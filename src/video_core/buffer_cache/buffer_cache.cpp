// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <span>
#include <vector>
#if defined(__ANDROID__)
#include <android/log.h>
#include <cstdio>
#include <cstdlib>
#endif
#include "common/alignment.h"
#include "common/content_fingerprint.h"
#include "common/debug.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "core/libraries/gnmdriver/gnmdriver.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace VideoCore {

#ifdef __ANDROID__
namespace {

bool ExecutorBufferReadbackContractTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EXECUTOR_READBACK_CONTRACT_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool ExecutorShouldLogBufferReadbackOrdinal(const u64 ordinal) {
    return ordinal <= 64 || (ordinal & (ordinal - 1)) == 0;
}

std::atomic<u64> executor_buffer_read_request_count{0};
std::atomic<u64> executor_buffer_download_count{0};
std::atomic<u64> executor_buffer_empty_download_count{0};
std::atomic<u64> executor_buffer_download_bytes{0};

// Precise readbacks are correctness-critical for GPU-authored buffers, but an Android SIGSEGV
// round-trip followed by a Vulkan submit/Finish for every 4 KiB page is prohibitively expensive.
// Once one byte in a cached buffer must become CPU-visible, downloading a bounded surrounding
// window is semantically equivalent: all queued GPU writes are completed by the same Finish and
// every copied page is re-armed if the GPU modifies it again. This trades a little transfer
// bandwidth for up to 256 times fewer signal, mprotect and queue-wait round-trips. Only ranges
// actually marked GPU-dirty are copied, so a sparse 1 MiB window does not imply a 1 MiB transfer.
constexpr u64 ExecutorPreciseReadbackWindow = 1_MB;
constexpr u64 ExecutorPreciseReadbackWaveWindow = 64_MB;
constexpr u64 ExecutorPreciseReadbackWaveMaxBytes = 16_MB;

} // namespace
#endif

static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      fault_manager{instance, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      stream_buffer{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      bda_pagetable_buffer{instance, scheduler, MemoryUsage::DeviceLocal,
                           0,        AllFlags,  BDA_PAGETABLE_SIZE} {
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);

    // Ensure the first slot is used for the null buffer
    const auto null_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
    ASSERT(null_id.index == 0);
    const vk::Buffer& null_buffer = slot_buffers[null_id].buffer;
    Vulkan::SetObjectName(instance.GetDevice(), null_buffer, "Null Buffer");

    // VK_EXT_robustness2 guarantees that reads through a true null descriptor return zero.  On
    // hosts without nullDescriptor support we bind this small real buffer instead, so it must
    // provide the same deterministic contract.  Device-local allocations have undefined initial
    // contents; leaving this uninitialised made optional guest resources return driver garbage.
    slot_buffers[null_id].Fill(0, 16, 0);

    // Set up garbage collection parameters
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    trigger_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_TRIGGER_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
}

BufferCache::~BufferCache() = default;

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size) {
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    memory_tracker->InvalidateRegion(
        device_addr, size, [this, device_addr, size] { ReadMemory(device_addr, size, true); });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
#ifdef __ANDROID__
    if (ExecutorBufferReadbackContractTraceEnabled()) {
        const u64 ordinal =
            executor_buffer_read_request_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (ExecutorShouldLogBufferReadbackOrdinal(ordinal)) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_READBACK_REQUEST] n=%llu addr=0x%llx bytes=%llu isWrite=%u",
                static_cast<unsigned long long>(ordinal),
                static_cast<unsigned long long>(device_addr),
                static_cast<unsigned long long>(size), is_write ? 1u : 0u);
        }
    }
#endif
    liverpool->SendCommand<true>([this, device_addr, size, is_write] {
        Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
        VAddr download_addr = device_addr;
        u64 download_size = size;
#ifdef __ANDROID__
        // Read faults in precise mode commonly walk adjacent shader/SRT/indirect-data pages. Keep
        // write faults exact (their CPU-modified bookkeeping has different semantics), but resolve
        // a bounded portion of the same already-cached buffer for reads. Clamping to the Buffer
        // avoids creating/merging cache objects and never touches an unrelated guest allocation.
        if (!is_write && size <= TRACKER_BYTES_PER_PAGE &&
            Config::getReadbacksMode() == Config::GpuReadbacksMode::Precise) {
            const VAddr buffer_begin = buffer.CpuAddr();
            const VAddr buffer_end = buffer_begin + buffer.SizeBytes();
            VAddr window_begin = Common::AlignDown(device_addr, ExecutorPreciseReadbackWindow);
            window_begin = std::max(window_begin, buffer_begin);
            const VAddr required_end = device_addr + size;
            VAddr window_end =
                std::min(buffer_end, window_begin + ExecutorPreciseReadbackWindow);
            if (window_end < required_end) {
                window_end = required_end;
                if (window_end - window_begin > ExecutorPreciseReadbackWindow) {
                    window_begin =
                        std::max(buffer_begin, window_end - ExecutorPreciseReadbackWindow);
                }
            }
            download_addr = window_begin;
            download_size = window_end - window_begin;
        }
#endif
#ifdef __ANDROID__
        if (!is_write && size <= TRACKER_BYTES_PER_PAGE &&
            Config::getReadbacksMode() == Config::GpuReadbacksMode::Precise) {
            DownloadBufferMemoryWave(buffer, download_addr, download_size);
        } else
#endif
        {
            DownloadBufferMemory<false>(buffer, download_addr, download_size, is_write);
        }
    });
}

#ifdef __ANDROID__
void BufferCache::DownloadBufferMemoryWave(Buffer& primary_buffer, VAddr device_addr, u64 size) {
    struct WaveCopy {
        Buffer* buffer = nullptr;
        VAddr guest_addr = 0;
        u64 size = 0;
        u64 source_offset = 0;
        u64 download_offset = 0;
    };

    boost::container::small_vector<WaveCopy, 32> copies;
    RangeSet scheduled_ranges;
    u64 total_size_bytes = 0;
    constexpr u64 CopyAlignment = 64;

    const auto queue_buffer_ranges = [&](Buffer& buffer, const VAddr range_addr,
                                         const u64 range_size) {
        if (range_size == 0 || total_size_bytes >= ExecutorPreciseReadbackWaveMaxBytes) {
            return;
        }
        memory_tracker->ForEachDownloadRange<false>(
            range_addr, range_size, [&](const u64 download_addr, const u64 download_size) {
                gpu_modified_ranges.ForEachInRange(
                    download_addr, download_size, [&](const VAddr start, const VAddr end) {
                        if (start >= end || scheduled_ranges.Intersects(start, end - start) ||
                            !buffer.IsInBounds(start, end - start) ||
                            total_size_bytes >= ExecutorPreciseReadbackWaveMaxBytes) {
                            return;
                        }
                        const u64 available =
                            ExecutorPreciseReadbackWaveMaxBytes - total_size_bytes;
                        const u64 copy_size = std::min<u64>(end - start, available);
                        if (copy_size == 0) {
                            return;
                        }
                        copies.push_back(WaveCopy{
                            .buffer = &buffer,
                            .guest_addr = start,
                            .size = copy_size,
                            .source_offset = buffer.Offset(start),
                            .download_offset = total_size_bytes,
                        });
                        scheduled_ranges.Add(start, copy_size);
                        total_size_bytes =
                            Common::AlignUp(total_size_bytes + copy_size, CopyAlignment);
                    });
            });
    };

    // The faulting range is mandatory and is queued before speculative neighbours, so the byte
    // which triggered SIGSEGV is always visible when this function returns.
    queue_buffer_ranges(primary_buffer, device_addr, size);

    // GPU-written indirect buffers, descriptor tables and readback labels are commonly separate
    // cache objects while living in the same guest allocation arena. Gather them into the same
    // download buffer and fence. The cap prevents a render target heap from turning one CPU read
    // into an unbounded transfer; any remainder keeps its protection and faults normally later.
    const VAddr wave_begin =
        Common::AlignDown(device_addr, ExecutorPreciseReadbackWaveWindow);
    const VAddr wave_end =
        wave_begin > std::numeric_limits<VAddr>::max() - ExecutorPreciseReadbackWaveWindow
            ? std::numeric_limits<VAddr>::max()
            : wave_begin + ExecutorPreciseReadbackWaveWindow;
    buffer_ranges.ForEachInRange(
        wave_begin, wave_end - wave_begin,
        [&](const VAddr range_begin, const VAddr range_end, const BufferId id) {
            if (total_size_bytes >= ExecutorPreciseReadbackWaveMaxBytes ||
                IsBufferInvalid(id)) {
                return;
            }
            Buffer& buffer = slot_buffers[id];
            const VAddr begin = std::max(range_begin, buffer.CpuAddr());
            const VAddr end =
                std::min<VAddr>(range_end, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin < end) {
                queue_buffer_ranges(buffer, begin, end - begin);
            }
        });

    if (copies.empty()) {
        return;
    }

    const auto [download, base_offset] = download_buffer.Map(total_size_bytes);
    if (download == nullptr) {
        // Stream-buffer exhaustion is exceptional; preserve correctness with the proven exact
        // path instead of returning to the faulting instruction with stale guest memory.
        DownloadBufferMemory<false>(primary_buffer, device_addr, size, false);
        return;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    for (const WaveCopy& copy : copies) {
        cmdbuf.copyBuffer(copy.buffer->buffer, download_buffer.Handle(), vk::BufferCopy{
            .srcOffset = copy.source_offset,
            .dstOffset = base_offset + copy.download_offset,
            .size = copy.size,
        });
    }
    scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::BufferReadbackWave);

    auto* guest_memory = Core::Memory::Instance();
    for (const WaveCopy& copy : copies) {
        if (!guest_memory->TryWriteBacking(std::bit_cast<u8*>(copy.guest_addr),
                                           download + copy.download_offset, copy.size)) {
            static std::atomic<u32> failure_count{0};
            const u32 failure = failure_count.fetch_add(1, std::memory_order_relaxed);
            if (failure < 16) {
                LOG_ERROR(Render_Vulkan,
                          "GPU wave readback could not write guest backing: addr={:#x}, "
                          "size={:#x}, occurrence={}",
                          copy.guest_addr, copy.size, failure + 1);
            }
            continue;
        }
        // Publish the backing-store change before clearing GPU-dirty state. A shader identity
        // lookup which races this readback will then observe either dirty memory or a new
        // generation, never clean memory paired with stale metadata.
        NotifyGuestReadback();
        gpu_modified_ranges.Subtract(copy.guest_addr, copy.size);
        memory_tracker->UnmarkRegionAsGpuModified(copy.guest_addr, copy.size);
    }
}
#endif

template <bool async>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
        });
    if (total_size_bytes == 0) {
#ifdef __ANDROID__
        if (ExecutorBufferReadbackContractTraceEnabled()) {
            const u64 ordinal =
                executor_buffer_empty_download_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (ExecutorShouldLogBufferReadbackOrdinal(ordinal)) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_READBACK_DOWNLOAD_EMPTY] n=%llu addr=0x%llx bytes=%llu "
                    "bufferBase=0x%llx isWrite=%u",
                    static_cast<unsigned long long>(ordinal),
                    static_cast<unsigned long long>(device_addr),
                    static_cast<unsigned long long>(size),
                    static_cast<unsigned long long>(buffer.CpuAddr()), is_write ? 1u : 0u);
            }
        }
#endif
        return;
    }
#ifdef __ANDROID__
    if (ExecutorBufferReadbackContractTraceEnabled()) {
        const u64 ordinal =
            executor_buffer_download_count.fetch_add(1, std::memory_order_relaxed) + 1;
        const u64 aggregate_bytes =
            executor_buffer_download_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed) +
            total_size_bytes;
        if (ExecutorShouldLogBufferReadbackOrdinal(ordinal)) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_READBACK_DOWNLOAD] n=%llu addr=0x%llx requestBytes=%llu "
                "downloadBytes=%llu aggregateBytes=%llu copies=%zu bufferBase=0x%llx isWrite=%u "
                "async=%u",
                static_cast<unsigned long long>(ordinal),
                static_cast<unsigned long long>(device_addr),
                static_cast<unsigned long long>(size),
                static_cast<unsigned long long>(total_size_bytes),
                static_cast<unsigned long long>(aggregate_bytes), copies.size(),
                static_cast<unsigned long long>(buffer.CpuAddr()), is_write ? 1u : 0u,
                async ? 1u : 0u);
        }
    }
#endif
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
    const auto write_data = [&]() {
        auto* memory = Core::Memory::Instance();
        bool wrote_all = true;
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buffer.CpuAddr() + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - offset;
            if (!memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr),
                                         download + dst_offset, copy.size)) {
                wrote_all = false;
                static std::atomic<u32> failure_count{0};
                const u32 failure = failure_count.fetch_add(1, std::memory_order_relaxed);
                if (failure < 16) {
                    LOG_ERROR(Render_Vulkan,
                              "GPU buffer readback could not write guest backing: addr={:#x}, "
                              "size={:#x}, occurrence={}",
                              copy_device_addr, copy.size, failure + 1);
                }
            }
        }
        if (!wrote_all) {
            return;
        }
        // See DownloadBufferMemoryWave: generation-before-clean closes the interval in which
        // successful GPU data could otherwise look clean while retaining old shader metadata.
        NotifyGuestReadback();
        for (const auto& copy : copies) {
            gpu_modified_ranges.Subtract(buffer.CpuAddr() + copy.srcOffset, copy.size);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if constexpr (async) {
        scheduler.DeferOperation(write_data);
    } else {
        scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::BufferReadbackDirect);
        write_data();
    }
}

BufferCache::PreparedVertexBuffers BufferCache::PrepareVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
    const bool executor_probe_bound, const u64 executor_probe_sequence) {
    const auto& regs = liverpool->regs;
    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1);

    // EXECUTOR replay: the vertex-attribute V# base_address comes straight from the fetch shader's
    // user_data and never passes through DCB/table relocation (same class as constant/texture sharps in
    // BindBuffers/BindTextures). Unrelocated, the UV/color/position streams read the OLD captured VA ->
    // zero/garbage -> textured draws output black (fullscreen attributeless passes still cover because
    // they synthesize positions from vertex_id and have no vertex buffers). Rebase each stream here.
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        for (auto& buffer : guest_buffers) {
            if (buffer.base_address != 0) {
                const u64 nb = Libraries::GnmDriver::ExecutorReplayRelocate(buffer.base_address);
                if (nb) {
                    buffer.base_address = nb;
                }
#ifdef __ANDROID__
                u32 v0 = 0, v1 = 0;
                if (Libraries::GnmDriver::ExecutorReplayIsRelocated(buffer.base_address)) {
                    const u32* p = reinterpret_cast<const u32*>(buffer.base_address);
                    v0 = p[0];
                    v1 = p[1];
                }
                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                    "[EXECUTOR_VBUF] base=0x%llx stride=%u size=%u reloc=%d v=%08x %08x",
                                    (unsigned long long)buffer.base_address, (unsigned)buffer.GetStride(),
                                    (unsigned)buffer.GetSize(), nb ? 1 : 0, v0, v1);
#endif
            }
        }
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        vk::Buffer vk_buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    Vulkan::VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    Vulkan::VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size, false);
        range.vk_buffer = buffer->buffer;
        range.offset = offset;
        if (IsRegionGpuModified(range.base_address, size)) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    PreparedVertexBuffers prepared{};
    prepared.dynamic_vertex_input = instance.IsVertexInputDynamicState();
    prepared.attributes.assign(attributes.begin(), attributes.end());
    prepared.bindings.assign(bindings.begin(), bindings.end());

    // Resolve vertex buffers, but do not record any state yet. ObtainBuffer can wait for the stream
    // ring and rotate Scheduler::CommandBuffer; emitting only after every input is prepared prevents
    // a later allocation from silently discarding an earlier vertex/index binding.
    const auto null_buffer =
        instance.IsNullDescriptorSupported() ? VK_NULL_HANDLE : GetBuffer(NULL_BUFFER_ID).Handle();
    for (const auto& buffer : guest_buffers) {
        if (buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            prepared.buffers.emplace_back(host_buffer_info->vk_buffer);
            prepared.offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                       host_buffer_info->base_address);
        } else {
            prepared.buffers.emplace_back(null_buffer);
            prepared.offsets.push_back(0);
        }
        prepared.sizes.push_back(buffer.GetSize());
        prepared.strides.push_back(buffer.GetStride());
    }

#ifdef __ANDROID__
    // Compare the exact byte ranges about to be consumed by fixed-function vertex fetch.  The
    // ordinary GPU-content probe fingerprints guest memory before binding; that is insufficient on
    // a split CPU/GPU cache because ObtainBuffer may legally select an older VkBuffer allocation.
    // This oracle deliberately receives the already selected handle and offset and is only enabled
    // for the externally bounded draw capture.
    if (executor_probe_bound) {
        // Interleaved attributes normally overlap almost completely. Probe each merged allocation
        // once so the digest covers the highest referenced vertex as well as the first 64 KiB,
        // without issuing five redundant downloads of the same stream.
        for (u32 i = 0; i < ranges_merged.size(); ++i) {
            const auto& range = ranges_merged[i];
            if (range.base_address != 0 && range.GetSize() != 0 &&
                range.vk_buffer != VK_NULL_HANDLE) {
                ExecutorProbeBoundGpuBuffer(range.base_address, static_cast<u32>(range.GetSize()),
                                            range.vk_buffer, range.offset, "vertex_range",
                                            executor_probe_sequence, i);
            }
        }
    }
#else
    (void)executor_probe_bound;
    (void)executor_probe_sequence;
#endif

    return prepared;
}

void BufferCache::BindPreparedVertexBuffers(const PreparedVertexBuffers& prepared) {
    const auto cmdbuf = scheduler.CommandBuffer();
    if (prepared.dynamic_vertex_input) {
        cmdbuf.setVertexInputEXT(prepared.bindings, prepared.attributes);
    }
    if (prepared.buffers.empty()) {
        return;
    }

    const auto num_buffers = static_cast<u32>(prepared.buffers.size());
    if (prepared.dynamic_vertex_input) {
        // Match the desktop shadPS4 path. The dynamic vertex-input state already owns the guest
        // stride; supplying a second, exact byte range here changes robust vertex-fetch behaviour
        // on mobile drivers and can turn legal indexed fetch patterns into zero-filled vertices.
        cmdbuf.bindVertexBuffers(0, num_buffers, prepared.buffers.data(), prepared.offsets.data());
    } else {
        cmdbuf.bindVertexBuffers2(0, num_buffers, prepared.buffers.data(), prepared.offsets.data(),
                                  prepared.sizes.data(), prepared.strides.data());
    }
}

BufferCache::PreparedIndexBuffer BufferCache::PrepareIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
    const bool executor_probe_bound, const u64 executor_probe_sequence) {
    const auto& regs = liverpool->regs;

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [vk_buffer, offset] = ObtainBuffer(index_address, index_buffer_size, false);
    if (IsRegionGpuModified(index_address, index_buffer_size)) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers.emplace_back(*barrier);
        }
    }
#ifdef __ANDROID__
    if (executor_probe_bound) {
        ExecutorProbeBoundGpuBuffer(index_address, index_buffer_size, vk_buffer->Handle(), offset,
                                    "index_input", executor_probe_sequence, 0);
    }
#else
    (void)executor_probe_bound;
    (void)executor_probe_sequence;
#endif
    return PreparedIndexBuffer{
        .buffer = vk_buffer->Handle(),
        .offset = offset,
        .type = index_type,
    };
}

void BufferCache::BindPreparedIndexBuffer(const PreparedIndexBuffer& prepared) {
    scheduler.CommandBuffer().bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            return;
        }
    }
    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }
        const auto [buffer, offset] = ObtainBuffer(address, num_bytes, true);
        return buffer;
    }();
    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true);
        return buffer;
    }();
    auto& dst_buffer = [&] -> const Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true);
        gpu_modified_ranges.Add(dst, num_bytes);
        return buffer;
    }();
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_before,
    });
    cmdbuf.copyBuffer(src_buffer.Handle(), dst_buffer.Handle(), region);
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 2,
        .pBufferMemoryBarriers = buf_barriers_after,
    });
}

#ifdef __ANDROID__
// Sonic JIT corruption hunt (path B): flag a GPU buffer/image descriptor whose guest address or
// size is CLEARLY garbage — a legit PS4 GPU VA is < 2^40 (0x2xx direct / 0x8xx module, all < 64GB) and
// sizes are modest. A base >= 0x1000000000 (into the host lib/heap band ~0x77xx) or a > 256MB size
// means JIT produced a bad V#/T# (the user's thesis: our JIT emits a subtly-wrong sharp), which
// then drives the shared GPU code's unvalidated raw write into our .so/heap. Log the exact value so we
// can trace which draw/stage/shader-resource it came from.
static void ExecutorFlagSuspectGpuDescriptor(const char* site, VAddr addr, u64 size) {
    const bool bad_addr = addr >= 0x1000000000ull || (addr != 0 && addr < 0x10000ull);
    const bool bad_size = size > 0x10000000ull;
    if (!bad_addr && !bad_size) {
        return;
    }
    static std::atomic<u32> budget{64};
    if (budget.fetch_sub(1, std::memory_order_relaxed) == 0) {
        return;
    }
    if (budget.load(std::memory_order_relaxed) > 60) {
        // keep going
    }
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_SUSPECT_GPU_DESCRIPTOR] site=%s addr=0x%llx size=0x%llx "
                        "badAddr=%d badSize=%d (JIT garbage sharp)",
                        site, static_cast<unsigned long long>(addr),
                        static_cast<unsigned long long>(size), bad_addr ? 1 : 0, bad_size ? 1 : 0);
    if (std::FILE* cf = std::fopen(
            "/data/data/app.lsx4.android/files/lsx4-home/jit-checkfail.txt", "a")) {
        std::fprintf(cf, "SUSPECT_GPU_DESCRIPTOR site=%s addr=0x%llx size=0x%llx\n", site,
                     static_cast<unsigned long long>(addr), static_cast<unsigned long long>(size));
        std::fclose(cf);
    }
}
#endif

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written,
                                                  bool is_texel_buffer, BufferId buffer_id) {
#ifdef __ANDROID__
    ExecutorFlagSuspectGpuDescriptor("ObtainBuffer", device_addr, size);
#endif
    // EXECUTOR (Codex phase-2): every VB/IB/constant/storage buffer a draw reads passes through here.
    // Record the guest [addr,size] for the active .gnmcap capture (read-only inputs; skip RT writes).
    if (!is_written && Libraries::GnmDriver::ExecutorGnmCaptureActive()) {
        Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(3u /*buffer*/, device_addr, size);
    }
    // During replay, report any buffer address that was not relocated (a resource still to capture).
    Libraries::GnmDriver::ExecutorReplayCheckAddr(3u /*buffer*/, device_addr, size);
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= CACHING_PAGESIZE && !IsRegionGpuModified(device_addr, size) &&
        (Libraries::GnmDriver::ExecutorReplayActive() ||
         IsRegionCpuModified(device_addr, size))) {
        const u64 offset = stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
        return {&stream_buffer, offset};
    }
    if (IsBufferInvalid(buffer_id)) {
        buffer_id = FindBuffer(device_addr, size);
    }
    Buffer& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
#ifdef __ANDROID__
    ExecutorFlagSuspectGpuDescriptor("ObtainBufferForImage", gpu_addr, size);
#endif
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing GPU data.
    if (IsRegionGpuModified(gpu_addr, size)) {
        return ObtainBuffer(gpu_addr, size, false, false);
    }
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, 16);
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Check if we are missing some edge case here
    return buffer_ranges.Intersects(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

#ifdef __ANDROID__
void BufferCache::ExecutorProbeGpuBuffer(VAddr device_addr, u32 size, const char* role,
                                         u64 sequence) {
    if (device_addr == 0 || size == 0) {
        return;
    }

    // The diagnostic is externally bounded by the selected-draw content probe.  Cap each resource
    // anyway so an invalid guest descriptor cannot turn it into a large synchronous readback.
    constexpr u32 MaxProbeBytes = 64_KB;
    const u32 probe_size = std::min(size, MaxProbeBytes);
    const auto [source, source_offset] = ObtainBuffer(device_addr, probe_size, false);
    if (source == nullptr || !source->IsInBounds(device_addr, probe_size)) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_GPU_BUFFER_CONTENT] seq=%llu role=%s guest=0x%llx "
                            "bytes=%u status=unavailable",
                            static_cast<unsigned long long>(sequence), role,
                            static_cast<unsigned long long>(device_addr), probe_size);
        return;
    }

    const auto [download, download_offset] = download_buffer.Map(probe_size, 64);
    if (download == nullptr) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_GPU_BUFFER_CONTENT] seq=%llu role=%s guest=0x%llx "
                            "bytes=%u status=download_full",
                            static_cast<unsigned long long>(sequence), role,
                            static_cast<unsigned long long>(device_addr), probe_size);
        return;
    }
    download_buffer.Commit();

    // Use a conservative dependency without mutating Buffer::GetBarrier state.  Resource binding
    // has already assembled its tracked barriers for this draw; changing that state here would make
    // this diagnostic alter the following draw.  The two explicit barriers make every earlier GPU
    // producer visible to the transfer and the transfer read complete before later consumers.
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = source->Handle(),
        .offset = source_offset,
        .size = probe_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    const vk::BufferCopy copy = {
        .srcOffset = source_offset,
        .dstOffset = download_offset,
        .size = probe_size,
    };
    cmdbuf.copyBuffer(source->Handle(), download_buffer.Handle(), copy);
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .buffer = source->Handle(),
        .offset = source_offset,
        .size = probe_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
    scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::BufferProbe);

    const u64 digest = Common::FingerprintBytes(
        std::span<const std::uint8_t>{download, probe_size},
        Common::FingerprintDomain::BufferProbe);
    u32 nonzero = 0;
    u32 nan32 = 0;
    u32 inf32 = 0;
    float min_finite = std::numeric_limits<float>::infinity();
    float max_finite = -std::numeric_limits<float>::infinity();
    for (u32 i = 0; i < probe_size; ++i) {
        nonzero += download[i] != 0;
    }
    for (u32 i = 0; i + sizeof(u32) <= probe_size; i += sizeof(u32)) {
        u32 word{};
        std::memcpy(&word, download + i, sizeof(word));
        const float value = std::bit_cast<float>(word);
        nan32 += std::isnan(value);
        inf32 += std::isinf(value);
        if (std::isfinite(value)) {
            min_finite = std::min(min_finite, value);
            max_finite = std::max(max_finite, value);
        }
    }
    const auto word_at = [&](u32 index) {
        u32 value{};
        const u32 offset = index * sizeof(u32);
        if (offset + sizeof(u32) <= probe_size) {
            std::memcpy(&value, download + offset, sizeof(value));
        }
        return value;
    };
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_GPU_BUFFER_CONTENT] seq=%llu role=%s guest=0x%llx requested=%u bytes=%u "
        "digest=0x%llx nonzero=%u nan32=%u inf32=%u finiteMin=%g finiteMax=%g "
        "w0_15=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
        static_cast<unsigned long long>(sequence), role,
        static_cast<unsigned long long>(device_addr), size, probe_size,
        static_cast<unsigned long long>(digest), nonzero, nan32, inf32,
        std::isfinite(min_finite) ? min_finite : 0.0f,
        std::isfinite(max_finite) ? max_finite : 0.0f, word_at(0), word_at(1), word_at(2),
        word_at(3), word_at(4), word_at(5), word_at(6), word_at(7), word_at(8), word_at(9),
        word_at(10), word_at(11), word_at(12), word_at(13), word_at(14), word_at(15));
}

void BufferCache::ExecutorProbeBoundGpuBuffer(VAddr guest_addr, u32 size, vk::Buffer bound_buffer,
                                              u64 bound_offset, const char* role, u64 sequence,
                                              u32 slot) {
    if (guest_addr == 0 || size == 0 || bound_buffer == VK_NULL_HANDLE) {
        return;
    }

    // This is an oracle for the CPU->GPU boundary, not a general-purpose dump. Keep it bounded and
    // snapshot the guest first: Finish() below may let the guest recycle its transient ring range.
    constexpr u32 MaxProbeBytes = 512_KB;
    const u64 readable_size = memory->ClampRangeSize(guest_addr, size);
    const u32 probe_size = static_cast<u32>(std::min<u64>({size, MaxProbeBytes, readable_size}));
    if (probe_size == 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_BOUND_BUFFER_ORACLE] seq=%llu role=%s slot=%u "
                            "guest=0x%llx requested=%u status=guest_unreadable",
                            static_cast<unsigned long long>(sequence), role, slot,
                            static_cast<unsigned long long>(guest_addr), size);
        return;
    }

    std::vector<u8> guest(probe_size);
    memory->CopySparseMemory(guest_addr, guest.data(), probe_size);

    const auto [download, download_offset] = download_buffer.Map(probe_size, 64);
    if (download == nullptr) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_BOUND_BUFFER_ORACLE] seq=%llu role=%s slot=%u "
                            "guest=0x%llx bytes=%u status=download_full",
                            static_cast<unsigned long long>(sequence), role, slot,
                            static_cast<unsigned long long>(guest_addr), probe_size);
        return;
    }
    download_buffer.Commit();

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .buffer = bound_buffer,
        .offset = bound_offset,
        .size = probe_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(bound_buffer, download_buffer.Handle(), vk::BufferCopy{
        .srcOffset = bound_offset,
        .dstOffset = download_offset,
        .size = probe_size,
    });
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .buffer = bound_buffer,
        .offset = bound_offset,
        .size = probe_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
    scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::BufferProbe);

    const auto digest = [](const u8* bytes, const u32 count) {
        return Common::FingerprintBytes(
            std::span<const std::uint8_t>{bytes, count},
            Common::FingerprintDomain::BufferProbe);
    };
    u32 first_diff = probe_size;
    u32 guest_nonzero = 0;
    u32 gpu_nonzero = 0;
    for (u32 i = 0; i < probe_size; ++i) {
        guest_nonzero += guest[i] != 0;
        gpu_nonzero += download[i] != 0;
        if (first_diff == probe_size && guest[i] != download[i]) {
            first_diff = i;
        }
    }
    const auto word_at = [probe_size](const u8* bytes, const u32 index) {
        u32 value{};
        const u32 offset = index * sizeof(u32);
        if (offset + sizeof(u32) <= probe_size) {
            std::memcpy(&value, bytes + offset, sizeof(value));
        }
        return value;
    };
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_BOUND_BUFFER_ORACLE] seq=%llu role=%s slot=%u guest=0x%llx "
        "boundOffset=0x%llx requested=%u bytes=%u truncated=%u match=%u firstDiff=%u "
        "guestDigest=0x%llx gpuDigest=0x%llx guestNonzero=%u gpuNonzero=%u "
        "guestW0_7=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x "
        "gpuW0_7=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
        static_cast<unsigned long long>(sequence), role, slot,
        static_cast<unsigned long long>(guest_addr),
        static_cast<unsigned long long>(bound_offset), size, probe_size,
        probe_size != size ? 1u : 0u, first_diff == probe_size ? 1u : 0u, first_diff,
        static_cast<unsigned long long>(digest(guest.data(), probe_size)),
        static_cast<unsigned long long>(digest(download, probe_size)), guest_nonzero, gpu_nonzero,
        word_at(guest.data(), 0), word_at(guest.data(), 1), word_at(guest.data(), 2),
        word_at(guest.data(), 3), word_at(guest.data(), 4), word_at(guest.data(), 5),
        word_at(guest.data(), 6), word_at(guest.data(), 7), word_at(download, 0),
        word_at(download, 1), word_at(download, 2), word_at(download, 3), word_at(download, 4),
        word_at(download, 5), word_at(download, 6), word_at(download, 7));

    // Preserve enough exact input data to evaluate representative triangles offline without
    // another multi-minute game boot.  Constant buffers are small and are emitted in full up to
    // 4 KiB; interleaved vertex/index streams are capped at 512 bytes.  The summary above remains
    // the byte-for-byte GPU proof, so one guest copy is sufficient when match=1.
    const bool is_vs_buffer = role != nullptr && role[0] == 'v' && role[1] == 's' && role[2] == '_';
    const u32 dump_bytes = std::min(probe_size, is_vs_buffer ? u32{4_KB} : u32{512});
    const u32 dump_words = dump_bytes / sizeof(u32);
    for (u32 first_word = 0; first_word < dump_words; first_word += 8) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_BOUND_BUFFER_WORDS] seq=%llu role=%s slot=%u first=%u count=%u "
            "w=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
            static_cast<unsigned long long>(sequence), role, slot, first_word,
            std::min(8u, dump_words - first_word), word_at(guest.data(), first_word + 0),
            word_at(guest.data(), first_word + 1), word_at(guest.data(), first_word + 2),
            word_at(guest.data(), first_word + 3), word_at(guest.data(), first_word + 4),
            word_at(guest.data(), first_word + 5), word_at(guest.data(), first_word + 6),
            word_at(guest.data(), first_word + 7));
    }
}
#endif

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size) {
    if (device_addr == 0) {
        return NULL_BUFFER_ID;
    }
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    const auto expand_begin = [&](VAddr add_value) {
        static constexpr VAddr min_page = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    const auto expand_end = [&](VAddr add_value) {
        static constexpr VAddr max_page = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    const vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = dst_base_offset,
        .size = overlap.SizeBytes(),
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copy);

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size) {
    const VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
    auto& new_buffer = slot_buffers[new_buffer_id];
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    return new_buffer_id;
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page].buffer_id = buffer_id;
        } else {
            page_table[page].buffer_id = BufferId{};
        }
    }
    if constexpr (insert) {
        total_used_memory += Common::AlignUp(size, CACHING_PAGESIZE);
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        for (u64 i = 0; i < size_pages; ++i) {
            vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
            bda_addrs.push_back(addr);
        }
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
        bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] { src_buffer = UploadCopies(buffer, copies, total_size_bytes); });

    if (src_buffer) {
        ASSERT(!copies.empty());
        vk::DeviceSize upload_begin = copies.front().dstOffset;
        vk::DeviceSize upload_end = upload_begin + copies.front().size;
        for (const auto& copy : copies) {
            upload_begin = std::min(upload_begin, copy.dstOffset);
            upload_end = std::max(upload_end, copy.dstOffset + copy.size);
        }
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = buffer.Handle(),
            .offset = upload_begin,
            .size = upload_end - upload_begin,
        };
        const vk::BufferMemoryBarrier2 post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .buffer = buffer.Handle(),
            .offset = upload_begin,
            .size = upload_end - upload_begin,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
        cmdbuf.copyBuffer(src_buffer, buffer.buffer, copies);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &post_barrier,
        });
        TouchBuffer(buffer);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeBufferFromImage(buffer, device_addr, size);
    }
    return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            // Apply the staging offset
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        // For large one time transfers use a temporary host buffer.
        auto temp_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = temp_buffer->Handle();
        u8* const staging = temp_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        }
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
        return src_buffer;
    }
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size) {
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        return false;
    }
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
        copy_size += mip_info.size;
    }
    if (copy_size == 0) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    return true;
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size) {
    const VAddr device_addr_end = device_addr + size;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        SynchronizeBuffer(buffer, start, size, false, false);
    });
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

void BufferCache::RunGarbageCollector(const std::optional<size_t> sampled_device_memory) {
    SCOPE_EXIT {
        ++gc_tick;
    };
    if (sampled_device_memory) {
        total_used_memory = *sampled_device_memory;
    } else if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    const bool aggressive = total_used_memory >= critical_gc_memory;
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return;
        }
        --max_deletions;
        Buffer& buffer = slot_buffers[buffer_id];
        // InvalidateMemory(buffer.CpuAddr(), buffer.SizeBytes());
        DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes(), true);
        DeleteBuffer(buffer_id);
    };
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    Unregister(buffer_id);
    scheduler.DeferOperation([this, buffer_id] { slot_buffers.erase(buffer_id); });
    buffer.is_deleted = true;
}

} // namespace VideoCore
