// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <tuple>
#include <utility>
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
#include "video_core/page_manager.h"
#include "video_core/mobile_gpu_optimization.h"
#include "video_core/readback_optimization.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/vulkan_driver_optimization.h"

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

bool ExecutorReadOnlyRenameEnabled() {
    static const bool enabled = [] {
        constexpr const char* DisableMarker =
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-bisect-disable-read-only-rename";
        if (std::FILE* marker = std::fopen(DisableMarker, "rb")) {
            std::fclose(marker);
            return false;
        }
        return true;
    }();
    return enabled;
}

bool ExecutorReadOnlyRenameUsageEnabled(const BufferCache::ReadOnlyBufferUsage usage) {
    constexpr std::array<const char*, static_cast<std::size_t>(
                                          BufferCache::ReadOnlyBufferUsage::Count)>
        DisableMarkers{
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-bisect-disable-read-only-rename-vertex",
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-bisect-disable-read-only-rename-index",
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-bisect-disable-read-only-rename-uniform",
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-bisect-disable-read-only-rename-storage",
        };
    const auto read_mask = [&] {
        u32 result = (1u << DisableMarkers.size()) - 1u;
        for (std::size_t i = 0; i < DisableMarkers.size(); ++i) {
            if (std::FILE* marker = std::fopen(DisableMarkers[i], "rb")) {
                std::fclose(marker);
                result &= ~(1u << i);
            }
        }
        return result;
    };
    static std::atomic<u32> enabled_mask{read_mask()};
    static std::atomic<u64> poll_count{0};
    // Runtime polling is intentionally sparse: it makes type-by-type GPU
    // corruption bisection possible without another multi-minute title load,
    // while four marker probes per 65536 resources are immeasurable in play.
    if ((poll_count.fetch_add(1, std::memory_order_relaxed) & 0xffffu) == 0xffffu) {
        enabled_mask.store(read_mask(), std::memory_order_relaxed);
    }
    return (enabled_mask.load(std::memory_order_relaxed) &
            (1u << static_cast<std::size_t>(usage))) != 0;
}

bool ExecutorShouldLogBufferReadbackOrdinal(const u64 ordinal) {
    return ordinal <= 64 || (ordinal & (ordinal - 1)) == 0;
}

std::atomic<u64> executor_buffer_read_request_count{0};
std::atomic<u64> executor_buffer_download_count{0};
std::atomic<u64> executor_buffer_empty_download_count{0};
std::atomic<u64> executor_buffer_download_bytes{0};

constexpr u64 ExecutorPreciseReadbackWindow = TRACKER_BYTES_PER_PAGE;
u64 ExecutorElapsedNs(const std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

void ExecutorMaybeLogReadbackOptimizationStats() {
    const auto stats = GetReadbackOptimizationStats();
    static std::atomic<u64> next_log{1};
    u64 expected = next_log.load(std::memory_order_relaxed);
    if (stats.snapshots < expected) {
        return;
    }
    const u64 following = expected == 1 ? 256 : expected + 256;
    if (!next_log.compare_exchange_strong(expected, following, std::memory_order_relaxed)) {
        return;
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_READBACK_BATCH_STATS] enabled=%u mode=%d snapshots=%llu "
        "submit=%llu demand=%llu ranges=%llu pages=%llu copyCommands=%llu "
        "copyRegions=%llu sourceBuffers=%llu mergedRegions=%llu timelineWaits=%llu "
        "bytes=%llu cpuPlanMs=%.3f "
        "cpuPublishMs=%.3f cpuWaitMs=%.3f gpuCompletionMs=%.3f fallbacks=%llu "
        "publishFailures=%llu duplicateSubmitSkips=%llu overflowAllocations=%llu "
        "demandFlushes=%llu baselineRequests=%llu sidecarLabels=%llu sidecarWaits=%llu",
        Config::managedReadbackBatching() ? 1u : 0u, Config::getReadbacksMode(),
        static_cast<unsigned long long>(stats.snapshots),
        static_cast<unsigned long long>(stats.submit_snapshots),
        static_cast<unsigned long long>(stats.demand_snapshots),
        static_cast<unsigned long long>(stats.ranges),
        static_cast<unsigned long long>(stats.pages),
        static_cast<unsigned long long>(stats.copy_commands),
        static_cast<unsigned long long>(stats.copy_regions),
        static_cast<unsigned long long>(stats.source_buffers),
        static_cast<unsigned long long>(stats.regions_merged),
        static_cast<unsigned long long>(stats.timeline_waits),
        static_cast<unsigned long long>(stats.bytes),
        static_cast<double>(stats.cpu_plan_ns) / 1'000'000.0,
        static_cast<double>(stats.cpu_publish_ns) / 1'000'000.0,
        static_cast<double>(stats.cpu_wait_ns) / 1'000'000.0,
        static_cast<double>(stats.gpu_completion_ns) / 1'000'000.0,
        static_cast<unsigned long long>(stats.fallbacks),
        static_cast<unsigned long long>(stats.publication_failures),
        static_cast<unsigned long long>(stats.duplicate_submit_skips),
        static_cast<unsigned long long>(stats.overflow_allocations),
        static_cast<unsigned long long>(stats.demand_flushes),
        static_cast<unsigned long long>(stats.baseline_requests),
        static_cast<unsigned long long>(stats.sidecar_labels),
        static_cast<unsigned long long>(stats.sidecar_waits));
}

}
#endif

static constexpr size_t DataShareBufferSize = 64_KB;
#ifdef __ANDROID__
static constexpr size_t StagingBufferSize = 192_MB;
static constexpr size_t DownloadBufferSize = 16_MB;
static constexpr size_t UboStreamBufferSize = 32_MB;
static constexpr size_t DeviceBufferSize = 64_MB;
#else
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;
#endif

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      page_manager{&tracker}, memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
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

#ifdef __ANDROID__
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_READBACK_BATCH_CONFIG] enabled=%u effectiveMode=%d "
        "configuredBy=managed_optimization_7",
        Config::managedReadbackBatching() ? 1u : 0u, Config::getReadbacksMode());
#endif

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);

    const auto null_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
    ASSERT(null_id.index == 0);
    const vk::Buffer& null_buffer = slot_buffers[null_id].buffer;
    Vulkan::SetObjectName(instance.GetDevice(), null_buffer, "Null Buffer");

    slot_buffers[null_id].Fill(0, 16, 0);

    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
#ifdef __ANDROID__
    const auto adaptive_threshold = [device_local_memory](
                                        const s64 numerator,
                                        const s64 denominator,
                                        const s64 floor,
                                        const s64 ceiling) {
        const s64 proportional = device_local_memory * numerator / denominator;
        return static_cast<u64>(
            std::min(device_local_memory, std::clamp(proportional, floor, ceiling)));
    };
    const bool managed_mobile_gpu =
        VideoCore::IsManagedMobileGpuOptimization();
    trigger_gc_memory = managed_mobile_gpu
                            ? adaptive_threshold(7, 20, 512_MB, 1_GB + 256_MB)
                            : adaptive_threshold(9, 20, 768_MB, 1_GB + 512_MB);
    critical_gc_memory = std::min<u64>(
        device_local_memory,
        std::max<u64>(
            trigger_gc_memory + 256_MB,
            managed_mobile_gpu
                ? adaptive_threshold(7, 10, 1_GB, 2_GB + 256_MB)
                : adaptive_threshold(3, 4, 1_GB + 256_MB, 2_GB + 512_MB)));
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_BUFFER_GC_POLICY] budgetMiB=%lld triggerMiB=%llu criticalMiB=%llu",
        static_cast<long long>(device_local_memory / 1_MB),
        static_cast<unsigned long long>(trigger_gc_memory / 1_MB),
        static_cast<unsigned long long>(critical_gc_memory / 1_MB));
    return;
#endif
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
    scheduler.FlushPendingBufferTransfersForRange(device_addr, size);
#ifdef __ANDROID__
    WaitForPendingReadback(device_addr, size);
#endif
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    liverpool->NotifyMemoryPublicationForExecutor(device_addr, size);
    memory_tracker->InvalidateRegion(
        device_addr, size, [this, device_addr, size] { ReadMemory(device_addr, size, true); });
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write) {
    scheduler.FlushPendingBufferTransfersForRange(device_addr, size);
#ifdef __ANDROID__
    WaitForPendingReadback(device_addr, size);
#endif
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
        const bool managed_submit_readback =
            Config::managedReadbackBatching() &&
            Config::getReadbacksMode() == Config::GpuReadbacksMode::Precise;
        if (managed_submit_readback && size <= TRACKER_BYTES_PER_PAGE) {
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
        if (managed_submit_readback) {
            DownloadBufferMemoryWave(buffer, download_addr, download_size, is_write);
        } else
#endif
        {
#ifdef __ANDROID__
            AccumulateReadbackOptimizationStats(
                ReadbackOptimizationStats{.baseline_requests = 1});
#endif
            DownloadBufferMemory<false>(buffer, download_addr, download_size, is_write);
        }
    });
}

#ifdef __ANDROID__
void BufferCache::WaitForPendingReadback(VAddr device_addr, u64 size) {
    if (size == 0 || device_addr > std::numeric_limits<VAddr>::max() - size) {
        return;
    }
    u64 dependency_tick = 0;
    {
        std::scoped_lock lock{pending_readback_mutex};
        const VAddr request_end = device_addr + size;
        for (const PendingReadbackRange& pending : pending_readback_ranges) {
            if (pending.begin > std::numeric_limits<VAddr>::max() - pending.size) {
                continue;
            }
            const VAddr pending_end = pending.begin + pending.size;
            if (device_addr < pending_end && pending.begin < request_end) {
                dependency_tick = std::max(dependency_tick, pending.tick);
            }
        }
    }
    if (dependency_tick == 0) {
        return;
    }
    liverpool->SendCommand<true>([this, dependency_tick] {
        const auto wait_start = std::chrono::steady_clock::now();
        // A submit snapshot is normally carried by the surrounding guest
        // submit. A CPU fault can arrive before that boundary is materialized;
        // explicitly publish the current command buffer in that case so the
        // retirement worker can own the single host timeline wait.
        if (dependency_tick >= scheduler.CurrentTick()) {
            scheduler.Flush();
        }
        // The retirement worker owns the sole Vulkan timeline wait for this
        // tick. Waiting for retirement here avoids issuing a second host wait.
        scheduler.WaitRetirement(dependency_tick);
        AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
            .cpu_wait_ns = ExecutorElapsedNs(wait_start),
        });
    });
}

void BufferCache::PrefetchReadbackWindows() {
    const auto plan_start = std::chrono::steady_clock::now();
    if (!Config::managedReadbackBatching() ||
        recent_readback_window_count == 0 ||
        Config::getReadbacksMode() != Config::GpuReadbacksMode::Precise) {
        return;
    }
    const u64 snapshot_tick = scheduler.CurrentTick();
    if (snapshot_tick != 0 && snapshot_tick == last_submit_readback_snapshot_tick) {
        AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
            .duplicate_submit_skips = 1,
        });
        return;
    }
    scheduler.FlushPendingBufferTransfers();

    struct PrefetchCopy {
        vk::Buffer source{};
        VAddr guest_addr = 0;
        u64 size = 0;
        u64 source_offset = 0;
        u64 download_offset = 0;
    };

    boost::container::small_vector<PrefetchCopy, 32> copies;
    RangeSet scheduled_ranges;
    RangeSet excluded_ranges;
    u64 total_size_bytes = 0;
    constexpr u64 CopyAlignment = 64;

    // Start with ranges already owned by an earlier submit snapshot. New
    // ranges are added below, so partially overlapping dirty spans still
    // contribute their uncovered pages instead of being skipped wholesale.
    {
        std::scoped_lock lock{pending_readback_mutex};
        for (const PendingReadbackRange& pending : pending_readback_ranges) {
            if (pending.size != 0 &&
                pending.begin <= std::numeric_limits<VAddr>::max() - pending.size) {
                excluded_ranges.Add(pending.begin, pending.size);
            }
        }
    }

    const auto queue_buffer_ranges = [&](Buffer& buffer, const VAddr range_addr,
                                         const u64 range_size) {
        if (range_size == 0) {
            return;
        }
        memory_tracker->ForEachDownloadRange<false>(
            range_addr, range_size, [&](const u64 download_addr, const u64 download_size) {
                gpu_modified_ranges.ForEachInRange(
                    download_addr, download_size, [&](const VAddr start, const VAddr end) {
                        if (start >= end || !buffer.IsInBounds(start, end - start)) {
                            return;
                        }
                        boost::container::small_vector<std::pair<VAddr, VAddr>, 4> gaps;
                        excluded_ranges.ForEachNotInRange(
                            start, end - start,
                            [&](const VAddr gap_begin, const u64 gap_size) {
                                gaps.emplace_back(gap_begin, gap_begin + gap_size);
                            });
                        for (const auto [gap_begin, gap_end] : gaps) {
                            const u64 copy_size = gap_end - gap_begin;
                            if (copy_size == 0) {
                                continue;
                            }
                            const u64 source_offset = buffer.Offset(gap_begin);
                            if (!copies.empty()) {
                                PrefetchCopy& previous = copies.back();
                                if (previous.source == buffer.Handle() &&
                                    previous.guest_addr + previous.size == gap_begin &&
                                    previous.source_offset + previous.size == source_offset &&
                                    previous.download_offset + previous.size ==
                                        total_size_bytes) {
                                    previous.size += copy_size;
                                } else {
                                    copies.push_back(PrefetchCopy{
                                        .source = buffer.Handle(),
                                        .guest_addr = gap_begin,
                                        .size = copy_size,
                                        .source_offset = source_offset,
                                        .download_offset = total_size_bytes,
                                    });
                                }
                            } else {
                                copies.push_back(PrefetchCopy{
                                    .source = buffer.Handle(),
                                    .guest_addr = gap_begin,
                                    .size = copy_size,
                                    .source_offset = source_offset,
                                    .download_offset = total_size_bytes,
                                });
                            }
                            scheduled_ranges.Add(gap_begin, copy_size);
                            excluded_ranges.Add(gap_begin, copy_size);
                            total_size_bytes =
                                Common::AlignUp(total_size_bytes + copy_size, CopyAlignment);
                        }
                    });
            });
    };

    for (u32 index = 0; index < recent_readback_window_count; ++index) {
        const RecentReadbackWindow& window = recent_readback_windows[index];
        buffer_ranges.ForEachInRange(
            window.begin, window.size,
            [&](const VAddr range_begin, const VAddr range_end, const BufferId id) {
                if (IsBufferInvalid(id)) {
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
    }
    if (copies.empty()) {
        return;
    }

    const u64 planned_regions = copies.size();
    std::stable_sort(copies.begin(), copies.end(),
                     [](const PrefetchCopy& lhs, const PrefetchCopy& rhs) {
                         const VkBuffer lhs_handle = static_cast<VkBuffer>(lhs.source);
                         const VkBuffer rhs_handle = static_cast<VkBuffer>(rhs.source);
                         if (lhs_handle != rhs_handle) {
                             return std::less<VkBuffer>{}(lhs_handle, rhs_handle);
                         }
                         if (lhs.source_offset != rhs.source_offset) {
                             return lhs.source_offset < rhs.source_offset;
                         }
                         return lhs.guest_addr < rhs.guest_addr;
                     });
    boost::container::small_vector<PrefetchCopy, 32> coalesced_copies;
    for (const PrefetchCopy& copy : copies) {
        if (!coalesced_copies.empty()) {
            PrefetchCopy& previous = coalesced_copies.back();
            if (previous.source == copy.source &&
                previous.guest_addr + previous.size == copy.guest_addr &&
                previous.source_offset + previous.size == copy.source_offset) {
                previous.size += copy.size;
                continue;
            }
        }
        coalesced_copies.push_back(copy);
    }
    copies = std::move(coalesced_copies);
    total_size_bytes = 0;
    for (PrefetchCopy& copy : copies) {
        copy.download_offset = total_size_bytes;
        total_size_bytes =
            Common::AlignUp(total_size_bytes + copy.size, CopyAlignment);
    }

    std::unique_ptr<Buffer> overflow_download;
    u8* download = nullptr;
    u64 base_offset = 0;
    if (total_size_bytes <= download_buffer.SizeBytes()) {
        std::tie(download, base_offset) =
            download_buffer.Map(total_size_bytes, CopyAlignment, false);
    }
    if (download == nullptr) {
        overflow_download = std::make_unique<Buffer>(
            std::nothrow, instance, scheduler, MemoryUsage::Download, 0,
            vk::BufferUsageFlagBits::eTransferDst, total_size_bytes);
        if (overflow_download->IsValid() &&
            overflow_download->mapped_data.size() >= total_size_bytes) {
            download = overflow_download->mapped_data.data();
            base_offset = 0;
            AccumulateReadbackOptimizationStats(
                ReadbackOptimizationStats{.overflow_allocations = 1});
        }
    }
    if (download == nullptr &&
        total_size_bytes <= download_buffer.SizeBytes()) {
        // Dedicated allocation can fail under transient heap pressure. The
        // existing ring remains a correctness fallback; waiting here only
        // retires an older ring slice and does not create another submit.
        overflow_download.reset();
        std::tie(download, base_offset) =
            download_buffer.Map(total_size_bytes, CopyAlignment, true);
    }
    if (download == nullptr) {
        AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
            .cpu_plan_ns = ExecutorElapsedNs(plan_start),
            .fallbacks = 1,
        });
        return;
    }
    Buffer* const download_target =
        overflow_download ? overflow_download.get() : &download_buffer;
    if (!overflow_download) {
        download_buffer.Commit();
    }
    scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::BufferReadback);
    const auto cmdbuf = scheduler.CommandBuffer();
    boost::container::small_vector<vk::BufferMemoryBarrier2, 32> pre_barriers;
    vk::Buffer previous_source{};
    for (const PrefetchCopy& copy : copies) {
        if (copy.source == previous_source) {
            continue;
        }
        previous_source = copy.source;
        pre_barriers.push_back(vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eShaderWrite |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = copy.source,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        });
    }
    pre_barriers.push_back(vk::BufferMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eHost,
        .srcAccessMask = vk::AccessFlagBits2::eHostRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = download_target->Handle(),
        .offset = base_offset,
        .size = total_size_bytes,
    });
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });
    std::vector<bool> copy_recorded(copies.size(), false);
    u64 copy_commands = 0;
    u64 copy_regions = 0;
    for (size_t first = 0; first < copies.size(); ++first) {
        if (copy_recorded[first]) {
            continue;
        }
        const vk::Buffer source = copies[first].source;
        boost::container::small_vector<vk::BufferCopy, 32> regions;
        for (size_t index = first; index < copies.size(); ++index) {
            if (!copy_recorded[index] && copies[index].source == source) {
                const PrefetchCopy& copy = copies[index];
                regions.push_back(vk::BufferCopy{
                    .srcOffset = copy.source_offset,
                    .dstOffset = base_offset + copy.download_offset,
                    .size = copy.size,
                });
                copy_recorded[index] = true;
            }
        }
        cmdbuf.copyBuffer(source, download_target->Handle(), regions);
        ++copy_commands;
        copy_regions += regions.size();
    }
    const vk::BufferMemoryBarrier2 host_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = download_target->Handle(),
        .offset = base_offset,
        .size = total_size_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });

    std::vector<PendingReadbackRange> tracked_ranges;
    tracked_ranges.reserve(copies.size());
    const u64 completion_tick = snapshot_tick;
    u64 tracked_pages = 0;
    u64 copied_bytes = 0;
    for (const PrefetchCopy& copy : copies) {
        copied_bytes += copy.size;
    }
    scheduled_ranges.ForEach([&](const VAddr begin, const VAddr end) {
        const u64 size = end - begin;
        tracked_pages +=
            (size + static_cast<u64>(TRACKER_BYTES_PER_PAGE) - 1) /
            static_cast<u64>(TRACKER_BYTES_PER_PAGE);
        page_manager->UpdatePageWatchers<true>(begin, size);
        page_manager->TrackPendingCompletionRead(begin, size);
        tracked_ranges.push_back(PendingReadbackRange{begin, size, completion_tick});
    });
    {
        std::scoped_lock lock{pending_readback_mutex};
        pending_readback_ranges.insert(pending_readback_ranges.end(), tracked_ranges.begin(),
                                       tracked_ranges.end());
    }
    for (const PendingReadbackRange& range : tracked_ranges) {
        memory_tracker->UnmarkRegionAsGpuModified(range.begin, range.size);
        gpu_modified_ranges.Subtract(range.begin, range.size);
    }
    last_submit_readback_snapshot_tick = snapshot_tick;

    static u64 prefetch_batches = 0;
    static u64 prefetch_bytes = 0;
    prefetch_bytes += total_size_bytes;
    if (((++prefetch_batches) & 255u) == 0u) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_READBACK_PREFETCH] batches=256 bytesMiB=%.3f avgKiB=%.3f",
            static_cast<double>(prefetch_bytes) / static_cast<double>(1_MB),
            static_cast<double>(prefetch_bytes) / (256.0 * 1024.0));
        prefetch_bytes = 0;
    }

    // Publish prefetched readbacks on the retirement runner for this exact
    // timeline tick. This avoids leaving completed copies in the ordinary
    // pending queue until another submit and lets faulting guest threads wait
    // for one batched retirement point instead of repeatedly round-tripping
    // through the GPU command queue.
    const auto queued_at = std::chrono::steady_clock::now();
    AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
        .snapshots = 1,
        .submit_snapshots = 1,
        .ranges = static_cast<u64>(tracked_ranges.size()),
        .pages = tracked_pages,
        .copy_commands = copy_commands,
        .copy_regions = copy_regions,
        .source_buffers = copy_commands,
        .regions_merged = planned_regions - copies.size(),
        .bytes = copied_bytes,
        .cpu_plan_ns = ExecutorElapsedNs(plan_start),
    });
    ExecutorMaybeLogReadbackOptimizationStats();
    scheduler.DeferPriorityOperationAt(
        completion_tick,
        [this, copies = std::move(copies), tracked_ranges = std::move(tracked_ranges),
         download, base_offset, total_size_bytes, completion_tick, queued_at,
         overflow_download = std::move(overflow_download)]() mutable {
            const auto publish_start = std::chrono::steady_clock::now();
            RecordReadbackTimelineRetirement(completion_tick);
            if (overflow_download) {
                overflow_download->InvalidateMappedRange(0, total_size_bytes);
            } else {
                download_buffer.InvalidateMappedRange(base_offset, total_size_bytes);
            }
            auto* guest_memory = Core::Memory::Instance();
            bool wrote_all = true;
            for (const PrefetchCopy& copy : copies) {
                if (!guest_memory->TryWriteBacking(
                        std::bit_cast<u8*>(copy.guest_addr),
                        download + copy.download_offset, copy.size)) {
                    wrote_all = false;
                }
            }
            if (!wrote_all) {
                AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
                    .fallbacks = 1,
                    .publication_failures = 1,
                });
                for (const PendingReadbackRange& range : tracked_ranges) {
                    gpu_modified_ranges.Add(range.begin, range.size);
                    memory_tracker->MarkRegionAsGpuModified(range.begin, range.size);
                }
            } else {
                NotifyGuestReadback();
            }
            AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
                .cpu_publish_ns = ExecutorElapsedNs(publish_start),
                .gpu_completion_ns = ExecutorElapsedNs(queued_at),
            });
            {
                std::scoped_lock lock{pending_readback_mutex};
                std::erase_if(pending_readback_ranges, [&](const auto& pending) {
                    return pending.tick == completion_tick &&
                           std::ranges::any_of(tracked_ranges, [&](const auto& completed) {
                               return pending.begin == completed.begin &&
                                      pending.size == completed.size;
                           });
                });
            }
            for (const PendingReadbackRange& range : tracked_ranges) {
                page_manager->UntrackPendingCompletionRead(range.begin, range.size);
                page_manager->UpdatePageWatchers<false>(range.begin, range.size);
            }
            liverpool->NotifyMemoryPublicationForExecutor(0, 1);
        });
}

void BufferCache::DownloadBufferMemoryWave(Buffer& primary_buffer, const VAddr device_addr,
                                           const u64 size, const bool is_write) {
    const auto plan_start = std::chrono::steady_clock::now();
    // Demand publication is an explicit consumer boundary: all queued writes
    // must be in the command stream before the one readback snapshot.
    scheduler.FlushPendingBufferTransfers();
    struct WaveCopy {
        Buffer* buffer{};
        VAddr guest_addr{};
        u64 size{};
        u64 source_offset{};
        u64 download_offset{};
        bool primary{};
    };

    boost::container::small_vector<WaveCopy, 32> copies;
    RangeSet scheduled_ranges;
    constexpr u64 CopyAlignment = 64;

    const auto queue_buffer_ranges = [&](Buffer& buffer, const VAddr range_addr,
                                         const u64 range_size, const bool primary) {
        if (range_size == 0) {
            return;
        }
        memory_tracker->ForEachDownloadRange<false>(
            range_addr, range_size,
            [&](const u64 download_addr, const u64 download_size) {
                gpu_modified_ranges.ForEachInRange(
                    download_addr, download_size,
                    [&](const VAddr start, const VAddr end) {
                        if (start >= end ||
                            !buffer.IsInBounds(start, end - start)) {
                            return;
                        }
                        boost::container::small_vector<std::pair<VAddr, VAddr>, 4>
                            gaps;
                        scheduled_ranges.ForEachNotInRange(
                            start, end - start,
                            [&](const VAddr gap_begin, const u64 gap_size) {
                                gaps.emplace_back(gap_begin,
                                                  gap_begin + gap_size);
                            });
                        for (const auto [gap_begin, gap_end] : gaps) {
                            const u64 copy_size = gap_end - gap_begin;
                            if (copy_size == 0) {
                                continue;
                            }
                            copies.push_back(WaveCopy{
                                .buffer = &buffer,
                                .guest_addr = gap_begin,
                                .size = copy_size,
                                .source_offset = buffer.Offset(gap_begin),
                                .primary = primary,
                            });
                            scheduled_ranges.Add(gap_begin, copy_size);
                        }
                    });
            });
    };

    queue_buffer_ranges(primary_buffer, device_addr, size, true);
    const auto queue_window = [&](const RecentReadbackWindow& window) {
        if (window.size == 0) {
            return;
        }
        buffer_ranges.ForEachInRange(
            window.begin, window.size,
            [&](const VAddr range_begin, const VAddr range_end,
                const BufferId id) {
                if (IsBufferInvalid(id)) {
                    return;
                }
                Buffer& buffer = slot_buffers[id];
                const VAddr begin = std::max(range_begin, buffer.CpuAddr());
                const VAddr end = std::min<VAddr>(
                    range_end, buffer.CpuAddr() + buffer.SizeBytes());
                if (begin < end) {
                    queue_buffer_ranges(buffer, begin, end - begin, false);
                }
            });
    };
    for (u32 index = 0; index < recent_readback_window_count; ++index) {
        queue_window(recent_readback_windows[index]);
    }

    // Page-sized CPU writes are the dominant Bloodborne readback fault too:
    // before a partial overwrite the current GPU contents must be published.
    // Keep both read and write windows as submit-prefetch hints so the next
    // ordinary submit retires recurring ranges in one wave instead of forcing
    // one Vulkan submit/timeline wait for every guest fault.
    if (size <= TRACKER_BYTES_PER_PAGE) {
        const RecentReadbackWindow current_window{device_addr, size};
        u32 previous_index = recent_readback_window_count;
        for (u32 index = 0; index < recent_readback_window_count; ++index) {
            const auto& recent = recent_readback_windows[index];
            if (recent.begin == current_window.begin &&
                recent.size == current_window.size) {
                previous_index = index;
                break;
            }
        }
        const u32 shift_from =
            previous_index < recent_readback_window_count
                ? previous_index
                : std::min<u32>(recent_readback_window_count,
                                recent_readback_windows.size() - 1);
        for (u32 index = shift_from; index > 0; --index) {
            recent_readback_windows[index] =
                recent_readback_windows[index - 1];
        }
        recent_readback_windows[0] = current_window;
        if (previous_index == recent_readback_window_count) {
            recent_readback_window_count =
                std::min<u32>(recent_readback_window_count + 1,
                              recent_readback_windows.size());
        }
    }

    if (copies.empty()) {
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
        return;
    }

    const u64 planned_regions = copies.size();
    std::stable_sort(
        copies.begin(), copies.end(),
        [](const WaveCopy& lhs, const WaveCopy& rhs) {
            const VkBuffer lhs_handle =
                static_cast<VkBuffer>(lhs.buffer->Handle());
            const VkBuffer rhs_handle =
                static_cast<VkBuffer>(rhs.buffer->Handle());
            if (lhs_handle != rhs_handle) {
                return std::less<VkBuffer>{}(lhs_handle, rhs_handle);
            }
            if (lhs.source_offset != rhs.source_offset) {
                return lhs.source_offset < rhs.source_offset;
            }
            return lhs.guest_addr < rhs.guest_addr;
        });
    boost::container::small_vector<WaveCopy, 32> coalesced_copies;
    for (const WaveCopy& copy : copies) {
        if (!coalesced_copies.empty()) {
            WaveCopy& previous = coalesced_copies.back();
            if (previous.buffer == copy.buffer &&
                previous.guest_addr + previous.size == copy.guest_addr &&
                previous.source_offset + previous.size ==
                    copy.source_offset) {
                previous.size += copy.size;
                previous.primary |= copy.primary;
                continue;
            }
        }
        coalesced_copies.push_back(copy);
    }
    copies = std::move(coalesced_copies);

    u64 total_size_bytes = 0;
    u64 copied_bytes = 0;
    for (WaveCopy& copy : copies) {
        copy.download_offset = total_size_bytes;
        copied_bytes += copy.size;
        total_size_bytes =
            Common::AlignUp(total_size_bytes + copy.size, CopyAlignment);
    }

    std::unique_ptr<Buffer> overflow_download;
    u8* download = nullptr;
    u64 base_offset = 0;
    if (total_size_bytes <= download_buffer.SizeBytes()) {
        std::tie(download, base_offset) =
            download_buffer.Map(total_size_bytes, CopyAlignment, false);
    }
    if (download == nullptr) {
        overflow_download = std::make_unique<Buffer>(
            std::nothrow, instance, scheduler, MemoryUsage::Download, 0,
            vk::BufferUsageFlagBits::eTransferDst, total_size_bytes);
        if (overflow_download->IsValid() &&
            overflow_download->mapped_data.size() >= total_size_bytes) {
            download = overflow_download->mapped_data.data();
            base_offset = 0;
            AccumulateReadbackOptimizationStats(
                ReadbackOptimizationStats{.overflow_allocations = 1});
        }
    }
    if (download == nullptr &&
        total_size_bytes <= download_buffer.SizeBytes()) {
        overflow_download.reset();
        std::tie(download, base_offset) =
            download_buffer.Map(total_size_bytes, CopyAlignment, true);
    }
    if (download == nullptr) {
        AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
            .cpu_plan_ns = ExecutorElapsedNs(plan_start),
            .fallbacks = 1,
        });
        // A full-size dedicated allocation can exceed a device allocation
        // limit. Preserve correctness with the same managed retirement path
        // in bounded chunks: no scheduler::Finish loop is reintroduced.
        const u64 chunk_size =
            std::max<u64>(TRACKER_BYTES_PER_PAGE,
                          download_buffer.SizeBytes() / 2);
        const auto saved_windows = recent_readback_windows;
        const u32 saved_window_count = recent_readback_window_count;
        recent_readback_window_count = 0;
        for (u64 offset = 0; offset < size;) {
            const u64 bytes = std::min(chunk_size, size - offset);
            DownloadBufferMemoryWave(primary_buffer, device_addr + offset,
                                     bytes, is_write);
            offset += bytes;
        }
        recent_readback_windows = saved_windows;
        recent_readback_window_count = saved_window_count;
        return;
    }

    Buffer* const download_target =
        overflow_download ? overflow_download.get() : &download_buffer;
    if (!overflow_download) {
        download_buffer.Commit();
    }
    scheduler.EndRendering(
        Vulkan::Scheduler::EndRenderingReason::BufferReadback);
    const auto cmdbuf = scheduler.CommandBuffer();
    boost::container::small_vector<vk::BufferMemoryBarrier2, 32> pre_barriers;
    Buffer* previous_source{};
    for (const WaveCopy& copy : copies) {
        if (copy.buffer == previous_source) {
            continue;
        }
        previous_source = copy.buffer;
        pre_barriers.push_back(vk::BufferMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eShaderWrite |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = copy.buffer->Handle(),
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        });
    }
    pre_barriers.push_back(vk::BufferMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eHost,
        .srcAccessMask = vk::AccessFlagBits2::eHostRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = download_target->Handle(),
        .offset = base_offset,
        .size = total_size_bytes,
    });
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount =
            static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    u64 copy_commands = 0;
    u64 copy_regions = 0;
    for (size_t first = 0; first < copies.size();) {
        const vk::Buffer source = copies[first].buffer->Handle();
        boost::container::small_vector<vk::BufferCopy, 32> regions;
        size_t end = first;
        while (end < copies.size() &&
               copies[end].buffer->Handle() == source) {
            const WaveCopy& copy = copies[end++];
            regions.push_back(vk::BufferCopy{
                .srcOffset = copy.source_offset,
                .dstOffset = base_offset + copy.download_offset,
                .size = copy.size,
            });
        }
        cmdbuf.copyBuffer(source, download_target->Handle(), regions);
        ++copy_commands;
        copy_regions += regions.size();
        first = end;
    }
    const vk::BufferMemoryBarrier2 host_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .buffer = download_target->Handle(),
        .offset = base_offset,
        .size = total_size_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });

    u64 tracked_ranges = 0;
    u64 tracked_pages = 0;
    scheduled_ranges.ForEach([&](const VAddr begin, const VAddr end) {
        ++tracked_ranges;
        const u64 range_size = end - begin;
        tracked_pages +=
            (range_size + static_cast<u64>(TRACKER_BYTES_PER_PAGE) - 1) /
            static_cast<u64>(TRACKER_BYTES_PER_PAGE);
    });
    const u64 completion_tick = scheduler.CurrentTick();
    const auto queued_at = std::chrono::steady_clock::now();
    scheduler.DeferPriorityOperationAt(
        completion_tick,
        [this, copies = std::move(copies), download, base_offset,
         total_size_bytes, completion_tick, queued_at, device_addr, size,
         is_write,
         overflow_download = std::move(overflow_download)]() mutable {
            const auto publish_start = std::chrono::steady_clock::now();
            RecordReadbackTimelineRetirement(completion_tick);
            if (overflow_download) {
                overflow_download->InvalidateMappedRange(0, total_size_bytes);
            } else {
                download_buffer.InvalidateMappedRange(base_offset,
                                                      total_size_bytes);
            }
            auto* guest_memory = Core::Memory::Instance();
            RangeSet published_ranges;
            bool published_any = false;
            bool primary_failed = false;
            for (const WaveCopy& copy : copies) {
                if (!guest_memory->TryWriteBacking(
                        std::bit_cast<u8*>(copy.guest_addr),
                        download + copy.download_offset, copy.size)) {
                    primary_failed |= copy.primary;
                    AccumulateReadbackOptimizationStats(
                        ReadbackOptimizationStats{
                            .fallbacks = 1,
                            .publication_failures = 1,
                        });
                    continue;
                }
                published_ranges.Add(copy.guest_addr, copy.size);
                published_any = true;
            }
            published_ranges.ForEach(
                [&](const VAddr begin, const VAddr end) {
                    gpu_modified_ranges.Subtract(begin, end - begin);
                    memory_tracker->UnmarkRegionAsGpuModified(begin,
                                                              end - begin);
                });
            if (is_write && !primary_failed) {
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
            }
            if (published_any) {
                NotifyGuestReadback();
                liverpool->NotifyMemoryPublicationForExecutor(device_addr,
                                                              size);
            }
            AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
                .cpu_publish_ns = ExecutorElapsedNs(publish_start),
                .gpu_completion_ns = ExecutorElapsedNs(queued_at),
            });
        });

    AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
        .snapshots = 1,
        .demand_snapshots = 1,
        .ranges = tracked_ranges,
        .pages = tracked_pages,
        .copy_commands = copy_commands,
        .copy_regions = copy_regions,
        .source_buffers = copy_commands,
        .regions_merged = planned_regions - copy_regions,
        .bytes = copied_bytes,
        .cpu_plan_ns = ExecutorElapsedNs(plan_start),
        .demand_flushes = 1,
    });
    ExecutorMaybeLogReadbackOptimizationStats();

    const auto wait_start = std::chrono::steady_clock::now();
    scheduler.Flush();
    scheduler.WaitRetirement(completion_tick);
    AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
        .cpu_wait_ns = ExecutorElapsedNs(wait_start),
    });
}
#endif

template <bool async>
void BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write) {
#ifdef __ANDROID__
    const auto plan_start = std::chrono::steady_clock::now();
#endif
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
#ifdef __ANDROID__
    if (download == nullptr) {
        AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
            .cpu_plan_ns = ExecutorElapsedNs(plan_start),
            .fallbacks = 1,
            .publication_failures = 1,
        });
        return;
    }
#endif
    for (auto& copy : copies) {
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::BufferReadback);
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
#ifdef __ANDROID__
    u64 copied_bytes = 0;
    u64 copied_pages = 0;
    for (const auto& copy : copies) {
        copied_bytes += copy.size;
        copied_pages +=
            (copy.size + static_cast<u64>(TRACKER_BYTES_PER_PAGE) - 1) /
            static_cast<u64>(TRACKER_BYTES_PER_PAGE);
    }
    const auto queued_at = std::chrono::steady_clock::now();
    AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
        .snapshots = 1,
        .demand_snapshots = 1,
        .ranges = static_cast<u64>(copies.size()),
        .pages = copied_pages,
        .copy_commands = 1,
        .copy_regions = static_cast<u64>(copies.size()),
        .bytes = copied_bytes,
        .cpu_plan_ns = ExecutorElapsedNs(plan_start),
    });
    ExecutorMaybeLogReadbackOptimizationStats();
#endif
    const auto write_data =
        [this, buffer_cpu_addr = buffer.CpuAddr(), copies, download, offset, device_addr, size,
         is_write
#ifdef __ANDROID__
         ,
         queued_at
#endif
    ]() {
#ifdef __ANDROID__
        const auto publish_start = std::chrono::steady_clock::now();
#endif
        auto* memory = Core::Memory::Instance();
        bool wrote_all = true;
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buffer_cpu_addr + copy.srcOffset;
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
#ifdef __ANDROID__
            AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
                .cpu_publish_ns = ExecutorElapsedNs(publish_start),
                .fallbacks = 1,
                .publication_failures = 1,
            });
#endif
            return;
        }
        NotifyGuestReadback();
        for (const auto& copy : copies) {
            gpu_modified_ranges.Subtract(buffer_cpu_addr + copy.srcOffset, copy.size);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
        liverpool->NotifyMemoryPublicationForExecutor(device_addr, size);
#ifdef __ANDROID__
        AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
            .timeline_waits = async ? 1u : 0u,
            .cpu_publish_ns = ExecutorElapsedNs(publish_start),
            .gpu_completion_ns = async ? ExecutorElapsedNs(queued_at) : 0u,
        });
#endif
    };
    if constexpr (async) {
        scheduler.DeferOperation(write_data);
    } else {
#ifdef __ANDROID__
        const auto wait_start = std::chrono::steady_clock::now();
#endif
        scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::BufferReadbackDirect);
#ifdef __ANDROID__
        const u64 wait_ns = ExecutorElapsedNs(wait_start);
        AccumulateReadbackOptimizationStats(ReadbackOptimizationStats{
            .timeline_waits = 1,
            .cpu_wait_ns = wait_ns,
            .gpu_completion_ns = wait_ns,
        });
#endif
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

    Vulkan::VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

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

    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        const auto [buffer, offset] =
            ObtainReadOnlyBuffer(range.base_address, static_cast<u32>(size),
                                 ReadOnlyBufferUsage::Vertex);
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
    if (executor_probe_bound) {
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

#ifdef __ANDROID__
void BufferCache::ExecutorMaybeLogBindDedup() {
    const u64 total_attempts = executor_vertex_input_bind_attempts +
                               executor_vertex_buffer_bind_attempts +
                               executor_index_buffer_bind_attempts;
    if (total_attempts < executor_bind_dedup_next_log) {
        return;
    }
    executor_bind_dedup_next_log =
        (total_attempts / 16384u + 1u) * 16384u;
    const auto hit_percent = [](const u64 attempts, const u64 hits) {
        return attempts != 0
                   ? 100.0 * static_cast<double>(hits) /
                         static_cast<double>(attempts)
                   : 0.0;
    };
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VK_BIND_DEDUP] kind=vertexIndex total=%llu "
        "input=%llu/%llu(%.1f%%) vertex=%llu/%llu(%.1f%%) "
        "index=%llu/%llu(%.1f%%)",
        static_cast<unsigned long long>(total_attempts),
        static_cast<unsigned long long>(executor_vertex_input_bind_hits),
        static_cast<unsigned long long>(executor_vertex_input_bind_attempts),
        hit_percent(executor_vertex_input_bind_attempts,
                    executor_vertex_input_bind_hits),
        static_cast<unsigned long long>(executor_vertex_buffer_bind_hits),
        static_cast<unsigned long long>(executor_vertex_buffer_bind_attempts),
        hit_percent(executor_vertex_buffer_bind_attempts,
                    executor_vertex_buffer_bind_hits),
        static_cast<unsigned long long>(executor_index_buffer_bind_hits),
        static_cast<unsigned long long>(executor_index_buffer_bind_attempts),
        hit_percent(executor_index_buffer_bind_attempts,
                    executor_index_buffer_bind_hits));
}
#endif

void BufferCache::BindPreparedVertexBuffers(const PreparedVertexBuffers& prepared) {
    const auto cmdbuf = scheduler.CommandBuffer();
#ifdef __ANDROID__
    const bool dedup_enabled = IsManagedVulkanVertexBindDeduplication();
    const u64 recording_tick = scheduler.CurrentTick();
    const u64 graphics_pipeline_epoch =
        scheduler.PipelineBindEpoch(vk::PipelineBindPoint::eGraphics);
    if (executor_vertex_bind_cmdbuf != cmdbuf ||
        executor_vertex_bind_recording_tick != recording_tick ||
        executor_vertex_bind_graphics_pipeline_epoch != graphics_pipeline_epoch) {
        executor_vertex_bind_cmdbuf = cmdbuf;
        executor_vertex_bind_recording_tick = recording_tick;
        executor_vertex_bind_graphics_pipeline_epoch = graphics_pipeline_epoch;
        // Qualcomm's proprietary driver can consume stale vertex input/buffer
        // state after a graphics-pipeline transition. Preserve deduplication
        // within the pipeline epoch, but replay both binds at its boundary.
        executor_vertex_input_bind_valid = false;
        executor_vertex_buffer_bind_valid = false;
    }

    if (prepared.dynamic_vertex_input) {
        ++executor_vertex_input_bind_attempts;
        const u64 input_epoch = scheduler.VertexInputEpoch();
        const bool input_hit =
            dedup_enabled &&
            executor_vertex_input_bind_valid &&
            executor_vertex_input_bind_epoch == input_epoch &&
            std::ranges::equal(executor_bound_vertex_state.bindings,
                               prepared.bindings) &&
            std::ranges::equal(executor_bound_vertex_state.attributes,
                               prepared.attributes);
        executor_vertex_input_bind_hits += static_cast<u64>(input_hit);
        if (!input_hit) {
            cmdbuf.setVertexInputEXT(prepared.bindings, prepared.attributes);
            executor_vertex_input_bind_epoch = scheduler.MarkVertexInput();
            executor_bound_vertex_state.bindings = prepared.bindings;
            executor_bound_vertex_state.attributes = prepared.attributes;
            executor_vertex_input_bind_valid = true;
        }
    }
#else
    if (prepared.dynamic_vertex_input) {
        cmdbuf.setVertexInputEXT(prepared.bindings, prepared.attributes);
        scheduler.MarkVertexInput();
    }
#endif
    if (prepared.buffers.empty()) {
#ifdef __ANDROID__
        ExecutorMaybeLogBindDedup();
#endif
        return;
    }

    const auto num_buffers = static_cast<u32>(prepared.buffers.size());
#ifdef __ANDROID__
    ++executor_vertex_buffer_bind_attempts;
    const u64 vertex_buffer_epoch = scheduler.VertexBufferBindEpoch();
    const bool buffer_hit =
        dedup_enabled &&
        executor_vertex_buffer_bind_valid &&
        executor_vertex_buffer_bind_epoch == vertex_buffer_epoch &&
        executor_bound_vertex_state.dynamic_vertex_input ==
            prepared.dynamic_vertex_input &&
        std::ranges::equal(executor_bound_vertex_state.buffers,
                           prepared.buffers) &&
        std::ranges::equal(executor_bound_vertex_state.offsets,
                           prepared.offsets) &&
        (prepared.dynamic_vertex_input ||
         (std::ranges::equal(executor_bound_vertex_state.sizes,
                             prepared.sizes) &&
          std::ranges::equal(executor_bound_vertex_state.strides,
                             prepared.strides)));
    executor_vertex_buffer_bind_hits += static_cast<u64>(buffer_hit);
    if (!buffer_hit) {
#endif
        [[maybe_unused]] u64 post_vertex_buffer_epoch{};
        if (prepared.dynamic_vertex_input) {
            cmdbuf.bindVertexBuffers(0, num_buffers, prepared.buffers.data(),
                                     prepared.offsets.data());
            post_vertex_buffer_epoch = scheduler.MarkVertexBufferBind();
        } else {
            cmdbuf.bindVertexBuffers2(0, num_buffers, prepared.buffers.data(),
                                      prepared.offsets.data(), prepared.sizes.data(),
                                      prepared.strides.data());
            post_vertex_buffer_epoch = scheduler.MarkVertexBufferBind();
        }
#ifdef __ANDROID__
        executor_vertex_buffer_bind_epoch = post_vertex_buffer_epoch;
        executor_bound_vertex_state.buffers = prepared.buffers;
        executor_bound_vertex_state.offsets = prepared.offsets;
        executor_bound_vertex_state.sizes = prepared.sizes;
        executor_bound_vertex_state.strides = prepared.strides;
        executor_bound_vertex_state.dynamic_vertex_input =
            prepared.dynamic_vertex_input;
        executor_vertex_buffer_bind_valid = true;
    }
    ExecutorMaybeLogBindDedup();
#endif
}

BufferCache::PreparedIndexBuffer BufferCache::PrepareIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers,
    const bool executor_probe_bound, const u64 executor_probe_sequence) {
    const auto& regs = liverpool->regs;

    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [vk_buffer, offset] =
        ObtainReadOnlyBuffer(index_address, index_buffer_size, ReadOnlyBufferUsage::Index);
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
    const auto cmdbuf = scheduler.CommandBuffer();
#ifdef __ANDROID__
    const bool dedup_enabled = IsManagedVulkanVertexBindDeduplication();
    const u64 recording_tick = scheduler.CurrentTick();
    const u64 graphics_pipeline_epoch =
        scheduler.PipelineBindEpoch(vk::PipelineBindPoint::eGraphics);
    if (executor_index_bind_cmdbuf != cmdbuf ||
        executor_index_bind_recording_tick != recording_tick ||
        executor_index_bind_graphics_pipeline_epoch != graphics_pipeline_epoch) {
        executor_index_bind_cmdbuf = cmdbuf;
        executor_index_bind_recording_tick = recording_tick;
        executor_index_bind_graphics_pipeline_epoch = graphics_pipeline_epoch;
        executor_index_buffer_bind_valid = false;
    }
    ++executor_index_buffer_bind_attempts;
    const u64 index_buffer_epoch = scheduler.IndexBufferBindEpoch();
    const bool index_hit =
        dedup_enabled &&
        executor_index_buffer_bind_valid &&
        executor_index_buffer_bind_epoch == index_buffer_epoch &&
        executor_bound_index_state.buffer == prepared.buffer &&
        executor_bound_index_state.offset == prepared.offset &&
        executor_bound_index_state.type == prepared.type;
    executor_index_buffer_bind_hits += static_cast<u64>(index_hit);
    if (!index_hit) {
#endif
        cmdbuf.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
#ifdef __ANDROID__
        executor_index_buffer_bind_epoch = scheduler.MarkIndexBufferBind();
        executor_bound_index_state = prepared;
        executor_index_buffer_bind_valid = true;
    }
    ExecutorMaybeLogBindDedup();
#else
        scheduler.MarkIndexBufferBind();
#endif
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    if (num_bytes == 0) {
        return;
    }
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

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds,
                             bool defer_gnm_dma) {
    if (num_bytes == 0) {
        return;
    }
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            return;
        }
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
#ifdef __ANDROID__
    if (defer_gnm_dma && IsManagedVulkanTransferCoalescing()) {
        scheduler.QueueBufferCopies(
            src_buffer.Handle(), dst_buffer.Handle(), std::span<const vk::BufferCopy>{&region, 1},
            src_gds ? VAddr{0} : src, src_gds ? u64{0} : num_bytes,
            dst_gds ? VAddr{0} : dst, dst_gds ? u64{0} : num_bytes,
            Vulkan::Scheduler::BufferTransferOrigin::GnmDma);
        return;
    }
#else
    (void)defer_gnm_dma;
#endif
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
    scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::BufferCopy);
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
                                                  bool is_texel_buffer, BufferId buffer_id,
                                                  const bool transfer_range_checked) {
    if (!transfer_range_checked) {
        scheduler.FlushPendingBufferTransfersForRange(device_addr, size);
    }
#ifdef __ANDROID__
    ExecutorFlagSuspectGpuDescriptor("ObtainBuffer", device_addr, size);
#endif
    if (!is_written && Libraries::GnmDriver::ExecutorGnmCaptureActive()) {
        Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(3u, device_addr, size);
    }
    Libraries::GnmDriver::ExecutorReplayCheckAddr(3u, device_addr, size);
    if (IsBufferInvalid(buffer_id)) {
        buffer_id = FindBuffer(device_addr, size, true);
    }
    Buffer& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer);
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
    }
    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::ObtainReadOnlyBuffer(
    const VAddr device_addr, const u32 size, const ReadOnlyBufferUsage usage,
    const BufferId fallback_buffer_id) {
#ifndef __ANDROID__
    (void)usage;
    return ObtainBuffer(device_addr, size, false, false, fallback_buffer_id);
#else
    ASSERT(usage < ReadOnlyBufferUsage::Count);
    scheduler.FlushPendingBufferTransfersForRange(device_addr, size);
    if (!ExecutorReadOnlyRenameEnabled()) {
        return ObtainBuffer(device_addr, size, false, false,
                            fallback_buffer_id);
    }

    ++read_only_rename_attempts;
    ++read_only_rename_usage_counts[static_cast<std::size_t>(usage)];
    const auto maybe_log = [&] {
        if (Config::managedGpuFastPath()) {
            return;
        }
        if (read_only_rename_attempts != 256 &&
            (read_only_rename_attempts & 4095u) != 0u) {
            return;
        }
        const auto fallback_count = [&](const ReadOnlyRenameFallback reason) {
            return read_only_rename_fallback_counts[static_cast<std::size_t>(reason)];
        };
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_READ_ONLY_RENAME] attempts=%llu renamed=%llu bytes=%llu "
            "usage=vtx:%llu,idx:%llu,ubo:%llu,ssbo:%llu "
            "renamedUsage=vtx:%llu,idx:%llu,ubo:%llu,ssbo:%llu "
            "fallback=gpu:%llu,image:%llu,clean:%llu,read:%llu,large:%llu,"
            "coherent:%llu,budget:%llu,ring:%llu,copy:%llu",
            static_cast<unsigned long long>(read_only_rename_attempts),
            static_cast<unsigned long long>(read_only_rename_ranges),
            static_cast<unsigned long long>(read_only_rename_bytes),
            static_cast<unsigned long long>(
                read_only_rename_usage_counts[static_cast<std::size_t>(
                    ReadOnlyBufferUsage::Vertex)]),
            static_cast<unsigned long long>(
                read_only_rename_usage_counts[static_cast<std::size_t>(
                    ReadOnlyBufferUsage::Index)]),
            static_cast<unsigned long long>(
                read_only_rename_usage_counts[static_cast<std::size_t>(
                    ReadOnlyBufferUsage::Uniform)]),
            static_cast<unsigned long long>(
                read_only_rename_usage_counts[static_cast<std::size_t>(
                    ReadOnlyBufferUsage::Storage)]),
            static_cast<unsigned long long>(
                read_only_rename_success_usage_counts[static_cast<std::size_t>(
                    ReadOnlyBufferUsage::Vertex)]),
            static_cast<unsigned long long>(
                read_only_rename_success_usage_counts[static_cast<std::size_t>(
                    ReadOnlyBufferUsage::Index)]),
            static_cast<unsigned long long>(
                read_only_rename_success_usage_counts[static_cast<std::size_t>(
                    ReadOnlyBufferUsage::Uniform)]),
            static_cast<unsigned long long>(
                read_only_rename_success_usage_counts[static_cast<std::size_t>(
                    ReadOnlyBufferUsage::Storage)]),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::GpuModified)),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::ImageAlias)),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::NotCpuDirty)),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::NotReadable)),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::TooLarge)),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::StreamNotCoherent)),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::TickBudget)),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::RingBusy)),
            static_cast<unsigned long long>(
                fallback_count(ReadOnlyRenameFallback::CopyFailed)));
    };
    const auto fallback = [&](const ReadOnlyRenameFallback reason) {
        ++read_only_rename_fallback_counts[static_cast<std::size_t>(reason)];
        maybe_log();
        return ObtainBuffer(device_addr, size, false, false,
                            fallback_buffer_id, true);
    };

    if (!ExecutorReadOnlyRenameUsageEnabled(usage)) {
        return ObtainBuffer(device_addr, size, false, false,
                            fallback_buffer_id, true);
    }

    if (device_addr == 0 || size == 0) {
        ++read_only_rename_fallback_counts[static_cast<std::size_t>(
            ReadOnlyRenameFallback::NotReadable)];
        maybe_log();
        return {&GetBuffer(NULL_BUFFER_ID), 0};
    }
    if (IsRegionGpuModified(device_addr, size) ||
        gpu_modified_ranges.Intersects(device_addr, size)) {
        return fallback(ReadOnlyRenameFallback::GpuModified);
    }
    const bool replay_memory = Libraries::GnmDriver::ExecutorReplayActive();
    if (IsRegionRegistered(device_addr, size) &&
        !IsRegionCpuModified(device_addr, size) && !replay_memory) {
        return fallback(ReadOnlyRenameFallback::NotCpuDirty);
    }
    if (texture_cache.FindImageFromRange(device_addr, size)) {
        return fallback(ReadOnlyRenameFallback::ImageAlias);
    }
    if (!memory->IsReadableMapping(device_addr, size) &&
        !replay_memory) {
        return fallback(ReadOnlyRenameFallback::NotReadable);
    }

    // The stream ring is a win for small per-draw constants, but putting
    // multi-megabyte vertex/index ranges in it causes frequent wraparound and
    // lets Qualcomm observe a recycled range while an older draw is still in
    // flight. Keep the original one-cache-page safety contract for every
    // read-only usage; large geometry remains in the tracked buffer cache.
    static constexpr u64 MaxRenameRangeBytes = CACHING_PAGESIZE;
    static constexpr u64 MaxRenameBytesPerTick = 16_MB;
    if (size > MaxRenameRangeBytes || size > stream_buffer.SizeBytes()) {
        return fallback(ReadOnlyRenameFallback::TooLarge);
    }
    if (!stream_buffer.is_coherent || stream_buffer.mapped_data.empty()) {
        return fallback(ReadOnlyRenameFallback::StreamNotCoherent);
    }

    const u64 current_tick = scheduler.CurrentTick();
    if (read_only_rename_tick != current_tick) {
        read_only_rename_tick = current_tick;
        read_only_rename_tick_bytes = 0;
    }
    if (size > MaxRenameBytesPerTick -
                   std::min(read_only_rename_tick_bytes, MaxRenameBytesPerTick)) {
        return fallback(ReadOnlyRenameFallback::TickBudget);
    }

    const u64 alignment = [&] {
        switch (usage) {
        case ReadOnlyBufferUsage::Vertex:
            return u64{16};
        case ReadOnlyBufferUsage::Index:
            return u64{4};
        case ReadOnlyBufferUsage::Uniform:
            return static_cast<u64>(instance.UniformMinAlignment());
        case ReadOnlyBufferUsage::Storage:
            return static_cast<u64>(instance.StorageMinAlignment());
        case ReadOnlyBufferUsage::Count:
            break;
        }
        UNREACHABLE();
        return u64{16};
    }();
    const auto [stream_data, stream_offset] = stream_buffer.Map(size, alignment, false);
    if (stream_data == nullptr) {
        return fallback(ReadOnlyRenameFallback::RingBusy);
    }
    if (!memory->TryCopyReadableMemory(device_addr, stream_data, size, false)) {
        std::memset(stream_data, 0, size);
        stream_buffer.Commit();
        stream_buffer.RetainCurrentAllocation();
        return fallback(ReadOnlyRenameFallback::CopyFailed);
    }
    stream_buffer.Commit();
    stream_buffer.RetainCurrentAllocation();
    read_only_rename_tick_bytes += size;
    ++read_only_rename_ranges;
    ++read_only_rename_success_usage_counts[static_cast<std::size_t>(usage)];
    read_only_rename_bytes += size;
    maybe_log();
    ASSERT(stream_offset <= std::numeric_limits<u32>::max());
    return {&stream_buffer, static_cast<u32>(stream_offset)};
#endif
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
#ifdef __ANDROID__
    ExecutorFlagSuspectGpuDescriptor("ObtainBufferForImage", gpu_addr, size);
#endif
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    if (IsRegionGpuModified(gpu_addr, size)) {
        return ObtainBuffer(gpu_addr, size, false, false);
    }
    const auto [data, offset] = staging_buffer.Map(size, 16);
    if (!memory->TryCopyBackingMemory(gpu_addr, data, size)) {
        memory->CopySparseMemory(gpu_addr, data, size);
    }
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
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

    scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::BufferProbe);
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

    scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::BufferProbe);
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

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size,
                                 const bool transfer_range_checked) {
    if (device_addr == 0) {
        return NULL_BUFFER_ID;
    }
    if (!transfer_range_checked) {
        scheduler.FlushPendingBufferTransfersForRange(device_addr, size);
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
    scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::BufferCopy);
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
    // Creating a wider cache buffer can absorb and retire buffers outside the
    // requested range. Materialize queued copies before changing that topology.
    scheduler.FlushPendingBufferTransfers();
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
        if (Config::directMemoryAccess()) {
            boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
            bda_addrs.reserve(size_pages);
            for (u64 i = 0; i < size_pages; ++i) {
                vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
                bda_addrs.push_back(addr);
            }
            WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                            bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        }
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        if (Config::directMemoryAccess()) {
            const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
            bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
        }
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    std::unique_ptr<Buffer> transient_upload_buffer;
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] {
            src_buffer = UploadCopies(
                buffer, copies, total_size_bytes, transient_upload_buffer);
        });

    if (src_buffer) {
        ASSERT(!copies.empty());
        vk::DeviceSize upload_begin = copies.front().dstOffset;
        vk::DeviceSize upload_end = upload_begin + copies.front().size;
        for (const auto& copy : copies) {
            upload_begin = std::min(upload_begin, copy.dstOffset);
            upload_end = std::max(upload_end, copy.dstOffset + copy.size);
        }
#ifdef __ANDROID__
        if (IsManagedVulkanTransferCoalescing()) {
            scheduler.QueueBufferUploadCopies(
                src_buffer, buffer.buffer, copies,
                buffer_start + upload_begin, upload_end - upload_begin);
            if (transient_upload_buffer) {
                scheduler.FlushPendingBufferTransfers();
            }
        } else
#endif
        {
            scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::BufferUpload);
            const auto cmdbuf = scheduler.CommandBuffer();
            const vk::BufferMemoryBarrier2 pre_barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask =
                    vk::AccessFlagBits2::eMemoryRead |
                    vk::AccessFlagBits2::eMemoryWrite |
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
                .dstAccessMask =
                    vk::AccessFlagBits2::eMemoryRead |
                    vk::AccessFlagBits2::eMemoryWrite,
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
        }
        if (transient_upload_buffer) {
            scheduler.DeferOperation(
                [upload = std::move(transient_upload_buffer)]() mutable {
                    upload.reset();
                });
        }
        TouchBuffer(buffer);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeBufferFromImage(buffer, device_addr, size);
    }
    return false;
}

vk::Buffer BufferCache::UploadCopies(
    Buffer& buffer, std::span<vk::BufferCopy> copies,
    size_t total_size_bytes, std::unique_ptr<Buffer>& transient_buffer) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            // These ranges were selected as CPU-modified by MemoryTracker, so
            // the physical backing is authoritative. Read it through the
            // unprotected alias: the guest VA page can remain read-protected
            // solely because a disjoint range on the same 4-KiB page is still
            // GPU-modified.
            if (!memory->TryCopyBackingMemory(device_addr, src_pointer, copy.size)) {
                memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            }
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        transient_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = transient_buffer->Handle();
        u8* const staging = transient_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            if (!memory->TryCopyBackingMemory(device_addr, src_pointer, copy.size)) {
                memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            }
        }
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
    scheduler.FlushPendingBufferTransfersForRange(device_addr, size);
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
    if (num_bytes == 0) {
        return;
    }
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    std::unique_ptr<Buffer> transient_upload_buffer;
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        transient_upload_buffer = std::make_unique<Buffer>(
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes);
        src_buffer = transient_upload_buffer->Handle();
        u8* const staging = transient_upload_buffer->mapped_data.data();
        std::memcpy(staging, value, num_bytes);
    }
#ifdef __ANDROID__
    if (IsManagedVulkanTransferCoalescing()) {
        const bool has_guest_range = buffer.CpuAddr() != 0;
        scheduler.QueueBufferUploadCopies(
            src_buffer, buffer.Handle(), std::span{&copy, 1},
            has_guest_range ? address : 0, has_guest_range ? num_bytes : 0);
        if (transient_upload_buffer) {
            scheduler.FlushPendingBufferTransfers();
        }
    } else
#endif
    {
        scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::BufferUpload);
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
            .dstAccessMask =
                vk::AccessFlagBits2::eMemoryRead |
                vk::AccessFlagBits2::eMemoryWrite,
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
    if (transient_upload_buffer) {
        scheduler.DeferOperation(
            [upload = std::move(transient_upload_buffer)]() mutable {
                upload.reset();
            });
    }
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
    scheduler.FlushPendingBufferTransfers();
    const bool aggressive = total_used_memory >= critical_gc_memory;
#ifdef __ANDROID__
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 16 : 64, gc_tick);
    int max_deletions = aggressive ? 48 : 16;
#else
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
#endif
    const int initial_max_deletions = max_deletions;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return true;
        }
        --max_deletions;
        Buffer& buffer = slot_buffers[buffer_id];
        DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes(), true);
        DeleteBuffer(buffer_id);
        return false;
    };
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    VideoCore::RecordMobileGpuResidencyGc(
        static_cast<u64>(initial_max_deletions - max_deletions));
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    // Pending buffer-transfer batches retain raw Vulkan handles until their commands are
    // materialized. Never let a cache slot enter deferred destruction first.
    scheduler.FlushPendingBufferTransfers();
    Buffer& buffer = slot_buffers[buffer_id];
    Unregister(buffer_id);
    scheduler.DeferOperation([this, buffer_id] { slot_buffers.erase(buffer_id); });
    buffer.is_deleted = true;
}

}
