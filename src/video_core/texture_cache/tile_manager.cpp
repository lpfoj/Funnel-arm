// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/tile_manager.h"
#include "core/libraries/gnmdriver/gnmdriver.h"
#include "core/memory.h"
#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include "video_core/host_shaders/tiling_comp.h"

#include <magic_enum/magic_enum.hpp>
#include <vk_mem_alloc.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace VideoCore {

struct TilingInfo {
    u32 bank_swizzle;
    u32 num_slices;
    u32 num_mips;
    std::array<ImageInfo::MipInfo, 16> mips;
};

namespace {

constexpr u64 ScratchPoolMaximumBudgetBytes = 128ull * 1024ull * 1024ull;
constexpr u64 ScratchPoolMinimumBucketBytes = 4ull * 1024ull;

u64 ScratchBucketSize(const u32 requested_size, const u64 budget_bytes) noexcept {
    if (requested_size == 0) {
        return 0;
    }
    const u64 normalized =
        std::max<u64>(requested_size, ScratchPoolMinimumBucketBytes);
    const u64 bucket = std::bit_ceil(normalized);
    return bucket <= budget_bytes ? bucket : 0;
}

} // namespace

struct TileManager::ScratchPoolState {
    struct IdleEntry {
        VkBuffer buffer{VK_NULL_HANDLE};
        VmaAllocation allocation{};
        u64 bucket{};
        size_t bucket_slot{};
    };

    using IdleIterator = std::list<IdleEntry>::iterator;

    struct BucketState {
        std::vector<IdleIterator> entries;
        u32 miss_pressure{};
    };

    struct ReleasedEntry {
        VkBuffer buffer{VK_NULL_HANDLE};
        VmaAllocation allocation{};
    };

    ScratchPoolState(const VmaAllocator allocator_, const u64 budget_bytes_)
        : allocator{allocator_}, budget_bytes{budget_bytes_} {}

    u64 BucketSize(const u32 requested_size) const noexcept {
        return ScratchBucketSize(requested_size, budget_bytes);
    }

    bool TryAcquire(const u64 bucket, ScratchBuffer& scratch) {
        if (bucket == 0) {
            return false;
        }
        {
            std::scoped_lock lock{mutex};
            if (stopping) {
                return false;
            }
            auto& state = buckets[bucket];
            if (state.entries.empty()) {
                state.miss_pressure =
                    std::min(state.miss_pressure + 1, MaximumMissPressure);
                misses.fetch_add(1, std::memory_order_relaxed);
                return false;
            }

            const IdleIterator idle = state.entries.back();
            state.entries.pop_back();
            scratch = {
                .buffer = vk::Buffer{idle->buffer},
                .allocation = idle->allocation,
                .bucket = bucket,
                .reused = true,
            };
            idle_buffers.erase(idle);
            pooled_bytes.fetch_sub(bucket, std::memory_order_relaxed);
            pooled_buffers.fetch_sub(1, std::memory_order_relaxed);
            if (state.miss_pressure != 0) {
                --state.miss_pressure;
            }
        }
        reused.fetch_add(1, std::memory_order_relaxed);
        MaybeLog("reuse");
        return true;
    }

    bool TryPool(const ScratchBuffer& scratch) {
        const u64 bucket = scratch.bucket;
        if (bucket == 0 || bucket > budget_bytes) {
            rejected.fetch_add(1, std::memory_order_relaxed);
            MaybeLog("destroy_unpooled");
            return false;
        }

        std::vector<ReleasedEntry> evicted_entries;
        bool admitted = false;
        try {
            std::scoped_lock lock{mutex};
            if (!stopping) {
                auto& target = buckets[bucket];
                u64 current_bytes = pooled_bytes.load(std::memory_order_relaxed);
                bool may_admit = true;

                if (current_bytes > budget_bytes - bucket) {
                    const u64 bytes_needed = current_bytes - (budget_bytes - bucket);
                    const u64 same_bucket_bytes =
                        static_cast<u64>(target.entries.size()) * bucket;
                    const u64 evictable_bytes = current_bytes - same_bucket_bytes;
                    const u32 admission_misses = RequiredAdmissionMisses(bucket);

                    // Do not let a one-shot large texture flush a warmed cache. A
                    // buffer acquired from this pool always gets its slot back;
                    // newly created size classes must demonstrate repeated misses.
                    may_admit =
                        evictable_bytes >= bytes_needed &&
                        (scratch.reused || target.miss_pressure >= admission_misses);

                    if (may_admit) {
                        evicted_entries.reserve(8);
                        while (current_bytes > budget_bytes - bucket) {
                            const auto oldest = std::ranges::find_if(
                                idle_buffers,
                                [bucket](const IdleEntry& entry) {
                                    return entry.bucket != bucket;
                                });
                            if (oldest == idle_buffers.end()) {
                                may_admit = false;
                                break;
                            }

                            const u64 evicted_bucket = oldest->bucket;
                            evicted_entries.push_back(
                                {oldest->buffer, oldest->allocation});
                            RemoveIdle(oldest);
                            current_bytes -= evicted_bucket;
                            pooled_bytes.store(current_bytes,
                                               std::memory_order_relaxed);
                            pooled_buffers.fetch_sub(1,
                                                     std::memory_order_relaxed);
                        }
                        if (may_admit && !scratch.reused) {
                            target.miss_pressure -= admission_misses;
                        }
                    }
                }

                if (may_admit) {
                    idle_buffers.push_back({
                        .buffer = static_cast<VkBuffer>(scratch.buffer),
                        .allocation = scratch.allocation,
                        .bucket = bucket,
                        .bucket_slot = target.entries.size(),
                    });
                    auto idle = idle_buffers.end();
                    --idle;
                    try {
                        target.entries.push_back(idle);
                    } catch (...) {
                        idle_buffers.erase(idle);
                        throw;
                    }
                    pooled_bytes.store(current_bytes + bucket,
                                       std::memory_order_relaxed);
                    pooled_buffers.fetch_add(1, std::memory_order_relaxed);
                    admitted = true;
                }
            }
        } catch (...) {
        }

        for (const ReleasedEntry& entry : evicted_entries) {
            vmaDestroyBuffer(allocator, entry.buffer, entry.allocation);
        }
        if (!evicted_entries.empty()) {
            evicted.fetch_add(evicted_entries.size(), std::memory_order_relaxed);
        }
        if (!admitted) {
            rejected.fetch_add(1, std::memory_order_relaxed);
        }
        MaybeLog(admitted ? (evicted_entries.empty() ? "retire_to_pool"
                                                     : "retire_with_evict")
                          : "destroy_rejected");
        return admitted;
    }

    void NoteCreated() {
        created.fetch_add(1, std::memory_order_relaxed);
        MaybeLog("create");
    }

    void StopAndDestroy() {
        std::list<IdleEntry> retired_entries;
        {
            std::scoped_lock lock{mutex};
            stopping = true;
            retired_entries.splice(retired_entries.end(), idle_buffers);
            buckets.clear();
            pooled_bytes.store(0, std::memory_order_relaxed);
            pooled_buffers.store(0, std::memory_order_relaxed);
        }
        for (const IdleEntry& entry : retired_entries) {
            vmaDestroyBuffer(allocator, entry.buffer, entry.allocation);
        }
    }

    void MaybeLog(const char* event) {
#ifdef __ANDROID__
        const u64 sequence = log_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
        if (sequence != 1 && (sequence & 255u) != 0) {
            return;
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_TILE_SCRATCH_POOL] event=%s created=%llu reused=%llu "
            "pooledBytes=%llu budgetBytes=%llu misses=%llu evicted=%llu "
            "rejected=%llu pooledBuffers=%llu",
            event,
            static_cast<unsigned long long>(created.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(reused.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(pooled_bytes.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(budget_bytes),
            static_cast<unsigned long long>(misses.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(evicted.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(rejected.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                pooled_buffers.load(std::memory_order_relaxed)));
#else
        (void)event;
#endif
    }

    u32 RequiredAdmissionMisses(const u64 bucket) const noexcept {
        const u64 admission_unit =
            std::max(budget_bytes / 8, ScratchPoolMinimumBucketBytes);
        const u64 proportional_misses =
            (bucket + admission_unit - 1) / admission_unit;
        return static_cast<u32>(std::clamp<u64>(proportional_misses, 2, 8));
    }

    void RemoveIdle(const IdleIterator idle) {
        auto& state = buckets.at(idle->bucket);
        const size_t slot = idle->bucket_slot;
        const size_t last_slot = state.entries.size() - 1;
        if (slot != last_slot) {
            state.entries[slot] = state.entries[last_slot];
            state.entries[slot]->bucket_slot = slot;
        }
        state.entries.pop_back();
        idle_buffers.erase(idle);
    }

    static constexpr u32 MaximumMissPressure = 255;

    VmaAllocator allocator{};
    u64 budget_bytes{};
    std::mutex mutex;
    std::list<IdleEntry> idle_buffers;
    std::unordered_map<u64, BucketState> buckets;
    std::atomic<u64> created{};
    std::atomic<u64> reused{};
    std::atomic<u64> misses{};
    std::atomic<u64> evicted{};
    std::atomic<u64> rejected{};
    std::atomic<u64> pooled_buffers{};
    std::atomic<u64> pooled_bytes{};
    std::atomic<u64> log_sequence{};
    bool stopping{};
};

TileManager::TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         StreamBuffer& stream_buffer_)
    : instance{instance}, scheduler{scheduler}, stream_buffer{stream_buffer_},
      scratch_pool{std::make_shared<ScratchPoolState>(
          instance.GetAllocator(),
          std::min(ScratchPoolMaximumBudgetBytes,
                   std::max(ScratchPoolMinimumBucketBytes,
                            instance.GetTotalMemoryBudget() / 32)))}
#ifdef __ANDROID__
      ,
      detile_command_pool{std::make_unique<Vulkan::CommandPool>(
          instance, scheduler.GetMasterSemaphore(),
          vk::CommandBufferLevel::eSecondary)}
#endif
{
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        {
            .binding = 2,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    }};

    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto desc_layout_result = device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(desc_layout_result.result == vk::Result::eSuccess,
               "Failed to create descriptor set layout: {}",
               vk::to_string(desc_layout_result.result));
    desc_layout = std::move(desc_layout_result.value);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 0U,
        .pPushConstantRanges = nullptr,
    };
    auto [layout_result, layout] = device.createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create pipeline layout: {}",
               vk::to_string(layout_result));
    pl_layout = std::move(layout);
}

TileManager::~TileManager() {
    scratch_pool->StopAndDestroy();
}

TileManager::ScratchBuffer TileManager::GetScratchBuffer(u32 size) {
    const u64 bucket = scratch_pool->BucketSize(size);
    ScratchBuffer scratch{};
    if (bucket != 0 && scratch_pool->TryAcquire(bucket, scratch)) {
        return scratch;
    }

    constexpr auto usage =
        vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;

    const vk::BufferCreateInfo buffer_ci = {
        .size = bucket != 0 ? bucket : size,
        .usage = usage,
    };

    const VmaAllocationCreateInfo alloc_info{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };

    VkBuffer buffer;
    VmaAllocation allocation;
    const auto buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
    const auto result = vmaCreateBuffer(instance.GetAllocator(), &buffer_ci_unsafe, &alloc_info,
                                        &buffer, &allocation, nullptr);
    ASSERT(result == VK_SUCCESS);
    scratch_pool->NoteCreated();
    return {
        .buffer = vk::Buffer{buffer},
        .allocation = allocation,
        .bucket = bucket,
        .reused = false,
    };
}

void TileManager::RetireScratchBuffer(ScratchBuffer scratch) {
    const std::shared_ptr<ScratchPoolState> pool = scratch_pool;
    const VmaAllocator allocator = instance.GetAllocator();
    // Return the allocation only after its command buffer retires. Pool hits
    // avoid both VMA allocation and Adreno object destruction in the hot
    // detile path; rejected entries retain the original deferred-destruction
    // lifetime.
    scheduler.DeferOperation([pool, allocator, scratch] {
        if (!pool->TryPool(scratch)) {
            vmaDestroyBuffer(allocator, static_cast<VkBuffer>(scratch.buffer),
                             scratch.allocation);
        }
    });
}

vk::Pipeline TileManager::GetTilingPipeline(const ImageInfo& info, bool is_tiler) {
    const u32 pl_id = u32(info.tile_mode) * NUM_BPPS + std::bit_width(info.num_bits) - 4;
    auto& tiling_pipelines = is_tiler ? tilers : detilers;
#ifdef __ANDROID__
    if (Libraries::GnmDriver::ExecutorReplayActive())
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_DETILE_PL] pl_id=%u arraySize=%zu tile_mode=%u num_bits=%u",
                            pl_id, tiling_pipelines.size(), (unsigned)info.tile_mode, info.num_bits);
#endif
    if (auto pipeline = *tiling_pipelines[pl_id]; pipeline != VK_NULL_HANDLE) {
        return pipeline;
    }
#ifdef __ANDROID__
    if (Libraries::GnmDriver::ExecutorReplayActive())
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_TILE] post_deref array_mode=%u",
                            (unsigned)info.array_mode);
#endif

    const auto device = instance.GetDevice();
    const auto micro_tile_mode = AmdGpu::GetMicroTileMode(info.tile_mode);
    std::vector<std::string> defines = {
        fmt::format("BITS_PER_PIXEL={}", info.num_bits),
        fmt::format("NUM_SAMPLES={}", info.num_samples),
        fmt::format("ARRAY_MODE={}", u32(info.array_mode)),
        fmt::format("MICRO_TILE_MODE={}", u32(micro_tile_mode)),
        fmt::format("MICRO_TILE_THICKNESS={}", AmdGpu::GetMicroTileThickness(info.array_mode)),
    };
#ifdef __ANDROID__
#define EXEC_TL_MARK(s) do { if (Libraries::GnmDriver::ExecutorReplayActive()) \
    __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_TILE] " s); } while(0)
#else
#define EXEC_TL_MARK(s) ((void)0)
#endif
    EXEC_TL_MARK("after_microtile");
    if (AmdGpu::IsMacroTiled(info.array_mode)) {
        EXEC_TL_MARK("macro:before_calcmode");
        const auto macro_tile_mode =
            AmdGpu::CalculateMacrotileMode(info.tile_mode, info.num_bits, info.num_samples);
        EXEC_TL_MARK("macro:before_numbanks");
        const u32 num_banks = AmdGpu::GetNumBanks(macro_tile_mode);
        defines.emplace_back(
            fmt::format("PIPE_CONFIG={}", u32(AmdGpu::GetPipeConfig(info.tile_mode))));
        defines.emplace_back(fmt::format("BANK_WIDTH={}", AmdGpu::GetBankWidth(macro_tile_mode)));
        defines.emplace_back(fmt::format("BANK_HEIGHT={}", AmdGpu::GetBankHeight(macro_tile_mode)));
        defines.emplace_back(fmt::format("NUM_BANKS={}", num_banks));
        defines.emplace_back(fmt::format("NUM_BANK_BITS={}", std::bit_width(num_banks) - 1));
        defines.emplace_back(fmt::format(
            "TILE_SPLIT_BYTES={}", AmdGpu::CalculateTileSplit(info.tile_mode, info.array_mode,
                                                              micro_tile_mode, info.num_bits)));
        defines.emplace_back(
            fmt::format("MACRO_TILE_ASPECT={}", AmdGpu::GetMacrotileAspect(macro_tile_mode)));
    }
    if (is_tiler) {
        defines.emplace_back(fmt::format("IS_TILER=1"));
    }

    EXEC_TL_MARK("before_vkcompile");
    const auto& module = Vulkan::Compile(HostShaders::TILING_COMP,
                                         vk::ShaderStageFlagBits::eCompute, device, defines);
    EXEC_TL_MARK("after_vkcompile");
    const auto module_name = fmt::format("{}_{} {}", magic_enum::enum_name(info.tile_mode),
                                         info.num_bits, is_tiler ? "tiler" : "detiler");
    EXEC_TL_MARK("after_modulename");
    LOG_INFO(Render_Vulkan, "Compiling shader {}", module_name);
    for (const auto& def : defines) {
        LOG_INFO(Render_Vulkan, "#define {}", def);
    }
    Vulkan::SetObjectName(device, module, module_name);
    EXEC_TL_MARK("after_setname before_createpipeline");
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };
    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .stage = shader_ci,
        .layout = *pl_layout,
    };
    auto [result, pipeline] =
        device.createComputePipelineUnique(VK_NULL_HANDLE, compute_pipeline_ci);
#ifdef __ANDROID__
    if (Libraries::GnmDriver::ExecutorReplayActive())
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_TILE] createpipeline result=%d handle=%p",
                            (int)result, (void*)(VkPipeline)*pipeline);
#endif
    ASSERT_MSG(result == vk::Result::eSuccess, "Detiler pipeline creation failed {}",
               vk::to_string(result));
    tiling_pipelines[pl_id] = std::move(pipeline);
    device.destroyShaderModule(module);
    EXEC_TL_MARK("getpipeline_done");
    return *tiling_pipelines[pl_id];
}

TileManager::Result TileManager::DetileImage(vk::Buffer in_buffer, u32 in_offset,
                                             const ImageInfo& info) {
    if (!info.props.is_tiled) {
        return {.buffer = in_buffer, .offset = in_offset};
    }

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    params.num_mips = info.resources.levels;
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const ScratchBuffer scratch = GetScratchBuffer(info.guest_size);
    const vk::Buffer out_buffer = scratch.buffer;

    scheduler.EndRendering(Vulkan::Scheduler::EndRenderingReason::TileCompute);
#ifdef __ANDROID__
    // Keep Qualcomm's compute state isolated without splitting the queue
    // epoch. The primary retains all barriers and ordering, while the
    // secondary starts from clean Vulkan command state.
    const auto primary_cmdbuf = scheduler.CommandBuffer();
    const auto cmdbuf = detile_command_pool->Commit();
    const vk::CommandBufferInheritanceInfo inheritance_info{};
    const vk::CommandBufferBeginInfo begin_info{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        .pInheritanceInfo = &inheritance_info,
    };
    const vk::Result begin_result = cmdbuf.begin(begin_info);
    ASSERT_MSG(begin_result == vk::Result::eSuccess,
               "Failed to begin isolated detile command buffer: {}",
               vk::to_string(begin_result));
#else
    const auto cmdbuf = scheduler.CommandBuffer();
#endif

    const vk::Pipeline detile_pipeline = GetTilingPipeline(info, false);
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, detile_pipeline);
    scheduler.MarkPipelineBind(vk::PipelineBindPoint::eCompute);

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = in_buffer,
        .offset = in_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = out_buffer,
        .offset = 0,
        .range = info.guest_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    EXEC_TL_MARK("before_pushdesc");
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);
    scheduler.MarkDescriptorBind(vk::PipelineBindPoint::eCompute);
    EXEC_TL_MARK("after_pushdesc");

    const auto dim_x = (info.guest_size / (info.num_bits / 8)) / 64;
#ifdef __ANDROID__
    static std::atomic<u32> detile_dispatch_sequence{};
    const u32 dispatch_sequence =
        detile_dispatch_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    if (dispatch_sequence <= 8) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_DETILE_DISPATCH] n=%u scheduler=%p recording=%d "
            "tick=%llu gpuTick=%llu cmdbuf=%p pipeline=%p layout=%p "
            "in=%p out=%p params=%p guestSize=%u bits=%u dimX=%u",
            dispatch_sequence, &scheduler,
            scheduler.HasRecordingCommandBuffer() ? 1 : 0,
            static_cast<unsigned long long>(scheduler.CurrentTick()),
            static_cast<unsigned long long>(scheduler.KnownGpuTick()),
            static_cast<void*>(static_cast<VkCommandBuffer>(cmdbuf)),
            static_cast<void*>(static_cast<VkPipeline>(detile_pipeline)),
            static_cast<void*>(static_cast<VkPipelineLayout>(*pl_layout)),
            static_cast<void*>(static_cast<VkBuffer>(in_buffer)),
            static_cast<void*>(static_cast<VkBuffer>(out_buffer)),
            static_cast<void*>(static_cast<VkBuffer>(params_buffer_info.buffer)),
            info.guest_size, info.num_bits, dim_x);
    }
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        Core::Memory::Instance()->VerifyReplayCanaries("before_detile_dispatch");
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_TILE] before_dispatch dim_x=%u guest_size=%u num_bits=%u cmdbuf=%p",
                            (unsigned)dim_x, info.guest_size, info.num_bits, (void*)(VkCommandBuffer)cmdbuf);
    }
#endif
    cmdbuf.dispatch(dim_x, 1, 1);
    const vk::BufferMemoryBarrier2 detile_to_upload{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = out_buffer,
        .offset = 0,
        .size = info.guest_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &detile_to_upload,
    });
#ifdef __ANDROID__
    const vk::Result end_result = cmdbuf.end();
    ASSERT_MSG(end_result == vk::Result::eSuccess,
               "Failed to end isolated detile command buffer: {}",
               vk::to_string(end_result));
    primary_cmdbuf.executeCommands(cmdbuf);
#endif
    EXEC_TL_MARK("after_dispatch");
    // The caller still records copyBufferToImage from this buffer. Hand
    // ownership back with the result and retire it only after that final use
    // has been recorded, so its timeline tick cannot precede its last Vulkan
    // command.
    return {.buffer = out_buffer, .offset = 0, .scratch = scratch};
}

void TileManager::RetireDetileResult(Result result) {
    if (result.scratch.buffer) {
        RetireScratchBuffer(result.scratch);
    }
}

void TileManager::TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                            vk::Buffer out_buffer, u32 out_offset, u32 copy_size) {
    const auto& info = in_image.info;
    if (!info.props.is_tiled) {
        for (auto& copy : buffer_copies) {
            copy.bufferOffset += out_offset;
        }
        in_image.Download(buffer_copies, out_buffer, out_offset, copy_size);
        return;
    }

    TilingInfo params{};
    params.bank_swizzle = info.bank_swizzle;
    params.num_slices = info.props.is_volume ? info.size.depth : info.resources.layers;
    params.num_mips = static_cast<u32>(buffer_copies.size());
    for (u32 mip = 0; mip < params.num_mips; ++mip) {
        auto& mip_info = params.mips[mip];
        mip_info = info.mips_layout[mip];
        if (info.props.is_block) {
            mip_info.pitch = std::max((mip_info.pitch + 3) / 4, 1U);
            mip_info.height = std::max((mip_info.height + 3) / 4, 1U);
        }
    }

    const vk::DescriptorBufferInfo params_buffer_info{
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&params, sizeof(params), instance.UniformMinAlignment()),
        .range = sizeof(params),
    };

    const ScratchBuffer scratch = GetScratchBuffer(info.guest_size);
    const vk::Buffer temp_buffer = scratch.buffer;

    const auto cmdbuf = scheduler.CommandBuffer();
    in_image.Download(buffer_copies, temp_buffer, 0, copy_size);

    const vk::BufferMemoryBarrier2 download_to_tiler{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = temp_buffer,
        .offset = 0,
        .size = info.guest_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &download_to_tiler,
    });

    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, GetTilingPipeline(info, true));
    scheduler.MarkPipelineBind(vk::PipelineBindPoint::eCompute);

    const vk::DescriptorBufferInfo tiled_buffer_info{
        .buffer = out_buffer,
        .offset = out_offset,
        .range = info.guest_size,
    };

    const vk::DescriptorBufferInfo linear_buffer_info{
        .buffer = temp_buffer,
        .offset = 0,
        .range = info.guest_size,
    };

    const std::array<vk::WriteDescriptorSet, 3> set_writes = {{
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &tiled_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &linear_buffer_info,
        },
        {
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &params_buffer_info,
        },
    }};
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pl_layout, 0, set_writes);
    scheduler.MarkDescriptorBind(vk::PipelineBindPoint::eCompute);

    const auto dim_x = (info.guest_size / (info.num_bits / 8)) / 64;
    cmdbuf.dispatch(dim_x, 1, 1);

    const vk::BufferMemoryBarrier2 tiler_to_consumers{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = out_buffer,
        .offset = out_offset,
        .size = info.guest_size,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &tiler_to_consumers,
    });
    RetireScratchBuffer(scratch);
}

}
