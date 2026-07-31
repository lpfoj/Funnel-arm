// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/texture_cache/texture_cache.h"

#include <list>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <vector>

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class Scheduler;
class RenderState;
class GraphicsPipeline;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] bool IsDeviceTerminal() const noexcept;
    void MarkDeviceTerminal(vk::Result result, const char* where) noexcept;

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds,
                    bool defer_gnm_dma = false);
    void FlushPendingGnmDmaTransfers();
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size);
    bool InvalidateMemoryFromGuestThread(VAddr addr, u64 size);
    bool ReadMemoryFromGuestThread(VAddr addr, u64 size);
    void TrackPendingCompletionRead(VAddr addr, u64 size);
    void UntrackPendingCompletionRead(VAddr addr, u64 size);
    void TrackMemoryWaitWrite(VAddr addr, u64 size);
    void UntrackMemoryWaitWrite(VAddr addr, u64 size);
    [[nodiscard]] bool ConsumeMemoryWaitWriteFault(VAddr addr);
    bool IsMemoryGpuModified(VAddr addr, u64 size);
    void ProcessDownloadImages();
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    void CpSync();
    void CompletionWaitBarrier(u64 dependency_tick);
    u64 Flush();
    void Finish();
    void OnSubmit();

#ifdef __ANDROID__
    /// Returns the most recently resolved physically-scaled color target when
    /// it is the image being scanned out.  The returned view is transitioned
    /// for sampling so the presenter can feed it directly to FSR instead of
    /// first expanding it with a nearest-neighbour blit.
    bool ExecutorAcquirePhysicalDrsPresentationSource(VideoCore::ImageId image_id,
                                                      vk::ImageView& image_view,
                                                      vk::Extent2D& extent);
#endif

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

private:
    struct GuestInvalidationRequest {
        VAddr addr{};
        u64 size{};
        bool complete{};
        bool result{};
        std::condition_variable completion_cv;
    };

    void DrainGuestInvalidations();

    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;
    void UpdateFragmentShadingRate(const GraphicsPipeline* pipeline,
                                   const RenderState& state) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            texture_cache.GetImage(image_id).binding = {};
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

#ifdef __ANDROID__
    struct ExecutorPhysicalDrsSurface;
    struct ExecutorPhysicalDrsDepthSurface;
    struct ExecutorPhysicalDrsCleanupControl;

    void ExecutorBindGuestGraphicsPipeline(const GraphicsPipeline* pipeline,
                                           vk::CommandBuffer cmdbuf);
    void ExecutorSealGuestGraphicsPipelineState(const GraphicsPipeline* pipeline,
                                                vk::CommandBuffer cmdbuf);
    void ExecutorBindGuestComputePipeline(const ComputePipeline* pipeline,
                                          vk::CommandBuffer cmdbuf);
    void ExecutorPreparePhysicalDrsDraw(const GraphicsPipeline* pipeline);
    bool ExecutorBeginPhysicalDrs(const GraphicsPipeline* pipeline, RenderState& state);
    void ExecutorResolvePhysicalDrs(const char* reason);
    void ExecutorTransitionPhysicalDrsSurface(vk::ImageLayout layout,
                                              vk::AccessFlags2 access,
                                              vk::PipelineStageFlags2 stage);
    void ExecutorTransitionPhysicalDrsDepthSurface(vk::ImageLayout layout,
                                                   vk::AccessFlags2 access,
                                                   vk::PipelineStageFlags2 stage);
    bool ExecutorPhysicalDrsPipelineCompatible(const GraphicsPipeline* pipeline) const;
    void ExecutorTouchPhysicalDrsSurface(ExecutorPhysicalDrsSurface& surface) noexcept;
    void ExecutorTouchPhysicalDrsDepthSurface(
        ExecutorPhysicalDrsDepthSurface& surface) noexcept;
    [[nodiscard]] u64 ExecutorPhysicalDrsImageAllocationBytes(
        const vk::ImageCreateInfo& image_ci) const noexcept;
    [[nodiscard]] bool ExecutorTryCreatePhysicalDrsImage(
        VideoCore::UniqueImage& image, const vk::ImageCreateInfo& image_ci,
        u64& allocation_bytes);
    [[nodiscard]] u64 ExecutorPhysicalDrsResidencyBudgetBytes() const noexcept;
    [[nodiscard]] bool ExecutorRunPhysicalDrsResidencyGc(
        u64 reserve_bytes = 0, bool force_cleanup = false);
    void ExecutorResetPhysicalDrsTransientBindings() noexcept;
    static void ExecutorManagedMobileGpuTransition(
        void* context, bool enabled) noexcept;
    void ExecutorNotifyManagedMobileGpuTransition(bool enabled) noexcept;
    void ExecutorSchedulePhysicalDrsOffCleanup();
    void ExecutorCompletePhysicalDrsOffCleanup();

    struct ExecutorGpuLatePassProbeState {
        struct ImageIdentity {
            VideoCore::ImageId image_id{};
            VAddr desc_guest_address{};
            VAddr image_guest_address{};
            u64 image_uid{};
            u64 backing_identity{};
            u64 image_handle{};
            VideoCore::SubresourceRange view_range{};
        };

        u64 epoch{};
        u64 attachment_signature{};
        u64 selected_sequence{};
        u64 fragment_hash{};
        std::array<ImageIdentity, AmdGpu::NUM_COLOR_BUFFERS> color_images{};
        u32 color_count{};
        ImageIdentity depth_image{};
        u32 depth_render_control_raw{};
        u32 depth_control_raw{};
        u32 depth_clear_enable{};
        u32 depth_write_enable{};
        u32 depth_effective_write{};
        u32 scheduler_depth_test{};
        u32 scheduler_depth_write{};
        u32 scheduler_depth_compare{};
        VAddr depth_read_address{};
        VAddr depth_write_address{};
        VAddr stencil_read_address{};
        VAddr stencil_write_address{};
        VAddr htile_address{};
        VAddr desc_stencil_address{};
        VAddr desc_htile_address{};
        bool armed{};
    };

    void ExecutorGpuLatePassTransition(u64 next_attachment_signature, u64 next_sequence,
                                       const char* reason);
    void ExecutorGpuLatePassArm(u64 attachment_signature, u64 selected_sequence,
                                u64 fragment_hash);
    void ExecutorGpuLatePassFlush(u64 next_attachment_signature, u64 next_sequence,
                                  const char* reason);
#endif

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    std::mutex guest_invalidation_mutex;
    std::vector<GuestInvalidationRequest*> pending_guest_invalidations;
    bool guest_invalidation_drain_queued{};

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;

    using BufferBindingInfo = std::tuple<VideoCore::BufferId, AmdGpu::Buffer, u64>;
    boost::container::static_vector<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings;
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    struct ExecutorGpuProbeBinding {
        VideoCore::ImageId image_id{};
        u32 stage{};
        u64 stage_hash{};
    };
    boost::container::static_vector<ExecutorGpuProbeBinding, Shader::NUM_IMAGES>
        executor_gpu_probe_sampled_images;
    boost::container::static_vector<ExecutorGpuProbeBinding, Shader::NUM_IMAGES>
        executor_gpu_probe_storage_images;
    u64 executor_gpu_probe_draw_sequence{};
    u64 executor_gpu_probe_dispatch_sequence{};
    u64 memory_budget_submit_counter{};
    size_t cached_device_memory_usage{};
    bool memory_budget_sample_valid{};
#ifdef __ANDROID__
    ExecutorGpuLatePassProbeState executor_gpu_late_pass_probe{};
    vk::CommandBuffer executor_guest_graphics_cmdbuf{};
    u64 executor_guest_graphics_pipeline_identity{};
    u64 executor_guest_graphics_recording_tick{};
    u64 executor_guest_graphics_pipeline_epoch{};
    bool executor_guest_graphics_pipeline_valid{};
    vk::CommandBuffer executor_guest_compute_cmdbuf{};
    u64 executor_guest_compute_pipeline_identity{};
    u64 executor_guest_compute_recording_tick{};
    u64 executor_guest_compute_pipeline_epoch{};
    bool executor_guest_compute_pipeline_valid{};
    u64 executor_unpublished_actual_draws{};
    u64 executor_completion_barrier_epoch{};
    u64 executor_completion_barrier_tick{};
    u64 executor_completion_barrier_dependency_tick{};
    u64 executor_completion_work_serial{};
    u64 executor_completion_barrier_work_serial{};
    u64 executor_completion_barrier_calls{};
    u64 executor_completion_barrier_emitted{};
    u64 executor_completion_barrier_elided{};
    bool executor_completion_barrier_valid{};

    struct ExecutorPhysicalDrsSurface {
        VideoCore::UniqueImage image{};
        vk::UniqueImageView view{};
        vk::Format format{vk::Format::eUndefined};
        vk::Extent2D extent{};
        vk::ImageLayout layout{vk::ImageLayout::eUndefined};
        vk::AccessFlags2 access{vk::AccessFlagBits2::eNone};
        vk::PipelineStageFlags2 stage{vk::PipelineStageFlagBits2::eTopOfPipe};
        VideoCore::ImageId owner_image_id{};
        u64 owner_image_uid{};
        u64 allocation_bytes{};
        u64 last_used_generation{};
        u64 last_use_tick{};
    };
    struct ExecutorPhysicalDrsDepthSurface {
        VideoCore::UniqueImage image{};
        vk::UniqueImageView view{};
        vk::Format format{vk::Format::eUndefined};
        vk::Extent2D extent{};
        vk::ImageAspectFlags aspects{};
        vk::ImageLayout layout{vk::ImageLayout::eUndefined};
        vk::AccessFlags2 access{vk::AccessFlagBits2::eNone};
        vk::PipelineStageFlags2 stage{vk::PipelineStageFlagBits2::eTopOfPipe};
        VideoCore::ImageId owner_image_id{};
        u64 owner_image_uid{};
        u64 allocation_bytes{};
        u64 last_used_generation{};
        u64 last_use_tick{};
    };
    std::list<ExecutorPhysicalDrsSurface> executor_physical_drs_surfaces;
    std::list<ExecutorPhysicalDrsDepthSurface> executor_physical_drs_depth_surfaces;
    std::array<ExecutorPhysicalDrsSurface*, AmdGpu::NUM_COLOR_BUFFERS>
        executor_physical_drs_color_surfaces{};
    std::array<VideoCore::ImageId, AmdGpu::NUM_COLOR_BUFFERS>
        executor_physical_drs_color_image_ids{};
    std::array<u64, AmdGpu::NUM_COLOR_BUFFERS>
        executor_physical_drs_color_image_uids{};
    std::array<VAddr, AmdGpu::NUM_COLOR_BUFFERS>
        executor_physical_drs_color_guest_addresses{};
    u32 executor_physical_drs_color_mask{};
    ExecutorPhysicalDrsSurface* executor_physical_drs_surface{};
    ExecutorPhysicalDrsDepthSurface* executor_physical_drs_depth_surface{};
    VideoCore::ImageId executor_physical_drs_image_id{};
    u64 executor_physical_drs_image_uid{};
    VideoCore::ImageId executor_physical_drs_depth_image_id{};
    u64 executor_physical_drs_depth_image_uid{};
    VAddr executor_physical_drs_guest_address{};
    VAddr executor_physical_drs_depth_guest_address{};
    u32 executor_physical_drs_scale_percent{100};
    bool executor_physical_drs_active{};
    bool executor_physical_drs_draw_active{};
    ExecutorPhysicalDrsSurface* executor_physical_drs_present_surface{};
    VideoCore::ImageId executor_physical_drs_present_image_id{};
    u64 executor_physical_drs_present_image_uid{};
    u64 executor_physical_drs_generation{};
    u64 executor_physical_drs_resident_bytes{};
    u64 executor_physical_drs_peak_bytes{};
    u64 executor_physical_drs_budget_bytes{};
    u64 executor_physical_drs_gc_passes{};
    u64 executor_physical_drs_gc_evictions{};
    u64 executor_physical_drs_gc_blocked{};
    bool executor_physical_drs_building{};
    bool executor_physical_drs_off_cleanup_scheduled{};
    bool executor_physical_drs_shutting_down{};
    // Resolve/admission/scheduling are one state transaction and deliberately
    // nest their locked helpers.
    std::recursive_mutex executor_physical_drs_mutex;
    std::shared_ptr<ExecutorPhysicalDrsCleanupControl>
        executor_physical_drs_cleanup_control;
#endif
    bool fault_process_pending{};
    bool attachment_feedback_loop{};
};

}
