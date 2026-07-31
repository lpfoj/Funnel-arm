// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <span>
#include <thread>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/unique_function.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace tracy {
class VkCtxScope;
}

namespace Libraries::GnmDriver {
bool ExecutorReplayActive();
}

namespace Vulkan {

class Instance;

#ifdef __ANDROID__
enum class ExecutorVkJournalDrawType : u32 {
    Direct = 1,
    IndexedDirect = 2,
    Indirect = 3,
    IndexedIndirect = 4,
};

bool ExecutorVkJournalRequested() noexcept;

u64 ExecutorVkJournalGraphicsDraw(const void* stream, u64 tick, u64 pipeline_key_hash,
                                  u64 pipeline_handle, u64 vs_hash, u64 ps_hash,
                                  ExecutorVkJournalDrawType draw_type, u32 primitive_type,
                                  u32 element_count,
                                  u32 instance_count, u32 max_draw_count, u32 stride,
                                  VAddr argument_address, VAddr count_address,
                                  VAddr color_address, VAddr depth_address,
                                  u64 attachment_signature, u32 color_attachment_count,
                                  u32 color_width, u32 color_height, u32 color_format,
                                  u32 depth_format, u32 descriptor_writes,
                                  u32 buffer_descriptors, u32 image_descriptors,
                                  bool force_record = false) noexcept;
void ExecutorVkJournalComputeDispatch(const void* stream, u64 tick, u64 pipeline_key_hash,
                                      u64 pipeline_handle, u64 cs_hash, bool indirect,
                                      u32 dim_x, u32 dim_y, u32 dim_z,
                                      VAddr argument_address, u32 descriptor_writes,
                                      u32 buffer_descriptors, u32 image_descriptors,
                                      bool force_record = false) noexcept;
void ExecutorVkJournalSubmit(const void* stream, u64 tick, bool completed,
                             vk::Result result) noexcept;
void ExecutorVkJournalArmContentProbe(const void* stream, u64 journal_sequence,
                                      u64 selected_draw_sequence, u64 fragment_hash) noexcept;
void ExecutorVkJournalArmFrameCapture(u64 epoch) noexcept;
void ExecutorVkJournalFinishFrameCapture(u64 epoch) noexcept;
void ExecutorVkJournalDump(const char* reason, u64 failing_tick) noexcept;
#endif

struct RenderAttachment {
    vk::ImageView image_view;
    vk::ImageLayout image_layout;
    std::array<u32, 4> clear_value;
    union {
        u32 is_clear;
        struct {
            bool has_depth;
            bool depth_clear;
            bool has_stencil;
            bool stencil_clear;
        };
    };
};
static_assert(std::has_unique_object_representations_v<RenderAttachment>);

struct RenderState {
    std::array<RenderAttachment, 8> color_attachments;
    RenderAttachment depth_stencil_attachment;
    u16 width;
    u16 height;
    u16 num_layers;
    u16 num_color_attachments;

    bool operator==(const RenderState& other) const noexcept {
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<RenderState>);

struct SubmitInfo {
    std::array<vk::Semaphore, 3> wait_semas;
    std::array<u64, 3> wait_ticks;
    std::array<vk::PipelineStageFlags, 3> wait_stage_masks;
    std::array<vk::Semaphore, 3> signal_semas;
    std::array<u64, 3> signal_ticks;
    vk::Fence fence;
    u32 num_wait_semas;
    u32 num_signal_semas;

    void AddWait(vk::Semaphore semaphore, u64 tick = 1,
                 vk::PipelineStageFlags stage_mask = vk::PipelineStageFlagBits::eAllCommands) {
        ASSERT(num_wait_semas < wait_semas.size());
        wait_semas[num_wait_semas] = semaphore;
        wait_ticks[num_wait_semas] = tick;
        wait_stage_masks[num_wait_semas++] = stage_mask;
    }

    void AddSignal(vk::Semaphore semaphore, u64 tick = 1) {
        signal_semas[num_signal_semas] = semaphore;
        signal_ticks[num_signal_semas++] = tick;
    }

    void AddSignal(vk::Fence fence) {
        this->fence = fence;
    }
};

using Viewports = boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS>;
using Scissors = boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS>;
using ColorWriteMasks = std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS>;
struct StencilOps {
    vk::StencilOp fail_op{};
    vk::StencilOp pass_op{};
    vk::StencilOp depth_fail_op{};
    vk::CompareOp compare_op{};

    bool operator==(const StencilOps& other) const {
        return fail_op == other.fail_op && pass_op == other.pass_op &&
               depth_fail_op == other.depth_fail_op && compare_op == other.compare_op;
    }
};
struct DynamicState {
    struct {
        bool viewports : 1;
        bool scissors : 1;

        bool depth_test_enabled : 1;
        bool depth_write_enabled : 1;
        bool depth_compare_op : 1;

        bool depth_bounds_test_enabled : 1;
        bool depth_bounds : 1;

        bool depth_bias_enabled : 1;
        bool depth_bias : 1;

        bool stencil_test_enabled : 1;
        bool stencil_front_ops : 1;
        bool stencil_front_reference : 1;
        bool stencil_front_write_mask : 1;
        bool stencil_front_compare_mask : 1;
        bool stencil_back_ops : 1;
        bool stencil_back_reference : 1;
        bool stencil_back_write_mask : 1;
        bool stencil_back_compare_mask : 1;

        bool primitive_restart_enable : 1;
        bool rasterizer_discard_enable : 1;
        bool cull_mode : 1;
        bool front_face : 1;

        bool blend_constants : 1;
        bool color_write_masks : 1;
        bool line_width : 1;
        bool feedback_loop_enabled : 1;
        bool fragment_shading_rate : 1;
    } dirty_state{};

    Viewports viewports{};
    Scissors scissors{};

    bool depth_test_enabled{};
    bool depth_write_enabled{};
    vk::CompareOp depth_compare_op{};

    bool depth_bounds_test_enabled{};
    float depth_bounds_min{};
    float depth_bounds_max{};

    bool depth_bias_enabled{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};

    bool stencil_test_enabled{};
    StencilOps stencil_front_ops{};
    u32 stencil_front_reference{};
    u32 stencil_front_write_mask{};
    u32 stencil_front_compare_mask{};
    StencilOps stencil_back_ops{};
    u32 stencil_back_reference{};
    u32 stencil_back_write_mask{};
    u32 stencil_back_compare_mask{};

    bool primitive_restart_enable{};
    bool rasterizer_discard_enable{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};

    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    float line_width{};
    bool feedback_loop_enabled{};
    // One renderer is active per runtime. Keep managed VRS state outside the
    // DynamicState object so its ABI and all following Scheduler offsets remain stable.
    inline static vk::Extent2D fragment_shading_rate{1, 1};
    u64 invalidation_epoch{};

    void Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf);

    void Invalidate() {
        std::memset(&dirty_state, 0xFF, sizeof(dirty_state));
        ++invalidation_epoch;
    }

    [[nodiscard]] u64 InvalidationEpoch() const noexcept {
        return invalidation_epoch;
    }

    void SetViewports(const Viewports& viewports_) {
        if (!std::ranges::equal(viewports, viewports_)) {
            viewports = viewports_;
            dirty_state.viewports = true;
        }
    }

    void SetScissors(const Scissors& scissors_) {
        if (!std::ranges::equal(scissors, scissors_)) {
            scissors = scissors_;
            dirty_state.scissors = true;
        }
    }

    void SetDepthTestEnabled(const bool enabled) {
        if (depth_test_enabled != enabled) {
            depth_test_enabled = enabled;
            dirty_state.depth_test_enabled = true;
        }
    }

    void SetDepthWriteEnabled(const bool enabled) {
        if (depth_write_enabled != enabled) {
            depth_write_enabled = enabled;
            dirty_state.depth_write_enabled = true;
        }
    }

    void SetDepthCompareOp(const vk::CompareOp compare_op) {
        if (depth_compare_op != compare_op) {
            depth_compare_op = compare_op;
            dirty_state.depth_compare_op = true;
        }
    }

    void SetDepthBoundsTestEnabled(const bool enabled) {
        if (depth_bounds_test_enabled != enabled) {
            depth_bounds_test_enabled = enabled;
            dirty_state.depth_bounds_test_enabled = true;
        }
    }

    void SetDepthBounds(const float min, const float max) {
        if (depth_bounds_min != min || depth_bounds_max != max) {
            depth_bounds_min = min;
            depth_bounds_max = max;
            dirty_state.depth_bounds = true;
        }
    }

    void SetDepthBiasEnabled(const bool enabled) {
        if (depth_bias_enabled != enabled) {
            depth_bias_enabled = enabled;
            dirty_state.depth_bias_enabled = true;
        }
    }

    void SetDepthBias(const float constant, const float clamp, const float slope) {
        if (depth_bias_constant != constant || depth_bias_clamp != clamp ||
            depth_bias_slope != slope) {
            depth_bias_constant = constant;
            depth_bias_clamp = clamp;
            depth_bias_slope = slope;
            dirty_state.depth_bias = true;
        }
    }

    void SetStencilTestEnabled(const bool enabled) {
        if (stencil_test_enabled != enabled) {
            stencil_test_enabled = enabled;
            dirty_state.stencil_test_enabled = true;
        }
    }

    void SetStencilOps(const StencilOps& front_ops, const StencilOps& back_ops) {
        if (stencil_front_ops != front_ops) {
            stencil_front_ops = front_ops;
            dirty_state.stencil_front_ops = true;
        }
        if (stencil_back_ops != back_ops) {
            stencil_back_ops = back_ops;
            dirty_state.stencil_back_ops = true;
        }
    }

    void SetStencilReferences(const u32 front_reference, const u32 back_reference) {
        if (stencil_front_reference != front_reference) {
            stencil_front_reference = front_reference;
            dirty_state.stencil_front_reference = true;
        }
        if (stencil_back_reference != back_reference) {
            stencil_back_reference = back_reference;
            dirty_state.stencil_back_reference = true;
        }
    }

    void SetStencilWriteMasks(const u32 front_write_mask, const u32 back_write_mask) {
        if (stencil_front_write_mask != front_write_mask) {
            stencil_front_write_mask = front_write_mask;
            dirty_state.stencil_front_write_mask = true;
        }
        if (stencil_back_write_mask != back_write_mask) {
            stencil_back_write_mask = back_write_mask;
            dirty_state.stencil_back_write_mask = true;
        }
    }

    void SetStencilCompareMasks(const u32 front_compare_mask, const u32 back_compare_mask) {
        if (stencil_front_compare_mask != front_compare_mask) {
            stencil_front_compare_mask = front_compare_mask;
            dirty_state.stencil_front_compare_mask = true;
        }
        if (stencil_back_compare_mask != back_compare_mask) {
            stencil_back_compare_mask = back_compare_mask;
            dirty_state.stencil_back_compare_mask = true;
        }
    }

    void SetPrimitiveRestartEnabled(const bool enabled) {
        if (primitive_restart_enable != enabled) {
            primitive_restart_enable = enabled;
            dirty_state.primitive_restart_enable = true;
        }
    }

    void SetCullMode(const vk::CullModeFlags cull_mode_) {
        if (cull_mode != cull_mode_) {
            cull_mode = cull_mode_;
            dirty_state.cull_mode = true;
        }
    }

    void SetFrontFace(const vk::FrontFace front_face_) {
        if (front_face != front_face_) {
            front_face = front_face_;
            dirty_state.front_face = true;
        }
    }

    void SetBlendConstants(const std::array<float, 4> blend_constants_) {
        if (blend_constants != blend_constants_) {
            blend_constants = blend_constants_;
            dirty_state.blend_constants = true;
        }
    }

    void SetRasterizerDiscardEnabled(const bool enabled) {
        if (rasterizer_discard_enable != enabled) {
            rasterizer_discard_enable = enabled;
            dirty_state.rasterizer_discard_enable = true;
        }
    }

    void SetColorWriteMasks(const ColorWriteMasks& color_write_masks_) {
        if (!std::ranges::equal(color_write_masks, color_write_masks_)) {
            color_write_masks = color_write_masks_;
            dirty_state.color_write_masks = true;
        }
    }

    void SetLineWidth(const float width) {
        if (line_width != width) {
            line_width = width;
            dirty_state.line_width = true;
        }
    }

    void SetAttachmentFeedbackLoopEnabled(const bool enabled) {
        if (feedback_loop_enabled != enabled) {
            feedback_loop_enabled = enabled;
            dirty_state.feedback_loop_enabled = true;
        }
    }

    void SetFragmentShadingRate(const vk::Extent2D rate) {
        if (fragment_shading_rate.width != rate.width ||
            fragment_shading_rate.height != rate.height) {
            fragment_shading_rate = rate;
            dirty_state.fragment_shading_rate = true;
        }
    }
};

class Scheduler {
public:
    enum class ExecutorFinishReason : u32 {
        Unknown,
        BufferReadbackWave,
        BufferReadbackDirect,
        BufferProbe,
        TextureReadback,
        TextureProbe,
        Rasterizer,
        PresenterShutdown,
        Count,
    };

    enum class EndRenderingReason : u32 {
        Unknown,
        BeginStateChange,
        PipelineResourceBarrier,
        RasterizerBarrier,
        RasterizerDraw,
        RasterizerClear,
        RasterizerCopy,
        RasterizerDispatch,
        ImageUpload,
        ImageDownload,
        ImageTransit,
        TextureCacheRefresh,
        BufferFill,
        BufferReadback,
        BufferCopy,
        BufferProbe,
        BufferUpload,
        FaultCompute,
        TileCompute,
        BlitHelper,
        ShaderHleCopy,
        Presenter,
        Flush,
        Submit,
        Count,
    };

    enum class BufferTransferOrigin : u32 {
        Hle,
        Upload,
        GnmDma,
        Count,
    };

    explicit Scheduler(const Instance& instance);
    ~Scheduler();

    void Flush(SubmitInfo& info);

    void Flush();

    void Finish(ExecutorFinishReason reason = ExecutorFinishReason::Unknown);

    void Wait(u64 tick);

    void WaitRetirement(u64 tick);

    void PopPendingOperations(bool force = false, bool timeline_current = false);

    void BeginRendering(const RenderState& new_state);

    void EndRendering(EndRenderingReason reason);

    /// Preserve an active dynamic-rendering scope for a guest completion
    /// dependency. This is legal only with dynamicRenderingLocalRead and uses
    /// a graphics-shader-only memory dependency; attachment transitions and
    /// transfer/compute work continue to use the conservative end-render path.
    [[nodiscard]] bool TryCompletionBarrierInsideRendering();

    void QueueBufferCopies(vk::Buffer source, vk::Buffer destination,
                           std::span<const vk::BufferCopy> regions,
                           VAddr source_guest_address, u64 source_size,
                           VAddr destination_guest_address, u64 destination_size,
                           BufferTransferOrigin origin);

    void QueueBufferUploadCopies(vk::Buffer source, vk::Buffer destination,
                                 std::span<const vk::BufferCopy> regions,
                                 VAddr destination_guest_address, u64 destination_size);

    void FlushPendingBufferTransfers();

    void FlushPendingBufferTransfersForRange(VAddr guest_address, u64 size);

    void FlushPendingGnmDmaTransfers();

    [[nodiscard]] bool HasPendingBufferTransfers() const noexcept {
        return !pending_buffer_transfers.empty();
    }

    [[nodiscard]] u64 PendingBufferTransferRequestEpoch() const noexcept {
        return pending_buffer_transfer_request_epoch;
    }

    const RenderState& GetRenderState() const {
        return render_state;
    }

    u64 RenderingEpoch() const {
        return rendering_epoch;
    }

    DynamicState& GetDynamicState() {
        return dynamic_state;
    }

    vk::CommandBuffer CommandBuffer() const {
        command_buffer_used = true;
        command_recording_epoch.fetch_add(1, std::memory_order_relaxed);
        return current_cmdbuf;
    }

    vk::CommandBuffer CommandBuffer(u64& access_epoch) const {
        command_buffer_used = true;
        access_epoch =
            command_recording_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
        return current_cmdbuf;
    }

    [[nodiscard]] u64 CommandRecordingEpoch() const noexcept {
        return command_recording_epoch.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 PipelineBindEpoch(vk::PipelineBindPoint bind_point) const noexcept {
        return pipeline_bind_epochs[BindPointIndex(bind_point)].load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 DescriptorBindEpoch(vk::PipelineBindPoint bind_point) const noexcept {
        return descriptor_bind_epochs[BindPointIndex(bind_point)].load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 PushConstantEpoch(vk::PipelineBindPoint bind_point) const noexcept {
        return push_constant_epochs[BindPointIndex(bind_point)].load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 VertexInputEpoch() const noexcept {
        return vertex_input_epoch.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 VertexBufferBindEpoch() const noexcept {
        return vertex_buffer_bind_epoch.load(std::memory_order_relaxed);
    }

    [[nodiscard]] u64 IndexBufferBindEpoch() const noexcept {
        return index_buffer_bind_epoch.load(std::memory_order_relaxed);
    }

    u64 MarkPipelineBind(vk::PipelineBindPoint bind_point) noexcept {
        return pipeline_bind_epochs[BindPointIndex(bind_point)].fetch_add(
                   1, std::memory_order_relaxed) +
               1;
    }

    u64 MarkDescriptorBind(vk::PipelineBindPoint bind_point) noexcept {
        return descriptor_bind_epochs[BindPointIndex(bind_point)].fetch_add(
                   1, std::memory_order_relaxed) +
               1;
    }

    u64 MarkPushConstants(vk::PipelineBindPoint bind_point) noexcept {
        return push_constant_epochs[BindPointIndex(bind_point)].fetch_add(
                   1, std::memory_order_relaxed) +
               1;
    }

    u64 MarkVertexInput() noexcept {
        return vertex_input_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    u64 MarkVertexBufferBind() noexcept {
        return vertex_buffer_bind_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    u64 MarkIndexBufferBind() noexcept {
        return index_buffer_bind_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore.CurrentTick();
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return master_semaphore.KnownGpuTick();
    }

    [[nodiscard]] bool HasPendingWork() const noexcept {
        return command_buffer_used ||
               !pending_buffer_transfers.empty() ||
#ifdef __ANDROID__
               pending_submit_count.load(std::memory_order_acquire) != 0 ||
#endif
               pending_ops_count.load(std::memory_order_acquire) != 0;
    }

    [[nodiscard]] bool IsDeviceLost() const noexcept {
        return master_semaphore.IsDeviceLost();
    }

    [[nodiscard]] bool HasRecordingCommandBuffer() const noexcept {
        return command_buffer_recording;
    }

    [[nodiscard]] bool IsRendering() const noexcept {
        return is_rendering;
    }

    [[nodiscard]] bool IsFree(u64 tick) noexcept {
        if (master_semaphore.IsDeviceLost()) {
            return false;
        }
        if (master_semaphore.IsFree(tick)) {
            return true;
        }
        master_semaphore.Refresh();
        return master_semaphore.IsFree(tick);
    }

    [[nodiscard]] MasterSemaphore* GetMasterSemaphore() noexcept {
        return &master_semaphore;
    }

    void MarkTerminal(vk::Result result, const char* where) noexcept;

    void DeferOperation(Common::UniqueFunction<void>&& func) {
        std::scoped_lock lk{pending_ops_mutex};
        pending_ops.emplace(std::move(func), CurrentTick());
        const u64 previous_count =
            pending_ops_count.fetch_add(1, std::memory_order_release);
        if (previous_count == 0) {
            pending_ops_poll_sequence.store(0, std::memory_order_relaxed);
        }
    }

    void DeferPriorityOperation(Common::UniqueFunction<void>&& func);

    void DeferPriorityOperationAt(u64 gpu_tick, Common::UniqueFunction<void>&& func);

    static std::mutex submit_mutex;

private:
    static constexpr std::size_t BindPointIndex(vk::PipelineBindPoint bind_point) noexcept {
        return bind_point == vk::PipelineBindPoint::eCompute ? 1u : 0u;
    }

    enum class ExecutorSubmitKind : u32 {
        FlushInfo,
        FlushPlain,
        Finish,
        WaitCurrent,
        Count,
    };

    void AllocateWorkerCommandBuffers();

    void SubmitExecution(SubmitInfo& info, ExecutorSubmitKind kind,
                         ExecutorFinishReason finish_reason = ExecutorFinishReason::Unknown);
#ifdef __ANDROID__
    void PendingSubmitThread(std::stop_token stoken);
    void PublishPendingSubmitBatch();
    bool SubmitPendingBatchLocked(const vk::SubmitInfo* trailing_submit = nullptr,
                                  vk::Fence trailing_fence = {},
                                  u64 trailing_tick = 0,
                                  bool forced_boundary = false);
    bool QueueSubmitLocked(vk::ArrayProxy<const vk::SubmitInfo> submits,
                           vk::Fence fence, std::span<const u64> logical_ticks,
                           u64 command_buffer_count, bool forced_boundary);
#endif

    void PriorityPendingOpsThread(std::stop_token stoken);
    void RetirementPendingOpsThread(std::stop_token stoken);

private:
    struct PendingBufferTransfer {
        vk::Buffer source;
        vk::Buffer destination;
        boost::container::small_vector<vk::BufferCopy, 8> regions;
        VAddr source_begin{};
        VAddr source_end{};
        VAddr destination_begin{};
        VAddr destination_end{};
        vk::DeviceSize source_offset_begin{};
        vk::DeviceSize source_offset_end{};
        vk::DeviceSize destination_offset_begin{};
        vk::DeviceSize destination_offset_end{};
        bool dependency_barrier_before{};
    };

    const Instance& instance;
    MasterSemaphore master_semaphore;
    mutable bool command_buffer_used{};
    bool command_buffer_recording{};
    mutable std::atomic<u64> command_recording_epoch{};
    // Command-state epochs belong to this command stream. Presenter owns
    // separate draw/present/flip schedulers, so sharing these epochs across
    // instances can incorrectly reuse state recorded into another buffer.
    std::array<std::atomic<u64>, 2> pipeline_bind_epochs{};
    std::array<std::atomic<u64>, 2> descriptor_bind_epochs{};
    std::array<std::atomic<u64>, 2> push_constant_epochs{};
    std::atomic<u64> vertex_input_epoch{};
    std::atomic<u64> vertex_buffer_bind_epoch{};
    std::atomic<u64> index_buffer_bind_epoch{};
    CommandPool command_pool;
    DynamicState dynamic_state;
    vk::CommandBuffer current_cmdbuf;
#ifdef __ANDROID__
    // Only internal, synchronization-free FlushPlain packets enter this batch.
    // FlushInfo/fence/wait/Finish packets remain publication boundaries and are
    // appended as a distinct VkSubmitInfo in the same vkQueueSubmit call.
    // A batch must never cross Scheduler instances: each instance owns a
    // different timeline semaphore and command pool.
    boost::container::small_vector<vk::CommandBuffer, 8>
        pending_submit_command_buffers;
    boost::container::small_vector<u64, 8> pending_submit_ticks;
    std::chrono::steady_clock::time_point pending_submit_deadline{};
    std::atomic<u32> pending_submit_count{};
    std::atomic<u64> pending_submit_last_tick{};
    std::condition_variable_any pending_submit_cv;
    std::jthread pending_submit_thread;
#endif
    std::condition_variable_any event_cv;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    std::queue<PendingOp> pending_ops;
    std::atomic<u64> pending_ops_count{};
    std::atomic<u32> pending_ops_poll_sequence{};
    std::mutex pending_ops_mutex;
    std::queue<PendingOp> priority_pending_ops;
    std::mutex priority_pending_ops_mutex;
    std::condition_variable_any priority_pending_ops_cv;
    std::jthread priority_pending_ops_thread;
    std::map<u64, std::queue<PendingOp>> retirement_pending_ops;
    std::map<u64, u64> retirement_outstanding_ops;
    std::atomic<u64> retirement_outstanding_count{};
    std::mutex retirement_pending_ops_mutex;
    std::condition_variable_any retirement_pending_ops_cv;
    std::jthread retirement_pending_ops_thread;
    std::atomic<u64> retirement_completed_tick{};
    std::vector<PendingBufferTransfer> pending_buffer_transfers;
    u64 pending_buffer_transfer_requests{};
    u64 pending_buffer_transfer_request_epoch{};
    u64 pending_buffer_transfer_regions{};
    u64 pending_gnm_dma_transfers{};
    u32 pending_buffer_transfer_origins{};
    bool flushing_buffer_transfers{};
#ifdef __ANDROID__
    std::array<u64, static_cast<std::size_t>(ExecutorSubmitKind::Count)>
        executor_submit_kind_counts{};
    std::array<u64, static_cast<std::size_t>(ExecutorFinishReason::Count)>
        executor_finish_reason_counts{};
    std::array<u64, static_cast<std::size_t>(ExecutorFinishReason::Count)>
        executor_finish_wait_counts{};
    std::array<u64, static_cast<std::size_t>(ExecutorFinishReason::Count)>
        executor_finish_wait_ns{};
    std::array<u64, static_cast<std::size_t>(ExecutorFinishReason::Count)>
        executor_finish_wait_max_ns{};
    u64 executor_finish_wait_total{};
    u64 executor_submit_kind_total{};
    inline static u64 executor_buffer_transfers_queued{};
    inline static u64 executor_buffer_transfer_regions_queued{};
    inline static u64 executor_buffer_transfer_flushes{};
    inline static u64 executor_buffer_transfer_copy_calls{};
    inline static u64 executor_buffer_transfer_copy_regions_emitted{};
    inline static u64 executor_buffer_transfer_barriers_requested{};
    inline static u64 executor_buffer_transfer_barriers_emitted{};
    inline static u64 executor_buffer_transfer_max_queued{};
    inline static u64 executor_buffer_transfer_dependency_barriers{};
    inline static u64 executor_hle_buffer_transfers_queued{};
    inline static u64 executor_buffer_uploads_queued{};
    inline static u64 executor_gnm_dma_transfers_queued{};
    inline static u64 executor_gnm_dma_regions_queued{};
    inline static u64 executor_submit_batches{};
    inline static u64 executor_submit_logical_packets{};
    inline static u64 executor_submit_command_buffers{};
    inline static u64 executor_submit_calls_saved{};
    inline static u64 executor_submit_forced_boundaries{};
    inline static u64 executor_submit_timer_flushes{};
    inline static u64 executor_submit_max_batch{};
#endif
    RenderState render_state;
    u64 rendering_epoch{};
    bool is_rendering = false;
    tracy::VkCtxScope* profiler_scope{};
};

}
