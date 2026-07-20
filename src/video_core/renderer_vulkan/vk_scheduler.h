// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <thread>

#include "common/unique_function.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace tracy {
class VkCtxScope;
}

// EXECUTOR .gnmcap replay: during replay everything must stay on the GPU command-processor thread to
// avoid the renderer-state race (de-tile heisenbug). DeferPriorityOperation normally runs on a separate
// PriorityPendingOps thread; during replay it is routed to the GPU-thread-drained pending_ops queue.
namespace Libraries::GnmDriver {
bool ExecutorReplayActive();
}

namespace Vulkan {

class Instance;

#ifdef __ANDROID__
// A lock-free, in-memory journal of the most recent GPU commands.  It stays silent on the hot path
// and is emitted only after Vulkan reports a terminal error, so production runs retain enough
// context to identify the command buffer/pipeline which killed a mobile driver without enabling the
// extremely expensive per-draw trace.
enum class ExecutorVkJournalDrawType : u32 {
    Direct = 1,
    IndexedDirect = 2,
    Indirect = 3,
    IndexedIndirect = 4,
};

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
// The content probe already gives us an externally bounded, title-independent draw selector.
// Preserve a compact render-graph window around that exact journal record and emit it once the
// same scheduler submits the command buffer containing it.
void ExecutorVkJournalArmContentProbe(const void* stream, u64 journal_sequence,
                                      u64 selected_draw_sequence, u64 fragment_hash) noexcept;
// One-shot, title-independent full guest-frame graph capture. Arm is called at a PM4 PatchedFlip;
// every draw/dispatch until the next PatchedFlip is aggregated by attachment set and shader
// transition. This is intentionally independent of the host swapchain cadence.
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
    u64 invalidation_epoch{};

    /// Commits the dynamic state to the provided command buffer.
    void Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf);

    /// Invalidates all dynamic state to be flushed into the next command buffer.
    void Invalidate() {
        std::memset(&dirty_state, 0xFF, sizeof(dirty_state));
        ++invalidation_epoch;
    }

    /// Changes whenever a new command buffer or an external graphics pass invalidates guest state.
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

    explicit Scheduler(const Instance& instance);
    ~Scheduler();

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush(SubmitInfo& info);

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush();

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish(ExecutorFinishReason reason = ExecutorFinishReason::Unknown);

    /// Waits for the given tick to trigger on the GPU.
    void Wait(u64 tick);

    /// Waits until exact guest completion callbacks through the given timeline tick have run.
    /// Unlike Wait(), this also guarantees that EOP/EOS/RELEASE_MEM bytes and IRQs are visible.
    void WaitRetirement(u64 tick);

    /// Attempts to execute operations whose tick the GPU has caught up with.
    void PopPendingOperations(bool force = false, bool timeline_current = false);

    /// Starts a new rendering scope with provided state.
    void BeginRendering(const RenderState& new_state);

    /// Ends current rendering scope.
    void EndRendering();

    /// Returns the current render state.
    const RenderState& GetRenderState() const {
        return render_state;
    }

    /// Returns the identity of the most recently started dynamic-rendering scope.
    u64 RenderingEpoch() const {
        return rendering_epoch;
    }

    /// Returns the current pipeline dynamic state tracking.
    DynamicState& GetDynamicState() {
        return dynamic_state;
    }

    /// Returns the current command buffer.
    vk::CommandBuffer CommandBuffer() const {
        // Every external caller obtains the handle in order to record a command. This lets the
        // guest-submit tail distinguish a genuinely empty command buffer without instrumenting
        // every Vulkan-Hpp command wrapper.
        command_buffer_used = true;
        command_recording_epoch.fetch_add(1, std::memory_order_relaxed);
        return current_cmdbuf;
    }

    /// Returns the current command buffer and the exact recording-access generation assigned to
    /// this caller. Completion barriers use the token to prove that no draw/copy/dispatch access
    /// occurred before a later logically equivalent barrier.
    vk::CommandBuffer CommandBuffer(u64& access_epoch) const {
        command_buffer_used = true;
        access_epoch =
            command_recording_epoch.fetch_add(1, std::memory_order_relaxed) + 1;
        return current_cmdbuf;
    }

    [[nodiscard]] u64 CommandRecordingEpoch() const noexcept {
        return command_recording_epoch.load(std::memory_order_relaxed);
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore.CurrentTick();
    }

    /// Returns whether the current command buffer or its deferred operations require a timeline
    /// submission. Liverpool uses this to attach independent guest completion points to one host
    /// command buffer without inventing an empty timeline tick.
    [[nodiscard]] bool HasPendingWork() const noexcept {
        return command_buffer_used ||
               pending_ops_count.load(std::memory_order_acquire) != 0;
    }

    [[nodiscard]] bool IsDeviceLost() const noexcept {
        return master_semaphore.IsDeviceLost();
    }

    /// Returns true when a tick has been triggered by the GPU.
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

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore* GetMasterSemaphore() noexcept {
        return &master_semaphore;
    }

    /// Enter the shared terminal/drop state and wake scheduler-side waiters. No further command
    /// buffers are submitted after this transition.
    void MarkTerminal(vk::Result result, const char* where) noexcept;

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Will be run when submitting or calling PopPendingOperations.
    void DeferOperation(Common::UniqueFunction<void>&& func) {
        // EXECUTOR: pending_ops is touched by BOTH the GPU coroutine (Rasterizer::Draw ->
        // PopPendingOperations / cache DeferOperation) AND the render/present thread
        // (Presenter -> draw_scheduler.Flush -> SubmitExecution -> PopPendingOperations). submit_mutex
        // only guards the SubmitExecution side, so the queue was mutated concurrently and unlocked ->
        // the std::deque tears and a moved UniqueFunction (pointer-valued) scribbles a wild host
        // address. Guard every pending_ops access with a dedicated lock.
        std::scoped_lock lk{pending_ops_mutex};
        pending_ops.emplace(std::move(func), CurrentTick());
        // Publish only after the queue entry is complete. PopPendingOperations uses this as an
        // empty fast path; the mutex remains the authority for queue contents and ordering.
        const u64 previous_count =
            pending_ops_count.fetch_add(1, std::memory_order_release);
        if (previous_count == 0) {
            pending_ops_poll_sequence.store(0, std::memory_order_relaxed);
        }
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Runs as soon as possible in another thread.
    void DeferPriorityOperation(Common::UniqueFunction<void>&& func);

    /// Defers an operation until the gpu has reached an explicitly captured timeline tick.
    /// The wait is performed only by the dedicated retirement helper thread.
    void DeferPriorityOperationAt(u64 gpu_tick, Common::UniqueFunction<void>&& func);

    // vkQueueSubmit/vkQueuePresentKHR require external synchronization.
    static std::mutex submit_mutex;

private:
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

    void PriorityPendingOpsThread(std::stop_token stoken);
    void RetirementPendingOpsThread(std::stop_token stoken);

private:
    const Instance& instance;
    MasterSemaphore master_semaphore;
    mutable bool command_buffer_used{};
    mutable std::atomic<u64> command_recording_epoch{};
    CommandPool command_pool;
    DynamicState dynamic_state;
    vk::CommandBuffer current_cmdbuf;
    std::condition_variable_any event_cv;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    std::queue<PendingOp> pending_ops;
    // Published queue size. A zero acquire-load lets the per-draw poll avoid both a Vulkan timeline
    // query and pending_ops_mutex when no deferred resource destruction is waiting.
    std::atomic<u64> pending_ops_count{};
    // A deferred object can remain pending across hundreds of draws while the GPU completes an
    // older tick. Polling the driver on every draw cannot make that tick finish sooner; normal draw
    // polls are sampled while submit/wait boundaries force an immediate drain.
    std::atomic<u32> pending_ops_poll_sequence{};
    std::mutex pending_ops_mutex;
    std::queue<PendingOp> priority_pending_ops;
    std::mutex priority_pending_ops_mutex;
    std::condition_variable_any priority_pending_ops_cv;
    std::jthread priority_pending_ops_thread;
    // Exact frame-retirement waits have a dedicated worker. They must not sit behind a generic
    // readback callback whose newer timeline tick may depend on the guest seeing a VideoOut label.
    // Explicit timeline ordering is required because a completion callback may be registered for
    // an already-submitted tick after newer ticks have entered the queue. FIFO insertion order is
    // not necessarily GPU timeline order.
    std::map<u64, std::queue<PendingOp>> retirement_pending_ops;
    // Includes both queued and currently executing callbacks. WaitRetirement must not infer callback
    // completion from the GPU timeline alone, nor from a queue that the worker has already popped.
    std::map<u64, u64> retirement_outstanding_ops;
    std::mutex retirement_pending_ops_mutex;
    std::condition_variable_any retirement_pending_ops_cv;
    std::jthread retirement_pending_ops_thread;
    std::atomic<u64> retirement_completed_tick{};
#ifdef __ANDROID__
    std::array<u64, static_cast<std::size_t>(ExecutorSubmitKind::Count)>
        executor_submit_kind_counts{};
    std::array<u64, static_cast<std::size_t>(ExecutorFinishReason::Count)>
        executor_finish_reason_counts{};
    u64 executor_submit_kind_total{};
#endif
    RenderState render_state;
    u64 rendering_epoch{};
    bool is_rendering = false;
    tracy::VkCtxScope* profiler_scope{};
};

} // namespace Vulkan
