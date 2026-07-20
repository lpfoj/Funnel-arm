// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>
#include <tuple>
#include <vector>
#include <queue>

#include "common/assert.h"
#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

namespace Vulkan {
class Rasterizer;
}

namespace Libraries::VideoOut {
struct VideoOutPort;
}

namespace AmdGpu {

struct PM4CmdWaitRegMem;

#ifdef __ANDROID__
// Marker-gated, bounded flight recorder for the guest EOP contract. These hooks intentionally live
// at the HLE/Liverpool boundary so one dump can prove exactly where a submitted IRQ-only EOP was
// lost without enabling the per-packet PM4/logcat firehose.
void ExecutorEopTraceHleSubmit(std::span<const u32> dcb, std::span<const u32> ccb,
                               const char* label, u32 workload, u32 cbpair);
void ExecutorEopTraceEqTrigger(u64 id, s64 eq);
void ExecutorEopTraceEqWait(bool entering, s64 eq, s32 result, u64 id, s16 filter);
void ExecutorEopTraceSemSync(u32 operation, uintptr_t slot, uintptr_t native, s32 before,
                             s32 after, s32 result, u32 thread_kind);
void ExecutorEopTraceSubmitDonePulse(u64 pulse);
#endif

struct Liverpool {
    static constexpr u32 GfxQueueId = 0u;
    static constexpr u32 NumGfxRings = 1u;     // actually 2, but HP is reserved by system software
    static constexpr u32 NumComputePipes = 7u; // actually 8, but #7 is reserved by system software
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u); // need to fit into u64 bitmap for ffs

    enum ContextRegs : u32 {
        DbZInfo = 0xA010,
        CbColor0Base = 0xA318,
        CbColor1Base = 0xA327,
        CbColor2Base = 0xA336,
        CbColor3Base = 0xA345,
        CbColor4Base = 0xA354,
        CbColor5Base = 0xA363,
        CbColor6Base = 0xA372,
        CbColor7Base = 0xA381,
        CbColor0Cmask = 0xA31F,
        CbColor1Cmask = 0xA32E,
        CbColor2Cmask = 0xA33D,
        CbColor3Cmask = 0xA34C,
        CbColor4Cmask = 0xA35B,
        CbColor5Cmask = 0xA36A,
        CbColor6Cmask = 0xA379,
        CbColor7Cmask = 0xA388,
    };

    Regs regs{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};

public:
    explicit Liverpool();
    ~Liverpool();

    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);

    void SubmitDone() noexcept {
        // Keep the predicate transition and notification under the same mutex used by Process().
        // Otherwise Process() can observe a false predicate, then go to sleep after this notify and
        // leave the submit-done tail (OnSubmit/Flush/GpuIdle) stranded until unrelated future work.
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        // Lock-free (Sonic JIT): a host wild write during draw processing corrupts submit_mutex,
        // so a unique_lock here FORTIFY-aborts the render thread before the present. num_submits is
        // atomic; poll it with a bound instead of locking the (possibly corrupted) mutex/cv.
        for (int i = 0; i < 250 && num_submits.load(std::memory_order_acquire) != 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    // PM4 frequently replays large SET_*_REG packets whose values are identical to the current
    // state. PipelineCache uses this generation to bypass rebuilding and rehashing the complete
    // graphics key (including shader/SRT discovery) between genuinely unchanged draws.
    [[nodiscard]] u64 GraphicsStateGeneration() const noexcept {
        return graphics_state_generation;
    }

    void NotifyGraphicsStateChanged() noexcept {
        ++graphics_state_generation;
    }

    /// Shared renderer terminal state used to stop accepting GPU work after an Android driver wait
    /// has exceeded its bounded retry budget.
    [[nodiscard]] bool IsRendererTerminal() const noexcept;
    void MarkRendererTerminal(const char* where) noexcept;

    // EXECUTOR .gnmcap replay: wait for BOTH the PM4 coroutine (num_submits==0) AND the Vulkan queue
    // (rasterizer->Finish -> scheduler.Finish) before the replay frees its overlay-backed memory.
    // Defined in liverpool.cpp where Vulkan::Rasterizer is a complete type.
    void WaitRasterizerIdleForExecutor();

    // Android CPU-read fault bridge. Completion writes may still live in the scheduler's current
    // (unsubmitted) Vulkan batch. A guest read of such a fence must force that batch out on the
    // Liverpool thread before the normal buffer-cache readback is allowed to continue.
    [[nodiscard]] bool ResolvePendingCompletionReadForExecutor(VAddr address, u64 size);

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    // Layer3Pm4Parse (parse-only): drive a DCB through the PRODUCTION PM4 parser (ProcessGraphics)
    // synchronously with NO rasterizer/Vulkan/presenter -- regs mutate, draw packets are detected but
    // never executed. Returns observed packet stats. See native_runtime_api ExecutorRunGnmPm4ParseSmoke.
    struct ExecutorPm4ParseStats {
        u32 packets;
        bool set_context_reg;
        bool set_sh_reg;
        bool draw_index_auto;
        bool regs_mutated;
        u32 num_indices;
        bool rasterizer_bound;
    };
    ExecutorPm4ParseStats ExecutorParsePm4NoRaster(std::span<const u32> dcb);

    // Read-only access to the parsed register file (for Layer3PipelinePrep: inspect ps_program/
    // vs_program after a parse-only DCB sets them via SetShReg). No GPU state is touched.
    const Regs& ExecutorRegs() const {
        return regs;
    }

    // Layer3CaptureSourcePrep: the .gnmcap WRITER hook -- serialize a submitted frame (DCB/CCB raw
    // dwords + a full raw Regs snapshot) into a .gnmcap v1 byte buffer. This is the integration point a
    // real capture would call from SubmitGfx (after ProcessGraphics populates `regs`). NO draw / no GPU.
    std::vector<u8> ExecutorCaptureGnmcapFrame(std::span<const u32> dcb, std::span<const u32> ccb) const;

    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if (std::this_thread::get_id() == gpu_id || IsRendererTerminal()) {
            return func();
        }
        if constexpr (wait_done) {
            std::binary_semaphore sem{0};
            {
                std::scoped_lock lk{submit_mutex};
                command_queue.emplace([&sem, &func] {
                    func();
                    sem.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            sem.acquire();
        } else {
            std::scoped_lock lk{submit_mutex};
            command_queue.emplace(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
    }

    void ReserveCopyBufferSpace() {
        GpuQueue& gfx_queue = mapped_queues[GfxQueueId];
        std::scoped_lock lk(gfx_queue.m_access);
        constexpr size_t GfxReservedSize = 2_MB >> 2;
        gfx_queue.ccb_buffer.reserve(GfxReservedSize);
        gfx_queue.dcb_buffer.reserve(GfxReservedSize);
    }

    inline ComputeProgram& GetCsRegs() {
        return mapped_queues[curr_qid].cs_state;
    }

    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet;
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{};

private:
    u64 graphics_state_generation{1};

    enum class Pm4Engine : u8 {
        Graphics,
        Constant,
        Compute,
    };

    struct Pm4SnapshotKey {
        VAddr address{};
        u32 words{};
        Pm4Engine engine{};

        auto operator<=>(const Pm4SnapshotKey&) const = default;
    };

    // Shared by one published submission and all of its nested IB coroutines. The budgets turn a
    // malformed cycle/count into a controlled submit abort instead of unbounded allocation.
    struct Pm4SubmissionState {
        struct CompletionWrite {
            VAddr address{};
            u64 value{};
            u64 gpu_tick{};
            u32 num_bytes{};
            Pm4Engine engine{};
        };

        bool abort{};
        u64 copied_words{};
        u32 copied_nodes{};
        // Render-wave oracle metadata. It is inert unless the Android marker is enabled and lets
        // every nested IB report which immutable root submission it belongs to.
        u64 trace_submit_sequence{};
        VAddr trace_root_dcb{};
        u64 trace_root_hash{};
        u64 trace_semantic_sequence{};
        // CPU-authored secondary command buffers belong to the submitted command graph. Keeping
        // their bytes here prevents the guest from rebuilding an alternating child allocation while
        // a slow host pipeline compile has the GPU parser queued before that child. GPU-generated
        // children are deliberately overridden at encounter time when BufferCache marks them dirty.
        std::map<Pm4SnapshotKey, std::vector<u32>> submit_snapshots{};
#ifdef __ANDROID__
        // One-shot, submission-owned completion generations. Consuming an entry is deliberate:
        // an unobserved guest CPU store cannot otherwise be distinguished from the old EOP value.
        std::vector<CompletionWrite> completion_writes{};
#endif
    };
    using Pm4SubmissionStatePtr = std::shared_ptr<Pm4SubmissionState>;

    struct Task {
        struct promise_type {
            auto get_return_object() {
                Task task{};
                task.handle = std::coroutine_handle<promise_type>::from_promise(*this);
                return task;
            }
            static constexpr std::suspend_always initial_suspend() noexcept {
                // We want the task to be suspended at start
                return {};
            }
            static constexpr std::suspend_always final_suspend() noexcept {
                return {};
            }
            void unhandled_exception() {
                try {
                    std::rethrow_exception(std::current_exception());
                } catch (const std::exception& e) {
                    UNREACHABLE_MSG("Unhandled exception: {}", e.what());
                }
            }
            void return_void() {}
            struct empty {};
            std::suspend_always yield_value(empty&&) {
                return {};
            }
        };

        using Handle = std::coroutine_handle<promise_type>;
        Handle handle;
    };

    using CmdBuffer = std::pair<std::span<const u32>, std::span<const u32>>;
    CmdBuffer CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb);
    bool CopyIndirectAtEncounter(Pm4Engine engine, VAddr address, u32 num_words,
                                 std::vector<u32>& owned,
                                 const Pm4SubmissionStatePtr& state) const;
    void InvalidateIndirectSnapshotsForWrite(const Pm4SubmissionStatePtr& state, VAddr address,
                                             u64 num_bytes) const;
#ifdef __ANDROID__
    void RecordCompletionWriteForExecutor(const Pm4SubmissionStatePtr& state, Pm4Engine engine,
                                          VAddr address, u64 value, u32 num_bytes,
                                          u64 gpu_tick);
    void CompletePendingCompletionWriteForExecutor(VAddr address, u32 num_bytes, u64 gpu_tick);
    [[nodiscard]] u64 ConsumeCompletionWaitForExecutor(
        const Pm4SubmissionStatePtr& state, Pm4Engine engine,
        const PM4CmdWaitRegMem& wait) const;
    [[nodiscard]] bool SynchronizeUntrackedMemoryWaitForExecutor(
        const PM4CmdWaitRegMem& wait, u32& value);
#endif
    void SnapshotCpuIndirectGraph(Pm4Engine engine, std::span<const u32> stream,
                                  const Pm4SubmissionStatePtr& state, u32 depth = 0) const;
    Task ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                         std::vector<u32> owned_dcb = {}, std::vector<u32> owned_ccb = {},
                         Pm4SubmissionStatePtr state = {}, u32 ib_depth = 0,
                         Task* inherited_ce_task = nullptr, VAddr logical_dcb_base = 0);
    Task ProcessCeUpdate(std::span<const u32> ccb, std::vector<u32> owned_ccb = {},
                         Pm4SubmissionStatePtr state = {}, u32 ib_depth = 0);
    template <bool is_indirect = false>
    Task ProcessCompute(std::span<const u32> acb, u32 vqid,
                        std::vector<u32> owned_acb = {}, Pm4SubmissionStatePtr state = {},
                        u32 ib_depth = 0, VAddr logical_acb_base = 0);

    void ProcessCommands();
    void Process(std::stop_token stoken);
    [[nodiscard]] bool PublishSubmit(u32 qid, Task::Handle handle,
                                     u32* published_submit_count = nullptr);
    [[nodiscard]] u32 RecountQueuedSubmits();
#ifdef __ANDROID__
    [[nodiscard]] u64 ArmGpuCompletionTickForExecutor();
    void FlushGpuCompletionBatchForExecutor(bool force, bool synchronization_wait);
#endif

    // Multi-producer/single-consumer queue for coroutine handles. Submit producers are serialized
    // by submit_mutex only for the condition-variable publication transaction; Liverpool alone
    // owns head/front/pop. The linked handoff lets that consumer inspect and retire its current
    // task without contending on a per-queue pthread mutex.
    class SubmitQueue {
        struct Node {
            explicit Node(Task::Handle handle_ = {}) : handle{handle_} {}
            std::atomic<Node*> next{nullptr};
            Task::Handle handle{};
        };

    public:
        SubmitQueue() : head{new Node{}}, tail{head} {}
        ~SubmitQueue() {
            Node* node = head;
            while (node != nullptr) {
                Node* next = node->next.load(std::memory_order_relaxed);
                delete node;
                node = next;
            }
        }
        SubmitQueue(const SubmitQueue&) = delete;
        SubmitQueue& operator=(const SubmitQueue&) = delete;

        void emplace(Task::Handle handle) {
            auto* node = new Node{handle};
            Node* previous = tail.exchange(node, std::memory_order_acq_rel);
            previous->next.store(node, std::memory_order_release);
            count.fetch_add(1, std::memory_order_release);
        }

        [[nodiscard]] Task::Handle front() const {
            Node* next = head->next.load(std::memory_order_acquire);
            return next != nullptr ? next->handle : Task::Handle{};
        }

        [[nodiscard]] bool empty() const {
            return front() == Task::Handle{};
        }

        [[nodiscard]] std::size_t size() const {
            return count.load(std::memory_order_acquire);
        }

        void pop() {
            Node* next = head->next.load(std::memory_order_acquire);
            ASSERT_MSG(next != nullptr, "Liverpool submit queue underflow");
            Node* previous = head;
            head = next;
            count.fetch_sub(1, std::memory_order_release);
            delete previous;
        }

    private:
        Node* head{};
        std::atomic<Node*> tail{};
        std::atomic<std::size_t> count{};
    };

    struct GpuQueue {
        std::mutex m_access{};
        std::atomic<u32> dcb_buffer_offset;
        std::atomic<u32> ccb_buffer_offset;
        std::vector<u32> dcb_buffer;
        std::vector<u32> ccb_buffer;
        SubmitQueue submits{};
        ComputeProgram cs_state{};
    };
    std::array<GpuQueue, NumTotalQueues> mapped_queues{};
    // SubmitAsc publishes new queue indices from guest threads while Process round-robins them on
    // the GPU thread. Monotonic atomic publication removes that otherwise-unsynchronised read/write.
    std::atomic<u32> num_mapped_queues{1u}; // GFX is always available
    // One bit per queue with at least one published coroutine. Producers set their bit after the
    // MPSC link becomes visible; the sole consumer clears it only after retiring the last node.
    // This avoids scanning all 57 mapped queue slots for every small PM4 coroutine resume.
    std::atomic<u64> submit_ready_mask{};

    VAddr indirect_args_addr{};
    u32 num_counter_pairs{};
    u64 pixel_counter{};

    struct ConstantEngine {
        void Reset() {
            ce_count = 0;
            de_count = 0;
            ce_compare_count = 0;
        }

        [[nodiscard]] u32 Diff() const {
            ASSERT_MSG(ce_count >= de_count, "DE counter is ahead of CE");
            return ce_count - de_count;
        }

        u32 ce_compare_count{};
        u32 ce_count{};
        u32 de_count{};
        static std::array<u8, 48_KB> constants_heap;
    } cblock{};

    Vulkan::Rasterizer* rasterizer{};
    Libraries::VideoOut::VideoOutPort* vo_port{};
    std::jthread process_thread{};
    std::atomic<u32> num_submits{};
    std::atomic<u32> num_commands{};
    std::atomic<bool> submit_done{};
#ifdef __ANDROID__
    struct ExecutorPendingCompletionWrite {
        VAddr address{};
        u64 gpu_tick{};
        u32 num_bytes{};

        bool operator<(const ExecutorPendingCompletionWrite& other) const noexcept {
            return std::tie(gpu_tick, address, num_bytes) <
                   std::tie(other.gpu_tick, other.address, other.num_bytes);
        }
    };
    std::mutex executor_pending_completion_mutex;
    // Duplicate completion triples are reference-counted. Retirement callbacks can now erase their
    // exact registry entry in O(log N), instead of linearly rescanning hundreds of pending writes
    // under the page-watch mutex for every EOP/EOS/RELEASE callback.
    std::map<ExecutorPendingCompletionWrite, u32> executor_pending_completion_writes{};
    std::map<VAddr, u32> executor_pending_completion_page_refs{};
    std::atomic<u64> executor_pending_read_flushes{};
    // Completion ticks for guest submit boundaries. The bounded Android pipeline keeps at most two
    // guest frames outstanding: enough CPU/GPU overlap to remove the old per-frame queue drain, but
    // never an unbounded Vulkan backlog that lets guest resources lap the GPU.
    std::array<u64, 2> executor_async_frame_ticks{};
    u64 executor_async_last_tick{};
    u32 executor_async_frame_head{};
    u32 executor_async_frame_count{};
    bool executor_completion_flush_pending{};
    u64 executor_guest_submit_count{};
    u64 executor_completion_point_count{};
    u64 executor_submit_done_frame_count{};
    u64 executor_metrics_last_guest_submits{};
    u64 executor_metrics_last_completion_points{};
    u64 executor_metrics_last_scheduler_tick{1};
    u64 executor_metrics_wait_flushes{};
    u64 executor_metrics_last_wait_flushes{};
    u64 executor_completion_wait_stalls{};
    u64 executor_completion_wait_stall_ns{};
    u64 executor_metrics_last_completion_wait_stalls{};
    u64 executor_metrics_last_completion_wait_stall_ns{};
    u64 executor_completion_wait_stall_max_ns{};
    u64 executor_logical_wait_count{};
    u64 executor_metrics_last_logical_waits{};
    u64 executor_admission_wait_count{};
    u64 executor_admission_wait_ns{};
    u64 executor_admission_wait_max_ns{};
    u64 executor_admission_free_retirements{};
    u64 executor_metrics_last_admission_wait_count{};
    u64 executor_metrics_last_admission_wait_ns{};
    u64 executor_metrics_last_admission_free_retirements{};
#endif
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::queue<Common::UniqueFunction<void>> command_queue{};
    std::thread::id gpu_id;
    s32 curr_qid{-1};
};

} // namespace AmdGpu
