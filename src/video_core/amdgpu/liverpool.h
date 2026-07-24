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
    static constexpr u32 NumGfxRings = 1u;
    static constexpr u32 NumComputePipes = 7u;
    static constexpr u32 NumQueuesPerPipe = 8u;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    static_assert(NumTotalQueues < 64u);

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
        std::scoped_lock lk{submit_mutex};
        mapped_queues[GfxQueueId].ccb_buffer_offset = 0;
        mapped_queues[GfxQueueId].dcb_buffer_offset = 0;
        submit_done = true;
        submit_cv.notify_one();
    }

    void WaitGpuIdle() noexcept {
        for (int i = 0; i < 250 && num_submits.load(std::memory_order_acquire) != 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    bool IsGpuIdle() const {
        return num_submits == 0;
    }

    [[nodiscard]] u64 GraphicsStateGeneration() const noexcept {
        return graphics_state_generation;
    }

    void NotifyGraphicsStateChanged() noexcept {
        ++graphics_state_generation;
    }

    [[nodiscard]] bool IsRendererTerminal() const noexcept;
    void MarkRendererTerminal(const char* where) noexcept;

    void WaitRasterizerIdleForExecutor();

    [[nodiscard]] bool ResolvePendingCompletionReadForExecutor(VAddr address, u64 size);

    void NotifyMemoryPublicationForExecutor(VAddr address, u64 size) noexcept;

    void SetVoPort(Libraries::VideoOut::VideoOutPort* port) {
        vo_port = port;
    }

    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

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

    const Regs& ExecutorRegs() const {
        return regs;
    }

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

    struct Pm4StreamFingerprint {
        u64 primary{};
        u64 verifier{};
        u32 words{};
        Pm4Engine engine{};

        auto operator<=>(const Pm4StreamFingerprint&) const = default;
    };

    struct Pm4PacketDescriptor {
        u32 offset_words{};
        u32 packet_words{};
        u16 opcode{};
        u8 type{};
    };

    struct Pm4DescriptorPlan {
        Pm4StreamFingerprint fingerprint{};
        std::vector<Pm4PacketDescriptor> packets{};
    };

    struct Pm4DescriptorCacheLine {
        Pm4StreamFingerprint fingerprint{};
        std::shared_ptr<const Pm4DescriptorPlan> plan{};
        u64 last_use{};
    };

    struct Pm4DescriptorAdmissionLine {
        Pm4StreamFingerprint fingerprint{};
        bool occupied{};
    };

    static constexpr u32 Pm4DescriptorCacheSets = 256;
    static constexpr u32 Pm4DescriptorCacheWays = 2;
    static constexpr u32 Pm4DescriptorAdmissionLines = 1024;

    struct Pm4SnapshotKey {
        VAddr address{};
        u32 words{};
        Pm4Engine engine{};

        auto operator<=>(const Pm4SnapshotKey&) const = default;
    };

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
        u64 trace_submit_sequence{};
        VAddr trace_root_dcb{};
        u64 trace_root_hash{};
        u64 trace_semantic_sequence{};
        std::map<Pm4SnapshotKey, std::vector<u32>> submit_snapshots{};
#ifdef __ANDROID__
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
    void CompletePendingCompletionWriteForExecutor(VAddr address, u64 value, Pm4Engine engine,
                                                   u32 num_bytes, u64 gpu_tick);
    [[nodiscard]] u64 ConsumeCompletionWaitForExecutor(
        const Pm4SubmissionStatePtr& state, Pm4Engine engine,
        const PM4CmdWaitRegMem& wait);
    [[nodiscard]] bool SynchronizeUntrackedMemoryWaitForExecutor(
        const PM4CmdWaitRegMem& wait, u32& value, bool may_defer_gpu_publication,
        bool& gpu_publication_deferred);
    void ParkMemoryWaitForExecutor(u32 queue_id, const PM4CmdWaitRegMem& wait,
                                  bool gpu_publication_deferred);
    void ReleaseParkedMemoryWaitForExecutor(u32 queue_id);
    void ClearParkedMemoryWaitsForExecutor();
    void RefreshParkedMemoryWaitsForExecutor();
#endif
    void SnapshotCpuIndirectGraph(Pm4Engine engine, std::span<const u32> stream,
                                  const Pm4SubmissionStatePtr& state, u32 depth = 0) const;
    [[nodiscard]] static Pm4StreamFingerprint FingerprintPm4Stream(
        Pm4Engine engine, std::span<const u32> stream) noexcept;
    [[nodiscard]] static std::shared_ptr<const Pm4DescriptorPlan> BuildPm4DescriptorPlan(
        const Pm4StreamFingerprint& fingerprint, std::span<const u32> stream);
    [[nodiscard]] std::shared_ptr<const Pm4DescriptorPlan> FindPm4DescriptorPlan(
        const Pm4StreamFingerprint& fingerprint);
    [[nodiscard]] bool AdmitPm4DescriptorPlanBuild(
        const Pm4StreamFingerprint& fingerprint) noexcept;
    void RememberPm4DescriptorPlan(std::shared_ptr<const Pm4DescriptorPlan> plan);
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
    void QueueGpuCompletionForExecutor(u64 gpu_tick,
                                       Common::UniqueFunction<void>&& completion);
    void PublishGpuCompletionBatchForExecutor();
    void OrderGpuWorkAfterCompletionForExecutor(u64 gpu_tick);
    void FlushGpuCompletionBatchForExecutor(bool force, bool synchronization_wait);
#endif

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
    std::array<Pm4DescriptorCacheLine,
               Pm4DescriptorCacheSets * Pm4DescriptorCacheWays>
        pm4_descriptor_cache_{};
    std::array<Pm4DescriptorAdmissionLine, Pm4DescriptorAdmissionLines>
        pm4_descriptor_admission_{};
    u64 pm4_descriptor_cache_clock_{};
    u64 pm4_descriptor_cache_hits_{};
    u64 pm4_descriptor_cache_misses_{};
    u64 pm4_descriptor_cache_inserts_{};
    std::atomic<u32> num_mapped_queues{1u};
    std::atomic<u64> submit_ready_mask{};
#ifdef __ANDROID__
    struct ExecutorParkedMemoryWait {
        VAddr address{};
        u32 function{};
        u32 reference{};
        u32 mask{};
        bool active{};
    };
    std::array<ExecutorParkedMemoryWait, NumTotalQueues> executor_parked_memory_waits{};
    u64 executor_parked_memory_wait_mask{};
    u64 executor_deferred_wait_flush_mask{};
    u32 executor_parked_wait_poll_counter{};
#endif

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
        u64 value{};
        u64 gpu_tick{};
        u32 num_bytes{};
        Pm4Engine engine{};

        bool operator<(const ExecutorPendingCompletionWrite& other) const noexcept {
            return std::tie(address, gpu_tick, num_bytes, value, engine) <
                   std::tie(other.address, other.gpu_tick, other.num_bytes, other.value,
                            other.engine);
        }
    };
    std::mutex executor_pending_completion_mutex;
    std::map<ExecutorPendingCompletionWrite, u32> executor_pending_completion_writes{};
    std::map<VAddr, u32> executor_pending_completion_page_refs{};
    std::atomic<u64> executor_pending_read_flushes{};
    std::array<u64, 2> executor_async_frame_ticks{};
    u64 executor_async_last_tick{};
    u32 executor_async_frame_head{};
    u32 executor_async_frame_count{};
    bool executor_completion_flush_pending{};
    u64 executor_completion_batch_tick{};
    std::vector<Common::UniqueFunction<void>> executor_completion_batch{};
    u64 executor_guest_submit_count{};
    u64 executor_completion_point_count{};
    u64 executor_submit_done_frame_count{};
    u64 executor_metrics_last_guest_submits{};
    u64 executor_metrics_last_completion_points{};
    u64 executor_metrics_last_scheduler_tick{1};
    u64 executor_metrics_wait_flushes{};
    u64 executor_metrics_last_wait_flushes{};
    u64 executor_untracked_wait_stalls{};
    u64 executor_untracked_wait_stall_ns{};
    u64 executor_untracked_wait_stall_max_ns{};
    u64 executor_metrics_last_untracked_wait_stalls{};
    u64 executor_metrics_last_untracked_wait_stall_ns{};
    u64 executor_parked_wait_count{};
    u64 executor_parked_wait_wake_count{};
    u64 executor_parked_wait_poll_count{};
    std::atomic<u64> executor_wait_event_generation{1};
    u64 executor_parked_event_sleep_count{};
    u64 executor_parked_event_wake_count{};
    u64 executor_parked_event_timeout_count{};
    u64 executor_parked_event_sleep_ns{};
    u64 executor_metrics_last_parked_wait_count{};
    u64 executor_metrics_last_parked_wait_wake_count{};
    u64 executor_metrics_last_parked_wait_poll_count{};
    u64 executor_metrics_last_parked_event_sleep_count{};
    u64 executor_metrics_last_parked_event_wake_count{};
    u64 executor_metrics_last_parked_event_timeout_count{};
    u64 executor_metrics_last_parked_event_sleep_ns{};
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

}
