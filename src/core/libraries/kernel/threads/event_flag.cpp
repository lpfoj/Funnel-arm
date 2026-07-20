// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <condition_variable>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/libs.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "core/libraries/kernel/threads/pthread.h"
#endif

namespace Libraries::Kernel {

#ifdef __ANDROID__
extern "C" bool executor_live_mono_abba_consume_resume_event_escape(const char* event_name,
                                                                    std::uint64_t bits)
    __attribute__((weak));
extern "C" void executor_live_mono_run_pending_signal_safe_point() __attribute__((weak));
extern "C" int executor_live_mono_synthetic_resume_ack(const char* reason)
    __attribute__((weak));
extern "C" void executor_live_mono_note_resume_event_waiter(void* guest_thread,
                                                            const char* event_name)
    __attribute__((weak));

static std::atomic<int> g_executor_live_mono_resume_waiters_to_release{0};

static bool ExecutorTraceHotEventFlagSet() {
    // NxSync/Game:Main event flags are set thousands of times during a normal render run. Preserve
    // the frontier probe, but keep it completely off the production path unless explicitly asked.
    static const bool enabled = [] {
        const char* value = std::getenv("EXECUTOR_TRACE_EVENTFLAG_SET");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool ExecutorAllowAndroidMonoResumeEventRelease() {
    const char* value = std::getenv("EXECUTOR_ALLOW_ANDROID_MONO_RESUME_EVENT_RELEASE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool ExecutorAllowAndroidMonoPostAckResumeEventRelease() {
    const char* value = std::getenv("EXECUTOR_ALLOW_ANDROID_MONO_POST_ACK_RESUME_EVENT_RELEASE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

extern "C" void executor_live_mono_arm_resume_event_after_waiters(int waiters,
                                                                  const char* reason) {
    if (!ExecutorAllowAndroidMonoResumeEventRelease()) {
        static std::atomic_int log_budget{16};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_MONO_RESUME_EVENT_ARM] waiters=%d "
                                "reason=%s result=disabled_android_release",
                                waiters, reason ? reason : "");
        }
        return;
    }
    if (waiters <= 0) {
        return;
    }
    int previous = g_executor_live_mono_resume_waiters_to_release.load(std::memory_order_acquire);
    while (previous < waiters &&
           !g_executor_live_mono_resume_waiters_to_release.compare_exchange_weak(
               previous, waiters, std::memory_order_acq_rel, std::memory_order_acquire)) {
    }
    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_MONO_RESUME_EVENT_ARM] waiters=%d previous=%d "
                            "reason=%s",
                            waiters, previous, reason ? reason : "");
    }
}

static void RunMonoPendingSignalSafePointFromEventFlagWait(std::unique_lock<std::mutex>& lock) {
    if (!executor_live_mono_run_pending_signal_safe_point) {
        return;
    }

    lock.unlock();
    executor_live_mono_run_pending_signal_safe_point();
    lock.lock();
}
#endif

constexpr int ORBIS_KERNEL_EVF_ATTR_TH_FIFO = 0x01;
constexpr int ORBIS_KERNEL_EVF_ATTR_TH_PRIO = 0x02;
constexpr int ORBIS_KERNEL_EVF_ATTR_SINGLE = 0x10;
constexpr int ORBIS_KERNEL_EVF_ATTR_MULTI = 0x20;

constexpr int ORBIS_KERNEL_EVF_WAITMODE_AND = 0x01;
constexpr int ORBIS_KERNEL_EVF_WAITMODE_OR = 0x02;
constexpr int ORBIS_KERNEL_EVF_WAITMODE_CLEAR_ALL = 0x10;
constexpr int ORBIS_KERNEL_EVF_WAITMODE_CLEAR_PAT = 0x20;

class EventFlagInternal {
public:
    enum class ClearMode { None, All, Bits };
    enum class WaitMode { And, Or };
    enum class ThreadMode { Single, Multi };
    enum class QueueMode { Fifo, ThreadPrio };

    EventFlagInternal(const std::string& name, ThreadMode thread_mode, QueueMode queue_mode,
                      uint64_t bits)
        : m_name(name), m_thread_mode(thread_mode), m_queue_mode(queue_mode), m_bits(bits) {};

    int Wait(u64 bits, WaitMode wait_mode, ClearMode clear_mode, u64* result, u32* ptr_micros) {
        std::unique_lock lock{m_mutex};

        uint32_t micros = 0;
        bool infinitely = true;
        if (ptr_micros != nullptr) {
            micros = *ptr_micros;
            infinitely = false;
        }

        if (m_thread_mode == ThreadMode::Single && m_waiting_threads > 0) {
            return ORBIS_KERNEL_ERROR_EPERM;
        }

        auto const start = std::chrono::system_clock::now();
        m_waiting_threads++;
        auto waitFunc = [this, wait_mode, bits] {
            return (m_status == Status::Canceled || m_status == Status::Deleted ||
                    (wait_mode == WaitMode::And && (m_bits & bits) == bits) ||
                    (wait_mode == WaitMode::Or && (m_bits & bits) != 0));
        };

#ifdef __ANDROID__
        if (m_name == "resumeEvent" && (bits & 1) != 0) {
            const int need =
                g_executor_live_mono_resume_waiters_to_release.load(std::memory_order_acquire);
            if (need > 0 && m_waiting_threads >= need && !waitFunc()) {
                int expected = need;
                if (g_executor_live_mono_resume_waiters_to_release.compare_exchange_strong(
                        expected, 0, std::memory_order_acq_rel, std::memory_order_acquire)) {
                    const u64 before = m_bits;
                    m_bits |= bits;
                    m_cond_var.notify_all();
                    static std::atomic_int log_budget{64};
                    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                        __android_log_print(
                            ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_MONO_RESUME_EVENT_RELEASE] waiters=%d need=%d "
                            "before=0x%llx after=0x%llx bits=0x%llx thread=%s",
                            m_waiting_threads, need, static_cast<unsigned long long>(before),
                            static_cast<unsigned long long>(m_bits),
                            static_cast<unsigned long long>(bits),
                            g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>");
                    }
                }
            }
        }
        bool executor_abba_resume_escape = false;
        auto shouldEscapeMonoAbbaResume = [this, bits] {
            return executor_live_mono_abba_consume_resume_event_escape &&
                   executor_live_mono_abba_consume_resume_event_escape(m_name.c_str(), bits);
        };

        if (infinitely) {
            while (!waitFunc()) {
                if (shouldEscapeMonoAbbaResume()) {
                    executor_abba_resume_escape = true;
                    break;
                }
                RunMonoPendingSignalSafePointFromEventFlagWait(lock);
                if (waitFunc()) {
                    break;
                }
                m_cond_var.wait_for(lock, std::chrono::milliseconds(1));
            }
        } else {
            const auto deadline = start + std::chrono::microseconds(micros);
            while (!waitFunc()) {
                if (shouldEscapeMonoAbbaResume()) {
                    executor_abba_resume_escape = true;
                    break;
                }
                RunMonoPendingSignalSafePointFromEventFlagWait(lock);
                if (waitFunc()) {
                    break;
                }
                const auto now = std::chrono::system_clock::now();
                if (now >= deadline) {
                    if (result != nullptr) {
                        *result = m_bits;
                    }
                    *ptr_micros = 0;
                    --m_waiting_threads;
                    return ORBIS_KERNEL_ERROR_ETIMEDOUT;
                }
                const auto remaining = deadline - now;
                const auto slice = remaining < std::chrono::milliseconds(1)
                                       ? remaining
                                       : std::chrono::system_clock::duration{
                                             std::chrono::milliseconds(1)};
                m_cond_var.wait_for(lock, slice);
            }
        }
#else
        if (infinitely) {
            m_cond_var.wait(lock, waitFunc);
        } else {
            if (!m_cond_var.wait_for(lock, std::chrono::microseconds(micros), waitFunc)) {
                if (result != nullptr) {
                    *result = m_bits;
                }
                *ptr_micros = 0;
                --m_waiting_threads;
                return ORBIS_KERNEL_ERROR_ETIMEDOUT;
            }
        }
#endif
        --m_waiting_threads;
        if (result != nullptr) {
            *result = m_bits;
#ifdef __ANDROID__
            if (executor_abba_resume_escape) {
                *result = bits;
            }
#endif
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::system_clock::now() - start)
                           .count();
        if (result != nullptr) {
            *result = m_bits;
#ifdef __ANDROID__
            if (executor_abba_resume_escape) {
                *result = bits;
            }
#endif
        }

        if (ptr_micros != nullptr) {
            *ptr_micros = (elapsed >= micros ? 0 : micros - elapsed);
        }

        if (m_status == Status::Canceled) {
            return ORBIS_KERNEL_ERROR_ECANCELED;
        } else if (m_status == Status::Deleted) {
            return ORBIS_KERNEL_ERROR_EACCES;
        }

        if (clear_mode == ClearMode::All) {
            m_bits = 0;
        } else if (clear_mode == ClearMode::Bits) {
            m_bits &= ~bits;
        }

        return ORBIS_OK;
    }

    int Poll(u64 bits, WaitMode wait_mode, ClearMode clear_mode, u64* result) {
        u32 micros = 0;
        auto ret = Wait(bits, wait_mode, clear_mode, result, &micros);
        if (ret == ORBIS_KERNEL_ERROR_ETIMEDOUT) {
            // Poll returns EBUSY instead.
            ret = ORBIS_KERNEL_ERROR_EBUSY;
        }
        return ret;
    }

    void Set(u64 bits) {
        std::unique_lock lock{m_mutex};

        while (m_status != Status::Set) {
            m_mutex.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            m_mutex.lock();
        }

        m_bits |= bits;
        m_cond_var.notify_all();
#ifdef __ANDROID__
        // FRONTIER-2: a producer worker signalling job-completion via an event flag. If the scene
        // build stalls with ready=0, NO Set fires here from a UnityWorker (candidate a). Seeing one
        // proves the producer reached completion and the stall is elsewhere. Gate to the render/
        // preload/worker threads so the common event-flag traffic stays quiet.
        const bool ef_interesting = ExecutorTraceHotEventFlagSet() &&
            g_curthread && (g_curthread->name.find("UnityWorker") != std::string::npos ||
                            g_curthread->name.find("UnityPreload") != std::string::npos ||
                            g_curthread->name.find("Game:Main") != std::string::npos ||
                            g_curthread->name.find("mono") != std::string::npos ||
                            g_curthread->name.find("Mono") != std::string::npos);
        if (ef_interesting) {
            static std::atomic_int ef_set_log_budget{2048};
            if (ef_set_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_EVENTFLAG_SET] name=%s bits=0x%llx newBits=0x%llx "
                                    "waiters=%d thread=%s",
                                    m_name.c_str(), static_cast<unsigned long long>(bits),
                                    static_cast<unsigned long long>(m_bits), m_waiting_threads,
                                    g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>");
            }
        }
#endif
    }

    void Clear(u64 bits) {
        std::unique_lock lock{m_mutex};
        while (m_status != Status::Set) {
            m_mutex.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            m_mutex.lock();
        }

        m_bits &= bits;
    }

    void Cancel(u64 setPattern, int* numWaitThreads) {
        std::unique_lock lock{m_mutex};

        while (m_status != Status::Set) {
            m_mutex.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            m_mutex.lock();
        }

        if (numWaitThreads) {
            *numWaitThreads = m_waiting_threads;
        }

        m_status = Status::Canceled;
        m_bits = setPattern;

        m_cond_var.notify_all();

        while (m_waiting_threads > 0) {
            m_mutex.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            m_mutex.lock();
        }

        m_status = Status::Set;
    }

#ifdef __ANDROID__
    u64 DebugBits() {
        std::lock_guard lock{m_mutex};
        return m_bits;
    }

    int DebugWaiterCount() {
        std::lock_guard lock{m_mutex};
        return m_waiting_threads;
    }

    const std::string& DebugName() const {
        return m_name;
    }
#endif

private:
    enum class Status { Set, Canceled, Deleted };

    std::mutex m_mutex;
    std::condition_variable m_cond_var;
    Status m_status = Status::Set;
    int m_waiting_threads = 0;
    std::string m_name;
    ThreadMode m_thread_mode = ThreadMode::Single;
    QueueMode m_queue_mode = QueueMode::Fifo;
    u64 m_bits = 0;
};

using OrbisKernelUseconds = u32;
using OrbisKernelEventFlag = EventFlagInternal*;

struct OrbisKernelEventFlagOptParam {
    size_t size;
};

#ifdef __ANDROID__
static std::atomic<OrbisKernelEventFlag> g_executor_live_mono_resume_event{nullptr};
static void TraceLiveEventFlag(const char* op, OrbisKernelEventFlag ef, u64 before, u64 after,
                               u64 bitPattern, u32 waitMode, u64 result,
                               const OrbisKernelUseconds* timeout, int ret);

extern "C" bool executor_live_signal_mono_resume_event_for_live_abba() {
    if (!ExecutorAllowAndroidMonoPostAckResumeEventRelease()) {
        static std::atomic_int disabled_budget{16};
        if (disabled_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_MONO_RESUME_EVENT_SIGNAL] result=disabled_post_ack_release");
        }
        return false;
    }

    OrbisKernelEventFlag ef = g_executor_live_mono_resume_event.load(std::memory_order_acquire);
    if (ef == nullptr) {
        static std::atomic_int missing_budget{32};
        if (missing_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_MONO_RESUME_EVENT_SIGNAL] result=missing");
        }
        return false;
    }

    const u64 before = ef->DebugBits();
    ef->Set(1);
    TraceLiveEventFlag("set_mono_abba_resume", ef, before, ef->DebugBits(), 1, 0, 0, nullptr,
                       ORBIS_OK);
    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_MONO_RESUME_EVENT_SIGNAL] result=ok ef=%p before=0x%llx "
                            "after=0x%llx",
                            ef, static_cast<unsigned long long>(before),
                            static_cast<unsigned long long>(ef->DebugBits()));
    }
    return true;
}

static bool ExecutorEnvFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

struct EventFlagTraceConfig {
    bool light_oracle = false;
    bool live_wide = false;
    bool live_sync = false;
    bool render_thread_only = false;
};

static const EventFlagTraceConfig& GetEventFlagTraceConfig() {
    // Trace policy is fixed before the first guest event flag is touched. Keeping the four
    // answers together removes repeated getenv walks from every production wait/set operation.
    static const EventFlagTraceConfig config{
        .light_oracle = ExecutorEnvFlag("EXECUTOR_LIGHT_ORACLE"),
        .live_wide = ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_WIDE"),
        .live_sync = ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_SYNC"),
        .render_thread_only = ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY"),
    };
    return config;
}

static bool IsWatchedEventFlag(OrbisKernelEventFlag ef, bool mark);

static bool IsLiveRenderThreadName(const std::string& name) {
    return name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("Game:Main") != std::string::npos ||
           name.find("Submit Done Thread") != std::string::npos;
}

static bool IsLiveInterestingThreadName(const std::string& name) {
    if (GetEventFlagTraceConfig().render_thread_only) {
        return IsLiveRenderThreadName(name);
    }
    return IsLiveRenderThreadName(name) || name.find("mono") != std::string::npos ||
           name.find("Mono") != std::string::npos ||
           name.find("Finalizer") != std::string::npos ||
           name.find("Thread Pool") != std::string::npos;
}

static bool IsMonoSuspendThreadName(const std::string& name) {
    return name.find("UnityPreload") != std::string::npos ||
           name.find("mono") != std::string::npos ||
           name.find("Mono") != std::string::npos;
}

static bool IsEventFlagFlowOp(const char* op) {
    return op && (std::strcmp(op, "create") == 0 || std::strcmp(op, "open_synthetic") == 0 ||
                  std::strcmp(op, "open") == 0 || std::strcmp(op, "set") == 0 ||
                  std::strcmp(op, "clear") == 0 || std::strcmp(op, "cancel") == 0 ||
                  std::strcmp(op, "wait_enter") == 0 ||
                  std::strcmp(op, "wait_return") == 0);
}

static bool ShouldTraceTargetedEventFlag(const char* op, OrbisKernelEventFlag ef) {
    if (!GetEventFlagTraceConfig().light_oracle || !g_curthread || !IsEventFlagFlowOp(op)) {
        return false;
    }
    if (IsMonoSuspendThreadName(g_curthread->name)) {
        return true;
    }
    return ef != nullptr && IsWatchedEventFlag(ef, false);
}

static bool ShouldTraceLiveEventFlag() {
    const auto& config = GetEventFlagTraceConfig();
    return config.live_wide || config.live_sync;
}

static bool IsCurrentLiveInterestingThread() {
    return ShouldTraceLiveEventFlag() && g_curthread && IsLiveInterestingThreadName(g_curthread->name);
}

static bool IsWatchedEventFlag(OrbisKernelEventFlag ef, bool mark) {
    static std::atomic<std::uintptr_t> watched[128]{};
    const auto value = reinterpret_cast<std::uintptr_t>(ef);
    if (value == 0) {
        return false;
    }
    for (auto& slot : watched) {
        if (slot.load(std::memory_order_relaxed) == value) {
            return true;
        }
    }
    if (!mark) {
        return false;
    }
    for (auto& slot : watched) {
        std::uintptr_t expected = 0;
        if (slot.compare_exchange_strong(expected, value, std::memory_order_relaxed) ||
            expected == value) {
            return true;
        }
    }
    return false;
}

static void TraceLiveEventFlag(const char* op, OrbisKernelEventFlag ef, u64 before, u64 after,
                               u64 bit_pattern, u32 wait_mode, u64 result_pat, const u32* timeout,
                               int ret) {
    const bool interesting = IsCurrentLiveInterestingThread();
    const bool targeted = ShouldTraceTargetedEventFlag(op, ef);
    if (interesting) {
        (void)IsWatchedEventFlag(ef, true);
    }
    if (targeted) {
        (void)IsWatchedEventFlag(ef, true);
    }
    if (!interesting && !targeted && !IsWatchedEventFlag(ef, false)) {
        return;
    }
    if (std::strcmp(op ? op : "", "wait_enter") == 0 && ef != nullptr &&
        executor_live_mono_note_resume_event_waiter != nullptr) {
        const auto& name = ef->DebugName();
        if (name == "resumeEvent") {
            executor_live_mono_note_resume_event_waiter(g_curthread, name.c_str());
        }
    }
    static std::atomic_int budget{8192};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_EVF] op=%s thread=%s tid=%ld watched=%u ef=%p before=0x%llx "
        "after=0x%llx bits=0x%llx waitMode=0x%x result=0x%llx timeout=%lld ret=0x%x "
        "waiters=%d name=%s",
        op, g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
        static_cast<long>(::syscall(SYS_gettid)), interesting ? 0u : 1u, ef,
        static_cast<unsigned long long>(before), static_cast<unsigned long long>(after),
        static_cast<unsigned long long>(bit_pattern), wait_mode,
        static_cast<unsigned long long>(result_pat),
        timeout ? static_cast<long long>(*timeout) : -1LL, static_cast<u32>(ret),
        ef ? ef->DebugWaiterCount() : -1, ef ? ef->DebugName().c_str() : "<null>");
}

static void TraceLiveEventFlagDuration(OrbisKernelEventFlag ef, int ret, long long elapsed_us,
                                       const u32* timeout) {
    if (!IsCurrentLiveInterestingThread() &&
        !ShouldTraceTargetedEventFlag("wait_return", ef) && !IsWatchedEventFlag(ef, false)) {
        return;
    }
    static std::atomic_int budget{4096};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0 && elapsed_us < 1000) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_EVF_WAIT_TIME] thread=%s tid=%ld ef=%p "
                        "elapsed_us=%lld timeout_after=%lld ret=0x%x waiters=%d name=%s",
                        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                        static_cast<long>(::syscall(SYS_gettid)), ef, elapsed_us,
                        timeout ? static_cast<long long>(*timeout) : -1LL, static_cast<u32>(ret),
                        ef ? ef->DebugWaiterCount() : -1,
                        ef ? ef->DebugName().c_str() : "<null>");
}

static std::shared_ptr<std::atomic_bool> BeginLiveEventFlagWaitWatch(OrbisKernelEventFlag ef,
                                                                     u64 before,
                                                                     u64 bit_pattern,
                                                                     u32 wait_mode,
                                                                     const u32* timeout) {
    if ((!ShouldTraceLiveEventFlag() || !IsCurrentLiveInterestingThread()) &&
        !ShouldTraceTargetedEventFlag("wait_enter", ef)) {
        return {};
    }
    const auto done = std::make_shared<std::atomic_bool>(false);
    const std::string thread_name = g_curthread ? g_curthread->name : "<no-gcurthread>";
    const long tid = static_cast<long>(::syscall(SYS_gettid));
    const long long timeout_value = timeout ? static_cast<long long>(*timeout) : -1LL;
    const std::string name = ef ? ef->DebugName() : "<null>";
    std::thread([done, ef, before, bit_pattern, wait_mode, timeout_value, thread_name, tid, name] {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (done->load(std::memory_order_acquire)) {
            return;
        }
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_EVF_STUCK] elapsed_ms=2000 thread=%s tid=%ld "
                            "ef=%p before=0x%llx bits=0x%llx waitMode=0x%x timeout=%lld "
                            "waiters=%d name=%s note=wait_enter_without_return",
                            thread_name.c_str(), tid, ef,
                            static_cast<unsigned long long>(before),
                            static_cast<unsigned long long>(bit_pattern), wait_mode,
                            timeout_value, ef ? ef->DebugWaiterCount() : -1, name.c_str());
    }).detach();
    return done;
}
#endif

int PS4_SYSV_ABI sceKernelCreateEventFlag(OrbisKernelEventFlag* ef, const char* pName, u32 attr,
                                          u64 initPattern,
                                          const OrbisKernelEventFlagOptParam* pOptParam) {
    LOG_TRACE(Kernel_Event, "called name = {} attr = {:#x} initPattern = {:#x}", pName, attr,
              initPattern);
    if (ef == nullptr || pName == nullptr) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (pOptParam || attr > (ORBIS_KERNEL_EVF_ATTR_MULTI | ORBIS_KERNEL_EVF_ATTR_TH_PRIO)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (strlen(pName) >= 32) {
        return ORBIS_KERNEL_ERROR_ENAMETOOLONG;
    }

    auto thread_mode = EventFlagInternal::ThreadMode::Single;
    auto queue_mode = EventFlagInternal::QueueMode::Fifo;
    switch (attr & 0xfu) {
    case 0x01:
        queue_mode = EventFlagInternal::QueueMode::Fifo;
        break;
    case 0x02:
        queue_mode = EventFlagInternal::QueueMode::ThreadPrio;
        break;
    case 0x00:
        break;
    default:
        UNREACHABLE();
    }

    switch (attr & 0xf0) {
    case 0x10:
        thread_mode = EventFlagInternal::ThreadMode::Single;
        break;
    case 0x20:
        thread_mode = EventFlagInternal::ThreadMode::Multi;
        break;
    case 0x00:
        break;
    default:
        UNREACHABLE();
    }

    if (queue_mode == EventFlagInternal::QueueMode::ThreadPrio) {
        LOG_ERROR(Kernel_Event, "ThreadPriority attr is not supported!");
    }

    *ef = new EventFlagInternal(std::string(pName), thread_mode, queue_mode, initPattern);
#ifdef __ANDROID__
    if (std::strcmp(pName, "resumeEvent") == 0) {
        g_executor_live_mono_resume_event.store(*ef, std::memory_order_release);
    }
    TraceLiveEventFlag("create", *ef, initPattern, initPattern, initPattern, attr, 0, nullptr,
                       ORBIS_OK);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelDeleteEventFlag(OrbisKernelEventFlag ef) {
    if (ef == nullptr) {
#ifdef __ANDROID__
        TraceLiveEventFlag("delete_missing", ef, 0, 0, 0, 0, 0, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }

#ifdef __ANDROID__
    const u64 before = ef->DebugBits();
    OrbisKernelEventFlag expected = ef;
    g_executor_live_mono_resume_event.compare_exchange_strong(expected, nullptr,
                                                              std::memory_order_acq_rel,
                                                              std::memory_order_acquire);
    TraceLiveEventFlag("delete", ef, before, before, 0, 0, 0, nullptr, ORBIS_OK);
#endif
    delete ef;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelOpenEventFlag(OrbisKernelEventFlag* ef, const char* pName, s32 flags,
                                        const OrbisKernelEventFlagOptParam* pOptParam) {
    LOG_INFO(Kernel_Event, "called name = {} flags = {:#x}", pName ? pName : "<null>", flags);
    if (ef == nullptr) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#ifdef __ANDROID__
    // Some homebrew uses OpenEventFlag for firmware-provided events before it has created
    // local flags. Android has no kernel object namespace here yet, so provide a signaled
    // synthetic object instead of leaving the guest output pointer untouched.
    *ef = new EventFlagInternal(pName ? std::string(pName) : std::string("android-opened-event"),
                                EventFlagInternal::ThreadMode::Multi,
                                EventFlagInternal::QueueMode::Fifo,
                                std::numeric_limits<u64>::max());
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_EVENT_FLAG_OPEN] name=%s flags=0x%x synthetic=%p",
                        pName ? pName : "<null>", flags, *ef);
    TraceLiveEventFlag("open_synthetic", *ef, std::numeric_limits<u64>::max(),
                       std::numeric_limits<u64>::max(), 0, static_cast<u32>(flags), 0, nullptr,
                       ORBIS_OK);
#else
    if (pOptParam != nullptr) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    *ef = new EventFlagInternal(pName ? std::string(pName) : std::string("opened-event"),
                                EventFlagInternal::ThreadMode::Multi,
                                EventFlagInternal::QueueMode::Fifo, 0);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelCloseEventFlag(OrbisKernelEventFlag ef) {
    LOG_INFO(Kernel_Event, "called ef = {}", static_cast<void*>(ef));
    if (ef == nullptr) {
#ifdef __ANDROID__
        TraceLiveEventFlag("close_missing", ef, 0, 0, 0, 0, 0, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
#ifdef __ANDROID__
    const u64 before = ef->DebugBits();
    TraceLiveEventFlag("close", ef, before, before, 0, 0, 0, nullptr, ORBIS_OK);
#endif
    delete ef;
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelClearEventFlag(OrbisKernelEventFlag ef, u64 bitPattern) {
    LOG_DEBUG(Kernel_Event, "called");
    if (ef == nullptr) {
#ifdef __ANDROID__
        TraceLiveEventFlag("clear_missing", ef, 0, 0, bitPattern, 0, 0, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
#ifdef __ANDROID__
    const u64 before = ef->DebugBits();
#endif
    ef->Clear(bitPattern);
#ifdef __ANDROID__
    TraceLiveEventFlag("clear", ef, before, ef->DebugBits(), bitPattern, 0, 0, nullptr, ORBIS_OK);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelCancelEventFlag(OrbisKernelEventFlag ef, u64 setPattern,
                                          int* pNumWaitThreads) {
    LOG_DEBUG(Kernel_Event, "called");
    if (ef == nullptr) {
#ifdef __ANDROID__
        TraceLiveEventFlag("cancel_missing", ef, 0, 0, setPattern, 0, 0, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
#ifdef __ANDROID__
    const u64 before = ef->DebugBits();
#endif
    ef->Cancel(setPattern, pNumWaitThreads);
#ifdef __ANDROID__
    TraceLiveEventFlag("cancel", ef, before, ef->DebugBits(), setPattern, 0,
                       pNumWaitThreads ? static_cast<u64>(*pNumWaitThreads) : 0, nullptr,
                       ORBIS_OK);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelSetEventFlag(OrbisKernelEventFlag ef, u64 bitPattern) {
    LOG_TRACE(Kernel_Event, "called");
    if (ef == nullptr) {
#ifdef __ANDROID__
        TraceLiveEventFlag("set_missing", ef, 0, 0, bitPattern, 0, 0, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
#ifdef __ANDROID__
    const u64 before = ef->DebugBits();
#endif
    ef->Set(bitPattern);
#ifdef __ANDROID__
    TraceLiveEventFlag("set", ef, before, ef->DebugBits(), bitPattern, 0, 0, nullptr, ORBIS_OK);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelPollEventFlag(OrbisKernelEventFlag ef, u64 bitPattern, u32 waitMode,
                                        u64* pResultPat) {
    LOG_DEBUG(Kernel_Event, "called bitPattern = {:#x} waitMode = {:#x}", bitPattern, waitMode);

    if (ef == nullptr) {
#ifdef __ANDROID__
        TraceLiveEventFlag("poll_missing", ef, 0, 0, bitPattern, waitMode, 0, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }

    if (bitPattern == 0) {
#ifdef __ANDROID__
        TraceLiveEventFlag("poll_invalid", ef, ef->DebugBits(), ef->DebugBits(), bitPattern,
                           waitMode, 0, nullptr, ORBIS_KERNEL_ERROR_EINVAL);
#endif
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto wait = EventFlagInternal::WaitMode::And;
    auto clear = EventFlagInternal::ClearMode::None;
    switch (waitMode & 0xf) {
    case 0x01:
        wait = EventFlagInternal::WaitMode::And;
        break;
    case 0x02:
        wait = EventFlagInternal::WaitMode::Or;
        break;
    default:
        UNREACHABLE();
    }

    switch (waitMode & 0xf0) {
    case 0x00:
        clear = EventFlagInternal::ClearMode::None;
        break;
    case 0x10:
        clear = EventFlagInternal::ClearMode::All;
        break;
    case 0x20:
        clear = EventFlagInternal::ClearMode::Bits;
        break;
    default:
        UNREACHABLE();
    }

#ifdef __ANDROID__
    const u64 before = ef->DebugBits();
#endif
    auto result = ef->Poll(bitPattern, wait, clear, pResultPat);
#ifdef __ANDROID__
    TraceLiveEventFlag("poll", ef, before, ef->DebugBits(), bitPattern, waitMode,
                       pResultPat ? *pResultPat : 0, nullptr, result);
#endif

    if (result != ORBIS_OK && result != ORBIS_KERNEL_ERROR_EBUSY) {
        LOG_DEBUG(Kernel_Event, "returned {:#x}", result);
    }

    return result;
}
int PS4_SYSV_ABI sceKernelWaitEventFlag(OrbisKernelEventFlag ef, u64 bitPattern, u32 waitMode,
                                        u64* pResultPat, OrbisKernelUseconds* pTimeout) {
    LOG_DEBUG(Kernel_Event, "called bitPattern = {:#x} waitMode = {:#x}", bitPattern, waitMode);
    if (ef == nullptr) {
#ifdef __ANDROID__
        TraceLiveEventFlag("wait_missing", ef, 0, 0, bitPattern, waitMode, 0, pTimeout,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }

    if (bitPattern == 0) {
#ifdef __ANDROID__
        TraceLiveEventFlag("wait_invalid", ef, ef->DebugBits(), ef->DebugBits(), bitPattern,
                           waitMode, 0, pTimeout, ORBIS_KERNEL_ERROR_EINVAL);
#endif
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto wait = EventFlagInternal::WaitMode::And;
    auto clear = EventFlagInternal::ClearMode::None;
    switch (waitMode & 0xf) {
    case 0x01:
        wait = EventFlagInternal::WaitMode::And;
        break;
    case 0x02:
        wait = EventFlagInternal::WaitMode::Or;
        break;
    default:
        UNREACHABLE();
    }

    switch (waitMode & 0xf0) {
    case 0x00:
        clear = EventFlagInternal::ClearMode::None;
        break;
    case 0x10:
        clear = EventFlagInternal::ClearMode::All;
        break;
    case 0x20:
        clear = EventFlagInternal::ClearMode::Bits;
        break;
    default:
        UNREACHABLE();
    }

#ifdef __ANDROID__
    const u64 before = ef->DebugBits();
    TraceLiveEventFlag("wait_enter", ef, before, before, bitPattern, waitMode,
                       pResultPat ? *pResultPat : 0, pTimeout, ORBIS_OK);
    auto executor_wait_watch = BeginLiveEventFlagWaitWatch(ef, before, bitPattern, waitMode, pTimeout);
    const auto executor_wait_start = std::chrono::steady_clock::now();
#endif
    const int result = ef->Wait(bitPattern, wait, clear, pResultPat, pTimeout);
#ifdef __ANDROID__
    if (executor_wait_watch) {
        executor_wait_watch->store(true, std::memory_order_release);
    }
    const auto executor_wait_end = std::chrono::steady_clock::now();
    TraceLiveEventFlag("wait_return", ef, before, ef->DebugBits(), bitPattern, waitMode,
                       pResultPat ? *pResultPat : 0, pTimeout, result);
    TraceLiveEventFlagDuration(
        ef, result,
        std::chrono::duration_cast<std::chrono::microseconds>(executor_wait_end -
                                                              executor_wait_start)
            .count(),
        pTimeout);
#endif
    if (result != ORBIS_OK && result != ORBIS_KERNEL_ERROR_ETIMEDOUT) {
        LOG_DEBUG(Kernel_Event, "returned {:#x}", result);
    }

    return result;
}

void RegisterKernelEventFlag(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("PZku4ZrXJqg", "libkernel", 1, "libkernel", sceKernelCancelEventFlag);
    LIB_FUNCTION("7uhBFWRAS60", "libkernel", 1, "libkernel", sceKernelClearEventFlag);
    LIB_FUNCTION("s9-RaxukuzQ", "libkernel", 1, "libkernel", sceKernelCloseEventFlag);
    LIB_FUNCTION("BpFoboUJoZU", "libkernel", 1, "libkernel", sceKernelCreateEventFlag);
    LIB_FUNCTION("8mql9OcQnd4", "libkernel", 1, "libkernel", sceKernelDeleteEventFlag);
    LIB_FUNCTION("1vDaenmJtyA", "libkernel", 1, "libkernel", sceKernelOpenEventFlag);
    LIB_FUNCTION("9lvj5DjHZiA", "libkernel", 1, "libkernel", sceKernelPollEventFlag);
    LIB_FUNCTION("IOnSvHzqu6A", "libkernel", 1, "libkernel", sceKernelSetEventFlag);
    LIB_FUNCTION("JTvBflhYazQ", "libkernel", 1, "libkernel", sceKernelWaitEventFlag);
}

} // namespace Libraries::Kernel
