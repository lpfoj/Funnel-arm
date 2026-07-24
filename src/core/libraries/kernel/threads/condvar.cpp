// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <cstdlib>
#ifdef __ANDROID__
#include <android/log.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#endif
#include "common/assert.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/threads/sleepq.h"
#include "core/libraries/kernel/threads/thread_state.h"
#include "core/libraries/libs.h"

#ifdef __ANDROID__
extern "C" void executor_live_mono_run_pending_signal_safe_point() __attribute__((weak));
#endif

namespace Libraries::Kernel {

static std::mutex CondStaticLock;

#ifdef __ANDROID__
static bool ExecutorCondSpuriousBackoffDisabled() {
    static const bool off =
        std::fopen("/data/data/app.lsx4.android/files/lsx4-home/run-no-cond-spurious-backoff",
                   "r") != nullptr;
    return off;
}
static void ExecutorCondPersistSpurious(const char* name, std::uint64_t n) {
    static std::atomic<int> budget{200};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    if (std::FILE* f =
            std::fopen("/data/data/app.lsx4.android/files/executor-condspurious.log", "a")) {
        std::fprintf(f, "spurious thread=%s count=%llu\n", name ? name : "?",
                     static_cast<unsigned long long>(n));
        std::fclose(f);
    }
}

static bool ShouldTraceUnityCond() {
    if (std::getenv("EXECUTOR_TRACE_LIVE_WIDE") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_SYNC") == nullptr &&
        std::getenv("EXECUTOR_LIGHT_ORACLE") == nullptr) {
        return false;
    }
    if (!g_curthread) {
        return false;
    }
    const auto& name = g_curthread->name;
    if (std::getenv("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY") != nullptr) {
        return name.find("Game:Main") != std::string::npos ||
               name.find("UnityGfxDeviceWorker") != std::string::npos ||
               name.find("UnityWorker") != std::string::npos ||
               name.find("Submit Done Thread") != std::string::npos ||
               name.find("SceFios") != std::string::npos;
    }
    return name.find("Game:Main") != std::string::npos ||
           name.find("UnityPreload") != std::string::npos ||
           name.find("mono") != std::string::npos ||
           name.find("Mono") != std::string::npos ||
           name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("Submit Done Thread") != std::string::npos ||
           name.find("SceFios") != std::string::npos;
}

static void TraceUnityCond(const char* op, PthreadCondT* cond, PthreadCond* native,
                           PthreadMutexT* mutex, s32 ret) {
    if (!ShouldTraceUnityCond()) {
        return;
    }
    if (std::getenv("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY") != nullptr && native &&
        native->name == "SceFiosOpWait" && !native->has_user_waiters &&
        std::strstr(op, "broadcast") == op) {
        static std::atomic_uint fios_broadcast_count{0};
        const unsigned count = fios_broadcast_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((count & (count - 1)) != 0) {
            return;
        }
    }
    static std::atomic_int budget{1024};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_COND] op=%s thread=%s cond=%p native=%p mutex=%p "
                        "waiters=%u ret=%d name=%s",
                        op, g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>", cond,
                        native, mutex, native && native->has_user_waiters ? 1u : 0u, ret,
                        native ? native->name.c_str() : "<null>");
}

static void TraceFiosCondDefer(const char* op, PthreadCond* cv, Pthread* waiter,
                               PthreadMutex* mutex, bool deferred) {
    if (std::getenv("EXECUTOR_LIGHT_ORACLE") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_WIDE") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_SYNC") == nullptr) {
        return;
    }
    const bool is_fios_cond = cv && cv->name.find("SceFios") != std::string::npos;
    const bool is_fios_cur = g_curthread && g_curthread->name.find("SceFios") != std::string::npos;
    const bool is_fios_waiter = waiter && waiter->name.find("SceFios") != std::string::npos;
    if (!is_fios_cond && !is_fios_cur && !is_fios_waiter) {
        return;
    }
    static std::atomic_int budget{256};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_FIOS_COND_DEFER] op=%s cond=%p condName=%s cur=%p curName=%s "
                        "waiter=%p waiterName=%s mutex=%p owner=%p ownerName=%s flags=0x%x "
                        "deferred=%d curDefer=%d",
                        op ? op : "?", cv, cv ? cv->name.c_str() : "<null>", g_curthread,
                        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>", waiter,
                        waiter ? waiter->name.c_str() : "<none>", mutex,
                        mutex ? mutex->m_owner : nullptr,
                        mutex && mutex->m_owner ? mutex->m_owner->name.c_str() : "<none>",
                        mutex ? static_cast<u32>(mutex->m_flags) : 0u, deferred ? 1 : 0,
                        g_curthread ? g_curthread->nwaiter_defer : 0);
}
#endif

#define THR_COND_INITIALIZER ((PthreadCond*)nullptr)
#define THR_COND_DESTROYED ((PthreadCond*)1)

static constexpr PthreadCondAttr PthreadCondattrDefault = {
    .c_pshared = 0,
    .c_clockid = ClockId::Realtime,
};

static int CondInit(PthreadCondT* cond, const PthreadCondAttrT* cond_attr, const char* name) {
    auto* cvp = new (std::nothrow) PthreadCond{};
    if (cvp == nullptr) {
        return POSIX_ENOMEM;
    }

    if (name) {
        cvp->name = name;
    } else {
        static int CondId = 0;
        cvp->name = fmt::format("Cond{}", CondId++);
    }

    if (cond_attr == nullptr || *cond_attr == nullptr) {
        cvp->clock_id = ClockId::Realtime;
    } else {
        cvp->clock_id = (*cond_attr)->c_clockid;
    }
    *cond = cvp;
    return 0;
}

static int InitStatic(Pthread* thread, PthreadCondT* cond) {
    std::scoped_lock lk{CondStaticLock};
    if (*cond == nullptr) {
        return CondInit(cond, nullptr, nullptr);
    }
    return 0;
}

#define CHECK_AND_INIT_COND                                                                        \
    if (cvp = *cond; cvp <= THR_COND_DESTROYED) [[unlikely]] {                                     \
        if (cvp == THR_COND_INITIALIZER) {                                                         \
            int ret;                                                                               \
            ret = InitStatic(g_curthread, cond);                                                   \
            if (ret)                                                                               \
                return (ret);                                                                      \
        } else if (cvp == THR_COND_DESTROYED) {                                                    \
            return POSIX_EINVAL;                                                                   \
        }                                                                                          \
        cvp = *cond;                                                                               \
    }

int PS4_SYSV_ABI posix_pthread_cond_init(PthreadCondT* cond, const PthreadCondAttrT* cond_attr) {
    *cond = nullptr;
    return CondInit(cond, cond_attr, nullptr);
}

int PS4_SYSV_ABI scePthreadCondInit(PthreadCondT* cond, const PthreadCondAttrT* cond_attr,
                                    const char* name) {
    *cond = nullptr;
    return CondInit(cond, cond_attr, name);
}

int PS4_SYSV_ABI posix_pthread_cond_destroy(PthreadCondT* cond) {
    PthreadCond* cvp = *cond;
    if (cvp == THR_COND_INITIALIZER) {
        return 0;
    }
    if (cvp == THR_COND_DESTROYED) {
        return POSIX_EINVAL;
    }
    cvp = *cond;
    *cond = THR_COND_DESTROYED;
    delete cvp;
    return 0;
}

int PthreadCond::Wait(PthreadMutexT* mutex, const OrbisKernelTimespec* abstime, u64 usec) {
    PthreadMutex* mp = *mutex;
    if (const int error = mp->IsOwned(g_curthread); error != 0) {
        return error;
    }

    Pthread* curthread = g_curthread;
    ASSERT_MSG(curthread->wchan == nullptr, "Thread was already on queue.");
    SleepqLock(this);

    has_user_waiters = true;
    curthread->will_sleep = true;

    int recurse;
    mp->CvUnlock(&recurse);

    curthread->mutex_obj = mp;
    SleepqAdd(this, curthread);

    int error = 0;
    for (;;) {
        curthread->ClearWake();
        SleepqUnlock(this);

#ifdef __ANDROID__
        if (executor_live_mono_run_pending_signal_safe_point) {
            executor_live_mono_run_pending_signal_safe_point();
        }
#endif
        error = curthread->Sleep(abstime, usec) ? 0 : POSIX_ETIMEDOUT;

        SleepqLock(this);
        if (curthread->wchan == nullptr) {
            error = 0;
            break;
        } else if (curthread->ShouldCancel()) {
            SleepQueue* sq = SleepqLookup(this);
            has_user_waiters = SleepqRemove(sq, curthread);
            SleepqUnlock(this);
            curthread->mutex_obj = nullptr;
            mp->CvLock(recurse);
            return 0;
        } else if (error == POSIX_ETIMEDOUT) {
            SleepQueue* sq = SleepqLookup(this);
            has_user_waiters = SleepqRemove(sq, curthread);
            break;
        }
#ifdef __ANDROID__
        {
            const std::uint64_t n =
                curthread->spurious_cond_wakes.fetch_add(1, std::memory_order_relaxed) + 1;
            if ((n & (n - 1)) == 0) {
                ExecutorCondPersistSpurious(curthread->name.c_str(), n);
            }
            SleepqUnlock(this);
            if (!ExecutorCondSpuriousBackoffDisabled()) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            SleepqLock(this);
            continue;
        }
#else
        UNREACHABLE();
#endif
    }
    SleepqUnlock(this);
    curthread->mutex_obj = nullptr;
    const int error2 = mp->CvLock(recurse);
    if (error == 0) {
        error = error2;
    }
    return error;
}

int PS4_SYSV_ABI posix_pthread_cond_wait(PthreadCondT* cond, PthreadMutexT* mutex) {
    PthreadCond* cvp{};
    CHECK_AND_INIT_COND
#ifdef __ANDROID__
    TraceUnityCond("wait_enter", cond, cvp, mutex, 0);
#endif
    const int ret = cvp->Wait(mutex, nullptr);
#ifdef __ANDROID__
    TraceUnityCond("wait_return", cond, cvp, mutex, ret);
#endif
    return ret;
}

int PS4_SYSV_ABI posix_pthread_cond_timedwait(PthreadCondT* cond, PthreadMutexT* mutex,
                                              const OrbisKernelTimespec* abstime) {
    if (abstime == nullptr || abstime->tv_sec < 0 || abstime->tv_nsec < 0 ||
        abstime->tv_nsec >= 1000000000) {
        return POSIX_EINVAL;
    }

    PthreadCond* cvp{};
    CHECK_AND_INIT_COND
#ifdef __ANDROID__
    TraceUnityCond("timedwait_enter", cond, cvp, mutex, 0);
#endif
    const int ret = cvp->Wait(mutex, abstime);
#ifdef __ANDROID__
    TraceUnityCond("timedwait_return", cond, cvp, mutex, ret);
#endif
    return ret;
}

int PS4_SYSV_ABI posix_pthread_cond_reltimedwait_np(PthreadCondT* cond, PthreadMutexT* mutex,
                                                    u64 usec) {
    PthreadCond* cvp{};
    CHECK_AND_INIT_COND
#ifdef __ANDROID__
    TraceUnityCond("reltimedwait_enter", cond, cvp, mutex, 0);
#endif
    const int ret = cvp->Wait(mutex, THR_RELTIME, usec);
#ifdef __ANDROID__
    TraceUnityCond("reltimedwait_return", cond, cvp, mutex, ret);
#endif
    return ret;
}

int PthreadCond::Signal(Pthread* thread) {
    Pthread* curthread = g_curthread;
    if (thread != nullptr) {
        auto* thread_state = ThrState::Instance();
        const int ret = thread_state->FindThread(thread, false);
        if (ret != ORBIS_OK) {
            return ret;
        }
        thread->lock.unlock();
    }

    SleepqLock(this);
    SleepQueue* sq = SleepqLookup(this);
    if (sq == nullptr) {
        SleepqUnlock(this);
        return 0;
    }

    Pthread* td = thread ? thread : sq->sq_blocked.front();

    PthreadMutex* mp = td->mutex_obj;
    has_user_waiters = SleepqRemove(sq, td);

    BinarySemaphore* waddr = nullptr;
    if (mp->m_owner == curthread) {
        if (curthread->nwaiter_defer >= Pthread::MaxDeferWaiters) {
            curthread->WakeAll();
        }
        curthread->defer_waiters[curthread->nwaiter_defer++] = &td->wake_sema;
        mp->m_flags |= PthreadMutexFlags::Deferred;
#ifdef __ANDROID__
        TraceFiosCondDefer("signal_defer", this, td, mp, true);
#endif
    } else {
        waddr = &td->wake_sema;
#ifdef __ANDROID__
        TraceFiosCondDefer("signal_direct", this, td, mp, false);
#endif
    }

    SleepqUnlock(this);
    if (waddr != nullptr) {
        waddr->release();
    }
    return 0;
}

struct BroadcastArg {
    Pthread* curthread;
    BinarySemaphore* waddrs[Pthread::MaxDeferWaiters];
    int count;
};

int PthreadCond::Broadcast() {
    BroadcastArg ba;
    ba.curthread = g_curthread;
    ba.count = 0;

    const auto drop_cb = [](Pthread* td, void* arg) {
        auto* ba2 = static_cast<BroadcastArg*>(arg);
        Pthread* curthread = ba2->curthread;
        PthreadMutex* mp = td->mutex_obj;

        if (mp->m_owner == curthread) {
            if (curthread->nwaiter_defer >= Pthread::MaxDeferWaiters) {
                curthread->WakeAll();
            }
            curthread->defer_waiters[curthread->nwaiter_defer++] = &td->wake_sema;
            mp->m_flags |= PthreadMutexFlags::Deferred;
#ifdef __ANDROID__
            TraceFiosCondDefer("broadcast_defer", nullptr, td, mp, true);
#endif
        } else {
            if (ba2->count >= Pthread::MaxDeferWaiters) {
                for (int i = 0; i < ba2->count; i++) {
                    ba2->waddrs[i]->release();
                }
                ba2->count = 0;
            }
            ba2->waddrs[ba2->count++] = &td->wake_sema;
#ifdef __ANDROID__
            TraceFiosCondDefer("broadcast_direct", nullptr, td, mp, false);
#endif
        }
    };

    SleepqLock(this);
    SleepQueue* sq = SleepqLookup(this);
    if (sq == nullptr) {
        SleepqUnlock(this);
        return 0;
    }

    SleepqDrop(sq, drop_cb, &ba);
    has_user_waiters = false;
    SleepqUnlock(this);

    for (int i = 0; i < ba.count; i++) {
        ba.waddrs[i]->release();
    }
    return 0;
}

int PS4_SYSV_ABI posix_pthread_cond_signal(PthreadCondT* cond) {
    PthreadCond* cvp{};
    CHECK_AND_INIT_COND
    const int ret = cvp->Signal(nullptr);
#ifdef __ANDROID__
    TraceUnityCond("signal", cond, cvp, nullptr, ret);
#endif
    return ret;
}

int PS4_SYSV_ABI posix_pthread_cond_signalto_np(PthreadCondT* cond, Pthread* thread) {
    PthreadCond* cvp{};
    CHECK_AND_INIT_COND
    const int ret = cvp->Signal(thread);
#ifdef __ANDROID__
    TraceUnityCond("signalto", cond, cvp, nullptr, ret);
#endif
    return ret;
}

int PS4_SYSV_ABI posix_pthread_cond_broadcast(PthreadCondT* cond) {
    PthreadCond* cvp{};
    CHECK_AND_INIT_COND
#ifdef __ANDROID__
    TraceUnityCond("broadcast_enter", cond, cvp, nullptr, 0);
#endif
    const int ret = cvp->Broadcast();
#ifdef __ANDROID__
    TraceUnityCond("broadcast_return", cond, cvp, nullptr, ret);
#endif
    return 0;
}

int PS4_SYSV_ABI posix_pthread_condattr_init(PthreadCondAttrT* attr) {
    auto* pattr = new (std::nothrow) PthreadCondAttr{};
    if (pattr == nullptr) {
        return POSIX_ENOMEM;
    }
    memcpy(pattr, &PthreadCondattrDefault, sizeof(PthreadCondAttr));
    *attr = pattr;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_condattr_destroy(PthreadCondAttrT* attr) {
    if (attr == nullptr || *attr == nullptr) {
        return POSIX_EINVAL;
    }
    delete *attr;
    *attr = nullptr;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_condattr_getclock(const PthreadCondAttrT* attr, ClockId* clock_id) {
    if (attr == nullptr || *attr == nullptr) {
        return POSIX_EINVAL;
    }
    *clock_id = (*attr)->c_clockid;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_condattr_setclock(PthreadCondAttrT* attr, ClockId clock_id) {
    if (attr == nullptr || *attr == nullptr) {
        return POSIX_EINVAL;
    }
    if (clock_id != ClockId::Realtime && clock_id != ClockId::Virtual &&
        clock_id != ClockId::Prof && clock_id != ClockId::Monotonic) {
        return POSIX_EINVAL;
    }
    (*attr)->c_clockid = clock_id;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_condattr_getpshared(const PthreadCondAttrT* attr, int* pshared) {
    if (attr == nullptr || *attr == nullptr) {
        return POSIX_EINVAL;
    }
    *pshared = 0;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_condattr_setpshared(PthreadCondAttrT* attr, int pshared) {
    if (attr == nullptr || *attr == nullptr) {
        return POSIX_EINVAL;
    }
    if (pshared != 0) {
        return POSIX_EINVAL;
    }
    return 0;
}

void RegisterCond(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("mKoTx03HRWA", "libScePosix", 1, "libkernel", posix_pthread_condattr_init);
    LIB_FUNCTION("3BpP850hBT4", "libScePosix", 1, "libkernel",
                 posix_pthread_condattr_setpshared);
    LIB_FUNCTION("dJcuQVn6-Iw", "libScePosix", 1, "libkernel", posix_pthread_condattr_destroy);
    LIB_FUNCTION("EjllaAqAPZo", "libScePosix", 1, "libkernel", posix_pthread_condattr_setclock);
    LIB_FUNCTION("h0qUqSuOmC8", "libScePosix", 1, "libkernel",
                 posix_pthread_condattr_getpshared);
    LIB_FUNCTION("cTDYxTUNPhM", "libScePosix", 1, "libkernel",
                 posix_pthread_condattr_getclock);
    LIB_FUNCTION("0TyVk4MSLt0", "libScePosix", 1, "libkernel", posix_pthread_cond_init);
    LIB_FUNCTION("K953PF5u6Pc", "libScePosix", 1, "libkernel",
                 posix_pthread_cond_reltimedwait_np);
    LIB_FUNCTION("2MOy+rUfuhQ", "libScePosix", 1, "libkernel", posix_pthread_cond_signal);
    LIB_FUNCTION("CI6Qy73ae10", "libScePosix", 1, "libkernel",
                 posix_pthread_cond_signalto_np);
    LIB_FUNCTION("RXXqi4CtF8w", "libScePosix", 1, "libkernel", posix_pthread_cond_destroy);
    LIB_FUNCTION("Op8TBGY5KHg", "libScePosix", 1, "libkernel", posix_pthread_cond_wait);
    LIB_FUNCTION("27bAgiJmOh0", "libScePosix", 1, "libkernel", posix_pthread_cond_timedwait);
    LIB_FUNCTION("mkx2fVhNMsg", "libScePosix", 1, "libkernel", posix_pthread_cond_broadcast);

    LIB_FUNCTION("0TyVk4MSLt0", "libkernel", 1, "libkernel", posix_pthread_cond_init);
    LIB_FUNCTION("3BpP850hBT4", "libkernel", 1, "libkernel", posix_pthread_condattr_setpshared);
    LIB_FUNCTION("EjllaAqAPZo", "libkernel", 1, "libkernel", posix_pthread_condattr_setclock);
    LIB_FUNCTION("h0qUqSuOmC8", "libkernel", 1, "libkernel", posix_pthread_condattr_getpshared);
    LIB_FUNCTION("cTDYxTUNPhM", "libkernel", 1, "libkernel", posix_pthread_condattr_getclock);
    LIB_FUNCTION("K953PF5u6Pc", "libkernel", 1, "libkernel",
                 posix_pthread_cond_reltimedwait_np);
    LIB_FUNCTION("Op8TBGY5KHg", "libkernel", 1, "libkernel", posix_pthread_cond_wait);
    LIB_FUNCTION("mkx2fVhNMsg", "libkernel", 1, "libkernel", posix_pthread_cond_broadcast);
    LIB_FUNCTION("2MOy+rUfuhQ", "libkernel", 1, "libkernel", posix_pthread_cond_signal);
    LIB_FUNCTION("CI6Qy73ae10", "libkernel", 1, "libkernel", posix_pthread_cond_signalto_np);
    LIB_FUNCTION("RXXqi4CtF8w", "libkernel", 1, "libkernel", posix_pthread_cond_destroy);
    LIB_FUNCTION("27bAgiJmOh0", "libkernel", 1, "libkernel", posix_pthread_cond_timedwait);
    LIB_FUNCTION("mKoTx03HRWA", "libkernel", 1, "libkernel", posix_pthread_condattr_init);
    LIB_FUNCTION("dJcuQVn6-Iw", "libkernel", 1, "libkernel", posix_pthread_condattr_destroy);

    LIB_FUNCTION("2Tb92quprl0", "libkernel", 1, "libkernel", ORBIS(scePthreadCondInit));
    LIB_FUNCTION("m5-2bsNfv7s", "libkernel", 1, "libkernel", ORBIS(posix_pthread_condattr_init));
    LIB_FUNCTION("6xMew9+rZwI", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_condattr_setpshared));
    LIB_FUNCTION("c-bxj027czs", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_condattr_setclock));
    LIB_FUNCTION("Dn-DRWi9t54", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_condattr_getpshared));
    LIB_FUNCTION("6qM3kO5S3Oo", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_condattr_getclock));
    LIB_FUNCTION("JGgj7Uvrl+A", "libkernel", 1, "libkernel", ORBIS(posix_pthread_cond_broadcast));
    LIB_FUNCTION("WKAXJ4XBPQ4", "libkernel", 1, "libkernel", ORBIS(posix_pthread_cond_wait));
    LIB_FUNCTION("waPcxYiR3WA", "libkernel", 1, "libkernel", ORBIS(posix_pthread_condattr_destroy));
    LIB_FUNCTION("kDh-NfxgMtE", "libkernel", 1, "libkernel", ORBIS(posix_pthread_cond_signal));
    LIB_FUNCTION("BmMjYxmew1w", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_cond_reltimedwait_np));
    LIB_FUNCTION("g+PZd2hiacg", "libkernel", 1, "libkernel", ORBIS(posix_pthread_cond_destroy));
    LIB_FUNCTION("o69RpYO-Mu0", "libkernel", 1, "libkernel", ORBIS(posix_pthread_cond_signalto_np));
}

}
