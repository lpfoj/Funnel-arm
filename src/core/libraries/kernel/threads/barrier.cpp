// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <climits>
#include <new>

#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/threads/sleepq.h"
#include "core/libraries/libs.h"

#ifdef __ANDROID__
extern "C" void executor_live_mono_run_pending_signal_safe_point() __attribute__((weak));
#endif

namespace Libraries::Kernel {
namespace {

PthreadBarrier* const ThrBarrierDestroyed = reinterpret_cast<PthreadBarrier*>(1);
constexpr int PthreadProcessPrivate = 0;
constexpr int PthreadProcessShared = 1;

void WakeBarrierWaiter(Pthread* thread, void*) {
    thread->wake_sema.release();
}

[[nodiscard]] bool IsValidPshared(int pshared) {
    return pshared == PthreadProcessPrivate || pshared == PthreadProcessShared;
}

}

s32 PS4_SYSV_ABI posix_pthread_barrier_init(PthreadBarrierT* barrier,
                                             const PthreadBarrierAttrT* attr, u32 count) {
    if (barrier == nullptr || count == 0 || count > INT_MAX) {
        return POSIX_EINVAL;
    }
    if (attr != nullptr && *attr != nullptr && !IsValidPshared((*attr)->pshared)) {
        return POSIX_EINVAL;
    }

    auto* native = new (std::nothrow) PthreadBarrier{};
    if (native == nullptr) {
        return POSIX_ENOMEM;
    }
    native->count = static_cast<int>(count);
    *barrier = native;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_barrier_destroy(PthreadBarrierT* barrier) {
    if (barrier == nullptr || *barrier == nullptr || *barrier == ThrBarrierDestroyed) {
        return POSIX_EINVAL;
    }

    PthreadBarrier* native = *barrier;
    SleepqLock(native);
    if (native->destroying) {
        SleepqUnlock(native);
        return POSIX_EBUSY;
    }

    native->destroying = true;
    if (native->waiters != 0) {
        native->destroying = false;
        SleepqUnlock(native);
        return POSIX_EBUSY;
    }

    while (native->refcount != 0) {
        void(native->destroy_sema.try_acquire());
        SleepqUnlock(native);
        native->destroy_sema.acquire();
        SleepqLock(native);
    }

    *barrier = ThrBarrierDestroyed;
    SleepqUnlock(native);
    delete native;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_barrier_wait(PthreadBarrierT* barrier) {
    if (barrier == nullptr || *barrier == nullptr || *barrier == ThrBarrierDestroyed) {
        return POSIX_EINVAL;
    }

    PthreadBarrier* native = *barrier;
    SleepqLock(native);
    if (native->destroying) {
        SleepqUnlock(native);
        return POSIX_EBUSY;
    }

    if (++native->waiters == native->count) {
        native->waiters = 0;
        ++native->cycle;
        if (SleepQueue* queue = SleepqLookup(native); queue != nullptr) {
            SleepqDrop(queue, WakeBarrierWaiter, nullptr);
        }
        SleepqUnlock(native);
        return PthreadBarrierSerialThread;
    }

    Pthread* curthread = g_curthread;
    if (curthread == nullptr || curthread->sleepqueue == nullptr) {
        --native->waiters;
        SleepqUnlock(native);
        return POSIX_EINVAL;
    }

    const s64 cycle = native->cycle;
    ++native->refcount;
    curthread->will_sleep = true;
    SleepqAdd(native, curthread);

    for (;;) {
        curthread->ClearWake();
        SleepqUnlock(native);
#ifdef __ANDROID__
        if (executor_live_mono_run_pending_signal_safe_point != nullptr) {
            executor_live_mono_run_pending_signal_safe_point();
        }
#endif
        curthread->Sleep(nullptr, 0);
        SleepqLock(native);
        if (native->cycle != cycle) {
            break;
        }
    }

    --native->refcount;
    const bool wake_destroyer = native->destroying && native->refcount == 0;
    SleepqUnlock(native);
    if (wake_destroyer) {
        native->destroy_sema.release();
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_barrierattr_init(PthreadBarrierAttrT* attr) {
    if (attr == nullptr) {
        return POSIX_EINVAL;
    }
    auto* native = new (std::nothrow) PthreadBarrierAttr{};
    if (native == nullptr) {
        return POSIX_ENOMEM;
    }
    native->pshared = PthreadProcessPrivate;
    *attr = native;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_barrierattr_destroy(PthreadBarrierAttrT* attr) {
    if (attr == nullptr || *attr == nullptr) {
        return POSIX_EINVAL;
    }
    delete *attr;
    *attr = nullptr;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_barrierattr_getpshared(const PthreadBarrierAttrT* attr,
                                                       int* pshared) {
    if (attr == nullptr || *attr == nullptr || pshared == nullptr) {
        return POSIX_EINVAL;
    }
    *pshared = (*attr)->pshared;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_barrierattr_setpshared(PthreadBarrierAttrT* attr, int pshared) {
    if (attr == nullptr || *attr == nullptr || !IsValidPshared(pshared)) {
        return POSIX_EINVAL;
    }
    (*attr)->pshared = pshared;
    return 0;
}

void RegisterBarrier(Core::Loader::SymbolsResolver* sym) {
    const auto register_library = [sym](const char* library) {
        LIB_FUNCTION("+Pqub9HZCPo", library, 1, "libkernel", posix_pthread_barrier_destroy);
        LIB_FUNCTION("ZsXLFtd2jqQ", library, 1, "libkernel", posix_pthread_barrier_init);
        LIB_FUNCTION("CawZgCYqXWk", library, 1, "libkernel", posix_pthread_barrier_wait);
        LIB_FUNCTION("AsCQCYTbe80", library, 1, "libkernel", posix_pthread_barrierattr_destroy);
        LIB_FUNCTION("a5JZMyjFV68", library, 1, "libkernel",
                     posix_pthread_barrierattr_getpshared);
        LIB_FUNCTION("4nqCnLJSvck", library, 1, "libkernel", posix_pthread_barrierattr_init);
        LIB_FUNCTION("jqrGJJxFhmU", library, 1, "libkernel",
                     posix_pthread_barrierattr_setpshared);
    };

    register_library("libScePosix");
    register_library("libkernel");
}

}
