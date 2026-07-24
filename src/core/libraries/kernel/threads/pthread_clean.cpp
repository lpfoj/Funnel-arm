// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/libs.h"

#ifdef __ANDROID__
#include <android/log.h>

static constexpr const char* ExecutorAndroidLogTag = "LSX4Native";

extern "C" int executor_lsx4_android_run_guest_callback(void* callback, void* arg,
                                                           const char* reason)
    __attribute__((weak));
extern "C" bool executor_lsx4_android_is_guest_address(void* address) __attribute__((weak));
#endif

namespace Libraries::Kernel {

void RunPthreadCleanupCallback(PthreadCleanupFunc routine, void* arg, const char* reason) {
    if (routine == nullptr) {
        return;
    }
#ifdef __ANDROID__
    if (executor_lsx4_android_run_guest_callback != nullptr) {
        const int rc = executor_lsx4_android_run_guest_callback(
            reinterpret_cast<void*>(routine), arg, reason ? reason : "pthread_cleanup");
        if (rc > 0) {
            return;
        }
        if (rc < 0) {
            __android_log_print(ANDROID_LOG_ERROR, ExecutorAndroidLogTag,
                                "[EXECUTOR_GUEST_CALLBACK] failed kind=cleanup reason=%s "
                                "routine=%p arg=%p rc=%d",
                                reason ? reason : "pthread_cleanup", reinterpret_cast<void*>(routine),
                                arg, rc);
            return;
        }
    }
    if (executor_lsx4_android_is_guest_address != nullptr &&
        executor_lsx4_android_is_guest_address(reinterpret_cast<void*>(routine))) {
        __android_log_print(ANDROID_LOG_ERROR, ExecutorAndroidLogTag,
                            "[EXECUTOR_GUEST_CALLBACK] blocked_direct_guest_cleanup reason=%s "
                            "routine=%p arg=%p",
                            reason ? reason : "pthread_cleanup", reinterpret_cast<void*>(routine),
                            arg);
        return;
    }
#endif
    routine(arg);
}

void PS4_SYSV_ABI __pthread_cleanup_push_imp(PthreadCleanupFunc routine, void* arg,
                                             PthreadCleanup* newbuf) {
    Pthread* curthread = CurrentOrFallbackPthread();
    if (curthread == nullptr || newbuf == nullptr) {
        return;
    }
    newbuf->routine = routine;
    newbuf->routine_arg = arg;
    newbuf->onheap = 0;
    curthread->cleanup.push_front(newbuf);
}

void PS4_SYSV_ABI posix_pthread_cleanup_push(PthreadCleanupFunc routine, void* arg) {
    Pthread* curthread = CurrentOrFallbackPthread();
    if (curthread == nullptr) {
        return;
    }
    auto* newbuf = new (std::nothrow) PthreadCleanup{};
    if (newbuf == nullptr) {
        return;
    }

    newbuf->routine = routine;
    newbuf->routine_arg = arg;
    newbuf->onheap = 1;
    curthread->cleanup.push_front(newbuf);
}

void PS4_SYSV_ABI posix_pthread_cleanup_pop(int execute) {
    Pthread* curthread = CurrentOrFallbackPthread();
    if (curthread == nullptr) {
        return;
    }
    if (!curthread->cleanup.empty()) {
        PthreadCleanup* old = curthread->cleanup.front();
        curthread->cleanup.pop_front();
        if (execute) {
            RunPthreadCleanupCallback(old->routine, old->routine_arg, "pthread_cleanup_pop");
        }
        if (old->onheap) {
            delete old;
        }
    }
}

void RegisterPthreadClean(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("4ZeZWcMsAV0", "libScePosix", 1, "libkernel", posix_pthread_cleanup_push);
    LIB_FUNCTION("RVxb0Ssa5t0", "libScePosix", 1, "libkernel", posix_pthread_cleanup_pop);

    LIB_FUNCTION("1xvtUVx1-Sg", "libkernel", 1, "libkernel", __pthread_cleanup_push_imp);
    LIB_FUNCTION("iWsFlYMf3Kw", "libkernel", 1, "libkernel", posix_pthread_cleanup_pop);
}

}
