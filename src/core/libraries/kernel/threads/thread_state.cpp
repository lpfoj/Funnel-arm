// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>
#ifdef __ANDROID__
#include <android/log.h>
#include <string_view>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "common/alignment.h"
#include "common/scope_exit.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/threads/sleepq.h"
#include "core/libraries/kernel/threads/thread_state.h"
#include "core/memory.h"
#include "core/tls.h"

namespace Libraries::Kernel {

thread_local Pthread* g_curthread{};

#ifdef __ANDROID__
static thread_local Pthread* g_host_bound_thread = nullptr;
static bool ExecutorHostThreadBindDisabled() {
    static const bool off = [] {
        return std::fopen("/data/data/app.lsx4.android/files/lsx4-home/"
                          "run-no-host-thread-bind", "r") != nullptr;
    }();
    return off;
}
#endif

Pthread* CurrentOrFallbackPthread() {
    if (g_curthread != nullptr) {
#ifdef __ANDROID__
        if (!ExecutorHostThreadBindDisabled()) {
            g_host_bound_thread = g_curthread;
        }
#endif
        return g_curthread;
    }
#ifdef __ANDROID__
    if (!ExecutorHostThreadBindDisabled() && g_host_bound_thread != nullptr &&
        g_host_bound_thread->state != PthreadState::Dead) {
        static thread_local std::uint64_t s_bind_log_budget = 8;
        if (s_bind_log_budget) {
            --s_bind_log_budget;
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_PTHREAD_HOST_BIND] reuse thread=%p name=%s",
                                g_host_bound_thread, g_host_bound_thread->name.c_str());
        }
        return g_host_bound_thread;
    }
    char host_name[64]{};
    if (pthread_getname_np(pthread_self(), host_name, sizeof(host_name)) == 0 && host_name[0]) {
        auto* thread_state = ThrState::Instance();
        Pthread* match = nullptr;
        bool ambiguous = false;
        {
            std::scoped_lock lk{thread_state->thread_list_lock};
            for (Pthread* thread : thread_state->threads) {
                if (thread == nullptr || thread->state == PthreadState::Dead ||
                    thread->name.empty()) {
                    continue;
                }
                const std::string_view guest_name{thread->name};
                const std::string_view host_view{host_name};
                const bool exact = guest_name == host_view;
                const bool truncated =
                    (guest_name.size() > host_view.size() &&
                     guest_name.substr(0, host_view.size()) == host_view) ||
                    (host_view.size() > guest_name.size() &&
                     host_view.substr(0, guest_name.size()) == guest_name);
                if (!exact && !truncated) {
                    continue;
                }
                if (match != nullptr && match != thread) {
                    ambiguous = true;
                    break;
                }
                match = thread;
            }
        }
        if (match != nullptr && !ambiguous) {
            g_curthread = match;
            if (!ExecutorHostThreadBindDisabled()) {
                g_host_bound_thread = match;
            }
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_PTHREAD_TLS_RECOVER] host=%s thread=%p name=%s",
                                host_name, match, match->name.c_str());
            return g_curthread;
        }
    }
    thread_local Pthread fallback_thread{};
    thread_local bool fallback_initialized = false;
    if (!fallback_initialized) {
        const auto tid = static_cast<s32>(syscall(__NR_gettid));
        fallback_thread.tid.store(tid, std::memory_order_relaxed);
        fallback_thread.attr.sched_policy = SchedPolicy::Fifo;
        fallback_thread.attr.prio = ORBIS_KERNEL_PRIO_FIFO_DEFAULT;
        fallback_thread.state = PthreadState::Running;
        fallback_thread.name = std::string("ExecutorBridgeFallback:") + std::to_string(tid);
        fallback_initialized = true;
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_PTHREAD_FALLBACK] created thread=%p name=%s",
                            &fallback_thread, fallback_thread.name.c_str());
        static std::atomic<int> s_diag_budget{200};
        if (s_diag_budget.fetch_sub(1, std::memory_order_relaxed) > 0 &&
            std::fopen("/data/data/app.lsx4.android/files/lsx4-home/run-no-bridge-diag",
                       "r") == nullptr) {
            if (std::FILE* f = std::fopen(
                    "/data/data/app.lsx4.android/files/executor-bridge-threads.log", "a")) {
                std::fprintf(f, "FALLBACK host='%s' tid=%d guest=[", host_name, tid);
                auto* ts = ThrState::Instance();
                std::scoped_lock lk{ts->thread_list_lock};
                int n = 0;
                for (Pthread* t : ts->threads) {
                    if (t && t->state != PthreadState::Dead && !t->name.empty() && n++ < 40) {
                        std::fprintf(f, "%s|", t->name.c_str());
                    }
                }
                std::fprintf(f, "]\n");
                std::fclose(f);
            }
        }
    }
    return &fallback_thread;
#else
    return nullptr;
#endif
}

Core::Tcb* TcbCtor(Pthread* thread, int initial);
void TcbDtor(Core::Tcb* oldtls);

ThreadState::ThreadState() {
    auto* memory = Core::Memory::Instance();
    auto& impl = memory->GetAddressSpace();
    static constexpr u32 ThrHeapSize = Common::AlignUp(sizeof(Pthread) * MaxThreads, 16_KB);
    void* heap_addr{};
    const int ret = memory->MapMemory(&heap_addr, impl.SystemReservedVirtualBase(), ThrHeapSize,
                                      Core::MemoryProt::CpuReadWrite, Core::MemoryMapFlags::NoFlags,
                                      Core::VMAType::File, "ThrHeap");
    ASSERT_MSG(ret == 0, "Unable to allocate thread heap memory {}", ret);
    thread_heap.Initialize(heap_addr, ThrHeapSize);
}

void ThreadState::Collect(Pthread* curthread) {
    boost::container::small_vector<Pthread*, 8> work_list;
    {
        std::scoped_lock lk{thread_list_lock};
        for (auto it = gc_list.begin(); it != gc_list.end();) {
            Pthread* td = *it;
            if (td->tid != TidTerminated) {
                ++it;
                continue;
            }
            FreeStack(&td->attr);
            work_list.push_back(td);
            it = gc_list.erase(it);
        }
    }
    for (Pthread* td : work_list) {
        Free(curthread, td);
    }
}

void ThreadState::TryCollect(Pthread* thread) {
    SCOPE_EXIT {
        thread->lock.unlock();
    };
    if (!thread->ShouldCollect()) {
        return;
    }

    thread->refcount++;
    thread->lock.unlock();
    std::scoped_lock lk{thread_list_lock};
    thread->lock.lock();
    thread->refcount--;
    if (thread->ShouldCollect()) {
        threads.erase(thread);
        gc_list.push_back(thread);
    }
}

Pthread* ThreadState::Alloc(Pthread* curthread) {
    Pthread* thread = nullptr;
    if (curthread != nullptr) {
        if (GcNeeded()) {
            Collect(curthread);
        }
        if (!free_threads.empty()) {
            std::scoped_lock lk{free_thread_lock};
            if (!free_threads.empty()) {
                thread = free_threads.back();
                free_threads.pop_back();
            }
        }
    }
    if (thread == nullptr) {
        if (total_threads > MaxThreads) {
            return nullptr;
        }
        total_threads.fetch_add(1);
        thread = thread_heap.Allocate();
        if (thread == nullptr) {
            total_threads.fetch_sub(1);
            return nullptr;
        }
    }
    Core::Tcb* tcb = nullptr;
    if (curthread != nullptr) {
        std::scoped_lock lk{tcb_lock};
        tcb = TcbCtor(thread, 0);
    } else {
        tcb = TcbCtor(thread, 1);
    }
    if (tcb != nullptr) {
        std::memset(static_cast<void*>(thread), 0, sizeof(Pthread));
        std::construct_at(thread);
        thread->tcb = tcb;
        thread->sleepqueue = new SleepQueue{};
    } else {
        thread_heap.Free(thread);
        total_threads.fetch_sub(1);
        thread = nullptr;
    }
    return thread;
}

void ThreadState::Free(Pthread* curthread, Pthread* thread) {
    if (curthread != nullptr) {
        std::scoped_lock lk{tcb_lock};
        TcbDtor(thread->tcb);
    } else {
        TcbDtor(thread->tcb);
    }
    thread->tcb = nullptr;
    auto* sleepqueue = thread->sleepqueue;
    std::destroy_at(thread);
    bool should_free;
    {
        std::scoped_lock lk{free_thread_lock};
        if (free_threads.size() >= MaxCachedThreads) {
            should_free = true;
        } else {
            should_free = false;
            free_threads.push_back(thread);
        }
    }
    if (should_free) {
        delete sleepqueue;
        thread_heap.Free(thread);
        total_threads.fetch_sub(1);
    }
}

int ThreadState::FindThread(Pthread* thread, const bool include_dead) {
    if (thread == nullptr) {
        return POSIX_EINVAL;
    }
    std::scoped_lock lk{thread_list_lock};
    const auto it = threads.find(thread);
    if (it == threads.end()) {
        return POSIX_ESRCH;
    }
    thread->lock.lock();
    if (!include_dead && thread->state == PthreadState::Dead) {
        thread->lock.unlock();
        return POSIX_ESRCH;
    }
    return 0;
}

int ThreadState::RefAdd(Pthread* thread, bool include_dead) {
    if (thread == nullptr) {
        return POSIX_EINVAL;
    }

    if (int ret = FindThread(thread, include_dead); ret != 0) {
        return ret;
    }

    thread->refcount++;
    thread->lock.unlock();
    return 0;
}

void ThreadState::RefDelete(Pthread* thread) {
    thread->lock.lock();
    thread->refcount--;
    TryCollect(thread);
}

}
