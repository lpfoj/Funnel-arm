// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/threads/thread_state.h"
#include "core/memory.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <atomic>
#include <cstdio>
#include <cstring>

static constexpr const char* ExecutorAndroidLogTag = "LSX4Native";
#endif

namespace Libraries::Kernel {

static constexpr size_t RoundUp(size_t size) {
    if (size % ThrPageSize != 0) {
        size = ((size / ThrPageSize) + 1) * ThrPageSize;
    }
    return size;
}

int ThreadState::CreateStack(PthreadAttr* attr) {
    if ((attr->stackaddr_attr) != nullptr) {
        attr->guardsize_attr = 0;
        attr->flags |= PthreadAttrFlags::StackUser;
        return 0;
    }

    size_t stacksize = RoundUp(attr->stacksize_attr);
#ifdef __ANDROID__
    if (attr->guardsize_attr != 0) {
        static std::atomic<std::uint32_t> guard_trace_count{0};
        if (guard_trace_count.fetch_add(1, std::memory_order_relaxed) < 4) {
            __android_log_print(
                ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                "[EXECUTOR_STACK_GUARD_DISABLED] stackSize=0x%zx oldGuard=0x%zx "
                "reason=android-fex-boehm-stack-scan",
                attr->stacksize_attr, attr->guardsize_attr);
        }
        attr->guardsize_attr = 0;
    }
#endif
    size_t guardsize = RoundUp(attr->guardsize_attr);

    attr->stackaddr_attr = nullptr;
    attr->flags &= ~PthreadAttrFlags::StackUser;

    thread_list_lock.lock();

    if (stacksize == ThrStackDefault && guardsize == ThrGuardDefault) {
        if (!dstackq.empty()) {
            Stack* spare_stack = dstackq.top();
            dstackq.pop();
            attr->stackaddr_attr = spare_stack->stackaddr;
        }
    }
    else {
        const auto it = std::ranges::find_if(mstackq, [&](Stack* stack) {
            return stack->stacksize == stacksize && stack->guardsize == guardsize;
        });
        if (it != mstackq.end()) {
            attr->stackaddr_attr = (*it)->stackaddr;
            mstackq.erase(it);
        }
    }

    if (attr->stackaddr_attr != nullptr) {
        thread_list_lock.unlock();
        return 0;
    }

    if (last_stack == 0) {
        static constexpr VAddr UsrStack = 0x7EFFF8000ULL;
        last_stack = UsrStack - ThrStackInitial - ThrGuardDefault;
    }

    VAddr stackaddr = last_stack - stacksize - guardsize;

    last_stack -= (stacksize + guardsize);

    thread_list_lock.unlock();

    auto* memory = Core::Memory::Instance();
    int ret = memory->MapMemory(reinterpret_cast<void**>(&stackaddr), stackaddr,
                                stacksize + guardsize, Core::MemoryProt::CpuReadWrite,
                                Core::MemoryMapFlags::NoFlags, Core::VMAType::Stack);
    ASSERT_MSG(ret == 0, "Unable to map stack memory");

    if (guardsize != 0) {
        ret = memory->Protect(stackaddr, guardsize, Core::MemoryProt::NoAccess);
        ASSERT_MSG(ret == 0, "Unable to protect guard page");
    }

    stackaddr += guardsize;
    attr->stackaddr_attr = (void*)stackaddr;

    if (attr->stackaddr_attr != nullptr) {
        return 0;
    }
    return -1;
}

void ThreadState::FreeStack(PthreadAttr* attr) {
    if (!attr || True(attr->flags & PthreadAttrFlags::StackUser) || !attr->stackaddr_attr) {
        return;
    }

    auto* stack_base = static_cast<char*>(attr->stackaddr_attr);
#ifdef __ANDROID__
    static const bool scrub_gc_root_stacks = [] {
        std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/lsx4-home/run-gc-guest-roots", "r");
        if (f) {
            std::fclose(f);
            return true;
        }
        return false;
    }();
    if (scrub_gc_root_stacks) {
        std::memset(stack_base, 0, attr->stacksize_attr);
    }
#endif
    auto* spare_stack = reinterpret_cast<Stack*>(stack_base + attr->stacksize_attr - sizeof(Stack));
    spare_stack->stacksize = RoundUp(attr->stacksize_attr);
    spare_stack->guardsize = RoundUp(attr->guardsize_attr);
    spare_stack->stackaddr = attr->stackaddr_attr;

    if (spare_stack->stacksize == ThrStackDefault && spare_stack->guardsize == ThrGuardDefault) {
        dstackq.push(spare_stack);
    } else {
        mstackq.push_back(spare_stack);
    }
    attr->stackaddr_attr = nullptr;
}

}
