// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/kernel/threads/pthread.h"

namespace Libraries::Kernel {

void PS4_SYSV_ABI ClearStack() {
    void* const stackaddr_attr = Libraries::Kernel::g_curthread->attr.stackaddr_attr;
    void* volatile sp = nullptr;
#if defined(ARCH_X86_64)
    asm("mov %%rsp, %0" : "=rm"(sp));
#elif defined(ARCH_ARM64)
    asm("mov %0, sp" : "=r"(sp));
#else
    sp = __builtin_frame_address(0);
#endif
    const auto stack_base = reinterpret_cast<uintptr_t>(stackaddr_attr);
    const auto stack_top = reinterpret_cast<uintptr_t>(sp);
    if (stack_top <= stack_base + 64) {
        return;
    }
    const size_t size = (stack_top - stack_base) - 64;
    void* volatile buf = alloca(size);
    memset(buf, 0, size);
    sp = nullptr;
}

void RegisterThreads(Core::Loader::SymbolsResolver* sym) {
    RegisterMutex(sym);
    RegisterCond(sym);
    RegisterBarrier(sym);
    RegisterRwlock(sym);
    RegisterSemaphore(sym);
    RegisterSpec(sym);
    RegisterThreadAttr(sym);
    RegisterThread(sym);
    RegisterRtld(sym);
    RegisterPthreadClean(sym);
}

}
