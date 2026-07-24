// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstring>
#include "common/types.h"
#ifdef _WIN32
#include <malloc.h>
#endif

namespace Xbyak {
class CodeGenerator;
}

namespace Libraries::Fiber {
struct OrbisFiberContext;
}

namespace Core {

union DtvEntry {
    std::size_t counter;
    u8* pointer;
};

struct Tcb {
    Tcb* tcb_self;
    DtvEntry* tcb_dtv;
    void* tcb_thread;
    ::Libraries::Fiber::OrbisFiberContext* tcb_fiber;
};

#ifdef _WIN32
u32 GetTcbKey();
#endif

void SetTcbBase(void* image_address);

Tcb* GetTcbBase();

void InitializeTLS();

template <class F, F f>
struct HostCallWrapperImpl;

template <class ReturnType, class... Args, PS4_SYSV_ABI ReturnType (*func)(Args...)>
struct HostCallWrapperImpl<PS4_SYSV_ABI ReturnType (*)(Args...), func> {
    static ReturnType PS4_SYSV_ABI wrap(Args... args) {
        return func(args...);
    }
};

#define HOST_CALL(func) (Core::HostCallWrapperImpl<decltype(&(func)), func>::wrap)

}
