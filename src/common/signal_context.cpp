// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/assert.h"
#include "common/signal_context.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ucontext.h>
#if defined(__ANDROID__) && defined(ARCH_ARM64)
#include <asm/sigcontext.h>
#endif
#endif

namespace Common {

void* GetXmmPointer(void* ctx, u8 index) {
#if defined(_WIN32)
#define CASE(index)                                                                                \
    case index:                                                                                    \
        return (void*)(&((EXCEPTION_POINTERS*)ctx)->ContextRecord->Xmm##index.Low)
#elif defined(__APPLE__)
#define CASE(index)                                                                                \
    case index:                                                                                    \
        return (void*)(&((ucontext_t*)ctx)->uc_mcontext->__fs.__fpu_xmm##index);
#elif defined(__ANDROID__) && defined(ARCH_ARM64)
#define CASE(index)                                                                                \
    case index:                                                                                    \
        return nullptr
#else
#define CASE(index)                                                                                \
    case index:                                                                                    \
        return (void*)(&((ucontext_t*)ctx)->uc_mcontext.fpregs->_xmm[index].element[0])
#endif
    switch (index) {
        CASE(0);
        CASE(1);
        CASE(2);
        CASE(3);
        CASE(4);
        CASE(5);
        CASE(6);
        CASE(7);
        CASE(8);
        CASE(9);
        CASE(10);
        CASE(11);
        CASE(12);
        CASE(13);
        CASE(14);
        CASE(15);
    default: {
        UNREACHABLE_MSG("Invalid XMM register index: {}", index);
        return nullptr;
    }
    }
#undef CASE
}

void* GetRip(void* ctx) {
#if defined(_WIN32)
    return (void*)((EXCEPTION_POINTERS*)ctx)->ContextRecord->Rip;
#elif defined(__APPLE__)
    return (void*)((ucontext_t*)ctx)->uc_mcontext->__ss.__rip;
#elif defined(__ANDROID__) && defined(ARCH_ARM64)
    return (void*)((ucontext_t*)ctx)->uc_mcontext.pc;
#else
    return (void*)((ucontext_t*)ctx)->uc_mcontext.gregs[REG_RIP];
#endif
}

void IncrementRip(void* ctx, u64 length) {
#if defined(_WIN32)
    ((EXCEPTION_POINTERS*)ctx)->ContextRecord->Rip += length;
#elif defined(__APPLE__)
    ((ucontext_t*)ctx)->uc_mcontext->__ss.__rip += length;
#elif defined(__ANDROID__) && defined(ARCH_ARM64)
    ((ucontext_t*)ctx)->uc_mcontext.pc += length;
#else
    ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_RIP] += length;
#endif
}

bool IsWriteError(void* ctx) {
#if defined(_WIN32)
    return ((EXCEPTION_POINTERS*)ctx)->ExceptionRecord->ExceptionInformation[0] == 1;
#elif defined(__APPLE__)
#if defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext->__es.__err & 0x2;
#elif defined(ARCH_ARM64)
    return ((ucontext_t*)ctx)->uc_mcontext->__es.__esr & 0x40;
#endif
#else
#if defined(ARCH_X86_64)
    return ((ucontext_t*)ctx)->uc_mcontext.gregs[REG_ERR] & 0x2;
#elif defined(__ANDROID__) && defined(ARCH_ARM64)
    auto* record = reinterpret_cast<_aarch64_ctx*>(
        reinterpret_cast<ucontext_t*>(ctx)->uc_mcontext.__reserved);
    while (record->magic != 0) {
        if (record->size < sizeof(_aarch64_ctx)) {
            return false;
        }
        if (record->magic == ESR_MAGIC && record->size >= sizeof(esr_context)) {
            return (reinterpret_cast<esr_context*>(record)->esr & 0x40) != 0;
        }
        record = reinterpret_cast<_aarch64_ctx*>(reinterpret_cast<u8*>(record) + record->size);
    }
    return false;
#else
#error "Unsupported architecture"
#endif
#endif
}
}
