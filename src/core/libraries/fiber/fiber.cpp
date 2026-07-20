// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "fiber.h"

#include "common/elf_info.h"
#include "common/logging/log.h"
#include "core/libraries/fiber/fiber_error.h"
#include "core/libraries/libs.h"
#include "core/tls.h"
#ifdef __ANDROID__
#include "executor/jit/lsx_translation_engine.h"
#endif

#include <cstdio>
#include <new>

namespace Libraries::Fiber {

static constexpr u32 kFiberSignature0 = 0xdef1649c;
static constexpr u32 kFiberSignature1 = 0xb37592a0;
static constexpr u32 kFiberOptSignature = 0xbb40e64d;
static constexpr u64 kFiberStackSignature = 0x7149f2ca7149f2ca;
static constexpr u64 kFiberStackSizeCheck = 0xdeadbeefdeadbeef;

static std::atomic<u32> context_size_check = false;

#ifdef __ANDROID__
namespace {

using Executor::Jit::LsxGpr;
using Executor::Jit::LsxMachineImage;
using Executor::Jit::LsxStackLease;

extern "C" int executor_live_get_current_hle_call_site(
    u64* guest_return, u64* return_off, u64* arg0, char* symbol,
    std::size_t symbol_size, char* module, std::size_t module_size);
extern "C" int executor_live_get_current_hle_guest_rsp(u64* guest_rsp);

struct AndroidFiberContinuation {
    LsxMachineImage machine{};
    u64* resume_arg_slot = nullptr;
    LsxStackLease stack_lease{};
    bool has_machine = false;
    bool owns_stack_lease = false;
};

struct AndroidFiberThreadRuntime {
    LsxMachineImage thread_machine{};
    OrbisFiberContext compatibility_context{};
    OrbisFiber* current_fiber = nullptr;
    u64* thread_return_slot = nullptr;
    bool active = false;
};

thread_local AndroidFiberThreadRuntime g_android_fiber_runtime{};

AndroidFiberContinuation* GetAndroidContinuation(OrbisFiber* fiber, const bool create) {
    if (fiber->context != nullptr) {
        return reinterpret_cast<AndroidFiberContinuation*>(fiber->context);
    }
    if (!create) {
        return nullptr;
    }
    auto* continuation = new (std::nothrow) AndroidFiberContinuation{};
    if (continuation != nullptr) {
        fiber->context = reinterpret_cast<OrbisFiberContext*>(continuation);
    }
    return continuation;
}

void DestroyAndroidContinuation(OrbisFiber* fiber) {
    auto* continuation = GetAndroidContinuation(fiber, false);
    if (continuation == nullptr) {
        return;
    }
    if (continuation->owns_stack_lease) {
        Executor::Jit::ReleaseLsxStackLease(continuation->stack_lease);
    }
    delete continuation;
    fiber->context = nullptr;
}

bool CapturePostHleContinuation(LsxMachineImage& destination, const u64 result) {
    auto* current = Executor::Jit::CurrentMachineImage();
    if (current == nullptr) {
        return false;
    }

    u64 guest_rsp = 0;
    u64 guest_return = 0;
    if (executor_live_get_current_hle_guest_rsp(&guest_rsp) != 0 || guest_rsp == 0) {
        guest_rsp = Executor::Jit::GetGpr64(*current, LsxGpr::Rsp);
    }
    (void)executor_live_get_current_hle_call_site(
        &guest_return, nullptr, nullptr, nullptr, 0, nullptr, 0);
    if (guest_return == 0 && guest_rsp != 0) {
        guest_return = Executor::Jit::ReadMemory(guest_rsp, sizeof(u64));
    }
    if (guest_rsp == 0 || guest_return == 0) {
        return false;
    }

    destination = *current;
    Executor::Jit::SetGpr64(destination, LsxGpr::Rax, result);
    Executor::Jit::SetGpr64(destination, LsxGpr::Rsp, guest_rsp + sizeof(u64));
    destination.rip_or_exit = guest_return;
    destination.direct_chain_budget = 0;
    return true;
}

bool PublishAndroidFiberState(const LsxMachineImage& source) {
    auto* current = Executor::Jit::CurrentMachineImage();
    if (current == nullptr) {
        return false;
    }
    *current = source;
    return Executor::Jit::MarkCurrentGuestStateOverride();
}

bool BuildInitialAndroidFiberState(OrbisFiber* fiber, const u64 arg_on_run_to,
                                   AndroidFiberContinuation& continuation,
                                   LsxMachineImage& destination) {
    const auto* current = Executor::Jit::CurrentMachineImage();
    if (current == nullptr || fiber->entry == nullptr) {
        return false;
    }

    // Preserve thread-local segment bases, SIMD contents and translator-private slots. Reset the
    // integer register file and architectural entry state exactly once for a new fiber.
    destination = *current;
    destination.gpr.fill(0);
    destination.rip_or_exit = reinterpret_cast<u64>(fiber->entry);
    destination.rflags = 0x202;
    destination.direct_chain_budget = 0;
    destination.x87_status_word = 0;
    destination.x87_last_opcode = 0;
    destination.x87_instruction_pointer = 0;
    destination.x87_data_pointer = 0;
    if (fiber->flags & FiberFlags::SetFpuRegs) {
        destination.x87_control_word = 0x037f;
        destination.mxcsr = 0x9fc0;
    }
    Executor::Jit::SetGpr64(destination, LsxGpr::Rdi, fiber->arg_on_initialize);
    Executor::Jit::SetGpr64(destination, LsxGpr::Rsi, arg_on_run_to);

    u64 stack_top = 0;
    if (fiber->addr_context != nullptr && fiber->size_context != 0) {
        stack_top = reinterpret_cast<u64>(fiber->addr_context) + fiber->size_context;
    } else {
        continuation.stack_lease =
            Executor::Jit::AcquireLsxStackLease(
                Executor::Jit::CurrentMachineImage());
        continuation.owns_stack_lease =
            continuation.stack_lease.current_sp != 0;
        stack_top = continuation.stack_lease.current_sp;
    }
    stack_top &= ~u64{0xf};
    if (stack_top < sizeof(u64)) {
        return false;
    }

    // A fiber entry is required to return through sceFiberReturnToThread. Keep a fail-closed zero
    // return at the top rather than borrowing the suspended caller's return chain.
    const u64 stack_pointer = stack_top - sizeof(u64);
    Executor::Jit::WriteMemory(stack_pointer, 0, sizeof(u64));
    Executor::Jit::SetGpr64(destination, LsxGpr::Rsp, stack_pointer);
    return true;
}

s32 RunAndroidFiber(OrbisFiber* fiber, const u64 arg_on_run_to, u64* arg_on_return) {
    auto& runtime = g_android_fiber_runtime;
    if (runtime.active) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    LsxMachineImage thread_continuation{};
    if (!CapturePostHleContinuation(thread_continuation, ORBIS_OK)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    auto* continuation = GetAndroidContinuation(fiber, true);
    if (continuation == nullptr) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    LsxMachineImage next{};
    if (continuation->has_machine) {
        next = continuation->machine;
        if (continuation->resume_arg_slot != nullptr) {
            *continuation->resume_arg_slot = arg_on_run_to;
        }
        Executor::Jit::SetGpr64(next, LsxGpr::Rax, ORBIS_OK);
    } else if (!BuildInitialAndroidFiberState(
                   fiber, arg_on_run_to, *continuation, next)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    runtime.thread_machine = thread_continuation;
    runtime.current_fiber = fiber;
    runtime.thread_return_slot = arg_on_return;
    runtime.active = true;
    runtime.compatibility_context.current_fiber = fiber;
    runtime.compatibility_context.prev_fiber = nullptr;
    Core::GetTcbBase()->tcb_fiber = &runtime.compatibility_context;
    continuation->has_machine = false;
    continuation->resume_arg_slot = nullptr;
    return PublishAndroidFiberState(next) ? ORBIS_OK : ORBIS_FIBER_ERROR_INVALID;
}

s32 SwitchAndroidFiber(OrbisFiber* fiber, const u64 arg_on_run_to, u64* arg_on_run) {
    auto& runtime = g_android_fiber_runtime;
    if (!runtime.active || runtime.current_fiber == nullptr) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    auto* previous_continuation =
        GetAndroidContinuation(runtime.current_fiber, true);
    auto* next_continuation = GetAndroidContinuation(fiber, true);
    if (previous_continuation == nullptr || next_continuation == nullptr) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (!CapturePostHleContinuation(previous_continuation->machine, ORBIS_OK)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    previous_continuation->resume_arg_slot = arg_on_run;
    previous_continuation->has_machine = true;

    LsxMachineImage next{};
    if (next_continuation->has_machine) {
        next = next_continuation->machine;
        if (next_continuation->resume_arg_slot != nullptr) {
            *next_continuation->resume_arg_slot = arg_on_run_to;
        }
        Executor::Jit::SetGpr64(next, LsxGpr::Rax, ORBIS_OK);
    } else if (!BuildInitialAndroidFiberState(
                   fiber, arg_on_run_to, *next_continuation, next)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    runtime.current_fiber->state = FiberState::Idle;
    runtime.current_fiber = fiber;
    runtime.compatibility_context.current_fiber = fiber;
    next_continuation->has_machine = false;
    next_continuation->resume_arg_slot = nullptr;
    return PublishAndroidFiberState(next) ? ORBIS_OK : ORBIS_FIBER_ERROR_INVALID;
}

s32 ReturnAndroidFiberToThread(const u64 arg_on_return, u64* arg_on_run) {
    auto& runtime = g_android_fiber_runtime;
    if (!runtime.active || runtime.current_fiber == nullptr) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    auto* continuation = GetAndroidContinuation(runtime.current_fiber, true);
    if (continuation == nullptr) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (!CapturePostHleContinuation(continuation->machine, ORBIS_OK)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    continuation->resume_arg_slot = arg_on_run;
    continuation->has_machine = true;

    if (runtime.thread_return_slot != nullptr) {
        *runtime.thread_return_slot = arg_on_return;
    }
    Executor::Jit::SetGpr64(runtime.thread_machine, LsxGpr::Rax, ORBIS_OK);
    runtime.current_fiber->state = FiberState::Idle;
    runtime.current_fiber = nullptr;
    runtime.thread_return_slot = nullptr;
    runtime.active = false;
    runtime.compatibility_context = {};
    Core::GetTcbBase()->tcb_fiber = nullptr;
    return PublishAndroidFiberState(runtime.thread_machine)
               ? ORBIS_OK
               : ORBIS_FIBER_ERROR_INVALID;
}

} // namespace
#endif

#ifdef __ANDROID__
namespace {

std::atomic<u32> g_android_fiber_trace_budget{96};

s32 TraceAndroidFiberResult(const char* operation, const s32 result, const void* subject,
                            const u64 detail0 = 0, const u64 detail1 = 0) {
    if (result != ORBIS_OK) {
        LOG_ERROR(Lib_Fiber,
                  "[LSX4_FIBER_API_ERROR] op={} result={:#x} subject={} detail0={:#x} "
                  "detail1={:#x}",
                  operation, static_cast<u32>(result), subject, detail0, detail1);
        if (std::FILE* trace = std::fopen(
                "/data/data/app.lsx4.android/files/lsx4-home/fiber-errors.txt", "a");
            trace != nullptr) {
            std::fprintf(trace,
                         "op=%s result=0x%08x subject=%p detail0=0x%llx detail1=0x%llx\n",
                         operation, static_cast<u32>(result), subject,
                         static_cast<unsigned long long>(detail0),
                         static_cast<unsigned long long>(detail1));
            std::fclose(trace);
        }
    }
    u32 remaining = g_android_fiber_trace_budget.load(std::memory_order_relaxed);
    while (remaining != 0 &&
           !g_android_fiber_trace_budget.compare_exchange_weak(
               remaining, remaining - 1, std::memory_order_relaxed)) {
    }
    if (remaining != 0) {
        LOG_INFO(Lib_Fiber,
                 "[LSX4_FIBER_API] op={} result={:#x} subject={} detail0={:#x} detail1={:#x}",
                 operation, static_cast<u32>(result), subject, detail0, detail1);
    }
    return result;
}

} // namespace
#endif

OrbisFiberContext* GetFiberContext() {
    return Core::GetTcbBase()->tcb_fiber;
}

extern "C" s32 PS4_SYSV_ABI _sceFiberSetJmp(OrbisFiberContext* ctx) asm("_sceFiberSetJmp");
extern "C" s32 PS4_SYSV_ABI _sceFiberLongJmp(OrbisFiberContext* ctx) asm("_sceFiberLongJmp");
extern "C" void PS4_SYSV_ABI _sceFiberSwitchEntry(OrbisFiberData* data,
                                                  bool set_fpu) asm("_sceFiberSwitchEntry");
extern "C" void PS4_SYSV_ABI _sceFiberForceQuit(u64 ret) asm("_sceFiberForceQuit");

extern "C" void PS4_SYSV_ABI _sceFiberForceQuit(u64 ret) {
    OrbisFiberContext* g_ctx = GetFiberContext();
    g_ctx->return_val = ret;
    _sceFiberLongJmp(g_ctx);
}

void PS4_SYSV_ABI _sceFiberCheckStackOverflow(OrbisFiberContext* ctx) {
    u64* stack_base = reinterpret_cast<u64*>(ctx->current_fiber->addr_context);
    u64 stack_size = ctx->current_fiber->size_context;
    if (stack_base && *stack_base != kFiberStackSignature) {
        UNREACHABLE_MSG("Stack overflow detected in fiber with size = 0x{:x}", stack_size);
    }
}

s32 PS4_SYSV_ABI _sceFiberAttachContext(OrbisFiber* fiber, void* addr_context, u64 size_context) {
    if (size_context && size_context < ORBIS_FIBER_CONTEXT_MINIMUM_SIZE) {
        return ORBIS_FIBER_ERROR_RANGE;
    }
    if (size_context & 15) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (!addr_context || !size_context) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (fiber->addr_context) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    fiber->addr_context = addr_context;
    fiber->size_context = size_context;
    fiber->context_start = addr_context;
    fiber->context_end = reinterpret_cast<u8*>(addr_context) + size_context;

    /* Apply signature to start of stack */
    *(u64*)addr_context = kFiberStackSignature;

    if (fiber->flags & FiberFlags::ContextSizeCheck) {
        u64* stack_start = reinterpret_cast<u64*>(fiber->context_start);
        u64* stack_end = reinterpret_cast<u64*>(fiber->context_end);

        u64* stack_ptr = stack_start + 1;
        while (stack_ptr < stack_end) {
            *stack_ptr++ = kFiberStackSizeCheck;
        }
    }

    return ORBIS_OK;
}

void PS4_SYSV_ABI _sceFiberSwitchToFiber(OrbisFiber* fiber, u64 arg_on_run_to,
                                         OrbisFiberContext* ctx) {
    OrbisFiberContext* fiber_ctx = fiber->context;
    if (fiber_ctx) {
        ctx->arg_on_run_to = arg_on_run_to;
        _sceFiberLongJmp(fiber_ctx);
        __builtin_trap();
    }

    OrbisFiberData data{};
    if (ctx->prev_fiber) {
        OrbisFiber* prev_fiber = ctx->prev_fiber;
        ctx->prev_fiber = nullptr;
        data.state = reinterpret_cast<u32*>(&prev_fiber->state);
    } else {
        data.state = nullptr;
    }

    data.entry = fiber->entry;
    data.arg_on_initialize = fiber->arg_on_initialize;
    data.arg_on_run_to = arg_on_run_to;
    data.stack_addr = reinterpret_cast<u8*>(fiber->addr_context) + fiber->size_context;
    if (fiber->flags & FiberFlags::SetFpuRegs) {
        data.fpucw = 0x037f;
        data.mxcsr = 0x9fc0;
        _sceFiberSwitchEntry(&data, true);
    } else {
        _sceFiberSwitchEntry(&data, false);
    }

    __builtin_trap();
}

void PS4_SYSV_ABI _sceFiberSwitch(OrbisFiber* cur_fiber, OrbisFiber* fiber, u64 arg_on_run_to,
                                  OrbisFiberContext* ctx) {
    ctx->prev_fiber = cur_fiber;
    ctx->current_fiber = fiber;

    if (fiber->addr_context == nullptr) {
        ctx->prev_fiber = nullptr;

        OrbisFiberData data{};
        data.entry = fiber->entry;
        data.arg_on_initialize = fiber->arg_on_initialize;
        data.arg_on_run_to = arg_on_run_to;
        data.stack_addr = reinterpret_cast<void*>(ctx->rsp & ~15);
        data.state = reinterpret_cast<u32*>(&cur_fiber->state);

        if (fiber->flags & FiberFlags::SetFpuRegs) {
            data.fpucw = 0x037f;
            data.mxcsr = 0x9fc0;
            _sceFiberSwitchEntry(&data, true);
        } else {
            _sceFiberSwitchEntry(&data, false);
        }

        __builtin_trap();
    }

    _sceFiberSwitchToFiber(fiber, arg_on_run_to, ctx);
    __builtin_trap();
}

void PS4_SYSV_ABI _sceFiberTerminate(OrbisFiber* fiber, u64 arg_on_return, OrbisFiberContext* ctx) {
    ctx->arg_on_return = arg_on_return;
    _sceFiberLongJmp(ctx);
    __builtin_trap();
}

s32 PS4_SYSV_ABI sceFiberInitializeImpl(OrbisFiber* fiber, const char* name, OrbisFiberEntry entry,
                                        u64 arg_on_initialize, void* addr_context, u64 size_context,
                                        const OrbisFiberOptParam* opt_param, u32 flags,
                                        u32 build_ver) {
    if (!fiber || !name || !entry) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((u64)fiber & 7 || (u64)addr_context & 15) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (opt_param && (u64)opt_param & 7) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (size_context && size_context < ORBIS_FIBER_CONTEXT_MINIMUM_SIZE) {
        return ORBIS_FIBER_ERROR_RANGE;
    }
    if (size_context & 15) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (!addr_context && size_context) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (addr_context && !size_context) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (opt_param && opt_param->magic != kFiberOptSignature) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    u32 user_flags = flags;
    if (build_ver >= Common::ElfInfo::FW_35) {
        user_flags |= FiberFlags::SetFpuRegs;
    }
    if (context_size_check) {
        user_flags |= FiberFlags::ContextSizeCheck;
    }

    strncpy(fiber->name, name, ORBIS_FIBER_MAX_NAME_LENGTH);

    fiber->entry = entry;
    fiber->arg_on_initialize = arg_on_initialize;
    fiber->addr_context = addr_context;
    fiber->size_context = size_context;
    fiber->context = nullptr;
    fiber->flags = user_flags;

    /*
        A low stack area is problematic, as we can easily
        cause a stack overflow with our HLE.
    */
    if (size_context && size_context <= 4096) {
        LOG_WARNING(Lib_Fiber, "Fiber initialized with small stack area.");
    }

    fiber->magic_start = kFiberSignature0;
    fiber->magic_end = kFiberSignature1;

    if (addr_context != nullptr) {
        fiber->context_start = addr_context;
        fiber->context_end = reinterpret_cast<u8*>(addr_context) + size_context;

        /* Apply signature to start of stack */
        *(u64*)addr_context = kFiberStackSignature;

        if (flags & FiberFlags::ContextSizeCheck) {
            u64* stack_start = reinterpret_cast<u64*>(fiber->context_start);
            u64* stack_end = reinterpret_cast<u64*>(fiber->context_end);

            u64* stack_ptr = stack_start + 1;
            while (stack_ptr < stack_end) {
                *stack_ptr++ = kFiberStackSizeCheck;
            }
        }
    }

    fiber->state = FiberState::Idle;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberOptParamInitialize(OrbisFiberOptParam* opt_param) {
    if (!opt_param) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((u64)opt_param & 7) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }

    opt_param->magic = kFiberOptSignature;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberFinalize(OrbisFiber* fiber) {
    if (!fiber) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((u64)fiber & 7) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (fiber->magic_start != kFiberSignature0 || fiber->magic_end != kFiberSignature1) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    FiberState expected = FiberState::Idle;
    if (!fiber->state.compare_exchange_strong(expected, FiberState::Terminated)) {
        return ORBIS_FIBER_ERROR_STATE;
    }

#ifdef __ANDROID__
    DestroyAndroidContinuation(fiber);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberRunImpl(OrbisFiber* fiber, void* addr_context, u64 size_context,
                                 u64 arg_on_run_to, u64* arg_on_return) {
    if (!fiber) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((u64)fiber & 7 || (u64)addr_context & 15) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (fiber->magic_start != kFiberSignature0 || fiber->magic_end != kFiberSignature1) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    Core::Tcb* tcb = Core::GetTcbBase();
    if (tcb->tcb_fiber) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    /* Caller wants to attach context and run. */
    if (addr_context != nullptr || size_context != 0) {
        s32 res = _sceFiberAttachContext(fiber, addr_context, size_context);
        if (res < 0) {
            return res;
        }
    }

    FiberState expected = FiberState::Idle;
    if (!fiber->state.compare_exchange_strong(expected, FiberState::Run)) {
        return ORBIS_FIBER_ERROR_STATE;
    }

#ifdef __ANDROID__
    const s32 android_result = RunAndroidFiber(fiber, arg_on_run_to, arg_on_return);
    if (android_result != ORBIS_OK) {
        fiber->state = FiberState::Idle;
    }
    return android_result;
#else
    OrbisFiberContext ctx{};
    ctx.current_fiber = fiber;
    ctx.prev_fiber = nullptr;
    ctx.return_val = 0;

    tcb->tcb_fiber = &ctx;

    s32 jmp = _sceFiberSetJmp(&ctx);
    if (!jmp) {
        if (fiber->addr_context) {
            _sceFiberSwitchToFiber(fiber, arg_on_run_to, &ctx);
            __builtin_trap();
        }

        OrbisFiberData data{};
        data.entry = fiber->entry;
        data.arg_on_initialize = fiber->arg_on_initialize;
        data.arg_on_run_to = arg_on_run_to;
        data.stack_addr = reinterpret_cast<void*>(ctx.rsp & ~15);
        data.state = nullptr;
        if (fiber->flags & FiberFlags::SetFpuRegs) {
            data.fpucw = 0x037f;
            data.mxcsr = 0x9fc0;
            _sceFiberSwitchEntry(&data, true);
        } else {
            _sceFiberSwitchEntry(&data, false);
        }
    }

    OrbisFiber* cur_fiber = ctx.current_fiber;
    ctx.current_fiber = nullptr;
    cur_fiber->state = FiberState::Idle;

    if (ctx.return_val != 0) {
        /* Fiber entry returned! This should never happen. */
        UNREACHABLE_MSG("Fiber entry function returned.");
    }

    if (arg_on_return) {
        *arg_on_return = ctx.arg_on_return;
    }

    tcb->tcb_fiber = nullptr;
    return ORBIS_OK;
#endif
}

s32 PS4_SYSV_ABI sceFiberSwitchImpl(OrbisFiber* fiber, void* addr_context, u64 size_context,
                                    u64 arg_on_run_to, u64* arg_on_run) {
    if (!fiber) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((u64)fiber & 7 || (u64)addr_context & 15) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (fiber->magic_start != kFiberSignature0 || fiber->magic_end != kFiberSignature1) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    OrbisFiberContext* g_ctx = GetFiberContext();
    if (!g_ctx) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    /* Caller wants to attach context and switch. */
    if (addr_context != nullptr || size_context != 0) {
        s32 res = _sceFiberAttachContext(fiber, addr_context, size_context);
        if (res < 0) {
            return res;
        }
    }

    FiberState expected = FiberState::Idle;
    if (!fiber->state.compare_exchange_strong(expected, FiberState::Run)) {
        return ORBIS_FIBER_ERROR_STATE;
    }

#ifdef __ANDROID__
    const s32 android_result = SwitchAndroidFiber(fiber, arg_on_run_to, arg_on_run);
    if (android_result != ORBIS_OK) {
        fiber->state = FiberState::Idle;
    }
    return android_result;
#else
    OrbisFiber* cur_fiber = g_ctx->current_fiber;
    if (cur_fiber->addr_context == nullptr) {
        _sceFiberSwitch(cur_fiber, fiber, arg_on_run_to, g_ctx);
        __builtin_trap();
    }

    OrbisFiberContext ctx{};
    s32 jmp = _sceFiberSetJmp(&ctx);
    if (!jmp) {
        cur_fiber->context = &ctx;
        _sceFiberCheckStackOverflow(g_ctx);
        _sceFiberSwitch(cur_fiber, fiber, arg_on_run_to, g_ctx);
        __builtin_trap();
    }

    g_ctx = GetFiberContext();
    if (g_ctx->prev_fiber) {
        g_ctx->prev_fiber->state = FiberState::Idle;
        g_ctx->prev_fiber = nullptr;
    }

    if (arg_on_run) {
        *arg_on_run = g_ctx->arg_on_run_to;
    }

    return ORBIS_OK;
#endif
}

s32 PS4_SYSV_ABI sceFiberGetSelf(OrbisFiber** fiber) {
    if (!fiber) {
        return ORBIS_FIBER_ERROR_NULL;
    }

#ifdef __ANDROID__
    if (!g_android_fiber_runtime.active ||
        g_android_fiber_runtime.current_fiber == nullptr) {
        return TraceAndroidFiberResult(
            "GetSelf", ORBIS_FIBER_ERROR_PERMISSION, nullptr);
    }
    *fiber = g_android_fiber_runtime.current_fiber;
    return TraceAndroidFiberResult("GetSelf", ORBIS_OK, *fiber);
#else
    OrbisFiberContext* g_ctx = GetFiberContext();
    if (!g_ctx) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    *fiber = g_ctx->current_fiber;
    return ORBIS_OK;
#endif
}

s32 PS4_SYSV_ABI sceFiberReturnToThread(u64 arg_on_return, u64* arg_on_run) {
#ifdef __ANDROID__
    const s32 result = ReturnAndroidFiberToThread(arg_on_return, arg_on_run);
    return TraceAndroidFiberResult(
        "ReturnToThread", result, g_android_fiber_runtime.current_fiber,
        arg_on_return, reinterpret_cast<u64>(arg_on_run));
#else
    OrbisFiberContext* g_ctx = GetFiberContext();
    if (!g_ctx) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    OrbisFiber* cur_fiber = g_ctx->current_fiber;
    if (cur_fiber->addr_context) {
        OrbisFiberContext ctx{};
        s32 jmp = _sceFiberSetJmp(&ctx);
        if (jmp) {
            g_ctx = GetFiberContext();
            if (g_ctx->prev_fiber) {
                g_ctx->prev_fiber->state = FiberState::Idle;
                g_ctx->prev_fiber = nullptr;
            }
            if (arg_on_run) {
                *arg_on_run = g_ctx->arg_on_run_to;
            }
            return ORBIS_OK;
        }

        cur_fiber->context = &ctx;
        _sceFiberCheckStackOverflow(g_ctx);
    }

    _sceFiberTerminate(cur_fiber, arg_on_return, g_ctx);
    __builtin_trap();
#endif
}

s32 PS4_SYSV_ABI sceFiberGetInfo(OrbisFiber* fiber, OrbisFiberInfo* fiber_info) {
    if (!fiber || !fiber_info) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((u64)fiber & 7 || (u64)fiber_info & 7) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (fiber_info->size != sizeof(OrbisFiberInfo)) {
        return ORBIS_FIBER_ERROR_INVALID;
    }
    if (fiber->magic_start != kFiberSignature0 || fiber->magic_end != kFiberSignature1) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    fiber_info->entry = fiber->entry;
    fiber_info->arg_on_initialize = fiber->arg_on_initialize;
    fiber_info->addr_context = fiber->addr_context;
    fiber_info->size_context = fiber->size_context;
    strncpy(fiber_info->name, fiber->name, ORBIS_FIBER_MAX_NAME_LENGTH);

    fiber_info->size_context_margin = -1;
    if (fiber->flags & FiberFlags::ContextSizeCheck && fiber->addr_context != nullptr) {
        u64 stack_margin = 0;
        u64* stack_start = reinterpret_cast<u64*>(fiber->context_start);
        u64* stack_end = reinterpret_cast<u64*>(fiber->context_end);

        if (*stack_start == kFiberStackSignature) {
            u64* stack_ptr = stack_start + 1;
            while (stack_ptr < stack_end) {
                if (*stack_ptr == kFiberStackSizeCheck) {
                    stack_ptr++;
                }
            }

            stack_margin =
                reinterpret_cast<u64>(stack_ptr) - reinterpret_cast<u64>(stack_start + 1);
        }

        fiber_info->size_context_margin = stack_margin;
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberStartContextSizeCheck(u32 flags) {
    if (flags != 0) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    u32 expected = 0;
    if (!context_size_check.compare_exchange_strong(expected, 1u)) {
        return ORBIS_FIBER_ERROR_STATE;
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberStopContextSizeCheck() {
    u32 expected = 1;
    if (!context_size_check.compare_exchange_strong(expected, 0u)) {
        return ORBIS_FIBER_ERROR_STATE;
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberRename(OrbisFiber* fiber, const char* name) {
    if (!fiber || !name) {
        return ORBIS_FIBER_ERROR_NULL;
    }
    if ((u64)fiber & 7) {
        return ORBIS_FIBER_ERROR_ALIGNMENT;
    }
    if (fiber->magic_start != kFiberSignature0 || fiber->magic_end != kFiberSignature1) {
        return ORBIS_FIBER_ERROR_INVALID;
    }

    strncpy(fiber->name, name, ORBIS_FIBER_MAX_NAME_LENGTH);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceFiberGetThreadFramePointerAddress(u64* addr_frame_pointer) {
    if (!addr_frame_pointer) {
        return ORBIS_FIBER_ERROR_NULL;
    }

#ifdef __ANDROID__
    if (!g_android_fiber_runtime.active) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }
    *addr_frame_pointer = Executor::Jit::GetGpr64(
        g_android_fiber_runtime.thread_machine, LsxGpr::Rbp);
    return ORBIS_OK;
#else
    OrbisFiberContext* g_ctx = GetFiberContext();
    if (!g_ctx) {
        return ORBIS_FIBER_ERROR_PERMISSION;
    }

    *addr_frame_pointer = g_ctx->rbp;
    return ORBIS_OK;
#endif
}

s32 PS4_SYSV_ABI sceFiberInitialize(OrbisFiber* fiber, const char* name, OrbisFiberEntry entry,
                                    u64 arg_on_initialize, void* addr_context, u64 size_context,
                                    const OrbisFiberOptParam* opt_param, u32 build_ver) {
    const s32 result =
        sceFiberInitializeImpl(fiber, name, entry, arg_on_initialize, addr_context, size_context,
                               opt_param, 0, build_ver);
#ifdef __ANDROID__
    return TraceAndroidFiberResult("Initialize", result, fiber, size_context, build_ver);
#else
    return result;
#endif
}

s32 PS4_SYSV_ABI sceFiberRun(OrbisFiber* fiber, u64 arg_on_run_to, u64* arg_on_return) {
    const s32 result = sceFiberRunImpl(fiber, nullptr, 0, arg_on_run_to, arg_on_return);
#ifdef __ANDROID__
    return TraceAndroidFiberResult("Run", result, fiber, arg_on_run_to,
                                   reinterpret_cast<u64>(arg_on_return));
#else
    return result;
#endif
}

s32 PS4_SYSV_ABI sceFiberSwitch(OrbisFiber* fiber, u64 arg_on_run_to, u64* arg_on_run) {
    const s32 result = sceFiberSwitchImpl(fiber, nullptr, 0, arg_on_run_to, arg_on_run);
#ifdef __ANDROID__
    return TraceAndroidFiberResult("Switch", result, fiber, arg_on_run_to,
                                   reinterpret_cast<u64>(arg_on_run));
#else
    return result;
#endif
}

#ifdef __ANDROID__
s32 PS4_SYSV_ABI TraceFiberInitializeImpl(OrbisFiber* fiber, const char* name,
                                          OrbisFiberEntry entry, u64 arg_on_initialize,
                                          void* addr_context, u64 size_context,
                                          const OrbisFiberOptParam* opt_param, u32 flags,
                                          u32 build_ver) {
    const s32 result =
        sceFiberInitializeImpl(fiber, name, entry, arg_on_initialize, addr_context, size_context,
                               opt_param, flags, build_ver);
    return TraceAndroidFiberResult("InitializeImpl", result, fiber, size_context, build_ver);
}

s32 PS4_SYSV_ABI TraceFiberRunImpl(OrbisFiber* fiber, void* addr_context, u64 size_context,
                                   u64 arg_on_run_to, u64* arg_on_return) {
    const s32 result =
        sceFiberRunImpl(fiber, addr_context, size_context, arg_on_run_to, arg_on_return);
    return TraceAndroidFiberResult("RunImpl", result, fiber, size_context, arg_on_run_to);
}

s32 PS4_SYSV_ABI TraceFiberSwitchImpl(OrbisFiber* fiber, void* addr_context, u64 size_context,
                                      u64 arg_on_run_to, u64* arg_on_run) {
    const s32 result =
        sceFiberSwitchImpl(fiber, addr_context, size_context, arg_on_run_to, arg_on_run);
    return TraceAndroidFiberResult("SwitchImpl", result, fiber, size_context, arg_on_run_to);
}
#endif

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("hVYD7Ou2pCQ", "libSceFiber", 1, "libSceFiber", sceFiberInitialize);
#ifdef __ANDROID__
    LIB_FUNCTION("7+OJIpko9RY", "libSceFiber", 1, "libSceFiber",
                 TraceFiberInitializeImpl); // _sceFiberInitializeWithInternalOptionImpl
#else
    LIB_FUNCTION("7+OJIpko9RY", "libSceFiber", 1, "libSceFiber",
                 sceFiberInitializeImpl); // _sceFiberInitializeWithInternalOptionImpl
#endif
    LIB_FUNCTION("asjUJJ+aa8s", "libSceFiber", 1, "libSceFiber", sceFiberOptParamInitialize);
    LIB_FUNCTION("JeNX5F-NzQU", "libSceFiber", 1, "libSceFiber", sceFiberFinalize);

    LIB_FUNCTION("a0LLrZWac0M", "libSceFiber", 1, "libSceFiber", sceFiberRun);
    LIB_FUNCTION("PFT2S-tJ7Uk", "libSceFiber", 1, "libSceFiber", sceFiberSwitch);
    LIB_FUNCTION("p+zLIOg27zU", "libSceFiber", 1, "libSceFiber", sceFiberGetSelf);
    LIB_FUNCTION("B0ZX2hx9DMw", "libSceFiber", 1, "libSceFiber", sceFiberReturnToThread);

#ifdef __ANDROID__
    LIB_FUNCTION("avfGJ94g36Q", "libSceFiber", 1, "libSceFiber",
                 TraceFiberRunImpl); // _sceFiberAttachContextAndRun
    LIB_FUNCTION("ZqhZFuzKT6U", "libSceFiber", 1, "libSceFiber",
                 TraceFiberSwitchImpl); // _sceFiberAttachContextAndSwitch
#else
    LIB_FUNCTION("avfGJ94g36Q", "libSceFiber", 1, "libSceFiber",
                 sceFiberRunImpl); // _sceFiberAttachContextAndRun
    LIB_FUNCTION("ZqhZFuzKT6U", "libSceFiber", 1, "libSceFiber",
                 sceFiberSwitchImpl); // _sceFiberAttachContextAndSwitch
#endif

    LIB_FUNCTION("uq2Y5BFz0PE", "libSceFiber", 1, "libSceFiber", sceFiberGetInfo);
    LIB_FUNCTION("Lcqty+QNWFc", "libSceFiber", 1, "libSceFiber", sceFiberStartContextSizeCheck);
    LIB_FUNCTION("Kj4nXMpnM8Y", "libSceFiber", 1, "libSceFiber", sceFiberStopContextSizeCheck);
    LIB_FUNCTION("JzyT91ucGDc", "libSceFiber", 1, "libSceFiber", sceFiberRename);

    LIB_FUNCTION("0dy4JtMUcMQ", "libSceFiber", 1, "libSceFiber",
                 sceFiberGetThreadFramePointerAddress);
}

} // namespace Libraries::Fiber
