// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/signal_context.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/signals.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#ifdef __ANDROID__
#include <android/log.h>
#include <fcntl.h>
#include <unistd.h>
#ifdef __ANDROID__
#include <dlfcn.h>
#include <unwind.h>
#endif
#include <atomic>
#include <chrono>
#include <thread>
#endif
#ifdef ARCH_X86_64
#include "common/x86_decoder.h"
#endif
#endif

#ifndef _WIN32
namespace Libraries::Kernel {
void SigactionHandler(int native_signum, siginfo_t* inf, ucontext_t* raw_context);
} // namespace Libraries::Kernel
#endif

namespace Core {

#if defined(_WIN32)

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    const auto* signals = Signals::Instance();

    bool handled = false;
    switch (pExp->ExceptionRecord->ExceptionCode) {
    case EXCEPTION_ACCESS_VIOLATION:
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions.
        return EXCEPTION_CONTINUE_EXECUTION;
    default:
        break;
    }

    return handled ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

#else

#ifdef __ANDROID__
static struct sigaction g_previous_sigsegv {};
static struct sigaction g_previous_sigbus {};
static struct sigaction g_previous_sigill {};
static struct sigaction g_previous_sigurg {};
static struct sigaction g_previous_sigvtalrm {};
static bool g_fault_signal_handlers_installed = false;
static bool g_jit_signal_handlers_installed = false;

extern "C" int executor_lsx4_android_runtime_jit_active() __attribute__((weak));
extern "C" int executor_jit_has_synchronous_guest_fault_frame() __attribute__((weak));
extern "C" int executor_jit_defer_synchronous_guest_fault(
    s32 native_sig, s32 si_code, s32 si_errno, s32 source_pid, u32 source_uid, u64 fault_addr,
    s32 is_write) __attribute__((weak));
extern "C" u64 executor_jit_current_fault_guest_rip() __attribute__((weak));
extern "C" u64 executor_jit_current_fault_active_block_rip() __attribute__((weak));
extern "C" u64 executor_jit_current_fault_instruction() __attribute__((weak));
extern "C" u32 executor_jit_current_fault_instruction_meta() __attribute__((weak));
extern "C" u64 executor_jit_current_fault_gpr(u32 index) __attribute__((weak));
extern "C" u32 executor_jit_current_fault_recent_rips(u64* out, u32 capacity)
    __attribute__((weak));
extern "C" u32 executor_jit_current_fault_active_block_bytes(u8* out, u32 capacity)
    __attribute__((weak));
extern "C" u32 executor_jit_current_fault_recent_block(u32 newest_index, u64* rip_out,
                                                               u8* bytes_out, u32 capacity)
    __attribute__((weak));

static bool AndroidJitActive() {
    return executor_lsx4_android_runtime_jit_active != nullptr &&
           executor_lsx4_android_runtime_jit_active() != 0;
}

static bool JitSynchronousGuestFaultCanResume() {
    return executor_jit_has_synchronous_guest_fault_frame != nullptr &&
           executor_jit_has_synchronous_guest_fault_frame() != 0;
}

static bool DeferJitSynchronousGuestFault(int sig, const siginfo_t* info,
                                               bool is_write) {
    if (executor_jit_defer_synchronous_guest_fault == nullptr) {
        return false;
    }
    // This is the only JIT action permitted while the POSIX signal frame is live.  The
    // callee copies these scalars into the armed frame and siglongjmps out of emitted AArch64.
    // Guest code, logging, allocation and ucontext writeback all happen later in Execute().
    return executor_jit_defer_synchronous_guest_fault(
               sig, info ? info->si_code : 0, info ? info->si_errno : 0,
               info ? info->si_pid : 0, info ? info->si_uid : 0,
               reinterpret_cast<u64>(info ? info->si_addr : nullptr), is_write ? 1 : 0) != 0;
}

static struct sigaction& PreviousActionForSignal(int sig) {
    switch (sig) {
    case SIGSEGV:
        return g_previous_sigsegv;
    case SIGBUS:
        return g_previous_sigbus;
    case SIGILL:
        return g_previous_sigill;
    default:
        return g_previous_sigsegv;
    }
}

static bool AddressIsInNamedMap(void* address, const char* needle) {
    if (!address || !needle) {
        return false;
    }
    const auto target = reinterpret_cast<uintptr_t>(address);
    FILE* maps = std::fopen("/proc/self/maps", "re");
    if (!maps) {
        return false;
    }
    char line[512];
    while (std::fgets(line, sizeof(line), maps)) {
        unsigned long long start = 0;
        unsigned long long end = 0;
        if (std::sscanf(line, "%llx-%llx", &start, &end) == 2 && target >= start &&
            target < end && std::strstr(line, needle)) {
            std::fclose(maps);
            return true;
        }
    }
    std::fclose(maps);
    return false;
}

static bool IsGuestNullCheckFault(int sig, const siginfo_t* info) {
    if (sig != SIGSEGV || info == nullptr) {
        return false;
    }
    const auto fault = reinterpret_cast<std::uintptr_t>(info->si_addr);
    return info->si_code == SEGV_MAPERR && fault < 0x10000;
}

static bool ChainPreviousSignalHandler(int sig, siginfo_t* info, void* raw_context,
                                       const char* reason) {
    auto& previous = PreviousActionForSignal(sig);
    if ((previous.sa_flags & SA_SIGINFO) && previous.sa_sigaction &&
        previous.sa_sigaction != SignalHandler) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_FEX_SIGNAL_CHAIN] sig=%d reason=%s pc=%p fault=%p "
                            "prev_sigaction=%p flags=0x%x",
                            sig, reason ? reason : "<none>", Common::GetRip(raw_context),
                            info ? info->si_addr : nullptr,
                            reinterpret_cast<void*>(previous.sa_sigaction), previous.sa_flags);
        previous.sa_sigaction(sig, info, raw_context);
        return true;
    }
    if (previous.sa_handler && previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN &&
        reinterpret_cast<void*>(previous.sa_handler) != reinterpret_cast<void*>(SignalHandler)) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_FEX_SIGNAL_CHAIN] sig=%d reason=%s pc=%p fault=%p "
                            "prev_handler=%p flags=0x%x",
                            sig, reason ? reason : "<none>", Common::GetRip(raw_context),
                            info ? info->si_addr : nullptr,
                            reinterpret_cast<void*>(previous.sa_handler), previous.sa_flags);
        previous.sa_handler(sig);
        return true;
    }
    return false;
}
#endif

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    X86DecodedInstruction instruction;
    X86DecodedOperand operands[X86_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (X86_SUCCESS(status)) {
        X86Formatter formatter;
        X86FormatterInit(&formatter, X86_FORMATTER_STYLE_INTEL);
        X86FormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), nullptr);
    }
#endif

    return buffer;
}

#ifdef __ANDROID__
// FEX emulates x86 Total-Store-Order with AArch64 STLR/LDAR (store-release / load-acquire), which
// REQUIRE natural alignment and raise SIGBUS(BUS_ADRALN) on a misaligned guest access — routine in
// x86 code that stores unaligned. In split signal mode FEX nominally owns SIGBUS, but the embed
// build does not backpatch these, so the faulting guest thread (e.g. UnityGfxDeviceWorker) dies and
// the Unity render handshake deadlocks (Game:Main waits forever on the dead worker's semaphore post).
// Emulate the faulting release/acquire access here (plain load/store bracketed by a full barrier)
// and step past it so the guest thread survives.
static bool TryFixupMisalignedReleaseAcquire(int sig, siginfo_t* info, void* raw_context) {
    if (sig != SIGBUS || raw_context == nullptr || info == nullptr || info->si_code != BUS_ADRALN) {
        return false;
    }
    auto& mc = reinterpret_cast<ucontext_t*>(raw_context)->uc_mcontext;
    const auto insn = *reinterpret_cast<const std::uint32_t*>(mc.pc);
    const unsigned rt = insn & 0x1fu;
    const unsigned rn = (insn >> 5) & 0x1fu;
    const std::uint64_t addr = (rn == 31) ? mc.sp : mc.regs[rn];
    const auto load_reg = [&](unsigned i) -> std::uint64_t { return i == 31 ? 0 : mc.regs[i]; };
    const auto store_reg = [&](unsigned i, std::uint64_t v) { if (i != 31) mc.regs[i] = v; };
    std::atomic_thread_fence(std::memory_order_seq_cst);
    bool handled = true;
    switch (insn & 0xFFFFFC00u) {
    case 0xC89FFC00u: { const std::uint64_t v = load_reg(rt);
        std::memcpy(reinterpret_cast<void*>(addr), &v, 8); break; }                 // STLR  Xt,[Xn]
    case 0x889FFC00u: { const std::uint32_t v = static_cast<std::uint32_t>(load_reg(rt));
        std::memcpy(reinterpret_cast<void*>(addr), &v, 4); break; }                 // STLR  Wt,[Xn]
    case 0x489FFC00u: { const std::uint16_t v = static_cast<std::uint16_t>(load_reg(rt));
        std::memcpy(reinterpret_cast<void*>(addr), &v, 2); break; }                 // STLRH Wt,[Xn]
    case 0xC8DFFC00u: { std::uint64_t v; std::memcpy(&v, reinterpret_cast<const void*>(addr), 8);
        store_reg(rt, v); break; }                                                  // LDAR  Xt,[Xn]
    case 0x88DFFC00u: { std::uint32_t v; std::memcpy(&v, reinterpret_cast<const void*>(addr), 4);
        store_reg(rt, static_cast<std::uint64_t>(v)); break; }                      // LDAR  Wt,[Xn]
    case 0x48DFFC00u: { std::uint16_t v; std::memcpy(&v, reinterpret_cast<const void*>(addr), 2);
        store_reg(rt, static_cast<std::uint64_t>(v)); break; }                      // LDARH Wt,[Xn]
    case 0xF8BFC000u: { std::uint64_t v; std::memcpy(&v, reinterpret_cast<const void*>(addr), 8);
        store_reg(rt, v); break; }                                                  // LDAPR  Xt,[Xn]
    case 0xB8BFC000u: { std::uint32_t v; std::memcpy(&v, reinterpret_cast<const void*>(addr), 4);
        store_reg(rt, static_cast<std::uint64_t>(v)); break; }                      // LDAPR  Wt,[Xn]
    case 0x78BFC000u: { std::uint16_t v; std::memcpy(&v, reinterpret_cast<const void*>(addr), 2);
        store_reg(rt, static_cast<std::uint64_t>(v)); break; }                      // LDAPRH Wt,[Xn]
    default: handled = false; break;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!handled) {
        return false;
    }
    mc.pc += 4;
    static std::atomic<int> total{0};
    const int n = total.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 64) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_FEX_UNALIGN_FIXUP] n=%d insn=0x%08x addr=0x%llx rt=%u rn=%u",
                            n, insn, static_cast<unsigned long long>(addr), rt, rn);
    }
    // Persistent, logcat-eviction-proof witness (async-signal-safe open/write/close): first few and
    // then every 4096th fixup, so a later run can confirm the handler is actually rescuing STLR/LDAR.
    if (n <= 8 || (n & 0xfff) == 0) {
        char line[128];
        const int len = std::snprintf(line, sizeof(line),
                                      "UNALIGN_FIXUP n=%d insn=0x%08x addr=0x%llx\n", n, insn,
                                      static_cast<unsigned long long>(addr));
        const int fd = ::open("/data/data/app.lsx4.android/files/executor-unalign.log",
                              O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (fd >= 0) {
            (void)::write(fd, line, static_cast<size_t>(len > 0 ? len : 0));
            ::close(fd);
        }
    }
    return true;
}
#endif

#ifdef __ANDROID__
// Survival escape hatch: a JIT reconstruction gap in the RSDK audio worker (Thread2) computes
// a corrupted store target (e.g. a host-lib address 0x77xx) and the native JIT block's STR SIGSEGVs,
// taking the WHOLE process down (killing the render main thread too). When
// EXECUTOR_SKIP_FAULTING_STORES is set, discard a faulting guest STORE (the value is lost but the
// offending thread survives, exactly as a real guest fault would be handled) and step past the
// single 4-byte AArch64 store instruction so the renderer can be observed. Only WRITE faults are
// skipped; reads/execs still crash. Not the default; opt-in for bring-up.
static bool TrySkipFaultingGuestStore(int sig, siginfo_t* info, void* raw_context) {
    if (sig != SIGSEGV || raw_context == nullptr) {
        return false;
    }
    static const bool enabled = [] {
        const char* v = std::getenv("EXECUTOR_SKIP_FAULTING_STORES");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    if (!enabled || !Common::IsWriteError(raw_context)) {
        return false;
    }
    auto& mc = reinterpret_cast<ucontext_t*>(raw_context)->uc_mcontext;
    const auto insn = *reinterpret_cast<const std::uint32_t*>(mc.pc);
    // Confirm the faulting instruction is a general STR/STUR/STRB/STRH/STP form (top bits 0b*0111000
    // unscaled/register/immediate stores, or 0b*01010010 STP) before stepping over it, so we never
    // silently skip something that is not a store.
    const bool is_str = ((insn & 0x3B000000u) == 0x38000000u) ||  // STR/STUR/STRB/STRH (imm/reg/unscaled)
                        ((insn & 0x3BC00000u) == 0x29000000u) ||  // STP (signed offset)
                        ((insn & 0x3BC00000u) == 0x28800000u) ||  // STP (post-index)
                        ((insn & 0x3BC00000u) == 0x29800000u);    // STP (pre-index)
    if (!is_str) {
        return false;
    }
    mc.pc += 4;
    static std::atomic<int> total{0};
    const int n = total.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n <= 128) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_SKIP_FAULTING_STORE] n=%d insn=0x%08x faultAddr=%p pc=0x%llx",
                            n, insn, info ? info->si_addr : nullptr,
                            static_cast<unsigned long long>(mc.pc - 4));
    }
    return true;
}
#endif

#ifdef __ANDROID__
// tid of the guest render/present thread (the one driving sceGnmSubmitAndFlip). Set by the GNM
// driver so the fault handler can tell the frame-producing thread apart from worker threads.
std::atomic<int> g_executor_render_tid{0};
} // namespace Core (reopened below)

// boost::container::static_vector / small_vector bounds preconditions use BOOST_ASSERT, a no-op under
// NDEBUG (silent UB on overflow). BOOST_ENABLE_ASSERT_HANDLER (defined for the GPU-path TUs) routes
// them here. The Sonic JIT corruption is a suspected static_vector overflow scribbling a
// std::mutex; log the EXACT expr/file:line to the persistent checkfail file so it can be pinned, then
// return (do not abort) so we can also see whether the render survives.
namespace boost {
void assertion_failed(char const* expr, char const* function, char const* file, long line) {
    static std::atomic<int> budget{64};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_BOOST_ASSERT] expr='%s' fn='%s' at %s:%ld", expr ? expr : "?",
                        function ? function : "?", file ? file : "?", line);
    char buf[512];
    const int n = std::snprintf(buf, sizeof(buf), "BOOST_ASSERT expr='%s' fn='%s' %s:%ld\n",
                                expr ? expr : "?", function ? function : "?", file ? file : "?", line);
    const int fd = ::open("/data/data/app.lsx4.android/files/lsx4-home/jit-checkfail.txt",
                          O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd >= 0) {
        (void)::write(fd, buf, static_cast<size_t>(n > 0 ? n : 0));
        ::close(fd);
    }
}
void assertion_failed_msg(char const* expr, char const* msg, char const* function, char const* file,
                          long line) {
    assertion_failed(expr, function, file, line);
    (void)msg;
}
} // namespace boost

namespace Core {

// Capture the calling thread's backtrace (async-signal-safe-ish via _Unwind_Backtrace) to the
// persistent checkfail file on an abort. With libc++ hardening on, a silent std::vector::operator[]
// OOB aborts AT THE WRITE SITE (before it can corrupt a mutex), so this backtrace pins the exact
// offending frame — the Sonic JIT host-memory corruptor. Offsets are resolved later with
// llvm-addr2line against the full-probe .so.
struct BtState {
    void** pcs;
    int count;
    int cap;
};
static _Unwind_Reason_Code BtTrace(struct _Unwind_Context* ctx, void* arg) {
    auto* st = static_cast<BtState*>(arg);
    if (st->count >= st->cap) {
        return _URC_END_OF_STACK;
    }
    const uintptr_t ip = _Unwind_GetIP(ctx);
    if (ip != 0) {
        st->pcs[st->count++] = reinterpret_cast<void*>(ip);
    }
    return _URC_NO_REASON;
}
static void CaptureAbortBacktrace(int sig) {
    static std::atomic<int> budget{8};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    void* pcs[64];
    BtState st{pcs, 0, 64};
    _Unwind_Backtrace(&BtTrace, &st);
    Dl_info info{};
    char line[256];
    const int fd = ::open("/data/data/app.lsx4.android/files/lsx4-home/jit-checkfail.txt",
                          O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd < 0) {
        return;
    }
    int n = std::snprintf(line, sizeof(line), "ABORT_BACKTRACE sig=%d tid=%d frames=%d\n", sig,
                          static_cast<int>(gettid()), st.count);
    (void)::write(fd, line, static_cast<size_t>(n > 0 ? n : 0));
    for (int i = 0; i < st.count; ++i) {
        const char* fname = "?";
        uintptr_t off = 0;
        if (dladdr(pcs[i], &info) && info.dli_fname) {
            fname = info.dli_fname;
            off = reinterpret_cast<uintptr_t>(pcs[i]) - reinterpret_cast<uintptr_t>(info.dli_fbase);
        }
        n = std::snprintf(line, sizeof(line), "  #%02d pc=%p %s+0x%lx\n", i, pcs[i], fname,
                          static_cast<unsigned long>(off));
        (void)::write(fd, line, static_cast<size_t>(n > 0 ? n : 0));
    }
    ::close(fd);
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_ABORT_BACKTRACE] sig=%d tid=%d frames=%d (see jit-checkfail.txt)",
                        sig, static_cast<int>(gettid()), st.count);
}

// Last-resort survival: a native JIT codegen bug on a WORKER thread (e.g. the RSDK audio
// worker) corrupts host memory and faults, taking the whole process — and the render thread that is
// mid-present — down with it. When EXECUTOR_PARK_NONRENDER_ON_FAULT is set and an unrecoverable
// fault lands on a non-render, non-main thread, park that thread forever instead of aborting the
// process, so the render thread can finish presenting the frame. Opt-in bring-up aid, not default.
static bool TryParkNonRenderThreadOnFault(int sig, void* code_address, siginfo_t* info) {
    static const bool enabled = [] {
        const char* v = std::getenv("EXECUTOR_PARK_NONRENDER_ON_FAULT");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    if (!enabled) {
        return false;
    }
    static const bool park_all = [] {
        const char* v = std::getenv("EXECUTOR_PARK_ALL_ON_FAULT");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    const int tid = static_cast<int>(gettid());
    const int render_tid = g_executor_render_tid.load(std::memory_order_relaxed);
    if (tid == getpid()) {
        return false; // never park the Android main/UI thread
    }
    if (!park_all && render_tid != 0 && tid == render_tid) {
        return false; // default: let the render thread crash for diagnosis
    }
    // EXECUTOR_PARK_ALL_ON_FAULT: park EVERY non-UI thread (including the render thread) on an
    // unrecoverable fault, so a FORTIFY on a corrupted mutex in the sceGnmSubmitDone/present tail does
    // not kill the process — the independent VideoOut present-pump thread can still present the flip
    // the game already queued (submitAndFlip buf 0), and the intact rendered swapchain frame displays.
    static std::atomic<int> parked{0};
    const int n = parked.fetch_add(1, std::memory_order_relaxed) + 1;
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_PARK_NONRENDER_ON_FAULT] n=%d sig=%d tid=%d renderTid=%d pc=%p "
                        "fault=%p (parking worker thread so render can present)",
                        n, sig, tid, render_tid, code_address, info ? info->si_addr : nullptr);
    for (;;) {
        struct timespec ts{3600, 0};
        nanosleep(&ts, nullptr);
    }
}
#endif

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    const auto* signals = Signals::Instance();
#ifdef __ANDROID__
    const bool jit_parity = AndroidJitActive();
    if (!jit_parity) {
        if (TryFixupMisalignedReleaseAcquire(sig, info, raw_context)) {
            return;
        }
        if (TrySkipFaultingGuestStore(sig, info, raw_context)) {
            return;
        }
    }
    // On any abort, capture the backtrace to the checkfail file: with libc++ hardening a silent
    // std::vector::operator[] OOB aborts at the write site, pinning the corruptor.
    if (sig == SIGABRT) {
        CaptureAbortBacktrace(sig);
    }
    // A FORTIFY abort (SIGABRT) from a JIT worker's corrupted-mutex lock would take the whole
    // process down; park the worker instead so the render thread survives to present.
    if (sig == SIGABRT && TryParkNonRenderThreadOnFault(sig, Common::GetRip(raw_context), info)) {
        return;
    }
#endif

    auto* code_address = Common::GetRip(raw_context);

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
#ifdef __ANDROID__
        if (jit_parity && JitSynchronousGuestFaultCanResume()) {
            // HLE memory tracking gets first refusal.  Every remaining fault raised by an armed
            // JIT Arm64BlockEntry belongs to guest execution, including an unhandled guest
            // fault.  Defer both cases: a registered guest handler resumes normally, while a
            // missing handler becomes a controlled mapped-entry error in Execute().  Falling
            // through to UNREACHABLE here used to turn an ordinary unhandled guest stack fault
            // into a process-wide host SIGTRAP.
            if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
                (void)DeferJitSynchronousGuestFault(sig, info, is_write);
                UNREACHABLE_MSG(
                    "JIT synchronous guest access fault could not be deferred");
            }
            break;
        }
        // Persistent, signal-safe fault capture (logcat evicts; the boundary json isn't written on a
        // crash). si_addr + rip + si_code tell us whether the Mono-init SIGSEGV is a DELIBERATE null-check
        // (si_addr null/small, rip in guest/AOT code, code=SEGV_MAPERR) that should be routed to Mono's
        // guest handler, or a REAL bad deref. guest_handler = Mono's registered SIGSEGV handler.
        {
            static std::atomic<int> budget{12};
            if (budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                const auto gh = Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)];
                const u64 guest_rip = executor_jit_current_fault_guest_rip != nullptr
                                          ? executor_jit_current_fault_guest_rip()
                                          : 0;
                const u64 active_rip =
                    executor_jit_current_fault_active_block_rip != nullptr
                        ? executor_jit_current_fault_active_block_rip()
                        : 0;
                const u64 instruction = executor_jit_current_fault_instruction != nullptr
                                            ? executor_jit_current_fault_instruction()
                                            : 0;
                const u32 instruction_meta =
                    executor_jit_current_fault_instruction_meta != nullptr
                        ? executor_jit_current_fault_instruction_meta()
                        : 0;
                const auto fault_gpr = [](const u32 index) -> u64 {
                    return executor_jit_current_fault_gpr != nullptr
                               ? executor_jit_current_fault_gpr(index)
                               : 0;
                };
                u64 recent_rips[16]{};
                const u32 recent_count = executor_jit_current_fault_recent_rips != nullptr
                                             ? executor_jit_current_fault_recent_rips(
                                                   recent_rips, 16u)
                                             : 0;
                u8 active_bytes[64]{};
                const u32 active_bytes_count =
                    executor_jit_current_fault_active_block_bytes != nullptr
                        ? executor_jit_current_fault_active_block_bytes(
                              active_bytes, 64u)
                        : 0;
                char line[768];
                int n = std::snprintf(
                    line, sizeof(line),
                    "SEGV sig=%d code=%d si_addr=%p host_rip=%p is_write=%d guest_handler=%p "
                    "guest_rip=0x%llx active_rip=0x%llx mnemonic=%u length=%u bytes_le=0x%llx "
                    "rax=0x%llx rcx=0x%llx rdx=0x%llx rbx=0x%llx rsp=0x%llx rbp=0x%llx "
                    "rsi=0x%llx rdi=0x%llx recent=",
                    sig, info ? info->si_code : -1, info ? info->si_addr : nullptr,
                    code_address, is_write ? 1 : 0, reinterpret_cast<void*>(gh),
                    static_cast<unsigned long long>(guest_rip),
                    static_cast<unsigned long long>(active_rip), instruction_meta & 0x00ffffffu,
                    instruction_meta >> 24u, static_cast<unsigned long long>(instruction),
                    // LsxMachineImage::gpr uses the physical-state order
                    // RAX,RBX,RCX,RDX,RSI,RDI,RBP,RSP rather than the x86
                    // encoding order used by the human-readable line above.
                    static_cast<unsigned long long>(fault_gpr(0)),
                    static_cast<unsigned long long>(fault_gpr(2)),
                    static_cast<unsigned long long>(fault_gpr(3)),
                    static_cast<unsigned long long>(fault_gpr(1)),
                    static_cast<unsigned long long>(fault_gpr(7)),
                    static_cast<unsigned long long>(fault_gpr(6)),
                    static_cast<unsigned long long>(fault_gpr(4)),
                    static_cast<unsigned long long>(fault_gpr(5)));
                for (u32 index = 0; index < recent_count && n > 0 &&
                                    static_cast<size_t>(n) < sizeof(line); ++index) {
                    n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n),
                                       "%s%llx", index == 0 ? "" : ",",
                                       static_cast<unsigned long long>(recent_rips[index]));
                }
                if (n > 0 && static_cast<size_t>(n) < sizeof(line)) {
                    n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n),
                                       " blockBytes=");
                }
                for (u32 index = 0; index < active_bytes_count && n > 0 &&
                                    static_cast<size_t>(n) + 3 < sizeof(line); ++index) {
                    n += std::snprintf(line + n, sizeof(line) - static_cast<size_t>(n), "%02x",
                                       active_bytes[index]);
                }
                if (n > 0 && static_cast<size_t>(n) + 2 < sizeof(line)) {
                    line[n++] = '\n';
                    line[n] = '\0';
                }
                int fd = ::open("/data/data/app.lsx4.android/files/executor-segv.log",
                                O_WRONLY | O_CREAT | O_APPEND, 0600);
                if (fd >= 0) {
                    (void)::write(fd, line, static_cast<size_t>(n > 0 ? n : 0));
                    if (executor_jit_current_fault_recent_block != nullptr) {
                        for (u32 recent_index = 0; recent_index < 12; ++recent_index) {
                            u64 recent_rip = 0;
                            u8 recent_bytes[128]{};
                            const u32 recent_size = executor_jit_current_fault_recent_block(
                                recent_index, &recent_rip, recent_bytes, 128u);
                            if (recent_size == 0) {
                                break;
                            }
                            char recent_line[320];
                            int recent_n = std::snprintf(
                                recent_line, sizeof(recent_line),
                                "RECENT_BLOCK newest=%u rip=0x%llx bytes=", recent_index,
                                static_cast<unsigned long long>(recent_rip));
                            for (u32 byte_index = 0; byte_index < recent_size && recent_n > 0 &&
                                                   static_cast<size_t>(recent_n) + 3 <
                                                       sizeof(recent_line);
                                 ++byte_index) {
                                recent_n += std::snprintf(
                                    recent_line + recent_n,
                                    sizeof(recent_line) - static_cast<size_t>(recent_n), "%02x",
                                    recent_bytes[byte_index]);
                            }
                            if (recent_n > 0 &&
                                static_cast<size_t>(recent_n) + 2 < sizeof(recent_line)) {
                                recent_line[recent_n++] = '\n';
                            }
                            (void)::write(fd, recent_line,
                                          static_cast<size_t>(recent_n > 0 ? recent_n : 0));
                        }
                    }
                    ::close(fd);
                }
            }
        }
#endif
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
#ifdef __ANDROID__
            if (jit_parity) {
                const s32 guest_sig = Libraries::Kernel::NativeToOrbisSignal(sig);
                if (Libraries::Kernel::HasSignalHandler(guest_sig) &&
                    JitSynchronousGuestFaultCanResume()) {
                    (void)DeferJitSynchronousGuestFault(sig, info, is_write);
                    UNREACHABLE_MSG(
                        "JIT synchronous guest access fault could not be deferred");
                }
                // A process-wide SIGSEGV/SIGBUS owner also observes faults from ART, libc and
                // vendor libraries.  They are host faults whenever JIT has no armed
                // Arm64BlockEntry frame; let the pre-existing Android handler finish its probe.
                if (!JitSynchronousGuestFaultCanResume() &&
                    ChainPreviousSignalHandler(sig, info, raw_context,
                                               "jit_host_access_no_resume")) {
                    return;
                }
                UNREACHABLE_MSG("Unhandled JIT access violation at code address {}: {} address {}",
                                fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                                fmt::ptr(info->si_addr));
            }
#endif
#ifdef __ANDROID__
            if (IsGuestNullCheckFault(sig, info) &&
                Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                static std::atomic<int> log_budget{64};
                if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                        "[EXECUTOR_FEX_SIGNAL_ROUTE] sig=%d reason=guest_null_check "
                                        "pc=%p fault=%p",
                                        sig, code_address, info ? info->si_addr : nullptr);
                }
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
#endif
#ifdef __ANDROID__
            if (AddressIsInNamedMap(code_address, "FEXMem") &&
                ChainPreviousSignalHandler(sig, info, raw_context, "fexmem_access")) {
                return;
            }
            // Before handing an unrecoverable worker-thread fault to FEX/default (which aborts the
            // whole process), park it so the render thread can still present its frame.
            if (TryParkNonRenderThreadOnFault(sig, code_address, info)) {
                return;
            }
            if (ChainPreviousSignalHandler(sig, info, raw_context, "android_unhandled_access")) {
                return;
            }
#endif
            // If the guest has installed a custom signal handler, and the access violation didn't
            // come from HLE memory tracking, pass the signal on
            if (Libraries::Kernel::HasSignalHandler(
                    Libraries::Kernel::NativeToOrbisSignal(sig))) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
        }
        break;
    }
    case SIGILL:
        if (!signals->DispatchIllegalInstruction(raw_context)) {
#ifdef __ANDROID__
            if (jit_parity) {
                const s32 guest_sig = Libraries::Kernel::NativeToOrbisSignal(sig);
                if (Libraries::Kernel::HasSignalHandler(guest_sig) &&
                    JitSynchronousGuestFaultCanResume()) {
                    (void)DeferJitSynchronousGuestFault(sig, info, false);
                    UNREACHABLE_MSG(
                        "JIT synchronous guest illegal instruction could not be deferred");
                }
                // Android and native libraries legitimately use SIGILL for host feature probes.
                // JIT owns SIGILL process-wide, but those probes are not guest faults when
                // no Arm64BlockEntry resume frame is armed.  Preserve the handler that was installed
                // before JIT instead of routing a host probe through the PS4 SIGILL table.
                if (!JitSynchronousGuestFaultCanResume() &&
                    ChainPreviousSignalHandler(sig, info, raw_context,
                                               "jit_host_ill_no_resume")) {
                    return;
                }
                UNREACHABLE_MSG("Unhandled JIT illegal instruction at code address {}: {}",
                                fmt::ptr(code_address), DisassembleInstruction(code_address));
            }
            if (AddressIsInNamedMap(code_address, "FEXMem") &&
                ChainPreviousSignalHandler(sig, info, raw_context, "fexmem_ill")) {
                return;
            }
            if (ChainPreviousSignalHandler(sig, info, raw_context, "android_unhandled_ill")) {
                return;
            }
#endif
            if (Libraries::Kernel::HasSignalHandler(
                    Libraries::Kernel::NativeToOrbisSignal(sig))) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled illegal instruction at code address {}: {}",
                            fmt::ptr(code_address), DisassembleInstruction(code_address));
        }
        break;
    default:
        if (sig == SIGSLEEP) {
            // Sleep thread until signal is received again
            sigset_t sigset;
            sigemptyset(&sigset);
            sigaddset(&sigset, SIGSLEEP);
            sigwait(&sigset, &sig);
        }
        break;
    }
}

#endif

#ifdef __ANDROID__
// Re-assert shadps4 ownership of SIGBUS AFTER the embedded FEX installs its own (crash-only) SIGBUS
// handler when it creates the guest CPU context — that races ahead of SignalDispatch's ctor, so the
// ctor-time install gets clobbered. Called from the guest-entry scope during boot; once re-owned it
// stays (FEX installs its host handler once). Lets TryFixupMisalignedReleaseAcquire rescue FEX's
// unaligned STLR/LDAR TSO stores instead of the guest render thread dying with SIGBUS(BUS_ADRALN).
static void ReassertBusHandlerOnce() {
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    struct sigaction prev{};
    if (sigaction(SIGBUS, &action, &prev) == 0 && prev.sa_sigaction != SignalHandler) {
        g_previous_sigbus = prev;
        static std::atomic<int> budget{8};
        if (budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_FEX_UNALIGN_REINSTALL] re-owned SIGBUS prev=%p",
                                reinterpret_cast<void*>(prev.sa_sigaction));
        }
    }
}

// A low-frequency watchdog re-asserts our SIGBUS handler every 100ms for the first 60s. FEX installs
// its own host SIGBUS handler when it builds the guest CPU context (and possibly again per guest
// thread), racing ahead of our ctor-time install; a periodic re-assert reclaims it cheaply (~10
// sigaction/s) without the per-guest-entry syscall storm that stalled boot.
extern "C" void executor_lsx4_reinstall_bus_handler() {
    static std::once_flag once;
    std::call_once(once, [] {
        ReassertBusHandlerOnce();
        std::thread([] {
            for (int i = 0; i < 600; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                ReassertBusHandlerOnce();
            }
        }).detach();
    });
}

extern "C" void executor_lsx4_install_jit_signal_handlers() {
    if (!AndroidJitActive() || g_jit_signal_handlers_installed) {
        return;
    }
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    ASSERT_MSG(sigaction(SIGSEGV, &action, &g_previous_sigsegv) == 0,
               "Failed to register JIT SIGSEGV handler.");
    ASSERT_MSG(sigaction(SIGBUS, &action, &g_previous_sigbus) == 0,
               "Failed to register JIT SIGBUS handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, &g_previous_sigill) == 0,
               "Failed to register JIT SIGILL handler.");
    ASSERT_MSG(sigaction(SIGURG, &action, &g_previous_sigurg) == 0,
               "Failed to register JIT SIGURG handler.");
    ASSERT_MSG(sigaction(SIGVTALRM, &action, &g_previous_sigvtalrm) == 0,
               "Failed to register JIT SIGVTALRM handler.");
    g_fault_signal_handlers_installed = true;
    g_jit_signal_handlers_installed = true;
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_JIT_SIGNAL_OWNER] "
                        "signals=SIGSEGV,SIGBUS,SIGILL,SIGURG,SIGVTALRM");
}
#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

#ifdef __ANDROID__
    const bool jit_active = AndroidJitActive();
    if (jit_active) {
        executor_lsx4_install_jit_signal_handlers();
    } else if (std::getenv("EXECUTOR_FEX_OWNS_FAULT_SIGNALS") != nullptr) {
        g_fault_signal_handlers_installed = false;
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_FEX_SIGNAL_MODE] owner=fex "
                            "skip_shadps4_fault_handlers=1");
    } else {
        ASSERT_MSG(sigaction(SIGSEGV, &action, &g_previous_sigsegv) == 0,
                   "Failed to register access violation signal handler.");
        // Own SIGBUS too so TryFixupMisalignedReleaseAcquire can rescue FEX's unaligned STLR/LDAR
        // TSO stores (BUS_ADRALN); non-alignment SIGBUS chains to the saved previous handler.
        ASSERT_MSG(sigaction(SIGBUS, &action, &g_previous_sigbus) == 0,
                   "Failed to register bus error signal handler.");
        // Own SIGABRT only when the worker-park survival aid is enabled, so a FORTIFY abort (a
        // corrupted-mutex pthread_mutex_lock from a JIT worker) parks that worker instead of
        // taking the whole process — and the render thread — down.
        if (const char* pv = std::getenv("EXECUTOR_PARK_NONRENDER_ON_FAULT");
            pv != nullptr && pv[0] != '\0' && std::strcmp(pv, "0") != 0) {
            struct sigaction abrt_action{};
            abrt_action.sa_sigaction = SignalHandler;
            abrt_action.sa_flags = SA_SIGINFO | SA_ONSTACK;
            sigemptyset(&abrt_action.sa_mask);
            sigaction(SIGABRT, &abrt_action, nullptr);
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_PARK_NONRENDER_ON_FAULT] owns SIGABRT=1");
        }
        g_fault_signal_handlers_installed = true;
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_FEX_SIGNAL_MODE] owner=%s "
                            "shadps4=SIGSEGV,SIGBUS fex=SIGILL jit=%d",
                            jit_active ? "jit" : "split",
                            jit_active ? 1 : 0);
    }
#else
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to register illegal instruction signal handler.");
#endif
#ifdef __ANDROID__
    if (!jit_active) {
        ASSERT_MSG(sigaction(SIGSLEEP, &action, nullptr) == 0,
                   "Failed to register sleep signal handler.");
    }
#else
    ASSERT_MSG(sigaction(SIGSLEEP, &action, nullptr) == 0,
               "Failed to register sleep signal handler.");
#endif
#endif
}

SignalDispatch::~SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(RemoveVectoredExceptionHandler(handle), "Failed to remove exception handler.");
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

#ifdef __ANDROID__
    if (g_fault_signal_handlers_installed) {
        ASSERT_MSG(sigaction(SIGSEGV, &g_previous_sigsegv, nullptr) == 0,
                   "Failed to restore access violation signal handler.");
        if (g_jit_signal_handlers_installed) {
            ASSERT_MSG(sigaction(SIGBUS, &g_previous_sigbus, nullptr) == 0,
                       "Failed to restore bus signal handler.");
            ASSERT_MSG(sigaction(SIGILL, &g_previous_sigill, nullptr) == 0,
                       "Failed to restore illegal-instruction signal handler.");
            ASSERT_MSG(sigaction(SIGURG, &g_previous_sigurg, nullptr) == 0,
                       "Failed to restore urgent signal handler.");
            ASSERT_MSG(sigaction(SIGVTALRM, &g_previous_sigvtalrm, nullptr) == 0,
                       "Failed to restore virtual-alarm signal handler.");
            g_jit_signal_handlers_installed = false;
        }
        g_fault_signal_handlers_installed = false;
    }
#else
    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to remove access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to remove illegal instruction signal handler.");
#endif
#endif
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core
