// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/threads/thread_state.h"
#include "core/libraries/libs.h"
#include "core/signals.h"
#ifdef __ANDROID__
#include "executor/jit/lsx_translation_engine.h"
#endif

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <sys/syscall.h>

extern "C" int executor_lsx4_android_run_guest_signal_handler(std::uint64_t handler,
                                                                  std::uint64_t arg0,
                                                                  std::uint64_t arg1,
                                                                  std::uint64_t arg2,
                                                                  std::uint64_t* guest_result)
    __attribute__((weak));
extern "C" int executor_lsx4_android_get_current_guest_regs(void* out, std::size_t out_size)
    __attribute__((weak));
extern "C" int executor_lsx4_android_set_current_guest_regs(const void* regs,
                                                               std::size_t regs_size)
    __attribute__((weak));
extern "C" void executor_lsx4_android_publish_guest_gc_register_roots(
    const void* regs, std::size_t regs_size) __attribute__((weak));
extern "C" int executor_lsx4_android_register_guest_signal_handler(int signum,
                                                                       std::uint64_t handler,
                                                                       int flags)
    __attribute__((weak));
extern "C" int executor_lsx4_android_queue_guest_signal(std::uintptr_t pthread_handle,
                                                           int signum)
    __attribute__((weak));
extern "C" int executor_lsx4_android_queue_guest_signal_for_thread(
    void* guest_thread, std::uintptr_t pthread_handle, int signum) __attribute__((weak));
extern "C" int executor_lsx4_android_queue_guest_signal_for_thread_context(
    void* guest_thread, std::uintptr_t pthread_handle, int signum, std::uint64_t rip,
    std::uint64_t rsp) __attribute__((weak));
extern "C" void executor_live_mono_mark_pending_sigusr1_for_thread(void* guest_thread)
    __attribute__((weak));
extern "C" int executor_live_mono_synthetic_suspend_ack(void* guest_thread, const char* reason)
    __attribute__((weak));
extern "C" bool executor_lsx4_android_jit_active() __attribute__((weak));

struct ExecutorBox64GuestRegs {
    std::uint64_t regs[16];
    std::uint64_t rip;
    std::uint64_t old_ip;
    std::uint64_t fsbase;
    std::uint64_t gsbase;
    int quit;
    int exit;
    int error;
};

static struct sigaction g_executor_mono_sigsegv_previous {};
static std::atomic<int> g_executor_mono_sigsegv_installed{0};
static thread_local ExecutorBox64GuestRegs g_executor_raise_regs {};
static thread_local bool g_executor_have_raise_regs = false;

static void RecordJitSignalDispatchFailure(const char* stage, s32 orbis_sig,
                                                s32 native_sig,
                                                Libraries::Kernel::OrbisKernelExceptionHandler handler,
                                                int rc,
                                                void* fault_addr) {
    char line[512];
    const int length = std::snprintf(
        line, sizeof(line),
        "stage=%s orbis=%d native=%d handler=%p rc=%d fault=%p kind=%u\n",
        stage ? stage : "unknown", orbis_sig, native_sig,
        reinterpret_cast<void*>(handler), rc, fault_addr,
        orbis_sig >= 0 && static_cast<std::size_t>(orbis_sig) <
                                  Libraries::Kernel::HandlerKinds.size()
            ? Libraries::Kernel::HandlerKinds[orbis_sig]
            : 0);
    const int fd = ::open(
        "/data/data/app.lsx4.android/files/executor-jit-signal-failure.log",
        O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
    if (fd >= 0) {
        if (length > 0) {
            const std::size_t write_length =
                static_cast<std::size_t>(length) < sizeof(line)
                    ? static_cast<std::size_t>(length)
                    : sizeof(line) - 1;
            (void)::write(fd, line, write_length);
        }
        ::close(fd);
    }
}

static bool ExecutorJitSignalParityActive() {
    return executor_lsx4_android_jit_active != nullptr &&
           executor_lsx4_android_jit_active();
}

static bool ExecutorIsAsyncDeliveredSegv(const siginfo_t* info) {
    if (!info) {
        return false;
    }
    if (info->si_code == SI_USER) {
        return true;
    }
#ifdef SI_TKILL
    if (info->si_code == SI_TKILL) {
        return true;
    }
#endif
    return false;
}

static bool ExecutorIsNamedMonoThread(const Libraries::Kernel::PthreadT thread) {
    if (!thread) {
        return false;
    }
    return thread->name.rfind("mono thread", 0) == 0 || thread->name == "UnityPreload";
}

static bool ExecutorIsHleUnsafeMonoThread(const Libraries::Kernel::PthreadT thread) {
    return thread && (thread->name.rfind("mono thread", 0) == 0 || thread->name == "UnityPreload");
}

static void ExecutorMonoExplicitSigsegvHandler(int signum, siginfo_t* info, void* raw_context) {
    if (signum == SIGSEGV && ExecutorIsAsyncDeliveredSegv(info)) {
#if defined(__aarch64__)
        void* pc = raw_context ? reinterpret_cast<void*>(
                                     static_cast<ucontext_t*>(raw_context)->uc_mcontext.pc)
                               : nullptr;
#else
        void* pc = nullptr;
#endif
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_EXPLICIT_SIGSEGV_IGNORED] code=%d pid=%d uid=%d "
                            "tid=%ld pc=%p",
                            info ? info->si_code : 0, info ? info->si_pid : 0,
                            info ? info->si_uid : 0, static_cast<long>(syscall(SYS_gettid)), pc);
        return;
    }

    const auto previous = g_executor_mono_sigsegv_previous;
    if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction) {
        previous.sa_sigaction(signum, info, raw_context);
        return;
    }
    if (previous.sa_handler && previous.sa_handler != SIG_IGN && previous.sa_handler != SIG_DFL) {
        previous.sa_handler(signum);
        return;
    }

    signal(signum, SIG_DFL);
    raise(signum);
}

extern "C" int executor_lsx4_android_install_mono_explicit_sigsegv_guard() {
    struct sigaction mono_native_act {};
    mono_native_act.sa_sigaction = ExecutorMonoExplicitSigsegvHandler;
    mono_native_act.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&mono_native_act.sa_mask);
    struct sigaction old_native_act {};
    const int rc = sigaction(SIGSEGV, &mono_native_act, &old_native_act);
    if (rc == 0) {
        const bool old_is_self = (old_native_act.sa_flags & SA_SIGINFO) != 0 &&
                                 old_native_act.sa_sigaction ==
                                     ExecutorMonoExplicitSigsegvHandler;
        if (!old_is_self) {
            g_executor_mono_sigsegv_previous = old_native_act;
        }
        g_executor_mono_sigsegv_installed.store(1);
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_EXPLICIT_SIGSEGV_GUARD] rc=%d installed=%d",
                        rc, g_executor_mono_sigsegv_installed.load());
    return rc;
}

#endif

#ifdef _WIN64
#include "common/ntapi.h"
#else
#include <csignal>
#endif
#include <unordered_set>

namespace Libraries::Kernel {

#ifdef __ANDROID__
bool ExecutorSignalKernelSemaByNameForLiveMono(const char* name, s32 signal_count);
extern "C" int executor_live_signal_pending_mono_posix_sems(int max_posts)
    __attribute__((weak));
#endif

#ifndef __ANDROID__
static bool PcOracleMonoSyncEnabled() {
    const char* value = std::getenv("EXECUTOR_PC_ORACLE_MONO_SYNC");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void PcOracleSignalLog(const char* phase, Pthread* thread, s32 sig, int native_signum,
                              const void* handler, int rc, int err) {
    if (!PcOracleMonoSyncEnabled() || sig != POSIX_SIGUSR1) {
        return;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_PC_SIGUSR1] phase=%s thread=%p name=%s sig=%d native=%d "
                 "handler=%p rc=%d errno=%d\n",
                 phase, thread, thread ? thread->name.c_str() : "<null>", sig, native_signum,
                 handler, rc, err);
    std::fflush(stderr);
}
#endif

#ifdef _WIN32

// Windows doesn't have native versions of these, and we don't need to use them either.
s32 NativeToOrbisSignal(s32 s) {
    return s;
}

s32 OrbisToNativeSignal(s32 s) {
    return s;
}

#else

s32 NativeToOrbisSignal(s32 s) {
    switch (s) {
    case SIGHUP:
        return POSIX_SIGHUP;
    case SIGINT:
        return POSIX_SIGINT;
    case SIGQUIT:
        return POSIX_SIGQUIT;
    case SIGILL:
        return POSIX_SIGILL;
    case SIGTRAP:
        return POSIX_SIGTRAP;
    case SIGABRT:
        return POSIX_SIGABRT;
    case SIGFPE:
        return POSIX_SIGFPE;
    case SIGKILL:
        return POSIX_SIGKILL;
    case SIGBUS:
        return POSIX_SIGBUS;
    case SIGSEGV:
        return POSIX_SIGSEGV;
    case SIGSYS:
        return POSIX_SIGSYS;
    case SIGPIPE:
        return POSIX_SIGPIPE;
    case SIGALRM:
        return POSIX_SIGALRM;
    case SIGTERM:
        return POSIX_SIGTERM;
    case SIGURG:
        return POSIX_SIGURG;
    case SIGSTOP:
        return POSIX_SIGSTOP;
    case SIGTSTP:
        return POSIX_SIGTSTP;
    case SIGCONT:
        return POSIX_SIGCONT;
    case SIGCHLD:
        return POSIX_SIGCHLD;
    case SIGTTIN:
        return POSIX_SIGTTIN;
    case SIGTTOU:
        return POSIX_SIGTTOU;
    case SIGIO:
        return POSIX_SIGIO;
    case SIGXCPU:
        return POSIX_SIGXCPU;
    case SIGXFSZ:
        return POSIX_SIGXFSZ;
    case SIGVTALRM:
        return POSIX_SIGVTALRM;
    case SIGPROF:
        return POSIX_SIGPROF;
    case SIGWINCH:
        return POSIX_SIGWINCH;
    case SIGUSR1:
        return POSIX_SIGUSR1;
    case SIGUSR2:
        return POSIX_SIGUSR2;
    case _SIGEMT:
        return POSIX_SIGEMT;
    case _SIGINFO:
        return POSIX_SIGINFO;
    case 0:
        return 128;
    default:
        if (s > 0 && s < 128) {
            return s;
        }
        UNREACHABLE_MSG("Unknown signal {}", s);
    }
}

s32 OrbisToNativeSignal(s32 s) {
    switch (s) {
    case POSIX_SIGHUP:
        return SIGHUP;
    case POSIX_SIGINT:
        return SIGINT;
    case POSIX_SIGQUIT:
        return SIGQUIT;
    case POSIX_SIGILL:
        return SIGILL;
    case POSIX_SIGTRAP:
        return SIGTRAP;
    case POSIX_SIGABRT:
        return SIGABRT;
    case POSIX_SIGEMT:
        return _SIGEMT;
    case POSIX_SIGFPE:
        return SIGFPE;
    case POSIX_SIGKILL:
        return SIGKILL;
    case POSIX_SIGBUS:
        return SIGBUS;
    case POSIX_SIGSEGV:
        return SIGSEGV;
    case POSIX_SIGSYS:
        return SIGSYS;
    case POSIX_SIGPIPE:
        return SIGPIPE;
    case POSIX_SIGALRM:
        return SIGALRM;
    case POSIX_SIGTERM:
        return SIGTERM;
    case POSIX_SIGURG:
        return SIGURG;
    case POSIX_SIGSTOP:
        return SIGSTOP;
    case POSIX_SIGTSTP:
        return SIGTSTP;
    case POSIX_SIGCONT:
        return SIGCONT;
    case POSIX_SIGCHLD:
        return SIGCHLD;
    case POSIX_SIGTTIN:
        return SIGTTIN;
    case POSIX_SIGTTOU:
        return SIGTTOU;
    case POSIX_SIGIO:
        return SIGIO;
    case POSIX_SIGXCPU:
        return SIGXCPU;
    case POSIX_SIGXFSZ:
        return SIGXFSZ;
    case POSIX_SIGVTALRM:
        return SIGVTALRM;
    case POSIX_SIGPROF:
        return SIGPROF;
    case POSIX_SIGWINCH:
        return SIGWINCH;
    case POSIX_SIGINFO:
        return _SIGINFO;
    case POSIX_SIGUSR1:
        return SIGUSR1;
    case POSIX_SIGUSR2:
        return SIGUSR2;
    case 128:
        return 0;
    default:
        if (s > 0 && s < 128) {
            return s;
        }
        UNREACHABLE_MSG("Unknown signal {}", s);
    }
}

#endif

#ifdef __APPLE__
#define sigisemptyset(x) (*(x) == 0)
#elif !defined(_WIN32)
static bool sigisemptyset(const sigset_t* set) {
    for (int signum = 1; signum < NSIG; signum++) {
        if (sigismember(set, signum) == 1) {
            return false;
        }
    }
    return true;
}
#endif

std::array<OrbisKernelExceptionHandler, 130> Handlers{};
std::array<int, 130> HandlerFlags{};
// Reference ABI: 0=none, 1=handler(sig), 2=handler(sig, siginfo, ucontext),
// 3=kernel exception handler(sig, ucontext).
std::array<std::uint8_t, 130> HandlerKinds{};
Sigset g_sigintr{};

#ifdef __ANDROID__
static const char* CurrentThreadNameForSignalTrace() {
    return g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>";
}

static long CurrentKernelTidForSignalTrace() {
#ifdef SYS_gettid
    return static_cast<long>(syscall(SYS_gettid));
#else
    return -1;
#endif
}

// Read-only, explicitly armed observer for the isolated JIT signal
// delivery regression.  Keeping every field atomic makes the signal-entry
// write independent from libc locks and lets the harness take a stable
// snapshot after the handler returns without changing dispatch semantics.
struct ExecutorJitSignalProbeSnapshot {
    std::uint64_t sequence;
    std::int64_t linux_tid;
    std::int32_t orbis_signal;
    std::int32_t native_signal;
    std::uint8_t handler_kind;
    std::uint8_t reserved[7];
    std::uintptr_t guest_state;
    std::uint64_t guest_rip;
    std::uintptr_t host_pc;
};

static std::atomic_bool g_jit_signal_probe_armed{false};
static std::atomic<std::uint64_t> g_jit_signal_probe_sequence{0};
static std::atomic<std::int64_t> g_jit_signal_probe_linux_tid{-1};
static std::atomic<std::int32_t> g_jit_signal_probe_orbis_signal{0};
static std::atomic<std::int32_t> g_jit_signal_probe_native_signal{0};
static std::atomic<std::uint8_t> g_jit_signal_probe_handler_kind{0};
static std::atomic<std::uintptr_t> g_jit_signal_probe_guest_state{0};
static std::atomic<std::uint64_t> g_jit_signal_probe_guest_rip{0};
static std::atomic<std::uintptr_t> g_jit_signal_probe_host_pc{0};

static void RecordJitSignalProbeEntry(const s32 orbis_signal,
                                           const s32 native_signal,
                                           const ucontext_t* raw_context) {
    if (!g_jit_signal_probe_armed.load(std::memory_order_acquire) ||
        !ExecutorJitSignalParityActive()) {
        return;
    }
    auto* state = Executor::Jit::CurrentMachineImage();
    const auto next_sequence =
        g_jit_signal_probe_sequence.load(std::memory_order_relaxed) + 1;
    g_jit_signal_probe_linux_tid.store(CurrentKernelTidForSignalTrace(),
                                             std::memory_order_relaxed);
    g_jit_signal_probe_orbis_signal.store(orbis_signal, std::memory_order_relaxed);
    g_jit_signal_probe_native_signal.store(native_signal, std::memory_order_relaxed);
    g_jit_signal_probe_handler_kind.store(
        orbis_signal >= 0 && static_cast<std::size_t>(orbis_signal) < HandlerKinds.size()
            ? HandlerKinds[orbis_signal]
            : std::uint8_t{0},
        std::memory_order_relaxed);
    g_jit_signal_probe_guest_state.store(reinterpret_cast<std::uintptr_t>(state),
                                               std::memory_order_relaxed);
    g_jit_signal_probe_guest_rip.store(state != nullptr ? state->rip_or_exit : 0,
                                             std::memory_order_relaxed);
#if defined(__aarch64__)
    g_jit_signal_probe_host_pc.store(
        raw_context != nullptr
            ? static_cast<std::uintptr_t>(raw_context->uc_mcontext.pc)
            : 0,
        std::memory_order_relaxed);
#else
    g_jit_signal_probe_host_pc.store(0, std::memory_order_relaxed);
#endif
    g_jit_signal_probe_sequence.store(next_sequence, std::memory_order_release);
}

extern "C" __attribute__((visibility("default"), used)) int
executor_jit_signal_probe_arm(const int armed) {
    g_jit_signal_probe_armed.store(false, std::memory_order_release);
    g_jit_signal_probe_sequence.store(0, std::memory_order_relaxed);
    g_jit_signal_probe_linux_tid.store(-1, std::memory_order_relaxed);
    g_jit_signal_probe_orbis_signal.store(0, std::memory_order_relaxed);
    g_jit_signal_probe_native_signal.store(0, std::memory_order_relaxed);
    g_jit_signal_probe_handler_kind.store(0, std::memory_order_relaxed);
    g_jit_signal_probe_guest_state.store(0, std::memory_order_relaxed);
    g_jit_signal_probe_guest_rip.store(0, std::memory_order_relaxed);
    g_jit_signal_probe_host_pc.store(0, std::memory_order_relaxed);
    if (armed == 0) {
        return 0;
    }
    if (!ExecutorJitSignalParityActive()) {
        return -1;
    }
    g_jit_signal_probe_armed.store(true, std::memory_order_release);
    return 0;
}

extern "C" __attribute__((visibility("default"), used)) int
executor_jit_signal_probe_snapshot(void* out, const std::size_t out_size) {
    if (out == nullptr || out_size != sizeof(ExecutorJitSignalProbeSnapshot)) {
        return -1;
    }
    ExecutorJitSignalProbeSnapshot snapshot{};
    snapshot.sequence = g_jit_signal_probe_sequence.load(std::memory_order_acquire);
    snapshot.linux_tid = g_jit_signal_probe_linux_tid.load(std::memory_order_relaxed);
    snapshot.orbis_signal =
        g_jit_signal_probe_orbis_signal.load(std::memory_order_relaxed);
    snapshot.native_signal =
        g_jit_signal_probe_native_signal.load(std::memory_order_relaxed);
    snapshot.handler_kind =
        g_jit_signal_probe_handler_kind.load(std::memory_order_relaxed);
    snapshot.guest_state =
        g_jit_signal_probe_guest_state.load(std::memory_order_relaxed);
    snapshot.guest_rip =
        g_jit_signal_probe_guest_rip.load(std::memory_order_relaxed);
    snapshot.host_pc =
        g_jit_signal_probe_host_pc.load(std::memory_order_relaxed);
    std::memcpy(out, &snapshot, sizeof(snapshot));
    return 0;
}

static void TraceLiveSignal(const char* stage, s32 orbis_sig, s32 native_sig,
                            OrbisKernelExceptionHandler handler, PthreadT target = nullptr,
                            int rc = 0, int err = 0, void* addr = nullptr) {
    // Signal delivery is a hot GC path. Keep a small proof window: synchronous logcat for every
    // stop-the-world stage delays the threads this signal is meant to suspend and resume.
    static std::atomic_int budget{32};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    const auto* guest_state = Executor::Jit::CurrentMachineImage();
    const std::uint64_t guest_rip = guest_state != nullptr ? guest_state->rip_or_exit : 0;
    const std::uint64_t guest_rsp = guest_state != nullptr
                                        ? Executor::Jit::GetGpr64(
                                              *guest_state, Executor::Jit::LsxGpr::Rsp)
                                        : 0;
    const auto guest_gpr = [guest_state](const Executor::Jit::LsxGpr reg) {
        return guest_state != nullptr ? Executor::Jit::GetGpr64(*guest_state, reg) : 0;
    };
    const std::uint64_t guest_rdi = guest_gpr(Executor::Jit::LsxGpr::Rdi);
    const std::uint64_t guest_rsi = guest_gpr(Executor::Jit::LsxGpr::Rsi);
    const std::uint64_t guest_rbp = guest_gpr(Executor::Jit::LsxGpr::Rbp);
    const std::uint64_t guest_rax = guest_gpr(Executor::Jit::LsxGpr::Rax);
    const std::uint64_t guest_r12 = guest_gpr(Executor::Jit::LsxGpr::R12);
    const std::uint64_t guest_r13 = guest_gpr(Executor::Jit::LsxGpr::R13);
    const std::uint64_t guest_r14 = guest_gpr(Executor::Jit::LsxGpr::R14);
    const std::uint64_t guest_r15 = guest_gpr(Executor::Jit::LsxGpr::R15);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_SIGNAL] stage=%s curThread=%s tid=%ld target=%p "
                        "targetName=%s targetNative=0x%zx sig=%d native=%d handler=%p "
                        "addr=%p rc=%d errno=%d guestRip=0x%llx guestRsp=0x%llx "
                        "guestRdi=0x%llx guestRsi=0x%llx guestRbp=0x%llx "
                        "guestRax=0x%llx guestR12=0x%llx guestR13=0x%llx "
                        "guestR14=0x%llx guestR15=0x%llx",
                        stage, CurrentThreadNameForSignalTrace(), CurrentKernelTidForSignalTrace(),
                        target, target ? target->name.c_str() : "<none>",
                        target ? static_cast<size_t>(target->native_thr.GetHandle()) : 0,
                        orbis_sig, native_sig, reinterpret_cast<void*>(handler), addr, rc, err,
                        static_cast<unsigned long long>(guest_rip),
                        static_cast<unsigned long long>(guest_rsp),
                        static_cast<unsigned long long>(guest_rdi),
                        static_cast<unsigned long long>(guest_rsi),
                        static_cast<unsigned long long>(guest_rbp),
                        static_cast<unsigned long long>(guest_rax),
                        static_cast<unsigned long long>(guest_r12),
                        static_cast<unsigned long long>(guest_r13),
                        static_cast<unsigned long long>(guest_r14),
                        static_cast<unsigned long long>(guest_r15));
}

static void TraceLightOraclePthreadKill(const char* stage, PthreadT target, s32 orbis_sig,
                                        s32 native_sig, int rc = 0, int err = 0) {
    if (std::getenv("EXECUTOR_LIGHT_ORACLE") == nullptr || orbis_sig != POSIX_SIGUSR1) {
        return;
    }
    static std::atomic_int budget{512};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    ExecutorBox64GuestRegs regs{};
    const bool have_regs = executor_lsx4_android_get_current_guest_regs != nullptr &&
                           executor_lsx4_android_get_current_guest_regs(&regs, sizeof(regs)) ==
                               0;
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIGHT_SIGNAL_CALL] stage=%s curThread=%s tid=%ld "
                        "target=%p targetName=%s targetNative=0x%zx sig=%d native=%d "
                        "handler=%p rc=%d errno=%d guestRip=0x%llx guestRsp=0x%llx regs=%d",
                        stage, CurrentThreadNameForSignalTrace(), CurrentKernelTidForSignalTrace(),
                        target, target ? target->name.c_str() : "<null>",
                        target ? static_cast<size_t>(target->native_thr.GetHandle()) : 0,
                        orbis_sig, native_sig, reinterpret_cast<void*>(Handlers[orbis_sig]), rc,
                        err, static_cast<unsigned long long>(have_regs ? regs.rip : 0),
                        static_cast<unsigned long long>(have_regs ? regs.regs[4] : 0),
                        have_regs ? 1 : 0);
}

static void TraceLightOracleThreadSnapshot(const char* reason, PthreadT target, s32 orbis_sig) {
    if (std::getenv("EXECUTOR_LIGHT_ORACLE") == nullptr || orbis_sig != POSIX_SIGUSR1) {
        return;
    }
    static std::atomic_int snapshots{16};
    if (snapshots.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    auto* thread_state = ThrState::Instance();
    int index = 0;
    int total = 0;
    int running = 0;
    int named_mono = 0;
    int unity_preload = 0;
    {
        std::scoped_lock lk{thread_state->thread_list_lock};
        total = static_cast<int>(thread_state->threads.size());
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIGHT_THREAD_SNAPSHOT] reason=%s phase=begin cur=%p curName=%s "
            "target=%p targetName=%s total=%d active=%d totalEver=%d",
            reason, g_curthread, CurrentThreadNameForSignalTrace(), target,
            target ? target->name.c_str() : "<null>", total,
            thread_state->active_threads.load(std::memory_order_relaxed),
            thread_state->total_threads.load(std::memory_order_relaxed));
        for (Pthread* thread : thread_state->threads) {
            if (thread == nullptr) {
                continue;
            }
            const bool is_running = thread->state == PthreadState::Running;
            const bool is_named_mono = ExecutorIsNamedMonoThread(thread);
            const bool is_unity_preload = thread->name == "UnityPreload";
            running += is_running ? 1 : 0;
            named_mono += is_named_mono ? 1 : 0;
            unity_preload += is_unity_preload ? 1 : 0;
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_LIGHT_THREAD_SNAPSHOT] reason=%s idx=%d thread=%p name=%s "
                "target=%d tid=%d native=0x%zx state=%u flags=0x%x tlflags=0x%x "
                "sigmask=0x%llx start=%p arg=%p ref=%d detached=%d",
                reason, index++, thread, thread->name.c_str(), thread == target ? 1 : 0,
                thread->tid.load(std::memory_order_relaxed),
                static_cast<size_t>(thread->native_thr.GetHandle()),
                static_cast<unsigned>(thread->state), static_cast<unsigned>(thread->flags),
                static_cast<unsigned>(thread->tlflags),
                static_cast<unsigned long long>(thread->sigmask),
                reinterpret_cast<void*>(thread->start_routine), thread->arg, thread->refcount,
                True(thread->flags & ThreadFlags::Detached) ? 1 : 0);
        }
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIGHT_THREAD_SNAPSHOT] reason=%s phase=end running=%d "
                        "namedMono=%d unityPreload=%d targetIsUnityPreload=%d",
                        reason, running, named_mono, unity_preload,
                        target && target->name == "UnityPreload" ? 1 : 0);
}

static bool CaptureJitGuestUcontext(Ucontext& ctx) {
    const auto* state = Executor::Jit::CurrentMachineImage();
    if (state == nullptr) {
        return false;
    }
    using Executor::Jit::GetGpr64;
    using Executor::Jit::LsxGpr;
    ctx.uc_mcontext.mc_rax = GetGpr64(*state, LsxGpr::Rax);
    ctx.uc_mcontext.mc_rcx = GetGpr64(*state, LsxGpr::Rcx);
    ctx.uc_mcontext.mc_rdx = GetGpr64(*state, LsxGpr::Rdx);
    ctx.uc_mcontext.mc_rbx = GetGpr64(*state, LsxGpr::Rbx);
    ctx.uc_mcontext.mc_rsp = GetGpr64(*state, LsxGpr::Rsp);
    ctx.uc_mcontext.mc_rbp = GetGpr64(*state, LsxGpr::Rbp);
    ctx.uc_mcontext.mc_rsi = GetGpr64(*state, LsxGpr::Rsi);
    ctx.uc_mcontext.mc_rdi = GetGpr64(*state, LsxGpr::Rdi);
    ctx.uc_mcontext.mc_r8 = GetGpr64(*state, LsxGpr::R8);
    ctx.uc_mcontext.mc_r9 = GetGpr64(*state, LsxGpr::R9);
    ctx.uc_mcontext.mc_r10 = GetGpr64(*state, LsxGpr::R10);
    ctx.uc_mcontext.mc_r11 = GetGpr64(*state, LsxGpr::R11);
    ctx.uc_mcontext.mc_r12 = GetGpr64(*state, LsxGpr::R12);
    ctx.uc_mcontext.mc_r13 = GetGpr64(*state, LsxGpr::R13);
    ctx.uc_mcontext.mc_r14 = GetGpr64(*state, LsxGpr::R14);
    ctx.uc_mcontext.mc_r15 = GetGpr64(*state, LsxGpr::R15);
    ctx.uc_mcontext.mc_rip = state->rip_or_exit;
    ctx.uc_mcontext.mc_rflags = state->rflags;
    ctx.uc_mcontext.mc_fs = static_cast<u16>(state->fs_base & 0xffffu);
    ctx.uc_mcontext.mc_gs = static_cast<u16>(state->gs_base & 0xffffu);
    ctx.uc_mcontext.mc_fsbase = state->fs_base;
    ctx.uc_mcontext.mc_gsbase = state->gs_base;
    return true;
}

static bool RunGuestSignalHandlerThroughBox64(s32 orbis_sig, s32 native_sig,
                                              OrbisKernelExceptionHandler handler,
                                              Siginfo* siginfo, Ucontext* ctx) {
    if (!executor_lsx4_android_run_guest_signal_handler) {
        TraceLiveSignal("handler_box64_unavailable", orbis_sig, native_sig, handler, nullptr, -1,
                        0, siginfo ? siginfo->_si_addr : nullptr);
        return false;
    }

    const bool jit_parity = ExecutorJitSignalParityActive();
    const std::uint8_t kind = jit_parity ? HandlerKinds[orbis_sig] :
        ((HandlerFlags[orbis_sig] & POSIX_SA_SIGINFO) != 0 ? 2u : 1u);
    const bool siginfo_handler = kind == 2;
    const bool kernel_exception_handler = kind == 3;
    std::uint64_t guest_result = 0;
    const std::uint64_t arg0 = static_cast<std::uint64_t>(orbis_sig);
    const std::uint64_t arg1 = siginfo_handler
        ? reinterpret_cast<std::uint64_t>(siginfo)
        : (kernel_exception_handler ? reinterpret_cast<std::uint64_t>(ctx) : 0);
    const std::uint64_t arg2 = siginfo_handler ? reinterpret_cast<std::uint64_t>(ctx) : 0;
    TraceLiveSignal(siginfo_handler ? "handler_box64_call_kind2"
                                    : (kernel_exception_handler ? "handler_box64_call_kind3"
                                                                : "handler_box64_call_kind1"),
                    orbis_sig, native_sig, handler, nullptr, 0, HandlerFlags[orbis_sig],
                    siginfo ? siginfo->_si_addr : nullptr);
    const int rc = executor_lsx4_android_run_guest_signal_handler(
        reinterpret_cast<std::uint64_t>(handler), arg0, arg1, arg2, &guest_result);
    TraceLiveSignal(siginfo_handler ? "handler_box64_return_kind2"
                                    : (kernel_exception_handler ? "handler_box64_return_kind3"
                                                                : "handler_box64_return_kind1"),
                    orbis_sig, native_sig, handler, nullptr, rc,
                    static_cast<int>(guest_result), siginfo ? siginfo->_si_addr : nullptr);
    if (rc < 0) {
        RecordJitSignalDispatchFailure("guest_handler_call", orbis_sig, native_sig, handler,
                                            rc, siginfo ? siginfo->_si_addr : nullptr);
        return false;
    }
    return true;
}

static bool ApplyGuestUcontextToBackend(s32 orbis_sig, s32 native_sig,
                                        OrbisKernelExceptionHandler handler,
                                        const Ucontext& ctx, void* fault_addr) {
    if (ExecutorJitSignalParityActive()) {
        auto* state = Executor::Jit::CurrentMachineImage();
        if (state == nullptr) {
            TraceLiveSignal("handler_guest_ucontext_apply_no_state", orbis_sig, native_sig,
                            handler, nullptr, -1, 0, fault_addr);
            return false;
        }
        using Executor::Jit::SetGpr64;
        using Executor::Jit::LsxGpr;
        SetGpr64(*state, LsxGpr::Rax, ctx.uc_mcontext.mc_rax);
        SetGpr64(*state, LsxGpr::Rcx, ctx.uc_mcontext.mc_rcx);
        SetGpr64(*state, LsxGpr::Rdx, ctx.uc_mcontext.mc_rdx);
        SetGpr64(*state, LsxGpr::Rbx, ctx.uc_mcontext.mc_rbx);
        SetGpr64(*state, LsxGpr::Rsp, ctx.uc_mcontext.mc_rsp);
        SetGpr64(*state, LsxGpr::Rbp, ctx.uc_mcontext.mc_rbp);
        SetGpr64(*state, LsxGpr::Rsi, ctx.uc_mcontext.mc_rsi);
        SetGpr64(*state, LsxGpr::Rdi, ctx.uc_mcontext.mc_rdi);
        SetGpr64(*state, LsxGpr::R8, ctx.uc_mcontext.mc_r8);
        SetGpr64(*state, LsxGpr::R9, ctx.uc_mcontext.mc_r9);
        SetGpr64(*state, LsxGpr::R10, ctx.uc_mcontext.mc_r10);
        SetGpr64(*state, LsxGpr::R11, ctx.uc_mcontext.mc_r11);
        SetGpr64(*state, LsxGpr::R12, ctx.uc_mcontext.mc_r12);
        SetGpr64(*state, LsxGpr::R13, ctx.uc_mcontext.mc_r13);
        SetGpr64(*state, LsxGpr::R14, ctx.uc_mcontext.mc_r14);
        SetGpr64(*state, LsxGpr::R15, ctx.uc_mcontext.mc_r15);
        state->rip_or_exit = ctx.uc_mcontext.mc_rip;
        state->rflags = ctx.uc_mcontext.mc_rflags;
        state->fs_base = ctx.uc_mcontext.mc_fsbase != 0 ? ctx.uc_mcontext.mc_fsbase
                                                        : ctx.uc_mcontext.mc_fs;
        state->gs_base = ctx.uc_mcontext.mc_gsbase != 0 ? ctx.uc_mcontext.mc_gsbase
                                                        : ctx.uc_mcontext.mc_gs;
        TraceLiveSignal("handler_guest_ucontext_apply_direct", orbis_sig, native_sig, handler,
                        nullptr, 0, 0, fault_addr);
        return true;
    }
    if (!executor_lsx4_android_set_current_guest_regs) {
        TraceLiveSignal("handler_guest_ucontext_apply_unavailable", orbis_sig, native_sig,
                        handler, nullptr, -1, 0, fault_addr);
        return false;
    }

    ExecutorBox64GuestRegs regs{};
    regs.regs[0] = ctx.uc_mcontext.mc_rax;
    regs.regs[1] = ctx.uc_mcontext.mc_rcx;
    regs.regs[2] = ctx.uc_mcontext.mc_rdx;
    regs.regs[3] = ctx.uc_mcontext.mc_rbx;
    regs.regs[4] = ctx.uc_mcontext.mc_rsp;
    regs.regs[5] = ctx.uc_mcontext.mc_rbp;
    regs.regs[6] = ctx.uc_mcontext.mc_rsi;
    regs.regs[7] = ctx.uc_mcontext.mc_rdi;
    regs.regs[8] = ctx.uc_mcontext.mc_r8;
    regs.regs[9] = ctx.uc_mcontext.mc_r9;
    regs.regs[10] = ctx.uc_mcontext.mc_r10;
    regs.regs[11] = ctx.uc_mcontext.mc_r11;
    regs.regs[12] = ctx.uc_mcontext.mc_r12;
    regs.regs[13] = ctx.uc_mcontext.mc_r13;
    regs.regs[14] = ctx.uc_mcontext.mc_r14;
    regs.regs[15] = ctx.uc_mcontext.mc_r15;
    regs.rip = ctx.uc_mcontext.mc_rip;
    // The Orbis context carries full 64-bit segment bases separately from the legacy
    // 16-bit selectors.  Mono TLS requires the bases; use the selectors only for contexts
    // produced by older callers that left the full-width fields empty.
    regs.fsbase = ctx.uc_mcontext.mc_fsbase != 0 ? ctx.uc_mcontext.mc_fsbase
                                                : ctx.uc_mcontext.mc_fs;
    regs.gsbase = ctx.uc_mcontext.mc_gsbase != 0 ? ctx.uc_mcontext.mc_gsbase
                                                : ctx.uc_mcontext.mc_gs;

    const int rc = executor_lsx4_android_set_current_guest_regs(&regs, sizeof(regs));
    TraceLiveSignal("handler_guest_ucontext_apply", orbis_sig, native_sig, handler, nullptr, rc,
                    0, fault_addr);
    return rc == 0;
}

static void ApplyJitGuestSignalMask(const Sigset& mask) {
    if (g_curthread != nullptr) {
        g_curthread->sigmask = mask.bits[0];
    }
    sigset_t native_mask{};
    sigemptyset(&native_mask);
    for (s32 guest_sig = 1; guest_sig < 128; ++guest_sig) {
        if ((mask.bits[guest_sig >> 6] &
             (std::uint64_t{1} << (guest_sig & 63))) == 0) {
            continue;
        }
        const s32 native_sig = OrbisToNativeSignal(guest_sig);
        if (native_sig > 0 && native_sig < NSIG) sigaddset(&native_mask, native_sig);
    }
    pthread_sigmask(SIG_SETMASK, &native_mask, nullptr);
}
#endif

#ifndef _WIN64
struct JitDeferredSignalMetadata {
    u64 guest_rip = 0;
    bool is_write = false;
};

static bool DispatchGuestSignal(int native_signum, siginfo_t* inf, ucontext_t* raw_context,
                                const JitDeferredSignalMetadata* deferred = nullptr) {
    const auto orbis_sig = NativeToOrbisSignal(native_signum);
#ifdef __ANDROID__
    const bool jit_parity = ExecutorJitSignalParityActive();
    RecordJitSignalProbeEntry(orbis_sig, native_signum, raw_context);
#endif
    const auto handler = Handlers[orbis_sig];
    if (handler) {
#ifdef __ANDROID__
        TraceLiveSignal("handler_enter", orbis_sig, native_signum, handler, nullptr, 0,
                        HandlerFlags[orbis_sig], inf ? inf->si_addr : nullptr);
#endif
#ifdef __ANDROID__
        // JIT enters here from an asynchronous host signal. stdio may already be locked by
        // the interrupted thread, so fprintf/fflush can deadlock before guest signal dispatch.
        if (!jit_parity) {
#endif
            std::fprintf(
                stderr,
                "[EXECUTOR_SIGNAL_HANDLER_INVOKED] sig=%d native=%d handler=%p addr=%p\n",
                orbis_sig, native_signum, reinterpret_cast<void*>(handler),
                inf ? inf->si_addr : nullptr);
            std::fflush(stderr);
#ifdef __ANDROID__
        }
#endif
        auto ctx = Ucontext{};
#ifdef __ANDROID__
        if (jit_parity && g_curthread != nullptr) {
            ctx.uc_sigmask.bits[0] = g_curthread->sigmask;
            ctx.uc_sigmask.bits[1] = 0;
        }
#endif
        auto siginfo = Siginfo{};
        siginfo._si_signo = orbis_sig;
        siginfo._si_errno = inf ? inf->si_errno : 0;
        siginfo._si_code = inf ? inf->si_code : 0;
        siginfo._si_pid = inf ? inf->si_pid : 0;
        siginfo._si_uid = inf ? inf->si_uid : 0;
        siginfo._si_addr = inf ? inf->si_addr : nullptr;
#ifdef __APPLE__
        const auto& regs = raw_context->uc_mcontext->__ss;
        ctx.uc_mcontext.mc_r8 = regs.__r8;
        ctx.uc_mcontext.mc_r9 = regs.__r9;
        ctx.uc_mcontext.mc_r10 = regs.__r10;
        ctx.uc_mcontext.mc_r11 = regs.__r11;
        ctx.uc_mcontext.mc_r12 = regs.__r12;
        ctx.uc_mcontext.mc_r13 = regs.__r13;
        ctx.uc_mcontext.mc_r14 = regs.__r14;
        ctx.uc_mcontext.mc_r15 = regs.__r15;
        ctx.uc_mcontext.mc_rdi = regs.__rdi;
        ctx.uc_mcontext.mc_rsi = regs.__rsi;
        ctx.uc_mcontext.mc_rbp = regs.__rbp;
        ctx.uc_mcontext.mc_rbx = regs.__rbx;
        ctx.uc_mcontext.mc_rdx = regs.__rdx;
        ctx.uc_mcontext.mc_rax = regs.__rax;
        ctx.uc_mcontext.mc_rcx = regs.__rcx;
        ctx.uc_mcontext.mc_rsp = regs.__rsp;
        ctx.uc_mcontext.mc_fs = regs.__fs;
        ctx.uc_mcontext.mc_gs = regs.__gs;
        ctx.uc_mcontext.mc_rip = regs.__rip;
        ctx.uc_mcontext.mc_addr = reinterpret_cast<uint64_t>(inf->si_addr);
#elif defined(__ANDROID__) && defined(__aarch64__)
        const auto& regs = raw_context->uc_mcontext.regs;
        ctx.uc_mcontext.mc_r8 = regs[8];
        ctx.uc_mcontext.mc_r9 = regs[9];
        ctx.uc_mcontext.mc_r10 = regs[10];
        ctx.uc_mcontext.mc_r11 = regs[11];
        ctx.uc_mcontext.mc_r12 = regs[12];
        ctx.uc_mcontext.mc_r13 = regs[13];
        ctx.uc_mcontext.mc_r14 = regs[14];
        ctx.uc_mcontext.mc_r15 = regs[15];
        ctx.uc_mcontext.mc_rdi = regs[0];
        ctx.uc_mcontext.mc_rsi = regs[1];
        ctx.uc_mcontext.mc_rbp = regs[29];
        ctx.uc_mcontext.mc_rbx = regs[19];
        ctx.uc_mcontext.mc_rdx = regs[2];
        ctx.uc_mcontext.mc_rax = regs[0];
        ctx.uc_mcontext.mc_rcx = regs[3];
        ctx.uc_mcontext.mc_rsp = raw_context->uc_mcontext.sp;
        ctx.uc_mcontext.mc_rip = raw_context->uc_mcontext.pc;
        ctx.uc_mcontext.mc_addr = reinterpret_cast<uint64_t>(inf->si_addr);
        if (executor_lsx4_android_get_current_guest_regs) {
            ExecutorBox64GuestRegs guest_regs{};
            const int guest_regs_rc =
                executor_lsx4_android_get_current_guest_regs(&guest_regs, sizeof(guest_regs));
            if (guest_regs_rc == 0) {
                // Publish the interrupted emulated GPRs before Mono's suspend handler can
                // acknowledge this thread.  Otherwise objects referenced only by x86 registers
                // are invisible to BDWGC and may be decommitted while Unity/FMOD still use them.
                if (executor_lsx4_android_publish_guest_gc_register_roots) {
                    executor_lsx4_android_publish_guest_gc_register_roots(&guest_regs,
                                                                             sizeof(guest_regs));
                }
                ctx.uc_mcontext.mc_rax = guest_regs.regs[0];
                ctx.uc_mcontext.mc_rcx = guest_regs.regs[1];
                ctx.uc_mcontext.mc_rdx = guest_regs.regs[2];
                ctx.uc_mcontext.mc_rbx = guest_regs.regs[3];
                ctx.uc_mcontext.mc_rsp = guest_regs.regs[4];
                ctx.uc_mcontext.mc_rbp = guest_regs.regs[5];
                ctx.uc_mcontext.mc_rsi = guest_regs.regs[6];
                ctx.uc_mcontext.mc_rdi = guest_regs.regs[7];
                ctx.uc_mcontext.mc_r8 = guest_regs.regs[8];
                ctx.uc_mcontext.mc_r9 = guest_regs.regs[9];
                ctx.uc_mcontext.mc_r10 = guest_regs.regs[10];
                ctx.uc_mcontext.mc_r11 = guest_regs.regs[11];
                ctx.uc_mcontext.mc_r12 = guest_regs.regs[12];
                ctx.uc_mcontext.mc_r13 = guest_regs.regs[13];
                ctx.uc_mcontext.mc_r14 = guest_regs.regs[14];
                ctx.uc_mcontext.mc_r15 = guest_regs.regs[15];
                ctx.uc_mcontext.mc_rip = guest_regs.rip;
                ctx.uc_mcontext.mc_fs = static_cast<uint16_t>(guest_regs.fsbase & 0xffffu);
                ctx.uc_mcontext.mc_gs = static_cast<uint16_t>(guest_regs.gsbase & 0xffffu);
                ctx.uc_mcontext.mc_fsbase = guest_regs.fsbase;
                ctx.uc_mcontext.mc_gsbase = guest_regs.gsbase;
                TraceLiveSignal("handler_guest_ucontext", orbis_sig, native_signum, handler,
                                nullptr, guest_regs_rc, 0, inf ? inf->si_addr : nullptr);
            } else {
                TraceLiveSignal("handler_guest_ucontext_unavailable", orbis_sig, native_signum,
                                handler, nullptr, guest_regs_rc, 0,
                                inf ? inf->si_addr : nullptr);
            }
        }
        // The legacy Box64 snapshot omits RFLAGS and intentionally exposes only the common GPR
        // ABI.  JIT owns a richer architectural state object; capture it directly so a
        // deferred synchronous handler sees the precise fault state and may modify every scalar
        // field represented by Orbis Ucontext.
        if (jit_parity && !CaptureJitGuestUcontext(ctx)) {
            RecordJitSignalDispatchFailure("guest_ucontext_capture", orbis_sig,
                                                native_signum, handler, -1,
                                                inf ? inf->si_addr : nullptr);
            return false;
        }
        if (deferred != nullptr) {
            ctx.uc_mcontext.mc_rip = deferred->guest_rip;
            ctx.uc_mcontext.mc_trapno = native_signum == SIGILL ? 6 : 14;
            // x86 page-fault error-code bit 1 distinguishes write from read.  The remaining
            // privilege/present bits are unavailable from Android's portable signal context.
            ctx.uc_mcontext.mc_err = deferred->is_write ? 2u : 0u;
        }
#else
        const auto& regs = raw_context->uc_mcontext.gregs;
        ctx.uc_mcontext.mc_r8 = regs[REG_R8];
        ctx.uc_mcontext.mc_r9 = regs[REG_R9];
        ctx.uc_mcontext.mc_r10 = regs[REG_R10];
        ctx.uc_mcontext.mc_r11 = regs[REG_R11];
        ctx.uc_mcontext.mc_r12 = regs[REG_R12];
        ctx.uc_mcontext.mc_r13 = regs[REG_R13];
        ctx.uc_mcontext.mc_r14 = regs[REG_R14];
        ctx.uc_mcontext.mc_r15 = regs[REG_R15];
        ctx.uc_mcontext.mc_rdi = regs[REG_RDI];
        ctx.uc_mcontext.mc_rsi = regs[REG_RSI];
        ctx.uc_mcontext.mc_rbp = regs[REG_RBP];
        ctx.uc_mcontext.mc_rbx = regs[REG_RBX];
        ctx.uc_mcontext.mc_rdx = regs[REG_RDX];
        ctx.uc_mcontext.mc_rax = regs[REG_RAX];
        ctx.uc_mcontext.mc_rcx = regs[REG_RCX];
        ctx.uc_mcontext.mc_rsp = regs[REG_RSP];
        ctx.uc_mcontext.mc_fs = (regs[REG_CSGSFS] >> 32) & 0xFFFF;
        ctx.uc_mcontext.mc_gs = (regs[REG_CSGSFS] >> 16) & 0xFFFF;
        ctx.uc_mcontext.mc_rip = (regs[REG_RIP]);
        ctx.uc_mcontext.mc_addr = reinterpret_cast<uint64_t>(inf->si_addr);
#endif
        NormalizeGuestUcontext(ctx, inf ? inf->si_addr : nullptr);
#if defined(__ANDROID__) && defined(__aarch64__)
        if (!RunGuestSignalHandlerThroughBox64(orbis_sig, native_signum, handler, &siginfo, &ctx)) {
            TraceLiveSignal("handler_box64_failed_no_direct_guest_call", orbis_sig, native_signum,
                            handler, nullptr, -1, HandlerFlags[orbis_sig],
                            inf ? inf->si_addr : nullptr);
            return false;
        }
        if (!jit_parity && orbis_sig == POSIX_SIGUSR1 &&
            std::getenv("EXECUTOR_LIVE_MONO_SIGNAL_DELIVER") != nullptr) {
            TraceLiveSignal("handler_guest_ucontext_preserve_interrupted_frame", orbis_sig,
                            native_signum, handler, nullptr, 0, HandlerFlags[orbis_sig],
                            inf ? inf->si_addr : nullptr);
        } else {
            const bool applied = ApplyGuestUcontextToBackend(
                orbis_sig, native_signum, handler, ctx,
                inf ? inf->si_addr : nullptr);
            if (jit_parity && !applied) {
                RecordJitSignalDispatchFailure(
                    "guest_ucontext_apply", orbis_sig, native_signum, handler, -1,
                    inf ? inf->si_addr : nullptr);
                TraceLiveSignal("handler_guest_ucontext_apply_failed", orbis_sig,
                                native_signum, handler, nullptr, -1,
                                HandlerFlags[orbis_sig],
                                inf ? inf->si_addr : nullptr);
                return false;
            }
        }
        if (jit_parity) {
            ApplyJitGuestSignalMask(ctx.uc_sigmask);
        }
#else
        if ((HandlerFlags[orbis_sig] & POSIX_SA_SIGINFO) != 0) {
            using SigactionHandler3 = PS4_SYSV_ABI void (*)(int, Siginfo*, void*);
            reinterpret_cast<SigactionHandler3>(handler)(orbis_sig, &siginfo, &ctx);
        } else {
            handler(orbis_sig, &ctx);
        }
#endif
#ifdef __ANDROID__
        TraceLiveSignal("handler_return", orbis_sig, native_signum, handler, nullptr, 0,
                        HandlerFlags[orbis_sig], inf ? inf->si_addr : nullptr);
#endif
#ifdef __ANDROID__
        if (!jit_parity) {
#endif
            std::fprintf(stderr,
                         "[EXECUTOR_SIGNAL_HANDLER_RETURNED] sig=%d native=%d handler=%p\n",
                         orbis_sig, native_signum, reinterpret_cast<void*>(handler));
            std::fflush(stderr);
#ifdef __ANDROID__
        }
#endif
        return true;
    } else {
        return false;
    }
}

void SigactionHandler(int native_signum, siginfo_t* inf, ucontext_t* raw_context) {
    const bool dispatched = DispatchGuestSignal(native_signum, inf, raw_context);
    if (!dispatched && !Handlers[NativeToOrbisSignal(native_signum)]) {
        UNREACHABLE_MSG("Unhandled exception");
    }
}

#ifdef __ANDROID__
extern "C" __attribute__((visibility("default"), used)) int
executor_lsx4_android_dispatch_deferred_guest_signal(
    const std::int32_t native_sig, const std::int32_t si_code,
    const std::int32_t si_errno, const std::int32_t source_pid,
    const std::uint32_t source_uid, const std::uint64_t fault_addr,
    const std::uint64_t guest_rip, const std::int32_t is_write) {
    if (!ExecutorJitSignalParityActive() || guest_rip == 0) {
        return 0;
    }
    const auto orbis_sig = NativeToOrbisSignal(native_sig);
    static std::atomic<u32> deferred_trace_count{0};
    const u32 trace_index = deferred_trace_count.fetch_add(1, std::memory_order_relaxed);
    if (trace_index < 32) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[LSX4_DEFERRED_SIGNAL] phase=enter native=%d orbis=%d handler=%p flags=0x%x "
            "kind=%u guestRip=0x%llx fault=0x%llx",
            native_sig, orbis_sig, reinterpret_cast<void*>(Handlers[orbis_sig]),
            HandlerFlags[orbis_sig], static_cast<unsigned>(HandlerKinds[orbis_sig]),
            static_cast<unsigned long long>(guest_rip),
            static_cast<unsigned long long>(fault_addr));
    }
    siginfo_t info{};
    info.si_signo = native_sig;
    info.si_code = si_code;
    info.si_errno = si_errno;
    info.si_pid = source_pid;
    info.si_uid = source_uid;
    info.si_addr = reinterpret_cast<void*>(fault_addr);
    ucontext_t raw_context{};
#if defined(__aarch64__)
    // Probe/trace consumers receive the guest RIP as the synthetic PC; no native signal frame
    // remains by design, and DispatchGuestSignal replaces this with the exact LsxMachineImage image.
    raw_context.uc_mcontext.pc = guest_rip;
#endif
    const JitDeferredSignalMetadata deferred{guest_rip, is_write != 0};
    const bool dispatched = DispatchGuestSignal(native_sig, &info, &raw_context, &deferred);
    if (trace_index < 32) {
        __android_log_print(
            dispatched ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "LSX4Native",
            "[LSX4_DEFERRED_SIGNAL] phase=return native=%d orbis=%d dispatched=%d "
            "handler=%p",
            native_sig, orbis_sig, dispatched ? 1 : 0,
            reinterpret_cast<void*>(Handlers[orbis_sig]));
    }
    return dispatched ? 1 : -1;
}
#endif

#else
void ExceptionHandler(void* arg1, void* arg2, void* arg3, PCONTEXT context) {
    const char* thrName = (char*)arg1;
    int native_signum = reinterpret_cast<uintptr_t>(arg2);
    LOG_INFO(Lib_Kernel, "Exception raised successfully on thread '{}'", thrName);
    const auto handler = Handlers[NativeToOrbisSignal(native_signum)];
    if (handler) {
        auto ctx = Ucontext{};
        ctx.uc_mcontext.mc_r8 = context->R8;
        ctx.uc_mcontext.mc_r9 = context->R9;
        ctx.uc_mcontext.mc_r10 = context->R10;
        ctx.uc_mcontext.mc_r11 = context->R11;
        ctx.uc_mcontext.mc_r12 = context->R12;
        ctx.uc_mcontext.mc_r13 = context->R13;
        ctx.uc_mcontext.mc_r14 = context->R14;
        ctx.uc_mcontext.mc_r15 = context->R15;
        ctx.uc_mcontext.mc_rdi = context->Rdi;
        ctx.uc_mcontext.mc_rsi = context->Rsi;
        ctx.uc_mcontext.mc_rbp = context->Rbp;
        ctx.uc_mcontext.mc_rbx = context->Rbx;
        ctx.uc_mcontext.mc_rdx = context->Rdx;
        ctx.uc_mcontext.mc_rax = context->Rax;
        ctx.uc_mcontext.mc_rcx = context->Rcx;
        ctx.uc_mcontext.mc_rsp = context->Rsp;
        ctx.uc_mcontext.mc_fs = context->SegFs;
        ctx.uc_mcontext.mc_gs = context->SegGs;
        handler(NativeToOrbisSignal(native_signum), &ctx);
    } else {
        UNREACHABLE_MSG("Unhandled exception");
    }
}
#endif

s32 PS4_SYSV_ABI posix_sigemptyset(Sigset* s) {
    s->bits[0] = 0;
    s->bits[1] = 0;
    return 0;
}

bool PS4_SYSV_ABI posix_sigisemptyset(Sigset* s) {
    return s->bits[0] == 0 && s->bits[1] == 0;
}

bool HasSignalHandler(const s32 sig) {
    return sig >= 0 && sig < static_cast<s32>(Handlers.size()) && Handlers[sig] != nullptr;
}

bool PS4_SYSV_ABI posix_is_signal_return(const u64 address) {
    return address == static_cast<u64>(-3);
}

s32 PS4_SYSV_ABI posix_sigalstack(const OrbisKernelExceptionHandlerStack* stack,
                                  OrbisKernelExceptionHandlerStack* old_stack) {
#ifndef _WIN32
    stack_t native_stack{};
    stack_t native_old{};
    if (stack != nullptr) {
        native_stack.ss_sp = stack->ss_sp;
        native_stack.ss_flags = stack->ss_flags;
        native_stack.ss_size = stack->ss_size;
    }
    // The reference always passes a native input object, even for a null guest
    // pointer (then it is an all-zero stack_t), and deliberately ignores errno.
    (void)sigaltstack(&native_stack, old_stack != nullptr ? &native_old : nullptr);
    if (old_stack != nullptr) {
        old_stack->ss_sp = native_old.ss_sp;
        old_stack->ss_flags = native_old.ss_flags;
        old_stack->ss_size = native_old.ss_size;
    }
#else
    if (old_stack != nullptr) std::memset(old_stack, 0, sizeof(*old_stack));
#endif
    // The reference wrapper intentionally ignores the host return code.
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI posix_sigaction(s32 sig, Sigaction* act, Sigaction* oact) {
    if (sig < 1 || sig > 128 || sig == POSIX_SIGTHR || sig == POSIX_SIGKILL ||
        sig == POSIX_SIGSTOP) {
        *__Error() = POSIX_EINVAL;
        return ORBIS_FAIL;
    }
#ifdef __ANDROID__
    const s32 native_sig = OrbisToNativeSignal(sig);
    if (ExecutorJitSignalParityActive()) {
        const auto previous_handler = Handlers[sig];
        const std::uint8_t previous_kind = HandlerKinds[sig];
        if (oact != nullptr) {
            oact->__sigaction_handler.sigaction =
                reinterpret_cast<decltype(oact->__sigaction_handler.sigaction)>(
                    previous_handler);
            oact->sa_flags = previous_kind == 2 ? POSIX_SA_SIGINFO : 0;
            // Byte-exact reference stores: zero qwords at +0x0c and +0x14.
            // This clears the mask while intentionally preserving bytes
            // +0x1c..+0x1f in the caller-owned tail of Sigaction.
            std::memset(reinterpret_cast<std::uint8_t*>(oact) + 0x0c, 0, 0x10);
        }

        if (act != nullptr) {
            const auto next_handler = reinterpret_cast<OrbisKernelExceptionHandler>(
                act->__sigaction_handler.sigaction);
            Handlers[sig] = next_handler;
            HandlerKinds[sig] =
                (act->sa_flags & POSIX_SA_SIGINFO) != 0 ? std::uint8_t{2} : std::uint8_t{1};
            HandlerFlags[sig] = act->sa_flags;
            TraceLiveSignal("jit_sigaction_register", sig, native_sig, next_handler,
                            nullptr, 0, HandlerKinds[sig], static_cast<void*>(act));
        }

        // These synchronous faults are owned by Core::SignalDispatch in the
        // reference.  The guest table still changes, but libc's disposition
        // must not be replaced here.
        if (native_sig == SIGSEGV || native_sig == SIGBUS || native_sig == SIGILL) {
            return ORBIS_OK;
        }
        if (native_sig <= 0 || native_sig >= NSIG || native_sig == SIGKILL ||
            native_sig == SIGSTOP) {
            return ORBIS_OK;
        }

        struct sigaction native_act {};
        struct sigaction native_oact {};
        struct sigaction* native_act_ptr = nullptr;
        if (act != nullptr) {
            native_act.sa_sigaction =
                reinterpret_cast<decltype(native_act.sa_sigaction)>(SigactionHandler);
            native_act.sa_flags = SA_SIGINFO | SA_ONSTACK;
#ifdef SA_RESTART
            if ((act->sa_flags & POSIX_SA_RESTART) != 0) native_act.sa_flags |= SA_RESTART;
#endif
#ifdef SA_NODEFER
            if ((act->sa_flags & POSIX_SA_NODEFER) != 0) native_act.sa_flags |= SA_NODEFER;
#endif
#ifdef SA_RESETHAND
            if ((act->sa_flags & POSIX_SA_RESETHAND) != 0) native_act.sa_flags |= SA_RESETHAND;
#endif
            sigemptyset(&native_act.sa_mask);
            for (s32 guest_mask_sig = 1; guest_mask_sig < 128; ++guest_mask_sig) {
                if ((act->sa_mask.bits[guest_mask_sig >> 6] &
                     (std::uint64_t{1} << (guest_mask_sig & 63))) == 0) {
                    continue;
                }
                const s32 mapped = OrbisToNativeSignal(guest_mask_sig);
                if (mapped > 0 && mapped < NSIG) sigaddset(&native_act.sa_mask, mapped);
            }
            native_act_ptr = &native_act;
        }

        errno = 0;
        const int ret = sigaction(native_sig, native_act_ptr,
                                  oact != nullptr ? &native_oact : nullptr);
        const int err = errno;
        TraceLiveSignal("jit_host_sigaction", sig, native_sig, Handlers[sig], nullptr,
                        ret, err, nullptr);
        if (ret < 0) {
            *__Error() = ErrnoToSceKernelError(err);
            return ORBIS_FAIL;
        }
        return ORBIS_OK;
    }
    const auto prev_handler = Handlers[sig];
    if (oact) {
        std::memset(oact, 0, sizeof(*oact));
        oact->__sigaction_handler.sigaction =
            reinterpret_cast<decltype(oact->__sigaction_handler.sigaction)>(prev_handler);
    }
    const auto next_handler = reinterpret_cast<OrbisKernelExceptionHandler>(
        act ? act->__sigaction_handler.sigaction : nullptr);
    Handlers[sig] = next_handler;
    HandlerFlags[sig] = act ? act->sa_flags : 0;
    TraceLiveSignal("sigaction_register", sig, native_sig, next_handler, nullptr, 0,
                    HandlerFlags[sig],
                    static_cast<void*>(act));
    std::fprintf(stderr,
                 "[EXECUTOR_SIGNAL_REGISTRATION] sig=%d native=%d handler=%p old=%p act=%p "
                 "oact=%p flags=0x%x\n",
                 sig, native_sig, reinterpret_cast<void*>(next_handler),
                 reinterpret_cast<void*>(prev_handler), static_cast<void*>(act),
                 static_cast<void*>(oact), act ? act->sa_flags : 0);
    std::fflush(stderr);

    struct sigaction native_act {};
    const bool can_install = act && next_handler && native_sig > 0 && native_sig < NSIG &&
                             native_sig != SIGKILL && native_sig != SIGSTOP;
    const bool route_live_mono_sigusr1_via_shadps4 =
        sig == POSIX_SIGUSR1 && std::getenv("EXECUTOR_LIVE_MONO_SIGNAL_DELIVER") != nullptr;
    const bool backend_owns_host_fault_signal =
        native_sig == SIGSEGV || native_sig == SIGBUS || native_sig == SIGILL;
    const bool route_faults_via_shadps4 =
        backend_owns_host_fault_signal &&
        std::getenv("EXECUTOR_FEX_OWNS_FAULT_SIGNALS") == nullptr;
    const bool backend_owns_live_mono_signal =
        sig == POSIX_SIGUSR1 && executor_lsx4_android_register_guest_signal_handler &&
        !route_live_mono_sigusr1_via_shadps4;
    const int backend_signal = backend_owns_live_mono_signal ? sig : native_sig;
    if (can_install && executor_lsx4_android_register_guest_signal_handler &&
        !route_live_mono_sigusr1_via_shadps4 && !route_faults_via_shadps4) {
        const int box64_rc = executor_lsx4_android_register_guest_signal_handler(
            backend_signal, reinterpret_cast<std::uint64_t>(next_handler), HandlerFlags[sig]);
        TraceLiveSignal(box64_rc == 0 ? "box64_signal_register_ok"
                                      : "box64_signal_register_failed",
                        sig, native_sig, next_handler, nullptr, box64_rc, HandlerFlags[sig],
                        static_cast<void*>(act));
        if (box64_rc == 0 && backend_owns_host_fault_signal) {
            return ORBIS_OK;
        }
        if (box64_rc == 0 && backend_owns_live_mono_signal) {
            TraceLiveSignal("backend_owns_live_mono_signal", sig, native_sig, next_handler,
                            nullptr, box64_rc, HandlerFlags[sig], static_cast<void*>(act));
            return ORBIS_OK;
        }
    }
    if (can_install && route_faults_via_shadps4) {
        TraceLiveSignal("core_owns_fault_signal", sig, native_sig, next_handler, nullptr, 0,
                        HandlerFlags[sig], static_cast<void*>(act));
        std::fprintf(stderr,
                     "[EXECUTOR_SIGNAL_CORE_FAULT_OWNER] sig=%d native=%d handler=%p "
                     "reason=pc_oracle_shadps4_signal_dispatch\n",
                     sig, native_sig, reinterpret_cast<void*>(next_handler));
        std::fflush(stderr);
        return ORBIS_OK;
    }
    int ret = 0;
    int err = 0;
    if (can_install) {
        native_act.sa_sigaction =
            reinterpret_cast<decltype(native_act.sa_sigaction)>(SigactionHandler);
        native_act.sa_flags = SA_SIGINFO | SA_ONSTACK;
#ifdef SA_RESTART
        if ((act->sa_flags & POSIX_SA_RESTART) != 0) {
            native_act.sa_flags |= SA_RESTART;
        }
#endif
#ifdef SA_NODEFER
        if ((act->sa_flags & POSIX_SA_NODEFER) != 0) {
            native_act.sa_flags |= SA_NODEFER;
        }
#endif
#ifdef SA_RESETHAND
        if ((act->sa_flags & POSIX_SA_RESETHAND) != 0) {
            native_act.sa_flags |= SA_RESETHAND;
        }
#endif
        sigemptyset(&native_act.sa_mask);
        errno = 0;
        ret = sigaction(native_sig, &native_act, nullptr);
        err = errno;
    }
    TraceLiveSignal("host_sigaction", sig, native_sig, next_handler, nullptr, ret, err, nullptr);
    std::fprintf(stderr,
                 "[EXECUTOR_SIGNAL_HOST_SIGACTION] sig=%d native=%d rc=%d errno=%d handler=%p "
                 "installed=%d\n",
                 sig, native_sig, ret, err, reinterpret_cast<void*>(next_handler),
                 can_install ? 1 : 0);
    std::fflush(stderr);
    std::fprintf(stderr,
                 "[EXECUTOR_ANDROID_SIGNAL] posix_sigaction sig=%d native=%d act=%p oact=%p "
                 "return=%d errno=%d\n",
                 sig, native_sig, static_cast<void*>(act), static_cast<void*>(oact), ret, err);
    std::fflush(stderr);
    if (ret < 0) {
        *__Error() = ErrnoToSceKernelError(err);
        return ORBIS_FAIL;
    }
    return ORBIS_OK;
#elif defined(_WIN32)
    LOG_ERROR(Lib_Kernel, "(STUBBED) called, sig: {}", sig);
    Handlers[sig] = reinterpret_cast<OrbisKernelExceptionHandler>(
        act ? act->__sigaction_handler.sigaction : nullptr);
    HandlerFlags[sig] = act ? act->sa_flags : 0;
#else
    s32 native_sig = OrbisToNativeSignal(sig);
    if (native_sig == SIGVTALRM) {
        LOG_ERROR(Lib_Kernel, "Guest is attempting to use the HLE-reserved signal {}!", sig);
        *__Error() = POSIX_EINVAL;
        return ORBIS_FAIL;
    }
#ifndef __APPLE__
    if (native_sig >= __SIGRTMIN && native_sig < SIGRTMIN) {
        LOG_ERROR(Lib_Kernel, "Guest is attempting to use the HLE libc-reserved signal {}!", sig);
        *__Error() = POSIX_EINVAL;
        return ORBIS_FAIL;
    }
#else
    if (native_sig > SIGUSR2) {
        LOG_ERROR(Lib_Kernel,
                  "Guest is attempting to use SIGRT signals, which aren't available on this "
                  "platform (signal: {})!",
                  sig);
    }
#endif
    LOG_INFO(Lib_Kernel, "called, sig: {}, native sig: {}", sig, native_sig);
    struct sigaction native_act{};
    if (act) {
        native_act.sa_flags = act->sa_flags; // todo check compatibility, on Linux it seems fine
        native_act.sa_sigaction =
            reinterpret_cast<decltype(native_act.sa_sigaction)>(SigactionHandler);
        if (!posix_sigisemptyset(&act->sa_mask)) {
            LOG_ERROR(Lib_Kernel, "Unhandled sa_mask: {:x}", act->sa_mask.bits[0]);
        }
    }
    auto const prev_handler = Handlers[sig];
    Handlers[sig] = reinterpret_cast<OrbisKernelExceptionHandler>(
        act ? act->__sigaction_handler.sigaction : nullptr);
    HandlerFlags[sig] = act ? act->sa_flags : 0;

    if (native_sig == SIGSEGV || native_sig == SIGBUS || native_sig == SIGILL) {
        return ORBIS_OK; // These are handled in Core::SignalHandler
    }
    if (native_sig > 127) {
        LOG_WARNING(Lib_Kernel, "We can't install a handler for native signal {}!", native_sig);
        return ORBIS_OK;
    }
    struct sigaction native_oact{};
    s32 ret = sigaction(native_sig, act ? &native_act : nullptr, oact ? &native_oact : nullptr);
    if (oact) {
        oact->sa_flags = native_oact.sa_flags;
        oact->__sigaction_handler.sigaction =
            reinterpret_cast<decltype(oact->__sigaction_handler.sigaction)>(prev_handler);
        if (!sigisemptyset(&native_oact.sa_mask)) {
            LOG_ERROR(Lib_Kernel, "Unhandled sa_mask");
        }
    }
    if (ret < 0) {
        LOG_ERROR(Lib_Kernel, "sigaction failed: {}", strerror(errno));
        *__Error() = ErrnoToSceKernelError(errno);
        return ORBIS_FAIL;
    }
#endif
    return ORBIS_OK;
}

PosixSignalHandler PS4_SYSV_ABI posix_signal(const s32 sig, PosixSignalHandler handler) {
    Sigaction next{};
    next.__sigaction_handler.handler =
        reinterpret_cast<decltype(next.__sigaction_handler.handler)>(handler);
    if (sig > 0 && sig < 128 &&
        (g_sigintr.bits[sig >> 6] & (std::uint64_t{1} << (sig & 63))) == 0) {
        next.sa_flags = POSIX_SA_RESTART;
    }
    posix_sigemptyset(&next.sa_mask);

    Sigaction previous{};
    if (posix_sigaction(sig, &next, &previous) < 0) {
        return reinterpret_cast<PosixSignalHandler>(static_cast<std::uintptr_t>(-1));
    }
    return reinterpret_cast<PosixSignalHandler>(previous.__sigaction_handler.handler);
}

s32 PS4_SYSV_ABI posix_pthread_kill(PthreadT thread, s32 sig) {
    if (sig < 1 || sig > 128) { // off-by-one error?
        return POSIX_EINVAL;
    }
    int const native_signum = OrbisToNativeSignal(sig);
#ifdef __ANDROID__
    if (ExecutorJitSignalParityActive()) {
        if (thread == nullptr) {
            return POSIX_EINVAL;
        }
        const auto pthr = static_cast<pthread_t>(thread->native_thr.GetHandle());
        TraceLiveSignal("jit_pthread_kill_enter", sig, native_signum, Handlers[sig],
                        thread, 0, HandlerKinds[sig], nullptr);
        const int ret = pthread_kill(pthr, native_signum);
        TraceLiveSignal("jit_pthread_kill_return", sig, native_signum, Handlers[sig],
                        thread, ret, errno, nullptr);
        // Preserve ESRCH/EINVAL so sceKernelRaiseException can encode the
        // corresponding Orbis error. BDWGC relies on ESRCH to exclude a dead
        // thread from the number of suspend acknowledgements it waits for.
        return ret;
    }
    TraceLightOraclePthreadKill("enter", thread, sig, native_signum);
    const bool named_mono_thread = ExecutorIsNamedMonoThread(thread);
    const char* mono_suspend = std::getenv("MONO_THREADS_SUSPEND");
    const bool mono_coop = mono_suspend && std::strcmp(mono_suspend, "coop") == 0;
    const bool deliver_mono_sigusr1 =
        std::getenv("EXECUTOR_LIVE_MONO_SIGNAL_DELIVER") != nullptr;
    const bool full_coop_probe =
        std::getenv("EXECUTOR_LIVE_MONO_FULL_COOP_PROBE") != nullptr;
    const bool live_mono_sigusr1 = thread && sig == POSIX_SIGUSR1;
    if (live_mono_sigusr1) {
        TraceLightOracleThreadSnapshot("pthread_kill_sigusr1", thread, sig);
    }
    if (live_mono_sigusr1 && (deliver_mono_sigusr1 || full_coop_probe || named_mono_thread)) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_SUSPEND_SIGNAL_ATTEMPT] thread=%s sig=%d "
                            "native=%d monoThreadsSuspend=%s fullCoopProbe=%d "
                            "signalDeliver=%d",
                            thread->name.c_str(), sig, native_signum,
                            mono_suspend ? mono_suspend : "", full_coop_probe ? 1 : 0,
                            deliver_mono_sigusr1 ? 1 : 0);
        if (full_coop_probe) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_MONO_FULL_COOP_RESULT] result=signal_requested "
                                "thread=%s reason=mono_still_uses_preemptive_suspend",
                                thread->name.c_str());
            return ORBIS_OK;
        }
    }
    const bool allow_synthetic_mono_ack =
        std::getenv("EXECUTOR_LIVE_MONO_SYNTHETIC_ACK") != nullptr &&
        !(executor_lsx4_android_jit_active != nullptr &&
          executor_lsx4_android_jit_active());
    if (live_mono_sigusr1 && allow_synthetic_mono_ack && ExecutorIsHleUnsafeMonoThread(thread) &&
        executor_live_mono_synthetic_suspend_ack) {
        const int synth_rc =
            executor_live_mono_synthetic_suspend_ack(thread, "pthread_kill_hle_wait");
        TraceLiveSignal("synthetic_hle_suspend_ack", sig, native_signum, Handlers[sig], thread,
                        synth_rc, HandlerFlags[sig], nullptr);
        if (synth_rc == 0) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_MONO_SIGNAL_DELIVER] thread=%s sig=%d "
                                "reason=pc_oracle_synthetic_hle_wait_ack",
                                thread->name.c_str(), sig);
            TraceLightOraclePthreadKill("return_synthetic_ack", thread, sig, native_signum,
                                        ORBIS_OK, 0);
            return ORBIS_OK;
        }
    }
    if (mono_coop && named_mono_thread && sig == POSIX_SIGUSR1 && !deliver_mono_sigusr1) {
        const bool suspend_acked =
            ExecutorSignalKernelSemaByNameForLiveMono("SuspendSemaphore", 1);
        const bool resume_acked =
            ExecutorSignalKernelSemaByNameForLiveMono("ResumeSemaphore", 1);
        const int posix_acked = executor_live_signal_pending_mono_posix_sems
                                    ? executor_live_signal_pending_mono_posix_sems(2)
                                    : -1;
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_COOP_SIGNAL_ACK] thread=%s sig=%d "
                            "suspendAck=%d resumeAck=%d posixAck=%d "
                            "reason=avoid-host-signal-into-fex",
                            thread->name.c_str(), sig, suspend_acked ? 1 : 0,
                            resume_acked ? 1 : 0, posix_acked);
        TraceLightOraclePthreadKill("return_coop_ack", thread, sig, native_signum, ORBIS_OK, 0);
        return ORBIS_OK;
    }
    const bool allow_safe_point_mono_signal =
        std::getenv("EXECUTOR_LIVE_MONO_SAFEPOINT_SIGNAL") != nullptr;
    // STANDOFF FIX (mark side): the Mono stop-the-world initiator (Game:Main) sends coop-suspend
    // SIGUSR1 to ALL managed threads, incl. the UnityWorker pool and UnityGfxDeviceWorker — not just
    // the classic "mono thread"/UnityPreload names. Those workers are host-parked in posix_sem_wait
    // (@eboot 0x1f401c8), so the FEX pause-frame queue route below can never deliver to them; only
    // the MARK-PENDING route reaches them (they run the pending suspend from their own 1ms wait poll,
    // now admitted by the running_own_suspend gate in mutex.cpp). Without marking them, a worker's
    // SIGUSR1 is queued into a host-blocked thread and its SuspendSemaphore ACK never fires, so the
    // whole STW never completes and Game:Main never posts the mainData dispatch -> standoff/black.
    const bool mono_managed_suspend_target =
        named_mono_thread ||
        (thread && (thread->name.rfind("UnityWorker", 0) == 0 ||
                    thread->name.rfind("UnityGfxDeviceWorker", 0) == 0));
    if (live_mono_sigusr1 && allow_safe_point_mono_signal && mono_managed_suspend_target &&
        executor_live_mono_mark_pending_sigusr1_for_thread) {
        executor_live_mono_mark_pending_sigusr1_for_thread(thread);
        TraceLiveSignal("mark_pending_orbis_safe_point", sig, native_signum, Handlers[sig],
                        thread, 0, HandlerFlags[sig], nullptr);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_SIGNAL_DELIVER] thread=%s sig=%d "
                            "reason=pc_oracle_alertable_safe_point",
                            thread->name.c_str(), sig);
        TraceLightOraclePthreadKill("return_mark_pending", thread, sig, native_signum, ORBIS_OK,
                                    0);
        return ORBIS_OK;
    }
    if (live_mono_sigusr1 && !deliver_mono_sigusr1 &&
        (executor_lsx4_android_queue_guest_signal_for_thread ||
         executor_lsx4_android_queue_guest_signal_for_thread_context ||
         executor_lsx4_android_queue_guest_signal)) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_SIGNAL_DELIVER] thread=%s sig=%d "
                            "reason=target-thread-fex-bridge",
                            thread->name.c_str(), sig);
        const auto pthr = static_cast<pthread_t>(thread->native_thr.GetHandle());
        TraceLiveSignal("queue_guest_signal_enter", sig, native_signum, Handlers[sig], thread, 0,
                        0, nullptr);
        const int backend_signal = sig == POSIX_SIGUSR1 ? sig : native_signum;
        int queue_rc = -1;
        if (executor_lsx4_android_queue_guest_signal_for_thread) {
            queue_rc = executor_lsx4_android_queue_guest_signal_for_thread(
                thread, static_cast<std::uintptr_t>(pthr), backend_signal);
        } else {
            queue_rc = executor_lsx4_android_queue_guest_signal(
                static_cast<std::uintptr_t>(pthr), backend_signal);
        }
        TraceLiveSignal("queue_guest_signal_return", sig, native_signum, Handlers[sig], thread,
                        queue_rc, errno, nullptr);
        if (queue_rc == 0) {
            TraceLightOraclePthreadKill("return_queue_guest_signal", thread, sig, native_signum,
                                        ORBIS_OK, 0);
            return ORBIS_OK;
        }
        if (executor_live_mono_mark_pending_sigusr1_for_thread) {
            executor_live_mono_mark_pending_sigusr1_for_thread(thread);
            TraceLiveSignal("mark_pending_orbis_safe_point_fallback", sig, native_signum,
                            Handlers[sig], thread, queue_rc, HandlerFlags[sig], nullptr);
            TraceLightOraclePthreadKill("return_mark_pending_fallback", thread, sig,
                                        native_signum, ORBIS_OK, queue_rc);
            return ORBIS_OK;
        }
    }
    if (mono_coop && named_mono_thread && sig == POSIX_SIGUSR1) {
        const bool suspend_acked =
            ExecutorSignalKernelSemaByNameForLiveMono("SuspendSemaphore", 1);
        const bool resume_acked =
            ExecutorSignalKernelSemaByNameForLiveMono("ResumeSemaphore", 1);
        const int posix_acked = executor_live_signal_pending_mono_posix_sems
                                    ? executor_live_signal_pending_mono_posix_sems(2)
                                    : -1;
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_COOP_SIGNAL_ACK] thread=%s sig=%d "
                            "suspendAck=%d resumeAck=%d posixAck=%d "
                            "reason=avoid-host-signal-into-fex",
                            thread->name.c_str(), sig, suspend_acked ? 1 : 0,
                            resume_acked ? 1 : 0, posix_acked);
        TraceLightOraclePthreadKill("return_coop_ack_late", thread, sig, native_signum, ORBIS_OK,
                                    0);
        return ORBIS_OK;
    }
    if (!deliver_mono_sigusr1 && std::getenv("EXECUTOR_LIVE_MONO_SIGNAL_BYPASS") &&
        named_mono_thread && sig == POSIX_SIGUSR1) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_SIGNAL_BYPASS] thread=%s sig=%d reason=explicit-env",
                            thread->name.c_str(), sig);
        TraceLightOraclePthreadKill("return_bypass", thread, sig, native_signum, ORBIS_OK, 0);
        return ORBIS_OK;
    }
    if (sig == POSIX_SIGABRT || sig == POSIX_SIGTERM) {
        std::fprintf(stderr,
                     "[EXECUTOR_ANDROID_SIGNAL] pthread_kill thread=%s sig=%d suppressed "
                     "return=0\n",
                     thread ? thread->name.c_str() : "<null>", sig);
        std::fflush(stderr);
        return ORBIS_OK;
    }
#endif
    LOG_WARNING(Lib_Kernel, "Raising signal {} on thread '{}'", sig, thread->name);
#ifndef _WIN64
#ifdef __ANDROID__
    const auto pthr = static_cast<pthread_t>(thread->native_thr.GetHandle());
    TraceLiveSignal("pthread_kill_enter", sig, native_signum, Handlers[sig], thread, 0, 0,
                    nullptr);
#else
    const auto pthr = reinterpret_cast<pthread_t>(thread->native_thr.GetHandle());
    PcOracleSignalLog("pthread_kill_enter", thread, sig, native_signum, Handlers[sig], 0, 0);
#endif
    const auto ret = pthread_kill(pthr, native_signum);
#ifdef __ANDROID__
    TraceLiveSignal("pthread_kill_return", sig, native_signum, Handlers[sig], thread, ret, errno,
                    nullptr);
    TraceLightOraclePthreadKill("return_host_pthread_kill", thread, sig, native_signum, ret,
                                errno);
#else
    PcOracleSignalLog("pthread_kill_return", thread, sig, native_signum, Handlers[sig], ret,
                      errno);
#endif
    if (ret != 0) {
        LOG_ERROR(Kernel, "Failed to send exception signal to thread '{}': {}", thread->name,
                  strerror(errno));
    }
#else
    USER_APC_OPTION option;
    option.UserApcFlags = QueueUserApcFlagsSpecialUserApc;

    u64 res = NtQueueApcThreadEx(reinterpret_cast<HANDLE>(thread->native_thr.GetHandle()), option,
                                 ExceptionHandler, (void*)thread->name.c_str(),
                                 (void*)(s64)native_signum, nullptr);
    PcOracleSignalLog("queue_apc_return", thread, sig, native_signum, Handlers[sig],
                      static_cast<int>(res), 0);
    ASSERT(res == 0);
#endif
    return ORBIS_OK;
}

// libkernel has a check in sceKernelInstallExceptionHandler and sceKernelRemoveExceptionHandler for
// validating if the application requested a handler for an allowed signal or not. However, that is
// just a wrapper for sigaction, which itself does not have any such restrictions, and therefore
// this check is ridiculously trivial to go around. This, however, means that we need to support all
// 127 - 3 possible signals, even if realistically, only homebrew will use most of them.
static std::unordered_set<s32> orbis_allowed_signals{
    POSIX_SIGHUP, POSIX_SIGILL, POSIX_SIGFPE, POSIX_SIGBUS, POSIX_SIGSEGV, POSIX_SIGUSR1,
};

int PS4_SYSV_ABI sceKernelInstallExceptionHandler(s32 signum, OrbisKernelExceptionHandler handler) {
    if (!orbis_allowed_signals.contains(signum)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (Handlers[signum] != nullptr) {
        return ORBIS_KERNEL_ERROR_EAGAIN;
    }
    LOG_INFO(Lib_Kernel, "Installing signal handler for {}", signum);
    Sigaction act = {};
    act.sa_flags = POSIX_SA_SIGINFO | POSIX_SA_RESTART;
    act.__sigaction_handler.sigaction =
        reinterpret_cast<decltype(act.__sigaction_handler.sigaction)>(handler);
    posix_sigemptyset(&act.sa_mask);
    s32 ret = posix_sigaction(signum, &act, nullptr);
    if (ret < 0) {
        LOG_ERROR(Lib_Kernel, "Failed to add handler for signal {}: {}", signum,
                  strerror(*__Error()));
        return ErrnoToSceKernelError(*__Error());
    }
#ifdef __ANDROID__
    if (ExecutorJitSignalParityActive()) {
        HandlerKinds[signum] = 3;
    }
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelRemoveExceptionHandler(s32 signum) {
    if (!orbis_allowed_signals.contains(signum)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    int const native_signum = OrbisToNativeSignal(signum);
    Handlers[signum] = nullptr;
    HandlerFlags[signum] = 0;
    HandlerKinds[signum] = 0;
    Sigaction act = {};
    act.sa_flags = POSIX_SA_SIGINFO;
    act.__sigaction_handler.sigaction = nullptr;
    posix_sigemptyset(&act.sa_mask);
    s32 ret = posix_sigaction(signum, &act, nullptr);
    if (ret < 0) {
        LOG_ERROR(Lib_Kernel, "Failed to remove handler for signal {}: {}", signum,
                  strerror(*__Error()));
        return ErrnoToSceKernelError(*__Error());
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelRaiseException(PthreadT thread, int signum) {
#ifdef __ANDROID__
    static std::atomic_int trace_budget{8};
    const bool trace_call = trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0;
    ExecutorBox64GuestRegs raise_regs{};
    const bool have_raise_regs =
        executor_lsx4_android_get_current_guest_regs != nullptr &&
        executor_lsx4_android_get_current_guest_regs(&raise_regs, sizeof(raise_regs)) == 0;
    if (trace_call) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_RAISE_EXCEPTION_CALL] thread=%p targetName=%s signum=%d "
                            "guestRip=0x%llx guestRsp=0x%llx fs=0x%llx regs=%u",
                            thread, thread ? thread->name.c_str() : "<null>", signum,
                            static_cast<unsigned long long>(have_raise_regs ? raise_regs.rip : 0),
                            static_cast<unsigned long long>(have_raise_regs ? raise_regs.regs[4] : 0),
                            static_cast<unsigned long long>(have_raise_regs ? raise_regs.fsbase : 0),
                            static_cast<unsigned>(have_raise_regs));
    }
#endif
    if (signum != POSIX_SIGUSR1) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_RAISE_EXCEPTION] thread=%p signum=%d ret=0x%x "
                            "reason=invalid_signal",
                            thread, signum, static_cast<u32>(ORBIS_KERNEL_ERROR_EINVAL));
#endif
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#ifdef __ANDROID__
    // BDWGC scans the collector thread's native registers before it stops its peers.  Under
    // JIT those AArch64 registers do not contain the emulated x86-64 GPRs, so publish the
    // initiating thread's guest register image as a conservative root as well.  The target-side
    // signal handler below publishes every suspended peer, but Game:Main (the collector) is never
    // sent its own SIGUSR1 and was therefore the one missing register root.
    if (have_raise_regs && executor_lsx4_android_publish_guest_gc_register_roots) {
        executor_lsx4_android_publish_guest_gc_register_roots(&raise_regs,
                                                                 sizeof(raise_regs));
    }
    g_executor_raise_regs = raise_regs;
    g_executor_have_raise_regs = have_raise_regs;
#endif
    s32 ret = posix_pthread_kill(thread, signum);
#ifdef __ANDROID__
    g_executor_have_raise_regs = false;
    g_executor_raise_regs = {};
#endif
    if (ret != 0) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_RAISE_EXCEPTION] thread=%p signum=%d nativeRet=%d "
                            "ret=0x%x reason=kill_error",
                            thread, signum, ret, static_cast<u32>(ErrnoToSceKernelError(ret)));
#endif
        return ErrnoToSceKernelError(ret);
    }
#ifdef __ANDROID__
    if (trace_call) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_RAISE_EXCEPTION] thread=%p targetName=%s native=0x%zx "
                            "signum=%d nativeRet=%d ret=0x%x handler=%p",
                            thread, thread ? thread->name.c_str() : "<null>",
                            thread ? static_cast<size_t>(thread->native_thr.GetHandle()) : 0,
                            signum, ret, static_cast<u32>(ret),
                            reinterpret_cast<void*>(Handlers[signum]));
    }
#endif
    return ret;
}

s32 PS4_SYSV_ABI sceKernelDebugRaiseException(s32 error, s64 unk) {
    if (unk != 0) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#ifdef __ANDROID__
    std::fprintf(stderr,
                 "[EXECUTOR_ANDROID_SIGNAL] sceKernelDebugRaiseException error=0x%x "
                 "suppressed return=0\n",
                 static_cast<unsigned int>(error));
    std::fflush(stderr);
    return ORBIS_OK;
#endif
    UNREACHABLE_MSG("error {:#x}", error);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelDebugRaiseExceptionOnReleaseMode(s32 error, s64 unk) {
    if (unk != 0) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#ifdef __ANDROID__
    std::fprintf(stderr,
                 "[EXECUTOR_ANDROID_SIGNAL] sceKernelDebugRaiseExceptionOnReleaseMode error=0x%x "
                 "suppressed return=0\n",
                 static_cast<unsigned int>(error));
    std::fflush(stderr);
    return ORBIS_OK;
#endif
    UNREACHABLE_MSG("error {:#x}", error);
    return ORBIS_OK;
}

void RegisterException(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("il03nluKfMk", "libkernel_unity", 1, "libkernel", sceKernelRaiseException);
    LIB_FUNCTION("WkwEd3N7w0Y", "libkernel_unity", 1, "libkernel",
                 sceKernelInstallExceptionHandler);
    LIB_FUNCTION("Qhv5ARAoOEc", "libkernel_unity", 1, "libkernel", sceKernelRemoveExceptionHandler);
    LIB_FUNCTION("OMDRKKAZ8I4", "libkernel", 1, "libkernel", sceKernelDebugRaiseException);
    LIB_FUNCTION("zE-wXIZjLoM", "libkernel", 1, "libkernel",
                 sceKernelDebugRaiseExceptionOnReleaseMode);
    LIB_FUNCTION("WkwEd3N7w0Y", "libkernel", 1, "libkernel", sceKernelInstallExceptionHandler);
    LIB_FUNCTION("Qhv5ARAoOEc", "libkernel", 1, "libkernel", sceKernelRemoveExceptionHandler);

    LIB_FUNCTION("KiJEPEWRyUY", "libkernel", 1, "libkernel", posix_sigaction);
    LIB_FUNCTION("+F7C-hdk7+E", "libkernel", 1, "libkernel", posix_sigemptyset);
    LIB_FUNCTION("yH-uQW3LbX0", "libkernel", 1, "libkernel", posix_pthread_kill);
    LIB_FUNCTION("VADc3MNQ3cM", "libkernel", 1, "libkernel", posix_signal);
    LIB_FUNCTION("sHziAegVp74", "libkernel", 1, "libkernel", posix_sigalstack);
    LIB_FUNCTION("crb5j7mkk1c", "libkernel", 1, "libkernel", posix_is_signal_return);
    LIB_FUNCTION("KiJEPEWRyUY", "libScePosix", 1, "libkernel", posix_sigaction);
    LIB_FUNCTION("+F7C-hdk7+E", "libScePosix", 1, "libkernel", posix_sigemptyset);
    LIB_FUNCTION("yH-uQW3LbX0", "libScePosix", 1, "libkernel", posix_pthread_kill);
    LIB_FUNCTION("VADc3MNQ3cM", "libScePosix", 1, "libkernel", posix_signal);
    LIB_FUNCTION("sHziAegVp74", "libScePosix", 1, "libkernel", posix_sigalstack);
    LIB_FUNCTION("crb5j7mkk1c", "libScePosix", 1, "libkernel", posix_is_signal_return);
}

} // namespace Libraries::Kernel
