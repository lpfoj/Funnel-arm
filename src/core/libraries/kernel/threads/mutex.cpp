// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef __ANDROID__
#include <android/log.h>
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "common/arch.h"
#include "common/assert.h"
#include "common/types.h"
#include "core/memory.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/threads/thread_state.h"
#include "core/libraries/libs.h"

namespace Libraries::Kernel {

static constexpr u32 MUTEX_ADAPTIVE_SPINS = 2000;
static_assert(MUTEX_ADAPTIVE_SPINS == 2000);
static std::mutex MutxStaticLock;

#ifdef __ANDROID__
extern "C" bool executor_lsx4_android_should_relax_guest_mutex(void* mutex)
    __attribute__((weak));
extern "C" void executor_lsx4_android_dump_box64_emu_states(const char* reason)
    __attribute__((weak));
extern "C" void executor_live_hle_flight_dump(const char* reason, u64 builder_total,
                                              u64 builder_draw, u64 builder_shader,
                                              u64 active_cb, u64 active_dw)
    __attribute__((weak));
extern "C" bool executor_lsx4_android_should_yield_unity_preload_mutex()
    __attribute__((weak));
extern "C" int executor_box64_embed_get_guest_thread_regs_by_pthread(
    std::uintptr_t pthread_handle, u64* rip, u64* rsp, u64* rbp, long* tid, char* host_name,
    std::size_t host_name_size) __attribute__((weak));
extern "C" int executor_box64_embed_get_guest_thread_regs_by_tid(
    long host_tid, u64* rip, u64* rsp, u64* rbp, long* tid, char* host_name,
    std::size_t host_name_size) __attribute__((weak));
extern "C" int executor_jit_get_guest_thread_regs_by_pthread(
    std::uintptr_t pthread_handle, u64* rip, u64* rsp, u64* rbp, long* tid, char* host_name,
    std::size_t host_name_size) __attribute__((weak));
extern "C" int executor_jit_get_guest_thread_regs_by_tid(
    long host_tid, u64* rip, u64* rsp, u64* rbp, long* tid, char* host_name,
    std::size_t host_name_size) __attribute__((weak));
extern "C" int executor_box64_embed_get_guest_thread_full_regs_by_pthread(
    std::uintptr_t pthread_handle, void* out, std::size_t out_size, long* tid, char* host_name,
    std::size_t host_name_size) __attribute__((weak));
extern "C" int executor_lsx4_android_symbolize_guest_pc(
    u64 pc, char* module_name, std::size_t module_name_size, u64* module_base, u64* module_offset,
    char* symbol_name, std::size_t symbol_name_size, u64* symbol_offset, u64* symbol_delta)
    __attribute__((weak));
extern "C" int executor_lsx4_android_queue_guest_signal(std::uintptr_t pthread_handle,
                                                           int signum)
    __attribute__((weak));
extern "C" int executor_lsx4_android_queue_guest_signal_for_thread(
    void* guest_thread, std::uintptr_t pthread_handle, int signum) __attribute__((weak));
extern "C" int executor_lsx4_android_queue_guest_signal_for_thread_context(
    void* guest_thread, std::uintptr_t pthread_handle, int signum, std::uint64_t rip,
    std::uint64_t rsp) __attribute__((weak));
extern "C" int executor_lsx4_android_run_guest_signal_handler(std::uint64_t handler,
                                                                 std::uint64_t arg0,
                                                                 std::uint64_t arg1,
                                                                 std::uint64_t arg2,
                                                                 std::uint64_t* guest_result)
    __attribute__((weak));
extern "C" int executor_lsx4_android_get_current_guest_regs(void* out, std::size_t out_size)
    __attribute__((weak));
extern "C" int executor_lsx4_android_set_current_guest_regs(const void* in,
                                                               std::size_t in_size)
    __attribute__((weak));
extern "C" bool executor_lsx4_android_jit_active() __attribute__((weak));
extern "C" int executor_live_signal_pending_mono_posix_sems(int max_posts)
    __attribute__((weak));
extern "C" bool executor_live_signal_mono_resume_event_for_live_abba() __attribute__((weak));
extern "C" void executor_live_mono_arm_resume_event_after_waiters(int waiters, const char* reason)
    __attribute__((weak));
extern "C" int executor_live_mono_synthetic_suspend_ack(void* guest_thread, const char* reason)
    __attribute__((weak));
extern "C" int executor_live_get_current_hle_call_site(
    std::uint64_t* guest_return, std::uint64_t* return_off, std::uint64_t* arg0, char* symbol,
    std::size_t symbol_size, char* module, std::size_t module_size) __attribute__((weak));
bool ExecutorSignalKernelSemaByNameForLiveMono(const char* name, s32 signal_count);
bool ExecutorWaitKernelSemaByNameForLiveMono(const char* name, s32 need_count);
bool ExecutorLivePosixSemHasWaitEnter(void* sem_slot);

static bool ExecutorLightOracleMode() {
    static const bool enabled = std::getenv("EXECUTOR_LIGHT_ORACLE") != nullptr;
    return enabled;
}

static bool ExecutorVerboseMonoMutexLedgerEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("EXECUTOR_VERBOSE_MONO_MUTEX_LEDGER");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool ExecutorStallMutexLedgerEnabled() {
    static const bool marker_enabled = [] {
        const char* env = std::getenv("EXECUTOR_LIVE_STALL_THREAD_DUMP");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return marker_enabled && executor_lsx4_android_jit_active != nullptr &&
           executor_lsx4_android_jit_active();
}

static bool ExecutorMutexHotTraceEnabled() {
    static const bool enabled =
        std::getenv("EXECUTOR_LIGHT_ORACLE") != nullptr ||
        std::getenv("EXECUTOR_VERBOSE_MONO_MUTEX_LEDGER") != nullptr ||
        std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr ||
        std::getenv("EXECUTOR_TRACE_LIVE_SYNC") != nullptr ||
        std::getenv("EXECUTOR_TRACE_LIVE_UNITY_MUTEX_OWNER") != nullptr ||
        std::getenv("EXECUTOR_LIVE_STALL_THREAD_DUMP") != nullptr;
    return enabled;
}

using ExecutorGuestThreadRegsByPthreadFn = int (*)(std::uintptr_t, u64*, u64*, u64*, long*, char*,
                                                   std::size_t);
using ExecutorGuestThreadRegsByTidFn = int (*)(long, u64*, u64*, u64*, long*, char*, std::size_t);
using ExecutorGuestThreadFullRegsByPthreadFn = int (*)(std::uintptr_t, void*, std::size_t, long*,
                                                       char*, std::size_t);
using ExecutorSymbolizeGuestPcFn = int (*)(u64, char*, std::size_t, u64*, u64*, char*,
                                           std::size_t, u64*, u64*);

struct ExecutorFexGuestRegsMirror {
    u64 regs[16];
    u64 rip;
    u64 old_ip;
    u64 fsbase;
    u64 gsbase;
    int quit;
    int exit;
    int error;
};

extern std::array<OrbisKernelExceptionHandler, 130> Handlers;
extern std::array<int, 130> HandlerFlags;

static ExecutorGuestThreadRegsByPthreadFn ResolveGuestThreadRegsByPthread() {
    if (executor_jit_get_guest_thread_regs_by_pthread &&
        executor_lsx4_android_jit_active &&
        executor_lsx4_android_jit_active()) {
        return executor_jit_get_guest_thread_regs_by_pthread;
    }
    if (executor_box64_embed_get_guest_thread_regs_by_pthread) {
        return executor_box64_embed_get_guest_thread_regs_by_pthread;
    }
    static std::atomic<ExecutorGuestThreadRegsByPthreadFn> cached{nullptr};
    static std::atomic_bool attempted{false};
    if (auto* fn = cached.load(std::memory_order_acquire)) {
        return fn;
    }
    bool expected = false;
    if (!attempted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return cached.load(std::memory_order_acquire);
    }
    auto* sym = reinterpret_cast<ExecutorGuestThreadRegsByPthreadFn>(
        dlsym(RTLD_DEFAULT, "executor_box64_embed_get_guest_thread_regs_by_pthread"));
    if (!sym) {
        void* handle = dlopen("libexecutor_fexcore_embed.so", RTLD_NOW | RTLD_NOLOAD);
        if (handle) {
            sym = reinterpret_cast<ExecutorGuestThreadRegsByPthreadFn>(
                dlsym(handle, "executor_box64_embed_get_guest_thread_regs_by_pthread"));
        }
    }
    cached.store(sym, std::memory_order_release);
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MUTEX_OWNER_PC_RESOLVE] fn=%p weak=%p",
                        reinterpret_cast<void*>(sym),
                        reinterpret_cast<void*>(
                            executor_box64_embed_get_guest_thread_regs_by_pthread));
    return sym;
}

static ExecutorGuestThreadRegsByTidFn ResolveGuestThreadRegsByTid() {
    if (executor_jit_get_guest_thread_regs_by_tid &&
        executor_lsx4_android_jit_active &&
        executor_lsx4_android_jit_active()) {
        return executor_jit_get_guest_thread_regs_by_tid;
    }
    if (executor_box64_embed_get_guest_thread_regs_by_tid) {
        return executor_box64_embed_get_guest_thread_regs_by_tid;
    }
    static std::atomic<ExecutorGuestThreadRegsByTidFn> cached{nullptr};
    static std::atomic_bool attempted{false};
    if (auto* fn = cached.load(std::memory_order_acquire)) {
        return fn;
    }
    bool expected = false;
    if (!attempted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return cached.load(std::memory_order_acquire);
    }
    auto* sym = reinterpret_cast<ExecutorGuestThreadRegsByTidFn>(
        dlsym(RTLD_DEFAULT, "executor_box64_embed_get_guest_thread_regs_by_tid"));
    if (!sym) {
        void* handle = dlopen("libexecutor_fexcore_embed.so", RTLD_NOW | RTLD_NOLOAD);
        if (handle) {
            sym = reinterpret_cast<ExecutorGuestThreadRegsByTidFn>(
                dlsym(handle, "executor_box64_embed_get_guest_thread_regs_by_tid"));
        }
    }
    cached.store(sym, std::memory_order_release);
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MUTEX_OWNER_PC_RESOLVE_TID] fn=%p weak=%p",
                        reinterpret_cast<void*>(sym),
                        reinterpret_cast<void*>(
                            executor_box64_embed_get_guest_thread_regs_by_tid));
    return sym;
}

static ExecutorGuestThreadFullRegsByPthreadFn ResolveGuestThreadFullRegsByPthread() {
    if (executor_box64_embed_get_guest_thread_full_regs_by_pthread) {
        return executor_box64_embed_get_guest_thread_full_regs_by_pthread;
    }
    static std::atomic<ExecutorGuestThreadFullRegsByPthreadFn> cached{nullptr};
    static std::atomic_bool attempted{false};
    if (auto* fn = cached.load(std::memory_order_acquire)) {
        return fn;
    }
    bool expected = false;
    if (!attempted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return cached.load(std::memory_order_acquire);
    }
    auto* sym = reinterpret_cast<ExecutorGuestThreadFullRegsByPthreadFn>(
        dlsym(RTLD_DEFAULT, "executor_box64_embed_get_guest_thread_full_regs_by_pthread"));
    if (!sym) {
        void* handle = dlopen("libexecutor_fexcore_embed.so", RTLD_NOW | RTLD_NOLOAD);
        if (handle) {
            sym = reinterpret_cast<ExecutorGuestThreadFullRegsByPthreadFn>(dlsym(
                handle, "executor_box64_embed_get_guest_thread_full_regs_by_pthread"));
        }
    }
    cached.store(sym, std::memory_order_release);
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MUTEX_OWNER_FULLREG_RESOLVE] fn=%p weak=%p",
                        reinterpret_cast<void*>(sym),
                        reinterpret_cast<void*>(
                            executor_box64_embed_get_guest_thread_full_regs_by_pthread));
    return sym;
}

static ExecutorSymbolizeGuestPcFn ResolveSymbolizeGuestPc() {
    if (executor_lsx4_android_symbolize_guest_pc) {
        return executor_lsx4_android_symbolize_guest_pc;
    }
    static std::atomic<ExecutorSymbolizeGuestPcFn> cached{nullptr};
    static std::atomic_bool attempted{false};
    if (auto* fn = cached.load(std::memory_order_acquire)) {
        return fn;
    }
    bool expected = false;
    if (!attempted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        return cached.load(std::memory_order_acquire);
    }
    auto* sym = reinterpret_cast<ExecutorSymbolizeGuestPcFn>(
        dlsym(RTLD_DEFAULT, "executor_lsx4_android_symbolize_guest_pc"));
    cached.store(sym, std::memory_order_release);
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MUTEX_SYMBOLIZE_RESOLVE] fn=%p weak=%p",
                        reinterpret_cast<void*>(sym),
                        reinterpret_cast<void*>(executor_lsx4_android_symbolize_guest_pc));
    return sym;
}

static std::string SymbolizeOwnerPcForLog(u64 rip) {
    auto* symbolize = ResolveSymbolizeGuestPc();
    if (!symbolize) {
        char fallback[96]{};
        std::snprintf(fallback, sizeof(fallback), "module=<no-symbolizer> moduleOff=0x0 symbol=<none>");
        return fallback;
    }

    char module_name[192]{};
    char symbol_name[192]{};
    u64 module_base = 0;
    u64 module_offset = 0;
    u64 symbol_offset = 0;
    u64 symbol_delta = 0;
    const int rc = symbolize(rip, module_name, sizeof(module_name), &module_base, &module_offset,
                             symbol_name, sizeof(symbol_name), &symbol_offset, &symbol_delta);
    char line[640]{};
    std::snprintf(line, sizeof(line),
                  "symRc=%d module=%s moduleBase=0x%llx moduleOff=0x%llx symbol=%s "
                  "symbolOff=0x%llx symbolDelta=0x%llx",
                  rc, module_name[0] ? module_name : "<none>",
                  static_cast<unsigned long long>(module_base),
                  static_cast<unsigned long long>(module_offset),
                  symbol_name[0] ? symbol_name : "<none>",
                  static_cast<unsigned long long>(symbol_offset),
                  static_cast<unsigned long long>(symbol_delta));
    return line;
}

static bool ShouldRelaxGuestMutex(void* mutex) {
    (void)mutex;
    return false;
}

static bool IsLiveUnityThread() {
    if (!g_curthread) {
        return false;
    }
    const auto& name = g_curthread->name;
    return name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("Game:Main") != std::string::npos ||
           name.find("UnityPreload") != std::string::npos ||
           name.find("mono") != std::string::npos ||
           name.find("Submit Done Thread") != std::string::npos;
}

static Pthread* CurrentMutexThread();
static const char* PthreadName(const Pthread* thread);
static bool IsLiveDirectMemoryMutexCell(const void* mutex);
static bool ThreadNameContains(const Pthread* thread, const char* needle);

static bool IsLiveUnityOrMonoThreadName(const Pthread* thread) {
    if (thread == nullptr) {
        return false;
    }
    const auto& name = thread->name;
    return name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("Game:Main") != std::string::npos ||
           name.find("UnityPreload") != std::string::npos ||
           name.find("mono") != std::string::npos ||
           name.find("Mono") != std::string::npos ||
           name.find("Submit Done Thread") != std::string::npos;
}

static bool IsUnityPreloadMutexYieldSentinelEnabled() {
    static const bool enabled = [] {
        constexpr const char* PathA =
            "/data/user/0/app.lsx4.android/files/lsx4-home/"
            "run-live-yield-unity-preload-mutex";
        constexpr const char* PathB =
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-live-yield-unity-preload-mutex";
        const int rc_a = access(PathA, F_OK);
        const int errno_a = errno;
        const int rc_b = access(PathB, F_OK);
        const int errno_b = errno;
        __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                            "[EXECUTOR_LIVE_YIELD_SENTINEL] pathA_rc=%d pathA_errno=%d "
                            "pathB_rc=%d pathB_errno=%d enabled=%d",
                            rc_a, errno_a, rc_b, errno_b, rc_a == 0 || rc_b == 0);
        return rc_a == 0 || rc_b == 0;
    }();
    return enabled;
}

static bool ShouldYieldUnityPreloadMutexTryLock(PthreadMutexT* guest, PthreadMutex* native,
                                                s32 ret) {
    if (ret != POSIX_EBUSY || native == nullptr || native->m_owner == nullptr ||
        !IsUnityPreloadMutexYieldSentinelEnabled()) {
        return false;
    }
    const auto guest_addr = reinterpret_cast<std::uintptr_t>(guest);
    if (guest_addr < 0x200000000ULL || guest_addr >= 0x300000000ULL) {
        return false;
    }
    return native->m_owner->name.find("UnityPreload") != std::string::npos;
}

static bool IsMonoAbbaTrylockSignalHelperEnabled() {
    static const bool enabled = [] {
        constexpr const char* PathA =
            "/data/user/0/app.lsx4.android/files/lsx4-home/"
            "run-live-mono-abba-trylock-helper";
        constexpr const char* PathB =
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-live-mono-abba-trylock-helper";
        const int rc_a = access(PathA, F_OK);
        const int errno_a = errno;
        const int rc_b = access(PathB, F_OK);
        const int errno_b = errno;
        const bool value = rc_a == 0 || rc_b == 0;
        __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                            "[EXECUTOR_MONO_ABBA_HELPER_SENTINEL] pathA_rc=%d "
                            "pathA_errno=%d pathB_rc=%d pathB_errno=%d lightOracle=%d "
                            "enabled=%d",
                            rc_a, errno_a, rc_b, errno_b, ExecutorLightOracleMode() ? 1 : 0,
                            value ? 1 : 0);
        return value;
    }();
    return enabled;
}

static bool IsMonoAbbaPcOracleSignalContractEnabled() {
    static const bool enabled = [] {
        constexpr const char* PathA =
            "/data/user/0/app.lsx4.android/files/lsx4-home/"
            "run-live-mono-abba-trylock-helper";
        constexpr const char* PathB =
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-live-mono-abba-trylock-helper";
        const int rc_a = access(PathA, F_OK);
        const int errno_a = errno;
        const int rc_b = access(PathB, F_OK);
        const int errno_b = errno;
        const bool helper_marker_enabled = rc_a == 0 || rc_b == 0;
        const bool signal_delivery_enabled =
            std::getenv("EXECUTOR_FEX_PS4_SIGNAL_FRAME") != nullptr;
        const bool fex_orbis_signal_frame = signal_delivery_enabled &&
            std::getenv("EXECUTOR_FEX_OWNS_FAULT_SIGNALS") != nullptr;
        const bool jit_signal_frame =
            executor_lsx4_android_jit_active != nullptr &&
            executor_lsx4_android_jit_active();
        const bool guest_signal_frame_available =
            fex_orbis_signal_frame || jit_signal_frame;
        const bool explicit_contract =
            std::getenv("EXECUTOR_ENABLE_UNITYPRELOAD_CONTROL_SIGNAL_CONTRACT") != nullptr;
        const bool abba_standalone =
            access("/data/data/app.lsx4.android/files/lsx4-home/run-abba-standalone",
                   F_OK) == 0 ||
            access("/data/user/0/app.lsx4.android/files/lsx4-home/run-abba-standalone",
                   F_OK) == 0;
        const bool value = explicit_contract && guest_signal_frame_available &&
                           (ExecutorLightOracleMode() || abba_standalone);
        __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                            "[EXECUTOR_MONO_ABBA_CONTRACT] enabled=%d lightOracle=%d "
                            "helperMarker=%d signalDelivery=%d pathA_rc=%d pathA_errno=%d pathB_rc=%d "
                            "pathB_errno=%d fexOrbisSignalFrame=%d envSignalDeliver=%d "
                            "jitSignalFrame=%d explicitContract=%d "
                            "source=opt_in_real_signal_contract noSyntheticAck=1",
                            value ? 1 : 0, ExecutorLightOracleMode() ? 1 : 0,
                            helper_marker_enabled ? 1 : 0, signal_delivery_enabled ? 1 : 0,
                            rc_a, errno_a, rc_b, errno_b,
                            fex_orbis_signal_frame ? 1 : 0,
                            std::getenv("EXECUTOR_LIVE_MONO_SIGNAL_DELIVER") != nullptr ? 1 : 0,
                            jit_signal_frame ? 1 : 0,
                            explicit_contract ? 1 : 0);
        return value;
    }();
    return enabled;
}

static void TraceLightOracleMutexContentionSite(PthreadMutexT* guest, PthreadMutex* native,
                                                s32 ret) {
    if (!ExecutorLightOracleMode() || ret != POSIX_EBUSY || native == nullptr ||
        native->m_owner == nullptr ||
        !IsLiveDirectMemoryMutexCell(guest)) {
        return;
    }
    Pthread* cur = g_curthread != nullptr ? g_curthread : CurrentMutexThread();
    if (!IsLiveUnityOrMonoThreadName(cur) || !IsLiveUnityOrMonoThreadName(native->m_owner)) {
        return;
    }

    static std::atomic_int budget{48};
    const int previous = budget.fetch_sub(1, std::memory_order_relaxed);
    if (previous <= 0) {
        return;
    }

    u64 guest_return = 0;
    u64 return_off = 0;
    u64 hle_arg0 = 0;
    char hle_symbol[96]{};
    char hle_module[64]{};
    const int hle_rc = executor_live_get_current_hle_call_site
                           ? executor_live_get_current_hle_call_site(
                                 &guest_return, &return_off, &hle_arg0, hle_symbol,
                                 sizeof(hle_symbol), hle_module, sizeof(hle_module))
                           : -2;
    __android_log_print(
        ANDROID_LOG_WARN, "LSX4Mutex",
        "[EXECUTOR_LIGHT_MUTEX_CONTENTION_SITE] idx=%d guest=%p native=%p ret=%d cur=%p "
        "curName=%s owner=%p ownerName=%s hleRc=%d hleRet=%p returnOff=0x%llx "
        "hleModule=%s hleSymbol=%s hleArg0=%p",
        48 - previous + 1, guest, native, ret, cur, PthreadName(cur), native->m_owner,
        PthreadName(native->m_owner), hle_rc, reinterpret_cast<void*>(guest_return),
        static_cast<unsigned long long>(return_off), hle_module[0] ? hle_module : "<none>",
        hle_symbol[0] ? hle_symbol : "<none>", reinterpret_cast<void*>(hle_arg0));
}

static bool ShouldTolerateUnityOwnerMismatchUnlock(void* mutex, PthreadMutex* native) {
    (void)mutex;
    (void)native;
    return false;
}

static bool ShouldTolerateUnityUnlockedUnlock(void* mutex, PthreadMutex* native) {
    (void)mutex;
    (void)native;
    return false;
}

static Pthread* CurrentMutexThread();
static bool IsOwnedByCurrentHostThread(const PthreadMutex* mutex);
static bool HandleMonoSuspendTryLockAbbaCycle(void* mutex, PthreadMutex* native, s32 ret);

static bool ShouldBypassMonoSuspendTryLock(void* mutex, PthreadMutex* native, s32 ret) {
    (void)mutex;
    (void)native;
    (void)ret;
    return false;
}

static bool ShouldBypassMonoSuspendOwnerMismatchUnlock(void* mutex, PthreadMutex* native) {
    (void)mutex;
    (void)native;
    return false;
}

static Pthread* CurrentMutexThread();

static long CurrentHostTid() {
    return static_cast<long>(syscall(__NR_gettid));
}

static std::uintptr_t CurrentNativePthreadHandle() {
    return static_cast<std::uintptr_t>(pthread_self());
}

static std::uintptr_t NativePthreadHandle(Pthread* thread) {
    return thread != nullptr ? thread->native_thr.GetHandle() : 0;
}

static bool IsOwnedByCurrentHostThread(const PthreadMutex* mutex) {
    (void)mutex;
    return false;
}

static const char* PthreadStateName(const Pthread* thread) {
    if (thread == nullptr) {
        return "<none>";
    }
    switch (thread->state) {
    case PthreadState::Running:
        return "running";
    case PthreadState::Dead:
        return "dead";
    default:
        return "unknown";
    }
}

static const char* PthreadName(const Pthread* thread) {
    return thread != nullptr && !thread->name.empty() ? thread->name.c_str() : "<none>";
}

static long PthreadGuestTid(const Pthread* thread) {
    return thread != nullptr ? static_cast<long>(thread->tid.load(std::memory_order_relaxed)) : 0;
}

static bool IsLiveDirectMemoryMutexCell(const void* mutex) {
    const auto address = reinterpret_cast<std::uintptr_t>(mutex);
    return address >= 0x200000000ULL && address < 0x300000000ULL;
}

static bool IsLiveMonoRuntimeMutexCell(const void* mutex) {
    const auto address = reinterpret_cast<std::uintptr_t>(mutex);
    return address == 0x803397ca0ULL ||
           (address >= 0x803380000ULL && address < 0x8033c0000ULL);
}

static bool ThreadNameContains(const Pthread* thread, const char* needle) {
    return thread != nullptr && needle != nullptr && thread->name.find(needle) != std::string::npos;
}

static bool ShouldTraceUnityMutexOwnerEvent(PthreadMutexT* guest, PthreadMutex* native, s32 ret) {
    static const bool enabled =
        std::getenv("EXECUTOR_TRACE_LIVE_UNITY_MUTEX_OWNER") != nullptr;
    if (!enabled || !IsLiveDirectMemoryMutexCell(guest)) {
        return false;
    }

    const Pthread* cur = CurrentMutexThread();
    if (ret == 0 && ThreadNameContains(cur, "Game:Main") &&
        (native == nullptr || native->m_owner == nullptr ||
         ThreadNameContains(native->m_owner, "Game:Main"))) {
        return false;
    }

    if (ThreadNameContains(cur, "UnityPreload") || ThreadNameContains(cur, "Game:Main") ||
        ThreadNameContains(cur, "Thread2") || ThreadNameContains(cur, "UnityGfxDeviceWorker") ||
        ThreadNameContains(cur, "SceFios")) {
        return true;
    }

    if (native != nullptr && native->m_owner != nullptr &&
        (ThreadNameContains(native->m_owner, "UnityPreload") ||
         ThreadNameContains(native->m_owner, "Game:Main") ||
         ThreadNameContains(native->m_owner, "Thread2") ||
         ThreadNameContains(native->m_owner, "UnityGfxDeviceWorker") ||
         ThreadNameContains(native->m_owner, "SceFios"))) {
        return true;
    }

    return ret != 0 && IsLiveUnityThread();
}

static void TraceUnityMutexOwnerEvent(const char* op, PthreadMutexT* guest, PthreadMutex* native,
                                      s32 ret) {
    if (!ShouldTraceUnityMutexOwnerEvent(guest, native, ret)) {
        return;
    }

    static std::atomic_int budget{2048};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    const Pthread* cur = CurrentMutexThread();
    const u64 owner_arg = native && native->m_owner ? reinterpret_cast<u64>(native->m_owner->arg)
                                                    : 0;
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MUTEX_OWNER_EVENT] op=%s guest=%p native=%p ret=%d "
                        "cur=%p curName=%s curTid=%ld curHostTid=%ld owner=%p "
                        "ownerName=%s ownerState=%s ownerTid=%ld ownerHostTid=%ld "
                        "ownerArg=0x%llx flags=0x%x type=%u count=%d",
                        op ? op : "?", guest, native, ret, cur, PthreadName(cur),
                        PthreadGuestTid(cur), CurrentHostTid(), native ? native->m_owner : nullptr,
                        PthreadName(native ? native->m_owner : nullptr),
                        PthreadStateName(native ? native->m_owner : nullptr),
                        PthreadGuestTid(native ? native->m_owner : nullptr),
                        native ? native->m_owner_host_tid : 0,
                        static_cast<unsigned long long>(owner_arg),
                        native ? static_cast<u32>(native->m_flags) : 0u,
                        native ? static_cast<u32>(native->Type()) : 0u,
                        native ? native->m_count : 0);
}

static void TraceLiveMonoRuntimeMutexOwnerPc(const char* phase, int count, Pthread* owner,
                                             bool force_light_oracle = false,
                                             long owner_host_tid = 0) {
    if (ExecutorLightOracleMode() && !force_light_oracle) {
        return;
    }
    if (owner == nullptr) {
        return;
    }
    auto* get_guest_thread_regs_by_pthread = ResolveGuestThreadRegsByPthread();
    auto* get_guest_thread_regs_by_tid = ResolveGuestThreadRegsByTid();
    if (get_guest_thread_regs_by_pthread == nullptr && get_guest_thread_regs_by_tid == nullptr) {
        return;
    }

    u64 owner_rip = 0;
    u64 owner_rsp = 0;
    u64 owner_rbp = 0;
    long fex_tid = 0;
    char fex_host_name[32]{};
    const auto owner_pthread = owner->native_thr.GetHandle();
    int pc_rc = -1;
    const char* lookup = "none";
    if (owner_pthread != 0 && get_guest_thread_regs_by_pthread != nullptr) {
        pc_rc = get_guest_thread_regs_by_pthread(owner_pthread, &owner_rip, &owner_rsp,
                                                 &owner_rbp, &fex_tid, fex_host_name,
                                                 sizeof(fex_host_name));
        lookup = "pthread";
    }
    if (pc_rc != 0 && owner_host_tid != 0 && get_guest_thread_regs_by_tid != nullptr) {
        pc_rc = get_guest_thread_regs_by_tid(owner_host_tid, &owner_rip, &owner_rsp, &owner_rbp,
                                             &fex_tid, fex_host_name, sizeof(fex_host_name));
        lookup = "host_tid";
    }
    const std::string owner_symbol = SymbolizeOwnerPcForLog(owner_rip);
    __android_log_print(
        ANDROID_LOG_WARN, "LSX4Native",
        "[EXECUTOR_LIVE_MONO_MUTEX_OWNER_PC] phase=%s count=%d rc=%d lookup=%s ownerName=%s "
        "pthread=0x%zx ownerHostTid=%ld fexTid=%ld fexHost=%s rip=0x%llx rsp=0x%llx rbp=0x%llx "
        "ebootOff=0x%llx %s",
        phase ? phase : "?", count, pc_rc, lookup, PthreadName(owner),
        static_cast<std::size_t>(owner_pthread), owner_host_tid, fex_tid,
        fex_host_name[0] ? fex_host_name : "<none>",
        static_cast<unsigned long long>(owner_rip),
        static_cast<unsigned long long>(owner_rsp),
        static_cast<unsigned long long>(owner_rbp),
        static_cast<unsigned long long>(owner_rip >= 0x800000000ULL ? owner_rip - 0x800000000ULL
                                                                    : 0),
        owner_symbol.c_str());
}

static void TraceLiveMonoRuntimeMutex(const char* phase, PthreadMutexT* guest, PthreadMutex* native,
                                      s32 ret = 0) {
    if (ExecutorLightOracleMode() || !ExecutorVerboseMonoMutexLedgerEnabled()) {
        return;
    }
    if (!g_curthread || !IsLiveMonoRuntimeMutexCell(guest)) {
        return;
    }

    const auto guest_addr = reinterpret_cast<std::uintptr_t>(guest);
    const bool current_unitypreload = g_curthread->name == "UnityPreload";
    const bool target_global_lock = guest_addr == 0x803397ca0ULL;
    const bool has_interesting_owner =
        native != nullptr && native->m_owner != nullptr &&
        (ThreadNameContains(native->m_owner, "Game:Main") ||
         ThreadNameContains(native->m_owner, "UnityPreload") ||
         ThreadNameContains(native->m_owner, "mono"));
    if (!current_unitypreload && !has_interesting_owner) {
        return;
    }

    int count = 0;
    if (target_global_lock) {
        static std::atomic_int target_budget{1024};
        const int previous = target_budget.fetch_sub(1, std::memory_order_relaxed);
        if (previous <= 0) {
            return;
        }
        count = 1024 - previous + 1;
    } else {
        static std::atomic_int general_budget{128};
        const int previous = general_budget.fetch_sub(1, std::memory_order_relaxed);
        if (previous <= 0) {
            return;
        }
        count = 128 - previous + 1;
    }

    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MONO_MUTEX] phase=%s count=%d thread=%s guest=%p "
                        "native=%p ret=%d owner=%p ownerName=%s ownerState=%s ownerTid=%ld "
                        "ownerHostTid=%ld cur=%p curTid=%ld flags=0x%x type=%u lockCount=%d",
                        phase ? phase : "?", count,
                        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>", guest,
                        native, ret, native ? native->m_owner : nullptr,
                        PthreadName(native ? native->m_owner : nullptr),
                        PthreadStateName(native ? native->m_owner : nullptr),
                        PthreadGuestTid(native ? native->m_owner : nullptr),
                        native ? native->m_owner_host_tid : 0, CurrentMutexThread(),
                        PthreadGuestTid(CurrentMutexThread()),
                        native ? static_cast<u32>(native->m_flags) : 0u,
                        native ? static_cast<u32>(native->Type()) : 0u,
                        native ? native->m_count : 0);
    if (native != nullptr && native->m_owner != nullptr) {
        TraceLiveMonoRuntimeMutexOwnerPc(phase, count, native->m_owner, false,
                                         native->m_owner_host_tid);
    }
}

static void TraceMonoRuntimeMutexLedger(const char* op, const char* phase, PthreadMutexT* guest,
                                         PthreadMutex* native, s32 ret = 0) {
    const bool verbose = ExecutorVerboseMonoMutexLedgerEnabled();
    if (!ExecutorLightOracleMode() && !verbose) {
        return;
    }
    if (!g_curthread || !IsLiveMonoRuntimeMutexCell(guest)) {
        return;
    }

    const auto guest_addr = reinterpret_cast<std::uintptr_t>(guest);
    const bool target_global_lock = guest_addr == 0x803397ca0ULL;
    constexpr int TargetLedgerBudget = 4096;
    constexpr int GeneralLedgerBudget = 512;
    if (ExecutorLightOracleMode() && !verbose) {
        static std::atomic_int light_target_budget{96};
        static std::atomic_int light_general_budget{8};
        const int previous = target_global_lock
                                 ? light_target_budget.fetch_sub(1, std::memory_order_relaxed)
                                 : light_general_budget.fetch_sub(1, std::memory_order_relaxed);
        if (previous <= 0) {
            return;
        }
    }
    static std::atomic_int target_budget{TargetLedgerBudget};
    static std::atomic_int general_budget{GeneralLedgerBudget};
    const int previous = target_global_lock
                             ? target_budget.fetch_sub(1, std::memory_order_relaxed)
                             : general_budget.fetch_sub(1, std::memory_order_relaxed);
    if (previous <= 0) {
        return;
    }
    const int index = (target_global_lock ? TargetLedgerBudget : GeneralLedgerBudget) - previous + 1;
    u64 guest_return = 0;
    u64 return_off = 0;
    u64 hle_arg0 = 0;
    char hle_symbol[96]{};
    char hle_module[64]{};
    const int hle_rc = executor_live_get_current_hle_call_site
                           ? executor_live_get_current_hle_call_site(
                                 &guest_return, &return_off, &hle_arg0, hle_symbol,
                                 sizeof(hle_symbol), hle_module, sizeof(hle_module))
                           : -2;
    __android_log_print(
        ANDROID_LOG_WARN, "LSX4Native",
        "[EXECUTOR_MONO_RUNTIME_MUTEX_LEDGER] idx=%d op=%s phase=%s thread=%s guest=%p native=%p "
        "ret=%d owner=%p ownerName=%s ownerState=%s ownerTid=%ld ownerHostTid=%ld cur=%p "
        "curTid=%ld curHostTid=%ld flags=0x%x type=%u count=%d hleRc=%d hleRet=%p "
        "returnOff=0x%llx hleModule=%s hleSymbol=%s hleArg0=%p",
        index, op ? op : "?", phase ? phase : "?", g_curthread->name.c_str(), guest, native, ret,
        native ? native->m_owner : nullptr, PthreadName(native ? native->m_owner : nullptr),
        PthreadStateName(native ? native->m_owner : nullptr),
        PthreadGuestTid(native ? native->m_owner : nullptr),
        native ? native->m_owner_host_tid : 0, CurrentMutexThread(),
        PthreadGuestTid(CurrentMutexThread()), CurrentHostTid(),
        native ? static_cast<u32>(native->m_flags) : 0u,
        native ? static_cast<u32>(native->Type()) : 0u, native ? native->m_count : 0, hle_rc,
        reinterpret_cast<void*>(guest_return), static_cast<unsigned long long>(return_off),
        hle_module[0] ? hle_module : "<none>", hle_symbol[0] ? hle_symbol : "<none>",
        reinterpret_cast<void*>(hle_arg0));
}

static u64 GuestReadQword(u64 address) {
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || address == 0 || !memory->IsValidMapping(address, sizeof(u64))) {
        return 0;
    }
    u64 value = 0;
    memory->CopySparseMemory(address, reinterpret_cast<u8*>(&value), sizeof(value));
    return value;
}

static u32 GuestReadU32(u64 address) {
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || address == 0 || !memory->IsValidMapping(address, sizeof(u32))) {
        return 0;
    }
    u32 value = 0;
    memory->CopySparseMemory(address, reinterpret_cast<u8*>(&value), sizeof(value));
    return value;
}

static u8 GuestReadByte(u64 address) {
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || address == 0 || !memory->IsValidMapping(address, sizeof(u8))) {
        return 0;
    }
    u8 value = 0;
    memory->CopySparseMemory(address, &value, sizeof(value));
    return value;
}

static std::string GuestBytesHex(u64 address, std::size_t size) {
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || address == 0 || size == 0 || !memory->IsValidMapping(address, size)) {
        return "<unmapped>";
    }

    std::vector<u8> bytes(size);
    memory->CopySparseMemory(address, bytes.data(), size);
    static constexpr char Hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(size * 3);
    for (std::size_t i = 0; i < bytes.size(); i++) {
        if (i != 0) {
            out.push_back(' ');
        }
        out.push_back(Hex[bytes[i] >> 4]);
        out.push_back(Hex[bytes[i] & 0xf]);
    }
    return out;
}

static void LogMonoScriptCandidateObject(const char* label, int count, u64 ptr) {
    if (ptr == 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_MONOSCRIPT_CANDIDATE] count=%d label=%s ptr=0x0",
                            count, label ? label : "?");
        return;
    }
    auto* memory = Core::Memory::Instance();
    const bool mapped = memory != nullptr && memory->IsValidMapping(ptr, 0x110);
    __android_log_print(
        ANDROID_LOG_WARN, "LSX4Native",
        "[EXECUTOR_LIVE_MONOSCRIPT_CANDIDATE] count=%d label=%s ptr=0x%llx mapped=%d "
        "q00=0x%llx q08=0x%llx q10=0x%llx q18=0x%llx q20=0x%llx q28=0x%llx "
        "q30=0x%llx q38=0x%llx q50=0x%llx q88=0x%llx qa0=0x%llx qb0=0x%llx "
        "d08=0x%x d0c=0x%x b0e=0x%x b50=0x%x b51=0x%x b103=0x%x",
        count, label ? label : "?", static_cast<unsigned long long>(ptr), mapped ? 1 : 0,
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x00)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x08)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x10)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x18)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x20)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x28)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x30)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x38)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x50)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0x88)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0xa0)),
        static_cast<unsigned long long>(GuestReadQword(ptr + 0xb0)), GuestReadU32(ptr + 0x08),
        GuestReadU32(ptr + 0x0c), GuestReadByte(ptr + 0x0e), GuestReadByte(ptr + 0x50),
        GuestReadByte(ptr + 0x51), GuestReadByte(ptr + 0x103));
}

static std::mutex g_mono_suspend_stolen_mutex_lock;
static std::unordered_map<PthreadMutex*, Pthread*> g_mono_suspend_stolen_mutexes;

static void RecordMonoSuspendStolenMutex(PthreadMutex* native) {
    Pthread* cur = g_curthread != nullptr ? g_curthread : CurrentMutexThread();
    if (native == nullptr || cur == nullptr) {
        return;
    }
    std::scoped_lock lock(g_mono_suspend_stolen_mutex_lock);
    g_mono_suspend_stolen_mutexes[native] = cur;
}

static bool ConsumeMonoSuspendStolenMutex(PthreadMutex* native) {
    Pthread* cur = g_curthread != nullptr ? g_curthread : CurrentMutexThread();
    if (native == nullptr || cur == nullptr) {
        return false;
    }
    std::scoped_lock lock(g_mono_suspend_stolen_mutex_lock);
    const auto it = g_mono_suspend_stolen_mutexes.find(native);
    if (it == g_mono_suspend_stolen_mutexes.end() || it->second != cur) {
        return false;
    }
    g_mono_suspend_stolen_mutexes.erase(it);
    return true;
}

static void ForgetMonoSuspendStolenMutex(PthreadMutex* native) {
    if (native == nullptr) {
        return;
    }
    std::scoped_lock lock(g_mono_suspend_stolen_mutex_lock);
    g_mono_suspend_stolen_mutexes.erase(native);
}

struct MutexWaitSnapshot {
    PthreadMutex* waiting_on = nullptr;
    void* guest_cell = nullptr;
    Pthread* owner_snapshot = nullptr;
    std::chrono::steady_clock::time_point wait_started{};
    std::chrono::steady_clock::time_point next_report{};
    u64 generation_at_start = 0;
    u32 report_count = 0;
};

static std::mutex g_mutex_wait_snapshot_lock;
static std::unordered_map<Pthread*, MutexWaitSnapshot> g_mutex_wait_snapshots;

static std::mutex g_mono_suspend_pending_signal_lock;
static std::unordered_set<Pthread*> g_mono_suspend_pending_sigusr1;

static std::mutex g_mono_suspend_abba_resume_lock;
static std::unordered_set<Pthread*> g_mono_suspend_abba_resume_escape;
static std::unordered_set<Pthread*> g_mono_suspend_preload_peer_signaled;
static std::unordered_set<Pthread*> g_mono_suspend_resume_event_waiters;
static std::unordered_set<void*> g_mono_suspend_unity_control_signaled;

static void MarkMonoSuspendSigusr1Pending(Pthread* target) {
    if (target == nullptr) {
        return;
    }
    std::scoped_lock lock(g_mono_suspend_pending_signal_lock);
    g_mono_suspend_pending_sigusr1.insert(target);
}

extern "C" void executor_live_mono_mark_pending_sigusr1_for_thread(void* guest_thread) {
    auto* target = static_cast<Pthread*>(guest_thread);
    MarkMonoSuspendSigusr1Pending(target);
    static std::atomic_int log_budget{256};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                            "[EXECUTOR_MONO_PENDING_SIGNAL] action=mark target=%p "
                            "targetName=%s",
                            target, PthreadName(target));
    }
}

extern "C" void executor_live_mono_note_resume_event_waiter(void* guest_thread,
                                                            const char* event_name) {
    auto* target = static_cast<Pthread*>(guest_thread);
    if (target == nullptr || event_name == nullptr ||
        std::strcmp(event_name, "resumeEvent") != 0 ||
        (!ThreadNameContains(target, "UnityPreload") &&
         !ThreadNameContains(target, "mono thread") &&
         !ThreadNameContains(target, "Thread2"))) {
        return;
    }
    {
        std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
        g_mono_suspend_resume_event_waiters.insert(target);
    }
    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                            "[EXECUTOR_MONO_RESUME_EVENT_WAITER] target=%p targetName=%s "
                            "event=%s reason=pc_oracle_staged_peer_signal",
                            target, PthreadName(target), event_name);
    }
}

static bool IsMonoSuspendResumeEventWaiter(Pthread* target) {
    if (target == nullptr) {
        return false;
    }
    std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
    return g_mono_suspend_resume_event_waiters.contains(target);
}

static void AcknowledgeMonoSuspendSigusr1FromMutex(Pthread* target, int queue_rc) {
    if (target == nullptr) {
        return;
    }

    const bool suspend_acked = ExecutorSignalKernelSemaByNameForLiveMono("SuspendSemaphore", 1);
    const bool resume_acked = ExecutorSignalKernelSemaByNameForLiveMono("ResumeSemaphore", 1);
    const int posix_acked =
        executor_live_signal_pending_mono_posix_sems ? executor_live_signal_pending_mono_posix_sems(2)
                                                     : -1;
    static std::atomic_int log_budget{128};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Mutex",
            "[EXECUTOR_MONO_MUTEX_SIGNAL_ACK] target=%p targetName=%s queueRc=%d "
            "suspendAck=%d resumeAck=%d posixAck=%d reason=mutex-alertable-sigusr1",
            target, PthreadName(target), queue_rc, suspend_acked ? 1 : 0, resume_acked ? 1 : 0,
            posix_acked);
    }
}

static void MarkMonoSuspendAbbaResumeEscape(Pthread* target) {
    if (target == nullptr) {
        return;
    }
    std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
    g_mono_suspend_abba_resume_escape.insert(target);
}

static bool TryMarkMonoSuspendAbbaSignalActive(Pthread* target) {
    if (target == nullptr) {
        return false;
    }
    {
        std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
        const auto [it, inserted] = g_mono_suspend_abba_resume_escape.insert(target);
        (void)it;
        if (!inserted) {
            static std::atomic_int log_budget{64};
            if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                    "[EXECUTOR_MONO_ABBA_SIGNAL_COALESCED] target=%p "
                                    "targetName=%s reason=signal_already_active",
                                    target, PthreadName(target));
            }
            return false;
        }
    }
    return true;
}

static void ClearMonoSuspendAbbaSignalState(Pthread* target) {
    if (target == nullptr) {
        return;
    }
    {
        std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
        g_mono_suspend_abba_resume_escape.erase(target);
        g_mono_suspend_preload_peer_signaled.erase(target);
        g_mono_suspend_resume_event_waiters.erase(target);
    }
    {
        std::scoped_lock lock(g_mono_suspend_pending_signal_lock);
        g_mono_suspend_pending_sigusr1.erase(target);
    }
}

static bool TryMarkUnityPreloadControlSignalActive(void* mutex) {
    if (mutex == nullptr) {
        return false;
    }
    std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
    const auto [it, inserted] = g_mono_suspend_unity_control_signaled.insert(mutex);
    (void)it;
    if (!inserted) {
        static std::atomic_int log_budget{64};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MONO_ABBA_SIGNAL_COALESCED] guest=%p "
                                "reason=unitypreload_control_mutex_already_signaled",
                                mutex);
        }
        return false;
    }
    return true;
}

static void ClearUnityPreloadControlSignalState(void* mutex) {
    if (mutex == nullptr || !IsMonoAbbaPcOracleSignalContractEnabled()) {
        return;
    }
    std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
    g_mono_suspend_unity_control_signaled.erase(mutex);
}

extern "C" bool executor_live_mono_abba_consume_resume_event_escape(const char* event_name,
                                                                    std::uint64_t bits) {
    (void)event_name;
    (void)bits;
    static std::atomic_int log_budget{8};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                            "[EXECUTOR_MONO_ABBA_RESUME_ESCAPE] disabled=pc_oracle_real_ack");
    }
    return false;
}

static bool ConsumeMonoSuspendSigusr1Pending(Pthread* target) {
    if (target == nullptr) {
        return false;
    }
    std::scoped_lock lock(g_mono_suspend_pending_signal_lock);
    const auto it = g_mono_suspend_pending_sigusr1.find(target);
    if (it == g_mono_suspend_pending_sigusr1.end()) {
        return false;
    }
    g_mono_suspend_pending_sigusr1.erase(it);
    return true;
}

static Pthread* ConsumeMonoSuspendSigusr1PendingForCurrent(Pthread* curthread) {
    const auto current_native = CurrentNativePthreadHandle();
    std::scoped_lock lock(g_mono_suspend_pending_signal_lock);

    if (curthread != nullptr) {
        const auto exact = g_mono_suspend_pending_sigusr1.find(curthread);
        if (exact != g_mono_suspend_pending_sigusr1.end()) {
            Pthread* target = *exact;
            g_mono_suspend_pending_sigusr1.erase(exact);
            return target;
        }
    }

    for (auto it = g_mono_suspend_pending_sigusr1.begin();
         it != g_mono_suspend_pending_sigusr1.end(); ++it) {
        Pthread* target = *it;
        if (target == nullptr || NativePthreadHandle(target) == 0 ||
            NativePthreadHandle(target) != current_native) {
            continue;
        }
        g_mono_suspend_pending_sigusr1.erase(it);
        static std::atomic_int log_budget{64};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MONO_PENDING_SIGNAL] action=consume_native "
                                "cur=%p curName=%s target=%p targetName=%s native=0x%zx",
                                curthread, PthreadName(curthread), target, PthreadName(target),
                                static_cast<std::size_t>(current_native));
        }
        return target;
    }

    if (!g_mono_suspend_pending_sigusr1.empty()) {
        static std::atomic_int log_budget{64};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            Pthread* pending = *g_mono_suspend_pending_sigusr1.begin();
            __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                                "[EXECUTOR_MONO_PENDING_SIGNAL] action=skip_current "
                                "cur=%p curName=%s curNative=0x%zx pending=%p "
                                "pendingName=%s pendingNative=0x%zx pendingCount=%zu",
                                curthread, PthreadName(curthread),
                                static_cast<std::size_t>(current_native), pending,
                                PthreadName(pending),
                                static_cast<std::size_t>(NativePthreadHandle(pending)),
                                g_mono_suspend_pending_sigusr1.size());
        }
    }
    return nullptr;
}

enum class MonoSignalSafePointResult {
    None,
    HandledContinue,
};

#ifdef __ANDROID__
static thread_local int t_live_held_guest_mutexes = 0;
inline void ExecutorNoteGuestMutexAcquired(PthreadMutex* mutex) {
    ++t_live_held_guest_mutexes;
    if (mutex != nullptr && ExecutorStallMutexLedgerEnabled()) {
        mutex->m_lock_generation.fetch_add(1, std::memory_order_relaxed);
    }
}
inline void ExecutorNoteGuestMutexReleased(PthreadMutex* mutex) {
    if (t_live_held_guest_mutexes > 0) {
        --t_live_held_guest_mutexes;
    }
    if (mutex != nullptr && ExecutorStallMutexLedgerEnabled()) {
        mutex->m_lock_generation.fetch_add(1, std::memory_order_relaxed);
    }
}
static bool ExecutorLiveMonoSafepointSignalEnabled() {
    static const bool enabled =
        std::getenv("EXECUTOR_LIVE_MONO_SAFEPOINT_SIGNAL") != nullptr;
    return enabled;
}

static bool ExecutorAlertableMutexPollingEnabled() {
    static const bool enabled = std::getenv("EXECUTOR_LIVE_MONO_SAFEPOINT_SIGNAL") != nullptr ||
                                std::getenv("EXECUTOR_DIAGNOSTIC_ALERTABLE_MUTEX_POLL") != nullptr;
    return enabled;
}
#endif

static MonoSignalSafePointResult RunPendingMonoSuspendSigusr1AtSafePoint(Pthread* curthread,
                                                                        void* guest_cell) {
    Pthread* target = ConsumeMonoSuspendSigusr1PendingForCurrent(curthread);
    if (target == nullptr) {
        return MonoSignalSafePointResult::None;
    }
#ifdef __ANDROID__
    const bool running_own_suspend =
        (curthread == target) || (CurrentNativePthreadHandle() == NativePthreadHandle(target));
    static thread_local std::chrono::steady_clock::time_point t_defer_since{};
    if (t_live_held_guest_mutexes > 0 && !running_own_suspend) {
        const auto now = std::chrono::steady_clock::now();
        if (t_defer_since == std::chrono::steady_clock::time_point{}) {
            t_defer_since = now;
        }
        if (now - t_defer_since < std::chrono::seconds(2)) {
            MarkMonoSuspendSigusr1Pending(target);
            static std::atomic_int defer_log_budget{64};
            if (defer_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                                    "[EXECUTOR_MONO_SUSPEND_DEFER] thread=%s target=%s heldMutexes=%d "
                                    "reason=holds_runtime_lock",
                                    PthreadName(curthread), PthreadName(target),
                                    t_live_held_guest_mutexes);
            }
            return MonoSignalSafePointResult::None;
        }
        static std::atomic_int escape_log_budget{32};
        if (escape_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MONO_SUSPEND_DEFER_ESCAPE] thread=%s target=%s "
                                "heldMutexes=%d reason=defer_exceeded_2s_running_anyway",
                                PthreadName(curthread), PthreadName(target),
                                t_live_held_guest_mutexes);
        }
        t_defer_since = {};
    } else {
        t_defer_since = {};
    }
#endif

    const bool explicit_mutex_safe_point =
        (ThreadNameContains(curthread, "UnityPreload") || ThreadNameContains(target, "UnityPreload")) &&
        IsLiveMonoRuntimeMutexCell(guest_cell);
#ifdef __ANDROID__
    const bool running_own_suspend_hle =
        (curthread == target) || (CurrentNativePthreadHandle() == NativePthreadHandle(target));
#else
    const bool running_own_suspend_hle = (curthread == target);
#endif
    const bool hle_safe_point =
        running_own_suspend_hle ||
        (curthread != nullptr && (ThreadNameContains(curthread, "UnityPreload") ||
                                  ThreadNameContains(curthread, "mono thread"))) ||
        ThreadNameContains(target, "UnityPreload") || ThreadNameContains(target, "mono thread");
    if (!explicit_mutex_safe_point && !hle_safe_point) {
        MarkMonoSuspendSigusr1Pending(target);
        return MonoSignalSafePointResult::None;
    }

    const auto handler = Handlers[POSIX_SIGUSR1];
    if (!handler || !executor_lsx4_android_run_guest_signal_handler) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                            "[EXECUTOR_MONO_ALERTABLE_SIGNAL] result=unavailable thread=%s "
                            "handler=%p run=%p guest=%p",
                            PthreadName(target), reinterpret_cast<void*>(handler),
                            reinterpret_cast<void*>(executor_lsx4_android_run_guest_signal_handler),
                            guest_cell);
        MarkMonoSuspendSigusr1Pending(target);
        return MonoSignalSafePointResult::None;
    }

    ExecutorFexGuestRegsMirror regs{};
    int regs_rc = -1;
    const auto current_native = CurrentNativePthreadHandle();
    const auto target_native = NativePthreadHandle(target);
    const bool running_on_target = curthread == target || current_native == target_native;
    const char* regs_source = "none";
    if (auto* get_full_regs = ResolveGuestThreadFullRegsByPthread()) {
        long fex_tid = 0;
        char fex_host_name[32]{};
        regs_rc = get_full_regs(target->native_thr.GetHandle(), &regs, sizeof(regs), &fex_tid,
                                fex_host_name, sizeof(fex_host_name));
        if (regs_rc == 0) {
            regs_source = "pthread_full";
        }
    }
    if (regs_rc != 0) {
        if (auto* get_regs = ResolveGuestThreadRegsByPthread()) {
            u64 rbp = 0;
            long fex_tid = 0;
            char fex_host_name[32]{};
            regs_rc = get_regs(target->native_thr.GetHandle(), &regs.rip, &regs.regs[4], &rbp,
                               &fex_tid, fex_host_name, sizeof(fex_host_name));
            regs.regs[5] = rbp;
            if (regs_rc == 0) {
                regs_source = "pthread_min";
            }
        }
    }
    if (regs_rc != 0 && running_on_target && executor_lsx4_android_get_current_guest_regs != nullptr) {
        regs_rc = executor_lsx4_android_get_current_guest_regs(&regs, sizeof(regs));
        if (regs_rc == 0) {
            regs_source = "current";
        }
    }

    auto siginfo = Siginfo{};
    siginfo._si_signo = POSIX_SIGUSR1;
    siginfo._si_code = 0;
    siginfo._si_pid = getpid();
    siginfo._si_uid = getuid();

    auto ctx = Ucontext{};
    ctx.uc_mcontext.mc_rax = regs.regs[0];
    ctx.uc_mcontext.mc_rcx = regs.regs[1];
    ctx.uc_mcontext.mc_rdx = regs.regs[2];
    ctx.uc_mcontext.mc_rbx = regs.regs[3];
    ctx.uc_mcontext.mc_rsp = regs.regs[4];
    ctx.uc_mcontext.mc_rbp = regs.regs[5];
    ctx.uc_mcontext.mc_rsi = regs.regs[6];
    ctx.uc_mcontext.mc_rdi = regs.regs[7];
    ctx.uc_mcontext.mc_r8 = regs.regs[8];
    ctx.uc_mcontext.mc_r9 = regs.regs[9];
    ctx.uc_mcontext.mc_r10 = regs.regs[10];
    ctx.uc_mcontext.mc_r11 = regs.regs[11];
    ctx.uc_mcontext.mc_r12 = regs.regs[12];
    ctx.uc_mcontext.mc_r13 = regs.regs[13];
    ctx.uc_mcontext.mc_r14 = regs.regs[14];
    ctx.uc_mcontext.mc_r15 = regs.regs[15];
    ctx.uc_mcontext.mc_rip = regs.rip;
    ctx.uc_mcontext.mc_fs = static_cast<u16>(regs.fsbase & 0xffffu);
    ctx.uc_mcontext.mc_gs = static_cast<u16>(regs.gsbase & 0xffffu);
    ctx.uc_mcontext.mc_fsbase = regs.fsbase;
    ctx.uc_mcontext.mc_gsbase = regs.gsbase;
    ctx.uc_stack.ss_sp = target ? target->attr.stackaddr_attr : nullptr;
    ctx.uc_stack.ss_size = target ? target->attr.stacksize_attr : 0;
    ctx.uc_stack.ss_flags = 0;
    NormalizeGuestUcontext(ctx, nullptr);

    std::uint64_t guest_result = 0;
    __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                        "[EXECUTOR_MONO_ALERTABLE_SIGNAL] result=call thread=%s guest=%p "
                        "cur=%p curName=%s handler=%p flags=0x%x regsSource=%s regsRc=%d "
                        "rip=0x%llx rsp=0x%llx currentNative=0x%zx targetNative=0x%zx",
                        PthreadName(target), guest_cell, curthread, PthreadName(curthread),
                        reinterpret_cast<void*>(handler), HandlerFlags[POSIX_SIGUSR1],
                        regs_source, regs_rc,
                        static_cast<unsigned long long>(ctx.uc_mcontext.mc_rip),
                        static_cast<unsigned long long>(ctx.uc_mcontext.mc_rsp),
                        static_cast<std::size_t>(current_native),
                        static_cast<std::size_t>(target_native));
    const int rc = executor_lsx4_android_run_guest_signal_handler(
        reinterpret_cast<std::uint64_t>(handler), POSIX_SIGUSR1,
        reinterpret_cast<std::uint64_t>(&ctx), reinterpret_cast<std::uint64_t>(&siginfo),
        &guest_result);
    const int restore_rc = running_on_target ? -4 : -3;
    __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                        "[EXECUTOR_MONO_ALERTABLE_SIGNAL] result=run thread=%s guest=%p "
                        "cur=%p curName=%s handler=%p flags=0x%x regsSource=%s regsRc=%d rc=%d "
                        "restoreRc=%d guestResult=0x%llx rip=0x%llx rsp=0x%llx interruptWait=%d",
                        PthreadName(target), guest_cell, curthread, PthreadName(curthread),
                        reinterpret_cast<void*>(handler), HandlerFlags[POSIX_SIGUSR1],
                        regs_source, regs_rc, rc, restore_rc,
                        static_cast<unsigned long long>(guest_result),
                        static_cast<unsigned long long>(ctx.uc_mcontext.mc_rip),
                        static_cast<unsigned long long>(ctx.uc_mcontext.mc_rsp),
                        explicit_mutex_safe_point ? 1 : 0);
    if (rc < 0) {
        MarkMonoSuspendSigusr1Pending(target);
        return MonoSignalSafePointResult::None;
    }
    ClearMonoSuspendAbbaSignalState(target);
#ifdef __ANDROID__
    t_defer_since = {};
#endif
    (void)explicit_mutex_safe_point;
    return MonoSignalSafePointResult::HandledContinue;
}

extern "C" void executor_live_mono_run_pending_signal_safe_point() {
    (void)RunPendingMonoSuspendSigusr1AtSafePoint(CurrentMutexThread(), nullptr);
}

static void RecordMutexWaitSnapshot(Pthread* waiter, PthreadMutex* waiting_on, void* guest_cell,
                                    Pthread* owner_snapshot) {
    if (waiter == nullptr || waiting_on == nullptr) {
        return;
    }
    std::scoped_lock lock(g_mutex_wait_snapshot_lock);
    const auto now = std::chrono::steady_clock::now();
    auto [it, inserted] = g_mutex_wait_snapshots.try_emplace(waiter);
    MutexWaitSnapshot& snapshot = it->second;
    if (inserted || snapshot.waiting_on != waiting_on || snapshot.guest_cell != guest_cell) {
        snapshot = {};
        snapshot.waiting_on = waiting_on;
        snapshot.guest_cell = guest_cell;
        snapshot.owner_snapshot = owner_snapshot;
        snapshot.wait_started = now;
        snapshot.next_report = now + std::chrono::seconds(2);
        snapshot.generation_at_start =
            waiting_on->m_lock_generation.load(std::memory_order_relaxed);
        return;
    }
    snapshot.owner_snapshot = owner_snapshot;
}

static void EraseMutexWaitSnapshot(Pthread* waiter, PthreadMutex* waiting_on) {
    if (waiter == nullptr) {
        return;
    }
    std::scoped_lock lock(g_mutex_wait_snapshot_lock);
    const auto it = g_mutex_wait_snapshots.find(waiter);
    if (it != g_mutex_wait_snapshots.end() &&
        (waiting_on == nullptr || it->second.waiting_on == waiting_on)) {
        g_mutex_wait_snapshots.erase(it);
    }
}

static void ClearMutexWaitSnapshot(Pthread* waiter, PthreadMutex* waiting_on) {
    EraseMutexWaitSnapshot(waiter, waiting_on);
    ClearMonoSuspendAbbaSignalState(waiter);
}

extern "C" void executor_live_dump_mutex_wait_ledger(const char* reason, int pulse) {
    if (!ExecutorStallMutexLedgerEnabled()) {
        return;
    }

    struct DueWait {
        Pthread* waiter;
        MutexWaitSnapshot snapshot;
    };
    std::vector<DueWait> due;
    const auto now = std::chrono::steady_clock::now();
    {
        std::scoped_lock lock(g_mutex_wait_snapshot_lock);
        due.reserve(g_mutex_wait_snapshots.size());
        for (auto& [waiter, snapshot] : g_mutex_wait_snapshots) {
            if (snapshot.waiting_on == nullptr || snapshot.wait_started.time_since_epoch().count() == 0 ||
                now < snapshot.next_report) {
                continue;
            }
            due.push_back({waiter, snapshot});
            const u32 next_shift = std::min<u32>(snapshot.report_count + 2, 10);
            snapshot.report_count++;
            snapshot.next_report =
                snapshot.wait_started + std::chrono::seconds(1ULL << next_shift);
        }
    }

    static std::atomic_int global_budget{256};
    auto* regs_by_pthread = ResolveGuestThreadRegsByPthread();
    auto* regs_by_tid = ResolveGuestThreadRegsByTid();
    for (const DueWait& item : due) {
        if (global_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
            return;
        }
        Pthread* waiter = item.waiter;
        PthreadMutex* native = item.snapshot.waiting_on;
        Pthread* owner = native != nullptr ? native->m_owner : nullptr;
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now - item.snapshot.wait_started)
                                    .count();
        const u64 generation =
            native != nullptr ? native->m_lock_generation.load(std::memory_order_relaxed) : 0;

        const auto capture_rip = [&](Pthread* thread, u64& rip, u64& rsp, long& host_tid) {
            rip = 0;
            rsp = 0;
            u64 rbp = 0;
            long registry_tid = 0;
            char host_name[32]{};
            int rc = -1;
            if (thread != nullptr && regs_by_pthread != nullptr) {
                rc = regs_by_pthread(thread->native_thr.GetHandle(), &rip, &rsp, &rbp,
                                     &registry_tid, host_name, sizeof(host_name));
            }
            host_tid = thread != nullptr ? static_cast<long>(thread->native_thr.GetTid()) : 0;
            if (rc != 0 && host_tid > 0 && regs_by_tid != nullptr) {
                rc = regs_by_tid(host_tid, &rip, &rsp, &rbp, &registry_tid, host_name,
                                 sizeof(host_name));
            }
            return rc;
        };

        u64 waiter_rip = 0;
        u64 waiter_rsp = 0;
        long waiter_host_tid = 0;
        const int waiter_regs_rc =
            capture_rip(waiter, waiter_rip, waiter_rsp, waiter_host_tid);
        u64 owner_rip = 0;
        u64 owner_rsp = 0;
        long owner_host_tid = 0;
        const int owner_regs_rc = capture_rip(owner, owner_rip, owner_rsp, owner_host_tid);
        const std::string waiter_symbol = SymbolizeOwnerPcForLog(waiter_rip);
        const std::string owner_symbol = SymbolizeOwnerPcForLog(owner_rip);

        const s32 waiter_tid = waiter != nullptr
                                   ? waiter->tid.load(std::memory_order_relaxed)
                                   : TidTerminated;
        const s32 owner_tid = owner != nullptr ? owner->tid.load(std::memory_order_relaxed)
                                               : TidTerminated;
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Mutex",
            "[EXECUTOR_MUTEX_WAIT_LEDGER] reason=%s pulse=%d report=%u waitMs=%lld "
            "guest=%p native=%p generationStart=%llu generationNow=%llu ownerChanged=%d "
            "waiter=%p waiterName=%s waiterTid=%d waiterHostTid=%ld waiterState=%s "
            "waiterTerminated=%d waiterCancelling=%d waiterForceExit=%d waiterRegsRc=%d "
            "waiterRip=0x%llx waiterRsp=0x%llx waiterSymbol=%s "
            "ownerStart=%p owner=%p ownerName=%s ownerTid=%d ownerHostTid=%ld ownerState=%s "
            "ownerTerminated=%d ownerCancelling=%d ownerForceExit=%d ownerRegsRc=%d "
            "ownerRip=0x%llx ownerRsp=0x%llx ownerSymbol=%s",
            reason ? reason : "?", pulse, item.snapshot.report_count + 1,
            static_cast<long long>(elapsed_ms), item.snapshot.guest_cell, native,
            static_cast<unsigned long long>(item.snapshot.generation_at_start),
            static_cast<unsigned long long>(generation),
            owner != item.snapshot.owner_snapshot ? 1 : 0, waiter, PthreadName(waiter), waiter_tid,
            waiter_host_tid, PthreadStateName(waiter), waiter_tid == TidTerminated ? 1 : 0,
            waiter != nullptr && waiter->cancelling ? 1 : 0,
            waiter != nullptr && waiter->force_exit ? 1 : 0, waiter_regs_rc,
            static_cast<unsigned long long>(waiter_rip),
            static_cast<unsigned long long>(waiter_rsp), waiter_symbol.c_str(),
            item.snapshot.owner_snapshot, owner, PthreadName(owner), owner_tid, owner_host_tid,
            PthreadStateName(owner), owner_tid == TidTerminated ? 1 : 0,
            owner != nullptr && owner->cancelling ? 1 : 0,
            owner != nullptr && owner->force_exit ? 1 : 0, owner_regs_rc,
            static_cast<unsigned long long>(owner_rip),
            static_cast<unsigned long long>(owner_rsp), owner_symbol.c_str());
    }
}

static bool IsMonoSuspendPeerCandidate(Pthread* thread, Pthread* cur, Pthread* primary) {
    return thread != nullptr && thread != cur && thread != primary &&
           thread->state == PthreadState::Running &&
           (ThreadNameContains(thread, "mono thread") || ThreadNameContains(thread, "Thread2"));
}

static Pthread* FindMonoSuspendPeerForRuntimeLock(Pthread* cur, Pthread* primary,
                                                  PthreadMutex* runtime_lock) {
    Pthread* fallback = nullptr;
    {
        std::scoped_lock lock(g_mutex_wait_snapshot_lock);
        for (const auto& [thread, wait] : g_mutex_wait_snapshots) {
            if (!IsMonoSuspendPeerCandidate(thread, cur, primary)) {
                continue;
            }
            if (wait.waiting_on == runtime_lock) {
                return thread;
            }
            if (fallback == nullptr) {
                fallback = thread;
            }
        }
    }
    if (fallback != nullptr) {
        return fallback;
    }

    auto* state = ThrState::Instance();
    if (state == nullptr) {
        return nullptr;
    }
    std::scoped_lock lock(state->thread_list_lock);
    for (Pthread* thread : state->threads) {
        if (IsMonoSuspendPeerCandidate(thread, cur, primary) &&
            ThreadNameContains(thread, "mono thread")) {
            return thread;
        }
        if (fallback == nullptr && IsMonoSuspendPeerCandidate(thread, cur, primary)) {
            fallback = thread;
        }
    }
    return fallback;
}

static Pthread* FindMonoSuspendPeerWaitingOnRuntimeLockOwnedBy(Pthread* cur, Pthread* primary,
                                                               MutexWaitSnapshot* out_wait) {
    if (cur == nullptr) {
        return nullptr;
    }

    std::scoped_lock lock(g_mutex_wait_snapshot_lock);
    for (const auto& [thread, wait] : g_mutex_wait_snapshots) {
        if (!IsMonoSuspendPeerCandidate(thread, cur, primary)) {
            continue;
        }
        if (wait.waiting_on == nullptr || wait.waiting_on->m_owner != cur ||
            !IsLiveMonoRuntimeMutexCell(wait.guest_cell)) {
            continue;
        }
        if (out_wait != nullptr) {
            *out_wait = wait;
        }
        return thread;
    }
    return nullptr;
}

static int QueueMonoSuspendSigusr1(Pthread* target) {
    if (target == nullptr) {
        return -4;
    }
    if (std::getenv("EXECUTOR_LIVE_MONO_SAFEPOINT_SIGNAL") != nullptr) {
        MarkMonoSuspendSigusr1Pending(target);
        static std::atomic_int safepoint_log_budget{128};
        if (safepoint_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Mutex",
                "[EXECUTOR_MONO_ABBA_PENDING_SIGNAL] target=%p targetName=%s "
                "targetNative=0x%zx queueRc=0 reason=pc_oracle_alertable_safe_point",
                target, PthreadName(target),
                static_cast<std::size_t>(NativePthreadHandle(target)));
        }
        return 0;
    }
    if (std::getenv("EXECUTOR_LIVE_MONO_SYNTHETIC_ACK") &&
        !(executor_lsx4_android_jit_active != nullptr &&
          executor_lsx4_android_jit_active()) &&
        (ThreadNameContains(target, "mono thread") || ThreadNameContains(target, "UnityPreload")) &&
        executor_live_mono_synthetic_suspend_ack) {
        const int synth_rc =
            executor_live_mono_synthetic_suspend_ack(target, "abba_group_hle_wait");
        static std::atomic_int synth_log_budget{64};
        if (synth_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                                "[EXECUTOR_MONO_ABBA_PENDING_SIGNAL] target=%p targetName=%s "
                                "queueRc=%d reason=pc_oracle_synthetic_hle_wait_ack",
                                target, PthreadName(target), synth_rc);
        }
        return synth_rc;
    }
    const auto pthr = NativePthreadHandle(target);
    if (std::getenv("EXECUTOR_LIVE_MONO_SIGNAL_DELIVER") != nullptr) {
        const bool jit_active =
            executor_lsx4_android_jit_active != nullptr &&
            executor_lsx4_android_jit_active();
        if (jit_active) {
            MarkMonoSuspendSigusr1Pending(target);
            static std::atomic_int jit_log_budget{128};
            if (jit_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Mutex",
                    "[EXECUTOR_MONO_ABBA_PENDING_SIGNAL] target=%p targetName=%s "
                    "targetNative=0x%zx queueRc=0 "
                    "reason=jit_guest_safe_point_signal_delivery",
                    target, PthreadName(target), static_cast<std::size_t>(pthr));
            }
            return 0;
        }
        const int native_sig = OrbisToNativeSignal(POSIX_SIGUSR1);
        const int native_rc = pthread_kill(pthr, native_sig);
        if (native_rc != 0) {
            MarkMonoSuspendSigusr1Pending(target);
        }
        static std::atomic_int native_log_budget{64};
        if (native_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                                "[EXECUTOR_MONO_ABBA_PENDING_SIGNAL] target=%p targetName=%s "
                                "targetNative=0x%zx nativeSig=%d queueRc=%d "
                                "reason=pc_oracle_native_sigaction_signal",
                                target, PthreadName(target), static_cast<std::size_t>(pthr),
                                native_sig, native_rc);
        }
        return native_rc;
    }
    int queue_rc = -3;
    if (executor_lsx4_android_queue_guest_signal_for_thread != nullptr) {
        queue_rc = executor_lsx4_android_queue_guest_signal_for_thread(
            target, static_cast<std::uintptr_t>(pthr), POSIX_SIGUSR1);
    } else if (executor_lsx4_android_queue_guest_signal != nullptr) {
        queue_rc = executor_lsx4_android_queue_guest_signal(static_cast<std::uintptr_t>(pthr),
                                                               POSIX_SIGUSR1);
    }
    if (queue_rc != 0) {
        MarkMonoSuspendSigusr1Pending(target);
    }
    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                            "[EXECUTOR_MONO_ABBA_PENDING_SIGNAL] target=%p targetName=%s "
                            "targetNative=0x%zx queueRc=%d reason=pc_oracle_target_thread_signal",
                            target, PthreadName(target),
                            static_cast<std::size_t>(pthr), queue_rc);
    }
    return queue_rc;
}

static bool LooksLikeOrbisCodePc(u64 pc) {
    return pc >= 0x800000000ULL && pc < 0x900000000ULL;
}

static bool LooksLikeOrbisStackRsp(u64 rsp) {
    return rsp >= 0x60000000ULL && rsp < 0x800000000ULL;
}

static bool CaptureMonoSuspendQueueContext(Pthread* target, u64* rip, u64* rsp) {
    if (target == nullptr || rip == nullptr || rsp == nullptr) {
        return false;
    }
    auto* get_regs = ResolveGuestThreadRegsByPthread();
    if (get_regs == nullptr) {
        return false;
    }
    u64 rbp = 0;
    long fex_tid = 0;
    char fex_host_name[32]{};
    const int rc = get_regs(NativePthreadHandle(target), rip, rsp, &rbp, &fex_tid, fex_host_name,
                            sizeof(fex_host_name));
    bool ok = rc == 0 && LooksLikeOrbisCodePc(*rip) && LooksLikeOrbisStackRsp(*rsp);
    const bool mono_runtime_wait_return_site =
        ThreadNameContains(target, "mono thread") && *rip == 0x8031b8819ULL;
    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                            "[EXECUTOR_MONO_ABBA_QUEUE_CONTEXT] target=%p targetName=%s "
                            "pthread=0x%zx rc=%d ok=%d fexTid=%ld fexHost=%s "
                            "rip=0x%llx rsp=0x%llx rbp=0x%llx hleWaitReturn=%d",
                            target, PthreadName(target),
                            static_cast<std::size_t>(NativePthreadHandle(target)), rc, ok ? 1 : 0,
                            fex_tid, fex_host_name[0] ? fex_host_name : "<none>",
                            static_cast<unsigned long long>(*rip),
                            static_cast<unsigned long long>(*rsp),
                            static_cast<unsigned long long>(rbp),
                            mono_runtime_wait_return_site ? 1 : 0);
    }
    return ok;
}

static int QueueMonoSuspendSigusr1WithContext(Pthread* target, u64 rip, u64 rsp) {
    if (target == nullptr) {
        return -4;
    }
    if (executor_lsx4_android_queue_guest_signal_for_thread_context == nullptr ||
        !LooksLikeOrbisCodePc(rip) || !LooksLikeOrbisStackRsp(rsp)) {
        return QueueMonoSuspendSigusr1(target);
    }
    const auto pthr = NativePthreadHandle(target);
    const int queue_rc = executor_lsx4_android_queue_guest_signal_for_thread_context(
        target, static_cast<std::uintptr_t>(pthr), POSIX_SIGUSR1, rip, rsp);
    if (queue_rc != 0) {
        MarkMonoSuspendSigusr1Pending(target);
    }
    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                            "[EXECUTOR_MONO_ABBA_PENDING_SIGNAL] target=%p targetName=%s "
                            "targetNative=0x%zx queueRc=%d rip=0x%llx rsp=0x%llx "
                            "reason=pc_oracle_target_thread_signal_context",
                            target, PthreadName(target), static_cast<std::size_t>(pthr),
                            queue_rc, static_cast<unsigned long long>(rip),
                            static_cast<unsigned long long>(rsp));
    }
    return queue_rc;
}

static int QueueMonoSuspendSigusr1AtMutexSafePoint(Pthread* target,
                                                   const char* reason) {
    if (target == nullptr) {
        return -4;
    }
    MarkMonoSuspendSigusr1Pending(target);
    static std::atomic_int log_budget{128};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Mutex",
            "[EXECUTOR_MONO_ABBA_PENDING_SIGNAL] target=%p targetName=%s "
            "targetNative=0x%zx queueRc=0 reason=%s",
            target, PthreadName(target),
            static_cast<std::size_t>(NativePthreadHandle(target)),
            reason ? reason : "pc_oracle_mutex_safe_point");
    }
    return 0;
}

static bool ShouldAvoidRepeatSignalToParkedMonoPeer(Pthread* target) {
    (void)target;
    return false;
}

static bool UnityPreloadReachedPcSuspendFrontier(Pthread* target, const char* caller,
                                                 const void* observed_mutex) {
    if (target == nullptr || !ThreadNameContains(target, "UnityPreload")) {
        return true;
    }
    constexpr std::uintptr_t UnityThreadWakeSemOffset = 0x44;
    constexpr std::uintptr_t UnityThreadControlMutexOffset = 0x78;
    constexpr std::uintptr_t EbootBase = 0x800000000ULL;
    constexpr std::uintptr_t EbootEnd = 0x810000000ULL;
    const auto arg = reinterpret_cast<std::uintptr_t>(target->arg);
    auto* wait_sem = reinterpret_cast<void*>(arg + UnityThreadWakeSemOffset);
    const bool sem_wait_frontier = arg != 0 && ExecutorLivePosixSemHasWaitEnter(wait_sem);
    const auto mutex_addr = reinterpret_cast<std::uintptr_t>(observed_mutex);
    const u64 q1 = GuestReadQword(arg + 0x08);
    const u64 q4 = GuestReadQword(arg + 0x20);
    const u64 q5 = GuestReadQword(arg + 0x28);
    const bool control_block_frontier =
        arg >= 0x200000000ULL && arg < 0x300000000ULL &&
        observed_mutex != nullptr && mutex_addr == arg + UnityThreadControlMutexOffset &&
        q1 == 4 && q4 == arg && q5 >= EbootBase && q5 < EbootEnd;
    bool reached = sem_wait_frontier;
    static const bool poll_frontier =
        ::access("/data/data/app.lsx4.android/files/lsx4-home/run-abba-poll-frontier",
                 F_OK) == 0 ||
        ::access("/data/user/0/app.lsx4.android/files/lsx4-home/run-abba-poll-frontier",
                 F_OK) == 0;
    if (poll_frontier && !reached) {
        reached = true;
        static std::atomic_int poll_log_budget{32};
        if (poll_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MONO_ABBA_POLL_FRONTIER] target=%p targetName=%s arg=%p "
                                "caller=%s reason=deliver_sig30_at_poll_loop_replicate_pc",
                                target, PthreadName(target), reinterpret_cast<void*>(arg),
                                caller ? caller : "<unknown>");
        }
    }
    if (!reached) {
        static std::atomic_int log_budget{64};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Mutex",
                "[EXECUTOR_MONO_ABBA_SIGNAL_SKIP] target=%p targetName=%s arg=%p "
                "waitSem=%p observedMutex=%p q1=0x%llx q4=0x%llx q5=0x%llx "
                "reason=unitypreload_before_pc_suspend_frontier caller=%s",
                target, PthreadName(target), reinterpret_cast<void*>(arg), wait_sem,
                observed_mutex, static_cast<unsigned long long>(q1),
                static_cast<unsigned long long>(q4), static_cast<unsigned long long>(q5),
                caller ? caller : "<unknown>");
        }
    } else if (control_block_frontier) {
        static std::atomic_int log_budget{16};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Mutex",
                "[EXECUTOR_MONO_ABBA_FRONTIER] target=%p targetName=%s arg=%p "
                "observedMutex=%p waitSem=%p semWait=%d q1=0x%llx q4=0x%llx q5=0x%llx "
                "reason=unitypreload_control_block_pc_frontier caller=%s",
                target, PthreadName(target), reinterpret_cast<void*>(arg), observed_mutex,
                wait_sem, sem_wait_frontier ? 1 : 0, static_cast<unsigned long long>(q1),
                static_cast<unsigned long long>(q4), static_cast<unsigned long long>(q5),
                caller ? caller : "<unknown>");
        }
    }
    return reached;
}

static bool HandleMonoSuspendTryLockAbbaCycle(void* mutex, PthreadMutex* native, s32 ret) {
    if (ret != POSIX_EBUSY || native == nullptr || native->m_owner == nullptr) {
        return false;
    }
    Pthread* cur = g_curthread != nullptr ? g_curthread : CurrentMutexThread();
    if (cur == nullptr || cur == native->m_owner) {
        return false;
    }

    if (!IsLiveDirectMemoryMutexCell(mutex) || !ThreadNameContains(cur, "Game:Main") ||
        !ThreadNameContains(native->m_owner, "UnityPreload")) {
        return false;
    }
    MutexWaitSnapshot owner_wait{};
    bool owner_wait_matches = false;
    {
        std::scoped_lock lock(g_mutex_wait_snapshot_lock);
        const auto it = g_mutex_wait_snapshots.find(native->m_owner);
        if (it != g_mutex_wait_snapshots.end()) {
            owner_wait = it->second;
            owner_wait_matches = owner_wait.waiting_on != nullptr &&
                                 owner_wait.waiting_on->m_owner == cur &&
                                 IsLiveMonoRuntimeMutexCell(owner_wait.guest_cell);
        }
    }

    Pthread* target = native->m_owner;
    if (!UnityPreloadReachedPcSuspendFrontier(target, "trylock_abba_cycle", mutex)) {
        return false;
    }
    Pthread* peer = nullptr;
    MutexWaitSnapshot peer_wait{};
    const char* pattern = "owner_wait_runtime_lock";
    if (owner_wait_matches) {
        peer = FindMonoSuspendPeerForRuntimeLock(cur, target, owner_wait.waiting_on);
        peer_wait = owner_wait;
    } else {
        peer = FindMonoSuspendPeerWaitingOnRuntimeLockOwnedBy(cur, target, &peer_wait);
        if (peer == nullptr) {
            peer = FindMonoSuspendPeerForRuntimeLock(cur, target, nullptr);
            pattern = "unitypreload_eventflag_wait";
        } else {
            pattern = "peer_wait_runtime_lock";
        }
    }

    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Mutex",
            "[EXECUTOR_MUTEX_ABBA_DETECTED] guest=%p native=%p "
            "cur=%p curName=%s owner=%p ownerName=%s pattern=%s ownerWaitGuest=%p "
            "ownerWaitNative=%p ownerWaitOwner=%p ownerWaitOwnerName=%s peer=%p "
            "peerName=%s peerWaitGuest=%p peerWaitNative=%p",
            mutex, native, cur, PthreadName(cur), native->m_owner, PthreadName(native->m_owner),
            pattern, owner_wait.guest_cell, owner_wait.waiting_on,
            owner_wait.waiting_on ? owner_wait.waiting_on->m_owner : nullptr,
            PthreadName(owner_wait.waiting_on ? owner_wait.waiting_on->m_owner : nullptr), peer,
            PthreadName(peer), peer_wait.guest_cell, peer_wait.waiting_on);
    }

    const bool target_only_signal = peer == nullptr;
    if (target_only_signal && !owner_wait_matches) {
        static std::atomic_int skip_log_budget{64};
        if (skip_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Mutex",
                "[EXECUTOR_MONO_ABBA_SIGNAL_TARGET_ONLY] guest=%p cur=%p curName=%s target=%p "
                "targetName=%s pattern=%s reason=no_mono_peer_available",
                mutex, cur, PthreadName(cur), target, PthreadName(target), pattern);
        }
        return false;
    }

    const bool target_waiting_for_resume = IsMonoSuspendResumeEventWaiter(target);
    if (target_waiting_for_resume && peer != nullptr &&
        !ShouldAvoidRepeatSignalToParkedMonoPeer(peer)) {
        bool should_signal_peer = false;
        {
            std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
            const auto [it, inserted] = g_mono_suspend_preload_peer_signaled.insert(target);
            (void)it;
            should_signal_peer = inserted;
        }
        if (should_signal_peer) {
            const int peer_queue_rc = QueueMonoSuspendSigusr1(peer);
            if (peer_queue_rc != 0) {
                MarkMonoSuspendSigusr1Pending(peer);
            }
            static std::atomic_int peer_log_budget{64};
            if (peer_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(
                    ANDROID_LOG_WARN, "LSX4Mutex",
                    "[EXECUTOR_MONO_ABBA_PEER_SIGNAL] guest=%p cur=%p curName=%s target=%p "
                    "targetName=%s peer=%p peerName=%s peerNative=0x%zx sig=%d queueRc=%d "
                    "reason=pc_oracle_target_wait_observed_hot_path",
                    mutex, cur, PthreadName(cur), target, PthreadName(target), peer,
                    PthreadName(peer), static_cast<std::size_t>(NativePthreadHandle(peer)),
                    POSIX_SIGUSR1, peer_queue_rc);
            }
            return true;
        }
    }

    if (!TryMarkMonoSuspendAbbaSignalActive(target)) {
        return true;
    }
    if (!TryMarkUnityPreloadControlSignalActive(mutex)) {
        return true;
    }
    const auto pthr = NativePthreadHandle(target);
    const auto peer_pthr = NativePthreadHandle(peer);
    const int queue_rc = QueueMonoSuspendSigusr1(target);
    const bool skip_peer_signal = ShouldAvoidRepeatSignalToParkedMonoPeer(peer);
    const int peer_queue_rc =
        peer != nullptr && !skip_peer_signal ? QueueMonoSuspendSigusr1(peer) : -4;
    const bool fallback_pending = queue_rc != 0 && queue_rc != -4;
    const bool peer_fallback_pending = peer != nullptr && peer_queue_rc != 0 && peer_queue_rc != -4;
    if (fallback_pending) {
        MarkMonoSuspendSigusr1Pending(target);
    }
    if (peer_fallback_pending) {
        MarkMonoSuspendSigusr1Pending(peer);
    }
    const int queued_count = (queue_rc == 0 ? 1 : 0) + (peer_queue_rc == 0 ? 1 : 0);
    const int required_waiters = queued_count;
    static std::atomic_int signal_log_budget{64};
    if (signal_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Mutex",
            "[EXECUTOR_MONO_ABBA_SIGNAL] guest=%p cur=%p curName=%s target=%p "
            "targetName=%s targetNative=0x%zx peer=%p peerName=%s peerNative=0x%zx "
            "waitGuest=%p pattern=%s sig=%d mode=%s queueRc=%d "
            "peerQueueRc=%d fallbackPending=%d peerFallbackPending=%d queuedCount=%d "
            "requiredWaiters=%d directAck=%d skipPeer=%d",
            mutex, cur, PthreadName(cur), target, PthreadName(target),
            static_cast<std::size_t>(pthr), peer, PthreadName(peer),
            static_cast<std::size_t>(peer_pthr), peer_wait.guest_cell, pattern, POSIX_SIGUSR1,
            "pc_oracle_unitypreload_then_peer",
            queue_rc, peer_queue_rc, fallback_pending ? 1 : 0, peer_fallback_pending ? 1 : 0,
            queued_count, required_waiters, 0, skip_peer_signal ? 1 : 0);
    }
    return true;
}

static bool SignalMonoSuspendForUnityPreloadBlockingLock(void* mutex, PthreadMutex* native) {
    if (!IsMonoAbbaPcOracleSignalContractEnabled()) {
        return false;
    }
    return HandleMonoSuspendTryLockAbbaCycle(mutex, native, POSIX_EBUSY);
}

static bool SignalMonoSuspendPeerForObservedUnityPreloadSuspend(void* mutex, PthreadMutex* native,
                                                               s32 ret) {
    if (ret != POSIX_EBUSY || native == nullptr || native->m_owner == nullptr ||
        !IsLiveDirectMemoryMutexCell(mutex)) {
        return false;
    }
    Pthread* cur = g_curthread != nullptr ? g_curthread : CurrentMutexThread();
    Pthread* target = native->m_owner;
    if (!ThreadNameContains(cur, "Game:Main") || !ThreadNameContains(target, "UnityPreload")) {
        return false;
    }

    Pthread* peer = FindMonoSuspendPeerForRuntimeLock(cur, target, nullptr);
    if (peer == nullptr) {
        static std::atomic_int no_peer_log_budget{32};
        if (no_peer_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Mutex",
                "[EXECUTOR_MONO_ABBA_PEER_SIGNAL] guest=%p cur=%p curName=%s target=%p "
                "targetName=%s peer=%p reason=no_peer_after_unitypreload_signal",
                mutex, cur, PthreadName(cur), target, PthreadName(target), peer);
        }
        return false;
    }

    if (!IsMonoSuspendResumeEventWaiter(target)) {
        static std::atomic_int wait_log_budget{64};
        if (wait_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Mutex",
                "[EXECUTOR_MONO_ABBA_PEER_SIGNAL] guest=%p cur=%p curName=%s target=%p "
                "targetName=%s peer=%p peerName=%s reason=wait_for_target_resume_event",
                mutex, cur, PthreadName(cur), target, PthreadName(target), peer,
                PthreadName(peer));
        }
        return false;
    }

    {
        std::scoped_lock lock(g_mono_suspend_abba_resume_lock);
        const auto [it, inserted] = g_mono_suspend_preload_peer_signaled.insert(target);
        (void)it;
        if (!inserted) {
            return false;
        }
    }

    const bool skip_peer_signal = ShouldAvoidRepeatSignalToParkedMonoPeer(peer);
    const int peer_queue_rc = skip_peer_signal ? -4 : QueueMonoSuspendSigusr1(peer);
    if (peer_queue_rc != 0 && peer_queue_rc != -4) {
        MarkMonoSuspendSigusr1Pending(peer);
    }
    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Mutex",
            "[EXECUTOR_MONO_ABBA_PEER_SIGNAL] guest=%p cur=%p curName=%s target=%p "
            "targetName=%s peer=%p peerName=%s peerNative=0x%zx sig=%d queueRc=%d "
            "reason=pc_oracle_unitypreload_then_mono_pair skipPeer=%d",
            mutex, cur, PthreadName(cur), target, PthreadName(target), peer, PthreadName(peer),
            static_cast<std::size_t>(NativePthreadHandle(peer)), POSIX_SIGUSR1, peer_queue_rc,
            skip_peer_signal ? 1 : 0);
    }
    return peer_queue_rc == 0;
}

static bool ShouldTraceUnityMutex() {
    static const bool enabled = std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr ||
                                std::getenv("EXECUTOR_TRACE_LIVE_SYNC") != nullptr;
    if (!enabled) {
        return false;
    }
    if (!g_curthread) {
        return false;
    }
    const auto& name = g_curthread->name;
    const bool is_fios_thread = name.find("SceFios") != std::string::npos;
    if (ExecutorLightOracleMode() && std::getenv("EXECUTOR_VERBOSE_LIVE_MUTEX") == nullptr &&
        !is_fios_thread) {
        return false;
    }
    if (std::getenv("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY") != nullptr) {
        return name.find("Game:Main") != std::string::npos ||
               name.find("UnityGfxDeviceWorker") != std::string::npos ||
               name.find("UnityWorker") != std::string::npos ||
               name.find("UnityPreload") != std::string::npos ||
               name.find("Submit Done Thread") != std::string::npos ||
               is_fios_thread;
    }
    return name.find("Game:Main") != std::string::npos ||
           name.find("UnityPreload") != std::string::npos ||
           name.find("mono") != std::string::npos ||
           name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("Submit Done Thread") != std::string::npos ||
           is_fios_thread;
}

static bool IsInvalidGuestMutexCell(PthreadMutexT* mutex) {
    const auto address = reinterpret_cast<std::uintptr_t>(mutex);
    return address < 0x10000 || (address & (alignof(PthreadMutexT) - 1)) != 0;
}

static s32 RejectInvalidGuestMutexCell(const char* op, PthreadMutexT* mutex) {
    if (!IsInvalidGuestMutexCell(mutex)) {
        return 0;
    }
    static std::atomic_int log_budget{256};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_MUTEX_BADPTR] op=%s thread=%s guest=%p ret=%d",
                            op ? op : "?", g_curthread ? g_curthread->name.c_str()
                                                       : "<no-gcurthread>",
                            mutex, POSIX_EINVAL);
    }
    return POSIX_EINVAL;
}

static void TraceUnityMutex(const char* op, PthreadMutexT* guest, PthreadMutex* native, s32 ret) {
    if (!ShouldTraceUnityMutex()) {
        return;
    }
    static std::atomic_int budget{2048};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_MUTEX] op=%s thread=%s guest=%p native=%p owner=%p "
                        "ownerName=%s ownerState=%s ownerTid=%ld ownerHostTid=%ld "
                        "cur=%p flags=0x%x type=%u count=%d ret=%d",
                        op, g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>", guest,
                        native, native ? native->m_owner : nullptr,
                        PthreadName(native ? native->m_owner : nullptr),
                        PthreadStateName(native ? native->m_owner : nullptr),
                        PthreadGuestTid(native ? native->m_owner : nullptr),
                        native ? native->m_owner_host_tid : 0, CurrentMutexThread(),
                        native ? static_cast<u32>(native->m_flags) : 0u,
                        native ? static_cast<u32>(native->Type()) : 0u,
                        native ? native->m_count : 0, ret);
}

static void TraceFiosDeferredWakeRelease(PthreadMutex* mutex, Pthread* wake_thread,
                                         bool deferred) {
#ifdef __ANDROID__
    if (!deferred) {
        return;
    }
    if (std::getenv("EXECUTOR_LIGHT_ORACLE") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_WIDE") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_SYNC") == nullptr) {
        return;
    }
    const bool cur_fios = g_curthread && g_curthread->name.find("SceFios") != std::string::npos;
    const bool wake_fios = wake_thread && wake_thread->name.find("SceFios") != std::string::npos;
    if (!cur_fios && !wake_fios) {
        return;
    }
    static std::atomic_int budget{256};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_FIOS_DEFER_WAKE_RELEASE] cur=%p curName=%s mutex=%p "
                        "wakeThread=%p wakeName=%s wakeWillSleep=%d wakeDefer=%d flags=0x%x",
                        g_curthread, g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                        mutex, wake_thread, PthreadName(wake_thread),
                        wake_thread ? wake_thread->will_sleep : 0,
                        wake_thread ? wake_thread->nwaiter_defer : 0,
                        mutex ? static_cast<u32>(mutex->m_flags) : 0u);
#endif
}

static void TraceMutexLockEnter(PthreadMutexT* guest, PthreadMutex* native) {
    if (!ShouldTraceUnityMutex()) {
        return;
    }
    const auto guest_addr = reinterpret_cast<std::uintptr_t>(guest);
    const bool direct_mutex = guest_addr >= 0x200000000ULL && guest_addr < 0x300000000ULL;
    const bool has_owner = native != nullptr && native->m_owner != nullptr;
    if (!direct_mutex && !has_owner) {
        return;
    }
    static std::atomic_int budget{512};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_MUTEX_LOCK_ENTER] thread=%s guest=%p native=%p "
                        "owner=%p ownerName=%s ownerState=%s ownerTid=%ld ownerHostTid=%ld "
                        "cur=%p flags=0x%x type=%u count=%d tid=%ld",
                        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>", guest,
                        native, native ? native->m_owner : nullptr,
                        PthreadName(native ? native->m_owner : nullptr),
                        PthreadStateName(native ? native->m_owner : nullptr),
                        PthreadGuestTid(native ? native->m_owner : nullptr),
                        native ? native->m_owner_host_tid : 0, CurrentMutexThread(),
                        native ? static_cast<u32>(native->m_flags) : 0u,
                        native ? static_cast<u32>(native->Type()) : 0u,
                        native ? native->m_count : 0, CurrentHostTid());
}

struct MutexLifecycleTrace {
    const char* op = nullptr;
    PthreadMutex* old_value = nullptr;
    PthreadMutex* new_value = nullptr;
    u64 guest_return = 0;
    u64 return_off = 0;
    long host_tid = 0;
};

static std::mutex g_mutex_lifecycle_trace_lock;
static std::unordered_map<PthreadMutexT*, MutexLifecycleTrace> g_mutex_lifecycle_trace;

static void RecordMutexLifecycle(const char* op, PthreadMutexT* guest, PthreadMutex* old_value,
                                 PthreadMutex* new_value) {
    if (guest == nullptr) {
        return;
    }
    u64 guest_return = 0;
    u64 return_off = 0;
    u64 hle_arg0 = 0;
    char hle_symbol[2]{};
    char hle_module[2]{};
    if (executor_live_get_current_hle_call_site) {
        executor_live_get_current_hle_call_site(&guest_return, &return_off, &hle_arg0, hle_symbol,
                                                sizeof(hle_symbol), hle_module,
                                                sizeof(hle_module));
    }
    std::lock_guard lock(g_mutex_lifecycle_trace_lock);
    if (g_mutex_lifecycle_trace.size() < 8192 || g_mutex_lifecycle_trace.contains(guest)) {
        g_mutex_lifecycle_trace[guest] = {
            .op = op,
            .old_value = old_value,
            .new_value = new_value,
            .guest_return = guest_return,
            .return_off = return_off,
            .host_tid = CurrentHostTid(),
        };
    }
}

static void TraceLastMutexLifecycle(PthreadMutexT* guest) {
    MutexLifecycleTrace trace{};
    {
        std::lock_guard lock(g_mutex_lifecycle_trace_lock);
        const auto it = g_mutex_lifecycle_trace.find(guest);
        if (it == g_mutex_lifecycle_trace.end()) {
            return;
        }
        trace = it->second;
    }
    __android_log_print(
        ANDROID_LOG_ERROR, "LSX4Native",
        "[EXECUTOR_MUTEX_LAST_LIFECYCLE] guest=%p op=%s old=%p new=%p tid=%ld "
        "hleRet=%p returnOff=0x%llx",
        guest, trace.op ? trace.op : "?", trace.old_value, trace.new_value, trace.host_tid,
        reinterpret_cast<void*>(trace.guest_return),
        static_cast<unsigned long long>(trace.return_off));
}

static void TraceUnexpectedMutexReturn(const char* op, PthreadMutexT* guest, PthreadMutex* native,
                                       s32 ret) {
    if (ret == 0) {
        return;
    }
    static const bool verbose_api_results =
        std::getenv("EXECUTOR_VERBOSE_MUTEX_API_RESULTS") != nullptr;
    const bool lifecycle_failure = ret == POSIX_EINVAL;
    if (!verbose_api_results && !lifecycle_failure) {
        return;
    }
    if (ret == POSIX_EBUSY && op != nullptr && std::strcmp(op, "trylock") == 0) {
        return;
    }
    static std::atomic_int verbose_budget{512};
    static std::atomic_int lifecycle_budget{64};
    auto& budget = lifecycle_failure ? lifecycle_budget : verbose_budget;
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    u64 guest_return = 0;
    u64 return_off = 0;
    u64 hle_arg0 = 0;
    char hle_symbol[96]{};
    char hle_module[64]{};
    const int hle_rc = executor_live_get_current_hle_call_site
                           ? executor_live_get_current_hle_call_site(
                                 &guest_return, &return_off, &hle_arg0, hle_symbol,
                                 sizeof(hle_symbol), hle_module, sizeof(hle_module))
                           : -2;
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_MUTEX_RET] op=%s thread=%s guest=%p native=%p ret=%d "
                        "owner=%p ownerName=%s ownerState=%s ownerTid=%ld ownerHostTid=%ld "
                        "cur=%p flags=0x%x type=%u count=%d tid=%ld hleRc=%d hleRet=%p "
                        "returnOff=0x%llx hleModule=%s hleSymbol=%s hleArg0=%p",
                        op ? op : "?", g_curthread ? g_curthread->name.c_str()
                                                    : "<no-gcurthread>",
                        guest, native, ret, native ? native->m_owner : nullptr,
                        PthreadName(native ? native->m_owner : nullptr),
                        PthreadStateName(native ? native->m_owner : nullptr),
                        PthreadGuestTid(native ? native->m_owner : nullptr),
                        native ? native->m_owner_host_tid : 0,
                        CurrentMutexThread(), native ? static_cast<u32>(native->m_flags) : 0u,
                        native ? static_cast<u32>(native->Type()) : 0u,
                        native ? native->m_count : 0, CurrentHostTid(), hle_rc,
                        reinterpret_cast<void*>(guest_return),
                        static_cast<unsigned long long>(return_off),
                        hle_module[0] ? hle_module : "<none>",
                        hle_symbol[0] ? hle_symbol : "<none>",
                        reinterpret_cast<void*>(hle_arg0));
    if (lifecycle_failure) {
        TraceLastMutexLifecycle(guest);
    }
}

static void TraceMutexInvalidCellState(const char* op, PthreadMutexT* guest,
                                       PthreadMutex* stored, const char* reason) {
    static std::atomic_int budget{64};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    u64 guest_return = 0;
    u64 return_off = 0;
    u64 hle_arg0 = 0;
    char hle_symbol[96]{};
    char hle_module[64]{};
    const int hle_rc = executor_live_get_current_hle_call_site
                           ? executor_live_get_current_hle_call_site(
                                 &guest_return, &return_off, &hle_arg0, hle_symbol,
                                 sizeof(hle_symbol), hle_module, sizeof(hle_module))
                           : -2;
    __android_log_print(
        ANDROID_LOG_ERROR, "LSX4Native",
        "[EXECUTOR_MUTEX_EINVAL] op=%s reason=%s thread=%s guest=%p stored=%p tid=%ld "
        "hleRc=%d hleRet=%p returnOff=0x%llx hleModule=%s hleSymbol=%s hleArg0=%p",
        op ? op : "?", reason ? reason : "?",
        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>", guest, stored,
        CurrentHostTid(), hle_rc, reinterpret_cast<void*>(guest_return),
        static_cast<unsigned long long>(return_off),
        hle_module[0] ? hle_module : "<none>",
        hle_symbol[0] ? hle_symbol : "<none>", reinterpret_cast<void*>(hle_arg0));
    TraceLastMutexLifecycle(guest);
}

static u64 EbootOffsetForLog(u64 pc) {
    return pc >= 0x800000000ULL ? pc - 0x800000000ULL : 0;
}

static void TraceLiveUnityPreloadActiveWorkItem(int count, u64 owner_arg, u64 mutex_guest,
                                                bool include_bytes) {
    if (owner_arg == 0) {
        return;
    }

    const u64 queue_begin = GuestReadQword(owner_arg + 0x90);
    const u64 queue_end = GuestReadQword(owner_arg + 0x98);
    const u64 queue_cap = GuestReadQword(owner_arg + 0xa0);
    const u64 complete_begin = GuestReadQword(owner_arg + 0xb0);
    const u64 complete_end = GuestReadQword(owner_arg + 0xb8);
    const u64 complete_cap = GuestReadQword(owner_arg + 0xc0);
    const u64 active = GuestReadQword(owner_arg + 0xc8);
    const u64 ready = GuestReadQword(owner_arg + 0xd0);
    const u64 unknown_d8 = GuestReadQword(owner_arg + 0xd8);
    const u64 queue_count =
        queue_end >= queue_begin ? (queue_end - queue_begin) / sizeof(u64) : 0;
    const u64 complete_count =
        complete_end >= complete_begin ? (complete_end - complete_begin) / sizeof(u64) : 0;

    u64 vtable = 0;
    u64 slot20 = 0;
    u64 slot30 = 0;
    u64 slot40 = 0;
    u64 slot48 = 0;
    u64 slot50 = 0;
    u64 slot58 = 0;
    u64 item_q08 = 0;
    u64 item_q40 = 0;
    u64 item_q50 = 0;
    u64 item_q58 = 0;
    u64 item_q80 = 0;
    u64 item_qa8 = 0;
    u64 item_qb8 = 0;
    u64 item_qc0 = 0;
    if (active != 0) {
        vtable = GuestReadQword(active + 0x00);
        item_q08 = GuestReadQword(active + 0x08);
        item_q40 = GuestReadQword(active + 0x40);
        item_q50 = GuestReadQword(active + 0x50);
        item_q58 = GuestReadQword(active + 0x58);
        item_q80 = GuestReadQword(active + 0x80);
        item_qa8 = GuestReadQword(active + 0xa8);
        item_qb8 = GuestReadQword(active + 0xb8);
        item_qc0 = GuestReadQword(active + 0xc0);
        if (vtable != 0) {
            slot20 = GuestReadQword(vtable + 0x20);
            slot30 = GuestReadQword(vtable + 0x30);
            slot40 = GuestReadQword(vtable + 0x40);
            slot48 = GuestReadQword(vtable + 0x48);
            slot50 = GuestReadQword(vtable + 0x50);
            slot58 = GuestReadQword(vtable + 0x58);
        }
    }

    __android_log_print(
        ANDROID_LOG_WARN, "LSX4Native",
        "[EXECUTOR_LIVE_UNITY_ACTIVE_WORK_ITEM] count=%d ownerArg=0x%llx mutex=0x%llx "
        "mutexDelta=0x%llx queueBegin=0x%llx queueEnd=0x%llx queueCap=0x%llx queueCount=%llu "
        "completeBegin=0x%llx completeEnd=0x%llx completeCap=0x%llx completeCount=%llu "
        "active=0x%llx ready=0x%llx qD8=0x%llx vtable=0x%llx "
        "slot20=0x%llx off20=0x%llx slot30=0x%llx off30=0x%llx "
        "slot40=0x%llx off40=0x%llx slot48=0x%llx off48=0x%llx "
        "slot50=0x%llx off50=0x%llx slot58=0x%llx off58=0x%llx "
        "itemQ08=0x%llx itemQ40=0x%llx itemQ50=0x%llx itemQ58=0x%llx "
        "itemQ80=0x%llx itemQA8=0x%llx itemQB8=0x%llx itemQC0=0x%llx",
        count, static_cast<unsigned long long>(owner_arg),
        static_cast<unsigned long long>(mutex_guest),
        static_cast<unsigned long long>(mutex_guest >= owner_arg ? mutex_guest - owner_arg : 0),
        static_cast<unsigned long long>(queue_begin),
        static_cast<unsigned long long>(queue_end),
        static_cast<unsigned long long>(queue_cap),
        static_cast<unsigned long long>(queue_count),
        static_cast<unsigned long long>(complete_begin),
        static_cast<unsigned long long>(complete_end),
        static_cast<unsigned long long>(complete_cap),
        static_cast<unsigned long long>(complete_count),
        static_cast<unsigned long long>(active),
        static_cast<unsigned long long>(ready),
        static_cast<unsigned long long>(unknown_d8),
        static_cast<unsigned long long>(vtable),
        static_cast<unsigned long long>(slot20),
        static_cast<unsigned long long>(EbootOffsetForLog(slot20)),
        static_cast<unsigned long long>(slot30),
        static_cast<unsigned long long>(EbootOffsetForLog(slot30)),
        static_cast<unsigned long long>(slot40),
        static_cast<unsigned long long>(EbootOffsetForLog(slot40)),
        static_cast<unsigned long long>(slot48),
        static_cast<unsigned long long>(EbootOffsetForLog(slot48)),
        static_cast<unsigned long long>(slot50),
        static_cast<unsigned long long>(EbootOffsetForLog(slot50)),
        static_cast<unsigned long long>(slot58),
        static_cast<unsigned long long>(EbootOffsetForLog(slot58)),
        static_cast<unsigned long long>(item_q08),
        static_cast<unsigned long long>(item_q40),
        static_cast<unsigned long long>(item_q50),
        static_cast<unsigned long long>(item_q58),
        static_cast<unsigned long long>(item_q80),
        static_cast<unsigned long long>(item_qa8),
        static_cast<unsigned long long>(item_qb8),
        static_cast<unsigned long long>(item_qc0));

    if (include_bytes && active != 0) {
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_LIVE_UNITY_ACTIVE_WORK_ITEM_BYTES] count=%d active=0x%llx bytes=%s",
            count, static_cast<unsigned long long>(active), GuestBytesHex(active, 0x180).c_str());
        if (vtable != 0) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Native",
                "[EXECUTOR_LIVE_UNITY_ACTIVE_WORK_ITEM_VTABLE] count=%d vtable=0x%llx bytes=%s",
                count, static_cast<unsigned long long>(vtable),
                GuestBytesHex(vtable, 0x80).c_str());
        }
    }
}

static int ScanGuestStackReturns(std::uint64_t rsp, std::uint64_t* out, int max_out, int max_scan) {
    if (rsp == 0 || out == nullptr || max_out <= 0) {
        return 0;
    }
    if (max_scan <= 0 || max_scan > 512) {
        max_scan = 256;
    }
    const long ps_long = ::sysconf(_SC_PAGESIZE);
    const std::uint64_t ps = ps_long > 0 ? static_cast<std::uint64_t>(ps_long) : 4096ull;
    int found = 0;
    std::uint64_t prev = 0;
    unsigned char vec = 0;
    std::uint64_t checked_page = ~0ull;
    for (int i = 0; i < max_scan && found < max_out; ++i) {
        const std::uint64_t addr = rsp + static_cast<std::uint64_t>(i) * 8;
        const std::uint64_t page = addr & ~(ps - 1ull);
        if (page != checked_page) {
            if (::mincore(reinterpret_cast<void*>(page), ps, &vec) != 0) {
                break;
            }
            checked_page = page;
        }
        std::uint64_t v = 0;
        std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
        if (v >= 0x800000000ull && v < 0x804000000ull && v != prev) {
            out[found++] = v;
            prev = v;
        }
    }
    return found;
}

extern "C" void executor_live_dump_unity_thread_rips(const char* reason, int pulse) {
    auto* regs_fn = ResolveGuestThreadRegsByPthread();
    auto* regs_by_tid_fn = ResolveGuestThreadRegsByTid();
    if (regs_fn == nullptr && regs_by_tid_fn == nullptr) {
        return;
    }
    Pthread* targets[24];
    int target_count = 0;
    bool thread_list_locked = false;
    if (auto* state = ThrState::Instance()) {
        std::unique_lock lock(state->thread_list_lock, std::try_to_lock);
        thread_list_locked = lock.owns_lock();
        if (thread_list_locked) {
            for (Pthread* thread : state->threads) {
                if (thread == nullptr || target_count >= 24) {
                    continue;
                }
                if (ThreadNameContains(thread, "Unity") ||
                    ThreadNameContains(thread, "Game:Main") ||
                    ThreadNameContains(thread, "mono") || ThreadNameContains(thread, "Mono") ||
                    ThreadNameContains(thread, "Loading.") ||
                    ThreadNameContains(thread, "Background Job")) {
                    targets[target_count++] = thread;
                }
            }
        }
    }
    if (pulse == 12) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_THREAD_RIP_TARGETS] reason=%s pulse=%d count=%d "
                            "threadListLocked=%d",
                            reason ? reason : "?", pulse, target_count,
                            thread_list_locked ? 1 : 0);
    }
    for (int i = 0; i < target_count; ++i) {
        Pthread* thread = targets[i];
        u64 rip = 0;
        u64 rsp = 0;
        u64 rbp = 0;
        long fex_tid = 0;
        char fex_host[32]{};
        int rc = -1;
        if (regs_fn != nullptr) {
            rc = regs_fn(thread->native_thr.GetHandle(), &rip, &rsp, &rbp, &fex_tid,
                         fex_host, sizeof(fex_host));
        }
        if (rc != 0 && regs_by_tid_fn != nullptr) {
            const pid_t host_tid = pthread_gettid_np(thread->native_thr.GetHandle());
            if (host_tid > 0) {
                rc = regs_by_tid_fn(host_tid, &rip, &rsp, &rbp, &fex_tid, fex_host,
                                    sizeof(fex_host));
            }
        }
        const std::string symbol = SymbolizeOwnerPcForLog(rip);
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_LIVE_THREAD_RIP_PULSE] reason=%s pulse=%d idx=%d rc=%d name=%s tid=%ld "
            "wchan=%p state=%s rip=0x%llx ebootOff=0x%llx rsp=0x%llx %s",
            reason ? reason : "?", pulse, i, rc, PthreadName(thread), PthreadGuestTid(thread),
            thread->wchan, PthreadStateName(thread), static_cast<unsigned long long>(rip),
            static_cast<unsigned long long>(rip >= 0x800000000ULL ? rip - 0x800000000ULL : 0),
            static_cast<unsigned long long>(rsp), symbol.c_str());
        if (rc == 0 && rsp != 0) {
            std::uint64_t frames[16]{};
            const int nf = ScanGuestStackReturns(rsp, frames, 16, 256);
            char chain[512];
            int off = 0;
            for (int f = 0; f < nf && off < static_cast<int>(sizeof(chain)) - 24; ++f) {
                const std::uint64_t fa = frames[f];
                const char* mod = fa >= 0x803190000ULL ? "mono+" : "eboot+";
                const std::uint64_t moff =
                    fa >= 0x803190000ULL ? fa - 0x803190000ULL : fa - 0x800000000ULL;
                off += std::snprintf(chain + off, sizeof(chain) - off, "%s0x%llx ", mod,
                                     static_cast<unsigned long long>(moff));
            }
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_LIVE_THREAD_STACK] pulse=%d idx=%d name=%s frames=%d %s",
                                pulse, i, PthreadName(thread), nf, chain);
        }
    }
    if (pulse == 12 && regs_by_tid_fn != nullptr) {
        if (DIR* task_dir = ::opendir("/proc/self/task")) {
            while (dirent* entry = ::readdir(task_dir)) {
                char* end = nullptr;
                const long host_tid = std::strtol(entry->d_name, &end, 10);
                if (host_tid <= 0 || end == entry->d_name || *end != '\0') {
                    continue;
                }
                char path[96]{};
                std::snprintf(path, sizeof(path), "/proc/self/task/%ld/comm", host_tid);
                FILE* comm_file = std::fopen(path, "r");
                if (comm_file == nullptr) {
                    continue;
                }
                char name[64]{};
                const bool read_name = std::fgets(name, sizeof(name), comm_file) != nullptr;
                std::fclose(comm_file);
                if (!read_name) {
                    continue;
                }
                name[std::strcspn(name, "\r\n")] = '\0';
                const std::string_view thread_name{name};
                if (thread_name.find("Unity") == std::string_view::npos &&
                    thread_name.find("Game:Main") == std::string_view::npos &&
                    thread_name.find("mono") == std::string_view::npos &&
                    thread_name.find("Mono") == std::string_view::npos) {
                    continue;
                }
                u64 rip = 0;
                u64 rsp = 0;
                u64 rbp = 0;
                long fex_tid = 0;
                char fex_host[32]{};
                const int rc = regs_by_tid_fn(host_tid, &rip, &rsp, &rbp, &fex_tid, fex_host,
                                              sizeof(fex_host));
                const std::string symbol = SymbolizeOwnerPcForLog(rip);
                __android_log_print(
                    ANDROID_LOG_WARN, "LSX4Native",
                    "[EXECUTOR_LIVE_OS_THREAD_RIP] hostTid=%ld name=%s rc=%d fexTid=%ld "
                    "rip=0x%llx rsp=0x%llx rbp=0x%llx %s",
                    host_tid, name, rc, fex_tid, static_cast<unsigned long long>(rip),
                    static_cast<unsigned long long>(rsp), static_cast<unsigned long long>(rbp),
                    symbol.c_str());
            }
            ::closedir(task_dir);
        }
    }
}

static void TraceLiveStuckMutexOwner(const char* op, PthreadMutexT* guest, PthreadMutex* native,
                                     s32 ret) {
    if (ret != POSIX_EBUSY || native == nullptr || native->m_owner == nullptr ||
        g_curthread == nullptr || g_curthread->name != "Game:Main") {
        return;
    }
    static const bool enabled = std::getenv("EXECUTOR_TRACE_LIVE_MUTEX_OWNER") != nullptr ||
                                std::getenv("EXECUTOR_TRACE_LIVE_MUTEX_OWNER_DUMP") != nullptr ||
                                std::getenv("EXECUTOR_TRACE_LIVE_EMU_DUMP") != nullptr;
    if (!enabled) {
        return;
    }
    const auto guest_addr = reinterpret_cast<std::uintptr_t>(guest);
    if (guest_addr < 0x200000000ULL || guest_addr >= 0x300000000ULL) {
        return;
    }
    const auto& owner_name = native->m_owner->name;
    if (owner_name.find("Unity") == std::string::npos) {
        return;
    }
    const bool force_unitypreload_owner_dump =
        owner_name == "UnityPreload" &&
        std::getenv("EXECUTOR_TRACE_LIVE_MUTEX_OWNER_DUMP") != nullptr;
    const bool dump_owner =
        std::getenv("EXECUTOR_TRACE_LIVE_MUTEX_OWNER_DUMP") != nullptr ||
        std::getenv("EXECUTOR_TRACE_LIVE_EMU_DUMP") != nullptr;

    static std::atomic<std::uintptr_t> last_guest{0};
    static std::atomic<int> seen{0};
    static std::atomic<int> dump_budget{4};
    static std::atomic<long long> last_timed_log_ms{0};
    const auto previous = last_guest.exchange(guest_addr, std::memory_order_relaxed);
    int count = 1;
    if (previous == guest_addr) {
        count = seen.fetch_add(1, std::memory_order_relaxed) + 1;
    } else {
        seen.store(1, std::memory_order_relaxed);
        last_timed_log_ms.store(0, std::memory_order_relaxed);
    }
    const bool milestone = count == 1 || count == 16 || count == 128 || count == 1024;
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    if (!milestone) {
        const auto previous_ms = last_timed_log_ms.load(std::memory_order_relaxed);
        if (previous_ms != 0 && now_ms - previous_ms < 5000) {
            return;
        }
        last_timed_log_ms.store(now_ms, std::memory_order_relaxed);
    } else {
        last_timed_log_ms.store(now_ms, std::memory_order_relaxed);
    }

    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MUTEX_OWNER_STUCK] op=%s count=%d guest=%p native=%p "
                        "ret=%d owner=%p ownerName=%s ownerState=%s ownerTid=%ld "
                        "ownerHostTid=%ld cur=%p curName=%s",
                        op ? op : "?", count, guest, native, ret, native->m_owner,
                        PthreadName(native->m_owner), PthreadStateName(native->m_owner),
                        PthreadGuestTid(native->m_owner), native->m_owner_host_tid,
                        CurrentMutexThread(),
                        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>");
    __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                        "[EXECUTOR_MONO_ABBA_PREQUEUE] op=%s ownerName=%s isTry=%d isPreload=%d "
                        "guest=%p native=%p",
                        op ? op : "?", owner_name.c_str(),
                        std::strcmp(op ? op : "", "trylock") == 0 ? 1 : 0,
                        owner_name == "UnityPreload" ? 1 : 0, guest, native);
    if (owner_name == "UnityPreload") {
        TraceLiveUnityPreloadActiveWorkItem(
            count, reinterpret_cast<u64>(native->m_owner->arg), static_cast<u64>(guest_addr),
            false);
    }
    if (auto* worker_regs_fn = ResolveGuestThreadRegsByPthread()) {
        Pthread* unity_workers[16];
        int worker_count = 0;
        if (auto* state = ThrState::Instance()) {
            std::scoped_lock worker_lock(state->thread_list_lock);
            for (Pthread* worker : state->threads) {
                if (worker != nullptr && ThreadNameContains(worker, "UnityWorker") &&
                    worker_count < 16) {
                    unity_workers[worker_count++] = worker;
                }
            }
        }
        for (int wi = 0; wi < worker_count; ++wi) {
            Pthread* worker = unity_workers[wi];
            u64 w_rip = 0;
            u64 w_rsp = 0;
            u64 w_rbp = 0;
            long w_fex_tid = 0;
            char w_fex_host[32]{};
            const auto w_pthread = worker->native_thr.GetHandle();
            const int w_rc = worker_regs_fn(w_pthread, &w_rip, &w_rsp, &w_rbp, &w_fex_tid,
                                            w_fex_host, sizeof(w_fex_host));
            const std::string w_symbol = SymbolizeOwnerPcForLog(w_rip);
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Native",
                "[EXECUTOR_LIVE_UNITY_WORKER_PC] count=%d idx=%d rc=%d name=%s tid=%ld wchan=%p "
                "state=%s fexTid=%ld fexHost=%s rip=0x%llx ebootOff=0x%llx rsp=0x%llx rbp=0x%llx %s",
                count, wi, w_rc, PthreadName(worker), PthreadGuestTid(worker), worker->wchan,
                PthreadStateName(worker), w_fex_tid, w_fex_host[0] ? w_fex_host : "<none>",
                static_cast<unsigned long long>(w_rip),
                static_cast<unsigned long long>(w_rip >= 0x800000000ULL ? w_rip - 0x800000000ULL
                                                                        : 0),
                static_cast<unsigned long long>(w_rsp), static_cast<unsigned long long>(w_rbp),
                w_symbol.c_str());
        }
    }
    if (auto* get_guest_thread_regs = ResolveGuestThreadRegsByPthread()) {
        u64 owner_rip = 0;
        u64 owner_rsp = 0;
        u64 owner_rbp = 0;
        long fex_tid = 0;
        char fex_host_name[32]{};
        const auto owner_pthread = native->m_owner->native_thr.GetHandle();
        const int pc_rc = get_guest_thread_regs(owner_pthread, &owner_rip, &owner_rsp, &owner_rbp,
                                                &fex_tid, fex_host_name, sizeof(fex_host_name));
        const std::string owner_symbol = SymbolizeOwnerPcForLog(owner_rip);
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_LIVE_MUTEX_OWNER_PC] count=%d rc=%d ownerName=%s pthread=0x%zx "
            "fexTid=%ld fexHost=%s rip=0x%llx rsp=0x%llx rbp=0x%llx ebootOff=0x%llx %s",
            count, pc_rc, PthreadName(native->m_owner),
            static_cast<std::size_t>(owner_pthread), fex_tid,
            fex_host_name[0] ? fex_host_name : "<none>",
            static_cast<unsigned long long>(owner_rip),
            static_cast<unsigned long long>(owner_rsp),
            static_cast<unsigned long long>(owner_rbp),
            static_cast<unsigned long long>(owner_rip >= 0x800000000ULL
                                                ? owner_rip - 0x800000000ULL
                                                : 0),
            owner_symbol.c_str());
        constexpr u64 EbootBase = 0x800000000ULL;
        constexpr u64 CreateMonoScriptCacheStart = EbootBase + 0x0019aa40ULL;
        constexpr u64 CreateMonoScriptCacheEnd = CreateMonoScriptCacheStart + 0x1e24ULL;
        if (owner_name == "UnityPreload" && owner_rip >= CreateMonoScriptCacheStart &&
            owner_rip < CreateMonoScriptCacheEnd) {
            static std::atomic_int monoscript_budget{24};
            if (monoscript_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                if (auto* get_full_regs = ResolveGuestThreadFullRegsByPthread()) {
                    ExecutorFexGuestRegsMirror regs{};
                    long full_tid = 0;
                    char full_host_name[32]{};
                    const int full_rc =
                        get_full_regs(owner_pthread, &regs, sizeof(regs), &full_tid,
                                      full_host_name, sizeof(full_host_name));
                    const u64 mono_class = regs.regs[7];
                    __android_log_print(
                        ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MONOSCRIPT_CACHE_LIGHT] count=%d rc=%d tid=%ld "
                        "host=%s rip=0x%llx off=0x%llx class=0x%llx threaded=%llu "
                        "obj=0x%llx rax=0x%llx rbx=0x%llx rcx=0x%llx rdx=0x%llx "
                        "rsi=0x%llx r8=0x%llx r9=0x%llx r12=0x%llx r13=0x%llx "
                        "r14=0x%llx r15=0x%llx classQ0=0x%llx classQ1=0x%llx "
                        "classQ2=0x%llx classQ3=0x%llx",
                        count, full_rc, full_tid, full_host_name[0] ? full_host_name : "<none>",
                        static_cast<unsigned long long>(regs.rip),
                        static_cast<unsigned long long>(regs.rip - EbootBase),
                        static_cast<unsigned long long>(mono_class),
                        static_cast<unsigned long long>(regs.regs[6]),
                        static_cast<unsigned long long>(regs.regs[2]),
                        static_cast<unsigned long long>(regs.regs[0]),
                        static_cast<unsigned long long>(regs.regs[3]),
                        static_cast<unsigned long long>(regs.regs[1]),
                        static_cast<unsigned long long>(regs.regs[2]),
                        static_cast<unsigned long long>(regs.regs[6]),
                        static_cast<unsigned long long>(regs.regs[8]),
                        static_cast<unsigned long long>(regs.regs[9]),
                        static_cast<unsigned long long>(regs.regs[12]),
                        static_cast<unsigned long long>(regs.regs[13]),
                        static_cast<unsigned long long>(regs.regs[14]),
                        static_cast<unsigned long long>(regs.regs[15]),
                        static_cast<unsigned long long>(GuestReadQword(mono_class + 0x00)),
                        static_cast<unsigned long long>(GuestReadQword(mono_class + 0x08)),
                        static_cast<unsigned long long>(GuestReadQword(mono_class + 0x10)),
                        static_cast<unsigned long long>(GuestReadQword(mono_class + 0x18)));
                    const u64 old_cache_probe = GuestReadQword(EbootBase + 0x28b3fe0ULL);
                    const u64 stack_guard_slot = GuestReadQword(EbootBase + 0x2823f88ULL);
                    const u64 stack_guard = GuestReadQword(stack_guard_slot);
                    const u64 mono_lock_a = GuestReadQword(EbootBase + 0x2823f90ULL);
                    const u64 mono_lock_b = GuestReadQword(EbootBase + 0x2823f98ULL);
                    const u64 cache_vector = GuestReadQword(EbootBase + 0x28b96a8ULL);
                    const u64 cache_vector_q08 = GuestReadQword(EbootBase + 0x28b96b0ULL);
                    const u64 cache_vector_q10 = GuestReadQword(EbootBase + 0x28b96b8ULL);
                    const u64 cache_scan_key = GuestReadQword(EbootBase + 0x28b9699ULL);
                    const u64 cache_scan_next = GuestReadQword(EbootBase + 0x28b96a1ULL);
                    const u64 cache_lock = GuestReadQword(EbootBase + 0x28ab908ULL);
                    const u64 owner_arg = reinterpret_cast<u64>(native->m_owner->arg);
                    const u64 active_item = GuestReadQword(owner_arg + 0xc8);
                    const u64 active_item_q40 = GuestReadQword(active_item + 0x40);
                    const u64 active_item_q50 = GuestReadQword(active_item + 0x50);
                    const u64 active_item_q58 = GuestReadQword(active_item + 0x58);
                    const u64 active_item_q80 = GuestReadQword(active_item + 0x80);
                    const u64 active_item_qa8 = GuestReadQword(active_item + 0xa8);
                    const u64 active_item_qb8 = GuestReadQword(active_item + 0xb8);
                    __android_log_print(
                        ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_MONOSCRIPT_GLOBALS_LIGHT] count=%d oldCacheProbe=0x%llx "
                        "stackGuardSlot=0x%llx stackGuard=0x%llx monoLockA=0x%llx "
                        "monoLockB=0x%llx cacheVector=0x%llx cacheVectorQ08=0x%llx "
                        "cacheVectorQ10=0x%llx cacheScanKey=0x%llx cacheScanNext=0x%llx "
                        "cacheLock=0x%llx itemQ40=0x%llx itemQ50=0x%llx itemQ58=0x%llx "
                        "itemQ80=0x%llx itemQA8=0x%llx itemQB8=0x%llx",
                        count, static_cast<unsigned long long>(old_cache_probe),
                        static_cast<unsigned long long>(stack_guard_slot),
                        static_cast<unsigned long long>(stack_guard),
                        static_cast<unsigned long long>(mono_lock_a),
                        static_cast<unsigned long long>(mono_lock_b),
                        static_cast<unsigned long long>(cache_vector),
                        static_cast<unsigned long long>(cache_vector_q08),
                        static_cast<unsigned long long>(cache_vector_q10),
                        static_cast<unsigned long long>(cache_scan_key),
                        static_cast<unsigned long long>(cache_scan_next),
                        static_cast<unsigned long long>(cache_lock),
                        static_cast<unsigned long long>(active_item_q40),
                        static_cast<unsigned long long>(active_item_q50),
                        static_cast<unsigned long long>(active_item_q58),
                        static_cast<unsigned long long>(active_item_q80),
                        static_cast<unsigned long long>(active_item_qa8),
                        static_cast<unsigned long long>(active_item_qb8));
                }
            }
        }
    }
    if (!dump_owner) {
        return;
    }
    if (auto* get_guest_thread_regs = ResolveGuestThreadRegsByPthread()) {
        u64 owner_rip = 0;
        u64 owner_rsp = 0;
        u64 owner_rbp = 0;
        long fex_tid = 0;
        char fex_host_name[32]{};
        const auto owner_pthread = native->m_owner->native_thr.GetHandle();
        const int pc_rc = get_guest_thread_regs(owner_pthread, &owner_rip, &owner_rsp, &owner_rbp,
                                                &fex_tid, fex_host_name, sizeof(fex_host_name));
        const std::string owner_symbol = SymbolizeOwnerPcForLog(owner_rip);
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_LIVE_MUTEX_OWNER_PC] count=%d rc=%d ownerName=%s pthread=0x%zx "
            "fexTid=%ld fexHost=%s rip=0x%llx rsp=0x%llx rbp=0x%llx ebootOff=0x%llx %s",
            count, pc_rc, PthreadName(native->m_owner),
            static_cast<std::size_t>(owner_pthread), fex_tid,
            fex_host_name[0] ? fex_host_name : "<none>",
            static_cast<unsigned long long>(owner_rip),
            static_cast<unsigned long long>(owner_rsp),
            static_cast<unsigned long long>(owner_rbp),
            static_cast<unsigned long long>(owner_rip >= 0x800000000ULL
                                                ? owner_rip - 0x800000000ULL
                                                : 0),
            owner_symbol.c_str());
        constexpr u64 EbootBase = 0x800000000ULL;
        constexpr u64 CreateMonoScriptCacheStart = EbootBase + 0x0019aa40ULL;
        constexpr u64 CreateMonoScriptCacheEnd = CreateMonoScriptCacheStart + 0x1e24ULL;
        if (owner_rip >= CreateMonoScriptCacheStart && owner_rip < CreateMonoScriptCacheEnd) {
            if (auto* get_full_regs = ResolveGuestThreadFullRegsByPthread()) {
                ExecutorFexGuestRegsMirror regs{};
                long full_tid = 0;
                char full_host_name[32]{};
                const int full_rc = get_full_regs(owner_pthread, &regs, sizeof(regs), &full_tid,
                                                  full_host_name, sizeof(full_host_name));
                static std::atomic<u64> last_mono_class{0};
                const u64 mono_class = regs.regs[7];
                const u64 previous_class =
                    last_mono_class.exchange(mono_class, std::memory_order_relaxed);
                __android_log_print(
                    ANDROID_LOG_WARN, "LSX4Native",
                    "[EXECUTOR_LIVE_MONOSCRIPT_CACHE_ARGS] count=%d rc=%d tid=%ld host=%s "
                    "rip=0x%llx off=0x%llx class=0x%llx sameClass=%d threaded=%llu obj=0x%llx "
                    "rax=0x%llx rbx=0x%llx rcx=0x%llx r8=0x%llx r9=0x%llx r12=0x%llx "
                    "r13=0x%llx r14=0x%llx r15=0x%llx fs=0x%llx gs=0x%llx "
                    "classQ0=0x%llx classQ1=0x%llx classQ2=0x%llx classQ3=0x%llx",
                    count, full_rc, full_tid, full_host_name[0] ? full_host_name : "<none>",
                    static_cast<unsigned long long>(regs.rip),
                    static_cast<unsigned long long>(regs.rip - EbootBase),
                    static_cast<unsigned long long>(mono_class),
                    previous_class == mono_class ? 1 : 0,
                    static_cast<unsigned long long>(regs.regs[6]),
                    static_cast<unsigned long long>(regs.regs[2]),
                    static_cast<unsigned long long>(regs.regs[0]),
                    static_cast<unsigned long long>(regs.regs[3]),
                    static_cast<unsigned long long>(regs.regs[1]),
                    static_cast<unsigned long long>(regs.regs[8]),
                    static_cast<unsigned long long>(regs.regs[9]),
                    static_cast<unsigned long long>(regs.regs[12]),
                    static_cast<unsigned long long>(regs.regs[13]),
                    static_cast<unsigned long long>(regs.regs[14]),
                    static_cast<unsigned long long>(regs.regs[15]),
                    static_cast<unsigned long long>(regs.fsbase),
                    static_cast<unsigned long long>(regs.gsbase),
                    static_cast<unsigned long long>(GuestReadQword(mono_class + 0x00)),
                    static_cast<unsigned long long>(GuestReadQword(mono_class + 0x08)),
                    static_cast<unsigned long long>(GuestReadQword(mono_class + 0x10)),
                    static_cast<unsigned long long>(GuestReadQword(mono_class + 0x18)));
                __android_log_print(
                    ANDROID_LOG_WARN, "LSX4Native",
                    "[EXECUTOR_LIVE_MONOSCRIPT_GLOBALS] count=%d cache=0x%llx bits=0x%llx "
                    "mul=0x%x cacheQ08=0x%llx cacheQ38=0x%llx cacheQ40=0x%llx "
                    "cacheQ48=0x%llx",
                    count,
                    static_cast<unsigned long long>(GuestReadQword(EbootBase + 0x28b3fe0ULL)),
                    static_cast<unsigned long long>(GuestReadQword(EbootBase + 0x28b3fe8ULL)),
                    GuestReadU32(EbootBase + 0x28b3ff0ULL),
                    static_cast<unsigned long long>(
                        GuestReadQword(GuestReadQword(EbootBase + 0x28b3fe0ULL) + 0x08)),
                    static_cast<unsigned long long>(
                        GuestReadQword(GuestReadQword(EbootBase + 0x28b3fe0ULL) + 0x38)),
                    static_cast<unsigned long long>(
                        GuestReadQword(GuestReadQword(EbootBase + 0x28b3fe0ULL) + 0x40)),
                    static_cast<unsigned long long>(
                        GuestReadQword(GuestReadQword(EbootBase + 0x28b3fe0ULL) + 0x48)));
                LogMonoScriptCandidateObject("rax", count, regs.regs[0]);
                LogMonoScriptCandidateObject("rcx", count, regs.regs[1]);
                LogMonoScriptCandidateObject("rdx", count, regs.regs[2]);
                LogMonoScriptCandidateObject("rbx", count, regs.regs[3]);
                LogMonoScriptCandidateObject("rsi", count, regs.regs[6]);
                LogMonoScriptCandidateObject("rdi", count, regs.regs[7]);
                LogMonoScriptCandidateObject("r8", count, regs.regs[8]);
                LogMonoScriptCandidateObject("r9", count, regs.regs[9]);
                LogMonoScriptCandidateObject("r12", count, regs.regs[12]);
                LogMonoScriptCandidateObject("r13", count, regs.regs[13]);
                LogMonoScriptCandidateObject("r14", count, regs.regs[14]);
                LogMonoScriptCandidateObject("r15", count, regs.regs[15]);
            }
        }
    }
    if (dump_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        const u64 owner_arg = reinterpret_cast<u64>(native->m_owner->arg);
        const u64 guest_base = static_cast<u64>(guest_addr);
        if (owner_name == "UnityPreload") {
            TraceLiveUnityPreloadActiveWorkItem(count, owner_arg, guest_base, true);
        }
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_UNITY_JOB_ARG] count=%d ownerName=%s arg=0x%llx "
                            "mutex=0x%llx delta=0x%llx q0=0x%llx q1=0x%llx q2=0x%llx "
                            "q3=0x%llx q4=0x%llx q5=0x%llx q6=0x%llx q7=0x%llx",
                            count, PthreadName(native->m_owner),
                            static_cast<unsigned long long>(owner_arg),
                            static_cast<unsigned long long>(guest_base),
                            static_cast<unsigned long long>(
                                owner_arg != 0 && guest_base >= owner_arg ? guest_base - owner_arg
                                                                          : 0),
                            static_cast<unsigned long long>(GuestReadQword(owner_arg + 0x00)),
                            static_cast<unsigned long long>(GuestReadQword(owner_arg + 0x08)),
                            static_cast<unsigned long long>(GuestReadQword(owner_arg + 0x10)),
                            static_cast<unsigned long long>(GuestReadQword(owner_arg + 0x18)),
                            static_cast<unsigned long long>(GuestReadQword(owner_arg + 0x20)),
                            static_cast<unsigned long long>(GuestReadQword(owner_arg + 0x28)),
                            static_cast<unsigned long long>(GuestReadQword(owner_arg + 0x30)),
                            static_cast<unsigned long long>(GuestReadQword(owner_arg + 0x38)));
        const auto arg_bytes = GuestBytesHex(owner_arg, 0x100);
        const auto mutex_bytes =
            GuestBytesHex(guest_base >= 0x80 ? guest_base - 0x80 : guest_base, 0x120);
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_UNITY_JOB_ARG_BYTES] count=%d arg=0x%llx bytes=%s",
                            count, static_cast<unsigned long long>(owner_arg), arg_bytes.c_str());
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_UNITY_MUTEX_AROUND] count=%d base=0x%llx bytes=%s",
                            count,
                            static_cast<unsigned long long>(guest_base >= 0x80 ? guest_base - 0x80
                                                                               : guest_base),
                            mutex_bytes.c_str());
        if (executor_live_hle_flight_dump) {
            executor_live_hle_flight_dump("mutex-owner-stuck", 0, 0, 0,
                                          reinterpret_cast<u64>(guest), count);
        }
        if ((force_unitypreload_owner_dump ||
             std::getenv("EXECUTOR_TRACE_LIVE_EMU_DUMP") != nullptr ||
             std::getenv("EXECUTOR_TRACE_LIVE_MUTEX_OWNER_DUMP") != nullptr) &&
            executor_lsx4_android_dump_box64_emu_states) {
            executor_lsx4_android_dump_box64_emu_states("mutex-owner-stuck");
        }
    }
}
#else
static s32 RejectInvalidGuestMutexCell(const char*, PthreadMutexT*) {
    return 0;
}
#endif

static Pthread* CurrentMutexThread() {
#ifdef __ANDROID__
    return CurrentOrFallbackPthread();
#else
    return g_curthread;
#endif
}

#ifndef __ANDROID__
static bool PcOracleMonoSyncEnabled() {
    const char* value = std::getenv("EXECUTOR_PC_ORACLE_MONO_SYNC");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static const char* PcOracleThreadName(const Pthread* thread) {
    return thread && !thread->name.empty() ? thread->name.c_str() : "<none>";
}

static bool PcOracleRuntimeMutexCell(const void* guest_cell) {
    const auto cell = reinterpret_cast<std::uintptr_t>(guest_cell);
    return cell == 0x803397ca0ULL || (cell >= 0x800000000ULL && cell < 0x810000000ULL);
}

static bool PcOracleUnityMutexSite(const Pthread* thread, const void* guest_cell,
                                   std::uintptr_t return_off) {
    if (thread == nullptr || guest_cell == nullptr) {
        return false;
    }
    const bool unity_thread = thread->name == "Game:Main" || thread->name == "UnityPreload" ||
                              thread->name == "UnityGfxDeviceWorker";
    if (!unity_thread) {
        return false;
    }
    const auto cell = reinterpret_cast<std::uintptr_t>(guest_cell);
    const bool heap_cell = cell >= 0x200000000ULL && cell < 0x400000000ULL;
    const bool eboot_control_site =
        return_off == 0x993d9ULL || return_off == 0x99419ULL || return_off == 0x99459ULL;
    return heap_cell || eboot_control_site;
}

static void PcOracleMutexLog(const char* op, PthreadMutexT* guest_cell, PthreadMutex* native,
                             s32 ret, const void* guest_return) {
    if (!PcOracleMonoSyncEnabled()) {
        return;
    }
    const Pthread* cur = CurrentMutexThread();
    const Pthread* owner = native ? native->m_owner : nullptr;
    const auto return_va = reinterpret_cast<std::uintptr_t>(guest_return);
    const auto return_off = return_va >= 0x800000000ULL ? return_va - 0x800000000ULL : 0ULL;
    const bool mono_mutex = PcOracleRuntimeMutexCell(guest_cell);
    const bool unity_mutex = PcOracleUnityMutexSite(cur, guest_cell, return_off);
    if (!mono_mutex && !unity_mutex) {
        return;
    }
    static std::atomic_int mono_budget{40000};
    if (mono_mutex && mono_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        std::fprintf(stderr,
                     "[EXECUTOR_PC_MONO_MUTEX] op=%s thread=%p threadName=%s guest=%p native=%p "
                     "ret=%d owner=%p ownerName=%s count=%d flags=0x%x type=%u "
                     "retaddr=%p returnOff=0x%llx\n",
                     op, cur, PcOracleThreadName(cur), guest_cell, native, ret, owner,
                     PcOracleThreadName(owner), native ? native->m_count : -1,
                     native ? static_cast<u32>(native->m_flags) : 0u,
                     native ? static_cast<u32>(native->Type()) : 0u, guest_return,
                     static_cast<unsigned long long>(return_off));
    }
    static std::atomic_int unity_budget{24000};
    if (!unity_mutex || unity_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        std::fflush(stderr);
        return;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_PC_MUTEX] op=%s thread=%p threadName=%s guest=%p native=%p "
                 "ret=%d owner=%p ownerName=%s count=%d flags=0x%x type=%u "
                 "retaddr=%p returnOff=0x%llx\n",
                 op, cur, PcOracleThreadName(cur), guest_cell, native, ret, owner,
                 PcOracleThreadName(owner), native ? native->m_count : -1,
                 native ? static_cast<u32>(native->m_flags) : 0u,
                 native ? static_cast<u32>(native->Type()) : 0u, guest_return,
                 static_cast<unsigned long long>(return_off));
    std::fflush(stderr);
}

static bool IsOwnedByCurrentHostThread(const PthreadMutex*) {
    return false;
}
#endif

#ifndef __ANDROID__
static void TraceMutexInvalidCellState(const char*, PthreadMutexT*, PthreadMutex*, const char*) {}
static void RecordMutexLifecycle(const char*, PthreadMutexT*, PthreadMutex*, PthreadMutex*) {}
#endif

#define THR_MUTEX_INITIALIZER ((PthreadMutex*)NULL)
#define THR_ADAPTIVE_MUTEX_INITIALIZER ((PthreadMutex*)1)
#define THR_MUTEX_DESTROYED ((PthreadMutex*)2)

#if defined(ARCH_X86_64)
#define CPU_SPINWAIT __asm__ volatile("pause")
#elif defined(ARCH_ARM64)
#define CPU_SPINWAIT __asm__ volatile("yield")
#else
#if defined(__aarch64__)
#define CPU_SPINWAIT __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__)
#define CPU_SPINWAIT __asm__ __volatile__("pause" ::: "memory")
#else
#define CPU_SPINWAIT std::this_thread::yield()
#endif
#endif

static bool ExecutorIsModuleRegionStaticMutex(const void* mutex) {
    const auto cell = reinterpret_cast<std::uintptr_t>(mutex);
    return cell >= 0x800000000ULL && cell < 0x810000000ULL;
}

#define CHECK_AND_INIT_MUTEX                                                                       \
    if (s32 ret = RejectInvalidGuestMutexCell(__func__, mutex); ret) {                             \
        return ret;                                                                                \
    }                                                                                              \
    if (PthreadMutex* m = *mutex; m <= THR_MUTEX_DESTROYED) [[unlikely]] {                         \
        if (m == THR_MUTEX_DESTROYED && !ExecutorIsModuleRegionStaticMutex(mutex)) {               \
            TraceMutexInvalidCellState(__func__, mutex, m, "destroyed-cell");                     \
            return POSIX_EINVAL;                                                                   \
        }                                                                                          \
        if (s32 ret = InitStatic(g_curthread, mutex); ret) {                                       \
            if (ret == POSIX_EINVAL) {                                                             \
                TraceMutexInvalidCellState(__func__, mutex, m, "static-init");                    \
            }                                                                                      \
            return ret;                                                                            \
        }                                                                                          \
        m = *mutex;                                                                                \
    }

static constexpr PthreadMutexAttr PthreadMutexattrDefault = {
    .m_type = PthreadMutexType::ErrorCheck, .m_protocol = PthreadMutexProt::None, .m_ceiling = 0};

static constexpr PthreadMutexAttr PthreadMutexattrAdaptiveDefault = {
    .m_type = PthreadMutexType::AdaptiveNp, .m_protocol = PthreadMutexProt::None, .m_ceiling = 0};

static constexpr PthreadMutexAttr PthreadMutexattrRecursiveDefault = {
    .m_type = PthreadMutexType::Recursive, .m_protocol = PthreadMutexProt::None, .m_ceiling = 0};

using CallocFun = void* (*)(size_t, size_t);

static s32 MutexInit(PthreadMutexT* mutex, const PthreadMutexAttr* mutex_attr, const char* name) {
    PthreadMutex* const old_value = *mutex;
    const PthreadMutexAttr* attr;
    if (mutex_attr == nullptr) {
        attr = &PthreadMutexattrDefault;
    } else {
        attr = mutex_attr;
        if (attr->m_type < PthreadMutexType::ErrorCheck || attr->m_type >= PthreadMutexType::Max) {
            return POSIX_EINVAL;
        }
        if (attr->m_protocol > PthreadMutexProt::Protect) {
            return POSIX_EINVAL;
        }
    }
    auto* pmutex = new (std::nothrow) PthreadMutex{};
    if (pmutex == nullptr) {
        return POSIX_ENOMEM;
    }

    if (name) {
        pmutex->name = name;
    } else {
        static s32 MutexId = 0;
        pmutex->name = fmt::format("Mutex{}", MutexId++);
    }

    pmutex->m_flags = PthreadMutexFlags(attr->m_type);
    pmutex->m_owner = nullptr;
    pmutex->m_count = 0;
    pmutex->m_spinloops = 0;
    pmutex->m_yieldloops = 0;
    pmutex->m_protocol = attr->m_protocol;
    if (attr->m_type == PthreadMutexType::AdaptiveNp) {
        pmutex->m_spinloops = MUTEX_ADAPTIVE_SPINS;
    }

    *mutex = pmutex;
    RecordMutexLifecycle("init", mutex, old_value, pmutex);
    return 0;
}

static s32 InitStatic(Pthread* thread, PthreadMutexT* mutex) {
    std::scoped_lock lk{MutxStaticLock};

    if (*mutex == THR_MUTEX_INITIALIZER) {
        return MutexInit(mutex, &PthreadMutexattrDefault, nullptr);
    } else if (*mutex == THR_ADAPTIVE_MUTEX_INITIALIZER) {
        return MutexInit(mutex, &PthreadMutexattrAdaptiveDefault, nullptr);
    } else if (*mutex == THR_MUTEX_DESTROYED && ExecutorIsModuleRegionStaticMutex(mutex)) {
        return MutexInit(mutex, &PthreadMutexattrRecursiveDefault, nullptr);
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_init(PthreadMutexT* mutex,
                                          const PthreadMutexAttrT* mutex_attr) {
    if (s32 ret = RejectInvalidGuestMutexCell(__func__, mutex); ret) {
        return ret;
    }
    const PthreadMutexAttr* resolved_attr = mutex_attr ? *mutex_attr : nullptr;
    const s32 ret = MutexInit(mutex, resolved_attr, nullptr);
#ifdef __ANDROID__
    static std::atomic_int audit_budget{128};
    if (audit_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                            "[EXECUTOR_MUTEX_INIT_AUDIT] api=posix guest=%p native=%p "
                            "attrCell=%p attr=%p attrType=%u ret=%d thread=%s tid=%ld",
                            mutex, ret == 0 && mutex ? *mutex : nullptr, mutex_attr, resolved_attr,
                            resolved_attr ? static_cast<u32>(resolved_attr->m_type) : 0u, ret,
                            g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                            CurrentHostTid());
    }
#endif
    return ret;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_init_for_mono(PthreadMutexT* mutex,
                                                   const PthreadMutexAttrT* mutex_attr) {
    if (s32 ret = RejectInvalidGuestMutexCell(__func__, mutex); ret) {
        return ret;
    }
    const s32 ret = MutexInit(mutex, mutex_attr ? *mutex_attr : nullptr, "pthread_mutex_init_for_mono");
#ifdef __ANDROID__
    static std::atomic_int log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                            "[EXECUTOR_MONO_MUTEX_INIT] guest=%p native=%p attr=%p ret=%d "
                            "type=%u thread=%s tid=%ld",
                            mutex, mutex ? *mutex : nullptr, mutex_attr, ret,
                            (ret == 0 && mutex && *mutex) ? static_cast<u32>((*mutex)->Type()) : 0u,
                            g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                            CurrentHostTid());
    }
#endif
    return ret;
}

s32 PS4_SYSV_ABI scePthreadMutexInit(PthreadMutexT* mutex, const PthreadMutexAttrT* mutex_attr,
                                     const char* name) {
    if (s32 ret = RejectInvalidGuestMutexCell(__func__, mutex); ret) {
        return ret;
    }
    return MutexInit(mutex, mutex_attr ? *mutex_attr : nullptr, name);
}

#ifdef __ANDROID__
static void RetireDestroyedGuestMutex(PthreadMutex* m) {
    if (m == nullptr) {
        return;
    }
    static std::mutex retired_lock;
    static std::vector<PthreadMutex*> retired;
    std::lock_guard<std::mutex> guard(retired_lock);
    retired.push_back(m);
}
#endif

s32 PS4_SYSV_ABI posix_pthread_mutex_destroy(PthreadMutexT* mutex) {
    if (s32 ret = RejectInvalidGuestMutexCell(__func__, mutex); ret) {
        return ret;
    }
    PthreadMutexT m = *mutex;
    if (m < THR_MUTEX_DESTROYED) {
        return 0;
    }
    if (m == THR_MUTEX_DESTROYED) {
        TraceMutexInvalidCellState(__func__, mutex, m, "already-destroyed");
        return POSIX_EINVAL;
    }
    if (m->m_owner != nullptr) {
        return POSIX_EBUSY;
    }
    RecordMutexLifecycle("destroy", mutex, m, THR_MUTEX_DESTROYED);
    *mutex = THR_MUTEX_DESTROYED;
#ifdef __ANDROID__
    RetireDestroyedGuestMutex(m);
#else
    delete m;
#endif
    return 0;
}

s32 PthreadMutex::SelfTryLock() {
    switch (Type()) {
    case PthreadMutexType::ErrorCheck:
    case PthreadMutexType::Normal:
    case PthreadMutexType::AdaptiveNp:
        return POSIX_EBUSY;
    case PthreadMutexType::Recursive: {
        if (m_count + 1 > 0) {
            m_count++;
            return 0;
        }
        return POSIX_EAGAIN;
    }
    default:
        return POSIX_EINVAL;
    }
}

s32 PthreadMutex::SelfLock(const OrbisKernelTimespec* abstime, u64 usec) {
    const auto DoSleep = [&] {
        if (abstime == THR_RELTIME) {
            std::this_thread::sleep_for(std::chrono::microseconds(usec));
            return POSIX_ETIMEDOUT;
        } else {
            if (abstime->tv_sec < 0 || abstime->tv_nsec < 0 || abstime->tv_nsec >= 1000000000) {
                return POSIX_EINVAL;
            } else {
                std::this_thread::sleep_until(abstime->TimePoint());
                return POSIX_ETIMEDOUT;
            }
        }
    };
    switch (Type()) {
    case PthreadMutexType::ErrorCheck:
    case PthreadMutexType::AdaptiveNp: {
        if (abstime) {
            return DoSleep();
        }
        return POSIX_EDEADLK;
    }
    case PthreadMutexType::Normal: {
        if (abstime) {
            return DoSleep();
        }
        UNREACHABLE_MSG("Mutex deadlock occured");
        return 0;
    }
    case PthreadMutexType::Recursive: {
        if (m_count + 1 > 0) {
            m_count++;
            return 0;
        }
        return POSIX_EAGAIN;
    }
    default:
        return POSIX_EINVAL;
    }
}

s32 PthreadMutex::Lock(const OrbisKernelTimespec* abstime, u64 usec, void* guest_cell) {
    Pthread* curthread = CurrentMutexThread();
    if (m_owner == curthread || IsOwnedByCurrentHostThread(this)) {
        return SelfLock(abstime, usec);
    }

    if (m_protocol == PthreadMutexProt::None) [[likely]] {
        s32 count = m_spinloops;
        while (count--) {
            if (m_lock.try_lock()) {
                m_owner = curthread;
#ifdef __ANDROID__
        ExecutorNoteGuestMutexAcquired(this);
#endif
#ifdef __ANDROID__
                m_owner_host_tid =
                    ExecutorMutexHotTraceEnabled() ? CurrentHostTid() : 0;
#endif
                return 0;
            }
            CPU_SPINWAIT;
        }

        count = m_yieldloops;
        while (count--) {
            std::this_thread::yield();
            if (m_lock.try_lock()) {
                m_owner = curthread;
#ifdef __ANDROID__
        ExecutorNoteGuestMutexAcquired(this);
#endif
#ifdef __ANDROID__
                m_owner_host_tid =
                    ExecutorMutexHotTraceEnabled() ? CurrentHostTid() : 0;
#endif
                return 0;
            }
        }
    }

    s32 ret = 0;
    if (abstime == nullptr) {
#ifdef __ANDROID__
        if (!ExecutorAlertableMutexPollingEnabled()) {
            const bool record_stall = ExecutorStallMutexLedgerEnabled();
            if (record_stall) {
                RecordMutexWaitSnapshot(curthread, this, guest_cell, m_owner);
            }
            m_lock.lock();
            if (record_stall) {
                EraseMutexWaitSnapshot(curthread, this);
            }
        } else {
            const auto wait_start = std::chrono::steady_clock::now();
            const bool light_oracle = ExecutorLightOracleMode();
            auto next_log = wait_start + (light_oracle ? std::chrono::seconds(5)
                                                       : std::chrono::milliseconds(100));
            auto next_unitypreload_signal_check = wait_start;
            bool dumped_once = false;
            RecordMutexWaitSnapshot(curthread, this, guest_cell, m_owner);
            while (!m_lock.try_lock_for(std::chrono::milliseconds(1))) {
                RecordMutexWaitSnapshot(curthread, this, guest_cell, m_owner);
                (void)RunPendingMonoSuspendSigusr1AtSafePoint(curthread, guest_cell);
                const auto now = std::chrono::steady_clock::now();
                if (now >= next_unitypreload_signal_check) {
                    const bool signaled_owner =
                        SignalMonoSuspendForUnityPreloadBlockingLock(guest_cell, this);
                    if (signaled_owner) {
                        static std::atomic_int log_budget{64};
                        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                            __android_log_print(
                                ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MONO_MUTEX_SIGNAL_ASSIST] lock_wait_signal_owner "
                                "guest=%p native=%p owner=%p ownerName=%s cur=%p thread=%s tid=%ld",
                                guest_cell, this, m_owner, PthreadName(m_owner), curthread,
                                g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                                static_cast<long>(syscall(__NR_gettid)));
                        }
                    }
                    next_unitypreload_signal_check = now + std::chrono::milliseconds(10);
                }
                if (now >= next_log) {
                    const auto elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start)
                            .count();
                    __android_log_print(
                        ANDROID_LOG_WARN, "LSX4Mutex",
                        "[EXECUTOR_MUTEX_WAIT_STUCK] thread=%s guest=%p native=%p elapsedMs=%lld "
                        "owner=%p ownerName=%s ownerState=%s ownerTid=%ld ownerHostTid=%ld "
                        "cur=%p curTid=%ld flags=0x%x type=%u count=%d hostTid=%ld",
                        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>", guest_cell,
                        this, static_cast<long long>(elapsed_ms), m_owner, PthreadName(m_owner),
                        PthreadStateName(m_owner), PthreadGuestTid(m_owner), m_owner_host_tid,
                        curthread, PthreadGuestTid(curthread), static_cast<u32>(m_flags),
                        static_cast<u32>(Type()), m_count, CurrentHostTid());
                    const auto guest_addr = reinterpret_cast<std::uintptr_t>(guest_cell);
                    const bool force_runtime_owner_pc =
                        light_oracle && (IsLiveMonoRuntimeMutexCell(guest_cell) ||
                                         (guest_addr == 0x802898668ULL &&
                                          ThreadNameContains(curthread, "UnityGfxDeviceWorker")));
                    if ((!light_oracle || force_runtime_owner_pc) &&
                        (!dumped_once || force_runtime_owner_pc)) {
                        dumped_once = true;
                        if (auto* get_guest_thread_regs = ResolveGuestThreadRegsByPthread()) {
                            u64 rip = 0;
                            u64 rsp = 0;
                            u64 rbp = 0;
                            long fex_tid = 0;
                            char fex_host_name[32]{};
                            const auto cur_pthread =
                                g_curthread ? g_curthread->native_thr.GetHandle() : 0;
                            const int pc_rc =
                                get_guest_thread_regs(cur_pthread, &rip, &rsp, &rbp, &fex_tid,
                                                      fex_host_name, sizeof(fex_host_name));
                            const std::string symbol = SymbolizeOwnerPcForLog(rip);
                            __android_log_print(
                                ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MUTEX_WAIT_PC] thread=%s guest=%p native=%p rc=%d "
                                "pthread=0x%zx fexTid=%ld fexHost=%s rip=0x%llx rsp=0x%llx "
                                "rbp=0x%llx %s",
                                g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                                guest_cell, this, pc_rc, static_cast<std::size_t>(cur_pthread),
                                fex_tid, fex_host_name[0] ? fex_host_name : "<none>",
                                static_cast<unsigned long long>(rip),
                                static_cast<unsigned long long>(rsp),
                                static_cast<unsigned long long>(rbp), symbol.c_str());
                        }
                        if (m_owner != nullptr) {
                            TraceLiveMonoRuntimeMutexOwnerPc("wait_stuck_owner", 1, m_owner,
                                                             force_runtime_owner_pc,
                                                             m_owner_host_tid);
                        }
                    }
                    next_log =
                        now + (light_oracle ? std::chrono::seconds(5) : std::chrono::seconds(1));
                }
                std::this_thread::yield();
            }
            ClearMutexWaitSnapshot(curthread, this);
        }
#else
        m_lock.lock();
#endif
    } else if (abstime != THR_RELTIME && (abstime->tv_nsec < 0 || abstime->tv_nsec >= 1000000000))
        [[unlikely]] {
        ret = POSIX_EINVAL;
    } else {
        const bool record_stall = ExecutorStallMutexLedgerEnabled();
        if (record_stall) {
            RecordMutexWaitSnapshot(curthread, this, guest_cell, m_owner);
        }
        if (abstime == THR_RELTIME) {
            ret = m_lock.try_lock_for(std::chrono::microseconds(usec)) ? 0 : POSIX_ETIMEDOUT;
        } else {
            ret = m_lock.try_lock_until(abstime->TimePoint()) ? 0 : POSIX_ETIMEDOUT;
        }
        if (record_stall) {
            EraseMutexWaitSnapshot(curthread, this);
        }
    }
    if (ret == 0) {
        m_owner = curthread;
#ifdef __ANDROID__
        ExecutorNoteGuestMutexAcquired(this);
#endif
#ifdef __ANDROID__
        m_owner_host_tid =
            ExecutorMutexHotTraceEnabled() ? CurrentHostTid() : 0;
#endif
    }
    return ret;
}

s32 PthreadMutex::TryLock() {
    Pthread* curthread = CurrentMutexThread();
    if (m_owner == curthread || IsOwnedByCurrentHostThread(this)) {
        return SelfTryLock();
    }
    const s32 ret = m_lock.try_lock() ? 0 : POSIX_EBUSY;
    if (ret == 0) {
        m_owner = curthread;
#ifdef __ANDROID__
        ExecutorNoteGuestMutexAcquired(this);
#endif
#ifdef __ANDROID__
        m_owner_host_tid =
            ExecutorMutexHotTraceEnabled() ? CurrentHostTid() : 0;
#endif
    }
    return ret;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_trylock(PthreadMutexT* mutex) {
    CHECK_AND_INIT_MUTEX
#ifndef __ANDROID__
    const void* pc_oracle_return = __builtin_return_address(0);
    PcOracleMutexLog("trylock_enter", mutex, *mutex, 0, pc_oracle_return);
#else
    TraceMonoRuntimeMutexLedger("trylock", "enter", mutex, *mutex);
#endif
    s32 ret = (*mutex)->TryLock();
#ifdef __ANDROID__
    const bool mono_abba_signal_contract = IsMonoAbbaPcOracleSignalContractEnabled();
    if (ret == 0 && mono_abba_signal_contract) {
        ClearUnityPreloadControlSignalState(mutex);
    }
    TraceMonoRuntimeMutexLedger("trylock", "return", mutex, *mutex, ret);
    TraceLightOracleMutexContentionSite(mutex, *mutex, ret);
    if (ShouldYieldUnityPreloadMutexTryLock(mutex, *mutex, ret)) {
        static std::atomic_int log_budget{32};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_LIVE_MUTEX_YIELD] guest=%p native=%p owner=%p "
                                "ownerName=%s cur=%p curName=%s ret=%d",
                                mutex, *mutex, (*mutex)->m_owner, PthreadName((*mutex)->m_owner),
                                CurrentMutexThread(),
                                g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                                ret);
        }
        usleep(5000);
    }
    const bool unity_preload_group_signal_sent =
        mono_abba_signal_contract && HandleMonoSuspendTryLockAbbaCycle(mutex, *mutex, ret);
    const bool mono_suspend_signal_sent =
        unity_preload_group_signal_sent || ShouldBypassMonoSuspendTryLock(mutex, *mutex, ret);
    if (mono_suspend_signal_sent) {
        auto* mp = *mutex;
        static std::atomic_int log_budget{128};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MONO_MUTEX_SIGNAL_ASSIST] trylock_busy_signal_only guest=%p "
                                "native=%p owner=%p cur=%p thread=%s tid=%ld",
                                mutex, mp, mp ? mp->m_owner : nullptr, CurrentMutexThread(),
                                g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                                static_cast<long>(syscall(__NR_gettid)));
        }
    }
    TraceUnityMutex("trylock", mutex, *mutex, ret);
    TraceUnityMutexOwnerEvent("trylock", mutex, *mutex, ret);
    TraceUnexpectedMutexReturn("trylock", mutex, *mutex, ret);
    TraceLiveStuckMutexOwner("trylock", mutex, *mutex, ret);
#endif
#ifndef __ANDROID__
    PcOracleMutexLog("trylock_return", mutex, *mutex, ret, pc_oracle_return);
#endif
    return ret;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_lock(PthreadMutexT* mutex) {
#ifdef __ANDROID__
    const bool relax_guest_mutex = ShouldRelaxGuestMutex(mutex);
    if (relax_guest_mutex) {
        static std::atomic_int log_budget{64};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                                "[EXECUTOR_MUTEX_RELAX] lock no-op guest=%p tid=%ld",
                                mutex, static_cast<long>(syscall(__NR_gettid)));
        }
        return 0;
    }
#endif
    CHECK_AND_INIT_MUTEX
#ifdef __ANDROID__
    const bool trace_mutex = ExecutorMutexHotTraceEnabled();
    if (trace_mutex) {
        TraceMutexLockEnter(mutex, *mutex);
        TraceMonoRuntimeMutexLedger("lock", "enter", mutex, *mutex);
        TraceLiveMonoRuntimeMutex("lock_enter", mutex, *mutex);
    }
#else
    const void* pc_oracle_return = __builtin_return_address(0);
    PcOracleMutexLog("lock_enter", mutex, *mutex, 0, pc_oracle_return);
#endif
    const s32 ret = (*mutex)->Lock(nullptr, 0, mutex);
#ifdef __ANDROID__
    if (ret == 0) {
        ClearUnityPreloadControlSignalState(mutex);
    }
    if (trace_mutex) {
        TraceMonoRuntimeMutexLedger("lock", "return", mutex, *mutex, ret);
        TraceLiveMonoRuntimeMutex("lock_return", mutex, *mutex, ret);
        TraceUnityMutex("lock", mutex, *mutex, ret);
        TraceUnityMutexOwnerEvent("lock", mutex, *mutex, ret);
    }
    TraceUnexpectedMutexReturn("lock", mutex, *mutex, ret);
#else
    PcOracleMutexLog("lock_return", mutex, *mutex, ret, pc_oracle_return);
#endif
    return ret;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_timedlock(PthreadMutexT* mutex,
                                               const OrbisKernelTimespec* abstime) {
    CHECK_AND_INIT_MUTEX
#ifndef __ANDROID__
    const void* pc_oracle_return = __builtin_return_address(0);
    PcOracleMutexLog("timedlock_enter", mutex, *mutex, 0, pc_oracle_return);
#else
    TraceMonoRuntimeMutexLedger("timedlock", "enter", mutex, *mutex);
#endif
    const s32 ret = (*mutex)->Lock(abstime, 0, mutex);
#ifdef __ANDROID__
    if (ret == 0) {
        ClearUnityPreloadControlSignalState(mutex);
    }
    TraceMonoRuntimeMutexLedger("timedlock", "return", mutex, *mutex, ret);
    TraceLiveMonoRuntimeMutex("timedlock_return", mutex, *mutex, ret);
    TraceUnityMutex("timedlock", mutex, *mutex, ret);
    TraceUnityMutexOwnerEvent("timedlock", mutex, *mutex, ret);
    TraceUnexpectedMutexReturn("timedlock", mutex, *mutex, ret);
#else
    PcOracleMutexLog("timedlock_return", mutex, *mutex, ret, pc_oracle_return);
#endif
    return ret;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_reltimedlock_np(PthreadMutexT* mutex, u64 usec) {
    CHECK_AND_INIT_MUTEX
#ifndef __ANDROID__
    const void* pc_oracle_return = __builtin_return_address(0);
    PcOracleMutexLog("reltimedlock_enter", mutex, *mutex, 0, pc_oracle_return);
#else
    TraceMonoRuntimeMutexLedger("reltimedlock", "enter", mutex, *mutex);
#endif
    const s32 ret = (*mutex)->Lock(THR_RELTIME, usec, mutex);
#ifdef __ANDROID__
    if (ret == 0) {
        ClearUnityPreloadControlSignalState(mutex);
    }
    TraceMonoRuntimeMutexLedger("reltimedlock", "return", mutex, *mutex, ret);
    TraceLiveMonoRuntimeMutex("reltimedlock_return", mutex, *mutex, ret);
    TraceUnityMutex("reltimedlock", mutex, *mutex, ret);
    TraceUnityMutexOwnerEvent("reltimedlock", mutex, *mutex, ret);
    TraceUnexpectedMutexReturn("reltimedlock", mutex, *mutex, ret);
#else
    PcOracleMutexLog("reltimedlock_return", mutex, *mutex, ret, pc_oracle_return);
#endif
    return ret;
}

s32 PthreadMutex::Unlock() {
    Pthread* curthread = CurrentMutexThread();
    if (m_owner != curthread && !IsOwnedByCurrentHostThread(this)) [[unlikely]] {
        return POSIX_EPERM;
    }

    if (Type() == PthreadMutexType::Recursive && m_count > 0) [[unlikely]] {
        m_count--;
    } else {
        const bool deferred = True(m_flags & PthreadMutexFlags::Deferred);
        m_flags &= ~PthreadMutexFlags::Deferred;

        Pthread* wake_thread = m_owner != nullptr ? m_owner : curthread;
        m_owner = nullptr;
#ifdef __ANDROID__
        ExecutorNoteGuestMutexReleased(this);
#endif
#ifdef __ANDROID__
        m_owner_host_tid = 0;
#endif
        m_lock.unlock();

        if (wake_thread->will_sleep == 0 && deferred) {
#ifdef __ANDROID__
            TraceFiosDeferredWakeRelease(this, wake_thread, deferred);
#endif
            wake_thread->WakeAll();
        }
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_unlock(PthreadMutexT* mutex) {
#ifdef __ANDROID__
    const bool relax_guest_mutex = ShouldRelaxGuestMutex(mutex);
    if (relax_guest_mutex) {
        static std::atomic_int log_budget{64};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Mutex",
                                "[EXECUTOR_MUTEX_RELAX] unlock no-op guest=%p tid=%ld",
                                mutex, static_cast<long>(syscall(__NR_gettid)));
        }
        return 0;
    }
    if (s32 ret = RejectInvalidGuestMutexCell(__func__, mutex); ret) {
        return ret;
    }
#endif
    PthreadMutex* mp = *mutex;
    if (mp <= THR_MUTEX_DESTROYED) [[unlikely]] {
        if (mp == THR_MUTEX_DESTROYED) {
#ifdef __ANDROID__
            if (ExecutorIsModuleRegionStaticMutex(mutex)) {
                return 0;
            }
#endif
            TraceMutexInvalidCellState(__func__, mutex, mp, "destroyed-cell");
            return POSIX_EINVAL;
        }
        return POSIX_EPERM;
    }
#ifndef __ANDROID__
    const void* pc_oracle_return = __builtin_return_address(0);
    PcOracleMutexLog("unlock_enter", mutex, mp, 0, pc_oracle_return);
#else
    const bool trace_mutex = ExecutorMutexHotTraceEnabled();
    if (trace_mutex) {
        TraceMonoRuntimeMutexLedger("unlock", "enter", mutex, mp);
    }
#endif
#ifdef __ANDROID__
    if (ShouldBypassMonoSuspendOwnerMismatchUnlock(mutex, mp)) {
        static std::atomic_int log_budget{128};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MONO_MUTEX_BYPASS] "
                                "unlock_owner_mismatch_virtual_ok guest=%p native=%p "
                                "owner=%p cur=%p tid=%ld thread=%s flags=0x%x type=%u",
                                mutex, mp, mp ? mp->m_owner : nullptr,
                                CurrentMutexThread(), static_cast<long>(syscall(__NR_gettid)),
                                g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                                mp ? static_cast<u32>(mp->m_flags) : 0u,
                                mp ? static_cast<u32>(mp->Type()) : 0u);
        }
        ForgetMonoSuspendStolenMutex(mp);
        return 0;
    }
#endif
    const s32 ret = mp->Unlock();
#ifdef __ANDROID__
    if (ret == 0) {
        ClearUnityPreloadControlSignalState(mutex);
    }
    if (trace_mutex) {
        TraceMonoRuntimeMutexLedger("unlock", "return", mutex, mp, ret);
    }
    if (ret == 0) {
        ForgetMonoSuspendStolenMutex(mp);
    }
    if (trace_mutex) {
        TraceLiveMonoRuntimeMutex("unlock_return", mutex, mp, ret);
        TraceUnityMutex("unlock", mutex, mp, ret);
        TraceUnityMutexOwnerEvent("unlock", mutex, mp, ret);
    }
    TraceUnexpectedMutexReturn("unlock", mutex, mp, ret);
    if (ret != 0) {
        static const bool verbose_api_results =
            std::getenv("EXECUTOR_VERBOSE_MUTEX_API_RESULTS") != nullptr;
        if (verbose_api_results) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                "[EXECUTOR_MUTEX_ERROR] unlock guest=%p native=%p ret=%d tid=%ld "
                                "owner=%p cur=%p flags=0x%x type=%u count=%d",
                                mutex, mp, ret, CurrentHostTid(),
                                mp ? mp->m_owner : nullptr, CurrentMutexThread(),
                                mp ? static_cast<u32>(mp->m_flags) : 0u,
                                mp ? static_cast<u32>(mp->Type()) : 0u,
                                mp ? mp->m_count : 0);
        }
        if (ret == POSIX_EPERM && ShouldTolerateUnityOwnerMismatchUnlock(mutex, mp)) {
            mp->m_owner = nullptr;
            mp->m_count = 0;
#ifdef __ANDROID__
            mp->m_owner_host_tid = 0;
#endif
            mp->m_flags &= ~PthreadMutexFlags::Deferred;
            mp->m_lock.unlock();
            static std::atomic_int log_budget{128};
            if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                    "[EXECUTOR_MUTEX_TOLERATE] unlock_owner_mismatch_as_ok "
                                    "guest=%p native=%p tid=%ld thread=%s",
                                    mutex, mp, static_cast<long>(syscall(__NR_gettid)),
                                    g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>");
            }
            return 0;
        }
        if (ret == POSIX_EPERM && ConsumeMonoSuspendStolenMutex(mp)) {
            static std::atomic_int log_budget{128};
            if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                    "[EXECUTOR_MONO_MUTEX_BYPASS] "
                                    "unlock_stolen_owner_as_ok guest=%p native=%p "
                                    "owner=%p cur=%p tid=%ld thread=%s",
                                    mutex, mp, mp ? mp->m_owner : nullptr,
                                    CurrentMutexThread(), static_cast<long>(syscall(__NR_gettid)),
                                    g_curthread ? g_curthread->name.c_str()
                                                : "<no-gcurthread>");
            }
            return 0;
        }
        if (ret == POSIX_EPERM && ShouldTolerateUnityUnlockedUnlock(mutex, mp)) {
            mp->m_count = 0;
#ifdef __ANDROID__
            mp->m_owner_host_tid = 0;
#endif
            mp->m_flags &= ~PthreadMutexFlags::Deferred;
            static std::atomic_int log_budget{128};
            if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Mutex",
                                    "[EXECUTOR_MUTEX_TOLERATE] unlock_unowned_as_ok "
                                    "guest=%p native=%p tid=%ld thread=%s",
                                    mutex, mp, static_cast<long>(syscall(__NR_gettid)),
                                    g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>");
            }
            return 0;
        }
    }
    if (ret == 0 && ExecutorLiveMonoSafepointSignalEnabled()) {
        (void)RunPendingMonoSuspendSigusr1AtSafePoint(CurrentMutexThread(), mutex);
    }
#endif
#ifndef __ANDROID__
    PcOracleMutexLog("unlock_return", mutex, mp, ret, pc_oracle_return);
#endif
    return ret;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_getspinloops_np(PthreadMutexT* mutex, int* count) {
    CHECK_AND_INIT_MUTEX
    *count = (*mutex)->m_spinloops;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_setspinloops_np(PthreadMutexT* mutex, s32 count) {
    CHECK_AND_INIT_MUTEX(*mutex)->m_spinloops = count;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_getyieldloops_np(PthreadMutexT* mutex, int* count) {
    CHECK_AND_INIT_MUTEX
    *count = (*mutex)->m_yieldloops;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_setyieldloops_np(PthreadMutexT* mutex, s32 count) {
    CHECK_AND_INIT_MUTEX(*mutex)->m_yieldloops = count;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutex_isowned_np(PthreadMutexT* mutex) {
    if (s32 ret = RejectInvalidGuestMutexCell(__func__, mutex); ret) {
        return ret;
    }
    PthreadMutex* m = *mutex;
    if (m <= THR_MUTEX_DESTROYED) {
        return 0;
    }
    return m->m_owner == CurrentMutexThread() || IsOwnedByCurrentHostThread(m);
}

s32 PthreadMutex::IsOwned(Pthread* curthread) const {
    curthread = curthread ? curthread : CurrentMutexThread();
    if (this <= THR_MUTEX_DESTROYED) [[unlikely]] {
        if (this == THR_MUTEX_DESTROYED) {
            return POSIX_EINVAL;
        }
        return POSIX_EPERM;
    }
    if (m_owner != curthread && !IsOwnedByCurrentHostThread(this)) {
        return POSIX_EPERM;
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_init(PthreadMutexAttrT* attr) {
    auto pattr = new (std::nothrow) PthreadMutexAttr{};
    if (pattr == nullptr) {
        return POSIX_ENOMEM;
    }
    memcpy(pattr, &PthreadMutexattrDefault, sizeof(PthreadMutexAttr));
    *attr = pattr;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_setkind_np(PthreadMutexAttrT* attr,
                                                    PthreadMutexType kind) {
    if (attr == nullptr || *attr == nullptr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    (*attr)->m_type = kind;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_getkind_np(PthreadMutexAttrT attr) {
    if (attr == nullptr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    return static_cast<int>(attr->m_type);
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_settype(PthreadMutexAttrT* attr, PthreadMutexType type) {
    if (attr == nullptr || *attr == nullptr || type < PthreadMutexType::ErrorCheck ||
        type >= PthreadMutexType::Max) {
        return POSIX_EINVAL;
    }
    (*attr)->m_type = type;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_gettype(PthreadMutexAttrT* attr, PthreadMutexType* type) {
    if (attr == nullptr || *attr == nullptr || (*attr)->m_type >= PthreadMutexType::Max) {
        return POSIX_EINVAL;
    }
    *type = (*attr)->m_type;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_destroy(PthreadMutexAttrT* attr) {
    if (attr == nullptr || *attr == nullptr) {
        return POSIX_EINVAL;
    }
    delete *attr;
    *attr = nullptr;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_getprotocol(PthreadMutexAttrT* mattr,
                                                     PthreadMutexProt* protocol) {
    if (mattr == nullptr || *mattr == nullptr || protocol == nullptr) {
        return POSIX_EINVAL;
    }
    *protocol = (*mattr)->m_protocol;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_setprioceiling(PthreadMutexAttrT* attr,
                                                        int prioceiling) {
    if (attr == nullptr || *attr == nullptr ||
        (*attr)->m_protocol != PthreadMutexProt::Protect ||
        prioceiling > ORBIS_KERNEL_PRIO_FIFO_HIGHEST ||
        prioceiling < ORBIS_KERNEL_PRIO_FIFO_LOWEST) {
        return POSIX_EINVAL;
    }
    (*attr)->m_ceiling = prioceiling;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_getprioceiling(PthreadMutexAttrT* attr,
                                                        int* prioceiling) {
    if (attr == nullptr || *attr == nullptr || prioceiling == nullptr ||
        (*attr)->m_protocol != PthreadMutexProt::Protect) {
        return POSIX_EINVAL;
    }
    *prioceiling = (*attr)->m_ceiling;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_setprotocol(PthreadMutexAttrT* mattr,
                                                     PthreadMutexProt protocol) {
    if (mattr == nullptr || *mattr == nullptr || (protocol < PthreadMutexProt::None) ||
        (protocol > PthreadMutexProt::Protect)) {
        return POSIX_EINVAL;
    }
    (*mattr)->m_protocol = protocol;
    return 0;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_setpshared(PthreadMutexAttrT* attr, s32 pshared) {
    constexpr s32 POSIX_PTHREAD_PROCESS_PRIVATE = 0;
    if (!attr || !*attr || pshared != POSIX_PTHREAD_PROCESS_PRIVATE) {
        return POSIX_EINVAL;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI posix_pthread_mutexattr_getpshared(PthreadMutexAttrT* attr, s32* pshared) {
    if (attr == nullptr || *attr == nullptr || pshared == nullptr) {
        return POSIX_EINVAL;
    }
    *pshared = 0;
    return ORBIS_OK;
}

void RegisterMutex(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("ttHNfU+qDBU", "libScePosix", 1, "libkernel", posix_pthread_mutex_init);
    LIB_FUNCTION("gKqzW-zWhvY", "libScePosix", 1, "libkernel", posix_pthread_mutex_isowned_np);
    LIB_FUNCTION("7H0iTOciTLo", "libScePosix", 1, "libkernel", posix_pthread_mutex_lock);
    LIB_FUNCTION("Io9+nTKXZtA", "libScePosix", 1, "libkernel", posix_pthread_mutex_timedlock);
    LIB_FUNCTION("2Z+PpY6CaJg", "libScePosix", 1, "libkernel", posix_pthread_mutex_unlock);
    LIB_FUNCTION("x4vQj3JKKmc", "libScePosix", 1, "libkernel",
                 posix_pthread_mutex_getspinloops_np);
    LIB_FUNCTION("OxEIUqkByy4", "libScePosix", 1, "libkernel",
                 posix_pthread_mutex_getyieldloops_np);
    LIB_FUNCTION("5-ncLMtL5+g", "libScePosix", 1, "libkernel",
                 posix_pthread_mutex_setspinloops_np);
    LIB_FUNCTION("frFuGprJmPc", "libScePosix", 1, "libkernel",
                 posix_pthread_mutex_setyieldloops_np);
    LIB_FUNCTION("ltCfaGr2JGE", "libScePosix", 1, "libkernel", posix_pthread_mutex_destroy);
    LIB_FUNCTION("dQHWEsJtoE4", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_init);
    LIB_FUNCTION("U6SNV+RnyLQ", "libScePosix", 1, "libkernel",
                 posix_pthread_mutexattr_getkind_np);
    LIB_FUNCTION("+m8+quqOwhM", "libScePosix", 1, "libkernel",
                 posix_pthread_mutexattr_getprioceiling);
    LIB_FUNCTION("yDaWxUE50s0", "libScePosix", 1, "libkernel",
                 posix_pthread_mutexattr_getprotocol);
    LIB_FUNCTION("PmL-TwKUzXI", "libScePosix", 1, "libkernel",
                 posix_pthread_mutexattr_getpshared);
    LIB_FUNCTION("GZFlI7RhuQo", "libScePosix", 1, "libkernel",
                 posix_pthread_mutexattr_gettype);
    LIB_FUNCTION("J9rlRuQ8H5s", "libScePosix", 1, "libkernel",
                 posix_pthread_mutexattr_setkind_np);
    LIB_FUNCTION("ZLvf6lVAc4M", "libScePosix", 1, "libkernel",
                 posix_pthread_mutexattr_setprioceiling);
    LIB_FUNCTION("mDmgMOGVUqg", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_settype);
    LIB_FUNCTION("5txKfcMUAok", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_setprotocol);
    LIB_FUNCTION("HF7lK46xzjY", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_destroy);
    LIB_FUNCTION("K-jXhbt2gn4", "libScePosix", 1, "libkernel", posix_pthread_mutex_trylock);
    LIB_FUNCTION("EXv3ztGqtDM", "libScePosix", 1, "libkernel", posix_pthread_mutexattr_setpshared);
    LIB_FUNCTION("hLoEhSBhi84", "libScePosix", 1, "libkernel",
                 posix_pthread_mutex_init_for_mono);

    LIB_FUNCTION("ttHNfU+qDBU", "libkernel", 1, "libkernel", posix_pthread_mutex_init);
    LIB_FUNCTION("gKqzW-zWhvY", "libkernel", 1, "libkernel", posix_pthread_mutex_isowned_np);
    LIB_FUNCTION("7H0iTOciTLo", "libkernel", 1, "libkernel", posix_pthread_mutex_lock);
    LIB_FUNCTION("Io9+nTKXZtA", "libkernel", 1, "libkernel", posix_pthread_mutex_timedlock);
    LIB_FUNCTION("2Z+PpY6CaJg", "libkernel", 1, "libkernel", posix_pthread_mutex_unlock);
    LIB_FUNCTION("x4vQj3JKKmc", "libkernel", 1, "libkernel",
                 posix_pthread_mutex_getspinloops_np);
    LIB_FUNCTION("OxEIUqkByy4", "libkernel", 1, "libkernel",
                 posix_pthread_mutex_getyieldloops_np);
    LIB_FUNCTION("5-ncLMtL5+g", "libkernel", 1, "libkernel",
                 posix_pthread_mutex_setspinloops_np);
    LIB_FUNCTION("frFuGprJmPc", "libkernel", 1, "libkernel",
                 posix_pthread_mutex_setyieldloops_np);
    LIB_FUNCTION("ltCfaGr2JGE", "libkernel", 1, "libkernel", posix_pthread_mutex_destroy);
    LIB_FUNCTION("dQHWEsJtoE4", "libkernel", 1, "libkernel", posix_pthread_mutexattr_init);
    LIB_FUNCTION("U6SNV+RnyLQ", "libkernel", 1, "libkernel",
                 posix_pthread_mutexattr_getkind_np);
    LIB_FUNCTION("+m8+quqOwhM", "libkernel", 1, "libkernel",
                 posix_pthread_mutexattr_getprioceiling);
    LIB_FUNCTION("yDaWxUE50s0", "libkernel", 1, "libkernel",
                 posix_pthread_mutexattr_getprotocol);
    LIB_FUNCTION("PmL-TwKUzXI", "libkernel", 1, "libkernel",
                 posix_pthread_mutexattr_getpshared);
    LIB_FUNCTION("GZFlI7RhuQo", "libkernel", 1, "libkernel",
                 posix_pthread_mutexattr_gettype);
    LIB_FUNCTION("J9rlRuQ8H5s", "libkernel", 1, "libkernel",
                 posix_pthread_mutexattr_setkind_np);
    LIB_FUNCTION("ZLvf6lVAc4M", "libkernel", 1, "libkernel",
                 posix_pthread_mutexattr_setprioceiling);
    LIB_FUNCTION("mDmgMOGVUqg", "libkernel", 1, "libkernel", posix_pthread_mutexattr_settype);
    LIB_FUNCTION("5txKfcMUAok", "libkernel", 1, "libkernel",
                 posix_pthread_mutexattr_setprotocol);
    LIB_FUNCTION("HF7lK46xzjY", "libkernel", 1, "libkernel", posix_pthread_mutexattr_destroy);
    LIB_FUNCTION("K-jXhbt2gn4", "libkernel", 1, "libkernel", posix_pthread_mutex_trylock);
    LIB_FUNCTION("EXv3ztGqtDM", "libkernel", 1, "libkernel", posix_pthread_mutexattr_setpshared);
    LIB_FUNCTION("hLoEhSBhi84", "libkernel", 1, "libkernel",
                 posix_pthread_mutex_init_for_mono);

    LIB_FUNCTION("cmo1RIYva9o", "libkernel", 1, "libkernel", ORBIS(scePthreadMutexInit));
    LIB_FUNCTION("W6OrTBO95UY", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutex_isowned_np));
    LIB_FUNCTION("2Of0f+3mhhE", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_destroy));
    LIB_FUNCTION("F8bUHwAG284", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutexattr_init));
    LIB_FUNCTION("rH2mWEndluc", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_getkind_np));
    LIB_FUNCTION("SgjMpyH9Z9I", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_getprioceiling));
    LIB_FUNCTION("GoTmFeui+hQ", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_getprotocol));
    LIB_FUNCTION("losEubHc64c", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_getpshared));
    LIB_FUNCTION("gquEhBrS2iw", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_gettype));
    LIB_FUNCTION("UWZbVSFze24", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_setkind_np));
    LIB_FUNCTION("532IaQguwMg", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_setprioceiling));
    LIB_FUNCTION("smWEktiyyG0", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_destroy));
    LIB_FUNCTION("iMp8QpE+XO4", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_settype));
    LIB_FUNCTION("1FGvU0i9saQ", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_setprotocol));
    LIB_FUNCTION("mxKx9bxXF2I", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutexattr_setpshared));
    LIB_FUNCTION("9UK1vLZQft4", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_lock));
    LIB_FUNCTION("tn3VlD0hG60", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_unlock));
    LIB_FUNCTION("pOmNmyRKlIE", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutex_getspinloops_np));
    LIB_FUNCTION("AWS3NyViL9o", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutex_getyieldloops_np));
    LIB_FUNCTION("42YkUouoMI0", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutex_setspinloops_np));
    LIB_FUNCTION("bP+cqFmBW+A", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutex_setyieldloops_np));
    LIB_FUNCTION("upoVrzMHFeE", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_trylock));
    LIB_FUNCTION("IafI2PxcPnQ", "libkernel", 1, "libkernel",
                 ORBIS(posix_pthread_mutex_reltimedlock_np));
    LIB_FUNCTION("qH1gXoq71RY", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutex_init));
    LIB_FUNCTION("n2MMpvU8igI", "libkernel", 1, "libkernel", ORBIS(posix_pthread_mutexattr_init));
}

}
