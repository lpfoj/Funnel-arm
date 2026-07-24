// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef __ANDROID__
#include <android/log.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" void executor_lsx4_android_dump_box64_emu_states(const char* reason)
    __attribute__((weak));
extern "C" int executor_live_get_current_hle_call_site(std::uint64_t* guest_return,
                                                       std::uint64_t* return_off,
                                                       std::uint64_t* arg0,
                                                       char* symbol,
                                                       std::size_t symbol_size,
                                                       char* module,
                                                       std::size_t module_size)
    __attribute__((weak));
extern "C" void executor_live_dump_current_hle_context(const char* reason,
                                                       std::uint64_t focus)
    __attribute__((weak));
extern "C" void executor_jit_dump_thread_states(const char* reason) __attribute__((weak));

static std::atomic<std::uintptr_t> g_executor_unity_gfx_worker{0};

extern "C" void executor_live_note_unity_gfx_worker(void* worker) {
    const auto value = reinterpret_cast<std::uintptr_t>(worker);
    g_executor_unity_gfx_worker.store(value, std::memory_order_release);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_UNITY_GFX_WORKER_NOTE] worker=%p", worker);
}

static bool ExecutorLightOracleMode() {
    const char* value = std::getenv("EXECUTOR_LIGHT_ORACLE");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}
#endif

#include "core/libraries/kernel/sync/semaphore.h"

#include "common/logging/log.h"
#include "core/memory.h"
#include "common/slot_vector.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/libs.h"
#ifdef __ANDROID__
#include "video_core/amdgpu/liverpool.h"
#endif

namespace Libraries::Kernel {

constexpr s32 ORBIS_KERNEL_SEM_VALUE_MAX = 0x7FFFFFFF;

struct PthreadSem {
    explicit PthreadSem(s32 value_)
        :
#ifndef __ANDROID__
          semaphore{value_},
#endif
          value{value_} {
    }

#ifdef __ANDROID__
    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    u32 waiters = 0;
#else
    CountingSemaphore semaphore;
#endif
    std::atomic<s32> value;
};

#ifdef __ANDROID__
static bool PostPthreadSemToken(PthreadSem* native, s32* before_out = nullptr) {
    bool wake_waiter = false;
    {
        std::lock_guard lock{native->wait_mutex};
        const s32 before = native->value.load(std::memory_order_relaxed);
        if (before_out != nullptr) {
            *before_out = before;
        }
        if (before == ORBIS_KERNEL_SEM_VALUE_MAX) {
            return false;
        }
        native->value.store(before + 1, std::memory_order_release);
        wake_waiter = native->waiters != 0;
    }
    if (wake_waiter) {
        native->wait_cv.notify_one();
    }
    return true;
}

static u32 ExecutorEopTraceThreadKind() {
    if (!g_curthread) {
        return 0;
    }
    if (g_curthread->name == "Game:Main") {
        return 1;
    }
    if (g_curthread->name.find("UnityGfxDeviceWorker") != std::string::npos) {
        return 2;
    }
    return 0;
}
#endif

static std::mutex pthread_sem_pool_mutex;
static std::vector<std::unique_ptr<PthreadSem>> pthread_sem_pool;
static std::array<std::mutex, 64> pthread_sem_slot_mutexes;

static std::mutex& PthreadSemSlotMutex(PthreadSem** sem) {
    const auto address = reinterpret_cast<std::uintptr_t>(sem);
    return pthread_sem_slot_mutexes[(address >> 2) % pthread_sem_slot_mutexes.size()];
}

static PthreadSem* LoadPthreadSem(PthreadSem** sem) {
    if (sem == nullptr) {
        return nullptr;
    }
    std::scoped_lock lock{PthreadSemSlotMutex(sem)};
    PthreadSem* native;
    std::memcpy(&native, sem, sizeof(native));
    return native;
}

static void StorePthreadSem(PthreadSem** sem, PthreadSem* native) {
    std::scoped_lock lock{PthreadSemSlotMutex(sem)};
    std::memcpy(sem, &native, sizeof(native));
}

static PthreadSem* DetachPthreadSem(PthreadSem** sem) {
    if (sem == nullptr) {
        return nullptr;
    }
    std::scoped_lock lock{PthreadSemSlotMutex(sem)};
    PthreadSem* native;
    std::memcpy(&native, sem, sizeof(native));
    PthreadSem* const empty = nullptr;
    std::memcpy(sem, &empty, sizeof(empty));
    return native;
}

class OrbisSem;

#ifndef __ANDROID__
static bool PcOracleMonoSyncEnabled() {
    const char* value = std::getenv("EXECUTOR_PC_ORACLE_MONO_SYNC");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static const char* PcOracleCurrentThreadName() {
    return g_curthread && !g_curthread->name.empty() ? g_curthread->name.c_str() : "<none>";
}

static std::uint64_t PcOracleReturnOff(const void* guest_return) {
    const auto return_va = reinterpret_cast<std::uintptr_t>(guest_return);
    return return_va >= 0x800000000ULL ? return_va - 0x800000000ULL : 0ULL;
}

static bool PcOracleTryReadQword(const void* base, std::size_t index, std::uint64_t& out) {
    out = 0;
    auto* memory = Core::Memory::Instance();
    const auto addr = reinterpret_cast<VAddr>(base) + index * sizeof(out);
    if (base == nullptr || memory == nullptr || !memory->IsValidMapping(addr, sizeof(out))) {
        return false;
    }
    memory->CopySparseMemory(addr, reinterpret_cast<u8*>(&out), sizeof(out));
    return true;
}

static std::string PcOracleBytesHex(const void* base, std::size_t size) {
    auto* memory = Core::Memory::Instance();
    const auto addr = reinterpret_cast<VAddr>(base);
    if (base == nullptr || memory == nullptr || !memory->IsValidMapping(addr, size)) {
        return "<invalid>";
    }
    std::vector<u8> bytes(size);
    memory->CopySparseMemory(addr, bytes.data(), size);
    static constexpr char Hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(size * 3);
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        const auto byte = bytes[i];
        out.push_back(Hex[(byte >> 4) & 0xf]);
        out.push_back(Hex[byte & 0xf]);
    }
    return out;
}

static void PcOraclePosixSemLog(const char* op, PthreadSem** slot, PthreadSem* sem, s32 before,
                                s32 after, s32 ret, const void* guest_return) {
    static std::atomic_int budget{8192};
    if (!PcOracleMonoSyncEnabled() || budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_PC_PSEM] op=%s threadName=%s slot=%p native=%p before=%d after=%d "
                 "ret=%d retaddr=%p returnOff=0x%llx\n",
                 op, PcOracleCurrentThreadName(), slot, sem, before, after, ret, guest_return,
                 static_cast<unsigned long long>(PcOracleReturnOff(guest_return)));
    std::fflush(stderr);
}

static void PcOraclePosixSemInvalidLog(const char* op, PthreadSem** slot,
                                       const void* guest_return) {
    static std::atomic_int budget{256};
    if (!PcOracleMonoSyncEnabled() || budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    std::uint64_t q0 = 0;
    std::uint64_t q1 = 0;
    const bool q0_ok = PcOracleTryReadQword(slot, 0, q0);
    const bool q1_ok = PcOracleTryReadQword(slot, 1, q1);
    const auto address = reinterpret_cast<std::uintptr_t>(slot);
    const auto window = address >= 0x20 ? reinterpret_cast<const void*>(address - 0x20) : slot;
    std::fprintf(stderr,
                 "[EXECUTOR_PC_PSEM_INVALID] op=%s threadName=%s slot=%p q0=0x%llx q0Ok=%d "
                 "q1=0x%llx q1Ok=%d retaddr=%p returnOff=0x%llx window=%p bytes=%s\n",
                 op, PcOracleCurrentThreadName(), slot, static_cast<unsigned long long>(q0),
                 q0_ok ? 1 : 0, static_cast<unsigned long long>(q1), q1_ok ? 1 : 0,
                 guest_return, static_cast<unsigned long long>(PcOracleReturnOff(guest_return)),
                 window, PcOracleBytesHex(window, 0x80).c_str());
    std::fflush(stderr);
}
#endif

#ifdef __ANDROID__
class OrbisSem;

extern "C" void executor_live_mono_run_pending_signal_safe_point() __attribute__((weak));
extern "C" bool executor_lsx4_android_jit_active() __attribute__((weak));

static bool UseJitNativeSemaphorePark() {
    return executor_lsx4_android_jit_active != nullptr &&
           executor_lsx4_android_jit_active();
}

static void RunMonoPendingSignalSafePointFromWait() {
    if (executor_live_mono_run_pending_signal_safe_point != nullptr) {
        executor_live_mono_run_pending_signal_safe_point();
    }
}

static bool ExecutorEnvFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool ShouldBypassLiveMonoSemWait(PthreadSem** sem) {
    if (!ExecutorEnvFlag("EXECUTOR_LIVE_MONO_POSIX_SEM_WAIT_BYPASS") || sem == nullptr) {
        return false;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(sem);
    const bool module_data = address >= 0x800000000ULL && address < 0x810000000ULL;
    if (!module_data) {
        return false;
    }
    const std::string thread_name = g_curthread ? g_curthread->name : "";
    return thread_name.find("mono") != std::string::npos ||
           thread_name.find("Mono") != std::string::npos ||
           thread_name.find("Finalizer") != std::string::npos ||
           thread_name.find("Thread Pool") != std::string::npos ||
           thread_name.find("Game:Main") != std::string::npos;
}

static bool IsLiveRenderThreadName(const std::string& name) {
    return name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("Game:Main") != std::string::npos ||
           name.find("UnityPreload") != std::string::npos ||
           name.find("Submit Done Thread") != std::string::npos;
}

static bool IsLiveInterestingThreadName(const std::string& name) {
    if (ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY")) {
        return IsLiveRenderThreadName(name);
    }
    if (ExecutorEnvFlag("EXECUTOR_LIGHT_ORACLE") &&
        (name.find("FMOD") != std::string::npos ||
         name.find("AudioOut") != std::string::npos ||
         name.find("audio") != std::string::npos ||
         name.find("Audio") != std::string::npos)) {
        return false;
    }
    return IsLiveRenderThreadName(name) ||
           name.find("Gamedata_Worker") != std::string::npos ||
           name.find("Task_Bagger") != std::string::npos ||
           name.find("FMOD") != std::string::npos ||
           name.find("mono") != std::string::npos || name.find("Mono") != std::string::npos ||
           name.find("Finalizer") != std::string::npos ||
           name.find("Thread Pool") != std::string::npos;
}

static bool IsLiveKnownNoisyThreadName(const std::string& name) {
    return name.find("FMOD") != std::string::npos ||
           name.find("AudioOut") != std::string::npos ||
           name.find("audio") != std::string::npos ||
           name.find("Audio") != std::string::npos;
}

static bool ShouldTraceLiveSync() {
    static const bool enabled = ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_WIDE") ||
                                ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_SYNC") ||
                                ExecutorEnvFlag("EXECUTOR_LIGHT_ORACLE");
    return enabled;
}

static bool ShouldRecordKernelSemFrontier() {
    static const bool enabled = ShouldTraceLiveSync() ||
                                ExecutorEnvFlag("EXECUTOR_SYNC_FRONTIER") ||
                                ExecutorEnvFlag("EXECUTOR_LIVE_HLE_FLIGHT");
    return enabled;
}

static bool ShouldRecordPosixSemFrontier() {
    static const bool enabled = ShouldTraceLiveSync() ||
                                ExecutorEnvFlag("EXECUTOR_SYNC_FRONTIER") ||
                                ExecutorEnvFlag("EXECUTOR_LIVE_HLE_FLIGHT");
    return enabled;
}

static bool ShouldTraceLiveDirectPosixSem(PthreadSem** sem) {
    if (!ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_POSIX_DIRECT_SEM") || sem == nullptr) {
        return false;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(sem);
    return address >= 0x200000000ULL && address < 0x400000000ULL;
}

static bool IsCurrentLiveInterestingThread() {
    return ShouldTraceLiveSync() && g_curthread && IsLiveInterestingThreadName(g_curthread->name);
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

static u64 GuestReadQword(u64 address, bool* ok = nullptr) {
    if (ok != nullptr) {
        *ok = false;
    }
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || address == 0 || !memory->IsValidMapping(address, sizeof(u64))) {
        return 0;
    }
    u64 value = 0;
    memory->CopySparseMemory(address, reinterpret_cast<u8*>(&value), sizeof(value));
    if (ok != nullptr) {
        *ok = true;
    }
    return value;
}

static u32 GuestReadDword(u64 address, bool* ok = nullptr) {
    if (ok != nullptr) {
        *ok = false;
    }
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || address == 0 || !memory->IsValidMapping(address, sizeof(u32))) {
        return 0;
    }
    u32 value = 0;
    memory->CopySparseMemory(address, reinterpret_cast<u8*>(&value), sizeof(value));
    if (ok != nullptr) {
        *ok = true;
    }
    return value;
}

static u8 GuestReadByte(u64 address, bool* ok = nullptr) {
    if (ok != nullptr) {
        *ok = false;
    }
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || address == 0 || !memory->IsValidMapping(address, sizeof(u8))) {
        return 0;
    }
    u8 value = 0;
    memory->CopySparseMemory(address, &value, sizeof(value));
    if (ok != nullptr) {
        *ok = true;
    }
    return value;
}

static long long SignedDelta(std::uintptr_t lhs, std::uintptr_t rhs) {
    if (lhs >= rhs) {
        return static_cast<long long>(lhs - rhs);
    }
    const auto diff = rhs - lhs;
    return -static_cast<long long>(diff);
}

static bool IsWatchedPosixSem(PthreadSem** sem, bool mark) {
    static std::atomic<std::uintptr_t> watched[64]{};
    const auto value = reinterpret_cast<std::uintptr_t>(sem);
    if (value == 0) {
        return false;
    }
    for (auto& slot : watched) {
        if (slot.load(std::memory_order_relaxed) == value) {
            return true;
        }
    }
    if (!mark) {
        return false;
    }
    for (auto& slot : watched) {
        std::uintptr_t expected = 0;
        if (slot.compare_exchange_strong(expected, value, std::memory_order_relaxed) ||
            expected == value) {
            return true;
        }
    }
    return false;
}

static bool IsWatchedKernelSem(Common::SlotId sem, bool mark) {
    static std::atomic<u32> watched[128]{};
    const auto value = sem.index + 1;
    if (value == 0) {
        return false;
    }
    for (auto& slot : watched) {
        if (slot.load(std::memory_order_relaxed) == value) {
            return true;
        }
    }
    if (!mark) {
        return false;
    }
    for (auto& slot : watched) {
        u32 expected = 0;
        if (slot.compare_exchange_strong(expected, value, std::memory_order_relaxed) ||
            expected == value) {
            return true;
        }
    }
    return false;
}

struct LivePosixSemRecord {
    std::uintptr_t sem{};
    std::uintptr_t native{};
    s32 init_value{};
    s32 last_before{};
    s32 last_after{};
    s32 last_ret{};
    std::uint32_t init_count{};
    std::uint32_t wait_enter_count{};
    std::uint32_t wait_return_count{};
    std::uint32_t post_count{};
    std::uint32_t trywait_count{};
    std::uint32_t timedwait_count{};
    std::uint32_t invalid_count{};
    long init_tid{};
    long last_tid{};
    std::string init_thread;
    std::string last_thread;
    std::string last_op;
    std::uint64_t init_guest_off{};
    std::uint64_t last_wait_guest_off{};
    std::uint64_t last_post_guest_off{};
    std::string init_guest_module;
    std::string last_wait_guest_module;
    std::string last_post_guest_module;
};

struct LiveKernelSemRecord {
    u32 slot{};
    std::uintptr_t native{};
    s32 init_value{};
    s32 max_value{};
    s32 last_before{};
    s32 last_after{};
    s32 last_count{};
    s32 last_ret{};
    std::uint32_t create_count{};
    std::uint32_t wait_enter_count{};
    std::uint32_t wait_return_count{};
    std::uint32_t signal_count{};
    std::uint32_t poll_count{};
    std::uint32_t cancel_count{};
    std::uint32_t invalid_count{};
    long init_tid{};
    long last_tid{};
    std::string init_thread;
    std::string last_thread;
    std::string last_op;
    std::string name;
};

static std::mutex& LivePosixSemRecordsMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<std::uintptr_t, LivePosixSemRecord>& LivePosixSemRecords() {
    static std::unordered_map<std::uintptr_t, LivePosixSemRecord> records;
    return records;
}

static std::string ExecutorSemModuleName(const char* module_name) {
    if (module_name == nullptr || module_name[0] == '\0') {
        return "<unknown>";
    }
    const std::string value{module_name};
    const auto slash = value.find_last_of("/\\");
    return slash == std::string::npos ? value : value.substr(slash + 1);
}

extern "C" void executor_live_record_posix_sem_hle_site(const char* op, void* sem_addr,
                                                        std::uint64_t guest_return,
                                                        std::uint64_t module_offset,
                                                        const char* module_name) {
    if (!ShouldRecordPosixSemFrontier() || sem_addr == nullptr || op == nullptr) {
        return;
    }
    const auto key = reinterpret_cast<std::uintptr_t>(sem_addr);
    const std::string thread_name = g_curthread ? g_curthread->name : "<no-gcurthread>";
    const long tid = static_cast<long>(::syscall(SYS_gettid));
    const auto module = ExecutorSemModuleName(module_name);
    {
        std::lock_guard lock{LivePosixSemRecordsMutex()};
        auto& record = LivePosixSemRecords()[key];
        record.sem = key;
        const std::string_view opname{op};
        if (opname == "init") {
            record.init_guest_off = module_offset;
            record.init_guest_module = module;
        } else if (opname.find("wait") != std::string_view::npos) {
            record.last_wait_guest_off = module_offset;
            record.last_wait_guest_module = module;
        } else if (opname == "post") {
            record.last_post_guest_off = module_offset;
            record.last_post_guest_module = module;
        }
    }
    static std::atomic_int budget{2048};
    if (ShouldTraceLiveSync() && budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_SEM_SITE] op=%s sem=%p ret=%p off=0x%llx module=%s "
            "thread=%s tid=%ld",
            op, sem_addr, reinterpret_cast<void*>(guest_return),
            static_cast<unsigned long long>(module_offset), module.c_str(), thread_name.c_str(),
            tid);
    }
}

extern "C" void executor_live_dump_posix_sem_records(const char* reason) {
    if (!ShouldRecordPosixSemFrontier()) {
        return;
    }
    struct Snapshot {
        LivePosixSemRecord record;
        s32 pending{};
    };
    std::vector<Snapshot> snapshots;
    snapshots.reserve(64);
    {
        std::lock_guard lock{LivePosixSemRecordsMutex()};
        for (const auto& [_, record] : LivePosixSemRecords()) {
            const s32 pending = static_cast<s32>(record.wait_enter_count) -
                                static_cast<s32>(record.wait_return_count);
            const bool render_sem =
                record.init_thread.find("Unity") != std::string::npos ||
                record.last_thread.find("Unity") != std::string::npos ||
                record.last_thread.find("Game:Main") != std::string::npos ||
                record.last_thread.find("Submit Done Thread") != std::string::npos ||
                pending > 0;
            if (render_sem) {
                snapshots.push_back(Snapshot{record, pending});
            }
        }
    }
    std::sort(snapshots.begin(), snapshots.end(), [](const Snapshot& lhs, const Snapshot& rhs) {
        if (lhs.pending != rhs.pending) {
            return lhs.pending > rhs.pending;
        }
        if (lhs.record.wait_enter_count != rhs.record.wait_enter_count) {
            return lhs.record.wait_enter_count > rhs.record.wait_enter_count;
        }
        return lhs.record.sem < rhs.record.sem;
    });
    static std::atomic_int budget{256};
    std::uint32_t emitted = 0;
    for (const auto& snapshot : snapshots) {
        if (emitted++ >= 24 || budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
            break;
        }
        const auto& r = snapshot.record;
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_LIVE_SEM_PENDING_SUMMARY] reason=%s sem=%p native=%p pending=%d "
            "init=%u wait_enter=%u wait_return=%u post=%u value_before=%d value_after=%d "
            "last_ret=%d initThread=%s lastThread=%s lastOp=%s "
            "initSite=%s+0x%llx waitSite=%s+0x%llx postSite=%s+0x%llx",
            reason && reason[0] ? reason : "<unknown>", reinterpret_cast<void*>(r.sem),
            reinterpret_cast<void*>(r.native), snapshot.pending, r.init_count,
            r.wait_enter_count, r.wait_return_count, r.post_count, r.last_before, r.last_after,
            r.last_ret, r.init_thread.c_str(), r.last_thread.c_str(), r.last_op.c_str(),
            r.init_guest_module.empty() ? "<unknown>" : r.init_guest_module.c_str(),
            static_cast<unsigned long long>(r.init_guest_off),
            r.last_wait_guest_module.empty() ? "<unknown>" : r.last_wait_guest_module.c_str(),
            static_cast<unsigned long long>(r.last_wait_guest_off),
            r.last_post_guest_module.empty() ? "<unknown>" : r.last_post_guest_module.c_str(),
            static_cast<unsigned long long>(r.last_post_guest_off));
    }
}

extern "C" int executor_live_signal_pending_mono_posix_sems(int max_posts) {
    if (max_posts <= 0) {
        return 0;
    }
    struct Candidate {
        std::uintptr_t sem{};
        std::uintptr_t native{};
        std::uint32_t wait_enter{};
        std::uint32_t wait_return{};
        std::uint32_t post{};
        std::string init_module;
        std::string wait_module;
        std::string last_thread;
    };
    std::vector<Candidate> candidates;
    {
        std::lock_guard lock{LivePosixSemRecordsMutex()};
        for (const auto& [_, record] : LivePosixSemRecords()) {
            const auto pending = static_cast<s32>(record.wait_enter_count) -
                                 static_cast<s32>(record.wait_return_count);
            const bool mono_module =
                record.init_guest_module.find("mono-ps4") != std::string::npos ||
                record.last_wait_guest_module.find("mono-ps4") != std::string::npos;
            const bool mono_waiter = record.last_thread.find("mono thread") != std::string::npos ||
                                     record.last_thread.find("Mono") != std::string::npos;
            if (pending <= 0 || record.native == 0 || !mono_module || !mono_waiter) {
                continue;
            }
            candidates.push_back(Candidate{record.sem, record.native, record.wait_enter_count,
                                           record.wait_return_count, record.post_count,
                                           record.init_guest_module, record.last_wait_guest_module,
                                           record.last_thread});
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs,
                                                       const Candidate& rhs) {
        const auto lhs_pending = static_cast<s32>(lhs.wait_enter) -
                                 static_cast<s32>(lhs.wait_return);
        const auto rhs_pending = static_cast<s32>(rhs.wait_enter) -
                                 static_cast<s32>(rhs.wait_return);
        if (lhs_pending != rhs_pending) {
            return lhs_pending > rhs_pending;
        }
        return lhs.sem < rhs.sem;
    });

    int signaled = 0;
    for (const auto& candidate : candidates) {
        if (signaled >= max_posts) {
            break;
        }
        auto* native = reinterpret_cast<PthreadSem*>(candidate.native);
        if (native == nullptr) {
            continue;
        }
        s32 before = 0;
        if (!PostPthreadSemToken(native, &before)) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Native",
                "[EXECUTOR_MONO_POSIX_SEM_SIGNAL] sem=%p native=%p result=overflow before=%d "
                "initSite=%s waitSite=%s lastThread=%s",
                reinterpret_cast<void*>(candidate.sem), native, before,
                candidate.init_module.c_str(), candidate.wait_module.c_str(),
                candidate.last_thread.c_str());
            continue;
        }
        const s32 after = native->value.load(std::memory_order_relaxed);
        {
            std::lock_guard records_lock{LivePosixSemRecordsMutex()};
            auto it = LivePosixSemRecords().find(candidate.sem);
            if (it != LivePosixSemRecords().end()) {
                auto& record = it->second;
                record.post_count++;
                record.last_before = before;
                record.last_after = after;
                record.last_ret = 0;
                record.last_tid = static_cast<long>(::syscall(SYS_gettid));
                record.last_thread = g_curthread ? g_curthread->name : "<no-gcurthread>";
                record.last_op = "signal_mono_posix";
                record.last_post_guest_module = "<executor>";
                record.last_post_guest_off = 0;
            }
        }
        signaled++;
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_MONO_POSIX_SEM_SIGNAL] sem=%p native=%p before=%d after=%d "
            "wait_enter=%u wait_return=%u post_before=%u initSite=%s waitSite=%s "
            "lastThread=%s reason=coop-sigusr1",
            reinterpret_cast<void*>(candidate.sem), native, before, after, candidate.wait_enter,
            candidate.wait_return, candidate.post, candidate.init_module.c_str(),
            candidate.wait_module.c_str(), candidate.last_thread.c_str());
    }
    if (signaled == 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_MONO_POSIX_SEM_SIGNAL] result=none maxPosts=%d",
                            max_posts);
    }
    return signaled;
}

static std::mutex& LiveKernelSemRecordsMutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<u32, LiveKernelSemRecord>& LiveKernelSemRecords() {
    static std::unordered_map<u32, LiveKernelSemRecord> records;
    return records;
}

static void UpdateLivePosixSemRecord(const char* op, PthreadSem** sem, PthreadSem* native,
                                     s32 before, s32 after, s32 ret) {
    if (!ShouldRecordPosixSemFrontier() || sem == nullptr) {
        return;
    }
    const auto key = reinterpret_cast<std::uintptr_t>(sem);
    const auto native_value = reinterpret_cast<std::uintptr_t>(native);
    const long tid = static_cast<long>(::syscall(SYS_gettid));
    const std::string thread_name = g_curthread ? g_curthread->name : "<no-gcurthread>";
    std::lock_guard lock{LivePosixSemRecordsMutex()};
    auto& record = LivePosixSemRecords()[key];
    record.sem = key;
    record.native = native_value;
    record.last_before = before;
    record.last_after = after;
    record.last_ret = ret;
    record.last_tid = tid;
    record.last_thread = thread_name;
    record.last_op = op ? op : "<null>";

    const std::string_view opname{op ? op : ""};
    if (opname == "init") {
        record.init_value = before;
        record.init_count++;
        record.init_tid = tid;
        record.init_thread = thread_name;
    } else if (opname.find("wait_enter") != std::string_view::npos) {
        record.wait_enter_count++;
    } else if (opname.find("wait_return") != std::string_view::npos) {
        record.wait_return_count++;
    } else if (opname.find("post_return") != std::string_view::npos) {
        record.post_count++;
    } else if (opname.find("trywait") != std::string_view::npos) {
        record.trywait_count++;
    } else if (opname.find("timedwait") != std::string_view::npos) {
        record.timedwait_count++;
    } else if (opname.find("invalid") != std::string_view::npos ||
               opname.find("overflow") != std::string_view::npos) {
        record.invalid_count++;
    }
}

#ifdef __ANDROID__
bool ExecutorLivePosixSemHasWaitEnter(void* sem_slot) {
    if (!ShouldTraceLiveSync() || sem_slot == nullptr) {
        return false;
    }
    const auto key = reinterpret_cast<std::uintptr_t>(sem_slot);
    std::lock_guard lock{LivePosixSemRecordsMutex()};
    const auto it = LivePosixSemRecords().find(key);
    return it != LivePosixSemRecords().end() && it->second.wait_enter_count != 0;
}
#endif

static void DumpLivePosixSemNeighborhood(PthreadSem** stuck_sem) {
    if (!ShouldTraceLiveSync() || stuck_sem == nullptr) {
        return;
    }
    const auto stuck = reinterpret_cast<std::uintptr_t>(stuck_sem);
#ifdef __ANDROID__
    if (executor_jit_dump_thread_states != nullptr) {
        executor_jit_dump_thread_states("posix-sem-stuck");
    }
#endif
    struct Snapshot {
        LivePosixSemRecord record;
        std::uintptr_t distance{};
    };
    std::vector<Snapshot> snapshots;
    snapshots.reserve(64);
    {
        std::lock_guard lock{LivePosixSemRecordsMutex()};
        const auto& records = LivePosixSemRecords();
        const auto stuck_group = stuck & 0xffffffffff000000ULL;
        for (const auto& [key, record] : records) {
            const auto distance = key > stuck ? key - stuck : stuck - key;
            const bool same_group = (key & 0xffffffffff000000ULL) == stuck_group;
            if (key == stuck || same_group || distance <= 0x100000ULL) {
                snapshots.push_back(Snapshot{record, distance});
            }
        }
    }
    std::sort(snapshots.begin(), snapshots.end(), [](const Snapshot& lhs, const Snapshot& rhs) {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.record.sem < rhs.record.sem;
    });
    static std::atomic_int budget{512};
    std::uint32_t emitted = 0;
    for (const auto& snapshot : snapshots) {
        if (emitted++ >= 48 || budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
            break;
        }
        const auto& r = snapshot.record;
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_LIVE_SEM_LEDGER] stuck=%p sem=%p dist=0x%llx native=%p init=%u "
            "wait_enter=%u wait_return=%u post=%u trywait=%u timedwait=%u invalid=%u "
            "init_value=%d last_before=%d last_after=%d last_ret=%d init_thread=%s "
            "init_tid=%ld last_op=%s last_thread=%s last_tid=%ld "
            "initSite=%s+0x%llx waitSite=%s+0x%llx postSite=%s+0x%llx",
            reinterpret_cast<void*>(stuck), reinterpret_cast<void*>(r.sem),
            static_cast<unsigned long long>(snapshot.distance), reinterpret_cast<void*>(r.native),
            r.init_count, r.wait_enter_count, r.wait_return_count, r.post_count,
            r.trywait_count, r.timedwait_count, r.invalid_count, r.init_value, r.last_before,
            r.last_after, r.last_ret, r.init_thread.c_str(), r.init_tid, r.last_op.c_str(),
            r.last_thread.c_str(), r.last_tid,
            r.init_guest_module.empty() ? "<unknown>" : r.init_guest_module.c_str(),
            static_cast<unsigned long long>(r.init_guest_off),
            r.last_wait_guest_module.empty() ? "<unknown>" : r.last_wait_guest_module.c_str(),
            static_cast<unsigned long long>(r.last_wait_guest_off),
            r.last_post_guest_module.empty() ? "<unknown>" : r.last_post_guest_module.c_str(),
            static_cast<unsigned long long>(r.last_post_guest_off));
    }
}

static void UpdateLiveKernelSemRecord(const char* op, Common::SlotId sem, const OrbisSem* native,
                                      s32 before, s32 after, s32 count, s32 ret,
                                      const char* name) {
    if (!ShouldRecordKernelSemFrontier()) {
        return;
    }
    const auto key = sem.index + 1;
    if (key == 0) {
        return;
    }
    const long tid = static_cast<long>(::syscall(SYS_gettid));
    const std::string thread_name = g_curthread ? g_curthread->name : "<no-gcurthread>";
    std::lock_guard lock{LiveKernelSemRecordsMutex()};
    auto& record = LiveKernelSemRecords()[key];
    record.slot = sem.index;
    record.native = reinterpret_cast<std::uintptr_t>(native);
    record.last_before = before;
    record.last_after = after;
    record.last_count = count;
    record.last_ret = ret;
    record.last_tid = tid;
    record.last_thread = thread_name;
    record.last_op = op ? op : "<null>";
    if (name && name[0] != '\0') {
        record.name = name;
    }

    const std::string_view opname{op ? op : ""};
    if (opname == "create") {
        record.init_value = before;
        record.max_value = count;
        record.create_count++;
        record.init_tid = tid;
        record.init_thread = thread_name;
    } else if (opname.find("wait_enter") != std::string_view::npos) {
        record.wait_enter_count++;
    } else if (opname.find("wait_return") != std::string_view::npos) {
        record.wait_return_count++;
    } else if (opname.find("signal") != std::string_view::npos) {
        record.signal_count++;
    } else if (opname.find("poll") != std::string_view::npos) {
        record.poll_count++;
    } else if (opname.find("cancel") != std::string_view::npos) {
        record.cancel_count++;
    } else if (opname.find("missing") != std::string_view::npos ||
               opname.find("invalid") != std::string_view::npos) {
        record.invalid_count++;
    }
}

static void DumpLiveKernelSemLedger(Common::SlotId stuck_sem) {
    if (!ShouldRecordKernelSemFrontier()) {
        return;
    }
    const auto stuck_key = stuck_sem.index + 1;
    struct Snapshot {
        LiveKernelSemRecord record;
        u32 distance{};
    };
    std::vector<Snapshot> snapshots;
    snapshots.reserve(64);
    {
        std::lock_guard lock{LiveKernelSemRecordsMutex()};
        for (const auto& [key, record] : LiveKernelSemRecords()) {
            const auto distance = key > stuck_key ? key - stuck_key : stuck_key - key;
            if (key == stuck_key || distance <= 32) {
                snapshots.push_back(Snapshot{record, distance});
            }
        }
    }
    std::sort(snapshots.begin(), snapshots.end(), [](const Snapshot& lhs, const Snapshot& rhs) {
        if (lhs.distance != rhs.distance) {
            return lhs.distance < rhs.distance;
        }
        return lhs.record.slot < rhs.record.slot;
    });
    static std::atomic_int budget{512};
    std::uint32_t emitted = 0;
    for (const auto& snapshot : snapshots) {
        if (emitted++ >= 48 || budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
            break;
        }
        const auto& r = snapshot.record;
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_LIVE_KSEMA_LEDGER] stuck_slot=%u slot=%u dist=%u native=%p "
            "create=%u wait_enter=%u wait_return=%u signal=%u poll=%u cancel=%u invalid=%u "
            "init=%d max=%d last_before=%d last_after=%d last_count=%d last_ret=0x%x "
            "init_thread=%s init_tid=%ld last_op=%s last_thread=%s last_tid=%ld name=%s",
            stuck_sem.index, r.slot, snapshot.distance, reinterpret_cast<void*>(r.native),
            r.create_count, r.wait_enter_count, r.wait_return_count, r.signal_count, r.poll_count,
            r.cancel_count, r.invalid_count, r.init_value, r.max_value, r.last_before,
            r.last_after, r.last_count, static_cast<u32>(r.last_ret), r.init_thread.c_str(),
            r.init_tid, r.last_op.c_str(), r.last_thread.c_str(), r.last_tid,
            r.name.empty() ? "<none>" : r.name.c_str());
    }
}

extern "C" void executor_live_dump_kernel_sem_records(const char* reason) {
    if (!ShouldRecordKernelSemFrontier()) {
        return;
    }
    std::vector<LiveKernelSemRecord> snapshot;
    {
        std::lock_guard lock{LiveKernelSemRecordsMutex()};
        snapshot.reserve(LiveKernelSemRecords().size());
        for (const auto& [key, record] : LiveKernelSemRecords()) {
            (void)key;
            snapshot.push_back(record);
        }
    }
    std::sort(snapshot.begin(), snapshot.end(), [](const auto& lhs, const auto& rhs) {
        const auto lhs_outstanding =
            static_cast<std::int64_t>(lhs.wait_enter_count) - lhs.wait_return_count;
        const auto rhs_outstanding =
            static_cast<std::int64_t>(rhs.wait_enter_count) - rhs.wait_return_count;
        if (lhs_outstanding != rhs_outstanding) {
            return lhs_outstanding > rhs_outstanding;
        }
        return lhs.slot < rhs.slot;
    });

    std::size_t active = 0;
    for (const auto& record : snapshot) {
        const auto outstanding =
            static_cast<std::int64_t>(record.wait_enter_count) - record.wait_return_count;
        if (outstanding <= 0) {
            continue;
        }
        ++active;
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_KERNEL_SEM_FRONTIER] reason=%s slot=%u native=%p name=%s "
            "outstanding=%lld create=%u wait_enter=%u wait_return=%u signal=%u poll=%u "
            "cancel=%u invalid=%u init=%d max=%d last_before=%d last_after=%d "
            "last_count=%d last_ret=0x%x init_thread=%s last_op=%s last_thread=%s",
            reason && reason[0] ? reason : "<none>", record.slot,
            reinterpret_cast<void*>(record.native),
            record.name.empty() ? "<none>" : record.name.c_str(),
            static_cast<long long>(outstanding), record.create_count, record.wait_enter_count,
            record.wait_return_count, record.signal_count, record.poll_count,
            record.cancel_count, record.invalid_count, record.init_value, record.max_value,
            record.last_before, record.last_after, record.last_count,
            static_cast<u32>(record.last_ret), record.init_thread.c_str(),
            record.last_op.c_str(), record.last_thread.c_str());
    }
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_KERNEL_SEM_FRONTIER_SUMMARY] reason=%s records=%zu active=%zu",
                        reason && reason[0] ? reason : "<none>", snapshot.size(), active);
}

static void TraceLiveUnityGfxSemContext(const char* op, PthreadSem** sem) {
    if (!ShouldTraceLiveSync() || sem == nullptr) {
        return;
    }
    if (ExecutorLightOracleMode() && !ExecutorEnvFlag("EXECUTOR_VERBOSE_UNITY_GFX_SEM_CTX")) {
        return;
    }
    const auto worker = g_executor_unity_gfx_worker.load(std::memory_order_acquire);
    if (worker == 0) {
        return;
    }

    const std::string thread_name = g_curthread ? g_curthread->name : "<no-gcurthread>";
    const bool interesting_thread =
        thread_name.find("UnityGfxDeviceWorker") != std::string::npos ||
        thread_name.find("Game:Main") != std::string::npos ||
        thread_name.find("Submit Done Thread") != std::string::npos;
    if (!interesting_thread) {
        return;
    }

    const auto sem_addr = reinterpret_cast<std::uintptr_t>(sem);
    bool stream_ok = false;
    bool device_ok = false;
    bool stop_ok = false;
    const u64 stream = GuestReadQword(worker + 0x18, &stream_ok);
    const u64 device = GuestReadQword(worker + 0x208, &device_ok);
    const u8 stop = GuestReadByte(worker + 0x211, &stop_ok);

    const auto worker_sem_d0 = worker + 0xd0;
    const auto worker_sem_e4 = worker + 0xe4;
    const auto stream_sem_d0 = stream != 0 ? stream + 0xd0 : 0;
    const auto stream_sem_e4 = stream != 0 ? stream + 0xe4 : 0;
    const bool sem_related =
        sem_addr == worker_sem_d0 || sem_addr == worker_sem_e4 || sem_addr == stream_sem_d0 ||
        sem_addr == stream_sem_e4 ||
        (stream != 0 && sem_addr > stream && sem_addr < stream + 0x180) ||
        (sem_addr > worker && sem_addr < worker + 0x240);
    if (!sem_related && thread_name.find("UnityGfxDeviceWorker") == std::string::npos) {
        return;
    }

    bool stream_buf_ok = false;
    bool stream_pos_ok = false;
    bool stream_cap_ok = false;
    const u64 stream_buf = stream != 0 ? GuestReadQword(stream + 0x8, &stream_buf_ok) : 0;
    const u32 stream_pos = stream != 0 ? GuestReadDword(stream + 0x100, &stream_pos_ok) : 0;
    const u32 stream_cap = stream != 0 ? GuestReadDword(stream + 0x104, &stream_cap_ok) : 0;

    static std::atomic_int budget{4096};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_UNITY_GFX_SEM_CTX] op=%s thread=%s tid=%ld sem=%p worker=%p "
        "stop=%u stopOk=%d stream=%p streamOk=%d streamBuf=%p streamBufOk=%d "
        "streamPos=0x%x streamPosOk=%d streamCap=0x%x streamCapOk=%d device=%p deviceOk=%d "
        "dWorkerD0=%lld dWorkerE4=%lld dStreamD0=%lld dStreamE4=%lld "
        "workerBytes=%s streamBytes=%s",
        op ? op : "<unknown>", thread_name.c_str(), static_cast<long>(::syscall(SYS_gettid)),
        sem, reinterpret_cast<void*>(worker), static_cast<unsigned>(stop), stop_ok ? 1 : 0,
        reinterpret_cast<void*>(stream), stream_ok ? 1 : 0, reinterpret_cast<void*>(stream_buf),
        stream_buf_ok ? 1 : 0, stream_pos, stream_pos_ok ? 1 : 0, stream_cap,
        stream_cap_ok ? 1 : 0, reinterpret_cast<void*>(device), device_ok ? 1 : 0,
        SignedDelta(sem_addr, worker_sem_d0), SignedDelta(sem_addr, worker_sem_e4),
        stream != 0 ? SignedDelta(sem_addr, stream_sem_d0) : 0,
        stream != 0 ? SignedDelta(sem_addr, stream_sem_e4) : 0,
        GuestBytesHex(worker, 0x240).c_str(), stream != 0 ? GuestBytesHex(stream, 0x180).c_str()
                                                          : "<none>");
}

static void TraceLivePosixSem(const char* op, PthreadSem** sem, PthreadSem* native, s32 before,
                              s32 after, s32 ret) {
    UpdateLivePosixSemRecord(op, sem, native, before, after, ret);
    TraceLiveUnityGfxSemContext(op, sem);
    const bool direct_sem = ShouldTraceLiveDirectPosixSem(sem);
    const bool interesting = IsCurrentLiveInterestingThread();
    if (ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY") && g_curthread &&
        !interesting && !direct_sem && IsLiveKnownNoisyThreadName(g_curthread->name)) {
        return;
    }
    if (interesting) {
        (void)IsWatchedPosixSem(sem, true);
    }
    if (!direct_sem && !interesting && !IsWatchedPosixSem(sem, false)) {
        return;
    }
    static std::atomic_int budget{8192};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_SEM] op=%s thread=%s tid=%ld watched=%u direct=%u sem=%p native=%p "
                        "before=%d after=%d ret=%d",
                        op, g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                        static_cast<long>(::syscall(SYS_gettid)), interesting ? 0u : 1u,
                        direct_sem ? 1u : 0u, sem, native, before, after, ret);
}

static void TraceLivePosixSemDuration(const char* op, PthreadSem** sem, PthreadSem* native,
                                      s32 ret, long long elapsed_us) {
    if (elapsed_us >= 1000) {
        TraceLiveUnityGfxSemContext(op, sem);
    }
    const bool direct_sem = ShouldTraceLiveDirectPosixSem(sem);
    if (ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY") && g_curthread &&
        !IsCurrentLiveInterestingThread() && !direct_sem &&
        IsLiveKnownNoisyThreadName(g_curthread->name)) {
        return;
    }
    if (!direct_sem && !IsCurrentLiveInterestingThread() && !IsWatchedPosixSem(sem, false)) {
        return;
    }
    static std::atomic_int budget{4096};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0 && elapsed_us < 1000) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_SEM_WAIT_TIME] op=%s thread=%s tid=%ld sem=%p native=%p "
                        "elapsed_us=%lld ret=%d",
                        op, g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                        static_cast<long>(::syscall(SYS_gettid)), sem, native, elapsed_us, ret);
}

static void TraceLiveInvalidPosixSemSlot(const char* op, PthreadSem** sem) {
    if (!ShouldTraceLiveSync()) {
        return;
    }
    static std::atomic_int budget{128};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    const auto address = reinterpret_cast<u64>(sem);
    bool q0_ok = false;
    bool q1_ok = false;
    const u64 q0 = GuestReadQword(address, &q0_ok);
    const u64 q1 = GuestReadQword(address + sizeof(u64), &q1_ok);
    const u64 window = address >= 0x20 ? address - 0x20 : address;
    const std::string bytes = GuestBytesHex(window, 0x80);
    std::uint64_t hle_ret = 0;
    std::uint64_t hle_off = 0;
    std::uint64_t hle_arg0 = 0;
    char hle_symbol[128]{};
    char hle_module[128]{};
    const int hle_site_rc = executor_live_get_current_hle_call_site
                                ? executor_live_get_current_hle_call_site(
                                      &hle_ret, &hle_off, &hle_arg0, hle_symbol,
                                      sizeof(hle_symbol), hle_module, sizeof(hle_module))
                                : -1;
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_SEM_INVALID_SLOT] op=%s thread=%s tid=%ld sem=%p "
                        "slotQ0=%p slotQ0Ok=%d slotQ1=%p slotQ1Ok=%d "
                        "hleRc=%d hleRet=%p hleOff=0x%llx hleArg0=%p hleModule=%s "
                        "hleSymbol=%s window=%p bytes=%s",
                        op ? op : "<unknown>",
                        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                        static_cast<long>(::syscall(SYS_gettid)), sem,
                        reinterpret_cast<void*>(q0), q0_ok ? 1 : 0,
                        reinterpret_cast<void*>(q1), q1_ok ? 1 : 0, hle_site_rc,
                        reinterpret_cast<void*>(hle_ret),
                        static_cast<unsigned long long>(hle_off),
                        reinterpret_cast<void*>(hle_arg0),
                        hle_module[0] ? hle_module : "<unknown>",
                        hle_symbol[0] ? hle_symbol : "<unknown>",
                        reinterpret_cast<void*>(window), bytes.c_str());
    if (executor_live_dump_current_hle_context != nullptr &&
        (!ExecutorLightOracleMode() ||
         std::getenv("EXECUTOR_TRACE_LIGHT_INVALID_SEM_HLE_DUMP") != nullptr) &&
        g_curthread && g_curthread->name.find("UnityPreload") != std::string::npos) {
        executor_live_dump_current_hle_context(op ? op : "posix_sem_invalid", address);
    }
}

static std::shared_ptr<std::atomic_bool> BeginLivePosixSemWaitWatch(PthreadSem** sem,
                                                                    PthreadSem* native,
                                                                    s32 before) {
    if (!ShouldTraceLiveSync() || !IsCurrentLiveInterestingThread()) {
        return {};
    }
    const auto done = std::make_shared<std::atomic_bool>(false);
    const std::string thread_name = g_curthread ? g_curthread->name : "<no-gcurthread>";
    const long tid = static_cast<long>(::syscall(SYS_gettid));
    std::thread([done, sem, native, before, thread_name, tid] {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (done->load(std::memory_order_acquire)) {
            return;
        }
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_SEM_STUCK] elapsed_ms=2000 thread=%s tid=%ld "
                            "sem=%p native=%p before=%d note=wait_enter_without_return",
                            thread_name.c_str(), tid, sem, native, before);
        if (ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_EMU_DUMP") &&
            executor_lsx4_android_dump_box64_emu_states) {
            executor_lsx4_android_dump_box64_emu_states("posix-sem-stuck");
        }
        DumpLivePosixSemNeighborhood(sem);
    }).detach();
    return done;
}

static void TraceLiveKernelSem(const char* op, Common::SlotId sem, const OrbisSem* native,
                               s32 before, s32 after, s32 count, const u32* timeout, s32 ret,
                               const char* name = nullptr) {
    UpdateLiveKernelSemRecord(op, sem, native, before, after, count, ret, name);
    const bool interesting = IsCurrentLiveInterestingThread();
    if (ExecutorEnvFlag("EXECUTOR_LIGHT_ORACLE") && g_curthread &&
        IsLiveKnownNoisyThreadName(g_curthread->name)) {
        return;
    }
    if (ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY") && g_curthread &&
        !interesting && IsLiveKnownNoisyThreadName(g_curthread->name)) {
        return;
    }
    if (interesting) {
        (void)IsWatchedKernelSem(sem, true);
    }
    if (!interesting && !IsWatchedKernelSem(sem, false)) {
        return;
    }
    static std::atomic_int budget{8192};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_KSEMA] op=%s thread=%s tid=%ld watched=%u slot=%u "
                        "native=%p before=%d after=%d count=%d timeout=%lld ret=0x%x name=%s",
                        op, g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                        static_cast<long>(::syscall(SYS_gettid)), interesting ? 0u : 1u,
                        sem.index, native, before, after, count,
                        timeout ? static_cast<long long>(*timeout) : -1LL, static_cast<u32>(ret),
                        name ? name : "<none>");
}

static std::shared_ptr<std::atomic_bool> BeginLiveKernelSemWaitWatch(Common::SlotId sem,
                                                                     const OrbisSem* native,
                                                                     s32 before, s32 need_count,
                                                                     const u32* timeout) {
    if (!ShouldTraceLiveSync() || !IsCurrentLiveInterestingThread()) {
        return {};
    }
    const auto done = std::make_shared<std::atomic_bool>(false);
    const std::string thread_name = g_curthread ? g_curthread->name : "<no-gcurthread>";
    const long tid = static_cast<long>(::syscall(SYS_gettid));
    const long long timeout_value = timeout ? static_cast<long long>(*timeout) : -1LL;
    std::thread([done, sem, native, before, need_count, timeout_value, thread_name, tid] {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (done->load(std::memory_order_acquire)) {
            return;
        }
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_KSEMA_STUCK] elapsed_ms=2000 thread=%s tid=%ld "
                            "slot=%u native=%p before=%d need=%d timeout=%lld "
                            "note=wait_enter_without_return",
                            thread_name.c_str(), tid, sem.index, native, before, need_count,
                            timeout_value);
        if (ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_EMU_DUMP") &&
            executor_lsx4_android_dump_box64_emu_states) {
            executor_lsx4_android_dump_box64_emu_states("kernel-sem-stuck");
        }
        DumpLiveKernelSemLedger(sem);
    }).detach();
    return done;
}

static void TraceLiveKernelSemDuration(Common::SlotId sem, const OrbisSem* native, s32 ret,
                                       long long elapsed_us, const u32* timeout) {
    if (ExecutorEnvFlag("EXECUTOR_LIGHT_ORACLE") && g_curthread &&
        IsLiveKnownNoisyThreadName(g_curthread->name)) {
        return;
    }
    const bool watched = IsWatchedKernelSem(sem, false);
    if (g_curthread && IsLiveKnownNoisyThreadName(g_curthread->name) && !watched) {
        return;
    }
    if (ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY") && g_curthread &&
        !IsCurrentLiveInterestingThread() && IsLiveKnownNoisyThreadName(g_curthread->name)) {
        return;
    }
    if (!IsCurrentLiveInterestingThread() && !watched) {
        return;
    }
    static std::atomic_int budget{4096};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0 && elapsed_us < 1000) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_KSEMA_WAIT_TIME] thread=%s tid=%ld slot=%u native=%p "
                        "elapsed_us=%lld timeout_after=%lld ret=0x%x",
                        g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                        static_cast<long>(::syscall(SYS_gettid)), sem.index, native, elapsed_us,
                        timeout ? static_cast<long long>(*timeout) : -1LL, static_cast<u32>(ret));
}
#endif

class OrbisSem {
public:
    OrbisSem(s32 init_count, s32 max_count, std::string_view name, bool is_fifo)
        : name{name}, token_count{init_count}, max_count{max_count}, init_count{init_count},
          is_fifo{is_fifo} {}
    ~OrbisSem() = default;

    s32 Wait(bool can_block, s32 need_count, u32* timeout) {
        std::unique_lock lk{mutex};
        if (token_count >= need_count) {
            token_count -= need_count;
            return ORBIS_OK;
        }
        if (!can_block) {
            return ORBIS_KERNEL_ERROR_EBUSY;
        }

        if (timeout && *timeout == 0) {
            return ORBIS_KERNEL_ERROR_ETIMEDOUT;
        }

        WaitingThread waiter{need_count, is_fifo};
        const auto it = AddWaiter(&waiter);

        const s32 result = waiter.Wait(lk, timeout);
        if (result == ORBIS_KERNEL_ERROR_ETIMEDOUT) {
            wait_list.erase(it);
        }
        return result;
    }

    bool Signal(s32 signal_count) {
        std::scoped_lock lk{mutex};
        if (token_count + signal_count > max_count) {
            for (auto it = wait_list.begin(); it != wait_list.end();) {
                auto* waiter = *it;
                if (waiter->need_count > token_count) {
                    ++it;
                    continue;
                }
                it = wait_list.erase(it);
                token_count -= waiter->need_count;
                waiter->was_signaled = true;
                waiter->sem.release();
            }
            return false;
        }
        token_count += signal_count;

        for (auto it = wait_list.begin(); it != wait_list.end();) {
            auto* waiter = *it;
            if (waiter->need_count > token_count) {
                ++it;
                continue;
            }
            it = wait_list.erase(it);
            token_count -= waiter->need_count;
            waiter->was_signaled = true;
            waiter->sem.release();
        }

        return true;
    }

    const std::string& Name() const {
        return name;
    }

    s32 TokenCount() const {
        return token_count.load(std::memory_order_relaxed);
    }

    s32 Cancel(s32 set_count, s32* num_waiters) {
        std::scoped_lock lk{mutex};
        if (num_waiters) {
            *num_waiters = static_cast<s32>(wait_list.size());
        }
        for (auto* waiter : wait_list) {
            waiter->was_canceled = true;
            waiter->sem.release();
        }
        wait_list.clear();
        token_count = set_count < 0 ? init_count : set_count;
        return ORBIS_OK;
    }

    void Delete() {
        std::scoped_lock lk{mutex};
        for (auto* waiter : wait_list) {
            waiter->was_deleted = true;
            waiter->sem.release();
        }
        wait_list.clear();
    }

public:
    struct WaitingThread {
        BinarySemaphore sem;
        u32 priority;
        s32 need_count;
        std::string thr_name;
        bool was_signaled{};
        bool was_deleted{};
        bool was_canceled{};

        explicit WaitingThread(s32 need_count, bool is_fifo)
            : sem{0}, priority{0}, need_count{need_count} {
            const auto* curthread = CurrentOrFallbackPthread();
            if (!is_fifo && curthread != nullptr) {
                priority = curthread->attr.prio;
            }

            thr_name = curthread != nullptr ? curthread->name : "<no-gcurthread>";
        }

        [[nodiscard]] s32 GetResult() const {
            if (was_signaled) {
                return ORBIS_OK;
            }
            if (was_deleted) {
                return ORBIS_KERNEL_ERROR_EACCES;
            }
            if (was_canceled) {
                return ORBIS_KERNEL_ERROR_ECANCELED;
            }
            return ORBIS_KERNEL_ERROR_ETIMEDOUT;
        }

        s32 Wait(std::unique_lock<std::mutex>& lk, u32* timeout) {
            lk.unlock();
            if (!timeout) {
#ifdef __ANDROID__
                if (UseJitNativeSemaphorePark()) {
                    sem.acquire();
                } else {
                    while (!sem.try_acquire_for(std::chrono::milliseconds(1))) {
                        RunMonoPendingSignalSafePointFromWait();
                        std::this_thread::yield();
                    }
                }
#else
                sem.acquire();
#endif
                lk.lock();
            } else {
                const auto start = std::chrono::high_resolution_clock::now();
#ifdef __ANDROID__
                if (UseJitNativeSemaphorePark()) {
                    sem.try_acquire_for(std::chrono::microseconds(*timeout));
                } else {
                    const auto deadline = start + std::chrono::microseconds(*timeout);
                    while (!sem.try_acquire_for(std::chrono::milliseconds(1))) {
                        RunMonoPendingSignalSafePointFromWait();
                        if (std::chrono::high_resolution_clock::now() >= deadline) {
                            break;
                        }
                        std::this_thread::yield();
                    }
                }
#else
                sem.try_acquire_for(std::chrono::microseconds(*timeout));
#endif
                const auto end = std::chrono::high_resolution_clock::now();
                const auto time =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                lk.lock();
                if (was_signaled) {
                    *timeout = time >= *timeout ? 0 : *timeout - static_cast<u32>(time);
                } else {
                    *timeout = 0;
                }
            }
            return GetResult();
        }
    };

    using WaitList = std::list<WaitingThread*>;

    WaitList::iterator AddWaiter(WaitingThread* waiter) {
        if (is_fifo) {
            wait_list.push_back(waiter);
            return --wait_list.end();
        }
        auto it = wait_list.begin();
        while (it != wait_list.end() && (*it)->priority <= waiter->priority) {
            ++it;
        }
        return wait_list.insert(it, waiter);
    }

    WaitList wait_list;
    std::string name;
    std::atomic<s32> token_count;
    std::mutex mutex;
    s32 max_count;
    s32 init_count;
    bool is_fifo;
};

#ifndef __ANDROID__
static bool PcOracleKernelSemaName(const OrbisSem* sem) {
    if (!PcOracleMonoSyncEnabled() || sem == nullptr) {
        return false;
    }
    const char* wide = std::getenv("EXECUTOR_PC_ORACLE_WIDE_SEM");
    if (wide != nullptr && wide[0] != '\0' && std::strcmp(wide, "0") != 0) {
        return true;
    }
    return sem->Name() == "SuspendSemaphore" || sem->Name() == "ResumeSemaphore";
}

static void PcOracleKernelSemaLog(const char* op, Common::SlotId slot, const OrbisSem* sem,
                                  s32 before, s32 after, s32 count, s32 ret) {
    if (!PcOracleKernelSemaName(sem)) {
        return;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_PC_KSEMA] op=%s threadName=%s slot=%u native=%p name=%s before=%d "
                 "after=%d count=%d ret=%d\n",
                 op, PcOracleCurrentThreadName(), slot.index, sem, sem->Name().c_str(), before,
                 after, count, ret);
    std::fflush(stderr);
}
#endif

using OrbisKernelSema = Common::SlotId;

static Common::SlotVector<std::unique_ptr<OrbisSem>> orbis_sems;

#ifdef __ANDROID__
bool ExecutorSignalKernelSemaByNameForLiveMono(const char* name, s32 signal_count) {
    if (name == nullptr || signal_count <= 0) {
        return false;
    }
    for (auto& sem : orbis_sems) {
        if (!sem || sem->Name() != name) {
            continue;
        }
        const s32 before = sem->TokenCount();
        const bool ok = sem->Signal(signal_count);
        const s32 after = sem->TokenCount();
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_COOP_SEMA_SIGNAL] name=%s count=%d before=%d "
                            "after=%d ok=%d",
                            name, signal_count, before, after, ok ? 1 : 0);
        return ok;
    }
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_MONO_COOP_SEMA_SIGNAL] name=%s count=%d result=missing",
                        name, signal_count);
    return false;
}

bool ExecutorWaitKernelSemaByNameForLiveMono(const char* name, s32 need_count) {
    if (name == nullptr || need_count <= 0) {
        return false;
    }
    for (auto& sem : orbis_sems) {
        if (!sem || sem->Name() != name) {
            continue;
        }
        const s32 before = sem->TokenCount();
        const s32 ret = sem->Wait(true, need_count, nullptr);
        const s32 after = sem->TokenCount();
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_COOP_SEMA_WAIT] name=%s count=%d before=%d "
                            "after=%d ret=0x%x",
                            name, need_count, before, after, static_cast<u32>(ret));
        return ret == ORBIS_OK;
    }
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_MONO_COOP_SEMA_WAIT] name=%s count=%d result=missing",
                        name, need_count);
    return false;
}

static std::atomic_int g_executor_live_mono_synthetic_resume_acks{0};

extern "C" int executor_live_mono_synthetic_suspend_ack(void* guest_thread, const char* reason) {
    const bool suspend_acked = ExecutorSignalKernelSemaByNameForLiveMono("SuspendSemaphore", 1);
    if (!suspend_acked) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_MONO_SYNTH_SUSPEND] result=missing_suspend thread=%p "
                            "reason=%s",
                            guest_thread, reason ? reason : "");
        return -2;
    }
    const int before =
        g_executor_live_mono_synthetic_resume_acks.fetch_add(1, std::memory_order_acq_rel);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_SYNTH_SUSPEND] result=ok thread=%p pendingBefore=%d "
                        "pendingAfter=%d reason=%s",
                        guest_thread, before, before + 1, reason ? reason : "");
    return 0;
}

extern "C" int executor_live_mono_synthetic_resume_ack(const char* reason) {
    const int count = g_executor_live_mono_synthetic_resume_acks.exchange(
        0, std::memory_order_acq_rel);
    if (count <= 0) {
        return 0;
    }
    const bool resume_acked = ExecutorSignalKernelSemaByNameForLiveMono("ResumeSemaphore", count);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_SYNTH_RESUME] count=%d resumeAck=%d reason=%s", count,
                        resume_acked ? 1 : 0, reason ? reason : "");
    return resume_acked ? count : -count;
}
#endif

s32 PS4_SYSV_ABI sceKernelCreateSema(OrbisKernelSema* sem, const char* pName, u32 attr,
                                     s32 initCount, s32 maxCount, const void* pOptParam) {
    if (!pName || attr > 2 || initCount < 0 || maxCount <= 0 || initCount > maxCount) {
        LOG_ERROR(Lib_Kernel, "Semaphore creation parameters are invalid!");
#ifdef __ANDROID__
        TraceLiveKernelSem("create_invalid", Common::SlotId{}, nullptr, initCount, -1, maxCount,
                           nullptr, ORBIS_KERNEL_ERROR_EINVAL, pName);
#endif
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    *sem = orbis_sems.insert(
        std::move(std::make_unique<OrbisSem>(initCount, maxCount, pName, attr == 1)));
#ifdef __ANDROID__
    TraceLiveKernelSem("create", *sem, orbis_sems[*sem].get(), initCount, initCount, maxCount,
                       nullptr, ORBIS_OK, pName);
#else
    PcOracleKernelSemaLog("create", *sem, orbis_sems[*sem].get(), initCount, initCount, maxCount,
                          ORBIS_OK);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelWaitSema(OrbisKernelSema sem, s32 needCount, u32* pTimeout) {
    if (!orbis_sems.is_allocated(sem)) {
#ifdef __ANDROID__
        TraceLiveKernelSem("wait_missing", sem, nullptr, -1, -1, needCount, pTimeout,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    auto* native = orbis_sems[sem].get();
    const s32 before = native->token_count.load(std::memory_order_relaxed);
#ifdef __ANDROID__
    const char* mono_suspend = std::getenv("MONO_THREADS_SUSPEND");
    const bool mono_coop = mono_suspend && std::strcmp(mono_suspend, "coop") == 0;
    const bool mono_kernel_suspend_sem =
        native->name == "SuspendSemaphore" || native->name == "ResumeSemaphore";
    if (mono_coop && ExecutorEnvFlag("EXECUTOR_LIVE_MONO_SEM_WAIT_BYPASS") &&
        mono_kernel_suspend_sem) {
        static std::atomic_int log_budget{64};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Native",
                "[EXECUTOR_LIVE_MONO_KSEMA_BYPASS] op=wait slot=%u native=%p before=%d "
                "need=%d thread=%s tid=%ld name=%s reason=coop-suspend",
                sem.index, native, before, needCount,
                g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                static_cast<long>(::syscall(SYS_gettid)), native->name.c_str());
        }
        TraceLiveKernelSem("wait_bypass_mono_suspend", sem, native, before, before, needCount,
                           pTimeout, ORBIS_OK);
        return ORBIS_OK;
    }
    TraceLiveKernelSem("wait_enter", sem, native, before, before, needCount, pTimeout, ORBIS_OK);
    auto executor_wait_watch = BeginLiveKernelSemWaitWatch(sem, native, before, needCount, pTimeout);
    const auto executor_wait_start = std::chrono::steady_clock::now();
#else
    PcOracleKernelSemaLog("wait_enter", sem, native, before, before, needCount, ORBIS_OK);
#endif
    const s32 ret = native->Wait(true, needCount, pTimeout);
    const s32 after = native->token_count.load(std::memory_order_relaxed);
#ifdef __ANDROID__
    if (executor_wait_watch) {
        executor_wait_watch->store(true, std::memory_order_release);
    }
    TraceLiveKernelSem("wait_return", sem, native, before, after, needCount, pTimeout, ret);
    const auto executor_wait_end = std::chrono::steady_clock::now();
    TraceLiveKernelSemDuration(
        sem, native, ret,
        std::chrono::duration_cast<std::chrono::microseconds>(executor_wait_end -
                                                              executor_wait_start)
            .count(),
        pTimeout);
#else
    PcOracleKernelSemaLog("wait_return", sem, native, before, after, needCount, ret);
#endif
    return ret;
}

s32 PS4_SYSV_ABI sceKernelSignalSema(OrbisKernelSema sem, s32 signalCount) {
    if (!orbis_sems.is_allocated(sem)) {
#ifdef __ANDROID__
        TraceLiveKernelSem("signal_missing", sem, nullptr, -1, -1, signalCount, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    auto* native = orbis_sems[sem].get();
    const s32 before = native->token_count.load(std::memory_order_relaxed);
    if (!native->Signal(signalCount)) {
#ifdef __ANDROID__
        TraceLiveKernelSem("signal_invalid", sem, native, before,
                           native->token_count.load(std::memory_order_relaxed), signalCount,
                           nullptr, ORBIS_KERNEL_ERROR_EINVAL);
#endif
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#ifdef __ANDROID__
    TraceLiveKernelSem("signal", sem, native, before,
                       native->token_count.load(std::memory_order_relaxed), signalCount, nullptr,
                       ORBIS_OK);
#else
    PcOracleKernelSemaLog("signal", sem, native, before,
                          native->token_count.load(std::memory_order_relaxed), signalCount,
                          ORBIS_OK);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelPollSema(OrbisKernelSema sem, s32 needCount) {
    if (!orbis_sems.is_allocated(sem)) {
#ifdef __ANDROID__
        TraceLiveKernelSem("poll_missing", sem, nullptr, -1, -1, needCount, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    auto* native = orbis_sems[sem].get();
    const s32 before = native->token_count.load(std::memory_order_relaxed);
    const s32 ret = native->Wait(false, needCount, nullptr);
#ifdef __ANDROID__
    TraceLiveKernelSem("poll", sem, native, before,
                       native->token_count.load(std::memory_order_relaxed), needCount, nullptr,
                       ret);
#endif
    return ret;
}

s32 PS4_SYSV_ABI sceKernelCancelSema(OrbisKernelSema sem, s32 setCount, s32* pNumWaitThreads) {
    if (!orbis_sems.is_allocated(sem)) {
#ifdef __ANDROID__
        TraceLiveKernelSem("cancel_missing", sem, nullptr, -1, -1, setCount, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    auto* native = orbis_sems[sem].get();
    const s32 before = native->token_count.load(std::memory_order_relaxed);
    const s32 ret = native->Cancel(setCount, pNumWaitThreads);
#ifdef __ANDROID__
    TraceLiveKernelSem("cancel", sem, native, before,
                       native->token_count.load(std::memory_order_relaxed), setCount, nullptr,
                       ret);
#endif
    return ret;
}

s32 PS4_SYSV_ABI sceKernelDeleteSema(OrbisKernelSema sem) {
    if (!orbis_sems.is_allocated(sem)) {
#ifdef __ANDROID__
        TraceLiveKernelSem("delete_missing", sem, nullptr, -1, -1, 0, nullptr,
                           ORBIS_KERNEL_ERROR_ESRCH);
#endif
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
#ifdef __ANDROID__
    TraceLiveKernelSem("delete", sem, orbis_sems[sem].get(),
                       orbis_sems[sem]->token_count.load(std::memory_order_relaxed), -1, 0,
                       nullptr, ORBIS_OK);
#endif
    orbis_sems[sem]->Delete();
    orbis_sems.erase(sem);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI posix_sem_init(PthreadSem** sem, s32 pshared, u32 value) {
    if (sem == nullptr || value > ORBIS_KERNEL_SEM_VALUE_MAX) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    PthreadSem* native;
    {
        std::scoped_lock lock{pthread_sem_pool_mutex};
        native = pthread_sem_pool.emplace_back(
            std::make_unique<PthreadSem>(static_cast<s32>(value))).get();
    }
    StorePthreadSem(sem, native);
#ifdef __ANDROID__
    TraceLivePosixSem("init", sem, native, static_cast<s32>(value),
                      native->value.load(std::memory_order_relaxed), 0);
#else
    PcOraclePosixSemLog("init", sem, native, static_cast<s32>(value),
                        native->value.load(std::memory_order_relaxed), 0,
                        __builtin_return_address(0));
#endif
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_destroy(PthreadSem** sem) {
    if (DetachPthreadSem(sem) == nullptr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_wait(PthreadSem** sem) {
    PthreadSem* const native = LoadPthreadSem(sem);
    if (native == nullptr) {
        *__Error() = POSIX_EINVAL;
#ifdef __ANDROID__
        TraceLivePosixSem("wait_invalid", sem, nullptr, -1, -1, -1);
        TraceLiveInvalidPosixSemSlot("wait_invalid", sem);
#else
        PcOraclePosixSemLog("wait_invalid", sem, nullptr, -1, -1, -1,
                            __builtin_return_address(0));
        PcOraclePosixSemInvalidLog("wait_invalid", sem, __builtin_return_address(0));
#endif
        return -1;
    }
    const s32 before = native->value.load(std::memory_order_relaxed);
#ifdef __ANDROID__
    const u32 executor_eop_thread_kind = ExecutorEopTraceThreadKind();
    if (executor_eop_thread_kind != 0) {
        AmdGpu::ExecutorEopTraceSemSync(0, reinterpret_cast<uintptr_t>(sem),
                                       reinterpret_cast<uintptr_t>(native), before, before, 0,
                                       executor_eop_thread_kind);
    }
    if (ShouldBypassLiveMonoSemWait(sem)) {
        static std::atomic_int log_budget{128};
        if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_LIVE_MONO_SEM_BYPASS] op=wait sem=%p native=%p "
                                "before=%d thread=%s tid=%ld ret=0 note=nonblocking_live_probe",
                                sem, native, before,
                                g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>",
                                static_cast<long>(::syscall(SYS_gettid)));
        }
        TraceLivePosixSem("wait_bypass", sem, native, before, before, 0);
        return 0;
    }
#endif
#ifdef __ANDROID__
    TraceLivePosixSem("wait_enter", sem, native, before, before, 0);
    auto executor_wait_watch = BeginLivePosixSemWaitWatch(sem, native, before);
    const auto executor_wait_start = std::chrono::steady_clock::now();
#else
    const void* pc_oracle_return = __builtin_return_address(0);
    PcOraclePosixSemLog("wait_enter", sem, native, before, before, 0, pc_oracle_return);
#endif
#ifdef __ANDROID__
    {
        std::unique_lock wait_lock{native->wait_mutex};
        ++native->waiters;
        while (native->value.load(std::memory_order_acquire) == 0) {
            if (UseJitNativeSemaphorePark()) {
                native->wait_cv.wait(wait_lock);
                continue;
            }
            native->wait_cv.wait_for(wait_lock, std::chrono::milliseconds(1));
            if (native->value.load(std::memory_order_acquire) != 0) {
                break;
            }
            wait_lock.unlock();
            RunMonoPendingSignalSafePointFromWait();
            std::this_thread::yield();
            wait_lock.lock();
        }
        --native->waiters;
        native->value.fetch_sub(1, std::memory_order_acq_rel);
    }
#else
    native->semaphore.acquire();
    native->value.fetch_sub(1, std::memory_order_acq_rel);
#endif
#ifdef __ANDROID__
    if (executor_eop_thread_kind != 0) {
        AmdGpu::ExecutorEopTraceSemSync(
            1, reinterpret_cast<uintptr_t>(sem), reinterpret_cast<uintptr_t>(native), before,
            native->value.load(std::memory_order_relaxed), 0, executor_eop_thread_kind);
    }
    if (executor_wait_watch) {
        executor_wait_watch->store(true, std::memory_order_release);
    }
    TraceLivePosixSem("wait_return", sem, native, before,
                      native->value.load(std::memory_order_relaxed), 0);
    const auto executor_wait_end = std::chrono::steady_clock::now();
    TraceLivePosixSemDuration(
        "wait", sem, native, 0,
        std::chrono::duration_cast<std::chrono::microseconds>(executor_wait_end -
                                                              executor_wait_start)
            .count());
#else
    PcOraclePosixSemLog("wait_return", sem, native, before,
                        native->value.load(std::memory_order_relaxed), 0, pc_oracle_return);
#endif
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_trywait(PthreadSem** sem) {
    PthreadSem* const native = LoadPthreadSem(sem);
    if (native == nullptr) {
        *__Error() = POSIX_EINVAL;
#ifdef __ANDROID__
        TraceLivePosixSem("trywait_invalid", sem, nullptr, -1, -1, -1);
        TraceLiveInvalidPosixSemSlot("trywait_invalid", sem);
#else
        PcOraclePosixSemLog("trywait_invalid", sem, nullptr, -1, -1, -1,
                            __builtin_return_address(0));
        PcOraclePosixSemInvalidLog("trywait_invalid", sem, __builtin_return_address(0));
#endif
        return -1;
    }
    const s32 before = native->value.load(std::memory_order_relaxed);
#ifdef __ANDROID__
    {
        std::lock_guard wait_lock{native->wait_mutex};
        if (native->value.load(std::memory_order_acquire) == 0) {
            *__Error() = POSIX_EAGAIN;
            TraceLivePosixSem("trywait_busy", sem, native, before,
                              native->value.load(std::memory_order_relaxed), -1);
            return -1;
        }
        native->value.fetch_sub(1, std::memory_order_acq_rel);
    }
#else
    if (!native->semaphore.try_acquire()) {
        *__Error() = POSIX_EAGAIN;
#ifdef __ANDROID__
        TraceLivePosixSem("trywait_busy", sem, native, before,
                          native->value.load(std::memory_order_relaxed), -1);
#endif
        return -1;
    }
    native->value.fetch_sub(1, std::memory_order_acq_rel);
#endif
#ifdef __ANDROID__
    TraceLivePosixSem("trywait_return", sem, native, before,
                      native->value.load(std::memory_order_relaxed), 0);
#else
    PcOraclePosixSemLog("trywait_return", sem, native, before,
                        native->value.load(std::memory_order_relaxed), 0,
                        __builtin_return_address(0));
#endif
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_timedwait(PthreadSem** sem, const OrbisKernelTimespec* t) {
    PthreadSem* const native = LoadPthreadSem(sem);
    if (native == nullptr) {
        *__Error() = POSIX_EINVAL;
#ifdef __ANDROID__
        TraceLivePosixSem("timedwait_invalid", sem, nullptr, -1, -1, -1);
        TraceLiveInvalidPosixSemSlot("timedwait_invalid", sem);
#else
        PcOraclePosixSemLog("timedwait_invalid", sem, nullptr, -1, -1, -1,
                            __builtin_return_address(0));
        PcOraclePosixSemInvalidLog("timedwait_invalid", sem, __builtin_return_address(0));
#endif
        return -1;
    }
    const s32 before = native->value.load(std::memory_order_relaxed);
#ifdef __ANDROID__
    TraceLivePosixSem("timedwait_enter", sem, native, before, before, 0);
    auto executor_wait_watch = BeginLivePosixSemWaitWatch(sem, native, before);
    const auto executor_wait_start = std::chrono::steady_clock::now();
#else
    const void* pc_oracle_return = __builtin_return_address(0);
    PcOraclePosixSemLog("timedwait_enter", sem, native, before, before, 0, pc_oracle_return);
#endif
    {
        bool acquired = false;
#ifdef __ANDROID__
        {
            const auto deadline = t->TimePoint();
            std::unique_lock wait_lock{native->wait_mutex};
            ++native->waiters;
            while (native->value.load(std::memory_order_acquire) == 0) {
                if (UseJitNativeSemaphorePark()) {
                    if (native->wait_cv.wait_until(wait_lock, deadline) ==
                            std::cv_status::timeout &&
                        native->value.load(std::memory_order_acquire) == 0) {
                        break;
                    }
                    continue;
                }
                const auto now = std::chrono::system_clock::now();
                if (now >= deadline) {
                    break;
                }
                const auto slice_deadline =
                    std::min(deadline, now + std::chrono::milliseconds(1));
                native->wait_cv.wait_until(wait_lock, slice_deadline);
                if (native->value.load(std::memory_order_acquire) != 0) {
                    break;
                }
                wait_lock.unlock();
                RunMonoPendingSignalSafePointFromWait();
                std::this_thread::yield();
                wait_lock.lock();
            }
            acquired = native->value.load(std::memory_order_acquire) != 0;
            --native->waiters;
            if (acquired) {
                native->value.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
#else
        acquired = native->semaphore.try_acquire_until(t->TimePoint());
#endif
        if (!acquired) {
            *__Error() = POSIX_ETIMEDOUT;
#ifdef __ANDROID__
            if (executor_wait_watch) {
                executor_wait_watch->store(true, std::memory_order_release);
            }
            TraceLivePosixSem("timedwait_timeout", sem, native, before,
                              native->value.load(std::memory_order_relaxed), -1);
            const auto executor_wait_end = std::chrono::steady_clock::now();
            TraceLivePosixSemDuration(
                "timedwait_timeout", sem, native, -1,
                std::chrono::duration_cast<std::chrono::microseconds>(executor_wait_end -
                                                                      executor_wait_start)
                    .count());
#endif
            return -1;
        }
#ifndef __ANDROID__
        native->value.fetch_sub(1, std::memory_order_acq_rel);
#endif
    }
#ifdef __ANDROID__
    if (executor_wait_watch) {
        executor_wait_watch->store(true, std::memory_order_release);
    }
    TraceLivePosixSem("timedwait_return", sem, native, before,
                      native->value.load(std::memory_order_relaxed), 0);
    const auto executor_wait_end = std::chrono::steady_clock::now();
    TraceLivePosixSemDuration(
        "timedwait", sem, native, 0,
        std::chrono::duration_cast<std::chrono::microseconds>(executor_wait_end -
                                                              executor_wait_start)
            .count());
#else
    PcOraclePosixSemLog("timedwait_return", sem, native, before,
                        native->value.load(std::memory_order_relaxed), 0, pc_oracle_return);
#endif
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_post(PthreadSem** sem) {
    PthreadSem* const native = LoadPthreadSem(sem);
    if (native == nullptr) {
        *__Error() = POSIX_EINVAL;
#ifdef __ANDROID__
        TraceLivePosixSem("post_invalid", sem, nullptr, -1, -1, -1);
        TraceLiveInvalidPosixSemSlot("post_invalid", sem);
#else
        PcOraclePosixSemLog("post_invalid", sem, nullptr, -1, -1, -1,
                            __builtin_return_address(0));
        PcOraclePosixSemInvalidLog("post_invalid", sem, __builtin_return_address(0));
#endif
        return -1;
    }
    const s32 before = native->value.load(std::memory_order_relaxed);
#ifdef __ANDROID__
    const u32 executor_eop_thread_kind = ExecutorEopTraceThreadKind();
    if (!PostPthreadSemToken(native)) {
        *__Error() = POSIX_EOVERFLOW;
        TraceLivePosixSem("post_overflow", sem, native, before, before, -1);
        return -1;
    }
    if (executor_eop_thread_kind != 0) {
        AmdGpu::ExecutorEopTraceSemSync(
            2, reinterpret_cast<uintptr_t>(sem), reinterpret_cast<uintptr_t>(native), before,
            native->value.load(std::memory_order_relaxed), 0, executor_eop_thread_kind);
    }
#else
    if (native->value == ORBIS_KERNEL_SEM_VALUE_MAX) {
        *__Error() = POSIX_EOVERFLOW;
#ifdef __ANDROID__
        TraceLivePosixSem("post_overflow", sem, native, before, before, -1);
#endif
        return -1;
    }
    ++native->value;
    native->semaphore.release();
#endif
#ifdef __ANDROID__
    TraceLivePosixSem("post_return", sem, native, before,
                      native->value.load(std::memory_order_relaxed), 0);
#else
    PcOraclePosixSemLog("post_return", sem, native, before,
                        native->value.load(std::memory_order_relaxed), 0,
                        __builtin_return_address(0));
#endif
    return 0;
}

s32 PS4_SYSV_ABI posix_sem_getvalue(PthreadSem** sem, s32* sval) {
    PthreadSem* const native = LoadPthreadSem(sem);
    if (native == nullptr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    if (sval) {
        *sval = native->value;
    }
    return 0;
}

s32 PS4_SYSV_ABI scePthreadSemInit(PthreadSem** sem, s32 flag, u32 value, const char* name) {
    if (flag != 0) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    s32 ret = posix_sem_init(sem, 0, value);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemDestroy(PthreadSem** sem) {
    s32 ret = posix_sem_destroy(sem);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemWait(PthreadSem** sem) {
    s32 ret = posix_sem_wait(sem);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemTrywait(PthreadSem** sem) {
    s32 ret = posix_sem_trywait(sem);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemTimedwait(PthreadSem** sem, u32 usec) {
    OrbisKernelTimespec now{};
    sceKernelClockGettime(ORBIS_CLOCK_REALTIME, &now);
    const u64 total_nsec = static_cast<u64>(now.tv_nsec) +
                           static_cast<u64>(usec % 1000000) * 1000ULL;

    OrbisKernelTimespec time{};
    time.tv_sec = now.tv_sec + usec / 1000000 + total_nsec / 1000000000ULL;
    time.tv_nsec = total_nsec % 1000000000ULL;

    s32 ret = posix_sem_timedwait(sem, &time);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemPost(PthreadSem** sem) {
    s32 ret = posix_sem_post(sem);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI scePthreadSemGetvalue(PthreadSem** sem, s32* sval) {
    s32 ret = posix_sem_getvalue(sem, sval);
    if (ret != 0) {
        return ErrnoToSceKernelError(*__Error());
    }

    return ORBIS_OK;
}

void RegisterSemaphore(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("188x57JYp0g", "libkernel", 1, "libkernel", sceKernelCreateSema);
    LIB_FUNCTION("Zxa0VhQVTsk", "libkernel", 1, "libkernel", sceKernelWaitSema);
    LIB_FUNCTION("4czppHBiriw", "libkernel", 1, "libkernel", sceKernelSignalSema);
    LIB_FUNCTION("12wOHk8ywb0", "libkernel", 1, "libkernel", sceKernelPollSema);
    LIB_FUNCTION("4DM06U2BNEY", "libkernel", 1, "libkernel", sceKernelCancelSema);
    LIB_FUNCTION("R1Jvn8bSCW8", "libkernel", 1, "libkernel", sceKernelDeleteSema);

    LIB_FUNCTION("pDuPEf3m4fI", "libScePosix", 1, "libkernel", posix_sem_init);
    LIB_FUNCTION("cDW233RAwWo", "libScePosix", 1, "libkernel", posix_sem_destroy);
    LIB_FUNCTION("YCV5dGGBcCo", "libScePosix", 1, "libkernel", posix_sem_wait);
    LIB_FUNCTION("WBWzsRifCEA", "libScePosix", 1, "libkernel", posix_sem_trywait);
    LIB_FUNCTION("w5IHyvahg-o", "libScePosix", 1, "libkernel", posix_sem_timedwait);
    LIB_FUNCTION("IKP8typ0QUk", "libScePosix", 1, "libkernel", posix_sem_post);
    LIB_FUNCTION("Bq+LRV-N6Hk", "libScePosix", 1, "libkernel", posix_sem_getvalue);

    LIB_FUNCTION("GEnUkDZoUwY", "libkernel", 1, "libkernel", scePthreadSemInit);
    LIB_FUNCTION("Vwc+L05e6oE", "libkernel", 1, "libkernel", scePthreadSemDestroy);
    LIB_FUNCTION("C36iRE0F5sE", "libkernel", 1, "libkernel", scePthreadSemWait);
    LIB_FUNCTION("H2a+IN9TP0E", "libkernel", 1, "libkernel", scePthreadSemTrywait);
    LIB_FUNCTION("fjN6NQHhK8k", "libkernel", 1, "libkernel", scePthreadSemTimedwait);
    LIB_FUNCTION("aishVAiFaYM", "libkernel", 1, "libkernel", scePthreadSemPost);
    LIB_FUNCTION("DjpBvGlaWbQ", "libkernel", 1, "libkernel", scePthreadSemGetvalue);
}

}
