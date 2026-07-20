// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/threads/thread_state.h"
#include "core/libraries/libs.h"
#include "core/memory.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <cstdlib>
#else
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#endif
#endif

#ifndef _WIN32
#include <signal.h>
#endif

#include <cstring>
#include <atomic>
#include <algorithm>
#include <vector>

#ifdef __ANDROID__

static constexpr const char* ExecutorAndroidLogTag = "LSX4Native";

static size_t ExecutorAndroidGuestThreadMinStack() {
    static size_t value = [] {
        const char* env = std::getenv("EXECUTOR_GUEST_THREAD_MIN_STACK");
        if (env != nullptr && *env != '\0') {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(env, &end, 0);
            if (end != env && parsed >= 256_KB && parsed <= 16_MB) {
                return static_cast<size_t>(parsed);
            }
        }
        return static_cast<size_t>(Libraries::Kernel::ThrStackDefault);
    }();
    return value;
}

extern "C" bool executor_lsx4_android_should_suppress_guest_thread(void* start_routine,
                                                                       void* arg,
                                                                       const char* name)
    __attribute__((weak));
extern "C" void executor_lsx4_android_note_suppressed_guest_thread(void* thread,
                                                                       void* start_routine,
                                                                       void* arg,
                                                                       const char* name)
    __attribute__((weak));
extern "C" bool executor_lsx4_android_should_fail_guest_thread_create(void* start_routine,
                                                                         void* arg,
                                                                         const char* name)
    __attribute__((weak));
extern "C" void executor_lsx4_android_note_guest_thread_create(void* thread,
                                                                  void* start_routine,
                                                                  void* arg,
                                                                  const char* name)
    __attribute__((weak));
extern "C" void* executor_lsx4_android_run_guest_thread(void* thread,
                                                           void* start_routine,
                                                           void* arg,
                                                           const char* name)
    __attribute__((weak));
extern "C" int executor_lsx4_android_runtime_jit_active() __attribute__((weak));
extern "C" void executor_live_note_unity_gfx_worker(void* worker) __attribute__((weak));
extern "C" bool executor_lsx4_android_should_suppress_guest_once(void* once_control,
                                                                     void* init_routine)
    __attribute__((weak));
extern "C" void executor_lsx4_android_note_suppressed_guest_once(void* once_control,
                                                                     void* init_routine)
    __attribute__((weak));
extern "C" int executor_lsx4_android_run_guest_once(void* once_control, void* init_routine)
    __attribute__((weak));
extern "C" bool executor_lsx4_android_should_join_guest_thread(void* start_routine)
    __attribute__((weak));
extern "C" bool executor_lsx4_android_should_join_guest_thread_named(void* start_routine,
                                                                        void* arg,
                                                                        const char* name)
    __attribute__((weak));
extern "C" void executor_live_hle_flight_record_external(
    const char* phase, const char* name, const char* symbol_name, std::uint64_t native_function,
    std::uint64_t result, std::uint64_t arg0, std::uint64_t arg1, std::uint64_t arg2,
    std::uint64_t arg3, std::uint64_t arg4, std::uint64_t arg5, std::uint64_t guest_rsp)
    __attribute__((weak));

static bool ExecutorLightOraclePthreadTrace() {
    return std::getenv("EXECUTOR_LIGHT_ORACLE") != nullptr;
}

static bool ExecutorAndroidJitActive() {
    return executor_lsx4_android_runtime_jit_active != nullptr &&
           executor_lsx4_android_runtime_jit_active() == 1;
}

static const char* ExecutorThreadNameOrUnknown(Libraries::Kernel::Pthread* thread) {
    if (thread == nullptr || thread->name.empty()) {
        return "<unknown>";
    }
    return thread->name.c_str();
}

static std::uint64_t ExecutorGuestOffset(const void* pc) {
    const auto value = reinterpret_cast<std::uintptr_t>(pc);
    return value >= 0x800000000ULL ? value - 0x800000000ULL : 0ULL;
}

static bool ExecutorLooksLikeGuestVa(std::uint64_t value) {
    return (value >= 0x200000000ULL && value < 0x900000000ULL) ||
           (value >= 0x60000000ULL && value < 0x80000000ULL);
}

static bool ExecutorAndroidTryReadQword(const void* base, std::size_t index, std::uint64_t& out) {
    out = 0;
    auto* memory = Core::Memory::Instance();
    const auto addr = reinterpret_cast<VAddr>(base) + index * sizeof(out);
    if (base == nullptr || memory == nullptr || !memory->IsValidMapping(addr, sizeof(out))) {
        return false;
    }
    memory->CopySparseMemory(addr, reinterpret_cast<u8*>(&out), sizeof(out));
    return true;
}

static std::string ExecutorAndroidGuestBytesHex(const void* base, std::size_t size) {
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

static void ExecutorAndroidLogCodeBytes(const char* tag, const char* phase, const char* name,
                                        const char* label, const void* base,
                                        std::size_t total_size) {
    if (base == nullptr || total_size == 0) {
        __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                            "[%s] phase=%s name=%s label=%s base=<null>",
                            tag ? tag : "EXECUTOR_ANDROID_CODE_BYTES",
                            phase ? phase : "<unknown>", name ? name : "<unknown>",
                            label ? label : "<unknown>");
        return;
    }

    constexpr std::size_t Chunk = 0x60;
    const auto addr = reinterpret_cast<std::uintptr_t>(base);
    for (std::size_t offset = 0; offset < total_size; offset += Chunk) {
        const auto chunk = std::min<std::size_t>(Chunk, total_size - offset);
        const auto* ptr = reinterpret_cast<const void*>(addr + offset);
        __android_log_print(
            ANDROID_LOG_INFO, ExecutorAndroidLogTag,
            "[%s] phase=%s name=%s label=%s base=%p off=0x%zx size=0x%zx bytes=%s",
            tag ? tag : "EXECUTOR_ANDROID_CODE_BYTES", phase ? phase : "<unknown>",
            name ? name : "<unknown>", label ? label : "<unknown>", base, offset, chunk,
            ExecutorAndroidGuestBytesHex(ptr, chunk).c_str());
    }
}

static void ExecutorAndroidLogGuestBlock(const char* tag, const char* phase, const char* name,
                                         const void* base, std::size_t bytes) {
    if (base == nullptr) {
        __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                            "[%s] phase=%s name=%s base=<null>", tag ? tag : "EXECUTOR_ANDROID_BLOCK",
                            phase ? phase : "<unknown>", name ? name : "<unknown>");
        return;
    }

    auto* memory = Core::Memory::Instance();
    const auto addr = reinterpret_cast<VAddr>(base);
    const bool mapped = memory != nullptr && memory->IsValidMapping(addr, bytes);
    std::uint64_t q[32]{};
    std::uint64_t ok_mask = 0;
    for (std::size_t i = 0; i < std::size(q); ++i) {
        if (ExecutorAndroidTryReadQword(base, i, q[i])) {
            ok_mask |= 1ULL << i;
        }
    }

    __android_log_print(
        ANDROID_LOG_INFO, ExecutorAndroidLogTag,
        "[%s] phase=%s name=%s base=%p mapped=%d okMask=0x%08llx "
        "q0=0x%llx q1=0x%llx q2=0x%llx q3=0x%llx q4=0x%llx q5=0x%llx "
        "q6=0x%llx q7=0x%llx q8=0x%llx q9=0x%llx q10=0x%llx q11=0x%llx "
        "q12=0x%llx q13=0x%llx q14=0x%llx q15=0x%llx q16=0x%llx q17=0x%llx "
        "q18=0x%llx q19=0x%llx q20=0x%llx q21=0x%llx q22=0x%llx q23=0x%llx "
        "q24=0x%llx q25=0x%llx q26=0x%llx q27=0x%llx q28=0x%llx q29=0x%llx "
        "q30=0x%llx q31=0x%llx bytes=%s",
        tag ? tag : "EXECUTOR_ANDROID_BLOCK", phase ? phase : "<unknown>",
        name ? name : "<unknown>", base, mapped ? 1 : 0,
        static_cast<unsigned long long>(ok_mask), static_cast<unsigned long long>(q[0]),
        static_cast<unsigned long long>(q[1]), static_cast<unsigned long long>(q[2]),
        static_cast<unsigned long long>(q[3]), static_cast<unsigned long long>(q[4]),
        static_cast<unsigned long long>(q[5]), static_cast<unsigned long long>(q[6]),
        static_cast<unsigned long long>(q[7]), static_cast<unsigned long long>(q[8]),
        static_cast<unsigned long long>(q[9]), static_cast<unsigned long long>(q[10]),
        static_cast<unsigned long long>(q[11]), static_cast<unsigned long long>(q[12]),
        static_cast<unsigned long long>(q[13]), static_cast<unsigned long long>(q[14]),
        static_cast<unsigned long long>(q[15]), static_cast<unsigned long long>(q[16]),
        static_cast<unsigned long long>(q[17]), static_cast<unsigned long long>(q[18]),
        static_cast<unsigned long long>(q[19]), static_cast<unsigned long long>(q[20]),
        static_cast<unsigned long long>(q[21]), static_cast<unsigned long long>(q[22]),
        static_cast<unsigned long long>(q[23]), static_cast<unsigned long long>(q[24]),
        static_cast<unsigned long long>(q[25]), static_cast<unsigned long long>(q[26]),
        static_cast<unsigned long long>(q[27]), static_cast<unsigned long long>(q[28]),
        static_cast<unsigned long long>(q[29]), static_cast<unsigned long long>(q[30]),
        static_cast<unsigned long long>(q[31]),
        ExecutorAndroidGuestBytesHex(base, bytes).c_str());
}

static void ExecutorAndroidLogWorkItemDispatch(const char* phase, const char* name,
                                               std::uint64_t work_item) {
    if (work_item == 0) {
        return;
    }

    std::uint64_t vtable = 0;
    if (!ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(work_item), 0, vtable) ||
        vtable == 0) {
        __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                            "[EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH] phase=%s name=%s "
                            "workItem=0x%llx vtable=<missing>",
                            phase ? phase : "<unknown>", name ? name : "<unknown>",
                            static_cast<unsigned long long>(work_item));
        return;
    }

    std::uint64_t slot20 = 0;
    std::uint64_t slot28 = 0;
    std::uint64_t slot30 = 0;
    std::uint64_t slot38 = 0;
    std::uint64_t slot40 = 0;
    std::uint64_t slot48 = 0;
    std::uint64_t slot50 = 0;
    std::uint64_t slot58 = 0;
    (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(vtable + 0x20), 0, slot20);
    (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(vtable + 0x28), 0, slot28);
    (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(vtable + 0x30), 0, slot30);
    (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(vtable + 0x38), 0, slot38);
    (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(vtable + 0x40), 0, slot40);
    (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(vtable + 0x48), 0, slot48);
    (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(vtable + 0x50), 0, slot50);
    (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(vtable + 0x58), 0, slot58);

    __android_log_print(
        ANDROID_LOG_INFO, ExecutorAndroidLogTag,
        "[EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH] phase=%s name=%s workItem=0x%llx "
        "vtable=0x%llx slot20=0x%llx off20=0x%llx slot28=0x%llx off28=0x%llx "
        "slot30=0x%llx off30=0x%llx slot38=0x%llx off38=0x%llx "
        "slot40=0x%llx off40=0x%llx slot48=0x%llx off48=0x%llx "
        "slot50=0x%llx off50=0x%llx slot58=0x%llx off58=0x%llx",
        phase ? phase : "<unknown>", name ? name : "<unknown>",
        static_cast<unsigned long long>(work_item), static_cast<unsigned long long>(vtable),
        static_cast<unsigned long long>(slot20),
        static_cast<unsigned long long>(ExecutorGuestOffset(reinterpret_cast<const void*>(slot20))),
        static_cast<unsigned long long>(slot28),
        static_cast<unsigned long long>(ExecutorGuestOffset(reinterpret_cast<const void*>(slot28))),
        static_cast<unsigned long long>(slot30),
        static_cast<unsigned long long>(ExecutorGuestOffset(reinterpret_cast<const void*>(slot30))),
        static_cast<unsigned long long>(slot38),
        static_cast<unsigned long long>(ExecutorGuestOffset(reinterpret_cast<const void*>(slot38))),
        static_cast<unsigned long long>(slot40),
        static_cast<unsigned long long>(ExecutorGuestOffset(reinterpret_cast<const void*>(slot40))),
        static_cast<unsigned long long>(slot48),
        static_cast<unsigned long long>(ExecutorGuestOffset(reinterpret_cast<const void*>(slot48))),
        static_cast<unsigned long long>(slot50),
        static_cast<unsigned long long>(ExecutorGuestOffset(reinterpret_cast<const void*>(slot50))),
        static_cast<unsigned long long>(slot58),
        static_cast<unsigned long long>(ExecutorGuestOffset(reinterpret_cast<const void*>(slot58))));

    if (std::getenv("EXECUTOR_TRACE_WORK_ITEM_BYTES") != nullptr) {
        ExecutorAndroidLogCodeBytes("EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH_BYTES", phase,
                                    name, "vtable", reinterpret_cast<const void*>(vtable), 0x80);
        ExecutorAndroidLogCodeBytes("EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH_BYTES", phase,
                                    name, "slot20", reinterpret_cast<const void*>(slot20), 0x180);
        ExecutorAndroidLogCodeBytes("EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH_BYTES", phase,
                                    name, "slot28", reinterpret_cast<const void*>(slot28), 0x120);
        ExecutorAndroidLogCodeBytes("EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH_BYTES", phase,
                                    name, "slot30", reinterpret_cast<const void*>(slot30), 0x120);
        ExecutorAndroidLogCodeBytes("EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH_BYTES", phase,
                                    name, "slot38", reinterpret_cast<const void*>(slot38), 0x120);
        ExecutorAndroidLogCodeBytes("EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH_BYTES", phase,
                                    name, "slot40", reinterpret_cast<const void*>(slot40), 0x180);
        ExecutorAndroidLogCodeBytes("EXECUTOR_ANDROID_THREAD_WORK_ITEM_DISPATCH_BYTES", phase,
                                    name, "slot50", reinterpret_cast<const void*>(slot50), 0x80);
    }
}

static bool ExecutorShouldTraceThreadArg(const char* name, const void* start) {
    if (!ExecutorLightOraclePthreadTrace()) {
        return false;
    }
    if (ExecutorGuestOffset(start) == 0x51c20) {
        return true;
    }
    if (name == nullptr) {
        return false;
    }
    return std::strstr(name, "Unity") != nullptr || std::strstr(name, "FMOD") != nullptr ||
           std::strstr(name, "mono") != nullptr || std::strstr(name, "Mono") != nullptr ||
           std::strstr(name, "SceFios") != nullptr;
}

static bool ExecutorVerboseThreadArgBytes() {
    return std::getenv("EXECUTOR_VERBOSE_THREAD_ARG_BYTES") != nullptr;
}

static void ExecutorTraceAndroidThreadArg(const char* phase, Libraries::Kernel::Pthread* thread,
                                          const void* start, const void* arg) {
    const char* name = ExecutorThreadNameOrUnknown(thread);
    if (!ExecutorShouldTraceThreadArg(name, start)) {
        return;
    }
    static std::atomic_int budget{1024};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    auto* memory = Core::Memory::Instance();
    const auto arg_addr = reinterpret_cast<VAddr>(arg);
    const bool arg_mapped =
        arg != nullptr && memory != nullptr && memory->IsValidMapping(arg_addr, 0x120);
    std::uint64_t q[32]{};
    std::uint64_t ok_mask = 0;
    for (std::size_t i = 0; i < 32; ++i) {
        if (ExecutorAndroidTryReadQword(arg, i, q[i])) {
            ok_mask |= 1ULL << i;
        }
    }

    __android_log_print(
        ANDROID_LOG_INFO, ExecutorAndroidLogTag,
        "[EXECUTOR_ANDROID_THREAD_ARG] phase=%s thread=%p name=%s start=%p startOff=0x%llx "
        "arg=%p argMapped=%d okMask=0x%08llx q0=0x%llx q1=0x%llx q2=0x%llx q3=0x%llx "
        "q4=0x%llx q5=0x%llx q6=0x%llx q7=0x%llx q8=0x%llx q9=0x%llx "
        "q10=0x%llx q11=0x%llx q12=0x%llx q13=0x%llx q14=0x%llx q15=0x%llx "
        "q16=0x%llx q17=0x%llx q18=0x%llx q19=0x%llx q20=0x%llx q21=0x%llx "
        "q22=0x%llx q23=0x%llx q24=0x%llx q25=0x%llx q26=0x%llx q27=0x%llx "
        "q28=0x%llx q29=0x%llx q30=0x%llx q31=0x%llx",
        phase != nullptr ? phase : "<unknown>", thread, name, start,
        static_cast<unsigned long long>(ExecutorGuestOffset(start)), arg, arg_mapped ? 1 : 0,
        static_cast<unsigned long long>(ok_mask), static_cast<unsigned long long>(q[0]), static_cast<unsigned long long>(q[1]),
        static_cast<unsigned long long>(q[2]), static_cast<unsigned long long>(q[3]),
        static_cast<unsigned long long>(q[4]), static_cast<unsigned long long>(q[5]),
        static_cast<unsigned long long>(q[6]), static_cast<unsigned long long>(q[7]),
        static_cast<unsigned long long>(q[8]), static_cast<unsigned long long>(q[9]),
        static_cast<unsigned long long>(q[10]), static_cast<unsigned long long>(q[11]),
        static_cast<unsigned long long>(q[12]), static_cast<unsigned long long>(q[13]),
        static_cast<unsigned long long>(q[14]), static_cast<unsigned long long>(q[15]),
        static_cast<unsigned long long>(q[16]), static_cast<unsigned long long>(q[17]),
        static_cast<unsigned long long>(q[18]), static_cast<unsigned long long>(q[19]),
        static_cast<unsigned long long>(q[20]), static_cast<unsigned long long>(q[21]),
        static_cast<unsigned long long>(q[22]), static_cast<unsigned long long>(q[23]),
        static_cast<unsigned long long>(q[24]), static_cast<unsigned long long>(q[25]),
        static_cast<unsigned long long>(q[26]), static_cast<unsigned long long>(q[27]),
        static_cast<unsigned long long>(q[28]), static_cast<unsigned long long>(q[29]),
        static_cast<unsigned long long>(q[30]), static_cast<unsigned long long>(q[31]));

    if (ExecutorGuestOffset(start) == 0x51c20) {
        const auto queue_begin = q[18];
        const auto queue_end = q[19];
        const auto queue_cap = q[20];
        const auto queue_count =
            queue_end >= queue_begin ? (queue_end - queue_begin) / sizeof(std::uint64_t) : 0;
        if (ExecutorGuestOffset(reinterpret_cast<const void*>(q[5])) == 0x7c6f0 &&
            executor_live_note_unity_gfx_worker != nullptr) {
            executor_live_note_unity_gfx_worker(reinterpret_cast<void*>(q[4]));
        }
        if (ExecutorVerboseThreadArgBytes()) {
            __android_log_print(
                ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                "[EXECUTOR_ANDROID_THREAD_ARG_WIDE] phase=%s name=%s arg=%p bytes=%s",
                phase != nullptr ? phase : "<unknown>", name, arg,
                ExecutorAndroidGuestBytesHex(arg, 0x120).c_str());
        }
        if (ExecutorGuestOffset(reinterpret_cast<const void*>(q[5])) == 0x7c6f0) {
            std::uint64_t worker_q[72]{};
            std::uint64_t worker_ok_mask0 = 0;
            std::uint64_t worker_ok_mask1 = 0;
            for (std::size_t i = 0; i < std::size(worker_q); ++i) {
                if (ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(q[4]), i,
                                                worker_q[i])) {
                    if (i < 64) {
                        worker_ok_mask0 |= 1ULL << i;
                    } else {
                        worker_ok_mask1 |= 1ULL << (i - 64);
                    }
                }
            }
            std::uint64_t worker_call28 = 0;
            std::uint64_t stream_vtable = 0;
            std::uint64_t device_vtable = 0;
            std::uint64_t device_call3c8 = 0;
            std::uint64_t device_call578 = 0;
            (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(worker_q[0] + 0x28),
                                              0, worker_call28);
            (void)ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(worker_q[3]), 0,
                                              stream_vtable);
            if (ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(worker_q[65]), 0,
                                            device_vtable)) {
                (void)ExecutorAndroidTryReadQword(
                    reinterpret_cast<const void*>(device_vtable + 0x3c8), 0, device_call3c8);
                (void)ExecutorAndroidTryReadQword(
                    reinterpret_cast<const void*>(device_vtable + 0x578), 0, device_call578);
            }
            if (ExecutorVerboseThreadArgBytes()) {
                __android_log_print(
                    ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                    "[EXECUTOR_ANDROID_GFX_WORKER_ARG] phase=%s name=%s worker=0x%llx "
                    "ok0=0x%016llx ok1=0x%016llx workerVtable=0x%llx workerCall28=0x%llx "
                    "stream=0x%llx streamVtable=0x%llx semD0=0x%llx semE4=0x%llx "
                    "device=0x%llx deviceVtable=0x%llx deviceCall3c8=0x%llx "
                    "deviceCall578=0x%llx stopQ=0x%llx bytes=%s",
                    phase != nullptr ? phase : "<unknown>", name,
                    static_cast<unsigned long long>(q[4]),
                    static_cast<unsigned long long>(worker_ok_mask0),
                    static_cast<unsigned long long>(worker_ok_mask1),
                    static_cast<unsigned long long>(worker_q[0]),
                    static_cast<unsigned long long>(worker_call28),
                    static_cast<unsigned long long>(worker_q[3]),
                    static_cast<unsigned long long>(stream_vtable),
                    static_cast<unsigned long long>(q[4] + 0xd0),
                    static_cast<unsigned long long>(q[4] + 0xe4),
                    static_cast<unsigned long long>(worker_q[65]),
                    static_cast<unsigned long long>(device_vtable),
                    static_cast<unsigned long long>(device_call3c8),
                    static_cast<unsigned long long>(device_call578),
                    static_cast<unsigned long long>(worker_q[66]),
                    ExecutorAndroidGuestBytesHex(reinterpret_cast<const void*>(q[4]), 0x240)
                        .c_str());
            } else {
                __android_log_print(
                    ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                    "[EXECUTOR_ANDROID_GFX_WORKER_ARG] phase=%s name=%s worker=0x%llx "
                    "ok0=0x%016llx ok1=0x%016llx workerVtable=0x%llx workerCall28=0x%llx "
                    "stream=0x%llx streamVtable=0x%llx semD0=0x%llx semE4=0x%llx "
                    "device=0x%llx deviceVtable=0x%llx deviceCall3c8=0x%llx "
                    "deviceCall578=0x%llx stopQ=0x%llx",
                    phase != nullptr ? phase : "<unknown>", name,
                    static_cast<unsigned long long>(q[4]),
                    static_cast<unsigned long long>(worker_ok_mask0),
                    static_cast<unsigned long long>(worker_ok_mask1),
                    static_cast<unsigned long long>(worker_q[0]),
                    static_cast<unsigned long long>(worker_call28),
                    static_cast<unsigned long long>(worker_q[3]),
                    static_cast<unsigned long long>(stream_vtable),
                    static_cast<unsigned long long>(q[4] + 0xd0),
                    static_cast<unsigned long long>(q[4] + 0xe4),
                    static_cast<unsigned long long>(worker_q[65]),
                    static_cast<unsigned long long>(device_vtable),
                    static_cast<unsigned long long>(device_call3c8),
                    static_cast<unsigned long long>(device_call578),
                    static_cast<unsigned long long>(worker_q[66]));
            }
        }
        __android_log_print(
            ANDROID_LOG_INFO, ExecutorAndroidLogTag,
            "[EXECUTOR_ANDROID_THREAD_QUEUE] phase=%s name=%s arg=%p targetArg=0x%llx target=0x%llx "
            "queueBegin=0x%llx queueEnd=0x%llx queueCap=0x%llx queueCount=%llu queueBytes=%s",
            phase != nullptr ? phase : "<unknown>", name, arg,
            static_cast<unsigned long long>(q[4]), static_cast<unsigned long long>(q[5]),
            static_cast<unsigned long long>(queue_begin), static_cast<unsigned long long>(queue_end),
            static_cast<unsigned long long>(queue_cap), static_cast<unsigned long long>(queue_count),
            ExecutorAndroidGuestBytesHex(reinterpret_cast<const void*>(queue_begin),
                                         std::min<std::uint64_t>(0x80, queue_count * sizeof(std::uint64_t)))
                .c_str());
        if (queue_count != 0) {
            std::uint64_t work_item = 0;
            if (ExecutorAndroidTryReadQword(reinterpret_cast<const void*>(queue_begin), 0,
                                            work_item)) {
                ExecutorAndroidLogGuestBlock("EXECUTOR_ANDROID_THREAD_WORK_ITEM", phase, name,
                                             reinterpret_cast<const void*>(work_item), 0x180);
                ExecutorAndroidLogWorkItemDispatch(phase, name, work_item);
            }
        }
        if (ExecutorLooksLikeGuestVa(q[27])) {
            ExecutorAndroidLogGuestBlock("EXECUTOR_ANDROID_THREAD_Q27_BLOCK", phase, name,
                                         reinterpret_cast<const void*>(q[27]), 0x100);
        }
    }
}

static void ExecutorTracePthreadLifecycle(const char* op, Libraries::Kernel::Pthread* thread,
                                          const char* name, const void* start,
                                          const void* arg, const void* extra = nullptr,
                                          int result = 0) {
    if (!ExecutorLightOraclePthreadTrace()) {
        return;
    }
    static std::atomic_int budget{4096};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    const int tid = thread != nullptr ? thread->tid.load(std::memory_order_relaxed) : -1;
    const u32 flags = thread != nullptr ? static_cast<u32>(thread->flags) : 0;
    const u32 state = thread != nullptr ? static_cast<u32>(thread->state) : 0xffffffffu;
    const int refcount = thread != nullptr ? thread->refcount : -1;
    const bool detached =
        thread != nullptr && True(thread->flags & Libraries::Kernel::ThreadFlags::Detached);
    const void* stack = thread != nullptr ? thread->attr.stackaddr_attr : nullptr;
    const size_t stack_size = thread != nullptr ? thread->attr.stacksize_attr : 0;
    const size_t guard_size = thread != nullptr ? thread->attr.guardsize_attr : 0;
    const auto native_handle = thread != nullptr ? thread->native_thr.GetHandle() : 0;
    __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                        "[EXECUTOR_PTHREAD_LIFECYCLE] op=%s cur=%p thread=%p name=\"%s\" "
                        "tid=%d native=%p state=%u flags=0x%x ref=%d detached=%d start=%p "
                        "arg=%p extra=%p result=%d stack=%p stackSize=0x%zx guard=0x%zx",
                        op != nullptr ? op : "<unknown>", Libraries::Kernel::g_curthread, thread,
                        name != nullptr ? name : ExecutorThreadNameOrUnknown(thread), tid,
                        reinterpret_cast<void*>(native_handle), state, flags, refcount,
                        detached ? 1 : 0, start, arg, extra, result, stack, stack_size,
                        guard_size);
    if (executor_live_hle_flight_record_external != nullptr) {
        executor_live_hle_flight_record_external(
            op != nullptr ? op : "pthread", "pthread", name != nullptr ? name : "<unknown>",
            reinterpret_cast<std::uint64_t>(start), static_cast<std::uint64_t>(result),
            reinterpret_cast<std::uint64_t>(thread), reinterpret_cast<std::uint64_t>(start),
            reinterpret_cast<std::uint64_t>(arg), reinterpret_cast<std::uint64_t>(extra),
            reinterpret_cast<std::uint64_t>(stack), static_cast<std::uint64_t>(stack_size), 0);
    }
}
#endif

static void ClearGuestThreadSignalMask(const char* phase, const char* name) {
#ifndef _WIN32
    sigset_t oldset{};
    sigset_t emptyset{};
    sigemptyset(&emptyset);
    const int rc = pthread_sigmask(SIG_SETMASK, &emptyset, &oldset);
#ifdef __ANDROID__
    static std::atomic_uint32_t signal_mask_log_count{0};
    if (rc != 0 || signal_mask_log_count.fetch_add(1, std::memory_order_relaxed) < 4) {
        __android_log_print(rc == 0 ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR,
                            ExecutorAndroidLogTag,
                            "[EXECUTOR_GUEST_THREAD_SIGMASK] phase=%s name=%s rc=%d old_usr1=%d "
                            "old_segv=%d old_bus=%d old_ill=%d",
                            phase ? phase : "<none>", name ? name : "<null>", rc,
                            sigismember(&oldset, SIGUSR1), sigismember(&oldset, SIGSEGV),
                            sigismember(&oldset, SIGBUS), sigismember(&oldset, SIGILL));
    }
#endif
#endif
}

namespace Libraries::Kernel {

extern PthreadAttr PthreadAttrDefault;

#ifndef __ANDROID__
static bool PcOracleMonoSyncEnabled() {
    const char* value = std::getenv("EXECUTOR_PC_ORACLE_MONO_SYNC");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static std::uint64_t PcOracleGuestOff(const void* pc) {
    const auto value = reinterpret_cast<std::uintptr_t>(pc);
    return value >= 0x800000000ULL ? value - 0x800000000ULL : 0ULL;
}

static bool PcOracleLooksLikeGuestVa(std::uint64_t value) {
    return (value >= 0x200000000ULL && value < 0x900000000ULL) ||
           (value >= 0x60000000ULL && value < 0x80000000ULL);
}

static bool PcOracleTryReadQword(const void* base, std::size_t index, std::uint64_t& out) {
    const auto* q = reinterpret_cast<const std::uint64_t*>(base) + index;
#if defined(_MSC_VER)
    __try {
        out = *q;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        out = 0;
        return false;
    }
#else
    auto* memory = Core::Memory::Instance();
    const auto addr = reinterpret_cast<VAddr>(q);
    if (base == nullptr || memory == nullptr) {
        out = 0;
        return false;
    }
    if (memory->IsValidMapping(addr, sizeof(out))) {
        out = *q;
        return true;
    }

    // Desktop oracle fallback: some guest buffers pass byte-wise mapping checks
    // while the coarser qword check rejects the range. Decode through guarded
    // bytes so queue entries are not silently omitted from PC traces.
    std::uint64_t value = 0;
    const auto* p = reinterpret_cast<const std::uint8_t*>(q);
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        const auto byte_addr = reinterpret_cast<VAddr>(p + i);
        if (!memory->IsValidMapping(byte_addr, sizeof(std::uint8_t))) {
            out = 0;
            return false;
        }
        value |= static_cast<std::uint64_t>(p[i]) << (i * 8);
    }
    out = value;
    return true;
#endif
}

static bool PcOracleTryReadByte(const void* base, std::size_t index, std::uint8_t& out) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(base) + index;
#if defined(_MSC_VER)
    __try {
        out = *p;
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        out = 0;
        return false;
    }
#else
    auto* memory = Core::Memory::Instance();
    const auto addr = reinterpret_cast<VAddr>(p);
    if (base == nullptr || memory == nullptr || !memory->IsValidMapping(addr, sizeof(out))) {
        out = 0;
        return false;
    }
    out = *p;
    return true;
#endif
}

static std::string PcOracleBytesHex(const void* base, std::size_t size) {
    if (base == nullptr) {
        return "<null>";
    }
    static constexpr char Hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(size * 3);
    for (std::size_t i = 0; i < size; ++i) {
        std::uint8_t byte = 0;
        if (!PcOracleTryReadByte(base, i, byte)) {
            return "<invalid>";
        }
        if (i != 0) {
            out.push_back(' ');
        }
        out.push_back(Hex[(byte >> 4) & 0xf]);
        out.push_back(Hex[byte & 0xf]);
    }
    return out;
}

static void PcOracleLogBlock(const char* tag, const char* name, const void* base,
                             std::size_t bytes) {
    if (base == nullptr) {
        std::fprintf(stderr, "[%s] name=%s base=<null>\n",
                     tag ? tag : "EXECUTOR_PC_BLOCK", name ? name : "<unknown>");
        return;
    }

    auto* memory = Core::Memory::Instance();
    const auto addr = reinterpret_cast<VAddr>(base);
    const bool mapped = memory != nullptr && memory->IsValidMapping(addr, bytes);
    std::uint64_t q[32]{};
    std::uint64_t ok_mask = 0;
    for (std::size_t i = 0; i < std::size(q); ++i) {
        if (PcOracleTryReadQword(base, i, q[i])) {
            ok_mask |= 1ULL << i;
        }
    }

    std::fprintf(
        stderr,
        "[%s] name=%s base=%p mapped=%d okMask=0x%08llx "
        "q0=0x%llx q1=0x%llx q2=0x%llx q3=0x%llx q4=0x%llx q5=0x%llx "
        "q6=0x%llx q7=0x%llx q8=0x%llx q9=0x%llx q10=0x%llx q11=0x%llx "
        "q12=0x%llx q13=0x%llx q14=0x%llx q15=0x%llx q16=0x%llx q17=0x%llx "
        "q18=0x%llx q19=0x%llx q20=0x%llx q21=0x%llx q22=0x%llx q23=0x%llx "
        "q24=0x%llx q25=0x%llx q26=0x%llx q27=0x%llx q28=0x%llx q29=0x%llx "
        "q30=0x%llx q31=0x%llx bytes=%s\n",
        tag ? tag : "EXECUTOR_PC_BLOCK", name ? name : "<unknown>", base, mapped ? 1 : 0,
        static_cast<unsigned long long>(ok_mask), static_cast<unsigned long long>(q[0]),
        static_cast<unsigned long long>(q[1]), static_cast<unsigned long long>(q[2]),
        static_cast<unsigned long long>(q[3]), static_cast<unsigned long long>(q[4]),
        static_cast<unsigned long long>(q[5]), static_cast<unsigned long long>(q[6]),
        static_cast<unsigned long long>(q[7]), static_cast<unsigned long long>(q[8]),
        static_cast<unsigned long long>(q[9]), static_cast<unsigned long long>(q[10]),
        static_cast<unsigned long long>(q[11]), static_cast<unsigned long long>(q[12]),
        static_cast<unsigned long long>(q[13]), static_cast<unsigned long long>(q[14]),
        static_cast<unsigned long long>(q[15]), static_cast<unsigned long long>(q[16]),
        static_cast<unsigned long long>(q[17]), static_cast<unsigned long long>(q[18]),
        static_cast<unsigned long long>(q[19]), static_cast<unsigned long long>(q[20]),
        static_cast<unsigned long long>(q[21]), static_cast<unsigned long long>(q[22]),
        static_cast<unsigned long long>(q[23]), static_cast<unsigned long long>(q[24]),
        static_cast<unsigned long long>(q[25]), static_cast<unsigned long long>(q[26]),
        static_cast<unsigned long long>(q[27]), static_cast<unsigned long long>(q[28]),
        static_cast<unsigned long long>(q[29]), static_cast<unsigned long long>(q[30]),
        static_cast<unsigned long long>(q[31]), PcOracleBytesHex(base, bytes).c_str());
}

static void PcOracleLogCodeBytes(const char* tag, const char* name, const char* label,
                                 const void* base, std::size_t total_size) {
    if (base == nullptr || total_size == 0) {
        std::fprintf(stderr, "[%s] name=%s label=%s base=<null>\n",
                     tag ? tag : "EXECUTOR_PC_CODE_BYTES", name ? name : "<unknown>",
                     label ? label : "<unknown>");
        return;
    }

    constexpr std::size_t Chunk = 0x60;
    const auto addr = reinterpret_cast<std::uintptr_t>(base);
    for (std::size_t offset = 0; offset < total_size; offset += Chunk) {
        const auto chunk = std::min<std::size_t>(Chunk, total_size - offset);
        const auto* ptr = reinterpret_cast<const void*>(addr + offset);
        std::fprintf(stderr,
                     "[%s] name=%s label=%s base=%p off=0x%zx size=0x%zx bytes=%s\n",
                     tag ? tag : "EXECUTOR_PC_CODE_BYTES", name ? name : "<unknown>",
                     label ? label : "<unknown>", base, offset, chunk,
                     PcOracleBytesHex(ptr, chunk).c_str());
    }
}

static void PcOracleLogWorkItemDispatch(const char* name, std::uint64_t work_item) {
    if (work_item == 0) {
        return;
    }

    std::uint64_t vtable = 0;
    if (!PcOracleTryReadQword(reinterpret_cast<const void*>(work_item), 0, vtable) ||
        vtable == 0) {
        std::fprintf(stderr,
                     "[EXECUTOR_PC_THREAD_WORK_ITEM_DISPATCH] name=%s workItem=0x%llx "
                     "vtable=<missing>\n",
                     name ? name : "<unknown>",
                     static_cast<unsigned long long>(work_item));
        return;
    }

    std::uint64_t slot20 = 0;
    std::uint64_t slot28 = 0;
    std::uint64_t slot30 = 0;
    std::uint64_t slot38 = 0;
    (void)PcOracleTryReadQword(reinterpret_cast<const void*>(vtable + 0x20), 0, slot20);
    (void)PcOracleTryReadQword(reinterpret_cast<const void*>(vtable + 0x28), 0, slot28);
    (void)PcOracleTryReadQword(reinterpret_cast<const void*>(vtable + 0x30), 0, slot30);
    (void)PcOracleTryReadQword(reinterpret_cast<const void*>(vtable + 0x38), 0, slot38);

    std::fprintf(
        stderr,
        "[EXECUTOR_PC_THREAD_WORK_ITEM_DISPATCH] name=%s workItem=0x%llx vtable=0x%llx "
        "slot20=0x%llx off20=0x%llx slot28=0x%llx off28=0x%llx "
        "slot30=0x%llx off30=0x%llx slot38=0x%llx off38=0x%llx\n",
        name ? name : "<unknown>", static_cast<unsigned long long>(work_item),
        static_cast<unsigned long long>(vtable), static_cast<unsigned long long>(slot20),
        static_cast<unsigned long long>(PcOracleGuestOff(reinterpret_cast<const void*>(slot20))),
        static_cast<unsigned long long>(slot28),
        static_cast<unsigned long long>(PcOracleGuestOff(reinterpret_cast<const void*>(slot28))),
        static_cast<unsigned long long>(slot30),
        static_cast<unsigned long long>(PcOracleGuestOff(reinterpret_cast<const void*>(slot30))),
        static_cast<unsigned long long>(slot38),
        static_cast<unsigned long long>(PcOracleGuestOff(reinterpret_cast<const void*>(slot38))));

    PcOracleLogCodeBytes("EXECUTOR_PC_THREAD_WORK_ITEM_DISPATCH_BYTES", name, "vtable",
                         reinterpret_cast<const void*>(vtable), 0x80);
    PcOracleLogCodeBytes("EXECUTOR_PC_THREAD_WORK_ITEM_DISPATCH_BYTES", name, "slot20",
                         reinterpret_cast<const void*>(slot20), 0x180);
    PcOracleLogCodeBytes("EXECUTOR_PC_THREAD_WORK_ITEM_DISPATCH_BYTES", name, "slot28",
                         reinterpret_cast<const void*>(slot28), 0x120);
    PcOracleLogCodeBytes("EXECUTOR_PC_THREAD_WORK_ITEM_DISPATCH_BYTES", name, "slot30",
                         reinterpret_cast<const void*>(slot30), 0x120);
    PcOracleLogCodeBytes("EXECUTOR_PC_THREAD_WORK_ITEM_DISPATCH_BYTES", name, "slot38",
                         reinterpret_cast<const void*>(slot38), 0x120);
}

static void PcOracleThreadCreateLog(const Pthread* creator, const Pthread* created,
                                    PthreadEntryFunc start_routine, void* arg,
                                    const void* guest_return) {
    static std::atomic_int budget{1024};
    if (!PcOracleMonoSyncEnabled() || budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_PC_THREAD_CREATE] creator=%p creatorName=%s thread=%p name=%s "
                 "start=%p startOff=0x%llx arg=%p retaddr=%p returnOff=0x%llx\n",
                 creator, creator && !creator->name.empty() ? creator->name.c_str() : "<none>",
                 created, created && !created->name.empty() ? created->name.c_str() : "<none>",
                 reinterpret_cast<void*>(start_routine),
                 static_cast<unsigned long long>(
                     PcOracleGuestOff(reinterpret_cast<void*>(start_routine))),
                 arg, guest_return,
                 static_cast<unsigned long long>(PcOracleGuestOff(guest_return)));
    auto* memory = Core::Memory::Instance();
    const auto arg_addr = reinterpret_cast<VAddr>(arg);
    const bool arg_mapped = arg != nullptr && memory != nullptr && memory->IsValidMapping(arg_addr, 0x120);
    std::uint64_t q[32]{};
    std::uint64_t ok_mask = 0;
    if (arg != nullptr) {
        for (std::size_t i = 0; i < 32; ++i) {
            if (PcOracleTryReadQword(arg, i, q[i])) {
                ok_mask |= 1ULL << i;
            }
        }
    }
    std::fprintf(stderr,
                 "[EXECUTOR_PC_THREAD_ARG] thread=%p name=%s arg=%p argMapped=%d okMask=0x%08llx "
                 "q0=0x%llx q1=0x%llx q2=0x%llx q3=0x%llx "
                 "q4=0x%llx q5=0x%llx q6=0x%llx q7=0x%llx "
                 "q8=0x%llx q9=0x%llx q10=0x%llx q11=0x%llx "
                 "q12=0x%llx q13=0x%llx q14=0x%llx q15=0x%llx "
                 "q16=0x%llx q17=0x%llx q18=0x%llx q19=0x%llx "
                 "q20=0x%llx q21=0x%llx q22=0x%llx q23=0x%llx "
                 "q24=0x%llx q25=0x%llx q26=0x%llx q27=0x%llx "
                 "q28=0x%llx q29=0x%llx q30=0x%llx q31=0x%llx\n",
                 created, created && !created->name.empty() ? created->name.c_str() : "<none>",
                 arg, arg_mapped ? 1 : 0, static_cast<unsigned long long>(ok_mask), static_cast<unsigned long long>(q[0]),
                 static_cast<unsigned long long>(q[1]), static_cast<unsigned long long>(q[2]),
                 static_cast<unsigned long long>(q[3]), static_cast<unsigned long long>(q[4]),
                 static_cast<unsigned long long>(q[5]), static_cast<unsigned long long>(q[6]),
                 static_cast<unsigned long long>(q[7]), static_cast<unsigned long long>(q[8]),
                 static_cast<unsigned long long>(q[9]), static_cast<unsigned long long>(q[10]),
                 static_cast<unsigned long long>(q[11]), static_cast<unsigned long long>(q[12]),
                 static_cast<unsigned long long>(q[13]), static_cast<unsigned long long>(q[14]),
                 static_cast<unsigned long long>(q[15]), static_cast<unsigned long long>(q[16]),
                 static_cast<unsigned long long>(q[17]), static_cast<unsigned long long>(q[18]),
                 static_cast<unsigned long long>(q[19]), static_cast<unsigned long long>(q[20]),
                 static_cast<unsigned long long>(q[21]), static_cast<unsigned long long>(q[22]),
                 static_cast<unsigned long long>(q[23]), static_cast<unsigned long long>(q[24]),
                 static_cast<unsigned long long>(q[25]), static_cast<unsigned long long>(q[26]),
                 static_cast<unsigned long long>(q[27]), static_cast<unsigned long long>(q[28]),
                 static_cast<unsigned long long>(q[29]), static_cast<unsigned long long>(q[30]),
                 static_cast<unsigned long long>(q[31]));
    if (PcOracleGuestOff(reinterpret_cast<void*>(start_routine)) == 0x51c20) {
        const auto queue_begin = q[18];
        const auto queue_end = q[19];
        const auto queue_cap = q[20];
        const auto queue_count =
            queue_end >= queue_begin ? (queue_end - queue_begin) / sizeof(std::uint64_t) : 0;
        std::fprintf(stderr,
                     "[EXECUTOR_PC_THREAD_ARG_WIDE] name=%s arg=%p bytes=%s\n",
                     created && !created->name.empty() ? created->name.c_str() : "<none>", arg,
                     PcOracleBytesHex(arg, 0x120).c_str());
        std::fprintf(stderr,
                     "[EXECUTOR_PC_THREAD_QUEUE] name=%s arg=%p targetArg=0x%llx target=0x%llx "
                     "queueBegin=0x%llx queueEnd=0x%llx queueCap=0x%llx queueCount=%llu queueBytes=%s\n",
                     created && !created->name.empty() ? created->name.c_str() : "<none>", arg,
                     static_cast<unsigned long long>(q[4]), static_cast<unsigned long long>(q[5]),
                     static_cast<unsigned long long>(queue_begin),
                     static_cast<unsigned long long>(queue_end),
                     static_cast<unsigned long long>(queue_cap),
                     static_cast<unsigned long long>(queue_count),
                     PcOracleBytesHex(reinterpret_cast<const void*>(queue_begin),
                                      std::min<std::uint64_t>(0x80, queue_count * sizeof(std::uint64_t)))
                         .c_str());
        if (queue_count != 0) {
            std::uint64_t work_item = 0;
            if (PcOracleTryReadQword(reinterpret_cast<const void*>(queue_begin), 0,
                                     work_item)) {
                PcOracleLogBlock("EXECUTOR_PC_THREAD_WORK_ITEM",
                                 created && !created->name.empty() ? created->name.c_str()
                                                                   : "<none>",
                                 reinterpret_cast<const void*>(work_item), 0x180);
                PcOracleLogWorkItemDispatch(
                    created && !created->name.empty() ? created->name.c_str() : "<none>",
                    work_item);
            }
        }
        if (PcOracleLooksLikeGuestVa(q[27])) {
            PcOracleLogBlock("EXECUTOR_PC_THREAD_Q27_BLOCK",
                             created && !created->name.empty() ? created->name.c_str()
                                                               : "<none>",
                             reinterpret_cast<const void*>(q[27]), 0x100);
        }
        if (PcOracleGuestOff(reinterpret_cast<void*>(q[5])) == 0x7c6f0) {
            std::uint64_t worker_q[72]{};
            std::uint64_t worker_ok_mask0 = 0;
            std::uint64_t worker_ok_mask1 = 0;
            for (std::size_t i = 0; i < std::size(worker_q); ++i) {
                if (PcOracleTryReadQword(reinterpret_cast<const void*>(q[4]), i, worker_q[i])) {
                    if (i < 64) {
                        worker_ok_mask0 |= 1ULL << i;
                    } else {
                        worker_ok_mask1 |= 1ULL << (i - 64);
                    }
                }
            }
            std::uint64_t worker_call28 = 0;
            std::uint64_t stream_vtable = 0;
            std::uint64_t device_vtable = 0;
            std::uint64_t device_call3c8 = 0;
            std::uint64_t device_call578 = 0;
            (void)PcOracleTryReadQword(reinterpret_cast<const void*>(worker_q[0] + 0x28), 0,
                                       worker_call28);
            (void)PcOracleTryReadQword(reinterpret_cast<const void*>(worker_q[3]), 0,
                                       stream_vtable);
            if (PcOracleTryReadQword(reinterpret_cast<const void*>(worker_q[65]), 0,
                                     device_vtable)) {
                (void)PcOracleTryReadQword(reinterpret_cast<const void*>(device_vtable + 0x3c8),
                                           0, device_call3c8);
                (void)PcOracleTryReadQword(reinterpret_cast<const void*>(device_vtable + 0x578),
                                           0, device_call578);
            }
            std::fprintf(stderr,
                         "[EXECUTOR_PC_GFX_WORKER_ARG] name=%s worker=0x%llx "
                         "ok0=0x%016llx ok1=0x%016llx workerVtable=0x%llx "
                         "workerCall28=0x%llx stream=0x%llx streamVtable=0x%llx "
                         "semD0=0x%llx semE4=0x%llx device=0x%llx deviceVtable=0x%llx "
                         "deviceCall3c8=0x%llx deviceCall578=0x%llx stopQ=0x%llx bytes=%s\n",
                         created && !created->name.empty() ? created->name.c_str() : "<none>",
                         static_cast<unsigned long long>(q[4]),
                         static_cast<unsigned long long>(worker_ok_mask0),
                         static_cast<unsigned long long>(worker_ok_mask1),
                         static_cast<unsigned long long>(worker_q[0]),
                         static_cast<unsigned long long>(worker_call28),
                         static_cast<unsigned long long>(worker_q[3]),
                         static_cast<unsigned long long>(stream_vtable),
                         static_cast<unsigned long long>(q[4] + 0xd0),
                         static_cast<unsigned long long>(q[4] + 0xe4),
                         static_cast<unsigned long long>(worker_q[65]),
                         static_cast<unsigned long long>(device_vtable),
                         static_cast<unsigned long long>(device_call3c8),
                         static_cast<unsigned long long>(device_call578),
                         static_cast<unsigned long long>(worker_q[66]),
                         PcOracleBytesHex(reinterpret_cast<const void*>(q[4]), 0x240).c_str());
        }
    }
    std::fflush(stderr);
}

static void PcOracleThreadStartLog(const Pthread* thread, const char* phase) {
    static std::atomic_int budget{256};
    if (!PcOracleMonoSyncEnabled() || thread == nullptr ||
        budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    const auto start = reinterpret_cast<void*>(thread->start_routine);
    const bool thread_wrapper = PcOracleGuestOff(start) == 0x51c20;
    const bool interesting_name =
        thread->name.find("Unity") != std::string::npos ||
        thread->name.find("mono") != std::string::npos ||
        thread->name.find("Mono") != std::string::npos;
    if (!thread_wrapper && !interesting_name) {
        return;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_PC_THREAD_START] phase=%s thread=%p name=%s start=%p "
                 "startOff=0x%llx arg=%p\n",
                 phase ? phase : "<unknown>", thread,
                 !thread->name.empty() ? thread->name.c_str() : "<none>", start,
                 static_cast<unsigned long long>(PcOracleGuestOff(start)), thread->arg);
    PcOracleThreadCreateLog(nullptr, thread, thread->start_routine, thread->arg, nullptr);
}
#endif

#ifdef __ANDROID__
static const char* ExecutorThreadName(Pthread* thread) {
    return thread ? thread->name.c_str() : "<null>";
}

static void WaitForInitialResume(Pthread* curthread, const char* runner) {
    curthread->lock.lock();
    const bool suspended = True(curthread->flags & ThreadFlags::Suspended);
    curthread->lock.unlock();
    if (!suspended) {
        return;
    }

    __android_log_print(ANDROID_LOG_WARN, ExecutorAndroidLogTag,
                        "[EXECUTOR_PTHREAD_INITIAL_SUSPEND_WAIT] runner=%s thread=%p name=%s",
                        runner ? runner : "?", curthread, ExecutorThreadName(curthread));
    for (;;) {
        curthread->lock.lock();
        const bool still_suspended = True(curthread->flags & ThreadFlags::Suspended);
        curthread->lock.unlock();
        if (!still_suspended) {
            break;
        }
        curthread->wake_sema.acquire();
    }
    __android_log_print(ANDROID_LOG_WARN, ExecutorAndroidLogTag,
                        "[EXECUTOR_PTHREAD_INITIAL_SUSPEND_RUN] runner=%s thread=%p name=%s",
                        runner ? runner : "?", curthread, ExecutorThreadName(curthread));
}

static int ResumeThread(PthreadT thread, const char* api_name) {
    if (thread == nullptr) {
        return POSIX_EINVAL;
    }

    auto* thread_state = ThrState::Instance();
    if (const int ret = thread_state->FindThread(thread, /*include dead*/ false); ret != 0) {
        return ret;
    }

    const bool was_suspended = True(thread->flags & ThreadFlags::Suspended);
    thread->flags &= ~(ThreadFlags::Suspended | ThreadFlags::NeedSuspend);
    thread->attr.suspend = 0;
    thread->lock.unlock();
    if (was_suspended) {
        thread->wake_release_count.fetch_add(1, std::memory_order_relaxed);
        thread->last_wake_site = api_name ? api_name : "resume";
        thread->wake_sema.release();
    }

    __android_log_print(ANDROID_LOG_WARN, ExecutorAndroidLogTag,
                        "[EXECUTOR_PTHREAD_RESUME] api=%s thread=%p name=%s wasSuspended=%d",
                        api_name ? api_name : "?", thread, ExecutorThreadName(thread),
                        was_suspended ? 1 : 0);
    return 0;
}

static int SuspendThread(PthreadT thread, const char* api_name) {
    if (thread == nullptr) {
        return POSIX_EINVAL;
    }

    auto* thread_state = ThrState::Instance();
    if (const int ret = thread_state->FindThread(thread, /*include dead*/ false); ret != 0) {
        return ret;
    }

    const bool was_suspended = True(thread->flags & ThreadFlags::Suspended);
    thread->flags |= ThreadFlags::Suspended | ThreadFlags::NeedSuspend;
    thread->attr.suspend = 1;
    thread->lock.unlock();

    __android_log_print(ANDROID_LOG_WARN, ExecutorAndroidLogTag,
                        "[EXECUTOR_PTHREAD_SUSPEND] api=%s thread=%p name=%s wasSuspended=%d "
                        "note=cooperative-only",
                        api_name ? api_name : "?", thread, ExecutorThreadName(thread),
                        was_suspended ? 1 : 0);
    return 0;
}
#endif

void _thread_cleanupspecific();

using ThreadDtor = void PS4_SYSV_ABI (*)();
static ThreadDtor ThreadDtors{};

void PS4_SYSV_ABI _sceKernelSetThreadDtors(ThreadDtor dtor) {
    ThreadDtors = dtor;
}

static void RunRegisteredThreadDtors() {
    if (ThreadDtors == nullptr) {
        return;
    }
#ifdef __ANDROID__
    // ThreadDtors is guest code on Android. Re-enter the active guest backend instead of ever
    // attempting to branch from AArch64 directly to an x86-64 address. The callback takes no
    // arguments; the cleanup bridge's unused null arg is harmless under the guest SysV ABI.
    RunPthreadCleanupCallback(reinterpret_cast<PthreadCleanupFunc>(ThreadDtors), nullptr,
                              "pthread_thread_dtors");
#else
    ThreadDtors();
#endif
}

static void PrepareThreadExit(void* status, const char* cleanup_reason) {
    Pthread* curthread = g_curthread;

    ASSERT_MSG(!curthread->cancelling, "Thread {} has called pthread_exit from a destructor",
               fmt::ptr(curthread));

    curthread->cancelling = true;
    curthread->no_cancel = true;
    curthread->cancel_async = false;
    curthread->cancel_point = false;
    curthread->ret = status;

    while (!curthread->cleanup.empty()) {
        PthreadCleanup* old = curthread->cleanup.front();
        curthread->cleanup.pop_front();
        RunPthreadCleanupCallback(old->routine, old->routine_arg, cleanup_reason);
        if (old->onheap) {
            delete old;
        }
    }

    // Match the PS4/desktop order: libc's registered per-thread finalizer runs before the POSIX
    // key destructors in FinalizeThreadState(). It may itself create or update pthread-specific
    // values that the latter must observe.
    RunRegisteredThreadDtors();
}

static Pthread* FinalizeThreadState() {
    Pthread* curthread = g_curthread;

    if (curthread->specific != nullptr) {
        _thread_cleanupspecific();
    }

    auto* thread_state = ThrState::Instance();
    ASSERT(thread_state->active_threads.fetch_sub(1) != 1);

    curthread->lock.lock();
    curthread->state = PthreadState::Dead;
    ASSERT(False(curthread->flags & ThreadFlags::NeedSuspend));

    /*
     * Thread was created with initial refcount 1, we drop the
     * reference count to allow it to be garbage collected.
     */
    curthread->refcount--;
    thread_state->TryCollect(curthread); /* thread lock released */

    /*
     * Kernel will do wakeup at the address, so joiner thread
     * will be resumed if it is sleeping at the address.
     */
    // The default seq_cst store has release semantics: cleanup/completion writes become visible
    // before a joiner observes termination. Keep it identical to the desktop shadPS4 contract.
    curthread->tid.store(TidTerminated);
    curthread->tid.notify_all();

    return curthread;
}

static void ExitThread() {
    Pthread* curthread = FinalizeThreadState();

    curthread->native_thr.Exit();
    UNREACHABLE();
    /* Never reach! */
}

#ifdef __ANDROID__
static void FinishThreadAndReturn() {
    Pthread* curthread = g_curthread;
    ExecutorTracePthreadLifecycle("finish_return_begin", curthread, nullptr,
                                  curthread != nullptr
                                      ? reinterpret_cast<void*>(curthread->start_routine)
                                      : nullptr,
                                  curthread != nullptr ? curthread->arg : nullptr);

    curthread = FinalizeThreadState();

    curthread->native_thr.CleanupBeforeReturn();
    ExecutorTracePthreadLifecycle("finish_return_done", curthread, nullptr,
                                  reinterpret_cast<void*>(curthread->start_routine),
                                  curthread->arg);
    g_curthread = nullptr;
}

static void posix_pthread_exit_returning(void* status) {
    Pthread* curthread = g_curthread;
    ExecutorTracePthreadLifecycle("pthread_exit_returning", curthread, nullptr,
                                  curthread != nullptr
                                      ? reinterpret_cast<void*>(curthread->start_routine)
                                      : nullptr,
                                  curthread != nullptr ? curthread->arg : nullptr, status);

    PrepareThreadExit(status, "pthread_exit_returning");
    FinishThreadAndReturn();
}
#endif

void PS4_SYSV_ABI posix_pthread_exit(void* status) {
    Pthread* curthread = g_curthread;
#ifdef __ANDROID__
    ExecutorTracePthreadLifecycle("pthread_exit", curthread, nullptr,
                                  curthread != nullptr
                                      ? reinterpret_cast<void*>(curthread->start_routine)
                                      : nullptr,
                                  curthread != nullptr ? curthread->arg : nullptr, status);
#endif

    PrepareThreadExit(status, "pthread_exit");
    ExitThread();
}

static int JoinThread(PthreadT pthread, void** thread_return, const OrbisKernelTimespec* abstime) {
    Pthread* curthread = g_curthread;

    if (pthread == nullptr) {
        return POSIX_EINVAL;
    }

    if (pthread == curthread) {
        return POSIX_EDEADLK;
    }

    auto* thread_state = ThrState::Instance();
    if (int ret = thread_state->FindThread(pthread, true); ret != 0) {
        return POSIX_ESRCH;
    }

    int ret = 0;
    if (True(pthread->flags & ThreadFlags::Detached)) {
        ret = POSIX_EINVAL;
    } else if (pthread->joiner != nullptr) {
        /* Multiple joiners are not supported. */
        ret = POSIX_ENOTSUP;
    }
    if (ret) {
        pthread->lock.unlock();
        return ret;
    }
    /* Set the running thread to be the joiner: */
    pthread->joiner = curthread;
    pthread->lock.unlock();

    const auto backout_join = [](void* arg) PS4_SYSV_ABI {
        auto* pthread2 = static_cast<Pthread*>(arg);
        std::scoped_lock lk{pthread2->lock};
        pthread2->joiner = nullptr;
    };

    PthreadCleanup cup{backout_join, pthread, 0};
    curthread->cleanup.push_front(&cup);

    //_thr_cancel_enter(curthread);

    const s32 tid = pthread->tid;
    while (pthread->tid.load() != TidTerminated) {
        //_thr_testcancel(curthread);
        ASSERT(abstime == nullptr);
        pthread->tid.wait(tid);
    }

    //_thr_cancel_leave(curthread, 0);
    curthread->cleanup.pop_front();

    if (ret == POSIX_ETIMEDOUT) {
        backout_join(pthread);
        return ret;
    }

    void* tmp = pthread->ret;
    pthread->lock.lock();
    pthread->flags |= ThreadFlags::Detached;
    pthread->joiner = nullptr;
    thread_state->TryCollect(pthread); /* thread lock released */
    if (thread_return != nullptr) {
        *thread_return = tmp;
    }

    return 0;
}

int PS4_SYSV_ABI posix_pthread_join(PthreadT pthread, void** thread_return) {
    return JoinThread(pthread, thread_return, nullptr);
}

int PS4_SYSV_ABI posix_pthread_timedjoin_np(PthreadT pthread, void** thread_return,
                                            const OrbisKernelTimespec* abstime) {
    if (abstime == nullptr || abstime->tv_sec < 0 || abstime->tv_nsec < 0 ||
        abstime->tv_nsec >= 1000000000) {
        return POSIX_EINVAL;
    }

    return JoinThread(pthread, thread_return, abstime);
}

int PS4_SYSV_ABI posix_pthread_detach(PthreadT pthread) {
    if (pthread == nullptr) {
        return POSIX_EINVAL;
    }

    auto* thread_state = ThrState::Instance();
    if (int ret = thread_state->FindThread(pthread, true); ret != 0) {
        return ret;
    }

    /* Check if the thread is already detached or has a joiner. */
    if (True(pthread->flags & ThreadFlags::Detached) || pthread->joiner != nullptr) {
        pthread->lock.unlock();
        return POSIX_EINVAL;
    }

    /* Flag the thread as detached. */
    pthread->flags |= ThreadFlags::Detached;
    thread_state->TryCollect(pthread); /* thread lock released */
    return 0;
}

static void RunThread(void* arg) {
    auto* curthread = static_cast<Pthread*>(arg);
    g_curthread = curthread;
#ifdef __ANDROID__
    ExecutorTracePthreadLifecycle("native_begin", curthread, nullptr,
                                  reinterpret_cast<void*>(curthread->start_routine),
                                  curthread->arg);
#endif
    Common::SetCurrentThreadName(curthread->name.c_str());
    DebugState.AddCurrentThreadToGuestList();
    Core::InitializeTLS();

    curthread->native_thr.Initialize();
    ClearGuestThreadSignalMask("RunThread", curthread->name.c_str());
#ifdef __ANDROID__
    WaitForInitialResume(curthread, "RunThread");
#endif

    // Clear the stack before running the guest thread
    if (False(g_curthread->attr.flags & PthreadAttrFlags::StackUser)) {
        ClearStack();
    }

#ifndef __ANDROID__
    PcOracleThreadStartLog(curthread, "native_entry");
#endif

    /* Run the current thread's start routine with argument: */
    void* ret = curthread->start_routine(curthread->arg);
#ifdef __ANDROID__
    ExecutorTracePthreadLifecycle("native_return", curthread, nullptr,
                                  reinterpret_cast<void*>(curthread->start_routine),
                                  curthread->arg, ret);
#endif

    /* Remove thread from tracking */
    DebugState.RemoveCurrentThreadFromGuestList();
    posix_pthread_exit(ret);
}

#ifdef __ANDROID__
static void RunAndroidMappedGuestThread(void* arg) {
    auto* curthread = static_cast<Pthread*>(arg);
    const bool jit = ExecutorAndroidJitActive();
    static std::atomic_uint32_t mapped_thread_log_count{0};
    const bool trace_guest_thread =
        mapped_thread_log_count.fetch_add(1, std::memory_order_relaxed) < 4;
    const char* runner_name = jit ? "RunJitGuestThread" : "RunBox64GuestThread";
    g_curthread = curthread;
    std::atomic_thread_fence(std::memory_order_acquire);
    ExecutorTracePthreadLifecycle("mapped_begin", curthread, nullptr,
                                  reinterpret_cast<void*>(curthread->start_routine),
                                  curthread->arg);
    ExecutorTraceAndroidThreadArg("mapped_begin", curthread,
                                  reinterpret_cast<void*>(curthread->start_routine),
                                  curthread->arg);
    Common::SetCurrentThreadName(curthread->name.c_str());
    if (trace_guest_thread) {
        __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                            "[EXECUTOR_GUEST_THREAD_NATIVE] begin thread=%p name=%s start=%p arg=%p",
                            curthread, curthread->name.c_str(),
                            reinterpret_cast<void*>(curthread->start_routine), curthread->arg);
    }
    DebugState.AddCurrentThreadToGuestList();
    Core::InitializeTLS();

    curthread->native_thr.Initialize();
    ClearGuestThreadSignalMask(runner_name, curthread->name.c_str());
    WaitForInitialResume(curthread, runner_name);

    // On desktop the guest code runs on the native pthread stack, so RunThread::ClearStack()
    // zeroes only the range below the current stack pointer and leaves the active top frames
    // untouched. Android mapped guest backends use a separate guest x86_64 stack while the
    // host pthread executes on an ARM stack; ClearStack() would look at the wrong SP. Mirror
    // the desktop contract by clearing the lower guest stack while preserving the bootstrap area.
    if (False(curthread->attr.flags & PthreadAttrFlags::StackUser) &&
        curthread->attr.stackaddr_attr != nullptr && curthread->attr.stacksize_attr != 0) {
        constexpr size_t LowGuardBytes = 64;
        constexpr size_t TopBootstrapPreserveBytes = 0x1000;
        const auto stack_base = reinterpret_cast<std::uintptr_t>(curthread->attr.stackaddr_attr);
        const size_t stack_size = curthread->attr.stacksize_attr;
        if (stack_size > LowGuardBytes + TopBootstrapPreserveBytes) {
            const auto clear_base = stack_base + LowGuardBytes;
            const size_t clear_size = stack_size - LowGuardBytes - TopBootstrapPreserveBytes;
            std::memset(reinterpret_cast<void*>(clear_base), 0, clear_size);
            if (trace_guest_thread) {
                __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                                    "[EXECUTOR_GUEST_THREAD_STACK] clear_mapped_partial backend=%s "
                                    "thread=%p name=%s stack=%p size=0x%zx lowGuard=0x%zx "
                                    "topPreserve=0x%zx clearBase=%p clearSize=0x%zx",
                                    jit ? "jit-aarch64-jit" : "box64-fex",
                                    curthread, curthread->name.c_str(),
                                    curthread->attr.stackaddr_attr, stack_size, LowGuardBytes,
                                    TopBootstrapPreserveBytes,
                                    reinterpret_cast<void*>(clear_base), clear_size);
            }
        } else {
            if (trace_guest_thread) {
                __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                                    "[EXECUTOR_GUEST_THREAD_STACK] clear_mapped_skip_small backend=%s "
                                    "thread=%p name=%s stack=%p size=0x%zx lowGuard=0x%zx "
                                    "topPreserve=0x%zx",
                                    jit ? "jit-aarch64-jit" : "box64-fex",
                                    curthread, curthread->name.c_str(),
                                    curthread->attr.stackaddr_attr, stack_size, LowGuardBytes,
                                    TopBootstrapPreserveBytes);
            }
        }
    }

    void* ret = nullptr;
    if (executor_lsx4_android_run_guest_thread != nullptr) {
        ret = executor_lsx4_android_run_guest_thread(
            curthread, reinterpret_cast<void*>(curthread->start_routine), curthread->arg,
            curthread->name.c_str());
    }

    if (trace_guest_thread) {
        __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                            "[EXECUTOR_GUEST_THREAD_NATIVE] mapped_return thread=%p name=%s ret=%p",
                            curthread, curthread->name.c_str(), ret);
    }
    ExecutorTracePthreadLifecycle("mapped_return", curthread, nullptr,
                                  reinterpret_cast<void*>(curthread->start_routine),
                                  curthread->arg, ret);
    DebugState.RemoveCurrentThreadFromGuestList();
    if (trace_guest_thread) {
        __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                            "[EXECUTOR_GUEST_THREAD_NATIVE] pthread_exit_begin thread=%p name=%s ret=%p",
                            curthread, curthread->name.c_str(), ret);
    }
    posix_pthread_exit_returning(ret);
    if (trace_guest_thread) {
        __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                            "[EXECUTOR_GUEST_THREAD_NATIVE] native_return thread=%p", curthread);
    }
}
#endif

int PS4_SYSV_ABI posix_pthread_create_name_np(PthreadT* thread, const PthreadAttrT* attr,
                                              PthreadEntryFunc start_routine, void* arg,
                                              const char* name) {
#ifdef __ANDROID__
    std::atomic_thread_fence(std::memory_order_acquire);
    if (executor_lsx4_android_should_fail_guest_thread_create != nullptr &&
        executor_lsx4_android_should_fail_guest_thread_create(
            reinterpret_cast<void*>(start_routine), arg, name)) {
        return POSIX_EAGAIN;
    }
#endif

    Pthread* curthread = g_curthread;
    auto* thread_state = ThrState::Instance();
    Pthread* new_thread = thread_state->Alloc(curthread);
    if (new_thread == nullptr) {
        return POSIX_EAGAIN;
    }

    if (attr == nullptr || *attr == nullptr) {
        new_thread->attr = PthreadAttrDefault;
    } else {
        new_thread->attr = *(*attr);
        new_thread->attr.cpusetsize = 0;
    }
    if (curthread != nullptr && new_thread->attr.sched_inherit == PthreadInheritSched) {
        if (True(curthread->attr.flags & PthreadAttrFlags::ScopeSystem)) {
            new_thread->attr.flags |= PthreadAttrFlags::ScopeSystem;
        } else {
            new_thread->attr.flags &= ~PthreadAttrFlags::ScopeSystem;
        }
        new_thread->attr.prio = curthread->attr.prio;
        new_thread->attr.sched_policy = curthread->attr.sched_policy;
    }

    static int TidCounter = 1;
    new_thread->tid = ++TidCounter;

    const bool jit = ExecutorAndroidJitActive();
    const bool force_unique_android_mapped_guest_stack =
#ifdef __ANDROID__
        true;
#else
        false;
#endif
#ifdef __ANDROID__
    static std::atomic<std::uint32_t> mapped_stack_trace_count{0};
    const bool trace_mapped_stack =
        force_unique_android_mapped_guest_stack &&
        mapped_stack_trace_count.fetch_add(1, std::memory_order_relaxed) < 4;
#endif

    /* Add additional stack space for HLE */
    static constexpr size_t AdditionalStack = 128_KB;
#ifdef __ANDROID__
    if (force_unique_android_mapped_guest_stack) {
        // In the Android mapped-entry path each guest pthread runs with an explicit emulated
        // x86_64 stack. Unity can pass attr objects whose stack pointer aliases the creator's
        // stack; using that verbatim lets worker threads overwrite each other's guest frames and
        // eventually crash the producer thread. Allocate a fresh emulated stack before
        // ThreadState::CreateStack sees the attr.
        if (trace_mapped_stack) {
            LOG_INFO(Kernel_Pthread,
                     "Android mapped guest thread forcing unique stack, backend={}, old attr stack={}, size={:#x}, flags={:#x}",
                     jit ? "jit-aarch64-jit" : "box64-fex",
                     new_thread->attr.stackaddr_attr, new_thread->attr.stacksize_attr,
                     static_cast<u32>(new_thread->attr.flags));
            __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                                "[EXECUTOR_PTHREAD_STACK_FORCE] backend=%s old_stack=%p "
                                "old_size=0x%zx flags=0x%x",
                                jit ? "jit-aarch64-jit" : "box64-fex",
                                new_thread->attr.stackaddr_attr, new_thread->attr.stacksize_attr,
                                static_cast<unsigned>(new_thread->attr.flags));
        }
        new_thread->attr.stackaddr_attr = nullptr;
        new_thread->attr.flags &= ~PthreadAttrFlags::StackUser;
        if (new_thread->attr.guardsize_attr == 0) {
            new_thread->attr.guardsize_attr = ThrGuardDefault;
        }
        new_thread->attr.stacksize_attr += AdditionalStack;

        // Mapped guest execution needs a conservative x86_64 stack. Keep the minimum
        // configurable to avoid Android LMK while preserving the old default.
        const size_t MinGuestThreadStack = ExecutorAndroidGuestThreadMinStack();
        if (new_thread->attr.stacksize_attr < MinGuestThreadStack) {
            new_thread->attr.stacksize_attr = MinGuestThreadStack;
        }
    } else
#endif
    if (new_thread->attr.stackaddr_attr == nullptr) {
        /* Add additional stack space for HLE */
        new_thread->attr.stacksize_attr += AdditionalStack;
    }

    if (thread_state->CreateStack(&new_thread->attr) != 0) {
        /* Insufficient memory to create a stack: */
        thread_state->Free(curthread, new_thread);
        return POSIX_EAGAIN;
    }

#ifdef __ANDROID__
    if (force_unique_android_mapped_guest_stack) {
        if (trace_mapped_stack) {
            LOG_INFO(Kernel_Pthread,
                     "Android mapped guest thread stack ready backend={}, stack={}, size={:#x}, guard={:#x}, flags={:#x}",
                     jit ? "jit-aarch64-jit" : "box64-fex",
                     new_thread->attr.stackaddr_attr, new_thread->attr.stacksize_attr,
                     new_thread->attr.guardsize_attr, static_cast<u32>(new_thread->attr.flags));
            __android_log_print(ANDROID_LOG_INFO, ExecutorAndroidLogTag,
                                "[EXECUTOR_PTHREAD_STACK_READY] backend=%s stack=%p size=0x%zx "
                                "guard=0x%zx flags=0x%x",
                                jit ? "jit-aarch64-jit" : "box64-fex",
                                new_thread->attr.stackaddr_attr, new_thread->attr.stacksize_attr,
                                new_thread->attr.guardsize_attr,
                                static_cast<unsigned>(new_thread->attr.flags));
        }
    }
#endif

    /*
     * Write a magic value to the thread structure
     * to help identify valid ones:
     */
    new_thread->magic = Pthread::ThrMagic;
    new_thread->start_routine = start_routine;
    new_thread->arg = arg;
    new_thread->cancel_enable = true;
    new_thread->cancel_async = false;

    auto* memory = Core::Memory::Instance();
    if (name && memory->IsValidMapping(reinterpret_cast<VAddr>(name))) {
        new_thread->name = name;
    } else {
        new_thread->name = fmt::format("Thread{}", new_thread->tid.load());
    }

#ifdef __ANDROID__
    ExecutorTracePthreadLifecycle("create_ready", new_thread, new_thread->name.c_str(),
                                  reinterpret_cast<void*>(start_routine), arg, curthread);
#endif

#ifndef __ANDROID__
    PcOracleThreadCreateLog(curthread, new_thread, start_routine, arg,
                            __builtin_return_address(0));
#endif

#ifdef __ANDROID__
    if (executor_lsx4_android_note_guest_thread_create != nullptr) {
        executor_lsx4_android_note_guest_thread_create(
            new_thread, reinterpret_cast<void*>(start_routine), arg, new_thread->name.c_str());
    }
#endif

    new_thread->state = PthreadState::Running;

    if (True(new_thread->attr.flags & PthreadAttrFlags::Detached)) {
        new_thread->flags |= ThreadFlags::Detached;
    }
    if (new_thread->attr.suspend != 0) {
        new_thread->flags |= ThreadFlags::Suspended | ThreadFlags::NeedSuspend;
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, ExecutorAndroidLogTag,
                            "[EXECUTOR_PTHREAD_CREATE_SUSPENDED] thread=%p name=%s start=%p "
                            "arg=%p suspend=%d",
                            new_thread, new_thread->name.c_str(),
                            reinterpret_cast<void*>(start_routine), arg, new_thread->attr.suspend);
#endif
    }

    /* Add the new thread. */
    new_thread->refcount = 1;
    thread_state->Link(curthread, new_thread);
#ifdef __ANDROID__
    ExecutorTracePthreadLifecycle("linked", new_thread, new_thread->name.c_str(),
                                  reinterpret_cast<void*>(start_routine), arg, curthread);
    ExecutorTraceAndroidThreadArg("linked", new_thread, reinterpret_cast<void*>(start_routine),
                                  arg);
#endif

    /* Return thread pointer eariler so that new thread can use it. */
    (*thread) = new_thread;

#ifdef __ANDROID__
    if (executor_lsx4_android_should_suppress_guest_thread != nullptr &&
        executor_lsx4_android_should_suppress_guest_thread(
            reinterpret_cast<void*>(start_routine), arg, new_thread->name.c_str())) {
        if (executor_lsx4_android_run_guest_thread != nullptr) {
            new_thread->native_thr = Core::NativeThread();
            // Android mapped guest threads need two separate stacks:
            // the host pthread stack for native C++/backend frames, and the PS4
            // guest stack stored in new_thread->attr for the emulated x86_64
            // thread. Running the host pthread itself on the guest stack lets
            // guest execution overwrite native return frames.
            std::atomic_thread_fence(std::memory_order_release);
            int ret =
                new_thread->native_thr.CreateDefaultStack(RunAndroidMappedGuestThread, new_thread);
            ASSERT_MSG(ret == 0, "Failed to create Android mapped guest thread with error {}", ret);
            ExecutorTracePthreadLifecycle("mapped_create_ret", new_thread,
                                          new_thread->name.c_str(),
                                          reinterpret_cast<void*>(start_routine), arg, nullptr,
                                          ret);
            if (attr != nullptr && *attr != nullptr && (*attr)->cpuset != nullptr) {
                new_thread->SetAffinity((*attr)->cpuset);
            }
            if (ret) {
                *thread = nullptr;
            }
            // Some Store worker threads (e.g. the icons thread at base+0x290b0)
            // synchronise with the main thread through a guest-side futex that
            // Box64 cannot wake across host threads -> black-screen deadlock. For
            // those, join the worker here so it fully completes (and publishes its
            // state) before pthread_create returns to the guest.
            const bool join_by_start =
                executor_lsx4_android_should_join_guest_thread != nullptr &&
                executor_lsx4_android_should_join_guest_thread(
                    reinterpret_cast<void*>(start_routine));
            const bool join_by_name =
                executor_lsx4_android_should_join_guest_thread_named != nullptr &&
                executor_lsx4_android_should_join_guest_thread_named(
                    reinterpret_cast<void*>(start_routine), arg, new_thread->name.c_str());
            if (ret != 0 || join_by_start || join_by_name) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                    "[EXECUTOR_GUEST_THREAD_JOIN_DECISION] thread=%p name=%s "
                                    "start=%p ret=%d joinStart=%d joinName=%d",
                                    new_thread, new_thread->name.c_str(),
                                    reinterpret_cast<void*>(start_routine), ret,
                                    join_by_start ? 1 : 0, join_by_name ? 1 : 0);
            }
            if (ret == 0 && (join_by_start || join_by_name)) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                    "[EXECUTOR_GUEST_THREAD_JOIN] begin thread=%p name=%s "
                                    "start=%p reason=%s",
                                    new_thread, new_thread->name.c_str(),
                                    reinterpret_cast<void*>(start_routine),
                                    join_by_name ? "name" : "start");
                pthread_join(
                    static_cast<pthread_t>(new_thread->native_thr.GetHandle()), nullptr);
                __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                    "[EXECUTOR_GUEST_THREAD_JOIN] done thread=%p name=%s",
                                    new_thread, new_thread->name.c_str());
            }
            return ret;
        }

        new_thread->ret = nullptr;
        new_thread->state = PthreadState::Dead;
        new_thread->refcount = 0;
        new_thread->tid.store(TidTerminated);
        new_thread->tid.notify_all();
        thread_state->active_threads.fetch_sub(1);
        ExecutorTracePthreadLifecycle("suppressed", new_thread, new_thread->name.c_str(),
                                      reinterpret_cast<void*>(start_routine), arg);
        if (executor_lsx4_android_note_suppressed_guest_thread != nullptr) {
            executor_lsx4_android_note_suppressed_guest_thread(
                new_thread, reinterpret_cast<void*>(start_routine), arg, new_thread->name.c_str());
        }
        return 0;
    }
#endif

    /* Create thread */
    new_thread->native_thr = Core::NativeThread();
#ifdef __ANDROID__
    std::atomic_thread_fence(std::memory_order_release);
#endif
    int ret = new_thread->native_thr.Create(RunThread, new_thread, &new_thread->attr);

    ASSERT_MSG(ret == 0, "Failed to create thread with error {}", ret);
#ifdef __ANDROID__
    ExecutorTracePthreadLifecycle("native_create_ret", new_thread, new_thread->name.c_str(),
                                  reinterpret_cast<void*>(start_routine), arg, nullptr, ret);
#endif

    if (attr != nullptr && *attr != nullptr && (*attr)->cpuset != nullptr) {
        new_thread->SetAffinity((*attr)->cpuset);
    }
    if (ret) {
        *thread = nullptr;
    }
    return ret;
}

int PS4_SYSV_ABI posix_pthread_create(PthreadT* thread, const PthreadAttrT* attr,
                                      PthreadEntryFunc start_routine, void* arg) {
    return posix_pthread_create_name_np(thread, attr, start_routine, arg, nullptr);
}

int PS4_SYSV_ABI posix_pthread_getthreadid_np() {
    Pthread* curthread = CurrentOrFallbackPthread();
    return curthread != nullptr ? curthread->tid.load() : TidTerminated;
}

int PS4_SYSV_ABI posix_pthread_getname_np(PthreadT thread, char* name) {
    if (thread == g_curthread) {
        // The current thread owns its lifetime, so no reference or lock is required.
        std::memcpy(name, thread->name.data(), std::min<size_t>(thread->name.size(), 32));
        return ORBIS_OK;
    }

    auto* thread_state = ThrState::Instance();
    if (int ret = thread_state->RefAdd(thread, false); ret != 0) {
        return POSIX_ESRCH;
    }

    thread->lock.lock();
    if (thread->state != PthreadState::Dead) {
        std::memcpy(name, thread->name.data(), std::min<size_t>(thread->name.size(), 32));
    }
    thread->lock.unlock();
    thread_state->RefDelete(thread);
    return ORBIS_OK;
}

int PS4_SYSV_ABI posix_pthread_equal(PthreadT thread1, PthreadT thread2) {
    return (thread1 == thread2 ? 1 : 0);
}

PthreadT PS4_SYSV_ABI posix_pthread_self() {
    return CurrentOrFallbackPthread();
}

int PS4_SYSV_ABI posix_pthread_rename_np(PthreadT thread, const char* name);

void PS4_SYSV_ABI posix_pthread_set_name_np(PthreadT thread, const char* name) {
    posix_pthread_rename_np(thread, name);
}

void PS4_SYSV_ABI posix_pthread_yield() {
    std::this_thread::yield();
}

void PS4_SYSV_ABI sched_yield() {
    std::this_thread::yield();
}

int PS4_SYSV_ABI posix_getpid() {
    return GLOBAL_PID;
}

int PS4_SYSV_ABI posix_pthread_once(PthreadOnce* once_control,
                                    void PS4_SYSV_ABI (*init_routine)()) {
#ifdef __ANDROID__
    {
        static std::atomic_int once_log_budget{256};
        const int ticket = once_log_budget.fetch_sub(1, std::memory_order_relaxed);
        if (ticket > 0) {
            const char* thread_name = g_curthread ? g_curthread->name.c_str() : "<no-curthread>";
            const auto state = once_control ? once_control->state.load(std::memory_order_relaxed)
                                            : PthreadOnceState::NeverDone;
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_PTHREAD_ONCE] enter once=%p init=%p state=%u thread=%s budget=%d",
                static_cast<void*>(once_control), reinterpret_cast<void*>(init_routine),
                static_cast<unsigned>(state), thread_name ? thread_name : "<unnamed>", ticket);
        }
    }
#endif
    for (;;) {
        auto state = once_control->state.load();
        if (state == PthreadOnceState::Done) {
#ifdef __ANDROID__
            static std::atomic_int done_log_budget{128};
            if (done_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                const char* thread_name = g_curthread ? g_curthread->name.c_str() : "<no-curthread>";
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_PTHREAD_ONCE] already_done once=%p init=%p thread=%s",
                                    static_cast<void*>(once_control),
                                    reinterpret_cast<void*>(init_routine),
                                    thread_name ? thread_name : "<unnamed>");
            }
#endif
            return 0;
        }
        if (state == PthreadOnceState::NeverDone) {
            if (once_control->state.compare_exchange_strong(state, PthreadOnceState::InProgress,
                                                            std::memory_order_acquire)) {
                break;
            }
        } else if (state == PthreadOnceState::InProgress) {
            if (once_control->state.compare_exchange_strong(state, PthreadOnceState::Wait,
                                                            std::memory_order_acquire)) {
                once_control->state.wait(PthreadOnceState::Wait);
            }
        } else if (state == PthreadOnceState::Wait) {
            once_control->state.wait(state);
        } else {
            return POSIX_EINVAL;
        }
    }

#ifdef __ANDROID__
    if (executor_lsx4_android_should_suppress_guest_once != nullptr &&
        executor_lsx4_android_should_suppress_guest_once(
            once_control, reinterpret_cast<void*>(init_routine))) {
        once_control->state.store(PthreadOnceState::Done, std::memory_order_release);
        once_control->state.notify_all();
        if (executor_lsx4_android_note_suppressed_guest_once != nullptr) {
            executor_lsx4_android_note_suppressed_guest_once(
                once_control, reinterpret_cast<void*>(init_routine));
        }
        return 0;
    }

    if (executor_lsx4_android_run_guest_once != nullptr) {
        const int guest_once_result = executor_lsx4_android_run_guest_once(
            once_control, reinterpret_cast<void*>(init_routine));
        if (guest_once_result > 0) {
            once_control->state.store(PthreadOnceState::Done, std::memory_order_release);
            once_control->state.notify_all();
            static std::atomic_int guest_once_ok_budget{128};
            if (guest_once_ok_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                const char* thread_name = g_curthread ? g_curthread->name.c_str() : "<no-curthread>";
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_PTHREAD_ONCE] guest_init_done once=%p init=%p thread=%s",
                    static_cast<void*>(once_control), reinterpret_cast<void*>(init_routine),
                    thread_name ? thread_name : "<unnamed>");
            }
            return 0;
        }
        if (guest_once_result < 0) {
            once_control->state.store(PthreadOnceState::NeverDone, std::memory_order_release);
            once_control->state.notify_all();
            static std::atomic_int guest_once_fail_budget{128};
            if (guest_once_fail_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                const char* thread_name = g_curthread ? g_curthread->name.c_str() : "<no-curthread>";
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_PTHREAD_ONCE] guest_init_failed once=%p init=%p thread=%s",
                    static_cast<void*>(once_control), reinterpret_cast<void*>(init_routine),
                    thread_name ? thread_name : "<unnamed>");
            }
            return POSIX_EINVAL;
        }
    }
#endif

    const auto once_cancel_handler = [](void* arg) PS4_SYSV_ABI {
        auto* once_control2 = static_cast<PthreadOnce*>(arg);
        auto state = PthreadOnceState::InProgress;
        if (once_control2->state.compare_exchange_strong(state, PthreadOnceState::NeverDone,
                                                         std::memory_order_release)) {
            return;
        }

        once_control2->state.store(PthreadOnceState::NeverDone, std::memory_order_release);
        once_control2->state.notify_all();
    };

    PthreadCleanup cup{once_cancel_handler, once_control, 0};
    g_curthread->cleanup.push_front(&cup);
    init_routine();
    g_curthread->cleanup.pop_front();

    auto state = PthreadOnceState::InProgress;
    if (once_control->state.compare_exchange_strong(state, PthreadOnceState::Done,
                                                    std::memory_order_release)) {
        return 0;
    }
    once_control->state.store(PthreadOnceState::Done);
    once_control->state.notify_all();
    return 0;
}

int PS4_SYSV_ABI posix_sched_get_priority_max(SchedPolicy policy) {
    if (policy != SchedPolicy::Fifo && policy != SchedPolicy::RoundRobin) {
        return POSIX_EINVAL;
    }
    return ORBIS_KERNEL_PRIO_FIFO_HIGHEST;
}

int PS4_SYSV_ABI posix_sched_get_priority_min(SchedPolicy policy) {
    if (policy != SchedPolicy::Fifo && policy != SchedPolicy::RoundRobin) {
        return POSIX_EINVAL;
    }
    return ORBIS_KERNEL_PRIO_FIFO_LOWEST;
}

int PS4_SYSV_ABI posix_pthread_rename_np(PthreadT thread, const char* name) {
    LOG_INFO(Kernel_Pthread, "name = {}", name ? name : "(null)");
    auto* thread_state = ThrState::Instance();
    auto* memory = Core::Memory::Instance();

    if (thread == g_curthread) {
#ifdef __ANDROID__
        ExecutorTracePthreadLifecycle("rename_before", thread, thread->name.c_str(),
                                      reinterpret_cast<void*>(thread->start_routine), thread->arg,
                                      name);
#endif
        thread->name = name ? name : std::string{};
        Common::SetThreadName(reinterpret_cast<void*>(thread->native_thr.GetHandle()),
                              thread->name.data());
        if (name && False(thread->attr.flags & PthreadAttrFlags::StackUser)) {
            const VAddr stack_addr = std::bit_cast<VAddr>(thread->attr.stackaddr_attr);
            memory->NameVirtualRange(stack_addr, thread->attr.stacksize_attr, name);
        }
#ifdef __ANDROID__
        ExecutorTracePthreadLifecycle("rename_after", thread, thread->name.c_str(),
                                      reinterpret_cast<void*>(thread->start_routine), thread->arg,
                                      name);
#endif
        return ORBIS_OK;
    }

    if (int ret = thread_state->RefAdd(thread, false); ret != 0) {
        return POSIX_ESRCH;
    }

    thread->lock.lock();
    if (thread->state != PthreadState::Dead) {
#ifdef __ANDROID__
        ExecutorTracePthreadLifecycle("rename_before", thread, thread->name.c_str(),
                                      reinterpret_cast<void*>(thread->start_routine), thread->arg,
                                      name);
#endif
        thread->name = name ? name : std::string{};
        Common::SetThreadName(reinterpret_cast<void*>(thread->native_thr.GetHandle()),
                              thread->name.data());
        if (name && False(thread->attr.flags & PthreadAttrFlags::StackUser)) {
            const VAddr stack_addr = std::bit_cast<VAddr>(thread->attr.stackaddr_attr);
            memory->NameVirtualRange(stack_addr, thread->attr.stacksize_attr, name);
        }
#ifdef __ANDROID__
        ExecutorTracePthreadLifecycle("rename_after", thread, thread->name.c_str(),
                                      reinterpret_cast<void*>(thread->start_routine), thread->arg,
                                      name);
#endif
    }
    thread->lock.unlock();
    thread_state->RefDelete(thread);
    return ORBIS_OK;
}

int PS4_SYSV_ABI posix_pthread_getschedparam(PthreadT pthread, SchedPolicy* policy,
                                             SchedParam* param) {
    if (policy == nullptr || param == nullptr) {
        return POSIX_EINVAL;
    }

    if (pthread == g_curthread) {
        /*
         * Avoid searching the thread list when it is the current
         * thread.
         */
        std::scoped_lock lk{g_curthread->lock};
        *policy = g_curthread->attr.sched_policy;
        param->sched_priority = g_curthread->attr.prio;
        return 0;
    }
    auto* thread_state = ThrState::Instance();
    /* Find the thread in the list of active threads. */
    if (int ret = thread_state->RefAdd(pthread, /*include dead*/ false); ret != 0) {
        return ret;
    }
    pthread->lock.lock();
    *policy = pthread->attr.sched_policy;
    param->sched_priority = pthread->attr.prio;
    pthread->lock.unlock();
    thread_state->RefDelete(pthread);
    return 0;
}

int PS4_SYSV_ABI posix_pthread_setschedparam(PthreadT pthread, SchedPolicy policy,
                                             const SchedParam* param) {
    if (pthread == nullptr || param == nullptr) {
        return POSIX_EINVAL;
    }

    auto* thread_state = ThrState::Instance();
    if (pthread == g_curthread) {
        g_curthread->lock.lock();
    } else if (int ret = thread_state->FindThread(pthread, /*include dead*/ false); ret != 0) {
        return ret;
    }

    if (pthread->attr.sched_policy == policy &&
        (policy == SchedPolicy::Other || pthread->attr.prio == param->sched_priority)) {
        pthread->attr.prio = param->sched_priority;
        pthread->lock.unlock();
        return 0;
    }

    // TODO: _thr_setscheduler
    pthread->attr.sched_policy = policy;
    pthread->attr.prio = param->sched_priority;
    pthread->lock.unlock();
    return 0;
}

int PS4_SYSV_ABI scePthreadGetprio(PthreadT thread, int* priority) {
    SchedParam param;
    SchedPolicy policy;

    const int ret = posix_pthread_getschedparam(thread, &policy, &param);
    if (ret != 0) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    *priority = param.sched_priority;
    return 0;
}

int PS4_SYSV_ABI posix_pthread_setprio(PthreadT thread, int prio) {
    SchedParam param;
    param.sched_priority = prio;

    auto* thread_state = ThrState::Instance();
    if (thread != g_curthread) {
        const int ret = thread_state->RefAdd(thread, /*include dead*/ false);
        if (ret != 0) {
            return ret;
        }
    }

    thread->lock.lock();
    if (thread->attr.sched_policy == SchedPolicy::Other || thread->attr.prio == prio) {
        thread->attr.prio = prio;
    } else {
        // TODO: _thr_setscheduler
        thread->attr.prio = prio;
    }

    thread->lock.unlock();
    if (thread != g_curthread) {
        thread_state->RefDelete(thread);
    }
    return 0;
}

enum class PthreadCancelState : u32 {
    Enable = 0,
    Disable = 1,
};

#define POSIX_PTHREAD_CANCELED ((void*)1)

static inline void TestCancel(const Pthread* curthread) {
    if (curthread->ShouldCancel() && !curthread->InCritical()) [[unlikely]] {
        posix_pthread_exit(POSIX_PTHREAD_CANCELED);
    }
}

int PS4_SYSV_ABI posix_pthread_setcancelstate(PthreadCancelState state,
                                              PthreadCancelState* oldstate) {
    Pthread* curthread = g_curthread;
    int oldval = curthread->cancel_enable;
    switch (state) {
    case PthreadCancelState::Disable:
        curthread->cancel_enable = false;
        break;
    case PthreadCancelState::Enable:
        curthread->cancel_enable = true;
        TestCancel(curthread);
        break;
    default:
        return POSIX_EINVAL;
    }

    if (oldstate) {
        *oldstate = oldval ? PthreadCancelState::Enable : PthreadCancelState::Disable;
    }
    return 0;
}

int Pthread::SetAffinity(const Cpuset* cpuset) {
    const auto processor_count = std::thread::hardware_concurrency();
    if (processor_count < 8) {
        return 0;
    }
    if (cpuset == nullptr) {
        return POSIX_EINVAL;
    }

    uintptr_t handle = native_thr.GetHandle();
    if (handle == 0) {
        return POSIX_ESRCH;
    }

    // We don't use this currently because some games gets performance problems
    // when applying affinity even on strong hardware
    /*
    u64 mask = cpuset->bits;
    #ifdef _WIN64
        DWORD_PTR affinity_mask = static_cast<DWORD_PTR>(mask);
        if (!SetThreadAffinityMask(reinterpret_cast<HANDLE>(handle), affinity_mask)) {
            return POSIX_EINVAL;
        }

    #elif defined(__linux__)
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);

        u64 mask = cpuset->bits;
        for (int cpu = 0; cpu < std::min(64, CPU_SETSIZE); ++cpu) {
            if (mask & (1ULL << cpu)) {
                CPU_SET(cpu, &cpu_set);
            }
        }

        int result =
            pthread_setaffinity_np(static_cast<pthread_t>(handle), sizeof(cpu_set_t), &cpu_set);
        if (result != 0) {
            return POSIX_EINVAL;
        }
    #endif
    */
    return 0;
}

int PS4_SYSV_ABI posix_pthread_getaffinity_np(PthreadT thread, size_t cpusetsize, Cpuset* cpusetp) {
    if (thread == nullptr || cpusetp == nullptr) {
        return POSIX_EINVAL;
    }

    auto* thread_state = ThrState::Instance();
    if (thread == g_curthread) {
        g_curthread->lock.lock();
    } else if (const auto ret = thread_state->FindThread(thread, /*include dead*/ false);
               ret != 0) {
        return ret;
    }

    auto* attr_ptr = &thread->attr;
    auto ret = posix_pthread_attr_getaffinity_np(&attr_ptr, cpusetsize, cpusetp);

    thread->lock.unlock();
    return ret;
}

int PS4_SYSV_ABI posix_pthread_setaffinity_np(PthreadT thread, size_t cpusetsize,
                                              const Cpuset* cpusetp) {
    if (thread == nullptr || cpusetp == nullptr) {
        return POSIX_EINVAL;
    }

    auto* thread_state = ThrState::Instance();
    if (thread == g_curthread) {
        g_curthread->lock.lock();
    } else if (const auto ret = thread_state->FindThread(thread, /*include dead*/ false);
               ret != 0) {
        return ret;
    }

    auto* attr_ptr = &thread->attr;
    auto ret = posix_pthread_attr_setaffinity_np(&attr_ptr, cpusetsize, cpusetp);

    if (ret == ORBIS_OK) {
        ret = thread->SetAffinity(thread->attr.cpuset);
    }

    thread->lock.unlock();
    return ret;
}

int PS4_SYSV_ABI scePthreadGetaffinity(PthreadT thread, u64* mask) {
    Cpuset cpuset;
    const int ret = posix_pthread_getaffinity_np(thread, sizeof(Cpuset), &cpuset);
    if (ret == 0) {
        *mask = cpuset.bits;
    }
    return ret;
}

int PS4_SYSV_ABI scePthreadSetaffinity(PthreadT thread, const u64 mask) {
    const Cpuset cpuset = {.bits = mask};
    return posix_pthread_setaffinity_np(thread, sizeof(Cpuset), &cpuset);
}

int PS4_SYSV_ABI posix_pthread_resume_np(PthreadT thread) {
#ifdef __ANDROID__
    return ResumeThread(thread, "pthread_resume_np");
#else
    return POSIX_ENOTSUP;
#endif
}

int PS4_SYSV_ABI posix_pthread_suspend_np(PthreadT thread) {
#ifdef __ANDROID__
    return SuspendThread(thread, "pthread_suspend_np");
#else
    return POSIX_ENOTSUP;
#endif
}

int PS4_SYSV_ABI posix_pthread_resume_all_np() {
#ifdef __ANDROID__
    auto* thread_state = ThrState::Instance();
    std::vector<Pthread*> snapshot;
    {
        std::scoped_lock lk{thread_state->thread_list_lock};
        snapshot.assign(thread_state->threads.begin(), thread_state->threads.end());
    }
    int resumed = 0;
    for (Pthread* thread : snapshot) {
        if (thread != g_curthread && ResumeThread(thread, "pthread_resume_all_np") == 0) {
            resumed++;
        }
    }
    __android_log_print(ANDROID_LOG_WARN, ExecutorAndroidLogTag,
                        "[EXECUTOR_PTHREAD_RESUME_ALL] resumed=%d", resumed);
    return 0;
#else
    return POSIX_ENOTSUP;
#endif
}

int PS4_SYSV_ABI posix_pthread_suspend_all_np() {
#ifdef __ANDROID__
    auto* thread_state = ThrState::Instance();
    std::vector<Pthread*> snapshot;
    {
        std::scoped_lock lk{thread_state->thread_list_lock};
        snapshot.assign(thread_state->threads.begin(), thread_state->threads.end());
    }
    int suspended = 0;
    for (Pthread* thread : snapshot) {
        if (thread != g_curthread && SuspendThread(thread, "pthread_suspend_all_np") == 0) {
            suspended++;
        }
    }
    __android_log_print(ANDROID_LOG_WARN, ExecutorAndroidLogTag,
                        "[EXECUTOR_PTHREAD_SUSPEND_ALL] suspended=%d note=cooperative-only",
                        suspended);
    return 0;
#else
    return POSIX_ENOTSUP;
#endif
}

int PS4_SYSV_ABI scePthreadResume(PthreadT thread) {
    return posix_pthread_resume_np(thread);
}

int PS4_SYSV_ABI scePthreadSuspend(PthreadT thread) {
    return posix_pthread_suspend_np(thread);
}

int PS4_SYSV_ABI scePthreadResumeAll() {
    return posix_pthread_resume_all_np();
}

int PS4_SYSV_ABI scePthreadSuspendAll() {
    return posix_pthread_suspend_all_np();
}

void RegisterThread(Core::Loader::SymbolsResolver* sym) {
    // Posix
    LIB_FUNCTION("Z4QosVuAsA0", "libScePosix", 1, "libkernel", posix_pthread_once);
    LIB_FUNCTION("7Xl257M4VNI", "libScePosix", 1, "libkernel", posix_pthread_equal);
    LIB_FUNCTION("CBNtXOoef-E", "libScePosix", 1, "libkernel", posix_sched_get_priority_max);
    LIB_FUNCTION("m0iS6jNsXds", "libScePosix", 1, "libkernel", posix_sched_get_priority_min);
    LIB_FUNCTION("EotR8a3ASf4", "libScePosix", 1, "libkernel", posix_pthread_self);
    LIB_FUNCTION("B5GmVDKwpn0", "libScePosix", 1, "libkernel", posix_pthread_yield);
    LIB_FUNCTION("+U1R4WtXvoc", "libScePosix", 1, "libkernel", posix_pthread_detach);
    LIB_FUNCTION("FJrT5LuUBAU", "libScePosix", 1, "libkernel", posix_pthread_exit);
    LIB_FUNCTION("h9CcP3J0oVM", "libScePosix", 1, "libkernel", posix_pthread_join);
    LIB_FUNCTION("OxhIB8LB-PQ", "libScePosix", 1, "libkernel", posix_pthread_create);
    LIB_FUNCTION("Jmi+9w9u0E4", "libScePosix", 1, "libkernel", posix_pthread_create_name_np);
    LIB_FUNCTION("4e9dMKt+UYA", "libScePosix", 1, "libkernel", posix_pthread_suspend_np);
    LIB_FUNCTION("BYM3L--ojzI", "libScePosix", 1, "libkernel", posix_pthread_resume_np);
    LIB_FUNCTION("iWAnZ3ger+8", "libScePosix", 1, "libkernel", posix_pthread_suspend_all_np);
    LIB_FUNCTION("3gY5B0FCkNY", "libScePosix", 1, "libkernel", posix_pthread_resume_all_np);
    LIB_FUNCTION("lZzFeSxPl08", "libScePosix", 1, "libkernel", posix_pthread_setcancelstate);
    LIB_FUNCTION("a2P9wYGeZvc", "libScePosix", 1, "libkernel", posix_pthread_setprio);
    LIB_FUNCTION("9vyP6Z7bqzc", "libScePosix", 1, "libkernel", posix_pthread_rename_np);
    LIB_FUNCTION("FIs3-UQT9sg", "libScePosix", 1, "libkernel", posix_pthread_getschedparam);
    LIB_FUNCTION("Xs9hdiD7sAA", "libScePosix", 1, "libkernel", posix_pthread_setschedparam);
    LIB_FUNCTION("6XG4B33N09g", "libScePosix", 1, "libkernel", sched_yield);
    LIB_FUNCTION("HoLVWNanBBc", "libScePosix", 1, "libkernel", posix_getpid);

    // Posix-Kernel
    LIB_FUNCTION("Z4QosVuAsA0", "libkernel", 1, "libkernel", posix_pthread_once);
    LIB_FUNCTION("EotR8a3ASf4", "libkernel", 1, "libkernel", posix_pthread_self);
    LIB_FUNCTION("OxhIB8LB-PQ", "libkernel", 1, "libkernel", posix_pthread_create);
    LIB_FUNCTION("Jmi+9w9u0E4", "libkernel", 1, "libkernel", posix_pthread_create_name_np);
    LIB_FUNCTION("4e9dMKt+UYA", "libkernel", 1, "libkernel", posix_pthread_suspend_np);
    LIB_FUNCTION("BYM3L--ojzI", "libkernel", 1, "libkernel", posix_pthread_resume_np);
    LIB_FUNCTION("iWAnZ3ger+8", "libkernel", 1, "libkernel", posix_pthread_suspend_all_np);
    LIB_FUNCTION("3gY5B0FCkNY", "libkernel", 1, "libkernel", posix_pthread_resume_all_np);
    LIB_FUNCTION("lZzFeSxPl08", "libkernel", 1, "libkernel", posix_pthread_setcancelstate);
    LIB_FUNCTION("CBNtXOoef-E", "libkernel", 1, "libkernel", posix_sched_get_priority_max);
    LIB_FUNCTION("m0iS6jNsXds", "libkernel", 1, "libkernel", posix_sched_get_priority_min);
    LIB_FUNCTION("Xs9hdiD7sAA", "libkernel", 1, "libkernel", posix_pthread_setschedparam);
    LIB_FUNCTION("+U1R4WtXvoc", "libkernel", 1, "libkernel", posix_pthread_detach);
    LIB_FUNCTION("7Xl257M4VNI", "libkernel", 1, "libkernel", posix_pthread_equal);
    LIB_FUNCTION("h9CcP3J0oVM", "libkernel", 1, "libkernel", posix_pthread_join);
    LIB_FUNCTION("Jb2uGFMr688", "libkernel", 1, "libkernel", posix_pthread_getaffinity_np);
    LIB_FUNCTION("5KWrg7-ZqvE", "libkernel", 1, "libkernel", posix_pthread_setaffinity_np);
    LIB_FUNCTION("3eqs37G74-s", "libkernel", 1, "libkernel", posix_pthread_getthreadid_np);

    // Orbis
    LIB_FUNCTION("14bOACANTBo", "libkernel", 1, "libkernel", ORBIS(posix_pthread_once));
    LIB_FUNCTION("GBUY7ywdULE", "libkernel", 1, "libkernel", ORBIS(posix_pthread_rename_np));
    LIB_FUNCTION("6UgtwV+0zb4", "libkernel", 1, "libkernel", ORBIS(posix_pthread_create_name_np));
    LIB_FUNCTION("4qGrR6eoP9Y", "libkernel", 1, "libkernel", ORBIS(posix_pthread_detach));
    LIB_FUNCTION("onNY9Byn-W8", "libkernel", 1, "libkernel", ORBIS(posix_pthread_join));
    LIB_FUNCTION("P41kTWUS3EI", "libkernel", 1, "libkernel", ORBIS(posix_pthread_getschedparam));
    LIB_FUNCTION("oIRFTjoILbg", "libkernel", 1, "libkernel", ORBIS(posix_pthread_setschedparam));
    LIB_FUNCTION("How7B8Oet6k", "libkernel", 1, "libkernel", ORBIS(posix_pthread_getname_np));
    LIB_FUNCTION("3kg7rT0NQIs", "libkernel", 1, "libkernel", posix_pthread_exit);
    LIB_FUNCTION("aI+OeCz8xrQ", "libkernel", 1, "libkernel", posix_pthread_self);
    LIB_FUNCTION("oxMp8uPqa+U", "libkernel", 1, "libkernel", posix_pthread_set_name_np);
    LIB_FUNCTION("3PtV6p3QNX4", "libkernel", 1, "libkernel", posix_pthread_equal);
    LIB_FUNCTION("T72hz6ffq08", "libkernel", 1, "libkernel", posix_pthread_yield);
    LIB_FUNCTION("EI-5-jlq2dE", "libkernel", 1, "libkernel", posix_pthread_getthreadid_np);
    LIB_FUNCTION("1tKyG7RlMJo", "libkernel", 1, "libkernel", scePthreadGetprio);
    LIB_FUNCTION("W0Hpm2X0uPE", "libkernel", 1, "libkernel", ORBIS(posix_pthread_setprio));
    LIB_FUNCTION("rNhWz+lvOMU", "libkernel", 1, "libkernel", _sceKernelSetThreadDtors);
    LIB_FUNCTION("6XG4B33N09g", "libkernel", 1, "libkernel", sched_yield);
    LIB_FUNCTION("HoLVWNanBBc", "libkernel", 1, "libkernel", posix_getpid);
    LIB_FUNCTION("rcrVFJsQWRY", "libkernel", 1, "libkernel", ORBIS(scePthreadGetaffinity));
    LIB_FUNCTION("bt3CTBKmGyI", "libkernel", 1, "libkernel", ORBIS(scePthreadSetaffinity));
    LIB_FUNCTION("DB7Mkm+Pqzw", "libkernel", 1, "libkernel", ORBIS(scePthreadResume));
    LIB_FUNCTION("ywmONkF81ok", "libkernel", 1, "libkernel", ORBIS(scePthreadSuspend));
    LIB_FUNCTION("te+MBYMzDhY", "libkernel", 1, "libkernel", ORBIS(scePthreadResumeAll));
    LIB_FUNCTION("HlzHlgqiBo8", "libkernel", 1, "libkernel", ORBIS(scePthreadSuspendAll));
}

} // namespace Libraries::Kernel
