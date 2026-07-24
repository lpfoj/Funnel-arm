// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

#ifdef __ANDROID__
extern "C" int executor_lsx4_android_run_guest_callback6(
    void* callback, std::uint64_t arg0, std::uint64_t arg1, std::uint64_t arg2,
    std::uint64_t arg3, std::uint64_t arg4, std::uint64_t arg5,
    std::uint32_t arg_count, std::uint64_t* result_out,
    const char* reason) __attribute__((weak));

extern "C" int executor_lsx4_android_run_guest_callback4(
    void* callback, std::uint64_t arg0, std::uint64_t arg1, std::uint64_t arg2,
    std::uint64_t arg3, std::uint32_t arg_count, std::uint64_t* result_out,
    const char* reason) __attribute__((weak));
#endif

namespace Libraries {

inline int RunGuestCallback6(void* callback, const std::uint64_t arg0,
                             const std::uint64_t arg1, const std::uint64_t arg2,
                             const std::uint64_t arg3, const std::uint64_t arg4,
                             const std::uint64_t arg5, const std::uint32_t arg_count,
                             std::uint64_t* result_out, const char* reason) {
#ifdef __ANDROID__
    if (executor_lsx4_android_run_guest_callback6 != nullptr) {
        return executor_lsx4_android_run_guest_callback6(
            callback, arg0, arg1, arg2, arg3, arg4, arg5, arg_count, result_out,
            reason);
    }
    if (executor_lsx4_android_run_guest_callback4 != nullptr) {
        return executor_lsx4_android_run_guest_callback4(
            callback, arg0, arg1, arg2, arg3, arg_count, result_out, reason);
    }
#else
    (void)callback;
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg_count;
    (void)result_out;
    (void)reason;
#endif
    return 0;
}

inline int RunGuestCallback4(void* callback, const std::uint64_t arg0,
                             const std::uint64_t arg1, const std::uint64_t arg2,
                             const std::uint64_t arg3, const std::uint32_t arg_count,
                             std::uint64_t* result_out, const char* reason) {
#ifdef __ANDROID__
    if (executor_lsx4_android_run_guest_callback4 != nullptr) {
        return executor_lsx4_android_run_guest_callback4(
            callback, arg0, arg1, arg2, arg3, arg_count, result_out, reason);
    }
    if (executor_lsx4_android_run_guest_callback6 != nullptr) {
        if (arg_count > 4) {
            return -2;
        }
        return executor_lsx4_android_run_guest_callback6(
            callback, arg0, arg1, arg2, arg3, 0, 0, arg_count, result_out, reason);
    }
#else
    (void)callback;
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg_count;
    (void)result_out;
    (void)reason;
#endif
    return 0;
}

}
