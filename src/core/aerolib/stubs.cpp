// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "common/singleton.h"
#include "core/aerolib/aerolib.h"
#include "core/aerolib/stubs.h"
#include "core/file_sys/fs.h"
#include "core/libraries/kernel/file_system.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/libc_internal/libc_internal_io.h"
#include "core/memory.h"
#include "executor/dynamic_translation/hle_thunk_identity.h"
#include "video_core/page_manager.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cctype>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#if defined(_WIN32)
static inline void* dlopen(const char*, int) {
    return nullptr;
}
static inline void* dlsym(void*, const char*) {
    return nullptr;
}
static inline int dlclose(void*) {
    return 0;
}
static inline const char* dlerror() {
    return "dlfcn unavailable on win32";
}
#ifndef RTLD_NOW
#define RTLD_NOW 0
#endif
#ifndef RTLD_GLOBAL
#define RTLD_GLOBAL 0
#endif
#ifndef RTLD_LAZY
#define RTLD_LAZY 0
#endif
#else
#include <dlfcn.h>
#endif
#include <cmath>
#include <unordered_set>
#include <thread>
#include <utility>
#include <vector>
#ifdef __ANDROID__
#include <android/log.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#include <pthread.h>
#endif
#if defined(__ANDROID__) || defined(__linux__)
#include <malloc.h>
#endif

namespace Core::AeroLib {

#ifdef __ANDROID__
extern "C" int executor_live_get_current_hle_call_site(std::uint64_t* guest_return,
                                                       std::uint64_t* return_off,
                                                       std::uint64_t* arg0,
                                                       char* symbol,
                                                       std::size_t symbol_size,
                                                       char* module,
                                                       std::size_t module_size)
    __attribute__((weak));
extern "C" int executor_live_get_current_hle_guest_rsp(std::uint64_t* guest_rsp)
    __attribute__((weak));
extern "C" void executor_jit_record_current_thread_state(const char* reason)
    __attribute__((weak));
extern "C" void executor_jit_dump_thread_states(const char* reason) __attribute__((weak));
extern "C" bool ExecutorJitReadGuestBytes(std::uint64_t address, void* dst,
                                                 std::size_t size) __attribute__((weak));
extern "C" bool ExecutorJitWriteGuestBytes(std::uint64_t address, const void* src,
                                                  std::size_t size) __attribute__((weak));
extern "C" bool ExecutorJitIsReadableGuestRange(std::uint64_t address,
                                                       std::size_t size)
    __attribute__((weak));

static bool ExecutorReadGuestBytes(u64 ptr, void* out, u64 size);
#endif

#ifdef __ANDROID__
constexpr u32 MAX_STUBS = 8192;
#else
constexpr u32 MAX_STUBS = 2048;
#endif

u64 UnresolvedStub() {
    LOG_ERROR(Core, "Returning zero to {}", __builtin_return_address(0));
    return 0;
}

static u64 UnknownStub() {
    LOG_ERROR(Core, "Returning zero to {}", __builtin_return_address(0));
    return 0;
}

static const NidEntry* stub_nids[MAX_STUBS];
static std::string stub_nids_unknown[MAX_STUBS];
static u32 UsedStubEntries;

#ifdef __ANDROID__
static void* g_android_x64_zero_stub_slab = nullptr;
static void* g_android_x64_hle_sideband_slab = nullptr;
static constexpr std::size_t AndroidX64StubSize = 512;
static constexpr std::size_t AndroidX64HleSidebandSlotSize = 96;
static constexpr std::size_t AndroidX64HleSidebandBuckets = 64;
static constexpr std::size_t AndroidX64HleSafePoolSize = 256 * 1024 * 1024;
static constexpr u32 AndroidX64HleSafeSlotSize = 0x1000;
static constexpr u32 AndroidX64HleSafeSlotRsp = 0x800;
static constexpr u32 AndroidX64HleStackScratchSize = 0x1000;
static constexpr u32 ExecutorBox64HleBridgeSyscall = 0x5ad00001u;
static std::mutex g_android_x64_hle_names_mutex;
static std::unordered_map<u64, std::string> g_android_x64_hle_names_by_native;
static std::unordered_map<u64, std::string> g_android_x64_hle_names_by_stub;
static std::unordered_map<u64, u64> g_android_x64_hle_target_by_stub;
static std::unordered_set<u64> g_android_x64_hle_return_zero_by_stub;
static std::unordered_map<u64, u64> g_android_x64_hle_stub_by_native;
static std::unordered_map<u64, u64> g_android_x64_return_zero_stub_by_native;
static constexpr u32 AndroidX64FallbackStubSlot = MAX_STUBS - 1;
static u32 g_android_x64_native_hle_next_slot = AndroidX64FallbackStubSlot;
static std::mutex g_android_x64_low_stub_mutex;
static std::unordered_map<std::string, u64> g_android_x64_low_stub_by_nid;

bool ExecutorRuntimeFlagExists(const char* filename);

bool ExecutorJitActiveForStubGeneration() {
    using Fn = int (*)();
    static Fn fn = reinterpret_cast<Fn>(
        dlsym(RTLD_DEFAULT, "executor_lsx4_android_runtime_jit_active"));
    return fn != nullptr && fn() != 0;
}

using ExecutorFexRegisterGuestExecutableRangeFn = int (*)(std::uint64_t address,
                                                          std::uint64_t size,
                                                          std::uint32_t prot);

ExecutorFexRegisterGuestExecutableRangeFn ResolveFexExecRangeRegistrar() {
    void* sym = dlsym(RTLD_DEFAULT, "executor_fexcore_register_guest_executable_range");
    if (sym == nullptr) {
        void* handle = dlopen("libexecutor_fexcore_embed.so", RTLD_NOW | RTLD_NOLOAD);
        if (handle != nullptr) {
            sym = dlsym(handle, "executor_fexcore_register_guest_executable_range");
        }
    }
    return reinterpret_cast<ExecutorFexRegisterGuestExecutableRangeFn>(sym);
}

void RegisterAndroidX64StubSlabWithFex() {
    if (!g_android_x64_zero_stub_slab) {
        return;
    }
    static void* s_registered_base = nullptr;
    if (s_registered_base == g_android_x64_zero_stub_slab) {
        return;
    }
    static ExecutorFexRegisterGuestExecutableRangeFn registrar = ResolveFexExecRangeRegistrar();
    if (registrar == nullptr) {
        return;
    }
    const std::uint64_t base = reinterpret_cast<std::uint64_t>(g_android_x64_zero_stub_slab);
    const std::uint64_t size = MAX_STUBS * AndroidX64StubSize;
    constexpr std::uint32_t kPfReadExec = 0x5u;
    const int rc = registrar(base, size, kPfReadExec);
    s_registered_base = g_android_x64_zero_stub_slab;
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_X64_STUB_FEX_RANGE] base=0x%llx size=0x%llx rc=%d",
                        static_cast<unsigned long long>(base),
                        static_cast<unsigned long long>(size), rc);
}

void* EnsureAndroidX64StubSlab() {
    if (!g_android_x64_zero_stub_slab) {
        const std::size_t slab_size = MAX_STUBS * AndroidX64StubSize;
        g_android_x64_zero_stub_slab = mmap(nullptr, slab_size, PROT_READ | PROT_WRITE | PROT_EXEC,
                                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (g_android_x64_zero_stub_slab == MAP_FAILED) {
            g_android_x64_zero_stub_slab = nullptr;
            return nullptr;
        }
        auto* code = static_cast<u8*>(g_android_x64_zero_stub_slab);
        for (u32 i = 0; i < MAX_STUBS; ++i) {
            auto* stub = code + i * AndroidX64StubSize;
            stub[0] = 0x48;
            stub[1] = 0x31;
            stub[2] = 0xc0;
            stub[3] = 0xc3;
            for (std::size_t j = 4; j < AndroidX64StubSize; ++j) {
                stub[j] = 0xcc;
            }
        }
        __builtin___clear_cache(reinterpret_cast<char*>(code),
                                reinterpret_cast<char*>(code + slab_size));
    }
    if (!ExecutorJitActiveForStubGeneration()) {
        RegisterAndroidX64StubSlabWithFex();
    }
    return g_android_x64_zero_stub_slab;
}

u64 GetAndroidX64HleSidebandSlot(u32 slot) {
    if (slot >= MAX_STUBS) {
        slot = MAX_STUBS - 1;
    }
    if (!g_android_x64_hle_sideband_slab) {
        const std::size_t slab_size =
            MAX_STUBS * AndroidX64HleSidebandBuckets * AndroidX64HleSidebandSlotSize;
        g_android_x64_hle_sideband_slab = mmap(nullptr, slab_size, PROT_READ | PROT_WRITE,
                                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (g_android_x64_hle_sideband_slab == MAP_FAILED) {
            g_android_x64_hle_sideband_slab = nullptr;
            return 0;
        }
        std::memset(g_android_x64_hle_sideband_slab, 0, slab_size);
    }
    auto* slot_ptr = static_cast<u8*>(g_android_x64_hle_sideband_slab) +
                     slot * AndroidX64HleSidebandBuckets * AndroidX64HleSidebandSlotSize;
    return reinterpret_cast<u64>(slot_ptr);
}

u8* GetAndroidX64StubSlot(u32 slot) {
    if (slot >= MAX_STUBS) {
        slot = MAX_STUBS - 1;
    }
    auto* slab = static_cast<u8*>(EnsureAndroidX64StubSlab());
    return slab ? slab + slot * AndroidX64StubSize : nullptr;
}

u64 GetAndroidX64ZeroStub(u32 slot) {
    auto* stub = GetAndroidX64StubSlot(slot);
    return reinterpret_cast<u64>(stub);
}

u64 GetAndroidX64ReturnZeroStub(u32 slot) {
    auto* code = GetAndroidX64StubSlot(slot);
    if (!code) {
        return 0;
    }
    std::size_t cursor = 0;
    code[cursor++] = 0x31;
    code[cursor++] = 0xc0;
    code[cursor++] = 0xc3;
    while (cursor < AndroidX64StubSize) {
        code[cursor++] = 0xcc;
    }
    __builtin___clear_cache(reinterpret_cast<char*>(code),
                            reinterpret_cast<char*>(code + AndroidX64StubSize));
    return reinterpret_cast<u64>(code);
}

bool ShouldAndroidX64ReturnZeroDirectly(const char* name) {
    const char* flag = std::getenv("EXECUTOR_FEX_DIRECT_RETURN_USLEEP");
    if (!flag || flag[0] == '\0' || std::strcmp(flag, "0") == 0) {
        return false;
    }
    const std::string_view n = name ? name : "";
    return n.find("sceKernelUsleep") != std::string_view::npos ||
           n.find("posix_usleep") != std::string_view::npos ||
           n.find("1jfXLRVzisc") != std::string_view::npos ||
           n.find("QcteRwbsnV0") != std::string_view::npos;
}

u64 GetAndroidX64HleStub(u32 slot, u64 native_function) {
    auto* code = GetAndroidX64StubSlot(slot);
    if (!code) {
        return 0;
    }

    static u64 s_hle_pool_base = 0;
    static u64 s_hle_pool_cursor = 0;
    if (s_hle_pool_base == 0) {
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
        auto alloc_safe = [](std::size_t sz) -> void* {
            for (u64 base = 0x400000000ULL; base < 0x5c0000000ULL; base += 0x40000000ULL) {
                void* m = mmap(reinterpret_cast<void*>(base), sz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                                   MAP_FIXED_NOREPLACE,
                               -1, 0);
                if (m == reinterpret_cast<void*>(base)) return m;
                if (m != MAP_FAILED) munmap(m, sz);
            }
            for (int a = 0; a < 8; ++a) {
                void* m = mmap(nullptr, sz, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
                if (m == MAP_FAILED) break;
                const u64 v = reinterpret_cast<u64>(m);
                if (v < 0x200000000ULL || v >= 0x300000000ULL) return m;
                munmap(m, sz);
            }
            return MAP_FAILED;
        };
        void* pool = alloc_safe(AndroidX64HleSafePoolSize);
        void* cur = alloc_safe(4096);
        if (pool == MAP_FAILED || cur == MAP_FAILED) {
            return 0;
        }
        s_hle_pool_base = reinterpret_cast<u64>(pool);
        s_hle_pool_cursor = reinterpret_cast<u64>(cur);
        *reinterpret_cast<volatile u64*>(cur) = 0;
        std::fprintf(stderr, "[EXECUTOR_HLE_STUB_SAFE_POOL] base=0x%llx cursor=0x%llx\n",
                     static_cast<unsigned long long>(s_hle_pool_base),
                     static_cast<unsigned long long>(s_hle_pool_cursor));
    }

    std::size_t cursor = 0;
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0xe0;
    code[cursor++] = 0x48;
    code[cursor++] = 0xc1;
    code[cursor++] = 0xe8;
    code[cursor++] = 0x20;
    code[cursor++] = 0x85;
    code[cursor++] = 0xc0;
    code[cursor++] = 0x0f;
    code[cursor++] = 0x84;
    const std::size_t je_stack_zero_rel_pos = cursor;
    {
        std::int32_t rel = 0;
        std::memcpy(code + cursor, &rel, sizeof(rel));
        cursor += sizeof(rel);
    }
    code[cursor++] = 0x83;
    code[cursor++] = 0xf8;
    code[cursor++] = 0x07;
    code[cursor++] = 0x0f;
    code[cursor++] = 0x84;
    const std::size_t je_stack_7_rel_pos = cursor;
    {
        std::int32_t rel = 0;
        std::memcpy(code + cursor, &rel, sizeof(rel));
        cursor += sizeof(rel);
    }
    code[cursor++] = 0xe9;
    const std::size_t jmp_pool_rel_pos = cursor;
    {
        std::int32_t rel = 0;
        std::memcpy(code + cursor, &rel, sizeof(rel));
        cursor += sizeof(rel);
    }

    const std::size_t stack_path = cursor;
    {
        const std::int32_t rel = static_cast<std::int32_t>(
            reinterpret_cast<u64>(code + stack_path) -
            reinterpret_cast<u64>(code + je_stack_zero_rel_pos + sizeof(std::int32_t)));
        std::memcpy(code + je_stack_zero_rel_pos, &rel, sizeof(rel));
    }
    {
        const std::int32_t rel = static_cast<std::int32_t>(
            reinterpret_cast<u64>(code + stack_path) -
            reinterpret_cast<u64>(code + je_stack_7_rel_pos + sizeof(std::int32_t)));
        std::memcpy(code + je_stack_7_rel_pos, &rel, sizeof(rel));
    }
    code[cursor++] = 0x48;
    code[cursor++] = 0x81;
    code[cursor++] = 0xec;
    {
        const u32 stack_slot = AndroidX64HleStackScratchSize;
        std::memcpy(code + cursor, &stack_slot, sizeof(stack_slot));
        cursor += sizeof(stack_slot);
    }
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0xe0;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x8d;
    code[cursor++] = 0x94;
    code[cursor++] = 0x24;
    {
        const u32 stack_slot = AndroidX64HleStackScratchSize;
        std::memcpy(code + cursor, &stack_slot, sizeof(stack_slot));
        cursor += sizeof(stack_slot);
    }
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x50;
    code[cursor++] = 0x40;
    code[cursor++] = 0xe9;
    const std::size_t jmp_ready_rel_pos = cursor;
    {
        std::int32_t rel = 0;
        std::memcpy(code + cursor, &rel, sizeof(rel));
        cursor += sizeof(rel);
    }

    const std::size_t pool_path = cursor;
    {
        const std::int32_t rel = static_cast<std::int32_t>(
            reinterpret_cast<u64>(code + pool_path) -
            reinterpret_cast<u64>(code + jmp_pool_rel_pos + sizeof(std::int32_t)));
        std::memcpy(code + jmp_pool_rel_pos, &rel, sizeof(rel));
    }
    code[cursor++] = 0x48;
    code[cursor++] = 0xb8;
    std::memcpy(code + cursor, &s_hle_pool_cursor, sizeof(s_hle_pool_cursor));
    cursor += sizeof(s_hle_pool_cursor);
    code[cursor++] = 0x41;
    code[cursor++] = 0xba;
    {
        const u32 slot_sz = AndroidX64HleSafeSlotSize;
        std::memcpy(code + cursor, &slot_sz, sizeof(slot_sz));
        cursor += sizeof(slot_sz);
    }
    code[cursor++] = 0xf0;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x0f;
    code[cursor++] = 0xc1;
    code[cursor++] = 0x10;
    code[cursor++] = 0x49;
    code[cursor++] = 0x81;
    code[cursor++] = 0xe2;
    {
        const u32 mask = static_cast<u32>(AndroidX64HleSafePoolSize - 1);
        std::memcpy(code + cursor, &mask, sizeof(mask));
        cursor += sizeof(mask);
    }
    code[cursor++] = 0x48;
    code[cursor++] = 0xb8;
    std::memcpy(code + cursor, &s_hle_pool_base, sizeof(s_hle_pool_base));
    cursor += sizeof(s_hle_pool_base);
    code[cursor++] = 0x4c;
    code[cursor++] = 0x01;
    code[cursor++] = 0xd0;
    code[cursor++] = 0x48;
    code[cursor++] = 0x05;
    {
        const u32 rsp_off = AndroidX64HleSafeSlotRsp;
        std::memcpy(code + cursor, &rsp_off, sizeof(rsp_off));
        cursor += sizeof(rsp_off);
    }
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0x60;
    code[cursor++] = 0x40;

    const std::size_t sideband_ready = cursor;
    {
        const std::int32_t rel = static_cast<std::int32_t>(
            reinterpret_cast<u64>(code + sideband_ready) -
            reinterpret_cast<u64>(code + jmp_ready_rel_pos + sizeof(std::int32_t)));
        std::memcpy(code + jmp_ready_rel_pos, &rel, sizeof(rel));
    }
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x58;
    code[cursor++] = 0x48;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x50;
    code[cursor++] = 0x40;
    code[cursor++] = 0x4d;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x12;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x90;
    {
        const u32 ret_off = 0x80;
        std::memcpy(code + cursor, &ret_off, sizeof(ret_off));
        cursor += sizeof(ret_off);
    }
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0x58;
    code[cursor++] = 0x50;
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0x68;
    code[cursor++] = 0x58;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x60;
    code[cursor++] = 0x60;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x68;
    code[cursor++] = 0x68;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x70;
    code[cursor++] = 0x70;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x78;
    code[cursor++] = 0x78;
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0xc4;
    code[cursor++] = 0x49;
    code[cursor++] = 0xbb;
    std::memcpy(code + cursor, &native_function, sizeof(native_function));
    cursor += sizeof(native_function);
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x58;
    code[cursor++] = 0x08;
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0x78;
    code[cursor++] = 0x10;
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0x70;
    code[cursor++] = 0x18;
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0x50;
    code[cursor++] = 0x20;
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0x48;
    code[cursor++] = 0x28;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x40;
    code[cursor++] = 0x30;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0x48;
    code[cursor++] = 0x38;
    code[cursor++] = 0x48;
    code[cursor++] = 0x89;
    code[cursor++] = 0xc7;
    const std::size_t call_pos = cursor;
    code[cursor++] = 0xe8;
    const std::size_t rel_pos = cursor;
    std::int32_t rel32 = 0;
    std::memcpy(code + cursor, &rel32, sizeof(rel32));
    cursor += sizeof(rel32);
    code[cursor++] = 0x48;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x04;
    code[cursor++] = 0x24;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x5c;
    code[cursor++] = 0x24;
    code[cursor++] = 0x48;
    code[cursor++] = 0x48;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x5c;
    code[cursor++] = 0x24;
    code[cursor++] = 0x50;
    code[cursor++] = 0x48;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x6c;
    code[cursor++] = 0x24;
    code[cursor++] = 0x58;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x64;
    code[cursor++] = 0x24;
    code[cursor++] = 0x60;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x6c;
    code[cursor++] = 0x24;
    code[cursor++] = 0x68;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x74;
    code[cursor++] = 0x24;
    code[cursor++] = 0x70;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x7c;
    code[cursor++] = 0x24;
    code[cursor++] = 0x78;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x54;
    code[cursor++] = 0x24;
    code[cursor++] = 0x40;
    code[cursor++] = 0x48;
    code[cursor++] = 0x8b;
    code[cursor++] = 0x8c;
    code[cursor++] = 0x24;
    {
        const u32 ret_off = 0x80;
        std::memcpy(code + cursor, &ret_off, sizeof(ret_off));
        cursor += sizeof(ret_off);
    }
    code[cursor++] = 0x49;
    code[cursor++] = 0x39;
    code[cursor++] = 0x0a;
    code[cursor++] = 0x74;
    code[cursor++] = 0x03;
    code[cursor++] = 0x49;
    code[cursor++] = 0x89;
    code[cursor++] = 0x0a;
    code[cursor++] = 0x4c;
    code[cursor++] = 0x89;
    code[cursor++] = 0xd4;
    code[cursor++] = 0xc3;
    const u64 helper = reinterpret_cast<u64>(code) + cursor;
    rel32 = static_cast<std::int32_t>(helper - (reinterpret_cast<u64>(code) + call_pos + 5));
    std::memcpy(code + rel_pos, &rel32, sizeof(rel32));
    code[cursor++] = 0x0f;
    code[cursor++] = 0x3f;
    const auto& hle_hash =
        ExecutorJitActiveForStubGeneration()
            ? Lsx4::Translation::TranslationThunkIdentity
            : Lsx4::Translation::FexThunkIdentity;
    std::memcpy(code + cursor, hle_hash.data(), hle_hash.size());
    cursor += hle_hash.size();
    while (cursor < AndroidX64StubSize) {
        code[cursor++] = 0xcc;
    }
    __builtin___clear_cache(reinterpret_cast<char*>(code),
                            reinterpret_cast<char*>(code + AndroidX64StubSize));
    return reinterpret_cast<u64>(code);
}

bool AndroidHasLowStubSlot() {
    return UsedStubEntries < g_android_x64_native_hle_next_slot;
}

u64 GetAndroidX64NativeHleStub(const char* name, u64 native_function, const char* tag) {
    if (!native_function) {
        return GetAndroidX64ZeroStub(AndroidX64FallbackStubSlot);
    }
    std::scoped_lock lock(g_android_x64_hle_names_mutex);
    g_android_x64_hle_names_by_native[native_function] = name ? name : "";
    const bool direct_return_zero = ShouldAndroidX64ReturnZeroDirectly(name);
    auto& cache = direct_return_zero ? g_android_x64_return_zero_stub_by_native
                                     : g_android_x64_hle_stub_by_native;
    if (const auto cached = cache.find(native_function); cached != cache.end()) {
        if (name != nullptr && g_android_x64_hle_names_by_stub[cached->second].empty()) {
            g_android_x64_hle_names_by_stub[cached->second] = name;
        }
        return cached->second;
    }
    if (g_android_x64_native_hle_next_slot <= UsedStubEntries) {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_NATIVE_HLE_EXHAUSTED] tag=%s name=%s low=%u high=%u\n",
                     tag ? tag : "?", name ? name : "<unnamed>", UsedStubEntries,
                     g_android_x64_native_hle_next_slot);
        std::fflush(stderr);
        return GetAndroidX64ZeroStub(AndroidX64FallbackStubSlot);
    }
    const u32 slot = --g_android_x64_native_hle_next_slot;
    stub_nids[slot] = nullptr;
    stub_nids_unknown[slot] = name ? name : "<native-hle>";
    const u64 address = direct_return_zero ? GetAndroidX64ReturnZeroStub(slot)
                                           : GetAndroidX64HleStub(slot, native_function);
    if (address != 0) {
        cache.emplace(native_function, address);
        g_android_x64_hle_names_by_stub[address] = name ? name : "";
        if (direct_return_zero) {
            g_android_x64_hle_return_zero_by_stub.insert(address);
        } else {
            g_android_x64_hle_target_by_stub[address] = native_function;
        }
    }
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_NATIVE_HLE] tag=%s index=%u name=%s address=%p target=%p "
                 "directReturnZero=%d low=%u high_next=%u\n",
                 tag ? tag : "?", slot, name ? name : "<unnamed>",
                 reinterpret_cast<void*>(address), reinterpret_cast<void*>(native_function),
                 direct_return_zero ? 1 : 0, UsedStubEntries, g_android_x64_native_hle_next_slot);
    std::fflush(stderr);
    return address;
}

u64 EnsureAndroidX64ZeroStubSlab() {
    return reinterpret_cast<u64>(EnsureAndroidX64StubSlab());
}

u64 GetAndroidX64ZeroStubSlabBase() {
    return reinterpret_cast<u64>(g_android_x64_zero_stub_slab);
}

u64 GetAndroidX64ZeroStubSlabSize() {
    return g_android_x64_zero_stub_slab ? MAX_STUBS * AndroidX64StubSize : 0;
}

std::string GetAndroidX64HleNameForNative(u64 native_function) {
    std::scoped_lock lock(g_android_x64_hle_names_mutex);
    const auto it = g_android_x64_hle_names_by_native.find(native_function);
    return it != g_android_x64_hle_names_by_native.end() ? it->second : std::string{};
}

bool IsAndroidX64NativeHleStub(u64 address, std::string* name) {
    std::scoped_lock lock(g_android_x64_hle_names_mutex);
    const auto it = g_android_x64_hle_names_by_stub.find(address);
    if (it == g_android_x64_hle_names_by_stub.end()) {
        return false;
    }
    if (name != nullptr) {
        *name = it->second;
    }
    return true;
}

bool TryGetAndroidX64NativeHleTarget(u64 address, u64* native_function) {
    std::scoped_lock lock(g_android_x64_hle_names_mutex);
    const auto it = g_android_x64_hle_target_by_stub.find(address);
    if (it == g_android_x64_hle_target_by_stub.end() || it->second == 0) {
        return false;
    }
    if (native_function != nullptr) {
        *native_function = it->second;
    }
    return true;
}

bool IsAndroidX64NativeHleReturnZeroStub(u64 address) {
    std::scoped_lock lock(g_android_x64_hle_names_mutex);
    return g_android_x64_hle_return_zero_by_stub.find(address) !=
           g_android_x64_hle_return_zero_by_stub.end();
}

bool ExecutorUseFastGuestLibcStrings() {
    static const bool enabled = [] {
        const char* env = std::getenv("EXECUTOR_FAST_GUEST_LIBC_STRINGS");
        if (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) {
            return true;
        }
        return ExecutorRuntimeFlagExists("executor-fast-guest-libc-strings.flag") ||
               ExecutorRuntimeFlagExists("lsx4-home/run-live-fast-guest-libc-strings");
    }();
    return enabled;
}

bool IsFastGuestLibcStringName(const char* name) {
    if (name == nullptr) {
        return false;
    }
    return std::strcmp(name, "strlen") == 0 || std::strcmp(name, "memcmp") == 0 ||
           std::strcmp(name, "strcmp") == 0 || std::strcmp(name, "strncmp") == 0;
}

u64 GetAndroidX64FastGuestLibcStringStub(const char* name) {
    if (!ExecutorUseFastGuestLibcStrings() || !IsFastGuestLibcStringName(name)) {
        return 0;
    }
    if (ExecutorJitActiveForStubGeneration()) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_FAST_LIBC_STRING_STUB] disabled=1 "
                                "backend=jit route=native-checked-hle");
        }
        return 0;
    }
    std::scoped_lock lock(g_android_x64_hle_names_mutex);
    if (g_android_x64_native_hle_next_slot <= UsedStubEntries) {
        std::fprintf(stderr,
                     "[EXECUTOR_FAST_LIBC_STRING_STUB_EXHAUSTED] name=%s low=%u high=%u\n",
                     name ? name : "<null>", UsedStubEntries, g_android_x64_native_hle_next_slot);
        std::fflush(stderr);
        return 0;
    }
    const u32 slot = --g_android_x64_native_hle_next_slot;
    stub_nids[slot] = nullptr;
    stub_nids_unknown[slot] = name ? name : "<fast-libc-string>";
    u8* code = GetAndroidX64StubSlot(slot);
    if (!code) {
        return 0;
    }

    std::size_t c = 0;
    auto emit = [&](u8 b) { code[c++] = b; };
    auto emit_rel8 = [&](std::size_t at, std::size_t target) {
        code[at] = static_cast<u8>(static_cast<std::int8_t>(target - (at + 1)));
    };

    if (std::strcmp(name, "strlen") == 0) {
        emit(0x31); emit(0xc0);
        const std::size_t loop = c;
        emit(0x80); emit(0x3c); emit(0x07); emit(0x00);
        emit(0x74); const std::size_t je_done = c++;
        emit(0x48); emit(0xff); emit(0xc0);
        emit(0xeb); const std::size_t j_loop = c++;
        const std::size_t done = c;
        emit(0xc3);
        emit_rel8(je_done, done);
        emit_rel8(j_loop, loop);
    } else if (std::strcmp(name, "strcmp") == 0) {
        const std::size_t loop = c;
        emit(0x0f); emit(0xb6); emit(0x07);
        emit(0x0f); emit(0xb6); emit(0x16);
        emit(0x38); emit(0xd0);
        emit(0x75); const std::size_t jne_diff = c++;
        emit(0x84); emit(0xc0);
        emit(0x74); const std::size_t je_eq = c++;
        emit(0x48); emit(0xff); emit(0xc7);
        emit(0x48); emit(0xff); emit(0xc6);
        emit(0xeb); const std::size_t j_loop = c++;
        const std::size_t diff = c;
        emit(0x29); emit(0xd0);
        emit(0xc3);
        const std::size_t eq = c;
        emit(0x31); emit(0xc0);
        emit(0xc3);
        emit_rel8(jne_diff, diff);
        emit_rel8(je_eq, eq);
        emit_rel8(j_loop, loop);
    } else if (std::strcmp(name, "strncmp") == 0 || std::strcmp(name, "memcmp") == 0) {
        const bool is_strncmp = std::strcmp(name, "strncmp") == 0;
        emit(0x48); emit(0x85); emit(0xd2);
        emit(0x74); const std::size_t je_eq0 = c++;
        const std::size_t loop = c;
        emit(0x0f); emit(0xb6); emit(0x07);
        emit(0x0f); emit(0xb6); emit(0x0e);
        emit(0x38); emit(0xc8);
        emit(0x75); const std::size_t jne_diff = c++;
        if (is_strncmp) {
            emit(0x84); emit(0xc0);
            emit(0x74); const std::size_t je_eqz = c++;
            emit(0x48); emit(0xff); emit(0xc7);
            emit(0x48); emit(0xff); emit(0xc6);
            emit(0x48); emit(0xff); emit(0xca);
            emit(0x75); const std::size_t jne_loop = c++;
            const std::size_t eq = c;
            emit(0x31); emit(0xc0);
            emit(0xc3);
            const std::size_t diff = c;
            emit(0x29); emit(0xc8);
            emit(0xc3);
            emit_rel8(je_eq0, eq);
            emit_rel8(jne_diff, diff);
            emit_rel8(je_eqz, eq);
            emit_rel8(jne_loop, loop);
        } else {
            emit(0x48); emit(0xff); emit(0xc7);
            emit(0x48); emit(0xff); emit(0xc6);
            emit(0x48); emit(0xff); emit(0xca);
            emit(0x75); const std::size_t jne_loop = c++;
            const std::size_t eq = c;
            emit(0x31); emit(0xc0);
            emit(0xc3);
            const std::size_t diff = c;
            emit(0x29); emit(0xc8);
            emit(0xc3);
            emit_rel8(je_eq0, eq);
            emit_rel8(jne_diff, diff);
            emit_rel8(jne_loop, loop);
        }
    } else {
        return 0;
    }

    while (c < AndroidX64StubSize) {
        code[c++] = 0xcc;
    }
    __builtin___clear_cache(reinterpret_cast<char*>(code),
                            reinterpret_cast<char*>(code + AndroidX64StubSize));
    const u64 address = reinterpret_cast<u64>(code);
    g_android_x64_hle_names_by_stub[address] = name ? name : "";
    std::fprintf(stderr,
                 "[EXECUTOR_FAST_LIBC_STRING_STUB] index=%u name=%s address=%p low=%u high_next=%u\n",
                 slot, name ? name : "<null>", reinterpret_cast<void*>(address), UsedStubEntries,
                 g_android_x64_native_hle_next_slot);
    std::fflush(stderr);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_FAST_LIBC_STRING_STUB] index=%u name=%s address=0x%llx",
                        slot, name ? name : "<null>",
                        static_cast<unsigned long long>(address));
    return address;
}
#endif

constexpr u64 ExecutorDefaultMspaceHandle = 0x5eec00000001ULL;
constexpr u64 ExecutorMspaceHandleBase = 0x5eec00001000ULL;
constexpr u64 ExecutorLazyLibcMspaceBase = Core::EXECUTOR_LIBC_MSPACE_BASE;
constexpr u64 ExecutorLazyLibcMspaceSize = Core::EXECUTOR_LIBC_MSPACE_INITIAL_SIZE;
constexpr std::size_t ExecutorAllocRedzoneSize = 64;
constexpr std::size_t ExecutorAllocGuardPageDefaultThreshold = static_cast<std::size_t>(-1);
constexpr std::uint8_t ExecutorAllocPrefixGuard = 0xa5;
constexpr std::uint8_t ExecutorAllocSuffixGuard = 0x5a;
std::mutex g_executor_alloc_mutex;
struct ExecutorAllocationInfo {
    void* base{};
    std::size_t mapping_size{};
    std::size_t requested_size{};
    std::size_t allocated_size{};
    std::uint64_t mspace_handle{};
    std::uint64_t mspace_chunk_begin{};
    std::size_t mspace_chunk_size{};
    std::uint64_t serial{};
    bool fios_cache_pinned{};
};
std::unordered_map<void*, ExecutorAllocationInfo> g_executor_allocations;
std::unordered_map<u64, std::map<u64, u64>> g_executor_live_mspace_intervals;
std::atomic<int> g_executor_alloc_small_log_budget{256};
std::atomic<int> g_executor_alloc_large_log_budget{512};
std::atomic<int> g_executor_alloc_png_log_budget{512};
std::atomic<std::uint64_t> g_executor_alloc_serial{1};
std::atomic<int> g_executor_alloc_guard_violation_budget{16};
std::atomic<int> g_executor_alloc_guard_free_log_budget{128};
struct ExecutorMspaceState {
    u64 base{};
    u64 size{};
    u64 offset{};
    u64 alloc_start{};
    u64 top_chunk{};
    u64 top_size{};
    std::map<u64, u64> free_blocks{};
};
std::mutex g_executor_mspace_mutex;
std::unordered_map<u64, ExecutorMspaceState> g_executor_mspaces;
u64 g_executor_next_mspace_handle = ExecutorMspaceHandleBase;
u64 g_executor_default_mspace_handle = 0;

constexpr u64 kExecutorSonyMspaceAlignment = 0x20ULL;
constexpr u64 kExecutorSonyMspaceChunkOverhead = 0x10ULL;
constexpr u64 kExecutorSonyMspaceMinChunkSize = 0x20ULL;
constexpr u64 kExecutorSonyMspaceStateChunkSize = 0x520ULL;
constexpr u64 kExecutorSonyMspaceTopFootSize = 0x60ULL;
constexpr u64 kExecutorGuestMspaceMagic = 0x32505342534D5845ULL;

bool ExecutorAlignUpSonyMspace(u64 value, u64 alignment, u64* out) {
    if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        value > std::numeric_limits<u64>::max() - (alignment - 1)) {
        return false;
    }
    *out = (value + alignment - 1) & ~(alignment - 1);
    return true;
}

bool ExecutorSonyMspaceRequestToChunkSize(u64 request, u64* out) {
    if (out == nullptr) {
        return false;
    }
    if (request < kExecutorSonyMspaceChunkOverhead) {
        *out = kExecutorSonyMspaceMinChunkSize;
        return true;
    }
    constexpr u64 Pad = kExecutorSonyMspaceChunkOverhead +
                        (kExecutorSonyMspaceAlignment - 1);
    if (request > std::numeric_limits<u64>::max() - Pad) {
        return false;
    }
    *out = (request + Pad) & ~(kExecutorSonyMspaceAlignment - 1);
    return *out >= kExecutorSonyMspaceMinChunkSize;
}

bool ExecutorPlaceSonyMspaceChunkInFreeBlock(u64 block_begin, u64 block_size, u64 alignment,
                                             u64 requested_chunk_size, u64* chunk_begin,
                                             u64* chunk_size, u64* prefix_size,
                                             u64* suffix_size) {
    if (chunk_begin == nullptr || chunk_size == nullptr || prefix_size == nullptr ||
        suffix_size == nullptr || block_size < requested_chunk_size ||
        block_begin > std::numeric_limits<u64>::max() - kExecutorSonyMspaceChunkOverhead) {
        return false;
    }
    alignment = std::max(alignment, kExecutorSonyMspaceAlignment);
    u64 user = 0;
    if (!ExecutorAlignUpSonyMspace(block_begin + kExecutorSonyMspaceChunkOverhead, alignment,
                                   &user) ||
        user < kExecutorSonyMspaceChunkOverhead) {
        return false;
    }
    u64 candidate = user - kExecutorSonyMspaceChunkOverhead;
    u64 prefix = candidate - block_begin;
    if (prefix != 0 && prefix < kExecutorSonyMspaceMinChunkSize) {
        if (user > std::numeric_limits<u64>::max() - alignment) {
            return false;
        }
        user += alignment;
        candidate = user - kExecutorSonyMspaceChunkOverhead;
        prefix = candidate - block_begin;
    }
    if (prefix > block_size || requested_chunk_size > block_size - prefix) {
        return false;
    }
    u64 used = requested_chunk_size;
    u64 suffix = block_size - prefix - used;
    if (suffix != 0 && suffix < kExecutorSonyMspaceMinChunkSize) {
        used += suffix;
        suffix = 0;
    }
    *chunk_begin = candidate;
    *chunk_size = used;
    *prefix_size = prefix;
    *suffix_size = suffix;
    return true;
}

bool ExecutorComputeSonyMspaceGeometry(u64 base, u64 size, u64* handle, u64* top_chunk,
                                       u64* top_size) {
    if (handle == nullptr || top_chunk == nullptr || top_size == nullptr ||
        base > std::numeric_limits<u64>::max() - size ||
        base > std::numeric_limits<u64>::max() - kExecutorSonyMspaceChunkOverhead) {
        return false;
    }
    u64 state_user = 0;
    if (!ExecutorAlignUpSonyMspace(base + kExecutorSonyMspaceChunkOverhead,
                                   kExecutorSonyMspaceAlignment, &state_user) ||
        state_user < kExecutorSonyMspaceChunkOverhead) {
        return false;
    }
    const u64 state_chunk = state_user - kExecutorSonyMspaceChunkOverhead;
    if (state_chunk > std::numeric_limits<u64>::max() - kExecutorSonyMspaceStateChunkSize ||
        state_chunk + kExecutorSonyMspaceStateChunkSize >
            std::numeric_limits<u64>::max() - kExecutorSonyMspaceChunkOverhead) {
        return false;
    }
    u64 first_user = 0;
    if (!ExecutorAlignUpSonyMspace(state_chunk + kExecutorSonyMspaceStateChunkSize +
                                       kExecutorSonyMspaceChunkOverhead,
                                   kExecutorSonyMspaceAlignment, &first_user) ||
        first_user < kExecutorSonyMspaceChunkOverhead) {
        return false;
    }
    const u64 first_top = first_user - kExecutorSonyMspaceChunkOverhead;
    const u64 end = base + size;
    if (first_top >= end || end - first_top <= kExecutorSonyMspaceTopFootSize) {
        return false;
    }
    const u64 first_top_size = end - first_top - kExecutorSonyMspaceTopFootSize;
    if (first_top_size < kExecutorSonyMspaceMinChunkSize) {
        return false;
    }
    *handle = state_user;
    *top_chunk = first_top;
    *top_size = first_top_size;
    return true;
}
#ifdef __ANDROID__
constexpr std::size_t ExecutorArenaMappingSentinel = static_cast<std::size_t>(-1);
constexpr std::size_t ExecutorAndroidGuestHeapArenaSize = 256 * 1024 * 1024;
constexpr std::uint8_t ExecutorGuestMallocUninitializedFill = 0xAF;
bool ExecutorShouldPoisonGuestMalloc() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ExecutorRuntimeFlagExists("run-poison-guest-malloc") ? 1 : 0;
    }
    return enabled == 1;
}

bool ExecutorMspaceFreeReuseEnabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const bool explicitly_disabled =
            ExecutorRuntimeFlagExists("run-disable-mspace-free-reuse");
        enabled = explicitly_disabled ? 0 : 1;
    }
    return enabled == 1;
}

bool ExecutorMemsetVtableGuardEnabled() {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = ExecutorRuntimeFlagExists("lsx4-home/run-enable-memset-vtable-guard") ? 1 : 0;
    }
    return enabled == 1;
}
std::mutex g_executor_arena_mutex;
std::uint8_t* g_executor_arena_base = nullptr;
std::size_t g_executor_arena_size = 0;


extern "C" std::uint64_t executor_aot_got_install(std::uint64_t amodule);
#ifdef __ANDROID__
void ExecutorStartVtZeroPoller();
#endif

u64 ExecutorMakeGuestMspaceLocked(u64 base, u64 size, u64 flags) {
    u64 handle = 0;
    u64 top_chunk = 0;
    u64 top_size = 0;
    if (!ExecutorComputeSonyMspaceGeometry(base, size, &handle, &top_chunk, &top_size)) {
        return 0;
    }
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || !memory->IsWritableMapping(base, size)) {
        return 0;
    }

    std::memset(reinterpret_cast<void*>(handle), 0, kExecutorSonyMspaceStateChunkSize);
    auto* hdr = reinterpret_cast<u64*>(handle);
    hdr[0] = kExecutorGuestMspaceMagic;
    hdr[1] = 1;
    hdr[2] = base;
    hdr[3] = size;
    hdr[4] = top_chunk + kExecutorSonyMspaceChunkOverhead;
    hdr[5] = top_chunk - base;
    hdr[6] = flags;

    const u64 state_chunk = handle - kExecutorSonyMspaceChunkOverhead;
    *reinterpret_cast<u64*>(state_chunk + sizeof(u64)) =
        kExecutorSonyMspaceStateChunkSize | 0x3ULL;
    *reinterpret_cast<u64*>(top_chunk + sizeof(u64)) = top_size | 0x1ULL;
    *reinterpret_cast<u64*>(base + size - (kExecutorSonyMspaceTopFootSize - sizeof(u64))) =
        kExecutorSonyMspaceTopFootSize;

    g_executor_mspaces[handle] = ExecutorMspaceState{
        .base = base,
        .size = size,
        .offset = top_chunk - base,
        .alloc_start = top_chunk + kExecutorSonyMspaceChunkOverhead,
        .top_chunk = top_chunk,
        .top_size = top_size,
    };
    return handle;
}

u64 ExecutorEnsureDefaultGuestMspaceLocked() {
    if (g_executor_default_mspace_handle != 0) {
        return g_executor_default_mspace_handle;
    }
    const u64 handle = ExecutorMakeGuestMspaceLocked(ExecutorLazyLibcMspaceBase,
                                                     ExecutorLazyLibcMspaceSize, 0);
    if (handle == 0) {
        return 0;
    }
    g_executor_default_mspace_handle = handle;
#ifdef __ANDROID__
    ExecutorStartVtZeroPoller();
#endif
    return g_executor_default_mspace_handle;
}

u64 ExecutorFindMspaceByBaseLocked(u64 base) {
    for (const auto& [handle, state] : g_executor_mspaces) {
        if (state.base == base) {
            return handle;
        }
    }
    return 0;
}
std::size_t g_executor_arena_offset = 0;

std::atomic<int> g_vt_poll_started{0};
bool g_vt_repair_enabled = false;
void ExecutorVtPollLog(const char* fmt, ...) {
    std::FILE* f = std::fopen("/data/data/app.lsx4.android/files/executor-vt-poll.log", "a");
    if (!f) {
        return;
    }
    std::va_list ap;
    va_start(ap, fmt);
    std::vfprintf(f, fmt, ap);
    va_end(ap);
    std::fputc('\n', f);
    std::fclose(f);
}
void ExecutorVtDecodeMonoString(const char* label, std::uintptr_t off, std::uint64_t field) {
    if (field < 0x200000000ull || field >= 0x220000000ull) {
        return;
    }
    std::uint32_t len = 0;
    std::memcpy(&len, reinterpret_cast<const void*>(field + 0x10), 4);
    if (len == 0 || len > 240) {
        return;
    }
    char buf[252] = {};
    for (std::uint32_t i = 0; i < len && i < 240; ++i) {
        std::uint16_t ch = 0;
        std::memcpy(&ch, reinterpret_cast<const void*>(field + 0x14 + i * 2), 2);
        buf[i] = (ch >= 0x20 && ch < 0x7f) ? static_cast<char>(ch) : '.';
    }
    ExecutorVtPollLog("    str @+0x%llx (%s) len=%u = \"%s\"", (unsigned long long)off, label, len, buf);
}
void ExecutorVtDumpCorruptedObject(const char* tag, std::uintptr_t addr, std::uint64_t old_vt) {
    ExecutorVtPollLog("[%s] addr=0x%llx old_vt=0x%llx", tag, (unsigned long long)addr,
                      (unsigned long long)old_vt);
    if (old_vt >= 0x200000000ull && old_vt < 0x810000000ull) {
        std::uint64_t klass = 0;
        std::memcpy(&klass, reinterpret_cast<const void*>(old_vt), 8);
        if (klass >= 0x200000000ull && klass < 0x810000000ull) {
            for (std::uintptr_t ko = 0x28; ko <= 0x50; ko += 8) {
                std::uint64_t namep = 0;
                std::memcpy(&namep, reinterpret_cast<const void*>(klass + ko), 8);
                if (namep < 0x200000000ull || namep >= 0x810000000ull) {
                    continue;
                }
                char nb[80] = {};
                bool ascii = true;
                int n = 0;
                for (; n < 79; ++n) {
                    unsigned char c = 0;
                    std::memcpy(&c, reinterpret_cast<const void*>(namep + n), 1);
                    if (c == 0) break;
                    if (c < 0x20 || c >= 0x7f) {
                        ascii = false;
                        break;
                    }
                    nb[n] = static_cast<char>(c);
                }
                if (ascii && n >= 2) {
                    ExecutorVtPollLog("    klass.name?@+0x%llx = \"%s\"", (unsigned long long)ko, nb);
                }
            }
        }
    }
    for (std::uintptr_t off = 0x10; off <= 0x58; off += 8) {
        std::uint64_t field = 0;
        std::memcpy(&field, reinterpret_cast<const void*>(addr + off), 8);
        ExecutorVtDecodeMonoString("field", off, field);
    }
}
bool ExecutorValidateMonoVTable(std::uint64_t vt, char* name_out, std::size_t name_sz) {
    if (name_out && name_sz) {
        name_out[0] = '\0';
    }
    if (vt < 0x200000000ull || vt >= 0x810000000ull) {
        return false;
    }
    std::uint64_t klass = 0;
    std::memcpy(&klass, reinterpret_cast<const void*>(vt), 8);
    if (klass < 0x200000000ull || klass >= 0x810000000ull) {
        return false;
    }
    {
        std::uint32_t isize = 0;
        std::memcpy(&isize, reinterpret_cast<const void*>(klass + 0x1c), 4);
        if (isize < 0x10 || isize > 0x10000) {
            return false;
        }
    }
    for (std::uintptr_t ko = 0x28; ko <= 0x50; ko += 8) {
        std::uint64_t namep = 0;
        std::memcpy(&namep, reinterpret_cast<const void*>(klass + ko), 8);
        if (namep < 0x200000000ull || namep >= 0x810000000ull) {
            continue;
        }
        char nb[80] = {};
        bool ascii = true;
        int n = 0;
        for (; n < 79; ++n) {
            unsigned char c = 0;
            std::memcpy(&c, reinterpret_cast<const void*>(namep + n), 1);
            if (c == 0) {
                break;
            }
            if (c < 0x20 || c >= 0x7f) {
                ascii = false;
                break;
            }
            nb[n] = static_cast<char>(c);
        }
        if (ascii && n >= 2) {
            if (name_out && name_sz) {
                std::strncpy(name_out, nb, name_sz - 1);
                name_out[name_sz - 1] = '\0';
            }
            return true;
        }
    }
    return false;
}
void ExecutorVtZeroPollerLoop() {
    ExecutorVtPollLog("[VT_POLL] started");
    std::unordered_map<std::uintptr_t, std::uint64_t> seen;
    std::unordered_map<std::uintptr_t, std::uint64_t> chunk_seen;
    std::unordered_map<std::uintptr_t, std::pair<std::uint64_t, std::uint64_t>> chunk_victims;
    int classified = 0;
    std::vector<std::pair<std::uintptr_t, std::size_t>> allocs;
    int zeroed = 0;
    int chunk_dumps = 0;
    long long vtset_total = 0;
    std::uint64_t scans = 0;
    for (;;) {
        allocs.clear();
        {
            std::scoped_lock lk{g_executor_alloc_mutex};
            allocs.reserve(g_executor_allocations.size());
            for (const auto& [p, info] : g_executor_allocations) {
                const auto a = reinterpret_cast<std::uintptr_t>(p);
                if (info.mapping_size == ExecutorArenaMappingSentinel && info.allocated_size >= 8 &&
                    a >= 0x200000000ull && a < 0x220000000ull) {
                    allocs.emplace_back(a, info.allocated_size);
                }
            }
        }
        int plausible_now = 0;
        for (const auto& [a, sz] : allocs) {
            std::uint64_t w = 0;
            std::memcpy(&w, reinterpret_cast<const void*>(a), 8);
            const bool now_vtable = w >= 0x200000000ull && w < 0x810000000ull;
            if (now_vtable) {
                ++plausible_now;
            }
            auto it = seen.find(a);
            const bool was_vtable =
                (it != seen.end()) && it->second >= 0x200000000ull && it->second < 0x810000000ull;
            if (!was_vtable && now_vtable) {
                ++vtset_total;
            }
            if (was_vtable && w == 0) {
                ++zeroed;
                if (g_vt_repair_enabled) {
                    *reinterpret_cast<volatile std::uint64_t*>(a) = it->second;
                    if (zeroed <= 64) {
                        ExecutorVtPollLog("[VT_REPAIR] addr=0x%llx vtable=0x%llx scans=%llu", a,
                                          it->second, scans);
                    }
                    seen[a] = it->second;
                } else {
                    if (zeroed <= 64) {
                        ExecutorVtPollLog("[VT_ZEROED] addr=0x%llx old_vtable=0x%llx scans=%llu", a,
                                          it->second, scans);
                    }
                    seen[a] = w;
                }
            } else {
                seen[a] = w;
            }
            if (sz >= 0x8000) {
                const std::uintptr_t end = a + sz;
                for (std::uintptr_t q = a + 0x10; q + 8 <= end; q += 0x10) {
                    std::uint64_t v = 0;
                    std::memcpy(&v, reinterpret_cast<const void*>(q), 8);
                    auto cit = chunk_seen.find(q);
                    std::uint64_t vk = 0;
                    if (v >= 0x200000000ull && v < 0x220000000ull) {
                        std::memcpy(&vk, reinterpret_cast<const void*>(v), 8);
                    }
                    if (v >= 0x200000000ull && v < 0x220000000ull && vk >= 0x200000000ull &&
                        vk < 0x810000000ull) {
                        chunk_seen[q] = v;
                    } else if (cit != chunk_seen.end()) {
                        if (v == 0 && chunk_dumps < 200) {
                            char vname[80] = {};
                            const bool real =
                                ExecutorValidateMonoVTable(cit->second, vname, sizeof(vname));
                            if (real) {
                                ++chunk_dumps;
                                auto isize_of = [](std::uint64_t vt) -> std::uint64_t {
                                    if (vt < 0x200000000ull || vt >= 0x810000000ull) return 0;
                                    std::uint64_t kls = 0;
                                    std::memcpy(&kls, reinterpret_cast<const void*>(vt), 8);
                                    if (kls < 0x200000000ull || kls >= 0x810000000ull) return 0;
                                    std::uint32_t s = 0;
                                    std::memcpy(&s, reinterpret_cast<const void*>(kls + 0x1c), 4);
                                    return s;
                                };
                                const std::uint64_t visize = isize_of(cit->second);
                                std::uintptr_t prev = 0;
                                std::uint64_t prev_vt = 0;
                                for (const auto& [pq, pv] : chunk_seen) {
                                    if (pq < q && pq > prev) {
                                        prev = pq;
                                        prev_vt = pv;
                                    }
                                }
                                const std::uint64_t pisize = isize_of(prev_vt);
                                const bool overrun = prev && pisize && (prev + pisize > q);
                                ExecutorVtPollLog(
                                    "[VT_REAL_VICTIM] addr=0x%llx old_vt=0x%llx class=%s isize=0x%llx "
                                    "prev=0x%llx prev_isize=0x%llx prev_end=0x%llx overrun=%d",
                                    (unsigned long long)q, (unsigned long long)cit->second, vname,
                                    (unsigned long long)visize, (unsigned long long)prev,
                                    (unsigned long long)pisize,
                                    (unsigned long long)(prev + pisize), overrun ? 1 : 0);
                                ExecutorVtDumpCorruptedObject("VT_CHUNK_ZEROED", q, cit->second);
                                if (chunk_victims.size() < 256) {
                                    chunk_victims[q] = {cit->second, scans};
                                }
                            }
                        }
                        if (v == 0 && g_vt_repair_enabled) {
                            *reinterpret_cast<volatile std::uint64_t*>(q) = cit->second;
                            ++zeroed;
                            if (zeroed <= 96) {
                                ExecutorVtPollLog(
                                    "[VT_CHUNK_REPAIR] addr=0x%llx vtable=0x%llx scans=%llu", q,
                                    cit->second, scans);
                            }
                        } else {
                            chunk_seen.erase(cit);
                        }
                    }
                }
            }
        }
        for (auto vi = chunk_victims.begin(); vi != chunk_victims.end();) {
            const std::uintptr_t q = vi->first;
            const std::uint64_t old_vt = vi->second.first;
            const std::uint64_t detect_scan = vi->second.second;
            std::uint64_t cur = 0;
            std::memcpy(&cur, reinterpret_cast<const void*>(q), 8);
            if (cur != 0) {
                std::uint64_t curk = 0;
                if (cur >= 0x200000000ull && cur < 0x220000000ull) {
                    std::memcpy(&curk, reinterpret_cast<const void*>(cur), 8);
                }
                const bool valid_vt = cur >= 0x200000000ull && cur < 0x220000000ull &&
                                      curk >= 0x200000000ull && curk < 0x810000000ull;
                if (classified < 64) {
                    ++classified;
                    ExecutorVtPollLog(
                        "[VT_VICTIM_CLASS] addr=0x%llx old_vt=0x%llx new=0x%llx %s after_scans=%llu",
                        q, old_vt, cur,
                        valid_vt ? (cur == old_vt ? "RESTORED" : "REUSED_NEW_VTABLE")
                                 : "NONZERO_NONVT",
                        scans - detect_scan);
                }
                vi = chunk_victims.erase(vi);
            } else if (scans - detect_scan > 30) {
                if (classified < 64) {
                    ++classified;
                    ExecutorVtPollLog(
                        "[VT_VICTIM_CLASS] addr=0x%llx old_vt=0x%llx STAYED_ZERO after_scans=%llu", q,
                        old_vt, scans - detect_scan);
                }
                vi = chunk_victims.erase(vi);
            } else {
                ++vi;
            }
        }
        if ((++scans % 1500) == 0) {
            ExecutorVtPollLog("[VT_HEARTBEAT] scans=%llu objs=%zu plausible_now=%d vtset_total=%lld "
                              "zeroed=%d chunk_dumps=%d chunk_tracked=%zu",
                              scans, allocs.size(), plausible_now, vtset_total, zeroed, chunk_dumps,
                              chunk_seen.size());
        }
        ::usleep(100);
    }
}
void ExecutorStartVtZeroPoller() {
    int e = 0;
    if (!g_vt_poll_started.compare_exchange_strong(e, 1)) {
        return;
    }
    std::FILE* m = std::fopen(
        "/data/data/app.lsx4.android/files/lsx4-home/run-vt-zero-poll", "r");
    if (!m) {
        return;
    }
    std::fclose(m);
    if (std::FILE* rm = std::fopen(
            "/data/data/app.lsx4.android/files/lsx4-home/run-vt-repair", "r")) {
        std::fclose(rm);
        g_vt_repair_enabled = true;
    }
    std::thread(ExecutorVtZeroPollerLoop).detach();
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_VT_POLL] started repair=%d", g_vt_repair_enabled ? 1 : 0);
}
#endif

#ifndef __ANDROID__
bool ExecutorMemsetVtableGuardEnabled() {
    return false;
}

bool ExecutorMspaceFreeReuseEnabled() {
    return false;
}

u64 ExecutorMakeGuestMspaceLocked(u64 base, u64 size, u64 flags) {
    u64 handle = 0;
    u64 top_chunk = 0;
    u64 top_size = 0;
    if (!ExecutorComputeSonyMspaceGeometry(base, size, &handle, &top_chunk, &top_size)) {
        return 0;
    }
    auto* memory = Core::Memory::Instance();
    if (memory == nullptr || !memory->IsWritableMapping(base, size)) {
        return 0;
    }
    std::memset(reinterpret_cast<void*>(handle), 0, kExecutorSonyMspaceStateChunkSize);
    const u64 state_chunk = handle - kExecutorSonyMspaceChunkOverhead;
    *reinterpret_cast<u64*>(state_chunk + sizeof(u64)) =
        kExecutorSonyMspaceStateChunkSize | 0x3ULL;
    *reinterpret_cast<u64*>(top_chunk + sizeof(u64)) = top_size | 0x1ULL;
    *reinterpret_cast<u64*>(base + size - (kExecutorSonyMspaceTopFootSize - sizeof(u64))) =
        kExecutorSonyMspaceTopFootSize;
    g_executor_mspaces[handle] = ExecutorMspaceState{
        .base = base,
        .size = size,
        .offset = top_chunk - base,
        .alloc_start = top_chunk + kExecutorSonyMspaceChunkOverhead,
        .top_chunk = top_chunk,
        .top_size = top_size,
    };
    return handle;
}

u64 ExecutorEnsureDefaultGuestMspaceLocked() {
    if (g_executor_default_mspace_handle == 0) {
        g_executor_default_mspace_handle = ExecutorMakeGuestMspaceLocked(
            ExecutorLazyLibcMspaceBase, ExecutorLazyLibcMspaceSize, 0);
    }
    return g_executor_default_mspace_handle;
}

u64 ExecutorFindMspaceByBaseLocked(u64 base) {
    for (const auto& [handle, state] : g_executor_mspaces) {
        if (state.base == base) {
            return handle;
        }
    }
    return 0;
}
#endif

std::size_t ExecutorPageSize() {
#ifdef __ANDROID__
    const long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? static_cast<std::size_t>(page_size) : 4096;
#else
    return 4096;
#endif
}

std::size_t RoundUpToMultiple(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

std::uintptr_t AlignUpAddress(std::uintptr_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
}

std::uintptr_t AlignDownAddress(std::uintptr_t value, std::size_t alignment) {
    return value & ~(static_cast<std::uintptr_t>(alignment) - 1);
}

#ifdef __ANDROID__
void* ExecutorArenaAllocAligned(const std::size_t alignment, const std::size_t size) {
    std::scoped_lock lk{g_executor_arena_mutex};
    if (g_executor_arena_base == nullptr) {
        void* mapping = mmap(nullptr, ExecutorAndroidGuestHeapArenaSize, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (mapping == MAP_FAILED) {
            return nullptr;
        }
        g_executor_arena_base = static_cast<std::uint8_t*>(mapping);
        g_executor_arena_size = ExecutorAndroidGuestHeapArenaSize;
        g_executor_arena_offset = 0;
        std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC_ARENA] mmap base=%p size=%zu\n",
                     g_executor_arena_base, g_executor_arena_size);
        std::fflush(stderr);
    }

    const std::uintptr_t arena_addr = reinterpret_cast<std::uintptr_t>(g_executor_arena_base);
    const std::uintptr_t raw_addr = arena_addr + g_executor_arena_offset;
    const std::uintptr_t user_addr = AlignUpAddress(raw_addr, alignment);
    const std::size_t next_offset =
        static_cast<std::size_t>((user_addr - arena_addr) + size);
    if (next_offset > g_executor_arena_size) {
        static int arena_growth_disabled = -1;
        if (arena_growth_disabled < 0) {
            arena_growth_disabled = ExecutorRuntimeFlagExists("run-no-arena-growth") ? 1 : 0;
        }
        if (arena_growth_disabled == 1) {
            return nullptr;
        }
        std::size_t chunk_size = ExecutorAndroidGuestHeapArenaSize;
        const std::size_t needed = size + alignment;
        if (needed > chunk_size) {
            const std::size_t page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
            chunk_size = (needed + page - 1) & ~(page - 1);
        }
        void* mapping = mmap(nullptr, chunk_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (mapping == MAP_FAILED) {
            return nullptr;
        }
        g_executor_arena_base = static_cast<std::uint8_t*>(mapping);
        g_executor_arena_size = chunk_size;
        g_executor_arena_offset = 0;
        static std::atomic<int> grow_log_budget{16};
        if (grow_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_AEROLIB_ALLOC_ARENA] grow new-chunk base=%p size=%zu "
                                "reason=exhausted",
                                g_executor_arena_base, g_executor_arena_size);
            std::fprintf(stderr,
                         "[EXECUTOR_AEROLIB_ALLOC_ARENA] grow new-chunk base=%p size=%zu "
                         "reason=exhausted\n",
                         g_executor_arena_base, g_executor_arena_size);
            std::fflush(stderr);
        }
        const std::uintptr_t new_addr = reinterpret_cast<std::uintptr_t>(g_executor_arena_base);
        const std::uintptr_t new_user = AlignUpAddress(new_addr, alignment);
        g_executor_arena_offset = static_cast<std::size_t>((new_user - new_addr) + size);
        return reinterpret_cast<void*>(new_user);
    }
    g_executor_arena_offset = next_offset;
    return reinterpret_cast<void*>(user_addr);
}
#endif

bool ExecutorRuntimeFlagExists(const char* filename) {
#ifdef __ANDROID__
    if (filename == nullptr || filename[0] == '\0') {
        return false;
    }
    char path[256] = {};
    std::snprintf(path, sizeof(path), "/data/data/app.lsx4.android/files/%s", filename);
    return access(path, F_OK) == 0;
#else
    (void)filename;
    return false;
#endif
}

#ifdef __ANDROID__
bool ExecutorVerboseAerolibAllocLogsEnabled() {
    static const bool enabled = [] {
        if ((ExecutorRuntimeFlagExists("run-light-oracle") ||
             std::getenv("EXECUTOR_LIGHT_ORACLE")) &&
            std::getenv("EXECUTOR_ALLOW_LIGHT_ORACLE_AEROLIB_ALLOC_LOGS") == nullptr) {
            return false;
        }
        const char* env = std::getenv("EXECUTOR_VERBOSE_AEROLIB_ALLOC_LOGS");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool ExecutorLightOracleSuppressesAerolibAllocLogs() {
    static const bool suppressed = [] {
        return (ExecutorRuntimeFlagExists("run-light-oracle") ||
                std::getenv("EXECUTOR_LIGHT_ORACLE")) &&
               std::getenv("EXECUTOR_ALLOW_LIGHT_ORACLE_AEROLIB_ALLOC_LOGS") == nullptr;
    }();
    return suppressed;
}

bool ExecutorVerboseLibcStringLogsEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("EXECUTOR_VERBOSE_LIBC_STRING_LOGS");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool ExecutorVerboseLibcBsearchLogsEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("EXECUTOR_VERBOSE_LIBC_BSEARCH");
        return (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ||
               ExecutorRuntimeFlagExists("run-live-libc-bsearch-trace");
    }();
    return enabled;
}

bool ExecutorVerboseLibcFormattedLogsEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("EXECUTOR_TRACE_LIBC_FORMATTED");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}
#endif

bool ExecutorLeakGuardPageFreesEnabled(std::size_t requested_size) {
#ifdef __ANDROID__
    static const int configured = [] {
        const char* env = std::getenv("EXECUTOR_AEROLIB_LEAK_GUARD_PAGE_FREES");
        if (env == nullptr || env[0] == '\0') {
            return -1;
        }
        return std::strcmp(env, "0") != 0 ? 1 : 0;
    }();
    if (configured >= 0) {
        return configured == 1;
    }
    return requested_size >= 64 * 1024;
#else
    (void)requested_size;
    return false;
#endif
}

std::size_t ExecutorGuardPageThreshold() {
#ifdef __ANDROID__
    static const std::size_t threshold = [] {
        const char* env = std::getenv("EXECUTOR_AEROLIB_GUARD_PAGE_THRESHOLD");
        if (env != nullptr && env[0] != '\0') {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(env, &end, 10);
            if (end != env && value == 0) {
                return static_cast<std::size_t>(-1);
            }
            if (end != env && value > 0) {
                return static_cast<std::size_t>(value);
            }
        }
        if (ExecutorRuntimeFlagExists("executor-aerolib-guard-all.flag")) {
            return std::size_t{1};
        }
        return ExecutorAllocGuardPageDefaultThreshold;
    }();
    return threshold;
#else
    return ExecutorAllocGuardPageDefaultThreshold;
#endif
}

std::size_t ExecutorGuardSmallMax() {
#ifdef __ANDROID__
    static const std::size_t small_max = [] {
        const char* env = std::getenv("EXECUTOR_AEROLIB_GUARD_PAGE_SMALL_MAX");
        if (env != nullptr && env[0] != '\0') {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(env, &end, 10);
            if (end != env && value > 0) {
                return static_cast<std::size_t>(value);
            }
        }
        if (ExecutorRuntimeFlagExists("executor-aerolib-guard-small.flag")) {
            return std::size_t{256};
        }
        return std::size_t{0};
    }();
    return small_max;
#else
    return 0;
#endif
}

bool ExecutorValidateMspaceReuseEnabled() {
#ifdef __ANDROID__
    static const bool enabled = [] {
        const char* env = std::getenv("EXECUTOR_VALIDATE_MSPACE_REUSE");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
#else
    return false;
#endif
}

bool ExecutorShouldUseGuardPage(std::size_t normalized_size) {
#ifdef __ANDROID__
    const std::size_t small_max = ExecutorGuardSmallMax();
    if (small_max > 0 && normalized_size >= 1 && normalized_size <= small_max) {
        return true;
    }
    return normalized_size >= ExecutorGuardPageThreshold();
#else
    (void)normalized_size;
    return false;
#endif
}

void ExecutorAllocLog(bool png_path, u64 size_hint, const char* format, ...) {
#ifdef __ANDROID__
    if (!ExecutorVerboseAerolibAllocLogsEnabled()) {
        return;
    }
#endif
    const bool large = size_hint >= 4096 || size_hint == 0;
    auto& budget = png_path ? g_executor_alloc_png_log_budget
                            : (large ? g_executor_alloc_large_log_budget
                                     : g_executor_alloc_small_log_budget);
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    va_list args;
    va_start(args, format);
#ifdef __ANDROID__
    __android_log_vprint(ANDROID_LOG_INFO, "LSX4Native", format, args);
#else
    std::vfprintf(stderr, format, args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
#endif
    va_end(args);
}

void ExecutorAllocFailLog(const char* format, ...) {
    static std::atomic<int> s_alloc_fail_log_budget{256};
    if (s_alloc_fail_log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    char buffer[768];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    std::fprintf(stderr, "%s\n", buffer);
    std::fflush(stderr);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native", "%s", buffer);
#endif
}

bool ExecutorFindLiveMspaceOverlapLocked(u64 handle, u64 begin, u64 size,
                                        u64* overlap_begin = nullptr,
                                        u64* overlap_size = nullptr) {
    if (handle == 0 || size == 0 || begin > std::numeric_limits<u64>::max() - size) {
        return false;
    }
    const auto owner = g_executor_live_mspace_intervals.find(handle);
    if (owner == g_executor_live_mspace_intervals.end()) {
        return false;
    }
    const u64 end = begin + size;
    const auto& intervals = owner->second;
    auto next = intervals.lower_bound(begin);
    auto report = [&](const std::map<u64, u64>::const_iterator it) {
        if (overlap_begin != nullptr) {
            *overlap_begin = it->first;
        }
        if (overlap_size != nullptr) {
            *overlap_size = it->second;
        }
        return true;
    };
    if (next != intervals.end() && next->first < end) {
        return report(next);
    }
    if (next != intervals.begin()) {
        const auto prev = std::prev(next);
        if (prev->second > std::numeric_limits<u64>::max() - prev->first ||
            prev->first + prev->second > begin) {
            return report(prev);
        }
    }
    return false;
}

bool RememberExecutorAllocation(void* ptr, void* base = nullptr, std::size_t mapping_size = 0,
                                std::size_t requested_size = 0,
                                std::size_t allocated_size = 0,
                                std::uint64_t mspace_handle = 0,
                                std::uint64_t mspace_chunk_begin = 0,
                                std::size_t mspace_chunk_size = 0) {
    if (!ptr) {
        return false;
    }
    std::scoped_lock lk{g_executor_alloc_mutex};
    ExecutorAllocationInfo info{
        .base = base ? base : ptr,
        .mapping_size = mapping_size,
        .requested_size = requested_size,
        .allocated_size = allocated_size ? allocated_size : requested_size,
        .mspace_handle = mspace_handle,
        .mspace_chunk_begin = mspace_chunk_begin,
        .mspace_chunk_size = mspace_chunk_size,
        .serial = g_executor_alloc_serial.fetch_add(1, std::memory_order_relaxed),
    };
    const auto existing = g_executor_allocations.find(ptr);
    if (existing != g_executor_allocations.end()) {
#ifdef __ANDROID__
        static std::atomic<int> s_duplicate_log_budget{64};
        if (s_duplicate_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_AEROLIB_ALLOC_DUPLICATE] ptr=%p oldSerial=%llu newSerial=%llu "
                "oldSize=%zu newSize=%zu action=keep-old",
                ptr, static_cast<unsigned long long>(existing->second.serial),
                static_cast<unsigned long long>(info.serial), existing->second.allocated_size,
                info.allocated_size);
        }
#endif
        return false;
    }

    const u64 begin = reinterpret_cast<u64>(ptr);
    const u64 live_size = static_cast<u64>(info.allocated_size != 0 ? info.allocated_size
                                                                    : info.requested_size);
    if (live_size != 0 && begin > std::numeric_limits<u64>::max() - live_size) {
        return false;
    }
    u64 overlap_begin = 0;
    u64 overlap_size = 0;
    if (ExecutorFindLiveMspaceOverlapLocked(mspace_handle, begin, live_size, &overlap_begin,
                                            &overlap_size)) {
#ifdef __ANDROID__
        static std::atomic<int> s_interval_log_budget{64};
        if (s_interval_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_AEROLIB_ALLOC_INTERVAL_CONFLICT] handle=%p ptr=%p size=%llu "
                "live=%p:%llu action=reject",
                reinterpret_cast<void*>(mspace_handle), ptr,
                static_cast<unsigned long long>(live_size),
                reinterpret_cast<void*>(overlap_begin),
                static_cast<unsigned long long>(overlap_size));
        }
#endif
        return false;
    }

    const auto [it, inserted] = g_executor_allocations.emplace(ptr, info);
    if (!inserted) {
        return false;
    }
    if (live_size != 0) {
        const bool interval_inserted =
            g_executor_live_mspace_intervals[mspace_handle].emplace(begin, live_size).second;
        if (!interval_inserted) {
            g_executor_allocations.erase(it);
            return false;
        }
    }
    return true;
}

bool ExecutorValidateAllocationGuards(void* ptr, const ExecutorAllocationInfo& info,
                                      const char* op) {
    if (!ptr || !info.base || info.allocated_size == 0) {
        return true;
    }
#ifdef __ANDROID__
    if (info.mapping_size == ExecutorArenaMappingSentinel) {
        return true;
    }
#endif
    if (info.mapping_size != 0) {
        if (info.allocated_size <= info.requested_size) {
            return true;
        }
        const auto* suffix = static_cast<const std::uint8_t*>(ptr) + info.requested_size;
        const std::size_t suffix_size = info.allocated_size - info.requested_size;
        std::size_t bad_suffix = suffix_size;
        for (std::size_t i = 0; i < suffix_size; ++i) {
            if (suffix[i] != ExecutorAllocSuffixGuard) {
                bad_suffix = std::min(bad_suffix, i);
            }
        }
        if (bad_suffix != suffix_size &&
            g_executor_alloc_guard_violation_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            ExecutorAllocLog(false, info.requested_size,
                             "[EXECUTOR_AEROLIB_ALLOC_GUARD] op=%s ptr=%p base=%p serial=%llu requested=%zu protected=%zu suffix_padding_bad=%zu",
                             op ? op : "<unknown>", ptr, info.base,
                             static_cast<unsigned long long>(info.serial), info.requested_size,
                             info.allocated_size, bad_suffix);
            std::fprintf(stderr,
                         "[EXECUTOR_AEROLIB_ALLOC_GUARD] op=%s ptr=%p base=%p serial=%llu requested=%zu protected=%zu suffix_padding_bad=%zu\n",
                         op ? op : "<unknown>", ptr, info.base,
                         static_cast<unsigned long long>(info.serial), info.requested_size,
                         info.allocated_size, bad_suffix);
            std::fflush(stderr);
            return false;
        }
        return true;
    }

    const auto* user = static_cast<const std::uint8_t*>(ptr);
    const auto* prefix = user - ExecutorAllocRedzoneSize;
    const auto* suffix = user + info.requested_size;
    bool ok = true;
    std::size_t bad_prefix = ExecutorAllocRedzoneSize;
    std::size_t bad_suffix = ExecutorAllocRedzoneSize;
    for (std::size_t i = 0; i < ExecutorAllocRedzoneSize; ++i) {
        if (prefix[i] != ExecutorAllocPrefixGuard) {
            ok = false;
            bad_prefix = std::min(bad_prefix, i);
        }
        if (suffix[i] != ExecutorAllocSuffixGuard) {
            ok = false;
            bad_suffix = std::min(bad_suffix, i);
        }
    }

    if (!ok && g_executor_alloc_guard_violation_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        ExecutorAllocLog(false, info.requested_size,
                         "[EXECUTOR_AEROLIB_ALLOC_GUARD] op=%s ptr=%p base=%p serial=%llu requested=%zu allocated=%zu prefix_bad=%zu suffix_bad=%zu",
                         op ? op : "<unknown>", ptr, info.base,
                         static_cast<unsigned long long>(info.serial), info.requested_size,
                         info.allocated_size, bad_prefix, bad_suffix);
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC_GUARD] op=%s ptr=%p base=%p serial=%llu requested=%zu allocated=%zu prefix_bad=%zu suffix_bad=%zu\n",
                     op ? op : "<unknown>", ptr, info.base,
                     static_cast<unsigned long long>(info.serial), info.requested_size,
                     info.allocated_size, bad_prefix, bad_suffix);
        std::fflush(stderr);
    }
    return ok;
}

void ExecutorValidateAllAllocationGuards(const char* op) {
    std::scoped_lock lk{g_executor_alloc_mutex};
    for (const auto& [ptr, info] : g_executor_allocations) {
        ExecutorValidateAllocationGuards(ptr, info, op);
    }
}

bool ForgetExecutorAllocation(void* ptr, ExecutorAllocationInfo* info = nullptr,
                              u64 expected_mspace_handle = 0) {
    if (!ptr) {
        return true;
    }
    std::scoped_lock lk{g_executor_alloc_mutex};
    const auto it = g_executor_allocations.find(ptr);
    if (it == g_executor_allocations.end()) {
        return false;
    }
    if (expected_mspace_handle != 0 && it->second.mspace_handle != expected_mspace_handle) {
#ifdef __ANDROID__
        static std::atomic<int> s_wrong_owner_log_budget{64};
        if (s_wrong_owner_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_AEROLIB_FREE_OWNER_MISMATCH] ptr=%p serial=%llu expected=%p "
                "actual=%p action=reject",
                ptr, static_cast<unsigned long long>(it->second.serial),
                reinterpret_cast<void*>(expected_mspace_handle),
                reinterpret_cast<void*>(it->second.mspace_handle));
        }
#endif
        return false;
    }
    if (it->second.fios_cache_pinned) {
#ifdef __ANDROID__
        static std::atomic<int> s_fios_pin_free_log_budget{128};
        if (s_fios_pin_free_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_FIOS_CACHE_FREE_QUARANTINE] ptr=%p serial=%llu size=%zu handle=%p "
                "action=keep-live",
                ptr, static_cast<unsigned long long>(it->second.serial),
                it->second.allocated_size, reinterpret_cast<void*>(it->second.mspace_handle));
        }
#endif
        return false;
    }
    const ExecutorAllocationInfo removed = it->second;
    if (info != nullptr) {
        *info = removed;
    }
    const auto owner = g_executor_live_mspace_intervals.find(removed.mspace_handle);
    if (owner != g_executor_live_mspace_intervals.end()) {
        owner->second.erase(reinterpret_cast<u64>(ptr));
        if (owner->second.empty()) {
            g_executor_live_mspace_intervals.erase(owner);
        }
    }
    g_executor_allocations.erase(it);
    return true;
}

bool ExecutorPinFiosCacheAllocation(void* buffer, std::size_t request, const char* guest_path,
                                    s64 file_offset) {
    if (buffer == nullptr || request != 0x10000 || guest_path == nullptr ||
        std::strstr(guest_path, "sharedassets4.assets") == nullptr) {
        return false;
    }
    const u64 needle = reinterpret_cast<u64>(buffer);
    if (needle > std::numeric_limits<u64>::max() - request) {
        return false;
    }
    const u64 read_end = needle + request;
    std::scoped_lock lk{g_executor_alloc_mutex};
    auto pin = [&](auto it) {
        const u64 begin = reinterpret_cast<u64>(it->first);
        const u64 size = static_cast<u64>(it->second.allocated_size != 0
                                              ? it->second.allocated_size
                                              : it->second.requested_size);
        if (size == 0 || begin > needle || size > std::numeric_limits<u64>::max() - begin ||
            read_end > begin + size) {
            return false;
        }
        const bool first_pin = !it->second.fios_cache_pinned;
        it->second.fios_cache_pinned = true;
#ifdef __ANDROID__
        static std::atomic<int> s_fios_pin_log_budget{128};
        if (first_pin && s_fios_pin_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_FIOS_CACHE_PIN] base=%p buffer=%p size=%llu serial=%llu handle=%p "
                "path=%s offset=%lld",
                it->first, buffer, static_cast<unsigned long long>(size),
                static_cast<unsigned long long>(it->second.serial),
                reinterpret_cast<void*>(it->second.mspace_handle),
                guest_path != nullptr ? guest_path : "<unknown>", static_cast<long long>(file_offset));
        }
#else
        (void)first_pin;
        (void)guest_path;
        (void)file_offset;
#endif
        return true;
    };
    const auto exact = g_executor_allocations.find(buffer);
    if (exact != g_executor_allocations.end() && pin(exact)) {
        return true;
    }
    for (const auto& [_, intervals] : g_executor_live_mspace_intervals) {
        auto next = intervals.upper_bound(needle);
        if (next == intervals.begin()) {
            continue;
        }
        const auto prev = std::prev(next);
        if (prev->second > std::numeric_limits<u64>::max() - prev->first ||
            needle < prev->first || read_end > prev->first + prev->second) {
            continue;
        }
        const auto owner = g_executor_allocations.find(reinterpret_cast<void*>(prev->first));
        if (owner != g_executor_allocations.end() && pin(owner)) {
            return true;
        }
    }
    return false;
}

bool IsExecutorAllocation(void* ptr) {
    if (!ptr) {
        return false;
    }
    std::scoped_lock lk{g_executor_alloc_mutex};
    return g_executor_allocations.find(ptr) != g_executor_allocations.end();
}

bool FindExecutorAllocationContaining(void* ptr, void** user_base,
                                      ExecutorAllocationInfo* info = nullptr,
                                      std::size_t* offset = nullptr) {
    if (user_base != nullptr) {
        *user_base = nullptr;
    }
    if (offset != nullptr) {
        *offset = 0;
    }
    if (!ptr) {
        return false;
    }

    const auto needle = reinterpret_cast<std::uintptr_t>(ptr);
    std::scoped_lock lk{g_executor_alloc_mutex};
    const auto exact = g_executor_allocations.find(ptr);
    if (exact != g_executor_allocations.end()) {
        if (user_base != nullptr) {
            *user_base = exact->first;
        }
        if (info != nullptr) {
            *info = exact->second;
        }
        return true;
    }
    for (const auto& [candidate, candidate_info] : g_executor_allocations) {
        const auto begin = reinterpret_cast<std::uintptr_t>(candidate);
        const auto size = candidate_info.allocated_size;
        if (size == 0 || needle < begin || needle >= begin + size) {
            continue;
        }
        if (user_base != nullptr) {
            *user_base = candidate;
        }
        if (info != nullptr) {
            *info = candidate_info;
        }
        if (offset != nullptr) {
            *offset = static_cast<std::size_t>(needle - begin);
        }
        return true;
    }
    return false;
}

std::size_t GetExecutorAllocationRequestedSize(void* ptr) {
    if (!ptr) {
        return 0;
    }
    std::scoped_lock lk{g_executor_alloc_mutex};
    const auto it = g_executor_allocations.find(ptr);
    return it != g_executor_allocations.end() ? it->second.requested_size : 0;
}

std::size_t GetExecutorAllocationUsableSize(void* ptr) {
    if (!ptr) {
        return 0;
    }
    const auto needle = reinterpret_cast<std::uintptr_t>(ptr);
    std::scoped_lock lk{g_executor_alloc_mutex};
    const auto exact = g_executor_allocations.find(ptr);
    if (exact != g_executor_allocations.end()) {
        return exact->second.allocated_size ? exact->second.allocated_size
                                            : exact->second.requested_size;
    }
    for (const auto& [candidate, info] : g_executor_allocations) {
        const auto begin = reinterpret_cast<std::uintptr_t>(candidate);
        const auto size = info.allocated_size ? info.allocated_size : info.requested_size;
        if (size == 0 || needle < begin || needle >= begin + size) {
            continue;
        }
        return size - static_cast<std::size_t>(needle - begin);
    }
    return 0;
}

#ifdef __ANDROID__
bool DescribeExecutorAllocation(u64 guest_ptr, ExecutorAllocationSnapshot* out) {
    if (out == nullptr) {
        return false;
    }
    *out = {};
    {
        std::scoped_lock lk{g_executor_mspace_mutex};
        for (const auto& [handle, state] : g_executor_mspaces) {
            if (guest_ptr < state.base || guest_ptr >= state.base + state.size) {
                continue;
            }
            out->in_region = true;
            out->region_handle = handle;
            out->region_base = state.base;
            out->region_size = state.size;
            out->region_offset = guest_ptr - state.base;
            break;
        }
    }
    void* user_base = nullptr;
    ExecutorAllocationInfo info{};
    std::size_t offset = 0;
    if (!FindExecutorAllocationContaining(reinterpret_cast<void*>(guest_ptr), &user_base, &info,
                                          &offset)) {
        return out->in_region;
    }
    out->found = true;
    out->exact = reinterpret_cast<u64>(user_base) == guest_ptr;
    out->user_base = reinterpret_cast<u64>(user_base);
    out->mapping_base = reinterpret_cast<u64>(info.base);
    out->mapping_size = static_cast<u64>(info.mapping_size);
    out->requested_size = static_cast<u64>(info.requested_size);
    out->allocated_size = static_cast<u64>(info.allocated_size);
    out->mspace_handle = info.mspace_handle;
    out->serial = info.serial;
    out->offset = static_cast<u64>(offset);
    return true;
}

void TraceExecutorAllocationsContaining(u64 guest_ptr, const char* phase) {
    const auto needle = static_cast<std::uintptr_t>(guest_ptr);
    std::scoped_lock lk{g_executor_alloc_mutex};
    std::uint32_t matches = 0;
    for (const auto& [candidate, info] : g_executor_allocations) {
        const auto begin = reinterpret_cast<std::uintptr_t>(candidate);
        const auto size = info.allocated_size ? info.allocated_size : info.requested_size;
        if (size == 0 || needle < begin || needle >= begin + size) {
            continue;
        }
        if (matches < 32) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_ALLOCATION_CONTAINING] phase=%s needle=0x%llx match=%u "
                "base=%p offset=0x%llx requested=0x%llx allocated=0x%llx "
                "mappingBase=%p mappingSize=0x%llx mspace=0x%llx serial=%llu",
                phase ? phase : "unknown", static_cast<unsigned long long>(guest_ptr), matches,
                candidate, static_cast<unsigned long long>(needle - begin),
                static_cast<unsigned long long>(info.requested_size),
                static_cast<unsigned long long>(info.allocated_size), info.base,
                static_cast<unsigned long long>(info.mapping_size),
                static_cast<unsigned long long>(info.mspace_handle),
                static_cast<unsigned long long>(info.serial));
        }
        ++matches;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_ALLOCATION_CONTAINING_SUMMARY] phase=%s needle=0x%llx matches=%u",
                        phase ? phase : "unknown", static_cast<unsigned long long>(guest_ptr),
                        matches);
}
#endif

bool ResizeExecutorAllocationInPlace(void* ptr, std::size_t requested_size,
                                     const char* source) {
    if (!ptr) {
        return false;
    }
    std::scoped_lock lk{g_executor_alloc_mutex};
    const auto it = g_executor_allocations.find(ptr);
    if (it == g_executor_allocations.end()) {
        return false;
    }
    auto& info = it->second;
    const std::size_t usable = info.allocated_size ? info.allocated_size : info.requested_size;
    if (requested_size > usable) {
        return false;
    }
    info.requested_size = requested_size;
#ifdef __ANDROID__
    static std::atomic<int> s_inplace_log_budget{128};
    if (ExecutorVerboseAerolibAllocLogsEnabled() &&
        s_inplace_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_AEROLIB_REALLOC_INPLACE] source=%s ptr=%p request=%zu usable=%zu serial=%llu",
                            source ? source : "unknown", ptr, requested_size, usable,
                            static_cast<unsigned long long>(info.serial));
    }
#else
    (void)source;
#endif
    return true;
}

bool IsPowerOfTwo(const u64 value) {
    return value != 0 && (value & (value - 1)) == 0;
}

bool ExecutorIsPlausibleGuestPointer(const u64 value) {
    return value >= 0x10000 && value < (1ULL << 48);
}

std::size_t NormalizeAlignment(u64 alignment) {
    alignment = std::max<u64>(alignment, sizeof(void*));
    if (alignment > std::numeric_limits<std::size_t>::max()) {
        return 0;
    }
    if (!IsPowerOfTwo(alignment)) {
        std::size_t normalized = sizeof(void*);
        while (normalized < alignment) {
            if (normalized > std::numeric_limits<std::size_t>::max() / 2) {
                return 0;
            }
            normalized <<= 1;
        }
        return normalized;
    }
    return static_cast<std::size_t>(alignment);
}

bool ExecutorIsValidPosixAlignment(u64 alignment) {
    return alignment >= sizeof(void*) && alignment <= std::numeric_limits<std::size_t>::max() &&
           IsPowerOfTwo(alignment);
}

void* ExecutorAllocAligned(u64 alignment, u64 size);
void ExecutorFree(void* ptr, const ExecutorAllocationInfo& info);

bool ExecutorRecycleMspaceAllocation(void* ptr, const ExecutorAllocationInfo& info,
                                     u64 fallback_handle = 0) {
    if (!ExecutorMspaceFreeReuseEnabled()) {
        return false;
    }
    if (ptr == nullptr || info.allocated_size == 0 ||
        info.mspace_chunk_size < kExecutorSonyMspaceMinChunkSize) {
        return false;
    }
    const u64 handle = info.mspace_handle != 0 ? info.mspace_handle : fallback_handle;
    if (handle == 0) {
        return false;
    }

    std::scoped_lock lk{g_executor_mspace_mutex};
    const auto it = g_executor_mspaces.find(handle);
    if (it == g_executor_mspaces.end()) {
        return false;
    }
    auto& mspace = it->second;
    const u64 user_addr = reinterpret_cast<u64>(ptr);
    const u64 addr = info.mspace_chunk_begin;
    const u64 chunk_size = static_cast<u64>(info.mspace_chunk_size);
    if (addr > std::numeric_limits<u64>::max() - chunk_size ||
        mspace.base > std::numeric_limits<u64>::max() - mspace.size) {
        return false;
    }
    const u64 addr_end = addr + chunk_size;
    const u64 segment_end = mspace.base + mspace.size;
    if (mspace.alloc_start < kExecutorSonyMspaceChunkOverhead ||
        addr < mspace.alloc_start - kExecutorSonyMspaceChunkOverhead ||
        addr_end > segment_end - kExecutorSonyMspaceTopFootSize ||
        user_addr != addr + kExecutorSonyMspaceChunkOverhead) {
        return false;
    }

    {
        std::scoped_lock alk{g_executor_alloc_mutex};
        u64 live_begin = 0;
        u64 live_size = 0;
        if (ExecutorFindLiveMspaceOverlapLocked(handle, user_addr,
                                                static_cast<u64>(info.allocated_size),
                                                &live_begin, &live_size)) {
#ifdef __ANDROID__
            static std::atomic<int> s_same_mspace_overlap_log_budget{64};
            if (s_same_mspace_overlap_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(
                    ANDROID_LOG_ERROR, "LSX4Native",
                    "[EXECUTOR_AEROLIB_MSPACE] recycle-quarantine ptr=%p size=%zu "
                    "handle=%p live=%p:%llu reason=same-mspace-live-overlap",
                    ptr, info.allocated_size, reinterpret_cast<void*>(handle),
                    reinterpret_cast<void*>(live_begin),
                    static_cast<unsigned long long>(live_size));
            }
#endif
            return true;
        }
    }

    if (ExecutorValidateMspaceReuseEnabled()) {
        std::scoped_lock alk{g_executor_alloc_mutex};
        for (const auto& [live_ptr, live_info] : g_executor_allocations) {
            const u64 live_begin = reinterpret_cast<u64>(live_ptr);
            const u64 live_size = live_info.allocated_size != 0 ? live_info.allocated_size
                                                                 : live_info.requested_size;
            if (live_size == 0) {
                continue;
            }
            const u64 live_end = live_begin + live_size;
            if (addr < live_end && live_begin < addr_end) {
#ifdef __ANDROID__
                static std::atomic<int> s_live_overlap_log_budget{64};
                if (s_live_overlap_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    __android_log_print(
                        ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_AEROLIB_MSPACE] recycle-quarantine ptr=%p size=%zu "
                        "live=%p:%llu reason=overlaps-live-allocation",
                        ptr, info.allocated_size, reinterpret_cast<void*>(live_begin),
                        static_cast<unsigned long long>(live_size));
                }
#endif
                return true;
            }
        }
    }

    auto next = mspace.free_blocks.lower_bound(addr);
    if (next != mspace.free_blocks.end() && addr_end > next->first) {
#ifdef __ANDROID__
        static std::atomic<int> s_overlap_log_budget{64};
        if (s_overlap_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Native",
                "[EXECUTOR_AEROLIB_MSPACE] recycle-drop-overlap ptr=%p size=%zu "
                "existing=%p:%llu reason=duplicate_or_overlap",
                ptr, info.allocated_size, reinterpret_cast<void*>(next->first),
                static_cast<unsigned long long>(next->second));
        }
#endif
        return true;
    }
    if (next != mspace.free_blocks.begin()) {
        const auto prev = std::prev(next);
        if (prev->first + prev->second > addr) {
#ifdef __ANDROID__
            static std::atomic<int> s_overlap_log_budget{64};
            if (s_overlap_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(
                    ANDROID_LOG_WARN, "LSX4Native",
                    "[EXECUTOR_AEROLIB_MSPACE] recycle-drop-overlap ptr=%p size=%zu "
                    "existing=%p:%llu reason=duplicate_or_overlap",
                    ptr, info.allocated_size, reinterpret_cast<void*>(prev->first),
                    static_cast<unsigned long long>(prev->second));
            }
#endif
            return true;
        }
    }

    u64 merged_begin = addr;
    u64 merged_end = addr_end;
    next = mspace.free_blocks.lower_bound(addr);
    if (next != mspace.free_blocks.begin()) {
        const auto prev = std::prev(next);
        if (prev->first + prev->second == addr) {
            merged_begin = prev->first;
            mspace.free_blocks.erase(prev);
        }
    }
    next = mspace.free_blocks.lower_bound(addr);
    if (next != mspace.free_blocks.end() && merged_end == next->first) {
        merged_end = next->first + next->second;
        mspace.free_blocks.erase(next);
    }
    if (merged_end == mspace.top_chunk) {
        mspace.top_size += merged_end - merged_begin;
        mspace.top_chunk = merged_begin;
        mspace.offset = mspace.top_chunk - mspace.base;
        *reinterpret_cast<u64*>(mspace.top_chunk + sizeof(u64)) = mspace.top_size | 0x1ULL;
#ifdef __ANDROID__
        if (handle >= mspace.base && handle - mspace.base <= mspace.size - sizeof(u64) * 6) {
            auto* hdr = reinterpret_cast<u64*>(handle);
            if (hdr[0] == kExecutorGuestMspaceMagic && hdr[2] == mspace.base) {
                hdr[5] = mspace.offset;
            }
        }
#endif
    } else {
        mspace.free_blocks.emplace(merged_begin, merged_end - merged_begin);
    }
#ifdef __ANDROID__
    static std::atomic<int> s_recycle_log_budget{256};
    if (ExecutorVerboseAerolibAllocLogsEnabled() &&
        s_recycle_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_AEROLIB_MSPACE] recycle handle=%p ptr=%p size=%zu freeCount=%zu",
                            reinterpret_cast<void*>(handle), ptr, info.allocated_size,
                            mspace.free_blocks.size());
    }
#endif
    return true;
}

bool ExecutorAllocFromMspace(u64 handle, u64 alignment, u64 size, void** out) {
    if (out != nullptr) {
        *out = nullptr;
    }
    if (handle == 0) {
#ifdef __ANDROID__
        std::scoped_lock lk{g_executor_mspace_mutex};
        handle = ExecutorEnsureDefaultGuestMspaceLocked();
#else
        handle = g_executor_default_mspace_handle;
#endif
    }
    if (handle == 0 || out == nullptr) {
        return false;
    }

    const std::size_t requested_alignment = NormalizeAlignment(alignment);
    if (requested_alignment == 0) {
        return true;
    }
    const std::size_t normalized_alignment =
        std::max<std::size_t>(requested_alignment, kExecutorSonyMspaceAlignment);
    const std::size_t normalized_size = static_cast<std::size_t>(std::max<u64>(size, 1));
    u64 requested_chunk_size = 0;
    if (!ExecutorSonyMspaceRequestToChunkSize(static_cast<u64>(normalized_size),
                                               &requested_chunk_size)) {
        return true;
    }
#ifdef __ANDROID__
    std::unique_lock lk{g_executor_mspace_mutex};
#else
    std::scoped_lock lk{g_executor_mspace_mutex};
#endif
    auto it = g_executor_mspaces.find(handle);
    if (it == g_executor_mspaces.end()) {
        return false;
    }

    auto& mspace = it->second;
    if (ExecutorMspaceFreeReuseEnabled()) {
        for (auto free_it = mspace.free_blocks.begin(); free_it != mspace.free_blocks.end();) {
            const u64 block_begin = free_it->first;
            const u64 block_size = free_it->second;
            u64 chunk_begin = 0;
            u64 allocation_chunk_size = 0;
            u64 prefix = 0;
            u64 suffix_size = 0;
            if (!ExecutorPlaceSonyMspaceChunkInFreeBlock(
                    block_begin, block_size, static_cast<u64>(normalized_alignment),
                    requested_chunk_size, &chunk_begin, &allocation_chunk_size, &prefix,
                    &suffix_size)) {
                ++free_it;
                continue;
            }
            const u64 user_addr = chunk_begin + kExecutorSonyMspaceChunkOverhead;
            const u64 usable_size = allocation_chunk_size - kExecutorSonyMspaceChunkOverhead;

            bool overlaps_live = false;
            u64 overlap_begin = 0;
            u64 overlap_size = 0;
            {
                std::scoped_lock alk{g_executor_alloc_mutex};
                overlaps_live = ExecutorFindLiveMspaceOverlapLocked(
                    handle, block_begin, block_size,
                    &overlap_begin, &overlap_size);
            }
            if (!overlaps_live && ExecutorValidateMspaceReuseEnabled()) {
                std::scoped_lock alk{g_executor_alloc_mutex};
                const u64 block_end = block_begin + block_size;
                for (const auto& [live_ptr, live_info] : g_executor_allocations) {
                    const u64 live_begin = reinterpret_cast<u64>(live_ptr);
                    const u64 live_size = live_info.allocated_size != 0
                                              ? live_info.allocated_size
                                              : live_info.requested_size;
                    if (live_size == 0 || live_begin > std::numeric_limits<u64>::max() - live_size) {
                        continue;
                    }
                    const u64 live_end = live_begin + live_size;
                    if (block_begin < live_end && live_begin < block_end) {
                        overlaps_live = true;
                        overlap_begin = live_begin;
                        overlap_size = live_size;
                        break;
                    }
                }
            }
            if (overlaps_live) {
#ifdef __ANDROID__
                static std::atomic<int> s_reuse_overlap_log_budget{64};
                if (s_reuse_overlap_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    __android_log_print(
                        ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_AEROLIB_MSPACE] reuse-quarantine block=%p:%llu "
                        "handle=%p live=%p:%llu reason=live-overlap",
                        reinterpret_cast<void*>(block_begin),
                        static_cast<unsigned long long>(block_size),
                        reinterpret_cast<void*>(handle),
                        reinterpret_cast<void*>(overlap_begin),
                        static_cast<unsigned long long>(overlap_size));
                }
#endif
                free_it = mspace.free_blocks.erase(free_it);
                continue;
            }

            const u64 suffix_ptr = chunk_begin + allocation_chunk_size;
            const auto old_block = *free_it;
            mspace.free_blocks.erase(free_it);
            if (prefix != 0) {
                mspace.free_blocks.emplace(old_block.first, prefix);
            }
            if (suffix_size != 0 && suffix_ptr >= mspace.alloc_start -
                                                               kExecutorSonyMspaceChunkOverhead &&
                suffix_ptr <= mspace.base + mspace.size - kExecutorSonyMspaceTopFootSize &&
                suffix_size <= mspace.base + mspace.size - kExecutorSonyMspaceTopFootSize -
                                   suffix_ptr) {
                mspace.free_blocks.emplace(suffix_ptr, suffix_size);
            }

            *reinterpret_cast<u64*>(chunk_begin + sizeof(u64)) =
                allocation_chunk_size | 0x3ULL;
            auto* ptr = reinterpret_cast<void*>(user_addr);
#ifdef __ANDROID__
            const bool remembered =
                RememberExecutorAllocation(ptr, ptr, ExecutorArenaMappingSentinel,
                                           normalized_size, static_cast<std::size_t>(usable_size),
                                           handle, chunk_begin,
                                           static_cast<std::size_t>(allocation_chunk_size));
#else
            const bool remembered =
                RememberExecutorAllocation(ptr, ptr, 0, normalized_size,
                                           static_cast<std::size_t>(usable_size), handle,
                                           chunk_begin,
                                           static_cast<std::size_t>(allocation_chunk_size));
#endif
            if (!remembered) {
#ifdef __ANDROID__
                static std::atomic<int> s_reuse_commit_log_budget{64};
                if (s_reuse_commit_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    __android_log_print(
                        ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_AEROLIB_MSPACE] reuse-quarantine block=%p:%llu "
                        "handle=%p candidate=%p:%zu reason=registry-commit-rejected",
                        reinterpret_cast<void*>(old_block.first),
                        static_cast<unsigned long long>(old_block.second),
                        reinterpret_cast<void*>(handle), ptr, normalized_size);
                }
#endif
                free_it = mspace.free_blocks.lower_bound(suffix_ptr);
                continue;
            }
#ifdef __ANDROID__
            if (ExecutorShouldPoisonGuestMalloc()) {
                std::memset(ptr, ExecutorGuestMallocUninitializedFill, normalized_size);
            }
#endif
            *out = ptr;
            ExecutorAllocLog(false, normalized_size,
                             "[EXECUTOR_AEROLIB_MSPACE] reuse handle=%p align=%zu size=%zu ptr=%p block=%p blockSize=%llu freeCount=%zu",
                             reinterpret_cast<void*>(handle), normalized_alignment,
                             normalized_size, ptr, reinterpret_cast<void*>(old_block.first),
                             static_cast<unsigned long long>(old_block.second),
                             mspace.free_blocks.size());
#ifdef __ANDROID__
            static std::atomic<int> s_reuse_log_budget{256};
            if (ExecutorVerboseAerolibAllocLogsEnabled() &&
                s_reuse_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_AEROLIB_MSPACE] reuse handle=%p align=%zu size=%zu ptr=%p block=%p blockSize=%llu freeCount=%zu",
                                    reinterpret_cast<void*>(handle), normalized_alignment,
                                    normalized_size, ptr, reinterpret_cast<void*>(old_block.first),
                                    static_cast<unsigned long long>(old_block.second),
                                    mspace.free_blocks.size());
            }
#endif
            return true;
        }
    }

    if (mspace.top_chunk > std::numeric_limits<u64>::max() -
                               kExecutorSonyMspaceChunkOverhead) {
        return true;
    }
    u64 aligned_user = 0;
    if (!ExecutorAlignUpSonyMspace(mspace.top_chunk + kExecutorSonyMspaceChunkOverhead,
                                   static_cast<u64>(normalized_alignment), &aligned_user) ||
        aligned_user < kExecutorSonyMspaceChunkOverhead) {
        return true;
    }
    u64 chunk_begin = aligned_user - kExecutorSonyMspaceChunkOverhead;
    u64 prefix = chunk_begin - mspace.top_chunk;
    if (prefix != 0 && prefix < kExecutorSonyMspaceMinChunkSize) {
        if (aligned_user > std::numeric_limits<u64>::max() - normalized_alignment) {
            return true;
        }
        aligned_user += normalized_alignment;
        chunk_begin = aligned_user - kExecutorSonyMspaceChunkOverhead;
        prefix = chunk_begin - mspace.top_chunk;
    }
    if (prefix > std::numeric_limits<u64>::max() - requested_chunk_size) {
        return true;
    }
    const u64 required_top = prefix + requested_chunk_size;

    if (required_top >= mspace.top_size && handle == g_executor_default_mspace_handle &&
        mspace.base == ExecutorLazyLibcMspaceBase) {
        const u64 top_offset = mspace.top_chunk - mspace.base;
        u64 required_capacity = 0;
        if (top_offset <= std::numeric_limits<u64>::max() - required_top &&
            top_offset + required_top <=
                std::numeric_limits<u64>::max() - kExecutorSonyMspaceTopFootSize -
                    kExecutorSonyMspaceMinChunkSize) {
            required_capacity = top_offset + required_top + kExecutorSonyMspaceTopFootSize +
                                kExecutorSonyMspaceMinChunkSize;
        }
        const u64 old_size = mspace.size;
        const u64 grown_size = required_capacity != 0
                                   ? Core::Memory::Instance()->GrowExecutorLibcMspace(
                                         required_capacity)
                                   : 0;
        if (grown_size > old_size) {
            mspace.size = grown_size;
            mspace.top_size += grown_size - old_size;
            *reinterpret_cast<u64*>(mspace.base + grown_size -
                                    (kExecutorSonyMspaceTopFootSize - sizeof(u64))) =
                kExecutorSonyMspaceTopFootSize;

#ifdef __ANDROID__
            if (handle >= mspace.base && handle - mspace.base <= mspace.size - sizeof(u64) * 4) {
                auto* hdr = reinterpret_cast<u64*>(handle);
                if (hdr[0] == kExecutorGuestMspaceMagic && hdr[2] == mspace.base) {
                    hdr[3] = grown_size;
                }
            }
#endif
        }
    }
    if (required_top >= mspace.top_size) {
        ExecutorAllocLog(false, normalized_size,
                         "[EXECUTOR_AEROLIB_MSPACE] oom handle=%p base=%p size=%llu top=%p topSize=%llu align=%zu request=%zu nb=%llu",
                         reinterpret_cast<void*>(handle), reinterpret_cast<void*>(mspace.base),
                         static_cast<unsigned long long>(mspace.size),
                         reinterpret_cast<void*>(mspace.top_chunk),
                         static_cast<unsigned long long>(mspace.top_size), normalized_alignment,
                         normalized_size, static_cast<unsigned long long>(requested_chunk_size));
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_MSPACE] oom handle=%p base=%p size=%llu top=%p topSize=%llu align=%zu request=%zu nb=%llu\n",
                     reinterpret_cast<void*>(handle), reinterpret_cast<void*>(mspace.base),
                     static_cast<unsigned long long>(mspace.size),
                     reinterpret_cast<void*>(mspace.top_chunk),
                     static_cast<unsigned long long>(mspace.top_size), normalized_alignment,
                     normalized_size, static_cast<unsigned long long>(requested_chunk_size));
        std::fflush(stderr);
#ifdef __ANDROID__
        {
            static std::atomic<int> s_mspace_oom_log_budget{64};
            if (s_mspace_oom_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                    "[EXECUTOR_AEROLIB_MSPACE] oom handle=%p base=%p size=0x%llx "
                                    "top=%p topSize=0x%llx align=%zu request=%zu nb=0x%llx default=%p",
                                    reinterpret_cast<void*>(handle),
                                    reinterpret_cast<void*>(mspace.base),
                                    static_cast<unsigned long long>(mspace.size),
                                    reinterpret_cast<void*>(mspace.top_chunk),
                                    static_cast<unsigned long long>(mspace.top_size),
                                    normalized_alignment, normalized_size,
                                    static_cast<unsigned long long>(requested_chunk_size),
                                    reinterpret_cast<void*>(g_executor_default_mspace_handle));
            }
        }
#endif
        return true;
    }

    const u64 old_top = mspace.top_chunk;
    const u64 new_top = old_top + required_top;
    const u64 new_top_size = mspace.top_size - required_top;
    const u64 usable_size = requested_chunk_size - kExecutorSonyMspaceChunkOverhead;
    const auto aligned = static_cast<std::uintptr_t>(aligned_user);

#ifdef __ANDROID__
    if (ExecutorValidateMspaceReuseEnabled() && normalized_size >= 0x1000) {
        static std::atomic<int> s_overlap_budget{64};
        const std::uintptr_t a0 = aligned, a1 = aligned + normalized_size;
        std::scoped_lock alk{g_executor_alloc_mutex};
        for (const auto& [p, info] : g_executor_allocations) {
            const std::uintptr_t b0 = reinterpret_cast<std::uintptr_t>(info.base);
            const std::uintptr_t b1 = b0 + info.allocated_size;
            if (info.allocated_size != 0 && a0 < b1 && b0 < a1) {
                if (s_overlap_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    __android_log_print(
                        ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_MSPACE_OVERLAP] NEW=[0x%llx,0x%llx) sz=%zu align=%zu handle=%p "
                        "OVERLAPS LIVE=[0x%llx,0x%llx) reqsz=%zu serial=%llu",
                        (unsigned long long)a0, (unsigned long long)a1, normalized_size,
                        normalized_alignment, reinterpret_cast<void*>(handle),
                        (unsigned long long)b0, (unsigned long long)b1, info.requested_size,
                        (unsigned long long)info.serial);
                }
            }
        }
    }
#endif

    auto* ptr = reinterpret_cast<void*>(aligned);
#ifdef __ANDROID__
    const bool remembered =
        RememberExecutorAllocation(ptr, ptr, ExecutorArenaMappingSentinel, normalized_size,
                                   static_cast<std::size_t>(usable_size), handle, chunk_begin,
                                   static_cast<std::size_t>(requested_chunk_size));
#else
    const bool remembered =
        RememberExecutorAllocation(ptr, ptr, 0, normalized_size,
                                   static_cast<std::size_t>(usable_size), handle, chunk_begin,
                                   static_cast<std::size_t>(requested_chunk_size));
#endif
    if (!remembered) {
#ifdef __ANDROID__
        static std::atomic<int> s_bump_commit_log_budget{64};
        if (s_bump_commit_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_AEROLIB_MSPACE] alloc-reject handle=%p ptr=%p size=%zu "
                "reason=registry-commit-rejected",
                reinterpret_cast<void*>(handle), ptr, normalized_size);
        }
#endif
        return true;
    }
    if (prefix >= kExecutorSonyMspaceMinChunkSize) {
        mspace.free_blocks.emplace(old_top, prefix);
    }
    *reinterpret_cast<u64*>(chunk_begin + sizeof(u64)) = requested_chunk_size | 0x3ULL;
    *reinterpret_cast<u64*>(new_top + sizeof(u64)) = new_top_size | 0x1ULL;
    mspace.top_chunk = new_top;
    mspace.top_size = new_top_size;
    mspace.offset = new_top - mspace.base;
#ifdef __ANDROID__
    if (handle >= mspace.base && handle - mspace.base <= mspace.size - sizeof(u64) * 6) {
        auto* hdr = reinterpret_cast<u64*>(handle);
        if (hdr[0] == kExecutorGuestMspaceMagic && hdr[2] == mspace.base) {
            hdr[5] = mspace.offset;
        }
    }
#endif
#ifdef __ANDROID__
    if (ExecutorShouldPoisonGuestMalloc()) {
        std::memset(ptr, ExecutorGuestMallocUninitializedFill, normalized_size);
    }
#endif
    *out = ptr;
    ExecutorAllocLog(false, normalized_size,
                     "[EXECUTOR_AEROLIB_MSPACE] alloc handle=%p align=%zu size=%zu ptr=%p next=%llu",
                     reinterpret_cast<void*>(handle), normalized_alignment, normalized_size, ptr,
                     static_cast<unsigned long long>(mspace.offset));
    return true;
}

void* ExecutorAllocDefaultMspaceOrFallback(u64 alignment, u64 size, bool zero = false) {
    void* ptr = nullptr;
    if (ExecutorAllocFromMspace(0, alignment, size, &ptr)) {
        if (ptr != nullptr && zero) {
            std::memset(ptr, 0, static_cast<std::size_t>(std::max<u64>(size, 1)));
        }
        return ptr;
    }
    ptr = ExecutorAllocAligned(alignment, size);
    if (ptr != nullptr && zero) {
        std::memset(ptr, 0, static_cast<std::size_t>(std::max<u64>(size, 1)));
    }
    return ptr;
}

void* ExecutorReallocUsingAllocator(u64 handle, void* old, u64 size, u64 alignment) {
    const std::size_t new_size = static_cast<std::size_t>(std::max<u64>(size, 1));
    if (old != nullptr) {
        const std::size_t usable = GetExecutorAllocationUsableSize(old);
        if (usable != 0 && new_size <= usable && ResizeExecutorAllocationInPlace(
                                                       old, new_size, "exact-shrink-or-fit")) {
            return old;
        }
    }
    void* ptr = nullptr;
    if (!ExecutorAllocFromMspace(handle, alignment, new_size, &ptr)) {
        ptr = ExecutorAllocAligned(alignment, new_size);
    }
    if (old != nullptr && ptr != nullptr) {
        ExecutorAllocationInfo old_info{};
        const std::size_t old_size = GetExecutorAllocationRequestedSize(old);
        if (old_size != 0) {
            std::memcpy(ptr, old, std::min(old_size, new_size));
        }
        if (ForgetExecutorAllocation(old, &old_info, handle)) {
            if (!ExecutorRecycleMspaceAllocation(old, old_info, handle)) {
                ExecutorFree(old, old_info);
            }
        }
    }
    return ptr;
}

#ifdef __ANDROID__
static std::size_t ExecutorCopyUntrackedReallocBytes(void* dst, const void* old,
                                                     std::size_t request_size) {
    if (dst == nullptr || old == nullptr || request_size == 0) {
        return 0;
    }

    constexpr std::size_t MaxUnknownReallocCopy = 1024 * 1024;
    constexpr std::size_t Page = 4096;
    const std::size_t copy_limit = std::min(request_size, MaxUnknownReallocCopy);
    const auto old_addr = reinterpret_cast<u64>(old);
    auto* out = static_cast<std::uint8_t*>(dst);
    std::size_t copied = 0;
    while (copied < copy_limit) {
        const std::size_t page_remaining =
            Page - static_cast<std::size_t>((old_addr + copied) & (Page - 1));
        const std::size_t chunk = std::min(page_remaining, copy_limit - copied);
        if (!ExecutorReadGuestBytes(old_addr + copied, out + copied, chunk)) {
            break;
        }
        copied += chunk;
    }
    return copied;
}
#endif

void* ExecutorReallocPossiblyInterior(u64 handle, void* old, u64 size, u64 alignment,
                                      bool* known_old, std::size_t* old_offset = nullptr) {
    if (known_old != nullptr) {
        *known_old = old == nullptr;
    }
    if (old_offset != nullptr) {
        *old_offset = 0;
    }
    if (old == nullptr || IsExecutorAllocation(old)) {
        if (known_old != nullptr) {
            *known_old = true;
        }
        return ExecutorReallocUsingAllocator(handle, old, size, alignment);
    }

    void* old_base = nullptr;
    ExecutorAllocationInfo old_info{};
    std::size_t offset = 0;
    const std::size_t new_size = static_cast<std::size_t>(std::max<u64>(size, 1));
    if (FindExecutorAllocationContaining(old, &old_base, &old_info, &offset)) {
        if (known_old != nullptr) {
            *known_old = true;
        }
        if (old_offset != nullptr) {
            *old_offset = offset;
        }
        const std::size_t old_usable =
            old_info.allocated_size > offset ? old_info.allocated_size - offset : 0;
        if (old_usable != 0 && new_size <= old_usable) {
#ifdef __ANDROID__
            static std::atomic<int> s_interior_inplace_log_budget{128};
            if (s_interior_inplace_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_AEROLIB_REALLOC_INPLACE] source=interior-shrink-or-fit old=%p base=%p offset=%zu request=%zu usable=%zu serial=%llu",
                                    old, old_base, offset, new_size, old_usable,
                                    static_cast<unsigned long long>(old_info.serial));
            }
#endif
            return old;
        }
        void* ptr = nullptr;
        if (!ExecutorAllocFromMspace(handle, alignment, new_size, &ptr)) {
            ptr = ExecutorAllocAligned(alignment, new_size);
        }
        if (ptr != nullptr) {
            const auto copy_size =
                std::min(old_info.requested_size > offset ? old_info.requested_size - offset : 0,
                         new_size);
            if (copy_size != 0) {
                std::memcpy(ptr, old, copy_size);
            }
            if (offset == 0 && ForgetExecutorAllocation(old_base, &old_info, handle)) {
                if (!ExecutorRecycleMspaceAllocation(old_base, old_info, handle)) {
                    ExecutorFree(old_base, old_info);
                }
            }
        }
        return ptr;
    }

#ifdef __ANDROID__
    if (ExecutorIsPlausibleGuestPointer(reinterpret_cast<u64>(old))) {
        if (old_offset != nullptr) {
            *old_offset = SIZE_MAX;
        }
        void* ptr = nullptr;
        if (!ExecutorAllocFromMspace(handle, alignment, new_size, &ptr)) {
            ptr = ExecutorAllocAligned(alignment, new_size);
        }
        if (ptr != nullptr) {
            const std::size_t copied = ExecutorCopyUntrackedReallocBytes(ptr, old, new_size);
            static std::atomic<int> s_untracked_copy_log_budget{256};
            if (s_untracked_copy_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                    "[EXECUTOR_AEROLIB_REALLOC_UNTRACKED_COPY] old=%p ptr=%p request=%zu copied=%zu full=%d handle=%p",
                                    old, ptr, new_size, copied, copied == new_size ? 1 : 0,
                                    reinterpret_cast<void*>(handle));
            }
        }
        return ptr;
    }
#endif

    return nullptr;
}

void* ExecutorAllocAligned(const u64 alignment, const u64 size) {
    const std::size_t normalized_alignment = NormalizeAlignment(alignment);
    if (normalized_alignment == 0) {
        return nullptr;
    }
    const std::size_t normalized_size = static_cast<std::size_t>(std::max<u64>(size, 1));
    if (normalized_size > SIZE_MAX - normalized_alignment - (ExecutorAllocRedzoneSize * 2)) {
        return nullptr;
    }

#ifdef __ANDROID__
    const std::size_t guard_page_threshold = ExecutorGuardPageThreshold();
    const std::size_t guard_small_max = ExecutorGuardSmallMax();
    if (ExecutorShouldUseGuardPage(normalized_size)) {
        const std::size_t page_size = ExecutorPageSize();
        const std::size_t protected_user_size =
            RoundUpToMultiple(normalized_size + normalized_alignment, page_size);
        if (protected_user_size > SIZE_MAX - page_size - page_size) {
            return nullptr;
        }
        const std::size_t mapping_size = page_size + protected_user_size + page_size;
        void* mapping = mmap(nullptr, mapping_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) {
            return nullptr;
        }

        const auto mapping_addr = reinterpret_cast<std::uintptr_t>(mapping);
        const auto user_region_addr = mapping_addr + page_size;
        const auto guard_addr = user_region_addr + protected_user_size;
        const auto user_addr =
            AlignDownAddress(guard_addr - normalized_size, normalized_alignment);
        if (user_addr < user_region_addr || user_addr + normalized_size > guard_addr) {
            munmap(mapping, mapping_size);
            return nullptr;
        }
        auto* user = reinterpret_cast<std::uint8_t*>(user_addr);
        auto* user_region = reinterpret_cast<std::uint8_t*>(user_region_addr);
        if (mprotect(user_region, protected_user_size, PROT_READ | PROT_WRITE) != 0) {
            munmap(mapping, mapping_size);
            return nullptr;
        }
        const std::size_t prefix_slack = user_addr - user_region_addr;
        const std::size_t suffix_slack = guard_addr - (user_addr + normalized_size);
        if (prefix_slack > 0) {
            std::memset(user_region, ExecutorAllocPrefixGuard, prefix_slack);
        }
        if (suffix_slack > 0) {
            std::memset(user + normalized_size, ExecutorAllocSuffixGuard, suffix_slack);
        }
        if (ExecutorShouldPoisonGuestMalloc()) {
            std::memset(user, ExecutorGuestMallocUninitializedFill, normalized_size);
        }

        RememberExecutorAllocation(user, mapping, mapping_size, normalized_size,
                                   normalized_size + suffix_slack);
        ExecutorAllocLog(false, normalized_size,
                         "[EXECUTOR_AEROLIB_ALLOC_GUARD] op=guard-page-alloc ptr=%p userEnd=%p guard=%p base=%p requested=%zu protected=%zu mapping=%zu page=%zu threshold=%zu smallMax=%zu prefixSlack=%zu suffixSlack=%zu",
                         user, user + normalized_size, reinterpret_cast<void*>(guard_addr),
                         mapping, normalized_size, protected_user_size, mapping_size, page_size,
                         guard_page_threshold, guard_small_max, prefix_slack, suffix_slack);
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC_GUARD] op=guard-page-alloc ptr=%p userEnd=%p guard=%p base=%p requested=%zu protected=%zu mapping=%zu page=%zu threshold=%zu smallMax=%zu prefixSlack=%zu suffixSlack=%zu\n",
                     user, user + normalized_size, reinterpret_cast<void*>(guard_addr), mapping,
                     normalized_size, protected_user_size, mapping_size, page_size,
                     guard_page_threshold, guard_small_max, prefix_slack, suffix_slack);
        std::fflush(stderr);
        return user;
    }
#endif

    const std::size_t allocated_size =
        normalized_size + normalized_alignment + (ExecutorAllocRedzoneSize * 2);
    ExecutorValidateAllAllocationGuards("pre-alloc");
#ifdef __ANDROID__
    void* arena_user = ExecutorArenaAllocAligned(normalized_alignment, normalized_size);
    if (arena_user != nullptr) {
        if (ExecutorShouldPoisonGuestMalloc()) {
            std::memset(arena_user, ExecutorGuestMallocUninitializedFill, normalized_size);
        }
        RememberExecutorAllocation(arena_user, arena_user, ExecutorArenaMappingSentinel,
                                   normalized_size, normalized_size);
        return arena_user;
    }
#endif
    void* raw = std::malloc(allocated_size);
    if (!raw) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=aligned_malloc align=%zu size=%zu allocated=%zu source=std_malloc",
            normalized_alignment, normalized_size, allocated_size);
        return nullptr;
    }

    const auto raw_addr = reinterpret_cast<std::uintptr_t>(raw);
    const auto user_min = raw_addr + ExecutorAllocRedzoneSize;
    const auto user_addr = (user_min + normalized_alignment - 1) & ~(normalized_alignment - 1);
    auto* user = reinterpret_cast<std::uint8_t*>(user_addr);
    std::memset(user - ExecutorAllocRedzoneSize, ExecutorAllocPrefixGuard,
                ExecutorAllocRedzoneSize);
#ifdef __ANDROID__
    if (ExecutorShouldPoisonGuestMalloc()) {
        std::memset(user, ExecutorGuestMallocUninitializedFill, normalized_size);
    }
#endif
    std::memset(user + normalized_size, ExecutorAllocSuffixGuard, ExecutorAllocRedzoneSize);
    RememberExecutorAllocation(user, raw, 0, normalized_size, normalized_size);
    return user;
}

void* ExecutorAlloc(const u64 size, const bool zero = false) {
    void* ptr = ExecutorAllocAligned(alignof(std::max_align_t), size);
    if (ptr != nullptr && zero) {
        std::memset(ptr, 0, static_cast<std::size_t>(std::max<u64>(size, 1)));
    }
    return ptr;
}

void ExecutorFree(void* ptr, const ExecutorAllocationInfo& info) {
#ifdef __ANDROID__
    if (info.mapping_size == ExecutorArenaMappingSentinel) {
        ExecutorValidateAllocationGuards(ptr, info, "free");
        return;
    }
    if (info.mapping_size != 0 && info.base != nullptr) {
        ExecutorValidateAllocationGuards(ptr, info, "free");
        const bool leak_guard_page = ExecutorLeakGuardPageFreesEnabled(info.requested_size);
        if (g_executor_alloc_guard_free_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            ExecutorAllocLog(false, info.requested_size,
                             "[EXECUTOR_AEROLIB_ALLOC_GUARD] op=%s ptr=%p base=%p serial=%llu requested=%zu protected=%zu mapping=%zu munmap=%d",
                             leak_guard_page ? "guard-page-free-leak"
                                             : "guard-page-free-munmap",
                             ptr, info.base, static_cast<unsigned long long>(info.serial),
                             info.requested_size, info.allocated_size, info.mapping_size,
                             leak_guard_page ? 0 : 1);
            std::fprintf(stderr,
                         "[EXECUTOR_AEROLIB_ALLOC_GUARD] op=%s ptr=%p base=%p serial=%llu requested=%zu protected=%zu mapping=%zu munmap=%d\n",
                         leak_guard_page ? "guard-page-free-leak"
                                         : "guard-page-free-munmap",
                         ptr, info.base, static_cast<unsigned long long>(info.serial),
                         info.requested_size, info.allocated_size, info.mapping_size,
                         leak_guard_page ? 0 : 1);
            std::fflush(stderr);
        }
        if (leak_guard_page) {
            return;
        }
        munmap(info.base, info.mapping_size);
        return;
    }
#endif
    ExecutorValidateAllocationGuards(ptr, info, "free");
    std::free(info.base ? info.base : ptr);
}

u64 ExecutorMspaceCreate(u64 name, u64 base, u64 size) {
    u64 handle = 0;
    bool guest_supplied_mspace = false;
    u64 old_default = 0;
    bool promoted_default = false;
    {
        std::scoped_lock lk{g_executor_mspace_mutex};
        auto* memory = Core::Memory::Instance();
        if (ExecutorIsPlausibleGuestPointer(base) && size >= 0x10000 && memory != nullptr &&
            memory->IsWritableMapping(base, size)) {
            guest_supplied_mspace = true;
            handle = ExecutorFindMspaceByBaseLocked(base);
            if (handle == 0) {
                handle = ExecutorMakeGuestMspaceLocked(base, size, 0);
            } else {
                auto& existing = g_executor_mspaces[handle];
                if (size > existing.size) {
                    existing.top_size += size - existing.size;
                    existing.size = size;
                    *reinterpret_cast<u64*>(existing.top_chunk + sizeof(u64)) =
                        existing.top_size | 0x1ULL;
                    *reinterpret_cast<u64*>(existing.base + existing.size -
                                            (kExecutorSonyMspaceTopFootSize - sizeof(u64))) =
                        kExecutorSonyMspaceTopFootSize;
#ifdef __ANDROID__
                    auto* hdr = reinterpret_cast<u64*>(handle);
                    if (hdr[0] == kExecutorGuestMspaceMagic && hdr[2] == existing.base) {
                        hdr[3] = existing.size;
                    }
#endif
                }
            }
        }
        if (handle == 0) {
            handle = ExecutorEnsureDefaultGuestMspaceLocked();
        }
        if (!guest_supplied_mspace && g_executor_default_mspace_handle == 0) {
            g_executor_default_mspace_handle = handle;
        }
    }
    ExecutorAllocLog(false, 0,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceCreate name=%p base=%p size=%llu handle=%p default=%p",
                     reinterpret_cast<void*>(name), reinterpret_cast<void*>(base),
                     static_cast<unsigned long long>(size), reinterpret_cast<void*>(handle),
                     reinterpret_cast<void*>(g_executor_default_mspace_handle));
    if (promoted_default) {
        ExecutorAllocLog(false, 0,
                         "[EXECUTOR_AEROLIB_ALLOC] default mspace promoted old=%p new=%p base=%p "
                         "size=%llu",
                         reinterpret_cast<void*>(old_default), reinterpret_cast<void*>(handle),
                         reinterpret_cast<void*>(base), static_cast<unsigned long long>(size));
    }
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled()) {
        static std::atomic<int> s_mspace_create_log_budget{64};
        if (s_mspace_create_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_AEROLIB_MSPACE] create name=%p base=%p size=0x%llx "
                                "handle=%p default=%p guest=%d promoted=%d oldDefault=%p",
                                reinterpret_cast<void*>(name), reinterpret_cast<void*>(base),
                                static_cast<unsigned long long>(size),
                                reinterpret_cast<void*>(handle),
                                reinterpret_cast<void*>(g_executor_default_mspace_handle),
                                guest_supplied_mspace ? 1 : 0, promoted_default ? 1 : 0,
                                reinterpret_cast<void*>(old_default));
        }
    }
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceCreate name=%p base=%p size=%llu handle=%p default=%p\n",
                     reinterpret_cast<void*>(name), reinterpret_cast<void*>(base),
                     static_cast<unsigned long long>(size), reinterpret_cast<void*>(handle),
                     reinterpret_cast<void*>(g_executor_default_mspace_handle));
        if (promoted_default) {
            std::fprintf(stderr,
                         "[EXECUTOR_AEROLIB_ALLOC] default mspace promoted old=%p new=%p base=%p "
                         "size=%llu\n",
                         reinterpret_cast<void*>(old_default), reinterpret_cast<void*>(handle),
                         reinterpret_cast<void*>(base), static_cast<unsigned long long>(size));
        }
        std::fflush(stderr);
    }
    return handle;
}

u64 ExecutorMspaceDestroy(u64 handle) {
    {
        std::scoped_lock lk{g_executor_mspace_mutex};
        g_executor_mspaces.erase(handle);
        if (g_executor_default_mspace_handle == handle) {
            g_executor_default_mspace_handle = 0;
        }
        std::scoped_lock alk{g_executor_alloc_mutex};
        for (auto it = g_executor_allocations.begin(); it != g_executor_allocations.end();) {
            if (it->second.mspace_handle == handle) {
                it = g_executor_allocations.erase(it);
            } else {
                ++it;
            }
        }
        g_executor_live_mspace_intervals.erase(handle);
    }
    ExecutorAllocLog(false, 0,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceDestroy handle=%p return=0",
                     reinterpret_cast<void*>(handle));
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceDestroy handle=%p return=0\n",
                     reinterpret_cast<void*>(handle));
        std::fflush(stderr);
    }
    return 0;
}

u64 ExecutorMspaceMalloc(u64 handle, u64 size) {
    void* ptr = nullptr;
    if (!ExecutorAllocFromMspace(handle, alignof(std::max_align_t), size, &ptr)) {
        ptr = ExecutorAlloc(size);
    }
    if (ptr == nullptr) {
        ExecutorAllocFailLog("[EXECUTOR_ALLOC_FAIL] op=sceLibcMspaceMalloc handle=%p size=%llu",
                             reinterpret_cast<void*>(handle),
                             static_cast<unsigned long long>(size));
    }
    ExecutorAllocLog(false, size, "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceMalloc size=%llu return=%p",
                     static_cast<unsigned long long>(size), ptr);
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceMalloc size=%llu return=%p\n",
                     static_cast<unsigned long long>(size), ptr);
        std::fflush(stderr);
    }
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorMspaceCalloc(u64 handle, u64 count, u64 size) {
    if (count != 0 && size > UINT64_MAX / count) {
        ExecutorAllocLog(true, UINT64_MAX,
                         "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceCalloc overflow count=%llu size=%llu return=null",
                         static_cast<unsigned long long>(count),
                         static_cast<unsigned long long>(size));
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceCalloc overflow count=%llu size=%llu return=null\n",
                     static_cast<unsigned long long>(count),
                     static_cast<unsigned long long>(size));
        std::fflush(stderr);
        return 0;
    }
    const u64 bytes = std::max<u64>(count * size, 1);
    void* ptr = nullptr;
    if (!ExecutorAllocFromMspace(handle, alignof(std::max_align_t), bytes, &ptr)) {
        ptr = ExecutorAlloc(bytes, true);
    } else if (ptr != nullptr) {
        std::memset(ptr, 0, static_cast<std::size_t>(bytes));
    }
    if (ptr == nullptr) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=sceLibcMspaceCalloc handle=%p count=%llu size=%llu bytes=%llu",
            reinterpret_cast<void*>(handle), static_cast<unsigned long long>(count),
            static_cast<unsigned long long>(size), static_cast<unsigned long long>(bytes));
    }
    ExecutorAllocLog(false, bytes,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceCalloc count=%llu size=%llu bytes=%llu return=%p",
                     static_cast<unsigned long long>(count), static_cast<unsigned long long>(size),
                     static_cast<unsigned long long>(bytes), ptr);
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceCalloc count=%llu size=%llu bytes=%llu return=%p\n",
                     static_cast<unsigned long long>(count), static_cast<unsigned long long>(size),
                     static_cast<unsigned long long>(bytes), ptr);
        std::fflush(stderr);
    }
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorMspaceRealloc(u64 handle, u64 old_ptr, u64 size) {
    void* old = reinterpret_cast<void*>(old_ptr);
    bool known_old = false;
    std::size_t old_offset = 0;
    void* ptr = ExecutorReallocPossiblyInterior(handle, old, size, alignof(std::max_align_t),
                                                &known_old, &old_offset);
    if (ptr == nullptr) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=sceLibcMspaceRealloc handle=%p old=%p size=%llu known_old=%d old_offset=%zu",
            reinterpret_cast<void*>(handle), reinterpret_cast<void*>(old_ptr),
            static_cast<unsigned long long>(size), known_old ? 1 : 0, old_offset);
    }
    ExecutorAllocLog(!known_old, size,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceRealloc old=%p size=%llu return=%p known_old=%d old_offset=%zu",
                     reinterpret_cast<void*>(old_ptr), static_cast<unsigned long long>(size), ptr,
                     known_old ? 1 : 0, old_offset);
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceRealloc old=%p size=%llu return=%p known_old=%d old_offset=%zu\n",
                 reinterpret_cast<void*>(old_ptr), static_cast<unsigned long long>(size), ptr,
                 known_old ? 1 : 0, old_offset);
    std::fflush(stderr);
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorMspaceReallocalign(u64 handle, u64 old_ptr, u64 size, u64 alignment) {
    void* old = reinterpret_cast<void*>(old_ptr);
    bool known_old = false;
    std::size_t old_offset = 0;
    void* ptr = ExecutorReallocPossiblyInterior(handle, old, size, alignment, &known_old,
                                                &old_offset);
    if (ptr == nullptr) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=sceLibcMspaceReallocalign handle=%p old=%p align=%llu size=%llu known_old=%d old_offset=%zu",
            reinterpret_cast<void*>(handle), reinterpret_cast<void*>(old_ptr),
            static_cast<unsigned long long>(alignment), static_cast<unsigned long long>(size),
            known_old ? 1 : 0, old_offset);
    }
    ExecutorAllocLog(!known_old, size,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceReallocalign old=%p size=%llu align=%llu return=%p known_old=%d old_offset=%zu",
                     reinterpret_cast<void*>(old_ptr), static_cast<unsigned long long>(size),
                     static_cast<unsigned long long>(alignment), ptr, known_old ? 1 : 0,
                     old_offset);
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceReallocalign old=%p size=%llu align=%llu return=%p known_old=%d old_offset=%zu\n",
                 reinterpret_cast<void*>(old_ptr), static_cast<unsigned long long>(size),
                 static_cast<unsigned long long>(alignment), ptr, known_old ? 1 : 0,
                 old_offset);
    std::fflush(stderr);
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorMspaceLock(u64) {
    return 0;
}

u64 ExecutorMspaceUnlock(u64) {
    return 0;
}

u64 ExecutorMspaceFree(u64 handle, u64 ptr) {
    void* value = reinterpret_cast<void*>(ptr);
    ExecutorAllocationInfo info{};
    const bool known = ForgetExecutorAllocation(value, &info, handle);
    if (known) {
        if (!ExecutorRecycleMspaceAllocation(value, info, handle)) {
            ExecutorFree(value, info);
        }
    }
    ExecutorAllocLog(!known && value != nullptr, 0,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceFree ptr=%p known=%d return=0",
                     value, known ? 1 : 0);
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceFree ptr=%p known=%d return=0\n",
                     value, known ? 1 : 0);
        std::fflush(stderr);
    }
    return 0;
}

u64 ExecutorMspaceMemalign(u64 handle, u64 alignment, u64 size) {
    void* ptr = nullptr;
    if (!ExecutorAllocFromMspace(handle, alignment, size, &ptr)) {
        ptr = ExecutorAllocAligned(alignment, size);
    }
    if (ptr == nullptr) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=sceLibcMspaceMemalign handle=%p align=%llu size=%llu",
            reinterpret_cast<void*>(handle), static_cast<unsigned long long>(alignment),
            static_cast<unsigned long long>(size));
    }
    ExecutorAllocLog(false, size,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceMemalign align=%llu size=%llu return=%p",
                     static_cast<unsigned long long>(alignment),
                     static_cast<unsigned long long>(size), ptr);
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceMemalign align=%llu size=%llu return=%p\n",
                     static_cast<unsigned long long>(alignment),
                     static_cast<unsigned long long>(size), ptr);
        std::fflush(stderr);
    }
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorMspaceAlignedAlloc(u64 handle, u64 alignment, u64 size) {
    return ExecutorMspaceMemalign(handle, alignment, size);
}

u64 ExecutorMspacePosixMemalign(u64 handle, u64 out_ptr, u64 alignment, u64 size) {
    if (!ExecutorIsValidPosixAlignment(alignment)) {
        return static_cast<u64>(EINVAL);
    }
    void* ptr = nullptr;
    if (!ExecutorAllocFromMspace(handle, alignment, size, &ptr)) {
        ptr = ExecutorAllocAligned(alignment, size);
    }
    if (ptr == nullptr) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=sceLibcMspacePosixMemalign handle=%p out=%p align=%llu size=%llu",
            reinterpret_cast<void*>(handle), reinterpret_cast<void*>(out_ptr),
            static_cast<unsigned long long>(alignment), static_cast<unsigned long long>(size));
    }
    if (out_ptr) {
        std::memcpy(reinterpret_cast<void*>(out_ptr), &ptr, sizeof(ptr));
    }
    const u64 result = ptr ? 0 : static_cast<u64>(ENOMEM);
    ExecutorAllocLog(false, size,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspacePosixMemalign out=%p align=%llu size=%llu ptr=%p return=%llu",
                     reinterpret_cast<void*>(out_ptr), static_cast<unsigned long long>(alignment),
                     static_cast<unsigned long long>(size), ptr,
                     static_cast<unsigned long long>(result));
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspacePosixMemalign out=%p align=%llu size=%llu ptr=%p return=%llu\n",
                 reinterpret_cast<void*>(out_ptr), static_cast<unsigned long long>(alignment),
                 static_cast<unsigned long long>(size), ptr, static_cast<unsigned long long>(result));
    std::fflush(stderr);
    return result;
}

u64 ExecutorMspaceMallocUsableSize(u64, u64 ptr) {
#if defined(__ANDROID__) || defined(__linux__)
    void* value = reinterpret_cast<void*>(ptr);
    std::size_t size = value ? GetExecutorAllocationUsableSize(value) : 0;
#ifndef __ANDROID__
    if (size == 0 && value && IsExecutorAllocation(value)) {
        size = malloc_usable_size(value);
    }
#endif
#else
    const std::size_t size = 0;
#endif
    ExecutorAllocLog(false, size,
                     "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceMallocUsableSize ptr=%p return=%zu",
                     reinterpret_cast<void*>(ptr), size);
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceMallocUsableSize ptr=%p return=%zu\n",
                 reinterpret_cast<void*>(ptr), size);
    std::fflush(stderr);
    return static_cast<u64>(size);
}

u64 ExecutorMspaceNoopStats() {
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] mspace stats noop return=0\n");
    std::fflush(stderr);
    return 0;
}

u64 ExecutorMspaceNoopTrim(u64, u64 pad) {
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] mspace trim pad=%llu return=0\n",
                 static_cast<unsigned long long>(pad));
    std::fflush(stderr);
    return 0;
}

u64 ExecutorMallocNoopStats() {
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] malloc stats noop return=0\n");
    std::fflush(stderr);
    return 0;
}

u64 ExecutorMspaceIsHeapEmpty(u64) {
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] sceLibcMspaceIsHeapEmpty return=0\n");
    std::fflush(stderr);
    return 0;
}

u64 ExecutorMspaceFooterValue() {
    return 0;
}

static std::atomic<int> g_mono_loadhook_ready{0};
#ifdef __ANDROID__
void ExecutorMaybeInstallMonoLoadHookLogger();
#else
void ExecutorMaybeInstallMonoLoadHookLogger() {}
#endif
void ExecutorMonoFixImageGuid(u64 image);

u64 ExecutorLibcMalloc(u64 size) {
    void* ptr = ExecutorAllocDefaultMspaceOrFallback(alignof(std::max_align_t), size);
    if (g_mono_loadhook_ready.load(std::memory_order_relaxed)) {
        ExecutorMaybeInstallMonoLoadHookLogger();
    }
    if (ptr == nullptr) {
        ExecutorAllocFailLog("[EXECUTOR_ALLOC_FAIL] op=malloc size=%llu",
                             static_cast<unsigned long long>(size));
    }
    ExecutorAllocLog(false, size, "[EXECUTOR_AEROLIB_ALLOC] malloc size=%llu return=%p",
                     static_cast<unsigned long long>(size), ptr);
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] malloc size=%llu return=%p\n",
                     static_cast<unsigned long long>(size), ptr);
        std::fflush(stderr);
    }
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorLibcCalloc(u64 count, u64 size) {
    if (count != 0 && size > UINT64_MAX / count) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=calloc-overflow count=%llu size=%llu return=null",
            static_cast<unsigned long long>(count), static_cast<unsigned long long>(size));
        ExecutorAllocLog(true, UINT64_MAX,
                         "[EXECUTOR_AEROLIB_ALLOC] calloc overflow count=%llu size=%llu return=null",
                         static_cast<unsigned long long>(count),
                         static_cast<unsigned long long>(size));
        return 0;
    }
    const u64 bytes = std::max<u64>(count * size, 1);
    void* ptr = ExecutorAllocDefaultMspaceOrFallback(alignof(std::max_align_t), bytes, true);
    if (ptr == nullptr) {
        ExecutorAllocFailLog("[EXECUTOR_ALLOC_FAIL] op=calloc count=%llu size=%llu bytes=%llu",
                             static_cast<unsigned long long>(count),
                             static_cast<unsigned long long>(size),
                             static_cast<unsigned long long>(bytes));
    }
    ExecutorAllocLog(false, count * size,
                     "[EXECUTOR_AEROLIB_ALLOC] calloc count=%llu size=%llu return=%p",
                     static_cast<unsigned long long>(count), static_cast<unsigned long long>(size),
                     ptr);
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] calloc count=%llu size=%llu return=%p\n",
                     static_cast<unsigned long long>(count),
                     static_cast<unsigned long long>(size), ptr);
        std::fflush(stderr);
    }
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorLibcRealloc(u64 old_ptr, u64 size) {
    void* old = reinterpret_cast<void*>(old_ptr);
    bool known_old = false;
    std::size_t old_offset = 0;
    void* ptr = ExecutorReallocPossiblyInterior(0, old, size, alignof(std::max_align_t),
                                                &known_old, &old_offset);
    if (ptr == nullptr) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=realloc old=%p size=%llu known_old=%d old_offset=%zu",
            old, static_cast<unsigned long long>(size), known_old ? 1 : 0, old_offset);
    }
    ExecutorAllocLog(!known_old, size,
                     "[EXECUTOR_AEROLIB_ALLOC] realloc old=%p size=%llu return=%p known_old=%d old_offset=%zu",
                     old, static_cast<unsigned long long>(size), ptr, known_old ? 1 : 0,
                     old_offset);
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_ALLOC] realloc old=%p size=%llu return=%p known_old=%d old_offset=%zu\n",
                 old, static_cast<unsigned long long>(size), ptr, known_old ? 1 : 0,
                 old_offset);
    std::fflush(stderr);
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorLibcReallocalign(u64 old_ptr, u64 size, u64 alignment) {
    void* old = reinterpret_cast<void*>(old_ptr);
    bool known_old = false;
    std::size_t old_offset = 0;
    void* ptr =
        ExecutorReallocPossiblyInterior(0, old, size, alignment, &known_old, &old_offset);
    if (ptr == nullptr) {
        ExecutorAllocFailLog(
            "[EXECUTOR_ALLOC_FAIL] op=reallocalign old=%p align=%llu size=%llu known_old=%d old_offset=%zu",
            old, static_cast<unsigned long long>(alignment),
            static_cast<unsigned long long>(size), known_old ? 1 : 0, old_offset);
    }
    ExecutorAllocLog(!known_old, size,
                     "[EXECUTOR_AEROLIB_ALLOC] reallocalign old=%p size=%llu align=%llu return=%p known_old=%d old_offset=%zu",
                     old, static_cast<unsigned long long>(size),
                     static_cast<unsigned long long>(alignment), ptr, known_old ? 1 : 0,
                     old_offset);
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_ALLOC] reallocalign old=%p size=%llu align=%llu return=%p known_old=%d old_offset=%zu\n",
                 old, static_cast<unsigned long long>(size),
                 static_cast<unsigned long long>(alignment), ptr, known_old ? 1 : 0, old_offset);
    std::fflush(stderr);
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorLibcFree(u64 ptr) {
    void* value = reinterpret_cast<void*>(ptr);
    ExecutorAllocationInfo info{};
    const bool known = ForgetExecutorAllocation(value, &info);
    if (known) {
        if (!ExecutorRecycleMspaceAllocation(value, info)) {
            ExecutorFree(value, info);
        }
    }
    ExecutorAllocLog(!known && value != nullptr, 0,
                     "[EXECUTOR_AEROLIB_ALLOC] free ptr=%p known=%d return=0", value,
                     known ? 1 : 0);
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] free ptr=%p known=%d return=0\n", value,
                     known ? 1 : 0);
        std::fflush(stderr);
    }
    return 0;
}

#ifdef __ANDROID__
static bool ExecutorLibcCopyTraceEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("EXECUTOR_TRACE_LIBC_COPY");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool ExecutorLooksLikeGuestDataPtr(u64 ptr) {
    return ptr >= 0x200000000ull && ptr < 0x810000000ull;
}

static bool ExecutorCopyRangeHasAsciiToken(u64 ptr, u64 size) {
    if (ptr < 0x10000ull || ptr >= 0x810000000ull || size == 0) {
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(std::min<u64>(size, 0x180));
    const auto* bytes = reinterpret_cast<const unsigned char*>(ptr);
    auto has_token = [&](const char* token) {
        const std::size_t token_len = std::strlen(token);
        if (token_len == 0 || token_len > n) {
            return false;
        }
        for (std::size_t i = 0; i + token_len <= n; ++i) {
            if (std::memcmp(bytes + i, token, token_len) == 0) {
                return true;
            }
        }
        return false;
    };
    return has_token("/app0/") || has_token("Save") || has_token("SavedGam") ||
           has_token("mainData") || has_token("Unity") || has_token("library/");
}

static std::string ExecutorCopyBytesHex(u64 ptr, u64 size) {
    if (ptr < 0x10000ull || ptr >= 0x810000000ull || size == 0) {
        return {};
    }
    const std::size_t n = static_cast<std::size_t>(std::min<u64>(size, 0x80));
    const auto* bytes = reinterpret_cast<const unsigned char*>(ptr);
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        out.push_back(kHex[bytes[i] >> 4]);
        out.push_back(kHex[bytes[i] & 0xf]);
    }
    return out;
}

static void ExecutorMaybeLogLibcCopy(const char* op, u64 dest, u64 src, u64 size) {
    if (!ExecutorLibcCopyTraceEnabled()) {
        return;
    }
    std::uint64_t guest_return = 0;
    std::uint64_t return_off = 0;
    std::uint64_t call_arg0 = 0;
    char symbol[128] = {};
    char module[160] = {};
    if (executor_live_get_current_hle_call_site != nullptr) {
        (void)executor_live_get_current_hle_call_site(&guest_return, &return_off, &call_arg0,
                                                      symbol, sizeof(symbol), module,
                                                      sizeof(module));
    }

    const bool token = ExecutorCopyRangeHasAsciiToken(dest, size) ||
                       ExecutorCopyRangeHasAsciiToken(src, size);
    const bool focused_return =
        (return_off >= 0x50000ull && return_off < 0x54000ull) ||
        (return_off >= 0x90000ull && return_off < 0xa1000ull) ||
        (return_off >= 0x190000ull && return_off < 0x1a1000ull);
    const bool small_heap_copy =
        ExecutorLooksLikeGuestDataPtr(dest) && size != 0 && size <= 0x400ull;
    static std::atomic_int token_budget{256};
    static std::atomic_int focused_budget{512};
    static std::atomic_int sample_budget{128};
    std::atomic_int* budget = nullptr;
    if (token) {
        budget = &token_budget;
    } else if (focused_return && small_heap_copy) {
        budget = &focused_budget;
    } else if (small_heap_copy && sample_budget.load(std::memory_order_relaxed) > 0) {
        budget = &sample_budget;
    }
    if (budget == nullptr) {
        return;
    }
    if (budget->fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    const std::string dst_hex = ExecutorCopyBytesHex(dest, size);
    const std::string src_hex = ExecutorCopyBytesHex(src, size);
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIBC_COPY] op=%s dest=0x%llx src=0x%llx size=0x%llx token=%d "
        "guestRet=0x%llx retOff=0x%llx symbol=%s module=%s dst=%s src=%s",
        op != nullptr ? op : "copy", static_cast<unsigned long long>(dest),
        static_cast<unsigned long long>(src), static_cast<unsigned long long>(size),
        token ? 1 : 0, static_cast<unsigned long long>(guest_return),
        static_cast<unsigned long long>(return_off), symbol[0] ? symbol : "?",
        module[0] ? module : "?", dst_hex.c_str(), src_hex.c_str());
}
#else
static void ExecutorMaybeLogLibcCopy(const char*, u64, u64, u64) {}
#endif

static inline void ExecutorWatchGateWrite(const char* who, u64 dest, u64 size, u64 val) {
#ifdef __ANDROID__
    if (dest < 0x800345ac0ull && dest + size > 0x8003452b0ull) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_GATE_WRITE] who=%s dest=0x%llx size=%llu val=0x%llx "
                            "covers_0x8003452d0=%d",
                            who, static_cast<unsigned long long>(dest),
                            static_cast<unsigned long long>(size),
                            static_cast<unsigned long long>(val),
                            (dest <= 0x8003452d0ull && dest + size > 0x8003452d0ull) ? 1 : 0);
    }
#else
    (void)who; (void)dest; (void)size; (void)val;
#endif
}

static bool ExecutorLibcOwnHostImageRange(u64& lo, u64& hi) {
#if defined(__ANDROID__) || defined(__linux__)
    static u64 s_lo = 0, s_hi = 0;
    static std::once_flag s_once;
    std::call_once(s_once, [] {
        const u64 self_addr = reinterpret_cast<u64>(&ExecutorLibcOwnHostImageRange);
        std::ifstream maps("/proc/self/maps");
        std::string line, self_path;
        while (std::getline(maps, line)) {
            unsigned long long b = 0, e = 0;
            char perms[8] = {};
            int poff = 0;
            if (std::sscanf(line.c_str(), "%llx-%llx %7s %*x %*x:%*x %*u %n", &b, &e, perms, &poff) < 3)
                continue;
            if (self_addr >= b && self_addr < e && poff > 0 &&
                static_cast<std::size_t>(poff) < line.size()) {
                self_path = line.substr(static_cast<std::size_t>(poff));
                while (!self_path.empty() && (self_path.back() == '\n' || self_path.back() == ' '))
                    self_path.pop_back();
                break;
            }
        }
        if (self_path.empty())
            return;
        std::ifstream maps2("/proc/self/maps");
        u64 mn = ~u64{0}, mx = 0;
        while (std::getline(maps2, line)) {
            if (line.find(self_path) == std::string::npos)
                continue;
            unsigned long long b = 0, e = 0;
            if (std::sscanf(line.c_str(), "%llx-%llx", &b, &e) != 2)
                continue;
            mn = std::min<u64>(mn, b);
            mx = std::max<u64>(mx, e);
        }
        if (mx > mn) { s_lo = mn; s_hi = mx; }
    });
    lo = s_lo; hi = s_hi;
    return s_hi > s_lo;
#else
    (void)lo; (void)hi; return false;
#endif
}

static bool ExecutorLibcRejectHostImageWrite(const char* op, u64 dest, u64 size) {
    u64 lo = 0, hi = 0;
    if (size == 0 || !ExecutorLibcOwnHostImageRange(lo, hi))
        return false;
    if (dest >= hi || (dest + size) <= lo)
        return false;
    static std::atomic<int> budget{128};
    if (budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        LOG_ERROR(Core,
                  "EXECUTOR_LIBC_WRITE_INTO_HOST_IMAGE op={} dest={:#x} size={:#x} image=[{:#x},{:#x}) "
                  "ret={} (REJECTED; guest passed a garbage host-address dest)",
                  op, dest, size, lo, hi,
                  reinterpret_cast<u64>(__builtin_return_address(0)));
        std::FILE* cf = std::fopen(
            "/data/data/app.lsx4.android/files/lsx4-home/jit-checkfail.txt", "a");
        if (cf != nullptr) {
            std::fprintf(cf, "LIBC_WRITE_INTO_HOST_IMAGE op=%s dest=0x%llx size=0x%llx\n", op,
                         static_cast<unsigned long long>(dest), static_cast<unsigned long long>(size));
            std::fclose(cf);
        }
    }
    return true;
}

extern "C" std::uint64_t ExecutorGetLastHleGuestRsp();

u64 ExecutorLibcMemcpy(u64 dest, u64 src, u64 size) {
    if (ExecutorLibcRejectHostImageWrite("memcpy", dest, size))
        return dest;
#ifdef __ANDROID__
    constexpr u64 kMaximumSingleGuestLibcCopy = 1ull << 30;
    if (size >= kMaximumSingleGuestLibcCopy) {
        std::uint64_t guest_return = 0;
        std::uint64_t return_off = 0;
        std::uint64_t call_arg0 = 0;
        char symbol[128] = {};
        char module[160] = {};
        if (executor_live_get_current_hle_call_site != nullptr) {
            (void)executor_live_get_current_hle_call_site(
                &guest_return, &return_off, &call_arg0, symbol, sizeof(symbol), module,
                sizeof(module));
        }
        const std::uint64_t guest_rsp = ExecutorGetLastHleGuestRsp();
        if (guest_return == 0 && guest_rsp >= 0x10000) {
            std::memcpy(&guest_return, reinterpret_cast<const void*>(guest_rsp),
                        sizeof(guest_return));
        }
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4Native",
            "[EXECUTOR_LIBC_INVALID_COPY] op=memcpy reason=oversized dest=0x%llx "
            "src=0x%llx size=0x%llx guestRsp=0x%llx guestRet=0x%llx retOff=0x%llx "
            "symbol=%s module=%s",
            static_cast<unsigned long long>(dest), static_cast<unsigned long long>(src),
            static_cast<unsigned long long>(size),
            static_cast<unsigned long long>(guest_rsp),
            static_cast<unsigned long long>(guest_return),
            static_cast<unsigned long long>(return_off), symbol[0] ? symbol : "?",
            module[0] ? module : "?");
        return dest;
    }
    if (size != 0 && (dest < 0x10000ull || src < 0x10000ull)) {
        std::uint64_t guest_return = 0;
        std::uint64_t return_off = 0;
        std::uint64_t call_arg0 = 0;
        char symbol[128] = {};
        char module[160] = {};
        if (executor_live_get_current_hle_call_site != nullptr) {
            (void)executor_live_get_current_hle_call_site(
                &guest_return, &return_off, &call_arg0, symbol, sizeof(symbol), module,
                sizeof(module));
        }
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4Native",
            "[EXECUTOR_LIBC_INVALID_COPY] op=memcpy dest=0x%llx src=0x%llx size=0x%llx "
            "guestRet=0x%llx retOff=0x%llx symbol=%s module=%s",
            static_cast<unsigned long long>(dest), static_cast<unsigned long long>(src),
            static_cast<unsigned long long>(size),
            static_cast<unsigned long long>(guest_return),
            static_cast<unsigned long long>(return_off), symbol[0] ? symbol : "?",
            module[0] ? module : "?");
    }
#endif
    VideoCore::PageManager::PrepareBulkGuestRead(src, size);
    VideoCore::PageManager::PrepareBulkGuestWrite(dest, size);
    VideoCore::PageManager::EnterBulkGuestWrite(dest, size);
    VideoCore::BeginBulkGuestWrite(dest, size);
    std::memcpy(reinterpret_cast<void*>(dest), reinterpret_cast<const void*>(src),
                static_cast<std::size_t>(size));
    VideoCore::PageManager::LeaveBulkGuestWrite(dest, size);
    VideoCore::EndBulkGuestWrite();
    ExecutorWatchGateWrite("memcpy", dest, size,
                           size >= 8 ? *reinterpret_cast<const u64*>(src) : 0);
    ExecutorMaybeLogLibcCopy("memcpy", dest, src, size);
    return dest;
}

u64 ExecutorLibcMemmove(u64 dest, u64 src, u64 size) {
    if (ExecutorLibcRejectHostImageWrite("memmove", dest, size))
        return dest;
    VideoCore::PageManager::PrepareBulkGuestRead(src, size);
    VideoCore::PageManager::PrepareBulkGuestWrite(dest, size);
    VideoCore::PageManager::EnterBulkGuestWrite(dest, size);
    VideoCore::BeginBulkGuestWrite(dest, size);
    std::memmove(reinterpret_cast<void*>(dest), reinterpret_cast<const void*>(src),
                 static_cast<std::size_t>(size));
    VideoCore::PageManager::LeaveBulkGuestWrite(dest, size);
    VideoCore::EndBulkGuestWrite();
    ExecutorWatchGateWrite("memmove", dest, size,
                           size >= 8 ? *reinterpret_cast<const u64*>(dest) : 0);
    ExecutorMaybeLogLibcCopy("memmove", dest, src, size);
    return dest;
}

std::atomic<int> g_memset_guard_log{300};
std::atomic<long long> g_memset_guard_hits{0};
u64 ExecutorLibcMemset(u64 dest, u64 val, u64 size) {
    if (ExecutorLibcRejectHostImageWrite("memset", dest, size))
        return dest;
    const std::uint8_t fill = static_cast<std::uint8_t>(val & 0xffu);
    ExecutorWatchGateWrite("memset", dest, size, fill);
    if (ExecutorMemsetVtableGuardEnabled() && fill == 0 && dest >= 0x200000000ull &&
        dest < 0x220000000ull && size >= 8 && size <= 0x20000ull) {
        std::uintptr_t a = dest;
        const std::uintptr_t e = dest + size;
        for (; a + 8 <= e; a += 8) {
            std::uint64_t w = 0;
            std::memcpy(&w, reinterpret_cast<const void*>(a), 8);
            if (w >= 0x200000000ull && w < 0x220000000ull) {
                std::uint64_t klass = 0;
                std::memcpy(&klass, reinterpret_cast<const void*>(w), 8);
                if (klass >= 0x200000000ull && klass < 0x810000000ull) {
                    g_memset_guard_hits.fetch_add(1, std::memory_order_relaxed);
                    if (g_memset_guard_log.fetch_sub(1, std::memory_order_relaxed) > 0) {
#ifdef __ANDROID__
                        __android_log_print(
                            ANDROID_LOG_ERROR, "LSX4Native",
                            "[MEMSET_GUARD] protected vtable=0x%llx @0x%llx in memset(dest=0x%llx "
                            "size=0x%llx)",
                            (unsigned long long)w, (unsigned long long)a, (unsigned long long)dest,
                            (unsigned long long)size);
#else
                        std::fprintf(stderr,
                                     "[MEMSET_GUARD] protected vtable=0x%llx @0x%llx in "
                                     "memset(dest=0x%llx size=0x%llx)\n",
                                     (unsigned long long)w, (unsigned long long)a,
                                     (unsigned long long)dest, (unsigned long long)size);
                        std::fflush(stderr);
#endif
                    }
                    continue;
                }
            }
            *reinterpret_cast<volatile std::uint64_t*>(a) = 0;
        }
        for (; a < e; ++a) {
            *reinterpret_cast<volatile std::uint8_t*>(a) = 0;
        }
        return dest;
    }
    VideoCore::PageManager::PrepareBulkGuestWrite(dest, size);
    VideoCore::PageManager::EnterBulkGuestWrite(dest, size);
    VideoCore::BeginBulkGuestWrite(dest, size);
    std::memset(reinterpret_cast<void*>(dest), fill, static_cast<std::size_t>(size));
    VideoCore::PageManager::LeaveBulkGuestWrite(dest, size);
    VideoCore::EndBulkGuestWrite();
    return dest;
}

extern "C" void executor_lsx4_register_guest_gc_roots() __attribute__((weak));

u64 ExecutorLibcMemalign(u64 alignment, u64 size) {
    void* ptr = ExecutorAllocDefaultMspaceOrFallback(alignment, size);
    if (executor_lsx4_register_guest_gc_roots) {
        static std::atomic<unsigned> memalign_calls{0};
        if ((memalign_calls.fetch_add(1, std::memory_order_relaxed) & 0xf) == 0) {
            executor_lsx4_register_guest_gc_roots();
        }
    }
    if (ptr == nullptr) {
        ExecutorAllocFailLog("[EXECUTOR_ALLOC_FAIL] op=memalign align=%llu size=%llu",
                             static_cast<unsigned long long>(alignment),
                             static_cast<unsigned long long>(size));
    }
    ExecutorAllocLog(false, size,
                     "[EXECUTOR_AEROLIB_ALLOC] memalign align=%llu size=%llu return=%p",
                     static_cast<unsigned long long>(alignment),
                     static_cast<unsigned long long>(size), ptr);
#ifdef __ANDROID__
    if (ExecutorVerboseAerolibAllocLogsEnabled())
#endif
    {
        std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] memalign align=%llu size=%llu return=%p\n",
                     static_cast<unsigned long long>(alignment),
                     static_cast<unsigned long long>(size), ptr);
        std::fflush(stderr);
    }
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorLibcPosixMemalign(u64 out_ptr, u64 alignment, u64 size) {
    if (!ExecutorIsValidPosixAlignment(alignment)) {
        return static_cast<u64>(EINVAL);
    }
    void* ptr = ExecutorAllocDefaultMspaceOrFallback(alignment, size);
    if (ptr == nullptr) {
        ExecutorAllocFailLog("[EXECUTOR_ALLOC_FAIL] op=posix_memalign out=%p align=%llu size=%llu",
                             reinterpret_cast<void*>(out_ptr),
                             static_cast<unsigned long long>(alignment),
                             static_cast<unsigned long long>(size));
    }
    if (out_ptr) {
        std::memcpy(reinterpret_cast<void*>(out_ptr), &ptr, sizeof(ptr));
    }
    const u64 result = ptr ? 0 : static_cast<u64>(ENOMEM);
    ExecutorAllocLog(false, size,
                     "[EXECUTOR_AEROLIB_ALLOC] posix_memalign out=%p align=%llu size=%llu ptr=%p return=%llu",
                     reinterpret_cast<void*>(out_ptr), static_cast<unsigned long long>(alignment),
                     static_cast<unsigned long long>(size), ptr,
                     static_cast<unsigned long long>(result));
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_ALLOC] posix_memalign out=%p align=%llu size=%llu ptr=%p return=%llu\n",
                 reinterpret_cast<void*>(out_ptr), static_cast<unsigned long long>(alignment),
                 static_cast<unsigned long long>(size), ptr, static_cast<unsigned long long>(result));
    std::fflush(stderr);
    return result;
}

u64 ExecutorLibcMallocUsableSize(u64 ptr) {
#if defined(__ANDROID__) || defined(__linux__)
    void* value = reinterpret_cast<void*>(ptr);
    const std::size_t size = value ? GetExecutorAllocationUsableSize(value) : 0;
#else
    const std::size_t size = 0;
#endif
    ExecutorAllocLog(false, size, "[EXECUTOR_AEROLIB_ALLOC] malloc_usable_size ptr=%p return=%zu",
                     reinterpret_cast<void*>(ptr), size);
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] malloc_usable_size ptr=%p return=%zu\n",
                 reinterpret_cast<void*>(ptr), size);
    std::fflush(stderr);
    return static_cast<u64>(size);
}

#ifdef __ANDROID__
static u64 ExecutorUntagHostPointer(u64 ptr) {
#if defined(__aarch64__)
    constexpr u64 TopByteMask = 0xff00000000000000ull;
    constexpr u64 CanonicalMask = 0x00ffffffffffffffull;
    if ((ptr & TopByteMask) != 0) {
        return ptr & CanonicalMask;
    }
#endif
    return ptr;
}

static std::size_t ExecutorHostPageSize() {
    static const std::size_t page_size = [] {
        const long value = sysconf(_SC_PAGESIZE);
        return value > 0 ? static_cast<std::size_t>(value) : std::size_t{4096};
    }();
    return page_size;
}

static bool ExecutorDirectReadableGuestAddress(u64 ptr) {
    if (ptr == 0) {
        return false;
    }
    ptr = ExecutorUntagHostPointer(ptr);
    const std::size_t page_size = ExecutorHostPageSize();
    const auto page = static_cast<uintptr_t>(ptr) & ~(static_cast<uintptr_t>(page_size) - 1u);
    unsigned char vec = 0;
    return mincore(reinterpret_cast<void*>(page), page_size, &vec) == 0;
}

static bool ExecutorDirectReadableGuestRange(u64 ptr, u64 size) {
    if (ptr == 0) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    ptr = ExecutorUntagHostPointer(ptr);
    if (size - 1 > std::numeric_limits<u64>::max() - ptr) {
        return false;
    }

    const std::size_t page_size = ExecutorHostPageSize();
    const u64 page_mask = static_cast<u64>(page_size - 1);
    const u64 first_page = ptr & ~page_mask;
    const u64 last_page = (ptr + size - 1) & ~page_mask;
    for (u64 page = first_page;; page += page_size) {
        unsigned char vec = 0;
        if (mincore(reinterpret_cast<void*>(page), page_size, &vec) != 0) {
            return false;
        }
        if (page == last_page) {
            return true;
        }
        if (page > std::numeric_limits<u64>::max() - page_size) {
            return false;
        }
    }
}

struct ExecutorGuestReadSpan {
    const unsigned char* data{};
    std::size_t size{};
    bool direct{};
};

static bool ExecutorFindHostReadableRange(u64 address, u64& range_begin, u64& range_end) {
    range_begin = 0;
    range_end = 0;
    if (address == 0) {
        return false;
    }
#if defined(__ANDROID__) || defined(__linux__)
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long long begin = 0;
        unsigned long long end = 0;
        char perms[8]{};
        if (std::sscanf(line.c_str(), "%llx-%llx %7s", &begin, &end, perms) != 3) {
            continue;
        }
        if (address >= begin && address < end) {
            if (perms[0] != 'r') {
                return false;
            }
            range_begin = static_cast<u64>(begin);
            range_end = static_cast<u64>(end);
            return range_end > range_begin;
        }
    }
    return false;
#else
    range_begin = address;
    range_end = std::numeric_limits<u64>::max();
    return true;
#endif
}

static bool ExecutorHostPageReadable(u64 address) {
#if (defined(__ANDROID__) || defined(__linux__)) && defined(SYS_process_vm_readv)
    unsigned char byte = 0;
    iovec local{
        .iov_base = &byte,
        .iov_len = sizeof(byte),
    };
    iovec remote{
        .iov_base = reinterpret_cast<void*>(address),
        .iov_len = sizeof(byte),
    };
    errno = 0;
    const long result = syscall(SYS_process_vm_readv, getpid(), &local, 1, &remote, 1, 0);
    if (result == 1) {
        return true;
    }
    if (errno != ENOSYS && errno != EPERM && errno != EACCES) {
        return false;
    }
#endif
    u64 range_begin = 0;
    u64 range_end = 0;
    return ExecutorFindHostReadableRange(address, range_begin, range_end);
}

static std::size_t ExecutorTrackedFallbackReadablePrefix(u64 address, std::size_t requested) {
    std::scoped_lock lk{g_executor_alloc_mutex};
    const auto owner = g_executor_live_mspace_intervals.find(0);
    if (owner == g_executor_live_mspace_intervals.end()) {
        return 0;
    }
    const auto& intervals = owner->second;
    auto next = intervals.upper_bound(address);
    if (next == intervals.begin()) {
        return 0;
    }
    const auto allocation = std::prev(next);
    const u64 begin = allocation->first;
    const u64 size = allocation->second;
    if (size == 0 || size > std::numeric_limits<u64>::max() - begin || address < begin ||
        address >= begin + size) {
        return 0;
    }
    return static_cast<std::size_t>(
        std::min<u64>(requested, begin + size - address));
}

class ExecutorGuestPageReader {
public:
    bool ReadSpan(u64 ptr, u64 requested, ExecutorGuestReadSpan& out) {
        out = {};
        if (ptr == 0 || requested == 0) {
            return false;
        }

        const std::size_t page_size = ExecutorHostPageSize();
        const u64 page_mask = static_cast<u64>(page_size - 1);
        const u64 host_ptr = ExecutorUntagHostPointer(ptr);
        const std::size_t page_remaining =
            page_size - static_cast<std::size_t>(host_ptr & page_mask);
        std::size_t candidate = static_cast<std::size_t>(
            std::min<u64>(requested, static_cast<u64>(page_remaining)));
        constexpr u64 GuestAddressLimit = 0x10000000000ull;
        if (ExecutorJitIsReadableGuestRange != nullptr) {
            const bool complete_registered_span =
                ExecutorJitIsReadableGuestRange(host_ptr, candidate);
            if (complete_registered_span) {
                out = {
                    .data = reinterpret_cast<const unsigned char*>(host_ptr),
                    .size = candidate,
                    .direct = true,
                };
                return true;
            }
            if (candidate > 1 &&
                ExecutorJitIsReadableGuestRange(host_ptr, 1)) {
                std::size_t low = 1;
                std::size_t high = candidate;
                while (low < high) {
                    const std::size_t mid = low + (high - low + 1) / 2;
                    if (ExecutorJitIsReadableGuestRange(host_ptr, mid)) {
                        low = mid;
                    } else {
                        high = mid - 1;
                    }
                }
                out = {
                    .data = reinterpret_cast<const unsigned char*>(host_ptr),
                    .size = low,
                    .direct = true,
                };
                return true;
            }
        }
        decltype(Core::Memory::Instance()) memory = nullptr;
        bool guest_managed = false;
        bool guest_readable = false;
        if (host_ptr < GuestAddressLimit) {
            memory = Core::Memory::Instance();
            if (memory != nullptr) {
                guest_readable =
                    memory->IsReadableMapping(ptr, candidate, &guest_managed);
            }
        }
        if (!guest_managed) {
            const std::size_t tracked_prefix =
                ExecutorTrackedFallbackReadablePrefix(host_ptr, candidate);
            if (tracked_prefix != 0) {
                out = {
                    .data = reinterpret_cast<const unsigned char*>(host_ptr),
                    .size = tracked_prefix,
                    .direct = true,
                };
                return true;
            }
            if (host_ptr < GuestAddressLimit) {
                return false;
            }
            if (!ExecutorHostPageReadable(host_ptr)) {
                return false;
            }
            out = {
                .data = reinterpret_cast<const unsigned char*>(host_ptr),
                .size = candidate,
                .direct = true,
            };
            return true;
        }

        if (!guest_readable) {
            bool first_byte_managed = false;
            if (memory == nullptr ||
                !memory->IsReadableMapping(ptr, 1, &first_byte_managed) ||
                !first_byte_managed) {
                return false;
            }

            std::size_t low = 1;
            std::size_t high = candidate;
            while (low < high) {
                const std::size_t mid = low + (high - low + 1) / 2;
                bool prefix_managed = false;
                if (memory->IsReadableMapping(ptr, mid, &prefix_managed) && prefix_managed) {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            candidate = low;
            guest_readable = true;
        }

        if (guest_readable) {
            out = {
                .data = reinterpret_cast<const unsigned char*>(host_ptr),
                .size = candidate,
                .direct = true,
            };
            return true;
        }

        return false;
    }
};

extern "C" bool ExecutorAeroLibGuestPageReaderSelfTest(const u64 address,
                                                        const std::size_t size) {
    if (address == 0 || size == 0 || size == std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    const u64 host_address = ExecutorUntagHostPointer(address);
    const std::size_t page_size = ExecutorHostPageSize();
    const std::size_t page_remaining =
        page_size - static_cast<std::size_t>(host_address & (page_size - 1));
    if (size + 1 > page_remaining) {
        return false;
    }

    ExecutorGuestPageReader reader;
    ExecutorGuestReadSpan exact{};
    if (!reader.ReadSpan(address, size, exact) || !exact.direct || exact.size != size ||
        exact.data != reinterpret_cast<const unsigned char*>(host_address)) {
        return false;
    }
    ExecutorGuestReadSpan prefix{};
    return reader.ReadSpan(address, size + 1, prefix) && prefix.direct &&
           prefix.size == size &&
           prefix.data == reinterpret_cast<const unsigned char*>(host_address);
}

enum class ExecutorGuestCStringStatus : u8 {
    Terminated,
    Fault,
    AddressOverflow,
};

struct ExecutorGuestCStringScan {
    u64 length{};
    u64 fault_address{};
    ExecutorGuestCStringStatus status{ExecutorGuestCStringStatus::Fault};
    bool all_direct{true};
};

static ExecutorGuestCStringScan ExecutorScanGuestCString(u64 str) {
    ExecutorGuestCStringScan result{};
    if (str == 0) {
        result.fault_address = 0;
        return result;
    }

    ExecutorGuestPageReader reader;
    while (true) {
        if (result.length > std::numeric_limits<u64>::max() - str) {
            result.fault_address = std::numeric_limits<u64>::max();
            result.status = ExecutorGuestCStringStatus::AddressOverflow;
            return result;
        }
        const u64 address = str + result.length;
        const u64 remaining = std::numeric_limits<u64>::max() - address + 1;
        ExecutorGuestReadSpan span{};
        if (remaining == 0 || !reader.ReadSpan(address, remaining, span) || span.size == 0) {
            result.fault_address = address;
            return result;
        }
        result.all_direct = result.all_direct && span.direct;
        if (const void* nul = std::memchr(span.data, 0, span.size); nul != nullptr) {
            result.length += static_cast<u64>(
                static_cast<const unsigned char*>(nul) - span.data);
            result.status = ExecutorGuestCStringStatus::Terminated;
            return result;
        }
        result.length += static_cast<u64>(span.size);
    }
}

static bool ExecutorReadGuestByte(u64 ptr, unsigned char& out) {
    ExecutorGuestPageReader reader;
    ExecutorGuestReadSpan span{};
    if (!reader.ReadSpan(ptr, 1, span) || span.size != 1) {
        return false;
    }
    out = span.data[0];
    return true;
}

static bool ExecutorReadGuestBytes(u64 ptr, void* out, u64 size) {
    if (ptr == 0 || !out) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    if (size - 1 > std::numeric_limits<u64>::max() - ptr) {
        return false;
    }

    ExecutorGuestPageReader reader;
    auto* destination = static_cast<unsigned char*>(out);
    u64 copied = 0;
    while (copied < size) {
        ExecutorGuestReadSpan span{};
        if (!reader.ReadSpan(ptr + copied, size - copied, span) || span.size == 0) {
            return false;
        }
        std::memcpy(destination + static_cast<std::size_t>(copied), span.data, span.size);
        copied += static_cast<u64>(span.size);
    }
    return true;
}

static bool ExecutorWriteGuestByte(u64 ptr, unsigned char value) {
    if (ptr == 0) {
        return false;
    }
    if (ExecutorDirectReadableGuestAddress(ptr)) {
        *reinterpret_cast<volatile unsigned char*>(ExecutorUntagHostPointer(ptr)) = value;
        return true;
    }
    auto* memory = Core::Memory::Instance();
    if (memory && memory->IsValidMapping(ptr, 1)) {
        const u8 byte = static_cast<u8>(value);
        return memory->TryWriteBacking(reinterpret_cast<void*>(ptr), &byte, sizeof(byte));
    }
    return false;
}

static bool ExecutorReadGuestQword(u64 ptr, u64& out) {
    return ExecutorReadGuestBytes(ptr, &out, sizeof(out));
}

static bool ExecutorWriteGuestQword(u64 ptr, u64 value) {
    if (ptr == 0) {
        return false;
    }
    if (ExecutorDirectReadableGuestAddress(ptr) &&
        ExecutorDirectReadableGuestAddress(ptr + sizeof(value) - 1)) {
        *reinterpret_cast<volatile u64*>(ExecutorUntagHostPointer(ptr)) = value;
        return true;
    }
    auto* memory = Core::Memory::Instance();
    if (memory && memory->IsValidMapping(ptr, sizeof(value))) {
        return memory->TryWriteBacking(reinterpret_cast<void*>(ptr), &value, sizeof(value));
    }
    return false;
}

static bool ExecutorReadableGuestAddress(u64 ptr) {
    unsigned char ignored = 0;
    return ExecutorReadGuestByte(ptr, ignored);
}

static u64 ExecutorCanonicalizeTruncatedMspaceCString(const char* func, u64 ptr) {
    if (ptr == 0 || ptr > 0xffffffffull || ExecutorReadableGuestAddress(ptr)) {
        return ptr;
    }

    constexpr u64 MspaceHighBits = 0x200000000ull;
    const u64 candidate = MspaceHighBits | ptr;
    unsigned char first = 0;
    if (!ExecutorReadGuestByte(candidate, first) ||
        (first != 0 && !std::isprint(static_cast<unsigned char>(first)))) {
        return ptr;
    }

    static std::atomic<int> log_budget{128};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIBC_PTR_CANON] func=%s ptr=%p fixed=%p first=0x%02x",
                            func ? func : "?", reinterpret_cast<void*>(ptr),
                            reinterpret_cast<void*>(candidate), first);
    }
    return candidate;
}

static void ExecutorLogInvalidLibcPtr(const char* func, u64 ptr) {
    static std::atomic<int> log_budget{128};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIBC_INVALID_PTR] func=%s ptr=%p", func,
                            reinterpret_cast<void*>(ptr));
    }
}

static int ExecutorSafeStrcmp(const char* func, u64 lhs, u64 rhs, u64 limit, bool bounded) {
    lhs = ExecutorCanonicalizeTruncatedMspaceCString(func, lhs);
    rhs = ExecutorCanonicalizeTruncatedMspaceCString(func, rhs);
    if (lhs == 0 || rhs == 0) {
        return lhs == rhs ? 0 : (lhs ? 1 : -1);
    }
    ExecutorGuestPageReader lhs_reader;
    ExecutorGuestPageReader rhs_reader;
    u64 compared = 0;
    while (!bounded || compared < limit) {
        if (compared > std::numeric_limits<u64>::max() - lhs) {
            ExecutorLogInvalidLibcPtr(func, std::numeric_limits<u64>::max());
            return 1;
        }
        if (compared > std::numeric_limits<u64>::max() - rhs) {
            ExecutorLogInvalidLibcPtr(func, std::numeric_limits<u64>::max());
            return -1;
        }

        const u64 lhs_address = lhs + compared;
        const u64 rhs_address = rhs + compared;
        u64 requested = bounded ? limit - compared : std::numeric_limits<u64>::max();
        requested = std::min(requested, std::numeric_limits<u64>::max() - lhs_address + 1);
        requested = std::min(requested, std::numeric_limits<u64>::max() - rhs_address + 1);
        ExecutorGuestReadSpan lhs_span{};
        if (requested == 0 || !lhs_reader.ReadSpan(lhs_address, requested, lhs_span) ||
            lhs_span.size == 0) {
            ExecutorLogInvalidLibcPtr(func, lhs_address);
            return 1;
        }
        ExecutorGuestReadSpan rhs_span{};
        if (!rhs_reader.ReadSpan(rhs_address, requested, rhs_span) || rhs_span.size == 0) {
            ExecutorLogInvalidLibcPtr(func, rhs_address);
            return -1;
        }
        const std::size_t common = std::min(lhs_span.size, rhs_span.size);
        const auto* lhs_nul = static_cast<const unsigned char*>(
            std::memchr(lhs_span.data, 0, common));
        const auto* rhs_nul = static_cast<const unsigned char*>(
            std::memchr(rhs_span.data, 0, common));
        std::size_t semantic_size = common;
        if (lhs_nul != nullptr) {
            semantic_size = std::min(
                semantic_size, static_cast<std::size_t>(lhs_nul - lhs_span.data) + 1);
        }
        if (rhs_nul != nullptr) {
            semantic_size = std::min(
                semantic_size, static_cast<std::size_t>(rhs_nul - rhs_span.data) + 1);
        }

        if (std::memcmp(lhs_span.data, rhs_span.data, semantic_size) != 0) {
            for (std::size_t i = 0; i < semantic_size; ++i) {
                if (lhs_span.data[i] != rhs_span.data[i]) {
                    return static_cast<int>(lhs_span.data[i]) -
                           static_cast<int>(rhs_span.data[i]);
                }
            }
        }
        if (lhs_nul != nullptr || rhs_nul != nullptr) {
            return 0;
        }
        compared += static_cast<u64>(common);
    }
    return 0;
}
#endif

u64 ExecutorLibcStrlen(u64 str) {
#ifdef __ANDROID__
    str = ExecutorCanonicalizeTruncatedMspaceCString("strlen", str);
    const ExecutorGuestCStringScan scan = ExecutorScanGuestCString(str);
    const u64 len = scan.length;
    if (str != 0 && scan.status != ExecutorGuestCStringStatus::Terminated) {
        ExecutorLogInvalidLibcPtr("strlen", scan.fault_address);
    }
    const char* text = scan.status == ExecutorGuestCStringStatus::Terminated && scan.all_direct
                           ? reinterpret_cast<const char*>(ExecutorUntagHostPointer(str))
                           : nullptr;
#else
    const char* text = reinterpret_cast<const char*>(str);
    const u64 len = text ? static_cast<u64>(std::strlen(text)) : 0;
#endif
#ifdef __ANDROID__
    static std::atomic<int> log_budget{128};
    if (ExecutorVerboseLibcStringLogsEnabled() &&
        log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        char preview[96]{};
        if (text) {
            std::size_t i = 0;
            for (; i + 1 < sizeof(preview) && text[i] != '\0'; ++i) {
                const unsigned char ch = static_cast<unsigned char>(text[i]);
                preview[i] = std::isprint(ch) ? static_cast<char>(ch) : '.';
            }
            preview[i] = '\0';
        }
        std::fprintf(stderr, "[EXECUTOR_LIBC_STRLEN] str=%p len=%llu text=\"%s\"\n",
                     reinterpret_cast<void*>(str), static_cast<unsigned long long>(len),
                     text ? preview : "<null>");
        std::fflush(stderr);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_STRLEN] str=%p len=%llu text=\"%s\"",
                            reinterpret_cast<void*>(str), static_cast<unsigned long long>(len),
                            text ? preview : "<null>");
    }
#endif
    return len;
}

#ifdef __ANDROID__
static std::string ExecutorReadGuestNumericString(u64 ptr, std::size_t limit) {
    std::string out;
    if (ptr == 0 || limit == 0) {
        return out;
    }
    out.reserve(std::min<std::size_t>(limit, 64));
    for (std::size_t i = 0; i < limit; ++i) {
        unsigned char ch = 0;
        if (!ExecutorReadGuestByte(ptr + i, ch) || ch == '\0') {
            break;
        }
        out.push_back(static_cast<char>(ch));
    }
    return out;
}
#endif

u64 ExecutorLibcStrtoull(u64 str, u64 endptr, u64 base) {
#ifdef __ANDROID__
    str = ExecutorCanonicalizeTruncatedMspaceCString("strtoull", str);
    std::string text = ExecutorReadGuestNumericString(str, 4096);
    if (text.empty()) {
        if (endptr) {
            *reinterpret_cast<u64*>(endptr) = str;
        }
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value =
        std::strtoull(text.c_str(), &end, static_cast<int>(base));
    if (endptr) {
        const std::size_t consumed =
            end != nullptr && end >= text.c_str()
                ? static_cast<std::size_t>(end - text.c_str())
                : 0;
        *reinterpret_cast<u64*>(endptr) = str + consumed;
    }
    static std::atomic<int> log_budget{4};
    if (ExecutorVerboseLibcStringLogsEnabled() &&
        log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_STRTOULL] str=%p base=%llu value=%llu text=\"%s\"",
                            reinterpret_cast<void*>(str),
                            static_cast<unsigned long long>(base), value, text.c_str());
    }
    return static_cast<u64>(value);
#else
    char* end = nullptr;
    const unsigned long long value =
        std::strtoull(reinterpret_cast<const char*>(str), &end, static_cast<int>(base));
    if (endptr) {
        *reinterpret_cast<char**>(endptr) = end;
    }
    return static_cast<u64>(value);
#endif
}

s64 ExecutorLibcStrtoll(u64 str, u64 endptr, u64 base) {
#ifdef __ANDROID__
    str = ExecutorCanonicalizeTruncatedMspaceCString("strtoll", str);
    std::string text = ExecutorReadGuestNumericString(str, 4096);
    if (text.empty()) {
        if (endptr) {
            *reinterpret_cast<u64*>(endptr) = str;
        }
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text.c_str(), &end, static_cast<int>(base));
    if (endptr) {
        const std::size_t consumed =
            end != nullptr && end >= text.c_str()
                ? static_cast<std::size_t>(end - text.c_str())
                : 0;
        *reinterpret_cast<u64*>(endptr) = str + consumed;
    }
    return static_cast<s64>(value);
#else
    char* end = nullptr;
    const long long value =
        std::strtoll(reinterpret_cast<const char*>(str), &end, static_cast<int>(base));
    if (endptr) {
        *reinterpret_cast<char**>(endptr) = end;
    }
    return static_cast<s64>(value);
#endif
}

u64 ExecutorLibcStoul(u64 str, u64 endptr, u64 base) {
    return static_cast<u32>(ExecutorLibcStrtoull(str, endptr, base));
}

u64 ExecutorLibcStoull(u64 str, u64 endptr, u64 base) {
    return ExecutorLibcStrtoull(str, endptr, base);
}

s64 ExecutorLibcStol(u64 str, u64 endptr, u64 base) {
    return static_cast<s32>(ExecutorLibcStrtoll(str, endptr, base));
}

s64 ExecutorLibcStoll(u64 str, u64 endptr, u64 base) {
    return ExecutorLibcStrtoll(str, endptr, base);
}

s64 ExecutorLibcMemcmp(u64 lhs, u64 rhs, u64 size) {
    if (size == 0) {
        return 0;
    }
    const auto* a = reinterpret_cast<const unsigned char*>(lhs);
    const auto* b = reinterpret_cast<const unsigned char*>(rhs);
    if (!a || !b) {
        return a == b ? 0 : (a ? 1 : -1);
    }
    int result = 0;
#ifdef __ANDROID__
    auto* memory = Core::Memory::Instance();
    const bool lhs_direct = ExecutorDirectReadableGuestRange(lhs, size);
    if (!lhs_direct && (!memory || !memory->IsValidMapping(lhs, size))) {
        ExecutorLogInvalidLibcPtr("memcmp", lhs);
        return 1;
    }
    const bool rhs_direct = ExecutorDirectReadableGuestRange(rhs, size);
    if (!rhs_direct && (!memory || !memory->IsValidMapping(rhs, size))) {
        ExecutorLogInvalidLibcPtr("memcmp", rhs);
        return -1;
    }

    constexpr std::size_t CompareChunkSize = 1024;
    std::array<unsigned char, CompareChunkSize> lhs_chunk{};
    std::array<unsigned char, CompareChunkSize> rhs_chunk{};
    std::array<unsigned char, 16> lhs_preview_bytes{};
    std::array<unsigned char, 16> rhs_preview_bytes{};
    if (lhs_direct && rhs_direct) {
        const auto* direct_lhs = reinterpret_cast<const unsigned char*>(
            ExecutorUntagHostPointer(lhs));
        const auto* direct_rhs = reinterpret_cast<const unsigned char*>(
            ExecutorUntagHostPointer(rhs));
        result = std::memcmp(direct_lhs, direct_rhs, static_cast<std::size_t>(size));
        const std::size_t preview_size = static_cast<std::size_t>(std::min<u64>(size, 16));
        std::memcpy(lhs_preview_bytes.data(), direct_lhs, preview_size);
        std::memcpy(rhs_preview_bytes.data(), direct_rhs, preview_size);
    } else {
        for (u64 offset = 0; offset < size;) {
            const std::size_t chunk_size = static_cast<std::size_t>(
                std::min<u64>(size - offset, static_cast<u64>(CompareChunkSize)));
            if (lhs_direct) {
                std::memcpy(lhs_chunk.data(),
                            reinterpret_cast<const void*>(ExecutorUntagHostPointer(lhs) + offset),
                            chunk_size);
            } else {
                memory->CopySparseMemory(lhs + offset, lhs_chunk.data(), chunk_size);
            }
            if (rhs_direct) {
                std::memcpy(rhs_chunk.data(),
                            reinterpret_cast<const void*>(ExecutorUntagHostPointer(rhs) + offset),
                            chunk_size);
            } else {
                memory->CopySparseMemory(rhs + offset, rhs_chunk.data(), chunk_size);
            }
            if (offset == 0) {
                const std::size_t preview_size = std::min<std::size_t>(chunk_size, 16);
                std::memcpy(lhs_preview_bytes.data(), lhs_chunk.data(), preview_size);
                std::memcpy(rhs_preview_bytes.data(), rhs_chunk.data(), preview_size);
            }
            result = std::memcmp(lhs_chunk.data(), rhs_chunk.data(), chunk_size);
            if (result != 0) {
                break;
            }
            offset += chunk_size;
        }
    }
    const auto is_low_stackish = [](const u64 value) {
        return value >= 0x60000000ULL && value < 0x80000000ULL;
    };
    if ((is_low_stackish(lhs) || is_low_stackish(rhs)) && size <= 0x20) {
        static std::atomic<int> s_low_memcmp_budget{4};
        if (s_low_memcmp_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            char lhs_preview[33]{};
            char rhs_preview[33]{};
            const std::size_t n = static_cast<std::size_t>(std::min<u64>(size, 16));
            for (std::size_t i = 0; i < n; ++i) {
                const unsigned char la = lhs_preview_bytes[i];
                const unsigned char rb = rhs_preview_bytes[i];
                lhs_preview[i] = std::isprint(la) ? static_cast<char>(la) : '.';
                rhs_preview[i] = std::isprint(rb) ? static_cast<char>(rb) : '.';
            }
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIBC_MEMCMP_LOW] lhs=%p rhs=%p size=0x%llx result=%d "
                                "lhs0=%02x rhs0=%02x lhsText=\"%s\" rhsText=\"%s\"",
                                reinterpret_cast<void*>(lhs), reinterpret_cast<void*>(rhs),
                                static_cast<unsigned long long>(size), result,
                                lhs_preview_bytes[0], rhs_preview_bytes[0], lhs_preview,
                                rhs_preview);
        }
    }
#else
    result = std::memcmp(a, b, static_cast<std::size_t>(size));
#endif
#ifdef __ANDROID__
    static std::atomic<int> log_budget{128};
    if (ExecutorVerboseLibcStringLogsEnabled() &&
        log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_MEMCMP] lhs=%p rhs=%p size=%llu result=%d "
                            "lhs0=0x%02x rhs0=0x%02x",
                            reinterpret_cast<void*>(lhs), reinterpret_cast<void*>(rhs),
                            static_cast<unsigned long long>(size), result,
                            lhs_preview_bytes[0], rhs_preview_bytes[0]);
    }
#endif
    return static_cast<s64>(result);
}

extern "C" int executor_lsx4_android_run_guest_signal_handler(std::uint64_t handler,
                                                                 std::uint64_t arg0,
                                                                 std::uint64_t arg1,
                                                                 std::uint64_t arg2,
                                                                 std::uint64_t* guest_result)
    __attribute__((weak));

extern "C" int executor_lsx4_android_run_guest_comparator(std::uint64_t func,
                                                             std::uint64_t arg0,
                                                             std::uint64_t arg1,
                                                             std::uint64_t* result_lo)
    __attribute__((weak));

#ifdef __ANDROID__
s64 ExecutorLibcStrcmp(u64 lhs, u64 rhs);

constexpr u64 ExecutorMonoClassInterfaceIdOffset = 0x5a;
constexpr std::array<u8, 16> ExecutorMonoCompareInterfaceIdsSignature = {
    0x0f, 0xb7, 0x47, 0x5a,
    0x48, 0x8b, 0x0e,
    0x0f, 0xb7, 0x49, 0x5a,
    0x29, 0xc8,
    0xc3,
    0x66, 0x90,
};

static bool ExecutorHasMonoCompareInterfaceIdsSignature(const u64 compar) {
    std::array<u8, ExecutorMonoCompareInterfaceIdsSignature.size()> bytes{};
    return ExecutorReadGuestBytes(compar, bytes.data(), bytes.size()) &&
           bytes == ExecutorMonoCompareInterfaceIdsSignature;
}

static bool ExecutorIsMonoCompareInterfaceIds(const u64 compar) {
    struct SignatureCache {
        std::array<u64, 2> addresses{};
        std::array<bool, 2> matches{};
        unsigned next = 0;
    };
    thread_local SignatureCache cache;
    for (std::size_t i = 0; i < cache.addresses.size(); ++i) {
        if (cache.addresses[i] == compar) {
            return cache.matches[i];
        }
    }
    const bool matches = ExecutorHasMonoCompareInterfaceIdsSignature(compar);
    const unsigned slot = cache.next++ % cache.addresses.size();
    cache.addresses[slot] = compar;
    cache.matches[slot] = matches;
    return matches;
}

static bool ExecutorReadMonoInterfaceFastPathBytes(const u64 address, void* out,
                                                   const std::size_t size,
                                                   const bool use_jit_reader) {
    if (use_jit_reader && ExecutorJitReadGuestBytes != nullptr &&
        ExecutorJitReadGuestBytes(address, out, size)) {
        return true;
    }
    return ExecutorReadGuestBytes(address, out, size);
}

static bool ExecutorReadMonoClassInterfaceId(const u64 klass, std::uint16_t& interface_id,
                                             const bool use_jit_reader) {
    if (klass == 0 ||
        klass > std::numeric_limits<u64>::max() - ExecutorMonoClassInterfaceIdOffset) {
        return false;
    }
    return ExecutorReadMonoInterfaceFastPathBytes(klass + ExecutorMonoClassInterfaceIdOffset,
                                                  &interface_id, sizeof(interface_id),
                                                  use_jit_reader);
}

static bool ExecutorCompareMonoInterfaceIds(const std::uint16_t key_interface_id, const u64 element,
                                            int& result, const bool use_jit_reader) {
    u64 element_class = 0;
    std::uint16_t element_interface_id = 0;
    if (!ExecutorReadMonoInterfaceFastPathBytes(element, &element_class, sizeof(element_class),
                                                use_jit_reader) ||
        !ExecutorReadMonoClassInterfaceId(element_class, element_interface_id,
                                          use_jit_reader)) {
        return false;
    }
    result = static_cast<int>(key_interface_id) - static_cast<int>(element_interface_id);
    return true;
}

static bool ExecutorMonoCompareInterfaceIdsSelfCheck() {
    auto exact_signature = ExecutorMonoCompareInterfaceIdsSignature;
    auto drifted_signature = exact_signature;
    drifted_signature[3] ^= 1;
    if (!ExecutorHasMonoCompareInterfaceIdsSignature(
            reinterpret_cast<u64>(exact_signature.data())) ||
        ExecutorHasMonoCompareInterfaceIdsSignature(
            reinterpret_cast<u64>(drifted_signature.data()))) {
        return false;
    }

    std::array<u8, ExecutorMonoClassInterfaceIdOffset + sizeof(std::uint16_t)> key{};
    std::array<u8, ExecutorMonoClassInterfaceIdOffset + sizeof(std::uint16_t)> element_class{};
    const std::uint16_t key_id = 0x8123;
    const std::uint16_t element_id = 0x1234;
    std::memcpy(key.data() + ExecutorMonoClassInterfaceIdOffset, &key_id, sizeof(key_id));
    std::memcpy(element_class.data() + ExecutorMonoClassInterfaceIdOffset, &element_id,
                sizeof(element_id));
    u64 element_class_ptr = reinterpret_cast<u64>(element_class.data());

    std::uint16_t decoded_key_id = 0;
    int result = 0;
    if (!ExecutorReadMonoClassInterfaceId(reinterpret_cast<u64>(key.data()), decoded_key_id,
                                          false) ||
        decoded_key_id != key_id ||
        !ExecutorCompareMonoInterfaceIds(decoded_key_id, reinterpret_cast<u64>(&element_class_ptr),
                                         result, false) ||
        result != static_cast<int>(key_id) - static_cast<int>(element_id)) {
        return false;
    }

    element_class_ptr = 0;
    return !ExecutorCompareMonoInterfaceIds(
        decoded_key_id, reinterpret_cast<u64>(&element_class_ptr), result, false);
}

constexpr std::array<u8, 8> ExecutorMonoIndirectStrcmpComparatorSignature = {
    0x48, 0x8b, 0x36, 0xe9, 0xf8, 0xa2, 0xfb, 0xff,
};

static bool ExecutorHasMonoIndirectStrcmpComparatorSignature(const u64 compar) {
    std::array<u8, ExecutorMonoIndirectStrcmpComparatorSignature.size()> bytes{};
    return ExecutorReadGuestBytes(compar, bytes.data(), bytes.size()) &&
           bytes == ExecutorMonoIndirectStrcmpComparatorSignature;
}

static bool ExecutorIsMonoIndirectStrcmpComparator(const u64 compar) {
    struct SignatureCache {
        std::array<u64, 2> addresses{};
        std::array<bool, 2> matches{};
        unsigned next = 0;
    };
    thread_local SignatureCache cache;
    for (std::size_t i = 0; i < cache.addresses.size(); ++i) {
        if (cache.addresses[i] == compar) {
            return cache.matches[i];
        }
    }
    const bool matches = ExecutorHasMonoIndirectStrcmpComparatorSignature(compar);
    const unsigned slot = cache.next++ % cache.addresses.size();
    cache.addresses[slot] = compar;
    cache.matches[slot] = matches;
    return matches;
}

static bool ExecutorCompareMonoIndirectStrcmp(const u64 key, const u64 element, int& result,
                                              const bool use_jit_reader) {
    u64 element_string = 0;
    if (!ExecutorReadMonoInterfaceFastPathBytes(element, &element_string,
                                                sizeof(element_string),
                                                use_jit_reader) ||
        element_string == 0) {
        return false;
    }
    result = static_cast<int>(ExecutorLibcStrcmp(key, element_string));
    return true;
}

static bool ExecutorMonoIndirectStrcmpSelfCheck() {
    auto exact_signature = ExecutorMonoIndirectStrcmpComparatorSignature;
    auto drifted_signature = exact_signature;
    drifted_signature[2] ^= 1;
    if (!ExecutorHasMonoIndirectStrcmpComparatorSignature(
            reinterpret_cast<u64>(exact_signature.data())) ||
        ExecutorHasMonoIndirectStrcmpComparatorSignature(
            reinterpret_cast<u64>(drifted_signature.data()))) {
        return false;
    }

    std::array<char, 6> key = {'c', 'o', 'a', 'c', 'h', '\0'};
    std::array<char, 6> equal = key;
    std::array<char, 7> after = {'d', 'r', 'i', 'v', 'e', 'r', '\0'};
    u64 element_string = reinterpret_cast<u64>(equal.data());
    int result = 1;
    if (!ExecutorCompareMonoIndirectStrcmp(
            reinterpret_cast<u64>(key.data()), reinterpret_cast<u64>(&element_string),
            result, false) ||
        result != 0) {
        return false;
    }
    element_string = reinterpret_cast<u64>(after.data());
    return ExecutorCompareMonoIndirectStrcmp(
               reinterpret_cast<u64>(key.data()), reinterpret_cast<u64>(&element_string),
               result, false) &&
           result < 0;
}

constexpr auto ExecutorMonoMetadataBsearchComparatorSignature = std::to_array<u8>({
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x56, 0x53, 0x48, 0x89, 0xfb, 0x48, 0x8b, 0x7b,
    0x08, 0x48, 0x2b, 0x37, 0x0f, 0xb6, 0x4f, 0x0b, 0x48, 0x89, 0xf0, 0x48, 0x99,
    0x48, 0xf7, 0xf9, 0x49, 0x89, 0xc6, 0x8b, 0x53, 0x04, 0x44, 0x89, 0xf6, 0xe8,
    0xf5, 0xc2, 0xff, 0xff, 0x39, 0x03, 0x75, 0x08, 0x44, 0x89, 0x73, 0x10, 0x31,
    0xc0, 0xeb, 0x0d, 0xb9, 0xff, 0xff, 0xff, 0xff, 0xb8, 0x01, 0x00, 0x00, 0x00,
    0x0f, 0x42, 0xc1, 0x5b, 0x41, 0x5e, 0x5d, 0xc3,
});

constexpr auto ExecutorMonoMetadataRangeBsearchComparatorSignature = std::to_array<u8>({
    0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x48, 0x89, 0xfb,
    0x48, 0x8b, 0x7b, 0x08, 0x48, 0x2b, 0x37, 0x0f, 0xb6, 0x4f, 0x0b, 0x48, 0x89,
    0xf0, 0x48, 0x99, 0x48, 0xf7, 0xf9, 0x8b, 0x53, 0x04, 0x49, 0x89, 0xc6, 0x44,
    0x89, 0xf6, 0xe8, 0x42, 0xc4, 0xff, 0xff, 0x41, 0x89, 0xc7, 0xb8, 0xff, 0xff,
    0xff, 0xff, 0x44, 0x39, 0x3b, 0x72, 0x32, 0x48, 0x8b, 0x7b, 0x08, 0xb8, 0xff,
    0xff, 0xff, 0x00, 0x41, 0x8d, 0x76, 0x01, 0x23, 0x47, 0x08, 0x39, 0xc6, 0x7d,
    0x18, 0x8b, 0x53, 0x04, 0xe8, 0x19, 0xc4, 0xff, 0xff, 0x89, 0xc1, 0xb8, 0x01,
    0x00, 0x00, 0x00, 0x41, 0x39, 0xcf, 0x74, 0x0a, 0x39, 0x0b, 0x73, 0x06, 0x31,
    0xc0, 0x44, 0x89, 0x73, 0x10, 0x48, 0x83, 0xc4, 0x08, 0x5b, 0x41, 0x5e, 0x41,
    0x5f, 0x5d, 0xc3,
});

static bool ExecutorMatchesMonoMetadataBsearchComparatorShape(
    const std::span<const u8, ExecutorMonoMetadataBsearchComparatorSignature.size()> bytes) {
    constexpr std::size_t kScheduledPairOffset = 29;
    constexpr std::size_t kScheduledPairSize = 6;
    constexpr std::array<u8, kScheduledPairSize> kMoveResultThenColumn = {
        0x49, 0x89, 0xc6, 0x8b, 0x53, 0x04,
    };
    constexpr std::array<u8, kScheduledPairSize> kMoveColumnThenResult = {
        0x8b, 0x53, 0x04, 0x49, 0x89, 0xc6,
    };
    constexpr std::size_t kDecodeCallOpcodeOffset = 38;
    constexpr std::size_t kDecodeCallDisplacementOffset = kDecodeCallOpcodeOffset + 1;
    constexpr std::size_t kDecodeCallDisplacementSize = sizeof(std::int32_t);
    constexpr std::size_t kSuffixOffset =
        kDecodeCallDisplacementOffset + kDecodeCallDisplacementSize;

    if (!std::equal(bytes.begin(), bytes.begin() + kScheduledPairOffset,
                    ExecutorMonoMetadataBsearchComparatorSignature.begin())) {
        return false;
    }
    const auto scheduled_pair =
        bytes.subspan<kScheduledPairOffset, kScheduledPairSize>();
    if (!std::equal(scheduled_pair.begin(), scheduled_pair.end(),
                    kMoveResultThenColumn.begin()) &&
        !std::equal(scheduled_pair.begin(), scheduled_pair.end(),
                    kMoveColumnThenResult.begin())) {
        return false;
    }
    if (!std::equal(bytes.begin() + kScheduledPairOffset + kScheduledPairSize,
                    bytes.begin() + kDecodeCallDisplacementOffset,
                    ExecutorMonoMetadataBsearchComparatorSignature.begin() +
                        kScheduledPairOffset + kScheduledPairSize)) {
        return false;
    }
    return std::equal(bytes.begin() + kSuffixOffset, bytes.end(),
                      ExecutorMonoMetadataBsearchComparatorSignature.begin() + kSuffixOffset);
}

static bool ExecutorHasMonoMetadataBsearchComparatorSignature(const u64 compar) {
    std::array<u8, ExecutorMonoMetadataBsearchComparatorSignature.size()> bytes{};
    return ExecutorReadGuestBytes(compar, bytes.data(), bytes.size()) &&
           ExecutorMatchesMonoMetadataBsearchComparatorShape(bytes);
}

static bool ExecutorMatchesMonoMetadataRangeBsearchComparatorShape(
    const std::span<const u8,
                    ExecutorMonoMetadataRangeBsearchComparatorSignature.size()> bytes) {
    constexpr std::size_t kScheduledPairOffset = 32;
    constexpr std::size_t kScheduledPairSize = 6;
    constexpr std::array<u8, kScheduledPairSize> kMoveColumnThenResult = {
        0x8b, 0x53, 0x04, 0x49, 0x89, 0xc6,
    };
    constexpr std::array<u8, kScheduledPairSize> kMoveResultThenColumn = {
        0x49, 0x89, 0xc6, 0x8b, 0x53, 0x04,
    };
    constexpr std::array<std::size_t, 2> kCallOpcodes = {41, 82};

    if (!std::equal(bytes.begin(), bytes.begin() + kScheduledPairOffset,
                    ExecutorMonoMetadataRangeBsearchComparatorSignature.begin())) {
        return false;
    }
    const auto scheduled_pair =
        bytes.subspan<kScheduledPairOffset, kScheduledPairSize>();
    if (!std::equal(scheduled_pair.begin(), scheduled_pair.end(),
                    kMoveColumnThenResult.begin()) &&
        !std::equal(scheduled_pair.begin(), scheduled_pair.end(),
                    kMoveResultThenColumn.begin())) {
        return false;
    }
    std::size_t checked = kScheduledPairOffset + kScheduledPairSize;
    for (const std::size_t call_opcode : kCallOpcodes) {
        if (!std::equal(bytes.begin() + checked, bytes.begin() + call_opcode + 1,
                        ExecutorMonoMetadataRangeBsearchComparatorSignature.begin() + checked) ||
            bytes[call_opcode] != 0xe8) {
            return false;
        }
        checked = call_opcode + 5;
    }
    return std::equal(bytes.begin() + checked, bytes.end(),
                      ExecutorMonoMetadataRangeBsearchComparatorSignature.begin() + checked);
}

static bool ExecutorHasMonoMetadataRangeBsearchComparatorSignature(const u64 compar) {
    std::array<u8, ExecutorMonoMetadataRangeBsearchComparatorSignature.size()> bytes{};
    return ExecutorReadGuestBytes(compar, bytes.data(), bytes.size()) &&
           ExecutorMatchesMonoMetadataRangeBsearchComparatorShape(bytes);
}

static bool ExecutorIsMonoMetadataBsearchComparator(const u64 compar) {
    struct SignatureCache {
        std::array<u64, 2> addresses{};
        std::array<bool, 2> matches{};
        unsigned next = 0;
    };
    thread_local SignatureCache cache;
    for (std::size_t i = 0; i < cache.addresses.size(); ++i) {
        if (cache.addresses[i] == compar) {
            return cache.matches[i];
        }
    }
    const bool matches = ExecutorHasMonoMetadataBsearchComparatorSignature(compar);
    const unsigned slot = cache.next++ % cache.addresses.size();
    cache.addresses[slot] = compar;
    cache.matches[slot] = matches;
    return matches;
}

static bool ExecutorIsMonoMetadataRangeBsearchComparator(const u64 compar) {
    struct SignatureCache {
        std::array<u64, 2> addresses{};
        std::array<bool, 2> matches{};
        unsigned next = 0;
    };
    thread_local SignatureCache cache;
    for (std::size_t i = 0; i < cache.addresses.size(); ++i) {
        if (cache.addresses[i] == compar) {
            return cache.matches[i];
        }
    }
    const bool matches = ExecutorHasMonoMetadataRangeBsearchComparatorSignature(compar);
    const unsigned slot = cache.next++ % cache.addresses.size();
    cache.addresses[slot] = compar;
    cache.matches[slot] = matches;
    return matches;
}

struct ExecutorMonoMetadataBsearchDecoded {
    std::uint32_t wanted{};
    std::uint32_t value{};
    std::uint32_t row{};
};

static bool ExecutorDecodeMonoMetadataBsearchElement(
    const u64 key, const u64 element, ExecutorMonoMetadataBsearchDecoded& decoded,
    const bool use_jit_reader) {
    struct KeyPrefix {
        std::uint32_t wanted;
        std::uint32_t column;
        u64 table;
    } key_prefix{};
    struct TablePrefix {
        u64 base;
        std::uint32_t rows_and_stride;
        std::uint32_t packed_columns;
    } table{};

    if (!ExecutorReadMonoInterfaceFastPathBytes(key, &key_prefix, sizeof(key_prefix),
                                                use_jit_reader) ||
        key_prefix.table == 0 ||
        !ExecutorReadMonoInterfaceFastPathBytes(key_prefix.table, &table, sizeof(table),
                                                use_jit_reader)) {
        return false;
    }

    const std::uint32_t rows = table.rows_and_stride & 0x00ffffffu;
    const std::uint32_t stride = table.rows_and_stride >> 24;
    const std::uint32_t column_count = table.packed_columns >> 24;
    if (table.base == 0 || rows == 0 || stride == 0 || key_prefix.column >= column_count ||
        key_prefix.column >= 12 || element < table.base) {
        return false;
    }

    const u64 delta = element - table.base;
    if (delta % stride != 0) {
        return false;
    }
    const u64 row64 = delta / stride;
    if (row64 >= rows || row64 > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }

    std::uint32_t column_offset = 0;
    for (std::uint32_t i = 0; i < key_prefix.column; ++i) {
        const std::uint32_t width_code = (table.packed_columns >> (i * 2)) & 3u;
        if (width_code == 2) {
            return false;
        }
        column_offset += width_code + 1;
    }
    const std::uint32_t width_code =
        (table.packed_columns >> (key_prefix.column * 2)) & 3u;
    if (width_code == 2) {
        return false;
    }
    const std::uint32_t width = width_code + 1;
    if (column_offset > stride || width > stride - column_offset ||
        row64 > (std::numeric_limits<u64>::max() - table.base) / stride) {
        return false;
    }
    const u64 row_address = table.base + row64 * stride;
    if (column_offset > std::numeric_limits<u64>::max() - row_address) {
        return false;
    }
    const u64 value_address = row_address + column_offset;

    std::uint32_t value = 0;
    if (width_code == 0) {
        std::int8_t byte = 0;
        if (!ExecutorReadMonoInterfaceFastPathBytes(value_address, &byte, sizeof(byte),
                                                    use_jit_reader)) {
            return false;
        }
        value = static_cast<std::uint32_t>(static_cast<std::int32_t>(byte));
    } else if (width_code == 1) {
        std::uint16_t word = 0;
        if (!ExecutorReadMonoInterfaceFastPathBytes(value_address, &word, sizeof(word),
                                                    use_jit_reader)) {
            return false;
        }
        value = word;
    } else {
        if (!ExecutorReadMonoInterfaceFastPathBytes(value_address, &value, sizeof(value),
                                                    use_jit_reader)) {
            return false;
        }
    }

    decoded.wanted = key_prefix.wanted;
    decoded.value = value;
    decoded.row = static_cast<std::uint32_t>(row64);
    return true;
}

static bool ExecutorCompareMonoMetadataBsearch(const u64 key, const u64 element, int& result,
                                               const bool use_jit_reader) {
    ExecutorMonoMetadataBsearchDecoded decoded{};
    if (!ExecutorDecodeMonoMetadataBsearchElement(key, element, decoded,
                                                  use_jit_reader)) {
        return false;
    }
    if (decoded.wanted == decoded.value) {
        if (!use_jit_reader || ExecutorJitWriteGuestBytes == nullptr ||
            key > std::numeric_limits<u64>::max() - 0x10 ||
            !ExecutorJitWriteGuestBytes(key + 0x10, &decoded.row, sizeof(decoded.row))) {
            return false;
        }
        result = 0;
    } else {
        result = decoded.wanted < decoded.value ? -1 : 1;
    }
    return true;
}

static bool ExecutorCompareMonoMetadataRangeBsearch(const u64 key, const u64 element,
                                                    int& result,
                                                    const bool use_jit_reader) {
    ExecutorMonoMetadataBsearchDecoded current{};
    if (!ExecutorDecodeMonoMetadataBsearchElement(key, element, current,
                                                  use_jit_reader)) {
        return false;
    }
    if (current.wanted < current.value) {
        result = -1;
        return true;
    }

    struct KeyPrefix {
        std::uint32_t wanted;
        std::uint32_t column;
        u64 table;
    } key_prefix{};
    struct TablePrefix {
        u64 base;
        std::uint32_t rows_and_stride;
        std::uint32_t packed_columns;
    } table{};
    if (!ExecutorReadMonoInterfaceFastPathBytes(key, &key_prefix, sizeof(key_prefix),
                                                use_jit_reader) ||
        key_prefix.table == 0 ||
        !ExecutorReadMonoInterfaceFastPathBytes(key_prefix.table, &table, sizeof(table),
                                                use_jit_reader)) {
        return false;
    }
    const std::uint32_t rows = table.rows_and_stride & 0x00ffffffu;
    const std::uint32_t stride = table.rows_and_stride >> 24;
    if (stride == 0 || current.row >= rows) {
        return false;
    }
    if (current.row + 1 < rows) {
        if (element > std::numeric_limits<u64>::max() - stride) {
            return false;
        }
        ExecutorMonoMetadataBsearchDecoded next{};
        if (!ExecutorDecodeMonoMetadataBsearchElement(key, element + stride, next,
                                                      use_jit_reader) ||
            next.row != current.row + 1) {
            return false;
        }
        if (current.value == next.value || current.wanted >= next.value) {
            result = 1;
            return true;
        }
    }
    if (!use_jit_reader || ExecutorJitWriteGuestBytes == nullptr ||
        key > std::numeric_limits<u64>::max() - 0x10 ||
        !ExecutorJitWriteGuestBytes(key + 0x10, &current.row, sizeof(current.row))) {
        return false;
    }
    result = 0;
    return true;
}

static bool ExecutorMonoMetadataBsearchSelfCheck() {
    auto exact_signature = ExecutorMonoMetadataBsearchComparatorSignature;
    auto rescheduled_signature = exact_signature;
    constexpr std::array<u8, 6> kMoveColumnThenResult = {
        0x8b, 0x53, 0x04, 0x49, 0x89, 0xc6,
    };
    std::copy(kMoveColumnThenResult.begin(), kMoveColumnThenResult.end(),
              rescheduled_signature.begin() + 29);
    rescheduled_signature[39] = 0x05;
    rescheduled_signature[40] = 0xc0;
    rescheduled_signature[41] = 0xff;
    rescheduled_signature[42] = 0xff;
    auto drifted_signature = exact_signature;
    drifted_signature[38] ^= 1;
    auto range_signature = ExecutorMonoMetadataRangeBsearchComparatorSignature;
    auto range_relinked_signature = range_signature;
    range_relinked_signature[42] ^= 0x55;
    range_relinked_signature[83] ^= 0xaa;
    auto range_drifted_signature = range_signature;
    range_drifted_signature[87] ^= 1;
    if (!ExecutorHasMonoMetadataBsearchComparatorSignature(
            reinterpret_cast<u64>(exact_signature.data())) ||
        !ExecutorHasMonoMetadataBsearchComparatorSignature(
            reinterpret_cast<u64>(rescheduled_signature.data())) ||
        ExecutorHasMonoMetadataBsearchComparatorSignature(
            reinterpret_cast<u64>(drifted_signature.data())) ||
        !ExecutorHasMonoMetadataRangeBsearchComparatorSignature(
            reinterpret_cast<u64>(range_signature.data())) ||
        !ExecutorHasMonoMetadataRangeBsearchComparatorSignature(
            reinterpret_cast<u64>(range_relinked_signature.data())) ||
        ExecutorHasMonoMetadataRangeBsearchComparatorSignature(
            reinterpret_cast<u64>(range_drifted_signature.data()))) {
        return false;
    }

    constexpr std::uint32_t kRows = 4;
    constexpr std::uint32_t kStride = 10;
    constexpr std::uint32_t kColumn = 2;
    constexpr std::uint32_t kWanted = 0x89abcdefu;
    constexpr std::uint32_t kPackedColumns =
        (3u << 24) | (3u << 0) | (1u << 2) | (3u << 4);
    std::array<u8, kRows * kStride> rows{};
    std::memcpy(rows.data() + 2 * kStride + 6, &kWanted, sizeof(kWanted));

    struct TablePrefix {
        u64 base;
        std::uint32_t rows_and_stride;
        std::uint32_t packed_columns;
    } table{reinterpret_cast<u64>(rows.data()), kRows | (kStride << 24), kPackedColumns};
    struct KeyPrefix {
        std::uint32_t wanted;
        std::uint32_t column;
        u64 table;
        std::uint32_t row;
    } key{kWanted, kColumn, reinterpret_cast<u64>(&table), 0xffffffffu};

    ExecutorMonoMetadataBsearchDecoded decoded{};
    if (!ExecutorDecodeMonoMetadataBsearchElement(
            reinterpret_cast<u64>(&key), reinterpret_cast<u64>(rows.data() + 2 * kStride),
            decoded, false) ||
        decoded.wanted != kWanted || decoded.value != kWanted || decoded.row != 2) {
        return false;
    }

    std::uint32_t next_value = kWanted + 7;
    std::memcpy(rows.data() + 3 * kStride + 6, &next_value, sizeof(next_value));
    int range_result = 1;
    key.wanted = kWanted - 1;
    if (!ExecutorCompareMonoMetadataRangeBsearch(
            reinterpret_cast<u64>(&key),
            reinterpret_cast<u64>(rows.data() + 2 * kStride), range_result, false) ||
        range_result != -1) {
        return false;
    }
    key.wanted = next_value;
    if (!ExecutorCompareMonoMetadataRangeBsearch(
            reinterpret_cast<u64>(&key),
            reinterpret_cast<u64>(rows.data() + 2 * kStride), range_result, false) ||
        range_result != 1) {
        return false;
    }
    key.wanted = kWanted;

    key.column = 3;
    if (ExecutorDecodeMonoMetadataBsearchElement(
            reinterpret_cast<u64>(&key), reinterpret_cast<u64>(rows.data()), decoded, false)) {
        return false;
    }
    key.column = kColumn;
    return !ExecutorDecodeMonoMetadataBsearchElement(
        reinterpret_cast<u64>(&key), reinterpret_cast<u64>(rows.data() + 1), decoded, false);
}
#endif

u64 ExecutorLibcBsearch(u64 key, u64 base, u64 nmemb, u64 size, u64 compar) {
    if (key == 0 || base == 0 || size == 0 || compar == 0) {
        return 0;
    }
    if (nmemb != 0 && size > UINT64_MAX / nmemb) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_LIBC_BSEARCH_GUEST] overflow key=%p base=%p nmemb=%llu "
                            "size=%llu compar=%p result=null",
                            reinterpret_cast<void*>(key), reinterpret_cast<void*>(base),
                            static_cast<unsigned long long>(nmemb),
                            static_cast<unsigned long long>(size), reinterpret_cast<void*>(compar));
#endif
        return 0;
    }

    u64 low = 0;
    u64 high = nmemb;
    u64 result = 0;
    int last_cmp = 0;
    u32 iterations = 0;
#ifdef __ANDROID__
    u64 hle_native_fn = 0;
    const bool compar_is_native_strcmp =
        TryGetAndroidX64NativeHleTarget(compar, &hle_native_fn) &&
        hle_native_fn == reinterpret_cast<u64>(&ExecutorLibcStrcmp);
    static const bool mono_interface_id_selfcheck_ok = ExecutorMonoCompareInterfaceIdsSelfCheck();
    static const bool mono_indirect_strcmp_selfcheck_ok =
        ExecutorMonoIndirectStrcmpSelfCheck();
    static const bool mono_metadata_bsearch_selfcheck_ok =
        ExecutorMonoMetadataBsearchSelfCheck();
    static const bool mono_interface_id_selfcheck_logged = [] {
        if (!mono_interface_id_selfcheck_ok) {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_LIBC_BSEARCH_MONO_IFACE_SELFCHECK] result=failed "
                                "fallback=guest");
        }
        return true;
    }();
    (void)mono_interface_id_selfcheck_logged;
    static const bool mono_indirect_strcmp_selfcheck_logged = [] {
        if (!mono_indirect_strcmp_selfcheck_ok) {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_LIBC_BSEARCH_MONO_INDIRECT_STRCMP_SELFCHECK] "
                                "result=failed fallback=guest");
        }
        return true;
    }();
    (void)mono_indirect_strcmp_selfcheck_logged;
    static const bool mono_metadata_bsearch_selfcheck_logged = [] {
        if (!mono_metadata_bsearch_selfcheck_ok) {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_LIBC_BSEARCH_MONO_METADATA_SELFCHECK] result=failed "
                                "fallback=guest");
        }
        return true;
    }();
    (void)mono_metadata_bsearch_selfcheck_logged;
    const bool use_jit_mono_reader =
        ExecutorJitActiveForStubGeneration() && ExecutorJitReadGuestBytes != nullptr;
    std::uint16_t mono_key_interface_id = 0;
    const bool compar_is_mono_interface_ids =
        mono_interface_id_selfcheck_ok && size == sizeof(u64) &&
        ExecutorIsMonoCompareInterfaceIds(compar) &&
        ExecutorReadMonoClassInterfaceId(key, mono_key_interface_id, use_jit_mono_reader);
    const bool compar_is_mono_indirect_strcmp =
        mono_indirect_strcmp_selfcheck_ok && size == sizeof(u64) &&
        ExecutorIsMonoIndirectStrcmpComparator(compar);
    const bool compar_is_mono_metadata_bsearch =
        mono_metadata_bsearch_selfcheck_ok && use_jit_mono_reader &&
        ExecutorJitWriteGuestBytes != nullptr &&
        ExecutorIsMonoMetadataBsearchComparator(compar);
    const bool compar_is_mono_metadata_range_bsearch =
        mono_metadata_bsearch_selfcheck_ok && use_jit_mono_reader &&
        ExecutorJitWriteGuestBytes != nullptr &&
        ExecutorIsMonoMetadataRangeBsearchComparator(compar);
#endif
    while (low < high) {
        const u64 mid = low + ((high - low) / 2);
        const u64 elem = base + (mid * size);
#ifdef __ANDROID__
        int cmp;
        if (compar_is_native_strcmp) {
            cmp = static_cast<int>(ExecutorLibcStrcmp(key, elem));
        } else if (compar_is_mono_indirect_strcmp &&
                   ExecutorCompareMonoIndirectStrcmp(key, elem, cmp,
                                                     use_jit_mono_reader)) {
        } else if (compar_is_mono_interface_ids &&
                   ExecutorCompareMonoInterfaceIds(mono_key_interface_id, elem, cmp,
                                                   use_jit_mono_reader)) {
        } else if (compar_is_mono_metadata_bsearch &&
                   ExecutorCompareMonoMetadataBsearch(key, elem, cmp,
                                                      use_jit_mono_reader)) {
        } else if (compar_is_mono_metadata_range_bsearch &&
                   ExecutorCompareMonoMetadataRangeBsearch(
                       key, elem, cmp, use_jit_mono_reader)) {
        } else {
            u64 raw_cmp = 0;
            const int cb_rc =
                executor_lsx4_android_run_guest_comparator != nullptr
                    ? executor_lsx4_android_run_guest_comparator(compar, key, elem, &raw_cmp)
                    : (executor_lsx4_android_run_guest_signal_handler != nullptr
                           ? executor_lsx4_android_run_guest_signal_handler(compar, key, elem, 0,
                                                                               &raw_cmp)
                           : -1);
            if (cb_rc != 0) {
                static std::atomic<int> s_bsearch_cb_fail_budget{32};
                if (s_bsearch_cb_fail_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    __android_log_print(
                        ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_LIBC_BSEARCH_GUEST] comparator call failed rc=%d "
                        "compar=%p key=%p elem=%p iterations=%u reason=abort_search",
                        cb_rc, reinterpret_cast<void*>(compar), reinterpret_cast<void*>(key),
                        reinterpret_cast<void*>(elem), iterations);
                }
                break;
            }
            cmp = static_cast<int>(static_cast<s32>(raw_cmp));
        }
#else
        using BsearchComparator = int (*)(const void*, const void*);
        const auto guest_cmp = reinterpret_cast<BsearchComparator>(compar);
        const int cmp =
            guest_cmp(reinterpret_cast<const void*>(key), reinterpret_cast<const void*>(elem));
#endif
        last_cmp = cmp;
        ++iterations;
        if (cmp < 0) {
            high = mid;
        } else if (cmp > 0) {
            low = mid + 1;
        } else {
            result = elem;
            break;
        }
    }

#ifdef __ANDROID__
    static std::atomic<int> s_bsearch_log_budget{128};
    if (ExecutorVerboseLibcBsearchLogsEnabled() &&
        s_bsearch_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_BSEARCH_GUEST] key=%p base=%p nmemb=%llu size=%llu "
                            "compar=%p result=%p iterations=%u lastCmp=%d",
                            reinterpret_cast<void*>(key), reinterpret_cast<void*>(base),
                            static_cast<unsigned long long>(nmemb),
                            static_cast<unsigned long long>(size), reinterpret_cast<void*>(compar),
                            reinterpret_cast<void*>(result), iterations, last_cmp);
    }
#endif
    return result;
}

s64 ExecutorLibcStrncmp(u64 lhs, u64 rhs, u64 size) {
    if (size == 0) {
        return 0;
    }
#ifdef __ANDROID__
    lhs = ExecutorCanonicalizeTruncatedMspaceCString("strncmp", lhs);
    rhs = ExecutorCanonicalizeTruncatedMspaceCString("strncmp", rhs);
#endif
    const char* a = reinterpret_cast<const char*>(lhs);
    const char* b = reinterpret_cast<const char*>(rhs);
    if (!a || !b) {
        return a == b ? 0 : (a ? 1 : -1);
    }
#ifdef __ANDROID__
    const int result = ExecutorSafeStrcmp("strncmp", lhs, rhs, size, true);
#else
    const int result = std::strncmp(a, b, static_cast<std::size_t>(size));
#endif
#ifdef __ANDROID__
    if (std::getenv("EXECUTOR_TRACE_LIBC_STRCMP_STRINGS") != nullptr) {
        std::uint64_t guest_return = 0;
        std::uint64_t return_off = 0;
        char symbol[96]{};
        char module[64]{};
        if (executor_live_get_current_hle_call_site != nullptr) {
            executor_live_get_current_hle_call_site(&guest_return, &return_off, nullptr, symbol,
                                                    sizeof(symbol), module, sizeof(module));
        }
        const bool is_jit_fault_search_site =
            std::strcmp(module[0] ? module : "", "mono-ps4.sprx") == 0 &&
            return_off >= 0x10200ULL && return_off < 0x10400ULL;
        static std::atomic<int> s_fault_site_budget{512};
        if (is_jit_fault_search_site &&
            s_fault_site_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            char la[48]{};
            char rbuf[48]{};
            const int preview_n =
                static_cast<int>(std::min<u64>(size, static_cast<u64>(sizeof(la) - 1)));
            for (int i = 0; i < preview_n; ++i) {
                unsigned char ch = 0;
                if (!ExecutorReadGuestByte(lhs + static_cast<u64>(i), ch) || ch == 0) break;
                la[i] = std::isprint(ch) ? static_cast<char>(ch) : '.';
            }
            for (int i = 0; i < preview_n; ++i) {
                unsigned char ch = 0;
                if (!ExecutorReadGuestByte(rhs + static_cast<u64>(i), ch) || ch == 0) break;
                rbuf[i] = std::isprint(ch) ? static_cast<char>(ch) : '.';
            }
            char thread_name[32]{};
            pthread_getname_np(pthread_self(), thread_name, sizeof(thread_name));
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_STRNCMP_STR] thread=\"%s\" ret=%p off=0x%llx module=%s "
                                "size=%llu result=%d lhs=%p rhs=%p lhsStr=\"%s\" rhsStr=\"%s\"",
                                thread_name[0] ? thread_name : "<unnamed>",
                                reinterpret_cast<void*>(guest_return),
                                static_cast<unsigned long long>(return_off),
                                module[0] ? module : "<unknown>",
                                static_cast<unsigned long long>(size), result,
                                reinterpret_cast<void*>(lhs), reinterpret_cast<void*>(rhs), la,
                                rbuf);
        }
    }
#endif
#ifdef __ANDROID__
    static std::atomic<int> log_budget{128};
    if (ExecutorVerboseLibcStringLogsEnabled() &&
        log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        unsigned char a0 = 0;
        unsigned char b0 = 0;
        ExecutorReadGuestByte(lhs, a0);
        ExecutorReadGuestByte(rhs, b0);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_STRNCMP] lhs=%p rhs=%p size=%llu result=%d "
                            "lhs0=0x%02x rhs0=0x%02x",
                            reinterpret_cast<void*>(lhs), reinterpret_cast<void*>(rhs),
                            static_cast<unsigned long long>(size), result, a0, b0);
    }
#endif
    return static_cast<s64>(result);
}

s64 ExecutorLibcStrcmp(u64 lhs, u64 rhs) {
#ifdef __ANDROID__
    lhs = ExecutorCanonicalizeTruncatedMspaceCString("strcmp", lhs);
    rhs = ExecutorCanonicalizeTruncatedMspaceCString("strcmp", rhs);
#endif
    const char* a = reinterpret_cast<const char*>(lhs);
    const char* b = reinterpret_cast<const char*>(rhs);
    if (!a || !b) {
        return a == b ? 0 : (a ? 1 : -1);
    }
#ifdef __ANDROID__
    const int result = ExecutorSafeStrcmp("strcmp", lhs, rhs, 0, false);
#else
    const int result = std::strcmp(a, b);
#endif
    if (std::getenv("EXECUTOR_TRACE_LIBC_STRCMP_STRINGS") != nullptr) {
        static std::atomic<unsigned long long> s_diag_n{0};
        const unsigned long long dn = s_diag_n.fetch_add(1, std::memory_order_relaxed);
        std::uint64_t guest_return = 0;
        std::uint64_t return_off = 0;
        char symbol[96]{};
        char module[64]{};
#ifdef __ANDROID__
        if (executor_live_get_current_hle_call_site != nullptr) {
            executor_live_get_current_hle_call_site(&guest_return, &return_off, nullptr, symbol,
                                                    sizeof(symbol), module, sizeof(module));
        }
        char thread_name[32]{};
        pthread_getname_np(pthread_self(), thread_name, sizeof(thread_name));
#else
        guest_return = reinterpret_cast<std::uint64_t>(__builtin_return_address(0));
        const char* thread_name = "desktop";
        if (guest_return >= 0x803190000ULL && guest_return < 0x804000000ULL) {
            return_off = guest_return - 0x803190000ULL;
            std::snprintf(module, sizeof(module), "mono-ps4.sprx");
        } else if (guest_return >= 0x800000000ULL && guest_return < 0x900000000ULL) {
            return_off = guest_return - 0x800000000ULL;
            std::snprintf(module, sizeof(module), "eboot.bin");
        }
#endif
        const bool is_unity_preload = std::strcmp(thread_name, "UnityPreload") == 0;
        const bool is_monoscript_cache_site =
            return_off >= 0x19b800ULL && return_off < 0x19c000ULL &&
            std::strcmp(module[0] ? module : "", "eboot.bin") == 0;
        const bool is_jit_fault_search_site =
            std::strcmp(module[0] ? module : "", "mono-ps4.sprx") == 0 &&
            return_off >= 0x10200ULL && return_off < 0x10400ULL;
        static std::atomic<int> s_focused_budget{4096};
        const bool focused =
            is_unity_preload || is_monoscript_cache_site || is_jit_fault_search_site;
        const bool should_log =
            ((dn & 0xF) == 0 && dn < 64000ULL) ||
            (focused && s_focused_budget.fetch_sub(1, std::memory_order_relaxed) > 0);
        if (should_log) {
            char la[48]{};
            char rbuf[48]{};
            for (int i = 0; i + 1 < static_cast<int>(sizeof(la)); ++i) {
                unsigned char ch = 0;
#ifdef __ANDROID__
                if (!ExecutorReadGuestByte(lhs + i, ch) || ch == 0) break;
#else
                ch = reinterpret_cast<const unsigned char*>(lhs)[i];
                if (ch == 0) break;
#endif
                la[i] = std::isprint(ch) ? static_cast<char>(ch) : '.';
            }
            for (int i = 0; i + 1 < static_cast<int>(sizeof(rbuf)); ++i) {
                unsigned char ch = 0;
#ifdef __ANDROID__
                if (!ExecutorReadGuestByte(rhs + i, ch) || ch == 0) break;
#else
                ch = reinterpret_cast<const unsigned char*>(rhs)[i];
                if (ch == 0) break;
#endif
                rbuf[i] = std::isprint(ch) ? static_cast<char>(ch) : '.';
            }
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_STRCMP_STR] n=%llu thread=\"%s\" ret=%p off=0x%llx module=%s "
                                "result=%d lhs=%p rhs=%p lhsStr=\"%s\" rhsStr=\"%s\"",
                                dn, thread_name[0] ? thread_name : "<unnamed>",
                                reinterpret_cast<void*>(guest_return),
                                static_cast<unsigned long long>(return_off),
                                module[0] ? module : "<unknown>", result,
                                reinterpret_cast<void*>(lhs), reinterpret_cast<void*>(rhs), la,
                                rbuf);
#else
            std::fprintf(stderr,
                         "[EXECUTOR_STRCMP_STR] n=%llu thread=\"%s\" ret=%p off=0x%llx "
                         "module=%s result=%d lhs=%p rhs=%p lhsStr=\"%s\" rhsStr=\"%s\"\n",
                         dn, thread_name, reinterpret_cast<void*>(guest_return),
                         static_cast<unsigned long long>(return_off),
                         module[0] ? module : "<unknown>", result, reinterpret_cast<void*>(lhs),
                         reinterpret_cast<void*>(rhs), la, rbuf);
            std::fflush(stderr);
#endif
        }
    }
#ifdef __ANDROID__
    static std::atomic<int> log_budget{128};
    if (ExecutorVerboseLibcStringLogsEnabled() &&
        log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_STRCMP] lhs=%p rhs=%p result=%d",
                            reinterpret_cast<void*>(lhs), reinterpret_cast<void*>(rhs), result);
    }
#endif
    return static_cast<s64>(result);
}

u64 ExecutorLibcStrtokImpl(u64 str, u64 delim, u64 saveptr, const bool explicit_saveptr,
                           const char* func_name) {
#ifdef __ANDROID__
    str = ExecutorCanonicalizeTruncatedMspaceCString(func_name, str);
    delim = ExecutorCanonicalizeTruncatedMspaceCString(func_name, delim);

    std::array<bool, 256> delimiter{};
    bool have_delimiter = false;
    if (delim != 0) {
        for (u64 i = 0; i < 256; ++i) {
            unsigned char ch = 0;
            if (!ExecutorReadGuestByte(delim + i, ch)) {
                ExecutorLogInvalidLibcPtr(func_name, delim + i);
                break;
            }
            if (ch == '\0') {
                have_delimiter = true;
                break;
            }
            delimiter[ch] = true;
        }
    }
    if (!have_delimiter) {
        return 0;
    }

    thread_local u64 strtok_cursor = 0;
    u64 cursor = str;
    if (cursor == 0) {
        if (explicit_saveptr) {
            if (saveptr == 0 || !ExecutorReadGuestQword(saveptr, cursor)) {
                return 0;
            }
        } else {
            cursor = strtok_cursor;
        }
    }

    constexpr u64 MaxScan = 1 << 20;
    u64 scanned = 0;
    unsigned char ch = 0;
    while (cursor != 0 && scanned < MaxScan) {
        if (!ExecutorReadGuestByte(cursor, ch)) {
            ExecutorLogInvalidLibcPtr(func_name, cursor);
            cursor = 0;
            break;
        }
        if (ch == '\0') {
            cursor = 0;
            break;
        }
        if (!delimiter[ch]) {
            break;
        }
        ++cursor;
        ++scanned;
    }

    u64 token = cursor;
    u64 next_cursor = 0;
    if (token != 0) {
        while (scanned < MaxScan) {
            if (!ExecutorReadGuestByte(cursor, ch)) {
                ExecutorLogInvalidLibcPtr(func_name, cursor);
                token = 0;
                break;
            }
            if (ch == '\0') {
                next_cursor = 0;
                break;
            }
            if (delimiter[ch]) {
                if (!ExecutorWriteGuestByte(cursor, 0)) {
                    ExecutorLogInvalidLibcPtr(func_name, cursor);
                    token = 0;
                } else {
                    next_cursor = cursor + 1;
                }
                break;
            }
            ++cursor;
            ++scanned;
        }
        if (scanned >= MaxScan) {
            token = 0;
            next_cursor = 0;
        }
    }

    if (explicit_saveptr) {
        if (saveptr != 0) {
            ExecutorWriteGuestQword(saveptr, next_cursor);
        }
    } else {
        strtok_cursor = next_cursor;
    }

    static std::atomic<int> log_budget{128};
    if ((ExecutorVerboseLibcStringLogsEnabled() || std::getenv("EXECUTOR_TRACE_LIBC_STRTOK")) &&
        log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        char token_preview[96]{};
        if (token != 0) {
            for (int i = 0; i + 1 < static_cast<int>(sizeof(token_preview)); ++i) {
                unsigned char pc = 0;
                if (!ExecutorReadGuestByte(token + static_cast<u64>(i), pc) || pc == 0) {
                    break;
                }
                token_preview[i] = std::isprint(pc) ? static_cast<char>(pc) : '.';
            }
        }
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_STRTOK] func=%s str=%p delim=%p saveptr=%p "
                            "token=%p next=%p tokenText=\"%s\"",
                            func_name, reinterpret_cast<void*>(str), reinterpret_cast<void*>(delim),
                            reinterpret_cast<void*>(saveptr), reinterpret_cast<void*>(token),
                            reinterpret_cast<void*>(next_cursor),
                            token != 0 ? token_preview : "<null>");
    }
    return token;
#else
    if (explicit_saveptr) {
        char** save = reinterpret_cast<char**>(saveptr);
        return reinterpret_cast<u64>(
            strtok_r(reinterpret_cast<char*>(str), reinterpret_cast<const char*>(delim), save));
    }
    return reinterpret_cast<u64>(std::strtok(reinterpret_cast<char*>(str),
                                             reinterpret_cast<const char*>(delim)));
#endif
}

u64 ExecutorLibcStrtok(u64 str, u64 delim) {
    return ExecutorLibcStrtokImpl(str, delim, 0, false, "strtok");
}

u64 ExecutorLibcStrtokR(u64 str, u64 delim, u64 saveptr) {
    return ExecutorLibcStrtokImpl(str, delim, saveptr, true, "strtok_r");
}

u64 ExecutorLibcStrtokS(u64 str, u64 delim, u64 context) {
    return ExecutorLibcStrtokImpl(str, delim, context, true, "strtok_s");
}

std::mutex g_executor_env_mutex;
std::unordered_map<std::string, std::string> g_executor_env_strings;

static bool ExecutorReadEnvString(u64 value, std::string& out) {
    out.clear();
    const char* text = reinterpret_cast<const char*>(value);
    if (!text) {
        return false;
    }
    for (std::size_t i = 0; i < 4096; ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (ch == '\0') {
            return !out.empty();
        }
        if (ch < 0x20 || ch > 0x7e) {
            out.clear();
            return false;
        }
        out.push_back(static_cast<char>(ch));
    }
    out.clear();
    return false;
}

u64 ExecutorLibcGetenv(u64 name_ptr) {
    std::string name;
    if (!ExecutorReadEnvString(name_ptr, name)) {
        return 0;
    }
    const char* value = std::getenv(name.c_str());
    if (value == nullptr && name == "MONO_PATH") {
        value = "/app0/Media/Managed";
    }
    static const int s_boehm_no_gc = [] {
        std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/lsx4-home/run-boehm-no-gc", "r");
        if (f) {
            std::fclose(f);
            return 1;
        }
        return 0;
    }();
    if (s_boehm_no_gc) {
        if (value == nullptr && name == "GC_DONT_GC") {
            value = "1";
        }
        if (value == nullptr && name == "GC_INITIAL_HEAP_SIZE") {
            value = "268435456";
        }
    }
    if (value == nullptr && (name == "MONO_LOG_LEVEL" || name == "MONO_LOG_MASK")) {
        static const int s_mono_verbose = [] {
            std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/lsx4-home/run-mono-verbose-log", "r");
            if (f) {
                std::fclose(f);
                return 1;
            }
            return 0;
        }();
        if (s_mono_verbose) {
            value = (name == "MONO_LOG_LEVEL") ? "debug" : "all";
        }
    }
#ifdef __ANDROID__
    if (std::FILE* gf = std::fopen(
            "/data/data/app.lsx4.android/files/executor-getenv.log", "a")) {
        std::fprintf(gf, "%s = %s\n", name.c_str(), value ? value : "<null>");
        std::fclose(gf);
    }
#endif
    if (!value) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_GETENV] name=%s result=<null>", name.c_str());
#endif
        return 0;
    }
    std::lock_guard lock(g_executor_env_mutex);
    auto& stored = g_executor_env_strings[name];
    stored = value;
#ifdef __ANDROID__
    static std::atomic<int> log_budget{256};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_GETENV] name=%s value=%s ptr=%p", name.c_str(),
                            stored.c_str(), stored.c_str());
    }
#endif
    return reinterpret_cast<u64>(stored.c_str());
}

s64 ExecutorLibcSetenv(u64 name_ptr, u64 value_ptr, u64 overwrite) {
    std::string name;
    std::string value;
    if (!ExecutorReadEnvString(name_ptr, name)) {
        return -1;
    }
    if (!ExecutorReadEnvString(value_ptr, value)) {
        value.clear();
    }
#ifdef _WIN32
    int rc = 0;
    if (overwrite || std::getenv(name.c_str()) == nullptr) {
        rc = _putenv_s(name.c_str(), value.c_str());
    }
#else
    const int rc = setenv(name.c_str(), value.c_str(), overwrite ? 1 : 0);
#endif
    if (rc == 0) {
        std::lock_guard lock(g_executor_env_mutex);
        g_executor_env_strings[name] = value;
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIBC_SETENV] name=%s value=%s overwrite=%llu rc=%d",
                        name.c_str(), value.c_str(), static_cast<unsigned long long>(overwrite),
                        rc);
#endif
    return rc == 0 ? 1 : 0;
}

s64 ExecutorLibcUnsetenv(u64 name_ptr) {
    std::string name;
    if (!ExecutorReadEnvString(name_ptr, name)) {
        return -1;
    }
#ifdef _WIN32
    const int rc = _putenv_s(name.c_str(), "");
#else
    const int rc = unsetenv(name.c_str());
#endif
    {
        std::lock_guard lock(g_executor_env_mutex);
        g_executor_env_strings.erase(name);
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIBC_UNSETENV] name=%s rc=%d", name.c_str(), rc);
#endif
    return rc == 0 ? 1 : 0;
}

u64 ExecutorPngMalloc(u64 png_ptr, u64 size) {
    void* ptr = std::malloc(static_cast<std::size_t>(std::max<u64>(size, 1)));
    RememberExecutorAllocation(ptr);
    ExecutorAllocLog(true, size, "[EXECUTOR_AEROLIB_ALLOC] png_malloc png=%p size=%llu return=%p",
                     reinterpret_cast<void*>(png_ptr), static_cast<unsigned long long>(size), ptr);
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] png_malloc png=%p size=%llu return=%p\n",
                 reinterpret_cast<void*>(png_ptr), static_cast<unsigned long long>(size), ptr);
    std::fflush(stderr);
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorPngCalloc(u64 png_ptr, u64 size) {
    void* ptr = std::calloc(1, static_cast<std::size_t>(std::max<u64>(size, 1)));
    RememberExecutorAllocation(ptr);
    ExecutorAllocLog(true, size, "[EXECUTOR_AEROLIB_ALLOC] png_calloc png=%p size=%llu return=%p",
                     reinterpret_cast<void*>(png_ptr), static_cast<unsigned long long>(size), ptr);
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_ALLOC] png_calloc png=%p size=%llu return=%p\n",
                 reinterpret_cast<void*>(png_ptr), static_cast<unsigned long long>(size), ptr);
    std::fflush(stderr);
    return reinterpret_cast<u64>(ptr);
}

u64 ExecutorPngFree(u64 png_ptr, u64 ptr) {
    void* value = reinterpret_cast<void*>(ptr);
    const bool known = ForgetExecutorAllocation(value);
    if (known) {
        std::free(value);
    }
    ExecutorAllocLog(true, 0, "[EXECUTOR_AEROLIB_ALLOC] png_free png=%p ptr=%p known=%d return=0",
                     reinterpret_cast<void*>(png_ptr), value, known ? 1 : 0);
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_ALLOC] png_free png=%p ptr=%p known=%d return=0\n",
                 reinterpret_cast<void*>(png_ptr), value, known ? 1 : 0);
    std::fflush(stderr);
    return 0;
}

u64 ExecutorRaise(u64 signal) {
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_SIGNAL] raise signal=%llu ignored return=0\n",
                 static_cast<unsigned long long>(signal));
    std::fflush(stderr);
    return 0;
}

u64 ExecutorAbort() {
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_SIGNAL] abort ignored return=0\n");
    std::fflush(stderr);
    return 0;
}

u64 ExecutorKill(u64 pid, u64 signal) {
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_SIGNAL] kill pid=%llu signal=%llu ignored return=0\n",
                 static_cast<unsigned long long>(pid), static_cast<unsigned long long>(signal));
    std::fflush(stderr);
    return 0;
}

u64 ExecutorPthreadSigmask(u64 how, u64, u64 old_set) {
    if (old_set) {
        const u64 empty_sigset[2] = {};
        std::memcpy(reinterpret_cast<void*>(old_set), empty_sigset, sizeof(empty_sigset));
    }
    std::fprintf(stderr,
                 "[EXECUTOR_AEROLIB_SIGNAL] pthread_sigmask how=%llu old_set=%p return=0\n",
                 static_cast<unsigned long long>(how), reinterpret_cast<void*>(old_set));
    std::fflush(stderr);
    return 0;
}

struct ExecutorDirHandle {
    s32 fd = -1;
    s64 base = 0;
    std::vector<std::uint8_t> buffer = std::vector<std::uint8_t>(8192);
    std::size_t valid = 0;
    std::size_t offset = 0;
    Libraries::Kernel::OrbisKernelDirent current{};
    std::string path;
};

u64 ExecutorOpendir(u64 raw_path) {
    const auto* path = reinterpret_cast<const char*>(raw_path);
    if (path == nullptr || path[0] == '\0') {
        errno = EFAULT;
        std::fprintf(stderr, "[EXECUTOR_AEROLIB_DIR] opendir path=<null> return=0 errno=%d\n",
                     errno);
        std::fflush(stderr);
        return 0;
    }

    const s32 fd = Libraries::Kernel::posix_open(
        path, Libraries::Kernel::ORBIS_KERNEL_O_RDONLY |
                  Libraries::Kernel::ORBIS_KERNEL_O_DIRECTORY,
        0);
    if (fd < 0) {
        std::fprintf(stderr, "[EXECUTOR_AEROLIB_DIR] opendir path=%s return=0 errno=%d\n", path,
                     errno);
        std::fflush(stderr);
        return 0;
    }

    auto* dir = new ExecutorDirHandle();
    dir->fd = fd;
    dir->path = path;
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_DIR] opendir path=%s fd=%d return=%p\n", path, fd,
                 dir);
    std::fflush(stderr);
    return reinterpret_cast<u64>(dir);
}

u64 ExecutorOpendir2(u64 raw_path, u64) {
    return ExecutorOpendir(raw_path);
}

u64 ExecutorReaddir(u64 raw_dir) {
    auto* dir = reinterpret_cast<ExecutorDirHandle*>(raw_dir);
    if (dir == nullptr || dir->fd < 0) {
        errno = EBADF;
        std::fprintf(stderr, "[EXECUTOR_AEROLIB_DIR] readdir dir=%p return=0 errno=%d\n", dir,
                     errno);
        std::fflush(stderr);
        return 0;
    }

    for (;;) {
        while (dir->offset + sizeof(u32) + sizeof(u16) <= dir->valid) {
            const auto* source = reinterpret_cast<const Libraries::Kernel::OrbisKernelDirent*>(
                dir->buffer.data() + dir->offset);
            const std::size_t reclen = source->d_reclen;
            if (reclen < 8 || dir->offset + reclen > dir->valid) {
                dir->offset = dir->valid;
                break;
            }
            dir->offset += reclen;
            if (source->d_namlen == 0) {
                continue;
            }
            std::memset(&dir->current, 0, sizeof(dir->current));
            std::memcpy(&dir->current, source, std::min<std::size_t>(sizeof(dir->current), reclen));
            dir->current.d_name[Libraries::Kernel::ORBIS_MAX_PATH] = '\0';
            std::fprintf(stderr,
                         "[EXECUTOR_AEROLIB_DIR] readdir path=%s name=%s type=%u reclen=%u\n",
                         dir->path.c_str(), dir->current.d_name, dir->current.d_type,
                         dir->current.d_reclen);
            std::fflush(stderr);
            return reinterpret_cast<u64>(&dir->current);
        }

        const s64 read = Libraries::Kernel::getdirentries(
            dir->fd, reinterpret_cast<char*>(dir->buffer.data()), dir->buffer.size(), &dir->base);
        if (read <= 0) {
            std::fprintf(stderr, "[EXECUTOR_AEROLIB_DIR] readdir path=%s eof read=%lld\n",
                         dir->path.c_str(), static_cast<long long>(read));
            std::fflush(stderr);
            return 0;
        }
        dir->valid = static_cast<std::size_t>(read);
        dir->offset = 0;
    }
}

u64 ExecutorReaddirR(u64 raw_dir, u64 raw_entry, u64 raw_result) {
    const u64 result = ExecutorReaddir(raw_dir);
    auto** result_out = reinterpret_cast<void**>(raw_result);
    if (result == 0) {
        if (result_out != nullptr) {
            *result_out = nullptr;
        }
        return 0;
    }
    auto* entry_out = reinterpret_cast<Libraries::Kernel::OrbisKernelDirent*>(raw_entry);
    if (entry_out != nullptr) {
        std::memcpy(entry_out, reinterpret_cast<void*>(result), sizeof(*entry_out));
        if (result_out != nullptr) {
            *result_out = entry_out;
        }
    } else if (result_out != nullptr) {
        *result_out = nullptr;
    }
    return 0;
}

u64 ExecutorClosedir(u64 raw_dir) {
    auto* dir = reinterpret_cast<ExecutorDirHandle*>(raw_dir);
    if (dir == nullptr) {
        errno = EBADF;
        return static_cast<u64>(-1);
    }
    const s32 fd = dir->fd;
    const auto path = dir->path;
    if (fd >= 0) {
        Libraries::Kernel::posix_close(fd);
    }
    delete dir;
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_DIR] closedir path=%s fd=%d return=0\n",
                 path.c_str(), fd);
    std::fflush(stderr);
    return 0;
}

struct ExecutorBgftDownloadParam {
    int user_id;
    int entitlement_type;
    u64 id;
    u64 content_url;
    u64 content_ex_url;
    u64 content_name;
    u64 icon_path;
    u64 sku_id;
    int option;
    int reserved;
    u64 playgo_scenario_id;
    u64 release_date;
    u64 package_type;
    u64 package_sub_type;
    u64 package_size;
};

struct ExecutorBgftDownloadParamEx {
    ExecutorBgftDownloadParam param;
    u32 slot;
};

struct ExecutorBgftTaskProgress {
    u32 bits;
    int error_result;
    u64 length;
    u64 transferred;
    u64 length_total;
    u64 transferred_total;
    u32 num_index;
    u32 num_total;
    u32 rest_sec;
    u32 rest_sec_total;
    int preparing_percent;
    int local_copy_percent;
};

struct ExecutorInstallTask {
    std::string title_id;
    std::string content_url;
    std::string content_name;
    u64 length = 1;
    bool started = false;
    bool completed = false;
};

std::mutex g_executor_install_mutex;
std::unordered_map<int, ExecutorInstallTask> g_executor_install_tasks;
std::unordered_set<std::string> g_executor_installed_titles;
int g_executor_next_install_task_id = 0x4100;

std::string ExecutorExtractTitleId(const std::string& path);

std::filesystem::path ExecutorRuntimeRoot() {
    const char* root = std::getenv("EXECUTOR_LSX4_RUNTIME_ROOT");
    return root && root[0] ? std::filesystem::path(root) : std::filesystem::path{};
}

std::filesystem::path ExecutorInstallMarkerPath(const std::string& title_id) {
    const auto root = ExecutorRuntimeRoot();
    if (root.empty() || title_id.empty()) {
        return {};
    }
    return root / "runtime-fs" / "user" / "app" / title_id / "executor-installed.json";
}

bool ExecutorTitleHasInstallMarker(const std::string& title_id) {
    const auto marker = ExecutorInstallMarkerPath(title_id);
    if (marker.empty()) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::exists(marker, ec);
}

void ExecutorPersistInstalledTitle(const std::string& title_id, const std::string& path) {
    const auto marker = ExecutorInstallMarkerPath(title_id);
    if (marker.empty()) {
        return;
    }
    std::error_code ec;
    std::filesystem::create_directories(marker.parent_path(), ec);
    std::ofstream out(marker, std::ios::trunc);
    if (!out) {
        return;
    }
    out << "{\"title\":\"" << title_id << "\",\"path\":\"" << path
        << "\",\"source\":\"executor-install-manager\"}\n";
}

extern "C" void executor_install_manager_note_local_package_open(const char* guest_path) {
    const std::string path = guest_path ? guest_path : "";
    const auto title_id = ExecutorExtractTitleId(path);
    if (title_id.empty() || title_id == "NPXS39041") {
        return;
    }

    bool first_seen = false;
    int task_id = 0;
    {
        std::scoped_lock lk{g_executor_install_mutex};
        first_seen = g_executor_installed_titles.insert(title_id).second;
        task_id = g_executor_next_install_task_id++;
        g_executor_install_tasks[task_id] = ExecutorInstallTask{
            .title_id = title_id,
            .content_url = path,
            .content_name = title_id,
            .length = 1,
            .started = true,
            .completed = true,
        };
    }
    ExecutorPersistInstalledTitle(title_id, path);

    const auto log_line = [](const char* format, auto... args) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native", format, args...);
#endif
        std::fprintf(stderr, format, args...);
        std::fputc('\n', stderr);
    };
    log_line("[EXECUTOR_INSTALL_MANAGER] op=file_transport_install path=%s title=%s first_seen=%d result=0",
             path.c_str(), title_id.c_str(), first_seen ? 1 : 0);
    log_line("[EXECUTOR_INSTALL_MANAGER] op=get_title_id path=%s title=%s is_app=1 result=0",
             path.c_str(), title_id.c_str());
    log_line("[EXECUTOR_INSTALL_MANAGER] op=bgft_register task=%d title=%s url=%s name=%s slot=0 length=1 result=0",
             task_id, title_id.c_str(), path.c_str(), title_id.c_str());
    log_line("[EXECUTOR_INSTALL_MANAGER] op=bgft_start task=%d title=%s result=0", task_id,
             title_id.c_str());
    log_line("[EXECUTOR_INSTALL_MANAGER] op=bgft_progress task=%d title=%s transferred=1 length=1 result=0",
             task_id, title_id.c_str());
    log_line("[EXECUTOR_INSTALL_MANAGER] op=app_install_pkg path=%s title=%s result=0",
             path.c_str(), title_id.c_str());
    std::fflush(stderr);
}

std::string ExecutorReadGuestString(u64 ptr, std::size_t limit = 4096) {
    if (!ptr) {
        return {};
    }
    const char* value = reinterpret_cast<const char*>(ptr);
    std::string out;
    out.reserve(64);
    for (std::size_t i = 0; i < limit; ++i) {
        const char c = value[i];
        if (c == '\0') {
            break;
        }
        out.push_back(c);
    }
    return out;
}

s64 ExecutorLibcVasprintf(u64 out_ptr, u64 fmt_ptr, u64 va_ptr) {
    auto write_null_out = [&]() {
        if (out_ptr) {
            *reinterpret_cast<u64*>(out_ptr) = 0;
        }
    };
    if (fmt_ptr == 0) {
        write_null_out();
        return -1;
    }
    const std::string fmt = ExecutorReadGuestString(fmt_ptr, 1 << 16);

    u64 vp = va_ptr;
    auto rd32 = [](u64 a) -> u32 { return a ? *reinterpret_cast<u32*>(a) : 0; };
    auto rd64 = [](u64 a) -> u64 { return a ? *reinterpret_cast<u64*>(a) : 0; };
    auto wr32 = [](u64 a, u32 v) { if (a) *reinterpret_cast<u32*>(a) = v; };
    auto wr64 = [](u64 a, u64 v) { if (a) *reinterpret_cast<u64*>(a) = v; };
    auto next_gp = [&]() -> u64 {
        if (!vp) return 0;
        const u32 gp = rd32(vp + 0);
        const u64 reg_save = rd64(vp + 16);
        const u64 overflow = rd64(vp + 8);
        if (gp < 48 && reg_save) {
            const u64 v = rd64(reg_save + gp);
            wr32(vp + 0, gp + 8);
            return v;
        }
        const u64 v = rd64(overflow);
        wr64(vp + 8, overflow + 8);
        return v;
    };
    auto next_fp = [&]() -> double {
        if (!vp) return 0.0;
        const u32 fp = rd32(vp + 4);
        const u64 reg_save = rd64(vp + 16);
        const u64 overflow = rd64(vp + 8);
        u64 bits;
        if (fp < 176 && reg_save) {
            bits = rd64(reg_save + fp);
            wr32(vp + 4, fp + 16);
        } else {
            bits = rd64(overflow);
            wr64(vp + 8, overflow + 8);
        }
        double dv;
        std::memcpy(&dv, &bits, sizeof(dv));
        return dv;
    };

    std::string out;
    char buf[512];
    for (std::size_t i = 0; i < fmt.size(); ++i) {
        const char c = fmt[i];
        if (c != '%') {
            out.push_back(c);
            continue;
        }
        const std::size_t start = i;
        std::string spec = "%";
        ++i;
        while (i < fmt.size() && std::strchr("-+ #0", fmt[i])) spec.push_back(fmt[i++]);
        while (i < fmt.size() && (std::isdigit(static_cast<unsigned char>(fmt[i])) || fmt[i] == '.')) {
            spec.push_back(fmt[i++]);
        }
        if (i < fmt.size() && fmt[i] == '*') {
            spec += std::to_string(static_cast<int>(static_cast<s64>(next_gp())));
            ++i;
        }
        bool wide64 = false;
        while (i < fmt.size() && std::strchr("lhjztLq", fmt[i])) {
            if (std::strchr("ljztLq", fmt[i])) wide64 = true;
            ++i;
        }
        if (i >= fmt.size()) {
            out.append(fmt.substr(start));
            break;
        }
        const char conv = fmt[i];
        switch (conv) {
        case '%':
            out.push_back('%');
            break;
        case 'd':
        case 'i': {
            s64 v = wide64 ? static_cast<s64>(next_gp())
                           : static_cast<s64>(static_cast<s32>(next_gp() & 0xFFFFFFFFull));
            std::snprintf(buf, sizeof(buf), (spec + "lld").c_str(), static_cast<long long>(v));
            out += buf;
            break;
        }
        case 'u':
        case 'x':
        case 'X':
        case 'o': {
            u64 v = wide64 ? next_gp() : (next_gp() & 0xFFFFFFFFull);
            std::snprintf(buf, sizeof(buf), (spec + "ll" + conv).c_str(),
                          static_cast<unsigned long long>(v));
            out += buf;
            break;
        }
        case 'p': {
            std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(next_gp()));
            out += buf;
            break;
        }
        case 'c':
            out.push_back(static_cast<char>(next_gp()));
            break;
        case 's': {
            const u64 sp = next_gp();
            if (sp == 0) {
                out += "(null)";
            } else {
                out += ExecutorReadGuestString(sp, 1 << 16);
            }
            break;
        }
        case 'f':
        case 'F':
        case 'g':
        case 'G':
        case 'e':
        case 'E':
        case 'a':
        case 'A': {
            std::snprintf(buf, sizeof(buf), (spec + conv).c_str(), next_fp());
            out += buf;
            break;
        }
        default:
            out.append(fmt.substr(start, i - start + 1));
            break;
        }
    }

    const std::size_t len = out.size();
    void* gbuf = ExecutorAllocDefaultMspaceOrFallback(16, len + 1, false);
    if (!gbuf) {
        write_null_out();
        return -1;
    }
    std::memcpy(gbuf, out.data(), len);
    reinterpret_cast<char*>(gbuf)[len] = '\0';
    if (out_ptr) {
        *reinterpret_cast<u64*>(out_ptr) = reinterpret_cast<u64>(gbuf);
    }
#ifdef __ANDROID__
    static std::atomic<int> log_budget{4};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_VASPRINTF] fmt=\"%s\" -> len=%zu out=\"%.120s\"",
                            fmt.c_str(), len, out.c_str());
    }
#endif
    return static_cast<s64>(len);
}

extern "C" u64 ExecutorLibcFwrite(u64 buffer_ptr, u64 element_size, u64 element_count,
                                  u64 file_ptr);

s64 ExecutorLibcVfprintf(u64 file_ptr, u64 fmt_ptr, u64 va_ptr) {
    u64 out_ptr = 0;
    const s64 len = ExecutorLibcVasprintf(reinterpret_cast<u64>(&out_ptr), fmt_ptr, va_ptr);
    if (len < 0 || out_ptr == 0) {
        return len;
    }

    const std::string text = ExecutorReadGuestString(out_ptr, 1 << 20);
    u64 written = 0;
    if (!text.empty()) {
        written = ExecutorLibcFwrite(out_ptr, 1, text.size(), file_ptr);
        if (written != text.size()) {
            std::fwrite(text.data(), 1, text.size(), stderr);
            std::fflush(stderr);
            written = text.size();
        }
    }
#ifdef __ANDROID__
    static std::atomic<int> log_budget{4};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_VFPRINTF] file=0x%llx fmt=0x%llx va=0x%llx len=%lld text=\"%.160s\"",
                            static_cast<unsigned long long>(file_ptr),
                            static_cast<unsigned long long>(fmt_ptr),
                            static_cast<unsigned long long>(va_ptr), static_cast<long long>(len),
                            text.c_str());
    }
#endif
    ExecutorLibcFree(out_ptr);
    return static_cast<s64>(written);
}

static u64 ExecutorCurrentGuestVarargOverflowArea() {
#ifdef __ANDROID__
    u64 guest_rsp = 0;
    if (executor_live_get_current_hle_guest_rsp != nullptr &&
        executor_live_get_current_hle_guest_rsp(&guest_rsp) == 0 && guest_rsp != 0) {
        return guest_rsp + sizeof(u64);
    }
#endif
    return 0;
}

static void ExecutorLogFormattedCopy(const char* tag, u64 dst, u64 size, u64 fmt_ptr, s64 len) {
#ifdef __ANDROID__
    if (!ExecutorVerboseLibcFormattedLogsEnabled()) {
        return;
    }
    static std::atomic<int> log_budget{192};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    const std::string fmt = ExecutorReadGuestString(fmt_ptr, 256);
    const std::string out = dst != 0 ? ExecutorReadGuestString(dst, 256) : std::string{};
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIBC_FORMATTED] fn=%s dst=0x%llx size=%llu fmt=0x%llx "
                        "len=%lld fmtStr=\"%.160s\" out=\"%.160s\"",
                        tag ? tag : "?", static_cast<unsigned long long>(dst),
                        static_cast<unsigned long long>(size),
                        static_cast<unsigned long long>(fmt_ptr), static_cast<long long>(len),
                        fmt.c_str(), out.c_str());
#else
    (void)tag;
    (void)dst;
    (void)size;
    (void)fmt_ptr;
    (void)len;
#endif
}

s64 ExecutorLibcFprintf(u64 file_ptr, u64 fmt_ptr, u64 a2, u64 a3, u64 a4, u64 a5) {
    alignas(16) u64 reg_save[6] = {file_ptr, fmt_ptr, a2, a3, a4, a5};
    alignas(16) u64 overflow[4] = {};
    const u64 guest_overflow = ExecutorCurrentGuestVarargOverflowArea();
    struct GuestVaList {
        u32 gp_offset;
        u32 fp_offset;
        u64 overflow_arg_area;
        u64 reg_save_area;
    } va{16, 48, guest_overflow ? guest_overflow : reinterpret_cast<u64>(overflow),
         reinterpret_cast<u64>(reg_save)};

    const s64 len = ExecutorLibcVfprintf(file_ptr, fmt_ptr, reinterpret_cast<u64>(&va));
#ifdef __ANDROID__
    static std::atomic<int> log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_FPRINTF] file=0x%llx fmt=0x%llx len=%lld",
                            static_cast<unsigned long long>(file_ptr),
                            static_cast<unsigned long long>(fmt_ptr), static_cast<long long>(len));
    }
#endif
    return len;
}

u64 ExecutorLibcStrcpy(u64 dst, u64 src) {
    if (dst == 0) {
        return dst;
    }
    if (ExecutorLibcRejectHostImageWrite("strcpy", dst, 1)) {
        return dst;
    }
#ifdef __ANDROID__
    src = ExecutorCanonicalizeTruncatedMspaceCString("strcpy", src);
    u64 i = 0;
    for (; i < (1ULL << 20); ++i) {
        unsigned char ch = 0;
        if (!ExecutorReadGuestByte(src + i, ch)) {
            ExecutorLogInvalidLibcPtr("strcpy", src + i);
            reinterpret_cast<char*>(dst)[i] = '\0';
            break;
        }
        reinterpret_cast<unsigned char*>(dst)[i] = ch;
        if (ch == '\0') {
            break;
        }
    }
#else
    if (src != 0) {
        std::strcpy(reinterpret_cast<char*>(dst), reinterpret_cast<const char*>(src));
    } else {
        reinterpret_cast<char*>(dst)[0] = '\0';
    }
#endif
    ExecutorWatchGateWrite("strcpy", dst, 0x40, 0);
#ifdef __ANDROID__
    if (dst >= 0x8003452b0ull && dst < 0x800345ac0ull) {
        char preview[17] = {0};
        std::memcpy(preview, reinterpret_cast<const char*>(dst), 16);
        for (int p = 0; p < 16; ++p) {
            if (preview[p] != 0 && (preview[p] < 0x20 || preview[p] > 0x7e)) {
                preview[p] = '.';
            }
        }
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_STRCPY_NAME] dst=0x%llx src=0x%llx name='%s'",
                            static_cast<unsigned long long>(dst),
                            static_cast<unsigned long long>(src), preview);
    }
#endif
    return dst;
}

u64 ExecutorLibcStrdup(u64 src) {
    if (src == 0) {
        return 0;
    }
#ifdef __ANDROID__
    src = ExecutorCanonicalizeTruncatedMspaceCString("strdup", src);
    const ExecutorGuestCStringScan scan = ExecutorScanGuestCString(src);
    if (scan.status != ExecutorGuestCStringStatus::Terminated ||
        scan.length == std::numeric_limits<u64>::max()) {
        ExecutorLogInvalidLibcPtr("strdup", scan.fault_address);
        return 0;
    }
    const u64 copy_size = scan.length + 1;
#else
    const u64 copy_size = static_cast<u64>(std::strlen(reinterpret_cast<const char*>(src))) + 1;
#endif
    const u64 dst = ExecutorLibcMalloc(copy_size);
    if (dst == 0) {
        return 0;
    }
#ifdef __ANDROID__
    if (!ExecutorReadGuestBytes(src, reinterpret_cast<void*>(dst), copy_size)) {
        ExecutorLibcFree(dst);
        ExecutorLogInvalidLibcPtr("strdup", src);
        return 0;
    }
#else
    std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), copy_size);
#endif
    return dst;
}

u64 ExecutorLibcStrncpy(u64 dst, u64 src, u64 count) {
    if (dst == 0) {
        return dst;
    }
    if (ExecutorLibcRejectHostImageWrite("strncpy", dst, count ? count : 1)) {
        return dst;
    }
#ifdef __ANDROID__
    src = ExecutorCanonicalizeTruncatedMspaceCString("strncpy", src);
    bool done = false;
    for (u64 i = 0; i < count; ++i) {
        unsigned char ch = 0;
        if (!done) {
            if (!ExecutorReadGuestByte(src + i, ch)) {
                ExecutorLogInvalidLibcPtr("strncpy", src + i);
                done = true;
                ch = 0;
            } else if (ch == '\0') {
                done = true;
            }
        }
        reinterpret_cast<unsigned char*>(dst)[i] = done ? 0 : ch;
        if (!done && ch == '\0') {
            done = true;
        }
    }
#else
    if (src != 0) {
        std::strncpy(reinterpret_cast<char*>(dst), reinterpret_cast<const char*>(src),
                     static_cast<std::size_t>(count));
    } else if (count != 0) {
        std::memset(reinterpret_cast<void*>(dst), 0, static_cast<std::size_t>(count));
    }
#endif
    return dst;
}

u64 ExecutorLibcStrlcpy(u64 dst, u64 src, u64 size) {
#ifdef __ANDROID__
    src = ExecutorCanonicalizeTruncatedMspaceCString("strlcpy", src);
    u64 source_len = 0;
    if (src != 0) {
        for (; source_len < (1ULL << 20); ++source_len) {
            unsigned char ch = 0;
            if (!ExecutorReadGuestByte(src + source_len, ch)) {
                ExecutorLogInvalidLibcPtr("strlcpy", src + source_len);
                break;
            }
            if (ch == '\0') {
                break;
            }
            if (dst != 0 && size != 0 && source_len + 1 < size) {
                reinterpret_cast<unsigned char*>(dst)[source_len] = ch;
            }
        }
    }
    if (dst != 0 && size != 0) {
        const u64 nul = std::min(source_len, size - 1);
        reinterpret_cast<char*>(dst)[nul] = '\0';
    }
    return source_len;
#else
    const char* source = reinterpret_cast<const char*>(src);
    const std::size_t source_len = source ? std::strlen(source) : 0;
    if (dst != 0 && size != 0) {
        const std::size_t copy_len = std::min<std::size_t>(source_len, size - 1);
        if (copy_len != 0 && source) {
            std::memcpy(reinterpret_cast<void*>(dst), source, copy_len);
        }
        reinterpret_cast<char*>(dst)[copy_len] = '\0';
    }
    return source_len;
#endif
}

u64 ExecutorLibcStrcat(u64 dst, u64 src) {
    if (dst == 0) {
        return dst;
    }
    const u64 dst_len = ExecutorLibcStrlen(dst);
    ExecutorLibcStrcpy(dst + dst_len, src);
    return dst;
}

u64 ExecutorLibcStrlcat(u64 dst, u64 src, u64 size) {
    u64 dst_len = 0;
    if (dst != 0) {
        for (; dst_len < size; ++dst_len) {
#ifdef __ANDROID__
            unsigned char ch = 0;
            if (!ExecutorReadGuestByte(dst + dst_len, ch) || ch == '\0') {
                break;
            }
#else
            if (reinterpret_cast<unsigned char*>(dst)[dst_len] == '\0') {
                break;
            }
#endif
        }
    }
    const u64 src_len = ExecutorLibcStrlen(src);
    if (dst == 0 || size == 0 || dst_len >= size) {
        return dst_len + src_len;
    }
    u64 i = 0;
    for (; i < src_len && dst_len + i + 1 < size; ++i) {
#ifdef __ANDROID__
        unsigned char ch = 0;
        if (!ExecutorReadGuestByte(src + i, ch)) {
            break;
        }
        reinterpret_cast<unsigned char*>(dst)[dst_len + i] = ch;
#else
        reinterpret_cast<unsigned char*>(dst)[dst_len + i] =
            reinterpret_cast<const unsigned char*>(src)[i];
#endif
    }
    reinterpret_cast<char*>(dst)[dst_len + i] = '\0';
    return dst_len + src_len;
}

u64 ExecutorLibcStrncat(u64 dst, u64 src, u64 count) {
    if (dst == 0) {
        return dst;
    }
    const u64 dst_len = ExecutorLibcStrlen(dst);
    u64 i = 0;
    for (; i < count; ++i) {
#ifdef __ANDROID__
        unsigned char ch = 0;
        if (!ExecutorReadGuestByte(src + i, ch) || ch == '\0') {
            break;
        }
        reinterpret_cast<unsigned char*>(dst)[dst_len + i] = ch;
#else
        const unsigned char ch = reinterpret_cast<const unsigned char*>(src)[i];
        if (ch == '\0') {
            break;
        }
        reinterpret_cast<unsigned char*>(dst)[dst_len + i] = ch;
#endif
    }
    reinterpret_cast<char*>(dst)[dst_len + i] = '\0';
    return dst;
}

static s64 ExecutorLibcCopyFormattedResult(u64 dst, u64 size, u64 fmt_ptr, u64 va_ptr,
                                           bool bounded) {
    if (dst == 0 && (!bounded || size != 0)) {
        return -1;
    }
    u64 out_ptr = 0;
    const s64 len = ExecutorLibcVasprintf(reinterpret_cast<u64>(&out_ptr), fmt_ptr, va_ptr);
    if (len < 0 || out_ptr == 0) {
        if (bounded && size != 0) {
            reinterpret_cast<char*>(dst)[0] = '\0';
        }
        return len;
    }
    const auto* text = reinterpret_cast<const char*>(out_ptr);
    const std::size_t text_len = static_cast<std::size_t>(len);
    if (bounded) {
        if (size != 0) {
            const std::size_t copy_len =
                std::min<std::size_t>(text_len, static_cast<std::size_t>(size - 1));
            if (copy_len != 0) {
                std::memcpy(reinterpret_cast<void*>(dst), text, copy_len);
            }
            reinterpret_cast<char*>(dst)[copy_len] = '\0';
        }
    } else {
        if (text_len != 0) {
            std::memcpy(reinterpret_cast<void*>(dst), text, text_len);
        }
        reinterpret_cast<char*>(dst)[text_len] = '\0';
    }
    ExecutorLibcFree(out_ptr);
    return len;
}

s64 ExecutorLibcVsnprintf(u64 dst, u64 size, u64 fmt_ptr, u64 va_ptr) {
    const s64 len = ExecutorLibcCopyFormattedResult(dst, size, fmt_ptr, va_ptr, true);
#ifdef __ANDROID__
    static std::atomic<int> log_budget{64};
    if (ExecutorVerboseLibcFormattedLogsEnabled() &&
        log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_VSNPRINTF] dst=0x%llx size=%llu fmt=0x%llx len=%lld",
                            static_cast<unsigned long long>(dst),
                            static_cast<unsigned long long>(size),
                            static_cast<unsigned long long>(fmt_ptr), static_cast<long long>(len));
    }
#endif
    ExecutorLogFormattedCopy("vsnprintf", dst, size, fmt_ptr, len);
    return len;
}

s64 ExecutorLibcVsprintf(u64 dst, u64 fmt_ptr, u64 va_ptr) {
    const s64 len = ExecutorLibcCopyFormattedResult(dst, 0, fmt_ptr, va_ptr, false);
    ExecutorLogFormattedCopy("vsprintf", dst, 0, fmt_ptr, len);
    return len;
}

s64 ExecutorLibcSnprintf(u64 dst, u64 size, u64 fmt_ptr, u64 a3, u64 a4, u64 a5) {
    alignas(16) u64 reg_save[6] = {dst, size, fmt_ptr, a3, a4, a5};
    alignas(16) u64 overflow[4] = {};
    const u64 guest_overflow = ExecutorCurrentGuestVarargOverflowArea();
    struct GuestVaList {
        u32 gp_offset;
        u32 fp_offset;
        u64 overflow_arg_area;
        u64 reg_save_area;
    } va{24, 48, guest_overflow ? guest_overflow : reinterpret_cast<u64>(overflow),
         reinterpret_cast<u64>(reg_save)};
    return ExecutorLibcVsnprintf(dst, size, fmt_ptr, reinterpret_cast<u64>(&va));
}

s64 ExecutorLibcSprintf(u64 dst, u64 fmt_ptr, u64 a2, u64 a3, u64 a4, u64 a5) {
    alignas(16) u64 reg_save[6] = {dst, fmt_ptr, a2, a3, a4, a5};
    alignas(16) u64 overflow[4] = {};
    const u64 guest_overflow = ExecutorCurrentGuestVarargOverflowArea();
    struct GuestVaList {
        u32 gp_offset;
        u32 fp_offset;
        u64 overflow_arg_area;
        u64 reg_save_area;
    } va{16, 48, guest_overflow ? guest_overflow : reinterpret_cast<u64>(overflow),
         reinterpret_cast<u64>(reg_save)};
    return ExecutorLibcVsprintf(dst, fmt_ptr, reinterpret_cast<u64>(&va));
}

std::string ExecutorExtractTitleId(const std::string& path) {
    auto clean_path = path;
    if (const auto query_pos = clean_path.find_first_of("?#"); query_pos != std::string::npos) {
        clean_path.resize(query_pos);
    }
    std::replace(clean_path.begin(), clean_path.end(), '\\', '/');

    const auto filename_slash = clean_path.find_last_of('/');
    auto filename = filename_slash == std::string::npos ? clean_path : clean_path.substr(filename_slash + 1);
    if (const auto dot_pos = filename.find_last_of('.'); dot_pos != std::string::npos) {
        filename.resize(dot_pos);
    }
    if (filename.size() >= 4 && filename.size() <= 16 && filename != "app" &&
        std::all_of(filename.begin(), filename.end(), [](unsigned char ch) {
            return std::isdigit(ch) || (ch >= 'A' && ch <= 'Z');
        })) {
        return filename;
    }

    const std::string marker = "/user/app/";
    const auto app_pos = clean_path.find(marker);
    if (app_pos != std::string::npos) {
        const auto begin = app_pos + marker.size();
        const auto end = clean_path.find('/', begin);
        if (end != std::string::npos && end > begin) {
            return clean_path.substr(begin, end - begin);
        }
    }

    const auto slash = clean_path.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        const auto parent_end = slash;
        const auto parent_begin = clean_path.find_last_of('/', parent_end - 1);
        if (parent_begin != std::string::npos && parent_begin + 1 < parent_end) {
            const auto parent = clean_path.substr(parent_begin + 1, parent_end - parent_begin - 1);
            if (!parent.empty() && parent.size() <= 15) {
                return parent;
            }
        }
    }

    return "EXEC00001";
}

void ExecutorWriteCString(u64 guest_dst, const std::string& value, std::size_t capacity) {
    if (!guest_dst || capacity == 0) {
        return;
    }
    auto* dst = reinterpret_cast<char*>(guest_dst);
    const auto copy_size = std::min<std::size_t>(value.size(), capacity - 1);
    std::memcpy(dst, value.data(), copy_size);
    dst[copy_size] = '\0';
}

u64 ExecutorAppInstUtilInitialize() {
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=appinst_init result=0\n");
    std::fflush(stderr);
    return 0;
}

u64 ExecutorAppInstUtilTerminate() {
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=appinst_term result=0\n");
    std::fflush(stderr);
    return 0;
}

u64 ExecutorAppInstUtilGetTitleIdFromPkg(u64 pkg_path_ptr, u64 title_id_ptr, u64 is_app_ptr) {
    const auto pkg_path = ExecutorReadGuestString(pkg_path_ptr);
    const auto title_id = ExecutorExtractTitleId(pkg_path);
    ExecutorWriteCString(title_id_ptr, title_id, 16);
    if (is_app_ptr) {
        *reinterpret_cast<int*>(is_app_ptr) = 1;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_INSTALL_MANAGER] op=get_title_id path=%s title=%s is_app=1 result=0\n",
                 pkg_path.c_str(), title_id.c_str());
    std::fflush(stderr);
    return 0;
}

u64 ExecutorAppInstUtilAppExists(u64 title_id_ptr, u64 flag_ptr) {
    const auto title_id = ExecutorReadGuestString(title_id_ptr, 64);
    bool exists = false;
    const bool persisted = ExecutorTitleHasInstallMarker(title_id);
    {
        std::scoped_lock lk{g_executor_install_mutex};
        if (persisted) {
            g_executor_installed_titles.insert(title_id);
        }
        exists = g_executor_installed_titles.find(title_id) != g_executor_installed_titles.end();
    }
    if (flag_ptr) {
        *reinterpret_cast<int*>(flag_ptr) = exists ? 1 : 0;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_INSTALL_MANAGER] op=app_exists title=%s flag=%d persisted=%d result=0\n",
                 title_id.c_str(), exists ? 1 : 0, persisted ? 1 : 0);
    std::fflush(stderr);
    return 0;
}

u64 ExecutorAppInstUtilAppUnInstall(u64 title_id_ptr) {
    const auto title_id = ExecutorReadGuestString(title_id_ptr, 64);
    {
        std::scoped_lock lk{g_executor_install_mutex};
        g_executor_installed_titles.erase(title_id);
    }
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=app_uninstall title=%s result=0\n",
                 title_id.c_str());
    std::fflush(stderr);
    return 0;
}

u64 ExecutorAppInstUtilAppPrepareOverwritePkg(u64 pkg_path_ptr) {
    const auto pkg_path = ExecutorReadGuestString(pkg_path_ptr);
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=prepare_overwrite path=%s result=0\n",
                 pkg_path.c_str());
    std::fflush(stderr);
    return 0;
}

u64 ExecutorAppInstUtilGetPrimaryAppSlot(u64 title_id_ptr, u64 slot_ptr) {
    const auto title_id = ExecutorReadGuestString(title_id_ptr, 64);
    if (slot_ptr) {
        *reinterpret_cast<int*>(slot_ptr) = 0;
    }
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=get_primary_slot title=%s slot=0 result=0\n",
                 title_id.c_str());
    std::fflush(stderr);
    return 0;
}

u64 ExecutorAppInstUtilAppInstallPkg(u64 pkg_path_ptr, u64) {
    const auto pkg_path = ExecutorReadGuestString(pkg_path_ptr);
    const auto title_id = ExecutorExtractTitleId(pkg_path);
    {
        std::scoped_lock lk{g_executor_install_mutex};
        g_executor_installed_titles.insert(title_id);
    }
    ExecutorPersistInstalledTitle(title_id, pkg_path);
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=app_install_pkg path=%s title=%s result=0\n",
                 pkg_path.c_str(), title_id.c_str());
    std::fflush(stderr);
    return 0;
}

u64 ExecutorBgftServiceInit(u64 params_ptr) {
    u64 heap_size = 0;
    if (params_ptr) {
        const auto* params = reinterpret_cast<const u64*>(params_ptr);
        heap_size = params[1];
    }
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=bgft_init heap_size=%llu result=0\n",
                 static_cast<unsigned long long>(heap_size));
    std::fflush(stderr);
    return 0;
}

u64 ExecutorBgftServiceTerm() {
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=bgft_term result=0\n");
    std::fflush(stderr);
    return 0;
}

u64 ExecutorBgftRegisterTaskByStorageEx(u64 params_ptr, u64 task_id_ptr) {
    std::string content_url;
    std::string content_name;
    u64 package_size = 0;
    u32 slot = 0;
    if (params_ptr) {
        const auto* params = reinterpret_cast<const ExecutorBgftDownloadParamEx*>(params_ptr);
        content_url = ExecutorReadGuestString(params->param.content_url);
        content_name = ExecutorReadGuestString(params->param.content_name);
        package_size = params->param.package_size;
        slot = params->slot;
    }
    const auto title_id = ExecutorExtractTitleId(content_url);
    const u64 length = std::max<u64>(package_size, 1);
    int task_id = 0;
    {
        std::scoped_lock lk{g_executor_install_mutex};
        task_id = g_executor_next_install_task_id++;
        ExecutorInstallTask task;
        task.title_id = title_id;
        task.content_url = content_url;
        task.content_name = content_name;
        task.length = length;
        g_executor_install_tasks[task_id] = task;
    }
    if (task_id_ptr) {
        *reinterpret_cast<int*>(task_id_ptr) = task_id;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_INSTALL_MANAGER] op=bgft_register task=%d title=%s url=%s name=%s slot=%u length=%llu result=0\n",
                 task_id, title_id.c_str(), content_url.c_str(), content_name.c_str(), slot,
                 static_cast<unsigned long long>(length));
    std::fflush(stderr);
    return 0;
}

u64 ExecutorBgftServiceDownloadStartTask(u64 task_id_value) {
    const int task_id = static_cast<int>(task_id_value);
    std::string title_id;
    {
        std::scoped_lock lk{g_executor_install_mutex};
        auto it = g_executor_install_tasks.find(task_id);
        if (it != g_executor_install_tasks.end()) {
            it->second.started = true;
            title_id = it->second.title_id;
        }
    }
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=bgft_start task=%d title=%s result=0\n",
                 task_id, title_id.c_str());
    std::fflush(stderr);
    return 0;
}

u64 ExecutorBgftServiceDownloadStartTaskAll() {
    std::scoped_lock lk{g_executor_install_mutex};
    for (auto& [task_id, task] : g_executor_install_tasks) {
        task.started = true;
    }
    std::fprintf(stderr, "[EXECUTOR_INSTALL_MANAGER] op=bgft_start_all tasks=%zu result=0\n",
                 g_executor_install_tasks.size());
    std::fflush(stderr);
    return 0;
}

u64 ExecutorBgftServiceDownloadGetProgress(u64 task_id_value, u64 progress_ptr) {
    const int task_id = static_cast<int>(task_id_value);
    ExecutorInstallTask task{};
    bool found = false;
    {
        std::scoped_lock lk{g_executor_install_mutex};
        auto it = g_executor_install_tasks.find(task_id);
        if (it != g_executor_install_tasks.end()) {
            it->second.started = true;
            it->second.completed = true;
            g_executor_installed_titles.insert(it->second.title_id);
            task = it->second;
            found = true;
        }
    }
    if (found) {
        ExecutorPersistInstalledTitle(task.title_id, task.content_url);
    }
    if (!found) {
        task.title_id = "UNKNOWN";
        task.length = 1;
    }
    if (progress_ptr) {
        auto* progress = reinterpret_cast<ExecutorBgftTaskProgress*>(progress_ptr);
        std::memset(progress, 0, sizeof(*progress));
        progress->length = task.length;
        progress->transferred = task.length;
        progress->length_total = task.length;
        progress->transferred_total = task.length;
        progress->num_index = 1;
        progress->num_total = 1;
        progress->preparing_percent = 100;
        progress->local_copy_percent = 100;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_INSTALL_MANAGER] op=bgft_progress task=%d title=%s transferred=%llu length=%llu result=0\n",
                 task_id, task.title_id.c_str(), static_cast<unsigned long long>(task.length),
                 static_cast<unsigned long long>(task.length));
    std::fflush(stderr);
    return 0;
}

extern "C" int ExecutorSceKeyboardInit() {
    return 0;
}
extern "C" int ExecutorSceKeyboardOpen(int, int, int, void*) {
    return 1;
}
extern "C" int ExecutorSceKeyboardReadState(int, void* state) {
    if (state) {
        std::memset(state, 0, 0x100);
    }
    return 0;
}
extern "C" int ExecutorSceKeyboardGetKey2Char(int, unsigned int, void* out) {
    if (out) {
        *static_cast<int*>(out) = 0;
    }
    return 0;
}

extern "C" int ExecutorCxaGuardAcquire(unsigned char* guard) {
    if (!guard) {
        return 0;
    }
    return guard[0] ? 0 : 1;
}
extern "C" u64 ExecutorCxaGuardRelease(unsigned char* guard) {
    if (guard) {
        guard[0] = 1;
    }
    return 0;
}
extern "C" u64 ExecutorCxaGuardAbort(unsigned char*) {
    return 0;
}

extern "C" u64 ExecutorCppStaticInitNoop(void* self) {
    return reinterpret_cast<u64>(self);
}

extern "C" u64 ExecutorCppStaticDtorNoop(void*) {
    return 0;
}

std::array<std::recursive_mutex, 64>& ExecutorLibcSyslocks() {
    static std::array<std::recursive_mutex, 64> locks{};
    return locks;
}

extern "C" u64 ExecutorLibcLocksyslock(int lock) {
    static std::atomic_int log_budget{16};
    auto& locks = ExecutorLibcSyslocks();
    const auto index = static_cast<std::size_t>(static_cast<unsigned int>(lock)) % locks.size();
    locks[index].lock();
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_DEBUG, "LSX4Native",
                            "[EXECUTOR_LIBC_SYSLOCK] lock id=%d index=%zu", lock, index);
#else
        std::fprintf(stderr, "[EXECUTOR_LIBC_SYSLOCK] lock id=%d index=%zu\n", lock, index);
#endif
    }
    return 0;
}

extern "C" u64 ExecutorLibcUnlocksyslock(int lock) {
    static std::atomic_int log_budget{16};
    auto& locks = ExecutorLibcSyslocks();
    const auto index = static_cast<std::size_t>(static_cast<unsigned int>(lock)) % locks.size();
    locks[index].unlock();
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_DEBUG, "LSX4Native",
                            "[EXECUTOR_LIBC_SYSLOCK] unlock id=%d index=%zu", lock, index);
#else
        std::fprintf(stderr, "[EXECUTOR_LIBC_SYSLOCK] unlock id=%d index=%zu\n", lock, index);
#endif
    }
    return 0;
}

extern "C" int ExecutorCxaAtexit(void (*func)(void*), void* arg, void* dso_handle) {
    static std::atomic_int log_budget{16};
    if (std::getenv("EXECUTOR_TRACE_CXA_ATEXIT") == nullptr &&
        log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return 0;
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_CXA_ATEXIT] func=%p arg=%p dso=%p result=0",
                        reinterpret_cast<void*>(func), arg, dso_handle);
#else
    std::fprintf(stderr, "[EXECUTOR_CXA_ATEXIT] func=%p arg=%p dso=%p result=0\n",
                 reinterpret_cast<void*>(func), arg, dso_handle);
#endif
    return 0;
}

namespace {

struct ExecutorFiosFile {
    std::FILE* fp{};
    std::string guest_path;
    std::filesystem::path host_path;
};

struct ExecutorFiosOp {
    s64 error{};
    u64 actual{};
    bool done{true};
    std::string guest_path;
};

std::mutex g_executor_fios_mutex;
std::unordered_map<u64, ExecutorFiosFile> g_executor_fios_files;
std::unordered_map<u64, ExecutorFiosOp> g_executor_fios_ops;
u64 g_executor_fios_next_handle = 0x10001;
u64 g_executor_fios_next_op = 0x20001;
std::vector<std::pair<std::string, std::string>> g_executor_fios_archive_mounts;

#ifdef __ANDROID__
void ExecutorFiosLog(const char* fmt, ...) {
    if (std::getenv("EXECUTOR_TRACE_FIOS") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_WIDE") == nullptr) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    __android_log_vprint(ANDROID_LOG_INFO, "LSX4Native", fmt, args);
    va_end(args);
}
#else
void ExecutorFiosLog(const char* fmt, ...) {
    if (std::getenv("EXECUTOR_TRACE_FIOS") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_WIDE") == nullptr) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}
#endif

bool ExecutorTraceFiosEnabled() {
    return std::getenv("EXECUTOR_TRACE_FIOS") != nullptr ||
           std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr;
}

bool ExecutorTraceFiosProgressEnabled() {
    return std::getenv("EXECUTOR_TRACE_FIOS_PROGRESS") != nullptr;
}

bool ExecutorIsInterestingFiosPath(std::string_view path) {
    return path.find(".assets") != std::string_view::npos ||
           path.find(".resG") != std::string_view::npos ||
           path.find(".psarc") != std::string_view::npos ||
           path.find("sharedassets") != std::string_view::npos ||
           path.find("level") != std::string_view::npos;
}

void ExecutorFiosProgressLog(const char* op, const ExecutorFiosFile* file, s64 offset,
                             u64 request, u64 actual, bool error) {
    if (!ExecutorTraceFiosProgressEnabled() || file == nullptr ||
        !ExecutorIsInterestingFiosPath(file->guest_path)) {
        return;
    }
    static std::atomic_uint64_t counter{0};
    const u64 index = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool short_read = request != 0 && actual != request;
    if (!error && !short_read && index > 32 && (index % 256) != 0) {
        return;
    }
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_FIOS_PROGRESS] n=%llu op=%s guest=%s offset=%lld "
                        "request=%llu actual=%llu short=%d error=%d",
                        static_cast<unsigned long long>(index), op ? op : "?",
                        file->guest_path.c_str(), static_cast<long long>(offset),
                        static_cast<unsigned long long>(request),
                        static_cast<unsigned long long>(actual), short_read ? 1 : 0,
                        error ? 1 : 0);
#else
    std::fprintf(stderr,
                 "[EXECUTOR_FIOS_PROGRESS] n=%llu op=%s guest=%s offset=%lld request=%llu "
                 "actual=%llu short=%d error=%d\n",
                 static_cast<unsigned long long>(index), op ? op : "?", file->guest_path.c_str(),
                 static_cast<long long>(offset), static_cast<unsigned long long>(request),
                 static_cast<unsigned long long>(actual), short_read ? 1 : 0, error ? 1 : 0);
#endif
}

bool ExecutorAddressMapped(u64 value, std::size_t size, bool require_write = false) {
    if (value < 0x10000 || size == 0) {
        return false;
    }
#if defined(__ANDROID__) || defined(__linux__)
    std::ifstream maps("/proc/self/maps");
    std::string line;
    const u64 end = value + size;
    while (std::getline(maps, line)) {
        unsigned long long start_addr = 0;
        unsigned long long end_addr = 0;
        char r = '-';
        char w = '-';
        if (std::sscanf(line.c_str(), "%llx-%llx %c%c", &start_addr, &end_addr, &r, &w) != 4) {
            continue;
        }
        if (value >= start_addr && end <= end_addr && r == 'r' &&
            (!require_write || w == 'w')) {
            return true;
        }
    }
    return false;
#else
    (void)require_write;
    return true;
#endif
}

bool ExecutorReadGuestCString(u64 value, std::string& out) {
    out.clear();
    if (!ExecutorAddressMapped(value, 1, false)) {
        return false;
    }
    const auto* ptr = reinterpret_cast<const char*>(value);
    for (std::size_t i = 0; i < 4096; ++i) {
        if (!ExecutorAddressMapped(value + i, 1, false)) {
            out.clear();
            return false;
        }
        const unsigned char ch = static_cast<unsigned char>(ptr[i]);
        if (ch == '\0') {
            return !out.empty();
        }
        if (ch < 0x20 || ch > 0x7e) {
            out.clear();
            return false;
        }
        out.push_back(static_cast<char>(ch));
    }
    out.clear();
    return false;
}

bool ExecutorLooksLikeFiosPath(std::string_view value) {
    return !value.empty() &&
           (value.find('/') != std::string_view::npos ||
            value.find(':') != std::string_view::npos ||
            value.find('.') != std::string_view::npos);
}

std::string ExecutorNormalizeFiosPath(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    if (path.starts_with("app0:")) {
        path = "/app0" + path.substr(5);
        if (path == "/app0") {
            path = "/app0/";
        }
    } else if (path.starts_with("app0/")) {
        path.insert(path.begin(), '/');
    } else if (path.starts_with("hostapp:")) {
        path = "/hostapp" + path.substr(8);
    } else if (path.starts_with("savedata0:")) {
        path = "/savedata0" + path.substr(10);
    } else if (!path.empty() && !path.starts_with("/")) {
        path = "/app0/" + path;
    }
    return path;
}

bool ExecutorReadFiosPath(u64 value, std::string& path) {
    if (!ExecutorReadGuestCString(value, path)) {
        return false;
    }
    path = ExecutorNormalizeFiosPath(std::move(path));
    return !path.empty();
}

std::string ExecutorManagedMediaFallbackPath(std::string_view guest_path) {
    static constexpr std::string_view kManagedPrefix = "/app0/Managed/";
    if (!guest_path.starts_with(kManagedPrefix)) {
        return {};
    }
    std::string remapped{"/app0/Media/Managed/"};
    remapped += guest_path.substr(kManagedPrefix.size());
    return remapped;
}

bool ExecutorPickFiosPath(const u64* args, std::string& path, int* arg_index = nullptr) {
    for (int i = 0; i < 6; ++i) {
        std::string candidate;
        if (ExecutorReadGuestCString(args[i], candidate) && ExecutorLooksLikeFiosPath(candidate)) {
            path = ExecutorNormalizeFiosPath(candidate);
            if (arg_index) {
                *arg_index = i;
            }
            return true;
        }
    }
    return false;
}

std::vector<std::pair<int, std::string>> ExecutorCollectFiosStrings(const u64* args) {
    std::vector<std::pair<int, std::string>> strings;
    for (int i = 0; i < 6; ++i) {
        std::string candidate;
        if (ExecutorReadGuestCString(args[i], candidate) && ExecutorLooksLikeFiosPath(candidate)) {
            strings.emplace_back(i, ExecutorNormalizeFiosPath(candidate));
        }
    }
    return strings;
}

u64* ExecutorPickWritableU64(const u64* args, const int skip_index = -1) {
    for (int i = 0; i < 6; ++i) {
        if (i == skip_index) {
            continue;
        }
        if (ExecutorAddressMapped(args[i], sizeof(u64), true)) {
            return reinterpret_cast<u64*>(args[i]);
        }
    }
    return nullptr;
}

void* ExecutorPickWritableBuffer(const u64* args, u64* size_out, const int skip_a = -1,
                                 const int skip_b = -1) {
    for (int i = 0; i < 6; ++i) {
        if (i == skip_a || i == skip_b || !ExecutorAddressMapped(args[i], 1, true)) {
            continue;
        }
        for (int j = 0; j < 6; ++j) {
            if (j == i || j == skip_a || j == skip_b) {
                continue;
            }
            const u64 size = args[j];
            if (size > 0 && size < (1ull << 31) && ExecutorAddressMapped(args[i], size, true)) {
                if (size_out) {
                    *size_out = size;
                }
                return reinterpret_cast<void*>(args[i]);
            }
        }
    }
    return nullptr;
}

std::filesystem::path ExecutorHostPathFromGuestPath(std::string_view guest_path, bool* exists_out = nullptr,
                                                    bool* read_only_out = nullptr) {
    std::string archive_remap;
    {
        std::scoped_lock lock(g_executor_fios_mutex);
        for (const auto& [mount, target] : g_executor_fios_archive_mounts) {
            if (guest_path == mount ||
                (guest_path.starts_with(mount) && guest_path.size() > mount.size() &&
                 guest_path[mount.size()] == '/')) {
                archive_remap = target;
                if (guest_path.size() > mount.size()) {
                    if (!archive_remap.ends_with('/')) {
                        archive_remap.push_back('/');
                    }
                    archive_remap += std::string(guest_path.substr(mount.size() + 1));
                }
                break;
            }
        }
    }
    if (!archive_remap.empty()) {
        return ExecutorHostPathFromGuestPath(archive_remap, exists_out, read_only_out);
    }
    bool read_only = false;
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    auto host = mnt->GetHostPath(guest_path, &read_only);
    bool exists = std::filesystem::exists(host);
    if (!exists) {
        const std::string managed_fallback = ExecutorManagedMediaFallbackPath(guest_path);
        if (!managed_fallback.empty()) {
            bool fallback_read_only = false;
            const auto fallback_host = mnt->GetHostPath(managed_fallback, &fallback_read_only);
            if (std::filesystem::exists(fallback_host)) {
                ExecutorFiosLog("[EXECUTOR_FIOS_MANAGED_FALLBACK] guest=%s remap=%s host=%s",
                                std::string(guest_path).c_str(), managed_fallback.c_str(),
                                fallback_host.string().c_str());
                host = fallback_host;
                read_only = fallback_read_only;
                exists = true;
            }
        }
    }
    if (read_only_out) {
        *read_only_out = read_only;
    }
    if (exists_out) {
        *exists_out = exists;
    }
    return host;
}

class ExecutorLibcErrnoTls {
public:
    ExecutorLibcErrnoTls& operator=(s32 value);
    s32 Value();
    u64 GuestAddress();

private:
    s32 value{};
    u64 guest_address{};
};

thread_local ExecutorLibcErrnoTls g_executor_libc_errno;

template <typename Object>
class ExecutorGuestObjectTable {
public:
    bool Insert(const u64 guest_token, std::shared_ptr<Object> object) {
        if (guest_token == 0 || !object) {
            return false;
        }
        std::scoped_lock lock(mutex);
        return objects.emplace(guest_token, std::move(object)).second;
    }

    std::shared_ptr<Object> Acquire(const u64 guest_token) const {
        std::scoped_lock lock(mutex);
        const auto it = objects.find(guest_token);
        return it != objects.end() ? it->second : nullptr;
    }

    std::shared_ptr<Object> Remove(const u64 guest_token) {
        std::scoped_lock lock(mutex);
        const auto it = objects.find(guest_token);
        if (it == objects.end()) {
            return nullptr;
        }
        auto object = std::move(it->second);
        objects.erase(it);
        return object;
    }

    std::vector<std::shared_ptr<Object>> Snapshot() const {
        std::scoped_lock lock(mutex);
        std::vector<std::shared_ptr<Object>> result;
        result.reserve(objects.size());
        for (const auto& [_, object] : objects) {
            result.emplace_back(object);
        }
        return result;
    }

private:
    mutable std::mutex mutex;
    std::unordered_map<u64, std::shared_ptr<Object>> objects;
};

struct ExecutorLibcFileObject {
    std::mutex io_mutex;
    Libraries::LibcInternal::OrbisFILE* guest_file{};
    s32 fd{-1};
    int pushback{EOF};
    std::string guest_path;
};

ExecutorGuestObjectTable<ExecutorLibcFileObject> g_executor_libc_files;
std::atomic<u64> g_executor_libc_file_serial{1};

void ExecutorLogUnknownLibcFile(const char* op, u64 file_ptr);

bool ExecutorCopyToGuestFileBuffer(const u64 guest_ptr, const void* source,
                                   const std::size_t size) {
    if (size == 0) {
        return true;
    }
    if (guest_ptr == 0 || source == nullptr ||
        guest_ptr > std::numeric_limits<u64>::max() - (size - 1)) {
        return false;
    }
#ifdef __ANDROID__
    if (ExecutorJitWriteGuestBytes != nullptr) {
        return ExecutorJitWriteGuestBytes(guest_ptr, source, size);
    }
    if (!ExecutorAddressMapped(guest_ptr, size, true)) {
        return false;
    }
#endif
    std::memcpy(reinterpret_cast<void*>(guest_ptr), source, size);
    return true;
}

bool ExecutorCopyFromGuestFileBuffer(void* destination, const u64 guest_ptr,
                                     const std::size_t size) {
    if (size == 0) {
        return true;
    }
    if (destination == nullptr || guest_ptr == 0 ||
        guest_ptr > std::numeric_limits<u64>::max() - (size - 1)) {
        return false;
    }
#ifdef __ANDROID__
    return ExecutorReadGuestBytes(guest_ptr, destination, size);
#else
    std::memcpy(destination, reinterpret_cast<const void*>(guest_ptr), size);
    return true;
#endif
}

ExecutorLibcErrnoTls& ExecutorLibcErrnoTls::operator=(const s32 new_value) {
    value = new_value;
    if (guest_address != 0) {
        ExecutorCopyToGuestFileBuffer(guest_address, &value, sizeof(value));
    }
    return *this;
}

s32 ExecutorLibcErrnoTls::Value() {
    if (guest_address != 0) {
        s32 guest_value = value;
        if (ExecutorCopyFromGuestFileBuffer(&guest_value, guest_address, sizeof(guest_value))) {
            value = guest_value;
        }
    }
    return value;
}

u64 ExecutorLibcErrnoTls::GuestAddress() {
    if (guest_address != 0) {
        return guest_address;
    }

    const u64 new_guest_address = ExecutorLibcCalloc(1, sizeof(s32));
    if (new_guest_address == 0 ||
        !ExecutorCopyToGuestFileBuffer(new_guest_address, &value, sizeof(value))) {
        if (new_guest_address != 0) {
            ExecutorLibcFree(new_guest_address);
        }
        return 0;
    }
    guest_address = new_guest_address;
    return guest_address;
}

int ExecutorGuestStandardStreamFd(const u64 file_ptr) {
    if (file_ptr == 0 || g_executor_libc_files.Acquire(file_ptr)) {
        return -1;
    }
    constexpr std::size_t HeaderSize = offsetof(Libraries::LibcInternal::OrbisFILE, _Buf);
    std::array<unsigned char, HeaderSize> header{};
    if (!ExecutorCopyFromGuestFileBuffer(header.data(), file_ptr, header.size())) {
        return -1;
    }
    Libraries::LibcInternal::OrbisFILE file{};
    std::memcpy(&file, header.data(), header.size());
    return file._Handle >= 0 && file._Handle <= 2 ? file._Handle : -1;
}

bool ExecutorParseLibcFileMode(const std::string& mode, u16& orbis_mode, s32& open_flags) {
    if (mode.empty()) {
        return false;
    }
    switch (mode.front()) {
    case 'r':
        orbis_mode = 0x80 | 0x01;
        open_flags = Libraries::Kernel::ORBIS_KERNEL_O_RDONLY;
        break;
    case 'w':
        orbis_mode = 0x80 | 0x1a;
        open_flags = Libraries::Kernel::ORBIS_KERNEL_O_WRONLY |
                     Libraries::Kernel::ORBIS_KERNEL_O_CREAT |
                     Libraries::Kernel::ORBIS_KERNEL_O_TRUNC;
        break;
    case 'a':
        orbis_mode = 0x80 | 0x16;
        open_flags = Libraries::Kernel::ORBIS_KERNEL_O_WRONLY |
                     Libraries::Kernel::ORBIS_KERNEL_O_CREAT |
                     Libraries::Kernel::ORBIS_KERNEL_O_APPEND;
        break;
    default:
        return false;
    }
    if (mode.find('+') != std::string::npos) {
        open_flags &= ~(Libraries::Kernel::ORBIS_KERNEL_O_RDONLY |
                        Libraries::Kernel::ORBIS_KERNEL_O_WRONLY);
        open_flags |= Libraries::Kernel::ORBIS_KERNEL_O_RDWR;
        orbis_mode = static_cast<u16>((orbis_mode & ~0x3u) | 0x3u);
    }
    if (mode.find('x') != std::string::npos) {
        open_flags |= Libraries::Kernel::ORBIS_KERNEL_O_EXCL;
    }
    return true;
}

void ExecutorInitializeGuestFile(Libraries::LibcInternal::OrbisFILE* file, const u16 mode,
                                 const s32 fd, const u64 serial) {
    std::memset(file, 0, sizeof(*file));
    file->_Mode = mode;
    file->_Idx = static_cast<u8>(5 + (serial % 0xfb));
    file->_Handle = fd;
    file->_Buf = &file->_Cbuf;
    file->_Bend = &file->unk2;
    file->_Next = &file->_Cbuf;
    file->_Rend = &file->_Cbuf;
    file->_Wend = &file->_Cbuf;
    file->_WWend = &file->_Cbuf;
    file->_Rback = &file->_Cbuf;
    file->_WRback = &file->unk1;
    file->_Mutex = nullptr;
}

void ExecutorRetireGuestFile(Libraries::LibcInternal::OrbisFILE* file) {
    if (!file) {
        return;
    }
    file->_Mode = 0;
    file->_Handle = -1;
    file->_Buf = &file->_Cbuf;
    file->_Next = &file->_Cbuf;
    file->_Rend = &file->_Cbuf;
    file->_WRend = &file->_Cbuf;
    file->_Wend = &file->_Cbuf;
    file->_WWend = &file->_Cbuf;
    file->_Rback = &file->_Cbuf;
    file->_WRback = &file->unk1;
}

void ExecutorLogUnknownLibcFile(const char* op, const u64 file_ptr) {
#ifdef __ANDROID__
    static std::atomic<int> log_budget{32};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIBC_FILE_UNKNOWN] op=%s guestFile=0x%llx action=no_host_FILE_passthrough",
                            op, static_cast<unsigned long long>(file_ptr));
    }
#else
    (void)op;
    (void)file_ptr;
#endif
}

extern "C" u64 ExecutorLibcError() {
    return g_executor_libc_errno.GuestAddress();
}

extern "C" u64 ExecutorLibcStrstr(u64 haystack, u64 needle) {
#ifdef __ANDROID__
    haystack = ExecutorCanonicalizeTruncatedMspaceCString("strstr", haystack);
    needle = ExecutorCanonicalizeTruncatedMspaceCString("strstr", needle);
#endif
    if (!haystack || !needle) {
        return 0;
    }
    const char* result = std::strstr(reinterpret_cast<const char*>(haystack),
                                     reinterpret_cast<const char*>(needle));
    return reinterpret_cast<u64>(result);
}

extern "C" u64 ExecutorLibcFclose(u64 file_ptr);

extern "C" u64 ExecutorLibcFopen(u64 path_ptr, u64 mode_ptr) {
    const std::string guest_path = ExecutorReadGuestString(path_ptr, 4096);
    std::string mode = ExecutorReadGuestString(mode_ptr, 32);
    if (guest_path.empty()) {
        g_executor_libc_errno = EINVAL;
        return 0;
    }
    if (mode.empty()) {
        mode = "rb";
    }

    u16 orbis_mode = 0;
    s32 open_flags = 0;
    if (!ExecutorParseLibcFileMode(mode, orbis_mode, open_flags)) {
        g_executor_libc_errno = EINVAL;
        return 0;
    }
    const s32 fd = Libraries::Kernel::posix_open(guest_path.c_str(), open_flags, 0666);
    u64 guest_file = 0;
    if (fd < 0) {
        g_executor_libc_errno = *Libraries::Kernel::__Error();
    } else {
        guest_file = ExecutorLibcCalloc(1, sizeof(Libraries::LibcInternal::OrbisFILE));
        if (guest_file == 0) {
            Libraries::Kernel::posix_close(fd);
            g_executor_libc_errno = ENOMEM;
        } else {
            const u64 serial = g_executor_libc_file_serial.fetch_add(1, std::memory_order_relaxed);
            auto* guest_orbis_file =
                reinterpret_cast<Libraries::LibcInternal::OrbisFILE*>(guest_file);
            ExecutorInitializeGuestFile(guest_orbis_file, orbis_mode, fd, serial);
            auto object = std::make_shared<ExecutorLibcFileObject>();
            object->guest_file = guest_orbis_file;
            object->fd = fd;
            object->guest_path = guest_path;
            if (!g_executor_libc_files.Insert(guest_file, std::move(object))) {
                ExecutorRetireGuestFile(guest_orbis_file);
                Libraries::Kernel::posix_close(fd);
                guest_file = 0;
                g_executor_libc_errno = EMFILE;
            }
        }
    }
#ifdef __ANDROID__
    static std::atomic<int> log_budget{64};
    if (log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_FILE] fopen guest=%s mode=%s token=0x%llx fd=%d "
                            "orbisFileSize=%zu errno=%d",
                            guest_path.c_str(), mode.c_str(),
                            static_cast<unsigned long long>(guest_file), fd,
                            sizeof(Libraries::LibcInternal::OrbisFILE),
                            g_executor_libc_errno.Value());
    }
#endif
    return guest_file;
}

extern "C" u64 ExecutorLibcFopenS(u64 out_file_ptr, u64 path_ptr, u64 mode_ptr) {
    if (out_file_ptr == 0) {
        g_executor_libc_errno = EINVAL;
        return EINVAL;
    }
    const u64 file_ptr = ExecutorLibcFopen(path_ptr, mode_ptr);
    if (!ExecutorWriteGuestQword(out_file_ptr, file_ptr)) {
        if (file_ptr != 0) {
            ExecutorLibcFclose(file_ptr);
        }
        g_executor_libc_errno = EFAULT;
        return EFAULT;
    }
    const s32 error = g_executor_libc_errno.Value();
    return file_ptr != 0 ? 0 : static_cast<u64>(error != 0 ? error : ENOENT);
}

extern "C" u64 ExecutorLibcFclose(u64 file_ptr) {
    if (!file_ptr) {
        g_executor_libc_errno = EINVAL;
        return static_cast<u64>(EOF);
    }
    auto file = g_executor_libc_files.Remove(file_ptr);
    if (!file) {
        if (ExecutorGuestStandardStreamFd(file_ptr) >= 0) {
            g_executor_libc_errno = EPERM;
            return static_cast<u64>(static_cast<s64>(EOF));
        }
        ExecutorLogUnknownLibcFile("fclose", file_ptr);
        g_executor_libc_errno = EBADF;
        return static_cast<u64>(static_cast<s64>(EOF));
    }
    std::scoped_lock io_lock(file->io_mutex);
    const s32 fd = std::exchange(file->fd, -1);
    if (fd < 0) {
        g_executor_libc_errno = EBADF;
        return static_cast<u64>(static_cast<s64>(EOF));
    }
    const int rc = Libraries::Kernel::posix_close(fd);
    ExecutorRetireGuestFile(file->guest_file);
    if (rc != 0) {
        g_executor_libc_errno = *Libraries::Kernel::__Error();
    }
    return static_cast<u64>(static_cast<s64>(rc));
}

extern "C" u64 ExecutorLibcFread(u64 buffer_ptr, u64 element_size, u64 element_count,
                                 u64 file_ptr) {
    if (element_size == 0 || element_count == 0) {
        return 0;
    }
    if (!buffer_ptr || !file_ptr) {
        g_executor_libc_errno = EINVAL;
        return 0;
    }
    if (element_count > std::numeric_limits<u64>::max() / element_size) {
        g_executor_libc_errno = EOVERFLOW;
        return 0;
    }
    const u64 total = element_size * element_count;
    if (buffer_ptr > std::numeric_limits<u64>::max() - (total - 1)) {
        g_executor_libc_errno = EFAULT;
        return 0;
    }
    auto file = g_executor_libc_files.Acquire(file_ptr);
    if (!file) {
        if (ExecutorGuestStandardStreamFd(file_ptr) == 0) {
            return 0;
        }
        ExecutorLogUnknownLibcFile("fread", file_ptr);
        g_executor_libc_errno = EBADF;
        return 0;
    }

    constexpr std::size_t ChunkSize = 64 * 1024;
    std::array<unsigned char, ChunkSize> chunk{};
    u64 delivered = 0;
    std::scoped_lock io_lock(file->io_mutex);
    if (file->fd < 0 || !file->guest_file) {
        g_executor_libc_errno = EBADF;
        return 0;
    }
    if (file->pushback != EOF && delivered < total) {
        const unsigned char byte = static_cast<unsigned char>(file->pushback);
        if (!ExecutorCopyToGuestFileBuffer(buffer_ptr, &byte, 1)) {
            g_executor_libc_errno = EFAULT;
            file->guest_file->_Mode |= 0x0200;
            return 0;
        }
        file->pushback = EOF;
        file->guest_file->_Mode &= static_cast<u16>(~0x4000u);
        ++delivered;
    }
    while (delivered < total) {
        const auto request = static_cast<std::size_t>(std::min<u64>(ChunkSize, total - delivered));
        const s64 read_result =
            Libraries::Kernel::sceKernelRead(file->fd, chunk.data(), request);
        if (read_result < 0) {
            g_executor_libc_errno = *Libraries::Kernel::__Error();
            file->guest_file->_Mode |= 0x0200;
            break;
        }
        const auto actual = static_cast<std::size_t>(read_result);
        if (actual != 0 &&
            !ExecutorCopyToGuestFileBuffer(buffer_ptr + delivered, chunk.data(), actual)) {
            Libraries::Kernel::posix_lseek(file->fd, -static_cast<s64>(actual), SEEK_CUR);
            g_executor_libc_errno = EFAULT;
            file->guest_file->_Mode |= 0x0200;
            break;
        }
        delivered += actual;
        if (actual < request) {
            if (actual == 0) {
                file->guest_file->_Mode |= 0x0100;
            }
            break;
        }
    }
#ifdef __ANDROID__
    static std::atomic<int> trace_budget{32};
    if (trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_FILE_IO] op=fread token=0x%llx dst=0x%llx "
                            "size=%llu count=%llu bytes=%llu elements=%llu errno=%d",
                            static_cast<unsigned long long>(file_ptr),
                            static_cast<unsigned long long>(buffer_ptr),
                            static_cast<unsigned long long>(element_size),
                            static_cast<unsigned long long>(element_count),
                            static_cast<unsigned long long>(delivered),
                            static_cast<unsigned long long>(delivered / element_size),
                            g_executor_libc_errno.Value());
    }
#endif
    return delivered / element_size;
}

extern "C" u64 ExecutorLibcFwrite(u64 buffer_ptr, u64 element_size, u64 element_count,
                                  u64 file_ptr) {
    if (element_size == 0 || element_count == 0) {
        return 0;
    }
    if (!buffer_ptr || !file_ptr) {
        g_executor_libc_errno = EINVAL;
        return 0;
    }
    if (element_count > std::numeric_limits<u64>::max() / element_size) {
        g_executor_libc_errno = EOVERFLOW;
        return 0;
    }
    const u64 total = element_size * element_count;
    if (buffer_ptr > std::numeric_limits<u64>::max() - (total - 1)) {
        g_executor_libc_errno = EFAULT;
        return 0;
    }
    auto file = g_executor_libc_files.Acquire(file_ptr);
    if (!file) {
        const int standard_fd = ExecutorGuestStandardStreamFd(file_ptr);
        if (standard_fd != 1 && standard_fd != 2) {
            ExecutorLogUnknownLibcFile("fwrite", file_ptr);
            g_executor_libc_errno = EBADF;
            return 0;
        }
        constexpr std::size_t ChunkSize = 64 * 1024;
        std::array<unsigned char, ChunkSize> chunk{};
        u64 written = 0;
        while (written < total) {
            const auto request =
                static_cast<std::size_t>(std::min<u64>(ChunkSize, total - written));
            if (!ExecutorCopyFromGuestFileBuffer(chunk.data(), buffer_ptr + written, request)) {
                g_executor_libc_errno = EFAULT;
                break;
            }
            const s64 result = Libraries::Kernel::sceKernelWrite(standard_fd, chunk.data(), request);
            if (result <= 0) {
                g_executor_libc_errno = *Libraries::Kernel::__Error();
                break;
            }
            written += static_cast<u64>(result);
            if (static_cast<u64>(result) < request) {
                break;
            }
        }
        return written / element_size;
    }

    constexpr std::size_t ChunkSize = 64 * 1024;
    std::array<unsigned char, ChunkSize> chunk{};
    u64 written = 0;
    std::scoped_lock io_lock(file->io_mutex);
    if (file->fd < 0 || !file->guest_file) {
        g_executor_libc_errno = EBADF;
        return 0;
    }
    while (written < total) {
        const auto request = static_cast<std::size_t>(std::min<u64>(ChunkSize, total - written));
        if (!ExecutorCopyFromGuestFileBuffer(chunk.data(), buffer_ptr + written, request)) {
            g_executor_libc_errno = EFAULT;
            break;
        }
        const s64 write_result = Libraries::Kernel::sceKernelWrite(file->fd, chunk.data(), request);
        if (write_result <= 0) {
            g_executor_libc_errno = *Libraries::Kernel::__Error();
            file->guest_file->_Mode |= 0x0200;
            break;
        }
        const auto actual = static_cast<u64>(write_result);
        written += actual;
        if (actual < request) {
            break;
        }
    }
    return written / element_size;
}

extern "C" s64 ExecutorLibcGetc(u64 file_ptr) {
    if (!file_ptr) {
        g_executor_libc_errno = EINVAL;
        return EOF;
    }
    auto file = g_executor_libc_files.Acquire(file_ptr);
    if (!file) {
        if (ExecutorGuestStandardStreamFd(file_ptr) == 0) {
            return EOF;
        }
        ExecutorLogUnknownLibcFile("getc", file_ptr);
        g_executor_libc_errno = EBADF;
        return EOF;
    }
    std::scoped_lock io_lock(file->io_mutex);
    if (file->fd < 0 || !file->guest_file) {
        g_executor_libc_errno = EBADF;
        return EOF;
    }
    if (file->pushback != EOF) {
        const int ch = std::exchange(file->pushback, EOF);
        file->guest_file->_Mode &= static_cast<u16>(~0x4000u);
        return ch;
    }
    unsigned char byte = 0;
    const s64 result = Libraries::Kernel::sceKernelRead(file->fd, &byte, 1);
    if (result < 0) {
        g_executor_libc_errno = *Libraries::Kernel::__Error();
        file->guest_file->_Mode |= 0x0200;
        return EOF;
    }
    if (result == 0) {
        file->guest_file->_Mode |= 0x0100;
        return EOF;
    }
    return byte;
}

extern "C" s64 ExecutorLibcUngetc(s32 ch, u64 file_ptr) {
    auto file = file_ptr ? g_executor_libc_files.Acquire(file_ptr) : nullptr;
    if (!file) {
        if (!file_ptr) {
            g_executor_libc_errno = EINVAL;
        } else if (ExecutorGuestStandardStreamFd(file_ptr) != 0) {
            ExecutorLogUnknownLibcFile("ungetc", file_ptr);
            g_executor_libc_errno = EBADF;
        }
        return EOF;
    }
    std::scoped_lock io_lock(file->io_mutex);
    if (file->fd < 0 || !file->guest_file) {
        g_executor_libc_errno = EBADF;
        return EOF;
    }
    if (ch == EOF || file->pushback != EOF) {
        return EOF;
    }
    file->pushback = static_cast<unsigned char>(ch);
    file->guest_file->_Mode = static_cast<u16>((file->guest_file->_Mode & ~0x0100u) | 0x4000u);
    return file->pushback;
}

extern "C" u64 ExecutorLibcFeof(u64 file_ptr) {
    auto file = file_ptr ? g_executor_libc_files.Acquire(file_ptr) : nullptr;
    if (!file) {
        if (!file_ptr) {
            g_executor_libc_errno = EINVAL;
        } else if (ExecutorGuestStandardStreamFd(file_ptr) >= 0) {
            return 0;
        } else {
            ExecutorLogUnknownLibcFile("feof", file_ptr);
            g_executor_libc_errno = EBADF;
        }
        return 0;
    }
    std::scoped_lock io_lock(file->io_mutex);
    if (file->fd < 0 || !file->guest_file) {
        g_executor_libc_errno = EBADF;
        return 0;
    }
    return (file->guest_file->_Mode & 0x0100u) != 0 ? 1 : 0;
}

extern "C" u64 ExecutorLibcFerror(u64 file_ptr) {
    if (!file_ptr) {
        return 1;
    }
    auto file = g_executor_libc_files.Acquire(file_ptr);
    if (!file) {
        if (ExecutorGuestStandardStreamFd(file_ptr) >= 0) {
            return 0;
        }
        ExecutorLogUnknownLibcFile("ferror", file_ptr);
        g_executor_libc_errno = EBADF;
        return 1;
    }
    std::scoped_lock io_lock(file->io_mutex);
    return file->fd >= 0 && file->guest_file
               ? static_cast<u64>((file->guest_file->_Mode & 0x0200u) != 0)
               : 1;
}

extern "C" u64 ExecutorLibcClearerr(u64 file_ptr) {
    if (auto file = g_executor_libc_files.Acquire(file_ptr)) {
        std::scoped_lock io_lock(file->io_mutex);
        if (file->guest_file) {
            file->guest_file->_Mode &= static_cast<u16>(~0x0300u);
        }
    } else if (file_ptr && ExecutorGuestStandardStreamFd(file_ptr) < 0) {
        ExecutorLogUnknownLibcFile("clearerr", file_ptr);
        g_executor_libc_errno = EBADF;
    }
    return 0;
}

extern "C" s64 ExecutorLibcFputs(u64 str_ptr, u64 file_ptr) {
    if (!str_ptr || !file_ptr) {
        g_executor_libc_errno = EINVAL;
        return EOF;
    }
    auto file = g_executor_libc_files.Acquire(file_ptr);
    if (!file) {
        const int standard_fd = ExecutorGuestStandardStreamFd(file_ptr);
        if (standard_fd != 1 && standard_fd != 2) {
            ExecutorLogUnknownLibcFile("fputs", file_ptr);
            g_executor_libc_errno = EBADF;
            return EOF;
        }
    }
    const std::string text = ExecutorReadGuestString(str_ptr, 1 << 20);
    return ExecutorLibcFwrite(str_ptr, 1, text.size(), file_ptr) == text.size() ? 0 : EOF;
}

extern "C" u64 ExecutorLibcFflush(u64 file_ptr) {
    if (!file_ptr) {
        for (auto& file : g_executor_libc_files.Snapshot()) {
            std::scoped_lock io_lock(file->io_mutex);
        }
        return 0;
    }
    auto file = g_executor_libc_files.Acquire(file_ptr);
    if (!file) {
        if (ExecutorGuestStandardStreamFd(file_ptr) >= 0) {
            return 0;
        }
        ExecutorLogUnknownLibcFile("fflush", file_ptr);
        g_executor_libc_errno = EBADF;
        return static_cast<u64>(static_cast<s64>(EOF));
    }
    std::scoped_lock io_lock(file->io_mutex);
    if (file->fd < 0) {
        g_executor_libc_errno = EBADF;
        return static_cast<u64>(static_cast<s64>(EOF));
    }
    return 0;
}

extern "C" s64 ExecutorLibcFseek(u64 file_ptr, s64 offset, s32 whence) {
    if (!file_ptr) {
        g_executor_libc_errno = EINVAL;
        return -1;
    }
    auto file = g_executor_libc_files.Acquire(file_ptr);
    if (!file) {
        if (ExecutorGuestStandardStreamFd(file_ptr) >= 0) {
            g_executor_libc_errno = ESPIPE;
            return -1;
        }
        ExecutorLogUnknownLibcFile("fseek", file_ptr);
        g_executor_libc_errno = EBADF;
        return -1;
    }
    std::scoped_lock io_lock(file->io_mutex);
    if (file->fd < 0 || whence < SEEK_SET || whence > SEEK_END) {
        g_executor_libc_errno = file->fd < 0 ? EBADF : EINVAL;
        return -1;
    }
    const s64 position = Libraries::Kernel::posix_lseek(file->fd, offset, whence);
    const int rc = position < 0 ? -1 : 0;
    if (rc != 0) {
        g_executor_libc_errno = *Libraries::Kernel::__Error();
        if (file->guest_file) {
            file->guest_file->_Mode |= 0x0200;
        }
    } else if (file->guest_file) {
        file->guest_file->_Mode &= static_cast<u16>(~0x0100u);
        file->pushback = EOF;
    }
    static std::atomic<u32> s_fseek_log{0};
    if (s_fseek_log.fetch_add(1) < 8) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_FILE] fseek fp=0x%llx offset=%lld whence=%d rc=%d",
                            static_cast<unsigned long long>(file_ptr),
                            static_cast<long long>(offset), static_cast<int>(whence), rc);
    }
    return static_cast<s64>(rc);
}

extern "C" s64 ExecutorLibcFtell(u64 file_ptr) {
    if (!file_ptr) {
        g_executor_libc_errno = EINVAL;
        return -1;
    }
    auto file = g_executor_libc_files.Acquire(file_ptr);
    if (!file) {
        if (ExecutorGuestStandardStreamFd(file_ptr) >= 0) {
            g_executor_libc_errno = ESPIPE;
            return -1;
        }
        ExecutorLogUnknownLibcFile("ftell", file_ptr);
        g_executor_libc_errno = EBADF;
        return -1;
    }
    std::scoped_lock io_lock(file->io_mutex);
    if (file->fd < 0) {
        g_executor_libc_errno = EBADF;
        return -1;
    }
    s64 pos = Libraries::Kernel::posix_lseek(file->fd, 0, SEEK_CUR);
    if (pos >= 0 && file->pushback != EOF) {
        --pos;
    }
    if (pos < 0) {
        g_executor_libc_errno = *Libraries::Kernel::__Error();
    }
    return pos;
}

extern "C" u64 ExecutorLibcRewind(u64 file_ptr) {
    if (!file_ptr) {
        return 0;
    }
    if (ExecutorLibcFseek(file_ptr, 0, SEEK_SET) == 0) {
        if (auto file = g_executor_libc_files.Acquire(file_ptr)) {
            std::scoped_lock io_lock(file->io_mutex);
            if (file->guest_file) {
                file->guest_file->_Mode &= static_cast<u16>(~0x0300u);
            }
        }
    }
    return 0;
}

extern "C" s64 ExecutorLibcFgetpos(u64 file_ptr, u64 pos_ptr) {
    if (!file_ptr || !pos_ptr) {
        g_executor_libc_errno = EINVAL;
        return -1;
    }
    const s64 pos = ExecutorLibcFtell(file_ptr);
    if (pos < 0) {
        return -1;
    }
    Libraries::LibcInternal::Orbisfpos_t guest_pos{};
    guest_pos._Off = pos;
    if (!ExecutorCopyToGuestFileBuffer(pos_ptr, &guest_pos, sizeof(guest_pos))) {
        g_executor_libc_errno = EFAULT;
        return -1;
    }
    return 0;
}

extern "C" s64 ExecutorLibcFsetpos(u64 file_ptr, u64 pos_ptr) {
    if (!file_ptr || !pos_ptr) {
        g_executor_libc_errno = EINVAL;
        return -1;
    }
    Libraries::LibcInternal::Orbisfpos_t guest_pos{};
    if (!ExecutorCopyFromGuestFileBuffer(&guest_pos, pos_ptr, sizeof(guest_pos))) {
        g_executor_libc_errno = EFAULT;
        return -1;
    }
    return ExecutorLibcFseek(file_ptr, guest_pos._Off, SEEK_SET);
}

ExecutorFiosFile* ExecutorFindFiosFile(u64 handle) {
    auto it = g_executor_fios_files.find(handle);
    return it == g_executor_fios_files.end() ? nullptr : &it->second;
}

ExecutorFiosOp* ExecutorFindFiosOp(u64 handle) {
    auto it = g_executor_fios_ops.find(handle);
    return it == g_executor_fios_ops.end() ? nullptr : &it->second;
}

u64 ExecutorFindFiosHandleArg(const u64* args, int* arg_index = nullptr) {
    std::scoped_lock lock(g_executor_fios_mutex);
    for (int i = 0; i < 6; ++i) {
        if (g_executor_fios_files.contains(args[i])) {
            if (arg_index) {
                *arg_index = i;
            }
            return args[i];
        }
    }
    return 0;
}

u64 ExecutorFindFiosOpArg(const u64* args, int* arg_index = nullptr) {
    std::scoped_lock lock(g_executor_fios_mutex);
    for (int i = 0; i < 6; ++i) {
        if (g_executor_fios_ops.contains(args[i])) {
            if (arg_index) {
                *arg_index = i;
            }
            return args[i];
        }
    }
    return 0;
}

u64 ExecutorCreateFiosOp(s64 error, u64 actual, std::string guest_path) {
    std::scoped_lock lock(g_executor_fios_mutex);
    const u64 handle = g_executor_fios_next_op++;
    g_executor_fios_ops.emplace(handle, ExecutorFiosOp{error, actual, true, std::move(guest_path)});
    return handle;
}

extern "C" s64 ExecutorFiosInitialize(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    ExecutorFiosLog("[EXECUTOR_FIOS] op=Initialize args=%llx,%llx,%llx,%llx,%llx,%llx result=0",
                    (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                    (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    return 0;
}

extern "C" s64 ExecutorFiosTerminate() {
    std::scoped_lock lock(g_executor_fios_mutex);
    for (auto& [_, file] : g_executor_fios_files) {
        if (file.fp) {
            std::fclose(file.fp);
        }
    }
    g_executor_fios_files.clear();
    g_executor_fios_ops.clear();
    g_executor_fios_archive_mounts.clear();
    ExecutorFiosLog("[EXECUTOR_FIOS] op=Terminate result=0");
    return 0;
}

extern "C" s64 ExecutorFiosNoop(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    ExecutorFiosLog("[EXECUTOR_FIOS] op=Noop args=%llx,%llx,%llx,%llx,%llx,%llx result=0",
                    (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                    (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    return 0;
}

extern "C" s64 ExecutorFiosArchiveGetMountBufferSizeSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4,
                                                           u64 a5) {
    std::string path;
    ExecutorReadFiosPath(a1, path);
    const s64 size = 0x10000;
    ExecutorFiosLog("[EXECUTOR_FIOS] op=ArchiveGetMountBufferSizeSync path=%s size=%lld",
                    path.empty() ? "<none>" : path.c_str(), (long long)size);
    return size;
}

extern "C" s64 ExecutorFiosArchiveMountSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    std::string path;
    const bool has_path = ExecutorReadFiosPath(a2, path);
    bool exists = false;
    const auto host = has_path ? ExecutorHostPathFromGuestPath(path, &exists) : std::filesystem::path{};
    std::string mount_point;
    if (ExecutorReadFiosPath(a3, mount_point)) {
        while (mount_point.size() > 1 && mount_point.ends_with('/')) {
            mount_point.pop_back();
        }
    }
    if (!mount_point.empty() && has_path) {
        std::string target = path;
        const auto slash = target.find_last_of('/');
        target = slash == std::string::npos ? "/app0" : target.substr(0, slash);
        std::scoped_lock lock(g_executor_fios_mutex);
        auto it = std::find_if(g_executor_fios_archive_mounts.begin(),
                               g_executor_fios_archive_mounts.end(),
                               [&](const auto& item) { return item.first == mount_point; });
        if (it == g_executor_fios_archive_mounts.end()) {
            g_executor_fios_archive_mounts.emplace_back(mount_point, target);
        } else {
            it->second = target;
        }
    }
    if (ExecutorAddressMapped(a1, sizeof(u32), true)) {
        *reinterpret_cast<u32*>(a1) = 1;
    }
    ExecutorFiosLog("[EXECUTOR_FIOS] op=ArchiveMountSync path=%s mount=%s host=%s exists=%d result=0",
                    has_path ? path.c_str() : "<none>",
                    mount_point.empty() ? "<none>" : mount_point.c_str(), host.string().c_str(),
                    exists ? 1 : 0);
    return 0;
}

extern "C" s64 ExecutorFiosFileExistsSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    std::string path;
    const bool has_path = ExecutorReadFiosPath(a1, path);
    bool exists = false;
    const auto host = has_path ? ExecutorHostPathFromGuestPath(path, &exists) : std::filesystem::path{};
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FileExistsSync path=%s host=%s exists=%d",
                    has_path ? path.c_str() : "<none>", host.string().c_str(), exists ? 1 : 0);
    return exists ? 1 : 0;
}

extern "C" s64 ExecutorFiosFileGetSizeSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    std::string path;
    const bool has_path = ExecutorReadFiosPath(a1, path);
    bool exists = false;
    const auto host = has_path ? ExecutorHostPathFromGuestPath(path, &exists) : std::filesystem::path{};
    u64 size = 0;
    if (exists && std::filesystem::is_regular_file(host)) {
        size = static_cast<u64>(std::filesystem::file_size(host));
    }
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FileGetSizeSync path=%s host=%s exists=%d size=%llu result=%lld",
                    has_path ? path.c_str() : "<none>", host.string().c_str(), exists ? 1 : 0,
                    (unsigned long long)size, (long long)(exists ? static_cast<s64>(size) : -1));
    return exists ? static_cast<s64>(size) : -1;
}

extern "C" s64 ExecutorFiosFHOpenSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    std::string path;
    const bool has_path = ExecutorReadFiosPath(a2, path);
    const bool out_mapped = ExecutorAddressMapped(a1, sizeof(u32), true);
    bool exists = false;
    const auto host = has_path ? ExecutorHostPathFromGuestPath(path, &exists) : std::filesystem::path{};
    if (!out_mapped) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHOpenSync path=%s host=%s out=0x%llx result=-1 "
                        "reason=invalid_output",
                        has_path ? path.c_str() : "<none>", host.string().c_str(),
                        (unsigned long long)a1);
        return -1;
    }
    std::FILE* fp = exists ? std::fopen(host.string().c_str(), "rb") : nullptr;
    if (!fp) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHOpenSync path=%s host=%s exists=%d result=-1",
                        has_path ? path.c_str() : "<none>", host.string().c_str(), exists ? 1 : 0);
        return -1;
    }
    u64 handle = 0;
    {
        std::scoped_lock lock(g_executor_fios_mutex);
        handle = g_executor_fios_next_handle++;
        const auto [_, inserted] =
            g_executor_fios_files.emplace(handle, ExecutorFiosFile{fp, path, host});
        if (!inserted) {
            std::fclose(fp);
            ExecutorFiosLog("[EXECUTOR_FIOS] op=FHOpenSync path=%s host=%s "
                            "handle=0x%llx result=-1 reason=handle_collision",
                            path.c_str(), host.string().c_str(),
                            (unsigned long long)handle);
            return -1;
        }
    }
    *reinterpret_cast<u32*>(a1) = static_cast<u32>(handle);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHOpenSync path=%s host=%s handle=0x%llx result=0",
                    path.c_str(), host.string().c_str(), (unsigned long long)handle);
    return 0;
}

extern "C" s64 ExecutorFiosFHCloseSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 args[] = {a0, a1, a2, a3, a4, a5};
    int handle_index = -1;
    const u64 handle = ExecutorFindFiosHandleArg(args, &handle_index);
    std::scoped_lock lock(g_executor_fios_mutex);
    auto it = g_executor_fios_files.find(handle);
    if (it == g_executor_fios_files.end()) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHCloseSync handle=0x%llx result=-1",
                        (unsigned long long)handle);
        return -1;
    }
    if (it->second.fp) {
        std::fclose(it->second.fp);
    }
    g_executor_fios_files.erase(it);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHCloseSync handle=0x%llx arg=%d result=0",
                    (unsigned long long)handle, handle_index);
    return 0;
}

extern "C" s64 ExecutorFiosFHSeek(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 handle = a0;
    const s64 offset = static_cast<s64>(a1);
    const int whence = static_cast<int>(a2);
    std::scoped_lock lock(g_executor_fios_mutex);
    auto* file = ExecutorFindFiosFile(handle);
    if (!file || !file->fp || whence < SEEK_SET || whence > SEEK_END) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHSeek handle=0x%llx offset=%lld whence=%d "
                        "result=-1",
                        (unsigned long long)handle, (long long)offset, whence);
        return -1;
    }
    const int rc = std::fseek(file->fp, static_cast<long>(offset), whence);
    const long pos = std::ftell(file->fp);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHSeek handle=0x%llx offset=%lld whence=%d pos=%ld result=%d",
                    (unsigned long long)handle, (long long)offset, whence, pos, rc == 0 ? 0 : -1);
    return rc == 0 ? 0 : -1;
}

extern "C" s64 ExecutorFiosFHGetSize(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 handle = a0;
    std::scoped_lock lock(g_executor_fios_mutex);
    auto* file = ExecutorFindFiosFile(handle);
    if (!file || !file->fp) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHGetSize handle=0x%llx result=-1",
                        (unsigned long long)handle);
        return -1;
    }

    const long saved_pos = std::ftell(file->fp);
    if (saved_pos < 0 || std::fseek(file->fp, 0, SEEK_END) != 0) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHGetSize handle=0x%llx result=-1",
                        (unsigned long long)handle);
        return -1;
    }
    const long size = std::ftell(file->fp);
    const int restore_rc = std::fseek(file->fp, saved_pos, SEEK_SET);
    const s64 result = size >= 0 && restore_rc == 0 ? static_cast<s64>(size) : -1;
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHGetSize handle=0x%llx size=%ld result=%lld",
                    (unsigned long long)handle, size, (long long)result);
    return result;
}

extern "C" s64 ExecutorFiosFHTell(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 handle = a0;
    std::scoped_lock lock(g_executor_fios_mutex);
    auto* file = ExecutorFindFiosFile(handle);
    const long pos = file && file->fp ? std::ftell(file->fp) : -1;
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHTell handle=0x%llx result=%ld",
                    (unsigned long long)handle, pos);
    return static_cast<s64>(pos);
}

extern "C" s64 ExecutorFiosFHSyncSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 handle = a1;
    std::scoped_lock lock(g_executor_fios_mutex);
    auto* file = ExecutorFindFiosFile(handle);
    if (!file || !file->fp) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHSyncSync handle=0x%llx result=-1",
                        (unsigned long long)handle);
        return -1;
    }
    const int rc = std::fflush(file->fp);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHSyncSync handle=0x%llx result=%d",
                    (unsigned long long)handle, rc == 0 ? 0 : -1);
    return rc == 0 ? 0 : -1;
}

extern "C" s64 ExecutorFiosFHReadSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 handle = a1;
    void* const buffer = reinterpret_cast<void*>(a2);
    const u64 request = a3;
    const bool buffer_mapped =
        request != 0 && ExecutorAddressMapped(a2, static_cast<std::size_t>(request), true);
    if (ExecutorTraceFiosEnabled()) {
        ExecutorFiosLog("[EXECUTOR_FIOS_RAW] op=FHReadSync args=%llx,%llx,%llx,%llx,%llx,%llx "
                        "handle=0x%llx handleArg=1 buffer=%p request=%llu",
                        (unsigned long long)a0, (unsigned long long)a1,
                        (unsigned long long)a2, (unsigned long long)a3,
                        (unsigned long long)a4, (unsigned long long)a5,
                        (unsigned long long)handle, buffer,
                        (unsigned long long)request);
    }
    std::scoped_lock lock(g_executor_fios_mutex);
    auto* file = ExecutorFindFiosFile(handle);
    if (!file || !file->fp || !buffer || !buffer_mapped) {
        ExecutorFiosProgressLog("FHReadSync", file, -1, request, 0, true);
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHReadSync handle=0x%llx buffer=%p request=%llu result=-1",
                        (unsigned long long)handle, buffer, (unsigned long long)request);
        return -1;
    }
    const long pos_before = std::ftell(file->fp);
    ExecutorPinFiosCacheAllocation(buffer, static_cast<std::size_t>(request),
                                   file->guest_path.c_str(), static_cast<s64>(pos_before));
    const auto count = std::fread(buffer, 1, static_cast<std::size_t>(request), file->fp);
    const long pos_after = std::ftell(file->fp);
    ExecutorFiosProgressLog("FHReadSync", file,
                            static_cast<s64>(pos_after) - static_cast<s64>(count), request,
                            static_cast<u64>(count), false);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHReadSync handle=0x%llx guest=%s request=%llu read=%llu result=%llu",
                    (unsigned long long)handle, file->guest_path.c_str(),
                    (unsigned long long)request, (unsigned long long)count,
                    (unsigned long long)count);
    if (count != request) {
        ExecutorFiosLog("[EXECUTOR_FIOS_TAIL] op=FHReadSync handle=0x%llx guest=%s posAfter=%ld "
                        "request=%llu read=%llu short=1",
                        (unsigned long long)handle, file->guest_path.c_str(), pos_after,
                        (unsigned long long)request, (unsigned long long)count);
    }
    return static_cast<s64>(count);
}

extern "C" s64 ExecutorFiosFHPreadSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 handle = a1;
    void* const buffer = reinterpret_cast<void*>(a2);
    const u64 request = a3;
    const s64 offset = static_cast<s64>(a4);
    const bool buffer_mapped =
        request != 0 && ExecutorAddressMapped(a2, static_cast<std::size_t>(request), true);
    if (ExecutorTraceFiosEnabled()) {
        ExecutorFiosLog("[EXECUTOR_FIOS_RAW] op=FHPreadSync args=%llx,%llx,%llx,%llx,%llx,%llx "
                        "handle=0x%llx handleArg=1 buffer=%p request=%llu offsetA4=%lld",
                        (unsigned long long)a0, (unsigned long long)a1,
                        (unsigned long long)a2, (unsigned long long)a3,
                        (unsigned long long)a4, (unsigned long long)a5,
                        (unsigned long long)handle, buffer,
                        (unsigned long long)request, (long long)offset);
    }
    std::scoped_lock lock(g_executor_fios_mutex);
    auto* file = ExecutorFindFiosFile(handle);
    if (!file || !file->fp || !buffer || !buffer_mapped || offset < 0) {
        ExecutorFiosProgressLog("FHPreadSync", file, offset, request, 0, true);
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHPreadSync handle=0x%llx buffer=%p request=%llu offset=%lld result=-1",
                        (unsigned long long)handle, buffer, (unsigned long long)request,
                        (long long)offset);
        return -1;
    }
    ExecutorPinFiosCacheAllocation(buffer, static_cast<std::size_t>(request),
                                   file->guest_path.c_str(), offset);
    const long saved = std::ftell(file->fp);
    std::fseek(file->fp, static_cast<long>(offset), SEEK_SET);
    const auto count = std::fread(buffer, 1, static_cast<std::size_t>(request), file->fp);
    std::fseek(file->fp, saved, SEEK_SET);
    ExecutorFiosProgressLog("FHPreadSync", file, offset, request, static_cast<u64>(count), false);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHPreadSync handle=0x%llx guest=%s offset=%lld request=%llu read=%llu result=%llu",
                    (unsigned long long)handle, file->guest_path.c_str(), (long long)offset,
                    (unsigned long long)request, (unsigned long long)count,
                    (unsigned long long)count);
    if (count != request) {
        ExecutorFiosLog("[EXECUTOR_FIOS_TAIL] op=FHPreadSync handle=0x%llx guest=%s offset=%lld "
                        "request=%llu read=%llu short=1",
                        (unsigned long long)handle, file->guest_path.c_str(), (long long)offset,
                        (unsigned long long)request, (unsigned long long)count);
    }
    return static_cast<s64>(count);
}

extern "C" s64 ExecutorFiosFHOpenWithModeSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4,
                                                u64 a5) {
    return ExecutorFiosFHOpenSync(a0, a1, a2, a3, a4, a5);
}

extern "C" s64 ExecutorFiosFileReadSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 args[] = {a0, a1, a2, a3, a4, a5};
    int path_index = -1;
    std::string path;
    const bool has_path = ExecutorPickFiosPath(args, path, &path_index);
    bool exists = false;
    const auto host = has_path ? ExecutorHostPathFromGuestPath(path, &exists) : std::filesystem::path{};
    u64 request = 0;
    void* buffer = ExecutorPickWritableBuffer(args, &request, path_index);
    s64 offset = 0;
    for (int i = 0; i < 6; ++i) {
        if (i == path_index || args[i] == 0 || args[i] == request ||
            ExecutorAddressMapped(args[i], 1, false)) {
            continue;
        }
        if (args[i] < (1ull << 31)) {
            offset = static_cast<s64>(args[i]);
        }
    }
    if (!exists || !buffer || request == 0 || offset < 0) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FileReadSync path=%s host=%s buffer=%p request=%llu offset=%lld result=-1",
                        has_path ? path.c_str() : "<none>", host.string().c_str(), buffer,
                        (unsigned long long)request, (long long)offset);
        return -1;
    }
    std::FILE* fp = std::fopen(host.string().c_str(), "rb");
    if (!fp) {
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FileReadSync path=%s host=%s result=-1",
                        path.c_str(), host.string().c_str());
        return -1;
    }
    ExecutorPinFiosCacheAllocation(buffer, static_cast<std::size_t>(request), path.c_str(), offset);
    std::fseek(fp, static_cast<long>(offset), SEEK_SET);
    const auto count = std::fread(buffer, 1, static_cast<std::size_t>(request), fp);
    std::fclose(fp);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FileReadSync path=%s host=%s offset=%lld request=%llu read=%llu result=%llu",
                    path.c_str(), host.string().c_str(), (long long)offset,
                    (unsigned long long)request, (unsigned long long)count,
                    (unsigned long long)count);
    return static_cast<s64>(count);
}

extern "C" s64 ExecutorFiosFileRead(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 args[] = {a0, a1, a2, a3, a4, a5};
    int path_index = -1;
    std::string path;
    const bool has_path = ExecutorPickFiosPath(args, path, &path_index);
    bool exists = false;
    const auto host = has_path ? ExecutorHostPathFromGuestPath(path, &exists) : std::filesystem::path{};
    u64 request = 0;
    void* buffer = ExecutorPickWritableBuffer(args, &request, path_index);
    s64 offset = 0;
    for (int i = 0; i < 6; ++i) {
        if (i == path_index || args[i] == 0 || args[i] == request ||
            ExecutorAddressMapped(args[i], 1, false)) {
            continue;
        }
        if (args[i] < (1ull << 31)) {
            offset = static_cast<s64>(args[i]);
        }
    }

    u64 actual = 0;
    s64 error = 0;
    if (!exists || !buffer || request == 0 || offset < 0) {
        error = -1;
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FileRead path=%s host=%s buffer=%p request=%llu offset=%lld error=-1",
                        has_path ? path.c_str() : "<none>", host.string().c_str(), buffer,
                        (unsigned long long)request, (long long)offset);
    } else if (std::FILE* fp = std::fopen(host.string().c_str(), "rb")) {
        ExecutorPinFiosCacheAllocation(buffer, static_cast<std::size_t>(request), path.c_str(),
                                       offset);
        std::fseek(fp, static_cast<long>(offset), SEEK_SET);
        actual = static_cast<u64>(std::fread(buffer, 1, static_cast<std::size_t>(request), fp));
        std::fclose(fp);
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FileRead path=%s host=%s offset=%lld request=%llu read=%llu",
                        path.c_str(), host.string().c_str(), (long long)offset,
                        (unsigned long long)request, (unsigned long long)actual);
    } else {
        error = -1;
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FileRead path=%s host=%s fopen=0 error=-1",
                        has_path ? path.c_str() : "<none>", host.string().c_str());
    }

    const u64 op_handle = ExecutorCreateFiosOp(error, actual, path);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FileReadDone handle=0x%llx actual=%llu error=%lld",
                    (unsigned long long)op_handle, (unsigned long long)actual, (long long)error);
    return static_cast<s64>(op_handle);
}

bool ExecutorFiosTryReadvEntries(std::FILE* fp, u64 iov_addr, u64 iov_count,
                                 const char* guest_path, s64 file_offset, u64* total_out) {
    if (!fp || !total_out || iov_count == 0 || iov_count > 128 ||
        !ExecutorAddressMapped(iov_addr, iov_count * 16, false)) {
        return false;
    }
    u64 total = 0;
    for (u64 i = 0; i < iov_count; ++i) {
        u64 base = 0;
        u64 size = 0;
        std::memcpy(&base, reinterpret_cast<const void*>(iov_addr + i * 16), sizeof(base));
        std::memcpy(&size, reinterpret_cast<const void*>(iov_addr + i * 16 + 8), sizeof(size));
        if (size == 0) {
            continue;
        }
        if (size > (1ull << 31) || !ExecutorAddressMapped(base, size, true)) {
            return false;
        }
        ExecutorPinFiosCacheAllocation(reinterpret_cast<void*>(base),
                                       static_cast<std::size_t>(size), guest_path,
                                       file_offset + static_cast<s64>(total));
        const auto count =
            std::fread(reinterpret_cast<void*>(base), 1, static_cast<std::size_t>(size), fp);
        total += static_cast<u64>(count);
        if (count != size) {
            break;
        }
    }
    *total_out = total;
    return true;
}

extern "C" s64 ExecutorFiosFHReadvSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 args[] = {a0, a1, a2, a3, a4, a5};
    int handle_index = -1;
    const u64 handle = ExecutorFindFiosHandleArg(args, &handle_index);
    std::scoped_lock lock(g_executor_fios_mutex);
    auto* file = ExecutorFindFiosFile(handle);
    if (!file || !file->fp) {
        ExecutorFiosProgressLog("FHReadvSync", file, -1, 0, 0, true);
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHReadvSync handle=0x%llx result=-1",
                        (unsigned long long)handle);
        return -1;
    }
    for (int i = 0; i < 6; ++i) {
        if (i == handle_index || !ExecutorAddressMapped(args[i], 16, false)) {
            continue;
        }
        for (int j = 0; j < 6; ++j) {
            if (j == i || j == handle_index || args[j] == 0 || args[j] > 128) {
                continue;
            }
            u64 total = 0;
            const long saved = std::ftell(file->fp);
            if (ExecutorFiosTryReadvEntries(file->fp, args[i], args[j],
                                            file->guest_path.c_str(),
                                            static_cast<s64>(saved), &total)) {
                const long pos_after = std::ftell(file->fp);
                ExecutorFiosProgressLog("FHReadvSync", file,
                                        static_cast<s64>(pos_after) - static_cast<s64>(total),
                                        total, total, false);
                ExecutorFiosLog("[EXECUTOR_FIOS] op=FHReadvSync handle=0x%llx guest=%s iov=%p count=%llu read=%llu result=%llu",
                                (unsigned long long)handle, file->guest_path.c_str(),
                                reinterpret_cast<void*>(args[i]), (unsigned long long)args[j],
                                (unsigned long long)total, (unsigned long long)total);
                return static_cast<s64>(total);
            }
            std::fseek(file->fp, saved, SEEK_SET);
        }
    }
    ExecutorFiosProgressLog("FHReadvSync", file, -1, 0, 0, true);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHReadvSync handle=0x%llx result=-1",
                    (unsigned long long)handle);
    return -1;
}

extern "C" s64 ExecutorFiosFHPreadvSync(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 args[] = {a0, a1, a2, a3, a4, a5};
    int handle_index = -1;
    const u64 handle = ExecutorFindFiosHandleArg(args, &handle_index);
    const s64 offset = static_cast<s64>(a4);
    std::scoped_lock lock(g_executor_fios_mutex);
    auto* file = ExecutorFindFiosFile(handle);
    if (!file || !file->fp || offset < 0) {
        ExecutorFiosProgressLog("FHPreadvSync", file, offset, 0, 0, true);
        ExecutorFiosLog("[EXECUTOR_FIOS] op=FHPreadvSync handle=0x%llx offset=%lld result=-1",
                        (unsigned long long)handle, (long long)offset);
        return -1;
    }
    for (int i = 0; i < 6; ++i) {
        if (i == handle_index || !ExecutorAddressMapped(args[i], 16, false)) {
            continue;
        }
        for (int j = 0; j < 6; ++j) {
            if (j == i || j == handle_index || args[j] == 0 || args[j] > 128) {
                continue;
            }
            u64 total = 0;
            const long saved = std::ftell(file->fp);
            std::fseek(file->fp, static_cast<long>(offset), SEEK_SET);
            if (ExecutorFiosTryReadvEntries(file->fp, args[i], args[j],
                                            file->guest_path.c_str(), offset, &total)) {
                std::fseek(file->fp, saved, SEEK_SET);
                ExecutorFiosProgressLog("FHPreadvSync", file, offset, total, total, false);
                ExecutorFiosLog("[EXECUTOR_FIOS] op=FHPreadvSync handle=0x%llx guest=%s offset=%lld iov=%p count=%llu read=%llu result=%llu",
                                (unsigned long long)handle, file->guest_path.c_str(),
                                (long long)offset, reinterpret_cast<void*>(args[i]),
                                (unsigned long long)args[j], (unsigned long long)total,
                                (unsigned long long)total);
                return static_cast<s64>(total);
            }
            std::fseek(file->fp, saved, SEEK_SET);
        }
    }
    ExecutorFiosProgressLog("FHPreadvSync", file, offset, 0, 0, true);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=FHPreadvSync handle=0x%llx offset=%lld result=-1",
                    (unsigned long long)handle, (long long)offset);
    return -1;
}

extern "C" s64 ExecutorFiosOpImmediateDone(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    ExecutorFiosLog("[EXECUTOR_FIOS] op=OpImmediateDone args=%llx,%llx,%llx,%llx,%llx,%llx result=0",
                    (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                    (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    return 0;
}

extern "C" s64 ExecutorFiosOpGetError(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 args[] = {a0, a1, a2, a3, a4, a5};
    int op_index = -1;
    const u64 op_handle = ExecutorFindFiosOpArg(args, &op_index);
    s64 error = 0;
    {
        std::scoped_lock lock(g_executor_fios_mutex);
        if (auto* op = ExecutorFindFiosOp(op_handle)) {
            error = op->error;
        }
    }
    ExecutorFiosLog("[EXECUTOR_FIOS] op=OpGetError handle=0x%llx arg=%d result=%lld",
                    (unsigned long long)op_handle, op_index, (long long)error);
    return error;
}

extern "C" s64 ExecutorFiosOpWait(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const s64 error = ExecutorFiosOpGetError(a0, a1, a2, a3, a4, a5);
    ExecutorFiosLog("[EXECUTOR_FIOS] op=OpWait result=%lld", (long long)error);
    return error;
}

extern "C" s64 ExecutorFiosOpIsDone(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    const u64 args[] = {a0, a1, a2, a3, a4, a5};
    int op_index = -1;
    const u64 op_handle = ExecutorFindFiosOpArg(args, &op_index);
    bool done = true;
    {
        std::scoped_lock lock(g_executor_fios_mutex);
        if (auto* op = ExecutorFindFiosOp(op_handle)) {
            done = op->done;
        }
    }
    ExecutorFiosLog("[EXECUTOR_FIOS] op=OpIsDone handle=0x%llx arg=%d result=%d",
                    (unsigned long long)op_handle, op_index, done ? 1 : 0);
    return done ? 1 : 0;
}

extern "C" s64 ExecutorFiosOpGetActualCount(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4,
                                             u64 a5) {
    const u64 args[] = {a0, a1, a2, a3, a4, a5};
    int op_index = -1;
    const u64 op_handle = ExecutorFindFiosOpArg(args, &op_index);
    u64 actual = 0;
    {
        std::scoped_lock lock(g_executor_fios_mutex);
        if (auto* op = ExecutorFindFiosOp(op_handle)) {
            actual = op->actual;
        }
    }
    if (auto* out = ExecutorPickWritableU64(args, op_index)) {
        *out = actual;
    }
    ExecutorFiosLog("[EXECUTOR_FIOS] op=OpGetActualCount handle=0x%llx arg=%d actual=%llu result=%llu",
                    (unsigned long long)op_handle, op_index, (unsigned long long)actual,
                    (unsigned long long)actual);
    return static_cast<s64>(actual);
}

extern "C" s64 ExecutorFiosWriteOrTruncateNoop(u64 a0, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    ExecutorFiosLog("[EXECUTOR_FIOS] op=WriteOrTruncateNoop args=%llx,%llx,%llx,%llx,%llx,%llx result=0",
                    (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                    (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    return 0;
}

}

struct ExecutorStubOverride {
    const char* nid;
    const char* name;
    u64 address;
};

bool ExecutorUseRealLibcAlloc();
bool IsLibcAllocFamilyNid(const char* nid);

const ExecutorStubOverride* FindExecutorStubOverride(const char* nid) {
#ifdef __ANDROID__
    if (ExecutorUseRealLibcAlloc() && IsLibcAllocFamilyNid(nid)) {
        return nullptr;
    }
    if (std::getenv("EXECUTOR_LIVE_SWAP_LIBC_COMPARE_NIDS")) {
        static const ExecutorStubOverride swapped_compare_overrides[] = {
            {"Ovb2dSJOAuE", "strncmp", reinterpret_cast<u64>(&ExecutorLibcStrncmp)},
            {"aesyjrHVWy4", "strcmp", reinterpret_cast<u64>(&ExecutorLibcStrcmp)},
        };
        for (const auto& override : swapped_compare_overrides) {
            if (std::strcmp(override.nid, nid) == 0) {
                return &override;
            }
        }
    }
#endif
    static const ExecutorStubOverride overrides[] = {
        {"3GPpjQdAMTw", "__cxa_guard_acquire", reinterpret_cast<u64>(&ExecutorCxaGuardAcquire)},
        {"9rAeANT2tyE", "__cxa_guard_release", reinterpret_cast<u64>(&ExecutorCxaGuardRelease)},
        {"2emaaluWzUw", "__cxa_guard_abort", reinterpret_cast<u64>(&ExecutorCxaGuardAbort)},
        {"sqWytnhYdEg", "std::ios_base::Init::Init()",
         reinterpret_cast<u64>(&ExecutorCppStaticInitNoop)},
        {"bTQcNwRc8hE", "std::ios_base::Init::Init()",
         reinterpret_cast<u64>(&ExecutorCppStaticInitNoop)},
        {"-Bl9-SZ2noc", "std::_Winit::_Winit()",
         reinterpret_cast<u64>(&ExecutorCppStaticInitNoop)},
        {"57mMrw0l-40", "std::_Winit::_Winit()",
         reinterpret_cast<u64>(&ExecutorCppStaticInitNoop)},
        {"kxXCvcat1cM", "std::ios_base::Init::~Init()",
         reinterpret_cast<u64>(&ExecutorCppStaticDtorNoop)},
        {"bxLH5WHgMBY", "std::ios_base::Init::~Init()",
         reinterpret_cast<u64>(&ExecutorCppStaticDtorNoop)},
        {"Uw3OTZFPNt4", "std::_Winit::~_Winit()",
         reinterpret_cast<u64>(&ExecutorCppStaticDtorNoop)},
        {"2yOarodWACE", "std::_Winit::~_Winit()",
         reinterpret_cast<u64>(&ExecutorCppStaticDtorNoop)},
        {"kALvdgEv5ME", "_Locksyslock", reinterpret_cast<u64>(&ExecutorLibcLocksyslock)},
        {"9nf8joUTSaQ", "_Unlocksyslock", reinterpret_cast<u64>(&ExecutorLibcUnlocksyslock)},
        {"tsvEmnenz48", "__cxa_atexit", reinterpret_cast<u64>(&ExecutorCxaAtexit)},
        {"wadT3QBCGY0", "sceKeyboardInit", reinterpret_cast<u64>(&ExecutorSceKeyboardInit)},
        {"HJ+KnEHcaxI", "sceKeyboardOpen", reinterpret_cast<u64>(&ExecutorSceKeyboardOpen)},
        {"6HpE68bzX6M", "sceKeyboardReadState", reinterpret_cast<u64>(&ExecutorSceKeyboardReadState)},
        {"yO9JwdRhtSA", "sceKeyboardGetKey2Char",
         reinterpret_cast<u64>(&ExecutorSceKeyboardGetKey2Char)},
        {"wAKZ-det+yo", "sceFiosInitialize", reinterpret_cast<u64>(&ExecutorFiosInitialize)},
        {"3HAgZPl1v+4", "sceFiosTerminate", reinterpret_cast<u64>(&ExecutorFiosTerminate)},
        {"lgITuBsRo2o", "sceFiosIOFilterAdd", reinterpret_cast<u64>(&ExecutorFiosNoop)},
        {"UUriaXy7G90", "sceFiosArchiveGetMountBufferSizeSync",
         reinterpret_cast<u64>(&ExecutorFiosArchiveGetMountBufferSizeSync)},
        {"xutLbQdqyb4", "sceFiosArchiveMountSync",
         reinterpret_cast<u64>(&ExecutorFiosArchiveMountSync)},
        {"zF8-CRvRXnM", "sceFiosFileGetSizeSync",
         reinterpret_cast<u64>(&ExecutorFiosFileGetSizeSync)},
        {"b44anV2D7K0", "sceFiosFHOpenSync", reinterpret_cast<u64>(&ExecutorFiosFHOpenSync)},
        {"w13Ojm7ON9o", "sceFiosFHOpenWithModeSync",
         reinterpret_cast<u64>(&ExecutorFiosFHOpenWithModeSync)},
        {"xReSebwKApA", "sceFiosFHSeek", reinterpret_cast<u64>(&ExecutorFiosFHSeek)},
        {"FdjoqFQOlt0", "sceFiosFHGetSize", reinterpret_cast<u64>(&ExecutorFiosFHGetSize)},
        {"MrRFrdgpsx8", "sceFiosFHTell", reinterpret_cast<u64>(&ExecutorFiosFHTell)},
        {"EzzSJz6yuMc", "sceFiosFHSyncSync", reinterpret_cast<u64>(&ExecutorFiosFHSyncSync)},
        {"AOujSGqU+ms", "sceFiosFHCloseSync", reinterpret_cast<u64>(&ExecutorFiosFHCloseSync)},
        {"Bn2ZF4ZjeuQ", "sceFiosFHReadSync", reinterpret_cast<u64>(&ExecutorFiosFHReadSync)},
        {"ltWdd+agvD0", "sceFiosFHReadvSync", reinterpret_cast<u64>(&ExecutorFiosFHReadvSync)},
        {"2m9+Opco-hk", "sceFiosFHPreadSync", reinterpret_cast<u64>(&ExecutorFiosFHPreadSync)},
        {"OHl4kz+OCws", "sceFiosFHPreadvSync",
         reinterpret_cast<u64>(&ExecutorFiosFHPreadvSync)},
        {"kVMxSiYD6tc", "sceFiosFileReadSync",
         reinterpret_cast<u64>(&ExecutorFiosFileReadSync)},
        {"YlKCkfJL+Y8", "sceFiosFileRead", reinterpret_cast<u64>(&ExecutorFiosFileRead)},
        {"SnoQQWnGK9I", "sceFiosOpWait", reinterpret_cast<u64>(&ExecutorFiosOpWait)},
        {"2wvqS7Odb6M", "sceFiosOpSyncWait",
         reinterpret_cast<u64>(&ExecutorFiosOpWait)},
        {"X+7rIfY97Ps", "sceFiosOpGetError",
         reinterpret_cast<u64>(&ExecutorFiosOpGetError)},
        {"bfgo2Otmqz0", "sceFiosOpIsDone", reinterpret_cast<u64>(&ExecutorFiosOpIsDone)},
        {"+FRvKknUj1I", "sceFiosOpGetActualCount",
         reinterpret_cast<u64>(&ExecutorFiosOpGetActualCount)},
        {"Kl-TbrDU9YM", "sceFiosFHWriteSync", reinterpret_cast<u64>(&ExecutorFiosWriteOrTruncateNoop)},
        {"gMcfOtHW6zk", "sceFiosFHPwriteSync",
         reinterpret_cast<u64>(&ExecutorFiosWriteOrTruncateNoop)},
        {"gRA2pp3a1-k", "sceFiosFileTruncateSync",
         reinterpret_cast<u64>(&ExecutorFiosWriteOrTruncateNoop)},
        {"NwOHMRM2Ppw", "sceFiosFileExistsSync",
         reinterpret_cast<u64>(&ExecutorFiosFileExistsSync)},
        {"-hn1tcVHq5Q", "sceLibcMspaceCreate", reinterpret_cast<u64>(&ExecutorMspaceCreate)},
        {"pi90NsG3zPA", "sceLibcMspaceCreateForMonoMutex",
         reinterpret_cast<u64>(&ExecutorMspaceCreate)},
        {"W6SiVSiCDtI", "sceLibcMspaceDestroy", reinterpret_cast<u64>(&ExecutorMspaceDestroy)},
        {"OJjm-QOIHlI", "sceLibcMspaceMalloc", reinterpret_cast<u64>(&ExecutorMspaceMalloc)},
        {"LYo3GhIlB38", "sceLibcMspaceCalloc", reinterpret_cast<u64>(&ExecutorMspaceCalloc)},
        {"gigoVHZvVPE", "sceLibcMspaceRealloc", reinterpret_cast<u64>(&ExecutorMspaceRealloc)},
        {"xLXHyF8De0c", "_sceLibcMspaceRealloc", reinterpret_cast<u64>(&ExecutorMspaceRealloc)},
        {"Vla-Z+eXlxo", "sceLibcMspaceFree", reinterpret_cast<u64>(&ExecutorMspaceFree)},
        {"iF1iQHzxBJU", "sceLibcMspaceMemalign", reinterpret_cast<u64>(&ExecutorMspaceMemalign)},
        {"ljkqMcC4-mk", "sceLibcMspaceAlignedAlloc",
         reinterpret_cast<u64>(&ExecutorMspaceAlignedAlloc)},
        {"qWESlyXMI3E", "sceLibcMspacePosixMemalign",
         reinterpret_cast<u64>(&ExecutorMspacePosixMemalign)},
        {"fEoW6BJsPt4", "sceLibcMspaceMallocUsableSize",
         reinterpret_cast<u64>(&ExecutorMspaceMallocUsableSize)},
        {"k04jLXu3+Ic", "sceLibcMspaceMallocStatsFast",
         reinterpret_cast<u64>(&ExecutorMspaceNoopStats)},
        {"mfHdJTIvhuo", "sceLibcMspaceMallocStats",
         reinterpret_cast<u64>(&ExecutorMspaceNoopStats)},
        {"pzUa7KEoydw", "sceLibcMspaceIsHeapEmpty",
         reinterpret_cast<u64>(&ExecutorMspaceIsHeapEmpty)},
        {"gvqHvbjlHzA", "sceLibcMspaceGetFooterValue",
         reinterpret_cast<u64>(&ExecutorMspaceFooterValue)},
        {"ssIzVl7hH8g", "mspace_malloc", reinterpret_cast<u64>(&ExecutorMspaceMalloc)},
        {"ctoQMlFaZ-M", "mspace_calloc", reinterpret_cast<u64>(&ExecutorMspaceCalloc)},
        {"yP1IuW1xeIk", "mspace_realloc", reinterpret_cast<u64>(&ExecutorMspaceRealloc)},
        {"o2waU9UJw9E", "mspace_free", reinterpret_cast<u64>(&ExecutorMspaceFree)},
        {"MEE4xlWGmaY", "mspace_memalign", reinterpret_cast<u64>(&ExecutorMspaceMemalign)},
        {"G1C+IzPmhc0", "mspace_lock", reinterpret_cast<u64>(&ExecutorMspaceLock)},
        {"+ewEHVkVfcU", "mspace_unlock", reinterpret_cast<u64>(&ExecutorMspaceUnlock)},
        {"1HBNVdBWQVk", "_sceLibcMspaceLock", reinterpret_cast<u64>(&ExecutorMspaceLock)},
        {"NewD1IkVMeU", "_sceLibcMspaceUnlock", reinterpret_cast<u64>(&ExecutorMspaceUnlock)},
        {"HHKtLpzPl4A", "mspace_reallocalign", reinterpret_cast<u64>(&ExecutorMspaceReallocalign)},
        {"KfKBG2h2KdM", "_sceLibcMspaceReallocalign",
         reinterpret_cast<u64>(&ExecutorMspaceReallocalign)},
        {"p6lrRW8-MLY", "sceLibcMspaceReallocalign",
         reinterpret_cast<u64>(&ExecutorMspaceReallocalign)},
        {"ZgbOkbpXeqI", "mspace_posix_memalign",
         reinterpret_cast<u64>(&ExecutorMspacePosixMemalign)},
        {"EJZBym9sZQ0", "mspace_malloc_usable_size",
         reinterpret_cast<u64>(&ExecutorMspaceMallocUsableSize)},
        {"LDfSNfOIwFI", "mspace_malloc_stats", reinterpret_cast<u64>(&ExecutorMspaceNoopStats)},
        {"Tp3xgPS3Uwc", "mspace_trim", reinterpret_cast<u64>(&ExecutorMspaceNoopTrim)},
        {"QuZzFJD5Hrw", "sceLibcPafMspaceMalloc",
         reinterpret_cast<u64>(&ExecutorMspaceMalloc)},
        {"-lZdT34nAAE", "sceLibcPafMspaceCalloc",
         reinterpret_cast<u64>(&ExecutorMspaceCalloc)},
        {"u32UXVridxQ", "sceLibcPafMspaceRealloc",
         reinterpret_cast<u64>(&ExecutorMspaceRealloc)},
        {"9mMuuhXMwqQ", "sceLibcPafMspaceFree", reinterpret_cast<u64>(&ExecutorMspaceFree)},
        {"PKJcFUfhKtw", "sceLibcPafMspaceMemalign",
         reinterpret_cast<u64>(&ExecutorMspaceMemalign)},
        {"7hOUKGcT6jM", "sceLibcPafMspacePosixMemalign",
         reinterpret_cast<u64>(&ExecutorMspacePosixMemalign)},
        {"6JcY5RDA4jY", "sceLibcPafMspaceMallocUsableSize",
         reinterpret_cast<u64>(&ExecutorMspaceMallocUsableSize)},
        {"OmG3YPCBLJs", "sceLibcPafMspaceMallocStatsFast",
         reinterpret_cast<u64>(&ExecutorMspaceNoopStats)},
        {"mO8NB8whKy8", "sceLibcPafMspaceMallocStats",
         reinterpret_cast<u64>(&ExecutorMspaceNoopStats)},
        {"htdTOnMxDbQ", "sceLibcPafMspaceIsHeapEmpty",
         reinterpret_cast<u64>(&ExecutorMspaceIsHeapEmpty)},
        {"kv4kgdjswN0", "sceLibcPafMspaceGetFooterValue",
         reinterpret_cast<u64>(&ExecutorMspaceFooterValue)},
        {"4h3fLdA8LHw", "coil_mspace_malloc",
         reinterpret_cast<u64>(&ExecutorMspaceMalloc)},
        {"6AZDjebK-v8", "coil_mspace_calloc",
         reinterpret_cast<u64>(&ExecutorMspaceCalloc)},
        {"dQhl7bw-umE", "coil_mspace_free", reinterpret_cast<u64>(&ExecutorMspaceFree)},
        {"gQX+4GDQjpM", "malloc", reinterpret_cast<u64>(&ExecutorLibcMalloc)},
        {"4GN2t+VJdx8", "_malloc", reinterpret_cast<u64>(&ExecutorLibcMalloc)},
        {"4-GxiotYTWo", "_sceLibcMalloc", reinterpret_cast<u64>(&ExecutorLibcMalloc)},
        {"fJnpuVVBbKk", "_Znwm", reinterpret_cast<u64>(&ExecutorLibcMalloc)},
        {"hdm0YfMa7TQ", "_Znam", reinterpret_cast<u64>(&ExecutorLibcMalloc)},
        {"ryUxD-60bKM", "_ZnwmRKSt9nothrow_t", reinterpret_cast<u64>(&ExecutorLibcMalloc)},
        {"Jh5qUcwiSEk", "_ZnamRKSt9nothrow_t", reinterpret_cast<u64>(&ExecutorLibcMalloc)},
        {"z+P+xCnWLBk", "_ZdlPv", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"MLWl90SFWNE", "_ZdaPv", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"lYDzBVE5mZs", "_ZdlPvm", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"FOt55ZNaVJk", "_ZdaPvm", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"McsGnqV6yRE", "_ZdlPvRKSt9nothrow_t", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"m-fSo3EbxNA", "_ZdaPvRKSt9nothrow_t", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"bZx+FFSlkUM", "_ZdlPvSt11align_val_t", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"v09ZcAhZzSc", "_ZdaPvSt11align_val_t", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"nwujzxOPXzQ", "_ZdlPvmSt11align_val_t", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"Y1RR+IQy6Pg", "_ZdaPvmSt11align_val_t", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"Dt9kllUFXS0", "_ZdlPvSt11align_val_tRKSt9nothrow_t",
         reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"dH3ucvQhfSY", "_ZdaPvSt11align_val_tRKSt9nothrow_t",
         reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"2X5agFjKxMc", "calloc", reinterpret_cast<u64>(&ExecutorLibcCalloc)},
        {"mFBxBLtfT6Q", "_calloc", reinterpret_cast<u64>(&ExecutorLibcCalloc)},
        {"hIkkDt5bctc", "_sceLibcCalloc", reinterpret_cast<u64>(&ExecutorLibcCalloc)},
        {"Y7aJ1uydPMo", "realloc", reinterpret_cast<u64>(&ExecutorLibcRealloc)},
        {"FPxZe+7sKPI", "_realloc", reinterpret_cast<u64>(&ExecutorLibcRealloc)},
        {"SapHB+u0OPE", "_sceLibcRealloc", reinterpret_cast<u64>(&ExecutorLibcRealloc)},
        {"OGybVuPAhAY", "reallocalign", reinterpret_cast<u64>(&ExecutorLibcReallocalign)},
        {"yFs8CcUN6CU", "internal_reallocalign",
         reinterpret_cast<u64>(&ExecutorLibcReallocalign)},
        {"tIhsqj0qsFE", "free", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"fjbBk89tXmQ", "_free", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"OONgpXDsjFI", "_sceLibcFree", reinterpret_cast<u64>(&ExecutorLibcFree)},
        {"Q3VBxCXhUHs", "memcpy", reinterpret_cast<u64>(&ExecutorLibcMemcpy)},
        {"+P6FRGH4LfA", "memmove", reinterpret_cast<u64>(&ExecutorLibcMemmove)},
        {"8zTFvBIAIN8", "memset", reinterpret_cast<u64>(&ExecutorLibcMemset)},
        {"Ujf3KzMvRmI", "memalign", reinterpret_cast<u64>(&ExecutorLibcMemalign)},
        {"dQumQIBx1Iw", "_memalign", reinterpret_cast<u64>(&ExecutorLibcMemalign)},
        {"cVSk9y8URbc", "posix_memalign", reinterpret_cast<u64>(&ExecutorLibcPosixMemalign)},
        {"nNkYCQIDb+4", "_posix_memalign", reinterpret_cast<u64>(&ExecutorLibcPosixMemalign)},
        {"2Btkg8k24Zg", "aligned_alloc", reinterpret_cast<u64>(&ExecutorLibcMemalign)},
        {"FoARlup+lmg", "_malloc_usable_size",
         reinterpret_cast<u64>(&ExecutorLibcMallocUsableSize)},
        {"NDcSfcYZRC8", "malloc_usable_size",
         reinterpret_cast<u64>(&ExecutorLibcMallocUsableSize)},
        {"qjBlw2cVMAM", "vasprintf", reinterpret_cast<u64>(&ExecutorLibcVasprintf)},
        {"Q2V+iqvjgC0", "vsnprintf", reinterpret_cast<u64>(&ExecutorLibcVsnprintf)},
        {"jbz9I9vkqkk", "vsprintf", reinterpret_cast<u64>(&ExecutorLibcVsprintf)},
        {"eLdDw6l0-bU", "snprintf", reinterpret_cast<u64>(&ExecutorLibcSnprintf)},
        {"XPrn0-EWJu4", "g_snprintf", reinterpret_cast<u64>(&ExecutorLibcSnprintf)},
        {"xYY8intnuPs", "monoeg_g_snprintf", reinterpret_cast<u64>(&ExecutorLibcSnprintf)},
        {"tcVi5SivF7Q", "sprintf", reinterpret_cast<u64>(&ExecutorLibcSprintf)},
        {"fffwELXNVFA", "fprintf", reinterpret_cast<u64>(&ExecutorLibcFprintf)},
        {"pDBDcY6uLSA", "vfprintf", reinterpret_cast<u64>(&ExecutorLibcVfprintf)},
        {"xeYO4u7uyJ0", "fopen", reinterpret_cast<u64>(&ExecutorLibcFopen)},
        {"uodLYyUip20", "fclose", reinterpret_cast<u64>(&ExecutorLibcFclose)},
        {"lbB+UlZqVG0", "fread", reinterpret_cast<u64>(&ExecutorLibcFread)},
        {"MpxhMh8QFro", "fwrite", reinterpret_cast<u64>(&ExecutorLibcFwrite)},
        {"QrZZdJ8XsX0", "fputs", reinterpret_cast<u64>(&ExecutorLibcFputs)},
        {"MUjC4lbHrK4", "fflush", reinterpret_cast<u64>(&ExecutorLibcFflush)},
        {"E1iwBYkG3CM", "__fflush", reinterpret_cast<u64>(&ExecutorLibcFflush)},
        {"rQFVBXp-Cxg", "fseek", reinterpret_cast<u64>(&ExecutorLibcFseek)},
        {"pkYiKw09PRA", "fseek", reinterpret_cast<u64>(&ExecutorLibcFseek)},
        {"Qazy8LmXTvw", "ftell", reinterpret_cast<u64>(&ExecutorLibcFtell)},
        {"5qP1iVQkdck", "ftell", reinterpret_cast<u64>(&ExecutorLibcFtell)},
        {"3QIPIh-GDjw", "rewind", reinterpret_cast<u64>(&ExecutorLibcRewind)},
        {"7PkSz+qnTto", "fsetpos", reinterpret_cast<u64>(&ExecutorLibcFsetpos)},
        {"SHlt7EhOtqA", "fgetpos", reinterpret_cast<u64>(&ExecutorLibcFgetpos)},
        {"8Q60JLJ6Rv4", "getc", reinterpret_cast<u64>(&ExecutorLibcGetc)},
        {"AEuF3F2f8TA", "fgetc", reinterpret_cast<u64>(&ExecutorLibcGetc)},
        {"-LFO7jhD5CE", "ungetc", reinterpret_cast<u64>(&ExecutorLibcUngetc)},
        {"LxcEU+ICu8U", "feof", reinterpret_cast<u64>(&ExecutorLibcFeof)},
        {"NuydofHcR1w", "feof", reinterpret_cast<u64>(&ExecutorLibcFeof)},
        {"AHxyhN96dy4", "ferror", reinterpret_cast<u64>(&ExecutorLibcFerror)},
        {"yxbGzBQC5xA", "ferror_unlocked", reinterpret_cast<u64>(&ExecutorLibcFerror)},
        {"St9nbxSoezk", "clearerr", reinterpret_cast<u64>(&ExecutorLibcClearerr)},
        {"NL836gOLANs", "fopen_s", reinterpret_cast<u64>(&ExecutorLibcFopenS)},
        {"9BcDykPmo1I", "__error", reinterpret_cast<u64>(&ExecutorLibcError)},
        {"j4ViWNHEgww", "strlen", reinterpret_cast<u64>(&ExecutorLibcStrlen)},
        {"g7zzzLDYGw0", "strdup", reinterpret_cast<u64>(&ExecutorLibcStrdup)},
        {"QxmSHBCuKTk", "strtoul", reinterpret_cast<u64>(&ExecutorLibcStrtoull)},
        {"5OqszGpy7Mg", "strtoull", reinterpret_cast<u64>(&ExecutorLibcStrtoull)},
        {"mXlxhmLNMPg", "strtol", reinterpret_cast<u64>(&ExecutorLibcStrtoll)},
        {"VOBg+iNwB-4", "strtoll", reinterpret_cast<u64>(&ExecutorLibcStrtoll)},
        {"zlfEH8FmyUA", "_Stoul", reinterpret_cast<u64>(&ExecutorLibcStoul)},
        {"YDnLaav6W6Q", "_Stoulx", reinterpret_cast<u64>(&ExecutorLibcStoul)},
        {"q+9E0X3aWpU", "_Stoull", reinterpret_cast<u64>(&ExecutorLibcStoull)},
        {"pSpDCDyxkaY", "_Stoullx", reinterpret_cast<u64>(&ExecutorLibcStoull)},
        {"Ecwid6wJMhY", "_Stolx", reinterpret_cast<u64>(&ExecutorLibcStol)},
        {"7pNKcscKrf8", "_Stoll", reinterpret_cast<u64>(&ExecutorLibcStoll)},
        {"mOnfZ5aNDQE", "_Stollx", reinterpret_cast<u64>(&ExecutorLibcStoll)},
        {"kiZSXIWd9vg", "strcpy", reinterpret_cast<u64>(&ExecutorLibcStrcpy)},
        {"6sJWiWSRuqk", "strncpy", reinterpret_cast<u64>(&ExecutorLibcStrncpy)},
        {"SfQIZcqvvms", "strlcpy", reinterpret_cast<u64>(&ExecutorLibcStrlcpy)},
        {"kMXQ-OVHLrY", "g_strlcpy", reinterpret_cast<u64>(&ExecutorLibcStrlcpy)},
        {"s0v1hj9ip3A", "monoeg_g_strlcpy", reinterpret_cast<u64>(&ExecutorLibcStrlcpy)},
        {"Ls4tzzhimqQ", "strcat", reinterpret_cast<u64>(&ExecutorLibcStrcat)},
        {"ByfjUZsWiyg", "strlcat", reinterpret_cast<u64>(&ExecutorLibcStrlcat)},
        {"kHg45qPC6f0", "strncat", reinterpret_cast<u64>(&ExecutorLibcStrncat)},
        {"DfivPArhucg", "memcmp", reinterpret_cast<u64>(&ExecutorLibcMemcmp)},
        {"NesIgTmfF0Q", "bsearch", reinterpret_cast<u64>(&ExecutorLibcBsearch)},
        {"Ovb2dSJOAuE", "strcmp", reinterpret_cast<u64>(&ExecutorLibcStrcmp)},
        {"aesyjrHVWy4", "strncmp", reinterpret_cast<u64>(&ExecutorLibcStrncmp)},
        {"oVkZ8W8-Q8A", "strtok", reinterpret_cast<u64>(&ExecutorLibcStrtok)},
        {"enqPGLfmVNU", "strtok_r", reinterpret_cast<u64>(&ExecutorLibcStrtokR)},
        {"-vXEQdRADLI", "strtok_s", reinterpret_cast<u64>(&ExecutorLibcStrtokS)},
        {"viiwFMaNamA", "strstr", reinterpret_cast<u64>(&ExecutorLibcStrstr)},
        {"smbQukfxYJM", "getenv", reinterpret_cast<u64>(&ExecutorLibcGetenv)},
        {"5FdejCVZfZs", "__wrap_getenv", reinterpret_cast<u64>(&ExecutorLibcGetenv)},
        {"iHFQ4ROnqJ8", "g_getenv", reinterpret_cast<u64>(&ExecutorLibcGetenv)},
        {"n-IyBYJNbQ0", "monoeg_g_getenv", reinterpret_cast<u64>(&ExecutorLibcGetenv)},
        {"M4YYbSFfJ8g", "setenv", reinterpret_cast<u64>(&ExecutorLibcSetenv)},
        {"GM3gnBK6qP0", "g_setenv", reinterpret_cast<u64>(&ExecutorLibcSetenv)},
        {"4uYhaXaveRc", "monoeg_g_setenv", reinterpret_cast<u64>(&ExecutorLibcSetenv)},
        {"CRJcH8CnPSI", "unsetenv", reinterpret_cast<u64>(&ExecutorLibcUnsetenv)},
        {"oJ+sgVai4po", "g_unsetenv", reinterpret_cast<u64>(&ExecutorLibcUnsetenv)},
        {"4+Y0-XRlarQ", "monoeg_g_unsetenv", reinterpret_cast<u64>(&ExecutorLibcUnsetenv)},
        {"iJbG4E+cTL4", "png_malloc", reinterpret_cast<u64>(&ExecutorPngMalloc)},
        {"e4j-rkRbvNw", "png_calloc", reinterpret_cast<u64>(&ExecutorPngCalloc)},
        {"Ofz1AcKLESA", "png_free", reinterpret_cast<u64>(&ExecutorPngFree)},
        {"gC2k54oPeFY", "png_malloc_default", reinterpret_cast<u64>(&ExecutorPngMalloc)},
        {"+TF60FmY-hk", "png_free_default", reinterpret_cast<u64>(&ExecutorPngFree)},
        {"CC-BLMBu9-I", "malloc_stats", reinterpret_cast<u64>(&ExecutorMallocNoopStats)},
        {"K90PVQAEb6g", "_malloc_stats", reinterpret_cast<u64>(&ExecutorMallocNoopStats)},
        {"KuOuD58hqn4", "malloc_stats_fast", reinterpret_cast<u64>(&ExecutorMallocNoopStats)},
        {"JtiEJNQyI2o", "_malloc_stats_fast", reinterpret_cast<u64>(&ExecutorMallocNoopStats)},
        {"L1SBTkC+Cvw", "abort", reinterpret_cast<u64>(&ExecutorAbort)},
        {"W0xkN0+ZkCE", "kill", reinterpret_cast<u64>(&ExecutorKill)},
        {"0t0-MxQNwK4", "raise", reinterpret_cast<u64>(&ExecutorRaise)},
        {"2wDzNZ9oKro", "coil_raise", reinterpret_cast<u64>(&ExecutorRaise)},
        {"JZKw5+Wrnaw", "pthread_sigmask", reinterpret_cast<u64>(&ExecutorPthreadSigmask)},
        {"ay3uROQAc5A", "opendir", reinterpret_cast<u64>(&ExecutorOpendir)},
        {"KOy7MeQ7OAU", "__opendir2", reinterpret_cast<u64>(&ExecutorOpendir2)},
        {"lybyyKtP54c", "readdir", reinterpret_cast<u64>(&ExecutorReaddir)},
        {"J0kng1yac3M", "readdir_r", reinterpret_cast<u64>(&ExecutorReaddirR)},
        {"XepdqehVYe4", "closedir", reinterpret_cast<u64>(&ExecutorClosedir)},
        {"540lotO7oHE", "sceAppInstUtilInitialize",
         reinterpret_cast<u64>(&ExecutorAppInstUtilInitialize)},
        {"kLLazhNh6d4", "sceAppInstUtilTerminate",
         reinterpret_cast<u64>(&ExecutorAppInstUtilTerminate)},
        {"fAG-kjDm-io", "sceAppInstUtilGetTitleIdFromPkg",
         reinterpret_cast<u64>(&ExecutorAppInstUtilGetTitleIdFromPkg)},
        {"kUT4RpxclMQ", "sceAppInstUtilAppExists",
         reinterpret_cast<u64>(&ExecutorAppInstUtilAppExists)},
        {"Sx4TTyrQccE", "sceAppInstUtilAppUnInstall",
         reinterpret_cast<u64>(&ExecutorAppInstUtilAppUnInstall)},
        {"joyu2ZxJvZY", "sceAppInstUtilAppPrepareOverwritePkg",
         reinterpret_cast<u64>(&ExecutorAppInstUtilAppPrepareOverwritePkg)},
        {"PFfR4JFlCAM", "sceAppInstUtilGetPrimaryAppSlot",
         reinterpret_cast<u64>(&ExecutorAppInstUtilGetPrimaryAppSlot)},
        {"bpLyMf0oVwQ", "sceAppInstUtilAppInstallPkg",
         reinterpret_cast<u64>(&ExecutorAppInstUtilAppInstallPkg)},
        {"WQbIW7cm-Bc", "sceBgftServiceInit",
         reinterpret_cast<u64>(&ExecutorBgftServiceInit)},
        {"8P9BlyDO4xE", "sceBgftServiceTerm",
         reinterpret_cast<u64>(&ExecutorBgftServiceTerm)},
        {"nd+0DEOC68A", "sceBgftServiceIntDownloadRegisterTaskByStorageEx",
         reinterpret_cast<u64>(&ExecutorBgftRegisterTaskByStorageEx)},
        {"9pyartiGi-o", "sceBgftServiceDownloadStartTask",
         reinterpret_cast<u64>(&ExecutorBgftServiceDownloadStartTask)},
        {"Q7qj97IDGtU", "sceBgftServiceDownloadStartTaskAll",
         reinterpret_cast<u64>(&ExecutorBgftServiceDownloadStartTaskAll)},
        {"vSjGf4cBpmU", "sceBgftServiceDownloadGetProgress",
         reinterpret_cast<u64>(&ExecutorBgftServiceDownloadGetProgress)},
        {"DDHG1a6+3q0", "roundf", reinterpret_cast<u64>(&::roundf)},
        {"mKhVDmYciWA", "floorf", reinterpret_cast<u64>(&::floorf)},
        {"GAUuLKGhsCw", "ceilf", reinterpret_cast<u64>(&::ceilf)},
        {"Vo8rvWtZw3g", "truncf", reinterpret_cast<u64>(&::truncf)},
        {"fmT2cjPoWBs", "fabsf", reinterpret_cast<u64>(&::fabsf)},
        {"Q+xU11-h0xQ", "sqrtf", reinterpret_cast<u64>(&::sqrtf)},
        {"Q4rRL34CEeE", "sinf", reinterpret_cast<u64>(&::sinf)},
        {"-P6FNMzk2Kc", "cosf", reinterpret_cast<u64>(&::cosf)},
        {"H8ya2H00jbI", "sin", reinterpret_cast<u64>(static_cast<double (*)(double)>(&::sin))},
        {"2WE3BTYVwKM", "cos", reinterpret_cast<u64>(static_cast<double (*)(double)>(&::cos))},
        {"T7uyNqP7vQA", "tan", reinterpret_cast<u64>(static_cast<double (*)(double)>(&::tan))},
        {"nlaojL9hDtA", "round", reinterpret_cast<u64>(static_cast<double (*)(double)>(&::round))},
        {"mpcTgMzhUY8", "floor", reinterpret_cast<u64>(static_cast<double (*)(double)>(&::floor))},
        {"gacfOmO8hNs", "ceil", reinterpret_cast<u64>(static_cast<double (*)(double)>(&::ceil))},
        {"MXRNWnosNlM", "sqrt", reinterpret_cast<u64>(static_cast<double (*)(double)>(&::sqrt))},
    };
    for (const auto& override : overrides) {
        if (std::strcmp(override.nid, nid) == 0) {
            return &override;
        }
    }
    return nullptr;
}

#ifdef __ANDROID__
static bool IsExecutorTraceableGpuStubName(const char* name);
#endif

template <int stub_index>
static u64 CommonStub() {
    auto entry = stub_nids[stub_index];
#ifdef __ANDROID__
    if (entry) {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_STUB] index=%d nid=%s name=%s caller=%p return=0\n",
                     stub_index, entry->nid, entry->name, __builtin_return_address(0));
        if (IsExecutorTraceableGpuStubName(entry->name)) {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_AEROLIB_GPU_STUB_CALL] index=%d nid=%s name=%s caller=%p return=0",
                                stub_index, entry->nid, entry->name,
                                __builtin_return_address(0));
        }
    } else {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_STUB] index=%d nid=%s name=Unknown caller=%p return=0\n",
                     stub_index, stub_nids_unknown[stub_index].c_str(),
                     __builtin_return_address(0));
    }
    std::fflush(stderr);
    return 0;
#else
    if (entry) {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_STUB] index=%d nid=%s name=%s caller=%p return=0\n",
                     stub_index, entry->nid, entry->name, __builtin_return_address(0));
        LOG_ERROR(Core, "Stub: {} (nid: {}) called, returning zero to {}", entry->name, entry->nid,
                  __builtin_return_address(0));
    } else {
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_STUB] index=%d nid=%s name=Unknown caller=%p return=0\n",
                     stub_index, stub_nids_unknown[stub_index].c_str(),
                     __builtin_return_address(0));
        LOG_ERROR(Core, "Stub: Unknown (nid: {}) called, returning zero to {}",
                  stub_nids_unknown[stub_index], __builtin_return_address(0));
    }
    std::fflush(stderr);
    return 0;
#endif
}

#ifdef __ANDROID__
enum class ExecutorLibcAllocatorOwner : u8 {
    Unconfigured,
    HLE,
    GuestLibc,
};

static std::atomic<ExecutorLibcAllocatorOwner> g_executor_libc_allocator_owner{
    ExecutorLibcAllocatorOwner::Unconfigured};

static const char* ExecutorLibcAllocatorOwnerName(ExecutorLibcAllocatorOwner owner) {
    switch (owner) {
    case ExecutorLibcAllocatorOwner::Unconfigured:
        return "unconfigured";
    case ExecutorLibcAllocatorOwner::HLE:
        return "executor-hle";
    case ExecutorLibcAllocatorOwner::GuestLibc:
        return "guest-libc";
    }
    return "invalid";
}

bool ExecutorConfigureLibcAllocatorOwner(bool guest_libc_available) {
    const auto requested = guest_libc_available ? ExecutorLibcAllocatorOwner::GuestLibc
                                                : ExecutorLibcAllocatorOwner::HLE;
    auto observed = ExecutorLibcAllocatorOwner::Unconfigured;
    if (g_executor_libc_allocator_owner.compare_exchange_strong(
            observed, requested, std::memory_order_acq_rel, std::memory_order_acquire)) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_ALLOCATOR_OWNER] action=configured owner=%s "
                            "guest_libc_available=%d",
                            ExecutorLibcAllocatorOwnerName(requested),
                            guest_libc_available ? 1 : 0);
        return true;
    }
    if (observed == requested) {
        return true;
    }
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_LIBC_ALLOCATOR_OWNER] action=configuration_rejected "
                        "requested=%s latched=%s guest_libc_available=%d",
                        ExecutorLibcAllocatorOwnerName(requested),
                        ExecutorLibcAllocatorOwnerName(observed),
                        guest_libc_available ? 1 : 0);
    return false;
}

bool ExecutorUseRealLibcAlloc() {
    auto owner = g_executor_libc_allocator_owner.load(std::memory_order_acquire);
    if (owner == ExecutorLibcAllocatorOwner::Unconfigured) {
        auto expected = ExecutorLibcAllocatorOwner::Unconfigured;
        if (g_executor_libc_allocator_owner.compare_exchange_strong(
                expected, ExecutorLibcAllocatorOwner::HLE, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            owner = ExecutorLibcAllocatorOwner::HLE;
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_LIBC_ALLOCATOR_OWNER] action=early_read owner=executor-hle "
                "fail_closed=1 lifecycle_error=configure_before_memory_or_relocation");
        } else {
            owner = expected;
        }
    }
    return owner == ExecutorLibcAllocatorOwner::GuestLibc;
}

bool IsLibcAllocFamilyNid(const char* nid) {
    if (nid == nullptr) {
        return false;
    }
    static const std::unordered_set<std::string_view> kAllocNids = {
        "gQX+4GDQjpM", "4GN2t+VJdx8", "4-GxiotYTWo",
        "2X5agFjKxMc", "mFBxBLtfT6Q", "hIkkDt5bctc",
        "Y7aJ1uydPMo", "FPxZe+7sKPI", "SapHB+u0OPE",
        "tIhsqj0qsFE", "fjbBk89tXmQ", "OONgpXDsjFI",
        "g7zzzLDYGw0", "qjBlw2cVMAM",
        "fJnpuVVBbKk", "hdm0YfMa7TQ", "ryUxD-60bKM", "Jh5qUcwiSEk",
        "z+P+xCnWLBk", "MLWl90SFWNE", "lYDzBVE5mZs", "FOt55ZNaVJk",
        "McsGnqV6yRE", "m-fSo3EbxNA", "bZx+FFSlkUM", "v09ZcAhZzSc",
        "nwujzxOPXzQ", "Y1RR+IQy6Pg", "Dt9kllUFXS0", "dH3ucvQhfSY",
        "Ujf3KzMvRmI", "dQumQIBx1Iw", "cVSk9y8URbc", "nNkYCQIDb+4",
        "2Btkg8k24Zg", "NDcSfcYZRC8", "FoARlup+lmg", "OGybVuPAhAY",
        "yFs8CcUN6CU",
        "-hn1tcVHq5Q", "pi90NsG3zPA", "W6SiVSiCDtI", "OJjm-QOIHlI", "LYo3GhIlB38",
        "gigoVHZvVPE", "xLXHyF8De0c", "Vla-Z+eXlxo", "iF1iQHzxBJU", "ljkqMcC4-mk",
        "qWESlyXMI3E", "fEoW6BJsPt4", "KfKBG2h2KdM", "p6lrRW8-MLY", "1HBNVdBWQVk",
        "NewD1IkVMeU", "k04jLXu3+Ic", "mfHdJTIvhuo", "pzUa7KEoydw", "gvqHvbjlHzA",
        "ssIzVl7hH8g", "ctoQMlFaZ-M", "yP1IuW1xeIk", "o2waU9UJw9E", "MEE4xlWGmaY",
        "G1C+IzPmhc0", "+ewEHVkVfcU", "HHKtLpzPl4A", "ZgbOkbpXeqI", "EJZBym9sZQ0",
        "LDfSNfOIwFI", "Tp3xgPS3Uwc",
        "QuZzFJD5Hrw", "-lZdT34nAAE", "u32UXVridxQ", "9mMuuhXMwqQ", "PKJcFUfhKtw",
        "7hOUKGcT6jM", "6JcY5RDA4jY", "OmG3YPCBLJs", "mO8NB8whKy8", "htdTOnMxDbQ",
        "kv4kgdjswN0", "4h3fLdA8LHw", "6AZDjebK-v8", "dQhl7bw-umE",
    };
    return kAllocNids.find(std::string_view{nid}) != kAllocNids.end();
}

bool IsLibcStdioFamilyNid(const char* nid) {
    if (nid == nullptr) {
        return false;
    }
    static const std::unordered_set<std::string_view> kStdioNids = {
        "fffwELXNVFA",
        "pDBDcY6uLSA",
        "xeYO4u7uyJ0",
        "uodLYyUip20",
        "lbB+UlZqVG0",
        "MpxhMh8QFro",
        "QrZZdJ8XsX0",
        "MUjC4lbHrK4",
        "E1iwBYkG3CM",
        "rQFVBXp-Cxg",
        "pkYiKw09PRA",
        "Qazy8LmXTvw",
        "5qP1iVQkdck",
        "3QIPIh-GDjw",
        "7PkSz+qnTto",
        "SHlt7EhOtqA",
        "8Q60JLJ6Rv4",
        "AEuF3F2f8TA",
        "-LFO7jhD5CE",
        "LxcEU+ICu8U",
        "NuydofHcR1w",
        "AHxyhN96dy4",
        "yxbGzBQC5xA",
        "St9nbxSoezk",
        "NL836gOLANs",
    };
    return kStdioNids.contains(std::string_view{nid});
}

bool IsLibcDirectoryFamilyNid(const char* nid) {
    if (nid == nullptr) {
        return false;
    }
    static const std::unordered_set<std::string_view> kDirectoryNids = {
        "ay3uROQAc5A",
        "KOy7MeQ7OAU",
        "lybyyKtP54c",
        "J0kng1yac3M",
        "XepdqehVYe4",
    };
    return kDirectoryNids.contains(std::string_view{nid});
}

bool TryGetAndroidX64ExecutorOverride(const char* nid, u64* virtual_address, std::string* name) {
    auto is_live_guest_libc_string = [](const char* value) {
        return value && (std::strcmp(value, "j4ViWNHEgww") == 0 ||
                         std::strcmp(value, "DfivPArhucg") == 0 ||
                         std::strcmp(value, "Ovb2dSJOAuE") == 0 ||
                         std::strcmp(value, "aesyjrHVWy4") == 0);
    };
    if (std::getenv("EXECUTOR_UNSAFE_GUEST_LIBC_STRINGS") && is_live_guest_libc_string(nid)) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LINK_RESOLVE] source=executor_override_skip "
                            "reason=unsafe_guest_libc_strings nid=%s",
                            nid ? nid : "");
        return false;
    }
    if (ExecutorUseRealLibcAlloc() && IsLibcAllocFamilyNid(nid)) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LINK_RESOLVE] source=executor_override_skip "
                            "reason=use_real_libc_alloc nid=%s",
                            nid ? nid : "");
        return false;
    }
    const auto* override = FindExecutorStubOverride(nid);
    if (!override) {
        return false;
    }
    if (name) {
        *name = override->name ? override->name : "";
    }
    if (virtual_address) {
        *virtual_address = GetStub(nid);
    }
    return true;
}

u64 GetAndroidX64HleStubForNative(const char* name, u64 native_function) {
    return GetAndroidX64NativeHleStub(name, native_function, "dynamic");
}
#endif

#define XREP_1(x) &CommonStub<x>,

#define XREP_2(x) XREP_1(x) XREP_1(x + 1)
#define XREP_4(x) XREP_2(x) XREP_2(x + 2)
#define XREP_8(x) XREP_4(x) XREP_4(x + 4)
#define XREP_16(x) XREP_8(x) XREP_8(x + 8)
#define XREP_32(x) XREP_16(x) XREP_16(x + 16)
#define XREP_64(x) XREP_32(x) XREP_32(x + 32)
#define XREP_128(x) XREP_64(x) XREP_64(x + 64)
#define XREP_256(x) XREP_128(x) XREP_128(x + 128)
#define XREP_512(x) XREP_256(x) XREP_256(x + 256)
#define XREP_1024(x) XREP_512(x) XREP_512(x + 512)
#define XREP_2048(x) XREP_1024(x) XREP_1024(x + 1024)
#define XREP_4096(x) XREP_2048(x) XREP_2048(x + 2048)
#define XREP_8192(x) XREP_4096(x) XREP_4096(x + 4096)

#ifdef __ANDROID__
#define STUBS_LIST XREP_8192(0)
#else
#define STUBS_LIST XREP_2048(0)
#endif

static u64 (*stub_handlers[MAX_STUBS])() = {STUBS_LIST};

#ifdef __ANDROID__
static bool IsExecutorTraceableGpuStubName(const char* name) {
    if (!name) {
        return false;
    }
    return std::strstr(name, "sceAgc") != nullptr || std::strstr(name, "sceGnm") != nullptr ||
           std::strstr(name, "sceMatAgc") != nullptr ||
           std::strstr(name, "sceFontGraphicsAgc") != nullptr ||
           std::strstr(name, "sceCompositorSetAgc") != nullptr;
}
#endif

#ifdef __ANDROID__
static void* GetExecutorFreeTypeHandle() {
    static void* handle = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        handle = dlopen(
            "/data/data/app.lsx4.android/files/probes/libexecutor_freetype_hle.so",
            RTLD_NOW | RTLD_GLOBAL);
        const char* err = handle ? "" : dlerror();
        if (std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/executor-ft-hle.log", "a")) {
            std::fprintf(f, "dlopen handle=%p err=%s\n", handle, err ? err : "");
            std::fclose(f);
        }
    }
    return handle;
}

extern "C" int ExecutorFtNewFace(void* library, const char* guest_path, long face_index, void** aface) {
    void* handle = GetExecutorFreeTypeHandle();
    if (!handle || !guest_path || !aface) {
        return 1;
    }
    using NewMemFace = int (*)(void*, const unsigned char*, long, long, void**);
    auto fn = reinterpret_cast<NewMemFace>(dlsym(handle, "FT_New_Memory_Face"));
    if (!fn) {
        return 1;
    }
    std::filesystem::path host_path;
    try {
        auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
        host_path = mnt->GetHostPath(guest_path);
    } catch (...) {
        host_path = guest_path;
    }
    long size = 0;
    unsigned char* buf = nullptr;
    bool read_ok = false;
    {
        std::ifstream f(host_path, std::ios::binary | std::ios::ate);
        if (f) {
            size = static_cast<long>(f.tellg());
            f.seekg(0);
            if (size > 0) {
                buf = static_cast<unsigned char*>(std::malloc(static_cast<size_t>(size)));
                if (buf && f.read(reinterpret_cast<char*>(buf), size)) {
                    read_ok = true;
                }
            }
        }
    }
    int ret = read_ok ? fn(library, buf, size, face_index, aface) : 1;
    if (!read_ok && buf) {
        std::free(buf);
    }
    if (std::FILE* lf = std::fopen(
            "/data/data/app.lsx4.android/files/executor-ft-hle.log", "a")) {
        std::fprintf(lf,
                     "EXECUTOR_FT_NEW_FACE guest=%s host=%s size=%ld ret=%d face=%p\n", guest_path,
                     host_path.string().c_str(), size, ret, *aface);
        std::fclose(lf);
    }
    std::fprintf(stderr, "[EXECUTOR_FT_NEW_FACE] guest=%s size=%ld ret=%d face=%p\n", guest_path, size,
                 ret, *aface);
    std::fflush(stderr);
    return ret;
}

extern "C" int ExecutorFtRenderGlyph(void* slot, int render_mode) {
    void* handle = GetExecutorFreeTypeHandle();
    if (!handle) {
        return 1;
    }
    auto real = reinterpret_cast<int (*)(void*, int)>(dlsym(handle, "FT_Render_Glyph"));
    const int ret = real ? real(slot, render_mode) : 1;
    auto mirror =
        reinterpret_cast<void (*)(void*)>(dlsym(handle, "executor_ft_mirror_openorbis_slot"));
    if (mirror && slot) {
        mirror(slot);
    }
    return ret;
}

extern "C" int ExecutorFtLoadChar(void* face, unsigned long char_code, int load_flags) {
    void* handle = GetExecutorFreeTypeHandle();
    if (!handle) {
        return 1;
    }
    auto real =
        reinterpret_cast<int (*)(void*, unsigned long, int)>(dlsym(handle, "FT_Load_Char"));
    const int ret = real ? real(face, char_code, load_flags) : 1;
    if (face) {
        void* slot = *reinterpret_cast<void**>(static_cast<unsigned char*>(face) + 0x98);
        auto mirror =
            reinterpret_cast<void (*)(void*)>(dlsym(handle, "executor_ft_mirror_openorbis_slot"));
        if (mirror && slot) {
            mirror(slot);
        }
    }
    return ret;
}

u64 ResolveExecutorFreeTypeHle(const char* nid) {
    if (!nid) {
        return 0;
    }
    if (std::strcmp(nid, "j-uMQE+unFY") == 0) {
        return reinterpret_cast<u64>(&ExecutorFtNewFace);
    }
    struct FtNid {
        const char* nid;
        const char* sym;
    };
    static const FtNid kFtNids[] = {
        {"GNsiLzm4SH8", "FT_Init_FreeType"},   {"dH02CpEEcYk", "FT_Done_FreeType"},
        {"40Wa2neSLM0", "FT_New_Memory_Face"}, {"OgFczHko0w8", "FT_Done_Face"},
        {"xuY99Qcb108", "FT_Set_Char_Size"},   {"bcM5xfX2SeA", "FT_Set_Transform"},
        {"G4v5vCrrp40", "FT_Select_Charmap"},  {"M156RuYoUbU", "FT_Load_Glyph"},
        {"obnSzeb-KXk", "FT_Get_Char_Index"},  {"LPtlB241JSg", "FT_Get_Kerning"},
        {"eMKHL+Cj50k", "FT_Get_Glyph"},
        {"c1nbr8oSWlg", "FT_Glyph_To_Bitmap"}, {"0rKmI1ZbMAc", "FT_Glyph_Stroke"},
        {"BBRhYfYBDgU", "FT_Glyph_StrokeBorder"}, {"RdrrejUMiJ0", "FT_Done_Glyph"},
        {"eqGCaeizvWM", "FT_Stroker_New"},     {"B3uU5aDZn94", "FT_Stroker_Set"},
        {"VW4GD-TaIUc", "FT_Library_SetLcdFilter"},
        {"GiuNQjoj4fs", "FT_Library_SetLcdFilterWeights"},
        {"lJxMIwj81SQ", "FT_Set_Pixel_Sizes"}, {"TCRmigbq7lc", "FT_Render_Glyph"},
        {"wPQ0rMK4-JU", "FT_Load_Char"},
    };
    const char* sym = nullptr;
    for (const auto& e : kFtNids) {
        if (std::strcmp(e.nid, nid) == 0) {
            sym = e.sym;
            break;
        }
    }
    if (!sym) {
        return 0;
    }
    static void* handle = nullptr;
    static bool tried = false;
    if (!tried) {
        tried = true;
        handle = dlopen(
            "/data/data/app.lsx4.android/files/probes/libexecutor_freetype_hle.so",
            RTLD_NOW | RTLD_GLOBAL);
        const char* err = handle ? "" : dlerror();
        if (std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/executor-ft-hle.log", "a")) {
            std::fprintf(f, "dlopen handle=%p err=%s\n", handle, err ? err : "");
            std::fclose(f);
        }
    }
    if (!handle) {
        return 0;
    }
    void* fn = dlsym(handle, sym);
    if (fn) {
        if (std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/executor-ft-hle.log", "a")) {
            std::fprintf(f, "resolved nid=%s sym=%s fn=%p\n", nid, sym, fn);
            std::fclose(f);
        }
    }
    return reinterpret_cast<u64>(fn);
}
#endif

u64 GetStub(const char* nid) {
    if (const auto* override = FindExecutorStubOverride(nid)) {
#ifdef __ANDROID__
        if (const u64 fast = GetAndroidX64FastGuestLibcStringStub(override->name)) {
            std::fprintf(stderr,
                         "[EXECUTOR_AEROLIB_OVERRIDE_FAST_X64] nid=%s name=%s address=%p\n",
                         override->nid, override->name, reinterpret_cast<void*>(fast));
            std::fflush(stderr);
            return fast;
        }
        const u64 address =
            GetAndroidX64NativeHleStub(override->name, override->address, "override");
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_OVERRIDE_X64] nid=%s name=%s address=%p target=%p\n",
                     override->nid, override->name, reinterpret_cast<void*>(address),
                     reinterpret_cast<void*>(override->address));
        std::fflush(stderr);
        return address;
#else
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_OVERRIDE] nid=%s name=%s address=%p\n",
                     override->nid, override->name, reinterpret_cast<void*>(override->address));
        std::fflush(stderr);
        return override->address;
#endif
    }

#ifdef __ANDROID__
    if (const u64 ft = ResolveExecutorFreeTypeHle(nid)) {
        return GetAndroidX64NativeHleStub(nid, ft, "freetype");
    }

    const std::string semantic_nid = nid ? nid : "";
    std::scoped_lock low_stub_lock(g_android_x64_low_stub_mutex);
    if (const auto cached = g_android_x64_low_stub_by_nid.find(semantic_nid);
        cached != g_android_x64_low_stub_by_nid.end()) {
        return cached->second;
    }
#endif

    if (
#ifdef __ANDROID__
        !AndroidHasLowStubSlot()
#else
        UsedStubEntries >= MAX_STUBS
#endif
    ) {
#ifdef __ANDROID__
        return GetAndroidX64ZeroStub(AndroidX64FallbackStubSlot);
#else
        return (u64)&UnknownStub;
#endif
    }

    const u32 slot = UsedStubEntries++;
    const auto entry = FindByNid(nid);
    if (!entry) {
        stub_nids_unknown[slot] = nid;
    } else {
        stub_nids[slot] = entry;
    }

#ifdef __ANDROID__
    u64 address = 0;
    if (entry && IsExecutorTraceableGpuStubName(entry->name)) {
        const u64 native = reinterpret_cast<u64>(stub_handlers[slot]);
        address = GetAndroidX64HleStub(slot, native);
        {
            std::scoped_lock lock(g_android_x64_hle_names_mutex);
            g_android_x64_hle_names_by_native[native] = entry->name ? entry->name : "";
            g_android_x64_hle_names_by_stub[address] = entry->name ? entry->name : "";
            g_android_x64_hle_target_by_stub[address] = native;
        }
        std::fprintf(stderr,
                     "[EXECUTOR_AEROLIB_GPU_STUB_TRACE] index=%u nid=%s name=%s address=%p "
                     "target=%p\n",
                     slot, entry->nid, entry->name, reinterpret_cast<void*>(address),
                     reinterpret_cast<void*>(native));
        std::fflush(stderr);
        if (std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/executor-stub-nids.log", "a")) {
            std::fprintf(f, "slot=%u nid=%s name=%s addr=%p gpu_trace=1\n", slot,
                         entry->nid, entry->name, reinterpret_cast<void*>(address));
            std::fclose(f);
        }
        g_android_x64_low_stub_by_nid.emplace(semantic_nid, address);
        return address;
    }
    address = GetAndroidX64ZeroStub(slot);
    std::fprintf(stderr, "[EXECUTOR_AEROLIB_X64_STUB] index=%u nid=%s address=%p\n", slot, nid,
                 reinterpret_cast<void*>(address));
    std::fflush(stderr);
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-stub-nids.log", "a")) {
        std::fprintf(f, "slot=%u nid=%s addr=%p\n", slot, nid ? nid : "<null>",
                     reinterpret_cast<void*>(address));
        std::fclose(f);
    }
    g_android_x64_low_stub_by_nid.emplace(semantic_nid, address);
    return address;
#else
    return (u64)stub_handlers[slot];
#endif
}

#ifdef __ANDROID__
static u64 g_mono_managed_dir_str = 0;
u64 ExecutorMonoManagedDirString() {
    if (g_mono_managed_dir_str) {
        return g_mono_managed_dir_str;
    }
    static const char kPath[] = "/app0/Media/Managed";
    void* p = ExecutorAllocDefaultMspaceOrFallback(16, sizeof(kPath));
    if (!p) {
        return 0;
    }
    std::memcpy(p, kPath, sizeof(kPath));
    g_mono_managed_dir_str = reinterpret_cast<u64>(p);
    return g_mono_managed_dir_str;
}

static void ExecutorReadGuestCStr(u64 ptr, char* out, int cap) {
    out[0] = 0;
    if (!ptr) {
        return;
    }
    for (int k = 0; k < cap - 1; ++k) {
        char c = 0;
        std::memcpy(&c, reinterpret_cast<const void*>(ptr + k), 1);
        if (c == 0) { out[k] = 0; break; }
        out[k] = (c >= 0x20 && c < 0x7f) ? c : '.';
        out[k + 1] = 0;
    }
}

u64 ExecutorMonoPathLog(u64 path_ptr, u64 arg1_ptr) {
    g_mono_loadhook_ready.store(1, std::memory_order_relaxed);
    char buf[96];
    char buf1[96];
    ExecutorReadGuestCStr(path_ptr, buf, sizeof(buf));
    ExecutorReadGuestCStr(arg1_ptr, buf1, sizeof(buf1));
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_SETPATH_CALLED] arg0=0x%llx \"%s\" arg1=0x%llx \"%s\"",
                        static_cast<unsigned long long>(path_ptr), buf,
                        static_cast<unsigned long long>(arg1_ptr), buf1);
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-mono-path.log", "a")) {
        std::fprintf(f, "setpath arg0=0x%llx \"%s\" arg1=0x%llx \"%s\"\n",
                     static_cast<unsigned long long>(path_ptr), buf,
                     static_cast<unsigned long long>(arg1_ptr), buf1);
        std::fclose(f);
    }
    return 0;
}

extern "C" void executor_conformance_note_mono_domain_open(const char* name);

u64 ExecutorMonoDomainOpenLog(u64 name_ptr) {
    char nm[176];
    ExecutorReadGuestCStr(name_ptr, nm, sizeof(nm));
    executor_conformance_note_mono_domain_open(nm);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_DOMAIN_OPEN] name=0x%llx \"%s\"",
                        static_cast<unsigned long long>(name_ptr), nm);
    if (std::strstr(nm, "SonyVitaSavedGames") != nullptr) {
        if (executor_jit_record_current_thread_state != nullptr) {
            executor_jit_record_current_thread_state("domain-open-SonyVitaSavedGames");
        }
        if (executor_jit_dump_thread_states != nullptr) {
            executor_jit_dump_thread_states("domain-open-SonyVitaSavedGames");
        }
    }
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-mono-domain-open.log", "a")) {
        std::fprintf(f, "domain_open name=0x%llx \"%s\"\n",
                     static_cast<unsigned long long>(name_ptr), nm);
        std::fclose(f);
    }
    return 0;
}

static u64 g_mono_install_preload_hook_addr = 0;
void ExecutorSetMonoInstallPreloadHook(u64 addr) { g_mono_install_preload_hook_addr = addr; }

u64 ExecutorMonoAssemblyPreloadHook(u64 aname_ptr, u64 assemblies_path, u64 user_data) {
    char nm[96];
    nm[0] = 0;
    if (aname_ptr) {
        u64 name_ptr = 0;
        std::memcpy(&name_ptr, reinterpret_cast<const void*>(aname_ptr), 8);
        if (name_ptr) {
            for (int k = 0; k < 95; ++k) {
                char c = 0;
                std::memcpy(&c, reinterpret_cast<const void*>(name_ptr + k), 1);
                if (c == 0) { nm[k] = 0; break; }
                nm[k] = (c >= 0x20 && c < 0x7f) ? c : '.';
                nm[k + 1] = 0;
            }
        }
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_PRELOAD_HOOK] aname=0x%llx name=\"%s\"",
                        static_cast<unsigned long long>(aname_ptr), nm);
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-mono-preload.log", "a")) {
        std::fprintf(f, "preload \"%s\"\n", nm);
        std::fclose(f);
    }
    return 0;
}

static u64 g_mono_mscorlib_name_str = 0;
u64 ExecutorMonoMscorlibNameString() {
    if (g_mono_mscorlib_name_str) {
        return g_mono_mscorlib_name_str;
    }
    static const char kName[] = "mscorlib";
    void* p = ExecutorAllocDefaultMspaceOrFallback(16, sizeof(kName));
    if (!p) {
        return 0;
    }
    std::memcpy(p, kName, sizeof(kName));
    g_mono_mscorlib_name_str = reinterpret_cast<u64>(p);
    return g_mono_mscorlib_name_str;
}

static u64 g_mono_mscorlib_path_str = 0;
u64 ExecutorMonoMscorlibPathString() {
    if (g_mono_mscorlib_path_str) {
        return g_mono_mscorlib_path_str;
    }
    static const char kPath[] = "/app0/Media/Managed/mscorlib.dll";
    void* p = ExecutorAllocDefaultMspaceOrFallback(16, sizeof(kPath));
    if (!p) {
        return 0;
    }
    std::memcpy(p, kPath, sizeof(kPath));
    g_mono_mscorlib_path_str = reinterpret_cast<u64>(p);
    return g_mono_mscorlib_path_str;
}

static u64 g_mono_status_scratch = 0;
u64 ExecutorMonoStatusScratch() {
    if (g_mono_status_scratch) {
        return g_mono_status_scratch;
    }
    void* p = ExecutorAllocDefaultMspaceOrFallback(16, 16);
    if (!p) {
        return 0;
    }
    std::memset(p, 0, 16);
    g_mono_status_scratch = reinterpret_cast<u64>(p);
    return g_mono_status_scratch;
}

static u64 g_mono_install_search_hook_addr = 0;
void ExecutorSetMonoInstallSearchHook(u64 addr) { g_mono_install_search_hook_addr = addr; }
static u64 g_mono_image_loaded_addr = 0;
void ExecutorSetMonoImageLoaded(u64 addr) { g_mono_image_loaded_addr = addr; }
static u64 g_mono_image_get_assembly_addr = 0;
void ExecutorSetMonoImageGetAssembly(u64 addr) { g_mono_image_get_assembly_addr = addr; }
static u64 g_mono_load_from_full_addr = 0;
void ExecutorSetMonoLoadFromFull(u64 addr) { g_mono_load_from_full_addr = addr; }
static u64 g_mono_add_internal_call_addr = 0;
void ExecutorSetMonoAddInternalCall(u64 addr) { g_mono_add_internal_call_addr = addr; }
static u64 g_mono_string_new_addr = 0;
void ExecutorSetMonoStringNew(u64 addr) { g_mono_string_new_addr = addr; }
static u64 g_mono_domain_get_addr = 0;
void ExecutorSetMonoDomainGet(u64 addr) { g_mono_domain_get_addr = addr; }
static u64 g_mono_domain_assembly_open_addr = 0;
void ExecutorSetMonoDomainAssemblyOpen(u64 addr) { g_mono_domain_assembly_open_addr = addr; }
static u64 g_unitylog_icall_name = 0;
static u64 g_unitylog_icall_noop = 0;
static u64 g_unitylog_icall_once = 0;
static u64 g_env_newline_icall_name = 0;
static u64 g_env_newline_cstr = 0;
static u64 g_env_newline_cache = 0;
static u64 g_env_newline_stub = 0;
static u64 g_env_newline_once = 0;
static u64 g_unity_compare_base_objects_icall_name = 0;
static u64 g_unity_compare_base_objects_icall_target = 0;
static u64 g_unity_compare_base_objects_icall_once = 0;
u64 ExecutorGetEnvironmentNewLineIcallStubCell() {
    return reinterpret_cast<u64>(&g_env_newline_stub);
}
u64 ExecutorWriteStringToUnityLogNoop(u64 str_ptr) {
    static int n = 0;
    if (n < 8) {
        ++n;
        char s[160]; ExecutorReadGuestCStr(str_ptr, s, sizeof(s));
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_WRITELOG_NOOP] called #%d str=0x%llx \"%s\"", n,
                            static_cast<unsigned long long>(str_ptr), s);
    }
    return 0;
}
static void ExecutorEnsureUnityLogIcallReg() {
    if (g_unitylog_icall_name == 0) {
        static const char kName[] = "UnityEngine.UnityLogWriter::WriteStringToUnityLog";
        void* p = ExecutorAllocDefaultMspaceOrFallback(16, sizeof(kName));
        if (p) { std::memcpy(p, kName, sizeof(kName)); g_unitylog_icall_name = reinterpret_cast<u64>(p); }
    }
    if (g_unitylog_icall_once == 0) {
        void* f = ExecutorAllocDefaultMspaceOrFallback(8, 8, true);
        if (f) g_unitylog_icall_once = reinterpret_cast<u64>(f);
    }
    if (g_unitylog_icall_noop == 0) {
        g_unitylog_icall_noop = GetAndroidX64NativeHleStub(
            "writelog_noop", reinterpret_cast<u64>(&ExecutorWriteStringToUnityLogNoop), "mono");
    }
}

static void ExecutorEnsureEnvironmentNewLineIcallReg() {
    if (g_env_newline_icall_name == 0) {
        static const char kName[] = "System.Environment::get_NewLine";
        void* p = ExecutorAllocDefaultMspaceOrFallback(16, sizeof(kName));
        if (p) {
            std::memcpy(p, kName, sizeof(kName));
            g_env_newline_icall_name = reinterpret_cast<u64>(p);
        }
    }
    if (g_env_newline_cstr == 0) {
        static const char kNewLine[] = "\n";
        void* p = ExecutorAllocDefaultMspaceOrFallback(16, sizeof(kNewLine));
        if (p) {
            std::memcpy(p, kNewLine, sizeof(kNewLine));
            g_env_newline_cstr = reinterpret_cast<u64>(p);
        }
    }
    if (g_env_newline_cache == 0) {
        void* p = ExecutorAllocDefaultMspaceOrFallback(8, 8, true);
        if (p) {
            g_env_newline_cache = reinterpret_cast<u64>(p);
        }
    }
    if (g_env_newline_once == 0) {
        void* p = ExecutorAllocDefaultMspaceOrFallback(8, 8, true);
        if (p) {
            g_env_newline_once = reinterpret_cast<u64>(p);
        }
    }
    if (g_env_newline_stub != 0 || !g_mono_string_new_addr || !g_mono_domain_get_addr ||
        !g_env_newline_cstr || !g_env_newline_cache) {
        return;
    }
    if (g_android_x64_native_hle_next_slot <= UsedStubEntries) {
        return;
    }
    const u32 slot = --g_android_x64_native_hle_next_slot;
    u8* code = GetAndroidX64StubSlot(slot);
    if (!code) {
        return;
    }
    std::size_t c = 0;
    code[c++] = 0x53;
    code[c++] = 0x48; code[c++] = 0xbb;
    std::memcpy(code + c, &g_env_newline_cache, 8); c += 8;
    code[c++] = 0x48; code[c++] = 0x8b; code[c++] = 0x03;
    code[c++] = 0x48; code[c++] = 0x85; code[c++] = 0xc0;
    code[c++] = 0x75; const std::size_t j_done = c++;
    code[c++] = 0x48; code[c++] = 0xb8;
    std::memcpy(code + c, &g_mono_domain_get_addr, 8); c += 8;
    code[c++] = 0xff; code[c++] = 0xd0;
    code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xc7;
    code[c++] = 0x48; code[c++] = 0xbe;
    std::memcpy(code + c, &g_env_newline_cstr, 8); c += 8;
    code[c++] = 0x48; code[c++] = 0xb8;
    std::memcpy(code + c, &g_mono_string_new_addr, 8); c += 8;
    code[c++] = 0xff; code[c++] = 0xd0;
    code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0x03;
    code[j_done] = static_cast<u8>(c - (j_done + 1));
    code[c++] = 0x5b;
    code[c++] = 0xc3;
    while (c < AndroidX64StubSize) {
        code[c++] = 0xcc;
    }
    __builtin___clear_cache(reinterpret_cast<char*>(code),
                            reinterpret_cast<char*>(code + AndroidX64StubSize));
    g_env_newline_stub = reinterpret_cast<u64>(code);
}

static void ExecutorEnsureUnityCompareBaseObjectsIcallReg() {
    if (g_unity_compare_base_objects_icall_name == 0) {
        static constexpr char kName[] =
            "UnityEngine.Object::CompareBaseObjectsInternal";
        void* p = ExecutorAllocDefaultMspaceOrFallback(16, sizeof(kName));
        if (p != nullptr) {
            std::memcpy(p, kName, sizeof(kName));
            g_unity_compare_base_objects_icall_name = reinterpret_cast<u64>(p);
        }
    }
    if (g_unity_compare_base_objects_icall_once == 0) {
        void* p = ExecutorAllocDefaultMspaceOrFallback(8, 8, true);
        if (p != nullptr) {
            g_unity_compare_base_objects_icall_once = reinterpret_cast<u64>(p);
        }
    }
    if (g_unity_compare_base_objects_icall_target != 0) {
        return;
    }
    static constexpr u64 kReferenceTarget = 0x800225000ull;
    static constexpr std::array<u8, 14> kReferenceSignature = {
        0x55, 0x48, 0x89, 0xe5, 0xe8, 0x27, 0x3b,
        0xe1, 0xff, 0x0f, 0xb6, 0xc0, 0x5d, 0xc3,
    };
    std::array<u8, kReferenceSignature.size()> actual{};
    if (ExecutorReadGuestBytes(kReferenceTarget, actual.data(), actual.size()) &&
        actual == kReferenceSignature) {
        g_unity_compare_base_objects_icall_target = kReferenceTarget;
    }
}

extern "C" std::uint64_t ExecutorGetLastHleGuestRsp();

static std::atomic<int> g_corlib_stack_dumped{0};
static void ExecutorMonoCorlibDumpStack() {
    int expected = 0;
    if (!g_corlib_stack_dumped.compare_exchange_strong(expected, 1)) {
        return;
    }
    const u64 rsp = ExecutorGetLastHleGuestRsp();
    std::FILE* f = std::fopen(
        "/data/data/app.lsx4.android/files/executor-mono-search.log", "a");
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_CORLIB] stackdump rsp=0x%llx",
                        static_cast<unsigned long long>(rsp));
    if (f) {
        std::fprintf(f, "stackdump rsp=0x%llx\n", static_cast<unsigned long long>(rsp));
    }
    const u64 kMonoBase = 0x803190000ULL;
    int found = 0;
    for (int i = 0; i < 512 && found < 24; ++i) {
        u64 v = 0;
        std::memcpy(&v, reinterpret_cast<const void*>(rsp + static_cast<u64>(i) * 8), 8);
        if (v >= 0x800000000ULL && v < 0x810000000ULL) {
            const long long off = static_cast<long long>(v) - static_cast<long long>(kMonoBase);
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_MONO_CORLIB] stk[%d] 0x%llx mono+0x%llx", i,
                                static_cast<unsigned long long>(v), static_cast<long long>(off));
            if (f) {
                std::fprintf(f, "stk[%d] 0x%llx mono%+lld\n", i,
                             static_cast<unsigned long long>(v), off);
            }
            ++found;
        }
    }
    if (f) {
        std::fclose(f);
    }
}

static std::atomic<u32> g_corlib_decide_count{0};
struct ExecutorPHashRepair { std::atomic<u64> slot; std::atomic<u64> valid; std::atomic<u32> kind; };
static ExecutorPHashRepair g_phash_repair[64];
static std::atomic<u32> g_phash_repair_n{0};
static std::atomic<int> g_phash_poller_started{0};
static std::atomic<u64> g_phash_restores{0};
static std::atomic<u64> g_pin_restores{0};
static inline bool ExecutorPRAddrMapped(u64 a) { return a >= 0x200000000ULL && a < 0x300000000ULL; }
static inline bool ExecutorPRIsMonoFn(u64 a) { return a >= 0x803000000ULL && a < 0x803400000ULL; }
static void* ExecutorPHashRepairPoller(void*) {
    for (;;) {
        const u32 n = g_phash_repair_n.load(std::memory_order_acquire);
        for (u32 i = 0; i < n; ++i) {
            const u64 slot = g_phash_repair[i].slot.load(std::memory_order_relaxed);
            if (!ExecutorPRAddrMapped(slot)) continue;
            u64 cur = 0;
            std::memcpy(&cur, reinterpret_cast<const void*>(slot), 8);
            const u64 valid = g_phash_repair[i].valid.load(std::memory_order_relaxed);
            const u32 kind = g_phash_repair[i].kind.load(std::memory_order_relaxed);
            if (kind == 1) {
                if (cur != valid) {
                    std::memcpy(reinterpret_cast<void*>(slot), &valid, 8);
                    const u64 r = g_pin_restores.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (r <= 16 || (r & (r - 1)) == 0) {
                        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                            "[EXECUTOR_PIN_RESTORE] slot=0x%llx bad=0x%llx "
                                            "restored=0x%llx total=%llu",
                                            (unsigned long long)slot, (unsigned long long)cur,
                                            (unsigned long long)valid, (unsigned long long)r);
                    }
                }
                continue;
            }
            if (!ExecutorPRAddrMapped(cur)) {
                std::memcpy(reinterpret_cast<void*>(slot), &valid, 8);
                g_phash_restores.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            u64 hf = 0;
            std::memcpy(&hf, reinterpret_cast<const void*>(cur), 8);
            if (ExecutorPRIsMonoFn(hf)) {
                if (cur != valid) g_phash_repair[i].valid.store(cur, std::memory_order_relaxed);
            } else {
                std::memcpy(reinterpret_cast<void*>(slot), &valid, 8);
                g_phash_restores.fetch_add(1, std::memory_order_relaxed);
            }
        }
        usleep(20);
    }
    return nullptr;
}
extern "C" void ExecutorRecordPHashRepairKind(u64 slot, u64 valid, u32 kind);
extern "C" void ExecutorRecordPHashRepair(u64 slot, u64 valid) {
    ExecutorRecordPHashRepairKind(slot, valid, 0);
}
extern "C" void ExecutorRecordPHashRepairKind(u64 slot, u64 valid, u32 kind) {
    if (access("/data/data/app.lsx4.android/files/lsx4-home/run-no-phash-repair", F_OK) == 0) {
        return;
    }
    if (!ExecutorPRAddrMapped(slot) || !ExecutorPRAddrMapped(valid)) {
        return;
    }
    const u32 n = g_phash_repair_n.load(std::memory_order_acquire);
    for (u32 i = 0; i < n; ++i) {
        if (g_phash_repair[i].slot.load(std::memory_order_relaxed) == slot) {
            return;
        }
    }
    if (n >= 64) {
        return;
    }
    g_phash_repair[n].slot.store(slot, std::memory_order_relaxed);
    g_phash_repair[n].valid.store(valid, std::memory_order_relaxed);
    g_phash_repair[n].kind.store(kind, std::memory_order_relaxed);
    g_phash_repair_n.store(n + 1, std::memory_order_release);
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_PHASH_REPAIR] tracking slot=0x%llx valid=0x%llx kind=%u (#%u)",
                        static_cast<unsigned long long>(slot),
                        static_cast<unsigned long long>(valid), kind, n);
    int e = 0;
    if (g_phash_poller_started.compare_exchange_strong(e, 1)) {
        pthread_t t;
        if (pthread_create(&t, nullptr, &ExecutorPHashRepairPoller, nullptr) == 0) {
            pthread_detach(t);
        }
    }
}

void ExecutorMonoLoadHookLog(u64 assembly) {
    if (access("/data/data/app.lsx4.android/files/lsx4-home/run-mono-no-llvm", F_OK) != 0) {
        const std::uint32_t one = 1;
        std::memcpy(reinterpret_cast<void*>(0x803398058ULL), &one, sizeof(one));
    }
    static std::atomic<int> budget{160};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    auto rdstr = [](u64 p, char* out, int n) {
        out[0] = 0;
        if (p < 0x200000000ULL || p >= 0x300000000ULL) return;
        for (int k = 0; k < n - 1; ++k) {
            unsigned char c = 0;
            std::memcpy(&c, reinterpret_cast<const void*>(p + k), 1);
            if (!c) { out[k] = 0; break; }
            out[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
            out[k + 1] = 0;
        }
    };
    auto img_ok = [](u64 im) {
        if (im < 0x200000000ULL || im >= 0x300000000ULL) return false;
        u64 nm = 0;
        std::memcpy(&nm, reinterpret_cast<const void*>(im + 0x20), 8);
        return nm >= 0x200000000ULL && nm < 0x300000000ULL;
    };
    u64 aname_name = 0, image = 0;
    if (assembly >= 0x200000000ULL && assembly < 0x300000000ULL) {
        std::memcpy(&aname_name, reinterpret_cast<const void*>(assembly + 0x10), 8);
        std::memcpy(&image, reinterpret_cast<const void*>(assembly + 0x58), 8);
        if (!img_ok(image)) {
            u64 i2 = 0;
            std::memcpy(&i2, reinterpret_cast<const void*>(assembly + 0x60), 8);
            if (img_ok(i2)) image = i2;
        }
    }
    if (image >= 0x200000000ULL && image < 0x300000000ULL) {
        ExecutorMonoFixImageGuid(image);
    }
    char aname_s[64];
    rdstr(aname_name, aname_s, sizeof(aname_s));
    char ipath[96];
    ipath[0] = 0;
    u64 image_name_p = 0, atab_base = 0, atab_rr = 0, img_aot = 0;
    if (image >= 0x200000000ULL && image < 0x300000000ULL) {
        std::memcpy(&image_name_p, reinterpret_cast<const void*>(image + 0x20), 8);
        rdstr(image_name_p, ipath, sizeof(ipath));
        std::memcpy(&atab_base, reinterpret_cast<const void*>(image + 0xc0 + 32 * 16), 8);
        std::memcpy(&atab_rr, reinterpret_cast<const void*>(image + 0xc0 + 32 * 16 + 8), 8);
        std::memcpy(&img_aot, reinterpret_cast<const void*>(image + 0x3b8), 8);
    }
    const unsigned rows = static_cast<unsigned>(atab_rr & 0xffffff);
    u64 img_guid = 0, mvid_ptr = 0;
    char guid_hex[80] = {};
    char guid_str[48] = {};
    char mvid_hex[40] = {};
    if (image >= 0x200000000ULL && image < 0x300000000ULL) {
        std::memcpy(&img_guid, reinterpret_cast<const void*>(image + 0x48), 8);
        std::memcpy(&mvid_ptr, reinterpret_cast<const void*>(image + 0x98), 8);
        if (img_guid >= 0x200000000ULL && img_guid < 0x300000000ULL) {
            int w = 0;
            for (int k = 0; k < 24 && w < 76; ++k) {
                unsigned char c = 0;
                std::memcpy(&c, reinterpret_cast<const void*>(img_guid + k), 1);
                w += std::snprintf(guid_hex + w, sizeof(guid_hex) - w, "%02x", c);
            }
            for (int k = 0; k < 46; ++k) {
                unsigned char c = 0;
                std::memcpy(&c, reinterpret_cast<const void*>(img_guid + k), 1);
                if (!c) { guid_str[k] = 0; break; }
                guid_str[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
            }
        }
        if (mvid_ptr >= 0x200000000ULL && mvid_ptr < 0x300000000ULL) {
            int w = 0;
            for (int k = 0; k < 16 && w < 36; ++k) {
                unsigned char c = 0;
                std::memcpy(&c, reinterpret_cast<const void*>(mvid_ptr + k), 1);
                w += std::snprintf(mvid_hex + w, sizeof(mvid_hex) - w, "%02x", c);
            }
        }
    }
    {
        auto rdq = [](u64 a) -> u64 {
            if (a < 0x200000000ULL || a >= 0x810000000ULL) return 0;
            u64 v = 0; std::memcpy(&v, reinterpret_cast<const void*>(a), 8); return v;
        };
        if (image >= 0x200000000ULL && image < 0x300000000ULL) {
            const u64 c510 = rdq(image + 0x510);
            const u64 v510 = rdq(c510);
            const u64 tbl = v510;
            if (ExecutorPRAddrMapped(c510) && ExecutorPRIsMonoFn(rdq(tbl))) {
                ExecutorRecordPHashRepair(c510, v510);
            }
            const u64 v550 = rdq(image + 0x550);
            if (v550 != 0) {
                ExecutorRecordPHashRepairKind(image + 0x550, v550, 1);
            }
            if (std::FILE* cf = std::fopen(
                    "/data/data/app.lsx4.android/files/executor-loadhook.log", "a")) {
                std::fprintf(cf,
                    "CACHEPROBE image=0x%llx +0x510=0x%llx *(+0x510)=0x%llx mutex(+0x550)=0x%llx "
                    "tbl=0x%llx t0_hashfunc=0x%llx t8_equal=0x%llx t10_buckets=0x%llx t18_size=0x%llx\n",
                    (unsigned long long)image, (unsigned long long)c510, (unsigned long long)v510,
                    (unsigned long long)rdq(image + 0x550), (unsigned long long)tbl,
                    (unsigned long long)rdq(tbl), (unsigned long long)rdq(tbl + 8),
                    (unsigned long long)rdq(tbl + 0x10), (unsigned long long)rdq(tbl + 0x18));
                std::fclose(cf);
            }
        }
    }
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-loadhook.log", "a")) {
        std::fprintf(f,
                     "LOADHOOK asm=0x%llx aname.name=0x%llx \"%s\" image=0x%llx imgName=\"%s\" "
                     "ASSEMBLY rows=%u base=0x%llx img.aot=0x%llx guid(+0x48)=0x%llx hex=%s str=\"%s\" "
                     "mvid(+0x98)=%s\n",
                     static_cast<unsigned long long>(assembly),
                     static_cast<unsigned long long>(aname_name),
                     aname_name ? aname_s : "<NULL>",
                     static_cast<unsigned long long>(image), ipath, rows,
                     static_cast<unsigned long long>(atab_base),
                     static_cast<unsigned long long>(img_aot),
                     static_cast<unsigned long long>(img_guid), guid_hex, guid_str, mvid_hex);
        std::fclose(f);
    }
}

static void* ExecutorMonoDefaultsWatchdog(void*) {
    sleep(8);
    std::uint64_t df[24] = {};
    for (int i = 0; i < 24; ++i) {
        std::memcpy(&df[i], reinterpret_cast<const void*>(0x803398d98ULL + static_cast<u64>(i) * 8), 8);
    }
    u64 corlib_aot = 0;
    if (df[0] >= 0x200000000ULL && df[0] < 0x300000000ULL) {
        std::memcpy(&corlib_aot, reinterpret_cast<const void*>(df[0] + 0x3b8), 8);
    }
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-monodefaults.log", "a")) {
        std::fprintf(f, "WATCHDOG corlib=0x%llx corlib.aot=0x%llx object_class=0x%llx :",
                     static_cast<unsigned long long>(df[0]),
                     static_cast<unsigned long long>(corlib_aot),
                     static_cast<unsigned long long>(df[1]));
        for (int i = 0; i < 24; ++i) {
            std::fprintf(f, " [%d]=0x%llx", i, static_cast<unsigned long long>(df[i]));
        }
        std::fprintf(f, "\n");
        std::fclose(f);
    }
    return nullptr;
}

void ExecutorMonoLoadHookPostLog(u64 assembly) {
    static std::atomic<int> g_watchdog_started{0};
    int we = 0;
    if (g_watchdog_started.compare_exchange_strong(we, 1)) {
        pthread_t t;
        if (pthread_create(&t, nullptr, &ExecutorMonoDefaultsWatchdog, nullptr) == 0) {
            pthread_detach(t);
        }
    }
    static std::atomic<int> budget{160};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    u64 aname_name = 0, image = 0, img_aot = 0;
    if (assembly >= 0x200000000ULL && assembly < 0x300000000ULL) {
        std::memcpy(&aname_name, reinterpret_cast<const void*>(assembly + 0x10), 8);
        std::memcpy(&image, reinterpret_cast<const void*>(assembly + 0x58), 8);
    }
    char an[48] = {};
    if (aname_name >= 0x200000000ULL && aname_name < 0x300000000ULL) {
        for (int k = 0; k < 47; ++k) {
            unsigned char c = 0;
            std::memcpy(&c, reinterpret_cast<const void*>(aname_name + k), 1);
            if (!c) break;
            an[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
        }
    }
    if (image >= 0x200000000ULL && image < 0x300000000ULL) {
        std::memcpy(&img_aot, reinterpret_cast<const void*>(image + 0x3b8), 8);
    }
    if (img_aot >= 0x200000000ULL && img_aot < 0x300000000ULL) {
        executor_aot_got_install(img_aot);
    }
    if (image >= 0x200000000ULL && image < 0x300000000ULL) {
        auto rdq = [](u64 a) -> u64 {
            if (a < 0x200000000ULL || a >= 0x810000000ULL) return 0;
            u64 v = 0; std::memcpy(&v, reinterpret_cast<const void*>(a), 8); return v;
        };
        const u64 c510 = rdq(image + 0x510);
        const u64 tbl = rdq(c510);
        if (std::FILE* cf = std::fopen(
                "/data/data/app.lsx4.android/files/executor-loadhook.log", "a")) {
            std::fprintf(cf,
                "CACHEPROBE_POST image=0x%llx +0x510=0x%llx tbl=0x%llx t0_hashfunc=0x%llx "
                "t18_size=0x%llx aot(+0x3b8)=0x%llx\n",
                (unsigned long long)image, (unsigned long long)c510, (unsigned long long)tbl,
                (unsigned long long)rdq(tbl), (unsigned long long)rdq(tbl + 0x18),
                (unsigned long long)img_aot);
            std::fclose(cf);
        }
    }
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-loadhook.log", "a")) {
        std::fprintf(f, "POST asm=0x%llx aname=\"%s\" image=0x%llx img.aot(+0x3b8)=0x%llx\n",
                     static_cast<unsigned long long>(assembly), aname_name ? an : "<NULL>",
                     static_cast<unsigned long long>(image),
                     static_cast<unsigned long long>(img_aot));
        const bool of_interest = (std::strstr(an, "mscorlib") != nullptr ||
                                  std::strstr(an, "UnityEngine") != nullptr ||
                                  std::strstr(an, "System") != nullptr);
        if (of_interest && img_aot >= 0x200000000ULL && img_aot < 0x810000000ULL) {
            auto rd32 = [](u64 p) -> std::uint32_t {
                std::uint32_t v = 0; std::memcpy(&v, reinterpret_cast<const void*>(p), 4); return v;
            };
            auto rd64 = [](u64 p) -> u64 {
                u64 v = 0; std::memcpy(&v, reinterpret_cast<const void*>(p), 8); return v;
            };
            for (int t = 0; t < 4; ++t) {
                std::uint32_t num = rd32(img_aot + t * 4 + 0x144);
                std::uint32_t cnt = rd32(img_aot + t * 4 + 0x120);
                std::uint32_t sz = rd32(img_aot + t * 4 + 0x15c);
                std::uint32_t gb = rd32(img_aot + t * 4 + 0x150);
                u64 code = rd64(img_aot + t * 8 + 0x108);
                std::fprintf(f,
                             "  IMTPOOL[%s] type=%d num=%u count=%u size=%u got_base=%u code=0x%llx\n",
                             an, t, num, cnt, sz, gb, static_cast<unsigned long long>(code));
                if (t == 2 && code >= 0x200000000ULL && code < 0x810000000ULL && sz > 0 &&
                    sz <= 128) {
                    char hex[3 * 128 + 1];
                    int hp = 0;
                    for (std::uint32_t b = 0; b < sz && hp + 3 < static_cast<int>(sizeof(hex)); ++b) {
                        unsigned char by = 0;
                        std::memcpy(&by, reinterpret_cast<const void*>(code + b), 1);
                        hp += std::snprintf(hex + hp, sizeof(hex) - hp, "%02x ", by);
                    }
                    std::fprintf(f, "  IMTTHUNK0[%s] code=0x%llx size=%u bytes=%s\n", an,
                                 static_cast<unsigned long long>(code), sz, hex);
                }
            }
        }
        std::fclose(f);
    }
}

static std::atomic<int> g_loadhook_logger_state{0};
void ExecutorMaybeInstallMonoLoadHookLogger() {
    if (g_loadhook_logger_state.load(std::memory_order_relaxed) != 0) {
        return;
    }
    constexpr u64 kLoadHookList = 0x803392e10ULL;
    u64 head = 0;
    std::memcpy(&head, reinterpret_cast<const void*>(kLoadHookList), 8);
    if (head < 0x200000000ULL || head >= 0x810000000ULL) {
        return;
    }
    int e = 0;
    if (!g_loadhook_logger_state.compare_exchange_strong(e, 1)) {
        return;
    }
    const u64 stub = GetAndroidX64NativeHleStub(
        "mono_loadhook_log", reinterpret_cast<u64>(&ExecutorMonoLoadHookLog), "mono");
    void* node = stub ? ExecutorAllocDefaultMspaceOrFallback(16, 24) : nullptr;
    if (!stub || !node) {
        return;
    }
    const u64 np = reinterpret_cast<u64>(node);
    const u64 zero = 0;
    std::memcpy(reinterpret_cast<void*>(np + 0), &head, 8);
    std::memcpy(reinterpret_cast<void*>(np + 8), &stub, 8);
    std::memcpy(reinterpret_cast<void*>(np + 16), &zero, 8);
    std::memcpy(reinterpret_cast<void*>(kLoadHookList), &np, 8);
    const u64 post_stub = GetAndroidX64NativeHleStub(
        "mono_loadhook_post", reinterpret_cast<u64>(&ExecutorMonoLoadHookPostLog), "mono");
    void* post_node = post_stub ? ExecutorAllocDefaultMspaceOrFallback(16, 24) : nullptr;
    if (post_stub && post_node) {
        const u64 pp = reinterpret_cast<u64>(post_node);
        std::memcpy(reinterpret_cast<void*>(pp + 0), &zero, 8);
        std::memcpy(reinterpret_cast<void*>(pp + 8), &post_stub, 8);
        std::memcpy(reinterpret_cast<void*>(pp + 16), &zero, 8);
        u64 cur = np;
        for (int g = 0; g < 32; ++g) {
            u64 nx = 0;
            std::memcpy(&nx, reinterpret_cast<const void*>(cur + 0), 8);
            if (nx < 0x200000000ULL || nx >= 0x810000000ULL) break;
            cur = nx;
        }
        std::memcpy(reinterpret_cast<void*>(cur + 0), &pp, 8);
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[MONO_LOADHOOK_INSTALL] node=0x%llx stub=0x%llx oldhead=0x%llx",
                        static_cast<unsigned long long>(np), static_cast<unsigned long long>(stub),
                        static_cast<unsigned long long>(head));
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-loadhook.log", "a")) {
        std::fprintf(f, "[INSTALL] node=0x%llx stub=0x%llx oldhead=0x%llx\n",
                     static_cast<unsigned long long>(np), static_cast<unsigned long long>(stub),
                     static_cast<unsigned long long>(head));
        std::fclose(f);
    }
}

u64 ExecutorMonoCorlibShouldResolve(u64 aname_ptr) {
    ExecutorMaybeInstallMonoLoadHookLogger();
    char nm[96];
    nm[0] = 0;
    u64 name_ptr = 0;
    if (aname_ptr) {
        std::memcpy(&name_ptr, reinterpret_cast<const void*>(aname_ptr), 8);
    }
    if (!name_ptr) {
        return 0;
    }
    for (int k = 0; k < 95; ++k) {
        char c = 0;
        std::memcpy(&c, reinterpret_cast<const void*>(name_ptr + k), 1);
        if (c == 0) { nm[k] = 0; break; }
        nm[k] = c;
        nm[k + 1] = 0;
    }
    if (nm[0] == 0) {
        return 0;
    }
    const u32 n = g_corlib_decide_count.fetch_add(1, std::memory_order_relaxed);
    if (n < 200) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_CORLIB] #%u decide \"%s\"", n, nm);
        if (std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/executor-mono-search.log", "a")) {
            std::fprintf(f, "#%u decide \"%s\"\n", n, nm);
            std::fclose(f);
        }
    }
    return 1;
}

static std::mutex g_mono_search_open_mutex;
static std::unordered_set<std::string> g_mono_search_open_active;
static std::atomic<u32> g_mono_search_open_count{0};
static std::atomic<u32> g_mono_search_open_leave_count{0};

static bool ExecutorReadMonoAssemblyName(u64 aname_ptr, char* out, std::size_t cap) {
    if (!out || cap == 0) {
        return false;
    }
    out[0] = 0;
    u64 name_ptr = 0;
    if (aname_ptr) {
        std::memcpy(&name_ptr, reinterpret_cast<const void*>(aname_ptr), 8);
    }
    if (!name_ptr) {
        return false;
    }
    for (std::size_t k = 0; k + 1 < cap; ++k) {
        char c = 0;
        std::memcpy(&c, reinterpret_cast<const void*>(name_ptr + k), 1);
        if (c == 0) {
            out[k] = 0;
            return out[0] != 0;
        }
        if (c == '/' || c == '\\' || c == ':') {
            return false;
        }
        out[k] = (c >= 0x20 && c < 0x7f) ? c : '.';
        out[k + 1] = 0;
    }
    return out[0] != 0;
}

u64 ExecutorMonoSearchOpenEnter(u64 aname_ptr) {
    char name[128];
    if (!ExecutorReadMonoAssemblyName(aname_ptr, name, sizeof(name))) {
        return 0;
    }
    {
        std::lock_guard lk(g_mono_search_open_mutex);
        if (g_mono_search_open_active.find(name) != g_mono_search_open_active.end()) {
            const u32 n = g_mono_search_open_count.fetch_add(1, std::memory_order_relaxed);
            if (n < 128) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_MONO_SEARCH_OPEN] name=\"%s\" recursive=1 path=0x0",
                                    name);
            }
            return 0;
        }
        g_mono_search_open_active.insert(name);
    }
    const std::string path = std::string("/app0/Media/Managed/") + name + ".dll";
    void* p = ExecutorAllocDefaultMspaceOrFallback(16, path.size() + 1);
    if (!p) {
        std::lock_guard lk(g_mono_search_open_mutex);
        g_mono_search_open_active.erase(name);
        return 0;
    }
    std::memcpy(p, path.c_str(), path.size() + 1);
    const u64 guest_path = reinterpret_cast<u64>(p);
    const u32 n = g_mono_search_open_count.fetch_add(1, std::memory_order_relaxed);
    if (n < 160) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_SEARCH_OPEN] name=\"%s\" path=0x%llx \"%s\"",
                            name, static_cast<unsigned long long>(guest_path), path.c_str());
        if (std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/executor-mono-search.log", "a")) {
            std::fprintf(f, "[OPEN] name=\"%s\" path=0x%llx \"%s\"\n", name,
                         static_cast<unsigned long long>(guest_path), path.c_str());
            std::fclose(f);
        }
    }
    return guest_path;
}

u64 ExecutorMonoSearchOpenLeave(u64 aname_ptr, u64 assembly) {
    char name[128];
    if (ExecutorReadMonoAssemblyName(aname_ptr, name, sizeof(name))) {
        std::lock_guard lk(g_mono_search_open_mutex);
        g_mono_search_open_active.erase(name);
        const u32 n = g_mono_search_open_leave_count.fetch_add(1, std::memory_order_relaxed);
        if (n < 160) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_MONO_SEARCH_OPEN_RESULT] name=\"%s\" assembly=0x%llx",
                                name, static_cast<unsigned long long>(assembly));
            if (std::FILE* f = std::fopen(
                    "/data/data/app.lsx4.android/files/executor-mono-search.log", "a")) {
                std::fprintf(f, "[OPEN_RESULT] name=\"%s\" assembly=0x%llx\n", name,
                             static_cast<unsigned long long>(assembly));
                std::fclose(f);
            }
        }
    }
    return assembly;
}

static constexpr int kCorlibSynthSlots = 32;
struct ExecutorCorlibSynthEntry {
    u64 image = 0;
    u64 synth = 0;
};
static ExecutorCorlibSynthEntry g_corlib_synth_map[kCorlibSynthSlots] = {};
static u32 g_corlib_synth_used = 0;
static std::mutex g_corlib_synth_mutex;
static std::atomic<u32> g_corlib_synth_count{0};

void ExecutorMonoFixImageGuid(u64 image);

static std::atomic<int> g_mono_module_dumped{0};
void ExecutorMonoDumpModule() {
    int e = 0;
    if (!g_mono_module_dumped.compare_exchange_strong(e, 1)) {
        return;
    }
    const u64 lo = 0x803190000ULL, hi = 0x8033a0000ULL;
    std::FILE* f = std::fopen(
        "/data/data/app.lsx4.android/files/executor-mono-dump.bin", "wb");
    if (!f) {
        return;
    }
    u8 buf[8192];
    u64 total = 0;
    for (u64 a = lo; a < hi; a += sizeof(buf)) {
        const u64 n = (a + sizeof(buf) <= hi) ? sizeof(buf) : (hi - a);
        std::memcpy(buf, reinterpret_cast<const void*>(a), n);
        total += std::fwrite(buf, 1, n, f);
    }
    std::fclose(f);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[MONO_DUMP] wrote 0x%llx bytes from base 0x%llx",
                        static_cast<unsigned long long>(total),
                        static_cast<unsigned long long>(lo));
    std::FILE* ef = std::fopen(
        "/data/data/app.lsx4.android/files/executor-eboot-dump.bin", "wb");
    if (ef) {
        const u64 elo = 0x800000000ULL, ehi = 0x800100000ULL;
        u64 etotal = 0;
        for (u64 a = elo; a < ehi; a += sizeof(buf)) {
            const u64 n = (a + sizeof(buf) <= ehi) ? sizeof(buf) : (ehi - a);
            std::memcpy(buf, reinterpret_cast<const void*>(a), n);
            etotal += std::fwrite(buf, 1, n, ef);
        }
        std::fclose(ef);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EBOOT_DUMP] wrote 0x%llx bytes from base 0x%llx",
                            static_cast<unsigned long long>(etotal),
                            static_cast<unsigned long long>(elo));
    }
    std::FILE* pf = std::fopen(
        "/data/data/app.lsx4.android/files/executor-eboot-plt.bin", "wb");
    if (pf) {
        const u64 plo = 0x8020f0000ULL, phi = 0x802120000ULL;
        for (u64 a = plo; a < phi; a += sizeof(buf)) {
            const u64 n = (a + sizeof(buf) <= phi) ? sizeof(buf) : (phi - a);
            std::memcpy(buf, reinterpret_cast<const void*>(a), n);
            std::fwrite(buf, 1, n, pf);
        }
        std::fclose(pf);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EBOOT_PLT_DUMP] wrote from base 0x%llx", 0x8020f0000ULL);
    }
}

static std::atomic<int> g_aot_code_dumped{0};
void ExecutorMonoDumpAotCode() {
    int e = 0;
    if (!g_aot_code_dumped.compare_exchange_strong(e, 1)) {
        return;
    }
    const u64 addrs[] = {0x8033972d8ULL, 0x80333fa98ULL, 0x80333babdULL,
                         0x8032a49d9ULL, 0x8032a2f3bULL, 0x803241f4eULL};
    std::FILE* f = std::fopen(
        "/data/data/app.lsx4.android/files/executor-mono-search.log", "a");
    for (u64 ra : addrs) {
        const u64 startv = ra - 0xC0;
        for (int row = 0; row < 0xE0; row += 32) {
            char line[160];
            int w = 0;
            for (int b = 0; b < 32; ++b) {
                u8 c = 0;
                std::memcpy(&c, reinterpret_cast<const void*>(startv + static_cast<u64>(row + b)), 1);
                w += std::snprintf(line + w, sizeof(line) - w, "%02x", c);
            }
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[AOT_CODE] ra=0x%llx @0x%llx: %s",
                                static_cast<unsigned long long>(ra),
                                static_cast<unsigned long long>(startv + static_cast<u64>(row)), line);
            if (f) {
                std::fprintf(f, "[AOT_CODE] ra=0x%llx @0x%llx: %s\n",
                             static_cast<unsigned long long>(ra),
                             static_cast<unsigned long long>(startv + static_cast<u64>(row)), line);
            }
        }
    }
    if (f) {
        std::fclose(f);
    }
}

static std::atomic<u32> g_loaded_asms_log{0};
void ExecutorMonoRegisterInLoadedAsms(u64 assembly) {
    if (!assembly) {
        return;
    }
    constexpr u64 kHead = 0x803392e70ULL;
    u64 head = 0;
    std::memcpy(&head, reinterpret_cast<const void*>(kHead), 8);
    u64 n = head;
    for (int guard = 0; n && guard < 512; ++guard) {
        u64 data = 0;
        std::memcpy(&data, reinterpret_cast<const void*>(n), 8);
        if (data == assembly) {
            return;
        }
        std::memcpy(&n, reinterpret_cast<const void*>(n + 8), 8);
    }
    void* node = ExecutorAllocDefaultMspaceOrFallback(16, 24);
    if (!node) {
        return;
    }
    const u64 np = reinterpret_cast<u64>(node);
    const u64 zero = 0;
    std::memcpy(reinterpret_cast<void*>(np + 0), &assembly, 8);
    std::memcpy(reinterpret_cast<void*>(np + 8), &head, 8);
    std::memcpy(reinterpret_cast<void*>(np + 16), &zero, 8);
    if (head) {
        std::memcpy(reinterpret_cast<void*>(head + 16), &np, 8);
    }
    std::memcpy(reinterpret_cast<void*>(kHead), &np, 8);
    const u32 lg = g_loaded_asms_log.fetch_add(1, std::memory_order_relaxed);
    if (lg < 48) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[MONO_LOADED_ADD] assembly=0x%llx node=0x%llx prev_head=0x%llx",
                            static_cast<unsigned long long>(assembly),
                            static_cast<unsigned long long>(np),
                            static_cast<unsigned long long>(head));
        if (std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/executor-mono-search.log", "a")) {
            std::fprintf(f, "[MONO_LOADED_ADD] assembly=0x%llx node=0x%llx prev_head=0x%llx\n",
                         static_cast<unsigned long long>(assembly),
                         static_cast<unsigned long long>(np),
                         static_cast<unsigned long long>(head));
            std::fclose(f);
        }
    }
}

extern "C" int executor_lsx4_android_run_guest_signal_handler(std::uint64_t handler,
                                                                 std::uint64_t arg0,
                                                                 std::uint64_t arg1,
                                                                 std::uint64_t arg2,
                                                                 std::uint64_t* guest_result)
    __attribute__((weak));

void ExecutorMonoBindCorlibAot(u64 assembly) {
    if (!assembly) {
        return;
    }
    if (access("/data/data/app.lsx4.android/files/lsx4-home/run-mono-bind-aot", F_OK) != 0) {
        return;
    }
    if (!executor_lsx4_android_run_guest_signal_handler) {
        return;
    }
    constexpr u64 kLoadHookList = 0x803392e10ULL;
    u64 node = 0;
    std::memcpy(&node, reinterpret_cast<const void*>(kLoadHookList), 8);
    int n = 0;
    for (int guard = 0; node >= 0x200000000ULL && node < 0x810000000ULL && guard < 16; ++guard) {
        u64 next = 0, func = 0, user_data = 0;
        std::memcpy(&next, reinterpret_cast<const void*>(node + 0), 8);
        std::memcpy(&func, reinterpret_cast<const void*>(node + 8), 8);
        std::memcpy(&user_data, reinterpret_cast<const void*>(node + 16), 8);
        if (func >= 0x800000000ULL && func < 0x810000000ULL) {
            std::uint64_t result = 0;
            const int rc = executor_lsx4_android_run_guest_signal_handler(func, assembly, user_data,
                                                                             0, &result);
            ++n;
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[MONO_BIND_AOT] hook#%d func=0x%llx asm=0x%llx rc=%d result=0x%llx",
                                n, static_cast<unsigned long long>(func),
                                static_cast<unsigned long long>(assembly), rc,
                                static_cast<unsigned long long>(result));
            if (std::FILE* f = std::fopen(
                    "/data/data/app.lsx4.android/files/executor-mono-search.log", "a")) {
                std::fprintf(f,
                             "[MONO_BIND_AOT] hook#%d func=0x%llx asm=0x%llx ud=0x%llx rc=%d "
                             "result=0x%llx\n",
                             n, static_cast<unsigned long long>(func),
                             static_cast<unsigned long long>(assembly),
                             static_cast<unsigned long long>(user_data), rc,
                             static_cast<unsigned long long>(result));
                std::fclose(f);
            }
        }
        node = next;
    }
}

u64 ExecutorMonoCorlibSynth(u64 img) {
    if (!img) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(g_corlib_synth_mutex);
    u64 synth = 0;
    bool fresh = false;
    for (u32 i = 0; i < g_corlib_synth_used; ++i) {
        if (g_corlib_synth_map[i].image == img) {
            synth = g_corlib_synth_map[i].synth;
            break;
        }
    }
    if (!synth) {
        if (g_corlib_synth_used >= kCorlibSynthSlots) {
            return 0;
        }
        void* p = ExecutorAllocDefaultMspaceOrFallback(16, 256);
        if (!p) {
            return 0;
        }
        synth = reinterpret_cast<u64>(p);
        g_corlib_synth_map[g_corlib_synth_used].image = img;
        g_corlib_synth_map[g_corlib_synth_used].synth = synth;
        ++g_corlib_synth_used;
        fresh = true;
        u64* q = reinterpret_cast<u64*>(synth);
        for (int i = 0; i < 256 / 8; ++i) {
            q[i] = img;
        }
        u64 name_ptr = 0;
        std::memcpy(&name_ptr, reinterpret_cast<const void*>(img + 0x28), 8);
        if (name_ptr >= 0x200000000ULL && name_ptr < 0x300000000ULL) {
            q[2] = name_ptr;
        }
        ExecutorMonoFixImageGuid(img);
        ExecutorMonoRegisterInLoadedAsms(synth);
    }
    const u32 n = g_corlib_synth_count.fetch_add(1, std::memory_order_relaxed);
    if (n < 64) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_CORLIB] #%u synth=0x%llx img=0x%llx fresh=%d", n,
                            static_cast<unsigned long long>(synth),
                            static_cast<unsigned long long>(img), fresh ? 1 : 0);
        if (std::FILE* f = std::fopen(
                "/data/data/app.lsx4.android/files/executor-mono-search.log", "a")) {
            std::fprintf(f, "#%u synth=0x%llx img=0x%llx fresh=%d\n", n,
                         static_cast<unsigned long long>(synth),
                         static_cast<unsigned long long>(img), fresh ? 1 : 0);
            std::fclose(f);
        }
    }
    if (fresh) {
        const u64 synth_local = synth;
        lock.unlock();
        ExecutorMonoBindCorlibAot(synth_local);
        return synth_local;
    }
    return synth;
}

static constexpr int kRegInProgressMax = 16;
static u64 g_reg_in_progress[kRegInProgressMax] = {};
static std::mutex g_reg_in_progress_mutex;
static std::atomic<u32> g_reg_log_count{0};
static void ExecutorMonoRegLog(const char* fmt, u64 a, u64 b) {
    const u32 n = g_reg_log_count.fetch_add(1, std::memory_order_relaxed);
    if (n >= 48) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native", fmt, (unsigned long long)a,
                        (unsigned long long)b);
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-mono-search.log", "a")) {
        std::fprintf(f, fmt, (unsigned long long)a, (unsigned long long)b);
        std::fprintf(f, "\n");
        std::fclose(f);
    }
}
u64 ExecutorMonoEnterRegister(u64 image) {
    if (!image) {
        return 0;
    }
    std::lock_guard<std::mutex> lk(g_reg_in_progress_mutex);
    for (int i = 0; i < kRegInProgressMax; ++i) {
        if (g_reg_in_progress[i] == image) {
            ExecutorMonoRegLog("[MONO_REG] ENTER image=0x%llx own=0(recurse)", image, 0);
            return 0;
        }
    }
    for (int i = 0; i < kRegInProgressMax; ++i) {
        if (g_reg_in_progress[i] == 0) {
            g_reg_in_progress[i] = image;
            ExecutorMonoRegLog("[MONO_REG] ENTER image=0x%llx own=1(register)", image, 0);
            return 1;
        }
    }
    ExecutorMonoRegLog("[MONO_REG] ENTER image=0x%llx own=0(full)", image, 0);
    return 0;
}
void ExecutorMonoLeaveRegister(u64 image, u64 result) {
    ExecutorMonoRegLog("[MONO_REG] LEAVE image=0x%llx result=0x%llx", image, result);
    std::lock_guard<std::mutex> lk(g_reg_in_progress_mutex);
    for (int i = 0; i < kRegInProgressMax; ++i) {
        if (g_reg_in_progress[i] == image) {
            g_reg_in_progress[i] = 0;
            return;
        }
    }
}

static std::atomic<int> g_mono_img_probe_done{0};
void ExecutorMonoImageProbe(u64 image) {
    if (!image) {
        return;
    }
    int expected = 0;
    if (!g_mono_img_probe_done.compare_exchange_strong(expected, 1)) {
        return;
    }
    std::FILE* f = std::fopen(
        "/data/data/app.lsx4.android/files/executor-mono-search.log", "a");
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[MONO_IMG_PROBE] image=0x%llx", static_cast<unsigned long long>(image));
    if (f) {
        std::fprintf(f, "[MONO_IMG_PROBE] image=0x%llx\n", static_cast<unsigned long long>(image));
    }
    auto mapped = [](u64 p) {
        return (p >= 0x200000000ULL && p < 0x300000000ULL) ||
               (p >= 0x800000000ULL && p < 0x810000000ULL) ||
               (p >= 0x6f0000000ULL && p < 0x720000000ULL);
    };
    for (int i = 0; i < 96; ++i) {
        u64 p = 0;
        std::memcpy(&p, reinterpret_cast<const void*>(image + static_cast<u64>(i) * 8), 8);
        if (!mapped(p)) {
            continue;
        }
        char str[64];
        int n = 0;
        bool printable = true;
        for (; n < 63; ++n) {
            u8 c = 0;
            std::memcpy(&c, reinterpret_cast<const void*>(p + static_cast<u64>(n)), 1);
            if (c == 0) {
                break;
            }
            if (c < 0x20 || c >= 0x7f) {
                printable = false;
                break;
            }
            str[n] = static_cast<char>(c);
        }
        str[n] = 0;
        if (printable && n >= 3) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[MONO_IMG_PROBE] [+0x%x]=0x%llx -> \"%s\"", i * 8,
                                static_cast<unsigned long long>(p), str);
            if (f) {
                std::fprintf(f, "[MONO_IMG_PROBE] [+0x%x]=0x%llx -> \"%s\"\n", i * 8,
                             static_cast<unsigned long long>(p), str);
            }
        }
    }
    auto hexdump = [&](const char* tag, u64 base, int bytes) {
        for (int off = 0; off < bytes; off += 32) {
            char line[160];
            int w = 0;
            for (int b = 0; b < 32 && off + b < bytes; ++b) {
                u8 c = 0;
                std::memcpy(&c, reinterpret_cast<const void*>(base + static_cast<u64>(off + b)), 1);
                w += std::snprintf(line + w, sizeof(line) - w, "%02x", c);
            }
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[MONO_IMG_PROBE] %s+0x%02x: %s", tag, off, line);
            if (f) {
                std::fprintf(f, "[MONO_IMG_PROBE] %s+0x%02x: %s\n", tag, off, line);
            }
        }
    };
    hexdump("IMG", image, 0xA0);
    ExecutorMonoDumpModule();
    ExecutorMonoDumpAotCode();
    hexdump("AOTMOD", 0x80261fda8ULL, 0x40);
    for (int off = 0; off < 0x40; off += 8) {
        u64 p = 0;
        std::memcpy(&p, reinterpret_cast<const void*>(0x80261fda8ULL + static_cast<u64>(off)), 8);
        if (!mapped(p)) {
            continue;
        }
        char a[40];
        char h[80];
        int wh = 0;
        for (int b = 0; b < 36; ++b) {
            u8 c = 0;
            std::memcpy(&c, reinterpret_cast<const void*>(p + static_cast<u64>(b)), 1);
            a[b] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
            if (b < 20) {
                wh += std::snprintf(h + wh, sizeof(h) - wh, "%02x", c);
            }
        }
        a[36] = 0;
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[MONO_IMG_PROBE] AOTtgt[+0x%x]=0x%llx a=\"%s\" h=%s", off,
                            static_cast<unsigned long long>(p), a, h);
        if (f) {
            std::fprintf(f, "[MONO_IMG_PROBE] AOTtgt[+0x%x]=0x%llx a=\"%s\" h=%s\n", off,
                         static_cast<unsigned long long>(p), a, h);
        }
    }
    for (int off = 0x40; off < 0xA0; off += 8) {
        u64 p = 0;
        std::memcpy(&p, reinterpret_cast<const void*>(image + static_cast<u64>(off)), 8);
        if (!mapped(p)) {
            continue;
        }
        char line[80];
        int w = 0;
        for (int b = 0; b < 16; ++b) {
            u8 c = 0;
            std::memcpy(&c, reinterpret_cast<const void*>(p + static_cast<u64>(b)), 1);
            w += std::snprintf(line + w, sizeof(line) - w, "%02x", c);
        }
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[MONO_IMG_PROBE] IMGtgt[+0x%x]=0x%llx: %s", off,
                            static_cast<unsigned long long>(p), line);
        if (f) {
            std::fprintf(f, "[MONO_IMG_PROBE] IMGtgt[+0x%x]=0x%llx: %s\n", off,
                         static_cast<unsigned long long>(p), line);
        }
    }
    if (f) {
        std::fclose(f);
    }
}

static std::atomic<u32> g_fix_guid_log{0};
void ExecutorMonoFixImageGuid(u64 image) {
    if (!image) {
        return;
    }
    u64 guid_field = 0;
    u64 heap_guid = 0;
    std::memcpy(&guid_field, reinterpret_cast<const void*>(image + 0x48), 8);
    std::memcpy(&heap_guid, reinterpret_cast<const void*>(image + 0x98), 8);
    const bool heap_ok = heap_guid >= 0x200000000ULL && heap_guid < 0x300000000ULL;
    bool already_string = false;
    if (guid_field >= 0x200000000ULL && guid_field < 0x300000000ULL) {
        unsigned char c8 = 0;
        std::memcpy(&c8, reinterpret_cast<const void*>(guid_field + 8), 1);
        if (c8 == '-') already_string = true;
    }
    if (!already_string && heap_ok) {
        unsigned char m[16] = {};
        for (int k = 0; k < 16; ++k) {
            std::memcpy(&m[k], reinterpret_cast<const void*>(heap_guid + static_cast<u64>(k)), 1);
        }
        char gs[40];
        std::snprintf(gs, sizeof(gs),
                      "%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X", m[3], m[2],
                      m[1], m[0], m[5], m[4], m[7], m[6], m[8], m[9], m[10], m[11], m[12], m[13], m[14],
                      m[15]);
        void* buf = ExecutorAllocDefaultMspaceOrFallback(8, sizeof(gs));
        if (buf) {
            std::memcpy(buf, gs, sizeof(gs));
            const u64 sp = reinterpret_cast<u64>(buf);
            std::memcpy(reinterpret_cast<void*>(image + 0x48), &sp, 8);
            const u32 n = g_fix_guid_log.fetch_add(1, std::memory_order_relaxed);
            if (n < 16) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[MONO_FIX_GUID] image=0x%llx guid(+0x48) 0x%llx -> string \"%s\" @0x%llx",
                                    static_cast<unsigned long long>(image),
                                    static_cast<unsigned long long>(guid_field), gs,
                                    static_cast<unsigned long long>(sp));
                if (std::FILE* f = std::fopen(
                        "/data/data/app.lsx4.android/files/executor-mono-search.log", "a")) {
                    std::fprintf(f, "[MONO_FIX_GUID] image=0x%llx guid -> \"%s\" @0x%llx\n",
                                 static_cast<unsigned long long>(image), gs,
                                 static_cast<unsigned long long>(sp));
                    std::fclose(f);
                }
            }
        }
    }
}

static u64 g_mono_corlib_search_hook_stub = 0;
u64 ExecutorBuildMonoCorlibSearchHook() {
    if (g_mono_corlib_search_hook_stub) {
        return g_mono_corlib_search_hook_stub;
    }
    const u64 image_loaded = g_mono_image_loaded_addr;
    const u64 get_assembly = g_mono_image_get_assembly_addr;
    if (!image_loaded || !get_assembly) {
        return 0;
    }
    const u64 should = GetAndroidX64NativeHleStub(
        "mono_corlib_decide", reinterpret_cast<u64>(&ExecutorMonoCorlibShouldResolve), "mono");
    const u64 synthfn = GetAndroidX64NativeHleStub(
        "mono_corlib_synth", reinterpret_cast<u64>(&ExecutorMonoCorlibSynth), "mono");
    const u64 imgprobe = GetAndroidX64NativeHleStub(
        "mono_img_probe", reinterpret_cast<u64>(&ExecutorMonoImageProbe), "mono");
    const u64 fixguid = GetAndroidX64NativeHleStub(
        "mono_fix_guid", reinterpret_cast<u64>(&ExecutorMonoFixImageGuid), "mono");
    if (!should || !synthfn || !imgprobe || !fixguid) {
        return 0;
    }
    const u64 domain_get = g_mono_domain_get_addr;
    const u64 domain_open = g_mono_domain_assembly_open_addr;
    const u64 open_enter = GetAndroidX64NativeHleStub(
        "mono_search_open_enter", reinterpret_cast<u64>(&ExecutorMonoSearchOpenEnter), "mono");
    const u64 open_leave = GetAndroidX64NativeHleStub(
        "mono_search_open_leave", reinterpret_cast<u64>(&ExecutorMonoSearchOpenLeave), "mono");
    const bool use_domain_open =
        domain_get && domain_open && open_enter && open_leave &&
        access("/data/data/app.lsx4.android/files/lsx4-home/run-disable-mono-search-open",
               F_OK) != 0;
    const bool use_search_synth =
        access("/data/data/app.lsx4.android/files/lsx4-home/run-mono-search-synth",
               F_OK) == 0;
    constexpr u32 kSearchHookSlots = 3;
    if (g_android_x64_native_hle_next_slot <= UsedStubEntries + (kSearchHookSlots - 1)) {
        return 0;
    }
    g_android_x64_native_hle_next_slot -= kSearchHookSlots;
    const u32 slot = g_android_x64_native_hle_next_slot;
    u8* code = GetAndroidX64StubSlot(slot);
    if (!code) {
        return 0;
    }
    std::size_t c = 0;
    auto emit64 = [&](u64 value) {
        std::memcpy(code + c, &value, 8);
        c += 8;
    };
    auto emit_movabs_rax = [&](u64 value) {
        code[c++] = 0x48; code[c++] = 0xb8; emit64(value);
    };
    auto emit_call_rax = [&]() {
        code[c++] = 0xff; code[c++] = 0xd0;
    };
    auto emit_jcc32 = [&](u8 op2) {
        code[c++] = 0x0f; code[c++] = op2;
        const std::size_t pos = c;
        const s32 zero = 0;
        std::memcpy(code + c, &zero, 4);
        c += 4;
        return pos;
    };
    auto emit_jmp32 = [&]() {
        code[c++] = 0xe9;
        const std::size_t pos = c;
        const s32 zero = 0;
        std::memcpy(code + c, &zero, 4);
        c += 4;
        return pos;
    };
    auto patch32 = [&](std::size_t pos, std::size_t target) {
        const s64 delta = static_cast<s64>(target) - static_cast<s64>(pos + 4);
        const s32 rel = static_cast<s32>(delta);
        std::memcpy(code + pos, &rel, 4);
    };

    code[c++] = 0x53;
    code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xfb;
    emit_movabs_rax(should); emit_call_rax();
    code[c++] = 0x48; code[c++] = 0x85; code[c++] = 0xc0;
    const std::size_t j_skip = emit_jcc32(0x84);
    code[c++] = 0x48; code[c++] = 0x8b; code[c++] = 0x3b;
    code[c++] = 0x48; code[c++] = 0x85; code[c++] = 0xff;
    const std::size_t j_skip0 = emit_jcc32(0x84);
    emit_movabs_rax(image_loaded); emit_call_rax();
    code[c++] = 0x48; code[c++] = 0x85; code[c++] = 0xc0;
    const std::size_t j_loaded = emit_jcc32(0x85);

    std::size_t j_open_null = static_cast<std::size_t>(-1);
    std::size_t j_open_done = static_cast<std::size_t>(-1);
    std::size_t j_no_domain_done = static_cast<std::size_t>(-1);
    std::size_t j_no_loader = static_cast<std::size_t>(-1);
    if (use_domain_open) {
        code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xdf;
        emit_movabs_rax(open_enter); emit_call_rax();
        code[c++] = 0x48; code[c++] = 0x85; code[c++] = 0xc0;
        j_open_null = emit_jcc32(0x84);
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xec; code[c++] = 0x10;
        code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0x04; code[c++] = 0x24;
        emit_movabs_rax(domain_get); emit_call_rax();
        code[c++] = 0x48; code[c++] = 0x85; code[c++] = 0xc0;
        const std::size_t j_no_domain = emit_jcc32(0x84);
        code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xc7;
        code[c++] = 0x48; code[c++] = 0x8b; code[c++] = 0x34; code[c++] = 0x24;
        emit_movabs_rax(domain_open); emit_call_rax();
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xc4; code[c++] = 0x10;
        code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xc6;
        code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xdf;
        emit_movabs_rax(open_leave); emit_call_rax();
        j_open_done = emit_jmp32();
        const std::size_t no_domain = c;
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xc4; code[c++] = 0x10;
        code[c++] = 0x31; code[c++] = 0xf6;
        code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xdf;
        emit_movabs_rax(open_leave); emit_call_rax();
        j_no_domain_done = emit_jmp32();
        patch32(j_no_domain, no_domain);
    } else {
        j_no_loader = emit_jmp32();
    }

    const std::size_t loaded_image = c;
    code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xc3;
    code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xdf;
    emit_movabs_rax(fixguid); emit_call_rax();
    code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xdf;
    emit_movabs_rax(imgprobe); emit_call_rax();
    code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xdf;
    emit_movabs_rax(get_assembly); emit_call_rax();
    code[c++] = 0x48; code[c++] = 0x85; code[c++] = 0xc0;
    const std::size_t j_done = emit_jcc32(0x85);
    if (use_search_synth) {
        code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xdf;
        emit_movabs_rax(synthfn); emit_call_rax();
    } else {
        code[c++] = 0x31; code[c++] = 0xc0;
    }
    const std::size_t done = c;
    code[c++] = 0x5b;
    code[c++] = 0xc3;
    const std::size_t ret_null = c;
    code[c++] = 0x5b;
    code[c++] = 0x31; code[c++] = 0xc0;
    code[c++] = 0xc3;

    patch32(j_skip, ret_null);
    patch32(j_skip0, ret_null);
    patch32(j_loaded, loaded_image);
    patch32(j_done, done);
    if (use_domain_open) {
        patch32(j_open_null, ret_null);
        patch32(j_open_done, done);
        patch32(j_no_domain_done, done);
    } else {
        patch32(j_no_loader, ret_null);
    }
    __builtin___clear_cache(reinterpret_cast<char*>(code), reinterpret_cast<char*>(code + c));
    g_mono_corlib_search_hook_stub = reinterpret_cast<u64>(code);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_SEARCH_HOOK_BUILT] stub=0x%llx image_loaded=0x%llx "
                        "get_assembly=0x%llx domain_get=0x%llx domain_open=0x%llx mode=%s size=%zu",
                        static_cast<unsigned long long>(g_mono_corlib_search_hook_stub),
                        static_cast<unsigned long long>(image_loaded),
                        static_cast<unsigned long long>(get_assembly),
                        static_cast<unsigned long long>(domain_get),
                        static_cast<unsigned long long>(domain_open),
                        use_domain_open ? (use_search_synth ? "domain-open+synth" : "domain-open") :
                                          (use_search_synth ? "synth" : "real-or-null"),
                        c);
    return g_mono_corlib_search_hook_stub;
}

u64 ExecutorMakeMonoConfigInterposer(u64 real_addr) {
    if (!real_addr) {
        return real_addr;
    }
    const u64 logger = GetAndroidX64NativeHleStub("mono_path_log",
                                                  reinterpret_cast<u64>(&ExecutorMonoPathLog),
                                                  "mono");
    if (g_android_x64_native_hle_next_slot <= UsedStubEntries) {
        return real_addr;
    }
    const u32 slot = --g_android_x64_native_hle_next_slot;
    u8* code = GetAndroidX64StubSlot(slot);
    if (!code || !logger) {
        return real_addr;
    }
    std::size_t c = 0;
    code[c++] = 0x57;
    code[c++] = 0x48; code[c++] = 0xb8; std::memcpy(code + c, &logger, 8); c += 8;
    code[c++] = 0xff; code[c++] = 0xd0;
    code[c++] = 0x5f;
    code[c++] = 0x48; code[c++] = 0x85; code[c++] = 0xff;
    code[c++] = 0x74; const std::size_t j_skip = c++;
    code[c++] = 0x48; code[c++] = 0xb8; std::memcpy(code + c, &real_addr, 8); c += 8;
    code[c++] = 0xff; code[c++] = 0xe0;
    const std::size_t skip = c;
    code[c++] = 0x31; code[c++] = 0xc0;
    code[c++] = 0xc3;
    code[j_skip] = static_cast<u8>(skip - (j_skip + 1));
    __builtin___clear_cache(reinterpret_cast<char*>(code), reinterpret_cast<char*>(code + c));
    const u64 addr = reinterpret_cast<u64>(code);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_CONFIG_INTERPOSE] real=0x%llx interposer=0x%llx",
                        static_cast<unsigned long long>(real_addr),
                        static_cast<unsigned long long>(addr));
    return addr;
}

u64 ExecutorMakeMonoDomainOpenInterposer(u64 real_addr) {
    if (!real_addr) {
        return real_addr;
    }
    ExecutorSetMonoDomainAssemblyOpen(real_addr);
    ExecutorEnsureUnityLogIcallReg();
    ExecutorEnsureEnvironmentNewLineIcallReg();
    ExecutorEnsureUnityCompareBaseObjectsIcallReg();
    const bool do_unitylog_reg = g_mono_add_internal_call_addr && g_unitylog_icall_name &&
                                 g_unitylog_icall_noop && g_unitylog_icall_once;
    const bool do_newline_reg = g_mono_add_internal_call_addr && g_env_newline_icall_name &&
                                g_env_newline_stub && g_env_newline_once;
    const bool do_compare_base_objects_reg =
        g_mono_add_internal_call_addr && g_unity_compare_base_objects_icall_name &&
        g_unity_compare_base_objects_icall_target && g_unity_compare_base_objects_icall_once;
    const u64 logger = GetAndroidX64NativeHleStub(
        "mono_domain_open_log", reinterpret_cast<u64>(&ExecutorMonoDomainOpenLog), "mono");
    if (g_android_x64_native_hle_next_slot <= UsedStubEntries) {
        return real_addr;
    }
    const u32 slot = --g_android_x64_native_hle_next_slot;
    u8* code = GetAndroidX64StubSlot(slot);
    if (!code || !logger) {
        return real_addr;
    }
    std::size_t c = 0;
    code[c++] = 0x57;
    code[c++] = 0x56;
    code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xec; code[c++] = 0x08;
    code[c++] = 0x48; code[c++] = 0x89; code[c++] = 0xf7;
    code[c++] = 0x48; code[c++] = 0xb8; std::memcpy(code + c, &logger, 8); c += 8;
    code[c++] = 0xff; code[c++] = 0xd0;
    code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xc4; code[c++] = 0x08;
    code[c++] = 0x5e;
    code[c++] = 0x5f;
    if (do_unitylog_reg) {
        code[c++] = 0x48; code[c++] = 0xb8;
        std::memcpy(code + c, &g_unitylog_icall_once, 8); c += 8;
        code[c++] = 0x80; code[c++] = 0x38; code[c++] = 0x00;
        code[c++] = 0x75; const std::size_t jskip = c; code[c++] = 0x00;
        code[c++] = 0xc6; code[c++] = 0x00; code[c++] = 0x01;
        code[c++] = 0x57; code[c++] = 0x56;
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xec; code[c++] = 0x08;
        code[c++] = 0x48; code[c++] = 0xbf;
        std::memcpy(code + c, &g_unitylog_icall_name, 8); c += 8;
        code[c++] = 0x48; code[c++] = 0xbe;
        std::memcpy(code + c, &g_unitylog_icall_noop, 8); c += 8;
        code[c++] = 0x48; code[c++] = 0xb8;
        std::memcpy(code + c, &g_mono_add_internal_call_addr, 8); c += 8;
        code[c++] = 0xff; code[c++] = 0xd0;
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xc4; code[c++] = 0x08;
        code[c++] = 0x5e; code[c++] = 0x5f;
        code[jskip] = static_cast<u8>(c - (jskip + 1));
    }
    if (do_newline_reg) {
        code[c++] = 0x48; code[c++] = 0xb8;
        std::memcpy(code + c, &g_env_newline_once, 8); c += 8;
        code[c++] = 0x80; code[c++] = 0x38; code[c++] = 0x00;
        code[c++] = 0x75; const std::size_t jskip = c; code[c++] = 0x00;
        code[c++] = 0xc6; code[c++] = 0x00; code[c++] = 0x01;
        code[c++] = 0x57; code[c++] = 0x56;
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xec; code[c++] = 0x08;
        code[c++] = 0x48; code[c++] = 0xbf;
        std::memcpy(code + c, &g_env_newline_icall_name, 8); c += 8;
        code[c++] = 0x48; code[c++] = 0xbe;
        std::memcpy(code + c, &g_env_newline_stub, 8); c += 8;
        code[c++] = 0x48; code[c++] = 0xb8;
        std::memcpy(code + c, &g_mono_add_internal_call_addr, 8); c += 8;
        code[c++] = 0xff; code[c++] = 0xd0;
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xc4; code[c++] = 0x08;
        code[c++] = 0x5e; code[c++] = 0x5f;
        code[jskip] = static_cast<u8>(c - (jskip + 1));
    }
    if (do_compare_base_objects_reg) {
        code[c++] = 0x48; code[c++] = 0xb8;
        std::memcpy(code + c, &g_unity_compare_base_objects_icall_once, 8); c += 8;
        code[c++] = 0x80; code[c++] = 0x38; code[c++] = 0x00;
        code[c++] = 0x75; const std::size_t jskip = c; code[c++] = 0x00;
        code[c++] = 0xc6; code[c++] = 0x00; code[c++] = 0x01;
        code[c++] = 0x57; code[c++] = 0x56;
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xec; code[c++] = 0x08;
        code[c++] = 0x48; code[c++] = 0xbf;
        std::memcpy(code + c, &g_unity_compare_base_objects_icall_name, 8); c += 8;
        code[c++] = 0x48; code[c++] = 0xbe;
        std::memcpy(code + c, &g_unity_compare_base_objects_icall_target, 8); c += 8;
        code[c++] = 0x48; code[c++] = 0xb8;
        std::memcpy(code + c, &g_mono_add_internal_call_addr, 8); c += 8;
        code[c++] = 0xff; code[c++] = 0xd0;
        code[c++] = 0x48; code[c++] = 0x83; code[c++] = 0xc4; code[c++] = 0x08;
        code[c++] = 0x5e; code[c++] = 0x5f;
        code[jskip] = static_cast<u8>(c - (jskip + 1));
    }
    code[c++] = 0x48; code[c++] = 0xb8; std::memcpy(code + c, &real_addr, 8); c += 8;
    code[c++] = 0xff; code[c++] = 0xe0;
    __builtin___clear_cache(reinterpret_cast<char*>(code), reinterpret_cast<char*>(code + c));
    const u64 addr = reinterpret_cast<u64>(code);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_ICALL_REG] WriteStringToUnityLog do_reg=%d add_internal_call=0x%llx "
                        "name=0x%llx noop=0x%llx",
                        do_unitylog_reg ? 1 : 0,
                        (unsigned long long)g_mono_add_internal_call_addr,
                        (unsigned long long)g_unitylog_icall_name,
                        (unsigned long long)g_unitylog_icall_noop);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_ICALL_REG] Environment.get_NewLine do_reg=%d "
                        "domain_get=0x%llx string_new=0x%llx name=0x%llx stub=0x%llx",
                        do_newline_reg ? 1 : 0,
                        (unsigned long long)g_mono_domain_get_addr,
                        (unsigned long long)g_mono_string_new_addr,
                        (unsigned long long)g_env_newline_icall_name,
                        (unsigned long long)g_env_newline_stub);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_ICALL_REG] Object.CompareBaseObjectsInternal do_reg=%d "
                        "name=0x%llx target=0x%llx",
                        do_compare_base_objects_reg ? 1 : 0,
                        (unsigned long long)g_unity_compare_base_objects_icall_name,
                        (unsigned long long)g_unity_compare_base_objects_icall_target);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_DOMAIN_OPEN_INTERPOSE] real=0x%llx interposer=0x%llx",
                        static_cast<unsigned long long>(real_addr),
                        static_cast<unsigned long long>(addr));
    return addr;
}

u64 ExecutorMakeMonoPathInterposer(u64 real_addr, bool register_preload_hook) {
    const u64 str = ExecutorMonoManagedDirString();
    if (!str || !real_addr) {
        return real_addr;
    }
    const u64 logger = GetAndroidX64NativeHleStub("mono_path_log",
                                                  reinterpret_cast<u64>(&ExecutorMonoPathLog),
                                                  "mono");
    (void)register_preload_hook;
    const bool do_hook = false;
    const u64 hook_stub = do_hook ? ExecutorBuildMonoCorlibSearchHook() : 0;
    if (g_android_x64_native_hle_next_slot <= UsedStubEntries) {
        return real_addr;
    }
    const u32 slot = --g_android_x64_native_hle_next_slot;
    u8* code = GetAndroidX64StubSlot(slot);
    if (!code || !logger) {
        return real_addr;
    }
    std::size_t c = 0;
    code[c++] = 0x56;
    code[c++] = 0x48; code[c++] = 0xb8; std::memcpy(code + c, &logger, 8); c += 8;
    code[c++] = 0xff; code[c++] = 0xd0;
    if (do_hook && hook_stub) {
        code[c++] = 0x48; code[c++] = 0xbf; std::memcpy(code + c, &hook_stub, 8); c += 8;
        code[c++] = 0x31; code[c++] = 0xf6;
        code[c++] = 0x48; code[c++] = 0xb8;
        std::memcpy(code + c, &g_mono_install_search_hook_addr, 8); c += 8;
        code[c++] = 0xff; code[c++] = 0xd0;
    }
    code[c++] = 0x5e;
    code[c++] = 0x48; code[c++] = 0xbf; std::memcpy(code + c, &str, 8); c += 8;
    code[c++] = 0x48; code[c++] = 0xb8; std::memcpy(code + c, &real_addr, 8); c += 8;
    code[c++] = 0xff; code[c++] = 0xe0;
    __builtin___clear_cache(reinterpret_cast<char*>(code), reinterpret_cast<char*>(code + c));
    const u64 addr = reinterpret_cast<u64>(code);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_PATH_INTERPOSE] real=0x%llx str=0x%llx logger=0x%llx "
                        "stub=0x%llx do_hook=%d hook_stub=0x%llx install=0x%llx",
                        static_cast<unsigned long long>(real_addr),
                        static_cast<unsigned long long>(str),
                        static_cast<unsigned long long>(logger),
                        static_cast<unsigned long long>(addr), do_hook ? 1 : 0,
                        static_cast<unsigned long long>(hook_stub),
                        static_cast<unsigned long long>(g_mono_install_search_hook_addr));
    return addr;
}

void ExecutorPatchLibcInternalAllocators(u64 exec_seg_addr, u64 exec_seg_size) {
    if (ExecutorRuntimeFlagExists("executor-disable-libc-internal-redirect")) {
        std::fprintf(stderr, "[EXECUTOR_LIBC_INTERNAL_REDIRECT] disabled by flag\n");
        std::fflush(stderr);
        return;
    }
    static constexpr std::array<u8, 20> kMallocSig = {
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
        0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48, 0x48, 0x89, 0x4d};
    static constexpr std::array<u8, 20> kMallocSig2 = {
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
        0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x48, 0x48, 0x89, 0x55};
    static constexpr std::array<u8, 20> kMallocSig3 = {
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
        0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x38, 0x8a, 0x05, 0x99};
    static constexpr std::array<u8, 20> kMspaceCallocSig = {
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50,
        0x49, 0x89, 0xd0, 0xb8, 0x58, 0x58, 0x58, 0x58, 0x45, 0x31};
    static constexpr std::array<u8, 20> kReallocSig = {
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
        0x41, 0x54, 0x53, 0x50, 0x49, 0x89, 0xcc, 0x49, 0x89, 0xd1};
    static constexpr std::array<u8, 22> kFreeSig = {
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41,
        0x54, 0x53, 0x48, 0x83, 0xec, 0x28, 0x49, 0x89, 0xf4, 0x48, 0x89};
    static constexpr std::array<u8, 22> kFreeSig2 = {
        0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41,
        0x54, 0x53, 0x48, 0x83, 0xec, 0x18, 0x48, 0x89, 0xf3, 0x45, 0x31};

    const u64 malloc_stub = GetAndroidX64NativeHleStub(
        "executor_int_malloc", reinterpret_cast<u64>(&ExecutorLibcMalloc), "libc_internal");
    const u64 mspace_calloc_stub = GetAndroidX64NativeHleStub(
        "executor_int_mspace_calloc", reinterpret_cast<u64>(&ExecutorMspaceCalloc),
        "libc_internal");
    const u64 realloc_stub = GetAndroidX64NativeHleStub(
        "executor_int_realloc", reinterpret_cast<u64>(&ExecutorLibcRealloc), "libc_internal");
    const u64 free_stub = GetAndroidX64NativeHleStub(
        "executor_int_free", reinterpret_cast<u64>(&ExecutorLibcFree), "libc_internal");
    if (!malloc_stub || !mspace_calloc_stub || !realloc_stub || !free_stub) {
        std::fprintf(stderr,
                     "[EXECUTOR_LIBC_INTERNAL_REDIRECT] stub alloc failed m=%p c=%p r=%p f=%p (skipped)\n",
                     reinterpret_cast<void*>(malloc_stub),
                     reinterpret_cast<void*>(mspace_calloc_stub),
                     reinterpret_cast<void*>(realloc_stub), reinterpret_cast<void*>(free_stub));
        std::fflush(stderr);
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIBC_INTERNAL_REDIRECT] stub alloc failed m=%p c=%p "
                            "r=%p f=%p (skipped)",
                            reinterpret_cast<void*>(malloc_stub),
                            reinterpret_cast<void*>(mspace_calloc_stub),
                            reinterpret_cast<void*>(realloc_stub),
                            reinterpret_cast<void*>(free_stub));
        return;
    }

    auto* bytes = reinterpret_cast<u8*>(exec_seg_addr);
    auto find_one = [&](const u8* sig, std::size_t n) -> u8* {
        for (u64 off = 0; off + n <= exec_seg_size; ++off) {
            if (std::memcmp(bytes + off, sig, n) == 0) {
                return bytes + off;
            }
        }
        return nullptr;
    };

    auto write_shuffle1_jmp = [](u8* p, u64 stub) {
        std::size_t c = 0;
        p[c++] = 0x48; p[c++] = 0x89; p[c++] = 0xf7;
        p[c++] = 0x48; p[c++] = 0xb8;
        std::memcpy(p + c, &stub, 8); c += 8;
        p[c++] = 0xff; p[c++] = 0xe0;
        __builtin___clear_cache(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p + c));
    };
    auto write_direct_jmp = [](u8* p, u64 stub) {
        std::size_t c = 0;
        p[c++] = 0x48; p[c++] = 0xb8;
        std::memcpy(p + c, &stub, 8); c += 8;
        p[c++] = 0xff; p[c++] = 0xe0;
        __builtin___clear_cache(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p + c));
    };

    u8* m = find_one(kMallocSig.data(), kMallocSig.size());
    if (m) {
        write_shuffle1_jmp(m, malloc_stub);
    }
    u8* m2 = find_one(kMallocSig2.data(), kMallocSig2.size());
    if (m2 && m2 != m) {
        write_shuffle1_jmp(m2, malloc_stub);
    }
    u8* m3 = find_one(kMallocSig3.data(), kMallocSig3.size());
    if (m3 && m3 != m && m3 != m2) {
        write_shuffle1_jmp(m3, malloc_stub);
    }
    u8* c = find_one(kMspaceCallocSig.data(), kMspaceCallocSig.size());
    if (c) {
        write_direct_jmp(c, mspace_calloc_stub);
    }
    u8* f = find_one(kFreeSig.data(), kFreeSig.size());
    if (f) {
        write_shuffle1_jmp(f, free_stub);
    }
    u8* f2 = find_one(kFreeSig2.data(), kFreeSig2.size());
    if (f2 && f2 != f) {
        write_shuffle1_jmp(f2, free_stub);
    }
    u8* r = find_one(kReallocSig.data(), kReallocSig.size());
    if (r) {
        std::size_t c = 0;
        r[c++] = 0x48; r[c++] = 0x89; r[c++] = 0xf7;
        r[c++] = 0x48; r[c++] = 0x89; r[c++] = 0xd6;
        r[c++] = 0x48; r[c++] = 0xb8;
        std::memcpy(r + c, &realloc_stub, 8); c += 8;
        r[c++] = 0xff; r[c++] = 0xe0;
        __builtin___clear_cache(reinterpret_cast<char*>(r), reinterpret_cast<char*>(r + c));
    }
    std::fprintf(
        stderr,
        "[EXECUTOR_LIBC_INTERNAL_REDIRECT] seg=0x%llx size=0x%llx malloc@%p malloc2@%p "
        "malloc3@%p calloc@%p free@%p free2@%p realloc@%p stubs m=%p c=%p f=%p r=%p\n",
        (unsigned long long)exec_seg_addr, (unsigned long long)exec_seg_size, (void*)m,
        (void*)m2, (void*)m3, (void*)c, (void*)f, (void*)f2, (void*)r, (void*)malloc_stub,
        (void*)mspace_calloc_stub, (void*)free_stub, (void*)realloc_stub);
    std::fflush(stderr);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIBC_INTERNAL_REDIRECT] seg=0x%llx size=0x%llx malloc=%d "
                        "malloc2=%d malloc3=%d calloc=%d free=%d free2=%d realloc=%d "
                        "stubs m=%p c=%p f=%p r=%p",
                        (unsigned long long)exec_seg_addr, (unsigned long long)exec_seg_size,
                        m ? 1 : 0, m2 ? 1 : 0, m3 ? 1 : 0, c ? 1 : 0, f ? 1 : 0,
                        f2 ? 1 : 0, r ? 1 : 0,
                        (void*)malloc_stub, (void*)mspace_calloc_stub, (void*)free_stub,
                        (void*)realloc_stub);
    if (std::FILE* lf = std::fopen(
            "/data/data/app.lsx4.android/files/executor-libc-patch.log", "a")) {
        std::fprintf(lf,
                     "REDIRECT seg=0x%llx size=0x%llx malloc=%d malloc2=%d malloc3=%d "
                     "calloc=%d free=%d free2=%d realloc=%d m=%p c=%p f=%p r=%p\n",
                     (unsigned long long)exec_seg_addr, (unsigned long long)exec_seg_size,
                     m ? 1 : 0, m2 ? 1 : 0, m3 ? 1 : 0, c ? 1 : 0, f ? 1 : 0,
                     f2 ? 1 : 0, r ? 1 : 0,
                     (void*)malloc_stub, (void*)mspace_calloc_stub, (void*)free_stub,
                     (void*)realloc_stub);
        std::fclose(lf);
    }
}
#endif

}
