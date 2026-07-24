// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

#ifdef __ANDROID__
#include <string>
#endif

namespace Core::AeroLib {

u64 UnresolvedStub();

u64 GetStub(const char* nid);

#ifdef __ANDROID__
struct ExecutorAllocationSnapshot {
    u64 user_base{};
    u64 mapping_base{};
    u64 mapping_size{};
    u64 requested_size{};
    u64 allocated_size{};
    u64 mspace_handle{};
    u64 serial{};
    u64 offset{};
    u64 region_handle{};
    u64 region_base{};
    u64 region_size{};
    u64 region_offset{};
    bool found{};
    bool exact{};
    bool in_region{};
};

bool DescribeExecutorAllocation(u64 guest_ptr, ExecutorAllocationSnapshot* out);
void TraceExecutorAllocationsContaining(u64 guest_ptr, const char* phase);

u64 ExecutorLibcMalloc(u64 size);
u64 ExecutorLibcCalloc(u64 count, u64 size);
u64 ExecutorLibcFree(u64 ptr);

u64 EnsureAndroidX64ZeroStubSlab();
u64 GetAndroidX64ZeroStubSlabBase();
u64 GetAndroidX64ZeroStubSlabSize();
u64 GetAndroidX64HleStubForNative(const char* name, u64 native_function);
std::string GetAndroidX64HleNameForNative(u64 native_function);
bool IsAndroidX64NativeHleStub(u64 address, std::string* name);
bool TryGetAndroidX64NativeHleTarget(u64 address, u64* native_function);
bool IsAndroidX64NativeHleReturnZeroStub(u64 address);
bool TryGetAndroidX64ExecutorOverride(const char* nid, u64* virtual_address, std::string* name);
bool IsLibcAllocFamilyNid(const char* nid);
bool IsLibcStdioFamilyNid(const char* nid);
bool IsLibcDirectoryFamilyNid(const char* nid);
bool ExecutorConfigureLibcAllocatorOwner(bool guest_libc_available);
bool ExecutorUseRealLibcAlloc();
u64 ExecutorMakeMonoPathInterposer(u64 real_addr, bool register_preload_hook);
u64 ExecutorMakeMonoConfigInterposer(u64 real_addr);
u64 ExecutorMakeMonoDomainOpenInterposer(u64 real_addr);
void ExecutorSetMonoInstallPreloadHook(u64 addr);
void ExecutorSetMonoInstallSearchHook(u64 addr);
void ExecutorSetMonoImageLoaded(u64 addr);
void ExecutorSetMonoImageGetAssembly(u64 addr);
void ExecutorSetMonoLoadFromFull(u64 addr);
void ExecutorSetMonoAddInternalCall(u64 addr);
void ExecutorSetMonoStringNew(u64 addr);
void ExecutorSetMonoDomainGet(u64 addr);
void ExecutorSetMonoDomainAssemblyOpen(u64 addr);
u64 ExecutorGetEnvironmentNewLineIcallStubCell();
s64 ExecutorLibcStrcmp(u64 lhs, u64 rhs);
s64 ExecutorLibcStrncmp(u64 lhs, u64 rhs, u64 size);
u64 ExecutorLibcBsearch(u64 key, u64 base, u64 nmemb, u64 size, u64 compar);
#endif

}
