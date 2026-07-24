// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <string_view>
#include <system_error>

#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#ifdef __ANDROID__
#include "core/aerolib/stubs.h"
#endif
#include "core/file_sys/fs.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/libs.h"
#include "core/libraries/piglet/piglet_android.h"
#include "core/linker.h"

#ifdef __ANDROID__
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <pthread.h>

#include <android/log.h>

extern "C" std::uint64_t ExecutorGetLastHleGuestRsp();

extern "C" int executor_lsx4_android_run_guest_module_start(void* module,
                                                               u64 args,
                                                               const void* argp,
                                                               void* param,
                                                               int* out_result)
    __attribute__((weak));
extern "C" int executor_lsx4_android_patch_box64_hle_imports(void* module)
    __attribute__((weak));
extern "C" int executor_lsx4_android_register_box64_module_segments(void* module)
    __attribute__((weak));
#endif

namespace Libraries::Kernel {

#ifdef __ANDROID__
namespace {

constexpr s32 AndroidSyntheticModuleHandleBase = 0x5ad40000;
constexpr s32 AndroidPigletModuleHandle = AndroidSyntheticModuleHandleBase + 1;
constexpr s32 AndroidStoreJailbreakModuleHandle = AndroidSyntheticModuleHandleBase + 2;
constexpr s32 AndroidStoreRsaModuleHandle = AndroidSyntheticModuleHandleBase + 3;
constexpr s32 AndroidGenericSystemModuleHandle = AndroidSyntheticModuleHandleBase + 4;
constexpr s32 AndroidStoreCurlModuleHandle = AndroidSyntheticModuleHandleBase + 5;

bool IsAndroidSyntheticModuleHandle(s32 handle) {
    return handle >= AndroidSyntheticModuleHandleBase &&
           handle <= AndroidSyntheticModuleHandleBase + 0xfff;
}

bool AndroidSkipModuleStartRequested() {
    if (std::getenv("EXECUTOR_PS4_SKIP_MODULE_START") != nullptr) {
        return true;
    }
    std::error_code ec;
    if (std::filesystem::exists("lsx4-home/translator/EXECUTOR_PS4_SKIP_MODULE_START", ec)) {
        return true;
    }
    ec.clear();
    return std::filesystem::exists(
        "/data/data/app.lsx4.android/files/lsx4-home/translator/EXECUTOR_PS4_SKIP_MODULE_START",
        ec);
}

bool IsAndroidPigletModuleHandle(s32 handle) {
    return handle == AndroidPigletModuleHandle;
}

bool IsAndroidStoreCurlModuleHandle(s32 handle) {
    return handle == AndroidStoreCurlModuleHandle;
}

bool Contains(std::string_view value, std::string_view needle) {
    return value.find(needle) != std::string_view::npos;
}

bool Equals(std::string_view value, std::string_view expected) {
    return value == expected;
}

bool EndsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

std::atomic_int g_android_curl_log_budget{64};
constexpr u64 AndroidCurlEasyHandle = 0x5ad4c001ULL;
constexpr u64 AndroidCurlOfflineHttpCode = 0;
constexpr u64 AndroidCurlCouldntConnect = 7;

void AndroidCurlLog(const char* symbol, u64 arg0 = 0, u64 arg1 = 0, u64 arg2 = 0) {
    if (g_android_curl_log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_CURL_STUB] %s arg0=0x%llx arg1=0x%llx arg2=0x%llx",
                        symbol ? symbol : "<null>", static_cast<unsigned long long>(arg0),
                        static_cast<unsigned long long>(arg1),
                        static_cast<unsigned long long>(arg2));
}

u64 AndroidCurlGlobalInit(u64 flags, u64, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_global_init", flags);
    return 0;
}

u64 AndroidCurlGlobalCleanup(u64, u64, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_global_cleanup");
    return 0;
}

u64 AndroidCurlVersionStub(u64, u64, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_version");
    return 0;
}

u64 AndroidCurlEasyInit(u64, u64, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_easy_init");
    return AndroidCurlEasyHandle;
}

u64 AndroidCurlEasySetopt(u64 handle, u64 option, u64 value, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_easy_setopt", handle, option, value);
    return 0;
}

u64 AndroidCurlEasyPerform(u64 handle, u64, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_easy_perform", handle);
    return AndroidCurlCouldntConnect;
}

u64 AndroidCurlEasyCleanup(u64 handle, u64, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_easy_cleanup", handle);
    return 0;
}

u64 AndroidCurlEasyGetinfo(u64 handle, u64 info, u64 codep, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_easy_getinfo", handle, info, codep);
    if (codep != 0) {
        *reinterpret_cast<long*>(codep) = static_cast<long>(AndroidCurlOfflineHttpCode);
    }
    return 0;
}

u64 AndroidCurlEasyStrerror(u64 error, u64, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_easy_strerror", error);
    return 0;
}

u64 AndroidCurlSlistAppend(u64 list, u64 text, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_slist_append", list, text);
    return list != 0 ? list : AndroidCurlEasyHandle + 1;
}

u64 AndroidCurlSlistFreeAll(u64 list, u64, u64, u64, u64, u64, u64, u64) {
    AndroidCurlLog("curl_slist_free_all", list);
    return 0;
}

void* ResolveAndroidCurlSymbol(const char* symbol) {
    if (symbol == nullptr) {
        return nullptr;
    }

    const auto stub = [&](const char* name, auto* function) -> void* {
        const auto address =
            Core::AeroLib::GetAndroidX64HleStubForNative(name, reinterpret_cast<u64>(function));
        return reinterpret_cast<void*>(address);
    };

    const std::string_view name(symbol);
    if (Equals(name, "curl_global_init")) {
        return stub(symbol, &AndroidCurlGlobalInit);
    }
    if (Equals(name, "curl_global_cleanup")) {
        return stub(symbol, &AndroidCurlGlobalCleanup);
    }
    if (Equals(name, "curl_version")) {
        return stub(symbol, &AndroidCurlVersionStub);
    }
    if (Equals(name, "curl_easy_init")) {
        return stub(symbol, &AndroidCurlEasyInit);
    }
    if (Equals(name, "curl_easy_setopt")) {
        return stub(symbol, &AndroidCurlEasySetopt);
    }
    if (Equals(name, "curl_easy_perform")) {
        return stub(symbol, &AndroidCurlEasyPerform);
    }
    if (Equals(name, "curl_easy_cleanup")) {
        return stub(symbol, &AndroidCurlEasyCleanup);
    }
    if (Equals(name, "curl_easy_getinfo")) {
        return stub(symbol, &AndroidCurlEasyGetinfo);
    }
    if (Equals(name, "curl_easy_strerror")) {
        return stub(symbol, &AndroidCurlEasyStrerror);
    }
    if (Equals(name, "curl_slist_append")) {
        return stub(symbol, &AndroidCurlSlistAppend);
    }
    if (Equals(name, "curl_slist_free_all")) {
        return stub(symbol, &AndroidCurlSlistFreeAll);
    }
    return nullptr;
}

s32 TryLoadAndroidSyntheticModule(std::string_view guest_path, s32* pRes) {
    const auto finish = [&](s32 handle, std::string_view reason) {
        if (pRes != nullptr) {
            *pRes = ORBIS_OK;
        }
        LOG_INFO(Lib_Kernel, "[EXECUTOR_MODULE_STUB] {} => handle {:#x} ({})", guest_path, handle,
                 reason);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MODULE_STUB] %.*s => handle=0x%x (%.*s)",
                            static_cast<int>(guest_path.size()), guest_path.data(), handle,
                            static_cast<int>(reason.size()), reason.data());
        return handle;
    };

    if (Contains(guest_path, "libScePigletv2VSH.sprx") || Contains(guest_path, "/data/piglet.sprx") ||
        Contains(guest_path, "/data/compiler.sprx")) {
        return finish(AndroidPigletModuleHandle, "Store Piglet/OpenGL bootstrap");
    }

    if (Contains(guest_path, "/app0/Media/jb.prx")) {
        return finish(AndroidStoreJailbreakModuleHandle, "Store jailbreak helper no-op");
    }

    if (Contains(guest_path, "/app0/Media/rsa.prx")) {
        return finish(AndroidStoreRsaModuleHandle, "Store loader RSA helper no-op");
    }

    if (Contains(guest_path, "curl.prx")) {
        return finish(AndroidStoreCurlModuleHandle, "Itemzflow curl compatibility");
    }

    if (Contains(guest_path, "/system/common/lib/") || Contains(guest_path, "/system/priv/lib/") ||
        Contains(guest_path, "/sys/common/lib/") || Contains(guest_path, "/sys/priv/lib/")) {
        return finish(AndroidGenericSystemModuleHandle, "missing PS4 system PRX (sys/system alias)");
    }

    if ((EndsWith(guest_path, ".prx") || EndsWith(guest_path, ".sprx")) &&
        !guest_path.starts_with("/app0/")) {
        return finish(AndroidGenericSystemModuleHandle,
                      "Box64 guest PRX start deferred to synthetic dlsym stubs");
    }

    return -1;
}

}
#endif

s32 PS4_SYSV_ABI sceKernelIsInSandbox() {
    return 1;
}

s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    return Config::isNeoModeConsole() &&
           Common::ElfInfo::Instance().GetPSFAttributes().support_neo_mode;
}

s32 PS4_SYSV_ABI sceKernelHasNeoMode() {
    return Config::isNeoModeConsole();
}

s32 PS4_SYSV_ABI sceKernelGetMainSocId() {
    LOG_DEBUG(Lib_Kernel, "called");
    if (Config::isNeoModeConsole()) {
        return 0x740f30;
    }
    return 0x710f10;
}

s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* ver) {
    if (!ver) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    *ver = Common::ElfInfo::Instance().CompiledSdkVer();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetCpumode() {
    LOG_DEBUG(Lib_Kernel, "called");
    auto& attrs = Common::ElfInfo::Instance().GetPSFAttributes();
    u32 is_cpu6 = attrs.six_cpu_mode.Value();
    u32 is_cpu7 = attrs.seven_cpu_mode.Value();
    if (is_cpu6 == 1 && is_cpu7 == 1) {
        return 2;
    }
    if (is_cpu7 == 1) {
        return 5;
    }
    return 0;
}

s32 PS4_SYSV_ABI sceKernelGetCurrentCpu() {
    LOG_DEBUG(Lib_Kernel, "called");
    return 0;
}

void* PS4_SYSV_ABI sceKernelGetProcParam() {
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    return linker->GetProcParam();
}

s32 PS4_SYSV_ABI sceKernelLoadStartModule(const char* moduleFileName, u64 args, const void* argp,
                                          u32 flags, const void* pOpt, s32* pRes) {
    LOG_INFO(Lib_Kernel, "called filename = {}, args = {}", moduleFileName, args);
#ifdef __ANDROID__
    static std::atomic_int module_trace_budget{32};
    const bool trace_module =
        module_trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0;
    if (trace_module) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_MODULE_LOAD] path=%s args=0x%llx flags=0x%x",
                            moduleFileName ? moduleFileName : "<null>",
                            static_cast<unsigned long long>(args), flags);
    }
    if (moduleFileName == nullptr) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (flags != 0) {
        LOG_INFO(Lib_Kernel, "[EXECUTOR_MODULE_STUB] ignoring Android probe flags {:#x}", flags);
    }
#else
    ASSERT(flags == 0);
#endif

    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    auto* linker = Common::Singleton<Core::Linker>::Instance();

    std::filesystem::path path;
    std::string guest_path(moduleFileName);

    s32 handle = -1;

    auto load_and_start = [&](const std::filesystem::path& host_path) -> s32 {
#ifdef __ANDROID__
        const u32 existing = linker->FindByName(host_path);
        if (existing != static_cast<u32>(-1)) {
            handle = static_cast<s32>(existing);
            Core::Module* existing_module = linker->GetModule(handle);
            if (existing_module == nullptr) {
                return -1;
            }
            const auto lifecycle =
                existing_module->lifecycle_state.load(std::memory_order_acquire);
            if (lifecycle == Core::Module::LifecycleState::Started ||
                lifecycle == Core::Module::LifecycleState::Starting) {
                if (pRes != nullptr) {
                    *pRes = 0;
                }
                if (trace_module) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x "
                        "start=ALREADY_STARTED_SKIP_START lifecycle=%u",
                        guest_path.c_str(), host_path.string().c_str(), handle,
                        static_cast<unsigned>(lifecycle));
                }
                return handle;
            }

            if (executor_lsx4_android_run_guest_module_start == nullptr) {
                __android_log_print(
                    ANDROID_LOG_ERROR, "LSX4Native",
                    "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x "
                    "start=PRELOADED_NO_START_BRIDGE",
                    guest_path.c_str(), host_path.string().c_str(), handle);
                return -1;
            }
            void* param = existing_module->GetProcParam<void*>();
            int start_result = 0;
            existing_module->lifecycle_state.store(Core::Module::LifecycleState::Starting,
                                                    std::memory_order_release);
            const int rc = executor_lsx4_android_run_guest_module_start(
                existing_module, args, argp, param, &start_result);
            existing_module->lifecycle_state.store(
                rc >= 0 ? Core::Module::LifecycleState::Started
                        : Core::Module::LifecycleState::Loaded,
                std::memory_order_release);
            if (pRes != nullptr) {
                *pRes = start_result;
            }
            __android_log_print(
                rc >= 0 ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x "
                "start=PRELOADED_START startRc=%d startResult=0x%x args=0x%llx argp=%p",
                guest_path.c_str(), host_path.string().c_str(), handle, rc,
                static_cast<u32>(start_result), static_cast<unsigned long long>(args), argp);
            return rc >= 0 ? handle : -1;
        }
        handle = linker->LoadModule(host_path, true);
        if (handle == -1) {
            return -1;
        }
        Core::Module* module = linker->GetModule(handle);
        if (module == nullptr) {
            return -1;
        }
        if (executor_lsx4_android_register_box64_module_segments != nullptr) {
            executor_lsx4_android_register_box64_module_segments(module);
        }
        linker->RelocateAnyImports(module);
        if (executor_lsx4_android_patch_box64_hle_imports != nullptr) {
            executor_lsx4_android_patch_box64_hle_imports(module);
        }
        if (module->tls.image_size != 0) {
            linker->AdvanceGenerationCounter();
        }
        void* param = module->GetProcParam<void*>();
        int start_result = 0;
        if (AndroidSkipModuleStartRequested()) {
            if (pRes != nullptr) {
                *pRes = 0;
            }
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x start=SKIPPED_BY_ENV",
                                guest_path.c_str(), host_path.string().c_str(), handle);
            return handle;
        }
        if (executor_lsx4_android_run_guest_module_start == nullptr) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x start=NO_BOX64_CURRENT_EMU3",
                                guest_path.c_str(), host_path.string().c_str(), handle);
            return -1;
        }
        module->lifecycle_state.store(Core::Module::LifecycleState::Starting,
                                      std::memory_order_release);
        const int rc = executor_lsx4_android_run_guest_module_start(
            module, args, argp, param, &start_result);
        module->lifecycle_state.store(rc >= 0 ? Core::Module::LifecycleState::Started
                                              : Core::Module::LifecycleState::Loaded,
                                      std::memory_order_release);
        if (pRes != nullptr) {
            *pRes = start_result;
        }
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x startRc=%d startResult=0x%x",
                            guest_path.c_str(), host_path.string().c_str(), handle, rc,
                            static_cast<u32>(start_result));
        return rc >= 0 ? handle : -1;
#else
        return linker->LoadAndStartModule(host_path, args, argp, pRes);
#endif
    };

#ifdef __ANDROID__
    if (const s32 synthetic_handle = TryLoadAndroidSyntheticModule(guest_path, pRes);
        synthetic_handle >= 0) {
        return synthetic_handle;
    }
#endif

    if (guest_path[0] == '/') {
        path = mnt->GetHostPath(guest_path);
        handle = load_and_start(path);
        if (handle != -1) {
#ifdef __ANDROID__
            if (trace_module) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x",
                    guest_path.c_str(), path.string().c_str(), handle);
            }
#endif
            return handle;
        }
    } else {
        if (!guest_path.contains('/')) {
            path = mnt->GetHostPath("/app0/" + guest_path);
            handle = load_and_start(path);
            if (handle != -1) {
#ifdef __ANDROID__
                if (trace_module) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x",
                        guest_path.c_str(), path.string().c_str(), handle);
                }
#endif
                return handle;
            }
        } else {
            path = mnt->GetHostPath(guest_path);
            handle = load_and_start(path);
            if (handle != -1) {
#ifdef __ANDROID__
                if (trace_module) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s host=%s handle=0x%x",
                        guest_path.c_str(), path.string().c_str(), handle);
                }
#endif
                return handle;
            }
        }
    }

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_MODULE_LOAD_RESULT] path=%s handle=ENOENT",
                        guest_path.c_str());
#endif
    return ORBIS_KERNEL_ERROR_ENOENT;
}

s32 PS4_SYSV_ABI sceKernelDlsym(s32 handle, const char* symbol, void** addrp) {
#ifdef __ANDROID__
    static std::atomic_int dlsym_trace_budget{64};
    if (dlsym_trace_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_DLSYM] handle=0x%x symbol=%s", handle,
                            symbol ? symbol : "<null>");
    }
    if (addrp == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    if (IsAndroidSyntheticModuleHandle(handle)) {
        if (IsAndroidPigletModuleHandle(handle)) {
            if (void* piglet_symbol = Libraries::Piglet::ResolveSymbolByName(symbol)) {
                *addrp = piglet_symbol;
                LOG_INFO(Lib_Kernel,
                         "[EXECUTOR_MODULE_STUB] dlsym Piglet handle {:#x} symbol {} => {}",
                         handle, symbol ? symbol : "<null>", piglet_symbol);
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_MODULE_STUB] dlsym Piglet handle=0x%x symbol=%s => %p",
                                    handle, symbol ? symbol : "<null>", piglet_symbol);
                return ORBIS_OK;
            }
        }
        if (IsAndroidStoreCurlModuleHandle(handle)) {
            if (void* curl_symbol = ResolveAndroidCurlSymbol(symbol)) {
                *addrp = curl_symbol;
                LOG_INFO(Lib_Kernel,
                         "[EXECUTOR_MODULE_STUB] dlsym curl handle {:#x} symbol {} => {}",
                         handle, symbol ? symbol : "<null>", curl_symbol);
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_MODULE_STUB] dlsym curl handle=0x%x symbol=%s => %p",
                                    handle, symbol ? symbol : "<null>", curl_symbol);
                return ORBIS_OK;
            }
        }

        const auto stub = Core::AeroLib::GetAndroidX64ZeroStubSlabBase();
        if (stub == 0) {
            LOG_ERROR(Lib_Kernel,
                      "[EXECUTOR_MODULE_STUB] synthetic dlsym failed: x64 zero stub slab missing");
            return ORBIS_KERNEL_ERROR_ESRCH;
        }
        *addrp = reinterpret_cast<void*>(stub);
        LOG_INFO(Lib_Kernel, "[EXECUTOR_MODULE_STUB] dlsym handle {:#x} symbol {} => {:#x}", handle,
                 symbol ? symbol : "<null>", stub);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MODULE_STUB] dlsym handle=0x%x symbol=%s => 0x%llx",
                            handle, symbol ? symbol : "<null>",
                            static_cast<unsigned long long>(stub));
        return ORBIS_OK;
    }
#endif

    auto* linker = Common::Singleton<Core::Linker>::Instance();
    auto* module = linker->GetModule(handle);
    if (module == nullptr) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    *addrp = module->FindByName(symbol);
    if (*addrp == nullptr) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetModuleInfoForUnwind(VAddr addr, s32 flags,
                                                 OrbisModuleInfoForUnwind* info) {
    if (flags >= 3) {
        std::memset(info, 0, sizeof(OrbisModuleInfoForUnwind));
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (!info) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    if (info->st_size < sizeof(OrbisModuleInfoForUnwind)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    LOG_INFO(Lib_Kernel, "called addr = {:#x}, flags = {:#x}", addr, flags);
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    auto* module = linker->FindByAddress(addr);
    if (!module) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    const auto mod_info = module->GetModuleInfoEx();

    std::memset(info, 0, sizeof(OrbisModuleInfoForUnwind));
    info->name = mod_info.name;
    info->eh_frame_hdr_addr = mod_info.eh_frame_hdr_addr;
    info->eh_frame_addr = mod_info.eh_frame_addr;
    info->eh_frame_size = mod_info.eh_frame_size;
    info->seg0_addr = mod_info.segments[0].address;
    info->seg0_size = mod_info.segments[0].size;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetModuleInfoFromAddr(VAddr addr, s32 flags,
                                                Core::OrbisKernelModuleInfoEx* info) {
    if (flags >= 3) {
        std::memset(info, 0, sizeof(Core::OrbisKernelModuleInfoEx));
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    if (info == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }

    LOG_INFO(Lib_Kernel, "called addr = {:#x}, flags = {:#x}", addr, flags);
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    auto* module = linker->FindByAddress(addr);
    if (!module) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }

    *info = module->GetModuleInfoEx();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetModuleInfo(s32 handle, Core::OrbisKernelModuleInfo* info) {
    if (info == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    if (info->st_size != sizeof(Core::OrbisKernelModuleInfo)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto* linker = Common::Singleton<Core::Linker>::Instance();
    auto* module = linker->GetModule(handle);
    if (module == nullptr) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    *info = module->GetModuleInfo();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetModuleInfo2(s32 handle, Core::OrbisKernelModuleInfo* info) {
    if (info == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    if (info->st_size != sizeof(Core::OrbisKernelModuleInfo)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto* linker = Common::Singleton<Core::Linker>::Instance();
    auto* module = linker->GetModule(handle);
    if (module == nullptr) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    if (module->IsSystemLib()) {
        return ORBIS_KERNEL_ERROR_EPERM;
    }
    *info = module->GetModuleInfo();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetModuleInfoInternal(s32 handle, Core::OrbisKernelModuleInfoEx* info) {
    if (info == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
    if (info->st_size != sizeof(Core::OrbisKernelModuleInfoEx)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto* linker = Common::Singleton<Core::Linker>::Instance();
    auto* module = linker->GetModule(handle);
    if (module == nullptr) {
        return ORBIS_KERNEL_ERROR_ESRCH;
    }
    *info = module->GetModuleInfoEx();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetModuleList(s32* handles, u64 num_array, u64* out_count) {
    if (handles == nullptr || out_count == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }

    auto* linker = Common::Singleton<Core::Linker>::Instance();
    u64 count = 0;
    auto* module = linker->GetModule(count);
    while (module != nullptr && count < num_array) {
        handles[count] = count;
        count++;
        module = linker->GetModule(count);
    }

    if (count == num_array && module != nullptr) {
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }

    *out_count = count;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetModuleList2(s32* handles, u64 num_array, u64* out_count) {
    if (handles == nullptr || out_count == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }

    auto* linker = Common::Singleton<Core::Linker>::Instance();
    u64 id = 0;
    u64 index = 0;
    auto* module = linker->GetModule(id);
    while (module != nullptr && index < num_array) {
        if (!module->IsSystemLib()) {
            handles[index++] = id;
        }
        id++;
        module = linker->GetModule(id);
    }

    if (index == num_array && module != nullptr) {
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }

    *out_count = index;
    return ORBIS_OK;
}

u32 PS4_SYSV_ABI posix_getuid() {
    return 1;
}

s32 PS4_SYSV_ABI exit(s32 status) {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_GUEST_EXIT_HLE] status=%d pthreadExit=%d",
                        status, std::getenv("EXECUTOR_ANDROID_GUEST_EXIT_PTHREAD") ? 1 : 0);
    {
        const std::uint64_t rsp = ExecutorGetLastHleGuestRsp();
        if (std::FILE* lf = std::fopen(
                "/data/data/app.lsx4.android/files/executor-exit-trace.log", "a")) {
            std::fprintf(lf, "EXIT status=%d rsp=0x%llx callers:", status,
                         static_cast<unsigned long long>(rsp));
            std::uint64_t callers[10] = {};
            int ncallers = 0;
            if (rsp >= 0x10000) {
                int found = 0;
                std::uint64_t last = 0;
                for (std::uint64_t off = 0; off < 0x3000 && found < 32; off += 8) {
                    std::uint64_t v = 0;
                    std::memcpy(&v, reinterpret_cast<const void*>(rsp + off), sizeof(v));
                    if (v >= 0x800000000ULL && v < 0x810000000ULL) {
                        std::fprintf(lf, " 0x%llx", static_cast<unsigned long long>(v));
                        ++found;
                        if (ncallers < 10 && (last == 0 || (v > last + 16) || (last > v + 16))) {
                            callers[ncallers++] = v;
                        }
                        last = v;
                    }
                }
            }
            std::fprintf(lf, "\n");
            for (int i = 0; i < ncallers; ++i) {
                const std::uint64_t a = callers[i];
                std::uint8_t win[80] = {};
                for (int b = 0; b < 80; ++b) {
                    std::memcpy(&win[b], reinterpret_cast<const void*>(a - 48 + b), 1);
                }
                std::fprintf(lf, "CODE @0x%llx :", static_cast<unsigned long long>(a));
                for (int b = 16; b < 64; ++b) {
                    std::fprintf(lf, " %02x", win[b]);
                }
                std::fprintf(lf, "\n");
                for (int b = 0; b + 7 <= 80; ++b) {
                    if (win[b] == 0x48 && win[b + 1] == 0x8d &&
                        (win[b + 2] == 0x05 || win[b + 2] == 0x0d || win[b + 2] == 0x15 ||
                         win[b + 2] == 0x1d || win[b + 2] == 0x25 || win[b + 2] == 0x2d ||
                         win[b + 2] == 0x35 || win[b + 2] == 0x3d)) {
                        std::int32_t disp = 0;
                        std::memcpy(&disp, &win[b + 3], 4);
                        const std::uint64_t lea_va = a - 48 + b;
                        const std::uint64_t target = lea_va + 7 + static_cast<std::int64_t>(disp);
                        if (target < 0x800000000ULL || target >= 0x810000000ULL) {
                            continue;
                        }
                        char str[97] = {};
                        int printable = 0;
                        for (int k = 0; k < 96; ++k) {
                            std::uint8_t c = 0;
                            std::memcpy(&c, reinterpret_cast<const void*>(target + k), 1);
                            if (c == 0) break;
                            str[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                            ++printable;
                        }
                        if (printable >= 4) {
                            std::fprintf(lf, "  STR @0x%llx: %s\n",
                                         static_cast<unsigned long long>(target), str);
                        }
                    }
                }
            }
            struct AotProbe { const char* name; std::uint64_t va; };
            const AotProbe probes[] = {
                {"mscorlib", 0x802624628ULL},
                {"Assembly_CSharp", 0x802519708ULL},
                {"MoveTrackerAssembly", 0x80257f528ULL},
            };
            for (const auto& p : probes) {
                std::fprintf(lf, "AOTINFO %s @0x%llx :", p.name,
                             static_cast<unsigned long long>(p.va));
                std::uint64_t qw[8] = {};
                for (int q = 0; q < 8; ++q) {
                    std::memcpy(&qw[q], reinterpret_cast<const void*>(p.va + q * 8), 8);
                    std::fprintf(lf, " 0x%llx", static_cast<unsigned long long>(qw[q]));
                }
                std::fprintf(lf, "\n");
                for (int q = 0; q < 8; ++q) {
                    const std::uint64_t ptr = qw[q];
                    if (ptr < 0x800000000ULL || ptr >= 0x810000000ULL) continue;
                    char s[40] = {};
                    int pr = 0;
                    for (int k = 0; k < 39; ++k) {
                        std::uint8_t c = 0;
                        std::memcpy(&c, reinterpret_cast<const void*>(ptr + k), 1);
                        if (c == 0) break;
                        if (c < 0x20 || c >= 0x7f) { pr = 0; break; }
                        s[k] = static_cast<char>(c);
                        ++pr;
                    }
                    if (pr >= 3) {
                        std::fprintf(lf, "  q%d->str @0x%llx: %s\n", q,
                                     static_cast<unsigned long long>(ptr), s);
                    }
                }
            }
            {
                const std::uint64_t globals[] = {0x803397ca8ULL, 0x803397c90ULL, 0x803397ca0ULL};
                for (std::uint64_t g : globals) {
                    std::uint64_t v = 0;
                    std::memcpy(&v, reinterpret_cast<const void*>(g), 8);
                    std::fprintf(lf, "GLOBAL [0x%llx] = 0x%llx\n",
                                 static_cast<unsigned long long>(g),
                                 static_cast<unsigned long long>(v));
                }
                {
                    const std::uint64_t kMonoDefaults = 0x803398d98ULL;
                    std::fprintf(lf, "MONO_DEFAULTS @0x%llx:",
                                 static_cast<unsigned long long>(kMonoDefaults));
                    std::uint64_t df[24] = {};
                    for (int i = 0; i < 24; ++i) {
                        std::memcpy(&df[i], reinterpret_cast<const void*>(kMonoDefaults + i * 8), 8);
                        std::fprintf(lf, " [%d]=0x%llx", i,
                                     static_cast<unsigned long long>(df[i]));
                    }
                    std::fprintf(lf, "\n");
                    const std::uint64_t corlib = df[0];
                    if (corlib >= 0x200000000ULL && corlib < 0x810000000ULL) {
                        const std::uint64_t soff[] = {0x20, 0x28, 0x30, 0x38, 0x48};
                        for (std::uint64_t io : soff) {
                            std::uint64_t p = 0;
                            std::memcpy(&p, reinterpret_cast<const void*>(corlib + io), 8);
                            char s[64] = {};
                            if (p >= 0x200000000ULL && p < 0x810000000ULL) {
                                for (int k = 0; k < 63; ++k) {
                                    std::uint8_t c = 0;
                                    std::memcpy(&c, reinterpret_cast<const void*>(p + k), 1);
                                    if (!c) break;
                                    s[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                                }
                            }
                            std::fprintf(lf, "  CORLIB+0x%llx=0x%llx \"%s\"\n",
                                         static_cast<unsigned long long>(io),
                                         static_cast<unsigned long long>(p), s);
                        }
                        std::uint64_t aot_mod = 0, name_cache = 0;
                        std::memcpy(&aot_mod, reinterpret_cast<const void*>(corlib + 0x3b8), 8);
                        std::fprintf(lf, "  CORLIB+0x3b8(aot_module)=0x%llx\n",
                                     static_cast<unsigned long long>(aot_mod));
                        std::fprintf(lf, "  CORLIB_RAW @0x%llx:",
                                     static_cast<unsigned long long>(corlib));
                        for (int b = 0; b < 0x420; ++b) {
                            std::uint8_t c = 0;
                            std::memcpy(&c, reinterpret_cast<const void*>(corlib + b), 1);
                            std::fprintf(lf, " %02x", c);
                        }
                        std::fprintf(lf, "\n");
                        std::uint64_t htab = 0, hstr = 0;
                        std::memcpy(&htab, reinterpret_cast<const void*>(corlib + 0xa8), 8);
                        std::memcpy(&hstr, reinterpret_cast<const void*>(corlib + 0x68), 8);
                        std::fprintf(lf, "  HEAP_TABLES.data=0x%llx HEAP_STRINGS.data=0x%llx\n",
                                     static_cast<unsigned long long>(htab),
                                     static_cast<unsigned long long>(hstr));
                        if (htab >= 0x200000000ULL && htab < 0x810000000ULL) {
                            std::fprintf(lf, "  TILDE_HDR @0x%llx:",
                                         static_cast<unsigned long long>(htab));
                            for (int b = 0; b < 0xc0; ++b) {
                                std::uint8_t c = 0;
                                std::memcpy(&c, reinterpret_cast<const void*>(htab + b), 1);
                                std::fprintf(lf, " %02x", c);
                            }
                            std::fprintf(lf, "\n");
                        }
                        if (hstr >= 0x200000000ULL && hstr < 0x810000000ULL) {
                            std::fprintf(lf, "  STRINGS_HEAD @0x%llx: ",
                                         static_cast<unsigned long long>(hstr));
                            for (int b = 0; b < 0x300; ++b) {
                                std::uint8_t c = 0;
                                std::memcpy(&c, reinterpret_cast<const void*>(hstr + b), 1);
                                std::fprintf(lf, "%c",
                                             (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.');
                            }
                            std::fprintf(lf, "\n");
                        }
                    }
                }
                const std::uint64_t msgs[] = {0x80334574dULL, 0x803345796ULL, 0x803345642ULL,
                                              0x80334553eULL, 0x803345810ULL};
                for (std::uint64_t ms : msgs) {
                    char s[80] = {};
                    for (int k = 0; k < 79; ++k) {
                        std::uint8_t c = 0;
                        std::memcpy(&c, reinterpret_cast<const void*>(ms + k), 1);
                        if (c == 0) break;
                        s[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                    }
                    std::fprintf(lf, "MSG @0x%llx: %s\n", static_cast<unsigned long long>(ms), s);
                }
                {
                    const std::uint64_t gl[] = {0x803392e18ULL, 0x803392e10ULL, 0x803392e08ULL};
                    for (std::uint64_t g : gl) {
                        std::uint64_t v = 0;
                        std::memcpy(&v, reinterpret_cast<const void*>(g), 8);
                        std::fprintf(lf, "MGLOBAL [0x%llx] = 0x%llx\n",
                                     static_cast<unsigned long long>(g),
                                     static_cast<unsigned long long>(v));
                        std::uint64_t node = v;
                        for (int n = 0; n < 4 && node >= 0x200000000ULL && node < 0x810000000ULL;
                             ++n) {
                            std::uint64_t nxt = 0, fn = 0;
                            std::memcpy(&nxt, reinterpret_cast<const void*>(node), 8);
                            std::memcpy(&fn, reinterpret_cast<const void*>(node + 8), 8);
                            std::fprintf(lf, "   node@0x%llx next=0x%llx func=0x%llx\n",
                                         static_cast<unsigned long long>(node),
                                         static_cast<unsigned long long>(nxt),
                                         static_cast<unsigned long long>(fn));
                            node = nxt;
                        }
                    }
                    std::uint64_t head = 0;
                    std::memcpy(&head, reinterpret_cast<const void*>(0x803392e70ULL), 8);
                    std::fprintf(lf, "LOADED_ASMS head=0x%llx:\n",
                                 static_cast<unsigned long long>(head));
                    std::uint64_t cur = head;
                    int count = 0;
                    while (cur >= 0x200000000ULL && cur < 0x810000000ULL && count < 60) {
                        std::uint64_t asm_ptr = 0, next = 0;
                        std::memcpy(&asm_ptr, reinterpret_cast<const void*>(cur), 8);
                        std::memcpy(&next, reinterpret_cast<const void*>(cur + 8), 8);
                        char nm[48] = {};
                        if (asm_ptr >= 0x200000000ULL && asm_ptr < 0x810000000ULL) {
                            for (int o = 0; o < 0xa0; o += 8) {
                                std::uint64_t p = 0;
                                std::memcpy(&p, reinterpret_cast<const void*>(asm_ptr + o), 8);
                                if (p < 0x200000000ULL || p >= 0x810000000ULL) continue;
                                char t[40] = {};
                                int pr = 0;
                                for (int k = 0; k < 39; ++k) {
                                    std::uint8_t c = 0;
                                    std::memcpy(&c, reinterpret_cast<const void*>(p + k), 1);
                                    if (c == 0) break;
                                    if (c < 0x20 || c >= 0x7f) { pr = 0; break; }
                                    t[k] = static_cast<char>(c); ++pr;
                                }
                                if (pr >= 4 && pr < 36) { std::snprintf(nm, sizeof(nm), "+0x%x:%s", o, t); break; }
                            }
                        }
                        char anm[48] = {};
                        std::uint64_t aname_name = 0;
                        if (asm_ptr >= 0x200000000ULL && asm_ptr < 0x810000000ULL) {
                            std::memcpy(&aname_name, reinterpret_cast<const void*>(asm_ptr + 0x10), 8);
                            if (aname_name >= 0x200000000ULL && aname_name < 0x810000000ULL) {
                                for (int k = 0; k < 47; ++k) {
                                    std::uint8_t c = 0;
                                    std::memcpy(&c, reinterpret_cast<const void*>(aname_name + k), 1);
                                    if (!c) break;
                                    anm[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                                }
                            }
                        }
                        std::fprintf(lf, "  asm[%d]=0x%llx scan=%s aname.name@+0x10=0x%llx \"%s\"\n",
                                     count, static_cast<unsigned long long>(asm_ptr), nm,
                                     static_cast<unsigned long long>(aname_name), anm);
                        if (count == 0 && asm_ptr >= 0x200000000ULL && asm_ptr < 0x810000000ULL) {
                            for (int o = 0x40; o <= 0x78; o += 8) {
                                std::uint64_t img = 0;
                                std::memcpy(&img, reinterpret_cast<const void*>(asm_ptr + o), 8);
                                if (img < 0x200000000ULL || img >= 0x810000000ULL) continue;
                                std::uint64_t pathp = 0;
                                std::memcpy(&pathp, reinterpret_cast<const void*>(img + 0x20), 8);
                                if (pathp < 0x200000000ULL || pathp >= 0x810000000ULL) continue;
                                char pth[72] = {};
                                int pr = 0;
                                for (int k = 0; k < 71; ++k) {
                                    std::uint8_t c = 0;
                                    std::memcpy(&c, reinterpret_cast<const void*>(pathp + k), 1);
                                    if (!c) break;
                                    if (c < 0x20 || c >= 0x7f) { pr = 0; break; }
                                    pth[k] = static_cast<char>(c); ++pr;
                                }
                                if (pr < 6) continue;
                                std::uint64_t atab_base = 0, atab_rowsz = 0;
                                std::memcpy(&atab_base, reinterpret_cast<const void*>(img + 0xc0 + 32 * 16), 8);
                                std::memcpy(&atab_rowsz, reinterpret_cast<const void*>(img + 0xc0 + 32 * 16 + 8), 8);
                                std::fprintf(lf, "  HEAD_IMG asm+0x%x=img0x%llx path=\"%s\" "
                                             "tables[ASSEMBLY].base=0x%llx rows_rowsize=0x%llx\n",
                                             o, static_cast<unsigned long long>(img), pth,
                                             static_cast<unsigned long long>(atab_base),
                                             static_cast<unsigned long long>(atab_rowsz));
                                break;
                            }
                        }
                        cur = next; ++count;
                    }
                    std::fprintf(lf, "LOADED_ASMS count=%d\n", count);
                }
                {
                    const std::uint64_t a = 0x803241f06ULL;
                    const std::uint64_t b = 0x809ee8000ULL + 0xb1f06ULL;
                    std::uint8_t ba = 0, bb = 0;
                    std::memcpy(&ba, reinterpret_cast<const void*>(a), 1);
                    std::memcpy(&bb, reinterpret_cast<const void*>(b), 1);
                    std::uint64_t ha = 0, hb = 0;
                    std::memcpy(&ha, reinterpret_cast<const void*>(0x803190000ULL), 8);
                    std::memcpy(&hb, reinterpret_cast<const void*>(0x809ee8000ULL), 8);
                    std::fprintf(lf,
                                 "MONOCOPIES gateA@0x%llx=0x%02x gateB@0x%llx=0x%02x "
                                 "hdrA=0x%llx hdrB=0x%llx\n",
                                 (unsigned long long)a, ba, (unsigned long long)b, bb,
                                 (unsigned long long)ha, (unsigned long long)hb);
                }
                {
                    auto rd_str = [](std::uint64_t p, char* out, int n) {
                        out[0] = 0;
                        if (p < 0x200000000ULL || p >= 0x810000000ULL) return;
                        for (int k = 0; k < n - 1; ++k) {
                            std::uint8_t c = 0;
                            std::memcpy(&c, reinterpret_cast<const void*>(p + k), 1);
                            if (c == 0) { out[k] = 0; return; }
                            out[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                        }
                        out[n - 1] = 0;
                    };
                    std::uint64_t ht = 0;
                    std::memcpy(&ht, reinterpret_cast<const void*>(0x803397ca8ULL), 8);
                    if (ht >= 0x200000000ULL && ht < 0x810000000ULL) {
                        std::uint64_t tbl = 0;
                        std::uint32_t tsz = 0;
                        std::memcpy(&tbl, reinterpret_cast<const void*>(ht + 0x10), 8);
                        std::memcpy(&tsz, reinterpret_cast<const void*>(ht + 0x18), 4);
                        std::fprintf(lf, "AOTHT tbl=0x%llx size=%u\n",
                                     static_cast<unsigned long long>(tbl), tsz);
                        for (std::uint32_t b = 0; b < tsz && b < 512; ++b) {
                            std::uint64_t slot = 0;
                            std::memcpy(&slot, reinterpret_cast<const void*>(tbl + b * 8), 8);
                            int guard = 0;
                            while (slot >= 0x200000000ULL && slot < 0x810000000ULL && guard++ < 8) {
                                std::uint64_t key = 0, val = 0, next = 0;
                                std::memcpy(&key, reinterpret_cast<const void*>(slot), 8);
                                std::memcpy(&val, reinterpret_cast<const void*>(slot + 8), 8);
                                std::memcpy(&next, reinterpret_cast<const void*>(slot + 0x10), 8);
                                char kn[48];
                                rd_str(key, kn, sizeof(kn));
                                std::uint32_t ood = 0, itl = 0;
                                if (val >= 0x200000000ULL && val < 0x810000000ULL) {
                                    std::memcpy(&ood, reinterpret_cast<const void*>(val + 0x54), 4);
                                    std::memcpy(&itl, reinterpret_cast<const void*>(val + 0x50), 4);
                                }
                                std::fprintf(lf, "AOTMOD key=%s amodule=0x%llx out_of_date=%u itl=%u\n",
                                             kn, static_cast<unsigned long long>(val), ood, itl);
                                if (val >= 0x200000000ULL && val < 0x810000000ULL) {
                                    auto looks_str = [](char* s) {
                                        return s[0] >= 0x20 && s[0] < 0x7f && s[1] >= 0x20 && s[1] < 0x7f;
                                    };
                                    for (int q = 0; q < 32; ++q) {
                                        std::uint64_t f = 0;
                                        std::memcpy(&f, reinterpret_cast<const void*>(val + q * 8), 8);
                                        if (f < 0x200000000ULL || f >= 0x810000000ULL) continue;
                                        char s[48];
                                        rd_str(f, s, sizeof(s));
                                        if (looks_str(s)) {
                                            std::fprintf(lf, "  f[+0x%x]->\"%s\"\n", q * 8, s);
                                            continue;
                                        }
                                        for (int e = 0; e < 4; ++e) {
                                            std::uint64_t ep = 0;
                                            std::memcpy(&ep, reinterpret_cast<const void*>(f + e * 8), 8);
                                            if (ep < 0x200000000ULL || ep >= 0x810000000ULL) break;
                                            char es[48];
                                            rd_str(ep, es, sizeof(es));
                                            if (looks_str(es))
                                                std::fprintf(lf, "  f[+0x%x][%d]->\"%s\"\n", q * 8, e, es);
                                        }
                                    }
                                }
                                slot = next;
                            }
                        }
                    }
                }
                std::uint64_t table = 0;
                std::memcpy(&table, reinterpret_cast<const void*>(0x803397ca8ULL), 8);
                if (table != 0) {
                    std::fprintf(lf, "AOT_MODULES_TABLE @0x%llx hdr:",
                                 static_cast<unsigned long long>(table));
                    for (int k = 0; k < 64; ++k) {
                        std::uint8_t b = 0;
                        std::memcpy(&b, reinterpret_cast<const void*>(table + k), 1);
                        std::fprintf(lf, " %02x", b);
                    }
                    std::fprintf(lf, "\n");
                }
            }
            std::fclose(lf);
        }
    }
    if (std::getenv("EXECUTOR_ANDROID_GUEST_EXIT_PTHREAD") != nullptr) {
        pthread_exit(reinterpret_cast<void*>(static_cast<std::intptr_t>(status)));
    }
#endif
    UNREACHABLE_MSG("Exiting with status code {}", status);
    return 0;
}

void RegisterProcess(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("xeu-pV8wkKs", "libkernel", 1, "libkernel", sceKernelIsInSandbox);
    LIB_FUNCTION("WB66evu8bsU", "libkernel", 1, "libkernel", sceKernelGetCompiledSdkVersion);
    LIB_FUNCTION("WslcK1FQcGI", "libkernel", 1, "libkernel", sceKernelIsNeoMode);
    LIB_FUNCTION("rNRtm1uioyY", "libkernel", 1, "libkernel", sceKernelHasNeoMode);
    LIB_FUNCTION("0vTn5IDMU9A", "libkernel", 1, "libkernel", sceKernelGetMainSocId);
    LIB_FUNCTION("VOx8NGmHXTs", "libkernel", 1, "libkernel", sceKernelGetCpumode);
    LIB_FUNCTION("g0VTBxfJyu0", "libkernel", 1, "libkernel", sceKernelGetCurrentCpu);
    LIB_FUNCTION("959qrazPIrg", "libkernel", 1, "libkernel", sceKernelGetProcParam);
    LIB_FUNCTION("wzvqT4UqKX8", "libkernel", 1, "libkernel", sceKernelLoadStartModule);
    LIB_FUNCTION("LwG8g3niqwA", "libkernel", 1, "libkernel", sceKernelDlsym);
    LIB_FUNCTION("RpQJJVKTiFM", "libkernel", 1, "libkernel", sceKernelGetModuleInfoForUnwind);
    LIB_FUNCTION("f7KBOafysXo", "libkernel", 1, "libkernel", sceKernelGetModuleInfoFromAddr);
    LIB_FUNCTION("kUpgrXIrz7Q", "libkernel", 1, "libkernel", sceKernelGetModuleInfo);
    LIB_FUNCTION("QgsKEUfkqMA", "libkernel", 1, "libkernel", sceKernelGetModuleInfo2);
    LIB_FUNCTION("QgsKEUfkqMA", "libkernel_module_info", 1, "libkernel", sceKernelGetModuleInfo2);
    LIB_FUNCTION("HZO7xOos4xc", "libkernel", 1, "libkernel", sceKernelGetModuleInfoInternal);
    LIB_FUNCTION("IuxnUuXk6Bg", "libkernel", 1, "libkernel", sceKernelGetModuleList);
    LIB_FUNCTION("ZzzC3ZGVAkc", "libkernel", 1, "libkernel", sceKernelGetModuleList2);
    LIB_FUNCTION("kg4x8Prhfxw", "libkernel", 1, "libkernel", posix_getuid);
    LIB_FUNCTION("6Z83sYWFlA8", "libkernel", 1, "libkernel", exit);
}

}
