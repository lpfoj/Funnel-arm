// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "core/file_sys/fs.h"
#include "core/libraries/disc_map/disc_map.h"
#include "core/libraries/font/font.h"
#include "core/libraries/font/fontft.h"
#include "core/libraries/jpeg/jpegenc.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/libc_internal/libc_internal.h"
#include "core/libraries/libpng/pngenc.h"
#include "core/libraries/libs.h"
#include "core/libraries/ngs2/ngs2.h"
#include "core/libraries/rtc/rtc.h"
#include "core/libraries/sysmodule/sysmodule_error.h"
#include "core/libraries/sysmodule/sysmodule_internal.h"
#include "core/libraries/sysmodule/sysmodule_table.h"
#include "core/linker.h"
#include "emulator.h"

namespace Libraries::SysModule {

s32 getModuleHandle(s32 id, s32* handle) {
    if (id == 0) {
        return ORBIS_SYSMODULE_INVALID_ID;
    }
    for (OrbisSysmoduleModuleInternal mod : g_modules_array) {
        if (mod.id != id) {
            continue;
        }
        if (mod.is_loaded < 1) {
            return ORBIS_SYSMODULE_NOT_LOADED;
        }
        if (handle != nullptr) {
            *handle = mod.handle;
        }
        return ORBIS_OK;
    }
    return ORBIS_SYSMODULE_INVALID_ID;
}

bool shouldHideName(const char* module_name) {
    for (u64 i = 0; i < g_num_modules; i++) {
        OrbisSysmoduleModuleInternal mod = g_modules_array[i];
        if ((mod.flags & OrbisSysmoduleModuleInternalFlags::IsGame) == 0) {
            continue;
        }
        u64 name_length = std::strlen(mod.name);
        char name_copy[0x100];
        std::strncpy(name_copy, mod.name, sizeof(name_copy));
        std::strncpy(&name_copy[name_length], ".prx", 4);
        s32 result = std::strncmp(module_name, name_copy, sizeof(name_copy));
        if (result == 0) {
            return true;
        }

        if (i == 3) {
            result = std::strncmp(module_name, "libSceFios2.sprx", sizeof(name_copy));
        } else if (i == 4) {
            result = std::strncmp(module_name, "libc.sprx", sizeof(name_copy));
        }

        if (result == 0) {
            return true;
        }
    }
    return false;
}

bool isDebugModule(s32 id) {
    for (OrbisSysmoduleModuleInternal mod : g_modules_array) {
        if (mod.id == id && (mod.flags & OrbisSysmoduleModuleInternalFlags::IsDebug) != 0) {
            return true;
        }
    }
    return false;
}

bool validateModuleId(s32 id) {
    if ((id & 0x7fffffff) == 0) {
        return ORBIS_SYSMODULE_INVALID_ID;
    }

    s32 sdk_ver = 0;
    ASSERT_MSG(!Kernel::sceKernelGetCompiledSdkVersion(&sdk_ver),
               "Failed to retrieve compiled SDK version");

    if (id == 0xb8 && sdk_ver >= Common::ElfInfo::FW_75) {
        return ORBIS_SYSMODULE_INVALID_ID;
    }

    if (id == 0xb0 && sdk_ver >= Common::ElfInfo::FW_70) {
        return ORBIS_SYSMODULE_INVALID_ID;
    }

    if (id == 0x80 && sdk_ver >= Common::ElfInfo::FW_30) {
        return ORBIS_SYSMODULE_INVALID_ID;
    }

    if (isDebugModule(id) && !Config::isDevKitConsole()) {
        return ORBIS_SYSMODULE_INVALID_ID;
    }

    return ORBIS_OK;
}

s32 loadModuleInternal(s32 index, s32 argc, const void* argv, s32* res_out) {
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    auto* game_info = Common::Singleton<Common::ElfInfo>::Instance();

    OrbisSysmoduleModuleInternal& mod = g_modules_array[index];
    if (mod.is_loaded > 0) {
        mod.is_loaded++;
        return ORBIS_OK;
    }

    s32 start_result = 0;
    if ((mod.flags & OrbisSysmoduleModuleInternalFlags::IsGame) != 0) {
        std::string guest_path = std::string("/app0/sce_module/").append(mod.name);
        guest_path.append(".prx");
        const auto& host_path = mnt->GetHostPath(guest_path);

        s32 result = linker->LoadAndStartModule(host_path, argc, argv, &start_result);
        if (result < 0) {
            LOG_ERROR(Lib_SysModule, "Failed to load game library {}", guest_path);
            return result;
        } else {
            mod.handle = result;
            mod.is_loaded++;
        }
    } else {
        std::string mod_name = std::string(mod.name);

        if (index == 0xd && Config::isDevKitConsole()) {
            mod_name.append("_padebug");
        }

        if (index == 0x27 && false) {
            mod_name.clear();
            mod_name.append(g_modules_array[0x15].name);
        }

        if (index == 0xb3 && Config::isDevKitConsole()) {
            mod_name.append("_debug");
        }

        if ((mod.flags & OrbisSysmoduleModuleInternalFlags::IsNeo) == 0 &&
            (mod.flags & OrbisSysmoduleModuleInternalFlags::IsNeoMode) != 0 &&
            Kernel::sceKernelIsNeoMode() == 1) {
            mod_name.append("ForNeoMode");
        } else if ((mod.flags & OrbisSysmoduleModuleInternalFlags::IsNeo) != 0 &&
                   Config::isNeoModeConsole()) {
            mod_name.append("ForNeo");
        }

        mod_name.append(".sprx");

        const auto& sys_module_path = Config::getSysModulesPath();
        const auto& game_specific_module_path =
            sys_module_path / game_info->GameSerial() / mod_name;
        if (std::filesystem::exists(game_specific_module_path)) {
            LOG_INFO(Loader, "Loading {} from game serial file {}", mod_name,
                     game_info->GameSerial());
            s32 handle =
                linker->LoadAndStartModule(game_specific_module_path, argc, argv, &start_result);
            ASSERT_MSG(handle >= 0, "Failed to load module {}", mod_name);
            mod.handle = handle;
            mod.is_loaded++;
            if (res_out != nullptr) {
                *res_out = start_result;
            }
            return ORBIS_OK;
        }

        static s32 stub_handle = 100;
        constexpr auto ModulesToLoad = std::to_array<Core::SysModules>(
            {{"libSceNgs2.sprx", &Libraries::Ngs2::RegisterLib},
             {"libSceUlt.sprx", nullptr},
             {"libSceRtc.sprx", &Libraries::Rtc::RegisterLib},
             {"libSceJpegDec.sprx", nullptr},
             {"libSceJpegEnc.sprx", &Libraries::JpegEnc::RegisterLib},
             {"libScePngEnc.sprx", &Libraries::PngEnc::RegisterLib},
             {"libSceJson.sprx", nullptr},
             {"libSceJson2.sprx", nullptr},
             {"libSceLibcInternal.sprx", &Libraries::LibcInternal::RegisterLib},
             {"libSceCesCs.sprx", nullptr},
             {"libSceAudiodec.sprx", nullptr},
             {"libSceFont.sprx", &Libraries::Font::RegisterlibSceFont},
             {"libSceFontFt.sprx", &Libraries::FontFt::RegisterlibSceFontFt},
             {"libSceFreeTypeOt.sprx", nullptr}});

        const auto it = std::ranges::find_if(
            ModulesToLoad, [&](Core::SysModules module) { return mod_name == module.module_name; });
        if (it == ModulesToLoad.end()) {
            mod.is_loaded++;
            mod.handle = stub_handle++;
            if (res_out != nullptr) {
                *res_out = ORBIS_OK;
            }
            return ORBIS_OK;
        }

        const auto& module_path = sys_module_path / mod_name;
        if (std::filesystem::exists(module_path)) {
            LOG_INFO(Loader, "Loading {}", mod_name);
            s32 handle = linker->LoadAndStartModule(module_path, argc, argv, &start_result);
            ASSERT_MSG(handle >= 0, "Failed to load module {}", mod_name);
            mod.handle = handle;
        } else {
            auto& [name, init_func] = *it;
            if (init_func) {
                LOG_INFO(Loader, "Can't Load {} switching to HLE", mod_name);
                init_func(&linker->GetHLESymbols());

                linker->RelocateAllImports();
            } else {
                LOG_INFO(Loader, "No HLE available for {} module", mod_name);
            }
            mod.handle = stub_handle++;
        }

        mod.is_loaded++;
    }

    if (res_out != nullptr) {
        *res_out = start_result;
    }

    return ORBIS_OK;
}

s32 loadModule(s32 id, s32 argc, const void* argv, s32* res_out) {
    OrbisSysmoduleModuleInternal requested_module{};
    for (OrbisSysmoduleModuleInternal mod : g_modules_array) {
        if (mod.id == id) {
            requested_module = mod;
            break;
        }
    }
    if (requested_module.id != id || requested_module.id == 0) {
        return ORBIS_SYSMODULE_INVALID_ID;
    }

    if (requested_module.to_load == nullptr) {
        return ORBIS_SYSMODULE_LOCK_FAILED;
    }

    LOG_INFO(Lib_SysModule, "Loading {}", requested_module.name);

    for (s64 i = requested_module.num_to_load - 1; i >= 0; i--) {
        u32 mod_index = requested_module.to_load[i];
        if ((!Config::isDevKitConsole() &&
             g_modules_array[mod_index].flags & OrbisSysmoduleModuleInternalFlags::IsDebug) != 0) {
            continue;
        }

        s32 result = 0;
        if (i != 0) {
            result = loadModuleInternal(mod_index, 0, nullptr, nullptr);
        } else {
            result = loadModuleInternal(mod_index, argc, argv, res_out);
        }

        if (result != ORBIS_OK) {
            return result;
        }
    }
    return ORBIS_OK;
}

s32 unloadModule(s32 id, s32 argc, const void* argv, s32* res_out, bool is_internal) {
    OrbisSysmoduleModuleInternal mod{};
    for (s32 i = 0; i < g_modules_array.size(); i++) {
        mod = g_modules_array[i];
        if (mod.id != id) {
            continue;
        }

        if (i == 0x22) {
            continue;
        }

        for (s32 index : g_preload_list_2) {
            if (index == i && mod.is_loaded == 1) {
                return ORBIS_OK;
            }
        }

        break;
    }

    if (mod.id != id || mod.id == 0) {
        return ORBIS_SYSMODULE_INVALID_ID;
    }

    if (mod.num_to_load == 0 || mod.to_load == nullptr) {
        return ORBIS_SYSMODULE_LOCK_FAILED;
    }

    for (s64 i = 0; i < mod.num_to_load; i++) {
        OrbisSysmoduleModuleInternal dep_mod = g_modules_array[mod.to_load[i]];
        if ((dep_mod.flags & OrbisSysmoduleModuleInternalFlags::IsDebug) != 0 &&
            !Config::isDevKitConsole()) {
            continue;
        }

        if (dep_mod.is_loaded == 0) {
            return ORBIS_SYSMODULE_NOT_LOADED;
        }

        dep_mod.is_loaded--;


        if (i == 0 && res_out != nullptr) {
            *res_out = ORBIS_OK;
        }
    }
    return ORBIS_OK;
}

s32 preloadModulesForLibkernel() {
    s32 sdk_ver = 0;
    ASSERT_MSG(Kernel::sceKernelGetCompiledSdkVersion(&sdk_ver) == 0,
               "Failed to get compiled SDK version");
    for (u32 module_index : g_preload_list_3) {
        if ((module_index == 0x12 || module_index == 0x1e || module_index == 0x24 ||
             module_index == 0x26) &&
            !Config::isDevKitConsole()) {
            continue;
        }

        if (module_index == 0x22 && sdk_ver >= Common::ElfInfo::FW_20) {
            continue;
        }

        if (module_index == 0x23 && !Config::isDevKitConsole()) {
            continue;
        }

        if (module_index == 0x25 && sdk_ver < Common::ElfInfo::FW_45 &&
            !Config::isDevKitConsole()) {
            continue;
        }

        if (module_index == 0x28 && sdk_ver < Common::ElfInfo::FW_70) {
            continue;
        }

        if ((module_index == 0x29 || module_index == 0x2a) && sdk_ver < Common::ElfInfo::FW_75) {
            continue;
        }

        s32 result = loadModuleInternal(module_index, 0, nullptr, nullptr);
        if (result != ORBIS_OK) {
            LOG_CRITICAL(Lib_SysModule, "Failed to preload {}, expect crashes",
                         g_modules_array[module_index].name);
        }
    }
    return ORBIS_OK;
}

}
