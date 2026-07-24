// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <string_view>

#include "assert.h"
#include "bit_field.h"
#include "singleton.h"
#include "types.h"

namespace Core {
class Emulator;
}

namespace Common {

union PSFAttributes {
    BitField<0, 1, u32> support_initial_user_logout;
    BitField<1, 1, u32> enter_button_cross;
    BitField<2, 1, u32> ps_move_warning;
    BitField<3, 1, u32> support_stereoscopic_3d;
    BitField<4, 1, u32> ps_button_suspend;
    BitField<5, 1, u32> enter_button_system;
    BitField<6, 1, u32> override_share_menu;
    BitField<8, 1, u32> special_res_ps_button_suspend;
    BitField<9, 1, u32> enable_hdcp;
    BitField<10, 1, u32> disable_hdcp_non_game;
    BitField<14, 1, u32> support_ps_vr;
    BitField<15, 1, u32> six_cpu_mode;
    BitField<16, 1, u32> seven_cpu_mode;
    BitField<23, 1, u32> support_neo_mode;
    BitField<26, 1, u32> require_ps_vr;
    BitField<29, 1, u32> support_hdr;
    BitField<31, 1, u32> display_location;

    u32 raw{};
};
static_assert(sizeof(PSFAttributes) == 4);

class ElfInfo {
    friend class Core::Emulator;

    bool initialized = false;

    std::string game_serial{};
    std::string title{};
    std::string app_ver{};
    u32 firmware_ver = 0;
    u32 raw_firmware_ver = 0;
    u32 sdk_ver = 0;
    PSFAttributes psf_attributes{};

    std::filesystem::path splash_path{};
    std::filesystem::path game_folder{};

public:
    static constexpr u32 FW_10 = 0x1000000;
    static constexpr u32 FW_15 = 0x1500000;
    static constexpr u32 FW_16 = 0x1600000;
    static constexpr u32 FW_17 = 0x1700000;
    static constexpr u32 FW_20 = 0x2000000;
    static constexpr u32 FW_25 = 0x2500000;
    static constexpr u32 FW_30 = 0x3000000;
    static constexpr u32 FW_35 = 0x3500000;
    static constexpr u32 FW_40 = 0x4000000;
    static constexpr u32 FW_45 = 0x4500000;
    static constexpr u32 FW_50 = 0x5000000;
    static constexpr u32 FW_55 = 0x5500000;
    static constexpr u32 FW_60 = 0x6000000;
    static constexpr u32 FW_70 = 0x7000000;
    static constexpr u32 FW_75 = 0x7500000;
    static constexpr u32 FW_80 = 0x8000000;
    static constexpr u32 FW_115 = 0x11500000;

    static ElfInfo& Instance() {
        return *Singleton<ElfInfo>::Instance();
    }

#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
    void InitializeForAndroidProbe(const std::filesystem::path& folder, u32 compiled_sdk_ver) {
        initialized = true;
        game_serial = "EXECANDROID";
        title = "Executor Android native probe";
        app_ver = "0.0";
        firmware_ver = FW_45;
        raw_firmware_ver = FW_45;
        sdk_ver = compiled_sdk_ver != 0 ? compiled_sdk_ver : FW_45;
        psf_attributes.raw = 0;
        splash_path.clear();
        game_folder = folder;
    }

    void InitializeForAndroidTitle(const std::filesystem::path& folder, std::string serial,
                                   std::string title_name, std::string app_version,
                                   u32 firmware, u32 raw_firmware, u32 compiled_sdk_ver,
                                   PSFAttributes attributes,
                                   std::filesystem::path splash = {}) {
        initialized = true;
        game_serial = std::move(serial);
        title = std::move(title_name);
        app_ver = std::move(app_version);
        firmware_ver = firmware;
        raw_firmware_ver = raw_firmware;
        sdk_ver = compiled_sdk_ver != 0 ? compiled_sdk_ver : firmware;
        psf_attributes = attributes;
        splash_path = std::move(splash);
        game_folder = folder;
    }
#endif

    [[nodiscard]] std::string_view GameSerial() const {
        ASSERT(initialized);
        return Instance().game_serial;
    }

    [[nodiscard]] std::string_view Title() const {
        ASSERT(initialized);
        return title;
    }

    [[nodiscard]] std::string_view AppVer() const {
        ASSERT(initialized);
        return app_ver;
    }

    [[nodiscard]] u32 FirmwareVer() const {
        ASSERT(initialized);
        return firmware_ver;
    }

    [[nodiscard]] u32 RawFirmwareVer() const {
        ASSERT(initialized);
        return raw_firmware_ver;
    }

    [[nodiscard]] u32 CompiledSdkVer() const {
        ASSERT(initialized);
        return sdk_ver;
    }

    [[nodiscard]] const PSFAttributes& GetPSFAttributes() const {
        ASSERT(initialized);
        return psf_attributes;
    }

    [[nodiscard]] const std::filesystem::path& GetSplashPath() const {
        return splash_path;
    }

    [[nodiscard]] const std::filesystem::path& GetGameFolder() const {
        return game_folder;
    }
};

}
