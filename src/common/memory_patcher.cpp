// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <codecvt>
#include <cstdio>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>
#ifdef __ANDROID__
#include <android/log.h>
#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_state.h"
#include "core/file_format/psf.h"
#include "memory_patcher.h"

namespace MemoryPatcher {

EXPORT uintptr_t g_eboot_address;
uint64_t g_eboot_image_size;
std::string g_game_serial;
std::string patch_file;
bool patches_applied = false;
std::vector<patchInfo> pending_patches;

#ifdef __ANDROID__
static bool ExecutorWriteSimplePatch(void* destination,
                                     const std::vector<unsigned char>& bytes);
#endif

namespace {

#ifdef __ANDROID__
struct EffectFlagSite {
    uintptr_t guest_address;
    const char* name;
};

struct EffectToggleSpec {
    std::size_t flag_index;
    const char* name;
    const char* marker;
};

enum class PatchValueSource : u8 {
    Literal,
    TargetWidthU32,
    TargetHeightU32,
    TargetWidthMovEax,
    TargetHeightMovEax,
    TargetHeightMovEcx,
};

struct PatchWriteSpec {
    const char* address;
    const char* value;
    bool little_endian;
    PatchValueSource value_source = PatchValueSource::Literal;
};

using ResolutionGuardSequence = std::array<u8, 9>;

struct ResolutionGuardSpec {
    uintptr_t guest_address;
    ResolutionGuardSequence expected;
};

struct TitlePatchProfile {
    std::string_view serial;
    std::string_view version;
    uintptr_t guest_image_base;
    std::span<const EffectFlagSite> effect_flag_sites;
    std::span<const EffectToggleSpec> effect_toggles;
    uintptr_t dof_branch_site;
    std::span<const u8> dof_disabled_code;
    std::string_view dof_marker;
    uintptr_t chromatic_aberration_site;
    std::span<const u8> chromatic_aberration_original_code;
    std::span<const u8> chromatic_aberration_disabled_code;
    std::string_view chromatic_aberration_marker;
    std::span<const PatchWriteSpec> resolution_writes;
    std::span<const ResolutionGuardSpec> resolution_guards;
    uintptr_t width_site;
    uintptr_t height_site;
    std::array<uintptr_t, 2> internal_width_sites;
    std::array<uintptr_t, 2> internal_height_sites;
    u32 native_width;
    u32 native_height;
};

constexpr std::array<EffectFlagSite, 4> kVerifiedEffectFlagSites{{
    {0x026C2538, "dynamic_light_shadows"},
    {0x026C2548, "ssao"},
    {0x026C2549, "motion_blur"},
    {0x026C254A, "anti_aliasing"},
}};

constexpr std::array kVerifiedEffectToggles{
    EffectToggleSpec{0, "dynamic_light_shadows", "run-disable-dynamic-shadows"},
    EffectToggleSpec{1, "ssao", "run-disable-ssao"},
    EffectToggleSpec{2, "motion_blur", "run-disable-motion-blur"},
    EffectToggleSpec{3, "anti_aliasing", "run-disable-anti-aliasing"},
};

constexpr uintptr_t kVerifiedDofBranchSite = 0x025D7BBC;
constexpr std::array<u8, 6> kVerifiedDofDisabledCode{0xE9, 0xDF, 0x00, 0x00, 0x00, 0x90};
constexpr std::string_view kDisableDofMarker = "run-disable-dof";

constexpr uintptr_t kVerifiedChromaticAberrationSite = 0x0269FAA8;
constexpr std::array<u8, 12> kVerifiedChromaticAberrationOriginalCode{
    0x8B, 0x85, 0x90, 0xF5, 0xFF, 0xFF, 0x89, 0x83, 0xAC, 0x00, 0x00, 0x00};
constexpr std::array<u8, 12> kVerifiedChromaticAberrationDisabledCode{
    0xC7, 0x83, 0xAC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x90};
constexpr std::string_view kDisableChromaticAberrationMarker =
    "run-disable-chromatic-aberration";

bool ExecutorRuntimeFlagExists(const std::filesystem::path& user_dir,
                               const std::string_view marker) {
    const std::filesystem::path marker_path{marker};
    return std::filesystem::exists(user_dir / marker_path) ||
           std::filesystem::exists(user_dir.parent_path() / marker_path);
}

constexpr std::array kVerifiedResolutionWrites{
    PatchWriteSpec{"0x055289f8", "00000280", true, PatchValueSource::TargetWidthU32},
    PatchWriteSpec{"0x055289fc", "00000168", true, PatchValueSource::TargetHeightU32},
    PatchWriteSpec{"0x0183A35D", "398EE33F", false},
    PatchWriteSpec{"0x02196A6B", "000280B8", true, PatchValueSource::TargetWidthMovEax},
    PatchWriteSpec{"0x02196A6F", "401F0F00", true},
    PatchWriteSpec{"0x02196A73", "00", false},
    PatchWriteSpec{"0x02358554", "000280B8", true, PatchValueSource::TargetWidthMovEax},
    PatchWriteSpec{"0x02358558", "401F0F00", true},
    PatchWriteSpec{"0x0235855C", "00", false},
    PatchWriteSpec{"0x02196A7A", "000168B8", true, PatchValueSource::TargetHeightMovEax},
    PatchWriteSpec{"0x02196A7E", "401F0F00", true},
    PatchWriteSpec{"0x02196A82", "00", false},
    PatchWriteSpec{"0x0235855D", "000168B9", true, PatchValueSource::TargetHeightMovEcx},
    PatchWriteSpec{"0x02358561", "401F0F00", true},
    PatchWriteSpec{"0x02358565", "00", false},
    PatchWriteSpec{"0x019E83AF", "000780B8", true},
    PatchWriteSpec{"0x019E83B3", "401F0F00", true},
    PatchWriteSpec{"0x019E83B7", "00", false},
    PatchWriteSpec{"0x01FFD491", "000780B8", true},
    PatchWriteSpec{"0x01FFD495", "401F0F00", true},
    PatchWriteSpec{"0x01FFD499", "00", false},
    PatchWriteSpec{"0x01FFD4D1", "000438B8", true},
    PatchWriteSpec{"0x01FFD4D5", "401F0F00", true},
    PatchWriteSpec{"0x01FFD4D9", "00", false},
    PatchWriteSpec{"0x01A44357", "000780B8", true},
    PatchWriteSpec{"0x01A4435B", "401F0F00", true},
    PatchWriteSpec{"0x01A4435F", "00", false},
    PatchWriteSpec{"0x01A44C55", "000780B8", true},
    PatchWriteSpec{"0x01A44C59", "401F0F00", true},
    PatchWriteSpec{"0x01A44C5D", "00", false},
    PatchWriteSpec{"0x01A452C7", "000780B8", true},
    PatchWriteSpec{"0x01A452CB", "401F0F00", true},
    PatchWriteSpec{"0x01A452CF", "00", false},
    PatchWriteSpec{"0x01A44365", "000438B8", true},
    PatchWriteSpec{"0x01A44369", "401F0F00", true},
    PatchWriteSpec{"0x01A4436D", "00", false},
    PatchWriteSpec{"0x01A44C63", "000438B8", true},
    PatchWriteSpec{"0x01A44C67", "401F0F00", true},
    PatchWriteSpec{"0x01A44C6B", "00", false},
    PatchWriteSpec{"0x01A452D5", "000438B8", true},
    PatchWriteSpec{"0x01A452D9", "401F0F00", true},
    PatchWriteSpec{"0x01A452DD", "00", false},
    PatchWriteSpec{"0x0212C674", "000780BE", true},
    PatchWriteSpec{"0x0212C678", "401F0F00", true},
    PatchWriteSpec{"0x0212C67C", "00", false},
    PatchWriteSpec{"0x0241848F", "000780B8", true},
    PatchWriteSpec{"0x02418493", "401F0F00", true},
    PatchWriteSpec{"0x02418497", "00", false},
    PatchWriteSpec{"0x02438BB9", "000780B8", true},
    PatchWriteSpec{"0x02438BBD", "401F0F00", true},
    PatchWriteSpec{"0x02438BC1", "00", false},
    PatchWriteSpec{"0x02438F59", "000780B8", true},
    PatchWriteSpec{"0x02438F5D", "401F0F00", true},
    PatchWriteSpec{"0x02438F61", "00", false},
    PatchWriteSpec{"0x0212C6A4", "000438BE", true},
    PatchWriteSpec{"0x0212C6A8", "401F0F00", true},
    PatchWriteSpec{"0x0212C6AC", "00", false},
    PatchWriteSpec{"0x02418498", "000438B9", true},
    PatchWriteSpec{"0x0241849C", "401F0F00", true},
    PatchWriteSpec{"0x024184A0", "00", false},
    PatchWriteSpec{"0x02438BC7", "000438B8", true},
    PatchWriteSpec{"0x02438BCB", "401F0F00", true},
    PatchWriteSpec{"0x02438BCF", "00", false},
    PatchWriteSpec{"0x02438F67", "000438B8", true},
    PatchWriteSpec{"0x02438F6B", "401F0F00", true},
    PatchWriteSpec{"0x02438F6F", "00", false},
};

constexpr std::array<ResolutionGuardSpec, 4> kVerifiedResolutionGuards{{
        {0x02196A6B, {0x48, 0x8D, 0x05, 0x86, 0x1F, 0x39, 0x03, 0x8B, 0x00}},
        {0x02196A7A, {0x48, 0x8D, 0x05, 0x7B, 0x1F, 0x39, 0x03, 0x8B, 0x00}},
        {0x02358554, {0x48, 0x8D, 0x05, 0x9D, 0x04, 0x1D, 0x03, 0x8B, 0x00}},
        {0x0235855D, {0x48, 0x8D, 0x0D, 0x98, 0x04, 0x1D, 0x03, 0x8B, 0x09}},
    }};

constexpr TitlePatchProfile kVerifiedRenderProfile{
    .serial = "CUSA03173",
    .version = "01.09",
    .guest_image_base = 0x00400000,
    .effect_flag_sites = std::span<const EffectFlagSite>{kVerifiedEffectFlagSites},
    .effect_toggles = std::span<const EffectToggleSpec>{kVerifiedEffectToggles},
    .dof_branch_site = kVerifiedDofBranchSite,
    .dof_disabled_code = std::span<const u8>{kVerifiedDofDisabledCode},
    .dof_marker = kDisableDofMarker,
    .chromatic_aberration_site = kVerifiedChromaticAberrationSite,
    .chromatic_aberration_original_code =
        std::span<const u8>{kVerifiedChromaticAberrationOriginalCode},
    .chromatic_aberration_disabled_code =
        std::span<const u8>{kVerifiedChromaticAberrationDisabledCode},
    .chromatic_aberration_marker = kDisableChromaticAberrationMarker,
    .resolution_writes = std::span<const PatchWriteSpec>{kVerifiedResolutionWrites},
    .resolution_guards = std::span<const ResolutionGuardSpec>{kVerifiedResolutionGuards},
    .width_site = 0x055289F8,
    .height_site = 0x055289FC,
    .internal_width_sites = {0x02196A6C, 0x02358555},
    .internal_height_sites = {0x02196A7B, 0x0235855E},
    .native_width = 1920,
    .native_height = 1080,
};

constexpr std::array kVerifiedRenderProfiles{&kVerifiedRenderProfile};

const TitlePatchProfile* FindVerifiedRenderProfile(const std::string_view serial) {
    const auto it = std::find_if(kVerifiedRenderProfiles.begin(), kVerifiedRenderProfiles.end(),
                                 [serial](const TitlePatchProfile* profile) {
                                     return profile->serial == serial;
                                 });
    return it != kVerifiedRenderProfiles.end() ? *it : nullptr;
}

std::string FormatResolutionPatchValue(const PatchWriteSpec& line,
                                       const u32 target_width,
                                       const u32 target_height) {
    char value[9]{};
    switch (line.value_source) {
    case PatchValueSource::TargetWidthU32:
        std::snprintf(value, sizeof(value), "%08X",
                      static_cast<unsigned int>(target_width));
        return value;
    case PatchValueSource::TargetHeightU32:
        std::snprintf(value, sizeof(value), "%08X",
                      static_cast<unsigned int>(target_height));
        return value;
    case PatchValueSource::TargetWidthMovEax:
        std::snprintf(value, sizeof(value), "%06XB8",
                      static_cast<unsigned int>(target_width));
        return value;
    case PatchValueSource::TargetHeightMovEax:
        std::snprintf(value, sizeof(value), "%06XB8",
                      static_cast<unsigned int>(target_height));
        return value;
    case PatchValueSource::TargetHeightMovEcx:
        std::snprintf(value, sizeof(value), "%06XB9",
                      static_cast<unsigned int>(target_height));
        return value;
    case PatchValueSource::Literal:
        return line.value;
    }
    return line.value;
}

void ApplyConfiguredTitleEffectOverrides() {
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MEMORY_PATCHER_GAME] serial=%s imageSize=0x%llx",
                        g_game_serial.c_str(),
                        static_cast<unsigned long long>(g_eboot_image_size));

    const auto user_dir = Common::FS::GetUserPath(Common::FS::PathType::UserDir);
    const TitlePatchProfile* profile = FindVerifiedRenderProfile(g_game_serial);
    const TitlePatchProfile& marker_profile =
        profile != nullptr ? *profile : kVerifiedRenderProfile;
    const bool bundle_requested =
        ExecutorRuntimeFlagExists(user_dir, "run-disable-render-effects-bundle");
    std::vector<u8> independent_requested(marker_profile.effect_flag_sites.size(), 0);
    bool any_independent_requested = false;
    for (const EffectToggleSpec& toggle : marker_profile.effect_toggles) {
        const bool requested = ExecutorRuntimeFlagExists(user_dir, toggle.marker);
        if (toggle.flag_index < independent_requested.size()) {
            independent_requested[toggle.flag_index] = requested ? 1 : 0;
        }
        any_independent_requested |= requested;
    }
    const bool dof_requested =
        ExecutorRuntimeFlagExists(user_dir, marker_profile.dof_marker);
    const bool chromatic_aberration_requested =
        ExecutorRuntimeFlagExists(user_dir, marker_profile.chromatic_aberration_marker);

    auto* param_sfo = Common::Singleton<PSF>::Instance();
    const std::string game_version{
        param_sfo->GetString("VERSION").value_or(std::string_view{})};
    const std::string profile_serial =
        profile != nullptr ? std::string{profile->serial} : g_game_serial;

    const auto log_option = [&](const char* name, const char* source, const bool requested,
                                const bool supported, const bool layout_valid, const bool applied,
                                const char* reason) {
        LOG_INFO(Loader,
                 "EXECUTOR_TITLE_PROFILE_EFFECT_OPTION serial={} version={} name={} source={} "
                 "requested={} supported={} layoutValid={} applied={} reason={}",
                 profile_serial, game_version, name, source, requested, supported, layout_valid,
                 applied, reason);
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_TITLE_PROFILE_EFFECT_OPTION] serial=%s version=%s name=%s source=%s "
            "requested=%u supported=%u layoutValid=%u applied=%u reason=%s",
            profile_serial.c_str(), game_version.c_str(), name, source, requested ? 1u : 0u,
            supported ? 1u : 0u, layout_valid ? 1u : 0u, applied ? 1u : 0u, reason);
    };

    if (profile == nullptr) {
        if (bundle_requested || any_independent_requested || dof_requested ||
            chromatic_aberration_requested) {
            for (const EffectToggleSpec& toggle : marker_profile.effect_toggles) {
                if (toggle.flag_index < independent_requested.size() &&
                    independent_requested[toggle.flag_index]) {
                    log_option(toggle.name, "independent",
                               true, false, false, false, "no_verified_title_profile");
                }
            }
            if (dof_requested) {
                log_option("depth_of_field", "independent", true, false, false, false,
                           "no_verified_title_profile");
            }
            if (chromatic_aberration_requested) {
                log_option("chromatic_aberration", "independent", true, false, false, false,
                           "no_verified_title_profile");
            }
            if (bundle_requested) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_TITLE_PROFILE_EFFECT_BUNDLE] serial=%s version=%s "
                    "result=ignored reason=title_mismatch",
                    profile_serial.c_str(), game_version.c_str());
            }
        }
        return;
    }

    const bool version_valid = game_version == profile->version;
    const auto range_in_image = [profile](const uintptr_t guest_address,
                                          const std::size_t size) {
        if (guest_address < profile->guest_image_base) {
            return false;
        }
        const uintptr_t offset = guest_address - profile->guest_image_base;
        return offset <= g_eboot_image_size && size <= g_eboot_image_size - offset;
    };
    std::vector<u8> values(profile->effect_flag_sites.size(), 0);
    std::vector<u8> flag_layout_valid(profile->effect_flag_sites.size(), 0);
    for (std::size_t index = 0; index < profile->effect_flag_sites.size(); ++index) {
        const auto guest_address = profile->effect_flag_sites[index].guest_address;
        if (!range_in_image(guest_address, sizeof(u8))) {
            continue;
        }
        const uintptr_t image_offset = guest_address - profile->guest_image_base;
        values[index] = *reinterpret_cast<const u8*>(g_eboot_address + image_offset);
        flag_layout_valid[index] = version_valid && values[index] <= 1 ? 1 : 0;
    }
    const bool bundle_layout_valid =
        std::all_of(flag_layout_valid.begin(), flag_layout_valid.end(),
                    [](const u8 valid) { return valid != 0; });

    LOG_INFO(Loader,
             "EXECUTOR_TITLE_PROFILE_EFFECT_LAYOUT serial={} version={} requested={} valid={} "
             "imageSize={:#x} flags={}",
             profile->serial, game_version,
             bundle_requested || any_independent_requested || dof_requested ||
                 chromatic_aberration_requested,
             bundle_layout_valid, g_eboot_image_size, values.size());
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_TITLE_PROFILE_EFFECT_LAYOUT] serial=%s version=%s requested=%u valid=%u "
        "imageSize=0x%llx flags=%zu",
        profile_serial.c_str(), game_version.c_str(),
        (bundle_requested || any_independent_requested || dof_requested ||
         chromatic_aberration_requested)
            ? 1u
            : 0u,
        bundle_layout_valid ? 1u : 0u,
        static_cast<unsigned long long>(g_eboot_image_size), values.size());

    bool bundle_all_applied = true;
    std::size_t bundle_applied_count = 0;
    for (std::size_t index = 0; index < profile->effect_flag_sites.size(); ++index) {
        bool individually_requested = false;
        if (index < independent_requested.size()) {
            individually_requested = independent_requested[index] != 0;
        }
        const bool requested = bundle_requested || individually_requested;
        const char* source = individually_requested
                                 ? (bundle_requested ? "independent+bundle" : "independent")
                                 : (bundle_requested ? "bundle" : "none");
        const auto& site = profile->effect_flag_sites[index];

        if (!requested) {
            continue;
        }
        if (!flag_layout_valid[index]) {
            log_option(site.name, source, true, true, false, false,
                       version_valid ? "live_byte_not_boolean" : "app_version_mismatch");
            if (bundle_requested) {
                bundle_all_applied = false;
            }
            continue;
        }

        bool applied = values[index] == 0;
        const char* reason = applied ? "already_disabled" : "write_failed";
        if (!applied) {
            auto* flag = reinterpret_cast<u8*>(
                g_eboot_address + (site.guest_address - profile->guest_image_base));
            applied = ExecutorWriteSimplePatch(flag, std::vector<unsigned char>{0});
            reason = applied ? "patched" : "write_failed";
        }
        log_option(site.name, source, true, true, true, applied, reason);
        if (applied) {
            LOG_INFO(Loader,
                     "EXECUTOR_TITLE_PROFILE_EFFECT_WRITE serial={} version={} name={} "
                     "address={:#x} value=0",
                     profile->serial, game_version, site.name, site.guest_address);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_TITLE_PROFILE_EFFECT_WRITE] serial=%s version=%s name=%s "
                "address=0x%llx value=0",
                profile_serial.c_str(), game_version.c_str(), site.name,
                static_cast<unsigned long long>(site.guest_address));
        }
        if (bundle_requested) {
            bundle_all_applied &= applied;
            bundle_applied_count += applied ? 1 : 0;
        }
    }

    if (dof_requested) {
        const uintptr_t dof_offset = profile->dof_branch_site - profile->guest_image_base;
        bool range_valid =
            profile->dof_disabled_code.size() == 6 &&
            range_in_image(profile->dof_branch_site, profile->dof_disabled_code.size());
        bool already_disabled = false;
        bool original_branch_valid = false;
        if (range_valid) {
            const auto* current =
                reinterpret_cast<const u8*>(g_eboot_address + dof_offset);
            already_disabled =
                std::equal(profile->dof_disabled_code.begin(), profile->dof_disabled_code.end(),
                           current);
            original_branch_valid =
                current[0] == 0x0F && (current[1] == 0x84 || current[1] == 0x85) &&
                current[2] == 0xDE && current[3] == 0x00 && current[4] == 0x00 &&
                current[5] == 0x00;
        }
        const bool layout_valid =
            version_valid && range_valid && (already_disabled || original_branch_valid);
        bool applied = already_disabled;
        const char* reason = already_disabled ? "already_disabled" : "live_opcode_mismatch";
        if (layout_valid && !already_disabled) {
            auto* destination = reinterpret_cast<u8*>(g_eboot_address + dof_offset);
            applied = ExecutorWriteSimplePatch(
                destination, std::vector<unsigned char>(profile->dof_disabled_code.begin(),
                                                         profile->dof_disabled_code.end()));
            reason = applied ? "patched" : "write_failed";
        } else if (!version_valid) {
            reason = "app_version_mismatch";
        } else if (!range_valid) {
            reason = "address_out_of_image";
        }
        log_option("depth_of_field", "independent", true, true, layout_valid, applied, reason);
    }

    if (chromatic_aberration_requested) {
        const uintptr_t chromatic_aberration_offset =
            profile->chromatic_aberration_site - profile->guest_image_base;
        const bool range_valid =
            profile->chromatic_aberration_original_code.size() ==
                profile->chromatic_aberration_disabled_code.size() &&
            range_in_image(profile->chromatic_aberration_site,
                           profile->chromatic_aberration_disabled_code.size());
        bool already_disabled = false;
        bool original_sequence_valid = false;
        if (range_valid) {
            const auto* current = reinterpret_cast<const u8*>(
                g_eboot_address + chromatic_aberration_offset);
            already_disabled =
                std::equal(profile->chromatic_aberration_disabled_code.begin(),
                           profile->chromatic_aberration_disabled_code.end(), current);
            original_sequence_valid =
                std::equal(profile->chromatic_aberration_original_code.begin(),
                           profile->chromatic_aberration_original_code.end(), current);
        }
        const bool layout_valid =
            version_valid && range_valid && (already_disabled || original_sequence_valid);
        bool applied = already_disabled;
        const char* reason =
            already_disabled ? "already_disabled" : "live_opcode_mismatch";
        if (layout_valid && !already_disabled) {
            auto* destination =
                reinterpret_cast<u8*>(g_eboot_address + chromatic_aberration_offset);
            applied = ExecutorWriteSimplePatch(
                destination,
                std::vector<unsigned char>(profile->chromatic_aberration_disabled_code.begin(),
                                           profile->chromatic_aberration_disabled_code.end()));
            reason = applied ? "patched" : "write_failed";
        } else if (!version_valid) {
            reason = "app_version_mismatch";
        } else if (!range_valid) {
            reason = "address_out_of_image";
        }
        log_option("chromatic_aberration", "independent", true, true, layout_valid, applied,
                   reason);
    }

    if (bundle_requested) {
        __android_log_print(bundle_all_applied ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR,
                            "LSX4Native",
                            "[EXECUTOR_TITLE_PROFILE_EFFECT_BUNDLE] serial=%s version=%s "
                            "result=%s flags=%zu",
                            profile_serial.c_str(), game_version.c_str(),
                            bundle_all_applied ? "applied" : "failed", bundle_applied_count);
    }
}

void ApplyConfiguredTitleResolutionOverride() {
    const TitlePatchProfile* profile = FindVerifiedRenderProfile(g_game_serial);
    if (profile == nullptr) {
        return;
    }
    const auto user_dir = Common::FS::GetUserPath(Common::FS::PathType::UserDir);
    const bool force_360 = ExecutorRuntimeFlagExists(
        user_dir, "run-force-internal-resolution-640x360");
    const u32 configured_width = Config::getInternalScreenWidth();
    const u32 configured_height = Config::getInternalScreenHeight();
    const u32 target_width = force_360 ? 640 : configured_width;
    const u32 target_height = force_360 ? 360 : configured_height;
    const bool target_is_16_9 =
        target_width != 0 && target_height != 0 &&
        static_cast<u64>(target_width) * 9 == static_cast<u64>(target_height) * 16;
    const bool target_is_downscale =
        target_width >= 320 && target_height >= 180 && target_width < profile->native_width &&
        target_height < profile->native_height;
    const bool requested = force_360 || (target_is_16_9 && target_is_downscale);
    auto* param_sfo = Common::Singleton<PSF>::Instance();
    const std::string game_version{
        param_sfo->GetString("VERSION").value_or(std::string_view{})};
    const std::string profile_serial{profile->serial};
    const std::string expected_version{profile->version};
    const auto range_in_image = [profile](const uintptr_t guest_address, const std::size_t size) {
        if (guest_address < profile->guest_image_base) {
            return false;
        }
        const uintptr_t offset = guest_address - profile->guest_image_base;
        return offset <= g_eboot_image_size && size <= g_eboot_image_size - offset;
    };
    const auto read_u32 = [&](const uintptr_t guest_address) {
        u32 value{};
        if (!range_in_image(guest_address, sizeof(value))) {
            return value;
        }
        std::memcpy(&value,
                    reinterpret_cast<const void*>(g_eboot_address +
                                                  (guest_address -
                                                   profile->guest_image_base)),
                    sizeof(value));
        return value;
    };
    bool bounds_valid = range_in_image(profile->width_site, sizeof(u32)) &&
                        range_in_image(profile->height_site, sizeof(u32));
    for (const auto& line : profile->resolution_writes) {
        const auto guest_address = static_cast<uintptr_t>(std::stoull(line.address, nullptr, 16));
        const std::string patch_value =
            FormatResolutionPatchValue(line, target_width, target_height);
        const std::size_t byte_count = patch_value.size() / 2;
        bounds_valid &= byte_count != 0 && range_in_image(guest_address, byte_count);
    }
    bool opcode_layout_valid = true;
    for (const ResolutionGuardSpec& guard : profile->resolution_guards) {
        if (!range_in_image(guard.guest_address, guard.expected.size())) {
            opcode_layout_valid = false;
            continue;
        }
        const auto* current = reinterpret_cast<const u8*>(
            g_eboot_address + (guard.guest_address - profile->guest_image_base));
        opcode_layout_valid &=
            std::equal(guard.expected.begin(), guard.expected.end(), current);
    }
    const u32 width = bounds_valid ? read_u32(profile->width_site) : 0;
    const u32 height = bounds_valid ? read_u32(profile->height_site) : 0;
    bool layout_valid = requested && target_is_16_9 && target_is_downscale &&
                        game_version == profile->version && bounds_valid &&
                        width == profile->native_width && height == profile->native_height &&
                        opcode_layout_valid;
    for (const EffectFlagSite& site : profile->effect_flag_sites) {
        if (!range_in_image(site.guest_address, sizeof(u8))) {
            layout_valid = false;
            continue;
        }
        const auto value = *reinterpret_cast<const u8*>(
            g_eboot_address + (site.guest_address - profile->guest_image_base));
        layout_valid &= value <= 1;
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_TITLE_PROFILE_RESOLUTION_LAYOUT] serial=%s version=%s expectedVersion=%s "
        "requested=%u valid=%u bounds=%u opcodes=%u imageSize=0x%llx width=%u height=%u "
        "target=%ux%u source=%s lines=%zu",
        profile_serial.c_str(), game_version.c_str(), expected_version.c_str(),
        requested ? 1u : 0u, layout_valid ? 1u : 0u,
        bounds_valid ? 1u : 0u, opcode_layout_valid ? 1u : 0u,
        static_cast<unsigned long long>(g_eboot_image_size), width, height, target_width,
        target_height, force_360 ? "marker" : "config", profile->resolution_writes.size());
    if (!layout_valid) {
        return;
    }

    const std::string patch_name = profile_serial + " " + expected_version + " " +
                                   std::to_string(target_width) + "x" +
                                   std::to_string(target_height);
    for (const auto& line : profile->resolution_writes) {
        PatchMemory(patch_name, line.address,
                    FormatResolutionPatchValue(line, target_width, target_height), {}, {},
                    false, line.little_endian);
    }
    const u32 patched_width = read_u32(profile->width_site);
    const u32 patched_height = read_u32(profile->height_site);
    const u32 internal_width_a = read_u32(profile->internal_width_sites[0]);
    const u32 internal_width_b = read_u32(profile->internal_width_sites[1]);
    const u32 internal_height_a = read_u32(profile->internal_height_sites[0]);
    const u32 internal_height_b = read_u32(profile->internal_height_sites[1]);
    const bool internal_target_applied =
        internal_width_a == target_width && internal_width_b == target_width &&
        internal_height_a == target_height && internal_height_b == target_height;
    __android_log_print(
        patched_width == target_width && patched_height == target_height &&
                internal_target_applied
            ? ANDROID_LOG_INFO
            : ANDROID_LOG_ERROR,
        "LSX4Native",
        "[EXECUTOR_TITLE_PROFILE_RESOLUTION_APPLIED] serial=%s version=%s width=%u height=%u "
        "expected=%ux%u internal=%u internalWidths=%u,%u internalHeights=%u,%u "
        "source=%s lines=%zu",
        profile_serial.c_str(), game_version.c_str(), patched_width, patched_height,
        target_width, target_height,
        internal_target_applied ? 1u : 0u, internal_width_a, internal_width_b,
        internal_height_a, internal_height_b, force_360 ? "marker" : "config",
        profile->resolution_writes.size());
}
#endif

}

std::string toHex(u64 value, size_t byteSize) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(byteSize * 2) << value;
    return ss.str();
}

std::string convertValueToHex(const std::string type, const std::string valueStr) {
    std::string result;

    if (type == "byte") {
        const u32 value = std::stoul(valueStr, nullptr, 16);
        result = toHex(value, 1);
    } else if (type == "bytes16") {
        const u32 value = std::stoul(valueStr, nullptr, 16);
        result = toHex(value, 2);
    } else if (type == "bytes32") {
        const u32 value = std::stoul(valueStr, nullptr, 16);
        result = toHex(value, 4);
    } else if (type == "bytes64") {
        const u64 value = std::stoull(valueStr, nullptr, 16);
        result = toHex(value, 8);
    } else if (type == "float32") {
        union {
            float f;
            uint32_t i;
        } floatUnion;
        floatUnion.f = std::stof(valueStr);
        result = toHex(std::byteswap(floatUnion.i), sizeof(floatUnion.i));
    } else if (type == "float64") {
        union {
            double d;
            uint64_t i;
        } doubleUnion;
        doubleUnion.d = std::stod(valueStr);
        result = toHex(std::byteswap(doubleUnion.i), sizeof(doubleUnion.i));
    } else if (type == "utf8") {
        std::vector<unsigned char> byteArray =
            std::vector<unsigned char>(valueStr.begin(), valueStr.end());
        byteArray.push_back('\0');
        std::stringstream ss;
        for (unsigned char c : byteArray) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(c);
        }
        result = ss.str();
    } else if (type == "utf16") {
        std::wstring wide_str(valueStr.size(), L'\0');
        std::mbstowcs(&wide_str[0], valueStr.c_str(), valueStr.size());
        wide_str.resize(std::wcslen(wide_str.c_str()));

        std::u16string valueStringU16;

        for (wchar_t wc : wide_str) {
            if (wc <= 0xFFFF) {
                valueStringU16.push_back(static_cast<char16_t>(wc));
            } else {
                wc -= 0x10000;
                valueStringU16.push_back(static_cast<char16_t>(0xD800 | (wc >> 10)));
                valueStringU16.push_back(static_cast<char16_t>(0xDC00 | (wc & 0x3FF)));
            }
        }

        std::vector<unsigned char> byteArray;
        for (char16_t ch : valueStringU16) {
            unsigned char low_byte = static_cast<unsigned char>(ch & 0x00FF);
            unsigned char high_byte = static_cast<unsigned char>((ch >> 8) & 0x00FF);

            byteArray.push_back(low_byte);
            byteArray.push_back(high_byte);
        }
        byteArray.push_back('\0');
        byteArray.push_back('\0');
        std::stringstream ss;

        for (unsigned char ch : byteArray) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
        }
        result = ss.str();
    } else if (type == "bytes") {
        result = valueStr;
    } else if (type == "mask" || type == "mask_jump32") {
        result = valueStr;
    } else {
        LOG_INFO(Loader, "Error applying Patch, unknown type: {}", type);
    }
    return result;
}

void ApplyPendingPatches();

void ApplyPatchesFromXML(std::filesystem::path path) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.c_str());

    auto* param_sfo = Common::Singleton<PSF>::Instance();
    auto app_version = param_sfo->GetString("APP_VER").value_or("Unknown version");

    if (result) {
        auto patchXML = doc.child("Patch");
        for (pugi::xml_node_iterator it = patchXML.children().begin();
             it != patchXML.children().end(); ++it) {

            if (std::string(it->name()) == "Metadata") {
                if (std::string(it->attribute("isEnabled").value()) == "true") {
                    std::string currentPatchName = it->attribute("Name").value();
                    std::string metadataAppVer = it->attribute("AppVer").value();
                    bool versionMatches = metadataAppVer == app_version;

                    auto patchList = it->first_child();
                    for (pugi::xml_node_iterator patchLineIt = patchList.children().begin();
                         patchLineIt != patchList.children().end(); ++patchLineIt) {

                        std::string type = patchLineIt->attribute("Type").value();
                        if (!versionMatches && type != "mask" && type != "mask_jump32")
                            continue;

                        std::string address = patchLineIt->attribute("Address").value();
                        std::string patchValue = patchLineIt->attribute("Value").value();
                        std::string maskOffsetStr = patchLineIt->attribute("Offset").value();
                        std::string targetStr = "";
                        std::string sizeStr = "";
                        if (type == "mask_jump32") {
                            targetStr = patchLineIt->attribute("Target").value();
                            sizeStr = patchLineIt->attribute("Size").value();
                        } else {
                            patchValue = convertValueToHex(type, patchValue);
                        }

                        bool littleEndian = false;
                        if (type == "bytes16" || type == "bytes32" || type == "bytes64") {
                            littleEndian = true;
                        }

                        MemoryPatcher::PatchMask patchMask = MemoryPatcher::PatchMask::None;
                        int maskOffsetValue = 0;

                        if (type == "mask")
                            patchMask = MemoryPatcher::PatchMask::Mask;

                        if (type == "mask_jump32")
                            patchMask = MemoryPatcher::PatchMask::Mask_Jump32;

                        if ((type == "mask" || type == "mask_jump32") && !maskOffsetStr.empty()) {
                            maskOffsetValue = std::stoi(maskOffsetStr, 0, 10);
                        }

                        MemoryPatcher::PatchMemory(currentPatchName, address, patchValue, targetStr,
                                                   sizeStr, false, littleEndian, patchMask,
                                                   maskOffsetValue);
                    }
                }
            }
        }
    } else {
        LOG_ERROR(Loader, "Could not parse patch XML: {}", result.description());
    }
}

void OnGameLoaded() {
    if (g_game_serial.empty()) {
        auto* param_sfo = Common::Singleton<PSF>::Instance();
        if (const auto title_id = param_sfo->GetString("TITLE_ID"); title_id.has_value()) {
            g_game_serial = *title_id;
        }
    }
#ifdef __ANDROID__
    ApplyConfiguredTitleEffectOverrides();
    ApplyConfiguredTitleResolutionOverride();
#endif
    std::filesystem::path patch_dir = Common::FS::GetUserPath(Common::FS::PathType::PatchesDir);
    if (!patch_file.empty()) {

        auto file_path = (patch_dir / patch_file).native();
        if (std::filesystem::exists(patch_file)) {
            ApplyPatchesFromXML(patch_file);
        } else {
            ApplyPatchesFromXML(file_path);
        }
    } else if (EmulatorState::GetInstance()->IsAutoPatchesLoadEnabled()) {
        for (auto const& repo : std::filesystem::directory_iterator(patch_dir)) {
            if (!repo.is_directory()) {
                continue;
            }
            std::ifstream json_file{repo.path() / "files.json"};
            nlohmann::json available_patches = nlohmann::json::parse(json_file);
            std::filesystem::path game_patch_file;
            for (auto const& [filename, serials] : available_patches.items()) {
                if (std::find(serials.begin(), serials.end(), g_game_serial) != serials.end()) {
                    game_patch_file = repo.path() / filename;
                    break;
                }
            }
            if (std::filesystem::exists(game_patch_file)) {
                ApplyPatchesFromXML(game_patch_file);
            }
        }
    }
    ApplyPendingPatches();
}

void AddPatchToQueue(patchInfo patchToAdd) {
    if (patches_applied) {
        PatchMemory(patchToAdd.modNameStr, patchToAdd.offsetStr, patchToAdd.valueStr,
                    patchToAdd.targetStr, patchToAdd.sizeStr, patchToAdd.isOffset,
                    patchToAdd.littleEndian, patchToAdd.patchMask, patchToAdd.maskOffset);
        return;
    }
    pending_patches.push_back(patchToAdd);
}

void ApplyPendingPatches() {
    patches_applied = true;
    for (size_t i = 0; i < pending_patches.size(); ++i) {
        const patchInfo& currentPatch = pending_patches[i];

        if (currentPatch.gameSerial != "*" && currentPatch.gameSerial != g_game_serial)
            continue;

        PatchMemory(currentPatch.modNameStr, currentPatch.offsetStr, currentPatch.valueStr,
                    currentPatch.targetStr, currentPatch.sizeStr, currentPatch.isOffset,
                    currentPatch.littleEndian, currentPatch.patchMask, currentPatch.maskOffset);
    }

    pending_patches.clear();
}

#ifdef __ANDROID__
static bool ExecutorWriteSimplePatch(void* destination,
                                     const std::vector<unsigned char>& bytes) {
    if (bytes.empty()) {
        return true;
    }
    const auto address = reinterpret_cast<uintptr_t>(destination);
    int original_protection = 0;
    std::ifstream maps{"/proc/self/maps"};
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long long begin{};
        unsigned long long end{};
        char permissions[5]{};
        if (std::sscanf(line.c_str(), "%llx-%llx %4s", &begin, &end, permissions) != 3 ||
            address < begin || address >= end) {
            continue;
        }
        original_protection |= permissions[0] == 'r' ? PROT_READ : 0;
        original_protection |= permissions[1] == 'w' ? PROT_WRITE : 0;
        original_protection |= permissions[2] == 'x' ? PROT_EXEC : 0;
        break;
    }
    if (original_protection == 0) {
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4Native",
            "[EXECUTOR_PATCH_PROTECTION] result=failed reason=map_not_found address=0x%llx",
            static_cast<unsigned long long>(address));
        return false;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return false;
    }
    const uintptr_t page_mask = static_cast<uintptr_t>(page_size - 1);
    const uintptr_t page_begin = address & ~page_mask;
    const uintptr_t page_end =
        (address + bytes.size() + page_mask) & ~page_mask;
    const std::size_t page_span = page_end - page_begin;
    const bool protection_change = (original_protection & PROT_WRITE) == 0;
    if (protection_change &&
        mprotect(reinterpret_cast<void*>(page_begin), page_span,
                 (original_protection | PROT_READ | PROT_WRITE) & ~PROT_EXEC) != 0) {
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4Native",
            "[EXECUTOR_PATCH_PROTECTION] result=failed phase=rw address=0x%llx errno=%d",
            static_cast<unsigned long long>(address), errno);
        return false;
    }

    std::memcpy(destination, bytes.data(), bytes.size());
    if (original_protection & PROT_EXEC) {
        __builtin___clear_cache(reinterpret_cast<char*>(destination),
                                reinterpret_cast<char*>(destination) + bytes.size());
    }
    if (protection_change &&
        mprotect(reinterpret_cast<void*>(page_begin), page_span, original_protection) != 0) {
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4Native",
            "[EXECUTOR_PATCH_PROTECTION] result=failed phase=restore address=0x%llx errno=%d",
            static_cast<unsigned long long>(address), errno);
        return false;
    }
    return true;
}
#endif

void PatchMemory(std::string modNameStr, std::string offsetStr, std::string valueStr,
                 std::string targetStr, std::string sizeStr, bool isOffset, bool littleEndian,
                 PatchMask patchMask, int maskOffset) {
    void* cheatAddress = nullptr;

    if (patchMask == PatchMask::None) {
        if (isOffset) {
            cheatAddress = reinterpret_cast<void*>(g_eboot_address + std::stoi(offsetStr, 0, 16));
        } else {
            cheatAddress =
                reinterpret_cast<void*>(g_eboot_address + (std::stoi(offsetStr, 0, 16) - 0x400000));
        }
    }

    if (patchMask == PatchMask::Mask) {
        cheatAddress = reinterpret_cast<void*>(PatternScan(offsetStr) + maskOffset);
    }

    if (patchMask == PatchMask::Mask_Jump32) {
        int jumpSize = std::stoi(sizeStr);

        constexpr int MAX_PATTERN_LENGTH = 256;
        if (jumpSize < 5) {
            LOG_ERROR(Loader, "Jump size must be at least 5 bytes");
            return;
        }
        if (jumpSize > MAX_PATTERN_LENGTH) {
            LOG_ERROR(Loader, "Jump size must be no more than {} bytes.", MAX_PATTERN_LENGTH);
            return;
        }

        uintptr_t baseAddress = PatternScan(offsetStr);
        if (baseAddress == 0) {
            LOG_ERROR(Loader, "PatternScan failed for mask_jump32 with pattern: {}", offsetStr);
            return;
        }
        uintptr_t patchAddress = baseAddress + maskOffset;

        std::vector<u8> nopBytes(jumpSize, 0x90);
        std::memcpy(reinterpret_cast<void*>(patchAddress), nopBytes.data(), nopBytes.size());

        uintptr_t jump_target = PatternScan(targetStr);
        if (jump_target == 0) {
            LOG_ERROR(Loader, "PatternScan failed to Target with pattern: {}", targetStr);
            return;
        }

        std::vector<u8> payload;
        for (size_t i = 0; i < valueStr.length(); i += 2) {

            std::string tempStr = valueStr.substr(i, 2);
            const char* byteStr = tempStr.c_str();
            char* endPtr;
            unsigned int byteVal = std::strtoul(byteStr, &endPtr, 16);

            if (endPtr != byteStr + 2) {
                LOG_ERROR(Loader, "Invalid byte in Value: {}", valueStr.substr(i, 2));
                return;
            }
            payload.push_back(static_cast<u8>(byteVal));
        }

        uintptr_t code_cave_end = jump_target + payload.size();

        std::memcpy(reinterpret_cast<void*>(jump_target), payload.data(), payload.size());

        u8 jumpInstruction[5];
        jumpInstruction[0] = 0xE9;
        s32 relJump = static_cast<s32>(jump_target - patchAddress - 5);
        std::memcpy(&jumpInstruction[1], &relJump, sizeof(relJump));
        std::memcpy(reinterpret_cast<void*>(patchAddress), jumpInstruction,
                    sizeof(jumpInstruction));

        u8 jumpBack[5];
        jumpBack[0] = 0xE9;
        s32 target_return = static_cast<s32>((patchAddress + jumpSize) - (code_cave_end + 5));
        std::memcpy(&jumpBack[1], &target_return, sizeof(target_return));
        std::memcpy(reinterpret_cast<void*>(code_cave_end), jumpBack, sizeof(jumpBack));

        LOG_INFO(Loader,
                 "Applied Patch mask_jump32: {}, PatchAddress: {:#x}, JumpTarget: {:#x}, "
                 "CodeCaveEnd: {:#x}, JumpSize: {}",
                 modNameStr, patchAddress, jump_target, code_cave_end, jumpSize);
        return;
    }

    if (cheatAddress == nullptr) {
        LOG_ERROR(Loader, "Failed to get address for patch {}", modNameStr);
        return;
    }

    std::vector<unsigned char> bytePatch;

    for (size_t i = 0; i < valueStr.length(); i += 2) {
        unsigned char byte =
            static_cast<unsigned char>(std::strtol(valueStr.substr(i, 2).c_str(), nullptr, 16));

        bytePatch.push_back(byte);
    }

    if (littleEndian) {
        std::reverse(bytePatch.begin(), bytePatch.end());
    }

#ifdef __ANDROID__
    if (!ExecutorWriteSimplePatch(cheatAddress, bytePatch)) {
        LOG_ERROR(Loader, "Failed to apply patch: {}, Offset: {}", modNameStr,
                  (uintptr_t)cheatAddress);
        return;
    }
#else
    std::memcpy(cheatAddress, bytePatch.data(), bytePatch.size());
#endif

    LOG_INFO(Loader, "Applied patch: {}, Offset: {}, Value: {}", modNameStr,
             (uintptr_t)cheatAddress, valueStr);
}

static std::vector<int32_t> PatternToByte(const std::string& pattern) {
    std::vector<int32_t> bytes;
    const char* start = pattern.data();
    const char* end = start + pattern.size();

    for (const char* current = start; current < end; ++current) {
        if (*current == '?') {
            ++current;
            if (*current == '?')
                ++current;
            bytes.push_back(-1);
        } else {
            bytes.push_back(strtoul(current, const_cast<char**>(&current), 16));
        }
    }

    return bytes;
}

uintptr_t PatternScan(const std::string& signature) {
    std::vector<int32_t> patternBytes = PatternToByte(signature);
    const auto scanBytes = static_cast<uint8_t*>((void*)g_eboot_address);

    const int32_t* sigPtr = patternBytes.data();
    const size_t sigSize = patternBytes.size();

    uint32_t foundResults = 0;
    for (uint32_t i = 0; i < g_eboot_image_size - sigSize; ++i) {
        bool found = true;
        for (uint32_t j = 0; j < sigSize; ++j) {
            if (scanBytes[i + j] != sigPtr[j] && sigPtr[j] != -1) {
                found = false;
                break;
            }
        }

        if (found) {
            foundResults++;
            return reinterpret_cast<uintptr_t>(&scanBytes[i]);
        }
    }

    return 0;
}

}
