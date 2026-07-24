// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <optional>
#include <vector>

namespace Common::FS {

enum class PathType {
    UserDir,
    LogDir,
    ScreenshotsDir,
    ShaderDir,
    TempDataDir,
    GameDataDir,
    SysModuleDir,
    DownloadDir,
    CapturesDir,
    CheatsDir,
    PatchesDir,
    MetaDataDir,
    CustomTrophy,
    CustomConfigs,
    CacheDir,
    FontsDir,
};

constexpr auto PORTABLE_DIR = "user";

constexpr auto LOG_DIR = "log";
constexpr auto SCREENSHOTS_DIR = "screenshots";
constexpr auto SHADER_DIR = "shader";
constexpr auto GAMEDATA_DIR = "data";
constexpr auto TEMPDATA_DIR = "temp";
constexpr auto SYSMODULES_DIR = "sys_modules";
constexpr auto DOWNLOAD_DIR = "download";
constexpr auto CAPTURES_DIR = "captures";
constexpr auto CHEATS_DIR = "cheats";
constexpr auto PATCHES_DIR = "patches";
constexpr auto METADATA_DIR = "game_data";
constexpr auto CUSTOM_TROPHY = "custom_trophy";
constexpr auto CUSTOM_CONFIGS = "custom_configs";
constexpr auto CACHE_DIR = "cache";
constexpr auto FONTS_DIR = "fonts";

constexpr auto LOG_FILE = "shad_log.txt";

[[nodiscard]] bool ValidatePath(const std::filesystem::path& path);

[[nodiscard]] std::string PathToUTF8String(const std::filesystem::path& path);

[[nodiscard]] const std::filesystem::path& GetUserPath(PathType user_path);

[[nodiscard]] std::string GetUserPathString(PathType user_path);

void SetUserPath(PathType user_path, const std::filesystem::path& new_path);

[[nodiscard]] std::optional<std::filesystem::path> FindGameByID(const std::filesystem::path& dir,
                                                                const std::string& game_id,
                                                                int max_depth);

}
