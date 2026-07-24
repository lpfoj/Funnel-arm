// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cctype>

#include "core/libraries/avplayer/avplayer_common.h"

namespace Libraries::AvPlayer {

static bool iequals(std::string_view l, std::string_view r) {
    return std::ranges::equal(l, r, [](u8 a, u8 b) { return std::tolower(a) == std::tolower(b); });
}

AvPlayerSourceType GetSourceType(std::string_view path) {
    if (path.empty()) {
        return AvPlayerSourceType::Unknown;
    }

    std::string_view name = path;
    if (path.find("://") != std::string_view::npos) {
        name = path.substr(0, path.find_first_of("?#"));
        if (name.empty()) {
            return AvPlayerSourceType::Unknown;
        }
    }


    auto dot_pos = name.rfind('.');
    if (dot_pos == std::string_view::npos) {
        return AvPlayerSourceType::Unknown;
    }

    auto ext = name.substr(dot_pos);
    if (ext.empty()) {
        return AvPlayerSourceType::Unknown;
    }

    ext = ext.substr(0, ext.find('/'));

    if (iequals(ext, ".mp4") || iequals(ext, ".m4v") || iequals(ext, ".m3d") ||
        iequals(ext, ".m4a") || iequals(ext, ".mov")) {
        return AvPlayerSourceType::FileMp4;
    }

    if (iequals(ext, ".m3u8")) {
        return AvPlayerSourceType::Hls;
    }

    return AvPlayerSourceType::Unknown;
}

}
