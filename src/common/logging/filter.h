// SPDX-FileCopyrightText: Copyright 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#include "common/logging/types.h"

namespace Common::Log {

const char* GetLogClassName(Class log_class);

const char* GetLevelName(Level log_level);

class Filter {
public:
    explicit Filter(Level default_level = Level::Info);

    void ResetAll(Level level);

    void SetClassLevel(Class log_class, Level level);

    void ParseFilterString(std::string_view filter_view);

    bool CheckMessage(Class log_class, Level level) const;

    bool IsDebug() const;

private:
    std::array<Level, static_cast<std::size_t>(Class::Count)> class_levels;
};

}
