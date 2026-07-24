// SPDX-FileCopyrightText: Copyright 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>

#include "common/logging/types.h"

namespace Common::Log {

struct Entry {
    std::chrono::microseconds timestamp;
    Class log_class{};
    Level log_level{};
    const char* filename = nullptr;
    u32 line_num = 0;
    std::string function;
    std::string message;
    std::string thread;
    u32 counter = 0;
};

}
