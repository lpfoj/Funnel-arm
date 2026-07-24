// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

namespace Common {

[[nodiscard]] std::string GetLastErrorMsg();

[[nodiscard]] std::string NativeErrorToString(int e);

}
