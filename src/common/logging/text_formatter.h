// SPDX-FileCopyrightText: Copyright 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

namespace Common::Log {

struct Entry;

std::string FormatLogMessage(const Entry& entry);

void PrintMessage(const Entry& entry);

void PrintColoredMessage(const Entry& entry);

}
