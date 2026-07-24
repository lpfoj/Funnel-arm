// SPDX-FileCopyrightText: Copyright 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string_view>
#include "common/logging/filter.h"

namespace Common::Log {

class Filter;

void Initialize(std::string_view log_file = "");

bool IsActive();

void Start();

void Stop();

void Denitializer();

void SetGlobalFilter(const Filter& filter);

void SetColorConsoleBackendEnabled(bool enabled);

void SetAppend();

}
