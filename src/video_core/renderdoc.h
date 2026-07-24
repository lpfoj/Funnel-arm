// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

namespace VideoCore {

void LoadRenderDoc();

void StartCapture();

void EndCapture();

void TriggerCapture();

void SetOutputDir(const std::filesystem::path& path, const std::string& prefix);

}
