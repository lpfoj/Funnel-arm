// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once
#include <string>

#include "imgui/imgui_layer.h"

namespace Core::Devtools {

class Layer final : public ImGui::Layer {
public:
    static void SetupSettings();

    bool CapturesGamepadInput() const override {
        return false;
    }
    void Draw() override;

    static void DrawNullGpuNotice();

private:
    static void DrawMenuBar();
    static void DrawAdvanced();
    static void DrawSimple();

    static void TextCentered(const std::string& text);
};

}

namespace Overlay {

void ToggleSimpleFps();
void SetSimpleFps(bool enabled);
void ToggleQuitWindow();
void ShowVolume();

}
