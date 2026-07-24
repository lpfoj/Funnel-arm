// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

#include "common/types.h"
#include "core/libraries/pad/pad.h"
#include "input/controller.h"

struct SDL_Window;
struct SDL_Gamepad;
union SDL_Event;

namespace Input {

class SDLInputEngine : public Engine {
public:
    ~SDLInputEngine() override;
    void Init() override;
    void SetLightBarRGB(u8 r, u8 g, u8 b) override;
    void SetVibration(u8 smallMotor, u8 largeMotor) override;
    float GetGyroPollRate() const override;
    float GetAccelPollRate() const override;
    State ReadState() override;

private:
    float m_gyro_poll_rate = 0.0f;
    float m_accel_poll_rate = 0.0f;
};

}

namespace Frontend {

enum class WindowSystemType : u8 {
    Headless,
    Android,
    Windows,
    X11,
    Wayland,
    Metal,
};

struct WindowSystemInfo {
    void* display_connection = nullptr;

    void* render_surface = nullptr;

    float render_surface_scale = 1.0f;

    WindowSystemType type = WindowSystemType::Headless;
};

class WindowSDL {
    int keyboard_grab = 0;

public:
    explicit WindowSDL(s32 width, s32 height, Input::GameController* controller,
                       std::string_view window_title);
#if defined(__ANDROID__)
    WindowSDL(s32 width, s32 height, void* native_window);
#endif
    ~WindowSDL();

    s32 GetWidth() const {
        return width;
    }

    s32 GetHeight() const {
        return height;
    }

    bool IsOpen() const {
        return is_open;
    }

    [[nodiscard]] SDL_Window* GetSDLWindow() const {
        return window;
    }

    WindowSystemInfo GetWindowInfo() const {
        return window_info;
    }

    void WaitEvent();
    void InitTimers();

    void RequestKeyboard();
    void ReleaseKeyboard();

private:
    void OnResize();
    void OnKeyboardMouseInput(const SDL_Event* event);
    void OnGamepadEvent(const SDL_Event* event);

private:
    s32 width;
    s32 height;
    Input::GameController* controller;
    WindowSystemInfo window_info{};
    SDL_Window* window{};
    bool is_shown{};
    bool is_open{true};
};

}
