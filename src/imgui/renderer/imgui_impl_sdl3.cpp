// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <imgui.h>
#include "common/config.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "imgui_impl_sdl3.h"
#include "input/controller.h"
#include "input/input_handler.h"
#include "sdl_window.h"

#include <SDL3/SDL.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

namespace ImGui::Sdl {

struct SdlData {
    SDL_Window* window{};
    SDL_WindowID window_id{};
    Uint64 time{};
    Uint64 nonReusedtime{};
    const char* clipboard_text_data{};

    SDL_Window* ime_window{};

    Uint32 mouse_window_id{};
    int mouse_buttons_down{};
    SDL_Cursor* mouse_cursors[ImGuiMouseCursor_COUNT]{};
    SDL_Cursor* mouse_last_cursor{};
    int mouse_pending_leave_frame{};
    ImVec2 prev_mouse_pos{0, 0};
    Uint64 lastCursorMoveTime{};

    ImVector<SDL_Gamepad*> gamepads{};
    GamepadMode gamepad_mode{};
    bool want_update_gamepads_list{};

    std::array<float, 60> framerateSecPerFrame;
    int framerateSecPerFrameIdx{};
    float framerateSecPerFrameAcc{};
};

static SdlData* GetBackendData() {
    return ImGui::GetCurrentContext() ? (SdlData*)ImGui::GetIO().BackendPlatformUserData : nullptr;
}

static const char* GetClipboardText(ImGuiContext*) {
    SdlData* bd = GetBackendData();
    if (bd->clipboard_text_data)
        SDL_free((void*)bd->clipboard_text_data);
    const char* sdl_clipboard_text = SDL_GetClipboardText();
    bd->clipboard_text_data = sdl_clipboard_text;
    return bd->clipboard_text_data;
}

static void SetClipboardText(ImGuiContext*, const char* text) {
    SDL_SetClipboardText(text);
}

static void PlatformSetImeData(ImGuiContext*, ImGuiViewport* viewport, ImGuiPlatformImeData* data) {
    SdlData* bd = GetBackendData();
    auto window_id = (SDL_WindowID)(intptr_t)viewport->PlatformHandle;
    SDL_Window* window = SDL_GetWindowFromID(window_id);
    if ((!data->WantVisible || bd->ime_window != window) && bd->ime_window != nullptr) {
        SDL_RunOnMainThread(
            [](void* userdata) { SDL_StopTextInput(static_cast<SDL_Window*>(userdata)); },
            bd->ime_window, true);
        bd->ime_window = nullptr;
    }
    if (data->WantVisible) {
        std::pair<SDL_Window*, SDL_Rect> usr_data;
        usr_data.first = window;
        usr_data.second.x = (int)data->InputPos.x;
        usr_data.second.y = (int)data->InputPos.y;
        usr_data.second.w = 1;
        usr_data.second.h = (int)data->InputLineHeight;
        SDL_RunOnMainThread(
            [](void* userdata) {
                auto* params = static_cast<std::pair<SDL_Window*, SDL_Rect>*>(userdata);
                SDL_SetTextInputArea(params->first, &params->second, 0);
                SDL_StartTextInput(params->first);
            },
            &usr_data, true);
        bd->ime_window = window;
    }
}

static ImGuiKey KeyEventToImGuiKey(SDL_Keycode keycode, SDL_Scancode scancode) {
    switch (scancode) {
    case SDL_SCANCODE_KP_0:
        return ImGuiKey_Keypad0;
    case SDL_SCANCODE_KP_1:
        return ImGuiKey_Keypad1;
    case SDL_SCANCODE_KP_2:
        return ImGuiKey_Keypad2;
    case SDL_SCANCODE_KP_3:
        return ImGuiKey_Keypad3;
    case SDL_SCANCODE_KP_4:
        return ImGuiKey_Keypad4;
    case SDL_SCANCODE_KP_5:
        return ImGuiKey_Keypad5;
    case SDL_SCANCODE_KP_6:
        return ImGuiKey_Keypad6;
    case SDL_SCANCODE_KP_7:
        return ImGuiKey_Keypad7;
    case SDL_SCANCODE_KP_8:
        return ImGuiKey_Keypad8;
    case SDL_SCANCODE_KP_9:
        return ImGuiKey_Keypad9;
    case SDL_SCANCODE_KP_PERIOD:
        return ImGuiKey_KeypadDecimal;
    case SDL_SCANCODE_KP_DIVIDE:
        return ImGuiKey_KeypadDivide;
    case SDL_SCANCODE_KP_MULTIPLY:
        return ImGuiKey_KeypadMultiply;
    case SDL_SCANCODE_KP_MINUS:
        return ImGuiKey_KeypadSubtract;
    case SDL_SCANCODE_KP_PLUS:
        return ImGuiKey_KeypadAdd;
    case SDL_SCANCODE_KP_ENTER:
        return ImGuiKey_KeypadEnter;
    case SDL_SCANCODE_KP_EQUALS:
        return ImGuiKey_KeypadEqual;
    default:
        break;
    }
    switch (keycode) {
    case SDLK_TAB:
        return ImGuiKey_Tab;
    case SDLK_LEFT:
        return ImGuiKey_LeftArrow;
    case SDLK_RIGHT:
        return ImGuiKey_RightArrow;
    case SDLK_UP:
        return ImGuiKey_UpArrow;
    case SDLK_DOWN:
        return ImGuiKey_DownArrow;
    case SDLK_PAGEUP:
        return ImGuiKey_PageUp;
    case SDLK_PAGEDOWN:
        return ImGuiKey_PageDown;
    case SDLK_HOME:
        return ImGuiKey_Home;
    case SDLK_END:
        return ImGuiKey_End;
    case SDLK_INSERT:
        return ImGuiKey_Insert;
    case SDLK_DELETE:
        return ImGuiKey_Delete;
    case SDLK_BACKSPACE:
        return ImGuiKey_Backspace;
    case SDLK_SPACE:
        return ImGuiKey_Space;
    case SDLK_RETURN:
        return ImGuiKey_Enter;
    case SDLK_ESCAPE:
        return ImGuiKey_Escape;
    case SDLK_APOSTROPHE:
        return ImGuiKey_Apostrophe;
    case SDLK_COMMA:
        return ImGuiKey_Comma;
    case SDLK_MINUS:
        return ImGuiKey_Minus;
    case SDLK_PERIOD:
        return ImGuiKey_Period;
    case SDLK_SLASH:
        return ImGuiKey_Slash;
    case SDLK_SEMICOLON:
        return ImGuiKey_Semicolon;
    case SDLK_EQUALS:
        return ImGuiKey_Equal;
    case SDLK_LEFTBRACKET:
        return ImGuiKey_LeftBracket;
    case SDLK_BACKSLASH:
        return ImGuiKey_Backslash;
    case SDLK_RIGHTBRACKET:
        return ImGuiKey_RightBracket;
    case SDLK_GRAVE:
        return ImGuiKey_GraveAccent;
    case SDLK_CAPSLOCK:
        return ImGuiKey_CapsLock;
    case SDLK_SCROLLLOCK:
        return ImGuiKey_ScrollLock;
    case SDLK_NUMLOCKCLEAR:
        return ImGuiKey_NumLock;
    case SDLK_PRINTSCREEN:
        return ImGuiKey_PrintScreen;
    case SDLK_PAUSE:
        return ImGuiKey_Pause;
    case SDLK_LCTRL:
        return ImGuiKey_LeftCtrl;
    case SDLK_LSHIFT:
        return ImGuiKey_LeftShift;
    case SDLK_LALT:
        return ImGuiKey_LeftAlt;
    case SDLK_LGUI:
        return ImGuiKey_LeftSuper;
    case SDLK_RCTRL:
        return ImGuiKey_RightCtrl;
    case SDLK_RSHIFT:
        return ImGuiKey_RightShift;
    case SDLK_RALT:
        return ImGuiKey_RightAlt;
    case SDLK_RGUI:
        return ImGuiKey_RightSuper;
    case SDLK_APPLICATION:
        return ImGuiKey_Menu;
    case SDLK_0:
        return ImGuiKey_0;
    case SDLK_1:
        return ImGuiKey_1;
    case SDLK_2:
        return ImGuiKey_2;
    case SDLK_3:
        return ImGuiKey_3;
    case SDLK_4:
        return ImGuiKey_4;
    case SDLK_5:
        return ImGuiKey_5;
    case SDLK_6:
        return ImGuiKey_6;
    case SDLK_7:
        return ImGuiKey_7;
    case SDLK_8:
        return ImGuiKey_8;
    case SDLK_9:
        return ImGuiKey_9;
    case SDLK_A:
        return ImGuiKey_A;
    case SDLK_B:
        return ImGuiKey_B;
    case SDLK_C:
        return ImGuiKey_C;
    case SDLK_D:
        return ImGuiKey_D;
    case SDLK_E:
        return ImGuiKey_E;
    case SDLK_F:
        return ImGuiKey_F;
    case SDLK_G:
        return ImGuiKey_G;
    case SDLK_H:
        return ImGuiKey_H;
    case SDLK_I:
        return ImGuiKey_I;
    case SDLK_J:
        return ImGuiKey_J;
    case SDLK_K:
        return ImGuiKey_K;
    case SDLK_L:
        return ImGuiKey_L;
    case SDLK_M:
        return ImGuiKey_M;
    case SDLK_N:
        return ImGuiKey_N;
    case SDLK_O:
        return ImGuiKey_O;
    case SDLK_P:
        return ImGuiKey_P;
    case SDLK_Q:
        return ImGuiKey_Q;
    case SDLK_R:
        return ImGuiKey_R;
    case SDLK_S:
        return ImGuiKey_S;
    case SDLK_T:
        return ImGuiKey_T;
    case SDLK_U:
        return ImGuiKey_U;
    case SDLK_V:
        return ImGuiKey_V;
    case SDLK_W:
        return ImGuiKey_W;
    case SDLK_X:
        return ImGuiKey_X;
    case SDLK_Y:
        return ImGuiKey_Y;
    case SDLK_Z:
        return ImGuiKey_Z;
    case SDLK_F1:
        return ImGuiKey_F1;
    case SDLK_F2:
        return ImGuiKey_F2;
    case SDLK_F3:
        return ImGuiKey_F3;
    case SDLK_F4:
        return ImGuiKey_F4;
    case SDLK_F5:
        return ImGuiKey_F5;
    case SDLK_F6:
        return ImGuiKey_F6;
    case SDLK_F7:
        return ImGuiKey_F7;
    case SDLK_F8:
        return ImGuiKey_F8;
    case SDLK_F9:
        return ImGuiKey_F9;
    case SDLK_F10:
        return ImGuiKey_F10;
    case SDLK_F11:
        return ImGuiKey_F11;
    case SDLK_F12:
        return ImGuiKey_F12;
    case SDLK_F13:
        return ImGuiKey_F13;
    case SDLK_F14:
        return ImGuiKey_F14;
    case SDLK_F15:
        return ImGuiKey_F15;
    case SDLK_F16:
        return ImGuiKey_F16;
    case SDLK_F17:
        return ImGuiKey_F17;
    case SDLK_F18:
        return ImGuiKey_F18;
    case SDLK_F19:
        return ImGuiKey_F19;
    case SDLK_F20:
        return ImGuiKey_F20;
    case SDLK_F21:
        return ImGuiKey_F21;
    case SDLK_F22:
        return ImGuiKey_F22;
    case SDLK_F23:
        return ImGuiKey_F23;
    case SDLK_F24:
        return ImGuiKey_F24;
    case SDLK_AC_BACK:
        return ImGuiKey_AppBack;
    case SDLK_AC_FORWARD:
        return ImGuiKey_AppForward;
    default:
        break;
    }
    return ImGuiKey_None;
}

static void UpdateKeyModifiers(SDL_Keymod sdl_key_mods) {
    ImGuiIO& io = ImGui::GetIO();
    io.AddKeyEvent(ImGuiMod_Ctrl, (sdl_key_mods & SDL_KMOD_CTRL) != 0);
    io.AddKeyEvent(ImGuiMod_Shift, (sdl_key_mods & SDL_KMOD_SHIFT) != 0);
    io.AddKeyEvent(ImGuiMod_Alt, (sdl_key_mods & SDL_KMOD_ALT) != 0);
    io.AddKeyEvent(ImGuiMod_Super, (sdl_key_mods & SDL_KMOD_GUI) != 0);
}

static ImGuiViewport* GetViewportForWindowId(SDL_WindowID window_id) {
    SdlData* bd = GetBackendData();
    return (window_id == bd->window_id) ? ImGui::GetMainViewport() : nullptr;
}

bool ProcessEvent(const SDL_Event* event) {
    SdlData* bd = GetBackendData();
    IM_ASSERT(bd != nullptr &&
              "Context or backend not initialized! Did you call ImGui_ImplSDL3_Init()?");
    ImGuiIO& io = ImGui::GetIO();

    switch (event->type) {
    case SDL_EVENT_MOUSE_MOTION: {
        if (GetViewportForWindowId(event->motion.windowID) == NULL)
            return false;
        ImVec2 mouse_pos((float)event->motion.x, (float)event->motion.y);
        io.AddMouseSourceEvent(event->motion.which == SDL_TOUCH_MOUSEID
                                   ? ImGuiMouseSource_TouchScreen
                                   : ImGuiMouseSource_Mouse);
        io.AddMousePosEvent(mouse_pos.x, mouse_pos.y);
        if (mouse_pos.x != bd->prev_mouse_pos.x || mouse_pos.y != bd->prev_mouse_pos.y) {
            bd->prev_mouse_pos.x = mouse_pos.x;
            bd->prev_mouse_pos.y = mouse_pos.y;
            if (Config::getCursorState() == Config::HideCursorState::Idle) {
                bd->lastCursorMoveTime = bd->time;
            }
        }
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL: {
        if (GetViewportForWindowId(event->wheel.windowID) == NULL)
            return false;
        float wheel_x = -event->wheel.x;
        float wheel_y = event->wheel.y;
#ifdef __EMSCRIPTEN__
        wheel_x /= 100.0f;
#endif
        io.AddMouseSourceEvent(event->wheel.which == SDL_TOUCH_MOUSEID
                                   ? ImGuiMouseSource_TouchScreen
                                   : ImGuiMouseSource_Mouse);
        io.AddMouseWheelEvent(wheel_x, wheel_y);
        return true;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (GetViewportForWindowId(event->button.windowID) == NULL)
            return false;
        int mouse_button = -1;
        if (event->button.button == SDL_BUTTON_LEFT) {
            mouse_button = 0;
        }
        if (event->button.button == SDL_BUTTON_RIGHT) {
            mouse_button = 1;
        }
        if (event->button.button == SDL_BUTTON_MIDDLE) {
            mouse_button = 2;
        }
        if (event->button.button == SDL_BUTTON_X1) {
            mouse_button = 3;
        }
        if (event->button.button == SDL_BUTTON_X2) {
            mouse_button = 4;
        }
        if (mouse_button == -1)
            break;
        io.AddMouseSourceEvent(event->button.which == SDL_TOUCH_MOUSEID
                                   ? ImGuiMouseSource_TouchScreen
                                   : ImGuiMouseSource_Mouse);
        io.AddMouseButtonEvent(mouse_button, (event->type == SDL_EVENT_MOUSE_BUTTON_DOWN));
        bd->mouse_buttons_down = (event->type == SDL_EVENT_MOUSE_BUTTON_DOWN)
                                     ? (bd->mouse_buttons_down | (1 << mouse_button))
                                     : (bd->mouse_buttons_down & ~(1 << mouse_button));
        return true;
    }
    case SDL_EVENT_TEXT_INPUT: {
        if (GetViewportForWindowId(event->text.windowID) == NULL)
            return false;
        io.AddInputCharactersUTF8(event->text.text);
        return true;
    }
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        if (GetViewportForWindowId(event->key.windowID) == NULL)
            return false;
        UpdateKeyModifiers((SDL_Keymod)event->key.mod);
        ImGuiKey key = KeyEventToImGuiKey(event->key.key, event->key.scancode);
        io.AddKeyEvent(key, (event->type == SDL_EVENT_KEY_DOWN));
        io.SetKeyEventNativeData(key, event->key.key, event->key.scancode,
                                 event->key.scancode);
        return true;
    }
    case SDL_EVENT_WINDOW_MOUSE_ENTER: {
        if (GetViewportForWindowId(event->window.windowID) == NULL)
            return false;
        bd->mouse_window_id = event->window.windowID;
        bd->mouse_pending_leave_frame = 0;
        bd->lastCursorMoveTime = bd->time;
        return true;
    }
    case SDL_EVENT_WINDOW_MOUSE_LEAVE: {
        if (GetViewportForWindowId(event->window.windowID) == NULL)
            return false;
        bd->mouse_pending_leave_frame = ImGui::GetFrameCount() + 1;
        bd->lastCursorMoveTime = bd->time;
        return true;
    }
    case SDL_EVENT_WINDOW_FOCUS_GAINED:
    case SDL_EVENT_WINDOW_FOCUS_LOST: {
        if (GetViewportForWindowId(event->window.windowID) == NULL)
            return false;
        io.AddFocusEvent(event->type == SDL_EVENT_WINDOW_FOCUS_GAINED);
        return true;
    }
    case SDL_EVENT_GAMEPAD_ADDED:
    case SDL_EVENT_GAMEPAD_REMOVED: {
        bd->want_update_gamepads_list = true;
        return true;
    }
    case SDL_EVENT_CHANGE_CONTROLLER:
        bd->want_update_gamepads_list = true;
        return false;
    }
    return false;
}

static void SetupPlatformHandles(ImGuiViewport* viewport, SDL_Window* window) {
    viewport->PlatformHandle = (void*)(intptr_t)SDL_GetWindowID(window);
    viewport->PlatformHandleRaw = nullptr;
#if defined(_WIN32) && !defined(__WINRT__)
    viewport->PlatformHandleRaw = (HWND)SDL_GetPointerProperty(
        SDL_GetWindowProperties(window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
#elif defined(__APPLE__) && defined(SDL_VIDEO_DRIVER_COCOA)
    viewport->PlatformHandleRaw = SDL_GetPointerProperty(
        SDL_GetWindowProperties(window), SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr);
#endif
}

bool Init(SDL_Window* window) {
    ImGuiIO& io = ImGui::GetIO();
    IMGUI_CHECKVERSION();
    IM_ASSERT(io.BackendPlatformUserData == nullptr && "Already initialized a platform backend!");

    SdlData* bd = IM_NEW(SdlData)();
    io.BackendPlatformUserData = (void*)bd;
    io.BackendPlatformName = "imgui_impl_sdl3_shadps4";
    io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
    io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;

    bd->window = window;
    bd->window_id = SDL_GetWindowID(window);

    ImGuiPlatformIO& platform_io = ImGui::GetPlatformIO();
    platform_io.Platform_SetClipboardTextFn = SetClipboardText;
    platform_io.Platform_GetClipboardTextFn = GetClipboardText;
    platform_io.Platform_SetImeDataFn = PlatformSetImeData;

    bd->gamepad_mode = ImGui_ImplSDL3_GamepadMode_AutoFirst;
    bd->want_update_gamepads_list = true;

#define CURSOR(left, right)                                                                        \
    bd->mouse_cursors[ImGuiMouseCursor_##left] = SDL_CreateSystemCursor(SDL_SYSTEM_CURSOR_##right)
    CURSOR(Arrow, DEFAULT);
    CURSOR(TextInput, TEXT);
    CURSOR(ResizeAll, MOVE);
    CURSOR(ResizeNS, NS_RESIZE);
    CURSOR(ResizeEW, EW_RESIZE);
    CURSOR(ResizeNESW, NESW_RESIZE);
    CURSOR(ResizeNWSE, NWSE_RESIZE);
    CURSOR(Hand, POINTER);
    CURSOR(NotAllowed, NOT_ALLOWED);
#undef CURSOR

    ImGuiViewport* main_viewport = ImGui::GetMainViewport();
    SetupPlatformHandles(main_viewport, window);

#ifdef SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH
    SDL_SetHint(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, "1");
#endif

#ifdef SDL_HINT_MOUSE_AUTO_CAPTURE
    SDL_SetHint(SDL_HINT_MOUSE_AUTO_CAPTURE, "0");
#endif

    return true;
}

static void CloseGamepads();

void Shutdown() {
    SdlData* bd = GetBackendData();
    IM_ASSERT(bd != nullptr && "No platform backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();

    if (bd->clipboard_text_data) {
        SDL_free((void*)bd->clipboard_text_data);
    }
    for (ImGuiMouseCursor cursor_n = 0; cursor_n < ImGuiMouseCursor_COUNT; cursor_n++)
        SDL_DestroyCursor(bd->mouse_cursors[cursor_n]);
    CloseGamepads();

    io.BackendPlatformName = nullptr;
    io.BackendPlatformUserData = nullptr;
    io.BackendFlags &= ~(ImGuiBackendFlags_HasMouseCursors | ImGuiBackendFlags_HasSetMousePos |
                         ImGuiBackendFlags_HasGamepad);
    IM_DELETE(bd);
}

static void UpdateMouseData() {
    SdlData* bd = GetBackendData();
    ImGuiIO& io = ImGui::GetIO();

    SDL_CaptureMouse((bd->mouse_buttons_down != 0) ? true : false);
    SDL_Window* focused_window = SDL_GetKeyboardFocus();
    const bool is_app_focused = (bd->window == focused_window);

    if (is_app_focused) {
        if (io.WantSetMousePos)
            SDL_WarpMouseInWindow(bd->window, io.MousePos.x, io.MousePos.y);

        if (bd->mouse_buttons_down == 0) {
            float mouse_x_global, mouse_y_global;
            int window_x, window_y;
            SDL_GetGlobalMouseState(&mouse_x_global, &mouse_y_global);
            SDL_GetWindowPosition(focused_window, &window_x, &window_y);
            mouse_x_global -= (float)window_x;
            mouse_y_global -= (float)window_y;
            io.AddMousePosEvent(mouse_x_global, mouse_y_global);
            if (mouse_x_global != bd->prev_mouse_pos.x || mouse_y_global != bd->prev_mouse_pos.y &&
                                                              bd->prev_mouse_pos.y == 0 &&
                                                              bd->prev_mouse_pos.x == 0) {
                bd->prev_mouse_pos.x = mouse_x_global;
                bd->prev_mouse_pos.y = mouse_y_global;
                bd->lastCursorMoveTime = bd->time;
            }
        }
    }
}

static void UpdateMouseCursor() {
    ImGuiIO& io = ImGui::GetIO();
    if (io.ConfigFlags & ImGuiConfigFlags_NoMouseCursorChange)
        return;
    SdlData* bd = GetBackendData();

    s16 cursorState = Config::getCursorState();
    ImGuiMouseCursor imgui_cursor = ImGui::GetMouseCursor();
    if (io.MouseDrawCursor || imgui_cursor == ImGuiMouseCursor_None ||
        cursorState == Config::HideCursorState::Always) {
        SDL_HideCursor();

    } else if (cursorState == Config::HideCursorState::Idle &&
               bd->time - bd->lastCursorMoveTime >=
                   Config::getCursorHideTimeout() * SDL_GetPerformanceFrequency()) {

        bool wasCursorVisible = SDL_CursorVisible();
        SDL_HideCursor();

        if (wasCursorVisible) {
            SDL_WarpMouseInWindow(SDL_GetKeyboardFocus(), bd->prev_mouse_pos.x,
                                  bd->prev_mouse_pos.y);
        }

    } else {
        SDL_Cursor* expected_cursor = bd->mouse_cursors[imgui_cursor]
                                          ? bd->mouse_cursors[imgui_cursor]
                                          : bd->mouse_cursors[ImGuiMouseCursor_Arrow];
        if (bd->mouse_last_cursor != expected_cursor) {
            SDL_SetCursor(expected_cursor);
            bd->mouse_last_cursor = expected_cursor;
        }
        SDL_ShowCursor();
    }
}

static void CloseGamepads() {
    SdlData* bd = GetBackendData();
    if (bd->gamepad_mode != ImGui_ImplSDL3_GamepadMode_Manual)
        for (SDL_Gamepad* gamepad : bd->gamepads)
            SDL_CloseGamepad(gamepad);
    bd->gamepads.resize(0);
}

void SetGamepadMode(GamepadMode mode, SDL_Gamepad** manual_gamepads_array,
                    int manual_gamepads_count) {
    SdlData* bd = GetBackendData();
    CloseGamepads();
    if (mode == ImGui_ImplSDL3_GamepadMode_Manual) {
        IM_ASSERT(manual_gamepads_array != nullptr && manual_gamepads_count > 0);
        for (int n = 0; n < manual_gamepads_count; n++)
            bd->gamepads.push_back(manual_gamepads_array[n]);
    } else {
        IM_ASSERT(manual_gamepads_array == nullptr && manual_gamepads_count <= 0);
        bd->want_update_gamepads_list = true;
    }
    bd->gamepad_mode = mode;
}

static void UpdateGamepadButton(SdlData* bd, ImGuiIO& io, ImGuiKey key,
                                SDL_GamepadButton button_no) {
    bool merged_value = false;
    for (SDL_Gamepad* gamepad : bd->gamepads)
        merged_value |= SDL_GetGamepadButton(gamepad, button_no) != 0;
    io.AddKeyEvent(key, merged_value);
}

static inline float Saturate(float v) {
    return v < 0.0f ? 0.0f : v > 1.0f ? 1.0f : v;
}
static void UpdateGamepadAnalog(SdlData* bd, ImGuiIO& io, ImGuiKey key, SDL_GamepadAxis axis_no,
                                float v0, float v1) {
    float merged_value = 0.0f;
    for (SDL_Gamepad* gamepad : bd->gamepads) {
        float vn = Saturate((float)(SDL_GetGamepadAxis(gamepad, axis_no) - v0) / (float)(v1 - v0));
        if (merged_value < vn)
            merged_value = vn;
    }
    io.AddKeyAnalogEvent(key, merged_value > 0.1f, merged_value);
}

static void UpdateGamepads() {
    ImGuiIO& io = ImGui::GetIO();
    SdlData* bd = GetBackendData();

    auto controller = Common::Singleton<Input::GameController>::Instance();
    auto engine = controller->GetEngine();
    SDL_Gamepad* SDLGamepad = engine->m_gamepad;
    if (bd->want_update_gamepads_list && bd->gamepad_mode != ImGui_ImplSDL3_GamepadMode_Manual) {
        if (SDLGamepad) {
            bd->gamepads.push_back(SDLGamepad);
            bd->want_update_gamepads_list = false;
        } else {
            CloseGamepads();
            int sdl_gamepads_count = 0;
            const SDL_JoystickID* sdl_gamepads = SDL_GetGamepads(&sdl_gamepads_count);
            for (int n = 0; n < sdl_gamepads_count; n++)
                if (SDL_Gamepad* gamepad = SDL_OpenGamepad(sdl_gamepads[n])) {
                    bd->gamepads.push_back(gamepad);
                    if (bd->gamepad_mode == ImGui_ImplSDL3_GamepadMode_AutoFirst)
                        break;
                }
            bd->want_update_gamepads_list = false;
        }
    }

    if ((io.ConfigFlags & ImGuiConfigFlags_NavEnableGamepad) == 0)
        return;
    io.BackendFlags &= ~ImGuiBackendFlags_HasGamepad;
    if (bd->gamepads.Size == 0)
        return;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;

    const int thumb_dead_zone = 8000;
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadStart, SDL_GAMEPAD_BUTTON_START);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadBack, SDL_GAMEPAD_BUTTON_BACK);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadFaceRight,
                        SDL_GAMEPAD_BUTTON_EAST);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadFaceUp,
                        SDL_GAMEPAD_BUTTON_NORTH);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadFaceDown,
                        SDL_GAMEPAD_BUTTON_SOUTH);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadDpadLeft, SDL_GAMEPAD_BUTTON_DPAD_LEFT);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadDpadRight, SDL_GAMEPAD_BUTTON_DPAD_RIGHT);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadDpadUp, SDL_GAMEPAD_BUTTON_DPAD_UP);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadDpadDown, SDL_GAMEPAD_BUTTON_DPAD_DOWN);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadL1, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadR1, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadL2, SDL_GAMEPAD_AXIS_LEFT_TRIGGER, 0.0f, 32767);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadR2, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, 0.0f, 32767);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadL3, SDL_GAMEPAD_BUTTON_LEFT_STICK);
    UpdateGamepadButton(bd, io, ImGuiKey_GamepadR3, SDL_GAMEPAD_BUTTON_RIGHT_STICK);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadLStickLeft, SDL_GAMEPAD_AXIS_LEFTX,
                        -thumb_dead_zone, -32768);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadLStickRight, SDL_GAMEPAD_AXIS_LEFTX,
                        +thumb_dead_zone, +32767);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadLStickUp, SDL_GAMEPAD_AXIS_LEFTY, -thumb_dead_zone,
                        -32768);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadLStickDown, SDL_GAMEPAD_AXIS_LEFTY,
                        +thumb_dead_zone, +32767);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadRStickLeft, SDL_GAMEPAD_AXIS_RIGHTX,
                        -thumb_dead_zone, -32768);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadRStickRight, SDL_GAMEPAD_AXIS_RIGHTX,
                        +thumb_dead_zone, +32767);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadRStickUp, SDL_GAMEPAD_AXIS_RIGHTY, -thumb_dead_zone,
                        -32768);
    UpdateGamepadAnalog(bd, io, ImGuiKey_GamepadRStickDown, SDL_GAMEPAD_AXIS_RIGHTY,
                        +thumb_dead_zone, +32767);
}

void NewFrame(bool is_reusing_frame) {
    SdlData* bd = GetBackendData();
    IM_ASSERT(bd != nullptr && "No platform backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();

    static Uint64 frequency = SDL_GetPerformanceFrequency();
    Uint64 current_time = SDL_GetPerformanceCounter();
    if (current_time <= bd->time)
        current_time = bd->time + 1;
    io.DeltaTime = bd->time > 0 ? (float)((double)(current_time - bd->time) / (double)frequency)
                                : 1.0f / 60.0f;
    bd->time = current_time;

    if (!is_reusing_frame) {
        if (current_time <= bd->nonReusedtime)
            current_time = bd->nonReusedtime + 1;
        float deltaTime =
            bd->nonReusedtime > 0
                ? (float)((double)(current_time - bd->nonReusedtime) / (double)frequency)
                : 1.0f / 60.0f;
        bd->nonReusedtime = current_time;
        DebugState.FrameDeltaTime = deltaTime;

        int& frameIdx = bd->framerateSecPerFrameIdx;
        float& framerateSec = bd->framerateSecPerFrame[frameIdx];
        float& acc = bd->framerateSecPerFrameAcc;
        int count = bd->framerateSecPerFrame.size();
        acc += deltaTime - framerateSec;
        framerateSec = deltaTime;
        frameIdx = (frameIdx + 1) % count;
        DebugState.Framerate = acc > 0.0f ? 1.0f / (acc / (float)count) : FLT_MAX;
    }

    if (bd->mouse_pending_leave_frame && bd->mouse_pending_leave_frame >= ImGui::GetFrameCount() &&
        bd->mouse_buttons_down == 0) {
        bd->mouse_window_id = 0;
        bd->mouse_pending_leave_frame = 0;
        io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
    }

    UpdateMouseData();
    UpdateMouseCursor();

    UpdateGamepads();
}

void OnResize() {
    SdlData* bd = GetBackendData();
    ImGuiIO& io = ImGui::GetIO();

    int w, h;
    int display_w, display_h;
    SDL_GetWindowSize(bd->window, &w, &h);
    if (SDL_GetWindowFlags(bd->window) & SDL_WINDOW_MINIMIZED) {
        w = h = 0;
    }
    SDL_GetWindowSizeInPixels(bd->window, &display_w, &display_h);
    io.DisplaySize = ImVec2((float)w, (float)h);
    if (w > 0 && h > 0) {
        io.DisplayFramebufferScale = {(float)display_w / (float)w, (float)display_h / (float)h};
    }
}

}
