// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstdarg>
#include <filesystem>
#include <mutex>

#ifdef __ANDROID__
#include <android/log.h>
#endif

#include "common/config.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "core/libraries/libs.h"
#include "core/libraries/pad/pad_errors.h"
#include "input/controller.h"
#include "pad.h"

namespace Libraries::Pad {

using Input::GameController;

static bool g_initialized = false;
static bool g_opened = false;
static std::mutex g_executor_pad_latch_mutex;
static u32 g_executor_host_down_buttons = 0;
static u32 g_executor_guest_observed_down_buttons = 0;
static u32 g_executor_pending_tap_buttons = 0;
static std::atomic<bool> g_executor_ui_launch_pending{false};
static std::atomic<u32> g_executor_ui_nav_pending{0};
static std::atomic<u32> g_executor_live_auto_pad_reads{0};

#ifdef __ANDROID__
static std::atomic_int g_android_pad_log_budget{8};

static bool AndroidPadTraceEnabled() {
    static const bool enabled = std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr ||
                                std::getenv("EXECUTOR_TRACE_PAD") != nullptr;
    return enabled;
}

static void AndroidPadLog(const char* format, ...) {
    va_list args;
    va_start(args, format);
    __android_log_vprint(ANDROID_LOG_INFO, "LSX4Native", format, args);
    va_end(args);
}

static void TracePadRead(const char* api, s32 handle, const OrbisPadData* data, s32 result) {
    const bool has_input = data && (static_cast<u32>(data[0].buttons) != 0 ||
                                    data[0].leftStick.x != 128 ||
                                    data[0].leftStick.y != 128 || data[0].rightStick.x != 128 ||
                                    data[0].rightStick.y != 128);
    if (!has_input && g_android_pad_log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }

    if (!data) {
        AndroidPadLog("[EXECUTOR_PAD_READ] api=%s handle=%d result=%d data=null", api, handle,
                      result);
        return;
    }

    AndroidPadLog("[EXECUTOR_PAD_READ] api=%s handle=%d result=%d connected=%d "
                  "connectedCount=%u buttons=0x%08x ls=%u,%u rs=%u,%u l2=%u r2=%u "
                  "touchNum=%u ts=%llu",
                  api, handle, result, data[0].connected ? 1 : 0, data[0].connectedCount,
                  data[0].buttons, data[0].leftStick.x, data[0].leftStick.y,
                  data[0].rightStick.x, data[0].rightStick.y, data[0].analogButtons.l2,
                  data[0].analogButtons.r2, data[0].touchData.touchNum,
                  static_cast<unsigned long long>(data[0].timestamp));
}
#else
static void AndroidPadLog(const char*, ...) {}
static void TracePadRead(const char*, s32, const OrbisPadData*, s32) {}
#endif

static bool ExecutorLiveAutoPadEnabled() {
    static int cached = -1;
    if (cached >= 0) {
        return cached != 0;
    }
    const char* root = std::getenv("EXECUTOR_LSX4_RUNTIME_ROOT");
    if (!root || root[0] == '\0') {
        cached = 0;
        return false;
    }
    std::error_code ec;
    const auto path = std::filesystem::path(root) / "run-live-pad-auto-cross";
    cached = std::filesystem::exists(path, ec) && !ec ? 1 : 0;
    if (cached != 0) {
        AndroidPadLog("[EXECUTOR_PAD_AUTO] source=sentinel path=%s enabled=1", path.string().c_str());
    }
    return cached != 0;
}

static u32 ExecutorLiveAutoPadMaskForRead() {
    if (!ExecutorLiveAutoPadEnabled()) {
        return 0;
    }
    const u32 read = g_executor_live_auto_pad_reads.fetch_add(1, std::memory_order_relaxed);
    if (read < 40 || (read >= 80 && read < 120)) {
        return static_cast<u32>(OrbisPadButtonDataOffset::Cross);
    }
    if (read < 80) {
        return static_cast<u32>(OrbisPadButtonDataOffset::Options);
    }
    return 0;
}

void ExecutorPadRecordHostButtonEvent(u32 button_mask, bool pressed) {
    if (pressed && (button_mask & static_cast<u32>(OrbisPadButtonDataOffset::Cross)) != 0) {
        g_executor_ui_launch_pending.store(true, std::memory_order_relaxed);
        AndroidPadLog("[EXECUTOR_PAD_LATCH] Cross press -> UI launch request latched");
    }
    if (pressed) {
        const u32 nav = button_mask & (static_cast<u32>(OrbisPadButtonDataOffset::Up) |
                                       static_cast<u32>(OrbisPadButtonDataOffset::Down) |
                                       static_cast<u32>(OrbisPadButtonDataOffset::Left) |
                                       static_cast<u32>(OrbisPadButtonDataOffset::Right) |
                                       static_cast<u32>(OrbisPadButtonDataOffset::Circle));
        if (nav != 0) {
            g_executor_ui_nav_pending.fetch_or(nav, std::memory_order_relaxed);
            AndroidPadLog("[EXECUTOR_PAD_LATCH] UI nav press latched mask=0x%08x", nav);
        }
    }
    std::scoped_lock lock{g_executor_pad_latch_mutex};
    if (pressed) {
        g_executor_host_down_buttons |= button_mask;
        g_executor_guest_observed_down_buttons &= ~button_mask;
    } else {
        if ((g_executor_guest_observed_down_buttons & button_mask) == 0) {
            g_executor_pending_tap_buttons |= button_mask;
            AndroidPadLog("[EXECUTOR_PAD_LATCH] queued buttons=0x%08x pending=0x%08x",
                          button_mask, g_executor_pending_tap_buttons);
        }
        g_executor_host_down_buttons &= ~button_mask;
        g_executor_guest_observed_down_buttons &= ~button_mask;
    }
}

bool ExecutorConsumeUiLaunchRequest() {
    return g_executor_ui_launch_pending.exchange(false, std::memory_order_relaxed);
}

u32 ExecutorConsumeUiNavButtons() {
    return g_executor_ui_nav_pending.exchange(0, std::memory_order_relaxed);
}

static void ApplyExecutorPadTapLatch(Input::State* states, s32 num) {
    if (states == nullptr || num <= 0) {
        return;
    }

    std::scoped_lock lock{g_executor_pad_latch_mutex};
    const u32 inject = g_executor_pending_tap_buttons;
    if (inject == 0) {
        return;
    }

    const auto current = static_cast<u32>(states[0].buttonsState);
    states[0].buttonsState = static_cast<OrbisPadButtonDataOffset>(current | inject);
    g_executor_pending_tap_buttons &= ~inject;
    AndroidPadLog("[EXECUTOR_PAD_LATCH] inject buttons=0x%08x state=0x%08x", inject,
                  current | inject);
}

static void MarkExecutorPadObservedButtons(OrbisPadButtonDataOffset buttons) {
    std::scoped_lock lock{g_executor_pad_latch_mutex};
    g_executor_guest_observed_down_buttons |=
        static_cast<u32>(buttons) & g_executor_host_down_buttons;
}

int PS4_SYSV_ABI scePadClose(s32 handle) {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadConnectPort() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadDeviceClassGetExtendedInformation(
    s32 handle, OrbisPadDeviceClassExtendedInformation* pExtInfo) {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    std::memset(pExtInfo, 0, sizeof(OrbisPadDeviceClassExtendedInformation));
    if (Config::getUseSpecialPad()) {
        pExtInfo->deviceClass = (OrbisPadDeviceClass)Config::getSpecialPadClass();
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadDeviceClassParseData(s32 handle, const OrbisPadData* pData,
                                            OrbisPadDeviceClassData* pDeviceClassData) {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadDeviceOpen() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadDisableVibration() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadDisconnectDevice() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadDisconnectPort() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadEnableAutoDetect() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadEnableExtensionPort() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadEnableSpecificDeviceClass() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadEnableUsbConnection() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetBluetoothAddress() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetCapability() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetControllerInformation(s32 handle, OrbisPadControllerInformation* pInfo) {
    if (AndroidPadTraceEnabled()) {
        AndroidPadLog("[EXECUTOR_PAD_TRACE] enter scePadGetControllerInformation handle=%d pInfo=%p",
                      handle, (void*)pInfo);
    }
    if (!pInfo) {
        return ORBIS_OK;
    }
    LOG_DEBUG(Lib_Pad, "called handle = {}", handle);
    pInfo->touchPadInfo.pixelDensity = 1;
    pInfo->touchPadInfo.resolution.x = 1920;
    pInfo->touchPadInfo.resolution.y = 950;
    pInfo->stickInfo.deadZoneLeft = 1;
    pInfo->stickInfo.deadZoneRight = 1;
    pInfo->connectionType = ORBIS_PAD_PORT_TYPE_STANDARD;
    pInfo->connectedCount = 1;
    pInfo->deviceClass = OrbisPadDeviceClass::Standard;
    if (handle < 0) {
        pInfo->connected = false;
        return ORBIS_OK;
    }
    pInfo->connected = true;
    if (Config::getUseSpecialPad()) {
        pInfo->connectionType = ORBIS_PAD_PORT_TYPE_SPECIAL;
        pInfo->deviceClass = (OrbisPadDeviceClass)Config::getSpecialPadClass();
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetDataInternal() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetDeviceId() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetDeviceInfo() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetExtControllerInformation(s32 handle,
                                                   OrbisPadExtendedControllerInformation* pInfo) {
    LOG_INFO(Lib_Pad, "called handle = {}", handle);

    pInfo->padType1 = 0;
    pInfo->padType2 = 0;
    pInfo->capability = 0;

    auto res = scePadGetControllerInformation(handle, &pInfo->base);
    return res;
}

int PS4_SYSV_ABI scePadGetExtensionUnitInfo() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetFeatureReport() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetHandle(Libraries::UserService::OrbisUserServiceUserId userId, s32 type,
                                 s32 index) {
    if (!g_initialized) {
        AndroidPadLog("[EXECUTOR_PAD_HANDLE] auto-init for scePadGetHandle user=%d type=%d index=%d",
                      static_cast<int>(userId), type, index);
        g_initialized = true;
    }
    if (userId == -1 || !g_opened) {
        AndroidPadLog("[EXECUTOR_PAD_HANDLE] dummy-open for scePadGetHandle user=%d type=%d index=%d",
                      static_cast<int>(userId), type, index);
        g_opened = true;
    }
    LOG_DEBUG(Lib_Pad, "(DUMMY) called");
    return 1;
}

int PS4_SYSV_ABI scePadGetIdleCount() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetInfo() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetInfoByPortType() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetLicenseControllerInformation() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetMotionSensorPosition() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetMotionTimerUnit() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetSphereRadius() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadGetVersionInfo() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadInit() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    g_initialized = true;
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadIsBlasterConnected() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadIsDS4Connected() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadIsLightBarBaseBrightnessControllable() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadIsMoveConnected() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadIsMoveReproductionModel() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadIsValidHandle() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadMbusInit() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadMbusTerm() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadOpen(Libraries::UserService::OrbisUserServiceUserId userId, s32 type,
                            s32 index, const OrbisPadOpenParam* pParam) {
    if (!g_initialized) {
        AndroidPadLog("[EXECUTOR_PAD_OPEN] auto-init user=%d type=%d index=%d",
                      static_cast<int>(userId), type, index);
        g_initialized = true;
    }
    if (userId == -1) {
        return ORBIS_PAD_ERROR_DEVICE_NO_HANDLE;
    }
    if (Config::getUseSpecialPad()) {
        if (type != ORBIS_PAD_PORT_TYPE_SPECIAL)
            return ORBIS_PAD_ERROR_DEVICE_NOT_CONNECTED;
    } else {
        if (type != ORBIS_PAD_PORT_TYPE_STANDARD && type != ORBIS_PAD_PORT_TYPE_REMOTE_CONTROL)
            return ORBIS_PAD_ERROR_DEVICE_NOT_CONNECTED;
    }
    LOG_INFO(Lib_Pad, "(DUMMY) called user_id = {} type = {} index = {}", userId, type, index);
    AndroidPadLog("[EXECUTOR_PAD_OPEN] api=scePadOpen user=%d type=%d index=%d handle=1",
                  static_cast<int>(userId), type, index);
    g_opened = true;
    scePadResetLightBar(userId);
    scePadResetOrientation(userId);
    return 1;
}

int PS4_SYSV_ABI scePadOpenExt(Libraries::UserService::OrbisUserServiceUserId userId, s32 type,
                               s32 index, const OrbisPadOpenExtParam* pParam) {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    if (Config::getUseSpecialPad()) {
        if (type != ORBIS_PAD_PORT_TYPE_SPECIAL)
            return ORBIS_PAD_ERROR_DEVICE_NOT_CONNECTED;
    } else {
        if (type != ORBIS_PAD_PORT_TYPE_STANDARD && type != ORBIS_PAD_PORT_TYPE_REMOTE_CONTROL)
            return ORBIS_PAD_ERROR_DEVICE_NOT_CONNECTED;
    }
    AndroidPadLog("[EXECUTOR_PAD_OPEN] api=scePadOpenExt user=%d type=%d index=%d handle=1",
                  static_cast<int>(userId), type, index);
    g_opened = true;
    return 1;
}

int PS4_SYSV_ABI scePadOpenExt2() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadOutputReport() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int ProcessStates(s32 handle, OrbisPadData* pData, Input::State* states, s32 num, bool connected,
                  u32 connected_count) {
    if (!connected) {
        pData[0] = {};
        pData[0].orientation = {0.0f, 0.0f, 0.0f, 1.0f};
        pData[0].connected = false;
        return 1;
    }

    ApplyExecutorPadTapLatch(states, num);

    for (int i = 0; i < num; i++) {
        pData[i].buttons = states[i].buttonsState;
        pData[i].leftStick.x = states[i].axes[static_cast<int>(Input::Axis::LeftX)];
        pData[i].leftStick.y = states[i].axes[static_cast<int>(Input::Axis::LeftY)];
        pData[i].rightStick.x = states[i].axes[static_cast<int>(Input::Axis::RightX)];
        pData[i].rightStick.y = states[i].axes[static_cast<int>(Input::Axis::RightY)];
        pData[i].analogButtons.l2 = states[i].axes[static_cast<int>(Input::Axis::TriggerLeft)];
        pData[i].analogButtons.r2 = states[i].axes[static_cast<int>(Input::Axis::TriggerRight)];
        pData[i].acceleration.x = states[i].acceleration.x * 0.098;
        pData[i].acceleration.y = states[i].acceleration.y * 0.098;
        pData[i].acceleration.z = states[i].acceleration.z * 0.098;
        pData[i].angularVelocity.x = states[i].angularVelocity.x;
        pData[i].angularVelocity.y = states[i].angularVelocity.y;
        pData[i].angularVelocity.z = states[i].angularVelocity.z;
        pData[i].orientation = {0.0f, 0.0f, 0.0f, 1.0f};

        auto* controller = Common::Singleton<GameController>::Instance();
        const auto* engine = controller->GetEngine();
        if (engine && handle == 1) {
            const auto gyro_poll_rate = engine->GetAccelPollRate();
            if (gyro_poll_rate != 0.0f) {
                auto now = std::chrono::steady_clock::now();
                float deltaTime = std::chrono::duration_cast<std::chrono::microseconds>(
                                      now - controller->GetLastUpdate())
                                      .count() /
                                  1000000.0f;
                controller->SetLastUpdate(now);
                Libraries::Pad::OrbisFQuaternion lastOrientation = controller->GetLastOrientation();
                Libraries::Pad::OrbisFQuaternion outputOrientation = {0.0f, 0.0f, 0.0f, 1.0f};
                GameController::CalculateOrientation(pData->acceleration, pData->angularVelocity,
                                                     deltaTime, lastOrientation, outputOrientation);
                pData[i].orientation = outputOrientation;
                controller->SetLastOrientation(outputOrientation);
            }
        }
        pData[i].touchData.touchNum =
            (states[i].touchpad[0].state ? 1 : 0) + (states[i].touchpad[1].state ? 1 : 0);

        if (handle == 1) {
            if (controller->GetTouchCount() >= 127) {
                controller->SetTouchCount(0);
            }

            if (controller->GetSecondaryTouchCount() >= 127) {
                controller->SetSecondaryTouchCount(0);
            }

            if (pData->touchData.touchNum == 1 && controller->GetPreviousTouchNum() == 0) {
                controller->SetTouchCount(controller->GetTouchCount() + 1);
                controller->SetSecondaryTouchCount(controller->GetTouchCount());
            } else if (pData->touchData.touchNum == 2 && controller->GetPreviousTouchNum() == 1) {
                controller->SetSecondaryTouchCount(controller->GetSecondaryTouchCount() + 1);
            } else if (pData->touchData.touchNum == 0 && controller->GetPreviousTouchNum() > 0) {
                if (controller->GetTouchCount() < controller->GetSecondaryTouchCount()) {
                    controller->SetTouchCount(controller->GetSecondaryTouchCount());
                } else {
                    if (controller->WasSecondaryTouchReset()) {
                        controller->SetTouchCount(controller->GetSecondaryTouchCount());
                        controller->UnsetSecondaryTouchResetBool();
                    }
                }
            }

            controller->SetPreviousTouchNum(pData->touchData.touchNum);

            if (pData->touchData.touchNum == 1) {
                states[i].touchpad[0].ID = controller->GetTouchCount();
                states[i].touchpad[1].ID = 0;
            } else if (pData->touchData.touchNum == 2) {
                states[i].touchpad[0].ID = controller->GetTouchCount();
                states[i].touchpad[1].ID = controller->GetSecondaryTouchCount();
            }
        } else {
            states[i].touchpad[0].ID = 1;
            states[i].touchpad[1].ID = 2;
        }

        pData[i].touchData.touch[0].x = states[i].touchpad[0].x;
        pData[i].touchData.touch[0].y = states[i].touchpad[0].y;
        pData[i].touchData.touch[0].id = states[i].touchpad[0].ID;
        pData[i].touchData.touch[1].x = states[i].touchpad[1].x;
        pData[i].touchData.touch[1].y = states[i].touchpad[1].y;
        pData[i].touchData.touch[1].id = states[i].touchpad[1].ID;
        pData[i].connected = connected;
        pData[i].timestamp = states[i].time;
        pData[i].connectedCount = connected_count;
        pData[i].deviceUniqueDataLen = 0;
        MarkExecutorPadObservedButtons(pData[i].buttons);
    }

    return num;
}

int PS4_SYSV_ABI scePadRead(s32 handle, OrbisPadData* pData, s32 num) {
    if (AndroidPadTraceEnabled()) {
        AndroidPadLog("[EXECUTOR_PAD_TRACE] enter scePadRead handle=%d pData=%p num=%d", handle,
                      (void*)pData, num);
    }
    LOG_TRACE(Lib_Pad, "called");
    if (!g_initialized) {
        AndroidPadLog("[EXECUTOR_PAD_TRACE] auto-init for scePadRead");
        g_initialized = true;
    }
    bool executor_dummy_handle = false;
    if (handle == ORBIS_PAD_ERROR_NOT_INITIALIZED || handle == ORBIS_PAD_ERROR_DEVICE_NO_HANDLE) {
        AndroidPadLog("[EXECUTOR_PAD_TRACE] normalizing error handle=0x%08x to dummy handle=1",
                      static_cast<u32>(handle));
        handle = 1;
        g_opened = true;
        executor_dummy_handle = true;
    }
    if (handle < 1 || !pData) {
        return ORBIS_PAD_ERROR_INVALID_HANDLE;
    }
    const s32 safe_num = std::clamp(num, 1, 64);
    int connected_count = 0;
    bool connected = false;
    std::vector<Input::State> states(64);
    auto* controller = Common::Singleton<GameController>::Instance();
    const auto* engine = controller->GetEngine();
    int ret_num = controller->ReadStates(states.data(), safe_num, &connected, &connected_count);
    ret_num = std::clamp(ret_num, 1, safe_num);
    const int result = ProcessStates(handle, pData, states.data(), ret_num, connected, connected_count);
    const u32 auto_mask = ExecutorLiveAutoPadMaskForRead();
    if (auto_mask != 0 && result > 0 && pData[0].connected) {
        pData[0].buttons = static_cast<OrbisPadButtonDataOffset>(
            static_cast<u32>(pData[0].buttons) | auto_mask);
        AndroidPadLog("[EXECUTOR_PAD_AUTO] read=%u mask=0x%08x buttons=0x%08x",
                      g_executor_live_auto_pad_reads.load(std::memory_order_relaxed), auto_mask,
                      static_cast<u32>(pData[0].buttons));
    }
    if (executor_dummy_handle) {
        pData[0].connected = false;
        pData[0].connectedCount = 0;
        TracePadRead("scePadRead", handle, pData, 0);
        AndroidPadLog("[EXECUTOR_PAD_READ] dummy normalized handle: returning zero samples");
        return 0;
    }
    TracePadRead("scePadRead", handle, pData, result);
    return result;
}

int PS4_SYSV_ABI scePadReadBlasterForTracker() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadReadExt(s32 handle, OrbisPadData* pData, s32 num) {
    LOG_TRACE(Lib_Pad, "called handle = {} num = {}", handle, num);
    const int result = scePadRead(handle, pData, num);
    TracePadRead("scePadReadExt", handle, pData, result);
    return result;
}

int PS4_SYSV_ABI scePadReadForTracker() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadReadHistory() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadReadState(s32 handle, OrbisPadData* pData) {
    if (AndroidPadTraceEnabled()) {
        AndroidPadLog("[EXECUTOR_PAD_TRACE] enter scePadReadState handle=%d pData=%p", handle,
                      (void*)pData);
    }
    LOG_TRACE(Lib_Pad, "called");
    if (handle < 1 || !pData) {
        return ORBIS_PAD_ERROR_INVALID_HANDLE;
    }
    auto* controller = Common::Singleton<GameController>::Instance();
    const auto* engine = controller->GetEngine();
    int connected_count = 0;
    bool connected = false;
    Input::State state;
    controller->ReadState(&state, &connected, &connected_count);
    ProcessStates(handle, pData, &state, 1, connected, connected_count);
    TracePadRead("scePadReadState", handle, pData, ORBIS_OK);
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadReadStateExt(s32 handle, OrbisPadData* pData) {
    LOG_TRACE(Lib_Pad, "called handle = {}", handle);
    const int result = scePadReadState(handle, pData);
    TracePadRead("scePadReadStateExt", handle, pData, result);
    return result;
}

int PS4_SYSV_ABI scePadResetLightBar(s32 handle) {
    LOG_INFO(Lib_Pad, "(DUMMY) called");
    if (handle != 1) {
        return ORBIS_PAD_ERROR_INVALID_HANDLE;
    }
    auto* controller = Common::Singleton<GameController>::Instance();
    int* rgb = Config::GetControllerCustomColor();
    controller->SetLightBarRGB(rgb[0], rgb[1], rgb[2]);
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadResetLightBarAll() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadResetLightBarAllByPortType() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadResetOrientation(s32 handle) {
    LOG_INFO(Lib_Pad, "scePadResetOrientation called handle = {}", handle);

    if (handle != 1) {
        return ORBIS_PAD_ERROR_INVALID_HANDLE;
    }

    auto* controller = Common::Singleton<GameController>::Instance();
    Libraries::Pad::OrbisFQuaternion defaultOrientation = {0.0f, 0.0f, 0.0f, 1.0f};
    controller->SetLastOrientation(defaultOrientation);
    controller->SetLastUpdate(std::chrono::steady_clock::now());

    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadResetOrientationForTracker() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetAngularVelocityDeadbandState(s32 handle, bool bEnable) {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetAutoPowerOffCount() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetButtonRemappingInfo() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetConnection() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetExtensionReport() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetFeatureReport() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetForceIntercepted() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetLightBar(s32 handle, const OrbisPadLightBarParam* pParam) {
    if (Config::GetOverrideControllerColor()) {
        return ORBIS_OK;
    }
    if (pParam != nullptr) {
        LOG_DEBUG(Lib_Pad, "called handle = {} rgb = {} {} {}", handle, pParam->r, pParam->g,
                  pParam->b);

        if (pParam->r < 0xD && pParam->g < 0xD && pParam->b < 0xD) {
            LOG_INFO(Lib_Pad, "Invalid lightbar setting");
            return ORBIS_PAD_ERROR_INVALID_LIGHTBAR_SETTING;
        }

        auto* controller = Common::Singleton<GameController>::Instance();
        controller->SetLightBarRGB(pParam->r, pParam->g, pParam->b);
        return ORBIS_OK;
    }
    return ORBIS_PAD_ERROR_INVALID_ARG;
}

int PS4_SYSV_ABI scePadSetLightBarBaseBrightness() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetLightBarBlinking() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetLightBarForTracker() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetLoginUserNumber() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetMotionSensorState(s32 handle, bool bEnable) {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetProcessFocus() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetProcessPrivilege() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetProcessPrivilegeOfButtonRemapping() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetShareButtonMaskForRemotePlay() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetTiltCorrectionState(s32 handle, bool bEnable) {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetUserColor() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetVibration(s32 handle, const OrbisPadVibrationParam* pParam) {
    if (pParam != nullptr) {
        LOG_DEBUG(Lib_Pad, "scePadSetVibration called handle = {} data = {} , {}", handle,
                  pParam->smallMotor, pParam->largeMotor);
        auto* controller = Common::Singleton<GameController>::Instance();
        controller->SetVibration(pParam->smallMotor, pParam->largeMotor);
        return ORBIS_OK;
    }
    return ORBIS_PAD_ERROR_INVALID_ARG;
}

int PS4_SYSV_ABI scePadSetVibrationForce() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSetVrTrackingMode() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadShareOutputData() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadStartRecording() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadStopRecording() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadSwitchConnection() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadVertualDeviceAddDevice() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadVirtualDeviceAddDevice() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadVirtualDeviceDeleteDevice() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadVirtualDeviceDisableButtonRemapping() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadVirtualDeviceGetRemoteSetting() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI scePadVirtualDeviceInsertData() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_28B998C7D8A3DA1D() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_298D21481F94C9FA() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_51E514BCD3A05CA5() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_89C9237E393DA243() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_EF103E845B6F0420() {
    LOG_ERROR(Lib_Pad, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("6ncge5+l5Qs", "libScePad", 1, "libScePad", scePadClose);
    LIB_FUNCTION("kazv1NzSB8c", "libScePad", 1, "libScePad", scePadConnectPort);
    LIB_FUNCTION("AcslpN1jHR8", "libScePad", 1, "libScePad",
                 scePadDeviceClassGetExtendedInformation);
    LIB_FUNCTION("IHPqcbc0zCA", "libScePad", 1, "libScePad", scePadDeviceClassParseData);
    LIB_FUNCTION("d7bXuEBycDI", "libScePad", 1, "libScePad", scePadDeviceOpen);
    LIB_FUNCTION("0aziJjRZxqQ", "libScePad", 1, "libScePad", scePadDisableVibration);
    LIB_FUNCTION("pnZXireDoeI", "libScePad", 1, "libScePad", scePadDisconnectDevice);
    LIB_FUNCTION("9ez71nWSvD0", "libScePad", 1, "libScePad", scePadDisconnectPort);
    LIB_FUNCTION("77ooWxGOIVs", "libScePad", 1, "libScePad", scePadEnableAutoDetect);
    LIB_FUNCTION("+cE4Jx431wc", "libScePad", 1, "libScePad", scePadEnableExtensionPort);
    LIB_FUNCTION("E1KEw5XMGQQ", "libScePad", 1, "libScePad", scePadEnableSpecificDeviceClass);
    LIB_FUNCTION("DD-KiRLBqkQ", "libScePad", 1, "libScePad", scePadEnableUsbConnection);
    LIB_FUNCTION("Q66U8FdrMaw", "libScePad", 1, "libScePad", scePadGetBluetoothAddress);
    LIB_FUNCTION("qtasqbvwgV4", "libScePad", 1, "libScePad", scePadGetCapability);
    LIB_FUNCTION("gjP9-KQzoUk", "libScePad", 1, "libScePad", scePadGetControllerInformation);
    LIB_FUNCTION("Uq6LgTJEmQs", "libScePad", 1, "libScePad", scePadGetDataInternal);
    LIB_FUNCTION("hDgisSGkOgw", "libScePad", 1, "libScePad", scePadGetDeviceId);
    LIB_FUNCTION("4rS5zG7RFaM", "libScePad", 1, "libScePad", scePadGetDeviceInfo);
    LIB_FUNCTION("hGbf2QTBmqc", "libScePad", 1, "libScePad", scePadGetExtControllerInformation);
    LIB_FUNCTION("1DmZjZAuzEM", "libScePad", 1, "libScePad", scePadGetExtensionUnitInfo);
    LIB_FUNCTION("PZSoY8j0Pko", "libScePad", 1, "libScePad", scePadGetFeatureReport);
    LIB_FUNCTION("u1GRHp+oWoY", "libScePad", 1, "libScePad", scePadGetHandle);
    LIB_FUNCTION("kiA9bZhbnAg", "libScePad", 1, "libScePad", scePadGetIdleCount);
    LIB_FUNCTION("1Odcw19nADw", "libScePad", 1, "libScePad", scePadGetInfo);
    LIB_FUNCTION("4x5Im8pr0-4", "libScePad", 1, "libScePad", scePadGetInfoByPortType);
    LIB_FUNCTION("vegw8qax5MI", "libScePad", 1, "libScePad", scePadGetLicenseControllerInformation);
    LIB_FUNCTION("WPIB7zBWxVE", "libScePad", 1, "libScePad", scePadGetMotionSensorPosition);
    LIB_FUNCTION("k4+nDV9vbT0", "libScePad", 1, "libScePad", scePadGetMotionTimerUnit);
    LIB_FUNCTION("do-JDWX+zRs", "libScePad", 1, "libScePad", scePadGetSphereRadius);
    LIB_FUNCTION("QuOaoOcSOw0", "libScePad", 1, "libScePad", scePadGetVersionInfo);
    LIB_FUNCTION("hv1luiJrqQM", "libScePad", 1, "libScePad", scePadInit);
    LIB_FUNCTION("bi0WNvZ1nug", "libScePad", 1, "libScePad", scePadIsBlasterConnected);
    LIB_FUNCTION("mEC+xJKyIjQ", "libScePad", 1, "libScePad", scePadIsDS4Connected);
    LIB_FUNCTION("d2Qk-i8wGak", "libScePad", 1, "libScePad",
                 scePadIsLightBarBaseBrightnessControllable);
    LIB_FUNCTION("4y9RNPSBsqg", "libScePad", 1, "libScePad", scePadIsMoveConnected);
    LIB_FUNCTION("9e56uLgk5y0", "libScePad", 1, "libScePad", scePadIsMoveReproductionModel);
    LIB_FUNCTION("pFTi-yOrVeQ", "libScePad", 1, "libScePad", scePadIsValidHandle);
    LIB_FUNCTION("CfwUlQtCFi4", "libScePad", 1, "libScePad", scePadMbusInit);
    LIB_FUNCTION("s7CvzS+9ZIs", "libScePad", 1, "libScePad", scePadMbusTerm);
    LIB_FUNCTION("xk0AcarP3V4", "libScePad", 1, "libScePad", scePadOpen);
    LIB_FUNCTION("WFIiSfXGUq8", "libScePad", 1, "libScePad", scePadOpenExt);
    LIB_FUNCTION("71E9e6n+2R8", "libScePad", 1, "libScePad", scePadOpenExt2);
    LIB_FUNCTION("DrUu8cPrje8", "libScePad", 1, "libScePad", scePadOutputReport);
    LIB_FUNCTION("q1cHNfGycLI", "libScePad", 1, "libScePad", scePadRead);
    LIB_FUNCTION("fm1r2vv5+OU", "libScePad", 1, "libScePad", scePadReadBlasterForTracker);
    LIB_FUNCTION("QjwkT2Ycmew", "libScePad", 1, "libScePad", scePadReadExt);
    LIB_FUNCTION("2NhkFTRnXHk", "libScePad", 1, "libScePad", scePadReadForTracker);
    LIB_FUNCTION("3u4M8ck9vJM", "libScePad", 1, "libScePad", scePadReadHistory);
    LIB_FUNCTION("YndgXqQVV7c", "libScePad", 1, "libScePad", scePadReadState);
    LIB_FUNCTION("5Wf4q349s+Q", "libScePad", 1, "libScePad", scePadReadStateExt);
    LIB_FUNCTION("DscD1i9HX1w", "libScePad", 1, "libScePad", scePadResetLightBar);
    LIB_FUNCTION("+4c9xRLmiXQ", "libScePad", 1, "libScePad", scePadResetLightBarAll);
    LIB_FUNCTION("+Yp6+orqf1M", "libScePad", 1, "libScePad", scePadResetLightBarAllByPortType);
    LIB_FUNCTION("rIZnR6eSpvk", "libScePad", 1, "libScePad", scePadResetOrientation);
    LIB_FUNCTION("jbAqAvLEP4A", "libScePad", 1, "libScePad", scePadResetOrientationForTracker);
    LIB_FUNCTION("r44mAxdSG+U", "libScePad", 1, "libScePad", scePadSetAngularVelocityDeadbandState);
    LIB_FUNCTION("ew647HuKi2Y", "libScePad", 1, "libScePad", scePadSetAutoPowerOffCount);
    LIB_FUNCTION("MbTt1EHYCTg", "libScePad", 1, "libScePad", scePadSetButtonRemappingInfo);
    LIB_FUNCTION("MLA06oNfF+4", "libScePad", 1, "libScePad", scePadSetConnection);
    LIB_FUNCTION("bsbHFI0bl5s", "libScePad", 1, "libScePad", scePadSetExtensionReport);
    LIB_FUNCTION("xqgVCEflEDY", "libScePad", 1, "libScePad", scePadSetFeatureReport);
    LIB_FUNCTION("lrjFx4xWnY8", "libScePad", 1, "libScePad", scePadSetForceIntercepted);
    LIB_FUNCTION("RR4novUEENY", "libScePad", 1, "libScePad", scePadSetLightBar);
    LIB_FUNCTION("dhQXEvmrVNQ", "libScePad", 1, "libScePad", scePadSetLightBarBaseBrightness);
    LIB_FUNCTION("etaQhgPHDRY", "libScePad", 1, "libScePad", scePadSetLightBarBlinking);
    LIB_FUNCTION("iHuOWdvQVpg", "libScePad", 1, "libScePad", scePadSetLightBarForTracker);
    LIB_FUNCTION("o-6Y99a8dKU", "libScePad", 1, "libScePad", scePadSetLoginUserNumber);
    LIB_FUNCTION("clVvL4ZDntw", "libScePad", 1, "libScePad", scePadSetMotionSensorState);
    LIB_FUNCTION("flYYxek1wy8", "libScePad", 1, "libScePad", scePadSetProcessFocus);
    LIB_FUNCTION("DmBx8K+jDWw", "libScePad", 1, "libScePad", scePadSetProcessPrivilege);
    LIB_FUNCTION("FbxEpTRDou8", "libScePad", 1, "libScePad",
                 scePadSetProcessPrivilegeOfButtonRemapping);
    LIB_FUNCTION("yah8Bk4TcYY", "libScePad", 1, "libScePad", scePadSetShareButtonMaskForRemotePlay);
    LIB_FUNCTION("vDLMoJLde8I", "libScePad", 1, "libScePad", scePadSetTiltCorrectionState);
    LIB_FUNCTION("z+GEemoTxOo", "libScePad", 1, "libScePad", scePadSetUserColor);
    LIB_FUNCTION("yFVnOdGxvZY", "libScePad", 1, "libScePad", scePadSetVibration);
    LIB_FUNCTION("8BOObG94-tc", "libScePad", 1, "libScePad", scePadSetVibrationForce);
    LIB_FUNCTION("--jrY4SHfm8", "libScePad", 1, "libScePad", scePadSetVrTrackingMode);
    LIB_FUNCTION("zFJ35q3RVnY", "libScePad", 1, "libScePad", scePadShareOutputData);
    LIB_FUNCTION("80XdmVYsNPA", "libScePad", 1, "libScePad", scePadStartRecording);
    LIB_FUNCTION("gAHvg6JPIic", "libScePad", 1, "libScePad", scePadStopRecording);
    LIB_FUNCTION("Oi7FzRWFr0Y", "libScePad", 1, "libScePad", scePadSwitchConnection);
    LIB_FUNCTION("0MB5x-ieRGI", "libScePad", 1, "libScePad", scePadVertualDeviceAddDevice);
    LIB_FUNCTION("N7tpsjWQ87s", "libScePad", 1, "libScePad", scePadVirtualDeviceAddDevice);
    LIB_FUNCTION("PFec14-UhEQ", "libScePad", 1, "libScePad", scePadVirtualDeviceDeleteDevice);
    LIB_FUNCTION("pjPCronWdxI", "libScePad", 1, "libScePad",
                 scePadVirtualDeviceDisableButtonRemapping);
    LIB_FUNCTION("LKXfw7VJYqg", "libScePad", 1, "libScePad", scePadVirtualDeviceGetRemoteSetting);
    LIB_FUNCTION("IWOyO5jKuZg", "libScePad", 1, "libScePad", scePadVirtualDeviceInsertData);
    LIB_FUNCTION("KLmYx9ij2h0", "libScePad", 1, "libScePad", Func_28B998C7D8A3DA1D);
    LIB_FUNCTION("KY0hSB+Uyfo", "libScePad", 1, "libScePad", Func_298D21481F94C9FA);
    LIB_FUNCTION("UeUUvNOgXKU", "libScePad", 1, "libScePad", Func_51E514BCD3A05CA5);
    LIB_FUNCTION("ickjfjk9okM", "libScePad", 1, "libScePad", Func_89C9237E393DA243);
    LIB_FUNCTION("7xA+hFtvBCA", "libScePad", 1, "libScePad", Func_EF103E845B6F0420);
};

}
