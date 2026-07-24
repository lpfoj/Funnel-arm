// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/parity_trace.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/libraries/gnmdriver/gnmdriver.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/videoout_error.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#include <cstdlib>

#ifdef __ANDROID__
#include <android/log.h>
#endif

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

extern "C" int executor_lsx4_runtime_present_guest_frame(const void* source, u32 width,
                                                            u32 height, u32 stride_bytes,
                                                            u32 pixel_format,
                                                            u32 tiling_mode)
    __attribute__((weak));

namespace Libraries::VideoOut {

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

#ifdef __ANDROID__
static bool ExecutorTraceLiveWide() {
    return std::getenv("EXECUTOR_TRACE_LIVE_WIDE_HOT") != nullptr;
}

static void ExecutorLogVideoOutPortState(const char* stage, const VideoOutPort* port,
                                         s32 requested_index, s32 effective_index, s64 flip_arg,
                                         bool is_eop) {
    if (!ExecutorTraceLiveWide() || !port) {
        return;
    }
    const auto& fs = port->flip_status;
    const auto labels = port->SnapshotVoLabels();
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_VIDEOOUT_STATE] stage=%s requested=%d effective=%d eop=%u arg=%lld pending=%d gc=%d count=%lld current=%d prev=%d labels=%llu,%llu,%llu,%llu reg=%d,%d,%d,%d",
        stage ? stage : "<null>", requested_index, effective_index, is_eop ? 1u : 0u,
        static_cast<long long>(flip_arg), fs.flip_pending_num, fs.gc_queue_num,
        static_cast<long long>(fs.count), fs.current_buffer, port->prev_index,
        static_cast<unsigned long long>(labels[0]),
        static_cast<unsigned long long>(labels[1]),
        static_cast<unsigned long long>(labels[2]),
        static_cast<unsigned long long>(labels[3]),
        port->buffer_slots[0].group_index, port->buffer_slots[1].group_index,
        port->buffer_slots[2].group_index, port->buffer_slots[3].group_index);
}

static bool ExecutorForceLiveFlipDoneEnabled() {
    return std::getenv("EXECUTOR_LIVE_FORCE_FLIP_DONE") != nullptr;
}

static void ExecutorForceLiveFlipDoneLabel(VideoOutPort* port, s32 index, const char* stage) {
    if (!ExecutorForceLiveFlipDoneEnabled() || !port || index < 0 ||
        index >= static_cast<s32>(MaxDisplayBuffers)) {
        return;
    }
    const auto old_value = port->StoreVoLabel(index, 0);
    const auto labels = port->SnapshotVoLabels();
    const auto& fs = port->flip_status;
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_FORCE_FLIP_DONE] op=release_current_label stage=%s index=%d old=%llu pending=%d gc=%d count=%lld current=%d prev=%d labels=%llu,%llu,%llu,%llu",
        stage ? stage : "<null>", index, static_cast<unsigned long long>(old_value),
        fs.flip_pending_num, fs.gc_queue_num, static_cast<long long>(fs.count), fs.current_buffer,
        port->prev_index, static_cast<unsigned long long>(labels[0]),
        static_cast<unsigned long long>(labels[1]),
        static_cast<unsigned long long>(labels[2]),
        static_cast<unsigned long long>(labels[3]));
}
#endif

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    if (main_port.is_open) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.is_open = true;
    if (liverpool) {
        liverpool->SetVoPort(&main_port);
    }
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock lock{mutex};

    main_port.is_open = false;
    main_port.flip_rate = 0;
    main_port.prev_index = -1;
    main_port.prev_ready_tick = 0;
    ASSERT(main_port.flip_events.empty());
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };

        port->StoreVoLabel(startIndex + i, 0);

        if (presenter) {
            presenter->RegisterVideoOutSurface(group, address);
        }
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

void VideoOutDriver::ReplayBufferRegistrationsToPresenter() {
    if (liverpool) {
        liverpool->SetVoPort(&main_port);
    }
    if (!presenter) {
        return;
    }
    std::scoped_lock lock{mutex};
    int replayed = 0;
    for (const auto& buffer : main_port.buffer_slots) {
        if (buffer.group_index < 0) {
            continue;
        }
        const auto& group = main_port.groups[buffer.group_index];
        if (!group.is_occupied) {
            continue;
        }
        presenter->RegisterVideoOutSurface(group, buffer.address_left);
        ++replayed;
    }
    LOG_INFO(Lib_VideoOut, "EXECUTOR_GNM_PRESENTER phase=replay_buffers count={}", replayed);
}

void VideoOutDriver::Flip(const Request& req) {
    if (!presenter) {
        return;
    }
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        return;
    }
    presenter->SetHDR(req.port->is_hdr);

    if (req.frame) {
        presenter->Present(req.frame);
    }

    auto* port = req.port;
    {
        std::unique_lock lock{port->port_mutex};
        auto& flip_status = port->flip_status;
        flip_status.count++;
        flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        flip_status.flip_arg = req.flip_arg;
        flip_status.current_buffer = req.index;
        if (req.eop) {
            --flip_status.gc_queue_num;
        }
        --flip_status.flip_pending_num;
    }
#ifdef __ANDROID__
    ExecutorLogVideoOutPortState("flip_after_status", port, req.index, req.index, req.flip_arg,
                                 req.eop);
#endif

    for (auto& event : port->flip_events) {
        if (event != nullptr) {
            event->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    const s32 release_index = port->prev_index;
    const u64 release_tick = port->prev_ready_tick;
    port->prev_index = req.index;
    port->prev_ready_tick = req.ready_tick;
    if (release_index != -1) {
        if (release_tick != 0) {
#ifdef __ANDROID__
            if (std::getenv("EXECUTOR_TRACE_LIVE_VIDEOOUT") != nullptr) {
                const auto* semaphore =
                    presenter->GetRasterizer().GetScheduler().GetMasterSemaphore();
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_VIDEOOUT_RETIRE] stage=queued index=%d tick=%llu current=%llu gpu=%llu",
                    release_index, static_cast<unsigned long long>(release_tick),
                    static_cast<unsigned long long>(semaphore->CurrentTick()),
                    static_cast<unsigned long long>(semaphore->KnownGpuTick()));
            }
#endif
            presenter->GetRasterizer().GetScheduler().DeferPriorityOperationAt(
                release_tick, [port, release_index] {
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_LIVE_VIDEOOUT") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_VIDEOOUT_RETIRE] stage=callback index=%d",
                                            release_index);
                    }
#endif
                    ExecutorRetireVoLabel(port, release_index);
                });
        } else {
            port->StoreVoLabel(release_index, 0);
        }
    }
#ifdef __ANDROID__
    ExecutorLogVideoOutPortState("flip_after_labels", port, req.index, req.index, req.flip_arg,
                                 req.eop);
#endif
}

void VideoOutDriver::DrawBlankFrame() {
    if (!presenter || Libraries::GnmDriver::ExecutorReplayActive()) {
        return;
    }
    const auto empty_frame = presenter->PrepareBlankFrame(false);
    presenter->Present(empty_frame);
}

void VideoOutDriver::DrawLastFrame() {
    if (!presenter || Libraries::GnmDriver::ExecutorReplayActive()) {
        return;
    }
    const auto frame = presenter->PrepareLastFrame();
    if (frame != nullptr) {
        presenter->Present(frame, true);
    }
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg,
                                bool is_eop) {
    const s32 requested_index = index;
    index = port->ResolveRegisteredBufferIndex(index);
    if (requested_index != -1 && index == -1) {
        LOG_ERROR(Lib_VideoOut, "Flip requested with no registered buffers index={}",
                  requested_index);
        return false;
    }
#ifdef __ANDROID__
    if (requested_index != index) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_VIDEOOUT_SUBMIT_INDEX_FIXUP] requested=%d effective=%d isEop=%u",
                            requested_index, index, is_eop ? 1u : 0u);
    }
    ExecutorLogVideoOutPortState("submit_enter", port, requested_index, index, flip_arg, is_eop);
#endif
    u64 parity_flip_ordinal{};
    {
        std::unique_lock lock{port->port_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        ++port->flip_status.flip_pending_num;
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
        parity_flip_ordinal = port->flip_status.count +
                              static_cast<u64>(port->flip_status.flip_pending_num) - 1;
    }

    if (Common::ParityTrace::IsEnabled()) {
        Common::ParityTrace::Json data = Common::ParityTrace::Json::object();
        data["requestedBufferIndex"] = requested_index;
        data["bufferIndex"] = index;
        data["blank"] = index == -1;
        data["isEop"] = is_eop;
        if (index != -1) {
            const auto& buffer = port->buffer_slots[index];
            ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
            const auto& attribute = port->groups[buffer.group_index].attrib;
            data["guestAddress"] = Common::ParityTrace::Hex(buffer.address_left);
            data["pixelFormat"] =
                Common::ParityTrace::Hex(static_cast<u32>(attribute.pixel_format));
            data["width"] = attribute.width;
            data["height"] = attribute.height;
            data["pitch"] = attribute.pitch_in_pixel;
            data["tilingMode"] = static_cast<s32>(attribute.tiling_mode);
        } else {
            data["guestAddress"] = "0x0";
            data["pixelFormat"] = "0x0";
            data["width"] = 0;
            data["height"] = 0;
            data["pitch"] = 0;
            data["tilingMode"] = -1;
        }
        Common::ParityTrace::Emit(
            Common::ParityTrace::Domain::VideoOut, "flip",
            Common::ParityTrace::Contract::MustEqual,
            {.submit = -1,
             .draw = -1,
             .stage = Common::ParityTrace::Stage::None,
             .slot = index,
             .ordinal = parity_flip_ordinal},
            std::move(data));
    }
#ifdef __ANDROID__
    ExecutorLogVideoOutPortState("submit_after_pending", port, requested_index, index, flip_arg,
                                 is_eop);
#endif

    if (!is_eop && liverpool) {
        liverpool->SendCommand([=, this]() { SubmitFlipInternal(port, index, flip_arg, is_eop); });
    } else {
        SubmitFlipInternal(port, index, flip_arg, is_eop);
    }

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop) {
#ifdef __ANDROID__
    ExecutorLogVideoOutPortState("internal_enter", port, index, index, flip_arg, is_eop);
#endif
    if (!presenter) {
        if (index != -1 && executor_lsx4_runtime_present_guest_frame) {
            const auto& buffer = port->buffer_slots[index];
            ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
            const auto& group = port->groups[buffer.group_index];
            const auto& attribute = group.attrib;
            const u32 stride = attribute.pitch_in_pixel * PixelFormatBpp(attribute.pixel_format);
            executor_lsx4_runtime_present_guest_frame(
                reinterpret_cast<const void*>(buffer.address_left), attribute.width, attribute.height,
                stride, static_cast<u32>(attribute.pixel_format),
                static_cast<u32>(attribute.tiling_mode));
        }

        {
            std::unique_lock lock{port->port_mutex};
            auto& flip_status = port->flip_status;
            flip_status.count++;
            flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            flip_status.flip_arg = flip_arg;
            flip_status.current_buffer = index;
            if (is_eop && flip_status.gc_queue_num > 0) {
                --flip_status.gc_queue_num;
            }
            if (flip_status.flip_pending_num > 0) {
                --flip_status.flip_pending_num;
            }
        }

        for (auto& event : port->flip_events) {
            if (event != nullptr) {
                event->TriggerEvent(
                    static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                    Kernel::OrbisKernelEvent::Filter::VideoOut,
                    reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                            (flip_arg << 16)));
            }
        }

        if (port->prev_index != -1) {
            port->StoreVoLabel(port->prev_index, 0);
        }
#ifdef __ANDROID__
        ExecutorForceLiveFlipDoneLabel(port, index, "internal_no_presenter");
#endif
        port->prev_index = index;
        port->prev_ready_tick = 0;
#ifdef __ANDROID__
        ExecutorLogVideoOutPortState("internal_no_presenter_done", port, index, index, flip_arg,
                                     is_eop);
#endif
        return;
    }

    Vulkan::Frame* frame{};
    const bool suppress_black_videoout = Libraries::GnmDriver::ExecutorLivePresentRtEnabled() &&
                                         Libraries::GnmDriver::ExecutorLiveDirectPresentActive();
    if (suppress_black_videoout) {
        frame = nullptr;
    } else if (index == -1) {
        frame = presenter->PrepareBlankFrame(false);
    } else {
        const auto& buffer = port->buffer_slots[index];
        ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
        const auto& group = port->groups[buffer.group_index];
        frame = presenter->PrepareFrame(group, buffer.address_left);
    }

    std::scoped_lock lock{mutex};
    requests.push({
        .frame = frame,
        .ready_tick = frame ? frame->ready_tick : 0,
        .port = port,
        .flip_arg = flip_arg,
        .index = index,
        .eop = is_eop,
    });
#ifdef __ANDROID__
    ExecutorLogVideoOutPortState("internal_queued", port, index, index, flip_arg, is_eop);
#endif
}

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 / Config::vblankFreq());

    Common::SetCurrentThreadName("shadPS4:PresentThread");
    Common::SetCurrentThreadRealtime(vblank_period);

    Common::AccurateTimer timer{vblank_period};

    const auto receive_request = [this] -> Request {
        std::scoped_lock lk{mutex};
        if (!requests.empty()) {
            const auto request = requests.front();
            requests.pop();
            return request;
        }
        return {};
    };

    while (!token.stop_requested()) {
        timer.Start();

        if (DebugState.IsGuestThreadsPaused()) {
            DrawLastFrame();
            timer.End();
            continue;
        }

        auto& vblank_status = main_port.vblank_status;
        if (vblank_status.count % (main_port.flip_rate + 1) == 0) {
            const auto request = receive_request();
            if (!request) {
                if (timer.GetTotalWait().count() < 0) {
                    if (!main_port.is_open) {
                        DrawBlankFrame();
                    } else if (ImGui::Core::MustKeepDrawing()) {
                        DrawLastFrame();
                    }
                }
            } else {
                Flip(request);
                FRAME_END;
            }
        }

        {
            std::scoped_lock lock{main_port.vo_mutex};

            for (auto& event : main_port.vblank_events) {
                if (event != nullptr) {
                    event->TriggerEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                                        reinterpret_cast<void*>(
                                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                                            (vblank_status.count << 16)));
                }
            }

            vblank_status.count++;
            vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
        }

        timer.End();
    }
}

}
