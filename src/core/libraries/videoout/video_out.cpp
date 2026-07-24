// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "core/libraries/libs.h"
#include "core/libraries/system/userservice.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/video_out.h"
#include "core/libraries/videoout/videoout_error.h"
#include "core/platform.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

#include <array>
#include <atomic>
#include <cstdlib>

#ifdef __ANDROID__
#include <android/log.h>
#endif

extern std::unique_ptr<Vulkan::Presenter> presenter;

namespace Libraries::VideoOut {

static std::unique_ptr<VideoOutDriver> driver;

namespace {

struct VoLabelAlias {
    uintptr_t canonical_address{};
    VideoOutPort* port{};
    u32 buffer_index{};
};

std::mutex g_vo_label_alias_mutex;
std::array<VoLabelAlias, MaxDisplayBuffers * 4> g_vo_label_aliases{};
std::size_t g_vo_label_alias_count = 0;

#ifdef __ANDROID__
bool ExecutorTraceVideoOutLabels() {
    static const bool enabled = std::getenv("EXECUTOR_TRACE_LIVE_VIDEOOUT") != nullptr;
    return enabled;
}
#endif

uintptr_t CanonicalLabelAddress(uintptr_t address) {
    return VideoOutPort::CanonicalLabelAddress(address);
}

bool FindVoLabelAliasLocked(uintptr_t address, u32 size, VoLabelAlias* out, uintptr_t* offset) {
    const uintptr_t canonical = CanonicalLabelAddress(address);
    for (std::size_t index = 0; index < g_vo_label_alias_count; ++index) {
        const auto& alias = g_vo_label_aliases[index];
        const uintptr_t begin = alias.canonical_address;
        const uintptr_t end = begin + sizeof(u64);
        if (alias.port != nullptr && alias.buffer_index < MaxDisplayBuffers &&
            canonical >= begin && size <= end - canonical) {
            if (out) {
                *out = alias;
            }
            if (offset) {
                *offset = canonical - begin;
            }
            return true;
        }
    }
    return false;
}

}

void PS4_SYSV_ABI sceVideoOutSetBufferAttribute(BufferAttribute* attribute, PixelFormat pixelFormat,
                                                u32 tilingMode, u32 aspectRatio, u32 width,
                                                u32 height, u32 pitchInPixel) {
    const u32 original_pitch = pitchInPixel;
    if (pitchInPixel < width || pitchInPixel > 16384) {
        pitchInPixel = width;
    }
    LOG_INFO(Lib_VideoOut,
             "pixelFormat = {}, tilingMode = {}, aspectRatio = {}, width = {}, height = {}, "
             "pitchInPixel = {}",
             GetPixelFormatString(pixelFormat), tilingMode, aspectRatio, width, height,
             pitchInPixel);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_ATTR] fmt=0x%x tiling=%u width=%u height=%u pitch=%u originalPitch=%u normalized=%u",
                        static_cast<u32>(pixelFormat), tilingMode, width, height, pitchInPixel,
                        original_pitch, original_pitch != pitchInPixel ? 1u : 0u);
    if (original_pitch != pitchInPixel) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_VIDEOOUT_ATTR_FIXUP] originalPitch=%u width=%u height=%u",
                            original_pitch, width, height);
    }
#endif

    std::memset(attribute, 0, sizeof(BufferAttribute));
    attribute->pixel_format = static_cast<PixelFormat>(pixelFormat);
    attribute->tiling_mode = static_cast<TilingMode>(tilingMode);
    attribute->aspect_ratio = aspectRatio;
    attribute->width = width;
    attribute->height = height;
    attribute->pitch_in_pixel = pitchInPixel;
    attribute->option = SCE_VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_NONE;
}

s32 PS4_SYSV_ABI sceVideoOutAddFlipEvent(Kernel::OrbisKernelEqueue eq, s32 handle, void* udata) {
    LOG_INFO(Lib_VideoOut, "handle = {}", handle);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_ADD_FLIP_EVENT] eq=%d handle=%d udata=%p", eq,
                        handle, udata);
#endif

    auto* port = driver->GetPort(handle);
    if (port == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    auto equeue = Kernel::GetEqueue(eq);
    if (equeue == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
    }

    Kernel::EqueueEvent event{};
    event.event.ident = static_cast<u64>(OrbisVideoOutInternalEventId::Flip);
    event.event.filter = Kernel::OrbisKernelEvent::Filter::VideoOut;
    event.event.flags = Kernel::OrbisKernelEvent::Flags::Add;
    event.event.udata = udata;
    event.event.fflags = 0;
    event.event.data = 0;
    event.data = port;
    equeue->AddEvent(event);

    port->flip_events.push_back(equeue);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_ADD_FLIP_EVENT_RESULT] eq=%d handle=%d rc=0",
                        eq, handle);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutDeleteFlipEvent(Kernel::OrbisKernelEqueue eq, s32 handle) {
    auto* port = driver->GetPort(handle);
    if (port == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    auto equeue = Kernel::GetEqueue(eq);
    if (equeue == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
    }
    equeue->RemoveEvent(handle, Kernel::OrbisKernelEvent::Filter::VideoOut);
    port->flip_events.erase(find(port->flip_events.begin(), port->flip_events.end(), equeue));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutAddVblankEvent(Kernel::OrbisKernelEqueue eq, s32 handle, void* udata) {
    LOG_INFO(Lib_VideoOut, "handle = {}", handle);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_ADD_VBLANK_EVENT] eq=%d handle=%d udata=%p",
                        eq, handle, udata);
#endif

    auto* port = driver->GetPort(handle);
    if (port == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    auto equeue = Kernel::GetEqueue(eq);
    if (equeue == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
    }

    Kernel::EqueueEvent event{};
    event.event.ident = static_cast<u64>(OrbisVideoOutInternalEventId::Vblank);
    event.event.filter = Kernel::OrbisKernelEvent::Filter::VideoOut;
    event.event.flags = Kernel::OrbisKernelEvent::Flags::Add;
    event.event.udata = udata;
    event.event.fflags = 0;
    event.event.data = 0;
    event.data = port;
    equeue->AddEvent(event);

    port->vblank_events.push_back(equeue);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_ADD_VBLANK_EVENT_RESULT] eq=%d handle=%d rc=0",
                        eq, handle);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutDeleteVblankEvent(Kernel::OrbisKernelEqueue eq, s32 handle) {
    auto* port = driver->GetPort(handle);
    if (port == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    auto equeue = Kernel::GetEqueue(eq);
    if (equeue == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
    }
    equeue->RemoveEvent(handle, Kernel::OrbisKernelEvent::Filter::VideoOut);
    port->vblank_events.erase(find(port->vblank_events.begin(), port->vblank_events.end(), equeue));
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutRegisterBuffers(s32 handle, s32 startIndex, void* const* addresses,
                                            s32 bufferNum, const BufferAttribute* attribute) {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_REGISTER] handle=%d start=%d count=%d addrs=%p attr=%p",
                        handle, startIndex, bufferNum, addresses, attribute);
    if (addresses != nullptr) {
        for (s32 i = 0; i < bufferNum && i < 8; ++i) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_VIDEOOUT_REGISTER_ADDR] slot=%d addr=%p",
                                startIndex + i, addresses[i]);
        }
    }
    if (attribute != nullptr) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_VIDEOOUT_REGISTER_ATTR] fmt=0x%x tiling=%u width=%u height=%u pitch=%u",
                            static_cast<u32>(attribute->pixel_format),
                            static_cast<u32>(attribute->tiling_mode), attribute->width,
                            attribute->height, attribute->pitch_in_pixel);
    }
#endif
    if (!addresses || !attribute) {
        LOG_ERROR(Lib_VideoOut, "Addresses are null");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }

    auto* port = driver->GetPort(handle);
    if (!port || !port->is_open) {
        LOG_ERROR(Lib_VideoOut, "Invalid handle = {}", handle);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    const s32 rc = driver->RegisterBuffers(port, startIndex, addresses, bufferNum, attribute);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_REGISTER_RESULT] handle=%d start=%d count=%d rc=0x%x",
                        handle, startIndex, bufferNum, static_cast<u32>(rc));
#endif
    return rc;
}

s32 PS4_SYSV_ABI sceVideoOutSetFlipRate(s32 handle, s32 rate) {
    LOG_TRACE(Lib_VideoOut, "called");
#ifdef __ANDROID__
    if (ExecutorTraceVideoOutLabels()) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_VIDEOOUT_FLIP_RATE] handle=%d rate=%d", handle, rate);
    }
#endif
    driver->GetPort(handle)->flip_rate = rate;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutIsFlipPending(s32 handle) {
    LOG_TRACE(Lib_VideoOut, "called");
    auto* port = driver->GetPort(handle);
    std::unique_lock lock{port->port_mutex};
    s32 pending = port->flip_status.flip_pending_num;
#ifdef __ANDROID__
    const auto labels = port->SnapshotVoLabels();
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_VIDEOOUT_IS_PENDING] handle=%d pending=%d gc=%d count=%lld current=%d prev=%d labels=%llu,%llu,%llu,%llu",
        handle, pending, port->flip_status.gc_queue_num,
        static_cast<long long>(port->flip_status.count), port->flip_status.current_buffer,
        port->prev_index, static_cast<unsigned long long>(labels[0]),
        static_cast<unsigned long long>(labels[1]),
        static_cast<unsigned long long>(labels[2]),
        static_cast<unsigned long long>(labels[3]));
#endif
    return pending;
}

s32 PS4_SYSV_ABI sceVideoOutSubmitFlip(s32 handle, s32 bufferIndex, s32 flipMode, s64 flipArg) {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_FLIP] handle=%d index=%d mode=%d arg=%lld",
                        handle, bufferIndex, flipMode, static_cast<long long>(flipArg));
#endif
    auto* port = driver->GetPort(handle);
    if (!port) {
        LOG_ERROR(Lib_VideoOut, "Invalid handle = {}", handle);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    if (flipMode != 1) {
        LOG_WARNING(Lib_VideoOut, "flipmode = {}", flipMode);
    }

    if (bufferIndex < -1 || bufferIndex > 15) {
        LOG_ERROR(Lib_VideoOut, "Invalid bufferIndex = {}", bufferIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_INDEX;
    }

    if (bufferIndex != -1 && port->buffer_slots[bufferIndex].group_index < 0) {
        LOG_ERROR(Lib_VideoOut, "Slot in bufferIndex = {} is not registered", bufferIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_INDEX;
    }

    LOG_DEBUG(Lib_VideoOut, "bufferIndex = {}, flipMode = {}, flipArg = {}", bufferIndex, flipMode,
              flipArg);

    const bool submitted = driver->SubmitFlip(port, bufferIndex, flipArg);
#ifdef __ANDROID__
    {
        std::unique_lock lock{port->port_mutex};
        const auto labels = port->SnapshotVoLabels();
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_VIDEOOUT_FLIP_RESULT] handle=%d index=%d mode=%d arg=%lld submitted=%u rc=0x%x pending=%d gc=%d count=%lld current=%d prev=%d labels=%llu,%llu,%llu,%llu",
            handle, bufferIndex, flipMode, static_cast<long long>(flipArg), submitted ? 1u : 0u,
            submitted ? 0u : static_cast<u32>(ORBIS_VIDEO_OUT_ERROR_FLIP_QUEUE_FULL),
            port->flip_status.flip_pending_num, port->flip_status.gc_queue_num,
            static_cast<long long>(port->flip_status.count), port->flip_status.current_buffer,
            port->prev_index, static_cast<unsigned long long>(labels[0]),
            static_cast<unsigned long long>(labels[1]),
            static_cast<unsigned long long>(labels[2]),
            static_cast<unsigned long long>(labels[3]));
    }
#endif
    if (!submitted) {
        LOG_ERROR(Lib_VideoOut, "Flip queue is full");
        return ORBIS_VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutGetEventId(const Kernel::OrbisKernelEvent* ev) {
    if (ev == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    if (ev->filter != Kernel::OrbisKernelEvent::Filter::VideoOut) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_EVENT;
    }

    OrbisVideoOutInternalEventId internal_event_id =
        static_cast<OrbisVideoOutInternalEventId>(ev->ident);
    switch (internal_event_id) {
    case OrbisVideoOutInternalEventId::Flip:
        return static_cast<s32>(OrbisVideoOutEventId::Flip);
    case OrbisVideoOutInternalEventId::Vblank:
    case OrbisVideoOutInternalEventId::SysVblank:
        return static_cast<s32>(OrbisVideoOutEventId::Vblank);
    case OrbisVideoOutInternalEventId::PreVblankStart:
        return static_cast<s32>(OrbisVideoOutEventId::PreVblankStart);
    case OrbisVideoOutInternalEventId::SetMode:
        return static_cast<s32>(OrbisVideoOutEventId::SetMode);
    case OrbisVideoOutInternalEventId::Position:
        return static_cast<s32>(OrbisVideoOutEventId::Position);
    default: {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_EVENT;
    }
    }
}

s32 PS4_SYSV_ABI sceVideoOutGetEventData(const Kernel::OrbisKernelEvent* ev, s64* data) {
    if (ev == nullptr || data == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    if (ev->filter != Kernel::OrbisKernelEvent::Filter::VideoOut) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_EVENT;
    }

    auto event_data = ev->data >> 0x10;
    if (ev->ident != static_cast<s32>(OrbisVideoOutInternalEventId::Flip) || ev->data >= 0) {
        *data = event_data;
    } else {
        *data = event_data | 0xffff000000000000;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutGetEventCount(const Kernel::OrbisKernelEvent* ev) {
    if (ev == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    if (ev->filter != Kernel::OrbisKernelEvent::Filter::VideoOut) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_EVENT;
    }

    auto event_data = static_cast<OrbisVideoOutEventData>(ev->data);
    return event_data.count;
}

s32 PS4_SYSV_ABI sceVideoOutGetFlipStatus(s32 handle, FlipStatus* status) {
    if (!status) {
        LOG_ERROR(Lib_VideoOut, "Flip status is null");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }

    auto* port = driver->GetPort(handle);
    if (!port) {
        LOG_ERROR(Lib_VideoOut, "Invalid port handle = {}", handle);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    {
        std::unique_lock lock{port->port_mutex};
        *status = port->flip_status;
    }
#ifdef __ANDROID__
    static std::atomic<s64> last_sampled_count{-1};
    const s64 count = status->count;
    const s64 previous = last_sampled_count.load(std::memory_order_relaxed);
    const bool sample = count != previous && (count < 16 || (count & 0xff) == 0);
    if (ExecutorTraceVideoOutLabels() ||
        (sample && last_sampled_count.exchange(count, std::memory_order_relaxed) != count)) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_VIDEOOUT_FLIP_STATUS] handle=%d count=%lld pending=%d gc=%d current=%d arg=%lld",
            handle, static_cast<long long>(count), status->flip_pending_num, status->gc_queue_num,
            status->current_buffer, static_cast<long long>(status->flip_arg));
    }
#endif

    LOG_TRACE(Lib_VideoOut,
              "count = {}, processTime = {}, tsc = {}, submitTsc = {}, flipArg = {}, gcQueueNum = "
              "{}, flipPendingNum = {}, currentBuffer = {}",
              status->count, status->process_time, status->tsc, status->submit_tsc,
              status->flip_arg, status->gc_queue_num, status->flip_pending_num,
              status->current_buffer);

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutGetVblankStatus(int handle, SceVideoOutVblankStatus* status) {
    if (status == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }

    auto* port = driver->GetPort(handle);
    if (!port) {
        LOG_ERROR(Lib_VideoOut, "Invalid port handle = {}", handle);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    std::unique_lock lock{port->vo_mutex};
    *status = port->vblank_status;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutGetResolutionStatus(s32 handle, SceVideoOutResolutionStatus* status) {
    LOG_INFO(Lib_VideoOut, "called");
    auto* port = driver->GetPort(handle);
    if (!port || !port->is_open) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_VIDEOOUT_RESOLUTION_RESULT] handle=%d rc=0x%x status=%p",
                            handle, static_cast<u32>(ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE),
                            status);
#endif
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    *status = port->resolution;
#ifdef __ANDROID__
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_VIDEOOUT_RESOLUTION_RESULT] handle=%d rc=0 full=%ux%u pane=%ux%u status=%p",
        handle, status->full_width, status->full_height, status->pane_width, status->pane_height,
        status);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutOpen(Libraries::UserService::OrbisUserServiceUserId userId, s32 busType,
                                 s32 index, const void* param) {
    LOG_INFO(Lib_VideoOut, "called");
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_OPEN] user=%d bus=%d index=%d param=%p", userId,
                        busType, index, param);
#endif
    ASSERT(busType == SCE_VIDEO_OUT_BUS_TYPE_MAIN);

    if (index != 0) {
        LOG_ERROR(Lib_VideoOut, "Index != 0");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto* params = reinterpret_cast<const ServiceThreadParams*>(param);
    int handle = driver->Open(params);

    if (handle < 0) {
        LOG_ERROR(Lib_VideoOut, "All available handles are open");
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_OPEN_RESULT] handle=%d", handle);
#endif
    return handle;
}

s32 PS4_SYSV_ABI sceVideoOutClose(s32 handle) {
    driver->Close(handle);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutUnregisterBuffers(s32 handle, s32 attributeIndex) {
    auto* port = driver->GetPort(handle);
    if (!port || !port->is_open) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    return driver->UnregisterBuffers(port, attributeIndex);
}

s32 PS4_SYSV_ABI sceVideoOutGetBufferLabelAddress(s32 handle, uintptr_t* label_addr) {
    if (label_addr == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }
    auto* port = driver->GetPort(handle);
    if (!port) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    *label_addr = reinterpret_cast<uintptr_t>(port->buffer_labels.data());
    for (u32 index = 0; index < MaxDisplayBuffers; ++index) {
        ExecutorRegisterVoLabelAlias(handle, *label_addr + index * sizeof(port->buffer_labels[0]),
                                     index);
    }
#ifdef __ANDROID__
    if (ExecutorTraceVideoOutLabels()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_VIDEOOUT_LABEL_ADDR] handle=%d base=0x%llx canonical=0x%llx labels=%u port=%p",
            handle, static_cast<unsigned long long>(*label_addr),
            static_cast<unsigned long long>(CanonicalLabelAddress(*label_addr)), MaxDisplayBuffers,
            port);
    }
#endif
    return 16;
}

bool ExecutorRegisterVoLabelAlias(s32 handle, uintptr_t address, u32 buffer_index) {
    if (!driver) {
        return false;
    }
    auto* port = driver->GetPort(handle);
    if (!port || buffer_index >= MaxDisplayBuffers) {
        return false;
    }

    const uintptr_t canonical = CanonicalLabelAddress(address);
    bool updated = false;
    bool full = false;
    std::size_t alias_count_for_log = 0;
    {
        std::scoped_lock lock{g_vo_label_alias_mutex};
        for (std::size_t index = 0; index < g_vo_label_alias_count; ++index) {
            auto& alias = g_vo_label_aliases[index];
            if (alias.canonical_address == canonical) {
                alias.port = port;
                alias.buffer_index = buffer_index;
                updated = true;
                break;
            }
        }
        if (!updated) {
            if (g_vo_label_alias_count >= g_vo_label_aliases.size()) {
                full = true;
                alias_count_for_log = g_vo_label_alias_count;
            } else {
                g_vo_label_aliases[g_vo_label_alias_count++] = {canonical, port, buffer_index};
                alias_count_for_log = g_vo_label_alias_count;
            }
        } else {
            alias_count_for_log = g_vo_label_alias_count;
        }
    }
    if (full) {
#ifdef __ANDROID__
        if (ExecutorTraceVideoOutLabels()) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_LIVE_VIDEOOUT_LABEL_ALIAS] full handle=%d buf=%u raw=0x%llx canonical=0x%llx count=%zu",
                                handle, buffer_index, static_cast<unsigned long long>(address),
                                static_cast<unsigned long long>(canonical), alias_count_for_log);
        }
#endif
        return false;
    }
#ifdef __ANDROID__
    if (ExecutorTraceVideoOutLabels()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_VIDEOOUT_LABEL_ALIAS] %s handle=%d buf=%u raw=0x%llx canonical=0x%llx port=%p labelStart=0x%llx count=%zu",
            updated ? "update" : "add", handle, buffer_index,
            static_cast<unsigned long long>(address), static_cast<unsigned long long>(canonical),
            port,
            static_cast<unsigned long long>(
                CanonicalLabelAddress(reinterpret_cast<uintptr_t>(port->buffer_labels.data()))),
            alias_count_for_log);
    }
#endif
    return true;
}

bool ExecutorTryWriteVoLabelAlias(const void* address, const void* data, u32 size) {
    std::scoped_lock alias_lock{g_vo_label_alias_mutex};
    VoLabelAlias alias{};
    uintptr_t offset{};
    if (!FindVoLabelAliasLocked(reinterpret_cast<uintptr_t>(address), size, &alias, &offset) ||
        !driver || driver->GetPort(1) != alias.port) {
        return false;
    }

    auto* dst = reinterpret_cast<std::uint8_t*>(&alias.port->buffer_labels[alias.buffer_index]) +
                offset;
    if (!alias.port->TryWriteVoLabel(dst, data, size)) {
        return false;
    }
#ifdef __ANDROID__
    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_PM4_WRITEDATA] vo_label_alias_hit raw=0x%llx buf=%u offset=%llu size=%u data0=0x%08x",
                            static_cast<unsigned long long>(
                                reinterpret_cast<uintptr_t>(address)),
                            alias.buffer_index, static_cast<unsigned long long>(offset), size,
                            size >= sizeof(u32) ? *static_cast<const u32*>(data) : 0u);
    }
#endif
    return true;
}

bool ExecutorTryReadVoLabelAlias(const void* address, u32* value) {
    if (!value) {
        return false;
    }
    std::scoped_lock alias_lock{g_vo_label_alias_mutex};
    VoLabelAlias alias{};
    uintptr_t offset{};
    if (!FindVoLabelAliasLocked(reinterpret_cast<uintptr_t>(address), sizeof(u32), &alias,
                                &offset) ||
        !driver || driver->GetPort(1) != alias.port) {
        return false;
    }
    const auto* src =
        reinterpret_cast<const std::uint8_t*>(&alias.port->buffer_labels[alias.buffer_index]) +
        offset;
    return alias.port->TryReadVoLabel(src, value, sizeof(u32));
}

bool ExecutorRetireVoLabel(VideoOutPort* expected_port, u32 buffer_index) {
    if (!expected_port || buffer_index >= MaxDisplayBuffers) {
        return false;
    }
    std::scoped_lock alias_lock{g_vo_label_alias_mutex};
    if (!driver) {
        return false;
    }
    auto* const current_port = driver->GetPort(1);
    if (current_port != expected_port) {
        return false;
    }
    current_port->StoreVoLabel(buffer_index, 0);
    return true;
}

bool ExecutorResolveVoBufferIndex(s32 handle, u32 requested, u32* effective) {
    if (!driver || !effective) {
        return false;
    }
    auto* port = driver->GetPort(handle);
    if (!port) {
        return false;
    }
    const s32 resolved = port->ResolveRegisteredBufferIndex(static_cast<s32>(requested));
    if (resolved < 0) {
        return false;
    }
    *effective = static_cast<u32>(resolved);
    return true;
}

s32 sceVideoOutSubmitEopFlip(s32 handle, u32 buf_id, u32 mode, s64 flip_arg, void** unk) {
    auto* port = driver->GetPort(handle);
    if (!port) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }
    const s32 effective_buf_id = port->ResolveRegisteredBufferIndex(static_cast<s32>(buf_id));
    if (effective_buf_id < 0) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_LIVE_VIDEOOUT_EOP_FLIP_NO_BUFFER] handle=%d requested=%u",
                            handle, buf_id);
#endif
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
#ifdef __ANDROID__
    if (effective_buf_id != static_cast<s32>(buf_id)) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_VIDEOOUT_EOP_FLIP_INDEX_FIXUP] handle=%d requested=%u effective=%d",
                            handle, buf_id, effective_buf_id);
    }
#endif

    Platform::IrqC::Instance()->RegisterOnce(
        Platform::InterruptId::GfxFlip, [=](Platform::InterruptId irq) {
#ifdef __ANDROID__
            static std::atomic<u32> eop_flip_callback_log_count{0};
            const u32 eop_flip_callback_log =
                eop_flip_callback_log_count.fetch_add(1, std::memory_order_relaxed);
            if (eop_flip_callback_log < 8) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_EOP_FLIP_CALLBACK] ordinal=%u handle=%d requested=%u effective=%d port=%p",
                    eop_flip_callback_log + 1, handle, buf_id, effective_buf_id, port);
            }
            if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr ||
                std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_LIVE_VIDEOOUT_EOP_FLIP_ENTER] handle=%d requested=%u effective=%d port=%p",
                                    handle, buf_id, effective_buf_id, port);
            }
#endif
            ASSERT_MSG(irq == Platform::InterruptId::GfxFlip, "An unexpected IRQ occured");
#ifdef __ANDROID__
            const u64 old_label = port->LoadVoLabel(effective_buf_id);
            if (old_label != 1) {
                if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr ||
                    std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr) {
                    __android_log_print(
                        ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_EOP_FLIP_LABEL_FIXUP] handle=%d requested=%u effective=%d old=%llu",
                        handle, buf_id, effective_buf_id,
                        static_cast<unsigned long long>(old_label));
                }
                port->StoreVoLabel(effective_buf_id, 1);
            }
#else
            ASSERT_MSG(port->LoadVoLabel(effective_buf_id) == 1, "Out of order flip IRQ");
#endif
#ifdef __ANDROID__
            if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr ||
                std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_LIVE_VIDEOOUT_EOP_FLIP_BEFORE_SUBMIT] handle=%d requested=%u effective=%d",
                                    handle, buf_id, effective_buf_id);
            }
#endif
            const auto result = driver->SubmitFlip(port, effective_buf_id, flip_arg, true);
#ifdef __ANDROID__
            if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr ||
                std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_LIVE_VIDEOOUT_EOP_FLIP_AFTER_SUBMIT] handle=%d requested=%u effective=%d result=%u",
                                    handle, buf_id, effective_buf_id, result ? 1u : 0u);
            }
#endif
            ASSERT_MSG(result, "EOP flip submission failed");
        });

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutGetDeviceCapabilityInfo(
    s32 handle, SceVideoOutDeviceCapabilityInfo* pDeviceCapabilityInfo) {
    pDeviceCapabilityInfo->capability = 0;
    if (presenter && presenter->IsHDRSupported()) {
        auto& game_info = Common::ElfInfo::Instance();
        if (game_info.GetPSFAttributes().support_hdr) {
            pDeviceCapabilityInfo->capability |= ORBIS_VIDEO_OUT_DEVICE_CAPABILITY_BT2020_PQ;
        }
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutWaitVblank(s32 handle) {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_WAIT_VBLANK_ENTER] handle=%d", handle);
#endif
    auto* port = driver->GetPort(handle);
    if (!port) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    std::unique_lock lock{port->vo_mutex};
    const auto prev_counter = port->vblank_status.count;
    port->vblank_cv.wait(lock, [&]() { return prev_counter != port->vblank_status.count; });
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_VIDEOOUT_WAIT_VBLANK_EXIT] handle=%d count=%lld", handle,
                        static_cast<long long>(port->vblank_status.count));
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutColorSettingsSetGamma(SceVideoOutColorSettings* settings, float gamma) {
    if (gamma < 0.1f || gamma > 2.0f) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    settings->gamma = gamma;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutAdjustColor(s32 handle, const SceVideoOutColorSettings* settings) {
    if (settings == nullptr) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ADDRESS;
    }

    auto* port = driver->GetPort(handle);
    if (!port) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    if (presenter) {
        presenter->GetPPSettingsRef().gamma = settings->gamma;
    }
    return ORBIS_OK;
}

struct Mode {
    u32 size;
    u8 encoding;
    u8 range;
    u8 colorimetry;
    u8 depth;
    u64 refresh_rate;
    u64 resolution;
    u8 reserved[8];
};

void PS4_SYSV_ABI sceVideoOutModeSetAny_(Mode* mode, u32 size) {
    std::memset(mode, 0xff, size);
    mode->size = size;
}

s32 PS4_SYSV_ABI sceVideoOutConfigureOutputMode_(s32 handle, u32 reserved, const Mode* mode,
                                                 const void* options, u32 size_mode,
                                                 u32 size_options) {
    auto* port = driver->GetPort(handle);
    if (!port) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_HANDLE;
    }

    if (reserved != 0) {
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    switch (mode->colorimetry) {
    case OrbisVideoOutColorimetry::Any:
        port->is_hdr = false;
        break;
    case OrbisVideoOutColorimetry::Bt2020PQ:
        if (Common::ElfInfo::Instance().GetPSFAttributes().support_hdr) {
            port->is_hdr = true;
        }
        break;
    default:
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceVideoOutSetWindowModeMargins(s32 handle, s32 top, s32 bottom) {
    LOG_ERROR(Lib_VideoOut, "(STUBBED) called top = {}, bottom = {}", top, bottom);
    return ORBIS_OK;
}

void ReplayBufferRegistrationsToPresenter() {
    if (driver) {
        driver->ReplayBufferRegistrationsToPresenter();
    }
}

bool ExecutorGetVideoOutBufferSnapshot(VAddr candidate_address, u64 candidate_size,
                                       ExecutorVideoOutBufferSnapshot* snapshot) noexcept {
    if (!snapshot) {
        return false;
    }
    *snapshot = {};
    if (!driver) {
        return false;
    }

    auto* port = driver->GetPort(1);
    if (!port) {
        return false;
    }

    std::scoped_lock lock{port->port_mutex};
    snapshot->requested_index = port->flip_status.current_buffer;
    snapshot->effective_index =
        port->ResolveRegisteredBufferIndex(snapshot->requested_index);
    snapshot->registered_buffers = static_cast<u32>(port->NumRegisteredBuffers());
    for (s32 index = 0; index < static_cast<s32>(MaxDisplayBuffers); ++index) {
        const auto& registered = port->buffer_slots[index];
        if (registered.group_index < 0 ||
            registered.group_index >= static_cast<s32>(MaxDisplayBufferGroups)) {
            continue;
        }
        const auto& registered_attrib = port->groups[registered.group_index].attrib;
        const u64 registered_bpp =
            registered_attrib.pixel_format == PixelFormat::A16R16G16B16Float ? 8u : 4u;
        const u64 registered_size = static_cast<u64>(registered_attrib.pitch_in_pixel) *
                                    registered_attrib.height * registered_bpp;
        const VAddr registered_address = registered.address_left;
        if (candidate_address != 0 && candidate_address == registered_address &&
            snapshot->exact_address_index < 0) {
            snapshot->exact_address_index = index;
        }
        if (candidate_address != 0 && candidate_size != 0 && registered_address != 0 &&
            candidate_address < registered_address + registered_size &&
            registered_address < candidate_address + candidate_size &&
            snapshot->overlapping_address_index < 0) {
            snapshot->overlapping_address_index = index;
        }
    }
    if (snapshot->effective_index < 0 ||
        snapshot->effective_index >= static_cast<s32>(MaxDisplayBuffers)) {
        return false;
    }

    const auto& buffer = port->buffer_slots[snapshot->effective_index];
    if (buffer.group_index < 0 ||
        buffer.group_index >= static_cast<s32>(MaxDisplayBufferGroups)) {
        return false;
    }
    const auto& attrib = port->groups[buffer.group_index].attrib;
    const u64 bytes_per_pixel =
        attrib.pixel_format == PixelFormat::A16R16G16B16Float ? 8u : 4u;
    snapshot->valid = true;
    snapshot->address = buffer.address_left;
    snapshot->size_bytes = static_cast<u64>(attrib.pitch_in_pixel) * attrib.height *
                           bytes_per_pixel;
    snapshot->width = attrib.width;
    snapshot->height = attrib.height;
    snapshot->pitch_in_pixel = attrib.pitch_in_pixel;
    snapshot->pixel_format = static_cast<u32>(attrib.pixel_format);
    snapshot->tiling_mode = static_cast<u32>(attrib.tiling_mode);
    return true;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
    {
        std::scoped_lock alias_lock{g_vo_label_alias_mutex};
        g_vo_label_aliases.fill({});
        g_vo_label_alias_count = 0;
        driver = std::make_unique<VideoOutDriver>(Config::getInternalScreenWidth(),
                                                  Config::getInternalScreenHeight());
    }
    if (presenter == nullptr) {
        LOG_WARNING(Lib_VideoOut,
                    "Using Android native-window VideoOut fallback without Vulkan presenter");
    }
#else
    {
        std::scoped_lock alias_lock{g_vo_label_alias_mutex};
        g_vo_label_aliases.fill({});
        g_vo_label_alias_count = 0;
        driver = std::make_unique<VideoOutDriver>(Config::getInternalScreenWidth(),
                                                  Config::getInternalScreenHeight());
    }
#endif

    LIB_FUNCTION("SbU3dwp80lQ", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutGetFlipStatus);
    LIB_FUNCTION("U46NwOiJpys", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutSubmitFlip);
    LIB_FUNCTION("w3BY+tAEiQY", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutRegisterBuffers);
    LIB_FUNCTION("HXzjK9yI30k", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutAddFlipEvent);
    LIB_FUNCTION("Xru92wHJRmg", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutAddVblankEvent);
    LIB_FUNCTION("CBiu4mCE1DA", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutSetFlipRate);
    LIB_FUNCTION("i6-sR91Wt-4", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutSetBufferAttribute);
    LIB_FUNCTION("6kPnj51T62Y", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutGetResolutionStatus);
    LIB_FUNCTION("Up36PTk687E", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutOpen);
    LIB_FUNCTION("zgXifHT9ErY", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutIsFlipPending);
    LIB_FUNCTION("N5KDtkIjjJ4", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutUnregisterBuffers);
    LIB_FUNCTION("OcQybQejHEY", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutGetBufferLabelAddress);
    LIB_FUNCTION("uquVH4-Du78", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutClose);
    LIB_FUNCTION("1FZBKy8HeNU", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutGetVblankStatus);
    LIB_FUNCTION("kGVLc3htQE8", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutGetDeviceCapabilityInfo);
    LIB_FUNCTION("j6RaAUlaLv0", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutWaitVblank);
    LIB_FUNCTION("U2JJtSqNKZI", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutGetEventId);
    LIB_FUNCTION("rWUTcKdkUzQ", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutGetEventData);
    LIB_FUNCTION("Mt4QHHkxkOc", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutGetEventCount);
    LIB_FUNCTION("DYhhWbJSeRg", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutColorSettingsSetGamma);
    LIB_FUNCTION("pv9CI5VC+R0", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutAdjustColor);
    LIB_FUNCTION("-Ozn0F1AFRg", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutDeleteFlipEvent);
    LIB_FUNCTION("oNOQn3knW6s", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutDeleteVblankEvent);
    LIB_FUNCTION("pjkDsgxli6c", "libSceVideoOut", 1, "libSceVideoOut", sceVideoOutModeSetAny_);
    LIB_FUNCTION("N1bEoJ4SRw4", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutConfigureOutputMode_);
    LIB_FUNCTION("MTxxrOCeSig", "libSceVideoOut", 1, "libSceVideoOut",
                 sceVideoOutSetWindowModeMargins);
}

}
