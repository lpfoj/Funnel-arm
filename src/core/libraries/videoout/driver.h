// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/debug.h"
#include "common/polyfill_thread.h"
#include "core/libraries/videoout/video_out.h"

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <queue>

namespace Vulkan {
struct Frame;
}

namespace Libraries::VideoOut {

struct VideoOutPort {
    SceVideoOutResolutionStatus resolution;
    std::array<VideoOutBuffer, MaxDisplayBuffers> buffer_slots;
    std::array<u64, MaxDisplayBuffers> buffer_labels;
    static_assert(sizeof(buffer_labels[0]) == 8u);
    std::array<BufferAttributeGroup, MaxDisplayBufferGroups> groups;
    FlipStatus flip_status;
    SceVideoOutVblankStatus vblank_status;
    std::vector<Kernel::EqueueInternal*> flip_events;
    std::vector<Kernel::EqueueInternal*> vblank_events;
    mutable std::mutex vo_mutex;
    std::mutex port_mutex;
    std::condition_variable vo_cv;
    std::condition_variable vblank_cv;
    int flip_rate = 0;
    int prev_index = -1;
    u64 prev_ready_tick = 0;
    bool is_open = false;
    bool is_hdr = false;

    static uintptr_t CanonicalLabelAddress(uintptr_t address) {
#ifdef __ANDROID__
        return address & 0x00ff'ffff'ffff'ffffull;
#else
        return address;
#endif
    }

    s32 FindFreeGroup() const {
        s32 index = 0;
        while (index < groups.size() && groups[index].is_occupied) {
            index++;
        }
        return index;
    }

    bool IsVoLabel(const u64* address) const {
        const uintptr_t addr = CanonicalLabelAddress(reinterpret_cast<uintptr_t>(address));
        const uintptr_t start =
            CanonicalLabelAddress(reinterpret_cast<uintptr_t>(&buffer_labels[0]));
        const uintptr_t end = start + sizeof(buffer_labels);
        return addr >= start && addr < end;
    }

    bool TryWriteVoLabel(const void* address, const void* data, u32 size) {
        const uintptr_t addr = CanonicalLabelAddress(reinterpret_cast<uintptr_t>(address));
        const uintptr_t start =
            CanonicalLabelAddress(reinterpret_cast<uintptr_t>(&buffer_labels[0]));
        const uintptr_t end = start + sizeof(buffer_labels);
        if (addr < start || size > end - addr) {
            return false;
        }
        std::scoped_lock lk{vo_mutex};
        auto* dst = reinterpret_cast<std::uint8_t*>(buffer_labels.data()) + (addr - start);
        std::memcpy(dst, data, size);
        vo_cv.notify_all();
        return true;
    }

    bool TryReadVoLabel(const void* address, void* data, u32 size) const {
        std::scoped_lock lk{vo_mutex};
        return TryReadVoLabelLocked(address, data, size);
    }

    bool TryReadVoLabelLocked(const void* address, void* data, u32 size) const {
        const uintptr_t addr = CanonicalLabelAddress(reinterpret_cast<uintptr_t>(address));
        const uintptr_t start =
            CanonicalLabelAddress(reinterpret_cast<uintptr_t>(&buffer_labels[0]));
        const uintptr_t end = start + sizeof(buffer_labels);
        if (addr < start || size > end - addr) {
            return false;
        }
        const auto* src =
            reinterpret_cast<const std::uint8_t*>(buffer_labels.data()) + (addr - start);
        std::memcpy(data, src, size);
        return true;
    }

    u64 StoreVoLabel(u32 index, u64 value) {
        ASSERT(index < buffer_labels.size());
        std::scoped_lock lk{vo_mutex};
        const u64 previous = buffer_labels[index];
        buffer_labels[index] = value;
        vo_cv.notify_all();
        return previous;
    }

    u64 LoadVoLabel(u32 index) const {
        ASSERT(index < buffer_labels.size());
        std::scoped_lock lk{vo_mutex};
        return buffer_labels[index];
    }

    std::array<u64, MaxDisplayBuffers> SnapshotVoLabels() const {
        std::scoped_lock lk{vo_mutex};
        return buffer_labels;
    }

    void WaitVoLabel(auto&& pred) {
        std::unique_lock lk{vo_mutex};
        vo_cv.wait(lk, pred);
    }

    void SignalVoLabel() {
        std::scoped_lock lk{vo_mutex};
        vo_cv.notify_all();
    }

    [[nodiscard]] int NumRegisteredBuffers() const {
        return std::count_if(buffer_slots.cbegin(), buffer_slots.cend(),
                             [](auto& buffer) { return buffer.group_index != -1; });
    }

    [[nodiscard]] s32 ResolveRegisteredBufferIndex(s32 requested) const {
        if (requested >= 0 && requested < static_cast<s32>(MaxDisplayBuffers) &&
            buffer_slots[requested].group_index >= 0) {
            return requested;
        }
        if (prev_index >= 0 && prev_index < static_cast<s32>(MaxDisplayBuffers) &&
            buffer_slots[prev_index].group_index >= 0) {
            return prev_index;
        }
        for (s32 index = 0; index < static_cast<s32>(MaxDisplayBuffers); ++index) {
            if (buffer_slots[index].group_index >= 0) {
                return index;
            }
        }
        return -1;
    }
};

struct ServiceThreadParams {
    u32 unknown;
    bool set_priority;
    u32 priority;
    bool set_affinity;
    u64 affinity;
};

class VideoOutDriver {
public:
    VideoOutDriver(u32 width, u32 height);
    ~VideoOutDriver();

    int Open(const ServiceThreadParams* params);
    void Close(s32 handle);

    VideoOutPort* GetPort(s32 handle);

    int RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses, s32 bufferNum,
                        const BufferAttribute* attribute);
    int UnregisterBuffers(VideoOutPort* port, s32 attributeIndex);

    bool SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop = false);

    void ReplayBufferRegistrationsToPresenter();

private:
    struct Request {
        Vulkan::Frame* frame;
        u64 ready_tick;
        VideoOutPort* port;
        s64 flip_arg;
        s32 index;
        bool eop;

        operator bool() const noexcept {
            return frame != nullptr;
        }
    };

    void Flip(const Request& req);
    void DrawBlankFrame();
    void DrawLastFrame();
    void SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop = false);
    void PresentThread(std::stop_token token);

    std::mutex mutex;
    VideoOutPort main_port{};
    std::jthread present_thread;
    std::queue<Request> requests;
};

}
