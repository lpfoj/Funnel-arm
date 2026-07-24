// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <thread>
#include <queue>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

class MasterSemaphore {
public:
    explicit MasterSemaphore(const Instance& instance_);
    ~MasterSemaphore();

    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsDeviceLost() const noexcept {
        return device_lost.load(std::memory_order_acquire) ||
               device_terminal.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return !IsDeviceLost() && KnownGpuTick() >= tick;
    }

    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return semaphore.get();
    }

    void Refresh();

    void Wait(u64 tick);

    void MarkDeviceLost(vk::Result result, const char* where) noexcept;

    void MarkTerminal(vk::Result result, const char* where) noexcept;

    [[nodiscard]] static bool IsDeviceTerminal() noexcept {
        return device_terminal.load(std::memory_order_acquire);
    }

protected:
    const Instance& instance;
    vk::UniqueSemaphore semaphore;
    std::atomic<u64> gpu_tick{0};
    std::atomic<u64> current_tick{1};
    std::atomic_bool device_lost{false};
    static std::atomic_bool device_terminal;
    static std::atomic<u32> live_instances;
};

}
