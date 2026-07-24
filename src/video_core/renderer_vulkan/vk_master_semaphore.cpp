// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <limits>
#ifdef __ANDROID__
#include <android/log.h>
#include <atomic>
#include <chrono>
#endif
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

#include "common/assert.h"

namespace Vulkan {

constexpr u64 WAIT_TIMEOUT = std::numeric_limits<u64>::max();
#ifdef __ANDROID__
constexpr u64 ANDROID_WAIT_SLICE = 250'000'000ULL;
#endif

std::atomic_bool MasterSemaphore::device_terminal{false};
std::atomic<u32> MasterSemaphore::live_instances{0};

MasterSemaphore::MasterSemaphore(const Instance& instance_) : instance{instance_} {
    if (live_instances.fetch_add(1, std::memory_order_acq_rel) == 0) {
        device_terminal.store(false, std::memory_order_release);
    }
    const vk::StructureChain semaphore_chain = {
        vk::SemaphoreCreateInfo{},
        vk::SemaphoreTypeCreateInfo{
            .semaphoreType = vk::SemaphoreType::eTimeline,
            .initialValue = 0,
        },
    };
    auto [semaphore_result, sem] =
        instance.GetDevice().createSemaphoreUnique(semaphore_chain.get());
    ASSERT_MSG(semaphore_result == vk::Result::eSuccess, "Failed to create master semaphore: {}",
               vk::to_string(semaphore_result));
    semaphore = std::move(sem);
}

MasterSemaphore::~MasterSemaphore() {
    live_instances.fetch_sub(1, std::memory_order_acq_rel);
}

void MasterSemaphore::MarkDeviceLost(vk::Result result, const char* where) noexcept {
    if (result != vk::Result::eErrorDeviceLost) {
        return;
    }

    MarkTerminal(result, where);
}

void MasterSemaphore::MarkTerminal(vk::Result result, const char* where) noexcept {
    device_lost.store(true, std::memory_order_release);

    bool expected = false;
    if (device_terminal.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
#ifdef __ANDROID__
        const std::string result_name = vk::to_string(result);
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_VK_TERMINAL_ORIGIN] where=%s result=%d name=%s", where,
                            static_cast<int>(result), result_name.c_str());
#endif
        LOG_ERROR(Render_Vulkan, "[EXECUTOR_VK_TERMINAL] where={} result={} terminal=1", where,
                  vk::to_string(result));
    }
}

void MasterSemaphore::Refresh() {
    if (IsDeviceLost()) {
        return;
    }

    u64 this_tick{};
    u64 counter{};
    do {
        this_tick = gpu_tick.load(std::memory_order_acquire);
        auto [counter_result, cntr] = instance.GetDevice().getSemaphoreCounterValue(*semaphore);
        if (counter_result != vk::Result::eSuccess) [[unlikely]] {
            MarkTerminal(counter_result, "timeline_counter");
            if (counter_result != vk::Result::eErrorDeviceLost) {
                LOG_ERROR(Render_Vulkan, "Failed to get master semaphore value: {}",
                          vk::to_string(counter_result));
            }
            return;
        }
        counter = cntr;
        if (counter < this_tick) {
            return;
        }
    } while (!gpu_tick.compare_exchange_weak(this_tick, counter, std::memory_order_release,
                                             std::memory_order_relaxed));
}

void MasterSemaphore::Wait(u64 tick) {
    if (IsDeviceLost()) {
        return;
    }

    if (IsFree(tick)) {
        return;
    }
    Refresh();
    if (IsDeviceLost() || IsFree(tick)) {
        return;
    }

    const vk::SemaphoreWaitInfo wait_info = {
        .semaphoreCount = 1,
        .pSemaphores = &semaphore.get(),
        .pValues = &tick,
    };

    vk::Result wait_result{};
#ifdef __ANDROID__
    const auto executor_wait_begin = std::chrono::steady_clock::now();
    do {
        wait_result = instance.GetDevice().waitSemaphores(&wait_info, ANDROID_WAIT_SLICE);
    } while (wait_result == vk::Result::eTimeout);
    const u64 executor_wait_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - executor_wait_begin)
            .count());
    static std::atomic<u64> executor_wait_samples{};
    static std::atomic<u64> executor_wait_total_ns{};
    static std::atomic<u64> executor_wait_max_ns{};
    executor_wait_total_ns.fetch_add(executor_wait_ns, std::memory_order_relaxed);
    u64 observed_max = executor_wait_max_ns.load(std::memory_order_relaxed);
    while (observed_max < executor_wait_ns &&
           !executor_wait_max_ns.compare_exchange_weak(
               observed_max, executor_wait_ns, std::memory_order_relaxed)) {
    }
    const u64 wait_sample =
        executor_wait_samples.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((wait_sample & 1023u) == 0u) {
        const u64 total_ns =
            executor_wait_total_ns.exchange(0, std::memory_order_relaxed);
        const u64 max_ns =
            executor_wait_max_ns.exchange(0, std::memory_order_relaxed);
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_VK_TIMELINE_WAIT] samples=1024 totalMs=%.3f avgUs=%.3f maxMs=%.3f",
            static_cast<double>(total_ns) / 1'000'000.0,
            static_cast<double>(total_ns) / 1'024'000.0,
            static_cast<double>(max_ns) / 1'000'000.0);
    }
#else
    wait_result = instance.GetDevice().waitSemaphores(&wait_info, WAIT_TIMEOUT);
#endif
    if (wait_result != vk::Result::eSuccess) [[unlikely]] {
        MarkTerminal(wait_result, "timeline_wait");
        if (wait_result != vk::Result::eErrorDeviceLost) {
            LOG_ERROR(Render_Vulkan, "Failed waiting for master semaphore: {}",
                      vk::to_string(wait_result));
        }
        return;
    }
    Refresh();
}

}
