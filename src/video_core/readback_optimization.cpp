// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>

#include "video_core/readback_optimization.h"

namespace VideoCore {
namespace {

struct AtomicReadbackOptimizationStats {
    std::atomic<u64> snapshots{};
    std::atomic<u64> submit_snapshots{};
    std::atomic<u64> demand_snapshots{};
    std::atomic<u64> ranges{};
    std::atomic<u64> pages{};
    std::atomic<u64> copy_commands{};
    std::atomic<u64> copy_regions{};
    std::atomic<u64> source_buffers{};
    std::atomic<u64> regions_merged{};
    std::atomic<u64> timeline_waits{};
    std::atomic<u64> bytes{};
    std::atomic<u64> cpu_plan_ns{};
    std::atomic<u64> cpu_publish_ns{};
    std::atomic<u64> cpu_wait_ns{};
    std::atomic<u64> gpu_completion_ns{};
    std::atomic<u64> fallbacks{};
    std::atomic<u64> publication_failures{};
    std::atomic<u64> duplicate_submit_skips{};
    std::atomic<u64> overflow_allocations{};
    std::atomic<u64> demand_flushes{};
    std::atomic<u64> baseline_requests{};
    std::atomic<u64> sidecar_labels{};
    std::atomic<u64> sidecar_waits{};
};

AtomicReadbackOptimizationStats g_readback_stats;
std::atomic<u64> g_last_readback_retirement_tick{};

#define LSX4_READBACK_STATS_FIELDS(X)                                                         \
    X(snapshots)                                                                              \
    X(submit_snapshots)                                                                       \
    X(demand_snapshots)                                                                       \
    X(ranges)                                                                                 \
    X(pages)                                                                                  \
    X(copy_commands)                                                                          \
    X(copy_regions)                                                                           \
    X(source_buffers)                                                                         \
    X(regions_merged)                                                                         \
    X(timeline_waits)                                                                         \
    X(bytes)                                                                                  \
    X(cpu_plan_ns)                                                                            \
    X(cpu_publish_ns)                                                                         \
    X(cpu_wait_ns)                                                                            \
    X(gpu_completion_ns)                                                                      \
    X(fallbacks)                                                                              \
    X(publication_failures)                                                                   \
    X(duplicate_submit_skips)                                                                 \
    X(overflow_allocations)                                                                   \
    X(demand_flushes)                                                                         \
    X(baseline_requests)                                                                      \
    X(sidecar_labels)                                                                         \
    X(sidecar_waits)

} // namespace

ReadbackOptimizationStats GetReadbackOptimizationStats() noexcept {
    ReadbackOptimizationStats result{};
#define LSX4_LOAD_READBACK_STAT(field)                                                        \
    result.field = g_readback_stats.field.load(std::memory_order_relaxed);
    LSX4_READBACK_STATS_FIELDS(LSX4_LOAD_READBACK_STAT)
#undef LSX4_LOAD_READBACK_STAT
    return result;
}

void ResetReadbackOptimizationStats() noexcept {
#define LSX4_RESET_READBACK_STAT(field)                                                       \
    g_readback_stats.field.store(0, std::memory_order_relaxed);
    LSX4_READBACK_STATS_FIELDS(LSX4_RESET_READBACK_STAT)
#undef LSX4_RESET_READBACK_STAT
    g_last_readback_retirement_tick.store(0, std::memory_order_relaxed);
}

void AccumulateReadbackOptimizationStats(const ReadbackOptimizationStats& delta) noexcept {
#define LSX4_ADD_READBACK_STAT(field)                                                         \
    if (delta.field != 0) {                                                                    \
        g_readback_stats.field.fetch_add(delta.field, std::memory_order_relaxed);              \
    }
    LSX4_READBACK_STATS_FIELDS(LSX4_ADD_READBACK_STAT)
#undef LSX4_ADD_READBACK_STAT
}

void RecordReadbackTimelineRetirement(const u64 gpu_tick) noexcept {
    if (gpu_tick == 0) {
        return;
    }
    const u64 previous =
        g_last_readback_retirement_tick.exchange(gpu_tick, std::memory_order_relaxed);
    if (previous != gpu_tick) {
        g_readback_stats.timeline_waits.fetch_add(1, std::memory_order_relaxed);
    }
}

#undef LSX4_READBACK_STATS_FIELDS

} // namespace VideoCore
