// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

// Process-wide counters intentionally use monotonically increasing totals.
// Managed Android can sample/reset them around an A/B interval without
// retaining a BufferCache or Liverpool instance.
struct ReadbackOptimizationStats {
    u64 snapshots{};
    u64 submit_snapshots{};
    u64 demand_snapshots{};
    u64 ranges{};
    u64 pages{};
    u64 copy_commands{};
    u64 copy_regions{};
    u64 source_buffers{};
    u64 regions_merged{};
    u64 timeline_waits{};
    u64 bytes{};
    u64 cpu_plan_ns{};
    u64 cpu_publish_ns{};
    u64 cpu_wait_ns{};
    u64 gpu_completion_ns{};
    u64 fallbacks{};
    u64 publication_failures{};
    u64 duplicate_submit_skips{};
    u64 overflow_allocations{};
    u64 demand_flushes{};
    u64 baseline_requests{};
    u64 sidecar_labels{};
    u64 sidecar_waits{};
};

[[nodiscard]] ReadbackOptimizationStats GetReadbackOptimizationStats() noexcept;
void ResetReadbackOptimizationStats() noexcept;

// Internal event API shared by BufferCache and the PM4 fence/label sidecar.
void AccumulateReadbackOptimizationStats(const ReadbackOptimizationStats& delta) noexcept;
// Retirement callbacks for the same scheduler tick share one Vulkan timeline
// wait. This helper deduplicates their metric without changing callback order.
void RecordReadbackTimelineRetirement(u64 gpu_tick) noexcept;

} // namespace VideoCore
