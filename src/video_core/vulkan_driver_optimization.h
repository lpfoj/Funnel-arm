// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

struct VulkanDriverOptimizationStats {
    u64 enabled{};
    u64 pipeline_bind_attempts{};
    u64 pipeline_binds_elided{};
    u64 descriptor_bind_attempts{};
    u64 descriptor_binds_elided{};
    u64 push_constant_attempts{};
    u64 push_constants_elided{};
    u64 dynamic_state_commits{};
    u64 dynamic_state_full_replays{};
    u64 barriers_elided{};
    u64 render_scope_reuses{};
    u64 transfer_requests{};
    u64 transfer_flushes{};
    u64 transfer_regions{};
    u64 transfer_copy_calls{};
    u64 pm4_plan_lookups{};
    u64 pm4_plan_hits{};
    u64 barrier_requests{};
    u64 barrier_batches{};
    u64 barrier_regions{};
    u64 barrier_regions_merged{};
    // Physical vkQueueSubmit calls versus the logical packets and command
    // buffers carried by them. A saved call is one logical packet folded into
    // an already-required physical call.
    u64 queue_submits{};
    u64 queue_submit_logical_packets{};
    u64 queue_submit_command_buffers{};
    u64 queue_submit_calls_saved{};
    u64 queue_submit_forced_boundaries{};
};

void SetManagedVulkanDriverOptimization(bool enabled) noexcept;
[[nodiscard]] bool IsManagedVulkanDriverOptimization() noexcept;
[[nodiscard]] bool IsManagedVulkanStateDeduplication() noexcept;
[[nodiscard]] bool IsManagedVulkanPipelineBindDeduplication() noexcept;
[[nodiscard]] bool IsManagedVulkanDescriptorDeduplication() noexcept;
[[nodiscard]] bool IsManagedVulkanVertexBindDeduplication() noexcept;
[[nodiscard]] bool IsManagedVulkanTransferCoalescing() noexcept;
[[nodiscard]] bool IsManagedVulkanSubmitCoalescing() noexcept;
[[nodiscard]] bool IsManagedVulkanRenderScopeReuse() noexcept;

void RecordVulkanPipelineBind(bool elided) noexcept;
void RecordVulkanDescriptorBind(bool elided) noexcept;
void RecordVulkanPushConstants(bool elided) noexcept;
void RecordVulkanDynamicState(bool full_replay) noexcept;
void RecordVulkanBarrierElision() noexcept;
void RecordVulkanRenderScopeReuse() noexcept;
void RecordVulkanTransferRequest(u64 regions) noexcept;
void RecordVulkanTransferFlush(u64 regions, u64 copy_calls) noexcept;
void RecordVulkanPm4Plan(bool hit) noexcept;
void RecordVulkanBarrierBatch(u64 requested_regions, u64 emitted_regions) noexcept;
void RecordVulkanQueueSubmit(u64 logical_packets = 1,
                             u64 command_buffers = 1,
                             bool forced_boundary = false) noexcept;

[[nodiscard]] VulkanDriverOptimizationStats
GetVulkanDriverOptimizationStats() noexcept;
void ResetVulkanDriverOptimizationStats() noexcept;

} // namespace VideoCore
