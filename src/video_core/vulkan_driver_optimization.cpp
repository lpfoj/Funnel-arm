// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/vulkan_driver_optimization.h"

#include <atomic>
#include <string>

#ifdef __ANDROID__
#include <unistd.h>
#endif

namespace VideoCore {
namespace {

std::atomic_bool enabled{false};
// Even values are active collection epochs; odd values mean reset in progress.
std::atomic<u64> stats_generation{2};
std::atomic<u64> pipeline_bind_attempts{0};
std::atomic<u64> pipeline_binds_elided{0};
std::atomic<u64> descriptor_bind_attempts{0};
std::atomic<u64> descriptor_binds_elided{0};
std::atomic<u64> push_constant_attempts{0};
std::atomic<u64> push_constants_elided{0};
std::atomic<u64> dynamic_state_commits{0};
std::atomic<u64> dynamic_state_full_replays{0};
std::atomic<u64> barriers_elided{0};
std::atomic<u64> render_scope_reuses{0};
std::atomic<u64> transfer_requests{0};
std::atomic<u64> transfer_flushes{0};
std::atomic<u64> transfer_regions{0};
std::atomic<u64> transfer_copy_calls{0};
std::atomic<u64> pm4_plan_lookups{0};
std::atomic<u64> pm4_plan_hits{0};
std::atomic<u64> barrier_requests{0};
std::atomic<u64> barrier_batches{0};
std::atomic<u64> barrier_regions{0};
std::atomic<u64> barrier_regions_merged{0};
std::atomic<u64> queue_submits{0};
std::atomic<u64> queue_submit_logical_packets{0};
std::atomic<u64> queue_submit_command_buffers{0};
std::atomic<u64> queue_submit_calls_saved{0};
std::atomic<u64> queue_submit_forced_boundaries{0};

#ifdef __ANDROID__
bool BisectionMarkerAbsent(const char* name) noexcept {
    const std::string path =
        std::string{"/data/data/app.lsx4.android/files/lsx4-home/"} + name;
    return ::access(path.c_str(), F_OK) != 0;
}
#endif

struct LocalStats {
    u64 generation{};
    u32 events{};
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
    u64 queue_submits{};
    u64 queue_submit_logical_packets{};
    u64 queue_submit_command_buffers{};
    u64 queue_submit_calls_saved{};
    u64 queue_submit_forced_boundaries{};
};

LocalStats& ThreadStats() noexcept {
    thread_local LocalStats stats{
        .generation = stats_generation.load(std::memory_order_relaxed),
    };
    return stats;
}

void FlushThreadStats(LocalStats& stats) noexcept {
    const u64 current_generation =
        stats_generation.load(std::memory_order_acquire);
    if ((current_generation & 1u) != 0 ||
        stats.generation != current_generation) {
        stats = {.generation = current_generation};
        return;
    }

    const auto accumulate = [](std::atomic<u64>& destination, const u64 value) {
        if (value != 0) {
            destination.fetch_add(value, std::memory_order_relaxed);
        }
    };
    accumulate(pipeline_bind_attempts, stats.pipeline_bind_attempts);
    accumulate(pipeline_binds_elided, stats.pipeline_binds_elided);
    accumulate(descriptor_bind_attempts, stats.descriptor_bind_attempts);
    accumulate(descriptor_binds_elided, stats.descriptor_binds_elided);
    accumulate(push_constant_attempts, stats.push_constant_attempts);
    accumulate(push_constants_elided, stats.push_constants_elided);
    accumulate(dynamic_state_commits, stats.dynamic_state_commits);
    accumulate(dynamic_state_full_replays, stats.dynamic_state_full_replays);
    accumulate(barriers_elided, stats.barriers_elided);
    accumulate(render_scope_reuses, stats.render_scope_reuses);
    accumulate(transfer_requests, stats.transfer_requests);
    accumulate(transfer_flushes, stats.transfer_flushes);
    accumulate(transfer_regions, stats.transfer_regions);
    accumulate(transfer_copy_calls, stats.transfer_copy_calls);
    accumulate(pm4_plan_lookups, stats.pm4_plan_lookups);
    accumulate(pm4_plan_hits, stats.pm4_plan_hits);
    accumulate(barrier_requests, stats.barrier_requests);
    accumulate(barrier_batches, stats.barrier_batches);
    accumulate(barrier_regions, stats.barrier_regions);
    accumulate(barrier_regions_merged, stats.barrier_regions_merged);
    accumulate(queue_submits, stats.queue_submits);
    accumulate(queue_submit_logical_packets,
               stats.queue_submit_logical_packets);
    accumulate(queue_submit_command_buffers,
               stats.queue_submit_command_buffers);
    accumulate(queue_submit_calls_saved, stats.queue_submit_calls_saved);
    accumulate(queue_submit_forced_boundaries,
               stats.queue_submit_forced_boundaries);
    stats = {.generation = current_generation};
}

void MaybeFlushThreadStats(LocalStats& stats) noexcept {
    constexpr u32 BatchSize = 256;
    if (++stats.events >= BatchSize) {
        FlushThreadStats(stats);
    }
}

} // namespace

void SetManagedVulkanDriverOptimization(const bool active) noexcept {
    enabled.store(active, std::memory_order_relaxed);
}

bool IsManagedVulkanDriverOptimization() noexcept {
    return enabled.load(std::memory_order_relaxed);
}

bool IsManagedVulkanStateDeduplication() noexcept {
#ifdef __ANDROID__
    static const bool allowed =
        BisectionMarkerAbsent("run-bisect-disable-vk-state-dedup");
    return IsManagedVulkanDriverOptimization() && allowed;
#else
    return IsManagedVulkanDriverOptimization();
#endif
}

bool IsManagedVulkanPipelineBindDeduplication() noexcept {
#ifdef __ANDROID__
    static const bool allowed =
        BisectionMarkerAbsent("run-bisect-disable-vk-pipeline-bind-dedup");
    return IsManagedVulkanStateDeduplication() && allowed;
#else
    return IsManagedVulkanStateDeduplication();
#endif
}

bool IsManagedVulkanDescriptorDeduplication() noexcept {
#ifdef __ANDROID__
    // Re-emitting push descriptors is required on the Android path.  Some
    // Qualcomm drivers internally rotate push-descriptor storage and a
    // signature-only elision can leave a draw observing an older image
    // binding for one frame.  It also costs more to fingerprint the complete
    // descriptor stream than to record the command in descriptor-heavy
    // scenes.  Keep pipeline, push-constant, vertex and dynamic-state
    // deduplication enabled; only the unsafe descriptor elision is disabled.
    return false;
#else
    return IsManagedVulkanStateDeduplication();
#endif
}

bool IsManagedVulkanVertexBindDeduplication() noexcept {
#ifdef __ANDROID__
    static const bool allowed =
        BisectionMarkerAbsent("run-bisect-disable-vk-vertex-bind-dedup");
    return IsManagedVulkanStateDeduplication() && allowed;
#else
    return IsManagedVulkanStateDeduplication();
#endif
}

bool IsManagedVulkanTransferCoalescing() noexcept {
#ifdef __ANDROID__
    static const bool allowed =
        BisectionMarkerAbsent("run-bisect-disable-vk-transfer-coalescing");
    return IsManagedVulkanDriverOptimization() && allowed;
#else
    return IsManagedVulkanDriverOptimization();
#endif
}

bool IsManagedVulkanSubmitCoalescing() noexcept {
#ifdef __ANDROID__
    return false;
#else
    return IsManagedVulkanDriverOptimization();
#endif
}

bool IsManagedVulkanRenderScopeReuse() noexcept {
#ifdef __ANDROID__
    static const bool allowed =
        BisectionMarkerAbsent("run-bisect-disable-vk-render-scope-reuse");
    return IsManagedVulkanDriverOptimization() && allowed;
#else
    return IsManagedVulkanDriverOptimization();
#endif
}

void RecordVulkanPipelineBind(const bool elided) noexcept {
    auto& stats = ThreadStats();
    ++stats.pipeline_bind_attempts;
    if (elided) {
        ++stats.pipeline_binds_elided;
    }
    MaybeFlushThreadStats(stats);
}

void RecordVulkanDescriptorBind(const bool elided) noexcept {
    auto& stats = ThreadStats();
    ++stats.descriptor_bind_attempts;
    if (elided) {
        ++stats.descriptor_binds_elided;
    }
    MaybeFlushThreadStats(stats);
}

void RecordVulkanPushConstants(const bool elided) noexcept {
    auto& stats = ThreadStats();
    ++stats.push_constant_attempts;
    if (elided) {
        ++stats.push_constants_elided;
    }
    MaybeFlushThreadStats(stats);
}

void RecordVulkanDynamicState(const bool full_replay) noexcept {
    auto& stats = ThreadStats();
    ++stats.dynamic_state_commits;
    if (full_replay) {
        ++stats.dynamic_state_full_replays;
    }
    MaybeFlushThreadStats(stats);
}

void RecordVulkanBarrierElision() noexcept {
    auto& stats = ThreadStats();
    ++stats.barriers_elided;
    MaybeFlushThreadStats(stats);
}

void RecordVulkanRenderScopeReuse() noexcept {
    auto& stats = ThreadStats();
    ++stats.render_scope_reuses;
    MaybeFlushThreadStats(stats);
}

void RecordVulkanTransferRequest(const u64 regions) noexcept {
    auto& stats = ThreadStats();
    ++stats.transfer_requests;
    stats.transfer_regions += regions;
    MaybeFlushThreadStats(stats);
}

void RecordVulkanTransferFlush(const u64 regions, const u64 copy_calls) noexcept {
    auto& stats = ThreadStats();
    ++stats.transfer_flushes;
    stats.transfer_copy_calls += copy_calls;
    (void)regions;
    MaybeFlushThreadStats(stats);
}

void RecordVulkanPm4Plan(const bool hit) noexcept {
    auto& stats = ThreadStats();
    ++stats.pm4_plan_lookups;
    if (hit) {
        ++stats.pm4_plan_hits;
    }
    MaybeFlushThreadStats(stats);
}

void RecordVulkanBarrierBatch(const u64 requested_regions,
                              const u64 emitted_regions) noexcept {
    auto& stats = ThreadStats();
    ++stats.barrier_requests;
    ++stats.barrier_batches;
    stats.barrier_regions += emitted_regions;
    stats.barrier_regions_merged +=
        requested_regions > emitted_regions ? requested_regions - emitted_regions : 0;
    MaybeFlushThreadStats(stats);
}

void RecordVulkanQueueSubmit(const u64 logical_packets,
                             const u64 command_buffers,
                             const bool forced_boundary) noexcept {
    auto& stats = ThreadStats();
    ++stats.queue_submits;
    stats.queue_submit_logical_packets += logical_packets;
    stats.queue_submit_command_buffers += command_buffers;
    stats.queue_submit_calls_saved +=
        logical_packets > 1 ? logical_packets - 1 : 0;
    if (forced_boundary) {
        ++stats.queue_submit_forced_boundaries;
    }
    MaybeFlushThreadStats(stats);
}

VulkanDriverOptimizationStats GetVulkanDriverOptimizationStats() noexcept {
    return {
        .enabled = IsManagedVulkanDriverOptimization() ? 1ULL : 0ULL,
        .pipeline_bind_attempts =
            pipeline_bind_attempts.load(std::memory_order_relaxed),
        .pipeline_binds_elided =
            pipeline_binds_elided.load(std::memory_order_relaxed),
        .descriptor_bind_attempts =
            descriptor_bind_attempts.load(std::memory_order_relaxed),
        .descriptor_binds_elided =
            descriptor_binds_elided.load(std::memory_order_relaxed),
        .push_constant_attempts =
            push_constant_attempts.load(std::memory_order_relaxed),
        .push_constants_elided =
            push_constants_elided.load(std::memory_order_relaxed),
        .dynamic_state_commits =
            dynamic_state_commits.load(std::memory_order_relaxed),
        .dynamic_state_full_replays =
            dynamic_state_full_replays.load(std::memory_order_relaxed),
        .barriers_elided = barriers_elided.load(std::memory_order_relaxed),
        .render_scope_reuses =
            render_scope_reuses.load(std::memory_order_relaxed),
        .transfer_requests = transfer_requests.load(std::memory_order_relaxed),
        .transfer_flushes = transfer_flushes.load(std::memory_order_relaxed),
        .transfer_regions = transfer_regions.load(std::memory_order_relaxed),
        .transfer_copy_calls =
            transfer_copy_calls.load(std::memory_order_relaxed),
        .pm4_plan_lookups = pm4_plan_lookups.load(std::memory_order_relaxed),
        .pm4_plan_hits = pm4_plan_hits.load(std::memory_order_relaxed),
        .barrier_requests = barrier_requests.load(std::memory_order_relaxed),
        .barrier_batches = barrier_batches.load(std::memory_order_relaxed),
        .barrier_regions = barrier_regions.load(std::memory_order_relaxed),
        .barrier_regions_merged =
            barrier_regions_merged.load(std::memory_order_relaxed),
        .queue_submits = queue_submits.load(std::memory_order_relaxed),
        .queue_submit_logical_packets =
            queue_submit_logical_packets.load(std::memory_order_relaxed),
        .queue_submit_command_buffers =
            queue_submit_command_buffers.load(std::memory_order_relaxed),
        .queue_submit_calls_saved =
            queue_submit_calls_saved.load(std::memory_order_relaxed),
        .queue_submit_forced_boundaries =
            queue_submit_forced_boundaries.load(std::memory_order_relaxed),
    };
}

void ResetVulkanDriverOptimizationStats() noexcept {
    // In-flight per-thread batches from the previous A/B epoch are discarded
    // on their next flush instead of leaking into the reset counters.
    stats_generation.fetch_add(1, std::memory_order_acq_rel);
    pipeline_bind_attempts.store(0, std::memory_order_relaxed);
    pipeline_binds_elided.store(0, std::memory_order_relaxed);
    descriptor_bind_attempts.store(0, std::memory_order_relaxed);
    descriptor_binds_elided.store(0, std::memory_order_relaxed);
    push_constant_attempts.store(0, std::memory_order_relaxed);
    push_constants_elided.store(0, std::memory_order_relaxed);
    dynamic_state_commits.store(0, std::memory_order_relaxed);
    dynamic_state_full_replays.store(0, std::memory_order_relaxed);
    barriers_elided.store(0, std::memory_order_relaxed);
    render_scope_reuses.store(0, std::memory_order_relaxed);
    transfer_requests.store(0, std::memory_order_relaxed);
    transfer_flushes.store(0, std::memory_order_relaxed);
    transfer_regions.store(0, std::memory_order_relaxed);
    transfer_copy_calls.store(0, std::memory_order_relaxed);
    pm4_plan_lookups.store(0, std::memory_order_relaxed);
    pm4_plan_hits.store(0, std::memory_order_relaxed);
    barrier_requests.store(0, std::memory_order_relaxed);
    barrier_batches.store(0, std::memory_order_relaxed);
    barrier_regions.store(0, std::memory_order_relaxed);
    barrier_regions_merged.store(0, std::memory_order_relaxed);
    queue_submits.store(0, std::memory_order_relaxed);
    queue_submit_logical_packets.store(0, std::memory_order_relaxed);
    queue_submit_command_buffers.store(0, std::memory_order_relaxed);
    queue_submit_calls_saved.store(0, std::memory_order_relaxed);
    queue_submit_forced_boundaries.store(0, std::memory_order_relaxed);
    stats_generation.fetch_add(1, std::memory_order_release);
}

} // namespace VideoCore
