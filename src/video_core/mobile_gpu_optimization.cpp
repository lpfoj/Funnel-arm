// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/mobile_gpu_optimization.h"

#include <algorithm>
#include <atomic>
#include <mutex>

namespace VideoCore {
namespace {

struct AtomicMobileGpuState {
    std::atomic_bool enabled{false};
    std::atomic_bool coarse_shading_available{false};
    std::atomic_bool physical_drs_available{false};
    std::atomic_bool physical_drs_resolution_floor{false};
    std::atomic<u32> scale_percent{100};
    std::atomic<u32> overload_streak{0};
    std::atomic<u32> recovery_streak{0};
    std::atomic<u64> frame_interval_ema_ns{0};
    std::atomic<u64> memory_used_bytes{0};
    std::atomic<u64> memory_budget_bytes{0};

    std::atomic<u64> observed_frames{0};
    std::atomic<u64> observed_memory_samples{0};
    std::atomic<u64> submitted_ticks{0};
    std::atomic<u64> completed_ticks{0};
    std::atomic<u64> max_gpu_backlog{0};
    std::atomic<u64> scale_down_events{0};
    std::atomic<u64> scale_up_events{0};
    std::atomic<u64> full_rate_draws{0};
    std::atomic<u64> coarse_rate_draws{0};
    std::atomic<u64> physical_scaled_draws{0};
    std::atomic<u64> physical_scale_fallbacks{0};
    std::atomic<u64> physical_resolve_blits{0};
    std::atomic<u64> physical_pixels_saved{0};
    std::atomic<u64> shading_stats_epoch{1};
    std::atomic<u64> fsr_frames{0};
    std::atomic<u64> source_smaller_than_output_frames{0};
    std::atomic<u64> exact_surface_reuses{0};
    std::atomic<u64> exact_surface_reuse_bytes{0};
    std::atomic<u64> surface_stats_epoch{1};
    std::atomic<u64> residency_gc_passes{0};
    std::atomic<u64> residency_gc_evictions{0};
};

AtomicMobileGpuState state;
std::mutex transition_callback_mutex;
MobileGpuOptimizationTransitionCallback transition_callback{};
void* transition_callback_context{};

void UpdateMaximum(std::atomic<u64>& destination, const u64 value) noexcept {
    u64 current = destination.load(std::memory_order_relaxed);
    while (current < value &&
           !destination.compare_exchange_weak(current, value, std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
    }
}

u64 UpdateEma(std::atomic<u64>& destination, const u64 sample) noexcept {
    u64 current = destination.load(std::memory_order_relaxed);
    for (;;) {
        // 1/8 new sample: responsive enough for scene changes without toggling
        // quality on individual shader/pipeline stutters.
        const u64 next = current == 0 ? sample : current - current / 8 + sample / 8;
        if (destination.compare_exchange_weak(current, next, std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
            return next;
        }
    }
}

bool MemoryPressureAtLeast(const u64 numerator, const u64 denominator) noexcept {
    const u64 budget = state.memory_budget_bytes.load(std::memory_order_relaxed);
    const u64 used = state.memory_used_bytes.load(std::memory_order_relaxed);
    return budget != 0 && used >= budget / denominator * numerator;
}

u32 NextLowerScale(const u32 current) noexcept {
    if (current > 85) {
        return 85;
    }
    if (current > 70) {
        return 70;
    }
    if (current > 60) {
        return 60;
    }
    return 50;
}

u32 NextHigherScale(const u32 current) noexcept {
    if (current < 60) {
        return 60;
    }
    if (current < 70) {
        return 70;
    }
    if (current < 85) {
        return 85;
    }
    return 100;
}

} // namespace

void SetManagedMobileGpuOptimization(const bool enabled) noexcept {
    // Calls are also a per-launch controller reset. Temporarily unpublish the
    // policy even when its value is unchanged so a new title cannot inherit
    // the previous title's EMA, memory pressure, or reduced shading rate.
    state.enabled.store(false, std::memory_order_release);
    state.overload_streak.store(0, std::memory_order_relaxed);
    state.recovery_streak.store(0, std::memory_order_relaxed);
    state.frame_interval_ema_ns.store(0, std::memory_order_relaxed);
    state.memory_used_bytes.store(0, std::memory_order_relaxed);
    state.memory_budget_bytes.store(0, std::memory_order_relaxed);
    state.physical_drs_resolution_floor.store(false,
                                              std::memory_order_relaxed);
    state.submitted_ticks.store(0, std::memory_order_relaxed);
    state.completed_ticks.store(0, std::memory_order_relaxed);
    state.scale_percent.store(100, std::memory_order_release);
    if (enabled) {
        // Publish the policy only after all controller inputs have been reset,
        // so the render thread cannot briefly act on stale pressure data.
        state.enabled.store(true, std::memory_order_release);
    }
    std::scoped_lock callback_lock{transition_callback_mutex};
    if (transition_callback != nullptr) {
        transition_callback(transition_callback_context, enabled);
    }
}

bool IsManagedMobileGpuOptimization() noexcept {
    return state.enabled.load(std::memory_order_acquire);
}

void SetManagedMobileGpuOptimizationTransitionCallback(
    const MobileGpuOptimizationTransitionCallback callback,
    void* const context) noexcept {
    std::scoped_lock callback_lock{transition_callback_mutex};
    transition_callback = callback;
    transition_callback_context = context;
}

void SetMobileGpuCoarseShadingAvailable(const bool available) noexcept {
    state.coarse_shading_available.store(available, std::memory_order_release);
    if (!available &&
        !state.physical_drs_available.load(std::memory_order_acquire)) {
        state.scale_percent.store(100, std::memory_order_release);
        state.overload_streak.store(0, std::memory_order_relaxed);
    }
}

void SetMobileGpuPhysicalDrsAvailable(const bool available) noexcept {
    state.physical_drs_available.store(available, std::memory_order_release);
    if (!available &&
        !state.coarse_shading_available.load(std::memory_order_acquire)) {
        state.scale_percent.store(100, std::memory_order_release);
        state.overload_streak.store(0, std::memory_order_relaxed);
    }
}

void SetMobileGpuPhysicalDrsResolutionFloor(const bool active) noexcept {
    state.physical_drs_resolution_floor.store(active,
                                              std::memory_order_release);
    if (active) {
        state.scale_percent.store(100, std::memory_order_release);
        state.overload_streak.store(0, std::memory_order_relaxed);
        state.recovery_streak.store(0, std::memory_order_relaxed);
    }
}

void ObserveMobileGpuFrame(const u64 interval_nanoseconds, const u64 submitted_tick,
                           const u64 completed_tick) noexcept {
    if (!IsManagedMobileGpuOptimization()) {
        return;
    }

    state.observed_frames.fetch_add(1, std::memory_order_relaxed);
    state.submitted_ticks.store(submitted_tick, std::memory_order_relaxed);
    state.completed_ticks.store(completed_tick, std::memory_order_relaxed);
    const u64 backlog = submitted_tick > completed_tick ? submitted_tick - completed_tick : 0;
    UpdateMaximum(state.max_gpu_backlog, backlog);
    const u64 interval_ema =
        interval_nanoseconds == 0
            ? state.frame_interval_ema_ns.load(std::memory_order_relaxed)
            : UpdateEma(state.frame_interval_ema_ns, interval_nanoseconds);
    if (state.physical_drs_resolution_floor.load(
            std::memory_order_acquire)) {
        state.scale_percent.store(100, std::memory_order_release);
        state.overload_streak.store(0, std::memory_order_relaxed);
        state.recovery_streak.store(0, std::memory_order_relaxed);
        return;
    }

    // A slow presented-frame interval by itself is not GPU pressure: emulated
    // games frequently miss their frame target on the CPU while the graphics
    // timeline is already caught up. Treating that case as overload reduced
    // the complete render target (including text and HUD) to 50% without
    // improving frame rate. Scale only when timeline or memory pressure proves
    // that reducing attachment/depth/fragment work can help.
    const bool severely_slow =
        interval_ema >= 50'000'000ULL;
    const bool overloaded =
        backlog >= 3 || MemoryPressureAtLeast(9, 10) ||
        (backlog >= 2 && interval_ema >= 36'000'000ULL);
    const bool recovered =
        backlog <= 1 && !MemoryPressureAtLeast(3, 4) &&
        (interval_ema == 0 || interval_ema <= 29'000'000ULL);

    if (overloaded) {
        state.recovery_streak.store(0, std::memory_order_relaxed);
        const u32 streak =
            state.overload_streak.fetch_add(1, std::memory_order_relaxed) + 1;
        if (streak >= 8 && (streak % 8) == 0 &&
            (state.coarse_shading_available.load(std::memory_order_acquire) ||
             state.physical_drs_available.load(std::memory_order_acquire))) {
            const u32 current =
                state.scale_percent.load(std::memory_order_acquire);
            // At <=20 FPS, 85% is too small a reduction to separate fragment
            // pressure from noise. Move directly to 70% (51% fewer pixels),
            // then to 50% (75% fewer pixels) if the interval remains severe.
            const u32 next =
                severely_slow ? (current > 70 ? 70u : 50u)
                              : NextLowerScale(current);
            if (next != current) {
                state.scale_percent.store(next, std::memory_order_release);
                state.scale_down_events.fetch_add(1,
                                                  std::memory_order_relaxed);
            }
        }
        return;
    }

    state.overload_streak.store(0, std::memory_order_relaxed);
    if (!recovered) {
        state.recovery_streak.store(0, std::memory_order_relaxed);
        return;
    }
    const u32 recovery =
        state.recovery_streak.fetch_add(1, std::memory_order_relaxed) + 1;
    if (recovery >= 120) {
        const u32 current =
            state.scale_percent.load(std::memory_order_acquire);
        const u32 next = NextHigherScale(current);
        if (next != current) {
            state.scale_percent.store(next, std::memory_order_release);
            state.scale_up_events.fetch_add(1, std::memory_order_relaxed);
        }
        state.recovery_streak.store(0, std::memory_order_relaxed);
    }
}

void ObserveMobileGpuMemory(const u64 used_bytes, const u64 budget_bytes) noexcept {
    if (!IsManagedMobileGpuOptimization()) {
        return;
    }
    state.observed_memory_samples.fetch_add(1, std::memory_order_relaxed);
    state.memory_used_bytes.store(used_bytes, std::memory_order_relaxed);
    state.memory_budget_bytes.store(budget_bytes, std::memory_order_relaxed);
}

u32 MobileGpuEffectiveScalePercent() noexcept {
    return IsManagedMobileGpuOptimization()
               ? state.scale_percent.load(std::memory_order_acquire)
               : 100;
}

bool MobileGpuWantsCoarseShading() noexcept {
    return IsManagedMobileGpuOptimization() &&
           state.coarse_shading_available.load(std::memory_order_acquire) &&
           state.scale_percent.load(std::memory_order_acquire) <= 50;
}

bool MobileGpuWantsPhysicalDrs() noexcept {
    return IsManagedMobileGpuOptimization() &&
           state.physical_drs_available.load(std::memory_order_acquire) &&
           state.scale_percent.load(std::memory_order_acquire) < 100;
}

bool MobileGpuWantsFsr(const u32 source_width, const u32 source_height,
                       const u32 output_width, const u32 output_height) noexcept {
    if (!IsManagedMobileGpuOptimization()) {
        return false;
    }
    const bool upscale =
        source_width < output_width && source_height < output_height;
    if (upscale) {
        state.source_smaller_than_output_frames.fetch_add(1, std::memory_order_relaxed);
        state.fsr_frames.fetch_add(1, std::memory_order_relaxed);
    }
    return upscale;
}

void RecordMobileGpuShadingDecision(const bool coarse) noexcept {
    if (!state.enabled.load(std::memory_order_relaxed)) {
        return;
    }
    // This is a per-draw hot path. Keep the common operation thread-local and
    // amortize the atomic publication so telemetry cannot become a renderer
    // bottleneck of its own.
    struct LocalShadingStats {
        u64 epoch{};
        u32 full{};
        u32 coarse{};
    };
    static thread_local LocalShadingStats local;
    const u64 epoch = state.shading_stats_epoch.load(std::memory_order_relaxed);
    if (local.epoch != epoch) {
        local = {.epoch = epoch};
    }
    if (coarse) {
        ++local.coarse;
    } else {
        ++local.full;
    }
    constexpr u32 PublishInterval = 256;
    if (local.full + local.coarse < PublishInterval) {
        return;
    }
    if (state.shading_stats_epoch.load(std::memory_order_relaxed) == epoch) {
        state.full_rate_draws.fetch_add(local.full, std::memory_order_relaxed);
        state.coarse_rate_draws.fetch_add(local.coarse, std::memory_order_relaxed);
    }
    local.full = 0;
    local.coarse = 0;
}

void RecordMobileGpuPhysicalDrsDecision(const bool scaled, const bool fallback,
                                        const u64 canonical_pixels,
                                        const u64 physical_pixels) noexcept {
    if (!state.enabled.load(std::memory_order_relaxed)) {
        return;
    }
    if (scaled) {
        state.physical_scaled_draws.fetch_add(1, std::memory_order_relaxed);
        if (canonical_pixels > physical_pixels) {
            state.physical_pixels_saved.fetch_add(canonical_pixels - physical_pixels,
                                                  std::memory_order_relaxed);
        }
    } else if (fallback) {
        state.physical_scale_fallbacks.fetch_add(1, std::memory_order_relaxed);
    }
}

void RecordMobileGpuPhysicalDrsResolve() noexcept {
    if (state.enabled.load(std::memory_order_relaxed)) {
        state.physical_resolve_blits.fetch_add(1, std::memory_order_relaxed);
    }
}

void RecordMobileGpuSurfaceReuse(const u64 bytes) noexcept {
    if (!state.enabled.load(std::memory_order_relaxed)) {
        return;
    }
    // FindImage can hit this path many times per draw. Batch both counters just
    // like shading telemetry to keep cache-hit accounting out of the hot path.
    struct LocalSurfaceStats {
        u64 epoch{};
        u64 count{};
        u64 bytes{};
    };
    static thread_local LocalSurfaceStats local;
    const u64 epoch = state.surface_stats_epoch.load(std::memory_order_relaxed);
    if (local.epoch != epoch) {
        local = {.epoch = epoch};
    }
    ++local.count;
    local.bytes += bytes;
    constexpr u64 PublishInterval = 128;
    if (local.count < PublishInterval) {
        return;
    }
    if (state.surface_stats_epoch.load(std::memory_order_relaxed) == epoch) {
        state.exact_surface_reuses.fetch_add(local.count, std::memory_order_relaxed);
        state.exact_surface_reuse_bytes.fetch_add(local.bytes, std::memory_order_relaxed);
    }
    local.count = 0;
    local.bytes = 0;
}

void RecordMobileGpuResidencyGc(const u64 evictions) noexcept {
    if (!IsManagedMobileGpuOptimization()) {
        return;
    }
    state.residency_gc_passes.fetch_add(1, std::memory_order_relaxed);
    state.residency_gc_evictions.fetch_add(evictions, std::memory_order_relaxed);
}

MobileGpuOptimizationStats GetMobileGpuOptimizationStats() noexcept {
    return {
        .enabled = IsManagedMobileGpuOptimization() ? 1ULL : 0ULL,
        .coarse_shading_available =
            state.coarse_shading_available.load(std::memory_order_relaxed) ? 1ULL : 0ULL,
        .physical_drs_available =
            state.physical_drs_available.load(std::memory_order_relaxed) ? 1ULL : 0ULL,
        .observed_frames = state.observed_frames.load(std::memory_order_relaxed),
        .observed_memory_samples =
            state.observed_memory_samples.load(std::memory_order_relaxed),
        .submitted_ticks = state.submitted_ticks.load(std::memory_order_relaxed),
        .completed_ticks = state.completed_ticks.load(std::memory_order_relaxed),
        .max_gpu_backlog = state.max_gpu_backlog.load(std::memory_order_relaxed),
        .frame_interval_nanoseconds =
            state.frame_interval_ema_ns.load(std::memory_order_relaxed),
        .memory_used_bytes = state.memory_used_bytes.load(std::memory_order_relaxed),
        .memory_budget_bytes = state.memory_budget_bytes.load(std::memory_order_relaxed),
        .effective_scale_percent = MobileGpuEffectiveScalePercent(),
        .scale_down_events = state.scale_down_events.load(std::memory_order_relaxed),
        .scale_up_events = state.scale_up_events.load(std::memory_order_relaxed),
        .full_rate_draws = state.full_rate_draws.load(std::memory_order_relaxed),
        .coarse_rate_draws = state.coarse_rate_draws.load(std::memory_order_relaxed),
        .physical_scaled_draws =
            state.physical_scaled_draws.load(std::memory_order_relaxed),
        .physical_scale_fallbacks =
            state.physical_scale_fallbacks.load(std::memory_order_relaxed),
        .physical_resolve_blits =
            state.physical_resolve_blits.load(std::memory_order_relaxed),
        .physical_pixels_saved =
            state.physical_pixels_saved.load(std::memory_order_relaxed),
        .fsr_frames = state.fsr_frames.load(std::memory_order_relaxed),
        .source_smaller_than_output_frames =
            state.source_smaller_than_output_frames.load(std::memory_order_relaxed),
        .exact_surface_reuses =
            state.exact_surface_reuses.load(std::memory_order_relaxed),
        .exact_surface_reuse_bytes =
            state.exact_surface_reuse_bytes.load(std::memory_order_relaxed),
        .residency_gc_passes =
            state.residency_gc_passes.load(std::memory_order_relaxed),
        .residency_gc_evictions =
            state.residency_gc_evictions.load(std::memory_order_relaxed),
    };
}

void ResetMobileGpuOptimizationStats() noexcept {
    // Per-thread batches from the previous generation are discarded on their
    // next draw instead of being published into freshly reset counters.
    state.shading_stats_epoch.fetch_add(1, std::memory_order_acq_rel);
    state.surface_stats_epoch.fetch_add(1, std::memory_order_acq_rel);
    state.observed_frames.store(0, std::memory_order_relaxed);
    state.observed_memory_samples.store(0, std::memory_order_relaxed);
    state.submitted_ticks.store(0, std::memory_order_relaxed);
    state.completed_ticks.store(0, std::memory_order_relaxed);
    state.max_gpu_backlog.store(0, std::memory_order_relaxed);
    state.frame_interval_ema_ns.store(0, std::memory_order_relaxed);
    state.memory_used_bytes.store(0, std::memory_order_relaxed);
    state.memory_budget_bytes.store(0, std::memory_order_relaxed);
    state.scale_down_events.store(0, std::memory_order_relaxed);
    state.scale_up_events.store(0, std::memory_order_relaxed);
    state.full_rate_draws.store(0, std::memory_order_relaxed);
    state.coarse_rate_draws.store(0, std::memory_order_relaxed);
    state.physical_scaled_draws.store(0, std::memory_order_relaxed);
    state.physical_scale_fallbacks.store(0, std::memory_order_relaxed);
    state.physical_resolve_blits.store(0, std::memory_order_relaxed);
    state.physical_pixels_saved.store(0, std::memory_order_relaxed);
    state.fsr_frames.store(0, std::memory_order_relaxed);
    state.source_smaller_than_output_frames.store(0, std::memory_order_relaxed);
    state.exact_surface_reuses.store(0, std::memory_order_relaxed);
    state.exact_surface_reuse_bytes.store(0, std::memory_order_relaxed);
    state.residency_gc_passes.store(0, std::memory_order_relaxed);
    state.residency_gc_evictions.store(0, std::memory_order_relaxed);
}

} // namespace VideoCore
