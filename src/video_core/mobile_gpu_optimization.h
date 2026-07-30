// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace VideoCore {

using MobileGpuOptimizationTransitionCallback =
    void (*)(void* context, bool enabled) noexcept;

struct MobileGpuOptimizationStats {
    u64 enabled{};
    u64 coarse_shading_available{};
    u64 physical_drs_available{};
    u64 observed_frames{};
    u64 observed_memory_samples{};
    u64 submitted_ticks{};
    u64 completed_ticks{};
    u64 max_gpu_backlog{};
    u64 frame_interval_nanoseconds{};
    u64 memory_used_bytes{};
    u64 memory_budget_bytes{};
    u64 effective_scale_percent{};
    u64 scale_down_events{};
    u64 scale_up_events{};
    u64 full_rate_draws{};
    u64 coarse_rate_draws{};
    u64 physical_scaled_draws{};
    u64 physical_scale_fallbacks{};
    u64 physical_resolve_blits{};
    u64 physical_pixels_saved{};
    u64 fsr_frames{};
    u64 source_smaller_than_output_frames{};
    u64 exact_surface_reuses{};
    u64 exact_surface_reuse_bytes{};
    u64 residency_gc_passes{};
    u64 residency_gc_evictions{};
};

/// Enables the Android/mobile GPU policy without changing the configured output
/// extent. Disabling it immediately restores full-rate shading and leaves all
/// existing renderer configuration untouched.
void SetManagedMobileGpuOptimization(bool enabled) noexcept;

[[nodiscard]] bool IsManagedMobileGpuOptimization() noexcept;

/// Registers the single live renderer for managed-policy transition
/// notifications. Registration and removal are serialized with callbacks, so
/// renderer shutdown cannot race an in-progress notification.
void SetManagedMobileGpuOptimizationTransitionCallback(
    MobileGpuOptimizationTransitionCallback callback,
    void* context) noexcept;

/// Publishes device capability after Vulkan feature/rate negotiation.
void SetMobileGpuCoarseShadingAvailable(bool available) noexcept;

/// Publishes whether the renderer can use a physically smaller color target
/// while preserving the canonical guest image and resolving it before any
/// guest-visible hazard.
void SetMobileGpuPhysicalDrsAvailable(bool available) noexcept;
void SetMobileGpuPhysicalDrsResolutionFloor(bool active) noexcept;

/// Feed one presented-frame sample to the hysteretic controller. Timeline
/// backlog is used to avoid reducing quality for CPU-bound frames.
void ObserveMobileGpuFrame(u64 interval_nanoseconds, u64 submitted_tick,
                           u64 completed_tick) noexcept;

/// Feed the latest VK_EXT_memory_budget sample.
void ObserveMobileGpuMemory(u64 used_bytes, u64 budget_bytes) noexcept;

/// 100 is full-rate shading. 50 is the linear-resolution workload equivalent
/// of a safe 2x2 coarse-fragment rate, preserving the configured output size.
/// This does not resize guest GNM render targets: their address/pitch/tile and
/// inter-pass alias contracts cannot be changed by a local presenter policy.
[[nodiscard]] u32 MobileGpuEffectiveScalePercent() noexcept;

[[nodiscard]] bool MobileGpuWantsCoarseShading() noexcept;
[[nodiscard]] bool MobileGpuWantsPhysicalDrs() noexcept;

/// FSR is selected only for a genuine upscale. It is never inserted for an
/// equal-size image or a downscale.
[[nodiscard]] bool MobileGpuWantsFsr(u32 source_width, u32 source_height,
                                    u32 output_width, u32 output_height) noexcept;

void RecordMobileGpuShadingDecision(bool coarse) noexcept;
void RecordMobileGpuPhysicalDrsDecision(bool scaled, bool fallback,
                                        u64 canonical_pixels,
                                        u64 physical_pixels) noexcept;
void RecordMobileGpuPhysicalDrsResolve() noexcept;
void RecordMobileGpuSurfaceReuse(u64 bytes) noexcept;
void RecordMobileGpuResidencyGc(u64 evictions) noexcept;

[[nodiscard]] MobileGpuOptimizationStats GetMobileGpuOptimizationStats() noexcept;
void ResetMobileGpuOptimizationStats() noexcept;

} // namespace VideoCore
