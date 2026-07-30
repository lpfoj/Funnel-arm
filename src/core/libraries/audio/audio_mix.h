// SPDX-FileCopyrightText: Copyright 2024-2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <span>

#include "common/types.h"

namespace Libraries::AudioOut {

struct AudioOptimizationStats {
    u64 conversion_calls{};
    u64 converted_frames{};
    u64 simd_frames{};
    u64 scalar_frames{};
    u64 conversion_nanoseconds{};
    u64 write_nanoseconds{};
    u64 partial_writes{};
    u64 write_errors{};
    u64 xruns{};
    u64 scheduler_wakeups{};
    u64 scheduler_batches{};
    u64 scheduler_ports{};
    u64 scheduler_mixed_ports{};
    u64 scheduler_nanoseconds{};
    u64 scheduler_max_batch_ports{};
    u64 zero_copy_handoffs{};
    u64 guest_copy_calls{};
    u64 guest_copy_bytes{};
    u64 guest_copy_wait_nanoseconds{};
    u64 continuity_failures{};
    u64 timestamp_order_failures{};
    u64 scheduler_underruns{};
    u64 channel_guard_passes{};
    u64 channel_guard_failures{};
    u64 scalar_fallback_buffers{};
    u64 regression_guard_runs{};
    u64 regression_guard_passes{};
    u64 regression_guard_failures{};
    u64 toggle_generation{};
    u64 managed_enabled{};
    u64 global_scalar_fallback{};
};

void SetManagedAudioSimdEnabled(bool enabled);
[[nodiscard]] bool IsManagedAudioSimdEnabled();
[[nodiscard]] AudioOptimizationStats GetAudioOptimizationStats();
void ResetAudioOptimizationStats();

enum class AudioSampleType : u8 {
    S16,
    Float32,
};

struct AudioBufferFormat {
    AudioSampleType sample_type{AudioSampleType::S16};
    u8 channels{2};
    bool standard_8ch{};
};

/// Per-stream differential guard. A generation change makes a managed toggle
/// effective on the very next buffer and re-arms SIMD/scalar comparison.
struct AudioConversionGuard {
    u64 generation{};
    bool validated{};
    bool force_scalar{};
};

struct AudioStreamGuard {
    u64 last_submitted_sequence{};
    u64 last_consumed_sequence{};
    u64 last_submit_timestamp_ns{};
    u64 last_consume_timestamp_ns{};
};

struct AudioStreamGuardResult {
    bool continuity_ok{true};
    bool timestamp_order_ok{true};
    bool underrun{};
};

struct AudioConvertResult {
    bool any_nonzero{};
    bool used_simd{};
    bool differential_guard_passed{};
    bool scalar_fallback{};
};

/// Converts one guest buffer into interleaved host float samples. Supported
/// input/output channel counts are 1, 2 and 8. Eight-channel input is mapped
/// semantically before stereo downmixing, including the Orbis "Std" layout.
[[nodiscard]] AudioConvertResult ConvertAudioBufferToFloat(const void* source, u32 frames,
                                                           AudioBufferFormat format,
                                                           u32 output_channels, float gain,
                                                           std::span<float> output,
                                                           AudioConversionGuard& guard);

/// Mixes two identically formatted guest buffers without an intermediate copy.
/// The destination is replaced atomically at buffer granularity by its caller.
/// S16 uses saturation and float uses [-1, 1] clipping.
[[nodiscard]] AudioConvertResult MixAudioBufferInPlace(void* destination, const void* source,
                                                       u32 frames, AudioBufferFormat format,
                                                       float source_gain,
                                                       AudioConversionGuard& guard);

[[nodiscard]] u64 AudioMonotonicNanoseconds() noexcept;

/// Title-agnostic stream-order guard used by both PS4 and isolated PS5 paths.
/// The known Bloodborne and Dreaming Sarah profiles are covered by the same
/// format/sequence contracts; no title or gameplay branch exists here.
void ObserveAudioStreamSubmit(AudioStreamGuard& guard, u64 sequence, u64 timestamp_ns) noexcept;
[[nodiscard]] AudioStreamGuardResult ObserveAudioStreamConsume(AudioStreamGuard& guard,
                                                               u64 sequence,
                                                               u64 submit_timestamp_ns,
                                                               u64 consume_timestamp_ns,
                                                               u64 expected_period_ns) noexcept;

void RecordAudioGuestCopy(u64 bytes, u64 wait_nanoseconds) noexcept;
void RecordAudioSchedulerWakeup() noexcept;
void RecordAudioSchedulerBatch(u32 ports, u32 mixed_ports, u64 scheduling_nanoseconds) noexcept;
void RecordAudioZeroCopyHandoff() noexcept;
void RecordAudioBackendWrite(u64 write_nanoseconds, s64 written_frames, u32 requested_frames,
                             u32 xrun_delta) noexcept;

/// Runs pure CPU differential and invariant fixtures. It is called
/// automatically when managed audio optimization is enabled. A failure forces
/// all subsequent buffers through the scalar reference until the next enable.
[[nodiscard]] bool RunAudioOptimizationRegressionGuards() noexcept;

} // namespace Libraries::AudioOut
