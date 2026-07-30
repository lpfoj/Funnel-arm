// SPDX-FileCopyrightText: Copyright 2024-2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/audio/audio_mix.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define LSX4_AUDIO_HAS_NEON
#endif

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace Libraries::AudioOut {
namespace {

constexpr float S16Scale = 1.0f / 32768.0f;
constexpr float CenterMix = 0.7071067811865475f;
constexpr float SurroundMix = 0.7071067811865475f;

#if defined(LSX4_AUDIO_HAS_NEON)
constexpr bool HasAudioSimd = true;
#else
constexpr bool HasAudioSimd = false;
#endif

struct AudioOptimizationCounters {
    // Optimized execution is armed only after the automatic differential
    // fixtures pass; merely compiling for a SIMD-capable CPU is insufficient.
    std::atomic_bool enabled{false};
    std::atomic_bool global_scalar_fallback{false};
    std::atomic<u64> generation{1};
    std::atomic<u64> conversion_calls{};
    std::atomic<u64> converted_frames{};
    std::atomic<u64> simd_frames{};
    std::atomic<u64> scalar_frames{};
    std::atomic<u64> conversion_nanoseconds{};
    std::atomic<u64> write_nanoseconds{};
    std::atomic<u64> partial_writes{};
    std::atomic<u64> write_errors{};
    std::atomic<u64> xruns{};
    std::atomic<u64> scheduler_wakeups{};
    std::atomic<u64> scheduler_batches{};
    std::atomic<u64> scheduler_ports{};
    std::atomic<u64> scheduler_mixed_ports{};
    std::atomic<u64> scheduler_nanoseconds{};
    std::atomic<u64> scheduler_max_batch_ports{};
    std::atomic<u64> zero_copy_handoffs{};
    std::atomic<u64> guest_copy_calls{};
    std::atomic<u64> guest_copy_bytes{};
    std::atomic<u64> guest_copy_wait_nanoseconds{};
    std::atomic<u64> continuity_failures{};
    std::atomic<u64> timestamp_order_failures{};
    std::atomic<u64> scheduler_underruns{};
    std::atomic<u64> channel_guard_passes{};
    std::atomic<u64> channel_guard_failures{};
    std::atomic<u64> scalar_fallback_buffers{};
    std::atomic<u64> regression_guard_runs{};
    std::atomic<u64> regression_guard_passes{};
    std::atomic<u64> regression_guard_failures{};
};

AudioOptimizationCounters counters{};
std::mutex regression_guard_mutex;

template <typename T>
void AtomicMax(std::atomic<T>& target, const T value) noexcept {
    T current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

u64 ElapsedNanoseconds(const std::chrono::steady_clock::time_point start) noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

bool ApproximatelyEqual(const float actual, const float expected) noexcept {
    if (std::isnan(expected)) {
        return std::isnan(actual);
    }
    if (!std::isfinite(actual) || !std::isfinite(expected)) {
        return actual == expected;
    }
    const float tolerance = 1.0e-5f * std::max(1.0f, std::abs(expected));
    return std::abs(actual - expected) <= tolerance;
}

float SanitizedFloat(const float value) noexcept {
    return std::isfinite(value) ? value : 0.0f;
}

std::array<u8, 8> SemanticSourceIndices(const AudioBufferFormat format) noexcept {
    // Non-Std: FL FR FC LFE SL SR BL BR
    // Std:     FL FR FC LFE BL BR SL SR
    if (format.channels != 8 || !format.standard_8ch) {
        return {0, 1, 2, 3, 4, 5, 6, 7};
    }
    return {0, 1, 2, 3, 6, 7, 4, 5};
}

float ReadSample(const void* source, const size_t index,
                 const AudioSampleType sample_type) noexcept {
    if (sample_type == AudioSampleType::Float32) {
        return SanitizedFloat(static_cast<const float*>(source)[index]);
    }
    return static_cast<float>(static_cast<const s16*>(source)[index]) * S16Scale;
}

bool ConvertScalar(const void* source, const u32 frames, const AudioBufferFormat format,
                   const u32 output_channels, const float gain, std::span<float> output) noexcept {
    if (source == nullptr || frames == 0 ||
        (format.channels != 1 && format.channels != 2 && format.channels != 8) ||
        (output_channels != 1 && output_channels != 2 && output_channels != 8) ||
        output.size() < static_cast<size_t>(frames) * output_channels) {
        return false;
    }

    const auto semantic = SemanticSourceIndices(format);
    bool any_nonzero = false;
    for (u32 frame = 0; frame < frames; ++frame) {
        const size_t source_base = static_cast<size_t>(frame) * format.channels;
        const size_t output_base = static_cast<size_t>(frame) * output_channels;
        if (format.channels == 8 && output_channels == 2) {
            const float center =
                ReadSample(source, source_base + semantic[2], format.sample_type) * CenterMix;
            const float left =
                (ReadSample(source, source_base + semantic[0], format.sample_type) + center +
                 (ReadSample(source, source_base + semantic[4], format.sample_type) +
                  ReadSample(source, source_base + semantic[6], format.sample_type)) *
                     SurroundMix) *
                gain;
            const float right =
                (ReadSample(source, source_base + semantic[1], format.sample_type) + center +
                 (ReadSample(source, source_base + semantic[5], format.sample_type) +
                  ReadSample(source, source_base + semantic[7], format.sample_type)) *
                     SurroundMix) *
                gain;
            output[output_base] = left;
            output[output_base + 1] = right;
            any_nonzero |= left != 0.0f || right != 0.0f;
            continue;
        }

        if (format.channels == 8 && output_channels == 8) {
            for (u32 channel = 0; channel < 8; ++channel) {
                const float value =
                    ReadSample(source, source_base + semantic[channel], format.sample_type) * gain;
                output[output_base + channel] = value;
                any_nonzero |= value != 0.0f;
            }
            continue;
        }

        const u32 copy_channels = std::min<u32>(format.channels, output_channels);
        for (u32 channel = 0; channel < copy_channels; ++channel) {
            const float value =
                ReadSample(source, source_base + channel, format.sample_type) * gain;
            output[output_base + channel] = value;
            any_nonzero |= value != 0.0f;
        }
        for (u32 channel = copy_channels; channel < output_channels; ++channel) {
            const float value = format.channels == 1
                                    ? ReadSample(source, source_base, format.sample_type) * gain
                                    : 0.0f;
            output[output_base + channel] = value;
            any_nonzero |= value != 0.0f;
        }
    }
    return any_nonzero;
}

#if defined(LSX4_AUDIO_HAS_NEON)
float32x4_t SanitizeFloat32x4(const float32x4_t value) noexcept {
    const float32x4_t absolute = vabsq_f32(value);
    const uint32x4_t finite = vcleq_f32(absolute, vdupq_n_f32(std::numeric_limits<float>::max()));
    return vbslq_f32(finite, value, vdupq_n_f32(0.0f));
}

bool ConvertSimd(const void* source, const u32 frames, const AudioBufferFormat format,
                 const u32 output_channels, const float gain, std::span<float> output) noexcept {
    if (source == nullptr || frames == 0 ||
        output.size() < static_cast<size_t>(frames) * output_channels) {
        return false;
    }

    bool any_nonzero = false;
    if (format.channels == 8 && output_channels == 2) {
        const auto semantic = SemanticSourceIndices(format);
        std::array<float, 8> left_weights{};
        std::array<float, 8> right_weights{};
        left_weights[semantic[0]] = 1.0f;
        left_weights[semantic[2]] = CenterMix;
        left_weights[semantic[4]] = SurroundMix;
        left_weights[semantic[6]] = SurroundMix;
        right_weights[semantic[1]] = 1.0f;
        right_weights[semantic[2]] = CenterMix;
        right_weights[semantic[5]] = SurroundMix;
        right_weights[semantic[7]] = SurroundMix;
        const float32x4_t left_low = vld1q_f32(left_weights.data());
        const float32x4_t left_high = vld1q_f32(left_weights.data() + 4);
        const float32x4_t right_low = vld1q_f32(right_weights.data());
        const float32x4_t right_high = vld1q_f32(right_weights.data() + 4);
        const float scale = gain * (format.sample_type == AudioSampleType::S16 ? S16Scale : 1.0f);
        for (u32 frame = 0; frame < frames; ++frame) {
            float32x4_t source_low;
            float32x4_t source_high;
            if (format.sample_type == AudioSampleType::Float32) {
                const float* input = static_cast<const float*>(source) + frame * 8;
                source_low = SanitizeFloat32x4(vld1q_f32(input));
                source_high = SanitizeFloat32x4(vld1q_f32(input + 4));
            } else {
                const int16x8_t input = vld1q_s16(static_cast<const s16*>(source) + frame * 8);
                source_low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(input)));
                source_high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(input)));
            }
            const float left = (vaddvq_f32(vmulq_f32(source_low, left_low)) +
                                vaddvq_f32(vmulq_f32(source_high, left_high))) *
                               scale;
            const float right = (vaddvq_f32(vmulq_f32(source_low, right_low)) +
                                 vaddvq_f32(vmulq_f32(source_high, right_high))) *
                                scale;
            output[frame * 2] = left;
            output[frame * 2 + 1] = right;
            any_nonzero |= left != 0.0f || right != 0.0f;
        }
        return any_nonzero;
    }

    if (format.channels == output_channels && !format.standard_8ch) {
        const size_t samples = static_cast<size_t>(frames) * output_channels;
        const float32x4_t vector_gain =
            vdupq_n_f32(gain * (format.sample_type == AudioSampleType::S16 ? S16Scale : 1.0f));
        size_t index = 0;
        if (format.sample_type == AudioSampleType::Float32) {
            const float* input = static_cast<const float*>(source);
            for (; index + 4 <= samples; index += 4) {
                const float32x4_t converted =
                    vmulq_f32(SanitizeFloat32x4(vld1q_f32(input + index)), vector_gain);
                vst1q_f32(output.data() + index, converted);
                any_nonzero |= vminvq_u32(vceqq_f32(converted, vdupq_n_f32(0.0f))) !=
                               std::numeric_limits<u32>::max();
            }
        } else {
            const s16* input = static_cast<const s16*>(source);
            for (; index + 8 <= samples; index += 8) {
                const int16x8_t packed = vld1q_s16(input + index);
                const float32x4_t low =
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(packed))), vector_gain);
                const float32x4_t high =
                    vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(packed))), vector_gain);
                vst1q_f32(output.data() + index, low);
                vst1q_f32(output.data() + index + 4, high);
                any_nonzero |= vminvq_u32(vandq_u32(vceqq_f32(low, vdupq_n_f32(0.0f)),
                                                    vceqq_f32(high, vdupq_n_f32(0.0f)))) !=
                               std::numeric_limits<u32>::max();
            }
        }
        for (; index < samples; ++index) {
            const float value = ReadSample(source, index, format.sample_type) * gain;
            output[index] = value;
            any_nonzero |= value != 0.0f;
        }
        return any_nonzero;
    }

    // Layout transforms are uncommon and remain on the scalar reference path.
    return ConvertScalar(source, frames, format, output_channels, gain, output);
}
#endif

bool MixScalar(void* destination, const void* source, const u32 frames,
               const AudioBufferFormat format, const float source_gain) noexcept {
    if (destination == nullptr || source == nullptr || frames == 0 ||
        (format.channels != 1 && format.channels != 2 && format.channels != 8)) {
        return false;
    }
    const size_t samples = static_cast<size_t>(frames) * format.channels;
    bool any_nonzero = false;
    if (format.sample_type == AudioSampleType::Float32) {
        auto* dst = static_cast<float*>(destination);
        const auto* src = static_cast<const float*>(source);
        for (size_t index = 0; index < samples; ++index) {
            const float mixed = std::clamp(
                SanitizedFloat(dst[index]) + SanitizedFloat(src[index]) * source_gain, -1.0f, 1.0f);
            dst[index] = mixed;
            any_nonzero |= mixed != 0.0f;
        }
    } else {
        auto* dst = static_cast<s16*>(destination);
        const auto* src = static_cast<const s16*>(source);
        for (size_t index = 0; index < samples; ++index) {
            const s32 mixed =
                static_cast<s32>(dst[index]) +
                static_cast<s32>(std::lrint(static_cast<float>(src[index]) * source_gain));
            dst[index] = static_cast<s16>(
                std::clamp(mixed, static_cast<s32>(std::numeric_limits<s16>::min()),
                           static_cast<s32>(std::numeric_limits<s16>::max())));
            any_nonzero |= dst[index] != 0;
        }
    }
    return any_nonzero;
}

#if defined(LSX4_AUDIO_HAS_NEON)
bool MixSimd(void* destination, const void* source, const u32 frames,
             const AudioBufferFormat format, const float source_gain) noexcept {
    if (destination == nullptr || source == nullptr || frames == 0) {
        return false;
    }
    const size_t samples = static_cast<size_t>(frames) * format.channels;
    bool any_nonzero = false;
    size_t index = 0;
    if (format.sample_type == AudioSampleType::Float32) {
        auto* dst = static_cast<float*>(destination);
        const auto* src = static_cast<const float*>(source);
        const float32x4_t gain = vdupq_n_f32(source_gain);
        const float32x4_t low = vdupq_n_f32(-1.0f);
        const float32x4_t high = vdupq_n_f32(1.0f);
        for (; index + 4 <= samples; index += 4) {
            const float32x4_t mixed = vmaxq_f32(
                low, vminq_f32(high, vaddq_f32(SanitizeFloat32x4(vld1q_f32(dst + index)),
                                               vmulq_f32(SanitizeFloat32x4(vld1q_f32(src + index)),
                                                         gain))));
            vst1q_f32(dst + index, mixed);
            any_nonzero |=
                vminvq_u32(vceqq_f32(mixed, vdupq_n_f32(0.0f))) != std::numeric_limits<u32>::max();
        }
    } else if (source_gain == 1.0f) {
        auto* dst = static_cast<s16*>(destination);
        const auto* src = static_cast<const s16*>(source);
        for (; index + 8 <= samples; index += 8) {
            const int16x8_t mixed = vqaddq_s16(vld1q_s16(dst + index), vld1q_s16(src + index));
            vst1q_s16(dst + index, mixed);
            any_nonzero |=
                vminvq_u16(vceqq_s16(mixed, vdupq_n_s16(0))) != std::numeric_limits<u16>::max();
        }
    }
    if (index == 0 && format.sample_type == AudioSampleType::S16 && source_gain != 1.0f) {
        return MixScalar(destination, source, frames, format, source_gain);
    }
    if (index < samples) {
        AudioBufferFormat tail_format = format;
        tail_format.channels = 1;
        const size_t sample_size =
            format.sample_type == AudioSampleType::Float32 ? sizeof(float) : sizeof(s16);
        any_nonzero |= MixScalar(static_cast<u8*>(destination) + index * sample_size,
                                 static_cast<const u8*>(source) + index * sample_size,
                                 static_cast<u32>(samples - index), tail_format, source_gain);
    }
    return any_nonzero;
}
#endif

bool FloatBuffersEqual(const std::span<const float> lhs,
                       const std::span<const float> rhs) noexcept {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t index = 0; index < lhs.size(); ++index) {
        if (!ApproximatelyEqual(lhs[index], rhs[index])) {
            return false;
        }
    }
    return true;
}

AudioStreamGuardResult EvaluateStreamConsume(AudioStreamGuard& guard, const u64 sequence,
                                             const u64 submit_timestamp_ns,
                                             const u64 consume_timestamp_ns,
                                             const u64 expected_period_ns) noexcept {
    AudioStreamGuardResult result{};
    result.continuity_ok =
        sequence != 0 &&
        (guard.last_consumed_sequence == 0 || sequence == guard.last_consumed_sequence + 1) &&
        sequence <= guard.last_submitted_sequence;
    result.timestamp_order_ok = consume_timestamp_ns >= submit_timestamp_ns &&
                                (guard.last_consume_timestamp_ns == 0 ||
                                 consume_timestamp_ns >= guard.last_consume_timestamp_ns);
    const u64 latency = consume_timestamp_ns >= submit_timestamp_ns
                            ? consume_timestamp_ns - submit_timestamp_ns
                            : 0;
    result.underrun = expected_period_ns != 0 && latency > expected_period_ns * 2;
    guard.last_consumed_sequence = sequence;
    guard.last_consume_timestamp_ns = consume_timestamp_ns;
    return result;
}

bool RunRegressionFixtures() {
    // Known heavy profile: CUSA03173 / Bloodborne uses the general large
    // surround path. The title is not consulted at runtime; this fixture
    // protects S16 8ch, both Orbis channel orders, 1024-frame downmix and mix.
    constexpr u32 BloodborneFrames = 1024;
    std::vector<s16> surround(static_cast<size_t>(BloodborneFrames) * 8);
    for (size_t index = 0; index < surround.size(); ++index) {
        surround[index] = static_cast<s16>(static_cast<s32>((index * 97u) % 60001u) - 30000);
    }
    std::vector<float> scalar_stereo(static_cast<size_t>(BloodborneFrames) * 2);
    std::vector<float> optimized_stereo(scalar_stereo.size());
    for (const bool standard_layout : {false, true}) {
        const AudioBufferFormat format{
            .sample_type = AudioSampleType::S16,
            .channels = 8,
            .standard_8ch = standard_layout,
        };
        ConvertScalar(surround.data(), BloodborneFrames, format, 2, 0.75f, scalar_stereo);
#if defined(LSX4_AUDIO_HAS_NEON)
        ConvertSimd(surround.data(), BloodborneFrames, format, 2, 0.75f, optimized_stereo);
#else
        ConvertScalar(surround.data(), BloodborneFrames, format, 2, 0.75f, optimized_stereo);
#endif
        if (!FloatBuffersEqual(scalar_stereo, optimized_stereo)) {
            return false;
        }
    }
    std::array<s16, 8> standard_impulse{};
    standard_impulse[6] = std::numeric_limits<s16>::max();
    std::array<float, 8> canonical_impulse{};
    ConvertScalar(standard_impulse.data(), 1,
                  {
                      .sample_type = AudioSampleType::S16,
                      .channels = 8,
                      .standard_8ch = true,
                  },
                  8, 1.0f, canonical_impulse);
    if (!ApproximatelyEqual(canonical_impulse[4], 32767.0f * S16Scale) ||
        canonical_impulse[6] != 0.0f) {
        return false;
    }

    auto surround_scalar = surround;
    auto surround_optimized = surround;
    std::vector<s16> surround_aux(surround.size(), 10000);
    const AudioBufferFormat surround_format{
        .sample_type = AudioSampleType::S16,
        .channels = 8,
    };
    MixScalar(surround_scalar.data(), surround_aux.data(), BloodborneFrames, surround_format, 1.0f);
#if defined(LSX4_AUDIO_HAS_NEON)
    MixSimd(surround_optimized.data(), surround_aux.data(), BloodborneFrames, surround_format,
            1.0f);
#else
    MixScalar(surround_optimized.data(), surround_aux.data(), BloodborneFrames, surround_format,
              1.0f);
#endif
    if (surround_scalar != surround_optimized) {
        return false;
    }

    // Known light profile: Dreaming Sarah exercises the same generic stereo
    // contract with both S16 and float buffers. No title-specific branch is
    // present in production conversion or scheduling.
    constexpr u32 DreamingSarahFrames = 1024;
    std::vector<float> stereo_float(static_cast<size_t>(DreamingSarahFrames) * 2);
    for (size_t index = 0; index < stereo_float.size(); ++index) {
        stereo_float[index] = std::sin(static_cast<float>(index) * 0.03125f) * 0.8f;
    }
    stereo_float[13] = std::numeric_limits<float>::quiet_NaN();
    scalar_stereo.resize(stereo_float.size());
    optimized_stereo.resize(stereo_float.size());
    const AudioBufferFormat float_stereo{
        .sample_type = AudioSampleType::Float32,
        .channels = 2,
    };
    ConvertScalar(stereo_float.data(), DreamingSarahFrames, float_stereo, 2, 0.625f, scalar_stereo);
#if defined(LSX4_AUDIO_HAS_NEON)
    ConvertSimd(stereo_float.data(), DreamingSarahFrames, float_stereo, 2, 0.625f,
                optimized_stereo);
#else
    ConvertScalar(stereo_float.data(), DreamingSarahFrames, float_stereo, 2, 0.625f,
                  optimized_stereo);
#endif
    if (!FloatBuffersEqual(scalar_stereo, optimized_stereo)) {
        return false;
    }

    // Continuity, timestamp/order and underrun invariants shared by all titles.
    AudioStreamGuard stream_guard{};
    ObserveAudioStreamSubmit(stream_guard, 1, 100);
    const auto first = ObserveAudioStreamConsume(stream_guard, 1, 100, 150, 100);
    ObserveAudioStreamSubmit(stream_guard, 2, 200);
    const auto second = ObserveAudioStreamConsume(stream_guard, 2, 200, 250, 100);
    AudioStreamGuard broken_guard{.last_submitted_sequence = 1};
    const auto broken = EvaluateStreamConsume(broken_guard, 2, 100, 90, 100);
    AudioStreamGuard late_guard{.last_submitted_sequence = 1};
    const auto late = EvaluateStreamConsume(late_guard, 1, 100, 401, 100);
    return first.continuity_ok && first.timestamp_order_ok && !first.underrun &&
           second.continuity_ok && second.timestamp_order_ok && !second.underrun &&
           !broken.continuity_ok && !broken.timestamp_order_ok && !broken.underrun &&
           late.continuity_ok && late.timestamp_order_ok && late.underrun;
}

} // namespace

void SetManagedAudioSimdEnabled(const bool enabled) {
    if (!enabled) {
        counters.enabled.store(false, std::memory_order_release);
        counters.global_scalar_fallback.store(false, std::memory_order_release);
        counters.generation.fetch_add(1, std::memory_order_acq_rel);
        return;
    }

    // Arm the optimized path only after its pure-CPU fixtures pass. Concurrent
    // streams remain on scalar code while the short guard is running.
    counters.enabled.store(false, std::memory_order_release);
    counters.global_scalar_fallback.store(false, std::memory_order_release);
    counters.generation.fetch_add(1, std::memory_order_acq_rel);
    const bool guards_passed = RunAudioOptimizationRegressionGuards();
    counters.enabled.store(guards_passed && HasAudioSimd, std::memory_order_release);
    counters.generation.fetch_add(1, std::memory_order_acq_rel);
}

bool IsManagedAudioSimdEnabled() {
    return counters.enabled.load(std::memory_order_acquire);
}

AudioOptimizationStats GetAudioOptimizationStats() {
    return {
        .conversion_calls = counters.conversion_calls.load(std::memory_order_relaxed),
        .converted_frames = counters.converted_frames.load(std::memory_order_relaxed),
        .simd_frames = counters.simd_frames.load(std::memory_order_relaxed),
        .scalar_frames = counters.scalar_frames.load(std::memory_order_relaxed),
        .conversion_nanoseconds = counters.conversion_nanoseconds.load(std::memory_order_relaxed),
        .write_nanoseconds = counters.write_nanoseconds.load(std::memory_order_relaxed),
        .partial_writes = counters.partial_writes.load(std::memory_order_relaxed),
        .write_errors = counters.write_errors.load(std::memory_order_relaxed),
        .xruns = counters.xruns.load(std::memory_order_relaxed),
        .scheduler_wakeups = counters.scheduler_wakeups.load(std::memory_order_relaxed),
        .scheduler_batches = counters.scheduler_batches.load(std::memory_order_relaxed),
        .scheduler_ports = counters.scheduler_ports.load(std::memory_order_relaxed),
        .scheduler_mixed_ports = counters.scheduler_mixed_ports.load(std::memory_order_relaxed),
        .scheduler_nanoseconds = counters.scheduler_nanoseconds.load(std::memory_order_relaxed),
        .scheduler_max_batch_ports =
            counters.scheduler_max_batch_ports.load(std::memory_order_relaxed),
        .zero_copy_handoffs = counters.zero_copy_handoffs.load(std::memory_order_relaxed),
        .guest_copy_calls = counters.guest_copy_calls.load(std::memory_order_relaxed),
        .guest_copy_bytes = counters.guest_copy_bytes.load(std::memory_order_relaxed),
        .guest_copy_wait_nanoseconds =
            counters.guest_copy_wait_nanoseconds.load(std::memory_order_relaxed),
        .continuity_failures = counters.continuity_failures.load(std::memory_order_relaxed),
        .timestamp_order_failures =
            counters.timestamp_order_failures.load(std::memory_order_relaxed),
        .scheduler_underruns = counters.scheduler_underruns.load(std::memory_order_relaxed),
        .channel_guard_passes = counters.channel_guard_passes.load(std::memory_order_relaxed),
        .channel_guard_failures = counters.channel_guard_failures.load(std::memory_order_relaxed),
        .scalar_fallback_buffers = counters.scalar_fallback_buffers.load(std::memory_order_relaxed),
        .regression_guard_runs = counters.regression_guard_runs.load(std::memory_order_relaxed),
        .regression_guard_passes = counters.regression_guard_passes.load(std::memory_order_relaxed),
        .regression_guard_failures =
            counters.regression_guard_failures.load(std::memory_order_relaxed),
        .toggle_generation = counters.generation.load(std::memory_order_relaxed),
        .managed_enabled = counters.enabled.load(std::memory_order_relaxed) ? 1u : 0u,
        .global_scalar_fallback =
            counters.global_scalar_fallback.load(std::memory_order_relaxed) ? 1u : 0u,
    };
}

void ResetAudioOptimizationStats() {
    counters.conversion_calls.store(0, std::memory_order_relaxed);
    counters.converted_frames.store(0, std::memory_order_relaxed);
    counters.simd_frames.store(0, std::memory_order_relaxed);
    counters.scalar_frames.store(0, std::memory_order_relaxed);
    counters.conversion_nanoseconds.store(0, std::memory_order_relaxed);
    counters.write_nanoseconds.store(0, std::memory_order_relaxed);
    counters.partial_writes.store(0, std::memory_order_relaxed);
    counters.write_errors.store(0, std::memory_order_relaxed);
    counters.xruns.store(0, std::memory_order_relaxed);
    counters.scheduler_wakeups.store(0, std::memory_order_relaxed);
    counters.scheduler_batches.store(0, std::memory_order_relaxed);
    counters.scheduler_ports.store(0, std::memory_order_relaxed);
    counters.scheduler_mixed_ports.store(0, std::memory_order_relaxed);
    counters.scheduler_nanoseconds.store(0, std::memory_order_relaxed);
    counters.scheduler_max_batch_ports.store(0, std::memory_order_relaxed);
    counters.zero_copy_handoffs.store(0, std::memory_order_relaxed);
    counters.guest_copy_calls.store(0, std::memory_order_relaxed);
    counters.guest_copy_bytes.store(0, std::memory_order_relaxed);
    counters.guest_copy_wait_nanoseconds.store(0, std::memory_order_relaxed);
    counters.continuity_failures.store(0, std::memory_order_relaxed);
    counters.timestamp_order_failures.store(0, std::memory_order_relaxed);
    counters.scheduler_underruns.store(0, std::memory_order_relaxed);
    counters.channel_guard_passes.store(0, std::memory_order_relaxed);
    counters.channel_guard_failures.store(0, std::memory_order_relaxed);
    counters.scalar_fallback_buffers.store(0, std::memory_order_relaxed);
    counters.regression_guard_runs.store(0, std::memory_order_relaxed);
    counters.regression_guard_passes.store(0, std::memory_order_relaxed);
    counters.regression_guard_failures.store(0, std::memory_order_relaxed);
}

AudioConvertResult ConvertAudioBufferToFloat(const void* source, const u32 frames,
                                             const AudioBufferFormat format,
                                             const u32 output_channels, const float gain,
                                             const std::span<float> output,
                                             AudioConversionGuard& guard) {
    const auto started = std::chrono::steady_clock::now();
    const u64 generation = counters.generation.load(std::memory_order_acquire);
    if (guard.generation != generation) {
        guard = {
            .generation = generation,
            .validated = false,
            .force_scalar = false,
        };
    }
    const bool managed = counters.enabled.load(std::memory_order_acquire);
    const bool global_fallback = counters.global_scalar_fallback.load(std::memory_order_acquire);
    const bool simd_shape = (format.channels == 8 && output_channels == 2) ||
                            (format.channels == output_channels && !format.standard_8ch);
    const bool try_simd =
        HasAudioSimd && simd_shape && managed && !global_fallback && !guard.force_scalar;

    bool any_nonzero = false;
    bool used_simd = false;
#if defined(LSX4_AUDIO_HAS_NEON)
    if (try_simd) {
        any_nonzero = ConvertSimd(source, frames, format, output_channels, gain, output);
        used_simd = true;
    } else
#else
    (void)try_simd;
#endif
    {
        any_nonzero = ConvertScalar(source, frames, format, output_channels, gain, output);
    }

    bool guard_passed = false;
    bool scalar_fallback = false;
    if (used_simd && !guard.validated && any_nonzero) {
        std::vector<float> reference(output.size());
        ConvertScalar(source, frames, format, output_channels, gain, reference);
        if (FloatBuffersEqual(output, reference)) {
            guard.validated = true;
            guard_passed = true;
            counters.channel_guard_passes.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::ranges::copy(reference, output.begin());
            guard.force_scalar = true;
            used_simd = false;
            scalar_fallback = true;
            counters.channel_guard_failures.fetch_add(1, std::memory_order_relaxed);
            counters.scalar_fallback_buffers.fetch_add(1, std::memory_order_relaxed);
        }
    }

    counters.conversion_calls.fetch_add(1, std::memory_order_relaxed);
    counters.converted_frames.fetch_add(frames, std::memory_order_relaxed);
    (used_simd ? counters.simd_frames : counters.scalar_frames)
        .fetch_add(frames, std::memory_order_relaxed);
    counters.conversion_nanoseconds.fetch_add(ElapsedNanoseconds(started),
                                              std::memory_order_relaxed);
    return {
        .any_nonzero = any_nonzero,
        .used_simd = used_simd,
        .differential_guard_passed = guard_passed,
        .scalar_fallback = scalar_fallback,
    };
}

AudioConvertResult MixAudioBufferInPlace(void* destination, const void* source, const u32 frames,
                                         const AudioBufferFormat format, const float source_gain,
                                         AudioConversionGuard& guard) {
    const auto started = std::chrono::steady_clock::now();
    const u64 generation = counters.generation.load(std::memory_order_acquire);
    if (guard.generation != generation) {
        guard = {
            .generation = generation,
            .validated = false,
            .force_scalar = false,
        };
    }
    const bool simd_shape = format.sample_type == AudioSampleType::Float32 || source_gain == 1.0f;
    const bool try_simd = HasAudioSimd && counters.enabled.load(std::memory_order_acquire) &&
                          !counters.global_scalar_fallback.load(std::memory_order_acquire) &&
                          !guard.force_scalar && simd_shape;

    const size_t sample_size =
        format.sample_type == AudioSampleType::Float32 ? sizeof(float) : sizeof(s16);
    const size_t bytes = static_cast<size_t>(frames) * format.channels * sample_size;
    std::vector<u8> original;
    if (try_simd && !guard.validated) {
        original.resize(bytes);
        std::memcpy(original.data(), destination, bytes);
    }

    bool any_nonzero = false;
    bool used_simd = false;
#if defined(LSX4_AUDIO_HAS_NEON)
    if (try_simd) {
        any_nonzero = MixSimd(destination, source, frames, format, source_gain);
        used_simd = true;
    } else
#else
    (void)try_simd;
#endif
    {
        any_nonzero = MixScalar(destination, source, frames, format, source_gain);
    }

    bool guard_passed = false;
    bool scalar_fallback = false;
    if (used_simd && !guard.validated) {
        std::vector<u8> reference = original;
        MixScalar(reference.data(), source, frames, format, source_gain);
        const bool equal =
            format.sample_type == AudioSampleType::S16
                ? std::memcmp(reference.data(), destination, bytes) == 0
                : FloatBuffersEqual(
                      std::span<const float>(reinterpret_cast<const float*>(reference.data()),
                                             bytes / sizeof(float)),
                      std::span<const float>(static_cast<const float*>(destination),
                                             bytes / sizeof(float)));
        if (equal) {
            guard.validated = true;
            guard_passed = true;
            counters.channel_guard_passes.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::memcpy(destination, reference.data(), bytes);
            guard.force_scalar = true;
            used_simd = false;
            scalar_fallback = true;
            counters.channel_guard_failures.fetch_add(1, std::memory_order_relaxed);
            counters.scalar_fallback_buffers.fetch_add(1, std::memory_order_relaxed);
        }
    }

    counters.conversion_calls.fetch_add(1, std::memory_order_relaxed);
    counters.converted_frames.fetch_add(frames, std::memory_order_relaxed);
    (used_simd ? counters.simd_frames : counters.scalar_frames)
        .fetch_add(frames, std::memory_order_relaxed);
    counters.conversion_nanoseconds.fetch_add(ElapsedNanoseconds(started),
                                              std::memory_order_relaxed);
    return {
        .any_nonzero = any_nonzero,
        .used_simd = used_simd,
        .differential_guard_passed = guard_passed,
        .scalar_fallback = scalar_fallback,
    };
}

u64 AudioMonotonicNanoseconds() noexcept {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

void ObserveAudioStreamSubmit(AudioStreamGuard& guard, const u64 sequence,
                              const u64 timestamp_ns) noexcept {
    if ((guard.last_submitted_sequence != 0 && sequence != guard.last_submitted_sequence + 1) ||
        (guard.last_submit_timestamp_ns != 0 && timestamp_ns < guard.last_submit_timestamp_ns)) {
        counters.continuity_failures.fetch_add(1, std::memory_order_relaxed);
    }
    guard.last_submitted_sequence = sequence;
    guard.last_submit_timestamp_ns = timestamp_ns;
}

AudioStreamGuardResult ObserveAudioStreamConsume(AudioStreamGuard& guard, const u64 sequence,
                                                 const u64 submit_timestamp_ns,
                                                 const u64 consume_timestamp_ns,
                                                 const u64 expected_period_ns) noexcept {
    const AudioStreamGuardResult result = EvaluateStreamConsume(
        guard, sequence, submit_timestamp_ns, consume_timestamp_ns, expected_period_ns);
    if (!result.continuity_ok) {
        counters.continuity_failures.fetch_add(1, std::memory_order_relaxed);
    }
    if (!result.timestamp_order_ok) {
        counters.timestamp_order_failures.fetch_add(1, std::memory_order_relaxed);
    }
    if (result.underrun) {
        counters.scheduler_underruns.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
}

void RecordAudioGuestCopy(const u64 bytes, const u64 wait_nanoseconds) noexcept {
    counters.guest_copy_calls.fetch_add(1, std::memory_order_relaxed);
    counters.guest_copy_bytes.fetch_add(bytes, std::memory_order_relaxed);
    counters.guest_copy_wait_nanoseconds.fetch_add(wait_nanoseconds, std::memory_order_relaxed);
}

void RecordAudioSchedulerWakeup() noexcept {
    counters.scheduler_wakeups.fetch_add(1, std::memory_order_relaxed);
}

void RecordAudioSchedulerBatch(const u32 ports, const u32 mixed_ports,
                               const u64 scheduling_nanoseconds) noexcept {
    counters.scheduler_batches.fetch_add(1, std::memory_order_relaxed);
    counters.scheduler_ports.fetch_add(ports, std::memory_order_relaxed);
    counters.scheduler_mixed_ports.fetch_add(mixed_ports, std::memory_order_relaxed);
    counters.scheduler_nanoseconds.fetch_add(scheduling_nanoseconds, std::memory_order_relaxed);
    AtomicMax(counters.scheduler_max_batch_ports, static_cast<u64>(ports));
}

void RecordAudioZeroCopyHandoff() noexcept {
    counters.zero_copy_handoffs.fetch_add(1, std::memory_order_relaxed);
}

void RecordAudioBackendWrite(const u64 write_nanoseconds, const s64 written_frames,
                             const u32 requested_frames, const u32 xrun_delta) noexcept {
    counters.write_nanoseconds.fetch_add(write_nanoseconds, std::memory_order_relaxed);
    if (written_frames < 0) {
        counters.write_errors.fetch_add(1, std::memory_order_relaxed);
    } else if (static_cast<u64>(written_frames) != requested_frames) {
        counters.partial_writes.fetch_add(1, std::memory_order_relaxed);
    }
    counters.xruns.fetch_add(xrun_delta, std::memory_order_relaxed);
}

bool RunAudioOptimizationRegressionGuards() noexcept {
    std::scoped_lock lock{regression_guard_mutex};
    counters.regression_guard_runs.fetch_add(1, std::memory_order_relaxed);
    bool passed = false;
    try {
        passed = RunRegressionFixtures();
    } catch (...) {
        passed = false;
    }
    if (passed) {
        counters.regression_guard_passes.fetch_add(1, std::memory_order_relaxed);
    } else {
        counters.regression_guard_failures.fetch_add(1, std::memory_order_relaxed);
        counters.global_scalar_fallback.store(true, std::memory_order_release);
        counters.generation.fetch_add(1, std::memory_order_acq_rel);
    }
#ifdef __ANDROID__
    __android_log_print(
        passed ? ANDROID_LOG_INFO : ANDROID_LOG_ERROR, "LSX4Native",
        "[EXECUTOR_AUDIO_RUNTIME_GUARD] result=%s profiles=CUSA03173-surround,DreamingSarah-stereo "
        "contracts=continuity,underrun,timestamp-order,channel-map fallback=%s",
        passed ? "pass" : "fail", passed ? "none" : "scalar");
#endif
    return passed;
}

} // namespace Libraries::AudioOut
