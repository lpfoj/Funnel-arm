// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

#ifdef __ANDROID__
#include <aaudio/AAudio.h>
#include <android/log.h>
#endif

#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_hints.h>

#include "common/config.h"
#include "common/logging/log.h"
#include "core/libraries/audio/audio_mix.h"
#include "core/libraries/audio/audioout.h"
#include "core/libraries/audio/audioout_backend.h"
#include "core/libraries/kernel/threads.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAS_SSE2
#endif

#define SDL_INVALID_AUDIODEVICEID 0

namespace Libraries::AudioOut {

namespace {

u64 ElapsedNanoseconds(const std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

} // namespace

constexpr float VOLUME_0DB = 32768.0f;
constexpr float INV_VOLUME_0DB = 1.0f / VOLUME_0DB;
constexpr float VOLUME_EPSILON = 0.001f;
constexpr u64 VOLUME_CHECK_INTERVAL_US = 50000;
constexpr u64 MIN_SLEEP_THRESHOLD_US = 10;
constexpr u64 TIMING_RESYNC_THRESHOLD_US = 100000;

constexpr u32 QUEUE_MULTIPLIER = 4;
constexpr size_t AUDIO_BUFFER_ALIGNMENT = 32;

#ifdef __ANDROID__
static void AndroidAudioLog(const char* format, ...) {
    va_list args;
    va_start(args, format);
    __android_log_vprint(ANDROID_LOG_INFO, "LSX4Native", format, args);
    va_end(args);
}
#endif

enum ChannelPos : u8 {
    FL = 0,
    FR = 1,
    FC = 2,
    LF = 3,
    SL = 4,
    SR = 5,
    BL = 6,
    BR = 7,
    STD_SL = 6,
    STD_SR = 7,
    STD_BL = 4,
    STD_BR = 5
};

class SDLPortBackend : public PortBackend {
public:
    explicit SDLPortBackend(const PortOut& port)
        : frame_size(port.format_info.FrameSize()), guest_buffer_size(port.BufferSize()),
          buffer_frames(port.buffer_frames), sample_rate(port.sample_rate),
          num_channels(port.format_info.num_channels), is_float(port.format_info.is_float),
          is_std(port.format_info.is_std), channel_layout(port.format_info.channel_layout) {

        if (!Initialize(port.type)) {
            LOG_ERROR(Lib_AudioOut, "Failed to initialize SDL audio backend");
        }
    }

    ~SDLPortBackend() override {
        Cleanup();
    }

    void Output(void* ptr) override {
        if (!stream || !internal_buffer) [[unlikely]] {
            return;
        }

        if (ptr == nullptr) [[unlikely]] {
            return;
        }

        const AudioBufferFormat format{
            .sample_type = is_float ? AudioSampleType::Float32 : AudioSampleType::S16,
            .channels = static_cast<u8>(num_channels),
            // SDL's input channel map consumes the guest's physical Std order
            // directly. Semantic normalization is only needed before software
            // downmixing/mixing, not on this native eight-channel route.
            .standard_8ch = false,
        };
        (void)ConvertAudioBufferToFloat(
            ptr, buffer_frames, format, num_channels, 1.0f,
            std::span<float>{static_cast<float*>(internal_buffer),
                             static_cast<size_t>(buffer_frames) * num_channels},
            conversion_guard);
        SubmitFloat(static_cast<const float*>(internal_buffer));
    }

    void OutputFloat(const float* ptr, const u32 frames, const u32 channels) override {
        if (!stream || !ptr) [[unlikely]] {
            return;
        }
        if (frames != buffer_frames || channels != num_channels) [[unlikely]] {
            LOG_ERROR(Lib_AudioOut,
                      "Invalid normalized SDL audio buffer: frames={}/{} channels={}/{}", frames,
                      buffer_frames, channels, num_channels);
            return;
        }
        if (is_std && channels == 8) {
            // The shared mixer exposes canonical FL/FR/FC/LFE/SL/SR/BL/BR.
            // Restore the Orbis Std physical order expected by the SDL input
            // map: FL/FR/FC/LFE/BL/BR/SL/SR.
            auto* physical = static_cast<float*>(internal_buffer);
            for (u32 frame = 0; frame < frames; ++frame) {
                const size_t offset = static_cast<size_t>(frame) * 8;
                physical[offset + 0] = ptr[offset + 0];
                physical[offset + 1] = ptr[offset + 1];
                physical[offset + 2] = ptr[offset + 2];
                physical[offset + 3] = ptr[offset + 3];
                physical[offset + 4] = ptr[offset + 6];
                physical[offset + 5] = ptr[offset + 7];
                physical[offset + 6] = ptr[offset + 4];
                physical[offset + 7] = ptr[offset + 5];
            }
            SubmitFloat(physical);
            return;
        }
        SubmitFloat(ptr);
    }

    void SetVolume(const std::array<int, 8>& ch_volumes) override {
        if (!stream) [[unlikely]] {
            return;
        }

        float max_channel_gain = 0.0f;
        const u32 channels_to_check = std::min(num_channels, 8u);

        for (u32 i = 0; i < channels_to_check; i++) {
            const float channel_gain = static_cast<float>(ch_volumes[i]) * INV_VOLUME_0DB;
            max_channel_gain = std::max(max_channel_gain, channel_gain);
        }

        const float slider_gain = Config::getVolumeSlider() * 0.01f;
        const float total_gain = max_channel_gain * slider_gain;

        const float current = current_gain.load(std::memory_order_acquire);
        if (std::abs(total_gain - current) < VOLUME_EPSILON) {
            return;
        }

        if (SDL_SetAudioStreamGain(stream, total_gain)) {
            current_gain.store(total_gain, std::memory_order_release);
            LOG_DEBUG(Lib_AudioOut,
                      "Set combined audio gain to {:.3f} (channel: {:.3f}, slider: {:.3f})",
                      total_gain, max_channel_gain, slider_gain);
        } else {
            LOG_ERROR(Lib_AudioOut, "Failed to set audio stream gain: {}", SDL_GetError());
        }
    }

    u64 GetLastOutputTime() const {
        return last_output_time.load(std::memory_order_acquire);
    }

private:
    void SubmitFloat(const float* ptr) {
        UpdateVolumeIfChanged();
        const u64 current_time = Kernel::sceKernelGetProcessTime();
        HandleTiming(current_time);

        if ((output_count++ & 0xF) == 0) {
            ManageAudioQueue();
        }

        const auto write_started = std::chrono::steady_clock::now();
        const bool write_ok = SDL_PutAudioStreamData(stream, ptr, internal_buffer_size);
        RecordAudioBackendWrite(ElapsedNanoseconds(write_started),
                                write_ok ? static_cast<s64>(buffer_frames) : -1, buffer_frames, 0);
        if (!write_ok) [[unlikely]] {
            LOG_ERROR(Lib_AudioOut, "Failed to output to SDL audio stream: {}", SDL_GetError());
        }

        last_output_time.store(current_time, std::memory_order_release);
    }
    bool Initialize(OrbisAudioOutPort type) {
        period_us = (1000000ULL * buffer_frames + sample_rate / 2) / sample_rate;

        internal_buffer_size = buffer_frames * sizeof(float) * num_channels;

#ifdef _WIN32
        internal_buffer = _aligned_malloc(internal_buffer_size, AUDIO_BUFFER_ALIGNMENT);
#else
        if (posix_memalign(&internal_buffer, AUDIO_BUFFER_ALIGNMENT, internal_buffer_size) != 0) {
            internal_buffer = nullptr;
        }
#endif

        if (!internal_buffer) {
            LOG_ERROR(Lib_AudioOut, "Failed to allocate aligned audio buffer of size {}",
                      internal_buffer_size);
            return false;
        }

        current_gain.store(Config::getVolumeSlider() * 0.01f, std::memory_order_relaxed);

        if (!SelectConverter()) {
            FreeAlignedBuffer();
            return false;
        }

        if (!OpenDevice(type)) {
            FreeAlignedBuffer();
            return false;
        }

        CalculateQueueThreshold();
        return true;
    }

    void Cleanup() {
        if (stream) {
            SDL_DestroyAudioStream(stream);
            stream = nullptr;
        }
        FreeAlignedBuffer();
    }

    void FreeAlignedBuffer() {
        if (internal_buffer) {
#ifdef _WIN32
            _aligned_free(internal_buffer);
#else
            free(internal_buffer);
#endif
            internal_buffer = nullptr;
        }
    }

    void UpdateVolumeIfChanged() {
        const u64 current_time = Kernel::sceKernelGetProcessTime();

        if (current_time - last_volume_check_time < VOLUME_CHECK_INTERVAL_US) {
            return;
        }

        last_volume_check_time = current_time;

        const float config_volume = Config::getVolumeSlider() * 0.01f;
        const float stored_gain = current_gain.load(std::memory_order_acquire);

        if (std::abs(config_volume - stored_gain) > VOLUME_EPSILON) {
            if (SDL_SetAudioStreamGain(stream, config_volume)) {
                current_gain.store(config_volume, std::memory_order_release);
                LOG_DEBUG(Lib_AudioOut, "Updated audio gain to {:.3f}", config_volume);
            } else {
                LOG_ERROR(Lib_AudioOut, "Failed to set audio stream gain: {}", SDL_GetError());
            }
        }
    }

    void HandleTiming(u64 current_time) {
        if (next_output_time == 0) [[unlikely]] {
            next_output_time = current_time + period_us;
            return;
        }

        const s64 time_diff = static_cast<s64>(current_time - next_output_time);

        if (time_diff > static_cast<s64>(TIMING_RESYNC_THRESHOLD_US)) [[unlikely]] {
            next_output_time = current_time + period_us;
        } else if (time_diff < 0) {
            const u64 time_to_wait = static_cast<u64>(-time_diff);
            next_output_time += period_us;

            if (time_to_wait > MIN_SLEEP_THRESHOLD_US) {
                const u64 sleep_duration = time_to_wait - MIN_SLEEP_THRESHOLD_US;
                std::this_thread::sleep_for(std::chrono::microseconds(sleep_duration));
            }
        } else {
            next_output_time += period_us;
        }
    }

    void ManageAudioQueue() {
        const auto queued = SDL_GetAudioStreamQueued(stream);

        if (queued >= queue_threshold) [[unlikely]] {
            LOG_DEBUG(Lib_AudioOut, "Clearing backed up audio queue ({} >= {})", queued,
                      queue_threshold);
            SDL_ClearAudioStream(stream);
            CalculateQueueThreshold();
        }
    }

    bool OpenDevice(OrbisAudioOutPort type) {
        const SDL_AudioSpec fmt = {
            .format = SDL_AUDIO_F32LE,
            .channels = static_cast<u8>(num_channels),
            .freq = static_cast<int>(sample_rate),
        };

        const std::string device_name = GetDeviceName(type);
        const SDL_AudioDeviceID dev_id = SelectAudioDevice(device_name, type);

        if (dev_id == SDL_INVALID_AUDIODEVICEID) {
            return false;
        }

        stream = SDL_OpenAudioDeviceStream(dev_id, &fmt, nullptr, nullptr);
        if (!stream) {
            LOG_ERROR(Lib_AudioOut, "Failed to create SDL audio stream: {}", SDL_GetError());
            return false;
        }

        if (!ConfigureChannelMap()) {
            SDL_DestroyAudioStream(stream);
            stream = nullptr;
            return false;
        }

        const float initial_gain = current_gain.load(std::memory_order_relaxed);
        if (!SDL_SetAudioStreamGain(stream, initial_gain)) {
            LOG_WARNING(Lib_AudioOut, "Failed to set initial audio gain: {}", SDL_GetError());
        }

        if (!SDL_ResumeAudioStreamDevice(stream)) {
            LOG_ERROR(Lib_AudioOut, "Failed to resume audio stream: {}", SDL_GetError());
            SDL_DestroyAudioStream(stream);
            stream = nullptr;
            return false;
        }

        LOG_INFO(Lib_AudioOut, "Opened audio device: {} ({} Hz, {} ch, gain: {:.3f})", device_name,
                 sample_rate, num_channels, initial_gain);
        return true;
    }

    SDL_AudioDeviceID SelectAudioDevice(const std::string& device_name, OrbisAudioOutPort type) {
        if (device_name == "None") {
            LOG_INFO(Lib_AudioOut, "Audio device disabled for port type {}",
                     static_cast<int>(type));
            return SDL_INVALID_AUDIODEVICEID;
        }

        if (device_name.empty() || device_name == "Default Device") {
            return SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
        }

        int num_devices = 0;
        SDL_AudioDeviceID* dev_array = SDL_GetAudioPlaybackDevices(&num_devices);

        if (!dev_array) {
            LOG_WARNING(Lib_AudioOut, "No audio devices found, using default");
            return SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
        }

        SDL_AudioDeviceID selected_device = SDL_INVALID_AUDIODEVICEID;

        for (int i = 0; i < num_devices; i++) {
            const char* dev_name = SDL_GetAudioDeviceName(dev_array[i]);
            if (dev_name && device_name == dev_name) {
                selected_device = dev_array[i];
                break;
            }
        }

        SDL_free(dev_array);

        if (selected_device == SDL_INVALID_AUDIODEVICEID) {
            LOG_WARNING(Lib_AudioOut, "Audio device '{}' not found, using default", device_name);
            return SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
        }

        return selected_device;
    }

    bool ConfigureChannelMap() {
        if (num_channels == 0) {
            return true;
        }

        std::vector<int> channel_map(num_channels);

        if (is_std && num_channels == 8) {
            channel_map = {FL, FR, FC, LF, STD_SL, STD_SR, STD_BL, STD_BR};
        } else {
            std::copy_n(channel_layout.begin(), num_channels, channel_map.begin());
        }

        if (!SDL_SetAudioStreamInputChannelMap(stream, channel_map.data(), num_channels)) {
            LOG_ERROR(Lib_AudioOut, "Failed to set channel map: {}", SDL_GetError());
            return false;
        }

        return true;
    }

    std::string GetDeviceName(OrbisAudioOutPort type) const {
        switch (type) {
        case OrbisAudioOutPort::Main:
        case OrbisAudioOutPort::Bgm:
            return Config::getMainOutputDevice();
        case OrbisAudioOutPort::PadSpk:
            return Config::getPadSpkOutputDevice();
        default:
            return Config::getMainOutputDevice();
        }
    }

    bool SelectConverter() {
        if (is_float) {
            switch (num_channels) {
            case 1:
                convert = &ConvertF32Mono;
                break;
            case 2:
                convert = &ConvertF32Stereo;
                break;
            case 8:
                convert = is_std ? &ConvertF32Std8CH : &ConvertF32_8CH;
                break;
            default:
                LOG_ERROR(Lib_AudioOut, "Unsupported float channel count: {}", num_channels);
                return false;
            }
        } else {
            switch (num_channels) {
            case 1:
                convert = &ConvertS16Mono;
                break;
            case 2:
#if defined(HAS_SSE2)
                convert = &ConvertS16StereoSIMD;
#else
                convert = &ConvertS16Stereo;
#endif
                break;
            case 8:
#if defined(HAS_SSE2)
                convert = &ConvertS16_8CH_SIMD;
#else
                convert = &ConvertS16_8CH;
#endif
                break;
            default:
                LOG_ERROR(Lib_AudioOut, "Unsupported S16 channel count: {}", num_channels);
                return false;
            }
        }

        return true;
    }

    void CalculateQueueThreshold() {
        if (!stream) {
            return;
        }

        SDL_AudioSpec discard;
        int sdl_buffer_frames = 0;

        if (!SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(stream), &discard,
                                      &sdl_buffer_frames)) {
            LOG_WARNING(Lib_AudioOut, "Failed to get SDL buffer size: {}", SDL_GetError());
        }

        const u32 sdl_buffer_size = sdl_buffer_frames * sizeof(float) * num_channels;
        queue_threshold = std::max(guest_buffer_size, sdl_buffer_size) * QUEUE_MULTIPLIER;

        LOG_DEBUG(Lib_AudioOut, "Audio queue threshold: {} bytes (SDL buffer: {} frames)",
                  queue_threshold, sdl_buffer_frames);
    }

    using ConverterFunc = void (*)(const void* src, void* dst, u32 frames, const float* volumes);

    static void ConvertS16Mono(const void* src, void* dst, u32 frames, const float*) {
        const s16* s = static_cast<const s16*>(src);
        float* d = static_cast<float*>(dst);

        for (u32 i = 0; i < frames; i++) {
            d[i] = s[i] * INV_VOLUME_0DB;
        }
    }

    static void ConvertS16Stereo(const void* src, void* dst, u32 frames, const float*) {
        const s16* s = static_cast<const s16*>(src);
        float* d = static_cast<float*>(dst);

        const u32 num_samples = frames << 1;
        for (u32 i = 0; i < num_samples; i++) {
            d[i] = s[i] * INV_VOLUME_0DB;
        }
    }

#ifdef HAS_SSE2
    static void ConvertS16StereoSIMD(const void* src, void* dst, u32 frames, const float*) {
        const s16* s = static_cast<const s16*>(src);
        float* d = static_cast<float*>(dst);

        const __m128 scale = _mm_set1_ps(INV_VOLUME_0DB);
        const u32 num_samples = frames << 1;
        u32 i = 0;

        for (; i + 8 <= num_samples; i += 8) {
            __m128i s16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&s[i]));

            __m128i s32_lo = _mm_cvtepi16_epi32(s16_vals);
            __m128i s32_hi = _mm_cvtepi16_epi32(_mm_srli_si128(s16_vals, 8));

            __m128 f_lo = _mm_mul_ps(_mm_cvtepi32_ps(s32_lo), scale);
            __m128 f_hi = _mm_mul_ps(_mm_cvtepi32_ps(s32_hi), scale);

            _mm_storeu_ps(&d[i], f_lo);
            _mm_storeu_ps(&d[i + 4], f_hi);
        }

        for (; i < num_samples; i++) {
            d[i] = s[i] * INV_VOLUME_0DB;
        }
    }
#endif

    static void ConvertS16_8CH(const void* src, void* dst, u32 frames, const float*) {
        const s16* s = static_cast<const s16*>(src);
        float* d = static_cast<float*>(dst);

        const u32 num_samples = frames << 3;
        for (u32 i = 0; i < num_samples; i++) {
            d[i] = s[i] * INV_VOLUME_0DB;
        }
    }

#ifdef HAS_SSE2
    static void ConvertS16_8CH_SIMD(const void* src, void* dst, u32 frames, const float*) {
        const s16* s = static_cast<const s16*>(src);
        float* d = static_cast<float*>(dst);

        const __m128 scale = _mm_set1_ps(INV_VOLUME_0DB);
        const u32 num_samples = frames << 3;
        u32 i = 0;

        for (; i + 8 <= num_samples; i += 8) {
            __m128i s16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&s[i]));
            __m128i s32_lo = _mm_cvtepi16_epi32(s16_vals);
            __m128i s32_hi = _mm_cvtepi16_epi32(_mm_srli_si128(s16_vals, 8));
            __m128 f_lo = _mm_mul_ps(_mm_cvtepi32_ps(s32_lo), scale);
            __m128 f_hi = _mm_mul_ps(_mm_cvtepi32_ps(s32_hi), scale);
            _mm_storeu_ps(&d[i], f_lo);
            _mm_storeu_ps(&d[i + 4], f_hi);
        }

        for (; i < num_samples; i++) {
            d[i] = s[i] * INV_VOLUME_0DB;
        }
    }
#endif

    static void ConvertF32Mono(const void* src, void* dst, u32 frames, const float*) {
        std::memcpy(dst, src, frames * sizeof(float));
    }

    static void ConvertF32Stereo(const void* src, void* dst, u32 frames, const float*) {
        std::memcpy(dst, src, frames * 2 * sizeof(float));
    }

    static void ConvertF32_8CH(const void* src, void* dst, u32 frames, const float*) {
        std::memcpy(dst, src, frames * 8 * sizeof(float));
    }

    static void ConvertF32Std8CH(const void* src, void* dst, u32 frames, const float*) {
        const float* s = static_cast<const float*>(src);
        float* d = static_cast<float*>(dst);

        for (u32 i = 0; i < frames; i++) {
            const u32 offset = i << 3;

            d[offset + FL] = s[offset + FL];
            d[offset + FR] = s[offset + FR];
            d[offset + FC] = s[offset + FC];
            d[offset + LF] = s[offset + LF];
            d[offset + SL] = s[offset + STD_SL];
            d[offset + SR] = s[offset + STD_SR];
            d[offset + BL] = s[offset + STD_BL];
            d[offset + BR] = s[offset + STD_BR];
        }
    }

    const u32 frame_size;
    const u32 guest_buffer_size;
    const u32 buffer_frames;
    const u32 sample_rate;
    const u32 num_channels;
    const bool is_float;
    const bool is_std;
    const std::array<int, 8> channel_layout;

    alignas(64) u64 period_us{0};
    alignas(64) std::atomic<u64> last_output_time{0};
    u64 next_output_time{0};
    u64 last_volume_check_time{0};
    u32 output_count{0};

    u32 internal_buffer_size{0};
    void* internal_buffer{nullptr};

    ConverterFunc convert{nullptr};
    AudioConversionGuard conversion_guard{};

    alignas(64) std::atomic<float> current_gain{1.0f};

    SDL_AudioStream* stream{nullptr};
    u32 queue_threshold{0};
};

#ifdef __ANDROID__
class AndroidAAudioPortBackend : public PortBackend {
public:
    explicit AndroidAAudioPortBackend(const PortOut& port)
        : buffer_frames(port.buffer_frames), sample_rate(port.sample_rate),
          guest_channels(port.format_info.num_channels),
          output_channels(port.format_info.num_channels == 1 ? 1u : 2u),
          is_float(port.format_info.is_float), is_std(port.format_info.is_std) {
        output_buffer.resize(static_cast<size_t>(buffer_frames) * output_channels);
        if (!OpenStream()) {
            Cleanup();
        }
    }

    ~AndroidAAudioPortBackend() override {
        Cleanup();
    }

    bool IsReady() const {
        return stream != nullptr;
    }

    void Output(void* ptr) override {
        if (!stream || !ptr) {
            return;
        }

        if (AudioOutputDisabled()) {
            return;
        }

        SubmitOutputBuffer(ConvertToOutput(ptr));
    }

    void OutputFloat(const float* ptr, const u32 frames, const u32 channels) override {
        if (!stream || !ptr || AudioOutputDisabled()) {
            return;
        }
        if (frames != buffer_frames || (channels != 1 && channels != 2 && channels != 8)) {
            AndroidAudioLog(
                "[EXECUTOR_AUDIO_AAUDIO_OUTPUT] invalid_normalized frames=%u/%u channels=%u",
                frames, buffer_frames, channels);
            return;
        }
        const AudioBufferFormat format{
            .sample_type = AudioSampleType::Float32,
            .channels = static_cast<u8>(channels),
            .standard_8ch = false,
        };
        const auto converted = ConvertAudioBufferToFloat(
            ptr, frames, format, output_channels, gain.load(std::memory_order_acquire),
            output_buffer, normalized_conversion_guard);
        SubmitOutputBuffer(converted.any_nonzero);
    }

    void SetVolume(const std::array<int, 8>& ch_volumes) override {
        float max_channel_gain = 0.0f;
        const u32 channels_to_check = std::min(guest_channels, 8u);
        for (u32 i = 0; i < channels_to_check; i++) {
            max_channel_gain =
                std::max(max_channel_gain, static_cast<float>(ch_volumes[i]) * INV_VOLUME_0DB);
        }
        gain.store(max_channel_gain * Config::getVolumeSlider() * 0.01f, std::memory_order_release);
    }

private:
    static bool AudioOutputDisabled() {
        static const bool disabled = [] {
            const char* value = std::getenv("EXECUTOR_DISABLE_AUDIO_OUTPUT");
            return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
        }();
        return disabled;
    }

    void SubmitOutputBuffer(const bool any_nonzero) {
        if (!any_nonzero && !output_buffer.empty()) {
            silence_dither_phase = !silence_dither_phase;
            output_buffer[0] = silence_dither_phase ? 1.0e-8f : -1.0e-8f;
        }
        const auto write_start = std::chrono::steady_clock::now();
        const aaudio_result_t written =
            AAudioStream_write(stream, output_buffer.data(), buffer_frames, 20'000'000);
        const u64 write_nanoseconds = ElapsedNanoseconds(write_start);

        if (written < 0) {
            RecordAudioBackendWrite(write_nanoseconds, written, buffer_frames, 0);
            if (TakeOutputLogBudget()) {
                AndroidAudioLog("[EXECUTOR_AUDIO_AAUDIO_OUTPUT] error=%s",
                                AAudio_convertResultToText(written));
            }
            return;
        }
        u32 xrun_delta{};
        if ((++output_call_count & 255u) == 0u) {
            const int32_t xruns = AAudioStream_getXRunCount(stream);
            if (xruns > last_xrun_count) {
                xrun_delta = static_cast<u32>(xruns - last_xrun_count);
                last_xrun_count = xruns;
            }
        }
        RecordAudioBackendWrite(write_nanoseconds, written, buffer_frames, xrun_delta);

        if (TakeOutputLogBudget()) {
            float peak = 0.0f;
            const size_t sample_count =
                std::min<size_t>(output_buffer.size(), static_cast<size_t>(512));
            for (size_t i = 0; i < sample_count; i++) {
                peak = std::max(peak, std::abs(output_buffer[i]));
            }
            AndroidAudioLog("[EXECUTOR_AUDIO_AAUDIO_OUTPUT] frames=%d/%u outCh=%u guestCh=%u "
                            "float=%d peak=%.4f",
                            static_cast<int>(written), buffer_frames, output_channels,
                            guest_channels, is_float ? 1 : 0, peak);
        }
    }
    bool OpenStream() {
        if (AudioOutputDisabled()) {
            AndroidAudioLog(
                "[EXECUTOR_AUDIO_AAUDIO_OPEN] disabled=1 (MIUI AudioTrack crash escape)");
            return false;
        }
        AAudioStreamBuilder* builder = nullptr;
        aaudio_result_t result = AAudio_createStreamBuilder(&builder);
        if (result != AAUDIO_OK || !builder) {
            AndroidAudioLog("[EXECUTOR_AUDIO_AAUDIO_OPEN] create_builder error=%s",
                            AAudio_convertResultToText(result));
            return false;
        }

        AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_OUTPUT);
        AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
        AAudioStreamBuilder_setSharingMode(builder, AAUDIO_SHARING_MODE_SHARED);
        AAudioStreamBuilder_setSampleRate(builder, static_cast<int32_t>(sample_rate));
        AAudioStreamBuilder_setChannelCount(builder, static_cast<int32_t>(output_channels));
        AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
        AAudioStreamBuilder_setBufferCapacityInFrames(
            builder, static_cast<int32_t>(std::max<u32>(buffer_frames * 4, 512)));

        result = AAudioStreamBuilder_openStream(builder, &stream);
        AAudioStreamBuilder_delete(builder);
        if (result != AAUDIO_OK || !stream) {
            AndroidAudioLog("[EXECUTOR_AUDIO_AAUDIO_OPEN] open error=%s",
                            AAudio_convertResultToText(result));
            return false;
        }

        result = AAudioStream_requestStart(stream);
        if (result != AAUDIO_OK) {
            AndroidAudioLog("[EXECUTOR_AUDIO_AAUDIO_OPEN] start error=%s",
                            AAudio_convertResultToText(result));
            return false;
        }

        AndroidAudioLog("[EXECUTOR_AUDIO_AAUDIO_OPEN] ok sampleRate=%u guestCh=%u outCh=%u "
                        "frames=%u",
                        sample_rate, guest_channels, output_channels, buffer_frames);
        return true;
    }

    void Cleanup() {
        if (stream) {
            AAudioStream_requestStop(stream);
            AAudioStream_close(stream);
            stream = nullptr;
        }
    }

    bool TakeOutputLogBudget() {
        int budget = output_log_budget.load(std::memory_order_relaxed);
        while (budget > 0 && !output_log_budget.compare_exchange_weak(budget, budget - 1,
                                                                      std::memory_order_relaxed,
                                                                      std::memory_order_relaxed)) {
        }
        return budget > 0;
    }

#if 0
    // Retained temporarily as a readable scalar oracle for older captures.
    // Production conversion below is shared with SDL and the isolated PS5
    // runtime through audio_mix.cpp.
    bool ValidateSimdOutput(const void* ptr, const float current_gain) const {
        const auto matches = [](const float actual, const float expected) {
            if (std::isnan(expected)) {
                return std::isnan(actual);
            }
            if (!std::isfinite(actual) || !std::isfinite(expected)) {
                return actual == expected;
            }
            const float tolerance =
                1.0e-5f * std::max(1.0f, std::abs(expected));
            return std::abs(actual - expected) <= tolerance;
        };
        if (guest_channels == 8) {
            constexpr float CenterMix = 0.70710678f;
            constexpr float SurroundMix = 0.70710678f;
            for (u32 frame = 0; frame < buffer_frames; ++frame) {
                const u32 src_base = frame * 8;
                const u32 dst_base = frame * 2;
                float left{};
                float right{};
                if (is_float) {
                    const float* src = static_cast<const float*>(ptr);
                    const float center = src[src_base + FC] * CenterMix;
                    left = (src[src_base + FL] + center +
                            (src[src_base + SL] + src[src_base + BL]) *
                                SurroundMix) *
                           current_gain;
                    right = (src[src_base + FR] + center +
                             (src[src_base + SR] + src[src_base + BR]) *
                                 SurroundMix) *
                            current_gain;
                } else {
                    const s16* src = static_cast<const s16*>(ptr);
                    const float center =
                        static_cast<float>(src[src_base + FC]) * CenterMix;
                    left = (static_cast<float>(src[src_base + FL]) + center +
                            static_cast<float>(
                                src[src_base + SL] + src[src_base + BL]) *
                                SurroundMix) *
                           INV_VOLUME_0DB * current_gain;
                    right = (static_cast<float>(src[src_base + FR]) + center +
                             static_cast<float>(
                                 src[src_base + SR] + src[src_base + BR]) *
                                 SurroundMix) *
                            INV_VOLUME_0DB * current_gain;
                }
                if (!matches(output_buffer[dst_base], left) ||
                    !matches(output_buffer[dst_base + 1], right)) {
                    return false;
                }
            }
            return true;
        }

        for (u32 frame = 0; frame < buffer_frames; ++frame) {
            const u32 src_base = frame * guest_channels;
            const u32 dst_base = frame * output_channels;
            float left{};
            float right{};
            if (is_float) {
                const float* src = static_cast<const float*>(ptr);
                left = src[src_base] * current_gain;
                if (output_channels == 2) {
                    right =
                        src[src_base + (guest_channels > 1 ? 1 : 0)] * current_gain;
                }
            } else {
                const s16* src = static_cast<const s16*>(ptr);
                left = static_cast<float>(src[src_base]) *
                       INV_VOLUME_0DB * current_gain;
                if (output_channels == 2) {
                    right = static_cast<float>(
                                src[src_base + (guest_channels > 1 ? 1 : 0)]) *
                            INV_VOLUME_0DB * current_gain;
                }
            }
            if (!matches(output_buffer[dst_base], left) ||
                (output_channels == 2 &&
                 !matches(output_buffer[dst_base + 1], right))) {
                return false;
            }
        }
        return true;
    }

    bool ConvertToOutput(const void* ptr) {
        const auto conversion_start = std::chrono::steady_clock::now();
        const float current_gain = gain.load(std::memory_order_acquire);
        bool any_nonzero = false;
        bool used_simd = false;

#if defined(LSX4_HAS_ARM64_NEON)
        if (IsManagedAudioSimdEnabled() && !simd_guard_failed) {
            used_simd = true;
            if (guest_channels == 8) {
                constexpr float CenterMix = 0.70710678f;
                constexpr float SurroundMix = 0.70710678f;
                const float32x4_t left_low_weights =
                    {1.0f, 0.0f, CenterMix, 0.0f};
                const float32x4_t right_low_weights =
                    {0.0f, 1.0f, CenterMix, 0.0f};
                const float32x4_t left_high_weights =
                    {SurroundMix, 0.0f, SurroundMix, 0.0f};
                const float32x4_t right_high_weights =
                    {0.0f, SurroundMix, 0.0f, SurroundMix};
                const float scale =
                    current_gain * (is_float ? 1.0f : INV_VOLUME_0DB);
                if (is_float) {
                    const float* src = static_cast<const float*>(ptr);
                    for (u32 frame = 0; frame < buffer_frames; ++frame) {
                        const float32x4_t low = vld1q_f32(src + frame * 8);
                        const float32x4_t high =
                            vld1q_f32(src + frame * 8 + 4);
                        const float left =
                            (vaddvq_f32(vmulq_f32(low, left_low_weights)) +
                             vaddvq_f32(vmulq_f32(high, left_high_weights))) *
                            scale;
                        const float right =
                            (vaddvq_f32(vmulq_f32(low, right_low_weights)) +
                             vaddvq_f32(vmulq_f32(high, right_high_weights))) *
                            scale;
                        output_buffer[frame * 2] = left;
                        output_buffer[frame * 2 + 1] = right;
                        any_nonzero |= left != 0.0f || right != 0.0f;
                    }
                } else {
                    const s16* src = static_cast<const s16*>(ptr);
                    for (u32 frame = 0; frame < buffer_frames; ++frame) {
                        const int16x8_t samples =
                            vld1q_s16(src + frame * 8);
                        const float32x4_t low = vcvtq_f32_s32(
                            vmovl_s16(vget_low_s16(samples)));
                        const float32x4_t high = vcvtq_f32_s32(
                            vmovl_s16(vget_high_s16(samples)));
                        const float left =
                            (vaddvq_f32(vmulq_f32(low, left_low_weights)) +
                             vaddvq_f32(vmulq_f32(high, left_high_weights))) *
                            scale;
                        const float right =
                            (vaddvq_f32(vmulq_f32(low, right_low_weights)) +
                             vaddvq_f32(vmulq_f32(high, right_high_weights))) *
                            scale;
                        output_buffer[frame * 2] = left;
                        output_buffer[frame * 2 + 1] = right;
                        any_nonzero |= left != 0.0f || right != 0.0f;
                    }
                }
            } else if (is_float) {
                const float* src = static_cast<const float*>(ptr);
                const size_t samples =
                    static_cast<size_t>(buffer_frames) * output_channels;
                const float32x4_t scale = vdupq_n_f32(current_gain);
                size_t index = 0;
                for (; index + 4 <= samples; index += 4) {
                    const float32x4_t output =
                        vmulq_f32(vld1q_f32(src + index), scale);
                    vst1q_f32(output_buffer.data() + index, output);
                    const uint32x4_t zero = vceqq_f32(
                        output, vdupq_n_f32(0.0f));
                    any_nonzero |=
                        vminvq_u32(zero) != std::numeric_limits<u32>::max();
                }
                for (; index < samples; ++index) {
                    const float output = src[index] * current_gain;
                    output_buffer[index] = output;
                    any_nonzero |= output != 0.0f;
                }
            } else {
                const s16* src = static_cast<const s16*>(ptr);
                const size_t samples =
                    static_cast<size_t>(buffer_frames) * output_channels;
                const float32x4_t scale =
                    vdupq_n_f32(INV_VOLUME_0DB * current_gain);
                size_t index = 0;
                for (; index + 8 <= samples; index += 8) {
                    const int16x8_t input = vld1q_s16(src + index);
                    const float32x4_t low = vmulq_f32(
                        vcvtq_f32_s32(vmovl_s16(vget_low_s16(input))),
                        scale);
                    const float32x4_t high = vmulq_f32(
                        vcvtq_f32_s32(vmovl_s16(vget_high_s16(input))),
                        scale);
                    vst1q_f32(output_buffer.data() + index, low);
                    vst1q_f32(output_buffer.data() + index + 4, high);
                    const uint32x4_t low_zero =
                        vceqq_f32(low, vdupq_n_f32(0.0f));
                    const uint32x4_t high_zero =
                        vceqq_f32(high, vdupq_n_f32(0.0f));
                    any_nonzero |=
                        vminvq_u32(vandq_u32(low_zero, high_zero)) !=
                        std::numeric_limits<u32>::max();
                }
                for (; index < samples; ++index) {
                    const float output =
                        static_cast<float>(src[index]) *
                        INV_VOLUME_0DB * current_gain;
                    output_buffer[index] = output;
                    any_nonzero |= output != 0.0f;
                }
            }
        }
#endif

        if (used_simd) {
            if (!simd_guard_completed && any_nonzero) {
                if (!ValidateSimdOutput(ptr, current_gain)) {
                    simd_guard_failed = true;
                    used_simd = false;
                    any_nonzero = false;
                    AndroidAudioLog(
                        "[EXECUTOR_AUDIO_SIMD_GUARD] result=fallback guestCh=%u "
                        "outCh=%u float=%u frames=%u",
                        guest_channels, output_channels, is_float ? 1u : 0u,
                        buffer_frames);
                } else {
                    simd_guard_completed = true;
                    AndroidAudioLog(
                        "[EXECUTOR_AUDIO_SIMD_GUARD] result=pass guestCh=%u "
                        "outCh=%u float=%u frames=%u",
                        guest_channels, output_channels, is_float ? 1u : 0u,
                        buffer_frames);
                }
            }
        }

        if (used_simd) {
            g_audio_conversion_calls.fetch_add(1, std::memory_order_relaxed);
            g_audio_converted_frames.fetch_add(
                buffer_frames, std::memory_order_relaxed);
            g_audio_simd_frames.fetch_add(
                buffer_frames, std::memory_order_relaxed);
            g_audio_conversion_nanoseconds.fetch_add(
                ElapsedNanoseconds(conversion_start),
                std::memory_order_relaxed);
            return any_nonzero;
        }

        if (guest_channels == 8) {
            constexpr float CenterMix = 0.70710678f;
            constexpr float SurroundMix = 0.70710678f;
            if (is_float) {
                const float* src = static_cast<const float*>(ptr);
                for (u32 frame = 0; frame < buffer_frames; frame++) {
                    const u32 src_base = frame * 8;
                    const u32 dst_base = frame * 2;
                    const float center = src[src_base + FC] * CenterMix;
                    output_buffer[dst_base] =
                        (src[src_base + FL] + center +
                         (src[src_base + SL] + src[src_base + BL]) * SurroundMix) *
                        current_gain;
                    output_buffer[dst_base + 1] =
                        (src[src_base + FR] + center +
                         (src[src_base + SR] + src[src_base + BR]) * SurroundMix) *
                        current_gain;
                    any_nonzero |= output_buffer[dst_base] != 0.0f ||
                                   output_buffer[dst_base + 1] != 0.0f;
                }
            } else {
                const s16* src = static_cast<const s16*>(ptr);
                for (u32 frame = 0; frame < buffer_frames; frame++) {
                    const u32 src_base = frame * 8;
                    const u32 dst_base = frame * 2;
                    const float center =
                        static_cast<float>(src[src_base + FC]) * CenterMix;
                    output_buffer[dst_base] =
                        (static_cast<float>(src[src_base + FL]) + center +
                         static_cast<float>(
                             src[src_base + SL] + src[src_base + BL]) *
                             SurroundMix) *
                        INV_VOLUME_0DB * current_gain;
                    output_buffer[dst_base + 1] =
                        (static_cast<float>(src[src_base + FR]) + center +
                         static_cast<float>(
                             src[src_base + SR] + src[src_base + BR]) *
                             SurroundMix) *
                        INV_VOLUME_0DB * current_gain;
                    any_nonzero |= output_buffer[dst_base] != 0.0f ||
                                   output_buffer[dst_base + 1] != 0.0f;
                }
            }
        } else if (is_float) {
            const float* src = static_cast<const float*>(ptr);
            for (u32 frame = 0; frame < buffer_frames; frame++) {
                const u32 src_base = frame * guest_channels;
                const u32 dst_base = frame * output_channels;
                output_buffer[dst_base] = src[src_base] * current_gain;
                if (output_channels == 2) {
                    output_buffer[dst_base + 1] =
                        src[src_base + (guest_channels > 1 ? 1 : 0)] * current_gain;
                }
                any_nonzero |= output_buffer[dst_base] != 0.0f ||
                               (output_channels == 2 &&
                                output_buffer[dst_base + 1] != 0.0f);
            }
        } else {
            const s16* src = static_cast<const s16*>(ptr);
            for (u32 frame = 0; frame < buffer_frames; frame++) {
                const u32 src_base = frame * guest_channels;
                const u32 dst_base = frame * output_channels;
                output_buffer[dst_base] =
                    src[src_base] * INV_VOLUME_0DB * current_gain;
                if (output_channels == 2) {
                    output_buffer[dst_base + 1] =
                        src[src_base + (guest_channels > 1 ? 1 : 0)] *
                        INV_VOLUME_0DB * current_gain;
                }
                any_nonzero |= output_buffer[dst_base] != 0.0f ||
                               (output_channels == 2 &&
                                output_buffer[dst_base + 1] != 0.0f);
            }
        }
        g_audio_conversion_calls.fetch_add(1, std::memory_order_relaxed);
        g_audio_converted_frames.fetch_add(
            buffer_frames, std::memory_order_relaxed);
        g_audio_scalar_frames.fetch_add(
            buffer_frames, std::memory_order_relaxed);
        g_audio_conversion_nanoseconds.fetch_add(
            ElapsedNanoseconds(conversion_start), std::memory_order_relaxed);
        return any_nonzero;
    }
#endif

    bool ConvertToOutput(const void* ptr) {
        const AudioBufferFormat format{
            .sample_type = is_float ? AudioSampleType::Float32 : AudioSampleType::S16,
            .channels = static_cast<u8>(guest_channels),
            .standard_8ch = is_std,
        };
        const auto result = ConvertAudioBufferToFloat(ptr, buffer_frames, format, output_channels,
                                                      gain.load(std::memory_order_acquire),
                                                      output_buffer, conversion_guard);
        return result.any_nonzero;
    }

    const u32 buffer_frames;
    const u32 sample_rate;
    const u32 guest_channels;
    const u32 output_channels;
    const bool is_float;
    const bool is_std;
    std::vector<float> output_buffer;
    bool silence_dither_phase = false;
    AAudioStream* stream{nullptr};
    std::atomic<float> gain{Config::getVolumeSlider() * 0.01f};
    std::atomic_int output_log_budget{4};
    u32 output_call_count{};
    int32_t last_xrun_count{};
    AudioConversionGuard conversion_guard{};
    AudioConversionGuard normalized_conversion_guard{};
};
#endif

std::unique_ptr<PortBackend> SDLAudioOut::Open(PortOut& port) {
#ifdef __ANDROID__
    static const bool audio_out_disabled = [] {
        const char* v = std::getenv("EXECUTOR_DISABLE_AUDIO_OUTPUT");
        return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
    }();
    auto android_backend = std::make_unique<AndroidAAudioPortBackend>(port);
    if (android_backend->IsReady() || audio_out_disabled) {
        return android_backend;
    }
    AndroidAudioLog("[EXECUTOR_AUDIO_AAUDIO_OPEN] falling back to SDL");
#endif
    return std::make_unique<SDLPortBackend>(port);
}

} // namespace Libraries::AudioOut
