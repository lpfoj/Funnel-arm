// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/ngs2/ngs2_runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/guest_callback.h"
#include "core/libraries/ngs2/ngs2_custom.h"
#include "core/libraries/ngs2/ngs2_error.h"
#include "libatrac9.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/samplefmt.h>
}

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace Libraries::Ngs2::Runtime {
namespace {

constexpr u32 DefaultMaxVoices = 64;
constexpr u32 MaxVoiceParams = 64;
constexpr u32 MaxWaveformBlocks = 64;
constexpr u32 MaxVoicePorts = ORBIS_NGS2_CUSTOM_MAX_PORTS;
constexpr u32 MaxVoiceMatrices = 16;
constexpr float MaxRouteGain = 16.0f;
constexpr u32 VoiceStateActive = 1;
constexpr u32 VoiceStatePaused = 2;
constexpr u32 VoiceStateNeedsData = 4;
constexpr u32 OutputPcmF32Interleaved = 0x18;
constexpr u32 WaveformPcm = 0x00000001;
constexpr u32 WaveformFloat = 0x00000003;
constexpr u32 WaveformPcmS8 = 0x11;
constexpr u32 WaveformPcmS16 = 0x12;
constexpr u32 WaveformPcmS24 = 0x13;
constexpr u32 WaveformPcmS32 = 0x14;
constexpr u32 WaveformPcmF32 = 0x18;
constexpr u32 WaveformAtrac9 = 0x00010000;

constexpr u32 ParamMatrixLevels = 1;
constexpr u32 ParamPortVolume = 2;
constexpr u32 ParamPortDelay = 3;
constexpr u32 ParamPortMatrix = 4;
constexpr u32 ParamPatch = 5;
constexpr u32 ParamEvent = 6;
constexpr u32 ParamCallback = 7;
constexpr u32 ParamSamplerSetup = 0x40010000;
constexpr u32 ParamSamplerBlocks = 0x40010001;
constexpr u32 ParamSamplerAddress = 0x40010002;
constexpr u32 ParamSamplerFrameOffset = 0x40010003;
constexpr u32 ParamSamplerExitLoop = 0x40010004;
constexpr u32 ParamSamplerPitch = 0x40010005;
constexpr u32 ParamSubmixerSetup = 0x20010000;
constexpr u32 ParamMasteringSetup = 0x30000000;
constexpr u32 ParamMasteringOutput = 0x30000001;
constexpr u32 LegacyParamSamplerSetup = 0x10000000;
constexpr u32 LegacyParamSamplerPitch = 0x10000005;
constexpr u32 LegacyParamSubmixerSetup = 0x20000000;

constexpr u32 NormalizeVoiceParamId(u32 id) {
    if (id >= LegacyParamSamplerSetup && id <= LegacyParamSamplerPitch) {
        return ParamSamplerSetup + (id - LegacyParamSamplerSetup);
    }
    if (id == LegacyParamSubmixerSetup) {
        return ParamSubmixerSetup;
    }
    return id;
}

constexpr bool IsIntegerPcmWaveform(u32 waveform_type) {
    return waveform_type == WaveformPcm ||
           (waveform_type >= WaveformPcmS8 && waveform_type <= WaveformPcmS32);
}

constexpr bool IsFloatPcmWaveform(u32 waveform_type) {
    return waveform_type == WaveformFloat || waveform_type == WaveformPcmF32;
}

constexpr bool IsPcmWaveform(u32 waveform_type) {
    return IsIntegerPcmWaveform(waveform_type) || IsFloatPcmWaveform(waveform_type);
}

constexpr u32 OutputPcmBytesPerSample(u32 waveform_type) {
    switch (waveform_type) {
    case WaveformPcmS8:
        return 1;
    case WaveformPcmS16:
        return 2;
    case WaveformPcmS24:
        return 3;
    case WaveformPcmS32:
    case WaveformPcmF32:
        return 4;
    default:
        return 0;
    }
}

void EncodeOutputPcm(void* output, u32 waveform_type, const float* samples,
                     size_t sample_count) {
    switch (waveform_type) {
    case WaveformPcmS8: {
        auto* destination = static_cast<s8*>(output);
        for (size_t index = 0; index < sample_count; ++index) {
            const float sample = samples[index];
            destination[index] =
                sample <= -1.0f ? std::numeric_limits<s8>::min()
                                : static_cast<s8>(sample * std::numeric_limits<s8>::max());
        }
        break;
    }
    case WaveformPcmS16: {
        auto* destination = static_cast<s16*>(output);
        for (size_t index = 0; index < sample_count; ++index) {
            const float sample = samples[index];
            destination[index] =
                sample <= -1.0f ? std::numeric_limits<s16>::min()
                                : static_cast<s16>(sample * std::numeric_limits<s16>::max());
        }
        break;
    }
    case WaveformPcmS24: {
        auto* destination = static_cast<u8*>(output);
        for (size_t index = 0; index < sample_count; ++index) {
            const float sample = samples[index];
            const s32 encoded = sample <= -1.0f ? -0x800000
                                                 : static_cast<s32>(sample * 0x7fffff);
            destination[index * 3] = static_cast<u8>(encoded);
            destination[index * 3 + 1] = static_cast<u8>(encoded >> 8);
            destination[index * 3 + 2] = static_cast<u8>(encoded >> 16);
        }
        break;
    }
    case WaveformPcmS32: {
        auto* destination = static_cast<s32*>(output);
        for (size_t index = 0; index < sample_count; ++index) {
            const float sample = samples[index];
            destination[index] =
                sample <= -1.0f
                    ? std::numeric_limits<s32>::min()
                    : static_cast<s32>(static_cast<double>(sample) *
                                       std::numeric_limits<s32>::max());
        }
        break;
    }
    default:
        break;
    }
}

constexpr u32 PcmBitDepth(const OrbisNgs2WaveformFormat& format) {
    switch (format.waveformType) {
    case WaveformPcmS8:
        return 8;
    case WaveformPcmS16:
        return 16;
    case WaveformPcmS24:
        return 24;
    case WaveformPcmS32:
    case WaveformPcmF32:
        return 32;
    default:
        return format.configData & 0xffffu;
    }
}

enum class VoiceEvent : u32 {
    Play = 0,
    Stop = 1,
    StopImmediate = 2,
    Resume = 3,
    Kill = 4,
    Pause = 5,
};

enum class VoiceKind : u8 {
    Unknown,
    Sampler,
    Submixer,
    Mastering,
};

bool ReadAtrac9CodecInfo(std::array<u8, ATRAC9_CONFIG_DATA_SIZE>& config,
                         Atrac9CodecInfo& info) {
    const auto try_config = [&]() {
        void* decoder = Atrac9GetHandle();
        if (decoder == nullptr) {
            return false;
        }
        info = {};
        const bool ready = Atrac9InitDecoder(decoder, config.data()) == 0 &&
                           Atrac9GetCodecInfo(decoder, &info) == 0;
        Atrac9ReleaseHandle(decoder);
        return ready;
    };

    if (try_config()) {
        return true;
    }
    std::reverse(config.begin(), config.end());
    return try_config();
}

#ifdef __ANDROID__
void TraceVoiceParam(OrbisNgs2Handle voice_handle, OrbisNgs2Handle rack_handle,
                     u32 voice_index, const OrbisNgs2VoiceParamHeader* param) {
    static std::unordered_map<u32, u32> counts;
    u32& count = counts[param->id];
    if (count >= 64) {
        return;
    }
    ++count;
    std::array<u32, 6> words{};
    const size_t payload_size =
        param->size > sizeof(*param) ? param->size - sizeof(*param) : 0;
    std::memcpy(words.data(), reinterpret_cast<const u8*>(param) + sizeof(*param),
                std::min(payload_size, sizeof(words)));
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[LSX4_NGS2_PARAM] handle=0x%llx rack=0x%llx voice=%u id=0x%08x size=%u "
        "next=%d data=%08x,%08x,%08x,%08x,%08x,%08x",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(voice_handle)),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rack_handle)),
        voice_index, param->id, param->size, param->next, words[0], words[1], words[2],
        words[3], words[4], words[5]);
}
#endif

struct SystemState {
    OrbisNgs2Handle handle{};
    OrbisNgs2ContextBufferInfo buffer_info{};
    uintptr_t user_data{};
    u32 sample_rate{48000};
    u32 max_grain_samples{512};
    u32 grain_samples{256};
    u64 render_count{};
};

struct RackState {
    OrbisNgs2Handle handle{};
    OrbisNgs2Handle system_handle{};
    OrbisNgs2ContextBufferInfo buffer_info{};
    uintptr_t user_data{};
    u32 rack_id{};
    u32 max_voices{DefaultMaxVoices};
    u64 render_count{};
    std::vector<OrbisNgs2Handle> voices;
};

struct PortState {
    s32 matrix_id{-1};
    float volume{1.0f};
    u32 delay_samples{};
    u32 dest_input{};
    OrbisNgs2Handle dest_handle{};
};

struct MatrixState {
    u32 num_levels{};
    std::array<float, ORBIS_NGS2_MAX_MATRIX_LEVELS> levels{};
};

struct SamplerVoiceState {
    OrbisNgs2VoiceState voice_state{};
    float envelope_height{};
    float peak_height{};
    u32 reserved{};
    u64 decoded_samples{};
    u64 decoded_bytes{};
    u64 user_data{};
    const void* waveform_data{};
};
static_assert(sizeof(SamplerVoiceState) == 48);

struct VoiceState {
    ~VoiceState() {
        if (decoder != nullptr) {
            avcodec_free_context(&decoder);
        }
        if (decoder_frame != nullptr) {
            av_frame_free(&decoder_frame);
        }
    }

    OrbisNgs2Handle handle{};
    OrbisNgs2Handle rack_handle{};
    u32 voice_index{};
    u32 state_flags{};
    VoiceKind kind{VoiceKind::Unknown};
    bool play_requested{};
    bool paused{};
    bool starved{};
    bool callback_pending{};
    OrbisNgs2WaveformFormat format{};
    bool has_format{};
    const u8* data{};
    std::vector<OrbisNgs2WaveformBlock> blocks;
    std::vector<const u8*> block_data;
    std::vector<u64> block_fingerprints;
    u32 block_index{};
    u32 block_repeats_left{};
    u32 block_byte_offset{};
    u32 block_skip_left{};
    u64 block_samples_left{};
    bool block_samples_bounded{};
    float pitch{1.0f};
    u32 frame_offset{};
    bool exit_loop{};
    OrbisNgs2VoiceCallbackHandler callback{};
    uintptr_t callback_data{};
    u32 callback_flags{};
    std::array<PortState, MaxVoicePorts> ports{};
    std::array<MatrixState, MaxVoiceMatrices> matrices{};
    u32 input_channels{};
    u32 output_channels{};
    u32 output_id{};
    AVCodecContext* decoder{};
    AVFrame* decoder_frame{};
    Atrac9CodecInfo codec{};
    bool decoder_ready{};
    std::vector<float> decoded_frame;
    u32 decoded_frame_pos{};
    u32 decoded_frame_samples{};
    std::vector<u8> compressed_packet;
    std::array<float, ORBIS_NGS2_MAX_VOICE_CHANNELS> source_frame{};
    std::array<float, ORBIS_NGS2_MAX_VOICE_CHANNELS> next_source_frame{};
    u32 source_frame_channels{};
    u32 next_source_frame_channels{};
    double source_phase{};
    bool source_frame_ready{};
    bool next_source_frame_ready{};
    bool decoder_output_reported{};
    bool decoder_failure_reported{};
    bool route_output_reported{};
    bool mixed_output_reported{};
    bool block_layout_reported{};
    u64 decoded_samples{};
    u64 decoded_bytes{};
};

struct PendingCallback {
    void* handler{};
    OrbisNgs2VoiceCallbackInfo info{};
};

bool SameWaveformBlock(const OrbisNgs2WaveformBlock& left,
                       const OrbisNgs2WaveformBlock& right) {
    return left.dataOffset == right.dataOffset && left.dataSize == right.dataSize &&
           left.numRepeats == right.numRepeats &&
           left.numSkipSamples == right.numSkipSamples &&
           left.numSamples == right.numSamples;
}

const u8* BlockData(const VoiceState& voice, size_t index) {
    return index < voice.block_data.size() ? voice.block_data[index] : voice.data;
}

u64 FingerprintBlockData(const VoiceState& voice, size_t index) {
    if (index >= voice.blocks.size()) {
        return 0;
    }
    const u8* base = BlockData(voice, index);
    if (base == nullptr) {
        return 0;
    }
    const OrbisNgs2WaveformBlock& block = voice.blocks[index];
    const u8* bytes = base + block.dataOffset;
    u64 hash = 1469598103934665603ull;
    for (size_t offset = 0; offset < block.dataSize; ++offset) {
        hash ^= bytes[offset];
        hash *= 1099511628211ull;
    }
    return hash;
}

void RefreshBlockFingerprints(VoiceState& voice) {
    voice.block_fingerprints.resize(voice.blocks.size());
    for (size_t index = 0; index < voice.blocks.size(); ++index) {
        voice.block_fingerprints[index] = FingerprintBlockData(voice, index);
    }
}

bool AnyBlockDataChanged(const VoiceState& voice) {
    if (voice.block_fingerprints.size() != voice.blocks.size()) {
        return true;
    }
    for (size_t index = 0; index < voice.blocks.size(); ++index) {
        if (voice.block_fingerprints[index] != FingerprintBlockData(voice, index)) {
            return true;
        }
    }
    return false;
}

bool HasAudiblePort(const VoiceState& voice) {
    return std::abs(voice.ports[0].volume) > 0.000001f;
}

bool SameWaveformBlocks(const std::vector<OrbisNgs2WaveformBlock>& current,
                        const OrbisNgs2WaveformBlock* incoming, u32 count) {
    if (incoming == nullptr || current.size() != count) {
        return count == 0 && current.empty();
    }
    for (u32 index = 0; index < count; ++index) {
        if (!SameWaveformBlock(current[index], incoming[index])) {
            return false;
        }
    }
    return true;
}

class Ngs2Runtime {
public:
    static Ngs2Runtime& Instance() {
        static Ngs2Runtime runtime;
        return runtime;
    }

    s32 CreateSystem(const OrbisNgs2SystemOption* option,
                     const OrbisNgs2ContextBufferInfo* buffer_info,
                     OrbisNgs2Handle* out_handle) {
        if (out_handle == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        auto system = std::make_unique<SystemState>();
        std::scoped_lock lock(mutex_);
        system->handle = AllocateHandle();
        if (buffer_info != nullptr) {
            system->buffer_info = *buffer_info;
            system->user_data = buffer_info->userData;
        }
        if (option != nullptr) {
            system->sample_rate = option->sampleRate;
            system->max_grain_samples = option->maxGrainSamples;
            system->grain_samples = option->numGrainSamples;
        }
        *out_handle = system->handle;
        systems_.emplace(system->handle, std::move(system));
        return ORBIS_OK;
    }

    s32 DestroySystem(OrbisNgs2Handle handle, OrbisNgs2ContextBufferInfo* out_buffer_info) {
        std::scoped_lock lock(mutex_);
        const auto system_it = systems_.find(handle);
        if (system_it == systems_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        if (out_buffer_info != nullptr) {
            *out_buffer_info = system_it->second->buffer_info;
            out_buffer_info->userData = system_it->second->user_data;
        }
        std::vector<OrbisNgs2Handle> racks;
        for (const auto& [rack_handle, rack] : racks_) {
            if (rack->system_handle == handle) {
                racks.push_back(rack_handle);
            }
        }
        for (const OrbisNgs2Handle rack_handle : racks) {
            DestroyRackLocked(rack_handle, nullptr);
        }
        systems_.erase(system_it);
        return ORBIS_OK;
    }

    s32 GetSystemInfo(OrbisNgs2Handle handle, OrbisNgs2SystemInfo* out_info,
                      size_t info_size) {
        if (out_info == nullptr || info_size < sizeof(OrbisNgs2SystemInfo)) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = systems_.find(handle);
        if (it == systems_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        OrbisNgs2SystemInfo info{};
        info.systemHandle = handle;
        info.bufferInfo = it->second->buffer_info;
        info.bufferInfo.userData = it->second->user_data;
        info.minGrainSamples = 64;
        info.maxGrainSamples = it->second->max_grain_samples;
        info.sampleRate = it->second->sample_rate;
        info.numGrainSamples = it->second->grain_samples;
        info.renderCount = static_cast<s64>(it->second->render_count);
        info.stateFlags = 1;
        for (const auto& [rack_handle, rack] : racks_) {
            if (rack->system_handle == handle) {
                ++info.rackCount;
            }
        }
        *out_info = info;
        return ORBIS_OK;
    }

    s32 GetSystemUserData(OrbisNgs2Handle handle, uintptr_t* out_user_data) {
        if (out_user_data == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = systems_.find(handle);
        if (it == systems_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        *out_user_data = it->second->user_data;
        return ORBIS_OK;
    }

    s32 SetSystemUserData(OrbisNgs2Handle handle, uintptr_t user_data) {
        std::scoped_lock lock(mutex_);
        const auto it = systems_.find(handle);
        if (it == systems_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        it->second->user_data = user_data;
        return ORBIS_OK;
    }

    s32 SetSystemGrainSamples(OrbisNgs2Handle handle, u32 samples) {
        std::scoped_lock lock(mutex_);
        const auto it = systems_.find(handle);
        if (it == systems_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        it->second->grain_samples = samples;
        return ORBIS_OK;
    }

    s32 SetSystemSampleRate(OrbisNgs2Handle handle, u32 sample_rate) {
        std::scoped_lock lock(mutex_);
        const auto it = systems_.find(handle);
        if (it == systems_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        it->second->sample_rate = sample_rate;
        return ORBIS_OK;
    }

    s32 CreateRack(OrbisNgs2Handle system_handle, u32 rack_id,
                   const OrbisNgs2RackOption* option,
                   const OrbisNgs2ContextBufferInfo* buffer_info,
                   OrbisNgs2Handle* out_handle) {
        if (out_handle == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        auto rack = std::make_unique<RackState>();
        std::scoped_lock lock(mutex_);
        if (!systems_.contains(system_handle)) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        rack->handle = AllocateHandle();
        rack->system_handle = system_handle;
        rack->rack_id = rack_id;
        if (option != nullptr && option->maxVoices != 0) {
            rack->max_voices = option->maxVoices;
        }
        if (buffer_info != nullptr) {
            rack->buffer_info = *buffer_info;
            rack->user_data = buffer_info->userData;
        }
        rack->voices.resize(rack->max_voices);
        *out_handle = rack->handle;
        racks_.emplace(rack->handle, std::move(rack));
        return ORBIS_OK;
    }

    s32 DestroyRack(OrbisNgs2Handle handle, OrbisNgs2ContextBufferInfo* out_buffer_info) {
        std::scoped_lock lock(mutex_);
        return DestroyRackLocked(handle, out_buffer_info);
    }

    s32 GetRackInfo(OrbisNgs2Handle handle, OrbisNgs2RackInfo* out_info, size_t info_size) {
        if (out_info == nullptr || info_size < sizeof(OrbisNgs2RackInfo)) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = racks_.find(handle);
        if (it == racks_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
        }
        OrbisNgs2RackInfo info{};
        info.rackHandle = handle;
        info.ownerSystemHandle = it->second->system_handle;
        info.bufferInfo = it->second->buffer_info;
        info.bufferInfo.userData = it->second->user_data;
        info.rackId = it->second->rack_id;
        info.maxVoices = it->second->max_voices;
        info.minGrainSamples = 64;
        info.maxGrainSamples = 1024;
        info.renderCount = it->second->render_count;
        for (const OrbisNgs2Handle voice_handle : it->second->voices) {
            const auto voice_it = voices_.find(voice_handle);
            if (voice_it != voices_.end() && voice_it->second->state_flags != 0) {
                ++info.activeVoiceCount;
            }
        }
        *out_info = info;
        return ORBIS_OK;
    }

    s32 GetRackUserData(OrbisNgs2Handle handle, uintptr_t* out_user_data) {
        if (out_user_data == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = racks_.find(handle);
        if (it == racks_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
        }
        *out_user_data = it->second->user_data;
        return ORBIS_OK;
    }

    s32 SetRackUserData(OrbisNgs2Handle handle, uintptr_t user_data) {
        std::scoped_lock lock(mutex_);
        const auto it = racks_.find(handle);
        if (it == racks_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
        }
        it->second->user_data = user_data;
        return ORBIS_OK;
    }

    s32 GetVoiceHandle(OrbisNgs2Handle rack_handle, u32 voice_index,
                       OrbisNgs2Handle* out_handle) {
        if (out_handle == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto rack_it = racks_.find(rack_handle);
        if (rack_it == racks_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
        }
        if (voice_index >= rack_it->second->max_voices) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_INDEX;
        }
        OrbisNgs2Handle& handle = rack_it->second->voices[voice_index];
        if (handle == 0) {
            auto voice = std::make_unique<VoiceState>();
            handle = AllocateHandle();
            voice->handle = handle;
            voice->rack_handle = rack_handle;
            voice->voice_index = voice_index;
            voices_.emplace(handle, std::move(voice));
        }
        *out_handle = handle;
        return ORBIS_OK;
    }

    s32 ControlVoice(OrbisNgs2Handle handle, const OrbisNgs2VoiceParamHeader* params) {
        if (params == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto voice_it = voices_.find(handle);
        if (voice_it == voices_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        VoiceState& voice = *voice_it->second;
        const OrbisNgs2VoiceParamHeader* current = params;
        for (u32 param_index = 0; param_index < MaxVoiceParams; ++param_index) {
            if (current->size < sizeof(OrbisNgs2VoiceParamHeader) || current->size > 4096) {
                break;
            }
            ApplyVoiceParam(voice, current);
            if (current->next == 0) {
                break;
            }
            current = reinterpret_cast<const OrbisNgs2VoiceParamHeader*>(
                reinterpret_cast<const u8*>(current) + current->next);
        }
        return ORBIS_OK;
    }

    s32 GetVoiceOwner(OrbisNgs2Handle handle, OrbisNgs2Handle* out_rack_handle,
                      u32* out_voice_index) {
        if (out_rack_handle == nullptr || out_voice_index == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = voices_.find(handle);
        if (it == voices_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        *out_rack_handle = it->second->rack_handle;
        *out_voice_index = it->second->voice_index;
        return ORBIS_OK;
    }

    s32 GetVoiceState(OrbisNgs2Handle handle, OrbisNgs2VoiceState* out_state,
                      size_t state_size) {
        if (out_state == nullptr || state_size < sizeof(OrbisNgs2VoiceState)) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = voices_.find(handle);
        if (it == voices_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        std::memset(out_state, 0, state_size);
        const VoiceState& voice = *it->second;
        out_state->stateFlags = voice.state_flags;
        const void* waveform_data = BlockData(voice, voice.block_index);
        if (state_size >= sizeof(OrbisNgs2CustomSamplerVoiceState)) {
            auto* sampler = reinterpret_cast<OrbisNgs2CustomSamplerVoiceState*>(out_state);
            sampler->waveformData = waveform_data;
            sampler->numDecodedSamples = voice.decoded_samples;
            sampler->decodedDataSize = voice.decoded_bytes;
        } else if (state_size >= sizeof(SamplerVoiceState)) {
            auto* sampler = reinterpret_cast<SamplerVoiceState*>(out_state);
            sampler->decoded_samples = voice.decoded_samples;
            sampler->decoded_bytes = voice.decoded_bytes;
            sampler->waveform_data = waveform_data;
        }
#ifdef __ANDROID__
        static std::atomic<u32> state_trace_count{0};
        const u32 trace_index = state_trace_count.fetch_add(1, std::memory_order_relaxed);
        if (trace_index < 160) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[LSX4_NGS2_STATE] handle=0x%llx rack=0x%llx voice=%u size=%zu "
                "flags=0x%x play=%d paused=%d starved=%d block=%u/%zu data=%p "
                "samples=%llu bytes=%llu",
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(voice.handle)),
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(voice.rack_handle)),
                voice.voice_index, state_size, voice.state_flags,
                voice.play_requested, voice.paused, voice.starved,
                voice.block_index, voice.blocks.size(), waveform_data,
                static_cast<unsigned long long>(voice.decoded_samples),
                static_cast<unsigned long long>(voice.decoded_bytes));
        }
        if (voice.starved && voice.play_requested) {
            static std::atomic<u32> starved_state_trace_count{0};
            if (starved_state_trace_count.fetch_add(1, std::memory_order_relaxed) < 96) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[LSX4_NGS2_STARVED_STATE] handle=0x%llx rack=0x%llx voice=%u "
                    "size=%zu flags=0x%x block=%u/%zu data=%p samples=%llu bytes=%llu "
                    "callback=%d",
                    static_cast<unsigned long long>(
                        reinterpret_cast<uintptr_t>(voice.handle)),
                    static_cast<unsigned long long>(
                        reinterpret_cast<uintptr_t>(voice.rack_handle)),
                    voice.voice_index, state_size, voice.state_flags, voice.block_index,
                    voice.blocks.size(), waveform_data,
                    static_cast<unsigned long long>(voice.decoded_samples),
                    static_cast<unsigned long long>(voice.decoded_bytes),
                    voice.callback != nullptr);
            }
        }
#endif
        return ORBIS_OK;
    }

    s32 GetVoiceStateFlags(OrbisNgs2Handle handle, u32* out_state_flags) {
        if (out_state_flags == nullptr) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = voices_.find(handle);
        if (it == voices_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        *out_state_flags = it->second->state_flags;
#ifdef __ANDROID__
        static std::atomic<u32> flags_trace_count{0};
        if (flags_trace_count.fetch_add(1, std::memory_order_relaxed) < 160) {
            const VoiceState& voice = *it->second;
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[LSX4_NGS2_STATE_FLAGS] handle=0x%llx rack=0x%llx voice=%u "
                "flags=0x%x play=%d paused=%d starved=%d block=%u/%zu",
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(voice.handle)),
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(voice.rack_handle)),
                voice.voice_index, voice.state_flags, voice.play_requested,
                voice.paused, voice.starved, voice.block_index, voice.blocks.size());
        }
        if (it->second->starved && it->second->play_requested) {
            static std::atomic<u32> starved_flags_trace_count{0};
            if (starved_flags_trace_count.fetch_add(1, std::memory_order_relaxed) < 96) {
                const VoiceState& voice = *it->second;
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[LSX4_NGS2_STARVED_FLAGS] handle=0x%llx rack=0x%llx voice=%u "
                    "flags=0x%x block=%u/%zu callback=%d",
                    static_cast<unsigned long long>(
                        reinterpret_cast<uintptr_t>(voice.handle)),
                    static_cast<unsigned long long>(
                        reinterpret_cast<uintptr_t>(voice.rack_handle)),
                    voice.voice_index, voice.state_flags, voice.block_index,
                    voice.blocks.size(), voice.callback != nullptr);
            }
        }
#endif
        return ORBIS_OK;
    }

    s32 GetVoiceMatrixInfo(OrbisNgs2Handle handle, u32 matrix_id,
                           OrbisNgs2VoiceMatrixInfo* out_info, size_t info_size) {
        if (out_info == nullptr || info_size < sizeof(OrbisNgs2VoiceMatrixInfo)) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = voices_.find(handle);
        if (it == voices_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        *out_info = {};
        if (matrix_id < it->second->matrices.size()) {
            const MatrixState& matrix = it->second->matrices[matrix_id];
            out_info->numLevels = matrix.num_levels;
            std::copy_n(matrix.levels.begin(), matrix.num_levels, out_info->aLevel);
        }
        return ORBIS_OK;
    }

    s32 GetVoicePortInfo(OrbisNgs2Handle handle, u32 port, OrbisNgs2VoicePortInfo* out_info,
                          size_t info_size) {
        if (out_info == nullptr || info_size < sizeof(OrbisNgs2VoicePortInfo)) {
            return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        }
        std::scoped_lock lock(mutex_);
        const auto it = voices_.find(handle);
        if (it == voices_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_VOICE_HANDLE;
        }
        OrbisNgs2VoicePortInfo info{};
        info.matrixId = -1;
        info.volume = 1.0f;
        if (port < it->second->ports.size()) {
            const PortState& state = it->second->ports[port];
            info.matrixId = state.matrix_id;
            info.volume = state.volume;
            info.numDelaySamples = state.delay_samples;
            info.destInputId = state.dest_input;
            info.destHandle = state.dest_handle;
        }
        *out_info = info;
        return ORBIS_OK;
    }

    s32 Render(OrbisNgs2Handle system_handle, const OrbisNgs2RenderBufferInfo* buffers,
               u32 num_buffers) {
        std::unique_lock lock(mutex_);
        const auto system_it = systems_.find(system_handle);
        if (system_it == systems_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
        }
        ++system_it->second->render_count;
        for (auto& [rack_handle, rack] : racks_) {
            if (rack->system_handle == system_handle) {
                ++rack->render_count;
            }
        }
        if (buffers == nullptr || num_buffers == 0) {
            return ORBIS_OK;
        }
        std::vector<PendingCallback> pending_callbacks;
        for (u32 buffer_index = 0; buffer_index < num_buffers; ++buffer_index) {
            const OrbisNgs2RenderBufferInfo& output = buffers[buffer_index];
            if (output.buffer == nullptr || output.bufferSize == 0 || output.numChannels == 0) {
                continue;
            }
            std::memset(output.buffer, 0, output.bufferSize);
            const u32 bytes_per_sample = OutputPcmBytesPerSample(output.waveformType);
            if (bytes_per_sample == 0) {
                continue;
            }
            const size_t sample_count = output.bufferSize / bytes_per_sample;
            const size_t frame_capacity = sample_count / output.numChannels;
            const size_t frame_count =
                std::min(frame_capacity,
                          static_cast<size_t>(system_it->second->grain_samples));
            float* samples = nullptr;
            if (output.waveformType == OutputPcmF32Interleaved) {
                samples = static_cast<float*>(output.buffer);
            } else {
                render_scratch_.resize(sample_count);
                std::fill(render_scratch_.begin(), render_scratch_.end(), 0.0f);
                samples = render_scratch_.data();
            }
            for (auto& [voice_handle, voice_ptr] : voices_) {
                VoiceState& voice = *voice_ptr;
                const auto rack_it = racks_.find(voice.rack_handle);
                if (rack_it == racks_.end() || rack_it->second->system_handle != system_handle ||
                    voice.kind != VoiceKind::Sampler || voice.state_flags == 0 || voice.paused) {
                    continue;
                }
                if (voice.callback == nullptr && !HasAudiblePort(voice)) {
                    continue;
                }
                MixVoice(voice, samples, frame_count, output.numChannels,
                         system_it->second->sample_rate, pending_callbacks);
            }
            float peak = 0.0f;
            for (size_t index = 0; index < sample_count; ++index) {
                samples[index] = std::clamp(samples[index], -1.0f, 1.0f);
                peak = std::max(peak, std::abs(samples[index]));
            }
            if (output.waveformType != OutputPcmF32Interleaved) {
                EncodeOutputPcm(output.buffer, output.waveformType, samples, sample_count);
            }
#ifdef __ANDROID__
            if (peak > 0.000001f && !reported_audio_) {
                reported_audio_ = true;
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[LSX4_NGS2_AUDIO] peak=%.6f activeVoices=%zu", peak,
                                    ActiveVoiceCount(system_handle));
            }
#endif
        }
        lock.unlock();
        for (PendingCallback& pending : pending_callbacks) {
            u64 ignored_result = 0;
            Libraries::RunGuestCallback4(
                pending.handler, reinterpret_cast<uintptr_t>(&pending.info), 0, 0, 0, 1,
                &ignored_result, "ngs2_voice_callback");
        }
        return ORBIS_OK;
    }

private:
    OrbisNgs2Handle AllocateHandle() {
        return 0x100000 + next_handle_++;
    }

    s32 DestroyRackLocked(OrbisNgs2Handle handle,
                          OrbisNgs2ContextBufferInfo* out_buffer_info) {
        const auto it = racks_.find(handle);
        if (it == racks_.end()) {
            return ORBIS_NGS2_ERROR_INVALID_RACK_HANDLE;
        }
        if (out_buffer_info != nullptr) {
            *out_buffer_info = it->second->buffer_info;
            out_buffer_info->userData = it->second->user_data;
        }
        for (const OrbisNgs2Handle voice_handle : it->second->voices) {
            voices_.erase(voice_handle);
        }
        racks_.erase(it);
        return ORBIS_OK;
    }

    void ResetDecoder(VoiceState& voice) {
        if (voice.decoder != nullptr) {
            avcodec_free_context(&voice.decoder);
        }
        if (voice.decoder_frame != nullptr) {
            av_frame_free(&voice.decoder_frame);
        }
        voice.decoder_ready = false;
        voice.decoded_frame.clear();
        voice.decoded_frame_pos = 0;
        voice.decoded_frame_samples = 0;
        voice.compressed_packet.clear();
        if (!voice.has_format) {
            return;
        }
        if (IsPcmWaveform(voice.format.waveformType)) {
            if (voice.format.numChannels == 0 ||
                voice.format.numChannels > ORBIS_NGS2_MAX_VOICE_CHANNELS) {
                return;
            }
            voice.codec = {};
            voice.codec.channels = static_cast<int>(voice.format.numChannels);
            voice.codec.frameSamples = 1;
            voice.decoded_frame.resize(voice.format.numChannels);
            voice.decoded_frame_samples = 1;
            voice.decoder_ready = true;
            return;
        }
        std::array<u8, ATRAC9_CONFIG_DATA_SIZE> config{};
        std::memcpy(config.data(), &voice.format.configData, config.size());
        const bool atrac9_candidate = voice.format.waveformType == WaveformAtrac9 ||
                                      config.front() == 0xfe || config.back() == 0xfe;
        if (!atrac9_candidate) {
            return;
        }
        const bool info_ready = ReadAtrac9CodecInfo(config, voice.codec);
        if (!info_ready || voice.codec.channels <= 0 ||
            voice.codec.channels > ORBIS_NGS2_MAX_VOICE_CHANNELS ||
            voice.codec.frameSamples <= 0 || voice.codec.superframeSize <= 0 ||
            voice.codec.framesInSuperframe <= 0) {
            return;
        }

        const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_ATRAC9);
        voice.decoder = codec == nullptr ? nullptr : avcodec_alloc_context3(codec);
        if (voice.decoder == nullptr) {
            return;
        }
        voice.decoder->block_align = voice.codec.superframeSize;
        voice.decoder->thread_count = 1;
        voice.decoder->extradata_size = 12;
        voice.decoder->extradata = static_cast<u8*>(
            av_mallocz(voice.decoder->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE));
        if (voice.decoder->extradata == nullptr) {
            avcodec_free_context(&voice.decoder);
            return;
        }
        voice.decoder->extradata[0] = 1;
        std::memcpy(voice.decoder->extradata + 4, config.data(), config.size());
        if (avcodec_open2(voice.decoder, codec, nullptr) < 0) {
            avcodec_free_context(&voice.decoder);
            return;
        }
        voice.decoder_frame = av_frame_alloc();
        if (voice.decoder_frame == nullptr) {
            avcodec_free_context(&voice.decoder);
            return;
        }
        const u32 superframe_samples =
            static_cast<u32>(voice.codec.frameSamples * voice.codec.framesInSuperframe);
        voice.decoded_frame.resize(static_cast<size_t>(superframe_samples) *
                                   voice.codec.channels);
        voice.compressed_packet.resize(static_cast<size_t>(voice.codec.superframeSize) +
                                       AV_INPUT_BUFFER_PADDING_SIZE);
        voice.decoder_ready = true;
    }

    void StartBlock(VoiceState& voice, bool reset_repeats) {
        if (voice.block_index >= voice.blocks.size()) {
            voice.starved = true;
            return;
        }
        const OrbisNgs2WaveformBlock& block = voice.blocks[voice.block_index];
        if (reset_repeats) {
            voice.block_repeats_left = block.numRepeats;
        }
        voice.block_byte_offset = 0;
        voice.block_skip_left = block.numSkipSamples;
        voice.block_samples_left = block.numSamples;
        voice.block_samples_bounded = block.numSamples != 0;
        voice.decoded_frame_pos = voice.decoded_frame_samples;
        voice.starved = false;
        voice.callback_pending = false;
    }

    void QueueBlockCallback(VoiceState& voice, const OrbisNgs2WaveformBlock& block,
                            u32 repeated_count,
                            std::vector<PendingCallback>& pending_callbacks) {
        if (voice.callback == nullptr) {
            return;
        }
        PendingCallback pending{};
        pending.handler = reinterpret_cast<void*>(voice.callback);
        pending.info.callbackData = voice.callback_data;
        pending.info.voiceHandle = voice.handle;
        pending.info.flag = 1;
        pending.info.param.waveformBlock.userData = block.userData;
        const u8* block_data = BlockData(voice, voice.block_index);
        pending.info.param.waveformBlock.data =
            block_data == nullptr ? nullptr : block_data + block.dataOffset;
        pending.info.param.waveformBlock.dataSize = block.dataSize;
        pending.info.param.waveformBlock.repeatedCount = repeated_count;
        pending.info.param.waveformBlock.attributeFlags = block.reserved;
        pending_callbacks.push_back(pending);
    }

    void FinishPlayback(VoiceState& voice) {
        voice.starved = true;
#ifdef __ANDROID__
        static std::atomic<u32> finish_trace_count{0};
        if (finish_trace_count.fetch_add(1, std::memory_order_relaxed) < 96) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[LSX4_NGS2_FINISH] handle=0x%llx rack=0x%llx voice=%u play=%d "
                "callback=%d blocks=%zu decoded=%llu volume=%.6f",
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(voice.handle)),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(voice.rack_handle)),
                voice.voice_index, voice.play_requested, voice.callback != nullptr,
                voice.blocks.size(),
                static_cast<unsigned long long>(voice.decoded_samples),
                voice.ports[0].volume);
        }
#endif
        if (voice.play_requested &&
            (voice.callback != nullptr || voice.kind == VoiceKind::Sampler)) {
            voice.state_flags =
                voice.callback == nullptr && voice.kind == VoiceKind::Sampler
                    ? VoiceStateActive | VoiceStateNeedsData
                    : VoiceStateActive;
            return;
        }
        voice.state_flags = 0;
    }

    void AdvanceBlock(VoiceState& voice, std::vector<PendingCallback>& pending_callbacks) {
        if (voice.block_index >= voice.blocks.size()) {
            FinishPlayback(voice);
            return;
        }
        const OrbisNgs2WaveformBlock& completed = voice.blocks[voice.block_index];
#ifdef __ANDROID__
        static std::atomic<u32> block_done_trace_count{0};
        if (block_done_trace_count.fetch_add(1, std::memory_order_relaxed) < 96) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[LSX4_NGS2_BLOCK_DONE] handle=0x%llx rack=0x%llx voice=%u "
                "block=%u/%zu size=%u samples=%u volume=%.6f callback=%d",
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(voice.handle)),
                static_cast<unsigned long long>(
                    reinterpret_cast<uintptr_t>(voice.rack_handle)),
                voice.voice_index, voice.block_index, voice.blocks.size(),
                completed.dataSize, completed.numSamples, voice.ports[0].volume,
                voice.callback != nullptr);
        }
#endif
        const u32 repeated_count = completed.numRepeats == std::numeric_limits<u32>::max()
                                       ? std::numeric_limits<u32>::max()
                                       : completed.numRepeats - voice.block_repeats_left;
        QueueBlockCallback(voice, completed, repeated_count, pending_callbacks);
        if (!voice.exit_loop && voice.block_repeats_left != 0) {
            if (voice.block_repeats_left != std::numeric_limits<u32>::max()) {
                --voice.block_repeats_left;
            }
            StartBlock(voice, false);
            return;
        }
        voice.exit_loop = false;
        ++voice.block_index;
        if (voice.block_index >= voice.blocks.size()) {
            voice.callback_pending = voice.callback != nullptr;
            FinishPlayback(voice);
            return;
        }
        StartBlock(voice, true);
    }

    bool DecodePcmFrame(VoiceState& voice) {
        const u8* block_data = BlockData(voice, voice.block_index);
        if (voice.block_index >= voice.blocks.size() || block_data == nullptr) {
            return false;
        }
        const OrbisNgs2WaveformBlock& block = voice.blocks[voice.block_index];
        const u32 channels = voice.format.numChannels;
        const u32 bits = PcmBitDepth(voice.format);
        const u32 bytes_per_sample = bits / 8;
        const u32 configured_stride = voice.format.configData >> 16;
        const u32 frame_stride =
            configured_stride != 0 ? configured_stride : channels * bytes_per_sample;
        if (channels == 0 || bytes_per_sample == 0 || frame_stride == 0 ||
            voice.block_byte_offset > block.dataSize ||
            frame_stride > block.dataSize - voice.block_byte_offset) {
            return false;
        }
        const u8* frame = block_data + block.dataOffset + voice.block_byte_offset;
        for (u32 channel = 0; channel < channels; ++channel) {
            const u8* sample = frame + channel * bytes_per_sample;
            float value = 0.0f;
            if (IsFloatPcmWaveform(voice.format.waveformType) && bits == 32) {
                std::memcpy(&value, sample, sizeof(value));
            } else if (IsFloatPcmWaveform(voice.format.waveformType) && bits == 64) {
                double wide = 0.0;
                std::memcpy(&wide, sample, sizeof(wide));
                value = static_cast<float>(wide);
            } else if (bits == 8) {
                value = (static_cast<int>(*sample) - 128) / 128.0f;
            } else if (bits == 16) {
                s16 integer = 0;
                std::memcpy(&integer, sample, sizeof(integer));
                value = integer / 32768.0f;
            } else if (bits == 24) {
                s32 integer = static_cast<s32>(sample[0]) |
                              (static_cast<s32>(sample[1]) << 8) |
                              (static_cast<s32>(sample[2]) << 16);
                if ((integer & 0x00800000) != 0) {
                    integer |= static_cast<s32>(0xff000000);
                }
                value = integer / 8388608.0f;
            } else if (bits == 32) {
                s32 integer = 0;
                std::memcpy(&integer, sample, sizeof(integer));
                value = integer / 2147483648.0f;
            } else {
                return false;
            }
            voice.decoded_frame[channel] = std::isfinite(value) ? value : 0.0f;
        }
        voice.block_byte_offset += frame_stride;
        voice.decoded_bytes += frame_stride;
        voice.decoded_frame_samples = 1;
        voice.decoded_frame_pos = voice.block_skip_left == 0 ? 0 : 1;
        if (voice.block_skip_left != 0) {
            --voice.block_skip_left;
        }
#ifdef __ANDROID__
        if (!voice.decoder_output_reported) {
            float peak = 0.0f;
            for (u32 channel = 0; channel < channels; ++channel) {
                peak = std::max(peak, std::abs(voice.decoded_frame[channel]));
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[LSX4_NGS2_DECODE] voice=%u pcmBits=%u channels=%u peak=%.6f",
                voice.voice_index, bits, channels, peak);
            voice.decoder_output_reported = true;
        }
#endif
        return true;
    }

    bool DecodeFrame(VoiceState& voice) {
        const u8* block_data = BlockData(voice, voice.block_index);
        if (!voice.decoder_ready || voice.block_index >= voice.blocks.size() ||
            block_data == nullptr) {
            return false;
        }
        if (IsPcmWaveform(voice.format.waveformType)) {
            return DecodePcmFrame(voice);
        }
        const OrbisNgs2WaveformBlock& block = voice.blocks[voice.block_index];
        if (voice.block_byte_offset >= block.dataSize) {
            return false;
        }
        const u32 packet_size = static_cast<u32>(voice.codec.superframeSize);
        const u32 remaining = block.dataSize - voice.block_byte_offset;
        if (packet_size == 0 || packet_size > remaining ||
            voice.compressed_packet.size() <
                static_cast<size_t>(packet_size) + AV_INPUT_BUFFER_PADDING_SIZE ||
            voice.decoder_frame == nullptr) {
            return false;
        }
        const u8* source = block_data + block.dataOffset + voice.block_byte_offset;
        std::memcpy(voice.compressed_packet.data(), source, packet_size);
        std::memset(voice.compressed_packet.data() + packet_size, 0,
                    AV_INPUT_BUFFER_PADDING_SIZE);
        AVPacket packet{};
        packet.data = voice.compressed_packet.data();
        packet.size = static_cast<int>(packet_size);
        int send_result = 0;
        int receive_result = 0;
        auto decode_packet = [&] {
            av_frame_unref(voice.decoder_frame);
            send_result = avcodec_send_packet(voice.decoder, &packet);
            receive_result =
                send_result >= 0 ? avcodec_receive_frame(voice.decoder, voice.decoder_frame)
                                 : send_result;
        };
        auto valid_output = [&] {
            const u32 channels =
                static_cast<u32>(voice.decoder_frame->ch_layout.nb_channels);
            const u32 samples =
                receive_result >= 0 ? static_cast<u32>(voice.decoder_frame->nb_samples) : 0;
            return receive_result >= 0 && samples != 0 &&
                   channels == static_cast<u32>(voice.codec.channels) &&
                   voice.decoder_frame->format == AV_SAMPLE_FMT_FLTP &&
                   voice.decoder_frame->extended_data != nullptr &&
                   samples <= static_cast<u32>(voice.codec.frameSamples *
                                               voice.codec.framesInSuperframe);
        };
        decode_packet();
        bool recovered = false;
        if (!valid_output()) {
            avcodec_flush_buffers(voice.decoder);
            decode_packet();
            recovered = valid_output();
        }
        const bool concealed = !valid_output();
        const u32 decoded_channels = static_cast<u32>(voice.codec.channels);
        const u32 decoded_samples =
            concealed
                ? static_cast<u32>(voice.codec.frameSamples * voice.codec.framesInSuperframe)
                : static_cast<u32>(voice.decoder_frame->nb_samples);
        if (concealed) {
            std::fill_n(voice.decoded_frame.begin(),
                        static_cast<size_t>(decoded_samples) * decoded_channels, 0.0f);
            avcodec_flush_buffers(voice.decoder);
#ifdef __ANDROID__
            if (!voice.decoder_failure_reported) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[LSX4_NGS2_DECODE] voice=%u concealed send=%d receive=%d offset=%u",
                    voice.voice_index, send_result, receive_result, voice.block_byte_offset);
                voice.decoder_failure_reported = true;
            }
#endif
        } else {
            for (u32 sample = 0; sample < decoded_samples; ++sample) {
                for (u32 channel = 0; channel < decoded_channels; ++channel) {
                    const auto* plane = reinterpret_cast<const float*>(
                        voice.decoder_frame->extended_data[channel]);
                    const float value = plane[sample];
                    voice.decoded_frame[static_cast<size_t>(sample) * decoded_channels +
                                        channel] = std::isfinite(value) ? value : 0.0f;
                }
            }
        }
        voice.block_byte_offset += packet_size;
        voice.decoded_bytes += packet_size;
        voice.decoded_frame_samples = decoded_samples;
        voice.decoded_frame_pos = std::min(voice.block_skip_left, decoded_samples);
        voice.block_skip_left -= voice.decoded_frame_pos;
#ifdef __ANDROID__
        if (!voice.decoder_output_reported && !concealed) {
            float peak = 0.0f;
            const size_t decoded_values =
                static_cast<size_t>(decoded_samples) * decoded_channels;
            for (size_t index = 0; index < decoded_values; ++index) {
                const float sample = voice.decoded_frame[index];
                if (std::isfinite(sample)) {
                    peak = std::max(peak, std::abs(sample));
                }
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[LSX4_NGS2_DECODE] voice=%u bytes=%u samples=%u channels=%u "
                "peak=%.6f",
                voice.voice_index, packet_size, decoded_samples, decoded_channels, peak);
            voice.decoder_output_reported = true;
        }
        if (recovered) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[LSX4_NGS2_DECODE] voice=%u recovered offset=%u",
                                voice.voice_index, voice.block_byte_offset - packet_size);
        }
#endif
        return true;
    }

    bool NextSourceFrame(VoiceState& voice,
                         std::array<float, ORBIS_NGS2_MAX_VOICE_CHANNELS>& out,
                         u32& out_channels,
                         std::vector<PendingCallback>& pending_callbacks) {
        for (u32 attempts = 0; attempts < MaxWaveformBlocks; ++attempts) {
            if (voice.state_flags == 0 || voice.paused || voice.starved ||
                voice.block_index >= voice.blocks.size()) {
                return false;
            }
            if (voice.block_samples_bounded && voice.block_samples_left == 0) {
                AdvanceBlock(voice, pending_callbacks);
                continue;
            }
            if (voice.decoded_frame_pos >= voice.decoded_frame_samples) {
                if (!DecodeFrame(voice)) {
                    AdvanceBlock(voice, pending_callbacks);
                    continue;
                }
                if (voice.decoded_frame_pos >= voice.decoded_frame_samples) {
                    continue;
                }
            }
            out.fill(0.0f);
            out_channels = static_cast<u32>(voice.codec.channels);
            const size_t source_base = static_cast<size_t>(voice.decoded_frame_pos) * out_channels;
            for (u32 channel = 0; channel < out_channels; ++channel) {
                out[channel] = voice.decoded_frame[source_base + channel];
            }
            ++voice.decoded_frame_pos;
            ++voice.decoded_samples;
            if (voice.block_samples_bounded) {
                --voice.block_samples_left;
            }
            return true;
        }
        FinishPlayback(voice);
        return false;
    }

    void BuildRouteMatrix(const VoiceState& source, u32 source_channels, u32 output_channels,
                          std::array<float, ORBIS_NGS2_MAX_MATRIX_LEVELS>& out_matrix) const {
        constexpr u32 width = ORBIS_NGS2_MAX_VOICE_CHANNELS;
        std::array<float, ORBIS_NGS2_MAX_MATRIX_LEVELS> transform{};
        for (u32 channel = 0; channel < source_channels && channel < width; ++channel) {
            transform[channel * width + channel] = 1.0f;
        }
        u32 current_channels = std::min(source_channels, width);
        const VoiceState* current = &source;
        std::array<OrbisNgs2Handle, 8> visited{};
        u32 visited_count = 0;

        for (u32 depth = 0; depth < visited.size() && current != nullptr; ++depth) {
            if (std::find(visited.begin(), visited.begin() + visited_count, current->handle) !=
                visited.begin() + visited_count) {
                break;
            }
            visited[visited_count++] = current->handle;
            const PortState& port = current->ports[0];
            const s32 selected_matrix = port.matrix_id;
            if (selected_matrix >= 0 &&
                static_cast<size_t>(selected_matrix) < current->matrices.size()) {
                const MatrixState& matrix = current->matrices[selected_matrix];
                if (current_channels != 0 && matrix.num_levels >= current_channels &&
                    matrix.num_levels % current_channels == 0) {
                    const u32 next_channels =
                        std::min(matrix.num_levels / current_channels, width);
                    std::array<float, ORBIS_NGS2_MAX_MATRIX_LEVELS> next{};
                    for (u32 dest = 0; dest < next_channels; ++dest) {
                        for (u32 original = 0; original < source_channels; ++original) {
                            float level = 0.0f;
                            for (u32 input = 0; input < current_channels; ++input) {
                                level += matrix.levels[dest * current_channels + input] *
                                         transform[input * width + original];
                            }
                            next[dest * width + original] = level;
                        }
                    }
                    transform = next;
                    current_channels = next_channels;
                }
            }
            if (port.volume != 1.0f) {
                for (u32 channel = 0; channel < current_channels; ++channel) {
                    for (u32 original = 0; original < source_channels; ++original) {
                        transform[channel * width + original] *= port.volume;
                    }
                }
            }
            const auto destination = voices_.find(port.dest_handle);
            current = destination == voices_.end() ? nullptr : destination->second.get();
        }

        out_matrix.fill(0.0f);
        if (current_channels == 1 && output_channels > 1) {
            for (u32 original = 0; original < source_channels; ++original) {
                out_matrix[original] = transform[original];
                out_matrix[width + original] = transform[original];
            }
            return;
        }
        const u32 channels = std::min(current_channels, output_channels);
        for (u32 dest = 0; dest < channels; ++dest) {
            for (u32 original = 0; original < source_channels; ++original) {
                out_matrix[dest * width + original] =
                    transform[dest * width + original];
            }
        }
    }

    void MixVoice(VoiceState& voice, float* output, size_t frame_count, u32 output_channels,
                  u32 output_sample_rate,
                  std::vector<PendingCallback>& pending_callbacks) {
        std::array<float, ORBIS_NGS2_MAX_MATRIX_LEVELS> route{};
        const u32 expected_channels =
            voice.decoder_ready ? static_cast<u32>(voice.codec.channels)
                                : std::min(voice.format.numChannels,
                                           static_cast<u32>(ORBIS_NGS2_MAX_VOICE_CHANNELS));
        BuildRouteMatrix(voice, expected_channels, output_channels, route);
#ifdef __ANDROID__
        if (!voice.route_output_reported) {
            float peak = 0.0f;
            for (const float level : route) {
                peak = std::max(peak, std::abs(level));
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[LSX4_NGS2_MIX] voice=%u volume=%.6f matrix=%d routePeak=%.6f",
                voice.voice_index, voice.ports[0].volume, voice.ports[0].matrix_id, peak);
            voice.route_output_reported = true;
        }
#endif
        if (!voice.source_frame_ready) {
            if (!NextSourceFrame(voice, voice.source_frame, voice.source_frame_channels,
                                 pending_callbacks)) {
                return;
            }
            voice.source_frame_ready = true;
        }
        if (!voice.next_source_frame_ready) {
            voice.next_source_frame_ready =
                NextSourceFrame(voice, voice.next_source_frame,
                                voice.next_source_frame_channels, pending_callbacks);
        }
        const double source_rate =
            voice.format.sampleRate == 0 ? output_sample_rate : voice.format.sampleRate;
        const double step =
            std::clamp(static_cast<double>(voice.pitch) * source_rate /
                           std::max<u32>(1, output_sample_rate),
                       1.0 / 16.0, 16.0);
        float voice_peak = 0.0f;
        for (size_t frame = 0; frame < frame_count; ++frame) {
            const size_t output_base = frame * output_channels;
            for (u32 dest = 0; dest < output_channels; ++dest) {
                float mixed = 0.0f;
                for (u32 input = 0; input < voice.source_frame_channels; ++input) {
                    const float current = voice.source_frame[input];
                    const float following =
                        voice.next_source_frame_ready &&
                                input < voice.next_source_frame_channels
                            ? voice.next_source_frame[input]
                            : current;
                    const float interpolated =
                        current + (following - current) * static_cast<float>(voice.source_phase);
                    mixed += interpolated *
                             route[dest * ORBIS_NGS2_MAX_VOICE_CHANNELS + input];
                }
                output[output_base + dest] += mixed;
                voice_peak = std::max(voice_peak, std::abs(mixed));
            }
            voice.source_phase += step;
            while (voice.source_phase >= 1.0) {
                voice.source_phase -= 1.0;
                if (!voice.next_source_frame_ready) {
                    voice.source_frame_ready = false;
                    return;
                }
                voice.source_frame = voice.next_source_frame;
                voice.source_frame_channels = voice.next_source_frame_channels;
                voice.next_source_frame_ready =
                    NextSourceFrame(voice, voice.next_source_frame,
                                    voice.next_source_frame_channels, pending_callbacks);
            }
        }
#ifdef __ANDROID__
        if (voice_peak > 0.000001f && !voice.mixed_output_reported) {
            static std::atomic<u32> mixed_trace_count{0};
            const u32 trace_index =
                mixed_trace_count.fetch_add(1, std::memory_order_relaxed);
            if (trace_index < 64) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[LSX4_NGS2_VOICE_OUTPUT] handle=0x%llx rack=0x%llx voice=%u "
                    "peak=%.6f volume=%.6f format=0x%x channels=%u rate=%u "
                    "block=%u/%zu offset=%u callbacks=%d",
                    static_cast<unsigned long long>(
                        reinterpret_cast<uintptr_t>(voice.handle)),
                    static_cast<unsigned long long>(
                        reinterpret_cast<uintptr_t>(voice.rack_handle)),
                    voice.voice_index, voice_peak, voice.ports[0].volume,
                    voice.format.waveformType, voice.format.numChannels,
                    voice.format.sampleRate, voice.block_index, voice.blocks.size(),
                    voice.block_byte_offset, voice.callback != nullptr);
            }
            voice.mixed_output_reported = true;
        }
#endif
    }

    void ApplyVoiceParam(VoiceState& voice, const OrbisNgs2VoiceParamHeader* param) {
#ifdef __ANDROID__
        TraceVoiceParam(voice.handle, voice.rack_handle, voice.voice_index, param);
#endif
        switch (NormalizeVoiceParamId(param->id)) {
        case ParamMatrixLevels: {
            if (param->size >= sizeof(OrbisNgs2VoiceMatrixLevelsParam)) {
                const auto* levels =
                    reinterpret_cast<const OrbisNgs2VoiceMatrixLevelsParam*>(param);
                if (levels->matrixId < voice.matrices.size() && levels->aLevel != nullptr) {
                    MatrixState& matrix = voice.matrices[levels->matrixId];
                    matrix.num_levels =
                        std::min(levels->numLevels,
                                 static_cast<u32>(ORBIS_NGS2_MAX_MATRIX_LEVELS));
                    for (u32 index = 0; index < matrix.num_levels; ++index) {
                        const float level = levels->aLevel[index];
                        matrix.levels[index] =
                            std::isfinite(level) && std::abs(level) <= MaxRouteGain
                                ? level
                                : 0.0f;
                    }
                    std::fill(matrix.levels.begin() + matrix.num_levels,
                              matrix.levels.end(), 0.0f);
#ifdef __ANDROID__
                    float peak = 0.0f;
                    for (u32 index = 0; index < matrix.num_levels; ++index) {
                        peak = std::max(peak, std::abs(matrix.levels[index]));
                    }
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[LSX4_NGS2_ROUTE] voice=%u matrix=%u levels=%u peak=%.6f "
                        "first=%.6f,%.6f,%.6f,%.6f",
                        voice.voice_index, levels->matrixId, matrix.num_levels, peak,
                        matrix.levels[0], matrix.levels[1], matrix.levels[2],
                        matrix.levels[3]);
#endif
                }
            }
            break;
        }
        case ParamCallback: {
            if (param->size >= sizeof(OrbisNgs2VoiceCallbackParam)) {
                const auto* callback = reinterpret_cast<const OrbisNgs2VoiceCallbackParam*>(param);
                voice.callback = callback->callbackHandler;
                voice.callback_data = callback->callbackData;
                voice.callback_flags = callback->flags;
            }
            break;
        }
        case ParamPortVolume: {
            if (param->size >= sizeof(OrbisNgs2VoicePortVolumeParam)) {
                const auto* volume = reinterpret_cast<const OrbisNgs2VoicePortVolumeParam*>(param);
                if (volume->port < voice.ports.size() && std::isfinite(volume->level)) {
#ifdef __ANDROID__
                    const float previous = voice.ports[volume->port].volume;
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[LSX4_NGS2_VOLUME] voice=%u kind=%u port=%u old=%.6f new=%.6f "
                        "raw=%08x",
                        voice.voice_index, static_cast<u32>(voice.kind), volume->port, previous,
                        volume->level, std::bit_cast<u32>(volume->level));
#endif
                    voice.ports[volume->port].volume = volume->level;
                }
            }
            break;
        }
        case ParamPortDelay: {
            if (param->size >= sizeof(OrbisNgs2VoicePortDelayParam)) {
                const auto* delay = reinterpret_cast<const OrbisNgs2VoicePortDelayParam*>(param);
                if (delay->port < voice.ports.size()) {
                    voice.ports[delay->port].delay_samples = delay->numSamples;
                }
            }
            break;
        }
        case ParamPortMatrix: {
            if (param->size >= sizeof(OrbisNgs2VoicePortMatrixParam)) {
                const auto* matrix = reinterpret_cast<const OrbisNgs2VoicePortMatrixParam*>(param);
                if (matrix->port < voice.ports.size()) {
                    voice.ports[matrix->port].matrix_id = matrix->matrixId;
                }
            }
            break;
        }
        case ParamPatch: {
            if (param->size >= sizeof(OrbisNgs2VoicePatchParam)) {
                const auto* patch = reinterpret_cast<const OrbisNgs2VoicePatchParam*>(param);
                if (patch->port < voice.ports.size()) {
                    voice.ports[patch->port].dest_handle = patch->destHandle;
                    voice.ports[patch->port].dest_input = patch->destInputId;
                }
            }
            break;
        }
        case ParamEvent: {
            if (param->size >= sizeof(OrbisNgs2VoiceEventParam)) {
                const auto* event = reinterpret_cast<const OrbisNgs2VoiceEventParam*>(param);
#ifdef __ANDROID__
                static std::atomic<u32> event_trace_count{0};
                if (event_trace_count.fetch_add(1, std::memory_order_relaxed) < 128) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[LSX4_NGS2_EVENT] handle=0x%llx rack=0x%llx voice=%u event=%u "
                        "beforeFlags=0x%x play=%d paused=%d starved=%d block=%u/%zu",
                        static_cast<unsigned long long>(
                            reinterpret_cast<uintptr_t>(voice.handle)),
                        static_cast<unsigned long long>(
                            reinterpret_cast<uintptr_t>(voice.rack_handle)),
                        voice.voice_index, event->eventId, voice.state_flags,
                        voice.play_requested, voice.paused, voice.starved,
                        voice.block_index, voice.blocks.size());
                }
#endif
                switch (static_cast<VoiceEvent>(event->eventId)) {
                case VoiceEvent::Play:
                    if (voice.play_requested && voice.state_flags != 0) {
                        voice.paused = false;
                        voice.state_flags = VoiceStateActive;
                        break;
                    }
                    voice.play_requested = true;
                    voice.paused = false;
                    voice.state_flags = VoiceStateActive;
                    voice.block_index = 0;
                    voice.source_frame_ready = false;
                    voice.next_source_frame_ready = false;
                    voice.source_phase = 0.0;
                    voice.decoded_samples = 0;
                    voice.decoded_bytes = 0;
                    voice.decoder_output_reported = false;
                    voice.decoder_failure_reported = false;
                    voice.route_output_reported = false;
                    voice.mixed_output_reported = false;
                    ResetDecoder(voice);
                    StartBlock(voice, true);
                    break;
                case VoiceEvent::Stop:
                case VoiceEvent::StopImmediate:
                    voice.play_requested = false;
                    voice.paused = false;
                    voice.starved = false;
                    voice.state_flags = 0;
                    voice.block_index = 0;
                    voice.decoded_frame_pos = 0;
                    voice.source_frame_ready = false;
                    voice.next_source_frame_ready = false;
                    voice.source_phase = 0.0;
                    break;
                case VoiceEvent::Pause:
                    if (voice.play_requested) {
                        voice.paused = true;
                        voice.state_flags = VoiceStateActive | VoiceStatePaused;
                    }
                    break;
                case VoiceEvent::Resume:
                    if (voice.play_requested || voice.kind == VoiceKind::Sampler) {
                        voice.play_requested = true;
                        voice.paused = false;
                        voice.state_flags = VoiceStateActive;
                        if (!voice.blocks.empty()) {
                            if (voice.starved && voice.block_index >= voice.blocks.size()) {
                                if (!AnyBlockDataChanged(voice)) {
                                    break;
                                }
                                voice.block_index = 0;
                                voice.decoded_samples = 0;
                                voice.decoded_bytes = 0;
                                RefreshBlockFingerprints(voice);
                            }
                            voice.starved = false;
                            if (voice.block_index >= voice.blocks.size()) {
                                voice.block_index = 0;
                            }
                            if (voice.decoded_frame_pos >= voice.decoded_frame_samples) {
                                StartBlock(voice, false);
                            }
                        } else {
                            voice.starved = false;
                        }
                    }
                    break;
                case VoiceEvent::Kill:
                    voice.play_requested = false;
                    voice.paused = false;
                    voice.starved = false;
                    voice.state_flags = 0;
                    voice.blocks.clear();
                    voice.block_data.clear();
                    voice.block_fingerprints.clear();
                    voice.data = nullptr;
                    voice.source_frame_ready = false;
                    voice.next_source_frame_ready = false;
                    voice.source_phase = 0.0;
                    ResetDecoder(voice);
                    break;
                default:
                    LOG_WARNING(Lib_Ngs2, "Ignoring unknown voice event {}", event->eventId);
                    break;
                }
            }
            break;
        }
        case ParamSamplerSetup: {
            if (param->size >= sizeof(OrbisNgs2CustomSamplerVoiceSetupParam)) {
                const auto* setup =
                    reinterpret_cast<const OrbisNgs2CustomSamplerVoiceSetupParam*>(param);
                if (voice.has_format && voice.decoder_ready &&
                    std::memcmp(&voice.format, &setup->format, sizeof(voice.format)) == 0) {
                    voice.kind = VoiceKind::Sampler;
                    break;
                }
                voice.kind = VoiceKind::Sampler;
                voice.format = setup->format;
                voice.has_format = true;
                voice.source_frame_ready = false;
                voice.next_source_frame_ready = false;
                voice.source_phase = 0.0;
                ResetDecoder(voice);
            }
            break;
        }
        case ParamSamplerBlocks: {
            if (param->size >= sizeof(OrbisNgs2CustomSamplerVoiceWaveformBlocksParam)) {
                const auto* waveform =
                    reinterpret_cast<const OrbisNgs2CustomSamplerVoiceWaveformBlocksParam*>(param);
                const u32 requested_count = std::min(waveform->numBlocks, MaxWaveformBlocks);
                const u32 count = waveform->aBlock == nullptr ? 0 : requested_count;
                const auto* incoming_data = static_cast<const u8*>(waveform->data);
                const bool unchanged =
                    voice.data == incoming_data &&
                    SameWaveformBlocks(voice.blocks, waveform->aBlock, count);
                const bool append_blocks =
                    count != 0 && !voice.blocks.empty() && (waveform->flags & 1u) != 0;
#ifdef __ANDROID__
                if (!unchanged && voice.voice_index == 0 && !voice.blocks.empty() &&
                    count != 0) {
                    static u32 block_change_count = 0;
                    if (block_change_count < 24) {
                        const OrbisNgs2WaveformBlock& previous = voice.blocks.front();
                        const OrbisNgs2WaveformBlock& incoming = waveform->aBlock[0];
                        __android_log_print(
                            ANDROID_LOG_INFO, "LSX4Native",
                            "[LSX4_NGS2_BLOCK_CHANGE] n=%u ptr=%d/%d "
                            "offset=%u/%u size=%u/%u repeats=%u/%u skip=%u/%u "
                            "samples=%u/%u reserved=%u/%u",
                            count, voice.data == incoming_data,
                            voice.blocks.data() == waveform->aBlock, previous.dataOffset,
                            incoming.dataOffset, previous.dataSize, incoming.dataSize,
                            previous.numRepeats, incoming.numRepeats, previous.numSkipSamples,
                            incoming.numSkipSamples, previous.numSamples, incoming.numSamples,
                            previous.reserved, incoming.reserved);
                        ++block_change_count;
                    }
                }
#endif
                if (unchanged && !voice.starved) {
                    break;
                }
                voice.data = incoming_data;
                if (count == 0) {
                    voice.blocks.clear();
                    voice.block_data.clear();
                    voice.block_fingerprints.clear();
                    voice.block_index = 0;
                } else if (append_blocks) {
                    if (voice.block_index != 0) {
                        const size_t consumed =
                            std::min<size_t>(voice.block_index, voice.blocks.size());
                        voice.blocks.erase(voice.blocks.begin(),
                                           voice.blocks.begin() + consumed);
                        voice.block_data.erase(voice.block_data.begin(),
                                               voice.block_data.begin() + consumed);
                        const size_t consumed_fingerprints =
                            std::min(consumed, voice.block_fingerprints.size());
                        voice.block_fingerprints.erase(
                            voice.block_fingerprints.begin(),
                            voice.block_fingerprints.begin() + consumed_fingerprints);
                        voice.block_index = 0;
                    }
                    voice.blocks.insert(voice.blocks.end(), waveform->aBlock,
                                        waveform->aBlock + count);
                    voice.block_data.insert(voice.block_data.end(), count, incoming_data);
                } else {
                    voice.blocks.assign(waveform->aBlock, waveform->aBlock + count);
                    voice.block_data.assign(count, incoming_data);
                    voice.block_index = 0;
                }
                RefreshBlockFingerprints(voice);
#ifdef __ANDROID__
                if (!voice.block_layout_reported && !voice.blocks.empty()) {
                    const OrbisNgs2WaveformBlock& block = voice.blocks.front();
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[LSX4_NGS2_BLOCK] voice=%u offset=%u size=%u repeats=%u skip=%u "
                        "samples=%u reserved=%u config=%08x frame=%d/%d source=%02x%02x%02x%02x"
                        "%02x%02x%02x%02x boundaries=%02x,%02x,%02x,%02x",
                        voice.voice_index, block.dataOffset, block.dataSize, block.numRepeats,
                        block.numSkipSamples, block.numSamples, block.reserved,
                        voice.format.configData, voice.codec.superframeSize,
                        voice.codec.framesInSuperframe,
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 1],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 2],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 3],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 4],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 5],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 6],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 7],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 96],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 192],
                        incoming_data == nullptr ? 0 : incoming_data[block.dataOffset + 288]);
                    voice.block_layout_reported = true;
                }
#endif
                if (!append_blocks || voice.starved) {
                    voice.source_frame_ready = false;
                    voice.next_source_frame_ready = false;
                    voice.source_phase = 0.0;
                    StartBlock(voice, true);
                }
                voice.decoded_samples = 0;
                voice.decoded_bytes = 0;
                if (voice.play_requested) {
                    voice.state_flags = VoiceStateActive;
                }
#ifdef __ANDROID__
                static std::atomic<u32> blocks_trace_count{0};
                if (blocks_trace_count.fetch_add(1, std::memory_order_relaxed) < 128) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[LSX4_NGS2_BLOCKS_COMMIT] handle=0x%llx rack=0x%llx voice=%u "
                        "count=%u append=%d starved=%d flags=0x%x total=%zu data=%p "
                        "volume=%.6f",
                        static_cast<unsigned long long>(
                            reinterpret_cast<uintptr_t>(voice.handle)),
                        static_cast<unsigned long long>(
                            reinterpret_cast<uintptr_t>(voice.rack_handle)),
                        voice.voice_index, count, append_blocks, voice.starved,
                        voice.state_flags, voice.blocks.size(), voice.data,
                        voice.ports[0].volume);
                }
#endif
            }
            break;
        }
        case ParamSamplerAddress: {
            if (param->size >= sizeof(OrbisNgs2CustomSamplerVoiceWaveformAddressParam)) {
                const auto* address =
                    reinterpret_cast<const OrbisNgs2CustomSamplerVoiceWaveformAddressParam*>(param);
                if (voice.data == address->from) {
                    voice.data = static_cast<const u8*>(address->to);
                }
                for (const u8*& block_data : voice.block_data) {
                    if (block_data == address->from) {
                        block_data = static_cast<const u8*>(address->to);
                    }
                }
            }
            break;
        }
        case ParamSamplerFrameOffset: {
            if (param->size >= sizeof(OrbisNgs2CustomSamplerVoiceWaveformFrameOffsetParam)) {
                const auto* offset = reinterpret_cast<
                    const OrbisNgs2CustomSamplerVoiceWaveformFrameOffsetParam*>(param);
                voice.frame_offset = offset->frameOffset;
            }
            break;
        }
        case ParamSamplerExitLoop:
            voice.exit_loop = true;
            break;
        case ParamSamplerPitch: {
            if (param->size >= sizeof(OrbisNgs2CustomSamplerVoicePitchParam)) {
                const auto* pitch =
                    reinterpret_cast<const OrbisNgs2CustomSamplerVoicePitchParam*>(param);
                if (std::isfinite(pitch->ratio) && pitch->ratio > 0.0f) {
                    voice.pitch = pitch->ratio;
                }
            }
            break;
        }
        case ParamSubmixerSetup: {
            if (param->size >= sizeof(OrbisNgs2CustomSubmixerVoiceSetupParam)) {
                const auto* setup =
                    reinterpret_cast<const OrbisNgs2CustomSubmixerVoiceSetupParam*>(param);
                voice.kind = VoiceKind::Submixer;
                voice.input_channels =
                    std::min(setup->numInputChannels,
                             static_cast<u32>(ORBIS_NGS2_MAX_VOICE_CHANNELS));
                voice.output_channels =
                    std::min(setup->numOutputChannels,
                             static_cast<u32>(ORBIS_NGS2_MAX_VOICE_CHANNELS));
            }
            break;
        }
        case ParamMasteringSetup: {
            if (param->size >= sizeof(OrbisNgs2CustomMasteringVoiceSetupParam)) {
                const auto* setup =
                    reinterpret_cast<const OrbisNgs2CustomMasteringVoiceSetupParam*>(param);
                voice.kind = VoiceKind::Mastering;
                voice.input_channels =
                    std::min(setup->numInputChannels,
                             static_cast<u32>(ORBIS_NGS2_MAX_VOICE_CHANNELS));
                voice.output_channels = voice.input_channels;
            }
            break;
        }
        case ParamMasteringOutput: {
            if (param->size >= sizeof(OrbisNgs2CustomMasteringVoiceOutputParam)) {
                const auto* output =
                    reinterpret_cast<const OrbisNgs2CustomMasteringVoiceOutputParam*>(param);
                voice.output_id = output->outputId;
            }
            break;
        }
        default:
            break;
        }
    }

    size_t ActiveVoiceCount(OrbisNgs2Handle system_handle) const {
        size_t count = 0;
        for (const auto& [voice_handle, voice] : voices_) {
            const auto rack_it = racks_.find(voice->rack_handle);
            if (rack_it != racks_.end() && rack_it->second->system_handle == system_handle &&
                voice->state_flags != 0) {
                ++count;
            }
        }
        return count;
    }

    std::mutex mutex_;
    u64 next_handle_{1};
    std::unordered_map<OrbisNgs2Handle, std::unique_ptr<SystemState>> systems_;
    std::unordered_map<OrbisNgs2Handle, std::unique_ptr<RackState>> racks_;
    std::unordered_map<OrbisNgs2Handle, std::unique_ptr<VoiceState>> voices_;
    std::vector<float> render_scratch_;
    bool reported_audio_{};
};

} // namespace

s32 CreateSystem(const OrbisNgs2SystemOption* option,
                 const OrbisNgs2ContextBufferInfo* buffer_info,
                 OrbisNgs2Handle* out_handle) {
    return Ngs2Runtime::Instance().CreateSystem(option, buffer_info, out_handle);
}

s32 DestroySystem(OrbisNgs2Handle system_handle, OrbisNgs2ContextBufferInfo* out_buffer_info) {
    return Ngs2Runtime::Instance().DestroySystem(system_handle, out_buffer_info);
}

s32 GetSystemInfo(OrbisNgs2Handle system_handle, OrbisNgs2SystemInfo* out_info,
                  size_t info_size) {
    return Ngs2Runtime::Instance().GetSystemInfo(system_handle, out_info, info_size);
}

s32 GetSystemUserData(OrbisNgs2Handle system_handle, uintptr_t* out_user_data) {
    return Ngs2Runtime::Instance().GetSystemUserData(system_handle, out_user_data);
}

s32 SetSystemUserData(OrbisNgs2Handle system_handle, uintptr_t user_data) {
    return Ngs2Runtime::Instance().SetSystemUserData(system_handle, user_data);
}

s32 SetSystemGrainSamples(OrbisNgs2Handle system_handle, u32 num_samples) {
    return Ngs2Runtime::Instance().SetSystemGrainSamples(system_handle, num_samples);
}

s32 SetSystemSampleRate(OrbisNgs2Handle system_handle, u32 sample_rate) {
    return Ngs2Runtime::Instance().SetSystemSampleRate(system_handle, sample_rate);
}

s32 CreateRack(OrbisNgs2Handle system_handle, u32 rack_id,
               const OrbisNgs2RackOption* option,
               const OrbisNgs2ContextBufferInfo* buffer_info,
               OrbisNgs2Handle* out_handle) {
    return Ngs2Runtime::Instance().CreateRack(system_handle, rack_id, option, buffer_info,
                                              out_handle);
}

s32 DestroyRack(OrbisNgs2Handle rack_handle, OrbisNgs2ContextBufferInfo* out_buffer_info) {
    return Ngs2Runtime::Instance().DestroyRack(rack_handle, out_buffer_info);
}

s32 GetRackInfo(OrbisNgs2Handle rack_handle, OrbisNgs2RackInfo* out_info, size_t info_size) {
    return Ngs2Runtime::Instance().GetRackInfo(rack_handle, out_info, info_size);
}

s32 GetRackUserData(OrbisNgs2Handle rack_handle, uintptr_t* out_user_data) {
    return Ngs2Runtime::Instance().GetRackUserData(rack_handle, out_user_data);
}

s32 SetRackUserData(OrbisNgs2Handle rack_handle, uintptr_t user_data) {
    return Ngs2Runtime::Instance().SetRackUserData(rack_handle, user_data);
}

s32 GetVoiceHandle(OrbisNgs2Handle rack_handle, u32 voice_index,
                   OrbisNgs2Handle* out_handle) {
    return Ngs2Runtime::Instance().GetVoiceHandle(rack_handle, voice_index, out_handle);
}

s32 ControlVoice(OrbisNgs2Handle voice_handle, const OrbisNgs2VoiceParamHeader* params) {
    return Ngs2Runtime::Instance().ControlVoice(voice_handle, params);
}

s32 GetVoiceOwner(OrbisNgs2Handle voice_handle, OrbisNgs2Handle* out_rack_handle,
                  u32* out_voice_index) {
    return Ngs2Runtime::Instance().GetVoiceOwner(voice_handle, out_rack_handle,
                                                 out_voice_index);
}

s32 GetVoiceState(OrbisNgs2Handle voice_handle, OrbisNgs2VoiceState* out_state,
                  size_t state_size) {
    return Ngs2Runtime::Instance().GetVoiceState(voice_handle, out_state, state_size);
}

s32 GetVoiceStateFlags(OrbisNgs2Handle voice_handle, u32* out_state_flags) {
    return Ngs2Runtime::Instance().GetVoiceStateFlags(voice_handle, out_state_flags);
}

s32 GetVoiceMatrixInfo(OrbisNgs2Handle voice_handle, u32 matrix_id,
                       OrbisNgs2VoiceMatrixInfo* out_info, size_t info_size) {
    return Ngs2Runtime::Instance().GetVoiceMatrixInfo(voice_handle, matrix_id, out_info,
                                                      info_size);
}

s32 GetVoicePortInfo(OrbisNgs2Handle voice_handle, u32 port,
                      OrbisNgs2VoicePortInfo* out_info, size_t info_size) {
    return Ngs2Runtime::Instance().GetVoicePortInfo(voice_handle, port, out_info, info_size);
}

s32 Render(OrbisNgs2Handle system_handle, const OrbisNgs2RenderBufferInfo* buffers,
           u32 num_buffers) {
    return Ngs2Runtime::Instance().Render(system_handle, buffers, num_buffers);
}

} // namespace Libraries::Ngs2::Runtime
