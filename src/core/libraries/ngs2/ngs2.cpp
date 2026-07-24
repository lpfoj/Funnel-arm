// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/guest_callback.h"
#include "core/libraries/libs.h"
#include "core/libraries/ngs2/ngs2.h"
#include "core/libraries/ngs2/ngs2_custom.h"
#include "core/libraries/ngs2/ngs2_error.h"
#include "core/libraries/ngs2/ngs2_geom.h"
#include "core/libraries/ngs2/ngs2_impl.h"
#include "core/libraries/ngs2/ngs2_pan.h"
#include "core/libraries/ngs2/ngs2_report.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

#include "libatrac9.h"

namespace Libraries::Ngs2 {

namespace {

constexpr u32 RiffTag = 0x46464952;
constexpr u32 WaveTag = 0x45564157;
constexpr u32 FormatTag = 0x20746d66;
constexpr u32 FactTag = 0x74636166;
constexpr u32 DataTag = 0x61746164;
constexpr u32 SampleTag = 0x6c706d73;
constexpr u16 WaveFormatPcm = 0x0001;
constexpr u16 WaveFormatFloat = 0x0003;
constexpr u16 WaveFormatExtensible = 0xfffe;
constexpr u32 Ngs2WaveformAtrac9 = 0x00010000;
constexpr std::array<u8, 16> Atrac9Guid{0xd2, 0x42, 0xe1, 0x47, 0xba, 0x36, 0x8d, 0x4d,
                                                0x88, 0xfc, 0x61, 0x65, 0x4f, 0x8c, 0x83, 0x6c};

template <typename T>
bool ReadWaveValue(const u8* bytes, size_t size, size_t offset, T& value) {
    if (offset > size || sizeof(T) > size - offset) {
        return false;
    }
    std::memcpy(&value, bytes + offset, sizeof(T));
    return true;
}

bool IsAtrac9Format(const OrbisNgs2WaveformFormat& format) {
    if (format.waveformType == Ngs2WaveformAtrac9) {
        return true;
    }
    const u8 first = static_cast<u8>(format.configData);
    return first == 0xfe;
}

bool GetAtrac9Info(const OrbisNgs2WaveformFormat& format, Atrac9CodecInfo& info) {
    std::array<u8, ATRAC9_CONFIG_DATA_SIZE> config{};
    std::memcpy(config.data(), &format.configData, config.size());
    void* handle = Atrac9GetHandle();
    if (handle == nullptr) {
        return false;
    }
    const int init_result = Atrac9InitDecoder(handle, config.data());
    const int info_result = init_result == 0 ? Atrac9GetCodecInfo(handle, &info) : init_result;
    Atrac9ReleaseHandle(handle);
    return init_result == 0 && info_result == 0 && info.superframeSize > 0 &&
           info.framesInSuperframe > 0 && info.frameSamples > 0;
}

s32 ParseRiffWaveform(const void* data, size_t data_size, OrbisNgs2WaveformInfo* out_info) {
    if (data == nullptr) {
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_ADDRESS;
    }
    if (out_info == nullptr) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }
    if (data_size < 12) {
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_SIZE;
    }

    const auto* bytes = static_cast<const u8*>(data);
    u32 riff = 0;
    u32 wave = 0;
    if (!ReadWaveValue(bytes, data_size, 0, riff) ||
        !ReadWaveValue(bytes, data_size, 8, wave) || riff != RiffTag || wave != WaveTag) {
        return ORBIS_NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT;
    }

    OrbisNgs2WaveformInfo parsed{};
    u16 format_tag = 0;
    u16 block_align = 0;
    u16 bits_per_sample = 0;
    u32 average_bytes_per_second = 0;
    u32 fact_samples = 0;
    u32 encoder_delay = 0;
    bool found_format = false;
    bool found_data = false;
    bool atrac9 = false;
    Atrac9CodecInfo atrac9_info{};

    size_t cursor = 12;
    while (cursor <= data_size && data_size - cursor >= 8) {
        u32 chunk_tag = 0;
        u32 chunk_size = 0;
        ReadWaveValue(bytes, data_size, cursor, chunk_tag);
        ReadWaveValue(bytes, data_size, cursor + 4, chunk_size);
        const size_t payload = cursor + 8;
        const size_t available = payload <= data_size ? data_size - payload : 0;

        if (chunk_tag == FormatTag) {
            if (chunk_size < 16 || available < std::min<size_t>(chunk_size, 16)) {
                return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_FORMAT;
            }
            u16 channels = 0;
            u32 sample_rate = 0;
            ReadWaveValue(bytes, data_size, payload, format_tag);
            ReadWaveValue(bytes, data_size, payload + 2, channels);
            ReadWaveValue(bytes, data_size, payload + 4, sample_rate);
            ReadWaveValue(bytes, data_size, payload + 8, average_bytes_per_second);
            ReadWaveValue(bytes, data_size, payload + 12, block_align);
            ReadWaveValue(bytes, data_size, payload + 14, bits_per_sample);
            if (channels == 0 || channels > ORBIS_NGS2_MAX_VOICE_CHANNELS) {
                return ORBIS_NGS2_ERROR_INVALID_NUM_CHANNELS;
            }
            if (sample_rate == 0) {
                return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_SAMPLE_RATE;
            }

            parsed.format.numChannels = channels;
            parsed.format.sampleRate = sample_rate;
            parsed.format.waveformType = format_tag;
            parsed.format.configData = static_cast<u32>(bits_per_sample) |
                                       (static_cast<u32>(block_align) << 16);

            if (format_tag == WaveFormatExtensible && chunk_size >= 48 && available >= 48) {
                atrac9 = std::equal(Atrac9Guid.begin(), Atrac9Guid.end(), bytes + payload + 24);
                if (atrac9) {
                    parsed.format.waveformType = Ngs2WaveformAtrac9;
                    std::memcpy(&parsed.format.configData, bytes + payload + 44,
                                sizeof(parsed.format.configData));
                    if (!GetAtrac9Info(parsed.format, atrac9_info)) {
                        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_CONFIG;
                    }
                    parsed.audioUnitSize = static_cast<u32>(atrac9_info.superframeSize);
                    parsed.numAudioUnitSamples = static_cast<u32>(
                        atrac9_info.framesInSuperframe * atrac9_info.frameSamples);
                    parsed.numAudioUnitPerFrame =
                        static_cast<u32>(atrac9_info.framesInSuperframe);
                    parsed.audioFrameSize = static_cast<u32>(
                        atrac9_info.superframeSize / atrac9_info.framesInSuperframe);
                    parsed.numAudioFrameSamples = static_cast<u32>(atrac9_info.frameSamples);
                }
            }

            if (!atrac9 && format_tag != WaveFormatPcm && format_tag != WaveFormatFloat) {
                return ORBIS_NGS2_ERROR_UNKNOWN_WAVEFORM_FORMAT;
            }
            if (!atrac9) {
                const u32 frame_size = block_align != 0
                                           ? block_align
                                           : static_cast<u32>(channels) *
                                                 std::max<u32>(1, bits_per_sample / 8);
                parsed.audioUnitSize = frame_size;
                parsed.numAudioUnitSamples = 1;
                parsed.numAudioUnitPerFrame = 1;
                parsed.audioFrameSize = frame_size;
                parsed.numAudioFrameSamples = 1;
            }
            found_format = true;
        } else if (chunk_tag == FactTag && chunk_size >= 4 && available >= 4) {
            ReadWaveValue(bytes, data_size, payload, fact_samples);
            if (chunk_size >= 8 && available >= 8) {
                ReadWaveValue(bytes, data_size, payload + 4, encoder_delay);
            }
        } else if (chunk_tag == SampleTag && chunk_size >= 60 && available >= 60) {
            u32 loop_count = 0;
            ReadWaveValue(bytes, data_size, payload + 28, loop_count);
            if (loop_count != 0) {
                ReadWaveValue(bytes, data_size, payload + 44, parsed.loopBeginPosition);
                ReadWaveValue(bytes, data_size, payload + 48, parsed.loopEndPosition);
                if (parsed.loopEndPosition != std::numeric_limits<u32>::max()) {
                    ++parsed.loopEndPosition;
                }
            }
        } else if (chunk_tag == DataTag) {
            if (payload > std::numeric_limits<u32>::max()) {
                return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_SIZE;
            }
            parsed.dataOffset = static_cast<u32>(payload);
            parsed.dataSize = chunk_size;
            found_data = true;
            break;
        }

        if (chunk_size > std::numeric_limits<size_t>::max() - payload) {
            return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_SIZE;
        }
        const size_t next = payload + chunk_size + (chunk_size & 1u);
        if (next <= cursor || next > data_size) {
            break;
        }
        cursor = next;
    }

    if (!found_format || !found_data || parsed.dataSize == 0) {
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_DATA;
    }

    parsed.numDelaySamples = encoder_delay;
    if (fact_samples != 0) {
        parsed.numSamples = fact_samples;
    } else if (parsed.audioUnitSize != 0 && parsed.numAudioUnitSamples != 0) {
        const u64 units = parsed.dataSize / parsed.audioUnitSize;
        parsed.numSamples = static_cast<u32>(std::min<u64>(
            units * parsed.numAudioUnitSamples, std::numeric_limits<u32>::max()));
    } else if (average_bytes_per_second != 0) {
        const u64 samples = static_cast<u64>(parsed.dataSize) * parsed.format.sampleRate /
                            average_bytes_per_second;
        parsed.numSamples =
            static_cast<u32>(std::min<u64>(samples, std::numeric_limits<u32>::max()));
    }
    if (parsed.numSamples == 0) {
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_DATA;
    }

    parsed.numBlocks = 1;
    parsed.aBlock[0].dataOffset = 0;
    parsed.aBlock[0].dataSize = parsed.dataSize;
    parsed.aBlock[0].numSkipSamples = encoder_delay;
    parsed.aBlock[0].numSamples = parsed.numSamples;
    *out_info = parsed;
    return ORBIS_OK;
}

}


s32 PS4_SYSV_ABI sceNgs2CalcWaveformBlock(const OrbisNgs2WaveformFormat* format, u32 samplePos,
                                          u32 numSamples, OrbisNgs2WaveformBlock* outBlock) {
    if (format == nullptr) {
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_FORMAT;
    }
    if (outBlock == nullptr) {
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_BLOCK_ADDRESS;
    }
    if (format->numChannels == 0 || format->numChannels > ORBIS_NGS2_MAX_VOICE_CHANNELS) {
        return ORBIS_NGS2_ERROR_INVALID_NUM_CHANNELS;
    }

    OrbisNgs2WaveformBlock block{};
    if (IsAtrac9Format(*format)) {
        Atrac9CodecInfo info{};
        if (!GetAtrac9Info(*format, info)) {
            return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_CONFIG;
        }
        const u64 unit_samples =
            static_cast<u64>(info.framesInSuperframe) * info.frameSamples;
        const u64 first_unit = samplePos / unit_samples;
        const u64 skip = samplePos % unit_samples;
        const u64 requested = static_cast<u64>(numSamples) + skip;
        const u64 units = requested == 0 ? 0 : (requested + unit_samples - 1) / unit_samples;
        const u64 offset = first_unit * static_cast<u64>(info.superframeSize);
        const u64 size = units * static_cast<u64>(info.superframeSize);
        if (offset > std::numeric_limits<u32>::max() || size > std::numeric_limits<u32>::max()) {
            return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_SIZE;
        }
        block.dataOffset = static_cast<u32>(offset);
        block.dataSize = static_cast<u32>(size);
        block.numSkipSamples = static_cast<u32>(skip);
        block.numSamples = numSamples;
    } else {
        const u32 block_align = format->configData >> 16;
        const u32 bytes_per_frame = block_align != 0 ? block_align : format->numChannels * 2;
        const u64 offset = static_cast<u64>(samplePos) * bytes_per_frame;
        const u64 size = static_cast<u64>(numSamples) * bytes_per_frame;
        if (offset > std::numeric_limits<u32>::max() || size > std::numeric_limits<u32>::max()) {
            return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_SIZE;
        }
        block.dataOffset = static_cast<u32>(offset);
        block.dataSize = static_cast<u32>(size);
        block.numSamples = numSamples;
    }
    *outBlock = block;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2GetWaveformFrameInfo(const OrbisNgs2WaveformFormat* format,
                                             u32* outFrameSize, u32* outNumFrameSamples,
                                             u32* outUnitsPerFrame, u32* outNumDelaySamples) {
    if (format == nullptr) {
        return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_FORMAT;
    }
    if (outFrameSize == nullptr && outNumFrameSamples == nullptr && outUnitsPerFrame == nullptr &&
        outNumDelaySamples == nullptr) {
        return ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
    }

    u32 frame_size = 0;
    u32 frame_samples = 0;
    u32 units_per_frame = 0;
    u32 delay_samples = 0;
    if (IsAtrac9Format(*format)) {
        Atrac9CodecInfo info{};
        if (!GetAtrac9Info(*format, info)) {
            return ORBIS_NGS2_ERROR_INVALID_WAVEFORM_CONFIG;
        }
        frame_size = static_cast<u32>(info.superframeSize / info.framesInSuperframe);
        frame_samples = static_cast<u32>(info.frameSamples);
        units_per_frame = static_cast<u32>(info.framesInSuperframe);
        delay_samples = format->frameOffset;
    } else {
        const u32 block_align = format->configData >> 16;
        frame_size = block_align != 0 ? block_align : format->numChannels * 2;
        frame_samples = 1;
        units_per_frame = 1;
    }
    if (outFrameSize != nullptr) {
        *outFrameSize = frame_size;
    }
    if (outNumFrameSamples != nullptr) {
        *outNumFrameSamples = frame_samples;
    }
    if (outUnitsPerFrame != nullptr) {
        *outUnitsPerFrame = units_per_frame;
    }
    if (outNumDelaySamples != nullptr) {
        *outNumDelaySamples = delay_samples;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2ParseWaveformData(const void* data, size_t dataSize,
                                          OrbisNgs2WaveformInfo* outInfo) {
    return ParseRiffWaveform(data, dataSize, outInfo);
}

s32 PS4_SYSV_ABI sceNgs2ParseWaveformFile(const char* path, u64 offset,
                                          OrbisNgs2WaveformInfo* outInfo) {
    LOG_ERROR(Lib_Ngs2, "path = {}, offset = {}", path, offset);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2ParseWaveformUser(OrbisNgs2ParseReadHandler handler, uintptr_t userData,
                                          OrbisNgs2WaveformInfo* outInfo) {
    LOG_ERROR(Lib_Ngs2, "userData = {}", userData);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackCreate(OrbisNgs2Handle systemHandle, u32 rackId,
                                   const OrbisNgs2RackOption* option,
                                   const OrbisNgs2ContextBufferInfo* bufferInfo,
                                   OrbisNgs2Handle* outHandle) {
    LOG_ERROR(Lib_Ngs2, "rackId = {}", rackId);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackCreateWithAllocator(OrbisNgs2Handle systemHandle, u32 rackId,
                                                const OrbisNgs2RackOption* option,
                                                const OrbisNgs2BufferAllocator* allocator,
                                                OrbisNgs2Handle* outHandle) {
    LOG_ERROR(Lib_Ngs2, "rackId = {}", rackId);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackDestroy(OrbisNgs2Handle rackHandle,
                                    OrbisNgs2ContextBufferInfo* outBufferInfo) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackGetInfo(OrbisNgs2Handle rackHandle, OrbisNgs2RackInfo* outInfo,
                                    size_t infoSize) {
    LOG_ERROR(Lib_Ngs2, "infoSize = {}", infoSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackGetUserData(OrbisNgs2Handle rackHandle, uintptr_t* outUserData) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackGetVoiceHandle(OrbisNgs2Handle rackHandle, u32 voiceIndex,
                                           OrbisNgs2Handle* outHandle) {
    LOG_DEBUG(Lib_Ngs2, "(STUBBED) voiceIndex = {}", voiceIndex);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackLock(OrbisNgs2Handle rackHandle) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackQueryBufferSize(u32 rackId, const OrbisNgs2RackOption* option,
                                            OrbisNgs2ContextBufferInfo* outBufferInfo) {
    LOG_ERROR(Lib_Ngs2, "rackId = {}", rackId);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackSetUserData(OrbisNgs2Handle rackHandle, uintptr_t userData) {
    LOG_ERROR(Lib_Ngs2, "userData = {}", userData);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2RackUnlock(OrbisNgs2Handle rackHandle) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemCreate(const OrbisNgs2SystemOption* option,
                                     const OrbisNgs2ContextBufferInfo* bufferInfo,
                                     OrbisNgs2Handle* outHandle) {
    s32 result;
    OrbisNgs2ContextBufferInfo localInfo;
    if (!bufferInfo || !outHandle) {
        if (!bufferInfo) {
            result = ORBIS_NGS2_ERROR_INVALID_BUFFER_INFO;
            LOG_ERROR(Lib_Ngs2, "Invalid system buffer info {}", (void*)bufferInfo);
        } else {
            result = ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
            LOG_ERROR(Lib_Ngs2, "Invalid system handle address {}", (void*)outHandle);
        }

    } else {
        localInfo.hostBuffer = bufferInfo->hostBuffer;
        localInfo.hostBufferSize = bufferInfo->hostBufferSize;
        for (int i = 0; i < 5; i++) {
            localInfo.reserved[i] = bufferInfo->reserved[i];
        }
        localInfo.userData = bufferInfo->userData;

        result = SystemSetup(option, &localInfo, 0, outHandle);
    }


    LOG_INFO(Lib_Ngs2, "called");
    return result;
}

s32 PS4_SYSV_ABI sceNgs2SystemCreateWithAllocator(const OrbisNgs2SystemOption* option,
                                                  const OrbisNgs2BufferAllocator* allocator,
                                                  OrbisNgs2Handle* outHandle) {
    s32 result;
    if (allocator && allocator->allocHandler != 0) {
        OrbisNgs2BufferAllocHandler hostAlloc = allocator->allocHandler;
        if (outHandle) {
            OrbisNgs2BufferFreeHandler hostFree = allocator->freeHandler;
            OrbisNgs2ContextBufferInfo bufferInfo{};
            result = SystemSetup(option, &bufferInfo, 0, 0);
            if (result >= 0) {
                bufferInfo.userData = allocator->userData;
                u64 callback_result = 0;
                const int callback_bridge = Libraries::RunGuestCallback4(
                    reinterpret_cast<void*>(hostAlloc),
                    reinterpret_cast<u64>(&bufferInfo), 0, 0, 0, 1, &callback_result,
                    "ngs2_buffer_alloc");
                if (callback_bridge > 0) {
                    result = static_cast<s32>(callback_result);
                } else if (callback_bridge < 0) {
                    result = ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR;
                } else {
                    result = hostAlloc(&bufferInfo);
                }
                if (result >= 0) {
                    OrbisNgs2Handle* handleCopy = outHandle;
                    result = SystemSetup(option, &bufferInfo, hostFree, handleCopy);
                    if (result < 0) {
                        if (hostFree) {
                            callback_result = 0;
                            const int free_bridge = Libraries::RunGuestCallback4(
                                reinterpret_cast<void*>(hostFree),
                                reinterpret_cast<u64>(&bufferInfo), 0, 0, 0, 1,
                                &callback_result, "ngs2_buffer_free_rollback");
                            if (free_bridge == 0) {
                                hostFree(&bufferInfo);
                            }
                        }
                    }
                }
            }
        } else {
            result = ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
            LOG_ERROR(Lib_Ngs2, "Invalid system handle address {}", (void*)outHandle);
        }
    } else {
        result = ORBIS_NGS2_ERROR_INVALID_BUFFER_ALLOCATOR;
        LOG_ERROR(Lib_Ngs2, "Invalid system buffer allocator {}", (void*)allocator);
    }
    LOG_INFO(Lib_Ngs2, "called");
    return result;
}

s32 PS4_SYSV_ABI sceNgs2SystemDestroy(OrbisNgs2Handle systemHandle,
                                      OrbisNgs2ContextBufferInfo* outBufferInfo) {
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    LOG_INFO(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemEnumHandles(OrbisNgs2Handle* aOutHandle, u32 maxHandles) {
    LOG_ERROR(Lib_Ngs2, "maxHandles = {}", maxHandles);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemEnumRackHandles(OrbisNgs2Handle systemHandle,
                                              OrbisNgs2Handle* aOutHandle, u32 maxHandles) {
    LOG_ERROR(Lib_Ngs2, "maxHandles = {}", maxHandles);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemGetInfo(OrbisNgs2Handle rackHandle, OrbisNgs2SystemInfo* outInfo,
                                      size_t infoSize) {
    LOG_ERROR(Lib_Ngs2, "infoSize = {}", infoSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemGetUserData(OrbisNgs2Handle systemHandle, uintptr_t* outUserData) {
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemLock(OrbisNgs2Handle systemHandle) {
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemQueryBufferSize(const OrbisNgs2SystemOption* option,
                                              OrbisNgs2ContextBufferInfo* outBufferInfo) {
    s32 result;
    if (outBufferInfo) {
        result = SystemSetup(option, outBufferInfo, 0, 0);
        LOG_INFO(Lib_Ngs2, "called");
    } else {
        result = ORBIS_NGS2_ERROR_INVALID_OUT_ADDRESS;
        LOG_ERROR(Lib_Ngs2, "Invalid system buffer info {}", (void*)outBufferInfo);
    }

    return result;
}

s32 PS4_SYSV_ABI sceNgs2SystemRender(OrbisNgs2Handle systemHandle,
                                     const OrbisNgs2RenderBufferInfo* aBufferInfo,
                                     u32 numBufferInfo) {
    LOG_DEBUG(Lib_Ngs2, "(STUBBED) numBufferInfo = {}", numBufferInfo);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    return ORBIS_OK;
}

static s32 PS4_SYSV_ABI sceNgs2SystemResetOption(OrbisNgs2SystemOption* outOption) {
    static const OrbisNgs2SystemOption option = {
        sizeof(OrbisNgs2SystemOption), "", 0, 512, 256, 48000, {0}};

    if (!outOption) {
        LOG_ERROR(Lib_Ngs2, "Invalid system option address {}", (void*)outOption);
        return ORBIS_NGS2_ERROR_INVALID_OPTION_ADDRESS;
    }
    *outOption = option;

    LOG_INFO(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemSetGrainSamples(OrbisNgs2Handle systemHandle, u32 numSamples) {
    LOG_ERROR(Lib_Ngs2, "numSamples = {}", numSamples);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemSetSampleRate(OrbisNgs2Handle systemHandle, u32 sampleRate) {
    LOG_ERROR(Lib_Ngs2, "sampleRate = {}", sampleRate);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemSetUserData(OrbisNgs2Handle systemHandle, uintptr_t userData) {
    LOG_ERROR(Lib_Ngs2, "userData = {}", userData);
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2SystemUnlock(OrbisNgs2Handle systemHandle) {
    if (!systemHandle) {
        LOG_ERROR(Lib_Ngs2, "systemHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_SYSTEM_HANDLE;
    }
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceControl(OrbisNgs2Handle voiceHandle,
                                     const OrbisNgs2VoiceParamHeader* paramList) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetMatrixInfo(OrbisNgs2Handle voiceHandle, u32 matrixId,
                                           OrbisNgs2VoiceMatrixInfo* outInfo, size_t outInfoSize) {
    LOG_ERROR(Lib_Ngs2, "matrixId = {}, outInfoSize = {}", matrixId, outInfoSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetOwner(OrbisNgs2Handle voiceHandle, OrbisNgs2Handle* outRackHandle,
                                      u32* outVoiceId) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetPortInfo(OrbisNgs2Handle voiceHandle, u32 port,
                                         OrbisNgs2VoicePortInfo* outInfo, size_t outInfoSize) {
    LOG_ERROR(Lib_Ngs2, "port = {}, outInfoSize = {}", port, outInfoSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetState(OrbisNgs2Handle voiceHandle, OrbisNgs2VoiceState* outState,
                                      size_t stateSize) {
    LOG_ERROR(Lib_Ngs2, "stateSize = {}", stateSize);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2VoiceGetStateFlags(OrbisNgs2Handle voiceHandle, u32* outStateFlags) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}


s32 PS4_SYSV_ABI sceNgs2CustomRackGetModuleInfo(OrbisNgs2Handle rackHandle, u32 moduleIndex,
                                                OrbisNgs2CustomModuleInfo* outInfo,
                                                size_t infoSize) {
    LOG_ERROR(Lib_Ngs2, "moduleIndex = {}, infoSize = {}", moduleIndex, infoSize);
    return ORBIS_OK;
}


s32 PS4_SYSV_ABI sceNgs2GeomResetListenerParam(OrbisNgs2GeomListenerParam* outListenerParam) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2GeomResetSourceParam(OrbisNgs2GeomSourceParam* outSourceParam) {
    LOG_ERROR(Lib_Ngs2, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2GeomCalcListener(const OrbisNgs2GeomListenerParam* param,
                                         OrbisNgs2GeomListenerWork* outWork, u32 flags) {
    LOG_ERROR(Lib_Ngs2, "flags = {}", flags);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2GeomApply(const OrbisNgs2GeomListenerWork* listener,
                                  const OrbisNgs2GeomSourceParam* source,
                                  OrbisNgs2GeomAttribute* outAttrib, u32 flags) {
    LOG_ERROR(Lib_Ngs2, "flags = {}", flags);
    return ORBIS_OK;
}


s32 PS4_SYSV_ABI sceNgs2PanInit(OrbisNgs2PanWork* work, const float* aSpeakerAngle, float unitAngle,
                                u32 numSpeakers) {
    LOG_ERROR(Lib_Ngs2, "unitAngle = {}, numSpeakers = {}", unitAngle, numSpeakers);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2PanGetVolumeMatrix(OrbisNgs2PanWork* work, const OrbisNgs2PanParam* aParam,
                                           u32 numParams, u32 matrixFormat,
                                           float* outVolumeMatrix) {
    LOG_ERROR(Lib_Ngs2, "numParams = {}, matrixFormat = {}", numParams, matrixFormat);
    return ORBIS_OK;
}


s32 PS4_SYSV_ABI sceNgs2ReportRegisterHandler(u32 reportType, OrbisNgs2ReportHandler handler,
                                              uintptr_t userData, OrbisNgs2Handle* outHandle) {
    LOG_INFO(Lib_Ngs2, "reportType = {}, userData = {}", reportType, userData);
    if (!handler) {
        LOG_ERROR(Lib_Ngs2, "handler is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_REPORT_HANDLE;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceNgs2ReportUnregisterHandler(OrbisNgs2Handle reportHandle) {
    if (!reportHandle) {
        LOG_ERROR(Lib_Ngs2, "reportHandle is nullptr");
        return ORBIS_NGS2_ERROR_INVALID_REPORT_HANDLE;
    }
    LOG_INFO(Lib_Ngs2, "called");
    return ORBIS_OK;
}


int PS4_SYSV_ABI sceNgs2FftInit() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2FftProcess() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2FftQuerySize() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2JobSchedulerResetOption() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2ModuleArrayEnumItems() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2ModuleEnumConfigs() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2ModuleQueueEnumItems() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2RackQueryInfo() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2RackRunCommands() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2SystemQueryInfo() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2SystemRunCommands() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2SystemSetLoudThreshold() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamCreate() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamCreateWithAllocator() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamDestroy() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamQueryBufferSize() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamQueryInfo() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamResetOption() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2StreamRunCommands() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2VoiceQueryInfo() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceNgs2VoiceRunCommands() {
    LOG_ERROR(Lib_Ngs2, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("3pCNbVM11UA", "libSceNgs2", 1, "libSceNgs2", sceNgs2CalcWaveformBlock);
    LIB_FUNCTION("6qN1zaEZuN0", "libSceNgs2", 1, "libSceNgs2", sceNgs2CustomRackGetModuleInfo);
    LIB_FUNCTION("Kg1MA5j7KFk", "libSceNgs2", 1, "libSceNgs2", sceNgs2FftInit);
    LIB_FUNCTION("D8eCqBxSojA", "libSceNgs2", 1, "libSceNgs2", sceNgs2FftProcess);
    LIB_FUNCTION("-YNfTO6KOMY", "libSceNgs2", 1, "libSceNgs2", sceNgs2FftQuerySize);
    LIB_FUNCTION("eF8yRCC6W64", "libSceNgs2", 1, "libSceNgs2", sceNgs2GeomApply);
    LIB_FUNCTION("1WsleK-MTkE", "libSceNgs2", 1, "libSceNgs2", sceNgs2GeomCalcListener);
    LIB_FUNCTION("7Lcfo8SmpsU", "libSceNgs2", 1, "libSceNgs2", sceNgs2GeomResetListenerParam);
    LIB_FUNCTION("0lbbayqDNoE", "libSceNgs2", 1, "libSceNgs2", sceNgs2GeomResetSourceParam);
    LIB_FUNCTION("ekGJmmoc8j4", "libSceNgs2", 1, "libSceNgs2", sceNgs2GetWaveformFrameInfo);
    LIB_FUNCTION("BcoPfWfpvVI", "libSceNgs2", 1, "libSceNgs2", sceNgs2JobSchedulerResetOption);
    LIB_FUNCTION("EEemGEQCjO8", "libSceNgs2", 1, "libSceNgs2", sceNgs2ModuleArrayEnumItems);
    LIB_FUNCTION("TaoNtmMKkXQ", "libSceNgs2", 1, "libSceNgs2", sceNgs2ModuleEnumConfigs);
    LIB_FUNCTION("ve6bZi+1sYQ", "libSceNgs2", 1, "libSceNgs2", sceNgs2ModuleQueueEnumItems);
    LIB_FUNCTION("gbMKV+8Enuo", "libSceNgs2", 1, "libSceNgs2", sceNgs2PanGetVolumeMatrix);
    LIB_FUNCTION("xa8oL9dmXkM", "libSceNgs2", 1, "libSceNgs2", sceNgs2PanInit);
    LIB_FUNCTION("hyVLT2VlOYk", "libSceNgs2", 1, "libSceNgs2", sceNgs2ParseWaveformData);
    LIB_FUNCTION("iprCTXPVWMI", "libSceNgs2", 1, "libSceNgs2", sceNgs2ParseWaveformFile);
    LIB_FUNCTION("t9T0QM17Kvo", "libSceNgs2", 1, "libSceNgs2", sceNgs2ParseWaveformUser);
    LIB_FUNCTION("cLV4aiT9JpA", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackCreate);
    LIB_FUNCTION("U546k6orxQo", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackCreateWithAllocator);
    LIB_FUNCTION("lCqD7oycmIM", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackDestroy);
    LIB_FUNCTION("M4LYATRhRUE", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackGetInfo);
    LIB_FUNCTION("Mn4XNDg03XY", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackGetUserData);
    LIB_FUNCTION("MwmHz8pAdAo", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackGetVoiceHandle);
    LIB_FUNCTION("MzTa7VLjogY", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackLock);
    LIB_FUNCTION("0eFLVCfWVds", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackQueryBufferSize);
    LIB_FUNCTION("TZqb8E-j3dY", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackQueryInfo);
    LIB_FUNCTION("MI2VmBx2RbM", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackRunCommands);
    LIB_FUNCTION("JNTMIaBIbV4", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackSetUserData);
    LIB_FUNCTION("++YZ7P9e87U", "libSceNgs2", 1, "libSceNgs2", sceNgs2RackUnlock);
    LIB_FUNCTION("uBIN24Tv2MI", "libSceNgs2", 1, "libSceNgs2", sceNgs2ReportRegisterHandler);
    LIB_FUNCTION("nPzb7Ly-VjE", "libSceNgs2", 1, "libSceNgs2", sceNgs2ReportUnregisterHandler);
    LIB_FUNCTION("koBbCMvOKWw", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemCreate);
    LIB_FUNCTION("mPYgU4oYpuY", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemCreateWithAllocator);
    LIB_FUNCTION("u-WrYDaJA3k", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemDestroy);
    LIB_FUNCTION("vubFP0T6MP0", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemEnumHandles);
    LIB_FUNCTION("U-+7HsswcIs", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemEnumRackHandles);
    LIB_FUNCTION("vU7TQ62pItw", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemGetInfo);
    LIB_FUNCTION("4lFaRxd-aLs", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemGetUserData);
    LIB_FUNCTION("gThZqM5PYlQ", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemLock);
    LIB_FUNCTION("pgFAiLR5qT4", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemQueryBufferSize);
    LIB_FUNCTION("3oIK7y7O4k0", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemQueryInfo)
    LIB_FUNCTION("i0VnXM-C9fc", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemRender);
    LIB_FUNCTION("AQkj7C0f3PY", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemResetOption);
    LIB_FUNCTION("gXiormHoZZ4", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemRunCommands);
    LIB_FUNCTION("l4Q2dWEH6UM", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemSetGrainSamples);
    LIB_FUNCTION("Wdlx0ZFTV9s", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemSetLoudThreshold);
    LIB_FUNCTION("-tbc2SxQD60", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemSetSampleRate);
    LIB_FUNCTION("GZB2v0XnG0k", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemSetUserData);
    LIB_FUNCTION("JXRC5n0RQls", "libSceNgs2", 1, "libSceNgs2", sceNgs2SystemUnlock);
    LIB_FUNCTION("sU2St3agdjg", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamCreate);
    LIB_FUNCTION("I+RLwaauggA", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamCreateWithAllocator);
    LIB_FUNCTION("bfoMXnTRtwE", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamDestroy);
    LIB_FUNCTION("dxulc33msHM", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamQueryBufferSize);
    LIB_FUNCTION("rfw6ufRsmow", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamQueryInfo);
    LIB_FUNCTION("q+2W8YdK0F8", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamResetOption);
    LIB_FUNCTION("qQHCi9pjDps", "libSceNgs2", 1, "libSceNgs2", sceNgs2StreamRunCommands);
    LIB_FUNCTION("uu94irFOGpA", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceControl);
    LIB_FUNCTION("jjBVvPN9964", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetMatrixInfo);
    LIB_FUNCTION("W-Z8wWMBnhk", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetOwner);
    LIB_FUNCTION("WCayTgob7-o", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetPortInfo);
    LIB_FUNCTION("-TOuuAQ-buE", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetState);
    LIB_FUNCTION("rEh728kXk3w", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceGetStateFlags);
    LIB_FUNCTION("9eic4AmjGVI", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceQueryInfo);
    LIB_FUNCTION("AbYvTOZ8Pts", "libSceNgs2", 1, "libSceNgs2", sceNgs2VoiceRunCommands);
};

}
