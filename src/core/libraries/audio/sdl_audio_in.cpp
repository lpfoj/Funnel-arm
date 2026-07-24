// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <SDL3/SDL.h>
#include <common/config.h>
#include <common/logging/log.h>
#include "audioin.h"
#include "audioin_backend.h"

namespace Libraries::AudioIn {

class SDLInPortBackend : public PortInBackend {
public:
    explicit SDLInPortBackend(const PortIn& port) : port(port) {
        SDL_AudioFormat sampleFormat = SDL_AUDIO_S16;

        SDL_AudioSpec fmt;
        SDL_zero(fmt);
        fmt.format = sampleFormat;
        fmt.channels = static_cast<Uint8>(port.channels_num);
        fmt.freq = static_cast<int>(port.freq);

        uint32_t devId = 0;
#ifdef __ANDROID__
        nullDevice = true;
        LOG_INFO(Lib_AudioIn, "Using silent audio input backend on Android");
#else
        std::string micDevStr = Config::getMicDevice();
        if (micDevStr == "None") {
            nullDevice = true;
            LOG_INFO(Lib_AudioIn, "Audio input disabled by configuration");
        } else if (micDevStr == "Default Device") {
            devId = SDL_AUDIO_DEVICE_DEFAULT_RECORDING;
            LOG_INFO(Lib_AudioIn, "Using default audio input device");
        } else {
            try {
                devId = static_cast<uint32_t>(std::stoul(micDevStr));
                LOG_INFO(Lib_AudioIn, "Using audio input device ID: {}", devId);
            } catch (const std::exception& e) {
                nullDevice = true;
                LOG_WARNING(Lib_AudioIn, "Invalid device ID '{}', disabling input", micDevStr);
            }
        }
#endif

        if (!nullDevice) {
            stream = SDL_OpenAudioDeviceStream(devId, &fmt, nullptr, nullptr);
            if (stream) {
                if (SDL_ResumeAudioStreamDevice(stream)) {
                    LOG_INFO(Lib_AudioIn, "Audio input opened: {} Hz, {} channels, format {}",
                             port.freq, port.channels_num, static_cast<u32>(port.format));
                } else {
                    SDL_DestroyAudioStream(stream);
                    stream = nullptr;
                    LOG_ERROR(Lib_AudioIn, "Failed to resume audio input stream");
                }
            } else {
                LOG_ERROR(Lib_AudioIn, "Failed to open audio input device: {}", SDL_GetError());
            }
        }

        if (!stream) {
            const size_t bufferSize = port.samples_num * port.sample_size * port.channels_num;
            internal_buffer = std::malloc(bufferSize);
            if (internal_buffer) {
                std::memset(internal_buffer, 0, bufferSize);
                LOG_INFO(Lib_AudioIn, "Created null audio input buffer of {} bytes", bufferSize);
            }
        }
    }

    ~SDLInPortBackend() override {
        {
            std::scoped_lock lock{capture_clock_mutex};
            capture_clock_stopping = true;
        }
        capture_clock_event.notify_all();
        if (stream) {
            SDL_DestroyAudioStream(stream);
        }
        if (internal_buffer) {
            std::free(internal_buffer);
            internal_buffer = nullptr;
        }
    }

    int Read(void* out_buffer) override {
        const int bytesToRead = port.samples_num * port.sample_size * port.channels_num;

        if (stream) {
            int attempts = 0;
            while (SDL_GetAudioStreamAvailable(stream) < bytesToRead) {
                SDL_Delay(1);
                if (++attempts > 1000) {
                    return 0;
                }
            }

            const int bytesRead = SDL_GetAudioStreamData(stream, out_buffer, bytesToRead);
            if (bytesRead < 0) {
                LOG_ERROR(Lib_AudioIn, "Audio input read error: {}", SDL_GetError());
                return 0;
            }

            const int framesRead = bytesRead / (port.sample_size * port.channels_num);
            return framesRead;
        } else if (internal_buffer) {
            const auto packet_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(
                    static_cast<double>(port.samples_num) / static_cast<double>(port.freq)));
            {
                std::unique_lock lock{capture_clock_mutex};
                const auto now = std::chrono::steady_clock::now();
                if (next_capture_packet.time_since_epoch().count() == 0 ||
                    now > next_capture_packet + packet_period * 4) {
                    next_capture_packet = now + packet_period;
                }
                capture_clock_event.wait_until(lock, next_capture_packet, [&] {
                    return capture_clock_stopping ||
                           std::chrono::steady_clock::now() >= next_capture_packet;
                });
                if (capture_clock_stopping) {
                    return 0;
                }
                next_capture_packet += packet_period;
            }
            std::memcpy(out_buffer, internal_buffer, bytesToRead);
            return port.samples_num;
        } else {
            return 0;
        }
    }

    void Clear() override {
        if (stream) {
            SDL_ClearAudioStream(stream);
        }
    }
    bool IsAvailable() override {
        if (nullDevice) {
            return false;
        } else {
            return true;
        }
    }

private:
    const PortIn& port;
    SDL_AudioStream* stream = nullptr;
    void* internal_buffer = nullptr;
    bool nullDevice = false;
    std::mutex capture_clock_mutex;
    std::condition_variable capture_clock_event;
    std::chrono::steady_clock::time_point next_capture_packet{};
    bool capture_clock_stopping = false;
};

std::unique_ptr<PortInBackend> SDLAudioIn::Open(PortIn& port) {
    return std::make_unique<SDLInPortBackend>(port);
}

}
