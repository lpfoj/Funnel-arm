// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

namespace Common {

class ContentFingerprint64 {
public:
    explicit ContentFingerprint64(const std::uint64_t domain) noexcept
        : state_{Avalanche(kInitialState ^ domain)}, domain_{domain} {}

    void Update(const std::span<const std::uint8_t> bytes) noexcept {
        if (bytes.empty()) {
            return;
        }

        size_ += bytes.size();
        std::size_t offset = 0;
        if (tail_size_ != 0) {
            const std::size_t copied =
                std::min(bytes.size(), tail_.size() - tail_size_);
            std::memcpy(tail_.data() + tail_size_, bytes.data(), copied);
            tail_size_ += copied;
            offset += copied;
            if (tail_size_ == tail_.size()) {
                MixWord(ReadLittleEndian64(tail_.data()));
                tail_size_ = 0;
            }
        }

        while (offset + sizeof(std::uint64_t) <= bytes.size()) {
            MixWord(ReadLittleEndian64(bytes.data() + offset));
            offset += sizeof(std::uint64_t);
        }
        if (offset != bytes.size()) {
            tail_size_ = bytes.size() - offset;
            std::memcpy(tail_.data(), bytes.data() + offset, tail_size_);
        }
    }

    void Update(const void* data, const std::size_t size) noexcept {
        Update({static_cast<const std::uint8_t*>(data), size});
    }

    template <typename T>
        requires std::is_integral_v<T>
    void UpdateLittleEndian(const T value) noexcept {
        using Unsigned = std::make_unsigned_t<T>;
        Unsigned encoded = static_cast<Unsigned>(value);
        std::array<std::uint8_t, sizeof(encoded)> bytes{};
        for (std::size_t byte = 0; byte < sizeof(encoded); ++byte) {
            bytes[byte] = static_cast<std::uint8_t>(encoded >> (byte * 8u));
        }
        Update(bytes);
    }

    template <typename T>
        requires std::is_enum_v<T>
    void UpdateLittleEndian(const T value) noexcept {
        UpdateLittleEndian(static_cast<std::underlying_type_t<T>>(value));
    }

    [[nodiscard]] std::uint64_t Finish() const noexcept {
        std::uint64_t completed = state_;
        if (tail_size_ != 0) {
            std::uint64_t tail_word = 0;
            for (std::size_t index = 0; index < tail_size_; ++index) {
                tail_word |= static_cast<std::uint64_t>(tail_[index]) << (index * 8u);
            }
            completed ^= tail_word + kTailBias +
                         (processed_size_ << 1u) +
                         (static_cast<std::uint64_t>(tail_size_) << 56u);
            completed = std::rotl(completed, 27) * kStepMultiplier;
            completed ^= completed >> 31;
        }
        return Avalanche(completed ^ std::rotl(domain_, 17) ^
                         (size_ * kLengthMultiplier));
    }

private:
    [[nodiscard]] static std::uint64_t ReadLittleEndian64(
        const std::uint8_t* bytes) noexcept {
        std::uint64_t value = 0;
        std::memcpy(&value, bytes, sizeof(value));
        if constexpr (std::endian::native == std::endian::big) {
            value = ((value & 0x00000000000000ffull) << 56u) |
                    ((value & 0x000000000000ff00ull) << 40u) |
                    ((value & 0x0000000000ff0000ull) << 24u) |
                    ((value & 0x00000000ff000000ull) << 8u) |
                    ((value & 0x000000ff00000000ull) >> 8u) |
                    ((value & 0x0000ff0000000000ull) >> 24u) |
                    ((value & 0x00ff000000000000ull) >> 40u) |
                    ((value & 0xff00000000000000ull) >> 56u);
        }
        return value;
    }

    void MixWord(const std::uint64_t word) noexcept {
        state_ ^= word + kStepBias + processed_size_;
        state_ = std::rotl(state_, 23) * kStepMultiplier;
        state_ ^= state_ >> 29;
        processed_size_ += sizeof(word);
    }

    [[nodiscard]] static constexpr std::uint64_t Avalanche(std::uint64_t value) noexcept {
        value ^= value >> 30;
        value *= kAvalancheMultiplier1;
        value ^= value >> 27;
        value *= kAvalancheMultiplier2;
        value ^= value >> 31;
        return value;
    }

    static constexpr std::uint64_t kInitialState = 0x2bf4f48456829b32ull;
    static constexpr std::uint64_t kStepBias = 0x13198a2e03707344ull;
    static constexpr std::uint64_t kStepMultiplier = 0xd6e8feb86659fd93ull;
    static constexpr std::uint64_t kTailBias = 0x082efa98ec4e6c89ull;
    static constexpr std::uint64_t kLengthMultiplier = 0xa4093822299f31d0ull;
    static constexpr std::uint64_t kAvalancheMultiplier1 = 0xbf58476d1ce4e5b9ull;
    static constexpr std::uint64_t kAvalancheMultiplier2 = 0x94d049bb133111ebull;

    std::uint64_t state_;
    std::uint64_t domain_;
    std::uint64_t size_ = 0;
    std::uint64_t processed_size_ = 0;
    std::array<std::uint8_t, sizeof(std::uint64_t)> tail_{};
    std::size_t tail_size_ = 0;
};

[[nodiscard]] inline std::uint64_t FingerprintBytes(
    const std::span<const std::uint8_t> bytes, const std::uint64_t domain) noexcept {
    ContentFingerprint64 fingerprint{domain};
    fingerprint.Update(bytes);
    return fingerprint.Finish();
}

namespace FingerprintDomain {
inline constexpr std::uint64_t ExecutableImage = 0x4558454355544142ull;
inline constexpr std::uint64_t GuestCode = 0x4755455354434f44ull;
inline constexpr std::uint64_t IrCachePayload = 0x4952435041594c44ull;
inline constexpr std::uint64_t NativeSegment = 0x4e41544956455347ull;
inline constexpr std::uint64_t WarmCodeProfile = 0x5741524d434f4445ull;
inline constexpr std::uint64_t CacheIdentityBase = 0x4341434845494430ull;
inline constexpr std::uint64_t DescriptorWrites = 0x4445534357524954ull;
inline constexpr std::uint64_t CaptureBody = 0x4341505455524542ull;
inline constexpr std::uint64_t CaptureSection = 0x4341505453454354ull;
inline constexpr std::uint64_t AttachmentSet = 0x4154544143484d54ull;
inline constexpr std::uint64_t GuestRangeSample = 0x4755455354524e47ull;
inline constexpr std::uint64_t ShaderUserData = 0x5348445255534552ull;
inline constexpr std::uint64_t BufferProbe = 0x42554650524f4245ull;
inline constexpr std::uint64_t RenderWaveKey = 0x52454e4457415645ull;
inline constexpr std::uint64_t RenderWaveWords = 0x57415645574f5244ull;
inline constexpr std::uint64_t CommandStream = 0x434f4d4d414e4453ull;
inline constexpr std::uint64_t DeviceFaultBinary = 0x4445564641554c54ull;
}

}
