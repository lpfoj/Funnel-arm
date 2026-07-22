// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdint>

namespace Lsx4::Translation::SelfTestValues {

// Deliberately irregular LSX4-owned canaries. Their alternating byte weight
// catches truncation, lane reversal and accidental zero extension without
// retaining the conventional counting/DEADBEEF fixtures from the retired suite.
inline constexpr std::uint64_t kWideCanary = 0x6d3a91c5e7b2048full;
inline constexpr std::uint64_t kReverseCanary = 0x92c56e3a184dfb70ull;
inline constexpr std::uint64_t kStatePoison = 0xb6e13c47a529d80full;
inline constexpr std::uint64_t kUpperPoison = 0xb6e13c4700000000ull;
inline constexpr std::uint64_t kReturnMarker = 0x4a8d72f19c36e5b0ull;
inline constexpr std::uint32_t kWordCanary = 0x6d3a91c5u;
inline constexpr std::uint64_t kLaneCanary = 0x2f84c16ba75de309ull;
inline constexpr std::uint64_t kAuxCanaryA = 0x71e4b8362dc90fa5ull;
inline constexpr std::uint64_t kAuxCanaryB = 0xc25907ed34a681bfull;
inline constexpr std::uint64_t kAuxCanaryC = 0x18bd63f2a9475ce0ull;
inline constexpr std::uint64_t kAuxCanaryD = 0xe46a1f93b8520dc7ull;
inline constexpr std::uint64_t kAuxCanaryE = 0x53c8a27d9614ef0bull;
inline constexpr std::uint32_t kQuietNan32 = 0x7fc6d3a9u;
inline constexpr std::uint32_t kSignalingNan32 = 0x7fa2c5e7u;

[[nodiscard]] consteval std::array<std::uint8_t, 6> EncodeMovEaxReturn(
    const std::uint32_t value) {
    return {0xb8,
            static_cast<std::uint8_t>(value),
            static_cast<std::uint8_t>(value >> 8u),
            static_cast<std::uint8_t>(value >> 16u),
            static_cast<std::uint8_t>(value >> 24u),
            0xc3};
}

} // namespace Lsx4::Translation::SelfTestValues
