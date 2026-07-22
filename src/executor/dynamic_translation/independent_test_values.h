// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

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

} // namespace Lsx4::Translation::SelfTestValues
