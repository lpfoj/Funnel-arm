// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Lsx4::Translation {

using GuestCompareExchange = bool (*)(std::uint64_t address, std::uint32_t width,
                                      std::uint64_t expected, std::uint64_t desired,
                                      std::uint64_t& observed,
                                      void* context) noexcept;

enum class AtomicMutation : std::uint8_t {
    Exchange,
    Add,
    AddCarry,
    Subtract,
    SubtractBorrow,
    BitAnd,
    BitOr,
    BitXor,
    Increment,
    Decrement,
    Negate,
    Invert,
};

enum class AtomicCommit : std::uint8_t {
    Stored,
    Compared,
    AccessFault,
    RetryLimit,
    InvalidRequest,
};

struct AtomicMutationResult {
    AtomicCommit commit{AtomicCommit::InvalidRequest};
    std::uint64_t before{};
    std::uint64_t after{};
    std::uint32_t attempts{};
};

struct AtomicComparisonResult {
    AtomicCommit commit{AtomicCommit::InvalidRequest};
    std::uint64_t observed{};
    bool exchanged{};
};

[[nodiscard]] AtomicMutationResult MutateGuestAtomically(
    GuestCompareExchange compare_exchange, void* context, std::uint64_t address,
    std::uint32_t width, AtomicMutation mutation, std::uint64_t operand = 0,
    bool carry_or_borrow = false) noexcept;

[[nodiscard]] AtomicComparisonResult CompareGuestAtomically(
    GuestCompareExchange compare_exchange, void* context, std::uint64_t address,
    std::uint32_t width, std::uint64_t expected, std::uint64_t desired) noexcept;

} // namespace Lsx4::Translation
