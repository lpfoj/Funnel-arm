// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/atomic_memory.h"

#include "executor/dynamic_translation/integer_math.h"
#include "executor/dynamic_translation/machine_state.h"

namespace Lsx4::Translation {
namespace {

constexpr std::uint32_t MaximumCasAttempts = 128;

bool IsAtomicWidth(const std::uint32_t width) noexcept {
    return width == 8 || width == 16 || width == 32 || width == 64;
}

std::uint64_t Transform(const AtomicMutation mutation, const std::uint64_t current,
                        const std::uint64_t operand, const bool carry_or_borrow,
                        const std::uint32_t width) noexcept {
    const std::uint64_t mask = WidthMask(width);
    switch (mutation) {
    case AtomicMutation::Exchange:
        return operand & mask;
    case AtomicMutation::Add:
        return AddWithCarry(current, operand, false, width).value;
    case AtomicMutation::AddCarry:
        return AddWithCarry(current, operand, carry_or_borrow, width).value;
    case AtomicMutation::Subtract:
        return SubtractWithBorrow(current, operand, false, width).value;
    case AtomicMutation::SubtractBorrow:
        return SubtractWithBorrow(current, operand, carry_or_borrow, width).value;
    case AtomicMutation::BitAnd:
        return (current & operand) & mask;
    case AtomicMutation::BitOr:
        return (current | operand) & mask;
    case AtomicMutation::BitXor:
        return (current ^ operand) & mask;
    case AtomicMutation::Increment:
        return (current + 1) & mask;
    case AtomicMutation::Decrement:
        return (current - 1) & mask;
    case AtomicMutation::Negate:
        return (0 - current) & mask;
    case AtomicMutation::Invert:
        return (~current) & mask;
    }
    return current;
}

} // namespace

AtomicMutationResult MutateGuestAtomically(
    const GuestCompareExchange compare_exchange, void* const context,
    const std::uint64_t address, const std::uint32_t width,
    const AtomicMutation mutation, const std::uint64_t operand,
    const bool carry_or_borrow) noexcept {
    if (!compare_exchange || !IsAtomicWidth(width)) {
        return {};
    }
    const std::uint64_t mask = WidthMask(width);
    std::uint64_t expected = 0;
    for (std::uint32_t attempt = 1; attempt <= MaximumCasAttempts; ++attempt) {
        const std::uint64_t desired =
            Transform(mutation, expected, operand, carry_or_borrow, width);
        std::uint64_t observed = 0;
        if (!compare_exchange(address, width, expected, desired, observed, context)) {
            return {AtomicCommit::AccessFault, observed & mask, desired, attempt};
        }
        observed &= mask;
        if (observed == expected) {
            return {AtomicCommit::Stored, observed, desired, attempt};
        }
        expected = observed;
    }
    return {AtomicCommit::RetryLimit, expected, expected, MaximumCasAttempts};
}

AtomicComparisonResult CompareGuestAtomically(
    const GuestCompareExchange compare_exchange, void* const context,
    const std::uint64_t address, const std::uint32_t width,
    const std::uint64_t expected, const std::uint64_t desired) noexcept {
    if (!compare_exchange || !IsAtomicWidth(width)) {
        return {};
    }
    const std::uint64_t mask = WidthMask(width);
    std::uint64_t observed = 0;
    if (!compare_exchange(address, width, expected & mask, desired & mask, observed,
                          context)) {
        return {AtomicCommit::AccessFault, observed & mask, false};
    }
    observed &= mask;
    return {AtomicCommit::Compared, observed, observed == (expected & mask)};
}

} // namespace Lsx4::Translation
