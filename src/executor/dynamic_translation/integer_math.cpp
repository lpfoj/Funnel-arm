// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/integer_math.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/machine_state.h"

namespace Lsx4::Translation {
namespace {

std::uint64_t StatusBits(const std::uint64_t result, const std::uint32_t width) noexcept {
    const std::uint64_t narrowed = result & WidthMask(width);
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    std::uint64_t flags = EvenLowByteParity(narrowed) ? ParityFlag : 0;
    flags |= narrowed == 0 ? ZeroFlag : 0;
    flags |= (narrowed & sign) != 0 ? SignFlag : 0;
    return flags;
}

__int128 SignedValue(const std::uint64_t value, const std::uint32_t width) noexcept {
    const std::uint64_t narrowed = value & WidthMask(width);
    if ((narrowed & (std::uint64_t{1} << (width - 1))) == 0) {
        return static_cast<__int128>(narrowed);
    }
    return static_cast<__int128>(narrowed) - (static_cast<__int128>(1) << width);
}

bool OutsideSignedWidth(const __int128 value, const std::uint32_t width) noexcept {
    const __int128 limit = static_cast<__int128>(1) << (width - 1);
    return value < -limit || value >= limit;
}

std::uint64_t RotateLeftWidth(const std::uint64_t value, const std::uint32_t amount,
                              const std::uint32_t width) noexcept {
    const std::uint64_t mask = WidthMask(width);
    return ((value << amount) | (value >> (width - amount))) & mask;
}

std::uint64_t RotateRightWidth(const std::uint64_t value, const std::uint32_t amount,
                               const std::uint32_t width) noexcept {
    const std::uint64_t mask = WidthMask(width);
    return ((value >> amount) | (value << (width - amount))) & mask;
}

} // namespace

IntegerMathResult AddWithCarry(std::uint64_t left, std::uint64_t right,
                               const bool carry, const std::uint32_t width) noexcept {
    const std::uint64_t mask = WidthMask(width);
    left &= mask;
    right &= mask;
    const unsigned __int128 full = static_cast<unsigned __int128>(left) + right + carry;
    const std::uint64_t result = static_cast<std::uint64_t>(full) & mask;
    std::uint64_t flags = StatusBits(result, width);
    flags |= full > mask ? CarryFlag : 0;
    flags |= ((left ^ right ^ result) & 0x10u) != 0 ? AuxiliaryFlag : 0;
    const __int128 signed_sum = SignedValue(left, width) + SignedValue(right, width) + carry;
    flags |= OutsideSignedWidth(signed_sum, width) ? OverflowFlag : 0;
    return {result, flags, ArithmeticFlagMask, true};
}

IntegerMathResult SubtractWithBorrow(std::uint64_t left, std::uint64_t right,
                                     const bool borrow,
                                     const std::uint32_t width) noexcept {
    const std::uint64_t mask = WidthMask(width);
    left &= mask;
    right &= mask;
    const unsigned __int128 subtrahend =
        static_cast<unsigned __int128>(right) + static_cast<unsigned>(borrow);
    const std::uint64_t result = (left - right - static_cast<unsigned>(borrow)) & mask;
    std::uint64_t flags = StatusBits(result, width);
    flags |= static_cast<unsigned __int128>(left) < subtrahend ? CarryFlag : 0;
    flags |= ((left ^ right ^ result) & 0x10u) != 0 ? AuxiliaryFlag : 0;
    const __int128 signed_difference =
        SignedValue(left, width) - SignedValue(right, width) - borrow;
    flags |= OutsideSignedWidth(signed_difference, width) ? OverflowFlag : 0;
    return {result, flags, ArithmeticFlagMask, true};
}

std::optional<IntegerMathResult> EvaluateBitMovement(
    const std::uint32_t mnemonic, std::uint64_t value, const std::uint64_t count,
    const std::uint32_t width) noexcept {
    if (width != 8 && width != 16 && width != 32 && width != 64) {
        return std::nullopt;
    }
    const std::uint32_t masked_count =
        static_cast<std::uint32_t>(count) & (width == 64 ? 0x3fu : 0x1fu);
    value &= WidthMask(width);
    if (masked_count == 0) {
        return IntegerMathResult{value, 0, 0, false};
    }

    if (mnemonic == X86_MNEMONIC_ROL || mnemonic == X86_MNEMONIC_ROR) {
        const std::uint32_t rotation = masked_count % width;
        if (rotation == 0) {
            return IntegerMathResult{value, 0, 0, false};
        }
        const bool left = mnemonic == X86_MNEMONIC_ROL;
        const std::uint64_t result = left ? RotateLeftWidth(value, rotation, width)
                                          : RotateRightWidth(value, rotation, width);
        const std::uint64_t sign = std::uint64_t{1} << (width - 1);
        const bool carry = left ? (result & 1u) != 0 : (result & sign) != 0;
        std::uint64_t flags = carry ? CarryFlag : 0;
        if (masked_count == 1) {
            const bool top = (result & sign) != 0;
            const bool next = (result & (sign >> 1)) != 0;
            const bool overflow = left ? top != carry : top != next;
            flags |= overflow ? OverflowFlag : 0;
        }
        return IntegerMathResult{result, flags, CarryFlag | OverflowFlag, true};
    }

    if (mnemonic != X86_MNEMONIC_SHL && mnemonic != X86_MNEMONIC_SHR &&
        mnemonic != X86_MNEMONIC_SAR) {
        return std::nullopt;
    }
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    const bool original_sign = (value & sign) != 0;
    std::uint64_t result = 0;
    bool carry = false;
    if (mnemonic == X86_MNEMONIC_SHL) {
        carry = masked_count <= width &&
                ((value >> (width - masked_count)) & 1u) != 0;
        result = masked_count < width ? (value << masked_count) & WidthMask(width) : 0;
    } else if (mnemonic == X86_MNEMONIC_SHR) {
        carry = masked_count <= width && ((value >> (masked_count - 1)) & 1u) != 0;
        result = masked_count < width ? value >> masked_count : 0;
    } else {
        carry = masked_count <= width && ((value >> (masked_count - 1)) & 1u) != 0;
        if (masked_count >= width) {
            result = original_sign ? WidthMask(width) : 0;
        } else if (original_sign) {
            result = (value >> masked_count) |
                     (WidthMask(width) << (width - masked_count));
        } else {
            result = value >> masked_count;
        }
        result &= WidthMask(width);
    }
    std::uint64_t flags = StatusBits(result, width) | (carry ? CarryFlag : 0);
    if (masked_count == 1) {
        const bool result_sign = (result & sign) != 0;
        const bool overflow = mnemonic == X86_MNEMONIC_SHL
                                  ? result_sign != carry
                              : mnemonic == X86_MNEMONIC_SHR ? original_sign
                                                             : false;
        flags |= overflow ? OverflowFlag : 0;
    }
    return IntegerMathResult{result, flags, ArithmeticFlagMask, true};
}

} // namespace Lsx4::Translation
