// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/machine_state.h"

#include <bit>

namespace Lsx4::Translation {

std::uint64_t ReadInteger(const CpuFrame& frame, const IntegerRegister id,
                          const std::uint32_t width, const bool upper_byte) noexcept {
    const std::uint64_t stored = frame.integer[RegisterIndex(id)];
    if (width == 8 && upper_byte) {
        return (stored >> 8) & 0xff;
    }
    return stored & WidthMask(width);
}

void WriteInteger(CpuFrame& frame, const IntegerRegister id, const std::uint64_t value,
                  const std::uint32_t width, const bool upper_byte) noexcept {
    std::uint64_t& destination = frame.integer[RegisterIndex(id)];
    if (width == 32) {
        destination = static_cast<std::uint32_t>(value);
        return;
    }
    if (width >= 64) {
        destination = value;
        return;
    }
    const std::uint32_t shift = upper_byte && width == 8 ? 8 : 0;
    const std::uint64_t field = WidthMask(width) << shift;
    destination = (destination & ~field) | ((value << shift) & field);
}

std::uint64_t SignExtend(std::uint64_t value, const std::uint32_t width) noexcept {
    if (width >= 64) {
        return value;
    }
    const std::uint64_t mask = WidthMask(width);
    value &= mask;
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    return (value ^ sign) - sign;
}

namespace {

void ReplaceArithmeticFlags(CpuFrame& frame, const ArithmeticFlags flags) noexcept {
    frame.condition_word = (frame.condition_word & ~ArithmeticFlagMask) | flags.bits;
}

} // namespace

void ApplyAdditionFlags(CpuFrame& frame, const std::uint64_t left,
                        const std::uint64_t right, const std::uint32_t width) noexcept {
    ReplaceArithmeticFlags(frame, FlagsForAddition(left, right, width));
}

void ApplySubtractionFlags(CpuFrame& frame, const std::uint64_t left,
                           const std::uint64_t right, const std::uint32_t width) noexcept {
    ReplaceArithmeticFlags(frame, FlagsForSubtraction(left, right, width));
}

void ApplyLogicalFlags(CpuFrame& frame, const std::uint64_t value,
                       const std::uint32_t width) noexcept {
    const std::uint64_t narrowed = value & WidthMask(width);
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    std::uint64_t bits = EvenLowByteParity(narrowed) ? ParityFlag : 0;
    bits |= narrowed == 0 ? ZeroFlag : 0;
    bits |= (narrowed & sign) != 0 ? SignFlag : 0;
    ReplaceArithmeticFlags(frame, {bits});
}

} // namespace Lsx4::Translation
