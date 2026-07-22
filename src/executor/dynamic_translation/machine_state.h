// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Lsx4::Translation {

enum class IntegerRegister : std::uint8_t {
    A, C, D, B, Stack, Frame, Source, Destination,
    R8, R9, R10, R11, R12, R13, R14, R15,
};

[[nodiscard]] constexpr std::size_t RegisterIndex(IntegerRegister id) noexcept {
    return static_cast<std::size_t>(id);
}

struct alignas(32) CpuFrame {
    std::array<std::uint64_t, 16> integer{};
    std::uint64_t resume_address{};
    std::uint64_t gs_origin{};
    std::uint64_t fs_origin{};
    std::uint64_t condition_word{};
    std::uint16_t x87_control{0x037f};
    std::uint8_t x87_stack_cursor{};
    std::uint8_t x87_validity{};
    std::uint32_t simd_control{0x1f80};
    std::uint32_t dispatch_phase{};
    std::uint16_t x87_status{};
    std::uint16_t x87_opcode{};
    std::array<std::array<std::uint8_t, 16>, 8> x87_registers{};
    std::uint64_t x87_code_address{};
    std::uint64_t x87_data_address{};
    std::array<std::array<std::uint8_t, 32>, 16> vectors{};
    std::uint64_t fault_context_slot{};
    std::uint64_t active_instruction{};
};

static_assert(std::is_standard_layout_v<CpuFrame>);
static_assert(alignof(CpuFrame) == 32);
static_assert(offsetof(CpuFrame, integer) == 0x000);
static_assert(offsetof(CpuFrame, resume_address) == 0x080);
static_assert(offsetof(CpuFrame, gs_origin) == 0x088);
static_assert(offsetof(CpuFrame, fs_origin) == 0x090);
static_assert(offsetof(CpuFrame, condition_word) == 0x098);
static_assert(offsetof(CpuFrame, simd_control) == 0x0a4);
static_assert(offsetof(CpuFrame, vectors) == 0x140);
static_assert(offsetof(CpuFrame, fault_context_slot) == 0x340);
static_assert(offsetof(CpuFrame, active_instruction) == 0x348);
static_assert(sizeof(CpuFrame) == 0x360);

inline constexpr std::uint64_t CarryFlag = 1ull << 0;
inline constexpr std::uint64_t ParityFlag = 1ull << 2;
inline constexpr std::uint64_t AuxiliaryFlag = 1ull << 4;
inline constexpr std::uint64_t ZeroFlag = 1ull << 6;
inline constexpr std::uint64_t SignFlag = 1ull << 7;
inline constexpr std::uint64_t OverflowFlag = 1ull << 11;
inline constexpr std::uint64_t ArithmeticFlagMask =
    CarryFlag | ParityFlag | AuxiliaryFlag | ZeroFlag | SignFlag | OverflowFlag;

struct ArithmeticFlags {
    std::uint64_t bits{};
};

[[nodiscard]] constexpr std::uint64_t ReplaceArithmeticFlags(
    const std::uint64_t condition_word, const std::uint64_t bits) noexcept {
    return condition_word ^ ((condition_word ^ bits) & ArithmeticFlagMask);
}

[[nodiscard]] constexpr std::uint64_t WidthMask(std::uint32_t width) noexcept {
    return width >= 64 ? ~std::uint64_t{} : ((std::uint64_t{1} << width) - 1);
}

[[nodiscard]] constexpr bool EvenLowByteParity(std::uint64_t value) noexcept {
    std::uint8_t byte = static_cast<std::uint8_t>(value);
    byte ^= static_cast<std::uint8_t>(byte >> 4);
    byte ^= static_cast<std::uint8_t>(byte >> 2);
    byte ^= static_cast<std::uint8_t>(byte >> 1);
    return (byte & 1u) == 0;
}

[[nodiscard]] constexpr ArithmeticFlags FlagsForAddition(
    std::uint64_t left, std::uint64_t right, std::uint32_t width) noexcept {
    const std::uint64_t mask = WidthMask(width);
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    left &= mask;
    right &= mask;
    const std::uint64_t result = (left + right) & mask;
    std::uint64_t bits = EvenLowByteParity(result) ? ParityFlag : 0;
    bits |= result == 0 ? ZeroFlag : 0;
    bits |= (result & sign) != 0 ? SignFlag : 0;
    bits |= ((left ^ right ^ result) & 0x10) != 0 ? AuxiliaryFlag : 0;
    bits |= left > mask - right ? CarryFlag : 0;
    bits |= ((~(left ^ right) & (left ^ result) & sign) != 0) ? OverflowFlag : 0;
    return {bits};
}

[[nodiscard]] constexpr ArithmeticFlags FlagsForSubtraction(
    std::uint64_t left, std::uint64_t right, std::uint32_t width) noexcept {
    const std::uint64_t mask = WidthMask(width);
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    left &= mask;
    right &= mask;
    const std::uint64_t result = (left - right) & mask;
    std::uint64_t bits = EvenLowByteParity(result) ? ParityFlag : 0;
    bits |= result == 0 ? ZeroFlag : 0;
    bits |= (result & sign) != 0 ? SignFlag : 0;
    bits |= ((left ^ right ^ result) & 0x10) != 0 ? AuxiliaryFlag : 0;
    bits |= left < right ? CarryFlag : 0;
    bits |= (((left ^ right) & (left ^ result) & sign) != 0) ? OverflowFlag : 0;
    return {bits};
}

[[nodiscard]] constexpr ArithmeticFlags FlagsForLogicalResult(
    const std::uint64_t value, const std::uint32_t width) noexcept {
    const std::uint64_t narrowed = value & WidthMask(width);
    const std::uint64_t sign = std::uint64_t{1} << (width - 1);
    std::uint64_t bits = EvenLowByteParity(narrowed) ? ParityFlag : 0;
    bits |= narrowed == 0 ? ZeroFlag : 0;
    bits |= (narrowed & sign) != 0 ? SignFlag : 0;
    return {bits};
}

static_assert(FlagsForAddition(0x7f, 1, 8).bits == (AuxiliaryFlag | SignFlag | OverflowFlag));
static_assert((FlagsForAddition(0xff, 1, 8).bits & (CarryFlag | ZeroFlag)) ==
              (CarryFlag | ZeroFlag));
static_assert((FlagsForSubtraction(0, 1, 8).bits & (CarryFlag | SignFlag)) ==
              (CarryFlag | SignFlag));
static_assert((FlagsForSubtraction(0x80, 1, 8).bits & OverflowFlag) != 0);

[[nodiscard]] std::uint64_t ReadInteger(const CpuFrame& frame, IntegerRegister id,
                                        std::uint32_t width, bool upper_byte = false) noexcept;
void WriteInteger(CpuFrame& frame, IntegerRegister id, std::uint64_t value,
                  std::uint32_t width, bool upper_byte = false) noexcept;
[[nodiscard]] std::uint64_t SignExtend(std::uint64_t value, std::uint32_t width) noexcept;
void ApplyAdditionFlags(CpuFrame& frame, std::uint64_t left, std::uint64_t right,
                        std::uint32_t width) noexcept;
void ApplySubtractionFlags(CpuFrame& frame, std::uint64_t left, std::uint64_t right,
                           std::uint32_t width) noexcept;
void ApplyLogicalFlags(CpuFrame& frame, std::uint64_t value,
                       std::uint32_t width) noexcept;

} // namespace Lsx4::Translation
