// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace Lsx4::Translation {

inline constexpr std::size_t MaximumInstructionBytes = 15;
inline constexpr std::size_t MaximumVisibleOperands = 5;

enum class OperandForm : std::uint8_t {
    None,
    Register,
    Memory,
    FarAddress,
    Immediate,
};

struct AddressExpression {
    std::uint32_t segment{};
    std::uint32_t base{};
    std::uint32_t index{};
    std::int64_t displacement{};
    std::uint8_t scale{};
    std::uint8_t displacement_offset{};
    std::uint8_t displacement_bytes{};
};

struct ImmediateValue {
    std::uint64_t bits{};
    std::uint8_t encoded_offset{};
    std::uint8_t encoded_bytes{};
    bool signed_value{};
    bool relative{};
};

struct Operand {
    OperandForm form{OperandForm::None};
    std::uint8_t source_ordinal{};
    std::uint8_t visibility{};
    std::uint8_t access{};
    std::uint8_t encoding{};
    std::uint8_t attributes{};
    std::uint16_t bit_width{};
    std::uint16_t element_kind{};
    std::uint16_t element_width{};
    std::uint16_t element_count{};
    std::uint32_t register_id{};
    AddressExpression address{};
    ImmediateValue immediate{};
    std::uint16_t far_segment{};
    std::uint32_t far_offset{};
};

struct Instruction {
    std::uint64_t address{};
    std::uint64_t prefix_attributes{};
    std::uint32_t mnemonic{};
    std::array<std::uint8_t, MaximumInstructionBytes> encoding_bytes{};
    std::array<Operand, MaximumVisibleOperands> operands{};
    std::uint8_t length{};
    std::uint8_t encoding_family{};
    std::uint8_t address_width{};
    std::uint8_t operand_width{};
    std::uint8_t operand_count{};
};

enum class DecodeOutcome : std::uint8_t {
    Complete,
    InvalidEncoding,
    DecoderUnavailable,
};

[[nodiscard]] DecodeOutcome DecodeInstruction(std::span<const std::uint8_t> source,
                                              std::uint64_t address,
                                              Instruction& output) noexcept;
[[nodiscard]] bool StartsWithInstruction(std::span<const std::uint8_t> source) noexcept;
[[nodiscard]] const char* RegisterSpelling(std::uint32_t register_id) noexcept;

} // namespace Lsx4::Translation
