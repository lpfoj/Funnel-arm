// SPDX-FileCopyrightText: Copyright 2026 PS4Run Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

using X86DecodeStatus = std::int32_t;
using X86Mnemonic = std::uint32_t;
using X86Register = std::uint32_t;

inline constexpr X86DecodeStatus X86_STATUS_OK = 0;
inline constexpr X86DecodeStatus X86_STATUS_INVALID_ARGUMENT = -1;
inline constexpr X86DecodeStatus X86_STATUS_DECODE_FAILED = -2;

[[nodiscard]] inline constexpr bool X86_SUCCESS(const X86DecodeStatus status) noexcept {
    return status == X86_STATUS_OK;
}

// Bump whenever the public operand normalization changes. Persistent JIT
// IR must never consume records produced under an older decoder contract.
inline constexpr std::uint32_t X86_VERSION = 0x00011501u;
inline constexpr std::size_t X86_MAX_INSTRUCTION_LENGTH = 15;
inline constexpr std::size_t X86_MAX_OPERAND_COUNT = 10;
inline constexpr std::size_t X86_MAX_OPERAND_COUNT_VISIBLE = 5;

inline constexpr std::uint32_t X86_MACHINE_MODE_LONG_64 = 64;
inline constexpr std::uint32_t X86_STACK_WIDTH_64 = 64;
inline constexpr std::uint32_t X86_FORMATTER_STYLE_INTEL = 0;

inline constexpr std::uint8_t X86_OPERAND_TYPE_UNUSED = 0;
inline constexpr std::uint8_t X86_OPERAND_TYPE_REGISTER = 1;
inline constexpr std::uint8_t X86_OPERAND_TYPE_MEMORY = 2;
inline constexpr std::uint8_t X86_OPERAND_TYPE_POINTER = 3;
inline constexpr std::uint8_t X86_OPERAND_TYPE_IMMEDIATE = 4;

inline constexpr std::uint8_t X86_OPERAND_ACTION_MASK_READ = 1;
inline constexpr std::uint8_t X86_OPERAND_ACTION_MASK_WRITE = 2;

inline constexpr std::uint64_t X86_ATTRIB_HAS_LOCK = 1ull << 0;
inline constexpr std::uint64_t X86_ATTRIB_HAS_REP = 1ull << 1;
inline constexpr std::uint64_t X86_ATTRIB_HAS_REPE = 1ull << 2;
inline constexpr std::uint64_t X86_ATTRIB_HAS_REPNE = 1ull << 3;

inline constexpr std::uint8_t X86_INSTRUCTION_ENCODING_LEGACY = 0;
inline constexpr std::uint8_t X86_INSTRUCTION_ENCODING_VEX = 1;
inline constexpr std::uint8_t X86_INSTRUCTION_ENCODING_EVEX = 2;
inline constexpr std::uint8_t X86_INSTRUCTION_ENCODING_XOP = 3;
inline constexpr std::uint8_t X86_INSTRUCTION_ENCODING_MVEX = 5;

#include "common/x86_iced_mnemonics.inc"
#include "common/x86_iced_registers.inc"

struct X86RegisterOperand {
    X86Register value;
};

struct X86Displacement {
    std::int64_t value;
    std::uint8_t offset;
    std::uint8_t size;
};

struct X86MemoryOperand {
    std::uint32_t type;
    X86Register segment;
    X86Register base;
    X86Register index;
    std::uint8_t scale;
    X86Displacement disp;
};

struct X86PointerOperand {
    std::uint16_t segment;
    std::uint32_t offset;
};

union X86ImmediateValue {
    std::uint64_t u;
    std::int64_t s;
};

struct X86ImmediateOperand {
    std::uint8_t is_signed;
    std::uint8_t is_relative;
    X86ImmediateValue value;
    std::uint8_t offset;
    std::uint8_t size;
};

struct X86DecodedOperand {
    std::uint8_t id;
    std::uint8_t visibility;
    std::uint8_t actions;
    std::uint8_t encoding;
    std::uint16_t size;
    std::uint16_t element_type;
    std::uint16_t element_size;
    std::uint16_t element_count;
    std::uint8_t attributes;
    std::uint8_t type;
    union {
        X86RegisterOperand reg;
        X86MemoryOperand mem;
        X86PointerOperand ptr;
        X86ImmediateOperand imm;
    };
};

struct X86DecodedInstruction {
    std::uint64_t attributes;
    X86Mnemonic mnemonic;
    std::uint8_t length;
    std::uint8_t encoding;
    std::uint8_t address_width;
    std::uint8_t operand_width;
    std::uint8_t operand_count;
    std::uint8_t operand_count_visible;
    std::uint8_t bytes[X86_MAX_INSTRUCTION_LENGTH];
};

static_assert(sizeof(X86Displacement) == 16);
static_assert(sizeof(X86MemoryOperand) == 40);
static_assert(sizeof(X86ImmediateOperand) == 24);
static_assert(sizeof(X86DecodedOperand) == 56);
static_assert(sizeof(X86DecodedInstruction) == 40);

struct X86Decoder {
    std::uint32_t machine_mode = X86_MACHINE_MODE_LONG_64;
};

struct X86Formatter {
    std::uint32_t style = X86_FORMATTER_STYLE_INTEL;
};

X86DecodeStatus X86DecoderInit(X86Decoder* decoder, std::uint32_t machine_mode,
                               std::uint32_t stack_width) noexcept;
X86DecodeStatus X86DecoderDecodeFull(const X86Decoder* decoder, const void* data,
                                     std::size_t size, X86DecodedInstruction* instruction,
                                     X86DecodedOperand* operands) noexcept;
X86DecodeStatus X86DecoderDecodeInstruction(const X86Decoder* decoder, const void* context,
                                            const void* data, std::size_t size,
                                            X86DecodedInstruction* instruction) noexcept;

X86DecodeStatus X86FormatterInit(X86Formatter* formatter, std::uint32_t style) noexcept;
X86DecodeStatus X86FormatterFormatInstruction(
    const X86Formatter* formatter, const X86DecodedInstruction* instruction,
    const X86DecodedOperand* operands, std::size_t operand_count, char* output,
    std::size_t output_capacity, std::uint64_t runtime_address, void* context) noexcept;

X86DecodeStatus X86CalcAbsoluteAddress(const X86DecodedInstruction* instruction,
                                      const X86DecodedOperand* operand,
                                      std::uint64_t runtime_address,
                                      std::uint64_t* absolute_address) noexcept;

const char* X86MnemonicGetString(X86Mnemonic mnemonic) noexcept;
const char* X86RegisterGetString(X86Register reg) noexcept;
