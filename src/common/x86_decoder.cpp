// SPDX-FileCopyrightText: Copyright 2026 PS4Run Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/x86_decoder.h"

#include <iterator>

extern "C" {
X86DecodeStatus lsx_iced_decode(const std::uint8_t* data, std::size_t size,
                                X86DecodedInstruction* instruction,
                                X86DecodedOperand* operands, std::size_t operand_capacity);
X86DecodeStatus lsx_iced_format(const std::uint8_t* data, std::size_t size,
                                std::uint64_t runtime_address, char* output,
                                std::size_t output_capacity);
std::uint32_t lsx_iced_decoder_contract_version();
}

namespace {

constexpr const char* MnemonicNames[] = {
#include "common/x86_iced_mnemonic_names.inc"
};

constexpr const char* RegisterNames[] = {
#include "common/x86_iced_register_names.inc"
};

static_assert(std::size(MnemonicNames) == X86_MNEMONIC_MAX_VALUE + 1);
static_assert(std::size(RegisterNames) == X86_REGISTER_MAX_VALUE + 1);

}

X86DecodeStatus X86DecoderInit(X86Decoder* decoder, const std::uint32_t machine_mode,
                               const std::uint32_t stack_width) noexcept {
    if (decoder == nullptr || machine_mode != X86_MACHINE_MODE_LONG_64 ||
        stack_width != X86_STACK_WIDTH_64 ||
        lsx_iced_decoder_contract_version() != X86_VERSION) {
        return X86_STATUS_INVALID_ARGUMENT;
    }
    decoder->machine_mode = machine_mode;
    return X86_STATUS_OK;
}

X86DecodeStatus X86DecoderDecodeFull(const X86Decoder* decoder, const void* data,
                                     const std::size_t size,
                                     X86DecodedInstruction* instruction,
                                     X86DecodedOperand* operands) noexcept {
    if (decoder == nullptr || decoder->machine_mode != X86_MACHINE_MODE_LONG_64) {
        return X86_STATUS_INVALID_ARGUMENT;
    }
    return lsx_iced_decode(static_cast<const std::uint8_t*>(data), size, instruction, operands,
                           X86_MAX_OPERAND_COUNT);
}

X86DecodeStatus X86DecoderDecodeInstruction(const X86Decoder* decoder, const void*,
                                            const void* data, const std::size_t size,
                                            X86DecodedInstruction* instruction) noexcept {
    if (decoder == nullptr || decoder->machine_mode != X86_MACHINE_MODE_LONG_64) {
        return X86_STATUS_INVALID_ARGUMENT;
    }
    return lsx_iced_decode(static_cast<const std::uint8_t*>(data), size, instruction, nullptr, 0);
}

X86DecodeStatus X86FormatterInit(X86Formatter* formatter, const std::uint32_t style) noexcept {
    if (formatter == nullptr || style != X86_FORMATTER_STYLE_INTEL) {
        return X86_STATUS_INVALID_ARGUMENT;
    }
    formatter->style = style;
    return X86_STATUS_OK;
}

X86DecodeStatus X86FormatterFormatInstruction(
    const X86Formatter* formatter, const X86DecodedInstruction* instruction,
    const X86DecodedOperand*, const std::size_t, char* output,
    const std::size_t output_capacity, const std::uint64_t runtime_address, void*) noexcept {
    if (formatter == nullptr || formatter->style != X86_FORMATTER_STYLE_INTEL ||
        instruction == nullptr) {
        return X86_STATUS_INVALID_ARGUMENT;
    }
    return lsx_iced_format(instruction->bytes, instruction->length, runtime_address, output,
                           output_capacity);
}

X86DecodeStatus X86CalcAbsoluteAddress(const X86DecodedInstruction* instruction,
                                      const X86DecodedOperand* operand,
                                      const std::uint64_t runtime_address,
                                      std::uint64_t* absolute_address) noexcept {
    if (instruction == nullptr || operand == nullptr || absolute_address == nullptr) {
        return X86_STATUS_INVALID_ARGUMENT;
    }
    const std::uint64_t next_ip = runtime_address + instruction->length;
    if (operand->type == X86_OPERAND_TYPE_IMMEDIATE && operand->imm.is_relative) {
        *absolute_address = next_ip + static_cast<std::uint64_t>(operand->imm.value.s);
        return X86_STATUS_OK;
    }
    if (operand->type == X86_OPERAND_TYPE_MEMORY &&
        (operand->mem.base == X86_REGISTER_RIP || operand->mem.base == X86_REGISTER_EIP)) {
        *absolute_address = next_ip + static_cast<std::uint64_t>(operand->mem.disp.value);
        return X86_STATUS_OK;
    }
    return X86_STATUS_DECODE_FAILED;
}

const char* X86MnemonicGetString(const X86Mnemonic mnemonic) noexcept {
    return mnemonic < std::size(MnemonicNames) ? MnemonicNames[mnemonic] : "invalid";
}

const char* X86RegisterGetString(const X86Register reg) noexcept {
    return reg < std::size(RegisterNames) ? RegisterNames[reg] : "none";
}
