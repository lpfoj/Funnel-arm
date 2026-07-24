// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fmt/format.h>
#include "common/decoder.h"

namespace Common {

DecoderImpl::DecoderImpl() {
    X86DecoderInit(&m_decoder, X86_MACHINE_MODE_LONG_64, X86_STACK_WIDTH_64);
    X86FormatterInit(&m_formatter, X86_FORMATTER_STYLE_INTEL);
}

DecoderImpl::~DecoderImpl() = default;

std::string DecoderImpl::disassembleInst(X86DecodedInstruction& inst,
                                         X86DecodedOperand* operands, u64 address) {
    const int bufLen = 256;
    char szBuffer[bufLen];
    X86FormatterFormatInstruction(&m_formatter, &inst, operands, inst.operand_count_visible,
                                    szBuffer, sizeof(szBuffer), address, nullptr);
    return szBuffer;
}

void DecoderImpl::printInstruction(void* code, u64 address) {
    X86DecodedInstruction instruction;
    X86DecodedOperand operands[X86_MAX_OPERAND_COUNT_VISIBLE];
    X86DecodeStatus status = X86DecoderDecodeFull(
        &m_decoder, code, X86_MAX_INSTRUCTION_LENGTH, &instruction, operands);
    if (!X86_SUCCESS(status)) {
        fmt::print("decode instruction failed at {}\n", fmt::ptr(code));
    } else {
        printInst(instruction, operands, address);
    }
}

void DecoderImpl::printInst(X86DecodedInstruction& inst, X86DecodedOperand* operands,
                            u64 address) {
    std::string s = disassembleInst(inst, operands, address);
    fmt::print("instruction: {}\n", s);
}

X86DecodeStatus DecoderImpl::decodeInstruction(X86DecodedInstruction& inst,
                                          X86DecodedOperand* operands, void* data, u64 size) {
    return X86DecoderDecodeFull(&m_decoder, data, size, &inst, operands);
}

}
