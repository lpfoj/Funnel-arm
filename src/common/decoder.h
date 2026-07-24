// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/x86_decoder.h"
#include "common/singleton.h"
#include "common/types.h"

namespace Common {

class DecoderImpl {
public:
    DecoderImpl();
    ~DecoderImpl();

    std::string disassembleInst(X86DecodedInstruction& inst, X86DecodedOperand* operands,
                                u64 address);
    void printInst(X86DecodedInstruction& inst, X86DecodedOperand* operands, u64 address);
    void printInstruction(void* code, u64 address);
    X86DecodeStatus decodeInstruction(X86DecodedInstruction& inst, X86DecodedOperand* operands,
                                 void* data, u64 size = 15);

private:
    X86Decoder m_decoder;
    X86Formatter m_formatter;
};

using Decoder = Common::Singleton<DecoderImpl>;

}
