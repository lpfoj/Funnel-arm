// SPDX-FileCopyrightText: Copyright 2026 PS4Run Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/jit/machine_code_lens.h"

#include <algorithm>
#include <cstring>

#include "common/x86_decoder.h"

#include "executor/jit/lsx_translation_engine.h"

namespace Executor::Jit {
namespace {

struct DecoderState {
    DecoderState() noexcept {
        ready = X86_SUCCESS(
            X86DecoderInit(&decoder, X86_MACHINE_MODE_LONG_64, X86_STACK_WIDTH_64));
    }

    X86Decoder decoder{};
    bool ready{};
};

const DecoderState& MachineDecoder() noexcept {
    static const DecoderState state{};
    return state;
}

void TranslateOperand(const X86DecodedOperand& source, LsxOperandRecord& target) noexcept {
    target = LsxOperandRecord{};
    target.id = source.id;
    target.visibility = static_cast<std::uint8_t>(source.visibility);
    target.actions = source.actions;
    target.encoding = static_cast<std::uint8_t>(source.encoding);
    target.size = source.size;
    target.element_type = static_cast<std::uint16_t>(source.element_type);
    target.element_size = source.element_size;
    target.element_count = source.element_count;
    target.attributes = source.attributes;
    target.type = static_cast<std::uint8_t>(source.type);

    switch (source.type) {
    case X86_OPERAND_TYPE_UNUSED:
        break;
    case X86_OPERAND_TYPE_REGISTER:
        target.reg.value = static_cast<LsxRegisterCode>(source.reg.value);
        break;
    case X86_OPERAND_TYPE_MEMORY:
        target.mem.type = static_cast<std::uint32_t>(source.mem.type);
        target.mem.segment = static_cast<LsxRegisterCode>(source.mem.segment);
        target.mem.base = static_cast<LsxRegisterCode>(source.mem.base);
        target.mem.index = static_cast<LsxRegisterCode>(source.mem.index);
        target.mem.scale = source.mem.scale;
        target.mem.disp.value = source.mem.disp.value;
        target.mem.disp.offset = source.mem.disp.offset;
        target.mem.disp.size = source.mem.disp.size;
        break;
    case X86_OPERAND_TYPE_POINTER:
        target.ptr.segment = source.ptr.segment;
        target.ptr.offset = source.ptr.offset;
        break;
    case X86_OPERAND_TYPE_IMMEDIATE:
        target.imm.is_signed = static_cast<std::uint8_t>(source.imm.is_signed);
        target.imm.is_relative = static_cast<std::uint8_t>(source.imm.is_relative);
        target.imm.value.u = source.imm.value.u;
        target.imm.offset = source.imm.offset;
        target.imm.size = source.imm.size;
        break;
    default:
        break;
    }
}

} // namespace

ByteLensResult LiftOneMachineInstruction(const std::span<const std::uint8_t> bytes,
                                         LsxDecodedOp& operation) noexcept {
    const DecoderState& state = MachineDecoder();
    if (!state.ready) {
        return ByteLensResult::Unavailable;
    }
    if (bytes.empty()) {
        return ByteLensResult::Rejected;
    }

    X86DecodedInstruction decoded{};
    X86DecodedOperand operands[X86_MAX_OPERAND_COUNT]{};
    const X86DecodeStatus status = X86DecoderDecodeFull(
        &state.decoder, bytes.data(), bytes.size(), &decoded, operands);
    if (!X86_SUCCESS(status)) {
        return ByteLensResult::Rejected;
    }

    operation.length = decoded.length;
    operation.attributes = static_cast<std::uint16_t>(decoded.attributes & 0xffffu);
    operation.mnemonic = static_cast<std::uint32_t>(decoded.mnemonic);
    std::memcpy(operation.bytes.data(), bytes.data(), decoded.length);
    operation.decoded.attributes = decoded.attributes;
    operation.decoded.mnemonic = static_cast<std::uint32_t>(decoded.mnemonic);
    operation.decoded.length = decoded.length;
    operation.decoded.encoding = static_cast<std::uint8_t>(decoded.encoding);
    operation.decoded.address_width = decoded.address_width;
    operation.decoded.operand_width = decoded.operand_width;
    operation.decoded.operand_count = decoded.operand_count;
    operation.decoded.operand_count_visible = decoded.operand_count_visible;
    operation.operand_count = static_cast<std::uint8_t>(
        std::min<std::uint32_t>(decoded.operand_count_visible,
                                X86_MAX_OPERAND_COUNT_VISIBLE));
    for (std::uint8_t index = 0; index < operation.operand_count; ++index) {
        TranslateOperand(operands[index], operation.operands[index]);
    }
    return ByteLensResult::Accepted;
}

bool HasMachineInstruction(const std::span<const std::uint8_t> bytes) noexcept {
    const DecoderState& state = MachineDecoder();
    if (!state.ready || bytes.empty()) {
        return false;
    }

    X86DecodedInstruction decoded{};
    return X86_SUCCESS(X86DecoderDecodeInstruction(
        &state.decoder, nullptr, bytes.data(), bytes.size(), &decoded));
}

const char* MachineRegisterName(const std::uint32_t code) noexcept {
    return X86RegisterGetString(static_cast<X86Register>(code));
}

} // namespace Executor::Jit
