// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/instruction_model.h"

#include <algorithm>
#include <cstring>

#include "common/x86_decoder.h"

namespace Lsx4::Translation {
namespace {

struct DecoderService {
    DecoderService() noexcept
        : available{X86_SUCCESS(X86DecoderInit(&handle, X86_MACHINE_MODE_LONG_64,
                                               X86_STACK_WIDTH_64))} {}

    X86Decoder handle{};
    bool available{};
};

const DecoderService& Service() noexcept {
    static const DecoderService service;
    return service;
}

Operand ConvertOperand(const X86DecodedOperand& source) noexcept {
    Operand result{};
    result.source_ordinal = source.id;
    result.visibility = static_cast<std::uint8_t>(source.visibility);
    result.access = source.actions;
    result.encoding = static_cast<std::uint8_t>(source.encoding);
    result.attributes = source.attributes;
    result.bit_width = source.size;
    result.element_kind = static_cast<std::uint16_t>(source.element_type);
    result.element_width = source.element_size;
    result.element_count = source.element_count;

    if (source.type == X86_OPERAND_TYPE_REGISTER) {
        result.form = OperandForm::Register;
        result.register_id = static_cast<std::uint32_t>(source.reg.value);
    } else if (source.type == X86_OPERAND_TYPE_MEMORY) {
        result.form = OperandForm::Memory;
        result.address.segment = static_cast<std::uint32_t>(source.mem.segment);
        result.address.base = static_cast<std::uint32_t>(source.mem.base);
        result.address.index = static_cast<std::uint32_t>(source.mem.index);
        result.address.scale = source.mem.scale;
        result.address.displacement = source.mem.disp.value;
        result.address.displacement_offset = source.mem.disp.offset;
        result.address.displacement_bytes = source.mem.disp.size;
    } else if (source.type == X86_OPERAND_TYPE_POINTER) {
        result.form = OperandForm::FarAddress;
        result.far_segment = source.ptr.segment;
        result.far_offset = source.ptr.offset;
    } else if (source.type == X86_OPERAND_TYPE_IMMEDIATE) {
        result.form = OperandForm::Immediate;
        result.immediate.bits = source.imm.value.u;
        result.immediate.encoded_offset = source.imm.offset;
        result.immediate.encoded_bytes = source.imm.size;
        result.immediate.signed_value = source.imm.is_signed != 0;
        result.immediate.relative = source.imm.is_relative != 0;
    }
    return result;
}

} // namespace

DecodeOutcome DecodeInstruction(const std::span<const std::uint8_t> source,
                                const std::uint64_t address,
                                Instruction& output) noexcept {
    output = {};
    output.address = address;
    const DecoderService& service = Service();
    if (!service.available) {
        return DecodeOutcome::DecoderUnavailable;
    }
    if (source.empty()) {
        return DecodeOutcome::InvalidEncoding;
    }

    X86DecodedInstruction decoded{};
    std::array<X86DecodedOperand, X86_MAX_OPERAND_COUNT> decoded_operands{};
    if (!X86_SUCCESS(X86DecoderDecodeFull(&service.handle, source.data(), source.size(),
                                          &decoded, decoded_operands.data()))) {
        return DecodeOutcome::InvalidEncoding;
    }

    output.prefix_attributes = decoded.attributes;
    output.mnemonic = static_cast<std::uint32_t>(decoded.mnemonic);
    output.length = decoded.length;
    output.encoding_family = static_cast<std::uint8_t>(decoded.encoding);
    output.address_width = decoded.address_width;
    output.operand_width = decoded.operand_width;
    std::memcpy(output.encoding_bytes.data(), source.data(), output.length);
    output.operand_count = static_cast<std::uint8_t>(std::min<std::size_t>(
        decoded.operand_count_visible, output.operands.size()));
    for (std::size_t index = 0; index < output.operand_count; ++index) {
        output.operands[index] = ConvertOperand(decoded_operands[index]);
    }
    return DecodeOutcome::Complete;
}

bool StartsWithInstruction(const std::span<const std::uint8_t> source) noexcept {
    const DecoderService& service = Service();
    if (!service.available || source.empty()) {
        return false;
    }
    X86DecodedInstruction decoded{};
    return X86_SUCCESS(X86DecoderDecodeInstruction(&service.handle, nullptr, source.data(),
                                                    source.size(), &decoded));
}

const char* RegisterSpelling(const std::uint32_t register_id) noexcept {
    return X86RegisterGetString(static_cast<X86Register>(register_id));
}

} // namespace Lsx4::Translation
