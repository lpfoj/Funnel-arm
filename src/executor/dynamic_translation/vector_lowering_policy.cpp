// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/vector_lowering_policy.h"

#include "common/x86_decoder.h"

namespace Executor::Jit {
namespace {

bool IsVectorRegister(const LsxOperandRecord& operand) noexcept {
    if (operand.type != X86_OPERAND_TYPE_REGISTER) {
        return false;
    }
    const LsxRegisterCode reg = operand.reg.value;
    return (reg >= X86_REGISTER_XMM0 && reg <= X86_REGISTER_XMM15) ||
           (reg >= X86_REGISTER_YMM0 && reg <= X86_REGISTER_YMM15);
}

} // namespace

DirectVectorPlan PlanDirectVectorKernel(
    const LsxDecodedOp& instruction) noexcept {
    if (instruction.operand_count < 2 ||
        !IsVectorRegister(instruction.operands[0])) {
        return {};
    }
    const auto& source = instruction.operands[1];
    const bool source_is_vector = IsVectorRegister(source);
    const bool source_is_memory =
        source.type == X86_OPERAND_TYPE_MEMORY;
    const std::uint32_t width_bits = instruction.operands[0].size;
    if ((width_bits != 128 && width_bits != 256) ||
        (!source_is_vector && !source_is_memory) ||
        source.size != width_bits) {
        return {};
    }

    switch (static_cast<X86Mnemonic>(instruction.mnemonic)) {
    case X86_MNEMONIC_PTEST:
    case X86_MNEMONIC_VPTEST:
        return {DirectVectorKernel::PackedBitTest, width_bits / 8};
    case X86_MNEMONIC_MOVSLDUP:
    case X86_MNEMONIC_VMOVSLDUP:
        return {DirectVectorKernel::DuplicateEvenDwords, width_bits / 8};
    case X86_MNEMONIC_MOVSHDUP:
    case X86_MNEMONIC_VMOVSHDUP:
        return {DirectVectorKernel::DuplicateOddDwords, width_bits / 8};
    case X86_MNEMONIC_MOVDDUP:
    case X86_MNEMONIC_VMOVDDUP:
        return {DirectVectorKernel::DuplicateLowQwords, width_bits / 8};
    default:
        return {};
    }
}

} // namespace Executor::Jit
