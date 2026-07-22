// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/region_plan.h"

#include <array>
#include <limits>

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/machine_state.h"

namespace Lsx4::Translation {

FlowBoundary ClassifyFlow(const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_JMP:
    case X86_MNEMONIC_JMPE:
        return FlowBoundary::Jump;
    case X86_MNEMONIC_JA:
    case X86_MNEMONIC_JAE:
    case X86_MNEMONIC_JB:
    case X86_MNEMONIC_JBE:
    case X86_MNEMONIC_JCXZ:
    case X86_MNEMONIC_JE:
    case X86_MNEMONIC_JECXZ:
    case X86_MNEMONIC_JG:
    case X86_MNEMONIC_JGE:
    case X86_MNEMONIC_JL:
    case X86_MNEMONIC_JLE:
    case X86_MNEMONIC_JNE:
    case X86_MNEMONIC_JNO:
    case X86_MNEMONIC_JNP:
    case X86_MNEMONIC_JNS:
    case X86_MNEMONIC_JO:
    case X86_MNEMONIC_JP:
    case X86_MNEMONIC_JRCXZ:
    case X86_MNEMONIC_JS:
    case X86_MNEMONIC_LOOP:
    case X86_MNEMONIC_LOOPE:
    case X86_MNEMONIC_LOOPNE:
        return FlowBoundary::ConditionalJump;
    case X86_MNEMONIC_CALL:
        return FlowBoundary::Call;
    case X86_MNEMONIC_RET:
    case X86_MNEMONIC_RETF:
    case X86_MNEMONIC_IRET:
    case X86_MNEMONIC_IRETD:
    case X86_MNEMONIC_IRETQ:
        return FlowBoundary::Return;
    case X86_MNEMONIC_INT:
    case X86_MNEMONIC_INT1:
    case X86_MNEMONIC_INT3:
    case X86_MNEMONIC_INTO:
    case X86_MNEMONIC_SYSCALL:
    case X86_MNEMONIC_SYSRET:
    case X86_MNEMONIC_SYSRETQ:
        return FlowBoundary::SystemTransfer;
    default:
        return FlowBoundary::None;
    }
}

RegionPlan PlanRegion(const std::uint64_t first_address,
                      const std::size_t instruction_limit,
                      const InstructionFetcher fetch,
                      void* const fetch_context) noexcept {
    RegionPlan plan{};
    plan.first_address = first_address;
    plan.continuation = first_address;
    if (fetch == nullptr || instruction_limit == 0) {
        plan.stop = RegionStop::ReadFailure;
        return plan;
    }
    plan.sequence.reserve(instruction_limit);

    std::array<std::uint8_t, MaximumInstructionBytes> bytes{};
    for (std::size_t ordinal = 0; ordinal < instruction_limit; ++ordinal) {
        const std::size_t available = fetch(plan.continuation, bytes, fetch_context);
        if (available == 0 || available > bytes.size()) {
            plan.stop = RegionStop::ReadFailure;
            return plan;
        }
        Instruction instruction{};
        if (DecodeInstruction(std::span<const std::uint8_t>{bytes.data(), available},
                              plan.continuation, instruction) != DecodeOutcome::Complete ||
            instruction.length == 0) {
            plan.stop = RegionStop::InvalidInstruction;
            return plan;
        }
        if (instruction.length > std::numeric_limits<std::uint64_t>::max() - plan.continuation) {
            plan.stop = RegionStop::AddressOverflow;
            return plan;
        }
        const FlowBoundary boundary = ClassifyFlow(instruction.mnemonic);
        plan.continuation += instruction.length;
        plan.sequence.push_back({instruction, boundary});
        if (boundary != FlowBoundary::None) {
            plan.stop = RegionStop::FlowBoundary;
            return plan;
        }
    }
    plan.stop = RegionStop::InstructionLimit;
    return plan;
}

bool ResolveRelativeTarget(const Instruction& instruction,
                           std::uint64_t& target) noexcept {
    if (instruction.operand_count == 0) {
        return false;
    }
    const Operand& operand = instruction.operands[0];
    if (operand.form != OperandForm::Immediate || !operand.immediate.relative ||
        operand.immediate.encoded_bytes == 0) {
        return false;
    }
    const std::uint32_t width = static_cast<std::uint32_t>(operand.immediate.encoded_bytes) * 8;
    const std::int64_t displacement = static_cast<std::int64_t>(
        SignExtend(operand.immediate.bits, width));
    const std::uint64_t next = instruction.address + instruction.length;
    const std::uint64_t magnitude = displacement < 0
        ? static_cast<std::uint64_t>(-(displacement + 1)) + 1
        : static_cast<std::uint64_t>(displacement);
    if ((displacement > 0 && magnitude > std::numeric_limits<std::uint64_t>::max() - next) ||
        (displacement < 0 && magnitude > next)) {
        return false;
    }
    target = displacement < 0 ? next - magnitude : next + magnitude;
    return true;
}

} // namespace Lsx4::Translation
