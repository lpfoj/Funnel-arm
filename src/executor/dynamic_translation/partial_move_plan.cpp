// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/partial_move_plan.h"

#include "common/x86_decoder.h"

namespace Executor::Jit {
namespace {

enum class LaneSelection : std::uint8_t {
    Low,
    High,
};

struct PartialMoveKind {
    bool supported{};
    bool vex{};
    LaneSelection lane{LaneSelection::Low};
};

[[nodiscard]] PartialMoveKind DescribeMnemonic(const X86Mnemonic mnemonic) noexcept {
    const bool high = mnemonic == X86_MNEMONIC_MOVHPS ||
                      mnemonic == X86_MNEMONIC_MOVHPD ||
                      mnemonic == X86_MNEMONIC_VMOVHPS ||
                      mnemonic == X86_MNEMONIC_VMOVHPD;
    const bool low = mnemonic == X86_MNEMONIC_MOVLPS ||
                     mnemonic == X86_MNEMONIC_MOVLPD ||
                     mnemonic == X86_MNEMONIC_VMOVLPS ||
                     mnemonic == X86_MNEMONIC_VMOVLPD;
    const bool vex = mnemonic == X86_MNEMONIC_VMOVLPS ||
                     mnemonic == X86_MNEMONIC_VMOVLPD ||
                     mnemonic == X86_MNEMONIC_VMOVHPS ||
                     mnemonic == X86_MNEMONIC_VMOVHPD;
    return {.supported = high || low,
            .vex = vex,
            .lane = high ? LaneSelection::High : LaneSelection::Low};
}

[[nodiscard]] std::optional<std::uint32_t> LaneOffset(
    const LsxOperandRecord& operand, const LaneSelection lane,
    const std::uint32_t ymm_bank_offset) noexcept {
    if (operand.type != X86_OPERAND_TYPE_REGISTER ||
        operand.reg.value < X86_REGISTER_XMM0 ||
        operand.reg.value > X86_REGISTER_XMM15) {
        return std::nullopt;
    }
    const auto register_index =
        static_cast<std::uint32_t>(operand.reg.value - X86_REGISTER_XMM0);
    const std::uint32_t lane_offset = lane == LaneSelection::High ? 8u : 0u;
    return ymm_bank_offset + register_index * 32u + lane_offset;
}

[[nodiscard]] bool IsQwordMemory(const LsxOperandRecord& operand) noexcept {
    return operand.type == X86_OPERAND_TYPE_MEMORY && operand.size == 64;
}

} // namespace

std::optional<PartialMovePlan> PlanPartialPackedMove(
    const LsxDecodedOp& instruction, const std::uint32_t ymm_bank_offset) noexcept {
    const auto kind = DescribeMnemonic(static_cast<X86Mnemonic>(instruction.mnemonic));
    if (!kind.supported || instruction.operand_count < 2) {
        return std::nullopt;
    }

    const auto opposite_lane = kind.lane == LaneSelection::High
                                   ? LaneSelection::Low
                                   : LaneSelection::High;
    if (instruction.operands[0].type == X86_OPERAND_TYPE_MEMORY) {
        if (instruction.operand_count != 2 || !IsQwordMemory(instruction.operands[0])) {
            return std::nullopt;
        }
        const auto source = LaneOffset(instruction.operands[1], kind.lane,
                                       ymm_bank_offset);
        if (!source) {
            return std::nullopt;
        }
        return PartialMovePlan{.memory_operand = 0,
                               .writes_memory = true,
                               .source_offset = *source};
    }

    const std::uint8_t memory_index = kind.vex ? 2 : 1;
    if (instruction.operand_count != (kind.vex ? 3 : 2) ||
        !IsQwordMemory(instruction.operands[memory_index])) {
        return std::nullopt;
    }
    const auto destination = LaneOffset(instruction.operands[0], kind.lane,
                                        ymm_bank_offset);
    if (!destination) {
        return std::nullopt;
    }

    PartialMovePlan plan{.memory_operand = memory_index,
                         .clears_upper_half = kind.vex,
                         .has_merge_lane = kind.vex,
                         .destination_lane_offset =
                             static_cast<std::uint8_t>(
                                 kind.lane == LaneSelection::High ? 8u : 0u),
                         .destination_offset = *destination};
    if (kind.vex) {
        const auto merge = LaneOffset(instruction.operands[1], opposite_lane,
                                      ymm_bank_offset);
        if (!merge) {
            return std::nullopt;
        }
        plan.merge_offset = *merge;
    }
    return plan;
}

std::optional<CrossLaneMovePlan> PlanCrossLanePackedMove(
    const LsxDecodedOp& instruction) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(instruction.mnemonic);
    const bool low_to_high = mnemonic == X86_MNEMONIC_MOVLHPS ||
                             mnemonic == X86_MNEMONIC_VMOVLHPS;
    const bool high_to_low = mnemonic == X86_MNEMONIC_MOVHLPS ||
                             mnemonic == X86_MNEMONIC_VMOVHLPS;
    if ((!low_to_high && !high_to_low) || instruction.operand_count < 2) {
        return std::nullopt;
    }

    const bool vector_prefix = mnemonic == X86_MNEMONIC_VMOVLHPS ||
                               mnemonic == X86_MNEMONIC_VMOVHLPS;
    const std::uint8_t old_destination = vector_prefix ? 1u : 0u;
    const std::uint8_t new_source = vector_prefix ? 2u : 1u;
    if (instruction.operand_count <= new_source) {
        return std::nullopt;
    }

    // Both instructions assemble two qwords, but MOVHLPS takes the upper qword
    // from each input whereas MOVLHPS takes each input's lower qword.
    return CrossLaneMovePlan{
        .lower_operand = high_to_low ? new_source : old_destination,
        .upper_operand = high_to_low ? old_destination : new_source,
        .lower_source_offset = static_cast<std::uint8_t>(high_to_low ? 8u : 0u),
        .upper_source_offset = static_cast<std::uint8_t>(high_to_low ? 8u : 0u),
        .clears_upper_half = vector_prefix,
    };
}

std::optional<HalfTransferPlan> PlanHalfVectorTransfer(
    const LsxDecodedOp& instruction) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(instruction.mnemonic);
    if ((mnemonic == X86_MNEMONIC_VINSERTF128 ||
         mnemonic == X86_MNEMONIC_VINSERTI128) &&
        instruction.operand_count >= 4) {
        return HalfTransferPlan{.direction = HalfTransferDirection::Insert,
                                .primary_operand = 1,
                                .secondary_operand = 2,
                                .control_operand = 3};
    }
    if ((mnemonic == X86_MNEMONIC_VEXTRACTF128 ||
         mnemonic == X86_MNEMONIC_VEXTRACTI128) &&
        instruction.operand_count >= 3) {
        return HalfTransferPlan{.direction = HalfTransferDirection::Extract,
                                .primary_operand = 1,
                                .control_operand = 2};
    }
    return std::nullopt;
}

} // namespace Executor::Jit
