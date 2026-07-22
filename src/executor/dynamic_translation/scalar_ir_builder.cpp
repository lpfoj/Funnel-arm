// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/scalar_ir_builder.h"

#include "common/x86_decoder.h"

#include <array>
#include <optional>

namespace Lsx4::Translation {
namespace {

std::optional<IntegerAccess> DecodeIntegerAccess(const Operand& operand) noexcept {
    if (operand.form != OperandForm::Register || operand.bit_width == 0 ||
        operand.bit_width > 64) {
        return std::nullopt;
    }
    const std::uint32_t id = operand.register_id;
    IntegerAccess result{.width = static_cast<std::uint8_t>(operand.bit_width)};
    if (id >= X86_REGISTER_AL && id <= X86_REGISTER_BL) {
        result.slot = static_cast<std::uint8_t>(id - X86_REGISTER_AL);
    } else if (id >= X86_REGISTER_AH && id <= X86_REGISTER_BH) {
        result.slot = static_cast<std::uint8_t>(id - X86_REGISTER_AH);
        result.upper_byte = true;
    } else if (id >= X86_REGISTER_SPL && id <= X86_REGISTER_R15L) {
        result.slot = static_cast<std::uint8_t>(id - X86_REGISTER_SPL + 4);
    } else if (id >= X86_REGISTER_AX && id <= X86_REGISTER_R15W) {
        result.slot = static_cast<std::uint8_t>(id - X86_REGISTER_AX);
    } else if (id >= X86_REGISTER_EAX && id <= X86_REGISTER_R15D) {
        result.slot = static_cast<std::uint8_t>(id - X86_REGISTER_EAX);
    } else if (id >= X86_REGISTER_RAX && id <= X86_REGISTER_R15) {
        result.slot = static_cast<std::uint8_t>(id - X86_REGISTER_RAX);
    } else {
        return std::nullopt;
    }
    return result;
}

IrType IntegerType(const std::uint16_t width) noexcept {
    switch (width) {
    case 8:
        return IrType::Integer8;
    case 16:
        return IrType::Integer16;
    case 32:
        return IrType::Integer32;
    case 64:
        return IrType::Integer64;
    default:
        return IrType::None;
    }
}

IrValue ReadScalarOperand(TranslationProgram& program, const Operand& operand,
                          const std::uint64_t guest_address) {
    const IrType type = IntegerType(operand.bit_width);
    if (type == IrType::None) {
        return {};
    }
    if (operand.form == OperandForm::Immediate) {
        return program.ir.AddValue(IrAction::Constant, type, {}, operand.immediate.bits,
                                   0, guest_address);
    }
    const auto access = DecodeIntegerAccess(operand);
    if (!access) {
        return {};
    }
    return program.ir.AddValue(IrAction::ReadInteger, type, {}, 0,
                               PackIntegerAccess(*access), guest_address);
}

bool WriteScalarOperand(TranslationProgram& program, const Operand& operand,
                        const IrValue value, const std::uint64_t guest_address) {
    const auto access = DecodeIntegerAccess(operand);
    if (!access || !value.Exists()) {
        return false;
    }
    const std::array<IrValue, 1> input{value};
    return program.ir.AddEffect(IrAction::WriteInteger, input, 0,
                                PackIntegerAccess(*access), guest_address);
}

bool EmitFlagUpdate(TranslationProgram& program, const FlagFormula formula,
                    const IrValue left, const IrValue right, const IrValue result,
                    const std::uint64_t guest_address) {
    const std::array<IrValue, 3> inputs{left, right, result};
    const IrValue flags = program.ir.AddValue(
        IrAction::ComposeFlags, IrType::FlagSet, inputs, 0,
        static_cast<std::uint32_t>(formula), guest_address);
    if (!flags.Exists()) {
        return false;
    }
    const std::array<IrValue, 1> flag_input{flags};
    return program.ir.AddEffect(IrAction::WriteFlags, flag_input, 0, 0, guest_address);
}

bool EmitMove(TranslationProgram& program, const Instruction& instruction) {
    if (instruction.operand_count != 2) {
        return false;
    }
    const IrValue source =
        ReadScalarOperand(program, instruction.operands[1], instruction.address);
    return WriteScalarOperand(program, instruction.operands[0], source,
                              instruction.address);
}

struct BinaryMeaning {
    IrAction action{};
    FlagFormula flags{};
    bool write_result{};
};

std::optional<BinaryMeaning> MeaningOf(const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_ADD:
        return BinaryMeaning{IrAction::Add, FlagFormula::Addition, true};
    case X86_MNEMONIC_SUB:
        return BinaryMeaning{IrAction::Subtract, FlagFormula::Subtraction, true};
    case X86_MNEMONIC_CMP:
        return BinaryMeaning{IrAction::Subtract, FlagFormula::Subtraction, false};
    case X86_MNEMONIC_AND:
        return BinaryMeaning{IrAction::BitAnd, FlagFormula::Logical, true};
    case X86_MNEMONIC_OR:
        return BinaryMeaning{IrAction::BitOr, FlagFormula::Logical, true};
    case X86_MNEMONIC_XOR:
        return BinaryMeaning{IrAction::BitXor, FlagFormula::Logical, true};
    case X86_MNEMONIC_TEST:
        return BinaryMeaning{IrAction::BitAnd, FlagFormula::Logical, false};
    default:
        return std::nullopt;
    }
}

bool EmitBinary(TranslationProgram& program, const Instruction& instruction,
                const BinaryMeaning meaning) {
    if (instruction.operand_count != 2 ||
        instruction.operands[0].form != OperandForm::Register) {
        return false;
    }
    const IrValue left =
        ReadScalarOperand(program, instruction.operands[0], instruction.address);
    const IrValue right =
        ReadScalarOperand(program, instruction.operands[1], instruction.address);
    if (!left.Exists() || !right.Exists() || left.type != right.type) {
        return false;
    }
    const std::array<IrValue, 2> inputs{left, right};
    const IrValue result =
        program.ir.AddValue(meaning.action, left.type, inputs, 0, 0, instruction.address);
    if (!result.Exists() ||
        (meaning.write_result && !WriteScalarOperand(
                                     program, instruction.operands[0], result,
                                     instruction.address))) {
        return false;
    }
    return EmitFlagUpdate(program, meaning.flags, left, right, result,
                          instruction.address);
}

bool EmitIncrement(TranslationProgram& program, const Instruction& instruction,
                   const bool decrement) {
    if (instruction.operand_count != 1 ||
        instruction.operands[0].form != OperandForm::Register) {
        return false;
    }
    const IrValue old =
        ReadScalarOperand(program, instruction.operands[0], instruction.address);
    if (!old.Exists()) {
        return false;
    }
    const IrValue one = program.ir.AddValue(IrAction::Constant, old.type, {}, 1, 0,
                                            instruction.address);
    const std::array<IrValue, 2> inputs{old, one};
    const IrAction action = decrement ? IrAction::Subtract : IrAction::Add;
    const IrValue result =
        program.ir.AddValue(action, old.type, inputs, 0, 0, instruction.address);
    if (!WriteScalarOperand(program, instruction.operands[0], result,
                            instruction.address)) {
        return false;
    }
    return EmitFlagUpdate(program,
                          decrement ? FlagFormula::Decrement : FlagFormula::Increment,
                          old, one, result, instruction.address);
}

bool EmitTerminal(TranslationProgram& program, const Instruction& instruction) {
    if (instruction.mnemonic == X86_MNEMONIC_RET) {
        return program.ir.AddEffect(IrAction::ExitReturn, {}, 0, 0, instruction.address);
    }
    if (instruction.mnemonic != X86_MNEMONIC_JMP) {
        return false;
    }
    std::uint64_t target = 0;
    if (!ResolveRelativeTarget(instruction, target)) {
        return false;
    }
    return program.ir.AddEffect(IrAction::ExitDirect, {}, target, 0,
                                instruction.address);
}

bool EmitKnown(TranslationProgram& program, const Instruction& instruction) {
    if (instruction.mnemonic == X86_MNEMONIC_MOV) {
        return EmitMove(program, instruction);
    }
    if (const auto meaning = MeaningOf(instruction.mnemonic)) {
        return EmitBinary(program, instruction, *meaning);
    }
    if (instruction.mnemonic == X86_MNEMONIC_INC ||
        instruction.mnemonic == X86_MNEMONIC_DEC) {
        return EmitIncrement(program, instruction,
                             instruction.mnemonic == X86_MNEMONIC_DEC);
    }
    return EmitTerminal(program, instruction);
}

void EmitFallback(TranslationProgram& program, const Instruction& instruction) {
    const std::size_t index = program.semantic_steps.size();
    program.semantic_steps.push_back(instruction);
    (void)program.ir.AddEffect(IrAction::SemanticFallback, {}, index,
                               instruction.mnemonic, instruction.address);
}

} // namespace

std::uint32_t PackIntegerAccess(const IntegerAccess access) noexcept {
    return static_cast<std::uint32_t>(access.slot) |
           (static_cast<std::uint32_t>(access.width) << 8) |
           (static_cast<std::uint32_t>(access.upper_byte) << 16);
}

IntegerAccess UnpackIntegerAccess(const std::uint32_t detail) noexcept {
    return {
        .slot = static_cast<std::uint8_t>(detail),
        .width = static_cast<std::uint8_t>(detail >> 8),
        .upper_byte = ((detail >> 16) & 1u) != 0,
    };
}

TranslationProgram BuildScalarProgram(const RegionPlan& plan) {
    TranslationProgram program{plan.first_address};
    for (const PlannedInstruction& planned : plan.sequence) {
        if (!EmitKnown(program, planned.instruction)) {
            EmitFallback(program, planned.instruction);
        }
    }
    if (!plan.sequence.empty() && plan.sequence.back().boundary == FlowBoundary::None) {
        (void)program.ir.AddEffect(IrAction::ExitDirect, {}, plan.continuation, 0,
                                   plan.sequence.back().instruction.address);
    }
    return program;
}

} // namespace Lsx4::Translation
