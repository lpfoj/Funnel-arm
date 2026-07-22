// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/scalar_semantics.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/integer_math.h"
#include "executor/dynamic_translation/register_access.h"
#include "executor/dynamic_translation/region_plan.h"

#include <array>
#include <bit>
#include <optional>

namespace Lsx4::Translation {
namespace {

struct ScalarValue {
    std::uint64_t bits{};
    std::uint64_t memory_address{};
    std::uint32_t width{};
    bool from_memory{};
};

enum class Predicate : std::uint8_t {
    Above,
    AboveOrEqual,
    Below,
    BelowOrEqual,
    Equal,
    Greater,
    GreaterOrEqual,
    Less,
    LessOrEqual,
    NotEqual,
    NoOverflow,
    NoParity,
    NonNegative,
    Overflow,
    Parity,
    Negative,
};

SemanticResult Unsupported(const Instruction& instruction) noexcept {
    return {SemanticStop::Unsupported, instruction.address, 0};
}

SemanticResult Fault(const Instruction& instruction, const std::uint64_t address) noexcept {
    return {SemanticStop::MemoryFault, instruction.address, address};
}

std::optional<Predicate> PredicateFor(const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_JA:
    case X86_MNEMONIC_CMOVA:
    case X86_MNEMONIC_SETA:
        return Predicate::Above;
    case X86_MNEMONIC_JAE:
    case X86_MNEMONIC_CMOVAE:
    case X86_MNEMONIC_SETAE:
        return Predicate::AboveOrEqual;
    case X86_MNEMONIC_JB:
    case X86_MNEMONIC_CMOVB:
    case X86_MNEMONIC_SETB:
        return Predicate::Below;
    case X86_MNEMONIC_JBE:
    case X86_MNEMONIC_CMOVBE:
    case X86_MNEMONIC_SETBE:
        return Predicate::BelowOrEqual;
    case X86_MNEMONIC_JE:
    case X86_MNEMONIC_CMOVE:
    case X86_MNEMONIC_SETE:
        return Predicate::Equal;
    case X86_MNEMONIC_JG:
    case X86_MNEMONIC_CMOVG:
    case X86_MNEMONIC_SETG:
        return Predicate::Greater;
    case X86_MNEMONIC_JGE:
    case X86_MNEMONIC_CMOVGE:
    case X86_MNEMONIC_SETGE:
        return Predicate::GreaterOrEqual;
    case X86_MNEMONIC_JL:
    case X86_MNEMONIC_CMOVL:
    case X86_MNEMONIC_SETL:
        return Predicate::Less;
    case X86_MNEMONIC_JLE:
    case X86_MNEMONIC_CMOVLE:
    case X86_MNEMONIC_SETLE:
        return Predicate::LessOrEqual;
    case X86_MNEMONIC_JNE:
    case X86_MNEMONIC_CMOVNE:
    case X86_MNEMONIC_SETNE:
        return Predicate::NotEqual;
    case X86_MNEMONIC_JNO:
    case X86_MNEMONIC_CMOVNO:
    case X86_MNEMONIC_SETNO:
        return Predicate::NoOverflow;
    case X86_MNEMONIC_JNP:
    case X86_MNEMONIC_CMOVNP:
    case X86_MNEMONIC_SETNP:
        return Predicate::NoParity;
    case X86_MNEMONIC_JNS:
    case X86_MNEMONIC_CMOVNS:
    case X86_MNEMONIC_SETNS:
        return Predicate::NonNegative;
    case X86_MNEMONIC_JO:
    case X86_MNEMONIC_CMOVO:
    case X86_MNEMONIC_SETO:
        return Predicate::Overflow;
    case X86_MNEMONIC_JP:
    case X86_MNEMONIC_CMOVP:
    case X86_MNEMONIC_SETP:
        return Predicate::Parity;
    case X86_MNEMONIC_JS:
    case X86_MNEMONIC_CMOVS:
    case X86_MNEMONIC_SETS:
        return Predicate::Negative;
    default:
        return std::nullopt;
    }
}

bool TestPredicate(const Predicate predicate, const std::uint64_t flags) noexcept {
    const bool carry = (flags & CarryFlag) != 0;
    const bool parity = (flags & ParityFlag) != 0;
    const bool zero = (flags & ZeroFlag) != 0;
    const bool sign = (flags & SignFlag) != 0;
    const bool overflow = (flags & OverflowFlag) != 0;
    switch (predicate) {
    case Predicate::Above:
        return !carry && !zero;
    case Predicate::AboveOrEqual:
        return !carry;
    case Predicate::Below:
        return carry;
    case Predicate::BelowOrEqual:
        return carry || zero;
    case Predicate::Equal:
        return zero;
    case Predicate::Greater:
        return !zero && sign == overflow;
    case Predicate::GreaterOrEqual:
        return sign == overflow;
    case Predicate::Less:
        return sign != overflow;
    case Predicate::LessOrEqual:
        return zero || sign != overflow;
    case Predicate::NotEqual:
        return !zero;
    case Predicate::NoOverflow:
        return !overflow;
    case Predicate::NoParity:
        return !parity;
    case Predicate::NonNegative:
        return !sign;
    case Predicate::Overflow:
        return overflow;
    case Predicate::Parity:
        return parity;
    case Predicate::Negative:
        return sign;
    }
    return false;
}

std::optional<std::uint64_t> ReadAddressRegister(const std::uint32_t register_id,
                                                 const std::uint8_t address_width,
                                                 const CpuFrame& frame) noexcept {
    Operand synthetic{};
    synthetic.form = OperandForm::Register;
    synthetic.register_id = register_id;
    synthetic.bit_width = address_width;
    const auto access = DecodeIntegerRegister(synthetic);
    if (!access) {
        return std::nullopt;
    }
    return ReadInteger(frame, static_cast<IntegerRegister>(access->slot), access->width,
                       access->upper_byte);
}

std::optional<std::uint64_t> EffectiveAddress(const Instruction& instruction,
                                              const Operand& operand,
                                              const CpuFrame& frame) noexcept {
    if (operand.form != OperandForm::Memory) {
        return std::nullopt;
    }
    std::uint64_t address = 0;
    if (operand.address.base == X86_REGISTER_RIP ||
        operand.address.base == X86_REGISTER_EIP) {
        address = instruction.address + instruction.length;
    } else if (operand.address.base != X86_REGISTER_NONE) {
        const auto base = ReadAddressRegister(operand.address.base,
                                              instruction.address_width, frame);
        if (!base) {
            return std::nullopt;
        }
        address = *base;
    }
    if (operand.address.index != X86_REGISTER_NONE) {
        const auto index = ReadAddressRegister(operand.address.index,
                                               instruction.address_width, frame);
        if (!index) {
            return std::nullopt;
        }
        address += *index * operand.address.scale;
    }
    address += static_cast<std::uint64_t>(operand.address.displacement);
    if (operand.address.segment == X86_REGISTER_FS) {
        address += frame.fs_origin;
    } else if (operand.address.segment == X86_REGISTER_GS) {
        address += frame.gs_origin;
    }
    if (instruction.address_width == 32) {
        address = static_cast<std::uint32_t>(address);
    }
    return address;
}

bool ReadMemoryScalar(const GuestMemoryPort& memory, const std::uint64_t address,
                      const std::uint32_t width, std::uint64_t& value) noexcept {
    const std::size_t size = width / 8;
    if (!memory.read || size == 0 || size > sizeof(value)) {
        return false;
    }
    std::array<std::uint8_t, 8> bytes{};
    if (!memory.read(address, std::span<std::uint8_t>{bytes.data(), size},
                     memory.context)) {
        return false;
    }
    value = 0;
    for (std::size_t index = 0; index < size; ++index) {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8u);
    }
    return true;
}

bool WriteMemoryScalar(const GuestMemoryPort& memory, const std::uint64_t address,
                       const std::uint32_t width, const std::uint64_t value) noexcept {
    const std::size_t size = width / 8;
    if (!memory.write || size == 0 || size > sizeof(value)) {
        return false;
    }
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t index = 0; index < size; ++index) {
        bytes[index] = static_cast<std::uint8_t>(value >> (index * 8u));
    }
    return memory.write(address, std::span<const std::uint8_t>{bytes.data(), size},
                        memory.context);
}

std::optional<ScalarValue> ReadOperand(const Instruction& instruction,
                                       const Operand& operand, const CpuFrame& frame,
                                       const GuestMemoryPort& memory) noexcept {
    if (operand.bit_width == 0 || operand.bit_width > 64) {
        return std::nullopt;
    }
    if (operand.form == OperandForm::Immediate) {
        return ScalarValue{operand.immediate.bits & WidthMask(operand.bit_width), 0,
                           operand.bit_width, false};
    }
    if (const auto access = DecodeIntegerRegister(operand)) {
        return ScalarValue{ReadInteger(frame, static_cast<IntegerRegister>(access->slot),
                                       access->width, access->upper_byte),
                           0, access->width, false};
    }
    const auto address = EffectiveAddress(instruction, operand, frame);
    std::uint64_t value = 0;
    if (!address || !ReadMemoryScalar(memory, *address, operand.bit_width, value)) {
        return std::nullopt;
    }
    return ScalarValue{value, *address, operand.bit_width, true};
}

std::optional<std::uint64_t> DestinationAddress(const Instruction& instruction,
                                                const Operand& operand,
                                                const CpuFrame& frame) noexcept {
    return operand.form == OperandForm::Memory
               ? EffectiveAddress(instruction, operand, frame)
               : std::optional<std::uint64_t>{0};
}

bool WriteOperand(const Operand& operand, const std::optional<std::uint64_t> address,
                  CpuFrame& frame, const GuestMemoryPort& memory,
                  const std::uint64_t value) noexcept {
    if (const auto access = DecodeIntegerRegister(operand)) {
        WriteInteger(frame, static_cast<IntegerRegister>(access->slot), value,
                     access->width, access->upper_byte);
        return true;
    }
    return operand.form == OperandForm::Memory && address &&
           WriteMemoryScalar(memory, *address, operand.bit_width, value);
}

void ApplyLogic(CpuFrame& frame, const std::uint64_t result,
                const std::uint32_t width) noexcept {
    ApplyLogicalFlags(frame, result, width);
}

SemanticResult AtomicFailure(const Instruction& instruction,
                             const std::uint64_t address,
                             const AtomicCommit commit) noexcept {
    if (commit == AtomicCommit::AccessFault) {
        return Fault(instruction, address);
    }
    if (commit == AtomicCommit::RetryLimit) {
        return {SemanticStop::Retry, instruction.address, 0};
    }
    return Unsupported(instruction);
}

std::optional<AtomicMutation> BinaryAtomicMutation(
    const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_ADD:
        return AtomicMutation::Add;
    case X86_MNEMONIC_ADC:
        return AtomicMutation::AddCarry;
    case X86_MNEMONIC_SUB:
        return AtomicMutation::Subtract;
    case X86_MNEMONIC_SBB:
        return AtomicMutation::SubtractBorrow;
    case X86_MNEMONIC_AND:
        return AtomicMutation::BitAnd;
    case X86_MNEMONIC_OR:
        return AtomicMutation::BitOr;
    case X86_MNEMONIC_XOR:
        return AtomicMutation::BitXor;
    default:
        return std::nullopt;
    }
}

SemanticResult ExecuteLockedAlu(const Instruction& instruction, CpuFrame& frame,
                                const GuestMemoryPort& memory) noexcept {
    if (instruction.operand_count != 2 ||
        instruction.operands[0].form != OperandForm::Memory ||
        instruction.operands[1].form == OperandForm::Memory) {
        return Unsupported(instruction);
    }
    const auto mutation = BinaryAtomicMutation(instruction.mnemonic);
    const auto address = EffectiveAddress(instruction, instruction.operands[0], frame);
    const auto source = ReadOperand(instruction, instruction.operands[1], frame, memory);
    const std::uint32_t width = instruction.operands[0].bit_width;
    if (!mutation || !address || !source || source->width != width) {
        return Unsupported(instruction);
    }
    const bool carry = (frame.condition_word & CarryFlag) != 0;
    const AtomicMutationResult transaction = MutateGuestAtomically(
        memory.compare_exchange, memory.context, *address, width, *mutation,
        source->bits, carry);
    if (transaction.commit != AtomicCommit::Stored) {
        return AtomicFailure(instruction, *address, transaction.commit);
    }
    if (instruction.mnemonic == X86_MNEMONIC_ADD) {
        ApplyAdditionFlags(frame, transaction.before, source->bits, width);
    } else if (instruction.mnemonic == X86_MNEMONIC_SUB) {
        ApplySubtractionFlags(frame, transaction.before, source->bits, width);
    } else if (instruction.mnemonic == X86_MNEMONIC_ADC) {
        const IntegerMathResult result =
            AddWithCarry(transaction.before, source->bits, carry, width);
        frame.condition_word = (frame.condition_word & ~result.flag_mask) |
                               result.flag_bits;
    } else if (instruction.mnemonic == X86_MNEMONIC_SBB) {
        const IntegerMathResult result =
            SubtractWithBorrow(transaction.before, source->bits, carry, width);
        frame.condition_word = (frame.condition_word & ~result.flag_mask) |
                               result.flag_bits;
    } else {
        ApplyLogic(frame, transaction.after, width);
    }
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteMove(const Instruction& instruction, CpuFrame& frame,
                           const GuestMemoryPort& memory) noexcept {
    if (instruction.operand_count != 2) {
        return Unsupported(instruction);
    }
    const auto destination = DestinationAddress(instruction, instruction.operands[0], frame);
    const auto source = ReadOperand(instruction, instruction.operands[1], frame, memory);
    if (!destination || !source) {
        const std::uint64_t address = destination ? 0 : instruction.address;
        return Fault(instruction, address);
    }
    if (!WriteOperand(instruction.operands[0], destination, frame, memory, source->bits)) {
        return Fault(instruction, *destination);
    }
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteExtension(const Instruction& instruction, CpuFrame& frame,
                                const GuestMemoryPort& memory, const bool sign_extend) noexcept {
    if (instruction.operand_count != 2) {
        return Unsupported(instruction);
    }
    const auto destination = DestinationAddress(instruction, instruction.operands[0], frame);
    const auto source = ReadOperand(instruction, instruction.operands[1], frame, memory);
    if (!destination || !source) {
        return Fault(instruction, source ? 0 : instruction.address);
    }
    const std::uint64_t value =
        sign_extend ? SignExtend(source->bits, source->width) : source->bits;
    if (!WriteOperand(instruction.operands[0], destination, frame, memory, value)) {
        return Fault(instruction, *destination);
    }
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteLea(const Instruction& instruction, CpuFrame& frame) noexcept {
    if (instruction.operand_count != 2) {
        return Unsupported(instruction);
    }
    const auto address = EffectiveAddress(instruction, instruction.operands[1], frame);
    const auto destination = DestinationAddress(instruction, instruction.operands[0], frame);
    if (!address || !destination ||
        !WriteOperand(instruction.operands[0], destination, frame, {}, *address)) {
        return Unsupported(instruction);
    }
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteAlu(const Instruction& instruction, CpuFrame& frame,
                          const GuestMemoryPort& memory) noexcept {
    if ((instruction.prefix_attributes & X86_ATTRIB_HAS_LOCK) != 0) {
        return ExecuteLockedAlu(instruction, frame, memory);
    }
    if (instruction.operand_count != 2) {
        return Unsupported(instruction);
    }
    const auto left = ReadOperand(instruction, instruction.operands[0], frame, memory);
    const auto right = ReadOperand(instruction, instruction.operands[1], frame, memory);
    if (!left || !right || left->width != right->width) {
        return Fault(instruction, left && left->from_memory ? left->memory_address
                                                            : instruction.address);
    }
    const std::uint64_t mask = WidthMask(left->width);
    std::uint64_t result = 0;
    bool write = true;
    enum class Flags : std::uint8_t { Add, Subtract, Logic, Direct } flags = Flags::Logic;
    IntegerMathResult direct{};
    switch (instruction.mnemonic) {
    case X86_MNEMONIC_ADD:
        result = (left->bits + right->bits) & mask;
        flags = Flags::Add;
        break;
    case X86_MNEMONIC_SUB:
        result = (left->bits - right->bits) & mask;
        flags = Flags::Subtract;
        break;
    case X86_MNEMONIC_ADC:
        direct = AddWithCarry(left->bits, right->bits,
                              (frame.condition_word & CarryFlag) != 0, left->width);
        result = direct.value;
        flags = Flags::Direct;
        break;
    case X86_MNEMONIC_SBB:
        direct = SubtractWithBorrow(left->bits, right->bits,
                                    (frame.condition_word & CarryFlag) != 0,
                                    left->width);
        result = direct.value;
        flags = Flags::Direct;
        break;
    case X86_MNEMONIC_CMP:
        result = (left->bits - right->bits) & mask;
        flags = Flags::Subtract;
        write = false;
        break;
    case X86_MNEMONIC_AND:
        result = left->bits & right->bits;
        break;
    case X86_MNEMONIC_OR:
        result = left->bits | right->bits;
        break;
    case X86_MNEMONIC_XOR:
        result = left->bits ^ right->bits;
        break;
    case X86_MNEMONIC_TEST:
        result = left->bits & right->bits;
        write = false;
        break;
    default:
        return Unsupported(instruction);
    }
    if (write && !WriteOperand(instruction.operands[0], left->from_memory
                                                            ? std::optional{left->memory_address}
                                                            : std::optional<std::uint64_t>{0},
                               frame, memory, result)) {
        return Fault(instruction, left->memory_address);
    }
    if (flags == Flags::Add) {
        ApplyAdditionFlags(frame, left->bits, right->bits, left->width);
    } else if (flags == Flags::Subtract) {
        ApplySubtractionFlags(frame, left->bits, right->bits, left->width);
    } else if (flags == Flags::Direct) {
        frame.condition_word = (frame.condition_word & ~direct.flag_mask) |
                               direct.flag_bits;
    } else {
        ApplyLogic(frame, result, left->width);
    }
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteBitMovement(const Instruction& instruction, CpuFrame& frame,
                                  const GuestMemoryPort& memory) noexcept {
    if (instruction.operand_count != 2 ||
        (instruction.prefix_attributes & X86_ATTRIB_HAS_LOCK) != 0) {
        return Unsupported(instruction);
    }
    const auto value = ReadOperand(instruction, instruction.operands[0], frame, memory);
    const auto count = ReadOperand(instruction, instruction.operands[1], frame, memory);
    if (!value || !count) {
        return Fault(instruction, value && value->from_memory ? value->memory_address
                                                              : instruction.address);
    }
    const auto evaluated =
        EvaluateBitMovement(instruction.mnemonic, value->bits, count->bits, value->width);
    if (!evaluated) {
        return Unsupported(instruction);
    }
    if (!evaluated->effective) {
        return {SemanticStop::Continue, instruction.address + instruction.length, 0};
    }
    const std::optional<std::uint64_t> destination =
        value->from_memory ? std::optional{value->memory_address}
                           : std::optional<std::uint64_t>{0};
    if (!WriteOperand(instruction.operands[0], destination, frame, memory,
                      evaluated->value)) {
        return Fault(instruction, value->memory_address);
    }
    frame.condition_word = (frame.condition_word & ~evaluated->flag_mask) |
                           evaluated->flag_bits;
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteUnary(const Instruction& instruction, CpuFrame& frame,
                            const GuestMemoryPort& memory) noexcept {
    if (instruction.operand_count != 1) {
        return Unsupported(instruction);
    }
    if ((instruction.prefix_attributes & X86_ATTRIB_HAS_LOCK) != 0) {
        if (instruction.operands[0].form != OperandForm::Memory) {
            return Unsupported(instruction);
        }
        const auto address = EffectiveAddress(instruction, instruction.operands[0], frame);
        std::optional<AtomicMutation> mutation;
        if (instruction.mnemonic == X86_MNEMONIC_INC) {
            mutation = AtomicMutation::Increment;
        } else if (instruction.mnemonic == X86_MNEMONIC_DEC) {
            mutation = AtomicMutation::Decrement;
        } else if (instruction.mnemonic == X86_MNEMONIC_NEG) {
            mutation = AtomicMutation::Negate;
        } else if (instruction.mnemonic == X86_MNEMONIC_NOT) {
            mutation = AtomicMutation::Invert;
        }
        if (!address || !mutation) {
            return Unsupported(instruction);
        }
        const AtomicMutationResult transaction = MutateGuestAtomically(
            memory.compare_exchange, memory.context, *address,
            instruction.operands[0].bit_width, *mutation);
        if (transaction.commit != AtomicCommit::Stored) {
            return AtomicFailure(instruction, *address, transaction.commit);
        }
        const std::uint64_t carry = frame.condition_word & CarryFlag;
        if (instruction.mnemonic == X86_MNEMONIC_INC) {
            ApplyAdditionFlags(frame, transaction.before, 1,
                               instruction.operands[0].bit_width);
            frame.condition_word = (frame.condition_word & ~CarryFlag) | carry;
        } else if (instruction.mnemonic == X86_MNEMONIC_DEC) {
            ApplySubtractionFlags(frame, transaction.before, 1,
                                  instruction.operands[0].bit_width);
            frame.condition_word = (frame.condition_word & ~CarryFlag) | carry;
        } else if (instruction.mnemonic == X86_MNEMONIC_NEG) {
            ApplySubtractionFlags(frame, 0, transaction.before,
                                  instruction.operands[0].bit_width);
        }
        return {SemanticStop::Continue, instruction.address + instruction.length, 0};
    }
    const auto old = ReadOperand(instruction, instruction.operands[0], frame, memory);
    if (!old) {
        return Fault(instruction, instruction.address);
    }
    const std::uint64_t mask = WidthMask(old->width);
    std::uint64_t result = 0;
    if (instruction.mnemonic == X86_MNEMONIC_NOT) {
        result = ~old->bits & mask;
    } else if (instruction.mnemonic == X86_MNEMONIC_NEG) {
        result = (0 - old->bits) & mask;
    } else if (instruction.mnemonic == X86_MNEMONIC_INC) {
        result = (old->bits + 1) & mask;
    } else if (instruction.mnemonic == X86_MNEMONIC_DEC) {
        result = (old->bits - 1) & mask;
    } else {
        return Unsupported(instruction);
    }
    const std::optional<std::uint64_t> destination =
        old->from_memory ? std::optional{old->memory_address}
                         : std::optional<std::uint64_t>{0};
    if (!WriteOperand(instruction.operands[0], destination, frame, memory, result)) {
        return Fault(instruction, old->memory_address);
    }
    if (instruction.mnemonic == X86_MNEMONIC_NEG) {
        ApplySubtractionFlags(frame, 0, old->bits, old->width);
    } else if (instruction.mnemonic == X86_MNEMONIC_INC ||
               instruction.mnemonic == X86_MNEMONIC_DEC) {
        const std::uint64_t carry = frame.condition_word & CarryFlag;
        if (instruction.mnemonic == X86_MNEMONIC_INC) {
            ApplyAdditionFlags(frame, old->bits, 1, old->width);
        } else {
            ApplySubtractionFlags(frame, old->bits, 1, old->width);
        }
        frame.condition_word = (frame.condition_word & ~CarryFlag) | carry;
    }
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteExchangeAdd(const Instruction& instruction, CpuFrame& frame,
                                  const GuestMemoryPort& memory) noexcept {
    if (instruction.operand_count != 2 ||
        instruction.operands[1].form != OperandForm::Register ||
        instruction.operands[0].bit_width != instruction.operands[1].bit_width) {
        return Unsupported(instruction);
    }
    const auto source = ReadOperand(instruction, instruction.operands[1], frame, memory);
    if (!source) {
        return Unsupported(instruction);
    }
    const bool memory_destination =
        instruction.operands[0].form == OperandForm::Memory;
    const bool requires_atomic =
        instruction.mnemonic == X86_MNEMONIC_XCHG && memory_destination;
    const bool requested_atomic =
        (instruction.prefix_attributes & X86_ATTRIB_HAS_LOCK) != 0;
    if (requires_atomic || requested_atomic) {
        if (!memory_destination) {
            return Unsupported(instruction);
        }
        const auto address = EffectiveAddress(instruction, instruction.operands[0], frame);
        if (!address) {
            return Unsupported(instruction);
        }
        const AtomicMutation mutation = instruction.mnemonic == X86_MNEMONIC_XADD
                                            ? AtomicMutation::Add
                                            : AtomicMutation::Exchange;
        const AtomicMutationResult transaction = MutateGuestAtomically(
            memory.compare_exchange, memory.context, *address, source->width, mutation,
            source->bits);
        if (transaction.commit != AtomicCommit::Stored) {
            return AtomicFailure(instruction, *address, transaction.commit);
        }
        if (!WriteOperand(instruction.operands[1], 0, frame, memory,
                          transaction.before)) {
            return Unsupported(instruction);
        }
        if (instruction.mnemonic == X86_MNEMONIC_XADD) {
            ApplyAdditionFlags(frame, transaction.before, source->bits, source->width);
        }
        return {SemanticStop::Continue, instruction.address + instruction.length, 0};
    }

    const auto destination = ReadOperand(instruction, instruction.operands[0], frame, memory);
    if (!destination || destination->width != source->width) {
        return Fault(instruction, instruction.address);
    }
    const std::uint64_t new_destination =
        instruction.mnemonic == X86_MNEMONIC_XADD
            ? (destination->bits + source->bits) & WidthMask(source->width)
            : source->bits;
    const std::optional<std::uint64_t> destination_address =
        destination->from_memory ? std::optional{destination->memory_address}
                                 : std::optional<std::uint64_t>{0};
    if (destination->from_memory &&
        !WriteOperand(instruction.operands[0], destination_address, frame, memory,
                      new_destination)) {
        return Fault(instruction, destination->memory_address);
    }
    if (!WriteOperand(instruction.operands[1], 0, frame, memory, destination->bits)) {
        return Unsupported(instruction);
    }
    if (!destination->from_memory &&
        !WriteOperand(instruction.operands[0], destination_address, frame, memory,
                      new_destination)) {
        return Unsupported(instruction);
    }
    if (instruction.mnemonic == X86_MNEMONIC_XADD) {
        ApplyAdditionFlags(frame, destination->bits, source->bits, source->width);
    }
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteCompareExchange(const Instruction& instruction, CpuFrame& frame,
                                      const GuestMemoryPort& memory) noexcept {
    if (instruction.operand_count != 2 ||
        instruction.operands[1].form != OperandForm::Register ||
        instruction.operands[0].bit_width != instruction.operands[1].bit_width) {
        return Unsupported(instruction);
    }
    const std::uint32_t width = instruction.operands[0].bit_width;
    const std::uint64_t expected = ReadInteger(frame, IntegerRegister::A, width);
    const auto replacement = ReadOperand(instruction, instruction.operands[1], frame, memory);
    if (!replacement) {
        return Unsupported(instruction);
    }
    std::uint64_t observed = 0;
    bool exchanged = false;
    if ((instruction.prefix_attributes & X86_ATTRIB_HAS_LOCK) != 0) {
        if (instruction.operands[0].form != OperandForm::Memory) {
            return Unsupported(instruction);
        }
        const auto address = EffectiveAddress(instruction, instruction.operands[0], frame);
        if (!address) {
            return Unsupported(instruction);
        }
        const AtomicComparisonResult comparison = CompareGuestAtomically(
            memory.compare_exchange, memory.context, *address, width, expected,
            replacement->bits);
        if (comparison.commit != AtomicCommit::Compared) {
            return AtomicFailure(instruction, *address, comparison.commit);
        }
        observed = comparison.observed;
        exchanged = comparison.exchanged;
    } else {
        const auto destination = ReadOperand(instruction, instruction.operands[0], frame, memory);
        if (!destination) {
            return Fault(instruction, instruction.address);
        }
        observed = destination->bits;
        exchanged = observed == expected;
        if (exchanged) {
            const std::optional<std::uint64_t> address =
                destination->from_memory ? std::optional{destination->memory_address}
                                         : std::optional<std::uint64_t>{0};
            if (!WriteOperand(instruction.operands[0], address, frame, memory,
                              replacement->bits)) {
                return Fault(instruction, destination->memory_address);
            }
        }
    }
    ApplySubtractionFlags(frame, expected, observed, width);
    if (!exchanged) {
        WriteInteger(frame, IntegerRegister::A, observed, width);
    }
    return {SemanticStop::Continue, instruction.address + instruction.length, 0};
}

SemanticResult ExecuteConditional(const Instruction& instruction, CpuFrame& frame,
                                  const GuestMemoryPort& memory,
                                  const Predicate predicate) noexcept {
    const bool take = TestPredicate(predicate, frame.condition_word);
    if (instruction.mnemonic >= X86_MNEMONIC_CMOVA &&
        instruction.mnemonic <= X86_MNEMONIC_CMOVS) {
        if (!take) {
            return {SemanticStop::Continue, instruction.address + instruction.length, 0};
        }
        return ExecuteMove(instruction, frame, memory);
    }
    if (instruction.mnemonic >= X86_MNEMONIC_SETA &&
        instruction.mnemonic <= X86_MNEMONIC_SETS) {
        if (instruction.operand_count != 1) {
            return Unsupported(instruction);
        }
        const auto destination = DestinationAddress(instruction, instruction.operands[0], frame);
        if (!destination ||
            !WriteOperand(instruction.operands[0], destination, frame, memory, take ? 1 : 0)) {
            return Fault(instruction, destination ? *destination : instruction.address);
        }
        return {SemanticStop::Continue, instruction.address + instruction.length, 0};
    }
    std::uint64_t target = instruction.address + instruction.length;
    if (take && !ResolveRelativeTarget(instruction, target)) {
        return Unsupported(instruction);
    }
    return {SemanticStop::Transfer, target, 0};
}

SemanticResult ExecuteStack(const Instruction& instruction, CpuFrame& frame,
                            const GuestMemoryPort& memory) noexcept {
    constexpr IntegerRegister stack = IntegerRegister::Stack;
    const std::uint64_t old_stack = ReadInteger(frame, stack, 64);
    if (instruction.mnemonic == X86_MNEMONIC_PUSH) {
        if (instruction.operand_count != 1) {
            return Unsupported(instruction);
        }
        const auto value = ReadOperand(instruction, instruction.operands[0], frame, memory);
        const std::uint64_t new_stack = old_stack - 8;
        if (!value || !WriteMemoryScalar(memory, new_stack, 64, value->bits)) {
            return Fault(instruction, new_stack);
        }
        WriteInteger(frame, stack, new_stack, 64);
        return {SemanticStop::Continue, instruction.address + instruction.length, 0};
    }
    if (instruction.mnemonic == X86_MNEMONIC_POP) {
        if (instruction.operand_count != 1 ||
            instruction.operands[0].form != OperandForm::Register) {
            return Unsupported(instruction);
        }
        std::uint64_t value = 0;
        if (!ReadMemoryScalar(memory, old_stack, 64, value)) {
            return Fault(instruction, old_stack);
        }
        WriteInteger(frame, stack, old_stack + 8, 64);
        if (!WriteOperand(instruction.operands[0], 0, frame, memory, value)) {
            return Unsupported(instruction);
        }
        return {SemanticStop::Continue, instruction.address + instruction.length, 0};
    }
    return Unsupported(instruction);
}

SemanticResult ExecuteControl(const Instruction& instruction, CpuFrame& frame,
                              const GuestMemoryPort& memory) noexcept {
    if (instruction.mnemonic == X86_MNEMONIC_RET) {
        const std::uint64_t stack = ReadInteger(frame, IntegerRegister::Stack, 64);
        std::uint64_t target = 0;
        if (!ReadMemoryScalar(memory, stack, 64, target)) {
            return Fault(instruction, stack);
        }
        std::uint64_t adjustment = 8;
        if (instruction.operand_count != 0) {
            const auto immediate = ReadOperand(instruction, instruction.operands[0], frame, memory);
            if (!immediate) {
                return Unsupported(instruction);
            }
            adjustment += immediate->bits & 0xffffu;
        }
        WriteInteger(frame, IntegerRegister::Stack, stack + adjustment, 64);
        return {SemanticStop::Return, target, 0};
    }
    if (instruction.mnemonic != X86_MNEMONIC_JMP &&
        instruction.mnemonic != X86_MNEMONIC_CALL) {
        return Unsupported(instruction);
    }
    if (instruction.operand_count != 1) {
        return Unsupported(instruction);
    }
    std::uint64_t target = 0;
    if (instruction.operands[0].form == OperandForm::Immediate &&
        instruction.operands[0].immediate.relative) {
        if (!ResolveRelativeTarget(instruction, target)) {
            return Unsupported(instruction);
        }
    } else {
        const auto value = ReadOperand(instruction, instruction.operands[0], frame, memory);
        if (!value) {
            return Fault(instruction, instruction.address);
        }
        target = value->bits;
    }
    if (instruction.mnemonic == X86_MNEMONIC_CALL) {
        const std::uint64_t stack = ReadInteger(frame, IntegerRegister::Stack, 64) - 8;
        if (!WriteMemoryScalar(memory, stack, 64,
                               instruction.address + instruction.length)) {
            return Fault(instruction, stack);
        }
        WriteInteger(frame, IntegerRegister::Stack, stack, 64);
    }
    return {SemanticStop::Transfer, target, 0};
}

} // namespace

SemanticResult ExecuteScalarInstruction(const Instruction& instruction, CpuFrame& frame,
                                        const GuestMemoryPort& memory) noexcept {
    if (instruction.length == 0) {
        return Unsupported(instruction);
    }
    if (const auto predicate = PredicateFor(instruction.mnemonic)) {
        return ExecuteConditional(instruction, frame, memory, *predicate);
    }
    switch (instruction.mnemonic) {
    case X86_MNEMONIC_NOP:
        return {SemanticStop::Continue, instruction.address + instruction.length, 0};
    case X86_MNEMONIC_MOV:
        return ExecuteMove(instruction, frame, memory);
    case X86_MNEMONIC_MOVZX:
        return ExecuteExtension(instruction, frame, memory, false);
    case X86_MNEMONIC_MOVSX:
    case X86_MNEMONIC_MOVSXD:
        return ExecuteExtension(instruction, frame, memory, true);
    case X86_MNEMONIC_LEA:
        return ExecuteLea(instruction, frame);
    case X86_MNEMONIC_ADD:
    case X86_MNEMONIC_ADC:
    case X86_MNEMONIC_SUB:
    case X86_MNEMONIC_SBB:
    case X86_MNEMONIC_CMP:
    case X86_MNEMONIC_AND:
    case X86_MNEMONIC_OR:
    case X86_MNEMONIC_XOR:
    case X86_MNEMONIC_TEST:
        return ExecuteAlu(instruction, frame, memory);
    case X86_MNEMONIC_SHL:
    case X86_MNEMONIC_SHR:
    case X86_MNEMONIC_SAR:
    case X86_MNEMONIC_ROL:
    case X86_MNEMONIC_ROR:
        return ExecuteBitMovement(instruction, frame, memory);
    case X86_MNEMONIC_INC:
    case X86_MNEMONIC_DEC:
    case X86_MNEMONIC_NEG:
    case X86_MNEMONIC_NOT:
        return ExecuteUnary(instruction, frame, memory);
    case X86_MNEMONIC_XADD:
    case X86_MNEMONIC_XCHG:
        return ExecuteExchangeAdd(instruction, frame, memory);
    case X86_MNEMONIC_CMPXCHG:
        return ExecuteCompareExchange(instruction, frame, memory);
    case X86_MNEMONIC_PUSH:
    case X86_MNEMONIC_POP:
        return ExecuteStack(instruction, frame, memory);
    case X86_MNEMONIC_CALL:
    case X86_MNEMONIC_JMP:
    case X86_MNEMONIC_RET:
        return ExecuteControl(instruction, frame, memory);
    default:
        return Unsupported(instruction);
    }
}

} // namespace Lsx4::Translation
