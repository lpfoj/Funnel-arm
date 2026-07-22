// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/x87_semantics.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/address_resolver.h"
#include "executor/dynamic_translation/x87_state.h"

#include <array>
#include <cmath>
#include <cstring>
#include <numbers>
#include <optional>

namespace Lsx4::Translation {
namespace {

X87SemanticResult Unsupported(const Instruction& instruction) noexcept {
    return {X87Stop::Unsupported, instruction.address, 0};
}

X87SemanticResult Fault(const Instruction& instruction,
                        const std::uint64_t address) noexcept {
    return {X87Stop::MemoryFault, instruction.address, address};
}

X87SemanticResult StackFaultResult(const Instruction& instruction) noexcept {
    return {X87Stop::StackFault, instruction.address, 0};
}

std::optional<std::uint8_t> StackIndex(const Operand& operand) noexcept {
    if (operand.form != OperandForm::Register ||
        operand.register_id < X86_REGISTER_ST0 ||
        operand.register_id > X86_REGISTER_ST7) {
        return std::nullopt;
    }
    return static_cast<std::uint8_t>(operand.register_id - X86_REGISTER_ST0);
}

std::optional<long double> ReadFloatingMemory(const Instruction& instruction,
                                              const Operand& operand,
                                              const CpuFrame& frame,
                                              const GuestMemoryPort& memory,
                                              std::uint64_t& address) noexcept {
    const auto resolved = ResolveGuestAddress(instruction, operand, frame);
    if (!resolved || !memory.read) {
        return std::nullopt;
    }
    address = *resolved;
    if (operand.bit_width == 32) {
        float value{};
        if (!memory.read(address,
                         std::span<std::uint8_t>{
                             reinterpret_cast<std::uint8_t*>(&value), sizeof(value)},
                         memory.context)) {
            return std::nullopt;
        }
        return static_cast<long double>(value);
    }
    if (operand.bit_width == 64) {
        double value{};
        if (!memory.read(address,
                         std::span<std::uint8_t>{
                             reinterpret_cast<std::uint8_t*>(&value), sizeof(value)},
                         memory.context)) {
            return std::nullopt;
        }
        return static_cast<long double>(value);
    }
    return std::nullopt;
}

std::optional<long double> ReadIntegerMemory(const Instruction& instruction,
                                             const Operand& operand,
                                             const CpuFrame& frame,
                                             const GuestMemoryPort& memory,
                                             std::uint64_t& address) noexcept {
    const auto resolved = ResolveGuestAddress(instruction, operand, frame);
    const std::size_t bytes = operand.bit_width / 8;
    if (!resolved || !memory.read || (bytes != 2 && bytes != 4 && bytes != 8)) {
        return std::nullopt;
    }
    address = *resolved;
    std::array<std::uint8_t, 8> data{};
    if (!memory.read(address, std::span<std::uint8_t>{data.data(), bytes},
                     memory.context)) {
        return std::nullopt;
    }
    std::uint64_t raw = 0;
    for (std::size_t index = 0; index < bytes; ++index) {
        raw |= static_cast<std::uint64_t>(data[index]) << (index * 8u);
    }
    const std::uint32_t bits = operand.bit_width;
    if (bits < 64 && (raw & (std::uint64_t{1} << (bits - 1))) != 0) {
        raw |= ~std::uint64_t{} << bits;
    }
    return static_cast<long double>(static_cast<std::int64_t>(raw));
}

bool WriteFloatingMemory(const Instruction& instruction, const Operand& operand,
                         const CpuFrame& frame, const GuestMemoryPort& memory,
                         const long double value, std::uint64_t& address) noexcept {
    const auto resolved = ResolveGuestAddress(instruction, operand, frame);
    if (!resolved || !memory.write) {
        return false;
    }
    address = *resolved;
    if (operand.bit_width == 32) {
        const float narrowed = static_cast<float>(value);
        return memory.write(
            address,
            std::span<const std::uint8_t>{
                reinterpret_cast<const std::uint8_t*>(&narrowed), sizeof(narrowed)},
            memory.context);
    }
    if (operand.bit_width == 64) {
        const double narrowed = static_cast<double>(value);
        return memory.write(
            address,
            std::span<const std::uint8_t>{
                reinterpret_cast<const std::uint8_t*>(&narrowed), sizeof(narrowed)},
            memory.context);
    }
    return false;
}

std::optional<long double> ConstantFor(const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_FLDZ:
        return 0.0L;
    case X86_MNEMONIC_FLD1:
        return 1.0L;
    case X86_MNEMONIC_FLDPI:
        return std::numbers::pi_v<long double>;
    case X86_MNEMONIC_FLDL2E:
        return std::numbers::log2e_v<long double>;
    case X86_MNEMONIC_FLDL2T:
        return std::numbers::log2e_v<long double> *
               std::numbers::ln10_v<long double>;
    case X86_MNEMONIC_FLDLG2:
        return std::numbers::log10e_v<long double> *
               std::numbers::ln2_v<long double>;
    case X86_MNEMONIC_FLDLN2:
        return std::numbers::ln2_v<long double>;
    default:
        return std::nullopt;
    }
}

long double ApplyBinary(const std::uint32_t mnemonic, const long double destination,
                        const long double source) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_FADD:
    case X86_MNEMONIC_FADDP:
        return destination + source;
    case X86_MNEMONIC_FMUL:
    case X86_MNEMONIC_FMULP:
        return destination * source;
    case X86_MNEMONIC_FSUB:
    case X86_MNEMONIC_FSUBP:
        return destination - source;
    case X86_MNEMONIC_FSUBR:
    case X86_MNEMONIC_FSUBRP:
        return source - destination;
    case X86_MNEMONIC_FDIV:
    case X86_MNEMONIC_FDIVP:
        return destination / source;
    case X86_MNEMONIC_FDIVR:
    case X86_MNEMONIC_FDIVRP:
        return source / destination;
    default:
        return destination;
    }
}

bool PopsResult(const std::uint32_t mnemonic) noexcept {
    return mnemonic == X86_MNEMONIC_FADDP || mnemonic == X86_MNEMONIC_FMULP ||
           mnemonic == X86_MNEMONIC_FSUBP || mnemonic == X86_MNEMONIC_FSUBRP ||
           mnemonic == X86_MNEMONIC_FDIVP || mnemonic == X86_MNEMONIC_FDIVRP;
}

bool IsBinary(const std::uint32_t mnemonic) noexcept {
    return mnemonic == X86_MNEMONIC_FADD || mnemonic == X86_MNEMONIC_FADDP ||
           mnemonic == X86_MNEMONIC_FMUL || mnemonic == X86_MNEMONIC_FMULP ||
           mnemonic == X86_MNEMONIC_FSUB || mnemonic == X86_MNEMONIC_FSUBP ||
           mnemonic == X86_MNEMONIC_FSUBR || mnemonic == X86_MNEMONIC_FSUBRP ||
           mnemonic == X86_MNEMONIC_FDIV || mnemonic == X86_MNEMONIC_FDIVP ||
           mnemonic == X86_MNEMONIC_FDIVR || mnemonic == X86_MNEMONIC_FDIVRP;
}

} // namespace

X87SemanticResult ExecuteX87Instruction(const Instruction& instruction,
                                        CpuFrame& frame,
                                        const GuestMemoryPort& memory) noexcept {
    if (instruction.length == 0 ||
        (instruction.prefix_attributes & X86_ATTRIB_HAS_LOCK) != 0) {
        return Unsupported(instruction);
    }
    frame.x87_code_address = instruction.address;
    frame.x87_opcode = static_cast<std::uint16_t>(
        instruction.encoding_bytes[0] |
        (static_cast<std::uint16_t>(instruction.encoding_bytes[1]) << 8));

    if (const auto constant = ConstantFor(instruction.mnemonic)) {
        return PushX87(frame, *constant) == X87StackOutcome::Complete
                   ? X87SemanticResult{X87Stop::Continue,
                                       instruction.address + instruction.length, 0}
                   : StackFaultResult(instruction);
    }
    if (instruction.mnemonic == X86_MNEMONIC_FLD ||
        instruction.mnemonic == X86_MNEMONIC_FILD) {
        if (instruction.operand_count != 1) {
            return Unsupported(instruction);
        }
        std::uint64_t address = 0;
        const auto value = instruction.mnemonic == X86_MNEMONIC_FLD
                               ? ReadFloatingMemory(instruction, instruction.operands[0],
                                                    frame, memory, address)
                               : ReadIntegerMemory(instruction, instruction.operands[0],
                                                   frame, memory, address);
        if (!value) {
            return Fault(instruction, address);
        }
        frame.x87_data_address = address;
        return PushX87(frame, *value) == X87StackOutcome::Complete
                   ? X87SemanticResult{X87Stop::Continue,
                                       instruction.address + instruction.length, 0}
                   : StackFaultResult(instruction);
    }
    if (instruction.mnemonic == X86_MNEMONIC_FSTP) {
        if (instruction.operand_count != 1) {
            return Unsupported(instruction);
        }
        const auto top = ReadX87(frame, 0);
        std::uint64_t address = 0;
        if (!top) {
            return StackFaultResult(instruction);
        }
        if (!WriteFloatingMemory(instruction, instruction.operands[0], frame, memory,
                                 *top, address)) {
            return Fault(instruction, address);
        }
        frame.x87_data_address = address;
        return PopX87(frame) == X87StackOutcome::Complete
                   ? X87SemanticResult{X87Stop::Continue,
                                       instruction.address + instruction.length, 0}
                   : StackFaultResult(instruction);
    }
    if (instruction.mnemonic == X86_MNEMONIC_FXCH) {
        const std::uint8_t logical = instruction.operand_count == 0
                                         ? 1
                                         : StackIndex(instruction.operands[0]).value_or(8);
        return ExchangeX87(frame, logical) == X87StackOutcome::Complete
                   ? X87SemanticResult{X87Stop::Continue,
                                       instruction.address + instruction.length, 0}
                   : StackFaultResult(instruction);
    }
    if (!IsBinary(instruction.mnemonic)) {
        return Unsupported(instruction);
    }

    std::uint8_t destination_index = 0;
    std::uint8_t source_index = PopsResult(instruction.mnemonic) ? 0 : 1;
    std::optional<long double> source;
    std::uint64_t memory_address = 0;
    if (instruction.operand_count == 1 &&
        instruction.operands[0].form == OperandForm::Memory) {
        source = ReadFloatingMemory(instruction, instruction.operands[0], frame, memory,
                                    memory_address);
        source_index = 8;
    } else if (instruction.operand_count >= 2) {
        destination_index = StackIndex(instruction.operands[0]).value_or(8);
        source_index = StackIndex(instruction.operands[1]).value_or(8);
        source = ReadX87(frame, source_index);
    } else if (PopsResult(instruction.mnemonic)) {
        destination_index = 1;
        source_index = 0;
        source = ReadX87(frame, 0);
    }
    const auto destination = ReadX87(frame, destination_index);
    if (!destination || !source) {
        return source_index == 8 && memory_address != 0
                   ? Fault(instruction, memory_address)
                   : StackFaultResult(instruction);
    }
    if (memory_address != 0) {
        frame.x87_data_address = memory_address;
    }
    const long double result =
        ApplyBinary(instruction.mnemonic, *destination, *source);
    if (WriteX87(frame, destination_index, result) != X87StackOutcome::Complete) {
        return StackFaultResult(instruction);
    }
    if (PopsResult(instruction.mnemonic) &&
        PopX87(frame) != X87StackOutcome::Complete) {
        return StackFaultResult(instruction);
    }
    return {X87Stop::Continue, instruction.address + instruction.length, 0};
}

} // namespace Lsx4::Translation
