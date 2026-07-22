// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/address_resolver.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/register_access.h"

namespace Lsx4::Translation {
namespace {

std::optional<std::uint64_t> AddressRegister(const std::uint32_t register_id,
                                             const std::uint8_t address_width,
                                             const CpuFrame& frame) noexcept {
    Operand description{};
    description.form = OperandForm::Register;
    description.register_id = register_id;
    description.bit_width = address_width;
    const auto access = DecodeIntegerRegister(description);
    if (!access) {
        return std::nullopt;
    }
    return ReadInteger(frame, static_cast<IntegerRegister>(access->slot), access->width,
                       access->upper_byte);
}

} // namespace

std::optional<std::uint64_t> ResolveGuestAddress(
    const Instruction& instruction, const Operand& operand,
    const CpuFrame& frame) noexcept {
    if (operand.form != OperandForm::Memory) {
        return std::nullopt;
    }
    std::uint64_t address = 0;
    if (operand.address.base == X86_REGISTER_RIP ||
        operand.address.base == X86_REGISTER_EIP) {
        address = instruction.address + instruction.length;
    } else if (operand.address.base != X86_REGISTER_NONE) {
        const auto base =
            AddressRegister(operand.address.base, instruction.address_width, frame);
        if (!base) {
            return std::nullopt;
        }
        address = *base;
    }
    if (operand.address.index != X86_REGISTER_NONE) {
        const auto index =
            AddressRegister(operand.address.index, instruction.address_width, frame);
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
    return instruction.address_width == 32 ? static_cast<std::uint32_t>(address)
                                           : address;
}

} // namespace Lsx4::Translation
