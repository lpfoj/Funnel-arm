// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/register_access.h"

#include "common/x86_decoder.h"

namespace Lsx4::Translation {

std::optional<IntegerAccess> DecodeIntegerRegister(const Operand& operand) noexcept {
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

} // namespace Lsx4::Translation
