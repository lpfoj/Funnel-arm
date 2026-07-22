// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/instruction_diagnostics.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/retiring_execution_core.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>

namespace Lsx4::Translation {
namespace {

const char* RegisterLabel(const std::uint32_t code) noexcept {
    const char* label = X86RegisterGetString(code);
    return label == nullptr ? "?" : label;
}

} // namespace

std::string RenderInstructionEvent(const Executor::Jit::LsxDecodedOp& instruction) {
    std::ostringstream event;
    const char* operation =
        X86MnemonicGetString(static_cast<X86Mnemonic>(instruction.mnemonic));
    event << "{pc=" << std::hex << instruction.guest_rip
          << ",op=" << (operation == nullptr ? "?" : operation)
          << ",raw=";
    for (std::size_t index = 0;
         index < std::min<std::size_t>(instruction.length, instruction.bytes.size());
         ++index) {
        event << std::setw(2) << std::setfill('0')
              << static_cast<unsigned>(instruction.bytes[index]);
    }
    event << std::dec << ",args=[";
    const std::size_t count =
        std::min<std::size_t>(instruction.operand_count, instruction.operands.size());
    for (std::size_t index = 0; index < count; ++index) {
        if (index != 0) {
            event << '|';
        }
        const auto& argument = instruction.operands[index];
        event << static_cast<unsigned>(argument.type) << '/' << argument.size;
        if (argument.type == X86_OPERAND_TYPE_REGISTER) {
            event << "@" << RegisterLabel(argument.reg.value);
        } else if (argument.type == X86_OPERAND_TYPE_MEMORY) {
            event << "@[" << RegisterLabel(argument.mem.segment) << ':'
                  << RegisterLabel(argument.mem.base) << '+'
                  << RegisterLabel(argument.mem.index) << '*'
                  << static_cast<unsigned>(argument.mem.scale) << '+'
                  << std::showbase << std::hex << argument.mem.disp.value
                  << std::noshowbase << std::dec << ']';
        } else if (argument.type == X86_OPERAND_TYPE_IMMEDIATE) {
            event << "@" << std::showbase << std::hex
                  << (argument.imm.is_signed
                          ? static_cast<std::uint64_t>(argument.imm.value.s)
                          : argument.imm.value.u)
                  << std::noshowbase << std::dec;
        }
    }
    event << "]}";
    return event.str();
}

std::string RenderRegisterEvent(const Executor::Jit::LsxMachineImage& state) {
    static constexpr std::array<const char*, 16> labels = {
        "a", "c", "d", "b", "sp", "bp", "si", "di",
        "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15",
    };
    std::ostringstream event;
    event << " registers={condition:" << std::showbase << std::hex << state.rflags;
    for (std::size_t index = 0; index < state.gpr.size(); ++index) {
        event << ',' << labels[index] << ':' << state.gpr[index];
    }
    event << ",fs:" << state.fs_base << ",gs:" << state.gs_base
          << std::noshowbase << std::dec << '}';
    return event.str();
}

} // namespace Lsx4::Translation
