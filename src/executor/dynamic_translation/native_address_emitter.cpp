// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/native_address_emitter.h"

#include "common/x86_decoder.h"

#include <array>

namespace Executor::Jit::NativeAddress {
namespace {

constexpr std::uint8_t kInvalidShift = 0xff;

[[nodiscard]] std::uint8_t ScaleShift(const std::uint8_t scale) noexcept {
    constexpr std::array<std::uint8_t, 9> shifts = {
        0, 0, 1, kInvalidShift, 2, kInvalidShift, kInvalidShift, kInvalidShift, 3,
    };
    return scale < shifts.size() ? shifts[scale] : kInvalidShift;
}

[[nodiscard]] bool LoadAddressRegister(Xbyak_aarch64::CodeGenerator& code,
                                       const LsxRegisterCode guest_register,
                                       const XReg& destination,
                                       const MachineAddressLayout& layout) {
    if (guest_register < X86_REGISTER_RAX || guest_register > X86_REGISTER_R15) {
        return false;
    }
    const auto ordinal = static_cast<std::uint32_t>(guest_register - X86_REGISTER_RAX);
    const auto offset = static_cast<std::uint32_t>(
        layout.gpr_base_offset + ordinal * sizeof(std::uint64_t));
    code.ldr(destination, Xbyak_aarch64::ptr(code.x0, offset));
    return true;
}

} // namespace

bool EmitGuestEffectiveAddress(Xbyak_aarch64::CodeGenerator& code,
                               const LsxDecodedOp& instruction,
                               const LsxOperandRecord& memory_operand,
                               const XReg& destination,
                               const XReg& index_scratch,
                               const XReg& displacement_scratch,
                               const MachineAddressLayout& layout,
                               const SegmentPolicy segment_policy) {
    if (memory_operand.type != X86_OPERAND_TYPE_MEMORY) {
        return false;
    }

    const auto& address = memory_operand.mem;
    const bool instruction_relative = address.base == X86_REGISTER_RIP ||
                                      address.base == X86_REGISTER_EIP;
    if (instruction_relative) {
        const auto absolute = instruction.guest_rip + instruction.length +
                              static_cast<std::uint64_t>(address.disp.value);
        code.mov(destination, absolute);
    } else {
        if (address.base == X86_REGISTER_NONE) {
            code.mov(destination, 0);
        } else if (!LoadAddressRegister(code, address.base, destination, layout)) {
            return false;
        }
        if (address.disp.value != 0) {
            code.mov(displacement_scratch,
                     static_cast<std::uint64_t>(address.disp.value));
            code.add(destination, destination, displacement_scratch);
        }
    }

    if (address.index != X86_REGISTER_NONE) {
        const std::uint8_t shift = ScaleShift(address.scale);
        if (shift == kInvalidShift ||
            !LoadAddressRegister(code, address.index, index_scratch, layout)) {
            return false;
        }
        if (shift != 0) {
            code.lsl(index_scratch, index_scratch, shift);
        }
        code.add(destination, destination, index_scratch);
    }

    if (segment_policy == SegmentPolicy::Apply &&
        (address.segment == X86_REGISTER_FS || address.segment == X86_REGISTER_GS)) {
        const std::uint32_t origin_offset = address.segment == X86_REGISTER_FS
                                                ? layout.fs_origin_offset
                                                : layout.gs_origin_offset;
        code.ldr(displacement_scratch,
                 Xbyak_aarch64::ptr(code.x0, origin_offset));
        code.add(destination, destination, displacement_scratch);
    }
    return true;
}

} // namespace Executor::Jit::NativeAddress
