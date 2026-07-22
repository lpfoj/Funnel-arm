// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/retiring_execution_core.h"

#include <cstdint>

#include <xbyak_aarch64/xbyak_aarch64.h>

namespace Executor::Jit::NativeAddress {

enum class SegmentPolicy : std::uint8_t {
    Ignore,
    Apply,
};

struct MachineAddressLayout {
    std::uint32_t gpr_base_offset{};
    std::uint32_t fs_origin_offset{};
    std::uint32_t gs_origin_offset{};
};

[[nodiscard]] bool EmitGuestEffectiveAddress(
    Xbyak_aarch64::CodeGenerator& code, const LsxDecodedOp& instruction,
    const LsxOperandRecord& memory_operand, const XReg& destination,
    const XReg& index_scratch, const XReg& displacement_scratch,
    const MachineAddressLayout& layout, SegmentPolicy segment_policy);

} // namespace Executor::Jit::NativeAddress
