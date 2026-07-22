// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

#include <xbyak_aarch64/xbyak_aarch64.h>

namespace Executor::Jit::NativeStatus {

struct X86FlagLayout {
    std::uint32_t state_offset{};
    std::uint32_t parity_bit{};
    std::uint32_t adjust_bit{};
    std::uint32_t zero_bit{};
    std::uint32_t sign_bit{};
    std::uint32_t overflow_bit{};
    std::uint64_t arithmetic_mask{};
};

void EmitTestByte(Xbyak_aarch64::CodeGenerator& code, const WReg& value,
                  bool include_sign, const X86FlagLayout& layout);
void EmitTest32(Xbyak_aarch64::CodeGenerator& code, const WReg& value,
                const X86FlagLayout& layout);
void EmitTest64(Xbyak_aarch64::CodeGenerator& code, const XReg& value,
                const X86FlagLayout& layout);
void EmitAdd64(Xbyak_aarch64::CodeGenerator& code, const XReg& left,
               const XReg& right, const XReg& result,
               const X86FlagLayout& layout);
void EmitSubtract32(Xbyak_aarch64::CodeGenerator& code, const WReg& left,
                    const WReg& right, const WReg& result,
                    const X86FlagLayout& layout);
void EmitSubtract8(Xbyak_aarch64::CodeGenerator& code, const WReg& left,
                   const WReg& right, const WReg& result,
                   const X86FlagLayout& layout);
void EmitIncrement64(Xbyak_aarch64::CodeGenerator& code, const XReg& input,
                     const XReg& result, const X86FlagLayout& layout);
void EmitDecrement64(Xbyak_aarch64::CodeGenerator& code, const XReg& input,
                     const XReg& result, const X86FlagLayout& layout);
void EmitIncrement32(Xbyak_aarch64::CodeGenerator& code, const WReg& input,
                     const WReg& result, const X86FlagLayout& layout);

} // namespace Executor::Jit::NativeStatus
