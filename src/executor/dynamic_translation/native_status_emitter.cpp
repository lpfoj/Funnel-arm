// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/native_status_emitter.h"

#include <array>

namespace Executor::Jit::NativeStatus {
namespace {

// Produces the even-parity bit in W15. W14 is reserved for the caller's
// accumulated x86 status word; W16 is temporary. The nibble lookup avoids
// reproducing the reference XOR-fold implementation.
void EmitEvenParity(Xbyak_aarch64::CodeGenerator& code, const WReg& source) {
    code.mov(code.w15, source);
    code.lsr(code.w16, code.w15, 4);
    code.eor(code.w15, code.w15, code.w16);
    code.and_(code.w15, code.w15, 0xf);
    code.mov(code.w16, 0x9669);
    code.lsrv(code.w15, code.w16, code.w15);
    code.and_(code.w15, code.w15, 1);
}

void PublishStatus(Xbyak_aarch64::CodeGenerator& code,
                   const X86FlagLayout& layout, const std::uint64_t mask) {
    code.mov(code.x15, mask);
    code.ldr(code.x16, Xbyak_aarch64::ptr(code.x0, layout.state_offset));
    code.bic(code.x16, code.x16, code.x15);
    code.orr(code.x16, code.x16, code.x14);
    code.str(code.x16, Xbyak_aarch64::ptr(code.x0, layout.state_offset));
}

void AppendNzv(Xbyak_aarch64::CodeGenerator& code,
                const X86FlagLayout& layout) {
    struct NativeConditionBit {
        Xbyak_aarch64::Cond condition;
        std::uint32_t x86_bit;
    };
    const std::array<NativeConditionBit, 3> status_bits = {{
        {Xbyak_aarch64::EQ, layout.zero_bit},
        {Xbyak_aarch64::MI, layout.sign_bit},
        {Xbyak_aarch64::VS, layout.overflow_bit},
    }};
    for (const auto& status : status_bits) {
        code.cset(code.x15, status.condition);
        code.lsl(code.x15, code.x15, status.x86_bit);
        code.orr(code.x14, code.x14, code.x15);
    }
}

void AppendParity(Xbyak_aarch64::CodeGenerator& code, const WReg& result,
                  const X86FlagLayout& layout) {
    EmitEvenParity(code, result);
    code.lsl(code.x15, code.x15, layout.parity_bit);
    code.orr(code.x14, code.x14, code.x15);
}

void AppendAdjust64(Xbyak_aarch64::CodeGenerator& code, const XReg& left,
                    const XReg& right, const XReg& result,
                    const X86FlagLayout& layout) {
    code.eor(code.x15, left, right);
    code.eor(code.x15, code.x15, result);
    code.ubfx(code.x15, code.x15, 4, 1);
    code.lsl(code.x15, code.x15, layout.adjust_bit);
    code.orr(code.x14, code.x14, code.x15);
}

void AppendAdjust32(Xbyak_aarch64::CodeGenerator& code, const WReg& left,
                    const WReg& right, const WReg& result,
                    const X86FlagLayout& layout) {
    code.eor(code.w15, left, right);
    code.eor(code.w15, code.w15, result);
    code.ubfx(code.w15, code.w15, 4, 1);
    code.lsl(code.x15, code.x15, layout.adjust_bit);
    code.orr(code.x14, code.x14, code.x15);
}

} // namespace

void EmitTestByte(Xbyak_aarch64::CodeGenerator& code, const WReg& value,
                  const bool include_sign, const X86FlagLayout& layout) {
    code.mov(code.w14, value);
    EmitEvenParity(code, code.w14);
    code.lsl(code.x15, code.x15, layout.parity_bit);
    if (include_sign) {
        code.lsr(code.w16, code.w14, 7);
        code.and_(code.w16, code.w16, 1);
        code.lsl(code.x16, code.x16, layout.sign_bit);
        code.orr(code.x15, code.x15, code.x16);
    }
    code.and_(code.w16, code.w14, 0xff);
    code.cmp(code.w16, 0);
    code.cset(code.x16, Xbyak_aarch64::EQ);
    code.lsl(code.x16, code.x16, layout.zero_bit);
    code.orr(code.x14, code.x15, code.x16);
    PublishStatus(code, layout, layout.arithmetic_mask);
}

void EmitTest32(Xbyak_aarch64::CodeGenerator& code, const WReg& value,
                const X86FlagLayout& layout) {
    code.mov(code.w14, value);
    EmitEvenParity(code, code.w14);
    code.lsl(code.x15, code.x15, layout.parity_bit);
    code.lsr(code.w16, code.w14, 31);
    code.lsl(code.x16, code.x16, layout.sign_bit);
    code.orr(code.x15, code.x15, code.x16);
    code.cmp(code.w14, 0);
    code.cset(code.x16, Xbyak_aarch64::EQ);
    code.lsl(code.x16, code.x16, layout.zero_bit);
    code.orr(code.x14, code.x15, code.x16);
    PublishStatus(code, layout, layout.arithmetic_mask);
}

void EmitTest64(Xbyak_aarch64::CodeGenerator& code, const XReg& value,
                const X86FlagLayout& layout) {
    code.mov(code.x14, value);
    EmitEvenParity(code, WReg(code.x14.getIdx()));
    code.lsl(code.x15, code.x15, layout.parity_bit);
    code.lsr(code.x16, code.x14, 63);
    code.lsl(code.x16, code.x16, layout.sign_bit);
    code.orr(code.x15, code.x15, code.x16);
    code.cmp(code.x14, 0);
    code.cset(code.x16, Xbyak_aarch64::EQ);
    code.lsl(code.x16, code.x16, layout.zero_bit);
    code.orr(code.x14, code.x15, code.x16);
    PublishStatus(code, layout, layout.arithmetic_mask);
}

void EmitAdd64(Xbyak_aarch64::CodeGenerator& code, const XReg& left,
               const XReg& right, const XReg& result,
               const X86FlagLayout& layout) {
    code.adds(result, left, right);
    code.cset(code.x14, Xbyak_aarch64::CS);
    AppendNzv(code, layout);
    AppendParity(code, WReg(result.getIdx()), layout);
    AppendAdjust64(code, left, right, result, layout);
    PublishStatus(code, layout, layout.arithmetic_mask);
}

void EmitSubtract64(Xbyak_aarch64::CodeGenerator& code, const XReg& left,
                    const XReg& right, const XReg& result,
                    const X86FlagLayout& layout) {
    code.subs(result, left, right);
    code.cset(code.x14, Xbyak_aarch64::CC);
    AppendNzv(code, layout);
    AppendParity(code, WReg(result.getIdx()), layout);
    AppendAdjust64(code, left, right, result, layout);
    PublishStatus(code, layout, layout.arithmetic_mask);
}

void EmitSubtract32(Xbyak_aarch64::CodeGenerator& code, const WReg& left,
                    const WReg& right, const WReg& result,
                    const X86FlagLayout& layout) {
    code.subs(result, left, right);
    code.cset(code.x14, Xbyak_aarch64::CC);
    AppendNzv(code, layout);
    AppendParity(code, result, layout);
    AppendAdjust32(code, left, right, result, layout);
    PublishStatus(code, layout, layout.arithmetic_mask);
}

void EmitSubtract8(Xbyak_aarch64::CodeGenerator& code, const WReg& left,
                   const WReg& right, const WReg& result,
                   const X86FlagLayout& layout) {
    code.sub(result, left, right);
    code.and_(result, result, 0xff);
    code.cmp(left, right);
    code.cset(code.x14, Xbyak_aarch64::CC);
    code.cmp(result, 0);
    code.cset(code.x15, Xbyak_aarch64::EQ);
    code.lsl(code.x15, code.x15, layout.zero_bit);
    code.orr(code.x14, code.x14, code.x15);
    code.lsr(code.w15, result, 7);
    code.lsl(code.x15, code.x15, layout.sign_bit);
    code.orr(code.x14, code.x14, code.x15);
    code.eor(code.w15, left, right);
    code.eor(code.w16, left, result);
    code.and_(code.w15, code.w15, code.w16);
    code.lsr(code.w15, code.w15, 7);
    code.and_(code.w15, code.w15, 1);
    code.lsl(code.x15, code.x15, layout.overflow_bit);
    code.orr(code.x14, code.x14, code.x15);
    AppendAdjust32(code, left, right, result, layout);
    AppendParity(code, result, layout);
    PublishStatus(code, layout, layout.arithmetic_mask);
}

void EmitIncrement64(Xbyak_aarch64::CodeGenerator& code, const XReg& input,
                     const XReg& result, const X86FlagLayout& layout) {
    code.adds(result, input, 1);
    code.mov(code.x14, 0);
    AppendNzv(code, layout);
    AppendParity(code, WReg(result.getIdx()), layout);
    code.mov(code.x16, 1);
    AppendAdjust64(code, input, code.x16, result, layout);
    PublishStatus(code, layout, layout.arithmetic_mask & ~std::uint64_t{1});
}

void EmitDecrement64(Xbyak_aarch64::CodeGenerator& code, const XReg& input,
                     const XReg& result, const X86FlagLayout& layout) {
    code.subs(result, input, 1);
    code.mov(code.x14, 0);
    AppendNzv(code, layout);
    AppendParity(code, WReg(result.getIdx()), layout);
    code.mov(code.x16, 1);
    AppendAdjust64(code, input, code.x16, result, layout);
    PublishStatus(code, layout, layout.arithmetic_mask & ~std::uint64_t{1});
}

void EmitIncrement32(Xbyak_aarch64::CodeGenerator& code, const WReg& input,
                     const WReg& result, const X86FlagLayout& layout) {
    code.adds(result, input, 1);
    code.mov(code.x14, 0);
    AppendNzv(code, layout);
    AppendParity(code, result, layout);
    code.mov(code.w16, 1);
    AppendAdjust32(code, input, code.w16, result, layout);
    PublishStatus(code, layout, layout.arithmetic_mask & ~std::uint64_t{1});
}

} // namespace Executor::Jit::NativeStatus
