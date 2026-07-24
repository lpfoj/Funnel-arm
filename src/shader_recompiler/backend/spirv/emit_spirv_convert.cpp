// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
Id ExtractU16(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U16, value);
}

Id ExtractS16(EmitContext& ctx, Id value) {
    return ctx.OpSConvert(ctx.S16, ctx.OpBitcast(ctx.S32[1], value));
}

Id ExtractU8(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U8, value);
}

Id ExtractS8(EmitContext& ctx, Id value) {
    return ctx.OpSConvert(ctx.S8, ctx.OpBitcast(ctx.S32[1], value));
}

Id Signed32AsU32(EmitContext& ctx, Id value) {
    return ctx.OpBitcast(ctx.U32[1], value);
}

Id ConvertFToSignedBits32(EmitContext& ctx, Id float_type, Id zero, Id value) {
    const Id is_negative = ctx.OpFOrdLessThan(ctx.U1[1], value, zero);
    const Id magnitude = ctx.OpConvertFToU(ctx.U32[1], ctx.OpFAbs(float_type, value));
    const Id negative = ctx.OpISub(ctx.U32[1], ctx.u32_zero_value, magnitude);
    return ctx.OpSelect(ctx.U32[1], is_negative, negative, magnitude);
}

Id ConvertFToSignedBits64(EmitContext& ctx, Id float_type, Id zero, Id value) {
    const Id is_negative = ctx.OpFOrdLessThan(ctx.U1[1], value, zero);
    const Id magnitude = ctx.OpConvertFToU(ctx.U64, ctx.OpFAbs(float_type, value));
    const Id negative = ctx.OpISub(ctx.U64, ctx.u64_zero_value, magnitude);
    return ctx.OpSelect(ctx.U64, is_negative, negative, magnitude);
}
}

Id EmitConvertS16F32(EmitContext& ctx, Id value) {
    return ConvertFToSignedBits32(ctx, ctx.F32[1], ctx.f32_zero_value, value);
}

Id EmitConvertS16F64(EmitContext& ctx, Id value) {
    const Id zero = ctx.Constant(ctx.F64[1], f64{0.0});
    return ConvertFToSignedBits32(ctx, ctx.F64[1], zero, value);
}

Id EmitConvertS32F32(EmitContext& ctx, Id value) {
    return ConvertFToSignedBits32(ctx, ctx.F32[1], ctx.f32_zero_value, value);
}

Id EmitConvertS32F64(EmitContext& ctx, Id value) {
    const Id zero = ctx.Constant(ctx.F64[1], f64{0.0});
    return ConvertFToSignedBits32(ctx, ctx.F64[1], zero, value);
}

Id EmitConvertS64F32(EmitContext& ctx, Id value) {
    return ConvertFToSignedBits64(ctx, ctx.F32[1], ctx.f32_zero_value, value);
}

Id EmitConvertS64F64(EmitContext& ctx, Id value) {
    const Id zero = ctx.Constant(ctx.F64[1], f64{0.0});
    return ConvertFToSignedBits64(ctx, ctx.F64[1], zero, value);
}

Id EmitConvertU16F32(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U32[1], ctx.OpConvertFToU(ctx.U16, value));
}

Id EmitConvertU16F64(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U32[1], ctx.OpConvertFToU(ctx.U16, value));
}

Id EmitConvertU32F32(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U32[1], value);
}

Id EmitConvertU32F64(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U32[1], value);
}

Id EmitConvertU64F32(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U64, value);
}

Id EmitConvertU64F64(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U64, value);
}

Id EmitConvertU64U32(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U64, value);
}

Id EmitConvertU32U64(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U32[1], value);
}

Id EmitConvertF16F32(EmitContext& ctx, Id value) {
    return ctx.OpFConvert(ctx.F16[1], value);
}

Id EmitConvertF32F16(EmitContext& ctx, Id value) {
    return ctx.OpFConvert(ctx.F32[1], value);
}

Id EmitConvertF32F64(EmitContext& ctx, Id value) {
    return ctx.OpFConvert(ctx.F32[1], value);
}

Id EmitConvertF64F32(EmitContext& ctx, Id value) {
    return ctx.OpFConvert(ctx.F64[1], value);
}

Id EmitConvertF32S8(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F32[1], ExtractS8(ctx, value));
}

Id EmitConvertF32S16(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F32[1], ExtractS16(ctx, value));
}

Id EmitConvertF32S32(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F32[1], ctx.OpBitcast(ctx.S32[1], value));
}

Id EmitConvertF32S64(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F32[1], ctx.OpBitcast(ctx.S64, value));
}

Id EmitConvertF32U8(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F32[1], ExtractU8(ctx, value));
}

Id EmitConvertF32U16(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F32[1], ExtractU16(ctx, value));
}

Id EmitConvertF32U32(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F32[1], value);
}

Id EmitConvertF32U64(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F32[1], value);
}

Id EmitConvertF64S8(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F64[1], ExtractS8(ctx, value));
}

Id EmitConvertF64S16(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F64[1], ExtractS16(ctx, value));
}

Id EmitConvertF64S32(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F64[1], ctx.OpBitcast(ctx.S32[1], value));
}

Id EmitConvertF64S64(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F64[1], ctx.OpBitcast(ctx.S64, value));
}

Id EmitConvertF64U8(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F64[1], ExtractU8(ctx, value));
}

Id EmitConvertF64U16(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F64[1], ExtractU16(ctx, value));
}

Id EmitConvertF64U32(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F64[1], value);
}

Id EmitConvertF64U64(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F64[1], value);
}

Id EmitConvertU16U32(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U16, value);
}

Id EmitConvertU32U16(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U32[1], value);
}

Id EmitConvertU8U32(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U8, value);
}

Id EmitConvertU32U8(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U32[1], value);
}

Id EmitConvertS32S8(EmitContext& ctx, Id value) {
    const Id signed_value = ctx.OpBitcast(ctx.S8, value);
    return Signed32AsU32(ctx, ctx.OpSConvert(ctx.S32[1], signed_value));
}

Id EmitConvertS32S16(EmitContext& ctx, Id value) {
    const Id signed_value = ctx.OpBitcast(ctx.S16, value);
    return Signed32AsU32(ctx, ctx.OpSConvert(ctx.S32[1], signed_value));
}

}
