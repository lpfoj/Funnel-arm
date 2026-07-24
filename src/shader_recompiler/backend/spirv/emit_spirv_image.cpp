// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/static_vector.hpp>
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

Id SignedImageScalar(EmitContext& ctx, const IR::Value& ir_value, Id value) {
    if (!Sirit::ValidId(value)) {
        return value;
    }
    if (ir_value.IsImmediate()) {
        return ctx.ConstS32(static_cast<s32>(ir_value.U32()));
    }
    return ctx.OpBitcast(ctx.S32[1], value);
}

Id SignedImageOffset(EmitContext& ctx, const IR::Value& offset) {
    const Id value = ctx.Def(offset);
    switch (offset.Type()) {
    case IR::Type::U32:
        return ctx.OpBitcast(ctx.S32[1], value);
    case IR::Type::U32x2:
        return ctx.OpBitcast(ctx.S32[2], value);
    case IR::Type::U32x3:
        return ctx.OpBitcast(ctx.S32[3], value);
    default:
        UNREACHABLE_MSG("Invalid runtime image offset type {}", offset.Type());
    }
}

struct ImageOperands {
    [[nodiscard]] bool Empty() const {
        return static_cast<u32>(mask) == 0;
    }

    void Add(spv::ImageOperandsMask new_mask, Id value) {
        if (!Sirit::ValidId(value)) {
            return;
        }
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value);
    }
    void Add(spv::ImageOperandsMask new_mask, Id value1, Id value2) {
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value1);
        operands.push_back(value2);
    }

    void AddOffset(EmitContext& ctx, const IR::Value& offset,
                   bool can_use_runtime_offsets = false) {
        if (offset.IsEmpty()) {
            return;
        }
        if (offset.IsImmediate()) {
            const s32 operand = offset.U32();
            Add(spv::ImageOperandsMask::ConstOffset, ctx.ConstS32(operand));
            return;
        }
        IR::Inst* const inst{offset.InstRecursive()};
        if (inst->AreAllArgsImmediates()) {
            switch (inst->GetOpcode()) {
            case IR::Opcode::CompositeConstructU32x2:
                Add(spv::ImageOperandsMask::ConstOffset,
                    ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                 static_cast<s32>(inst->Arg(1).U32())));
                return;
            case IR::Opcode::CompositeConstructU32x3:
                Add(spv::ImageOperandsMask::ConstOffset,
                    ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                 static_cast<s32>(inst->Arg(1).U32()),
                                 static_cast<s32>(inst->Arg(2).U32())));
                return;
            default:
                break;
            }
        }
        if (can_use_runtime_offsets) {
            Add(spv::ImageOperandsMask::Offset, SignedImageOffset(ctx, offset));
        } else {
            LOG_WARNING(Render_Vulkan,
                        "Runtime offset provided to unsupported image sample instruction");
        }
    }

    void AddDerivatives(EmitContext& ctx, Id derivatives_dx, Id derivatives_dy) {
        if (!Sirit::ValidId(derivatives_dx) || !Sirit::ValidId(derivatives_dy)) {
            return;
        }
        Add(spv::ImageOperandsMask::Grad, derivatives_dx, derivatives_dy);
    }

    spv::ImageOperandsMask mask{};
    boost::container::static_vector<Id, 4> operands;
};

namespace {

u32 TexelCoordinateCount(AmdGpu::ImageType view_type) {
    switch (view_type) {
    case AmdGpu::ImageType::Color1D:
        return 1;
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return 2;
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaaArray:
    case AmdGpu::ImageType::Color3D:
        return 3;
    default:
        return 0;
    }
}

struct CoordinateVectorLane {
    Id* built_in_vector{};
    IR::Inst* composite_vector{};
    IR::Inst* scalar_inst{};
    u32 width{};
    u32 component{};

    [[nodiscard]] bool HasSource() const {
        return built_in_vector || composite_vector;
    }

    [[nodiscard]] Id VectorId() const {
        return built_in_vector ? *built_in_vector : composite_vector->Definition<Id>();
    }

    [[nodiscard]] bool HasSameSource(const CoordinateVectorLane& other) const {
        return built_in_vector == other.built_in_vector &&
               composite_vector == other.composite_vector && width == other.width;
    }
};

CoordinateVectorLane GetAttributeVectorLane(EmitContext& ctx, const IR::Value& value) {
    IR::Inst* const inst = value.TryInstRecursive();
    if (!inst || inst->GetOpcode() != IR::Opcode::GetAttribute ||
        inst->Arg(0).Type() != IR::Type::Attribute || !inst->Arg(1).IsImmediate()) {
        return {};
    }

    const IR::Attribute attr = inst->Arg(0).Attribute();
    u32 component = inst->Arg(1).U32();
    Id* vector{};
    u32 width{};
    if (IR::IsParam(attr)) {
        const u32 param_index = u32(attr) - u32(IR::Attribute::Param0);
        auto& param = ctx.input_params.at(param_index);
        if (param.is_array || param.is_loaded || param.is_integer || param.num_components < 2 ||
            component >= param.num_components) {
            return {};
        }
        vector = &ctx.input_param_values[param_index];
        width = param.num_components;
        if (!Sirit::ValidId(*vector)) {
            *vector = ctx.OpLoad(ctx.F32[width], param.id);
        }
        return CoordinateVectorLane{vector, nullptr, inst, width, component};
    }
    switch (attr) {
    case IR::Attribute::FragCoord:
        vector = &ctx.frag_coord_value;
        width = 4;
        break;
    case IR::Attribute::TessellationEvaluationPointU:
        vector = &ctx.tess_coord_value;
        width = 3;
        component = 0;
        break;
    case IR::Attribute::TessellationEvaluationPointV:
        vector = &ctx.tess_coord_value;
        width = 3;
        component = 1;
        break;
    case IR::Attribute::BaryCoordSmooth:
        vector = &ctx.bary_coord_smooth_value;
        break;
    case IR::Attribute::BaryCoordSmoothCentroid:
        vector = &ctx.bary_coord_smooth_centroid_value;
        break;
    case IR::Attribute::BaryCoordSmoothSample:
        vector = &ctx.bary_coord_smooth_sample_value;
        break;
    case IR::Attribute::BaryCoordNoPersp:
        vector = &ctx.bary_coord_nopersp_value;
        break;
    case IR::Attribute::BaryCoordNoPerspCentroid:
        vector = &ctx.bary_coord_nopersp_centroid_value;
        break;
    case IR::Attribute::BaryCoordNoPerspSample:
        vector = &ctx.bary_coord_nopersp_sample_value;
        break;
    case IR::Attribute::BaryCoordPullModel:
        vector = &ctx.bary_coord_pull_model_value;
        width = 3;
        break;
    default:
        return {};
    }

    if (IR::IsBarycentricCoord(attr)) {
        width = ctx.profile.supports_amd_shader_explicit_vertex_parameter ? 2 : 3;
        if (!ctx.profile.supports_amd_shader_explicit_vertex_parameter &&
            ctx.profile.supports_fragment_shader_barycentric) {
            ++component;
        }
    }
    return vector && Sirit::ValidId(*vector) && component < width
               ? CoordinateVectorLane{vector, nullptr, inst, width, component}
               : CoordinateVectorLane{};
}

CoordinateVectorLane GetCompositeVectorLane(const IR::Value& value) {
    IR::Inst* const inst = value.TryInstRecursive();
    if (!inst) {
        return {};
    }

    u32 width{};
    switch (inst->GetOpcode()) {
    case IR::Opcode::CompositeExtractF32x2:
        width = 2;
        break;
    case IR::Opcode::CompositeExtractF32x3:
        width = 3;
        break;
    case IR::Opcode::CompositeExtractF32x4:
        width = 4;
        break;
    default:
        return {};
    }
    if (!inst->Arg(1).IsImmediate()) {
        return {};
    }
    const u32 component = inst->Arg(1).U32();
    IR::Inst* const vector = inst->Arg(0).TryInstRecursive();
    return vector && component < width
               ? CoordinateVectorLane{nullptr, vector, inst, width, component}
               : CoordinateVectorLane{};
}

bool TryGetCoordinateVectorLanes(EmitContext& ctx, const std::array<IR::Value, 3>& inputs,
                                 u32 num_components,
                                 std::array<CoordinateVectorLane, 3>& lanes) {
    for (u32 component = 0; component < num_components; ++component) {
        lanes[component] = GetAttributeVectorLane(ctx, inputs[component]);
        if (!lanes[component].HasSource()) {
            lanes[component] = GetCompositeVectorLane(inputs[component]);
        }
        if (!lanes[component].HasSource() ||
            (component > 0 && !lanes[component].HasSameSource(lanes[0]))) {
            return false;
        }
    }
    return true;
}

Id EmitCoordinateVectorShuffle(EmitContext& ctx,
                               const std::array<CoordinateVectorLane, 3>& lanes,
                               u32 num_components) {
    const Id vector = lanes[0].VectorId();
    if (!Sirit::ValidId(vector)) {
        return {};
    }
    return num_components == 2
               ? ctx.OpVectorShuffle(ctx.F32[2], vector, vector,
                                     lanes[0].component, lanes[1].component)
               : ctx.OpVectorShuffle(ctx.F32[3], vector, vector,
                                     lanes[0].component, lanes[1].component,
                                     lanes[2].component);
}

struct VectorSignedCoordinateMatch {
    IR::Inst* construct{};
    std::array<IR::Inst*, 3> converts{};
    std::array<IR::Inst*, 3> floors{};
    std::array<IR::Value, 3> inputs{};
    bool all_inputs_are_floor{};
};

struct VectorFloatCoordinateMatch {
    IR::Inst* construct{};
    std::array<IR::Value, 3> inputs{};
};

bool MatchVectorFloatCoordinates(const IR::Value& coords, u32 num_components,
                                 VectorFloatCoordinateMatch& match) {
    if (num_components < 2 || num_components > 3) {
        return false;
    }
    IR::Inst* const construct = coords.TryInstRecursive();
    const IR::Opcode expected_construct = num_components == 2
                                              ? IR::Opcode::CompositeConstructF32x2
                                              : IR::Opcode::CompositeConstructF32x3;
    if (!construct || construct->GetOpcode() != expected_construct) {
        return false;
    }
    match.construct = construct;
    for (u32 component = 0; component < num_components; ++component) {
        match.inputs[component] = construct->Arg(component);
    }
    return true;
}

u32 ZippedFloatArity(IR::Opcode opcode) {
    switch (opcode) {
    case IR::Opcode::FPAbs32:
    case IR::Opcode::FPNeg32:
    case IR::Opcode::FPSaturate32:
    case IR::Opcode::FPRoundEven32:
    case IR::Opcode::FPFloor32:
    case IR::Opcode::FPCeil32:
    case IR::Opcode::FPTrunc32:
    case IR::Opcode::FPFract32:
        return 1;
    case IR::Opcode::FPAdd32:
    case IR::Opcode::FPSub32:
    case IR::Opcode::FPMul32:
    case IR::Opcode::FPDiv32:
        return 2;
    case IR::Opcode::FPFma32:
    case IR::Opcode::FPClamp32:
        return 3;
    default:
        return 0;
    }
}

bool MatchZippedFloatOperation(const std::array<IR::Value, 3>& values, u32 num_components,
                               IR::Opcode& opcode, u32& arity,
                               std::array<IR::Inst*, 3>& scalar_insts) {
    if (num_components < 2 || num_components > 3) {
        return false;
    }
    for (u32 component = 0; component < num_components; ++component) {
        scalar_insts[component] = values[component].TryInstRecursive();
        if (!scalar_insts[component]) {
            return false;
        }
        if (component == 0) {
            opcode = scalar_insts[component]->GetOpcode();
            arity = ZippedFloatArity(opcode);
            if (arity == 0) {
                return false;
            }
        } else if (scalar_insts[component]->GetOpcode() != opcode) {
            return false;
        }
    }
    return true;
}

Id ConstructFloatVector(EmitContext& ctx, const std::array<IR::Value, 3>& values,
                        u32 num_components) {
    return num_components == 2
               ? ctx.OpCompositeConstruct(ctx.F32[2], ctx.Def(values[0]), ctx.Def(values[1]))
               : ctx.OpCompositeConstruct(ctx.F32[3], ctx.Def(values[0]), ctx.Def(values[1]),
                                          ctx.Def(values[2]));
}

void AppendIdentityInstructions(IR::Value value, std::vector<IR::Inst*>& instructions);

Id TryEmitZippedFloatExpression(EmitContext& ctx, const std::array<IR::Value, 3>& values,
                                u32 num_components) {
    std::array<CoordinateVectorLane, 3> lanes{};
    if (TryGetCoordinateVectorLanes(ctx, values, num_components, lanes)) {
        return EmitCoordinateVectorShuffle(ctx, lanes, num_components);
    }

    IR::Opcode opcode{};
    u32 arity{};
    std::array<IR::Inst*, 3> scalar_insts{};
    if (!MatchZippedFloatOperation(values, num_components, opcode, arity, scalar_insts)) {
        return {};
    }

    std::array<Id, 3> args{};
    for (u32 arg = 0; arg < arity; ++arg) {
        std::array<IR::Value, 3> arg_values{};
        for (u32 component = 0; component < num_components; ++component) {
            arg_values[component] = scalar_insts[component]->Arg(arg);
        }
        args[arg] = TryEmitZippedFloatExpression(ctx, arg_values, num_components);
        if (!Sirit::ValidId(args[arg])) {
            args[arg] = ConstructFloatVector(ctx, arg_values, num_components);
        }
    }

    const Id type = ctx.F32[num_components];
    const auto no_contraction = [&](Id result) {
        ctx.Decorate(result, spv::Decoration::NoContraction);
        return result;
    };
    switch (opcode) {
    case IR::Opcode::FPAbs32:
        return ctx.OpFAbs(type, args[0]);
    case IR::Opcode::FPAdd32:
        return no_contraction(ctx.OpFAdd(type, args[0], args[1]));
    case IR::Opcode::FPSub32:
        return no_contraction(ctx.OpFSub(type, args[0], args[1]));
    case IR::Opcode::FPFma32:
        return no_contraction(ctx.OpFma(type, args[0], args[1], args[2]));
    case IR::Opcode::FPMul32:
        return no_contraction(ctx.OpFMul(type, args[0], args[1]));
    case IR::Opcode::FPDiv32:
        return no_contraction(ctx.OpFDiv(type, args[0], args[1]));
    case IR::Opcode::FPNeg32:
        return ctx.OpFNegate(type, args[0]);
    case IR::Opcode::FPSaturate32:
        return num_components == 2
                   ? ctx.OpFClamp(type, args[0], ctx.ConstF32(0.0f, 0.0f),
                                  ctx.ConstF32(1.0f, 1.0f))
                   : ctx.OpFClamp(type, args[0], ctx.ConstF32(0.0f, 0.0f, 0.0f),
                                  ctx.ConstF32(1.0f, 1.0f, 1.0f));
    case IR::Opcode::FPClamp32:
        return ctx.OpFClamp(type, args[0], args[1], args[2]);
    case IR::Opcode::FPRoundEven32:
        return ctx.OpRoundEven(type, args[0]);
    case IR::Opcode::FPFloor32:
        return ctx.OpFloor(type, args[0]);
    case IR::Opcode::FPCeil32:
        return ctx.OpCeil(type, args[0]);
    case IR::Opcode::FPTrunc32:
        return ctx.OpTrunc(type, args[0]);
    case IR::Opcode::FPFract32:
        return ctx.OpFract(type, args[0]);
    default:
        return {};
    }
}

bool CollectZippedFloatExpression(EmitContext& ctx, const std::array<IR::Value, 3>& values,
                                  u32 num_components,
                                  std::vector<IR::Inst*>& instructions) {
    std::array<CoordinateVectorLane, 3> lanes{};
    if (TryGetCoordinateVectorLanes(ctx, values, num_components, lanes)) {
        for (u32 component = 0; component < num_components; ++component) {
            AppendIdentityInstructions(values[component], instructions);
            instructions.push_back(lanes[component].scalar_inst);
        }
        return true;
    }

    IR::Opcode opcode{};
    u32 arity{};
    std::array<IR::Inst*, 3> scalar_insts{};
    if (!MatchZippedFloatOperation(values, num_components, opcode, arity, scalar_insts)) {
        return false;
    }
    for (u32 component = 0; component < num_components; ++component) {
        AppendIdentityInstructions(values[component], instructions);
        instructions.push_back(scalar_insts[component]);
    }
    for (u32 arg = 0; arg < arity; ++arg) {
        std::array<IR::Value, 3> arg_values{};
        for (u32 component = 0; component < num_components; ++component) {
            arg_values[component] = scalar_insts[component]->Arg(arg);
        }
        CollectZippedFloatExpression(ctx, arg_values, num_components, instructions);
    }
    return true;
}

Id TryEmitVectorFloatCoordinates(EmitContext& ctx, const IR::Value& coords,
                                 u32 num_components) {
    VectorFloatCoordinateMatch match{};
    if (!MatchVectorFloatCoordinates(coords, num_components, match)) {
        return {};
    }
    return TryEmitZippedFloatExpression(ctx, match.inputs, num_components);
}

bool MatchVectorSignedCoordinates(const IR::Value& coords, u32 num_components,
                                  VectorSignedCoordinateMatch& match) {
    if (num_components < 2 || num_components > 3) {
        return false;
    }

    IR::Inst* const construct = coords.TryInstRecursive();
    const IR::Opcode expected_construct = num_components == 2
                                              ? IR::Opcode::CompositeConstructU32x2
                                              : IR::Opcode::CompositeConstructU32x3;
    if (!construct || construct->GetOpcode() != expected_construct) {
        return false;
    }

    std::array<IR::Value, 3> convert_inputs{};
    std::array<IR::Value, 3> floor_inputs{};
    bool all_inputs_are_floor = true;
    for (u32 component = 0; component < num_components; ++component) {
        IR::Inst* const convert = construct->Arg(component).TryInstRecursive();
        if (!convert || convert->GetOpcode() != IR::Opcode::ConvertS32F32) {
            return false;
        }
        match.converts[component] = convert;
        convert_inputs[component] = convert->Arg(0);
        IR::Inst* const floor = convert_inputs[component].TryInstRecursive();
        if (!floor || floor->GetOpcode() != IR::Opcode::FPFloor32) {
            all_inputs_are_floor = false;
            continue;
        }
        match.floors[component] = floor;
        floor_inputs[component] = floor->Arg(0);
    }

    match.construct = construct;
    match.all_inputs_are_floor = all_inputs_are_floor;
    match.inputs = all_inputs_are_floor ? floor_inputs : convert_inputs;
    return true;
}

Id TryEmitVectorSignedCoordinates(EmitContext& ctx, const IR::Value& coords,
                                  u32 num_components) {
    VectorSignedCoordinateMatch match{};
    if (!MatchVectorSignedCoordinates(coords, num_components, match)) {
        return {};
    }

    std::array<CoordinateVectorLane, 3> lanes{};
    Id float_coords{};
    if (TryGetCoordinateVectorLanes(ctx, match.inputs, num_components, lanes)) {
        float_coords = EmitCoordinateVectorShuffle(ctx, lanes, num_components);
    }
    if (!Sirit::ValidId(float_coords)) {
        float_coords =
            num_components == 2
                ? ctx.OpCompositeConstruct(ctx.F32[2], ctx.Def(match.inputs[0]),
                                           ctx.Def(match.inputs[1]))
                : ctx.OpCompositeConstruct(ctx.F32[3], ctx.Def(match.inputs[0]),
                                           ctx.Def(match.inputs[1]), ctx.Def(match.inputs[2]));
    }
    const Id rounded_coords = match.all_inputs_are_floor
                                  ? ctx.OpFloor(ctx.F32[num_components], float_coords)
                                  : float_coords;
    return ctx.OpConvertFToS(ctx.S32[num_components], rounded_coords);
}

void AppendIdentityInstructions(IR::Value value, std::vector<IR::Inst*>& instructions) {
    while (value.IsIdentity()) {
        IR::Inst* const identity = value.Inst();
        instructions.push_back(identity);
        value = identity->Arg(0);
    }
}

Id LoadedSampledImage(EmitContext& ctx, u32 index) {
    auto& texture = ctx.images[index];
    ASSERT(!texture.is_storage);
    ASSERT(texture.mip_fallback_mode != MipStorageFallbackMode::DynamicIndex);
    if (!Sirit::ValidId(texture.value)) {
        texture.value = ctx.OpLoad(texture.image_type, texture.id);
    }
    return texture.value;
}

Id LoadedSampler(EmitContext& ctx, u32 index) {
    ASSERT(index < ctx.sampler_values.size());
    Id& sampler = ctx.sampler_values[index];
    if (!Sirit::ValidId(sampler)) {
        sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[index]);
    }
    return sampler;
}

}

bool CollectAbsorbedVectorSignedCoordinateChain(EmitContext& ctx,
                                                AmdGpu::ImageType view_type,
                                                const IR::Value& ir_coords,
                                                std::vector<IR::Inst*>& instructions) {
    const u32 num_components = TexelCoordinateCount(view_type);
    VectorSignedCoordinateMatch match{};
    if (!MatchVectorSignedCoordinates(ir_coords, num_components, match)) {
        return false;
    }

    std::array<CoordinateVectorLane, 3> lanes{};
    if (!TryGetCoordinateVectorLanes(ctx, match.inputs, num_components, lanes)) {
        return false;
    }

    AppendIdentityInstructions(ir_coords, instructions);
    instructions.push_back(match.construct);
    for (u32 component = 0; component < num_components; ++component) {
        const IR::Value convert_value = match.construct->Arg(component);
        AppendIdentityInstructions(convert_value, instructions);
        instructions.push_back(match.converts[component]);

        if (match.all_inputs_are_floor) {
            const IR::Value floor_value = match.converts[component]->Arg(0);
            AppendIdentityInstructions(floor_value, instructions);
            instructions.push_back(match.floors[component]);
        }

        AppendIdentityInstructions(match.inputs[component], instructions);
        instructions.push_back(lanes[component].scalar_inst);
    }
    return true;
}

bool CollectAbsorbedVectorFloatCoordinateChain(EmitContext& ctx,
                                               AmdGpu::ImageType view_type,
                                               const IR::Value& ir_coords,
                                               std::vector<IR::Inst*>& instructions) {
    const u32 num_components = TexelCoordinateCount(view_type);
    VectorFloatCoordinateMatch match{};
    if (!MatchVectorFloatCoordinates(ir_coords, num_components, match)) {
        return false;
    }
    if (!CollectZippedFloatExpression(ctx, match.inputs, num_components, instructions)) {
        return false;
    }

    AppendIdentityInstructions(ir_coords, instructions);
    instructions.push_back(match.construct);
    return true;
}

Id FloatSampleCoordinates(EmitContext& ctx, AmdGpu::ImageType view_type,
                          const IR::Value& ir_coords, Id coords) {
    const u32 num_components = TexelCoordinateCount(view_type);
    if (const Id vector_coords =
            TryEmitVectorFloatCoordinates(ctx, ir_coords, num_components);
        Sirit::ValidId(vector_coords)) {
        return vector_coords;
    }
    return coords;
}

Id SignedTexelCoordinates(EmitContext& ctx, AmdGpu::ImageType view_type,
                          const IR::Value& ir_coords, Id coords) {
    const u32 num_components = TexelCoordinateCount(view_type);
    if (const Id vector_coords =
            TryEmitVectorSignedCoordinates(ctx, ir_coords, num_components);
        Sirit::ValidId(vector_coords)) {
        return vector_coords;
    }
    switch (view_type) {
    case AmdGpu::ImageType::Color1D:
        return ctx.OpBitcast(ctx.S32[1], coords);
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return ctx.OpBitcast(ctx.S32[2], coords);
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaaArray:
    case AmdGpu::ImageType::Color3D:
        return ctx.OpBitcast(ctx.S32[3], coords);
    default:
        return coords;
    }
}

Id EmitImageSampleRaw(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address1, Id address2,
                      Id address3, Id address4) {
    UNREACHABLE_MSG("Unreachable instruction");
}

Id EmitImageSampleImplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id bias,
                              const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    coords = FloatSampleCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = LoadedSampler(ctx, handle >> 16);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, offset);
    const Id sample = operands.Empty()
                          ? ctx.OpImageSampleImplicitLod(result_type, sampled_image, coords)
                          : ctx.OpImageSampleImplicitLod(result_type, sampled_image, coords,
                                                         operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageSampleExplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod,
                              const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    coords = FloatSampleCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = LoadedSampler(ctx, handle >> 16);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageSampleDrefImplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id dref,
                                  Id bias, const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    coords = FloatSampleCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
    const Id result_type = texture.data_types->Get(1);
    const Id sampler = LoadedSampler(ctx, handle >> 16);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, offset);
    const Id sample = operands.Empty()
                          ? ctx.OpImageSampleDrefImplicitLod(result_type, sampled_image, coords,
                                                            dref)
                          : ctx.OpImageSampleDrefImplicitLod(
                                result_type, sampled_image, coords, dref, operands.mask,
                                operands.operands);
    const Id sample_typed = texture.is_integer ? ctx.OpBitcast(ctx.F32[1], sample) : sample;
    return ctx.OpCompositeConstruct(ctx.F32[4], sample_typed, ctx.f32_zero_value,
                                    ctx.f32_zero_value, ctx.f32_zero_value);
}

Id EmitImageSampleDrefExplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id dref,
                                  Id lod, const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    coords = FloatSampleCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
    const Id result_type = texture.data_types->Get(1);
    const Id sampler = LoadedSampler(ctx, handle >> 16);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleDrefExplicitLod(result_type, sampled_image, coords, dref,
                                                       operands.mask, operands.operands);
    const Id sample_typed = texture.is_integer ? ctx.OpBitcast(ctx.F32[1], sample) : sample;
    return ctx.OpCompositeConstruct(ctx.F32[4], sample_typed, ctx.f32_zero_value,
                                    ctx.f32_zero_value, ctx.f32_zero_value);
}

Id EmitImageGather(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords,
                   const IR::Value& offset) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    coords = FloatSampleCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = LoadedSampler(ctx, handle >> 16);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const u32 comp = inst->Flags<IR::TextureInstInfo>().gather_comp.Value();
    ImageOperands operands;
    operands.AddOffset(ctx, offset, true);
    const Id component = ctx.ConstU32(comp);
    const Id texels = operands.Empty()
                          ? ctx.OpImageGather(result_type, sampled_image, coords, component)
                          : ctx.OpImageGather(result_type, sampled_image, coords, component,
                                              operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texels) : texels;
}

Id EmitImageGatherDref(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords,
                       const IR::Value& offset, Id dref) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    coords = FloatSampleCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = LoadedSampler(ctx, handle >> 16);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.AddOffset(ctx, offset, true);
    const Id texels = operands.Empty()
                          ? ctx.OpImageDrefGather(result_type, sampled_image, coords, dref)
                          : ctx.OpImageDrefGather(result_type, sampled_image, coords, dref,
                                                  operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texels) : texels;
}

Id EmitImageQueryDimensions(EmitContext& ctx, IR::Inst* inst, u32 handle, Id lod, bool has_mips) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = texture.is_storage ? ctx.OpLoad(texture.image_type, texture.id)
                                        : LoadedSampledImage(ctx, handle & 0xFFFF);
    const auto sharp = ctx.info.images[handle & 0xFFFF].GetSharp(ctx.info);
    const Id zero = ctx.u32_zero_value;
    const auto mips{[&] { return has_mips ? ctx.OpImageQueryLevels(ctx.U32[1], image) : zero; }};
    const bool uses_lod{texture.view_type != AmdGpu::ImageType::Color2DMsaa && !texture.is_storage};
    const Id signed_lod = SignedImageScalar(ctx, inst->Arg(1), lod);
    const auto query{[&](Id type) {
        return uses_lod ? ctx.OpImageQuerySizeLod(type, image, signed_lod)
                        : ctx.OpImageQuerySize(type, image);
    }};
    switch (texture.view_type) {
    case AmdGpu::ImageType::Color1D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[1]), zero, zero, mips());
    case AmdGpu::ImageType::Color1DArray:
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[2]), zero, mips());
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color3D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[3]), mips());
    default:
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
}

Id EmitImageQueryLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    coords = FloatSampleCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
    const Id sampler = LoadedSampler(ctx, handle >> 16);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    const Id zero{ctx.f32_zero_value};
    return ctx.OpImageQueryLod(ctx.F32[2], sampled_image, coords);
}

Id EmitImageGradient(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id derivatives_dx,
                     Id derivatives_dy, const IR::Value& offset, const IR::Value& lod_clamp) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    coords = FloatSampleCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
    const Id result_type = texture.data_types->Get(4);
    const Id sampler = LoadedSampler(ctx, handle >> 16);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    ImageOperands operands;
    operands.AddDerivatives(ctx, derivatives_dx, derivatives_dy);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(result_type, sampled_image, coords,
                                                   operands.mask, operands.operands);
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], sample) : sample;
}

Id EmitImageRead(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod, Id ms) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id signed_coords =
        SignedTexelCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    const Id color_type = texture.data_types->Get(4);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Sample, SignedImageScalar(ctx, inst->Arg(3), ms));
    Id texel;
    if (!texture.is_storage) {
        const Id image = LoadedSampledImage(ctx, handle & 0xFFFF);
        if (texture.view_type != AmdGpu::ImageType::Color2DMsaa) {
            if (Sirit::ValidId(ms)) {
                LOG_ERROR(Render_Recompiler, "image is not MS but ms operand is provided");
            }
            operands.Add(spv::ImageOperandsMask::Lod, SignedImageScalar(ctx, inst->Arg(2), lod));
        }
        texel = operands.Empty()
                    ? ctx.OpImageFetch(color_type, image, signed_coords)
                    : ctx.OpImageFetch(color_type, image, signed_coords, operands.mask,
                                       operands.operands);
    } else {
        Id image_ptr = texture.id;
        if (ctx.profile.supports_image_load_store_lod) {
            operands.Add(spv::ImageOperandsMask::Lod, SignedImageScalar(ctx, inst->Arg(2), lod));
        } else if (Sirit::ValidId(lod)) {
            UNREACHABLE_MSG("Unsupported ImageRead with Lod");
        }
        const Id image = ctx.OpLoad(texture.image_type, image_ptr);
        texel = operands.Empty()
                    ? ctx.OpImageRead(color_type, image, signed_coords)
                    : ctx.OpImageRead(color_type, image, signed_coords, operands.mask,
                                      operands.operands);
    }
    return texture.is_integer ? ctx.OpBitcast(ctx.F32[4], texel) : texel;
}

void EmitImageWrite(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod, Id ms,
                    Id color) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id signed_coords =
        SignedTexelCoordinates(ctx, texture.view_type, inst->Arg(1), coords);
    Id image_ptr = texture.id;
    const Id color_type = texture.data_types->Get(4);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Sample, SignedImageScalar(ctx, inst->Arg(3), ms));
    if (ctx.profile.supports_image_load_store_lod) {
        operands.Add(spv::ImageOperandsMask::Lod, SignedImageScalar(ctx, inst->Arg(2), lod));
    } else if (Sirit::ValidId(lod)) {
        LOG_WARNING(Render, "Fallback for ImageWrite with LOD");
        ASSERT(texture.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex);
        const Id single_image_ptr_type =
            ctx.TypePointer(spv::StorageClass::UniformConstant, texture.image_type);
        image_ptr = ctx.OpAccessChain(single_image_ptr_type, image_ptr, std::array{lod});
    }
    const Id image = ctx.OpLoad(texture.image_type, image_ptr);
    const Id texel = texture.is_integer ? ctx.OpBitcast(color_type, color) : color;
    if (operands.Empty()) {
        ctx.OpImageWrite(image, signed_coords, texel);
    } else {
        ctx.OpImageWrite(image, signed_coords, texel, operands.mask, operands.operands);
    }
}

Id EmitCubeFaceIndex(EmitContext& ctx, IR::Inst* inst, Id cube_coords) {
    if (ctx.profile.supports_native_cube_calc) {
        return ctx.OpCubeFaceIndexAMD(ctx.F32[1], cube_coords);
    } else {
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
}

}
