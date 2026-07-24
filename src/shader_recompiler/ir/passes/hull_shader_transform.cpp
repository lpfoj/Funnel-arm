// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>
#include "common/assert.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/attribute.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/operand_helper.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/pattern_matching.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Optimization {

namespace {

using namespace Shader::Optimiation::PatternMatching;

static void InitTessConstants(IR::ScalarReg sharp_ptr_base, s32 sharp_dword_offset,
                              Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                              TessellationDataConstantBuffer& tess_constants) {
    info.tess_consts_ptr_base = sharp_ptr_base;
    info.tess_consts_dword_offset = sharp_dword_offset;
    info.ReadTessConstantBuffer(tess_constants);
    runtime_info.InitFromTessConstants(tess_constants);
    return;
}

struct TessSharpLocation {
    IR::ScalarReg ptr_base;
    u32 dword_off;
};

std::optional<TessSharpLocation> FindTessConstantSharp(IR::Inst* read_const_buffer) {
    IR::Value sharp_ptr_base;
    IR::Value sharp_dword_offset;

    IR::Value rv = IR::Value{read_const_buffer};
    IR::Value handle = read_const_buffer->Arg(0);

    if (M_COMPOSITECONSTRUCTU32X4(M_GETUSERDATA(MatchImm(sharp_dword_offset)), MatchIgnore(),
                                  MatchIgnore(), MatchIgnore())
            .Match(handle)) {
        return TessSharpLocation{.ptr_base = IR::ScalarReg::Max,
                                 .dword_off = static_cast<u32>(sharp_dword_offset.ScalarReg())};
    } else if (M_COMPOSITECONSTRUCTU32X4(
                   M_READCONST(M_COMPOSITECONSTRUCTU32X2(M_GETUSERDATA(MatchImm(sharp_ptr_base)),
                                                         MatchIgnore()),
                               MatchImm(sharp_dword_offset)),
                   MatchIgnore(), MatchIgnore(), MatchIgnore())
                   .Match(handle)) {
        return TessSharpLocation{.ptr_base = sharp_ptr_base.ScalarReg(),
                                 .dword_off = sharp_dword_offset.U32()};
    }
    return {};
}

class TessConstantUseWalker {
public:
    void WalkUsersOfTessConstant(IR::Inst* read_const_buffer, TessConstantAttribute attr) {
        u32 inc;
        switch (attr) {
        case TessConstantAttribute::HsNumPatch:
        case TessConstantAttribute::HsOutputBase:
            inc = 1;
            break;
        case TessConstantAttribute::PatchConstBase:
            inc = 2;
            break;
        default:
            UNREACHABLE();
        }

        for (IR::Use use : read_const_buffer->Uses()) {
            WalkUsersOfTessConstantHelper(use, inc, false);
        }

        ++seq_num;
    }

private:
    struct PhiInfo {
        u32 seq_num;
        u32 unique_edge;
    };

    void WalkUsersOfTessConstantHelper(IR::Use use, u32 inc, bool propagateError) {
        IR::Inst* inst = use.user;

        switch (use.user->GetOpcode()) {
        case IR::Opcode::LoadSharedU32:
        case IR::Opcode::LoadSharedU64:
        case IR::Opcode::WriteSharedU32:
        case IR::Opcode::WriteSharedU64: {
            bool is_addr_operand = use.operand == 0;
            if (is_addr_operand) {
                u32 counter = inst->Flags<u32>();
                inst->SetFlags<u32>(counter + inc);
                ASSERT_MSG(!propagateError, "LDS instruction {} accesses ambiguous attribute type",
                           fmt::ptr(use.user));
                return;
            }
        }
        case IR::Opcode::Phi: {
            auto it = phi_infos.find(use.user);
            if (it == phi_infos.end()) {
                phi_infos[inst] = {.seq_num = seq_num,
                                   .unique_edge = static_cast<u16>(use.operand)};
            } else if (it->second.seq_num < seq_num) {
                it->second.seq_num = seq_num;
                if (it->second.unique_edge != use.operand) {
                    propagateError = true;
                }
            } else {
                ASSERT(it->second.seq_num == seq_num);
                return;
            }
            break;
        }
        default:
            break;
        }

        for (IR::Use use : inst->Uses()) {
            WalkUsersOfTessConstantHelper(use, inc, propagateError);
        }
    }

    std::unordered_map<const IR::Inst*, PhiInfo> phi_infos;
    u32 seq_num{1u};
};

enum class AttributeRegion : u32 { InputCP, OutputCP, PatchConst };

static AttributeRegion GetAttributeRegionKind(IR::Inst* ring_access, const Shader::Info& info,
                                              const Shader::RuntimeInfo& runtime_info) {
    u32 count = ring_access->Flags<u32>();
    if (count == 0) {
        return AttributeRegion::InputCP;
    } else if (info.l_stage == LogicalStage::TessellationControl &&
               runtime_info.hs_info.IsPassthrough()) {
        ASSERT(count <= 1);
        return AttributeRegion::PatchConst;
    } else {
        ASSERT(count <= 2);
        return AttributeRegion(count);
    }
}

static bool IsDivisibleByStride(IR::Value term, u32 stride) {
    IR::Value a, b;
    if (MatchU32(stride).Match(term)) {
        return true;
    } else if (M_BITFIELDUEXTRACT(MatchValue(a), MatchU32(0), MatchU32(24)).Match(term) ||
               M_BITFIELDSEXTRACT(MatchValue(a), MatchU32(0), MatchU32(24)).Match(term)) {
        return IsDivisibleByStride(a, stride);
    } else if (M_IMUL32(MatchValue(a), MatchValue(b)).Match(term)) {
        return IsDivisibleByStride(a, stride) || IsDivisibleByStride(b, stride);
    }
    return false;
}

static bool TryOptimizeAddendInModulo(IR::Value addend, u32 stride, std::vector<IR::U32>& addends) {
    IR::Value a, b;
    if (M_IADD32(MatchValue(a), MatchValue(b)).Match(addend)) {
        bool ret = false;
        ret = TryOptimizeAddendInModulo(a, stride, addends);
        ret |= TryOptimizeAddendInModulo(b, stride, addends);
        return ret;
    } else if (!IsDivisibleByStride(addend, stride)) {
        addends.push_back(IR::U32{addend});
        return false;
    } else {
        return true;
    }
}

static IR::U32 TryOptimizeAddressModulo(IR::U32 addr, u32 stride, IR::IREmitter& ir) {
    std::vector<IR::U32> addends;
    if (TryOptimizeAddendInModulo(addr, stride, addends)) {
        addr = ir.Imm32(0);
        for (auto& addend : addends) {
            addr = ir.IAdd(addr, addend);
        }
    }
    return addr;
}


static IR::F32 ReadTessControlPointAttribute(IR::U32 addr, const u32 stride, IR::IREmitter& ir,
                                             u32 off_dw, bool is_output_read_in_tcs) {
    if (off_dw > 0) {
        addr = ir.IAdd(addr, ir.Imm32(off_dw));
    }
    const IR::U32 control_point_index = ir.IDiv(addr, ir.Imm32(stride));
    const IR::U32 opt_addr = TryOptimizeAddressModulo(addr, stride, ir);
    const IR::U32 offset = ir.IMod(opt_addr, ir.Imm32(stride));
    const IR::U32 attr_index = ir.ShiftRightLogical(offset, ir.Imm32(4u));
    const IR::U32 comp_index =
        ir.ShiftRightLogical(ir.BitwiseAnd(offset, ir.Imm32(0xFU)), ir.Imm32(2u));
    if (is_output_read_in_tcs) {
        return ir.ReadTcsGenericOuputAttribute(control_point_index, attr_index, comp_index);
    } else {
        return ir.GetTessGenericAttribute(control_point_index, attr_index, comp_index);
    }
}

}

void HullShaderTransform(IR::Program& program, const RuntimeInfo& runtime_info) {
    const Info& info = program.info;

    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const auto opcode = inst.GetOpcode();
            switch (opcode) {
            case IR::Opcode::StoreBufferU32:
            case IR::Opcode::StoreBufferU32x2:
            case IR::Opcode::StoreBufferU32x3:
            case IR::Opcode::StoreBufferU32x4: {
                IR::Value soffset = IR::GetBufferSOffsetArg(&inst);
                if (!M_GETATTRIBUTEU32(MatchAttribute(IR::Attribute::TessFactorsBufferBase),
                                       MatchIgnore())
                         .Match(soffset)) {
                    break;
                }

                const auto info = inst.Flags<IR::BufferInstInfo>();
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};

                IR::Value voffset;
                bool success =
                    M_COMPOSITECONSTRUCTU32X3(MatchU32(0), MatchImm(voffset), MatchIgnore())
                        .Match(inst.Arg(IR::StoreBufferArgs::Address));
                ASSERT_MSG(success, "unhandled pattern in tess factor store");

                const u32 gcn_factor_idx = (info.inst_offset.Value() + voffset.U32()) >> 2;
                const IR::Value data = inst.Arg(IR::StoreBufferArgs::Data);

                const u32 num_dwords = u32(opcode) - u32(IR::Opcode::StoreBufferU32) + 1;

                const auto GetValue = [&](IR::Value data) -> IR::F32 {
                    if (auto* inst = data.TryInstRecursive();
                        inst && inst->GetOpcode() == IR::Opcode::BitCastU32F32) {
                        return IR::F32{inst->Arg(0)};
                    }
                    return ir.BitCast<IR::F32, IR::U32>(IR::U32{data});
                };

                auto get_factor_attr = [&](u32 gcn_factor_idx) -> IR::Patch {
                    switch (runtime_info.hs_info.tess_type) {
                    case AmdGpu::TessellationType::Isoline:
                        ASSERT(gcn_factor_idx < 2);
                        return IR::PatchFactor(gcn_factor_idx);
                    case AmdGpu::TessellationType::Triangle:
                        ASSERT(gcn_factor_idx < 4);
                        if (gcn_factor_idx == 3) {
                            return IR::Patch::TessellationLodInteriorU;
                        }
                        return IR::PatchFactor(gcn_factor_idx);
                    case AmdGpu::TessellationType::Quad:
                        ASSERT(gcn_factor_idx < 6);
                        return IR::PatchFactor(gcn_factor_idx);
                    default:
                        UNREACHABLE();
                    }
                };

                inst.Invalidate();
                if (num_dwords == 1) {
                    ir.SetPatch(get_factor_attr(gcn_factor_idx), GetValue(data));
                    break;
                }
                auto* inst = data.TryInstRecursive();
                ASSERT(inst && (inst->GetOpcode() == IR::Opcode::CompositeConstructU32x2 ||
                                inst->GetOpcode() == IR::Opcode::CompositeConstructU32x3 ||
                                inst->GetOpcode() == IR::Opcode::CompositeConstructU32x4));
                for (s32 i = 0; i < num_dwords; i++) {
                    ir.SetPatch(get_factor_attr(gcn_factor_idx + i), GetValue(inst->Arg(i)));
                }
                break;
            }

            case IR::Opcode::WriteSharedU32:
            case IR::Opcode::WriteSharedU64: {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                const u32 num_dwords = opcode == IR::Opcode::WriteSharedU32 ? 1 : 2;
                const IR::U32 addr{inst.Arg(0)};
                const IR::Value data = num_dwords == 2
                                           ? ir.UnpackUint2x32(IR::U64{inst.Arg(1).Resolve()})
                                           : inst.Arg(1).Resolve();

                const auto SetOutput = [&](IR::U32 addr, IR::U32 value, AttributeRegion output_kind,
                                           u32 off_dw) {
                    const IR::F32 data_component = ir.BitCast<IR::F32, IR::U32>(value);

                    if (output_kind == AttributeRegion::OutputCP) {
                        if (off_dw > 0) {
                            addr = ir.IAdd(addr, ir.Imm32(off_dw));
                        }
                        const u32 stride = runtime_info.hs_es_vs_info.hs_output_cp_stride;
                        const IR::U32 opt_addr = TryOptimizeAddressModulo(addr, stride, ir);
                        const IR::U32 offset = ir.IMod(opt_addr, ir.Imm32(stride));
                        const IR::U32 attr_index = ir.ShiftRightLogical(offset, ir.Imm32(4u));
                        const IR::U32 comp_index = ir.ShiftRightLogical(
                            ir.BitwiseAnd(offset, ir.Imm32(0xFU)), ir.Imm32(2u));
                        ir.SetTcsGenericAttribute(data_component, attr_index, comp_index);
                    } else {
                        ASSERT(output_kind == AttributeRegion::PatchConst);
                        ASSERT_MSG(addr.IsImmediate(), "patch addr non imm, inst {}",
                                   fmt::ptr(addr.Inst()));
                        ir.SetPatch(IR::PatchGeneric((addr.U32() >> 2) + off_dw), data_component);
                    }
                };

                AttributeRegion region = GetAttributeRegionKind(&inst, info, runtime_info);
                if (num_dwords == 1) {
                    SetOutput(addr, IR::U32{data}, region, 0);
                } else {
                    for (auto i = 0; i < num_dwords; i++) {
                        SetOutput(addr, IR::U32{ir.CompositeExtract(data, i)}, region, i);
                    }
                }
                inst.Invalidate();
                break;
            }

            case IR::Opcode::LoadSharedU32:
            case IR::Opcode::LoadSharedU64: {
                IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
                const IR::U32 addr{inst.Arg(0)};
                const AttributeRegion region = GetAttributeRegionKind(&inst, info, runtime_info);
                const u32 num_dwords = opcode == IR::Opcode::LoadSharedU32 ? 1 : 2;
                ASSERT_MSG(region == AttributeRegion::InputCP ||
                               region == AttributeRegion::OutputCP,
                           "Unhandled read of patchconst attribute in hull shader");
                const bool is_tcs_output_read = region == AttributeRegion::OutputCP;
                const u32 stride = is_tcs_output_read
                                       ? runtime_info.hs_es_vs_info.hs_output_cp_stride
                                       : runtime_info.hs_info.ls_stride;
                IR::Value attr_read;
                if (num_dwords == 1) {
                    attr_read = ir.BitCast<IR::U32>(
                        ReadTessControlPointAttribute(addr, stride, ir, 0, is_tcs_output_read));
                } else {
                    boost::container::static_vector<IR::Value, 4> read_components;
                    for (auto i = 0; i < num_dwords; i++) {
                        const IR::F32 component =
                            ReadTessControlPointAttribute(addr, stride, ir, i, is_tcs_output_read);
                        read_components.push_back(ir.BitCast<IR::U32>(component));
                    }
                    attr_read = ir.PackUint2x32(ir.CompositeConstruct(read_components));
                }
                inst.ReplaceUsesWithAndRemove(attr_read);
                break;
            }

            default:
                break;
            }
        }
    }

    if (runtime_info.hs_info.IsPassthrough()) {
        IR::Block* entry_block = *program.blocks.begin();
        auto it = std::ranges::find_if(entry_block->Instructions(), [](IR::Inst& inst) {
            return inst.GetOpcode() == IR::Opcode::Prologue;
        });
        ASSERT(it != entry_block->end());
        ++it;
        ASSERT(it != entry_block->end());
        ++it;
        IR::IREmitter ir{*entry_block, it};

        u32 num_attributes = Common::AlignUp(runtime_info.hs_info.ls_stride, 16) >> 4;
        const auto invocation_id = ir.GetAttributeU32(IR::Attribute::InvocationId);
        for (u32 attr_no = 0; attr_no < num_attributes; attr_no++) {
            for (u32 comp = 0; comp < 4; comp++) {
                IR::F32 attr_read =
                    ir.GetTessGenericAttribute(invocation_id, ir.Imm32(attr_no), ir.Imm32(comp));
                ir.SetTcsGenericAttribute(attr_read, ir.Imm32(attr_no), ir.Imm32(comp));
            }
        }
    }
}

void DomainShaderTransform(const IR::Program& program, const RuntimeInfo& runtime_info) {
    const Info& info = program.info;

    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const auto opcode = inst.GetOpcode();
            switch (inst.GetOpcode()) {
            case IR::Opcode::LoadSharedU32:
            case IR::Opcode::LoadSharedU64: {
                const IR::U32 addr{inst.Arg(0)};
                AttributeRegion region = GetAttributeRegionKind(&inst, info, runtime_info);
                const u32 num_dwords = opcode == IR::Opcode::LoadSharedU32 ? 1 : 2;
                const auto GetInput = [&](IR::U32 addr, u32 off_dw) -> IR::F32 {
                    if (region == AttributeRegion::OutputCP) {
                        return ReadTessControlPointAttribute(
                            addr, runtime_info.hs_es_vs_info.hs_output_cp_stride, ir, off_dw,
                            false);
                    } else {
                        ASSERT(region == AttributeRegion::PatchConst);
                        return ir.GetPatch(IR::PatchGeneric((addr.U32() >> 2) + off_dw));
                    }
                };
                IR::Value attr_read;
                if (num_dwords == 1) {
                    attr_read = ir.BitCast<IR::U32>(GetInput(addr, 0));
                } else {
                    boost::container::static_vector<IR::Value, 4> read_components;
                    for (auto i = 0; i < num_dwords; i++) {
                        const IR::F32 component = GetInput(addr, i);
                        read_components.push_back(ir.BitCast<IR::U32>(component));
                    }
                    attr_read = ir.PackUint2x32(ir.CompositeConstruct(read_components));
                }
                inst.ReplaceUsesWithAndRemove(attr_read);
                break;
            }
            default:
                break;
            }
        }
    }
}

void TessellationPreprocess(IR::Program& program, RuntimeInfo& runtime_info) {
    TessellationDataConstantBuffer tess_constants;
    Shader::Info& info = program.info;
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            auto found_tess_consts_sharp = [&]() -> bool {
                switch (inst.GetOpcode()) {
                case IR::Opcode::LoadSharedU32:
                case IR::Opcode::LoadSharedU64:
                case IR::Opcode::WriteSharedU32:
                case IR::Opcode::WriteSharedU64: {
                    IR::Value addr = inst.Arg(0);
                    auto read_const_buffer = IR::BreadthFirstSearch(
                        addr, [](IR::Inst* maybe_tess_const) -> std::optional<IR::Inst*> {
                            if (maybe_tess_const->GetOpcode() == IR::Opcode::ReadConstBuffer) {
                                return maybe_tess_const;
                            }
                            return std::nullopt;
                        });
                    if (read_const_buffer) {
                        auto sharp_location = FindTessConstantSharp(read_const_buffer.value());
                        if (sharp_location) {
                            if (info.tess_consts_dword_offset >= 0) {
                                ASSERT_MSG(static_cast<s32>(sharp_location->dword_off) ==
                                                   info.tess_consts_dword_offset &&
                                               sharp_location->ptr_base ==
                                                   info.tess_consts_ptr_base,
                                           "TessConstants V# is ambiguous");
                            }
                            InitTessConstants(sharp_location->ptr_base,
                                              static_cast<s32>(sharp_location->dword_off), info,
                                              runtime_info, tess_constants);
                            return true;
                        }
                        UNREACHABLE_MSG("Failed to match tess constant sharp");
                    }
                    return false;
                }
                default:
                    return false;
                }
            }();

            if (found_tess_consts_sharp) {
                break;
            }
        }
    }

    ASSERT(info.tess_consts_dword_offset >= 0);

    TessConstantUseWalker walker;

    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
                auto sharp_location = FindTessConstantSharp(&inst);
                if (sharp_location && sharp_location->ptr_base == info.tess_consts_ptr_base &&
                    sharp_location->dword_off == info.tess_consts_dword_offset) {
                    IR::Value index = inst.Arg(1);

                    ASSERT_MSG(index.IsImmediate(),
                               "Tessellation constant read with dynamic index");
                    u32 off_dw = index.U32();
                    ASSERT(off_dw <=
                           static_cast<u32>(TessConstantAttribute::FirstEdgeTessFactorIndex));

                    auto tess_const_attr = static_cast<TessConstantAttribute>(off_dw);
                    switch (tess_const_attr) {
                    case TessConstantAttribute::LsStride:
                        ASSERT(info.l_stage == LogicalStage::TessellationControl);
                        inst.ReplaceUsesWithAndRemove(IR::Value(tess_constants.ls_stride));
                        break;
                    case TessConstantAttribute::HsCpStride:
                        inst.ReplaceUsesWithAndRemove(IR::Value(tess_constants.hs_cp_stride));
                        break;
                    case TessConstantAttribute::HsNumPatch:
                    case TessConstantAttribute::HsOutputBase:
                    case TessConstantAttribute::PatchConstBase:
                        walker.WalkUsersOfTessConstant(&inst, tess_const_attr);
                        inst.ReplaceUsesWithAndRemove(IR::Value(0u));
                        break;
                    case Shader::TessConstantAttribute::PatchConstSize:
                    case Shader::TessConstantAttribute::PatchOutputSize:
                    case Shader::TessConstantAttribute::OffChipTessellationFactorThreshold:
                    case Shader::TessConstantAttribute::FirstEdgeTessFactorIndex:
                        break;
                    default:
                        UNREACHABLE_MSG("Read past end of TessConstantsBuffer");
                    }
                }
            }
        }
    }

    if (info.l_stage == LogicalStage::TessellationControl) {
        for (IR::Block* block : program.blocks) {
            for (auto it = block->Instructions().begin(); it != block->Instructions().end(); it++) {
                IR::Inst& inst = *it;
                if (M_BITFIELDUEXTRACT(
                        M_GETATTRIBUTEU32(MatchAttribute(IR::Attribute::PackedHullInvocationInfo),
                                          MatchIgnore()),
                        MatchU32(0), MatchU32(8))
                        .Match(IR::Value{&inst})) {
                    IR::IREmitter emit(*block, it);
                    IR::Value replacement(0u);
                    inst.ReplaceUsesWithAndRemove(replacement);
                } else if (M_BITFIELDUEXTRACT(
                               M_GETATTRIBUTEU32(
                                   MatchAttribute(IR::Attribute::PackedHullInvocationInfo),
                                   MatchIgnore()),
                               MatchU32(8), MatchU32(5))
                               .Match(IR::Value{&inst})) {
                    IR::IREmitter ir(*block, it);
                    IR::Value replacement;
                    if (runtime_info.hs_info.IsPassthrough()) {
                        replacement = ir.Imm32(0);
                    } else {
                        replacement = ir.GetAttributeU32(IR::Attribute::InvocationId);
                    }
                    inst.ReplaceUsesWithAndRemove(replacement);
                }
            }
        }
    }

    ConstantPropagationPass(program.post_order_blocks);
}

}
