// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifdef __ANDROID__
#include <android/log.h>
#endif
#include <algorithm>
#include <fmt/format.h>
#include "common/config.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/frontend/structured_control_flow.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/post_order.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"

namespace Shader {

IR::BlockList GenerateBlocks(const IR::AbstractSyntaxList& syntax_list) {
    size_t num_syntax_blocks{};
    for (const auto& [_, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            ++num_syntax_blocks;
        }
    }
    IR::BlockList blocks{};
    blocks.reserve(num_syntax_blocks);
    for (const auto& [data, type] : syntax_list) {
        if (type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(data.block);
        }
    }
    return blocks;
}

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile) {
    constexpr u32 token_mov_vcchi = 0xBEEB03FF;
    if (code[0] != token_mov_vcchi) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Gcn::GcnDecodeContext decoder;

    IR::Program program{info};
    program.ins_list.reserve(code.size());
    while (!slice.atEnd()) {
        program.ins_list.emplace_back(decoder.decodeInstruction(slice));
    }

    const bool has_terminator =
        std::ranges::any_of(program.ins_list, [](const Gcn::GcnInst& inst) {
            return inst.opcode == Gcn::Opcode::S_ENDPGM;
        });
    if (!has_terminator) {
        Gcn::GcnInst endpgm{};
        endpgm.opcode = Gcn::Opcode::S_ENDPGM;
        endpgm.category = Gcn::InstCategory::FlowControl;
        endpgm.length = 4;
        program.ins_list.push_back(endpgm);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_SHADER_NO_ENDPGM] stage=%u hash=%#llx appended implicit S_ENDPGM "
                            "(malformed shader; using a bounded terminal path)",
                            static_cast<u32>(info.stage),
                            static_cast<unsigned long long>(info.pgm_hash));
#endif
    }

    pools.ReleaseContents();

    Common::ObjectPool<Gcn::Block> gcn_block_pool{64};
    Gcn::CFG cfg{gcn_block_pool, program.ins_list};
    if (Config::dumpShaders()) {
        const auto dump_dir =
            Common::FS::GetUserPath(Common::FS::PathType::ShaderDir) / "dumps";
        std::filesystem::create_directories(dump_dir);
        const auto filename =
            fmt::format("{}_{:#018x}.cfg.dot", info.stage, info.pgm_hash);
        Common::FS::IOFile file{dump_dir / filename, Common::FS::FileAccessMode::Create,
                                Common::FS::FileType::TextFile};
        file.WriteString(cfg.Dot());
    }

    program.syntax_list =
        Shader::Gcn::BuildASL(pools.inst_pool, pools.block_pool, cfg, info, runtime_info, profile);
    program.blocks = GenerateBlocks(program.syntax_list);
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());

    Shader::InjectClipDistanceAttributes(program, runtime_info);

    if (!profile.support_float64) {
        Shader::Optimization::LowerFp64ToFp32(program);
    }
    Shader::Optimization::SsaRewritePass(program.post_order_blocks);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    if (info.l_stage == LogicalStage::TessellationControl) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::HullShaderTransform(program, runtime_info);
    } else if (info.l_stage == LogicalStage::TessellationEval) {
        Shader::Optimization::TessellationPreprocess(program, runtime_info);
        Shader::Optimization::DomainShaderTransform(program, runtime_info);
    }
    Shader::Optimization::RingAccessElimination(program, runtime_info);
    Shader::Optimization::ReadLaneEliminationPass(program);
    Shader::Optimization::FlattenExtendedUserdataPass(program);
    Shader::Optimization::ResourceTrackingPass(program, profile);
    Shader::Optimization::LowerBufferFormatToRaw(program);
    Shader::Optimization::SharedMemorySimplifyPass(program, profile);
    Shader::Optimization::SharedMemoryToStoragePass(program, runtime_info, profile);
    Shader::Optimization::SharedMemoryBarrierPass(program, runtime_info, profile);
    Shader::Optimization::IdentityRemovalPass(program.blocks);
    Shader::Optimization::DeadCodeEliminationPass(program);
    Shader::Optimization::ConstantPropagationPass(program.post_order_blocks);
    Shader::Optimization::CollectShaderInfoPass(program, profile);

    Shader::IR::DumpProgram(program, info);

    return program;
}

}
