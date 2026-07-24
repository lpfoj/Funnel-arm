// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/frontend/control_flow_graph.h"

namespace Shader::Gcn {

struct Compare {
    bool operator()(const Block& lhs, u32 rhs) const noexcept {
        return lhs.begin < rhs;
    }

    bool operator()(u32 lhs, const Block& rhs) const noexcept {
        return lhs < rhs.begin;
    }

    bool operator()(const Block& lhs, const Block& rhs) const noexcept {
        return lhs.begin < rhs.begin;
    }
};

static IR::Condition MakeCondition(const GcnInst& inst) {
    if (inst.IsCmpx()) {
        return IR::Condition::Execnz;
    }

    switch (inst.opcode) {
    case Opcode::S_CBRANCH_SCC0:
        return IR::Condition::Scc0;
    case Opcode::S_CBRANCH_SCC1:
        return IR::Condition::Scc1;
    case Opcode::S_CBRANCH_VCCZ:
        return IR::Condition::Vccz;
    case Opcode::S_CBRANCH_VCCNZ:
        return IR::Condition::Vccnz;
    case Opcode::S_CBRANCH_EXECZ:
        return IR::Condition::Execz;
    case Opcode::S_CBRANCH_EXECNZ:
        return IR::Condition::Execnz;
    default:
        return IR::Condition::True;
    }
}

static bool IgnoresExecMask(const GcnInst& inst) {
    switch (inst.category) {
    case InstCategory::ScalarALU:
    case InstCategory::ScalarMemory:
    case InstCategory::FlowControl:
        return true;
    default:
        break;
    }
    switch (inst.opcode) {
    case Opcode::V_READLANE_B32:
    case Opcode::V_WRITELANE_B32:
    case Opcode::V_READFIRSTLANE_B32:
        return true;
    default:
        break;
    }
    return false;
}

static std::optional<u32> ResolveSetPcTarget(std::span<const GcnInst> list, u32 setpc_index,
                                             std::span<const u32> pc_map) {
    if (setpc_index < 3) {
        return std::nullopt;
    }

    const auto& getpc = list[setpc_index - 3];
    const auto& arith = list[setpc_index - 2];
    const auto& setpc = list[setpc_index];

    if (getpc.opcode != Opcode::S_GETPC_B64 ||
        !(arith.opcode == Opcode::S_ADD_U32 || arith.opcode == Opcode::S_SUB_U32) ||
        setpc.opcode != Opcode::S_SETPC_B64)
        return std::nullopt;

    if (getpc.dst[0].code != setpc.src[0].code || arith.dst[0].code != setpc.src[0].code)
        return std::nullopt;

    if (arith.src_count < 2 || arith.src[1].field != OperandField::LiteralConst)
        return std::nullopt;

    const u32 imm = arith.src[1].code;

    const s32 signed_offset =
        (arith.opcode == Opcode::S_ADD_U32) ? static_cast<s32>(imm) : -static_cast<s32>(imm);

    const u32 base_pc = pc_map[setpc_index - 3] + getpc.length;

    const u32 result_pc = static_cast<u32>(static_cast<s32>(base_pc) + signed_offset);
    LOG_DEBUG(Render_Recompiler, "SetPC target: {} + {} = {}", base_pc, signed_offset, result_pc);
    return result_pc & ~0x3u;
}

static constexpr size_t LabelReserveSize = 32;

CFG::CFG(Common::ObjectPool<Block>& block_pool_, std::span<const GcnInst> inst_list_)
    : block_pool{block_pool_}, inst_list{inst_list_} {
    index_to_pc.resize(inst_list.size() + 1);
    labels.reserve(LabelReserveSize);
    EmitLabels();
    EmitBlocks();
    LinkBlocks();
    SplitDivergenceScopes();
}

void CFG::EmitLabels() {
    u32 pc = 0;
    AddLabel(pc);

    for (u32 i = 0; i < inst_list.size(); i++) {
        index_to_pc[i] = pc;
        const GcnInst inst = inst_list[i];
        if (inst.IsUnconditionalBranch()) {
            u32 target = inst.BranchTarget(pc);
            if (inst.opcode == Opcode::S_SETPC_B64) {
                if (auto t = ResolveSetPcTarget(inst_list, i, index_to_pc)) {
                    target = *t;
                } else {
                    LOG_WARNING(Render_Recompiler,
                                "S_SETPC_B64 without a resolvable offset at PC {:#x} (Index {}); "
                                "using fall-through target (reference-release parity)",
                                pc, i);
                }
            }
            AddLabel(target);
            AddLabel(pc + inst.length);
        } else if (inst.IsConditionalBranch()) {
            const u32 true_label = inst.BranchTarget(pc);
            const u32 false_label = pc + inst.length;
            if (true_label != false_label) {
                AddLabel(true_label);
                AddLabel(false_label);
            }
        } else if (inst.opcode == Opcode::S_ENDPGM) {
            const u32 next_label = pc + inst.length;
            AddLabel(next_label);
        }

        pc += inst.length;
    }
    index_to_pc[inst_list.size()] = pc;

    std::ranges::sort(labels);
}

void CFG::SplitDivergenceScopes() {
    const auto is_open_scope = [](const GcnInst& inst) {
        return inst.opcode == Opcode::S_AND_SAVEEXEC_B64 ||
               (inst.opcode == Opcode::S_ANDN2_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               inst.IsCmpx();
    };
    const auto is_close_scope = [](const GcnInst& inst) {
        return (inst.opcode == Opcode::S_MOV_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               inst.opcode == Opcode::S_CBRANCH_EXECZ || inst.opcode == Opcode::S_ENDPGM ||
               (inst.opcode == Opcode::S_ANDN2_B64 && inst.dst[0].field == OperandField::ExecLo);
    };

    for (auto blk = blocks.begin(); blk != blocks.end(); blk++) {
        auto next_blk = std::next(blk);
        s32 curr_begin = -1;
        for (size_t index = blk->begin_index; index <= blk->end_index; index++) {
            const auto& inst = inst_list[index];
            const bool is_close = is_close_scope(inst);
            if ((is_close || index == blk->end_index) && curr_begin != -1) {
                if (index - curr_begin == 1 && is_close) {
                    curr_begin = -1;
                    continue;
                }
                const auto start = inst_list.begin() + curr_begin + 1;
                if (!std::ranges::all_of(start, inst_list.begin() + index + !is_close,
                                         IgnoresExecMask)) {
                    do {
                        ++curr_begin;
                    } while (IgnoresExecMask(inst_list[curr_begin]));

                    s32 curr_end = index;
                    while (IgnoresExecMask(inst_list[curr_end])) {
                        --curr_end;
                    }

                    Block* block = block_pool.Create();
                    block->begin = index_to_pc[curr_begin];
                    block->end = index_to_pc[curr_end];
                    block->begin_index = curr_begin;
                    block->end_index = curr_end;
                    block->end_inst = inst_list[curr_end];
                    blocks.insert_before(next_blk, *block);

                    if (curr_end != blk->end_index) {
                        Block* epi_block = block_pool.Create();
                        epi_block->begin = index_to_pc[curr_end + 1];
                        epi_block->end = blk->end;
                        epi_block->begin_index = curr_end + 1;
                        epi_block->end_index = blk->end_index;
                        epi_block->end_inst = blk->end_inst;
                        epi_block->cond = blk->cond;
                        epi_block->end_class = blk->end_class;
                        epi_block->branch_true = blk->branch_true;
                        epi_block->branch_false = blk->branch_false;
                        blocks.insert_before(next_blk, *epi_block);

                        block->cond = IR::Condition::True;
                        block->branch_true = epi_block;
                        block->branch_false = nullptr;

                        blk->branch_false = epi_block;
                    } else {
                        auto& parent_blk = *blk;
                        ASSERT(blk->cond == IR::Condition::True && blk->branch_true);
                        block->cond = IR::Condition::True;
                        block->branch_true = blk->branch_true;
                        block->branch_false = nullptr;

                        blk->branch_false = blk->branch_true;
                    }

                    --curr_begin;
                    blk->end = index_to_pc[curr_begin];
                    blk->end_index = curr_begin;
                    blk->end_inst = inst_list[curr_begin];
                    blk->cond = IR::Condition::Execnz;
                    blk->end_class = EndClass::Branch;
                    blk->branch_true = block;
                }
                curr_begin = -1;
            }
            if (is_open_scope(inst)) {
                curr_begin = index;
            }
        }
    }
}

void CFG::EmitBlocks() {
    for (auto it = labels.cbegin(); it != labels.cend(); ++it) {
        const Label start = *it;
        const auto next_it = std::next(it);
        const bool is_last = (next_it == labels.cend());
        if (is_last) {
            return;
        }
        const Label end = *next_it;
        const size_t end_index = GetIndex(end) - 1;
        const auto& end_inst = inst_list[end_index];

        Block* block = block_pool.Create();
        block->begin = start;
        block->end = end;
        block->begin_index = GetIndex(start);
        block->end_index = end_index;
        block->end_inst = end_inst;
        block->cond = MakeCondition(end_inst);
        blocks.insert(*block);
    }
}

void CFG::LinkBlocks() {
    const auto get_block = [this](u32 address) -> Block* {
        auto it = blocks.find(address, Compare{});
        if (it == blocks.cend() || it->begin != address) {
            LOG_ERROR(Render_Recompiler, "LinkBlocks: no block at target {:#x}", address);
            return nullptr;
        }
        return &*it;
    };

    for (auto it = blocks.begin(); it != blocks.end(); it++) {
        auto& block = *it;
        const auto end_inst{block.end_inst};

        if (!end_inst.IsTerminateInstruction()) {
            auto* next_block = get_block(block.end);
            if (!next_block) {
                block.end_class = EndClass::Exit;
                block.branch_true = nullptr;
                block.branch_false = nullptr;
                continue;
            }
            block.branch_true = next_block;
            block.end_class = EndClass::Branch;
            continue;
        }

        const u32 branch_pc = block.end - end_inst.length;
        u32 target_pc = 0;
        if (end_inst.opcode == Opcode::S_SETPC_B64) {
            auto tgt = ResolveSetPcTarget(inst_list, block.end_index, index_to_pc);
            if (tgt) {
                target_pc = *tgt;
            } else {
                LOG_WARNING(Render_Recompiler,
                            "S_SETPC_B64 without a resolvable offset at PC {:#x} (Index {}); "
                            "using fall-through target (reference-release parity)",
                            branch_pc, block.end_index);
                target_pc = end_inst.BranchTarget(branch_pc);
            }
        } else {
            target_pc = end_inst.BranchTarget(branch_pc);
        }

        if (end_inst.IsUnconditionalBranch()) {
            auto* target_block = get_block(target_pc);
            if (!target_block) {
                block.end_class = EndClass::Exit;
                block.branch_true = nullptr;
                block.branch_false = nullptr;
                continue;
            }
            block.branch_true = target_block;
            block.end_class = EndClass::Branch;
        } else if (end_inst.IsConditionalBranch()) {
            auto* target_block = get_block(target_pc);
            auto* end_block = get_block(block.end);
            if (!end_block) {
                block.end_class = EndClass::Exit;
                block.branch_true = nullptr;
                block.branch_false = nullptr;
                continue;
            }
            if (!target_block) {
                block.cond = IR::Condition::True;
                block.branch_true = end_block;
                block.branch_false = nullptr;
                block.end_class = EndClass::Branch;
                continue;
            }
            block.branch_true = target_block;
            block.branch_false = end_block;
            block.end_class = EndClass::Branch;
        } else if (end_inst.opcode == Opcode::S_ENDPGM) {
            block.end_class = EndClass::Exit;
        } else {
            UNREACHABLE();
        }
    }
}

std::string CFG::Dot() const {
    int node_uid{0};

    const auto name_of = [](const Block& block) { return fmt::format("\"{:#x}\"", block.begin); };

    std::string dot{"digraph shader {\n"};
    dot += fmt::format("\tsubgraph cluster_{} {{\n", 0);
    dot += fmt::format("\t\tnode [style=filled];\n");
    for (const Block& block : blocks) {
        const std::string name{name_of(block)};
        const auto add_branch = [&](Block* branch, bool add_label) {
            dot += fmt::format("\t\t{}->{}", name, name_of(*branch));
            if (add_label && block.cond != IR::Condition::True &&
                block.cond != IR::Condition::False) {
                dot += fmt::format(" [label=\"{}\"]", block.cond);
            }
            dot += '\n';
        };
        dot += fmt::format("\t\t{};\n", name);
        switch (block.end_class) {
        case EndClass::Branch:
            if (block.cond != IR::Condition::False) {
                add_branch(block.branch_true, true);
            }
            if (block.cond != IR::Condition::True) {
                add_branch(block.branch_false, false);
            }
            break;
        case EndClass::Exit:
            dot += fmt::format("\t\t{}->N{};\n", name, node_uid);
            dot +=
                fmt::format("\t\tN{} [label=\"Exit\"][shape=square][style=stripped];\n", node_uid);
            ++node_uid;
            break;
        }
    }
    dot += "\t\tlabel = \"main\";\n\t}\n";
    if (blocks.empty()) {
        dot += "Start;\n";
    } else {
        dot += fmt::format("\tStart -> {};\n", name_of(*blocks.begin()));
    }
    dot += fmt::format("\tStart [shape=diamond];\n");
    dot += "}\n";
    return dot;
}

}
