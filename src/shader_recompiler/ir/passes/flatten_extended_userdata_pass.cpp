
// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(__ANDROID__)
#include <android/log.h>
#endif
#include <boost/container/flat_map.hpp>
#include <boost/container/small_vector.hpp>
#include "common/arch.h"
#include "core/libraries/gnmdriver/gnmdriver.h"
#ifdef ARCH_X86_64
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
#endif
#include "common/config.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/signal_context.h"
#include "core/signals.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/ir/srt_gvn_table.h"
#include "shader_recompiler/ir/value.h"
#include "src/common/decoder.h"

#ifdef ARCH_X86_64
using namespace Xbyak::util;

static Xbyak::CodeGenerator g_srt_codegen(32_MB);
static const u8* g_srt_codegen_start = nullptr;
#endif

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
#ifdef ARCH_X86_64
    const auto func_addr = (PFN_SrtWalker)g_srt_codegen.getCurr();
    g_srt_codegen.db(ptr, size);
    g_srt_codegen.ready();
    return func_addr;
#else
    (void)ptr;
    (void)size;
    return nullptr;
#endif
}

}

namespace {

static void DumpSrtProgram(const Shader::Info& info, const u8* code, size_t codesize) {
#ifdef ARCH_X86_64
    using namespace Common::FS;

    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}_{:#018x}.srtprogram.txt", info.stage, info.pgm_hash);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create, FileType::TextFile};

    u64 address = reinterpret_cast<u64>(code);
    u64 code_end = address + codesize;
    X86DecodedInstruction instruction;
    X86DecodedOperand operands[X86_MAX_OPERAND_COUNT];
    X86DecodeStatus status = X86_STATUS_OK;
    while (address < code_end && X86_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                                     instruction, operands, reinterpret_cast<void*>(address)))) {
        std::string s =
            Common::Decoder::Instance()->disassembleInst(instruction, operands, address);
        s += "\n";
        file.WriteString(s);
        address += instruction.length;
    }
#endif
}

#ifdef ARCH_X86_64
static bool SrtWalkerSignalHandler(void* context, void* fault_address) {
    const u8* code_start = g_srt_codegen_start;
    const u8* code_end = code_start + g_srt_codegen.getSize();
    const void* code = Common::GetRip(context);
    if (code < code_start || code >= code_end) {
        return false;
    }

    X86DecodedInstruction instruction;
    X86DecodedOperand operands[X86_MAX_OPERAND_COUNT];
    X86DecodeStatus status = Common::Decoder::Instance()->decodeInstruction(instruction, operands,
                                                                       const_cast<void*>(code), 15);

    ASSERT(X86_SUCCESS(status) && instruction.mnemonic == X86_MNEMONIC_MOV &&
           operands[0].type == X86_OPERAND_TYPE_REGISTER &&
           operands[1].type == X86_OPERAND_TYPE_MEMORY);

    size_t len = instruction.length;
    const size_t patch_size = 3;
    u8* code_patch = const_cast<u8*>(reinterpret_cast<const u8*>(code));

    switch (operands[0].reg.value) {
    case X86_REGISTER_RDI:
        code_patch[0] = 0x48;
        code_patch[1] = 0x31;
        code_patch[2] = 0xFF;
        break;
    case X86_REGISTER_R10D:
        code_patch[0] = 0x45;
        code_patch[1] = 0x31;
        code_patch[2] = 0xD2;
        break;
    default:
        UNREACHABLE_MSG("Unsupported register for SRT walker patch");
        return false;
    }

    memset(code_patch + patch_size, 0x90, len - patch_size);

    LOG_DEBUG(Render_Recompiler, "Patched SRT walker at {}", code);

    return true;
}
#endif

using namespace Shader;

struct PassInfo {
    using PtrUserList = boost::container::flat_map<u32, Shader::IR::Inst*>;

    Optimization::SrtGvnTable gvn_table;
    std::unordered_map<IR::Inst*, PtrUserList> pointer_uses;
    boost::container::small_flat_map<IR::ScalarReg, IR::Inst*, 1> srt_roots;

    std::unordered_map<u32, IR::Inst*> vn_to_inst;

    u32 dst_off_dw;

    PtrUserList* GetUsesAsPointer(IR::Inst* inst) {
        auto it = pointer_uses.find(inst);
        if (it != pointer_uses.end()) {
            return &it->second;
        }
        return nullptr;
    }

    IR::Inst* DeduplicateInstruction(IR::Inst* inst) {
        auto it = vn_to_inst.try_emplace(gvn_table.GetValueNumber(inst), inst);
        return it.first->second;
    }
};
}

namespace Shader::Optimization {

namespace {

#ifdef ARCH_X86_64
static inline void PushPtr(Xbyak::CodeGenerator& c, u32 off_dw) {
    c.push(rdi);
    c.mov(rdi, ptr[rdi + (off_dw << 2)]);
    c.mov(r10, 0xFFFFFFFFFFFFULL);
    c.and_(rdi, r10);
}

static inline void PopPtr(Xbyak::CodeGenerator& c) {
    c.pop(rdi);
};

static void VisitPointer(u32 off_dw, IR::Inst* subtree, PassInfo& pass_info,
                         Xbyak::CodeGenerator& c) {
    PushPtr(c, off_dw);
    PassInfo::PtrUserList* use_list = pass_info.GetUsesAsPointer(subtree);
    ASSERT(use_list);

    for (auto [src_off_dw, use] : *use_list) {
        c.mov(r10d, ptr[rdi + (src_off_dw << 2)]);
        c.mov(ptr[rsi + (pass_info.dst_off_dw << 2)], r10d);

        use->SetFlags<u32>(pass_info.dst_off_dw);
        pass_info.dst_off_dw++;
    }

    for (const auto [src_off_dw, use] : *use_list) {
        if (pass_info.GetUsesAsPointer(use)) {
            VisitPointer(src_off_dw, use, pass_info, c);
        }
    }

    PopPtr(c);
}
#endif

static void BuildSrtRecipeSubtree(IR::Inst* subtree, PassInfo& pass_info,
                                  Shader::PersistentSrtInfo& srt, s32 cur_node) {
    PassInfo::PtrUserList* use_list = pass_info.GetUsesAsPointer(subtree);
    ASSERT(use_list);

    for (auto [src_off_dw, use] : *use_list) {
        srt.portable_copies.push_back(Shader::PortableSrtCopy{
            .node = static_cast<u32>(cur_node),
            .src_off_dw = src_off_dw,
            .dst_off_dw = pass_info.dst_off_dw,
        });
        use->SetFlags<u32>(pass_info.dst_off_dw);
        pass_info.dst_off_dw++;
    }

    for (const auto [src_off_dw, use] : *use_list) {
        if (pass_info.GetUsesAsPointer(use)) {
            const s32 child_node = static_cast<s32>(srt.portable_nodes.size());
            srt.portable_nodes.push_back(Shader::PortableSrtNode{
                .parent = cur_node,
                .ptr_off_dw = src_off_dw,
            });
            BuildSrtRecipeSubtree(use, pass_info, srt, child_node);
        }
    }
}

static void BuildSrtRecipe(Info& info, PassInfo& pass_info) {
    info.srt_info.portable_nodes.clear();
    info.srt_info.portable_copies.clear();
    pass_info.dst_off_dw = NUM_USER_DATA_REGS;
    for (const auto& [sgpr_base, root] : pass_info.srt_roots) {
        const s32 root_node = static_cast<s32>(info.srt_info.portable_nodes.size());
        info.srt_info.portable_nodes.push_back(Shader::PortableSrtNode{
            .parent = -1,
            .ptr_off_dw = static_cast<u32>(sgpr_base),
        });
        BuildSrtRecipeSubtree(root, pass_info, info.srt_info, root_node);
    }
    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
}

static void GenerateSrtProgram(Info& info, PassInfo& pass_info) {
#ifdef ARCH_X86_64
    Xbyak::CodeGenerator& c = g_srt_codegen;

    if (pass_info.srt_roots.empty()) {
        return;
    }

    if (g_srt_codegen_start == nullptr) {
        g_srt_codegen_start = c.getCurr();
        auto* signals = Core::Signals::Instance();
        constexpr u32 priority = 1;
        signals->RegisterAccessViolationHandler(SrtWalkerSignalHandler, priority);
    }

    info.srt_info.walker_func = c.getCurr<PFN_SrtWalker>();
    pass_info.dst_off_dw = NUM_USER_DATA_REGS;
    ASSERT(pass_info.dst_off_dw == info.srt_info.flattened_bufsize_dw);

    for (const auto& [sgpr_base, root] : pass_info.srt_roots) {
        VisitPointer(static_cast<u32>(sgpr_base), root, pass_info, c);
    }

    c.ret();
    c.ready();

    info.srt_info.walker_func_size =
        c.getCurr() - reinterpret_cast<const u8*>(info.srt_info.walker_func);

    if (Config::dumpShaders()) {
        DumpSrtProgram(info, reinterpret_cast<const u8*>(info.srt_info.walker_func),
                       info.srt_info.walker_func_size);
    }

    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
    BuildSrtRecipe(info, pass_info);
#else
    if (pass_info.srt_roots.empty()) {
        return;
    }

    info.srt_info.walker_func = nullptr;
    info.srt_info.walker_func_size = 0;
    BuildSrtRecipe(info, pass_info);
#endif
}

};

void FlattenExtendedUserdataPass(IR::Program& program) {
    Shader::Info& info = program.info;
    PassInfo pass_info;

    boost::container::small_vector<IR::Inst*, 32> all_readconsts;

    for (auto r_it = program.post_order_blocks.rbegin(); r_it != program.post_order_blocks.rend();
         r_it++) {
        IR::Block* block = *r_it;
        for (IR::Inst& inst : *block) {
            if (inst.GetOpcode() == IR::Opcode::ReadConst) {
                if (!inst.Arg(1).IsImmediate()) {
                    LOG_WARNING(Render_Recompiler, "ReadConst has non-immediate offset");
                    continue;
                }

                all_readconsts.push_back(&inst);
                if (pass_info.DeduplicateInstruction(&inst) != &inst) {
                    continue;
                }

                IR::Inst* ptr_composite = inst.Arg(0).InstRecursive();

                const auto pred = [](IR::Inst* inst) -> std::optional<IR::Inst*> {
                    if (inst->GetOpcode() == IR::Opcode::GetUserData ||
                        inst->GetOpcode() == IR::Opcode::ReadConst) {
                        return inst;
                    }
                    return std::nullopt;
                };
                auto base0 = IR::BreadthFirstSearch(ptr_composite->Arg(0), pred);
                auto base1 = IR::BreadthFirstSearch(ptr_composite->Arg(1), pred);
                ASSERT_MSG(base0 && base1, "ReadConst not from constant memory");

                IR::Inst* ptr_lo = base0.value();
                ptr_lo = pass_info.DeduplicateInstruction(ptr_lo);

                auto ptr_uses_kv =
                    pass_info.pointer_uses.try_emplace(ptr_lo, PassInfo::PtrUserList{});
                PassInfo::PtrUserList& user_list = ptr_uses_kv.first->second;

                user_list[inst.Arg(1).U32()] = &inst;

                if (ptr_lo->GetOpcode() == IR::Opcode::GetUserData) {
                    IR::ScalarReg ud_reg = ptr_lo->Arg(0).ScalarReg();
                    pass_info.srt_roots[ud_reg] = ptr_lo;
                }
            }
        }
    }

    GenerateSrtProgram(info, pass_info);

    for (IR::Inst* readconst : all_readconsts) {
        ASSERT(pass_info.vn_to_inst.contains(pass_info.gvn_table.GetValueNumber(readconst)));
        IR::Inst* original = pass_info.DeduplicateInstruction(readconst);
        readconst->SetFlags<u32>(original->Flags<u32>());
    }

    info.RefreshFlatBuf();
}

}

namespace Shader {

void RunPortableSrtWalker(const u32* user_data, u32* flat, const PersistentSrtInfo& srt,
                           u32 stage, u64 pgm_hash,
                           SrtMemoryRanges* memory_ranges,
                           PFN_SrtMemoryRead memory_reader) {
    if (srt.portable_nodes.empty()) {
        return;
    }
    const bool replay = Libraries::GnmDriver::ExecutorReplayActive();
    const bool capture = Libraries::GnmDriver::ExecutorGnmCaptureActive();
#if defined(__ANDROID__)
    static const bool srt_trace_enabled =
        std::getenv("EXECUTOR_TRACE_SRT_WALKER") != nullptr;
    const bool live_diag = !replay && srt_trace_enabled;
    static std::mutex live_srt_seen_mutex;
    static std::vector<std::pair<u32, u64>> live_srt_seen;
    bool live_diag_log = false;
    if (live_diag) {
        std::scoped_lock lock{live_srt_seen_mutex};
        const std::pair key{stage, pgm_hash};
        if (live_srt_seen.size() < 128 &&
            std::ranges::find(live_srt_seen, key) == live_srt_seen.end()) {
            live_srt_seen.push_back(key);
            live_diag_log = true;
        }
    }
#endif
    for (const auto& c : srt.portable_copies) {
        if (c.node >= srt.portable_nodes.size()) {
            return;
        }
    }
    for (size_t i = 0; i < srt.portable_nodes.size(); ++i) {
        const s32 parent = srt.portable_nodes[i].parent;
        if (parent >= 0 && static_cast<size_t>(parent) >= i) {
            return;
        }
    }
    boost::container::small_vector<u32, 32> node_span_dw;
    if (capture || memory_ranges) {
        node_span_dw.resize(srt.portable_nodes.size(), 0);
        for (const auto& c : srt.portable_copies) {
            node_span_dw[c.node] = std::max(node_span_dw[c.node], c.src_off_dw + 1u);
        }
        for (const auto& n : srt.portable_nodes) {
            if (n.parent >= 0) {
                node_span_dw[n.parent] = std::max(node_span_dw[n.parent], n.ptr_off_dw + 2u);
            }
        }
    }
    const bool bulk_reader_requested = !replay && memory_reader != nullptr && memory_ranges != nullptr;
    bool bulk_reader_enabled = bulk_reader_requested;
    boost::container::small_vector<size_t, 32> node_bulk_offsets(srt.portable_nodes.size(), 0);
    boost::container::small_vector<u8, 32> node_bulk_valid(srt.portable_nodes.size(), 0);
    size_t total_bulk_dwords = 0;
    constexpr size_t kMaxBulkSrtDwords = 64u * 1024u;
    if (bulk_reader_enabled) {
        for (size_t i = 0; i < node_span_dw.size(); ++i) {
            node_bulk_offsets[i] = total_bulk_dwords;
            if (node_span_dw[i] > kMaxBulkSrtDwords - total_bulk_dwords) {
                bulk_reader_enabled = false;
                break;
            }
            total_bulk_dwords += node_span_dw[i];
        }
    }
    boost::container::small_vector<u32, 256> bulk_node_words;
    if (bulk_reader_enabled) {
        bulk_node_words.resize(total_bulk_dwords);
    }
    boost::container::small_vector<VAddr, 32> bases(srt.portable_nodes.size(), 0);
    for (size_t i = 0; i < srt.portable_nodes.size(); ++i) {
        const auto& n = srt.portable_nodes[i];
        u64 ptr{};
        const u64 byte_offset = static_cast<u64>(n.ptr_off_dw) * sizeof(u32);
        if (n.parent < 0) {
            std::memcpy(&ptr, reinterpret_cast<const u8*>(user_data) + byte_offset, sizeof(ptr));
        } else {
            const VAddr parent_base = bases[n.parent];
            if (parent_base == 0 ||
                parent_base > std::numeric_limits<VAddr>::max() - byte_offset) {
                continue;
            }
            const VAddr pointer_address = parent_base + byte_offset;
            if (!replay && memory_reader) {
                const size_t parent = static_cast<size_t>(n.parent);
                const bool pointer_is_in_bulk_span =
                    bulk_reader_enabled && node_bulk_valid[parent] != 0 &&
                    n.ptr_off_dw <= node_span_dw[parent] &&
                    node_span_dw[parent] - n.ptr_off_dw >= sizeof(ptr) / sizeof(u32);
                if (pointer_is_in_bulk_span) {
                    std::memcpy(
                        &ptr,
                        bulk_node_words.data() + node_bulk_offsets[parent] + n.ptr_off_dw,
                        sizeof(ptr));
                } else if (!memory_reader(pointer_address, &ptr, sizeof(ptr))) {
                    continue;
                }
            } else {
                std::memcpy(&ptr, reinterpret_cast<const void*>(pointer_address), sizeof(ptr));
            }
        }
        ptr &= 0xFFFFFFFFFFFFULL;
        const u64 raw_ptr = ptr;
        if (replay && ptr != 0) {
            const u64 nb = Libraries::GnmDriver::ExecutorReplayRelocate(ptr);
            if (nb) {
                ptr = nb;
            } else if (!Libraries::GnmDriver::ExecutorReplayIsRelocated(ptr)) {
                ptr = 0;
            }
        }
#if defined(__ANDROID__)
        if (replay || live_diag_log) {
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_SRT_NODE] mode=%s stage=%u hash=0x%llx i=%zu parent=%d "
                                "ptrOff=%u raw=0x%llx resolved=0x%llx",
                                replay ? "replay" : "live",
                                stage, (unsigned long long)pgm_hash, i, n.parent, n.ptr_off_dw,
                                (unsigned long long)raw_ptr,
                                (unsigned long long)ptr);
        }
#endif
        bases[i] = ptr;
        if (memory_ranges && ptr != 0 && node_span_dw[i] != 0) {
            memory_ranges->push_back(
                SrtMemoryRange{ptr, static_cast<u64>(node_span_dw[i]) * 4u});
        }
        if (bulk_reader_enabled && ptr != 0 && node_span_dw[i] != 0) {
            auto* const node_words = bulk_node_words.data() + node_bulk_offsets[i];
            node_bulk_valid[i] =
                memory_reader(ptr, node_words,
                              static_cast<u64>(node_span_dw[i]) * sizeof(u32));
        }
        if (capture && raw_ptr != 0 && node_span_dw[i] != 0) {
            Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(
                6u, raw_ptr, static_cast<u64>(node_span_dw[i]) * 4u);
        }
    }
    u32 filled = 0, nonzero = 0;
    for (const auto& c : srt.portable_copies) {
        const VAddr base = bases[c.node];
        const u64 byte_offset = static_cast<u64>(c.src_off_dw) * sizeof(u32);
        if (base == 0 || base > std::numeric_limits<VAddr>::max() - byte_offset) {
            continue;
        }
        const VAddr source_address = base + byte_offset;
        u32 v{};
        if (!replay && memory_reader) {
            const bool value_is_in_bulk_span =
                bulk_reader_enabled && node_bulk_valid[c.node] != 0 &&
                c.src_off_dw < node_span_dw[c.node];
            if (value_is_in_bulk_span) {
                v = bulk_node_words[node_bulk_offsets[c.node] + c.src_off_dw];
            } else if (!memory_reader(source_address, &v, sizeof(v))) {
                continue;
            }
        } else {
            std::memcpy(&v, reinterpret_cast<const void*>(source_address), sizeof(v));
        }
        flat[c.dst_off_dw] = v;
        ++filled;
        if (v != 0) {
            ++nonzero;
        }
    }
#if defined(__ANDROID__)
    if (replay || live_diag_log) {
        char flat_dump[384]{};
        size_t flat_dump_len = 0;
        const u32 dump_dw = std::min<u32>(srt.flattened_bufsize_dw, 24u);
        for (u32 i = 0; i < dump_dw && flat_dump_len < sizeof(flat_dump); ++i) {
            const int written = std::snprintf(flat_dump + flat_dump_len,
                                              sizeof(flat_dump) - flat_dump_len,
                                              "%s%08x", i == 0 ? "" : ",", flat[i]);
            if (written <= 0) {
                break;
            }
            flat_dump_len += std::min<size_t>(static_cast<size_t>(written),
                                              sizeof(flat_dump) - flat_dump_len);
        }
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_SRT_WALK] mode=%s stage=%u hash=0x%llx nodes=%zu copies=%zu "
                            "filled=%u nonzero=%u flatDw=%u flat0_23=%s",
                            replay ? "replay" : "live",
                            stage, (unsigned long long)pgm_hash, srt.portable_nodes.size(),
                            srt.portable_copies.size(), filled, nonzero, srt.flattened_bufsize_dw,
                            flat_dump);
    }
#endif
}

}
