// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "common/arch.h"
#include "common/assert.h"
#include "common/config.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/string_util.h"
#include "common/thread.h"
#include "core/aerolib/aerolib.h"
#include "core/aerolib/stubs.h"
#include "core/devtools/widget/module_list.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/memory.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/sysmodule/sysmodule.h"
#include "core/linker.h"
#include "core/memory.h"
#include "core/tls.h"
#include "ipc/ipc.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#include <signal.h>
#include <sys/mman.h>
#endif
#ifdef __ANDROID__
#include <android/log.h>
extern "C" int executor_lsx4_android_run_main_entry(std::uint64_t entry_addr,
                                                       void* params,
                                                       void* exit_func);
extern "C" int executor_lsx4_probe_env_newline_icall_table(std::uint64_t mono_base);
extern "C" void executor_conformance_note_reloc_audit(std::uint64_t relative,
                                                       std::uint64_t imagebase_addends,
                                                       std::uint64_t icall_hits,
                                                       std::uint64_t fallback_repairs);
extern "C" int executor_lsx4_patch_stack_chk_calls(void* module_base, std::size_t module_size,
                                                       std::uint64_t stack_chk_plt,
                                                       const char* module_name);
extern "C" void executor_lsx4_android_register_guest_gc_sideband_range(
    const void* base, std::size_t size, const char* label) __attribute__((weak));

static bool ExecutorGcGuestRootsMarkerEnabled() {
    static const bool enabled = [] {
        std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/lsx4-home/run-gc-guest-roots", "r");
        if (f) {
            std::fclose(f);
            return true;
        }
        return false;
    }();
    return enabled;
}

static bool ExecutorVerboseLinkerTraceEnabled() {
    const char* env = std::getenv("EXECUTOR_VERBOSE_LINKER_TRACE");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static void ExecutorAuditLibcFamilyOwner(const char* nid, const char* importer,
                                         const char* source, const char* actual_owner,
                                         std::uint64_t address) {
    const bool allocator = Core::AeroLib::IsLibcAllocFamilyNid(nid);
    const bool stdio = Core::AeroLib::IsLibcStdioFamilyNid(nid);
    const bool directory = Core::AeroLib::IsLibcDirectoryFamilyNid(nid);
    if (!allocator && !stdio && !directory) {
        return;
    }
    const char* family = allocator ? "allocator" : (stdio ? "stdio" : "directory");
    const char* expected_owner =
        allocator && Core::AeroLib::ExecutorUseRealLibcAlloc() ? "guest-libc" : "executor-hle";
    const char* actual = actual_owner != nullptr ? actual_owner : "unknown";

    static std::mutex owner_mutex;
    static std::unordered_map<std::string, std::string> owners;
    static std::unordered_map<std::string, bool> family_ok_logged;
    static std::unordered_map<std::string, bool> mixed_owner_logged;
    bool first = false;
    bool mixed = false;
    bool first_family_record = false;
    std::string previous;
    {
        std::lock_guard lock(owner_mutex);
        const std::string key = family;
        auto [it, inserted] = owners.emplace(key, actual);
        first = inserted;
        if (!inserted && it->second != actual) {
            mixed = mixed_owner_logged.emplace(key, true).second;
            previous = it->second;
        }
        first_family_record = family_ok_logged.emplace(family, true).second;
    }
    if (!first && !mixed) {
        return;
    }
    const bool owner_matches = std::strcmp(expected_owner, actual) == 0;
    if (owner_matches && !mixed && !first_family_record) {
        return;
    }
    __android_log_print(owner_matches && !mixed ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
                        "LSX4Native",
                        "[EXECUTOR_LIBC_FAMILY_OWNER] status=%s family=%s nid=%s "
                        "expected=%s actual=%s previous=%s source=%s importer=%s addr=0x%llx "
                        "mixed=%d action=warn-only",
                        owner_matches && !mixed ? "ok" : "violation", family,
                        nid != nullptr ? nid : "<null>", expected_owner, actual,
                        previous.empty() ? "<none>" : previous.c_str(),
                        source != nullptr ? source : "<unknown>",
                        importer != nullptr ? importer : "<unknown>",
                        static_cast<unsigned long long>(address), mixed ? 1 : 0);
}

static bool ExecutorStackCanaryBranchPatchEnabled() {
    if (const char* env = std::getenv("EXECUTOR_PATCH_STACK_CANARY_BRANCHES");
        env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) {
        return true;
    }
    if (std::FILE* marker = std::fopen(
            "/data/data/app.lsx4.android/files/lsx4-home/"
            "run-live-stack-canary-branch-patch",
            "r")) {
        std::fclose(marker);
        return true;
    }
    return false;
}
#endif

namespace Core {

static void ExecutorLinkerTrace(const char* stage, const std::filesystem::path& path,
                                const std::string& detail = {}) {
#ifdef __ANDROID__
    if (!ExecutorVerboseLinkerTraceEnabled()) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LINKER_STAGE] stage=%s path=%s detail=%s", stage,
                        path.string().c_str(), detail.empty() ? "-" : detail.c_str());
#else
    (void)stage;
    (void)path;
    (void)detail;
#endif
}

static PS4_SYSV_ABI void ProgramExitFunc() {
    LOG_ERROR(Core_Linker, "Exit function called");
}

#ifdef ARCH_X86_64
static PS4_SYSV_ABI void* RunMainEntry [[noreturn]] (EntryParams* params) {
    asm volatile("andq $-16, %%rsp\n"
                 "subq $8, %%rsp\n"

                 "pushq 8(%1)\n"
                 "pushq 0(%1)\n"

                 "movq %1, %%rdi\n"
                 "movq %2, %%rsi\n"

                 "jmp *%0\n"
                 :
                 : "r"(params->entry_addr), "r"(params), "r"(ProgramExitFunc)
                 : "rax", "rsi", "rdi");
    UNREACHABLE();
}
#else
static PS4_SYSV_ABI void* RunMainEntry(EntryParams* params) {
#ifdef __ANDROID__
    if (params != nullptr) {
        const int rc = executor_lsx4_android_run_main_entry(
            params->entry_addr, params, reinterpret_cast<void*>(ProgramExitFunc));
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MAIN_ENTRY_BRIDGE] entry=%p params=%p rc=%d",
                            reinterpret_cast<void*>(params->entry_addr), params, rc);
        return nullptr;
    }
#endif
    UNREACHABLE_MSG("Native host cannot jump into PS4 x86-64 entry; Android must route entry "
                    "execution through the Box64 guest bridge.");
}
#endif

Linker::Linker() : memory{Memory::Instance()} {}

Linker::~Linker() = default;

void Linker::Execute(const std::vector<std::string>& args) {
    if (Config::debugDump()) {
        DebugDump();
    }

    Module* module = m_modules[0].get();
    PrepareStaticTlsForMainModule();

    for (const auto& m : m_modules) {
        Relocate(m.get());
    }

    u64 fmem_size = ORBIS_FLEXIBLE_MEMORY_SIZE;
    bool use_extended_mem1 = true, use_extended_mem2 = true;

    const auto* proc_param = GetProcParam();
    ASSERT(proc_param);

    Core::OrbisKernelMemParam mem_param{};
    if (proc_param->size >= offsetof(OrbisProcParam, mem_param) + sizeof(OrbisKernelMemParam*)) {
        if (proc_param->mem_param) {
            mem_param = *proc_param->mem_param;
            if (mem_param.size >=
                offsetof(OrbisKernelMemParam, flexible_memory_size) + sizeof(u64*)) {
                if (const auto* flexible_size = mem_param.flexible_memory_size) {
                    fmem_size = *flexible_size + ORBIS_FLEXIBLE_MEMORY_BASE;
                }
            }
        }
    }

    if (mem_param.size < offsetof(OrbisKernelMemParam, extended_memory_1) + sizeof(u64*)) {
        mem_param.extended_memory_1 = nullptr;
    }
    if (mem_param.size < offsetof(OrbisKernelMemParam, extended_memory_2) + sizeof(u64*)) {
        mem_param.extended_memory_2 = nullptr;
    }

    const u64 sdk_ver = proc_param->sdk_version;
    if (sdk_ver < Common::ElfInfo::FW_50) {
        use_extended_mem1 = mem_param.extended_memory_1 ? *mem_param.extended_memory_1 : false;
        use_extended_mem2 = mem_param.extended_memory_2 ? *mem_param.extended_memory_2 : false;
    }

    memory->SetupMemoryRegions(fmem_size, use_extended_mem1, use_extended_mem2);

    main_thread.Run([this, module, &args](std::stop_token) {
        Common::SetCurrentThreadName("Game:Main");
#ifndef _WIN32
        sigset_t emptyset;
        sigemptyset(&emptyset);
        pthread_sigmask(SIG_SETMASK, &emptyset, nullptr);
#endif
        if (auto& ipc = IPC::Instance()) {
            ipc.WaitForStart();
        }

        Libraries::SysModule::sceSysmodulePreloadModuleForLibkernel();

        s64 phys_addr{};
        s32 result = Libraries::Kernel::sceKernelAllocateDirectMemory(
            0, Libraries::Kernel::sceKernelGetDirectMemorySize(), 0x10000, 0x10000, 3, &phys_addr);
        if (result == 0) {
            void* addr{reinterpret_cast<void*>(0xfe0000000)};
            result = Libraries::Kernel::sceKernelMapNamedDirectMemory(
                &addr, 0x10000, 0x13, 0, phys_addr, 0x10000, "SceGnmDriver");
        }
        ASSERT_MSG(result == 0, "Unable to emulate libSceGnmDriver initialization");

        EntryParams& params = Libraries::Kernel::entry_params;
        params.argc = 1;
        params.argv[0] = "eboot.bin";
        if (!args.empty()) {
            params.argc = args.size();
            for (int i = 0; i < args.size() && i < 33; i++) {
                params.argv[i] = args[i].c_str();
            }
        }
        params.entry_addr = module->GetEntryAddress();
        Libraries::Kernel::ClearStack();
        RunMainEntry(&params);
    });
}

void Linker::PrepareStaticTlsForMainModule() {
    std::scoped_lock lk{mutex};
    if (m_modules.empty()) {
        return;
    }

    Module* module = m_modules[0].get();
    static_tls_size = module->tls.offset = module->tls.image_size;
}

void Linker::PrepareStaticTlsForModule(Module* module) {
    std::scoped_lock lk{mutex};
    if (module == nullptr) {
        return;
    }

    static_tls_size = module->tls.offset = module->tls.image_size;
}

s32 Linker::LoadModule(const std::filesystem::path& elf_name, bool is_dynamic) {
    std::scoped_lock lk{mutex};
    ExecutorLinkerTrace("load_module_enter", elf_name,
                        std::string("is_dynamic=") + (is_dynamic ? "1" : "0"));

    if (!std::filesystem::exists(elf_name)) {
        LOG_ERROR(Core_Linker, "Provided file {} does not exist", elf_name.string());
        ExecutorLinkerTrace("exists_failed", elf_name);
        return -1;
    }
    ExecutorLinkerTrace("exists_ok", elf_name);

    const u32 existing = FindByName(elf_name);
    if (existing != static_cast<u32>(-1)) {
        ExecutorLinkerTrace("already_loaded", elf_name,
                            std::string("handle=") + std::to_string(existing));
        return static_cast<s32>(existing);
    }

    ExecutorLinkerTrace("module_ctor_begin", elf_name);
    auto module = std::make_unique<Module>(memory, elf_name, max_tls_index);
    ExecutorLinkerTrace("module_ctor_returned", elf_name,
                        std::string("valid=") + (module->IsValid() ? "1" : "0"));
    if (!module->IsValid()) {
        LOG_ERROR(Core_Linker, "Provided file {} is not valid ELF file", elf_name.string());
        ExecutorLinkerTrace("module_invalid", elf_name);
        return -1;
    }

    num_static_modules += !is_dynamic;
    ExecutorLinkerTrace("emplace_begin", elf_name,
                        std::string("next_index=") + std::to_string(m_modules.size()));
    m_modules.emplace_back(std::move(module));
    ExecutorLinkerTrace("emplace_done", elf_name,
                        std::string("index=") + std::to_string(m_modules.size() - 1));

    Core::Devtools::Widget::ModuleList::AddModule(elf_name.filename().string(), elf_name);
    ExecutorLinkerTrace("devtools_registered", elf_name);

    return m_modules.size() - 1;
}

s32 Linker::LoadAndStartModule(const std::filesystem::path& path, u64 args, const void* argp,
                               int* pRes) {
    u32 handle = FindByName(path);
    if (handle != -1) {
        return handle;
    }
    handle = LoadModule(path, true);
    if (handle == -1) {
        return -1;
    }
    auto* module = GetModule(handle);
    RelocateAnyImports(module);

    if (module->tls.image_size != 0) {
        AdvanceGenerationCounter();
    }

    auto* param = module->GetProcParam<OrbisProcParam*>();
    ASSERT_MSG(!param || param->size >= 0x18, "Invalid module param size: {}", param->size);
    s32 ret = module->Start(args, argp, param);
    if (pRes) {
        *pRes = ret;
    }

    return handle;
}

Module* Linker::FindByAddress(VAddr address) {
    for (auto& module : m_modules) {
        const VAddr base = module->GetBaseAddress();
        if (address >= base && address < base + module->aligned_base_size) {
            return module.get();
        }
    }
    return nullptr;
}

void Linker::Relocate(Module* module) {
#ifdef __ANDROID__
    const bool executor_trace_mono_rela =
        module && module->file.filename().string() == "mono-ps4.sprx";
    u64 executor_relative_count = 0;
    u64 executor_relative_imagebase_addend_count = 0;
    u64 executor_icall_rela_hits = 0;
#endif
    module->ForEachRelocation([&](elf_relocation* rel, u32 i, bool is_jmp_rel) {
        const u32 num_relocs = module->dynamic_info.relocation_table_size / sizeof(elf_relocation);
        const u32 bit_idx = (is_jmp_rel ? num_relocs : 0) + i;
        if (module->TestRelaBit(bit_idx)) {
            return;
        }
        auto type = rel->GetType();
        auto symbol = rel->GetSymbol();
        auto addend = rel->rel_addend;
        auto* symbol_table = module->dynamic_info.symbol_table;
        auto* names_tlb = module->dynamic_info.str_table;

        const VAddr rel_base_virtual_addr = module->GetBaseAddress();
        const VAddr rel_virtual_addr = rel_base_virtual_addr + rel->rel_offset;
        bool rel_is_resolved = false;
        u64 rel_value = 0;
        Loader::SymbolType rel_sym_type = Loader::SymbolType::Unknown;
        std::string rel_name;
        std::string rel_raw_name;

        switch (type) {
        case R_X86_64_RELATIVE: {
#ifdef __ANDROID__
            if (executor_trace_mono_rela) {
                ++executor_relative_count;
                constexpr u64 kMonoTypeNamesOff = 0x1f5130;
                constexpr u64 kMonoMethodNamesOff = 0x1f5420;
                constexpr u64 kMonoFuncsOff = 0x1f67d0;
                const u64 off = rel->rel_offset;
                if ((off >= kMonoTypeNamesOff && off < kMonoTypeNamesOff + 0x2f0) ||
                    (off >= kMonoMethodNamesOff && off < kMonoMethodNamesOff + 0x13b0) ||
                    (off >= kMonoFuncsOff && off < kMonoFuncsOff + 0x1400)) {
                    ++executor_icall_rela_hits;
                }
            }
#endif
            rel_value = rel_base_virtual_addr + addend;
#ifdef __ANDROID__
            constexpr s64 kLinkedImageBase = 0x100000000ll;
            if (addend >= kLinkedImageBase &&
                static_cast<u64>(addend - kLinkedImageBase) < module->aligned_base_size) {
                rel_value = rel_base_virtual_addr + static_cast<u64>(addend - kLinkedImageBase);
                if (executor_trace_mono_rela) {
                    ++executor_relative_imagebase_addend_count;
                }
            }
#endif
            rel_is_resolved = true;
            module->SetRelaBit(bit_idx);
            break;
        }
        case R_X86_64_DTPMOD64:
            rel_value = static_cast<u64>(module->tls.modid);
            rel_is_resolved = true;
            rel_sym_type = Loader::SymbolType::Tls;
            module->SetRelaBit(bit_idx);
            break;
        case R_X86_64_GLOB_DAT:
        case R_X86_64_JUMP_SLOT:
            addend = 0;
        case R_X86_64_64: {
            auto sym = symbol_table[symbol];
            auto sym_bind = sym.GetBind();
            auto sym_type = sym.GetType();
            auto sym_visibility = sym.GetVisibility();
            u64 symbol_virtual_addr = 0;
            Loader::SymbolRecord symrec{};
            switch (sym_type) {
            case STT_FUN:
                rel_sym_type = Loader::SymbolType::Function;
                break;
            case STT_OBJECT:
                rel_sym_type = Loader::SymbolType::Object;
                break;
            case STT_NOTYPE:
                rel_sym_type = Loader::SymbolType::NoType;
                break;
            default:
                ASSERT_MSG(0, "unknown symbol type {}", sym_type);
            }

            if (sym_visibility != 0) {
                LOG_INFO(Core_Linker, "symbol visibility !=0");
            }

            switch (sym_bind) {
            case STB_LOCAL:
                symbol_virtual_addr = rel_base_virtual_addr + sym.st_value;
                module->SetRelaBit(bit_idx);
                break;
            case STB_GLOBAL:
            case STB_WEAK: {
                rel_name = names_tlb + sym.st_name;
                rel_raw_name = rel_name;
                if (Resolve(rel_name, rel_sym_type, module, &symrec)) {
                    module->SetRelaBit(bit_idx);
                }
                symbol_virtual_addr = symrec.virtual_address;
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown bind type {}", sym_bind);
            }
            rel_is_resolved = (symbol_virtual_addr != 0);
            rel_value = (rel_is_resolved ? symbol_virtual_addr + addend : 0);
            rel_name = symrec.name;
            break;
        }
        default:
            LOG_INFO(Core_Linker, "UNK type {:#010x} rel symbol : {:#010x}", type, symbol);
        }

#ifdef __ANDROID__
        if (executor_trace_mono_rela && is_jmp_rel && ExecutorVerboseLinkerTraceEnabled()) {
            const u64 plt_guess = rel_base_virtual_addr + 0xe0ull + static_cast<u64>(i) * 0x10ull;
            std::string nid_name;
            if (!rel_raw_name.empty()) {
                const auto hash = rel_raw_name.find('#');
                const std::string nid = hash == std::string::npos ? rel_raw_name
                                                                  : rel_raw_name.substr(0, hash);
                if (const auto aeronid = AeroLib::FindByNid(nid.c_str())) {
                    nid_name = aeronid->name ? aeronid->name : "";
                }
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_MONO_PLT_MAP] index=%u plt=0x%llx slot=0x%llx relOff=0x%llx "
                "type=%u sym=%u raw=%s human=%s resolvedName=%s resolved=0x%llx ok=%u addend=0x%llx",
                i, static_cast<unsigned long long>(plt_guess),
                static_cast<unsigned long long>(rel_virtual_addr),
                static_cast<unsigned long long>(rel->rel_offset), static_cast<unsigned>(type),
                static_cast<unsigned>(symbol), rel_raw_name.empty() ? "-" : rel_raw_name.c_str(),
                nid_name.empty() ? "-" : nid_name.c_str(), rel_name.empty() ? "-" : rel_name.c_str(),
                static_cast<unsigned long long>(rel_value), rel_is_resolved ? 1u : 0u,
                static_cast<unsigned long long>(addend));
        }
#endif

        if (rel_is_resolved) {
#ifdef __ANDROID__
            u64 old_value = 0;
            std::memcpy(&old_value, reinterpret_cast<void*>(rel_virtual_addr), sizeof(old_value));
            const bool cusa_plt_probe =
                module->file.string().find("CUSA00754") != std::string::npos &&
                module->file.string().find("eboot.bin") != std::string::npos &&
                old_value >= 0x20f0000 && old_value < 0x2100000;
            const bool cusa_stack_guard_probe =
                module->file.string().find("CUSA00754") != std::string::npos &&
                module->file.string().find("eboot.bin") != std::string::npos &&
                (rel_virtual_addr == 0x802811a98ull || rel->rel_offset == 0x2811a98ull ||
                 rel_raw_name.find("f7uOxY9mM1U") != std::string::npos ||
                 rel_name.find("__stack_chk_guard") != std::string::npos);
            const bool trace_reloc =
                rel_raw_name.find("mono-ps4") != std::string::npos ||
                rel_raw_name.find("-pnj3-7a6QA") != std::string::npos ||
                rel_name.find("mono") != std::string::npos ||
                rel_name.find("unity_mono") != std::string::npos || cusa_plt_probe ||
                cusa_stack_guard_probe;
            if (trace_reloc && ExecutorVerboseLinkerTraceEnabled()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_RELOC_SLOT] module=%s raw=%s resolvedName=%s type=%u bit=%u "
                    "jmp=%u relOff=0x%llx relVA=0x%llx old=0x%llx new=0x%llx addend=0x%llx",
                    module->file.string().c_str(), rel_raw_name.c_str(), rel_name.c_str(),
                    static_cast<unsigned>(type), bit_idx, static_cast<unsigned>(is_jmp_rel),
                    static_cast<unsigned long long>(rel->rel_offset),
                    static_cast<unsigned long long>(rel_virtual_addr),
                    static_cast<unsigned long long>(old_value),
                    static_cast<unsigned long long>(rel_value),
                    static_cast<unsigned long long>(addend));
            }
#endif
            std::memcpy(reinterpret_cast<void*>(rel_virtual_addr), &rel_value, sizeof(rel_value));
#ifdef __ANDROID__
            const bool imported_stack_chk_fail =
                is_jmp_rel && (rel_raw_name.find("Ou3iL1abvng") != std::string::npos ||
                               rel_raw_name.find("__stack_chk_fail") != std::string::npos ||
                               rel_name.find("__stack_chk_fail") != std::string::npos);
            if (imported_stack_chk_fail) {
                u64 stack_chk_plt =
                    rel_base_virtual_addr + 0xe0ull + static_cast<u64>(i) * 0x10ull;
                if (old_value >= 6 && old_value < module->aligned_base_size) {
                    const u64 candidate = rel_base_virtual_addr + old_value - 6;
                    const auto* candidate_bytes =
                        reinterpret_cast<const std::uint8_t*>(candidate);
                    if (candidate_bytes[0] == 0xff && candidate_bytes[1] == 0x25) {
                        stack_chk_plt = candidate;
                    }
                }
                const bool patch_stack_canary_branches =
                    ExecutorStackCanaryBranchPatchEnabled();
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_STACK_CANARY_PATCH] stack_chk_plt_resolved module=%s "
                    "old=0x%llx plt=0x%llx fallback=0x%llx branchPatch=%u",
                    module->file.string().c_str(),
                    static_cast<unsigned long long>(old_value),
                    static_cast<unsigned long long>(stack_chk_plt),
                    static_cast<unsigned long long>(rel_base_virtual_addr + 0xe0ull +
                                                    static_cast<u64>(i) * 0x10ull),
                    patch_stack_canary_branches ? 1u : 0u);
                if (patch_stack_canary_branches) {
                    executor_lsx4_patch_stack_chk_calls(
                        reinterpret_cast<void*>(module->GetBaseAddress()),
                        static_cast<std::size_t>(module->aligned_base_size), stack_chk_plt,
                        module->file.string().c_str());
                }
            }
#endif
        } else {
            LOG_INFO(Core_Linker, "Function not patched! {}", rel_name);
        }
    });
#ifdef __ANDROID__
    if (executor_trace_mono_rela) {
        constexpr u64 kLinkedImageBase = 0x100000000ull;
        constexpr u64 kMonoBase = 0x803190000ull;
        constexpr u64 kMonoTypeNamesOff = 0x1f5130;
        constexpr u64 kMonoMethodNamesOff = 0x1f5420;
        constexpr u64 kMonoFuncsOff = 0x1f67d0;
        constexpr u64 kGetNewLineIndex = 101;

        auto readq = [](u64 addr) -> u64 {
            u64 v = 0;
            std::memcpy(&v, reinterpret_cast<const void*>(addr), sizeof(v));
            return v;
        };
        auto writeq = [](u64 addr, u64 value) {
            std::memcpy(reinterpret_cast<void*>(addr), &value, sizeof(value));
        };
        const u64 base = module->GetBaseAddress();
        const u64 funcs101_addr = base + kMonoFuncsOff + kGetNewLineIndex * sizeof(u64);
        const u64 funcs101_before = readq(funcs101_addr);
        u64 repaired = 0;

        auto repair_pointer_array = [&](u64 off, u64 count) {
            for (u64 i = 0; i < count; ++i) {
                const u64 addr = base + off + i * sizeof(u64);
                const u64 old = readq(addr);
                if (old >= kLinkedImageBase &&
                    old < kLinkedImageBase + module->aligned_base_size) {
                    writeq(addr, base + (old - kLinkedImageBase));
                    ++repaired;
                }
            }
        };

        if (funcs101_before >= kLinkedImageBase &&
            funcs101_before < kLinkedImageBase + module->aligned_base_size) {
            repair_pointer_array(kMonoTypeNamesOff, (kMonoMethodNamesOff - kMonoTypeNamesOff) / 8);
            repair_pointer_array(kMonoMethodNamesOff, (kMonoFuncsOff - kMonoMethodNamesOff) / 8);
            repair_pointer_array(kMonoFuncsOff, 700);
        }

        const u64 funcs101_after = readq(funcs101_addr);
        const u64 method_name101 = readq(base + kMonoMethodNamesOff + kGetNewLineIndex * sizeof(u64));
        const u64 type_name0 = readq(base + kMonoTypeNamesOff);
        __android_log_print(
            ANDROID_LOG_ERROR, "LSX4Native",
            "[EXECUTOR_MONO_RELA_PROBE] base=0x%llx linkedBase=0x%llx relRelative=%llu "
            "imagebaseAddends=%llu icallRelHits=%llu funcs101_before=0x%llx funcs101_after=0x%llx "
            "expected=0x%llx methodName101=0x%llx typeName0=0x%llx fallbackRepaired=%llu",
            static_cast<unsigned long long>(base),
            static_cast<unsigned long long>(kLinkedImageBase),
            static_cast<unsigned long long>(executor_relative_count),
            static_cast<unsigned long long>(executor_relative_imagebase_addend_count),
            static_cast<unsigned long long>(executor_icall_rela_hits),
            static_cast<unsigned long long>(funcs101_before),
            static_cast<unsigned long long>(funcs101_after),
            static_cast<unsigned long long>(kMonoBase + 0x3b3a0),
            static_cast<unsigned long long>(method_name101),
            static_cast<unsigned long long>(type_name0),
            static_cast<unsigned long long>(repaired));
        executor_conformance_note_reloc_audit(executor_relative_count,
                                              executor_relative_imagebase_addend_count,
                                              executor_icall_rela_hits, repaired);
        const int env_probe_rc = executor_lsx4_probe_env_newline_icall_table(base);
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_MONO_RELA_PROBE] envNewLineRetProbeRc=%d base=0x%llx",
                            env_probe_rc, static_cast<unsigned long long>(base));
    }
#endif
}

const Module* Linker::FindExportedModule(const ModuleInfo& module, const LibraryInfo& library) {
    const auto it = std::ranges::find_if(m_modules, [&](const auto& m) {
        return std::ranges::contains(m->GetExportLibs(), library) &&
               std::ranges::contains(m->GetExportModules(), module);
    });
    return it == m_modules.end() ? nullptr : it->get();
}

#ifdef __ANDROID__
static bool IsAndroidGuestLibcSurfaceName(const std::string& name) {
    return name == "libc" || name == "libSceLibcInternal";
}

static const Loader::SymbolRecord* FindAndroidGuestLibcExport(
    const Module& candidate, const Loader::SymbolResolver& requested) {
    const auto export_libraries = candidate.GetExportLibs();
    const auto export_modules = candidate.GetExportModules();
    for (const auto& export_library : export_libraries) {
        if (!IsAndroidGuestLibcSurfaceName(export_library.name)) {
            continue;
        }
        for (const auto& export_module : export_modules) {
            if (!IsAndroidGuestLibcSurfaceName(export_module.name)) {
                continue;
            }
            Loader::SymbolResolver alias = requested;
            alias.library = export_library.name;
            alias.library_version = export_library.version;
            alias.module = export_module.name;
            if (const auto* record = candidate.export_sym.FindSymbol(alias);
                record != nullptr && record->virtual_address != 0) {
                return record;
            }
        }
    }
    return nullptr;
}

static const Loader::SymbolRecord* FindAndroidMonoExportRelaxed(const Module& candidate,
                                                                const Loader::SymbolResolver& sr) {
    if (candidate.file.string().find("mono-ps4") == std::string::npos) {
        return nullptr;
    }
    if (sr.library != "mono-ps4" && sr.module != "mono-ps4") {
        return nullptr;
    }
    const std::string nid_prefix = sr.name + "#";
    const auto aeronid = AeroLib::FindByNid(sr.name.c_str());
    const char* nid_human = aeronid ? aeronid->name : nullptr;
    const auto symbols = candidate.export_sym.GetSymbols();
    for (const auto& record : candidate.export_sym.GetSymbols()) {
        const bool nid_prefix_match = record.name.rfind(nid_prefix, 0) == 0;
        const bool nid_human_match =
            nid_human != nullptr && !record.nid_name.empty() && record.nid_name == nid_human;
        if (!nid_prefix_match && !nid_human_match) {
            continue;
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_MONO_EXPORT_RELAXED_HIT] nid=%s human=%s record=%s recordHuman=%s "
            "addr=0x%llx exportModule=%s",
            sr.name.c_str(), nid_human ? nid_human : "", record.name.c_str(),
            record.nid_name.c_str(), static_cast<unsigned long long>(record.virtual_address),
            candidate.file.string().c_str());
        return &record;
    }
    static std::atomic<u32> miss_logs{0};
    const u32 miss_idx = miss_logs.fetch_add(1, std::memory_order_relaxed);
    if (miss_idx < 16) {
        std::string sample;
        for (size_t i = 0; i < std::min<size_t>(symbols.size(), 6); ++i) {
            if (!sample.empty()) {
                sample += ",";
            }
            sample += symbols[i].name + "/" + symbols[i].nid_name;
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_MONO_EXPORT_RELAXED_MISS] nid=%s human=%s exports=%zu sample=%s "
            "exportModule=%s",
            sr.name.c_str(), nid_human ? nid_human : "", symbols.size(), sample.c_str(),
            candidate.file.string().c_str());
    }
    return nullptr;
}
#endif

bool Linker::Resolve(const std::string& name, Loader::SymbolType sym_type, Module* m,
                     Loader::SymbolRecord* return_info) {
    const auto ids = Common::SplitString(name, '#');
    if (ids.size() != 3) {
        return_info->virtual_address = 0;
        return_info->name = name;
        LOG_ERROR(Core_Linker, "Not Resolved {}", name);
        return false;
    }

    const LibraryInfo* library = m->FindLibrary(ids[1]);
    const ModuleInfo* module = m->FindModule(ids[2]);
    ASSERT_MSG(library && module, "Unable to find library and module");

    Loader::SymbolResolver sr{};
    sr.name = ids.at(0);
    sr.library = library->name;
    sr.library_version = library->version;
    sr.module = module->name;
    sr.type = sym_type;

#ifdef __ANDROID__
    if (sr.name == "f7uOxY9mM1U" && sym_type != Loader::SymbolType::Function) {
        return_info->name = "__stack_chk_guard";
        return_info->virtual_address =
            reinterpret_cast<u64>(Libraries::Kernel::GetStackChkGuardExportStorage());
        const auto type_name = Loader::SymbolsResolver::SymbolTypeToS(sym_type);
        if (ExecutorVerboseLinkerTraceEnabled()) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_LINK_RESOLVE] source=executor_data_object importer=%s nid=%s lib=%s "
                "mod=%s type=%.*s addr=0x%llx value=0x%llx",
                m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(), sr.module.c_str(),
                static_cast<int>(type_name.size()), type_name.data(),
                static_cast<unsigned long long>(return_info->virtual_address),
                static_cast<unsigned long long>(
                    *reinterpret_cast<u64*>(return_info->virtual_address)));
        }
        return true;
    }
#endif

#ifdef __ANDROID__
    if (sym_type == Loader::SymbolType::Object) {
        if (const Loader::SymbolRecord* object_record = m_hle_symbols.FindSymbol(sr);
            object_record != nullptr && object_record->virtual_address != 0) {
            *return_info = *object_record;
            Core::Devtools::Widget::ModuleList::AddModule(sr.library);
            if (ExecutorVerboseLinkerTraceEnabled()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LINK_RESOLVE] source=hle_data_object_pc_order importer=%s "
                    "nid=%s lib=%s mod=%s addr=0x%llx name=%s",
                    m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(),
                    sr.module.c_str(),
                    static_cast<unsigned long long>(return_info->virtual_address),
                    return_info->name.c_str());
            }
            return true;
        }
    }
#endif

#ifdef __ANDROID__
    const bool guest_owned_libc_allocator =
        sym_type == Loader::SymbolType::Function && Core::AeroLib::ExecutorUseRealLibcAlloc() &&
        Core::AeroLib::IsLibcAllocFamilyNid(sr.name.c_str());
    if (guest_owned_libc_allocator) {
        for (const auto& candidate : m_modules) {
            const auto* record = FindAndroidGuestLibcExport(*candidate, sr);
            if (record == nullptr) {
                continue;
            }
            *return_info = *record;
            Core::Devtools::Widget::ModuleList::AddModule(sr.library);
            ExecutorAuditLibcFamilyOwner(sr.name.c_str(), m->file.string().c_str(),
                                         "guest_libc_allocator_export", "guest-libc",
                                         return_info->virtual_address);
            if (ExecutorVerboseLinkerTraceEnabled()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LINK_RESOLVE] source=guest_libc_allocator_export importer=%s "
                    "nid=%s lib=%s mod=%s addr=0x%llx exportModule=%s",
                    m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(),
                    sr.module.c_str(),
                    static_cast<unsigned long long>(return_info->virtual_address),
                    candidate->file.string().c_str());
            }
            return true;
        }

        return_info->name = name;
        return_info->virtual_address = 0;
        if (ExecutorVerboseLinkerTraceEnabled()) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_LINK_RESOLVE] source=guest_libc_allocator_deferred importer=%s "
                "nid=%s lib=%s mod=%s loadedModuleCount=%zu",
                m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(),
                sr.module.c_str(), m_modules.size());
        }
        return false;
    }

    const bool is_libc_os_surface =
        sym_type == Loader::SymbolType::Function &&
        (sr.module == "libc" || sr.module == "libSceLibcInternal" ||
         sr.library == "libc" || sr.library == "libSceLibcInternal");
    const bool importer_is_guest_libc =
        m->file.filename() == "libc.prx" || m->file.filename() == "libSceLibcInternal.prx";
    if (is_libc_os_surface && !importer_is_guest_libc) {
        if (Core::AeroLib::TryGetAndroidX64ExecutorOverride(
                sr.name.c_str(), &return_info->virtual_address, &return_info->name)) {
            ExecutorAuditLibcFamilyOwner(sr.name.c_str(), m->file.string().c_str(),
                                         "executor_override_libc_owner", "executor-hle",
                                         return_info->virtual_address);
            if (ExecutorVerboseLinkerTraceEnabled()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LINK_RESOLVE] source=executor_override_libc_owner importer=%s "
                    "nid=%s lib=%s mod=%s addr=0x%llx name=%s",
                    m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(),
                    sr.module.c_str(),
                    static_cast<unsigned long long>(return_info->virtual_address),
                    return_info->name.c_str());
            }
            return true;
        }

        Loader::SymbolResolver hle_sr = sr;
        const Loader::SymbolRecord* hle_record = m_hle_symbols.FindSymbol(hle_sr);
        if (hle_record == nullptr) {
            hle_sr.library = "libSceLibcInternal";
            hle_sr.module = "libSceLibcInternal";
            hle_record = m_hle_symbols.FindSymbol(hle_sr);
        }
        if (hle_record != nullptr && hle_record->virtual_address != 0) {
            const u64 x64_stub = Core::AeroLib::GetAndroidX64HleStubForNative(
                hle_record->name.c_str(), hle_record->virtual_address);
            if (x64_stub != 0) {
                *return_info = *hle_record;
                return_info->virtual_address = x64_stub;
                Core::Devtools::Widget::ModuleList::AddModule(sr.library);
                ExecutorAuditLibcFamilyOwner(sr.name.c_str(), m->file.string().c_str(),
                                             "hle_libc_owner", "upstream-hle",
                                             return_info->virtual_address);
                if (ExecutorVerboseLinkerTraceEnabled()) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LINK_RESOLVE] source=hle_libc_owner importer=%s nid=%s "
                        "lib=%s mod=%s native=0x%llx stub=0x%llx name=%s",
                        m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(),
                        sr.module.c_str(),
                        static_cast<unsigned long long>(hle_record->virtual_address),
                        static_cast<unsigned long long>(x64_stub), return_info->name.c_str());
                }
                return true;
            }
        }
    }
#endif

    const auto* p = FindExportedModule(*module, *library);
    if (p && p->export_sym.GetSize() > 0) {
        const auto* record = p->export_sym.FindSymbol(sr);
        if (record) {
            *return_info = *record;
#ifdef __ANDROID__
            if (sr.name == "WqHfqjSXV0k" || sr.name == "8GjM2JWuepQ") {
                {
                    Loader::SymbolResolver hr = sr;
                    hr.library = "mono-ps4";
                    hr.module = "mono-ps4";
                    hr.name = "PKaE-B7gmy8";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoInstallSearchHook(hrec->virtual_address);
                            break;
                        }
                    }
                    hr.name = "TYMPK-S1pDk";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoImageLoaded(hrec->virtual_address);
                            break;
                        }
                    }
                    hr.name = "o5qQozz5NCI";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoImageGetAssembly(hrec->virtual_address);
                            break;
                        }
                    }
                    hr.name = "YJp-6AZoYfM";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoLoadFromFull(hrec->virtual_address);
                            break;
                        }
                    }
                    hr.name = "-Vt1ihrzI0Q";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoAddInternalCall(hrec->virtual_address);
                            if (ExecutorVerboseLinkerTraceEnabled()) {
                                __android_log_print(
                                    ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_LINK_RESOLVE] captured "
                                    "mono_add_internal_call=0x%llx",
                                    static_cast<unsigned long long>(hrec->virtual_address));
                            }
                            break;
                        }
                    }
                    hr.name = "pXv-WCokNsY";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoStringNew(hrec->virtual_address);
                            if (ExecutorVerboseLinkerTraceEnabled()) {
                                __android_log_print(
                                    ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_LINK_RESOLVE] captured mono_string_new=0x%llx",
                                    static_cast<unsigned long long>(hrec->virtual_address));
                            }
                            break;
                        }
                    }
                    hr.name = "iAlTedwf89k";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoDomainGet(hrec->virtual_address);
                            if (ExecutorVerboseLinkerTraceEnabled()) {
                                __android_log_print(
                                    ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_LINK_RESOLVE] captured mono_domain_get=0x%llx",
                                    static_cast<unsigned long long>(hrec->virtual_address));
                            }
                            break;
                        }
                    }
                    hr.name = "dCeihPtadCM";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoDomainAssemblyOpen(hrec->virtual_address);
                            if (ExecutorVerboseLinkerTraceEnabled()) {
                                __android_log_print(
                                    ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_LINK_RESOLVE] captured "
                                    "mono_domain_assembly_open=0x%llx",
                                    static_cast<unsigned long long>(hrec->virtual_address));
                            }
                            break;
                        }
                    }
                }
                const u64 interp = Core::AeroLib::ExecutorMakeMonoPathInterposer(
                    record->virtual_address, sr.name == "WqHfqjSXV0k");
                if (ExecutorVerboseLinkerTraceEnabled()) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LINK_RESOLVE] source=mono_path_interpose importer=%s nid=%s "
                        "real=0x%llx interposer=0x%llx",
                        m->file.string().c_str(), sr.name.c_str(),
                        static_cast<unsigned long long>(record->virtual_address),
                        static_cast<unsigned long long>(interp));
                }
                return_info->virtual_address = interp;
                return true;
            }
            if (sr.name == "dCeihPtadCM") {
                Core::AeroLib::ExecutorSetMonoDomainAssemblyOpen(record->virtual_address);
                {
                    Loader::SymbolResolver hr = sr;
                    hr.library = "mono-ps4"; hr.module = "mono-ps4"; hr.name = "-Vt1ihrzI0Q";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoAddInternalCall(hrec->virtual_address);
                            break;
                        }
                    }
                    hr.name = "pXv-WCokNsY";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoStringNew(hrec->virtual_address);
                            break;
                        }
                    }
                    hr.name = "iAlTedwf89k";
                    for (const auto& candidate : m_modules) {
                        if (const auto* hrec = FindAndroidMonoExportRelaxed(*candidate, hr)) {
                            Core::AeroLib::ExecutorSetMonoDomainGet(hrec->virtual_address);
                            break;
                        }
                    }
                }
                const u64 interp =
                    Core::AeroLib::ExecutorMakeMonoDomainOpenInterposer(record->virtual_address);
                if (ExecutorVerboseLinkerTraceEnabled()) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LINK_RESOLVE] source=mono_domain_open_interpose nid=%s "
                        "real=0x%llx interposer=0x%llx",
                        sr.name.c_str(), static_cast<unsigned long long>(record->virtual_address),
                        static_cast<unsigned long long>(interp));
                }
                return_info->virtual_address = interp;
                return true;
            }
            if (sr.name == "O1ecX4iuwO8") {
                const u64 interp =
                    Core::AeroLib::ExecutorMakeMonoConfigInterposer(record->virtual_address);
                return_info->virtual_address = interp;
                return true;
            }
#endif
#ifdef __ANDROID__
            const bool is_locale_export_frontier =
                sr.module == "libc" &&
                (sr.name == "9rMML086SEE" || sr.name == "hEQ2Yi4PJXA");
            if (ExecutorVerboseLinkerTraceEnabled() &&
                (sr.name == "-pnj3-7a6QA" || sr.library == "mono-ps4" ||
                 sr.module == "mono-ps4" || is_locale_export_frontier)) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LINK_RESOLVE] source=export importer=%s nid=%s lib=%s mod=%s "
                    "addr=0x%llx exportModule=%s",
                    m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(),
                    sr.module.c_str(), static_cast<unsigned long long>(record->virtual_address),
                    p->file.string().c_str());
            }
#endif
#ifdef __ANDROID__
            ExecutorAuditLibcFamilyOwner(sr.name.c_str(), m->file.string().c_str(), "export",
                                         "guest-libc", return_info->virtual_address);
#endif
            return true;
        }
    }

#ifdef __ANDROID__
    if (sr.library == "mono-ps4" || sr.module == "mono-ps4") {
        for (const auto& candidate : m_modules) {
            const auto* record = FindAndroidMonoExportRelaxed(*candidate, sr);
            if (!record) {
                continue;
            }
            *return_info = *record;
            if (ExecutorVerboseLinkerTraceEnabled()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LINK_RESOLVE] source=export_mono_fallback importer=%s nid=%s "
                    "lib=%s mod=%s addr=0x%llx exportModule=%s",
                    m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(), sr.module.c_str(),
                    static_cast<unsigned long long>(record->virtual_address),
                    candidate->file.string().c_str());
            }
            return true;
        }
        if (ExecutorVerboseLinkerTraceEnabled()) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LINK_RESOLVE] source=export_mono_miss importer=%s nid=%s "
                                "lib=%s mod=%s loadedModuleCount=%zu",
                                m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(),
                                sr.module.c_str(), m_modules.size());
        }
    }
#endif

#ifdef __ANDROID__
    if (std::getenv("EXECUTOR_LIVE_EBOOT_LIBC_COMPARE_SPLIT") != nullptr &&
        sr.library == "libc" && sr.module == "libc" &&
        std::filesystem::path(m->file).filename() == "eboot.bin" &&
        (sr.name == "Ovb2dSJOAuE" || sr.name == "aesyjrHVWy4")) {
        if (sr.name == "Ovb2dSJOAuE") {
            return_info->name = "strncmp";
            return_info->virtual_address =
                reinterpret_cast<u64>(&Core::AeroLib::ExecutorLibcStrncmp);
        } else {
            return_info->name = "strcmp";
            return_info->virtual_address =
                reinterpret_cast<u64>(&Core::AeroLib::ExecutorLibcStrcmp);
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LINK_RESOLVE] source=executor_override_eboot_libc_split "
            "importer=%s nid=%s lib=%s mod=%s addr=0x%llx name=%s",
            m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(), sr.module.c_str(),
            static_cast<unsigned long long>(return_info->virtual_address),
            return_info->name.c_str());
        return true;
    }

    if (Core::AeroLib::TryGetAndroidX64ExecutorOverride(sr.name.c_str(),
                                                        &return_info->virtual_address,
                                                        &return_info->name)) {
        ExecutorAuditLibcFamilyOwner(sr.name.c_str(), m->file.string().c_str(),
                                     "executor_override", "executor-hle",
                                     return_info->virtual_address);
        if (ExecutorVerboseLinkerTraceEnabled()) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_LINK_RESOLVE] source=executor_override importer=%s nid=%s lib=%s mod=%s "
                "addr=0x%llx name=%s",
                m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(), sr.module.c_str(),
                static_cast<unsigned long long>(return_info->virtual_address),
                return_info->name.c_str());
        }
        return true;
    }
#endif

    const auto* record = m_hle_symbols.FindSymbol(sr);
    if (record) {
        *return_info = *record;
        Core::Devtools::Widget::ModuleList::AddModule(sr.library);
#ifdef __ANDROID__
        if (sym_type == Loader::SymbolType::Function && record->virtual_address != 0) {
            const u64 x64_stub = Core::AeroLib::GetAndroidX64HleStubForNative(
                return_info->name.c_str(), record->virtual_address);
            if (x64_stub != 0) {
                if (ExecutorVerboseLinkerTraceEnabled()) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LINK_RESOLVE] source=hle_x64_stub importer=%s nid=%s lib=%s "
                        "mod=%s native=0x%llx stub=0x%llx name=%s",
                        m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(),
                        sr.module.c_str(), static_cast<unsigned long long>(record->virtual_address),
                        static_cast<unsigned long long>(x64_stub), return_info->name.c_str());
                }
                return_info->virtual_address = x64_stub;
            }
        }
        if (ExecutorVerboseLinkerTraceEnabled() &&
            (sr.name == "-pnj3-7a6QA" || sr.library == "mono-ps4" ||
             sr.module == "mono-ps4")) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_LINK_RESOLVE] source=hle importer=%s nid=%s lib=%s mod=%s addr=0x%llx",
                m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(), sr.module.c_str(),
                static_cast<unsigned long long>(return_info->virtual_address));
        }
#endif
#ifdef __ANDROID__
        ExecutorAuditLibcFamilyOwner(sr.name.c_str(), m->file.string().c_str(), "hle",
                                     "upstream-hle", return_info->virtual_address);
#endif
        return true;
    }

    const auto aeronid = AeroLib::FindByNid(sr.name.c_str());
    if (aeronid) {
        return_info->name = aeronid->name;
        return_info->virtual_address = AeroLib::GetStub(aeronid->nid);
    } else {
        return_info->virtual_address = AeroLib::GetStub(sr.name.c_str());
        return_info->name = "Unknown !!!";
    }
    LOG_WARNING(Core_Linker, "Linker: Stub resolved {} as {} (lib: {}, mod: {})", sr.name,
                return_info->name, library->name, module->name);
#ifdef __ANDROID__
    ExecutorAuditLibcFamilyOwner(sr.name.c_str(), m->file.string().c_str(), "stub", "stub",
                                 return_info->virtual_address);
    if (ExecutorVerboseLinkerTraceEnabled() &&
        (sr.name == "-pnj3-7a6QA" || sr.library == "mono-ps4" || sr.module == "mono-ps4")) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LINK_RESOLVE] source=stub importer=%s nid=%s lib=%s mod=%s addr=0x%llx "
            "stubName=%s",
            m->file.string().c_str(), sr.name.c_str(), sr.library.c_str(), sr.module.c_str(),
            static_cast<unsigned long long>(return_info->virtual_address),
            return_info->name.c_str());
    }
#endif
    return false;
}

void* Linker::TlsGetAddr(u64 module_index, u64 offset) {
    std::scoped_lock lk{mutex};

    DtvEntry* dtv_table = GetTcbBase()->tcb_dtv;
    if (dtv_table[0].counter != dtv_generation_counter) {
        const u32 old_num_dtvs = dtv_table[1].counter;
        ASSERT_MSG(max_tls_index > old_num_dtvs, "Module unloading unsupported");
        DtvEntry* new_dtv_table = new DtvEntry[max_tls_index + 2]{};
        std::memcpy(new_dtv_table + 2, dtv_table + 2, old_num_dtvs * sizeof(DtvEntry));
        new_dtv_table[0].counter = dtv_generation_counter;
        new_dtv_table[1].counter = max_tls_index;
        delete[] dtv_table;

        GetTcbBase()->tcb_dtv = new_dtv_table;
        dtv_table = new_dtv_table;
    }

    u8* addr = dtv_table[module_index + 1].pointer;
    Module* module = m_modules[module_index - 1].get();
    if (!addr) {
        const u32 init_image_size = module->tls.init_image_size;
#ifdef __ANDROID__
        const size_t tls_size = module->tls.image_size ? module->tls.image_size : 1;
        const size_t alloc_size = Common::AlignUp(tls_size, size_t{0x1000});
        u8* dest = reinterpret_cast<u8*>(mmap(nullptr, alloc_size, PROT_READ | PROT_WRITE,
                                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        ASSERT_MSG(dest != MAP_FAILED, "Unable to allocate Android dynamic PRX TLS");
        if (executor_lsx4_android_register_guest_gc_sideband_range) {
            executor_lsx4_android_register_guest_gc_sideband_range(
                dest, alloc_size, "jit_dynamic_prx_tls");
        }
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_TLS_ALLOC] module=%llu offset=0x%llx size=0x%zx "
                            "alloc=0x%zx addr=%p path=%s",
                            static_cast<unsigned long long>(module_index),
                            static_cast<unsigned long long>(offset), tls_size, alloc_size, dest,
                            module->file.string().c_str());
#else
        u8* dest = reinterpret_cast<u8*>(heap_api->heap_malloc(module->tls.image_size));
#endif
        const u8* src = reinterpret_cast<const u8*>(module->tls.image_virtual_addr);
        std::memcpy(dest, src, init_image_size);
        std::memset(dest + init_image_size, 0, module->tls.image_size - init_image_size);
        dtv_table[module_index + 1].pointer = dest;
        addr = dest;
    }
    return addr + offset;
}

void* Linker::AllocateTlsForThread(bool is_primary) {
    static constexpr size_t TcbSize = 0x40;
    static constexpr size_t TlsAllocAlign = 0x20;
#ifdef __ANDROID__
    static constexpr size_t AndroidTlsTcbPrefix = 16_KB;
#else
    static constexpr size_t AndroidTlsTcbPrefix = 0;
#endif
    const size_t tls_payload_size = Common::AlignUp(static_tls_size, TlsAllocAlign) + TcbSize;
    const size_t total_tls_size = AndroidTlsTcbPrefix + tls_payload_size;

    static constexpr VAddr KernelAllocBase = 0x881000000ULL;

    void* addr_out{reinterpret_cast<void*>(KernelAllocBase)};
    if (is_primary) {
        const size_t tls_aligned = Common::AlignUp(total_tls_size, 16_KB);
        const int ret = Libraries::Kernel::sceKernelMapNamedFlexibleMemory(
            &addr_out, tls_aligned, 3, 0, "SceKernelPrimaryTcbTls");
#ifdef __ANDROID__
        if (ret == 0) {
            addr_out = reinterpret_cast<u8*>(addr_out) + AndroidTlsTcbPrefix;
        } else {
            const size_t host_aligned = Common::AlignUp(total_tls_size, 16_KB);
            addr_out = mmap(nullptr, host_aligned, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (addr_out == MAP_FAILED) {
                addr_out = nullptr;
            }
            if (addr_out) {
                addr_out = reinterpret_cast<u8*>(addr_out) + AndroidTlsTcbPrefix;
            }
            LOG_WARNING(Core_Linker,
                        "Falling back to mmap allocation for Android primary TLS+TCB, "
                        "sceKernelMapNamedFlexibleMemory returned {} addr={}",
                        ret, fmt::ptr(addr_out));
        }
#else
        ASSERT_MSG(ret == 0, "Unable to allocate TLS+TCB for the primary thread");
#endif
    } else {
#ifdef __ANDROID__
        const size_t host_aligned = Common::AlignUp(total_tls_size, 16_KB);
        void* raw_addr = mmap(nullptr, host_aligned, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw_addr == MAP_FAILED) {
            addr_out = nullptr;
        } else {
            addr_out = reinterpret_cast<u8*>(raw_addr) + AndroidTlsTcbPrefix;
        }
#else
        if (heap_api) {
            addr_out = heap_api->heap_malloc(total_tls_size);
        } else {
            addr_out = std::malloc(total_tls_size);
        }
#endif
    }
#ifdef __ANDROID__
    if (addr_out) {
        const size_t rooted_size = Common::AlignUp(total_tls_size, 16_KB);
        void* rooted_base = reinterpret_cast<u8*>(addr_out) - AndroidTlsTcbPrefix;
        if (executor_lsx4_android_register_guest_gc_sideband_range) {
            executor_lsx4_android_register_guest_gc_sideband_range(
                rooted_base, rooted_size,
                is_primary ? "jit_main_static_tls_tcb"
                           : "jit_worker_static_tls_tcb");
        }
        LOG_INFO(Core_Linker,
                 "Android TLS+TCB allocated primary={} tls_base={} tcb={} prefix={:#x} static_tls={:#x} total={:#x}",
                 is_primary, fmt::ptr(addr_out),
                 fmt::ptr(reinterpret_cast<u8*>(addr_out) + Common::AlignUp(static_tls_size, TlsAllocAlign)),
                 AndroidTlsTcbPrefix, static_tls_size, total_tls_size);
    }
#endif
    return addr_out;
}

void Linker::FreeTlsForNonPrimaryThread(void* pointer, size_t dynamic_size) {
#ifdef __ANDROID__
    if (dynamic_size != 0) {
        const size_t alloc_size = Common::AlignUp(dynamic_size, size_t{0x1000});
        if (ExecutorGcGuestRootsMarkerEnabled()) {
            std::memset(pointer, 0, alloc_size);
        } else {
            munmap(pointer, alloc_size);
        }
        return;
    }
    static constexpr size_t TcbSize = 0x40;
    static constexpr size_t TlsAllocAlign = 0x20;
    static constexpr size_t AndroidTlsTcbPrefix = 16_KB;
    if (pointer) {
        const size_t total_tls_size =
            AndroidTlsTcbPrefix + Common::AlignUp(static_tls_size, TlsAllocAlign) + TcbSize;
        void* raw_addr = reinterpret_cast<u8*>(pointer) - AndroidTlsTcbPrefix;
        const size_t alloc_size = Common::AlignUp(total_tls_size, 16_KB);
        if (ExecutorGcGuestRootsMarkerEnabled()) {
            std::memset(raw_addr, 0, alloc_size);
        } else {
            munmap(raw_addr, alloc_size);
        }
    }
#else
    if (heap_api) {
        heap_api->heap_free(pointer);
    } else {
        std::free(pointer);
    }
#endif
}

void Linker::DebugDump() {
    const auto& log_dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
    const std::filesystem::path debug(log_dir / "debugdump");
    std::filesystem::create_directory(debug);
    for (const auto& m : m_modules) {
        Module* module = m.get();
        auto& elf = module->elf;
        const std::filesystem::path filepath(debug / module->file.stem());
        std::filesystem::create_directory(filepath);
        module->import_sym.DebugDump(filepath / "imports.txt");
        module->export_sym.DebugDump(filepath / "exports.txt");
        if (elf.IsSelfFile()) {
            elf.SelfHeaderDebugDump(filepath / "selfHeader.txt");
            elf.SelfSegHeaderDebugDump(filepath / "selfSegHeaders.txt");
        }
        elf.ElfHeaderDebugDump(filepath / "elfHeader.txt");
        elf.PHeaderDebugDump(filepath / "elfPHeaders.txt");
    }
}

}
