// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>
#include "common/string_util.h"
#ifdef __ANDROID__
#include "core/aerolib/stubs.h"
#endif
#include "core/libraries/kernel/threads.h"
#include "core/module.h"
#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace Core {

struct DynamicModuleInfo;
class Linker;
class MemoryManager;

struct OrbisKernelMemParam {
    u64 size;
    u64* extended_page_table;
    u64* flexible_memory_size;
    u8* extended_memory_1;
    u64* extended_gpu_page_table;
    u8* extended_memory_2;
    u64* extended_cpu_page_table;
};
static_assert(sizeof(OrbisKernelMemParam) == 0x38);

struct OrbisProcParam {
    u64 size;
    u32 magic;
    u32 entry_count;
    u64 sdk_version;
    char* process_name;
    char* main_thread_name;
    u32* main_thread_prio;
    u32* main_thread_stack_size;
    void* libc_param;
    OrbisKernelMemParam* mem_param;
    void* fs_param;
    u32* process_preload_enable;
    u64 unknown1;
};

using ExitFunc = PS4_SYSV_ABI void (*)();

class Linker;

struct EntryParams {
    int argc;
    u32 padding;
    const char* argv[33];
    VAddr entry_addr;
};

struct HeapAPI {
    PS4_SYSV_ABI void* (*heap_malloc)(size_t);
    PS4_SYSV_ABI void (*heap_free)(void*);
    PS4_SYSV_ABI void* (*heap_calloc)(size_t, size_t);
    PS4_SYSV_ABI void* (*heap_realloc)(void*, size_t);
    PS4_SYSV_ABI void* (*heap_memalign)(size_t, size_t);
    PS4_SYSV_ABI int (*heap_posix_memalign)(void**, size_t, size_t);
    PS4_SYSV_ABI int (*heap_reallocalign)(void);
    PS4_SYSV_ABI void (*heap_malloc_stats)(void);
    PS4_SYSV_ABI int (*heap_malloc_stats_fast)(void);
    PS4_SYSV_ABI size_t (*heap_malloc_usable_size)(void*);
};

using AppHeapAPI = HeapAPI*;

class Linker {
public:
    explicit Linker();
    ~Linker();

    Loader::SymbolsResolver& GetHLESymbols() {
        return m_hle_symbols;
    }

    OrbisProcParam* GetProcParam() const {
        return m_modules[0]->GetProcParam<OrbisProcParam*>();
    }

    Module* GetModule(s32 index) const {
        if (index >= 0 && index < m_modules.size()) {
            return m_modules.at(index).get();
        }
        return nullptr;
    }

    std::size_t GetModuleCount() const {
        return m_modules.size();
    }

    u32 FindByName(const std::filesystem::path& name) const {
        for (u32 i = 0; i < m_modules.size(); i++) {
            if (name == m_modules[i]->file) {
                return i;
            }
#ifdef __ANDROID__
            std::error_code lhs_ec;
            std::error_code rhs_ec;
            const auto lhs = std::filesystem::weakly_canonical(name, lhs_ec);
            const auto rhs = std::filesystem::weakly_canonical(m_modules[i]->file, rhs_ec);
            if (!lhs_ec && !rhs_ec && lhs == rhs) {
                return i;
            }
#endif
        }
        return -1;
    }

    u32 MaxTlsIndex() const {
        return max_tls_index;
    }

    u32 GenerationCounter() const {
        return dtv_generation_counter;
    }

    size_t StaticTlsSize() const noexcept {
        return static_tls_size;
    }

    void RelocateAnyImports(Module* m) {
        std::scoped_lock lk{mutex};

        Relocate(m);
#ifdef __ANDROID__
        const auto trace_dynamic_reloc = [] {
            static std::atomic_uint32_t emitted{0};
            return emitted.fetch_add(1, std::memory_order_relaxed) < 8;
        };
        const bool android_loaded_mono_prx = m->file.string().find("mono-ps4") != std::string::npos;
        if (android_loaded_mono_prx) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_MONO_REPATCH] stage=loaded path=%s exports=%zu exportLibs=%zu",
                m->file.string().c_str(), m->GetExportModules().size(), m->GetExportLibs().size());
        }
#endif
        const auto exports = m->GetExportModules();
#ifdef __ANDROID__
        const bool guest_libc_allocator_owner = Core::AeroLib::ExecutorUseRealLibcAlloc();
        const auto is_libc_surface_name = [](const std::string& name) {
            return name == "libc" || name == "libSceLibcInternal";
        };
#endif
        for (auto& export_mod : exports) {
#ifdef __ANDROID__
            const bool loaded_guest_libc_surface =
                guest_libc_allocator_owner && is_libc_surface_name(export_mod.name);
#endif
            for (auto& module : m_modules) {
                const auto imports = module->GetImportModules();
                if (std::ranges::contains(imports, export_mod.name, &ModuleInfo::name)
#ifdef __ANDROID__
                    || loaded_guest_libc_surface
#endif
                ) {
#ifdef __ANDROID__
                    module->ForEachRelocation([&](elf_relocation* rel, u32 i, bool is_jmp_rel) {
                        if (!module->dynamic_info.symbol_table || !module->dynamic_info.str_table) {
                            return;
                        }
                        const u32 num_relocs =
                            module->dynamic_info.relocation_table_size / sizeof(elf_relocation);
                        const u32 bit_idx = (is_jmp_rel ? num_relocs : 0) + i;
                        const auto sym_index = rel->GetSymbol();
                        const elf_symbol& sym = module->dynamic_info.symbol_table[sym_index];
                        if (sym.GetBind() != STB_GLOBAL && sym.GetBind() != STB_WEAK) {
                            return;
                        }
                        const char* raw_name = module->dynamic_info.str_table + sym.st_name;
                        if (raw_name == nullptr) {
                            return;
                        }
                        const auto ids = Common::SplitString(raw_name, '#');
                        if (ids.size() != 3) {
                            return;
                        }
                        const ModuleInfo* imported_mod = module->FindModule(ids[2]);
                        const bool guest_libc_allocator_relocation =
                            loaded_guest_libc_surface && sym.GetType() == STT_FUN &&
                            Core::AeroLib::IsLibcAllocFamilyNid(ids[0].c_str());
                        if (imported_mod == nullptr ||
                            (imported_mod->name != export_mod.name &&
                             !guest_libc_allocator_relocation)) {
                            return;
                        }
                        const bool should_yield_bootstrap_override_to_loaded_prx =
                            export_mod.name == "libSceFios2" ||
                            guest_libc_allocator_relocation;
                        u64 executor_override = 0;
                        std::string executor_override_name;
                        if (!should_yield_bootstrap_override_to_loaded_prx &&
                            Core::AeroLib::TryGetAndroidX64ExecutorOverride(
                                ids[0].c_str(), nullptr, &executor_override_name)) {
                            if (trace_dynamic_reloc()) {
                                __android_log_print(
                                    ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_DYNAMIC_RELOC_KEEP_OVERRIDE] loaded=%s importer=%s "
                                    "symbol=%s module=%s bit=%u override=0x%llx name=%s",
                                    m->file.string().c_str(), module->file.string().c_str(), raw_name,
                                    export_mod.name.c_str(), bit_idx,
                                    static_cast<unsigned long long>(executor_override),
                                    executor_override_name.c_str());
                            }
                            return;
                        }
                        if (module->TestRelaBit(bit_idx)) {
                            const VAddr rel_virtual_addr =
                                module->GetBaseAddress() + rel->rel_offset;
                            u64 current_value = 0;
                            std::memcpy(&current_value,
                                        reinterpret_cast<const void*>(rel_virtual_addr),
                                        sizeof(current_value));
                            std::string existing_hle_name;
                            if (Core::AeroLib::IsAndroidX64NativeHleStub(
                                    current_value, &existing_hle_name)) {
                                if (!should_yield_bootstrap_override_to_loaded_prx) {
                                    if (trace_dynamic_reloc()) {
                                        __android_log_print(
                                            ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_DYNAMIC_RELOC_KEEP_HLE_STUB] loaded=%s "
                                            "importer=%s symbol=%s module=%s bit=%u current=0x%llx "
                                            "name=%s",
                                            m->file.string().c_str(), module->file.string().c_str(),
                                            raw_name, export_mod.name.c_str(), bit_idx,
                                            static_cast<unsigned long long>(current_value),
                                            existing_hle_name.c_str());
                                    }
                                    return;
                                }
                                if (trace_dynamic_reloc()) {
                                    __android_log_print(
                                        ANDROID_LOG_INFO, "LSX4Native",
                                        "[EXECUTOR_DYNAMIC_RELOC_YIELD_HLE_STUB] loaded=%s "
                                        "importer=%s symbol=%s module=%s bit=%u current=0x%llx "
                                        "name=%s",
                                        m->file.string().c_str(), module->file.string().c_str(),
                                        raw_name, export_mod.name.c_str(), bit_idx,
                                        static_cast<unsigned long long>(current_value),
                                        existing_hle_name.c_str());
                                }
                            }
                            module->ClearRelaBit(bit_idx);
                            if (trace_dynamic_reloc()) {
                                __android_log_print(
                                    ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_DYNAMIC_RELOC_CLEAR] loaded=%s importer=%s symbol=%s "
                                    "module=%s bit=%u",
                                    m->file.string().c_str(), module->file.string().c_str(), raw_name,
                                    export_mod.name.c_str(), bit_idx);
                            }
                        }
                    });
#endif
                    Relocate(module.get());
                }
            }
        }
#ifdef __ANDROID__
        if (android_loaded_mono_prx) {
            for (auto& module : m_modules) {
                if (module.get() == m || !module->dynamic_info.symbol_table ||
                    !module->dynamic_info.str_table) {
                    continue;
                }
                u32 candidate_count = 0;
                u32 cleared_count = 0;
                module->ForEachRelocation([&](elf_relocation* rel, u32 i, bool is_jmp_rel) {
                    const u32 num_relocs =
                        module->dynamic_info.relocation_table_size / sizeof(elf_relocation);
                    const u32 bit_idx = (is_jmp_rel ? num_relocs : 0) + i;
                    const auto sym_index = rel->GetSymbol();
                    const elf_symbol& sym = module->dynamic_info.symbol_table[sym_index];
                    if (sym.GetBind() != STB_GLOBAL && sym.GetBind() != STB_WEAK) {
                        return;
                    }
                    const char* raw_name = module->dynamic_info.str_table + sym.st_name;
                    if (raw_name == nullptr) {
                        return;
                    }
                    const auto ids = Common::SplitString(raw_name, '#');
                    if (ids.size() != 3) {
                        return;
                    }
                    const LibraryInfo* imported_lib = module->FindLibrary(ids[1]);
                    const ModuleInfo* imported_mod = module->FindModule(ids[2]);
                    if ((imported_lib == nullptr || imported_lib->name != "mono-ps4") &&
                        (imported_mod == nullptr || imported_mod->name != "mono-ps4")) {
                        return;
                    }
                    candidate_count++;
                    const bool was_set = module->TestRelaBit(bit_idx);
                    if (was_set) {
                        module->ClearRelaBit(bit_idx);
                        cleared_count++;
                    }
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_REPATCH] stage=candidate importer=%s symbol=%s bit=%u "
                        "wasSet=%u",
                        module->file.string().c_str(), raw_name, bit_idx,
                        static_cast<unsigned>(was_set));
                });
                if (candidate_count != 0) {
                    __android_log_print(
                        ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MONO_REPATCH] stage=relocate importer=%s candidates=%u "
                        "cleared=%u",
                        module->file.string().c_str(), candidate_count, cleared_count);
                    Relocate(module.get());
                }
            }
        }
#endif
    }

    void RelocateAllImports() {
        std::scoped_lock lk{mutex};
        for (auto& module : m_modules) {
            Relocate(module.get());
        }
    }

    void SetHeapAPI(void* func[]) {
        heap_api = reinterpret_cast<AppHeapAPI>(func);
    }

    void AdvanceGenerationCounter() noexcept {
        dtv_generation_counter++;
    }

    void* TlsGetAddr(u64 module_index, u64 offset);
    void* AllocateTlsForThread(bool is_primary);
    void FreeTlsForNonPrimaryThread(void* pointer, size_t dynamic_size = 0);

    s32 LoadModule(const std::filesystem::path& elf_name, bool is_dynamic = false);
    s32 LoadAndStartModule(const std::filesystem::path& path, u64 args, const void* argp,
                           int* pRes);
    Module* FindByAddress(VAddr address);

    void Relocate(Module* module);
    void PrepareStaticTlsForMainModule();
    void PrepareStaticTlsForModule(Module* module);
    bool Resolve(const std::string& name, Loader::SymbolType type, Module* module,
                 Loader::SymbolRecord* return_info);
    void Execute(const std::vector<std::string>& args = {});
    void DebugDump();

private:
    const Module* FindExportedModule(const ModuleInfo& m, const LibraryInfo& l);

    MemoryManager* memory;
    Libraries::Kernel::Thread main_thread;
    std::mutex mutex;
    u32 dtv_generation_counter{1};
    size_t static_tls_size{};
    u32 max_tls_index{};
    u32 num_static_modules{};
    AppHeapAPI heap_api{};
    std::vector<std::unique_ptr<Module>> m_modules;
    Loader::SymbolsResolver m_hle_symbols{};
};

}
