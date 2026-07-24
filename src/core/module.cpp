// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/alignment.h"
#include "common/arch.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/memory_patcher.h"
#include "common/sha1.h"
#include "common/string_util.h"
#include "core/aerolib/aerolib.h"
#include "core/aerolib/stubs.h"
#include "core/cpu_patches.h"
#include "core/loader/dwarf.h"
#include "core/memory.h"
#include "core/module.h"
#include "core/tls.h"
#if defined(__ANDROID__) || defined(__linux__)
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#endif
#ifdef __ANDROID__
#include <android/log.h>
#include <dlfcn.h>

extern "C" int executor_lsx4_android_run_guest_module_start(void* module_ptr,
                                                               std::uint64_t args,
                                                               const void* argp,
                                                               void* param,
                                                               int* out_result);
extern "C" int executor_lsx4_install_imt_dynamic_fix(void* seg_host, std::size_t seg_size);
extern "C" int executor_lsx4_install_vtnull_probe(void* seg_host, std::size_t seg_size);
extern "C" int executor_lsx4_install_gc_disable_collect(void* seg_host, std::size_t seg_size);
extern "C" int executor_lsx4_install_mono_value_copy_probe(void* target_addr);
#endif

#ifdef __ANDROID__
namespace Core::AeroLib {
void ExecutorPatchLibcInternalAllocators(u64 exec_seg_addr, u64 exec_seg_size);
}
#endif

namespace Core {

using EntryFunc = PS4_SYSV_ABI int (*)(size_t args, const void* argp, void* param);

static constexpr u64 ModuleLoadBase = 0x800000000;

static u64 GetAlignedSize(const elf_program_header& phdr) {
    return (phdr.p_align != 0 ? (phdr.p_memsz + (phdr.p_align - 1)) & ~(phdr.p_align - 1)
                              : phdr.p_memsz);
}

static std::string SegmentProtString(const u32 flags) {
    std::string out;
    out += (flags & PF_READ) ? 'r' : '-';
    out += (flags & PF_WRITE) ? 'w' : '-';
    out += (flags & PF_EXEC) ? 'x' : '-';
    return out;
}

static void ExecutorModuleTrace(const char* stage, const std::filesystem::path& path,
                                const std::string& detail = {}) {
#ifdef __ANDROID__
    const char* trace = std::getenv("EXECUTOR_VERBOSE_MODULE_STAGE");
    if (trace == nullptr || trace[0] == '\0' || std::strcmp(trace, "0") == 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MODULE_STAGE] stage=%s path=%s detail=%s", stage,
                        path.string().c_str(), detail.empty() ? "-" : detail.c_str());
#else
    (void)stage;
    (void)path;
    (void)detail;
#endif
}

#ifdef __ANDROID__
using ExecutorFexRegisterGuestExecutableRangeFn = int (*)(std::uint64_t address,
                                                          std::uint64_t size,
                                                          std::uint32_t prot);

static ExecutorFexRegisterGuestExecutableRangeFn ExecutorResolveFexExecRangeRegistrar() {
    void* sym = dlsym(RTLD_DEFAULT, "executor_fexcore_register_guest_executable_range");
    if (sym == nullptr) {
        void* handle = dlopen("libexecutor_fexcore_embed.so", RTLD_NOW | RTLD_NOLOAD);
        if (handle != nullptr) {
            sym = dlsym(handle, "executor_fexcore_register_guest_executable_range");
        }
    }
    return reinterpret_cast<ExecutorFexRegisterGuestExecutableRangeFn>(sym);
}

static void ExecutorRegisterFexGuestExecutableRange(std::uint64_t address, std::uint64_t size,
                                                    std::uint32_t prot,
                                                    const std::filesystem::path& path,
                                                    std::uint16_t index) {
    if ((prot & PF_EXEC) == 0 || size == 0) {
        return;
    }
    static ExecutorFexRegisterGuestExecutableRangeFn registrar =
        ExecutorResolveFexExecRangeRegistrar();
    if (registrar == nullptr) {
        return;
    }
    const int rc = registrar(address, size, prot);
    ExecutorModuleTrace("fex_exec_range_register", path,
                        std::string("index=") + std::to_string(index) +
                            ",addr=" + std::to_string(address) +
                            ",size=" + std::to_string(size) +
                            ",prot=" + SegmentProtString(prot) +
                            ",rc=" + std::to_string(rc));
}
#endif

static u64 CalculateBaseSize(const elf_header& ehdr, std::span<const elf_program_header> phdr) {
    u64 base_size = 0;
    for (u16 i = 0; i < ehdr.e_phnum; i++) {
        if (phdr[i].p_memsz != 0 && (phdr[i].p_type == PT_LOAD || phdr[i].p_type == PT_SCE_RELRO)) {
            const u64 last_addr = phdr[i].p_vaddr + GetAlignedSize(phdr[i]);
            base_size = std::max(last_addr, base_size);
        }
    }
    return base_size;
}

static std::string EncodeId(u64 nVal) {
    std::string enc;
    static constexpr std::string_view codes =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    if (nVal < 0x40u) {
        enc += codes[nVal];
    } else {
        if (nVal < 0x1000u) {
            enc += codes[static_cast<u16>(nVal >> 6u) & 0x3fu];
            enc += codes[nVal & 0x3fu];
        } else {
            enc += codes[static_cast<u16>(nVal >> 12u) & 0x3fu];
            enc += codes[static_cast<u16>(nVal >> 6u) & 0x3fu];
            enc += codes[nVal & 0x3fu];
        }
    }
    return enc;
}

static std::string StringToNid(std::string_view symbol) {
    static constexpr std::array<u8, 16> Salt = {0x51, 0x8D, 0x64, 0xA6, 0x35, 0xDE, 0xD8, 0xC1,
                                                0xE6, 0xB0, 0x39, 0xB1, 0xC3, 0xE5, 0x52, 0x30};
    std::vector<u8> input(symbol.size() + Salt.size());
    std::memcpy(input.data(), symbol.data(), symbol.size());
    std::memcpy(input.data() + symbol.size(), Salt.data(), Salt.size());

    sha1::SHA1::digest8_t hash;
    sha1::SHA1 sha;
    sha.processBytes(input.data(), input.size());
    sha.getDigestBytes(hash);

    u64 digest;
    std::memcpy(&digest, hash, sizeof(digest));

    static constexpr std::string_view codes =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
    std::string dst(11, '\0');

    for (int i = 0; i < 10; i++) {
        dst[i] = codes[(digest >> (58 - i * 6)) & 0x3f];
    }
    dst[10] = codes[(digest & 0xf) * 4];
    return dst;
}

Module::Module(Core::MemoryManager* memory_, const std::filesystem::path& file_, u32& max_tls_index)
    : memory{memory_}, file{file_}, name{file.filename().string()} {
    ExecutorModuleTrace("ctor_enter", file);
    ExecutorModuleTrace("elf_open_begin", file);
    elf.Open(file);
    ExecutorModuleTrace("elf_open_done", file,
                        std::string("is_elf=") + (elf.IsElfFile() ? "1" : "0"));
    if (elf.IsElfFile()) {
        ExecutorModuleTrace("load_memory_begin", file);
        LoadModuleToMemory(max_tls_index);
        ExecutorModuleTrace("load_memory_done", file,
                            std::string("base=") + std::to_string(base_virtual_addr));
        ExecutorModuleTrace("dynamic_info_begin", file);
        LoadDynamicInfo();
        ExecutorModuleTrace("dynamic_info_done", file,
                            std::string("needed=") +
                                std::to_string(dynamic_info.needed.size()) + ",imports=" +
                                std::to_string(dynamic_info.import_modules.size()) + ",libs=" +
                                std::to_string(dynamic_info.import_libs.size()));
        ExecutorModuleTrace("symbols_begin", file);
        LoadSymbols();
        ExecutorModuleTrace("symbols_done", file);
#ifdef __ANDROID__
        if (file.filename().string() == "mono-ps4.sprx") {
            static const char* const kMonoHookTargets[] = {
                "mono_value_copy",       "mono_value_copy_array", "mono_array_new",
                "mono_object_new_ptrfree", "mono_string_new_wrapper", "mono_raise_exception",
                "mono_gc_bzero",         "mono_field_set_value"};
            if (std::FILE* lf = std::fopen(
                    "/data/data/app.lsx4.android/files/executor-mono-exports.log", "a")) {
                for (const char* fn : kMonoHookTargets) {
                    void* a = FindByName(fn);
                    std::fprintf(lf, "MONO_EXPORT %-26s = %p\n", fn, a);
                }
                std::fclose(lf);
            }
            if (std::fopen("/data/data/app.lsx4.android/files/run-enable-legacy-patches", "r") &&
                FindByName("mono_value_copy")) {
                void* mvc = FindByName("mono_value_copy");
                const int rc = executor_lsx4_install_mono_value_copy_probe(mvc);
                if (std::FILE* lf = std::fopen(
                        "/data/data/app.lsx4.android/files/executor-mono-exports.log", "a")) {
                    std::fprintf(lf, "MVC_INSTALL target=%p rc=%d\n", mvc, rc);
                    std::fclose(lf);
                }
            }
        }
#endif
    }
    ExecutorModuleTrace("ctor_done", file,
                        std::string("valid=") + (IsValid() ? "1" : "0"));
}

static void RestoreHostSegmentProtection(const VAddr segment_addr, const u64 segment_size,
                                         const u32 flags, const std::filesystem::path& path,
                                         const u16 index) {
#if defined(__ANDROID__) || defined(__linux__)
    if ((flags & PF_EXEC) == 0 || segment_size == 0) {
        return;
    }

    constexpr u64 PageSize = 0x1000;
    const VAddr page_base = Common::AlignDown(segment_addr, PageSize);
    const VAddr page_end = Common::AlignUp(segment_addr + segment_size, PageSize);
    const u64 page_size = page_end - page_base;

    int host_prot = PROT_NONE;
    if (flags & PF_READ) {
        host_prot |= PROT_READ;
    }
    if (flags & PF_WRITE) {
        host_prot |= PROT_WRITE;
    }
    if (flags & PF_EXEC) {
        host_prot |= PROT_EXEC;
    }

    const int rc = ::mprotect(reinterpret_cast<void*>(page_base), page_size, host_prot);
    ExecutorModuleTrace(
        rc == 0 ? "host_exec_protect_done" : "host_exec_protect_failed", path,
        std::string("index=") + std::to_string(index) + ",addr=" + std::to_string(page_base) +
            ",size=" + std::to_string(page_size) + ",prot=" + SegmentProtString(flags) +
            ",errno=" + std::to_string(rc == 0 ? 0 : errno) +
            (rc == 0 ? std::string{} : std::string(",error=") + std::strerror(errno)));
#else
    (void)segment_addr;
    (void)segment_size;
    (void)flags;
    (void)path;
    (void)index;
#endif
}

Module::~Module() = default;

s32 Module::Start(u64 args, const void* argp, void* param) {
    LOG_INFO(Core_Linker, "Module started : {}", name);
    lifecycle_state.store(LifecycleState::Starting, std::memory_order_release);
    const VAddr addr = dynamic_info.init_virtual_addr + GetBaseAddress();
#ifdef __ANDROID__
    int result = 0;
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MODULE_START_BRIDGE] path=%s function=0x%llx args=0x%llx",
                        file.string().c_str(), static_cast<unsigned long long>(addr),
                        static_cast<unsigned long long>(args));
    const int rc = executor_lsx4_android_run_guest_module_start(this, args, argp, param, &result);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_MODULE_START_BRIDGE] result path=%s rc=%d guestResult=0x%x",
                        file.string().c_str(), rc, static_cast<unsigned>(result));
    lifecycle_state.store(rc >= 0 ? LifecycleState::Started : LifecycleState::Loaded,
                          std::memory_order_release);
    return rc >= 0 ? result : rc;
#else
    const s32 result = reinterpret_cast<EntryFunc>(addr)(args, argp, param);
    lifecycle_state.store(LifecycleState::Started, std::memory_order_release);
    return result;
#endif
}

void Module::LoadModuleToMemory(u32& max_tls_index) {
    static constexpr size_t BlockAlign = 0x1000;
    static constexpr u64 TrampolineSize = 8_MB;
    ExecutorModuleTrace("load_memory_enter", file);

    const auto elf_header = elf.GetElfHeader();
    const auto elf_pheader = elf.GetProgramHeader();
    const u64 base_size = CalculateBaseSize(elf_header, elf_pheader);
    aligned_base_size = Common::AlignUp(base_size, BlockAlign);
    ExecutorModuleTrace("headers_ready", file,
                        std::string("phnum=") + std::to_string(elf_header.e_phnum) +
                            ",base_size=" + std::to_string(base_size) +
                            ",aligned=" + std::to_string(aligned_base_size));

    void** out_addr = reinterpret_cast<void**>(&base_virtual_addr);
    ExecutorModuleTrace("map_memory_begin", file,
                        std::string("size=") +
                            std::to_string(aligned_base_size + TrampolineSize));
    memory->MapMemory(out_addr, ModuleLoadBase, aligned_base_size + TrampolineSize,
                      MemoryProt::CpuReadWrite | MemoryProt::CpuExec, MemoryMapFlags::NoFlags,
                      VMAType::Code, name);
    ExecutorModuleTrace("map_memory_done", file,
                        std::string("base=") + std::to_string(base_virtual_addr));
    LOG_INFO(Core_Linker, "Loading module {} to {}", name, fmt::ptr(*out_addr));

#ifdef ARCH_X86_64
    void* trampoline_addr = std::bit_cast<void*>(base_virtual_addr + aligned_base_size);
    RegisterPatchModule(*out_addr, aligned_base_size, trampoline_addr, TrampolineSize);
#endif

    LOG_INFO(Core_Linker, "======== Load Module to Memory ========");
    LOG_INFO(Core_Linker, "base_virtual_addr ......: {:#018x}", base_virtual_addr);
    LOG_INFO(Core_Linker, "base_size ..............: {:#018x}", base_size);
    LOG_INFO(Core_Linker, "aligned_base_size ......: {:#018x}", aligned_base_size);

    const auto add_segment = [this](const elf_program_header& phdr, bool do_map = true) {
        const VAddr segment_addr = base_virtual_addr + phdr.p_vaddr;
        if (do_map) {
            elf.LoadSegment(segment_addr, phdr.p_offset, phdr.p_filesz);
        }
        if (info.num_segments < 4) {
            auto& segment = info.segments[info.num_segments++];
            segment.address = segment_addr;
            segment.prot = phdr.p_flags;
            segment.size = GetAlignedSize(phdr);
        } else {
            LOG_ERROR(Core_Linker, "Attempting to add too many segments!");
        }
    };

    for (u16 i = 0; i < elf_header.e_phnum; i++) {
        const auto header_type = elf.ElfPheaderTypeStr(elf_pheader[i].p_type);
        ExecutorModuleTrace("program_header_begin", file,
                            std::string("index=") + std::to_string(i) + ",type=" +
                                std::to_string(elf_pheader[i].p_type) + ",filesz=" +
                                std::to_string(elf_pheader[i].p_filesz) + ",memsz=" +
                                std::to_string(elf_pheader[i].p_memsz));
        switch (elf_pheader[i].p_type) {
        case PT_LOAD:
        case PT_SCE_RELRO: {
            if (elf_pheader[i].p_memsz == 0) {
                LOG_ERROR(Core_Linker, "p_memsz==0 in type {}", header_type);
                continue;
            }

            const u64 segment_addr = elf_pheader[i].p_vaddr + base_virtual_addr;
            const u64 segment_file_size = elf_pheader[i].p_filesz;
            const u64 segment_memory_size = GetAlignedSize(elf_pheader[i]);
            const auto segment_mode = elf.ElfPheaderFlagsStr(elf_pheader[i].p_flags);
            LOG_INFO(Core_Linker, "program header = [{}] type = {}", i, header_type);
            LOG_INFO(Core_Linker, "segment_addr ..........: {:#018x}", segment_addr);
            LOG_INFO(Core_Linker, "segment_file_size .....: {}", segment_file_size);
            LOG_INFO(Core_Linker, "segment_memory_size ...: {}", segment_memory_size);
            LOG_INFO(Core_Linker, "segment_mode ..........: {}", segment_mode);

            add_segment(elf_pheader[i]);
            ExecutorModuleTrace("load_segment_done", file,
                                std::string("index=") + std::to_string(i));
#if defined(ARCH_X86_64) || defined(__ANDROID__)
            if (elf_pheader[i].p_flags & PF_EXEC) {
                if (file.filename().string() == "libc.prx") {
                    const bool use_guest_libc_allocator =
                        Core::AeroLib::ExecutorUseRealLibcAlloc();
                    if (use_guest_libc_allocator) {
                        std::fprintf(stderr,
                                     "[EXECUTOR_LIBC_CANARY_PATCH] skipped owner=guest-libc "
                                     "seg=0x%llx size=0x%llx\n",
                                     (unsigned long long)segment_addr,
                                     (unsigned long long)segment_file_size);
                        std::fflush(stderr);
#ifdef __ANDROID__
                        __android_log_print(
                            ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_OWNER] owner=guest-libc action=preserve-module-bytes "
                            "seg=0x%llx size=0x%llx",
                            (unsigned long long)segment_addr,
                            (unsigned long long)segment_file_size);
#endif
                    } else {
                    static constexpr std::array<u8, 19> kCanarySig = {
                        0xbe, 0xf8, 0xff, 0xff, 0xff, 0x49, 0x23, 0x77, 0xf8, 0x4a,
                        0x33, 0x54, 0x3e, 0xf0, 0x4c, 0x39, 0xc2, 0x0f, 0x85};
                    auto* bytes = reinterpret_cast<u8*>(segment_addr);
                    bool patched = false;
                    for (u64 off = 0; off + kCanarySig.size() <= segment_file_size; ++off) {
                        if (std::memcmp(bytes + off, kCanarySig.data(), kCanarySig.size()) == 0) {
                            std::memset(bytes + off + 17, 0x90, 6);
                            std::fprintf(stderr,
                                         "[EXECUTOR_LIBC_CANARY_PATCH] applied at va=0x%llx off=0x%llx\n",
                                         (unsigned long long)(segment_addr + off + 17),
                                         (unsigned long long)off);
                            std::fflush(stderr);
                            patched = true;
                            break;
                        }
                    }
                    std::fprintf(stderr,
                                 "[EXECUTOR_LIBC_CANARY_PATCH] libc.prx exec seg scanned: addr=0x%llx size=0x%llx patched=%d\n",
                                 (unsigned long long)segment_addr,
                                 (unsigned long long)segment_file_size, patched ? 1 : 0);
                    std::fflush(stderr);
                    if (std::FILE* lf = std::fopen(
                            "/data/data/app.lsx4.android/files/executor-libc-patch.log",
                            "a")) {
                        std::fprintf(lf,
                                     "CANARY seg=0x%llx size=0x%llx canary_patched=%d\n",
                                     (unsigned long long)segment_addr,
                                     (unsigned long long)segment_file_size, patched ? 1 : 0);
                        std::fclose(lf);
                    }
#ifdef __ANDROID__
                    Core::AeroLib::ExecutorPatchLibcInternalAllocators(segment_addr,
                                                                       segment_file_size);
#endif
                    }
                }
#ifdef __ANDROID__
                std::FILE* aot_flag =
                    file.filename().string() == "mono-ps4.sprx"
                        ? std::fopen(
                              "/data/data/app.lsx4.android/files/executor-force-aot-deps-ok",
                              "r")
                        : nullptr;
                if (aot_flag != nullptr) {
                    std::fclose(aot_flag);
                    static constexpr std::array<u8, 7> kAotDepSig = {0x41, 0x83, 0x7c, 0x24,
                                                                     0x54, 0x00, 0x74};
                    auto* bytes = reinterpret_cast<u8*>(segment_addr);
                    int patched = 0;
                    for (u64 off = 0; off + kAotDepSig.size() <= segment_file_size; ++off) {
                        if (std::memcmp(bytes + off, kAotDepSig.data(), kAotDepSig.size()) == 0) {
                            bytes[off + 6] = 0xeb;
                            ++patched;
                        }
                    }
                    if (std::FILE* lf = std::fopen(
                            "/data/data/app.lsx4.android/files/executor-libc-patch.log", "a")) {
                        std::fprintf(lf, "AOTDEP seg=0x%llx size=0x%llx je2jmp_patched=%d\n",
                                     (unsigned long long)segment_addr,
                                     (unsigned long long)segment_file_size, patched);
                        std::fclose(lf);
                    }
                }

                if (file.filename().string() == "mono-ps4.sprx") {
                    const bool disable_exc_init_jit = std::filesystem::exists(
                        "/data/data/app.lsx4.android/files/lsx4-home/run-no-exc-init-jit");
                    if (!disable_exc_init_jit) {
                        static constexpr std::array<u8, 19> kExcInitSig = {
                            0x48, 0x8d, 0x05, 0x2e, 0x96, 0x12, 0x00, 0x83, 0x38, 0x00,
                            0x74, 0x47, 0x48, 0x8d, 0x3d, 0xe0, 0xd2, 0x0d, 0x00};
                        auto* bytes = reinterpret_cast<u8*>(segment_addr);
                        int patched = 0;
                        for (u64 off = 0; off + kExcInitSig.size() <= segment_file_size; ++off) {
                            if (std::memcmp(bytes + off, kExcInitSig.data(), kExcInitSig.size()) ==
                                0) {
                                bytes[off + 10] = 0xeb;
                                ++patched;
                            }
                        }
                        static constexpr std::array<u8, 19> kCodemanSig = {
                            0x48, 0x8d, 0x05, 0xd1, 0x29, 0x12, 0x00, 0x83, 0x38, 0x00,
                            0x74, 0x15, 0x48, 0x8d, 0x15, 0x0f, 0x71, 0x0d, 0x00};
                        int codeman_patched = 0;
                        for (u64 off = 0; off + kCodemanSig.size() <= segment_file_size; ++off) {
                            if (std::memcmp(bytes + off, kCodemanSig.data(),
                                            kCodemanSig.size()) == 0) {
                                bytes[off + 10] = 0xeb;
                                ++codeman_patched;
                            }
                        }
                        if (std::FILE* lf = std::fopen(
                                "/data/data/app.lsx4.android/files/executor-libc-patch.log",
                                "a")) {
                            std::fprintf(lf,
                                         "EXC_INIT_JIT seg=0x%llx size=0x%llx je2jmp_patched=%d "
                                         "codeman_patched=%d\n",
                                         (unsigned long long)segment_addr,
                                         (unsigned long long)segment_file_size, patched,
                                         codeman_patched);
                            std::fclose(lf);
                        }
                    }
                    const bool disable_imt_fix = std::filesystem::exists(
                        "/data/data/app.lsx4.android/files/lsx4-home/run-disable-imt-fix");
                    const int rc = disable_imt_fix
                        ? 99
                        : executor_lsx4_install_imt_dynamic_fix(
                              reinterpret_cast<void*>(segment_addr),
                              static_cast<std::size_t>(segment_file_size));
                    if (std::FILE* lf = std::fopen(
                            "/data/data/app.lsx4.android/files/executor-libc-patch.log", "a")) {
                        std::fprintf(lf, "IMT_DYNAMIC_FIX seg=0x%llx size=0x%llx rc=%d disabled=%d\n",
                                     (unsigned long long)segment_addr,
                                     (unsigned long long)segment_file_size, rc, disable_imt_fix ? 1 : 0);
                        std::fclose(lf);
                    }
                    if (std::fopen("/data/data/app.lsx4.android/files/run-enable-legacy-patches", "r")) {
                        const int vrc = executor_lsx4_install_vtnull_probe(
                            reinterpret_cast<void*>(segment_addr),
                            static_cast<std::size_t>(segment_file_size));
                        (void)vrc;
                        const int grc = executor_lsx4_install_gc_disable_collect(
                            reinterpret_cast<void*>(segment_addr),
                            static_cast<std::size_t>(segment_file_size));
                        if (std::FILE* lf = std::fopen(
                                "/data/data/app.lsx4.android/files/executor-libc-patch.log", "a")) {
                            std::fprintf(lf, "GC_DISABLE seg=0x%llx size=0x%llx rc=%d\n",
                                         (unsigned long long)segment_addr,
                                         (unsigned long long)segment_file_size, grc);
                            std::fclose(lf);
                        }
                    }
                }
#endif
#ifdef ARCH_X86_64
                PrePatchInstructions(segment_addr, segment_file_size);
#endif
            }
#endif
            RestoreHostSegmentProtection(segment_addr, segment_memory_size, elf_pheader[i].p_flags,
                                         file, i);
#ifdef __ANDROID__
            ExecutorRegisterFexGuestExecutableRange(segment_addr, segment_memory_size,
                                                    elf_pheader[i].p_flags, file, i);
#endif
            break;
        }
        case PT_DYNAMIC:
            add_segment(elf_pheader[i], false);
            if (elf_pheader[i].p_filesz != 0) {
                m_dynamic.resize(elf_pheader[i].p_filesz);
                const VAddr segment_addr = std::bit_cast<VAddr>(m_dynamic.data());
                elf.LoadSegment(segment_addr, elf_pheader[i].p_offset, elf_pheader[i].p_filesz);
                ExecutorModuleTrace("dynamic_segment_loaded", file,
                                    std::string("size=") +
                                        std::to_string(elf_pheader[i].p_filesz));
            } else {
                LOG_ERROR(Core_Linker, "p_filesz==0 in type {}", header_type);
            }
            break;
        case PT_SCE_DYNLIBDATA:
            if (elf_pheader[i].p_filesz != 0) {
                m_dynamic_data.resize(elf_pheader[i].p_filesz);
                const VAddr segment_addr = std::bit_cast<VAddr>(m_dynamic_data.data());
                elf.LoadSegment(segment_addr, elf_pheader[i].p_offset, elf_pheader[i].p_filesz);
                ExecutorModuleTrace("dynlibdata_loaded", file,
                                    std::string("size=") +
                                        std::to_string(elf_pheader[i].p_filesz));
            } else {
                LOG_ERROR(Core_Linker, "p_filesz==0 in type {}", header_type);
            }
            break;
        case PT_TLS:
            tls.init_image_size = elf_pheader[i].p_filesz;
            tls.align = elf_pheader[i].p_align;
            tls.image_virtual_addr = elf_pheader[i].p_vaddr + base_virtual_addr;
            tls.image_size = GetAlignedSize(elf_pheader[i]);
            tls.modid = ++max_tls_index;
            LOG_INFO(Core_Linker, "TLS virtual address = {:#x}", tls.image_virtual_addr);
            LOG_INFO(Core_Linker, "TLS image size      = {}", tls.image_size);
            ExecutorModuleTrace("tls_seen", file,
                                std::string("image_size=") + std::to_string(tls.image_size) +
                                    ",modid=" + std::to_string(tls.modid));
            break;
        case PT_SCE_PROCPARAM:
            proc_param_virtual_addr = elf_pheader[i].p_vaddr + base_virtual_addr;
            ExecutorModuleTrace("proc_param_seen", file,
                                std::string("addr=") +
                                    std::to_string(proc_param_virtual_addr));
            break;
        case PT_GNU_EH_FRAME: {
            eh_frame_hdr_addr = elf_pheader[i].p_vaddr;
            eh_frame_hdr_size = elf_pheader[i].p_memsz;
            const VAddr eh_hdr_start = base_virtual_addr + eh_frame_hdr_addr;
            const VAddr eh_hdr_end = eh_hdr_start + eh_frame_hdr_size;
            Dwarf::EHHeaderInfo hdr_info;
            if (Dwarf::DecodeEHHdr(eh_hdr_start, eh_hdr_end, hdr_info)) {
                eh_frame_addr = hdr_info.eh_frame_ptr - base_virtual_addr;
                if (eh_frame_hdr_addr > eh_frame_addr) {
                    eh_frame_size = (eh_frame_hdr_addr - eh_frame_addr);
                } else {
                    eh_frame_size = (aligned_base_size - eh_frame_hdr_addr);
                }
            }
            break;
        }
        default:
            LOG_ERROR(Core_Linker, "Unimplemented type {}", header_type);
        }
    }

    const VAddr entry_addr = base_virtual_addr + elf.GetElfEntry();
    LOG_INFO(Core_Linker, "program entry addr ..........: {:#018x}", entry_addr);
    ExecutorModuleTrace("segments_done", file,
                        std::string("entry=") + std::to_string(entry_addr) + ",segments=" +
                            std::to_string(info.num_segments));

    if (MemoryPatcher::g_eboot_address == 0) {
        if (name == "eboot.bin") {
            ExecutorModuleTrace("memory_patcher_begin", file);
            MemoryPatcher::g_eboot_address = base_virtual_addr;
            MemoryPatcher::g_eboot_image_size = base_size;
            MemoryPatcher::OnGameLoaded();
            ExecutorModuleTrace("memory_patcher_done", file);
        }
    }
}

void Module::LoadDynamicInfo() {
    ExecutorModuleTrace("dynamic_parse_enter", file,
                        std::string("dynamic=") + std::to_string(m_dynamic.size()) +
                            ",data=" + std::to_string(m_dynamic_data.size()));
    const auto* dynamic_begin = reinterpret_cast<elf_dynamic*>(m_dynamic.data());
    const size_t dynamic_count = m_dynamic.size() / sizeof(elf_dynamic);
    for (size_t dynamic_index = 0; dynamic_index < dynamic_count; dynamic_index++) {
        const auto* dyn = dynamic_begin + dynamic_index;
        if (dyn->d_tag == DT_NULL) {
            break;
        }
        switch (dyn->d_tag) {
        case DT_SCE_HASH:
            dynamic_info.hash_table =
                reinterpret_cast<void*>(m_dynamic_data.data() + dyn->d_un.d_ptr);
            break;
        case DT_SCE_HASHSZ:
            dynamic_info.hash_table_size = dyn->d_un.d_val;
            break;
        case DT_SCE_STRTAB:
            dynamic_info.str_table =
                reinterpret_cast<char*>(m_dynamic_data.data() + dyn->d_un.d_ptr);
            break;
        case DT_SCE_STRSZ:
            dynamic_info.str_table_size = dyn->d_un.d_val;
            break;
        case DT_SCE_SYMTAB:
            dynamic_info.symbol_table =
                reinterpret_cast<elf_symbol*>(m_dynamic_data.data() + dyn->d_un.d_ptr);
            break;
        case DT_SCE_SYMTABSZ:
            dynamic_info.symbol_table_total_size = dyn->d_un.d_val;
            break;
        case DT_SCE_SYMENT:
            dynamic_info.symbol_table_entries_size = dyn->d_un.d_val;
            break;
        default:
            break;
        }
    }
    ExecutorModuleTrace("dynamic_prescan_done", file,
                        std::string("strtab=") + (dynamic_info.str_table ? "1" : "0") +
                            ",symtab=" + (dynamic_info.symbol_table ? "1" : "0") +
                            ",entries=" + std::to_string(dynamic_count));
    bool saw_dynamic_null = false;
    for (size_t dynamic_index = 0; dynamic_index < dynamic_count; dynamic_index++) {
        const auto* dyn = dynamic_begin + dynamic_index;
        ExecutorModuleTrace("dynamic_tag", file,
                            std::string("index=") + std::to_string(dynamic_index) + ",tag=" +
                                std::to_string(dyn->d_tag) + ",value=" +
                                std::to_string(dyn->d_un.d_val));
        if (dyn->d_tag == DT_NULL) {
            saw_dynamic_null = true;
            ExecutorModuleTrace("dynamic_null", file,
                                std::string("index=") + std::to_string(dynamic_index));
            break;
        }
        switch (dyn->d_tag) {
        case DT_SCE_HASH:
            dynamic_info.hash_table =
                reinterpret_cast<void*>(m_dynamic_data.data() + dyn->d_un.d_ptr);
            break;
        case DT_SCE_HASHSZ:
            dynamic_info.hash_table_size = dyn->d_un.d_val;
            break;
        case DT_SCE_STRTAB:
            dynamic_info.str_table =
                reinterpret_cast<char*>(m_dynamic_data.data() + dyn->d_un.d_ptr);
            break;
        case DT_SCE_STRSZ:
            dynamic_info.str_table_size = dyn->d_un.d_val;
            break;
        case DT_SCE_SYMTAB:
            dynamic_info.symbol_table =
                reinterpret_cast<elf_symbol*>(m_dynamic_data.data() + dyn->d_un.d_ptr);
            break;
        case DT_SCE_SYMTABSZ:
            dynamic_info.symbol_table_total_size = dyn->d_un.d_val;
            break;
        case DT_INIT:
            dynamic_info.init_virtual_addr = dyn->d_un.d_ptr;
            break;
        case DT_FINI:
            dynamic_info.fini_virtual_addr = dyn->d_un.d_ptr;
            break;
        case DT_SCE_PLTGOT:
            dynamic_info.pltgot_virtual_addr = dyn->d_un.d_ptr;
            break;
        case DT_SCE_JMPREL:
            dynamic_info.jmp_relocation_table =
                reinterpret_cast<elf_relocation*>(m_dynamic_data.data() + dyn->d_un.d_ptr);
            break;
        case DT_SCE_PLTRELSZ:
            dynamic_info.jmp_relocation_table_size = dyn->d_un.d_val;
            break;
        case DT_SCE_PLTREL:
            dynamic_info.jmp_relocation_type = dyn->d_un.d_val;
            if (dynamic_info.jmp_relocation_type != DT_RELA) {
                LOG_WARNING(Core_Linker, "DT_SCE_PLTREL is NOT DT_RELA should check!");
            }
            break;
        case DT_SCE_RELA:
            dynamic_info.relocation_table =
                reinterpret_cast<elf_relocation*>(m_dynamic_data.data() + dyn->d_un.d_ptr);
            break;
        case DT_SCE_RELASZ:
            dynamic_info.relocation_table_size = dyn->d_un.d_val;
            break;
        case DT_SCE_RELAENT:
            dynamic_info.relocation_table_entries_size = dyn->d_un.d_val;
            if (dynamic_info.relocation_table_entries_size != 0x18) {
                LOG_WARNING(Core_Linker, "DT_SCE_RELAENT is NOT 0x18 should check!");
            }
            break;
        case DT_INIT_ARRAY:
            dynamic_info.init_array_virtual_addr = dyn->d_un.d_ptr;
            break;
        case DT_FINI_ARRAY:
            dynamic_info.fini_array_virtual_addr = dyn->d_un.d_ptr;
            break;
        case DT_INIT_ARRAYSZ:
            dynamic_info.init_array_size = dyn->d_un.d_val;
            break;
        case DT_FINI_ARRAYSZ:
            dynamic_info.fini_array_size = dyn->d_un.d_val;
            break;
        case DT_PREINIT_ARRAY:
            dynamic_info.preinit_array_virtual_addr = dyn->d_un.d_ptr;
            break;
        case DT_PREINIT_ARRAYSZ:
            dynamic_info.preinit_array_size = dyn->d_un.d_val;
            break;
        case DT_SCE_SYMENT:
            dynamic_info.symbol_table_entries_size = dyn->d_un.d_val;
            if (dynamic_info.symbol_table_entries_size != 0x18) {
                LOG_WARNING(Core_Linker, "DT_SCE_SYMENT is NOT 0x18 should check!");
            }
            break;
        case DT_DEBUG:
            dynamic_info.debug = dyn->d_un.d_val;
            break;
        case DT_TEXTREL:
            dynamic_info.textrel = dyn->d_un.d_val;
            break;
        case DT_FLAGS:
            dynamic_info.flags = dyn->d_un.d_val;
            if (dynamic_info.flags != 0x04) {
                LOG_WARNING(Core_Linker, "DT_FLAGS is NOT 0x04 should check!");
            }
            break;
        case DT_NEEDED:
            if (dynamic_info.str_table) {
                dynamic_info.needed.push_back(dynamic_info.str_table + dyn->d_un.d_val);
            } else {
                LOG_ERROR(Core_Linker, "DT_NEEDED str table is not loaded should check!");
            }
            break;
        case DT_SCE_NEEDED_MODULE: {
            ModuleInfo& info = dynamic_info.import_modules.emplace_back();
            info.value = dyn->d_un.d_val;
            info.name = dynamic_info.str_table + info.name_offset;
            info.enc_id = EncodeId(info.id);
            break;
        }
        case DT_SCE_IMPORT_LIB: {
            LibraryInfo& info = dynamic_info.import_libs.emplace_back();
            info.value = dyn->d_un.d_val;
            info.name = dynamic_info.str_table + info.name_offset;
            info.enc_id = EncodeId(info.id);
            break;
        }
        case DT_SCE_FINGERPRINT:
            LOG_INFO(Core_Linker, "DT_SCE_FINGERPRINT value = {:#018x}", dyn->d_un.d_val);
            std::memcpy(info.fingerprint.data(), &dyn->d_un.d_val, sizeof(SCE_DBG_NUM_FINGERPRINT));
            break;
        case DT_SCE_IMPORT_LIB_ATTR:
            LOG_INFO(Core_Linker, "unsupported DT_SCE_IMPORT_LIB_ATTR value = ......: {:#018x}",
                     dyn->d_un.d_val);
            break;
        case DT_SCE_ORIGINAL_FILENAME:
            dynamic_info.filename = dynamic_info.str_table + dyn->d_un.d_val;
            break;
        case DT_SCE_MODULE_INFO: {
            ModuleInfo& info = dynamic_info.export_modules.emplace_back();
            info.value = dyn->d_un.d_val;
            info.name = dynamic_info.str_table + info.name_offset;
            info.enc_id = EncodeId(info.id);
            const std::string full_name = info.name + ".sprx";
            full_name.copy(this->info.name.data(), full_name.size());
            break;
        };
        case DT_SCE_MODULE_ATTR:
            LOG_INFO(Core_Linker, "unsupported DT_SCE_MODULE_ATTR value = ..........: {:#018x}",
                     dyn->d_un.d_val);
            break;
        case DT_SCE_EXPORT_LIB: {
            LibraryInfo& info = dynamic_info.export_libs.emplace_back();
            info.value = dyn->d_un.d_val;
            info.name = dynamic_info.str_table + info.name_offset;
            info.enc_id = EncodeId(info.id);
            break;
        }
        default:
            LOG_INFO(Core_Linker, "unsupported dynamic tag ..........: {:#018x}", dyn->d_tag);
        }
    }
    if (!saw_dynamic_null) {
        LOG_WARNING(Core_Linker, "Dynamic table ended without DT_NULL after {} entries",
                    dynamic_count);
        ExecutorModuleTrace("dynamic_parse_missing_null", file,
                            std::string("entries=") + std::to_string(dynamic_count));
    }
    const u32 relabits_num = dynamic_info.relocation_table_size / sizeof(elf_relocation) +
                             dynamic_info.jmp_relocation_table_size / sizeof(elf_relocation);
    rela_bits.resize((relabits_num + 7) / 8);
    ExecutorModuleTrace("dynamic_parse_done", file,
                        std::string("needed=") + std::to_string(dynamic_info.needed.size()) +
                            ",import_modules=" +
                            std::to_string(dynamic_info.import_modules.size()) +
                            ",import_libs=" +
                            std::to_string(dynamic_info.import_libs.size()) + ",relocs=" +
                            std::to_string(relabits_num));
}

void Module::LoadSymbols() {
    std::size_t export_count = 0;
    std::size_t import_count = 0;
#ifdef __ANDROID__
    std::size_t mono_named_exports = 0;
    std::size_t mono_named_imports = 0;
    std::string mono_export_samples;
    std::string mono_import_samples;
#endif
    const auto symbol_database = [this, &export_count, &import_count
#ifdef __ANDROID__
                                  ,
                                  &mono_named_exports, &mono_named_imports, &mono_export_samples,
                                  &mono_import_samples
#endif
    ](Loader::SymbolsResolver& symbol, bool export_func) {
        if (!dynamic_info.symbol_table || !dynamic_info.str_table ||
            dynamic_info.symbol_table_total_size == 0) {
            LOG_INFO(Core_Linker, "Symbol table not found!");
            return;
        }
        for (auto* sym = dynamic_info.symbol_table;
             reinterpret_cast<u8*>(sym) < reinterpret_cast<u8*>(dynamic_info.symbol_table) +
                                              dynamic_info.symbol_table_total_size;
             sym++) {
            const u8 bind = sym->GetBind();
            const u8 type = sym->GetType();
            const u8 visibility = sym->GetVisibility();
            const auto id = std::string(dynamic_info.str_table + sym->st_name);
            const auto ids = Common::SplitString(id, '#');
            if (ids.size() != 3) {
                continue;
            }

            const auto* library = FindLibrary(ids[1]);
            const auto* module = FindModule(ids[2]);
            ASSERT_MSG(library && module, "Unable to find library and module");
            if ((bind != STB_GLOBAL && bind != STB_WEAK) ||
                (type != STT_FUN && type != STT_OBJECT) || export_func != (sym->st_value != 0)) {
                continue;
            }

            const auto aeronid = AeroLib::FindByNid(ids.at(0).c_str());
            const auto nid_name = aeronid ? aeronid->name : "UNK";

            Loader::SymbolResolver sym_r{};
            sym_r.name = ids.at(0);
            sym_r.nidName = nid_name;
            sym_r.library = library->name;
            sym_r.library_version = library->version;
            sym_r.module = module->name;
            switch (type) {
            case STT_NOTYPE:
                sym_r.type = Loader::SymbolType::NoType;
                break;
            case STT_FUN:
                sym_r.type = Loader::SymbolType::Function;
                break;
            case STT_OBJECT:
                sym_r.type = Loader::SymbolType::Object;
                break;
            default:
                sym_r.type = Loader::SymbolType::Unknown;
                break;
            }
            const VAddr sym_addr = export_func ? sym->st_value + base_virtual_addr : 0;
            symbol.AddSymbol(sym_r, sym_addr);
#ifdef __ANDROID__
            auto append_sample = [](std::string& out, const std::string& value) {
                if (out.size() > 512) {
                    return;
                }
                if (!out.empty()) {
                    out += ",";
                }
                out += value;
            };
            const std::string nid_name_string = nid_name ? std::string(nid_name) : std::string{};
            if (id.find("mono") != std::string::npos ||
                nid_name_string.find("mono") != std::string::npos) {
                if (export_func) {
                    mono_named_exports++;
                    append_sample(mono_export_samples, nid_name_string + ":" + id);
                } else {
                    mono_named_imports++;
                    append_sample(mono_import_samples, nid_name_string + ":" + id);
                }
            }
#endif
            if (export_func) {
                export_count++;
            } else {
                import_count++;
            }
        }
    };
    symbol_database(export_sym, true);
    symbol_database(import_sym, false);
#ifdef __ANDROID__
    if (file.string().find("mono-ps4") != std::string::npos) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_MONO_SYMBOLS] path=%s symtabBytes=%llu strtabBytes=%llu exports=%zu "
            "imports=%zu exportResolver=%zu importResolver=%zu monoExports=%zu monoImports=%zu",
            file.string().c_str(),
            static_cast<unsigned long long>(dynamic_info.symbol_table_total_size),
            static_cast<unsigned long long>(dynamic_info.str_table_size), export_count,
            import_count, export_sym.GetSize(), import_sym.GetSize(), mono_named_exports,
            mono_named_imports);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_SYMBOL_SAMPLES] exports=%s",
                            mono_export_samples.empty() ? "-" : mono_export_samples.c_str());
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_MONO_SYMBOL_SAMPLES] imports=%s",
                            mono_import_samples.empty() ? "-" : mono_import_samples.c_str());
    }
#endif
}

OrbisKernelModuleInfoEx Module::GetModuleInfoEx() const {
    return OrbisKernelModuleInfoEx{
        .name = info.name,
        .tls_index = tls.modid,
        .tls_init_addr = tls.image_virtual_addr,
        .tls_init_size = tls.init_image_size,
        .tls_size = tls.image_size,
        .tls_offset = tls.offset,
        .tls_align = tls.align,
        .init_proc_addr = base_virtual_addr + dynamic_info.init_virtual_addr,
        .fini_proc_addr = base_virtual_addr + dynamic_info.fini_virtual_addr,
        .eh_frame_hdr_addr = base_virtual_addr + eh_frame_hdr_addr,
        .eh_frame_addr = base_virtual_addr + eh_frame_addr,
        .eh_frame_hdr_size = eh_frame_hdr_size,
        .eh_frame_size = eh_frame_size,
        .segments = info.segments,
        .segment_count = info.num_segments,
    };
}

const ModuleInfo* Module::FindModule(std::string_view id) {
    const auto& import_modules = dynamic_info.import_modules;
    for (u32 i = 0; const auto& mod : import_modules) {
        if (mod.enc_id == id) {
            return &import_modules[i];
        }
        i++;
    }
    const auto& export_modules = dynamic_info.export_modules;
    for (u32 i = 0; const auto& mod : export_modules) {
        if (mod.enc_id == id) {
            return &export_modules[i];
        }
        i++;
    }
    return nullptr;
}

const LibraryInfo* Module::FindLibrary(std::string_view id) {
    const auto& import_libs = dynamic_info.import_libs;
    for (u32 i = 0; const auto& lib : import_libs) {
        if (lib.enc_id == id) {
            return &import_libs[i];
        }
        i++;
    }
    const auto& export_libs = dynamic_info.export_libs;
    for (u32 i = 0; const auto& lib : export_libs) {
        if (lib.enc_id == id) {
            return &export_libs[i];
        }
        i++;
    }
    return nullptr;
}

void* Module::FindByName(std::string_view name) {
    const auto nid_str = StringToNid(name);
    const auto symbols = export_sym.GetSymbols();
    const auto it = std::ranges::find_if(
        symbols, [&](const Loader::SymbolRecord& record) { return record.name.contains(nid_str); });
    if (it != symbols.end()) {
        return reinterpret_cast<void*>(it->virtual_address);
    }
    return nullptr;
}

}
