// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <string>
#include <vector>

#include "common/io_file.h"
#include "common/types.h"

struct self_header {
    static constexpr u32 signature = 0x1D3D154Fu;

    u32 magic;
    u8 version;
    u8 mode;
    u8 endian;
    u8 attributes;
    u8 category;
    u8 program_type;
    u16 padding1;
    u16 header_size;
    u16 meta_size;
    u32 file_size;
    u32 padding2;
    u16 segment_count;
    u16 unknown1A;
    u32 padding3;
};

struct self_segment_header {
    bool IsBlocked() const {
        return (flags & 0x800) != 0;
    }

    u32 GetId() const {
        return (flags >> 20u) & 0xFFFu;
    }

    bool IsOrdered() const {
        return (flags & 1) != 0;
    }

    bool IsEncrypted() const {
        return (flags & 2) != 0;
    }

    bool IsSigned() const {
        return (flags & 4) != 0;
    }

    bool IsCompressed() const {
        return (flags & 8) != 0;
    }

    u64 flags;
    u64 file_offset;
    u64 file_size;
    u64 memory_size;
};

constexpr u8 EI_MAG0 = 0;
constexpr u8 EI_MAG1 = 1;
constexpr u8 EI_MAG2 = 2;
constexpr u8 EI_MAG3 = 3;
constexpr u8 EI_CLASS = 4;
constexpr u8 EI_DATA = 5;
constexpr u8 EI_VERSION = 6;
constexpr u8 EI_OSABI = 7;
constexpr u8 EI_ABIVERSION = 8;

constexpr u8 ELFMAG0 = 0x7F;
constexpr u8 ELFMAG1 = 'E';
constexpr u8 ELFMAG2 = 'L';
constexpr u8 ELFMAG3 = 'F';

typedef enum : u16 {
    ET_NONE = 0x0,
    ET_REL = 0x1,
    ET_EXEC = 0x2,
    ET_DYN = 0x3,
    ET_CORE = 0x4,
    ET_SCE_EXEC = 0xfe00,
    ET_SCE_STUBLIB = 0xfe0c,
    ET_SCE_DYNEXEC = 0xfe10,
    ET_SCE_DYNAMIC = 0xfe18
} e_type_s;

typedef enum : u16 {
    EM_NONE = 0,
    EM_M32 = 1,
    EM_SPARC = 2,
    EM_386 = 3,
    EM_68K = 4,
    EM_88K = 5,
    EM_860 = 7,
    EM_MIPS = 8,
    EM_S370 = 9,
    EM_MIPS_RS3_LE = 10,
    EM_PARISC = 15,
    EM_VPP500 = 17,
    EM_SPARC32PLUS = 18,
    EM_960 = 19,
    EM_PPC = 20,
    EM_PPC64 = 21,
    EM_S390 = 22,
    EM_V800 = 36,
    EM_FR20 = 37,
    EM_RH32 = 38,
    EM_RCE = 39,
    EM_ARM = 40,
    EM_ALPHA = 41,
    EM_SH = 42,
    EM_SPARCV9 = 43,
    EM_TRICORE = 44,
    EM_ARC = 45,
    EM_H8_300 = 46,
    EM_H8_300H = 47,
    EM_H8S = 48,
    EM_H8_500 = 49,
    EM_IA_64 = 50,
    EM_MIPS_X = 51,
    EM_COLDFIRE = 52,
    EM_68HC12 = 53,
    EM_MMA = 54,
    EM_PCP = 55,
    EM_NCPU = 56,
    EM_NDR1 = 57,
    EM_STARCORE = 58,
    EM_ME16 = 59,
    EM_ST100 = 60,
    EM_TINYJ = 61,
    EM_X86_64 = 62,
    EM_PDSP = 63,
    EM_PDP10 = 64,
    EM_PDP11 = 65,
    EM_FX66 = 66,
    EM_ST9PLUS = 67,
    EM_ST7 = 68,
    EM_68HC16 = 69,
    EM_68HC11 = 70,
    EM_68HC08 = 71,
    EM_68HC05 = 72,
    EM_SVX = 73,
    EM_ST19 = 75,
    EM_CRIS = 76,
    EM_JAVELIN = 77,
    EM_FIREPATH = 78,
    EM_ZSP = 79,
    EM_MMIX = 80,
    EM_HUANY = 81,
    EM_PRISM = 82,
    EM_AVR = 83,
    EM_FR30 = 84,
    EM_D10V = 85,
    EM_D30V = 86,
    EM_V850 = 87,
    EM_M32R = 88,
    EM_MN10300 = 89,
    EM_MN10200 = 90,
    EM_PJ = 91,
    EM_OPENRISC = 92,
    EM_ARC_A5 = 93,
    EM_XTENSA = 94,
    EM_VIDEOCORE = 95,
    EM_TMM_GPP = 96,
    EM_NS32K = 97,
    EM_TPC = 98,
    EM_SNP1K = 99,
    EM_ST200 = 100,
    EM_IP2K = 101,
    EM_MAX = 102,
    EM_CR = 103,
    EM_F2MC16 = 104,
    EM_MSP430 = 105,
    EM_BLACKFIN = 106,
    EM_SE_C33 = 107,
    EM_SEP = 108,
    EM_ARCA = 109,
    EM_UNICORE = 110
} e_machine_es;

typedef enum : u32 { EV_NONE = 0x0, EV_CURRENT = 0x1 } e_version_es;

typedef enum : u8 {
    ELF_CLASS_NONE = 0x0,
    ELF_CLASS_32 = 0x1,
    ELF_CLASS_64 = 0x2,
    ELF_CLASS_NUM = 0x3
} ident_class_es;

typedef enum : u8 {
    ELF_DATA_NONE = 0x0,
    ELF_DATA_2LSB = 0x1,
    ELF_DATA_2MSB = 0x2,
    ELF_DATA_NUM = 0x3
} ident_endian_es;

typedef enum : u8 {
    ELF_VERSION_NONE = 0x0,
    ELF_VERSION_CURRENT = 0x1,
    ELF_VERSION_NUM = 0x2
} ident_version_es;

typedef enum : u8 {
    ELF_OSABI_NONE = 0x0,
    ELF_OSABI_HPUX = 0x1,
    ELF_OSABI_NETBSD = 0x2,
    ELF_OSABI_LINUX = 0x3,
    ELF_OSABI_SOLARIS = 0x6,
    ELF_OSABI_AIX = 0x7,
    ELF_OSABI_IRIX = 0x8,
    ELF_OSABI_FREEBSD = 0x9,
    ELF_OSABI_TRU64 = 0xA,
    ELF_OSABI_MODESTO = 0xB,
    ELF_OSABI_OPENBSD = 0xC,
    ELF_OSABI_OPENVMS = 0xD,
    ELF_OSABI_NSK = 0xE,
    ELF_OSABI_AROS = 0xF,
    ELF_OSABI_ARM_AEABI = 0x40,
    ELF_OSABI_ARM = 0x61,
    ELF_OSABI_STANDALONE = 0xFF
} ident_osabi_es;

typedef enum : u8 {
    ELF_ABI_VERSION_AMDGPU_HSA_V2 = 0x0,
    ELF_ABI_VERSION_AMDGPU_HSA_V3 = 0x1,
    ELF_ABI_VERSION_AMDGPU_HSA_V4 = 0x2,
    ELF_ABI_VERSION_AMDGPU_HSA_V5 = 0x3
} ident_abiversion_es;

struct elf_ident {
    u8 magic[4];
    ident_class_es ei_class;
    ident_endian_es ei_data;
    ident_version_es ei_version;
    ident_osabi_es ei_osabi;
    ident_abiversion_es ei_abiversion;
    u8 pad[6];
};

struct elf_header {
    static const u32 signature = 0x7F454C46u;

    elf_ident e_ident;
    e_type_s e_type;
    e_machine_es e_machine;
    e_version_es e_version;
    u64 e_entry;
    u64 e_phoff;
    u64 e_shoff;
    u32 e_flags;
    u16 e_ehsize;
    u16 e_phentsize;
    u16 e_phnum;
    u16 e_shentsize;
    u16 e_shnum;
    u16 e_shstrndx;
};

typedef enum : u32 {
    PT_NULL = 0x0,
    PT_LOAD = 0x1,
    PT_DYNAMIC = 0x2,
    PT_INTERP = 0x3,
    PT_NOTE = 0x4,
    PT_SHLIB = 0x5,
    PT_PHDR = 0x6,
    PT_TLS = 0x7,
    PT_NUM = 0x8,
    PT_SCE_RELA = 0x60000000,
    PT_SCE_DYNLIBDATA = 0x61000000,
    PT_SCE_PROCPARAM = 0x61000001,
    PT_SCE_MODULE_PARAM = 0x61000002,
    PT_SCE_RELRO = 0x61000010,
    PT_GNU_EH_FRAME = 0x6474e550,
    PT_GNU_STACK = 0x6474e551,
    PT_GNU_RELRO = 0x6474e552,
    PT_SCE_COMMENT = 0x6fffff00,
    PT_SCE_LIBVERSION = 0x6fffff01,
    PT_LOSUNW = 0x6ffffffa,
    PT_SUNWBSS = 0x6ffffffa,
    PT_SUNWSTACK = 0x6ffffffb,
    PT_HISUNW = 0x6fffffff,
    PT_HIOS = 0x6fffffff,
    PT_LOPROC = 0x70000000,
    PT_HIPROC = 0x7fffffff
} elf_program_type;

typedef enum : u32 {
    PF_NONE = 0x0,
    PF_EXEC = 0x1,
    PF_WRITE = 0x2,
    PF_WRITE_EXEC = 0x3,
    PF_READ = 0x4,
    PF_READ_EXEC = 0x5,
    PF_READ_WRITE = 0x6,
    PF_READ_WRITE_EXEC = 0x7
} elf_program_flags;

struct elf_program_header {
    elf_program_type p_type;
    elf_program_flags p_flags;
    u64 p_offset;
    u64 p_vaddr;
    u64 p_paddr;
    u64 p_filesz;
    u64 p_memsz;
    u64 p_align;
};

struct elf_section_header {
    u32 sh_name;
    u32 sh_type;
    u64 sh_flags;
    u64 sh_addr;
    u64 sh_offset;
    u64 sh_size;
    u32 sh_link;
    u32 sh_info;
    u64 sh_addralign;
    u64 sh_entsize;
};

typedef enum : u64 {
    PT_FAKE = 0x1,
    PT_NPDRM_EXEC = 0x4,
    PT_NPDRM_DYNLIB = 0x5,
    PT_SYSTEM_EXEC = 0x8,
    PT_SYSTEM_DYNLIB = 0x9,
    PT_HOST_KERNEL = 0xC,
    PT_SECURE_MODULE = 0xE,
    PT_SECURE_KERNEL = 0xF
} program_type_es;

struct elf_program_id_header {
    u64 authid;
    program_type_es program_type;
    u64 appver;
    u64 firmver;
    u8 digest[32];
};

constexpr s64 DT_NULL = 0;
constexpr s64 DT_NEEDED = 0x00000001;
constexpr s64 DT_RELA = 0x00000007;
constexpr s64 DT_INIT = 0x0000000c;
constexpr s64 DT_FINI = 0x0000000d;
constexpr s64 DT_DEBUG = 0x00000015;
constexpr s64 DT_TEXTREL = 0x00000016;
constexpr s64 DT_INIT_ARRAY = 0x00000019;
constexpr s64 DT_FINI_ARRAY = 0x0000001a;
constexpr s64 DT_INIT_ARRAYSZ = 0x0000001b;
constexpr s64 DT_FINI_ARRAYSZ = 0x0000001c;
constexpr s64 DT_FLAGS = 0x0000001e;
constexpr s64 DT_PREINIT_ARRAY = 0x00000020;
constexpr s64 DT_PREINIT_ARRAYSZ = 0x00000021;
constexpr s64 DT_SCE_FINGERPRINT = 0x61000007;
constexpr s64 DT_SCE_ORIGINAL_FILENAME = 0x61000009;
constexpr s64 DT_SCE_MODULE_INFO = 0x6100000d;
constexpr s64 DT_SCE_NEEDED_MODULE = 0x6100000f;
constexpr s64 DT_SCE_MODULE_ATTR = 0x61000011;
constexpr s64 DT_SCE_EXPORT_LIB = 0x61000013;
constexpr s64 DT_SCE_IMPORT_LIB = 0x61000015;
constexpr s64 DT_SCE_IMPORT_LIB_ATTR = 0x61000019;
constexpr s64 DT_SCE_HASH = 0x61000025;
constexpr s64 DT_SCE_PLTGOT = 0x61000027;
constexpr s64 DT_SCE_JMPREL = 0x61000029;
constexpr s64 DT_SCE_PLTREL = 0x6100002b;
constexpr s64 DT_SCE_PLTRELSZ = 0x6100002d;
constexpr s64 DT_SCE_RELA = 0x6100002f;
constexpr s64 DT_SCE_RELASZ = 0x61000031;
constexpr s64 DT_SCE_RELAENT = 0x61000033;
constexpr s64 DT_SCE_SYMENT = 0x6100003b;
constexpr s64 DT_SCE_HASHSZ = 0x6100003d;
constexpr s64 DT_SCE_STRTAB = 0x61000035;
constexpr s64 DT_SCE_STRSZ = 0x61000037;
constexpr s64 DT_SCE_SYMTAB = 0x61000039;
constexpr s64 DT_SCE_SYMTABSZ = 0x6100003f;

struct elf_dynamic {
    s64 d_tag;
    union {
        u64 d_val;
        u64 d_ptr;
    } d_un;
};

constexpr u8 STB_LOCAL = 0;
constexpr u8 STB_GLOBAL = 1;
constexpr u8 STB_WEAK = 2;

constexpr u8 STT_NOTYPE = 0;
constexpr u8 STT_OBJECT = 1;
constexpr u8 STT_FUN = 2;
constexpr u8 STT_SECTION = 3;
constexpr u8 STT_FILE = 4;
constexpr u8 STT_COMMON = 5;
constexpr u8 STT_TLS = 6;
constexpr u8 STT_LOOS = 10;
constexpr u8 STT_SCE = 11;
constexpr u8 STT_HIOS = 12;
constexpr u8 STT_LOPRO = 13;
constexpr u8 STT_SPARC_REGISTER = 13;
constexpr u8 STT_HIPROC = 15;

constexpr u8 STV_DEFAULT = 0;
constexpr u8 STV_INTERNAL = 1;
constexpr u8 STV_HIDDEN = 2;
constexpr u8 STV_PROTECTED = 3;

struct elf_symbol {
    u8 GetBind() const {
        return st_info >> 4u;
    }
    u8 GetType() const {
        return st_info & 0xfu;
    }
    u8 GetVisibility() const {
        return st_other & 3u;
    }

    u32 st_name;
    u8 st_info;
    u8 st_other;
    u16 st_shndx;
    u64 st_value;
    u64 st_size;
};

struct elf_relocation {
    u32 GetSymbol() const {
        return static_cast<u32>(rel_info >> 32u);
    }
    u32 GetType() const {
        return static_cast<u32>(rel_info & 0xffffffff);
    }

    u64 rel_offset;
    u64 rel_info;
    s64 rel_addend;
};
constexpr u32 R_X86_64_64 = 1;
constexpr u32 R_X86_64_GLOB_DAT = 6;
constexpr u32 R_X86_64_JUMP_SLOT = 7;
constexpr u32 R_X86_64_RELATIVE = 8;
constexpr u32 R_X86_64_DTPMOD64 = 16;

struct eh_frame_hdr {
    uint8_t version;
    uint8_t eh_frame_ptr_enc;
    uint8_t fde_count_enc;
    uint8_t table_enc;
    uint32_t eh_frame_ptr;
    uint32_t fde_count;
};

namespace Core::Loader {

class Elf {
public:
    Elf() = default;
    ~Elf();

    void Open(const std::filesystem::path& file_name);
    bool IsSelfFile() const;
    bool IsElfFile() const;

    [[nodiscard]] self_header GetSElfHeader() const {
        return m_self;
    }

    [[nodiscard]] elf_header GetElfHeader() const {
        return m_elf_header;
    }

    [[nodiscard]] std::span<const elf_program_header> GetProgramHeader() const {
        return m_elf_phdr;
    }

    [[nodiscard]] std::span<const self_segment_header> GetSegmentHeader() const {
        return m_self_segments;
    }

    [[nodiscard]] u64 GetElfEntry() const {
        return m_elf_header.e_entry;
    }

    [[nodiscard]] bool IsSharedLib() const {
        return m_elf_header.e_type == ET_SCE_DYNAMIC;
    }

    std::string SElfHeaderStr();
    std::string SELFSegHeader(u16 no);
    std::string ElfHeaderStr();
    std::string ElfPHeaderStr(u16 no);
    std::string_view ElfPheaderTypeStr(u32 type);
    std::string ElfPheaderFlagsStr(u32 flags);

    void LoadSegment(u64 virtual_addr, u64 file_offset, u64 size);
    bool IsSharedLib();
    void ElfHeaderDebugDump(const std::filesystem::path& file_name);
    void SelfHeaderDebugDump(const std::filesystem::path& file_name);
    void SelfSegHeaderDebugDump(const std::filesystem::path& file_name);
    void PHeaderDebugDump(const std::filesystem::path& file_name);

private:
    Common::FS::IOFile m_f{};
    bool is_self{};
    self_header m_self{};
    std::vector<self_segment_header> m_self_segments;
    elf_header m_elf_header{};
    std::vector<elf_program_header> m_elf_phdr;
    std::vector<elf_section_header> m_elf_shdr;
    elf_program_id_header m_self_id_header{};
};

}
