// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ps5_desktop/nextgen_loader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace Funnel::Ps5Desktop {
namespace {

constexpr std::uint64_t GuestPageSize = 0x4000;
constexpr std::uint64_t GuestAddressLimit = 0x10000000000ull;
constexpr std::uint64_t FirstImageBase = 0x900000000ull;
constexpr std::uint64_t ImageBaseStep = 0x1000000ull;
constexpr std::uint32_t PtLoad = 1;
constexpr std::uint32_t PtDynamic = 2;
constexpr std::uint32_t PtTls = 7;
constexpr std::uint32_t PtSceDynlibData = 0x61000000;
constexpr std::uint32_t PtSceProcParam = 0x61000001;
constexpr std::uint32_t PtSceRelro = 0x61000010;
constexpr std::uint32_t PfExecute = 1;
constexpr std::uint32_t PfWrite = 2;
constexpr std::uint32_t PfRead = 4;
constexpr std::uint16_t EtDyn = 3;
constexpr std::uint16_t EtSceDynExec = 0xfe10;
constexpr std::uint16_t EtSceDynamic = 0xfe18;
constexpr std::uint32_t EmX86_64 = 62;
constexpr std::int64_t DtNull = 0;
constexpr std::int64_t DtNeeded = 1;
constexpr std::int64_t DtRela = 7;
constexpr std::int64_t DtRelaSize = 8;
constexpr std::int64_t DtRelaEntry = 9;
constexpr std::int64_t DtStringTable = 5;
constexpr std::int64_t DtSymbolTable = 6;
constexpr std::int64_t DtStringSize = 10;
constexpr std::int64_t DtSymbolEntry = 11;
constexpr std::int64_t DtInit = 12;
constexpr std::int64_t DtFini = 13;
constexpr std::int64_t DtJumpRel = 23;
constexpr std::int64_t DtInitArray = 25;
constexpr std::int64_t DtFiniArray = 26;
constexpr std::int64_t DtInitArraySize = 27;
constexpr std::int64_t DtFiniArraySize = 28;
constexpr std::int64_t DtPreinitArray = 32;
constexpr std::int64_t DtPreinitArraySize = 33;
constexpr std::int64_t DtPltRelSize = 2;
constexpr std::int64_t DtSceJumpRel = 0x61000029;
constexpr std::int64_t DtSceRela = 0x6100002f;
constexpr std::int64_t DtSceRelaSize = 0x61000031;
constexpr std::int64_t DtSceRelaEntry = 0x61000033;
constexpr std::int64_t DtSceStringTable = 0x61000035;
constexpr std::int64_t DtSceStringSize = 0x61000037;
constexpr std::int64_t DtSceSymbolTable = 0x61000039;
constexpr std::int64_t DtSceSymbolEntry = 0x6100003b;
constexpr std::int64_t DtSceSymbolTableSize = 0x6100003f;
constexpr std::int64_t DtScePltRelSize = 0x6100002d;
constexpr std::uint32_t RelocNone = 0;
constexpr std::uint32_t Reloc64 = 1;
constexpr std::uint32_t RelocPc32 = 2;
constexpr std::uint32_t RelocPlt32 = 4;
constexpr std::uint32_t RelocGlobDat = 6;
constexpr std::uint32_t RelocJumpSlot = 7;
constexpr std::uint32_t RelocRelative = 8;
constexpr std::uint32_t RelocUnsigned32 = 10;
constexpr std::uint32_t RelocSigned32 = 11;
constexpr std::uint32_t RelocDtpMod64 = 16;
constexpr std::uint32_t RelocDtpOff64 = 17;
constexpr std::uint32_t RelocTpOff64 = 18;
constexpr std::uint32_t RelocPc64 = 24;
constexpr std::uint32_t RelocSize32 = 32;
constexpr std::uint32_t RelocSize64 = 33;
constexpr std::uint32_t RelocRelative64 = 38;
constexpr std::uint32_t SectionTypeRela = 4;
constexpr std::uint32_t SectionTypeSymbolTable = 2;
constexpr std::uint8_t SymbolBindGlobal = 1;
constexpr std::uint8_t SymbolBindWeak = 2;
constexpr std::uint8_t SymbolTypeObject = 1;
constexpr std::uint8_t SymbolTypeFunction = 2;
constexpr std::uint8_t SymbolTypeTls = 6;

#pragma pack(push, 1)
struct SelfHeader {
    std::uint8_t ident[12];
    std::uint16_t size1;
    std::uint16_t size2;
    std::uint64_t file_size;
    std::uint16_t segment_count;
    std::uint16_t unknown;
    std::uint32_t padding;
};

struct SelfSegment {
    std::uint64_t type;
    std::uint64_t offset;
    std::uint64_t compressed_size;
    std::uint64_t decompressed_size;
};

struct ElfHeader {
    std::uint8_t ident[16];
    std::uint16_t type;
    std::uint16_t machine;
    std::uint32_t version;
    std::uint64_t entry;
    std::uint64_t program_offset;
    std::uint64_t section_offset;
    std::uint32_t flags;
    std::uint16_t header_size;
    std::uint16_t program_entry_size;
    std::uint16_t program_count;
    std::uint16_t section_entry_size;
    std::uint16_t section_count;
    std::uint16_t section_name_index;
};

struct ProgramHeader {
    std::uint32_t type;
    std::uint32_t flags;
    std::uint64_t offset;
    std::uint64_t virtual_address;
    std::uint64_t physical_address;
    std::uint64_t file_size;
    std::uint64_t memory_size;
    std::uint64_t alignment;
};

struct DynamicEntry {
    std::int64_t tag;
    std::uint64_t value;
};

struct Symbol {
    std::uint32_t name;
    std::uint8_t info;
    std::uint8_t other;
    std::uint16_t section;
    std::uint64_t value;
    std::uint64_t size;
};

struct Rela {
    std::uint64_t offset;
    std::uint64_t info;
    std::int64_t addend;
};

struct SectionHeader {
    std::uint32_t name;
    std::uint32_t type;
    std::uint64_t flags;
    std::uint64_t address;
    std::uint64_t offset;
    std::uint64_t size;
    std::uint32_t link;
    std::uint32_t info;
    std::uint64_t alignment;
    std::uint64_t entry_size;
};
#pragma pack(pop)

static_assert(sizeof(SelfHeader) == 32);
static_assert(sizeof(SelfSegment) == 32);
static_assert(sizeof(ElfHeader) == 64);
static_assert(sizeof(ProgramHeader) == 56);
static_assert(sizeof(DynamicEntry) == 16);
static_assert(sizeof(Symbol) == 24);
static_assert(sizeof(Rela) == 24);
static_assert(sizeof(SectionHeader) == 64);

struct FileImage {
    std::vector<std::uint8_t> bytes;
    std::uint64_t elf_offset{};
    std::vector<SelfSegment> self_segments;
    bool self{};
};

struct DynamicView {
    std::uint64_t string_table{};
    std::uint64_t string_size{};
    std::uint64_t symbol_table{};
    std::uint64_t symbol_table_size{};
    std::uint64_t symbol_entry_size{sizeof(Symbol)};
    std::uint64_t rela{};
    std::uint64_t rela_size{};
    std::uint64_t rela_entry_size{sizeof(Rela)};
    std::uint64_t jump_rela{};
    std::uint64_t jump_rela_size{};
    std::uint64_t init{};
    std::uint64_t fini{};
    std::uint64_t init_array{};
    std::uint64_t init_array_size{};
    std::uint64_t fini_array{};
    std::uint64_t fini_array_size{};
    std::uint64_t preinit_array{};
    std::uint64_t preinit_array_size{};
    std::uint64_t metadata_begin{};
    std::uint64_t metadata_size{};
    std::vector<std::uint64_t> needed_name_offsets;
};

template <typename T>
bool ReadObject(const std::vector<std::uint8_t>& bytes,
                const std::uint64_t offset, T& value) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
        return false;
    }
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return true;
}

bool AddFits(const std::uint64_t left, const std::uint64_t right,
             std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

std::uint64_t AlignDown(const std::uint64_t value,
                        const std::uint64_t alignment) {
    return value & ~(alignment - 1u);
}

bool AlignUp(const std::uint64_t value, const std::uint64_t alignment,
             std::uint64_t& result) {
    if (value > std::numeric_limits<std::uint64_t>::max() -
                    (alignment - 1u)) {
        return false;
    }
    result = (value + alignment - 1u) & ~(alignment - 1u);
    return true;
}

bool IsPowerOfTwo(const std::uint64_t value) {
    return value != 0 && (value & (value - 1u)) == 0;
}

bool CalculateStaticTlsOffset(const std::uint64_t previous_offset,
                              const std::uint64_t size,
                              const std::uint64_t alignment,
                              const std::uint64_t alignment_bias,
                              std::uint64_t& result) {
    if (!IsPowerOfTwo(alignment) ||
        size > std::numeric_limits<std::uint64_t>::max() - previous_offset ||
        previous_offset + size >
            std::numeric_limits<std::uint64_t>::max() - (alignment - 1u)) {
        return false;
    }
    const auto candidate =
        previous_offset + size + alignment - 1u;
    if (alignment_bias >
        std::numeric_limits<std::uint64_t>::max() - candidate) {
        return false;
    }
    result = candidate -
             ((candidate + alignment_bias) & (alignment - 1u));
    return result >= previous_offset && result >= size;
}

bool IsKnownSelf(const SelfHeader& header) {
    const bool magic =
        (header.ident[0] == 0x4f && header.ident[1] == 0x15 &&
         header.ident[2] == 0x3d && header.ident[3] == 0x1d) ||
        (header.ident[0] == 0x54 && header.ident[1] == 0x14 &&
         header.ident[2] == 0xf5 && header.ident[3] == 0xee);
    return magic && header.segment_count != 0 &&
           header.segment_count <= 4096;
}

bool ReadFile(const char* path, FileImage& file, std::string& error) {
    std::ifstream stream{std::filesystem::path{path},
                         std::ios::binary | std::ios::ate};
    if (!stream) {
        error = "cannot open executable";
        return false;
    }
    const auto end = stream.tellg();
    if (end < 0 ||
        static_cast<std::uint64_t>(end) >
            std::numeric_limits<std::size_t>::max()) {
        error = "executable is too large";
        return false;
    }
    file.bytes.resize(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!file.bytes.empty() &&
        !stream.read(reinterpret_cast<char*>(file.bytes.data()),
                     static_cast<std::streamsize>(file.bytes.size()))) {
        error = "cannot read executable";
        return false;
    }

    SelfHeader self{};
    if (!ReadObject(file.bytes, 0, self) || !IsKnownSelf(self)) {
        return true;
    }
    std::uint64_t table_size{};
    if (!AddFits(sizeof(SelfHeader),
                 static_cast<std::uint64_t>(self.segment_count) *
                     sizeof(SelfSegment),
                 table_size) ||
        table_size > file.bytes.size()) {
        error = "truncated SELF segment table";
        return false;
    }
    file.self = true;
    file.elf_offset = table_size;
    file.self_segments.resize(self.segment_count);
    std::memcpy(file.self_segments.data(),
                file.bytes.data() + sizeof(SelfHeader),
                file.self_segments.size() * sizeof(SelfSegment));
    return true;
}

bool ReadElfRelativeObject(const FileImage& file,
                           const std::uint64_t elf_relative_offset,
                           auto& value) {
    std::uint64_t absolute_offset{};
    return AddFits(file.elf_offset, elf_relative_offset, absolute_offset) &&
           ReadObject(file.bytes, absolute_offset, value);
}

bool ReadSectionHeader(const FileImage& file, const ElfHeader& header,
                       const std::uint32_t section_index,
                       SectionHeader& section) {
    if (section_index >= header.section_count ||
        header.section_entry_size < sizeof(SectionHeader)) {
        return false;
    }
    std::uint64_t relative_offset{};
    return AddFits(
               header.section_offset,
               static_cast<std::uint64_t>(section_index) *
                   header.section_entry_size,
               relative_offset) &&
           ReadElfRelativeObject(file, relative_offset, section);
}

bool ElfRelativeSpan(const FileImage& file,
                     const std::uint64_t elf_relative_offset,
                     const std::uint64_t byte_count,
                     std::span<const std::uint8_t>& span) {
    std::uint64_t absolute_offset{};
    if (!AddFits(file.elf_offset, elf_relative_offset, absolute_offset) ||
        absolute_offset > file.bytes.size() ||
        byte_count > file.bytes.size() - absolute_offset) {
        return false;
    }
    span = std::span{
        file.bytes.data() + static_cast<std::size_t>(absolute_offset),
        static_cast<std::size_t>(byte_count)};
    return true;
}

bool ReadSegmentBytes(const FileImage& file,
                      const std::vector<ProgramHeader>& programs,
                      const std::size_t program_index,
                      void* destination, const std::uint64_t byte_count,
                      std::string& error) {
    const auto& program = programs[program_index];
    std::uint64_t source_offset{};
    if (!file.self) {
        if (!AddFits(file.elf_offset, program.offset, source_offset)) {
            error = "ELF segment offset overflow";
            return false;
        }
    } else {
        const auto match = std::ranges::find_if(
            file.self_segments, [&](const SelfSegment& segment) {
                return (segment.type & 0x800u) != 0 &&
                       ((segment.type >> 20u) & 0xfffu) == program_index;
            });
        if (match == file.self_segments.end()) {
            error = "SELF has no unencrypted image for PT_LOAD";
            return false;
        }
        if (match->compressed_size != match->decompressed_size ||
            match->decompressed_size < byte_count) {
            error = "compressed or encrypted SELF segment is unsupported";
            return false;
        }
        source_offset = match->offset;
    }
    if (source_offset > file.bytes.size() ||
        byte_count > file.bytes.size() - source_offset) {
        error = "truncated PT_LOAD data";
        return false;
    }
    std::memcpy(destination, file.bytes.data() + source_offset,
                static_cast<std::size_t>(byte_count));
    return true;
}

bool ReadAuxiliarySegment(
    const FileImage& file,
    const std::vector<ProgramHeader>& programs,
    const std::size_t program_index,
    std::vector<std::uint8_t>& output, std::string& error) {
    const auto& program = programs[program_index];
    if (program.file_size >
        std::numeric_limits<std::size_t>::max()) {
        error = "PS5 metadata segment is too large";
        return false;
    }
    output.resize(static_cast<std::size_t>(program.file_size));
    return output.empty() ||
           ReadSegmentBytes(file, programs, program_index,
                            output.data(), output.size(), error);
}

std::uint32_t LoaderFlags(const std::uint32_t elf_flags) {
    std::uint32_t protection{};
    if ((elf_flags & PfRead) != 0) {
        protection |= LoaderRead;
    }
    if ((elf_flags & PfWrite) != 0) {
        protection |= LoaderWrite;
    }
    if ((elf_flags & PfExecute) != 0) {
        protection |= LoaderRead | LoaderExecute;
    }
    return protection;
}

int NativeProtection(const std::uint32_t protection) {
    int native = PROT_NONE;
    if ((protection & LoaderRead) != 0) {
        native |= PROT_READ;
    }
    if ((protection & LoaderWrite) != 0) {
        native |= PROT_WRITE;
    }
    if ((protection & LoaderExecute) != 0) {
        native |= PROT_EXEC;
    }
    return native;
}

bool AddressInImage(const LoadedImage& image, const std::uint64_t address,
                    const std::uint64_t size) {
    return address >= image.base && size <= image.mapped_size &&
           address - image.base <= image.mapped_size - size;
}

bool AddressInDynamicMetadata(const DynamicView& dynamic,
                              const std::uint64_t address,
                              const std::uint64_t size) {
    return address >= dynamic.metadata_begin &&
           size <= dynamic.metadata_size &&
           address - dynamic.metadata_begin <=
               dynamic.metadata_size - size;
}

bool AddressInLoaderData(const LoadedImage& image,
                         const DynamicView& dynamic,
                         const std::uint64_t address,
                         const std::uint64_t size) {
    return AddressInImage(image, address, size) ||
           AddressInDynamicMetadata(dynamic, address, size);
}

bool ParseDynamic(const LoadedImage& image,
                  const std::span<const std::uint8_t> entries_image,
                  const std::span<const std::uint8_t> metadata,
                  DynamicView& view, std::string& error) {
    if (entries_image.empty()) {
        return true;
    }
    if (entries_image.size() % sizeof(DynamicEntry) != 0) {
        error = "truncated PT_DYNAMIC data";
        return false;
    }
    view.metadata_begin =
        reinterpret_cast<std::uint64_t>(metadata.data());
    view.metadata_size = metadata.size();
    const auto count = entries_image.size() / sizeof(DynamicEntry);
    const auto* entries = reinterpret_cast<const DynamicEntry*>(
        entries_image.data());
    const auto sce_pointer = [&](const std::uint64_t offset) {
        return view.metadata_begin + offset;
    };
    for (std::uint64_t index = 0; index < count; ++index) {
        const auto& entry = entries[index];
        if (entry.tag == DtNull) {
            break;
        }
        switch (entry.tag) {
        case DtNeeded:
            view.needed_name_offsets.push_back(entry.value);
            break;
        case DtStringTable:
            view.string_table = image.base + entry.value;
            break;
        case DtSceStringTable:
            view.string_table = sce_pointer(entry.value);
            break;
        case DtStringSize:
        case DtSceStringSize:
            view.string_size = entry.value;
            break;
        case DtSymbolTable:
            view.symbol_table = image.base + entry.value;
            break;
        case DtSceSymbolTable:
            view.symbol_table = sce_pointer(entry.value);
            break;
        case DtSceSymbolTableSize:
            view.symbol_table_size = entry.value;
            break;
        case DtSymbolEntry:
        case DtSceSymbolEntry:
            view.symbol_entry_size = entry.value;
            break;
        case DtInit:
            view.init = image.base + entry.value;
            break;
        case DtFini:
            view.fini = image.base + entry.value;
            break;
        case DtInitArray:
            view.init_array = image.base + entry.value;
            break;
        case DtInitArraySize:
            view.init_array_size = entry.value;
            break;
        case DtFiniArray:
            view.fini_array = image.base + entry.value;
            break;
        case DtFiniArraySize:
            view.fini_array_size = entry.value;
            break;
        case DtPreinitArray:
            view.preinit_array = image.base + entry.value;
            break;
        case DtPreinitArraySize:
            view.preinit_array_size = entry.value;
            break;
        case DtRela:
            view.rela = image.base + entry.value;
            break;
        case DtSceRela:
            view.rela = sce_pointer(entry.value);
            break;
        case DtRelaSize:
        case DtSceRelaSize:
            view.rela_size = entry.value;
            break;
        case DtRelaEntry:
        case DtSceRelaEntry:
            view.rela_entry_size = entry.value;
            break;
        case DtJumpRel:
            view.jump_rela = image.base + entry.value;
            break;
        case DtSceJumpRel:
            view.jump_rela = sce_pointer(entry.value);
            break;
        case DtPltRelSize:
        case DtScePltRelSize:
            view.jump_rela_size = entry.value;
            break;
        default:
            break;
        }
    }
    if ((view.rela_size != 0 || view.jump_rela_size != 0 ||
         view.symbol_table_size != 0) &&
        view.symbol_entry_size != sizeof(Symbol)) {
        error = "unsupported PS5 symbol table entry size";
        return false;
    }
    if ((view.rela_size != 0 || view.jump_rela_size != 0) &&
        view.rela_entry_size != sizeof(Rela)) {
        error = "unsupported PS5 relocation table entry size";
        return false;
    }
    return true;
}

bool ReadDynamicString(const LoadedImage& image,
                       const DynamicView& dynamic,
                       const std::uint64_t offset,
                       std::string& value, std::string& error) {
    if (dynamic.string_table == 0 || offset >= dynamic.string_size ||
        !AddressInLoaderData(
            image, dynamic, dynamic.string_table + offset, 1)) {
        error = "invalid PS5 dynamic string offset";
        return false;
    }
    const auto* const begin =
        reinterpret_cast<const char*>(dynamic.string_table + offset);
    const auto remaining =
        static_cast<std::size_t>(dynamic.string_size - offset);
    const auto* const end = static_cast<const char*>(
        std::memchr(begin, '\0', remaining));
    if (end == nullptr) {
        error = "unterminated PS5 dynamic string";
        return false;
    }
    value.assign(begin, static_cast<std::size_t>(end - begin));
    return true;
}

void AppendExportAlias(LoadedImage& image, std::string name,
                       const std::uint64_t address,
                       const std::uint64_t size) {
    if (name.empty() ||
        std::ranges::any_of(
            image.exported_symbols,
            [&](const LoadedSymbol& symbol) {
                return symbol.name == name;
            })) {
        return;
    }
    image.exported_symbols.push_back(
        {std::move(name), address, size});
}

bool CollectDynamicMetadata(LoadedImage& image,
                            const DynamicView& dynamic,
                            std::string& error) {
    for (const auto offset : dynamic.needed_name_offsets) {
        std::string dependency;
        if (!ReadDynamicString(
                image, dynamic, offset, dependency, error)) {
            return false;
        }
        if (!dependency.empty() &&
            !std::ranges::contains(image.dependencies, dependency)) {
            image.dependencies.push_back(std::move(dependency));
        }
    }

    auto table_size = dynamic.symbol_table_size;
    if (table_size == 0 && dynamic.symbol_table != 0 &&
        dynamic.string_table > dynamic.symbol_table &&
        (dynamic.string_table - dynamic.symbol_table) %
                sizeof(Symbol) ==
            0) {
        table_size = dynamic.string_table - dynamic.symbol_table;
    }
    if (table_size == 0) {
        return true;
    }
    if (dynamic.symbol_table == 0 ||
        table_size % sizeof(Symbol) != 0 ||
        !AddressInLoaderData(
            image, dynamic, dynamic.symbol_table, table_size)) {
        error = "invalid PS5 dynamic symbol table";
        return false;
    }

    const auto* const symbols =
        reinterpret_cast<const Symbol*>(dynamic.symbol_table);
    const auto symbol_count = table_size / sizeof(Symbol);
    for (std::uint64_t index = 1; index < symbol_count; ++index) {
        const auto& symbol = symbols[index];
        const auto binding = static_cast<std::uint8_t>(symbol.info >> 4u);
        if ((binding != SymbolBindGlobal &&
             binding != SymbolBindWeak) ||
            symbol.section == 0 || symbol.name == 0) {
            continue;
        }
        std::string name;
        if (!ReadDynamicString(
                image, dynamic, symbol.name, name, error)) {
            return false;
        }
        std::uint64_t address{};
        if (!AddFits(image.base, symbol.value, address) ||
            !AddressInImage(
                image, address, symbol.size == 0 ? 1 : symbol.size)) {
            error = "PS5 exported symbol is outside mapped image";
            return false;
        }
        AppendExportAlias(image, name, address, symbol.size);
        if (const auto separator = name.find('#');
            separator != std::string::npos) {
            AppendExportAlias(
                image, name.substr(0, separator), address, symbol.size);
        }
        if (name.starts_with('_') && name.size() > 1) {
            AppendExportAlias(
                image, name.substr(1), address, symbol.size);
        }
    }
    return true;
}

void CollectSectionExports(const FileImage& file,
                           const ElfHeader& header,
                           LoadedImage& image) {
    if (header.section_offset == 0 || header.section_count == 0 ||
        header.section_entry_size < sizeof(SectionHeader)) {
        return;
    }
    for (std::uint32_t section_index = 0;
         section_index < header.section_count; ++section_index) {
        SectionHeader symbol_section{};
        if (!ReadSectionHeader(
                file, header, section_index, symbol_section) ||
            symbol_section.type != SectionTypeSymbolTable ||
            symbol_section.size == 0 ||
            symbol_section.entry_size < sizeof(Symbol) ||
            symbol_section.link >= header.section_count) {
            continue;
        }
        SectionHeader string_section{};
        std::span<const std::uint8_t> strings;
        if (!ReadSectionHeader(
                file, header, symbol_section.link, string_section) ||
            !ElfRelativeSpan(
                file, string_section.offset, string_section.size, strings)) {
            continue;
        }
        const auto symbol_count =
            symbol_section.size / symbol_section.entry_size;
        for (std::uint64_t symbol_index = 1;
             symbol_index < symbol_count; ++symbol_index) {
            std::uint64_t symbol_offset{};
            Symbol symbol{};
            if (!AddFits(
                    symbol_section.offset,
                    symbol_index * symbol_section.entry_size,
                    symbol_offset) ||
                !ReadElfRelativeObject(file, symbol_offset, symbol)) {
                break;
            }
            const auto binding =
                static_cast<std::uint8_t>(symbol.info >> 4u);
            if ((binding != SymbolBindGlobal &&
                 binding != SymbolBindWeak) ||
                symbol.section == 0 || symbol.name == 0 ||
                symbol.name >= strings.size()) {
                continue;
            }
            const auto* const begin = reinterpret_cast<const char*>(
                strings.data() + symbol.name);
            const auto* const end = static_cast<const char*>(
                std::memchr(
                    begin, '\0',
                    strings.size() - symbol.name));
            if (end == nullptr || end == begin) {
                continue;
            }
            std::uint64_t address{};
            if (!AddFits(image.base, symbol.value, address) ||
                !AddressInImage(
                    image, address,
                    symbol.size == 0 ? 1 : symbol.size)) {
                continue;
            }
            std::string name{
                begin, static_cast<std::size_t>(end - begin)};
            AppendExportAlias(image, name, address, symbol.size);
            if (const auto separator = name.find('#');
                separator != std::string::npos) {
                AppendExportAlias(
                    image, name.substr(0, separator),
                    address, symbol.size);
            }
            if (name.starts_with('_') && name.size() > 1) {
                AppendExportAlias(
                    image, name.substr(1), address, symbol.size);
            }
        }
    }
}

bool AppendInitializer(const LoadedImage& image,
                       const std::uint64_t raw_address,
                       std::vector<std::uint64_t>& functions,
                       std::string& error) {
    if (raw_address == 0 ||
        raw_address == std::numeric_limits<std::uint64_t>::max()) {
        return true;
    }
    auto address = raw_address;
    if (!AddressInImage(image, address, 1) &&
        raw_address < image.mapped_size) {
        address = image.base + raw_address;
    }
    if (!AddressInImage(image, address, 1)) {
        error = "PS5 initializer points outside mapped image";
        return false;
    }
    if (!std::ranges::contains(functions, address)) {
        functions.push_back(address);
    }
    return true;
}

bool AppendInitializerArray(const LoadedImage& image,
                            const std::uint64_t array_address,
                            const std::uint64_t array_size,
                            const bool reverse,
                            std::vector<std::uint64_t>& functions,
                            std::string& error) {
    if (array_size == 0) {
        return true;
    }
    if (array_size % sizeof(std::uint64_t) != 0 ||
        !AddressInImage(image, array_address, array_size)) {
        error = "invalid PS5 initializer array";
        return false;
    }
    const auto* entries =
        reinterpret_cast<const std::uint64_t*>(array_address);
    const auto count = array_size / sizeof(std::uint64_t);
    for (std::uint64_t position = 0; position < count; ++position) {
        const auto index = reverse ? count - position - 1 : position;
        if (!AppendInitializer(
                image, entries[index], functions, error)) {
            return false;
        }
    }
    return true;
}

bool CollectInitializers(LoadedImage& image, const DynamicView& dynamic,
                         std::string& error) {
    if (!AppendInitializerArray(
            image, dynamic.preinit_array, dynamic.preinit_array_size,
            false, image.preinitializer_functions, error) ||
        !AppendInitializer(
            image, dynamic.init, image.initializer_functions, error) ||
        !AppendInitializerArray(
            image, dynamic.init_array, dynamic.init_array_size,
            false, image.initializer_functions, error) ||
        !AppendInitializerArray(
            image, dynamic.fini_array, dynamic.fini_array_size,
            true, image.finalizer_functions, error) ||
        !AppendInitializer(
            image, dynamic.fini, image.finalizer_functions, error)) {
        return false;
    }
    return true;
}

bool ReadDynamicSymbol(const LoadedImage& image,
                       const DynamicView& dynamic,
                       const std::uint32_t symbol_index,
                       Symbol& symbol, std::string& error) {
    if (symbol_index == 0) {
        symbol = {};
        return true;
    }
    if (dynamic.symbol_table == 0 ||
        !AddressInLoaderData(
            image, dynamic,
            dynamic.symbol_table +
                static_cast<std::uint64_t>(symbol_index) *
                    sizeof(Symbol),
            sizeof(Symbol))) {
        error = "PS5 relocation symbol is unavailable";
        return false;
    }
    symbol = *reinterpret_cast<const Symbol*>(
        dynamic.symbol_table +
        static_cast<std::uint64_t>(symbol_index) * sizeof(Symbol));
    return true;
}

bool ResolveSymbol(const LoadedImage& image, const DynamicView& dynamic,
                   const LoaderHost& host, const std::uint32_t symbol_index,
                   std::uint64_t& value, std::uint64_t& symbol_size,
                   std::string& import_name,
                   std::uint32_t& import_flags, bool& deferred,
                   std::string& error) {
    import_name.clear();
    import_flags = 0;
    deferred = false;
    Symbol symbol{};
    if (!ReadDynamicSymbol(
            image, dynamic, symbol_index, symbol, error)) {
        return false;
    }
    symbol_size = symbol.size;
    if (symbol_index == 0) {
        value = 0;
        return true;
    }
    if (symbol.section != 0 || symbol.value != 0) {
        if (!AddFits(image.base, symbol.value, value) ||
            !AddressInImage(
                image, value, symbol.size == 0 ? 1 : symbol.size)) {
            error = "PS5 relocation symbol is outside mapped image";
            return false;
        }
        return true;
    }
    const auto symbol_bind = static_cast<std::uint8_t>(symbol.info >> 4u);
    std::string full_name;
    if (!ReadDynamicString(
            image, dynamic, symbol.name, full_name, error)) {
        error = "invalid PS5 import symbol name: " + error;
        return false;
    }
    const auto separator = full_name.find('#');
    import_name = full_name.substr(0, separator);
    const auto symbol_type =
        static_cast<std::uint8_t>(symbol.info & 0x0fu);
    import_flags =
        symbol_type == SymbolTypeObject ? LoaderImportData :
        symbol_type == SymbolTypeTls ? LoaderImportTls :
        LoaderImportFunction;
    if (symbol_bind == SymbolBindWeak) {
        import_flags |= LoaderImportWeak;
    }
    value = 0;
    if (host.bind_import == nullptr) {
        deferred = true;
        return true;
    }
    const auto bind_status = host.bind_import(
        host.context, import_name.c_str(), import_flags,
        &value, &symbol_size);
    if (bind_status == LoaderBindDeferred) {
        deferred = true;
        return true;
    }
    if (bind_status != LoaderBindResolved || value == 0) {
        if (symbol_bind == SymbolBindWeak) {
            value = 0;
            deferred = true;
            return true;
        }
        error = std::string{"cannot bind PS5 import: "} + import_name;
        return false;
    }
    return true;
}

bool ResolveTlsSymbolOffset(const LoadedImage& image,
                            const DynamicView& dynamic,
                            const std::uint32_t symbol_index,
                            const std::int64_t addend,
                            std::uint64_t& value,
                            std::string& error) {
    Symbol symbol{};
    if (!ReadDynamicSymbol(
            image, dynamic, symbol_index, symbol, error)) {
        return false;
    }
    value = symbol.value + static_cast<std::uint64_t>(addend);
    return true;
}

std::size_t RelocationWriteSize(const std::uint32_t type) {
    switch (type) {
    case RelocPc32:
    case RelocPlt32:
    case RelocUnsigned32:
    case RelocSigned32:
    case RelocSize32:
        return sizeof(std::uint32_t);
    default:
        return sizeof(std::uint64_t);
    }
}

bool WriteRelocationValue(const std::uint64_t patch,
                          const std::uint32_t type,
                          const std::uint64_t value,
                          std::string& error) {
    if (type == RelocUnsigned32 || type == RelocSize32) {
        if (value > std::numeric_limits<std::uint32_t>::max()) {
            error = "unsigned 32-bit PS5 relocation overflow";
            return false;
        }
        const auto narrowed = static_cast<std::uint32_t>(value);
        std::memcpy(reinterpret_cast<void*>(patch),
                    &narrowed, sizeof(narrowed));
        return true;
    }
    if (type == RelocPc32 || type == RelocPlt32 ||
        type == RelocSigned32) {
        const auto signed_value = std::bit_cast<std::int64_t>(value);
        if (signed_value < std::numeric_limits<std::int32_t>::min() ||
            signed_value > std::numeric_limits<std::int32_t>::max()) {
            error = "signed 32-bit PS5 relocation overflow";
            return false;
        }
        const auto narrowed = static_cast<std::int32_t>(signed_value);
        std::memcpy(reinterpret_cast<void*>(patch),
                    &narrowed, sizeof(narrowed));
        return true;
    }
    std::memcpy(reinterpret_cast<void*>(patch), &value, sizeof(value));
    return true;
}

bool ApplyRelaTable(LoadedImage& image, const DynamicView& dynamic,
                    const LoaderHost& host, const std::uint64_t table,
                    const std::uint64_t table_size, std::string& error) {
    if (table_size == 0) {
        return true;
    }
    if (!AddressInLoaderData(image, dynamic, table, table_size) ||
        table_size % sizeof(Rela) != 0) {
        error = "PS5 relocation table is outside mapped image";
        return false;
    }
    const auto* records = reinterpret_cast<const Rela*>(table);
    const auto count = table_size / sizeof(Rela);
    for (std::uint64_t index = 0; index < count; ++index) {
        const auto& record = records[index];
        const auto type = static_cast<std::uint32_t>(record.info);
        if (type == RelocNone) {
            continue;
        }
        const auto symbol_index =
            static_cast<std::uint32_t>(record.info >> 32u);
        std::uint64_t patch{};
        const auto write_size = RelocationWriteSize(type);
        if (!AddFits(image.base, record.offset, patch) ||
            !AddressInImage(image, patch, write_size)) {
            error = "PS5 relocation target is outside mapped image";
            return false;
        }
        std::uint64_t value{};
        std::uint64_t symbol_size{};
        std::string import_name;
        std::uint32_t import_flags{};
        bool deferred{};
        switch (type) {
        case RelocRelative:
        case RelocRelative64:
            value = image.base + static_cast<std::uint64_t>(record.addend);
            break;
        case RelocGlobDat:
        case RelocJumpSlot:
        case Reloc64:
        case RelocUnsigned32:
        case RelocSigned32:
        case RelocPc32:
        case RelocPlt32:
        case RelocPc64:
        case RelocSize32:
        case RelocSize64:
            if (!ResolveSymbol(
                    image, dynamic, host, symbol_index,
                    value, symbol_size, import_name,
                    import_flags, deferred, error)) {
                return false;
            }
            if (type == RelocSize32 || type == RelocSize64) {
                value = symbol_size;
            }
            if (type != RelocGlobDat && type != RelocJumpSlot) {
                value += static_cast<std::uint64_t>(record.addend);
            }
            if (type == RelocPc32 || type == RelocPlt32 ||
                type == RelocPc64) {
                value -= patch;
            }
            break;
        case RelocDtpMod64:
            if (image.tls_memory_size == 0 ||
                image.tls_module_id == 0) {
                error = "PS5 DTPMOD64 relocation references an image without PT_TLS";
                return false;
            }
            value = image.tls_module_id;
            break;
        case RelocDtpOff64:
        case RelocTpOff64:
            if (image.tls_memory_size == 0 ||
                !ResolveTlsSymbolOffset(
                    image, dynamic, symbol_index, record.addend,
                    value, error)) {
                if (error.empty()) {
                    error = "PS5 TLS relocation references an image without PT_TLS";
                }
                return false;
            }
            if (type == RelocTpOff64) {
                value -= image.tls_static_offset;
            }
            break;
        default:
            error = "unsupported PS5 relocation type " +
                    std::to_string(type);
            return false;
        }
        if (!WriteRelocationValue(patch, type, value, error)) {
            return false;
        }
        if (deferred && !import_name.empty()) {
            image.deferred_imports.push_back(
                {.name = std::move(import_name),
                 .patch = patch,
                 .addend = record.addend,
                 .symbol_size = symbol_size,
                 .type = type,
                 .flags = import_flags});
        }
    }
    return true;
}

bool ApplySectionRelaFallback(
    const FileImage& file, const ElfHeader& header,
    LoadedImage& image, const LoaderHost& host,
    std::string& error) {
    if (header.section_offset == 0 || header.section_count == 0 ||
        header.section_entry_size < sizeof(SectionHeader)) {
        return true;
    }
    for (std::uint32_t section_index = 0;
         section_index < header.section_count; ++section_index) {
        SectionHeader relocation_section{};
        if (!ReadSectionHeader(
                file, header, section_index, relocation_section) ||
            relocation_section.type != SectionTypeRela ||
            relocation_section.size == 0) {
            continue;
        }
        if (relocation_section.entry_size != sizeof(Rela) ||
            relocation_section.size % sizeof(Rela) != 0 ||
            relocation_section.link >= header.section_count) {
            if (file.self) {
                continue;
            }
            error = "invalid PS5 section relocation table";
            return false;
        }
        SectionHeader symbol_section{};
        SectionHeader string_section{};
        std::span<const std::uint8_t> relocations;
        std::span<const std::uint8_t> symbols;
        std::span<const std::uint8_t> strings;
        if (!ReadSectionHeader(
                file, header, relocation_section.link,
                symbol_section) ||
            symbol_section.entry_size != sizeof(Symbol) ||
            symbol_section.size % sizeof(Symbol) != 0 ||
            symbol_section.link >= header.section_count ||
            !ReadSectionHeader(
                file, header, symbol_section.link,
                string_section) ||
            !ElfRelativeSpan(
                file, relocation_section.offset,
                relocation_section.size, relocations) ||
            !ElfRelativeSpan(
                file, symbol_section.offset,
                symbol_section.size, symbols) ||
            !ElfRelativeSpan(
                file, string_section.offset,
                string_section.size, strings)) {
            if (file.self) {
                continue;
            }
            error = "truncated PS5 section relocation metadata";
            return false;
        }
        DynamicView section_view{};
        section_view.metadata_begin =
            reinterpret_cast<std::uint64_t>(file.bytes.data());
        section_view.metadata_size = file.bytes.size();
        section_view.string_table =
            reinterpret_cast<std::uint64_t>(strings.data());
        section_view.string_size = strings.size();
        section_view.symbol_table =
            reinterpret_cast<std::uint64_t>(symbols.data());
        section_view.symbol_table_size = symbols.size();
        section_view.symbol_entry_size = sizeof(Symbol);
        section_view.rela_entry_size = sizeof(Rela);
        if (!ApplyRelaTable(
                image, section_view, host,
                reinterpret_cast<std::uint64_t>(relocations.data()),
                relocations.size(), error)) {
            error = "PS5 section relocation fallback failed: " + error;
            return false;
        }
    }
    return true;
}

bool ProtectAndRegister(const LoaderHost& host, LoadedImage& image,
                        const std::vector<ProgramHeader>& programs,
                        std::string& error) {
    const auto page_count =
        static_cast<std::size_t>(image.mapped_size / GuestPageSize);
    std::vector<std::uint32_t> pages(page_count);
    for (const auto& program : programs) {
        if (program.memory_size == 0 ||
            (program.type != PtLoad && program.type != PtSceRelro)) {
            continue;
        }
        std::uint64_t end{};
        if (!AddFits(program.virtual_address, program.memory_size, end)) {
            error = "PT_LOAD address overflow";
            return false;
        }
        const auto first = AlignDown(program.virtual_address, GuestPageSize) /
                           GuestPageSize;
        std::uint64_t aligned_end{};
        if (!AlignUp(end, GuestPageSize, aligned_end) ||
            aligned_end > image.mapped_size) {
            error = "PT_LOAD exceeds mapped image";
            return false;
        }
        const auto last = aligned_end / GuestPageSize;
        for (std::uint64_t page = first; page < last; ++page) {
            pages[static_cast<std::size_t>(page)] |=
                LoaderFlags(program.flags);
        }
    }

    for (std::size_t first = 0; first < pages.size();) {
        if (pages[first] == 0) {
            ++first;
            continue;
        }
        std::size_t last = first + 1;
        while (last < pages.size() && pages[last] == pages[first]) {
            ++last;
        }
        const std::uint64_t address =
            image.base + first * GuestPageSize;
        const std::uint64_t byte_count =
            (last - first) * GuestPageSize;
        if (mprotect(reinterpret_cast<void*>(address),
                     static_cast<std::size_t>(byte_count),
                     NativeProtection(pages[first])) != 0) {
            error = "mprotect failed for PS5 image";
            return false;
        }
        if (host.register_mapping == nullptr ||
            host.register_mapping(host.context, address, byte_count,
                                  pages[first], "ps5-eboot") != 0) {
            error = "runtime rejected PS5 image mapping";
            return false;
        }
        image.registered_ranges.push_back(
            {address, byte_count, pages[first]});
        first = last;
    }
    return true;
}

void* AllocateImage(const std::uint64_t mapped_size,
                    const std::uint64_t preferred_base) {
    const auto first_base =
        preferred_base != 0 ? preferred_base : FirstImageBase;
    for (std::uint64_t attempt = 0; attempt != 256; ++attempt) {
        const std::uint64_t desired = first_base +
                                      attempt * ImageBaseStep;
        if (desired >= GuestAddressLimit ||
            mapped_size > GuestAddressLimit - desired) {
            break;
        }
        void* const mapping = mmap(
            reinterpret_cast<void*>(desired),
            static_cast<std::size_t>(mapped_size),
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) {
            continue;
        }
        const auto address = reinterpret_cast<std::uint64_t>(mapping);
        if ((address & (GuestPageSize - 1u)) == 0 &&
            address < GuestAddressLimit &&
            mapped_size <= GuestAddressLimit - address) {
            return mapping;
        }
        munmap(mapping, static_cast<std::size_t>(mapped_size));
    }
    return MAP_FAILED;
}

}

bool LoadNextGenExecutable(
    const char* const path, const LoaderHost& host, LoadedImage& image,
    std::string& error, const LoaderOptions options) {
    image = {};
    error.clear();
    if (path == nullptr || path[0] == '\0') {
        error = "empty PS5 executable path";
        return false;
    }

    FileImage file{};
    if (!ReadFile(path, file, error)) {
        return false;
    }
    ElfHeader header{};
    if (!ReadObject(file.bytes, file.elf_offset, header)) {
        error = "truncated ELF header";
        return false;
    }
    if (std::memcmp(header.ident, "\x7f" "ELF", 4) != 0 ||
        header.ident[4] != 2 || header.ident[5] != 1 ||
        header.ident[6] != 1 || header.machine != EmX86_64 ||
        header.version != 1 ||
        (header.type != EtDyn && header.type != EtSceDynExec &&
         header.type != EtSceDynamic) ||
        header.program_entry_size != sizeof(ProgramHeader) ||
        header.program_count == 0 || header.program_count > 4096) {
        error = "unsupported PS5 ELF identity";
        return false;
    }

    std::uint64_t program_offset{};
    std::uint64_t program_bytes{};
    if (!AddFits(file.elf_offset, header.program_offset, program_offset) ||
        !AddFits(program_offset,
                 static_cast<std::uint64_t>(header.program_count) *
                     sizeof(ProgramHeader),
                 program_bytes) ||
        program_bytes > file.bytes.size()) {
        error = "truncated ELF program table";
        return false;
    }
    std::vector<ProgramHeader> programs(header.program_count);
    std::memcpy(programs.data(), file.bytes.data() + program_offset,
                programs.size() * sizeof(ProgramHeader));

    std::uint64_t image_size{};
    for (const auto& program : programs) {
        if (program.memory_size == 0 ||
            (program.type != PtLoad && program.type != PtSceRelro)) {
            continue;
        }
        if (program.file_size > program.memory_size) {
            error = "PT_LOAD file size exceeds memory size";
            return false;
        }
        std::uint64_t end{};
        if (!AddFits(program.virtual_address, program.memory_size, end) ||
            !AlignUp(end, GuestPageSize, end)) {
            error = "PT_LOAD size overflow";
            return false;
        }
        image_size = std::max(image_size, end);
    }
    if (image_size == 0 || image_size >= GuestAddressLimit) {
        error = "PS5 ELF has no loadable image";
        return false;
    }

    void* const reservation = AllocateImage(
        image_size, options.preferred_image_base);
    if (reservation == MAP_FAILED) {
        error = "cannot reserve low PS5 guest address space";
        return false;
    }
    image.base = reinterpret_cast<std::uint64_t>(reservation);
    image.mapped_size = image_size;
    image.entry = image.base + header.entry;
    image.next_gen = header.ident[8] == 2;
    image.self_container = file.self;

    for (std::size_t index = 0; index < programs.size(); ++index) {
        const auto& program = programs[index];
        if (program.type == PtTls) {
            const auto alignment =
                program.alignment == 0 ? 1 : program.alignment;
            std::uint64_t tls_address{};
            if (options.tls_module_id == 0 ||
                image.tls_memory_size != 0 ||
                program.file_size > program.memory_size ||
                !IsPowerOfTwo(alignment) ||
                !AddFits(image.base, program.virtual_address, tls_address) ||
                !AddressInImage(
                    image, tls_address,
                    program.file_size == 0 ? 1 : program.file_size) ||
                !CalculateStaticTlsOffset(
                    options.previous_tls_static_offset,
                    program.memory_size, alignment,
                    program.virtual_address & (alignment - 1u),
                    image.tls_static_offset)) {
                error = "invalid or duplicate PS5 PT_TLS segment";
                UnloadNextGenExecutable(host, image);
                return false;
            }
            image.tls_image = tls_address;
            image.tls_file_size = program.file_size;
            image.tls_memory_size = program.memory_size;
            image.tls_alignment = alignment;
            image.tls_alignment_bias =
                program.virtual_address & (alignment - 1u);
            image.tls_module_id = options.tls_module_id;
        } else if (program.type == PtSceProcParam) {
            image.process_param = image.base + program.virtual_address;
        }
        if (program.file_size == 0 ||
            (program.type != PtLoad && program.type != PtSceRelro)) {
            continue;
        }
        const std::uint64_t destination =
            image.base + program.virtual_address;
        if (!AddressInImage(image, destination, program.file_size) ||
            !ReadSegmentBytes(file, programs, index,
                              reinterpret_cast<void*>(destination),
                              program.file_size, error)) {
            UnloadNextGenExecutable(host, image);
            return false;
        }
    }

    std::vector<std::uint8_t> dynamic_entries;
    std::vector<std::uint8_t> dynamic_metadata;
    for (std::size_t index = 0; index < programs.size(); ++index) {
        if (programs[index].type == PtDynamic) {
            const auto& dynamic_program = programs[index];
            std::uint64_t dynamic_address{};
            const bool mapped =
                AddFits(image.base, dynamic_program.virtual_address,
                        dynamic_address) &&
                AddressInImage(image, dynamic_address,
                               dynamic_program.file_size);
            if (mapped) {
                if (dynamic_program.file_size >
                    std::numeric_limits<std::size_t>::max()) {
                    error = "PS5 dynamic segment is too large";
                    UnloadNextGenExecutable(host, image);
                    return false;
                }
                dynamic_entries.resize(static_cast<std::size_t>(
                    dynamic_program.file_size));
                if (!dynamic_entries.empty()) {
                    std::memcpy(
                        dynamic_entries.data(),
                        reinterpret_cast<const void*>(dynamic_address),
                        dynamic_entries.size());
                }
            } else if (!ReadAuxiliarySegment(
                           file, programs, index, dynamic_entries,
                           error)) {
                UnloadNextGenExecutable(host, image);
                return false;
            }
        }
        if (programs[index].type == PtSceDynlibData &&
            !ReadAuxiliarySegment(file, programs, index,
                                  dynamic_metadata, error)) {
            UnloadNextGenExecutable(host, image);
            return false;
        }
    }

    DynamicView dynamic{};
    if (!ParseDynamic(image, dynamic_entries, dynamic_metadata,
                       dynamic, error) ||
        !CollectDynamicMetadata(image, dynamic, error) ||
        !ApplyRelaTable(image, dynamic, host, dynamic.rela,
                         dynamic.rela_size, error) ||
        !ApplyRelaTable(image, dynamic, host, dynamic.jump_rela,
                         dynamic.jump_rela_size, error) ||
        ((dynamic.rela_size == 0 && dynamic.jump_rela_size == 0) &&
         !ApplySectionRelaFallback(
             file, header, image, host, error)) ||
        !CollectInitializers(image, dynamic, error) ||
        !AddressInImage(image, image.entry, 1) ||
        !ProtectAndRegister(host, image, programs, error)) {
        UnloadNextGenExecutable(host, image);
        return false;
    }
    CollectSectionExports(file, header, image);
    return true;
}

bool RebindNextGenImports(
    const LoaderHost& host, LoadedImage& image, std::string& error) {
    if (image.deferred_imports.empty()) {
        return true;
    }
    if (host.bind_import == nullptr) {
        error = "PS5 import rebinding has no symbol resolver";
        return false;
    }

    std::vector<ImportedRelocation> remaining;
    remaining.reserve(image.deferred_imports.size());
    for (auto& relocation : image.deferred_imports) {
        std::uint64_t symbol_value{};
        auto symbol_size = relocation.symbol_size;
        const auto status = host.bind_import(
            host.context, relocation.name.c_str(),
            relocation.flags | LoaderImportRebind,
            &symbol_value, &symbol_size);
        if (status == LoaderBindDeferred) {
            remaining.push_back(std::move(relocation));
            continue;
        }
        if (status != LoaderBindResolved || symbol_value == 0) {
            error = "cannot rebind PS5 import: " + relocation.name;
            return false;
        }

        auto value = symbol_value;
        if (relocation.type == RelocSize32 ||
            relocation.type == RelocSize64) {
            value = symbol_size;
        }
        if (relocation.type != RelocGlobDat &&
            relocation.type != RelocJumpSlot) {
            value += static_cast<std::uint64_t>(relocation.addend);
        }
        if (relocation.type == RelocPc32 ||
            relocation.type == RelocPlt32 ||
            relocation.type == RelocPc64) {
            value -= relocation.patch;
        }

        const auto write_size = RelocationWriteSize(relocation.type);
        const auto range = std::ranges::find_if(
            image.registered_ranges,
            [&](const LoadedRange& candidate) {
                return relocation.patch >= candidate.address &&
                       write_size <= candidate.byte_count &&
                       relocation.patch - candidate.address <=
                           candidate.byte_count - write_size;
            });
        if (range == image.registered_ranges.end()) {
            error = "PS5 deferred relocation target is not registered";
            return false;
        }
        const auto host_page_size =
            static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
        if (host_page_size == 0 ||
            !std::has_single_bit(host_page_size)) {
            error = "invalid host page size during PS5 rebinding";
            return false;
        }
        const auto page_begin =
            AlignDown(relocation.patch, host_page_size);
        std::uint64_t patch_end{};
        std::uint64_t page_end{};
        if (!AddFits(relocation.patch, write_size, patch_end) ||
            !AlignUp(patch_end, host_page_size, page_end)) {
            error = "PS5 deferred relocation page overflow";
            return false;
        }
        const auto page_bytes =
            static_cast<std::size_t>(page_end - page_begin);
        if (mprotect(
                reinterpret_cast<void*>(page_begin), page_bytes,
                PROT_READ | PROT_WRITE) != 0) {
            error = "cannot unlock PS5 relocation page";
            return false;
        }
        const auto wrote = WriteRelocationValue(
            relocation.patch, relocation.type, value, error);
        const auto restored =
            mprotect(reinterpret_cast<void*>(page_begin), page_bytes,
                     NativeProtection(range->protection)) == 0;
        if (!wrote || !restored) {
            if (error.empty()) {
                error = "cannot restore PS5 relocation page protection";
            }
            return false;
        }
    }
    image.deferred_imports = std::move(remaining);
    return true;
}

void UnloadNextGenExecutable(
    const LoaderHost& host, LoadedImage& image) noexcept {
    if (host.unregister_mapping != nullptr) {
        for (const auto& range : image.registered_ranges) {
            (void)host.unregister_mapping(
                host.context, range.address, range.byte_count);
        }
    }
    if (image.base != 0 && image.mapped_size != 0) {
        munmap(reinterpret_cast<void*>(image.base),
               static_cast<std::size_t>(image.mapped_size));
    }
    image = {};
}

}
