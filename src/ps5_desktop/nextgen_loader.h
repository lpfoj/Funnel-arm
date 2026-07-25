// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Funnel::Ps5Desktop {

inline constexpr std::uint32_t NextGenLoaderAbiVersion = 2;

enum LoaderProtection : std::uint32_t {
    LoaderRead = 1u << 0,
    LoaderWrite = 1u << 1,
    LoaderExecute = 1u << 2,
};

enum LoaderImportFlag : std::uint32_t {
    LoaderImportFunction = 1u << 0,
    LoaderImportData = 1u << 1,
    LoaderImportTls = 1u << 2,
    LoaderImportWeak = 1u << 3,
    LoaderImportRebind = 1u << 4,
};

enum LoaderBindStatus : int {
    LoaderBindResolved = 0,
    LoaderBindDeferred = 1,
    LoaderBindError = -1,
};

struct LoaderHost {
    void* context{};
    int (*register_mapping)(void* context, std::uint64_t address,
                            std::uint64_t byte_count,
                            std::uint32_t protection,
                            const char* label){};
    int (*unregister_mapping)(void* context, std::uint64_t address,
                              std::uint64_t byte_count){};
    int (*bind_import)(void* context, const char* symbol,
                       std::uint32_t flags, std::uint64_t* value,
                       std::uint64_t* symbol_size){};
};

struct LoadedRange {
    std::uint64_t address{};
    std::uint64_t byte_count{};
    std::uint32_t protection{};
};

struct LoaderOptions {
    std::uint64_t tls_module_id{1};
    std::uint64_t previous_tls_static_offset{};
    std::uint64_t preferred_image_base{};
};

struct LoadedSymbol {
    std::string name;
    std::uint64_t address{};
    std::uint64_t size{};
};

struct ImportedRelocation {
    std::string name;
    std::uint64_t patch{};
    std::int64_t addend{};
    std::uint64_t symbol_size{};
    std::uint32_t type{};
    std::uint32_t flags{};
};

struct LoadedImage {
    std::uint64_t base{};
    std::uint64_t entry{};
    std::uint64_t mapped_size{};
    std::uint64_t tls_image{};
    std::uint64_t tls_file_size{};
    std::uint64_t tls_memory_size{};
    std::uint64_t tls_alignment{1};
    std::uint64_t tls_alignment_bias{};
    std::uint64_t tls_static_offset{};
    std::uint64_t tls_module_id{};
    std::uint64_t process_param{};
    bool next_gen{};
    bool self_container{};
    std::vector<std::uint64_t> preinitializer_functions;
    std::vector<std::uint64_t> initializer_functions;
    std::vector<std::uint64_t> finalizer_functions;
    std::vector<std::string> dependencies;
    std::vector<LoadedSymbol> exported_symbols;
    std::vector<ImportedRelocation> deferred_imports;
    std::vector<LoadedRange> registered_ranges;
};

[[nodiscard]] bool LoadNextGenExecutable(
    const char* path, const LoaderHost& host, LoadedImage& image,
    std::string& error, LoaderOptions options = {});
[[nodiscard]] bool RebindNextGenImports(
    const LoaderHost& host, LoadedImage& image, std::string& error);
void UnloadNextGenExecutable(
    const LoaderHost& host, LoadedImage& image) noexcept;

}
