// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace Lsx4::Translation {

struct EntryRequest {
    std::array<std::uint64_t, 8> arguments{};
    std::uint32_t argument_count{};
    std::uint8_t stack_argument_mode{};
    bool permit_return_sentinel{true};
    std::uint64_t thread_segment_origin{};
    std::uint64_t supplied_stack_base{};
    std::uint64_t supplied_stack_bytes{};
};

struct TranslationCounters {
    std::uint64_t decoded_regions{};
    std::uint64_t native_regions{};
    std::uint64_t semantic_regions{};
    std::uint64_t unsupported_regions{};
    std::uint64_t memory_cache_hits{};
    std::uint64_t stored_ir_enabled{};
    std::uint64_t stored_ir_hits{};
    std::uint64_t stored_ir_misses{};
    std::uint64_t stored_ir_loaded{};
    std::uint64_t stored_ir_written{};
    std::uint64_t stored_native_hits{};
    std::uint64_t stored_native_segments{};
    std::uint64_t stored_native_captured{};
    std::uint64_t stored_native_written{};
    std::uint64_t stored_native_fallbacks{};
    std::uint64_t stored_native_rejected{};
};

struct HleLeafOutcome {
    std::uint64_t value{};
    std::uint64_t accepted{};
};

[[nodiscard]] std::uint64_t ExecuteGuest(std::uint64_t address,
                                         const EntryRequest& request);
void ConfigureArtifactStore(const std::string& directory, const std::string& title,
                            std::uint64_t executable_identity, bool enabled);
[[nodiscard]] TranslationCounters ReadTranslationCounters() noexcept;
[[nodiscard]] std::string DescribeTranslationRuntime();
[[nodiscard]] std::string ExerciseTranslationRuntime();

} // namespace Lsx4::Translation
