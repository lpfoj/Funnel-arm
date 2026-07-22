// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/code_memory.h"
#include "executor/dynamic_translation/code_witness.h"
#include "executor/dynamic_translation/edge_portal.h"
#include "executor/dynamic_translation/region_registry.h"
#include "executor/dynamic_translation/scalar_ir_builder.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>

namespace Lsx4::Translation {

enum class AcquireStatus : std::uint8_t {
    Ready,
    NeedsSemanticFallback,
    GuestReadFailure,
    InvalidRegion,
    WitnessFailure,
    NativeCompileFailure,
};

struct AcquireResult {
    AcquireStatus status{AcquireStatus::InvalidRegion};
    std::shared_ptr<const PublishedRegion> region{};
    RegionPlan fallback_plan{};
};

struct TranslationServiceCounters {
    std::uint64_t lookup_hits{};
    std::uint64_t compiled_regions{};
    std::uint64_t semantic_regions{};
    std::uint64_t invalidated_regions{};
    std::uint64_t compile_failures{};
};

class TranslationService {
public:
    explicit TranslationService(std::size_t arena_slab_bytes = 2 * 1024 * 1024);
    ~TranslationService();

    TranslationService(const TranslationService&) = delete;
    TranslationService& operator=(const TranslationService&) = delete;

    [[nodiscard]] AcquireResult Acquire(std::uint64_t guest_address,
                                        InstructionFetcher fetch,
                                        void* fetch_context,
                                        const ReadGuestBytes& reader,
                                        std::size_t instruction_limit = 64);
    void NotifyMutation(std::uint64_t guest_address, std::size_t byte_count);
    void Clear() noexcept;

    [[nodiscard]] TranslationServiceCounters Counters() const noexcept;
    [[nodiscard]] std::uint64_t RuntimeEpoch() const noexcept;
    [[nodiscard]] std::size_t RegionCount() const noexcept;
    [[nodiscard]] std::size_t NativeBytes() const noexcept;

private:
    static constexpr std::size_t AdmissionShardCount = 67;

    [[nodiscard]] static std::size_t AdmissionShard(std::uint64_t address) noexcept;
    [[nodiscard]] bool RegionIsCurrent(const PublishedRegion& region,
                                       const ReadGuestBytes& reader) const;
    void Retire(const std::shared_ptr<const PublishedRegion>& region) noexcept;
    [[nodiscard]] std::vector<std::shared_ptr<ExitPortal>> BuildEdges(
        const RegionPlan& plan);

    ExecutableArena arena_;
    MutationLedger mutations_{};
    RegionRegistry regions_{};
    EdgeDirectory edges_{};
    mutable std::shared_mutex publication_gate_{};
    std::array<std::mutex, AdmissionShardCount> admission_{};
    std::atomic<std::uint64_t> runtime_epoch_{1};
    std::atomic<std::uint64_t> lookup_hits_{};
    std::atomic<std::uint64_t> compiled_regions_{};
    std::atomic<std::uint64_t> semantic_regions_{};
    std::atomic<std::uint64_t> invalidated_regions_{};
    std::atomic<std::uint64_t> compile_failures_{};
};

} // namespace Lsx4::Translation
