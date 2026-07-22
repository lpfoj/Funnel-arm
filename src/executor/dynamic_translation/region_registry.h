// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/code_witness.h"
#include "executor/dynamic_translation/edge_portal.h"
#include "executor/dynamic_translation/region_plan.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace Lsx4::Translation {

enum class ExecutionGrade : std::uint8_t {
    Semantic,
    Native,
    Optimized,
};

struct RegionPayload {
    RegionPlan plan{};
    CodeWitness source_identity{};
    std::uintptr_t entry_address{};
    std::uintptr_t code_address{};
    std::size_t code_bytes{};
    ExecutionGrade grade{ExecutionGrade::Semantic};
    std::vector<std::shared_ptr<ExitPortal>> outgoing_edges{};
};

class PublishedRegion final {
public:
    PublishedRegion(std::uint64_t publication, RegionPayload payload);

    [[nodiscard]] std::uint64_t Publication() const noexcept;
    [[nodiscard]] std::uint64_t GuestAddress() const noexcept;
    [[nodiscard]] const RegionPayload& Payload() const noexcept;

private:
    const std::uint64_t publication_;
    const RegionPayload payload_;
};

class RegionRegistry {
public:
    [[nodiscard]] std::shared_ptr<const PublishedRegion> Find(
        std::uint64_t guest_address) const noexcept;
    [[nodiscard]] std::shared_ptr<const PublishedRegion> Install(RegionPayload payload);
    [[nodiscard]] std::shared_ptr<const PublishedRegion> Remove(
        std::uint64_t guest_address) noexcept;
    [[nodiscard]] std::vector<std::shared_ptr<const PublishedRegion>> Snapshot() const;

    void Clear() noexcept;
    [[nodiscard]] std::uint64_t ChangeSequence() const noexcept;
    [[nodiscard]] std::size_t ActiveCount() const noexcept;

private:
    struct RegistryCell {
        explicit RegistryCell(std::uint64_t guest_address) noexcept
            : guest_address{guest_address} {}

        const std::uint64_t guest_address;
        std::shared_ptr<const PublishedRegion> current{};
    };

    struct Partition {
        mutable std::shared_mutex mutex{};
        std::unordered_map<std::uint64_t, std::unique_ptr<RegistryCell>> cells{};
    };

    static constexpr std::size_t kPartitionCount = 61;

    [[nodiscard]] static std::size_t PartitionIndex(std::uint64_t guest_address) noexcept;
    [[nodiscard]] RegistryCell* FindCell(std::uint64_t guest_address) const noexcept;

    mutable std::array<Partition, kPartitionCount> partitions_{};
    std::atomic<std::uint64_t> next_publication_{1};
    std::atomic<std::uint64_t> change_sequence_{1};
    std::atomic<std::size_t> active_count_{};
};

} // namespace Lsx4::Translation
