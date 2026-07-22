// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/region_registry.h"

#include <utility>

namespace Lsx4::Translation {

PublishedRegion::PublishedRegion(const std::uint64_t publication, RegionPayload payload)
    : publication_{publication}, payload_{std::move(payload)} {}

std::uint64_t PublishedRegion::Publication() const noexcept {
    return publication_;
}

std::uint64_t PublishedRegion::GuestAddress() const noexcept {
    return payload_.plan.first_address;
}

const RegionPayload& PublishedRegion::Payload() const noexcept {
    return payload_;
}

std::size_t RegionRegistry::PartitionIndex(std::uint64_t guest_address) noexcept {
    guest_address ^= guest_address >> 29;
    guest_address *= 0x9fb21c651e98df25ull;
    guest_address ^= guest_address >> 32;
    return static_cast<std::size_t>(guest_address % kPartitionCount);
}

RegionRegistry::RegistryCell* RegionRegistry::FindCell(
    const std::uint64_t guest_address) const noexcept {
    Partition& partition = partitions_[PartitionIndex(guest_address)];
    std::shared_lock lock{partition.mutex};
    const auto found = partition.cells.find(guest_address);
    return found == partition.cells.end() ? nullptr : found->second.get();
}

std::shared_ptr<const PublishedRegion> RegionRegistry::Find(
    const std::uint64_t guest_address) const noexcept {
    RegistryCell* const cell = FindCell(guest_address);
    return cell ? std::atomic_load_explicit(&cell->current, std::memory_order_acquire)
                : nullptr;
}

std::shared_ptr<const PublishedRegion> RegionRegistry::Install(RegionPayload payload) {
    const std::uint64_t guest_address = payload.plan.first_address;
    if (!payload.plan.IsExecutable() || payload.source_identity.guest_address != guest_address ||
        payload.entry_address == 0) {
        return {};
    }

    Partition& partition = partitions_[PartitionIndex(guest_address)];
    RegistryCell* cell = nullptr;
    {
        std::unique_lock lock{partition.mutex};
        auto [position, inserted] = partition.cells.try_emplace(guest_address);
        if (inserted) {
            position->second = std::make_unique<RegistryCell>(guest_address);
        }
        cell = position->second.get();
    }

    const std::uint64_t publication =
        next_publication_.fetch_add(1, std::memory_order_relaxed);
    auto replacement = std::make_shared<const PublishedRegion>(publication, std::move(payload));
    auto published = replacement;
    auto displaced = std::atomic_exchange_explicit(
        &cell->current, std::move(replacement), std::memory_order_acq_rel);
    if (!displaced) {
        active_count_.fetch_add(1, std::memory_order_relaxed);
    }
    change_sequence_.fetch_add(1, std::memory_order_release);
    return published;
}

std::shared_ptr<const PublishedRegion> RegionRegistry::Remove(
    const std::uint64_t guest_address) noexcept {
    RegistryCell* const cell = FindCell(guest_address);
    if (!cell) {
        return {};
    }
    auto displaced = std::atomic_exchange_explicit(
        &cell->current, std::shared_ptr<const PublishedRegion>{},
        std::memory_order_acq_rel);
    if (displaced) {
        active_count_.fetch_sub(1, std::memory_order_relaxed);
        change_sequence_.fetch_add(1, std::memory_order_release);
    }
    return displaced;
}

std::vector<std::shared_ptr<const PublishedRegion>> RegionRegistry::Snapshot() const {
    std::vector<std::shared_ptr<const PublishedRegion>> result{};
    result.reserve(ActiveCount());
    for (const Partition& partition : partitions_) {
        std::shared_lock lock{partition.mutex};
        for (const auto& [guest_address, cell] : partition.cells) {
            (void)guest_address;
            if (auto region =
                    std::atomic_load_explicit(&cell->current, std::memory_order_acquire)) {
                result.push_back(std::move(region));
            }
        }
    }
    return result;
}

void RegionRegistry::Clear() noexcept {
    bool changed = false;
    for (Partition& partition : partitions_) {
        std::shared_lock lock{partition.mutex};
        for (auto& [guest_address, cell] : partition.cells) {
            (void)guest_address;
            if (std::atomic_exchange_explicit(
                    &cell->current, std::shared_ptr<const PublishedRegion>{},
                    std::memory_order_acq_rel)) {
                changed = true;
                active_count_.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }
    if (changed) {
        change_sequence_.fetch_add(1, std::memory_order_release);
    }
}

std::uint64_t RegionRegistry::ChangeSequence() const noexcept {
    return change_sequence_.load(std::memory_order_acquire);
}

std::size_t RegionRegistry::ActiveCount() const noexcept {
    return active_count_.load(std::memory_order_relaxed);
}

} // namespace Lsx4::Translation
