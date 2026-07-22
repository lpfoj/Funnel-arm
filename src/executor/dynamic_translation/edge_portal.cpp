// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/edge_portal.h"

namespace Lsx4::Translation {

ExitPortal::ExitPortal(const std::uint64_t guest_destination,
                       const EdgePolicy policy) noexcept
    : guest_destination_{guest_destination}, policy_{policy} {}

std::uint64_t ExitPortal::GuestDestination() const noexcept {
    return guest_destination_;
}

EdgePolicy ExitPortal::Policy() const noexcept {
    return policy_;
}

ResolvedEdge ExitPortal::Resolve(const std::uint64_t required_revision,
                                 const std::uint64_t required_epoch) const noexcept {
    if (policy_ != EdgePolicy::Chainable) {
        return {};
    }
    for (unsigned attempt = 0; attempt < 3; ++attempt) {
        const std::uint64_t before = sequence_.load(std::memory_order_acquire);
        if ((before & 1u) != 0) {
            continue;
        }
        const std::uintptr_t host = host_address_.load(std::memory_order_relaxed);
        const std::uint64_t revision = guest_revision_.load(std::memory_order_relaxed);
        const std::uint64_t epoch = runtime_epoch_.load(std::memory_order_relaxed);
        const std::uint64_t after = sequence_.load(std::memory_order_acquire);
        if (before != after) {
            continue;
        }
        if (host == 0 || revision != required_revision || epoch != required_epoch) {
            return {};
        }
        return {host, revision, epoch};
    }
    return {};
}

void ExitPortal::Replace(const std::uintptr_t host_address,
                         const std::uint64_t guest_revision,
                         const std::uint64_t runtime_epoch) noexcept {
    sequence_.fetch_add(1, std::memory_order_acq_rel);
    host_address_.store(host_address, std::memory_order_relaxed);
    guest_revision_.store(guest_revision, std::memory_order_relaxed);
    runtime_epoch_.store(runtime_epoch, std::memory_order_relaxed);
    sequence_.fetch_add(1, std::memory_order_release);
}

void EdgeDirectory::Attach(ExitPortal& portal) {
    std::lock_guard lock{mutex_};
    listeners_[portal.GuestDestination()].insert(&portal);
}

void EdgeDirectory::Detach(ExitPortal& portal) noexcept {
    std::lock_guard lock{mutex_};
    const auto found = listeners_.find(portal.GuestDestination());
    if (found == listeners_.end()) {
        return;
    }
    found->second.erase(&portal);
    if (found->second.empty()) {
        listeners_.erase(found);
    }
    portal.Replace(0, 0, 0);
}

void EdgeDirectory::Announce(const std::uint64_t guest_address,
                             const std::uintptr_t host_address,
                             const std::uint64_t guest_revision,
                             const std::uint64_t runtime_epoch) {
    std::lock_guard lock{mutex_};
    const auto found = listeners_.find(guest_address);
    if (found == listeners_.end()) {
        return;
    }
    for (ExitPortal* const portal : found->second) {
        portal->Replace(host_address, guest_revision, runtime_epoch);
    }
}

void EdgeDirectory::Revoke(const std::uint64_t guest_address) noexcept {
    std::lock_guard lock{mutex_};
    const auto found = listeners_.find(guest_address);
    if (found == listeners_.end()) {
        return;
    }
    for (ExitPortal* const portal : found->second) {
        portal->Replace(0, 0, 0);
    }
}

void EdgeDirectory::RevokeAll() noexcept {
    std::lock_guard lock{mutex_};
    for (auto& [guest_address, portals] : listeners_) {
        (void)guest_address;
        for (ExitPortal* const portal : portals) {
            portal->Replace(0, 0, 0);
        }
    }
}

} // namespace Lsx4::Translation
