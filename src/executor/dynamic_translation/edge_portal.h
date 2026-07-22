// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace Lsx4::Translation {

enum class EdgePolicy : std::uint8_t {
    Chainable,
    DispatcherOnly,
    HleBoundary,
    SignalBoundary,
};

struct ResolvedEdge {
    std::uintptr_t host_address{};
    std::uint64_t guest_revision{};
    std::uint64_t runtime_epoch{};
};

class alignas(64) ExitPortal {
public:
    ExitPortal(std::uint64_t guest_destination, EdgePolicy policy) noexcept;

    ExitPortal(const ExitPortal&) = delete;
    ExitPortal& operator=(const ExitPortal&) = delete;

    [[nodiscard]] std::uint64_t GuestDestination() const noexcept;
    [[nodiscard]] EdgePolicy Policy() const noexcept;
    [[nodiscard]] ResolvedEdge Resolve(std::uint64_t required_revision,
                                       std::uint64_t required_epoch) const noexcept;

private:
    friend class EdgeDirectory;

    void Replace(std::uintptr_t host_address, std::uint64_t guest_revision,
                 std::uint64_t runtime_epoch) noexcept;

    const std::uint64_t guest_destination_;
    const EdgePolicy policy_;
    std::atomic<std::uint64_t> sequence_{};
    std::atomic<std::uintptr_t> host_address_{};
    std::atomic<std::uint64_t> guest_revision_{};
    std::atomic<std::uint64_t> runtime_epoch_{};
};

class EdgeDirectory {
public:
    void Attach(ExitPortal& portal);
    void Detach(ExitPortal& portal) noexcept;
    void Announce(std::uint64_t guest_address, std::uintptr_t host_address,
                  std::uint64_t guest_revision, std::uint64_t runtime_epoch);
    void Revoke(std::uint64_t guest_address) noexcept;
    void RevokeAll() noexcept;

private:
    std::mutex mutex_{};
    std::unordered_map<std::uint64_t, std::unordered_set<ExitPortal*>> listeners_{};
};

} // namespace Lsx4::Translation
