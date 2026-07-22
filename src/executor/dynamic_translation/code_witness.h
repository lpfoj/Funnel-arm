// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Lsx4::Translation {

using ReadGuestBytes =
    std::function<bool(std::uint64_t guest_address, std::span<std::uint8_t> destination)>;

class PageRevision final {
public:
    [[nodiscard]] std::uint64_t Observe() const noexcept;

private:
    friend class MutationLedger;
    std::atomic<std::uint64_t> value_{1};
};

struct PageObservation {
    std::uint64_t guest_page{};
    std::uint64_t revision{};
    std::shared_ptr<PageRevision> source{};
};

struct CodeWitness {
    std::uint64_t guest_address{};
    std::uint64_t byte_count{};
    std::uint64_t fingerprint{};
    std::vector<PageObservation> observations{};
};

class MutationLedger {
public:
    static constexpr std::uint64_t kGuestPageSize = 4096;
    static constexpr std::size_t kMaximumWitnessBytes = 1024 * 1024;

    [[nodiscard]] std::optional<CodeWitness> Capture(
        std::uint64_t guest_address, std::size_t byte_count,
        const ReadGuestBytes& reader);
    [[nodiscard]] bool Validate(const CodeWitness& witness,
                                const ReadGuestBytes& reader) const;

    void MarkChanged(std::uint64_t guest_address, std::size_t byte_count);
    [[nodiscard]] std::uint64_t GlobalRevision() const noexcept;

private:
    [[nodiscard]] std::vector<PageObservation> ObserveRange(
        std::uint64_t guest_address, std::size_t byte_count);
    [[nodiscard]] static bool RevisionsAreCurrent(
        const std::vector<PageObservation>& observations) noexcept;

    mutable std::mutex directory_mutex_{};
    std::unordered_map<std::uint64_t, std::shared_ptr<PageRevision>> pages_{};
    std::atomic<std::uint64_t> global_revision_{1};
};

} // namespace Lsx4::Translation
