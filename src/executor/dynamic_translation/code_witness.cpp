// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/code_witness.h"

#include "common/content_fingerprint.h"

#include <algorithm>
#include <limits>

namespace Lsx4::Translation {
namespace {

constexpr std::uint64_t PageBase(const std::uint64_t address) noexcept {
    return address & ~(MutationLedger::kGuestPageSize - 1);
}

constexpr std::uint64_t LastCoveredAddress(const std::uint64_t address,
                                           const std::size_t byte_count) noexcept {
    const std::uint64_t distance = static_cast<std::uint64_t>(byte_count - 1);
    if (distance > std::numeric_limits<std::uint64_t>::max() - address) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return address + distance;
}

std::uint64_t Fingerprint(const std::span<const std::uint8_t> bytes) noexcept {
    return Common::FingerprintBytes(bytes, Common::FingerprintDomain::GuestCode);
}

} // namespace

std::uint64_t PageRevision::Observe() const noexcept {
    return value_.load(std::memory_order_acquire);
}

std::vector<PageObservation> MutationLedger::ObserveRange(
    const std::uint64_t guest_address, const std::size_t byte_count) {
    std::vector<PageObservation> result{};
    if (byte_count == 0) {
        return result;
    }

    const std::uint64_t first_page = PageBase(guest_address);
    const std::uint64_t last_page = PageBase(LastCoveredAddress(guest_address, byte_count));
    const std::uint64_t page_distance = (last_page - first_page) / kGuestPageSize;
    result.reserve(static_cast<std::size_t>(page_distance + 1));

    std::lock_guard lock{directory_mutex_};
    for (std::uint64_t page = first_page;; page += kGuestPageSize) {
        auto& cell = pages_[page];
        if (!cell) {
            cell = std::make_shared<PageRevision>();
        }
        result.push_back({page, cell->Observe(), cell});
        if (page == last_page) {
            break;
        }
    }
    return result;
}

bool MutationLedger::RevisionsAreCurrent(
    const std::vector<PageObservation>& observations) noexcept {
    return std::all_of(observations.begin(), observations.end(),
                       [](const PageObservation& observation) {
                           return observation.source &&
                                  observation.source->Observe() == observation.revision;
                       });
}

std::optional<CodeWitness> MutationLedger::Capture(
    const std::uint64_t guest_address, const std::size_t byte_count,
    const ReadGuestBytes& reader) {
    if (byte_count == 0 || byte_count > kMaximumWitnessBytes || !reader) {
        return std::nullopt;
    }

    std::vector<std::uint8_t> image(byte_count);
    for (unsigned attempt = 0; attempt < 4; ++attempt) {
        auto observations = ObserveRange(guest_address, byte_count);
        if (!reader(guest_address, image) || !RevisionsAreCurrent(observations)) {
            continue;
        }
        return CodeWitness{guest_address, static_cast<std::uint64_t>(byte_count),
                           Fingerprint(image), std::move(observations)};
    }
    return std::nullopt;
}

bool MutationLedger::Validate(const CodeWitness& witness,
                              const ReadGuestBytes& reader) const {
    if (witness.byte_count == 0 || !reader ||
        witness.byte_count > kMaximumWitnessBytes ||
        !RevisionsAreCurrent(witness.observations)) {
        return false;
    }

    std::vector<std::uint8_t> image(static_cast<std::size_t>(witness.byte_count));
    if (!reader(witness.guest_address, image) ||
        !RevisionsAreCurrent(witness.observations)) {
        return false;
    }
    return Fingerprint(image) == witness.fingerprint;
}

void MutationLedger::MarkChanged(const std::uint64_t guest_address,
                                 const std::size_t byte_count) {
    if (byte_count == 0) {
        return;
    }
    auto observations = ObserveRange(guest_address, byte_count);
    for (const PageObservation& observation : observations) {
        observation.source->value_.fetch_add(1, std::memory_order_release);
    }
    global_revision_.fetch_add(1, std::memory_order_release);
}

std::uint64_t MutationLedger::GlobalRevision() const noexcept {
    return global_revision_.load(std::memory_order_acquire);
}

static_assert(PageBase(0x1234) == 0x1000);
static_assert(LastCoveredAddress(0x1000, 1) == 0x1000);
static_assert(LastCoveredAddress(std::numeric_limits<std::uint64_t>::max() - 2, 8) ==
              std::numeric_limits<std::uint64_t>::max());

} // namespace Lsx4::Translation
