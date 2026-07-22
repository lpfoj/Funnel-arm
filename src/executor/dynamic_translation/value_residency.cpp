// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/value_residency.h"

#include <algorithm>

namespace Lsx4::Translation {

ValueResidency::ValueResidency(const std::span<const std::uint8_t> integer_hosts,
                               const std::span<const std::uint8_t> vector_hosts,
                               const TransferObserver observer,
                               void* const observer_context)
    : observer_{observer}, observer_context_{observer_context} {
    lanes_.reserve(integer_hosts.size() + vector_hosts.size());
    for (const std::uint8_t host : integer_hosts) {
        lanes_.push_back({.host = {ValueBank::Integer, host}});
    }
    for (const std::uint8_t host : vector_hosts) {
        lanes_.push_back({.host = {ValueBank::Vector, host}});
    }
}

ValueResidency::Lane* ValueResidency::Locate(const ValueIdentity value) noexcept {
    const auto found = std::find_if(lanes_.begin(), lanes_.end(),
                                    [value](const Lane& lane) {
                                        return lane.occupied && lane.value == value;
                                    });
    return found == lanes_.end() ? nullptr : &*found;
}

const ValueResidency::Lane* ValueResidency::Locate(const ValueIdentity value) const noexcept {
    const auto found = std::find_if(lanes_.begin(), lanes_.end(),
                                    [value](const Lane& lane) {
                                        return lane.occupied && lane.value == value;
                                    });
    return found == lanes_.end() ? nullptr : &*found;
}

ValueResidency::Lane* ValueResidency::SelectLane(const ValueBank bank) noexcept {
    Lane* victim = nullptr;
    for (Lane& lane : lanes_) {
        if (lane.host.bank != bank) {
            continue;
        }
        if (!lane.occupied) {
            return &lane;
        }
        if (!lane.pinned && (!victim || lane.next_use > victim->next_use)) {
            victim = &lane;
        }
    }
    return victim;
}

void ValueResidency::Emit(const TransferDirection direction, const Lane& lane) const {
    if (observer_) {
        observer_({direction, lane.value, lane.host}, observer_context_);
    }
}

void ValueResidency::Vacate(Lane& lane, const bool preserve_value) {
    if (lane.occupied && lane.dirty && preserve_value) {
        Emit(TransferDirection::Spill, lane);
    }
    lane.value = {};
    lane.next_use = NoFutureUse;
    lane.occupied = false;
    lane.dirty = false;
    lane.pinned = false;
}

std::optional<HostRegister> ValueResidency::Acquire(const ValueIdentity value,
                                                    const ValueAccess access,
                                                    const std::size_t next_use,
                                                    const bool pin) {
    if (Lane* const existing = Locate(value)) {
        existing->next_use = next_use;
        existing->pinned = existing->pinned || pin;
        existing->dirty = existing->dirty || access != ValueAccess::Read;
        return existing->host;
    }

    Lane* const lane = SelectLane(value.bank);
    if (!lane) {
        return std::nullopt;
    }
    Vacate(*lane, true);
    lane->value = value;
    lane->next_use = next_use;
    lane->occupied = true;
    lane->dirty = access != ValueAccess::Read;
    lane->pinned = pin;
    if (access != ValueAccess::Write) {
        Emit(TransferDirection::Fill, *lane);
    }
    return lane->host;
}

void ValueResidency::SetPinned(const ValueIdentity value, const bool pinned) noexcept {
    if (Lane* const lane = Locate(value)) {
        lane->pinned = pinned;
    }
}

void ValueResidency::Forget(const ValueIdentity value) noexcept {
    if (Lane* const lane = Locate(value)) {
        Vacate(*lane, false);
    }
}

void ValueResidency::CrossCallBoundary(const std::span<const HostRegister> clobbered) {
    for (Lane& lane : lanes_) {
        if (!lane.occupied || std::find(clobbered.begin(), clobbered.end(), lane.host) ==
                                  clobbered.end()) {
            continue;
        }
        Vacate(lane, true);
    }
}

void ValueResidency::SpillAll() {
    for (Lane& lane : lanes_) {
        if (lane.occupied && lane.dirty) {
            Emit(TransferDirection::Spill, lane);
            lane.dirty = false;
        }
    }
}

void ValueResidency::Reset() noexcept {
    for (Lane& lane : lanes_) {
        Vacate(lane, false);
    }
}

std::size_t ValueResidency::ResidentCount() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        lanes_.begin(), lanes_.end(), [](const Lane& lane) { return lane.occupied; }));
}

bool ValueResidency::IsDirty(const ValueIdentity value) const noexcept {
    const Lane* const lane = Locate(value);
    return lane && lane->dirty;
}

} // namespace Lsx4::Translation
