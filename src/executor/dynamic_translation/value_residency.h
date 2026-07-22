// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace Lsx4::Translation {

enum class ValueBank : std::uint8_t {
    Integer,
    Vector,
};

enum class ValueAccess : std::uint8_t {
    Read,
    Write,
    ReadWrite,
};

struct ValueIdentity {
    ValueBank bank{ValueBank::Integer};
    std::uint16_t number{};

    friend constexpr bool operator==(ValueIdentity, ValueIdentity) = default;
};

struct HostRegister {
    ValueBank bank{ValueBank::Integer};
    std::uint8_t number{};

    friend constexpr bool operator==(HostRegister, HostRegister) = default;
};

enum class TransferDirection : std::uint8_t {
    Fill,
    Spill,
};

struct ResidencyTransfer {
    TransferDirection direction{TransferDirection::Fill};
    ValueIdentity value{};
    HostRegister host{};
};

using TransferObserver = void (*)(const ResidencyTransfer& transfer, void* context);

class ValueResidency {
public:
    static constexpr std::size_t NoFutureUse = std::numeric_limits<std::size_t>::max();

    ValueResidency(std::span<const std::uint8_t> integer_hosts,
                   std::span<const std::uint8_t> vector_hosts,
                   TransferObserver observer, void* observer_context);

    [[nodiscard]] std::optional<HostRegister> Acquire(ValueIdentity value,
                                                      ValueAccess access,
                                                      std::size_t next_use,
                                                      bool pin = false);
    void SetPinned(ValueIdentity value, bool pinned) noexcept;
    void Forget(ValueIdentity value) noexcept;
    void CrossCallBoundary(std::span<const HostRegister> clobbered);
    void SpillAll();
    void Reset() noexcept;

    [[nodiscard]] std::size_t ResidentCount() const noexcept;
    [[nodiscard]] bool IsDirty(ValueIdentity value) const noexcept;

private:
    struct Lane {
        HostRegister host{};
        ValueIdentity value{};
        std::size_t next_use{NoFutureUse};
        bool occupied{};
        bool dirty{};
        bool pinned{};
    };

    [[nodiscard]] Lane* Locate(ValueIdentity value) noexcept;
    [[nodiscard]] const Lane* Locate(ValueIdentity value) const noexcept;
    [[nodiscard]] Lane* SelectLane(ValueBank bank) noexcept;
    void Emit(TransferDirection direction, const Lane& lane) const;
    void Vacate(Lane& lane, bool preserve_value);

    std::vector<Lane> lanes_{};
    TransferObserver observer_{};
    void* observer_context_{};
};

} // namespace Lsx4::Translation
