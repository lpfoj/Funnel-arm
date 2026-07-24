// SPDX-FileCopyrightText: 2014 Tony Wasserka
// SPDX-FileCopyrightText: 2014 Dolphin Emulator Project
// SPDX-License-Identifier: BSD-3-Clause AND GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>

#pragma pack(1)
template <std::size_t Position, std::size_t Bits, typename T>
struct BitField {

    using Type = T;

    using UnderlyingType = typename std::conditional_t<std::is_enum_v<T>, std::underlying_type<T>,
                                                       std::enable_if<true, T>>::type;

    using StorageType = std::make_unsigned_t<UnderlyingType>;

    static constexpr std::size_t position = Position;
    static constexpr std::size_t bits = Bits;
    static constexpr StorageType mask = (((StorageType)~0) >> (8 * sizeof(T) - bits)) << position;

    [[nodiscard]] static constexpr StorageType FormatValue(const T& value) {
        return (static_cast<StorageType>(value) << position) & mask;
    }

    [[nodiscard]] static constexpr T ExtractValue(const StorageType& storage) {
        if constexpr (std::numeric_limits<UnderlyingType>::is_signed) {
            std::size_t shift = 8 * sizeof(T) - bits;
            return static_cast<T>(static_cast<UnderlyingType>(storage << (shift - position)) >>
                                  shift);
        } else {
            return static_cast<T>((storage & mask) >> position);
        }
    }

    BitField(T val) = delete;
    BitField& operator=(T val) = delete;

    constexpr BitField() noexcept = default;

    constexpr BitField(const BitField&) noexcept = default;
    constexpr BitField& operator=(const BitField&) noexcept = default;

    constexpr BitField(BitField&&) noexcept = default;
    constexpr BitField& operator=(BitField&&) noexcept = default;

    [[nodiscard]] constexpr operator T() const {
        return Value();
    }

    constexpr void Assign(const T& value) {
        storage = (static_cast<StorageType>(storage) & ~mask) | FormatValue(value);
    }

    [[nodiscard]] constexpr T Value() const {
        return ExtractValue(storage);
    }

    [[nodiscard]] constexpr explicit operator bool() const {
        return Value() != 0;
    }

private:
    StorageType storage;

    static_assert(bits + position <= 8 * sizeof(T), "Bitfield out of range");

    static_assert(position < 8 * sizeof(T), "Invalid position");
    static_assert(bits <= 8 * sizeof(T), "Invalid number of bits");
    static_assert(bits > 0, "Invalid number of bits");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable in a BitField");
};
#pragma pack()
