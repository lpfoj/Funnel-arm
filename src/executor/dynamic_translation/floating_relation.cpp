// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/floating_relation.h"

#include <array>
#include <cmath>

namespace Lsx4::Translation {

FloatingOrder ClassifyFloatingOrder(const double left, const double right) noexcept {
    if (std::isunordered(left, right)) {
        return FloatingOrder::Unordered;
    }
    if (left < right) {
        return FloatingOrder::Less;
    }
    if (left > right) {
        return FloatingOrder::Greater;
    }
    return FloatingOrder::Equal;
}

bool FloatingRelationMatches(const double left, const double right,
                             const std::uint32_t immediate) noexcept {
    enum RelationBit : std::uint8_t {
        Unordered = 1u,
        Equal = 2u,
        Less = 4u,
        Greater = 8u,
    };

    constexpr std::array<std::uint8_t, 16> accepted_relations = {
        Equal,
        Less,
        static_cast<std::uint8_t>(Less | Equal),
        Unordered,
        static_cast<std::uint8_t>(Unordered | Less | Greater),
        static_cast<std::uint8_t>(Unordered | Equal | Greater),
        static_cast<std::uint8_t>(Unordered | Greater),
        static_cast<std::uint8_t>(Less | Equal | Greater),
        static_cast<std::uint8_t>(Unordered | Equal),
        static_cast<std::uint8_t>(Unordered | Less),
        static_cast<std::uint8_t>(Unordered | Less | Equal),
        0,
        static_cast<std::uint8_t>(Less | Greater),
        static_cast<std::uint8_t>(Equal | Greater),
        Greater,
        static_cast<std::uint8_t>(Unordered | Less | Equal | Greater),
    };

    std::uint8_t observed = 0;
    switch (ClassifyFloatingOrder(left, right)) {
    case FloatingOrder::Unordered:
        observed = Unordered;
        break;
    case FloatingOrder::Less:
        observed = Less;
        break;
    case FloatingOrder::Equal:
        observed = Equal;
        break;
    case FloatingOrder::Greater:
        observed = Greater;
        break;
    }
    return (accepted_relations[immediate & 0x0fu] & observed) != 0;
}

double RoundFloatingValue(const double value, const std::uint32_t mode) noexcept {
    constexpr std::array<double (*)(double), 4> operations = {
        static_cast<double (*)(double)>(std::nearbyint),
        static_cast<double (*)(double)>(std::floor),
        static_cast<double (*)(double)>(std::ceil),
        static_cast<double (*)(double)>(std::trunc),
    };
    return operations[mode & 3u](value);
}

} // namespace Lsx4::Translation
