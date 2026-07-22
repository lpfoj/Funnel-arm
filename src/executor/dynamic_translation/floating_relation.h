// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Lsx4::Translation {

enum class FloatingOrder : std::uint8_t {
    Unordered,
    Less,
    Equal,
    Greater,
};

[[nodiscard]] FloatingOrder ClassifyFloatingOrder(double left, double right) noexcept;

[[nodiscard]] bool FloatingRelationMatches(double left, double right,
                                           std::uint32_t immediate) noexcept;

[[nodiscard]] double RoundFloatingValue(double value, std::uint32_t mode) noexcept;

} // namespace Lsx4::Translation
