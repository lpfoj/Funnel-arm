// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <optional>

namespace Lsx4::Translation {

struct IntegerMathResult {
    std::uint64_t value{};
    std::uint64_t flag_bits{};
    std::uint64_t flag_mask{};
    bool effective{};
};

[[nodiscard]] IntegerMathResult AddWithCarry(std::uint64_t left,
                                             std::uint64_t right,
                                             bool carry,
                                             std::uint32_t width) noexcept;
[[nodiscard]] IntegerMathResult SubtractWithBorrow(std::uint64_t left,
                                                   std::uint64_t right,
                                                   bool borrow,
                                                   std::uint32_t width) noexcept;
[[nodiscard]] std::optional<IntegerMathResult> EvaluateBitMovement(
    std::uint32_t mnemonic, std::uint64_t value, std::uint64_t count,
    std::uint32_t width) noexcept;

} // namespace Lsx4::Translation
