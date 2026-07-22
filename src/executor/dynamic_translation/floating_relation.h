// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <optional>

namespace Lsx4::Translation {

enum class FloatingOrder : std::uint8_t {
    Unordered,
    Less,
    Equal,
    Greater,
};

enum class ScalarRootOperation : std::uint8_t {
    SquareRoot,
    Reciprocal,
    ReciprocalSquareRoot,
};

struct ScalarRootPlan {
    ScalarRootOperation operation{};
    std::uint8_t merge_operand{};
    std::uint8_t source_operand{};
    bool double_precision{};
};

[[nodiscard]] FloatingOrder ClassifyFloatingOrder(double left, double right) noexcept;

[[nodiscard]] bool FloatingRelationMatches(double left, double right,
                                           std::uint32_t immediate) noexcept;

[[nodiscard]] double RoundFloatingValue(double value, std::uint32_t mode) noexcept;

[[nodiscard]] std::optional<ScalarRootPlan> DescribeScalarRootOperation(
    std::uint32_t mnemonic) noexcept;

[[nodiscard]] double EvaluateScalarRoot(ScalarRootOperation operation,
                                        double value) noexcept;
[[nodiscard]] float EvaluateScalarRoot(ScalarRootOperation operation,
                                       float value) noexcept;

} // namespace Lsx4::Translation
