// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/floating_relation.h"

#include "common/x86_decoder.h"

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

std::optional<ScalarRootPlan> DescribeScalarRootOperation(
    const std::uint32_t mnemonic_code) noexcept {
    const auto mnemonic = static_cast<X86Mnemonic>(mnemonic_code);
    const bool sqrt_single = mnemonic == X86_MNEMONIC_SQRTSS ||
                             mnemonic == X86_MNEMONIC_VSQRTSS;
    const bool sqrt_double = mnemonic == X86_MNEMONIC_SQRTSD ||
                             mnemonic == X86_MNEMONIC_VSQRTSD;
    const bool reciprocal = mnemonic == X86_MNEMONIC_RCPSS ||
                            mnemonic == X86_MNEMONIC_VRCPSS;
    const bool reciprocal_root = mnemonic == X86_MNEMONIC_RSQRTSS ||
                                 mnemonic == X86_MNEMONIC_VRSQRTSS;
    if (!sqrt_single && !sqrt_double && !reciprocal && !reciprocal_root) {
        return std::nullopt;
    }

    const bool vector_prefix = mnemonic == X86_MNEMONIC_VSQRTSS ||
                               mnemonic == X86_MNEMONIC_VSQRTSD ||
                               mnemonic == X86_MNEMONIC_VRCPSS ||
                               mnemonic == X86_MNEMONIC_VRSQRTSS;
    return ScalarRootPlan{
        .operation = reciprocal_root ? ScalarRootOperation::ReciprocalSquareRoot
                                     : reciprocal ? ScalarRootOperation::Reciprocal
                                                  : ScalarRootOperation::SquareRoot,
        .merge_operand = static_cast<std::uint8_t>(vector_prefix ? 1u : 0u),
        .source_operand = static_cast<std::uint8_t>(vector_prefix ? 2u : 1u),
        .double_precision = sqrt_double,
    };
}

double EvaluateScalarRoot(const ScalarRootOperation operation,
                          const double value) noexcept {
    switch (operation) {
    case ScalarRootOperation::SquareRoot:
        return std::sqrt(value);
    case ScalarRootOperation::Reciprocal:
        return 1.0 / value;
    case ScalarRootOperation::ReciprocalSquareRoot:
        return 1.0 / std::sqrt(value);
    }
    return value;
}

float EvaluateScalarRoot(const ScalarRootOperation operation,
                         const float value) noexcept {
    switch (operation) {
    case ScalarRootOperation::SquareRoot:
        return std::sqrt(value);
    case ScalarRootOperation::Reciprocal:
        return 1.0f / value;
    case ScalarRootOperation::ReciprocalSquareRoot:
        return 1.0f / std::sqrt(value);
    }
    return value;
}

} // namespace Lsx4::Translation
