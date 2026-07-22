// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/operation_ir.h"
#include "executor/dynamic_translation/region_plan.h"

#include <cstdint>
#include <vector>

namespace Lsx4::Translation {

enum class FlagFormula : std::uint8_t {
    Addition,
    Subtraction,
    Logical,
    Increment,
    Decrement,
};

struct IntegerAccess {
    std::uint8_t slot{};
    std::uint8_t width{};
    bool upper_byte{};
};

struct TranslationProgram {
    explicit TranslationProgram(std::uint64_t first_address) : ir{first_address} {}

    OperationIr ir;
    std::vector<Instruction> semantic_steps{};
};

[[nodiscard]] std::uint32_t PackIntegerAccess(IntegerAccess access) noexcept;
[[nodiscard]] IntegerAccess UnpackIntegerAccess(std::uint32_t detail) noexcept;
[[nodiscard]] TranslationProgram BuildScalarProgram(const RegionPlan& plan);

} // namespace Lsx4::Translation
