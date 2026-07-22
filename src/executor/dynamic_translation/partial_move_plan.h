// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/retiring_execution_core.h"

#include <cstdint>
#include <optional>

namespace Executor::Jit {

struct PartialMovePlan {
    std::uint8_t memory_operand{};
    bool writes_memory{};
    bool clears_upper_half{};
    bool has_merge_lane{};
    std::uint8_t destination_lane_offset{};
    std::uint32_t destination_offset{};
    std::uint32_t source_offset{};
    std::uint32_t merge_offset{};
};

struct CrossLaneMovePlan {
    std::uint8_t lower_operand{};
    std::uint8_t upper_operand{};
    std::uint8_t lower_source_offset{};
    std::uint8_t upper_source_offset{};
    bool clears_upper_half{};
};

[[nodiscard]] std::optional<PartialMovePlan> PlanPartialPackedMove(
    const LsxDecodedOp& instruction, std::uint32_t ymm_bank_offset) noexcept;

[[nodiscard]] std::optional<CrossLaneMovePlan> PlanCrossLanePackedMove(
    const LsxDecodedOp& instruction) noexcept;

} // namespace Executor::Jit
