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

[[nodiscard]] std::optional<PartialMovePlan> PlanPartialPackedMove(
    const LsxDecodedOp& instruction, std::uint32_t ymm_bank_offset) noexcept;

} // namespace Executor::Jit
