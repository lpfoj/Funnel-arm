// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "executor/dynamic_translation/instruction_model.h"

namespace Lsx4::Translation {

enum class FlowBoundary : std::uint8_t {
    None,
    Jump,
    ConditionalJump,
    Call,
    Return,
    SystemTransfer,
};

enum class RegionStop : std::uint8_t {
    FlowBoundary,
    InstructionLimit,
    ReadFailure,
    InvalidInstruction,
    AddressOverflow,
};

struct PlannedInstruction {
    Instruction instruction{};
    FlowBoundary boundary{FlowBoundary::None};
};

struct RegionPlan {
    std::uint64_t first_address{};
    std::uint64_t continuation{};
    std::vector<PlannedInstruction> sequence{};
    RegionStop stop{RegionStop::InstructionLimit};

    [[nodiscard]] bool IsExecutable() const noexcept {
        return !sequence.empty() && stop != RegionStop::ReadFailure &&
               stop != RegionStop::InvalidInstruction && stop != RegionStop::AddressOverflow;
    }
};

using InstructionFetcher = std::size_t (*)(std::uint64_t address,
                                           std::span<std::uint8_t> destination,
                                           void* context) noexcept;

[[nodiscard]] FlowBoundary ClassifyFlow(std::uint32_t mnemonic) noexcept;
[[nodiscard]] RegionPlan PlanRegion(std::uint64_t first_address,
                                    std::size_t instruction_limit,
                                    InstructionFetcher fetch,
                                    void* fetch_context) noexcept;
[[nodiscard]] bool ResolveRelativeTarget(const Instruction& instruction,
                                         std::uint64_t& target) noexcept;

} // namespace Lsx4::Translation
