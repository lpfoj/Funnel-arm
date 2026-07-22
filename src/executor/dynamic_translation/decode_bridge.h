// SPDX-FileCopyrightText: Copyright 2026 PS4Run Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <span>

namespace Executor::Jit {

struct LsxDecodedOp;

enum class ByteLensResult : std::uint8_t {
    Accepted,
    Rejected,
    Unavailable,
};

// Vendor-neutral decode boundary used by the translation engine. The implementation owns the
// concrete decoder and publishes only PS4Run's retained operation model.
[[nodiscard]] ByteLensResult LiftOneMachineInstruction(
    std::span<const std::uint8_t> bytes, LsxDecodedOp& operation) noexcept;

// Lightweight executable-target probe. It validates one complete long-mode instruction without
// exposing the decoder's instruction object to the execution engine.
[[nodiscard]] bool HasMachineInstruction(std::span<const std::uint8_t> bytes) noexcept;

// Diagnostic spelling for an architectural register code retained in LsxOperandRecord.
[[nodiscard]] const char* MachineRegisterName(std::uint32_t code) noexcept;

} // namespace Executor::Jit
