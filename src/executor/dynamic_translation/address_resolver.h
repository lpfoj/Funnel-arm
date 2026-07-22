// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/instruction_model.h"
#include "executor/dynamic_translation/machine_state.h"

#include <cstdint>
#include <optional>

namespace Lsx4::Translation {

[[nodiscard]] std::optional<std::uint64_t> ResolveGuestAddress(
    const Instruction& instruction, const Operand& operand,
    const CpuFrame& frame) noexcept;

} // namespace Lsx4::Translation
