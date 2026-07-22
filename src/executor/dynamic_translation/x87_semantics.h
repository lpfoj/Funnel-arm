// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/guest_memory.h"
#include "executor/dynamic_translation/instruction_model.h"
#include "executor/dynamic_translation/machine_state.h"

#include <cstdint>

namespace Lsx4::Translation {

enum class X87Stop : std::uint8_t {
    Continue,
    MemoryFault,
    StackFault,
    Unsupported,
};

struct X87SemanticResult {
    X87Stop stop{X87Stop::Unsupported};
    std::uint64_t next_address{};
    std::uint64_t fault_address{};
};

[[nodiscard]] X87SemanticResult ExecuteX87Instruction(
    const Instruction& instruction, CpuFrame& frame,
    const GuestMemoryPort& memory) noexcept;

} // namespace Lsx4::Translation
