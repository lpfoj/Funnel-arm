// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/atomic_memory.h"
#include "executor/dynamic_translation/instruction_model.h"
#include "executor/dynamic_translation/machine_state.h"

#include <cstdint>
#include <span>

namespace Lsx4::Translation {

using GuestRead = bool (*)(std::uint64_t address, std::span<std::uint8_t> destination,
                           void* context) noexcept;
using GuestWrite = bool (*)(std::uint64_t address,
                            std::span<const std::uint8_t> source,
                            void* context) noexcept;

struct GuestMemoryPort {
    GuestRead read{};
    GuestWrite write{};
    void* context{};
    GuestCompareExchange compare_exchange{};
};

enum class SemanticStop : std::uint8_t {
    Continue,
    Transfer,
    Return,
    MemoryFault,
    Retry,
    Unsupported,
};

struct SemanticResult {
    SemanticStop stop{SemanticStop::Unsupported};
    std::uint64_t next_address{};
    std::uint64_t fault_address{};
};

[[nodiscard]] SemanticResult ExecuteScalarInstruction(
    const Instruction& instruction, CpuFrame& frame,
    const GuestMemoryPort& memory) noexcept;

} // namespace Lsx4::Translation
