// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/execution_context.h"
#include "executor/dynamic_translation/guest_memory.h"
#include "executor/dynamic_translation/translation_service.h"

#include <cstddef>
#include <cstdint>

namespace Lsx4::Translation {

enum class BoundaryAction : std::uint8_t {
    NotHandled,
    Continue,
    Transfer,
    Complete,
    Fault,
};

struct BoundaryDecision {
    BoundaryAction action{BoundaryAction::NotHandled};
    std::uint64_t next_address{};
    std::uint64_t result{};
    std::uint64_t fault_address{};
};

using BoundaryPort = BoundaryDecision (*)(const Instruction& instruction,
                                          CpuFrame& frame,
                                          void* context) noexcept;

struct ExecutionPorts {
    InstructionFetcher fetch{};
    void* fetch_context{};
    ReadGuestBytes read_code{};
    GuestMemoryPort memory{};
    BoundaryPort boundary{};
    void* boundary_context{};
};

struct ExecutionLimits {
    std::uint64_t return_sentinel{};
    std::uint64_t maximum_instructions{20'000'000};
    std::size_t region_instruction_limit{64};
};

enum class DriverStop : std::uint8_t {
    Complete,
    GuestReadFailure,
    MemoryFault,
    UnsupportedInstruction,
    CompileFailure,
    RetryLimit,
    InstructionLimit,
    BoundaryFault,
};

struct DriverResult {
    DriverStop stop{DriverStop::GuestReadFailure};
    std::uint64_t result{};
    std::uint64_t final_address{};
    std::uint64_t fault_address{};
    std::uint64_t instructions{};
};

class ExecutionDriver {
public:
    explicit ExecutionDriver(std::size_t arena_slab_bytes = 2 * 1024 * 1024);

    [[nodiscard]] DriverResult Run(std::uint64_t first_address, CpuFrame& frame,
                                   const ExecutionPorts& ports,
                                   const ExecutionLimits& limits = {});
    [[nodiscard]] TranslationService& Service() noexcept;

private:
    TranslationService service_;
};

} // namespace Lsx4::Translation
