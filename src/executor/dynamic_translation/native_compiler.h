// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/code_memory.h"
#include "executor/dynamic_translation/machine_state.h"
#include "executor/dynamic_translation/scalar_ir_builder.h"

#include <cstddef>
#include <cstdint>

namespace Lsx4::Translation {

using NativeRegionEntry = std::uint64_t (*)(CpuFrame* frame);

enum class NativeCompileFailure : std::uint8_t {
    None,
    InvalidIr,
    UnsupportedAction,
    RegisterPressure,
    FrameTooLarge,
    EncodingFailure,
    ArenaExhausted,
};

struct NativeArtifact {
    NativeRegionEntry entry{};
    CodeReservation reservation{};
    NativeCompileFailure failure{NativeCompileFailure::None};

    [[nodiscard]] bool Succeeded() const noexcept {
        return entry != nullptr && reservation.IsValid() &&
               failure == NativeCompileFailure::None;
    }
};

[[nodiscard]] NativeArtifact CompileScalarIr(const TranslationProgram& program,
                                             ExecutableArena& arena);

} // namespace Lsx4::Translation
