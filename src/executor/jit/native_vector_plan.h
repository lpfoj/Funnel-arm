// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/jit/lsx_translation_engine.h"

namespace Executor::Jit {

enum class DirectVectorKernel : std::uint8_t {
    None,
    PackedBitTest,
    DuplicateEvenDwords,
    DuplicateOddDwords,
    DuplicateLowQwords,
};

struct DirectVectorPlan {
    DirectVectorKernel kernel{DirectVectorKernel::None};
    std::uint32_t width_bytes{};

    explicit operator bool() const noexcept {
        return kernel != DirectVectorKernel::None;
    }
};

// Convert immutable decoder records into a small lowering decision. The emitter
// consumes only this plan and no longer owns opcode-shape validation for these
// kernels.
DirectVectorPlan PlanDirectVectorKernel(const LsxDecodedOp& instruction) noexcept;

} // namespace Executor::Jit
