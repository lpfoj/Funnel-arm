// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/machine_state.h"

#include <cstdint>
#include <optional>

namespace Lsx4::Translation {

class ExecutionBinding {
public:
    explicit ExecutionBinding(CpuFrame& frame) noexcept;
    ~ExecutionBinding();

    ExecutionBinding(const ExecutionBinding&) = delete;
    ExecutionBinding& operator=(const ExecutionBinding&) = delete;

private:
    CpuFrame* previous_{};
};

[[nodiscard]] CpuFrame* CurrentExecutionFrame() noexcept;
[[nodiscard]] bool SnapshotExecutionFrame(CpuFrame& destination) noexcept;
void PostExecutionReplacement(const CpuFrame& replacement) noexcept;
[[nodiscard]] bool TakeExecutionReplacement(CpuFrame& destination) noexcept;

void PostExecutionCompletion(std::uint64_t result) noexcept;
[[nodiscard]] std::optional<std::uint64_t> TakeExecutionCompletion() noexcept;
void ClearExecutionMessages() noexcept;

} // namespace Lsx4::Translation
