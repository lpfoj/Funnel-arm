// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/machine_state.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace Lsx4::Translation {

enum class X87StackOutcome : std::uint8_t {
    Complete,
    StackFault,
    InvalidIndex,
};

[[nodiscard]] X87StackOutcome PushX87(CpuFrame& frame, long double value) noexcept;
[[nodiscard]] X87StackOutcome PopX87(CpuFrame& frame,
                                    long double* value = nullptr) noexcept;
[[nodiscard]] std::optional<long double> ReadX87(const CpuFrame& frame,
                                                 std::uint8_t logical_index) noexcept;
[[nodiscard]] X87StackOutcome WriteX87(CpuFrame& frame, std::uint8_t logical_index,
                                      long double value) noexcept;
[[nodiscard]] X87StackOutcome ExchangeX87(CpuFrame& frame,
                                         std::uint8_t logical_index) noexcept;
[[nodiscard]] std::size_t X87Depth(const CpuFrame& frame) noexcept;

} // namespace Lsx4::Translation
