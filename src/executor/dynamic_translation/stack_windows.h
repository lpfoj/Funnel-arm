// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/live_state_port.h"

#include <cstdint>

namespace Lsx4::Translation {

struct StackWindow {
    std::uint64_t lower_bound{};
    std::uint64_t stack_pointer{};
    std::uint64_t restore_pointer{};

    [[nodiscard]] bool IsValid() const noexcept {
        return lower_bound != 0 && stack_pointer > lower_bound;
    }
};

[[nodiscard]] StackWindow AcquireStackWindow(LiveStateView parent) noexcept;
void ReleaseStackWindow(const StackWindow& window) noexcept;

} // namespace Lsx4::Translation
