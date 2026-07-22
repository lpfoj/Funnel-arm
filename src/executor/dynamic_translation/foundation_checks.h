// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Lsx4::Translation {

struct FoundationCheckReport {
    std::uint64_t completed{};
    std::uint64_t failed{};

    [[nodiscard]] bool AllPassed() const noexcept {
        return completed != 0 && failed == 0;
    }
};

[[nodiscard]] FoundationCheckReport RunFoundationChecks();

} // namespace Lsx4::Translation
