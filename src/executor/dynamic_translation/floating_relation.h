// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Lsx4::Translation {

[[nodiscard]] bool FloatingRelationMatches(double left, double right,
                                           std::uint32_t immediate) noexcept;

} // namespace Lsx4::Translation
