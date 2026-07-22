// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Lsx4::Translation {

// Title-specific instruction offsets do not belong in the translation engine.
// Diagnostic clients may install a policy later; the production default is off.
[[nodiscard]] bool TraceAllocatorOffset(std::uint64_t module_offset) noexcept;
[[nodiscard]] bool TraceTranslationOffset(std::uint64_t module_offset) noexcept;

} // namespace Lsx4::Translation
