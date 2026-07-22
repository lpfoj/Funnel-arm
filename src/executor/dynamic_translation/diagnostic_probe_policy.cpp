// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/diagnostic_probe_policy.h"

namespace Lsx4::Translation {

bool TraceAllocatorOffset(const std::uint64_t module_offset) noexcept {
    (void)module_offset;
    return false;
}

bool TraceTranslationOffset(const std::uint64_t module_offset) noexcept {
    (void)module_offset;
    return false;
}

} // namespace Lsx4::Translation
