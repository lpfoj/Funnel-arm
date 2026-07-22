// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/instruction_model.h"

#include <cstdint>
#include <optional>

namespace Lsx4::Translation {

struct IntegerAccess {
    std::uint8_t slot{};
    std::uint8_t width{};
    bool upper_byte{};
};

[[nodiscard]] std::optional<IntegerAccess> DecodeIntegerRegister(
    const Operand& operand) noexcept;
[[nodiscard]] std::uint32_t PackIntegerAccess(IntegerAccess access) noexcept;
[[nodiscard]] IntegerAccess UnpackIntegerAccess(std::uint32_t detail) noexcept;

} // namespace Lsx4::Translation
