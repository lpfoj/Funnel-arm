// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdint>

namespace Lsx4::Translation {

struct GuestCapabilityPage {
    std::array<std::uint32_t, 4> registers{};
};

GuestCapabilityPage QueryGuestCapabilityPage(std::uint32_t page,
                                             std::uint32_t selector) noexcept;

} // namespace Lsx4::Translation
