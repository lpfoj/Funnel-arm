// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/atomic_memory.h"

#include <cstdint>
#include <span>

namespace Lsx4::Translation {

using GuestRead = bool (*)(std::uint64_t address, std::span<std::uint8_t> destination,
                           void* context) noexcept;
using GuestWrite = bool (*)(std::uint64_t address,
                            std::span<const std::uint8_t> source,
                            void* context) noexcept;

struct GuestMemoryPort {
    GuestRead read{};
    GuestWrite write{};
    void* context{};
    GuestCompareExchange compare_exchange{};
};

} // namespace Lsx4::Translation
