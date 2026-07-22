// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace Lsx4::Translation {

consteval std::array<std::uint8_t, 32> BuildThunkIdentity(
    const std::string_view domain) {
    std::array<std::uint8_t, 32> result{};
    std::uint64_t state = 0x7a53d491c2e68b0full;
    for (std::size_t index = 0; index < domain.size(); ++index) {
        state ^= static_cast<std::uint8_t>(domain[index]);
        state = (state << 9) | (state >> 55);
        state += 0x165667b19e3779f9ull + index;
        result[index % result.size()] ^=
            static_cast<std::uint8_t>(state >> ((index % 8) * 8));
    }
    for (std::size_t index = 0; index < result.size(); ++index) {
        state ^= state >> 13;
        state *= 0x9e6c63d0676a9a99ull;
        state ^= state >> 29;
        result[index] ^= static_cast<std::uint8_t>(state >> 41);
    }
    return result;
}

inline constexpr auto FexThunkIdentity =
    BuildThunkIdentity("LSX4/FEX/HLE bridge/2026-07");
inline constexpr auto TranslationThunkIdentity =
    BuildThunkIdentity("LSX4/dynamic translation/HLE bridge/v1");

} // namespace Lsx4::Translation
