// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <type_traits>

namespace Common {

template <typename N, typename D>
    requires std::is_integral_v<N> && std::is_unsigned_v<D>
[[nodiscard]] constexpr N DivCeil(N number, D divisor) {
    return static_cast<N>((static_cast<D>(number) + divisor - 1) / divisor);
}

template <typename N, typename D>
    requires std::is_integral_v<N> && std::is_unsigned_v<D>
[[nodiscard]] constexpr N DivCeilLog2(N value, D alignment_log2) {
    return static_cast<N>((static_cast<D>(value) + (D(1) << alignment_log2) - 1) >> alignment_log2);
}

}
