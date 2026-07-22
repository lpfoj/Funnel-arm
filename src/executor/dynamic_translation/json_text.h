// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>
#include <string_view>

namespace Lsx4::Translation {

[[nodiscard]] std::string EscapeJsonText(std::string_view input);

} // namespace Lsx4::Translation
