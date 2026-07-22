// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/json_text.h"

namespace Lsx4::Translation {

std::string EscapeJsonText(const std::string_view input) {
    constexpr char digits[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(input.size() + input.size() / 8);
    for (const unsigned char byte : input) {
        if (byte >= 0x20 && byte != '"' && byte != '\\') {
            escaped.push_back(static_cast<char>(byte));
        } else if (byte == '"' || byte == '\\') {
            escaped.push_back('\\');
            escaped.push_back(static_cast<char>(byte));
        } else {
            escaped.append("\\u00");
            escaped.push_back(digits[byte >> 4]);
            escaped.push_back(digits[byte & 0x0f]);
        }
    }
    return escaped;
}

} // namespace Lsx4::Translation
