// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/trace_memory.h"

#include "executor/dynamic_translation/process_memory.h"

#include <cctype>
#include <limits>

namespace Lsx4::Translation::TraceMemory {
namespace {

bool IsCompleteRange(const std::uint64_t address,
                     const std::size_t byte_count) noexcept {
    return byte_count != 0 &&
           byte_count - 1 <= std::numeric_limits<std::uint64_t>::max() - address &&
           IsPlausibleGuestAddress(address) &&
           IsPlausibleGuestAddress(address + byte_count - 1);
}

} // namespace

bool IsPlausibleGuestAddress(const std::uint64_t address) noexcept {
    const bool compact_guest = address >= UINT64_C(0x20000000) &&
                               address < UINT64_C(0x70000000);
    const bool extended_guest = address >= UINT64_C(0x100000000) &&
                                address < UINT64_C(0x900000000);
    return compact_guest || extended_guest;
}

bool ReadBytes(const std::uint64_t address, void* const destination,
               const std::size_t byte_count) noexcept {
    return destination != nullptr && IsCompleteRange(address, byte_count) &&
           ReadProcessGuestMemory(address, destination, byte_count);
}

bool ReadScalar(const std::uint64_t address, const std::size_t byte_count,
                std::uint64_t& value) noexcept {
    if (byte_count != 1 && byte_count != 2 && byte_count != 4 &&
        byte_count != 8) {
        value = 0;
        return false;
    }
    value = 0;
    return ReadBytes(address, &value, byte_count);
}

std::string ReadText(const std::uint64_t address,
                     const std::size_t maximum_length) {
    if (maximum_length == 0 || !IsPlausibleGuestAddress(address)) {
        return {};
    }
    std::string text;
    text.reserve(maximum_length);
    for (std::size_t offset = 0; offset < maximum_length; ++offset) {
        unsigned char byte = 0;
        if (!ReadBytes(address + offset, &byte, sizeof(byte))) {
            return text.empty() ? std::string{"<unreadable>"} : text;
        }
        if (byte == 0) {
            break;
        }
        text.push_back(std::isprint(byte) != 0 ? static_cast<char>(byte) : '.');
    }
    return text;
}

} // namespace Lsx4::Translation::TraceMemory
