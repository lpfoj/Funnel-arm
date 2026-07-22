// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace Lsx4::Translation::TraceMemory {

[[nodiscard]] bool IsPlausibleGuestAddress(std::uint64_t address) noexcept;
[[nodiscard]] bool ReadBytes(std::uint64_t address, void* destination,
                             std::size_t byte_count) noexcept;
[[nodiscard]] bool ReadScalar(std::uint64_t address, std::size_t byte_count,
                              std::uint64_t& value) noexcept;
[[nodiscard]] std::string ReadText(std::uint64_t address,
                                   std::size_t maximum_length);

} // namespace Lsx4::Translation::TraceMemory
