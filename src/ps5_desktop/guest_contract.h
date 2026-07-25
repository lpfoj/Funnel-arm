// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>

namespace Funnel::Ps5Desktop {

inline constexpr std::uint32_t GuestContractAbiVersion = 1;
inline constexpr std::uint32_t GuestPageSize = 0x4000;
inline constexpr std::uint64_t GuestAddressLimit = 0x10000000000ull;

enum CpuStateRequirement : std::uint64_t {
    CpuStateGprAndFlags = 1ull << 0,
    CpuStateFsGsBase = 1ull << 1,
    CpuStateX87 = 1ull << 2,
    CpuStateYmm = 1ull << 3,
};

enum InstructionRequirement : std::uint64_t {
    InstructionSse4aExtrqInsertq = 1ull << 0,
    InstructionMonitorxMwaitx = 1ull << 1,
};

struct GuestContract {
    std::uint32_t abi_version{};
    std::uint32_t guest_page_size{};
    std::uint64_t guest_address_limit{};
    std::uint64_t required_cpu_state{};
    std::uint64_t required_instructions{};
};

[[nodiscard]] GuestContract QueryGuestContract() noexcept;

}
