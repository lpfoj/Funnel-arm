// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ps5_desktop/guest_contract.h"

namespace Funnel::Ps5Desktop {

GuestContract QueryGuestContract() noexcept {
    return {
        .abi_version = GuestContractAbiVersion,
        .guest_page_size = GuestPageSize,
        .guest_address_limit = GuestAddressLimit,
        .required_cpu_state =
            CpuStateGprAndFlags | CpuStateFsGsBase | CpuStateX87 | CpuStateYmm,
        .required_instructions =
            InstructionSse4aExtrqInsertq | InstructionMonitorxMwaitx,
    };
}

}
