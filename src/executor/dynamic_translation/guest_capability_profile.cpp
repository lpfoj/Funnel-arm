// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/guest_capability_profile.h"

namespace Lsx4::Translation {
namespace {

struct CapabilityRecord {
    std::uint32_t page;
    GuestCapabilityPage value;
    bool selector_must_be_zero;
};

constexpr std::array kProfile{
    CapabilityRecord{0x00000000u,
                     {{{0x00000007u, 0x68747541u, 0x444d4163u, 0x69746e65u}}},
                     false},
    CapabilityRecord{0x00000001u,
                     {{{0x00630f01u, 0x00000800u, 0x3ed80201u, 0x07808131u}}},
                     false},
    CapabilityRecord{0x00000007u,
                     {{{0x00000000u, 0x00002128u, 0x00000000u, 0x00000000u}}},
                     true},
    CapabilityRecord{0x80000000u,
                     {{{0x80000008u, 0x00000000u, 0x00000000u, 0x00000000u}}},
                     false},
    CapabilityRecord{0x80000001u,
                     {{{0x00000000u, 0x00000000u, 0x00000020u, 0x2c100800u}}},
                     false},
    CapabilityRecord{0x80000008u,
                     {{{0x00003030u, 0x00000000u, 0x00000000u, 0x00000000u}}},
                     false},
};

} // namespace

GuestCapabilityPage QueryGuestCapabilityPage(const std::uint32_t page,
                                             const std::uint32_t selector) noexcept {
    for (const CapabilityRecord& record : kProfile) {
        if (record.page == page &&
            (!record.selector_must_be_zero || selector == 0)) {
            return record.value;
        }
    }
    return {};
}

} // namespace Lsx4::Translation
