// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/foundation_checks.h"

#include "executor/dynamic_translation/code_memory.h"
#include "executor/dynamic_translation/code_witness.h"
#include "executor/dynamic_translation/edge_portal.h"
#include "executor/dynamic_translation/machine_state.h"
#include "executor/dynamic_translation/region_plan.h"
#include "executor/dynamic_translation/region_registry.h"
#include "executor/dynamic_translation/value_residency.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <vector>

namespace Lsx4::Translation {
namespace {

enum CheckBit : std::uint64_t {
    MachineState = 1ull << 0,
    DecodeAndPlan = 1ull << 1,
    ExecutableMemory = 1ull << 2,
    MutationWitness = 1ull << 3,
    OutgoingEdge = 1ull << 4,
    RegionPublication = 1ull << 5,
    ResidencyEvents = 1ull << 6,
};

struct FetchContext {
    std::uint64_t first{};
    std::span<const std::uint8_t> bytes{};
};

std::size_t FetchFixture(const std::uint64_t address,
                         const std::span<std::uint8_t> destination,
                         void* const opaque) noexcept {
    const auto& context = *static_cast<const FetchContext*>(opaque);
    if (address < context.first) {
        return 0;
    }
    const std::uint64_t offset64 = address - context.first;
    if (offset64 >= context.bytes.size()) {
        return 0;
    }
    const std::size_t offset = static_cast<std::size_t>(offset64);
    const std::size_t copied = std::min(destination.size(), context.bytes.size() - offset);
    std::memcpy(destination.data(), context.bytes.data() + offset, copied);
    return copied;
}

bool CheckMachineState() {
    CpuFrame frame{};
    WriteInteger(frame, IntegerRegister::A, 0x1122334455667788ull, 64);
    WriteInteger(frame, IntegerRegister::A, 0xaa, 8, true);
    if (ReadInteger(frame, IntegerRegister::A, 64) != 0x112233445566aa88ull) {
        return false;
    }
    WriteInteger(frame, IntegerRegister::A, 0xfedcba98u, 32);
    if (ReadInteger(frame, IntegerRegister::A, 64) != 0xfedcba98u) {
        return false;
    }
    ApplyAdditionFlags(frame, 0xff, 1, 8);
    return (frame.condition_word & (CarryFlag | ZeroFlag)) == (CarryFlag | ZeroFlag);
}

bool CheckDecodeAndPlan() {
    constexpr std::array<std::uint8_t, 3> fixture{0x31, 0xc0, 0xc3};
    FetchContext context{0x7800, fixture};
    RegionPlan plan = PlanRegion(context.first, 5, FetchFixture, &context);
    return plan.IsExecutable() && plan.sequence.size() == 2 &&
           plan.sequence.back().boundary == FlowBoundary::Return &&
           plan.continuation == context.first + fixture.size();
}

bool CheckExecutableMemory() {
    ExecutableArena arena{64 * 1024};
    CodeReservation reservation = arena.Reserve(24, 16);
    if (!reservation.IsValid()) {
        return false;
    }
    for (std::size_t index = 0; index < reservation.writable.size(); ++index) {
        reservation.writable[index] = static_cast<std::uint8_t>((index * 29u) ^ 0x5du);
    }
    arena.Publish(reservation);
    return arena.CommittedBytes() == reservation.writable.size() &&
           arena.Owns(reinterpret_cast<std::uintptr_t>(reservation.executable.data()));
}

bool CheckMutationWitness() {
    constexpr std::uint64_t guest_base = 0x1ff0;
    std::vector<std::uint8_t> image(96);
    for (std::size_t index = 0; index < image.size(); ++index) {
        image[index] = static_cast<std::uint8_t>(index * 7u + 3u);
    }
    const ReadGuestBytes reader = [&image](const std::uint64_t address,
                                           const std::span<std::uint8_t> destination) {
        if (address < guest_base || address - guest_base > image.size()) {
            return false;
        }
        const std::size_t offset = static_cast<std::size_t>(address - guest_base);
        if (destination.size() > image.size() - offset) {
            return false;
        }
        std::memcpy(destination.data(), image.data() + offset, destination.size());
        return true;
    };

    MutationLedger ledger{};
    auto original = ledger.Capture(guest_base + 8, 40, reader);
    if (!original || !ledger.Validate(*original, reader)) {
        return false;
    }
    image[13] ^= 0x80;
    if (ledger.Validate(*original, reader)) {
        return false;
    }
    auto refreshed = ledger.Capture(guest_base + 8, 40, reader);
    ledger.MarkChanged(guest_base + 16, 1);
    return refreshed && !ledger.Validate(*refreshed, reader);
}

bool CheckOutgoingEdge() {
    ExitPortal portal{0x8a00, EdgePolicy::Chainable};
    EdgeDirectory directory{};
    directory.Attach(portal);
    directory.Announce(0x8a00, 0x440000, 19, 6);
    const ResolvedEdge exact = portal.Resolve(19, 6);
    const ResolvedEdge stale = portal.Resolve(18, 6);
    directory.Revoke(0x8a00);
    const ResolvedEdge revoked = portal.Resolve(19, 6);
    directory.Detach(portal);
    return exact.host_address == 0x440000 && stale.host_address == 0 &&
           revoked.host_address == 0;
}

bool CheckRegionPublication() {
    RegionPayload payload{};
    payload.plan.first_address = 0xa200;
    payload.plan.continuation = 0xa201;
    payload.plan.sequence.emplace_back();
    payload.plan.stop = RegionStop::FlowBoundary;
    payload.source_identity.guest_address = payload.plan.first_address;
    payload.source_identity.byte_count = 1;
    payload.entry_address = 0x600000;

    RegionRegistry registry{};
    const auto published = registry.Install(std::move(payload));
    const auto found = registry.Find(0xa200);
    const auto removed = registry.Remove(0xa200);
    return published && found == published && removed == published &&
           !registry.Find(0xa200) && registry.ActiveCount() == 0;
}

bool CheckResidencyEvents() {
    constexpr std::array<std::uint8_t, 2> integer_hosts{7, 9};
    constexpr std::array<std::uint8_t, 1> vector_hosts{4};
    std::vector<ResidencyTransfer> events{};
    const auto observe = [](const ResidencyTransfer& transfer, void* context) {
        static_cast<std::vector<ResidencyTransfer>*>(context)->push_back(transfer);
    };
    ValueResidency residency{integer_hosts, vector_hosts, observe, &events};
    const ValueIdentity far_dirty{ValueBank::Integer, 1};
    const ValueIdentity near_clean{ValueBank::Integer, 2};
    const ValueIdentity incoming{ValueBank::Integer, 3};
    if (!residency.Acquire(far_dirty, ValueAccess::Write, 100) ||
        !residency.Acquire(near_clean, ValueAccess::Read, 2) ||
        !residency.Acquire(incoming, ValueAccess::Read, 3)) {
        return false;
    }
    return events.size() == 3 && events[0].direction == TransferDirection::Fill &&
           events[0].value == near_clean &&
           events[1].direction == TransferDirection::Spill &&
           events[1].value == far_dirty &&
           events[2].direction == TransferDirection::Fill &&
           events[2].value == incoming;
}

} // namespace

FoundationCheckReport RunFoundationChecks() {
    FoundationCheckReport report{};
    const auto run = [&report](const std::uint64_t bit, const bool passed) {
        report.completed |= bit;
        if (!passed) {
            report.failed |= bit;
        }
    };
    run(MachineState, CheckMachineState());
    run(DecodeAndPlan, CheckDecodeAndPlan());
    run(ExecutableMemory, CheckExecutableMemory());
    run(MutationWitness, CheckMutationWitness());
    run(OutgoingEdge, CheckOutgoingEdge());
    run(RegionPublication, CheckRegionPublication());
    run(ResidencyEvents, CheckResidencyEvents());
    return report;
}

} // namespace Lsx4::Translation
