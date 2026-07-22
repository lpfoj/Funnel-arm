// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/foundation_checks.h"

#include "executor/dynamic_translation/code_memory.h"
#include "executor/dynamic_translation/code_witness.h"
#include "executor/dynamic_translation/edge_portal.h"
#include "executor/dynamic_translation/machine_state.h"
#include "executor/dynamic_translation/native_encoder.h"
#include "executor/dynamic_translation/native_compiler.h"
#include "executor/dynamic_translation/operation_ir.h"
#include "executor/dynamic_translation/region_plan.h"
#include "executor/dynamic_translation/region_registry.h"
#include "executor/dynamic_translation/scalar_ir_builder.h"
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
    NativeEncoding = 1ull << 7,
    IntermediateRepresentation = 1ull << 8,
    ScalarIrBuilder = 1ull << 9,
    NativeIrCompiler = 1ull << 10,
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

bool CheckNativeEncoding() {
    ArmWordStream stream{};
    const CodeLabel taken = stream.NewLabel();
    stream.PutConstant(ArmRegister::X0, 0);
    stream.JumpIfZero(ArmRegister::X0, taken);
    stream.PutConstant(ArmRegister::X0, 99);
    stream.Return();
    if (!stream.Place(taken)) {
        return false;
    }
    stream.PutConstant(ArmRegister::X0, 42);
    stream.Return();
    if (!stream.Finalize()) {
        return false;
    }

    ExecutableArena arena{64 * 1024};
    CodeReservation reservation = arena.Reserve(stream.ByteSize(), 16);
    if (!reservation.IsValid() || !stream.CopyTo(reservation.writable)) {
        return false;
    }
    arena.Publish(reservation);
#if defined(__aarch64__)
    using GeneratedFunction = std::uint64_t (*)();
    const auto function = reinterpret_cast<GeneratedFunction>(
        const_cast<std::uint8_t*>(reservation.executable.data()));
    return function() == 42;
#else
    return true;
#endif
}

bool CheckIntermediateRepresentation() {
    OperationIr ir{0xb400};
    const IrValue left = ir.AddValue(IrAction::Constant, IrType::Integer64, {}, 17);
    const IrValue right = ir.AddValue(IrAction::Constant, IrType::Integer64, {}, 25);
    const std::array<IrValue, 2> operands{left, right};
    const IrValue sum = ir.AddValue(IrAction::Add, IrType::Integer64, operands);
    const std::array<IrValue, 1> destination{sum};
    if (!ir.AddEffect(IrAction::WriteInteger, destination, 0, 3) ||
        !ir.AddEffect(IrAction::ExitDirect, {}, 0xb410)) {
        return false;
    }
    const auto lifetimes = ir.AnalyzeLifetimes();
    return ir.Validate() && ir.Steps().size() == 5 && lifetimes[left.id].use_count == 1 &&
           lifetimes[right.id].last_use == 2 && lifetimes[sum.id].last_use == 3;
}

bool CheckScalarIrBuilder() {
    constexpr std::array<std::uint8_t, 9> fixture{
        0xb8, 0x05, 0x00, 0x00, 0x00, 0x83, 0xc0, 0x07, 0xc3};
    FetchContext context{0xc800, fixture};
    const RegionPlan plan = PlanRegion(context.first, 6, FetchFixture, &context);
    if (!plan.IsExecutable() || plan.sequence.size() != 3) {
        return false;
    }
    const TranslationProgram program = BuildScalarProgram(plan);
    return program.ir.Validate() && program.semantic_steps.empty() &&
           program.ir.Steps().back().action == IrAction::ExitReturn;
}

bool CheckNativeIrCompiler() {
    TranslationProgram program{0xd000};
    const IrValue left =
        program.ir.AddValue(IrAction::Constant, IrType::Integer32, {}, 0xffffffffu);
    const IrValue right =
        program.ir.AddValue(IrAction::Constant, IrType::Integer32, {}, 2);
    const std::array<IrValue, 2> operands{left, right};
    const IrValue sum =
        program.ir.AddValue(IrAction::Add, IrType::Integer32, operands);
    const std::array<IrValue, 1> sum_input{sum};
    if (!program.ir.AddEffect(IrAction::WriteInteger, sum_input, 0,
                              PackIntegerAccess({0, 32, false}))) {
        return false;
    }
    const std::array<IrValue, 3> flag_inputs{left, right, sum};
    const IrValue flags = program.ir.AddValue(
        IrAction::ComposeFlags, IrType::FlagSet, flag_inputs, 0,
        static_cast<std::uint32_t>(FlagFormula::Addition));
    const std::array<IrValue, 1> flags_input{flags};
    if (!program.ir.AddEffect(IrAction::WriteFlags, flags_input) ||
        !program.ir.AddEffect(IrAction::ExitDirect, {}, 0xd040)) {
        return false;
    }

    ExecutableArena arena{64 * 1024};
    const NativeArtifact artifact = CompileScalarIr(program, arena);
    if (!artifact.Succeeded()) {
        return false;
    }
#if defined(__aarch64__)
    CpuFrame frame{};
    frame.integer[0] = ~std::uint64_t{};
    const std::uint64_t next = artifact.entry(&frame);
    return next == 0xd040 && frame.resume_address == 0xd040 &&
           frame.integer[0] == 1 && (frame.condition_word & CarryFlag) != 0;
#else
    return true;
#endif
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
    run(NativeEncoding, CheckNativeEncoding());
    run(IntermediateRepresentation, CheckIntermediateRepresentation());
    run(ScalarIrBuilder, CheckScalarIrBuilder());
    run(NativeIrCompiler, CheckNativeIrCompiler());
    return report;
}

} // namespace Lsx4::Translation
