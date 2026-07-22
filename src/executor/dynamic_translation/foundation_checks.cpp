// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/foundation_checks.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/code_memory.h"
#include "executor/dynamic_translation/code_witness.h"
#include "executor/dynamic_translation/edge_portal.h"
#include "executor/dynamic_translation/machine_state.h"
#include "executor/dynamic_translation/integer_math.h"
#include "executor/dynamic_translation/native_encoder.h"
#include "executor/dynamic_translation/native_compiler.h"
#include "executor/dynamic_translation/operation_ir.h"
#include "executor/dynamic_translation/region_plan.h"
#include "executor/dynamic_translation/region_registry.h"
#include "executor/dynamic_translation/scalar_ir_builder.h"
#include "executor/dynamic_translation/scalar_semantics.h"
#include "executor/dynamic_translation/value_residency.h"
#include "executor/dynamic_translation/vector_semantics.h"

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
    ScalarSemantics = 1ull << 11,
    IntegerMath = 1ull << 12,
    AtomicSemantics = 1ull << 13,
    VectorSemantics = 1ull << 14,
    VectorPacking = 1ull << 15,
    VectorShifts = 1ull << 16,
};

struct FetchContext {
    std::uint64_t first{};
    std::span<const std::uint8_t> bytes{};
};

struct SemanticMemoryImage {
    std::uint64_t base{};
    std::array<std::uint8_t, 128> bytes{};
};

bool ReadSemanticMemory(const std::uint64_t address,
                        const std::span<std::uint8_t> destination,
                        void* const opaque) noexcept {
    const auto& image = *static_cast<const SemanticMemoryImage*>(opaque);
    if (address < image.base || address - image.base > image.bytes.size()) {
        return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - image.base);
    if (destination.size() > image.bytes.size() - offset) {
        return false;
    }
    std::memcpy(destination.data(), image.bytes.data() + offset, destination.size());
    return true;
}

bool WriteSemanticMemory(const std::uint64_t address,
                         const std::span<const std::uint8_t> source,
                         void* const opaque) noexcept {
    auto& image = *static_cast<SemanticMemoryImage*>(opaque);
    if (address < image.base || address - image.base > image.bytes.size()) {
        return false;
    }
    const std::size_t offset = static_cast<std::size_t>(address - image.base);
    if (source.size() > image.bytes.size() - offset) {
        return false;
    }
    std::memcpy(image.bytes.data() + offset, source.data(), source.size());
    return true;
}

bool CompareExchangeSemanticMemory(const std::uint64_t address,
                                   const std::uint32_t width,
                                   const std::uint64_t expected,
                                   const std::uint64_t desired,
                                   std::uint64_t& observed,
                                   void* const opaque) noexcept {
    const std::size_t size = width / 8;
    if (size == 0 || size > sizeof(observed)) {
        return false;
    }
    std::array<std::uint8_t, 8> bytes{};
    if (!ReadSemanticMemory(address, std::span<std::uint8_t>{bytes.data(), size},
                            opaque)) {
        return false;
    }
    observed = 0;
    for (std::size_t index = 0; index < size; ++index) {
        observed |= static_cast<std::uint64_t>(bytes[index]) << (index * 8u);
    }
    if (observed != (expected & WidthMask(width))) {
        return true;
    }
    for (std::size_t index = 0; index < size; ++index) {
        bytes[index] = static_cast<std::uint8_t>(desired >> (index * 8u));
    }
    return WriteSemanticMemory(address,
                               std::span<const std::uint8_t>{bytes.data(), size},
                               opaque);
}

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

bool CheckScalarSemantics() {
    constexpr std::array<std::uint8_t, 18> fixture{
        0xb8, 0xff, 0xff, 0xff, 0xff, 0x83, 0xc0, 0x02, 0x50,
        0x59, 0x48, 0x83, 0xf9, 0x01, 0x74, 0x02, 0x31, 0xd2};
    constexpr std::uint64_t code_base = 0xe000;
    SemanticMemoryImage image{.base = 0x120000};
    GuestMemoryPort memory{ReadSemanticMemory, WriteSemanticMemory, &image};
    CpuFrame frame{};
    frame.integer[RegisterIndex(IntegerRegister::Stack)] = image.base + 96;
    const std::uint64_t original_stack = frame.integer[RegisterIndex(IntegerRegister::Stack)];

    std::size_t offset = 0;
    SemanticResult last{};
    unsigned executed = 0;
    while (offset < fixture.size() && executed < 6) {
        Instruction instruction{};
        if (DecodeInstruction(std::span<const std::uint8_t>{fixture}.subspan(offset),
                              code_base + offset, instruction) != DecodeOutcome::Complete) {
            return false;
        }
        last = ExecuteScalarInstruction(instruction, frame, memory);
        offset += instruction.length;
        ++executed;
        if (last.stop != SemanticStop::Continue) {
            break;
        }
    }
    std::uint64_t stacked = 0;
    std::memcpy(&stacked, image.bytes.data() + 88, sizeof(stacked));
    return executed == 6 && last.stop == SemanticStop::Transfer &&
           last.next_address == code_base + fixture.size() && frame.integer[0] == 1 &&
           frame.integer[1] == 1 && frame.integer[4] == original_stack && stacked == 1 &&
           (frame.condition_word & ZeroFlag) != 0 &&
           (frame.condition_word & CarryFlag) == 0;
}

bool CheckIntegerMath() {
    const IntegerMathResult adc = AddWithCarry(0xffffffffu, 0, true, 32);
    const IntegerMathResult sbb = SubtractWithBorrow(0, 0, true, 16);
    const auto rotate = EvaluateBitMovement(X86_MNEMONIC_ROL, 0x81, 1, 8);
    const auto arithmetic = EvaluateBitMovement(X86_MNEMONIC_SAR, 0x8000, 4, 16);
    return adc.value == 0 && (adc.flag_bits & (CarryFlag | ZeroFlag)) ==
                                 (CarryFlag | ZeroFlag) &&
           sbb.value == 0xffff && (sbb.flag_bits & CarryFlag) != 0 && rotate &&
           rotate->value == 0x03 && (rotate->flag_bits & CarryFlag) != 0 &&
           arithmetic && arithmetic->value == 0xf800 &&
           (arithmetic->flag_bits & SignFlag) != 0;
}

bool CheckAtomicSemantics() {
    constexpr std::array<std::uint8_t, 14> fixture{
        0xf0, 0x0f, 0xc1, 0x07, 0xf0, 0x0f, 0xb1,
        0x0f, 0xf0, 0x0f, 0xb1, 0x0f, 0x87, 0x17};
    constexpr std::uint64_t code_base = 0xe800;
    SemanticMemoryImage image{.base = 0x130000};
    const std::uint32_t initial = 10;
    std::memcpy(image.bytes.data(), &initial, sizeof(initial));
    GuestMemoryPort memory{ReadSemanticMemory, WriteSemanticMemory, &image,
                           CompareExchangeSemanticMemory};
    CpuFrame frame{};
    WriteInteger(frame, IntegerRegister::A, 5, 32);
    WriteInteger(frame, IntegerRegister::C, 21, 32);
    WriteInteger(frame, IntegerRegister::D, 7, 32);
    WriteInteger(frame, IntegerRegister::Destination, image.base, 64);

    std::size_t offset = 0;
    for (unsigned executed = 0; executed < 4; ++executed) {
        Instruction instruction{};
        if (DecodeInstruction(std::span<const std::uint8_t>{fixture}.subspan(offset),
                              code_base + offset, instruction) !=
            DecodeOutcome::Complete) {
            return false;
        }
        const SemanticResult result = ExecuteScalarInstruction(instruction, frame, memory);
        if (result.stop != SemanticStop::Continue) {
            return false;
        }
        offset += instruction.length;
    }
    std::uint32_t final_memory = 0;
    std::memcpy(&final_memory, image.bytes.data(), sizeof(final_memory));
    return offset == fixture.size() && final_memory == 7 &&
           ReadInteger(frame, IntegerRegister::A, 32) == 15 &&
           ReadInteger(frame, IntegerRegister::C, 32) == 21 &&
           ReadInteger(frame, IntegerRegister::D, 32) == 21 &&
           (frame.condition_word & ZeroFlag) != 0;
}

bool CheckVectorSemantics() {
    constexpr std::array<std::uint8_t, 25> fixture{
        0xf3, 0x0f, 0x6f, 0x07, 0xf3, 0x0f, 0x6f, 0x0e, 0x66,
        0x0f, 0xfc, 0xc1, 0x66, 0x0f, 0x70, 0xd0, 0x1b, 0x66,
        0x0f, 0xef, 0xd1, 0xf3, 0x0f, 0x7f, 0x12};
    constexpr std::uint64_t code_base = 0xec00;
    SemanticMemoryImage image{.base = 0x140000};
    for (std::size_t index = 0; index < 16; ++index) {
        image.bytes[index] = static_cast<std::uint8_t>(index);
        image.bytes[32 + index] = 1;
    }
    GuestMemoryPort memory{ReadSemanticMemory, WriteSemanticMemory, &image};
    CpuFrame frame{};
    WriteInteger(frame, IntegerRegister::Destination, image.base, 64);
    WriteInteger(frame, IntegerRegister::Source, image.base + 32, 64);
    WriteInteger(frame, IntegerRegister::D, image.base + 64, 64);

    std::size_t offset = 0;
    for (unsigned executed = 0; executed < 6; ++executed) {
        Instruction instruction{};
        if (DecodeInstruction(std::span<const std::uint8_t>{fixture}.subspan(offset),
                              code_base + offset, instruction) !=
            DecodeOutcome::Complete) {
            return false;
        }
        const VectorSemanticResult result =
            ExecuteVectorInstruction(instruction, frame, memory);
        if (result.stop != VectorStop::Continue) {
            return false;
        }
        offset += instruction.length;
    }
    for (std::size_t output = 0; output < 16; ++output) {
        const std::size_t source = 12 - (output / 4) * 4 + output % 4;
        const std::uint8_t expected =
            static_cast<std::uint8_t>(source + 1) ^ std::uint8_t{1};
        if (image.bytes[64 + output] != expected) {
            return false;
        }
    }
    return offset == fixture.size();
}

bool CheckVectorPacking() {
    constexpr std::array<std::uint8_t, 20> fixture{
        0xf3, 0x0f, 0x6f, 0x07, 0xf3, 0x0f, 0x6f, 0x0e, 0x66, 0x0f,
        0x63, 0xc1, 0x66, 0x0f, 0x74, 0x02, 0xf3, 0x0f, 0x7f, 0x01};
    constexpr std::array<std::int16_t, 8> left{
        -200, -128, -1, 0, 1, 127, 128, 300};
    constexpr std::array<std::int16_t, 8> right{
        32767, -32768, 42, -42, 255, -255, 7, 8};
    constexpr std::array<std::uint8_t, 16> expected{
        0x80, 0x80, 0xff, 0x00, 0x01, 0x7f, 0x7f, 0x7f,
        0x7f, 0x80, 0x2a, 0xd6, 0x7f, 0x80, 0x07, 0x08};
    constexpr std::uint64_t code_base = 0xee00;
    SemanticMemoryImage image{.base = 0x150000};
    std::memcpy(image.bytes.data(), left.data(), sizeof(left));
    std::memcpy(image.bytes.data() + 32, right.data(), sizeof(right));
    std::memcpy(image.bytes.data() + 64, expected.data(), expected.size());
    GuestMemoryPort memory{ReadSemanticMemory, WriteSemanticMemory, &image};
    CpuFrame frame{};
    WriteInteger(frame, IntegerRegister::Destination, image.base, 64);
    WriteInteger(frame, IntegerRegister::Source, image.base + 32, 64);
    WriteInteger(frame, IntegerRegister::D, image.base + 64, 64);
    WriteInteger(frame, IntegerRegister::C, image.base + 96, 64);

    std::size_t offset = 0;
    for (unsigned executed = 0; executed < 5; ++executed) {
        Instruction instruction{};
        if (DecodeInstruction(std::span<const std::uint8_t>{fixture}.subspan(offset),
                              code_base + offset, instruction) !=
            DecodeOutcome::Complete) {
            return false;
        }
        if (ExecuteVectorInstruction(instruction, frame, memory).stop !=
            VectorStop::Continue) {
            return false;
        }
        offset += instruction.length;
    }
    for (std::size_t index = 0; index < 16; ++index) {
        if (image.bytes[96 + index] != 0xff) {
            return false;
        }
    }
    return offset == fixture.size();
}

bool CheckVectorShifts() {
    constexpr std::array<std::uint8_t, 18> fixture{
        0xf3, 0x0f, 0x6f, 0x07, 0x66, 0x0f, 0x71, 0xe0, 0x04,
        0x66, 0x0f, 0x71, 0xf0, 0x01, 0xf3, 0x0f, 0x7f, 0x06};
    constexpr std::array<std::uint16_t, 8> source{
        0x8001, 0xffff, 0x7fff, 0x0010, 0xfff0, 0x1234, 0xedcc, 0x0000};
    constexpr std::array<std::uint16_t, 8> expected{
        0xf000, 0xfffe, 0x0ffe, 0x0002, 0xfffe, 0x0246, 0xfdb8, 0x0000};
    constexpr std::uint64_t code_base = 0xef00;
    SemanticMemoryImage image{.base = 0x160000};
    std::memcpy(image.bytes.data(), source.data(), sizeof(source));
    GuestMemoryPort memory{ReadSemanticMemory, WriteSemanticMemory, &image};
    CpuFrame frame{};
    WriteInteger(frame, IntegerRegister::Destination, image.base, 64);
    WriteInteger(frame, IntegerRegister::Source, image.base + 32, 64);

    std::size_t offset = 0;
    for (unsigned executed = 0; executed < 4; ++executed) {
        Instruction instruction{};
        if (DecodeInstruction(std::span<const std::uint8_t>{fixture}.subspan(offset),
                              code_base + offset, instruction) !=
            DecodeOutcome::Complete ||
            ExecuteVectorInstruction(instruction, frame, memory).stop !=
                VectorStop::Continue) {
            return false;
        }
        offset += instruction.length;
    }
    return offset == fixture.size() &&
           std::memcmp(image.bytes.data() + 32, expected.data(), sizeof(expected)) == 0;
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
    run(ScalarSemantics, CheckScalarSemantics());
    run(IntegerMath, CheckIntegerMath());
    run(AtomicSemantics, CheckAtomicSemantics());
    run(VectorSemantics, CheckVectorSemantics());
    run(VectorPacking, CheckVectorPacking());
    run(VectorShifts, CheckVectorShifts());
    return report;
}

} // namespace Lsx4::Translation
