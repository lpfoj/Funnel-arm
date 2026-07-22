// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/native_compiler.h"

#include "executor/dynamic_translation/native_encoder.h"
#include "executor/dynamic_translation/value_residency.h"

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>

namespace Lsx4::Translation {
namespace {

constexpr std::size_t Align16(const std::size_t value) noexcept {
    return (value + 15u) & ~std::size_t{15};
}

ArmIntegerWidth NativeWidth(const IrType type) noexcept {
    return type == IrType::Integer64 || type == IrType::Address ||
                   type == IrType::FlagSet
               ? ArmIntegerWidth::Bits64
               : ArmIntegerWidth::Bits32;
}

std::uint8_t StorageBytes(const std::uint8_t width) noexcept {
    return width == 8 ? 1 : width == 16 ? 2 : width == 32 ? 4 : width == 64 ? 8 : 0;
}

std::uint64_t CalculateFlagWord(const std::uint64_t existing, const std::uint64_t left,
                                const std::uint64_t right, const std::uint64_t result,
                                const std::uint64_t metadata) noexcept {
    const std::uint32_t width = static_cast<std::uint32_t>(metadata & 0xffu);
    const auto formula = static_cast<FlagFormula>((metadata >> 8) & 0xffu);
    ArithmeticFlags calculated{};
    switch (formula) {
    case FlagFormula::Addition:
    case FlagFormula::Increment:
        calculated = FlagsForAddition(left, right, width);
        break;
    case FlagFormula::Subtraction:
    case FlagFormula::Decrement:
        calculated = FlagsForSubtraction(left, right, width);
        break;
    case FlagFormula::Logical: {
        const std::uint64_t narrowed = result & WidthMask(width);
        const std::uint64_t sign = std::uint64_t{1} << (width - 1);
        calculated.bits = EvenLowByteParity(narrowed) ? ParityFlag : 0;
        calculated.bits |= narrowed == 0 ? ZeroFlag : 0;
        calculated.bits |= (narrowed & sign) != 0 ? SignFlag : 0;
        break;
    }
    }
    if (formula == FlagFormula::Increment || formula == FlagFormula::Decrement) {
        calculated.bits = (calculated.bits & ~CarryFlag) | (existing & CarryFlag);
    }
    return (existing & ~ArithmeticFlagMask) | calculated.bits;
}

struct TransferContext {
    ArmWordStream* stream{};
    bool failed{};
};

std::uint16_t SpillOffset(const ValueIdentity value) noexcept {
    return static_cast<std::uint16_t>(16u + (static_cast<std::size_t>(value.number) - 1u) * 8u);
}

ArmRegister ToArmRegister(const HostRegister host) noexcept {
    return static_cast<ArmRegister>(host.number);
}

void ObserveTransfer(const ResidencyTransfer& transfer, void* const opaque) {
    auto& context = *static_cast<TransferContext*>(opaque);
    if (transfer.value.number == 0) {
        context.failed = true;
        return;
    }
    const std::uint16_t offset = SpillOffset(transfer.value);
    const bool encoded = transfer.direction == TransferDirection::Fill
                             ? context.stream->Load(ToArmRegister(transfer.host),
                                                    ArmRegister::ZeroOrStack, offset, 8)
                             : context.stream->Store(ToArmRegister(transfer.host),
                                                     ArmRegister::ZeroOrStack, offset, 8);
    context.failed = context.failed || !encoded;
}

std::size_t NextUse(const std::span<const IrStep> steps, const std::size_t after,
                    const std::uint32_t value_id) noexcept {
    for (std::size_t index = after + 1; index < steps.size(); ++index) {
        for (std::size_t input = 0; input < steps[index].input_count; ++input) {
            if (steps[index].inputs[input].id == value_id) {
                return index;
            }
        }
    }
    return ValueResidency::NoFutureUse;
}

class CompilerSession {
public:
    explicit CompilerSession(const TranslationProgram& program)
        : program_{program}, steps_{program.ir.Steps()}, transfer_{&stream_},
          residency_{kIntegerHosts, {}, ObserveTransfer, &transfer_} {}

    NativeCompileFailure Build() {
        if (!program_.ir.Validate()) {
            return NativeCompileFailure::InvalidIr;
        }
        const auto lifetimes = program_.ir.AnalyzeLifetimes();
        if (lifetimes.size() > std::numeric_limits<std::uint16_t>::max()) {
            return NativeCompileFailure::FrameTooLarge;
        }
        stack_bytes_ = Align16(16u + (lifetimes.size() - 1u) * 8u);
        if (stack_bytes_ > 4080) {
            return NativeCompileFailure::FrameTooLarge;
        }
        if (!stream_.SubtractImmediate(ArmRegister::ZeroOrStack,
                                       ArmRegister::ZeroOrStack,
                                       static_cast<std::uint16_t>(stack_bytes_)) ||
            !stream_.Store(ArmRegister::X19, ArmRegister::ZeroOrStack, 0, 8) ||
            !stream_.Store(ArmRegister::X30, ArmRegister::ZeroOrStack, 8, 8)) {
            return NativeCompileFailure::EncodingFailure;
        }
        stream_.CopyRegister(ArmRegister::X19, ArmRegister::X0);

        for (std::size_t index = 0; index < steps_.size(); ++index) {
            const NativeCompileFailure failure = EmitStep(index, steps_[index]);
            if (failure != NativeCompileFailure::None) {
                return failure;
            }
        }
        if (!has_exit_ || transfer_.failed || !stream_.Finalize()) {
            return NativeCompileFailure::EncodingFailure;
        }
        return NativeCompileFailure::None;
    }

    ArmWordStream& Stream() noexcept {
        return stream_;
    }

private:
    std::optional<ArmRegister> Acquire(const IrValue value, const ValueAccess access,
                                       const std::size_t step, const bool pin = false) {
        if (!value.Exists() || value.id > std::numeric_limits<std::uint16_t>::max() ||
            !IsIntegerType(value.type)) {
            return std::nullopt;
        }
        const auto host = residency_.Acquire(
            {ValueBank::Integer, static_cast<std::uint16_t>(value.id)}, access,
            NextUse(steps_, step, value.id), pin);
        return host ? std::optional<ArmRegister>{ToArmRegister(*host)} : std::nullopt;
    }

    void ReleaseInputs(const IrStep& step, const std::size_t index) noexcept {
        for (std::size_t input = 0; input < step.input_count; ++input) {
            const ValueIdentity identity{ValueBank::Integer,
                                         static_cast<std::uint16_t>(step.inputs[input].id)};
            residency_.SetPinned(identity, false);
            if (NextUse(steps_, index, step.inputs[input].id) ==
                ValueResidency::NoFutureUse) {
                residency_.Forget(identity);
            }
        }
        if (step.result.Exists() && NextUse(steps_, index, step.result.id) ==
                                        ValueResidency::NoFutureUse) {
            residency_.Forget({ValueBank::Integer,
                               static_cast<std::uint16_t>(step.result.id)});
        }
    }

    NativeCompileFailure EmitStep(const std::size_t index, const IrStep& step) {
        if (has_exit_) {
            return NativeCompileFailure::InvalidIr;
        }
        NativeCompileFailure result = NativeCompileFailure::None;
        switch (step.action) {
        case IrAction::Constant:
            result = EmitConstant(index, step);
            break;
        case IrAction::ReadInteger:
            result = EmitReadInteger(index, step);
            break;
        case IrAction::WriteInteger:
            result = EmitWriteInteger(index, step);
            break;
        case IrAction::Add:
        case IrAction::Subtract:
        case IrAction::BitAnd:
        case IrAction::BitOr:
        case IrAction::BitXor:
            result = EmitBinary(index, step);
            break;
        case IrAction::ComposeFlags:
            result = EmitFlags(index, step);
            break;
        case IrAction::WriteFlags:
            result = EmitWriteFlags(index, step);
            break;
        case IrAction::Barrier:
            residency_.SpillAll();
            result = transfer_.failed ? NativeCompileFailure::EncodingFailure
                                      : NativeCompileFailure::None;
            break;
        case IrAction::ExitDirect:
            result = EmitDirectExit(index, step);
            break;
        default:
            result = NativeCompileFailure::UnsupportedAction;
            break;
        }
        if (result == NativeCompileFailure::None) {
            ReleaseInputs(step, index);
        }
        return result;
    }

    NativeCompileFailure EmitConstant(const std::size_t index, const IrStep& step) {
        const auto destination = Acquire(step.result, ValueAccess::Write, index);
        if (!destination) {
            return NativeCompileFailure::RegisterPressure;
        }
        stream_.PutConstant(*destination, step.immediate, NativeWidth(step.result.type));
        return NativeCompileFailure::None;
    }

    NativeCompileFailure EmitReadInteger(const std::size_t index, const IrStep& step) {
        const IntegerAccess access = UnpackIntegerAccess(step.detail);
        const std::uint8_t bytes = StorageBytes(access.width);
        const auto destination = Acquire(step.result, ValueAccess::Write, index);
        if (!destination || access.slot >= 16 || bytes == 0 || access.upper_byte ||
            !stream_.Load(*destination, ArmRegister::X19,
                          static_cast<std::uint16_t>(access.slot * 8u), bytes)) {
            return destination ? NativeCompileFailure::UnsupportedAction
                               : NativeCompileFailure::RegisterPressure;
        }
        return NativeCompileFailure::None;
    }

    NativeCompileFailure EmitWriteInteger(const std::size_t index, const IrStep& step) {
        const IntegerAccess access = UnpackIntegerAccess(step.detail);
        std::uint8_t bytes = StorageBytes(access.width);
        const auto source = Acquire(step.inputs[0], ValueAccess::Read, index, true);
        if (access.width == 32) {
            bytes = 8;
        }
        if (!source) {
            return NativeCompileFailure::RegisterPressure;
        }
        if (access.slot >= 16 || bytes == 0 || access.upper_byte ||
            !stream_.Store(*source, ArmRegister::X19,
                           static_cast<std::uint16_t>(access.slot * 8u), bytes)) {
            return NativeCompileFailure::UnsupportedAction;
        }
        return NativeCompileFailure::None;
    }

    NativeCompileFailure EmitBinary(const std::size_t index, const IrStep& step) {
        const auto left = Acquire(step.inputs[0], ValueAccess::Read, index, true);
        const auto right = Acquire(step.inputs[1], ValueAccess::Read, index, true);
        const auto destination = Acquire(step.result, ValueAccess::Write, index);
        if (!left || !right || !destination) {
            return NativeCompileFailure::RegisterPressure;
        }
        const ArmIntegerWidth width = NativeWidth(step.result.type);
        switch (step.action) {
        case IrAction::Add:
            stream_.AddRegisters(*destination, *left, *right, width);
            break;
        case IrAction::Subtract:
            stream_.SubtractRegisters(*destination, *left, *right, width);
            break;
        case IrAction::BitAnd:
            stream_.AndRegisters(*destination, *left, *right, width);
            break;
        case IrAction::BitOr:
            stream_.OrRegisters(*destination, *left, *right, width);
            break;
        case IrAction::BitXor:
            stream_.XorRegisters(*destination, *left, *right, width);
            break;
        default:
            return NativeCompileFailure::UnsupportedAction;
        }
        return NativeCompileFailure::None;
    }

    NativeCompileFailure EmitFlags(const std::size_t index, const IrStep& step) {
        residency_.SpillAll();
        if (transfer_.failed) {
            return NativeCompileFailure::EncodingFailure;
        }
        residency_.CrossCallBoundary(kClobberedHosts);
        if (transfer_.failed ||
            !stream_.Load(ArmRegister::X0, ArmRegister::X19,
                          static_cast<std::uint16_t>(offsetof(CpuFrame, condition_word)), 8)) {
            return NativeCompileFailure::EncodingFailure;
        }
        for (std::size_t input = 0; input < 3; ++input) {
            if (!stream_.Load(static_cast<ArmRegister>(input + 1),
                              ArmRegister::ZeroOrStack,
                              SpillOffset({ValueBank::Integer,
                                           static_cast<std::uint16_t>(step.inputs[input].id)}),
                              8)) {
                return NativeCompileFailure::EncodingFailure;
            }
        }
        const std::uint64_t metadata = IrTypeBitWidth(step.inputs[2].type) |
                                       (static_cast<std::uint64_t>(step.detail) << 8);
        stream_.PutConstant(ArmRegister::X4, metadata);
        stream_.PutConstant(
            ArmRegister::X16,
            reinterpret_cast<std::uintptr_t>(&CalculateFlagWord));
        stream_.CallRegister(ArmRegister::X16);
        const auto destination = Acquire(step.result, ValueAccess::Write, index);
        if (!destination) {
            return NativeCompileFailure::RegisterPressure;
        }
        stream_.CopyRegister(*destination, ArmRegister::X0);
        return NativeCompileFailure::None;
    }

    NativeCompileFailure EmitWriteFlags(const std::size_t index, const IrStep& step) {
        const auto source = Acquire(step.inputs[0], ValueAccess::Read, index, true);
        if (!source) {
            return NativeCompileFailure::RegisterPressure;
        }
        return stream_.Store(*source, ArmRegister::X19,
                             static_cast<std::uint16_t>(offsetof(CpuFrame, condition_word)), 8)
                   ? NativeCompileFailure::None
                   : NativeCompileFailure::EncodingFailure;
    }

    NativeCompileFailure EmitDirectExit(const std::size_t index, const IrStep& step) {
        (void)index;
        stream_.PutConstant(ArmRegister::X0, step.immediate);
        if (!stream_.Store(ArmRegister::X0, ArmRegister::X19,
                           static_cast<std::uint16_t>(offsetof(CpuFrame, resume_address)), 8) ||
            !stream_.Load(ArmRegister::X19, ArmRegister::ZeroOrStack, 0, 8) ||
            !stream_.Load(ArmRegister::X30, ArmRegister::ZeroOrStack, 8, 8) ||
            !stream_.AddImmediate(ArmRegister::ZeroOrStack,
                                  ArmRegister::ZeroOrStack,
                                  static_cast<std::uint16_t>(stack_bytes_))) {
            return NativeCompileFailure::EncodingFailure;
        }
        stream_.Return();
        has_exit_ = true;
        return NativeCompileFailure::None;
    }

    static constexpr std::array<std::uint8_t, 10> kIntegerHosts{
        5, 6, 7, 8, 10, 11, 12, 13, 14, 15};
    static constexpr std::array<HostRegister, 10> kClobberedHosts{
        HostRegister{ValueBank::Integer, 5}, HostRegister{ValueBank::Integer, 6},
        HostRegister{ValueBank::Integer, 7}, HostRegister{ValueBank::Integer, 8},
        HostRegister{ValueBank::Integer, 10}, HostRegister{ValueBank::Integer, 11},
        HostRegister{ValueBank::Integer, 12}, HostRegister{ValueBank::Integer, 13},
        HostRegister{ValueBank::Integer, 14}, HostRegister{ValueBank::Integer, 15}};

    const TranslationProgram& program_;
    std::span<const IrStep> steps_{};
    ArmWordStream stream_{};
    TransferContext transfer_{};
    ValueResidency residency_;
    std::size_t stack_bytes_{};
    bool has_exit_{};
};

} // namespace

NativeArtifact CompileScalarIr(const TranslationProgram& program, ExecutableArena& arena) {
    CompilerSession compiler{program};
    const NativeCompileFailure failure = compiler.Build();
    if (failure != NativeCompileFailure::None) {
        return {.failure = failure};
    }
    ArmWordStream& stream = compiler.Stream();
    CodeReservation reservation = arena.Reserve(stream.ByteSize(), 16);
    if (!reservation.IsValid()) {
        return {.failure = NativeCompileFailure::ArenaExhausted};
    }
    if (!stream.CopyTo(reservation.writable)) {
        return {.failure = NativeCompileFailure::EncodingFailure};
    }
    arena.Publish(reservation);
    const auto entry = reinterpret_cast<NativeRegionEntry>(
        const_cast<std::uint8_t*>(reservation.executable.data()));
    return {entry, reservation, NativeCompileFailure::None};
}

} // namespace Lsx4::Translation
