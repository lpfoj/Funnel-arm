// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/operation_ir.h"

#include <algorithm>

namespace Lsx4::Translation {
namespace {

bool ProducesValue(const IrAction action) noexcept {
    switch (action) {
    case IrAction::WriteInteger:
    case IrAction::WriteVector:
    case IrAction::StoreMemory:
    case IrAction::WriteFlags:
    case IrAction::Barrier:
    case IrAction::SemanticFallback:
    case IrAction::ExitDirect:
    case IrAction::ExitConditional:
    case IrAction::ExitIndirect:
    case IrAction::ExitReturn:
        return false;
    default:
        return true;
    }
}

bool IsBinary(const IrAction action) noexcept {
    return action == IrAction::Add || action == IrAction::Subtract ||
           action == IrAction::Multiply || action == IrAction::BitAnd ||
           action == IrAction::BitOr || action == IrAction::BitXor ||
           action == IrAction::ShiftLeft || action == IrAction::ShiftRightLogical ||
           action == IrAction::ShiftRightArithmetic || action == IrAction::RotateLeft ||
           action == IrAction::RotateRight || action == IrAction::CompareEqual ||
           action == IrAction::CompareUnsigned || action == IrAction::CompareSigned;
}

} // namespace

bool IsIntegerType(const IrType type) noexcept {
    return type == IrType::Integer8 || type == IrType::Integer16 ||
           type == IrType::Integer32 || type == IrType::Integer64 ||
           type == IrType::Address || type == IrType::FlagSet;
}

std::uint32_t IrTypeBitWidth(const IrType type) noexcept {
    switch (type) {
    case IrType::Integer8:
        return 8;
    case IrType::Integer16:
        return 16;
    case IrType::Integer32:
        return 32;
    case IrType::Integer64:
    case IrType::Address:
    case IrType::FlagSet:
        return 64;
    case IrType::Vector128:
        return 128;
    case IrType::Vector256:
        return 256;
    case IrType::None:
        return 0;
    }
    return 0;
}

OperationIr::OperationIr(const std::uint64_t first_guest_address) noexcept
    : first_guest_address_{first_guest_address} {}

bool OperationIr::InputsAreAvailable(const std::span<const IrValue> inputs) const noexcept {
    if (inputs.size() > IrStep{}.inputs.size()) {
        return false;
    }
    return std::all_of(inputs.begin(), inputs.end(), [this](const IrValue input) {
        return input.Exists() && input.id < next_value_id_;
    });
}

IrValue OperationIr::AddValue(const IrAction action, const IrType type,
                              const std::span<const IrValue> inputs,
                              const std::uint64_t immediate, const std::uint32_t detail,
                              const std::uint64_t guest_address) {
    if (!ProducesValue(action) || type == IrType::None || !InputsAreAvailable(inputs)) {
        construction_failed_ = true;
        return {};
    }
    IrStep step{};
    step.action = action;
    step.result = {next_value_id_++, type};
    std::copy(inputs.begin(), inputs.end(), step.inputs.begin());
    step.input_count = static_cast<std::uint8_t>(inputs.size());
    step.immediate = immediate;
    step.detail = detail;
    step.guest_address = guest_address;
    steps_.push_back(step);
    return step.result;
}

bool OperationIr::AddEffect(const IrAction action, const std::span<const IrValue> inputs,
                            const std::uint64_t immediate, const std::uint32_t detail,
                            const std::uint64_t guest_address) {
    if (ProducesValue(action) || !InputsAreAvailable(inputs)) {
        construction_failed_ = true;
        return false;
    }
    IrStep step{};
    step.action = action;
    std::copy(inputs.begin(), inputs.end(), step.inputs.begin());
    step.input_count = static_cast<std::uint8_t>(inputs.size());
    step.immediate = immediate;
    step.detail = detail;
    step.guest_address = guest_address;
    steps_.push_back(step);
    return true;
}

bool OperationIr::ShapeIsValid(const IrStep& step) noexcept {
    if (ProducesValue(step.action) != step.result.Exists()) {
        return false;
    }
    if (step.input_count > step.inputs.size()) {
        return false;
    }
    if (step.action == IrAction::Constant || step.action == IrAction::ReadInteger ||
        step.action == IrAction::ReadVector || step.action == IrAction::ReadFlags ||
        step.action == IrAction::ExitDirect || step.action == IrAction::ExitReturn ||
        step.action == IrAction::Barrier || step.action == IrAction::SemanticFallback) {
        return step.input_count == 0;
    }
    if (IsBinary(step.action) || step.action == IrAction::StoreMemory) {
        return step.input_count == 2;
    }
    if (step.action == IrAction::Select || step.action == IrAction::ComposeFlags) {
        return step.input_count == 3;
    }
    if (step.action == IrAction::EffectiveAddress) {
        return step.input_count <= 2;
    }
    return step.input_count == 1;
}

bool OperationIr::Validate() const noexcept {
    if (construction_failed_ || steps_.empty()) {
        return false;
    }
    std::uint32_t expected_id = 1;
    for (const IrStep& step : steps_) {
        if (!ShapeIsValid(step)) {
            return false;
        }
        if (step.result.Exists() && step.result.id != expected_id++) {
            return false;
        }
        for (std::size_t input = 0; input < step.input_count; ++input) {
            if (!step.inputs[input].Exists() || step.inputs[input].id >= expected_id) {
                return false;
            }
        }
    }
    return true;
}

std::vector<ValueLifetime> OperationIr::AnalyzeLifetimes() const {
    std::vector<ValueLifetime> result(next_value_id_);
    for (std::size_t index = 0; index < steps_.size(); ++index) {
        const IrStep& step = steps_[index];
        if (step.result.Exists()) {
            result[step.result.id] = {index, index, 0};
        }
        for (std::size_t input = 0; input < step.input_count; ++input) {
            ValueLifetime& lifetime = result[step.inputs[input].id];
            lifetime.last_use = index;
            ++lifetime.use_count;
        }
    }
    return result;
}

std::span<const IrStep> OperationIr::Steps() const noexcept {
    return steps_;
}

std::uint64_t OperationIr::FirstGuestAddress() const noexcept {
    return first_guest_address_;
}

} // namespace Lsx4::Translation
