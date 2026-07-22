// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/x87_state.h"

#include <bit>
#include <cstring>
#include <utility>

namespace Lsx4::Translation {
namespace {

static_assert(sizeof(long double) <= 16);

constexpr std::uint16_t InvalidOperation = 1u << 0;
constexpr std::uint16_t StackFault = 1u << 6;
constexpr std::uint16_t ConditionOne = 1u << 9;
constexpr std::uint16_t TopMask = 7u << 11;

std::uint8_t PhysicalSlot(const CpuFrame& frame,
                          const std::uint8_t logical_index) noexcept {
    return static_cast<std::uint8_t>((frame.x87_stack_cursor + logical_index) & 7u);
}

bool SlotPresent(const CpuFrame& frame, const std::uint8_t physical) noexcept {
    return (frame.x87_validity & (std::uint8_t{1} << physical)) != 0;
}

void SetTopField(CpuFrame& frame) noexcept {
    frame.x87_status = static_cast<std::uint16_t>(
        (frame.x87_status & ~TopMask) | (frame.x87_stack_cursor << 11));
}

void MarkFault(CpuFrame& frame, const bool overflow) noexcept {
    frame.x87_status |= InvalidOperation | StackFault;
    if (overflow) {
        frame.x87_status |= ConditionOne;
    } else {
        frame.x87_status &= ~ConditionOne;
    }
}

long double LoadSlot(const CpuFrame& frame, const std::uint8_t physical) noexcept {
    long double value{};
    std::memcpy(&value, frame.x87_registers[physical].data(), sizeof(value));
    return value;
}

void StoreSlot(CpuFrame& frame, const std::uint8_t physical,
               const long double value) noexcept {
    frame.x87_registers[physical].fill(0);
    std::memcpy(frame.x87_registers[physical].data(), &value, sizeof(value));
}

} // namespace

X87StackOutcome PushX87(CpuFrame& frame, const long double value) noexcept {
    const std::uint8_t destination =
        static_cast<std::uint8_t>((frame.x87_stack_cursor - 1u) & 7u);
    if (SlotPresent(frame, destination)) {
        MarkFault(frame, true);
        return X87StackOutcome::StackFault;
    }
    frame.x87_stack_cursor = destination;
    StoreSlot(frame, destination, value);
    frame.x87_validity |= std::uint8_t{1} << destination;
    SetTopField(frame);
    return X87StackOutcome::Complete;
}

X87StackOutcome PopX87(CpuFrame& frame, long double* const value) noexcept {
    const std::uint8_t source = frame.x87_stack_cursor;
    if (!SlotPresent(frame, source)) {
        MarkFault(frame, false);
        return X87StackOutcome::StackFault;
    }
    if (value) {
        *value = LoadSlot(frame, source);
    }
    frame.x87_validity &= static_cast<std::uint8_t>(~(std::uint8_t{1} << source));
    frame.x87_registers[source].fill(0);
    frame.x87_stack_cursor = static_cast<std::uint8_t>((source + 1u) & 7u);
    SetTopField(frame);
    return X87StackOutcome::Complete;
}

std::optional<long double> ReadX87(const CpuFrame& frame,
                                   const std::uint8_t logical_index) noexcept {
    if (logical_index >= 8) {
        return std::nullopt;
    }
    const std::uint8_t physical = PhysicalSlot(frame, logical_index);
    return SlotPresent(frame, physical) ? std::optional{LoadSlot(frame, physical)}
                                        : std::nullopt;
}

X87StackOutcome WriteX87(CpuFrame& frame, const std::uint8_t logical_index,
                         const long double value) noexcept {
    if (logical_index >= 8) {
        return X87StackOutcome::InvalidIndex;
    }
    const std::uint8_t physical = PhysicalSlot(frame, logical_index);
    if (!SlotPresent(frame, physical)) {
        MarkFault(frame, false);
        return X87StackOutcome::StackFault;
    }
    StoreSlot(frame, physical, value);
    return X87StackOutcome::Complete;
}

X87StackOutcome ExchangeX87(CpuFrame& frame,
                            const std::uint8_t logical_index) noexcept {
    if (logical_index >= 8) {
        return X87StackOutcome::InvalidIndex;
    }
    const std::uint8_t top = frame.x87_stack_cursor;
    const std::uint8_t other = PhysicalSlot(frame, logical_index);
    if (!SlotPresent(frame, top) || !SlotPresent(frame, other)) {
        MarkFault(frame, false);
        return X87StackOutcome::StackFault;
    }
    std::swap(frame.x87_registers[top], frame.x87_registers[other]);
    return X87StackOutcome::Complete;
}

std::size_t X87Depth(const CpuFrame& frame) noexcept {
    return std::popcount(frame.x87_validity);
}

} // namespace Lsx4::Translation
