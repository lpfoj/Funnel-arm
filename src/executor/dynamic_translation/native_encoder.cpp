// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/native_encoder.h"

#include <cstring>
#include <limits>

namespace Lsx4::Translation {
namespace {

constexpr std::uint32_t RegisterBits(const ArmRegister value) noexcept {
    return static_cast<std::uint32_t>(value);
}

bool FitsSigned(const std::int64_t value, const unsigned bits) noexcept {
    const std::int64_t lower = -(std::int64_t{1} << (bits - 1));
    const std::int64_t upper = (std::int64_t{1} << (bits - 1)) - 1;
    return value >= lower && value <= upper;
}

std::uint32_t MemoryTemplate(const bool load, const std::uint8_t width_bytes) noexcept {
    switch (width_bytes) {
    case 1:
        return load ? 0x39400000u : 0x39000000u;
    case 2:
        return load ? 0x79400000u : 0x79000000u;
    case 4:
        return load ? 0xb9400000u : 0xb9000000u;
    case 8:
        return load ? 0xf9400000u : 0xf9000000u;
    default:
        return 0;
    }
}

} // namespace

CodeLabel ArmWordStream::NewLabel() {
    label_positions_.push_back(std::numeric_limits<std::size_t>::max());
    return {static_cast<std::uint32_t>(label_positions_.size() - 1)};
}

bool ArmWordStream::Place(const CodeLabel label) noexcept {
    if (finalized_ || label.id >= label_positions_.size() ||
        label_positions_[label.id] != std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    label_positions_[label.id] = words_.size();
    return true;
}

void ArmWordStream::Append(const std::uint32_t word) {
    if (!finalized_) {
        words_.push_back(word);
    }
}

void ArmWordStream::NoOperation() {
    Append(0xd503201fu);
}

void ArmWordStream::Return(const ArmRegister target) {
    Append(0xd65f0000u | (RegisterBits(target) << 5));
}

void ArmWordStream::JumpRegister(const ArmRegister target) {
    Append(0xd61f0000u | (RegisterBits(target) << 5));
}

void ArmWordStream::CallRegister(const ArmRegister target) {
    Append(0xd63f0000u | (RegisterBits(target) << 5));
}

void ArmWordStream::CopyRegister(const ArmRegister destination, const ArmRegister source) {
    Append(0xaa0003e0u | (RegisterBits(source) << 16) | RegisterBits(destination));
}

void ArmWordStream::PutConstant(const ArmRegister destination, const std::uint64_t value) {
    bool emitted = false;
    for (unsigned lane = 0; lane < 4; ++lane) {
        const std::uint32_t part = static_cast<std::uint32_t>((value >> (lane * 16u)) & 0xffffu);
        if (part == 0 && (emitted || value != 0)) {
            continue;
        }
        const std::uint32_t base = emitted ? 0xf2800000u : 0xd2800000u;
        Append(base | (lane << 21) | (part << 5) | RegisterBits(destination));
        emitted = true;
    }
}

bool ArmWordStream::AddImmediate(const ArmRegister destination, const ArmRegister source,
                                 const std::uint16_t value, const bool left_shift_12) {
    if (value > 4095) {
        return false;
    }
    Append(0x91000000u | (static_cast<std::uint32_t>(left_shift_12) << 22) |
           (static_cast<std::uint32_t>(value) << 10) | (RegisterBits(source) << 5) |
           RegisterBits(destination));
    return true;
}

bool ArmWordStream::SubtractImmediate(const ArmRegister destination,
                                      const ArmRegister source, const std::uint16_t value,
                                      const bool left_shift_12) {
    if (value > 4095) {
        return false;
    }
    Append(0xd1000000u | (static_cast<std::uint32_t>(left_shift_12) << 22) |
           (static_cast<std::uint32_t>(value) << 10) | (RegisterBits(source) << 5) |
           RegisterBits(destination));
    return true;
}

void ArmWordStream::AddRegisters(const ArmRegister destination, const ArmRegister left,
                                 const ArmRegister right) {
    Append(0x8b000000u | (RegisterBits(right) << 16) | (RegisterBits(left) << 5) |
           RegisterBits(destination));
}

void ArmWordStream::SubtractRegisters(const ArmRegister destination,
                                      const ArmRegister left, const ArmRegister right) {
    Append(0xcb000000u | (RegisterBits(right) << 16) | (RegisterBits(left) << 5) |
           RegisterBits(destination));
}

void ArmWordStream::AndRegisters(const ArmRegister destination, const ArmRegister left,
                                 const ArmRegister right) {
    Append(0x8a000000u | (RegisterBits(right) << 16) | (RegisterBits(left) << 5) |
           RegisterBits(destination));
}

void ArmWordStream::OrRegisters(const ArmRegister destination, const ArmRegister left,
                                const ArmRegister right) {
    Append(0xaa000000u | (RegisterBits(right) << 16) | (RegisterBits(left) << 5) |
           RegisterBits(destination));
}

void ArmWordStream::XorRegisters(const ArmRegister destination, const ArmRegister left,
                                 const ArmRegister right) {
    Append(0xca000000u | (RegisterBits(right) << 16) | (RegisterBits(left) << 5) |
           RegisterBits(destination));
}

bool ArmWordStream::Load(const ArmRegister destination, const ArmRegister base,
                         const std::uint16_t byte_offset, const std::uint8_t width_bytes) {
    const std::uint32_t instruction = MemoryTemplate(true, width_bytes);
    if (instruction == 0 || byte_offset % width_bytes != 0 ||
        byte_offset / width_bytes > 4095) {
        return false;
    }
    Append(instruction | ((byte_offset / width_bytes) << 10) | (RegisterBits(base) << 5) |
           RegisterBits(destination));
    return true;
}

bool ArmWordStream::Store(const ArmRegister source, const ArmRegister base,
                          const std::uint16_t byte_offset, const std::uint8_t width_bytes) {
    const std::uint32_t instruction = MemoryTemplate(false, width_bytes);
    if (instruction == 0 || byte_offset % width_bytes != 0 ||
        byte_offset / width_bytes > 4095) {
        return false;
    }
    Append(instruction | ((byte_offset / width_bytes) << 10) | (RegisterBits(base) << 5) |
           RegisterBits(source));
    return true;
}

void ArmWordStream::AddFixup(const CodeLabel target, const FixupKind kind,
                             const std::uint32_t template_word) {
    if (finalized_) {
        return;
    }
    fixups_.push_back({words_.size(), target.id, kind});
    words_.push_back(template_word);
}

void ArmWordStream::Jump(const CodeLabel target) {
    AddFixup(target, FixupKind::Jump26, 0x14000000u);
}

void ArmWordStream::Call(const CodeLabel target) {
    AddFixup(target, FixupKind::Call26, 0x94000000u);
}

void ArmWordStream::JumpIf(const ArmCondition condition, const CodeLabel target) {
    AddFixup(target, FixupKind::Condition19,
             0x54000000u | static_cast<std::uint32_t>(condition));
}

void ArmWordStream::JumpIfZero(const ArmRegister value, const CodeLabel target,
                               const bool is_64_bit) {
    AddFixup(target, FixupKind::Zero19,
             0x34000000u | (static_cast<std::uint32_t>(is_64_bit) << 31) |
                 RegisterBits(value));
}

void ArmWordStream::JumpIfNonzero(const ArmRegister value, const CodeLabel target,
                                  const bool is_64_bit) {
    AddFixup(target, FixupKind::Nonzero19,
             0x35000000u | (static_cast<std::uint32_t>(is_64_bit) << 31) |
                 RegisterBits(value));
}

bool ArmWordStream::Finalize() noexcept {
    if (finalized_) {
        return true;
    }
    for (const Fixup& fixup : fixups_) {
        if (fixup.label_id >= label_positions_.size()) {
            return false;
        }
        const std::size_t target = label_positions_[fixup.label_id];
        if (target == std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        const std::int64_t delta = static_cast<std::int64_t>(target) -
                                   static_cast<std::int64_t>(fixup.word_index);
        const bool wide = fixup.kind == FixupKind::Jump26 || fixup.kind == FixupKind::Call26;
        const unsigned bits = wide ? 26 : 19;
        if (!FitsSigned(delta, bits)) {
            return false;
        }
        const std::uint32_t mask = (std::uint32_t{1} << bits) - 1;
        const unsigned shift = wide ? 0 : 5;
        words_[fixup.word_index] |= (static_cast<std::uint32_t>(delta) & mask) << shift;
    }
    finalized_ = true;
    return true;
}

std::size_t ArmWordStream::ByteSize() const noexcept {
    return words_.size() * sizeof(std::uint32_t);
}

std::span<const std::uint32_t> ArmWordStream::Words() const noexcept {
    return words_;
}

bool ArmWordStream::CopyTo(const std::span<std::uint8_t> destination) const noexcept {
    if (!finalized_ || destination.size() < ByteSize()) {
        return false;
    }
    std::memcpy(destination.data(), words_.data(), ByteSize());
    return true;
}

} // namespace Lsx4::Translation
