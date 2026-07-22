// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace Lsx4::Translation {

enum class ArmRegister : std::uint8_t {
    X0, X1, X2, X3, X4, X5, X6, X7,
    X8, X9, X10, X11, X12, X13, X14, X15,
    X16, X17, X18, X19, X20, X21, X22, X23,
    X24, X25, X26, X27, X28, X29, X30, ZeroOrStack,
};

enum class ArmCondition : std::uint8_t {
    Equal = 0,
    NotEqual = 1,
    CarrySet = 2,
    CarryClear = 3,
    Negative = 4,
    NonNegative = 5,
    Overflow = 6,
    NoOverflow = 7,
    UnsignedHigher = 8,
    UnsignedLowerOrSame = 9,
    SignedGreaterOrEqual = 10,
    SignedLess = 11,
    SignedGreater = 12,
    SignedLessOrEqual = 13,
    Always = 14,
};

struct CodeLabel {
    std::uint32_t id{};
};

class ArmWordStream {
public:
    [[nodiscard]] CodeLabel NewLabel();
    [[nodiscard]] bool Place(CodeLabel label) noexcept;

    void NoOperation();
    void Return(ArmRegister target = ArmRegister::X30);
    void JumpRegister(ArmRegister target);
    void CallRegister(ArmRegister target);
    void CopyRegister(ArmRegister destination, ArmRegister source);
    void PutConstant(ArmRegister destination, std::uint64_t value);

    [[nodiscard]] bool AddImmediate(ArmRegister destination, ArmRegister source,
                                    std::uint16_t value, bool left_shift_12 = false);
    [[nodiscard]] bool SubtractImmediate(ArmRegister destination, ArmRegister source,
                                         std::uint16_t value, bool left_shift_12 = false);
    void AddRegisters(ArmRegister destination, ArmRegister left, ArmRegister right);
    void SubtractRegisters(ArmRegister destination, ArmRegister left, ArmRegister right);
    void AndRegisters(ArmRegister destination, ArmRegister left, ArmRegister right);
    void OrRegisters(ArmRegister destination, ArmRegister left, ArmRegister right);
    void XorRegisters(ArmRegister destination, ArmRegister left, ArmRegister right);

    [[nodiscard]] bool Load(ArmRegister destination, ArmRegister base,
                            std::uint16_t byte_offset, std::uint8_t width_bytes);
    [[nodiscard]] bool Store(ArmRegister source, ArmRegister base,
                             std::uint16_t byte_offset, std::uint8_t width_bytes);

    void Jump(CodeLabel target);
    void Call(CodeLabel target);
    void JumpIf(ArmCondition condition, CodeLabel target);
    void JumpIfZero(ArmRegister value, CodeLabel target, bool is_64_bit = true);
    void JumpIfNonzero(ArmRegister value, CodeLabel target, bool is_64_bit = true);

    [[nodiscard]] bool Finalize() noexcept;
    [[nodiscard]] std::size_t ByteSize() const noexcept;
    [[nodiscard]] std::span<const std::uint32_t> Words() const noexcept;
    [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const noexcept;

private:
    enum class FixupKind : std::uint8_t {
        Jump26,
        Call26,
        Condition19,
        Zero19,
        Nonzero19,
    };

    struct Fixup {
        std::size_t word_index{};
        std::uint32_t label_id{};
        FixupKind kind{FixupKind::Jump26};
    };

    void Append(std::uint32_t word);
    void AddFixup(CodeLabel target, FixupKind kind, std::uint32_t template_word);

    std::vector<std::uint32_t> words_{};
    std::vector<std::size_t> label_positions_{};
    std::vector<Fixup> fixups_{};
    bool finalized_{};
};

} // namespace Lsx4::Translation
