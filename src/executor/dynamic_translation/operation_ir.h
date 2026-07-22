// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace Lsx4::Translation {

enum class IrType : std::uint8_t {
    None,
    Integer8,
    Integer16,
    Integer32,
    Integer64,
    Address,
    FlagSet,
    Vector128,
    Vector256,
};

enum class IrAction : std::uint8_t {
    Constant,
    ReadInteger,
    WriteInteger,
    ReadVector,
    WriteVector,
    EffectiveAddress,
    LoadMemory,
    StoreMemory,
    Add,
    Subtract,
    Multiply,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
    RotateLeft,
    RotateRight,
    CompareEqual,
    CompareUnsigned,
    CompareSigned,
    Select,
    Truncate,
    ZeroExtend,
    SignExtend,
    ComposeFlags,
    ReadFlags,
    WriteFlags,
    Barrier,
    SemanticFallback,
    ExitDirect,
    ExitConditional,
    ExitIndirect,
    ExitReturn,
};

struct IrValue {
    std::uint32_t id{};
    IrType type{IrType::None};

    [[nodiscard]] constexpr bool Exists() const noexcept {
        return id != 0 && type != IrType::None;
    }
    friend constexpr bool operator==(IrValue, IrValue) = default;
};

struct IrStep {
    IrAction action{IrAction::Barrier};
    IrValue result{};
    std::array<IrValue, 3> inputs{};
    std::uint64_t immediate{};
    std::uint64_t guest_address{};
    std::uint32_t detail{};
    std::uint8_t input_count{};
};

struct ValueLifetime {
    std::size_t definition{};
    std::size_t last_use{};
    std::size_t use_count{};
};

class OperationIr {
public:
    explicit OperationIr(std::uint64_t first_guest_address = 0) noexcept;

    [[nodiscard]] IrValue AddValue(IrAction action, IrType type,
                                   std::span<const IrValue> inputs = {},
                                   std::uint64_t immediate = 0,
                                   std::uint32_t detail = 0,
                                   std::uint64_t guest_address = 0);
    [[nodiscard]] bool AddEffect(IrAction action, std::span<const IrValue> inputs = {},
                                 std::uint64_t immediate = 0,
                                 std::uint32_t detail = 0,
                                 std::uint64_t guest_address = 0);

    [[nodiscard]] bool Validate() const noexcept;
    [[nodiscard]] std::vector<ValueLifetime> AnalyzeLifetimes() const;
    [[nodiscard]] std::span<const IrStep> Steps() const noexcept;
    [[nodiscard]] std::uint64_t FirstGuestAddress() const noexcept;

private:
    [[nodiscard]] bool InputsAreAvailable(std::span<const IrValue> inputs) const noexcept;
    [[nodiscard]] static bool ShapeIsValid(const IrStep& step) noexcept;

    std::uint64_t first_guest_address_{};
    std::vector<IrStep> steps_{};
    std::uint32_t next_value_id_{1};
    bool construction_failed_{};
};

[[nodiscard]] bool IsIntegerType(IrType type) noexcept;
[[nodiscard]] std::uint32_t IrTypeBitWidth(IrType type) noexcept;

} // namespace Lsx4::Translation
