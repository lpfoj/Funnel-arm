// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/vector_semantics.h"

#include "common/x86_decoder.h"
#include "executor/dynamic_translation/address_resolver.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>

namespace Lsx4::Translation {
namespace {

struct VectorRegisterAccess {
    std::uint8_t slot{};
    std::uint8_t bytes{};
};

struct VectorValue {
    std::array<std::uint8_t, 32> bytes{};
    std::uint64_t address{};
    std::uint8_t active_bytes{};
    bool from_memory{};
};

VectorSemanticResult Unsupported(const Instruction& instruction) noexcept {
    return {VectorStop::Unsupported, instruction.address, 0};
}

VectorSemanticResult Fault(const Instruction& instruction,
                           const std::uint64_t address) noexcept {
    return {VectorStop::MemoryFault, instruction.address, address};
}

std::optional<VectorRegisterAccess> VectorRegister(
    const std::uint32_t id) noexcept {
    if (id >= X86_REGISTER_XMM0 && id <= X86_REGISTER_XMM15) {
        return VectorRegisterAccess{static_cast<std::uint8_t>(id - X86_REGISTER_XMM0),
                                    16};
    }
    if (id >= X86_REGISTER_YMM0 && id <= X86_REGISTER_YMM15) {
        return VectorRegisterAccess{static_cast<std::uint8_t>(id - X86_REGISTER_YMM0),
                                    32};
    }
    return std::nullopt;
}

std::optional<VectorValue> ReadVectorOperand(const Instruction& instruction,
                                             const Operand& operand,
                                             const CpuFrame& frame,
                                             const GuestMemoryPort& memory) noexcept {
    const std::size_t requested = operand.bit_width / 8;
    if ((requested != 16 && requested != 32) || operand.bit_width % 8 != 0) {
        return std::nullopt;
    }
    VectorValue value{};
    value.active_bytes = static_cast<std::uint8_t>(requested);
    if (operand.form == OperandForm::Register) {
        const auto access = VectorRegister(operand.register_id);
        if (!access || requested > access->bytes) {
            return std::nullopt;
        }
        std::memcpy(value.bytes.data(), frame.vectors[access->slot].data(), requested);
        return value;
    }
    const auto address = ResolveGuestAddress(instruction, operand, frame);
    if (!address || !memory.read ||
        !memory.read(*address,
                     std::span<std::uint8_t>{value.bytes.data(), requested},
                     memory.context)) {
        return std::nullopt;
    }
    value.address = *address;
    value.from_memory = true;
    return value;
}

bool WriteVectorOperand(const Instruction& instruction, const Operand& operand,
                        const VectorValue& value, CpuFrame& frame,
                        const GuestMemoryPort& memory,
                        const std::optional<std::uint64_t> known_address =
                            std::nullopt) noexcept {
    const std::size_t requested = operand.bit_width / 8;
    if (requested != value.active_bytes) {
        return false;
    }
    if (operand.form == OperandForm::Register) {
        const auto access = VectorRegister(operand.register_id);
        if (!access || requested > access->bytes) {
            return false;
        }
        std::memcpy(frame.vectors[access->slot].data(), value.bytes.data(), requested);
        if (requested == 16 &&
            instruction.encoding_family != X86_INSTRUCTION_ENCODING_LEGACY) {
            std::memset(frame.vectors[access->slot].data() + 16, 0, 16);
        }
        return true;
    }
    const auto address = known_address ? known_address
                                       : ResolveGuestAddress(instruction, operand, frame);
    return address && memory.write &&
           memory.write(*address,
                        std::span<const std::uint8_t>{value.bytes.data(), requested},
                        memory.context);
}

std::uint32_t NormalizeMnemonic(const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_VMOVAPS:
        return X86_MNEMONIC_MOVAPS;
    case X86_MNEMONIC_VMOVUPS:
        return X86_MNEMONIC_MOVUPS;
    case X86_MNEMONIC_VMOVDQA:
    case X86_MNEMONIC_VMOVDQA32:
    case X86_MNEMONIC_VMOVDQA64:
        return X86_MNEMONIC_MOVDQA;
    case X86_MNEMONIC_VMOVDQU:
    case X86_MNEMONIC_VMOVDQU8:
    case X86_MNEMONIC_VMOVDQU16:
    case X86_MNEMONIC_VMOVDQU32:
    case X86_MNEMONIC_VMOVDQU64:
        return X86_MNEMONIC_MOVDQU;
    case X86_MNEMONIC_VPXOR:
    case X86_MNEMONIC_VPXORD:
    case X86_MNEMONIC_VPXORQ:
        return X86_MNEMONIC_PXOR;
    case X86_MNEMONIC_VPAND:
    case X86_MNEMONIC_VPANDD:
    case X86_MNEMONIC_VPANDQ:
        return X86_MNEMONIC_PAND;
    case X86_MNEMONIC_VPANDN:
    case X86_MNEMONIC_VPANDND:
    case X86_MNEMONIC_VPANDNQ:
        return X86_MNEMONIC_PANDN;
    case X86_MNEMONIC_VPOR:
    case X86_MNEMONIC_VPORD:
    case X86_MNEMONIC_VPORQ:
        return X86_MNEMONIC_POR;
    case X86_MNEMONIC_VPADDB:
        return X86_MNEMONIC_PADDB;
    case X86_MNEMONIC_VPADDW:
        return X86_MNEMONIC_PADDW;
    case X86_MNEMONIC_VPADDD:
        return X86_MNEMONIC_PADDD;
    case X86_MNEMONIC_VPADDQ:
        return X86_MNEMONIC_PADDQ;
    case X86_MNEMONIC_VPADDSB:
        return X86_MNEMONIC_PADDSB;
    case X86_MNEMONIC_VPADDSW:
        return X86_MNEMONIC_PADDSW;
    case X86_MNEMONIC_VPADDUSB:
        return X86_MNEMONIC_PADDUSB;
    case X86_MNEMONIC_VPADDUSW:
        return X86_MNEMONIC_PADDUSW;
    case X86_MNEMONIC_VPSUBB:
        return X86_MNEMONIC_PSUBB;
    case X86_MNEMONIC_VPSUBW:
        return X86_MNEMONIC_PSUBW;
    case X86_MNEMONIC_VPSUBD:
        return X86_MNEMONIC_PSUBD;
    case X86_MNEMONIC_VPSUBQ:
        return X86_MNEMONIC_PSUBQ;
    case X86_MNEMONIC_VPSUBSB:
        return X86_MNEMONIC_PSUBSB;
    case X86_MNEMONIC_VPSUBSW:
        return X86_MNEMONIC_PSUBSW;
    case X86_MNEMONIC_VPSUBUSB:
        return X86_MNEMONIC_PSUBUSB;
    case X86_MNEMONIC_VPSUBUSW:
        return X86_MNEMONIC_PSUBUSW;
    case X86_MNEMONIC_VPACKSSWB:
        return X86_MNEMONIC_PACKSSWB;
    case X86_MNEMONIC_VPACKSSDW:
        return X86_MNEMONIC_PACKSSDW;
    case X86_MNEMONIC_VPACKUSWB:
        return X86_MNEMONIC_PACKUSWB;
    case X86_MNEMONIC_VPACKUSDW:
        return X86_MNEMONIC_PACKUSDW;
    case X86_MNEMONIC_VPCMPEQB:
        return X86_MNEMONIC_PCMPEQB;
    case X86_MNEMONIC_VPCMPEQW:
        return X86_MNEMONIC_PCMPEQW;
    case X86_MNEMONIC_VPCMPEQD:
        return X86_MNEMONIC_PCMPEQD;
    case X86_MNEMONIC_VPCMPEQQ:
        return X86_MNEMONIC_PCMPEQQ;
    case X86_MNEMONIC_VPCMPGTB:
        return X86_MNEMONIC_PCMPGTB;
    case X86_MNEMONIC_VPCMPGTW:
        return X86_MNEMONIC_PCMPGTW;
    case X86_MNEMONIC_VPCMPGTD:
        return X86_MNEMONIC_PCMPGTD;
    case X86_MNEMONIC_VPCMPGTQ:
        return X86_MNEMONIC_PCMPGTQ;
    case X86_MNEMONIC_VPSHUFB:
        return X86_MNEMONIC_PSHUFB;
    case X86_MNEMONIC_VPSHUFD:
        return X86_MNEMONIC_PSHUFD;
    case X86_MNEMONIC_VSHUFPS:
        return X86_MNEMONIC_SHUFPS;
    case X86_MNEMONIC_VUNPCKLPS:
        return X86_MNEMONIC_UNPCKLPS;
    case X86_MNEMONIC_VUNPCKHPS:
        return X86_MNEMONIC_UNPCKHPS;
    case X86_MNEMONIC_VUNPCKLPD:
        return X86_MNEMONIC_UNPCKLPD;
    case X86_MNEMONIC_VUNPCKHPD:
        return X86_MNEMONIC_UNPCKHPD;
    default:
        return mnemonic;
    }
}

bool IsMove(const std::uint32_t mnemonic) noexcept {
    return mnemonic == X86_MNEMONIC_MOVAPS || mnemonic == X86_MNEMONIC_MOVUPS ||
           mnemonic == X86_MNEMONIC_MOVDQA || mnemonic == X86_MNEMONIC_MOVDQU;
}

std::uint8_t PackedLaneBytes(const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_PADDB:
    case X86_MNEMONIC_PSUBB:
        return 1;
    case X86_MNEMONIC_PADDW:
    case X86_MNEMONIC_PSUBW:
        return 2;
    case X86_MNEMONIC_PADDD:
    case X86_MNEMONIC_PSUBD:
        return 4;
    case X86_MNEMONIC_PADDQ:
    case X86_MNEMONIC_PSUBQ:
        return 8;
    default:
        return 0;
    }
}

std::uint64_t LoadLane(const std::uint8_t* source,
                       const std::uint8_t bytes) noexcept {
    std::uint64_t value = 0;
    for (std::uint8_t index = 0; index < bytes; ++index) {
        value |= static_cast<std::uint64_t>(source[index]) << (index * 8u);
    }
    return value;
}

void StoreLane(std::uint8_t* destination, const std::uint8_t bytes,
               const std::uint64_t value) noexcept {
    for (std::uint8_t index = 0; index < bytes; ++index) {
        destination[index] = static_cast<std::uint8_t>(value >> (index * 8u));
    }
}

VectorValue PackedBinary(const std::uint32_t mnemonic, const VectorValue& left,
                         const VectorValue& right) noexcept {
    VectorValue result{};
    result.active_bytes = left.active_bytes;
    const std::uint8_t lane_bytes = PackedLaneBytes(mnemonic);
    const std::uint64_t mask = lane_bytes == 8
                                   ? ~std::uint64_t{}
                                   : (std::uint64_t{1} << (lane_bytes * 8u)) - 1;
    for (std::size_t offset = 0; offset < result.active_bytes; offset += lane_bytes) {
        const std::uint64_t first = LoadLane(left.bytes.data() + offset, lane_bytes);
        const std::uint64_t second = LoadLane(right.bytes.data() + offset, lane_bytes);
        const std::uint64_t value =
            mnemonic == X86_MNEMONIC_PADDB || mnemonic == X86_MNEMONIC_PADDW ||
                    mnemonic == X86_MNEMONIC_PADDD || mnemonic == X86_MNEMONIC_PADDQ
                ? (first + second) & mask
                : (first - second) & mask;
        StoreLane(result.bytes.data() + offset, lane_bytes, value);
    }
    return result;
}

VectorValue LogicalBinary(const std::uint32_t mnemonic, const VectorValue& left,
                          const VectorValue& right) noexcept {
    VectorValue result{};
    result.active_bytes = left.active_bytes;
    for (std::size_t index = 0; index < result.active_bytes; ++index) {
        if (mnemonic == X86_MNEMONIC_PXOR) {
            result.bytes[index] = left.bytes[index] ^ right.bytes[index];
        } else if (mnemonic == X86_MNEMONIC_PAND) {
            result.bytes[index] = left.bytes[index] & right.bytes[index];
        } else if (mnemonic == X86_MNEMONIC_PANDN) {
            result.bytes[index] = static_cast<std::uint8_t>(~left.bytes[index]) &
                                  right.bytes[index];
        } else {
            result.bytes[index] = left.bytes[index] | right.bytes[index];
        }
    }
    return result;
}

std::uint8_t SaturatingLaneBytes(const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_PADDSB:
    case X86_MNEMONIC_PADDUSB:
    case X86_MNEMONIC_PSUBSB:
    case X86_MNEMONIC_PSUBUSB:
        return 1;
    case X86_MNEMONIC_PADDSW:
    case X86_MNEMONIC_PADDUSW:
    case X86_MNEMONIC_PSUBSW:
    case X86_MNEMONIC_PSUBUSW:
        return 2;
    default:
        return 0;
    }
}

bool IsUnsignedSaturation(const std::uint32_t mnemonic) noexcept {
    return mnemonic == X86_MNEMONIC_PADDUSB || mnemonic == X86_MNEMONIC_PADDUSW ||
           mnemonic == X86_MNEMONIC_PSUBUSB || mnemonic == X86_MNEMONIC_PSUBUSW;
}

bool IsSaturatingAddition(const std::uint32_t mnemonic) noexcept {
    return mnemonic == X86_MNEMONIC_PADDSB || mnemonic == X86_MNEMONIC_PADDSW ||
           mnemonic == X86_MNEMONIC_PADDUSB || mnemonic == X86_MNEMONIC_PADDUSW;
}

std::int64_t SignedLane(const std::uint64_t value,
                        const std::uint8_t bytes) noexcept {
    if (bytes == 8) {
        return static_cast<std::int64_t>(value);
    }
    const std::uint32_t bits = bytes * 8u;
    const std::uint64_t sign = std::uint64_t{1} << (bits - 1);
    return (value & sign) == 0
               ? static_cast<std::int64_t>(value)
               : static_cast<std::int64_t>(value | (~std::uint64_t{} << bits));
}

VectorValue SaturatingBinary(const std::uint32_t mnemonic, const VectorValue& left,
                             const VectorValue& right) noexcept {
    VectorValue result{};
    result.active_bytes = left.active_bytes;
    const std::uint8_t bytes = SaturatingLaneBytes(mnemonic);
    const bool addition = IsSaturatingAddition(mnemonic);
    const bool unsigned_values = IsUnsignedSaturation(mnemonic);
    const std::uint32_t bits = bytes * 8u;
    for (std::size_t offset = 0; offset < result.active_bytes; offset += bytes) {
        const std::uint64_t first = LoadLane(left.bytes.data() + offset, bytes);
        const std::uint64_t second = LoadLane(right.bytes.data() + offset, bytes);
        std::uint64_t value = 0;
        if (unsigned_values) {
            const std::uint64_t maximum = (std::uint64_t{1} << bits) - 1;
            value = addition ? std::min(maximum, first + second)
                             : (first < second ? 0 : first - second);
        } else {
            const std::int64_t minimum = -(std::int64_t{1} << (bits - 1));
            const std::int64_t maximum = (std::int64_t{1} << (bits - 1)) - 1;
            const std::int64_t wide = addition
                                          ? SignedLane(first, bytes) +
                                                SignedLane(second, bytes)
                                          : SignedLane(first, bytes) -
                                                SignedLane(second, bytes);
            value = static_cast<std::uint64_t>(std::clamp(wide, minimum, maximum));
        }
        StoreLane(result.bytes.data() + offset, bytes, value);
    }
    return result;
}

std::uint8_t ComparisonLaneBytes(const std::uint32_t mnemonic) noexcept {
    switch (mnemonic) {
    case X86_MNEMONIC_PCMPEQB:
    case X86_MNEMONIC_PCMPGTB:
        return 1;
    case X86_MNEMONIC_PCMPEQW:
    case X86_MNEMONIC_PCMPGTW:
        return 2;
    case X86_MNEMONIC_PCMPEQD:
    case X86_MNEMONIC_PCMPGTD:
        return 4;
    case X86_MNEMONIC_PCMPEQQ:
    case X86_MNEMONIC_PCMPGTQ:
        return 8;
    default:
        return 0;
    }
}

VectorValue ComparePacked(const std::uint32_t mnemonic, const VectorValue& left,
                          const VectorValue& right) noexcept {
    VectorValue result{};
    result.active_bytes = left.active_bytes;
    const std::uint8_t bytes = ComparisonLaneBytes(mnemonic);
    const bool greater = mnemonic == X86_MNEMONIC_PCMPGTB ||
                         mnemonic == X86_MNEMONIC_PCMPGTW ||
                         mnemonic == X86_MNEMONIC_PCMPGTD ||
                         mnemonic == X86_MNEMONIC_PCMPGTQ;
    for (std::size_t offset = 0; offset < result.active_bytes; offset += bytes) {
        const std::uint64_t first = LoadLane(left.bytes.data() + offset, bytes);
        const std::uint64_t second = LoadLane(right.bytes.data() + offset, bytes);
        const bool selected = greater ? SignedLane(first, bytes) > SignedLane(second, bytes)
                                      : first == second;
        StoreLane(result.bytes.data() + offset, bytes,
                  selected ? ~std::uint64_t{} : 0);
    }
    return result;
}

VectorValue PackNarrow(const std::uint32_t mnemonic, const VectorValue& left,
                       const VectorValue& right) noexcept {
    VectorValue result{};
    result.active_bytes = left.active_bytes;
    const bool dwords = mnemonic == X86_MNEMONIC_PACKSSDW ||
                        mnemonic == X86_MNEMONIC_PACKUSDW;
    const bool unsigned_output = mnemonic == X86_MNEMONIC_PACKUSWB ||
                                 mnemonic == X86_MNEMONIC_PACKUSDW;
    const std::uint8_t input_bytes = dwords ? 4 : 2;
    const std::uint8_t output_bytes = dwords ? 2 : 1;
    const std::int64_t minimum = unsigned_output
                                     ? 0
                                     : -(std::int64_t{1} << (output_bytes * 8u - 1));
    const std::int64_t maximum = unsigned_output
                                     ? (std::int64_t{1} << (output_bytes * 8u)) - 1
                                     : (std::int64_t{1} << (output_bytes * 8u - 1)) - 1;
    const std::size_t values_per_source = 16 / input_bytes;
    for (std::size_t lane = 0; lane < result.active_bytes; lane += 16) {
        for (std::size_t source_index = 0; source_index < 2; ++source_index) {
            const VectorValue& source = source_index == 0 ? left : right;
            for (std::size_t index = 0; index < values_per_source; ++index) {
                const std::uint64_t raw =
                    LoadLane(source.bytes.data() + lane + index * input_bytes,
                             input_bytes);
                const std::int64_t narrowed =
                    std::clamp(SignedLane(raw, input_bytes), minimum, maximum);
                const std::size_t output =
                    lane + (source_index * values_per_source + index) * output_bytes;
                StoreLane(result.bytes.data() + output, output_bytes,
                          static_cast<std::uint64_t>(narrowed));
            }
        }
    }
    return result;
}

VectorValue ShuffleBytes(const VectorValue& source,
                         const VectorValue& control) noexcept {
    VectorValue result{};
    result.active_bytes = source.active_bytes;
    for (std::size_t lane = 0; lane < result.active_bytes; lane += 16) {
        for (std::size_t index = 0; index < 16; ++index) {
            const std::uint8_t selector = control.bytes[lane + index];
            result.bytes[lane + index] =
                (selector & 0x80u) != 0 ? 0 : source.bytes[lane + (selector & 0x0fu)];
        }
    }
    return result;
}

VectorValue ShuffleDwords(const VectorValue& source,
                          const std::uint8_t control) noexcept {
    VectorValue result{};
    result.active_bytes = source.active_bytes;
    for (std::size_t lane = 0; lane < result.active_bytes; lane += 16) {
        for (std::size_t output = 0; output < 4; ++output) {
            const std::size_t input = (control >> (output * 2u)) & 3u;
            std::memcpy(result.bytes.data() + lane + output * 4,
                        source.bytes.data() + lane + input * 4, 4);
        }
    }
    return result;
}

VectorValue ShuffleFloats(const VectorValue& left, const VectorValue& right,
                          const std::uint8_t control) noexcept {
    VectorValue result{};
    result.active_bytes = left.active_bytes;
    for (std::size_t lane = 0; lane < result.active_bytes; lane += 16) {
        for (std::size_t output = 0; output < 4; ++output) {
            const std::size_t input = (control >> (output * 2u)) & 3u;
            const auto& source = output < 2 ? left : right;
            std::memcpy(result.bytes.data() + lane + output * 4,
                        source.bytes.data() + lane + input * 4, 4);
        }
    }
    return result;
}

VectorValue Unpack(const VectorValue& left, const VectorValue& right,
                   const std::uint8_t element_bytes, const bool high) noexcept {
    VectorValue result{};
    result.active_bytes = left.active_bytes;
    const std::size_t half_elements = 8 / element_bytes;
    for (std::size_t lane = 0; lane < result.active_bytes; lane += 16) {
        const std::size_t first = high ? half_elements : 0;
        for (std::size_t index = 0; index < half_elements; ++index) {
            std::memcpy(result.bytes.data() + lane + (index * 2) * element_bytes,
                        left.bytes.data() + lane + (first + index) * element_bytes,
                        element_bytes);
            std::memcpy(result.bytes.data() + lane + (index * 2 + 1) * element_bytes,
                        right.bytes.data() + lane + (first + index) * element_bytes,
                        element_bytes);
        }
    }
    return result;
}

} // namespace

VectorSemanticResult ExecuteVectorInstruction(
    const Instruction& instruction, CpuFrame& frame,
    const GuestMemoryPort& memory) noexcept {
    if (instruction.length == 0 ||
        (instruction.prefix_attributes & X86_ATTRIB_HAS_LOCK) != 0) {
        return Unsupported(instruction);
    }
    const std::uint32_t mnemonic = NormalizeMnemonic(instruction.mnemonic);
    if (IsMove(mnemonic)) {
        if (instruction.operand_count != 2) {
            return Unsupported(instruction);
        }
        const auto source =
            ReadVectorOperand(instruction, instruction.operands[1], frame, memory);
        if (!source) {
            const auto address =
                ResolveGuestAddress(instruction, instruction.operands[1], frame);
            return Fault(instruction, address.value_or(instruction.address));
        }
        const auto address =
            ResolveGuestAddress(instruction, instruction.operands[0], frame);
        if (!WriteVectorOperand(instruction, instruction.operands[0], *source, frame,
                                memory, address)) {
            return instruction.operands[0].form == OperandForm::Memory
                       ? Fault(instruction, address.value_or(instruction.address))
                       : Unsupported(instruction);
        }
        return {VectorStop::Continue, instruction.address + instruction.length, 0};
    }

    if (mnemonic == X86_MNEMONIC_PSHUFD) {
        if (instruction.operand_count != 3 ||
            instruction.operands[2].form != OperandForm::Immediate) {
            return Unsupported(instruction);
        }
        const auto source =
            ReadVectorOperand(instruction, instruction.operands[1], frame, memory);
        if (!source) {
            return Fault(instruction, instruction.address);
        }
        const VectorValue result =
            ShuffleDwords(*source, instruction.operands[2].immediate.bits);
        if (!WriteVectorOperand(instruction, instruction.operands[0], result, frame,
                                memory)) {
            return Unsupported(instruction);
        }
        return {VectorStop::Continue, instruction.address + instruction.length, 0};
    }

    const bool vex = instruction.encoding_family != X86_INSTRUCTION_ENCODING_LEGACY;
    const std::size_t left_index = vex ? 1 : 0;
    const std::size_t right_index = vex ? 2 : 1;
    const std::size_t immediate_index = vex ? 3 : 2;
    if (instruction.operand_count <= right_index) {
        return Unsupported(instruction);
    }
    const auto left =
        ReadVectorOperand(instruction, instruction.operands[left_index], frame, memory);
    const auto right =
        ReadVectorOperand(instruction, instruction.operands[right_index], frame, memory);
    if (!left || !right || left->active_bytes != right->active_bytes) {
        return Fault(instruction, instruction.address);
    }

    VectorValue result{};
    if (mnemonic == X86_MNEMONIC_PXOR || mnemonic == X86_MNEMONIC_PAND ||
        mnemonic == X86_MNEMONIC_PANDN || mnemonic == X86_MNEMONIC_POR) {
        result = LogicalBinary(mnemonic, *left, *right);
    } else if (PackedLaneBytes(mnemonic) != 0) {
        result = PackedBinary(mnemonic, *left, *right);
    } else if (SaturatingLaneBytes(mnemonic) != 0) {
        result = SaturatingBinary(mnemonic, *left, *right);
    } else if (ComparisonLaneBytes(mnemonic) != 0) {
        result = ComparePacked(mnemonic, *left, *right);
    } else if (mnemonic == X86_MNEMONIC_PACKSSWB ||
               mnemonic == X86_MNEMONIC_PACKSSDW ||
               mnemonic == X86_MNEMONIC_PACKUSWB ||
               mnemonic == X86_MNEMONIC_PACKUSDW) {
        result = PackNarrow(mnemonic, *left, *right);
    } else if (mnemonic == X86_MNEMONIC_PSHUFB) {
        result = ShuffleBytes(*left, *right);
    } else if (mnemonic == X86_MNEMONIC_SHUFPS) {
        if (instruction.operand_count <= immediate_index ||
            instruction.operands[immediate_index].form != OperandForm::Immediate) {
            return Unsupported(instruction);
        }
        result = ShuffleFloats(*left, *right,
                               instruction.operands[immediate_index].immediate.bits);
    } else if (mnemonic == X86_MNEMONIC_UNPCKLPS ||
               mnemonic == X86_MNEMONIC_UNPCKHPS ||
               mnemonic == X86_MNEMONIC_UNPCKLPD ||
               mnemonic == X86_MNEMONIC_UNPCKHPD) {
        const bool doubles = mnemonic == X86_MNEMONIC_UNPCKLPD ||
                             mnemonic == X86_MNEMONIC_UNPCKHPD;
        const bool high = mnemonic == X86_MNEMONIC_UNPCKHPS ||
                          mnemonic == X86_MNEMONIC_UNPCKHPD;
        result = Unpack(*left, *right, doubles ? 8 : 4, high);
    } else {
        return Unsupported(instruction);
    }

    if (!WriteVectorOperand(instruction, instruction.operands[0], result, frame,
                            memory)) {
        return Unsupported(instruction);
    }
    return {VectorStop::Continue, instruction.address + instruction.length, 0};
}

} // namespace Lsx4::Translation
