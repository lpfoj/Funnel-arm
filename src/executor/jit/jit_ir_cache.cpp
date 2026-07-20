// SPDX-FileCopyrightText: Copyright 2026 Executor Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "jit_ir_cache.h"
#include "common/content_fingerprint.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace Executor::Jit {
namespace {

constexpr std::array<std::uint8_t, 8> kCacheMagic = {
    'E', 'X', 'B', 'I', 'R', 'C', '0', '1',
};
constexpr std::uint32_t kHeaderSize = 88;
constexpr std::uint16_t kRecordSchemaVersion = 2;
constexpr std::uint32_t kKnownNativeFlags =
    kJitNativeFlagValid | kJitNativeFlagLeafHleFused |
    kJitNativeFlagScalarFloatDirectV1 |
    kJitNativeFlagScalarFloatDirectV2 |
    kJitNativeFlagPackedFloatDirectV3 |
    kJitNativeFlagMixedHotDirectV4 |
    kJitNativeFlagWideHotDirectV5 |
    kJitNativeFlagPostLoadDirectV6 |
    kJitNativeFlagPostLoadDirectV7 |
    kJitNativeFlagPostLoadDirectV8 |
    kJitNativeFlagPostLoadDirectV9 |
    kJitNativeFlagDirectFaultSlotV10 |
    kJitNativeFlagInlineLseXchgV11 |
    kJitNativeFlagDirectLseXchgV12 |
    kJitNativeFlagSharedChainAbiV13 |
    kJitNativeFlagScalarLogicDirectV14 |
    kJitNativeFlagVpslldImmDirectV15;
constexpr std::uint32_t kAarch64PointerMaterializationBytes = 4 * sizeof(std::uint32_t);

constexpr std::uint16_t kBlockFlagCanJitInitial = 1u << 0;
constexpr std::uint16_t kBlockFlagDecodeFailed = 1u << 1;
constexpr std::uint16_t kKnownBlockFlags = kBlockFlagCanJitInitial | kBlockFlagDecodeFailed;

constexpr std::uint8_t kInstructionFlagCanExecuteVector = 1u << 0;
constexpr std::uint8_t kInstructionFlagCanExecuteScalarFloat = 1u << 1;
constexpr std::uint8_t kInstructionFlagCanJitInitial = 1u << 2;
constexpr std::uint8_t kInstructionFlagTerminatesBlock = 1u << 3;
constexpr std::uint8_t kKnownInstructionFlags =
    kInstructionFlagCanExecuteVector | kInstructionFlagCanExecuteScalarFloat |
    kInstructionFlagCanJitInitial | kInstructionFlagTerminatesBlock;

// Stable limits of the LSX cache schema. Decoder values are normalized into these integer fields
// by machine_code_lens.cpp; changing a bound requires an explicit schema review.
constexpr std::uint8_t kOperandKindUnused = 0;
constexpr std::uint8_t kOperandKindRegister = 1;
constexpr std::uint8_t kOperandKindMemory = 2;
constexpr std::uint8_t kOperandKindPointer = 3;
constexpr std::uint8_t kOperandKindImmediate = 4;
constexpr std::uint8_t kOperandVisibilityLimit = 3;
constexpr std::uint8_t kOperandAttributeLimit = 0x01;
constexpr std::uint8_t kOperandEncodingLimit = 34;
constexpr std::uint16_t kElementTypeLimit = 10;
constexpr std::uint8_t kMemoryOperandKindLimit = 4;
constexpr std::uint32_t kRegisterCodeLimit = 266;
constexpr std::uint32_t kMnemonicCodeLimit = 1795;
constexpr std::uint8_t kInstructionEncodingLimit = 5;

[[nodiscard]] std::uint64_t FingerprintPayload(
    const std::span<const std::uint8_t> bytes) noexcept {
    return Common::FingerprintBytes(bytes, Common::FingerprintDomain::IrCachePayload);
}

[[nodiscard]] std::uint64_t FingerprintNativeSegment(
    const std::span<const std::uint8_t> bytes) noexcept {
    return Common::FingerprintBytes(bytes, Common::FingerprintDomain::NativeSegment);
}

class ByteWriter {
public:
    explicit ByteWriter(const std::uint64_t max_size) : max_size_{max_size} {}

    [[nodiscard]] bool ok() const noexcept {
        return ok_;
    }

    [[nodiscard]] const std::vector<std::uint8_t>& bytes() const noexcept {
        return bytes_;
    }

    [[nodiscard]] std::vector<std::uint8_t> TakeBytes() noexcept {
        return std::move(bytes_);
    }

    void U8(const std::uint8_t value) {
        Append(&value, sizeof(value));
    }

    void U16(const std::uint16_t value) {
        const std::array<std::uint8_t, 2> encoded = {
            static_cast<std::uint8_t>(value),
            static_cast<std::uint8_t>(value >> 8),
        };
        Append(encoded.data(), encoded.size());
    }

    void U32(const std::uint32_t value) {
        const std::array<std::uint8_t, 4> encoded = {
            static_cast<std::uint8_t>(value),
            static_cast<std::uint8_t>(value >> 8),
            static_cast<std::uint8_t>(value >> 16),
            static_cast<std::uint8_t>(value >> 24),
        };
        Append(encoded.data(), encoded.size());
    }

    void U64(const std::uint64_t value) {
        std::array<std::uint8_t, 8> encoded{};
        for (std::size_t index = 0; index < encoded.size(); ++index) {
            encoded[index] = static_cast<std::uint8_t>(value >> (index * 8));
        }
        Append(encoded.data(), encoded.size());
    }

    void Bytes(const std::span<const std::uint8_t> bytes) {
        Append(bytes.data(), bytes.size());
    }

private:
    void Append(const void* source, const std::size_t size) {
        if (size == 0) {
            return;
        }
        if (!ok_ || size > max_size_ || bytes_.size() > max_size_ - size) {
            ok_ = false;
            return;
        }
        const auto* begin = static_cast<const std::uint8_t*>(source);
        bytes_.insert(bytes_.end(), begin, begin + size);
    }

    std::uint64_t max_size_ = 0;
    std::vector<std::uint8_t> bytes_{};
    bool ok_ = true;
};

class ByteReader {
public:
    explicit ByteReader(const std::span<const std::uint8_t> bytes) : bytes_{bytes} {}

    [[nodiscard]] std::size_t remaining() const noexcept {
        return bytes_.size() - offset_;
    }

    [[nodiscard]] bool empty() const noexcept {
        return remaining() == 0;
    }

    bool U8(std::uint8_t& value) {
        if (remaining() < 1) {
            return false;
        }
        value = bytes_[offset_++];
        return true;
    }

    bool U16(std::uint16_t& value) {
        if (remaining() < 2) {
            return false;
        }
        value = static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes_[offset_]) |
                                           static_cast<std::uint16_t>(bytes_[offset_ + 1]) << 8);
        offset_ += 2;
        return true;
    }

    bool U32(std::uint32_t& value) {
        if (remaining() < 4) {
            return false;
        }
        value = static_cast<std::uint32_t>(bytes_[offset_]) |
                static_cast<std::uint32_t>(bytes_[offset_ + 1]) << 8 |
                static_cast<std::uint32_t>(bytes_[offset_ + 2]) << 16 |
                static_cast<std::uint32_t>(bytes_[offset_ + 3]) << 24;
        offset_ += 4;
        return true;
    }

    bool U64(std::uint64_t& value) {
        if (remaining() < 8) {
            return false;
        }
        value = 0;
        for (std::size_t index = 0; index < 8; ++index) {
            value |= static_cast<std::uint64_t>(bytes_[offset_ + index]) << (index * 8);
        }
        offset_ += 8;
        return true;
    }

    bool Bytes(const std::size_t size, std::span<const std::uint8_t>& bytes) {
        if (size > remaining()) {
            return false;
        }
        bytes = bytes_.subspan(offset_, size);
        offset_ += size;
        return true;
    }

private:
    std::span<const std::uint8_t> bytes_{};
    std::size_t offset_ = 0;
};

[[nodiscard]] bool ValidAddressWidth(const std::uint8_t width) noexcept {
    return width == 0 || width == 16 || width == 32 || width == 64;
}

[[nodiscard]] bool ValidOperandWidth(const std::uint8_t width) noexcept {
    return width == 0 || width == 8 || width == 16 || width == 32 || width == 64;
}

[[nodiscard]] bool ValidateOperand(const LsxOperandRecord& operand, std::string& error) {
    if (operand.visibility > kOperandVisibilityLimit || operand.actions > 0x0f ||
        operand.attributes > kOperandAttributeLimit ||
        operand.encoding > kOperandEncodingLimit || operand.element_type > kElementTypeLimit ||
        operand.type > kOperandKindImmediate) {
        error = "invalid cached operand enum";
        return false;
    }

    switch (operand.type) {
    case kOperandKindUnused:
        break;
    case kOperandKindRegister:
        if (operand.reg.value > kRegisterCodeLimit) {
            error = "invalid cached register operand";
            return false;
        }
        break;
    case kOperandKindMemory:
        if (operand.mem.type > kMemoryOperandKindLimit ||
            operand.mem.segment > kRegisterCodeLimit ||
            operand.mem.base > kRegisterCodeLimit ||
            operand.mem.index > kRegisterCodeLimit ||
            (operand.mem.scale != 0 && operand.mem.scale != 1 && operand.mem.scale != 2 &&
             operand.mem.scale != 4 && operand.mem.scale != 8) ||
            operand.mem.disp.offset > kLsxInstructionByteLimit ||
            (operand.mem.disp.size != 0 && operand.mem.disp.size != 8 &&
             operand.mem.disp.size != 16 && operand.mem.disp.size != 32 &&
             operand.mem.disp.size != 64)) {
            error = "invalid cached memory operand";
            return false;
        }
        break;
    case kOperandKindPointer:
        break;
    case kOperandKindImmediate:
        if ((operand.imm.is_signed != 0 && operand.imm.is_signed != 1) ||
            (operand.imm.is_relative != 0 && operand.imm.is_relative != 1) ||
            operand.imm.offset > kLsxInstructionByteLimit ||
            (operand.imm.size != 0 && operand.imm.size != 8 && operand.imm.size != 16 &&
             operand.imm.size != 32 && operand.imm.size != 64)) {
            error = "invalid cached immediate flags";
            return false;
        }
        break;
    default:
        error = "invalid cached operand type";
        return false;
    }
    return true;
}

[[nodiscard]] bool ValidateInstruction(const LsxDecodedOp& instruction,
                                       const JitIrCacheLimits& limits, std::string& error) {
    if (instruction.length == 0 || instruction.length > limits.max_synthetic_instruction_length) {
        error = "invalid instruction length";
        return false;
    }
    if (instruction.length > instruction.bytes.size() &&
        instruction.category != LsxOpClass::System) {
        error = "oversized non-system instruction";
        return false;
    }
    if (static_cast<std::uint32_t>(instruction.mnemonic) >
            kMnemonicCodeLimit ||
        static_cast<std::uint8_t>(instruction.category) >
            static_cast<std::uint8_t>(LsxOpClass::System) ||
        instruction.operand_count > kLsxVisibleOperandLimit) {
        error = "invalid instruction enum or operand count";
        return false;
    }
    if (instruction.decoded.mnemonic != instruction.mnemonic ||
        instruction.decoded.operand_count_visible != instruction.operand_count ||
        instruction.decoded.operand_count > kLsxDecodedOperandLimit ||
        static_cast<std::uint32_t>(instruction.decoded.encoding) >
            kInstructionEncodingLimit ||
        !ValidAddressWidth(instruction.decoded.address_width) ||
        !ValidOperandWidth(instruction.decoded.operand_width)) {
        error = "inconsistent decoded instruction metadata";
        return false;
    }
    if (instruction.length <= instruction.bytes.size()) {
        if (instruction.decoded.length != instruction.length) {
            error = "decoded instruction length mismatch";
            return false;
        }
    } else if (instruction.decoded.length != 0 || instruction.operand_count != 0) {
        // JIT's HLE bridge marker is a synthetic system instruction wider than x86's
        // architectural 15-byte maximum and intentionally has no decoded length/operands.
        error = "invalid synthetic system instruction metadata";
        return false;
    }
    for (std::uint8_t index = 0; index < instruction.operand_count; ++index) {
        if (!ValidateOperand(instruction.operands[index], error)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool ValidNativeRelocationKind(const JitNativeRelocationKind kind) noexcept {
    return static_cast<std::uint8_t>(kind) <=
           static_cast<std::uint8_t>(JitNativeRelocationKind::PolymorphicChainSite);
}

[[nodiscard]] std::uint32_t ReadLittleEndianU32(const std::uint8_t* bytes) noexcept {
    return static_cast<std::uint32_t>(bytes[0]) |
           static_cast<std::uint32_t>(bytes[1]) << 8 |
           static_cast<std::uint32_t>(bytes[2]) << 16 |
           static_cast<std::uint32_t>(bytes[3]) << 24;
}

[[nodiscard]] bool ValidPointerMaterializationSequence(
    const JitNativeSegment& segment, const JitNativeRelocation& relocation) noexcept {
    constexpr std::uint32_t kImmediateMask = 0x001fffe0u;
    const std::array<std::uint32_t, 4> expected = {
        0xd2800000u | relocation.register_index,
        0xf2a00000u | relocation.register_index,
        0xf2c00000u | relocation.register_index,
        0xf2e00000u | relocation.register_index,
    };
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const std::uint32_t instruction = ReadLittleEndianU32(
            segment.bytes.data() + relocation.code_offset + index * sizeof(std::uint32_t));
        if ((instruction & ~kImmediateMask) != expected[index]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool ValidateNativeSegments(const JitIrCacheRecord& record,
                                          const JitIrCacheLimits& limits,
                                          std::uint64_t& native_bytes,
                                          std::uint64_t& native_relocations,
                                          std::string& error) {
    native_bytes = 0;
    native_relocations = 0;
    if ((record.native_flags & ~kKnownNativeFlags) != 0) {
        error = "unknown cached native flags";
        return false;
    }

    const bool valid = (record.native_flags & kJitNativeFlagValid) != 0;
    if (!valid) {
        if (!record.native_segments.empty() ||
            record.entry_segment_index != kJitNativeNoSegment ||
            record.direct_segment_index != kJitNativeNoSegment) {
            error = "native segments present without valid marker";
            return false;
        }
        return true;
    }
    if (record.native_segments.empty() ||
        record.native_segments.size() > limits.max_native_segments_per_record) {
        error = "cached native segment count exceeds limit";
        return false;
    }
    if (record.entry_segment_index >= record.native_segments.size() ||
        (record.direct_segment_index != kJitNativeNoSegment &&
         record.direct_segment_index >= record.native_segments.size())) {
        error = "cached native entry/direct segment index out of bounds";
        return false;
    }

    for (const JitNativeSegment& segment : record.native_segments) {
        if (segment.bytes.empty() || segment.bytes.size() % sizeof(std::uint32_t) != 0 ||
            segment.bytes.size() > limits.max_native_segment_bytes) {
            error = "invalid cached native segment size";
            return false;
        }
        if (segment.relocs.size() > limits.max_native_relocations_per_segment) {
            error = "cached native relocation count exceeds limit";
            return false;
        }
        if (segment.bytes.size() > limits.max_total_native_bytes - native_bytes ||
            segment.relocs.size() >
                limits.max_total_native_relocations - native_relocations) {
            error = "cached native data exceeds total limit";
            return false;
        }
        native_bytes += segment.bytes.size();
        native_relocations += segment.relocs.size();

        std::vector<std::uint32_t> relocation_offsets;
        relocation_offsets.reserve(segment.relocs.size());
        for (const JitNativeRelocation& relocation : segment.relocs) {
            if (!ValidNativeRelocationKind(relocation.kind) || relocation.register_index > 30) {
                error = "invalid cached native relocation kind or register";
                return false;
            }
            if (relocation.code_offset % sizeof(std::uint32_t) != 0 ||
                relocation.code_offset > segment.bytes.size() ||
                segment.bytes.size() - relocation.code_offset <
                    kAarch64PointerMaterializationBytes) {
                error = "cached native relocation patch sequence out of bounds";
                return false;
            }
            if (!ValidPointerMaterializationSequence(segment, relocation)) {
                error = "cached native relocation does not reference fixed MOVZ/MOVK sequence";
                return false;
            }
            switch (relocation.kind) {
            case JitNativeRelocationKind::IrBlock:
                if (relocation.target_index != 0) {
                    error = "cached IR-block relocation has nonzero target index";
                    return false;
                }
                break;
            case JitNativeRelocationKind::IrInstruction:
                if (relocation.target_index >= record.block.instructions.size()) {
                    error = "cached IR-instruction relocation target out of bounds";
                    return false;
                }
                break;
            case JitNativeRelocationKind::SegmentAddress:
                if (relocation.target_index >= record.native_segments.size()) {
                    error = "cached segment-address relocation target out of bounds";
                    return false;
                }
                break;
            case JitNativeRelocationKind::ModuleRelative:
            case JitNativeRelocationKind::BackendRelative:
                break;
            case JitNativeRelocationKind::ChainPatchCell:
                if (relocation.target_index > 1) {
                    error = "cached direct-link relocation edge out of bounds";
                    return false;
                }
                break;
            case JitNativeRelocationKind::StableExternal:
                if (relocation.target_index == 0 || relocation.addend != 0) {
                    error = "invalid cached stable-external relocation";
                    return false;
                }
                break;
            case JitNativeRelocationKind::PolymorphicChainSite:
                if (relocation.target_index != 0 || relocation.addend != 0) {
                    error = "invalid cached indirect-link-site relocation";
                    return false;
                }
                break;
            }
            relocation_offsets.push_back(relocation.code_offset);
        }
        std::sort(relocation_offsets.begin(), relocation_offsets.end());
        for (std::size_t index = 1; index < relocation_offsets.size(); ++index) {
            if (relocation_offsets[index] - relocation_offsets[index - 1] <
                kAarch64PointerMaterializationBytes) {
                error = "overlapping cached native relocation patch sequences";
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] bool ValidateRecord(const JitIrCacheRecord& record,
                                  const JitIrCacheLimits& limits, std::string& error,
                                  std::uint64_t* native_bytes_out = nullptr,
                                  std::uint64_t* native_relocations_out = nullptr) {
    const LsxDecodedRegion& block = record.block;
    if (record.guest_rip != block.start_rip || block.end_rip < block.start_rip ||
        block.end_rip - block.start_rip > std::numeric_limits<std::uint32_t>::max()) {
        error = "invalid cached block guest span";
        return false;
    }
    if (block.instructions.size() > limits.max_instructions_per_block) {
        error = "cached block exceeds instruction limit";
        return false;
    }
    if (block.diagnostic.size() > limits.max_diagnostic_bytes) {
        error = "cached block diagnostic exceeds limit";
        return false;
    }

    std::uint64_t expected_rip = block.start_rip;
    for (const LsxDecodedOp& instruction : block.instructions) {
        if (instruction.guest_rip != expected_rip ||
            expected_rip > std::numeric_limits<std::uint64_t>::max() - instruction.length ||
            !ValidateInstruction(instruction, limits, error)) {
            if (error.empty()) {
                error = "non-contiguous cached block";
            }
            return false;
        }
        expected_rip += instruction.length;
    }
    if (expected_rip != block.end_rip) {
        error = "cached block end does not match instructions";
        return false;
    }
    std::uint64_t native_bytes = 0;
    std::uint64_t native_relocations = 0;
    if (!ValidateNativeSegments(record, limits, native_bytes, native_relocations, error)) {
        return false;
    }
    if (native_bytes_out != nullptr) {
        *native_bytes_out = native_bytes;
    }
    if (native_relocations_out != nullptr) {
        *native_relocations_out = native_relocations;
    }
    return true;
}

void EncodeOperandRecord(ByteWriter& writer, const LsxOperandRecord& operand) {
    writer.U8(operand.id);
    writer.U8(static_cast<std::uint8_t>(operand.visibility));
    writer.U8(operand.actions);
    writer.U8(static_cast<std::uint8_t>(operand.encoding));
    writer.U16(operand.size);
    writer.U16(static_cast<std::uint16_t>(operand.element_type));
    writer.U16(operand.element_size);
    writer.U16(operand.element_count);
    writer.U8(operand.attributes);
    writer.U8(static_cast<std::uint8_t>(operand.type));

    switch (operand.type) {
    case kOperandKindUnused:
        break;
    case kOperandKindRegister:
        writer.U32(static_cast<std::uint32_t>(operand.reg.value));
        break;
    case kOperandKindMemory:
        writer.U32(static_cast<std::uint32_t>(operand.mem.type));
        writer.U32(static_cast<std::uint32_t>(operand.mem.segment));
        writer.U32(static_cast<std::uint32_t>(operand.mem.base));
        writer.U32(static_cast<std::uint32_t>(operand.mem.index));
        writer.U8(operand.mem.scale);
        writer.U8(operand.mem.disp.offset);
        writer.U8(operand.mem.disp.size);
        writer.U64(std::bit_cast<std::uint64_t>(operand.mem.disp.value));
        break;
    case kOperandKindPointer:
        writer.U16(operand.ptr.segment);
        writer.U32(operand.ptr.offset);
        break;
    case kOperandKindImmediate:
        writer.U8(static_cast<std::uint8_t>(operand.imm.is_signed));
        writer.U8(static_cast<std::uint8_t>(operand.imm.is_relative));
        writer.U8(operand.imm.offset);
        writer.U8(operand.imm.size);
        writer.U64(operand.imm.value.u);
        break;
    default:
        break;
    }
}

void WriteInstruction(ByteWriter& writer, const LsxDecodedOp& instruction) {
    writer.U64(instruction.guest_rip);
    writer.U16(instruction.attributes);
    writer.U8(instruction.length);
    writer.U8(static_cast<std::uint8_t>(instruction.category));
    writer.U32(instruction.mnemonic);
    std::uint8_t flags = 0;
    flags |= instruction.can_execute_vector ? kInstructionFlagCanExecuteVector : 0;
    flags |= instruction.can_execute_scalar_float ? kInstructionFlagCanExecuteScalarFloat : 0;
    flags |= instruction.can_jit_initial ? kInstructionFlagCanJitInitial : 0;
    flags |= instruction.terminates_block ? kInstructionFlagTerminatesBlock : 0;
    writer.U8(flags);
    writer.U8(instruction.operand_count);
    writer.Bytes(instruction.bytes);
    writer.U64(static_cast<std::uint64_t>(instruction.decoded.attributes));
    writer.U32(instruction.decoded.mnemonic);
    writer.U8(instruction.decoded.length);
    writer.U8(instruction.decoded.encoding);
    writer.U8(instruction.decoded.address_width);
    writer.U8(instruction.decoded.operand_width);
    writer.U8(instruction.decoded.operand_count);
    writer.U8(instruction.decoded.operand_count_visible);
    for (std::uint8_t index = 0; index < instruction.operand_count; ++index) {
        EncodeOperandRecord(writer, instruction.operands[index]);
    }
}

void WriteNativeRelocation(ByteWriter& writer,
                           const JitNativeRelocation& relocation) {
    writer.U8(static_cast<std::uint8_t>(relocation.kind));
    writer.U8(relocation.register_index);
    writer.U16(0); // Reserved; a nonzero value requires a future record schema.
    writer.U32(relocation.code_offset);
    writer.U32(relocation.target_index);
    writer.U64(std::bit_cast<std::uint64_t>(relocation.addend));
}

[[nodiscard]] bool ReadNativeRelocation(ByteReader& reader,
                                        JitNativeRelocation& relocation,
                                        std::string& error) {
    std::uint8_t kind = 0;
    std::uint16_t reserved = 0;
    std::uint64_t addend = 0;
    if (!reader.U8(kind) || !reader.U8(relocation.register_index) || !reader.U16(reserved) ||
        !reader.U32(relocation.code_offset) || !reader.U32(relocation.target_index) ||
        !reader.U64(addend)) {
        error = "truncated cached native relocation";
        return false;
    }
    relocation.kind = static_cast<JitNativeRelocationKind>(kind);
    relocation.addend = std::bit_cast<std::int64_t>(addend);
    if (reserved != 0 || !ValidNativeRelocationKind(relocation.kind)) {
        error = "invalid cached native relocation kind or reserved bits";
        return false;
    }
    return true;
}

[[nodiscard]] bool DecodeOperandRecord(ByteReader& reader, LsxOperandRecord& operand,
                                       std::string& error) {
    operand = LsxOperandRecord{};
    std::uint8_t visibility = 0;
    std::uint8_t encoding = 0;
    std::uint16_t element_type = 0;
    std::uint8_t type = 0;
    if (!reader.U8(operand.id) || !reader.U8(visibility) || !reader.U8(operand.actions) ||
        !reader.U8(encoding) || !reader.U16(operand.size) || !reader.U16(element_type) ||
        !reader.U16(operand.element_size) || !reader.U16(operand.element_count) ||
        !reader.U8(operand.attributes) || !reader.U8(type)) {
        error = "truncated operand";
        return false;
    }
    operand.visibility = visibility;
    operand.encoding = encoding;
    operand.element_type = element_type;
    operand.type = type;

    switch (operand.type) {
    case kOperandKindUnused:
        break;
    case kOperandKindRegister: {
        std::uint32_t value = 0;
        if (!reader.U32(value)) {
            error = "truncated register operand";
            return false;
        }
        operand.reg.value = value;
        break;
    }
    case kOperandKindMemory: {
        std::uint32_t memory_type = 0;
        std::uint32_t segment = 0;
        std::uint32_t base = 0;
        std::uint32_t index = 0;
        std::uint64_t displacement = 0;
        if (!reader.U32(memory_type) || !reader.U32(segment) || !reader.U32(base) ||
            !reader.U32(index) || !reader.U8(operand.mem.scale) ||
            !reader.U8(operand.mem.disp.offset) || !reader.U8(operand.mem.disp.size) ||
            !reader.U64(displacement)) {
            error = "truncated memory operand";
            return false;
        }
        operand.mem.type = memory_type;
        operand.mem.segment = segment;
        operand.mem.base = base;
        operand.mem.index = index;
        operand.mem.disp.value = std::bit_cast<std::int64_t>(displacement);
        break;
    }
    case kOperandKindPointer:
        if (!reader.U16(operand.ptr.segment) || !reader.U32(operand.ptr.offset)) {
            error = "truncated pointer operand";
            return false;
        }
        break;
    case kOperandKindImmediate: {
        std::uint8_t is_signed = 0;
        std::uint8_t is_relative = 0;
        if (!reader.U8(is_signed) || !reader.U8(is_relative) || !reader.U8(operand.imm.offset) ||
            !reader.U8(operand.imm.size) || !reader.U64(operand.imm.value.u)) {
            error = "truncated immediate operand";
            return false;
        }
        operand.imm.is_signed = is_signed;
        operand.imm.is_relative = is_relative;
        break;
    }
    default:
        error = "invalid operand type";
        return false;
    }
    return ValidateOperand(operand, error);
}

[[nodiscard]] bool ReadInstruction(ByteReader& reader, LsxDecodedOp& instruction,
                                   const JitIrCacheLimits& limits, std::string& error) {
    std::uint8_t category = 0;
    std::uint8_t flags = 0;
    std::span<const std::uint8_t> instruction_bytes;
    std::uint64_t decoded_attributes = 0;
    if (!reader.U64(instruction.guest_rip) || !reader.U16(instruction.attributes) ||
        !reader.U8(instruction.length) || !reader.U8(category) ||
        !reader.U32(instruction.mnemonic) || !reader.U8(flags) ||
        !reader.U8(instruction.operand_count) ||
        !reader.Bytes(instruction.bytes.size(), instruction_bytes) ||
        !reader.U64(decoded_attributes) || !reader.U32(instruction.decoded.mnemonic) ||
        !reader.U8(instruction.decoded.length) || !reader.U8(instruction.decoded.encoding) ||
        !reader.U8(instruction.decoded.address_width) ||
        !reader.U8(instruction.decoded.operand_width) ||
        !reader.U8(instruction.decoded.operand_count) ||
        !reader.U8(instruction.decoded.operand_count_visible)) {
        error = "truncated instruction";
        return false;
    }
    if ((flags & ~kKnownInstructionFlags) != 0) {
        error = "unknown cached instruction flags";
        return false;
    }
    std::copy(instruction_bytes.begin(), instruction_bytes.end(), instruction.bytes.begin());
    instruction.category = static_cast<LsxOpClass>(category);
    instruction.can_execute_vector = (flags & kInstructionFlagCanExecuteVector) != 0;
    instruction.can_execute_scalar_float = (flags & kInstructionFlagCanExecuteScalarFloat) != 0;
    instruction.can_jit_initial = (flags & kInstructionFlagCanJitInitial) != 0;
    instruction.terminates_block = (flags & kInstructionFlagTerminatesBlock) != 0;
    instruction.decoded.attributes = decoded_attributes;

    if (instruction.operand_count > instruction.operands.size()) {
        error = "cached instruction operand count exceeds storage";
        return false;
    }
    for (std::uint8_t index = 0; index < instruction.operand_count; ++index) {
        if (!DecodeOperandRecord(reader, instruction.operands[index], error)) {
            return false;
        }
    }
    if (!ValidateInstruction(instruction, limits, error)) {
        return false;
    }

    // These fields are compiler products, not decoded guest facts.  Keep their serialized bytes
    // only for format compatibility/strict validation, then discard them fail-closed.  The
    // current JIT production classifiers reconstruct them before a loaded block can be
    // lowered or published.
    instruction.attributes = 0;
    instruction.category = LsxOpClass::Unsupported;
    instruction.can_execute_vector = false;
    instruction.can_execute_scalar_float = false;
    instruction.can_execute_x87 = false;
    instruction.can_jit_initial = false;
    instruction.terminates_block = false;
    return true;
}

[[nodiscard]] bool WriteRecord(const JitIrCacheRecord& record, ByteWriter& payload,
                               const JitIrCacheLimits& limits, std::string& error) {
    if (!ValidateRecord(record, limits, error)) {
        return false;
    }

    ByteWriter body{limits.max_file_bytes};
    body.U16(kRecordSchemaVersion);
    std::uint16_t flags = 0;
    flags |= record.block.can_jit_initial ? kBlockFlagCanJitInitial : 0;
    flags |= record.block.decode_failed ? kBlockFlagDecodeFailed : 0;
    body.U16(flags);
    body.U64(record.guest_rip);
    body.U64(record.guest_hash);
    body.U64(record.block.start_rip);
    body.U64(record.block.end_rip);
    body.U32(static_cast<std::uint32_t>(record.block.instructions.size()));
    body.U32(static_cast<std::uint32_t>(record.block.diagnostic.size()));
    body.U32(record.native_flags);
    body.U32(static_cast<std::uint32_t>(record.native_segments.size()));
    body.U32(record.entry_segment_index);
    body.U32(record.direct_segment_index);
    body.Bytes(std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(record.block.diagnostic.data()),
        record.block.diagnostic.size()});
    for (const LsxDecodedOp& instruction : record.block.instructions) {
        WriteInstruction(body, instruction);
    }
    for (const JitNativeSegment& segment : record.native_segments) {
        body.U32(static_cast<std::uint32_t>(segment.bytes.size()));
        body.U32(static_cast<std::uint32_t>(segment.relocs.size()));
        body.U64(FingerprintNativeSegment(segment.bytes));
        body.Bytes(segment.bytes);
        for (const JitNativeRelocation& relocation : segment.relocs) {
            WriteNativeRelocation(body, relocation);
        }
    }
    if (!body.ok() || body.bytes().size() > std::numeric_limits<std::uint32_t>::max()) {
        error = "serialized cache record exceeds size limit";
        return false;
    }
    payload.U32(static_cast<std::uint32_t>(body.bytes().size()));
    payload.Bytes(body.bytes());
    if (!payload.ok()) {
        error = "serialized cache payload exceeds size limit";
        return false;
    }
    return true;
}

[[nodiscard]] bool ReadRecord(ByteReader& body, JitIrCacheRecord& record,
                              std::uint64_t& total_instructions,
                              std::uint64_t& total_native_bytes,
                              std::uint64_t& total_native_relocations,
                              const JitIrCacheLimits& limits, std::string& error) {
    std::uint16_t schema = 0;
    std::uint16_t flags = 0;
    std::uint32_t instruction_count = 0;
    std::uint32_t diagnostic_size = 0;
    std::uint32_t native_segment_count = 0;
    if (!body.U16(schema) || !body.U16(flags) || !body.U64(record.guest_rip) ||
        !body.U64(record.guest_hash) || !body.U64(record.block.start_rip) ||
        !body.U64(record.block.end_rip) || !body.U32(instruction_count) ||
        !body.U32(diagnostic_size) || !body.U32(record.native_flags) ||
        !body.U32(native_segment_count) || !body.U32(record.entry_segment_index) ||
        !body.U32(record.direct_segment_index)) {
        error = "truncated cache record header";
        return false;
    }
    if (schema != kRecordSchemaVersion) {
        error = "unsupported cache record schema";
        return false;
    }
    if ((flags & ~kKnownBlockFlags) != 0) {
        error = "unknown cached block flags";
        return false;
    }
    if (instruction_count > limits.max_instructions_per_block ||
        instruction_count > limits.max_total_instructions ||
        total_instructions > limits.max_total_instructions - instruction_count) {
        error = "cached instruction count exceeds limit";
        return false;
    }
    if (diagnostic_size > limits.max_diagnostic_bytes) {
        error = "cached diagnostic exceeds limit";
        return false;
    }
    if (native_segment_count > limits.max_native_segments_per_record) {
        error = "cached native segment count exceeds limit";
        return false;
    }

    std::span<const std::uint8_t> diagnostic;
    if (!body.Bytes(diagnostic_size, diagnostic)) {
        error = "truncated cache diagnostic";
        return false;
    }
    // can_jit_initial is derived from current instruction capabilities.  Deserialize it only as
    // a known schema bit, then leave the block fail-closed until production canonicalization.
    record.block.can_jit_initial = false;
    record.block.decode_failed = (flags & kBlockFlagDecodeFailed) != 0;
    if (diagnostic.empty()) {
        record.block.diagnostic.clear();
    } else {
        record.block.diagnostic.assign(reinterpret_cast<const char*>(diagnostic.data()),
                                       diagnostic.size());
    }
    record.block.instructions.resize(instruction_count);
    for (std::uint32_t index = 0; index < instruction_count; ++index) {
        if (!ReadInstruction(body, record.block.instructions[index], limits, error)) {
            return false;
        }
    }

    std::uint64_t record_native_bytes = 0;
    std::uint64_t record_native_relocations = 0;
    record.native_segments.resize(native_segment_count);
    for (JitNativeSegment& segment : record.native_segments) {
        std::uint32_t segment_size = 0;
        std::uint32_t relocation_count = 0;
        std::uint64_t segment_hash = 0;
        if (!body.U32(segment_size) || !body.U32(relocation_count) ||
            !body.U64(segment_hash)) {
            error = "truncated cached native segment header";
            return false;
        }
        if (segment_size == 0 || segment_size % sizeof(std::uint32_t) != 0 ||
            segment_size > limits.max_native_segment_bytes ||
            relocation_count > limits.max_native_relocations_per_segment) {
            error = "cached native segment exceeds limit";
            return false;
        }
        if (segment_size > limits.max_total_native_bytes - record_native_bytes ||
            relocation_count >
                limits.max_total_native_relocations - record_native_relocations ||
            record_native_bytes + segment_size >
                limits.max_total_native_bytes - total_native_bytes ||
            record_native_relocations + relocation_count >
                limits.max_total_native_relocations - total_native_relocations) {
            error = "cached native data exceeds total limit";
            return false;
        }

        std::span<const std::uint8_t> segment_bytes;
        if (!body.Bytes(segment_size, segment_bytes)) {
            error = "truncated cached native segment bytes";
            return false;
        }
        if (FingerprintNativeSegment(segment_bytes) != segment_hash) {
            error = "cached native segment checksum mismatch";
            return false;
        }
        segment.bytes.assign(segment_bytes.begin(), segment_bytes.end());

        constexpr std::size_t kSerializedRelocationBytes = 20;
        if (relocation_count > body.remaining() / kSerializedRelocationBytes) {
            error = "truncated cached native relocation table";
            return false;
        }
        segment.relocs.resize(relocation_count);
        for (JitNativeRelocation& relocation : segment.relocs) {
            if (!ReadNativeRelocation(body, relocation, error)) {
                return false;
            }
        }
        record_native_bytes += segment_size;
        record_native_relocations += relocation_count;
    }
    total_instructions += instruction_count;
    std::uint64_t validated_native_bytes = 0;
    std::uint64_t validated_native_relocations = 0;
    if (!ValidateRecord(record, limits, error, &validated_native_bytes,
                        &validated_native_relocations) ||
        validated_native_bytes != record_native_bytes ||
        validated_native_relocations != record_native_relocations) {
        if (error.empty()) {
            error = "cached native accounting mismatch";
        }
        return false;
    }
    total_native_bytes += record_native_bytes;
    total_native_relocations += record_native_relocations;
    return true;
}

[[nodiscard]] std::string ErrnoDetail(const char* operation) {
    std::ostringstream message;
    message << operation << ": " << std::error_code(errno, std::generic_category()).message();
    return message.str();
}

[[nodiscard]] std::uint64_t CurrentProcessIdValue() noexcept {
#ifdef _WIN32
    return static_cast<std::uint64_t>(::GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

[[nodiscard]] bool WriteDurableFile(const std::filesystem::path& path,
                                    const std::span<const std::uint8_t> bytes, std::string& error) {
#ifdef _WIN32
    std::FILE* file = nullptr;
    if (::_wfopen_s(&file, path.c_str(), L"wb") != 0) {
        file = nullptr;
    }
#else
    std::FILE* file = std::fopen(path.c_str(), "wb");
#endif
    if (file == nullptr) {
        error = ErrnoDetail("open cache temporary file");
        return false;
    }

    bool success = true;
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const std::size_t written =
            std::fwrite(bytes.data() + offset, 1, bytes.size() - offset, file);
        if (written == 0) {
            error = ErrnoDetail("write cache temporary file");
            success = false;
            break;
        }
        offset += written;
    }
    if (success && std::fflush(file) != 0) {
        error = ErrnoDetail("flush cache temporary file");
        success = false;
    }
    if (success) {
#ifdef _WIN32
        if (::_commit(::_fileno(file)) != 0) {
#else
        if (::fsync(::fileno(file)) != 0) {
#endif
            error = ErrnoDetail("sync cache temporary file");
            success = false;
        }
    }
    if (std::fclose(file) != 0 && success) {
        error = ErrnoDetail("close cache temporary file");
        success = false;
    }
    return success;
}

[[nodiscard]] bool ReplaceFileAtomically(const std::filesystem::path& from,
                                         const std::filesystem::path& to, std::string& error) {
#ifdef _WIN32
    if (!::MoveFileExW(from.c_str(), to.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        std::ostringstream message;
        message << "replace cache file: win32 error " << ::GetLastError();
        error = message.str();
        return false;
    }
#else
    if (::rename(from.c_str(), to.c_str()) != 0) {
        error = ErrnoDetail("replace cache file");
        return false;
    }
#endif
    return true;
}

} // namespace

bool SerializeJitIrCache(const JitIrCacheIdentity& identity,
                              const std::span<const JitIrCacheRecord> records,
                              std::vector<std::uint8_t>& output, std::string& error,
                              const JitIrCacheLimits& limits) {
    output.clear();
    error.clear();
    if (limits.max_file_bytes < kHeaderSize || records.size() > limits.max_records) {
        error = "cache exceeds configured record/file limit";
        return false;
    }

    try {
        ByteWriter payload{limits.max_file_bytes - kHeaderSize};
        std::uint64_t total_instructions = 0;
        std::uint64_t total_native_bytes = 0;
        std::uint64_t total_native_relocations = 0;
        for (const JitIrCacheRecord& record : records) {
            if (record.block.instructions.size() > limits.max_total_instructions ||
                total_instructions >
                    limits.max_total_instructions - record.block.instructions.size()) {
                error = "cache exceeds total instruction limit";
                return false;
            }
            total_instructions += record.block.instructions.size();
            std::uint64_t record_native_bytes = 0;
            std::uint64_t record_native_relocations = 0;
            if (!ValidateRecord(record, limits, error, &record_native_bytes,
                                &record_native_relocations)) {
                return false;
            }
            if (record_native_bytes > limits.max_total_native_bytes - total_native_bytes ||
                record_native_relocations >
                    limits.max_total_native_relocations - total_native_relocations) {
                error = "cache exceeds total native code/relocation limit";
                return false;
            }
            total_native_bytes += record_native_bytes;
            total_native_relocations += record_native_relocations;
            if (!WriteRecord(record, payload, limits, error)) {
                return false;
            }
        }

        ByteWriter file{limits.max_file_bytes};
        file.Bytes(kCacheMagic);
        file.U32(kJitIrCacheFormatVersion);
        file.U32(kHeaderSize);
        file.U64(identity.jit_abi_version);
        file.Bytes(identity.content_key);
        file.U64(records.size());
        file.U64(payload.bytes().size());
        file.U64(FingerprintPayload(payload.bytes()));
        file.U64(0); // Header flags/reserved for a future compatible extension.
        file.Bytes(payload.bytes());
        if (!file.ok()) {
            error = "serialized cache exceeds file limit";
            return false;
        }
        output = file.TakeBytes();
        return true;
    } catch (const std::exception& exception) {
        error = std::string{"cache serialization failed: "} + exception.what();
        output.clear();
        return false;
    }
}

JitIrCacheLoadResult DeserializeJitIrCache(
    const std::span<const std::uint8_t> bytes, const JitIrCacheIdentity& expected_identity,
    const JitIrCacheLimits& limits) {
    JitIrCacheLoadResult result{};
    if (bytes.size() > limits.max_file_bytes) {
        result.status = JitIrCacheStatus::LimitExceeded;
        result.detail = "cache file exceeds configured size limit";
        return result;
    }
    if (bytes.size() < kHeaderSize) {
        result.status = JitIrCacheStatus::Corrupt;
        result.detail = "truncated cache header";
        return result;
    }

    try {
        ByteReader reader{bytes};
        std::span<const std::uint8_t> magic;
        std::uint32_t version = 0;
        std::uint32_t header_size = 0;
        JitIrCacheIdentity identity{};
        std::span<const std::uint8_t> content_key;
        std::uint64_t record_count = 0;
        std::uint64_t payload_size = 0;
        std::uint64_t payload_hash = 0;
        std::uint64_t header_flags = 0;
        if (!reader.Bytes(kCacheMagic.size(), magic) || !reader.U32(version) ||
            !reader.U32(header_size) || !reader.U64(identity.jit_abi_version) ||
            !reader.Bytes(identity.content_key.size(), content_key) || !reader.U64(record_count) ||
            !reader.U64(payload_size) || !reader.U64(payload_hash) || !reader.U64(header_flags)) {
            result.status = JitIrCacheStatus::Corrupt;
            result.detail = "truncated cache header";
            return result;
        }
        std::copy(content_key.begin(), content_key.end(), identity.content_key.begin());
        if (!std::equal(magic.begin(), magic.end(), kCacheMagic.begin())) {
            result.status = JitIrCacheStatus::Corrupt;
            result.detail = "invalid cache magic";
            return result;
        }
        if (version != kJitIrCacheFormatVersion) {
            result.status = JitIrCacheStatus::UnsupportedVersion;
            result.detail = "unsupported cache format version";
            return result;
        }
        if (header_size != kHeaderSize || header_flags != 0) {
            result.status = JitIrCacheStatus::Corrupt;
            result.detail = "invalid cache header size or flags";
            return result;
        }
        if (identity != expected_identity) {
            result.status = JitIrCacheStatus::IdentityMismatch;
            result.detail = "cache title/content key or JIT ABI does not match";
            return result;
        }
        if (record_count > limits.max_records) {
            result.status = JitIrCacheStatus::LimitExceeded;
            result.detail = "cache record count exceeds configured limit";
            return result;
        }
        if (payload_size != reader.remaining()) {
            result.status = JitIrCacheStatus::Corrupt;
            result.detail = "cache payload length mismatch";
            return result;
        }

        std::span<const std::uint8_t> payload;
        if (!reader.Bytes(static_cast<std::size_t>(payload_size), payload) ||
            FingerprintPayload(payload) != payload_hash) {
            result.status = JitIrCacheStatus::Corrupt;
            result.detail = "cache payload checksum mismatch";
            return result;
        }

        ByteReader payload_reader{payload};
        result.records.reserve(static_cast<std::size_t>(record_count));
        std::uint64_t total_instructions = 0;
        std::uint64_t total_native_bytes = 0;
        std::uint64_t total_native_relocations = 0;
        for (std::uint64_t index = 0; index < record_count; ++index) {
            std::uint32_t body_size = 0;
            if (!payload_reader.U32(body_size) || body_size > payload_reader.remaining()) {
                result.status = JitIrCacheStatus::Corrupt;
                result.detail = "truncated cache record";
                result.records.clear();
                return result;
            }
            std::span<const std::uint8_t> body_bytes;
            if (!payload_reader.Bytes(body_size, body_bytes)) {
                result.status = JitIrCacheStatus::Corrupt;
                result.detail = "truncated cache record body";
                result.records.clear();
                return result;
            }
            ByteReader body{body_bytes};
            JitIrCacheRecord record{};
            std::string error;
            if (!ReadRecord(body, record, total_instructions, total_native_bytes,
                            total_native_relocations, limits, error) ||
                !body.empty()) {
                result.status = error.find("limit") != std::string::npos
                                    ? JitIrCacheStatus::LimitExceeded
                                    : JitIrCacheStatus::Corrupt;
                result.detail = error.empty() ? "cache record has trailing bytes" : error;
                result.records.clear();
                return result;
            }
            result.records.push_back(std::move(record));
        }
        if (!payload_reader.empty()) {
            result.status = JitIrCacheStatus::Corrupt;
            result.detail = "cache payload has trailing bytes";
            result.records.clear();
            return result;
        }
        result.status = JitIrCacheStatus::Ok;
        result.detail.clear();
        return result;
    } catch (const std::exception& exception) {
        result.status = JitIrCacheStatus::Corrupt;
        result.detail = std::string{"cache deserialization failed: "} + exception.what();
        result.records.clear();
        return result;
    }
}

JitIrCacheLoadResult LoadJitIrCacheFile(const std::filesystem::path& path,
                                                  const JitIrCacheIdentity& expected_identity,
                                                  const JitIrCacheLimits& limits) {
    JitIrCacheLoadResult result{};
    std::error_code filesystem_error;
    const bool exists = std::filesystem::exists(path, filesystem_error);
    if (filesystem_error) {
        result.status = JitIrCacheStatus::IoError;
        result.detail = "stat cache file: " + filesystem_error.message();
        return result;
    }
    if (!exists) {
        result.status = JitIrCacheStatus::NotFound;
        result.detail = "cache file not found";
        return result;
    }
    const std::uintmax_t file_size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error) {
        result.status = JitIrCacheStatus::IoError;
        result.detail = "read cache file size: " + filesystem_error.message();
        return result;
    }
    if (file_size > limits.max_file_bytes ||
        file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()) ||
        file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        result.status = JitIrCacheStatus::LimitExceeded;
        result.detail = "cache file exceeds configured size limit";
        return result;
    }

    try {
        std::ifstream stream{path, std::ios::binary};
        if (!stream) {
            result.status = JitIrCacheStatus::IoError;
            result.detail = "open cache file failed";
            return result;
        }
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
        if (!bytes.empty()) {
            stream.read(reinterpret_cast<char*>(bytes.data()),
                        static_cast<std::streamsize>(bytes.size()));
            if (!stream || stream.gcount() != static_cast<std::streamsize>(bytes.size())) {
                result.status = JitIrCacheStatus::IoError;
                result.detail = "short read from cache file";
                return result;
            }
        }
        return DeserializeJitIrCache(bytes, expected_identity, limits);
    } catch (const std::exception& exception) {
        result.status = JitIrCacheStatus::IoError;
        result.detail = std::string{"read cache file failed: "} + exception.what();
        return result;
    }
}

bool SaveJitIrCacheFileAtomic(const std::filesystem::path& path,
                                   const JitIrCacheIdentity& identity,
                                   const std::span<const JitIrCacheRecord> records,
                                   std::string& error, const JitIrCacheLimits& limits) {
    std::vector<std::uint8_t> bytes;
    if (!SerializeJitIrCache(identity, records, bytes, error, limits)) {
        return false;
    }

    try {
        std::filesystem::path parent = path.parent_path();
        if (parent.empty()) {
            parent = ".";
        }
        std::error_code filesystem_error;
        std::filesystem::create_directories(parent, filesystem_error);
        if (filesystem_error) {
            error = "create cache directory: " + filesystem_error.message();
            return false;
        }

        static std::atomic<std::uint64_t> sequence{0};
        const std::uint64_t stamp =
            static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        std::ostringstream suffix;
        suffix << ".tmp." << CurrentProcessIdValue() << '.' << stamp << '.'
               << sequence.fetch_add(1, std::memory_order_relaxed);
        std::filesystem::path temporary = path;
        temporary += suffix.str();

        if (!WriteDurableFile(temporary, bytes, error)) {
            std::filesystem::remove(temporary, filesystem_error);
            return false;
        }
        if (!ReplaceFileAtomically(temporary, path, error)) {
            std::filesystem::remove(temporary, filesystem_error);
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        error = std::string{"write cache file failed: "} + exception.what();
        return false;
    }
}

} // namespace Executor::Jit
