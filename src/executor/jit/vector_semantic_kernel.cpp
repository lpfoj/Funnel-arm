// SPDX-FileCopyrightText: Copyright 2026 Executor Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/jit/vector_semantic_kernel.h"

#include <array>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Executor::Jit::VectorSemantic {
namespace {

constexpr std::size_t kMaximumVectorBytes = 32;

bool ValidWidth(const std::size_t width) noexcept {
    return width != 0 && width <= kMaximumVectorBytes;
}

template <typename T>
T LoadLane(const std::uint8_t* source) noexcept {
    T value{};
    std::memcpy(&value, source, sizeof(value));
    return value;
}

template <typename T>
void StoreLane(std::uint8_t* destination, const T value) noexcept {
    std::memcpy(destination, &value, sizeof(value));
}

std::uint64_t ApplyBooleanWord(const std::uint64_t left,
                               const std::uint64_t right,
                               const BooleanOperation operation) noexcept {
    switch (operation) {
    case BooleanOperation::ExclusiveOr:
        return left ^ right;
    case BooleanOperation::Conjunction:
        return left & right;
    case BooleanOperation::Disjunction:
        return left | right;
    case BooleanOperation::ConjunctionWithInvertedLeft:
        return right & ~left;
    }
    return left ^ right;
}

#if defined(__aarch64__)
uint8x16_t ApplyBooleanNeon(const uint8x16_t left, const uint8x16_t right,
                            const BooleanOperation operation) noexcept {
    switch (operation) {
    case BooleanOperation::ExclusiveOr:
        return veorq_u8(left, right);
    case BooleanOperation::Conjunction:
        return vandq_u8(left, right);
    case BooleanOperation::Disjunction:
        return vorrq_u8(left, right);
    case BooleanOperation::ConjunctionWithInvertedLeft:
        return vbicq_u8(right, left);
    }
    return veorq_u8(left, right);
}
#endif

} // namespace

bool ApplyBoolean(void* const destination, const void* const left,
                  const void* const right, const std::size_t width,
                  const BooleanOperation operation) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        !ValidWidth(width)) {
        return false;
    }

    std::array<std::uint8_t, kMaximumVectorBytes> left_snapshot{};
    std::array<std::uint8_t, kMaximumVectorBytes> right_snapshot{};
    std::array<std::uint8_t, kMaximumVectorBytes> result{};
    std::memcpy(left_snapshot.data(), left, width);
    std::memcpy(right_snapshot.data(), right, width);

#if defined(__aarch64__)
    if (width == 16 || width == 32) {
        for (std::size_t offset = 0; offset < width; offset += 16) {
            const uint8x16_t lhs = vld1q_u8(left_snapshot.data() + offset);
            const uint8x16_t rhs = vld1q_u8(right_snapshot.data() + offset);
            vst1q_u8(result.data() + offset,
                     ApplyBooleanNeon(lhs, rhs, operation));
        }
    } else
#endif
    {
        std::size_t offset = 0;
        for (; offset + sizeof(std::uint64_t) <= width;
             offset += sizeof(std::uint64_t)) {
            StoreLane(result.data() + offset,
                      ApplyBooleanWord(LoadLane<std::uint64_t>(
                                           left_snapshot.data() + offset),
                                       LoadLane<std::uint64_t>(
                                           right_snapshot.data() + offset),
                                       operation));
        }
        for (; offset < width; ++offset) {
            result[offset] = static_cast<std::uint8_t>(ApplyBooleanWord(
                left_snapshot[offset], right_snapshot[offset], operation));
        }
    }
    std::memcpy(destination, result.data(), result.size());
    return true;
}

PackedBitTestResult TestPackedBits(const void* const left,
                                   const void* const right,
                                   const std::size_t width) noexcept {
    if (left == nullptr || right == nullptr || !ValidWidth(width)) {
        return {};
    }

    const auto* lhs = static_cast<const std::uint8_t*>(left);
    const auto* rhs = static_cast<const std::uint8_t*>(right);
    bool intersection_nonzero = false;
    bool masked_complement_nonzero = false;
#if defined(__aarch64__)
    if (width == 16 || width == 32) {
        uint8x16_t intersection = vdupq_n_u8(0);
        uint8x16_t masked_complement = vdupq_n_u8(0);
        for (std::size_t offset = 0; offset < width; offset += 16) {
            const uint8x16_t left_lane = vld1q_u8(lhs + offset);
            const uint8x16_t right_lane = vld1q_u8(rhs + offset);
            intersection = vorrq_u8(intersection,
                                    vandq_u8(left_lane, right_lane));
            masked_complement = vorrq_u8(
                masked_complement, vbicq_u8(right_lane, left_lane));
        }
        intersection_nonzero = vmaxvq_u8(intersection) != 0;
        masked_complement_nonzero = vmaxvq_u8(masked_complement) != 0;
    } else
#endif
    {
        for (std::size_t offset = 0; offset < width; ++offset) {
            intersection_nonzero |= (lhs[offset] & rhs[offset]) != 0;
            masked_complement_nonzero |=
                (static_cast<std::uint8_t>(~lhs[offset]) & rhs[offset]) != 0;
        }
    }
    return {!intersection_nonzero, !masked_complement_nonzero};
}

bool DuplicateDwordParity(void* const destination, const void* const source,
                          const std::size_t width,
                          const bool odd_lanes) noexcept {
    if (destination == nullptr || source == nullptr ||
        (width != 16 && width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> source_snapshot{};
    std::array<std::uint8_t, kMaximumVectorBytes> result{};
    std::memcpy(source_snapshot.data(), source, width);
#if defined(__aarch64__)
    for (std::size_t offset = 0; offset < width; offset += 16) {
        const uint32x4_t input = vreinterpretq_u32_u8(
            vld1q_u8(source_snapshot.data() + offset));
        const uint32x4_t selected = odd_lanes
            ? vuzp2q_u32(input, input)
            : vuzp1q_u32(input, input);
        vst1q_u8(result.data() + offset,
                 vreinterpretq_u8_u32(vzip1q_u32(selected, selected)));
    }
#else
    for (std::size_t base = 0; base < width; base += 16) {
        const std::size_t first = base + (odd_lanes ? 4u : 0u);
        const std::size_t second = base + (odd_lanes ? 12u : 8u);
        const std::uint32_t a = LoadLane<std::uint32_t>(source_snapshot.data() + first);
        const std::uint32_t b = LoadLane<std::uint32_t>(source_snapshot.data() + second);
        StoreLane(result.data() + base, a);
        StoreLane(result.data() + base + 4, a);
        StoreLane(result.data() + base + 8, b);
        StoreLane(result.data() + base + 12, b);
    }
#endif
    std::memcpy(destination, result.data(), result.size());
    return true;
}

bool DuplicateLowQwordPerLane(void* const destination, const void* const source,
                              const std::size_t width,
                              const bool zero_upper) noexcept {
    if (destination == nullptr || source == nullptr ||
        (width != 16 && width != 32)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> source_snapshot{};
    std::array<std::uint8_t, kMaximumVectorBytes> result{};
    std::memcpy(source_snapshot.data(), source, width);
#if defined(__aarch64__)
    for (std::size_t base = 0; base < width; base += 16) {
        const uint64x2_t input = vreinterpretq_u64_u8(
            vld1q_u8(source_snapshot.data() + base));
        vst1q_u8(result.data() + base,
                 vreinterpretq_u8_u64(vdupq_laneq_u64(input, 0)));
    }
#else
    for (std::size_t base = 0; base < width; base += 16) {
        const std::uint64_t value =
            LoadLane<std::uint64_t>(source_snapshot.data() + base);
        StoreLane(result.data() + base, value);
        StoreLane(result.data() + base + 8, value);
    }
#endif
    std::memcpy(destination, result.data(), width);
    if (zero_upper && width < result.size()) {
        std::memset(static_cast<std::uint8_t*>(destination) + width, 0,
                    result.size() - width);
    }
    return true;
}

bool AddOrSubtractPacked(void* const destination, const void* const left,
                         const void* const right,
                         const std::size_t element_width,
                         const std::size_t total_width,
                         const bool subtract) noexcept {
    if (destination == nullptr || left == nullptr || right == nullptr ||
        (total_width != 16 && total_width != 32) ||
        (element_width != 1 && element_width != 2 && element_width != 4 &&
         element_width != 8)) {
        return false;
    }
    std::array<std::uint8_t, kMaximumVectorBytes> lhs{};
    std::array<std::uint8_t, kMaximumVectorBytes> rhs{};
    std::array<std::uint8_t, kMaximumVectorBytes> result{};
    std::memcpy(lhs.data(), left, total_width);
    std::memcpy(rhs.data(), right, total_width);
#if defined(__aarch64__)
    for (std::size_t offset = 0; offset < total_width; offset += 16) {
        const uint8x16_t left_bytes = vld1q_u8(lhs.data() + offset);
        const uint8x16_t right_bytes = vld1q_u8(rhs.data() + offset);
        uint8x16_t output{};
        switch (element_width) {
        case 1:
            output = subtract ? vsubq_u8(left_bytes, right_bytes)
                              : vaddq_u8(left_bytes, right_bytes);
            break;
        case 2:
            output = vreinterpretq_u8_u16(subtract
                ? vsubq_u16(vreinterpretq_u16_u8(left_bytes),
                            vreinterpretq_u16_u8(right_bytes))
                : vaddq_u16(vreinterpretq_u16_u8(left_bytes),
                            vreinterpretq_u16_u8(right_bytes)));
            break;
        case 4:
            output = vreinterpretq_u8_u32(subtract
                ? vsubq_u32(vreinterpretq_u32_u8(left_bytes),
                            vreinterpretq_u32_u8(right_bytes))
                : vaddq_u32(vreinterpretq_u32_u8(left_bytes),
                            vreinterpretq_u32_u8(right_bytes)));
            break;
        default:
            output = vreinterpretq_u8_u64(subtract
                ? vsubq_u64(vreinterpretq_u64_u8(left_bytes),
                            vreinterpretq_u64_u8(right_bytes))
                : vaddq_u64(vreinterpretq_u64_u8(left_bytes),
                            vreinterpretq_u64_u8(right_bytes)));
            break;
        }
        vst1q_u8(result.data() + offset, output);
    }
#else
    for (std::size_t offset = 0; offset < total_width;
         offset += element_width) {
        std::uint64_t left_value = 0;
        std::uint64_t right_value = 0;
        std::memcpy(&left_value, lhs.data() + offset, element_width);
        std::memcpy(&right_value, rhs.data() + offset, element_width);
        const std::uint64_t output = subtract ? left_value - right_value
                                              : left_value + right_value;
        std::memcpy(result.data() + offset, &output, element_width);
    }
#endif
    std::memcpy(destination, result.data(), result.size());
    return true;
}

} // namespace Executor::Jit::VectorSemantic
