// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

namespace Executor::Jit::VectorSemantic {

enum class BooleanOperation : std::uint8_t {
    ExclusiveOr,
    Conjunction,
    Disjunction,
    ConjunctionWithInvertedLeft,
};

struct PackedBitTestResult {
    bool intersection_is_zero{};
    bool masked_complement_is_zero{};
};

// These kernels are pure, alias-safe transformations. They are shared by the
// interpreter/semantic leaf while the native emitter uses the same operation
// classification to produce direct NEON. Keeping state/flags outside this layer
// makes fault ordering and architectural writeback explicit at the caller.
bool ApplyBoolean(void* destination, const void* left, const void* right,
                  std::size_t width, BooleanOperation operation) noexcept;
PackedBitTestResult TestPackedBits(const void* left, const void* right,
                                   std::size_t width) noexcept;
bool DuplicateDwordParity(void* destination, const void* source,
                          std::size_t width, bool odd_lanes) noexcept;
bool DuplicateLowQwordPerLane(void* destination, const void* source,
                              std::size_t width, bool zero_upper) noexcept;
bool AddOrSubtractPacked(void* destination, const void* left, const void* right,
                         std::size_t element_width, std::size_t total_width,
                         bool subtract) noexcept;
bool ReplicateElement(void* destination, const void* element,
                      std::size_t element_width,
                      std::size_t total_width) noexcept;
bool SquareRootElements(void* destination, const void* source,
                        std::size_t element_width,
                        std::size_t total_width) noexcept;
bool CompareIntegerElements(void* destination, const void* left,
                            const void* right, std::size_t element_width,
                            std::size_t total_width,
                            bool signed_greater_than) noexcept;

} // namespace Executor::Jit::VectorSemantic
