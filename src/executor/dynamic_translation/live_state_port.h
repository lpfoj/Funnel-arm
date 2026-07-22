// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "executor/dynamic_translation/machine_state.h"

#include <cstdint>

namespace Lsx4::Translation {

class LiveStateView {
public:
    constexpr LiveStateView() noexcept = default;
    explicit constexpr LiveStateView(void* storage) noexcept : storage_{storage} {}

    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return storage_ != nullptr;
    }
    [[nodiscard]] std::uint64_t Read(IntegerRegister id) const noexcept;
    void Write(IntegerRegister id, std::uint64_t value) const noexcept;
    [[nodiscard]] std::uint64_t ResumeAddress() const noexcept;
    void SetResumeAddress(std::uint64_t address) const noexcept;
    [[nodiscard]] bool CopyTo(CpuFrame& destination) const noexcept;
    [[nodiscard]] bool ReplaceWith(const CpuFrame& source) const noexcept;

private:
    void* storage_{};
};

[[nodiscard]] LiveStateView PublishedLiveState() noexcept;
[[nodiscard]] void* ExchangeLiveStateStorage(void* storage) noexcept;
[[nodiscard]] const void* ExchangeContinuationStorage(const void* storage) noexcept;
[[nodiscard]] bool SnapshotContinuationState(CpuFrame& destination) noexcept;

} // namespace Lsx4::Translation
