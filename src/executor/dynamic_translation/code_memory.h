// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace Lsx4::Translation {

struct CodeReservation {
    std::span<std::uint8_t> writable{};
    std::span<const std::uint8_t> executable{};

    [[nodiscard]] bool IsValid() const noexcept {
        return !writable.empty() && writable.size() == executable.size();
    }
};

class ExecutableArena {
public:
    explicit ExecutableArena(std::size_t slab_bytes = 2 * 1024 * 1024) noexcept;
    ~ExecutableArena();

    ExecutableArena(const ExecutableArena&) = delete;
    ExecutableArena& operator=(const ExecutableArena&) = delete;

    [[nodiscard]] CodeReservation Reserve(std::size_t bytes,
                                          std::size_t alignment = 16) noexcept;
    void Publish(const CodeReservation& reservation) noexcept;
    [[nodiscard]] bool Owns(std::uintptr_t executable_address) const noexcept;
    [[nodiscard]] std::size_t CommittedBytes() const noexcept;

private:
    struct Slab;

    [[nodiscard]] std::unique_ptr<Slab> AllocateSlab(std::size_t minimum_bytes) noexcept;

    const std::size_t preferred_slab_bytes_;
    mutable std::mutex mutex_{};
    std::vector<std::unique_ptr<Slab>> slabs_{};
    std::size_t committed_bytes_{};
};

} // namespace Lsx4::Translation
