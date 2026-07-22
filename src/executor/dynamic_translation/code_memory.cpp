// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/code_memory.h"

#include <algorithm>
#include <atomic>
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace Lsx4::Translation {
namespace {

std::size_t RoundUp(const std::size_t value, const std::size_t alignment) noexcept {
    if (alignment == 0 || value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

bool IsPowerOfTwo(const std::size_t value) noexcept {
    return value != 0 && (value & (value - 1)) == 0;
}

int CreateAnonymousFile() noexcept {
#if defined(SYS_memfd_create)
    return static_cast<int>(::syscall(SYS_memfd_create, "lsx4-dynamic-code", 0x0001u));
#else
    return -1;
#endif
}

} // namespace

struct ExecutableArena::Slab {
    int descriptor{-1};
    std::uint8_t* write_view{};
    std::uint8_t* execute_view{};
    std::size_t capacity{};
    std::size_t cursor{};

    ~Slab() {
        if (write_view != nullptr) {
            ::munmap(write_view, capacity);
        }
        if (execute_view != nullptr) {
            ::munmap(execute_view, capacity);
        }
        if (descriptor >= 0) {
            ::close(descriptor);
        }
    }
};

ExecutableArena::ExecutableArena(const std::size_t slab_bytes) noexcept
    : preferred_slab_bytes_{std::max<std::size_t>(slab_bytes, 64 * 1024)} {}

ExecutableArena::~ExecutableArena() = default;

std::unique_ptr<ExecutableArena::Slab> ExecutableArena::AllocateSlab(
    const std::size_t minimum_bytes) noexcept {
    const long page_value = ::sysconf(_SC_PAGESIZE);
    const std::size_t page_bytes = page_value > 0 ? static_cast<std::size_t>(page_value) : 4096;
    const std::size_t requested = std::max(preferred_slab_bytes_, minimum_bytes);
    const std::size_t capacity = RoundUp(requested, page_bytes);
    if (capacity == 0) {
        return {};
    }

    const int descriptor = CreateAnonymousFile();
    if (descriptor < 0 || ::ftruncate(descriptor, static_cast<off_t>(capacity)) != 0) {
        if (descriptor >= 0) {
            ::close(descriptor);
        }
        return {};
    }
    void* const write_view = ::mmap(nullptr, capacity, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, descriptor, 0);
    if (write_view == MAP_FAILED) {
        ::close(descriptor);
        return {};
    }
    void* const execute_view = ::mmap(nullptr, capacity, PROT_READ | PROT_EXEC,
                                      MAP_SHARED, descriptor, 0);
    if (execute_view == MAP_FAILED) {
        ::munmap(write_view, capacity);
        ::close(descriptor);
        return {};
    }

    auto slab = std::make_unique<Slab>();
    slab->descriptor = descriptor;
    slab->write_view = static_cast<std::uint8_t*>(write_view);
    slab->execute_view = static_cast<std::uint8_t*>(execute_view);
    slab->capacity = capacity;
    return slab;
}

CodeReservation ExecutableArena::Reserve(const std::size_t bytes,
                                          const std::size_t alignment) noexcept {
    if (bytes == 0 || !IsPowerOfTwo(alignment)) {
        return {};
    }
    std::lock_guard lock{mutex_};
    Slab* selected = nullptr;
    std::size_t offset = 0;
    for (const auto& slab : slabs_) {
        offset = RoundUp(slab->cursor, alignment);
        if (offset != 0 && offset <= slab->capacity && bytes <= slab->capacity - offset) {
            selected = slab.get();
            break;
        }
    }
    if (selected == nullptr) {
        if (bytes > std::numeric_limits<std::size_t>::max() - alignment) {
            return {};
        }
        std::unique_ptr<Slab> slab = AllocateSlab(bytes + alignment);
        if (!slab) {
            return {};
        }
        selected = slab.get();
        slabs_.push_back(std::move(slab));
        offset = RoundUp(selected->cursor, alignment);
    }
    selected->cursor = offset + bytes;
    committed_bytes_ += bytes;
    return {
        std::span<std::uint8_t>{selected->write_view + offset, bytes},
        std::span<const std::uint8_t>{selected->execute_view + offset, bytes},
    };
}

void ExecutableArena::Publish(const CodeReservation& reservation) noexcept {
    if (!reservation.IsValid()) {
        return;
    }
    auto* const begin = const_cast<char*>(reinterpret_cast<const char*>(
        reservation.executable.data()));
    __builtin___clear_cache(begin, begin + reservation.executable.size());
    std::atomic_thread_fence(std::memory_order_release);
}

bool ExecutableArena::Owns(const std::uintptr_t executable_address) const noexcept {
    std::lock_guard lock{mutex_};
    for (const auto& slab : slabs_) {
        const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(slab->execute_view);
        if (executable_address >= begin && executable_address - begin < slab->cursor) {
            return true;
        }
    }
    return false;
}

std::size_t ExecutableArena::CommittedBytes() const noexcept {
    std::lock_guard lock{mutex_};
    return committed_bytes_;
}

} // namespace Lsx4::Translation
