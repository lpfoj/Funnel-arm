// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/process_memory.h"

#include <cstring>
#include <limits>

#if defined(__linux__)
#include <sys/uio.h>
#include <unistd.h>
#endif

namespace Lsx4::Translation {
namespace {

bool AddressRangeExists(const std::uint64_t address,
                        const std::size_t byte_count) noexcept {
    return address >= 0x10000 && byte_count != 0 &&
           byte_count - 1 <= std::numeric_limits<std::uint64_t>::max() - address;
}

} // namespace

bool ReadProcessGuestMemory(const std::uint64_t address, void* const destination,
                            const std::size_t byte_count) noexcept {
    if (destination == nullptr || !AddressRangeExists(address, byte_count)) {
        return false;
    }
#if defined(__linux__)
    iovec local{destination, byte_count};
    iovec remote{reinterpret_cast<void*>(address), byte_count};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) ==
           static_cast<ssize_t>(byte_count);
#else
    std::memcpy(destination, reinterpret_cast<const void*>(address), byte_count);
    return true;
#endif
}

bool WriteProcessGuestMemory(const std::uint64_t address, const void* const source,
                             const std::size_t byte_count) noexcept {
    if (source == nullptr || !AddressRangeExists(address, byte_count)) {
        return false;
    }
#if defined(__linux__)
    iovec local{const_cast<void*>(source), byte_count};
    iovec remote{reinterpret_cast<void*>(address), byte_count};
    return process_vm_writev(getpid(), &local, 1, &remote, 1, 0) ==
           static_cast<ssize_t>(byte_count);
#else
    std::memcpy(reinterpret_cast<void*>(address), source, byte_count);
    return true;
#endif
}

bool ReadProcessGuestScalar(const std::uint64_t address,
                            std::uint64_t& value) noexcept {
    value = 0;
    return ReadProcessGuestMemory(address, &value, sizeof(value));
}

bool WriteProcessGuestScalar(const std::uint64_t address,
                             const std::uint64_t value) noexcept {
    return WriteProcessGuestMemory(address, &value, sizeof(value));
}

} // namespace Lsx4::Translation
