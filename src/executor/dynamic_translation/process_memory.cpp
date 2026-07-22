// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/process_memory.h"

#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

#if defined(__linux__)
#include <link.h>
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

#if defined(__linux__)
std::pair<std::uintptr_t, std::uintptr_t> OwningImageRange() noexcept {
    static std::once_flag once;
    static std::pair<std::uintptr_t, std::uintptr_t> range{};
    std::call_once(once, [] {
        struct SearchContext {
            std::uintptr_t needle{};
            std::pair<std::uintptr_t, std::uintptr_t> found{};
        } context{reinterpret_cast<std::uintptr_t>(&WriteProcessGuestMemory), {}};
        dl_iterate_phdr(
            [](dl_phdr_info* info, std::size_t, void* opaque) {
                auto& context = *static_cast<SearchContext*>(opaque);
                std::uintptr_t first = std::numeric_limits<std::uintptr_t>::max();
                std::uintptr_t last = 0;
                bool owns = false;
                for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
                    const ElfW(Phdr)& header = info->dlpi_phdr[index];
                    if (header.p_type != PT_LOAD) {
                        continue;
                    }
                    const std::uintptr_t begin = info->dlpi_addr + header.p_vaddr;
                    const std::uintptr_t end = begin + header.p_memsz;
                    owns = owns ||
                           (context.needle >= begin && context.needle < end);
                    first = begin < first ? begin : first;
                    last = end > last ? end : last;
                }
                if (owns && last > first) {
                    context.found = {first, last};
                    return 1;
                }
                return 0;
            },
            &context);
        range = context.found;
    });
    return range;
}
#endif

bool OverlapsOwningImage(const std::uint64_t address,
                         const std::size_t byte_count) noexcept {
#if defined(__linux__)
    const auto [first, last] = OwningImageRange();
    return last > first && address < last && address + byte_count > first;
#else
    (void)address;
    (void)byte_count;
    return false;
#endif
}

} // namespace

bool IsProcessImageMemory(const std::uint64_t address,
                          const std::size_t byte_count) noexcept {
    return AddressRangeExists(address, byte_count) &&
           OverlapsOwningImage(address, byte_count);
}

bool QueryProcessImageBounds(std::uint64_t& first,
                             std::uint64_t& last) noexcept {
#if defined(__linux__)
    const auto bounds = OwningImageRange();
    first = bounds.first;
    last = bounds.second;
    return last > first;
#else
    first = 0;
    last = 0;
    return false;
#endif
}

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
    if (IsProcessImageMemory(address, byte_count)) {
        return false;
    }
#if defined(__linux__)
    // Android permits the same-process readv path used by stable instruction
    // fetches on devices where process_vm_writev is filtered by seccomp.  Make
    // the guest destination the local vector and copy from the validated remote
    // source; the kernel still checks both mappings without a racy memcpy.
    iovec local{reinterpret_cast<void*>(address), byte_count};
    iovec remote{const_cast<void*>(source), byte_count};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) ==
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
