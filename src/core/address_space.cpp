// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <map>
#include <cstdio>
#include <cstring>
#include <limits>
#include "common/alignment.h"
#include "common/arch.h"
#include "common/assert.h"
#include "common/config.h"
#include "common/elf_info.h"
#include "common/error.h"
#include "core/address_space.h"
#include "core/libraries/kernel/memory.h"
#include "core/memory.h"
#include "libraries/error_codes.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#if defined(__ANDROID__)
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif
#if defined(__ANDROID__)
#include <android/log.h>
#endif

#if defined(__APPLE__) && defined(ARCH_X86_64)
asm(".zerofill SYSTEM_MANAGED,SYSTEM_MANAGED,__SYSTEM_MANAGED,0x7FFBFC000");
asm(".zerofill SYSTEM_RESERVED,SYSTEM_RESERVED,__SYSTEM_RESERVED,0x7C0004000");
asm(".zerofill USER_AREA,USER_AREA,__USER_AREA,0x5F9000000000");
#endif

namespace Core {

namespace {

#if defined(__ANDROID__)
[[nodiscard]] int CreateMemoryFile(const char* name, unsigned int flags) noexcept {
    return static_cast<int>(syscall(__NR_memfd_create, name, flags));
}
#else
[[nodiscard]] int CreateMemoryFile(const char* name, unsigned int flags) noexcept {
    return memfd_create(name, flags);
}
#endif

[[nodiscard]] constexpr bool RangeIsContained(VAddr address, u64 size, VAddr owner_base,
                                               u64 owner_size) noexcept {
    constexpr VAddr MaxAddress = std::numeric_limits<VAddr>::max();
    if (size == 0 || owner_size == 0 || address > MaxAddress - size ||
        owner_base > MaxAddress - owner_size || address < owner_base) {
        return false;
    }
    const u64 offset = address - owner_base;
    return offset < owner_size && size <= owner_size - offset;
}

static_assert(RangeIsContained(0x1000, 0x1000, 0x1000, 0x2000));
static_assert(RangeIsContained(0x2fff, 1, 0x1000, 0x2000));
static_assert(!RangeIsContained(0x3000, 1, 0x1000, 0x2000));
static_assert(!RangeIsContained(0x2fff, 2, 0x1000, 0x2000));
static_assert(!RangeIsContained(std::numeric_limits<VAddr>::max() - 7, 8, 0x1000,
                                std::numeric_limits<VAddr>::max() - 0x1000));

}

constexpr VAddr SYSTEM_MANAGED_MIN = 0x400000ULL;
constexpr VAddr SYSTEM_MANAGED_MAX = 0x7FFFFBFFFULL;
constexpr VAddr SYSTEM_RESERVED_MIN = 0x7FFFFC000ULL;
#if defined(__APPLE__) && defined(ARCH_X86_64)
constexpr VAddr SYSTEM_RESERVED_MAX = 0xFBFFFFFFFULL;
constexpr VAddr USER_MIN = 0x7000000000ULL;
#else
constexpr VAddr SYSTEM_RESERVED_MAX = 0xFFFFFFFFFULL;
constexpr VAddr USER_MIN = 0x1000000000ULL;
#endif
#if defined(__linux__)
constexpr VAddr USER_MAX = 0x54FFFFFFFFFFULL;
#else
constexpr VAddr USER_MAX = 0x5FFFFFFFFFFFULL;
#endif

static constexpr u64 SystemManagedSize = SYSTEM_MANAGED_MAX - SYSTEM_MANAGED_MIN + 1;
static constexpr u64 SystemReservedSize = SYSTEM_RESERVED_MAX - SYSTEM_RESERVED_MIN + 1;
static constexpr u64 UserSize = USER_MAX - USER_MIN + 1;

static u64 BackingSize = ORBIS_KERNEL_TOTAL_MEM_DEV_PRO;

#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
static constexpr VAddr AndroidProbeSystemManagedMin = DEFAULT_MAPPING_BASE;
static constexpr u64 AndroidProbeSystemManagedSize =
    SYSTEM_RESERVED_MIN - AndroidProbeSystemManagedMin;
static constexpr u64 AndroidProbeSystemReservedSize = SystemReservedSize;
static constexpr u64 AndroidProbeUserSize = 6_GB;
static constexpr VAddr AndroidProbeHighUserMin = 0xC000000000ULL;
static constexpr u64 AndroidProbeHighUserSize = 16_GB;
#endif

#ifdef _WIN32

[[nodiscard]] constexpr u64 ToWindowsProt(Core::MemoryProt prot) {
    const bool read =
        True(prot & Core::MemoryProt::CpuRead) || True(prot & Core::MemoryProt::GpuRead);
    const bool write =
        True(prot & Core::MemoryProt::CpuWrite) || True(prot & Core::MemoryProt::GpuWrite);
    const bool execute = True(prot & Core::MemoryProt::CpuExec);

    if (write && !read) {
        LOG_WARNING(Core, "Converting write-only mapping to read-write");
    }

    if (execute) {
        if (write) {
            return PAGE_EXECUTE_READWRITE;
        } else if (read && !write) {
            return PAGE_EXECUTE_READ;
        } else {
            return PAGE_EXECUTE;
        }
    } else {
        if (write) {
            return PAGE_READWRITE;
        } else if (read && !write) {
            return PAGE_READONLY;
        } else {
            return PAGE_NOACCESS;
        }
    }
}

struct MemoryRegion {
    VAddr base;
    PAddr phys_base;
    u64 size;
    u32 prot;
    s32 fd;
    bool is_mapped;
};

struct AddressSpace::Impl {
    Impl() : process{GetCurrentProcess()} {
        SYSTEM_INFO sys_info{};
        GetSystemInfo(&sys_info);
        u64 alignment = sys_info.dwAllocationGranularity;

        auto ntdll_handle = GetModuleHandleW(L"ntdll.dll");
        ASSERT_MSG(ntdll_handle, "Failed to retrieve ntdll handle");

        s64(WINAPI * RtlGetVersion)(LPOSVERSIONINFOW);
        *(FARPROC*)&RtlGetVersion = GetProcAddress(ntdll_handle, "RtlGetVersion");
        ASSERT_MSG(RtlGetVersion, "failed to retrieve function pointer for RtlGetVersion");

        RTL_OSVERSIONINFOW os_version_info{};
        RtlGetVersion(&os_version_info);

        u64 supported_user_max = USER_MAX;
        static constexpr s32 AffectedBuildNumber = 22621;

        s32 sdk_ver = Common::ElfInfo::Instance().CompiledSdkVer();
        if (os_version_info.dwBuildNumber <= AffectedBuildNumber ||
            sdk_ver >= Common::ElfInfo::FW_30) {
            supported_user_max = 0x10000000000ULL;
            if (sdk_ver < Common::ElfInfo::FW_30) {
                LOG_WARNING(
                    Core,
                    "Older Windows version detected, reducing user max to {:#x} to avoid problems",
                    supported_user_max);
            }
        }

        VAddr next_addr = SYSTEM_MANAGED_MIN;
        MEMORY_BASIC_INFORMATION info{};
        while (next_addr <= supported_user_max) {
            ASSERT_MSG(VirtualQuery(reinterpret_cast<PVOID>(next_addr), &info, sizeof(info)),
                       "Failed to query memory information for address {:#x}", next_addr);

            next_addr = reinterpret_cast<VAddr>(info.BaseAddress) + info.RegionSize;
            next_addr = Common::AlignUp(next_addr, alignment);

            u64 size = info.RegionSize;
            if (next_addr > supported_user_max) {
                size -= (next_addr - supported_user_max);
            }
            size = Common::AlignDown(size, alignment);

            if (info.State == MEM_FREE && info.RegionSize > 0x1000000) {
                VAddr addr = Common::AlignUp(reinterpret_cast<VAddr>(info.BaseAddress), alignment);
                regions.emplace(addr,
                                MemoryRegion{addr, PAddr(-1), size, PAGE_NOACCESS, -1, false});
            }
        }

        for (auto region : regions) {
            auto addr = static_cast<u8*>(VirtualAlloc2(
                process, reinterpret_cast<PVOID>(region.second.base), region.second.size,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, NULL, 0));
            ASSERT_MSG(addr, "Unable to reserve virtual address space: {}",
                       Common::GetLastErrorMsg());
        }

        system_managed_base = reinterpret_cast<u8*>(regions.begin()->first);
        system_managed_size = SystemManagedSize - (regions.begin()->first - SYSTEM_MANAGED_MIN);
        system_reserved_base = reinterpret_cast<u8*>(SYSTEM_RESERVED_MIN);
        system_reserved_size = SystemReservedSize;
        user_base = reinterpret_cast<u8*>(USER_MIN);
        user_size = supported_user_max - USER_MIN - 1;

        BackingSize += Config::getExtraDmemInMbytes() * 1_MB;

        backing_handle = CreateFileMapping2(INVALID_HANDLE_VALUE, nullptr, FILE_MAP_ALL_ACCESS,
                                            PAGE_EXECUTE_READWRITE, SEC_COMMIT, BackingSize,
                                            nullptr, nullptr, 0);

        ASSERT_MSG(backing_handle, "{}", Common::GetLastErrorMsg());
        backing_base = static_cast<u8*>(VirtualAlloc2(process, nullptr, BackingSize,
                                                      MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                                      PAGE_NOACCESS, nullptr, 0));
        ASSERT_MSG(backing_base, "{}", Common::GetLastErrorMsg());

        void* const ret =
            MapViewOfFile3(backing_handle, process, backing_base, 0, BackingSize,
                           MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0);
        ASSERT_MSG(ret == backing_base, "{}", Common::GetLastErrorMsg());
    }

    ~Impl() {
        if (virtual_base) {
            if (!VirtualFree(virtual_base, 0, MEM_RELEASE)) {
                LOG_CRITICAL(Core, "Failed to free virtual memory");
            }
        }
        if (backing_base) {
            if (!UnmapViewOfFile2(process, backing_base, MEM_PRESERVE_PLACEHOLDER)) {
                LOG_CRITICAL(Core, "Failed to unmap backing memory placeholder");
            }
            if (!VirtualFreeEx(process, backing_base, 0, MEM_RELEASE)) {
                LOG_CRITICAL(Core, "Failed to free backing memory");
            }
        }
        if (!CloseHandle(backing_handle)) {
            LOG_CRITICAL(Core, "Failed to free backing memory file handle");
        }
    }

    void* MapRegion(MemoryRegion* region) {
        VAddr virtual_addr = region->base;
        PAddr phys_addr = region->phys_base;
        u64 size = region->size;
        ULONG prot = region->prot;
        s32 fd = region->fd;

        void* ptr = nullptr;
        if (phys_addr != -1) {
            HANDLE backing = fd != -1 ? reinterpret_cast<HANDLE>(fd) : backing_handle;
            if (fd != -1 && prot == PAGE_READONLY) {
                DWORD resultvar;
                ptr = VirtualAlloc2(process, reinterpret_cast<PVOID>(virtual_addr), size,
                                    MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                                    PAGE_READWRITE, nullptr, 0);

                OVERLAPPED param{};
                param.Offset = phys_addr & 0xffffffffull;
                param.OffsetHigh = (phys_addr & 0xffffffff00000000ull) >> 32;
                bool ret = ReadFile(backing, ptr, size, &resultvar, &param);
                ASSERT_MSG(ret, "ReadFile failed. {}", Common::GetLastErrorMsg());
                ret = VirtualProtect(ptr, size, prot, &resultvar);
                ASSERT_MSG(ret, "VirtualProtect failed. {}", Common::GetLastErrorMsg());
            } else {
                ptr = MapViewOfFile3(backing, process, reinterpret_cast<PVOID>(virtual_addr),
                                     phys_addr, size, MEM_REPLACE_PLACEHOLDER, prot, nullptr, 0);
                ASSERT_MSG(ptr, "MapViewOfFile3 failed. {}", Common::GetLastErrorMsg());
            }
        } else {
            ptr =
                VirtualAlloc2(process, reinterpret_cast<PVOID>(virtual_addr), size,
                              MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER, prot, nullptr, 0);
        }
        ASSERT_MSG(ptr, "{}", Common::GetLastErrorMsg());
        return ptr;
    }

    void UnmapRegion(MemoryRegion* region) {
        VAddr virtual_addr = region->base;
        PAddr phys_base = region->phys_base;
        u64 size = region->size;
        ULONG prot = region->prot;
        s32 fd = region->fd;

        bool ret = false;
        if ((fd != -1 && prot != PAGE_READONLY) || (fd == -1 && phys_base != -1)) {
            ret = UnmapViewOfFile2(process, reinterpret_cast<PVOID>(virtual_addr),
                                   MEM_PRESERVE_PLACEHOLDER);
        } else {
            ret = VirtualFreeEx(process, reinterpret_cast<PVOID>(virtual_addr), size,
                                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
        }
        ASSERT_MSG(ret, "Unmap on virtual_addr {:#x}, size {:#x} failed: {}", virtual_addr, size,
                   Common::GetLastErrorMsg());
    }

    void SplitRegion(VAddr virtual_addr, u64 size) {
        auto it = std::prev(regions.upper_bound(virtual_addr));

        ASSERT_MSG(it->second.base + it->second.size >= virtual_addr + size,
                   "Cannot fit region into one placeholder");

        if (it->second.is_mapped) {
            ASSERT_MSG(it->second.phys_base != -1 || !it->second.is_mapped,
                       "Cannot split unbacked mapping");
            UnmapRegion(&it->second);
        }

        if (it->second.base != virtual_addr) {
            auto& region = it->second;
            u64 base_offset = virtual_addr - region.base;
            u64 next_region_size = region.size - base_offset;
            PAddr next_region_phys_base = -1;
            if (region.is_mapped) {
                next_region_phys_base = region.phys_base + base_offset;
            }
            region.size = base_offset;

            if (!VirtualFreeEx(process, LPVOID(region.base), region.size,
                               MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
                UNREACHABLE_MSG("Region splitting failed: {}", Common::GetLastErrorMsg());
            }

            if (region.is_mapped) {
                MapRegion(&region);
            }

            it = regions.emplace_hint(std::next(it), virtual_addr,
                                      MemoryRegion(virtual_addr, next_region_phys_base,
                                                   next_region_size, region.prot, region.fd,
                                                   region.is_mapped));
        }

        if (it->second.size != size) {
            auto& region = it->second;
            VAddr next_region_addr = region.base + size;
            u64 next_region_size = region.size - size;
            PAddr next_region_phys_base = -1;
            if (region.is_mapped) {
                next_region_phys_base = region.phys_base + size;
            }
            region.size = size;

            regions.emplace_hint(std::next(it), next_region_addr,
                                 MemoryRegion(next_region_addr, next_region_phys_base,
                                              next_region_size, region.prot, region.fd,
                                              region.is_mapped));

            if (!VirtualFreeEx(process, LPVOID(region.base), region.size,
                               MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
                UNREACHABLE_MSG("Region splitting failed: {}", Common::GetLastErrorMsg());
            }

            if (region.is_mapped) {
                MapRegion(&std::next(it)->second);
            }
        }

        if (it->second.is_mapped) {
            MapRegion(&it->second);
        }
    }

    void* Map(VAddr virtual_addr, PAddr phys_addr, u64 size, ULONG prot, s32 fd = -1) {
        std::scoped_lock lk{mutex};
        auto it = std::prev(regions.upper_bound(virtual_addr));

        if (it->first != virtual_addr || it->second.size != size) {
            SplitRegion(virtual_addr, size);
            it = std::prev(regions.upper_bound(virtual_addr));
        }

        auto& [base, region] = *it;
        ASSERT_MSG(!region.is_mapped, "Cannot overwrite mapped region");

        region.is_mapped = true;
        region.phys_base = phys_addr;
        region.prot = prot;
        region.fd = fd;
        return MapRegion(&region);
    }

    [[nodiscard]] bool ContainsOwnedRange(VAddr virtual_addr, u64 size) const noexcept {
        constexpr VAddr MaxAddress = std::numeric_limits<VAddr>::max();
        if (size == 0 || virtual_addr > MaxAddress - size || regions.empty()) {
            return false;
        }
        auto it = regions.upper_bound(virtual_addr);
        if (it == regions.begin()) {
            return false;
        }
        --it;

        VAddr cursor = virtual_addr;
        u64 remaining = size;
        while (remaining != 0) {
            const auto& region = it->second;
            if (!RangeIsContained(cursor, 1, region.base, region.size)) {
                return false;
            }
            const u64 consumed =
                std::min<u64>(region.size - (cursor - region.base), remaining);
            cursor += consumed;
            remaining -= consumed;
            if (remaining == 0) {
                return true;
            }
            ++it;
            if (it == regions.end() || it->second.base != cursor) {
                return false;
            }
        }
        return true;
    }

    void CoalesceFreeRegions(VAddr virtual_addr) {
        auto it = std::prev(regions.upper_bound(virtual_addr));
        ASSERT_MSG(!it->second.is_mapped, "Cannot coalesce mapped regions");

        bool can_coalesce = false;
        auto it_prev = it != regions.begin() ? std::prev(it) : regions.end();
        while (it_prev != regions.end() && !it_prev->second.is_mapped &&
               it_prev->first + it_prev->second.size == it->first) {
            it_prev->second.size = it_prev->second.size + it->second.size;
            regions.erase(it);
            it = it_prev;

            can_coalesce = true;

            it_prev = it != regions.begin() ? std::prev(it) : regions.end();
        }

        auto it_next = std::next(it);
        while (it_next != regions.end() && !it_next->second.is_mapped &&
               it->first + it->second.size == it_next->first) {
            it->second.size = it->second.size + it_next->second.size;
            regions.erase(it_next);

            can_coalesce = true;

            it_next = std::next(it);
        }

        if (can_coalesce) {
            if (!VirtualFreeEx(process, LPVOID(it->first), it->second.size,
                               MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
                UNREACHABLE_MSG("Region coalescing failed: {}", Common::GetLastErrorMsg());
            }
        }
    }

    void Unmap(VAddr virtual_addr, u64 size) {
        std::scoped_lock lk{mutex};
        u64 remaining_size = size;
        VAddr current_addr = virtual_addr;
        while (remaining_size > 0) {
            auto it = std::prev(regions.upper_bound(current_addr));

            u64 base_offset = current_addr - it->second.base;
            u64 size_to_unmap = std::min<u64>(it->second.size - base_offset, remaining_size);
            if (current_addr != it->second.base || size_to_unmap != it->second.size) {
                SplitRegion(current_addr, size_to_unmap);
                it = std::prev(regions.upper_bound(current_addr));
            }

            auto& [base, region] = *it;

            if (region.is_mapped) {
                UnmapRegion(&region);
            }

            region.is_mapped = false;
            region.fd = -1;
            region.phys_base = -1;
            region.prot = PAGE_NOACCESS;

            remaining_size -= size_to_unmap;
            current_addr += size_to_unmap;
        }

        CoalesceFreeRegions(virtual_addr);
    }

    void Protect(VAddr virtual_addr, u64 size, bool read, bool write, bool execute) {
        std::scoped_lock lk{mutex};
        DWORD new_flags{};

        if (write && !read) {
            LOG_WARNING(Core, "Converting write-only protection to read-write");
        }

        if (execute) {
            if (write) {
                new_flags = PAGE_EXECUTE_READWRITE;
            } else if (read && !write) {
                new_flags = PAGE_EXECUTE_READ;
            } else {
                new_flags = PAGE_EXECUTE;
            }
        } else {
            if (write) {
                new_flags = PAGE_READWRITE;
            } else if (read && !write) {
                new_flags = PAGE_READONLY;
            } else {
                new_flags = PAGE_NOACCESS;
            }
        }

        if (new_flags == 0) {
            LOG_CRITICAL(Core,
                         "Unsupported protection flag combination for address {:#x}, size {}, "
                         "read={}, write={}, execute={}",
                         virtual_addr, size, read, write, execute);
            return;
        }

        const VAddr virtual_end = virtual_addr + size;
        auto it = --regions.upper_bound(virtual_addr);
        ASSERT_MSG(it != regions.end(), "addr {:#x} out of bounds", virtual_addr);
        for (; it->first < virtual_end; it++) {
            if (!it->second.is_mapped) {
                continue;
            }
            const auto& region = it->second;
            const u64 range_addr = std::max(region.base, virtual_addr);
            const u64 range_size = std::min(region.base + region.size, virtual_end) - range_addr;
            DWORD old_flags{};
            if (!VirtualProtectEx(process, LPVOID(range_addr), range_size, new_flags, &old_flags)) {
                UNREACHABLE_MSG(
                    "Failed to change virtual memory protection for address {:#x}, size "
                    "{:#x}, error {}",
                    virtual_addr, size, Common::GetLastErrorMsg());
            }
        }
    }

    boost::icl::interval_set<VAddr> GetUsableRegions() {
        boost::icl::interval_set<VAddr> reserved_regions;
        for (auto region : regions) {
            reserved_regions.insert({region.second.base, region.second.base + region.second.size});
        }
        return reserved_regions;
    }

    std::mutex mutex;
    HANDLE process{};
    HANDLE backing_handle{};
    u8* backing_base{};
    u8* virtual_base{};
    u8* system_managed_base{};
    u64 system_managed_size{};
    u8* system_reserved_base{};
    u64 system_reserved_size{};
    u8* user_base{};
    u64 user_size{};
    std::map<VAddr, MemoryRegion> regions;
};
#else

enum PosixPageProtection {
    PAGE_NOACCESS = 0,
    PAGE_READONLY = PROT_READ,
    PAGE_READWRITE = PROT_READ | PROT_WRITE,
    PAGE_EXECUTE = PROT_EXEC,
    PAGE_EXECUTE_READ = PROT_EXEC | PROT_READ,
    PAGE_EXECUTE_READWRITE = PROT_EXEC | PROT_READ | PROT_WRITE
};

[[nodiscard]] constexpr PosixPageProtection ToPosixProt(Core::MemoryProt prot) {
    const bool read =
        True(prot & Core::MemoryProt::CpuRead) || True(prot & Core::MemoryProt::GpuRead);
    const bool write =
        True(prot & Core::MemoryProt::CpuWrite) || True(prot & Core::MemoryProt::GpuWrite);
    const bool execute = True(prot & Core::MemoryProt::CpuExec);

    if (write && !read) {
        LOG_WARNING(Core, "Converting write-only mapping to read-write");
    }

    if (execute) {
        if (write) {
            return PAGE_EXECUTE_READWRITE;
        } else if (read && !write) {
            return PAGE_EXECUTE_READ;
        } else {
            return PAGE_EXECUTE;
        }
    } else {
        if (write) {
            return PAGE_READWRITE;
        } else if (read && !write) {
            return PAGE_READONLY;
        } else {
            return PAGE_NOACCESS;
        }
    }
}

#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
#ifndef MAP_FIXED_NOREPLACE
static constexpr int MAP_FIXED_NOREPLACE = 0x100000;
#endif

static u8* ReserveAndroidCanonicalRegion(VAddr base, u64 size) {
    void* ret = mmap(reinterpret_cast<void*>(base), size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (ret == MAP_FAILED) {
        return nullptr;
    }
    if (reinterpret_cast<VAddr>(ret) != base) {
        munmap(ret, size);
        return nullptr;
    }
    return static_cast<u8*>(ret);
}
#endif

struct AddressSpace::Impl {
    Impl() {
        BackingSize += Config::getExtraDmemInMbytes() * 1_MB;
#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
        system_managed_size = AndroidProbeSystemManagedSize;
        system_reserved_size = AndroidProbeSystemReservedSize;
        user_size = AndroidProbeUserSize;
#else
        system_managed_size = SystemManagedSize;
        system_reserved_size = SystemReservedSize;
        user_size = UserSize;
#endif

        constexpr int protection_flags = PROT_READ | PROT_WRITE;
        constexpr int map_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED;
#if defined(__APPLE__) && defined(ARCH_X86_64)
        system_managed_base =
            reinterpret_cast<u8*>(mmap(reinterpret_cast<void*>(SYSTEM_MANAGED_MIN),
                                       system_managed_size, protection_flags, map_flags, -1, 0));
        system_reserved_base =
            reinterpret_cast<u8*>(mmap(reinterpret_cast<void*>(SYSTEM_RESERVED_MIN),
                                       system_reserved_size, protection_flags, map_flags, -1, 0));
        user_base = reinterpret_cast<u8*>(
            mmap(reinterpret_cast<void*>(USER_MIN), user_size, protection_flags, map_flags, -1, 0));
#else
        const auto virtual_size = system_managed_size + system_reserved_size + user_size;
#if defined(ARCH_X86_64)
        const auto virtual_base =
            reinterpret_cast<u8*>(mmap(reinterpret_cast<void*>(SYSTEM_MANAGED_MIN), virtual_size,
                                       protection_flags, map_flags, -1, 0));
        system_managed_base = virtual_base;
        system_reserved_base = reinterpret_cast<u8*>(SYSTEM_RESERVED_MIN);
        user_base = reinterpret_cast<u8*>(USER_MIN);
#else
#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
        u8* virtual_base = nullptr;
        system_managed_base =
            ReserveAndroidCanonicalRegion(AndroidProbeSystemManagedMin, system_managed_size);
        system_reserved_base =
            ReserveAndroidCanonicalRegion(SYSTEM_RESERVED_MIN, system_reserved_size);
        user_base = ReserveAndroidCanonicalRegion(USER_MIN, user_size);
        android_high_user_size = AndroidProbeHighUserSize;
        android_high_user_base =
            ReserveAndroidCanonicalRegion(AndroidProbeHighUserMin, android_high_user_size);
        if (!system_managed_base || !system_reserved_base || !user_base) {
            const int saved_errno = errno;
            if (system_managed_base) {
                munmap(system_managed_base, system_managed_size);
            }
            if (system_reserved_base) {
                munmap(system_reserved_base, system_reserved_size);
            }
            if (user_base) {
                munmap(user_base, user_size);
            }
            if (android_high_user_base) {
                munmap(android_high_user_base, android_high_user_size);
                android_high_user_base = nullptr;
                android_high_user_size = 0;
            }
            LOG_ERROR(Kernel_Vmm,
                      "Android canonical PS4 VA reservation failed (errno={}, {}). Falling back "
                      "to shifted probe VA; live commercial guests may reject direct memory VAs.",
                      saved_errno, strerror(saved_errno));
            virtual_base =
                reinterpret_cast<u8*>(mmap(nullptr, virtual_size, protection_flags,
                                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0));
            system_managed_base = virtual_base;
            system_reserved_base = virtual_base + system_managed_size;
            user_base = system_reserved_base + system_reserved_size;
        } else {
            virtual_base = system_managed_base;
            LOG_INFO(Kernel_Vmm,
                     "Android canonical PS4 VA reservation enabled for live direct memory.");
            if (android_high_user_base) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_ANDROID_VA] high_user reserved base=0x%llx "
                                    "end=0x%llx size=0x%llx",
                                    reinterpret_cast<unsigned long long>(android_high_user_base),
                                    reinterpret_cast<unsigned long long>(android_high_user_base +
                                                                        android_high_user_size),
                                    static_cast<unsigned long long>(android_high_user_size));
            } else {
                android_high_user_size = 0;
                __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                    "[EXECUTOR_ANDROID_VA] high_user reservation failed base=0x%llx "
                                    "size=0x%llx errno=%d",
                                    static_cast<unsigned long long>(AndroidProbeHighUserMin),
                                    static_cast<unsigned long long>(AndroidProbeHighUserSize), errno);
            }
        }
#else
        const auto virtual_base =
            reinterpret_cast<u8*>(mmap(nullptr, virtual_size, protection_flags, map_flags, -1, 0));
        system_managed_base = virtual_base;
        system_reserved_base = virtual_base + SYSTEM_RESERVED_MIN - SYSTEM_MANAGED_MIN;
        user_base = virtual_base + USER_MIN - SYSTEM_MANAGED_MIN;
#endif
#endif
#endif
        if (system_managed_base == MAP_FAILED || system_reserved_base == MAP_FAILED ||
            user_base == MAP_FAILED) {
            LOG_CRITICAL(Kernel_Vmm, "mmap failed: {}", strerror(errno));
            throw std::bad_alloc{};
        }

        LOG_INFO(Kernel_Vmm, "System managed virtual memory region: {} - {}",
                 fmt::ptr(system_managed_base),
                 fmt::ptr(system_managed_base + system_managed_size - 1));
        LOG_INFO(Kernel_Vmm, "System reserved virtual memory region: {} - {}",
                 fmt::ptr(system_reserved_base),
                 fmt::ptr(system_reserved_base + system_reserved_size - 1));
        LOG_INFO(Kernel_Vmm, "User virtual memory region: {} - {}", fmt::ptr(user_base),
                 fmt::ptr(user_base + user_size - 1));
#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_ANDROID_VA] system_managed=0x%llx..0x%llx "
                            "system_reserved=0x%llx..0x%llx user=0x%llx..0x%llx",
                            reinterpret_cast<unsigned long long>(system_managed_base),
                            reinterpret_cast<unsigned long long>(system_managed_base +
                                                                system_managed_size),
                            reinterpret_cast<unsigned long long>(system_reserved_base),
                            reinterpret_cast<unsigned long long>(system_reserved_base +
                                                                system_reserved_size),
                            reinterpret_cast<unsigned long long>(user_base),
                            reinterpret_cast<unsigned long long>(user_base + user_size));
#endif

        const VAddr system_managed_addr = reinterpret_cast<VAddr>(system_managed_base);
        const VAddr system_reserved_addr = reinterpret_cast<VAddr>(system_reserved_base);
        const VAddr user_addr = reinterpret_cast<VAddr>(user_base);
        m_free_regions.insert({system_managed_addr, system_managed_addr + system_managed_size});
        m_free_regions.insert({system_reserved_addr, system_reserved_addr + system_reserved_size});
        m_free_regions.insert({user_addr, user_addr + user_size});
#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
        if (android_high_user_base && android_high_user_size != 0) {
            const VAddr high_user_addr = reinterpret_cast<VAddr>(android_high_user_base);
            m_free_regions.insert({high_user_addr, high_user_addr + android_high_user_size});
        }
#endif
        m_owned_regions = m_free_regions;

#ifdef __APPLE__
        const auto shm_path = fmt::format("/BackingDmem{}", getpid());
        backing_fd = shm_open(shm_path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (backing_fd < 0) {
            LOG_CRITICAL(Kernel_Vmm, "shm_open failed: {}", strerror(errno));
            throw std::bad_alloc{};
        }
        shm_unlink(shm_path.c_str());
#else
        madvise(virtual_base, virtual_size, MADV_HUGEPAGE);

#ifndef MFD_EXEC
#define MFD_EXEC 0x0010U
#endif
        backing_fd = CreateMemoryFile("BackingDmem", MFD_EXEC);
        if (backing_fd < 0) {
            const int mfd_exec_errno = errno;
            backing_fd = CreateMemoryFile("BackingDmem", 0);
            if (backing_fd >= 0) {
                LOG_WARNING(Kernel_Vmm,
                            "memfd_create(MFD_EXEC) rejected ({}); fell back to non-exec memfd "
                            "(dynarec exec pages will fault on this kernel)",
                            strerror(mfd_exec_errno));
            }
        }
        if (backing_fd < 0) {
            LOG_CRITICAL(Kernel_Vmm, "memfd_create failed: {}", strerror(errno));
            throw std::bad_alloc{};
        }
#endif

        int ret = ftruncate(backing_fd, BackingSize);
        if (ret != 0) {
            LOG_CRITICAL(Kernel_Vmm, "ftruncate failed with {}, are you out-of-memory?",
                         strerror(errno));
            throw std::bad_alloc{};
        }

        backing_base = static_cast<u8*>(
            mmap(nullptr, BackingSize, PROT_READ | PROT_WRITE, MAP_SHARED, backing_fd, 0));
        if (backing_base == MAP_FAILED) {
            LOG_CRITICAL(Kernel_Vmm, "mmap failed: {}", strerror(errno));
            throw std::bad_alloc{};
        }
    }

    void* Map(VAddr virtual_addr, PAddr phys_addr, u64 size, PosixPageProtection prot,
              int fd = -1) {
        ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
                   "Refusing to MAP_FIXED non-owned range addr={:#x}, size={:#x}", virtual_addr,
                   size);
        m_free_regions.subtract({virtual_addr, virtual_addr + size});
        const int handle = phys_addr != -1 ? (fd == -1 ? backing_fd : fd) : -1;
        const off_t host_offset = phys_addr != -1 ? phys_addr : 0;
        const int flag = phys_addr != -1 ? MAP_SHARED : (MAP_ANONYMOUS | MAP_PRIVATE);
        void* ret = mmap(reinterpret_cast<void*>(virtual_addr), size, prot, MAP_FIXED | flag,
                         handle, host_offset);
        ASSERT_MSG(ret != MAP_FAILED, "mmap failed: {}", strerror(errno));
        return ret;
    }

    void ReserveRange(VAddr virtual_addr, u64 size) {
        ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
                   "Refusing to reserve non-owned range addr={:#x}, size={:#x}", virtual_addr,
                   size);
        m_free_regions.subtract({virtual_addr, virtual_addr + size});
    }

    void Unmap(VAddr virtual_addr, u64 size) {
        ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
                   "Refusing to unmap non-owned range addr={:#x}, size={:#x}", virtual_addr,
                   size);
        VAddr start_address = virtual_addr;
        VAddr end_address = start_address + size;
        const VAddr query_start = start_address == 0 ? 0 : start_address - 1;
        const VAddr query_end =
            end_address == std::numeric_limits<VAddr>::max() ? end_address : end_address + 1;
        auto it = m_free_regions.find({query_start, query_end});

        if (it != m_free_regions.end()) {
            start_address = std::min(start_address, it->lower());
            end_address = std::max(end_address, it->upper());
        }

        ASSERT_MSG(ContainsOwnedRange(start_address, end_address - start_address),
                   "Refusing to coalesce unmap beyond owned range addr={:#x}, size={:#x}",
                   start_address, end_address - start_address);

        m_free_regions.insert({start_address, end_address});

        void* ret = mmap(reinterpret_cast<void*>(start_address), end_address - start_address,
                         PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        ASSERT_MSG(ret != MAP_FAILED, "mmap failed: {}", strerror(errno));
    }

    void Protect(VAddr virtual_addr, u64 size, bool read, bool write, bool execute) {
        ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
                   "Refusing to mprotect non-owned range addr={:#x}, size={:#x}", virtual_addr,
                   size);
        int flags = PROT_NONE;
        if (read) {
            flags |= PROT_READ;
        }
        if (write) {
            flags |= PROT_WRITE;
        }
#if defined(ARCH_X86_64) || defined(__ANDROID__)
        if (execute) {
            flags |= PROT_EXEC;
        }
#endif
        int ret = mprotect(reinterpret_cast<void*>(virtual_addr), size, flags);
        ASSERT_MSG(ret == 0, "mprotect failed: {}", strerror(errno));
    }

    int backing_fd;
    u8* backing_base{};
    u8* system_managed_base{};
    u64 system_managed_size{};
    u8* system_reserved_base{};
    u64 system_reserved_size{};
    u8* user_base{};
    u64 user_size{};
#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
    u8* android_high_user_base{};
    u64 android_high_user_size{};
#endif
    boost::icl::interval_set<VAddr> m_free_regions;
    boost::icl::interval_set<VAddr> m_owned_regions;

    [[nodiscard]] bool ContainsOwnedRange(VAddr virtual_addr, u64 size) const noexcept {
        for (const auto& region : m_owned_regions) {
            if (RangeIsContained(virtual_addr, size, region.lower(),
                                 region.upper() - region.lower())) {
                return true;
            }
            if (virtual_addr < region.lower()) {
                return false;
            }
        }
        return false;
    }

    boost::icl::interval_set<VAddr> GetUsableRegions() const {
        return m_free_regions;
    }
};
#endif

AddressSpace::AddressSpace() : impl{std::make_unique<Impl>()} {
    backing_base = impl->backing_base;
    system_managed_base = impl->system_managed_base;
    system_managed_size = impl->system_managed_size;
    system_reserved_base = impl->system_reserved_base;
    system_reserved_size = impl->system_reserved_size;
    user_base = impl->user_base;
    user_size = impl->user_size;
}

AddressSpace::~AddressSpace() = default;

bool AddressSpace::ContainsOwnedRange(VAddr virtual_addr, u64 size) const noexcept {
    return impl->ContainsOwnedRange(virtual_addr, size);
}

void* AddressSpace::Map(VAddr virtual_addr, u64 size, PAddr phys_addr, bool is_exec) {
    ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
               "Attempted to map non-owned address range {:#x}, size {:#x}", virtual_addr, size);
#if defined(ARCH_X86_64) || defined(__ANDROID__)
    const auto prot = is_exec ? PAGE_EXECUTE_READWRITE : PAGE_READWRITE;
#else
    constexpr auto prot = PAGE_READWRITE;
#endif
    return impl->Map(virtual_addr, phys_addr, size, prot);
}

void* AddressSpace::MapFile(VAddr virtual_addr, u64 size, u64 offset, u32 prot, uintptr_t fd) {
    ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
               "Attempted to map file into non-owned address range {:#x}, size {:#x}",
               virtual_addr, size);
#ifdef _WIN32
    return impl->Map(virtual_addr, offset, size,
                     ToWindowsProt(std::bit_cast<Core::MemoryProt>(prot)), fd);
#else
    return impl->Map(virtual_addr, offset, size, ToPosixProt(std::bit_cast<Core::MemoryProt>(prot)),
                     fd);
#endif
}

void AddressSpace::Unmap(VAddr virtual_addr, u64 size) {
    ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
               "Attempted to unmap non-owned address range {:#x}, size {:#x}", virtual_addr,
               size);
    impl->Unmap(virtual_addr, size);
}

void AddressSpace::ReserveRange(VAddr virtual_addr, u64 size) {
#ifndef _WIN32
    ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
               "Attempted to reserve non-owned address range {:#x}, size {:#x}", virtual_addr,
               size);
    impl->ReserveRange(virtual_addr, size);
#endif
}

void AddressSpace::Protect(VAddr virtual_addr, u64 size, MemoryPermission perms) {
    ASSERT_MSG(ContainsOwnedRange(virtual_addr, size),
               "Attempted to protect non-owned address range {:#x}, size {:#x}", virtual_addr,
               size);
    const bool read = True(perms & MemoryPermission::Read);
    const bool write = True(perms & MemoryPermission::Write);
    const bool execute = True(perms & MemoryPermission::Execute);
    return impl->Protect(virtual_addr, size, read, write, execute);
}

boost::icl::interval_set<VAddr> AddressSpace::GetUsableRegions() {
#ifdef _WIN32
    return impl->GetUsableRegions();
#else
    return impl->GetUsableRegions();
#endif
}

}
