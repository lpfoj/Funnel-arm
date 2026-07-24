// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#if defined(__ANDROID__)
#include <android/log.h>
#include <cstdlib>
#endif
#include <limits>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "core/aerolib/stubs.h"
#include "core/file_sys/fs.h"
#include "core/libraries/kernel/memory.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/process.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace Core {

MemoryManager::MemoryManager() {
    LOG_INFO(Kernel_Vmm, "Virtual memory space initialized with regions:");

    auto regions = impl.GetUsableRegions();
    u64 total_usable_space = 0;
    for (auto region : regions) {
        vma_map.emplace(region.lower(),
                        VirtualMemoryArea{region.lower(), region.upper() - region.lower()});
        LOG_INFO(Kernel_Vmm, "{:#x} - {:#x}", region.lower(), region.upper());
    }

    if (!Core::AeroLib::ExecutorUseRealLibcAlloc()) {
        auto it = FindVMA(EXECUTOR_LIBC_MSPACE_BASE);
        if (it != vma_map.end() && it->second.IsFree() &&
            it->second.Contains(EXECUTOR_LIBC_MSPACE_BASE, EXECUTOR_LIBC_MSPACE_CAPACITY)) {
            const auto handle =
                CarveVMA(EXECUTOR_LIBC_MSPACE_BASE, EXECUTOR_LIBC_MSPACE_CAPACITY);
            auto& vma = handle->second;
            vma.type = VMAType::ExecutorHeap;
            impl.ReserveRange(EXECUTOR_LIBC_MSPACE_BASE, EXECUTOR_LIBC_MSPACE_CAPACITY);
            vma.prot = MemoryProt::CpuReadWrite;
            vma.name = "executor_libc_mspace";
            executor_libc_mspace_size = EXECUTOR_LIBC_MSPACE_INITIAL_SIZE;
            LOG_INFO(Kernel_Vmm,
                     "Reserved Executor libc mspace capacity {:#x} - {:#x} (initial size {:#x})",
                     EXECUTOR_LIBC_MSPACE_BASE,
                     EXECUTOR_LIBC_MSPACE_BASE + EXECUTOR_LIBC_MSPACE_CAPACITY,
                     executor_libc_mspace_size);
        } else {
            LOG_ERROR(Kernel_Vmm,
                      "Could not reserve Executor libc mspace capacity at {:#x}; "
                      "direct/flexible maps may collide with libc heap",
                      EXECUTOR_LIBC_MSPACE_BASE);
        }
    }

    ASSERT_MSG(Libraries::Kernel::sceKernelGetCompiledSdkVersion(&sdk_version) == 0,
               "Failed to get compiled SDK version");
}

MemoryManager::~MemoryManager() = default;

void MemoryManager::SetRasterizer(Vulkan::Rasterizer* rasterizer_) {
    u64 libc_mspace_size = 0;
    {
        std::unique_lock lk{mutex};
        rasterizer = rasterizer_;
        libc_mspace_size = executor_libc_mspace_size;
    }

#ifdef __ANDROID__
    if (rasterizer == nullptr) {
        return;
    }
    if (Core::AeroLib::ExecutorUseRealLibcAlloc()) {
        LOG_INFO(Kernel_Vmm,
                 "Guest libc allocator selected; bootstrap Executor mspace reservation disabled");
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIBC_OWNER] owner=guest-libc "
                            "action=skip-executor-fallback-reservation");
#endif
    } else {

        if (libc_mspace_size != 0) {
            rasterizer->MapMemory(EXECUTOR_LIBC_MSPACE_BASE, libc_mspace_size);
        }
    }
#endif
}

u64 MemoryManager::GrowExecutorLibcMspace(u64 required_size) {
    if (required_size == 0 || required_size > EXECUTOR_LIBC_MSPACE_CAPACITY) {
        return 0;
    }

    const u64 rounded_size =
        std::min(EXECUTOR_LIBC_MSPACE_CAPACITY,
                 (required_size + EXECUTOR_LIBC_MSPACE_GROW_GRANULARITY - 1) &
                     ~(EXECUTOR_LIBC_MSPACE_GROW_GRANULARITY - 1));
    u64 old_size = 0;
    Vulkan::Rasterizer* active_rasterizer = nullptr;
    {
        std::unique_lock lk{mutex};
        if (executor_libc_mspace_size == 0) {
            return 0;
        }
        if (required_size <= executor_libc_mspace_size) {
            return executor_libc_mspace_size;
        }

        const auto vma = FindVMA(EXECUTOR_LIBC_MSPACE_BASE);
        if (vma == vma_map.end() || vma->second.type != VMAType::ExecutorHeap ||
            vma->second.name != "executor_libc_mspace" ||
            !vma->second.Contains(EXECUTOR_LIBC_MSPACE_BASE, EXECUTOR_LIBC_MSPACE_CAPACITY)) {
            LOG_ERROR(Kernel_Vmm,
                      "Executor libc mspace growth rejected: mapped capacity invariant lost");
            return 0;
        }

        old_size = executor_libc_mspace_size;
        executor_libc_mspace_size = rounded_size;
        active_rasterizer = rasterizer;
    }

    if (active_rasterizer != nullptr && rounded_size > old_size) {
        active_rasterizer->MapMemory(EXECUTOR_LIBC_MSPACE_BASE + old_size,
                                     rounded_size - old_size);
    }

    LOG_INFO(Kernel_Vmm, "Grew Executor libc mspace from {:#x} to {:#x} (required {:#x})",
             old_size, rounded_size, required_size);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIBC_MSPACE_GROW] old=0x%llx new=0x%llx "
                        "required=0x%llx capacity=0x%llx",
                        static_cast<unsigned long long>(old_size),
                        static_cast<unsigned long long>(rounded_size),
                        static_cast<unsigned long long>(required_size),
                        static_cast<unsigned long long>(EXECUTOR_LIBC_MSPACE_CAPACITY));
#endif
    return rounded_size;
}

void MemoryManager::SetupMemoryRegions(u64 flexible_size, bool use_extended_mem1,
                                       bool use_extended_mem2) {
    const bool is_neo = ::Libraries::Kernel::sceKernelIsNeoMode();
    auto total_size = is_neo ? ORBIS_KERNEL_TOTAL_MEM_PRO : ORBIS_KERNEL_TOTAL_MEM;
    if (Config::isDevKitConsole()) {
        total_size = is_neo ? ORBIS_KERNEL_TOTAL_MEM_DEV_PRO : ORBIS_KERNEL_TOTAL_MEM_DEV;
    }
    s32 extra_dmem = Config::getExtraDmemInMbytes();
    if (Config::getExtraDmemInMbytes() != 0) {
        LOG_WARNING(Kernel_Vmm,
                    "extraDmemInMbytes is {} MB! Old Direct Size: {:#x} -> New Direct Size: {:#x}",
                    extra_dmem, total_size, total_size + extra_dmem * 1_MB);
        total_size += extra_dmem * 1_MB;
    }
    if (!use_extended_mem1 && is_neo) {
        total_size -= 256_MB;
    }
    if (!use_extended_mem2 && !is_neo) {
        total_size -= 128_MB;
    }
    total_flexible_size = flexible_size - ORBIS_FLEXIBLE_MEMORY_BASE;
    total_direct_size = total_size - flexible_size;

    dmem_map.clear();
    dmem_map.emplace(0, PhysicalMemoryArea{0, total_direct_size});

    const auto remaining_physical_space = total_size - total_direct_size;
    fmem_map.clear();
    fmem_map.emplace(total_direct_size,
                     PhysicalMemoryArea{total_direct_size, remaining_physical_space});

    LOG_INFO(Kernel_Vmm, "Configured memory regions: flexible size = {:#x}, direct size = {:#x}",
             total_flexible_size, total_direct_size);
}

u64 MemoryManager::ClampRangeSize(VAddr virtual_addr, u64 size) {
    static constexpr u64 MinSizeToClamp = 1_GB;
    if (size < MinSizeToClamp) {
        return size;
    }

    std::shared_lock lk{mutex};
    ASSERT_MSG(IsValidMapping(virtual_addr), "Attempted to access invalid address {:#x}",
               virtual_addr);

    auto vma = FindVMA(virtual_addr);
    u64 clamped_size = vma->second.base + vma->second.size - virtual_addr;
    ++vma;

    while (vma != vma_map.end() && vma->second.IsMapped() && clamped_size < size) {
        clamped_size += vma->second.size;
        ++vma;
    }
    clamped_size = std::min(clamped_size, size);

    if (size != clamped_size) {
        LOG_WARNING(Kernel_Vmm, "Clamped requested buffer range addr={:#x}, size={:#x} to {:#x}",
                    virtual_addr, size, clamped_size);
    }
    return clamped_size;
}

void MemoryManager::SetPrtArea(u32 id, VAddr address, u64 size) {
    PrtArea& area = prt_areas[id];
    if (rasterizer && area.mapped) {
        rasterizer->UnmapMemory(area.start, area.end - area.start);
    }

    area.start = address;
    area.end = address + size;
    area.mapped = true;

    if (rasterizer) {
        rasterizer->MapMemory(address, size);
    }
}

void MemoryManager::RegisterReplayMemory(VAddr base, u64 size, const u8* backing) {
    std::unique_lock lk{mutex};
    replay_ranges.push_back({base, size, backing});
}

void MemoryManager::ClearReplayMemory() {
    std::unique_lock lk{mutex};
    replay_ranges.clear();
    replay_canaries.clear();
}

void MemoryManager::RegisterReplayCanary(VAddr addr, u32 expected) {
    std::unique_lock lk{mutex};
    replay_canaries.push_back({addr, expected});
}

void MemoryManager::VerifyReplayCanaries(const char* where) {
#ifdef __ANDROID__
    std::shared_lock lk{mutex};
    u32 bad = 0;
    for (const auto& c : replay_canaries) {
        const u32 got = *reinterpret_cast<const volatile u32*>(c.addr);
        if (got != c.expected) {
            ++bad;
            __android_log_print(ANDROID_LOG_ERROR, "EXECUTOR",
                                "[EXECUTOR_CANARY] CORRUPT at=%s addr=0x%llx expected=0x%08x got=0x%08x",
                                where, (unsigned long long)c.addr, c.expected, got);
        }
    }
    if (bad == 0) {
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_CANARY] ok at=%s count=%zu", where,
                            replay_canaries.size());
    }
#endif
}

const MemoryManager::ReplayRange* MemoryManager::FindReplayRange(VAddr addr) const {
    for (const auto& r : replay_ranges) {
        if (addr >= r.base && addr < r.base + r.size) {
            return &r;
        }
    }
    return nullptr;
}

void MemoryManager::CopySparseMemory(VAddr virtual_addr, u8* dest, u64 size) {
    std::shared_lock lk{mutex};
    if (!replay_ranges.empty()) {
        while (size) {
            const ReplayRange* rr = FindReplayRange(virtual_addr);
            if (!rr) {
                break;
            }
            const u64 avail = rr->base + rr->size - virtual_addr;
            const u64 copy_size = std::min<u64>(avail, size);
            std::memcpy(dest, rr->backing + (virtual_addr - rr->base), copy_size);
            size -= copy_size;
            virtual_addr += copy_size;
            dest += copy_size;
        }
        if (size == 0) {
            return;
        }
    }
    ASSERT_MSG(IsValidMapping(virtual_addr), "Attempted to access invalid address {:#x}",
               virtual_addr);

    auto vma = FindVMA(virtual_addr);
    while (size) {
        u64 copy_size = std::min<u64>(vma->second.size - (virtual_addr - vma->first), size);
        if (vma->second.IsMapped()) {
            std::memcpy(dest, std::bit_cast<const u8*>(virtual_addr), copy_size);
        } else {
            std::memset(dest, 0, copy_size);
        }
        size -= copy_size;
        virtual_addr += copy_size;
        dest += copy_size;
        ++vma;
    }
}

bool MemoryManager::IsReadableMapping(VAddr virtual_addr, u64 size, bool* address_managed) {
    if (address_managed != nullptr) {
        *address_managed = false;
    }
    if (size == 0) {
        return false;
    }

    std::shared_lock lk{mutex};
    if (vma_map.empty() || virtual_addr < vma_map.begin()->first) {
        return false;
    }

    auto vma = FindVMA(virtual_addr);
    if (vma == vma_map.end()) {
        return false;
    }
    const auto& first_area = vma->second;
    if (first_area.size > std::numeric_limits<VAddr>::max() - first_area.base ||
        virtual_addr < first_area.base || virtual_addr >= first_area.base + first_area.size) {
        return false;
    }
    if (address_managed != nullptr) {
        *address_managed = true;
    }
    if (virtual_addr > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    const VAddr range_end = virtual_addr + size;
    VAddr cursor = virtual_addr;
    while (cursor < range_end) {
        if (vma == vma_map.end()) {
            return false;
        }
        const auto& area = vma->second;
        if (area.size > std::numeric_limits<VAddr>::max() - area.base) {
            return false;
        }
        const VAddr area_end = area.base + area.size;
        if (cursor < area.base || cursor >= area_end || !area.IsMapped() ||
            !True(area.prot & MemoryProt::CpuRead)) {
            return false;
        }

        cursor = std::min(range_end, area_end);
        if (cursor < range_end) {
            ++vma;
            if (vma == vma_map.end() || vma->second.base != cursor) {
                return false;
            }
        }
    }
    return true;
}

bool MemoryManager::TryCopyReadableMemory(VAddr virtual_addr, void* dest, u64 size,
                                          bool gpu_backing_synchronized) {
    if (size == 0) {
        return true;
    }
    if (dest == nullptr || virtual_addr > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    std::shared_lock lk{mutex};
    VAddr cursor = virtual_addr;
    u64 remaining = size;
    auto* output = static_cast<u8*>(dest);

    while (remaining != 0) {
        if (const ReplayRange* replay = FindReplayRange(cursor)) {
            if (replay->size > std::numeric_limits<VAddr>::max() - replay->base) {
                return false;
            }
            const u64 available = replay->base + replay->size - cursor;
            const u64 copy_size = std::min(remaining, available);
            std::memcpy(output, replay->backing + (cursor - replay->base), copy_size);
            cursor += copy_size;
            output += copy_size;
            remaining -= copy_size;
            continue;
        }

        if (vma_map.empty() || cursor < vma_map.begin()->first) {
            return false;
        }
        auto vma = FindVMA(cursor);
        if (vma == vma_map.end()) {
            return false;
        }
        const auto& area = vma->second;
        if (area.size > std::numeric_limits<VAddr>::max() - area.base) {
            return false;
        }
        const VAddr area_end = area.base + area.size;
        const bool cpu_readable = True(area.prot & MemoryProt::CpuRead);
        const bool synchronized_gpu_readable =
            gpu_backing_synchronized && True(area.prot & MemoryProt::GpuRead);
        if (cursor < area.base || cursor >= area_end || !area.IsMapped() ||
            (!cpu_readable && !synchronized_gpu_readable)) {
            return false;
        }

        const u64 copy_size = std::min<u64>(remaining, area_end - cursor);
        std::memcpy(output, std::bit_cast<const void*>(cursor), copy_size);
        cursor += copy_size;
        output += copy_size;
        remaining -= copy_size;
    }
    return true;
}

bool MemoryManager::TryCopyBackingMemory(VAddr virtual_addr, void* dest, u64 size) {
    if (size == 0) {
        return true;
    }
    if (dest == nullptr || virtual_addr > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    std::shared_lock lk{mutex};
    VAddr cursor = virtual_addr;
    u64 remaining = size;
    auto* output = static_cast<u8*>(dest);

    while (remaining != 0) {
        if (const ReplayRange* replay = FindReplayRange(cursor)) {
            if (replay->size > std::numeric_limits<VAddr>::max() - replay->base) {
                return false;
            }
            const u64 available = replay->base + replay->size - cursor;
            const u64 copy_size = std::min(remaining, available);
            std::memcpy(output, replay->backing + (cursor - replay->base), copy_size);
            cursor += copy_size;
            output += copy_size;
            remaining -= copy_size;
            continue;
        }

        if (vma_map.empty() || cursor < vma_map.begin()->first) {
            return false;
        }
        auto vma_handle = FindVMA(cursor);
        if (vma_handle == vma_map.end()) {
            return false;
        }
        const auto& vma = vma_handle->second;
        if (vma.size > std::numeric_limits<VAddr>::max() - vma.base ||
            cursor < vma.base || cursor >= vma.base + vma.size || !vma.IsMapped() ||
            !True(vma.prot & MemoryProt::CpuRead)) {
            return false;
        }

        if (vma.type == VMAType::ExecutorHeap || !HasPhysicalBacking(vma)) {
            return false;
        }

        const u64 offset_in_vma = cursor - vma.base;
        auto phys_handle = vma.phys_areas.upper_bound(offset_in_vma);
        if (phys_handle == vma.phys_areas.begin()) {
            return false;
        }
        --phys_handle;
        const u64 phys_offset = phys_handle->first;
        const auto& physical = phys_handle->second;
        if (offset_in_vma < phys_offset || offset_in_vma - phys_offset >= physical.size) {
            return false;
        }

        const u64 offset_in_physical = offset_in_vma - phys_offset;
        const u64 vma_available = vma.size - offset_in_vma;
        const u64 physical_available = physical.size - offset_in_physical;
        const u64 copy_size = std::min({remaining, vma_available, physical_available});
        if (copy_size == 0) {
            return false;
        }
        std::memcpy(output, impl.BackingBase() + physical.base + offset_in_physical, copy_size);
        cursor += copy_size;
        output += copy_size;
        remaining -= copy_size;
    }
    return true;
}

bool MemoryManager::IsWritableMapping(VAddr virtual_addr, u64 size) {
    if (size == 0 || virtual_addr > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    std::shared_lock lk{mutex};
    if (vma_map.empty() || virtual_addr < vma_map.begin()->first) {
        return false;
    }

    const VAddr range_end = virtual_addr + size;
    VAddr cursor = virtual_addr;
    auto vma = FindVMA(cursor);
    while (cursor < range_end) {
        if (vma == vma_map.end()) {
            return false;
        }
        const auto& area = vma->second;
        if (area.size > std::numeric_limits<VAddr>::max() - area.base) {
            return false;
        }
        const VAddr area_end = area.base + area.size;
        if (cursor < area.base || cursor >= area_end || !area.IsMapped() ||
            !True(area.prot & MemoryProt::CpuWrite)) {
            return false;
        }

        cursor = std::min(range_end, area_end);
        if (cursor < range_end) {
            ++vma;
            if (vma == vma_map.end() || vma->second.base != cursor) {
                return false;
            }
        }
    }
    return true;
}

bool MemoryManager::TryWriteBacking(void* address, const void* data, u64 size) {
    const VAddr virtual_addr = std::bit_cast<VAddr>(address);
    if (size == 0) {
        return true;
    }
    if (address == nullptr || data == nullptr || virtual_addr > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    std::shared_lock lk{mutex};
    if (!IsValidMapping(virtual_addr, size)) {
        return false;
    }

    const auto direct_vma = FindVMA(virtual_addr);
    if (direct_vma != vma_map.end() && direct_vma->second.type == VMAType::ExecutorHeap) {
        if (!direct_vma->second.Contains(virtual_addr, size)) {
            return false;
        }
        std::memcpy(std::bit_cast<void*>(virtual_addr), data, size);
        return true;
    }

    struct BackingSpan {
        u8* destination{};
        u64 size{};
    };
    std::vector<BackingSpan> spans;

    VAddr cursor = virtual_addr;
    u64 remaining = size;
    auto vma_handle = FindVMA(cursor);
    while (remaining != 0) {
        if (vma_handle == vma_map.end()) {
            return false;
        }
        const auto& vma = vma_handle->second;
        if (cursor < vma.base || cursor >= vma.base + vma.size || !HasPhysicalBacking(vma)) {
            return false;
        }

        const u64 offset_in_vma = cursor - vma.base;
        auto phys_handle = vma.phys_areas.upper_bound(offset_in_vma);
        if (phys_handle == vma.phys_areas.begin()) {
            return false;
        }
        --phys_handle;
        const u64 phys_offset = phys_handle->first;
        const auto& physical = phys_handle->second;
        if (offset_in_vma < phys_offset || offset_in_vma - phys_offset >= physical.size) {
            return false;
        }

        const u64 offset_in_physical = offset_in_vma - phys_offset;
        const u64 vma_available = vma.size - offset_in_vma;
        const u64 physical_available = physical.size - offset_in_physical;
        const u64 copy_size = std::min({remaining, vma_available, physical_available});
        if (copy_size == 0) {
            return false;
        }
        spans.push_back({impl.BackingBase() + physical.base + offset_in_physical, copy_size});
        cursor += copy_size;
        remaining -= copy_size;

        if (remaining != 0 && cursor == vma.base + vma.size) {
            ++vma_handle;
            if (vma_handle == vma_map.end() || vma_handle->second.base != cursor) {
                return false;
            }
        }
    }

    const auto* source = static_cast<const u8*>(data);
    u64 source_offset = 0;
    for (const auto& span : spans) {
        std::memcpy(span.destination, source + source_offset, span.size);
        source_offset += span.size;
    }
    return source_offset == size;
}

PAddr MemoryManager::PoolExpand(PAddr search_start, PAddr search_end, u64 size, u64 alignment) {
    std::scoped_lock lk{mutex, unmap_mutex};
    alignment = alignment > 0 ? alignment : 64_KB;

    auto dmem_area = FindDmemArea(search_start);
    auto mapping_start = search_start > dmem_area->second.base
                             ? Common::AlignUp(search_start, alignment)
                             : Common::AlignUp(dmem_area->second.base, alignment);
    auto mapping_end = mapping_start + size;

    while (dmem_area->second.dma_type != PhysicalMemoryType::Free ||
           dmem_area->second.GetEnd() < mapping_end) {
        dmem_area++;
        if (dmem_area == dmem_map.end()) {
            break;
        }

        mapping_start = Common::AlignUp(dmem_area->second.base, alignment);
        mapping_end = mapping_start + size;
    }

    if (dmem_area == dmem_map.end()) {
        LOG_ERROR(Kernel_Vmm, "Unable to find free direct memory area: size = {:#x}", size);
        return -1;
    }

    auto& area = CarvePhysArea(dmem_map, mapping_start, size)->second;
    area.dma_type = PhysicalMemoryType::Pooled;
    area.memory_type = 3;

    pool_budget += size;

    return mapping_start;
}

PAddr MemoryManager::Allocate(PAddr search_start, PAddr search_end, u64 size, u64 alignment,
                              s32 memory_type) {
    std::scoped_lock lk{mutex, unmap_mutex};
    alignment = alignment > 0 ? alignment : 16_KB;

    auto dmem_area = FindDmemArea(search_start);
    auto mapping_start =
        Common::AlignUp(std::max<PAddr>(search_start, dmem_area->second.base), alignment);
    auto mapping_end = mapping_start + size;

    while (dmem_area->second.dma_type != PhysicalMemoryType::Free ||
           dmem_area->second.GetEnd() < mapping_end) {
        dmem_area++;
        if (dmem_area == dmem_map.end()) {
            break;
        }

        mapping_start = Common::AlignUp(dmem_area->second.base, alignment);
        mapping_end = mapping_start + size;
    }

    if (dmem_area == dmem_map.end()) {
        LOG_ERROR(Kernel_Vmm, "Unable to find free direct memory area: size = {:#x}", size);
        return -1;
    }

    auto& area = CarvePhysArea(dmem_map, mapping_start, size)->second;
    area.memory_type = memory_type;
    area.dma_type = PhysicalMemoryType::Allocated;
    MergeAdjacent(dmem_map, dmem_area);

    return mapping_start;
}

s32 MemoryManager::Free(PAddr phys_addr, u64 size, bool is_checked) {
    if (phys_addr > total_direct_size || (is_checked && phys_addr + size > total_direct_size)) {
        LOG_ERROR(Kernel_Vmm, "phys_addr {:#x}, size {:#x} goes outside dmem map", phys_addr, size);
        if (is_checked) {
            return ORBIS_KERNEL_ERROR_ENOENT;
        }
        return ORBIS_OK;
    }

    std::scoped_lock lk{unmap_mutex};
    std::vector<std::pair<PAddr, u64>> free_list;
    u64 remaining_size = size;
    auto phys_handle = FindDmemArea(phys_addr);
    for (; phys_handle != dmem_map.end(); phys_handle++) {
        if (remaining_size == 0) {
            break;
        }
        auto& dmem_area = phys_handle->second;
        if (dmem_area.dma_type == PhysicalMemoryType::Free) {
            if (is_checked) {
                LOG_ERROR(Kernel_Vmm, "Attempting to release a free dmem area");
                return ORBIS_KERNEL_ERROR_ENOENT;
            }
            continue;
        }

        const PAddr current_phys_addr = std::max<PAddr>(phys_addr, phys_handle->first);
        const u64 start_in_dma = current_phys_addr - phys_handle->first;
        const u64 size_in_dma =
            std::min<u64>(remaining_size, phys_handle->second.size - start_in_dma);
        free_list.emplace_back(current_phys_addr, size_in_dma);

        remaining_size -= size_in_dma;
    }

    std::vector<std::pair<VAddr, u64>> remove_list;
    for (const auto& [addr, mapping] : vma_map) {
        if (mapping.type != VMAType::Direct) {
            continue;
        }
        for (auto& [offset_in_vma, phys_mapping] : mapping.phys_areas) {
            if (phys_addr + size > phys_mapping.base &&
                phys_addr < phys_mapping.base + phys_mapping.size) {
                const u64 phys_offset =
                    std::max<u64>(phys_mapping.base, phys_addr) - phys_mapping.base;
                const VAddr addr_in_vma = mapping.base + offset_in_vma + phys_offset;
                const u64 unmap_size = std::min<u64>(phys_mapping.size - phys_offset, size);

                remove_list.emplace_back(addr_in_vma, unmap_size);
            }
        }
    }

    for (auto& [addr, unmap_size] : remove_list) {
        if (rasterizer && IsValidGpuMapping(addr, unmap_size)) {
            rasterizer->UnmapMemory(addr, unmap_size);
        }
    }

    std::scoped_lock lk2{mutex};

    for (const auto& [addr, size] : remove_list) {
        LOG_INFO(Kernel_Vmm, "Unmapping direct mapping {:#x} with size {:#x}", addr, size);
        UnmapMemoryImpl(addr, size);
    }

    for (auto& [phys_addr, size] : free_list) {
        const auto dmem_handle = CarvePhysArea(dmem_map, phys_addr, size);
        auto& new_dmem_area = dmem_handle->second;
        new_dmem_area.dma_type = PhysicalMemoryType::Free;
        new_dmem_area.memory_type = 0;

        MergeAdjacent(dmem_map, dmem_handle);
    }

    return ORBIS_OK;
}

s32 MemoryManager::PoolCommit(VAddr virtual_addr, u64 size, MemoryProt prot, s32 mtype) {
    std::scoped_lock lk{unmap_mutex};
    std::unique_lock lk2{mutex};
    ASSERT_MSG(IsValidMapping(virtual_addr, size), "Attempted to access invalid address {:#x}",
               virtual_addr);

    const u64 alignment = 64_KB;
    VAddr mapped_addr = Common::AlignUp(virtual_addr, alignment);

    auto& vma = FindVMA(mapped_addr)->second;
    if (vma.type != VMAType::PoolReserved) {
        LOG_ERROR(Kernel_Vmm, "Attempting to commit non-pooled memory at {:#x}", mapped_addr);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (!vma.Contains(mapped_addr, size)) {
        LOG_ERROR(Kernel_Vmm,
                  "Pooled region {:#x} to {:#x} is not large enough to commit from {:#x} to {:#x}",
                  vma.base, vma.base + vma.size, mapped_addr, mapped_addr + size);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (pool_budget <= size) {
        LOG_ERROR(Kernel_Vmm, "Not enough pooled memory to perform mapping");
        return ORBIS_KERNEL_ERROR_ENOMEM;
    } else {
        pool_budget -= size;
    }

    if (True(prot & MemoryProt::CpuWrite)) {
        prot |= MemoryProt::CpuRead;
    }

    const auto new_vma_handle = CarveVMA(virtual_addr, size);
    auto& new_vma = new_vma_handle->second;
    new_vma.disallow_merge = false;
    new_vma.prot = prot;
    new_vma.name = "anon";
    new_vma.type = Core::VMAType::Pooled;
    new_vma.phys_areas.clear();

    auto handle = dmem_map.begin();
    u64 remaining_size = size;
    VAddr current_addr = mapped_addr;
    while (handle != dmem_map.end() && remaining_size > 0) {
        if (handle->second.dma_type != PhysicalMemoryType::Pooled) {
            handle++;
            continue;
        }

        u64 size_to_map = std::min<u64>(remaining_size, handle->second.size);

        const auto new_dmem_handle = CarvePhysArea(dmem_map, handle->second.base, size_to_map);
        auto& new_dmem_area = new_dmem_handle->second;
        new_dmem_area.dma_type = PhysicalMemoryType::Committed;
        new_dmem_area.memory_type = mtype;

        new_vma.phys_areas[current_addr - mapped_addr] = new_dmem_handle->second;
        MergeAdjacent(new_vma.phys_areas, new_vma.phys_areas.find(current_addr - mapped_addr));

        void* out_addr = impl.Map(current_addr, size_to_map, new_dmem_area.base);

        handle = MergeAdjacent(dmem_map, new_dmem_handle);
        current_addr += size_to_map;
        remaining_size -= size_to_map;
        handle++;
    }
    ASSERT_MSG(remaining_size == 0, "Failed to commit pooled memory");

    MergeAdjacent(vma_map, new_vma_handle);

    lk2.unlock();
    if (rasterizer && IsValidGpuMapping(mapped_addr, size)) {
        rasterizer->MapMemory(mapped_addr, size);
    }

    return ORBIS_OK;
}

MemoryManager::VMAHandle MemoryManager::CreateArea(VAddr virtual_addr, u64 size, MemoryProt prot,
                                                   MemoryMapFlags flags, VMAType type,
                                                   std::string_view name, u64 alignment) {
    auto vma = FindVMA(virtual_addr)->second;
    if (True(flags & MemoryMapFlags::Fixed)) {
        auto unmap_addr = virtual_addr;
        auto unmap_size = size;
        while (unmap_size > 0) {
            auto unmapped = UnmapBytesFromEntry(unmap_addr, vma, unmap_size);
            unmap_addr += unmapped;
            unmap_size -= unmapped;
            vma = FindVMA(unmap_addr)->second;
        }
    }
    vma = FindVMA(virtual_addr)->second;

    ASSERT_MSG(vma.IsFree(), "VMA to map is not free");

    const auto new_vma_handle = CarveVMA(virtual_addr, size);
    auto& new_vma = new_vma_handle->second;
    const bool is_exec = True(prot & MemoryProt::CpuExec);
    if (True(prot & MemoryProt::CpuWrite)) {
        prot |= MemoryProt::CpuRead;
    }

    new_vma.disallow_merge = True(flags & MemoryMapFlags::NoCoalesce);
    new_vma.prot = prot;
    new_vma.name = name;
    new_vma.type = type;
    new_vma.phys_areas.clear();
    return new_vma_handle;
}

s32 MemoryManager::MapMemory(void** out_addr, VAddr virtual_addr, u64 size, MemoryProt prot,
                             MemoryMapFlags flags, VMAType type, std::string_view name,
                             bool validate_dmem, PAddr phys_addr, u64 alignment) {
    if (type == VMAType::Flexible && flexible_usage + size > total_flexible_size) {
        LOG_ERROR(Kernel_Vmm,
                  "Out of flexible memory, available flexible memory = {:#x}"
                  " requested size = {:#x}",
                  total_flexible_size - flexible_usage, size);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    std::scoped_lock lk{unmap_mutex};

    PhysHandle dmem_area;
    if (phys_addr != -1) {
        if (phys_addr > total_direct_size || size > total_direct_size - phys_addr) {
            LOG_ERROR(Kernel_Vmm, "Unable to map {:#x} bytes at physical address {:#x}", size,
                      phys_addr);
            return ORBIS_KERNEL_ERROR_ENOMEM;
        }

        auto dmem_area = FindDmemArea(phys_addr);
        while (dmem_area != dmem_map.end() && dmem_area->second.base < phys_addr + size) {
            if (dmem_area->second.dma_type != PhysicalMemoryType::Allocated &&
                dmem_area->second.dma_type != PhysicalMemoryType::Mapped) {
                LOG_ERROR(Kernel_Vmm, "Unable to map {:#x} bytes at physical address {:#x}", size,
                          phys_addr);
                return ORBIS_KERNEL_ERROR_ENOMEM;
            }

            if (validate_dmem && dmem_area->second.dma_type == PhysicalMemoryType::Mapped) {
                LOG_ERROR(Kernel_Vmm, "Unable to map {:#x} bytes at physical address {:#x}", size,
                          phys_addr);
                return ORBIS_KERNEL_ERROR_EBUSY;
            }

            dmem_area++;
        }
    }

    if (True(flags & MemoryMapFlags::Fixed) && True(flags & MemoryMapFlags::NoOverwrite)) {
        if (!IsValidMapping(virtual_addr, size)) {
            LOG_ERROR(Kernel_Vmm, "Unable to map non-owned range addr={:#x}, size={:#x}",
                      virtual_addr, size);
            return ORBIS_KERNEL_ERROR_ENOMEM;
        }
        auto vma = FindVMA(virtual_addr)->second;
        auto remaining_size = vma.base + vma.size - virtual_addr;
        if (!vma.IsFree() || remaining_size < size) {
            LOG_ERROR(Kernel_Vmm, "Unable to map {:#x} bytes at address {:#x}", size, virtual_addr);
            return ORBIS_KERNEL_ERROR_ENOMEM;
        }
    } else if (False(flags & MemoryMapFlags::Fixed)) {
        alignment = alignment > 0 ? alignment : 16_KB;
        virtual_addr = virtual_addr == 0 ? DEFAULT_MAPPING_BASE : virtual_addr;
        virtual_addr = SearchFree(virtual_addr, size, alignment);
        if (virtual_addr == -1) {
            return ORBIS_KERNEL_ERROR_ENOMEM;
        }
    }

    if (!impl.ContainsOwnedRange(virtual_addr, size) || !IsValidMapping(virtual_addr, size)) {
        LOG_ERROR(Kernel_Vmm, "Unable to map non-owned range addr={:#x}, size={:#x}", virtual_addr,
                  size);
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }

    if (rasterizer && IsValidGpuMapping(virtual_addr, size)) {
        rasterizer->UnmapMemory(virtual_addr, size);
    }

    std::unique_lock lk2{mutex};

    auto new_vma_handle = CreateArea(virtual_addr, size, prot, flags, type, name, alignment);
    auto& new_vma = new_vma_handle->second;
    auto mapped_addr = new_vma.base;
    bool is_exec = True(prot & MemoryProt::CpuExec);

    if (type == VMAType::Flexible) {
        auto handle = fmem_map.begin();
        u64 remaining_size = size;
        VAddr current_addr = mapped_addr;
        while (handle != fmem_map.end() && remaining_size != 0) {
            if (handle->second.dma_type != PhysicalMemoryType::Free) {
                handle++;
                continue;
            }

            u64 size_to_map = std::min<u64>(remaining_size, handle->second.size);

            const auto new_fmem_handle = CarvePhysArea(fmem_map, handle->second.base, size_to_map);
            auto& new_fmem_area = new_fmem_handle->second;
            new_fmem_area.dma_type = PhysicalMemoryType::Flexible;

            new_vma.phys_areas[current_addr - mapped_addr] = new_fmem_handle->second;
            MergeAdjacent(new_vma.phys_areas, new_vma.phys_areas.find(current_addr - mapped_addr));

            void* out_addr = impl.Map(current_addr, size_to_map, new_fmem_area.base, is_exec);

            handle = MergeAdjacent(fmem_map, new_fmem_handle);

            current_addr += size_to_map;
            remaining_size -= size_to_map;
            flexible_usage += size_to_map;
            handle++;
        }
        ASSERT_MSG(remaining_size == 0, "Failed to map physical memory");
    } else if (type == VMAType::Direct) {
        auto current_phys_addr = phys_addr;
        u64 remaining_size = size;
        auto dmem_area = FindDmemArea(phys_addr);
        while (dmem_area != dmem_map.end() && remaining_size > 0) {
            const auto start_phys_addr = std::max<PAddr>(current_phys_addr, dmem_area->second.base);
            const auto offset_in_dma = start_phys_addr - dmem_area->second.base;
            const auto size_in_dma =
                std::min<u64>(dmem_area->second.size - offset_in_dma, remaining_size);
            const auto dmem_handle = CarvePhysArea(dmem_map, start_phys_addr, size_in_dma);
            auto& new_dmem_area = dmem_handle->second;
            new_dmem_area.dma_type = PhysicalMemoryType::Mapped;

            const u64 offset_in_vma = current_phys_addr - phys_addr;
            new_vma.phys_areas[offset_in_vma] = dmem_handle->second;
            MergeAdjacent(new_vma.phys_areas, new_vma.phys_areas.find(offset_in_vma));

            MergeAdjacent(dmem_map, dmem_handle);

            current_phys_addr += size_in_dma;
            remaining_size -= size_in_dma;
            dmem_area = FindDmemArea(current_phys_addr);
        }
        ASSERT_MSG(remaining_size == 0, "Failed to map physical memory");
    }

    if (new_vma.type != VMAType::Direct || sdk_version >= Common::ElfInfo::FW_20) {
        MergeAdjacent(vma_map, new_vma_handle);
    }

    *out_addr = std::bit_cast<void*>(mapped_addr);
    if (type != VMAType::Reserved && type != VMAType::PoolReserved) {
        if (type != VMAType::Flexible) {
            impl.Map(mapped_addr, size, phys_addr, is_exec);
        }

        lk2.unlock();

            if (rasterizer && IsValidGpuMapping(mapped_addr, size)) {
                rasterizer->MapMemory(mapped_addr, size);
            }
    }

    return ORBIS_OK;
}

s32 MemoryManager::MapFile(void** out_addr, VAddr virtual_addr, u64 size, MemoryProt prot,
                           MemoryMapFlags flags, s32 fd, s64 phys_addr) {
    uintptr_t handle = 0;
    std::scoped_lock lk{unmap_mutex};
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto file = h->GetFile(fd);
    if (file == nullptr) {
        LOG_WARNING(Kernel_Vmm, "Invalid file for mmap, fd {}", fd);
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    if (file->type != Core::FileSys::FileType::Regular) {
        LOG_WARNING(Kernel_Vmm, "Unsupported file type for mmap, fd {}", fd);
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    if (True(prot & MemoryProt::CpuWrite)) {
        prot |= MemoryProt::CpuRead;
    }

    handle = file->f.GetFileMapping();

    if (False(file->f.GetAccessMode() & Common::FS::FileAccessMode::Write) ||
        False(file->f.GetAccessMode() & Common::FS::FileAccessMode::Append)) {
        prot &= ~MemoryProt::CpuWrite;
    }

    if (prot >= MemoryProt::GpuRead) {
        ASSERT_MSG(false, "Files cannot be mapped to GPU memory");
    }

    if (True(prot & MemoryProt::CpuExec)) {
        prot &= ~MemoryProt::CpuExec;
    }

    if (True(flags & MemoryMapFlags::Fixed) && False(flags & MemoryMapFlags::NoOverwrite)) {
        if (!IsValidMapping(virtual_addr, size)) {
            LOG_ERROR(Kernel_Vmm, "Unable to file-map non-owned range addr={:#x}, size={:#x}",
                      virtual_addr, size);
            return ORBIS_KERNEL_ERROR_ENOMEM;
        }
        auto vma = FindVMA(virtual_addr)->second;

        auto remaining_size = vma.base + vma.size - virtual_addr;
        if (!vma.IsFree() || remaining_size < size) {
            LOG_ERROR(Kernel_Vmm, "Unable to map {:#x} bytes at address {:#x}", size, virtual_addr);
            return ORBIS_KERNEL_ERROR_ENOMEM;
        }
    } else if (False(flags & MemoryMapFlags::Fixed)) {
        virtual_addr = virtual_addr == 0 ? DEFAULT_MAPPING_BASE : virtual_addr;
        virtual_addr = SearchFree(virtual_addr, size, 16_KB);
        if (virtual_addr == -1) {
            return ORBIS_KERNEL_ERROR_ENOMEM;
        }
    }


    if (!impl.ContainsOwnedRange(virtual_addr, size) || !IsValidMapping(virtual_addr, size)) {
        LOG_ERROR(Kernel_Vmm, "Unable to file-map non-owned range addr={:#x}, size={:#x}",
                  virtual_addr, size);
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }

    if (rasterizer && IsValidGpuMapping(virtual_addr, size)) {
        rasterizer->UnmapMemory(virtual_addr, size);
    }

    std::scoped_lock lk2{mutex};

    auto new_vma_handle = CreateArea(virtual_addr, size, prot, flags, VMAType::File, "anon", 0);

    auto& new_vma = new_vma_handle->second;
    new_vma.fd = fd;
    auto mapped_addr = new_vma.base;
    bool is_exec = True(prot & MemoryProt::CpuExec);

    impl.MapFile(mapped_addr, size, phys_addr, std::bit_cast<u32>(prot), handle);

    *out_addr = std::bit_cast<void*>(mapped_addr);
    return ORBIS_OK;
}

s32 MemoryManager::PoolDecommit(VAddr virtual_addr, u64 size) {
    std::scoped_lock lk{unmap_mutex};
    if (!impl.ContainsOwnedRange(virtual_addr, size) || !IsValidMapping(virtual_addr, size)) {
        LOG_ERROR(Kernel_Vmm, "Unable to decommit non-owned range addr={:#x}, size={:#x}",
                  virtual_addr, size);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    auto it = FindVMA(virtual_addr);
    while (it != vma_map.end() && it->second.base + it->second.size <= virtual_addr + size) {
        if (it->second.type != VMAType::PoolReserved && it->second.type != VMAType::Pooled) {
            LOG_ERROR(Kernel_Vmm, "Attempting to decommit non-pooled memory!");
            return ORBIS_KERNEL_ERROR_EINVAL;
        }
        it++;
    }

    if (rasterizer && IsValidGpuMapping(virtual_addr, size)) {
        rasterizer->UnmapMemory(virtual_addr, size);
    }

    std::scoped_lock lk2{mutex};

    u64 remaining_size = size;
    VAddr current_addr = virtual_addr;
    while (remaining_size != 0) {
        const auto handle = FindVMA(current_addr);
        const auto& vma_base = handle->second;
        const auto start_in_vma = current_addr - vma_base.base;
        const auto size_in_vma = std::min<u64>(remaining_size, vma_base.size - start_in_vma);
        if (vma_base.type == VMAType::Pooled) {
            pool_budget += size_in_vma;

            u64 size_to_free = size_in_vma;
            auto phys_handle = std::prev(vma_base.phys_areas.upper_bound(start_in_vma));
            while (phys_handle != vma_base.phys_areas.end() && size_to_free > 0) {
                u64 dma_offset =
                    std::max<PAddr>(phys_handle->first, start_in_vma) - phys_handle->first;
                PAddr phys_addr = phys_handle->second.base + dma_offset;
                u64 size_in_dma =
                    std::min<u64>(size_to_free, phys_handle->second.size - dma_offset);

                const auto new_dmem_handle = CarvePhysArea(dmem_map, phys_addr, size_in_dma);
                auto& new_dmem_area = new_dmem_handle->second;
                new_dmem_area.dma_type = PhysicalMemoryType::Pooled;

                MergeAdjacent(dmem_map, new_dmem_handle);

                size_to_free -= size_in_dma;
                phys_handle++;
            }
            ASSERT_MSG(size_to_free == 0, "Failed to decommit pooled memory");
        }

        const auto new_it = CarveVMA(current_addr, size_in_vma);
        auto& vma = new_it->second;
        vma.type = VMAType::PoolReserved;
        vma.prot = MemoryProt::NoAccess;
        vma.disallow_merge = false;
        vma.name = "anon";
        vma.phys_areas.clear();
        MergeAdjacent(vma_map, new_it);

        current_addr += size_in_vma;
        remaining_size -= size_in_vma;
    }

    impl.Unmap(virtual_addr, size);

    return ORBIS_OK;
}

s32 MemoryManager::UnmapMemory(VAddr virtual_addr, u64 size) {
    if (size == 0) {
        return ORBIS_OK;
    }

    std::scoped_lock lk{unmap_mutex};
    virtual_addr = Common::AlignDown(virtual_addr, 16_KB);
    if (size > std::numeric_limits<u64>::max() - (16_KB - 1)) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    size = Common::AlignUp(size, 16_KB);
    if (!impl.ContainsOwnedRange(virtual_addr, size) || !IsValidMapping(virtual_addr, size)) {
        LOG_ERROR(Kernel_Vmm, "Unable to unmap non-owned range addr={:#x}, size={:#x}",
                  virtual_addr, size);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (rasterizer && IsValidGpuMapping(virtual_addr, size)) {
        rasterizer->UnmapMemory(virtual_addr, size);
    }

    std::scoped_lock lk2{mutex};
    return UnmapMemoryImpl(virtual_addr, size);
}

u64 MemoryManager::UnmapBytesFromEntry(VAddr virtual_addr, VirtualMemoryArea vma_base, u64 size) {
    const auto start_in_vma = virtual_addr - vma_base.base;
    const auto size_in_vma = std::min<u64>(vma_base.size - start_in_vma, size);
    const auto vma_type = vma_base.type;
    if (vma_base.type == VMAType::Free || vma_base.type == VMAType::Pooled) {
        return size_in_vma;
    }

    VAddr current_addr = virtual_addr;
    if (vma_base.phys_areas.size() > 0) {
        u64 size_to_free = size_in_vma;
        auto phys_handle = std::prev(vma_base.phys_areas.upper_bound(start_in_vma));
        while (phys_handle != vma_base.phys_areas.end() && size_to_free > 0) {
            u64 dma_offset = std::max<PAddr>(phys_handle->first, start_in_vma) - phys_handle->first;
            PAddr phys_addr = phys_handle->second.base + dma_offset;
            u64 size_in_dma = std::min<u64>(size_to_free, phys_handle->second.size - dma_offset);

            if (vma_type == VMAType::Direct) {
                const auto new_dmem_handle = CarvePhysArea(dmem_map, phys_addr, size_in_dma);
                auto& new_dmem_area = new_dmem_handle->second;
                new_dmem_area.dma_type = PhysicalMemoryType::Allocated;

                MergeAdjacent(dmem_map, new_dmem_handle);
            } else if (vma_type == VMAType::Flexible) {
                const auto new_fmem_handle = CarvePhysArea(fmem_map, phys_addr, size_in_dma);
                auto& new_fmem_area = new_fmem_handle->second;
                new_fmem_area.dma_type = PhysicalMemoryType::Free;

                MergeAdjacent(fmem_map, new_fmem_handle);

                const auto unmap_hardware_address = impl.BackingBase() + phys_addr;
                std::memset(unmap_hardware_address, 0, size_in_dma);

                flexible_usage -= size_in_dma;
            }

            size_to_free -= size_in_dma;
            phys_handle++;
        }
        ASSERT_MSG(size_to_free == 0, "Failed to unmap physical memory");
    }

    const auto new_it = CarveVMA(virtual_addr, size_in_vma);
    auto& vma = new_it->second;
    vma.type = VMAType::Free;
    vma.prot = MemoryProt::NoAccess;
    vma.phys_areas.clear();
    vma.disallow_merge = false;
    vma.name = "";
    MergeAdjacent(vma_map, new_it);

    if (vma_type != VMAType::Reserved && vma_type != VMAType::PoolReserved) {
        impl.Unmap(virtual_addr, size_in_vma);
    }
    return size_in_vma;
}

s32 MemoryManager::UnmapMemoryImpl(VAddr virtual_addr, u64 size) {
    u64 unmapped_bytes = 0;
    do {
        auto it = FindVMA(virtual_addr + unmapped_bytes);
        auto& vma_base = it->second;
        auto unmapped =
            UnmapBytesFromEntry(virtual_addr + unmapped_bytes, vma_base, size - unmapped_bytes);
        ASSERT_MSG(unmapped > 0, "Failed to unmap memory, progress is impossible");
        unmapped_bytes += unmapped;
    } while (unmapped_bytes < size);

    return ORBIS_OK;
}

s32 MemoryManager::QueryProtection(VAddr addr, void** start, void** end, u32* prot) {
    std::shared_lock lk{mutex};
    ASSERT_MSG(IsValidMapping(addr), "Attempted to access invalid address {:#x}", addr);

    const auto it = FindVMA(addr);
    const auto& vma = it->second;
    if (vma.IsFree()) {
        LOG_ERROR(Kernel_Vmm, "Address {:#x} is not mapped", addr);
        return ORBIS_KERNEL_ERROR_EACCES;
    }

    if (start != nullptr) {
        *start = reinterpret_cast<void*>(vma.base);
    }
    if (end != nullptr) {
        *end = reinterpret_cast<void*>(vma.base + vma.size);
    }
    if (prot != nullptr) {
        *prot = static_cast<u32>(vma.prot);
    }

    return ORBIS_OK;
}

s64 MemoryManager::ProtectBytes(VAddr addr, VirtualMemoryArea& vma_base, u64 size,
                                MemoryProt prot) {
    const auto start_in_vma = addr - vma_base.base;
    const auto adjusted_size = std::min<u64>(vma_base.size - start_in_vma, size);
    const MemoryProt old_prot = vma_base.prot;
    const MemoryProt new_prot = prot;

    if (vma_base.type == VMAType::Free || vma_base.type == VMAType::PoolReserved) {
        return adjusted_size;
    }

    if (True(prot & MemoryProt::CpuWrite)) {
        prot |= MemoryProt::CpuRead;
    }

    Core::MemoryPermission perms{};

    if (True(prot & MemoryProt::CpuRead)) {
        perms |= Core::MemoryPermission::Read;
    }
    if (True(prot & MemoryProt::CpuReadWrite)) {
        perms |= Core::MemoryPermission::ReadWrite;
    }
    if (True(prot & MemoryProt::CpuExec)) {
        perms |= Core::MemoryPermission::Execute;
    }
    if (True(prot & MemoryProt::GpuRead)) {
        perms |= Core::MemoryPermission::Read;
    }
    if (True(prot & MemoryProt::GpuWrite)) {
        perms |= Core::MemoryPermission::Write;
    }
    if (True(prot & MemoryProt::GpuReadWrite)) {
        perms |= Core::MemoryPermission::ReadWrite;
    }

    if (vma_base.type == VMAType::Direct || vma_base.type == VMAType::Pooled ||
        vma_base.type == VMAType::File) {
        prot &= ~MemoryProt::CpuExec;
    }

    const auto new_it = CarveVMA(addr, adjusted_size);
    auto& new_vma = new_it->second;
    new_vma.prot = prot;
    MergeAdjacent(vma_map, new_it);

    if (vma_base.type == VMAType::Reserved) {
        return adjusted_size;
    }

    if (new_prot != old_prot) {
        impl.Protect(addr, adjusted_size, perms);
    }

    return adjusted_size;
}

s32 MemoryManager::Protect(VAddr addr, u64 size, MemoryProt prot) {
    if (size == 0) {
        return ORBIS_OK;
    }

    std::scoped_lock lk{mutex, unmap_mutex};
    if (!impl.ContainsOwnedRange(addr, size) || !IsValidMapping(addr, size)) {
        LOG_ERROR(Kernel_Vmm, "Unable to protect non-owned range addr={:#x}, size={:#x}", addr,
                  size);
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    constexpr static MemoryProt flag_mask =
        MemoryProt::CpuReadWrite | MemoryProt::CpuExec | MemoryProt::GpuReadWrite;
    MemoryProt valid_flags = prot & flag_mask;

    s64 protected_bytes = 0;
    while (protected_bytes < size) {
        auto it = FindVMA(addr + protected_bytes);
        auto& vma_base = it->second;
        if (vma_base.base > addr + protected_bytes) {
            protected_bytes += vma_base.base - (addr + protected_bytes);
        }
        auto result = ProtectBytes(addr + protected_bytes, vma_base, size - protected_bytes, prot);
        if (result < 0) {
            return result;
        }
        protected_bytes += result;
    }

    return ORBIS_OK;
}

s32 MemoryManager::VirtualQuery(VAddr addr, s32 flags,
                                ::Libraries::Kernel::OrbisVirtualQueryInfo* info) {
    auto query_addr =
        addr < impl.SystemManagedVirtualBase() ? impl.SystemManagedVirtualBase() : addr;
    if (addr < query_addr && flags == 0) {
        LOG_WARNING(Kernel_Vmm, "VirtualQuery on free memory region");
        return ORBIS_KERNEL_ERROR_EACCES;
    }

    std::shared_lock lk{mutex};
    auto it = FindVMA(query_addr);

    while (it != vma_map.end() && it->second.type == VMAType::Free && flags == 1) {
        ++it;
    }
    if (it == vma_map.end() || it->second.type == VMAType::Free) {
        LOG_WARNING(Kernel_Vmm, "VirtualQuery on free memory region");
        return ORBIS_KERNEL_ERROR_EACCES;
    }

    const auto& vma = it->second;
    info->start = vma.base;
    info->end = vma.base + vma.size;
    info->offset = 0;
    info->protection = static_cast<s32>(vma.prot);
    info->is_flexible = vma.type == VMAType::Flexible ? 1 : 0;
    info->is_direct = vma.type == VMAType::Direct ? 1 : 0;
    info->is_stack = vma.type == VMAType::Stack ? 1 : 0;
    info->is_pooled = vma.type == VMAType::PoolReserved || vma.type == VMAType::Pooled ? 1 : 0;
    info->is_committed = vma.IsMapped() ? 1 : 0;
    info->memory_type = 0;
    if (vma.type == VMAType::Direct) {
        ASSERT_MSG(vma.phys_areas.size() > 0, "No physical backing for direct mapping?");
        info->offset = vma.phys_areas.begin()->second.base;
        info->memory_type = vma.phys_areas.begin()->second.memory_type;
    }
    if (vma.type == VMAType::Reserved || vma.type == VMAType::PoolReserved) {
        info->protection = 0;
    }

    strncpy(info->name, vma.name.data(), ::Libraries::Kernel::ORBIS_KERNEL_MAXIMUM_NAME_LENGTH);

    return ORBIS_OK;
}

s32 MemoryManager::DirectMemoryQuery(PAddr addr, bool find_next,
                                     ::Libraries::Kernel::OrbisQueryInfo* out_info) {
    if (addr >= total_direct_size) {
        LOG_WARNING(Kernel_Vmm, "Unable to find allocated direct memory region to query!");
        return ORBIS_KERNEL_ERROR_EACCES;
    }

    std::shared_lock lk{mutex};
    auto dmem_area = FindDmemArea(addr);
    while (dmem_area != dmem_map.end() && dmem_area->second.dma_type == PhysicalMemoryType::Free &&
           find_next) {
        dmem_area++;
    }

    if (dmem_area == dmem_map.end() || dmem_area->second.dma_type == PhysicalMemoryType::Free) {
        LOG_WARNING(Kernel_Vmm, "Unable to find allocated direct memory region to query!");
        return ORBIS_KERNEL_ERROR_EACCES;
    }

    out_info->start = dmem_area->second.base;
    out_info->memoryType = dmem_area->second.memory_type;

    while (dmem_area != dmem_map.end() && dmem_area->second.memory_type == out_info->memoryType &&
           (dmem_area->second.dma_type == PhysicalMemoryType::Mapped ||
            dmem_area->second.dma_type == PhysicalMemoryType::Allocated)) {
        out_info->end = dmem_area->second.GetEnd();
        dmem_area++;
    }

    return ORBIS_OK;
}

s32 MemoryManager::DirectQueryAvailable(PAddr search_start, PAddr search_end, u64 alignment,
                                        PAddr* phys_addr_out, u64* size_out) {
    std::shared_lock lk{mutex};

    auto dmem_area = FindDmemArea(search_start);
    PAddr paddr{};
    u64 max_size{};

    while (dmem_area != dmem_map.end()) {
        if (dmem_area->second.dma_type != PhysicalMemoryType::Free) {
            dmem_area++;
            continue;
        }

        auto aligned_base = alignment > 0 ? Common::AlignUp(dmem_area->second.base, alignment)
                                          : dmem_area->second.base;
        const auto alignment_size = aligned_base - dmem_area->second.base;
        auto remaining_size =
            dmem_area->second.size >= alignment_size ? dmem_area->second.size - alignment_size : 0;

        if (dmem_area->second.base < search_start) {
            remaining_size = remaining_size > (search_start - dmem_area->second.base)
                                 ? remaining_size - (search_start - dmem_area->second.base)
                                 : 0;
            aligned_base = alignment > 0 ? Common::AlignUp(search_start, alignment) : search_start;
        }

        if (dmem_area->second.GetEnd() > search_end) {
            remaining_size = remaining_size > (dmem_area->second.GetEnd() - search_end)
                                 ? remaining_size - (dmem_area->second.GetEnd() - search_end)
                                 : 0;
        }

        if (remaining_size > max_size) {
            paddr = aligned_base;
            max_size = remaining_size;
        }
        dmem_area++;
    }

    *phys_addr_out = paddr;
    *size_out = max_size;
    return ORBIS_OK;
}

s32 MemoryManager::SetDirectMemoryType(VAddr addr, u64 size, s32 memory_type) {
    std::scoped_lock lk{mutex, unmap_mutex};

    ASSERT_MSG(IsValidMapping(addr, size), "Attempted to access invalid address {:#x}", addr);

    VAddr current_addr = addr;
    u64 remaining_size = size;
    auto vma_handle = FindVMA(addr);
    while (vma_handle != vma_map.end() && remaining_size > 0) {
        const VAddr start_in_vma = current_addr - vma_handle->second.base;
        const u64 size_in_vma =
            std::min<u64>(remaining_size, vma_handle->second.size - start_in_vma);

        if (vma_handle->second.type == VMAType::Direct ||
            vma_handle->second.type == VMAType::Pooled) {
            vma_handle = CarveVMA(current_addr, size_in_vma);
            auto phys_handle = vma_handle->second.phys_areas.begin();
            while (phys_handle != vma_handle->second.phys_areas.end()) {
                phys_handle->second.memory_type = memory_type;

                auto dmem_handle =
                    CarvePhysArea(dmem_map, phys_handle->second.base, phys_handle->second.size);
                auto& dmem_area = dmem_handle->second;
                dmem_area.memory_type = memory_type;

                phys_handle++;
            }

            vma_handle = MergeAdjacent(vma_map, vma_handle);
        }
        current_addr += size_in_vma;
        remaining_size -= size_in_vma;
        vma_handle++;
    }

    return ORBIS_OK;
}

void MemoryManager::NameVirtualRange(VAddr virtual_addr, u64 size, std::string_view name) {
    std::scoped_lock lk{mutex, unmap_mutex};

    u64 aligned_size = Common::AlignUp(size, 16_KB);
    VAddr aligned_addr = Common::AlignDown(virtual_addr, 16_KB);

    ASSERT_MSG(IsValidMapping(aligned_addr, aligned_size),
               "Attempted to access invalid address {:#x}", aligned_addr);
    auto it = FindVMA(aligned_addr);
    u64 remaining_size = aligned_size;
    VAddr current_addr = aligned_addr;
    while (remaining_size > 0 && it != vma_map.end()) {
        const u64 start_in_vma = current_addr - it->second.base;
        const u64 size_in_vma = std::min<u64>(remaining_size, it->second.size - start_in_vma);
        if (!it->second.IsFree()) {
            if (size_in_vma < it->second.size) {
                it = CarveVMA(current_addr, size_in_vma);
                auto& new_vma = it->second;
                new_vma.name = name;
            } else {
                auto& vma = it->second;
                vma.name = name;
            }
        }
        it = MergeAdjacent(vma_map, it);
        remaining_size -= size_in_vma;
        current_addr += size_in_vma;
        it++;
    }
}

s32 MemoryManager::GetDirectMemoryType(PAddr addr, s32* directMemoryTypeOut,
                                       void** directMemoryStartOut, void** directMemoryEndOut) {
    if (addr >= total_direct_size) {
        LOG_ERROR(Kernel_Vmm, "Unable to find allocated direct memory region to check type!");
        return ORBIS_KERNEL_ERROR_ENOENT;
    }

    std::shared_lock lk{mutex};
    const auto& dmem_area = FindDmemArea(addr)->second;
    if (dmem_area.dma_type == PhysicalMemoryType::Free) {
        LOG_ERROR(Kernel_Vmm, "Unable to find allocated direct memory region to check type!");
        return ORBIS_KERNEL_ERROR_ENOENT;
    }

    *directMemoryStartOut = reinterpret_cast<void*>(dmem_area.base);
    *directMemoryEndOut = reinterpret_cast<void*>(dmem_area.GetEnd());
    *directMemoryTypeOut = dmem_area.memory_type;
    return ORBIS_OK;
}

s32 MemoryManager::IsStack(VAddr addr, void** start, void** end) {
    std::shared_lock lk{mutex};
    ASSERT_MSG(IsValidMapping(addr), "Attempted to access invalid address {:#x}", addr);
    const auto vma_it = FindVMA(addr);
    const auto& vma = vma_it->second;
    if (vma.IsFree()) {
        return ORBIS_KERNEL_ERROR_EACCES;
    }

    u64 stack_start = 0;
    u64 stack_end = 0;
    if (vma.type == VMAType::Stack) {
        auto first = vma_it;
        while (first != vma_map.begin()) {
            auto prev = std::prev(first);
            if (prev->second.type != VMAType::Stack ||
                prev->second.base + prev->second.size != first->second.base) {
                break;
            }
            first = prev;
        }

        auto last = vma_it;
        while (std::next(last) != vma_map.end()) {
            auto next = std::next(last);
            if (next->second.type != VMAType::Stack ||
                last->second.base + last->second.size != next->second.base) {
                break;
            }
            last = next;
        }

        bool found_readable = false;
        for (auto it = first;; ++it) {
            const auto& stack_vma = it->second;
            if (stack_vma.prot != MemoryProt::NoAccess) {
                if (!found_readable) {
                    stack_start = stack_vma.base;
                    found_readable = true;
                }
                stack_end = stack_vma.base + stack_vma.size;
            } else if (found_readable) {
                break;
            }
            if (it == last) {
                break;
            }
        }

#if defined(__ANDROID__)
        if (std::getenv("EXECUTOR_LIVE_MONO_SIGNAL_DELIVER") != nullptr &&
            found_readable && (stack_start != first->second.base || stack_end != last->second.base + last->second.size)) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Memory",
                "[EXECUTOR_IS_STACK_USABLE] addr=0x%llx vma=0x%llx..0x%llx usable=0x%llx..0x%llx",
                static_cast<unsigned long long>(addr),
                static_cast<unsigned long long>(first->second.base),
                static_cast<unsigned long long>(last->second.base + last->second.size),
                static_cast<unsigned long long>(stack_start),
                static_cast<unsigned long long>(stack_end));
        }
#endif
    }

    if (start != nullptr) {
        *start = reinterpret_cast<void*>(stack_start);
    }

    if (end != nullptr) {
        *end = reinterpret_cast<void*>(stack_end);
    }
    return ORBIS_OK;
}

s32 MemoryManager::GetMemoryPoolStats(::Libraries::Kernel::OrbisKernelMemoryPoolBlockStats* stats) {
    std::shared_lock lk{mutex};

    constexpr u64 block_size = 64_KB;
    u64 committed_size = 0;

    auto dma_handle = dmem_map.begin();
    while (dma_handle != dmem_map.end()) {
        if (dma_handle->second.dma_type == PhysicalMemoryType::Committed) {
            committed_size += dma_handle->second.size;
        }
        dma_handle++;
    }

    stats->allocated_flushed_blocks = committed_size / block_size;
    stats->available_flushed_blocks = committed_size / block_size;
    stats->allocated_cached_blocks = 0;
    stats->available_cached_blocks = 0;

    return ORBIS_OK;
}

void MemoryManager::InvalidateMemory(const VAddr addr, const u64 size) const {
    if (rasterizer) {
        rasterizer->InvalidateMemory(addr, size);
    }
}

VAddr MemoryManager::SearchFree(VAddr virtual_addr, u64 size, u32 alignment) {
    auto min_search_address = impl.SystemManagedVirtualBase();
    auto max_search_address = impl.UserVirtualBase() + impl.UserVirtualSize();

    if (virtual_addr < min_search_address) {
        virtual_addr = min_search_address;
    }

    ASSERT_MSG(IsValidMapping(virtual_addr), "Input address {:#x} is out of bounds", virtual_addr);

    virtual_addr = Common::AlignUp(virtual_addr, alignment);
    auto it = FindVMA(virtual_addr);

    if (it->second.IsFree() && it->second.Contains(virtual_addr, size)) {
        return virtual_addr;
    }

    it++;

    while (it != vma_map.end()) {
        if (!it->second.IsFree()) {
            it++;
            continue;
        }

        const auto& vma = it->second;
        virtual_addr = Common::AlignUp(vma.base, alignment);
        if (virtual_addr > vma.base + vma.size) {
            it++;
            continue;
        }

        if (virtual_addr >= max_search_address) {
            break;
        }

        const u64 remaining_size = vma.base + vma.size - virtual_addr;
        if (remaining_size >= size) {
            return virtual_addr;
        }
        it++;
    }

    LOG_ERROR(Kernel_Vmm, "Couldn't find a free mapping for address {:#x}, size {:#x}",
              virtual_addr, size);
    return -1;
}

MemoryManager::VMAHandle MemoryManager::MergeAdjacent(VMAMap& handle_map, VMAHandle iter) {
    const auto next_vma = std::next(iter);
    if (next_vma != handle_map.end() && iter->second.CanMergeWith(next_vma->second)) {
        u64 base_offset = iter->second.size;
        iter->second.size += next_vma->second.size;
        for (auto& area : next_vma->second.phys_areas) {
            iter->second.phys_areas[base_offset + area.first] = area.second;
        }
        handle_map.erase(next_vma);
    }

    if (iter != handle_map.begin()) {
        auto prev_vma = std::prev(iter);
        if (prev_vma->second.CanMergeWith(iter->second)) {
            u64 base_offset = prev_vma->second.size;
            prev_vma->second.size += iter->second.size;
            for (auto& area : iter->second.phys_areas) {
                prev_vma->second.phys_areas[base_offset + area.first] = area.second;
            }
            handle_map.erase(iter);
            iter = prev_vma;
        }
    }

    return iter;
}

MemoryManager::PhysHandle MemoryManager::MergeAdjacent(PhysMap& handle_map, PhysHandle iter) {
    const auto next_vma = std::next(iter);
    if (next_vma != handle_map.end() && iter->second.CanMergeWith(next_vma->second)) {
        iter->second.size += next_vma->second.size;
        handle_map.erase(next_vma);
    }

    if (iter != handle_map.begin()) {
        auto prev_vma = std::prev(iter);
        if (prev_vma->second.CanMergeWith(iter->second)) {
            prev_vma->second.size += iter->second.size;
            handle_map.erase(iter);
            iter = prev_vma;
        }
    }

    return iter;
}

MemoryManager::VMAHandle MemoryManager::CarveVMA(VAddr virtual_addr, u64 size) {
    auto vma_handle = FindVMA(virtual_addr);

    const VirtualMemoryArea& vma = vma_handle->second;
    ASSERT_MSG(vma.base <= virtual_addr, "Adding a mapping to already mapped region");

    const VAddr start_in_vma = virtual_addr - vma.base;
    const VAddr end_in_vma = start_in_vma + size;

    if (start_in_vma == 0 && size == vma.size) {
        return vma_handle;
    }

    ASSERT_MSG(end_in_vma <= vma.size, "Mapping cannot fit inside free region");

    if (end_in_vma != vma.size) {
        Split(vma_handle, end_in_vma);
    }
    if (start_in_vma != 0) {
        vma_handle = Split(vma_handle, start_in_vma);
    }

    return vma_handle;
}

MemoryManager::PhysHandle MemoryManager::CarvePhysArea(PhysMap& map, PAddr addr, u64 size) {
    auto pmem_handle = std::prev(map.upper_bound(addr));
    ASSERT_MSG(addr <= pmem_handle->second.GetEnd(), "Physical address not in map");

    const PhysicalMemoryArea& area = pmem_handle->second;
    ASSERT_MSG(area.base <= addr, "Adding an allocation to already allocated region");

    const PAddr start_in_area = addr - area.base;
    const PAddr end_in_vma = start_in_area + size;
    ASSERT_MSG(end_in_vma <= area.size, "Mapping cannot fit inside free region: size = {:#x}",
               size);

    if (end_in_vma != area.size) {
        Split(map, pmem_handle, end_in_vma);
    }
    if (start_in_area != 0) {
        pmem_handle = Split(map, pmem_handle, start_in_area);
    }

    return pmem_handle;
}

MemoryManager::VMAHandle MemoryManager::Split(VMAHandle vma_handle, u64 offset_in_vma) {
    auto& old_vma = vma_handle->second;
    ASSERT(offset_in_vma < old_vma.size && offset_in_vma > 0);

    auto new_vma = old_vma;
    old_vma.size = offset_in_vma;
    new_vma.base += offset_in_vma;
    new_vma.size -= offset_in_vma;

    if (HasPhysicalBacking(new_vma)) {
        new_vma.phys_areas.clear();

        std::map<uintptr_t, PhysicalMemoryArea> old_vma_phys_areas;
        for (auto& [offset, region] : old_vma.phys_areas) {
            if (offset + region.size <= offset_in_vma) {
                old_vma_phys_areas[offset] = region;
            }
            if (offset < offset_in_vma && offset + region.size > offset_in_vma) {
                u64 size_in_old = offset_in_vma - offset;
                old_vma_phys_areas[offset] = PhysicalMemoryArea{
                    region.base, size_in_old, region.memory_type, region.dma_type};
                PAddr new_base = region.base + size_in_old;
                u64 size_in_new = region.size - size_in_old;
                new_vma.phys_areas[0] =
                    PhysicalMemoryArea{new_base, size_in_new, region.memory_type, region.dma_type};
            }
            if (offset >= offset_in_vma) {
                new_vma.phys_areas[offset - offset_in_vma] = region;
            }
        }

        old_vma.phys_areas = old_vma_phys_areas;
    }

    return vma_map.emplace_hint(std::next(vma_handle), new_vma.base, new_vma);
}

MemoryManager::PhysHandle MemoryManager::Split(PhysMap& map, PhysHandle phys_handle,
                                               u64 offset_in_area) {
    auto& old_area = phys_handle->second;
    ASSERT(offset_in_area < old_area.size && offset_in_area > 0);

    auto new_area = old_area;
    old_area.size = offset_in_area;
    new_area.memory_type = old_area.memory_type;
    new_area.base += offset_in_area;
    new_area.size -= offset_in_area;

    return map.emplace_hint(std::next(phys_handle), new_area.base, new_area);
}

}
