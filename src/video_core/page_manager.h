// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include "common/alignment.h"
#include "common/types.h"
#include "video_core/buffer_cache//region_definitions.h"

namespace Vulkan {
class Rasterizer;
}

namespace VideoCore {

void BeginBulkGuestWrite(VAddr address, u64 size) noexcept;
void EndBulkGuestWrite() noexcept;

class PageManager {
    static constexpr size_t PAGE_BITS = TRACKER_PAGE_BITS;
    static constexpr size_t PAGE_BYTES = TRACKER_BYTES_PER_PAGE;

    static constexpr size_t PAGES_PER_LOCK = NUM_PAGES_PER_REGION;

public:
    explicit PageManager(Vulkan::Rasterizer* rasterizer);
    ~PageManager();

    static void PrepareBulkGuestRead(VAddr address, u64 size);
    static void PrepareBulkGuestWrite(VAddr address, u64 size);
    static void EnterBulkGuestWrite(VAddr address, u64 size);
    static void LeaveBulkGuestWrite(VAddr address, u64 size);

    void OnGpuMap(VAddr address, size_t size);

    void OnGpuUnmap(VAddr address, size_t size);

    template <bool track>
    void UpdatePageWatchers(VAddr addr, u64 size) const;

    void TrackPendingCompletionRead(VAddr addr, u64 size) const;
    void UntrackPendingCompletionRead(VAddr addr, u64 size) const;

    void TrackMemoryWaitWrite(VAddr addr, u64 size) const;
    void UntrackMemoryWaitWrite(VAddr addr, u64 size) const;
    [[nodiscard]] bool ConsumeMemoryWaitWriteFault(VAddr addr) const;

    template <bool track, bool is_read = false>
    void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const;

    static constexpr VAddr GetPageAddr(VAddr addr) {
        return Common::AlignDown(addr, PAGE_BYTES);
    }

    static constexpr VAddr GetNextPageAddr(VAddr addr) {
        return Common::AlignUp(addr + 1, PAGE_BYTES);
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}
