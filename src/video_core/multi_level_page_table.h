// SPDX-FileCopyrightText: 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#ifdef __ANDROID__
#include <atomic>
#include <android/log.h>
#endif

#include "common/object_pool.h"
#include "common/types.h"

namespace VideoCore {

template <class Traits>
class MultiLevelPageTable final {
    using Entry = typename Traits::Entry;

    static constexpr size_t AddressSpaceBits = Traits::AddressSpaceBits;
    static constexpr size_t FirstLevelBits = Traits::FirstLevelBits;
    static constexpr size_t PageBits = Traits::PageBits;
    static constexpr size_t FirstLevelShift = AddressSpaceBits - FirstLevelBits;
    static constexpr size_t SecondLevelBits = FirstLevelShift - PageBits;
    static constexpr size_t NumEntriesPerL1Page = 1ULL << SecondLevelBits;

    using L1Page = std::array<Entry, NumEntriesPerL1Page>;

public:
    explicit MultiLevelPageTable() : first_level_map{1ULL << FirstLevelBits, nullptr} {}

    ~MultiLevelPageTable() noexcept = default;

    static constexpr size_t NumL1Pages = 1ULL << FirstLevelBits;

    [[nodiscard]] Entry* find(size_t page) {
        const size_t l1_page = page >> SecondLevelBits;
        if (l1_page >= NumL1Pages) {
            return nullptr;
        }
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        if (!first_level_map[l1_page]) {
            return nullptr;
        }
        return &(*first_level_map[l1_page])[l2_page];
    }

    [[nodiscard]] const Entry* find(size_t page) const {
        const size_t l1_page = page >> SecondLevelBits;
        if (l1_page >= NumL1Pages) {
            return nullptr;
        }
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        if (!first_level_map[l1_page]) {
            return nullptr;
        }
        return &(*first_level_map[l1_page])[l2_page];
    }

    [[nodiscard]] const Entry& operator[](size_t page) const {
        const size_t l1_page = page >> SecondLevelBits;
        if (l1_page >= NumL1Pages) {
            return OutOfRangeEntry();
        }
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        if (!first_level_map[l1_page]) {
            first_level_map[l1_page] = page_alloc.Create();
        }
        return (*first_level_map[l1_page])[l2_page];
    }

    [[nodiscard]] Entry& operator[](size_t page) {
        const size_t l1_page = page >> SecondLevelBits;
        if (l1_page >= NumL1Pages) {
            return OutOfRangeEntry();
        }
        const size_t l2_page = page & (NumEntriesPerL1Page - 1);
        if (!first_level_map[l1_page]) {
            first_level_map[l1_page] = page_alloc.Create();
        }
        return (*first_level_map[l1_page])[l2_page];
    }

private:
    static Entry& OutOfRangeEntry() {
#ifdef __ANDROID__
        static std::atomic<unsigned> s_oob{0};
        const unsigned n = s_oob.fetch_add(1, std::memory_order_relaxed);
        if (n < 32) {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_PAGETABLE_OOB] n=%u (guest address >= GPU ceiling; "
                                "bounded no-op instead of ~8MB wild write)",
                                n);
        }
#endif
        static Entry dummy{};
        return dummy;
    }
    std::vector<L1Page*> first_level_map{};
    Common::ObjectPool<L1Page> page_alloc;
};

}
