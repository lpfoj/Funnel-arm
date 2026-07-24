// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/range_lock.h"
#include "common/signal_context.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include <unistd.h>
#include "common/adaptive_mutex.h"
#ifdef ENABLE_USERFAULTFD
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include "common/error.h"
#endif
#else
#include <windows.h>
#include "common/spin_lock.h"
#endif

#ifdef __linux__
#include "common/adaptive_mutex.h"
#endif

#ifdef __ANDROID__
#include <android/log.h>

extern "C" std::uint64_t executor_jit_current_fault_guest_rip();
#endif
#if !defined(__linux__) || !defined(PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP)
#include "common/spin_lock.h"
#endif

namespace VideoCore {

#ifdef __ANDROID__
namespace {

bool ExecutorReadbackContractTraceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EXECUTOR_READBACK_CONTRACT_TRACE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool ExecutorShouldLogReadbackOrdinal(const u64 ordinal) {
    return ordinal <= 64 || (ordinal & (ordinal - 1)) == 0;
}

std::atomic<u64> executor_readback_protect_count{0};
std::atomic<u64> executor_readback_fault_count{0};
std::atomic<u64> executor_readback_read_fault_count{0};
std::atomic<u64> executor_readback_write_fault_count{0};

}
#endif

constexpr size_t PAGE_SIZE = 4_KB;
constexpr size_t PAGE_BITS = 12;

#ifdef __ANDROID__
thread_local bool executor_handling_guest_fault = false;
thread_local VAddr executor_bulk_write_begin = 0;
thread_local VAddr executor_bulk_write_end = 0;
thread_local bool executor_bulk_write_consumed = false;
#endif

void BeginBulkGuestWrite(const VAddr address, const u64 size) noexcept {
#ifdef __ANDROID__
    executor_bulk_write_begin = address;
    executor_bulk_write_end =
        size <= std::numeric_limits<VAddr>::max() - address ? address + size : address;
    executor_bulk_write_consumed = false;
#else
    (void)address;
    (void)size;
#endif
}

void EndBulkGuestWrite() noexcept {
#ifdef __ANDROID__
    executor_bulk_write_begin = 0;
    executor_bulk_write_end = 0;
    executor_bulk_write_consumed = false;
#endif
}

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers : 5;
        u8 num_read_watchers : 2;
        u8 memory_wait_write_watcher : 1;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 && memory_wait_write_watcher == 0
                       ? Core::MemoryPermission::Write
                       : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <s32 delta, bool is_read>
        u8 AddDelta() {
            if constexpr (is_read) {
                if constexpr (delta == 1) {
                    return ++num_read_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                    return --num_read_watchers;
                } else {
                    return num_read_watchers;
                }
            } else {
                if constexpr (delta == 1) {
                    ASSERT_MSG(num_write_watchers < 31, "Too many write watchers");
                    return ++num_write_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                    return --num_write_watchers;
                } else {
                    return num_write_watchers;
                }
            }
        }
    };
    static_assert(sizeof(PageState) == 1,
                  "The 40-bit page-state table must remain byte-packed");

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / PAGES_PER_LOCK;
    inline static Vulkan::Rasterizer* rasterizer;
    inline static Impl* owner;
#ifdef ENABLE_USERFAULTFD
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
        owner = this;
        uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        ASSERT_MSG(uffd != -1, "{}", Common::GetLastErrorMsg());

        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        ASSERT(ret == 0 && api.api == UFFD_API);

        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    void OnMap(VAddr address, size_t size) {
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed");
    }

    void OnUnmap(VAddr address, size_t size) {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed");
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? 0 : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            ASSERT_MSG(readret != -1 || errno == EAGAIN, "Unexpected result of uffd read");
            if (errno == EAGAIN) {
                continue;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            const VAddr addr = msg.arg.pagefault.address;
            const bool wait_event = rasterizer->ConsumeMemoryWaitWriteFault(addr);
            const bool invalidated = rasterizer->InvalidateMemory(addr, 1);
            ASSERT_MSG(wait_event || invalidated,
                       "Unhandled userfaultfd write protection at {:#x}", addr);
        }
    }

    std::jthread ufd_thread;
    int uffd;
#else
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
        owner = this;

        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
#ifdef __ANDROID__
        if (ExecutorReadbackContractTraceEnabled()) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_READBACK_CONTRACT_INIT] mode=%d hostPageSize=%zu trackerPageSize=%zu",
                Config::getReadbacksMode(), static_cast<size_t>(::sysconf(_SC_PAGESIZE)),
                PAGE_SIZE);
        }
#endif
    }

    void OnMap(VAddr address, size_t size) {
        FillActualWriteProtection(address, size, false);
        FillActualReadProtection(address, size, false);
    }

    void OnUnmap(VAddr address, size_t size) {
        FillActualWriteProtection(address, size, false);
        FillActualReadProtection(address, size, false);
    }

    bool ActualReadProtected(const size_t page) const noexcept {
        const u64 word = actual_read_protection[page >> 6];
        return ((word >> (page & 63)) & 1u) != 0;
    }

    bool ActualWriteProtected(const size_t page) const noexcept {
        const u64 word = actual_write_protection[page >> 6];
        return ((word >> (page & 63)) & 1u) != 0;
    }

    bool WriteFaultBypassed(const size_t page) const noexcept {
        const u64 word = write_fault_bypass[page >> 6];
        return ((word >> (page & 63)) & 1u) != 0;
    }

    bool BulkWriteActive(const size_t page) const noexcept {
        const u64 word = active_bulk_write[page >> 6];
        return ((word >> (page & 63)) & 1u) != 0;
    }

    void SetBulkWriteActive(const size_t page, const bool enabled) noexcept {
        const u64 mask = u64{1} << (page & 63);
        if (enabled) {
            active_bulk_write[page >> 6] |= mask;
        } else {
            active_bulk_write[page >> 6] &= ~mask;
        }
    }

    void SetWriteFaultBypass(const size_t page, const bool enabled) noexcept {
        const u64 mask = u64{1} << (page & 63);
        if (enabled) {
            write_fault_bypass[page >> 6] |= mask;
        } else {
            write_fault_bypass[page >> 6] &= ~mask;
        }
    }

    void FillActualReadProtection(const VAddr address, const u64 size,
                                  const bool protected_state) noexcept {
        if (size == 0 || address >= (u64{1} << ADDRESS_BITS)) {
            return;
        }
        const u64 bounded_size = std::min(size, (u64{1} << ADDRESS_BITS) - address);
        const size_t first_page = address >> PAGE_BITS;
        const size_t end_page = Common::DivCeil(address + bounded_size, PAGE_SIZE);
        if (first_page >= end_page) {
            return;
        }
        const size_t first_word = first_page >> 6;
        const size_t last_word = (end_page - 1) >> 6;
        const u64 first_mask = ~u64{0} << (first_page & 63);
        const u64 last_mask =
            (end_page & 63) == 0 ? ~u64{0} : (u64{1} << (end_page & 63)) - 1;
        if (first_word == last_word) {
            const u64 mask = first_mask & last_mask;
            if (protected_state) {
                actual_read_protection[first_word] |= mask;
            } else {
                actual_read_protection[first_word] &= ~mask;
            }
            return;
        }
        if (protected_state) {
            actual_read_protection[first_word] |= first_mask;
            std::fill(actual_read_protection.begin() + first_word + 1,
                      actual_read_protection.begin() + last_word, ~u64{0});
            actual_read_protection[last_word] |= last_mask;
        } else {
            actual_read_protection[first_word] &= ~first_mask;
            std::fill(actual_read_protection.begin() + first_word + 1,
                      actual_read_protection.begin() + last_word, u64{0});
            actual_read_protection[last_word] &= ~last_mask;
        }
    }

    void FillActualWriteProtection(const VAddr address, const u64 size,
                                   const bool protected_state) noexcept {
        if (size == 0 || address >= (u64{1} << ADDRESS_BITS)) {
            return;
        }
        const u64 bounded_size =
            std::min(size, (u64{1} << ADDRESS_BITS) - address);
        const size_t first_page = address >> PAGE_BITS;
        const size_t end_page =
            Common::DivCeil(address + bounded_size, PAGE_SIZE);
        if (first_page >= end_page) {
            return;
        }

        const size_t first_word = first_page >> 6;
        const size_t last_word = (end_page - 1) >> 6;
        const u64 first_mask = ~u64{0} << (first_page & 63);
        const u64 last_mask =
            (end_page & 63) == 0 ? ~u64{0} : (u64{1} << (end_page & 63)) - 1;
        if (first_word == last_word) {
            const u64 mask = first_mask & last_mask;
            if (protected_state) {
                actual_write_protection[first_word] |= mask;
            } else {
                actual_write_protection[first_word] &= ~mask;
            }
            return;
        }

        if (protected_state) {
            actual_write_protection[first_word] |= first_mask;
            std::fill(actual_write_protection.begin() + first_word + 1,
                      actual_write_protection.begin() + last_word, ~u64{0});
            actual_write_protection[last_word] |= last_mask;
        } else {
            actual_write_protection[first_word] &= ~first_mask;
            std::fill(actual_write_protection.begin() + first_word + 1,
                      actual_write_protection.begin() + last_word, u64{0});
            actual_write_protection[last_word] &= ~last_mask;
        }
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        const auto apply = [&](const VAddr begin, const u64 bytes,
                               const Core::MemoryPermission actual_perms) {
            impl.Protect(begin, bytes, actual_perms);
            FillActualReadProtection(
                begin, bytes, !True(actual_perms & Core::MemoryPermission::Read));
            FillActualWriteProtection(
                begin, bytes, !True(actual_perms & Core::MemoryPermission::Write));
        };
        if (True(perms & Core::MemoryPermission::Write)) {
            apply(address, size, perms);
        } else {
            size_t page = address >> PAGE_BITS;
            const size_t page_end = Common::DivCeil(address + size, PAGE_SIZE);
            while (page < page_end) {
                const bool bypassed = WriteFaultBypassed(page);
                const size_t first = page++;
                while (page < page_end &&
                       WriteFaultBypassed(page) == bypassed) {
                    ++page;
                }
                Core::MemoryPermission actual_perms = perms;
                if (bypassed) {
                    actual_perms |= Core::MemoryPermission::Write;
                }
                apply(static_cast<VAddr>(first) << PAGE_BITS,
                      (page - first) << PAGE_BITS, actual_perms);
            }
        }
#ifdef __ANDROID__
        if (ExecutorReadbackContractTraceEnabled()) {
            const u64 ordinal =
                executor_readback_protect_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (ExecutorShouldLogReadbackOrdinal(ordinal)) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_READBACK_PROTECT] n=%llu addr=0x%llx bytes=%zu read=%u write=%u "
                    "execute=%u mode=%d",
                    static_cast<unsigned long long>(ordinal),
                    static_cast<unsigned long long>(address), size,
                    True(perms & Core::MemoryPermission::Read) ? 1u : 0u,
                    True(perms & Core::MemoryPermission::Write) ? 1u : 0u,
                    True(perms & Core::MemoryPermission::Execute) ? 1u : 0u,
                    Config::getReadbacksMode());
            }
        }
#endif
    }

    template <bool track, bool is_read>
    void ApplyWatcherProtection(const VAddr address, const u64 size,
                                const Core::MemoryPermission perms) {
        Protect(address, size, perms);
    }

    bool ReleaseDeferredWriteProtection(const VAddr address) {
        if (address >= (u64{1} << ADDRESS_BITS)) {
            return false;
        }
        const size_t page = address >> PAGE_BITS;
        std::scoped_lock lk{locks[page / PAGES_PER_LOCK]};
        PageState& state = cached_pages[page];
        if (!True(state.WritePerm() & Core::MemoryPermission::Write)) {
            return false;
        }
        Protect(static_cast<VAddr>(page) << PAGE_BITS, PAGE_SIZE, state.Perms());
        return true;
    }

    bool CommitWriteFault(const VAddr address, const bool persistent) {
        if (address >= (u64{1} << ADDRESS_BITS)) {
            return false;
        }
        (void)persistent;
        const size_t page = address >> PAGE_BITS;
        std::scoped_lock lk{locks[page / PAGES_PER_LOCK]};
        if (!ActualWriteProtected(page)) {
            return true;
        }
        const Core::MemoryPermission perms = cached_pages[page].Perms();
        if (!True(perms & Core::MemoryPermission::Write)) {
            return false;
        }
        Protect(static_cast<VAddr>(page) << PAGE_BITS, PAGE_SIZE, perms);
        return true;
    }

    void ReleaseDeferredWriteProtectionRange(const VAddr address, const u64 size) {
        if (size == 0 || address >= (u64{1} << ADDRESS_BITS)) {
            return;
        }
        const u64 bounded_size = std::min(size, (u64{1} << ADDRESS_BITS) - address);
        size_t page = address >> PAGE_BITS;
        const size_t page_end = Common::DivCeil(address + bounded_size, PAGE_SIZE);
        while (page < page_end) {
            const size_t lock_end =
                std::min(page_end, Common::AlignUp(page + 1, PAGES_PER_LOCK));
            std::scoped_lock lk{locks[page / PAGES_PER_LOCK]};
            while (page < lock_end) {
                while (page < lock_end &&
                       (!ActualWriteProtected(page) ||
                        !True(cached_pages[page].WritePerm() &
                              Core::MemoryPermission::Write))) {
                    ++page;
                }
                const size_t first = page;
                const Core::MemoryPermission perms =
                    first < lock_end ? cached_pages[first].Perms()
                                     : Core::MemoryPermission::None;
                while (page < lock_end && ActualWriteProtected(page) &&
                       True(cached_pages[page].WritePerm() &
                            Core::MemoryPermission::Write) &&
                       cached_pages[page].Perms() == perms) {
                    ++page;
                }
                if (first != page) {
                    Protect(static_cast<VAddr>(first) << PAGE_BITS,
                            (page - first) << PAGE_BITS, perms);
                }
            }
        }
    }

    void PrepareBulkGuestWrite(const VAddr address, const u64 size) {
        if (size == 0 || address >= (u64{1} << ADDRESS_BITS)) {
            return;
        }
        const u64 bounded_size = std::min(size, (u64{1} << ADDRESS_BITS) - address);
        size_t page = address >> PAGE_BITS;
        const size_t page_end = Common::DivCeil(address + bounded_size, PAGE_SIZE);
        bool protected_page = false;
        while (page < page_end) {
            if (ActualWriteProtected(page)) {
                protected_page = true;
                break;
            }
            ++page;
        }
        if (!protected_page) {
            return;
        }
        const VAddr page_begin = Common::AlignDown(address, PAGE_SIZE);
        const VAddr page_limit = Common::AlignUp(address + bounded_size, PAGE_SIZE);
        rasterizer->InvalidateMemory(page_begin, page_limit - page_begin);
        ReleaseDeferredWriteProtectionRange(page_begin, page_limit - page_begin);
    }

    void PrepareBulkGuestRead(const VAddr address, const u64 size) {
        if (size == 0 || address >= (u64{1} << ADDRESS_BITS)) {
            return;
        }
        const u64 bounded_size = std::min(size, (u64{1} << ADDRESS_BITS) - address);
        size_t page = address >> PAGE_BITS;
        const size_t page_end = Common::DivCeil(address + bounded_size, PAGE_SIZE);
        while (page < page_end && !ActualReadProtected(page)) {
            ++page;
        }
        if (page != page_end) {
            rasterizer->ReadMemory(address, bounded_size);
        }
    }

    void SetBulkGuestWriteActivity(const VAddr address, const u64 size,
                                   const bool active) {
        if (size == 0 || address >= (u64{1} << ADDRESS_BITS)) {
            return;
        }
        const u64 bounded_size = std::min(size, (u64{1} << ADDRESS_BITS) - address);
        size_t page = address >> PAGE_BITS;
        const size_t page_end = Common::DivCeil(address + bounded_size, PAGE_SIZE);
        while (page < page_end) {
            std::scoped_lock lk{locks[page / PAGES_PER_LOCK]};
            SetBulkWriteActive(page, active);
            if (active) {
                if (!WriteFaultBypassed(page)) {
                    SetWriteFaultBypass(page, true);
                    const Core::MemoryPermission perms =
                        cached_pages[page].ReadPerm() |
                        Core::MemoryPermission::Write;
                    Protect(static_cast<VAddr>(page) << PAGE_BITS, PAGE_SIZE,
                            perms);
                }
            } else if (WriteFaultBypassed(page)) {
                SetWriteFaultBypass(page, false);
                Protect(static_cast<VAddr>(page) << PAGE_BITS, PAGE_SIZE,
                        cached_pages[page].Perms());
            }
            ++page;
        }
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const bool is_write = Common::IsWriteError(context);
#ifdef __ANDROID__
        const bool previous_fault_state = executor_handling_guest_fault;
        executor_handling_guest_fault = true;
#endif
        const bool wait_event = is_write && rasterizer->ConsumeMemoryWaitWriteFault(addr);
        VAddr cache_addr = Common::AlignDown(addr, PAGE_SIZE);
        u64 cache_size = PAGE_SIZE;
#ifdef __ANDROID__
        if (is_write) {
            if (!executor_bulk_write_consumed &&
                addr >= executor_bulk_write_begin && addr < executor_bulk_write_end) {
                cache_addr = Common::AlignDown(executor_bulk_write_begin, PAGE_SIZE);
                cache_size =
                    Common::AlignUp(executor_bulk_write_end, PAGE_SIZE) - cache_addr;
                executor_bulk_write_consumed = true;
            }
        }
#endif
        const bool cache_event =
            is_write ? rasterizer->InvalidateMemory(cache_addr, cache_size)
                     : rasterizer->ReadMemory(cache_addr, cache_size);
        const bool deferred_release =
            is_write && owner != nullptr &&
            owner->ReleaseDeferredWriteProtection(addr);
        bool committed_write = false;
        if (is_write && owner != nullptr) {
            const bool persistent =
                addr >= executor_bulk_write_begin && addr < executor_bulk_write_end;
            committed_write = owner->CommitWriteFault(addr, persistent);
        }
        if (is_write && cache_size > PAGE_SIZE && owner != nullptr) {
            owner->ReleaseDeferredWriteProtectionRange(cache_addr, cache_size);
        }
        const bool handled =
            wait_event || cache_event || deferred_release || committed_write;
#ifdef __ANDROID__
        executor_handling_guest_fault = previous_fault_state;
        if (ExecutorReadbackContractTraceEnabled()) {
            const u64 ordinal =
                executor_readback_fault_count.fetch_add(1, std::memory_order_relaxed) + 1;
            const u64 typed_ordinal =
                (is_write ? executor_readback_write_fault_count
                          : executor_readback_read_fault_count)
                    .fetch_add(1, std::memory_order_relaxed) +
                1;
            if (ExecutorShouldLogReadbackOrdinal(ordinal)) {
                const u64 guest_rip = executor_jit_current_fault_guest_rip();
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_READBACK_FAULT] n=%llu typed=%llu addr=0x%llx isWrite=%u "
                    "handled=%u guestRip=0x%llx reads=%llu writes=%llu",
                    static_cast<unsigned long long>(ordinal),
                    static_cast<unsigned long long>(typed_ordinal),
                    static_cast<unsigned long long>(addr), is_write ? 1u : 0u,
                    handled ? 1u : 0u,
                    static_cast<unsigned long long>(guest_rip),
                    static_cast<unsigned long long>(
                        executor_readback_read_fault_count.load(std::memory_order_relaxed)),
                    static_cast<unsigned long long>(
                        executor_readback_write_fault_count.load(std::memory_order_relaxed)));
            }
        }
#endif
        return handled;
    }
#endif

    ~Impl() {
        if (owner == this) {
            owner = nullptr;
        }
    }

    template <bool track, bool is_read>
    void UpdatePageWatchers(VAddr addr, u64 size) {
        RENDERER_TRACE;

        size_t page = addr >> PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PAGE_SIZE);

        const auto lock_start = locks.begin() + (page / PAGES_PER_LOCK);
        const auto lock_end = locks.begin() + Common::DivCeil(page_end, PAGES_PER_LOCK);
        Common::RangeLockGuard lk(lock_start, lock_end);

        auto perms = cached_pages[page].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                ApplyWatcherProtection<track, is_read>(
                    range_begin << PAGE_BITS, range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        const u64 aligned_addr = page << PAGE_BITS;
        const u64 aligned_end = page_end << PAGE_BITS;
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
        }

        for (; page != page_end; ++page) {
            PageState& state = cached_pages[page];

            const u8 new_count = state.AddDelta<track ? 1 : -1, is_read>();
            if constexpr (track && !is_read) {
                if (new_count == 1 && WriteFaultBypassed(page) &&
                    !BulkWriteActive(page)) {
                    SetWriteFaultBypass(page, false);
                }
            }

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                potential_range_bytes += PAGE_SIZE;
            }

            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    range_begin = page;
                    potential_range_bytes = PAGE_SIZE;
                }
                range_bytes = potential_range_bytes;
            }
        }

        release_pending();
    }

    template <bool track, bool is_read>
    void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) {
        RENDERER_TRACE;
        auto start_range = mask.FirstRange();
        auto end_range = mask.LastRange();

        if (start_range.second == end_range.second) {
            const VAddr start_addr = base_addr + (start_range.first << PAGE_BITS);
            const u64 size = (start_range.second - start_range.first) << PAGE_BITS;
            return UpdatePageWatchers<track, is_read>(start_addr, size);
        }

        size_t base_page = (base_addr >> PAGE_BITS);
        ASSERT(base_page % PAGES_PER_LOCK == 0);
        std::scoped_lock lk(locks[base_page / PAGES_PER_LOCK]);
        auto perms = cached_pages[base_page + start_range.first].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                ApplyWatcherProtection<track, is_read>(
                    range_begin << PAGE_BITS, range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        for (size_t page = start_range.first; page < end_range.second; ++page) {
            PageState& state = cached_pages[base_page + page];
            const bool update = mask.Get(page);

            const u8 new_count =
                update ? state.AddDelta<track ? 1 : -1, is_read>() : state.AddDelta<0, is_read>();
            if constexpr (track && !is_read) {
                const size_t absolute_page = base_page + page;
                if (update && new_count == 1 &&
                    WriteFaultBypassed(absolute_page) &&
                    !BulkWriteActive(absolute_page)) {
                    SetWriteFaultBypass(absolute_page, false);
                }
            }

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                potential_range_bytes += PAGE_SIZE;
            }

            if (!update) {
                continue;
            }

            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    range_begin = base_page + page;
                    potential_range_bytes = PAGE_SIZE;
                }
                range_bytes = potential_range_bytes;
            }
        }

        release_pending();
    }

    template <bool track>
    void UpdateMemoryWaitWriteWatchers(VAddr addr, u64 size) {
        if (size == 0 || addr >= (u64{1} << ADDRESS_BITS) ||
            size > (u64{1} << ADDRESS_BITS) - addr) {
            return;
        }
        size_t page = addr >> PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PAGE_SIZE);
        const auto lock_start = locks.begin() + (page / PAGES_PER_LOCK);
        const auto lock_end = locks.begin() + Common::DivCeil(page_end, PAGES_PER_LOCK);
        Common::RangeLockGuard lk(lock_start, lock_end);

        for (; page != page_end; ++page) {
            PageState& state = cached_pages[page];
            const auto old_perms = state.Perms();
            const u8 desired = track ? 1u : 0u;
            if (state.memory_wait_write_watcher == desired) {
                continue;
            }
            if constexpr (track) {
                active_memory_wait_write_watchers.fetch_add(1, std::memory_order_release);
            }
            state.memory_wait_write_watcher = desired;
            const auto new_perms = state.Perms();
            if (new_perms != old_perms) {
                Protect(static_cast<VAddr>(page) << PAGE_BITS, PAGE_SIZE, new_perms);
            }
            if constexpr (!track) {
                active_memory_wait_write_watchers.fetch_sub(1, std::memory_order_release);
            }
        }
    }

    bool ConsumeMemoryWaitWriteFault(VAddr addr) {
        if (addr >= (u64{1} << ADDRESS_BITS) ||
            active_memory_wait_write_watchers.load(std::memory_order_acquire) == 0) {
            return false;
        }
        const size_t page = addr >> PAGE_BITS;
        std::scoped_lock lk{locks[page / PAGES_PER_LOCK]};
        PageState& state = cached_pages[page];
        if (state.memory_wait_write_watcher == 0) {
            return false;
        }
        const auto old_perms = state.Perms();
        state.memory_wait_write_watcher = 0;
        const auto new_perms = state.Perms();
        if (new_perms != old_perms) {
            Protect(static_cast<VAddr>(page) << PAGE_BITS, PAGE_SIZE, new_perms);
        }
        active_memory_wait_write_watchers.fetch_sub(1, std::memory_order_release);
        return true;
    }

    std::atomic<u32> active_memory_wait_write_watchers{0};
    std::array<PageState, NUM_ADDRESS_PAGES> cached_pages{};
    std::array<u64, NUM_ADDRESS_PAGES / 64> actual_read_protection{};
    std::array<u64, NUM_ADDRESS_PAGES / 64> actual_write_protection{};
    std::array<u64, NUM_ADDRESS_PAGES / 64> write_fault_bypass{};
    std::array<u64, NUM_ADDRESS_PAGES / 64> active_bulk_write{};
#ifdef __ANDROID__
    using LockType = Common::ParkingMutex;
#elif defined(PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP)
    using LockType = Common::AdaptiveMutex;
#else
    using LockType = Common::SpinLock;
#endif
    std::array<LockType, NUM_ADDRESS_LOCKS> locks{};
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_)
    : impl{std::make_unique<Impl>(rasterizer_)} {}

PageManager::~PageManager() = default;

void PageManager::PrepareBulkGuestRead(const VAddr address, const u64 size) {
    if (Impl::owner != nullptr) {
        Impl::owner->PrepareBulkGuestRead(address, size);
    }
}

void PageManager::PrepareBulkGuestWrite(const VAddr address, const u64 size) {
    if (Impl::owner != nullptr) {
        Impl::owner->PrepareBulkGuestWrite(address, size);
    }
}

void PageManager::EnterBulkGuestWrite(const VAddr address, const u64 size) {
    if (Impl::owner != nullptr) {
        Impl::owner->SetBulkGuestWriteActivity(address, size, true);
    }
}

void PageManager::LeaveBulkGuestWrite(const VAddr address, const u64 size) {
    if (Impl::owner != nullptr) {
        Impl::owner->SetBulkGuestWriteActivity(address, size, false);
    }
}

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::TrackPendingCompletionRead(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<true, true>(addr, size);
}

void PageManager::UntrackPendingCompletionRead(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<false, true>(addr, size);
}

void PageManager::TrackMemoryWaitWrite(VAddr addr, u64 size) const {
    impl->UpdateMemoryWaitWriteWatchers<true>(addr, size);
}

void PageManager::UntrackMemoryWaitWrite(VAddr addr, u64 size) const {
    impl->UpdateMemoryWaitWriteWatchers<false>(addr, size);
}

bool PageManager::ConsumeMemoryWaitWriteFault(VAddr addr) const {
    return impl->ConsumeMemoryWaitWriteFault(addr);
}

template <bool track>
void PageManager::UpdatePageWatchers(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<track, false>(addr, size);
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const {
    impl->UpdatePageWatchersForRegion<track, is_read>(base_addr, mask);
}

template void PageManager::UpdatePageWatchers<true>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchers<false>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchersForRegion<true, true>(VAddr base_addr,
                                                                   RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<true, false>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, true>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, false>(VAddr base_addr,
                                                                     RegionBits& mask) const;

}
