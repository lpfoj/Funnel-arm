// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>
#include <cstdlib>
#include <magic_enum/magic_enum.hpp>

#ifdef __ANDROID__
#include <android/log.h>
#include <atomic>
#endif

#include "common/assert.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "core/file_sys/fs.h"
#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/libs.h"
#ifdef __ANDROID__
#include "video_core/amdgpu/liverpool.h"
#endif

namespace Libraries::Kernel {

extern boost::asio::io_context io_context;
extern void KernelSignalRequest();

static std::unordered_map<s32, EqueueInternal*> kqueues;
static constexpr auto HrTimerSpinlockThresholdNs = 1200000u;

#ifdef __ANDROID__
static bool ExecutorTraceHotEqueueWait() {
    static const bool enabled = [] {
        const char* value = std::getenv("EXECUTOR_TRACE_EQUEUE_WAIT");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool ShouldTraceLiveEqueue() {
    if (std::getenv("EXECUTOR_TRACE_LIVE_WIDE") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_SYNC") == nullptr &&
        std::getenv("EXECUTOR_LIGHT_ORACLE") == nullptr) {
        return false;
    }
    if (!g_curthread) {
        return false;
    }
    const auto& name = g_curthread->name;
    return name.find("Game:Main") != std::string::npos ||
           name.find("mono") != std::string::npos ||
           name.find("UnityPreload") != std::string::npos ||
           name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("Submit Done Thread") != std::string::npos;
}

static const char* CurrentLiveThreadName() {
    return g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>";
}

static void TraceLiveEqueue(const char* op, OrbisKernelEqueue eq, u64 ident, s16 filter, u64 data,
                            void* udata, s32 rc) {
    if (!ShouldTraceLiveEqueue()) {
        return;
    }
    static std::atomic_int budget{4096};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_EQUEUE] op=%s thread=%s eq=%ld ident=0x%llx "
                        "filter=%d data=0x%llx udata=%p rc=0x%x",
                        op, CurrentLiveThreadName(), eq,
                        static_cast<unsigned long long>(ident), filter,
                        static_cast<unsigned long long>(data), udata, static_cast<u32>(rc));
}
#endif

EqueueInternal* GetEqueue(OrbisKernelEqueue eq) {
    if (!kqueues.contains(eq)) {
        return nullptr;
    }
    return kqueues[eq];
}

static void HrTimerCallback(OrbisKernelEqueue eq, const OrbisKernelEvent& kevent) {
    if (kqueues.contains(eq)) {
        kqueues[eq]->TriggerEvent(kevent.ident, OrbisKernelEvent::Filter::HrTimer, kevent.udata);
    }
}

static void TimerCallback(OrbisKernelEqueue eq, const OrbisKernelEvent& kevent) {
    if (kqueues.contains(eq) && kqueues[eq]->EventExists(kevent.ident, kevent.filter)) {
        kqueues[eq]->TriggerEvent(kevent.ident, OrbisKernelEvent::Filter::Timer, kevent.udata);
        if (!(kevent.flags & OrbisKernelEvent::Flags::OneShot)) {
            kqueues[eq]->ScheduleEvent(kevent.ident, kevent.filter, TimerCallback);
        }
    }
}

bool EqueueInternal::AddEvent(EqueueEvent& event) {
    const u64 id = event.event.ident;
    const auto filter = event.event.filter;
    {
        std::scoped_lock lock{m_mutex};

        event.time_added = std::chrono::steady_clock::now();
        if (event.event.filter == OrbisKernelEvent::Filter::Timer) {
            event.timer_interval = std::chrono::milliseconds(event.event.data);
        } else if (event.event.filter == OrbisKernelEvent::Filter::HrTimer) {
            OrbisKernelBintime* time = reinterpret_cast<OrbisKernelBintime*>(event.event.data);

            OrbisKernelTimespec ts;
            ts.tv_sec = time->sec;
            ts.tv_nsec = (1000000000 * (time->frac >> 32)) >> 32;

            event.timer_interval = std::chrono::nanoseconds(ts.tv_nsec + ts.tv_sec * 1000000000);
        }

        const auto& find_it = std::ranges::find_if(m_events, [id, filter](auto& ev) {
            return ev.event.ident == id && ev.event.filter == filter;
        });
        if (find_it != m_events.cend()) {
            auto& old_event = *find_it;
            old_event.timer_interval = event.timer_interval;
            old_event.event.udata = event.event.udata;
            return true;
        }

        event.event.data = 0;

        event.event.flags &= ~OrbisKernelEvent::Flags::Add;

        if (event.event.filter != OrbisKernelEvent::Filter::User) {
            event.event.flags |= OrbisKernelEvent::Flags::Clear;
        }

        const auto& it = std::ranges::find(m_events, event);
        if (it != m_events.cend()) {
            *it = std::move(event);
        } else {
            m_events.emplace_back(std::move(event));
        }
    }

    if (filter == OrbisKernelEvent::Timer) {
        return this->ScheduleEvent(id, OrbisKernelEvent::Filter::Timer, TimerCallback);
    } else if (filter == OrbisKernelEvent::HrTimer) {
        return this->ScheduleEvent(id, OrbisKernelEvent::Filter::HrTimer, HrTimerCallback);
    }

    return true;
}

bool EqueueInternal::ScheduleEvent(u64 id, s16 filter,
                                   void (*callback)(OrbisKernelEqueue, const OrbisKernelEvent&)) {
    std::scoped_lock lock{m_mutex};

    const auto& it = std::ranges::find_if(m_events, [id, filter](auto& ev) {
        return ev.event.ident == id && ev.event.filter == filter;
    });
    if (it == m_events.cend()) {
        return false;
    }

    const auto& event = *it;
    ASSERT(event.event.filter == OrbisKernelEvent::Filter::Timer ||
           event.event.filter == OrbisKernelEvent::Filter::HrTimer);

    if (!it->timer) {
        it->timer = std::make_unique<boost::asio::steady_timer>(io_context, event.timer_interval);
    } else {
        it->timer->expires_at(it->timer->expiry() + event.timer_interval);
    }

    it->timer->async_wait(
        [this, event_data = event.event, callback](const boost::system::error_code& ec) {
            if (ec) {
                if (ec != boost::system::errc::operation_canceled) {
                    LOG_ERROR(Kernel_Event, "Timer callback error: {}", ec.message());
                } else {
                    LOG_DEBUG(Kernel_Event, "Timer cancelled");
                }
                return;
            }
            callback(this->m_handle, event_data);
        });
    KernelSignalRequest();

    return true;
}

bool EqueueInternal::RemoveEvent(u64 id, s16 filter) {
    bool has_found = false;
    std::scoped_lock lock{m_mutex};

    const auto& it = std::ranges::find_if(m_events, [id, filter](auto& ev) {
        return ev.event.ident == id && ev.event.filter == filter;
    });
    if (it != m_events.cend()) {
        m_events.erase(it);
        has_found = true;
    }
    return has_found;
}

int EqueueInternal::WaitForEvents(OrbisKernelEvent* ev, int num, const OrbisKernelUseconds* timo) {
    if (timo != nullptr && *timo == 0) {
        return GetTriggeredEvents(ev, num);
    }
    const auto micros = timo ? *timo : 0u;

    if (HasSmallTimer()) {
        return WaitForSmallTimer(ev, num, micros);
    }

    int count = 0;
#ifdef __ANDROID__
    const bool is_gfx_eop_queue = m_name.find("EOP") != std::string::npos;
    if (is_gfx_eop_queue) {
        AmdGpu::ExecutorEopTraceEqWait(true, m_handle, 0, 0,
                                      OrbisKernelEvent::Filter::GraphicsCore);
    }
#endif

    const auto predicate = [&] {
        count = GetTriggeredEvents(ev, num);
        return count > 0;
    };

    if (micros == 0) {
        std::unique_lock lock{m_mutex};
        m_cond.wait(lock, predicate);
    } else {
        std::unique_lock lock{m_mutex};
        m_cond.wait_for(lock, std::chrono::microseconds(micros), predicate);
    }

#ifdef __ANDROID__
    if (is_gfx_eop_queue) {
        const u64 event_id = count > 0 ? ev[0].ident : 0;
        const s16 event_filter = count > 0 ? ev[0].filter : OrbisKernelEvent::Filter::None;
        AmdGpu::ExecutorEopTraceEqWait(false, m_handle, count, event_id, event_filter);
    }
#endif
    return count;
}

bool EqueueInternal::TriggerEvent(u64 ident, s16 filter, void* trigger_data) {
    bool has_found = false;
    {
        std::scoped_lock lock{m_mutex};
        for (auto& event : m_events) {
            if (event.event.ident == ident && event.event.filter == filter) {
                if (filter == OrbisKernelEvent::Filter::VideoOut) {
                    event.TriggerDisplay(trigger_data);
                } else if (filter == OrbisKernelEvent::Filter::User) {
                    event.TriggerUser(trigger_data);
                } else if (filter == OrbisKernelEvent::Filter::Timer ||
                           filter == OrbisKernelEvent::Filter::HrTimer) {
                    event.TriggerTimer();
                } else {
                    event.Trigger(trigger_data);
                }
                has_found = true;
            }
        }
    }
    m_cond.notify_one();
#ifdef __ANDROID__
    TraceLiveEqueue(has_found ? "trigger" : "trigger_miss", m_handle, ident, filter,
                    reinterpret_cast<u64>(trigger_data), trigger_data, has_found ? ORBIS_OK : -1);
#endif
    return has_found;
}

int EqueueInternal::GetTriggeredEvents(OrbisKernelEvent* ev, int num) {
    int count = 0;
    for (auto it = m_events.begin(); it != m_events.end();) {
        if (it->IsTriggered()) {
            ev[count++] = it->event;
            if (it->event.flags & OrbisKernelEvent::Flags::Clear) {
                it->Clear();
            }
            if (it->event.flags & OrbisKernelEvent::Flags::OneShot) {
                it = m_events.erase(it);
            } else {
                ++it;
            }

            if (count == num) {
                break;
            }
        } else {
            ++it;
        }
    }

    return count;
}

bool EqueueInternal::AddSmallTimer(EqueueEvent& ev) {
    OrbisKernelBintime* time = reinterpret_cast<OrbisKernelBintime*>(ev.event.data);
    OrbisKernelTimespec ts;
    ts.tv_sec = time->sec;
    ts.tv_nsec = ((1000000000 * (time->frac >> 32)) >> 32);

    SmallTimer st;
    st.event = ev.event;
    st.added = std::chrono::steady_clock::now();
    st.interval = std::chrono::nanoseconds(ts.tv_nsec + ts.tv_sec * 1000000000);
    {
        std::scoped_lock lock{m_mutex};
        m_small_timers[st.event.ident] = std::move(st);
    }
    return true;
}

int EqueueInternal::WaitForSmallTimer(OrbisKernelEvent* ev, int num, u32 micros) {
    ASSERT(num >= 1);

    auto curr_clock = std::chrono::steady_clock::now();
    const auto wait_end_us = (micros == 0) ? std::chrono::steady_clock::time_point::max()
                                           : curr_clock + std::chrono::microseconds{micros};
    int count = 0;
    do {
        curr_clock = std::chrono::steady_clock::now();
        {
            std::scoped_lock lock{m_mutex};
            for (auto it = m_small_timers.begin(); it != m_small_timers.end() && count < num;) {
                const SmallTimer& st = it->second;

                if (curr_clock - st.added >= st.interval) {
                    ev[count++] = st.event;
                    it = m_small_timers.erase(it);
                } else {
                    ++it;
                }
            }

            if (count > 0)
                return count;
        }
        std::this_thread::yield();
    } while (curr_clock < wait_end_us);

    return 0;
}

bool EqueueInternal::EventExists(u64 id, s16 filter) {
    std::scoped_lock lock{m_mutex};

    const auto& it = std::ranges::find_if(m_events, [id, filter](auto& ev) {
        return ev.event.ident == id && ev.event.filter == filter;
    });

    return it != m_events.cend();
}

s32 PS4_SYSV_ABI posix_kqueue() {
    auto* handles = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    s32 kqueue_handle = handles->CreateHandle();
    auto* kqueue_file = handles->GetFile(kqueue_handle);
    kqueue_file->type = Core::FileSys::FileType::Equeue;

    char name[32];
    memset(name, 0, sizeof(name));
    snprintf(name, sizeof(name), "kqueue%i", kqueue_handle);

    kqueues[kqueue_handle] = new EqueueInternal(kqueue_handle, name);
    LOG_INFO(Kernel_Event, "kqueue created with name {}", name);

    return kqueue_handle;
}

bool SupportedEqueueFilter(OrbisKernelEvent::Filter filter) {
    return filter == OrbisKernelEvent::Filter::GraphicsCore ||
           filter == OrbisKernelEvent::Filter::HrTimer ||
           filter == OrbisKernelEvent::Filter::Timer || filter == OrbisKernelEvent::Filter::User ||
           filter == OrbisKernelEvent::Filter::VideoOut;
}

s32 PS4_SYSV_ABI posix_kevent(s32 handle, OrbisKernelEvent* changelist, u64 nchanges,
                              OrbisKernelEvent* eventlist, u64 nevents,
                              OrbisKernelTimespec* timeout) {
    LOG_INFO(Kernel_Event, "called, eq = {}, nchanges = {}, nevents = {}", handle, nchanges,
             nevents);

    if (!kqueues.contains(handle)) {
        *__Error() = POSIX_EBADF;
        return ORBIS_FAIL;
    }
    auto equeue = kqueues[handle];

    for (u64 i = 0; i < nchanges; i++) {
        auto event = changelist[i];
        if (!SupportedEqueueFilter(event.filter)) {
            LOG_ERROR(Kernel_Event, "Unsupported event filter {}",
                      magic_enum::enum_name(event.filter));
            continue;
        }

        if (event.flags & OrbisKernelEvent::Flags::Add) {
            EqueueEvent internal_event{};
            internal_event.event = event;
            if (!equeue->AddEvent(internal_event)) {
                *__Error() = POSIX_ENOMEM;
                return ORBIS_FAIL;
            }
#ifdef __ANDROID__
            TraceLiveEqueue("kevent_add", handle, event.ident, event.filter, event.data,
                            event.udata, ORBIS_OK);
#endif
        }

        if (event.flags & OrbisKernelEvent::Flags::Delete) {
            if (!equeue->RemoveEvent(event.ident, event.filter)) {
                *__Error() = POSIX_ENOENT;
                return ORBIS_FAIL;
            }
#ifdef __ANDROID__
            TraceLiveEqueue("kevent_delete", handle, event.ident, event.filter, event.data,
                            event.udata, ORBIS_OK);
#endif
        }

        if (event.filter == OrbisKernelEvent::Filter::User && event.fflags == 0x1000000) {
            if (!equeue->TriggerEvent(event.ident, OrbisKernelEvent::Filter::User, event.udata)) {
                *__Error() = POSIX_ENOENT;
                return ORBIS_FAIL;
            }
        } else if (event.fflags != 0) {
            LOG_ERROR(Kernel_Event, "Unhandled fflags {:#x} for event filter {}", event.fflags,
                      magic_enum::enum_name(event.filter));
            continue;
        }
    }

    s32 count = 0;
    if (nevents > 0) {
        if (timeout != nullptr) {
            OrbisKernelUseconds micros = (timeout->tv_sec * 1000000) + (timeout->tv_nsec / 1000);
            count = equeue->WaitForEvents(eventlist, nevents, &micros);
        } else {
            count = equeue->WaitForEvents(eventlist, nevents, nullptr);
        }
    }
    return count;
}

int PS4_SYSV_ABI sceKernelCreateEqueue(OrbisKernelEqueue* eq, const char* name) {
    if (eq == nullptr) {
        LOG_ERROR(Kernel_Event, "Event queue is null!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    if (name == nullptr) {
        LOG_ERROR(Kernel_Event, "Event queue name is null!");
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    static constexpr u64 MaxEventQueueNameSize = 32;
    if (std::strlen(name) > MaxEventQueueNameSize) {
        LOG_ERROR(Kernel_Event, "Event queue name exceeds 32 bytes!");
        return ORBIS_KERNEL_ERROR_ENAMETOOLONG;
    }

    LOG_INFO(Kernel_Event, "name = {}", name);

    auto* handles = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    OrbisKernelEqueue kqueue_handle = handles->CreateHandle();
    auto* kqueue_file = handles->GetFile(kqueue_handle);
    kqueue_file->type = Core::FileSys::FileType::Equeue;

    kqueues[kqueue_handle] = new EqueueInternal(kqueue_handle, name);
    *eq = kqueue_handle;
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_EQUEUE_CREATE] eq=%ld name=%s", kqueue_handle, name);
#endif

    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelDeleteEqueue(OrbisKernelEqueue eq) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    auto* handles = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    handles->DeleteHandle(eq);
    kqueues.erase(eq);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelWaitEqueue(OrbisKernelEqueue eq, OrbisKernelEvent* ev, int num, int* out,
                                     OrbisKernelUseconds* timo) {
    HLE_TRACE;
#ifdef __ANDROID__
    if (ExecutorTraceHotEqueueWait()) {
        const long long timeout = timo ? static_cast<long long>(*timo) : -1;
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_EQUEUE_WAIT_ENTER] eq=%ld ev=%p num=%d out=%p timeout=%lld", eq,
            ev, num, out, timeout);
    }
#endif
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    auto& equeue = kqueues[eq];

    TRACE_HINT(equeue->GetName());
    LOG_TRACE(Kernel_Event, "equeue = {} num = {}", equeue->GetName(), num);

    if (ev == nullptr) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }

    if (num < 1) {
        *out = 0;
        return ORBIS_KERNEL_ERROR_EINVAL;
    }

    *out = equeue->WaitForEvents(ev, num, timo);

    if (*out == 0) {
#ifdef __ANDROID__
        if (ExecutorTraceHotEqueueWait()) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_EQUEUE_WAIT_EXIT] eq=%ld out=0 rc=0x%x", eq,
                                static_cast<u32>(ORBIS_KERNEL_ERROR_ETIMEDOUT));
        }
#endif
        return ORBIS_KERNEL_ERROR_ETIMEDOUT;
    }

#ifdef __ANDROID__
    if (ExecutorTraceHotEqueueWait()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_EQUEUE_WAIT_EXIT] eq=%ld out=%d ident=0x%llx filter=%d data=0x%llx rc=0",
            eq, *out, static_cast<unsigned long long>(ev[0].ident), ev[0].filter,
            static_cast<unsigned long long>(ev[0].data));
    }
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelAddHRTimerEvent(OrbisKernelEqueue eq, int id, OrbisKernelTimespec* ts,
                                          void* udata) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    const auto total_ns = ts->tv_sec * 1000000000 + ts->tv_nsec;

    EqueueEvent event{};
    event.event.ident = id;
    event.event.filter = OrbisKernelEvent::Filter::HrTimer;
    event.event.flags = OrbisKernelEvent::Flags::Add | OrbisKernelEvent::Flags::OneShot;
    event.event.fflags = 0;
    OrbisKernelBintime time{ts->tv_sec, ts->tv_nsec * 0x44b82fa09};
    event.event.data = reinterpret_cast<u64>(&time);
    event.event.udata = udata;

    auto& equeue = kqueues[eq];
    if (total_ns < HrTimerSpinlockThresholdNs) {
        return equeue->AddSmallTimer(event) ? ORBIS_OK : ORBIS_KERNEL_ERROR_ENOMEM;
    }

    if (!equeue->AddEvent(event)) {
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }
#ifdef __ANDROID__
    TraceLiveEqueue("add_hrtimer", eq, event.event.ident, event.event.filter, total_ns, udata,
                    ORBIS_OK);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelDeleteHRTimerEvent(OrbisKernelEqueue eq, int id) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    auto& equeue = kqueues[eq];
    if (equeue->HasSmallTimer()) {
        return equeue->RemoveSmallTimer(id) ? ORBIS_OK : ORBIS_KERNEL_ERROR_ENOENT;
    } else {
        return equeue->RemoveEvent(id, OrbisKernelEvent::Filter::HrTimer)
                   ? ORBIS_OK
                   : ORBIS_KERNEL_ERROR_ENOENT;
    }
}

int PS4_SYSV_ABI sceKernelAddTimerEvent(OrbisKernelEqueue eq, int id, OrbisKernelUseconds usec,
                                        void* udata) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    EqueueEvent event{};
    event.event.ident = static_cast<u64>(id);
    event.event.filter = OrbisKernelEvent::Filter::Timer;
    event.event.flags = OrbisKernelEvent::Flags::Add;
    event.event.fflags = 0;
    event.event.data = usec / 1000;
    event.event.udata = udata;

    auto& equeue = kqueues[eq];
    if (!equeue->AddEvent(event)) {
        return ORBIS_KERNEL_ERROR_ENOMEM;
    }
#ifdef __ANDROID__
    TraceLiveEqueue("add_timer", eq, event.event.ident, event.event.filter, usec, udata,
                    ORBIS_OK);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelDeleteTimerEvent(OrbisKernelEqueue eq, int id) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    return kqueues[eq]->RemoveEvent(id, OrbisKernelEvent::Filter::Timer)
               ? ORBIS_OK
               : ORBIS_KERNEL_ERROR_ENOENT;
}

int PS4_SYSV_ABI sceKernelAddUserEvent(OrbisKernelEqueue eq, int id) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    EqueueEvent event{};
    event.event.ident = id;
    event.event.filter = OrbisKernelEvent::Filter::User;
    event.event.udata = 0;
    event.event.flags = OrbisKernelEvent::Flags::Add;
    event.event.fflags = 0;
    event.event.data = 0;

    const int rc = kqueues[eq]->AddEvent(event) ? ORBIS_OK : ORBIS_KERNEL_ERROR_ENOMEM;
#ifdef __ANDROID__
    TraceLiveEqueue("add_user", eq, event.event.ident, event.event.filter, event.event.data,
                    event.event.udata, rc);
#endif
    return rc;
}

int PS4_SYSV_ABI sceKernelAddUserEventEdge(OrbisKernelEqueue eq, int id) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    EqueueEvent event{};
    event.event.ident = id;
    event.event.filter = OrbisKernelEvent::Filter::User;
    event.event.udata = 0;
    event.event.flags = OrbisKernelEvent::Flags::Add | OrbisKernelEvent::Flags::Clear;
    event.event.fflags = 0;
    event.event.data = 0;

    const int rc = kqueues[eq]->AddEvent(event) ? ORBIS_OK : ORBIS_KERNEL_ERROR_ENOMEM;
#ifdef __ANDROID__
    TraceLiveEqueue("add_user_edge", eq, event.event.ident, event.event.filter,
                    event.event.data, event.event.udata, rc);
#endif
    return rc;
}

void* PS4_SYSV_ABI sceKernelGetEventUserData(const OrbisKernelEvent* ev) {
    ASSERT(ev);
    return ev->udata;
}

u64 PS4_SYSV_ABI sceKernelGetEventId(const OrbisKernelEvent* ev) {
    return ev->ident;
}

int PS4_SYSV_ABI sceKernelTriggerUserEvent(OrbisKernelEqueue eq, int id, void* udata) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    if (!kqueues[eq]->TriggerEvent(id, OrbisKernelEvent::Filter::User, udata)) {
#ifdef __ANDROID__
        TraceLiveEqueue("trigger_user_miss", eq, id, OrbisKernelEvent::Filter::User, 0, udata,
                        ORBIS_KERNEL_ERROR_ENOENT);
#endif
        return ORBIS_KERNEL_ERROR_ENOENT;
    }
#ifdef __ANDROID__
    TraceLiveEqueue("trigger_user", eq, id, OrbisKernelEvent::Filter::User, 0, udata, ORBIS_OK);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelDeleteUserEvent(OrbisKernelEqueue eq, int id) {
    if (!kqueues.contains(eq)) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    if (!kqueues[eq]->RemoveEvent(id, OrbisKernelEvent::Filter::User)) {
        return ORBIS_KERNEL_ERROR_ENOENT;
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceKernelGetEventFilter(const OrbisKernelEvent* ev) {
    return ev->filter;
}

u64 PS4_SYSV_ABI sceKernelGetEventData(const OrbisKernelEvent* ev) {
    return ev->data;
}

void RegisterEventQueue(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("nh2IFMgKTv8", "libScePosix", 1, "libkernel", posix_kqueue);
    LIB_FUNCTION("RW-GEfpnsqg", "libScePosix", 1, "libkernel", posix_kevent);
    LIB_FUNCTION("D0OdFMjp46I", "libkernel", 1, "libkernel", sceKernelCreateEqueue);
    LIB_FUNCTION("jpFjmgAC5AE", "libkernel", 1, "libkernel", sceKernelDeleteEqueue);
    LIB_FUNCTION("fzyMKs9kim0", "libkernel", 1, "libkernel", sceKernelWaitEqueue);
    LIB_FUNCTION("vz+pg2zdopI", "libkernel", 1, "libkernel", sceKernelGetEventUserData);
    LIB_FUNCTION("4R6-OvI2cEA", "libkernel", 1, "libkernel", sceKernelAddUserEvent);
    LIB_FUNCTION("WDszmSbWuDk", "libkernel", 1, "libkernel", sceKernelAddUserEventEdge);
    LIB_FUNCTION("R74tt43xP6k", "libkernel", 1, "libkernel", sceKernelAddHRTimerEvent);
    LIB_FUNCTION("J+LF6LwObXU", "libkernel", 1, "libkernel", sceKernelDeleteHRTimerEvent);
    LIB_FUNCTION("57ZK+ODEXWY", "libkernel", 1, "libkernel", sceKernelAddTimerEvent);
    LIB_FUNCTION("YWQFUyXIVdU", "libkernel", 1, "libkernel", sceKernelDeleteTimerEvent);
    LIB_FUNCTION("F6e0kwo4cnk", "libkernel", 1, "libkernel", sceKernelTriggerUserEvent);
    LIB_FUNCTION("LJDwdSNTnDg", "libkernel", 1, "libkernel", sceKernelDeleteUserEvent);
    LIB_FUNCTION("mJ7aghmgvfc", "libkernel", 1, "libkernel", sceKernelGetEventId);
    LIB_FUNCTION("23CPPI1tyBY", "libkernel", 1, "libkernel", sceKernelGetEventFilter);
    LIB_FUNCTION("kwGyyjohI50", "libkernel", 1, "libkernel", sceKernelGetEventData);
}

}
