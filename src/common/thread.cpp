// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>

#include "core/libraries/kernel/threads/pthread.h"

#include "common/error.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "ntapi.h"
#ifdef __ANDROID__
#include <android/log.h>
#endif
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
#include "common/string_util.h"
#else
#if defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <pthread_np.h>
#else
#include <pthread.h>
#endif
#include <sched.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

namespace Common {

#ifdef __ANDROID__
namespace {

constexpr const char* AndroidThreadAffinityLogTag = "LSX4Native";

struct AndroidCpuTopology {
    cpu_set_t baseline_allowed{};
    cpu_set_t prime_cpus{};
    cpu_set_t performance_cpus{};
    cpu_set_t latency_cpus{};
    cpu_set_t graphics_cpus{};
    cpu_set_t compute_cpus{};
    std::array<std::uint64_t, CPU_SETSIZE> scores{};
    const char* score_source{"none"};
    std::uint64_t minimum_score{};
    std::uint64_t maximum_score{};
    int baseline_count{};
    int performance_count{};
    int prime_count{};
    int latency_count{};
    int graphics_count{};
    int compute_count{};
    bool usable{};
};

AndroidCpuTopology& GetAndroidCpuTopology() {
    static AndroidCpuTopology topology;
    return topology;
}

std::once_flag& GetAndroidCpuTopologyOnceFlag() {
    static std::once_flag once;
    return once;
}

int CountCpus(const cpu_set_t& cpus) {
    int count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        count += CPU_ISSET(cpu, &cpus) ? 1 : 0;
    }
    return count;
}

std::uint64_t ReadPositiveCpuValue(const int cpu, const char* relative_path) {
    std::ifstream input{"/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" +
                        relative_path};
    std::uint64_t value = 0;
    input >> value;
    return input ? value : 0;
}

bool DiscoverScoreSource(const cpu_set_t& allowed, const char* relative_path,
                         std::array<std::uint64_t, CPU_SETSIZE>& scores,
                         std::uint64_t& minimum, std::uint64_t& maximum) {
    minimum = std::numeric_limits<std::uint64_t>::max();
    maximum = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        const std::uint64_t value = ReadPositiveCpuValue(cpu, relative_path);
        if (value == 0) {
            return false;
        }
        scores[static_cast<std::size_t>(cpu)] = value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    return minimum != std::numeric_limits<std::uint64_t>::max() && maximum > minimum;
}

void DiscoverAndroidCpuTopology() {
    auto& topology = GetAndroidCpuTopology();
    CPU_ZERO(&topology.baseline_allowed);
    CPU_ZERO(&topology.prime_cpus);
    CPU_ZERO(&topology.performance_cpus);
    CPU_ZERO(&topology.latency_cpus);
    CPU_ZERO(&topology.graphics_cpus);
    CPU_ZERO(&topology.compute_cpus);

    if (sched_getaffinity(0, sizeof(topology.baseline_allowed),
                          &topology.baseline_allowed) != 0) {
        __android_log_print(ANDROID_LOG_WARN, AndroidThreadAffinityLogTag,
                            "[EXECUTOR_THREAD_AFFINITY_TOPOLOGY] enabled=0 "
                            "reason=affinity_read_failed errno=%d",
                            errno);
        return;
    }
    topology.baseline_count = CountCpus(topology.baseline_allowed);
    if (topology.baseline_count < 4) {
        __android_log_print(ANDROID_LOG_INFO, AndroidThreadAffinityLogTag,
                            "[EXECUTOR_THREAD_AFFINITY_TOPOLOGY] enabled=0 "
                            "reason=small_allowed_mask allowed=%d",
                            topology.baseline_count);
        return;
    }

    std::array<std::uint64_t, CPU_SETSIZE> candidate_scores{};
    if (DiscoverScoreSource(topology.baseline_allowed, "cpu_capacity", candidate_scores,
                            topology.minimum_score, topology.maximum_score)) {
        topology.score_source = "cpu_capacity";
        topology.scores = candidate_scores;
    } else {
        candidate_scores.fill(0);
        if (DiscoverScoreSource(topology.baseline_allowed, "cpufreq/cpuinfo_max_freq",
                                candidate_scores, topology.minimum_score,
                                topology.maximum_score)) {
            topology.score_source = "cpuinfo_max_freq";
            topology.scores = candidate_scores;
        } else {
            candidate_scores.fill(0);
            if (DiscoverScoreSource(topology.baseline_allowed, "cpufreq/scaling_max_freq",
                                    candidate_scores, topology.minimum_score,
                                    topology.maximum_score)) {
                topology.score_source = "scaling_max_freq";
                topology.scores = candidate_scores;
            } else {
                __android_log_print(
                    ANDROID_LOG_INFO, AndroidThreadAffinityLogTag,
                    "[EXECUTOR_THREAD_AFFINITY_TOPOLOGY] enabled=0 "
                    "reason=no_heterogeneous_capacity_data allowed=%d",
                    topology.baseline_count);
                return;
            }
        }
    }

    // Treat differences below ten percent as a homogeneous topology. Pinning
    // such a device only removes scheduler freedom without selecting faster CPUs.
    if (topology.maximum_score * 100u <= topology.minimum_score * 110u) {
        __android_log_print(
            ANDROID_LOG_INFO, AndroidThreadAffinityLogTag,
            "[EXECUTOR_THREAD_AFFINITY_TOPOLOGY] enabled=0 "
            "reason=effectively_homogeneous source=%s allowed=%d min=%llu max=%llu",
            topology.score_source, topology.baseline_count,
            static_cast<unsigned long long>(topology.minimum_score),
            static_cast<unsigned long long>(topology.maximum_score));
        return;
    }

    const std::uint64_t performance_floor = topology.maximum_score * 4u / 5u;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &topology.baseline_allowed)) {
            continue;
        }
        const auto score = topology.scores[static_cast<std::size_t>(cpu)];
        if (score >= performance_floor) {
            CPU_SET(cpu, &topology.prime_cpus);
            CPU_SET(cpu, &topology.performance_cpus);
        }
    }

    topology.prime_count = CountCpus(topology.prime_cpus);
    topology.performance_count = CountCpus(topology.performance_cpus);
    // A two-core prime island is too small for the guest main and GPU command
    // threads together on sustained workloads. Include the next cluster so the
    // Android scheduler can balance latency-critical work without a hot-spot.
    if (topology.performance_count <= 2) {
        std::uint64_t next_cluster_score = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &topology.baseline_allowed)) {
                continue;
            }
            const auto score = topology.scores[static_cast<std::size_t>(cpu)];
            if (score < topology.maximum_score) {
                next_cluster_score = std::max(next_cluster_score, score);
            }
        }
        if (next_cluster_score != 0) {
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (CPU_ISSET(cpu, &topology.baseline_allowed) &&
                    topology.scores[static_cast<std::size_t>(cpu)] >= next_cluster_score) {
                    CPU_SET(cpu, &topology.performance_cpus);
                }
            }
            topology.performance_count = CountCpus(topology.performance_cpus);
        }
    }

    // Keep the frame-latency guest and GPU command threads off each other's
    // CPU. Heavy Android emulators otherwise concentrate both, plus inherited
    // guest workers, on the same small prime island. Pick the two strongest
    // allowed CPUs deterministically and leave the remaining CPUs to audio,
    // physics and other guest workers.
    int latency_cpu = -1;
    int graphics_cpu = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &topology.baseline_allowed)) {
            continue;
        }
        if (latency_cpu < 0 ||
            topology.scores[static_cast<std::size_t>(cpu)] >
                topology.scores[static_cast<std::size_t>(latency_cpu)] ||
            (topology.scores[static_cast<std::size_t>(cpu)] ==
                 topology.scores[static_cast<std::size_t>(latency_cpu)] &&
             cpu > latency_cpu)) {
            latency_cpu = cpu;
        }
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (cpu == latency_cpu || !CPU_ISSET(cpu, &topology.baseline_allowed)) {
            continue;
        }
        if (graphics_cpu < 0 ||
            topology.scores[static_cast<std::size_t>(cpu)] >
                topology.scores[static_cast<std::size_t>(graphics_cpu)] ||
            (topology.scores[static_cast<std::size_t>(cpu)] ==
                 topology.scores[static_cast<std::size_t>(graphics_cpu)] &&
             cpu > graphics_cpu)) {
            graphics_cpu = cpu;
        }
    }
    if (latency_cpu >= 0) {
        CPU_SET(latency_cpu, &topology.latency_cpus);
    }
    if (graphics_cpu >= 0) {
        CPU_SET(graphics_cpu, &topology.graphics_cpus);
    }
    topology.latency_count = CountCpus(topology.latency_cpus);
    topology.graphics_count = CountCpus(topology.graphics_cpus);

    topology.compute_cpus = topology.baseline_allowed;
    if (topology.baseline_count >= 6) {
        if (latency_cpu >= 0) {
            CPU_CLR(latency_cpu, &topology.compute_cpus);
        }
        if (graphics_cpu >= 0) {
            CPU_CLR(graphics_cpu, &topology.compute_cpus);
        }
    }
    topology.compute_count = topology.baseline_count;
    topology.compute_count = CountCpus(topology.compute_cpus);
    topology.usable = topology.latency_count == 1 &&
                      topology.graphics_count == 1 &&
                      topology.compute_count >= 2;
    __android_log_print(
        topology.usable ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
        AndroidThreadAffinityLogTag,
        "[EXECUTOR_THREAD_AFFINITY_TOPOLOGY] enabled=%d source=%s allowed=%d "
        "prime=%d performance=%d latency=%d graphics=%d compute=%d "
        "latencyCpu=%d graphicsCpu=%d min=%llu max=%llu",
        topology.usable ? 1 : 0, topology.score_source, topology.baseline_count,
        topology.prime_count, topology.performance_count, topology.latency_count,
        topology.graphics_count, topology.compute_count, latency_cpu, graphics_cpu,
        static_cast<unsigned long long>(topology.minimum_score),
        static_cast<unsigned long long>(topology.maximum_score));
}

const char* ThreadAffinityClassName(const ThreadAffinityClass affinity_class) {
    switch (affinity_class) {
    case ThreadAffinityClass::LatencyPerformance:
        return "latency_performance";
    case ThreadAffinityClass::GuestCompute:
        return "guest_compute";
    case ThreadAffinityClass::GraphicsDriver:
        return "graphics_driver";
    }
    return "unknown";
}

} // namespace
#endif

void InitializeAndroidThreadAffinity() {
#ifdef __ANDROID__
    std::call_once(GetAndroidCpuTopologyOnceFlag(), DiscoverAndroidCpuTopology);
#endif
}

bool ApplyCurrentThreadAffinity(const ThreadAffinityClass affinity_class) {
#ifdef __ANDROID__
    InitializeAndroidThreadAffinity();
    const auto& topology = GetAndroidCpuTopology();
    if (!topology.usable) {
        return false;
    }

    cpu_set_t current_allowed;
    CPU_ZERO(&current_allowed);
    if (sched_getaffinity(0, sizeof(current_allowed), &current_allowed) != 0) {
        return false;
    }
    const int current_count = CountCpus(current_allowed);

    const cpu_set_t& role_cpus = [&]() -> const cpu_set_t& {
        switch (affinity_class) {
        case ThreadAffinityClass::LatencyPerformance:
            return topology.latency_cpus;
        case ThreadAffinityClass::GuestCompute:
            return topology.compute_cpus;
        case ThreadAffinityClass::GraphicsDriver:
            return topology.graphics_cpus;
        }
        return topology.baseline_allowed;
    }();
    cpu_set_t requested;
    CPU_ZERO(&requested);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &topology.baseline_allowed) &&
            CPU_ISSET(cpu, &role_cpus)) {
            CPU_SET(cpu, &requested);
        }
    }
    const int requested_count = CountCpus(requested);
    if (requested_count == 0) {
        return false;
    }

    const int result = sched_setaffinity(0, sizeof(requested), &requested);
    const int saved_errno = result == 0 ? 0 : errno;
    cpu_set_t actual;
    CPU_ZERO(&actual);
    const int actual_count =
        result == 0 && sched_getaffinity(0, sizeof(actual), &actual) == 0
            ? CountCpus(actual)
            : 0;

    static std::array<std::atomic_uint32_t, 3> log_counts{};
    const auto role_index = static_cast<std::size_t>(affinity_class);
    if (role_index < log_counts.size() &&
        log_counts[role_index].fetch_add(1, std::memory_order_relaxed) < 16) {
        __android_log_print(
            result == 0 ? ANDROID_LOG_INFO : ANDROID_LOG_WARN,
            AndroidThreadAffinityLogTag,
            "[EXECUTOR_THREAD_AFFINITY] role=%s before=%d requested=%d actual=%d "
            "result=%d errno=%d",
            ThreadAffinityClassName(affinity_class), current_count, requested_count,
            actual_count, result, saved_errno);
    }
    return result == 0 && actual_count != 0;
#else
    (void)affinity_class;
    return false;
#endif
}

#ifdef __APPLE__

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    const std::chrono::nanoseconds computation_ns = period_ns / 2;

    struct mach_timebase_info timebase{};
    mach_timebase_info(&timebase);
    const auto ticks_per_ns =
        static_cast<double>(timebase.denom) / static_cast<double>(timebase.numer);

    const auto period_ticks =
        static_cast<u32>(static_cast<double>(period_ns.count()) * ticks_per_ns);
    const auto computation_ticks =
        static_cast<u32>(static_cast<double>(computation_ns.count()) * ticks_per_ns);

    thread_time_constraint_policy policy = {
        .period = period_ticks,
        .computation = computation_ticks,
        .constraint = computation_ticks,
        .preemptible = false,
    };

    int ret = thread_policy_set(
        pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
        reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    if (ret != KERN_SUCCESS) {
        LOG_ERROR(Common, "Could not set thread to real-time with period {} ns: {}",
                  period_ns.count(), ret);
    }
}

#else

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
}

#endif

#ifdef _WIN32

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    auto handle = GetCurrentThread();
    int windows_priority = 0;
    switch (new_priority) {
    case ThreadPriority::Low:
        windows_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    case ThreadPriority::Normal:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    case ThreadPriority::High:
        windows_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case ThreadPriority::VeryHigh:
        windows_priority = THREAD_PRIORITY_HIGHEST;
        break;
    case ThreadPriority::Critical:
        windows_priority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    default:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    }
    SetThreadPriority(handle, windows_priority);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();

    LARGE_INTEGER interval{
        .QuadPart = -1 * (duration.count() / 100u),
    };
    HANDLE timer = ::CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &interval, 0, NULL, NULL, 0);
    const auto ret = WaitForSingleObjectEx(timer, INFINITE, interruptible);
    ::CloseHandle(timer);

    if (remaining) {
        const auto end_sleep = std::chrono::high_resolution_clock::now();
        const auto sleep_time = end_sleep - begin_sleep;
        *remaining = duration > sleep_time ? duration - sleep_time : std::chrono::nanoseconds(0);
    }
    return ret == WAIT_OBJECT_0;
}

#else

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    pthread_t this_thread = pthread_self();

    const auto scheduling_type = SCHED_OTHER;
    s32 max_prio = sched_get_priority_max(scheduling_type);
    s32 min_prio = sched_get_priority_min(scheduling_type);
    u32 level = std::max(static_cast<u32>(new_priority) + 1, 4U);

    struct sched_param params;
    if (max_prio > min_prio) {
        params.sched_priority = min_prio + ((max_prio - min_prio) * level) / 4;
    } else {
        params.sched_priority = min_prio - ((min_prio - max_prio) * level) / 4;
    }

    pthread_setschedparam(this_thread, scheduling_type, &params);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    timespec request = {
        .tv_sec = duration.count() / 1'000'000'000,
        .tv_nsec = duration.count() % 1'000'000'000,
    };
    timespec remain;
    int ret;
    while ((ret = nanosleep(&request, &remain)) < 0 && errno == EINTR) {
        if (interruptible) {
            break;
        }
        request = remain;
    }
    if (remaining) {
        *remaining = std::chrono::nanoseconds(remain.tv_sec * 1'000'000'000 + remain.tv_nsec);
    }
    return ret == 0 || errno != EINTR;
}

#endif

#ifdef _MSC_VER

void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
}

void SetThreadName(void* thread, const char* name) {
    SetThreadDescription(thread, UTF8ToUTF16W(name).data());
}

#else

#if !defined(_WIN32) || defined(_MSC_VER)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
#ifdef __APPLE__
    pthread_setname_np(name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__)
    std::string truncated(name, std::min(strlen(name), static_cast<std::size_t>(15)));
    if (int e = pthread_setname_np(pthread_self(), truncated.c_str())) {
        errno = e;
        LOG_ERROR(Common, "Failed to set thread name to '{}': {}", truncated, GetLastErrorMsg());
    }
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void SetThreadName(void* thread, const char* name) {
}
#endif

#if defined(_WIN32)
void SetCurrentThreadName(const char*) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
}

void SetThreadName(void* thread, const char* name) {
}
#endif

#endif

AccurateTimer::AccurateTimer(std::chrono::nanoseconds target_interval)
    : target_interval(target_interval) {}

void AccurateTimer::Start() {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();
    if (total_wait.count() > 0) {
        AccurateSleep(total_wait, nullptr, false);
    }
    start_time = std::chrono::high_resolution_clock::now();
    total_wait -= std::chrono::duration_cast<std::chrono::nanoseconds>(start_time - begin_sleep);
}

void AccurateTimer::End() {
    auto now = std::chrono::high_resolution_clock::now();
    total_wait +=
        target_interval - std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time);
}

std::string GetCurrentThreadName() {
    using namespace Libraries::Kernel;
    if (g_curthread && !g_curthread->name.empty()) {
        return g_curthread->name;
    }
#ifdef _WIN32
    PWSTR name;
    GetThreadDescription(GetCurrentThread(), &name);
    return Common::UTF16ToUTF8(name);
#else
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return std::string{name};
#endif
}

}
