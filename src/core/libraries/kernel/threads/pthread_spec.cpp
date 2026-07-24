// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/libs.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>
extern "C" bool executor_lsx4_android_should_auto_allocate_pthread_key(int key)
    __attribute__((weak));
extern "C" void executor_lsx4_android_note_auto_allocated_pthread_key(int key)
    __attribute__((weak));
extern "C" int executor_lsx4_android_run_guest_callback(void* callback, void* arg,
                                                           const char* reason)
    __attribute__((weak));
extern "C" bool executor_lsx4_android_is_guest_address(void* address) __attribute__((weak));
extern "C" const void* executor_get_captured_current_thread(unsigned real_key,
                                                            const std::uint64_t* per_thread_by_index)
    __attribute__((weak));
extern "C" std::size_t executor_pthread_mono_by_index_offset() {
    return __builtin_offsetof(::Libraries::Kernel::Pthread, mono_by_index);
}
extern "C" int executor_live_get_current_hle_call_site(std::uint64_t* guest_return,
                                                       std::uint64_t* return_off,
                                                       std::uint64_t* arg0,
                                                       char* symbol,
                                                       std::size_t symbol_size,
                                                       char* module,
                                                       std::size_t module_size)
    __attribute__((weak));

static bool ExecutorLightOracleMode() {
    return std::getenv("EXECUTOR_LIGHT_ORACLE") != nullptr;
}
#endif

namespace Libraries::Kernel {

static constexpr u32 PthreadKeysMax = 256;
static constexpr u32 PthreadDestructorIterations = 4;

static std::array<PthreadKey, PthreadKeysMax> ThreadKeytable{};
static std::mutex KeytableLock;

#ifdef __ANDROID__
static bool IsUnityTlsThread(const Pthread* pthread) {
    if (pthread == nullptr) {
        return false;
    }
    const auto& name = pthread->name;
    return name.find("Game:Main") != std::string::npos ||
           name.find("mono") != std::string::npos ||
           name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("UnityPreload") != std::string::npos ||
           name.find("Thread2") != std::string::npos ||
           name.find("Thread3") != std::string::npos ||
           name.find("ExecutorBridgeFallback") != std::string::npos;
}

static bool ShouldTraceUnityTls(const Pthread* pthread, PthreadKeyT key) {
    if (ExecutorLightOracleMode()) {
        return key < 16 || IsUnityTlsThread(pthread);
    }
    if (std::getenv("EXECUTOR_TRACE_LIVE_WIDE") == nullptr &&
        std::getenv("EXECUTOR_TRACE_LIVE_TLS") == nullptr) {
        return false;
    }
    if (std::getenv("EXECUTOR_TRACE_LIVE_RENDER_THREAD_ONLY") != nullptr) {
        if (pthread == nullptr) {
            return false;
        }
        const auto& name = pthread->name;
        return name.find("Game:Main") != std::string::npos ||
               name.find("UnityGfxDeviceWorker") != std::string::npos ||
               name.find("UnityWorker") != std::string::npos ||
               name.find("Submit Done Thread") != std::string::npos;
    }
    return IsUnityTlsThread(pthread);
}

static bool ConsumeLightOracleTlsBudget(const char* op) {
    if (!ExecutorLightOracleMode()) {
        return true;
    }
    if (op == nullptr) {
        return false;
    }

    static std::atomic_int key_budget{64};
    static std::atomic_int set_budget{128};
    static std::atomic_int stale_budget{16};
    static std::atomic_int get_budget{64};
    static std::atomic_int null_budget{64};

    std::atomic_int* selected = nullptr;
    if (std::strcmp(op, "key_create") == 0 || std::strcmp(op, "key_create_fail") == 0) {
        selected = &key_budget;
    } else if (std::strncmp(op, "setspecific", 11) == 0) {
        selected = &set_budget;
    } else if (std::strcmp(op, "getspecific_stale") == 0) {
        selected = &stale_budget;
    } else if (std::strcmp(op, "getspecific_null") == 0) {
        selected = &null_budget;
    } else if (std::strcmp(op, "getspecific") == 0) {
        selected = &get_budget;
    } else {
        return false;
    }

    return selected->fetch_sub(1, std::memory_order_relaxed) > 0;
}

static void TraceUnityTls(const char* op, PthreadKeyT key, const void* value, int ret,
                          const Pthread* pthread = nullptr) {
    if (pthread == nullptr) {
        pthread = g_curthread;
    }
    if (!ShouldTraceUnityTls(pthread, key)) {
        return;
    }
    if (!ConsumeLightOracleTlsBudget(op)) {
        return;
    }
    static std::atomic_int budget{1024};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    const bool allocated = key < PthreadKeysMax && ThreadKeytable[key].allocated;
    const u32 seq = key < PthreadKeysMax ? ThreadKeytable[key].seqno.load() : 0;
    const void* slot_value = nullptr;
    int slot_seq = 0;
    if (pthread != nullptr && pthread->specific != nullptr && key >= 0 && key < PthreadKeysMax) {
        slot_value = pthread->specific[key].data;
        slot_seq = pthread->specific[key].seqno;
    }
    std::uint64_t guest_return = 0;
    std::uint64_t return_off = 0;
    char symbol[96]{};
    char module[64]{};
    const int hle_rc = executor_live_get_current_hle_call_site != nullptr
                           ? executor_live_get_current_hle_call_site(
                                 &guest_return, &return_off, nullptr, symbol, sizeof(symbol),
                                 module, sizeof(module))
                           : -1;
    const auto host_tid = static_cast<int>(syscall(__NR_gettid));
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_TLS] op=%s hostTid=%d thread=%s pthread=%p gcur=%p "
                        "key=%u allocated=%u tableSeq=%u slotSeq=%d value=%p slotValue=%p "
                        "ret=%d specific=%p count=%u hleRc=%d guestReturn=%p returnOff=0x%llx "
                        "symbol=%s module=%s",
                        op, host_tid, pthread ? pthread->name.c_str() : "<no-pthread>", pthread,
                        g_curthread, key, allocated ? 1u : 0u, seq, slot_seq, value, slot_value,
                        ret, pthread ? pthread->specific : nullptr,
                        pthread ? pthread->specific_data_count : 0u, hle_rc,
                        reinterpret_cast<void*>(guest_return),
                        static_cast<unsigned long long>(return_off),
                        symbol[0] ? symbol : "<unknown>", module[0] ? module : "<unknown>");
}
#else
static bool PcOracleMonoSyncEnabled() {
    const char* value = std::getenv("EXECUTOR_PC_ORACLE_MONO_SYNC");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool PcOracleIsUnityTlsThread(const Pthread* pthread) {
    if (pthread == nullptr) {
        return false;
    }
    const auto& name = pthread->name;
    return name.find("Game:Main") != std::string::npos ||
           name.find("mono") != std::string::npos ||
           name.find("UnityGfxDeviceWorker") != std::string::npos ||
           name.find("UnityWorker") != std::string::npos ||
           name.find("UnityPreload") != std::string::npos ||
           name.find("Thread2") != std::string::npos ||
           name.find("Thread3") != std::string::npos;
}

static std::uint64_t PcOracleGuestOff(const void* pc) {
    const auto value = reinterpret_cast<std::uintptr_t>(pc);
    return value >= 0x800000000ULL ? value - 0x800000000ULL : 0ULL;
}

static void PcOracleTraceUnityTls(const char* op, PthreadKeyT key, const void* value, int ret,
                                  const Pthread* pthread, const void* guest_return) {
    if (!PcOracleMonoSyncEnabled()) {
        return;
    }
    if (!(key < 16 || PcOracleIsUnityTlsThread(pthread))) {
        return;
    }
    static std::atomic_int budget{16384};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    const bool allocated = key < PthreadKeysMax && ThreadKeytable[key].allocated;
    const u32 seq = key < PthreadKeysMax ? ThreadKeytable[key].seqno : 0;
    const void* slot_value = nullptr;
    int slot_seq = 0;
    if (pthread != nullptr && pthread->specific != nullptr && key < PthreadKeysMax) {
        slot_value = pthread->specific[key].data;
        slot_seq = pthread->specific[key].seqno;
    }
    std::fprintf(stderr,
                 "[EXECUTOR_PC_TLS] op=%s thread=%s pthread=%p gcur=%p key=%u allocated=%u "
                 "tableSeq=%u slotSeq=%d value=%p slotValue=%p ret=%d specific=%p count=%u "
                 "retaddr=%p returnOff=0x%llx\n",
                 op ? op : "<unknown>",
                 pthread && !pthread->name.empty() ? pthread->name.c_str() : "<no-pthread>",
                 pthread, g_curthread, key, allocated ? 1u : 0u, seq, slot_seq, value,
                 slot_value, ret, pthread ? pthread->specific : nullptr,
                 pthread ? pthread->specific_data_count : 0u, guest_return,
                 static_cast<unsigned long long>(PcOracleGuestOff(guest_return)));
    std::fflush(stderr);
}
#endif

void RunPthreadKeyDestructor(PthreadKeyDestructor destructor, const void* data,
                             const char* reason) {
    if (destructor == nullptr) {
        return;
    }
#ifdef __ANDROID__
    if (executor_lsx4_android_run_guest_callback != nullptr) {
        const int rc = executor_lsx4_android_run_guest_callback(
            reinterpret_cast<void*>(destructor), const_cast<void*>(data),
            reason ? reason : "pthread_key_destructor");
        if (rc > 0) {
            return;
        }
        if (rc < 0) {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_GUEST_CALLBACK] failed kind=tls_destructor reason=%s "
                                "destructor=%p data=%p rc=%d",
                                reason ? reason : "pthread_key_destructor",
                                reinterpret_cast<void*>(destructor), data, rc);
            return;
        }
    }
    if (executor_lsx4_android_is_guest_address != nullptr &&
        executor_lsx4_android_is_guest_address(reinterpret_cast<void*>(destructor))) {
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_GUEST_CALLBACK] blocked_direct_guest_tls_destructor "
                            "reason=%s destructor=%p data=%p",
                            reason ? reason : "pthread_key_destructor",
                            reinterpret_cast<void*>(destructor), data);
        return;
    }
#endif
    destructor(data);
}

int PS4_SYSV_ABI posix_pthread_key_create(PthreadKeyT* key, PthreadKeyDestructor destructor) {
    std::scoped_lock lk{KeytableLock};
    const auto it = std::ranges::find_if(
        ThreadKeytable, [](const PthreadKey& entry) { return entry.allocated.load() == 0; });
    if (it != ThreadKeytable.end()) {
        it->allocated = 1;
        it->destructor = destructor;
        it->seqno++;
        *key = static_cast<PthreadKeyT>(std::distance(ThreadKeytable.begin(), it));
#ifdef __ANDROID__
        TraceUnityTls("key_create", *key, reinterpret_cast<const void*>(destructor), 0);
#else
        PcOracleTraceUnityTls("key_create", *key, reinterpret_cast<const void*>(destructor), 0,
                              g_curthread, __builtin_return_address(0));
#endif
        return 0;
    }
#ifdef __ANDROID__
    TraceUnityTls("key_create_fail", PthreadKeysMax, reinterpret_cast<const void*>(destructor),
                  POSIX_EAGAIN);
#else
    PcOracleTraceUnityTls("key_create_fail", PthreadKeysMax,
                          reinterpret_cast<const void*>(destructor), POSIX_EAGAIN, g_curthread,
                          __builtin_return_address(0));
#endif
    return POSIX_EAGAIN;
}

int PS4_SYSV_ABI posix_pthread_key_delete(PthreadKeyT key) {
    if (key >= PthreadKeysMax) {
        return POSIX_EINVAL;
    }

    std::scoped_lock lk{KeytableLock};
    if (!ThreadKeytable[key].allocated) {
        return POSIX_EINVAL;
    }

    ThreadKeytable[key].allocated = 0;
    return 0;
}

void _thread_cleanupspecific() {
    Pthread* curthread = g_curthread;
    PthreadKeyDestructor destructor;
    const void* data = nullptr;

    if (curthread->specific == nullptr) {
        return;
    }

    std::unique_lock lk{KeytableLock};
    for (int i = 0; (i < PthreadDestructorIterations) && (curthread->specific_data_count > 0);
         i++) {
        for (int key = 0; (key < PthreadKeysMax) && (curthread->specific_data_count > 0); key++) {
            destructor = nullptr;

            if (ThreadKeytable[key].allocated && (curthread->specific[key].data != nullptr)) {
                if (curthread->specific[key].seqno == ThreadKeytable[key].seqno) {
                    data = curthread->specific[key].data;
                    destructor = ThreadKeytable[key].destructor;
                }
                curthread->specific[key].data = nullptr;
                curthread->specific_data_count--;
            } else if (curthread->specific[key].data != nullptr) {
                curthread->specific[key].data = nullptr;
                curthread->specific_data_count--;
            }

            if (destructor != nullptr) {
                lk.unlock();
                RunPthreadKeyDestructor(destructor, data, "pthread_key_destructor");
                lk.lock();
            }
        }
    }
    delete[] curthread->specific;
    curthread->specific = nullptr;
    if (curthread->specific_data_count > 0) {
        LOG_WARNING(Lib_Kernel, "Thread has exited with leftover thread-specific data");
    }
}

#ifdef __ANDROID__
struct HostTlsElem { const void* data = nullptr; u32 seqno = 0; bool set = false; };
static thread_local HostTlsElem g_host_tls[PthreadKeysMax];
static bool ExecutorHostTlsShadowDisabled() {
    static const bool off =
        std::fopen("/data/data/app.lsx4.android/files/lsx4-home/run-no-host-tls-shadow",
                   "r") != nullptr;
    return off;
}

struct NameTlsElem { const void* data = nullptr; u32 seqno = 0; };
static std::mutex g_name_tls_lock;
static std::unordered_map<std::string, std::array<NameTlsElem, PthreadKeysMax>> g_name_tls;
static std::atomic<std::uint32_t> g_global_set_count[PthreadKeysMax];
static const void* g_global_last_value[PthreadKeysMax];
static std::array<std::string, PthreadKeysMax> g_global_last_name;
static bool ExecutorTlsForensicsEnabled() {
    static const bool on = [] {
        const char* value = std::getenv("EXECUTOR_TRACE_TLS_FORENSICS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return on;
}
static bool ExecutorNameTlsSubstituteEnabled() {
    static const bool on =
        std::fopen("/data/data/app.lsx4.android/files/lsx4-home/run-tls-name-substitute",
                   "r") != nullptr;
    return on;
}
static void NameTlsRecord(const std::string& name, PthreadKeyT key, const void* value, u32 seqno) {
    if ((!ExecutorNameTlsSubstituteEnabled() && !ExecutorTlsForensicsEnabled()) || name.empty() ||
        key >= PthreadKeysMax) {
        return;
    }
    std::scoped_lock lk{g_name_tls_lock};
    auto& arr = g_name_tls[name];
    arr[key].data = value;
    arr[key].seqno = seqno;
    g_global_set_count[key].fetch_add(1, std::memory_order_relaxed);
    g_global_last_value[key] = value;
    g_global_last_name[key] = name;
}
static const void* NameTlsLookup(const std::string& name, PthreadKeyT key, u32 want_seqno) {
    if (name.empty() || key >= PthreadKeysMax) {
        return nullptr;
    }
    std::scoped_lock lk{g_name_tls_lock};
    const auto it = g_name_tls.find(name);
    if (it == g_name_tls.end()) {
        return nullptr;
    }
    const auto& e = it->second[key];
    if (e.data != nullptr && e.seqno == want_seqno) {
        return e.data;
    }
    return nullptr;
}
static void NameTlsPersistFile(const char* line) {
    static std::atomic<int> file_budget{600};
    if (file_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    static const bool disabled =
        std::fopen("/data/data/app.lsx4.android/files/lsx4-home/run-no-nametls-file",
                   "r") != nullptr;
    if (disabled) {
        return;
    }
    if (std::FILE* f =
            std::fopen("/data/data/app.lsx4.android/files/executor-nametls.log", "a")) {
        std::fputs(line, f);
        std::fputc('\n', f);
        std::fclose(f);
    }
}
static const void* NameTlsRecover(const Pthread* pthread, PthreadKeyT key) {
    if (pthread == nullptr || key >= PthreadKeysMax || !ThreadKeytable[key].allocated) {
        return nullptr;
    }
    const bool substitute = ExecutorNameTlsSubstituteEnabled();
    const bool trace = ExecutorTlsForensicsEnabled();
    if (!substitute && !trace) {
        return nullptr;
    }
    const void* nv = NameTlsLookup(pthread->name, key, ThreadKeytable[key].seqno);
    if (nv == nullptr) {
        static std::atomic<int> s_miss_log{256};
        if (trace && s_miss_log.fetch_sub(1, std::memory_order_relaxed) > 0) {
            const std::uint32_t gsc = g_global_set_count[key].load(std::memory_order_relaxed);
            std::scoped_lock lk{g_name_tls_lock};
            char line[256]{};
            std::snprintf(line, sizeof(line),
                          "MISS key=%u thread=%s globalSetCount=%u lastSetName=%s lastSetValue=%p",
                          key, pthread->name.c_str(), gsc,
                          gsc ? g_global_last_name[key].c_str() : "<none>",
                          gsc ? g_global_last_value[key] : nullptr);
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_NAME_TLS_MISS] %s (never-set if count=0)", line);
            NameTlsPersistFile(line);
        }
        return nullptr;
    }
    static std::atomic<int> s_name_log{128};
    if (trace && s_name_log.fetch_sub(1, std::memory_order_relaxed) > 0) {
        char line[256]{};
        std::snprintf(line, sizeof(line), "HIT key=%u value=%p thread=%s subst=%d", key, nv,
                      pthread->name.c_str(), ExecutorNameTlsSubstituteEnabled() ? 1 : 0);
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_NAME_TLS_HIT] %s (guest-Pthread+host-thread stores both missed)",
                            line);
        NameTlsPersistFile(line);
    }
    return substitute ? nv : nullptr;
}

static void SetspecTrace(const char* tag, PthreadKeyT key, const void* value, const Pthread* pthread) {
    if (!ExecutorTlsForensicsEnabled()) {
        return;
    }
    static std::atomic<int> budget{4000};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    static const bool disabled =
        std::fopen("/data/data/app.lsx4.android/files/lsx4-home/run-no-setspec-file",
                   "r") != nullptr;
    if (disabled) {
        return;
    }
    if (std::FILE* f =
            std::fopen("/data/data/app.lsx4.android/files/executor-setspec.log", "a")) {
        std::fprintf(f, "%s key=%u value=%p hostTid=%d pthread=%p name=%s\n", tag, key, value,
                     static_cast<int>(syscall(__NR_gettid)), pthread,
                     pthread ? pthread->name.c_str() : "<null>");
        std::fclose(f);
    }
}
#endif

int PS4_SYSV_ABI posix_pthread_setspecific(PthreadKeyT key, const void* value) {
#ifdef __ANDROID__
    SetspecTrace("call", key, value, g_curthread);
#endif
    Pthread* pthread = CurrentOrFallbackPthread();
    if (pthread == nullptr) {
#ifdef __ANDROID__
        SetspecTrace("fail_nopthread", key, value, nullptr);
#endif
        return POSIX_EINVAL;
    }

    if (!pthread->specific) {
        pthread->specific = new (std::nothrow) PthreadSpecificElem[PthreadKeysMax]{};
        if (pthread->specific == nullptr) {
            return POSIX_ENOMEM;
        }
    }
    if (key >= PthreadKeysMax) {
        return POSIX_EINVAL;
    }
    if (!ThreadKeytable[key].allocated) {
#ifdef __ANDROID__
        if (executor_lsx4_android_should_auto_allocate_pthread_key != nullptr &&
            executor_lsx4_android_should_auto_allocate_pthread_key(static_cast<int>(key))) {
            std::scoped_lock lk{KeytableLock};
            if (!ThreadKeytable[key].allocated) {
                ThreadKeytable[key].allocated = 1;
                ThreadKeytable[key].destructor = nullptr;
                ThreadKeytable[key].seqno++;
                if (executor_lsx4_android_note_auto_allocated_pthread_key != nullptr) {
                    executor_lsx4_android_note_auto_allocated_pthread_key(
                        static_cast<int>(key));
                }
                TraceUnityTls("setspecific_auto_alloc", key, value, 0, pthread);
            }
        }
#endif
    }
    if (!ThreadKeytable[key].allocated) {
#ifdef __ANDROID__
        SetspecTrace("fail_noalloc", key, value, pthread);
#endif
        return POSIX_EINVAL;
    }

    if (pthread->specific[key].data == nullptr) {
        if (value != nullptr) {
            pthread->specific_data_count++;
        }
    } else if (value == nullptr) {
        pthread->specific_data_count--;
    }
    pthread->specific[key].data = value;
    pthread->specific[key].seqno = ThreadKeytable[key].seqno;
#ifdef __ANDROID__
    if (!ExecutorHostTlsShadowDisabled() && key < PthreadKeysMax) {
        g_host_tls[key].data = value;
        g_host_tls[key].seqno = ThreadKeytable[key].seqno;
        g_host_tls[key].set = true;
    }
    NameTlsRecord(pthread->name, key, value, ThreadKeytable[key].seqno);
    SetspecTrace("ok", key, value, pthread);
    TraceUnityTls("setspecific", key, value, 0, pthread);
#else
    PcOracleTraceUnityTls("setspecific", key, value, 0, pthread, __builtin_return_address(0));
#endif
    return 0;
}

#ifdef __ANDROID__
static void ExecutorPersistTlsNull(const char* reason, PthreadKeyT key, const Pthread* pthread) {
    if (!ExecutorTlsForensicsEnabled()) {
        return;
    }
    static std::atomic<int> budget{400};
    if (budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    static const bool disabled =
        std::fopen("/data/data/app.lsx4.android/files/lsx4-home/run-no-tlsnull-diag",
                   "r") != nullptr;
    if (disabled) {
        return;
    }
    const bool allocated = key < PthreadKeysMax && ThreadKeytable[key].allocated;
    const u32 tseq = key < PthreadKeysMax ? ThreadKeytable[key].seqno.load() : 0;
    int sseq = -1;
    const void* sval = nullptr;
    const bool has_arr = pthread && pthread->specific;
    if (has_arr && key < PthreadKeysMax) {
        sseq = pthread->specific[key].seqno;
        sval = pthread->specific[key].data;
    }
    if (std::FILE* f = std::fopen(
            "/data/data/app.lsx4.android/files/executor-tlsnull.log", "a")) {
        std::fprintf(f, "%s key=%u alloc=%d tableSeq=%u slotSeq=%d slotVal=%p hasArr=%d thread=%s\n",
                     reason, key, allocated ? 1 : 0, tseq, sseq, sval, has_arr ? 1 : 0,
                     pthread ? pthread->name.c_str() : "<null>");
        std::fclose(f);
    }
}
#endif

const void* PS4_SYSV_ABI posix_pthread_getspecific(PthreadKeyT key) {
    Pthread* pthread = CurrentOrFallbackPthread();
    if (pthread == nullptr) {
        return nullptr;
    }

    if (!pthread->specific || key >= PthreadKeysMax) {
#ifdef __ANDROID__
        TraceUnityTls("getspecific_null", key, nullptr, 0, pthread);
        ExecutorPersistTlsNull("noarray", key, pthread);
        if (const void* nv = NameTlsRecover(pthread, key)) {
            return nv;
        }
#else
        PcOracleTraceUnityTls("getspecific_null", key, nullptr, 0, pthread,
                              __builtin_return_address(0));
#endif
        return nullptr;
    }

    if (ThreadKeytable[key].allocated &&
        (pthread->specific[key].seqno == ThreadKeytable[key].seqno)) {
        const void* value = pthread->specific[key].data;
#ifdef __ANDROID__
        TraceUnityTls("getspecific", key, value, 0, pthread);
#else
        PcOracleTraceUnityTls("getspecific", key, value, 0, pthread,
                              __builtin_return_address(0));
#endif
        return value;
    }

#ifdef __ANDROID__
    if (!ExecutorHostTlsShadowDisabled() && key < PthreadKeysMax && g_host_tls[key].set &&
        ThreadKeytable[key].allocated &&
        g_host_tls[key].seqno == ThreadKeytable[key].seqno && g_host_tls[key].data != nullptr) {
        static std::atomic<int> s_hit_log{32};
        if (s_hit_log.fetch_sub(1, std::memory_order_relaxed) > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_HOST_TLS_HIT] key=%u value=%p thread=%s (guest-slot missed)",
                                key, g_host_tls[key].data, pthread->name.c_str());
        }
        return g_host_tls[key].data;
    }
    TraceUnityTls("getspecific_stale", key, nullptr, 0, pthread);
    ExecutorPersistTlsNull("stale", key, pthread);
    if (executor_get_captured_current_thread != nullptr) {
        if (const void* cap = executor_get_captured_current_thread(static_cast<unsigned>(key),
                                                                   pthread->mono_by_index)) {
            static std::atomic<int> s_cap_log{32};
            if (s_cap_log.fetch_sub(1, std::memory_order_relaxed) > 0) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                    "[EXECUTOR_CURTHREAD_INJECT] key=%u value=%p thread=%s (from "
                                    "set-wrapper capture)",
                                    key, cap, pthread->name.c_str());
            }
            return cap;
        }
    }
    if (const void* nv = NameTlsRecover(pthread, key)) {
        return nv;
    }
#else
    PcOracleTraceUnityTls("getspecific_stale", key, nullptr, 0, pthread,
                          __builtin_return_address(0));
#endif
    return nullptr;
}

void RegisterSpec(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("mqULNdimTn0", "libScePosix", 1, "libkernel", posix_pthread_key_create);
    LIB_FUNCTION("6BpEZuDT7YI", "libScePosix", 1, "libkernel", posix_pthread_key_delete);
    LIB_FUNCTION("0-KXaS70xy4", "libScePosix", 1, "libkernel", posix_pthread_getspecific);
    LIB_FUNCTION("WrOLvHU0yQM", "libScePosix", 1, "libkernel", posix_pthread_setspecific);

    LIB_FUNCTION("mqULNdimTn0", "libkernel", 1, "libkernel", posix_pthread_key_create);
    LIB_FUNCTION("0-KXaS70xy4", "libkernel", 1, "libkernel", posix_pthread_getspecific);
    LIB_FUNCTION("WrOLvHU0yQM", "libkernel", 1, "libkernel", posix_pthread_setspecific);

    LIB_FUNCTION("geDaqgH9lTg", "libkernel", 1, "libkernel", ORBIS(posix_pthread_key_create));
    LIB_FUNCTION("PrdHuuDekhY", "libkernel", 1, "libkernel", ORBIS(posix_pthread_key_delete));
    LIB_FUNCTION("eoht7mQOCmo", "libkernel", 1, "libkernel", posix_pthread_getspecific);
    LIB_FUNCTION("+BzXYkqYeLE", "libkernel", 1, "libkernel", ORBIS(posix_pthread_setspecific));
}

}
