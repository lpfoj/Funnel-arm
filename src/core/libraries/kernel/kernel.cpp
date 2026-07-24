// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <random>
#include <thread>
#include <boost/asio/io_context.hpp>

#include "common/assert.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "common/va_ctx.h"
#include "core/file_sys/fs.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/debug.h"
#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/file_system.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/memory.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/kernel/threads.h"
#include "core/libraries/kernel/threads/exception.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/libs.h"
#include "core/libraries/network/sys_net.h"

#ifdef _WIN64
#include <Rpc.h>
#elif !defined(__ANDROID__)
#include <uuid/uuid.h>
#endif
#include <common/singleton.h>
#include <core/libraries/network/net_error.h>
#include <core/libraries/network/sockets.h>
#include <core/linker.h>
#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#include <atomic>
#include <sys/mman.h>
#endif
#include "aio.h"

namespace Libraries::Kernel {

#ifdef __ANDROID__
static u64 g_stack_chk_guard = 0;
#else
static u64 g_stack_chk_guard = 0xDEADBEEF54321ABC;
#endif

#ifdef __ANDROID__
u64* GetStackChkGuardExportStorage() {
    static u64* guard_storage = []() -> u64* {
        void* page = mmap(nullptr, 0x4000, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (page == MAP_FAILED) {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_STACK_GUARD_EXPORT] mmap failed, using native bss ptr=%p",
                                &g_stack_chk_guard);
            return &g_stack_chk_guard;
        }
        auto* guard = static_cast<u64*>(page);
        *guard = g_stack_chk_guard;
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_STACK_GUARD_EXPORT] storage=%p value=0x%llx mode=zero_model",
                            guard, static_cast<unsigned long long>(*guard));
        return guard;
    }();
    return guard_storage;
}

extern "C" int executor_live_get_current_hle_call_site(std::uint64_t* guest_return,
                                                       std::uint64_t* return_off,
                                                       std::uint64_t* arg0,
                                                       char* symbol,
                                                       std::size_t symbol_size,
                                                       char* module,
                                                       std::size_t module_size)
    __attribute__((weak));
extern "C" int executor_lsx4_android_get_current_guest_regs(void* out, std::size_t out_size)
    __attribute__((weak));
extern "C" int executor_lsx4_android_symbolize_guest_pc(
    std::uint64_t pc, char* module_name, std::size_t module_name_size, std::uint64_t* module_base,
    std::uint64_t* module_offset, char* symbol_name, std::size_t symbol_name_size,
    std::uint64_t* symbol_offset, std::uint64_t* symbol_delta) __attribute__((weak));

struct ExecutorStackChkGuestRegs {
    std::uint64_t regs[16];
    std::uint64_t rip;
    std::uint64_t old_ip;
    std::uint64_t fsbase;
    std::uint64_t gsbase;
    int quit;
    int exit;
    int error;
};

static void LogStackChkSymbol(const char* tag, std::uint64_t pc) {
    if (pc == 0 || executor_lsx4_android_symbolize_guest_pc == nullptr) {
        return;
    }
    char pc_module[192]{};
    char pc_symbol[128]{};
    std::uint64_t pc_base = 0;
    std::uint64_t pc_off = 0;
    std::uint64_t sym_off = 0;
    std::uint64_t sym_delta = 0;
    const int rc = executor_lsx4_android_symbolize_guest_pc(
        pc, pc_module, sizeof(pc_module), &pc_base, &pc_off, pc_symbol, sizeof(pc_symbol),
        &sym_off, &sym_delta);
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_STACK_CHK_SYMBOL] tag=%s pc=%p rc=%d module=%s "
                        "base=%p off=0x%llx symbol=%s symOff=0x%llx delta=0x%llx",
                        tag, reinterpret_cast<void*>(pc), rc,
                        pc_module[0] ? pc_module : "<unknown>", reinterpret_cast<void*>(pc_base),
                        static_cast<unsigned long long>(pc_off),
                        pc_symbol[0] ? pc_symbol : "<unknown>",
                        static_cast<unsigned long long>(sym_off),
                        static_cast<unsigned long long>(sym_delta));
}
#endif

boost::asio::io_context io_context;
static std::mutex m_asio_req;
static std::condition_variable_any cv_asio_req;
static std::atomic<u32> asio_requests;
static std::jthread service_thread;

Core::EntryParams entry_params{};

void KernelSignalRequest() {
    std::unique_lock lock{m_asio_req};
    ++asio_requests;
    cv_asio_req.notify_one();
}

static void KernelServiceThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:KernelServiceThread");

    while (!stoken.stop_requested()) {
        HLE_TRACE;
        {
            std::unique_lock lock{m_asio_req};
            Common::CondvarWait(cv_asio_req, lock, stoken, [] { return asio_requests != 0; });
        }
        if (stoken.stop_requested()) {
            break;
        }

        io_context.run();
        io_context.restart();

        asio_requests = 0;
    }
}

[[noreturn]] static PS4_SYSV_ABI void stack_chk_fail() {
#ifdef __ANDROID__
    std::uint64_t guest_return = 0;
    std::uint64_t return_off = 0;
    std::uint64_t arg0 = 0;
    char symbol[128]{};
    char module[192]{};
    const int site_rc = executor_live_get_current_hle_call_site
                            ? executor_live_get_current_hle_call_site(
                                  &guest_return, &return_off, &arg0, symbol, sizeof(symbol),
                                  module, sizeof(module))
                            : -1;
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_STACK_CHK_FAIL] guest __stack_chk_fail reached site_rc=%d "
                        "guest_return=%p returnOff=0x%llx symbol=%s module=%s arg0=%p",
                        site_rc, reinterpret_cast<void*>(guest_return),
                        static_cast<unsigned long long>(return_off),
                        symbol[0] ? symbol : "<unknown>",
                        module[0] ? module : "<unknown>", reinterpret_cast<void*>(arg0));
    ExecutorStackChkGuestRegs regs{};
    const int regs_rc = executor_lsx4_android_get_current_guest_regs
                            ? executor_lsx4_android_get_current_guest_regs(&regs, sizeof(regs))
                            : -1;
    std::uint64_t stack_ret = 0;
    std::uint64_t thunk_guest_rsp = 0;
    std::uint64_t thunk_guest_ret = 0;
    if (regs_rc == 0 && regs.regs[4] >= 0x10000) {
        std::memcpy(&stack_ret, reinterpret_cast<const void*>(regs.regs[4]), sizeof(stack_ret));
    }
    if (regs_rc == 0 && regs.regs[7] >= 0x10000) {
        std::memcpy(&thunk_guest_rsp, reinterpret_cast<const void*>(regs.regs[7] + 0x40),
                    sizeof(thunk_guest_rsp));
        if (thunk_guest_rsp >= 0x10000) {
            std::memcpy(&thunk_guest_ret, reinterpret_cast<const void*>(thunk_guest_rsp),
                        sizeof(thunk_guest_ret));
        }
    }
    std::uint64_t canary_m08 = 0;
    std::uint64_t canary_m10 = 0;
    std::uint64_t canary_m18 = 0;
    std::uint64_t canary_m20 = 0;
    std::uint64_t canary_m28 = 0;
    std::uint64_t canary_m30 = 0;
    std::uint64_t guard_value = 0;
    auto read_rbp_slot = [&](std::uint64_t offset, std::uint64_t& out) {
        if (regs_rc == 0 && regs.regs[5] >= offset + 0x10000) {
            std::memcpy(&out, reinterpret_cast<const void*>(regs.regs[5] - offset), sizeof(out));
        }
    };
    read_rbp_slot(0x08, canary_m08);
    read_rbp_slot(0x10, canary_m10);
    read_rbp_slot(0x18, canary_m18);
    read_rbp_slot(0x20, canary_m20);
    read_rbp_slot(0x28, canary_m28);
    read_rbp_slot(0x30, canary_m30);
    const auto* guard = GetStackChkGuardExportStorage();
    if (guard != nullptr) {
        std::memcpy(&guard_value, guard, sizeof(guard_value));
    }
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_STACK_CHK_REGS] regs_rc=%d rip=%p old_ip=%p rsp=%p rbp=%p "
                        "rax=%p rdi=%p rsi=%p rdx=%p rcx=%p stackRet=%p fsbase=%p gsbase=%p",
                        regs_rc, reinterpret_cast<void*>(regs.rip),
                        reinterpret_cast<void*>(regs.old_ip), reinterpret_cast<void*>(regs.regs[4]),
                        reinterpret_cast<void*>(regs.regs[5]), reinterpret_cast<void*>(regs.regs[0]),
                        reinterpret_cast<void*>(regs.regs[7]), reinterpret_cast<void*>(regs.regs[6]),
                        reinterpret_cast<void*>(regs.regs[2]), reinterpret_cast<void*>(regs.regs[1]),
                        reinterpret_cast<void*>(stack_ret), reinterpret_cast<void*>(regs.fsbase),
                        reinterpret_cast<void*>(regs.gsbase));
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_STACK_CHK_DETAIL] thunkGuestRsp=%p thunkGuestRet=%p "
                        "guardPtr=%p guard=0x%llx rbp_m08=0x%llx rbp_m10=0x%llx "
                        "rbp_m18=0x%llx rbp_m20=0x%llx rbp_m28=0x%llx rbp_m30=0x%llx",
                        reinterpret_cast<void*>(thunk_guest_rsp),
                        reinterpret_cast<void*>(thunk_guest_ret), guard,
                        static_cast<unsigned long long>(guard_value),
                        static_cast<unsigned long long>(canary_m08),
                        static_cast<unsigned long long>(canary_m10),
                        static_cast<unsigned long long>(canary_m18),
                        static_cast<unsigned long long>(canary_m20),
                        static_cast<unsigned long long>(canary_m28),
                        static_cast<unsigned long long>(canary_m30));
    LogStackChkSymbol("rip", regs.rip);
    LogStackChkSymbol("old_ip", regs.old_ip);
    LogStackChkSymbol("stack_ret", stack_ret);
    LogStackChkSymbol("thunk_guest_ret", thunk_guest_ret);
#endif
    std::abort();
    __builtin_unreachable();
}

static thread_local s32 g_posix_errno = 0;

s32* PS4_SYSV_ABI __Error() {
    return &g_posix_errno;
}

void ErrSceToPosix(s32 error) {
    g_posix_errno = error - ORBIS_KERNEL_ERROR_UNKNOWN;
}

s32 ErrnoToSceKernelError(s32 error) {
    return error + ORBIS_KERNEL_ERROR_UNKNOWN;
}

s32 PS4_SYSV_ABI sceKernelError(s32 posix_error) {
    if (posix_error == 0) {
        return 0;
    }
    return posix_error + ORBIS_KERNEL_ERROR_UNKNOWN;
}

void SetPosixErrno(s32 e) {
    switch (e) {
    case EPERM:
        g_posix_errno = POSIX_EPERM;
        break;
    case ENOENT:
        g_posix_errno = POSIX_ENOENT;
        break;
    case EDEADLK:
        g_posix_errno = POSIX_EDEADLK;
        break;
    case ENOMEM:
        g_posix_errno = POSIX_ENOMEM;
        break;
    case EACCES:
        g_posix_errno = POSIX_EACCES;
        break;
    case EFAULT:
        g_posix_errno = POSIX_EFAULT;
        break;
    case EINVAL:
        g_posix_errno = POSIX_EINVAL;
        break;
    case ENOSPC:
        g_posix_errno = POSIX_ENOSPC;
        break;
    case ERANGE:
        g_posix_errno = POSIX_ERANGE;
        break;
    case EAGAIN:
        g_posix_errno = POSIX_EAGAIN;
        break;
    case ETIMEDOUT:
        g_posix_errno = POSIX_ETIMEDOUT;
        break;
    default:
        LOG_WARNING(Kernel, "Unhandled errno {}", e);
        g_posix_errno = e;
    }
}

static u64 g_mspace_atomic_id_mask = 0;
static u64 g_mstate_table[64] = {0};

struct HeapInfoInfo {
    u64 size = sizeof(HeapInfoInfo);
    u32 flag;
    u32 getSegmentInfo;
    u64* mspace_atomic_id_mask;
    u64* mstate_table;
};

void PS4_SYSV_ABI sceLibcHeapGetTraceInfo(HeapInfoInfo* info) {
    info->mspace_atomic_id_mask = &g_mspace_atomic_id_mask;
    info->mstate_table = g_mstate_table;
    info->getSegmentInfo = 0;
}

struct OrbisKernelUuid {
    u32 timeLow;
    u16 timeMid;
    u16 timeHiAndVersion;
    u8 clockSeqHiAndReserved;
    u8 clockSeqLow;
    u8 node[6];
};
static_assert(sizeof(OrbisKernelUuid) == 0x10);

s32 PS4_SYSV_ABI sceKernelUuidCreate(OrbisKernelUuid* orbisUuid) {
    if (!orbisUuid) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
#ifdef _WIN64
    UUID uuid;
    if (UuidCreate(&uuid) != RPC_S_OK) {
        return ORBIS_KERNEL_ERROR_EFAULT;
    }
#elif defined(__ANDROID__)
    std::random_device rd;
    auto* uuid = reinterpret_cast<u8*>(orbisUuid);
    for (size_t i = 0; i < sizeof(OrbisKernelUuid); i++) {
        uuid[i] = static_cast<u8>(rd());
    }
    orbisUuid->timeHiAndVersion = (orbisUuid->timeHiAndVersion & 0x0fff) | 0x4000;
    orbisUuid->clockSeqHiAndReserved = (orbisUuid->clockSeqHiAndReserved & 0x3f) | 0x80;
    return ORBIS_OK;
#else
    uuid_t uuid;
    uuid_generate(uuid);
    std::memcpy(orbisUuid, &uuid, sizeof(OrbisKernelUuid));
    return ORBIS_OK;
#endif
}

s32 PS4_SYSV_ABI kernel_ioctl(s32 fd, u64 cmd, VA_ARGS) {
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        LOG_INFO(Lib_Kernel, "ioctl: fd = {:X} cmd = {:X} file == nullptr", fd, cmd);
        g_posix_errno = POSIX_EBADF;
        return -1;
    }
    if (file->type != Core::FileSys::FileType::Device) {
        LOG_WARNING(Lib_Kernel, "ioctl: fd = {:X} cmd = {:X} file->type != Device", fd, cmd);
        g_posix_errno = ENOTTY;
        return -1;
    }
    VA_CTX(ctx);
    s32 result = file->device->ioctl(cmd, &ctx);
    LOG_TRACE(Lib_Kernel, "ioctl: fd = {:X} cmd = {:X} result = {}", fd, cmd, result);
    if (result < 0) {
        ErrSceToPosix(result);
        return -1;
    }
    return result;
}

const char* PS4_SYSV_ABI sceKernelGetFsSandboxRandomWord() {
    const char* path = "executor";
    return path;
}

#if !defined(_WIN32)
static void OrbisSigsetToNative(const Sigset* in, sigset_t* out) {
    sigemptyset(out);
    if (!in) {
        return;
    }
    for (s32 sig = 1; sig < 128; ++sig) {
        const u64 mask = 1ULL << (sig & 63);
        if ((in->bits[sig >> 6] & mask) == 0) {
            continue;
        }
        if (sig == POSIX_SIGKILL || sig == POSIX_SIGSTOP || sig == POSIX_SIGTHR ||
            sig == POSIX_SIGLIBRT) {
            continue;
        }
        const s32 native = OrbisToNativeSignal(sig);
        if (native > 0 && native < NSIG) {
            sigaddset(out, native);
        }
    }
}

static void NativeSigsetToOrbis(const sigset_t* in, Sigset* out) {
    if (!out) {
        return;
    }
    out->bits[0] = 0;
    out->bits[1] = 0;
    if (!in) {
        return;
    }
    for (s32 native = 1; native < NSIG; ++native) {
        if (sigismember(in, native) != 1) {
            continue;
        }
        const s32 orbis = NativeToOrbisSignal(native);
        if (orbis > 0 && orbis < 128) {
            out->bits[orbis >> 6] |= 1ULL << (orbis & 63);
        }
    }
}
#endif

s32 PS4_SYSV_ABI _sigprocmask(s32 how, const Sigset* set, Sigset* oldset) {
#ifdef _WIN32
    if (oldset) {
        oldset->bits[0] = 0;
        oldset->bits[1] = 0;
    }
    return ORBIS_OK;
#else
    sigset_t native_set{};
    sigset_t native_old{};
    sigset_t* set_ptr = nullptr;
    if (set) {
        OrbisSigsetToNative(set, &native_set);
        set_ptr = &native_set;
    }

    int native_how = SIG_SETMASK;
    switch (how) {
    case 1:
        native_how = SIG_BLOCK;
        break;
    case 2:
        native_how = SIG_UNBLOCK;
        break;
    case 3:
        native_how = SIG_SETMASK;
        break;
    default:
        native_how = how;
        break;
    }

    errno = 0;
    const int rc = pthread_sigmask(native_how, set_ptr, oldset ? &native_old : nullptr);
    if (oldset) {
        NativeSigsetToOrbis(&native_old, oldset);
    }
#ifdef __ANDROID__
    static const bool trace_sigmask = std::getenv("EXECUTOR_TRACE_SIGMASK") != nullptr;
    static std::atomic_int budget{128};
    if (trace_sigmask && budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_SIGMASK] thread=%s how=%d nativeHow=%d set=%p "
                            "old=%p rc=%d errno=%d set0=0x%llx set1=0x%llx old0=0x%llx "
                            "old1=0x%llx",
                            g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>", how,
                            native_how, set, oldset, rc, errno,
                            static_cast<unsigned long long>(set ? set->bits[0] : 0),
                            static_cast<unsigned long long>(set ? set->bits[1] : 0),
                            static_cast<unsigned long long>(oldset ? oldset->bits[0] : 0),
                            static_cast<unsigned long long>(oldset ? oldset->bits[1] : 0));
    }
#endif
    if (rc != 0) {
        *__Error() = rc;
        return ORBIS_FAIL;
    }
    return ORBIS_OK;
#endif
}

s32 PS4_SYSV_ABI posix_getpagesize() {
    return 16_KB;
}

s32 PS4_SYSV_ABI sceKernelGetGPI() {
    LOG_DEBUG(Kernel, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelSetGPO() {
    LOG_DEBUG(Kernel, "called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetAllowedSdkVersionOnSystem(s32* ver) {
    if (ver == nullptr) {
        return ORBIS_KERNEL_ERROR_EINVAL;
    }
    *ver = CURRENT_FIRMWARE_VERSION | 0xfff;
    LOG_INFO(Lib_Kernel, "called, returned sw version: {}", *ver);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetSystemSwVersion(SwVersionStruct* ret) {
    if (ret == nullptr) {
        return ORBIS_OK;
    }
    u32 fake_fw = CURRENT_FIRMWARE_VERSION;
    ret->hex_representation = fake_fw;
    std::snprintf(ret->text_representation, 28, "%2x.%03x.%03x", fake_fw >> 0x18,
                  fake_fw >> 0xc & 0xfff, fake_fw & 0xfff);
    LOG_INFO(Lib_Kernel, "called, returned sw version: {}", ret->text_representation);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI getargc() {
    return entry_params.argc;
}

const char** PS4_SYSV_ABI getargv() {
    return entry_params.argv;
}

s32 PS4_SYSV_ABI get_authinfo(s32 pid, AuthInfoData* p2) {
    LOG_WARNING(Lib_Kernel, "(STUBBED) called, pid: {}", pid);
    if (p2 == nullptr) {
        *Kernel::__Error() = POSIX_EPERM;
        return -1;
    }
    if (pid != 0 && pid != GLOBAL_PID) {
        *Kernel::__Error() = POSIX_ESRCH;
        return -1;
    }

    *p2 = {};
    p2->caps[0] = 0x2000000000000000;
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelGetAppInfo(s32 pid, OrbisKernelAppInfo* app_info) {
    LOG_WARNING(Lib_Kernel, "(STUBBED) called, pid: {}", pid);
    if (pid != GLOBAL_PID) {
        return ORBIS_KERNEL_ERROR_EPERM;
    }
    if (app_info == nullptr) {
        return ORBIS_OK;
    }

    auto& game_info = Common::ElfInfo::Instance();
    *app_info = {};
    app_info->has_param_sfo = 1;
    strncpy(app_info->cusa_name, game_info.GameSerial().data(), 10);
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    service_thread = std::jthread{KernelServiceThread};

    Libraries::Kernel::RegisterFileSystem(sym);
    Libraries::Kernel::RegisterTime(sym);
    Libraries::Kernel::RegisterThreads(sym);
    Libraries::Kernel::RegisterKernelEventFlag(sym);
    Libraries::Kernel::RegisterMemory(sym);
    Libraries::Kernel::RegisterEventQueue(sym);
    Libraries::Kernel::RegisterProcess(sym);
    Libraries::Kernel::RegisterException(sym);
    Libraries::Kernel::RegisterAio(sym);
    Libraries::Kernel::RegisterDebug(sym);

#ifdef __ANDROID__
    LIB_OBJ("f7uOxY9mM1U", "libkernel", 1, "libkernel", GetStackChkGuardExportStorage());
#else
    LIB_OBJ("f7uOxY9mM1U", "libkernel", 1, "libkernel", &g_stack_chk_guard);
#endif
    LIB_FUNCTION("D4yla3vx4tY", "libkernel", 1, "libkernel", sceKernelError);
    LIB_FUNCTION("YeU23Szo3BM", "libkernel", 1, "libkernel", sceKernelGetAllowedSdkVersionOnSystem);
    LIB_FUNCTION("Mv1zUObHvXI", "libkernel", 1, "libkernel", sceKernelGetSystemSwVersion);
    LIB_FUNCTION("igMefp4SAv0", "libkernel", 1, "libkernel", get_authinfo);
    LIB_FUNCTION("G-MYv5erXaU", "libkernel", 1, "libkernel", sceKernelGetAppInfo);
    LIB_FUNCTION("PfccT7qURYE", "libkernel", 1, "libkernel", kernel_ioctl);
    LIB_FUNCTION("wW+k21cmbwQ", "libkernel", 1, "libkernel", kernel_ioctl);
    LIB_FUNCTION("JGfTMBOdUJo", "libkernel", 1, "libkernel", sceKernelGetFsSandboxRandomWord);
    LIB_FUNCTION("6xVpy0Fdq+I", "libkernel", 1, "libkernel", _sigprocmask);
    LIB_FUNCTION("Xjoosiw+XPI", "libkernel", 1, "libkernel", sceKernelUuidCreate);
    LIB_FUNCTION("Ou3iL1abvng", "libkernel", 1, "libkernel", stack_chk_fail);
    LIB_FUNCTION("9BcDykPmo1I", "libkernel", 1, "libkernel", __Error);
    LIB_FUNCTION("k+AXqu2-eBc", "libkernel", 1, "libkernel", posix_getpagesize);
    LIB_FUNCTION("k+AXqu2-eBc", "libScePosix", 1, "libkernel", posix_getpagesize);
    LIB_FUNCTION("NWtTN10cJzE", "libSceLibcInternalExt", 1, "libSceLibcInternal",
                 sceLibcHeapGetTraceInfo);

    LIB_FUNCTION("XVL8So3QJUk", "libkernel", 1, "libkernel", Libraries::Net::sys_connect);
    LIB_FUNCTION("pG70GT5yRo4", "libkernel", 1, "libkernel", Libraries::Net::sys_socketex);
    LIB_FUNCTION("KuOmgKoqCdY", "libkernel", 1, "libkernel", Libraries::Net::sys_bind);
    LIB_FUNCTION("6O8EwYOgH9Y", "libkernel", 1, "libkernel", Libraries::Net::sys_getsockopt);
    LIB_FUNCTION("fFxGkxF2bVo", "libkernel", 1, "libkernel", Libraries::Net::sys_setsockopt);
    LIB_FUNCTION("pxnCmagrtao", "libkernel", 1, "libkernel", Libraries::Net::sys_listen);
    LIB_FUNCTION("3e+4Iv7IJ8U", "libkernel", 1, "libkernel", Libraries::Net::sys_accept);
    LIB_FUNCTION("TUuiYS2kE8s", "libkernel", 1, "libkernel", Libraries::Net::sys_shutdown);
    LIB_FUNCTION("TU-d9PfIHPM", "libkernel", 1, "libkernel", Libraries::Net::sys_socket);
    LIB_FUNCTION("MZb0GKT3mo8", "libkernel", 1, "libkernel", Libraries::Net::sys_socketpair);
    LIB_FUNCTION("MZb0GKT3mo8", "libkernel_ps2emu", 1, "libkernel", Libraries::Net::sys_socketpair);
    LIB_FUNCTION("K1S8oc61xiM", "libkernel", 1, "libkernel", Libraries::Net::sys_htonl);
    LIB_FUNCTION("jogUIsOV3-U", "libkernel", 1, "libkernel", Libraries::Net::sys_htons);
    LIB_FUNCTION("fZOeZIOEmLw", "libkernel", 1, "libkernel", Libraries::Net::sys_send);
    LIB_FUNCTION("oBr313PppNE", "libkernel", 1, "libkernel", Libraries::Net::sys_sendto);
    LIB_FUNCTION("Ez8xjo9UF4E", "libkernel", 1, "libkernel", Libraries::Net::sys_recv);
    LIB_FUNCTION("lUk6wrGXyMw", "libkernel", 1, "libkernel", Libraries::Net::sys_recvfrom);

    LIB_FUNCTION("TU-d9PfIHPM", "libScePosix", 1, "libkernel", Libraries::Net::sys_socket);
    LIB_FUNCTION("fZOeZIOEmLw", "libScePosix", 1, "libkernel", Libraries::Net::sys_send);
    LIB_FUNCTION("oBr313PppNE", "libScePosix", 1, "libkernel", Libraries::Net::sys_sendto);
    LIB_FUNCTION("Ez8xjo9UF4E", "libScePosix", 1, "libkernel", Libraries::Net::sys_recv);
    LIB_FUNCTION("lUk6wrGXyMw", "libScePosix", 1, "libkernel", Libraries::Net::sys_recvfrom);
    LIB_FUNCTION("hI7oVeOluPM", "libScePosix", 1, "libkernel", Libraries::Net::sys_recvmsg);
    LIB_FUNCTION("TXFFFiNldU8", "libScePosix", 1, "libkernel", Libraries::Net::sys_getpeername);
    LIB_FUNCTION("6O8EwYOgH9Y", "libScePosix", 1, "libkernel", Libraries::Net::sys_getsockopt);
    LIB_FUNCTION("fFxGkxF2bVo", "libScePosix", 1, "libkernel", Libraries::Net::sys_setsockopt);
    LIB_FUNCTION("RenI1lL1WFk", "libScePosix", 1, "libkernel", Libraries::Net::sys_getsockname);
    LIB_FUNCTION("KuOmgKoqCdY", "libScePosix", 1, "libkernel", Libraries::Net::sys_bind);
    LIB_FUNCTION("5jRCs2axtr4", "libScePosix", 1, "libkernel",
                 Libraries::Net::sceNetInetNtop);
    LIB_FUNCTION("4n51s0zEf0c", "libScePosix", 1, "libkernel",
                 Libraries::Net::sceNetInetPton);
    LIB_FUNCTION("XVL8So3QJUk", "libScePosix", 1, "libkernel", Libraries::Net::sys_connect);
    LIB_FUNCTION("3e+4Iv7IJ8U", "libScePosix", 1, "libkernel", Libraries::Net::sys_accept);
    LIB_FUNCTION("aNeavPDNKzA", "libScePosix", 1, "libkernel", Libraries::Net::sys_sendmsg);
    LIB_FUNCTION("pxnCmagrtao", "libScePosix", 1, "libkernel", Libraries::Net::sys_listen);

    LIB_FUNCTION("4oXYe9Xmk0Q", "libkernel", 1, "libkernel", sceKernelGetGPI);
    LIB_FUNCTION("ca7v6Cxulzs", "libkernel", 1, "libkernel", sceKernelSetGPO);
    LIB_FUNCTION("iKJMWrAumPE", "libkernel", 1, "libkernel", getargc);
    LIB_FUNCTION("FJmglmTMdr4", "libkernel", 1, "libkernel", getargv);
}

}
