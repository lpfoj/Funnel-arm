// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdarg>
#include <cstdio>
#include <map>
#include <string_view>

#include <common/va_ctx.h>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/singleton.h"
#include "core/file_sys/fs.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/kernel/file_system.h"
#include "core/libraries/kernel/kernel.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/libc_internal/libc_internal_io.h"
#include "core/libraries/libc_internal/libc_internal_threads.h"
#include "core/libraries/libs.h"
#include "printf.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <array>
#include <atomic>
#include <mutex>
#include "core/aerolib/stubs.h"
#endif

namespace Libraries::LibcInternal {

#ifdef __ANDROID__
static constexpr std::size_t kFileLockStripes = 128;
static std::array<std::recursive_mutex, kFileLockStripes> g_file_lock_stripes{};

static std::recursive_mutex& FileLockFor(const OrbisFILE* file) {
    const auto key = reinterpret_cast<std::uintptr_t>(file);
    return g_file_lock_stripes[(key >> 4) % kFileLockStripes];
}
#endif

s32 PS4_SYSV_ABI internal_snprintf(char* s, u64 n, VA_ARGS) {
    VA_CTX(ctx);
    return snprintf_ctx(s, n, &ctx);
}

std::map<s32, OrbisFILE*> g_files{};
static constexpr s32 g_initial_files = 5;
static constexpr s32 g_max_files = 0x100;

#ifdef __ANDROID__
namespace {

std::atomic<int> g_executor_libc_io_log_budget{1024};

bool ExecutorTraceLibcIoPath(std::string_view path) {
    return path.find("/mnt/sandbox/pfsmnt/NPXS39041-app0/assets/") != std::string_view::npos ||
           path.find("/user/app/NPXS39041/storedata/") != std::string_view::npos ||
           path.find("/user/appmeta/NPXS39041/") != std::string_view::npos ||
           path.find("/user/appmeta/external/NPXS39041/") != std::string_view::npos;
}

const Core::FileSys::File* ExecutorLibcIoFile(const OrbisFILE* file) {
    if (file == nullptr || file->_Handle < 0) {
        return nullptr;
    }
    return Common::Singleton<Core::FileSys::HandleTable>::Instance()->GetFile(file->_Handle);
}

bool ExecutorTraceLibcIoFile(const OrbisFILE* file) {
    const auto* fs_file = ExecutorLibcIoFile(file);
    return fs_file != nullptr && ExecutorTraceLibcIoPath(fs_file->m_guest_name);
}

void ExecutorLibcIoLog(const char* format, ...) {
    if (g_executor_libc_io_log_budget.fetch_sub(1, std::memory_order_relaxed) <= 0) {
        return;
    }
    va_list args;
    va_start(args, format);
    va_list stderr_args;
    va_copy(stderr_args, args);
    __android_log_vprint(ANDROID_LOG_INFO, "LSX4Native", format, args);
    std::vfprintf(stderr, format, stderr_args);
    std::fputc('\n', stderr);
    std::fflush(stderr);
    va_end(stderr_args);
    va_end(args);
}

}
#endif

OrbisFILE* PS4_SYSV_ABI internal__Fofind() {
    u64 index = g_initial_files;
    while (index != g_max_files) {
        OrbisFILE* file = g_files[index];
        if (file == nullptr) {
#ifdef __ANDROID__
            file = reinterpret_cast<OrbisFILE*>(
                Core::AeroLib::ExecutorLibcCalloc(1, sizeof(OrbisFILE)));
#else
            file = new OrbisFILE();
#endif
            if (file == nullptr) {
                return nullptr;
            }
            g_files[index] = file;
            file->_Mode = 0x80;
            file->_Idx = index;
            return file;
        }
        if (file->_Mode == 0) {
            file->_Mode = 0xff7f;
            return file;
        }
        index++;
    }
    return nullptr;
}

void PS4_SYSV_ABI internal__Lockfilelock(OrbisFILE* file) {
#ifdef __ANDROID__
    if (file != nullptr) {
        FileLockFor(file).lock();
    }
#else
    if (file != nullptr && file->_Mutex != nullptr) {
        internal__Mtxlock(&file->_Mutex);
    }
#endif
}

void PS4_SYSV_ABI internal__Unlockfilelock(OrbisFILE* file) {
#ifdef __ANDROID__
    if (file != nullptr) {
        FileLockFor(file).unlock();
    }
#else
    if (file != nullptr && file->_Mutex != nullptr) {
        internal__Mtxunlock(&file->_Mutex);
    }
#endif
}

OrbisFILE* PS4_SYSV_ABI internal__Foprep(const char* path, const char* mode, OrbisFILE* file,
                                         s32 fd, s32 s_mode, s32 flag) {
    if (file == nullptr) {
        *Kernel::__Error() = POSIX_ENOMEM;
#ifdef __ANDROID__
        return nullptr;
#endif
    }

#ifndef __ANDROID__
    Libraries::Kernel::PthreadMutexT mtx = file->_Mutex;
    Libraries::Kernel::PthreadMutexT* mtx_ptr = &file->_Mutex;
#endif
    u8 file_index = file->_Idx;
    u16 file_mode = file->_Mode & 0x80;

    memset(file, 0, sizeof(OrbisFILE));
    file->_Handle = -1;

    u8* ptr = &file->_Cbuf;
    file->_Mutex = nullptr;
    file->_Idx = file_index;
    file->_Buf = ptr;
    file->_Bend = &file->unk2;
    file->_Next = ptr;
    file->_Rend = ptr;
    file->_WRend = ptr;
    file->_Wend = ptr;
    file->_WWend = ptr;
    file->_Rback = ptr;
    file->_WRback = &file->unk1;

    const char* mode_str = mode;
    u16 calc_mode = 0;
    u16 access_mode = 0;
    if (mode_str[0] == 'r') {
        calc_mode = 1 | file_mode;
    } else if (mode_str[0] == 'w') {
        calc_mode = 0x1a | file_mode;
    } else if (mode_str[0] == 'a') {
        calc_mode = 0x16 | file_mode;
    } else {
        file->_Mode = file_mode;
#ifdef __ANDROID__
        file->_Mutex = nullptr;
        if (flag != 0) {
            internal__Unlockfilelock(file);
        }
#else
        if (flag == 0) {
            internal__Mtxinit(mtx_ptr, nullptr);
        } else {
            file->_Mutex = mtx;
            internal__Unlockfilelock(file);
        }
#endif
        internal_fclose(file);
        *Kernel::__Error() = POSIX_EINVAL;
        return nullptr;
    }
    file->_Mode = calc_mode;

    do {
        if (mode_str[1] == '+') {
            file_mode = 3;
            if ((~calc_mode & 3) == 0) {
                break;
            }
        } else if (mode_str[1] != 'b') {
            file_mode = 0x20;
            if ((calc_mode & 0x20) != 0) {
                break;
            }
        }
        mode_str++;
        calc_mode = file_mode | calc_mode;
        file->_Mode = calc_mode;
    } while (true);

    if (path == nullptr && fd >= 0) {
        file->_Handle = fd;
    } else {
        fd = internal__Fopen(path, calc_mode, s_mode == 0x55);
        file->_Handle = fd;
    }

    if (fd < 0) {
#ifdef __ANDROID__
        file->_Mutex = nullptr;
        if (flag != 0) {
            internal__Unlockfilelock(file);
        }
#else
        if (flag == 0) {
            internal__Mtxinit(mtx_ptr, nullptr);
        } else {
            file->_Mutex = mtx;
            internal__Unlockfilelock(file);
        }
#endif
        s32 old_errno = *Kernel::__Error();
        internal_fclose(file);
        *Kernel::__Error() = old_errno;
        return nullptr;
    }

#ifdef __ANDROID__
    file->_Mutex = nullptr;
#else
    if (flag == 0) {
        char mtx_name[0x20];
        std::snprintf(mtx_name, 0x20, "FileFD:0x%08X", fd);
        internal__Mtxinit(mtx_ptr, mtx_name);
    } else {
        file->_Mutex = mtx;
    }
#endif
    return file;
}

s32 PS4_SYSV_ABI internal__Fopen(const char* path, u16 mode, bool flag) {
    u32 large_mode = mode;
    u16 open_mode = 0600;
    if (!flag) {
        open_mode = 0666;
    }
    s32 creat_flag = large_mode << 5 & 0x200;
    s32 excl_flag = large_mode << 5 & 0x800;
    s32 misc_flags = (large_mode & 8) * 0x80 + (large_mode & 4) * 2;
    s32 access_flag = std::max<s32>((large_mode & 3) - 1, 0);
    s32 open_flags = creat_flag | misc_flags | excl_flag | access_flag;
#ifdef __ANDROID__
    const bool trace = path != nullptr && ExecutorTraceLibcIoPath(path);
    if (trace) {
        ExecutorLibcIoLog(
            "[EXECUTOR_LIBC_IO] _Fopen path=%s mode=0x%x flag=%d open_flags=0x%x open_mode=0%o",
            path, mode, flag ? 1 : 0, open_flags, open_mode);
    }
#endif
    const s32 fd = Libraries::Kernel::posix_open(path, open_flags, open_mode);
#ifdef __ANDROID__
    if (trace) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] _Fopen_result path=%s fd=%d errno=%d", path, fd,
                          *Kernel::__Error());
    }
#endif
    return fd;
}

OrbisFILE* PS4_SYSV_ABI internal_fopen(const char* path, const char* mode) {
    std::scoped_lock lk{g_file_mtx};
    LOG_INFO(Lib_LibcInternal, "called, path {}, mode {}", path, mode);
#ifdef __ANDROID__
    const bool trace = path != nullptr && ExecutorTraceLibcIoPath(path);
    if (trace) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] fopen path=%s mode=%s", path,
                          mode ? mode : "<null>");
    }
#endif
    OrbisFILE* file = internal__Fofind();
    OrbisFILE* result = internal__Foprep(path, mode, file, -1, 0, 0);
#ifdef __ANDROID__
    if (trace) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] fopen_result path=%s file=%p fd=%d errno=%d", path,
                          result, result ? result->_Handle : -1, *Kernel::__Error());
    }
#endif
    return result;
}

s32 PS4_SYSV_ABI internal_fflush(OrbisFILE* file) {
    if (file == nullptr) {
        std::scoped_lock lk{g_file_mtx};
        s32 fflush_result = 0;
        for (auto& file : g_files) {
            s32 res = internal_fflush(file.second);
            if (res < 0) {
                fflush_result = -1;
            }
        }
        return fflush_result;
    }
    if ((file->_Mode & 0x2000) != 0) {
        internal__Lockfilelock(file);
        u16 file_mode = file->_Mode;
        u8* file_buf_start = file->_Buf;
        u8* file_buf_end = file->_Next;
        while (file_buf_start < file_buf_end) {
            u64 size_to_write = static_cast<u64>(file_buf_end - file_buf_start);
            s32 write_bytes =
                Libraries::Kernel::sceKernelWrite(file->_Handle, file_buf_start, size_to_write);
            if (write_bytes < 1) {
                file_buf_start = file->_Buf;
                file->_Next = file_buf_start;
                file->_Wend = file_buf_start;
                file->_WWend = file_buf_start;
                u8* off_mode = reinterpret_cast<u8*>(&file->_Mode) + 1;
                *off_mode = *off_mode | 2;
                internal__Unlockfilelock(file);
                return -1;
            }
            file_buf_end = file->_Next;
            file_buf_start += write_bytes;
        }
        file->_Next = file_buf_start;
        file->_Wend = file_buf_start;
        file->_WWend = file_buf_start;
        file->_Mode = file_mode & 0xdfff;
        internal__Unlockfilelock(file);
    }
    return 0;
}

s64 PS4_SYSV_ABI internal__Nnl(OrbisFILE* file, u8* val1, u8* val2) {
    if (val1 < val2) {
        return val2 - val1;
    }
    return 0;
}

s32 PS4_SYSV_ABI internal__Fspos(OrbisFILE* file, Orbisfpos_t* file_pos, s64 offset, s32 whence) {
    if ((file->_Mode & 3) == 0) {
        return -1;
    }
    if (internal_fflush(file) != 0) {
        return -1;
    }
    if (whence >= 3) {
        *Libraries::Kernel::__Error() = POSIX_EINVAL;
        return -1;
    }
    if (file_pos != nullptr) {
        offset = offset + file_pos->_Off;
    }
    if (whence == 1 && (file->_Mode & 0x1000) != 0) {
        s64 val1 = internal__Nnl(file, file->_Rback, &file->_Cbuf);
        u8* rsave_ptr = file->_Rsave;
        if (rsave_ptr == nullptr) {
            rsave_ptr = file->_Rend;
        }
        s64 val2 = internal__Nnl(file, file->_Next, rsave_ptr);
        s64 val3 = internal__Nnl(file, file->_Next, file->_WRend);
        offset = offset - (val1 + val2 + val3);
    }
    s64 result = 0;
    if (whence == 2 || (whence == 1 && offset != 0) || (whence == 0 && offset != -1)) {
        result = Libraries::Kernel::posix_lseek(file->_Handle, offset, whence);
    }
    if (result == -1) {
        return -1;
    }

    u16 file_mode = file->_Mode;
    if ((file_mode & 0x3000) != 0) {
        u8* file_buf = file->_Buf;
        file->_Next = file_buf;
        file->_Rend = file_buf;
        file->_WRend = file_buf;
        file->_Wend = file_buf;
        file->_WWend = file_buf;
        file->_Rback = &file->_Cbuf;
        file->_WRback = &file->unk1;
        file->_Rsave = nullptr;
    }
    if (file_pos != nullptr) {
        std::memcpy(&file->_Wstate, &file_pos->_Wstate, sizeof(Orbis_Mbstatet));
    }
    file->_Mode = file_mode & 0xceff;
    return 0;
}

s32 PS4_SYSV_ABI internal_fseek(OrbisFILE* file, s64 offset, s32 whence) {
    internal__Lockfilelock(file);
    LOG_TRACE(Lib_LibcInternal, "called, file handle {:#x}, offset {:#x}, whence {:#x}",
              file->_Handle, offset, whence);
#ifdef __ANDROID__
    const auto* fs_file = ExecutorLibcIoFile(file);
    const bool trace = fs_file != nullptr && ExecutorTraceLibcIoPath(fs_file->m_guest_name);
    if (trace) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] fseek fd=%d guest=%s offset=%lld whence=%d",
                          file->_Handle, fs_file->m_guest_name.c_str(),
                          static_cast<long long>(offset), whence);
    }
#endif
    s32 result = internal__Fspos(file, nullptr, offset, whence);
#ifdef __ANDROID__
    if (trace) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] fseek_result fd=%d result=%d errno=%d",
                          file->_Handle, result, *Kernel::__Error());
    }
#endif
    internal__Unlockfilelock(file);
    return result;
}

s64 PS4_SYSV_ABI internal_ftell(OrbisFILE* file) {
    if (file == nullptr || file->_Handle < 0) {
        *Kernel::__Error() = POSIX_EBADF;
        return -1;
    }
    internal__Lockfilelock(file);
    s64 result = Libraries::Kernel::posix_lseek(file->_Handle, 0, 1);
    if (result >= 0 && (file->_Mode & 0x1000) != 0) {
        u8* read_end = file->_Rsave != nullptr ? file->_Rsave : file->_Rend;
        if (read_end > file->_Next) {
            result -= read_end - file->_Next;
        }
        if ((file->_Mode & 0x4000) != 0 && file->_Rback < &file->_Cbuf) {
            result -= &file->_Cbuf - file->_Rback;
        }
    }
    LOG_TRACE(Lib_LibcInternal, "called, file handle {:#x}, result {:#x}", file->_Handle, result);
#ifdef __ANDROID__
    if (ExecutorTraceLibcIoFile(file)) {
        const auto* fs_file = ExecutorLibcIoFile(file);
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] ftell fd=%d guest=%s result=%lld errno=%d",
                          file->_Handle, fs_file ? fs_file->m_guest_name.c_str() : "<unknown>",
                          static_cast<long long>(result), *Kernel::__Error());
    }
#endif
    internal__Unlockfilelock(file);
    return result;
}

s32 PS4_SYSV_ABI internal__Frprep(OrbisFILE* file) {
#ifdef __ANDROID__
    const auto* trace_file = ExecutorLibcIoFile(file);
    const bool trace = trace_file != nullptr && ExecutorTraceLibcIoPath(trace_file->m_guest_name);
    if (trace) {
        ExecutorLibcIoLog(
            "[EXECUTOR_LIBC_IO] frprep_enter fd=%d guest=%s mode=0x%x next=%p rend=%p buf=%p bend=%p",
            file->_Handle, trace_file->m_guest_name.c_str(), file->_Mode, file->_Next,
            file->_Rend, file->_Buf, file->_Bend);
    }
#endif
    if (file->_Rend > file->_Next) {
#ifdef __ANDROID__
        if (trace) {
            ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] frprep_buffered fd=%d guest=%s available=%llu",
                              file->_Handle, trace_file->m_guest_name.c_str(),
                              static_cast<unsigned long long>(file->_Rend - file->_Next));
        }
#endif
        return 1;
    }
    if ((file->_Mode & 0x100) != 0) {
#ifdef __ANDROID__
        if (trace) {
            ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] frprep_eof_or_error fd=%d guest=%s mode=0x%x",
                              file->_Handle, trace_file->m_guest_name.c_str(), file->_Mode);
        }
#endif
        return 0;
    }
    u16 mode = file->_Mode;
    if ((mode & 0xa001) != 1) {
        file->_Mode = (((mode ^ 0x8000) >> 0xf) << 0xe) | mode | 0x200;
#ifdef __ANDROID__
        if (trace) {
            ExecutorLibcIoLog(
                "[EXECUTOR_LIBC_IO] frprep_reject_mode fd=%d guest=%s old_mode=0x%x new_mode=0x%x",
                file->_Handle, trace_file->m_guest_name.c_str(), mode, file->_Mode);
        }
#endif
        return -1;
    }

    u8* file_buf = file->_Buf;
    if ((mode & 0x800) == 0 && file_buf == &file->_Cbuf) {
#ifdef __ANDROID__
        u8* new_buffer = reinterpret_cast<u8*>(Core::AeroLib::ExecutorLibcMalloc(0x10000));
#else
        u8* new_buffer = std::bit_cast<u8*>(std::malloc(0x10000));
#endif
        if (new_buffer == nullptr) {
            file->_Buf = file_buf;
            file->_Bend = file_buf + 1;
        } else {
            file->_Mode = file->_Mode | 0x40;
            file->_Buf = new_buffer;
            file->_Bend = new_buffer + 0x10000;
            file->_WRend = new_buffer;
            file->_WWend = new_buffer;
            file_buf = new_buffer;
        }
#ifdef __ANDROID__
        if (trace) {
            ExecutorLibcIoLog(
                "[EXECUTOR_LIBC_IO] frprep_buffer_alloc fd=%d guest=%s host_buffer=%p bend=%p ok=%d",
                file->_Handle, trace_file->m_guest_name.c_str(), file->_Buf, file->_Bend,
                new_buffer != nullptr ? 1 : 0);
        }
#endif
    }
    file->_Next = file_buf;
    file->_Rend = file_buf;
    file->_Wend = file_buf;
    const auto read_request = static_cast<u64>(file->_Bend - file_buf);
#ifdef __ANDROID__
    if (trace) {
        ExecutorLibcIoLog(
            "[EXECUTOR_LIBC_IO] frprep_read_request fd=%d guest=%s request=%llu buf=%p bend=%p",
            file->_Handle, trace_file->m_guest_name.c_str(),
            static_cast<unsigned long long>(read_request), file_buf, file->_Bend);
    }
#endif
    s32 read_result =
        Libraries::Kernel::sceKernelRead(file->_Handle, file_buf, read_request);
#ifdef __ANDROID__
    if (trace) {
        ExecutorLibcIoLog(
            "[EXECUTOR_LIBC_IO] frprep_read_result fd=%d guest=%s result=%d errno=%d mode=0x%x",
            file->_Handle, trace_file->m_guest_name.c_str(), read_result, *Kernel::__Error(),
            file->_Mode);
    }
#endif
    if (read_result < 0) {
        u8* off_mode = reinterpret_cast<u8*>(&file->_Mode) + 1;
        *off_mode = *off_mode | 0x42;
        return -1;
    } else if (read_result != 0) {
        file->_Mode = file->_Mode | 0x5000;
        file->_Rend = file->_Rend + read_result;
        return 1;
    }
    file->_Mode = (file->_Mode & 0xaeff) | 0x4100;
    return 0;
}

u64 PS4_SYSV_ABI internal_fread(char* ptr, u64 size, u64 nmemb, OrbisFILE* file) {
    if (size == 0 || nmemb == 0) {
        return 0;
    }
    internal__Lockfilelock(file);
    LOG_TRACE(Lib_LibcInternal, "called, file handle {:#x}, size {:#x}, nmemb {:#x}", file->_Handle,
              size, nmemb);
#ifdef __ANDROID__
    const auto* fs_file = ExecutorLibcIoFile(file);
    const bool trace = fs_file != nullptr && ExecutorTraceLibcIoPath(fs_file->m_guest_name);
    if (trace) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] fread fd=%d guest=%s size=%llu nmemb=%llu ptr=%p",
                          file->_Handle, fs_file->m_guest_name.c_str(),
                          static_cast<unsigned long long>(size),
                          static_cast<unsigned long long>(nmemb), ptr);
    }
#endif
    s64 total_size = size * nmemb;
    s64 remaining_size = total_size;
    if ((file->_Mode & 0x4000) != 0) {
        while (remaining_size != 0) {
            u8* rback_ptr = file->_Rback;
            if (&file->_Cbuf <= rback_ptr) {
                break;
            }
            file->_Rback = rback_ptr + 1;
            *ptr = *rback_ptr;
            ptr++;
            remaining_size--;
        }
    }

    while (remaining_size != 0) {
        u8* file_ptr = file->_Rsave;
        if (file_ptr == nullptr) {
            file_ptr = file->_Rend;
        } else {
            file->_Rend = file_ptr;
            file->_Rsave = nullptr;
        }
        u8* src = file->_Next;
        if (file_ptr <= src) {
            s32 res = internal__Frprep(file);
            if (res < 1) {
                internal__Unlockfilelock(file);
                const u64 items = (total_size - remaining_size) / size;
#ifdef __ANDROID__
                if (trace) {
                    ExecutorLibcIoLog(
                        "[EXECUTOR_LIBC_IO] fread_result fd=%d guest=%s items=%llu bytes=%llu short=1 errno=%d",
                        file->_Handle, fs_file->m_guest_name.c_str(),
                        static_cast<unsigned long long>(items),
                        static_cast<unsigned long long>(total_size - remaining_size),
                        *Kernel::__Error());
                }
#endif
                return items;
            }
            src = file->_Next;
            file_ptr = file->_Rend;
        }
        u64 copy_bytes = std::min<u64>(file_ptr - src, remaining_size);
        std::memcpy(ptr, src, copy_bytes);
        file->_Next += copy_bytes;
        ptr += copy_bytes;
        remaining_size -= copy_bytes;
    }
    internal__Unlockfilelock(file);
    const u64 items = (total_size - remaining_size) / size;
#ifdef __ANDROID__
    if (trace) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] fread_result fd=%d guest=%s items=%llu bytes=%llu errno=%d",
                          file->_Handle, fs_file->m_guest_name.c_str(),
                          static_cast<unsigned long long>(items),
                          static_cast<unsigned long long>(total_size - remaining_size),
                          *Kernel::__Error());
    }
#endif
    return items;
}

u64 PS4_SYSV_ABI internal_fwrite(const char* ptr, u64 size, u64 nmemb, OrbisFILE* file) {
    if (size == 0 || nmemb == 0) {
        return 0;
    }
    if (file == nullptr || file->_Handle < 0) {
        *Kernel::__Error() = POSIX_EBADF;
        return 0;
    }
    internal__Lockfilelock(file);
    const u64 total_size = size * nmemb;
    u64 written = 0;
    while (written < total_size) {
        const auto chunk = total_size - written;
        const s64 result =
            Libraries::Kernel::sceKernelWrite(file->_Handle, ptr + written, chunk);
        if (result <= 0) {
            break;
        }
        written += static_cast<u64>(result);
    }
    internal__Unlockfilelock(file);
    return written / size;
}

s32 PS4_SYSV_ABI internal_fgetc(OrbisFILE* file) {
    unsigned char ch = 0;
    const s32 result = internal_fread(reinterpret_cast<char*>(&ch), 1, 1, file) == 1 ? ch : -1;
#ifdef __ANDROID__
    if (ExecutorTraceLibcIoFile(file)) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] fgetc fd=%d result=%d", file ? file->_Handle : -1,
                          result);
    }
#endif
    return result;
}

s32 PS4_SYSV_ABI internal_ungetc(s32 ch, OrbisFILE* file) {
    if (file == nullptr || ch == -1) {
        return -1;
    }
    internal__Lockfilelock(file);
    if (file->_Rback <= file->_Back) {
        internal__Unlockfilelock(file);
        return -1;
    }
    --file->_Rback;
    *file->_Rback = static_cast<u8>(ch);
    file->_Mode |= 0x4000;
    internal__Unlockfilelock(file);
    return static_cast<u8>(ch);
}

s32 PS4_SYSV_ABI internal_feof(OrbisFILE* file) {
    if (file == nullptr) {
        return 0;
    }
    return (file->_Mode & 0x0100) != 0 ? 1 : 0;
}

s32 PS4_SYSV_ABI internal_ferror(OrbisFILE* file) {
    if (file == nullptr) {
        return 0;
    }
    return (file->_Mode & 0x0200) != 0 ? 1 : 0;
}

void PS4_SYSV_ABI internal_rewind(OrbisFILE* file) {
    if (file == nullptr) {
        return;
    }
#ifdef __ANDROID__
    if (ExecutorTraceLibcIoFile(file)) {
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] rewind fd=%d", file->_Handle);
    }
#endif
    internal_fseek(file, 0, 0);
    file->_Mode &= static_cast<u16>(~0x0300u);
}

s32 PS4_SYSV_ABI internal_fopen_s(OrbisFILE** out_file, const char* path, const char* mode) {
    if (out_file == nullptr) {
        *Kernel::__Error() = POSIX_EFAULT;
        return POSIX_EFAULT;
    }
    *out_file = internal_fopen(path, mode);
    return *out_file != nullptr ? 0 : *Kernel::__Error();
}

void PS4_SYSV_ABI internal__Fofree(OrbisFILE* file) {
    u8* cbuf_ptr = &file->_Cbuf;
    s8 trunc_mode = static_cast<s8>(file->_Mode);

    file->_Mode = 0;
    file->_Handle = -1;
    file->_Buf = cbuf_ptr;
    file->_Next = cbuf_ptr;
    file->_Rend = cbuf_ptr;
    file->_WRend = cbuf_ptr;
    file->_Wend = cbuf_ptr;
    file->_WWend = cbuf_ptr;
    file->_Rback = cbuf_ptr;
    file->_WRback = &file->unk1;
#ifdef __ANDROID__
    file->_Mutex = nullptr;
#endif
    if (trunc_mode < 0) {
        g_files.erase(file->_Idx);
#ifdef __ANDROID__
        Core::AeroLib::ExecutorLibcFree(reinterpret_cast<u64>(file));
#else
        internal__Mtxdst(&file->_Mutex);
        free(file);
#endif
    }
}

s32 PS4_SYSV_ABI internal_fclose(OrbisFILE* file) {
    if (file == nullptr) {
        return -1;
    }

    LOG_INFO(Lib_LibcInternal, "called, file handle {:#x}", file->_Handle);
#ifdef __ANDROID__
    if (ExecutorTraceLibcIoFile(file)) {
        const auto* fs_file = ExecutorLibcIoFile(file);
        ExecutorLibcIoLog("[EXECUTOR_LIBC_IO] fclose fd=%d guest=%s", file->_Handle,
                          fs_file ? fs_file->m_guest_name.c_str() : "<unknown>");
    }
#endif
    if ((file->_Mode & 3) == 0 || file->_Handle < 0) {
        std::scoped_lock lk{g_file_mtx};
        internal__Fofree(file);
        *Libraries::Kernel::__Error() = POSIX_EBADF;
    } else {
        s32 fflush_result = internal_fflush(file);
        std::scoped_lock lk{g_file_mtx};
        if ((file->_Mode & 0x40) != 0) {
#ifdef __ANDROID__
            Core::AeroLib::ExecutorLibcFree(reinterpret_cast<u64>(file->_Buf));
#else
            std::free(file->_Buf);
#endif
        }
        file->_Buf = nullptr;
        s32 close_result = Libraries::Kernel::posix_close(file->_Handle);
        internal__Fofree(file);
        return ~-(close_result == 0) | fflush_result;
    }
    return 0;
}

void RegisterlibSceLibcInternalIo(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("eLdDw6l0-bU", "libSceLibcInternal", 1, "libSceLibcInternal", internal_snprintf);
    LIB_FUNCTION("MUjC4lbHrK4", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fflush);
    LIB_FUNCTION("xGT4Mc55ViQ", "libSceLibcInternal", 1, "libSceLibcInternal", internal__Fofind);
    LIB_FUNCTION("dREVnZkAKRE", "libSceLibcInternal", 1, "libSceLibcInternal", internal__Foprep);
    LIB_FUNCTION("sQL8D-jio7U", "libSceLibcInternal", 1, "libSceLibcInternal", internal__Fopen);
    LIB_FUNCTION("A+Y3xfrWLLo", "libSceLibcInternal", 1, "libSceLibcInternal", internal__Fspos);
    LIB_FUNCTION("Ss3108pBuZY", "libSceLibcInternal", 1, "libSceLibcInternal", internal__Nnl);
    LIB_FUNCTION("9s3P+LCvWP8", "libSceLibcInternal", 1, "libSceLibcInternal", internal__Frprep);
    LIB_FUNCTION("jVDuvE3s5Bs", "libSceLibcInternal", 1, "libSceLibcInternal", internal__Fofree);
    LIB_FUNCTION("vZkmJmvqueY", "libSceLibcInternal", 1, "libSceLibcInternal",
                 internal__Lockfilelock);
    LIB_FUNCTION("0x7rx8TKy2Y", "libSceLibcInternal", 1, "libSceLibcInternal",
                 internal__Unlockfilelock);
}

void ForceRegisterlibSceLibcInternalIo(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("xeYO4u7uyJ0", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fopen);
    LIB_FUNCTION("rQFVBXp-Cxg", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fseek);
    LIB_FUNCTION("Qazy8LmXTvw", "libSceLibcInternal", 1, "libSceLibcInternal", internal_ftell);
    LIB_FUNCTION("5qP1iVQkdck", "libSceLibcInternal", 1, "libSceLibcInternal", internal_ftell);
    LIB_FUNCTION("pkYiKw09PRA", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fseek);
    LIB_FUNCTION("lbB+UlZqVG0", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fread);
    LIB_FUNCTION("MpxhMh8QFro", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fwrite);
    LIB_FUNCTION("AEuF3F2f8TA", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fgetc);
    LIB_FUNCTION("8Q60JLJ6Rv4", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fgetc);
    LIB_FUNCTION("-LFO7jhD5CE", "libSceLibcInternal", 1, "libSceLibcInternal", internal_ungetc);
    LIB_FUNCTION("LxcEU+ICu8U", "libSceLibcInternal", 1, "libSceLibcInternal", internal_feof);
    LIB_FUNCTION("NuydofHcR1w", "libSceLibcInternal", 1, "libSceLibcInternal", internal_feof);
    LIB_FUNCTION("AHxyhN96dy4", "libSceLibcInternal", 1, "libSceLibcInternal", internal_ferror);
    LIB_FUNCTION("yxbGzBQC5xA", "libSceLibcInternal", 1, "libSceLibcInternal", internal_ferror);
    LIB_FUNCTION("3QIPIh-GDjw", "libSceLibcInternal", 1, "libSceLibcInternal", internal_rewind);
    LIB_FUNCTION("NL836gOLANs", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fopen_s);
    LIB_FUNCTION("uodLYyUip20", "libSceLibcInternal", 1, "libSceLibcInternal", internal_fclose);
}

}
