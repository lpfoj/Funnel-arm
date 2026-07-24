// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <ranges>
#include <magic_enum/magic_enum.hpp>

#include "common/assert.h"
#include "common/error.h"
#include "common/logging/log.h"
#include "common/scope_exit.h"
#include "common/singleton.h"
#include "core/file_sys/devices/console_device.h"
#include "core/file_sys/devices/deci_tty6_device.h"
#include "core/file_sys/devices/logger.h"
#include "core/file_sys/devices/nop_device.h"
#include "core/file_sys/devices/random_device.h"
#include "core/file_sys/devices/rng_device.h"
#include "core/file_sys/devices/srandom_device.h"
#include "core/file_sys/devices/urandom_device.h"
#include "core/file_sys/directories/normal_directory.h"
#include "core/file_sys/directories/pfs_directory.h"
#include "core/file_sys/fs.h"
#include "core/libraries/kernel/file_system.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/posix_error.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/libs.h"
#include "core/libraries/network/sockets.h"
#include "core/memory.h"
#include "kernel.h"

#ifdef _WIN32
#include <io.h>
#include <winsock2.h>
#else
#include <sys/select.h>
#include <sys/stat.h>
#endif

#ifdef __ANDROID__
#include <android/log.h>
extern "C" void executor_install_manager_note_local_package_open(const char* guest_path);
extern "C" void executor_store_note_stdout_write(const char* text, std::size_t size);

static void ExecutorLogKernelWriteText(s32 fd, const void* buf, u64 nbytes) {
    if (buf == nullptr || nbytes == 0 || nbytes > (1024ULL * 1024ULL)) {
        return;
    }
    const auto size = static_cast<std::size_t>(std::min<u64>(nbytes, 1024));
    const auto* src = static_cast<const unsigned char*>(buf);
    char text[1025]{};
    for (std::size_t i = 0; i < size; ++i) {
        const unsigned char c = src[i];
        if (c == '\n' || c == '\r' || c == '\t') {
            text[i] = static_cast<char>(c);
        } else if (std::isprint(c) != 0) {
            text[i] = static_cast<char>(c);
        } else {
            text[i] = '.';
        }
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_KERNEL_WRITE] fd=%d size=%llu text=\"%s\"",
                        static_cast<int>(fd), static_cast<unsigned long long>(nbytes), text);
}
#endif

namespace D = Core::Devices;
namespace fs = std::filesystem;
using FactoryDevice = std::function<std::shared_ptr<D::BaseDevice>(u32, const char*, int, u16)>;

#define GET_DEVICE_FD(fd)                                                                          \
    [](u32, const char*, int, u16) {                                                               \
        return Common::Singleton<Core::FileSys::HandleTable>::Instance()->GetFile(fd)->device;     \
    }

static std::map<std::string, FactoryDevice> available_device = {
    {"/dev/stdin", GET_DEVICE_FD(0)},
    {"/dev/stdout", GET_DEVICE_FD(1)},
    {"/dev/stderr", GET_DEVICE_FD(2)},

    {"/dev/fd/0", GET_DEVICE_FD(0)},
    {"/dev/fd/1", GET_DEVICE_FD(1)},
    {"/dev/fd/2", GET_DEVICE_FD(2)},

    {"/dev/deci_stdin", GET_DEVICE_FD(0)},
    {"/dev/deci_stdout", GET_DEVICE_FD(1)},
    {"/dev/deci_stderr", GET_DEVICE_FD(2)},

    {"/dev/null", GET_DEVICE_FD(0)},

    {"/dev/urandom",  &D::URandomDevice::Create },
    {"/dev/random",   &D::RandomDevice::Create },
    {"/dev/srandom",  &D::SRandomDevice::Create },
    {"/dev/console",  &D::ConsoleDevice::Create },
    {"/dev/deci_tty6",&D::DeciTty6Device::Create },
    {"/dev/rng",      &D::RngDevice::Create },
};

namespace Libraries::Kernel {

#ifdef __ANDROID__
namespace {

void AndroidFsLog(const char* format, ...) {
    va_list args;
    va_start(args, format);
    __android_log_vprint(ANDROID_LOG_INFO, "LSX4Native", format, args);
    va_end(args);
}

bool ShouldTraceLiveFileIo() {
    return std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr ||
           std::getenv("EXECUTOR_TRACE_LIVE_FILE") != nullptr;
}

bool ShouldTraceLiveFileErrors() {
    return ShouldTraceLiveFileIo() ||
           std::getenv("EXECUTOR_TRACE_LIVE_FILE_ERRORS") != nullptr;
}

bool ShouldTraceGuestStdout() {
    return std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr ||
           std::getenv("EXECUTOR_TRACE_STDOUT") != nullptr;
}

bool ShouldStoreGuestStdout(std::string_view preview) {
    if (preview.find("Forcing submitDone to avoid TRC R4089 breach") != std::string_view::npos) {
        return ShouldTraceGuestStdout();
    }
    return true;
}

bool IsPs4StoreCoverPath(std::string_view path) {
    return path.find("/user/app/NPXS39041/storedata/") != std::string_view::npos &&
           path.ends_with("_cover.png");
}

bool ShouldLogAndroidFsPath(std::string_view path) {
    if (!ShouldTraceLiveFileErrors()) {
        return false;
    }
    if (!IsPs4StoreCoverPath(path)) {
        return true;
    }
    static int cover_log_budget = 48;
    return cover_log_budget-- > 0;
}

bool ShouldTraceAndroidAssetPath(std::string_view path) {
    const bool live_wide = ShouldTraceLiveFileErrors();
    const bool high_value_runtime_path =
        path.find("archive.psarc") != std::string_view::npos ||
        path.find("mono-ps4") != std::string_view::npos ||
        path.find("/sce_module/") != std::string_view::npos;
    if (high_value_runtime_path) {
        return live_wide;
    }
    if (live_wide &&
        (path.starts_with("/app0/") || path.starts_with("/user/app/") ||
         path.starts_with("/mnt/sandbox/pfsmnt/") || path.starts_with("/system/"))) {
        return true;
    }
    return path.find("/mnt/sandbox/pfsmnt/NPXS39041-app0/assets/") != std::string_view::npos ||
           path.find("/mnt/sandbox/pfsmnt/ITEM00001-app0/assets/") != std::string_view::npos ||
           path.find("/user/app/NPXS39041/storedata/") != std::string_view::npos ||
           path.find("/user/appmeta/external/ITEM00001/") != std::string_view::npos ||
           path.find("/system_ex/app/NPXS20113/bdjstack/lib/fonts/") != std::string_view::npos ||
           path.find("/user/appmeta/NPXS39041/") != std::string_view::npos ||
           path.find("/user/appmeta/external/NPXS39041/") != std::string_view::npos;
}

bool ShouldTraceResourceManifest() {
    static const bool enabled = std::getenv("EXECUTOR_TRACE_RESOURCE_MANIFEST") != nullptr;
    return enabled;
}

const char* ResourceManifestClass(std::string_view path) {
    if (!ShouldTraceResourceManifest() || path.empty()) {
        return nullptr;
    }

    std::string lower{path};
    std::ranges::transform(lower, lower.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    struct Pattern {
        std::string_view needle;
        const char* resource_class;
    };
    static constexpr Pattern kPatterns[] = {
        {".flver", "geometry"}, {".mesh", "geometry"},
        {".model", "geometry"}, {".mdl", "geometry"},
        {".msb", "scene"},      {".scene", "scene"},
        {".world", "scene"},    {".level", "scene"},
        {".chrbnd", "actor"},   {".objbnd", "actor"},
        {".partsbnd", "actor"}, {".anibnd", "actor"},
        {".remobnd", "movie"},  {".gparam", "parameter"},
        {".parambnd", "parameter"},
        {".mtd", "material"},   {".matbin", "material"},
        {".material", "material"},
        {".tpf", "texture"},    {".gnf", "texture"},
        {".dds", "texture"},    {".texture", "texture"},
        {".hkxbhd", "physics"}, {".hkxbdt", "physics"},
        {".hkx", "physics"},    {".skeleton", "animation"},
        {".animation", "animation"}, {".anim", "animation"},
        {".bnd.dcx", "archive"}, {".bhd", "archive-index"},
        {".bdt", "archive-data"}, {".psarc", "archive"},
        {".pak", "archive"},    {".cpk", "archive"},
        {".dat", "archive-data"},
    };
    for (const auto& pattern : kPatterns) {
        if (lower.find(pattern.needle) != std::string::npos) {
            return pattern.resource_class;
        }
    }
    return nullptr;
}

u64 NextResourceManifestOpenId(std::string_view path) {
    if (ResourceManifestClass(path) == nullptr) {
        return 0;
    }
    static std::atomic<u64> next_id{1};
    constexpr u64 MaxResources = 2048;
    const u64 id = next_id.fetch_add(1, std::memory_order_relaxed);
    if (id > MaxResources) {
        return 0;
    }
    return id;
}

struct ResourceManifestState {
    u64 id{};
    std::string path;
    const char* resource_class{"resource"};
    u64 file_size{};
    u64 calls{};
    u64 requested{};
    u64 returned{};
    u64 max_end{};
    u64 short_reads{};
    u64 errors{};
    bool first_read_logged{};
};

std::mutex& ResourceManifestMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<s32, ResourceManifestState>& ResourceManifestFiles() {
    static std::map<s32, ResourceManifestState> files;
    return files;
}

u64 SaturatingResourceManifestAdd(u64 lhs, u64 rhs) {
    constexpr u64 Max = std::numeric_limits<u64>::max();
    return rhs > Max - lhs ? Max : lhs + rhs;
}

void RegisterResourceManifestFile(u64 id, s32 fd, std::string_view path, u64 file_size) {
    if (id == 0 || !ShouldTraceResourceManifest()) {
        return;
    }
    const char* resource_class = ResourceManifestClass(path);
    std::scoped_lock lock{ResourceManifestMutex()};
    ResourceManifestFiles().insert_or_assign(
        fd, ResourceManifestState{.id = id,
                                  .path = std::string{path},
                                  .resource_class = resource_class != nullptr ? resource_class
                                                                              : "resource",
                                  .file_size = file_size});
}

u64 ResourceManifestIovecRequest(const OrbisKernelIovec* iov, s32 iovcnt) {
    if (!ShouldTraceResourceManifest() || iov == nullptr || iovcnt <= 0) {
        return 0;
    }
    u64 requested = 0;
    for (s32 index = 0; index < iovcnt; ++index) {
        requested = SaturatingResourceManifestAdd(requested, iov[index].iov_len);
    }
    return requested;
}

void FormatHexPrefix(const void* data, s64 length, char* out, std::size_t out_size);
const char* CurrentLiveThreadName();

void NoteResourceManifestRead(s32 fd, const char* op, u64 request, s64 range_offset, s64 result,
                              const void* prefix_data, s64 prefix_length, s64 prefix_offset) {
    if (!ShouldTraceResourceManifest()) {
        return;
    }

    u64 id = 0;
    std::string path;
    const char* resource_class = "resource";
    bool emit_first_read = false;
    bool emit_detailed_read = false;
    u64 calls = 0;
    u64 returned = 0;
    u64 file_size = 0;
    u64 short_reads = 0;
    u64 errors = 0;
    {
        std::scoped_lock lock{ResourceManifestMutex()};
        const auto it = ResourceManifestFiles().find(fd);
        if (it == ResourceManifestFiles().end()) {
            return;
        }
        auto& state = it->second;
        state.calls = SaturatingResourceManifestAdd(state.calls, 1);
        state.requested = SaturatingResourceManifestAdd(state.requested, request);
        if (result < 0) {
            state.errors = SaturatingResourceManifestAdd(state.errors, 1);
        } else {
            state.returned = SaturatingResourceManifestAdd(state.returned, static_cast<u64>(result));
            if (request != 0 && static_cast<u64>(result) < request) {
                state.short_reads = SaturatingResourceManifestAdd(state.short_reads, 1);
            }
            if (result > 0 && range_offset >= 0) {
                const u64 end = SaturatingResourceManifestAdd(static_cast<u64>(range_offset),
                                                               static_cast<u64>(result));
                state.max_end = std::max(state.max_end, end);
            }
        }
        if (!state.first_read_logged && result > 0 && prefix_data != nullptr &&
            prefix_length > 0) {
            state.first_read_logged = true;
            emit_first_read = true;
            id = state.id;
            path = state.path;
            resource_class = state.resource_class;
        }
        static std::atomic_int detailed_read_budget{4096};
        if (state.resource_class == std::string_view{"archive-data"} &&
            state.file_size >= (64ull << 20) &&
            detailed_read_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            emit_detailed_read = true;
            id = state.id;
            path = state.path;
            resource_class = state.resource_class;
            calls = state.calls;
            returned = state.returned;
            file_size = state.file_size;
            short_reads = state.short_reads;
            errors = state.errors;
        }
    }

    if (emit_first_read) {
        char prefix[64]{};
        FormatHexPrefix(prefix_data, prefix_length, prefix, sizeof(prefix));
        AndroidFsLog("[EXECUTOR_RESOURCE_FIRST_READ] id=%llu fd=%d class=%s op=%s "
                     "offset=%lld request=%llu result=%lld prefix16=%s path=%s thread=%s",
                     static_cast<unsigned long long>(id), fd, resource_class, op,
                     static_cast<long long>(prefix_offset),
                     static_cast<unsigned long long>(request), static_cast<long long>(result),
                      prefix, path.c_str(), CurrentLiveThreadName());
    }
    if (emit_detailed_read) {
        AndroidFsLog("[EXECUTOR_RESOURCE_READ] id=%llu fd=%d class=%s op=%s call=%llu "
                     "offset=%lld request=%llu result=%lld returned=%llu size=%llu short=%llu "
                     "errors=%llu thread=%s path=%s",
                     static_cast<unsigned long long>(id), fd, resource_class, op,
                     static_cast<unsigned long long>(calls), static_cast<long long>(range_offset),
                     static_cast<unsigned long long>(request), static_cast<long long>(result),
                     static_cast<unsigned long long>(returned),
                     static_cast<unsigned long long>(file_size),
                     static_cast<unsigned long long>(short_reads),
                     static_cast<unsigned long long>(errors), CurrentLiveThreadName(),
                     path.c_str());
    }
}

void CloseResourceManifestFile(s32 fd) {
    if (!ShouldTraceResourceManifest()) {
        return;
    }

    ResourceManifestState state;
    {
        std::scoped_lock lock{ResourceManifestMutex()};
        const auto it = ResourceManifestFiles().find(fd);
        if (it == ResourceManifestFiles().end()) {
            return;
        }
        state = std::move(it->second);
        ResourceManifestFiles().erase(it);
    }
    AndroidFsLog("[EXECUTOR_RESOURCE_CLOSE] id=%llu fd=%d class=%s size=%llu calls=%llu "
                 "requested=%llu returned=%llu maxEnd=%llu short=%llu errors=%llu "
                 "firstRead=%d path=%s thread=%s",
                 static_cast<unsigned long long>(state.id), fd, state.resource_class,
                 static_cast<unsigned long long>(state.file_size),
                 static_cast<unsigned long long>(state.calls),
                 static_cast<unsigned long long>(state.requested),
                 static_cast<unsigned long long>(state.returned),
                 static_cast<unsigned long long>(state.max_end),
                 static_cast<unsigned long long>(state.short_reads),
                 static_cast<unsigned long long>(state.errors), state.first_read_logged ? 1 : 0,
                 state.path.c_str(), CurrentLiveThreadName());
}

std::string NormalizeAndroidGuestPathForFs(std::string_view raw_path) {
    std::string path{raw_path};
    std::replace(path.begin(), path.end(), '\\', '/');
    if (path.starts_with("app0/")) {
        path.insert(path.begin(), '/');
    } else if (path.starts_with("app0:")) {
        path = "/app0" + path.substr(5);
        if (path == "/app0") {
            path = "/app0/";
        }
    }
    return path;
}

std::string ManagedMediaFallbackPath(std::string_view guest_path) {
    static constexpr std::string_view kManagedPrefix = "/app0/Managed/";
    if (!guest_path.starts_with(kManagedPrefix)) {
        return {};
    }
    std::string remapped{"/app0/Media/Managed/"};
    remapped += guest_path.substr(kManagedPrefix.size());
    return remapped;
}

bool LooksLikeAndroidGuestPathStart(const char* value) {
    if (value == nullptr || value[0] != '/') {
        return false;
    }

    static constexpr std::string_view prefixes[] = {
        "/app0/",    "/data/", "/dev/",  "/mnt/",  "/preinst/",
        "/system/",  "/usb",   "/user/", "/update/"};
    for (const auto prefix : prefixes) {
        bool matches = true;
        for (std::size_t i = 0; i < prefix.size(); ++i) {
            if (value[i] != prefix[i]) {
                matches = false;
                break;
            }
        }
        if (matches) {
            return true;
        }
    }
    return false;
}

bool ReadBoundedPrintableCString(const char* value, std::size_t limit, std::string& out) {
    if (value == nullptr || !LooksLikeAndroidGuestPathStart(value)) {
        return false;
    }

    out.clear();
    out.reserve(64);
    for (std::size_t i = 0; i < limit; ++i) {
        const unsigned char ch = static_cast<unsigned char>(value[i]);
        if (ch == '\0') {
            return out.size() >= 2;
        }
        if (ch < 0x20 || ch > 0x7e) {
            out.clear();
            return false;
        }
        out.push_back(static_cast<char>(ch));
    }
    out.clear();
    return false;
}

std::string RecoverAndroidGuestPathNear(const char* raw_path) {
    if (raw_path == nullptr || raw_path[0] != '\0') {
        return {};
    }

    std::string candidate;
    for (int offset = 1; offset <= 256; ++offset) {
        if (ReadBoundedPrintableCString(raw_path + offset, 255, candidate)) {
            AndroidFsLog("[EXECUTOR_FS_PATH_RECOVER] base=%p offset=+%d path=%s", raw_path,
                         offset, candidate.c_str());
            return candidate;
        }
    }
    const auto page_offset = reinterpret_cast<std::uintptr_t>(raw_path) & 0xfffU;
    if (page_offset < 128U) {
        AndroidFsLog("[EXECUTOR_FS_PATH_RECOVER_SKIP_BACKSCAN] base=%p page_offset=%llu", raw_path,
                     static_cast<unsigned long long>(page_offset));
        return {};
    }
    for (int offset = -128; offset < 0; ++offset) {
        if (ReadBoundedPrintableCString(raw_path + offset, 255, candidate)) {
            AndroidFsLog("[EXECUTOR_FS_PATH_RECOVER] base=%p offset=%d path=%s", raw_path,
                         offset, candidate.c_str());
            return candidate;
        }
    }
    return {};
}

const char* ResolveAndroidGuestPathForFs(const char* raw_path, std::string& recovered) {
    recovered.clear();
    if (raw_path != nullptr && raw_path[0] == '\0') {
        recovered = RecoverAndroidGuestPathNear(raw_path);
        if (!recovered.empty()) {
            return recovered.c_str();
        }
    }
    return raw_path;
}

bool ShouldTraceAndroidAssetFile(const Core::FileSys::File* file) {
    return file != nullptr && ShouldTraceAndroidAssetPath(std::string_view{file->m_guest_name});
}

bool ShouldEmitLiveFileIoLog(const Core::FileSys::File* file, u64 request = 0, s64 result = 0) {
    if (!ShouldTraceLiveFileIo() && !(ShouldTraceLiveFileErrors() && result < 0)) {
        return false;
    }
    if (!ShouldTraceAndroidAssetFile(file)) {
        return false;
    }
    static std::atomic_int budget{8192};
    if (budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
        return true;
    }
    return result <= 0 || request >= (1u << 20);
}

const char* CurrentLiveThreadName() {
    return g_curthread ? g_curthread->name.c_str() : "<no-gcurthread>";
}

void FormatHexPrefix(const void* data, s64 length, char* out, std::size_t out_size) {
    if (out_size == 0) {
        return;
    }
    out[0] = '\0';
    if (data == nullptr || length <= 0) {
        return;
    }

    const auto* bytes = static_cast<const unsigned char*>(data);
    const std::size_t count = std::min<std::size_t>(static_cast<std::size_t>(length), 16);
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < count && cursor + 4 < out_size; ++i) {
        const int written = std::snprintf(out + cursor, out_size - cursor, "%s%02x",
                                          i == 0 ? "" : " ", bytes[i]);
        if (written <= 0) {
            break;
        }
        cursor += static_cast<std::size_t>(written);
    }
}

bool HasPngSignature(const void* data, s64 length) {
    if (data == nullptr || length < 8) {
        return false;
    }
    const auto* bytes = static_cast<const unsigned char*>(data);
    return bytes[0] == 0x89 && bytes[1] == 0x50 && bytes[2] == 0x4e && bytes[3] == 0x47 &&
           bytes[4] == 0x0d && bytes[5] == 0x0a && bytes[6] == 0x1a && bytes[7] == 0x0a;
}

void LogPs4StoreCoverPrefix(const char* op, s32 fd, const Core::FileSys::File* file, s32 index,
                            u64 request, s64 offset, s64 result, const void* buffer) {
    if (file == nullptr || !IsPs4StoreCoverPath(file->m_guest_name) || result <= 0 || offset != 0 ||
        buffer == nullptr) {
        return;
    }
    static int png_prefix_log_budget = 256;
    if (png_prefix_log_budget-- <= 0) {
        return;
    }

    char prefix[64] = {};
    FormatHexPrefix(buffer, result, prefix, sizeof(prefix));
    AndroidFsLog("[EXECUTOR_FS_PNG_PREFIX] op=%s fd=%d guest=%s index=%d offset=%lld "
                 "request=%llu result=%lld buf=%p prefix=%s png=%d",
                 op, fd, file->m_guest_name.c_str(), index, static_cast<long long>(offset),
                 static_cast<unsigned long long>(request), static_cast<long long>(result),
                 buffer, prefix, HasPngSignature(buffer, result) ? 1 : 0);
}

}
#endif

s32 PS4_SYSV_ABI open(const char* raw_path, s32 flags, u16 mode) {
#ifdef __ANDROID__
    std::string recovered_path;
    raw_path = ResolveAndroidGuestPathForFs(raw_path, recovered_path);
    if (raw_path != nullptr && raw_path[0] == '\0' &&
        (flags & 0x3) == ORBIS_KERNEL_O_RDONLY) {
        recovered_path = "/mnt/sandbox/pfsmnt/ITEM00001-app0/assets/ps4_cover_not_found.png";
        AndroidFsLog("[EXECUTOR_FS_EMPTY_PATH_FALLBACK] flags=0x%x mode=0%o fallback=%s", flags,
                     mode, recovered_path.c_str());
        raw_path = recovered_path.c_str();
    } else if (raw_path != nullptr && raw_path[0] == '\0' && flags == 0x601) {
        recovered_path = "/user/app/ITEM00001/logs/itemzflow_app.log";
        AndroidFsLog("[EXECUTOR_FS_EMPTY_PATH_FALLBACK] flags=0x%x mode=0%o fallback=%s", flags,
                     mode, recovered_path.c_str());
        raw_path = recovered_path.c_str();
    }
    std::string normalized_path;
    if (raw_path != nullptr && raw_path[0] != '\0') {
        normalized_path = NormalizeAndroidGuestPathForFs(raw_path);
        if (normalized_path != raw_path) {
            raw_path = normalized_path.c_str();
        }
    }
#endif
#ifdef __ANDROID__
    const u64 resource_manifest_id =
        raw_path != nullptr ? NextResourceManifestOpenId(std::string_view{raw_path}) : 0;
    if (resource_manifest_id != 0) {
        AndroidFsLog("[EXECUTOR_RESOURCE_OPEN] id=%llu class=%s path=%s flags=0x%x mode=0%o "
                     "thread=%s",
                     static_cast<unsigned long long>(resource_manifest_id),
                     ResourceManifestClass(raw_path), raw_path, flags, mode, CurrentLiveThreadName());
    }
#endif
    LOG_INFO(Kernel_Fs, "path = {} flags = {:#x} mode = {:#o}", raw_path, flags, mode);
#ifdef __ANDROID__
    const bool android_log_path =
        raw_path == nullptr || ShouldLogAndroidFsPath(std::string_view{raw_path});
    if (android_log_path) {
        AndroidFsLog("[EXECUTOR_FS_OPEN] path=%s flags=0x%x mode=0%o thread=%s",
                     raw_path ? raw_path : "<null>", flags, mode, CurrentLiveThreadName());
    }
#endif
    if (raw_path == nullptr) {
        *__Error() = POSIX_EFAULT;
#ifdef __ANDROID__
        if (ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_OPEN_ERROR] path=<null> flags=0x%x mode=0%o errno=%d "
                         "thread=%s",
                         flags, mode, *__Error(), CurrentLiveThreadName());
        }
#endif
        return -1;
    }

    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();

    bool read = (flags & 0x3) == ORBIS_KERNEL_O_RDONLY;
    bool write = (flags & 0x3) == ORBIS_KERNEL_O_WRONLY;
    bool rdwr = (flags & 0x3) == ORBIS_KERNEL_O_RDWR;

    if (!read && !write && !rdwr) {
        *__Error() = POSIX_EINVAL;
#ifdef __ANDROID__
        if (android_log_path || ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_OPEN_ERROR] path=%s flags=0x%x mode=0%o errno=%d "
                         "reason=invalid_flags thread=%s",
                         raw_path, flags, mode, *__Error(), CurrentLiveThreadName());
        }
#endif
        return -1;
    }

    if (strlen(raw_path) > 255) {
        *__Error() = POSIX_ENAMETOOLONG;
#ifdef __ANDROID__
        if (android_log_path || ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_OPEN_ERROR] path=%s flags=0x%x mode=0%o errno=%d "
                         "reason=too_long thread=%s",
                         raw_path, flags, mode, *__Error(), CurrentLiveThreadName());
        }
#endif
        return -1;
    }

    bool nonblock = (flags & ORBIS_KERNEL_O_NONBLOCK) != 0;
    bool append = (flags & ORBIS_KERNEL_O_APPEND) != 0;
    bool sync = (flags & ORBIS_KERNEL_O_SYNC) != 0 || (flags & ORBIS_KERNEL_O_FSYNC) != 0;
    bool create = (flags & ORBIS_KERNEL_O_CREAT) != 0;
    bool truncate = (flags & ORBIS_KERNEL_O_TRUNC) != 0;
    bool excl = (flags & ORBIS_KERNEL_O_EXCL) != 0;
    bool dsync = (flags & ORBIS_KERNEL_O_DSYNC) != 0;
    bool direct = (flags & ORBIS_KERNEL_O_DIRECT) != 0;
    bool directory = (flags & ORBIS_KERNEL_O_DIRECTORY) != 0;

    if (sync || direct || dsync || nonblock) {
        LOG_WARNING(Kernel_Fs, "flags {:#x} not fully handled", flags);
    }

    std::string_view path{raw_path};
    u32 handle = h->CreateHandle();
    auto* file = h->GetFile(handle);

    if (path.starts_with("/dev/")) {
        for (const auto& [prefix, factory] : available_device) {
            if (path.starts_with(prefix)) {
                file->is_opened = true;
                file->type = Core::FileSys::FileType::Device;
                file->m_guest_name = path;
                file->device = factory(handle, path.data(), flags, mode);
#ifdef __ANDROID__
                AndroidFsLog("[EXECUTOR_FS_OPEN_OK] fd=%u guest=%s device=true", handle,
                             file->m_guest_name.c_str());
                if (resource_manifest_id != 0) {
                    AndroidFsLog("[EXECUTOR_RESOURCE_OPEN_OK] id=%llu fd=%u class=%s guest=%s "
                                 "type=device thread=%s",
                                 static_cast<unsigned long long>(resource_manifest_id), handle,
                                 ResourceManifestClass(file->m_guest_name),
                                 file->m_guest_name.c_str(), CurrentLiveThreadName());
                }
#endif
                return handle;
            }
        }
    }

    bool read_only = false;
    file->m_guest_name = path;
    file->m_host_name = mnt->GetHostPath(file->m_guest_name, &read_only);
    bool exists = fs::exists(file->m_host_name);
#ifdef __ANDROID__
    if (!exists && read && !create) {
        const std::string managed_fallback = ManagedMediaFallbackPath(file->m_guest_name);
        if (!managed_fallback.empty()) {
            bool fallback_read_only = false;
            const auto fallback_host = mnt->GetHostPath(managed_fallback, &fallback_read_only);
            if (fs::exists(fallback_host)) {
                AndroidFsLog("[EXECUTOR_FS_MANAGED_FALLBACK] guest=%s remap=%s host=%s",
                             file->m_guest_name.c_str(), managed_fallback.c_str(),
                             fallback_host.string().c_str());
                file->m_guest_name = managed_fallback;
                file->m_host_name = fallback_host;
                read_only = fallback_read_only;
                exists = true;
            }
        }
    }
    if (file->m_guest_name.find("debug.log") != std::string::npos) {
        file->m_host_name = "/data/data/app.lsx4.android/files/app0-debug.log";
        read_only = false;
        exists = fs::exists(file->m_host_name);
        AndroidFsLog("[EXECUTOR_FS_DEBUGLOG] redirected guest=%s -> host=%s flags=0x%x",
                     file->m_guest_name.c_str(), file->m_host_name.string().c_str(), flags);
    }
#endif
#ifdef __ANDROID__
    if (android_log_path) {
            AndroidFsLog("[EXECUTOR_FS_MAP] guest=%s host=%s exists=%d read_only=%d thread=%s",
                     file->m_guest_name.c_str(), file->m_host_name.string().c_str(),
                     exists ? 1 : 0, read_only ? 1 : 0, CurrentLiveThreadName());
    }
    if (file->m_guest_name == "/user/app/NPXS39041/store.db" && exists &&
        (truncate || write || rdwr)) {
        std::error_code size_error;
        const auto current_size = fs::file_size(file->m_host_name, size_error);
        if (!size_error && current_size > 0) {
            AndroidFsLog("[EXECUTOR_FS_PROTECT] preserving seeded PS4 Store DB guest=%s host=%s "
                         "size=%llu flags=0x%x",
                         file->m_guest_name.c_str(), file->m_host_name.string().c_str(),
                         static_cast<unsigned long long>(current_size), flags);
            h->DeleteHandle(handle);
            *__Error() = POSIX_EACCES;
            return -1;
        }
    }
#endif
    s32 e = 0;

    if (create) {
        if (excl && exists) {
            h->DeleteHandle(handle);
            *__Error() = POSIX_EEXIST;
            return -1;
        }

        if (!exists) {
            if (read_only) {
                h->DeleteHandle(handle);
                *__Error() = POSIX_EROFS;
                return -1;
            }
            Common::FS::IOFile out(file->m_host_name, Common::FS::FileAccessMode::Create);
        }
    } else if (!exists) {
        h->DeleteHandle(handle);
        *__Error() = POSIX_ENOENT;
#ifdef __ANDROID__
        if (android_log_path || ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_OPEN_ERROR] path=%s host=%s flags=0x%x mode=0%o errno=%d "
                         "reason=missing thread=%s",
                         file->m_guest_name.c_str(), file->m_host_name.string().c_str(), flags,
                         mode, *__Error(), CurrentLiveThreadName());
        }
#endif
        return -1;
    }

    if (fs::is_directory(file->m_host_name) || directory) {
        directory = true;
    }

    if (directory) {
        if (!fs::is_directory(file->m_host_name)) {
            h->DeleteHandle(handle);
            *__Error() = POSIX_ENOTDIR;
            return -1;
        }

        if (write || rdwr) {
            h->DeleteHandle(handle);
            *__Error() = POSIX_EISDIR;
            return -1;
        }

        if (truncate) {
            h->DeleteHandle(handle);
            *__Error() = POSIX_EISDIR;
            return -1;
        }

        file->type = Core::FileSys::FileType::Directory;
        file->is_opened = true;
        if (file->m_guest_name.starts_with("/app0")) {
            file->directory = Core::Directories::PfsDirectory::Create(file->m_guest_name);
        } else {
            file->directory = Core::Directories::NormalDirectory::Create(file->m_guest_name);
        }
    } else {
        file->type = Core::FileSys::FileType::Regular;

        if (truncate && read_only) {
            h->DeleteHandle(handle);
            *__Error() = POSIX_EROFS;
            return -1;
        } else if (truncate) {
            e = file->f.Open(file->m_host_name, Common::FS::FileAccessMode::ReadWrite);
            if (e == 0) {
                file->f.SetSize(0);
            }
        }

        if (read) {
            e = file->f.Open(file->m_host_name, Common::FS::FileAccessMode::Read);
        } else if (read_only) {
            h->DeleteHandle(handle);
            *__Error() = POSIX_EROFS;
            return -1;
        } else if (write) {
            if (append) {
                e = file->f.Open(file->m_host_name, Common::FS::FileAccessMode::Append);
            } else {
                e = file->f.Open(file->m_host_name, Common::FS::FileAccessMode::Write);
            }
        } else if (rdwr) {
            if (append) {
                e = file->f.Open(file->m_host_name, Common::FS::FileAccessMode::ReadAppend);
            } else {
                e = file->f.Open(file->m_host_name, Common::FS::FileAccessMode::ReadWrite);
            }
        }
    }

    if (e != 0) {
        SetPosixErrno(e);
#ifdef __ANDROID__
        if (android_log_path) {
            AndroidFsLog("[EXECUTOR_FS_OPEN_FAIL] guest=%s host=%s platform_errno=%d posix=%d "
                         "thread=%s",
                         file->m_guest_name.c_str(), file->m_host_name.string().c_str(), e,
                         *__Error(), CurrentLiveThreadName());
        }
#endif
        h->DeleteHandle(handle);
        return -1;
    }

    file->is_opened = true;
    *__Error() = 0;
#ifdef __ANDROID__
    if (!directory && file->m_guest_name.starts_with("/user/app/") &&
        file->m_guest_name.ends_with("/app.pkg")) {
        AndroidFsLog("[EXECUTOR_INSTALL_MANAGER] op=file_transport_open_candidate path=%s",
                     file->m_guest_name.c_str());
        executor_install_manager_note_local_package_open(file->m_guest_name.c_str());
    }
    if (android_log_path) {
        std::error_code size_error;
        const auto host_size =
            !directory ? fs::file_size(file->m_host_name, size_error) : static_cast<u64>(0);
        AndroidFsLog("[EXECUTOR_FS_OPEN_OK] fd=%u guest=%s host=%s type=%s size=%llu thread=%s", handle,
                     file->m_guest_name.c_str(), file->m_host_name.string().c_str(),
                     directory ? "directory" : "regular",
                     size_error ? 0ULL : static_cast<unsigned long long>(host_size),
                     CurrentLiveThreadName());
    }
    if (resource_manifest_id != 0) {
        std::error_code size_error;
        const auto host_size =
            !directory ? fs::file_size(file->m_host_name, size_error) : static_cast<u64>(0);
        const u64 manifest_size = size_error ? 0ULL : static_cast<u64>(host_size);
        const char* resource_class = ResourceManifestClass(file->m_guest_name);
        AndroidFsLog("[EXECUTOR_RESOURCE_OPEN_OK] id=%llu fd=%u class=%s guest=%s host=%s "
                     "type=%s size=%llu thread=%s",
                     static_cast<unsigned long long>(resource_manifest_id), handle,
                     resource_class != nullptr ? resource_class : "resource",
                     file->m_guest_name.c_str(), file->m_host_name.string().c_str(),
                     directory ? "directory" : "regular",
                     static_cast<unsigned long long>(manifest_size), CurrentLiveThreadName());
        if (!directory) {
            RegisterResourceManifestFile(resource_manifest_id, static_cast<s32>(handle),
                                         file->m_guest_name, manifest_size);
        }
    }
#endif
    return handle;
}

s32 PS4_SYSV_ABI posix_open(const char* filename, s32 flags, u16 mode) {
    return open(filename, flags, mode);
}

s32 PS4_SYSV_ABI sceKernelOpen(const char* path, s32 flags, u16 mode) {
    s32 result = open(path, flags, mode);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s32 PS4_SYSV_ABI close(s32 fd) {
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
#ifdef __ANDROID__
        if (ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_CLOSE_ERROR] fd=%d guest=<bad-fd> errno=%d thread=%s", fd,
                         *__Error(), CurrentLiveThreadName());
        }
#endif
        return -1;
    }
    if (fd < 3) {
        *__Error() = POSIX_EPERM;
        return -1;
    }
    if (file->type == Core::FileSys::FileType::Regular) {
        file->f.Close();
    } else if (file->type == Core::FileSys::FileType::Socket) {
        file->socket->Close();
    }
#ifdef __ANDROID__
    if (ShouldEmitLiveFileIoLog(file)) {
        AndroidFsLog("[EXECUTOR_FS_CLOSE] thread=%s fd=%d guest=%s type=%u", CurrentLiveThreadName(),
                     fd, file->m_guest_name.c_str(), static_cast<u32>(file->type.load()));
    }
    CloseResourceManifestFile(fd);
#endif
    file->is_opened = false;
    LOG_INFO(Kernel_Fs, "Closing {}", file->m_guest_name);
    h->DeleteHandle(fd);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI posix_close(s32 fd) {
    return close(fd);
}

s32 PS4_SYSV_ABI sceKernelClose(s32 fd) {
    s32 result = close(fd);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

#ifdef __ANDROID__
extern "C" std::uint64_t ExecutorGetLastHleGuestRsp();
#endif

s64 PS4_SYSV_ABI write(s32 fd, const void* buf, u64 nbytes) {
#ifdef __ANDROID__
    if ((fd == 1 || fd == 2) && buf != nullptr && nbytes > 0) {
        if (std::FILE* gf = std::fopen(
                "/data/data/app.lsx4.android/files/executor-guest-stdout.log", "a")) {
            std::fwrite(buf, 1, static_cast<std::size_t>(std::min<u64>(nbytes, 1024)), gf);
            std::fclose(gf);
        }
        {
            static std::atomic<int> aot_dumped{0};
            const auto* amsg = static_cast<const char*>(buf);
            bool is_aot = false;
            for (u64 i = 0; i + 18 < nbytes && i < 220; ++i) {
                if (std::memcmp(amsg + i, "Failed to load AOT", 18) == 0) { is_aot = true; break; }
            }
            int aexp = 0;
            if (is_aot && aot_dumped.compare_exchange_strong(aexp, 1)) {
                AndroidFsLog("[EXECUTOR_AOTFAIL_DETECTED] corlib AOT '(null)' captured");
                const u64 rsp = ExecutorGetLastHleGuestRsp();
                if (std::FILE* sf = std::fopen(
                        "/data/data/app.lsx4.android/files/executor-aotfail.log", "a")) {
                    std::fprintf(sf, "AOTFAIL rsp=0x%llx\n", static_cast<unsigned long long>(rsp));
                    std::fflush(sf);
                    std::fprintf(sf, "RAWSTACK:\n");
                    for (int i = 0; i < 400; ++i) {
                        u64 v = 0;
                        std::memcpy(&v, reinterpret_cast<const void*>(rsp + static_cast<u64>(i) * 8), 8);
                        std::fprintf(sf, "%d:0x%llx ", i, static_cast<unsigned long long>(v));
                        if ((i % 6) == 5) std::fprintf(sf, "\n");
                    }
                    std::fprintf(sf, "\nENDRAW\n");
                    std::fflush(sf);
                    auto is_mscorlib_image = [](u64 cand, char* out, std::size_t n) -> bool {
                        if (cand < 0x200000000ULL || cand >= 0x300000000ULL) return false;
                        u64 namep = 0;
                        std::memcpy(&namep, reinterpret_cast<const void*>(cand + 0x20), 8);
                        if (namep < 0x200000000ULL || namep >= 0x300000000ULL) return false;
                        int pr = 0;
                        for (std::size_t k = 0; k + 1 < n; ++k) {
                            unsigned char c = 0;
                            std::memcpy(&c, reinterpret_cast<const void*>(namep + k), 1);
                            if (!c) { out[k] = 0; break; }
                            if (c < 0x20 || c >= 0x7f) { pr = 0; break; }
                            out[k] = static_cast<char>(c); ++pr;
                        }
                        return pr >= 8 && std::strstr(out, "mscorlib") != nullptr;
                    };
                    auto dump_img = [&](u64 img, const char* via) {
                        char path[96] = {};
                        is_mscorlib_image(img, path, sizeof(path));
                        u64 atab_base = 0, atab_rr = 0, asmname = 0, aot = 0;
                        std::memcpy(&atab_base, reinterpret_cast<const void*>(img + 0xc0 + 32 * 16), 8);
                        std::memcpy(&atab_rr, reinterpret_cast<const void*>(img + 0xc0 + 32 * 16 + 8), 8);
                        std::memcpy(&asmname, reinterpret_cast<const void*>(img + 0x28), 8);
                        std::memcpy(&aot, reinterpret_cast<const void*>(img + 0x3b8), 8);
                        char an[40] = {};
                        if (asmname >= 0x200000000ULL && asmname < 0x300000000ULL) {
                            for (int k = 0; k < 39; ++k) {
                                unsigned char c = 0;
                                std::memcpy(&c, reinterpret_cast<const void*>(asmname + k), 1);
                                if (!c) break; an[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                            }
                        }
                        std::fprintf(sf, "  IMG@0x%llx via=%s path=\"%s\" asmName=\"%s\" "
                                     "tables[ASSEMBLY] base=0x%llx rows_rowsize=0x%llx aot_module=0x%llx\n",
                                     static_cast<unsigned long long>(img), via, path, an,
                                     static_cast<unsigned long long>(atab_base),
                                     static_cast<unsigned long long>(atab_rr),
                                     static_cast<unsigned long long>(aot));
                        std::fflush(sf);
                    };
                    {
                        int al = 0;
                        for (int i = 0; i < 600 && al < 16; ++i) {
                            u64 p = 0;
                            std::memcpy(&p, reinterpret_cast<const void*>(rsp + static_cast<u64>(i) * 8), 8);
                            if (p < 0x200000000ULL || p >= 0x300000000ULL) continue;
                            for (u64 off : {0x58ULL, 0x60ULL}) {
                                u64 img = 0;
                                std::memcpy(&img, reinterpret_cast<const void*>(p + off), 8);
                                if (img < 0x200000000ULL || img >= 0x300000000ULL) continue;
                                u64 ipath = 0;
                                std::memcpy(&ipath, reinterpret_cast<const void*>(img + 0x20), 8);
                                if (ipath < 0x200000000ULL || ipath >= 0x300000000ULL) continue;
                                char ip[80] = {}; int pr = 0;
                                for (int k = 0; k < 79; ++k) { unsigned char c=0; std::memcpy(&c,reinterpret_cast<const void*>(ipath+k),1); if(!c){ip[k]=0;break;} if(c<0x20||c>=0x7f){pr=0;break;} ip[k]=(char)c; ++pr; }
                                if (pr < 6 || !std::strstr(ip, ".dll")) continue;
                                u64 anamep = 0;
                                std::memcpy(&anamep, reinterpret_cast<const void*>(p + 0x10), 8);
                                char an3[40] = {};
                                bool aname_ok = anamep >= 0x200000000ULL && anamep < 0x300000000ULL;
                                if (aname_ok) { for (int k=0;k<39;k++){unsigned char c=0; std::memcpy(&c,reinterpret_cast<const void*>(anamep+k),1); if(!c)break; an3[k]=(c>=0x20&&c<0x7f)?(char)c:'.';} }
                                u64 iaot=0; std::memcpy(&iaot, reinterpret_cast<const void*>(img+0x3b8),8);
                                std::fprintf(sf, "  ASMLIST@0x%llx aname.name=0x%llx \"%s\" img=0x%llx path=\"%s\" img.aot=0x%llx\n",
                                             (unsigned long long)p, (unsigned long long)anamep,
                                             aname_ok?an3:"<NULL>", (unsigned long long)img, ip, (unsigned long long)iaot);
                                std::fflush(sf);
                                ++al; break;
                            }
                        }
                    }
                    int hits = 0;
                    char tmp[96] = {};
                    for (int i = 0; i < 600 && hits < 8; ++i) {
                        u64 p = 0;
                        std::memcpy(&p, reinterpret_cast<const void*>(rsp + static_cast<u64>(i) * 8), 8);
                        if (p < 0x200000000ULL || p >= 0x300000000ULL) continue;
                        if (is_mscorlib_image(p, tmp, sizeof(tmp))) { dump_img(p, "stack-img"); ++hits; continue; }
                        for (u64 off : {0x58ULL, 0x60ULL}) {
                            u64 img = 0;
                            std::memcpy(&img, reinterpret_cast<const void*>(p + off), 8);
                            if (is_mscorlib_image(img, tmp, sizeof(tmp))) {
                                u64 anamep = 0;
                                std::memcpy(&anamep, reinterpret_cast<const void*>(p + 0x10), 8);
                                char an2[40] = {};
                                if (anamep >= 0x200000000ULL && anamep < 0x300000000ULL) {
                                    for (int k = 0; k < 39; ++k) {
                                        unsigned char c = 0;
                                        std::memcpy(&c, reinterpret_cast<const void*>(anamep + k), 1);
                                        if (!c) break; an2[k] = (c >= 0x20 && c < 0x7f) ? static_cast<char>(c) : '.';
                                    }
                                }
                                std::fprintf(sf, "  ASM@0x%llx aname.name@+0x10=0x%llx \"%s\" image@+0x%llx=0x%llx\n",
                                             static_cast<unsigned long long>(p),
                                             static_cast<unsigned long long>(anamep), an2,
                                             static_cast<unsigned long long>(off),
                                             static_cast<unsigned long long>(img));
                                std::fflush(sf);
                                dump_img(img, "asm-img");
                                ++hits;
                                break;
                            }
                        }
                    }
                    std::fclose(sf);
                }
            }
        }
        {
            static std::atomic<int> dumped{0};
            const auto* msg = static_cast<const char*>(buf);
            bool is_gfile = false;
            for (u64 i = 0; i + 5 < nbytes && i < 200; ++i) {
                if (msg[i] == 'g' && msg[i + 1] == 'f' && msg[i + 2] == 'i' && msg[i + 3] == 'l' &&
                    msg[i + 4] == 'e') { is_gfile = true; break; }
            }
            int expected = 0;
            if (is_gfile && dumped.compare_exchange_strong(expected, 1)) {
                AndroidFsLog("[EXECUTOR_GFILE_DETECTED] gfile NULL-filename assert observed");
                const u64 rsp = ExecutorGetLastHleGuestRsp();
                if (std::FILE* sf = std::fopen(
                        "/data/data/app.lsx4.android/files/executor-gfile-stack.log", "a")) {
                    std::fprintf(sf, "gfile stackdump rsp=0x%llx\n",
                                 static_cast<unsigned long long>(rsp));
                    int found = 0;
                    for (int i = 0; i < 768 && found < 32; ++i) {
                        u64 v = 0;
                        std::memcpy(&v, reinterpret_cast<const void*>(rsp + static_cast<u64>(i) * 8),
                                    8);
                        if (v >= 0x800000000ULL && v < 0x810000000ULL) {
                            std::fprintf(sf, "stk[%d] 0x%llx mono%+lld\n", i,
                                         static_cast<unsigned long long>(v),
                                         static_cast<long long>(v) - 0x803190000LL);
                            ++found;
                        }
                    }
                    const u64 base = 0x8032a2e00ULL, span = 0x200ULL;
                    std::fprintf(sf, "code@0x%llx:\n", static_cast<unsigned long long>(base));
                    for (u64 o = 0; o < span; o += 16) {
                        unsigned char b[16];
                        std::memcpy(b, reinterpret_cast<const void*>(base + o), 16);
                        std::fprintf(sf, "%06llx:", static_cast<unsigned long long>(o));
                        for (int k = 0; k < 16; ++k) std::fprintf(sf, " %02x", b[k]);
                        std::fprintf(sf, "\n");
                    }
                    for (u64 o = 0; o + 7 < span; ++o) {
                        unsigned char op[3];
                        std::memcpy(op, reinterpret_cast<const void*>(base + o), 3);
                        if (op[0] == 0x48 && op[1] == 0x8d && (op[2] & 0xc7) == 0x05) {
                            int disp = 0;
                            std::memcpy(&disp, reinterpret_cast<const void*>(base + o + 3), 4);
                            const u64 tgt = base + o + 7 + static_cast<s64>(disp);
                            char s[64];
                            int n = 0;
                            for (; n < 63; ++n) {
                                char c = 0;
                                std::memcpy(&c, reinterpret_cast<const void*>(tgt + n), 1);
                                if (c == 0) break;
                                s[n] = (c >= 0x20 && c < 0x7f) ? c : '.';
                            }
                            s[n] = 0;
                            if (n >= 3) {
                                std::fprintf(sf, "lea@+0x%llx -> 0x%llx \"%s\"\n",
                                             static_cast<unsigned long long>(o),
                                             static_cast<unsigned long long>(tgt), s);
                            }
                        }
                    }
                    std::fclose(sf);
                }
            }
        }
        constexpr u64 MaxPreview = 256;
        const auto preview_size = std::min<u64>(nbytes, MaxPreview);
        char preview[MaxPreview + 1]{};
        const auto* bytes = static_cast<const unsigned char*>(buf);
        for (u64 i = 0; i < preview_size; ++i) {
            const unsigned char c = bytes[i];
            preview[i] = (std::isprint(c) || c == '\n' || c == '\r' || c == '\t') ? char(c) : '.';
        }
        const std::string_view preview_view{preview, strnlen(preview, MaxPreview)};
        if (ShouldStoreGuestStdout(preview_view)) {
            if (ShouldTraceGuestStdout()) {
                std::fprintf(stderr,
                             "[EXECUTOR_ANDROID_WRITE] fd=%d size=%llu preview=\"%s\"%s\n", fd,
                             static_cast<unsigned long long>(nbytes), preview,
                             nbytes > MaxPreview ? "..." : "");
                std::fflush(stderr);
                AndroidFsLog("[EXECUTOR_ANDROID_WRITE] fd=%d size=%llu preview=\"%s\"%s", fd,
                             static_cast<unsigned long long>(nbytes), preview,
                             nbytes > MaxPreview ? "..." : "");
            }
            executor_store_note_stdout_write(static_cast<const char*>(buf),
                                             static_cast<std::size_t>(nbytes));
        }
    }
#endif
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    std::scoped_lock lk{file->m_mutex};
    if (file->type == Core::FileSys::FileType::Device) {
        s64 result = file->device->write(buf, nbytes);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    } else if (file->type == Core::FileSys::FileType::Socket) {
        return file->socket->SendPacket(buf, nbytes, 0, nullptr, 0);
    } else if (file->type == Core::FileSys::FileType::Directory) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    return file->f.WriteRaw<u8>(buf, nbytes);
}

s64 PS4_SYSV_ABI posix_write(s32 fd, const void* buf, u64 nbytes) {
    return write(fd, buf, nbytes);
}

s64 PS4_SYSV_ABI sceKernelWrite(s32 fd, const void* buf, u64 nbytes) {
#ifdef __ANDROID__
    if (fd == 1 || fd == 2) {
        ExecutorLogKernelWriteText(fd, buf, nbytes);
    }
#endif
    s64 result = write(fd, buf, nbytes);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 ReadFile(Common::FS::IOFile& file, void* buf, u64 nbytes) {
    const auto* memory = Core::Memory::Instance();
    const auto size = file.GetSize();
    const auto tell = file.Tell();
    const u64 remaining = tell >= 0 && static_cast<u64>(tell) < size ? size - static_cast<u64>(tell) : 0;
    memory->InvalidateMemory(reinterpret_cast<VAddr>(buf), std::min<u64>(nbytes, remaining));

    return file.ReadRaw<u8>(buf, nbytes);
}

s64 PS4_SYSV_ABI readv(s32 fd, const OrbisKernelIovec* iov, s32 iovcnt) {
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    std::scoped_lock lk{file->m_mutex};
    if (file->type == Core::FileSys::FileType::Device) {
        s64 result = file->device->readv(iov, iovcnt);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    } else if (file->type == Core::FileSys::FileType::Directory) {
        s64 result = file->directory->readv(iov, iovcnt);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }

    if (file->f.IsWriteOnly()) {
        *__Error() = POSIX_EBADF;
#ifdef __ANDROID__
        NoteResourceManifestRead(fd, "readv", ResourceManifestIovecRequest(iov, iovcnt),
                                 file->f.Tell(), -1, nullptr, 0, file->f.Tell());
#endif
        return -1;
    }

#ifdef __ANDROID__
    const bool trace_asset = ShouldEmitLiveFileIoLog(file);
    const s64 before = file->f.Tell();
    const u64 manifest_request = ResourceManifestIovecRequest(iov, iovcnt);
    const void* manifest_prefix_data = nullptr;
    s64 manifest_prefix_length = 0;
    s64 manifest_prefix_offset = before;
    if (trace_asset) {
        AndroidFsLog("[EXECUTOR_FS_READV] fd=%d guest=%s iovcnt=%d before=%lld size=%llu", fd,
                     file->m_guest_name.c_str(), iovcnt, static_cast<long long>(before),
                     static_cast<unsigned long long>(file->f.GetSize()));
    }
#endif
    s64 total_read = 0;
    for (s32 i = 0; i < iovcnt; i++) {
        const s64 chunk_before = file->f.Tell();
        const s64 read_result = ReadFile(file->f, iov[i].iov_base, iov[i].iov_len);
        total_read += read_result;
        if (read_result >= 0) {
            *__Error() = 0;
        }
#ifdef __ANDROID__
        if (manifest_prefix_data == nullptr && read_result > 0 && iov[i].iov_base != nullptr) {
            manifest_prefix_data = iov[i].iov_base;
            manifest_prefix_length = read_result;
            manifest_prefix_offset = chunk_before;
        }
        if (trace_asset && ShouldEmitLiveFileIoLog(file, iov[i].iov_len, read_result)) {
            AndroidFsLog("[EXECUTOR_FS_READV_IOV] fd=%d guest=%s index=%d request=%llu result=%lld "
                         "buf=%p thread=%s",
                         fd, file->m_guest_name.c_str(), i,
                         static_cast<unsigned long long>(iov[i].iov_len),
                         static_cast<long long>(read_result), iov[i].iov_base,
                         CurrentLiveThreadName());
            LogPs4StoreCoverPrefix("readv", fd, file, i,
                                   static_cast<unsigned long long>(iov[i].iov_len), chunk_before,
                                   read_result, iov[i].iov_base);
        }
#endif
    }
#ifdef __ANDROID__
    NoteResourceManifestRead(fd, "readv", manifest_request, before, total_read,
                             manifest_prefix_data, manifest_prefix_length,
                             manifest_prefix_offset);
    if (trace_asset) {
        AndroidFsLog("[EXECUTOR_FS_READV_RESULT] fd=%d guest=%s total=%lld after=%lld errno=%d",
                     fd, file->m_guest_name.c_str(), static_cast<long long>(total_read),
                     static_cast<long long>(file->f.Tell()), *__Error());
    }
#endif
    if (total_read >= 0) {
        *__Error() = 0;
    }
    return total_read;
}

s64 PS4_SYSV_ABI posix_readv(s32 fd, const OrbisKernelIovec* iov, s32 iovcnt) {
    return readv(fd, iov, iovcnt);
}

s64 PS4_SYSV_ABI sceKernelReadv(s32 fd, const OrbisKernelIovec* iov, s32 iovcnt) {
    s64 result = readv(fd, iov, iovcnt);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 PS4_SYSV_ABI writev(s32 fd, const OrbisKernelIovec* iov, s32 iovcnt) {
#ifdef __ANDROID__
    if ((fd == 1 || fd == 2) && iov != nullptr && iovcnt > 0) {
        for (s32 v = 0; v < iovcnt; ++v) {
            ExecutorLogKernelWriteText(fd, iov[v].iov_base, iov[v].iov_len);
        }
    }
    if (fd >= 0 && fd <= 2 && iov != nullptr && iovcnt > 0) {
        for (s32 v = 0; v < iovcnt; ++v) {
            const auto* base = static_cast<const unsigned char*>(iov[v].iov_base);
            const std::size_t len = iov[v].iov_len;
            if (base == nullptr || len == 0 || len > 4096) {
                continue;
            }
            constexpr std::size_t MaxPreview = 256;
            const std::size_t preview_size = std::min<std::size_t>(len, MaxPreview);
            char preview[MaxPreview + 1]{};
            for (std::size_t i = 0; i < preview_size; ++i) {
                const unsigned char c = base[i];
                preview[i] =
                    (std::isprint(c) || c == '\n' || c == '\r' || c == '\t') ? char(c) : '.';
            }
            const std::string_view preview_view{preview, strnlen(preview, MaxPreview)};
            if (preview_view.find("Forcing submitDone to avoid TRC R4089 breach") !=
                std::string_view::npos) {
                AndroidFsLog("[EXECUTOR_LIVE_SUBMITDONE_FORCED] fd=%d iov=%d size=%llu preview=\"%s\"",
                             fd, v, static_cast<unsigned long long>(len), preview);
            }
            if (ShouldStoreGuestStdout(preview_view)) {
                if (ShouldTraceGuestStdout()) {
                    AndroidFsLog("[EXECUTOR_ANDROID_WRITEV] fd=%d iov=%d size=%llu preview=\"%s\"%s",
                                 fd, v, static_cast<unsigned long long>(len), preview,
                                 len > MaxPreview ? "..." : "");
                }
                executor_store_note_stdout_write(static_cast<const char*>(iov[v].iov_base),
                                                 static_cast<std::size_t>(len));
            }
        }
    }
#endif
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    std::scoped_lock lk{file->m_mutex};

    if (file->type == Core::FileSys::FileType::Device) {
        s64 result = file->device->writev(iov, iovcnt);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    } else if (file->type == Core::FileSys::FileType::Directory) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    s64 total_written = 0;
    for (s32 i = 0; i < iovcnt; i++) {
        total_written += file->f.WriteRaw<u8>(iov[i].iov_base, iov[i].iov_len);
    }
    return total_written;
}

s64 PS4_SYSV_ABI posix_writev(s32 fd, const OrbisKernelIovec* iov, s32 iovcnt) {
    return writev(fd, iov, iovcnt);
}

s64 PS4_SYSV_ABI sceKernelWritev(s32 fd, const OrbisKernelIovec* iov, s32 iovcnt) {
    s64 result = writev(fd, iov, iovcnt);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 PS4_SYSV_ABI posix_lseek(s32 fd, s64 offset, s32 whence) {
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    std::scoped_lock lk{file->m_mutex};
#ifdef __ANDROID__
    const bool trace_asset = ShouldEmitLiveFileIoLog(file);
    const s64 before = file->type == Core::FileSys::FileType::Regular ? file->f.Tell() : 0;
#endif
    if (file->type == Core::FileSys::FileType::Device) {
        s64 result = file->device->lseek(offset, whence);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    } else if (file->type == Core::FileSys::FileType::Directory) {
        s64 result = file->directory->lseek(offset, whence);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }

    Common::FS::SeekOrigin origin{};
    if (whence == 0) {
        origin = Common::FS::SeekOrigin::SetOrigin;
    } else if (whence == 1) {
        origin = Common::FS::SeekOrigin::CurrentPosition;
    } else if (whence == 2) {
        origin = Common::FS::SeekOrigin::End;
    } else if (whence == 3 || whence == 4) {
        *__Error() = POSIX_ENOTTY;
        return -1;
    } else {
        *__Error() = POSIX_EINVAL;
        return -1;
    }

    if (!file->f.Seek(offset, origin)) {
        if (errno != 0) {
            SetPosixErrno(errno);
            return -1;
        }
        return -1;
    }

    s64 result = file->f.Tell();
    if (result < 0) {
        SetPosixErrno(errno);
        return -1;
    }
    *__Error() = 0;
#ifdef __ANDROID__
    if (trace_asset) {
        AndroidFsLog("[EXECUTOR_FS_LSEEK] fd=%d guest=%s offset=%lld whence=%d before=%lld result=%lld "
                     "size=%llu",
                     fd, file->m_guest_name.c_str(), static_cast<long long>(offset), whence,
                     static_cast<long long>(before), static_cast<long long>(result),
                     static_cast<unsigned long long>(file->f.GetSize()));
    }
#endif
    return result;
}

s64 PS4_SYSV_ABI sceKernelLseek(s32 fd, s64 offset, s32 whence) {
    s64 result = posix_lseek(fd, offset, whence);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 PS4_SYSV_ABI read(s32 fd, void* buf, u64 nbytes) {
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    std::scoped_lock lk{file->m_mutex};
    if (file->type == Core::FileSys::FileType::Device) {
        s64 result = file->device->read(buf, nbytes);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    } else if (file->type == Core::FileSys::FileType::Directory) {
        s64 result = file->directory->read(buf, nbytes);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    } else if (file->type == Core::FileSys::FileType::Socket) {
        return file->socket->ReceivePacket(buf, nbytes, 0, nullptr, 0);
    }

    if (file->f.IsWriteOnly()) {
        *__Error() = POSIX_EBADF;
#ifdef __ANDROID__
        const s64 manifest_offset = file->f.Tell();
        NoteResourceManifestRead(fd, "read", nbytes, manifest_offset, -1, nullptr, 0,
                                 manifest_offset);
#endif
        return -1;
    }

    const auto before = file->f.Tell();
    const auto size = file->f.GetSize();
    const auto result = ReadFile(file->f, buf, nbytes);
    if (result >= 0) {
        *__Error() = 0;
    }
#ifdef __ANDROID__
    NoteResourceManifestRead(fd, "read", nbytes, before, result, buf, result, before);
    if (ShouldEmitLiveFileIoLog(file, nbytes, result)) {
        const auto after = file->f.Tell();
        AndroidFsLog("[EXECUTOR_FS_READ] fd=%d guest=%s request=%llu before=%lld result=%lld "
                     "after=%lld size=%llu thread=%s",
                     fd, file->m_guest_name.c_str(), static_cast<unsigned long long>(nbytes),
                     static_cast<long long>(before), static_cast<long long>(result),
                     static_cast<long long>(after), static_cast<unsigned long long>(size),
                     CurrentLiveThreadName());
    }
#endif
    return result;
}

s64 PS4_SYSV_ABI posix_read(s32 fd, void* buf, u64 nbytes) {
    return read(fd, buf, nbytes);
}

s64 PS4_SYSV_ABI sceKernelRead(s32 fd, void* buf, u64 nbytes) {
    s64 result = read(fd, buf, nbytes);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s32 PS4_SYSV_ABI posix_mkdir(const char* path, u16 mode) {
#ifdef __ANDROID__
    std::string recovered_path;
    path = ResolveAndroidGuestPathForFs(path, recovered_path);
    AndroidFsLog("[EXECUTOR_FS_MKDIR] path=%s mode=0%o", path ? path : "<null>", mode);
#endif
    LOG_INFO(Kernel_Fs, "path = {} mode = {:#o}", path, mode);
    if (path == nullptr) {
        *__Error() = POSIX_ENOTDIR;
        return -1;
    }
    if (strlen(path) > 255) {
        *__Error() = POSIX_ENAMETOOLONG;
        return -1;
    }
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();

    bool ro = false;
    const auto dir_name = mnt->GetHostPath(path, &ro);

    if (fs::exists(dir_name)) {
        *__Error() = POSIX_EEXIST;
        return -1;
    }

    if (ro) {
        *__Error() = POSIX_EROFS;
        return -1;
    }

    std::error_code ec;
    if (dir_name.empty() || !fs::create_directory(dir_name, ec)) {
        *__Error() = POSIX_EIO;
        return -1;
    }

    if (!fs::exists(dir_name)) {
        *__Error() = POSIX_ENOENT;
        return -1;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelMkdir(const char* path, u16 mode) {
    s32 result = posix_mkdir(path, mode);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s32 PS4_SYSV_ABI posix_rmdir(const char* path) {
    if (strlen(path) > 255) {
        *__Error() = POSIX_ENAMETOOLONG;
        return -1;
    }
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    bool ro = false;

    const fs::path dir_name = mnt->GetHostPath(path, &ro);

    if (ro) {
        *__Error() = POSIX_EROFS;
        return -1;
    }

    if (dir_name.empty() || !fs::is_directory(dir_name)) {
        *__Error() = POSIX_ENOTDIR;
        return -1;
    }

    if (!fs::exists(dir_name)) {
        *__Error() = POSIX_ENOENT;
        return -1;
    }

    std::error_code ec;
    s32 result = fs::remove_all(dir_name, ec);

    if (ec) {
        *__Error() = POSIX_EIO;
        return -1;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelRmdir(const char* path) {
    s32 result = posix_rmdir(path);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s32 PS4_SYSV_ABI posix_stat(const char* path, OrbisKernelStat* sb) {
    LOG_DEBUG(Kernel_Fs, "(PARTIAL) path = {}", path);
    if (strlen(path) > 255) {
        *__Error() = POSIX_ENAMETOOLONG;
        return -1;
    }
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    const auto path_name = mnt->GetHostPath(path);
    std::memset(sb, 0, sizeof(OrbisKernelStat));
    const bool is_dir = fs::is_directory(path_name);
    const bool is_file = fs::is_regular_file(path_name);
    if (!is_dir && !is_file) {
#ifdef __ANDROID__
        if (path != nullptr && ShouldTraceLiveFileErrors() &&
            ShouldTraceAndroidAssetPath(std::string_view{path})) {
            AndroidFsLog("[EXECUTOR_FS_STAT_MISS] guest=%s host=%s errno=%d thread=%s", path,
                         path_name.string().c_str(), POSIX_ENOENT, CurrentLiveThreadName());
        }
#endif
        *__Error() = POSIX_ENOENT;
        return -1;
    }

    const auto now_sys = std::chrono::system_clock::now();
    const auto now_file = fs::file_time_type::clock::now();
    const auto mtime = fs::last_write_time(path_name);
    const auto mtimestamp = now_sys + (mtime - now_file);

    if (fs::is_directory(path_name)) {
        sb->st_mode = 0000777u | 0040000u;
        sb->st_size = 65536;
        sb->st_blksize = 65536;
        sb->st_blocks = 128;
        sb->st_mtim.tv_sec =
            std::chrono::duration_cast<std::chrono::seconds>(mtimestamp.time_since_epoch()).count();
    } else {
        sb->st_mode = 0000777u | 0100000u;
        sb->st_size = static_cast<s64>(fs::file_size(path_name));
        sb->st_blksize = 512;
        sb->st_blocks = (sb->st_size + 511) / 512;
        sb->st_mtim.tv_sec =
            std::chrono::duration_cast<std::chrono::seconds>(mtimestamp.time_since_epoch()).count();
    }

#ifdef __ANDROID__
    if (path != nullptr && ShouldTraceLiveFileIo() &&
        ShouldTraceAndroidAssetPath(std::string_view{path})) {
        AndroidFsLog("[EXECUTOR_FS_STAT] guest=%s host=%s result=0 mode=0%llo size=%lld "
                     "blksize=%lld blocks=%lld thread=%s",
                     path, path_name.string().c_str(), static_cast<unsigned long long>(sb->st_mode),
                     static_cast<long long>(sb->st_size), static_cast<long long>(sb->st_blksize),
                     static_cast<long long>(sb->st_blocks), CurrentLiveThreadName());
    }
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelStat(const char* path, OrbisKernelStat* sb) {
    s32 result = posix_stat(path, sb);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s32 PS4_SYSV_ABI sceKernelCheckReachability(const char* path) {
    if (strlen(path) > 255) {
        return ORBIS_KERNEL_ERROR_ENAMETOOLONG;
    }

    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    std::string_view guest_path{path};
    for (const auto& prefix : available_device | std::views::keys) {
        if (guest_path.starts_with(prefix)) {
            return ORBIS_OK;
        }
    }
    const auto path_name = mnt->GetHostPath(guest_path);
    if (!fs::exists(path_name)) {
        return ORBIS_KERNEL_ERROR_ENOENT;
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI fstat(s32 fd, OrbisKernelStat* sb) {
    LOG_DEBUG(Kernel_Fs, "(PARTIAL) fd = {}", fd);
    if (sb == nullptr) {
        *__Error() = POSIX_EFAULT;
        return -1;
    }
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
#ifdef __ANDROID__
        if (ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_FSTAT_ERROR] fd=%d guest=<bad-fd> errno=%d thread=%s", fd,
                         *__Error(), CurrentLiveThreadName());
        }
#endif
        return -1;
    }
    std::memset(sb, 0, sizeof(OrbisKernelStat));

    switch (file->type) {
    case Core::FileSys::FileType::Device: {
        s32 result = file->device->fstat(sb);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }
    case Core::FileSys::FileType::Regular: {
        sb->st_mode = 0000777u | 0100000u;
        sb->st_size = file->f.GetSize();
        sb->st_blksize = 512;
        sb->st_blocks = (sb->st_size + 511) / 512;
#if defined(__linux__) || defined(__FreeBSD__)
        struct stat filestat = {};
        stat(file->f.GetPath().c_str(), &filestat);
        sb->st_atim = *reinterpret_cast<OrbisKernelTimespec*>(&filestat.st_atim);
        sb->st_mtim = *reinterpret_cast<OrbisKernelTimespec*>(&filestat.st_mtim);
        sb->st_ctim = *reinterpret_cast<OrbisKernelTimespec*>(&filestat.st_ctim);
#elif defined(__APPLE__)
        struct stat filestat = {};
        stat(file->f.GetPath().c_str(), &filestat);
        sb->st_atim = *reinterpret_cast<OrbisKernelTimespec*>(&filestat.st_atimespec);
        sb->st_mtim = *reinterpret_cast<OrbisKernelTimespec*>(&filestat.st_mtimespec);
        sb->st_ctim = *reinterpret_cast<OrbisKernelTimespec*>(&filestat.st_ctimespec);
#else
        const auto ft = std::filesystem::last_write_time(file->f.GetPath());
        const auto sctp = std::chrono::time_point_cast<std::chrono::nanoseconds>(
            ft - std::filesystem::file_time_type::clock::now() + std::chrono::system_clock::now());
        const auto secs = std::chrono::time_point_cast<std::chrono::seconds>(sctp);
        const auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(sctp - secs);

        sb->st_mtim.tv_sec = static_cast<int64_t>(secs.time_since_epoch().count());
        sb->st_mtim.tv_nsec = static_cast<int64_t>(nsecs.count());
        sb->st_atim = sb->st_mtim;
        sb->st_ctim = sb->st_mtim;
#endif
        break;
    }
    case Core::FileSys::FileType::Directory: {
        s32 result = file->directory->fstat(sb);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }
    case Core::FileSys::FileType::Socket: {
        return file->socket->fstat(sb);
    }
    case Core::FileSys::FileType::Epoll:
    case Core::FileSys::FileType::Resolver:
    case Core::FileSys::FileType::Equeue: {
        LOG_ERROR(Kernel_Fs, "(STUBBED) file type {}", magic_enum::enum_name(file->type.load()));
        break;
    }
    default:
        UNREACHABLE_MSG("{}", u32(file->type.load()));
    }
#ifdef __ANDROID__
    if (ShouldEmitLiveFileIoLog(file)) {
        const auto type_name = std::string{magic_enum::enum_name(file->type.load())};
        AndroidFsLog("[EXECUTOR_FS_FSTAT] fd=%d guest=%s type=%s result=0 mode=0%llo size=%lld "
                     "blksize=%lld blocks=%lld",
                     fd, file->m_guest_name.c_str(), type_name.c_str(),
                     static_cast<unsigned long long>(sb->st_mode), static_cast<long long>(sb->st_size),
                     static_cast<long long>(sb->st_blksize), static_cast<long long>(sb->st_blocks));
    }
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI posix_fstat(s32 fd, OrbisKernelStat* sb) {
    return fstat(fd, sb);
}

s32 PS4_SYSV_ABI sceKernelFstat(s32 fd, OrbisKernelStat* sb) {
    s32 result = fstat(fd, sb);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s32 PS4_SYSV_ABI posix_ftruncate(s32 fd, s64 length) {
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);

    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    if (file->type == Core::FileSys::FileType::Device) {
        s32 result = file->device->ftruncate(length);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }

    if (file->m_host_name.empty()) {
        *__Error() = POSIX_EACCES;
        return -1;
    }

    file->f.SetSize(length);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelFtruncate(s32 fd, s64 length) {
    s32 result = posix_ftruncate(fd, length);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s32 PS4_SYSV_ABI posix_rename(const char* from, const char* to) {
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();
    bool ro = false;
    const auto src_path = mnt->GetHostPath(from, &ro);
    if (strlen(from) > 255) {
        *__Error() = POSIX_ENAMETOOLONG;
        return -1;
    }
    if (strlen(to) > 255) {
        *__Error() = POSIX_ENAMETOOLONG;
        return -1;
    }
    if (!fs::exists(src_path)) {
        *__Error() = POSIX_ENOENT;
        return -1;
    }
    if (ro) {
        *__Error() = POSIX_EROFS;
        return -1;
    }
    const auto dst_path = mnt->GetHostPath(to, &ro);
    if (ro) {
        *__Error() = POSIX_EROFS;
        return -1;
    }
    const bool src_is_dir = fs::is_directory(src_path);
    const bool dst_is_dir = fs::is_directory(dst_path);

    if (fs::exists(dst_path)) {
        if (src_is_dir && !dst_is_dir) {
            *__Error() = POSIX_ENOTDIR;
            return -1;
        }
        if (!src_is_dir && dst_is_dir) {
            *__Error() = POSIX_EISDIR;
            return -1;
        }
        if (dst_is_dir && !fs::is_empty(dst_path)) {
            *__Error() = POSIX_ENOTEMPTY;
            return -1;
        }
    }

    fs::copy(src_path, dst_path,
             fs::copy_options::overwrite_existing | fs::copy_options::recursive);
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto file = h->GetFile(src_path);
    if (file) {
        auto access_mode = file->f.GetAccessMode();
        file->f.Close();
        fs::remove(src_path);
        file->f.Open(dst_path, access_mode);
    } else {
        fs::remove_all(src_path);
    }

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelRename(const char* from, const char* to) {
    s32 result = posix_rename(from, to);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 PS4_SYSV_ABI posix_preadv(s32 fd, OrbisKernelIovec* iov, s32 iovcnt, s64 offset) {
    if (offset < 0) {
        *__Error() = POSIX_EINVAL;
#ifdef __ANDROID__
        NoteResourceManifestRead(fd, "preadv", 0, offset, -1, nullptr, 0, offset);
#endif
        return -1;
    }

    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    std::scoped_lock lk{file->m_mutex};
    if (file->type == Core::FileSys::FileType::Device) {
        s64 result = file->device->preadv(iov, iovcnt, offset);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    } else if (file->type == Core::FileSys::FileType::Directory) {
        s64 result = file->directory->preadv(iov, iovcnt, offset);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }

    if (file->f.IsWriteOnly()) {
        *__Error() = POSIX_EBADF;
#ifdef __ANDROID__
        NoteResourceManifestRead(fd, "preadv", ResourceManifestIovecRequest(iov, iovcnt), offset,
                                 -1, nullptr, 0, offset);
        if (ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_PREADV_ERROR] fd=%d guest=%s iovcnt=%d offset=%lld "
                         "errno=%d reason=write_only thread=%s",
                         fd, file->m_guest_name.c_str(), iovcnt, static_cast<long long>(offset),
                         *__Error(), CurrentLiveThreadName());
        }
#endif
        return -1;
    }

    const s64 pos = file->f.Tell();
    SCOPE_EXIT {
        file->f.Seek(pos);
    };
    if (!file->f.Seek(offset)) {
        *__Error() = POSIX_EIO;
#ifdef __ANDROID__
        NoteResourceManifestRead(fd, "preadv", ResourceManifestIovecRequest(iov, iovcnt), offset,
                                 -1, nullptr, 0, offset);
        if (ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_PREADV_ERROR] fd=%d guest=%s iovcnt=%d offset=%lld "
                         "errno=%d reason=seek thread=%s",
                         fd, file->m_guest_name.c_str(), iovcnt, static_cast<long long>(offset),
                         *__Error(), CurrentLiveThreadName());
        }
#endif
        return -1;
    }
#ifdef __ANDROID__
    const bool trace_asset = ShouldEmitLiveFileIoLog(file);
    const u64 manifest_request = ResourceManifestIovecRequest(iov, iovcnt);
    const void* manifest_prefix_data = nullptr;
    s64 manifest_prefix_length = 0;
    s64 manifest_prefix_offset = offset;
    if (trace_asset) {
        AndroidFsLog("[EXECUTOR_FS_PREADV] fd=%d guest=%s iovcnt=%d offset=%lld saved_pos=%lld "
                     "size=%llu",
                     fd, file->m_guest_name.c_str(), iovcnt, static_cast<long long>(offset),
                     static_cast<long long>(pos),
                     static_cast<unsigned long long>(file->f.GetSize()));
    }
#endif
    s64 total_read = 0;
    for (s32 i = 0; i < iovcnt; i++) {
        const s64 chunk_before = file->f.Tell();
        const s64 read_result = ReadFile(file->f, iov[i].iov_base, iov[i].iov_len);
        total_read += read_result;
        if (read_result >= 0) {
            *__Error() = 0;
        }
#ifdef __ANDROID__
        if (manifest_prefix_data == nullptr && read_result > 0 && iov[i].iov_base != nullptr) {
            manifest_prefix_data = iov[i].iov_base;
            manifest_prefix_length = read_result;
            manifest_prefix_offset = chunk_before;
        }
        if (trace_asset && ShouldEmitLiveFileIoLog(file, iov[i].iov_len, read_result)) {
            AndroidFsLog("[EXECUTOR_FS_PREADV_IOV] fd=%d guest=%s index=%d request=%llu "
                         "result=%lld buf=%p thread=%s",
                         fd, file->m_guest_name.c_str(), i,
                         static_cast<unsigned long long>(iov[i].iov_len),
                         static_cast<long long>(read_result), iov[i].iov_base,
                         CurrentLiveThreadName());
            LogPs4StoreCoverPrefix("preadv", fd, file, i,
                                   static_cast<unsigned long long>(iov[i].iov_len), chunk_before,
                                   read_result, iov[i].iov_base);
        }
        if (read_result < 0 && ShouldTraceLiveFileErrors()) {
            AndroidFsLog("[EXECUTOR_FS_PREADV_ERROR] fd=%d guest=%s index=%d request=%llu "
                         "result=%lld errno=%d offset=%lld chunk_before=%lld thread=%s",
                         fd, file->m_guest_name.c_str(), i,
                         static_cast<unsigned long long>(iov[i].iov_len),
                         static_cast<long long>(read_result), *__Error(),
                         static_cast<long long>(offset), static_cast<long long>(chunk_before),
                         CurrentLiveThreadName());
        }
#endif
    }
#ifdef __ANDROID__
    NoteResourceManifestRead(fd, "preadv", manifest_request, offset, total_read,
                             manifest_prefix_data, manifest_prefix_length,
                             manifest_prefix_offset);
    if (trace_asset) {
        AndroidFsLog("[EXECUTOR_FS_PREADV_RESULT] fd=%d guest=%s total=%lld after=%lld errno=%d",
                     fd, file->m_guest_name.c_str(), static_cast<long long>(total_read),
                     static_cast<long long>(file->f.Tell()), *__Error());
    }
#endif
    if (total_read >= 0) {
        *__Error() = 0;
    }
    return total_read;
}

s64 PS4_SYSV_ABI sceKernelPreadv(s32 fd, OrbisKernelIovec* iov, s32 iovcnt, s64 offset) {
    s64 result = posix_preadv(fd, iov, iovcnt, offset);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 PS4_SYSV_ABI posix_pread(s32 fd, void* buf, u64 nbytes, s64 offset) {
    OrbisKernelIovec iovec{buf, nbytes};
    return posix_preadv(fd, &iovec, 1, offset);
}

s64 PS4_SYSV_ABI sceKernelPread(s32 fd, void* buf, u64 nbytes, s64 offset) {
    OrbisKernelIovec iovec{buf, nbytes};
    return sceKernelPreadv(fd, &iovec, 1, offset);
}

s32 PS4_SYSV_ABI posix_fsync(s32 fd) {
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    if (file->type == Core::FileSys::FileType::Device) {
        s32 result = file->device->fsync();
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }
    file->f.Flush();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelFsync(s32 fd) {
    s32 result = posix_fsync(fd);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

static s64 GetDents(s32 fd, char* buf, u64 nbytes, s64* basep) {
    if (buf == nullptr) {
        *__Error() = POSIX_EFAULT;
        return -1;
    }
    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    if (nbytes < 512) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }

    switch (file->type) {
    case Core::FileSys::FileType::Directory: {
        s64 result = file->directory->getdents(buf, nbytes, basep);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }
    case Core::FileSys::FileType::Device: {
        s64 result = file->device->getdents(buf, nbytes, basep);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    }
    default: {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    }

    return ORBIS_OK;
}

s64 PS4_SYSV_ABI posix_getdents(s32 fd, char* buf, u64 nbytes) {
    return GetDents(fd, buf, nbytes, nullptr);
}

s64 PS4_SYSV_ABI sceKernelGetdents(s32 fd, char* buf, u64 nbytes) {
    s64 result = posix_getdents(fd, buf, nbytes);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 PS4_SYSV_ABI getdirentries(s32 fd, char* buf, u64 nbytes, s64* basep) {
    return GetDents(fd, buf, nbytes, basep);
}

s64 PS4_SYSV_ABI posix_getdirentries(s32 fd, char* buf, u64 nbytes, s64* basep) {
    return GetDents(fd, buf, nbytes, basep);
}

s64 PS4_SYSV_ABI sceKernelGetdirentries(s32 fd, char* buf, u64 nbytes, s64* basep) {
    s64 result = GetDents(fd, buf, nbytes, basep);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 PS4_SYSV_ABI posix_pwritev(s32 fd, const OrbisKernelIovec* iov, s32 iovcnt, s64 offset) {
    if (offset < 0) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }

    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* file = h->GetFile(fd);
    if (file == nullptr) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    std::scoped_lock lk{file->m_mutex};

    if (file->type == Core::FileSys::FileType::Device) {
        s64 result = file->device->pwritev(iov, iovcnt, offset);
        if (result < 0) {
            ErrSceToPosix(result);
            return -1;
        }
        return result;
    } else if (file->type == Core::FileSys::FileType::Directory) {
        *__Error() = POSIX_EBADF;
        return -1;
    }

    const s64 pos = file->f.Tell();
    SCOPE_EXIT {
        file->f.Seek(pos);
    };
    if (!file->f.Seek(offset)) {
        *__Error() = POSIX_EIO;
        return -1;
    }
    s64 total_written = 0;
    for (s32 i = 0; i < iovcnt; i++) {
        total_written += file->f.WriteRaw<u8>(iov[i].iov_base, iov[i].iov_len);
    }
    return total_written;
}

s64 PS4_SYSV_ABI posix_pwrite(s32 fd, void* buf, u64 nbytes, s64 offset) {
    OrbisKernelIovec iovec{buf, nbytes};
    return posix_pwritev(fd, &iovec, 1, offset);
}

s64 PS4_SYSV_ABI sceKernelPwrite(s32 fd, void* buf, u64 nbytes, s64 offset) {
    s64 result = posix_pwrite(fd, buf, nbytes, offset);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s64 PS4_SYSV_ABI sceKernelPwritev(s32 fd, const OrbisKernelIovec* iov, s32 iovcnt, s64 offset) {
    s64 result = posix_pwritev(fd, iov, iovcnt, offset);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

s32 PS4_SYSV_ABI posix_unlink(const char* path) {
#ifdef __ANDROID__
    std::string recovered_path;
    path = ResolveAndroidGuestPathForFs(path, recovered_path);
    AndroidFsLog("[EXECUTOR_FS_UNLINK] path=%s", path ? path : "<null>");
#endif
    if (path == nullptr) {
        *__Error() = POSIX_EINVAL;
        return -1;
    }
    if (strlen(path) > 255) {
        *__Error() = POSIX_ENAMETOOLONG;
        return -1;
    }

    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    auto* mnt = Common::Singleton<Core::FileSys::MntPoints>::Instance();

    bool ro = false;
    const auto host_path = mnt->GetHostPath(path, &ro);
    if (host_path.empty()) {
        *__Error() = POSIX_ENOENT;
        return -1;
    }

    if (ro) {
        *__Error() = POSIX_EROFS;
        return -1;
    }

    if (fs::is_directory(host_path)) {
        *__Error() = POSIX_EPERM;
        return -1;
    }

    auto* file = h->GetFile(host_path);
    if (file == nullptr) {
        Common::FS::IOFile file(host_path, Common::FS::FileAccessMode::ReadWrite);
        file.Unlink();
    } else {
        file->f.Unlink();
    }

    LOG_INFO(Kernel_Fs, "Unlinked {}", path);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceKernelUnlink(const char* path) {
    s32 result = posix_unlink(path);
    if (result < 0) {
        LOG_ERROR(Kernel_Fs, "error = {}", *__Error());
        return ErrnoToSceKernelError(*__Error());
    }
    return result;
}

#ifdef _WIN32
#define __FD_SETSIZE 1024

typedef struct {
    unsigned long fds_bits[__FD_SETSIZE / (8 * sizeof(unsigned long))];
} fd_set_posix;

#define FD_SET_POSIX(fd, set)                                                                      \
    ((set)->fds_bits[(fd) / (8 * sizeof(unsigned long))] |=                                        \
     (1UL << ((fd) % (8 * sizeof(unsigned long)))))

#define FD_CLR_POSIX(fd, set)                                                                      \
    ((set)->fds_bits[(fd) / (8 * sizeof(unsigned long))] &=                                        \
     ~(1UL << ((fd) % (8 * sizeof(unsigned long)))))

#define FD_ISSET_POSIX(fd, set)                                                                    \
    (((set)->fds_bits[(fd) / (8 * sizeof(unsigned long))] &                                        \
      (1UL << ((fd) % (8 * sizeof(unsigned long))))) != 0)

#define FD_ZERO_POSIX(set) memset((set), 0, sizeof(fd_set_posix))

s32 PS4_SYSV_ABI posix_select(s32 nfds, fd_set_posix* readfds, fd_set_posix* writefds,
                              fd_set_posix* exceptfds, OrbisKernelTimeval* timeout) {
    LOG_DEBUG(Kernel_Fs, "nfds = {}, readfds = {}, writefds = {}, exceptfds = {}, timeout = {}",
              nfds, fmt::ptr(readfds), fmt::ptr(writefds), fmt::ptr(exceptfds), fmt::ptr(timeout));

    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();

    fd_set read_host = {}, write_host = {}, except_host = {};
    FD_ZERO(&read_host);
    FD_ZERO(&write_host);
    FD_ZERO(&except_host);

    fd_set_posix read_ready, write_ready, except_ready;
    FD_ZERO_POSIX(&read_ready);
    FD_ZERO_POSIX(&write_ready);
    FD_ZERO_POSIX(&except_ready);

    std::map<s32, s32> host_to_guest;
    s32 socket_max_fd = -1;

    for (s32 i = 0; i < nfds; ++i) {
        bool want_read = readfds && FD_ISSET_POSIX(i, readfds);
        bool want_write = writefds && FD_ISSET_POSIX(i, writefds);
        bool want_except = exceptfds && FD_ISSET_POSIX(i, exceptfds);
        if (!(want_read || want_write || want_except)) {
            continue;
        }

        auto* file = h->GetFile(i);
        if (!file || ((file->type == Core::FileSys::FileType::Regular && !file->f.IsOpen()) ||
                      (file->type == Core::FileSys::FileType::Socket && !file->is_opened))) {
            LOG_ERROR(Kernel_Fs, "fd {} is null or not opened", i);
            *__Error() = POSIX_EBADF;
            return -1;
        }

        s32 native_fd = -1;
        switch (file->type) {
        case Core::FileSys::FileType::Regular:
            native_fd = static_cast<s32>(file->f.GetFileMapping());
            break;
        case Core::FileSys::FileType::Socket: {
            auto sock = file->socket->Native();
            native_fd = sock ? static_cast<s32>(*sock) : -1;
            break;
        }
        case Core::FileSys::FileType::Device:
            native_fd = -1;
            break;
        default:
            UNREACHABLE();
            break;
        }

        if (file->type == Core::FileSys::FileType::Regular ||
            file->type == Core::FileSys::FileType::Device) {
            if (want_read && i != 0) {
                FD_SET_POSIX(i, &read_ready);
            }
            if (want_write) {
                FD_SET_POSIX(i, &write_ready);
            }
        } else if (file->type == Core::FileSys::FileType::Socket) {
            if (want_read) {
                FD_SET(native_fd, &read_host);
            }
            if (want_write) {
                FD_SET(native_fd, &write_host);
            }
            if (want_except) {
                FD_SET(native_fd, &except_host);
            }
            socket_max_fd = std::max(socket_max_fd, native_fd);
        }

        if (native_fd == -1) {
            continue;
        }

        host_to_guest[native_fd] = i;
    }

    LOG_DEBUG(Kernel_Fs,
              "Before select(): read_host.fd_count = {}, write_host.fd_count = {}, "
              "except_host.fd_count = {}",
              read_host.fd_count, write_host.fd_count, except_host.fd_count);

    if (read_host.fd_count == 0 && write_host.fd_count == 0 && except_host.fd_count == 0) {
        LOG_WARNING(Kernel_Fs, "No sockets in fd_sets, select() will return immediately");
    }

    if (readfds) {
        FD_ZERO_POSIX(readfds);
    }
    if (writefds) {
        FD_ZERO_POSIX(writefds);
    }
    if (exceptfds) {
        FD_ZERO_POSIX(exceptfds);
    }

    s32 result = 0;
    if (socket_max_fd != -1) {
        timeval tv = {};
        timeval* tv_ptr = nullptr;
        if (timeout) {
            tv.tv_sec = timeout->tv_sec;
            tv.tv_usec = timeout->tv_usec;
            tv_ptr = &tv;
        }
        result = select(0, read_host.fd_count > 0 ? &read_host : nullptr,
                        write_host.fd_count > 0 ? &write_host : nullptr,
                        except_host.fd_count > 0 ? &except_host : nullptr, tv_ptr);
        if (result == SOCKET_ERROR) {
            s32 err = WSAGetLastError();
            LOG_ERROR(Kernel_Fs, "select() failed with error {}", err);
            switch (err) {
            case WSAEFAULT:
                *__Error() = POSIX_EFAULT;
                break;
            case WSAEINVAL:
                *__Error() = POSIX_EINVAL;
                break;
            case WSAENOBUFS:
                *__Error() = POSIX_ENOBUFS;
                break;
            default:
                LOG_ERROR(Kernel_Fs, "Unhandled error case {}", err);
                break;
            }
            return -1;
        }

        for (s32 i = 0; i < read_host.fd_count; ++i) {
            s32 fd = static_cast<s32>(read_host.fd_array[i]);
            FD_SET_POSIX(host_to_guest[fd], readfds);
        }
        for (s32 i = 0; i < write_host.fd_count; ++i) {
            s32 fd = static_cast<s32>(write_host.fd_array[i]);
            FD_SET_POSIX(host_to_guest[fd], writefds);
        }
        for (s32 i = 0; i < except_host.fd_count; ++i) {
            s32 fd = static_cast<s32>(except_host.fd_array[i]);
            FD_SET_POSIX(host_to_guest[fd], exceptfds);
        }
    }

    s32 disk_ready = 0;
    for (s32 i = 0; i < nfds; ++i) {
        if (FD_ISSET_POSIX(i, &read_ready)) {
            FD_SET_POSIX(i, readfds);
            disk_ready++;
        }
        if (FD_ISSET_POSIX(i, &write_ready)) {
            FD_SET_POSIX(i, writefds);
            disk_ready++;
        }
    }

    return result + disk_ready;
}
#else
s32 PS4_SYSV_ABI posix_select(s32 nfds, fd_set* readfds, fd_set* writefds, fd_set* exceptfds,
                              OrbisKernelTimeval* timeout) {
    LOG_DEBUG(Kernel_Fs, "nfds = {}, readfds = {}, writefds = {}, exceptfds = {}, timeout = {}",
              nfds, fmt::ptr(readfds), fmt::ptr(writefds), fmt::ptr(exceptfds), fmt::ptr(timeout));

    auto* h = Common::Singleton<Core::FileSys::HandleTable>::Instance();
    fd_set read_host, write_host, except_host;
    FD_ZERO(&read_host);
    FD_ZERO(&write_host);
    FD_ZERO(&except_host);

    std::map<s32, s32> host_to_guest;
    s32 max_fd = -1;

    for (s32 i = 0; i < nfds; ++i) {
        auto read = readfds && FD_ISSET(i, readfds);
        auto write = writefds && FD_ISSET(i, writefds);
        auto except = exceptfds && FD_ISSET(i, exceptfds);
        if (read || write || except) {
            auto* file = h->GetFile(i);
            if (file == nullptr ||
                ((file->type == Core::FileSys::FileType::Regular && !file->f.IsOpen()) ||
                 (file->type == Core::FileSys::FileType::Socket && !file->is_opened))) {
                LOG_ERROR(Kernel_Fs, "fd {} is null or not opened", i);
                *__Error() = POSIX_EBADF;
                return -1;
            }

            s32 native_fd = [&] {
                switch (file->type) {
                case Core::FileSys::FileType::Regular:
                    return static_cast<s32>(file->f.GetFileMapping());
                case Core::FileSys::FileType::Device:
                    return -1;
                case Core::FileSys::FileType::Socket: {
                    auto sock = file->socket->Native();
                    return sock ? static_cast<s32>(*sock) : -1;
                }
                default:
                    UNREACHABLE();
                }
            }();
            if (native_fd == -1) {
                continue;
            }
            host_to_guest.emplace(native_fd, i);

            max_fd = std::max(max_fd, native_fd);

            if (read) {
                FD_SET(native_fd, &read_host);
            }
            if (write) {
                FD_SET(native_fd, &write_host);
            }
            if (except) {
                FD_SET(native_fd, &except_host);
            }
        }
    }

    if (max_fd == -1) {
        LOG_WARNING(Kernel_Fs, "all requested file descriptors are unsupported");
        return 0;
    }

    s32 ret = select(max_fd + 1, &read_host, &write_host, &except_host, (timeval*)timeout);

    if (ret > 0) {
        if (readfds) {
            FD_ZERO(readfds);
        }
        if (writefds) {
            FD_ZERO(writefds);
        }
        if (exceptfds) {
            FD_ZERO(exceptfds);
        }

        for (s32 i = 0; i < max_fd + 1; ++i) {
            if (readfds && FD_ISSET(i, &read_host)) {
                FD_SET(host_to_guest[i], readfds);
            }
            if (writefds && FD_ISSET(i, &write_host)) {
                FD_SET(host_to_guest[i], writefds);
            }
            if (exceptfds && FD_ISSET(i, &except_host)) {
                FD_SET(host_to_guest[i], exceptfds);
            }
        }
    }
    if (ret < 0) {
        s32 error = errno;
        LOG_ERROR(Kernel_Fs, "native select call failed with {} ({})", error,
                  Common::NativeErrorToString(error));
        SetPosixErrno(error);
    }

    return ret;
}
#endif

void RegisterFileSystem(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("6c3rCVE-fTU", "libkernel", 1, "libkernel", open);
    LIB_FUNCTION("wuCroIGjt2g", "libScePosix", 1, "libkernel", posix_open);
    LIB_FUNCTION("wuCroIGjt2g", "libkernel", 1, "libkernel", posix_open);
    LIB_FUNCTION("1G3lF1Gg1k8", "libkernel", 1, "libkernel", sceKernelOpen);
    LIB_FUNCTION("NNtFaKJbPt0", "libkernel", 1, "libkernel", close);
    LIB_FUNCTION("bY-PO6JhzhQ", "libScePosix", 1, "libkernel", posix_close);
    LIB_FUNCTION("bY-PO6JhzhQ", "libkernel", 1, "libkernel", posix_close);
    LIB_FUNCTION("UK2Tl2DWUns", "libkernel", 1, "libkernel", sceKernelClose);
    LIB_FUNCTION("FxVZqBAA7ks", "libkernel", 1, "libkernel", write);
    LIB_FUNCTION("FN4gaPmuFV8", "libScePosix", 1, "libkernel", posix_write);
    LIB_FUNCTION("FN4gaPmuFV8", "libkernel", 1, "libkernel", posix_write);
    LIB_FUNCTION("4wSze92BhLI", "libkernel", 1, "libkernel", sceKernelWrite);
    LIB_FUNCTION("+WRlkKjZvag", "libkernel", 1, "libkernel", readv);
    LIB_FUNCTION("YSHRBRLn2pI", "libkernel", 1, "libkernel", writev);
    LIB_FUNCTION("kAt6VDbHmro", "libkernel", 1, "libkernel", sceKernelWritev);
    LIB_FUNCTION("Oy6IpwgtYOk", "libScePosix", 1, "libkernel", posix_lseek);
    LIB_FUNCTION("Oy6IpwgtYOk", "libkernel", 1, "libkernel", posix_lseek);
    LIB_FUNCTION("oib76F-12fk", "libkernel", 1, "libkernel", sceKernelLseek);
    LIB_FUNCTION("DRuBt2pvICk", "libkernel", 1, "libkernel", read);
    LIB_FUNCTION("AqBioC2vF3I", "libScePosix", 1, "libkernel", posix_read);
    LIB_FUNCTION("AqBioC2vF3I", "libkernel", 1, "libkernel", posix_read);
    LIB_FUNCTION("Cg4srZ6TKbU", "libkernel", 1, "libkernel", sceKernelRead);
    LIB_FUNCTION("JGMio+21L4c", "libScePosix", 1, "libkernel", posix_mkdir);
    LIB_FUNCTION("JGMio+21L4c", "libkernel", 1, "libkernel", posix_mkdir);
    LIB_FUNCTION("1-LFLmRFxxM", "libkernel", 1, "libkernel", sceKernelMkdir);
    LIB_FUNCTION("c7ZnT7V1B98", "libScePosix", 1, "libkernel", posix_rmdir);
    LIB_FUNCTION("c7ZnT7V1B98", "libkernel", 1, "libkernel", posix_rmdir);
    LIB_FUNCTION("naInUjYt3so", "libkernel", 1, "libkernel", sceKernelRmdir);
    LIB_FUNCTION("E6ao34wPw+U", "libScePosix", 1, "libkernel", posix_stat);
    LIB_FUNCTION("E6ao34wPw+U", "libkernel", 1, "libkernel", posix_stat);
    LIB_FUNCTION("eV9wAD2riIA", "libkernel", 1, "libkernel", sceKernelStat);
    LIB_FUNCTION("uWyW3v98sU4", "libkernel", 1, "libkernel", sceKernelCheckReachability);
    LIB_FUNCTION("mqQMh1zPPT8", "libScePosix", 1, "libkernel", posix_fstat);
    LIB_FUNCTION("mqQMh1zPPT8", "libkernel", 1, "libkernel", posix_fstat);
    LIB_FUNCTION("kBwCPsYX-m4", "libkernel", 1, "libkernel", sceKernelFstat);
    LIB_FUNCTION("ih4CD9-gghM", "libkernel", 1, "libkernel", posix_ftruncate);
    LIB_FUNCTION("VW3TVZiM4-E", "libkernel", 1, "libkernel", sceKernelFtruncate);
    LIB_FUNCTION("NN01qLRhiqU", "libScePosix", 1, "libkernel", posix_rename);
    LIB_FUNCTION("NN01qLRhiqU", "libkernel", 1, "libkernel", posix_rename);
    LIB_FUNCTION("52NcYU9+lEo", "libkernel", 1, "libkernel", sceKernelRename);
    LIB_FUNCTION("yTj62I7kw4s", "libkernel", 1, "libkernel", sceKernelPreadv);
    LIB_FUNCTION("ezv-RSBNKqI", "libScePosix", 1, "libkernel", posix_pread);
    LIB_FUNCTION("ezv-RSBNKqI", "libkernel", 1, "libkernel", posix_pread);
    LIB_FUNCTION("+r3rMFwItV4", "libkernel", 1, "libkernel", sceKernelPread);
    LIB_FUNCTION("juWbTNM+8hw", "libScePosix", 1, "libkernel", posix_fsync);
    LIB_FUNCTION("juWbTNM+8hw", "libkernel", 1, "libkernel", posix_fsync);
    LIB_FUNCTION("fTx66l5iWIA", "libkernel", 1, "libkernel", sceKernelFsync);
    LIB_FUNCTION("j2AIqSqJP0w", "libkernel", 1, "libkernel", sceKernelGetdents);
    LIB_FUNCTION("sfKygSjIbI8", "libkernel", 1, "libkernel", getdirentries);
    LIB_FUNCTION("2G6i6hMIUUY", "libkernel", 1, "libkernel", posix_getdents);
    LIB_FUNCTION("taRWhTJFTgE", "libkernel", 1, "libkernel", sceKernelGetdirentries);
    LIB_FUNCTION("C2kJ-byS5rM", "libkernel", 1, "libkernel", posix_pwrite);
    LIB_FUNCTION("FCcmRZhWtOk", "libScePosix", 1, "libkernel", posix_pwritev);
    LIB_FUNCTION("FCcmRZhWtOk", "libkernel", 1, "libkernel", posix_pwritev);
    LIB_FUNCTION("nKWi-N2HBV4", "libkernel", 1, "libkernel", sceKernelPwrite);
    LIB_FUNCTION("mBd4AfLP+u8", "libkernel", 1, "libkernel", sceKernelPwritev);
    LIB_FUNCTION("VAzswvTOCzI", "libkernel", 1, "libkernel", posix_unlink);
    LIB_FUNCTION("AUXVxWeJU-A", "libkernel", 1, "libkernel", sceKernelUnlink);
    LIB_FUNCTION("T8fER+tIGgk", "libScePosix", 1, "libkernel", posix_select);
    LIB_FUNCTION("T8fER+tIGgk", "libkernel", 1, "libkernel", posix_select);
}

}
