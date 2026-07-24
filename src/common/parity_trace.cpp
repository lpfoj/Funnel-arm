// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/parity_trace.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>

#include "common/scm_rev.h"

namespace Common::ParityTrace {
namespace {

constexpr std::uint64_t DefaultEventCap = 4096;
constexpr std::uint64_t MaximumEventCap = 1'000'000;

constexpr std::array<std::string_view, 15> ForbiddenSemanticKeys{
    "timestamp",       "timestampns",    "walltime",       "walltimens",
    "hosttid",         "hostthreadid",   "hostptr",        "vkhandle",
    "pipelinehandle",  "bufferhandle",   "imagehandle",    "imageviewhandle",
    "semaphorehandle", "hostpointer",    "nativepointer",
};

constexpr std::string_view ToString(Domain value) {
    switch (value) {
    case Domain::Loader:
        return "loader";
    case Domain::Vm:
        return "vm";
    case Domain::Hle:
        return "hle";
    case Domain::Thread:
        return "thread";
    case Domain::Fs:
        return "fs";
    case Domain::Audio:
        return "audio";
    case Domain::Gnm:
        return "gnm";
    case Domain::Gpu:
        return "gpu";
    case Domain::VideoOut:
        return "videoout";
    case Domain::Frame:
        return "frame";
    }
    return "hle";
}

constexpr std::string_view ToString(Contract value) {
    switch (value) {
    case Contract::MustEqual:
        return "must_equal";
    case Contract::Capability:
        return "capability";
    case Contract::Health:
        return "health";
    }
    return "health";
}

constexpr std::string_view ToString(Stage value) {
    switch (value) {
    case Stage::None:
        return "none";
    case Stage::Graphics:
        return "graphics";
    case Stage::Vs:
        return "vs";
    case Stage::Ps:
        return "ps";
    case Stage::Gs:
        return "gs";
    case Stage::Es:
        return "es";
    case Stage::Hs:
        return "hs";
    case Stage::Ls:
        return "ls";
    case Stage::Cs:
        return "cs";
    }
    return "none";
}

std::optional<std::string> ReadIdentityHash(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return std::nullopt;
    }
    std::string value{raw};
    if (value.size() != 64 ||
        !std::ranges::all_of(value, [](unsigned char ch) { return std::isxdigit(ch) != 0; })) {
        return std::nullopt;
    }
    std::ranges::transform(value, value.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::optional<std::uint64_t> ReadEventCap() {
    const char* raw = std::getenv("SHADPS4_PARITY_MAX_EVENTS");
    if (raw == nullptr || *raw == '\0') {
        return DefaultEventCap;
    }
    std::uint64_t value{};
    const std::string_view text{raw};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0 ||
        value > MaximumEventCap) {
        return std::nullopt;
    }
    return value;
}

bool ReadDirtyState() {
    if (const char* raw = std::getenv("SHADPS4_PARITY_DIRTY")) {
        const std::string_view value{raw};
        if (value == "1" || value == "true" || value == "TRUE") {
            return true;
        }
        if (value == "0" || value == "false" || value == "FALSE") {
            return false;
        }
    }
    return std::string_view{Common::g_scm_desc}.find("dirty") != std::string_view::npos;
}

std::string NormalizeSemanticKey(std::string_view key) {
    std::string normalized;
    normalized.reserve(key.size());
    for (const unsigned char ch : key) {
        if (std::isalnum(ch) != 0) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return normalized;
}

bool IsSemanticDataValid(const Json& value) {
    if (value.is_number_float()) {
        return false;
    }
    if (value.is_object()) {
        for (auto entry = value.cbegin(); entry != value.cend(); ++entry) {
            const std::string normalized = NormalizeSemanticKey(entry.key());
            if (std::ranges::find(ForbiddenSemanticKeys, normalized) !=
                ForbiddenSemanticKeys.end()) {
                return false;
            }
            if (!IsSemanticDataValid(entry.value())) {
                return false;
            }
        }
    } else if (value.is_array()) {
        for (const auto& entry : value) {
            if (!IsSemanticDataValid(entry)) {
                return false;
            }
        }
    }
    return true;
}

class Writer {
public:
    Writer() noexcept {
        try {
            Initialize();
        } catch (const std::exception& exception) {
            Disable(exception.what());
        } catch (...) {
            Disable("unknown initialization error");
        }
    }

    [[nodiscard]] bool IsEnabled() const noexcept {
        return enabled.load(std::memory_order_acquire);
    }

    void Emit(Domain domain, std::string_view event, Contract contract, const Key& key,
              Json data) noexcept {
        if (!IsEnabled()) {
            return;
        }
        try {
            if (event.empty() || !data.is_object() || key.submit < -1 || key.draw < -1 ||
                key.slot < -1 || (contract != Contract::Health && !IsSemanticDataValid(data))) {
                Reject("invalid event or non-canonical semantic data");
                return;
            }

            std::scoped_lock lock{mutex};
            if (!enabled.load(std::memory_order_relaxed)) {
                return;
            }
            if (event_count >= event_cap) {
                MarkTruncatedLocked();
                return;
            }

            Json record = Json::object();
            record["kind"] = "event";
            record["domain"] = std::string{ToString(domain)};
            record["event"] = std::string{event};
            record["contract"] = std::string{ToString(contract)};
            Json event_key = Json::object();
            event_key["submit"] = key.submit;
            event_key["draw"] = key.draw;
            event_key["stage"] = std::string{ToString(key.stage)};
            event_key["slot"] = key.slot;
            event_key["ordinal"] = key.ordinal;
            record["key"] = std::move(event_key);
            record["data"] = std::move(data);
            stream << record.dump() << '\n';
            stream.flush();
            if (!stream) {
                DisableLocked("write failed");
                return;
            }
            ++event_count;
            public_event_count.store(event_count, std::memory_order_release);
        } catch (const std::exception& exception) {
            Reject(exception.what());
        } catch (...) {
            Reject("unknown event serialization error");
        }
    }

    void Flush() noexcept {
        std::scoped_lock lock{mutex};
        if (stream.is_open()) {
            stream.flush();
        }
    }

    [[nodiscard]] std::uint64_t EventCount() const noexcept {
        return public_event_count.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool WasTruncated() const noexcept {
        return truncated.load(std::memory_order_acquire);
    }

private:
    void Initialize() {
        const char* trace_path = std::getenv("SHADPS4_PARITY_TRACE");
        if (trace_path == nullptr || *trace_path == '\0') {
            return;
        }

        const auto eboot = ReadIdentityHash("SHADPS4_PARITY_EBOOT_SHA256");
        const auto modules = ReadIdentityHash("SHADPS4_PARITY_MODULE_SET_SHA256");
        const auto config = ReadIdentityHash("SHADPS4_PARITY_SEMANTIC_CONFIG_SHA256");
        const auto cap = ReadEventCap();
        if (!eboot || !modules || !config || !cap) {
            Disable("missing/invalid identity hash or SHADPS4_PARITY_MAX_EVENTS");
            return;
        }

        Json host_caps = Json::object();
        if (const char* raw_caps = std::getenv("SHADPS4_PARITY_HOST_CAPS_JSON")) {
            host_caps = Json::parse(raw_caps);
            if (!host_caps.is_object()) {
                Disable("SHADPS4_PARITY_HOST_CAPS_JSON must be a JSON object");
                return;
            }
        }

        const std::filesystem::path path{trace_path};
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }
        stream.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!stream) {
            Disable("cannot open output file");
            return;
        }

        const char* configured_side = std::getenv("SHADPS4_PARITY_SIDE");
#ifdef __ANDROID__
        constexpr std::string_view default_side = "android";
#else
        constexpr std::string_view default_side = "pc-reference";
#endif
        const std::string side = configured_side != nullptr && *configured_side != '\0'
                                     ? configured_side
                                     : std::string{default_side};
        event_cap = *cap;

        Json identity = Json::object();
        identity["ebootSha256"] = *eboot;
        identity["moduleSetSha256"] = *modules;
        identity["semanticConfigSha256"] = *config;

        Json build = Json::object();
        build["side"] = side;
        build["commit"] = std::string{Common::g_scm_rev};
        build["dirty"] = ReadDirtyState();
        build["branch"] = std::string{Common::g_scm_branch};
        build["version"] = std::string{Common::g_version};

        Json header = Json::object();
        header["kind"] = "header";
        header["schema"] = "shadps4-pc-android-parity";
        header["version"] = 1;
        header["identity"] = std::move(identity);
        header["build"] = std::move(build);
        header["hostCaps"] = std::move(host_caps);
        stream << header.dump() << '\n';
        stream.flush();
        if (!stream) {
            Disable("header write failed");
            return;
        }
        enabled.store(true, std::memory_order_release);
    }

    void Reject(std::string_view reason) noexcept {
        std::scoped_lock lock{mutex};
        DisableLocked(reason);
    }

    void MarkTruncatedLocked() noexcept {
        if (!truncated.exchange(true, std::memory_order_acq_rel) && stream.is_open()) {
            stream << "# parity trace event cap reached: " << event_cap << '\n';
            stream.flush();
        }
        enabled.store(false, std::memory_order_release);
    }

    void Disable(std::string_view reason) noexcept {
        std::scoped_lock lock{mutex};
        DisableLocked(reason);
    }

    void DisableLocked(std::string_view reason) noexcept {
        if (stream.is_open()) {
            stream << "# parity trace disabled: " << reason << '\n';
            stream.flush();
        }
        enabled.store(false, std::memory_order_release);
        std::fprintf(stderr, "shadPS4 parity trace disabled: %.*s\n", static_cast<int>(reason.size()),
                     reason.data());
    }

    mutable std::mutex mutex;
    std::ofstream stream;
    std::atomic_bool enabled{false};
    std::atomic_bool truncated{false};
    std::atomic_uint64_t public_event_count{};
    std::uint64_t event_cap{DefaultEventCap};
    std::uint64_t event_count{};
};

Writer& GetWriter() {
    static Writer writer;
    return writer;
}

}

bool IsEnabled() noexcept {
    return GetWriter().IsEnabled();
}

std::string Hex(std::uint64_t value) {
    std::array<char, 18> buffer{};
    buffer[0] = '0';
    buffer[1] = 'x';
    const auto [end, error] =
        std::to_chars(buffer.data() + 2, buffer.data() + buffer.size(), value, 16);
    if (error != std::errc{}) {
        return "0x0";
    }
    return std::string{buffer.data(), end};
}

void Emit(Domain domain, std::string_view event, Contract contract, const Key& key,
          Json data) noexcept {
    GetWriter().Emit(domain, event, contract, key, std::move(data));
}

void Flush() noexcept {
    GetWriter().Flush();
}

std::uint64_t EventCount() noexcept {
    return GetWriter().EventCount();
}

bool WasTruncated() noexcept {
    return GetWriter().WasTruncated();
}

}
