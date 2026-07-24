// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace Common::ParityTrace {

enum class Domain {
    Loader,
    Vm,
    Hle,
    Thread,
    Fs,
    Audio,
    Gnm,
    Gpu,
    VideoOut,
    Frame,
};

enum class Contract {
    MustEqual,
    Capability,
    Health,
};

enum class Stage {
    None,
    Graphics,
    Vs,
    Ps,
    Gs,
    Es,
    Hs,
    Ls,
    Cs,
};

struct Key {
    std::int64_t submit{-1};
    std::int64_t draw{-1};
    Stage stage{Stage::None};
    std::int64_t slot{-1};
    std::uint64_t ordinal{};
};

using Json = nlohmann::json;

[[nodiscard]] bool IsEnabled() noexcept;

[[nodiscard]] std::string Hex(std::uint64_t value);

void Emit(Domain domain, std::string_view event, Contract contract, const Key& key,
          Json data) noexcept;

void Flush() noexcept;
[[nodiscard]] std::uint64_t EventCount() noexcept;
[[nodiscard]] bool WasTruncated() noexcept;

}
