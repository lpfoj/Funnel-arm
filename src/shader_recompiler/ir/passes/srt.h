// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include <vector>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32*, u32*);
using PFN_SrtMemoryRead = bool (*)(VAddr, void*, u64);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

struct PortableSrtNode {
    s32 parent;
    u32 ptr_off_dw;
};
struct PortableSrtCopy {
    u32 node;
    u32 src_off_dw;
    u32 dst_off_dw;
};

struct PersistentSrtInfo {
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16;
    std::vector<PortableSrtNode> portable_nodes;
    std::vector<PortableSrtCopy> portable_copies;

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

struct SrtMemoryRange {
    VAddr address{};
    u64 size{};
};
using SrtMemoryRanges = boost::container::small_vector<SrtMemoryRange, 16>;

void RunPortableSrtWalker(const u32* user_data, u32* flat, const PersistentSrtInfo& srt,
                           u32 stage, u64 pgm_hash,
                           SrtMemoryRanges* memory_ranges = nullptr,
                           PFN_SrtMemoryRead memory_reader = nullptr);

}
