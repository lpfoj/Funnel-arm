// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"

namespace Shader {

struct TessellationDataConstantBuffer {
    u32 ls_stride;
    u32 hs_cp_stride;
    u32 num_patches;
    u32 hs_output_base;
    u32 patch_const_size;
    u32 patch_const_base;
    u32 patch_output_size;
    f32 off_chip_tessellation_factor_threshold;
    u32 first_edge_tess_factor_index;
};

enum class TessConstantAttribute : u32 {
    LsStride,
    HsCpStride,
    HsNumPatch,
    HsOutputBase,
    PatchConstSize,
    PatchConstBase,
    PatchOutputSize,
    OffChipTessellationFactorThreshold,
    FirstEdgeTessFactorIndex,
};

}
