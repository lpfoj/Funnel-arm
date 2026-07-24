// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright © 2023 Skyline Team and Contributors (https://github.com/skyline-emu/)
// Copyright © 2015-2023 The Khronos Group Inc.
// Copyright © 2015-2023 Valve Corporation
// Copyright © 2015-2023 LunarG, Inc.

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

bool IsVulkanFormatCompatible(vk::Format base, vk::Format view);

}
