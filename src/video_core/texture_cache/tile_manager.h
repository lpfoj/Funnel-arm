// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>

#include "common/types.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/buffer_cache/buffer.h"

namespace Vulkan {
class CommandPool;
}

namespace VideoCore {

struct ImageInfo;
struct Image;
class StreamBuffer;

class TileManager {
    static constexpr size_t NUM_BPPS = 5;

public:
    struct ScratchBuffer {
        vk::Buffer buffer{};
        VmaAllocation allocation{};
        u64 bucket{};
        bool reused{};
    };
    struct Result {
        vk::Buffer buffer{};
        u32 offset{};
        ScratchBuffer scratch{};
    };

    explicit TileManager(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         StreamBuffer& stream_buffer);
    ~TileManager();

    void TileImage(Image& in_image, std::span<vk::BufferImageCopy> buffer_copies,
                   vk::Buffer out_buffer, u32 out_offset, u32 copy_size);

    Result DetileImage(vk::Buffer in_buffer, u32 in_offset, const ImageInfo& info);
    void RetireDetileResult(Result result);

private:
    struct ScratchPoolState;

    vk::Pipeline GetTilingPipeline(const ImageInfo& info, bool is_tiler);
    ScratchBuffer GetScratchBuffer(u32 size);
    void RetireScratchBuffer(ScratchBuffer scratch);

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    StreamBuffer& stream_buffer;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pl_layout;
    std::array<vk::UniquePipeline, AmdGpu::NUM_TILE_MODES * NUM_BPPS> detilers{};
    std::array<vk::UniquePipeline, AmdGpu::NUM_TILE_MODES * NUM_BPPS> tilers{};
    std::shared_ptr<ScratchPoolState> scratch_pool;
#ifdef __ANDROID__
    std::unique_ptr<Vulkan::CommandPool> detile_command_pool;
#endif
};

}
