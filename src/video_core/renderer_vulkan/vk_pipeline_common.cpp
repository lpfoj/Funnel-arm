// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/static_vector.hpp>

#include <bit>

#include "common/content_fingerprint.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {
namespace {

u64 DescriptorWritesSignature(const Pipeline::DescriptorWrites& writes) {
    Common::ContentFingerprint64 fingerprint{Common::FingerprintDomain::DescriptorWrites};
    const auto mix = [&fingerprint](const u64 value) {
        fingerprint.UpdateLittleEndian(value);
    };
    mix(writes.size());
    for (const auto& write : writes) {
        mix(write.dstBinding);
        mix(write.dstArrayElement);
        mix(write.descriptorCount);
        mix(static_cast<u32>(write.descriptorType));
        for (u32 index = 0; index < write.descriptorCount; ++index) {
            if (write.pBufferInfo != nullptr) {
                const auto& info = write.pBufferInfo[index];
                mix(std::bit_cast<u64>(static_cast<VkBuffer>(info.buffer)));
                mix(info.offset);
                mix(info.range);
            } else if (write.pImageInfo != nullptr) {
                const auto& info = write.pImageInfo[index];
                mix(std::bit_cast<u64>(static_cast<VkSampler>(info.sampler)));
                mix(std::bit_cast<u64>(static_cast<VkImageView>(info.imageView)));
                mix(static_cast<u32>(info.imageLayout));
            } else if (write.pTexelBufferView != nullptr) {
                mix(std::bit_cast<u64>(
                    static_cast<VkBufferView>(write.pTexelBufferView[index])));
            }
        }
    }
    return fingerprint.Finish();
}

struct DescriptorBindCache {
    const Pipeline* pipeline{};
    u64 cmdbuf{};
    u64 recording_tick{};
    u64 signature{};
    bool valid{};
};

}

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, vk::PipelineCache pipeline_cache,
                   bool is_compute_)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() = default;

void Pipeline::BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data) const {
    const auto cmdbuf = scheduler.CommandBuffer();
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;

    if (!buffer_barriers.empty()) {
        const auto dependencies = vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = u32(buffer_barriers.size()),
            .pBufferMemoryBarriers = buffer_barriers.data(),
        };
        scheduler.EndRendering();
        cmdbuf.pipelineBarrier2(dependencies);
    }

    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    cmdbuf.pushConstants(*pipeline_layout, stage_flags, 0u, sizeof(push_data), &push_data);

    if (set_writes.empty()) {
        return;
    }

    static thread_local std::array<DescriptorBindCache, 2> bind_caches{};
    auto& bind_cache = bind_caches[IsCompute() ? 1u : 0u];
    const u64 raw_cmdbuf =
        std::bit_cast<u64>(static_cast<VkCommandBuffer>(cmdbuf));
    const u64 recording_tick = scheduler.CurrentTick();
    const u64 descriptor_signature = DescriptorWritesSignature(set_writes);
    if (bind_cache.valid && bind_cache.pipeline == this &&
        bind_cache.cmdbuf == raw_cmdbuf &&
        bind_cache.recording_tick == recording_tick &&
        bind_cache.signature == descriptor_signature) {
        return;
    }

    if (uses_push_descriptors) {
        cmdbuf.pushDescriptorSetKHR(bind_point, *pipeline_layout, 0, set_writes);
        bind_cache = {
            .pipeline = this,
            .cmdbuf = raw_cmdbuf,
            .recording_tick = recording_tick,
            .signature = descriptor_signature,
            .valid = true,
        };
        return;
    }

    const auto desc_set = desc_heap.Commit(*desc_layout);
    for (auto& set_write : set_writes) {
        set_write.dstSet = desc_set;
    }
    instance.GetDevice().updateDescriptorSets(set_writes, {});
    cmdbuf.bindDescriptorSets(bind_point, *pipeline_layout, 0, desc_set, {});
    bind_cache = {
        .pipeline = this,
        .cmdbuf = raw_cmdbuf,
        .recording_tick = recording_tick,
        .signature = descriptor_signature,
        .valid = true,
    };
}

std::string Pipeline::GetDebugString() const {
    std::string stage_desc;
    for (const auto& stage : stages) {
        if (stage) {
            const auto shader_name = PipelineCache::GetShaderName(stage->stage, stage->pgm_hash);
            if (stage_desc.empty()) {
                stage_desc = shader_name;
            } else {
                stage_desc = fmt::format("{},{}", stage_desc, shader_name);
            }
        }
    }
    return stage_desc;
}

}
