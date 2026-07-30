// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>

#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module,
                                 SerializationSupport& sdata, bool preloading,
                                 vk::PipelineCreateFlags build_flags)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, true},
      compute_key{compute_key_} {
    auto& info = stages[int(Shader::LogicalStage::Compute)];
    info = &info_;
    const auto debug_str = GetDebugString();

    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .pNext = instance.IsSubgroupSize64Supported() ? &subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

    u32 binding{};
    size_t storage_buffer_index{};
    if (!preloading) {
        sdata.storage_buffers.clear();
        sdata.storage_buffers.reserve(info->buffers.size());
        sdata.image_bindings.clear();
        sdata.image_bindings.reserve(info->images.size());
    }
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    for (const auto& buffer : info->buffers) {
        const auto sharp = preloading ? AmdGpu::Buffer{} : buffer.GetSharp(*info);
        const bool is_storage =
            preloading
                ? storage_buffer_index < sdata.storage_buffers.size() &&
                      sdata.storage_buffers[storage_buffer_index] != 0
                : buffer.IsStorage(sharp);
        if (!preloading) {
            sdata.storage_buffers.push_back(is_storage ? 1u : 0u);
        }
        ++storage_buffer_index;
        bindings.push_back({
            .binding = binding++,
            .descriptorType = is_storage ? vk::DescriptorType::eStorageBuffer
                                         : vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    size_t image_binding_index{};
    for (const auto& image : info->images) {
        const u32 num_bindings =
            preloading
                ? (image_binding_index < sdata.image_bindings.size()
                       ? sdata.image_bindings[image_binding_index]
                       : 1u)
                : image.NumBindings(*info);
        if (!preloading) {
            sdata.image_bindings.push_back(num_bindings);
        }
        ++image_binding_index;
        bindings.push_back({
            .binding = binding,
            .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                               : vk::DescriptorType::eSampledImage,
            .descriptorCount = num_bindings,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
        binding += num_bindings;
    }
    for (const auto& sampler : info->samplers) {
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }

    const vk::PushConstantRange push_constants = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    uses_push_descriptors = binding <= instance.MaxPushDescriptors();
    const auto flags = uses_push_descriptors
                           ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                           : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    const auto device = instance.GetDevice();
    auto [descriptor_set_result, descriptor_set] =
        device.createDescriptorSetLayoutUnique(desc_layout_ci);
    if (descriptor_set_result != vk::Result::eSuccess) {
        build_result = descriptor_set_result;
        LOG_ERROR(Render_Vulkan, "Failed to create compute descriptor set layout: {}",
                  vk::to_string(descriptor_set_result));
        return;
    }
    desc_layout = std::move(descriptor_set);

    const vk::DescriptorSetLayout set_layout = *desc_layout;
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
    if (layout_result != vk::Result::eSuccess) {
        build_result = layout_result;
        LOG_ERROR(Render_Vulkan, "Failed to create compute pipeline layout: {}",
                  vk::to_string(layout_result));
        return;
    }
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "Compute PipelineLayout {}", debug_str);

    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .flags = build_flags,
        .stage = shader_ci,
        .layout = *pipeline_layout,
    };
    auto [pipeline_result, pipe] =
        instance.GetDevice().createComputePipelineUnique(pipeline_cache, compute_pipeline_ci);
    build_result = pipeline_result;
    if (pipeline_result != vk::Result::eSuccess) {
        if (pipeline_result != vk::Result::ePipelineCompileRequired) {
            LOG_ERROR(Render_Vulkan, "Failed to create compute pipeline: {}",
                      vk::to_string(pipeline_result));
        }
        return;
    }
    pipeline = std::move(pipe);
    SetObjectName(device, *pipeline, "Compute Pipeline {}", debug_str);
}

ComputePipeline::~ComputePipeline() = default;

}
