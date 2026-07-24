// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <span>
#include <unordered_map>

#include "video_core/renderer_vulkan/vk_platform.h"

#define TRACY_VK_USE_SYMBOL_TABLE
#include <tracy/TracyVulkan.hpp>

namespace Frontend {
class WindowSDL;
}

VK_DEFINE_HANDLE(VmaAllocator)

namespace Vulkan {

void ExecutorRunGcnSpirvSmoke(const char* marker_path);
void ExecutorRunLayer3PipelinePrepSmoke(const char* marker_path);
void ExecutorRunLayer3PipelineShaderCompileSmoke(const char* marker_path);
void ExecutorRunLayer3GnmcapRelocSmoke(const char* marker_path);
void ExecutorRunVsCompileClassifySmoke(const char* marker_path);
void ExecutorRunGnmcapReplayDryRunSmoke(const char* marker_path);
void ExecutorRunGnmReplayWithMemorySmoke(const char* marker_path);

class Instance {
public:
    explicit Instance(bool validation = false, bool crash_diagnostic = false,
                      bool create_headless_device = false);
    explicit Instance(Frontend::WindowSDL& window, s32 physical_device_index,
                      bool enable_validation = false, bool enable_crash_diagnostic = false);
    ~Instance();

    std::string GetDriverVersionName();

    [[nodiscard]] vk::Format GetSupportedFormat(vk::Format format,
                                                vk::FormatFeatureFlags2 flags) const;

    vk::Instance GetInstance() const {
        return *instance;
    }

    vk::PhysicalDevice GetPhysicalDevice() const {
        return physical_device;
    }

    vk::Device GetDevice() const {
        return *device;
    }

    VmaAllocator GetAllocator() const {
        return allocator;
    }

    std::span<const vk::PhysicalDevice> GetPhysicalDevices() const {
        return physical_devices;
    }

    u32 GetGraphicsQueueFamilyIndex() const {
        return queue_family_index;
    }

    u32 GetPresentQueueFamilyIndex() const {
        return queue_family_index;
    }

    vk::Queue GetGraphicsQueue() const {
        return graphics_queue;
    }

    vk::Queue GetPresentQueue() const {
        return present_queue;
    }

    TracyVkCtx GetProfilerContext() const {
        return profiler_context;
    }

    bool IsAnisotropicFilteringSupported() const {
        return features.samplerAnisotropy;
    }

    bool IsDepthBoundsSupported() const {
        return features.depthBounds;
    }

    bool IsShaderFloat16Supported() const {
        return vk12_features.shaderFloat16;
    }

    bool IsShaderFloat64Supported() const {
        return features.shaderFloat64;
    }

    bool IsShaderInt64Supported() const {
        return features.shaderInt64;
    }

    bool IsShaderInt16Supported() const {
        return features.shaderInt16;
    }

    bool IsShaderInt8Supported() const {
        return vk12_features.shaderInt8;
    }

    bool IsMaintenance8Supported() const {
        return maintenance_8;
    }

    bool IsAttachmentFeedbackLoopLayoutSupported() const {
        return attachment_feedback_loop;
    }

    bool IsCustomBorderColorSupported() const {
        return custom_border_color;
    }

    bool IsShaderStencilExportSupported() const {
        return shader_stencil_export;
    }

    bool IsDepthClipControlSupported() const {
        return depth_clip_control;
    }

    bool IsDepthClipEnableSupported() const {
        return depth_clip_enable;
    }

    bool IsDepthRangeUnrestrictedSupported() const {
        return depth_range_unrestricted;
    }

    bool IsExtendedDynamicState3Supported() const {
        return dynamic_state_3;
    }

    bool IsDynamicColorWriteMaskSupported() const {
        return dynamic_state_3 && dynamic_state_3_features.extendedDynamicState3ColorWriteMask;
    }

    bool IsCoarseFragmentShadingSupported() const {
        return fragment_shading_rate && coarse_fragment_size.width > 1 &&
               coarse_fragment_size.height > 1;
    }

    vk::Extent2D GetCoarseFragmentSize() const {
        return coarse_fragment_size;
    }

    bool IsVertexInputDynamicState() const {
        return vertex_input_dynamic_state;
    }

    bool IsRobustBufferAccess2Supported() const {
        return robustness2 && robustness2_features.robustBufferAccess2;
    }

    bool IsRobustImageAccess2Supported() const {
        return robustness2 && robustness2_features.robustImageAccess2;
    }

    bool IsNullDescriptorSupported() const {
        return robustness2 && robustness2_features.nullDescriptor;
    }

    bool IsDeviceFaultSupported() const {
        return device_fault && device_fault_features.deviceFault;
    }

    void DumpDeviceFaultInfo(const char* where, u64 failing_tick) const noexcept;

    bool IsFragmentShaderBarycentricSupported() const {
        return fragment_shader_barycentric;
    }

    bool IsAmdShaderExplicitVertexParameterSupported() const {
        return amd_shader_explicit_vertex_parameter;
    }

    bool IsListRestartSupported() const {
        return list_restart && list_restart_features.primitiveTopologyListRestart;
    }

    bool IsPatchListRestartSupported() const {
        return list_restart && list_restart_features.primitiveTopologyPatchListRestart;
    }

    bool IsLegacyVertexAttributesSupported() const {
        return legacy_vertex_attributes;
    }

    bool IsProvokingVertexSupported() const {
        return provoking_vertex;
    }

    bool IsImageLoadStoreLodSupported() const {
        return image_load_store_lod;
    }

    bool IsAmdGcnShaderSupported() const {
        return amd_gcn_shader;
    }

    bool IsAmdShaderTrinaryMinMaxSupported() const {
        return amd_shader_trinary_minmax;
    }

    bool IsShaderAtomicFloatBuffer32MinMaxSupported() const {
        return shader_atomic_float2 &&
               shader_atomic_float2_features.shaderBufferFloat32AtomicMinMax;
    }

    bool IsShaderAtomicFloatImage32MinMaxSupported() const {
        return shader_atomic_float2 && shader_atomic_float2_features.shaderImageFloat32AtomicMinMax;
    }

    bool IsBufferInt64AtomicsSupported() const {
        return vk12_features.shaderBufferInt64Atomics;
    }

    bool IsSharedInt64AtomicsSupported() const {
        return vk12_features.shaderSharedInt64Atomics;
    }

    bool IsSubgroupSize64Supported() const {
        return vk13_features.subgroupSizeControl && vk13_props.maxSubgroupSize >= 64;
    }

    bool IsWorkgroupMemoryExplicitLayoutSupported() const {
        return workgroup_memory_explicit_layout &&
               workgroup_memory_explicit_layout_features.workgroupMemoryExplicitLayout16BitAccess;
    }

    bool IsMixedDepthSamplesSupported() const {
        return nv_framebuffer_mixed_samples || amd_mixed_attachment_samples;
    }

    bool IsMixedAnySamplesSupported() const {
        return amd_mixed_attachment_samples;
    }

    bool IsGeometryStageSupported() const {
        return features.geometryShader;
    }

    bool IsTessellationSupported() const {
        return features.tessellationShader;
    }

    bool IsTessellationIsolinesSupported() const {
        return !portability_subset || portability_features.tessellationIsolines;
    }

    bool IsTessellationPointModeSupported() const {
        return !portability_subset || portability_features.tessellationPointMode;
    }

    u32 GetVendorID() const {
        return properties.vendorID;
    }

    u32 GetDeviceID() const {
        return properties.deviceID;
    }

    vk::DriverId GetDriverID() const {
        return driver_id;
    }

    u32 GetDriverVersion() const {
        return properties.driverVersion;
    }

    u32 ApiVersion() const {
        return properties.apiVersion;
    }

    std::string_view GetVendorName() const {
        return vendor_name;
    }

    std::span<const std::string> GetAvailableExtensions() const {
        return available_extensions;
    }

    std::string_view GetModelName() const {
        return properties.deviceName;
    }

    bool IsIntegrated() const {
        return properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
    }

    const auto GetPipelineCacheUUID() const {
        return properties.pipelineCacheUUID;
    }

    vk::DeviceSize UniformMinAlignment() const {
        return properties.limits.minUniformBufferOffsetAlignment;
    }

    vk::DeviceSize UniformMaxSize() const {
        return properties.limits.maxUniformBufferRange;
    }

    vk::DeviceSize StorageMinAlignment() const {
        return properties.limits.minStorageBufferOffsetAlignment;
    }

    vk::DeviceSize NonCoherentAtomSize() const {
        return properties.limits.nonCoherentAtomSize;
    }

    u32 SubgroupSize() const {
        return vk11_props.subgroupSize;
    }

    u32 MaxComputeSharedMemorySize() const {
        return properties.limits.maxComputeSharedMemorySize;
    }

    float MaxSamplerLodBias() const {
        return properties.limits.maxSamplerLodBias;
    }

    float MaxSamplerAnisotropy() const {
        return properties.limits.maxSamplerAnisotropy;
    }

    u32 MaxPushDescriptors() const {
        return push_descriptor_props.maxPushDescriptors;
    }

    const vk::PhysicalDeviceVulkan12Properties& GetVk12Properties() const noexcept {
        return vk12_props;
    }

    const vk::PhysicalDeviceMemoryProperties& GetMemoryProperties() const noexcept {
        return memory_properties;
    }

    bool IsShaderClipDistanceSupported() const {
        return features.shaderClipDistance;
    }

    u32 GetMaxViewportWidth() const {
        return properties.limits.maxViewportDimensions[0];
    }

    u32 GetMaxViewportHeight() const {
        return properties.limits.maxViewportDimensions[1];
    }

    u32 GetMaxFramebufferWidth() const {
        return properties.limits.maxFramebufferWidth;
    }

    u32 GetMaxFramebufferHeight() const {
        return properties.limits.maxFramebufferHeight;
    }

    u32 GetMaxSamplerAllocationCount() const {
        if (driver_id == vk::DriverId::eMesaHoneykrisp) {
            return 1024;
        }
        return properties.limits.maxSamplerAllocationCount;
    }

    vk::SampleCountFlags GetColorSampleCounts() const {
        return properties.limits.framebufferColorSampleCounts;
    }

    vk::SampleCountFlags GetDepthSampleCounts() const {
        return properties.limits.framebufferDepthSampleCounts &
               properties.limits.framebufferStencilSampleCounts;
    }

    bool IsLogicOpSupported() const {
        return features.logicOp;
    }

    bool IsBlockTexelViewSupported() const {
        return supports_block_texel_view;
    }

    bool Is2dViewOf3dSupported() const {
        return image_2d_view_of_3d && image_2d_view_of_3d_features.image2DViewOf3D &&
               image_2d_view_of_3d_features.sampler2DViewOf3D;
    }

    bool CanReportMemoryUsage() const {
        return supports_memory_budget;
    }

    [[nodiscard]] u64 GetDeviceMemoryUsage() const;

    [[nodiscard]] u64 GetTotalMemoryBudget() const {
        return total_memory_budget;
    }

    [[nodiscard]] bool IsFormatSupported(vk::Format format, vk::FormatFeatureFlags2 flags) const;

private:
    bool CreateDevice();

    void CreateAllocator();

    void CollectDeviceParameters();
    void CollectPhysicalMemoryInfo();
    void CollectImageFormatInfo();
    void CollectToolingInfo() const;

    [[nodiscard]] vk::FormatFeatureFlags2 GetFormatFeatureFlags(vk::Format format) const;

private:
    vk::UniqueInstance instance;
    vk::PhysicalDevice physical_device;
    vk::UniqueDevice device;
    vk::PhysicalDeviceProperties properties;
    vk::PhysicalDeviceMemoryProperties memory_properties;
    vk::PhysicalDeviceVulkan11Properties vk11_props;
    vk::PhysicalDeviceVulkan12Properties vk12_props;
    vk::PhysicalDeviceVulkan13Properties vk13_props;
    vk::PhysicalDevicePushDescriptorPropertiesKHR push_descriptor_props;
    vk::PhysicalDeviceFeatures features;
    vk::PhysicalDeviceVulkan12Features vk12_features;
    vk::PhysicalDeviceVulkan13Features vk13_features;
    vk::PhysicalDevicePortabilitySubsetFeaturesKHR portability_features;
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic_state_3_features;
    vk::PhysicalDeviceFragmentShadingRateFeaturesKHR fragment_shading_rate_features;
    vk::PhysicalDeviceRobustness2FeaturesEXT robustness2_features;
    vk::PhysicalDeviceFaultFeaturesEXT device_fault_features;
    vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT shader_atomic_float2_features;
    vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR
        workgroup_memory_explicit_layout_features;
    vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT image_2d_view_of_3d_features;
    vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT list_restart_features;
    vk::DriverIdKHR driver_id;
    vk::UniqueDebugUtilsMessengerEXT debug_callback{};
    std::string vendor_name;
    VmaAllocator allocator{};
    vk::Queue present_queue;
    vk::Queue graphics_queue;
    std::vector<vk::PhysicalDevice> physical_devices;
    std::vector<std::string> available_extensions;
    std::unordered_map<vk::Format, vk::FormatProperties3> format_properties;
    TracyVkCtx profiler_context{};
    u32 queue_family_index{0};
    vk::Extent2D coarse_fragment_size{1, 1};
    bool custom_border_color{};
    bool fragment_shader_barycentric{};
    bool amd_shader_explicit_vertex_parameter{};
    bool depth_clip_control{};
    bool depth_clip_enable{};
    bool dynamic_state_3{};
    bool fragment_shading_rate{};
    bool depth_range_unrestricted{};
    bool vertex_input_dynamic_state{};
    bool robustness2{};
    bool device_fault{};
    bool list_restart{};
    bool legacy_vertex_attributes{};
    bool provoking_vertex{};
    bool shader_stencil_export{};
    bool image_load_store_lod{};
    bool amd_gcn_shader{};
    bool amd_shader_trinary_minmax{};
    bool nv_framebuffer_mixed_samples{};
    bool amd_mixed_attachment_samples{};
    bool shader_atomic_float{};
    bool shader_atomic_float2{};
    bool workgroup_memory_explicit_layout{};
    bool image_2d_view_of_3d{};
    bool portability_subset{};
    bool maintenance_8{};
    bool attachment_feedback_loop{};
    bool supports_memory_budget{};
    bool supports_block_texel_view{};
    mutable std::atomic_flag device_fault_dumped = ATOMIC_FLAG_INIT;
    mutable std::atomic<u64> memory_usage_cache{};
    mutable std::atomic<u64> memory_usage_sample_time_ns{};
    mutable std::atomic_flag memory_usage_query_in_flight = ATOMIC_FLAG_INIT;
    u64 total_memory_budget{};
    std::vector<size_t> valid_heaps;
};

}
