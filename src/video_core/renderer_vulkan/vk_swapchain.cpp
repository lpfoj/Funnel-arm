// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <limits>
#include "common/assert.h"
#include "common/config.h"
#include "common/logging/log.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace Vulkan {

#ifdef __ANDROID__
static bool ExecutorTraceVulkanPresent() {
    static const bool enabled = std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr ||
                                std::getenv("EXECUTOR_TRACE_PM4") != nullptr ||
                                std::getenv("EXECUTOR_TRACE_VULKAN_PRESENT") != nullptr;
    return enabled;
}
#endif

static constexpr vk::SurfaceFormatKHR SURFACE_FORMAT_HDR = {
    .format = vk::Format::eA2B10G10R10UnormPack32,
    .colorSpace = vk::ColorSpaceKHR::eHdr10St2084EXT,
};

Swapchain::Swapchain(const Instance& instance_, const Frontend::WindowSDL& window_)
    : instance{instance_}, window{window_}, surface{CreateSurface(instance.GetInstance(), window)} {
    FindPresentFormat();
    FindPresentMode();

    Create(window.GetWidth(), window.GetHeight());
    ImGui::Core::Initialize(instance, window, image_count, surface_format.format);
}

Swapchain::~Swapchain() {
    Destroy();
    instance.GetInstance().destroySurfaceKHR(surface);
}

void Swapchain::Create(u32 width_, u32 height_) {
    width = width_;
    height = height_;
    needs_recreation = false;

    Destroy();

    if (!SetSurfaceProperties()) {
        needs_recreation = true;
        return;
    }

    const std::array queue_family_indices = {
        instance.GetGraphicsQueueFamilyIndex(),
        instance.GetPresentQueueFamilyIndex(),
    };

    const bool exclusive = queue_family_indices[0] == queue_family_indices[1];
    const u32 queue_family_indices_count = exclusive ? 1u : 2u;
    const vk::SharingMode sharing_mode =
        exclusive ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent;
    const auto format = needs_hdr ? SURFACE_FORMAT_HDR : surface_format;
    const vk::SwapchainCreateInfoKHR swapchain_info = {
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = sharing_mode,
        .queueFamilyIndexCount = queue_family_indices_count,
        .pQueueFamilyIndices = queue_family_indices.data(),
        .preTransform = transform,
        .compositeAlpha = composite_alpha,
        .presentMode = present_mode,
        .clipped = true,
        .oldSwapchain = nullptr,
    };

    LOG_INFO(Render_Vulkan,
             "EXECUTOR_VULKAN_WSI phase=create_before surfaceCurrentTransform={} "
             "swapchainPreTransform={} aligned={}",
             vk::to_string(surface_current_transform), vk::to_string(swapchain_info.preTransform),
             surface_current_transform == swapchain_info.preTransform ? "YES" : "NO");
    auto [swapchain_result, chain] = instance.GetDevice().createSwapchainKHR(swapchain_info);
    if (swapchain_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan,
                    "[EXECUTOR_VULKAN_WSI] createSwapchainKHR failed: {} — will retry next present",
                    vk::to_string(swapchain_result));
        needs_recreation = true;
        return;
    }
    swapchain = chain;
    LOG_INFO(Render_Vulkan,
             "EXECUTOR_VULKAN_WSI phase=create_after result={} surfaceCurrentTransform={} "
             "swapchainPreTransform={} aligned={}",
             vk::to_string(swapchain_result), vk::to_string(surface_current_transform),
             vk::to_string(swapchain_info.preTransform),
             surface_current_transform == swapchain_info.preTransform ? "YES" : "NO");

    SetupImages();
    RefreshSemaphores();
}

void Swapchain::Recreate(u32 width_, u32 height_) {
    LOG_DEBUG(Render_Vulkan, "Recreate the swapchain: width={} height={} HDR={}", width_, height_,
              needs_hdr);
    Create(width_, height_);
}

void Swapchain::SetHDR(bool hdr) {
    if (needs_hdr == hdr) {
        return;
    }

    auto result = instance.GetDevice().waitIdle();
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(ImGui, "Failed to wait for Vulkan device idle on mode change: {}",
                    vk::to_string(result));
    }

    needs_hdr = hdr;
    Recreate(width, height);
    ImGui::Core::OnSurfaceFormatChange(needs_hdr ? SURFACE_FORMAT_HDR.format
                                                 : surface_format.format);
}

bool Swapchain::AcquireNextImage() {
    if (terminal) {
        return false;
    }
    if (!swapchain || image_acquired.empty()) {
        needs_recreation = true;
        return false;
    }
    vk::Device device = instance.GetDevice();
    vk::Result result = device.acquireNextImageKHR(
        swapchain,
#ifdef __ANDROID__
        250'000'000ULL,
#else
        std::numeric_limits<u64>::max(),
#endif
        image_acquired[frame_index], VK_NULL_HANDLE, &image_index);

    switch (result) {
    case vk::Result::eSuccess:
        consecutive_acquire_timeouts = 0;
        break;
    case vk::Result::eErrorDeviceLost:
        terminal = true;
        terminal_result = result;
        needs_recreation = false;
        return false;
#ifdef __ANDROID__
    case vk::Result::eTimeout:
        needs_recreation = false;
        if (++consecutive_acquire_timeouts >= 8) {
            terminal = true;
            terminal_result = result;
            LOG_ERROR(Render_Vulkan,
                      "[EXECUTOR_VK_TERMINAL] where=swapchain_acquire result={} terminal=1",
                      vk::to_string(result));
        }
        return false;
#endif
    case vk::Result::eSuboptimalKHR:
    case vk::Result::eErrorSurfaceLostKHR:
    case vk::Result::eErrorOutOfDateKHR:
    case vk::Result::eErrorUnknown:
        needs_recreation = true;
        break;
    default:
        LOG_WARNING(Render_Vulkan,
                    "Swapchain acquire returned unexpected result {} — requesting recreate",
                    vk::to_string(result));
        needs_recreation = true;
        break;
    }

    return !needs_recreation;
}

bool Swapchain::Present() {

    if (terminal) {
        return false;
    }

    const vk::PresentInfoKHR present_info = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &present_ready[image_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    };

    auto result = instance.GetPresentQueue().presentKHR(present_info);
    if (result == vk::Result::eErrorDeviceLost) {
        terminal = true;
        terminal_result = result;
    }
#ifdef __ANDROID__
    const bool expected_identity_suboptimal =
        result == vk::Result::eSuboptimalKHR &&
        transform == vk::SurfaceTransformFlagBitsKHR::eIdentity &&
        surface_current_transform != vk::SurfaceTransformFlagBitsKHR::eIdentity;
#else
    const bool expected_identity_suboptimal = false;
#endif
    const bool present_requires_attention =
        result != vk::Result::eSuccess && !expected_identity_suboptimal;
#ifdef __ANDROID__
    const bool trace_present = ExecutorTraceVulkanPresent();
#else
    const bool trace_present = true;
#endif
    if (present_requires_attention || trace_present) {
        LOG_INFO(Render_Vulkan, "EXECUTOR_VULKAN_PRESENT result={} suboptimal={} outOfDate={}",
                 vk::to_string(result), result == vk::Result::eSuboptimalKHR ? "YES" : "NO",
                 result == vk::Result::eErrorOutOfDateKHR ? "YES" : "NO");
    }
#ifdef __ANDROID__
    if (present_requires_attention || trace_present) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_VULKAN_PRESENT] result=%s suboptimal=%d outOfDate=%d",
                            vk::to_string(result).c_str(),
                            result == vk::Result::eSuboptimalKHR ? 1 : 0,
                            result == vk::Result::eErrorOutOfDateKHR ? 1 : 0);
    }
#endif
    if (result == vk::Result::eErrorOutOfDateKHR ||
        (result == vk::Result::eSuboptimalKHR && !expected_identity_suboptimal)) {
        needs_recreation = true;
    } else if (!expected_identity_suboptimal) {
        ASSERT_MSG(result == vk::Result::eSuccess, "Swapchain presentation failed: {}",
                   vk::to_string(result));
    }

    frame_index = (frame_index + 1) % image_count;

    return !needs_recreation;
}

void Swapchain::FindPresentFormat() {
    const auto [formats_result, formats] =
        instance.GetPhysicalDevice().getSurfaceFormatsKHR(surface);
    ASSERT_MSG(formats_result == vk::Result::eSuccess, "Failed to query surface formats: {}",
               vk::to_string(formats_result));

    supports_hdr =
        std::find_if(formats.begin(), formats.end(), [](const vk::SurfaceFormatKHR& format) {
            return format == SURFACE_FORMAT_HDR;
        }) != formats.end();
    supports_hdr &= Config::allowHDR();

    if (formats[0].format == vk::Format::eUndefined) {
        surface_format.format = vk::Format::eR8G8B8A8Unorm;
        surface_format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
        return;
    }

    for (const vk::SurfaceFormatKHR& sformat : formats) {
        vk::Format format = sformat.format;
        if (format != vk::Format::eR8G8B8A8Unorm && format != vk::Format::eB8G8R8A8Unorm) {
            continue;
        }

        surface_format.format = format;
        surface_format.colorSpace = sformat.colorSpace;
        return;
    }

    UNREACHABLE_MSG("Unable to find required swapchain format!");
}

void Swapchain::FindPresentMode() {
    const auto [modes_result, modes] =
        instance.GetPhysicalDevice().getSurfacePresentModesKHR(surface);
    if (modes_result != vk::Result::eSuccess) {
        LOG_ERROR(Render, "Failed to query available present modes, falling back to Fifo as "
                          "guaranteed supported option.");
        present_mode = vk::PresentModeKHR::eFifo;
        return;
    }

    const auto requested_mode = Config::getPresentMode();
    if (requested_mode == "Mailbox") {
        present_mode = vk::PresentModeKHR::eMailbox;
    } else if (requested_mode == "Fifo") {
        present_mode = vk::PresentModeKHR::eFifo;
    } else if (requested_mode == "Immediate") {
        present_mode = vk::PresentModeKHR::eImmediate;
    } else {
        LOG_ERROR(Render_Vulkan, "Unknown present mode {}, defaulting to Mailbox.",
                  Config::getPresentMode());
        present_mode = vk::PresentModeKHR::eMailbox;
    }

    if (std::ranges::find(modes, present_mode) == modes.cend()) {
        constexpr auto fallback = vk::PresentModeKHR::eFifo;
        LOG_WARNING(Render, "Requested present mode {} is not supported, falling back to {}.",
                    vk::to_string(present_mode), vk::to_string(fallback));
        present_mode = fallback;
    }
}

bool Swapchain::SetSurfaceProperties() {
    const auto [capabilities_result, capabilities] =
        instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(surface);
    if (capabilities_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan,
                    "[EXECUTOR_VULKAN_WSI] surface capabilities query failed: {} — skipping "
                    "swapchain recreate, will retry",
                    vk::to_string(capabilities_result));
        return false;
    }

    extent = capabilities.currentExtent;
    if (capabilities.currentExtent.width == std::numeric_limits<u32>::max()) {
        extent.width = std::max(capabilities.minImageExtent.width,
                                std::min(capabilities.maxImageExtent.width, width));
        extent.height = std::max(capabilities.minImageExtent.height,
                                 std::min(capabilities.maxImageExtent.height, height));
    }

    image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    surface_current_transform = capabilities.currentTransform;

    constexpr auto identity = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    transform = capabilities.supportedTransforms & identity ? identity
                                                             : capabilities.currentTransform;
    LOG_INFO(Render_Vulkan,
             "EXECUTOR_VULKAN_WSI phase=surface_properties surfaceCurrentTransform={} "
             "swapchainPreTransform={} identitySupported={} aligned={}",
             vk::to_string(surface_current_transform), vk::to_string(transform),
             static_cast<bool>(capabilities.supportedTransforms & identity) ? "YES" : "NO",
             surface_current_transform == transform ? "YES" : "NO");

    composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    if (!(capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque)) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
    }
    return true;
}

void Swapchain::Destroy() {
    vk::Device device = instance.GetDevice();
    if (!terminal) {
        const auto wait_result = device.waitIdle();
        if (wait_result != vk::Result::eSuccess) {
            LOG_WARNING(Render_Vulkan, "Failed to wait for device to become idle: {}",
                        vk::to_string(wait_result));
        }
    }

    for (auto& image_view : images_view) {
        if (image_view) {
            device.destroyImageView(image_view);
        }
    }
    images_view.clear();
    images.clear();

    if (swapchain) {
        device.destroySwapchainKHR(swapchain);
        swapchain = nullptr;
    }

    for (const auto& sem : image_acquired) {
        device.destroySemaphore(sem);
    }
    for (const auto& sem : present_ready) {
        device.destroySemaphore(sem);
    }

    image_acquired.clear();
    present_ready.clear();
    image_count = 0;
    image_index = 0;
    frame_index = 0;
}

void Swapchain::RefreshSemaphores() {
    const vk::Device device = instance.GetDevice();
    image_acquired.resize(image_count);
    present_ready.resize(image_count);

    for (vk::Semaphore& semaphore : image_acquired) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create image acquired semaphore: {}",
                   vk::to_string(semaphore_result));
        semaphore = sem;
    }
    for (vk::Semaphore& semaphore : present_ready) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create present ready semaphore: {}", vk::to_string(semaphore_result));
        semaphore = sem;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, image_acquired[i], "Swapchain Semaphore: image_acquired {}", i);
        SetObjectName(device, present_ready[i], "Swapchain Semaphore: present_ready {}", i);
    }
}

void Swapchain::SetupImages() {
    vk::Device device = instance.GetDevice();
    auto [images_result, imgs] = device.getSwapchainImagesKHR(swapchain);
    ASSERT_MSG(images_result == vk::Result::eSuccess, "Failed to create swapchain images: {}",
               vk::to_string(images_result));
    images = std::move(imgs);
    image_count = static_cast<u32>(images.size());
    images_view.resize(image_count);
    for (u32 i = 0; i < image_count; ++i) {
        if (images_view[i]) {
            device.destroyImageView(images_view[i]);
        }
        auto [im_view_result, im_view] = device.createImageView(vk::ImageViewCreateInfo{
            .image = images[i],
            .viewType = vk::ImageViewType::e2D,
            .format = needs_hdr ? SURFACE_FORMAT_HDR.format : surface_format.format,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .levelCount = 1,
                    .layerCount = 1,
                },
        });
        ASSERT_MSG(im_view_result == vk::Result::eSuccess, "Failed to create image view: {}",
                   vk::to_string(im_view_result));
        images_view[i] = im_view;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, images[i], "Swapchain Image {}", i);
        SetObjectName(device, images_view[i], "Swapchain ImageView {}", i);
    }
}

}
