// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
#if defined(__ANDROID__)
#include <android/log.h>
#endif

#include "common/config.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/singleton.h"
#include "core/debug_state.h"
#include "core/devtools/layer.h"
#include "core/libraries/system/systemservice.h"
#include "imgui/renderer/imgui_core.h"
#include "imgui/renderer/imgui_impl_vulkan.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/image.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <imgui.h>
#include <mutex>
#include <vk_mem_alloc.h>

namespace Vulkan {

namespace {
std::atomic<u64> g_executor_live_present_count{0};
}

u64 ExecutorGetLivePresentCount() noexcept {
    return g_executor_live_present_count.load(std::memory_order_relaxed);
}

#if defined(__ANDROID__)
static void ExecutorLiveDirectPresentStats(bool present_ok, u32 src_width, u32 src_height,
                                           u32 dst_width, u32 dst_height, u32 target) {
    using Clock = std::chrono::steady_clock;
    static std::mutex stats_mutex;
    static auto start = Clock::now();
    static auto last = start;
    static u64 total = 0;
    static u64 ok = 0;
    static u64 fail = 0;
    static u64 last_total = 0;
    static u64 last_ok = 0;
    static u64 last_fail = 0;

    const auto now = Clock::now();
    std::scoped_lock lock{stats_mutex};
    ++total;
    if (present_ok) {
        ++ok;
        g_executor_live_present_count.fetch_add(1, std::memory_order_relaxed);
    } else {
        ++fail;
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
    if (elapsed_ms.count() < 1000) {
        return;
    }

    const double window_sec = static_cast<double>(elapsed_ms.count()) / 1000.0;
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_FPS] source=direct_present elapsedMs=%lld fps=%.2f okRate=%.2f "
        "failRate=%.2f presents=%llu ok=%llu fail=%llu src=%ux%u dst=%ux%u target=%u",
        static_cast<long long>(total_ms),
        static_cast<double>(total - last_total) / window_sec,
        static_cast<double>(ok - last_ok) / window_sec,
        static_cast<double>(fail - last_fail) / window_sec,
        static_cast<unsigned long long>(total), static_cast<unsigned long long>(ok),
        static_cast<unsigned long long>(fail), src_width, src_height, dst_width, dst_height,
        target);

    last = now;
    last_total = total;
    last_ok = ok;
    last_fail = fail;
}
#endif

bool CanBlitToSwapchain(const vk::PhysicalDevice physical_device, vk::Format format) {
    const vk::FormatProperties props{physical_device.getFormatProperties(format)};
    return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);
}

[[nodiscard]] vk::ImageSubresourceLayers MakeImageSubresourceLayers() {
    return vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlit(s32 frame_width, s32 frame_height, s32 dst_width,
                                          s32 dst_height, s32 offset_x, s32 offset_y) {
    return vk::ImageBlit{
        .srcSubresource = MakeImageSubresourceLayers(),
        .srcOffsets =
            std::array{
                vk::Offset3D{
                    .x = 0,
                    .y = 0,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = frame_width,
                    .y = frame_height,
                    .z = 1,
                },
            },
        .dstSubresource = MakeImageSubresourceLayers(),
        .dstOffsets =
            std::array{
                vk::Offset3D{
                    .x = offset_x,
                    .y = offset_y,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = offset_x + dst_width,
                    .y = offset_y + dst_height,
                    .z = 1,
                },
            },
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlitStretch(s32 frame_width, s32 frame_height,
                                                 s32 swapchain_width, s32 swapchain_height) {
    return MakeImageBlit(frame_width, frame_height, swapchain_width, swapchain_height, 0, 0);
}

static vk::Rect2D FitImage(s32 frame_width, s32 frame_height, s32 swapchain_width,
                           s32 swapchain_height) {
    float frame_aspect = static_cast<float>(frame_width) / frame_height;
    float swapchain_aspect = static_cast<float>(swapchain_width) / swapchain_height;

    u32 dst_width = swapchain_width;
    u32 dst_height = swapchain_height;

    if (frame_aspect > swapchain_aspect) {
        dst_height = static_cast<s32>(swapchain_width / frame_aspect);
    } else {
        dst_width = static_cast<s32>(swapchain_height * frame_aspect);
    }

    const s32 offset_x = (swapchain_width - dst_width) / 2;
    const s32 offset_y = (swapchain_height - dst_height) / 2;

    return vk::Rect2D{{offset_x, offset_y}, {dst_width, dst_height}};
}

[[nodiscard]] vk::ImageBlit MakeImageBlitFit(s32 frame_width, s32 frame_height, s32 swapchain_width,
                                             s32 swapchain_height) {
    const auto& dst_rect = FitImage(frame_width, frame_height, swapchain_width, swapchain_height);

    return MakeImageBlit(frame_width, frame_height, dst_rect.extent.width, dst_rect.extent.height,
                         dst_rect.offset.x, dst_rect.offset.y);
}

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_},
      instance{window, Config::getGpuId(), Config::vkValidationEnabled(),
               Config::getVkCrashDiagnosticEnabled()},
      draw_scheduler{instance}, present_scheduler{instance}, flip_scheduler{instance},
      swapchain{instance, window},
      rasterizer{std::make_unique<Rasterizer>(instance, draw_scheduler, liverpool)},
      texture_cache{rasterizer->GetTextureCache()} {
    const u32 num_images = swapchain.GetImageCount();
    const vk::Device device = instance.GetDevice();

    present_frames.resize(num_images);
    for (u32 i = 0; i < num_images; i++) {
        Frame& frame = present_frames[i];
        frame.id = i;
        auto fence = Check<"create present done fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        frame.present_done = fence;
        free_queue.push(&frame);
    }

    fsr_settings.enable = Config::getFsrEnabled();
    fsr_settings.use_rcas = Config::getRcasEnabled();
    fsr_settings.rcas_attenuation = static_cast<float>(Config::getRcasAttenuation() / 1000.f);

    fsr_pass.Create(device, instance.GetAllocator(), num_images);
    pp_pass.Create(device, swapchain.GetSurfaceFormat().format);

    ImGui::Layer::AddLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
}

Presenter::~Presenter() {
    ImGui::Layer::RemoveLayer(Common::Singleton<Core::Devtools::Layer>::Instance());
    draw_scheduler.Finish(Scheduler::ExecutorFinishReason::PresenterShutdown);
    const vk::Device device = instance.GetDevice();
    for (auto& frame : present_frames) {
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        device.destroyImageView(frame.image_view);
        device.destroyFence(frame.present_done);
    }
    ImGui::Core::Shutdown(device);
}

bool Presenter::IsDeviceLost() const noexcept {
    return device_lost.load(std::memory_order_acquire) || draw_scheduler.IsDeviceLost() ||
           present_scheduler.IsDeviceLost() || flip_scheduler.IsDeviceLost();
}

void Presenter::MarkDeviceLost(vk::Result result, const char* where) noexcept {
    if (result != vk::Result::eErrorDeviceLost) {
        return;
    }

    MarkTerminal(result, where);
}

void Presenter::MarkTerminal(vk::Result result, const char* where) noexcept {

    bool expected = false;
    if (!device_lost.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
        return;
    }

    LOG_ERROR(Render_Vulkan, "[EXECUTOR_VK_TERMINAL] where={} result={} terminal=1", where,
              vk::to_string(result));
    draw_scheduler.MarkTerminal(result, "presenter_draw");
    present_scheduler.MarkTerminal(result, "presenter_present");
    flip_scheduler.MarkTerminal(result, "presenter_flip");
    free_cv.notify_all();
    frame_cv.notify_all();
}

bool Presenter::WaitForFence(vk::Fence fence, const char* where) noexcept {
#ifdef __ANDROID__
    constexpr u64 WaitSlice = 250'000'000ULL;
    constexpr u32 MaxWaits = 8;
    for (u32 attempt = 0; attempt < MaxWaits; ++attempt) {
        const vk::Result result = instance.GetDevice().waitForFences(fence, false, WaitSlice);
        if (result == vk::Result::eSuccess) {
            return true;
        }
        if (result == vk::Result::eTimeout && attempt + 1 != MaxWaits) {
            continue;
        }
        MarkTerminal(result, where);
        return false;
    }
    return false;
#else
    const vk::Result result = instance.GetDevice().waitForFences(
        fence, false, std::numeric_limits<u64>::max());
    if (result == vk::Result::eSuccess) {
        return true;
    }
    if (result == vk::Result::eErrorDeviceLost) {
        MarkDeviceLost(result, where);
    } else {
        LOG_ERROR(Render_Vulkan, "Failed waiting for present fence: {}", vk::to_string(result));
    }
    return false;
#endif
}

bool Presenter::AcquireSwapchainImage(const char* where) {
    if (swapchain.AcquireNextImage()) {
        return true;
    }
    if (swapchain.IsTerminal()) {
        MarkTerminal(swapchain.TerminalResult(), where);
        return false;
    }
    if (!swapchain.NeedsRecreation()) {
        return false;
    }
    swapchain.Recreate(window.GetWidth(), window.GetHeight());
    if (swapchain.AcquireNextImage()) {
        return true;
    }
    if (swapchain.IsTerminal()) {
        MarkTerminal(swapchain.TerminalResult(), where);
    }
    return false;
}

bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

void Presenter::RecreateFrame(Frame* frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    if (frame->imgui_texture) {
        ImGui::Vulkan::RemoveTexture(frame->imgui_texture);
    }
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &frame->allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}",
                     vk::to_string(vk::Result{result}));
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};
    SetObjectName(device, frame->image, "Frame image #{}", frame->id);

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    auto view = Check<"create frame image view">(device.createImageView(view_info));
    frame->image_view = view;
    frame->width = width;
    frame->height = height;

    frame->imgui_texture = ImGui::Vulkan::AddTexture(view, vk::ImageLayout::eShaderReadOnlyOptimal);
    frame->is_hdr = swapchain.GetHDR();
}

Frame* Presenter::PrepareLastFrame() {
    if (IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "prepare_last_frame");
        return nullptr;
    }
    if (last_submit_frame == nullptr) {
        return nullptr;
    }

    Frame* frame = last_submit_frame;

    if (!WaitForFence(frame->present_done, "prepare_last_frame_fence")) {
        return nullptr;
    }

    auto& scheduler = flip_scheduler;
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier =
        vk::ImageMemoryBarrier2{.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
                                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                                .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                .newLayout = vk::ImageLayout::eGeneral,
                                .image = frame->image,
                                .subresourceRange{frame_subresources}};

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    frame->ready_semaphore = scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    if (scheduler.IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "prepare_last_frame_submit");
        return nullptr;
    }
    return frame;
}

static vk::Format GetFrameViewFormat(const Libraries::VideoOut::PixelFormat format) {
    switch (format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    if (IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "prepare_frame");
        return nullptr;
    }

    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    auto image_id = texture_cache.FindImage(desc);
#ifdef __ANDROID__
    const auto registered_image_id = image_id;
    bool using_live_drawn_target = false;
    const auto live_target = texture_cache.ExecutorTakeLiveFullResolutionTarget(
        attribute.attrib.width, attribute.attrib.height, registered_image_id);
    if (live_target) {
        image_id = live_target;
        using_live_drawn_target = true;
    }
#endif
    auto& source_image = texture_cache.GetImage(image_id);
    const bool source_view_descriptor_mismatch =
        source_image.info.pixel_format != desc.info.pixel_format ||
        source_image.info.type != desc.info.type ||
        source_image.info.resources != desc.info.resources;
    const bool source_is_distinct_rt_alias =
        source_image.usage.render_target && !source_image.usage.vo_surface;
    bool use_canonical_source_view =
        source_view_descriptor_mismatch || source_is_distinct_rt_alias;
#ifdef __ANDROID__
    use_canonical_source_view |= using_live_drawn_target;
#endif
#ifdef __ANDROID__
    const auto& registered_image = texture_cache.GetImage(registered_image_id);
    static std::atomic<u32> frame_source_log_count{0};
    const u32 frame_source_log =
        frame_source_log_count.fetch_add(1, std::memory_order_relaxed);
    if (frame_source_log < 8) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_PREPARE_FRAME_SOURCE] ordinal=%u bridge=%u scanoutImage=%u "
            "scanoutFmt=%u scanoutType=%llu image=%u scanout=%ux%u source=%ux%u "
            "sourceAddr=0x%llx sourceBytes=0x%llx sourceFmt=%u sourceType=%llu "
            "sourceTile=%u sourceBits=%u sourceUsageVo=%u sourceUsageRt=%u "
            "canonicalView=%u descriptorMismatch=%u rtAlias=%u",
            frame_source_log + 1, using_live_drawn_target ? 1u : 0u,
            registered_image_id.index, static_cast<u32>(registered_image.info.pixel_format),
            static_cast<unsigned long long>(registered_image.info.type), image_id.index,
            attribute.attrib.width, attribute.attrib.height, source_image.info.size.width,
            source_image.info.size.height,
            static_cast<unsigned long long>(source_image.info.guest_address),
            static_cast<unsigned long long>(source_image.info.guest_size),
            static_cast<u32>(source_image.info.pixel_format),
            static_cast<unsigned long long>(source_image.info.type),
            static_cast<u32>(source_image.info.tile_mode), source_image.info.num_bits,
            source_image.usage.vo_surface, source_image.usage.render_target,
            use_canonical_source_view ? 1u : 0u,
            source_view_descriptor_mismatch ? 1u : 0u,
            source_is_distinct_rt_alias ? 1u : 0u);
    }
#endif
    const u32 flags_before_refresh = static_cast<u32>(source_image.flags);
#ifdef __ANDROID__
    if (!using_live_drawn_target) {
        texture_cache.UpdateImage(image_id);
    }
#else
    texture_cache.UpdateImage(image_id);
#endif

#if defined(__ANDROID__)
    static const bool trace_videoout_source =
        std::getenv("EXECUTOR_TRACE_VIDEOOUT_SOURCE") != nullptr ||
        std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr;
    static std::atomic<u64> videoout_sequence{0};
    static std::atomic<bool> live_target_source_proved{false};
    const u64 sequence = videoout_sequence.fetch_add(1, std::memory_order_relaxed);
    const bool prove_first_live_target =
        using_live_drawn_target &&
        !live_target_source_proved.exchange(true, std::memory_order_relaxed);
    if (trace_videoout_source) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_VIDEOOUT_FRAME_SOURCE] seq=%llu image=%u registered=%u "
            "addr=0x%llx registeredAddr=0x%llx flagsBefore=0x%x flagsAfter=0x%x "
            "usageVo=%u usageRt=%u canonicalView=%u rtAlias=%u",
            static_cast<unsigned long long>(sequence), image_id.index,
            registered_image_id.index,
            static_cast<unsigned long long>(source_image.info.guest_address),
            static_cast<unsigned long long>(registered_image.info.guest_address),
            flags_before_refresh, static_cast<u32>(source_image.flags),
            source_image.usage.vo_surface, source_image.usage.render_target,
            use_canonical_source_view ? 1u : 0u,
            source_is_distinct_rt_alias ? 1u : 0u);
    }
    if (prove_first_live_target ||
        (trace_videoout_source && (sequence < 12 || (sequence % 60) < 3))) {
        texture_cache.ExecutorTraceVideoOutSource(image_id, sequence, flags_before_refresh);
    }
#endif

    Frame* frame = GetRenderFrame();
    if (frame == nullptr) {
        return nullptr;
    }

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange{frame_subresources},
    };

    draw_scheduler.EndRendering();
    const auto cmdbuf = draw_scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    VideoCore::ImageViewInfo view_info{};
    if (use_canonical_source_view) {
        view_info.type = source_image.info.type;
        const vk::Format scanout_format = GetFrameViewFormat(attribute.attrib.pixel_format);
        view_info.format = VideoCore::IsVulkanFormatCompatible(source_image.info.pixel_format,
                                                                scanout_format)
            ? scanout_format
            : source_image.info.pixel_format;
        view_info.range.extent = source_image.info.resources;
    } else {
        view_info.format = GetFrameViewFormat(attribute.attrib.pixel_format);
    }
    view_info.mapping.a = vk::ComponentSwizzle::eOne;

    auto& image = texture_cache.GetImage(image_id);
    auto image_view = *image.FindView(view_info).image_view;
    image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {});

    const vk::Extent2D image_size = {image.info.size.width, image.info.size.height};
    expected_ratio = static_cast<float>(image_size.width) / static_cast<float>(image_size.height);

    image_view = fsr_pass.Render(cmdbuf, image_view, image_size, {frame->width, frame->height},
                                 fsr_settings, frame->is_hdr);
    pp_pass.Render(cmdbuf, image_view, image_size, *frame, pp_settings);

    DebugState.game_resolution = {image_size.width, image_size.height};
    DebugState.output_resolution = {frame->width, frame->height};

    frame->ready_semaphore = draw_scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    SubmitInfo info{};
    draw_scheduler.Flush(info);
    if (draw_scheduler.IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "prepare_frame_submit");
        return nullptr;
    }
    return frame;
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    Frame* frame = GetRenderFrame();
    if (frame == nullptr) {
        return nullptr;
    }

    auto& scheduler = present_thread ? present_scheduler : draw_scheduler;
    scheduler.EndRendering();

    const auto cmdbuf = scheduler.CommandBuffer();

    constexpr vk::ImageSubresourceRange simple_subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const auto post_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const vk::RenderingAttachmentInfo attachment = {
        .imageView = frame->image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .extent = {frame->width, frame->height},
            },
        .layerCount = 1,
        .colorAttachmentCount = 1u,
        .pColorAttachments = &attachment,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    cmdbuf.beginRendering(rendering_info);
    cmdbuf.endRendering();

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &post_barrier,
    });

    frame->ready_semaphore = scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    if (scheduler.IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "prepare_blank_frame_submit");
        return nullptr;
    }
    return frame;
}

bool Presenter::ExecutorPresentReplayRT() {
#ifdef __ANDROID__
    if (IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "replay_present");
        return false;
    }
    static u32 replay_present_log_counter = 0;
    const bool live_present = std::getenv("EXECUTOR_LIVE_PRESENT_RT") != nullptr;
    const bool log_this_present =
        replay_present_log_counter == 0 || (replay_present_log_counter % 60) == 0;
    ++replay_present_log_counter;
    if (log_this_present && !live_present) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_PRESENT] replay candidates=%zu target=%u",
                            texture_cache.executor_replay_color_candidates.size(),
                            texture_cache.executor_replay_color_target.index);
    }
    VideoCore::ImageId image_id{};
    if (live_present) {
        const u32 videoout_width = executor_videoout_width.load(std::memory_order_acquire);
        const u32 videoout_height = executor_videoout_height.load(std::memory_order_acquire);
        image_id =
            texture_cache.ExecutorTakeLiveFullResolutionTarget(videoout_width, videoout_height);
        if (!image_id) {
            return false;
        }
    } else {
        if (!texture_cache.executor_replay_color_candidates.empty()) {
            texture_cache.ExecutorReplayReadbackColorTarget();
        }
        image_id = texture_cache.executor_replay_color_target;
    }
    if (!image_id) {
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_PRESENT] no replay color target");
        return false;
    }
    auto& src = texture_cache.GetImage(image_id);
    if (live_present) {
        std::scoped_lock present_lock{executor_present_mutex};
        if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
            swapchain.Recreate(window.GetWidth(), window.GetHeight());
        }
        if (!AcquireSwapchainImage("direct_present_acquire")) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_PRESENT] live direct skip=acquire_failed");
            return false;
        }

        auto& scheduler = draw_scheduler;
        scheduler.EndRendering();
        const auto cmdbuf = scheduler.CommandBuffer();
        const vk::Image swapchain_image = swapchain.Image();
        constexpr vk::ImageSubresourceRange subresource{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        };
        const vk::ImageMemoryBarrier pre_barrier{
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange = subresource,
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

        src.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
                    cmdbuf);
        const vk::Extent2D dst_extent = swapchain.GetExtent();
        const auto blit = MakeImageBlitStretch(static_cast<s32>(src.info.size.width),
                                               static_cast<s32>(src.info.size.height),
                                               static_cast<s32>(dst_extent.width),
                                               static_cast<s32>(dst_extent.height));
        cmdbuf.blitImage(src.GetImage(), vk::ImageLayout::eTransferSrcOptimal, swapchain_image,
                         vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eLinear);

        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eNone,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange = subresource,
        };
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eBottomOfPipe,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);

        SubmitInfo info{};
        info.AddWait(swapchain.GetImageAcquiredSemaphore(), 1,
                     vk::PipelineStageFlagBits::eTransfer);
        info.AddSignal(swapchain.GetPresentReadySemaphore());
        scheduler.Flush(info);
        if (scheduler.IsDeviceLost()) {
            MarkDeviceLost(vk::Result::eErrorDeviceLost, "direct_present_submit");
            return false;
        }

        std::scoped_lock submit_lock{Scheduler::submit_mutex};
        const bool present_ok = swapchain.Present();
        if (swapchain.IsTerminal()) {
            MarkTerminal(swapchain.TerminalResult(), "direct_queue_present");
        } else if (!present_ok) {
            swapchain.Recreate(window.GetWidth(), window.GetHeight());
        } else {
            executor_live_direct_present_active.store(true, std::memory_order_release);
        }
        ExecutorLiveDirectPresentStats(present_ok, src.info.size.width, src.info.size.height,
                                       dst_extent.width, dst_extent.height, image_id.index);
        if (log_this_present) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_PRESENT_DIRECT] ok=%d src=%ux%u dst=%ux%u target=%u",
                                present_ok ? 1 : 0, src.info.size.width, src.info.size.height,
                                dst_extent.width, dst_extent.height, image_id.index);
        }
        return present_ok;
    }
    Frame* frame = GetRenderFrame();
    if (frame == nullptr) {
        return false;
    }

    draw_scheduler.EndRendering();
    const auto cmdbuf = draw_scheduler.CommandBuffer();

    constexpr vk::ImageSubresourceRange frame_subresources{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto pre_barrier = vk::ImageMemoryBarrier{
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame->image,
        .subresourceRange = frame_subresources,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                           vk::PipelineStageFlagBits::eColorAttachmentOutput,
                           vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

    VideoCore::ImageViewInfo view_info{};
    view_info.type = src.info.type;
    view_info.format = src.info.pixel_format;
    view_info.range.extent = src.info.resources;
    view_info.mapping.a = vk::ComponentSwizzle::eOne;

    const auto& src_view = src.FindView(view_info);
    if (!src_view.image_view) {
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_PRESENT] no replay image view target=%u extent=%ux%u",
                            image_id.index, src.info.size.width, src.info.size.height);
        return false;
    }
    const auto image_view = *src_view.image_view;
    src.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {});

    const vk::Extent2D image_size = {src.info.size.width, src.info.size.height};
    expected_ratio = static_cast<float>(image_size.width) / static_cast<float>(image_size.height);
    pp_pass.Render(cmdbuf, image_view, image_size, *frame, pp_settings);

    frame->ready_semaphore = draw_scheduler.GetMasterSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    SubmitInfo info{};
    draw_scheduler.Flush(info);
    if (draw_scheduler.IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "replay_present_submit");
        return false;
    }
    if (log_this_present) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_PRESENT] presenting replay RT opaque-alpha %ux%u -> frame %ux%u",
                            src.info.size.width, src.info.size.height, frame->width, frame->height);
    }
    Present(frame);
    return true;
#else
    return false;
#endif
}

void Presenter::ExecutorRecyclePreparedFrame(Frame* frame) {
    if (!frame) {
        return;
    }
    draw_scheduler.GetMasterSemaphore()->Wait(frame->ready_tick);
    if (draw_scheduler.IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "recycle_frame_wait");
        return;
    }
    std::scoped_lock fl{free_mutex};
    free_queue.push(frame);
    free_cv.notify_one();
}

void Presenter::Present(Frame* frame, bool is_reusing_frame) {
    if (frame == nullptr) {
        return;
    }
    if (IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "present");
        return;
    }

    std::unique_lock present_lock{executor_present_mutex};
    if (!is_reusing_frame && std::getenv("EXECUTOR_LIVE_PRESENT_RT") != nullptr &&
        ExecutorLiveDirectPresentActive()) {
        present_lock.unlock();
        ExecutorRecyclePreparedFrame(frame);
        return;
    }

    const auto free_frame = [&] {
        if (!is_reusing_frame) {
            last_submit_frame = frame;
            std::scoped_lock fl{free_mutex};
            free_queue.push(frame);
            free_cv.notify_one();
        }
    };

    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }

    if (!AcquireSwapchainImage("videoout_acquire")) {
        LOG_WARNING(Render_Vulkan, "Skipping frame!");
        free_frame();
        return;
    }

    const auto reset_result = instance.GetDevice().resetFences(frame->present_done);
    if (reset_result != vk::Result::eSuccess) [[unlikely]] {
        if (reset_result == vk::Result::eErrorDeviceLost) {
            MarkDeviceLost(reset_result, "present_reset_fence");
        } else {
            LOG_ERROR(Render_Vulkan, "Unexpected error resetting present done fence: {}",
                      vk::to_string(reset_result));
        }
        free_frame();
        return;
    }

    ImGuiID dockId = ImGui::Core::NewFrame(is_reusing_frame);

    const vk::Image swapchain_image = swapchain.Image();
    const vk::ImageView swapchain_image_view = swapchain.ImageView();

    auto& scheduler = present_scheduler;
    const auto cmdbuf = scheduler.CommandBuffer();

    if (Config::getVkHostMarkersEnabled()) {
        cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
            .pLabelName = "Present",
        });
    }

    {
        auto* profiler_ctx = instance.GetProfilerContext();
        TracyVkNamedZoneC(profiler_ctx, renderer_gpu_zone, cmdbuf, "Host frame",
                          MarkersPalette::GpuMarkerColor, profiler_ctx != nullptr);

        const vk::Extent2D extent = swapchain.GetExtent();
        const std::array pre_barriers{
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = swapchain_image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = frame->image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
        };

        const vk::ImageMemoryBarrier post_barrier{
            .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
            .dstAccessMask = vk::AccessFlagBits::eNone,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchain_image,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits::eColorAttachmentOutput |
                                   vk::PipelineStageFlagBits::eFragmentShader,
                               vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);

        {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0f});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
            ImGui::SetNextWindowDockID(dockId, ImGuiCond_Once);
            if (ImGui::Begin("Display##game_display", nullptr, ImGuiWindowFlags_NoNav)) {
                auto game_texture = frame->imgui_texture;
                auto game_width = frame->width;
                auto game_height = frame->height;

                if (Libraries::SystemService::IsSplashVisible()) {
                    if (!splash_img.has_value()) {
                        splash_img.emplace();
                        auto splash_path = Common::ElfInfo::Instance().GetSplashPath();
                        if (!splash_path.empty()) {
                            splash_img = ImGui::RefCountedTexture::DecodePngFile(splash_path);
                        }
                    }
                    if (auto& splash_image = this->splash_img.value()) {
                        auto [im_id, width, height] = splash_image.GetTexture();
                        game_texture = im_id;
                        game_width = width;
                        game_height = height;
                    }
                }

                ImVec2 contentArea = ImGui::GetContentRegionAvail();
                SetExpectedGameSize((s32)contentArea.x, (s32)contentArea.y);

                const auto imgRect =
                    FitImage(game_width, game_height, (s32)contentArea.x, (s32)contentArea.y);
                ImVec2 offset{
                    static_cast<float>(imgRect.offset.x),
                    static_cast<float>(imgRect.offset.y),
                };
                ImVec2 size{
                    static_cast<float>(imgRect.extent.width),
                    static_cast<float>(imgRect.extent.height),
                };

                ImGui::SetCursorPos(ImGui::GetCursorStartPos() + offset);
                ImGui::Image(game_texture, size);

                if (Config::nullGpu()) {
                    Core::Devtools::Layer::DrawNullGpuNotice();
                }
            }
            ImGui::End();
            ImGui::PopStyleVar(3);
            ImGui::PopStyleColor();
        }
        ImGui::Core::Render(cmdbuf, swapchain_image_view, swapchain.GetExtent());

        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);

        if (profiler_ctx) {
            TracyVkCollect(profiler_ctx, cmdbuf);
        }
    }

    if (Config::getVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }

    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    info.AddWait(frame->ready_semaphore, frame->ready_tick);
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    info.AddSignal(frame->present_done);
    scheduler.Flush(info);
    if (scheduler.IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "present_submit");
        free_frame();
        return;
    }

    std::scoped_lock submit_lock{Scheduler::submit_mutex};
    const bool present_ok = swapchain.Present();
    if (!present_ok) {
        if (swapchain.IsTerminal()) {
            MarkTerminal(swapchain.TerminalResult(), "videoout_queue_present");
        } else {
            swapchain.Recreate(window.GetWidth(), window.GetHeight());
        }
    } else {
        g_executor_live_present_count.fetch_add(1, std::memory_order_relaxed);
    }

    free_frame();
    if (!is_reusing_frame) {
        DebugState.IncFlipFrameNum();
    }
}

Frame* Presenter::GetRenderFrame() {
    if (IsDeviceLost()) {
        MarkDeviceLost(vk::Result::eErrorDeviceLost, "get_render_frame");
        return nullptr;
    }

    Frame* frame;
    {
        std::unique_lock lock{free_mutex};
#ifdef __ANDROID__
        while (!IsDeviceLost() && free_queue.empty()) {
            free_cv.wait_for(lock, std::chrono::milliseconds{250});
        }
#else
        free_cv.wait(lock, [this] { return IsDeviceLost() || !free_queue.empty(); });
#endif
        if (IsDeviceLost()) {
            lock.unlock();
            MarkDeviceLost(vk::Result::eErrorDeviceLost, "get_render_frame_queue");
            return nullptr;
        }
        LOG_DEBUG(Render_Vulkan, "Got render frame, remaining {}", free_queue.size() - 1);

        frame = free_queue.front();
        free_queue.pop();
    }

    if (!WaitForFence(frame->present_done, "render_frame_fence")) {
        if (!IsDeviceLost()) {
            std::scoped_lock lock{free_mutex};
            free_queue.push(frame);
            free_cv.notify_one();
        }
        return nullptr;
    }

    if (frame->width != expected_frame_width || frame->height != expected_frame_height ||
        frame->is_hdr != swapchain.GetHDR()) {
        RecreateFrame(frame, expected_frame_width, expected_frame_height);
    }

    return frame;
}

void Presenter::SetExpectedGameSize(s32 width, s32 height) {
    const float ratio = (float)width / (float)height;

    expected_frame_height = height;
    expected_frame_width = width;
    if (ratio > expected_ratio) {
        expected_frame_width = static_cast<s32>(height * expected_ratio);
    } else {
        expected_frame_height = static_cast<s32>(width / expected_ratio);
    }
}

}
