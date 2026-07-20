// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <xxhash.h>

#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/scope_exit.h"
#include "core/libraries/videoout/video_out.h"
#include "core/memory.h"
#include "core/libraries/gnmdriver/gnmdriver.h"
#if defined(__ANDROID__)
#include <algorithm>
#include <android/log.h>
#include <cstdint>
#include <cstdlib>
#endif
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/amdgpu/render_wave_trace.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/texture_cache/tile_manager.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#if defined(__ANDROID__)
extern "C" int executor_lsx4_runtime_present_guest_frame(
    const void* source, std::uint32_t width, std::uint32_t height,
    std::uint32_t stride_bytes, std::uint32_t pixel_format, std::uint32_t tiling_mode);
#endif

namespace VideoCore {

#if defined(__ANDROID__)
static void ExecutorDumpRtPpm(const char* path, const u8* pixels, u32 width, u32 height, u32 stride,
                              u32 bpp) {
    if (!path || !pixels || width == 0 || height == 0 || bpp < 4) {
        return;
    }
    FILE* fp = std::fopen(path, "wb");
    if (!fp) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_RT_DUMP] failed_open path=%s", path);
        return;
    }
    std::fprintf(fp, "P6\n%u %u\n255\n", width, height);
    for (u32 y = 0; y < height; ++y) {
        const u8* row = pixels + static_cast<u64>(y) * stride;
        for (u32 x = 0; x < width; ++x) {
            const u8 rgb[3] = {row[static_cast<u64>(x) * bpp + 0],
                               row[static_cast<u64>(x) * bpp + 1],
                               row[static_cast<u64>(x) * bpp + 2]};
            std::fwrite(rgb, 1, sizeof(rgb), fp);
        }
    }
    std::fclose(fp);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_RT_DUMP] wrote path=%s extent=%ux%u stride=%u", path, width,
                        height, stride);
}
#endif

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           AmdGpu::Liverpool* liverpool_, BufferCache& buffer_cache_,
                           PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)} {
    // Create basic null image at fixed image ID.
    const auto null_id = GetNullImage(vk::Format::eR8G8B8A8Unorm);
    ASSERT(null_id.index == NULL_IMAGE_ID.index);

    const u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    pressure_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_PRESSURE_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
}

TextureCache::~TextureCache() = default;

ImageId TextureCache::GetNullImage(const vk::Format format) {
    const auto existing_image = null_images.find(format);
    if (existing_image != null_images.end()) {
        return existing_image->second;
    }

    ImageInfo info{};
    info.pixel_format = format;
    info.type = AmdGpu::ImageType::Color2D;
    info.tile_mode = AmdGpu::TileMode::Thin1DThin;
    info.num_bits = 32;
    info.UpdateSize();

    const ImageId null_id =
        slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
    auto& image = slot_images[null_id];
    Vulkan::SetObjectName(instance.GetDevice(), image.GetImage(),
                          fmt::format("Null Image ({})", vk::to_string(format)));

    // Match the VK_EXT_robustness2 nullDescriptor read contract on devices where the extension
    // cannot be used.  A freshly allocated Vulkan image contains undefined data and is initially
    // in eUndefined, while the descriptor fallback below is advertised as eGeneral.  Initialise
    // the fixed RGBA8 fallback to transparent black and put it in the declared layout before it
    // can be sampled by a guest shader.
    if (null_id.index == NULL_IMAGE_ID.index) {
        const vk::ClearValue zero = {
            .color = {.uint32 = std::array<u32, 4>{0, 0, 0, 0}},
        };
        image.Clear(zero, VideoCore::SubresourceRange{.extent = image.info.resources});
        image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, {});
    }

    image.flags = ImageFlagBits::Empty;
    image.track_addr = image.info.guest_address;
    image.track_addr_end = image.info.guest_address + image.info.guest_size;

    null_images.emplace(format, null_id);
    return null_id;
}

void TextureCache::ProcessDownloadImages() {
    for (auto it = download_images.begin(); it != download_images.end();) {
        if (DownloadImageMemory(*it, true)) {
            it = download_images.erase(it);
        } else {
            ++it;
        }
    }
}

bool TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return true;
    }
    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    ASSERT(download_size <= image.info.guest_size);
    const auto [download, offset] = download_buffer.Map(download_size);
    download_buffer.Commit();
    const vk::BufferImageCopy image_download = {
        .bufferOffset = offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             download_buffer.Handle(), image_download);

    if (sync) {
        scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::TextureReadback);
        const bool wrote = Core::Memory::Instance()->TryWriteBacking(
            std::bit_cast<u8*>(image.info.guest_address), download, download_size);
        if (!wrote) {
            static std::atomic<u32> failure_count{0};
            const u32 failure = failure_count.fetch_add(1, std::memory_order_relaxed);
            if (failure < 16) {
                LOG_ERROR(Render_Vulkan,
                          "GPU image readback could not write guest backing: addr={:#x}, "
                          "size={:#x}, sync=true, occurrence={}",
                          image.info.guest_address, download_size, failure + 1);
            }
            return false;
        }
    } else {
        scheduler.DeferPriorityOperation(
            [this, device_addr = image.info.guest_address, download, download_size] {
                const bool wrote = Core::Memory::Instance()->TryWriteBacking(
                    std::bit_cast<u8*>(device_addr), download, download_size);
                if (!wrote) {
                    static std::atomic<u32> failure_count{0};
                    const u32 failure = failure_count.fetch_add(1, std::memory_order_relaxed);
                    if (failure < 16) {
                        LOG_ERROR(Render_Vulkan,
                                  "Deferred GPU image readback could not write guest backing: "
                                  "addr={:#x}, size={:#x}, occurrence={}",
                                  device_addr, download_size, failure + 1);
                    }
                }
            });
    }
    return true;
}

#if defined(__ANDROID__)
namespace {

struct ExecutorGpuContentProbeConfig {
    bool enabled{};
    u32 budget{12};
    u64 max_bytes{16ull * 1024 * 1024};
    u64 draw_first{1};
    u64 draw_count{1};
    u64 draw_stride{1};
    u64 draw_hash{};
    u64 draw_hash1{};
    u64 draw_hash2{};
    u64 draw_hash3{};
    u64 draw_trigger_hash{};
    u64 dispatch_first{1};
    u64 dispatch_count{4};
};

// Mutable state is scoped by the one-shot PatchedFlip epoch. Ordinary startup-configured probes
// retain their old atomics/selection contract; the dynamic path can be re-armed repeatedly without
// restarting a title or inheriting a budget consumed by loading screens.
struct ExecutorGpuLateProbeState {
    std::mutex mutex;
    u64 epoch{};
    std::array<u64, 4> attachment_signatures{};
    u32 attachment_count{};
    u32 dispatch_count{};
    u32 issued{};
    u32 pass_end_issued{};
    u32 display_width{};
    u32 display_height{};
};

ExecutorGpuLateProbeState& ExecutorGpuLateProbeRuntime() {
    static ExecutorGpuLateProbeState state;
    return state;
}

void ExecutorGpuLateProbeResetLocked(ExecutorGpuLateProbeState& state, u64 epoch) {
    if (state.epoch == epoch) {
        return;
    }
    state.epoch = epoch;
    state.attachment_signatures.fill(0);
    state.attachment_count = 0;
    state.dispatch_count = 0;
    state.issued = 0;
    state.pass_end_issued = 0;
    state.display_width = 0;
    state.display_height = 0;
    Libraries::VideoOut::ExecutorVideoOutBufferSnapshot video_out{};
    if (Libraries::VideoOut::ExecutorGetVideoOutBufferSnapshot(0, 0, &video_out) &&
        video_out.valid) {
        state.display_width = video_out.width;
        state.display_height = video_out.height;
    }
}

u64 ExecutorGpuProbeEnvU64(const char* name, u64 fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char* end{};
    const u64 parsed = std::strtoull(value, &end, 0);
    return end != value ? parsed : fallback;
}

const ExecutorGpuContentProbeConfig& ExecutorGpuProbeConfig() {
    static const ExecutorGpuContentProbeConfig config = [] {
        ExecutorGpuContentProbeConfig value{};
        const char* marker = std::getenv("EXECUTOR_GPU_CONTENT_PROBE");
        value.enabled = marker && *marker && std::strcmp(marker, "0") != 0;
        value.budget = static_cast<u32>(
            std::clamp<u64>(ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_BUDGET", 12),
                            1, 32));
        value.max_bytes = std::clamp<u64>(
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_MAX_BYTES",
                                   16ull * 1024 * 1024),
            4096, 64ull * 1024 * 1024);
        value.draw_first =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DRAW_FIRST", 1);
        value.draw_count =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DRAW_COUNT", 1);
        value.draw_stride = std::max<u64>(
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DRAW_STRIDE", 1), 1);
        value.draw_hash =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DRAW_HASH", 0);
        value.draw_hash1 =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DRAW_HASH1", 0);
        value.draw_hash2 =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DRAW_HASH2", 0);
        value.draw_hash3 =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DRAW_HASH3", 0);
        value.draw_trigger_hash =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DRAW_TRIGGER_HASH", 0);
        value.dispatch_first =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DISPATCH_FIRST", 1);
        value.dispatch_count =
            ExecutorGpuProbeEnvU64("EXECUTOR_GPU_CONTENT_PROBE_DISPATCH_COUNT", 4);
        return value;
    }();
    return config;
}

bool ExecutorGpuProbeSequenceSelected(u64 sequence, u64 first, u64 count, u64 stride = 1) {
    if (count == 0 || sequence < first || stride == 0) {
        return false;
    }
    const u64 delta = sequence - first;
    return delta % stride == 0 && delta / stride < count;
}

const char* ExecutorGpuProbeRoleName(TextureCache::ExecutorGpuContentRole role) {
    switch (role) {
    case TextureCache::ExecutorGpuContentRole::SampledBeforeDraw:
        return "sampled_pre_draw";
    case TextureCache::ExecutorGpuContentRole::ColorBeforeDraw:
        return "color_pre_draw";
    case TextureCache::ExecutorGpuContentRole::DepthBeforeDraw:
        return "depth_pre_draw";
    case TextureCache::ExecutorGpuContentRole::StorageAfterDispatch:
        return "storage_post_dispatch";
    case TextureCache::ExecutorGpuContentRole::ColorAfterDraw:
        return "color_post_draw";
    case TextureCache::ExecutorGpuContentRole::DepthAfterDraw:
        return "depth_post_draw";
    case TextureCache::ExecutorGpuContentRole::ColorPassEnd:
        return "color_pass_end";
    case TextureCache::ExecutorGpuContentRole::DepthPassEnd:
        return "depth_pass_end";
    }
    return "unknown";
}

} // namespace
#endif

bool TextureCache::ExecutorGpuContentProbeEnabled() const {
#if defined(__ANDROID__)
    const auto& config = ExecutorGpuProbeConfig();
    if (config.enabled) {
        static std::atomic<bool> logged{false};
        if (!logged.exchange(true, std::memory_order_relaxed)) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_CONTENT_PROBE_CONFIG] budget=%u maxBytes=%llu "
                "drawFirst=%llu drawCount=%llu drawStride=%llu "
                "drawHashes=0x%llx,0x%llx,0x%llx,0x%llx triggerHash=0x%llx "
                "dispatchFirst=%llu "
                "dispatchCount=%llu",
                config.budget, static_cast<unsigned long long>(config.max_bytes),
                static_cast<unsigned long long>(config.draw_first),
                static_cast<unsigned long long>(config.draw_count),
                static_cast<unsigned long long>(config.draw_stride),
                static_cast<unsigned long long>(config.draw_hash),
                static_cast<unsigned long long>(config.draw_hash1),
                static_cast<unsigned long long>(config.draw_hash2),
                static_cast<unsigned long long>(config.draw_hash3),
                static_cast<unsigned long long>(config.draw_trigger_hash),
                static_cast<unsigned long long>(config.dispatch_first),
                static_cast<unsigned long long>(config.dispatch_count));
        }
    }
    return config.enabled || AmdGpu::RenderWaveTrace::LateFrameCaptureActive();
#else
    return false;
#endif
}

bool TextureCache::ExecutorGpuContentProbeDrawSelected(
    u64 sequence, u64 fragment_hash, u64 attachment_signature, u32 color_attachment_count,
    u32 color_width, u32 color_height, bool has_depth, u32 element_count, bool indirect) const {
#if defined(__ANDROID__)
    const auto& config = ExecutorGpuProbeConfig();
    if (AmdGpu::RenderWaveTrace::LateFrameCaptureActive()) {
        auto& runtime = ExecutorGpuLateProbeRuntime();
        std::scoped_lock lock{runtime.mutex};
        ExecutorGpuLateProbeResetLocked(runtime, AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch());
        // Select scene-scale geometry passes, not the clear, shadow-map and cubemap work which
        // normally leads a deferred frame. Half of each registered VideoOut dimension is a
        // resolution-independent lower bound that still admits dynamic-resolution rendering.
        // This remains a diagnostic pass classifier: no title, shader, guest address or format is
        // part of the decision.
        const bool scene_scale =
            color_width != 0 && color_height != 0 &&
            (runtime.display_width == 0 || color_width * 2 >= runtime.display_width) &&
            (runtime.display_height == 0 || color_height * 2 >= runtime.display_height);
        const bool substantial_geometry = indirect || element_count >= 64;
        if (fragment_hash == 0 || attachment_signature == 0 || color_attachment_count == 0 ||
            !has_depth || !scene_scale || !substantial_geometry) {
            return false;
        }
        // Read back one representative draw for each of the first four qualifying attachment sets.
        // This samples G-buffer and later scene-depth transitions without synchronously downloading
        // every mesh in a full gameplay frame.
        const u64 identity = attachment_signature != 0 ? attachment_signature : ~u64{0};
        if (std::ranges::find(runtime.attachment_signatures.begin(),
                             runtime.attachment_signatures.begin() + runtime.attachment_count,
                             identity) != runtime.attachment_signatures.begin() +
                                             runtime.attachment_count) {
            return false;
        }
        if (runtime.attachment_count >= runtime.attachment_signatures.size()) {
            return false;
        }
        runtime.attachment_signatures[runtime.attachment_count++] = identity;
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GPU_FRAME_ATTACHMENT_SELECT] epoch=%llu group=%u seq=%llu "
            "attach=0x%llx ps=0x%llx colors=%u extent=%ux%u depth=%u elements=%u indirect=%u",
            static_cast<unsigned long long>(runtime.epoch), runtime.attachment_count,
            static_cast<unsigned long long>(sequence),
            static_cast<unsigned long long>(attachment_signature),
            static_cast<unsigned long long>(fragment_hash), color_attachment_count, color_width,
            color_height, has_depth ? 1u : 0u, element_count, indirect ? 1u : 0u);
        return true;
    }
    if (!config.enabled) {
        return false;
    }
    const std::array hashes{config.draw_hash, config.draw_hash1, config.draw_hash2,
                            config.draw_hash3};
    // A trigger makes first/count relative to a later render-graph phase. It is a generic probe
    // primitive: early logo/loading occurrences of the same post shader no longer consume the
    // bounded capture intended for the first gameplay frame.
    if (config.draw_trigger_hash != 0) {
        static std::atomic<bool> trigger_seen{false};
        if (fragment_hash == config.draw_trigger_hash) {
            trigger_seen.store(true, std::memory_order_release);
        }
        if (!trigger_seen.load(std::memory_order_acquire)) {
            return false;
        }
    }
    const bool has_hash_filter = std::ranges::any_of(hashes, [](u64 hash) { return hash != 0; });
    if (!has_hash_filter) {
        return ExecutorGpuProbeSequenceSelected(sequence, config.draw_first, config.draw_count,
                                                config.draw_stride);
    }
    if (fragment_hash == 0 ||
        std::ranges::find_if(hashes, [fragment_hash](u64 hash) {
            return hash != 0 && hash == fragment_hash;
        }) == hashes.end()) {
        return false;
    }
    // With a hash filter, first/count/stride describe matching draws rather than global draw
    // ordinals. This makes the bounded readback useful for a recurring post-process pass without
    // adding a title- or shader-specific branch to the renderer.
    // Count each requested shader independently.  A frequently recurring first hash must not
    // consume the global budget before later render-graph stages are reached.
    const auto hash_it = std::ranges::find(hashes, fragment_hash);
    const auto hash_index = static_cast<std::size_t>(std::distance(hashes.begin(), hash_it));
    static std::array<std::atomic<u64>, 4> matching_sequences{};
    const u64 occurrence =
        matching_sequences[hash_index].fetch_add(1, std::memory_order_relaxed) + 1;
    return ExecutorGpuProbeSequenceSelected(occurrence, config.draw_first, config.draw_count,
                                            config.draw_stride);
#else
    (void)sequence;
    (void)fragment_hash;
    (void)attachment_signature;
    (void)color_attachment_count;
    (void)color_width;
    (void)color_height;
    (void)has_depth;
    (void)element_count;
    (void)indirect;
    return false;
#endif
}

bool TextureCache::ExecutorGpuContentProbeDispatchSelected(u64 sequence) const {
#if defined(__ANDROID__)
    const auto& config = ExecutorGpuProbeConfig();
    if (AmdGpu::RenderWaveTrace::LateFrameCaptureActive()) {
        auto& runtime = ExecutorGpuLateProbeRuntime();
        std::scoped_lock lock{runtime.mutex};
        ExecutorGpuLateProbeResetLocked(runtime, AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch());
        return runtime.dispatch_count++ < 2;
    }
    return config.enabled && ExecutorGpuProbeSequenceSelected(
                                 sequence, config.dispatch_first, config.dispatch_count);
#else
    (void)sequence;
    return false;
#endif
}

void TextureCache::ExecutorGpuContentProbe(ImageId image_id, ExecutorGpuContentRole role,
                                           u32 stage, u64 stage_hash, u64 sequence) {
#if defined(__ANDROID__)
    const auto& config = ExecutorGpuProbeConfig();
    const bool late_capture = AmdGpu::RenderWaveTrace::LateFrameCaptureActive();
    if ((!config.enabled && !late_capture) || !image_id) {
        return;
    }

    static std::atomic<u32> startup_issued{0};
    u32 ordinal{};
    const bool pass_end_role = role == ExecutorGpuContentRole::ColorPassEnd ||
                               role == ExecutorGpuContentRole::DepthPassEnd;
    // A complete deferred pass can expose eight MRTs plus depth. Keep pass-end accounting separate
    // from dispatch/storage probes so earlier diagnostics in the same late-frame epoch cannot
    // consume the evidence we armed the capture for. Four selected attachment groups fit within 64.
    const u32 effective_budget = late_capture ? (pass_end_role ? 64u : 32u) : config.budget;
    if (late_capture) {
        auto& runtime = ExecutorGpuLateProbeRuntime();
        std::scoped_lock lock{runtime.mutex};
        ExecutorGpuLateProbeResetLocked(runtime, AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch());
        ordinal = pass_end_role ? runtime.pass_end_issued++ : runtime.issued++;
    } else {
        ordinal = startup_issued.fetch_add(1, std::memory_order_relaxed);
    }
    if (ordinal >= effective_budget) {
        return;
    }

    Image& image = slot_images[image_id];
    const char* role_name = ExecutorGpuProbeRoleName(role);
    const u32 bpp = image.info.num_bits / 8u;
    const u32 width = image.info.size.width;
    const u32 height = image.info.size.height;
    const u32 depth = std::max(image.info.size.depth, 1u);
    const u32 layers = std::max(image.info.resources.layers, 1u);
    const u64 layer_stride = static_cast<u64>(image.info.pitch) * height * depth * bpp;
    const u64 download_size = layer_stride * layers;
    const bool unsupported = image.info.props.is_block || bpp == 0 ||
                             image.info.num_bits % 8u != 0 || image.info.num_samples != 1 ||
                             width == 0 || height == 0 || image.info.pitch < width ||
                             download_size == 0 || download_size > config.max_bytes ||
                             image.backing->state.layout == vk::ImageLayout::eUndefined;
    if (unsupported) {
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_GPU_CONTENT_PROBE] ordinal=%u role=%s seq=%llu stage=%u hash=0x%llx "
            "imageId=%u status=skip format=%s extent=%ux%ux%u pitch=%u layers=%u bits=%u "
            "samples=%u depth=%u block=%u bytes=%llu maxBytes=%llu layout=%u",
            ordinal + 1, role_name, static_cast<unsigned long long>(sequence), stage,
            static_cast<unsigned long long>(stage_hash), image_id.index,
            vk::to_string(image.info.pixel_format).c_str(), width, height, depth, image.info.pitch,
            layers, image.info.num_bits, image.info.num_samples,
            image.info.props.is_depth ? 1u : 0u, image.info.props.is_block ? 1u : 0u,
            static_cast<unsigned long long>(download_size),
            static_cast<unsigned long long>(config.max_bytes),
            static_cast<u32>(image.backing->state.layout));
        return;
    }

    // Preserve the state expected by the already-built descriptor/render state. The full-resource
    // diagnostic transition intentionally collapses subresource tracking; restoring the aggregate
    // state is conservative and subsequent partial bindings recreate per-subresource states.
    const Image::State restore_state = image.backing->state;
    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    const auto [download, offset] = download_buffer.Map(download_size);
    download_buffer.Commit();
    const vk::ImageAspectFlags probe_aspect = image.info.props.is_depth
                                                  ? vk::ImageAspectFlagBits::eDepth
                                                  : vk::ImageAspectFlagBits::eColor;
    const vk::BufferImageCopy region = {
        .bufferOffset = offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = height,
        .imageSubresource = {.aspectMask = probe_aspect,
                             .mipLevel = 0,
                             .baseArrayLayer = 0,
                             .layerCount = layers},
        .imageOffset = {0, 0, 0},
        .imageExtent = {width, height, depth},
    };

    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
                  cmdbuf);
    cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             download_buffer.Handle(), region);
    const auto restore_barriers = image.GetBarriers(
        restore_state.layout, restore_state.access_mask, restore_state.pl_stage, {});
    if (!restore_barriers.empty()) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .imageMemoryBarrierCount = static_cast<u32>(restore_barriers.size()),
            .pImageMemoryBarriers = restore_barriers.data(),
        });
    }
    scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::TextureProbe);

    const auto* bytes = reinterpret_cast<const u8*>(download);
    const u64 content_hash = XXH3_64bits(bytes, download_size);
    u64 nonzero = 0;
    for (u64 i = 0; i < download_size; ++i) {
        nonzero += bytes[i] != 0;
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_GPU_CONTENT_PROBE] ordinal=%u role=%s seq=%llu stage=%u hash=0x%llx "
        "imageId=%u status=ok format=%s extent=%ux%ux%u pitch=%u layers=%u bpp=%u "
        "bytes=%llu contentHash=0x%llx nonzero=%llu layout=%u access=0x%llx flags=0x%x "
        "usageTexture=%u usageStorage=%u usageRt=%u",
        ordinal + 1, role_name, static_cast<unsigned long long>(sequence), stage,
        static_cast<unsigned long long>(stage_hash), image_id.index,
        vk::to_string(image.info.pixel_format).c_str(), width, height, depth, image.info.pitch,
        layers, bpp, static_cast<unsigned long long>(download_size),
        static_cast<unsigned long long>(content_hash), static_cast<unsigned long long>(nonzero),
        static_cast<u32>(restore_state.layout),
        static_cast<unsigned long long>(static_cast<VkAccessFlags2>(restore_state.access_mask)),
        static_cast<u32>(image.flags), image.usage.texture, image.usage.storage,
        image.usage.render_target);

    // Exact D32 classes turn the destructive depth-write oracle into a binary result even when
    // the selected geometry covers none of the small grid samples logged below.  0.25f is the
    // oracle sentinel, 1.0f is the guest's ordinary far-plane clear, and every other bit pattern
    // proves that the draw produced a nontrivial depth value.  This is useful for every title and
    // intentionally does not depend on a shader hash or a known frame.
    if (image.info.props.is_depth && bpp >= sizeof(u32)) {
        constexpr u32 Sentinel025Bits = 0x3e800000u;
        constexpr u32 OneBits = 0x3f800000u;
        u64 sentinel_025_count = 0;
        u64 one_count = 0;
        u64 other_count = 0;
        u32 other_min = std::numeric_limits<u32>::max();
        u32 other_max = 0;
        for (u32 layer = 0; layer < layers; ++layer) {
            const u8* const layer_bytes = bytes + static_cast<u64>(layer) * layer_stride;
            for (u32 y = 0; y < height; ++y) {
                const u8* const row =
                    layer_bytes + static_cast<u64>(y) * image.info.pitch * bpp;
                for (u32 x = 0; x < width; ++x) {
                    u32 value{};
                    std::memcpy(&value, row + static_cast<u64>(x) * bpp, sizeof(value));
                    if (value == Sentinel025Bits) {
                        ++sentinel_025_count;
                    } else if (value == OneBits) {
                        ++one_count;
                    } else {
                        ++other_count;
                        other_min = std::min(other_min, value);
                        other_max = std::max(other_max, value);
                    }
                }
            }
        }
        if (other_count == 0) {
            other_min = 0;
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GPU_CONTENT_DEPTH_D32] ordinal=%u role=%s seq=%llu hash=0x%llx "
            "imageId=%u pixels=%llu sentinel025=%llu one=%llu other=%llu "
            "otherBitsMin=0x%08x otherBitsMax=0x%08x",
            ordinal + 1, role_name, static_cast<unsigned long long>(sequence),
            static_cast<unsigned long long>(stage_hash), image_id.index,
            static_cast<unsigned long long>(static_cast<u64>(width) * height * layers),
            static_cast<unsigned long long>(sentinel_025_count),
            static_cast<unsigned long long>(one_count),
            static_cast<unsigned long long>(other_count), other_min, other_max);
    }

    if (bpp == 4 && !image.info.props.is_depth) {
        std::array<u8, 4> channel_min{255, 255, 255, 255};
        std::array<u8, 4> channel_max{};
        std::array<u64, 4> channel_sum{};
        u64 rg_neutral = 0;
        u64 rg_saturated = 0;
        u64 horizontal_l1 = 0;
        u64 vertical_l1 = 0;
        const u64 pixel_count = static_cast<u64>(width) * height * layers;
        for (u32 layer = 0; layer < layers; ++layer) {
            const u8* const layer_bytes = bytes + static_cast<u64>(layer) * layer_stride;
            for (u32 y = 0; y < height; ++y) {
                const u8* const row = layer_bytes + static_cast<u64>(y) * image.info.pitch * bpp;
                const u8* const previous_row =
                    y == 0 ? nullptr
                           : layer_bytes + static_cast<u64>(y - 1) * image.info.pitch * bpp;
                for (u32 x = 0; x < width; ++x) {
                    const u8* const pixel = row + static_cast<u64>(x) * bpp;
                    for (u32 channel = 0; channel < 4; ++channel) {
                        channel_min[channel] = std::min(channel_min[channel], pixel[channel]);
                        channel_max[channel] = std::max(channel_max[channel], pixel[channel]);
                        channel_sum[channel] += pixel[channel];
                        if (x != 0) {
                            horizontal_l1 += std::abs(static_cast<int>(pixel[channel]) -
                                                      static_cast<int>((pixel - bpp)[channel]));
                        }
                        if (previous_row) {
                            vertical_l1 +=
                                std::abs(static_cast<int>(pixel[channel]) -
                                         static_cast<int>(previous_row[x * bpp + channel]));
                        }
                    }
                    const bool neutral_r = pixel[0] >= 127 && pixel[0] <= 129;
                    const bool neutral_g = pixel[1] >= 127 && pixel[1] <= 129;
                    rg_neutral += neutral_r && neutral_g;
                    rg_saturated += pixel[0] <= 1 || pixel[0] >= 254 || pixel[1] <= 1 ||
                                    pixel[1] >= 254;
                }
            }
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GPU_CONTENT_RGBA8] ordinal=%u role=%s seq=%llu hash=0x%llx "
            "imageId=%u pixels=%llu min=%u,%u,%u,%u max=%u,%u,%u,%u "
            "mean=%.3f,%.3f,%.3f,%.3f rgNeutral=%llu rgSaturated=%llu "
            "horizontalL1=%llu verticalL1=%llu",
            ordinal + 1, role_name, static_cast<unsigned long long>(sequence),
            static_cast<unsigned long long>(stage_hash), image_id.index,
            static_cast<unsigned long long>(pixel_count), channel_min[0], channel_min[1],
            channel_min[2], channel_min[3], channel_max[0], channel_max[1], channel_max[2],
            channel_max[3], static_cast<double>(channel_sum[0]) / pixel_count,
            static_cast<double>(channel_sum[1]) / pixel_count,
            static_cast<double>(channel_sum[2]) / pixel_count,
            static_cast<double>(channel_sum[3]) / pixel_count,
            static_cast<unsigned long long>(rg_neutral),
            static_cast<unsigned long long>(rg_saturated),
            static_cast<unsigned long long>(horizontal_l1),
            static_cast<unsigned long long>(vertical_l1));
    }

    // A center-row-only oracle cannot distinguish a valid 2-D image from a vertically clamped or
    // repeated surface.  Five bounded rows/columns make that distinction without dumping the image
    // or adding any title-specific interpretation to the renderer.
    const u32 logged_layers = std::min(layers, 2u);
    const std::array<u32, 5> xs{0u, width / 4, width / 2, (width * 3) / 4, width - 1};
    const std::array<u32, 5> ys{0u, height / 4, height / 2, (height * 3) / 4, height - 1};
    for (u32 layer = 0; layer < logged_layers; ++layer) {
        for (const u32 y : ys) {
            std::array<u64, 5> samples{};
            for (u32 sample = 0; sample < xs.size(); ++sample) {
                const u64 sample_offset =
                    static_cast<u64>(layer) * layer_stride +
                    (static_cast<u64>(y) * image.info.pitch + xs[sample]) * bpp;
                const u32 sample_bytes = std::min(bpp, 8u);
                std::memcpy(&samples[sample], bytes + sample_offset, sample_bytes);
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_CONTENT_GRID] ordinal=%u role=%s seq=%llu stage=%u hash=0x%llx "
                "imageId=%u layer=%u y=%u bpp=%u x0=%u:0x%016llx x1=%u:0x%016llx "
                "x2=%u:0x%016llx x3=%u:0x%016llx x4=%u:0x%016llx",
                ordinal + 1, role_name, static_cast<unsigned long long>(sequence), stage,
                static_cast<unsigned long long>(stage_hash), image_id.index, layer, y, bpp,
                xs[0], static_cast<unsigned long long>(samples[0]), xs[1],
                static_cast<unsigned long long>(samples[1]), xs[2],
                static_cast<unsigned long long>(samples[2]), xs[3],
                static_cast<unsigned long long>(samples[3]), xs[4],
                static_cast<unsigned long long>(samples[4]));
        }
    }
#else
    (void)image_id;
    (void)role;
    (void)stage;
    (void)stage_hash;
    (void)sequence;
#endif
}

void TextureCache::ExecutorTraceVideoOutSource(ImageId image_id, u64 sequence,
                                               u32 flags_before_refresh) {
#ifdef __ANDROID__
    Image& image = slot_images[image_id];
    const u32 bpp = std::max<u32>(image.info.num_bits / 8u, 1u);
    const u64 download_size = static_cast<u64>(image.info.pitch) * image.info.size.height * bpp;
    if (download_size == 0 || download_size > (64ull * 1024 * 1024)) {
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_VIDEOOUT_SOURCE] seq=%llu id=%u skip=bad_size addr=0x%llx "
            "extent=%ux%u pitch=%u bpp=%u bytes=%llu flagsBefore=0x%x flagsAfter=0x%x",
            static_cast<unsigned long long>(sequence), image_id.index,
            static_cast<unsigned long long>(image.info.guest_address), image.info.size.width,
            image.info.size.height, image.info.pitch, bpp,
            static_cast<unsigned long long>(download_size), flags_before_refresh,
            static_cast<u32>(image.flags));
        return;
    }

    // The CPU backing can legitimately be stale for a GPU-modified render target. Recording both
    // hashes tells us whether a black present originates before TextureCache, in TextureCache's
    // selected image, or later in the post-process/swapchain path.
    std::vector<u8> cpu_bytes(download_size);
    Core::Memory::Instance()->CopySparseMemory(image.info.guest_address, cpu_bytes.data(),
                                               download_size);
    const u64 cpu_hash = XXH3_64bits(cpu_bytes.data(), cpu_bytes.size());
    u64 cpu_nonzero = 0;
    for (const u8 value : cpu_bytes) {
        cpu_nonzero += value != 0;
    }

    auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
    const auto [download, offset] = download_buffer.Map(download_size);
    download_buffer.Commit();
    const vk::BufferImageCopy region = {
        .bufferOffset = offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                             .mipLevel = 0,
                             .baseArrayLayer = 0,
                             .layerCount = 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, 1},
    };
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();
    image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                             download_buffer.Handle(), region);
    scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::TextureProbe);

    const auto* gpu_bytes = reinterpret_cast<const u8*>(download);
    const u64 gpu_hash = XXH3_64bits(gpu_bytes, download_size);
    std::array<u64, 4> gpu_nonzero{};
    for (u64 offset_px = 0; offset_px < download_size; ++offset_px) {
        gpu_nonzero[offset_px & 3] += gpu_bytes[offset_px] != 0;
    }
    const u64 middle = (download_size / 2) & ~u64(15);

    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VIDEOOUT_SOURCE] seq=%llu id=%u addr=0x%llx format=%s extent=%ux%u "
        "pitch=%u bpp=%u tiled=%u bytes=%llu flagsBefore=0x%x flagsAfter=0x%x "
        "usageVo=%u usageRt=%u bound=%u target=%u cpuHash=0x%llx cpuNz=%llu "
        "gpuHash=0x%llx gpuNz=%llu,%llu,%llu,%llu mid=%02x%02x%02x%02x",
        static_cast<unsigned long long>(sequence), image_id.index,
        static_cast<unsigned long long>(image.info.guest_address),
        vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
        image.info.size.height, image.info.pitch, bpp, image.info.props.is_tiled ? 1u : 0u,
        static_cast<unsigned long long>(download_size), flags_before_refresh,
        static_cast<u32>(image.flags), image.usage.vo_surface, image.usage.render_target,
        image.binding.is_bound, image.binding.is_target,
        static_cast<unsigned long long>(cpu_hash), static_cast<unsigned long long>(cpu_nonzero),
        static_cast<unsigned long long>(gpu_hash),
        static_cast<unsigned long long>(gpu_nonzero[0]),
        static_cast<unsigned long long>(gpu_nonzero[1]),
        static_cast<unsigned long long>(gpu_nonzero[2]),
        static_cast<unsigned long long>(gpu_nonzero[3]), gpu_bytes[middle + 0],
        gpu_bytes[middle + 1], gpu_bytes[middle + 2], gpu_bytes[middle + 3]);

    static std::mutex dump_mutex;
    static std::array<u64, 8> dumped_source_addresses{};
    if (bpp >= 4 && (gpu_nonzero[0] + gpu_nonzero[1] + gpu_nonzero[2]) != 0) {
        std::scoped_lock lock{dump_mutex};
        const u64 source_address = image.info.guest_address;
        const bool already_dumped =
            std::ranges::find(dumped_source_addresses, source_address) !=
            dumped_source_addresses.end();
        const auto free_slot = std::ranges::find(dumped_source_addresses, 0);
        if (!already_dumped && free_slot != dumped_source_addresses.end()) {
            *free_slot = source_address;
            char path[256]{};
            std::snprintf(
                path, sizeof(path),
                "/data/user/0/app.lsx4.android/files/lsx4-home/"
                "live-videoout-source-%016llx.ppm",
                static_cast<unsigned long long>(source_address));
            ExecutorDumpRtPpm(path, gpu_bytes, image.info.size.width, image.info.size.height,
                              image.info.pitch * bpp, bpp);
        }
    }
#else
    (void)image_id;
    (void)sequence;
    (void)flags_before_refresh;
#endif
}

void TextureCache::ExecutorTrackLiveDrawnColorTarget(ImageId image_id) {
#ifdef __ANDROID__
    if (!image_id) {
        return;
    }
    const u32 image_index = image_id.index;
    if (executor_live_last_color_target.exchange(image_index, std::memory_order_relaxed) ==
        image_index) {
        return;
    }
    std::scoped_lock lk{executor_replay_candidates_mutex};
    // A frame can contain thousands of consecutive draws to one attachment. Keep only target
    // transitions: selection is based on write order, not draw count, and a bounded list avoids
    // turning the frame-boundary bridge itself into a hot allocation path.
    if (executor_live_drawn_color_candidates.empty() ||
        executor_live_drawn_color_candidates.back() != image_id) {
        executor_live_drawn_color_candidates.push_back(image_id);
    }
#else
    (void)image_id;
#endif
}

ImageId TextureCache::ExecutorTakeLiveFullResolutionTarget(u32 width, u32 height,
                                                           ImageId preferred) {
#ifdef __ANDROID__
    std::vector<ImageId> candidates;
    {
        std::scoped_lock lk{executor_replay_candidates_mutex};
        candidates.swap(executor_live_drawn_color_candidates);
        executor_live_last_color_target.store(0, std::memory_order_relaxed);
    }

    // The game can bind small bloom/shadow/resolve targets after its composite draw. Prefer the last
    // target that exactly matches the registered VideoOut resolution. Engines may also render at a
    // dynamic internal resolution (for example 1280x720 into a 1920x1080 scanout), so fall back to the
    // largest sufficiently-sized target with the same aspect ratio. The regular Presenter path scales
    // that source exactly as it scales a native VideoOut image.
    if (width == 0 || height == 0) {
        return {};
    }

    const bool preferred_valid = preferred && slot_images.is_allocated(preferred);
    const Image* preferred_image = preferred_valid ? &GetImage(preferred) : nullptr;
    const auto candidate_valid = [&](const ImageId id) {
        return id && slot_images.is_allocated(id);
    };

    // A narrow source trace must show every image that was actually drawn between two flips, not
    // only the scanout image eventually selected below. This is intentionally a one-shot diagnostic:
    // each readback waits for the corresponding GPU work and therefore cannot live on the frame hot
    // path. It lets a white/black registered scanout be distinguished from a valid intermediate
    // composite without changing the source-selection policy being measured.
    static std::atomic<bool> traced_live_candidate_set{false};
    if (std::getenv("EXECUTOR_TRACE_VIDEOOUT_SOURCE") != nullptr &&
        !traced_live_candidate_set.exchange(true, std::memory_order_relaxed)) {
        std::array<u32, 16> traced_ids{};
        std::size_t traced_count = 0;
        for (const ImageId id : candidates) {
            if (!candidate_valid(id) ||
                std::ranges::find(traced_ids.begin(), traced_ids.begin() + traced_count, id.index) !=
                    traced_ids.begin() + traced_count) {
                continue;
            }
            traced_ids[traced_count++] = id.index;
            ExecutorTraceVideoOutSource(id, 0x10000u + traced_count, 0);
            if (traced_count == traced_ids.size()) {
                break;
            }
        }
    }

    // A VideoOut flip names the scanout buffer that owns this presentation. Preserve that exact PC
    // contract whenever the selected buffer was among the images actually written in this batch.
    // The full-resolution heuristic below is only a fallback for engines that render the final
    // composite into a distinct intermediate and leave the registered scanout backing untouched.
    if (preferred_valid && std::ranges::find(candidates, preferred) != candidates.end()) {
        return preferred;
    }

    // FindImage can return a distinct RT alias for the exact VideoOut guest address. This is the
    // strongest source relation after identity and is the same address contract used by the PC
    // path; prefer it before any resolution heuristic.
    if (preferred_image) {
        for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
            if (!candidate_valid(*it)) {
                continue;
            }
            const auto& image = GetImage(*it);
            if (image.info.guest_address == preferred_image->info.guest_address &&
                IsVulkanFormatCompatible(image.info.pixel_format,
                                         preferred_image->info.pixel_format)) {
                return *it;
            }
        }
    }

    ImageId exact_best{};
    int exact_best_score = std::numeric_limits<int>::min();
    for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
        if (!candidate_valid(*it)) {
            continue;
        }
        const auto& image = GetImage(*it);
        if (!image.info.props.is_depth && image.info.size.width == width &&
            image.info.size.height == height) {
            int score = image.usage.vo_surface ? 8 : 0;
            if (preferred_image) {
                score += IsVulkanFormatCompatible(image.info.pixel_format,
                                                   preferred_image->info.pixel_format)
                    ? 4
                    : 0;
                score += image.info.type == preferred_image->info.type ? 2 : 0;
            }
            if (score > exact_best_score) {
                exact_best = *it;
                exact_best_score = score;
            }
        }
    }
    if (exact_best) {
        return exact_best;
    }

    ImageId best{};
    u64 best_area = 0;
    u32 best_width = 0;
    u32 best_height = 0;
    for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
        if (!candidate_valid(*it)) {
            continue;
        }
        const auto& image = GetImage(*it);
        if (image.info.props.is_depth) {
            continue;
        }
        const u32 candidate_width = image.info.size.width;
        const u32 candidate_height = image.info.size.height;
        if (candidate_width < width / 2 || candidate_height < height / 2) {
            continue;
        }
        const u64 lhs = static_cast<u64>(candidate_width) * height;
        const u64 rhs = static_cast<u64>(candidate_height) * width;
        const u64 difference = lhs > rhs ? lhs - rhs : rhs - lhs;
        const u64 aspect_scale = std::max(lhs, rhs);
        if (difference > std::max<u64>(1, aspect_scale / 100)) {
            continue;
        }
        const u64 area = static_cast<u64>(candidate_width) * candidate_height;
        if (area > best_area) {
            best = *it;
            best_area = area;
            best_width = candidate_width;
            best_height = candidate_height;
        }
    }
    if (best) {
        static std::atomic<bool> logged_dynamic_source{false};
        if (!logged_dynamic_source.exchange(true, std::memory_order_relaxed)) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_FRAME_SOURCE_BRIDGE] scanout=%ux%u source=%ux%u target=%u candidates=%zu",
                width, height, best_width, best_height, best.index, candidates.size());
        }
    }
    return best;
#else
    (void)width;
    (void)height;
    (void)preferred;
    return {};
#endif
}

void TextureCache::ExecutorReplayReadbackColorTarget() {
#ifdef __ANDROID__
    // Build the candidate list (fall back to the single last-bound target if none were tracked).
    // Snapshot under the lock so a concurrent GPU-coroutine push_back cannot tear the vector mid-copy.
    std::vector<ImageId> candidates;
    {
        std::scoped_lock lk{executor_replay_candidates_mutex};
        candidates = executor_replay_color_candidates;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_RT_READBACK] enter candidates=%zu fallback=%u",
                        candidates.size(), executor_replay_color_target.index);
    if (candidates.empty() && executor_replay_color_target) {
        candidates.push_back(executor_replay_color_target);
    }
    if (candidates.empty()) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_RT_READBACK] skip reason=no_candidates");
        return;
    }
    scheduler.EndRendering();

    // Read back one color target and return its stats. Forces GPU completion so the bytes are valid.
    struct RtStats {
        ImageId id{};
        u32 w{}, h{};
        u32 pitch{}, bpp{};
        u64 nzc[4]{};
        u32 unique_rgb{};
        u8 mid[8]{};
        std::vector<u8> rgba;
        bool ok{false};
    };
    auto readback = [&](ImageId id) -> RtStats {
        RtStats st{};
        st.id = id;
        Image& image = slot_images[id];
        const u32 bpp = std::max<u32>(image.info.num_bits / 8u, 1u);
        const u32 w = image.info.size.width;
        const u32 h = image.info.size.height;
        const u64 download_size = static_cast<u64>(image.info.pitch) * h * bpp;
        if (download_size == 0 || download_size > (64ull * 1024 * 1024)) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_RT_READBACK] skip_candidate id=%u reason=bad_size "
                                "extent=%ux%u pitch=%u bpp=%u size=%llu bits=%u",
                                id.index, w, h, image.info.pitch, bpp,
                                static_cast<unsigned long long>(download_size),
                                image.info.num_bits);
            return st;
        }
        auto& download_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
        const auto [download, offset] = download_buffer.Map(download_size);
        download_buffer.Commit();
        const vk::BufferImageCopy region = {
            .bufferOffset = offset,
            .bufferRowLength = image.info.pitch,
            .bufferImageHeight = h,
            .imageSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                 .mipLevel = 0,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {w, h, 1},
        };
        const auto cmdbuf = scheduler.CommandBuffer();
        image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
        cmdbuf.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                 download_buffer.Handle(), region);
        scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::TextureProbe);
        const u8* px = reinterpret_cast<const u8*>(download);
        const u64 scan = std::min<u64>(download_size, 8ull * 1024 * 1024);
        for (u64 i = 0; i < scan; ++i) {
            if (px[i] != 0) {
                ++st.nzc[i & 3];
            }
        }
        const u64 mid = (download_size / 2) & ~u64(15);
        for (int k = 0; k < 8; ++k) {
            st.mid[k] = px[mid + k];
        }
        if (bpp >= 4 && w > 0 && h > 0) {
            std::array<u32, 256> seen{};
            u32 seen_count = 0;
            constexpr u32 GridX = 64;
            constexpr u32 GridY = 36;
            for (u32 gy = 0; gy < GridY; ++gy) {
                const u32 y = std::min<u32>((static_cast<u64>(gy) * h) / GridY, h - 1);
                for (u32 gx = 0; gx < GridX; ++gx) {
                    const u32 x = std::min<u32>((static_cast<u64>(gx) * w) / GridX, w - 1);
                    const u64 off = (static_cast<u64>(y) * image.info.pitch + x) * bpp;
                    if (off + 2 >= download_size) {
                        continue;
                    }
                    const u32 rgb = static_cast<u32>(px[off]) |
                                    (static_cast<u32>(px[off + 1]) << 8) |
                                    (static_cast<u32>(px[off + 2]) << 16);
                    if (rgb == 0) {
                        continue;
                    }
                    bool known = false;
                    for (u32 i = 0; i < seen_count; ++i) {
                        if (seen[i] == rgb) {
                            known = true;
                            break;
                        }
                    }
                    if (!known) {
                        seen[seen_count++] = rgb;
                        if (seen_count == seen.size()) {
                            break;
                        }
                    }
                }
                if (seen_count == seen.size()) {
                    break;
                }
            }
            st.unique_rgb = seen_count;
        }
        st.w = w;
        st.h = h;
        st.pitch = image.info.pitch;
        st.bpp = bpp;
        if (bpp >= 4) {
            st.rgba.assign(px, px + download_size);
            for (u64 off = 3; off < st.rgba.size(); off += bpp) {
                st.rgba[off] = 0xff;
            }
        }
        st.ok = true;
        return st;
    };

    // Pick the final composite: prefer a full-resolution target whose RGB (not just alpha/single-channel)
    // is populated -- that is the displayed frame. Intermediate passes are small or single-channel.
    ImageId best{};
    u64 best_score = 0;
    RtStats best_stats{};
    u32 order = 0;
    FILE* candidate_meta =
        std::fopen("/data/user/0/app.lsx4.android/files/lsx4-home/live-rt-candidates.txt",
                   "wb");
    for (const ImageId id : candidates) {
        ++order;
        const RtStats st = readback(id);
        if (!st.ok) {
            continue;
        }
        const u32 rgb_channels = (st.nzc[0] > 0) + (st.nzc[1] > 0) + (st.nzc[2] > 0);
        const u64 area = static_cast<u64>(st.w) * st.h;
        const bool full_res = st.w >= 1280 && st.h >= 720;
        const u64 rgb_bytes = st.nzc[0] + st.nzc[1] + st.nzc[2];
        // score: full-resolution RGB targets dominate. Among them, prefer real composite content over
        // flat clear/background targets using sampled RGB diversity; if still tied, choose the later
        // bind in cb0 order because final composite passes are normally submitted after background.
        const u64 score = (full_res ? (1ull << 62) : 0) |
                          (static_cast<u64>(rgb_channels) << 58) |
                          (static_cast<u64>(st.unique_rgb) << 48) |
                          std::min<u64>(rgb_bytes, (1ull << 48) - 1);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_RT_READBACK] cand extent=%ux%u rgbCh=%u nzR=%llu nzG=%llu "
                            "nzB=%llu nzA=%llu uniqueRGB=%u order=%u id=%u midpx=%02x%02x%02x%02x "
                            "score=%llu",
                            st.w, st.h, rgb_channels, (unsigned long long)st.nzc[0],
                            (unsigned long long)st.nzc[1], (unsigned long long)st.nzc[2],
                            (unsigned long long)st.nzc[3], st.unique_rgb, order, id.index,
                            st.mid[0], st.mid[1], st.mid[2], st.mid[3],
                            (unsigned long long)score);
        if (st.bpp >= 4 && !st.rgba.empty()) {
            char path[192]{};
            std::snprintf(path, sizeof(path),
                          "/data/user/0/app.lsx4.android/files/lsx4-home/"
                          "live-rt-cand-%02u-id%u.ppm",
                          order, id.index);
            ExecutorDumpRtPpm(path, st.rgba.data(), st.w, st.h, st.pitch * st.bpp, st.bpp);
        }
        if (candidate_meta) {
            std::fprintf(candidate_meta,
                         "order=%u id=%u extent=%ux%u pitch=%u bpp=%u rgbCh=%u "
                         "nzR=%llu nzG=%llu nzB=%llu nzA=%llu uniqueRGB=%u "
                         "midpx=%02x%02x%02x%02x score=%llu\n",
                         order, id.index, st.w, st.h, st.pitch, st.bpp, rgb_channels,
                         (unsigned long long)st.nzc[0], (unsigned long long)st.nzc[1],
                         (unsigned long long)st.nzc[2], (unsigned long long)st.nzc[3],
                         st.unique_rgb, st.mid[0], st.mid[1], st.mid[2], st.mid[3],
                         (unsigned long long)score);
            std::fflush(candidate_meta);
        }
        if (score >= best_score) {
            best_score = score;
            best = id;
            best_stats = st;
        }
    }
    if (candidate_meta) {
        std::fclose(candidate_meta);
    }
    if (best) {
        // Present/keep the chosen composite target (overrides the last-bound fallback).
        executor_replay_color_target = best;
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_RT_READBACK] CHOSEN extent=%ux%u uniqueRGB=%u "
                            "midpx=%02x%02x%02x%02x",
                            best_stats.w, best_stats.h, best_stats.unique_rgb, best_stats.mid[0],
                            best_stats.mid[1], best_stats.mid[2], best_stats.mid[3]);
        if (best_stats.bpp >= 4 && !best_stats.rgba.empty()) {
            ExecutorDumpRtPpm(
                "/data/user/0/app.lsx4.android/files/lsx4-home/live-rt-chosen.ppm",
                best_stats.rgba.data(), best_stats.w, best_stats.h,
                best_stats.pitch * best_stats.bpp, best_stats.bpp);
        }
        if (std::getenv("EXECUTOR_LIVE_CPU_PRESENT_RT") != nullptr && best_stats.bpp >= 4 &&
            !best_stats.rgba.empty()) {
            const u32 stride = best_stats.pitch * best_stats.bpp;
            const int present_rc = executor_lsx4_runtime_present_guest_frame(
                best_stats.rgba.data(), best_stats.w, best_stats.h, stride, 1u, 1u);
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_RT_CPU_PRESENT] rc=%d extent=%ux%u stride=%u "
                                "bytes=%zu uniqueRGB=%u midpx=%02x%02x%02x%02x",
                                present_rc, best_stats.w, best_stats.h, stride,
                                best_stats.rgba.size(), best_stats.unique_rgb, best_stats.mid[0],
                                best_stats.mid[1], best_stats.mid[2], best_stats.mid[3]);
        }
    } else {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_RT_READBACK] no_valid_candidate count=%zu",
                            candidates.size());
    }
    {
        std::scoped_lock lk{executor_replay_candidates_mutex};
        executor_replay_color_candidates.clear();
    }
#endif
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    std::scoped_lock lock{mutex};
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
        }
    });
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    std::scoped_lock lock{mutex};
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            return;
        }
        // Ensure image is reuploaded when accessed again.
        image.flags |= ImageFlagBits::GpuDirty;
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lk{mutex};

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        const auto new_image_id =
            slot_images.insert(instance, scheduler, blit_helper, slot_image_views, new_info);
        RegisterImage(new_image_id);

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = cache_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;

        if (cache_image.info.num_samples == 1 && new_info.num_samples == 1) {
            // Perform depth<->color copy using the intermediate copy buffer.
            if (instance.IsMaintenance8Supported()) {
                new_image.CopyImage(cache_image);
            } else {
                const auto& copy_buffer = buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
                new_image.CopyImageWithBuffer(cache_image, copy_buffer.Handle(), 0);
            }
        } else if (cache_image.info.num_samples == 1 && new_info.props.is_depth &&
                   new_info.num_samples > 1) {
            // Perform a rendering pass to transfer the channels of source as samples in dest.
            cache_image.Transit(vk::ImageLayout::eShaderReadOnlyOptimal,
                                vk::AccessFlagBits2::eShaderRead, {});
            new_image.Transit(vk::ImageLayout::eDepthAttachmentOptimal,
                              vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {});
            blit_helper.ReinterpretColorAsMsDepth(
                new_info.size.width, new_info.size.height, new_info.num_samples,
                cache_image.info.pixel_format, new_info.pixel_format, cache_image.GetImage(),
                new_image.GetImage());
        } else {
            LOG_WARNING(Render_Vulkan, "Unimplemented depth overlap copy");
        }

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        if (auto mip = image_info.MipOf(cache_image.info); mip >= 0) {
            if (auto slice = image_info.SliceOf(cache_image.info, mip); slice >= 0) {
                return {cache_image_id, mip, slice};
            }
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        if (auto mip = cache_image.info.MipOf(image_info); mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    merged_image.CopyMip(cache_image, mip, slice);
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    const auto new_image_id =
        slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    new_image.CopyImage(src_image);

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);

    TrackImage(new_image_id);
    new_image.flags &= ~ImageFlagBits::Dirty;
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;

    if (info.guest_address == 0) [[unlikely]] {
        return GetNullImage(info.pixel_format);
    }

    std::scoped_lock lock{mutex};
    ImageIds image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        image_id = slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
        RegisterImage(image_id);
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    ImageIds image_ids;
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (Config::readbackLinearImages() && !image.info.props.is_tiled &&
            image.info.guest_address != 0) {
            download_images.emplace(image_id);
        }
    }
    UpdateImage(image_id);
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (Config::readbackLinearImages() && !image.info.props.is_tiled) {
        download_images.emplace(image_id);
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
#ifdef __ANDROID__
#define FDT_MARK(s) do { if (Libraries::GnmDriver::ExecutorReplayActive()) \
    __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_FDT] " s); } while(0)
#else
#define FDT_MARK(s) ((void)0)
#endif
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
#ifdef __ANDROID__
    if (Libraries::GnmDriver::ExecutorReplayActive())
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_FDT] before_updateimage depthAddr=0x%llx size=%u reloc=%d "
                            "stencilAddr=0x%llx htile=0x%llx",
                            (unsigned long long)image.info.guest_address, image.info.guest_size,
                            Libraries::GnmDriver::ExecutorReplayIsRelocated(image.info.guest_address) ? 1
                                                                                                      : 0,
                            (unsigned long long)desc.info.stencil_addr,
                            (unsigned long long)desc.info.meta_info.htile_addr);
#endif
    // EXECUTOR replay: the depth buffer is a render-target OUTPUT (the z-prepass writes it), and its
    // initial guest contents are not captured (depthAddr reloc=0). UpdateImage->RefreshImage would
    // TrackImage/de-tile that un-relocated guest address and crash. Skip it: the depth image (created by
    // FindImage) is used as a cleared attachment, the z-prepass populates it, the color draws test it.
    if (!Libraries::GnmDriver::ExecutorReplayActive()) {
        UpdateImage(image_id);
    }
    FDT_MARK("after_updateimage");

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaDataInfo::Type::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    FDT_MARK("before_stencil");
    // EXECUTOR replay: stencil_addr is likewise an un-captured/un-relocated guest address; creating a
    // stencil image from it would track/read dead memory. Skip stencil linkage for the first-pixel depth.
    if (desc.info.stencil_addr != 0 && !Libraries::GnmDriver::ExecutorReplayActive()) {
        FDT_MARK("stencil_path");
        ImageId stencil_id{};
        ForEachImageInRegion(desc.info.stencil_addr, desc.info.stencil_size,
                             [&](ImageId image_id, Image& image) {
                                 if (image.info.guest_address == desc.info.stencil_addr) {
                                     stencil_id = image_id;
                                 }
                             });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id =
                slot_images.insert(instance, scheduler, blit_helper, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, image.image_uid);
    }

    FDT_MARK("before_findview");
    auto& v = image.FindView(desc.view_info, false);
    FDT_MARK("after_findview");
    return v;
}

void TextureCache::RefreshImage(Image& image) {
#ifdef __ANDROID__
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        const u32* src = reinterpret_cast<const u32*>(image.info.guest_address);
        u32 s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        if (Libraries::GnmDriver::ExecutorReplayIsRelocated(image.info.guest_address)) {
            s0 = src[0]; s1 = src[1]; s2 = src[2]; s3 = src[3];
        }
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_RI] enter dirty=%d depth=%d addr=0x%llx size=%u src=%08x %08x %08x %08x",
                            (int)True(image.flags & ImageFlagBits::Dirty), image.info.props.is_depth ? 1 : 0,
                            (unsigned long long)image.info.guest_address, image.info.guest_size, s0, s1, s2,
                            s3);
    }
#endif
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }
#ifdef __ANDROID__
    // EXECUTOR multi-submit (Codex): if this image was rendered by an earlier replay submit (it is a
    // render target / GpuModified), do NOT re-upload it from the zero CPU backing -- that would erase the
    // GPU-produced content that a later submit samples as a texture (the RT-as-texture composite). Keep
    // the GPU content; just clear the Dirty flag so it isn't retried.
    if (Libraries::GnmDriver::ExecutorReplayActive() &&
        (image.usage.render_target || image.usage.depth_target ||
         True(image.flags & ImageFlagBits::GpuModified))) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }
#endif
    // EXECUTOR: the RefreshImage replay-skip crutch is removed. The texture source memory is captured on
    // the PC writer (BindTextures usage=4 range) and the T# descriptor base is relocated on Android, so
    // image.info.guest_address now points into the rebased arena and ObtainBufferForImage/DetileImage
    // read real, captured texture bytes -> the sampled texture is correct and the color draw produces
    // non-zero pixels. (If a texture range is still uncaptured, ExecutorReplayCheckAddr logs RELOC_MISS.)
#ifdef __ANDROID__
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_REFRESH_IMG] guest_addr=0x%llx guest_size=%llu tiled=%d "
                            "w=%u h=%u mips=%u layers=%u relocHit=%d",
                            (unsigned long long)image.info.guest_address,
                            (unsigned long long)image.info.guest_size, image.info.props.is_tiled ? 1 : 0,
                            image.info.size.width, image.info.size.height, image.info.resources.levels,
                            image.info.resources.layers,
                            Libraries::GnmDriver::ExecutorReplayIsRelocated(image.info.guest_address) ? 1
                                                                                                     : 0);
        Libraries::GnmDriver::ExecutorReplayCheckAddr(4u, image.info.guest_address,
                                                      image.info.guest_size);
    }
#endif

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));
        // Hash the same byte footprint as the image format.  The old compact expression parsed the
        // conditional as `(3 + is_block) ? 4 : 0`, so every format was shifted by four: ordinary
        // images hashed only half of their bytes while block-compressed images hashed far too many.
        // Besides reading outside the sampled pixel window this can leave a changed texture falsely
        // classified as clean.  Keep this byte-for-byte with the desktop texture-cache contract.
        const u32 sample_width = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 sample_height = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = sample_width * sample_height * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        if (is_gpu_modified && !is_gpu_dirty) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const u64 hash = XXH3_64bits(addr + mip_offset, mip_size);
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    // EXECUTOR PC-writer capture: record the EXACT source range the detiler is about to read. This is
    // the tiled backing footprint (image.info.guest_size for a tiled image = GetColorSliceSize*layers),
    // which is larger than the logical size captured by the BindTextures hook -- so capturing here gives
    // replay the full range and avoids the de-tile OOB read. usage=4 (texture). No-op on Android replay.
    if (Libraries::GnmDriver::ExecutorGnmCaptureActive()) {
        Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(4u, image.info.guest_address,
                                                               image.info.guest_size);
    }

#ifdef __ANDROID__
    const bool exec_ri_trace = Libraries::GnmDriver::ExecutorReplayActive();
    if (exec_ri_trace)
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_RI] before_obtain copies=%zu",
                            image_copies.size());
#endif
    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
#ifdef __ANDROID__
    if (exec_ri_trace)
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_RI] after_obtain in_offset=%u",
                            (unsigned)in_offset);
#endif
    // Tiled images are consumed by TileManager's compute shader. Linear images bypass the detiler
    // and the same buffer is consumed directly by Image::Upload's copy command. Track the actual
    // consumer: treating every source as compute leaves linear texture uploads ordered against the
    // wrong stage and can expose stale/missing levels on strict mobile drivers.
    const auto source_access = image.info.props.is_tiled
                                   ? vk::AccessFlagBits2::eShaderRead
                                   : vk::AccessFlagBits2::eTransferRead;
    const auto source_stage = image.info.props.is_tiled
                                  ? vk::PipelineStageFlagBits2::eComputeShader
                                  : vk::PipelineStageFlagBits2::eTransfer;
    if (auto barrier = in_buffer->GetBarrier(source_access, source_stage)) {
        scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier.value(),
        });
    }

    const auto [buffer, offset] =
        tile_manager.DetileImage(in_buffer->Handle(), in_offset, image.info);
#ifdef __ANDROID__
    if (exec_ri_trace)
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_RI] after_detile");
#endif
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    image.Upload(image_copies, buffer, offset);

#ifdef __ANDROID__
    // EXECUTOR replay diagnostic (Codex): read back the de-tiled VkImage to split "de-tile/upload broke"
    // from "view/sampler/UV/shader" -- if this is zero the de-tile produced nothing; if non-zero the
    // sampled-black is downstream. Only for the big sampled textures (not RTs), one-shot per image.
    if (Libraries::GnmDriver::ExecutorReplayActive() && !image.info.props.is_depth &&
        image.info.size.width >= 64 && image.info.guest_size <= (64ull << 20)) {
        auto& dl = buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
        const u32 bpp = std::max<u32>(image.info.num_bits / 8u, 1u);
        const u64 dsize = static_cast<u64>(image.info.pitch) * image.info.size.height * bpp;
        if (dsize > 0 && dsize <= (64ull << 20)) {
            const auto [dptr, doff] = dl.Map(dsize);
            dl.Commit();
            const vk::BufferImageCopy region{
                .bufferOffset = doff,
                .bufferRowLength = image.info.pitch,
                .bufferImageHeight = image.info.size.height,
                .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .imageOffset = {0, 0, 0},
                .imageExtent = {image.info.size.width, image.info.size.height, 1}};
            scheduler.EndRendering();
            const auto cb = scheduler.CommandBuffer();
            image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
            cb.copyImageToBuffer(image.GetImage(), vk::ImageLayout::eTransferSrcOptimal, dl.Handle(),
                                 region);
            scheduler.Finish(Vulkan::Scheduler::ExecutorFinishReason::TextureProbe);
            const u8* px = reinterpret_cast<const u8*>(dptr);
            u64 nz = 0;
            const u64 scan = std::min<u64>(dsize, 4ull << 20);
            for (u64 i = 0; i < scan; ++i)
                if (px[i]) ++nz;
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_TEX_READBACK] addr=0x%llx %ux%u bits=%u nonzeroBytes=%llu "
                                "first=%08x %08x",
                                (unsigned long long)image.info.guest_address, image.info.size.width,
                                image.info.size.height, image.info.num_bits, (unsigned long long)nz,
                                ((const u32*)px)[0], ((const u32*)px)[1]);
        }
    }
#endif
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base) {
    const u64 hash = XXH3_64bits(&sampler, sizeof(sampler));

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] = samplers.try_emplace(hash, instance, sampler, border_color_base);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }
    return it->second.Handle();
}

void TextureCache::RegisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    ForEachPage(image.info.guest_address, image.info.guest_size,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    lru_cache.Free(image.lru_id);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers<1>(image_begin, image.info.guest_size);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers<1>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers<1>(addr, size);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        tracker.UpdatePageWatchers<false>(addr, size);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(image_begin, size);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(addr, size);
}

void TextureCache::GarbageCollectImages(const std::optional<size_t> sampled_device_memory) {
    if (sampled_device_memory) {
        total_used_memory = *sampled_device_memory;
    } else if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        auto& image = slot_images[image_id];
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        if (download) {
            DownloadImageMemory(image_id);
        }
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::GarbageCollectSamplers() {
    std::scoped_lock lock{samplers_mutex};
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }

    bool pressured = false;
    bool aggressive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;
    const auto configure = [&](bool allow_aggressive) {
        total_used_samplers = samplers.size();
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggressive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggressive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggressive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        const auto sampler_it = samplers.find(hash);
        if (sampler_it == samplers.end()) {
            return false;
        }
        const size_t lru_id = sampler_it->second.lru_id;
        samplers.erase(sampler_it);
        sampler_lru_cache.Free(lru_id);
        return false;
    };

    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    if (samplers.size() >= critical_gc_samplers) {
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector(const std::optional<size_t> sampled_device_memory) {
    SCOPE_EXIT {
        ++gc_tick;
    };
    GarbageCollectImages(sampled_device_memory);
    GarbageCollectSamplers();
}

void TextureCache::TouchImage(const Image& image) {
    lru_cache.Touch(image.lru_id, gc_tick);
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");
    download_images.erase(image_id);
#ifdef __ANDROID__
    {
        // A draw target may retire between its last write and the next guest flip. Never let a
        // recycled SlotId turn that stale candidate into an unrelated clear/auxiliary surface.
        std::scoped_lock lk{executor_replay_candidates_mutex};
        std::erase(executor_live_drawn_color_candidates, image_id);
        u32 expected = image_id.index;
        executor_live_last_color_target.compare_exchange_strong(
            expected, 0, std::memory_order_relaxed, std::memory_order_relaxed);
    }
#endif

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

} // namespace VideoCore
