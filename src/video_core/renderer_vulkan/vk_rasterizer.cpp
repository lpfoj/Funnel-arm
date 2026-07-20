// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/config.h"
#include "common/content_fingerprint.h"
#include "common/debug.h"
#include "core/memory.h"
#include "core/libraries/videoout/video_out.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/render_wave_trace.h"
#include "core/libraries/gnmdriver/gnmdriver.h"
#ifdef __ANDROID__
#include <cstdlib>
#ifdef __ANDROID__
#include <unistd.h>
#endif
#include <android/log.h>
#endif
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#include <array>
#include <cstring>
#include <memory>
#include <span>

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

bool Rasterizer::IsDeviceTerminal() const noexcept {
    return scheduler.IsDeviceLost();
}

void Rasterizer::MarkDeviceTerminal(vk::Result result, const char* where) noexcept {
    scheduler.MarkTerminal(result, where);
}

#ifdef __ANDROID__
static u64 ExecutorVkPipelineHandle(vk::Pipeline pipeline) noexcept {
    return reinterpret_cast<u64>(static_cast<VkPipeline>(pipeline));
}

// Qualcomm's proprietary Android driver advertises EDS1/2 but can re-latch the all-disabled
// static VkPipelineDepthStencilStateCreateInfo when a graphics pipeline is bound. Vulkan permits
// dynamic state to be recorded before bind (the upstream PC order), yet on Adreno that legal order
// produced populated MRTs with an entirely clear D32 attachment. A marker-gated full-pass oracle
// proved that state recorded after bind immediately restores spatially varying depth.
//
// The production workaround therefore finishes every allocation which can rotate the command
// buffer first, then records pipeline + vertex/index + the complete dynamic state contiguously
// immediately before the draw. A newly started dynamic-rendering scope is treated as a host-state
// boundary, but a same-pipeline hit inside that scope remains cached. Read-only buffer transitions
// are coalesced before reaching this path, avoiding the repeated End/BeginRendering churn which
// exposed the Adreno state-loss bug without recording a dozen redundant EDS commands per draw.
static bool ExecutorNeedsQualcommPostBindDynamicState(const Instance& instance) noexcept {
    static const bool disabled_for_bisection =
        ::access("/data/data/app.lsx4.android/files/lsx4-home/"
                 "run-bisect-disable-qualcomm-post-bind",
                 F_OK) == 0;
    return !disabled_for_bisection &&
           instance.GetDriverID() == vk::DriverId::eQualcommProprietary &&
           !instance.IsExtendedDynamicState3Supported();
}

static void ExecutorLogQualcommPostBindDynamicState() noexcept {
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set(std::memory_order_relaxed)) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_QUALCOMM_PIPELINE_ORDER] enabled=1 driver=qualcomm noEds3=1 "
            "order=prepare_all_then_pipeline_vertex_index_dynamic scope_cached "
            "direct_and_indirect");
    }
}

void Rasterizer::ExecutorBindGuestGraphicsPipeline(const GraphicsPipeline* pipeline,
                                                   const vk::CommandBuffer cmdbuf) {
    if (!ExecutorNeedsQualcommPostBindDynamicState(instance)) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
        return;
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    const auto handle = pipeline->Handle();
    const u64 epoch = dynamic_state.InvalidationEpoch();
    const u64 rendering_epoch = scheduler.RenderingEpoch();
    const bool needs_bind = executor_guest_graphics_cmdbuf != cmdbuf ||
                            executor_guest_graphics_pipeline != handle ||
                            executor_guest_dynamic_epoch != epoch ||
                            executor_guest_rendering_epoch != rendering_epoch;
    if (!needs_bind) {
        return;
    }

    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
    dynamic_state.Invalidate();
    executor_guest_graphics_cmdbuf = cmdbuf;
    executor_guest_graphics_pipeline = handle;
    executor_guest_dynamic_epoch = dynamic_state.InvalidationEpoch();
    executor_guest_rendering_epoch = rendering_epoch;
    ExecutorLogQualcommPostBindDynamicState();
}

struct ExecutorDepthWriteOracleRuntime {
    std::atomic<u64> epoch{0};
    std::atomic<u64> attachment_signature{0};
};

static ExecutorDepthWriteOracleRuntime& ExecutorGetDepthWriteOracleRuntime() noexcept {
    static ExecutorDepthWriteOracleRuntime runtime{};
    return runtime;
}

static bool ExecutorDepthWriteOracleMarkerPresent() noexcept {
    // Check at the draw rather than process start so a live capture can arm the destructive oracle
    // only after the title reaches the scene under investigation.
    return ::access("/data/data/app.lsx4.android/files/lsx4-home/"
                    "run-depth-write-oracle",
                    F_OK) == 0;
}

// Claim one attachment span per late-frame epoch.  The sentinel is cleared once, but the forced
// post-bind state remains active for every draw in the same pass: the first selected object can be
// outside the camera while later objects still provide the decisive coverage.
static bool ExecutorClaimDepthWriteOracle(u64 attachment_signature) noexcept {
    if (!ExecutorDepthWriteOracleMarkerPresent() || attachment_signature == 0) {
        return false;
    }
    const u64 epoch = AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch();
    if (epoch == 0) {
        return false;
    }
    auto& runtime = ExecutorGetDepthWriteOracleRuntime();
    u64 previous = runtime.epoch.load(std::memory_order_acquire);
    while (previous != epoch) {
        if (runtime.epoch.compare_exchange_weak(previous, epoch, std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
            runtime.attachment_signature.store(attachment_signature, std::memory_order_release);
            return true;
        }
    }
    return false;
}

static bool ExecutorDepthWriteOracleActive(u64 attachment_signature) noexcept {
    if (!AmdGpu::RenderWaveTrace::LateFrameCaptureActive() ||
        !ExecutorDepthWriteOracleMarkerPresent() || attachment_signature == 0) {
        return false;
    }
    const auto& runtime = ExecutorGetDepthWriteOracleRuntime();
    const u64 epoch = AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch();
    return epoch != 0 && runtime.epoch.load(std::memory_order_acquire) == epoch &&
           runtime.attachment_signature.load(std::memory_order_acquire) == attachment_signature;
}

struct ExecutorVkAttachmentSummary {
    u64 signature{};
    VAddr depth_address{};
    u32 color_count{};
};

// Hash the complete guest attachment set rather than only CB0.  Deferred renderers often keep CB0
// stable while changing the other G-buffer surfaces, so CB0 alone cannot identify render-pass
// transitions.  This is metadata-only and runs only for commands already entering the journal.
template <typename ColorTargets, typename DepthTarget>
static ExecutorVkAttachmentSummary ExecutorSummarizeAttachments(
    const ColorTargets& color_targets, const DepthTarget& depth_target) noexcept {
    ExecutorVkAttachmentSummary summary{};
    Common::ContentFingerprint64 fingerprint{Common::FingerprintDomain::AttachmentSet};
    bool any_attachment = false;
    auto mix = [&](const u64 value) {
        fingerprint.UpdateLittleEndian(value);
    };
    for (u32 index = 0; index < color_targets.size(); ++index) {
        const auto& [image_id, desc] = color_targets[index];
        if (!image_id) {
            continue;
        }
        any_attachment = true;
        ++summary.color_count;
        const auto& info = desc.info;
        mix(0xc010000000000000ull | index);
        mix(info.guest_address);
        mix(info.guest_size);
        mix((static_cast<u64>(info.size.width) << 32) | info.size.height);
        mix(static_cast<u64>(info.pixel_format));
    }
    if (depth_target.first) {
        any_attachment = true;
        const auto& info = depth_target.second.info;
        summary.depth_address = info.guest_address;
        mix(0xd320000000000000ull);
        mix(info.guest_address);
        mix(info.guest_size);
        mix((static_cast<u64>(info.size.width) << 32) | info.size.height);
        mix(static_cast<u64>(info.pixel_format));
    }
    summary.signature = any_attachment ? fingerprint.Finish() : 0;
    return summary;
}

static bool ExecutorAlignedSsboScratchEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("EXECUTOR_ALIGNED_SSBO_SCRATCH");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

static bool ExecutorTracePm4Enabled() noexcept {
    static const bool enabled = std::getenv("EXECUTOR_TRACE_PM4") != nullptr;
    return enabled;
}

static bool ExecutorTraceLiveWideEnabled() noexcept {
    static const bool enabled = std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr;
    return enabled;
}

static bool ExecutorTraceLivePresentResourcesEnabled() noexcept {
    // Resource dumps are diagnostics. EXECUTOR_LIVE_PRESENT_RT is the normal production bridge and
    // must not implicitly enable descriptor logging or memory sampling on every shader stage.
    static const bool enabled =
        std::getenv("EXECUTOR_TRACE_LIVE_PRESENT_RT") != nullptr;
    return enabled;
}

struct ExecutorGuestRangeFingerprint {
    u64 digest{};
    u64 clamped_size{};
    bool readable{};
};

// A selected-draw fingerprint is deliberately bounded to the first/last 128 bytes.  It is enough
// to compare the exact guest VB/IB contract with PC without turning a world draw into a multi-MB
// CPU readback.  `gpu_modified` is logged separately; a digest from such a range describes the CPU
// shadow only and must not be mistaken for authoritative GPU contents.
static ExecutorGuestRangeFingerprint ExecutorFingerprintGuestRange(Core::MemoryManager* memory,
                                                                    VAddr address, u64 size) {
    ExecutorGuestRangeFingerprint result{};
    if (address == 0 || size == 0 || !memory->IsValidMapping(address, 1)) {
        return result;
    }
    result.clamped_size = memory->ClampRangeSize(address, size);
    if (result.clamped_size == 0) {
        return result;
    }
    result.readable = true;
    Common::ContentFingerprint64 fingerprint{Common::FingerprintDomain::GuestRangeSample};
    const auto hash_bytes = [&](const u8* bytes, const u64 count) {
        fingerprint.Update(
            std::span<const std::uint8_t>{bytes, static_cast<std::size_t>(count)});
    };
    const u64 head_size = std::min<u64>(result.clamped_size, 128);
    hash_bytes(reinterpret_cast<const u8*>(address), head_size);
    if (result.clamped_size > head_size) {
        const u64 tail_size = std::min<u64>(result.clamped_size - head_size, 128);
        hash_bytes(reinterpret_cast<const u8*>(address + result.clamped_size - tail_size),
                   tail_size);
    }
    // Make equal sampled bytes from differently sized descriptors distinguishable.
    fingerprint.UpdateLittleEndian(result.clamped_size);
    result.digest = fingerprint.Finish();
    return result;
}

static void ExecutorTraceDrawnColorTarget(
    const VideoCore::TextureCache::ImageDesc* color_target) {
    static const bool enabled = std::getenv("EXECUTOR_TRACE_VIDEOOUT_SOURCE") != nullptr;
    if (!enabled) {
        return;
    }

    const auto* info = color_target ? &color_target->info : nullptr;
    const u64 cb_address = info ? info->guest_address : 0;
    const u64 cb_size = info ? info->guest_size : 0;
    Libraries::VideoOut::ExecutorVideoOutBufferSnapshot videoout{};
    (void)Libraries::VideoOut::ExecutorGetVideoOutBufferSnapshot(cb_address, cb_size, &videoout);

    const u64 vo_address = videoout.valid ? videoout.address : 0;
    const u64 meta = info ? (static_cast<u64>(info->size.width) << 32) |
                                static_cast<u64>(info->size.height)
                          : 0;

    static std::atomic<u64> seen_draws{0};
    static std::atomic<u64> emitted{0};
    static std::atomic<u64> last_cb_address{~0ull};
    static std::atomic<u64> last_vo_address{~0ull};
    static std::atomic<u64> last_extent{~0ull};
    const u64 draw_sequence = seen_draws.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool changed = last_cb_address.exchange(cb_address, std::memory_order_relaxed) !=
                             cb_address ||
                         last_vo_address.exchange(vo_address, std::memory_order_relaxed) !=
                             vo_address ||
                         last_extent.exchange(meta, std::memory_order_relaxed) != meta;
    if (draw_sequence > 4 && !changed) {
        return;
    }
    const u64 log_sequence = emitted.fetch_add(1, std::memory_order_relaxed);
    if (log_sequence >= 32) {
        return;
    }

    const u64 cb_end = cb_address + cb_size;
    const u64 vo_end = vo_address + videoout.size_bytes;
    const bool exact = cb_address != 0 && cb_address == vo_address;
    const bool overlaps = cb_address != 0 && vo_address != 0 && cb_address < vo_end &&
                          vo_address < cb_end;
    const s64 delta = static_cast<s64>(cb_address - vo_address);
    const std::string cb_format =
        info ? vk::to_string(info->pixel_format) : std::string{"none"};

    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_DRAW_VIDEOOUT_MATCH] seq=%llu draw=%llu transition=%u "
        "cb0=0x%llx cbBytes=%llu cbExtent=%ux%u cbPitch=%u cbFormat=%s cbTiled=%u "
        "voValid=%u voCurrent=%d voEffective=%d voExactAny=%d voOverlapAny=%d "
        "voRegistered=%u voAddr=0x%llx "
        "voBytes=%llu voExtent=%ux%u voPitch=%u voFormat=0x%x voTiling=%u "
        "exact=%u overlap=%u delta=%lld",
        static_cast<unsigned long long>(log_sequence + 1),
        static_cast<unsigned long long>(draw_sequence), changed ? 1u : 0u,
        static_cast<unsigned long long>(cb_address),
        static_cast<unsigned long long>(cb_size), info ? info->size.width : 0,
        info ? info->size.height : 0, info ? info->pitch : 0, cb_format.c_str(),
        info && info->props.is_tiled ? 1u : 0u, videoout.valid ? 1u : 0u,
        videoout.requested_index, videoout.effective_index, videoout.exact_address_index,
        videoout.overlapping_address_index, videoout.registered_buffers,
        static_cast<unsigned long long>(vo_address),
        static_cast<unsigned long long>(videoout.size_bytes), videoout.width, videoout.height,
        videoout.pitch_in_pixel, videoout.pixel_format, videoout.tiling_mode, exact ? 1u : 0u,
        overlaps ? 1u : 0u, static_cast<long long>(delta));
}

static u64 ExecutorCountNonzeroSparse(Core::MemoryManager* memory, VAddr address, u64 size) {
    if (!memory || address == 0 || size == 0) {
        return 0;
    }
    std::array<u8, 4096> buffer{};
    u64 nonzero = 0;
    u64 remaining = std::min<u64>(size, 4ull << 20);
    VAddr cursor = address;
    while (remaining) {
        const u64 chunk = std::min<u64>(remaining, buffer.size());
        memory->CopySparseMemory(cursor, buffer.data(), chunk);
        for (u64 i = 0; i < chunk; ++i) {
            nonzero += buffer[i] != 0;
        }
        cursor += chunk;
        remaining -= chunk;
    }
    return nonzero;
}
#endif

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache} {
    if (!Config::nullGpu()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
}

Rasterizer::~Rasterizer() = default;

#ifdef __ANDROID__
void Rasterizer::ExecutorGpuLatePassTransition(u64 next_attachment_signature, u64 next_sequence,
                                               const char* reason) {
    if (!AmdGpu::RenderWaveTrace::LateFrameCaptureActive()) {
        executor_gpu_late_pass_probe = {};
        return;
    }

    const u64 epoch = AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch();
    if (executor_gpu_late_pass_probe.epoch != epoch) {
        executor_gpu_late_pass_probe = {};
        executor_gpu_late_pass_probe.epoch = epoch;
    }
    if (executor_gpu_late_pass_probe.armed &&
        executor_gpu_late_pass_probe.attachment_signature != next_attachment_signature) {
        ExecutorGpuLatePassFlush(next_attachment_signature, next_sequence, reason);
    }
}

void Rasterizer::ExecutorGpuLatePassArm(u64 attachment_signature, u64 selected_sequence,
                                        u64 fragment_hash) {
    if (!AmdGpu::RenderWaveTrace::LateFrameCaptureActive() || attachment_signature == 0) {
        return;
    }

    const u64 epoch = AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch();
    if (executor_gpu_late_pass_probe.epoch != epoch) {
        executor_gpu_late_pass_probe = {};
        executor_gpu_late_pass_probe.epoch = epoch;
    }
    // Selection is unique per attachment signature. Retain the first selected draw as the span's
    // diagnostic identity if a caller nevertheless attempts to arm it more than once.
    if (executor_gpu_late_pass_probe.armed) {
        return;
    }

    auto& pending = executor_gpu_late_pass_probe;
    const auto capture_identity = [&](VideoCore::ImageId image_id, VAddr desc_guest_address,
                                      const VideoCore::SubresourceRange& view_range) {
        ExecutorGpuLatePassProbeState::ImageIdentity identity{};
        if (!image_id) {
            return identity;
        }
        const auto& image = texture_cache.GetImage(image_id);
        identity.image_id = image_id;
        identity.desc_guest_address = desc_guest_address;
        identity.image_guest_address = image.info.guest_address;
        identity.image_uid = image.image_uid;
        identity.backing_identity = reinterpret_cast<uintptr_t>(image.backing);
        identity.image_handle =
            reinterpret_cast<u64>(static_cast<VkImage>(image.GetImage()));
        identity.view_range = view_range;
        return identity;
    };
    pending.attachment_signature = attachment_signature;
    pending.selected_sequence = selected_sequence;
    pending.fragment_hash = fragment_hash;
    pending.color_count = 0;
    for (const auto& color_desc : cb_descs) {
        if (!color_desc.first || pending.color_count >= pending.color_images.size()) {
            continue;
        }
        pending.color_images[pending.color_count++] =
            capture_identity(color_desc.first, color_desc.second.info.guest_address,
                             color_desc.second.view_info.range);
    }
    if (db_desc.first) {
        pending.depth_image =
            capture_identity(db_desc.first, db_desc.second.info.guest_address,
                             db_desc.second.view_info.range);
    }

    const auto& regs = liverpool->regs;
    pending.depth_render_control_raw = std::bit_cast<u32>(regs.depth_render_control);
    pending.depth_control_raw = std::bit_cast<u32>(regs.depth_control);
    pending.depth_clear_enable = regs.depth_render_control.depth_clear_enable;
    pending.depth_write_enable = regs.depth_control.depth_write_enable;
    pending.depth_effective_write =
        regs.depth_control.depth_write_enable && !regs.depth_render_control.depth_clear_enable;
    const auto& dynamic = scheduler.GetDynamicState();
    pending.scheduler_depth_test = dynamic.depth_test_enabled;
    pending.scheduler_depth_write = dynamic.depth_write_enabled;
    pending.scheduler_depth_compare = static_cast<u32>(dynamic.depth_compare_op);
    pending.depth_read_address = regs.depth_buffer.DepthAddress();
    pending.depth_write_address = regs.depth_buffer.DepthWriteAddress();
    pending.stencil_read_address = regs.depth_buffer.StencilAddress();
    pending.stencil_write_address = regs.depth_buffer.StencilWriteAddress();
    pending.htile_address = regs.depth_htile_data_base.GetAddress();
    pending.desc_stencil_address = db_desc.first ? db_desc.second.info.stencil_addr : 0;
    pending.desc_htile_address =
        db_desc.first ? db_desc.second.info.meta_info.htile_addr : 0;
    pending.armed = pending.color_count != 0 || static_cast<bool>(pending.depth_image.image_id);

    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_GPU_PASS_END_ARM] epoch=%llu seq=%llu attach=0x%llx ps=0x%llx "
        "colors=%u depth=%u dbRender=0x%x dbControl=0x%x clear=%u write=%u "
        "effectiveWrite=%u schedTest=%u schedWrite=%u schedCompare=%u "
        "depthRead=0x%llx depthWrite=0x%llx stencilRead=0x%llx stencilWrite=0x%llx "
        "htile=0x%llx descStencil=0x%llx descHtile=0x%llx",
        static_cast<unsigned long long>(pending.epoch),
        static_cast<unsigned long long>(pending.selected_sequence),
        static_cast<unsigned long long>(pending.attachment_signature),
        static_cast<unsigned long long>(pending.fragment_hash), pending.color_count,
        pending.depth_image.image_id ? 1u : 0u, pending.depth_render_control_raw,
        pending.depth_control_raw, pending.depth_clear_enable, pending.depth_write_enable,
        pending.depth_effective_write, pending.scheduler_depth_test,
        pending.scheduler_depth_write, pending.scheduler_depth_compare,
        static_cast<unsigned long long>(pending.depth_read_address),
        static_cast<unsigned long long>(pending.depth_write_address),
        static_cast<unsigned long long>(pending.stencil_read_address),
        static_cast<unsigned long long>(pending.stencil_write_address),
        static_cast<unsigned long long>(pending.htile_address),
        static_cast<unsigned long long>(pending.desc_stencil_address),
        static_cast<unsigned long long>(pending.desc_htile_address));

    const auto log_identity = [&](const char* role, u32 index,
                                  const ExecutorGpuLatePassProbeState::ImageIdentity& identity) {
        if (!identity.image_id) {
            return;
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GPU_PASS_END_IMAGE] phase=arm role=%s index=%u imageId=%u "
            "descGuest=0x%llx imageGuest=0x%llx imageUid=%llu backing=0x%llx "
            "vkImage=0x%llx mip=%u+%u layer=%u+%u",
            role, index, identity.image_id.index,
            static_cast<unsigned long long>(identity.desc_guest_address),
            static_cast<unsigned long long>(identity.image_guest_address),
            static_cast<unsigned long long>(identity.image_uid),
            static_cast<unsigned long long>(identity.backing_identity),
            static_cast<unsigned long long>(identity.image_handle), identity.view_range.base.level,
            identity.view_range.extent.levels, identity.view_range.base.layer,
            identity.view_range.extent.layers);
    };
    for (u32 index = 0; index < pending.color_count; ++index) {
        log_identity("color", index, pending.color_images[index]);
    }
    log_identity("depth", 0, pending.depth_image);
}

void Rasterizer::ExecutorGpuLatePassFlush(u64 next_attachment_signature, u64 next_sequence,
                                          const char* reason) {
    if (!executor_gpu_late_pass_probe.armed) {
        return;
    }

    // Clear the member before readback. ExecutorGpuContentProbe synchronously ends/submits the
    // current rendering scope; keeping no armed state across that re-entrant scheduler work makes
    // an accidental nested boundary harmless.
    const auto completed = executor_gpu_late_pass_probe;
    executor_gpu_late_pass_probe = {};
    executor_gpu_late_pass_probe.epoch = completed.epoch;

    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_GPU_PASS_END_FLUSH] epoch=%llu selectedSeq=%llu nextSeq=%llu "
        "attach=0x%llx nextAttach=0x%llx ps=0x%llx colors=%u depth=%u reason=%s",
        static_cast<unsigned long long>(completed.epoch),
        static_cast<unsigned long long>(completed.selected_sequence),
        static_cast<unsigned long long>(next_sequence),
        static_cast<unsigned long long>(completed.attachment_signature),
        static_cast<unsigned long long>(next_attachment_signature),
        static_cast<unsigned long long>(completed.fragment_hash), completed.color_count,
        completed.depth_image.image_id ? 1u : 0u, reason ? reason : "unknown");

    const auto log_current_identity = [&](const char* role, u32 index,
                                          const ExecutorGpuLatePassProbeState::ImageIdentity& captured) {
        if (!captured.image_id) {
            return;
        }
        const auto& image = texture_cache.GetImage(captured.image_id);
        const u64 backing_identity = reinterpret_cast<uintptr_t>(image.backing);
        const u64 image_handle =
            reinterpret_cast<u64>(static_cast<VkImage>(image.GetImage()));
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GPU_PASS_END_IMAGE] phase=flush role=%s index=%u imageId=%u "
            "capturedUid=%llu currentUid=%llu capturedBacking=0x%llx currentBacking=0x%llx "
            "capturedVkImage=0x%llx currentVkImage=0x%llx same=%u",
            role, index, captured.image_id.index,
            static_cast<unsigned long long>(captured.image_uid),
            static_cast<unsigned long long>(image.image_uid),
            static_cast<unsigned long long>(captured.backing_identity),
            static_cast<unsigned long long>(backing_identity),
            static_cast<unsigned long long>(captured.image_handle),
            static_cast<unsigned long long>(image_handle),
            captured.image_uid == image.image_uid && captured.backing_identity == backing_identity &&
                    captured.image_handle == image_handle
                ? 1u
                : 0u);
    };

    for (u32 index = 0; index < completed.color_count; ++index) {
        log_current_identity("color", index, completed.color_images[index]);
        texture_cache.ExecutorGpuContentProbe(
            completed.color_images[index].image_id,
            VideoCore::TextureCache::ExecutorGpuContentRole::ColorPassEnd,
            static_cast<u32>(Shader::LogicalStage::Fragment), completed.fragment_hash,
            completed.selected_sequence);
    }
    if (completed.depth_image.image_id) {
        log_current_identity("depth", 0, completed.depth_image);
        texture_cache.ExecutorGpuContentProbe(
            completed.depth_image.image_id,
            VideoCore::TextureCache::ExecutorGpuContentRole::DepthPassEnd,
            static_cast<u32>(Shader::LogicalStage::Fragment), completed.fragment_hash,
            completed.selected_sequence);
    }
}
#endif

void Rasterizer::CpSync() {
#ifdef __ANDROID__
    ++executor_completion_work_serial;
#endif
    scheduler.EndRendering();
    auto cmdbuf = scheduler.CommandBuffer();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

void Rasterizer::CompletionWaitBarrier() {
    // A real EOP -> WAIT_REG_MEM round-trip ends a command buffer and waits for full completion.
    // When those packets are folded into one host command buffer, recreate the device-memory
    // dependency explicitly. Ending dynamic rendering also preserves the old submission boundary's
    // attachment-store/load behavior, which is required for depth and transient target reuse.
    scheduler.EndRendering();
#ifdef __ANDROID__
    ++executor_completion_barrier_calls;
    const u64 current_tick = scheduler.CurrentTick();
    const u64 current_epoch = scheduler.CommandRecordingEpoch();
    const bool barrier_is_redundant =
        executor_completion_barrier_valid &&
        executor_completion_barrier_tick == current_tick &&
        executor_completion_barrier_epoch == current_epoch &&
        executor_completion_barrier_work_serial == executor_completion_work_serial;
    if (barrier_is_redundant) {
        ++executor_completion_barrier_elided;
        if ((executor_completion_barrier_calls & 4095u) == 0u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_COMPLETION_BARRIER_COALESCE] calls=%llu emitted=%llu elided=%llu "
                "elidePct=%.2f tick=%llu epoch=%llu",
                static_cast<unsigned long long>(executor_completion_barrier_calls),
                static_cast<unsigned long long>(executor_completion_barrier_emitted),
                static_cast<unsigned long long>(executor_completion_barrier_elided),
                executor_completion_barrier_calls != 0
                    ? static_cast<double>(executor_completion_barrier_elided) * 100.0 /
                          static_cast<double>(executor_completion_barrier_calls)
                    : 0.0,
                static_cast<unsigned long long>(current_tick),
                static_cast<unsigned long long>(current_epoch));
        }
        return;
    }
#endif
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask =
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
#ifdef __ANDROID__
    u64 barrier_epoch{};
    scheduler.CommandBuffer(barrier_epoch).pipelineBarrier2(vk::DependencyInfo{
#else
    scheduler.CommandBuffer().pipelineBarrier2(vk::DependencyInfo{
#endif
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
#ifdef __ANDROID__
    ++executor_completion_barrier_emitted;
    if (scheduler.CurrentTick() == current_tick) {
        executor_completion_barrier_tick = current_tick;
        executor_completion_barrier_epoch = barrier_epoch;
        executor_completion_barrier_work_serial = executor_completion_work_serial;
        executor_completion_barrier_valid = true;
    } else {
        executor_completion_barrier_valid = false;
    }
    if ((executor_completion_barrier_calls & 4095u) == 0u) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_COMPLETION_BARRIER_COALESCE] calls=%llu emitted=%llu elided=%llu "
            "elidePct=%.2f tick=%llu epoch=%llu",
            static_cast<unsigned long long>(executor_completion_barrier_calls),
            static_cast<unsigned long long>(executor_completion_barrier_emitted),
            static_cast<unsigned long long>(executor_completion_barrier_elided),
            executor_completion_barrier_calls != 0
                ? static_cast<double>(executor_completion_barrier_elided) * 100.0 /
                      static_cast<double>(executor_completion_barrier_calls)
                : 0.0,
            static_cast<unsigned long long>(current_tick),
            static_cast<unsigned long long>(barrier_epoch));
    }
#endif
}

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->regs;
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
#ifdef __ANDROID__
        static const bool kTracePrep =
            ExecutorTracePm4Enabled() ||
            (::access("/data/data/app.lsx4.android/files/lsx4-home/run-trace-draw", F_OK) == 0);
        if (kTracePrep)
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_DRAW_TRACE] prep_cb cb=%u skip=%d colBufValid=%d mask=%u mrt=%d base=0x%llx",
                                cb, (int)skip_cb_binding, (int)(bool)col_buf, target_mask,
                                (int)((key.mrt_mask & (1 << cb)) != 0),
                                (unsigned long long)col_buf.Address());
#endif
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->last_cb_extent[cb];
        std::construct_at(&desc, col_buf, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
        // EXECUTOR (Codex phase-2): record the bound color render target [addr,size] so replay can
        // allocate + rebase a compatible RT (contents become the clear; usage=5 render target).
        if (Libraries::GnmDriver::ExecutorGnmCaptureActive()) {
            Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(
                5u, col_buf.Address(), static_cast<u64>(col_buf.GetColorSliceSize()) * col_buf.NumSlices());
        }
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->last_db_extent;
        auto& [image_id, desc] = db_desc;
        std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                          htile_address, hint);
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
        // EXECUTOR (Codex phase-2): the depth/stencil + htile backing memory must also be captured, or
        // BeginRendering reads an un-relocated depth target on replay and crashes.
        if (Libraries::GnmDriver::ExecutorGnmCaptureActive()) {
            const auto& db = regs.depth_buffer;
            if (db.DepthValid()) {
                Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(5u, db.DepthAddress(),
                                                                       db.GetDepthSliceSize());
            }
            if (db.StencilValid()) {
                Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(5u, db.StencilAddress(),
                                                                       db.GetDepthSliceSize());
            }
            if (htile_address) {
                Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(5u, htile_address, 0x40000u);
            }
        }
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::Info& info,
    const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
#ifdef __ANDROID__
    ++executor_completion_work_serial;
    static const bool trace_actual_draw =
        ExecutorTraceLiveWideEnabled() ||
        ExecutorTracePm4Enabled() ||
        (::access("/data/data/app.lsx4.android/files/lsx4-home/run-trace-draw",
                  F_OK) == 0);
    if (trace_actual_draw) {
        LOG_INFO(Render_Vulkan,
                 "EXECUTOR_VK_ACTUAL_DRAW phase=rasterizer_draw indexed={} index_offset={}",
                 is_indexed ? "YES" : "NO", index_offset);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_VK_ACTUAL_DRAW] phase=rasterizer_draw indexed=%u "
                            "indexOffset=%u",
                            is_indexed ? 1u : 0u, index_offset);
    }
#else
    LOG_INFO(Render_Vulkan, "EXECUTOR_VK_ACTUAL_DRAW phase=rasterizer_draw indexed={} index_offset={}",
             is_indexed ? "YES" : "NO", index_offset);
#endif

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

#ifdef __ANDROID__
    // EXECUTOR bisection: localize the host-side wild write in the draw path. Level N returns after an
    // increasing prefix of the draw so a single build can binary-search phases across runs:
    //   1 = skip everything (no pipeline compile, no bind, no vkCmdDraw)
    //   2 = compile pipeline only, skip PrepareRenderState/BindResources/BeginRendering/draw
    //   3 = + PrepareRenderState + BindResources, skip BeginRendering/descriptor-write/draw
    //   4 = + everything except the final vkCmdDraw/DrawIndexed
    // If the "destroyed mutex" FORTIFY stops at level K but reproduces at K+1, the corruptor is in the
    // phase between them.
    static const int kBisectDrawLevel = [] {
        const char* e = std::getenv("EXECUTOR_BISECT_DRAW");
        const int lvl = e ? std::atoi(e) : 0;
        if (lvl > 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_BISECT_DRAW] active level=%d", lvl);
        }
        return lvl;
    }();
    if (kBisectDrawLevel == 1) {
        return;
    }
#endif

    const auto& regs = liverpool->regs;
    // EXECUTOR (Codex phase-2): record this draw's shaders into the active .gnmcap capture session. The
    // buffers it reads are recorded centrally in BufferCache::ObtainBuffer; here we add the VS/PS code.
    if (Libraries::GnmDriver::ExecutorGnmCaptureActive()) {
        Libraries::GnmDriver::ExecutorGnmCaptureOnDraw();
        const auto rec_shader = [](const auto& prog, u32 stage) {
            const u64 addr = reinterpret_cast<u64>(prog.template Address<const u8*>());
            if (addr == 0) {
                return;
            }
            const auto p = AmdGpu::GetParams(prog);
            Libraries::GnmDriver::ExecutorGnmCaptureRecordShader(
                stage, p.hash, addr, static_cast<u32>(p.code.size() * sizeof(u32)));
        };
        rec_shader(regs.ps_program, 0u);  // PS
        rec_shader(regs.vs_program, 1u);  // VS
    }
#ifdef __ANDROID__
    static const bool kTraceDraw =
        ExecutorTracePm4Enabled() ||
        (::access("/data/data/app.lsx4.android/files/lsx4-home/run-trace-draw", F_OK) == 0);
#define EXEC_DRAW_TRACE(stage)                                                                          \
    do {                                                                                               \
        if (kTraceDraw)                                                                                \
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_DRAW_TRACE] " stage);         \
    } while (0)
#else
#define EXEC_DRAW_TRACE(stage) ((void)0)
#endif
    EXEC_DRAW_TRACE("enter");
#ifdef __ANDROID__
    const bool executor_draw_diag = Libraries::GnmDriver::ExecutorReplayActive() ||
                                    ExecutorTraceLivePresentResourcesEnabled();
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        Core::Memory::Instance()->VerifyReplayCanaries("draw_enter");
    }
    if ((kTraceDraw && Libraries::GnmDriver::ExecutorReplayActive()) || executor_draw_diag) {
        const u64 ps_a = reinterpret_cast<u64>(regs.ps_program.Address<const u8*>());
        const u64 vs_a = reinterpret_cast<u64>(regs.vs_program.Address<const u8*>());
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_PIPELINE_INPUT] indexed=%d psAddr=0x%llx psReloc=%d vsAddr=0x%llx "
                            "vsReloc=%d numIndices=%u",
                            (int)is_indexed, (unsigned long long)ps_a,
                            (int)Libraries::GnmDriver::ExecutorReplayIsRelocated(ps_a),
                            (unsigned long long)vs_a,
                            (int)Libraries::GnmDriver::ExecutorReplayIsRelocated(vs_a),
                            regs.num_indices);
    }
#endif
    // A single PS4 QuadList is already a four-vertex perimeter suitable for Vulkan's
    // TriangleFan topology. Bypass auxiliary tessellation for this common fullscreen path:
    // it avoids a mobile-driver varying corruption between TES and fragment stages and is also
    // cheaper. Multi-quad and indirect draws retain the general tessellation fallback.
    const bool single_quad_fan =
        regs.primitive_type == AmdGpu::PrimitiveType::QuadList && regs.num_indices == 4 &&
        regs.stage_enable.raw == AmdGpu::ShaderStageEnable::VgtStages::Vs;
    const GraphicsPipeline* pipeline =
        pipeline_cache.GetGraphicsPipeline(single_quad_fan
                                               ? std::optional{AmdGpu::PrimitiveType::TriangleFan}
                                               : std::nullopt);
    if (!pipeline || !pipeline->IsValid()) {
        EXEC_DRAW_TRACE("no_pipeline_return");
        return;
    }
    EXEC_DRAW_TRACE("pipeline_ok");
#ifdef __ANDROID__
    if (kBisectDrawLevel == 2) {
        return;
    }
#endif

    PrepareRenderState(pipeline);
    EXEC_DRAW_TRACE("render_state_ok");
    if (!BindResources(pipeline)) {
        EXEC_DRAW_TRACE("bind_resources_return");
        return;
    }
    EXEC_DRAW_TRACE("bind_resources_ok");
#ifdef __ANDROID__
    if (kBisectDrawLevel == 3) {
        return;
    }
#endif
    const auto state = BeginRendering(pipeline);
    EXEC_DRAW_TRACE("begin_rendering_ok");
#ifdef __ANDROID__
    // EXECUTOR: skip attachment-less (depth-only z-prepass) draws while the depth attachment is skipped --
    // an empty framebuffer draw crashes the driver. Re-enable with the depth attachment together.
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        bool any_color = false;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            if (state.color_attachments[cb].image_view) {
                any_color = true;
                break;
            }
        }
        if (!any_color) {
            EXEC_DRAW_TRACE("skip_zprepass_replay");
            return;
        }
    }
#endif

    const bool executor_gpu_probe_enabled = texture_cache.ExecutorGpuContentProbeEnabled();
    const u64 executor_gpu_probe_draw =
        executor_gpu_probe_enabled ? ++executor_gpu_probe_draw_sequence : 0;
    const auto* executor_gpu_probe_fragment =
        pipeline->TryGetStage(Shader::LogicalStage::Fragment);
    const u64 executor_gpu_probe_fragment_hash =
        executor_gpu_probe_fragment ? executor_gpu_probe_fragment->pgm_hash : 0;
#ifdef __ANDROID__
    const auto executor_gpu_probe_attachments =
        executor_gpu_probe_enabled ? ExecutorSummarizeAttachments(cb_descs, db_desc)
                                   : ExecutorVkAttachmentSummary{};
    const u64 executor_gpu_probe_attachment_signature =
        executor_gpu_probe_attachments.signature;
    const bool executor_gpu_probe_late = AmdGpu::RenderWaveTrace::LateFrameCaptureActive();
    if (executor_gpu_probe_late) {
        ExecutorGpuLatePassTransition(executor_gpu_probe_attachment_signature,
                                      executor_gpu_probe_draw, "graphics_signature");
    }
#else
    const u64 executor_gpu_probe_attachment_signature = 0;
    const bool executor_gpu_probe_late = false;
#endif
    const bool executor_gpu_probe_selected =
        executor_gpu_probe_enabled &&
        texture_cache.ExecutorGpuContentProbeDrawSelected(executor_gpu_probe_draw,
                                                          executor_gpu_probe_fragment_hash,
                                                          executor_gpu_probe_attachment_signature,
                                                          executor_gpu_probe_attachments.color_count,
                                                          cb_descs[0].first
                                                              ? cb_descs[0].second.info.size.width
                                                              : 0,
                                                          cb_descs[0].first
                                                              ? cb_descs[0].second.info.size.height
                                                              : 0,
                                                          static_cast<bool>(db_desc.first),
                                                          regs.num_indices, false);
    if (executor_gpu_probe_selected) {
#ifdef __ANDROID__
        const auto& probe_key = pipeline->GetGraphicsKey();
        const auto& probe_vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
        const auto [probe_vertex_offset, probe_instance_offset] =
            GetDrawOffsets(regs, probe_vs_info, pipeline->GetFetchShader());
        const auto probe_stage_hash = [&](Shader::LogicalStage stage) {
            return probe_key.stage_hashes[static_cast<u32>(stage)];
        };
        const auto probe_is_list_topology = [](const AmdGpu::PrimitiveType type) {
            const auto topology = LiverpoolToVK::PrimitiveType(type);
            return topology == vk::PrimitiveTopology::ePointList ||
                   topology == vk::PrimitiveTopology::eLineList ||
                   topology == vk::PrimitiveTopology::eTriangleList ||
                   topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
                   topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
        };
        const auto probe_is_patch_topology = [](const AmdGpu::PrimitiveType type) {
            return type == AmdGpu::PrimitiveType::PatchPrimitive ||
                   type == AmdGpu::PrimitiveType::QuadList ||
                   type == AmdGpu::PrimitiveType::RectList;
        };
        const bool probe_restart_raw = (regs.enable_primitive_restart & 1) != 0;
        const bool probe_restart_effective =
            probe_restart_raw &&
            (instance.IsListRestartSupported() || !probe_is_list_topology(regs.primitive_type)) &&
            (instance.IsPatchListRestartSupported() ||
             !probe_is_patch_topology(regs.primitive_type));

        Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> probe_attributes;
        Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> probe_bindings;
        Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> probe_divisors;
        Vulkan::VertexInputs<AmdGpu::Buffer> probe_guest_buffers;
        pipeline->GetVertexInputs(probe_attributes, probe_bindings, probe_divisors,
                                  probe_guest_buffers, regs.vgt_instance_step_rate_0,
                                  regs.vgt_instance_step_rate_1);
        const auto& probe_viewport = regs.viewports[0];
        const auto& probe_depth_bounds = regs.viewport_depths[0];
        const auto& probe_viewport_control = regs.viewport_control;
        const float probe_zoffset =
            probe_viewport_control.zoffset_enable ? probe_viewport.zoffset : 0.0f;
        const float probe_zscale =
            probe_viewport_control.zscale_enable ? probe_viewport.zscale : 1.0f;
        float probe_min_depth = probe_zoffset;
        float probe_max_depth = probe_zoffset + probe_zscale;
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            probe_min_depth = probe_zoffset - probe_zscale;
            probe_max_depth = probe_zoffset + probe_zscale;
        }
        const float probe_effective_min_depth =
            instance.IsDepthRangeUnrestrictedSupported() ? probe_min_depth
                                                          : std::max(probe_min_depth, 0.0f);
        const float probe_effective_max_depth =
            instance.IsDepthRangeUnrestrictedSupported() ? probe_max_depth
                                                          : std::min(probe_max_depth, 1.0f);
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GPU_VERTEX_PROFILE] seq=%llu fs=0x%llx legacyAttrib=%u "
            "dynamicVertexInput=%u attributes=%zu bindings=%zu divisors=%zu",
            static_cast<unsigned long long>(executor_gpu_probe_draw),
            static_cast<unsigned long long>(executor_gpu_probe_fragment_hash),
            instance.IsLegacyVertexAttributesSupported() ? 1u : 0u,
            instance.IsVertexInputDynamicState() ? 1u : 0u, probe_attributes.size(),
            probe_bindings.size(), probe_divisors.size());

        // Record the descriptor base and the byte compensation actually consumed by the emitted
        // shader.  Guest-address modulo is not authoritative here: small CPU-dirty inputs may have
        // been copied into an already aligned stream slice, while GPU-dirty inputs stay in their
        // cached device-local allocation.  buffer_infos and buf_offsets share the global buffer
        // order used by BindResources (FS -> graphics stages), so this is an exact end-to-end check
        // of the host-alignment bridge rather than another CPU-shadow fingerprint.
        u32 probe_global_buffer = 0;
        for (const auto* probe_stage : pipeline->GetStages()) {
            if (!probe_stage) {
                continue;
            }
            for (u32 stage_buffer = 0; stage_buffer < probe_stage->buffers.size();
                 ++stage_buffer, ++probe_global_buffer) {
                if (probe_global_buffer >= buffer_infos.size() ||
                    probe_global_buffer >= push_data.buf_offsets.size()) {
                    break;
                }
                const auto& desc = probe_stage->buffers[stage_buffer];
                const auto sharp = desc.GetSharp(*probe_stage);
                const auto& host = buffer_infos[probe_global_buffer];
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_GPU_BUFFER_BINDING] seq=%llu global=%u stage=%u stageHash=0x%llx "
                    "slot=%u sharpIdx=%u guest=0x%llx guestSize=%u storage=%u written=%u "
                    "hostOffset=0x%llx hostRange=0x%llx compensation=0x%x storageAlign=%llu",
                    static_cast<unsigned long long>(executor_gpu_probe_draw), probe_global_buffer,
                    static_cast<unsigned>(probe_stage->l_stage),
                    static_cast<unsigned long long>(probe_stage->pgm_hash), stage_buffer,
                    static_cast<unsigned>(desc.sharp_idx),
                    static_cast<unsigned long long>(sharp.base_address),
                    static_cast<unsigned>(sharp.GetSize()), desc.IsStorage(sharp) ? 1u : 0u,
                    desc.is_written ? 1u : 0u,
                    static_cast<unsigned long long>(host.offset),
                    static_cast<unsigned long long>(host.range),
                    static_cast<unsigned>(push_data.buf_offsets[probe_global_buffer]),
                    static_cast<unsigned long long>(instance.StorageMinAlignment()));
                if (probe_stage->l_stage == Shader::LogicalStage::Vertex &&
                    sharp.base_address != 0 && sharp.GetSize() != 0 &&
                    host.buffer != VK_NULL_HANDLE) {
                    const u64 exact_bound_offset =
                        host.offset + push_data.buf_offsets[probe_global_buffer];
                    buffer_cache.ExecutorProbeBoundGpuBuffer(
                        static_cast<VAddr>(sharp.base_address), sharp.GetSize(), host.buffer,
                        exact_bound_offset, "vs_buffer", executor_gpu_probe_draw, stage_buffer);
                }
            }
        }
        for (u32 i = 0; i < probe_guest_buffers.size(); ++i) {
            const auto& buffer = probe_guest_buffers[i];
            const u64 size = buffer.GetSize();
            const auto fingerprint = ExecutorFingerprintGuestRange(
                memory, static_cast<VAddr>(buffer.base_address), size);
            const bool gpu_modified = size != 0 && buffer_cache.IsRegionGpuModified(
                                                       static_cast<VAddr>(buffer.base_address), size);
            const bool cpu_modified = size != 0 && buffer_cache.IsRegionCpuModified(
                                                       static_cast<VAddr>(buffer.base_address), size);
            const auto dst = buffer.DstSelect();
            const u32 location = i < probe_attributes.size() ? probe_attributes[i].location : ~0u;
            const u32 binding = i < probe_attributes.size() ? probe_attributes[i].binding : ~0u;
            const u32 host_format = i < probe_attributes.size()
                                        ? static_cast<u32>(probe_attributes[i].format)
                                        : 0u;
            const u32 divisor = i < probe_bindings.size() ? probe_bindings[i].divisor : 0u;
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_VERTEX_INPUT] seq=%llu slot=%u loc=%u bind=%u guest=0x%llx "
                "size=%llu clamped=%llu records=%u stride=%u dataFmt=%u numFmt=%u "
                "hostFmt=%u dst=%u,%u,%u,%u divisor=%u gpuDirty=%u cpuDirty=%u "
                "readable=%u sampleDigest=0x%llx",
                static_cast<unsigned long long>(executor_gpu_probe_draw), i, location, binding,
                static_cast<unsigned long long>(buffer.base_address),
                static_cast<unsigned long long>(size),
                static_cast<unsigned long long>(fingerprint.clamped_size), buffer.num_records,
                buffer.GetStride(), static_cast<unsigned>(buffer.GetDataFmt()),
                static_cast<unsigned>(buffer.GetNumberFmt()), host_format,
                static_cast<unsigned>(dst.r), static_cast<unsigned>(dst.g),
                static_cast<unsigned>(dst.b), static_cast<unsigned>(dst.a), divisor,
                gpu_modified ? 1u : 0u, cpu_modified ? 1u : 0u,
                fingerprint.readable ? 1u : 0u,
                static_cast<unsigned long long>(fingerprint.digest));
            if (gpu_modified) {
                buffer_cache.ExecutorProbeGpuBuffer(
                    static_cast<VAddr>(buffer.base_address), static_cast<u32>(size),
                    "vertex", executor_gpu_probe_draw);
            }
        }
        // Fingerprint every guest buffer consumed by the selected vertex shader as well as the
        // fixed-function V# streams above.  Skinned/compute-fed geometry commonly keeps bone or
        // transformed-vertex data in SSBOs; on a split CPU/GPU backing model a perfectly valid V#
        // can still produce exploded geometry when the shader sees a stale copy.  Keep this tied to
        // the externally selected, bounded content probe so normal gameplay pays no logging cost.
        for (u32 i = 0; i < probe_vs_info.buffers.size(); ++i) {
            const auto& desc = probe_vs_info.buffers[i];
            const auto buffer = desc.GetSharp(probe_vs_info);
            const bool special = desc.IsSpecial();
            const VAddr address = static_cast<VAddr>(buffer.base_address);
            const u64 size = special ? 0 : buffer.GetSize();
            const auto fingerprint = ExecutorFingerprintGuestRange(memory, address, size);
            const bool gpu_modified = size != 0 && buffer_cache.IsRegionGpuModified(address, size);
            const bool cpu_modified = size != 0 && buffer_cache.IsRegionCpuModified(address, size);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_VS_BUFFER_INPUT] seq=%llu slot=%u sharpIdx=%u type=%u "
                "special=%u storage=%u written=%u formatted=%u guest=0x%llx size=%llu "
                "clamped=%llu records=%u stride=%u dataFmt=%u numFmt=%u gpuDirty=%u "
                "cpuDirty=%u readable=%u sampleDigest=0x%llx",
                static_cast<unsigned long long>(executor_gpu_probe_draw), i,
                static_cast<unsigned>(desc.sharp_idx), static_cast<unsigned>(desc.buffer_type),
                special ? 1u : 0u, desc.IsStorage(buffer) ? 1u : 0u,
                desc.is_written ? 1u : 0u, desc.is_formatted ? 1u : 0u,
                static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
                static_cast<unsigned long long>(fingerprint.clamped_size), buffer.num_records,
                buffer.GetStride(), static_cast<unsigned>(buffer.GetDataFmt()),
                static_cast<unsigned>(buffer.GetNumberFmt()), gpu_modified ? 1u : 0u,
                cpu_modified ? 1u : 0u, fingerprint.readable ? 1u : 0u,
                static_cast<unsigned long long>(fingerprint.digest));
            if (gpu_modified) {
                buffer_cache.ExecutorProbeGpuBuffer(address, static_cast<u32>(size),
                                                    "vs_buffer", executor_gpu_probe_draw);
            }
        }
        if (is_indexed) {
            const bool index16 =
                regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
            const u32 index_size = index16 ? sizeof(u16) : sizeof(u32);
            const VAddr index_address = regs.index_base_address.Address<VAddr>() +
                                        static_cast<u64>(index_offset) * index_size;
            const u64 index_bytes = static_cast<u64>(regs.num_indices) * index_size;
            const auto fingerprint =
                ExecutorFingerprintGuestRange(memory, index_address, index_bytes);
            const bool gpu_modified = index_bytes != 0 &&
                                      buffer_cache.IsRegionGpuModified(index_address, index_bytes);
            const bool cpu_modified = index_bytes != 0 &&
                                      buffer_cache.IsRegionCpuModified(index_address, index_bytes);
            u32 min_index = ~0u;
            u32 max_index = 0;
            const u32 scan_count = static_cast<u32>(std::min<u64>(
                regs.num_indices, fingerprint.clamped_size / index_size));
            if (fingerprint.readable) {
                for (u32 i = 0; i < scan_count; ++i) {
                    u32 value{};
                    if (index16) {
                        u16 value16{};
                        std::memcpy(&value16,
                                    reinterpret_cast<const void*>(index_address +
                                                                  static_cast<u64>(i) * 2),
                                    sizeof(value16));
                        value = value16;
                    } else {
                        std::memcpy(&value,
                                    reinterpret_cast<const void*>(index_address +
                                                                  static_cast<u64>(i) * 4),
                                    sizeof(value));
                    }
                    min_index = std::min(min_index, value);
                    max_index = std::max(max_index, value);
                }
            }
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_INDEX_INPUT] seq=%llu guest=0x%llx bytes=%llu clamped=%llu "
                "type=%u indexOffset=%u count=%u scanned=%u min=%u max=%u gpuDirty=%u "
                "cpuDirty=%u readable=%u sampleDigest=0x%llx",
                static_cast<unsigned long long>(executor_gpu_probe_draw),
                static_cast<unsigned long long>(index_address),
                static_cast<unsigned long long>(index_bytes),
                static_cast<unsigned long long>(fingerprint.clamped_size), index16 ? 16u : 32u,
                index_offset, regs.num_indices, scan_count,
                scan_count == 0 ? 0u : min_index, max_index, gpu_modified ? 1u : 0u,
                cpu_modified ? 1u : 0u, fingerprint.readable ? 1u : 0u,
                static_cast<unsigned long long>(fingerprint.digest));
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GPU_DRAW_STATE] seq=%llu fs=0x%llx tcs=0x%llx tes=0x%llx "
            "vs=0x%llx gs=0x%llx indexed=%u numIdx=%u "
            "instances=%u psMrtMask=0x%x keyMrtMask=0x%x keyWriteMask0=0x%x ctMask0=0x%x "
            "discard=%u colorMode=%u colorCtlRaw=0x%x rop3=%u keyLogicOp=%u "
            "blend0=%u blendRaw0=0x%x disableRop0=%u src=%u dst=%u "
            "vp=%.2f,%.2f,%.1f,%.1f "
            "scissor=%d,%d-%d,%d cb0=0x%llx idxReg=%u voff=%u ioff=%u prim=%u "
            "polyRaw=0x%x cull=%u front=%u depthRaw=0x%x depthEn=%u depthWrite=%u depthFunc=%u "
            "stageEnable=0x%x hsInCp=%u hsOutCp=%u rasterKill=%u keyDiscard=%u clipDis=%u "
            "vpCtl=%u%u%u%u gen=%u,%u-%u,%u win=%u,%u-%u,%u winOff=%d,%d offDis=%u "
            "vpscEn=%u vpsc0=%u,%u-%u,%u restart=%u/%u restartIdx=0x%x "
            "listRestart=%u patchRestart=%u robustBuf2=%u robustImg2=%u nullDesc=%u "
            "writes=%zu buffers=%zu images=%zu feedback=%u/%u depthUnrestricted=%u "
            "clipSpace=%u zCtl=%u%u zRaw=%.6f,%.6f zEffective=%.6f,%.6f "
            "zBounds=%.6f,%.6f "
            "gsMode=0x%x gsScenario=%u cut=%u onchip=%u esPass=%u",
            static_cast<unsigned long long>(executor_gpu_probe_draw),
            static_cast<unsigned long long>(executor_gpu_probe_fragment_hash),
            static_cast<unsigned long long>(
                probe_stage_hash(Shader::LogicalStage::TessellationControl)),
            static_cast<unsigned long long>(
                probe_stage_hash(Shader::LogicalStage::TessellationEval)),
            static_cast<unsigned long long>(probe_stage_hash(Shader::LogicalStage::Vertex)),
            static_cast<unsigned long long>(probe_stage_hash(Shader::LogicalStage::Geometry)),
            is_indexed ? 1u : 0u, regs.num_indices, regs.num_instances.NumInstances(),
            executor_gpu_probe_fragment ? executor_gpu_probe_fragment->mrt_mask : 0u,
            probe_key.mrt_mask, static_cast<unsigned>(probe_key.write_masks[0]),
            regs.color_target_mask.GetMask(0),
            executor_gpu_probe_fragment && executor_gpu_probe_fragment->has_discard ? 1u : 0u,
            static_cast<unsigned>(regs.color_control.mode),
            std::bit_cast<u32>(regs.color_control),
            static_cast<unsigned>(regs.color_control.rop3),
            static_cast<unsigned>(probe_key.logic_op),
            static_cast<unsigned>(regs.blend_control[0].enable),
            std::bit_cast<u32>(regs.blend_control[0]),
            static_cast<unsigned>(regs.blend_control[0].disable_rop3),
            static_cast<unsigned>(regs.blend_control[0].color_src_factor),
            static_cast<unsigned>(regs.blend_control[0].color_dst_factor),
            regs.viewports[0].xscale, regs.viewports[0].yscale, regs.viewports[0].xoffset,
            regs.viewports[0].yoffset, static_cast<int>(regs.screen_scissor.top_left_x),
            static_cast<int>(regs.screen_scissor.top_left_y),
            static_cast<int>(regs.screen_scissor.bottom_right_x),
            static_cast<int>(regs.screen_scissor.bottom_right_y),
            static_cast<unsigned long long>(regs.color_buffers[0].Address()),
            regs.index_offset, probe_vertex_offset, probe_instance_offset,
            static_cast<unsigned>(regs.primitive_type),
            std::bit_cast<u32>(regs.polygon_control),
            static_cast<unsigned>(regs.polygon_control.CullingMode()),
            static_cast<unsigned>(regs.polygon_control.front_face),
            std::bit_cast<u32>(regs.depth_control),
            static_cast<unsigned>(regs.depth_control.depth_enable),
            static_cast<unsigned>(regs.depth_control.depth_write_enable),
            static_cast<unsigned>(regs.depth_control.depth_func),
            static_cast<unsigned>(regs.stage_enable.raw),
            static_cast<unsigned>(regs.ls_hs_config.hs_input_control_points),
            static_cast<unsigned>(regs.ls_hs_config.hs_output_control_points),
            static_cast<unsigned>(regs.clipper_control.dx_rasterization_kill),
            static_cast<unsigned>(probe_key.rasterizer_discard_enable),
            static_cast<unsigned>(regs.clipper_control.clip_disable),
            static_cast<unsigned>(regs.viewport_control.xscale_enable),
            static_cast<unsigned>(regs.viewport_control.xoffset_enable),
            static_cast<unsigned>(regs.viewport_control.yscale_enable),
            static_cast<unsigned>(regs.viewport_control.yoffset_enable),
            static_cast<unsigned>(regs.generic_scissor.top_left_x),
            static_cast<unsigned>(regs.generic_scissor.top_left_y),
            static_cast<unsigned>(regs.generic_scissor.bottom_right_x),
            static_cast<unsigned>(regs.generic_scissor.bottom_right_y),
            static_cast<unsigned>(regs.window_scissor.top_left_x),
            static_cast<unsigned>(regs.window_scissor.top_left_y),
            static_cast<unsigned>(regs.window_scissor.bottom_right_x),
            static_cast<unsigned>(regs.window_scissor.bottom_right_y),
            static_cast<int>(regs.window_offset.window_x_offset),
            static_cast<int>(regs.window_offset.window_y_offset),
            static_cast<unsigned>(regs.window_scissor.window_offset_disable),
            static_cast<unsigned>(regs.mode_control.vport_scissor_enable),
            static_cast<unsigned>(regs.viewport_scissors[0].top_left_x),
            static_cast<unsigned>(regs.viewport_scissors[0].top_left_y),
            static_cast<unsigned>(regs.viewport_scissors[0].bottom_right_x),
            static_cast<unsigned>(regs.viewport_scissors[0].bottom_right_y),
            probe_restart_raw ? 1u : 0u, probe_restart_effective ? 1u : 0u,
            regs.primitive_restart_index, instance.IsListRestartSupported() ? 1u : 0u,
            instance.IsPatchListRestartSupported() ? 1u : 0u,
            instance.IsRobustBufferAccess2Supported() ? 1u : 0u,
            instance.IsRobustImageAccess2Supported() ? 1u : 0u,
            instance.IsNullDescriptorSupported() ? 1u : 0u, set_writes.size(), buffer_infos.size(),
            image_infos.size(), attachment_feedback_loop ? 1u : 0u,
            instance.IsAttachmentFeedbackLoopLayoutSupported() ? 1u : 0u,
            instance.IsDepthRangeUnrestrictedSupported() ? 1u : 0u,
            static_cast<unsigned>(regs.clipper_control.clip_space),
            static_cast<unsigned>(probe_viewport_control.zscale_enable),
            static_cast<unsigned>(probe_viewport_control.zoffset_enable), probe_min_depth,
            probe_max_depth, probe_effective_min_depth, probe_effective_max_depth,
            probe_depth_bounds.zmin, probe_depth_bounds.zmax,
            std::bit_cast<u32>(regs.vgt_gs_mode),
            static_cast<unsigned>(regs.vgt_gs_mode.mode), regs.vgt_gs_mode.cut_mode,
            regs.vgt_gs_mode.onchip, regs.vgt_gs_mode.es_passthru);
        const auto log_probe_image = [&](const char* role, VideoCore::ImageId image_id,
                                         u64 guest_address) {
            if (!image_id) {
                __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                    "[EXECUTOR_GPU_ATTACHMENT] seq=%llu role=%s imageId=0 "
                                    "guest=0x%llx",
                                    static_cast<unsigned long long>(executor_gpu_probe_draw), role,
                                    static_cast<unsigned long long>(guest_address));
                return;
            }
            const auto& image = texture_cache.GetImage(image_id);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_ATTACHMENT] seq=%llu role=%s imageId=%u guest=0x%llx "
                "tracked=0x%llx format=%s extent=%ux%ux%u pitch=%u layers=%u bits=%u "
                "samples=%u layout=%u access=0x%llx depth=%u stencil=%u",
                static_cast<unsigned long long>(executor_gpu_probe_draw), role, image_id.index,
                static_cast<unsigned long long>(guest_address),
                static_cast<unsigned long long>(image.info.guest_address),
                vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                image.info.size.height, image.info.size.depth, image.info.pitch,
                image.info.resources.layers, image.info.num_bits, image.info.num_samples,
                static_cast<u32>(image.backing->state.layout),
                static_cast<unsigned long long>(
                    static_cast<VkAccessFlags2>(image.backing->state.access_mask)),
                image.info.props.is_depth ? 1u : 0u,
                image.info.props.has_stencil ? 1u : 0u);
        };
        static constexpr std::array<const char*, AmdGpu::NUM_COLOR_BUFFERS> color_roles{
            "color0", "color1", "color2", "color3", "color4", "color5", "color6", "color7"};
        for (u32 color_index = 0; color_index < cb_descs.size(); ++color_index) {
            log_probe_image(color_roles[color_index], cb_descs[color_index].first,
                            regs.color_buffers[color_index].Address());
            const auto& color_key = probe_key.color_buffers[color_index];
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_MRT_STATE] seq=%llu index=%u bound=%u guest=0x%llx "
                "dataFmt=%u numFmt=%u conversion=%u exportFmt=%u writeMask=0x%x "
                "targetMask=0x%x samples=%u",
                static_cast<unsigned long long>(executor_gpu_probe_draw), color_index,
                cb_descs[color_index].first ? 1u : 0u,
                static_cast<unsigned long long>(regs.color_buffers[color_index].Address()),
                static_cast<unsigned>(color_key.data_format),
                static_cast<unsigned>(color_key.num_format),
                static_cast<unsigned>(color_key.num_conversion),
                static_cast<unsigned>(color_key.export_format),
                static_cast<unsigned>(probe_key.write_masks[color_index]),
                regs.color_target_mask.GetMask(color_index), probe_key.color_samples[color_index]);
        }
        log_probe_image("depth", db_desc.first, regs.depth_buffer.DepthAddress());
        for (const auto* probe_stage : pipeline->GetStages()) {
            if (!probe_stage) {
                continue;
            }
            Common::ContentFingerprint64 flat_fingerprint{
                Common::FingerprintDomain::ShaderUserData};
            u32 flat_nonzero = 0;
            for (const u32 value : probe_stage->flattened_ud_buf) {
                flat_fingerprint.UpdateLittleEndian(value);
                flat_nonzero += value != 0;
            }
            const u64 flat_digest = flat_fingerprint.Finish();
            const auto& flat = probe_stage->flattened_ud_buf;
            const auto flat_at = [&](size_t index) {
                return index < flat.size() ? flat[index] : 0u;
            };
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_STAGE_SRT] seq=%llu stage=%u logical=%u hash=0x%llx "
                "flatDw=%zu flatNonzero=%u digest=0x%llx nodes=%zu copies=%zu "
                "flat0_15=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x,"
                "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
                static_cast<unsigned long long>(executor_gpu_probe_draw),
                static_cast<unsigned>(probe_stage->stage),
                static_cast<unsigned>(probe_stage->l_stage),
                static_cast<unsigned long long>(probe_stage->pgm_hash), flat.size(), flat_nonzero,
                static_cast<unsigned long long>(flat_digest),
                probe_stage->srt_info.portable_nodes.size(),
                probe_stage->srt_info.portable_copies.size(), flat_at(0), flat_at(1), flat_at(2),
                flat_at(3), flat_at(4), flat_at(5), flat_at(6), flat_at(7), flat_at(8), flat_at(9),
                flat_at(10), flat_at(11), flat_at(12), flat_at(13), flat_at(14), flat_at(15));
        }
        if (executor_gpu_probe_fragment) {
            for (u32 image_index = 0;
                 image_index < executor_gpu_probe_fragment->images.size(); ++image_index) {
                const auto& image_desc = executor_gpu_probe_fragment->images[image_index];
                const auto image = image_desc.GetSharp(*executor_gpu_probe_fragment);
                const auto swizzle = image.DstSelect();
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_GPU_IMAGE_STATE] seq=%llu fs=0x%llx slot=%u sharp=%u "
                    "guest=0x%llx type=%u extent=%ux%ux%u pitch=%u dataFmt=%u numFmt=%u "
                    "dst=%u,%u,%u,%u levels=%u..%u array=%u..%u depth=%u written=%u "
                    "isDepth=%u",
                    static_cast<unsigned long long>(executor_gpu_probe_draw),
                    static_cast<unsigned long long>(executor_gpu_probe_fragment_hash),
                    image_index, image_desc.sharp_idx,
                    static_cast<unsigned long long>(image.Address()),
                    static_cast<unsigned>(image.GetType()), static_cast<unsigned>(image.width + 1),
                    static_cast<unsigned>(image.height + 1),
                    static_cast<unsigned>(image.depth + 1), image.Pitch(),
                    static_cast<unsigned>(image.GetDataFmt()),
                    static_cast<unsigned>(image.GetNumberFmt()),
                    static_cast<unsigned>(swizzle.r), static_cast<unsigned>(swizzle.g),
                    static_cast<unsigned>(swizzle.b), static_cast<unsigned>(swizzle.a),
                    static_cast<unsigned>(image.base_level),
                    static_cast<unsigned>(image.last_level),
                    static_cast<unsigned>(image.base_array),
                    static_cast<unsigned>(image.last_array),
                    static_cast<unsigned>(image.depth), image_desc.is_written ? 1u : 0u,
                    image_desc.is_depth ? 1u : 0u);
            }
            for (u32 sampler_index = 0;
                 sampler_index < executor_gpu_probe_fragment->samplers.size(); ++sampler_index) {
                const auto& sampler_desc = executor_gpu_probe_fragment->samplers[sampler_index];
                const auto sampler = sampler_desc.GetSharp(*executor_gpu_probe_fragment);
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_GPU_SAMPLER_STATE] seq=%llu fs=0x%llx slot=%u sharp=%u "
                    "assocImage=%u raw0=0x%llx raw1=0x%llx clamp=%u,%u,%u filter=%u,%u,%u "
                    "forceUnnorm=%u",
                    static_cast<unsigned long long>(executor_gpu_probe_draw),
                    static_cast<unsigned long long>(executor_gpu_probe_fragment_hash),
                    sampler_index, sampler_desc.sharp_idx, sampler_desc.associated_image,
                    static_cast<unsigned long long>(sampler.raw0),
                    static_cast<unsigned long long>(sampler.raw1),
                    static_cast<unsigned>(sampler.clamp_x.Value()),
                    static_cast<unsigned>(sampler.clamp_y.Value()),
                    static_cast<unsigned>(sampler.clamp_z.Value()),
                    static_cast<unsigned>(sampler.xy_mag_filter.Value()),
                    static_cast<unsigned>(sampler.xy_min_filter.Value()),
                    static_cast<unsigned>(sampler.mip_filter.Value()),
                    static_cast<unsigned>(sampler.force_unnormalized.Value()));
            }
        }
#endif
        if (!executor_gpu_probe_late) {
            for (const auto& color_desc : cb_descs) {
                if (!color_desc.first) {
                    continue;
                }
                texture_cache.ExecutorGpuContentProbe(
                    color_desc.first,
                    VideoCore::TextureCache::ExecutorGpuContentRole::ColorBeforeDraw,
                    static_cast<u32>(Shader::LogicalStage::Fragment),
                    executor_gpu_probe_fragment_hash, executor_gpu_probe_draw);
            }
            if (db_desc.first) {
                texture_cache.ExecutorGpuContentProbe(
                    db_desc.first,
                    VideoCore::TextureCache::ExecutorGpuContentRole::DepthBeforeDraw,
                    static_cast<u32>(Shader::LogicalStage::Fragment),
                    executor_gpu_probe_fragment_hash, executor_gpu_probe_draw);
            }
        }
    }

    // Prepare every buffer before recording fixed-function state. ObtainBuffer and the bounded
    // diagnostics may wait/submit and rotate Scheduler::CommandBuffer; prepared Vulkan handles
    // remain valid and are emitted only after the final pipeline bind below.
    const auto prepared_vertex_buffers =
        buffer_cache.PrepareVertexBuffers(*pipeline, buffer_barriers,
                                          executor_gpu_probe_selected, executor_gpu_probe_draw);
    EXEC_DRAW_TRACE("vertex_buffers_ok");
    VideoCore::BufferCache::PreparedIndexBuffer prepared_index_buffer{};
    if (is_indexed) {
        prepared_index_buffer =
            buffer_cache.PrepareIndexBuffer(index_offset, buffer_barriers,
                                            executor_gpu_probe_selected, executor_gpu_probe_draw);
    }

    // Buffer/resource preparation above may submit and rotate the command buffer. Stream allocations
    // were initially watched under the preparation tick, but the descriptors and vertex bindings are
    // consumed by the current draw tick. Retain the populated ring prefix until that tick completes.
    buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream).RetainCurrentAllocation();

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
#ifdef __ANDROID__
    const bool executor_post_bind_dynamic =
        ExecutorNeedsQualcommPostBindDynamicState(instance);
    if (!executor_post_bind_dynamic) {
        UpdateDynamicState(pipeline, is_indexed);
    }
#else
    UpdateDynamicState(pipeline, is_indexed);
#endif
#ifdef __ANDROID__
    if (executor_gpu_probe_selected && executor_gpu_probe_late) {
        // Arm only after the guest depth state has been converted into Scheduler dynamic state, so
        // the oracle records both the raw PM4 bits and the exact depth-write value Vulkan will see.
        ExecutorGpuLatePassArm(executor_gpu_probe_attachment_signature,
                               executor_gpu_probe_draw, executor_gpu_probe_fragment_hash);
    }
#endif
    scheduler.BeginRendering(state);

    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);
#ifdef __ANDROID__
    // Diagnostic A/B for the generic DrawIndexAuto/RectList path. Some command streams leave
    // VGT_INDEX_OFFSET live across draws; this switch lets us prove whether feeding that stale base
    // into Vulkan's gl_VertexIndex moves the auxiliary fullscreen rect out of coverage. It is off by
    // default and deliberately keyed by a runtime marker, never by title or shader hash.
    static const bool kForceRectListFirstVertexZero =
        ::access("/data/data/app.lsx4.android/files/lsx4-home/"
                 "run-force-rectlist-first-vertex-zero",
                 F_OK) == 0;
    if (kForceRectListFirstVertexZero && !is_indexed &&
        regs.primitive_type == AmdGpu::PrimitiveType::RectList && vertex_offset != 0) {
        static std::atomic<u32> forced_count{0};
        const u32 count = forced_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count <= 32) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_RECTLIST_FIRST_VERTEX_AB] n=%u raw=%u effective=0 "
                                "indexReg=%u",
                                count, vertex_offset, regs.index_offset);
        }
        vertex_offset = 0;
    }
#endif

    const auto cmdbuf = scheduler.CommandBuffer();
#ifdef __ANDROID__
    ExecutorBindGuestGraphicsPipeline(pipeline, cmdbuf);
#else
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
#endif
    buffer_cache.BindPreparedVertexBuffers(prepared_vertex_buffers);
    if (is_indexed) {
        buffer_cache.BindPreparedIndexBuffer(prepared_index_buffer);
    }
#ifdef __ANDROID__
    if (executor_post_bind_dynamic) {
        // A real pipeline bind invalidates the complete guest state. Same-scope cache hits retain
        // ordinary changed-only tracking.
        UpdateDynamicState(pipeline, is_indexed);
    }
#endif
    EXEC_DRAW_TRACE("before_draw_cmd");
#ifdef __ANDROID__
    const bool executor_depth_write_oracle_clear =
        executor_gpu_probe_selected && executor_gpu_probe_late && db_desc.first &&
        ExecutorClaimDepthWriteOracle(executor_gpu_probe_attachment_signature);
    const bool executor_depth_write_oracle =
        db_desc.first && ExecutorDepthWriteOracleActive(executor_gpu_probe_attachment_signature);
    if (executor_depth_write_oracle_clear) {
        // Clear to a value which cannot be confused with the guest's normal 1.0 depth clear, then
        // replay the essential EDS state *after* pipeline binding for the complete attachment span.
        // The existing pass-end readback gives three disjoint outcomes:
        //   0.25 unchanged -> no fragment coverage or the Vulkan write command was ineffective;
        //   1.0 written    -> the shader/rasterizer produced far-plane depth;
        //   another value -> normal depth exists and the ordinary dynamic-state path suppressed it.
        vk::ClearValue clear_value{};
        clear_value.depthStencil = vk::ClearDepthStencilValue{0.25f, 0};
        const std::array clear_attachments{
            vk::ClearAttachment{vk::ImageAspectFlagBits::eDepth, 0, clear_value}};
        const std::array clear_rects{vk::ClearRect{
            vk::Rect2D{vk::Offset2D{0, 0}, vk::Extent2D{state.width, state.height}}, 0,
            state.num_layers}};
        cmdbuf.clearAttachments(clear_attachments, clear_rects);
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_DEPTH_WRITE_ORACLE] phase=armed epoch=%llu draw=%llu imageId=%u "
            "attach=0x%llx sentinel=0.25 extent=%ux%u layers=%u "
            "stateOrder=after_bind compare=always write=1 scope=complete_pass",
            static_cast<unsigned long long>(AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch()),
            static_cast<unsigned long long>(executor_gpu_probe_draw), db_desc.first.index,
            static_cast<unsigned long long>(executor_gpu_probe_attachment_signature), state.width,
            state.height, state.num_layers);
    }
    if (executor_depth_write_oracle) {
        cmdbuf.setDepthTestEnable(true);
        cmdbuf.setDepthWriteEnable(true);
        cmdbuf.setDepthCompareOp(vk::CompareOp::eAlways);
        cmdbuf.setStencilTestEnable(false);
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(false);
        }
        cmdbuf.setDepthBiasEnable(false);
        static std::atomic<u32> replay_logs{0};
        const u32 replay_log = replay_logs.fetch_add(1, std::memory_order_relaxed);
        if (replay_log < 4) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_DEPTH_WRITE_ORACLE] phase=post_bind_replay epoch=%llu draw=%llu "
                "attach=0x%llx ordinal=%u",
                static_cast<unsigned long long>(AmdGpu::RenderWaveTrace::LateFrameCaptureEpoch()),
                static_cast<unsigned long long>(executor_gpu_probe_draw),
                static_cast<unsigned long long>(executor_gpu_probe_attachment_signature),
                replay_log + 1);
        }
    }
    if (executor_draw_diag) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_DRAW_STATE] mode=%s idx=%d numIdx=%u inst=%u colorMode=%u tmask0=%u depthValid=%d "
            "vp xs=%.2f ys=%.2f xo=%.1f yo=%.1f voff=%d scis=[%d,%d-%d,%d] gscis=[%d,%d-%d,%d] "
            "blend0 en=%u src=%u dst=%u dctl en=%u zfunc=%u",
            Libraries::GnmDriver::ExecutorReplayActive() ? "replay" : "live",
            is_indexed ? 1 : 0, regs.num_indices, regs.num_instances.NumInstances(),
            (unsigned)regs.color_control.mode, regs.color_target_mask.GetMask(0),
            regs.depth_buffer.DepthValid() ? 1 : 0, regs.viewports[0].xscale, regs.viewports[0].yscale,
            regs.viewports[0].xoffset, regs.viewports[0].yoffset, (int)vertex_offset,
            (int)regs.screen_scissor.top_left_x, (int)regs.screen_scissor.top_left_y,
            (int)regs.screen_scissor.bottom_right_x, (int)regs.screen_scissor.bottom_right_y,
            (int)regs.generic_scissor.top_left_x, (int)regs.generic_scissor.top_left_y,
            (int)regs.generic_scissor.bottom_right_x, (int)regs.generic_scissor.bottom_right_y,
            (unsigned)regs.blend_control[0].enable, (unsigned)regs.blend_control[0].color_src_factor,
            (unsigned)regs.blend_control[0].color_dst_factor, (unsigned)regs.depth_control.depth_enable,
            (unsigned)regs.depth_control.depth_func);
        // RT-write diagnosis (Codex sec.16): does the real PS actually export MRT0 / is it discarded /
        // is color write masked off? If psMrtMask lacks bit0 or wmask0==0 the draw rasterizes but writes
        // nothing; hasDiscard=1 means the shader may KILL all fragments. cb0base ties the draw to its RT.
        const auto& gkey = pipeline->GetGraphicsKey();
        const auto* ps_info = pipeline->TryGetStage(Shader::LogicalStage::Fragment);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_PS_INFO] mode=%s psMrtMask=0x%x hasDiscard=%d keyMrtMask=0x%x "
                            "wmask0=0x%x ctMask0=0x%x cb0base=0x%llx",
                            Libraries::GnmDriver::ExecutorReplayActive() ? "replay" : "live",
                            ps_info ? (unsigned)ps_info->mrt_mask : 0u,
                            ps_info && ps_info->has_discard ? 1 : 0,
                            (unsigned)gkey.mrt_mask, (unsigned)gkey.write_masks[0],
                            (unsigned)regs.color_target_mask.GetMask(0),
                            (unsigned long long)regs.color_buffers[0].Address());
    }
#endif

#ifdef __ANDROID__
    if (kBisectDrawLevel == 4) {
        ResetBindings();
        return;
    }
    static const u64 kBisectDrawMax = [] {
        const char* value = std::getenv("EXECUTOR_BISECT_DRAW_MAX");
        return value ? std::strtoull(value, nullptr, 10) : 0ull;
    }();
    static std::atomic<u64> bisect_draw_sequence{0};
    const u64 bisect_draw_sequence_value =
        bisect_draw_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    if (kBisectDrawMax != 0 && bisect_draw_sequence_value > kBisectDrawMax) {
        ResetBindings();
        return;
    }
    if (kBisectDrawMax != 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_BISECT_DRAW_MAX] execute=%llu max=%llu pipe=0x%llx "
                            "vs=0x%llx ps=0x%llx elements=%u cb0=0x%llx",
                            static_cast<unsigned long long>(bisect_draw_sequence_value),
                            static_cast<unsigned long long>(kBisectDrawMax),
                            static_cast<unsigned long long>(pipeline->GetPipelineHash()),
                            static_cast<unsigned long long>(pipeline->GetGraphicsKey().stage_hashes[
                                u32(Shader::LogicalStage::Vertex)]),
                            static_cast<unsigned long long>(pipeline->GetGraphicsKey().stage_hashes[
                                u32(Shader::LogicalStage::Fragment)]),
                            regs.num_indices,
                            static_cast<unsigned long long>(regs.color_buffers[0].Address()));
    }
#endif
    if (is_indexed) {
        cmdbuf.drawIndexed(regs.num_indices, regs.num_instances.NumInstances(), 0,
                           s32(vertex_offset), instance_offset);
    } else {
        cmdbuf.draw(regs.num_indices, regs.num_instances.NumInstances(), vertex_offset,
                    instance_offset);
    }
#ifdef __ANDROID__
    if (executor_depth_write_oracle) {
        // The destructive oracle records state behind DynamicState's back. Make the next ordinary
        // draw restore the complete guest state even when it reuses the same pipeline.
        scheduler.GetDynamicState().Invalidate();
    }
#endif
    if (executor_gpu_probe_selected && !executor_gpu_probe_late) {
        for (const auto& color_desc : cb_descs) {
            if (!color_desc.first) {
                continue;
            }
            texture_cache.ExecutorGpuContentProbe(
                color_desc.first, VideoCore::TextureCache::ExecutorGpuContentRole::ColorAfterDraw,
                static_cast<u32>(Shader::LogicalStage::Fragment),
                executor_gpu_probe_fragment_hash, executor_gpu_probe_draw);
        }
    }
    if (executor_gpu_probe_selected && !executor_gpu_probe_late && db_desc.first) {
        texture_cache.ExecutorGpuContentProbe(
            db_desc.first, VideoCore::TextureCache::ExecutorGpuContentRole::DepthAfterDraw,
            static_cast<u32>(Shader::LogicalStage::Fragment), executor_gpu_probe_fragment_hash,
            executor_gpu_probe_draw);
    }
    if (executor_gpu_probe_selected && !executor_gpu_probe_late) {
        u32 sampled_probe_count = 0;
        for (const auto& probe : executor_gpu_probe_sampled_images) {
            if (sampled_probe_count++ >= 8) {
                break;
            }
            texture_cache.ExecutorGpuContentProbe(
                probe.image_id,
                VideoCore::TextureCache::ExecutorGpuContentRole::SampledBeforeDraw,
                probe.stage, probe.stage_hash, executor_gpu_probe_draw);
        }
    }
#ifdef __ANDROID__
    {
        const auto& key = pipeline->GetGraphicsKey();
        const auto& color = cb_descs[0].second.info;
        const u64 journal_sequence = ExecutorVkJournalGraphicsDraw(
            &scheduler, scheduler.CurrentTick(), pipeline->GetPipelineHash(),
            ExecutorVkPipelineHandle(pipeline->Handle()),
            key.stage_hashes[u32(Shader::LogicalStage::Vertex)],
            key.stage_hashes[u32(Shader::LogicalStage::Fragment)],
            is_indexed ? ExecutorVkJournalDrawType::IndexedDirect
                       : ExecutorVkJournalDrawType::Direct,
            static_cast<u32>(key.prim_type), regs.num_indices,
            regs.num_instances.NumInstances(), 0, 0, 0, 0,
            cb_descs[0].first ? color.guest_address : 0,
            executor_gpu_probe_attachments.depth_address,
            executor_gpu_probe_attachments.signature,
            executor_gpu_probe_attachments.color_count,
            cb_descs[0].first ? color.size.width : 0, cb_descs[0].first ? color.size.height : 0,
            static_cast<u32>(pipeline->GetColorFormat0()),
            static_cast<u32>(pipeline->GetDepthFormat()),
            static_cast<u32>(set_writes.size()), static_cast<u32>(buffer_infos.size()),
            static_cast<u32>(image_infos.size()), executor_gpu_probe_enabled);
        if (executor_gpu_probe_selected) {
            ExecutorVkJournalArmContentProbe(&scheduler, journal_sequence,
                                             executor_gpu_probe_draw,
                                             executor_gpu_probe_fragment_hash);
        }
    }
    ++executor_unpublished_actual_draws;
    {
        const auto& key = pipeline->GetGraphicsKey();
        AmdGpu::RenderWaveTrace::RasterDraw(
            regs, key.stage_hashes[u32(Shader::LogicalStage::Vertex)],
            key.stage_hashes[u32(Shader::LogicalStage::Fragment)], pipeline->GetPipelineHash(),
            is_indexed, false);
    }
    ExecutorTraceDrawnColorTarget(cb_descs[0].first ? &cb_descs[0].second : nullptr);
    // This is deliberately after vkCmdDraw: failed pipeline/resource paths must never make a merely
    // bound attachment eligible for direct presentation.
    if (Libraries::GnmDriver::ExecutorLivePresentRtEnabled() && cb_descs[0].first) {
        texture_cache.ExecutorTrackLiveDrawnColorTarget(cb_descs[0].first);
    }
#endif
#ifdef __ANDROID__
    if (trace_actual_draw) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_VK_ACTUAL_DRAW] phase=vk_cmd_draw indexed=%u numIdx=%u "
                            "instances=%u vertexOffset=%u instanceOffset=%u",
                            is_indexed ? 1u : 0u, regs.num_indices,
                            regs.num_instances.NumInstances(), vertex_offset, instance_offset);
    }
#endif
    // Pixel-proof readback deferred to OnSubmit (after all draws + EndRendering): reading the RT here,
    // while it is still the active color attachment of the open rendering scope, reads stale zeros
    // (Codex). The de-tile heisenbug that previously made the deferred path crash is fixed.
    ResetBindings();
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;
#ifdef __ANDROID__
    ++executor_completion_work_serial;
#endif

    scheduler.PopPendingOperations();

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline || !pipeline->IsValid()) {
        return;
    }

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);
    const auto& regs = liverpool->regs;

    const bool executor_gpu_probe_enabled = texture_cache.ExecutorGpuContentProbeEnabled();
    const u64 executor_gpu_probe_draw =
        executor_gpu_probe_enabled ? ++executor_gpu_probe_draw_sequence : 0;
    const auto* executor_gpu_probe_fragment =
        pipeline->TryGetStage(Shader::LogicalStage::Fragment);
    const u64 executor_gpu_probe_fragment_hash =
        executor_gpu_probe_fragment ? executor_gpu_probe_fragment->pgm_hash : 0;
#ifdef __ANDROID__
    const auto executor_gpu_probe_attachments =
        executor_gpu_probe_enabled ? ExecutorSummarizeAttachments(cb_descs, db_desc)
                                   : ExecutorVkAttachmentSummary{};
    const u64 executor_gpu_probe_attachment_signature =
        executor_gpu_probe_attachments.signature;
    const bool executor_gpu_probe_late = AmdGpu::RenderWaveTrace::LateFrameCaptureActive();
    if (executor_gpu_probe_late) {
        ExecutorGpuLatePassTransition(executor_gpu_probe_attachment_signature,
                                      executor_gpu_probe_draw, "graphics_signature");
    }
#else
    const u64 executor_gpu_probe_attachment_signature = 0;
    const bool executor_gpu_probe_late = false;
#endif
    const bool executor_gpu_probe_selected =
        executor_gpu_probe_enabled &&
        texture_cache.ExecutorGpuContentProbeDrawSelected(executor_gpu_probe_draw,
                                                          executor_gpu_probe_fragment_hash,
                                                          executor_gpu_probe_attachment_signature,
                                                          executor_gpu_probe_attachments.color_count,
                                                          cb_descs[0].first
                                                              ? cb_descs[0].second.info.size.width
                                                              : 0,
                                                          cb_descs[0].first
                                                              ? cb_descs[0].second.info.size.height
                                                              : 0,
                                                          static_cast<bool>(db_desc.first), 0, true);
    if (executor_gpu_probe_selected) {
#ifdef __ANDROID__
        const auto& probe_key = pipeline->GetGraphicsKey();
        const auto probe_stage_hash = [&](Shader::LogicalStage stage) {
            return probe_key.stage_hashes[static_cast<u32>(stage)];
        };
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GPU_DRAW_INDIRECT_STATE] seq=%llu fs=0x%llx tcs=0x%llx "
            "tes=0x%llx vs=0x%llx gs=0x%llx indexed=%u arg=0x%llx offset=%u "
            "stride=%u maxCount=%u count=0x%llx prim=%u stageEnable=0x%x cb0=0x%llx "
            "db=0x%llx writes=%zu buffers=%zu images=%zu",
            static_cast<unsigned long long>(executor_gpu_probe_draw),
            static_cast<unsigned long long>(executor_gpu_probe_fragment_hash),
            static_cast<unsigned long long>(
                probe_stage_hash(Shader::LogicalStage::TessellationControl)),
            static_cast<unsigned long long>(
                probe_stage_hash(Shader::LogicalStage::TessellationEval)),
            static_cast<unsigned long long>(probe_stage_hash(Shader::LogicalStage::Vertex)),
            static_cast<unsigned long long>(probe_stage_hash(Shader::LogicalStage::Geometry)),
            is_indexed ? 1u : 0u, static_cast<unsigned long long>(arg_address), offset, stride,
            max_count, static_cast<unsigned long long>(count_address),
            static_cast<unsigned>(regs.primitive_type), static_cast<unsigned>(regs.stage_enable.raw),
            static_cast<unsigned long long>(regs.color_buffers[0].Address()),
            static_cast<unsigned long long>(regs.depth_buffer.DepthAddress()), set_writes.size(),
            buffer_infos.size(), image_infos.size());
        for (const auto* probe_stage : pipeline->GetStages()) {
            if (!probe_stage) {
                continue;
            }
            Common::ContentFingerprint64 flat_fingerprint{
                Common::FingerprintDomain::ShaderUserData};
            u32 flat_nonzero = 0;
            for (const u32 value : probe_stage->flattened_ud_buf) {
                flat_fingerprint.UpdateLittleEndian(value);
                flat_nonzero += value != 0;
            }
            const u64 flat_digest = flat_fingerprint.Finish();
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GPU_STAGE_SRT] seq=%llu indirect=1 stage=%u logical=%u "
                "hash=0x%llx flatDw=%zu flatNonzero=%u digest=0x%llx nodes=%zu copies=%zu",
                static_cast<unsigned long long>(executor_gpu_probe_draw),
                static_cast<unsigned>(probe_stage->stage),
                static_cast<unsigned>(probe_stage->l_stage),
                static_cast<unsigned long long>(probe_stage->pgm_hash),
                probe_stage->flattened_ud_buf.size(), flat_nonzero,
                static_cast<unsigned long long>(flat_digest),
                probe_stage->srt_info.portable_nodes.size(),
                probe_stage->srt_info.portable_copies.size());
        }
#endif
        if (!executor_gpu_probe_late) {
            if (cb_descs[0].first) {
                texture_cache.ExecutorGpuContentProbe(
                    cb_descs[0].first,
                    VideoCore::TextureCache::ExecutorGpuContentRole::ColorBeforeDraw,
                    static_cast<u32>(Shader::LogicalStage::Fragment),
                    executor_gpu_probe_fragment_hash, executor_gpu_probe_draw);
            }
            if (db_desc.first) {
                texture_cache.ExecutorGpuContentProbe(
                    db_desc.first,
                    VideoCore::TextureCache::ExecutorGpuContentRole::DepthBeforeDraw,
                    static_cast<u32>(Shader::LogicalStage::Fragment),
                    executor_gpu_probe_fragment_hash, executor_gpu_probe_draw);
            }
        }
    }

    // Resolve indirect arguments first: these allocations can also rotate the command buffer.
    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    const auto prepared_vertex_buffers =
        buffer_cache.PrepareVertexBuffers(*pipeline, buffer_barriers);
    VideoCore::BufferCache::PreparedIndexBuffer prepared_index_buffer{};
    if (is_indexed) {
        prepared_index_buffer = buffer_cache.PrepareIndexBuffer(0, buffer_barriers);
    }

    // Indirect arguments may have just been produced by a compute shader.  They are not part of
    // the normal descriptor/vertex bindings, so BindResources cannot infer this dependency for us.
    // Match the PC renderer: make shader writes visible to the indirect command reader before the
    // command is recorded.  Missing this barrier can poison the whole submit even when the final
    // commands in it are ordinary direct draws.
    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    pipeline->BindResources(set_writes, buffer_barriers, push_data);
#ifdef __ANDROID__
    const bool executor_post_bind_dynamic =
        ExecutorNeedsQualcommPostBindDynamicState(instance);
    if (!executor_post_bind_dynamic) {
        UpdateDynamicState(pipeline, is_indexed);
    }
#else
    UpdateDynamicState(pipeline, is_indexed);
#endif
#ifdef __ANDROID__
    if (executor_gpu_probe_selected && executor_gpu_probe_late) {
        ExecutorGpuLatePassArm(executor_gpu_probe_attachment_signature,
                               executor_gpu_probe_draw, executor_gpu_probe_fragment_hash);
    }
#endif
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    const auto cmdbuf = scheduler.CommandBuffer();
#ifdef __ANDROID__
    ExecutorBindGuestGraphicsPipeline(pipeline, cmdbuf);
#else
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline->Handle());
#endif
    buffer_cache.BindPreparedVertexBuffers(prepared_vertex_buffers);
    if (is_indexed) {
        buffer_cache.BindPreparedIndexBuffer(prepared_index_buffer);
    }
#ifdef __ANDROID__
    if (executor_post_bind_dynamic) {
        UpdateDynamicState(pipeline, is_indexed);
    }
#endif

    if (is_indexed) {
        ASSERT(sizeof(VkDrawIndexedIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndexedIndirectCount(buffer->Handle(), base, count_buffer->Handle(),
                                            count_base, max_count, stride);
        } else {
            cmdbuf.drawIndexedIndirect(buffer->Handle(), base, max_count, stride);
        }
    } else {
        ASSERT(sizeof(VkDrawIndirectCommand) == stride);

        if (count_address != 0) {
            cmdbuf.drawIndirectCount(buffer->Handle(), base, count_buffer->Handle(), count_base,
                                     max_count, stride);
        } else {
            cmdbuf.drawIndirect(buffer->Handle(), base, max_count, stride);
        }
    }
    if (executor_gpu_probe_selected && !executor_gpu_probe_late && cb_descs[0].first) {
        texture_cache.ExecutorGpuContentProbe(
            cb_descs[0].first, VideoCore::TextureCache::ExecutorGpuContentRole::ColorAfterDraw,
            static_cast<u32>(Shader::LogicalStage::Fragment),
            executor_gpu_probe_fragment_hash,
            executor_gpu_probe_draw);
    }
    if (executor_gpu_probe_selected && !executor_gpu_probe_late && db_desc.first) {
        texture_cache.ExecutorGpuContentProbe(
            db_desc.first, VideoCore::TextureCache::ExecutorGpuContentRole::DepthAfterDraw,
            static_cast<u32>(Shader::LogicalStage::Fragment), executor_gpu_probe_fragment_hash,
            executor_gpu_probe_draw);
    }
    if (executor_gpu_probe_selected && !executor_gpu_probe_late) {
        u32 sampled_probe_count = 0;
        for (const auto& probe : executor_gpu_probe_sampled_images) {
            if (sampled_probe_count++ >= 8) {
                break;
            }
            texture_cache.ExecutorGpuContentProbe(
                probe.image_id,
                VideoCore::TextureCache::ExecutorGpuContentRole::SampledBeforeDraw,
                probe.stage, probe.stage_hash, executor_gpu_probe_draw);
        }
    }

#ifdef __ANDROID__
    {
        const auto& key = pipeline->GetGraphicsKey();
        const auto& color = cb_descs[0].second.info;
        const u64 journal_sequence = ExecutorVkJournalGraphicsDraw(
            &scheduler, scheduler.CurrentTick(), pipeline->GetPipelineHash(),
            ExecutorVkPipelineHandle(pipeline->Handle()),
            key.stage_hashes[u32(Shader::LogicalStage::Vertex)],
            key.stage_hashes[u32(Shader::LogicalStage::Fragment)],
            is_indexed ? ExecutorVkJournalDrawType::IndexedIndirect
                       : ExecutorVkJournalDrawType::Indirect,
            static_cast<u32>(key.prim_type), 0, 0, max_count, stride,
            arg_address + offset, count_address,
            cb_descs[0].first ? color.guest_address : 0,
            executor_gpu_probe_attachments.depth_address,
            executor_gpu_probe_attachments.signature,
            executor_gpu_probe_attachments.color_count,
            cb_descs[0].first ? color.size.width : 0, cb_descs[0].first ? color.size.height : 0,
            static_cast<u32>(pipeline->GetColorFormat0()),
            static_cast<u32>(pipeline->GetDepthFormat()), static_cast<u32>(set_writes.size()),
            static_cast<u32>(buffer_infos.size()), static_cast<u32>(image_infos.size()),
            executor_gpu_probe_enabled);
        if (executor_gpu_probe_selected) {
            ExecutorVkJournalArmContentProbe(&scheduler, journal_sequence,
                                             executor_gpu_probe_draw,
                                             executor_gpu_probe_fragment_hash);
        }
    }
    ++executor_unpublished_actual_draws;
    {
        const auto& key = pipeline->GetGraphicsKey();
        AmdGpu::RenderWaveTrace::RasterDraw(
            regs, key.stage_hashes[u32(Shader::LogicalStage::Vertex)],
            key.stage_hashes[u32(Shader::LogicalStage::Fragment)], pipeline->GetPipelineHash(),
            is_indexed, true);
    }
    ExecutorTraceDrawnColorTarget(cb_descs[0].first ? &cb_descs[0].second : nullptr);
    if (Libraries::GnmDriver::ExecutorLivePresentRtEnabled() && cb_descs[0].first) {
        texture_cache.ExecutorTrackLiveDrawnColorTarget(cb_descs[0].first);
    }
#endif

    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();
#ifdef __ANDROID__
    ++executor_completion_work_serial;
    if (AmdGpu::RenderWaveTrace::LateFrameCaptureActive()) {
        ExecutorGpuLatePassTransition(0, executor_gpu_probe_dispatch_sequence + 1,
                                      "compute_direct");
    }
#endif

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline || !pipeline->IsValid()) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatch(cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);

#ifdef __ANDROID__
    // Journal the command while it still belongs to this command buffer. A selected content probe
    // may call Finish() and rotate the scheduler before returning.
    ExecutorVkJournalComputeDispatch(
        &scheduler, scheduler.CurrentTick(), pipeline->GetPipelineHash(),
        ExecutorVkPipelineHandle(pipeline->Handle()), pipeline->GetPipelineHash(), false,
        cs_program.dim_x, cs_program.dim_y, cs_program.dim_z, 0,
        static_cast<u32>(set_writes.size()), static_cast<u32>(buffer_infos.size()),
        static_cast<u32>(image_infos.size()),
        texture_cache.ExecutorGpuContentProbeEnabled());
#endif

    if (texture_cache.ExecutorGpuContentProbeEnabled()) {
        const u64 executor_gpu_probe_dispatch = ++executor_gpu_probe_dispatch_sequence;
        if (texture_cache.ExecutorGpuContentProbeDispatchSelected(executor_gpu_probe_dispatch)) {
            for (const auto& probe : executor_gpu_probe_storage_images) {
                texture_cache.ExecutorGpuContentProbe(
                    probe.image_id,
                    VideoCore::TextureCache::ExecutorGpuContentRole::StorageAfterDispatch,
                    probe.stage, probe.stage_hash, executor_gpu_probe_dispatch);
            }
        }
    }

    ResetBindings();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();
#ifdef __ANDROID__
    ++executor_completion_work_serial;
    if (AmdGpu::RenderWaveTrace::LateFrameCaptureActive()) {
        ExecutorGpuLatePassTransition(0, executor_gpu_probe_dispatch_sequence + 1,
                                      "compute_indirect");
    }
#endif

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline || !pipeline->IsValid()) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    pipeline->BindResources(set_writes, buffer_barriers, push_data);

    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->Handle());
    cmdbuf.dispatchIndirect(buffer->Handle(), base);

#ifdef __ANDROID__
    // Keep the journal identity on the dispatch command buffer even when the optional probe below
    // submits it synchronously.
    ExecutorVkJournalComputeDispatch(
        &scheduler, scheduler.CurrentTick(), pipeline->GetPipelineHash(),
        ExecutorVkPipelineHandle(pipeline->Handle()), pipeline->GetPipelineHash(), true, 0, 0, 0,
        address + offset, static_cast<u32>(set_writes.size()),
        static_cast<u32>(buffer_infos.size()), static_cast<u32>(image_infos.size()),
        texture_cache.ExecutorGpuContentProbeEnabled());
#endif

    if (texture_cache.ExecutorGpuContentProbeEnabled()) {
        const u64 executor_gpu_probe_dispatch = ++executor_gpu_probe_dispatch_sequence;
        if (texture_cache.ExecutorGpuContentProbeDispatchSelected(executor_gpu_probe_dispatch)) {
            for (const auto& probe : executor_gpu_probe_storage_images) {
                texture_cache.ExecutorGpuContentProbe(
                    probe.image_id,
                    VideoCore::TextureCache::ExecutorGpuContentRole::StorageAfterDispatch,
                    probe.stage, probe.stage_hash, executor_gpu_probe_dispatch);
            }
        }
    }

    ResetBindings();
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    // The guest submit tail reaches this even when Presenter already flushed the shared draw
    // scheduler. Use Scheduler's empty-command-buffer guard instead of unconditionally emitting a
    // second host vkQueueSubmit with no work.
    scheduler.Flush();
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish(Scheduler::ExecutorFinishReason::Rasterizer);
}

void Rasterizer::OnSubmit() {
#ifdef __ANDROID__
    if (executor_unpublished_actual_draws != 0) {
        Libraries::GnmDriver::ExecutorGnmRecordActualDraw(executor_unpublished_actual_draws);
        executor_unpublished_actual_draws = 0;
    }
#endif
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
#ifdef __ANDROID__
    // Pixel-proof: at submit end (all draws recorded, rendering scope closes here) read back the last
    // bound color target with a proper COLOR_ATTACHMENT->TRANSFER_SRC transition + Finish, then count
    // non-zero pixels. Correct timing per Codex (not per-draw inside the open render pass).
    if (Libraries::GnmDriver::ExecutorReplayActive() &&
        texture_cache.executor_replay_color_target) {
        texture_cache.ExecutorReplayReadbackColorTarget();
    }
#endif
    texture_cache.ProcessDownloadImages();
    std::optional<size_t> sampled_device_memory;
    if (instance.CanReportMemoryUsage()) {
        // vkGetPhysicalDeviceMemoryProperties2 is a driver round-trip. Both caches used to issue it
        // independently on every guest submit (roughly 80 calls/s in Bloodborne). One shared sample
        // every 16 submits reacts to pressure within a fraction of a second while removing the
        // fixed per-submit cost from the gameplay path.
        constexpr u64 MemoryBudgetSampleInterval = 16;
        if (!memory_budget_sample_valid ||
            (memory_budget_submit_counter++ % MemoryBudgetSampleInterval) == 0) {
            cached_device_memory_usage = instance.GetDeviceMemoryUsage();
            memory_budget_sample_valid = true;
        }
        sampled_device_memory = cached_device_memory_usage;
    }
    texture_cache.RunGarbageCollector(sampled_device_memory);
    buffer_cache.RunGarbageCollector(sampled_device_memory);
    pipeline_cache.AdvanceShaderIdentitySubmit();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();
    executor_gpu_probe_sampled_images.clear();
    executor_gpu_probe_storage_images.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        stage->PushUd(binding, push_data);
        BindBuffers(*stage, binding, push_data);
        BindTextures(*stage, binding);
        uses_dma |= stage->uses_dma;
    }

    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
        fault_process_pending = true;
    }

    return true;
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(info).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    buffer_bindings.clear();
#ifdef __ANDROID__
    const bool exec_live_resource_diag =
        ExecutorTraceLivePresentResourcesEnabled() &&
        !Libraries::GnmDriver::ExecutorReplayActive();
    static int exec_live_cbuf_log_budget = 96;
#endif

    for (const auto& desc : stage.buffers) {
        auto vsharp = desc.GetSharp(stage);
        // EXECUTOR: V#/constant buffer descriptors are assembled into flattened_ud_buf at recompile
        // time and never pass through DCB/table relocation, so the base is still the captured (old) VA.
        // Translate it to the rebased arena address so FindBuffer/ObtainBuffer read real bytes (same
        // class of fix as the T# image base in BindTextures). base_address is a raw 44-bit byte address.
        if (Libraries::GnmDriver::ExecutorReplayActive() && !desc.IsSpecial() &&
            vsharp.base_address != 0) {
            const u64 nb = Libraries::GnmDriver::ExecutorReplayRelocate(vsharp.base_address);
            if (nb) {
                vsharp.base_address = nb;
            }
#ifdef __ANDROID__
            // Log PS/VS constant buffer content -- if a material constant the PS multiplies by is zero,
            // the textured fragment output is black even though the texture image is non-zero.
            const u64 cb = vsharp.base_address;
            u32 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            if (Libraries::GnmDriver::ExecutorReplayIsRelocated(cb)) {
                const u32* p = reinterpret_cast<const u32*>(cb);
                c0 = p[0]; c1 = p[1]; c2 = p[2]; c3 = p[3];
            }
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_CBUF] stage=%u base=0x%llx size=%u stride=%u buf=%08x %08x "
                                "%08x %08x",
                                (unsigned)stage.stage, (unsigned long long)cb, (unsigned)vsharp.GetSize(),
                                (unsigned)vsharp.stride, c0, c1, c2, c3);
#endif
        }
#ifdef __ANDROID__
        if (exec_live_resource_diag && exec_live_cbuf_log_budget > 0 && !desc.IsSpecial() &&
            vsharp.base_address != 0) {
            u32 c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            const u64 cb = vsharp.base_address;
            const u32 sample_size = std::min<u32>(16u, vsharp.GetSize());
            const bool mapped = sample_size != 0 && memory->IsValidMapping(cb, sample_size);
            if (mapped) {
                u8 tmp[16]{};
                memory->CopySparseMemory(cb, tmp, sample_size);
                std::memcpy(&c0, tmp + 0, sizeof(c0));
                std::memcpy(&c1, tmp + 4, sizeof(c1));
                std::memcpy(&c2, tmp + 8, sizeof(c2));
                std::memcpy(&c3, tmp + 12, sizeof(c3));
            }
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_LIVE_CBUF] stage=%u sharpIdx=%u type=%u base=0x%llx "
                                "size=%u stride=%u mapped=%d first=%08x %08x %08x %08x",
                                (unsigned)stage.stage, (unsigned)desc.sharp_idx,
                                (unsigned)desc.buffer_type, (unsigned long long)cb,
                                (unsigned)vsharp.GetSize(), (unsigned)vsharp.stride,
                                mapped ? 1 : 0, c0, c1, c2, c3);
            --exec_live_cbuf_log_budget;
        }
#endif
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
        }
    }

    // Second pass to re-bind buffers that were updated after binding
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                const auto& cs_program = liverpool->GetCsRegs();
                const u64 lds_size = static_cast<u64>(cs_program.SharedMemSize()) *
                                     static_cast<u64>(cs_program.NumWorkgroups());

                // SharedMemoryToStorage gives every workgroup a disjoint slice.  The ordinary
                // utility ring is large enough for most dispatches, but a legal dispatch can
                // exceed it on a host with a small native-LDS limit.  Do not turn that case into
                // memset(nullptr): allocate one deferred stream buffer with the same lifetime
                // contract as the ring.
                auto& utility_lds =
                    buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                std::unique_ptr<VideoCore::StreamBuffer> dedicated_lds;
                VideoCore::StreamBuffer* lds_buffer = &utility_lds;
                if (lds_size > utility_lds.SizeBytes()) {
                    dedicated_lds = std::make_unique<VideoCore::StreamBuffer>(
                        instance, scheduler, VideoCore::MemoryUsage::Stream, lds_size);
                    lds_buffer = dedicated_lds.get();
                }

                const auto [data, offset] = lds_buffer->Map(lds_size, alignment);
                ASSERT_MSG(data != nullptr,
                           "Failed to reserve {} bytes for shared-memory storage fallback",
                           lds_size);
                std::memset(data, 0, lds_size);
                // Commit flushes non-coherent host writes, advances the ring and attaches the
                // allocation to the current scheduler tick.  Omitting it made consecutive
                // dispatches reuse and overwrite the same range.
                lds_buffer->Commit();
                buffer_infos.emplace_back(lds_buffer->Handle(), offset, lds_size);
                if (auto barrier = lds_buffer->GetBarrier(
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        static_cast<u32>(offset))) {
                    buffer_barriers.emplace_back(*barrier);
                }
                if (dedicated_lds) {
                    scheduler.DeferOperation(
                        [buffer = std::move(dedicated_lds)]() mutable { buffer.reset(); });
                }
            } else if (instance.IsNullDescriptorSupported()) {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            } else {
                auto& null_buffer = buffer_cache.GetBuffer(VideoCore::NULL_BUFFER_ID);
                buffer_infos.emplace_back(null_buffer.Handle(), 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id);
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            bool bound_aligned_scratch = false;
#ifdef __ANDROID__
            // PS4 buffer descriptors may begin at any dword while Vulkan requires storage-buffer
            // descriptor offsets to satisfy minStorageBufferOffsetAlignment (64 bytes on Adreno).
            // The normal path aligns the descriptor down and adds the missing byte offset in the
            // shader. Diagnostic A/B only: the working Android reference binds the aligned cached buffer
            // and supplies the byte compensation through PushData even on Qualcomm; never select
            // this expensive exact-base copy from a vendor/title/shader heuristic.
            const bool needs_aligned_scratch =
                is_storage && !desc.is_written && adjust != 0 &&
                ExecutorAlignedSsboScratchEnabled();
            if (needs_aligned_scratch) {
                auto& stream_scratch =
                    buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                // Small CPU-dirty inputs may themselves be returned from stream_buffer. Vulkan
                // forbids overlapping copies within one buffer, so use the independent upload ring
                // as destination in that case.
                auto& scratch = vk_buffer->Handle() == stream_scratch.Handle()
                                    ? buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Upload)
                                    : stream_scratch;
                const auto [scratch_data, scratch_offset] = scratch.Map(size, alignment);
                if (scratch_data != nullptr) {
                    // Reserve the ring range before recording a GPU write into it.  Commit only
                    // advances/flushed the host-visible allocation; the authoritative contents are
                    // supplied by copyBuffer below.
                    scratch.Commit();
                    scheduler.EndRendering();
                    const auto cmdbuf = scheduler.CommandBuffer();

                    boost::container::small_vector<vk::BufferMemoryBarrier2, 2> pre_barriers;
                    if (auto barrier = vk_buffer->GetBarrier(
                            vk::AccessFlagBits2::eTransferRead,
                            vk::PipelineStageFlagBits2::eTransfer, offset)) {
                        pre_barriers.emplace_back(*barrier);
                    }
                    if (auto barrier = scratch.GetBarrier(
                            vk::AccessFlagBits2::eTransferWrite,
                            vk::PipelineStageFlagBits2::eTransfer,
                            static_cast<u32>(scratch_offset))) {
                        pre_barriers.emplace_back(*barrier);
                    }
                    if (!pre_barriers.empty()) {
                        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                            .bufferMemoryBarrierCount =
                                static_cast<u32>(pre_barriers.size()),
                            .pBufferMemoryBarriers = pre_barriers.data(),
                        });
                    }
                    cmdbuf.copyBuffer(vk_buffer->Handle(), scratch.Handle(),
                                      vk::BufferCopy{
                                          .srcOffset = offset,
                                          .dstOffset = scratch_offset,
                                          .size = size,
                                      });
                    if (auto barrier = scratch.GetBarrier(
                            vk::AccessFlagBits2::eShaderRead,
                            vk::PipelineStageFlagBits2::eAllCommands,
                            static_cast<u32>(scratch_offset))) {
                        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                            .bufferMemoryBarrierCount = 1,
                            .pBufferMemoryBarriers = &*barrier,
                        });
                    }

                    buffer_infos.emplace_back(scratch.Handle(), scratch_offset, size);
                    bound_aligned_scratch = true;
                    static std::atomic<u32> scratch_log_budget{64};
                    u32 remaining = scratch_log_budget.load(std::memory_order_relaxed);
                    while (remaining != 0 &&
                           !scratch_log_budget.compare_exchange_weak(
                               remaining, remaining - 1, std::memory_order_relaxed)) {
                    }
                    if (remaining != 0) {
                        __android_log_print(
                            ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_ALIGNED_SSBO_SCRATCH] binding=%u guest=0x%llx bytes=%llu "
                            "sourceOffset=0x%x originalAdjust=0x%x scratchOffset=0x%llx align=%u",
                            binding.buffer,
                            static_cast<unsigned long long>(vsharp.base_address),
                            static_cast<unsigned long long>(size), offset, adjust,
                            static_cast<unsigned long long>(scratch_offset), alignment);
                    }
                }
            }
#endif
            if (!bound_aligned_scratch) {
                push_data.AddOffset(binding.buffer, adjust);
                buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
                if (auto barrier =
                        vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                              : vk::AccessFlagBits2::eShaderRead,
                                              vk::PipelineStageFlagBits2::eAllCommands)) {
                    buffer_barriers.emplace_back(*barrier);
                }
            }
            if (desc.is_written && desc.is_formatted) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        set_writes.push_back({
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = is_storage ? vk::DescriptorType::eStorageBuffer
                                         : vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &buffer_infos.back(),
        });
        ++binding.buffer;
    }
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // Hosts without SPV_AMD_shader_image_load_store_lod emulate storage-mip access by binding one
    // view per mip. This records the descriptor-array width for each logical image resource.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
#ifdef __ANDROID__
    const bool exec_live_resource_diag =
        ExecutorTraceLivePresentResourcesEnabled() &&
        !Libraries::GnmDriver::ExecutorReplayActive();
    static int exec_live_tsharp_log_budget = 96;
    static int exec_replay_tsharp_log_budget = 512;
#endif

    u32 image_slot = 0;
    for (const auto& image_desc : stage.images) {
        auto tsharp = image_desc.GetSharp(stage);
        u64 old_byte = static_cast<u64>(tsharp.base_address) << 8;
        u64 new_byte = 0;
        bool reloc_hit = false;
        // EXECUTOR: the T# image descriptor is assembled into flattened_ud_buf at recompile time and so
        // never passes through DCB/table relocation; its base is still the captured (old) texture VA.
        // Translate it to the rebased arena address here so FindImage/RefreshImage read real bytes.
        if (Libraries::GnmDriver::ExecutorReplayActive() && tsharp.base_address != 0) {
            new_byte = Libraries::GnmDriver::ExecutorReplayRelocate(old_byte);
            reloc_hit = new_byte != 0;
            if (new_byte) {
                tsharp.base_address = new_byte >> 8;
            }
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_TSHARP] sharpIdx=%u old=0x%llx new=0x%llx relocHit=%d fmt=%u "
                                "w=%u h=%u",
                                (unsigned)image_desc.sharp_idx, (unsigned long long)old_byte,
                                (unsigned long long)new_byte, new_byte ? 1 : 0,
                                static_cast<u32>(tsharp.GetDataFmt()),
                                (unsigned)tsharp.width, (unsigned)tsharp.height);
#endif
        }
#ifdef __ANDROID__
        if (exec_live_resource_diag && exec_live_tsharp_log_budget > 0) {
            const u64 byte_addr = static_cast<u64>(tsharp.base_address) << 8;
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_LIVE_TSHARP] stage=%u hash=0x%llx sharpIdx=%u base=0x%llx fmt=%u "
                                "w=%u h=%u pitch=%u mips=%u valid=%d",
                                (unsigned)stage.stage, (unsigned long long)stage.pgm_hash,
                                (unsigned)image_desc.sharp_idx,
                                (unsigned long long)byte_addr,
                                static_cast<u32>(tsharp.GetDataFmt()), (unsigned)tsharp.width,
                                (unsigned)tsharp.height, (unsigned)tsharp.pitch,
                                (unsigned)tsharp.NumLevels(), tsharp.Valid() ? 1 : 0);
            --exec_live_tsharp_log_budget;
        }
#endif
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        // A descriptor with a null guest address is unbound even when its remaining format bits
        // are populated. Match upstream shadPS4 and route it through the null-descriptor path
        // instead of asking the texture cache to create an image backed by address zero.
        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
#ifdef __ANDROID__
            if (Libraries::GnmDriver::ExecutorReplayActive() &&
                (exec_replay_tsharp_log_budget > 0 || image_desc.sharp_idx == 16)) {
                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                    "[EXECUTOR_BIND_IMAGE] imageSlot=%u sharpIdx=%u oldBase=0x%llx "
                                    "newBase=0x%llx relocHit=%d fmt=%u w=%u h=%u imageId=0 "
                                    "readbackNonzero=0 nonzeroBytes=0 invalidFmt=%u nullAddr=%u",
                                    (unsigned)image_slot, (unsigned)image_desc.sharp_idx,
                                    (unsigned long long)old_byte, (unsigned long long)new_byte,
                                    reloc_hit ? 1 : 0, static_cast<u32>(tsharp.GetDataFmt()),
                                    (unsigned)tsharp.width, (unsigned)tsharp.height,
                                    tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid ? 1u
                                                                                           : 0u,
                                    tsharp.Address() == 0 ? 1u : 0u);
                --exec_replay_tsharp_log_budget;
            }
#endif
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_descriptor_array_sizes.push_back(1);
            ++image_slot;
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);

        for (u32 mip_index = 0; mip_index < num_bindings; ++mip_index) {
            auto& [image_id, desc] = image_bindings.emplace_back(
                std::piecewise_construct, std::tuple{}, std::tuple{tsharp, image_desc});

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += mip_index;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImage(desc);
            auto* image = &texture_cache.GetImage(image_id);
#ifdef __ANDROID__
            if (exec_live_resource_diag && exec_live_tsharp_log_budget > 0) {
                __android_log_print(
                    ANDROID_LOG_INFO, "EXECUTOR",
                    "[EXECUTOR_LIVE_IMAGE_BIND] stage=%u hash=0x%llx slot=%u sharpIdx=%u "
                    "imageId=%u guest=0x%llx bytes=0x%x imageFmt=%u extent=%ux%ux%u "
                    "pitch=%u levels=%u layers=%u tile=%u flags=0x%x block=%u "
                    "usageTexture=%u usageStorage=%u usageRt=%u",
                    (unsigned)stage.stage, (unsigned long long)stage.pgm_hash,
                    (unsigned)image_slot, (unsigned)image_desc.sharp_idx,
                    (unsigned)image_id.index,
                    (unsigned long long)image->info.guest_address,
                    (unsigned)image->info.guest_size,
                    (unsigned)image->info.pixel_format,
                    (unsigned)image->info.size.width, (unsigned)image->info.size.height,
                    (unsigned)image->info.size.depth, (unsigned)image->info.pitch,
                    (unsigned)image->info.resources.levels,
                    (unsigned)image->info.resources.layers,
                    (unsigned)image->info.tile_mode, (unsigned)image->flags,
                    image->info.props.is_block ? 1u : 0u, image->usage.texture,
                    image->usage.storage, image->usage.render_target);
            }
#endif
#ifdef __ANDROID__
            if (Libraries::GnmDriver::ExecutorReplayActive() &&
                (exec_replay_tsharp_log_budget > 0 || image_desc.sharp_idx == 16)) {
                const u64 nonzero_bytes = ExecutorCountNonzeroSparse(
                    memory, image->info.guest_address, image->info.guest_size);
                __android_log_print(
                    ANDROID_LOG_INFO, "EXECUTOR",
                    "[EXECUTOR_BIND_IMAGE] imageSlot=%u sharpIdx=%u oldBase=0x%llx "
                    "newBase=0x%llx relocHit=%d fmt=%u w=%u h=%u imageId=%u "
                    "readbackNonzero=%d nonzeroBytes=%llu guest=0x%llx guestSize=%llu "
                    "pitch=%u levels=%u layers=%u type=%u baseLevel=%u lastLevel=%u "
                    "baseArray=%u lastArray=%u dstSel=%u,%u,%u,%u isArray=%d",
                    (unsigned)image_slot, (unsigned)image_desc.sharp_idx,
                    (unsigned long long)old_byte, (unsigned long long)new_byte,
                    reloc_hit ? 1 : 0, static_cast<u32>(tsharp.GetDataFmt()),
                    (unsigned)tsharp.width, (unsigned)tsharp.height,
                    static_cast<unsigned>(image_id.index), nonzero_bytes ? 1 : 0,
                    (unsigned long long)nonzero_bytes,
                    (unsigned long long)image->info.guest_address,
                    (unsigned long long)image->info.guest_size, (unsigned)image->info.pitch,
                    (unsigned)image->info.resources.levels,
                    (unsigned)image->info.resources.layers, (unsigned)tsharp.type,
                    (unsigned)tsharp.base_level, (unsigned)tsharp.last_level,
                    (unsigned)tsharp.base_array, (unsigned)tsharp.last_array,
                    (unsigned)tsharp.dst_sel_x, (unsigned)tsharp.dst_sel_y,
                    (unsigned)tsharp.dst_sel_z, (unsigned)tsharp.dst_sel_w,
                    image_desc.is_array ? 1 : 0);
                --exec_replay_tsharp_log_budget;
            }
#endif
            // EXECUTOR (Codex phase-2): capture the sampled texture's guest memory so replay can
            // rebase it (else texture_cache.FindImage reads an un-relocated texture on device).
            if (Libraries::GnmDriver::ExecutorGnmCaptureActive()) {
                Libraries::GnmDriver::ExecutorGnmCaptureRecordResource(
                    4u, image->info.guest_address, image->info.guest_size);
            }
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we need
                // to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
            ++image_slot;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // Second pass to re-bind images that were updated after binding
    for (auto& [image_id, desc] : image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            if (instance.IsNullDescriptorSupported()) {
                image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            } else {
                auto& null_image_view = texture_cache.FindTexture(VideoCore::NULL_IMAGE_ID, desc);
                image_infos.emplace_back(VK_NULL_HANDLE, *null_image_view.image_view,
                                         vk::ImageLayout::eGeneral);
            }
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(image_id, desc);

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
            } else {
                if (is_storage) {
                    image.Transit(vk::ImageLayout::eGeneral,
                                  vk::AccessFlagBits2::eShaderRead |
                                      vk::AccessFlagBits2::eShaderWrite,
                                  desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range);
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            if (texture_cache.ExecutorGpuContentProbeEnabled()) {
                auto& probes = is_storage ? executor_gpu_probe_storage_images
                                          : executor_gpu_probe_sampled_images;
                bool already_recorded = false;
                for (const auto& probe : probes) {
                    if (probe.image_id == image_id &&
                        probe.stage == static_cast<u32>(stage.stage) &&
                        probe.stage_hash == stage.pgm_hash) {
                        already_recorded = true;
                        break;
                    }
                }
                if (!already_recorded) {
                    probes.push_back({image_id, static_cast<u32>(stage.stage), stage.pgm_hash});
                }
            }

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
        }
    }

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (const u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        set_writes.push_back({
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified,
            .dstArrayElement = 0,
            .descriptorCount = array_size,
            .descriptorType =
                is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage,
            .pImageInfo = &image_infos[image_info_idx],
        });

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    u32 sampler_slot = 0;
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(stage);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
#ifdef __ANDROID__
        if (Libraries::GnmDriver::ExecutorReplayActive()) {
            const u32 associated_image = sampler.associated_image;
            const u32 associated_sharp =
                associated_image < stage.images.size() ? stage.images[associated_image].sharp_idx
                                                       : std::numeric_limits<u32>::max();
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_BIND_SAMPLER] sampSlot=%u ssharpIdx=%u associatedImage=%u "
                                "associatedSharpIdx=%u clampX=%u clampY=%u clampZ=%u minFilter=%u "
                                "magFilter=%u mipFilter=%u filterMode=%u border=%u forceUnnorm=%u "
                                "minLod=%.2f maxLod=%.2f lodBias=%.2f disableAniso=%u raw0=0x%llx "
                                "raw1=0x%llx",
                                (unsigned)sampler_slot, (unsigned)sampler.sharp_idx,
                                (unsigned)associated_image, (unsigned)associated_sharp,
                                (unsigned)ssharp.clamp_x.Value(), (unsigned)ssharp.clamp_y.Value(),
                                (unsigned)ssharp.clamp_z.Value(),
                                (unsigned)ssharp.xy_min_filter.Value(),
                                (unsigned)ssharp.xy_mag_filter.Value(),
                                (unsigned)ssharp.mip_filter.Value(),
                                (unsigned)ssharp.filter_mode.Value(),
                                (unsigned)ssharp.border_color_type.Value(),
                                (unsigned)ssharp.force_unnormalized.Value(), ssharp.MinLod(),
                                ssharp.MaxLod(), ssharp.LodBias(), sampler.disable_aniso,
                                (unsigned long long)ssharp.raw0, (unsigned long long)ssharp.raw1);
        }
#endif
        const auto vk_sampler = texture_cache.GetSampler(ssharp, liverpool->regs.ta_bc_base);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        set_writes.push_back({
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eSampler,
            .pImageInfo = &image_infos.back(),
        });
        ++sampler_slot;
    }
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->regs;
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state{};
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
#ifdef __ANDROID__
    if (ExecutorTracePm4Enabled())
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_DRAW_TRACE] br_enter numColor=%u", state.num_color_attachments);
#endif
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            continue;
        }
#ifdef __ANDROID__
        if (ExecutorTracePm4Enabled())
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_DRAW_TRACE] br_getimage cb=%u", cb);
#endif
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
#ifdef __ANDROID__
        if (ExecutorTracePm4Enabled())
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_DRAW_TRACE] br_before_updateimage cb=%u", cb);
#endif
        texture_cache.UpdateImage(image_id);
#ifdef __ANDROID__
        if (ExecutorTracePm4Enabled())
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_DRAW_TRACE] br_before_findrt cb=%u", cb);
#endif
        image->SetBackingSamples(key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
#ifdef __ANDROID__
        if (ExecutorTracePm4Enabled())
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_DRAW_TRACE] br_rt_ok cb=%u", cb);
#endif
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);
        // EXECUTOR replay diagnostic: force a Clear to a known red so the readback can confirm it reads
        // the same image the draws render to. If RT_READBACK is still zero, the readback target is wrong;
        // if it's red, the draws' fragment output (blend/shader) is what produces zero. Gated by env.
        const bool exec_clear_probe =
            Libraries::GnmDriver::ExecutorReplayActive() && std::getenv("EXECUTOR_CLEAR_PROBE");
        const auto clear_value =
            exec_clear_probe
                ? vk::ClearValue{.color = {.float32 = {{1.0f, 0.0f, 0.0f, 1.0f}}}}
                : (is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{});
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear || exec_clear_probe;
        image->usage.render_target = 1u;
        const bool executor_track_present_rt = Libraries::GnmDriver::ExecutorReplayActive();
        if (executor_track_present_rt && cb == 0) {
            texture_cache.executor_replay_color_target = image_id;  // fallback (last-bound)
            // Track cb0 bind order, including repeated binds. Replay frames often visit several
            // full-resolution RTs; the first can be only a background/clear, while a later bind is the
            // composite to present. The picker collapses readback work but uses this order for ties.
            {
                std::scoped_lock lk{texture_cache.executor_replay_candidates_mutex};
                texture_cache.executor_replay_color_candidates.push_back(image_id);
            }
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_RT_CAND] mode=%s push id=%u count=%zu extent=%ux%u base=0x%llx",
                                Libraries::GnmDriver::ExecutorReplayActive() ? "replay" : "live",
                                image_id.index, texture_cache.executor_replay_color_candidates.size(),
                                image->info.size.width, image->info.size.height,
                                (unsigned long long)image->info.guest_address);
#endif
        }
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    // EXECUTOR (Codex phase-2): during replay, skip the depth attachment. The depth target's image-view
    // creation / htile path can read un-captured metadata and crashes; a first-pixel replay does not need
    // the depth test, and skipping it lets the (depth-only z-prepass) draw and the later color draws run.
    // EXECUTOR: depth attachment still skipped during replay -- the depth target's de-tile/creation has
    // a separate issue (crashes in BeginRendering even with the present-thread race fixed). Tackle after
    // the readback-timing/zero-pixel diagnosis (Codex priority order).
    if (auto image_id = db_desc.first; image_id) {  // depth temporarily enabled to localize FDT crash
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        auto& image = texture_cache.GetImage(image_id);

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear = regs.depth_render_control.depth_clear_enable ||
                                    texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can remain enabled while depth is read-only. Using a fully read-only
        // depth/stencil layout in that state is invalid and can make strict mobile drivers lose
        // the device at queue submit. Match the PC renderer's mixed-aspect layout contract.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);
#ifdef __ANDROID__
        if (Libraries::GnmDriver::ExecutorReplayActive()) __android_log_print(ANDROID_LOG_INFO,"EXECUTOR","[EXECUTOR_DRAW_TRACE] depth_after_transit");
#endif

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);
        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            // EXECUTOR replay: the depth image content isn't uploaded, so preserve the existing replay
            // clear policy while using the normalized v0.16 render-state representation.
            const bool depth_clear =
                is_depth_clear || Libraries::GnmDriver::ExecutorReplayActive();
            attachment.clear_value[0] = depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    const auto mrt0_id = texture_cache.FindImage(mrt0_desc, true);
    const auto mrt1_id = texture_cache.FindImage(mrt1_desc, true);
    auto& mrt0_image = texture_cache.GetImage(mrt0_id);
    auto& mrt1_image = texture_cache.GetImage(mrt1_id);

#ifdef __ANDROID__
    // FlusterCluck's final display operation is a GCN resolve, not a shader draw/copy: the PC oracle
    // binds the full-resolution MSAA scene as MRT0 and the registered VideoOut buffer as MRT1, then
    // emits a DrawIndexAuto while CB_COLOR_CONTROL is Resolve. Keep this trace independent from the
    // very expensive all-PM4 trace so one normal production run proves whether that exact tail packet
    // survives the guest DCB boundary. Sampling also keeps a 60 fps resolve loop cheap.
    static const bool trace_resolve_pair =
        std::getenv("EXECUTOR_TRACE_VIDEOOUT_SOURCE") != nullptr ||
        std::getenv("EXECUTOR_TRACE_RESOLVE_PAIR") != nullptr;
    static std::atomic<u64> resolve_sequence{0};
    const u64 resolve_seq = resolve_sequence.fetch_add(1, std::memory_order_relaxed);
    const bool log_resolve =
        trace_resolve_pair && (resolve_seq < 12 || (resolve_seq % 60) == 0);
    if (log_resolve) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_RESOLVE_PAIR] seq=%llu phase=before srcId=%u src=0x%llx "
            "srcExtent=%ux%u srcInfoSamples=%u srcBackingSamples=%u srcFlags=0x%x "
            "dstId=%u dst=0x%llx dstExtent=%ux%u dstInfoSamples=%u "
            "dstBackingSamples=%u dstFlags=0x%x dstUsageVo=%u dstUsageRt=%u",
            static_cast<unsigned long long>(resolve_seq), mrt0_id.index,
            static_cast<unsigned long long>(mrt0_image.info.guest_address),
            mrt0_image.info.size.width, mrt0_image.info.size.height, mrt0_image.info.num_samples,
            mrt0_image.backing ? mrt0_image.backing->num_samples : 0u,
            static_cast<u32>(mrt0_image.flags), mrt1_id.index,
            static_cast<unsigned long long>(mrt1_image.info.guest_address),
            mrt1_image.info.size.width, mrt1_image.info.size.height, mrt1_image.info.num_samples,
            mrt1_image.backing ? mrt1_image.backing->num_samples : 0u,
            static_cast<u32>(mrt1_image.flags), mrt1_image.usage.vo_surface,
            mrt1_image.usage.render_target);
    }
#endif

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
#ifdef __ANDROID__
    // Unity presents the resolve destination, not the multisampled draw target.  The normal
    // VideoOut path already finds this exact MRT1 image by its registered guest address; keep the
    // opt-in Android direct-present fallback on the same semantic source.  Otherwise it keeps
    // presenting the last vkCmdDraw target and, after its first successful present suppresses
    // VideoOut, which leaves a stale logo/flash while subsequent resolved frames are ignored.
    if (Libraries::GnmDriver::ExecutorLivePresentRtEnabled()) {
        texture_cache.ExecutorTrackLiveDrawnColorTarget(mrt1_id);
    }
    if (log_resolve) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_RESOLVE_PAIR] seq=%llu phase=recorded srcId=%u srcBackingSamples=%u "
            "dstId=%u dstBackingSamples=%u dstFlags=0x%x",
            static_cast<unsigned long long>(resolve_seq), mrt0_id.index,
            mrt0_image.backing ? mrt0_image.backing->num_samples : 0u, mrt1_id.index,
            mrt1_image.backing ? mrt1_image.backing->num_samples : 0u,
            static_cast<u32>(mrt1_image.flags));
    }
#endif
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.CommandBuffer().copyImage(read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                                        write_image.GetImage(),
                                        vk::ImageLayout::eTransferDstOptimal, region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
#ifdef __ANDROID__
    ++executor_completion_work_serial;
#endif
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
#ifdef __ANDROID__
    ++executor_completion_work_serial;
#endif
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    pipeline_cache.InvalidateShaderIdentityRange(addr, size);
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
#ifdef __ANDROID__
    // A CPU read fault can target a fence whose Vulkan commands and host publication callback are
    // still held in Liverpool's current deferred batch. Resolve that dependency first. The normal
    // buffer-cache path below is still required to restore its page watcher and synchronize any
    // unrelated GPU bytes sharing the page.
    (void)liverpool->ResolvePendingCompletionReadForExecutor(addr, size);
#endif
    // A precise GPU->guest readback is also a precise shader-identity invalidation boundary.
    // Keeping this range local lets immutable shader metadata survive unrelated readbacks without
    // ever reusing a hash after GPU-written code bytes have been published to guest memory.
    pipeline_cache.InvalidateShaderIdentityReadbackRange(addr, size);
    buffer_cache.ReadMemory(addr, size);
    return true;
}

void Rasterizer::TrackPendingCompletionRead(VAddr addr, u64 size) {
    page_manager.TrackPendingCompletionRead(addr, size);
}

void Rasterizer::UntrackPendingCompletionRead(VAddr addr, u64 size) {
    page_manager.UntrackPendingCompletionRead(addr, size);
}

bool Rasterizer::IsMemoryGpuModified(VAddr addr, u64 size) {
    return IsMapped(addr, size) && buffer_cache.IsRegionGpuModified(addr, size);
}

void Rasterizer::ProcessDownloadImages() {
#ifdef __ANDROID__
    ++executor_completion_work_serial;
#endif
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    pipeline_cache.InvalidateShaderIdentityRange(addr, size);
    pipeline_cache.InvalidateSrtSnapshots();
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.Commit(instance, scheduler.CommandBuffer());
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->regs;

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        dynamic_state.SetStencilReferences(front.stencil_test_val, back.stencil_test_val);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
#ifndef __ANDROID__
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
#endif
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !Config::getVkGuestMarkersEnabled()) ||
        (!from_guest && !Config::getVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !Config::getVkGuestMarkersEnabled()) ||
        (!from_guest && !Config::getVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !Config::getVkGuestMarkersEnabled()) ||
        (!from_guest && !Config::getVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !Config::getVkGuestMarkersEnabled()) ||
        (!from_guest && !Config::getVkHostMarkersEnabled())) {
        return;
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
