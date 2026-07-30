// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
#include "core/libraries/gnmdriver/gnmdriver.h"
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <charconv>
#include <cstdlib>
#include <ranges>
#ifdef __ANDROID__
#include <android/log.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "common/config.h"
#include "common/elf_info.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_driver_optimization.h"

namespace Vulkan {

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

namespace {

template <typename T>
u64 HashGraphicsStructuralValue(const u64 seed, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return HashCombine(seed, XXH3_64bits(&value, sizeof(value)));
}

u64 GraphicsStructuralSignature(
    const AmdGpu::Regs& regs,
    const std::optional<AmdGpu::PrimitiveType> host_primitive_override) {
    u64 signature = 0x4c53583447505331ULL;
    const auto mix = [&](const auto& value) {
        signature = HashGraphicsStructuralValue(signature, value);
    };

    // Shader addresses/settings and every register consumed by
    // BuildRuntimeInfo or RefreshGraphicsKey. Resource addresses and ordinary
    // viewport/attachment addresses are deliberately excluded: they do not
    // select a Vulkan pipeline and used to trigger a full shader-stage walk.
    const auto mix_program = [&](const AmdGpu::ShaderProgram& program) {
        signature = HashCombine(
            signature,
            XXH3_64bits(&program, offsetof(AmdGpu::ShaderProgram, user_data)));
    };
    mix_program(regs.ps_program);
    mix_program(regs.vs_program);
    mix_program(regs.gs_program);
    mix_program(regs.es_program);
    mix_program(regs.hs_program);
    mix_program(regs.ls_program);

    mix(regs.depth_render_override);
    const bool depth_valid = regs.depth_buffer.DepthValid();
    const bool stencil_valid = regs.depth_buffer.StencilValid();
    mix(depth_valid);
    mix(stencil_valid);
    if (depth_valid) {
        mix(regs.depth_buffer.z_info);
    }
    if (stencil_valid) {
        mix(regs.depth_buffer.stencil_info);
    }

    mix(regs.clipper_control);
    mix(regs.polygon_control);
    mix(regs.color_control);
    mix(regs.color_target_mask);
    mix(regs.color_shader_mask);
    mix(regs.color_export_format);
    mix(regs.blend_control);
    for (const auto& color : regs.color_buffers) {
        const bool active = static_cast<bool>(color);
        mix(active);
        if (active) {
            mix(color.info.raw);
            mix(color.attrib.raw);
        }
    }

    mix(regs.stage_enable);
    mix(regs.ls_hs_config);
    mix(regs.tess_config);
    mix(regs.vgt_esgs_ring_itemsize);
    mix(regs.vgt_gs_vert_itemsize);
    mix(regs.vgt_gs_max_vert_out);
    mix(regs.vgt_gs_instance_cnt);
    mix(regs.vgt_gs_out_prim_type);
    mix(regs.vgt_gs_mode);
    mix(regs.vgt_strmout_config);
    mix(regs.vgt_instance_step_rate_0);
    mix(regs.vgt_instance_step_rate_1);
    mix(regs.vs_output_control);
    mix(regs.ps_input_ena);
    mix(regs.ps_input_addr);
    mix(regs.ps_inputs);
    mix(static_cast<u32>(regs.num_interp));
    mix(regs.z_export_format);
    mix(regs.depth_shader_control);
    mix(regs.primitive_type);
    mix(host_primitive_override);
    return signature;
}

struct AtomicAsyncPipelineCompileStats {
    std::atomic_bool fail_fast_supported{};
    std::atomic<u32> worker_count{};
    std::atomic<u32> queue_capacity{};
    std::atomic<u32> queue_depth{};
    std::atomic<u32> max_queue_depth{};
    std::atomic<u32> shader_queue_capacity{};
    std::atomic<u32> shader_queue_depth{};
    std::atomic<u32> shader_max_queue_depth{};
    std::atomic<u64> shader_queue_byte_capacity{};
    std::atomic<u64> shader_queue_bytes{};
    std::atomic<u64> shader_max_queue_bytes{};
    std::atomic<u64> graphics_misses{};
    std::atomic<u64> compute_misses{};
    std::atomic<u64> fail_fast_hits{};
    std::atomic<u64> fail_fast_compile_required{};
    std::atomic<u64> pending_hits{};
    std::atomic<u64> fast_queued{};
    std::atomic<u64> optimized_queued{};
    std::atomic<u64> queue_drops{};
    std::atomic<u64> fast_completed{};
    std::atomic<u64> optimized_completed{};
    std::atomic<u64> build_failures{};
    std::atomic<u64> published{};
    std::atomic<u64> optimized_replacements{};
    std::atomic<u64> skipped_uses{};
    std::atomic<u64> foreground_fallbacks{};
    std::atomic<u64> shader_fast_compiled{};
    std::atomic<u64> shader_optimized_queued{};
    std::atomic<u64> shader_optimized_completed{};
    std::atomic<u64> shader_optimized_published{};
    std::atomic<u64> shader_queue_drops{};
    std::atomic<u64> shader_negative_hits{};
    std::atomic<u64> shader_build_failures{};
    std::atomic<u64> shader_replacements{};
    std::atomic<u64> shader_total_build_us{};
    std::atomic<u64> shader_max_build_us{};
    std::atomic<u64> total_build_us{};
    std::atomic<u64> max_build_us{};
};

AtomicAsyncPipelineCompileStats async_pipeline_stats{};

template <typename T>
void AtomicMax(std::atomic<T>& target, const T value) noexcept {
    T previous = target.load(std::memory_order_relaxed);
    while (previous < value &&
           !target.compare_exchange_weak(previous, value, std::memory_order_relaxed)) {
    }
}

size_t ReadBoundedAsyncPipelineOption(const char* name, const size_t fallback,
                                      const size_t minimum, const size_t maximum) {
    const char* const value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    size_t parsed{};
    const std::string_view text{value};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return fallback;
    }
    return std::clamp(parsed, minimum, maximum);
}

bool DeferAsyncLiveReplacement(const Instance& instance) noexcept {
#ifdef __ANDROID__
    // Qualcomm's proprietary driver may keep internal references to the
    // currently bound shader/pipeline even after the command-buffer lifetime
    // tracked by Vulkan has ended. Replacing a live fallback is therefore not
    // a safe publication boundary on these devices: Bloodborne can continue
    // presenting its loading overlay while guest progress is effectively
    // starved. Compile and persist the optimized artifact in the background,
    // but activate it on the next title launch through the disk caches.
    return instance.GetDriverID() == vk::DriverId::eQualcommProprietary;
#else
    (void)instance;
    return false;
#endif
}

u64 AsyncSteadyMilliseconds() noexcept {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

} // Anonymous namespace

AsyncPipelineCompileStats GetAsyncPipelineCompileStats() noexcept {
    return {
        .managed_enabled = Config::managedAsyncPipelineCompilation(),
        .fail_fast_supported =
            async_pipeline_stats.fail_fast_supported.load(std::memory_order_relaxed),
        .worker_count = async_pipeline_stats.worker_count.load(std::memory_order_relaxed),
        .queue_capacity = async_pipeline_stats.queue_capacity.load(std::memory_order_relaxed),
        .queue_depth = async_pipeline_stats.queue_depth.load(std::memory_order_relaxed),
        .max_queue_depth = async_pipeline_stats.max_queue_depth.load(std::memory_order_relaxed),
        .shader_queue_capacity =
            async_pipeline_stats.shader_queue_capacity.load(std::memory_order_relaxed),
        .shader_queue_depth =
            async_pipeline_stats.shader_queue_depth.load(std::memory_order_relaxed),
        .shader_max_queue_depth =
            async_pipeline_stats.shader_max_queue_depth.load(std::memory_order_relaxed),
        .shader_queue_byte_capacity =
            async_pipeline_stats.shader_queue_byte_capacity.load(std::memory_order_relaxed),
        .shader_queue_bytes =
            async_pipeline_stats.shader_queue_bytes.load(std::memory_order_relaxed),
        .shader_max_queue_bytes =
            async_pipeline_stats.shader_max_queue_bytes.load(std::memory_order_relaxed),
        .graphics_misses = async_pipeline_stats.graphics_misses.load(std::memory_order_relaxed),
        .compute_misses = async_pipeline_stats.compute_misses.load(std::memory_order_relaxed),
        .fail_fast_hits = async_pipeline_stats.fail_fast_hits.load(std::memory_order_relaxed),
        .fail_fast_compile_required =
            async_pipeline_stats.fail_fast_compile_required.load(std::memory_order_relaxed),
        .pending_hits = async_pipeline_stats.pending_hits.load(std::memory_order_relaxed),
        .fast_queued = async_pipeline_stats.fast_queued.load(std::memory_order_relaxed),
        .optimized_queued =
            async_pipeline_stats.optimized_queued.load(std::memory_order_relaxed),
        .queue_drops = async_pipeline_stats.queue_drops.load(std::memory_order_relaxed),
        .fast_completed = async_pipeline_stats.fast_completed.load(std::memory_order_relaxed),
        .optimized_completed =
            async_pipeline_stats.optimized_completed.load(std::memory_order_relaxed),
        .build_failures = async_pipeline_stats.build_failures.load(std::memory_order_relaxed),
        .published = async_pipeline_stats.published.load(std::memory_order_relaxed),
        .optimized_replacements =
            async_pipeline_stats.optimized_replacements.load(std::memory_order_relaxed),
        .skipped_uses = async_pipeline_stats.skipped_uses.load(std::memory_order_relaxed),
        .foreground_fallbacks =
            async_pipeline_stats.foreground_fallbacks.load(std::memory_order_relaxed),
        .shader_fast_compiled =
            async_pipeline_stats.shader_fast_compiled.load(std::memory_order_relaxed),
        .shader_optimized_queued =
            async_pipeline_stats.shader_optimized_queued.load(std::memory_order_relaxed),
        .shader_optimized_completed =
            async_pipeline_stats.shader_optimized_completed.load(std::memory_order_relaxed),
        .shader_optimized_published =
            async_pipeline_stats.shader_optimized_published.load(std::memory_order_relaxed),
        .shader_queue_drops =
            async_pipeline_stats.shader_queue_drops.load(std::memory_order_relaxed),
        .shader_negative_hits =
            async_pipeline_stats.shader_negative_hits.load(std::memory_order_relaxed),
        .shader_build_failures =
            async_pipeline_stats.shader_build_failures.load(std::memory_order_relaxed),
        .shader_replacements =
            async_pipeline_stats.shader_replacements.load(std::memory_order_relaxed),
        .shader_total_build_us =
            async_pipeline_stats.shader_total_build_us.load(std::memory_order_relaxed),
        .shader_max_build_us =
            async_pipeline_stats.shader_max_build_us.load(std::memory_order_relaxed),
        .total_build_us = async_pipeline_stats.total_build_us.load(std::memory_order_relaxed),
        .max_build_us = async_pipeline_stats.max_build_us.load(std::memory_order_relaxed),
    };
}

void ResetAsyncPipelineCompileStats() noexcept {
    async_pipeline_stats.max_queue_depth.store(
        async_pipeline_stats.queue_depth.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    async_pipeline_stats.shader_max_queue_depth.store(
        async_pipeline_stats.shader_queue_depth.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    async_pipeline_stats.shader_max_queue_bytes.store(
        async_pipeline_stats.shader_queue_bytes.load(std::memory_order_relaxed),
        std::memory_order_relaxed);
    async_pipeline_stats.graphics_misses.store(0, std::memory_order_relaxed);
    async_pipeline_stats.compute_misses.store(0, std::memory_order_relaxed);
    async_pipeline_stats.fail_fast_hits.store(0, std::memory_order_relaxed);
    async_pipeline_stats.fail_fast_compile_required.store(0, std::memory_order_relaxed);
    async_pipeline_stats.pending_hits.store(0, std::memory_order_relaxed);
    async_pipeline_stats.fast_queued.store(0, std::memory_order_relaxed);
    async_pipeline_stats.optimized_queued.store(0, std::memory_order_relaxed);
    async_pipeline_stats.queue_drops.store(0, std::memory_order_relaxed);
    async_pipeline_stats.fast_completed.store(0, std::memory_order_relaxed);
    async_pipeline_stats.optimized_completed.store(0, std::memory_order_relaxed);
    async_pipeline_stats.build_failures.store(0, std::memory_order_relaxed);
    async_pipeline_stats.published.store(0, std::memory_order_relaxed);
    async_pipeline_stats.optimized_replacements.store(0, std::memory_order_relaxed);
    async_pipeline_stats.skipped_uses.store(0, std::memory_order_relaxed);
    async_pipeline_stats.foreground_fallbacks.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_fast_compiled.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_optimized_queued.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_optimized_completed.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_optimized_published.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_queue_drops.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_negative_hits.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_build_failures.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_replacements.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_total_build_us.store(0, std::memory_order_relaxed);
    async_pipeline_stats.shader_max_build_us.store(0, std::memory_order_relaxed);
    async_pipeline_stats.total_build_us.store(0, std::memory_order_relaxed);
    async_pipeline_stats.max_build_us.store(0, std::memory_order_relaxed);
}

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(Stage stage, LogicalStage l_stage) {
    auto& info = runtime_infos[u32(l_stage)];
    const auto& regs = liverpool->regs;
    const auto BuildCommon = [&](const auto& program) {
        info.num_user_data = program.settings.num_user_regs;
        info.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.num_allocated_vgprs = program.NumVgprs();
        info.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.fp_round_mode32 = program.settings.fp_round_mode32;
    };
    info.Initialize(stage);
    switch (stage) {
    case Stage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = infos[u32(Shader::LogicalStage::TessellationControl)];
        hull_info->ReadTessConstantBuffer(tess_constants);
        info.ls_info.ls_stride = tess_constants.ls_stride;
        break;
    }
    case Stage::Hull: {
        BuildCommon(regs.hs_program);
        info.hs_info.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.hs_info.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.hs_info.tess_type = regs.tess_config.type;
        info.hs_info.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;

        break;
    }
    case Stage::Export: {
        BuildCommon(regs.es_program);
        info.es_info.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Vertex: {
        BuildCommon(regs.vs_program);
        info.vs_info.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.vs_info.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.vs_info.num_outputs = MapOutputs(info.vs_info.outputs, regs.vs_output_control);
        info.vs_info.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        const auto host_primitive =
            graphics_primitive_override.value_or(regs.primitive_type);
        info.vs_info.tess_emulated_primitive =
            host_primitive == AmdGpu::PrimitiveType::RectList ||
            host_primitive == AmdGpu::PrimitiveType::QuadList;
        info.vs_info.clip_disable = regs.IsClipDisabled();
        if (l_stage == LogicalStage::TessellationEval) {
            info.es_vs_info.tess_type = regs.tess_config.type;
            info.es_vs_info.tess_topology = regs.tess_config.topology;
            info.es_vs_info.tess_partitioning = regs.tess_config.partitioning;
        }
        break;
    }
    case Stage::Geometry: {
        BuildCommon(regs.gs_program);
        auto& gs_info = info.gs_info;
        gs_info.num_outputs = MapOutputs(gs_info.outputs, regs.vs_output_control);
        gs_info.output_vertices = regs.vgt_gs_max_vert_out;
        gs_info.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            gs_info.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            gs_info.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs_info.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs_info.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs_info.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs_info.mode = regs.vgt_gs_mode.mode;
        const auto params_vc =
            GetShaderParams(regs.vs_program.user_data.data(),
                            regs.vs_program.Address<const u32*>());
        gs_info.vs_copy = params_vc.code;
        gs_info.vs_copy_hash = params_vc.hash;
        DumpShader(gs_info.vs_copy, gs_info.vs_copy_hash, Shader::Stage::Vertex, 0, "copy.bin");
        break;
    }
    case Stage::Fragment: {
        BuildCommon(regs.ps_program);
        info.fs_info.en_flags = regs.ps_input_ena;
        info.fs_info.addr_flags = regs.ps_input_addr;
        info.fs_info.num_inputs = regs.num_interp;
        info.fs_info.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.fs_info.mrtz_mask = regs.depth_shader_control.z_export_enable |
                                 (stencil_ref_export_enable << 1) |
                                 (regs.depth_shader_control.mask_export_enable << 2) |
                                 (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.fs_info.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.fs_info.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.fs_info.dual_source_blending = false;
        }
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.fs_info.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.fs_info.color_buffers[i] = graphics_key.color_buffers[i];
        }
        info.fs_info.clip_distance_emulation =
            regs.vs_output_control.clip_distance_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Local)) &&
            profile.needs_clip_distance_emulation;
        break;
    }
    case Stage::Compute: {
        const auto& cs_pgm = liverpool->GetCsRegs();
        info.num_user_data = cs_pgm.settings.num_user_regs;
        info.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.cs_info.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                       cs_pgm.num_thread_z.full};
        info.cs_info.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                    cs_pgm.IsTgidEnabled(2)};
        info.cs_info.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    return info;
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_,
                             VideoCore::BufferCache& buffer_cache_)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      buffer_cache{&buffer_cache_},
      desc_heap{instance, scheduler.GetMasterSemaphore(), DescriptorHeapSizes} {
#ifdef __ANDROID__
    const bool managed_shader_identity_cache =
        VideoCore::IsManagedVulkanDriverOptimization();
    const char* const shader_identity_cache_flag =
        std::getenv("EXECUTOR_SHADER_IDENTITY_CACHE");
    shader_identity_cache_enabled =
        managed_shader_identity_cache ||
        (shader_identity_cache_flag != nullptr &&
         shader_identity_cache_flag[0] != '\0' &&
         shader_identity_cache_flag[0] != '0');
    const char* const immutable_shader_identity_cache_flag =
        std::getenv("EXECUTOR_IMMUTABLE_SHADER_IDENTITY_CACHE");
    immutable_shader_identity_cache_enabled = shader_identity_cache_enabled &&
        (managed_shader_identity_cache ||
         (immutable_shader_identity_cache_flag != nullptr &&
          immutable_shader_identity_cache_flag[0] != '\0' &&
          immutable_shader_identity_cache_flag[0] != '0'));
    const char* const immutable_srt_snapshot_cache_flag =
        std::getenv("EXECUTOR_IMMUTABLE_SRT_SNAPSHOT_CACHE");
    immutable_srt_snapshot_cache_enabled =
        immutable_srt_snapshot_cache_flag != nullptr &&
        immutable_srt_snapshot_cache_flag[0] != '\0' &&
        immutable_srt_snapshot_cache_flag[0] != '0';
    if (shader_identity_cache_enabled) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_SHADER_IDENTITY_CACHE] enabled=1 scope=%s maxEntries=%zu",
            immutable_shader_identity_cache_enabled ? "backing_invalidation+fetch_epoch"
                                                    : "backing_invalidation",
            MaxShaderIdentityEntriesPerSubmit);
    }
    if (immutable_srt_snapshot_cache_enabled) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_SRT_CACHE_CONFIG] immutableBackingValidation=1 "
            "gpuGeneratedSrt=unsupported");
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_SRT_VICTIM_CACHE_CONFIG] primarySlotsPerProgram=%zu "
        "globalVictimSlots=%zu replacement=CLOCK exactByteValidation=1",
        Program::NumSpecializationSnapshotSlots, MaxSrtVictimSnapshots);
#endif
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_ubo_size = instance.UniformMaxSize() - instance.UniformMinAlignment(),
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_legacy_vertex_attributes = instance_.IsLegacyVertexAttributesSupported(),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_robust_buffer_access =
#ifdef __ANDROID__
            !(instance.GetDriverID() == vk::DriverId::eQualcommProprietary &&
              ::access("/data/user/0/app.lsx4.android/files/run-manual-buffer-bounds", F_OK) == 0),
#else
            true,
#endif
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .has_incomplete_fragment_shader_barycentric =
            instance_.IsFragmentShaderBarycentricSupported() &&
            instance.GetDriverID() == vk::DriverId::eMoltenvk,
        .has_broken_spirv_float_controls =
            instance.GetDriverID() == vk::DriverId::eQualcommProprietary,
        .needs_manual_interpolation =
            instance.IsFragmentShaderBarycentricSupported() &&
            instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMoltenvk,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMoltenvk,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };

    std::vector<u8> host_cache_data = LoadHostPipelineCache();
    vk::PipelineCacheCreateInfo cache_info{};
    cache_info.initialDataSize = host_cache_data.size();
    cache_info.pInitialData = host_cache_data.data();
    auto cache_result_value =
        instance.GetDevice().createPipelineCacheUnique(cache_info);
    auto cache_result = cache_result_value.result;
    auto cache = std::move(cache_result_value.value);
    bool host_cache_rejected = false;
    if (cache_result != vk::Result::eSuccess && !host_cache_data.empty()) {
        LOG_WARNING(Render_Vulkan,
                    "Native Vulkan pipeline cache was rejected; rebuilding it");
        host_cache_rejected = true;
        cache_result_value = instance.GetDevice().createPipelineCacheUnique({});
        cache_result = cache_result_value.result;
        cache = std::move(cache_result_value.value);
    }
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
    async_pipeline_cache_seed = std::move(host_cache_data);
    next_pipeline_cache_checkpoint =
        std::chrono::steady_clock::now() + std::chrono::seconds{15};
    if (host_cache_rejected) {
        MarkHostPipelineCacheDirty();
    }

#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_VK_PIPELINE_CACHE] loadedBytes=%zu vendor=0x%x "
                        "device=0x%x driver=0x%x",
                        async_pipeline_cache_seed.size(), instance.GetVendorID(),
                        instance.GetDeviceID(), instance.GetDriverVersion());
#endif

    WarmUp();
    EnsureAsyncPipelineMode();
}

PipelineCache::~PipelineCache() {
    StopAsyncPipelineWorkers();
    PublishAsyncPipelineResults();
    DestroyRetiredShaderModules();
    StopBackgroundWarmUp();
    SaveHostPipelineCache();
    SavePipelineNegativeCache();
}

std::string PipelineCache::HostPipelineCacheName() const {
    return fmt::format("vk-native-{:08x}-{:08x}-{:08x}.bin",
                       instance.GetVendorID(), instance.GetDeviceID(),
                       instance.GetDriverVersion());
}

bool PipelineCache::WriteCacheFileAtomically(const std::filesystem::path& path,
                                             const std::vector<u8>& data) {
    auto temporary_path = path;
    temporary_path += ".tmp";
    {
        Common::FS::IOFile file{temporary_path, Common::FS::FileAccessMode::Create};
        if (!file.IsOpen() || file.Write(data) != data.size() || !file.Commit()) {
            file.Close();
            std::error_code remove_error;
            std::filesystem::remove(temporary_path, remove_error);
            return false;
        }
        file.Close();
    }

    std::error_code rename_error;
    std::filesystem::rename(temporary_path, path, rename_error);
#ifdef _WIN32
    // std::filesystem::rename cannot replace an existing file on Windows.
    // Android/POSIX takes the atomic replacement path above.
    if (rename_error) {
        std::error_code remove_error;
        std::filesystem::remove(path, remove_error);
        rename_error.clear();
        std::filesystem::rename(temporary_path, path, rename_error);
    }
#endif
    if (!rename_error) {
        return true;
    }

    std::error_code remove_error;
    std::filesystem::remove(temporary_path, remove_error);
    LOG_WARNING(Render_Vulkan, "Failed to atomically replace pipeline cache {}: {}",
                path.string(), rename_error.message());
    return false;
}

std::vector<u8> PipelineCache::LoadHostPipelineCache() const {
    if (!Config::isPipelineCacheEnabled()) {
        return {};
    }
    const auto& game_info = Common::ElfInfo::Instance();
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                      game_info.GameSerial() / HostPipelineCacheName();
    const Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read};
    if (!file.IsOpen() || file.GetSize() < sizeof(vk::PipelineCacheHeaderVersionOne)) {
        return {};
    }
    std::vector<u8> data(file.GetSize());
    if (file.Read(data) != data.size()) {
        return {};
    }
    vk::PipelineCacheHeaderVersionOne header{};
    std::memcpy(&header, data.data(), sizeof(header));
    if (header.headerSize < sizeof(header) || header.headerSize > data.size() ||
        header.headerVersion != vk::PipelineCacheHeaderVersion::eOne ||
        header.vendorID != instance.GetVendorID() ||
        header.deviceID != instance.GetDeviceID() ||
        std::memcmp(header.pipelineCacheUUID.data(),
                    instance.GetPipelineCacheUUID().data(), VK_UUID_SIZE) != 0) {
        LOG_WARNING(Render_Vulkan,
                    "Ignoring incompatible native Vulkan pipeline cache");
        return {};
    }
    return data;
}

void PipelineCache::SaveHostPipelineCache() {
    if (host_pipeline_cache_saved_generation == host_pipeline_cache_generation ||
        !pipeline_cache ||
        !Config::isPipelineCacheEnabled()) {
        return;
    }
    auto [result, data] =
        instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if (result != vk::Result::eSuccess || data.empty()) {
        return;
    }
    const auto& game_info = Common::ElfInfo::Instance();
    const auto directory = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                           game_info.GameSerial();
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        return;
    }
    if (!WriteCacheFileAtomically(directory / HostPipelineCacheName(), data)) {
        return;
    }
    host_pipeline_cache_saved_generation = host_pipeline_cache_generation;
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_VK_PIPELINE_CACHE] savedBytes=%zu generation=%llu",
                        data.size(),
                        static_cast<unsigned long long>(host_pipeline_cache_generation));
#endif
}

void PipelineCache::MarkHostPipelineCacheDirty() {
    ++host_pipeline_cache_generation;
}

void PipelineCache::MaybeCheckpointPipelineCaches() {
    if (!Config::isPipelineCacheEnabled()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now < next_pipeline_cache_checkpoint) {
        return;
    }
    // Capture early startup work quickly, then keep expensive driver cache
    // extraction and durable I/O off the per-frame path for at least a minute.
    next_pipeline_cache_checkpoint = now + std::chrono::minutes{1};

    auto background_lock = LockBackgroundWarmUpForForeground();
    if (host_pipeline_cache_saved_generation == host_pipeline_cache_generation &&
        !pipeline_negative_cache_dirty) {
        return;
    }
    SaveHostPipelineCache();
    SavePipelineNegativeCache();
}

void PipelineCache::ResetPipelineBuildState() {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
}

AsyncPipelineCompileStats PipelineCache::GetAsyncPipelineStats() const noexcept {
    return GetAsyncPipelineCompileStats();
}

void PipelineCache::EnsureAsyncPipelineMode() {
    if (Config::managedAsyncPipelineCompilation()) {
        StartAsyncPipelineWorkers();
    } else if (async_pipeline_started) {
        StopAsyncPipelineWorkers();
        PublishAsyncPipelineResults();
        DestroyRetiredShaderModules();
        std::scoped_lock background_lock{background_warmup_mutex};
        for (const auto& key : fallback_graphics_pipelines) {
            if (auto it = graphics_pipelines.find(key);
                it != graphics_pipelines.end()) {
                auto retired = std::move(it.value());
                graphics_pipelines.erase(it);
                if (retired) {
                    scheduler.DeferOperation(
                        [pipeline = std::move(retired)]() mutable {
                            pipeline.reset();
                        });
                }
            }
        }
        for (const auto& key : fallback_compute_pipelines) {
            if (auto it = compute_pipelines.find(key);
                it != compute_pipelines.end()) {
                auto retired = std::move(it.value());
                compute_pipelines.erase(it);
                if (retired) {
                    scheduler.DeferOperation(
                        [pipeline = std::move(retired)]() mutable {
                            pipeline.reset();
                        });
                }
            }
        }
        fallback_graphics_pipelines.clear();
        fallback_compute_pipelines.clear();
        last_graphics_pipeline = nullptr;
        last_graphics_state_generation = std::numeric_limits<u64>::max();
        last_graphics_lookup_resolved = false;
    }
}

void PipelineCache::StartAsyncPipelineWorkers() {
    if (async_pipeline_started || !Config::managedAsyncPipelineCompilation() ||
        scheduler.IsDeviceLost()) {
        return;
    }
    const size_t default_workers =
#ifdef __ANDROID__
        1;
#else
        2;
#endif
    const size_t worker_count =
        ReadBoundedAsyncPipelineOption("EXECUTOR_ASYNC_PIPELINE_WORKERS",
                                       default_workers, 1, 2);
    async_pipeline_queue_capacity =
        ReadBoundedAsyncPipelineOption("EXECUTOR_ASYNC_PIPELINE_QUEUE", 128, 16, 512);
    async_shader_queue_capacity =
        ReadBoundedAsyncPipelineOption("EXECUTOR_ASYNC_SHADER_QUEUE", 64, 8, 256);
    async_shader_queue_byte_capacity =
        ReadBoundedAsyncPipelineOption("EXECUTOR_ASYNC_SHADER_QUEUE_MB", 32, 4, 256) *
        1024 * 1024;
    {
        std::scoped_lock lock{async_pipeline_mutex};
        async_pipeline_accepting = true;
        async_pipeline_started = true;
    }
    async_pipeline_stats.fail_fast_supported.store(
        instance.IsPipelineCreationCacheControlSupported(), std::memory_order_relaxed);
    async_pipeline_stats.worker_count.store(static_cast<u32>(worker_count),
                                            std::memory_order_relaxed);
    async_pipeline_stats.queue_capacity.store(
        static_cast<u32>(async_pipeline_queue_capacity), std::memory_order_relaxed);
    async_pipeline_stats.shader_queue_capacity.store(
        static_cast<u32>(async_shader_queue_capacity),
        std::memory_order_relaxed);
    async_pipeline_stats.shader_queue_byte_capacity.store(
        async_shader_queue_byte_capacity, std::memory_order_relaxed);
    async_pipeline_workers.reserve(worker_count);
    for (u32 index = 0; index < worker_count; ++index) {
        async_pipeline_workers.emplace_back(
            [this, index](const std::stop_token stop_token) {
                AsyncPipelineWorker(stop_token, index);
            });
    }
    async_shader_worker = std::jthread{
        [this](const std::stop_token stop_token) {
            AsyncShaderWorker(stop_token);
        }};
#ifdef __ANDROID__
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_ASYNC_PIPELINE_CONFIG] enabled=1 workers=%zu queue=%zu "
        "shaderQueue=%zu shaderQueueMiB=%zu failFast=%u "
        "fallback=disableOptimization optimizedReplacement=%s",
        worker_count, async_pipeline_queue_capacity, async_shader_queue_capacity,
        async_shader_queue_byte_capacity / (1024 * 1024),
        instance.IsPipelineCreationCacheControlSupported() ? 1u : 0u,
        DeferAsyncLiveReplacement(instance) ? "nextLaunch" : "live");
#endif
}

void PipelineCache::StopAsyncPipelineWorkers() {
    if (!async_pipeline_started && async_pipeline_workers.empty() &&
        !async_shader_worker.joinable()) {
        return;
    }
    {
        std::scoped_lock lock{async_pipeline_mutex};
        async_pipeline_accepting = false;
        async_pipeline_jobs.clear();
        async_shader_jobs.clear();
        async_shader_queued_bytes = 0;
        pending_fast_graphics_pipelines.clear();
        pending_optimized_graphics_pipelines.clear();
        pending_fast_compute_pipelines.clear();
        pending_optimized_compute_pipelines.clear();
        pending_optimized_shaders.clear();
        async_pipeline_stats.queue_depth.store(0, std::memory_order_relaxed);
        async_pipeline_stats.shader_queue_depth.store(0,
                                                      std::memory_order_relaxed);
        async_pipeline_stats.shader_queue_bytes.store(0,
                                                      std::memory_order_relaxed);
    }
    for (auto& worker : async_pipeline_workers) {
        worker.request_stop();
    }
    if (async_shader_worker.joinable()) {
        async_shader_worker.request_stop();
    }
    async_pipeline_cv.notify_all();
    async_pipeline_workers.clear();
    async_shader_worker = {};
    {
        std::scoped_lock lock{async_pipeline_mutex};
        async_pipeline_started = false;
    }
    async_pipeline_stats.worker_count.store(0, std::memory_order_relaxed);
}

bool PipelineCache::QueueAsyncGraphicsPipeline(
    const GraphicsPipelineKey& key,
    const std::span<const Shader::Info* const, MaxShaderStages> stage_infos,
    const std::span<const Shader::RuntimeInfo, MaxShaderStages> stage_runtime_infos,
    std::optional<Shader::Gcn::FetchShaderData> stage_fetch_shader,
    const std::span<const vk::ShaderModule, MaxShaderStages> stage_modules,
    GraphicsPipeline::SerializationSupport sdata, const bool optimized) {
    auto payload = std::make_shared<AsyncGraphicsPayload>();
    payload->key = key;
    std::ranges::copy(stage_infos, payload->infos.begin());
    std::ranges::copy(stage_runtime_infos, payload->runtime_infos.begin());
    payload->fetch_shader = std::move(stage_fetch_shader);
    std::ranges::copy(stage_modules, payload->modules.begin());
    payload->sdata = std::move(sdata);

    std::scoped_lock lock{async_pipeline_mutex};
    if (!async_pipeline_accepting) {
        return false;
    }
    auto& pending = optimized ? pending_optimized_graphics_pipelines
                              : pending_fast_graphics_pipelines;
    if (pending.contains(key)) {
        async_pipeline_stats.pending_hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (async_pipeline_jobs.size() >= async_pipeline_queue_capacity) {
        async_pipeline_stats.queue_drops.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    pending.emplace(key);
    AsyncPipelineJob job{
        .payload = std::move(payload),
        .tier = optimized ? AsyncPipelineTier::Optimized : AsyncPipelineTier::Fast,
        .queued_at = std::chrono::steady_clock::now(),
    };
    if (optimized) {
        async_pipeline_jobs.push_back(std::move(job));
    } else {
        const auto first_optimized = std::find_if(
            async_pipeline_jobs.begin(), async_pipeline_jobs.end(),
            [](const AsyncPipelineJob& queued) {
                return queued.tier == AsyncPipelineTier::Optimized;
            });
        async_pipeline_jobs.insert(first_optimized, std::move(job));
    }
    (optimized ? async_pipeline_stats.optimized_queued
               : async_pipeline_stats.fast_queued)
        .fetch_add(1, std::memory_order_relaxed);
    const u32 queue_depth = static_cast<u32>(async_pipeline_jobs.size());
    async_pipeline_stats.queue_depth.store(queue_depth, std::memory_order_relaxed);
    AtomicMax(async_pipeline_stats.max_queue_depth, queue_depth);
    async_pipeline_cv.notify_one();
    return true;
}

bool PipelineCache::QueueAsyncComputePipeline(
    const ComputePipelineKey& key, const Shader::Info& info,
    const vk::ShaderModule module, ComputePipeline::SerializationSupport sdata,
    const bool optimized) {
    auto payload = std::make_shared<AsyncComputePayload>();
    payload->key = key;
    payload->info = &info;
    payload->module = module;
    payload->sdata = std::move(sdata);

    std::scoped_lock lock{async_pipeline_mutex};
    if (!async_pipeline_accepting) {
        return false;
    }
    auto& pending =
        optimized ? pending_optimized_compute_pipelines : pending_fast_compute_pipelines;
    if (pending.contains(key)) {
        async_pipeline_stats.pending_hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    if (async_pipeline_jobs.size() >= async_pipeline_queue_capacity) {
        async_pipeline_stats.queue_drops.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    pending.emplace(key);
    AsyncPipelineJob job{
        .payload = std::move(payload),
        .tier = optimized ? AsyncPipelineTier::Optimized : AsyncPipelineTier::Fast,
        .queued_at = std::chrono::steady_clock::now(),
    };
    if (optimized) {
        async_pipeline_jobs.push_back(std::move(job));
    } else {
        const auto first_optimized = std::find_if(
            async_pipeline_jobs.begin(), async_pipeline_jobs.end(),
            [](const AsyncPipelineJob& queued) {
                return queued.tier == AsyncPipelineTier::Optimized;
            });
        async_pipeline_jobs.insert(first_optimized, std::move(job));
    }
    (optimized ? async_pipeline_stats.optimized_queued
               : async_pipeline_stats.fast_queued)
        .fetch_add(1, std::memory_order_relaxed);
    const u32 queue_depth = static_cast<u32>(async_pipeline_jobs.size());
    async_pipeline_stats.queue_depth.store(queue_depth, std::memory_order_relaxed);
    AtomicMax(async_pipeline_stats.max_queue_depth, queue_depth);
    async_pipeline_cv.notify_one();
    return true;
}

bool PipelineCache::QueueAsyncShaderOptimization(
    std::shared_ptr<AsyncShaderPayload> payload) {
    if (!payload || !payload->program ||
        payload->perm_idx >= payload->program->modules.size()) {
        return false;
    }
    const AsyncShaderKey key{
        .program = payload->program,
        .perm_idx = payload->perm_idx,
        .generation = payload->expected_generation,
        .cache_generation = payload->cache_generation,
    };
    if (IsShaderBinaryTierKnownInvalid(payload->info->pgm_hash,
                                       payload->perm_idx,
                                       ShaderBinaryTier::Optimized)) {
        std::scoped_lock lock{async_pipeline_mutex};
        invalid_optimized_shaders.emplace(key);
        async_pipeline_stats.shader_negative_hits.fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }

    std::scoped_lock lock{async_pipeline_mutex};
    if (!async_pipeline_accepting || invalid_optimized_shaders.contains(key)) {
        return false;
    }
    if (pending_optimized_shaders.contains(key)) {
        async_pipeline_stats.pending_hits.fetch_add(1,
                                                    std::memory_order_relaxed);
        return true;
    }
    if (pending_optimized_shaders.size() >= async_shader_queue_capacity ||
        payload->queued_bytes > async_shader_queue_byte_capacity ||
        async_shader_queued_bytes >
            async_shader_queue_byte_capacity - payload->queued_bytes) {
        async_pipeline_stats.shader_queue_drops.fetch_add(
            1, std::memory_order_relaxed);
        return false;
    }
    pending_optimized_shaders.emplace(key);
    async_shader_queued_bytes += payload->queued_bytes;
    async_shader_jobs.push_back(std::move(payload));
    const u32 shader_queue_depth =
        static_cast<u32>(pending_optimized_shaders.size());
    async_pipeline_stats.shader_queue_depth.store(
        shader_queue_depth, std::memory_order_relaxed);
    async_pipeline_stats.shader_queue_bytes.store(
        async_shader_queued_bytes, std::memory_order_relaxed);
    AtomicMax(async_pipeline_stats.shader_max_queue_depth,
              shader_queue_depth);
    AtomicMax(async_pipeline_stats.shader_max_queue_bytes,
              static_cast<u64>(async_shader_queued_bytes));
    async_pipeline_stats.shader_optimized_queued.fetch_add(
        1, std::memory_order_relaxed);
    async_pipeline_cv.notify_all();
    return true;
}

void PipelineCache::AsyncShaderWorker(const std::stop_token stop_token) {
    Common::SetCurrentThreadName("shadPS4:VkShader");
#ifdef __ANDROID__
    (void)::setpriority(PRIO_PROCESS, 0, 7);
#endif
    const auto device = instance.GetDevice();
    for (;;) {
        std::shared_ptr<AsyncShaderPayload> payload;
        {
            std::unique_lock lock{async_pipeline_mutex};
            const bool ready = async_pipeline_cv.wait(
                lock, stop_token, [this] {
                    return !async_shader_jobs.empty() ||
                           !async_pipeline_accepting;
                });
            if (!ready || stop_token.stop_requested() ||
                (!async_pipeline_accepting && async_shader_jobs.empty())) {
                return;
            }
            if (DeferAsyncLiveReplacement(instance) &&
                AsyncSteadyMilliseconds() <
                    async_last_foreground_activity_ms.load(
                        std::memory_order_acquire) +
                        750u) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds{50});
                continue;
            }
            payload = std::move(async_shader_jobs.front());
            async_shader_jobs.pop_front();
        }

        AsyncShaderResult completed{
            .payload = payload,
            .result = vk::Result::eErrorUnknown,
        };
        const auto started = std::chrono::steady_clock::now();
        if (!scheduler.IsDeviceLost() && payload->info) {
            const Shader::ShaderParams owned_params{
                .user_data = payload->user_data,
                .code = payload->code,
                .hash = payload->info->pgm_hash,
            };
            auto optimized_info = std::make_unique<Shader::Info>(
                payload->info->stage, payload->info->l_stage, owned_params);
            auto optimized_pools = std::make_unique<Shader::Pools>();
            auto optimized_runtime_info = payload->source_runtime_info;
            auto optimized_ir = std::make_unique<Shader::IR::Program>(
                Shader::TranslateProgram(
                    payload->code, *optimized_pools, *optimized_info,
                    optimized_runtime_info, profile,
                    Shader::TranslationTier::Optimized));
            auto binding = payload->binding_start;
            completed.spv = Shader::Backend::SPIRV::EmitSPIRV(
                profile, optimized_runtime_info, *optimized_ir, binding);
            if (binding == payload->expected_binding_end &&
                !completed.spv.empty()) {
                const vk::ShaderModuleCreateInfo create_info{
                    .codeSize = completed.spv.size() * sizeof(u32),
                    .pCode = completed.spv.data(),
                };
                auto [result, module] = device.createShaderModule(create_info);
                completed.result = result;
                if (result == vk::Result::eSuccess) {
                    completed.module = module;
                    payload->info = std::move(optimized_info);
                    payload->pools = std::move(optimized_pools);
                    payload->ir = std::move(optimized_ir);
                    payload->runtime_info =
                        std::move(optimized_runtime_info);
                }
            }
        } else if (scheduler.IsDeviceLost()) {
            completed.result = vk::Result::eErrorDeviceLost;
        }
        completed.build_us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started)
                .count());
        async_pipeline_stats.shader_optimized_completed.fetch_add(
            1, std::memory_order_relaxed);
        async_pipeline_stats.shader_total_build_us.fetch_add(
            completed.build_us, std::memory_order_relaxed);
        AtomicMax(async_pipeline_stats.shader_max_build_us,
                  completed.build_us);
        if (completed.result != vk::Result::eSuccess ||
            !completed.module) {
            async_pipeline_stats.shader_build_failures.fetch_add(
                1, std::memory_order_relaxed);
        }
        {
            std::scoped_lock lock{async_pipeline_mutex};
            async_shader_results.push_back(std::move(completed));
        }
    }
}

void PipelineCache::PublishAsyncShaderResults() {
    std::deque<AsyncShaderResult> completed;
    {
        std::scoped_lock lock{async_pipeline_mutex};
        completed.swap(async_shader_results);
        for (const auto& result : completed) {
            const AsyncShaderKey key{
                .program = result.payload->program,
                .perm_idx = result.payload->perm_idx,
                .generation = result.payload->expected_generation,
                .cache_generation = result.payload->cache_generation,
            };
            pending_optimized_shaders.erase(key);
            async_shader_queued_bytes =
                result.payload->queued_bytes > async_shader_queued_bytes
                    ? 0
                    : async_shader_queued_bytes - result.payload->queued_bytes;
        }
        async_pipeline_stats.shader_queue_depth.store(
            static_cast<u32>(pending_optimized_shaders.size()),
            std::memory_order_relaxed);
        async_pipeline_stats.shader_queue_bytes.store(
            async_shader_queued_bytes, std::memory_order_relaxed);
    }
    if (completed.empty()) {
        return;
    }

    const auto device = instance.GetDevice();
    std::scoped_lock background_lock{background_warmup_mutex};
    for (auto& result : completed) {
        auto& payload = *result.payload;
        const AsyncShaderKey key{
            .program = payload.program,
            .perm_idx = payload.perm_idx,
            .generation = payload.expected_generation,
            .cache_generation = payload.cache_generation,
        };
        if (result.result == vk::Result::eErrorDeviceLost) {
            scheduler.MarkTerminal(result.result,
                                   "async optimized shader build");
            if (result.module) {
                device.destroyShaderModule(result.module);
            }
            continue;
        }
        const bool generation_matches =
            payload.program && payload.perm_idx < payload.program->modules.size() &&
            payload.cache_generation == shader_tier_generation &&
            payload.program->modules[payload.perm_idx].generation ==
                payload.expected_generation &&
            payload.program->modules[payload.perm_idx].module ==
                payload.expected_module &&
            payload.program->modules[payload.perm_idx].tier ==
                ShaderBinaryTier::Fast;
        if (!generation_matches || scheduler.IsDeviceLost()) {
            if (result.module) {
                device.destroyShaderModule(result.module);
            }
            continue;
        }
        if (result.result != vk::Result::eSuccess || !result.module) {
            invalid_optimized_shaders.emplace(key);
            RecordShaderBinaryTierFailure(payload.info->pgm_hash,
                                          payload.perm_idx,
                                          ShaderBinaryTier::Optimized);
            continue;
        }

        if (DeferAsyncLiveReplacement(instance)) {
            RegisterShaderBinary(std::move(result.spv),
                                 payload.info->pgm_hash,
                                 payload.perm_idx,
                                 ShaderBinaryTier::Optimized);
            invalid_optimized_shaders.erase(key);
            async_pipeline_stats.shader_optimized_published.fetch_add(
                1, std::memory_order_relaxed);
            // The foreground fast module stays active for this process.
            // The optimized module is only needed to validate driver
            // acceptance; its SPIR-V is now in the per-title disk cache.
            device.destroyShaderModule(result.module);
            continue;
        }

        auto& program_module = payload.program->modules[payload.perm_idx];
        const vk::ShaderModule old_module = program_module.module;
        const vk::ShaderModule new_module = result.module;
        program_module.module = new_module;
        program_module.tier = ShaderBinaryTier::Optimized;
        ++program_module.generation;
        ++payload.program->module_generation;

        std::vector<std::unique_ptr<GraphicsPipeline>> retired_graphics;
        std::vector<std::unique_ptr<ComputePipeline>> retired_compute;
        if (auto related = module_related_pipelines.find(old_module);
            related != module_related_pipelines.end()) {
            for (const auto& pipeline_key : related.value()) {
                if (const auto* graphics =
                        std::get_if<GraphicsPipelineKey>(&pipeline_key)) {
                    if (auto it = graphics_pipelines.find(*graphics);
                        it != graphics_pipelines.end()) {
                        retired_graphics.push_back(std::move(it.value()));
                        graphics_pipelines.erase(it);
                    }
                    fallback_graphics_pipelines.erase(*graphics);
                } else {
                    const auto& compute =
                        std::get<ComputePipelineKey>(pipeline_key);
                    if (auto it = compute_pipelines.find(compute);
                        it != compute_pipelines.end()) {
                        retired_compute.push_back(std::move(it.value()));
                        compute_pipelines.erase(it);
                    }
                    fallback_compute_pipelines.erase(compute);
                }
            }
            module_related_pipelines.erase(related);
        }
        for (auto& pipeline : retired_graphics) {
            scheduler.DeferOperation(
                [pipeline = std::move(pipeline)]() mutable { pipeline.reset(); });
        }
        for (auto& pipeline : retired_compute) {
            scheduler.DeferOperation(
                [pipeline = std::move(pipeline)]() mutable { pipeline.reset(); });
        }

        retired_shader_modules.push_back(old_module);
        const auto name =
            GetShaderName(payload.info->stage, payload.info->pgm_hash,
                          payload.perm_idx);
        Vulkan::SetObjectName(device, new_module, name);
        RegisterShaderBinary(std::move(result.spv), payload.info->pgm_hash,
                             payload.perm_idx,
                             ShaderBinaryTier::Optimized);
        invalid_optimized_shaders.erase(key);
        last_graphics_pipeline = nullptr;
        last_graphics_state_generation = std::numeric_limits<u64>::max();
        last_graphics_lookup_resolved = false;
        async_pipeline_stats.shader_optimized_published.fetch_add(
            1, std::memory_order_relaxed);
        async_pipeline_stats.shader_replacements.fetch_add(
            1, std::memory_order_relaxed);
    }
#ifdef __ANDROID__
    const auto stats = GetAsyncPipelineCompileStats();
    if (stats.shader_optimized_completed <= 8 ||
        (stats.shader_optimized_completed & 63u) == 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_ASYNC_SHADER] fast=%llu queued=%llu completed=%llu "
            "published=%llu replaced=%llu failed=%llu negative=%llu "
            "drops=%llu queue=%u/%u bytes=%llu/%llu buildUs=%llu maxUs=%llu",
            static_cast<unsigned long long>(stats.shader_fast_compiled),
            static_cast<unsigned long long>(stats.shader_optimized_queued),
            static_cast<unsigned long long>(stats.shader_optimized_completed),
            static_cast<unsigned long long>(stats.shader_optimized_published),
            static_cast<unsigned long long>(stats.shader_replacements),
            static_cast<unsigned long long>(stats.shader_build_failures),
            static_cast<unsigned long long>(stats.shader_negative_hits),
            static_cast<unsigned long long>(stats.shader_queue_drops),
            stats.shader_queue_depth, stats.shader_queue_capacity,
            static_cast<unsigned long long>(stats.shader_queue_bytes),
            static_cast<unsigned long long>(
                stats.shader_queue_byte_capacity),
            static_cast<unsigned long long>(stats.shader_total_build_us),
            static_cast<unsigned long long>(stats.shader_max_build_us));
    }
#endif
}

void PipelineCache::DestroyRetiredShaderModules() {
    if (retired_shader_modules.empty()) {
        return;
    }
    std::vector<vk::ShaderModule> destroyable;
    {
        std::scoped_lock lock{async_pipeline_mutex};
        const auto is_queued = [this](const vk::ShaderModule module) {
            return std::ranges::any_of(
                async_pipeline_jobs,
                [module](const AsyncPipelineJob& job) {
                    return std::visit(
                        [module](const auto& payload) {
                            using PayloadType =
                                std::decay_t<decltype(payload)>;
                            if constexpr (std::is_same_v<
                                              PayloadType,
                                              std::shared_ptr<
                                                  AsyncGraphicsPayload>>) {
                                for (size_t stage = 0;
                                     stage < MaxShaderStages; ++stage) {
                                    if (payload->infos[stage] &&
                                        payload->modules[stage] == module) {
                                        return true;
                                    }
                                }
                                return false;
                            } else {
                                return payload->module == module;
                            }
                        },
                        job.payload);
                });
        };
        const auto is_completed = [this](const vk::ShaderModule module) {
            return std::ranges::any_of(
                async_pipeline_results,
                [module](const AsyncPipelineResult& result) {
                    return std::visit(
                        [module](const auto& completed) {
                            using ResultType =
                                std::decay_t<decltype(completed)>;
                            if constexpr (std::is_same_v<
                                              ResultType,
                                              AsyncGraphicsResult>) {
                                for (size_t stage = 0;
                                     stage < MaxShaderStages; ++stage) {
                                    if (completed.payload->infos[stage] &&
                                        completed.payload->modules[stage] ==
                                            module) {
                                        return true;
                                    }
                                }
                                return false;
                            } else {
                                return completed.payload->module == module;
                            }
                        },
                        result);
                });
        };
        auto it = retired_shader_modules.begin();
        while (it != retired_shader_modules.end()) {
            if (!active_async_pipeline_modules.contains(*it) &&
                !is_queued(*it) && !is_completed(*it)) {
                destroyable.push_back(*it);
                it = retired_shader_modules.erase(it);
            } else {
                ++it;
            }
        }
    }
    const auto device = instance.GetDevice();
    for (const auto module : destroyable) {
        if (module) {
            device.destroyShaderModule(module);
        }
    }
}

void PipelineCache::AsyncPipelineWorker(const std::stop_token stop_token,
                                        const u32 worker_index) {
    Common::SetCurrentThreadName(
        fmt::format("shadPS4:VkPipe{}", worker_index).c_str());
#ifdef __ANDROID__
    (void)::setpriority(PRIO_PROCESS, 0, 6);
#endif
    const auto device = instance.GetDevice();
    for (;;) {
        AsyncPipelineJob job;
        {
            std::unique_lock lock{async_pipeline_mutex};
            const bool ready = async_pipeline_cv.wait(
                lock, stop_token, [this] {
                    return !async_pipeline_jobs.empty() || !async_pipeline_accepting;
                });
            if (!ready || stop_token.stop_requested() ||
                (!async_pipeline_accepting && async_pipeline_jobs.empty())) {
                return;
            }
            if (DeferAsyncLiveReplacement(instance) &&
                AsyncSteadyMilliseconds() <
                    async_last_foreground_activity_ms.load(
                        std::memory_order_acquire) +
                        750u) {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds{50});
                continue;
            }
            job = std::move(async_pipeline_jobs.front());
            async_pipeline_jobs.pop_front();
            std::visit(
                [this](const auto& payload) {
                    using PayloadType = std::decay_t<decltype(payload)>;
                    if constexpr (std::is_same_v<
                                      PayloadType,
                                      std::shared_ptr<AsyncGraphicsPayload>>) {
                        for (size_t stage = 0; stage < MaxShaderStages;
                             ++stage) {
                            if (payload->infos[stage] &&
                                payload->modules[stage]) {
                                ++active_async_pipeline_modules
                                      [payload->modules[stage]];
                            }
                        }
                    } else if (payload->module) {
                        ++active_async_pipeline_modules[payload->module];
                    }
                },
                job.payload);
            async_pipeline_stats.queue_depth.store(
                static_cast<u32>(async_pipeline_jobs.size()),
                std::memory_order_relaxed);
        }

        vk::PipelineCacheCreateInfo cache_info{};
        cache_info.initialDataSize = async_pipeline_cache_seed.size();
        cache_info.pInitialData = async_pipeline_cache_seed.data();
        auto [cache_result, local_cache] =
            device.createPipelineCacheUnique(cache_info);
        const auto started = std::chrono::steady_clock::now();
        const vk::PipelineCreateFlags build_flags =
            job.tier == AsyncPipelineTier::Fast
                ? vk::PipelineCreateFlagBits::eDisableOptimization
                : vk::PipelineCreateFlags{};

        AsyncPipelineResult completed;
        if (const auto* graphics =
                std::get_if<std::shared_ptr<AsyncGraphicsPayload>>(&job.payload)) {
            AsyncGraphicsResult result{
                .payload = *graphics,
                .tier = job.tier,
                .result = cache_result,
            };
            if (cache_result == vk::Result::eSuccess && !scheduler.IsDeviceLost()) {
                result.local_cache = std::move(local_cache);
                result.pipeline = std::make_unique<GraphicsPipeline>(
                    instance, scheduler, desc_heap, profile, (*graphics)->key,
                    *result.local_cache, (*graphics)->infos,
                    (*graphics)->runtime_infos, (*graphics)->fetch_shader,
                    (*graphics)->modules, (*graphics)->sdata, true, build_flags);
                result.result = result.pipeline->BuildResult();
            }
            result.build_us = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started)
                    .count());
            completed = std::move(result);
        } else {
            const auto& compute =
                std::get<std::shared_ptr<AsyncComputePayload>>(job.payload);
            AsyncComputeResult result{
                .payload = compute,
                .tier = job.tier,
                .result = cache_result,
            };
            if (cache_result == vk::Result::eSuccess && !scheduler.IsDeviceLost()) {
                result.local_cache = std::move(local_cache);
                result.pipeline = std::make_unique<ComputePipeline>(
                    instance, scheduler, desc_heap, profile, *result.local_cache,
                    compute->key, *compute->info, compute->module, compute->sdata,
                    true, build_flags);
                result.result = result.pipeline->BuildResult();
            }
            result.build_us = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started)
                    .count());
            completed = std::move(result);
        }

        const u64 build_us = std::visit(
            [](const auto& result) { return result.build_us; }, completed);
        const bool valid = std::visit(
            [](const auto& result) {
                return result.pipeline && result.pipeline->IsValid();
            },
            completed);
        (job.tier == AsyncPipelineTier::Fast
             ? async_pipeline_stats.fast_completed
             : async_pipeline_stats.optimized_completed)
            .fetch_add(1, std::memory_order_relaxed);
        async_pipeline_stats.total_build_us.fetch_add(build_us,
                                                      std::memory_order_relaxed);
        AtomicMax(async_pipeline_stats.max_build_us, build_us);
        if (!valid) {
            async_pipeline_stats.build_failures.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::scoped_lock lock{async_pipeline_mutex};
            std::visit(
                [this](const auto& payload) {
                    const auto release_module =
                        [this](const vk::ShaderModule module) {
                            if (auto it =
                                    active_async_pipeline_modules.find(module);
                                it != active_async_pipeline_modules.end()) {
                                if (--it->second == 0) {
                                    active_async_pipeline_modules.erase(it);
                                }
                            }
                        };
                    using PayloadType = std::decay_t<decltype(payload)>;
                    if constexpr (std::is_same_v<
                                      PayloadType,
                                      std::shared_ptr<AsyncGraphicsPayload>>) {
                        for (size_t stage = 0; stage < MaxShaderStages;
                             ++stage) {
                            if (payload->infos[stage] &&
                                payload->modules[stage]) {
                                release_module(payload->modules[stage]);
                            }
                        }
                    } else if (payload->module) {
                        release_module(payload->module);
                    }
                },
                job.payload);
            async_pipeline_results.push_back(std::move(completed));
        }
    }
}

void PipelineCache::PublishAsyncPipelineResults() {
    std::deque<AsyncPipelineResult> completed;
    {
        std::scoped_lock lock{async_pipeline_mutex};
        completed.swap(async_pipeline_results);
        for (const auto& item : completed) {
            std::visit(
                [this](const auto& result) {
                    const bool optimized =
                        result.tier == AsyncPipelineTier::Optimized;
                    using ResultType = std::decay_t<decltype(result)>;
                    if constexpr (std::is_same_v<ResultType, AsyncGraphicsResult>) {
                        (optimized ? pending_optimized_graphics_pipelines
                                   : pending_fast_graphics_pipelines)
                            .erase(result.payload->key);
                    } else {
                        (optimized ? pending_optimized_compute_pipelines
                                   : pending_fast_compute_pipelines)
                            .erase(result.payload->key);
                    }
                },
                item);
        }
    }
    if (completed.empty()) {
        PublishAsyncShaderResults();
        DestroyRetiredShaderModules();
        return;
    }

    std::vector<std::shared_ptr<AsyncGraphicsPayload>> optimize_graphics;
    std::vector<std::shared_ptr<AsyncComputePayload>> optimize_compute;
    {
        std::scoped_lock background_lock{background_warmup_mutex};
        const auto merge_local_cache = [&](auto& result) {
            if (!result.local_cache || !result.pipeline ||
                !result.pipeline->IsValid()) {
                return;
            }
            const std::array source_caches{*result.local_cache};
            const auto merge_result =
                instance.GetDevice().mergePipelineCaches(*pipeline_cache,
                                                         source_caches);
            if (merge_result == vk::Result::eSuccess) {
                MarkHostPipelineCacheDirty();
            } else if (merge_result == vk::Result::eErrorDeviceLost) {
                scheduler.MarkTerminal(merge_result,
                                       "async pipeline cache merge");
            } else {
                LOG_WARNING(Render_Vulkan,
                            "Failed to merge async pipeline cache: {}",
                            vk::to_string(merge_result));
            }
        };

        for (auto& item : completed) {
            std::visit(
                [&](auto& result) {
                    using ResultType = std::decay_t<decltype(result)>;
                    const bool optimized =
                        result.tier == AsyncPipelineTier::Optimized;
                    const bool valid =
                        result.pipeline && result.pipeline->IsValid();
                    const auto module_is_retired =
                        [this](const vk::ShaderModule module) {
                            return std::ranges::find(retired_shader_modules,
                                                     module) !=
                                   retired_shader_modules.end();
                        };
                    bool uses_retired_module = false;
                    if constexpr (std::is_same_v<ResultType,
                                                 AsyncGraphicsResult>) {
                        for (size_t stage = 0; stage < MaxShaderStages;
                             ++stage) {
                            uses_retired_module |=
                                result.payload->infos[stage] &&
                                module_is_retired(
                                    result.payload->modules[stage]);
                        }
                    } else {
                        uses_retired_module =
                            module_is_retired(result.payload->module);
                    }
                    if (uses_retired_module) {
                        return;
                    }
                    if (result.result == vk::Result::eErrorDeviceLost) {
                        scheduler.MarkTerminal(result.result,
                                               "async pipeline build");
                    }

                    if constexpr (std::is_same_v<ResultType,
                                                 AsyncGraphicsResult>) {
                        const auto& key = result.payload->key;
                        auto existing = graphics_pipelines.find(key);
                        const bool existing_valid =
                            existing != graphics_pipelines.end() &&
                            existing.value() && existing.value()->IsValid();
                        const bool existing_fallback =
                            fallback_graphics_pipelines.contains(key);
                        if (!valid) {
                            if (!optimized && !existing_valid &&
                                result.result != vk::Result::eErrorDeviceLost &&
                                result.result !=
                                    vk::Result::ePipelineCompileRequired) {
                                RecordInvalid(key);
                            }
                            return;
                        }

                        merge_local_cache(result);
                        if (optimized && existing_valid &&
                            DeferAsyncLiveReplacement(instance)) {
                            // The local cache was merged above, so the
                            // optimized pipeline is available to the next
                            // launch without hot-swapping a live Qualcomm
                            // object in this command stream.
                            return;
                        }
                        bool published = false;
                        bool replaced = false;
                        if (optimized) {
                            if (existing_valid && existing_fallback) {
                                auto retired = std::move(existing.value());
                                existing.value() = std::move(result.pipeline);
                                fallback_graphics_pipelines.erase(key);
                                scheduler.DeferOperation(
                                    [pipeline = std::move(retired)]() mutable {
                                        pipeline.reset();
                                    });
                                published = true;
                                replaced = true;
                            } else if (!existing_valid) {
                                if (existing == graphics_pipelines.end()) {
                                    graphics_pipelines.try_emplace(
                                        key, std::move(result.pipeline));
                                } else {
                                    existing.value() = std::move(result.pipeline);
                                }
                                fallback_graphics_pipelines.erase(key);
                                published = true;
                            }
                        } else if (!existing_valid) {
                            if (existing == graphics_pipelines.end()) {
                                graphics_pipelines.try_emplace(
                                    key, std::move(result.pipeline));
                            } else {
                                existing.value() = std::move(result.pipeline);
                            }
                            fallback_graphics_pipelines.emplace(key);
                            published = true;
                        }
                        if (!published) {
                            return;
                        }

                        RecordValid(key);
                        ++num_new_pipelines;
                        const auto pipeline_hash =
                            std::hash<GraphicsPipelineKey>{}(key);
                        RegisterPipelineData(key, pipeline_hash,
                                             result.payload->sdata);
                        for (size_t stage = 0; stage < MaxShaderStages;
                             ++stage) {
                            if (result.payload->infos[stage]) {
                                module_related_pipelines
                                    [result.payload->modules[stage]]
                                        .emplace_back(key);
                            }
                        }
                        last_graphics_pipeline = nullptr;
                        last_graphics_state_generation =
                            std::numeric_limits<u64>::max();
                        last_graphics_lookup_resolved = false;
                        async_pipeline_stats.published.fetch_add(
                            1, std::memory_order_relaxed);
                        if (replaced) {
                            async_pipeline_stats.optimized_replacements
                                .fetch_add(1, std::memory_order_relaxed);
                        }
                        if (!optimized &&
                            Config::managedAsyncPipelineCompilation()) {
                            optimize_graphics.push_back(result.payload);
                        }
                    } else {
                        const auto& key = result.payload->key;
                        auto existing = compute_pipelines.find(key);
                        const bool existing_valid =
                            existing != compute_pipelines.end() &&
                            existing.value() && existing.value()->IsValid();
                        const bool existing_fallback =
                            fallback_compute_pipelines.contains(key);
                        if (!valid) {
                            if (!optimized && !existing_valid &&
                                result.result != vk::Result::eErrorDeviceLost &&
                                result.result !=
                                    vk::Result::ePipelineCompileRequired) {
                                RecordInvalid(key);
                            }
                            return;
                        }

                        merge_local_cache(result);
                        if (optimized && existing_valid &&
                            DeferAsyncLiveReplacement(instance)) {
                            return;
                        }
                        bool published = false;
                        bool replaced = false;
                        if (optimized) {
                            if (existing_valid && existing_fallback) {
                                auto retired = std::move(existing.value());
                                existing.value() = std::move(result.pipeline);
                                fallback_compute_pipelines.erase(key);
                                scheduler.DeferOperation(
                                    [pipeline = std::move(retired)]() mutable {
                                        pipeline.reset();
                                    });
                                published = true;
                                replaced = true;
                            } else if (!existing_valid) {
                                if (existing == compute_pipelines.end()) {
                                    compute_pipelines.try_emplace(
                                        key, std::move(result.pipeline));
                                } else {
                                    existing.value() = std::move(result.pipeline);
                                }
                                fallback_compute_pipelines.erase(key);
                                published = true;
                            }
                        } else if (!existing_valid) {
                            if (existing == compute_pipelines.end()) {
                                compute_pipelines.try_emplace(
                                    key, std::move(result.pipeline));
                            } else {
                                existing.value() = std::move(result.pipeline);
                            }
                            fallback_compute_pipelines.emplace(key);
                            published = true;
                        }
                        if (!published) {
                            return;
                        }

                        RecordValid(key);
                        ++num_new_pipelines;
                        RegisterPipelineData(key, result.payload->sdata);
                        module_related_pipelines[result.payload->module]
                            .emplace_back(key);
                        async_pipeline_stats.published.fetch_add(
                            1, std::memory_order_relaxed);
                        if (replaced) {
                            async_pipeline_stats.optimized_replacements
                                .fetch_add(1, std::memory_order_relaxed);
                        }
                        if (!optimized &&
                            Config::managedAsyncPipelineCompilation()) {
                            optimize_compute.push_back(result.payload);
                        }
                    }
                },
                item);
        }
    }

    for (const auto& payload : optimize_graphics) {
        QueueAsyncGraphicsPipeline(
            payload->key, payload->infos, payload->runtime_infos,
            payload->fetch_shader, payload->modules, payload->sdata, true);
    }
    for (const auto& payload : optimize_compute) {
        QueueAsyncComputePipeline(payload->key, *payload->info,
                                  payload->module, payload->sdata, true);
    }
    // Shader publication runs after pipeline publication.  If an optimized
    // shader supersedes a module used by a just-finished pipeline, the module
    // swap and dependent-pipeline invalidation happen in this same turn.
    PublishAsyncShaderResults();
    DestroyRetiredShaderModules();

#ifdef __ANDROID__
    const auto stats = GetAsyncPipelineCompileStats();
    const u64 completed_count =
        stats.fast_completed + stats.optimized_completed;
    if (completed_count <= 8 || (completed_count & 63u) == 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_ASYNC_PIPELINE] fastDone=%llu optDone=%llu "
            "published=%llu replaced=%llu failed=%llu queue=%u/%u "
            "fallback=%llu skipped=%llu shaderFast=%llu shaderOpt=%llu/%llu "
            "shaderReplace=%llu shaderQueue=%u/%u shaderBytes=%llu/%llu "
            "totalBuildUs=%llu maxBuildUs=%llu",
            static_cast<unsigned long long>(stats.fast_completed),
            static_cast<unsigned long long>(stats.optimized_completed),
            static_cast<unsigned long long>(stats.published),
            static_cast<unsigned long long>(stats.optimized_replacements),
            static_cast<unsigned long long>(stats.build_failures),
            stats.queue_depth, stats.queue_capacity,
            static_cast<unsigned long long>(stats.foreground_fallbacks),
            static_cast<unsigned long long>(stats.skipped_uses),
            static_cast<unsigned long long>(stats.shader_fast_compiled),
            static_cast<unsigned long long>(stats.shader_optimized_published),
            static_cast<unsigned long long>(stats.shader_optimized_completed),
            static_cast<unsigned long long>(stats.shader_replacements),
            stats.shader_queue_depth, stats.shader_queue_capacity,
            static_cast<unsigned long long>(stats.shader_queue_bytes),
            static_cast<unsigned long long>(stats.shader_queue_byte_capacity),
            static_cast<unsigned long long>(stats.total_build_us),
            static_cast<unsigned long long>(stats.max_build_us));
    }
#endif
}

std::unique_lock<std::mutex> PipelineCache::LockBackgroundWarmUpForForeground() {
    std::unique_lock<std::mutex> background_lock;
    if (!background_warmup_active.load(std::memory_order_acquire)) {
        return background_lock;
    }

    background_warmup_foreground_waiters.fetch_add(1, std::memory_order_acq_rel);
    const auto wait_started = std::chrono::steady_clock::now();
    if (background_warmup_active.load(std::memory_order_acquire)) {
        background_lock = std::unique_lock{background_warmup_mutex};
    }
    background_warmup_foreground_waiters.fetch_sub(1, std::memory_order_acq_rel);

    const auto waited_us = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - wait_started)
            .count());
    if (waited_us < 250) {
        return background_lock;
    }

    const u64 wait_events =
        background_warmup_wait_events.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 total_wait_us =
        background_warmup_wait_us.fetch_add(waited_us, std::memory_order_relaxed) + waited_us;
    u64 previous_max = background_warmup_max_wait_us.load(std::memory_order_relaxed);
    while (previous_max < waited_us &&
           !background_warmup_max_wait_us.compare_exchange_weak(
               previous_max, waited_us, std::memory_order_relaxed)) {
    }
#ifdef __ANDROID__
    if (wait_events == 1 || (wait_events & 255u) == 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_PIPELINE_WARMUP_ARBITRATION] waits=%llu totalWaitUs=%llu "
            "maxWaitUs=%llu active=%d",
            static_cast<unsigned long long>(wait_events),
            static_cast<unsigned long long>(total_wait_us),
            static_cast<unsigned long long>(
                background_warmup_max_wait_us.load(std::memory_order_relaxed)),
            background_warmup_active.load(std::memory_order_relaxed) ? 1 : 0);
    }
#endif
    return background_lock;
}

void PipelineCache::DiscardBackgroundWarmUpPipelinesLocked(
    std::vector<BackgroundWarmUpInfos>& retired_infos,
    std::vector<std::unique_ptr<GraphicsPipeline>>& retired_graphics,
    std::vector<std::unique_ptr<ComputePipeline>>& retired_compute) {
    for (auto info_it = background_warmup_graphics_infos.begin();
         info_it != background_warmup_graphics_infos.end(); ++info_it) {
        const auto& key = info_it.key();
        if (auto pipeline_it = graphics_pipelines.find(key);
            pipeline_it != graphics_pipelines.end()) {
            retired_graphics.push_back(std::move(pipeline_it.value()));
            graphics_pipelines.erase(pipeline_it);
        }
        retired_infos.push_back(std::move(info_it.value()));
    }
    for (auto info_it = background_warmup_compute_infos.begin();
         info_it != background_warmup_compute_infos.end(); ++info_it) {
        const auto& key = info_it.key();
        if (auto pipeline_it = compute_pipelines.find(key);
            pipeline_it != compute_pipelines.end()) {
            retired_compute.push_back(std::move(pipeline_it.value()));
            compute_pipelines.erase(pipeline_it);
        }
        retired_infos.push_back(std::move(info_it.value()));
    }
    background_warmup_graphics_infos.clear();
    background_warmup_compute_infos.clear();
    last_graphics_pipeline = nullptr;
    last_graphics_state_generation = std::numeric_limits<u64>::max();
    last_graphics_lookup_resolved = false;
}

bool PipelineCache::CaptureSrtSourceBytes(
    const std::span<const Shader::SrtMemoryRange> ranges, std::vector<u8>& bytes) const {
    size_t total_size = 0;
    for (const auto& range : ranges) {
        if (range.size > std::numeric_limits<size_t>::max() - total_size) {
            bytes.clear();
            return false;
        }
        total_size += static_cast<size_t>(range.size);
    }
    bytes.resize(total_size);
    size_t offset = 0;
    for (const auto& range : ranges) {
        std::memcpy(bytes.data() + offset,
                    reinterpret_cast<const void*>(range.address),
                    static_cast<size_t>(range.size));
        offset += static_cast<size_t>(range.size);
    }
    return true;
}

bool PipelineCache::CaptureSrtBackingBytes(
    const std::span<const Shader::SrtMemoryRange> ranges, std::vector<u8>& bytes) const {
    size_t total_size = 0;
    for (const auto& range : ranges) {
        if (range.address == 0 || range.size == 0 ||
            range.size > std::numeric_limits<size_t>::max() - total_size) {
            bytes.clear();
            return false;
        }
        total_size += static_cast<size_t>(range.size);
    }
    bytes.resize(total_size);
    auto* memory = Core::Memory::Instance();
    size_t offset = 0;
    for (const auto& range : ranges) {
        if (!memory->TryCopyBackingMemory(range.address, bytes.data() + offset, range.size)) {
            bytes.clear();
            return false;
        }
        offset += static_cast<size_t>(range.size);
    }
    return true;
}

bool PipelineCache::AreSrtMemoryRangesGpuClean(
    const std::span<const Shader::SrtMemoryRange> ranges) const {
    return std::ranges::all_of(ranges, [this](const auto& range) {
        return range.address != 0 && range.size != 0 &&
               !buffer_cache->IsRegionGpuModifiedExact(range.address, range.size);
    });
}

bool PipelineCache::DoSrtSourceBytesMatch(
    const std::span<const Shader::SrtMemoryRange> ranges,
    const std::span<const u8> source_bytes) const {
    size_t offset = 0;
    for (const auto& range : ranges) {
        if (range.size > source_bytes.size() - std::min(offset, source_bytes.size()) ||
            std::memcmp(reinterpret_cast<const void*>(range.address),
                        source_bytes.data() + offset,
                        static_cast<size_t>(range.size)) != 0) {
            return false;
        }
        offset += static_cast<size_t>(range.size);
    }
    return offset == source_bytes.size();
}

Program::SpecializationSnapshot* PipelineCache::FindSrtVictimSnapshot(
    const Program& program, const u64 identity_key) {
    ++srt_snapshot_victim_lookups;
    const SrtVictimKey key{&program, identity_key};
    const auto it = srt_snapshot_victims.find(key);
    if (it == srt_snapshot_victims.end()) {
        return nullptr;
    }
    auto& entry = it.value();
    if (!entry.snapshot || !entry.snapshot->valid) {
        return nullptr;
    }
    entry.referenced = true;
    return entry.snapshot.get();
}

std::unique_ptr<Program::SpecializationSnapshot> PipelineCache::RetainSrtVictimSnapshot(
    const Program& program, std::unique_ptr<Program::SpecializationSnapshot> snapshot) {
    if (!snapshot || !snapshot->valid) {
        return snapshot;
    }

    const SrtVictimKey key{&program, snapshot->identity_key};
    if (const auto existing = srt_snapshot_victims.find(key);
        existing != srt_snapshot_victims.end()) {
        auto& entry = existing.value();
        std::swap(entry.snapshot, snapshot);
        entry.referenced = true;
        ++srt_snapshot_victim_replacements;
        return snapshot;
    }

    std::unique_ptr<Program::SpecializationSnapshot> recycled;
    while (srt_snapshot_victims.size() >= MaxSrtVictimSnapshots &&
           !srt_snapshot_victim_clock.empty()) {
        const SrtVictimKey candidate = srt_snapshot_victim_clock.front();
        srt_snapshot_victim_clock.pop_front();
        const auto it = srt_snapshot_victims.find(candidate);
        if (it == srt_snapshot_victims.end()) {
            continue;
        }
        auto& entry = it.value();
        if (entry.referenced) {
            entry.referenced = false;
            srt_snapshot_victim_clock.push_back(candidate);
            continue;
        }
        recycled = std::move(entry.snapshot);
        srt_snapshot_victims.erase(it);
        ++srt_snapshot_victim_evictions;
        break;
    }
    if (srt_snapshot_victims.size() >= MaxSrtVictimSnapshots) {
        return snapshot;
    }

    if (srt_snapshot_victims.empty()) {
        srt_snapshot_victims.reserve(MaxSrtVictimSnapshots);
    }
    auto [it, inserted] = srt_snapshot_victims.try_emplace(key);
    if (!inserted) {
        auto& entry = it.value();
        std::swap(entry.snapshot, snapshot);
        entry.referenced = true;
        ++srt_snapshot_victim_replacements;
        return snapshot;
    }
    it.value().snapshot = std::move(snapshot);
    it.value().referenced = true;
    srt_snapshot_victim_clock.push_back(key);
    ++srt_snapshot_victim_insertions;
    return recycled;
}

void PipelineCache::InvalidateSrtSnapshots() {
    // The background warm-up worker may insert programs and permutations.
    // Serialize the full-program walk with those mutations.
    std::scoped_lock background_lock{background_warmup_mutex};
    ++background_warmup_generation;
    srt_snapshot_victims.clear();
    srt_snapshot_victim_clock.clear();
    for (auto& [_, program] : program_cache) {
        if (!program) {
            continue;
        }
        for (auto& snapshot : program->specialization_snapshots) {
            if (snapshot) {
                snapshot->valid = false;
            }
        }
        program->specialization_snapshot_count = 0;
    }
}

bool PipelineCache::WasCpuRangeInvalidatedSinceLocked(const VAddr range_address,
                                                      const u64 range_size,
                                                      const u64 sequence) const {
    if (range_size == 0 ||
        range_address > std::numeric_limits<VAddr>::max() - range_size) {
        return true;
    }
    const u64 published_generation =
        shader_identity_published_generation.load(std::memory_order_acquire);
    if (published_generation <= sequence) {
        return false;
    }
    if (published_generation - sequence > ShaderIdentityInvalidationHistorySize) {
        return true;
    }
    const VAddr range_end = range_address + range_size;
    for (u64 current = sequence + 1; current <= published_generation; ++current) {
        const auto& invalidation =
            shader_identity_invalidation_history
                [(current - 1) % ShaderIdentityInvalidationHistorySize];
        const u64 invalidation_sequence =
            invalidation.sequence.load(std::memory_order_acquire);
        const VAddr invalidation_address =
            invalidation.address.load(std::memory_order_relaxed);
        const u64 invalidation_size =
            invalidation.size.load(std::memory_order_relaxed);
        if (invalidation_sequence != current || invalidation_size == 0 ||
            invalidation_address >
                std::numeric_limits<VAddr>::max() - invalidation_size) {
            return true;
        }
        const VAddr invalidation_end = invalidation_address + invalidation_size;
        if (invalidation_address < range_end && range_address < invalidation_end) {
            return true;
        }
    }
    return false;
}

bool PipelineCache::AreSrtMemoryRangesCpuStableSince(
    const std::span<const Shader::SrtMemoryRange> ranges, const u64 sequence) {
    if (shader_identity_published_generation.load(std::memory_order_acquire) <= sequence) {
        return true;
    }
    return std::ranges::all_of(ranges, [&](const auto& range) {
        return !WasCpuRangeInvalidatedSinceLocked(range.address, range.size, sequence);
    });
}

u64 PipelineCache::CaptureCpuInvalidationSequence() {
    return shader_identity_published_generation.load(std::memory_order_acquire);
}

Shader::ShaderParams PipelineCache::GetShaderParams(const u32* user_data, const u32* code) {
    ASSERT(user_data != nullptr);
    ASSERT(code != nullptr);
    const std::span<const u32, Shader::ShaderParams::NumShaderUserData> fresh_user_data{
        user_data, Shader::ShaderParams::NumShaderUserData};
    const auto scan_metadata = [&] {
        const auto& binary_info = AmdGpu::SearchBinaryInfo(code);
        return Shader::ShaderParams{
            .user_data = fresh_user_data,
            .code = std::span{code, binary_info.length / sizeof(u32)},
            .hash = binary_info.shader_hash,
        };
    };
    if (!shader_identity_cache_enabled) {
        return scan_metadata();
    }

    const VAddr address = reinterpret_cast<VAddr>(code);
    ++shader_identity_lookups;
    const u64 scan_generation = CaptureCpuInvalidationSequence();
    if (auto it = shader_identity_cache.find(address); it != shader_identity_cache.end()) {
        ShaderIdentityMetadata& cached_metadata = it.value();
        if (!WasCpuRangeInvalidatedSinceLocked(
                cached_metadata.address, cached_metadata.length_bytes,
                cached_metadata.invalidation_sequence)) {
            cached_metadata.invalidation_sequence = CaptureCpuInvalidationSequence();
            ++shader_identity_hits;
            ++shader_identity_gpu_alias_hits;
            return {
                .user_data = fresh_user_data,
                .code = std::span{code, cached_metadata.length_bytes / sizeof(u32)},
                .hash = cached_metadata.hash,
            };
        }
        ++shader_identity_invalidation_misses;
        ++shader_identity_cpu_erases;
        shader_identity_cache.erase(it);
    }

    const Shader::ShaderParams params = scan_metadata();
    const size_t length_bytes = params.code.size_bytes();
    const bool valid_length =
        length_bytes != 0 && length_bytes <= std::numeric_limits<u32>::max() &&
        address <= std::numeric_limits<VAddr>::max() - length_bytes;
    ++shader_identity_scans;
    if (!valid_length ||
        WasCpuRangeInvalidatedSinceLocked(address, length_bytes, scan_generation)) {
        return params;
    }
    if (!shader_identity_cache.contains(address) &&
        shader_identity_cache.size() >= MaxShaderIdentityEntriesPerSubmit) {
        ++shader_identity_capacity_misses;
        shader_identity_submit_erases += shader_identity_cache.size();
        shader_identity_cache.clear();
    }
    shader_identity_cache.insert_or_assign(
        address, ShaderIdentityMetadata{
                     .address = address,
                     .length_bytes = static_cast<u32>(length_bytes),
                     .hash = params.hash,
                     .invalidation_sequence = CaptureCpuInvalidationSequence(),
                 });
    return params;
}

void PipelineCache::InvalidateShaderIdentityRange(const VAddr address, const u64 size) {
    if (!shader_identity_cache_enabled || size == 0) {
        return;
    }

    const u64 invalidation_sequence =
        shader_identity_published_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
    auto& invalidation =
        shader_identity_invalidation_history
            [(invalidation_sequence - 1) % ShaderIdentityInvalidationHistorySize];
    // Publish the sequence last. Readers either observe this exact immutable
    // record or conservatively reject the cached snapshot while a writer is
    // filling it; no renderer-thread mutex or whole-cache scan is required.
    invalidation.sequence.store(0, std::memory_order_release);
    invalidation.address.store(address, std::memory_order_relaxed);
    invalidation.size.store(size, std::memory_order_relaxed);
    invalidation.sequence.store(invalidation_sequence, std::memory_order_release);
}

void PipelineCache::InvalidateShaderIdentityReadbackRange(const VAddr address, const u64 size) {
    // A GPU readback publishes the latest contents into the CPU backing store; it is not a
    // guest CPU write. Shader programs are CPU-uploaded and immutable while referenced by a
    // pipeline, so invalidating their identity here makes the metadata scan fault on the same
    // GPU-protected page again on every draw. Keep this behaviour opt-in because a title which
    // genuinely generates shader code on the GPU still needs conservative invalidation.
    if (immutable_shader_identity_cache_enabled) {
        return;
    }
    InvalidateShaderIdentityRange(address, size);
}

void PipelineCache::AdvanceShaderIdentitySubmit() {
    MaybeCheckpointPipelineCaches();
    if (!shader_identity_cache_enabled) {
        return;
    }

    ++shader_identity_submits;
    const size_t entries = shader_identity_cache.size();
#ifdef __ANDROID__
    if (shader_identity_submits <= 4 || (shader_identity_submits & 63u) == 0) {
        const double hit_percent =
            shader_identity_lookups != 0
                ? 100.0 * static_cast<double>(shader_identity_hits) /
                      static_cast<double>(shader_identity_lookups)
                : 0.0;
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_SHADER_IDENTITY_CACHE] submits=%llu lookups=%llu hits=%llu hitPct=%.1f "
            "scans=%llu gpuDirtyMiss=%llu backingHits=%llu invalidationMiss=%llu dirtyAfterScan=%llu "
            "cpuErases=%llu capacityErases=%llu capacityMiss=%llu entries=%zu",
            static_cast<unsigned long long>(shader_identity_submits),
            static_cast<unsigned long long>(shader_identity_lookups),
            static_cast<unsigned long long>(shader_identity_hits), hit_percent,
            static_cast<unsigned long long>(shader_identity_scans),
            static_cast<unsigned long long>(shader_identity_gpu_dirty_misses),
            static_cast<unsigned long long>(shader_identity_gpu_alias_hits),
            static_cast<unsigned long long>(shader_identity_invalidation_misses),
            static_cast<unsigned long long>(shader_identity_dirty_after_scan),
            static_cast<unsigned long long>(shader_identity_cpu_erases),
            static_cast<unsigned long long>(shader_identity_submit_erases),
            static_cast<unsigned long long>(shader_identity_capacity_misses), entries);
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_FETCH_SHADER_CACHE] lookups=%llu epochHits=%llu backingHits=%llu "
            "reparses=%llu backingFallbacks=%llu racedPublications=%llu "
            "tessLookups=%llu tessHits=%llu tessReads=%llu tessBacking=%llu",
            static_cast<unsigned long long>(fetch_shader_lookups),
            static_cast<unsigned long long>(fetch_shader_epoch_hits),
            static_cast<unsigned long long>(fetch_shader_backing_hits),
            static_cast<unsigned long long>(fetch_shader_reparses),
            static_cast<unsigned long long>(fetch_shader_backing_fallbacks),
            static_cast<unsigned long long>(fetch_shader_raced_publications),
            static_cast<unsigned long long>(tess_constants_lookups),
            static_cast<unsigned long long>(tess_constants_hits),
            static_cast<unsigned long long>(tess_constants_reads),
            static_cast<unsigned long long>(tess_constants_backing_copies));
    }
#endif
}

std::optional<Shader::TessellationDataConstantBuffer>
PipelineCache::ResolveTessConstants(const Shader::Info& info) {
    if (info.l_stage != Shader::LogicalStage::TessellationControl &&
        info.l_stage != Shader::LogicalStage::TessellationEval) {
        return std::nullopt;
    }
    ++tess_constants_lookups;
    const auto buffer = info.ReadUdReg<AmdGpu::Buffer>(
        static_cast<u32>(info.tess_consts_ptr_base),
        static_cast<u32>(info.tess_consts_dword_offset));
    const VAddr address = buffer.base_address;
    constexpr u64 size = sizeof(Shader::TessellationDataConstantBuffer);
    const std::array range{Shader::SrtMemoryRange{.address = address, .size = size}};
    if (const auto it = tess_constants_snapshots.find(address);
        it != tess_constants_snapshots.end() &&
        AreSrtMemoryRangesCpuStableSince(range, it.value().invalidation_sequence)) {
        it.value().invalidation_sequence = CaptureCpuInvalidationSequence();
        ++tess_constants_hits;
        return it.value().data;
    }

    const u64 capture_sequence = CaptureCpuInvalidationSequence();
    Shader::TessellationDataConstantBuffer data{};
    // This control block is authored by the guest CPU and consumed read-only
    // by tessellation. Large aliased upload buffers can nevertheless inherit a
    // GPU-modified page bit, making a 36-byte host read wait for the entire
    // Vulkan timeline. The managed identity mode already invalidates on every
    // guest CPU publication, so its backing copy is authoritative here.
    const bool copied_from_backing =
        immutable_shader_identity_cache_enabled &&
        Core::Memory::Instance()->TryCopyBackingMemory(address, &data, sizeof(data));
    if (copied_from_backing) {
        ++tess_constants_backing_copies;
    } else {
        std::memcpy(&data, reinterpret_cast<const void*>(address), sizeof(data));
    }
    ++tess_constants_reads;
    if (!AreSrtMemoryRangesCpuStableSince(range, capture_sequence)) {
        return data;
    }
    if (!tess_constants_snapshots.contains(address) &&
        tess_constants_snapshots.size() >= MaxTessConstantsSnapshots) {
        tess_constants_snapshots.clear();
    }
    tess_constants_snapshots.insert_or_assign(
        address, TessConstantsSnapshot{
                     .data = data,
                     .invalidation_sequence = CaptureCpuInvalidationSequence(),
                 });
    return data;
}

const std::optional<Shader::Gcn::FetchShaderData>& PipelineCache::ResolveFetchShader(
    Program& program, const Shader::Info& info) {
    auto& snapshot = program.fetch_shader_snapshot;
    ++fetch_shader_lookups;
    if (!info.has_fetch_shader) {
        snapshot.parsed.reset();
        snapshot.code.clear();
        snapshot.address = 0;
        snapshot.invalidation_sequence = CaptureCpuInvalidationSequence();
        snapshot.initialized = true;
        return snapshot.parsed;
    }

    const u32* code =
        Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
    const VAddr address = reinterpret_cast<VAddr>(code);
    if (snapshot.initialized && snapshot.parsed &&
        snapshot.code.size() == snapshot.parsed->size) {
        const size_t size_bytes = snapshot.code.size() * sizeof(u32);
        if (snapshot.address == address && AreSrtMemoryRangesCpuStableSince(
                                               std::array{Shader::SrtMemoryRange{
                                                   .address = address, .size = size_bytes}},
                                               snapshot.invalidation_sequence)) {
            snapshot.invalidation_sequence = CaptureCpuInvalidationSequence();
            ++fetch_shader_epoch_hits;
            return snapshot.parsed;
        }

        const u64 validation_sequence = CaptureCpuInvalidationSequence();
        bool bytes_match = false;
        if (immutable_shader_identity_cache_enabled) {
            fetch_shader_backing_validation_scratch.resize(snapshot.code.size());
            const bool copied =
                size_bytes == 0 ||
                Core::Memory::Instance()->TryCopyBackingMemory(
                    address, fetch_shader_backing_validation_scratch.data(), size_bytes);
            bytes_match = copied &&
                          std::ranges::equal(fetch_shader_backing_validation_scratch,
                                             snapshot.code);
        } else {
            bytes_match = size_bytes == 0 ||
                          std::memcmp(snapshot.code.data(), code, size_bytes) == 0;
        }
        u64 publish_sequence{};
        bool stable = false;
        const u64 generation_before_validation = CaptureCpuInvalidationSequence();
        stable = bytes_match &&
                 !WasCpuRangeInvalidatedSinceLocked(address, size_bytes,
                                                   validation_sequence);
        publish_sequence = CaptureCpuInvalidationSequence();
        stable &= publish_sequence == generation_before_validation;
        if (stable) {
            snapshot.address = address;
            snapshot.invalidation_sequence = publish_sequence;
            ++fetch_shader_backing_hits;
            return snapshot.parsed;
        }
    }

    ++fetch_shader_reparses;
    const u64 parse_sequence = CaptureCpuInvalidationSequence();
    auto parsed = Shader::Gcn::ParseFetchShader(info);
    std::vector<u32> parsed_code;
    bool captured_from_backing = false;
    if (parsed) {
        parsed_code.resize(parsed->size);
        if (!parsed_code.empty()) {
            if (immutable_shader_identity_cache_enabled) {
                captured_from_backing =
                    Core::Memory::Instance()->TryCopyBackingMemory(
                        address, parsed_code.data(), parsed_code.size() * sizeof(u32));
            }
            if (!captured_from_backing) {
                fetch_shader_backing_fallbacks += immutable_shader_identity_cache_enabled;
                std::memcpy(parsed_code.data(), code, parsed_code.size() * sizeof(u32));
            }
        }
    }

    u64 publish_sequence{};
    bool stable_publication = parsed.has_value();
    if (parsed) {
        const u64 generation_before_validation = CaptureCpuInvalidationSequence();
        stable_publication = !WasCpuRangeInvalidatedSinceLocked(
            address, parsed_code.size() * sizeof(u32), parse_sequence);
        publish_sequence = CaptureCpuInvalidationSequence();
        stable_publication &= publish_sequence == generation_before_validation;
    }
    snapshot.parsed = std::move(parsed);
    snapshot.code = std::move(parsed_code);
    snapshot.address = stable_publication ? address : 0;
    snapshot.invalidation_sequence = publish_sequence;
    snapshot.initialized = stable_publication;
    fetch_shader_raced_publications += !stable_publication;
    return snapshot.parsed;
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(
    std::optional<AmdGpu::PrimitiveType> host_primitive_override) {
    EnsureAsyncPipelineMode();
    if (Config::managedAsyncPipelineCompilation()) {
        async_last_foreground_activity_ms.store(
            AsyncSteadyMilliseconds(), std::memory_order_release);
    }
    PublishAsyncPipelineResults();
    const u64 state_generation = liverpool->GraphicsStateGeneration();
    const u64 other_generation = liverpool->GraphicsOtherGeneration();
    const u64 shader_program_generation =
        liverpool->GraphicsShaderProgramGeneration();
    const u64 shader_user_data_generation =
        liverpool->GraphicsShaderUserDataGeneration();
    const u64 structural_signature =
        GraphicsStructuralSignature(liverpool->regs, host_primitive_override);
    const GraphicsPipeline* const previous_pipeline = last_graphics_pipeline;
    const bool generation_changed = last_graphics_state_generation != state_generation;
    const bool managed_state_cache =
#ifdef __ANDROID__
        VideoCore::IsManagedVulkanDriverOptimization();
#else
        false;
#endif
    const auto commit_lookup_state = [&] {
        last_graphics_state_generation = state_generation;
        last_graphics_other_generation = other_generation;
        last_graphics_shader_program_generation = shader_program_generation;
        last_graphics_shader_user_data_generation = shader_user_data_generation;
        last_graphics_structural_signature = structural_signature;
        last_graphics_primitive_override = host_primitive_override;
        last_graphics_lookup_resolved = true;
    };
#ifdef __ANDROID__
    auto& telemetry = graphics_pipeline_telemetry;
    ++telemetry.lookups;
    telemetry.generation_misses += generation_changed;
    const auto maybe_log_telemetry = [&] {
        const u64 generation_misses = telemetry.generation_misses;
        const bool log_checkpoint =
            generation_misses == 256 ||
            (generation_misses != 0 && (generation_misses & 4095u) == 0);
        if (!log_checkpoint ||
            telemetry.last_logged_generation_miss == generation_misses) {
            return;
        }
        const auto dirty = liverpool->GetGraphicsDirtyTelemetrySnapshot();
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_GP_CACHE_TELEMETRY] lookups=%llu refreshes=%llu generationMiss=%llu "
            "refreshSameFinalKey=%llu sameStageHashes=%llu actualPipelineChanged=%llu "
            "refreshFailures=%llu irrelevantStateHits=%llu userDataOnlyRefreshes=%llu "
            "dirtyProgram=%llu dirtyUserData=%llu dirtyOther=%llu",
            static_cast<unsigned long long>(telemetry.lookups),
            static_cast<unsigned long long>(telemetry.refreshes),
            static_cast<unsigned long long>(generation_misses),
            static_cast<unsigned long long>(telemetry.refresh_same_final_key),
            static_cast<unsigned long long>(telemetry.same_stage_hashes),
            static_cast<unsigned long long>(telemetry.actual_pipeline_changed),
            static_cast<unsigned long long>(telemetry.refresh_failures),
            static_cast<unsigned long long>(telemetry.irrelevant_state_hits),
            static_cast<unsigned long long>(telemetry.user_data_only_refreshes),
            static_cast<unsigned long long>(dirty.shader_program -
                                            telemetry.last_logged_dirty_program),
            static_cast<unsigned long long>(dirty.shader_user_data -
                                            telemetry.last_logged_dirty_user_data),
            static_cast<unsigned long long>(dirty.other -
                                            telemetry.last_logged_dirty_other));
        telemetry.last_logged_generation_miss = generation_misses;
        telemetry.last_logged_dirty_program = dirty.shader_program;
        telemetry.last_logged_dirty_user_data = dirty.shader_user_data;
        telemetry.last_logged_dirty_other = dirty.other;
    };
#endif
    if (last_graphics_lookup_resolved &&
        last_graphics_state_generation == state_generation &&
        last_graphics_primitive_override == host_primitive_override) {
        return last_graphics_pipeline;
    }

    // Games rewrite attachment addresses, viewport/scissor state and other
    // dynamic registers between draws. Those writes must remain visible to
    // Rasterizer, but none of them selects a Vulkan pipeline. Avoid rebuilding
    // all shader stages when the normalized structural state is unchanged.
    if (managed_state_cache && last_graphics_lookup_resolved &&
        last_graphics_primitive_override == host_primitive_override &&
        last_graphics_shader_program_generation == shader_program_generation &&
        last_graphics_shader_user_data_generation == shader_user_data_generation &&
        last_graphics_structural_signature == structural_signature) {
        commit_lookup_state();
#ifdef __ANDROID__
        ++telemetry.irrelevant_state_hits;
        maybe_log_telemetry();
#endif
        return last_graphics_pipeline;
    }

    auto background_lock = LockBackgroundWarmUpForForeground();
    graphics_primitive_override = host_primitive_override;
#ifdef __ANDROID__
    ++telemetry.refreshes;
#endif
    const bool user_data_only_refresh =
        managed_state_cache && previous_pipeline != nullptr &&
        last_graphics_lookup_resolved &&
        last_graphics_primitive_override == host_primitive_override &&
        last_graphics_other_generation == other_generation &&
        last_graphics_shader_program_generation == shader_program_generation &&
        last_graphics_shader_user_data_generation != shader_user_data_generation &&
        last_graphics_structural_signature == structural_signature;
    const bool refresh_ok =
        user_data_only_refresh ? RefreshGraphicsStages(false)
                               : RefreshGraphicsKey(host_primitive_override);
#ifdef __ANDROID__
    telemetry.user_data_only_refreshes += user_data_only_refresh;
#endif
#ifdef __ANDROID__
    if (generation_changed && refresh_ok && previous_pipeline != nullptr) {
        const auto& previous_key = previous_pipeline->GetGraphicsKey();
        telemetry.refresh_same_final_key += graphics_key == previous_key;
        telemetry.same_stage_hashes += graphics_key.stage_hashes == previous_key.stage_hashes;
    }
#endif
#ifdef __ANDROID__
    {
        static std::atomic<int> s_gp{0};
        int gp_n = s_gp.load(std::memory_order_relaxed);
        if (gp_n < 24 &&
            (gp_n = s_gp.fetch_add(1, std::memory_order_relaxed)) < 24) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_GP] n=%d refreshKey=%d vsInfo=%d psInfo=%d", gp_n,
                                (int)refresh_ok, infos[static_cast<u32>(Shader::LogicalStage::Vertex)] != nullptr,
                                infos[static_cast<u32>(Shader::LogicalStage::Fragment)] != nullptr);
        }
    }
#endif
    if (!refresh_ok) {
        last_graphics_pipeline = nullptr;
        last_graphics_lookup_resolved = false;
#ifdef __ANDROID__
        ++telemetry.refresh_failures;
        maybe_log_telemetry();
#endif
        return nullptr;
    }
    // User-data/SRT addresses legitimately change on nearly every draw, but
    // they frequently resolve to the same shader permutations and fixed
    // graphics state. RefreshGraphicsKey must still update Info/bindings for
    // those resources; once the final key is identical, avoid another large
    // robin-map lookup and pipeline validity walk.
    if (previous_pipeline != nullptr &&
        graphics_key == previous_pipeline->GetGraphicsKey()) {
        last_graphics_pipeline = previous_pipeline;
        commit_lookup_state();
#ifdef __ANDROID__
        maybe_log_telemetry();
#endif
        return previous_pipeline;
    }
#ifdef __ANDROID__
    static const bool kBisectSkipPipelineCtor =
        std::getenv("EXECUTOR_BISECT_SKIP_PIPELINE_CTOR") != nullptr;
    if (kBisectSkipPipelineCtor) {
        static std::atomic<int> once{0};
        if (once.fetch_add(1) == 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_BISECT_SKIP_PIPELINE_CTOR] active (ran RefreshGraphicsKey, "
                                "skipping GraphicsPipeline ctor)");
        }
        return nullptr;
    }
#endif
    if (IsKnownInvalid(graphics_key)) {
        graphics_pipelines.try_emplace(graphics_key);
        last_graphics_pipeline = nullptr;
        commit_lookup_state();
        return nullptr;
    }
    auto it = graphics_pipelines.find(graphics_key);
    if (it == graphics_pipelines.end() &&
        Config::managedAsyncPipelineCompilation() &&
        !DeferAsyncLiveReplacement(instance)) {
        async_pipeline_stats.graphics_misses.fetch_add(1,
                                                       std::memory_order_relaxed);
        GraphicsPipeline::SerializationSupport sdata{};
        GraphicsPipeline::PrepareSerializationSupport(
            instance, graphics_key, infos, runtime_infos, fetch_shader, sdata);

        if (instance.IsPipelineCreationCacheControlSupported() &&
            sdata.tcs.empty() && sdata.tes.empty()) {
            const auto pipeline_hash =
                std::hash<GraphicsPipelineKey>{}(graphics_key);
            auto fast_pipeline = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, graphics_key,
                *pipeline_cache, infos, runtime_infos, fetch_shader, modules,
                sdata, true,
                vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired);
            if (fast_pipeline->IsValid()) {
                async_pipeline_stats.fail_fast_hits.fetch_add(
                    1, std::memory_order_relaxed);
                RecordValid(graphics_key);
                RegisterPipelineData(graphics_key, pipeline_hash, sdata);
                ++num_new_pipelines;
                auto [fast_it, inserted] = graphics_pipelines.try_emplace(
                    graphics_key, std::move(fast_pipeline));
                ASSERT(inserted);
                for (size_t stage = 0; stage < MaxShaderStages; ++stage) {
                    if (infos[stage]) {
                        module_related_pipelines[modules[stage]].emplace_back(
                            graphics_key);
                    }
                }
                MarkHostPipelineCacheDirty();
                it = fast_it;
            } else if (fast_pipeline->WasCompileRequired()) {
                async_pipeline_stats.fail_fast_compile_required.fetch_add(
                    1, std::memory_order_relaxed);
            } else {
                if (fast_pipeline->BuildResult() ==
                    vk::Result::eErrorDeviceLost) {
                    scheduler.MarkTerminal(fast_pipeline->BuildResult(),
                                           "graphics pipeline fail-fast");
                } else {
                    RecordInvalid(graphics_key);
                }
                graphics_pipelines.try_emplace(graphics_key);
                it = graphics_pipelines.find(graphics_key);
            }
        }

        if (it == graphics_pipelines.end()) {
            auto foreground_pipeline = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, graphics_key,
                *pipeline_cache, infos, runtime_infos, fetch_shader, modules,
                sdata, true,
                vk::PipelineCreateFlagBits::eDisableOptimization);
            if (foreground_pipeline->IsValid()) {
                const auto pipeline_hash =
                    std::hash<GraphicsPipelineKey>{}(graphics_key);
                RecordValid(graphics_key);
                RegisterPipelineData(graphics_key, pipeline_hash, sdata);
                ++num_new_pipelines;
                auto [foreground_it, inserted] =
                    graphics_pipelines.try_emplace(
                        graphics_key, std::move(foreground_pipeline));
                ASSERT(inserted);
                fallback_graphics_pipelines.emplace(graphics_key);
                for (size_t stage = 0; stage < MaxShaderStages; ++stage) {
                    if (infos[stage]) {
                        module_related_pipelines[modules[stage]].emplace_back(
                            graphics_key);
                    }
                }
                async_pipeline_stats.foreground_fallbacks.fetch_add(
                    1, std::memory_order_relaxed);
                MarkHostPipelineCacheDirty();
                it = foreground_it;
                QueueAsyncGraphicsPipeline(
                    graphics_key, infos, runtime_infos, fetch_shader, modules,
                    sdata, true);
            } else {
                if (foreground_pipeline->BuildResult() ==
                    vk::Result::eErrorDeviceLost) {
                    scheduler.MarkTerminal(foreground_pipeline->BuildResult(),
                                           "graphics foreground fallback");
                } else {
                    RecordInvalid(graphics_key);
                }
                graphics_pipelines.try_emplace(graphics_key);
                it = graphics_pipelines.find(graphics_key);
            }
        }
    }
    if (it == graphics_pipelines.end()) {
        const auto [new_it, is_new] =
            graphics_pipelines.try_emplace(graphics_key);
        it = new_it;
        ASSERT(is_new);
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);

        if (it.value()->IsValid()) {
            RecordValid(graphics_key);
            RegisterPipelineData(graphics_key, pipeline_hash, sdata);
            ++num_new_pipelines;
            MarkHostPipelineCacheDirty();
        } else {
            RecordInvalid(graphics_key);
        }

        for (auto stage = 0; stage < MaxShaderStages; ++stage) {
            if (infos[stage]) {
                auto& m = modules[stage];
                module_related_pipelines[m].emplace_back(graphics_key);
            }
        }
        fetch_shader.reset();
    }
    const auto* pipeline = it->second.get();
    last_graphics_pipeline = pipeline && pipeline->IsValid() ? pipeline : nullptr;
    commit_lookup_state();
#ifdef __ANDROID__
    telemetry.actual_pipeline_changed += last_graphics_pipeline != previous_pipeline;
    maybe_log_telemetry();
#endif
    return last_graphics_pipeline;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    EnsureAsyncPipelineMode();
    if (Config::managedAsyncPipelineCompilation()) {
        async_last_foreground_activity_ms.store(
            AsyncSteadyMilliseconds(), std::memory_order_release);
    }
    PublishAsyncPipelineResults();
    auto background_lock = LockBackgroundWarmUpForForeground();
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    if (IsKnownInvalid(compute_key)) {
        compute_pipelines.try_emplace(compute_key);
        return nullptr;
    }
    auto it = compute_pipelines.find(compute_key);
    if (it == compute_pipelines.end() &&
        Config::managedAsyncPipelineCompilation() &&
        !DeferAsyncLiveReplacement(instance)) {
        async_pipeline_stats.compute_misses.fetch_add(1,
                                                      std::memory_order_relaxed);
        ComputePipeline::SerializationSupport sdata{};
        sdata.storage_buffers.reserve(infos[0]->buffers.size());
        for (const auto& buffer : infos[0]->buffers) {
            sdata.storage_buffers.push_back(
                buffer.IsStorage(buffer.GetSharp(*infos[0])) ? 1u : 0u);
        }
        sdata.image_bindings.reserve(infos[0]->images.size());
        for (const auto& image : infos[0]->images) {
            sdata.image_bindings.push_back(image.NumBindings(*infos[0]));
        }

        if (instance.IsPipelineCreationCacheControlSupported()) {
            auto fast_pipeline = std::make_unique<ComputePipeline>(
                instance, scheduler, desc_heap, profile, *pipeline_cache,
                compute_key, *infos[0], modules[0], sdata, true,
                vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired);
            if (fast_pipeline->IsValid()) {
                async_pipeline_stats.fail_fast_hits.fetch_add(
                    1, std::memory_order_relaxed);
                RecordValid(compute_key);
                RegisterPipelineData(compute_key, sdata);
                ++num_new_pipelines;
                auto [fast_it, inserted] = compute_pipelines.try_emplace(
                    compute_key, std::move(fast_pipeline));
                ASSERT(inserted);
                module_related_pipelines[modules[0]].emplace_back(compute_key);
                MarkHostPipelineCacheDirty();
                it = fast_it;
            } else if (fast_pipeline->WasCompileRequired()) {
                async_pipeline_stats.fail_fast_compile_required.fetch_add(
                    1, std::memory_order_relaxed);
            } else {
                if (fast_pipeline->BuildResult() ==
                    vk::Result::eErrorDeviceLost) {
                    scheduler.MarkTerminal(fast_pipeline->BuildResult(),
                                           "compute pipeline fail-fast");
                } else {
                    RecordInvalid(compute_key);
                }
                compute_pipelines.try_emplace(compute_key);
                it = compute_pipelines.find(compute_key);
            }
        }

        if (it == compute_pipelines.end()) {
            auto foreground_pipeline = std::make_unique<ComputePipeline>(
                instance, scheduler, desc_heap, profile, *pipeline_cache,
                compute_key, *infos[0], modules[0], sdata, true,
                vk::PipelineCreateFlagBits::eDisableOptimization);
            if (foreground_pipeline->IsValid()) {
                RecordValid(compute_key);
                RegisterPipelineData(compute_key, sdata);
                ++num_new_pipelines;
                auto [foreground_it, inserted] =
                    compute_pipelines.try_emplace(
                        compute_key, std::move(foreground_pipeline));
                ASSERT(inserted);
                fallback_compute_pipelines.emplace(compute_key);
                module_related_pipelines[modules[0]].emplace_back(compute_key);
                async_pipeline_stats.foreground_fallbacks.fetch_add(
                    1, std::memory_order_relaxed);
                MarkHostPipelineCacheDirty();
                it = foreground_it;
                QueueAsyncComputePipeline(compute_key, *infos[0], modules[0],
                                          sdata, true);
            } else {
                if (foreground_pipeline->BuildResult() ==
                    vk::Result::eErrorDeviceLost) {
                    scheduler.MarkTerminal(foreground_pipeline->BuildResult(),
                                           "compute foreground fallback");
                } else {
                    RecordInvalid(compute_key);
                }
                compute_pipelines.try_emplace(compute_key);
                it = compute_pipelines.find(compute_key);
            }
        }
    }
    if (it == compute_pipelines.end()) {
        const auto [new_it, is_new] =
            compute_pipelines.try_emplace(compute_key);
        it = new_it;
        ASSERT(is_new);
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        if (it.value()->IsValid()) {
            RecordValid(compute_key);
            RegisterPipelineData(compute_key, sdata);
            ++num_new_pipelines;
            MarkHostPipelineCacheDirty();
        } else {
            RecordInvalid(compute_key);
        }

        auto& m = modules[0];
        module_related_pipelines[m].emplace_back(compute_key);
    }
    const auto* pipeline = it->second.get();
    return pipeline && pipeline->IsValid() ? pipeline : nullptr;
}

bool PipelineCache::RefreshGraphicsKey(
    std::optional<AmdGpu::PrimitiveType> host_primitive_override) {
    std::memset(&graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
#ifdef __ANDROID__
    key.rasterizer_discard_enable = regs.clipper_control.dx_rasterization_kill;
#endif
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = host_primitive_override.value_or(regs.primitive_type);
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            continue;
        }

        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
    }

    if (!RefreshGraphicsStages()) {
        return false;
    }

    u8 color_samples = 0;
    bool all_color_samples_same = true;
    bool has_active_color_target = false;
    bool has_rop3_enabled_target = false;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        has_active_color_target = true;
        has_rop3_enabled_target |= !regs.blend_control[cb].disable_rop3;

        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    if (has_active_color_target && !has_rop3_enabled_target) {
        key.logic_op = AmdGpu::ColorControl::LogicOp::Copy;
    }

    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages(const bool refresh_runtime_info) {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = std::nullopt;
    key.stage_hashes.fill(0);
    key.vertex_buffer_formats.fill({});
    key.mrt_mask = 0;
    key.num_color_attachments = 0;

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](Shader::Stage stage_in, Shader::LogicalStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params =
            GetShaderParams(pgm->user_data.data(), pgm->Address<const u32*>());
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader_;
        std::tie(infos[stage_out_idx], modules[stage_out_idx], fetch_shader_,
                 key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding, refresh_runtime_info);
        if (fetch_shader_) {
            fetch_shader = fetch_shader_;
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);
    if (refresh_runtime_info) {
        runtime_infos.fill({});
    }
#ifdef __ANDROID__
    {
        static std::atomic<int> s_rgs{0};
        int rgs_n = s_rgs.load(std::memory_order_relaxed);
        if (rgs_n < 24 &&
            (rgs_n = s_rgs.fetch_add(1, std::memory_order_relaxed)) < 24) {
            const auto* vpgm = regs.ProgramForStage(static_cast<u32>(Stage::Vertex));
            const auto* fpgm = regs.ProgramForStage(static_cast<u32>(Stage::Fragment));
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_RGS] n=%d stage_enable=0x%x fsEn=%d vsEn=%d fsAddr=0x%llx vsAddr=0x%llx",
                rgs_n, static_cast<unsigned>(regs.stage_enable.raw),
                (int)regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Fragment)),
                (int)regs.stage_enable.IsStageEnabled(static_cast<u32>(Stage::Vertex)),
                (unsigned long long)(fpgm ? reinterpret_cast<u64>(fpgm->Address<const u8*>()) : 0ull),
                (unsigned long long)(vpgm ? reinterpret_cast<u64>(vpgm->Address<const u8*>()) : 0ull));
        }
    }
#endif
    const auto result = bind_stage(Stage::Fragment, LogicalStage::Fragment);
    if (!result && regs.vs_output_control.clip_distance_enable &&
        profile.needs_clip_distance_emulation) {
        LOG_WARNING(Render_Vulkan,
                    "Clip distance emulation is ineffective due to absense of fragment shader");
    }

    const auto* fs_info = infos[static_cast<u32>(LogicalStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    const u32 raw_num_color_attachments = static_cast<u32>(std::bit_width(key.mrt_mask));
    key.num_color_attachments = std::min<u32>(raw_num_color_attachments, AmdGpu::NUM_COLOR_BUFFERS);
#ifdef __ANDROID__
    if (raw_num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_MRT_OOB] mrt_mask=0x%x raw_num_color_attachments=%u > %u -> "
                            "clamped (would overrun pipeline color arrays / driver attachmentCount)",
                            static_cast<unsigned>(key.mrt_mask), raw_num_color_attachments,
                            static_cast<unsigned>(AmdGpu::NUM_COLOR_BUFFERS));
    }
#endif

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported() ||
            (regs.tess_config.type == AmdGpu::TessellationType::Isoline &&
             !instance.IsTessellationIsolinesSupported())) {
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Vertex, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported() ||
            (regs.tess_config.type == AmdGpu::TessellationType::Isoline &&
             !instance.IsTessellationIsolinesSupported())) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(Stage::Hull, LogicalStage::TessellationControl)) {
            return false;
        }
        if (!bind_stage(Stage::Export, LogicalStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(Stage::Local, LogicalStage::Vertex)) {
            return false;
        }
        if (!bind_stage(Stage::Geometry, LogicalStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(Stage::Vertex, LogicalStage::Vertex);
        break;
    default:
        UNREACHABLE_MSG("unhandled stage_en: {}", static_cast<u32>(regs.stage_enable.raw));
    }

    const auto* vs_info = infos[static_cast<u32>(Shader::LogicalStage::Vertex)];
    if (vs_info && fetch_shader && !instance.IsVertexInputDynamicState()) {
        u32 vertex_binding = 0;
        for (const auto& attrib : fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT(vertex_binding < MaxVertexBufferCount);
            key.vertex_buffer_formats[vertex_binding++] =
                Vulkan::LiverpoolToVK::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto cs_params =
        GetShaderParams(cs_pgm.user_data.data(), cs_pgm.Address<const u32*>());
    std::tie(infos[0], modules[0], fetch_shader, compute_key.value) =
        GetProgram(Shader::Stage::Compute, LogicalStage::Compute, cs_params, binding);
    return true;
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                               const std::span<const u32>& code, size_t perm_idx,
                                               Shader::Backend::Bindings& binding,
                                               std::shared_ptr<AsyncShaderPayload>*
                                                   optimization_payload) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");

    if (optimization_payload) {
        optimization_payload->reset();
    }
    std::shared_ptr<AsyncShaderPayload> shader_payload;
    std::vector<u32> spv;
    bool use_fast_tier =
        optimization_payload != nullptr &&
        Config::managedAsyncPipelineCompilation() &&
        !DeferAsyncLiveReplacement(instance);
    if (use_fast_tier &&
        IsShaderBinaryTierKnownInvalid(info.pgm_hash, perm_idx,
                                       ShaderBinaryTier::Fast)) {
        use_fast_tier = false;
        async_pipeline_stats.shader_negative_hits.fetch_add(
            1, std::memory_order_relaxed);
    }
    if (use_fast_tier) {
        shader_payload = std::make_shared<AsyncShaderPayload>();
        std::ranges::copy(info.user_data, shader_payload->user_data.begin());
        shader_payload->code.assign(code.begin(), code.end());
        shader_payload->source_runtime_info = runtime_info;
        shader_payload->runtime_info = runtime_info;
        shader_payload->binding_start = binding;
        const Shader::ShaderParams owned_params{
            .user_data = shader_payload->user_data,
            .code = shader_payload->code,
            .hash = info.pgm_hash,
        };
        shader_payload->info =
            std::make_unique<Shader::Info>(info.stage, info.l_stage,
                                           owned_params);
        shader_payload->pools = std::make_unique<Shader::Pools>();
        shader_payload->ir = std::make_unique<Shader::IR::Program>(
            Shader::TranslateProgram(shader_payload->code,
                                     *shader_payload->pools,
                                     *shader_payload->info,
                                     shader_payload->runtime_info, profile,
                                     Shader::TranslationTier::Fast));
        spv = Shader::Backend::SPIRV::EmitSPIRV(
            profile, shader_payload->runtime_info, *shader_payload->ir,
            binding);
        shader_payload->expected_binding_end = binding;
        constexpr size_t IrMemoryEstimateMultiplier = 12;
        const size_t source_bytes =
            shader_payload->code.size() * sizeof(u32);
        constexpr size_t PoolAllocationFloor =
            Shader::Pools::InstPoolSize * sizeof(Shader::IR::Inst) +
            Shader::Pools::BlockPoolSize * sizeof(Shader::IR::Block);
        shader_payload->queued_bytes =
            source_bytes >
                    (std::numeric_limits<size_t>::max() -
                     PoolAllocationFloor) /
                        IrMemoryEstimateMultiplier
                ? std::numeric_limits<size_t>::max()
                : PoolAllocationFloor +
                      source_bytes * IrMemoryEstimateMultiplier;

        const auto foreground_user_data = info.user_data;
        const VAddr foreground_pgm_base = info.pgm_base;
        info = *shader_payload->info;
        info.user_data = foreground_user_data;
        info.pgm_base = foreground_pgm_base;
        async_pipeline_stats.shader_fast_compiled.fetch_add(
            1, std::memory_order_relaxed);
    } else {
        const auto ir_program = Shader::TranslateProgram(
            code, pools, info, runtime_info, profile,
            Shader::TranslationTier::Optimized);
        spv = Shader::Backend::SPIRV::EmitSPIRV(
            profile, runtime_info, ir_program, binding);
    }
#ifdef __ANDROID__
    if (info.stage == Shader::Stage::Fragment && info.has_discard) {
        u32 kills{};
        u32 demotes{};
        constexpr u32 OpKillOpcode = 252;
        constexpr u32 OpDemoteToHelperInvocationExtOpcode = 5380;
        for (size_t offset = 5; offset < spv.size();) {
            const u32 instruction = spv[offset];
            const u32 word_count = instruction >> 16;
            const u32 opcode = instruction & 0xffffu;
            kills += opcode == OpKillOpcode;
            demotes += opcode == OpDemoteToHelperInvocationExtOpcode;
            if (word_count == 0 || offset + word_count > spv.size()) {
                break;
            }
            offset += word_count;
        }
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_DOWNWELL_DISCARD_SPV] hash=0x%llx perm=%zu words=%zu "
                            "kills=%u demotes=%u",
                            static_cast<unsigned long long>(info.pgm_hash), perm_idx, spv.size(),
                            kills, demotes);
    }
#endif
    DumpShader(spv, info.pgm_hash, info.stage, perm_idx, "spv");

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.stage, perm_idx, "spv");
    const bool is_patched = patch && Config::patchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        const auto create_module = [this](const std::span<const u32> code) {
            const vk::ShaderModuleCreateInfo create_info{
                .codeSize = code.size_bytes(),
                .pCode = code.data(),
            };
            return instance.GetDevice().createShaderModule(create_info);
        };
        auto [module_result, created_module] = create_module(spv);
        if (module_result != vk::Result::eSuccess && use_fast_tier &&
            shader_payload && shader_payload->ir) {
            RecordShaderBinaryTierFailure(info.pgm_hash, perm_idx,
                                          ShaderBinaryTier::Fast);
            Shader::OptimizeProgram(*shader_payload->ir);
            auto optimized_binding = shader_payload->binding_start;
            spv = Shader::Backend::SPIRV::EmitSPIRV(
                profile, shader_payload->runtime_info, *shader_payload->ir,
                optimized_binding);
            ASSERT_MSG(optimized_binding ==
                           shader_payload->expected_binding_end,
                       "Optimized shader binding contract changed");
            binding = optimized_binding;
            use_fast_tier = false;
            shader_payload.reset();
            std::tie(module_result, created_module) = create_module(spv);
        }
        if (module_result != vk::Result::eSuccess) {
            RecordShaderBinaryTierFailure(
                info.pgm_hash, perm_idx,
                use_fast_tier ? ShaderBinaryTier::Fast
                              : ShaderBinaryTier::Optimized);
        }
        ASSERT_MSG(module_result == vk::Result::eSuccess,
                   "Failed to compile SPIR-V shader: {}",
                   vk::to_string(module_result));
        module = created_module;
    }

    auto cached_spv = spv;
    RegisterShaderBinary(
        std::move(cached_spv), info.pgm_hash, perm_idx,
        use_fast_tier ? ShaderBinaryTier::Fast
                      : ShaderBinaryTier::Optimized);

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (Config::collectShadersForDebug()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    if (shader_payload && !is_patched) {
        *optimization_payload = std::move(shader_payload);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding,
                                                const bool refresh_runtime_info) {
    auto runtime_info =
        refresh_runtime_info ? BuildRuntimeInfo(stage, l_stage)
                             : runtime_infos[static_cast<u32>(l_stage)];
    const auto binding_start = binding;
    std::array<u32, Shader::NUM_USER_DATA_REGS> specialization_user_data{};
    u32 specialization_user_data_mask = 0;
    u64 snapshot_key = 0;
    Shader::PFN_SrtMemoryRead srt_memory_reader{};
    if (immutable_srt_snapshot_cache_enabled) {
        srt_memory_reader = [](const VAddr address, void* const dest, const u64 size) {
            return Core::Memory::Instance()->TryCopyBackingMemory(address, dest, size);
        };
    }

    const auto build_snapshot_identity = [&](Program& program, const Shader::Info& info) {
        specialization_user_data.fill(0);
        if (!program.specialization_user_data_mask_initialized) {
            u32 mask = 0;
            const auto mark_words = [&](const u32 first, const u32 count) {
                const u32 end =
                    std::min<u32>(first + count, Shader::NUM_USER_DATA_REGS);
                for (u32 index = first; index < end; ++index) {
                    mask |= 1u << index;
                }
            };

            for (const auto& node : info.srt_info.portable_nodes) {
                if (node.parent < 0) {
                    mark_words(node.ptr_off_dw, 2);
                }
            }
            if (info.has_fetch_shader) {
                mark_words(info.fetch_shader_sgpr_base, 2);
            }
            for (const auto& resource : info.buffers) {
                if (!resource.inline_cbuf &&
                    resource.sharp_idx < Shader::NUM_USER_DATA_REGS) {
                    mark_words(resource.sharp_idx,
                               sizeof(AmdGpu::Buffer) / sizeof(u32));
                }
            }
            for (const auto& resource : info.images) {
                if (resource.sharp_idx < Shader::NUM_USER_DATA_REGS) {
                    mark_words(resource.sharp_idx,
                               sizeof(AmdGpu::Image) / sizeof(u32));
                }
            }
            for (const auto& resource : info.samplers) {
                if (!resource.is_inline_sampler &&
                    resource.sharp_idx < Shader::NUM_USER_DATA_REGS) {
                    mark_words(resource.sharp_idx,
                               sizeof(AmdGpu::Sampler) / sizeof(u32));
                }
            }
            for (const auto& resource : info.fmasks) {
                if (resource.sharp_idx < Shader::NUM_USER_DATA_REGS) {
                    mark_words(resource.sharp_idx,
                               sizeof(AmdGpu::Image) / sizeof(u32));
                }
            }
            if (info.tess_consts_ptr_base != Shader::IR::ScalarReg::Max) {
                mark_words(static_cast<u32>(info.tess_consts_ptr_base), 2);
            }
            program.specialization_user_data_mask = mask;
            program.specialization_user_data_mask_initialized = true;
        }
        specialization_user_data_mask = program.specialization_user_data_mask;
        for (u32 index = 0; index < Shader::NUM_USER_DATA_REGS; ++index) {
            if ((specialization_user_data_mask & (1u << index)) != 0) {
                specialization_user_data[index] =
                    index < params.user_data.size() ? params.user_data[index] : 0;
            }
        }

        snapshot_key = HashCombine(
            params.hash,
            static_cast<u64>(binding_start.unified) |
                (static_cast<u64>(binding_start.buffer) << 20) |
                (static_cast<u64>(binding_start.user_data) << 40));
        snapshot_key =
            HashCombine(snapshot_key, static_cast<u64>(specialization_user_data_mask));
        for (u32 index = 0; index < Shader::NUM_USER_DATA_REGS; ++index) {
            if ((specialization_user_data_mask & (1u << index)) != 0) {
                snapshot_key = HashCombine(
                    snapshot_key,
                    (static_cast<u64>(index) << 32) | specialization_user_data[index]);
            }
        }
    };

    const auto store_snapshot =
        [&](Program& program, const Shader::Info& info,
            Shader::SrtMemoryRanges&& ranges, const size_t module_index) {
            auto& primary_slot = program.SelectSpecializationSnapshotSlot(snapshot_key);
            const bool primary_had_valid_snapshot =
                primary_slot != nullptr && primary_slot->valid;
            if (primary_had_valid_snapshot &&
                primary_slot->identity_key != snapshot_key) {
                primary_slot =
                    RetainSrtVictimSnapshot(program, std::move(primary_slot));
            }
            if (!primary_slot) {
                primary_slot = std::make_unique<Program::SpecializationSnapshot>();
            }
            auto& snapshot = *primary_slot;
            snapshot.valid = false;
            snapshot.specialization_user_data = specialization_user_data;
            snapshot.specialization_user_data_mask = specialization_user_data_mask;
            snapshot.flattened_user_data = info.flattened_ud_buf;
            snapshot.runtime_info = runtime_info;
            snapshot.binding_start = binding_start;
            snapshot.srt_ranges = std::move(ranges);
            const u64 capture_sequence = CaptureCpuInvalidationSequence();
            const bool captured =
                immutable_srt_snapshot_cache_enabled
                    ? CaptureSrtBackingBytes(snapshot.srt_ranges, snapshot.srt_source_bytes)
                    : CaptureSrtSourceBytes(snapshot.srt_ranges, snapshot.srt_source_bytes);
            const bool stable_capture =
                !immutable_srt_snapshot_cache_enabled ||
                AreSrtMemoryRangesCpuStableSince(snapshot.srt_ranges, capture_sequence);
            if (!captured || !stable_capture) {
                if (primary_had_valid_snapshot) {
                    --program.specialization_snapshot_count;
                }
                return;
            }
            snapshot.module_index = module_index;
            snapshot.identity_key = snapshot_key;
            snapshot.cpu_invalidation_sequence = capture_sequence;
            snapshot.valid = true;
            if (!primary_had_valid_snapshot) {
                ++program.specialization_snapshot_count;
            }
        };

    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        std::shared_ptr<AsyncShaderPayload> shader_optimization;
        const auto module =
            CompileModule(program->info, runtime_info, params.code, 0,
                          binding, &shader_optimization);
        build_snapshot_identity(*program, program->info);
        Shader::SrtMemoryRanges srt_ranges;
        program->info.RefreshFlatBuf(&srt_ranges, srt_memory_reader);
        const auto& parsed_fetch_shader = ResolveFetchShader(*program, program->info);
        const auto tess_constants = ResolveTessConstants(program->info);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, binding_start,
                                                &parsed_fetch_shader,
                                                tess_constants ? &*tess_constants : nullptr);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(
            module, std::move(spec),
            shader_optimization ? ShaderBinaryTier::Fast
                                : ShaderBinaryTier::Optimized);
        if (shader_optimization) {
            shader_optimization->program = program.get();
            shader_optimization->perm_idx = 0;
            shader_optimization->expected_generation =
                program->modules[0].generation;
            shader_optimization->cache_generation = shader_tier_generation;
            shader_optimization->expected_module = module;
            QueueAsyncShaderOptimization(std::move(shader_optimization));
        }
        store_snapshot(*program, program->info, std::move(srt_ranges), 0);
        return std::make_tuple(&program->info, module, program->modules[0].spec.fetch_shader_data,
                               perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = params.Base();
    info.user_data = params.user_data;
    build_snapshot_identity(*program, info);
#ifdef __ANDROID__
    struct ProgramHotEntry {
        const PipelineCache* owner{};
        const Program* program{};
        u64 snapshot_key{};
        Shader::RuntimeInfo runtime_info{};
        Shader::Backend::Bindings binding_start{};
        Shader::Backend::Bindings binding_end{};
        Shader::SrtMemoryRanges srt_ranges{};
        std::vector<u32> flattened_user_data{};
        Result result{};
        u64 invalidation_sequence{};
        u64 module_generation{};
        u64 last_used{};
        bool valid{};
    };
    // Key this cache by the specialization identity instead of all user-data
    // words. Raw push-data changes on almost every Bloodborne draw, while only
    // the masked descriptor/SRT words select the flattened resources and
    // shader permutation. Comparing all words made the old cache miss 100%.
    // Bloodborne rotates through several thousand descriptor identities per
    // frame. A direct-mapped cache still discarded hot identities on hash
    // collisions and spent most misses rebuilding SRT vectors. Four exact
    // ways retain the same safety checks while sharply reducing allocator and
    // portable-SRT-walker traffic.
    static constexpr size_t ProgramHotSetCount = 8192;
    static constexpr size_t ProgramHotWayCount = 4;
    static thread_local std::array<std::array<ProgramHotEntry, ProgramHotWayCount>,
                                   ProgramHotSetCount>
        program_hot_sets{};
    static thread_local u64 program_hot_lookups{};
    static thread_local u64 program_hot_hits{};
    static thread_local u64 program_hot_clock{};
    u64 program_hot_hash =
        snapshot_key ^ static_cast<u64>(reinterpret_cast<uintptr_t>(program.get()));
    program_hot_hash ^= program_hot_hash >> 30;
    program_hot_hash *= 0xbf58476d1ce4e5b9ULL;
    program_hot_hash ^= program_hot_hash >> 27;
    program_hot_hash *= 0x94d049bb133111ebULL;
    program_hot_hash ^= program_hot_hash >> 31;
    auto& program_hot_set =
        program_hot_sets[program_hot_hash & (ProgramHotSetCount - 1)];
    ++program_hot_lookups;
    ++program_hot_clock;
    const bool program_hot_managed_safe =
        immutable_srt_snapshot_cache_enabled ||
        VideoCore::IsManagedVulkanDriverOptimization();
    ProgramHotEntry* program_hot_hit = nullptr;
    ProgramHotEntry* program_hot_invalid = nullptr;
    ProgramHotEntry* program_hot_lru = &program_hot_set.front();
    for (auto& candidate : program_hot_set) {
        if (!candidate.valid) {
            if (program_hot_invalid == nullptr) {
                program_hot_invalid = &candidate;
            }
            continue;
        }
        if (!program_hot_lru->valid ||
            candidate.last_used < program_hot_lru->last_used) {
            program_hot_lru = &candidate;
        }
        const bool identity =
            program_hot_managed_safe && candidate.owner == this &&
            candidate.program == program.get() &&
            candidate.snapshot_key == snapshot_key &&
            candidate.module_generation == program->module_generation &&
            candidate.runtime_info == runtime_info &&
            candidate.binding_start == binding_start;
        if (!identity) {
            continue;
        }
        if (AreSrtMemoryRangesCpuStableSince(candidate.srt_ranges,
                                             candidate.invalidation_sequence) &&
            (immutable_srt_snapshot_cache_enabled ||
             AreSrtMemoryRangesGpuClean(candidate.srt_ranges))) {
            program_hot_hit = &candidate;
        } else {
            candidate.valid = false;
            program_hot_invalid = &candidate;
        }
        break;
    }
    ProgramHotEntry* const program_hot_victim =
        program_hot_invalid != nullptr ? program_hot_invalid : program_hot_lru;
    if (program_hot_hit != nullptr) {
        ++program_hot_hits;
        program_hot_hit->last_used = program_hot_clock;
        program_hot_hit->invalidation_sequence = CaptureCpuInvalidationSequence();
        info.flattened_ud_buf = program_hot_hit->flattened_user_data;
        binding = program_hot_hit->binding_end;
        if ((program_hot_lookups & 65535u) == 0) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_PROGRAM_HOT_CACHE] lookups=%llu hits=%llu hitPct=%.1f "
                "sets=%zu ways=%zu",
                static_cast<unsigned long long>(program_hot_lookups),
                static_cast<unsigned long long>(program_hot_hits),
                100.0 * static_cast<double>(program_hot_hits) /
                    static_cast<double>(program_hot_lookups),
                ProgramHotSetCount, ProgramHotWayCount);
        }
        return program_hot_hit->result;
    }
    const auto remember_program_hot = [&](const Result& result,
                                            const Shader::SrtMemoryRanges& ranges,
                                            const u64 invalidation_sequence) {
        auto& program_hot = *program_hot_victim;
        program_hot.owner = this;
        program_hot.program = program.get();
        program_hot.snapshot_key = snapshot_key;
        program_hot.runtime_info = runtime_info;
        program_hot.binding_start = binding_start;
        program_hot.binding_end = binding;
        program_hot.srt_ranges = ranges;
        program_hot.flattened_user_data = info.flattened_ud_buf;
        program_hot.result = result;
        program_hot.invalidation_sequence = invalidation_sequence;
        program_hot.module_generation = program->module_generation;
        program_hot.last_used = program_hot_clock;
        program_hot.valid = true;
        if ((program_hot_lookups & 65535u) == 0) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_PROGRAM_HOT_CACHE] lookups=%llu hits=%llu hitPct=%.1f "
                "sets=%zu ways=%zu",
                static_cast<unsigned long long>(program_hot_lookups),
                static_cast<unsigned long long>(program_hot_hits),
                100.0 * static_cast<double>(program_hot_hits) /
                    static_cast<double>(program_hot_lookups),
                ProgramHotSetCount, ProgramHotWayCount);
        }
    };
#endif
    ++srt_snapshot_lookups;
    auto* snapshot_ptr = program->FindSpecializationSnapshot(snapshot_key);
    bool snapshot_from_victim = false;
    if (!snapshot_ptr) {
        snapshot_ptr = FindSrtVictimSnapshot(*program, snapshot_key);
        snapshot_from_victim = snapshot_ptr != nullptr;
        srt_snapshot_identity_misses += !snapshot_ptr;
    }
    if (snapshot_ptr) {
        auto& snapshot = *snapshot_ptr;
        const bool identity_matches =
            snapshot.module_index < program->modules.size() &&
            snapshot.binding_start == binding_start &&
            snapshot.runtime_info == runtime_info &&
            snapshot.specialization_user_data_mask == specialization_user_data_mask &&
            snapshot.specialization_user_data == specialization_user_data;
        const bool immutable_cpu_stable =
            identity_matches && immutable_srt_snapshot_cache_enabled &&
            AreSrtMemoryRangesCpuStableSince(snapshot.srt_ranges,
                                              snapshot.cpu_invalidation_sequence);
        const bool gpu_clean =
            identity_matches && !immutable_cpu_stable &&
            AreSrtMemoryRangesGpuClean(snapshot.srt_ranges);
        bool backing_validated = false;
        bool source_matches = false;
        if (identity_matches && immutable_srt_snapshot_cache_enabled) {
            if (immutable_cpu_stable) {
                backing_validated = true;
                source_matches = true;
                snapshot.cpu_invalidation_sequence = CaptureCpuInvalidationSequence();
                ++srt_snapshot_epoch_hits;
            } else {
                backing_validated =
                    CaptureSrtBackingBytes(snapshot.srt_ranges, srt_backing_validation_scratch);
                source_matches =
                    backing_validated &&
                    std::ranges::equal(srt_backing_validation_scratch,
                                       snapshot.srt_source_bytes);
            }
            srt_snapshot_backing_alias_hits += source_matches && !gpu_clean;
            srt_snapshot_backing_fallbacks += !backing_validated;
        }
        if (!backing_validated) {
            source_matches =
                gpu_clean &&
                DoSrtSourceBytesMatch(snapshot.srt_ranges, snapshot.srt_source_bytes);
        }
        srt_snapshot_gpu_misses +=
            identity_matches && !gpu_clean && !immutable_srt_snapshot_cache_enabled;
        srt_snapshot_identity_mismatches += !identity_matches;
        srt_snapshot_content_misses +=
            (gpu_clean || backing_validated) && !source_matches;
        if (source_matches) {
            ++srt_snapshot_hits;
            srt_snapshot_primary_hits += !snapshot_from_victim;
            srt_snapshot_victim_hits += snapshot_from_victim;
            info.flattened_ud_buf = snapshot.flattened_user_data;
            info.AddBindings(binding);
            const auto& cached_module = program->modules[snapshot.module_index];
#ifdef __ANDROID__
            static const bool trace_srt_cache =
                std::getenv("EXECUTOR_TRACE_SRT_CACHE") != nullptr;
            if (trace_srt_cache && (srt_snapshot_lookups & 0xffff) == 0) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_SRT_CACHE] lookups=%llu hits=%llu hitPct=%.1f "
                    "contentMiss=%llu gpuMiss=%llu backingAliasHit=%llu "
                    "backingFallback=%llu epochHit=%llu primaryHit=%llu victimLookup=%llu "
                    "victimHit=%llu keyMiss=%llu identityMismatch=%llu "
                    "primarySnapshots=%zu victimSnapshots=%zu victimInsert=%llu "
                    "victimReplace=%llu victimEvict=%llu",
                    static_cast<unsigned long long>(srt_snapshot_lookups),
                    static_cast<unsigned long long>(srt_snapshot_hits),
                    100.0 * static_cast<double>(srt_snapshot_hits) /
                        static_cast<double>(srt_snapshot_lookups),
                    static_cast<unsigned long long>(srt_snapshot_content_misses),
                    static_cast<unsigned long long>(srt_snapshot_gpu_misses),
                    static_cast<unsigned long long>(srt_snapshot_backing_alias_hits),
                    static_cast<unsigned long long>(srt_snapshot_backing_fallbacks),
                    static_cast<unsigned long long>(srt_snapshot_epoch_hits),
                    static_cast<unsigned long long>(srt_snapshot_primary_hits),
                    static_cast<unsigned long long>(srt_snapshot_victim_lookups),
                    static_cast<unsigned long long>(srt_snapshot_victim_hits),
                    static_cast<unsigned long long>(srt_snapshot_identity_misses),
                    static_cast<unsigned long long>(srt_snapshot_identity_mismatches),
                    program->specialization_snapshot_count, srt_snapshot_victims.size(),
                    static_cast<unsigned long long>(srt_snapshot_victim_insertions),
                    static_cast<unsigned long long>(srt_snapshot_victim_replacements),
                    static_cast<unsigned long long>(srt_snapshot_victim_evictions));
            }
#endif
            const Result result = std::make_tuple(
                &info, cached_module.module, cached_module.spec.fetch_shader_data,
                HashCombine(params.hash, snapshot.module_index));
#ifdef __ANDROID__
            remember_program_hot(result, snapshot.srt_ranges,
                                 CaptureCpuInvalidationSequence());
#endif
            return result;
        }
    }
#ifdef __ANDROID__
    static const bool trace_srt_cache =
        std::getenv("EXECUTOR_TRACE_SRT_CACHE") != nullptr;
    if (trace_srt_cache && (srt_snapshot_lookups & 0xffff) == 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_SRT_CACHE] lookups=%llu hits=%llu hitPct=%.1f "
            "contentMiss=%llu gpuMiss=%llu backingAliasHit=%llu "
            "backingFallback=%llu epochHit=%llu primaryHit=%llu victimLookup=%llu "
            "victimHit=%llu keyMiss=%llu identityMismatch=%llu "
            "primarySnapshots=%zu victimSnapshots=%zu victimInsert=%llu "
            "victimReplace=%llu victimEvict=%llu",
            static_cast<unsigned long long>(srt_snapshot_lookups),
            static_cast<unsigned long long>(srt_snapshot_hits),
            100.0 * static_cast<double>(srt_snapshot_hits) /
                static_cast<double>(srt_snapshot_lookups),
            static_cast<unsigned long long>(srt_snapshot_content_misses),
            static_cast<unsigned long long>(srt_snapshot_gpu_misses),
            static_cast<unsigned long long>(srt_snapshot_backing_alias_hits),
            static_cast<unsigned long long>(srt_snapshot_backing_fallbacks),
            static_cast<unsigned long long>(srt_snapshot_epoch_hits),
            static_cast<unsigned long long>(srt_snapshot_primary_hits),
            static_cast<unsigned long long>(srt_snapshot_victim_lookups),
            static_cast<unsigned long long>(srt_snapshot_victim_hits),
            static_cast<unsigned long long>(srt_snapshot_identity_misses),
            static_cast<unsigned long long>(srt_snapshot_identity_mismatches),
            program->specialization_snapshot_count, srt_snapshot_victims.size(),
            static_cast<unsigned long long>(srt_snapshot_victim_insertions),
            static_cast<unsigned long long>(srt_snapshot_victim_replacements),
            static_cast<unsigned long long>(srt_snapshot_victim_evictions));
    }
#endif

    Shader::SrtMemoryRanges srt_ranges;
    info.RefreshFlatBuf(&srt_ranges, srt_memory_reader);
    const auto& parsed_fetch_shader = ResolveFetchShader(*program, info);
    const auto tess_constants = ResolveTessConstants(info);
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding_start,
                                            &parsed_fetch_shader,
                                            tess_constants ? &*tess_constants : nullptr);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end()) {
        auto new_info = Shader::Info(stage, l_stage, params);
        std::shared_ptr<AsyncShaderPayload> shader_optimization;
        module = CompileModule(new_info, runtime_info, params.code, perm_idx,
                               binding, &shader_optimization);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(
            module, std::move(spec),
            shader_optimization ? ShaderBinaryTier::Fast
                                : ShaderBinaryTier::Optimized);
        if (shader_optimization) {
            shader_optimization->program = program.get();
            shader_optimization->perm_idx = perm_idx;
            shader_optimization->expected_generation =
                program->modules[perm_idx].generation;
            shader_optimization->cache_generation = shader_tier_generation;
            shader_optimization->expected_module = module;
            QueueAsyncShaderOptimization(std::move(shader_optimization));
        }
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    const Result result = std::make_tuple(&program->info, module,
                                          program->modules[perm_idx].spec.fetch_shader_data,
                                          perm_hash);
#ifdef __ANDROID__
    // A heavily instanced scene can rotate through more descriptor/SRT
    // identities than one program's resident snapshot table. Replacing a
    // snapshot then moving the victim into the global CLOCK cache allocated
    // several vectors per draw and cost more than the exact SRT walk itself.
    // Once this program is saturated, keep the already-hot identities and use
    // the allocation-free streaming result for cold identities.
    const bool streaming_cold_identity =
        VideoCore::IsManagedVulkanDriverOptimization() &&
        program->specialization_snapshot_count >=
            Program::NumSpecializationSnapshotSlots * 7 / 8;
    if (!streaming_cold_identity) {
        store_snapshot(*program, info, std::move(srt_ranges), perm_idx);
        if (const auto* stored_snapshot =
                program->FindSpecializationSnapshot(snapshot_key)) {
            remember_program_hot(result, stored_snapshot->srt_ranges,
                                 stored_snapshot->cpu_invalidation_sequence);
        }
    }
#else
    store_snapshot(*program, info, std::move(srt_ranges), perm_idx);
#endif
    return result;
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                              std::span<const u32> spv_code) {
    const bool restart_async = Config::managedAsyncPipelineCompilation();
    ++shader_tier_generation;
    StopAsyncPipelineWorkers();
    PublishAsyncPipelineResults();
    // Keep shader infos alive until the associated retired pipelines have
    // been destroyed. Declaration order makes the pipeline vectors retire
    // before the info vector after the cache lock is released.
    std::vector<BackgroundWarmUpInfos> retired_infos;
    std::vector<std::unique_ptr<GraphicsPipeline>> retired_graphics;
    std::vector<std::unique_ptr<ComputePipeline>> retired_compute;
    std::optional<vk::ShaderModule> new_module{};
    {
        // Shader replacement iterates and mutates every cache that the background
        // warm-up worker can grow.
        std::scoped_lock background_lock{background_warmup_mutex};
        ++background_warmup_generation;
        fallback_graphics_pipelines.clear();
        fallback_compute_pipelines.clear();
        DiscardBackgroundWarmUpPipelinesLocked(
            retired_infos, retired_graphics, retired_compute);
        last_graphics_pipeline = nullptr;
        last_graphics_state_generation = std::numeric_limits<u64>::max();
        for (const auto& [_, program] : program_cache) {
            for (auto& m : program->modules) {
                if (m.module == module) {
                    const auto& d = instance.GetDevice();
                    d.destroyShaderModule(m.module);
                    m.module = CompileSPV(spv_code, d);
                    m.tier = ShaderBinaryTier::Optimized;
                    ++m.generation;
                    ++program->module_generation;
                    new_module = m.module;
                }
            }
        }
        if (module_related_pipelines.contains(module)) {
            auto& pipeline_keys = module_related_pipelines[module];
            for (auto& key : pipeline_keys) {
                if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                    const auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                    if (auto it = graphics_pipelines.find(graphics_key);
                        it != graphics_pipelines.end()) {
                        retired_graphics.push_back(std::move(it.value()));
                        graphics_pipelines.erase(it);
                    }
                } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                    const auto& compute_key = std::get<ComputePipelineKey>(key);
                    if (auto it = compute_pipelines.find(compute_key);
                        it != compute_pipelines.end()) {
                        retired_compute.push_back(std::move(it.value()));
                        compute_pipelines.erase(it);
                    }
                }
            }
            module_related_pipelines.erase(module);
        }
    }
    if (restart_async) {
        StartAsyncPipelineWorkers();
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::Stage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!Config::dumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto full = dump_dir / filename;
    const auto file = IOFile{full, FileAccessMode::Create};
    file.WriteSpan(code);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_SHADER_DUMP] wrote path=%s bytes=%zu isOpen=%d",
                        full.string().c_str(), code.size_bytes(), file.IsOpen() ? 1 : 0);
#endif
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::Stage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
}
