// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
#include "core/libraries/gnmdriver/gnmdriver.h"
// SPDX-License-Identifier: GPL-2.0-or-later

#include <ranges>
#ifdef __ANDROID__
#include <android/log.h>
#include <atomic>
#include <cstdlib>
#include <unistd.h>
#endif

#include "common/config.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
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

namespace Vulkan {

using Shader::LogicalStage;
using Shader::Output;
using Shader::Stage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

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
    const char* const shader_identity_cache_flag =
        std::getenv("EXECUTOR_SHADER_IDENTITY_CACHE");
    shader_identity_cache_enabled =
        shader_identity_cache_flag == nullptr || shader_identity_cache_flag[0] == '\0' ||
        shader_identity_cache_flag[0] != '0';
    const char* const immutable_shader_identity_cache_flag =
        std::getenv("EXECUTOR_IMMUTABLE_SHADER_IDENTITY_CACHE");
    immutable_shader_identity_cache_enabled =
        shader_identity_cache_enabled && immutable_shader_identity_cache_flag != nullptr &&
        immutable_shader_identity_cache_flag[0] != '\0' &&
        immutable_shader_identity_cache_flag[0] != '0';
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

    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);

    WarmUp();
}

PipelineCache::~PipelineCache() {
    StopBackgroundWarmUp();
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
    if (shader_identity_generation <= sequence) {
        return false;
    }
    if (shader_identity_generation - sequence > ShaderIdentityInvalidationHistorySize) {
        return true;
    }
    const VAddr range_end = range_address + range_size;
    for (u64 current = sequence + 1; current <= shader_identity_generation; ++current) {
        const auto& invalidation =
            shader_identity_invalidation_history
                [(current - 1) % ShaderIdentityInvalidationHistorySize];
        if (invalidation.sequence != current || invalidation.size == 0 ||
            invalidation.address >
                std::numeric_limits<VAddr>::max() - invalidation.size) {
            return true;
        }
        const VAddr invalidation_end = invalidation.address + invalidation.size;
        if (invalidation.address < range_end && range_address < invalidation_end) {
            return true;
        }
    }
    return false;
}

u64 PipelineCache::CaptureCpuInvalidationSequence() {
    std::scoped_lock lock{shader_identity_cache_mutex};
    return shader_identity_generation;
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
    u64 scan_generation{};
    {
        std::scoped_lock lock{shader_identity_cache_mutex};
        ++shader_identity_lookups;
        scan_generation = shader_identity_generation;
        if (auto it = shader_identity_cache.find(address); it != shader_identity_cache.end()) {
            const ShaderIdentityMetadata& cached_metadata = it->second;
            ++shader_identity_hits;
            ++shader_identity_gpu_alias_hits;
            return {
                .user_data = fresh_user_data,
                .code = std::span{code, cached_metadata.length_bytes / sizeof(u32)},
                .hash = cached_metadata.hash,
            };
        }
    }

    const Shader::ShaderParams params = scan_metadata();
    const size_t length_bytes = params.code.size_bytes();
    const bool valid_length =
        length_bytes != 0 && length_bytes <= std::numeric_limits<u32>::max() &&
        address <= std::numeric_limits<VAddr>::max() - length_bytes;
    {
        std::scoped_lock lock{shader_identity_cache_mutex};
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
                     });
    }
    return params;
}

void PipelineCache::InvalidateShaderIdentityRange(const VAddr address, const u64 size) {
    if (!shader_identity_cache_enabled || size == 0) {
        return;
    }

    std::scoped_lock lock{shader_identity_cache_mutex};
    const u64 invalidation_sequence = ++shader_identity_generation;
    shader_identity_invalidation_history
        [(invalidation_sequence - 1) % ShaderIdentityInvalidationHistorySize] = {
            .address = address,
            .size = size,
            .sequence = invalidation_sequence,
        };
    if (address > std::numeric_limits<VAddr>::max() - size) {
        shader_identity_cpu_erases += shader_identity_cache.size();
        shader_identity_cache.clear();
        return;
    }

    const VAddr invalidate_end = address + size;
    for (auto it = shader_identity_cache.begin(); it != shader_identity_cache.end();) {
        const auto& metadata = it->second;
        if (metadata.address > std::numeric_limits<VAddr>::max() - metadata.length_bytes) {
            ++shader_identity_cpu_erases;
            it = shader_identity_cache.erase(it);
            continue;
        }
        const VAddr metadata_end = metadata.address + metadata.length_bytes;
        if (address < metadata_end && metadata.address < invalidate_end) {
            ++shader_identity_cpu_erases;
            it = shader_identity_cache.erase(it);
        } else {
            ++it;
        }
    }
}

void PipelineCache::InvalidateShaderIdentityReadbackRange(const VAddr address, const u64 size) {
    InvalidateShaderIdentityRange(address, size);
}

void PipelineCache::AdvanceShaderIdentitySubmit() {
    if (!shader_identity_cache_enabled) {
        return;
    }

    std::scoped_lock lock{shader_identity_cache_mutex};
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
            "reparses=%llu backingFallbacks=%llu racedPublications=%llu",
            static_cast<unsigned long long>(fetch_shader_lookups),
            static_cast<unsigned long long>(fetch_shader_epoch_hits),
            static_cast<unsigned long long>(fetch_shader_backing_hits),
            static_cast<unsigned long long>(fetch_shader_reparses),
            static_cast<unsigned long long>(fetch_shader_backing_fallbacks),
            static_cast<unsigned long long>(fetch_shader_raced_publications));
    }
#endif
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
        {
            std::scoped_lock lock{shader_identity_cache_mutex};
            if (snapshot.address == address &&
                !WasCpuRangeInvalidatedSinceLocked(address, size_bytes,
                                                   snapshot.invalidation_sequence)) {
                ++fetch_shader_epoch_hits;
                return snapshot.parsed;
            }
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
        {
            std::scoped_lock lock{shader_identity_cache_mutex};
            stable = bytes_match &&
                     !WasCpuRangeInvalidatedSinceLocked(address, size_bytes,
                                                       validation_sequence);
            publish_sequence = shader_identity_generation;
        }
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
        std::scoped_lock lock{shader_identity_cache_mutex};
        stable_publication = !WasCpuRangeInvalidatedSinceLocked(
            address, parsed_code.size() * sizeof(u32), parse_sequence);
        publish_sequence = shader_identity_generation;
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
    const u64 state_generation = liverpool->GraphicsStateGeneration();
    if (!background_warmup_active.load(std::memory_order_acquire) &&
        last_graphics_pipeline != nullptr &&
        last_graphics_state_generation == state_generation &&
        last_graphics_primitive_override == host_primitive_override) {
        return last_graphics_pipeline;
    }

    std::unique_lock<std::mutex> background_lock;
    if (background_warmup_active.load(std::memory_order_acquire)) {
        background_warmup_foreground_waiters.fetch_add(1, std::memory_order_acq_rel);
        if (background_warmup_active.load(std::memory_order_acquire)) {
            background_lock = std::unique_lock{background_warmup_mutex};
        }
        background_warmup_foreground_waiters.fetch_sub(1, std::memory_order_acq_rel);
    }
    graphics_primitive_override = host_primitive_override;
    const bool refresh_ok = RefreshGraphicsKey(host_primitive_override);
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
        return nullptr;
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
    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);

        GraphicsPipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, false);

        RegisterPipelineData(graphics_key, pipeline_hash, sdata);
        ++num_new_pipelines;

        if (Config::collectShadersForDebug()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (infos[stage]) {
                    auto& m = modules[stage];
                    module_related_pipelines[m].emplace_back(graphics_key);
                }
            }
        }
        fetch_shader.reset();
    }
    const auto* pipeline = it->second.get();
    last_graphics_pipeline = pipeline && pipeline->IsValid() ? pipeline : nullptr;
    last_graphics_state_generation = state_generation;
    last_graphics_primitive_override = host_primitive_override;
    return last_graphics_pipeline;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    std::unique_lock<std::mutex> background_lock;
    if (background_warmup_active.load(std::memory_order_acquire)) {
        background_warmup_foreground_waiters.fetch_add(1, std::memory_order_acq_rel);
        if (background_warmup_active.load(std::memory_order_acquire)) {
            background_lock = std::unique_lock{background_warmup_mutex};
        }
        background_warmup_foreground_waiters.fetch_sub(1, std::memory_order_acq_rel);
    }
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *infos[0],
                                                       modules[0], sdata, false);
        RegisterPipelineData(compute_key, sdata);
        ++num_new_pipelines;

        if (Config::collectShadersForDebug()) {
            auto& m = modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
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

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = liverpool->regs;
    auto& key = graphics_key;
    fetch_shader = std::nullopt;

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
            GetProgram(stage_in, stage_out, params, binding);
        if (fetch_shader_) {
            fetch_shader = fetch_shader_;
        }
        return true;
    };

    infos.fill(nullptr);
    modules.fill(nullptr);
    runtime_infos.fill({});
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
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.stage, perm_idx, "bin");

    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
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
        module = CompileSPV(spv, instance.GetDevice());
    }

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (Config::collectShadersForDebug()) {
        DebugState.CollectShader(name, info.l_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::Result PipelineCache::GetProgram(Stage stage, LogicalStage l_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(stage, l_stage);
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

    const auto build_snapshot_identity = [&](const Shader::Info& info) {
        specialization_user_data.fill(0);
        specialization_user_data_mask = 0;
        const auto mark_words = [&](const u32 first, const u32 count) {
            const u32 end = std::min<u32>(first + count, Shader::NUM_USER_DATA_REGS);
            for (u32 index = first; index < end; ++index) {
                specialization_user_data_mask |= 1u << index;
                specialization_user_data[index] =
                    index < params.user_data.size() ? params.user_data[index] : 0;
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
            if (!resource.inline_cbuf && resource.sharp_idx < Shader::NUM_USER_DATA_REGS) {
                mark_words(resource.sharp_idx, sizeof(AmdGpu::Buffer) / sizeof(u32));
            }
        }
        for (const auto& resource : info.images) {
            if (resource.sharp_idx < Shader::NUM_USER_DATA_REGS) {
                mark_words(resource.sharp_idx, sizeof(AmdGpu::Image) / sizeof(u32));
            }
        }
        for (const auto& resource : info.samplers) {
            if (!resource.is_inline_sampler &&
                resource.sharp_idx < Shader::NUM_USER_DATA_REGS) {
                mark_words(resource.sharp_idx, sizeof(AmdGpu::Sampler) / sizeof(u32));
            }
        }
        for (const auto& resource : info.fmasks) {
            if (resource.sharp_idx < Shader::NUM_USER_DATA_REGS) {
                mark_words(resource.sharp_idx, sizeof(AmdGpu::Image) / sizeof(u32));
            }
        }
        if (info.tess_consts_ptr_base != Shader::IR::ScalarReg::Max) {
            mark_words(static_cast<u32>(info.tess_consts_ptr_base), 2);
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
            const bool captured =
                immutable_srt_snapshot_cache_enabled
                    ? CaptureSrtBackingBytes(snapshot.srt_ranges, snapshot.srt_source_bytes)
                    : CaptureSrtSourceBytes(snapshot.srt_ranges, snapshot.srt_source_bytes);
            if (!captured) {
                if (primary_had_valid_snapshot) {
                    --program.specialization_snapshot_count;
                }
                return;
            }
            snapshot.module_index = module_index;
            snapshot.identity_key = snapshot_key;
            snapshot.valid = true;
            if (!primary_had_valid_snapshot) {
                ++program.specialization_snapshot_count;
            }
        };

    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>(stage, l_stage, params);
        auto& program = it_pgm.value();
        const auto module = CompileModule(program->info, runtime_info, params.code, 0, binding);
        build_snapshot_identity(program->info);
        Shader::SrtMemoryRanges srt_ranges;
        program->info.RefreshFlatBuf(&srt_ranges, srt_memory_reader);
        const auto& parsed_fetch_shader = ResolveFetchShader(*program, program->info);
        auto spec = Shader::StageSpecialization(program->info, runtime_info, profile, binding_start,
                                                &parsed_fetch_shader);
        const auto perm_hash = HashCombine(params.hash, 0);

        RegisterShaderMeta(program->info, spec.fetch_shader_data, spec, perm_hash, 0);
        program->AddPermut(module, std::move(spec));
        store_snapshot(*program, program->info, std::move(srt_ranges), 0);
        return std::make_tuple(&program->info, module, program->modules[0].spec.fetch_shader_data,
                               perm_hash);
    }

    auto& program = it_pgm.value();
    auto& info = program->info;
    info.pgm_base = params.Base();
    info.user_data = params.user_data;
    build_snapshot_identity(info);
    ++srt_snapshot_lookups;
    const auto* snapshot_ptr = program->FindSpecializationSnapshot(snapshot_key);
    bool snapshot_from_victim = false;
    if (!snapshot_ptr) {
        snapshot_ptr = FindSrtVictimSnapshot(*program, snapshot_key);
        snapshot_from_victim = snapshot_ptr != nullptr;
        srt_snapshot_identity_misses += !snapshot_ptr;
    }
    if (snapshot_ptr) {
        const auto& snapshot = *snapshot_ptr;
        const bool identity_matches =
            snapshot.module_index < program->modules.size() &&
            snapshot.binding_start == binding_start &&
            snapshot.runtime_info == runtime_info &&
            snapshot.specialization_user_data_mask == specialization_user_data_mask &&
            snapshot.specialization_user_data == specialization_user_data;
        const bool gpu_clean =
            identity_matches && AreSrtMemoryRangesGpuClean(snapshot.srt_ranges);
        bool backing_validated = false;
        bool source_matches = false;
        if (identity_matches && immutable_srt_snapshot_cache_enabled) {
            backing_validated =
                CaptureSrtBackingBytes(snapshot.srt_ranges, srt_backing_validation_scratch);
            source_matches =
                backing_validated &&
                std::ranges::equal(srt_backing_validation_scratch,
                                   snapshot.srt_source_bytes);
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
                    "backingFallback=%llu primaryHit=%llu victimLookup=%llu "
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
            return std::make_tuple(
                &info, cached_module.module, cached_module.spec.fetch_shader_data,
                HashCombine(params.hash, snapshot.module_index));
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
            "backingFallback=%llu primaryHit=%llu victimLookup=%llu "
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
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding_start,
                                            &parsed_fetch_shader);

    size_t perm_idx = program->modules.size();
    u64 perm_hash = HashCombine(params.hash, perm_idx);

    vk::ShaderModule module{};

    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end()) {
        auto new_info = Shader::Info(stage, l_stage, params);
        module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);

        RegisterShaderMeta(info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
        program->AddPermut(module, std::move(spec));
    } else {
        info.AddBindings(binding);
        module = it->module;
        perm_idx = std::distance(program->modules.begin(), it);
        perm_hash = HashCombine(params.hash, perm_idx);
    }
    store_snapshot(*program, info, std::move(srt_ranges), perm_idx);
    return std::make_tuple(&program->info, module,
                           program->modules[perm_idx].spec.fetch_shader_data, perm_hash);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    last_graphics_pipeline = nullptr;
    last_graphics_state_generation = std::numeric_limits<u64>::max();
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
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
