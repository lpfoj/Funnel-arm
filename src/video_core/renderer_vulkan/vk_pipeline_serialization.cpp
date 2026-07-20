// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/config.h"
#include "common/serdes.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include <chrono>

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/resource.h>
#endif

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
// Bump whenever emitted SPIR-V semantics or RuntimeInfo's raw serialized layout changes. Version 17
// restores the upstream/reference packed SSBO-offset extraction with OpBitFieldUExtract; V18 also
// restores raw sign-bit VOP3 integer source modifiers. Older modules must not survive either
// semantic change.
static constexpr u32 ShaderBinaryVersion = 22u;
static constexpr u32 ShaderMetaVersion = 1u;
static constexpr u32 PipelineKeyVersion = 1u;
} // namespace Serialization

namespace Vulkan {

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    info.Deserialize(ar);

    fetch_shader_data = spec.fetch_shader_data;
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    compute_key.Deserialize(ar);
    if (compute_pipelines.contains(compute_key)) {
        return true;
    }

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    graphics_key.Deserialize(ar);
    if (graphics_pipelines.contains(graphics_key)) {
        return true;
    }

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        if (meta_blob.empty()) {
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    size_t perm_idx{};
    if (!LoadShaderMeta(ar, program->info, fetch_shader, spec, perm_idx)) {
        return false;
    }

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", program->info.pgm_hash, perm_idx),
                                       spv);
    if (spv.empty()) {
        return false;
    }

    // Permutation hash depends on shader variation index. To prevent collisions, we need insert it
    // at the exact position rather than append

    vk::ShaderModule module{};

    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        module = CompileSPV(spv, instance.GetDevice());
        it_pgm.value() = std::move(program);
    } else {
        const auto& it = std::ranges::find(it_pgm.value()->modules, spec, &Program::Module::spec);
        if (it != it_pgm.value()->modules.end()) {
            // If the permutation is already preloaded, make sure it has the same permutation index
            const auto idx = std::distance(it_pgm.value()->modules.begin(), it);
            ASSERT_MSG(perm_idx == idx, "Permutation {} is already inserted at {}! ({}_{:x})",
                       perm_idx, idx, program->info.stage, program->info.pgm_hash);
            module = it->module;
        } else {
            module = CompileSPV(spv, instance.GetDevice());
        }
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);

    infos[stage] = &it_pgm.value()->info;
    modules[stage] = module;

    return true;
}

void PipelineCache::WarmUp() {
    // A serialized module bypasses GetShaderPatch entirely. When replacement shaders are enabled,
    // compile the live program path so the requested patch is guaranteed to reach Vulkan.
    if (!Config::isPipelineCacheEnabled() || Config::patchShaders()) {
        return;
    }

    Storage::DataBase::Instance().Open();

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        Storage::DataBase::Instance().FinishPreload();

        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Ignoring the cache",
                    profile_data.size(), sizeof(Shader::Profile));
        Storage::DataBase::Instance().FinishPreload();
        return;
    }

    // Comparing the raw struct also compares indeterminate padding bytes, which invalidated a
    // perfectly compatible cache on otherwise identical Android launches. Deserialize into a
    // value and compare members so the first level run can genuinely reuse its shader cache.
    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Ignoring the cache");
        Storage::DataBase::Instance().FinishPreload();
        return;
    }

    u32 num_pipelines{};
    u32 num_total_pipelines{};
#ifdef __ANDROID__
    // Desktop can afford eagerly creating every cached pipeline. On Android, a mature game cache
    // can contain hundreds of Adreno pipelines; compiling all of them blocked launch for 17 seconds
    // even though only the earliest subset is needed before the first frame. Cache files are visited
    // oldest-first (first game use); later variants are recreated by the normal on-demand compiler
    // only if gameplay actually reaches them. Bound both count and wall time so cache growth never
    // makes launch slower over time.
    constexpr u32 kWarmPipelineLimit = 24;
    constexpr auto kWarmTimeBudget = std::chrono::milliseconds{1500};
    const auto warm_start = std::chrono::steady_clock::now();
    bool budget_reached = false;
    std::vector<std::vector<u8>> deferred_pipeline_blobs;
#endif

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) -> bool {
            ++num_total_pipelines;
#ifdef __ANDROID__
            if (num_pipelines >= kWarmPipelineLimit ||
                std::chrono::steady_clock::now() - warm_start >= kWarmTimeBudget) {
                budget_reached = true;
                deferred_pipeline_blobs.push_back(std::move(data));
                return true;
            }
#endif
            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                return true;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
            }
            return true;
        });

    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
#ifdef __ANDROID__
    const auto warm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - warm_start)
                             .count();
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_PIPELINE_WARMUP_BOUNDED] loaded=%u visited=%u limit=%u "
                        "elapsedMs=%lld budgetReached=%d deferred=%zu",
                        num_pipelines, num_total_pipelines, kWarmPipelineLimit,
                        static_cast<long long>(warm_ms), budget_reached ? 1 : 0,
                        deferred_pipeline_blobs.size());
#endif
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
#ifdef __ANDROID__
    if (!deferred_pipeline_blobs.empty()) {
        StartBackgroundWarmUp(std::move(deferred_pipeline_blobs));
    }
#endif
}

void PipelineCache::StartBackgroundWarmUp(
    std::vector<std::vector<u8>>&& pipeline_blobs) {
#ifdef __ANDROID__
    StopBackgroundWarmUp();
    background_warmup_active.store(false, std::memory_order_release);
    background_warmup_worker = std::jthread{
        [this, blobs = std::move(pipeline_blobs)](const std::stop_token stop_token) mutable {
            // Pipeline warmup must never compete with the title's first-frame work. Linux nice is
            // per-thread here and inherited only by children, so foreground renderer priority is
            // unchanged while this cache worker uses idle CPU time.
            (void)::setpriority(PRIO_PROCESS, 0, 8);
            // Adreno serializes substantial portions of vkCreate*Pipelines inside the driver, so
            // compiling hundreds of cached pipelines immediately can delay first submit even from
            // a niced host thread. Give boot/first-frame work an uncontended window; foreground
            // lookups remain fully on-demand because the background-active gate stays false.
            constexpr auto kFirstFrameGrace = std::chrono::seconds{60};
            const auto grace_deadline = std::chrono::steady_clock::now() + kFirstFrameGrace;
            while (!stop_token.stop_requested() &&
                   std::chrono::steady_clock::now() < grace_deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds{10});
            }
            if (stop_token.stop_requested()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_PIPELINE_WARMUP_BACKGROUND] loaded=0 attempted=0 total=%zu "
                    "elapsedMs=0 stopped=1 graceMs=%lld",
                    blobs.size(),
                    static_cast<long long>(
                        std::chrono::duration_cast<std::chrono::milliseconds>(kFirstFrameGrace)
                            .count()));
                return;
            }
            background_warmup_active.store(true, std::memory_order_release);
            const auto started = std::chrono::steady_clock::now();
            u32 attempted = 0;
            u32 loaded = 0;
            for (std::vector<u8>& data : blobs) {
                if (stop_token.stop_requested()) {
                    break;
                }
                while (background_warmup_foreground_waiters.load(
                           std::memory_order_acquire) != 0 &&
                       !stop_token.stop_requested()) {
                    std::this_thread::yield();
                }
                if (stop_token.stop_requested()) {
                    break;
                }
                std::unique_lock background_lock{background_warmup_mutex};
                ++attempted;
                Serialization::Archive ar{std::move(data)};
                Serialization::Reader pldata{ar};
                u32 version{};
                pldata.Read(version);
                if (version != Serialization::PipelineKeyVersion) {
                    continue;
                }
                u32 is_compute{};
                pldata.Read(is_compute);
                const bool result = is_compute ? LoadComputePipeline(ar)
                                               : LoadGraphicsPipeline(ar);
                loaded += result ? 1u : 0u;
            }
            background_warmup_active.store(false, std::memory_order_release);
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - started)
                                        .count();
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_PIPELINE_WARMUP_BACKGROUND] loaded=%u attempted=%u total=%zu "
                "elapsedMs=%lld stopped=%d graceMs=%lld",
                loaded, attempted, blobs.size(), static_cast<long long>(elapsed_ms),
                stop_token.stop_requested() ? 1 : 0,
                static_cast<long long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(kFirstFrameGrace)
                        .count()));
        }};
#else
    (void)pipeline_blobs;
#endif
}

void PipelineCache::StopBackgroundWarmUp() {
    if (!background_warmup_worker.joinable()) {
        return;
    }
    background_warmup_worker.request_stop();
    background_warmup_worker.join();
    background_warmup_active.store(false, std::memory_order_release);
}

void PipelineCache::Sync() {
    StopBackgroundWarmUp();
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    // Field-by-field (the struct now holds std::vectors, so a raw sizeof() blob is unsafe). walker_func
    // itself is a host pointer -- not serialized; it is re-registered from the walker code on read.
    srt.Write(walker_func_size);
    srt.Write(flattened_bufsize_dw);
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
    srt.Write(portable_nodes);   // non-x86 SRT walker recipe
    srt.Write(portable_copies);
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    srt.Read(walker_func_size);
    srt.Read(flattened_bufsize_dw);
    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }
    portable_nodes.clear();
    portable_copies.clear();
    srt.Read(portable_nodes);
    srt.Read(portable_copies);

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (fetch_shader_data) {
        spec.Write(sizeof(*fetch_shader_data));
        fetch_shader_data->Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
