// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace VideoCore {
class BufferCache;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

enum class ShaderBinaryTier : u8 {
    Fast,
    Optimized,
};

struct AsyncPipelineCompileStats {
    bool managed_enabled{};
    bool fail_fast_supported{};
    u32 worker_count{};
    u32 queue_capacity{};
    u32 queue_depth{};
    u32 max_queue_depth{};
    u32 shader_queue_capacity{};
    u32 shader_queue_depth{};
    u32 shader_max_queue_depth{};
    u64 shader_queue_byte_capacity{};
    u64 shader_queue_bytes{};
    u64 shader_max_queue_bytes{};
    u64 graphics_misses{};
    u64 compute_misses{};
    u64 fail_fast_hits{};
    u64 fail_fast_compile_required{};
    u64 pending_hits{};
    u64 fast_queued{};
    u64 optimized_queued{};
    u64 queue_drops{};
    u64 fast_completed{};
    u64 optimized_completed{};
    u64 build_failures{};
    u64 published{};
    u64 optimized_replacements{};
    u64 skipped_uses{};
    u64 foreground_fallbacks{};
    u64 shader_fast_compiled{};
    u64 shader_optimized_queued{};
    u64 shader_optimized_completed{};
    u64 shader_optimized_published{};
    u64 shader_queue_drops{};
    u64 shader_negative_hits{};
    u64 shader_build_failures{};
    u64 shader_replacements{};
    u64 shader_total_build_us{};
    u64 shader_max_build_us{};
    u64 total_build_us{};
    u64 max_build_us{};
};

[[nodiscard]] AsyncPipelineCompileStats GetAsyncPipelineCompileStats() noexcept;
void ResetAsyncPipelineCompileStats() noexcept;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
        ShaderBinaryTier tier{ShaderBinaryTier::Optimized};
        u64 generation{1};
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    u64 module_generation{1};
    u32 specialization_user_data_mask{};
    bool specialization_user_data_mask_initialized{};
    struct SpecializationSnapshot {
        std::array<u32, Shader::NUM_USER_DATA_REGS> specialization_user_data{};
        u32 specialization_user_data_mask{};
        std::vector<u32> flattened_user_data{};
        Shader::RuntimeInfo runtime_info{};
        Shader::Backend::Bindings binding_start{};
        Shader::SrtMemoryRanges srt_ranges{};
        std::vector<u8> srt_source_bytes{};
        size_t module_index{};
        u64 identity_key{};
        u64 cpu_invalidation_sequence{};
        bool valid{};
    };
    static constexpr size_t NumSpecializationSnapshotWays = 4;
    static constexpr size_t NumSpecializationSnapshotSets = 512;
    static_assert((NumSpecializationSnapshotSets & (NumSpecializationSnapshotSets - 1)) == 0);
    static constexpr size_t NumSpecializationSnapshotSlots =
        NumSpecializationSnapshotWays * NumSpecializationSnapshotSets;
    std::array<std::unique_ptr<SpecializationSnapshot>, NumSpecializationSnapshotSlots>
        specialization_snapshots{};
    std::array<u8, NumSpecializationSnapshotSets> specialization_snapshot_next_way{};
    size_t specialization_snapshot_count{};

    [[nodiscard]] static size_t SpecializationSnapshotSet(const u64 identity_key) {
        u64 mixed = identity_key;
        mixed ^= mixed >> 30;
        mixed *= 0xbf58476d1ce4e5b9ULL;
        mixed ^= mixed >> 27;
        mixed *= 0x94d049bb133111ebULL;
        mixed ^= mixed >> 31;
        return static_cast<size_t>(mixed) & (NumSpecializationSnapshotSets - 1);
    }

    [[nodiscard]] SpecializationSnapshot* FindSpecializationSnapshot(u64 identity_key) {
        const size_t set = SpecializationSnapshotSet(identity_key);
        const size_t base = set * NumSpecializationSnapshotWays;
        for (size_t way = 0; way < NumSpecializationSnapshotWays; ++way) {
            auto& slot = specialization_snapshots[base + way];
            if (slot && slot->valid && slot->identity_key == identity_key) {
                return slot.get();
            }
        }
        return nullptr;
    }

    std::unique_ptr<SpecializationSnapshot>& SelectSpecializationSnapshotSlot(u64 identity_key) {
        const size_t set = SpecializationSnapshotSet(identity_key);
        const size_t base = set * NumSpecializationSnapshotWays;
        for (size_t way = 0; way < NumSpecializationSnapshotWays; ++way) {
            auto& slot = specialization_snapshots[base + way];
            if (slot && slot->identity_key == identity_key) {
                return slot;
            }
        }
        for (size_t way = 0; way < NumSpecializationSnapshotWays; ++way) {
            auto& slot = specialization_snapshots[base + way];
            if (!slot) {
                return slot;
            }
            if (!slot->valid) {
                return slot;
            }
        }
        const size_t way =
            specialization_snapshot_next_way[set]++ % NumSpecializationSnapshotWays;
        return specialization_snapshots[base + way];
    }
    struct FetchShaderSnapshot {
        std::optional<Shader::Gcn::FetchShaderData> parsed{};
        std::vector<u32> code{};
        VAddr address{};
        u64 invalidation_sequence{};
        bool initialized{};
    } fetch_shader_snapshot{};

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                   ShaderBinaryTier tier = ShaderBinaryTier::Optimized) {
        modules.emplace_back(module, std::move(spec), tier, 1);
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx,
                      ShaderBinaryTier tier = ShaderBinaryTier::Optimized) {
        modules.resize(std::max(modules.size(), perm_idx + 1));
        modules[perm_idx] = {module, std::move(spec), tier, 1};
    }
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool,
                           VideoCore::BufferCache& buffer_cache);
    ~PipelineCache();

    void WarmUp();
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline*
    GetGraphicsPipeline(std::optional<AmdGpu::PrimitiveType> host_primitive_override = std::nullopt);

    const ComputePipeline* GetComputePipeline();

    void InvalidateSrtSnapshots();
    void InvalidateShaderIdentityRange(VAddr address, u64 size);
    void InvalidateShaderIdentityReadbackRange(VAddr address, u64 size);
    void AdvanceShaderIdentitySubmit();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::optional<Shader::Gcn::FetchShaderData>, u64>;
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding,
                      bool refresh_runtime_info = true);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

    [[nodiscard]] AsyncPipelineCompileStats GetAsyncPipelineStats() const noexcept;

private:
    struct AsyncShaderPayload;

    using BackgroundWarmUpInfos =
        std::array<std::unique_ptr<Shader::Info>, MaxShaderStages>;

    void StartBackgroundWarmUp(std::vector<std::vector<u8>>&& pipeline_blobs);
    void StopBackgroundWarmUp();
    bool LoadIsolatedWarmUpPipeline(Serialization::Archive& ar, bool is_compute,
                                    u64 warmup_generation);
    std::unique_lock<std::mutex> LockBackgroundWarmUpForForeground();
    void DiscardBackgroundWarmUpPipelinesLocked(
        std::vector<BackgroundWarmUpInfos>& retired_infos,
        std::vector<std::unique_ptr<GraphicsPipeline>>& retired_graphics,
        std::vector<std::unique_ptr<ComputePipeline>>& retired_compute);
    void SaveHostPipelineCache();
    void MaybeCheckpointPipelineCaches();
    [[nodiscard]] std::vector<u8> LoadHostPipelineCache() const;
    [[nodiscard]] std::string HostPipelineCacheName() const;
    [[nodiscard]] static bool WriteCacheFileAtomically(const std::filesystem::path& path,
                                                       const std::vector<u8>& data);
    void LoadPipelineNegativeCache();
    void SavePipelineNegativeCache();
    [[nodiscard]] std::string PipelineNegativeCacheName() const;
    [[nodiscard]] bool IsKnownInvalid(const ComputePipelineKey& key) const;
    [[nodiscard]] bool IsKnownInvalid(const GraphicsPipelineKey& key) const;
    void RecordInvalid(const ComputePipelineKey& key);
    void RecordInvalid(const GraphicsPipelineKey& key);
    void RecordValid(const ComputePipelineKey& key);
    void RecordValid(const GraphicsPipelineKey& key);
    void MarkHostPipelineCacheDirty();
    void ResetPipelineBuildState();
    void EnsureAsyncPipelineMode();
    void StartAsyncPipelineWorkers();
    void StopAsyncPipelineWorkers();
    void PublishAsyncPipelineResults();
    bool QueueAsyncShaderOptimization(
        std::shared_ptr<AsyncShaderPayload> payload);
    void AsyncShaderWorker(std::stop_token stop_token);
    void PublishAsyncShaderResults();
    void DestroyRetiredShaderModules();
    bool QueueAsyncGraphicsPipeline(
        const GraphicsPipelineKey& key,
        std::span<const Shader::Info* const, MaxShaderStages> stage_infos,
        std::span<const Shader::RuntimeInfo, MaxShaderStages> stage_runtime_infos,
        std::optional<Shader::Gcn::FetchShaderData> stage_fetch_shader,
        std::span<const vk::ShaderModule, MaxShaderStages> stage_modules,
        GraphicsPipeline::SerializationSupport sdata, bool optimized);
    bool QueueAsyncComputePipeline(
        const ComputePipelineKey& key, const Shader::Info& info,
        vk::ShaderModule module, ComputePipeline::SerializationSupport sdata,
        bool optimized);
    void AsyncPipelineWorker(std::stop_token stop_token, u32 worker_index);
    bool RefreshGraphicsKey(std::optional<AmdGpu::PrimitiveType> host_primitive_override);
    bool RefreshGraphicsStages(bool refresh_runtime_info = true);
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding,
                                   std::shared_ptr<AsyncShaderPayload>*
                                       optimization_payload = nullptr);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage);
    bool CaptureSrtSourceBytes(std::span<const Shader::SrtMemoryRange> ranges,
                               std::vector<u8>& bytes) const;
    bool CaptureSrtBackingBytes(std::span<const Shader::SrtMemoryRange> ranges,
                                std::vector<u8>& bytes) const;
    bool DoSrtSourceBytesMatch(std::span<const Shader::SrtMemoryRange> ranges,
                               std::span<const u8> source_bytes) const;
    bool AreSrtMemoryRangesGpuClean(std::span<const Shader::SrtMemoryRange> ranges) const;
    Program::SpecializationSnapshot* FindSrtVictimSnapshot(const Program& program,
                                                           u64 identity_key);
    std::unique_ptr<Program::SpecializationSnapshot> RetainSrtVictimSnapshot(
        const Program& program, std::unique_ptr<Program::SpecializationSnapshot> snapshot);
    const std::optional<Shader::Gcn::FetchShaderData>& ResolveFetchShader(
        Program& program, const Shader::Info& info);
    std::optional<Shader::TessellationDataConstantBuffer> ResolveTessConstants(
        const Shader::Info& info);
    Shader::ShaderParams GetShaderParams(const u32* user_data, const u32* code);
    [[nodiscard]] bool WasCpuRangeInvalidatedSinceLocked(VAddr address, u64 size,
                                                         u64 sequence) const;
    [[nodiscard]] bool AreSrtMemoryRangesCpuStableSince(
        std::span<const Shader::SrtMemoryRange> ranges, u64 sequence);
    [[nodiscard]] u64 CaptureCpuInvalidationSequence();

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    VideoCore::BufferCache* buffer_cache;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    std::vector<u8> async_pipeline_cache_seed{};
    u64 host_pipeline_cache_generation{};
    u64 host_pipeline_cache_saved_generation{};
    std::chrono::steady_clock::time_point next_pipeline_cache_checkpoint{};
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<ComputePipelineKey, BackgroundWarmUpInfos>
        background_warmup_compute_infos;
    tsl::robin_map<GraphicsPipelineKey, BackgroundWarmUpInfos>
        background_warmup_graphics_infos;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    const GraphicsPipeline* last_graphics_pipeline{};
    u64 last_graphics_state_generation{std::numeric_limits<u64>::max()};
    u64 last_graphics_other_generation{std::numeric_limits<u64>::max()};
    u64 last_graphics_shader_program_generation{std::numeric_limits<u64>::max()};
    u64 last_graphics_shader_user_data_generation{std::numeric_limits<u64>::max()};
    u64 last_graphics_structural_signature{};
    std::optional<AmdGpu::PrimitiveType> graphics_primitive_override{};
    std::optional<AmdGpu::PrimitiveType> last_graphics_primitive_override{};
    bool last_graphics_lookup_resolved{};
    u32 num_new_pipelines{};
#ifdef __ANDROID__
    struct GraphicsPipelineTelemetry {
        u64 lookups{};
        u64 refreshes{};
        u64 generation_misses{};
        u64 refresh_same_final_key{};
        u64 same_stage_hashes{};
        u64 actual_pipeline_changed{};
        u64 refresh_failures{};
        u64 irrelevant_state_hits{};
        u64 user_data_only_refreshes{};
        u64 last_logged_generation_miss{};
        u64 last_logged_dirty_program{};
        u64 last_logged_dirty_user_data{};
        u64 last_logged_dirty_other{};
    } graphics_pipeline_telemetry{};
#endif

    u64 srt_snapshot_lookups{};
    u64 srt_snapshot_hits{};
    u64 srt_snapshot_content_misses{};
    u64 srt_snapshot_gpu_misses{};
    u64 srt_snapshot_backing_alias_hits{};
    u64 srt_snapshot_backing_fallbacks{};
    u64 srt_snapshot_epoch_hits{};
    u64 srt_snapshot_primary_hits{};
    u64 srt_snapshot_victim_lookups{};
    u64 srt_snapshot_victim_hits{};
    u64 srt_snapshot_identity_misses{};
    u64 srt_snapshot_identity_mismatches{};
    u64 srt_snapshot_victim_insertions{};
    u64 srt_snapshot_victim_replacements{};
    u64 srt_snapshot_victim_evictions{};
    bool immutable_srt_snapshot_cache_enabled{};
    mutable std::vector<u8> srt_backing_validation_scratch{};
    struct SrtVictimKey {
        const Program* program{};
        u64 identity_key{};

        friend bool operator==(const SrtVictimKey&, const SrtVictimKey&) = default;
    };
    struct SrtVictimKeyHash {
        size_t operator()(const SrtVictimKey& key) const noexcept {
            size_t seed = std::hash<const Program*>{}(key.program);
            seed ^= std::hash<u64>{}(key.identity_key) + 0x9e3779b97f4a7c15ULL +
                    (seed << 6) + (seed >> 2);
            return seed;
        }
    };
    struct SrtVictimEntry {
        std::unique_ptr<Program::SpecializationSnapshot> snapshot{};
        bool referenced{};
    };
    static constexpr size_t MaxSrtVictimSnapshots = 8192;
    tsl::robin_map<SrtVictimKey, SrtVictimEntry, SrtVictimKeyHash> srt_snapshot_victims{};
    std::deque<SrtVictimKey> srt_snapshot_victim_clock{};
    mutable std::vector<u32> fetch_shader_backing_validation_scratch{};
    u64 fetch_shader_lookups{};
    u64 fetch_shader_epoch_hits{};
    u64 fetch_shader_backing_hits{};
    u64 fetch_shader_reparses{};
    u64 fetch_shader_backing_fallbacks{};
    u64 fetch_shader_raced_publications{};
    struct TessConstantsSnapshot {
        Shader::TessellationDataConstantBuffer data{};
        u64 invalidation_sequence{};
    };
    static constexpr size_t MaxTessConstantsSnapshots = 32768;
    tsl::robin_map<VAddr, TessConstantsSnapshot> tess_constants_snapshots{};
    u64 tess_constants_lookups{};
    u64 tess_constants_hits{};
    u64 tess_constants_reads{};
    u64 tess_constants_backing_copies{};

    struct ShaderIdentityMetadata {
        VAddr address{};
        u32 length_bytes{};
        u64 hash{};
        u64 invalidation_sequence{};
    };
    struct ShaderIdentityInvalidation {
        std::atomic<VAddr> address{};
        std::atomic<u64> size{};
        std::atomic<u64> sequence{};
    };
    static constexpr size_t MaxShaderIdentityEntriesPerSubmit = 4096;
    static constexpr size_t ShaderIdentityInvalidationHistorySize = 4096;
    bool shader_identity_cache_enabled{};
    bool immutable_shader_identity_cache_enabled{};
    tsl::robin_map<VAddr, ShaderIdentityMetadata> shader_identity_cache;
    std::array<ShaderIdentityInvalidation, ShaderIdentityInvalidationHistorySize>
        shader_identity_invalidation_history{};
    // Lock-free publication for the overwhelmingly common "nothing changed" lookup path.
    std::atomic<u64> shader_identity_published_generation{};
    u64 shader_identity_submits{};
    u64 shader_identity_lookups{};
    u64 shader_identity_hits{};
    u64 shader_identity_scans{};
    u64 shader_identity_gpu_dirty_misses{};
    u64 shader_identity_gpu_alias_hits{};
    u64 shader_identity_invalidation_misses{};
    u64 shader_identity_dirty_after_scan{};
    u64 shader_identity_cpu_erases{};
    u64 shader_identity_submit_erases{};
    u64 shader_identity_capacity_misses{};

    std::jthread background_warmup_worker{};
    std::atomic<bool> background_warmup_active{false};
    std::atomic<u32> background_warmup_foreground_waiters{0};
    std::atomic<u64> background_warmup_wait_events{0};
    std::atomic<u64> background_warmup_wait_us{0};
    std::atomic<u64> background_warmup_max_wait_us{0};
    u64 background_warmup_generation{};
    std::mutex background_warmup_mutex{};
    struct PipelineNegativeCacheRecord {
        u32 failure_count{};
        u64 last_failure_epoch{};
        u64 retry_after_epoch{};
    };
    std::unordered_set<ComputePipelineKey> invalid_compute_pipelines{};
    std::unordered_set<GraphicsPipelineKey> invalid_graphics_pipelines{};
    std::unordered_map<ComputePipelineKey, PipelineNegativeCacheRecord>
        persistent_invalid_compute_pipelines{};
    std::unordered_map<GraphicsPipelineKey, PipelineNegativeCacheRecord>
        persistent_invalid_graphics_pipelines{};
    u64 pipeline_negative_cache_epoch{1};
    bool pipeline_negative_cache_loaded{};
    bool pipeline_negative_cache_dirty{};

    enum class AsyncPipelineTier : u8 {
        Fast,
        Optimized,
    };
    struct AsyncGraphicsPayload {
        GraphicsPipelineKey key{};
        std::array<const Shader::Info*, MaxShaderStages> infos{};
        std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
        std::array<vk::ShaderModule, MaxShaderStages> modules{};
        GraphicsPipeline::SerializationSupport sdata{};
    };
    struct AsyncComputePayload {
        ComputePipelineKey key{};
        const Shader::Info* info{};
        vk::ShaderModule module{};
        ComputePipeline::SerializationSupport sdata{};
    };
    using AsyncPipelinePayload =
        std::variant<std::shared_ptr<AsyncGraphicsPayload>,
                     std::shared_ptr<AsyncComputePayload>>;
    struct AsyncPipelineJob {
        AsyncPipelinePayload payload{};
        AsyncPipelineTier tier{AsyncPipelineTier::Fast};
        std::chrono::steady_clock::time_point queued_at{};
    };
    struct AsyncGraphicsResult {
        std::shared_ptr<AsyncGraphicsPayload> payload{};
        std::unique_ptr<GraphicsPipeline> pipeline{};
        vk::UniquePipelineCache local_cache{};
        AsyncPipelineTier tier{AsyncPipelineTier::Fast};
        vk::Result result{vk::Result::eNotReady};
        u64 build_us{};
    };
    struct AsyncComputeResult {
        std::shared_ptr<AsyncComputePayload> payload{};
        std::unique_ptr<ComputePipeline> pipeline{};
        vk::UniquePipelineCache local_cache{};
        AsyncPipelineTier tier{AsyncPipelineTier::Fast};
        vk::Result result{vk::Result::eNotReady};
        u64 build_us{};
    };
    using AsyncPipelineResult =
        std::variant<AsyncGraphicsResult, AsyncComputeResult>;

    struct AsyncShaderPayload {
        Program* program{};
        size_t perm_idx{};
        u64 expected_generation{};
        u64 cache_generation{};
        vk::ShaderModule expected_module{};
        std::array<u32, Shader::NUM_USER_DATA_REGS> user_data{};
        std::vector<u32> code{};
        std::unique_ptr<Shader::Info> info{};
        std::unique_ptr<Shader::Pools> pools{};
        std::unique_ptr<Shader::IR::Program> ir{};
        Shader::RuntimeInfo source_runtime_info{};
        Shader::RuntimeInfo runtime_info{};
        Shader::Backend::Bindings binding_start{};
        Shader::Backend::Bindings expected_binding_end{};
        size_t queued_bytes{};
    };
    struct AsyncShaderResult {
        std::shared_ptr<AsyncShaderPayload> payload{};
        vk::ShaderModule module{};
        std::vector<u32> spv{};
        vk::Result result{vk::Result::eNotReady};
        u64 build_us{};
    };
    struct AsyncShaderKey {
        Program* program{};
        size_t perm_idx{};
        u64 generation{};
        u64 cache_generation{};

        friend bool operator==(const AsyncShaderKey&, const AsyncShaderKey&) =
            default;
    };
    struct AsyncShaderKeyHash {
        size_t operator()(const AsyncShaderKey& key) const noexcept {
            size_t value = std::hash<Program*>{}(key.program);
            value ^= std::hash<size_t>{}(key.perm_idx) +
                     0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
            value ^= std::hash<u64>{}(key.generation) +
                     0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
            value ^= std::hash<u64>{}(key.cache_generation) +
                     0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2);
            return value;
        }
    };
    std::vector<std::jthread> async_pipeline_workers{};
    std::jthread async_shader_worker{};
    std::deque<AsyncPipelineJob> async_pipeline_jobs{};
    std::deque<AsyncPipelineResult> async_pipeline_results{};
    std::deque<std::shared_ptr<AsyncShaderPayload>> async_shader_jobs{};
    std::deque<AsyncShaderResult> async_shader_results{};
    std::unordered_set<GraphicsPipelineKey> pending_fast_graphics_pipelines{};
    std::unordered_set<GraphicsPipelineKey> pending_optimized_graphics_pipelines{};
    std::unordered_set<ComputePipelineKey> pending_fast_compute_pipelines{};
    std::unordered_set<ComputePipelineKey> pending_optimized_compute_pipelines{};
    std::unordered_set<GraphicsPipelineKey> fallback_graphics_pipelines{};
    std::unordered_set<ComputePipelineKey> fallback_compute_pipelines{};
    std::unordered_set<AsyncShaderKey, AsyncShaderKeyHash>
        pending_optimized_shaders{};
    std::unordered_set<AsyncShaderKey, AsyncShaderKeyHash>
        invalid_optimized_shaders{};
    std::unordered_map<vk::ShaderModule, u32>
        active_async_pipeline_modules{};
    std::mutex async_pipeline_mutex{};
    std::condition_variable_any async_pipeline_cv{};
    size_t async_pipeline_queue_capacity{128};
    size_t async_shader_queue_capacity{64};
    size_t async_shader_queue_byte_capacity{32 * 1024 * 1024};
    size_t async_shader_queued_bytes{};
    bool async_pipeline_accepting{};
    bool async_pipeline_started{};
    std::atomic<u64> async_last_foreground_activity_ms{};
    u64 shader_tier_generation{1};
    std::vector<vk::ShaderModule> retired_shader_modules{};

    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

}
