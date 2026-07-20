// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
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

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    struct SpecializationSnapshot {
        // Only user-data words that can affect StageSpecialization participate. Per-draw push
        // constants do not change the Vk shader permutation and including all 16 words made a
        // cache miss for every object transform in large games.
        std::array<u32, Shader::NUM_USER_DATA_REGS> specialization_user_data{};
        u32 specialization_user_data_mask{};
        std::vector<u32> flattened_user_data{};
        Shader::RuntimeInfo runtime_info{};
        Shader::Backend::Bindings binding_start{};
        Shader::SrtMemoryRanges srt_ranges{};
        std::vector<u8> srt_source_bytes{};
        size_t module_index{};
        u64 identity_key{};
        bool valid{};
    };
    // Four-way set associativity prevents cyclic SRT working sets from evicting one another at a
    // single modulo slot. Bloodborne's room workload filled the old 512-entry direct map and then
    // missed roughly 75% of lookups despite gpuMiss=0 and very few content mismatches. Slots remain
    // lazily allocated; a globally bounded victim tier below retains displaced hot snapshots.
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
        // HashCombine deliberately keeps this key cheap, but its low bits retain patterns from
        // aligned SRT pointers. Avalanche before selecting a power-of-two set so rotating guest
        // descriptor tables do not collapse into a handful of conflict-heavy sets.
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

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        modules[perm_idx] = {module, std::move(spec)};
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
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    void StartBackgroundWarmUp(std::vector<std::vector<u8>>&& pipeline_blobs);
    void StopBackgroundWarmUp();
    bool RefreshGraphicsKey(std::optional<AmdGpu::PrimitiveType> host_primitive_override);
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
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
    Shader::ShaderParams GetShaderParams(const u32* user_data, const u32* code);
    [[nodiscard]] bool WasCpuRangeInvalidatedSinceLocked(VAddr address, u64 size,
                                                         u64 sequence) const;
    [[nodiscard]] u64 CaptureCpuInvalidationSequence();

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    VideoCore::BufferCache* buffer_cache;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
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
    std::optional<AmdGpu::PrimitiveType> graphics_primitive_override{};
    std::optional<AmdGpu::PrimitiveType> last_graphics_primitive_override{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    u64 srt_snapshot_lookups{};
    u64 srt_snapshot_hits{};
    u64 srt_snapshot_content_misses{};
    u64 srt_snapshot_gpu_misses{};
    u64 srt_snapshot_backing_alias_hits{};
    u64 srt_snapshot_backing_fallbacks{};
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
    // The primary cache remains embedded and allocation-light for every shader. Only snapshots
    // displaced by a saturated hot program enter this shared CLOCK cache, so the additional mobile
    // memory is globally bounded rather than multiplied by the number of shader programs.
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

    struct ShaderIdentityMetadata {
        VAddr address{};
        u32 length_bytes{};
        u64 hash{};
    };
    struct ShaderIdentityInvalidation {
        VAddr address{};
        u64 size{};
        u64 sequence{};
    };
    static constexpr size_t MaxShaderIdentityEntriesPerSubmit = 4096;
    static constexpr size_t ShaderIdentityInvalidationHistorySize = 4096;
    // Android precise readback can protect a shader page after the GPU aliases it. Re-running
    // SearchBinaryInfo for every draw then faults, downloads and waits even though the same program
    // was already identified earlier in this guest submit. This cache stores metadata only; every
    // lookup still receives the current register user_data and must prove the complete code span is
    // GPU-clean with an unchanged readback generation. CPU invalidation and submit boundaries
    // invalidate entries under this mutex.
    bool shader_identity_cache_enabled{};
    // Aggressive opt-in for titles which keep shader binaries immutable after their first
    // successful scan. CPU writes still erase overlapping entries. Conservative GPU writable
    // descriptor ranges are ignored on cache hits because they describe possible writes to an
    // allocation, not proof that the shader bytes changed.
    bool immutable_shader_identity_cache_enabled{};
    tsl::robin_map<VAddr, ShaderIdentityMetadata> shader_identity_cache;
    std::mutex shader_identity_cache_mutex{};
    std::array<ShaderIdentityInvalidation, ShaderIdentityInvalidationHistorySize>
        shader_identity_invalidation_history{};
    u64 shader_identity_generation{};
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

    // Android's driver can spend tens of seconds recreating a mature VkPipeline set. The initial
    // bounded subset is loaded synchronously; the remainder is rebuilt while the guest performs
    // CPU-only boot work. Every foreground pipeline lookup joins this worker before touching the
    // cache maps, so the existing single-owner contract remains intact without a hot-path mutex.
    std::jthread background_warmup_worker{};
    std::atomic<bool> background_warmup_active{false};
    std::atomic<u32> background_warmup_foreground_waiters{0};
    std::mutex background_warmup_mutex{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;
};

} // namespace Vulkan
