// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/config.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/scm_rev.h"
#include "common/serdes.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <type_traits>

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/resource.h>
#endif

namespace Serialization {
// Version 24 invalidates optimized tier binaries produced by the former
// Fast-IR mutation path. Optimized binaries are now independently translated
// from guest code, matching the synchronous compiler contract.
static constexpr u32 ShaderBinaryVersion = 28u;
static constexpr u32 ShaderMetaVersion = 1u;
static constexpr u32 PipelineKeyVersion = 3u;
}

namespace Vulkan {

namespace {

constexpr u32 ShaderTierNegativeMagic = 0x53544e47u;
constexpr u32 ShaderTierNegativeVersion = 1u;

std::string ShaderBinaryName(const u64 pgm_hash, const size_t perm_idx,
                             const ShaderBinaryTier tier) {
    const auto suffix =
        tier == ShaderBinaryTier::Optimized ? "_opt" : "_fast";
    return fmt::format("{:#018x}_{}{}", pgm_hash, perm_idx, suffix);
}

std::string ShaderTierNegativeName(const u64 pgm_hash, const size_t perm_idx,
                                   const ShaderBinaryTier tier) {
    return fmt::format("{}_tier_negative",
                       ShaderBinaryName(pgm_hash, perm_idx, tier));
}

constexpr u64 PipelineNegativeCacheMagic = 0x4c5358344e454731ULL;
constexpr u32 PipelineNegativeCacheVersion = 2;
constexpr u32 PipelineNegativeCacheBuildAbiVersion = 1;
constexpr u32 MaxNegativePipelineEntries = 32768;
constexpr u64 MaxNegativePipelineCacheBytes = 64ULL * 1024 * 1024;
constexpr u32 PersistentNegativeFailureThreshold = 2;
constexpr u64 MaxNegativePipelineRetryBackoffRuns = 8;

struct PipelineNegativeCacheHeader {
    u64 magic{};
    u32 format_version{};
    u32 pipeline_key_version{};
    u32 shader_binary_version{};
    u32 pointer_size{};
    u64 build_identity{};
    u64 cache_epoch{};
    u32 vendor_id{};
    u32 device_id{};
    u32 driver_version{};
    u32 graphics_key_size{};
    u32 compute_key_size{};
    u32 graphics_record_size{};
    u32 compute_record_size{};
    u32 graphics_count{};
    u32 compute_count{};
    std::array<u8, VK_UUID_SIZE> pipeline_cache_uuid{};
};

template <typename Key>
struct PipelineNegativeCacheDiskRecord {
    Key key{};
    u32 failure_count{};
    u32 reserved{};
    u64 last_failure_epoch{};
    u64 retry_after_epoch{};
};

static_assert(std::is_trivially_copyable_v<
              PipelineNegativeCacheDiskRecord<GraphicsPipelineKey>>);
static_assert(std::is_trivially_copyable_v<
              PipelineNegativeCacheDiskRecord<ComputePipelineKey>>);

u64 PipelineNegativeCacheBuildIdentity() {
    // A clean build revision rejects negative results produced by another
    // emulator build. The explicit ABI revision also covers local/dirty builds
    // whenever the validity contract changes.
    constexpr u64 FnvOffsetBasis = 14695981039346656037ULL;
    constexpr u64 FnvPrime = 1099511628211ULL;
    u64 hash = FnvOffsetBasis;
    const auto mix_text = [&](const std::string_view text) {
        for (const char value : text) {
            hash ^= static_cast<u8>(value);
            hash *= FnvPrime;
        }
        hash ^= 0xff;
        hash *= FnvPrime;
    };
    const auto mix_u32 = [&](u32 value) {
        for (u32 byte = 0; byte < sizeof(value); ++byte) {
            hash ^= static_cast<u8>(value >> (byte * 8));
            hash *= FnvPrime;
        }
    };
    mix_text(Common::g_scm_rev);
    mix_text(Common::g_version);
    mix_u32(PipelineNegativeCacheBuildAbiVersion);
    mix_u32(Serialization::PipelineKeyVersion);
    mix_u32(Serialization::ShaderBinaryVersion);
    return hash;
}

u64 NegativePipelineRetryDelay(const u32 failure_count) {
    if (failure_count < PersistentNegativeFailureThreshold) {
        return 1;
    }
    const u32 shift = std::min(failure_count - 1, 3u);
    return std::min(u64{1} << shift, MaxNegativePipelineRetryBackoffRuns);
}

struct ScopedWarmUpShaderModules {
    explicit ScopedWarmUpShaderModules(const vk::Device device_) : device{device_} {}

    ~ScopedWarmUpShaderModules() {
        for (const vk::ShaderModule module : modules) {
            if (module != vk::ShaderModule{}) {
                device.destroyShaderModule(module);
            }
        }
    }

    void Add(const vk::ShaderModule module) {
        modules.push_back(module);
    }

    vk::Device device;
    std::vector<vk::ShaderModule> modules;
};

template <typename T>
void AppendObject(std::vector<u8>& bytes, const T& object) {
    const auto old_size = bytes.size();
    bytes.resize(old_size + sizeof(T));
    std::memcpy(bytes.data() + old_size, &object, sizeof(T));
}

} // Anonymous namespace

std::string PipelineCache::PipelineNegativeCacheName() const {
    return fmt::format("vk-negative-{:08x}-{:08x}-{:08x}-p{}-s{}.bin",
                       instance.GetVendorID(), instance.GetDeviceID(),
                       instance.GetDriverVersion(), Serialization::PipelineKeyVersion,
                       Serialization::ShaderBinaryVersion);
}

bool PipelineCache::IsKnownInvalid(const ComputePipelineKey& key) const {
    return invalid_compute_pipelines.contains(key);
}

bool PipelineCache::IsKnownInvalid(const GraphicsPipelineKey& key) const {
    return invalid_graphics_pipelines.contains(key);
}

void PipelineCache::RecordInvalid(const ComputePipelineKey& key) {
    invalid_compute_pipelines.emplace(key);
    auto& record = persistent_invalid_compute_pipelines[key];
    if (record.last_failure_epoch != pipeline_negative_cache_epoch) {
        if (record.failure_count != std::numeric_limits<u32>::max()) {
            ++record.failure_count;
        }
        record.last_failure_epoch = pipeline_negative_cache_epoch;
        record.retry_after_epoch = pipeline_negative_cache_epoch >
                                           std::numeric_limits<u64>::max() -
                                               MaxNegativePipelineRetryBackoffRuns
                                       ? std::numeric_limits<u64>::max()
                                       : pipeline_negative_cache_epoch +
                                             NegativePipelineRetryDelay(
                                                 record.failure_count);
        pipeline_negative_cache_dirty = true;
    }
}

void PipelineCache::RecordInvalid(const GraphicsPipelineKey& key) {
    invalid_graphics_pipelines.emplace(key);
    auto& record = persistent_invalid_graphics_pipelines[key];
    if (record.last_failure_epoch != pipeline_negative_cache_epoch) {
        if (record.failure_count != std::numeric_limits<u32>::max()) {
            ++record.failure_count;
        }
        record.last_failure_epoch = pipeline_negative_cache_epoch;
        record.retry_after_epoch = pipeline_negative_cache_epoch >
                                           std::numeric_limits<u64>::max() -
                                               MaxNegativePipelineRetryBackoffRuns
                                       ? std::numeric_limits<u64>::max()
                                       : pipeline_negative_cache_epoch +
                                             NegativePipelineRetryDelay(
                                                 record.failure_count);
        pipeline_negative_cache_dirty = true;
    }
}

void PipelineCache::RecordValid(const ComputePipelineKey& key) {
    invalid_compute_pipelines.erase(key);
    pipeline_negative_cache_dirty |=
        persistent_invalid_compute_pipelines.erase(key) != 0;
}

void PipelineCache::RecordValid(const GraphicsPipelineKey& key) {
    invalid_graphics_pipelines.erase(key);
    pipeline_negative_cache_dirty |=
        persistent_invalid_graphics_pipelines.erase(key) != 0;
}

void PipelineCache::LoadPipelineNegativeCache() {
    if (pipeline_negative_cache_loaded || !Config::isPipelineCacheEnabled()) {
        return;
    }
    pipeline_negative_cache_loaded = true;

    const auto& game_info = Common::ElfInfo::Instance();
    const auto path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                      game_info.GameSerial() / PipelineNegativeCacheName();
    const Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read};
    const u64 file_size = file.GetSize();
    if (!file.IsOpen() || file_size < sizeof(PipelineNegativeCacheHeader) ||
        file_size > MaxNegativePipelineCacheBytes) {
        return;
    }

    std::vector<u8> bytes(static_cast<size_t>(file_size));
    if (file.Read(bytes) != bytes.size()) {
        return;
    }
    PipelineNegativeCacheHeader header{};
    std::memcpy(&header, bytes.data(), sizeof(header));
    using GraphicsDiskRecord =
        PipelineNegativeCacheDiskRecord<GraphicsPipelineKey>;
    using ComputeDiskRecord =
        PipelineNegativeCacheDiskRecord<ComputePipelineKey>;
    const auto expected_size =
        sizeof(header) +
        static_cast<u64>(header.graphics_count) * sizeof(GraphicsDiskRecord) +
        static_cast<u64>(header.compute_count) * sizeof(ComputeDiskRecord);
    const auto& pipeline_cache_uuid = instance.GetPipelineCacheUUID();
    if (header.magic != PipelineNegativeCacheMagic ||
        header.format_version != PipelineNegativeCacheVersion ||
        header.pipeline_key_version != Serialization::PipelineKeyVersion ||
        header.shader_binary_version != Serialization::ShaderBinaryVersion ||
        header.pointer_size != sizeof(void*) ||
        header.build_identity != PipelineNegativeCacheBuildIdentity() ||
        header.cache_epoch == std::numeric_limits<u64>::max() ||
        header.vendor_id != instance.GetVendorID() ||
        header.device_id != instance.GetDeviceID() ||
        header.driver_version != instance.GetDriverVersion() ||
        header.graphics_key_size != sizeof(GraphicsPipelineKey) ||
        header.compute_key_size != sizeof(ComputePipelineKey) ||
        header.graphics_record_size != sizeof(GraphicsDiskRecord) ||
        header.compute_record_size != sizeof(ComputeDiskRecord) ||
        header.graphics_count > MaxNegativePipelineEntries ||
        header.compute_count > MaxNegativePipelineEntries ||
        expected_size != bytes.size() ||
        std::memcmp(header.pipeline_cache_uuid.data(), pipeline_cache_uuid.data(),
                    VK_UUID_SIZE) != 0) {
        LOG_WARNING(Render_Vulkan, "Ignoring incompatible pipeline negative cache");
        // Replace legacy, corrupt, or build-incompatible files at the next
        // checkpoint instead of reconsidering the same poisoned data forever.
        pipeline_negative_cache_dirty = true;
        return;
    }

    pipeline_negative_cache_epoch = header.cache_epoch + 1;
    size_t offset = sizeof(header);
    size_t blocked_graphics{};
    size_t blocked_compute{};
    size_t retry_graphics{};
    size_t retry_compute{};
    invalid_graphics_pipelines.reserve(header.graphics_count);
    persistent_invalid_graphics_pipelines.reserve(header.graphics_count);
    for (u32 i = 0; i < header.graphics_count; ++i) {
        GraphicsDiskRecord disk{};
        std::memcpy(&disk, bytes.data() + offset, sizeof(disk));
        offset += sizeof(disk);
        if (disk.failure_count == 0 ||
            disk.last_failure_epoch > header.cache_epoch ||
            disk.retry_after_epoch < disk.last_failure_epoch ||
            disk.retry_after_epoch - disk.last_failure_epoch >
                MaxNegativePipelineRetryBackoffRuns) {
            pipeline_negative_cache_dirty = true;
            continue;
        }
        persistent_invalid_graphics_pipelines.emplace(
            disk.key, PipelineNegativeCacheRecord{
                          .failure_count = disk.failure_count,
                          .last_failure_epoch = disk.last_failure_epoch,
                          .retry_after_epoch = disk.retry_after_epoch,
                      });
        if (disk.failure_count >= PersistentNegativeFailureThreshold &&
            pipeline_negative_cache_epoch < disk.retry_after_epoch) {
            invalid_graphics_pipelines.emplace(disk.key);
            ++blocked_graphics;
        } else {
            ++retry_graphics;
        }
    }
    invalid_compute_pipelines.reserve(header.compute_count);
    persistent_invalid_compute_pipelines.reserve(header.compute_count);
    for (u32 i = 0; i < header.compute_count; ++i) {
        ComputeDiskRecord disk{};
        std::memcpy(&disk, bytes.data() + offset, sizeof(disk));
        offset += sizeof(disk);
        if (disk.failure_count == 0 ||
            disk.last_failure_epoch > header.cache_epoch ||
            disk.retry_after_epoch < disk.last_failure_epoch ||
            disk.retry_after_epoch - disk.last_failure_epoch >
                MaxNegativePipelineRetryBackoffRuns) {
            pipeline_negative_cache_dirty = true;
            continue;
        }
        persistent_invalid_compute_pipelines.emplace(
            disk.key, PipelineNegativeCacheRecord{
                          .failure_count = disk.failure_count,
                          .last_failure_epoch = disk.last_failure_epoch,
                          .retry_after_epoch = disk.retry_after_epoch,
                      });
        if (disk.failure_count >= PersistentNegativeFailureThreshold &&
            pipeline_negative_cache_epoch < disk.retry_after_epoch) {
            invalid_compute_pipelines.emplace(disk.key);
            ++blocked_compute;
        } else {
            ++retry_compute;
        }
    }
    // Epoch-based backoff only advances when the new epoch is checkpointed.
    pipeline_negative_cache_dirty |=
        !persistent_invalid_graphics_pipelines.empty() ||
        !persistent_invalid_compute_pipelines.empty();
#ifdef __ANDROID__
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_PIPELINE_NEGATIVE_CACHE] epoch=%llu loadedGraphics=%zu "
        "loadedCompute=%zu blockedGraphics=%zu blockedCompute=%zu "
        "retryGraphics=%zu retryCompute=%zu",
        static_cast<unsigned long long>(pipeline_negative_cache_epoch),
        persistent_invalid_graphics_pipelines.size(),
        persistent_invalid_compute_pipelines.size(), blocked_graphics,
        blocked_compute, retry_graphics, retry_compute);
#endif
}

void PipelineCache::SavePipelineNegativeCache() {
    if (!pipeline_negative_cache_dirty || !Config::isPipelineCacheEnabled()) {
        return;
    }

    using GraphicsDiskRecord =
        PipelineNegativeCacheDiskRecord<GraphicsPipelineKey>;
    using ComputeDiskRecord =
        PipelineNegativeCacheDiskRecord<ComputePipelineKey>;

    std::vector<GraphicsDiskRecord> graphics;
    graphics.reserve(
        std::min<size_t>(persistent_invalid_graphics_pipelines.size(),
                         MaxNegativePipelineEntries));
    for (const auto& [key, record] : persistent_invalid_graphics_pipelines) {
        graphics.push_back({
            .key = key,
            .failure_count = record.failure_count,
            .last_failure_epoch = record.last_failure_epoch,
            .retry_after_epoch = record.retry_after_epoch,
        });
    }
    std::ranges::sort(graphics, [](const auto& lhs, const auto& rhs) {
        if (lhs.last_failure_epoch != rhs.last_failure_epoch) {
            return lhs.last_failure_epoch > rhs.last_failure_epoch;
        }
        return std::memcmp(&lhs.key, &rhs.key, sizeof(lhs.key)) < 0;
    });
    graphics.resize(
        std::min<size_t>(graphics.size(), MaxNegativePipelineEntries));

    std::vector<ComputeDiskRecord> compute;
    compute.reserve(
        std::min<size_t>(persistent_invalid_compute_pipelines.size(),
                         MaxNegativePipelineEntries));
    for (const auto& [key, record] : persistent_invalid_compute_pipelines) {
        compute.push_back({
            .key = key,
            .failure_count = record.failure_count,
            .last_failure_epoch = record.last_failure_epoch,
            .retry_after_epoch = record.retry_after_epoch,
        });
    }
    std::ranges::sort(compute, [](const auto& lhs, const auto& rhs) {
        if (lhs.last_failure_epoch != rhs.last_failure_epoch) {
            return lhs.last_failure_epoch > rhs.last_failure_epoch;
        }
        return lhs.key.value < rhs.key.value;
    });
    compute.resize(std::min<size_t>(compute.size(), MaxNegativePipelineEntries));

    PipelineNegativeCacheHeader header{
        .magic = PipelineNegativeCacheMagic,
        .format_version = PipelineNegativeCacheVersion,
        .pipeline_key_version = Serialization::PipelineKeyVersion,
        .shader_binary_version = Serialization::ShaderBinaryVersion,
        .pointer_size = static_cast<u32>(sizeof(void*)),
        .build_identity = PipelineNegativeCacheBuildIdentity(),
        .cache_epoch = pipeline_negative_cache_epoch,
        .vendor_id = instance.GetVendorID(),
        .device_id = instance.GetDeviceID(),
        .driver_version = instance.GetDriverVersion(),
        .graphics_key_size = static_cast<u32>(sizeof(GraphicsPipelineKey)),
        .compute_key_size = static_cast<u32>(sizeof(ComputePipelineKey)),
        .graphics_record_size = static_cast<u32>(sizeof(GraphicsDiskRecord)),
        .compute_record_size = static_cast<u32>(sizeof(ComputeDiskRecord)),
        .graphics_count = static_cast<u32>(graphics.size()),
        .compute_count = static_cast<u32>(compute.size()),
    };
    const auto& pipeline_cache_uuid = instance.GetPipelineCacheUUID();
    std::memcpy(header.pipeline_cache_uuid.data(), pipeline_cache_uuid.data(), VK_UUID_SIZE);

    std::vector<u8> bytes;
    bytes.reserve(sizeof(header) + graphics.size() * sizeof(GraphicsDiskRecord) +
                  compute.size() * sizeof(ComputeDiskRecord));
    AppendObject(bytes, header);
    for (const auto& key : graphics) {
        AppendObject(bytes, key);
    }
    for (const auto& key : compute) {
        AppendObject(bytes, key);
    }

    const auto& game_info = Common::ElfInfo::Instance();
    const auto directory = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                           game_info.GameSerial();
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        return;
    }
    if (!WriteCacheFileAtomically(directory / PipelineNegativeCacheName(), bytes)) {
        return;
    }
    pipeline_negative_cache_dirty = false;
#ifdef __ANDROID__
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_PIPELINE_NEGATIVE_CACHE] epoch=%llu savedGraphics=%zu "
        "savedCompute=%zu bytes=%zu",
        static_cast<unsigned long long>(pipeline_negative_cache_epoch),
        graphics.size(), compute.size(), bytes.size());
#endif
}

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1});

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
    pldata.Write(u32{0});

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

void RegisterShaderBinary(std::vector<u32>&& spv, const u64 pgm_hash,
                          const size_t perm_idx, const ShaderBinaryTier tier) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       ShaderBinaryName(pgm_hash, perm_idx, tier),
                                       std::move(spv));
    RecordShaderBinaryTierValid(pgm_hash, perm_idx, tier);
}

bool LoadShaderBinary(std::vector<u32>& spv, const u64 pgm_hash,
                      const size_t perm_idx, ShaderBinaryTier* loaded_tier) {
    spv.clear();
    auto& database = Storage::DataBase::Instance();
    database.Load(Storage::BlobType::ShaderBinary,
                  ShaderBinaryName(pgm_hash, perm_idx,
                                   ShaderBinaryTier::Optimized),
                  spv);
    if (!spv.empty()) {
        if (loaded_tier) {
            *loaded_tier = ShaderBinaryTier::Optimized;
        }
        return true;
    }
    // Pre-tiered caches stored the fully optimized translation under the
    // unsuffixed key.  Keep them valid and classify them as optimized.
    database.Load(Storage::BlobType::ShaderBinary,
                  fmt::format("{:#018x}_{}", pgm_hash, perm_idx), spv);
    if (!spv.empty()) {
        if (loaded_tier) {
            *loaded_tier = ShaderBinaryTier::Optimized;
        }
        return true;
    }
    database.Load(Storage::BlobType::ShaderBinary,
                  ShaderBinaryName(pgm_hash, perm_idx, ShaderBinaryTier::Fast),
                  spv);
    if (!spv.empty()) {
        if (loaded_tier) {
            *loaded_tier = ShaderBinaryTier::Fast;
        }
        return true;
    }
    return false;
}

bool IsShaderBinaryTierKnownInvalid(const u64 pgm_hash, const size_t perm_idx,
                                    const ShaderBinaryTier tier) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return false;
    }
    std::vector<u32> marker;
    Storage::DataBase::Instance().Load(
        Storage::BlobType::ShaderBinary,
        ShaderTierNegativeName(pgm_hash, perm_idx, tier), marker);
    return marker.size() >= 4 && marker[0] == ShaderTierNegativeMagic &&
           marker[1] == ShaderTierNegativeVersion &&
           marker[2] == Serialization::ShaderBinaryVersion && marker[3] != 0;
}

void RecordShaderBinaryTierFailure(const u64 pgm_hash, const size_t perm_idx,
                                   const ShaderBinaryTier tier) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }
    Storage::DataBase::Instance().Save(
        Storage::BlobType::ShaderBinary,
        ShaderTierNegativeName(pgm_hash, perm_idx, tier),
        std::vector<u32>{ShaderTierNegativeMagic, ShaderTierNegativeVersion,
                         Serialization::ShaderBinaryVersion, 1u});
}

void RecordShaderBinaryTierValid(const u64 pgm_hash, const size_t perm_idx,
                                 const ShaderBinaryTier tier) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }
    Storage::DataBase::Instance().Save(
        Storage::BlobType::ShaderBinary,
        ShaderTierNegativeName(pgm_hash, perm_idx, tier),
        std::vector<u32>{ShaderTierNegativeMagic, ShaderTierNegativeVersion,
                         Serialization::ShaderBinaryVersion, 0u});
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    constexpr size_t HeaderSize = sizeof(u32) * 2 + sizeof(u64) + sizeof(size_t);
    if (ar.SizeBytes() < HeaderSize) {
        return false;
    }
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
    Serialization::Writer sdata{ar};
    sdata.Write(storage_buffers);
    sdata.Write(image_bindings);
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};
    sdata.Read(storage_buffers);
    sdata.Read(image_bindings);
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    ResetPipelineBuildState();
    compute_key.Deserialize(ar);
    if (const auto it = compute_pipelines.find(compute_key);
        it != compute_pipelines.end()) {
        return it->second && it->second->IsValid();
    }
    if (IsKnownInvalid(compute_key)) {
        compute_pipelines.try_emplace(compute_key);
        return false;
    }

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        ResetPipelineBuildState();
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        ResetPipelineBuildState();
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);

    const bool valid = it.value()->IsValid();
    if (valid) {
        RecordValid(compute_key);
        module_related_pipelines[modules[0]].emplace_back(compute_key);
        MarkHostPipelineCacheDirty();
    } else {
        RecordInvalid(compute_key);
    }
    ResetPipelineBuildState();
    return valid;
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
    sdata.Write(storage_buffers);
    sdata.Write(image_bindings);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(storage_buffers);
    sdata.Read(image_bindings);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    ResetPipelineBuildState();
    graphics_key.Deserialize(ar);
    if (const auto it = graphics_pipelines.find(graphics_key);
        it != graphics_pipelines.end()) {
        return it->second && it->second->IsValid();
    }
    if (IsKnownInvalid(graphics_key)) {
        graphics_pipelines.try_emplace(graphics_key);
        return false;
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
            ResetPipelineBuildState();
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            ResetPipelineBuildState();
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);

    const bool valid = it.value()->IsValid();
    if (valid) {
        RecordValid(graphics_key);
        for (size_t stage = 0; stage < MaxShaderStages; ++stage) {
            if (infos[stage]) {
                module_related_pipelines[modules[stage]].emplace_back(
                    graphics_key);
            }
        }
        MarkHostPipelineCacheDirty();
    } else {
        RecordInvalid(graphics_key);
    }
    ResetPipelineBuildState();
    return valid;
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
    ShaderBinaryTier loaded_tier{};
    if (!LoadShaderBinary(spv, program->info.pgm_hash, perm_idx,
                          &loaded_tier)) {
        return false;
    }


    vk::ShaderModule module{};
    bool reused_permutation = false;

    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        module = CompileSPV(spv, instance.GetDevice());
        it_pgm.value() = std::move(program);
    } else {
        const auto& it = std::ranges::find(it_pgm.value()->modules, spec, &Program::Module::spec);
        if (it != it_pgm.value()->modules.end()) {
            const auto idx = std::distance(it_pgm.value()->modules.begin(), it);
            ASSERT_MSG(perm_idx == idx, "Permutation {} is already inserted at {}! ({}_{:x})",
                       perm_idx, idx, program->info.stage, program->info.pgm_hash);
            module = it->module;
            reused_permutation = true;
        } else {
            module = CompileSPV(spv, instance.GetDevice());
        }
    }
    // When the program already existed, the temporary `program` is destroyed
    // on return. Never publish its Info pointer through the specialization.
    if (!reused_permutation) {
        spec.info = &it_pgm.value()->info;
        it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx,
                                     loaded_tier);
    }

    infos[stage] = &it_pgm.value()->info;
    modules[stage] = module;

    return true;
}

bool PipelineCache::LoadIsolatedWarmUpPipeline(Serialization::Archive& ar,
                                               const bool is_compute,
                                               const u64 warmup_generation) {
    BackgroundWarmUpInfos owned_infos{};
    std::array<const Shader::Info*, MaxShaderStages> warm_infos{};
    std::array<vk::ShaderModule, MaxShaderStages> warm_modules{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> warm_runtime_infos{};
    std::optional<Shader::Gcn::FetchShaderData> warm_fetch_shader{};
    const vk::Device device = instance.GetDevice();
    ScopedWarmUpShaderModules shader_modules{device};

    const auto load_stage =
        [&](const u64 metadata_hash, const size_t stage) -> bool {
        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(
            Storage::BlobType::ShaderMeta,
            fmt::format("{:#018x}", metadata_hash), meta_blob);
        if (meta_blob.empty()) {
            return false;
        }

        auto info = std::make_unique<Shader::Info>();
        Shader::StageSpecialization spec{};
        spec.info = info.get();
        std::optional<Shader::Gcn::FetchShaderData> stage_fetch_shader;
        size_t perm_idx{};
        Serialization::Archive meta_ar{std::move(meta_blob)};
        if (!LoadShaderMeta(meta_ar, *info, stage_fetch_shader, spec, perm_idx)) {
            return false;
        }

        std::vector<u32> spv;
        if (!LoadShaderBinary(spv, info->pgm_hash, perm_idx)) {
            return false;
        }

        const vk::ShaderModule module = CompileSPV(spv, device);
        shader_modules.Add(module);
        warm_infos[stage] = info.get();
        warm_modules[stage] = module;
        owned_infos[stage] = std::move(info);
        if (stage_fetch_shader) {
            warm_fetch_shader = std::move(stage_fetch_shader);
        }
        return true;
    };

    if (is_compute) {
        ComputePipelineKey key{};
        key.Deserialize(ar);
        {
            std::scoped_lock lock{background_warmup_mutex};
            if (warmup_generation != background_warmup_generation) {
                return false;
            }
            if (const auto it = compute_pipelines.find(key);
                it != compute_pipelines.end()) {
                return it->second && it->second->IsValid();
            }
            if (IsKnownInvalid(key)) {
                return false;
            }
        }

        ComputePipeline::SerializationSupport sdata{};
        sdata.Deserialize(ar);
        if (!load_stage(key.value, 0)) {
            return false;
        }

        auto local_cache_result = device.createPipelineCacheUnique({});
        if (local_cache_result.result != vk::Result::eSuccess) {
            return false;
        }
        auto local_cache = std::move(local_cache_result.value);
        auto warmed_pipeline = std::make_unique<ComputePipeline>(
            instance, scheduler, desc_heap, profile, *local_cache, key,
            *warm_infos[0], warm_modules[0], sdata, true);
        const bool valid = warmed_pipeline->IsValid();

        std::scoped_lock lock{background_warmup_mutex};
        if (warmup_generation != background_warmup_generation) {
            return false;
        }
        if (const auto it = compute_pipelines.find(key);
            it != compute_pipelines.end()) {
            return it->second && it->second->IsValid();
        }
        if (!valid || IsKnownInvalid(key)) {
            if (!valid) {
                RecordInvalid(key);
                compute_pipelines.try_emplace(key);
            }
            return false;
        }
        RecordValid(key);
        const std::array source_caches{*local_cache};
        const auto merge_result =
            device.mergePipelineCaches(*pipeline_cache, source_caches);
        if (merge_result == vk::Result::eSuccess) {
            MarkHostPipelineCacheDirty();
        } else {
            LOG_WARNING(Render_Vulkan,
                        "Failed to merge background compute pipeline cache: {}",
                        vk::to_string(merge_result));
        }
        const bool info_is_new =
            background_warmup_compute_infos
                .try_emplace(key, std::move(owned_infos))
                .second;
        ASSERT(info_is_new);
        const auto [pipeline_it, pipeline_is_new] =
            compute_pipelines.try_emplace(key);
        ASSERT(pipeline_is_new);
        pipeline_it.value() = std::move(warmed_pipeline);
        return true;
    }

    GraphicsPipelineKey key{};
    key.Deserialize(ar);
    {
        std::scoped_lock lock{background_warmup_mutex};
        if (warmup_generation != background_warmup_generation) {
            return false;
        }
        if (const auto it = graphics_pipelines.find(key);
            it != graphics_pipelines.end()) {
            return it->second && it->second->IsValid();
        }
        if (IsKnownInvalid(key)) {
            return false;
        }
    }

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);
    for (size_t stage = 0; stage < MaxShaderStages; ++stage) {
        const u64 metadata_hash = key.stage_hashes[stage];
        if (metadata_hash != 0 && !load_stage(metadata_hash, stage)) {
            return false;
        }
    }

    auto local_cache_result = device.createPipelineCacheUnique({});
    if (local_cache_result.result != vk::Result::eSuccess) {
        return false;
    }
    auto local_cache = std::move(local_cache_result.value);
    auto warmed_pipeline = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, key, *local_cache,
        warm_infos, warm_runtime_infos, warm_fetch_shader, warm_modules,
        sdata, true);
    const bool valid = warmed_pipeline->IsValid();

    std::scoped_lock lock{background_warmup_mutex};
    if (warmup_generation != background_warmup_generation) {
        return false;
    }
    if (const auto it = graphics_pipelines.find(key);
        it != graphics_pipelines.end()) {
        return it->second && it->second->IsValid();
    }
    if (!valid || IsKnownInvalid(key)) {
        if (!valid) {
            RecordInvalid(key);
            graphics_pipelines.try_emplace(key);
        }
        return false;
    }
    RecordValid(key);
    const std::array source_caches{*local_cache};
    const auto merge_result =
        device.mergePipelineCaches(*pipeline_cache, source_caches);
    if (merge_result == vk::Result::eSuccess) {
        MarkHostPipelineCacheDirty();
    } else {
        LOG_WARNING(Render_Vulkan,
                    "Failed to merge background graphics pipeline cache: {}",
                    vk::to_string(merge_result));
    }
    const bool info_is_new =
        background_warmup_graphics_infos
            .try_emplace(key, std::move(owned_infos))
            .second;
    ASSERT(info_is_new);
    const auto [pipeline_it, pipeline_is_new] =
        graphics_pipelines.try_emplace(key);
    ASSERT(pipeline_is_new);
    pipeline_it.value() = std::move(warmed_pipeline);
    return true;
}

void PipelineCache::WarmUp() {
    if (!Config::isPipelineCacheEnabled() || Config::patchShaders()) {
        return;
    }

    Storage::DataBase::Instance().Open();

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

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Ignoring the cache");
        Storage::DataBase::Instance().FinishPreload();
        return;
    }
    LoadPipelineNegativeCache();

    u32 num_pipelines{};
    u32 num_total_pipelines{};
    u32 num_failed_pipelines{};
#ifdef __ANDROID__
    constexpr u32 kWarmPipelineLimit = 8;
    constexpr auto kWarmTimeBudget = std::chrono::milliseconds{250};
    const auto warm_start = std::chrono::steady_clock::now();
    bool budget_reached = false;
    std::vector<std::vector<u8>> deferred_pipeline_blobs;
#endif

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) -> bool {
            ++num_total_pipelines;
            constexpr size_t PipelineHeaderSize = sizeof(u32) * 2;
            if (data.size() < PipelineHeaderSize) {
                ++num_failed_pipelines;
                return true;
            }
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
                ++num_failed_pipelines;
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
            } else {
                ++num_failed_pipelines;
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
    if (num_failed_pipelines != 0) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_failed_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
#ifdef __ANDROID__
    if (!deferred_pipeline_blobs.empty()) {
        const char* const background_warmup_flag =
            std::getenv("EXECUTOR_BACKGROUND_PIPELINE_WARMUP");
        const bool background_warmup_enabled =
            Config::managedAsyncPipelineCompilation() &&
            !Config::isPipelineCacheArchived() &&
            (background_warmup_flag == nullptr ||
             (background_warmup_flag[0] != '\0' &&
              background_warmup_flag[0] != '0'));
        if (!background_warmup_enabled) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_PIPELINE_WARMUP_DEFERRED] disabled=1 queued=%zu "
                "reason=%s flag=EXECUTOR_BACKGROUND_PIPELINE_WARMUP",
                deferred_pipeline_blobs.size(),
                Config::isPipelineCacheArchived() ? "archivedCacheRequiresSerialIO"
                                                  : "explicitOptOut");
            return;
        }
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_PIPELINE_WARMUP_DEFERRED] queued=%zu graceMs=500 "
            "foregroundPriority=1 throttleMs=2 defaultEnabled=1",
            deferred_pipeline_blobs.size());
        StartBackgroundWarmUp(std::move(deferred_pipeline_blobs));
    }
#endif
}

void PipelineCache::StartBackgroundWarmUp(
    std::vector<std::vector<u8>>&& pipeline_blobs) {
#ifdef __ANDROID__
    StopBackgroundWarmUp();
    u64 run_generation{};
    {
        std::scoped_lock lock{background_warmup_mutex};
        run_generation = ++background_warmup_generation;
    }
    // Publish the gate before the worker can commit into shared maps. Slow
    // shader/driver work uses per-item state and a private VkPipelineCache.
    background_warmup_active.store(true, std::memory_order_release);
    background_warmup_worker = std::jthread{
        [this, blobs = std::move(pipeline_blobs),
         run_generation](const std::stop_token stop_token) mutable {
            (void)::setpriority(PRIO_PROCESS, 0, 8);
            constexpr auto kFirstFrameGrace = std::chrono::milliseconds{500};
            constexpr auto kPipelineThrottle = std::chrono::milliseconds{2};
            constexpr auto kForegroundCooldown = std::chrono::milliseconds{8};
            const auto grace_deadline = std::chrono::steady_clock::now() + kFirstFrameGrace;
            while (!stop_token.stop_requested() &&
                   std::chrono::steady_clock::now() < grace_deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds{2});
            }
            if (stop_token.stop_requested()) {
                background_warmup_active.store(false, std::memory_order_release);
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
            const auto started = std::chrono::steady_clock::now();
            u32 attempted = 0;
            u32 loaded = 0;
            u32 foreground_yields = 0;
            bool generation_invalidated = false;
            for (std::vector<u8>& data : blobs) {
                if (stop_token.stop_requested()) {
                    break;
                }
                constexpr size_t PipelineHeaderSize = sizeof(u32) * 2;
                if (data.size() < PipelineHeaderSize) {
                    continue;
                }

                while (!stop_token.stop_requested()) {
                    if (background_warmup_foreground_waiters.load(
                            std::memory_order_acquire) != 0) {
                        ++foreground_yields;
                        std::this_thread::sleep_for(kForegroundCooldown);
                        continue;
                    }
                    std::unique_lock background_lock{
                        background_warmup_mutex, std::try_to_lock};
                    if (!background_lock.owns_lock()) {
                        ++foreground_yields;
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds{1});
                        continue;
                    }
                    // A foreground miss may announce itself concurrently with
                    // try_lock. Hand it priority before starting the isolated
                    // driver compile.
                    std::this_thread::yield();
                    if (background_warmup_foreground_waiters.load(
                            std::memory_order_acquire) == 0) {
                        generation_invalidated =
                            background_warmup_generation != run_generation;
                        break;
                    }
                    ++foreground_yields;
                    background_lock.unlock();
                    std::this_thread::sleep_for(kForegroundCooldown);
                }
                if (stop_token.stop_requested() || generation_invalidated) {
                    break;
                }

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
                const bool result = LoadIsolatedWarmUpPipeline(
                    ar, is_compute != 0, run_generation);
                loaded += result ? 1u : 0u;
                {
                    std::scoped_lock lock{background_warmup_mutex};
                    generation_invalidated =
                        background_warmup_generation != run_generation;
                }
                if (generation_invalidated) {
                    break;
                }
                std::this_thread::sleep_for(kPipelineThrottle);
            }
            background_warmup_active.store(false, std::memory_order_release);
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - started)
                                        .count();
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_PIPELINE_WARMUP_BACKGROUND] loaded=%u attempted=%u total=%zu "
                "elapsedMs=%lld stopped=%d graceMs=%lld foregroundYields=%u "
                "foregroundWaits=%llu waitUs=%llu maxWaitUs=%llu isolated=1",
                loaded, attempted, blobs.size(), static_cast<long long>(elapsed_ms),
                (stop_token.stop_requested() || generation_invalidated) ? 1 : 0,
                static_cast<long long>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(kFirstFrameGrace)
                            .count()),
                foreground_yields,
                static_cast<unsigned long long>(
                    background_warmup_wait_events.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    background_warmup_wait_us.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    background_warmup_max_wait_us.load(std::memory_order_relaxed)));
        }};
#else
    (void)pipeline_blobs;
#endif
}

void PipelineCache::StopBackgroundWarmUp() {
    if (!background_warmup_worker.joinable()) {
        background_warmup_active.store(false, std::memory_order_release);
        return;
    }
    {
        std::scoped_lock lock{background_warmup_mutex};
        ++background_warmup_generation;
    }
    background_warmup_worker.request_stop();
    background_warmup_worker.join();
    background_warmup_active.store(false, std::memory_order_release);
}

void PipelineCache::Sync() {
    StopAsyncPipelineWorkers();
    PublishAsyncPipelineResults();
    DestroyRetiredShaderModules();
    StopBackgroundWarmUp();
    SaveHostPipelineCache();
    SavePipelineNegativeCache();
    Storage::DataBase::Instance().Close();
}

}

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

    srt.Write(walker_func_size);
    srt.Write(flattened_bufsize_dw);
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
    srt.Write(portable_nodes);
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

}
