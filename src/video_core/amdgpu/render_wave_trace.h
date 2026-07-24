// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#ifdef __ANDROID__

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <android/log.h>

#include "common/content_fingerprint.h"
#include "common/types.h"
#include "core/memory.h"
#include "video_core/amdgpu/regs.h"

namespace AmdGpu::RenderWaveTrace {

inline constexpr const char* LateFrameArmMarker =
    "/data/data/app.lsx4.android/files/lsx4-home/diag-gpu-frame-arm";

inline std::atomic<bool>& LateFrameCaptureFlag() noexcept {
    static std::atomic<bool> active{false};
    return active;
}

inline std::atomic<u64>& LateFrameCaptureEpochCounter() noexcept {
    static std::atomic<u64> epoch{0};
    return epoch;
}

inline bool BaseEnabled() noexcept {
    static const bool enabled = [] {
        const char* value = std::getenv("EXECUTOR_TRACE_RENDER_WAVE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

inline bool Enabled() noexcept {
    return BaseEnabled() || LateFrameCaptureFlag().load(std::memory_order_acquire);
}

enum class Stage : u8 {
    Ps,
    Vs,
    Gs,
    Es,
    Hs,
    Ls,
};

inline const char* StageName(Stage stage) noexcept {
    switch (stage) {
    case Stage::Ps:
        return "PS";
    case Stage::Vs:
        return "VS";
    case Stage::Gs:
        return "GS";
    case Stage::Es:
        return "ES";
    case Stage::Hs:
        return "HS";
    case Stage::Ls:
        return "LS";
    }
    return "?";
}

inline const ShaderProgram& ProgramForStage(const Regs& regs, Stage stage) noexcept {
    switch (stage) {
    case Stage::Ps:
        return regs.ps_program;
    case Stage::Vs:
        return regs.vs_program;
    case Stage::Gs:
        return regs.gs_program;
    case Stage::Es:
        return regs.es_program;
    case Stage::Hs:
        return regs.hs_program;
    case Stage::Ls:
        return regs.ls_program;
    }
    return regs.ps_program;
}

struct ShaderIdentity {
    VAddr address{};
    u64 hash{};
    bool readable{};
    bool found{};
};

struct StageKey {
    u8 layer{};
    u8 stage{};
    u64 identity{};

    bool operator==(const StageKey&) const = default;
};

struct PairKey {
    u8 layer{};
    u64 vs{};
    u64 ps{};

    bool operator==(const PairKey&) const = default;
};

struct KeyHash {
    static u64 Mix(std::initializer_list<u64> values) noexcept {
        Common::ContentFingerprint64 fingerprint{Common::FingerprintDomain::RenderWaveKey};
        for (const u64 value : values) {
            fingerprint.UpdateLittleEndian(value);
        }
        return fingerprint.Finish();
    }

    std::size_t operator()(const StageKey& key) const noexcept {
        return static_cast<std::size_t>(Mix({key.layer, key.stage, key.identity}));
    }

    std::size_t operator()(const PairKey& key) const noexcept {
        return static_cast<std::size_t>(Mix({key.layer, key.vs, key.ps}));
    }
};

struct State {
    static constexpr u32 MaxLogRecords = 4096;
    static constexpr std::size_t MaxShaderProbeBytes = 0x10000 + sizeof(BinaryInfo);

    std::mutex mutex;
    std::unordered_map<VAddr, ShaderIdentity> shader_cache;
    std::unordered_set<StageKey, KeyHash> unique_stages;
    std::unordered_set<PairKey, KeyHash> unique_pairs;
    u32 emitted{};
    std::atomic<u64> submit_sequence{};
};

inline State& GetState() {
    static State state;
    return state;
}

inline bool ConsumeLateFrameArmMarker() noexcept {
    std::FILE* marker = std::fopen(LateFrameArmMarker, "rb");
    if (marker == nullptr) {
        return false;
    }
    std::fclose(marker);
    if (std::remove(LateFrameArmMarker) != 0) {
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_GPU_FRAME_ARM] marker_remove_failed path=%s",
                            LateFrameArmMarker);
    }
    return true;
}

inline u64 ArmLateFrameCapture() noexcept {
    auto& state = GetState();
    {
        std::scoped_lock lock{state.mutex};
        state.unique_stages.clear();
        state.unique_pairs.clear();
        state.emitted = 0;
        state.submit_sequence.store(0, std::memory_order_relaxed);
    }
    const u64 epoch =
        LateFrameCaptureEpochCounter().fetch_add(1, std::memory_order_acq_rel) + 1;
    LateFrameCaptureFlag().store(true, std::memory_order_release);
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_GPU_FRAME_ARM] epoch=%llu boundary=patched_flip "
                        "capture=next_complete_guest_frame renderWaveBudget=%u",
                        static_cast<unsigned long long>(epoch), State::MaxLogRecords);
    return epoch;
}

inline bool LateFrameCaptureActive() noexcept {
    return LateFrameCaptureFlag().load(std::memory_order_acquire);
}

inline u64 LateFrameCaptureEpoch() noexcept {
    return LateFrameCaptureEpochCounter().load(std::memory_order_acquire);
}

inline void FinishLateFrameCapture() noexcept {
    if (!LateFrameCaptureFlag().exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    auto& state = GetState();
    u32 emitted{};
    {
        std::scoped_lock lock{state.mutex};
        emitted = state.emitted;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_GPU_FRAME_COMPLETE] epoch=%llu renderWaveRecords=%u",
                        static_cast<unsigned long long>(LateFrameCaptureEpoch()), emitted);
}

inline bool ReserveUniqueStage(u8 layer, Stage stage, const ShaderIdentity& shader) {
    auto& state = GetState();
    const u64 identity = shader.found ? shader.hash : shader.address;
    std::scoped_lock lock{state.mutex};
    if (state.emitted >= State::MaxLogRecords ||
        !state.unique_stages.emplace(layer, static_cast<u8>(stage), identity).second) {
        return false;
    }
    ++state.emitted;
    return true;
}

inline bool ReserveUniquePair(u8 layer, u64 vs, u64 ps) {
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    if (state.emitted >= State::MaxLogRecords ||
        !state.unique_pairs.emplace(layer, vs, ps).second) {
        return false;
    }
    ++state.emitted;
    return true;
}

inline bool ReserveRecord() {
    auto& state = GetState();
    std::scoped_lock lock{state.mutex};
    if (state.emitted >= State::MaxLogRecords) {
        return false;
    }
    ++state.emitted;
    return true;
}

inline ShaderIdentity ResolveShader(VAddr address) {
    ShaderIdentity result{.address = address};
    if (!Enabled() || address == 0) {
        return result;
    }

    auto& state = GetState();
    {
        std::scoped_lock lock{state.mutex};
        if (const auto it = state.shader_cache.find(address); it != state.shader_cache.end()) {
            return it->second;
        }
    }

    auto* memory = Core::Memory::Instance();
    std::vector<u8> bytes(State::MaxShaderProbeBytes);
    std::size_t copied = 0;
    if (memory != nullptr) {
        while (copied < bytes.size()) {
            const std::size_t chunk = std::min<std::size_t>(4096, bytes.size() - copied);
            if (!memory->TryCopyReadableMemory(address + copied, bytes.data() + copied, chunk)) {
                break;
            }
            copied += chunk;
        }
    }
    result.readable = copied >= sizeof(u32) * 2;

    const auto read_info = [&](std::size_t offset, BinaryInfo& info) {
        if (offset > copied || sizeof(BinaryInfo) > copied - offset) {
            return false;
        }
        std::memcpy(&info, bytes.data() + offset, sizeof(info));
        return info.Valid() && info.length != 0 && info.length <= 16_MB;
    };

    BinaryInfo info{};
    if (result.readable) {
        u32 token{};
        u32 offset_token{};
        std::memcpy(&token, bytes.data(), sizeof(token));
        std::memcpy(&offset_token, bytes.data() + sizeof(token), sizeof(offset_token));
        if (token == 0xBEEB03FFu) {
            const u64 info_offset = (u64{offset_token} + 1) * 2 * sizeof(u32);
            if (info_offset <= std::numeric_limits<std::size_t>::max() &&
                read_info(static_cast<std::size_t>(info_offset), info)) {
                result.hash = info.shader_hash;
                result.found = true;
            }
        }
    }
    if (!result.found && copied >= sizeof(BinaryInfo)) {
        const std::size_t scan_end =
            std::min<std::size_t>(0x10000, copied - sizeof(BinaryInfo) + 1);
        for (std::size_t offset = 0; offset < scan_end; offset += alignof(u32)) {
            if (std::memcmp(bytes.data() + offset, BinaryInfo::signature_ref.data(),
                            BinaryInfo::signature_ref.size()) == 0 &&
                read_info(offset, info)) {
                result.hash = info.shader_hash;
                result.found = true;
                break;
            }
        }
    }

    {
        std::scoped_lock lock{state.mutex};
        state.shader_cache.try_emplace(address, result);
    }
    return result;
}

inline ShaderIdentity ResolveShader(const ShaderProgram& program) {
    return ResolveShader(reinterpret_cast<VAddr>(program.Address<const u8*>()));
}

inline u64 HashWords(std::span<const u32> words) noexcept {
    Common::ContentFingerprint64 fingerprint{Common::FingerprintDomain::RenderWaveWords};
    const auto hash_span = [&](std::span<const u32> sample) {
        fingerprint.Update(sample.data(), sample.size_bytes());
    };
    constexpr std::size_t SampleWords = 64;
    const std::size_t head = std::min(words.size(), SampleWords);
    hash_span(words.first(head));
    if (words.size() > head) {
        hash_span(words.last(std::min(words.size() - head, SampleWords)));
    }
    fingerprint.UpdateLittleEndian(static_cast<u64>(words.size()));
    return fingerprint.Finish();
}

struct BuilderThreadState {
    uintptr_t cmdbuf{};
    std::array<ShaderIdentity, 6> stages{};
    u64 transition{};
};

inline thread_local BuilderThreadState builder_state{};

inline void BuilderBegin(uintptr_t cmdbuf) {
    if (!Enabled()) {
        return;
    }
    builder_state = {};
    builder_state.cmdbuf = cmdbuf;
}

inline void BuilderShader(Stage stage, uintptr_t cmdbuf, uintptr_t packet, VAddr program_address) {
    if (!Enabled()) {
        return;
    }
    if (builder_state.cmdbuf == 0) {
        builder_state.cmdbuf = cmdbuf != 0 ? cmdbuf : packet;
    }
    const ShaderIdentity shader = ResolveShader(program_address);
    builder_state.stages[static_cast<u8>(stage)] = shader;
    if (!ReserveUniqueStage(0, stage, shader)) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_RENDER_WAVE] layer=builder event=shader stage=%s hash=0x%llx "
                        "program=0x%llx readable=%u found=%u cmdbuf=0x%llx packet=0x%llx",
                        StageName(stage), static_cast<unsigned long long>(shader.hash),
                        static_cast<unsigned long long>(program_address), shader.readable ? 1u : 0u,
                        shader.found ? 1u : 0u,
                        static_cast<unsigned long long>(builder_state.cmdbuf),
                        static_cast<unsigned long long>(packet));
}

inline void BuilderDraw(uintptr_t cmdbuf, uintptr_t packet, const char* api) {
    if (!Enabled()) {
        return;
    }
    if (builder_state.cmdbuf == 0) {
        builder_state.cmdbuf = cmdbuf != 0 ? cmdbuf : packet;
    }
    const u64 vs = builder_state.stages[static_cast<u8>(Stage::Vs)].hash;
    const u64 ps = builder_state.stages[static_cast<u8>(Stage::Ps)].hash;
    ++builder_state.transition;
    if (!ReserveUniquePair(0, vs, ps)) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_RENDER_WAVE] layer=builder event=draw transition=%llu api=%s "
                        "vs=0x%llx ps=0x%llx cmdbuf=0x%llx packet=0x%llx",
                        static_cast<unsigned long long>(builder_state.transition), api ? api : "?",
                        static_cast<unsigned long long>(vs), static_cast<unsigned long long>(ps),
                        static_cast<unsigned long long>(builder_state.cmdbuf),
                        static_cast<unsigned long long>(packet));
}

inline u64 NextSubmitSequence() noexcept {
    return GetState().submit_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
}

inline void Submit(u64 sequence, VAddr root, u64 root_hash, u32 dwords, u32 ccb_dwords) {
    if (!Enabled() || !(sequence <= 4 || std::has_single_bit(sequence)) || !ReserveRecord()) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_RENDER_WAVE] layer=submit event=root submit=%llu root=0x%llx "
                        "rootHash=0x%llx dwords=%u ccbDwords=%u",
                        static_cast<unsigned long long>(sequence),
                        static_cast<unsigned long long>(root),
                        static_cast<unsigned long long>(root_hash), dwords, ccb_dwords);
}

inline void Pm4Shader(Stage stage, const ShaderIdentity& shader, u64 submit, VAddr root,
                      u64 root_hash, VAddr stream, u32 depth, u64 semantic_sequence) {
    if (!Enabled() || !ReserveUniqueStage(1, stage, shader)) {
        return;
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_RENDER_WAVE] layer=pm4 event=shader stage=%s hash=0x%llx "
        "program=0x%llx readable=%u found=%u submit=%llu root=0x%llx rootHash=0x%llx "
        "stream=0x%llx depth=%u semantic=%llu",
        StageName(stage), static_cast<unsigned long long>(shader.hash),
        static_cast<unsigned long long>(shader.address), shader.readable ? 1u : 0u,
        shader.found ? 1u : 0u, static_cast<unsigned long long>(submit),
        static_cast<unsigned long long>(root), static_cast<unsigned long long>(root_hash),
        static_cast<unsigned long long>(stream), depth,
        static_cast<unsigned long long>(semantic_sequence));
}

inline void Pm4Draw(const Regs& regs, u64 submit, VAddr root, u64 root_hash, VAddr stream,
                    u32 depth, u32 opcode, u64 semantic_sequence) {
    if (!Enabled()) {
        return;
    }
    const ShaderIdentity vs_shader = ResolveShader(regs.vs_program);
    const ShaderIdentity ps_shader = ResolveShader(regs.ps_program);
    const u64 vs = vs_shader.hash;
    const u64 ps = ps_shader.hash;
    if (!ReserveUniquePair(1, vs, ps)) {
        return;
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_RENDER_WAVE] layer=pm4 event=draw submit=%llu semantic=%llu "
        "opcode=0x%x vs=0x%llx ps=0x%llx root=0x%llx rootHash=0x%llx "
        "stream=0x%llx depth=%u",
        static_cast<unsigned long long>(submit), static_cast<unsigned long long>(semantic_sequence),
        opcode, static_cast<unsigned long long>(vs), static_cast<unsigned long long>(ps),
        static_cast<unsigned long long>(root), static_cast<unsigned long long>(root_hash),
        static_cast<unsigned long long>(stream), depth);
}

inline void RasterDraw(const Regs& regs, u64 pipeline_vs, u64 pipeline_ps, u64 pipeline,
                       bool indexed, bool indirect) {
    if (!Enabled()) {
        return;
    }
    const ShaderIdentity vs_shader = ResolveShader(regs.vs_program);
    const ShaderIdentity ps_shader = ResolveShader(regs.ps_program);
    const u64 vs = vs_shader.hash;
    const u64 ps = ps_shader.hash;
    if (!ReserveUniquePair(2, vs, ps)) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_RENDER_WAVE] layer=raster event=draw vs=0x%llx ps=0x%llx "
                        "pipelineVs=0x%llx pipelinePs=0x%llx pipeline=0x%llx "
                        "indexed=%u indirect=%u",
                        static_cast<unsigned long long>(vs), static_cast<unsigned long long>(ps),
                        static_cast<unsigned long long>(pipeline_vs),
                        static_cast<unsigned long long>(pipeline_ps),
                        static_cast<unsigned long long>(pipeline), indexed ? 1u : 0u,
                        indirect ? 1u : 0u);
}

}

#endif
