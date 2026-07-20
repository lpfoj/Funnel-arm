// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#ifdef __ANDROID__
#include <android/log.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" int executor_lsx4_android_install_mono_explicit_sigsegv_guard()
    __attribute__((weak));
extern "C" void executor_live_dump_posix_sem_records(const char* reason) __attribute__((weak));
#endif

#include "executor/gnmcap_format.h"
#include "gnm_error.h"
#include "gnmdriver.h"

#include "common/assert.h"
#include "common/config.h"
#include "common/content_fingerprint.h"
#include "common/debug.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/slot_vector.h"
#include "core/address_space.h"
#include "core/debug_state.h"
#include "core/signals.h"
#include "core/libraries/gnmdriver/gnm_error.h"
#include "core/libraries/gnmdriver/gnmdriver_init.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/libs.h"
#include "core/libraries/videoout/video_out.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/render_wave_trace.h"
#include "video_core/renderer_vulkan/vk_presenter.h"

extern Frontend::WindowSDL* g_window;
std::unique_ptr<Vulkan::Presenter> presenter;
std::unique_ptr<AmdGpu::Liverpool> liverpool;

#ifdef __ANDROID__
extern "C" std::uint64_t executor_imt_overflow_count() __attribute__((weak));
extern "C" std::uint64_t executor_imt_alloc_overflow_count() __attribute__((weak));
extern "C" std::uint64_t executor_imt_tail_overflow_count() __attribute__((weak));
extern "C" void executor_mono_vtnull_dump() __attribute__((weak));
extern "C" void executor_mono_vcall_entry_dump() __attribute__((weak));
extern "C" void executor_lsx4_register_guest_gc_roots() __attribute__((weak));
extern "C" void executor_mono_value_copy_dump() __attribute__((weak));
extern "C" void executor_lsx4_android_dump_box64_emu_states(const char* reason)
    __attribute__((weak));
extern "C" void executor_live_hle_flight_dump(const char* reason, u64 builder_total,
                                              u64 builder_draw, u64 builder_shader,
                                              u64 active_cb, u64 active_dw)
    __attribute__((weak));
extern "C" void executor_jit_dump_thread_states(const char* reason) __attribute__((weak));
extern "C" void executor_live_dump_unity_thread_rips(const char* reason, int pulse);
extern "C" void executor_live_dump_kernel_sem_records(const char* reason) __attribute__((weak));
extern "C" void executor_live_dump_mutex_wait_ledger(const char* reason, int pulse)
    __attribute__((weak));
extern "C" int executor_lsx4_android_runtime_jit_active() __attribute__((weak));

static bool ExecutorTraceLiveWide() {
    // FPS: this gates ~20 per-submit/per-DCB GPU traces (DCB_SET_SH/CTX, GNM_BUILDER/DWORDS/STAGE,
    // DCB_DMA ~ 100k+ synchronous logcat lines/run) that fired under the broad EXECUTOR_TRACE_LIVE_WIDE
    // and dominate the render-path time. Gate behind a dedicated opt-in so ordinary widecheck runs do
    // not pay for them; set EXECUTOR_TRACE_LIVE_WIDE_HOT=1 to restore the old verbose behaviour.
    static const bool enabled = std::getenv("EXECUTOR_TRACE_LIVE_WIDE_HOT") != nullptr;
    return enabled;
}

static bool ExecutorLiveShouldPulse(u64 value) {
    return value <= 4 || (value != 0 && (value & (value - 1)) == 0);
}

static std::atomic<u64> g_live_gnm_builder_total{0};
static std::atomic<u64> g_live_gnm_builder_draw{0};
static std::atomic<u64> g_live_gnm_builder_shader{0};
// Draw helpers are not a draw counter. Unity writes DRAW_* PM4 packets directly into its DCB, so
// g_live_gnm_builder_draw legitimately remains zero while Liverpool executes real Vulkan draws.
// Keep a separate post-vkCmdDraw counter for the on-screen HUD.
static std::atomic<u64> g_live_actual_draws{0};
static std::atomic<u64> g_live_gnm_builder_submitdone{0};
static std::atomic<uintptr_t> g_live_active_cmdbuf{0};
static std::atomic<u32> g_live_active_cmdbuf_size{0};
static std::atomic<uintptr_t> g_live_active_cmdbuf_end{0};
static std::atomic<u64> g_live_autosubmitted_hash{0};
static std::atomic<bool> g_live_post_shader_watchdog_started{false};
static std::atomic<u64> g_live_submit_calls{0};
static std::atomic<u64> g_live_submit_done_calls{0};
static std::atomic<u64> g_live_present_rt_requests{0};

struct ExecutorLiveSubmittedDcb {
    uintptr_t begin{};
    u32 valid_dwords{};
    u64 hash{};
};
static std::mutex g_live_submitted_dcb_mutex;
static std::array<ExecutorLiveSubmittedDcb, 16> g_live_submitted_dcbs{};
static std::size_t g_live_submitted_dcb_cursor{};

static void ExecutorLiveFrameStats(const char* event, u64 frames) {
    using Clock = std::chrono::steady_clock;
    static std::mutex stats_mutex;
    static auto start = Clock::now();
    static auto last = start;
    static u64 last_submits = 0;
    static u64 last_done = 0;
    static u64 last_present = 0;

    const auto now = Clock::now();
    std::scoped_lock lock{stats_mutex};
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
    if (elapsed_ms.count() < 1000) {
        return;
    }

    const auto submits = g_live_submit_calls.load(std::memory_order_relaxed);
    const auto done = g_live_submit_done_calls.load(std::memory_order_relaxed);
    const auto present = g_live_present_rt_requests.load(std::memory_order_relaxed);
    const double window_sec = static_cast<double>(elapsed_ms.count()) / 1000.0;
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();

    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_FPS] source=gnm event=%s elapsedMs=%lld frames=%llu "
        "submitRate=%.2f submitDoneRate=%.2f presentReqRate=%.2f submits=%llu "
        "submitDone=%llu presentReq=%llu builderTotal=%llu builderDraw=%llu builderShader=%llu",
        event ? event : "unknown", static_cast<long long>(total_ms),
        static_cast<unsigned long long>(frames),
        static_cast<double>(submits - last_submits) / window_sec,
        static_cast<double>(done - last_done) / window_sec,
        static_cast<double>(present - last_present) / window_sec,
        static_cast<unsigned long long>(submits), static_cast<unsigned long long>(done),
        static_cast<unsigned long long>(present),
        static_cast<unsigned long long>(g_live_gnm_builder_total.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_live_gnm_builder_draw.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(g_live_gnm_builder_shader.load(std::memory_order_relaxed)));

    last = now;
    last_submits = submits;
    last_done = done;
    last_present = present;
}

static bool ExecutorLiveGnmApiIsActualDraw(const char* api) {
    return api &&
           (std::strcmp(api, "sceGnmDrawIndex") == 0 ||
            std::strcmp(api, "sceGnmDrawIndexAuto") == 0 ||
            std::strcmp(api, "sceGnmDrawIndexIndirect") == 0 ||
            std::strcmp(api, "sceGnmDrawIndexIndirectCountMulti") == 0 ||
            std::strcmp(api, "sceGnmDrawIndexIndirectMulti") == 0 ||
            std::strcmp(api, "sceGnmDrawIndexMultiInstanced") == 0 ||
            std::strcmp(api, "sceGnmDrawIndexOffset") == 0 ||
            std::strcmp(api, "sceGnmDrawIndirect") == 0 ||
            std::strcmp(api, "sceGnmDrawIndirectCountMulti") == 0 ||
            std::strcmp(api, "sceGnmDrawIndirectMulti") == 0 ||
            std::strcmp(api, "sceGnmDrawOpaqueAuto") == 0);
}

static void ExecutorLiveClassifyDcb(const char* tag, const u32* dcb, u32 dwords);

static void ExecutorLiveMaybeStartPostShaderWatchdog(const char* api, const char* result,
                                                     const bool is_shader) {
    if (!api || !result || std::strcmp(result, "OK") != 0 || !is_shader) {
        return;
    }
    static const bool watchdog_enabled = std::getenv("EXECUTOR_LIVE_HLE_FLIGHT") != nullptr ||
                                         std::getenv("EXECUTOR_LIGHT_ORACLE") != nullptr;
    if (!watchdog_enabled ||
        g_live_post_shader_watchdog_started.load(std::memory_order_relaxed)) {
        return;
    }
    bool expected = false;
    if (!g_live_post_shader_watchdog_started.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return;
    }
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        const auto total = g_live_gnm_builder_total.load(std::memory_order_relaxed);
        const auto draw = g_live_gnm_builder_draw.load(std::memory_order_relaxed);
        const auto shader = g_live_gnm_builder_shader.load(std::memory_order_relaxed);
        const auto active_cb = g_live_active_cmdbuf.load(std::memory_order_relaxed);
        const auto active_dw = g_live_active_cmdbuf_size.load(std::memory_order_relaxed);
        if (shader >= 2 && draw == 0) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_STALL] reason=post_shader_no_draw "
                                "builderTotal=%llu builderDraw=%llu builderShader=%llu "
                                "activeCb=%p activeDw=%llu",
                                static_cast<unsigned long long>(total),
                                static_cast<unsigned long long>(draw),
                                static_cast<unsigned long long>(shader),
                                reinterpret_cast<void*>(active_cb),
                                static_cast<unsigned long long>(active_dw));
            if (active_cb && active_dw != 0) {
                ExecutorLiveClassifyDcb("post_shader_no_draw_active_cb",
                                        reinterpret_cast<const u32*>(active_cb),
                                        static_cast<u32>(std::min<u64>(active_dw, 4096)));
            }
            if (executor_live_hle_flight_dump) {
                executor_live_hle_flight_dump("post_shader_no_draw", total, draw, shader,
                                              active_cb, active_dw);
            } else {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                    "[EXECUTOR_LIVE_HLE_FLIGHT_DUMP] unavailable=1");
            }
            if (executor_jit_dump_thread_states) {
                executor_jit_dump_thread_states("post-shader-no-draw");
            }
            if (executor_lsx4_android_dump_box64_emu_states) {
                executor_lsx4_android_dump_box64_emu_states("post-shader-no-draw");
            }
        }
    }).detach();
}

enum class ExecutorLiveGnmBuilderKind : u8 {
    Auto,
    Shader,
};

static uintptr_t ExecutorRenderWaveCmdbufIdentity(const u32* packet) {
    const auto address = reinterpret_cast<uintptr_t>(packet);
    const auto active = g_live_active_cmdbuf.load(std::memory_order_relaxed);
    // GNM helpers receive a packet-local cursor, not the allocation base. DrawInit gives us the
    // base; keep using it while the cursor is within a deliberately generous command-buffer bound.
    if (active != 0 && address >= active && address - active < 16_MB) {
        return active;
    }
    return address;
}

static void ExecutorLiveGnmBuilderLog(const char* api, const char* result, const u32* cmdbuf,
                                      u32 size, const char* detail = "",
                                      const ExecutorLiveGnmBuilderKind kind =
                                          ExecutorLiveGnmBuilderKind::Auto) {
    if (api) {
        const bool is_shader = kind == ExecutorLiveGnmBuilderKind::Shader ||
                               (kind == ExecutorLiveGnmBuilderKind::Auto &&
                                std::strstr(api, "Shader") != nullptr);
        const bool is_draw =
            kind == ExecutorLiveGnmBuilderKind::Auto && ExecutorLiveGnmApiIsActualDraw(api);
        const bool is_submit =
            kind == ExecutorLiveGnmBuilderKind::Auto &&
            (std::strstr(api, "SubmitDone") || std::strstr(api, "Submit"));

        // These counters feed a one-Hz HUD; doing four contended atomic RMWs for every GNM helper
        // is pure observer overhead. Unity/Bloodborne issues thousands of builder calls per second
        // from several guest workers. Accumulate per host thread and publish in bounded batches;
        // the displayed value may trail by at most 255 calls per active thread while the command
        // buffer tracking below remains immediate and exact.
        struct BuilderCounterBatch {
            u32 total{};
            u32 draw{};
            u32 shader{};
            u32 submit{};
        };
        static thread_local BuilderCounterBatch batch;
        ++batch.total;
        batch.draw += is_draw ? 1u : 0u;
        batch.shader += is_shader ? 1u : 0u;
        batch.submit += is_submit ? 1u : 0u;
        if (batch.total >= 256u) {
            g_live_gnm_builder_total.fetch_add(batch.total, std::memory_order_relaxed);
            if (batch.draw != 0) {
                g_live_gnm_builder_draw.fetch_add(batch.draw, std::memory_order_relaxed);
            }
            if (batch.shader != 0) {
                g_live_gnm_builder_shader.fetch_add(batch.shader, std::memory_order_relaxed);
            }
            if (batch.submit != 0) {
                g_live_gnm_builder_submitdone.fetch_add(batch.submit, std::memory_order_relaxed);
            }
            batch = {};
        }
        if (cmdbuf && result && std::strcmp(result, "OK") == 0) {
            const auto begin = reinterpret_cast<uintptr_t>(cmdbuf);
            const auto end = begin + static_cast<uintptr_t>(size) * sizeof(u32);
            if (std::strstr(api, "DrawInit") != nullptr) {
                g_live_active_cmdbuf.store(begin, std::memory_order_relaxed);
                g_live_active_cmdbuf_end.store(end, std::memory_order_relaxed);
                g_live_active_cmdbuf_size.store(size, std::memory_order_relaxed);
#ifdef __ANDROID__
                AmdGpu::RenderWaveTrace::BuilderBegin(begin);
#endif
            } else {
                const auto base = g_live_active_cmdbuf.load(std::memory_order_relaxed);
                // Track the high-water mark inside the same builder command buffer. Several GNM
                // helpers receive packet-local pointers, while DrawInit receives the buffer base.
                // Autosubmit/classification must use the built range, not the init packet size.
                if (base != 0 && begin >= base && begin - base < (1u << 20)) {
                    auto previous_end = g_live_active_cmdbuf_end.load(std::memory_order_relaxed);
                    while (end > previous_end &&
                           !g_live_active_cmdbuf_end.compare_exchange_weak(
                               previous_end, end, std::memory_order_relaxed)) {
                    }
                    const auto tracked_end = g_live_active_cmdbuf_end.load(std::memory_order_relaxed);
                    if (tracked_end > base) {
                        g_live_active_cmdbuf_size.store(
                            static_cast<u32>((tracked_end - base) / sizeof(u32)),
                            std::memory_order_relaxed);
                    }
                }
            }
#ifdef __ANDROID__
            if (kind == ExecutorLiveGnmBuilderKind::Auto &&
                ExecutorLiveGnmApiIsActualDraw(api)) {
                AmdGpu::RenderWaveTrace::BuilderDraw(ExecutorRenderWaveCmdbufIdentity(cmdbuf),
                                                      begin, api);
            }
#endif
        }
        ExecutorLiveMaybeStartPostShaderWatchdog(api, result, is_shader);
    }
    static std::atomic<int> first_builder_log_budget{8};
    int first_log_slot = first_builder_log_budget.load(std::memory_order_relaxed);
    while (first_log_slot > 0 &&
           !first_builder_log_budget.compare_exchange_weak(
               first_log_slot, first_log_slot - 1, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    if (first_log_slot > 0) {
        char thread_name[32]{};
        pthread_getname_np(pthread_self(), thread_name, sizeof(thread_name));
        const auto active = g_live_active_cmdbuf.load(std::memory_order_relaxed);
        const auto active_end = g_live_active_cmdbuf_end.load(std::memory_order_relaxed);
        const auto begin = reinterpret_cast<uintptr_t>(cmdbuf);
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_BUILDER_FIRST] slot=%d api=%s result=%s "
                            "cmdbuf=0x%llx size=%u activeCb=0x%llx activeEnd=0x%llx "
                            "activeSize=%u inActive=%d thread=%s total=%llu draw=%llu "
                            "shader=%llu detail=%s",
                            9 - first_log_slot, api ? api : "<null>",
                            result ? result : "<null>",
                            static_cast<unsigned long long>(begin), size,
                            static_cast<unsigned long long>(active),
                            static_cast<unsigned long long>(active_end),
                            g_live_active_cmdbuf_size.load(std::memory_order_relaxed),
                            active != 0 && begin >= active && begin < active_end ? 1 : 0,
                            thread_name[0] ? thread_name : "<unnamed>",
                            static_cast<unsigned long long>(
                                g_live_gnm_builder_total.load(std::memory_order_relaxed)),
                            static_cast<unsigned long long>(
                                g_live_gnm_builder_draw.load(std::memory_order_relaxed)),
                            static_cast<unsigned long long>(
                                g_live_gnm_builder_shader.load(std::memory_order_relaxed)),
                            detail ? detail : "");
    }
    if (!ExecutorTraceLiveWide()) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_GNM_BUILDER] api=%s result=%s cmdbuf=%p size=%u %s",
                        api ? api : "<null>", result ? result : "<null>", cmdbuf, size,
                        detail ? detail : "");
}

static void ExecutorLiveGnmUnknownLog(const char* name) {
    char thread_name[32]{};
    pthread_getname_np(pthread_self(), thread_name, sizeof(thread_name));
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_LIVE_GNM_UNKNOWN] api=%s thread=%s self=0x%llx "
                        "builderTotal=%llu builderDraw=%llu activeCb=0x%llx activeSize=%u",
                        name ? name : "<null>", thread_name[0] ? thread_name : "<unnamed>",
                        static_cast<unsigned long long>(
                            static_cast<uintptr_t>(pthread_self())),
                        static_cast<unsigned long long>(
                            g_live_gnm_builder_total.load(std::memory_order_relaxed)),
                        static_cast<unsigned long long>(
                            g_live_gnm_builder_draw.load(std::memory_order_relaxed)),
                        static_cast<unsigned long long>(
                            g_live_active_cmdbuf.load(std::memory_order_relaxed)),
                        g_live_active_cmdbuf_size.load(std::memory_order_relaxed));
}

static void ExecutorLiveDumpDwords(const char* tag, const u32* data, u32 dwords) {
    if (!ExecutorTraceLiveWide()) {
        return;
    }
    if (!data || dwords == 0) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_DWORDS] tag=%s addr=%p dwords=%u empty=1",
                            tag ? tag : "<null>", data, dwords);
        return;
    }
    const u32 limit = std::min<u32>(dwords, 16u);
    char line[320]{};
    int used = std::snprintf(line, sizeof(line), "tag=%s addr=%p dwords=%u sample=",
                             tag ? tag : "<null>", data, dwords);
    for (u32 i = 0; i < limit && used > 0 && used < static_cast<int>(sizeof(line)); ++i) {
        used += std::snprintf(line + used, sizeof(line) - used, "%s%08x", i ? "," : "",
                              data[i]);
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_GNM_DWORDS] %s", line);
}

static void ExecutorLiveDumpBuilderTail(const char* api, const u32* cmdbuf, u32 size) {
    if (!ExecutorTraceLiveWide() || !cmdbuf || size == 0) {
        return;
    }
    ExecutorLiveDumpDwords(api, cmdbuf, std::min<u32>(size, 16u));
}

static void ExecutorLiveClassifyDcb(const char* tag, const u32* dcb, u32 dwords) {
    if (!ExecutorTraceLiveWide()) {
        return;
    }
    ExecutorLiveDumpDwords(tag, dcb, std::min<u32>(dwords, 16u));
    if (!dcb || dwords == 0) {
        return;
    }

    u32 packets = 0;
    u32 draws = 0;
    u32 dispatches = 0;
    u32 waits = 0;
    u32 set_sh = 0;
    u32 set_context = 0;
    u32 set_uconfig = 0;
    u32 dma = 0;
    u32 writes = 0;
    u32 indirect = 0;
    u32 set_sh_detail = 0;
    u32 set_ctx_detail = 0;
    u32 invalid = 0;
    u32 first_op = 0xff;
    u32 last_op = 0xff;
    u32 first_draw_off = 0xffffffffu;
    u32 first_setsh_off = 0xffffffffu;
    u32 off = 0;
    u32 opcode_counts[256]{};

    while (off < dwords && packets < 4096) {
        const auto* header = reinterpret_cast<const AmdGpu::PM4Header*>(dcb + off);
        if (header->type.Value() == 2) {
            ++packets;
            ++off;
            continue;
        }
        if (header->type.Value() != 3) {
            ++invalid;
            break;
        }
        const u32 body_words = header->type3.NumWords();
        const u32 packet_words = body_words + 1;
        if (packet_words == 0 || off + packet_words > dwords) {
            ++invalid;
            break;
        }
        const auto opcode = header->type3.opcode.Value();
        opcode_counts[static_cast<u32>(opcode) & 0xffu]++;
        if (first_op == 0xff) {
            first_op = static_cast<u32>(opcode);
        }
        last_op = static_cast<u32>(opcode);
        ++packets;
        switch (opcode) {
        case AmdGpu::PM4ItOpcode::DrawIndex2:
        case AmdGpu::PM4ItOpcode::DrawIndexOffset2:
        case AmdGpu::PM4ItOpcode::DrawIndexAuto:
        case AmdGpu::PM4ItOpcode::DrawIndirect:
        case AmdGpu::PM4ItOpcode::DrawIndirectMulti:
        case AmdGpu::PM4ItOpcode::DrawIndexIndirect:
        case AmdGpu::PM4ItOpcode::DrawIndexIndirectMulti:
        case AmdGpu::PM4ItOpcode::DrawIndexIndirectCountMulti:
            if (first_draw_off == 0xffffffffu) {
                first_draw_off = off;
            }
            ++draws;
            break;
        case AmdGpu::PM4ItOpcode::DispatchDirect:
        case AmdGpu::PM4ItOpcode::DispatchIndirect:
            ++dispatches;
            break;
        case AmdGpu::PM4ItOpcode::WaitRegMem:
        case AmdGpu::PM4ItOpcode::WaitOnCeCounter:
            ++waits;
            break;
        case AmdGpu::PM4ItOpcode::SetShReg:
            if (first_setsh_off == 0xffffffffu) {
                first_setsh_off = off;
            }
            ++set_sh;
            if (set_sh_detail++ < 16) {
                const auto* set_data = reinterpret_cast<const AmdGpu::PM4CmdSetData*>(&dcb[off]);
                const u32 value_words = body_words > 0 ? body_words - 1 : 0;
                const u32* values = dcb + off + 2;
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_DCB_SET_SH] tag=%s n=%u off=%u reg=0x%x words=%u "
                    "v=%08x %08x %08x %08x %08x %08x %08x %08x",
                    tag ? tag : "<null>", set_sh, off, set_data->reg_offset.Value(), value_words,
                    value_words > 0 ? values[0] : 0u, value_words > 1 ? values[1] : 0u,
                    value_words > 2 ? values[2] : 0u, value_words > 3 ? values[3] : 0u,
                    value_words > 4 ? values[4] : 0u, value_words > 5 ? values[5] : 0u,
                    value_words > 6 ? values[6] : 0u, value_words > 7 ? values[7] : 0u);
            }
            break;
        case AmdGpu::PM4ItOpcode::SetContextReg:
            ++set_context;
            if (set_ctx_detail++ < 12) {
                const auto* set_data = reinterpret_cast<const AmdGpu::PM4CmdSetData*>(&dcb[off]);
                const u32 value_words = body_words > 0 ? body_words - 1 : 0;
                const u32* values = dcb + off + 2;
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_DCB_SET_CTX] tag=%s n=%u off=%u reg=0x%x words=%u "
                    "v=%08x %08x %08x %08x",
                    tag ? tag : "<null>", set_context, off, set_data->reg_offset.Value(),
                    value_words, value_words > 0 ? values[0] : 0u,
                    value_words > 1 ? values[1] : 0u, value_words > 2 ? values[2] : 0u,
                    value_words > 3 ? values[3] : 0u);
            }
            break;
        case AmdGpu::PM4ItOpcode::SetUconfigReg:
            ++set_uconfig;
            break;
        case AmdGpu::PM4ItOpcode::DmaData:
            ++dma;
            if (dma <= 8) {
                const auto* dma_data = reinterpret_cast<const AmdGpu::PM4DmaData*>(&dcb[off]);
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_DCB_DMA] tag=%s n=%u off=%u srcSel=%u dstSel=%u "
                    "src=0x%llx dst=0x%llx bytes=%u data=%08x command=%08x",
                    tag ? tag : "<null>", dma, off, u32(dma_data->src_sel.Value()),
                    u32(dma_data->dst_sel.Value()),
                    static_cast<unsigned long long>(
                        dma_data->src_sel == AmdGpu::DmaDataSrc::Data
                            ? 0ull
                            : dma_data->SrcAddress<VAddr>()),
                    static_cast<unsigned long long>(dma_data->DstAddress<VAddr>()),
                    dma_data->NumBytes(), dma_data->data, dma_data->command);
            }
            break;
        case AmdGpu::PM4ItOpcode::WriteData:
            ++writes;
            break;
        case AmdGpu::PM4ItOpcode::IndirectBuffer:
        case AmdGpu::PM4ItOpcode::IndirectBufferConst:
            ++indirect;
            break;
        default:
            break;
        }
        off += packet_words;
    }

    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_DCB_CLASSIFY] tag=%s dwords=%u packets=%u draws=%u dispatches=%u "
        "waits=%u setSh=%u setCtx=%u setUcfg=%u dma=%u writes=%u indirect=%u invalid=%u "
        "firstOp=0x%02x lastOp=0x%02x firstDrawOff=%u firstSetShOff=%u",
        tag ? tag : "<null>", dwords, packets, draws, dispatches, waits, set_sh, set_context,
        set_uconfig, dma, writes, indirect, invalid, first_op, last_op, first_draw_off,
        first_setsh_off);

    char hist[512]{};
    int used = 0;
    for (u32 pass = 0; pass < 12; ++pass) {
        u32 best_op = 0;
        u32 best_count = 0;
        for (u32 op = 0; op < 256; ++op) {
            if (opcode_counts[op] > best_count) {
                best_op = op;
                best_count = opcode_counts[op];
            }
        }
        if (best_count == 0 || used >= static_cast<int>(sizeof(hist)) - 32) {
            break;
        }
        used += std::snprintf(hist + used, sizeof(hist) - used, "%s0x%02x:%u",
                              used ? "," : "", best_op, best_count);
        opcode_counts[best_op] = 0;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_DCB_OPCODE_HIST] tag=%s top=%s",
                        tag ? tag : "<null>", hist);
}

struct ExecutorLiveDcbAnalysis {
    u32 valid_dwords = 0;
    u32 draws = 0;
    u32 packets = 0;
    u64 hash = 0;
};

static ExecutorLiveDcbAnalysis ExecutorLiveAnalyzeDcb(const u32* dcb, u32 max_dwords) {
    ExecutorLiveDcbAnalysis result{};
    if (!dcb || max_dwords == 0) {
        return result;
    }
    Common::ContentFingerprint64 fingerprint{Common::FingerprintDomain::CommandStream};
    u32 off = 0;
    while (off < max_dwords && result.packets < 4096) {
        const auto* header = reinterpret_cast<const AmdGpu::PM4Header*>(dcb + off);
        u32 packet_words = 0;
        if (header->type.Value() == 2) {
            packet_words = 1;
        } else if (header->type.Value() == 3) {
            const u32 body_words = header->type3.NumWords();
            packet_words = body_words + 1;
            if (packet_words == 0 || off + packet_words > max_dwords) {
                break;
            }
            switch (header->type3.opcode.Value()) {
            case AmdGpu::PM4ItOpcode::DrawIndex2:
            case AmdGpu::PM4ItOpcode::DrawIndexOffset2:
            case AmdGpu::PM4ItOpcode::DrawIndexAuto:
            case AmdGpu::PM4ItOpcode::DrawIndirect:
            case AmdGpu::PM4ItOpcode::DrawIndirectMulti:
            case AmdGpu::PM4ItOpcode::DrawIndexIndirect:
            case AmdGpu::PM4ItOpcode::DrawIndexIndirectMulti:
            case AmdGpu::PM4ItOpcode::DrawIndexIndirectCountMulti:
                ++result.draws;
                break;
            default:
                break;
            }
        } else {
            break;
        }
        for (u32 i = 0; i < packet_words; ++i) {
            fingerprint.UpdateLittleEndian(dcb[off + i]);
        }
        off += packet_words;
        result.valid_dwords = off;
        ++result.packets;
    }
    result.hash = fingerprint.Finish();
    return result;
}

static void ExecutorLiveRememberExplicitDcb(const u32* dcb,
                                            const ExecutorLiveDcbAnalysis& analysis) {
    if (!dcb || analysis.valid_dwords == 0) {
        return;
    }
    std::scoped_lock lock{g_live_submitted_dcb_mutex};
    auto& record = g_live_submitted_dcbs[g_live_submitted_dcb_cursor++ %
                                         g_live_submitted_dcbs.size()];
    record.begin = reinterpret_cast<uintptr_t>(dcb);
    record.valid_dwords = analysis.valid_dwords;
    record.hash = analysis.hash;
}

static bool ExecutorLiveWasExplicitlySubmitted(const u32* dcb,
                                               const ExecutorLiveDcbAnalysis& analysis) {
    if (!dcb || analysis.valid_dwords == 0) {
        return false;
    }
    const auto begin = reinterpret_cast<uintptr_t>(dcb);
    std::scoped_lock lock{g_live_submitted_dcb_mutex};
    return std::ranges::any_of(g_live_submitted_dcbs, [&](const auto& record) {
        return record.begin == begin && record.valid_dwords == analysis.valid_dwords &&
               record.hash == analysis.hash;
    });
}
#endif

namespace {
std::mutex g_presenter_init_mutex;

// Lazily create the Vulkan presenter on the first real GNM submit. On Android the presenter is
// intentionally NOT created eagerly (its swapchain ctor would steal the ANativeWindow from Piglet/
// GLES, which 2D homebrew like Itemzflow/Store render through). GNM games issue GPU work, so we
// create the presenter here at the HLE submit boundary -- NOT on Liverpool's GPU thread, which
// would risk WSI init racing GPU queue processing / deadlocks. Piglet-only guests never reach this.
// Design per Codex layer3-lazy-gnm-presenter-design (2026-05-31).
// FOLLOW-UP (needs a real arm64 device to validate): the one-way Piglet->Vulkan surface-owner
// handoff (eglMakeCurrent(NO_SURFACE)+eglDestroySurface) and full g_android_wsi_mutex coordination
// with RefreshAndroidVulkanWindowLocked(). A pure GNM guest has no Piglet EGL surface, so no detach
// is required for the first GNM-only bring-up; mixed Piglet+GNM needs the owner switch.
bool EnsureGnmPresenter(const char* reason) {
    if (presenter) {
#ifdef __ANDROID__
        if (ExecutorTraceLiveWide()) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_PRESENTER] result=already reason=%s",
                                reason ? reason : "<null>");
        }
#endif
        return true;
    }
    std::scoped_lock init_lock{g_presenter_init_mutex};
    if (presenter) {
#ifdef __ANDROID__
        if (ExecutorTraceLiveWide()) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_PRESENTER] result=already_after_lock reason=%s",
                                reason ? reason : "<null>");
        }
#endif
        return true;
    }
    if (!g_window) {
        LOG_ERROR(Lib_GnmDriver, "EXECUTOR_GNM_PRESENTER phase=ensure result=no_window reason={}",
                  reason);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_PRESENTER] result=no_window reason=%s",
                            reason ? reason : "<null>");
#endif
        return false;
    }
    try {
        presenter = std::make_unique<Vulkan::Presenter>(*g_window, liverpool.get());
        // Replay VideoOut buffers registered before the presenter existed (Codex review #1): a GNM
        // game calls sceVideoOutRegisterBuffers during init, before its first GNM submit, when the
        // presenter is still null -- without this the presenter has no display surfaces (black screen).
        Libraries::VideoOut::ReplayBufferRegistrationsToPresenter();
        LOG_INFO(Lib_GnmDriver, "EXECUTOR_GNM_PRESENTER phase=ensure result=OK reason={}", reason);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_PRESENTER] result=OK reason=%s",
                            reason ? reason : "<null>");
#endif
        return true;
    } catch (const std::exception& e) {
        presenter.reset();
        LOG_ERROR(Lib_GnmDriver,
                  "EXECUTOR_GNM_PRESENTER phase=ensure result=failed reason={} error={}", reason,
                  e.what());
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_PRESENTER] result=failed reason=%s error=%s",
                            reason ? reason : "<null>", e.what());
#endif
        return false;
    } catch (...) {
        presenter.reset();
        LOG_ERROR(Lib_GnmDriver,
                  "EXECUTOR_GNM_PRESENTER phase=ensure result=failed reason={} error=unknown",
                  reason);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_PRESENTER] result=failed reason=%s error=unknown",
                            reason ? reason : "<null>");
#endif
        return false;
    }
}
} // namespace

namespace Libraries::GnmDriver {

using namespace AmdGpu;

enum GnmEventType : u64 {
    Compute0RelMem = 0x00,
    Compute1RelMem = 0x01,
    Compute2RelMem = 0x02,
    Compute3RelMem = 0x03,
    Compute4RelMem = 0x04,
    Compute5RelMem = 0x05,
    Compute6RelMem = 0x06,
    GfxEop = 0x40
};

enum ShaderStages : u32 {
    Cs,
    Ps,
    Vs,
    Gs,
    Es,
    Hs,
    Ls,

    Max
};

static constexpr std::array indirect_sgpr_offsets{0u, 0u, 0x4cu, 0u, 0xccu, 0u, 0x14cu};

// Gates use of what appear to be the neo-mode init sequences but with the older
// IA_MULTI_VGT_PARAM register address. No idea what this is for as the ioctl
// that controls it is still a mystery, but leaving the sequences in gated behind
// this flag in case we need it in the future.
static constexpr bool UseNeoCompatSequences = false;

// In case if `submitDone` is issued we need to block submissions until GPU idle
static u32 submission_lock{};
std::condition_variable cv_lock{};
std::mutex m_submission{};
static u64 frames_submitted{};      // frame counter
static bool send_init_packet{true}; // initialize HW state before first game's submit in a frame
static s32 sdk_version{0};

static u32 asc_next_offs_dw[Liverpool::NumComputeRings];

// This address is initialized in sceGnmGetTheTessellationFactorRingBufferBaseAddress
static VAddr tessellation_factors_ring_addr = -1;
static constexpr u32 tessellation_offchip_buffer_size = 0x800000u;

static void ResetSubmissionLock(Platform::InterruptId irq) {
    std::unique_lock lock{m_submission};
    submission_lock = 0;
    cv_lock.notify_all();
}

static bool WaitGpuIdle(const char* where = "gnm_wait_gpu_idle") {
    HLE_TRACE;
    std::unique_lock lock{m_submission};
#ifdef __ANDROID__
    (void)where;
    constexpr auto WaitSlice = std::chrono::milliseconds{250};
    while (submission_lock != 0) {
        if (liverpool && liverpool->IsRendererTerminal()) {
            submission_lock = 0;
            cv_lock.notify_all();
            return false;
        }
        // Match the PC contract: submission_lock is released by the real GpuIdle IRQ.  Pipeline
        // compilation and first-use shader work can legitimately take longer than five seconds on
        // Android; treating that latency as a renderer failure permanently dropped every later DCB
        // while the HLE still returned success.  Keep the bounded wait slice only so an actual
        // Vulkan terminal transition is observed promptly, not as an artificial terminal deadline.
        cv_lock.wait_for(lock, WaitSlice, [] { return submission_lock == 0; });
    }
    if (liverpool && liverpool->IsRendererTerminal()) {
        return false;
    }
    return true;
#else
    cv_lock.wait(lock, [] { return submission_lock == 0; });
    return true;
#endif
}

#ifdef __ANDROID__
static bool ExecutorEnvFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static bool ExecutorTraceHotGnmEqEvent() {
    // GfxEop fires once per submitted frame. Keep its IRQ trace separate from broad live/wide
    // diagnostics so production rendering never performs a synchronous logcat write per IRQ.
    static const bool enabled = ExecutorEnvFlag("EXECUTOR_TRACE_GNM_EQ_EVENT");
    return enabled;
}

static bool ExecutorLiveGnmSyncSubmit() {
    // Reference parity: the reference submits GPU work ASYNC — SubmitGfx enqueues and the game
    // thread continues (it goes on to submit the flip, whose present sets the videoout flip label
    // that the content DCB's WAIT_REG_MEM waits on). Our executor-added synchronous drain
    // (WaitRasterizerIdleForExecutor right after SubmitGfx) blocks Game:Main BEFORE it can submit
    // that flip → the flip never presents → the label never gets set → the GPU coroutine's WAIT
    // never resolves → deadlock (drain hangs forever, ~2 frames then black). Force async off to
    // match the reference and break the deadlock. Opt back in with EXECUTOR_FORCE_GNM_SYNC_SUBMIT.
    static const bool enabled = ExecutorEnvFlag("EXECUTOR_FORCE_GNM_SYNC_SUBMIT");
    return enabled;
}

static void ExecutorLogLiveGnmStage(const char* stage, u32 workload, u32 cbpair, std::size_t dcb_dw,
                                    std::size_t ccb_dw) {
    // FPS: per-submit stage trace (~4.5k lines/run). Opt-in via dedicated env so it does not fire
    // under the broad EXECUTOR_TRACE_LIVE_WIDE and cost per-frame logcat I/O on the render path.
    static const bool trace_gnm_stage = std::getenv("EXECUTOR_TRACE_GNM_STAGE") != nullptr;
    if (!trace_gnm_stage || !ExecutorTraceLiveWide()) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_GNM_STAGE] stage=%s workload=%u cbpair=%u dcbDw=%zu ccbDw=%zu",
                        stage ? stage : "<null>", workload, cbpair, dcb_dw, ccb_dw);
}
#endif

// Write a special ending NOP packet with N DWs data block
static inline u32* WriteTrailingNop(u32* cmdbuf, u32 data_block_size) {
    auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
    nop->header = PM4Type3Header{PM4ItOpcode::Nop, data_block_size - 1};
    nop->data_block[0] = 0u; // only one out of `data_block_size` is initialized
    return cmdbuf + data_block_size + 1 /* header */;
}

// Write a special ending NOP packet with N DWs data block
template <u32 data_block_size>
static inline u32* WriteTrailingNop(u32* cmdbuf) {
    auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
    nop->header = PM4Type3Header{PM4ItOpcode::Nop, data_block_size - 1};
    nop->data_block[0] = 0u; // only one out of `data_block_size` is initialized
    return cmdbuf + data_block_size + 1 /* header */;
}

static inline u32* ClearContextState(u32* cmdbuf) {
    static constexpr std::array ClearStateSequence{
        0xc0012800u, 0x80000000u, 0x80000000u, 0xc0001200u, 0u, 0xc0055800u,
        0x2ec47fc0u, 0xffffffffu, 0u,          0u,          0u, 10u,
    };
    static_assert(ClearStateSequence.size() == 0xc);

    std::memcpy(cmdbuf, ClearStateSequence.data(), ClearStateSequence.size() * 4);
    return cmdbuf + ClearStateSequence.size();
}

static inline bool IsValidEventType(Platform::InterruptId id) {
    return (static_cast<u32>(id) >= static_cast<u32>(Platform::InterruptId::Compute0RelMem) &&
            static_cast<u32>(id) <= static_cast<u32>(Platform::InterruptId::Compute6RelMem)) ||
           static_cast<u32>(id) == static_cast<u32>(Platform::InterruptId::GfxEop);
}

s32 PS4_SYSV_ABI sceGnmAddEqEvent(OrbisKernelEqueue eq, u64 id, void* udata) {
    LOG_TRACE(Lib_GnmDriver, "called");
#ifdef __ANDROID__
    if (ExecutorTraceHotGnmEqEvent()) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_EQ_EVENT] op=add_enter eq=%ld id=0x%llx udata=%p",
                            eq, static_cast<unsigned long long>(id), udata);
    }
#endif

    auto equeue = GetEqueue(eq);
    if (!equeue) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_EQ_EVENT] op=add_missing_eq eq=%ld id=0x%llx",
                            eq, static_cast<unsigned long long>(id));
#endif
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    EqueueEvent kernel_event{};
    kernel_event.event.ident = id;
    kernel_event.event.filter = OrbisKernelEvent::Filter::GraphicsCore;
    kernel_event.event.flags = OrbisKernelEvent::Flags::Add;
    kernel_event.event.fflags = 0;
    kernel_event.event.data = id;
    kernel_event.event.udata = udata;

    equeue->AddEvent(kernel_event);

    Platform::IrqC::Instance()->Register(
        static_cast<Platform::InterruptId>(id),
        [=](Platform::InterruptId irq) {
            ASSERT_MSG(irq == static_cast<Platform::InterruptId>(id), "An unexpected IRQ occured");

            // We need to convert IRQ# to event id
            if (!IsValidEventType(irq))
                return;

            // Event data is expected to be an event type as per sceGnmGetEqEventType.
            equeue->TriggerEvent(static_cast<GnmEventType>(id),
                                 OrbisKernelEvent::Filter::GraphicsCore,
                                 reinterpret_cast<void*>(id));
#ifdef __ANDROID__
            if (id == static_cast<u64>(GnmEventType::GfxEop)) {
                AmdGpu::ExecutorEopTraceEqTrigger(id, static_cast<s64>(eq));
            }
            if (ExecutorTraceHotGnmEqEvent()) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_GNM_EQ_EVENT] op=trigger irq=%u id=0x%llx eq=%ld",
                    static_cast<u32>(irq), static_cast<unsigned long long>(id), eq);
            }
#endif
        },
        equeue);
#ifdef __ANDROID__
    if (ExecutorTraceHotGnmEqEvent()) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_EQ_EVENT] op=add_return eq=%ld id=0x%llx rc=0",
                            eq, static_cast<unsigned long long>(id));
    }
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmAreSubmitsAllowed() {
    LOG_TRACE(Lib_GnmDriver, "called");
    return submission_lock == 0;
}

int PS4_SYSV_ABI sceGnmBeginWorkload(u32 workload_stream, u64* workload) {
    if (workload) {
        *workload = (-(u32)(workload_stream < 0x10) & 1);
        return 0xf < workload_stream;
    }
    return 3;
}

s32 PS4_SYSV_ABI sceGnmComputeWaitOnAddress(u32* cmdbuf, u32 size, uintptr_t addr, u32 mask,
                                            u32 cmp_func, u32 ref) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 0xe)) {
        cmdbuf = WriteHeader<PM4ItOpcode::Nop>(cmdbuf, 3);
        cmdbuf = WriteBody(cmdbuf, 0u);
        cmdbuf += 2;

        const u32 is_mem = addr > 0xffffu;
        const u32 addr_mask = is_mem ? 0xfffffffcu : 0xffffu;
        auto* wait_reg_mem = reinterpret_cast<PM4CmdWaitRegMem*>(cmdbuf);
        wait_reg_mem->header = PM4Type3Header{PM4ItOpcode::WaitRegMem, 5};
        wait_reg_mem->raw = (is_mem << 4u) | (cmp_func & 7u);
        wait_reg_mem->poll_addr_lo_raw = u32(addr & addr_mask);
        wait_reg_mem->poll_addr_hi = u32(addr >> 32u);
        wait_reg_mem->ref = ref;
        wait_reg_mem->mask = mask;
        wait_reg_mem->poll_interval = 10u;

        WriteTrailingNop<2>(cmdbuf + 7);
        return ORBIS_OK;
    }
    return -1;
}

int PS4_SYSV_ABI sceGnmComputeWaitSemaphore() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmCreateWorkloadStream(u64 param1, u32* workload_stream) {
    if (param1 != 0 && workload_stream) {
        *workload_stream = 1;
        return 0;
    }
    return 3;
}

int PS4_SYSV_ABI sceGnmDebuggerGetAddressWatch() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerHaltWavefront() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerReadGds() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerReadSqIndirectRegister() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerResumeWavefront() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerResumeWavefrontCreation() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerSetAddressWatch() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerWriteGds() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebuggerWriteSqIndirectRegister() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebugHardwareStatus() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmDeleteEqEvent(OrbisKernelEqueue eq, u64 id) {
    LOG_TRACE(Lib_GnmDriver, "called");

    auto equeue = GetEqueue(eq);
    if (!equeue) {
        return ORBIS_KERNEL_ERROR_EBADF;
    }

    equeue->RemoveEvent(id, OrbisKernelEvent::Filter::GraphicsCore);

    Platform::IrqC::Instance()->Unregister(static_cast<Platform::InterruptId>(id), equeue);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmDestroyWorkloadStream() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

void PS4_SYSV_ABI sceGnmDingDong(u32 gnm_vqid, u32 next_offs_dw) {
    HLE_TRACE;
    LOG_DEBUG(Lib_GnmDriver, "vqid {}, offset_dw {}", gnm_vqid, next_offs_dw);

    if (gnm_vqid == 0) {
        return;
    }

    if (!WaitGpuIdle("sceGnmDingDong")) {
        return;
    }

    if (DebugState.ShouldPauseInSubmit()) {
        DebugState.PauseGuestThreads();
    }

    auto vqid = gnm_vqid - 1;
    auto& asc_queue = liverpool->asc_queues[{vqid}];

    auto& offs_dw = asc_next_offs_dw[vqid];

    if (next_offs_dw == offs_dw) {
        return;
    }

    if (next_offs_dw < offs_dw && next_offs_dw != 0) {
        // For cases if a submission is split at the end of the ring buffer, we need to submit it in
        // two parts to handle the wrap
        liverpool->SubmitAsc(gnm_vqid, {reinterpret_cast<const u32*>(asc_queue.map_addr) + offs_dw,
                                        asc_queue.ring_size_dw - offs_dw});
        offs_dw = 0;
    }

    const auto* acb_ptr = reinterpret_cast<const u32*>(asc_queue.map_addr) + offs_dw;
    const auto acb_size_dw = (next_offs_dw ? next_offs_dw : asc_queue.ring_size_dw) - offs_dw;
    const std::span acb_span{acb_ptr, acb_size_dw};

    asc_next_offs_dw[vqid] = next_offs_dw;

    if (DebugState.DumpingCurrentFrame()) {
        static auto last_frame_num = -1LL;
        static u32 seq_num{};
        if (last_frame_num == frames_submitted) {
            ++seq_num;
        } else {
            last_frame_num = frames_submitted;
            seq_num = 0u;
        }

        // Up to this point, all ACB submissions have been stored in a secondary command buffer.
        // Dumping them using the current ring pointer would result in files containing only the
        // `IndirectBuffer` command. To access the actual command stream, we need to unwrap the IB.
        auto acb = acb_span;
        auto base_addr = reinterpret_cast<uintptr_t>(acb_ptr);
        const auto* indirect_buffer =
            reinterpret_cast<const PM4CmdIndirectBuffer*>(acb_span.data());
        if (indirect_buffer->header.opcode == PM4ItOpcode::IndirectBuffer) {
            base_addr = reinterpret_cast<uintptr_t>(indirect_buffer->Address<const u32>());
            acb = {reinterpret_cast<const u32*>(base_addr), indirect_buffer->ib_size};
        }

        using namespace DebugStateType;

        DebugState.PushQueueDump({
            .type = QueueType::acb,
            .submit_num = seq_num,
            .num2 = gnm_vqid,
            .data = {acb.begin(), acb.end()},
            .base_addr = base_addr,
        });
    }
    liverpool->SubmitAsc(gnm_vqid, acb_span);
}

void PS4_SYSV_ABI sceGnmDingDongForWorkload(u32 gnm_vqid, u32 next_offs_dw, u64 workload_id) {
    LOG_DEBUG(Lib_GnmDriver, "called, redirecting to sceGnmDingDong");
    sceGnmDingDong(gnm_vqid, next_offs_dw);
}

int PS4_SYSV_ABI sceGnmDisableMipStatsReport() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmDispatchDirect(u32* cmdbuf, u32 size, u32 threads_x, u32 threads_y,
                                      u32 threads_z, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 9) && ((s32)(threads_x | threads_y | threads_z) > -1)) {
        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DispatchDirect>(cmdbuf, 4, PM4ShaderType::ShaderCompute,
                                                          predicate);
        cmdbuf = WriteBody(cmdbuf, threads_x, threads_y, threads_z);
        cmdbuf[0] = (flags & 0x18) + 1; // ordered append mode

        WriteTrailingNop<3>(cmdbuf + 1);
        return ORBIS_OK;
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDispatchIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 7)) {
        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DispatchIndirect>(cmdbuf, 2, PM4ShaderType::ShaderCompute,
                                                            predicate);
        cmdbuf[0] = data_offset;
        cmdbuf[1] = (flags & 0x18) + 1; // ordered append mode

        WriteTrailingNop<3>(cmdbuf + 2);
        return ORBIS_OK;
    }
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDispatchIndirectOnMec(u32* cmdbuf, u32 size, VAddr args, u32 modifier) {
    if (cmdbuf != nullptr && size == 8 && args != 0 && ((args & 3u) == 0)) {
        cmdbuf[0] = 0xc0021602 | (modifier & 1u);
        *(VAddr*)(&cmdbuf[1]) = args;
        cmdbuf[3] = (modifier & 0x18) | 1u;
        cmdbuf[4] = 0xc0021000;
        cmdbuf[5] = 0;
        return ORBIS_OK;
    }
    return ORBIS_FAIL;
}

u32 PS4_SYSV_ABI sceGnmDispatchInitDefaultHardwareState(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
        return 0;
    }

    cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
        cmdbuf, 0x216u,
        0xffffffffu); // COMPUTE_STATIC_THREAD_MGMT_SE0
    cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
        cmdbuf, 0x217u,
        0xffffffffu); // COMPUTE_STATIC_THREAD_MGMT_SE1

    if (sceKernelIsNeoMode()) {
        cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
            cmdbuf, 0x219u,
            0xffffffffu); // COMPUTE_STATIC_THREAD_MGMT_SE2
        cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
            cmdbuf, 0x21au,
            0xffffffffu); // COMPUTE_STATIC_THREAD_MGMT_SE3
    }

    cmdbuf = PM4CmdSetData::SetShReg<PM4ShaderType::ShaderCompute>(
        cmdbuf, 0x215u, 0x170u); // COMPUTE_RESOURCE_LIMITS

    cmdbuf = WriteHeader<PM4ItOpcode::AcquireMem>(cmdbuf, 6);
    cmdbuf = WriteBody(cmdbuf, 0x28000000u, 0u, 0u, 0u, 0u, 0xau);

    cmdbuf = WriteHeader<PM4ItOpcode::Nop>(cmdbuf, sceKernelIsNeoMode() ? 0xe9 : 0xef);
    cmdbuf = WriteBody(cmdbuf, 0u);
    return HwInitPacketSize;
}

s32 PS4_SYSV_ABI sceGnmDrawIndex(u32* cmdbuf, u32 size, u32 index_count, uintptr_t index_addr,
                                 u32 flags, u32 type) {
    LOG_TRACE(Lib_GnmDriver, "called");
#ifdef __ANDROID__
    if (ExecutorTraceLiveWide()) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_DRAW] api=sceGnmDrawIndex cmdbuf=%p size=%u count=%u index=0x%llx flags=0x%x type=%u",
                            cmdbuf, size, index_count,
                            static_cast<unsigned long long>(index_addr), flags, type);
        LOG_INFO(Lib_GnmDriver,
                 "EXECUTOR_GNM_DRAW_ATTEMPT api=sceGnmDrawIndex index_count={} size={} flags={:#x}",
                 index_count, size, flags);
        LOG_INFO(Lib_GnmDriver,
                 "EXECUTOR_GNM_DRAW api=sceGnmDrawIndex index_count={} size={} flags={:#x}",
                 index_count, size, flags);
    }
#else
    LOG_INFO(Lib_GnmDriver,
             "EXECUTOR_GNM_DRAW_ATTEMPT api=sceGnmDrawIndex index_count={} size={} flags={:#x}",
             index_count, size, flags);
    LOG_INFO(Lib_GnmDriver,
             "EXECUTOR_GNM_DRAW api=sceGnmDrawIndex index_count={} size={} flags={:#x}",
             index_count, size, flags);
#endif

    if (cmdbuf && (size == 10) && (index_addr != 0) && (index_addr & 1) == 0 &&
        (flags & 0x1ffffffe) == 0) { // no predication will be set in the packet
        auto* draw_index = reinterpret_cast<PM4CmdDrawIndex2*>(cmdbuf);
        draw_index->header =
            PM4Type3Header{PM4ItOpcode::DrawIndex2, 4, PM4ShaderType::ShaderGraphics};
        draw_index->max_size = index_count;
        draw_index->index_base_lo = u32(index_addr);
        draw_index->index_base_hi = u32(index_addr >> 32);
        draw_index->index_count = index_count;
        draw_index->draw_initiator = sceKernelIsNeoMode() ? flags & 0xe0000000u : 0;

        WriteTrailingNop<3>(cmdbuf + 6);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawIndex", "OK", cmdbuf, size);
        ExecutorLiveDumpBuilderTail("sceGnmDrawIndex", cmdbuf, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawIndex", "bad_args", cmdbuf, size);
#endif
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexAuto(u32* cmdbuf, u32 size, u32 index_count, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");
#ifdef __ANDROID__
    if (ExecutorTraceLiveWide()) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_DRAW] api=sceGnmDrawIndexAuto cmdbuf=%p size=%u count=%u flags=0x%x",
                            cmdbuf, size, index_count, flags);
        LOG_INFO(Lib_GnmDriver,
                 "EXECUTOR_GNM_DRAW_ATTEMPT api=sceGnmDrawIndexAuto index_count={} size={} flags={:#x}",
                 index_count, size, flags);
        LOG_INFO(Lib_GnmDriver,
                 "EXECUTOR_GNM_DRAW api=sceGnmDrawIndexAuto index_count={} size={} flags={:#x}",
                 index_count, size, flags);
    }
#else
    LOG_INFO(Lib_GnmDriver,
             "EXECUTOR_GNM_DRAW_ATTEMPT api=sceGnmDrawIndexAuto index_count={} size={} flags={:#x}",
             index_count, size, flags);
    LOG_INFO(Lib_GnmDriver,
             "EXECUTOR_GNM_DRAW api=sceGnmDrawIndexAuto index_count={} size={} flags={:#x}",
             index_count, size, flags);
#endif

    if (cmdbuf && (size == 7) &&
        (flags & 0x1ffffffe) == 0) { // no predication will be set in the packet
        cmdbuf = WritePacket<PM4ItOpcode::DrawIndexAuto>(
            cmdbuf, PM4ShaderType::ShaderGraphics, index_count,
            sceKernelIsNeoMode() ? flags & 0xe0000000u | 2u : 2u);
        WriteTrailingNop<3>(cmdbuf);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawIndexAuto", "OK", cmdbuf - 3, size);
        ExecutorLiveDumpBuilderTail("sceGnmDrawIndexAuto", cmdbuf - 3, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawIndexAuto", "bad_args", cmdbuf, size);
#endif
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 shader_stage,
                                         u32 vertex_sgpr_offset, u32 instance_sgpr_offset,
                                         u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 9) && (shader_stage < ShaderStages::Max) &&
        (vertex_sgpr_offset < 0x10u) && (instance_sgpr_offset < 0x10u)) {

        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndexIndirect>(
            cmdbuf, 4, PM4ShaderType::ShaderGraphics, predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];

        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = sceKernelIsNeoMode() ? flags & 0xe0000000u : 0u;

        cmdbuf += 4;
        WriteTrailingNop<3>(cmdbuf);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawIndexIndirect", "OK", cmdbuf - 7, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawIndexIndirect", "bad_args", cmdbuf, size);
#endif
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexIndirectCountMulti(u32* cmdbuf, u32 size, u32 data_offset,
                                                   u32 max_count, u64 count_addr, u32 shader_stage,
                                                   u32 vertex_sgpr_offset, u32 instance_sgpr_offset,
                                                   u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if ((!sceKernelIsNeoMode() || !UseNeoCompatSequences) && cmdbuf && (size == 16) &&
        (vertex_sgpr_offset < 0x10u) && (instance_sgpr_offset < 0x10u) &&
        (shader_stage == ShaderStages::Vs || shader_stage == ShaderStages::Es ||
         shader_stage == ShaderStages::Ls)) {

        cmdbuf = WriteHeader<PM4ItOpcode::Nop>(cmdbuf, 2);
        cmdbuf = WriteBody(cmdbuf, 0u);
        cmdbuf += 1;

        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndexIndirectCountMulti>(
            cmdbuf, 9, PM4ShaderType::ShaderGraphics, predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];

        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = (count_addr != 0 ? 1u : 0u) << 0x1e;
        cmdbuf[4] = max_count;
        *(u64*)(&cmdbuf[5]) = count_addr;
        cmdbuf[7] = sizeof(DrawIndexedIndirectArgs);
        cmdbuf[8] = sceKernelIsNeoMode() ? flags & 0xe0000000u : 0;

        cmdbuf += 9;
        WriteTrailingNop<2>(cmdbuf);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawIndexIndirectCountMulti", "OK", cmdbuf - 14, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawIndexIndirectCountMulti", "bad_args", cmdbuf, size);
#endif
    return -1;
}

int PS4_SYSV_ABI sceGnmDrawIndexIndirectMulti(u32* cmdbuf, u32 size, u32 data_offset, u32 max_count,
                                              u32 shader_stage, u32 vertex_sgpr_offset,
                                              u32 instance_sgpr_offset, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 11) && (vertex_sgpr_offset < 0x10u) && (instance_sgpr_offset < 0x10u) &&
        (shader_stage == ShaderStages::Vs || shader_stage == ShaderStages::Es ||
         shader_stage == ShaderStages::Ls)) {

        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndexIndirectMulti>(
            cmdbuf, 6, PM4ShaderType::ShaderGraphics, predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];

        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = max_count;
        cmdbuf[4] = sizeof(DrawIndexedIndirectArgs);
        cmdbuf[5] = sceKernelIsNeoMode() ? flags & 0xe0000000u : 0;

        cmdbuf += 6;
        WriteTrailingNop<3>(cmdbuf);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawIndexIndirectMulti", "OK", cmdbuf - 9, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawIndexIndirectMulti", "bad_args", cmdbuf, size);
#endif
    return -1;
}

int PS4_SYSV_ABI sceGnmDrawIndexMultiInstanced() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    UNREACHABLE();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexOffset(u32* cmdbuf, u32 size, u32 index_offset, u32 index_count,
                                       u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 9)) {
        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndexOffset2>(
            cmdbuf, 4, PM4ShaderType::ShaderGraphics, predicate);
        cmdbuf = WriteBody(cmdbuf, index_count, index_offset, index_count,
                           sceKernelIsNeoMode() ? flags & 0xe0000000u : 0u);

        WriteTrailingNop<3>(cmdbuf);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawIndexOffset", "OK", cmdbuf - 5, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawIndexOffset", "bad_args", cmdbuf, size);
#endif
    return -1;
}

s32 PS4_SYSV_ABI sceGnmDrawIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 shader_stage,
                                    u32 vertex_sgpr_offset, u32 instance_sgpr_offset, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 9) && (shader_stage < ShaderStages::Max) &&
        (vertex_sgpr_offset < 0x10u) && (instance_sgpr_offset < 0x10u)) {

        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndirect>(cmdbuf, 4, PM4ShaderType::ShaderGraphics,
                                                        predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];

        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = sceKernelIsNeoMode() ? flags & 0xe0000000u | 2u : 2u; // auto index

        cmdbuf += 4;
        WriteTrailingNop<3>(cmdbuf);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawIndirect", "OK", cmdbuf - 7, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawIndirect", "bad_args", cmdbuf, size);
#endif
    return -1;
}

int PS4_SYSV_ABI sceGnmDrawIndirectCountMulti() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    UNREACHABLE();
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmDrawIndirectMulti(u32* cmdbuf, u32 size, u32 data_offset, u32 max_count,
                                         u32 shader_stage, u32 vertex_sgpr_offset,
                                         u32 instance_sgpr_offset, u32 flags) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && size == 11 && shader_stage < ShaderStages::Max && vertex_sgpr_offset < 0x10 &&
        instance_sgpr_offset < 0x10) {
        const auto predicate = flags & 1 ? PM4Predicate::PredEnable : PM4Predicate::PredDisable;
        cmdbuf = WriteHeader<PM4ItOpcode::DrawIndirectMulti>(
            cmdbuf, 4, PM4ShaderType::ShaderGraphics, predicate);

        const auto sgpr_offset = indirect_sgpr_offsets[shader_stage];
        cmdbuf[0] = data_offset;
        cmdbuf[1] = vertex_sgpr_offset == 0 ? 0 : (vertex_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[2] = instance_sgpr_offset == 0 ? 0 : (instance_sgpr_offset & 0xffffu) + sgpr_offset;
        cmdbuf[3] = max_count;
        cmdbuf[4] = sizeof(DrawIndirectArgs);
        cmdbuf[5] = sceKernelIsNeoMode() ? flags & 0xe0000000u | 2u : 2u; // auto index

        cmdbuf += 6;
        WriteTrailingNop<3>(cmdbuf);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawIndirectMulti", "OK", cmdbuf - 9, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawIndirectMulti", "bad_args", cmdbuf, size);
#endif
    return -1;
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawInitDefaultHardwareState", "bad_size", cmdbuf, size);
#endif
        return 0;
    }

    const auto& SetupContext = [](u32* cmdbuf, u32 size, bool clear_state) {
        const auto* cmdbuf_end = cmdbuf + HwInitPacketSize;
        if (clear_state) {
            cmdbuf = ClearContextState(cmdbuf);
        }

        std::memcpy(cmdbuf, &InitSequence[2], (InitSequence.size() - 2) * 4);
        cmdbuf += InitSequence.size() - 2;

        const auto cmdbuf_left = cmdbuf_end - cmdbuf - 1;
        WriteTrailingNop(cmdbuf, cmdbuf_left);

        return HwInitPacketSize;
    };

    const u32 written = SetupContext(cmdbuf, size, true);
#ifdef __ANDROID__
    // `size` is the caller's remaining DCB capacity (often ~1M dwords), not bytes written.
    // Tracking that capacity makes active_cmdbuf cover unwritten memory and defeats DCB analysis.
    ExecutorLiveGnmBuilderLog("sceGnmDrawInitDefaultHardwareState", "OK", cmdbuf, written,
                              "written=256");
#endif
    return written;
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState175(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawInitDefaultHardwareState175", "bad_size", cmdbuf,
                                  size);
#endif
        return 0;
    }

    auto* const cmdbuf_begin = cmdbuf;
    const auto* cmdbuf_end = cmdbuf + HwInitPacketSize;
    cmdbuf = ClearContextState(cmdbuf);
    std::memcpy(cmdbuf, &InitSequence175[2], (InitSequence175.size() - 2) * 4);
    cmdbuf += InitSequence175.size() - 2;

    const auto cmdbuf_left = cmdbuf_end - cmdbuf - 1;
    WriteTrailingNop(cmdbuf, cmdbuf_left);

#ifdef __ANDROID__
    // cmdbuf has advanced into the init sequence here. Record the real DCB base and written range.
    ExecutorLiveGnmBuilderLog("sceGnmDrawInitDefaultHardwareState175", "OK", cmdbuf_begin,
                              HwInitPacketSize,
                              "written=256");
#endif
    return HwInitPacketSize;
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState200(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawInitDefaultHardwareState200", "bad_size", cmdbuf,
                                  size);
#endif
        return 0;
    }

    const auto& SetupContext200 = [](u32* cmdbuf, u32 size, bool clear_state) {
        const auto* cmdbuf_end = cmdbuf + HwInitPacketSize;
        if (clear_state) {
            cmdbuf = ClearContextState(cmdbuf);
        }

        if (sceKernelIsNeoMode()) {
            if (!UseNeoCompatSequences) {
                std::memcpy(cmdbuf, &InitSequence200Neo[2], (InitSequence200Neo.size() - 2) * 4);
                cmdbuf += InitSequence200Neo.size() - 2;
            } else {
                std::memcpy(cmdbuf, &InitSequence200NeoCompat[2],
                            (InitSequence200NeoCompat.size() - 2) * 4);
                cmdbuf += InitSequence200NeoCompat.size() - 2;
            }
        } else {
            std::memcpy(cmdbuf, &InitSequence200[2], (InitSequence200.size() - 2) * 4);
            cmdbuf += InitSequence200.size() - 2;
        }

        const auto cmdbuf_left = cmdbuf_end - cmdbuf - 1;
        WriteTrailingNop(cmdbuf, cmdbuf_left);

        return HwInitPacketSize;
    };

    const u32 written = SetupContext200(cmdbuf, size, true);
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawInitDefaultHardwareState200", "OK", cmdbuf, written,
                              "written=256");
#endif
    return written;
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState350(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size < HwInitPacketSize) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawInitDefaultHardwareState350", "bad_size", cmdbuf,
                                  size);
#endif
        return 0;
    }

    const auto& SetupContext350 = [](u32* cmdbuf, u32 size, bool clear_state) {
        const auto* cmdbuf_end = cmdbuf + HwInitPacketSize;
        if (clear_state) {
            cmdbuf = ClearContextState(cmdbuf);
        }

        if (sceKernelIsNeoMode()) {
            if (!UseNeoCompatSequences) {
                std::memcpy(cmdbuf, &InitSequence350Neo[2], (InitSequence350Neo.size() - 2) * 4);
                cmdbuf += InitSequence350Neo.size() - 2;
            } else {
                std::memcpy(cmdbuf, &InitSequence350NeoCompat[2],
                            (InitSequence350NeoCompat.size() - 2) * 4);
                cmdbuf += InitSequence350NeoCompat.size() - 2;
            }
        } else {
            std::memcpy(cmdbuf, &InitSequence350[2], (InitSequence350.size() - 2) * 4);
            cmdbuf += InitSequence350.size() - 2;
        }

        const auto cmdbuf_left = cmdbuf_end - cmdbuf - 1;
        WriteTrailingNop(cmdbuf, cmdbuf_left);

        return HwInitPacketSize;
    };

    const u32 written = SetupContext350(cmdbuf, size, true);
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawInitDefaultHardwareState350", "OK", cmdbuf, written,
                              "written=256");
#endif
    return written;
}

u32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextState(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    constexpr auto CtxInitPacketSize = 0x20u;
    if (size != CtxInitPacketSize) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawInitToDefaultContextState", "bad_size", cmdbuf, size);
#endif
        return 0;
    }

    if (sceKernelIsNeoMode()) {
        std::memcpy(cmdbuf, CtxInitSequenceNeo.data(), CtxInitSequenceNeo.size() * 4);
    } else {
        std::memcpy(cmdbuf, CtxInitSequence.data(), CtxInitSequence.size() * 4);
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawInitToDefaultContextState", "OK", cmdbuf, size,
                              "written=32");
#endif
    return CtxInitPacketSize;
}

u32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextState400(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    constexpr auto CtxInitPacketSize = 0x100u;
    if (size != CtxInitPacketSize) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmDrawInitToDefaultContextState400", "bad_size", cmdbuf,
                                  size);
#endif
        return 0;
    }

    if (sceKernelIsNeoMode()) {
        if (!UseNeoCompatSequences) {
            std::memcpy(cmdbuf, CtxInitSequence400Neo.data(), CtxInitSequence400Neo.size() * 4);
        } else {
            std::memcpy(cmdbuf, CtxInitSequence400NeoCompat.data(),
                        CtxInitSequence400NeoCompat.size() * 4);
        }
    } else {
        std::memcpy(cmdbuf, CtxInitSequence400.data(), CtxInitSequence400.size() * 4);
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmDrawInitToDefaultContextState400", "OK", cmdbuf, size,
                              "written=256");
#endif
    return CtxInitPacketSize;
}

int PS4_SYSV_ABI sceGnmDrawOpaqueAuto() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

bool PS4_SYSV_ABI sceGnmDriverCaptureInProgress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterface() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForGpuDebugger() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForGpuException() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForHDRScopes() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForReplay() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForResourceRegistration() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

u32 PS4_SYSV_ABI sceGnmDriverInternalRetrieveGnmInterfaceForValidation() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0x80000000;
}

int PS4_SYSV_ABI sceGnmDriverInternalVirtualQuery() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

bool PS4_SYSV_ABI sceGnmDriverTraceInProgress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

int PS4_SYSV_ABI sceGnmDriverTriggerCapture() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_RAZOR_NOT_LOADED;
}

int PS4_SYSV_ABI sceGnmEndWorkload(u64 workload) {
    if (workload != 0) {
        return (0xf < ((workload >> 0x38) & 0xff)) * 2;
    }
    return 2;
}

s32 PS4_SYSV_ABI sceGnmFindResourcesPublic() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

void PS4_SYSV_ABI sceGnmFlushGarlic() {
    LOG_TRACE(Lib_GnmDriver, "(STUBBED) called");
}

int PS4_SYSV_ABI sceGnmGetCoredumpAddress() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetCoredumpMode() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetCoredumpProtectionFaultTimestamp() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetDbgGcHandle() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return -1;
}

int PS4_SYSV_ABI sceGnmGetDebugTimestamp() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetEqEventType(const OrbisKernelEvent* ev) {
    LOG_TRACE(Lib_GnmDriver, "called");
    return sceKernelGetEventData(ev);
}

int PS4_SYSV_ABI sceGnmGetEqTimeStamp() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetGpuBlockStatus() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

u32 PS4_SYSV_ABI sceGnmGetGpuCoreClockFrequency() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // On console this uses an ioctl check, but we assume it is equal to just checking for neo mode.
    return sceKernelIsNeoMode() ? 911'000'000 : 800'000'000;
}

int PS4_SYSV_ABI sceGnmGetGpuInfoStatus() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetLastWaitedAddress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetNumTcaUnits() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetOffChipTessellationBufferSize() {
    LOG_TRACE(Lib_GnmDriver, "called");
    return tessellation_offchip_buffer_size;
}

int PS4_SYSV_ABI sceGnmGetOwnerName() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetPhysicalCounterFromVirtualized() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

u32 PS4_SYSV_ABI sceGnmGetProtectionFaultTimeStamp() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0;
}

int PS4_SYSV_ABI sceGnmGetResourceBaseAddressAndSizeInBytes() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceName() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceShaderGuid() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceType() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceUserData() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetShaderProgramBaseAddress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmGetShaderStatus() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

VAddr PS4_SYSV_ABI sceGnmGetTheTessellationFactorRingBufferBaseAddress() {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (tessellation_factors_ring_addr == -1) {
        auto* memory = Core::Memory::Instance();
        auto& address_space = memory->GetAddressSpace();
        tessellation_factors_ring_addr = address_space.SystemReservedVirtualBase() +
                                         address_space.SystemReservedVirtualSize() - 0x10000000;
    }

    return tessellation_factors_ring_addr;
}

void PS4_SYSV_ABI sceGnmGpuPaDebugEnter() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
}

void PS4_SYSV_ABI sceGnmGpuPaDebugLeave() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
}

s32 PS4_SYSV_ABI sceGnmInsertDingDongMarker(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

#ifdef __ANDROID__
    static std::atomic<bool> traced{false};
    if (!traced.exchange(true, std::memory_order_relaxed)) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_GNM_PC_ABI] api=sceGnmInsertDingDongMarker "
                            "cmdbuf=%p size=%u",
                            cmdbuf, size);
    }
#endif
    if (cmdbuf == nullptr || size != 4) {
        return -1;
    }
    WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0u, 0u, 0u);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmInsertPopMarker(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (cmdbuf && (size == 6)) {
        cmdbuf =
            WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics,
                                          PM4CmdNop::PayloadType::DebugMarkerPop, 0u, 0u, 0u, 0u);
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmInsertPopMarker", "OK", cmdbuf - 6, size);
#endif
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmInsertPopMarker", "bad_args", cmdbuf, size);
#endif
    return -1;
}

s32 PS4_SYSV_ABI sceGnmInsertPushColorMarker(u32* cmdbuf, u32 size, const char* marker, u32 color) {
    LOG_TRACE(Lib_GnmDriver, "called");
#ifdef __ANDROID__
    char detail[160]{};
    std::snprintf(detail, sizeof(detail), "marker=\"%.96s\" color=0x%x",
                  marker ? marker : "<null>", color);
#endif

    if (cmdbuf && marker) {
        const auto len = std::strlen(marker);
        const u32 packet_size = ((len + 0xc) >> 2) + ((len + 0x10) >> 3) * 2;
        if (packet_size + 2 == size) {
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
            nop->header =
                PM4Type3Header{PM4ItOpcode::Nop, packet_size, PM4ShaderType::ShaderGraphics};
            nop->data_block[0] = PM4CmdNop::PayloadType::DebugColorMarkerPush;
            const auto marker_len = len + 1;
            std::memcpy(&nop->data_block[1], marker, marker_len);
            *reinterpret_cast<u32*>(reinterpret_cast<u8*>(&nop->data_block[1]) + marker_len + 8) =
                color;
            std::memset(reinterpret_cast<u8*>(&nop->data_block[1]) + marker_len + 8 + sizeof(u32),
                        0, packet_size * 4 - marker_len - 8 - sizeof(u32));
#ifdef __ANDROID__
            ExecutorLiveGnmBuilderLog("sceGnmInsertPushColorMarker", "OK", cmdbuf, size, detail);
#endif
            return ORBIS_OK;
        }
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmInsertPushColorMarker", "bad_args", cmdbuf, size, detail);
#endif
    return -1;
}

s32 PS4_SYSV_ABI sceGnmInsertPushMarker(u32* cmdbuf, u32 size, const char* marker) {
    LOG_TRACE(Lib_GnmDriver, "called");
#ifdef __ANDROID__
    char detail[144]{};
    std::snprintf(detail, sizeof(detail), "marker=\"%.96s\"", marker ? marker : "<null>");
#endif

    if (cmdbuf && marker) {
        const auto len = std::strlen(marker);
        const u32 packet_size = ((len + 8) >> 2) + ((len + 0xc) >> 3) * 2;
        if (packet_size + 2 == size) {
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
            nop->header =
                PM4Type3Header{PM4ItOpcode::Nop, packet_size, PM4ShaderType::ShaderGraphics};
            nop->data_block[0] = PM4CmdNop::PayloadType::DebugMarkerPush;
            const auto marker_len = len + 1;
            std::memcpy(&nop->data_block[1], marker, marker_len);
            std::memset(reinterpret_cast<u8*>(&nop->data_block[1]) + marker_len, 0,
                        packet_size * 4 - marker_len);
#ifdef __ANDROID__
            ExecutorLiveGnmBuilderLog("sceGnmInsertPushMarker", "OK", cmdbuf, size, detail);
#endif
            return ORBIS_OK;
        }
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmInsertPushMarker", "bad_args", cmdbuf, size, detail);
#endif
    return -1;
}

int PS4_SYSV_ABI sceGnmInsertSetColorMarker() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmInsertSetMarker(u32* cmdbuf, u32 size, const char* marker) {
    LOG_TRACE(Lib_GnmDriver, "called");
#ifdef __ANDROID__
    char detail[144]{};
    std::snprintf(detail, sizeof(detail), "marker=\"%.96s\"", marker ? marker : "<null>");
#endif

    if (cmdbuf && marker) {
        const auto len = std::strlen(marker);
        const u32 packet_size = ((len + 8) >> 2) + ((len + 0xc) >> 3) * 2;
        if (packet_size + 2 == size) {
            auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf);
            nop->header =
                PM4Type3Header{PM4ItOpcode::Nop, packet_size, PM4ShaderType::ShaderGraphics};
            nop->data_block[0] = PM4CmdNop::PayloadType::DebugSetMarker;
            const auto marker_len = len + 1;
            std::memcpy(&nop->data_block[1], marker, marker_len);
            std::memset(reinterpret_cast<u8*>(&nop->data_block[1]) + marker_len, 0,
                        packet_size * 4 - marker_len);
#ifdef __ANDROID__
            ExecutorLiveGnmBuilderLog("sceGnmInsertSetMarker", "OK", cmdbuf, size, detail);
#endif
            return ORBIS_OK;
        }
    }
#ifdef __ANDROID__
    ExecutorLiveGnmBuilderLog("sceGnmInsertSetMarker", "bad_args", cmdbuf, size, detail);
#endif
    return -1;
}

int PS4_SYSV_ABI sceGnmInsertThreadTraceMarker() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmInsertWaitFlipDone(u32* cmdbuf, u32 size, s32 vo_handle, u32 buf_idx) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (size != 7) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_WAIT_FLIP] result=bad_size cmdbuf=%p size=%u handle=%d buf=%u",
                            cmdbuf, size, vo_handle, buf_idx);
        char detail[96]{};
        std::snprintf(detail, sizeof(detail), "handle=%d buf=%u", vo_handle, buf_idx);
        ExecutorLiveGnmBuilderLog("sceGnmInsertWaitFlipDone", "bad_size", cmdbuf, size, detail);
#endif
        return -1;
    }

    uintptr_t label_addr{};
    ASSERT_MSG(VideoOut::sceVideoOutGetBufferLabelAddress(vo_handle, &label_addr) == 16,
               "sceVideoOutGetBufferLabelAddress call failed");
    u32 effective_buf_idx = buf_idx;
    const bool resolved_buf =
        VideoOut::ExecutorResolveVoBufferIndex(vo_handle, buf_idx, &effective_buf_idx);
    const auto* labels = reinterpret_cast<const u64*>(label_addr);
#ifdef __ANDROID__
    if (ExecutorTraceLiveWide()) {
        const u64 label0 = labels ? labels[0] : 0;
        const u64 label1 = labels ? labels[1] : 0;
        const u64 label2 = labels ? labels[2] : 0;
        const u64 label3 = labels ? labels[3] : 0;
        const u64 requested_label =
            labels && buf_idx < 16 ? labels[buf_idx] : 0xffff'ffff'ffff'ffffull;
        const u64 effective_label =
            labels && resolved_buf && effective_buf_idx < 16 ? labels[effective_buf_idx]
                                                             : 0xffff'ffff'ffff'ffffull;
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_GNM_WAIT_FLIP] cmdbuf=%p size=%u handle=%d requested=%u effective=%u resolved=%u labelAddr=0x%llx reqLabel=%llu effLabel=%llu labels=%llu,%llu,%llu,%llu",
            cmdbuf, size, vo_handle, buf_idx, effective_buf_idx, resolved_buf ? 1u : 0u,
            static_cast<unsigned long long>(label_addr),
            static_cast<unsigned long long>(requested_label),
            static_cast<unsigned long long>(effective_label),
            static_cast<unsigned long long>(label0), static_cast<unsigned long long>(label1),
            static_cast<unsigned long long>(label2), static_cast<unsigned long long>(label3));
    }
#endif

    auto* wait_reg_mem = reinterpret_cast<PM4CmdWaitRegMem*>(cmdbuf);
    wait_reg_mem->header = PM4Type3Header{PM4ItOpcode::WaitRegMem, 5};
    wait_reg_mem->raw = 0x13u;
    const u32 wait_buf_idx = resolved_buf ? effective_buf_idx : buf_idx;
    *reinterpret_cast<uintptr_t*>(&wait_reg_mem->poll_addr_lo) =
        (label_addr + wait_buf_idx * sizeof(uintptr_t)) & ~0x3ull;
    wait_reg_mem->ref = 0u;
    wait_reg_mem->mask = 0xffff'ffffu;
    wait_reg_mem->poll_interval = 10u;
#ifdef __ANDROID__
    if (std::getenv("EXECUTOR_LIVE_FORCE_FLIP_DONE") != nullptr) {
        const uintptr_t poll_addr = *reinterpret_cast<uintptr_t*>(&wait_reg_mem->poll_addr_lo);
        const u32 value = wait_reg_mem->ref;
        const bool wrote = VideoOut::ExecutorTryWriteVoLabelAlias(
            reinterpret_cast<const void*>(poll_addr), &value, sizeof(value));
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_FORCE_FLIP_DONE] op=wait_flip_label handle=%d requested=%u effective=%u resolved=%u wait=%u poll=0x%llx ref=0x%x wrote=%u",
            vo_handle, buf_idx, effective_buf_idx, resolved_buf ? 1u : 0u, wait_buf_idx,
            static_cast<unsigned long long>(poll_addr), value, wrote ? 1u : 0u);
    }
    if (ExecutorTraceLiveWide()) {
        const uintptr_t poll_addr = *reinterpret_cast<uintptr_t*>(&wait_reg_mem->poll_addr_lo);
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_GNM_WAIT_FLIP_PACKET] poll=0x%llx ref=0x%x mask=0x%x interval=%u raw=0x%x",
            static_cast<unsigned long long>(poll_addr), wait_reg_mem->ref, wait_reg_mem->mask,
            wait_reg_mem->poll_interval, wait_reg_mem->raw);
    }
#endif
#ifdef __ANDROID__
    char builder_detail[160]{};
    const uintptr_t builder_poll_addr = *reinterpret_cast<uintptr_t*>(&wait_reg_mem->poll_addr_lo);
    std::snprintf(builder_detail, sizeof(builder_detail),
                  "handle=%d requested=%u effective=%u resolved=%u wait=%u poll=0x%llx",
                  vo_handle, buf_idx, effective_buf_idx, resolved_buf ? 1u : 0u, wait_buf_idx,
                  static_cast<unsigned long long>(builder_poll_addr));
    ExecutorLiveGnmBuilderLog("sceGnmInsertWaitFlipDone", "OK", cmdbuf, size, builder_detail);
    ExecutorLiveDumpBuilderTail("sceGnmInsertWaitFlipDone", cmdbuf, size);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmIsCoredumpValid() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

bool PS4_SYSV_ABI sceGnmIsUserPaEnabled() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware. Keep the upstream default, but allow a live bring-up
    // gate for titles that branch their GNM builder path on this query.
    const bool enabled = std::getenv("EXECUTOR_LIVE_GNM_USER_PA") != nullptr;
#ifdef __ANDROID__
    if (ExecutorTraceLiveWide()) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_RETURN] api=sceGnmIsUserPaEnabled value=%u",
                            enabled ? 1u : 0u);
    }
#endif
    return enabled;
}

int PS4_SYSV_ABI sceGnmLogicalCuIndexToPhysicalCuIndex() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmLogicalCuMaskToPhysicalCuMask(s64, s32 logical_cu_mask) {
    LOG_INFO(Lib_GnmDriver, "called, logical_cu_mask: {}", logical_cu_mask);
#ifdef __ANDROID__
    static std::atomic<bool> traced{false};
    if (!traced.exchange(true, std::memory_order_relaxed)) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_GNM_PC_ABI] "
                            "api=sceGnmLogicalCuMaskToPhysicalCuMask mask=0x%x",
                            static_cast<u32>(logical_cu_mask));
    }
#endif
    return logical_cu_mask;
}

int PS4_SYSV_ABI sceGnmLogicalTcaUnitToPhysical() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmMapComputeQueue(u32 pipe_id, u32 queue_id, VAddr ring_base_addr,
                                       u32 ring_size_dw, u32* read_ptr_addr) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (pipe_id >= Liverpool::NumComputePipes) {
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_PIPE_ID;
    }

    if (queue_id >= Liverpool::NumQueuesPerPipe) {
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_QUEUE_ID;
    }

    if (VAddr(ring_base_addr) % 256 != 0) { // alignment check
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_RING_BASE_ADDR;
    }

    if (!std::has_single_bit(ring_size_dw)) {
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_RING_SIZE;
    }

    if (VAddr(read_ptr_addr) % 4 != 0) { // alignment check
        return ORBIS_GNM_ERROR_COMPUTEQUEUE_INVALID_READ_PTR_ADDR;
    }

    const auto vqid =
        liverpool->asc_queues.insert(VAddr(ring_base_addr), read_ptr_addr, ring_size_dw, pipe_id);
    // We need to offset index as `dingDong` assumes it to be from the range [1..64]
    const auto gnm_vqid = vqid.index + 1;
    LOG_INFO(Lib_GnmDriver, "ASC pipe {} queue {} mapped to vqueue {}", pipe_id, queue_id,
             gnm_vqid);

    const auto& queue = liverpool->asc_queues[vqid];
    *queue.read_addr = 0u;

    return gnm_vqid;
}

int PS4_SYSV_ABI sceGnmMapComputeQueueWithPriority(u32 pipe_id, u32 queue_id, VAddr ring_base_addr,
                                                   u32 ring_size_dw, u32* read_ptr_addr,
                                                   u32 pipePriority) {
    LOG_TRACE(Lib_GnmDriver, "called");

    (void)pipePriority;
    return sceGnmMapComputeQueue(pipe_id, queue_id, ring_base_addr, ring_size_dw, read_ptr_addr);
}

int PS4_SYSV_ABI sceGnmPaDisableFlipCallbacks() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmPaEnableFlipCallbacks() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmPaHeartbeat() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmQueryResourceRegistrationUserMemoryRequirements() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmRaiseUserExceptionEvent() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmRegisterGdsResource() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

void PS4_SYSV_ABI sceGnmRegisterGnmLiveCallbackConfig() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
}

s32 PS4_SYSV_ABI sceGnmRegisterOwner(void* handle, const char* name) {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

s32 PS4_SYSV_ABI sceGnmRegisterResource(void* res_handle, void* owner_handle, const void* addr,
                                        size_t size, const char* name, int res_type,
                                        u64 user_data) {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmRequestFlipAndSubmitDone() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_LIVE_GNM_REQUEST_SUBMIT_DONE] api=sceGnmRequestFlipAndSubmitDone");
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmRequestFlipAndSubmitDoneForWorkload() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
#ifdef __ANDROID__
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_LIVE_GNM_REQUEST_SUBMIT_DONE] api=sceGnmRequestFlipAndSubmitDoneForWorkload");
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmRequestMipStatsReportAndReset() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmResetVgtControl(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (cmdbuf == nullptr || size != 3) {
        return -1;
    }
    if (sceKernelIsNeoMode()) {
        if (!UseNeoCompatSequences) {
            PM4CmdSetData::SetUconfigReg(cmdbuf, 0x40000258u, 0x6d007fu); // IA_MULTI_VGT_PARAM
        } else {
            PM4CmdSetData::SetContextReg(cmdbuf, 0x100002aau, 0xd00ffu); // IA_MULTI_VGT_PARAM
        }
    } else {
        PM4CmdSetData::SetContextReg(cmdbuf, 0x2aau, 0xffu); // IA_MULTI_VGT_PARAM
    }
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSdmaClose() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaConstFill() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaCopyLinear() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaCopyTiled() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaCopyWindow() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaFlush() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaGetMinCmdSize() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSdmaOpen() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

s32 PS4_SYSV_ABI sceGnmSetCsShader(u32* cmdbuf, u32 size, const u32* cs_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x18) {
        return -1;
    }
    if (!cs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer in shader registers.");
        return -1;
    }
    if (cs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x20cu, cs_regs[0],
                                     0u); // COMPUTE_PGM_LO/COMPUTE_PGM_HI
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x212u, cs_regs[2],
                                     cs_regs[3]); // COMPUTE_PGM_RSRC1/COMPUTE_PGM_RSRC2
    cmdbuf = PM4CmdSetData::SetShReg(
        cmdbuf, 0x207u, cs_regs[4], cs_regs[5],
        cs_regs[6]); // COMPUTE_NUM_THREAD_X/COMPUTE_NUM_THREAD_Y/COMPUTE_NUM_THREAD_Z

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetCsShaderWithModifier(u32* cmdbuf, u32 size, const u32* cs_regs,
                                               u32 modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x18) {
        return -1;
    }
    if (!cs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer in shader registers.");
        return -1;
    }
    if ((modifier & 0xfffffc3fu) != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask.");
        return -1;
    }
    if (cs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x20cu, cs_regs[0],
                                     0u); // COMPUTE_PGM_LO/COMPUTE_PGM_HI
    const u32 rsrc1 = modifier == 0 ? cs_regs[2] : (cs_regs[2] & 0xfffffc3fu) | modifier;
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x212u, rsrc1,
                                     cs_regs[3]); // COMPUTE_PGM_RSRC1/COMPUTE_PGM_RSRC2
    cmdbuf = PM4CmdSetData::SetShReg(
        cmdbuf, 0x207u, cs_regs[4], cs_regs[5],
        cs_regs[6]); // COMPUTE_NUM_THREAD_X/COMPUTE_NUM_THREAD_Y/COMPUTE_NUM_THREAD_Z

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetEmbeddedPsShader(u32* cmdbuf, u32 size, u32 shader_id,
                                           u32 shader_modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (shader_id > 1) {
        LOG_ERROR(Lib_GnmDriver, "Unknown shader id {}", shader_id);
        return ORBIS_GNM_ERROR_FAILURE;
    }

    // clang-format off
    constexpr static std::array ps0_code alignas(256) = {
        0xbeeb03ffu, 0x00000003u, // s_mov_b32     vcc_hi, $0x00000003
        0x7e000280u,              // v_mov_b32     v0, 0
        0x5e000100u,              // v_cvt_pkrtz_f16_f32 v0, v0, v0
        0xbf800000u,              // s_nop
        0xf8001c0fu, 0x00000000u, // exp           mrt0, v0, v0 compr vm done
        0xbf810000u,              // s_endpgm

        // Binary header
        0x5362724fu, 0x07726468u, 0x00002043u, 0u, 0xb0a45b2bu, 0x1d39766du, 0x72044b7bu, 0x0000000fu,
        // PS regs
        0x0fe000f0u, 0u, 0xc0000u, 4u, 0u, 4u, 2u, 2u, 0u, 0u, 0x10u, 0xfu, 0xcu, 0u, 0u, 0u,
    };

    const auto shader0_addr = uintptr_t(ps0_code.data()); // Original address is 0xfe000f00
    const static u32 ps0_regs[] = {
        u32(shader0_addr >> 8), u32(shader0_addr >> 40), 0xc0000u, 4u, 0u, 4u, 2u, 2u, 0u, 0u, 0x10u, 0xfu, 0xcu};

    constexpr static std::array ps1_code alignas(256) = {
        0xbeeb03ffu, 0x00000003u, // s_mov_b32     vcc_hi, $0x00000003
        0x7e040280u,              // v_mov_b32     v2, 0 
        0xf8001803u, 0x02020202u, // exp           mrt0, v2, v2, off, off vm done
        0xbf810000u,              // s_endpgm

        // Binary header
        0x5362724fu, 0x07726468u, 0x00001841u, 0x04080002u, 0x98b9cb94u, 0u, 0x6f130734u, 0x0000000fu,
        // PS regs
        0x0fe000f2u, 0u, 0x2000u, 0u, 0u, 2u, 2u, 2u, 0u, 0u, 0x10u, 3u, 0xcu,
    };

    const auto shader1_addr = uintptr_t(ps1_code.data()); // Original address is 0xfe000f20
    const static u32 ps1_regs[] = {
        u32(shader1_addr >> 8), u32(shader1_addr >> 40), 0x2000u, 0u, 0u, 2u, 2u, 2u, 0u, 0u, 0x10u, 3u, 0xcu};
    // clang-format on

    const auto ps_regs = shader_id == 0 ? ps0_regs : ps1_regs;

    // Normally the driver will do a call to `sceGnmSetPsShader350()`, but this function has
    // a check for zero in the upper part of shader address. In our case, the address is a
    // pointer to a stack memory, so the check will likely fail. To workaround it we will
    // repeat set shader functionality here as it is trivial.
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                     ps_regs[1]); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                     ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1c4u, ps_regs[4],
                                          ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b3u, ps_regs[6],
                                          ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b6u, ps_regs[8]);  // SPI_PS_IN_CONTROL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b8u, ps_regs[9]);  // SPI_BARYC_CNTL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, ps_regs[10]); // DB_SHADER_CONTROL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, ps_regs[11]);  // CB_SHADER_MASK

    WriteTrailingNop<11>(cmdbuf);

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetEmbeddedVsShader(u32* cmdbuf, u32 size, u32 shader_id,
                                           u32 shader_modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (shader_id != 0) {
        LOG_ERROR(Lib_GnmDriver, "Unknown shader id {}", shader_id);
        return ORBIS_GNM_ERROR_FAILURE;
    }

    // A fullscreen triangle with one uv set
    // clang-format off
    constexpr static std::array shader_code alignas(256) = {
        0xbeeb03ffu, 0x00000007u, // s_mov_b32     vcc_hi, $0x00000007
        0x36020081u,              // v_and_b32     v1, 1, v0
        0x34020281u,              // v_lshlrev_b32 v1, 1, v1
        0x360000c2u,              // v_and_b32     v0, -2, v0
        0x4a0202c1u,              // v_add_i32     v1, vcc, -1, v1
        0x4a0000c1u,              // v_add_i32     v0, vcc, -1, v0
        0x7e020b01u,              // v_cvt_f32_i32 v1, v1
        0x7e000b00U,              // v_cvt_f32_i32 v0, v0
        0x7e040280u,              // v_mov_b32     v2, 0
        0x7e0602f2u,              // v_mov_b32     v3, 1.0
        0xf80008cfu, 0x03020001u, // exp           pos0, v1, v0, v2, v3 done
        0xf800020fu, 0x03030303u, // exp           param0, v3, v3, v3, v3
        0xbf810000u,              // s_endpgm

        // Binary header
        0x5362724fu, 0x07726468u, 0x00004047u, 0u, 0x47f8c29fu, 0x9b2da5cfu, 0xff7c5b7du, 0x00000017u,
        // VS regs
        0x0fe000f1u, 0u, 0x000c0000u, 4u, 0u, 4u, 0u, 7u,
    };
    // clang-format on

    const auto shader_addr = uintptr_t(shader_code.data()); // Original address is 0xfe000f10
    const static u32 vs_regs[] = {
        u32(shader_addr >> 8), u32(shader_addr >> 40), 0xc0000u, 4, 0, 4, 0, 7};

    // Normally the driver will do a call to `sceGnmSetVsShader()`, but this function has
    // a check for zero in the upper part of shader address. In our case, the address is a
    // pointer to a stack memory, so the check will likely fail. To workaround it we will
    // repeat set shader functionality here as it is trivial.
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x48u, vs_regs[0], vs_regs[1]); // SPI_SHADER_PGM_LO_VS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x4au, vs_regs[2],
                                     vs_regs[3]); // SPI_SHADER_PGM_RSRC1_VS/SPI_SHADER_PGM_RSRC2_VS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x207u, vs_regs[6]); // PA_CL_VS_OUT_CNTL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b1u, vs_regs[4]); // SPI_VS_OUT_CONFIG
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1c3u, vs_regs[5]); // SPI_SHADER_POS_FORMAT

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetEsShader(u32* cmdbuf, u32 size, const u32* es_regs, u32 shader_modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");
    const u32* cmdbuf_begin = cmdbuf;

    if (!cmdbuf || size < 0x14) {
        return -1;
    }

    if (!es_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (shader_modifier & 0xfcfffc3f) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask");
        return -1;
    }

    if (es_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    const u32 var =
        shader_modifier == 0 ? es_regs[2] : ((es_regs[2] & 0xfcfffc3f) | shader_modifier);
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0xc8u, es_regs[0], 0u);  // SPI_SHADER_PGM_LO_ES
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0xcau, var, es_regs[3]); // SPI_SHADER_PGM_RSRC1_ES

    WriteTrailingNop<11>(cmdbuf);
#ifdef __ANDROID__
    AmdGpu::RenderWaveTrace::BuilderShader(
        AmdGpu::RenderWaveTrace::Stage::Es, ExecutorRenderWaveCmdbufIdentity(cmdbuf_begin),
        reinterpret_cast<uintptr_t>(cmdbuf_begin), u64{es_regs[0]} << 8);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSetGsRingSizes() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetGsShader(u32* cmdbuf, u32 size, const u32* gs_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");
    const u32* cmdbuf_begin = cmdbuf;

    if (!cmdbuf || size < 0x1d) {
        return -1;
    }

    if (!gs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (gs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x88u, gs_regs[0], 0u); // SPI_SHADER_PGM_LO_GS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x8au, gs_regs[2],
                                     gs_regs[3]); // SPI_SHADER_PGM_RSRC1_GS/SPI_SHADER_PGM_RSRC2_GS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x2e5u, gs_regs[4]); // VGT_STRMOUT_CONFIG
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x29bu, gs_regs[5]); // VGT_GS_OUT_PRIM_TYPE
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x2e4u, gs_regs[6]); // VGT_GS_INSTANCE_CNT

    WriteTrailingNop<11>(cmdbuf);
#ifdef __ANDROID__
    AmdGpu::RenderWaveTrace::BuilderShader(
        AmdGpu::RenderWaveTrace::Stage::Gs, ExecutorRenderWaveCmdbufIdentity(cmdbuf_begin),
        reinterpret_cast<uintptr_t>(cmdbuf_begin), u64{gs_regs[0]} << 8);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetHsShader(u32* cmdbuf, u32 size, const u32* hs_regs, u32 param4) {
    LOG_TRACE(Lib_GnmDriver, "called");
    const u32* cmdbuf_begin = cmdbuf;
    if (!cmdbuf || size < 0x1E) {
        return -1;
    }

    if (!hs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (hs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x108u, hs_regs[0], 0u); // SPI_SHADER_PGM_LO_HS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x10au, hs_regs[2],
                                     hs_regs[3]); // SPI_SHADER_PGM_RSRC1_HS/SPI_SHADER_PGM_RSRC2_HS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x286u,
                                          hs_regs[5],                  // VGT_HOS_MAX_TESS_LEVEL
                                          hs_regs[6]);                 // VGT_HOS_MIN_TESS_LEVEL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x2dbu, hs_regs[4]); // VGT_TF_PARAM
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x2d6u, param4);     // VGT_LS_HS_CONFIG

    // right padding?
    WriteTrailingNop<11>(cmdbuf);
#ifdef __ANDROID__
    AmdGpu::RenderWaveTrace::BuilderShader(
        AmdGpu::RenderWaveTrace::Stage::Hs, ExecutorRenderWaveCmdbufIdentity(cmdbuf_begin),
        reinterpret_cast<uintptr_t>(cmdbuf_begin), u64{hs_regs[0]} << 8);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetLsShader(u32* cmdbuf, u32 size, const u32* ls_regs, u32 shader_modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");
    const u32* cmdbuf_begin = cmdbuf;

    if (!cmdbuf || size < 0x17) {
        return -1;
    }

    if (!ls_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    const auto modifier_mask = ((shader_modifier & 0xfffffc3f) == 0) ? 0xfffffc3f : 0xfcfffc3f;
    if (shader_modifier & modifier_mask) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask");
        return -1;
    }

    if (ls_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    const u32 var =
        shader_modifier == 0 ? ls_regs[2] : ((ls_regs[2] & modifier_mask) | shader_modifier);
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x148u, ls_regs[0], 0u);  // SPI_SHADER_PGM_LO_LS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x14bu, ls_regs[3]);      // SPI_SHADER_PGM_RSRC2_LS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x14au, var, ls_regs[3]); // SPI_SHADER_PGM_RSRC1_LS

    WriteTrailingNop<11>(cmdbuf);
#ifdef __ANDROID__
    AmdGpu::RenderWaveTrace::BuilderShader(
        AmdGpu::RenderWaveTrace::Stage::Ls, ExecutorRenderWaveCmdbufIdentity(cmdbuf_begin),
        reinterpret_cast<uintptr_t>(cmdbuf_begin), u64{ls_regs[0]} << 8);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetPsShader(u32* cmdbuf, u32 size, const u32* ps_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");
    const u32* cmdbuf_begin = cmdbuf;
#ifdef __ANDROID__
    if (ExecutorTraceLiveWide()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_GNM_CMD_WRITE] api=sceGnmSetPsShader cmdbuf=%p size=%u regs=%p pgmLo=0x%x rsrc1=0x%x rsrc2=0x%x",
            cmdbuf, size, ps_regs, ps_regs ? ps_regs[0] : 0u, ps_regs ? ps_regs[2] : 0u,
            ps_regs ? ps_regs[3] : 0u);
    }
#endif

    if (!cmdbuf || size <= 0x27) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmSetPsShader", "bad_size", cmdbuf, size, "",
                                  ExecutorLiveGnmBuilderKind::Shader);
#endif
        return -1;
    }
    if (!ps_regs) {
        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, 0u,
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, 0u); // DB_SHADER_CONTROL

        WriteTrailingNop<0x20>(cmdbuf);
    } else {
        if (ps_regs[1] != 0) {
            LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
#ifdef __ANDROID__
            ExecutorLiveGnmBuilderLog("sceGnmSetPsShader", "bad_shader_addr", cmdbuf, size, "",
                                      ExecutorLiveGnmBuilderKind::Shader);
#endif
            return -1;
        }

        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf =
            PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                    ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
        cmdbuf = PM4CmdSetData::SetContextReg(
            cmdbuf, 0x1c4u, ps_regs[4], ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b3u, ps_regs[6],
                                              ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b6u, ps_regs[8]);  // SPI_PS_IN_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b8u, ps_regs[9]);  // SPI_BARYC_CNTL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, ps_regs[10]); // DB_SHADER_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, ps_regs[11]);  // CB_SHADER_MASK

        WriteTrailingNop<11>(cmdbuf);
    }
#ifdef __ANDROID__
    AmdGpu::RenderWaveTrace::BuilderShader(
        AmdGpu::RenderWaveTrace::Stage::Ps, ExecutorRenderWaveCmdbufIdentity(cmdbuf_begin),
        reinterpret_cast<uintptr_t>(cmdbuf_begin), ps_regs ? u64{ps_regs[0]} << 8 : 0);
    ExecutorLiveGnmBuilderLog("sceGnmSetPsShader", "OK", cmdbuf, size,
                              ps_regs ? "regs=bound" : "regs=null",
                              ExecutorLiveGnmBuilderKind::Shader);
    ExecutorLiveDumpBuilderTail("sceGnmSetPsShader", cmdbuf_begin, size);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetPsShader350(u32* cmdbuf, u32 size, const u32* ps_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");
    const u32* cmdbuf_begin = cmdbuf;
#ifdef __ANDROID__
    if (ExecutorTraceLiveWide()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_GNM_CMD_WRITE] api=sceGnmSetPsShader350 cmdbuf=%p size=%u regs=%p pgmLo=0x%x rsrc1=0x%x rsrc2=0x%x",
            cmdbuf, size, ps_regs, ps_regs ? ps_regs[0] : 0u, ps_regs ? ps_regs[2] : 0u,
            ps_regs ? ps_regs[3] : 0u);
    }
#endif

    if (!cmdbuf || size <= 0x27) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmSetPsShader350", "bad_size", cmdbuf, size, "",
                                  ExecutorLiveGnmBuilderKind::Shader);
#endif
        return -1;
    }
    if (!ps_regs) {
        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, 0u,
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, 0u);  // DB_SHADER_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, 0xfu); // CB_SHADER_MASK

        WriteTrailingNop<0x1d>(cmdbuf);
    } else {
        if (ps_regs[1] != 0) {
            LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
#ifdef __ANDROID__
            ExecutorLiveGnmBuilderLog("sceGnmSetPsShader350", "bad_shader_addr", cmdbuf, size,
                                      "", ExecutorLiveGnmBuilderKind::Shader);
#endif
            return -1;
        }

        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf =
            PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                    ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
        cmdbuf = PM4CmdSetData::SetContextReg(
            cmdbuf, 0x1c4u, ps_regs[4], ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b3u, ps_regs[6],
                                              ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b6u, ps_regs[8]);  // SPI_PS_IN_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b8u, ps_regs[9]);  // SPI_BARYC_CNTL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x203u, ps_regs[10]); // DB_SHADER_CONTROL
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, ps_regs[11]);  // CB_SHADER_MASK

        WriteTrailingNop<11>(cmdbuf);
    }
#ifdef __ANDROID__
    AmdGpu::RenderWaveTrace::BuilderShader(
        AmdGpu::RenderWaveTrace::Stage::Ps, ExecutorRenderWaveCmdbufIdentity(cmdbuf_begin),
        reinterpret_cast<uintptr_t>(cmdbuf_begin), ps_regs ? u64{ps_regs[0]} << 8 : 0);
    ExecutorLiveGnmBuilderLog("sceGnmSetPsShader350", "OK", cmdbuf, size,
                              ps_regs ? "regs=bound" : "regs=null",
                              ExecutorLiveGnmBuilderKind::Shader);
    ExecutorLiveDumpBuilderTail("sceGnmSetPsShader350", cmdbuf_begin, size);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSetResourceRegistrationUserMemory() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSetResourceUserData() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSetSpiEnableSqCounters() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSetSpiEnableSqCountersForUnitInstance() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSetupMipStatsReport() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetVgtControl(u32* cmdbuf, u32 size, u32 prim_group_sz_minus_one,
                                     u32 partial_vs_wave_mode, u32 wd_switch_only_on_eop_mode) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size != 3 || (prim_group_sz_minus_one >= 0x100) ||
        ((wd_switch_only_on_eop_mode | partial_vs_wave_mode) >= 2)) {
        return -1;
    }

    if (sceKernelIsNeoMode()) {
        const u32 wd_switch_on_eop = u32(wd_switch_only_on_eop_mode != 0) << 0x14;
        const u32 switch_on_eoi = u32(wd_switch_only_on_eop_mode == 0) << 0x13;
        const u32 reg_value =
            wd_switch_only_on_eop_mode != 0
                ? (partial_vs_wave_mode & 1) << 0x10 | prim_group_sz_minus_one | wd_switch_on_eop |
                      switch_on_eoi | 0x40000u
                : prim_group_sz_minus_one & 0x1cffffu | wd_switch_on_eop | switch_on_eoi | 0x50000u;
        if (!UseNeoCompatSequences) {
            PM4CmdSetData::SetUconfigReg(cmdbuf, 0x40000258u,
                                         reg_value | 0x600000u); // IA_MULTI_VGT_PARAM
        } else {
            PM4CmdSetData::SetContextReg(cmdbuf, 0x100002aau, reg_value); // IA_MULTI_VGT_PARAM
        }
    } else {
        const u32 reg_value =
            ((partial_vs_wave_mode & 1) << 0x10) | (prim_group_sz_minus_one & 0xffffu);
        PM4CmdSetData::SetContextReg(cmdbuf, 0x2aau, reg_value); // IA_MULTI_VGT_PARAM
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSetVsShader(u32* cmdbuf, u32 size, const u32* vs_regs, u32 shader_modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");
    const u32* cmdbuf_begin = cmdbuf;
#ifdef __ANDROID__
    if (ExecutorTraceLiveWide()) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_GNM_CMD_WRITE] api=sceGnmSetVsShader cmdbuf=%p size=%u regs=%p pgmLo=0x%x rsrc1=0x%x rsrc2=0x%x modifier=0x%x",
            cmdbuf, size, vs_regs, vs_regs ? vs_regs[0] : 0u, vs_regs ? vs_regs[2] : 0u,
            vs_regs ? vs_regs[3] : 0u, shader_modifier);
    }
#endif

    if (!cmdbuf || size <= 0x1c) {
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmSetVsShader", "bad_size", cmdbuf, size, "",
                                  ExecutorLiveGnmBuilderKind::Shader);
#endif
        return -1;
    }

    if (!vs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmSetVsShader", "null_regs", cmdbuf, size, "",
                                  ExecutorLiveGnmBuilderKind::Shader);
#endif
        return -1;
    }

    if (shader_modifier & 0xfcfffc3f) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask");
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmSetVsShader", "bad_modifier", cmdbuf, size, "",
                                  ExecutorLiveGnmBuilderKind::Shader);
#endif
        return -1;
    }

    if (vs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
#ifdef __ANDROID__
        ExecutorLiveGnmBuilderLog("sceGnmSetVsShader", "bad_shader_addr", cmdbuf, size, "",
                                  ExecutorLiveGnmBuilderKind::Shader);
#endif
        return -1;
    }

    const u32 var = shader_modifier == 0 ? vs_regs[2] : (vs_regs[2] & 0xfcfffc3f) | shader_modifier;
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x48u, vs_regs[0], 0u);   // SPI_SHADER_PGM_LO_VS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x4au, var, vs_regs[3]);  // SPI_SHADER_PGM_RSRC1_VS
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x207u, vs_regs[6]); // PA_CL_VS_OUT_CNTL
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1b1u, vs_regs[4]); // SPI_VS_OUT_CONFIG
    cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x1c3u, vs_regs[5]); // SPI_SHADER_POS_FORMAT

    WriteTrailingNop<11>(cmdbuf);
#ifdef __ANDROID__
    AmdGpu::RenderWaveTrace::BuilderShader(
        AmdGpu::RenderWaveTrace::Stage::Vs, ExecutorRenderWaveCmdbufIdentity(cmdbuf_begin),
        reinterpret_cast<uintptr_t>(cmdbuf_begin), u64{vs_regs[0]} << 8);
    ExecutorLiveGnmBuilderLog("sceGnmSetVsShader", "OK", cmdbuf, size, "regs=bound",
                              ExecutorLiveGnmBuilderKind::Shader);
    ExecutorLiveDumpBuilderTail("sceGnmSetVsShader", cmdbuf_begin, size);
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSetWaveLimitMultiplier() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSetWaveLimitMultipliers() {
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmSpmEndSpm() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmInit() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmInit2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetDelay() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetMuxRam() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetMuxRam2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetSelectCounter() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetSpmSelects() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmSetSpmSelects2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSpmStartSpm() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttFini() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttFinishTrace() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetBcInfo() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetGpuClocks() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetHiWater() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetStatus() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetTraceCounter() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetTraceWptr() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetWrapCounts() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetWrapCounts2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttGetWritebackLabels() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttInit() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSelectMode() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSelectTarget() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSelectTokens() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetCuPerfMask() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetDceEventWrite() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetHiWater() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetTraceBuffer2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetTraceBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetUserData() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSetUserdataTimer() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttStartTrace() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttStopTrace() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSwitchTraceBuffer() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttSwitchTraceBuffer2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmSqttWaitForEvent() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

static inline s32 PatchFlipRequest(u32* cmdbuf, u32 size, u32 vo_handle, u32 buf_idx, u32 flip_mode,
                                   s64 flip_arg, void* unk) {
    // check for `prepareFlip` packet
    cmdbuf += size - 64;
    ASSERT_MSG(cmdbuf[0] == 0xc03e1000, "Can't find `prepareFlip` packet");

    std::array<u32, 7> backup{};
    std::memcpy(backup.data(), cmdbuf, backup.size() * sizeof(decltype(backup)::value_type));

    ASSERT_MSG(((backup[2] & 3) == 0u) || (backup[1] != PM4CmdNop::PayloadType::PrepareFlipLabel),
               "Invalid flip packet");
    ASSERT_MSG(buf_idx != 0xffff'ffffu, "Invalid VO buffer index");

    u32 effective_buf_idx = buf_idx;
    if (VideoOut::ExecutorResolveVoBufferIndex(static_cast<s32>(vo_handle), buf_idx,
                                               &effective_buf_idx)) {
#ifdef __ANDROID__
        if (effective_buf_idx != buf_idx) {
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_GNM_FLIP_INDEX_FIXUP] handle=%u requested=%u effective=%u",
                                vo_handle, buf_idx, effective_buf_idx);
        }
#endif
        buf_idx = effective_buf_idx;
    }

    const s32 flip_result = VideoOut::sceVideoOutSubmitEopFlip(vo_handle, buf_idx, flip_mode,
                                                               flip_arg, nullptr /*unk*/);
    if (flip_result != 0) {
        if (flip_result == 0x80290012) {
            LOG_ERROR(Lib_GnmDriver, "Flip queue is full");
            return 0x80d11081;
        } else {
            LOG_ERROR(Lib_GnmDriver, "Flip request failed");
            return flip_result;
        }
    }

    uintptr_t label_addr{};
    ASSERT_MSG(VideoOut::sceVideoOutGetBufferLabelAddress(vo_handle, &label_addr) == 16,
               "sceVideoOutGetBufferLabelAddress call failed");

    // Write event to lock the VO surface
    auto* write_lock = reinterpret_cast<PM4CmdWriteData*>(cmdbuf);
    write_lock->header = PM4Type3Header{PM4ItOpcode::WriteData, 3};
    write_lock->raw = 0x500u;
    const auto addr = (label_addr + buf_idx * sizeof(label_addr)) & ~0x3ull;
    write_lock->Address<uintptr_t>(addr);
    write_lock->data[0] = 1;
    VideoOut::ExecutorRegisterVoLabelAlias(vo_handle, addr, buf_idx);

    auto* nop = reinterpret_cast<PM4CmdNop*>(cmdbuf + 5);

    if (backup[1] == PM4CmdNop::PayloadType::PrepareFlip) {
        nop->header = PM4Type3Header{PM4ItOpcode::Nop, 0x39};
        nop->data_block[0] = PM4CmdNop::PayloadType::PatchedFlip;
    } else {
        if (backup[1] == PM4CmdNop::PayloadType::PrepareFlipLabel) {
            nop->header = PM4Type3Header{PM4ItOpcode::Nop, 0x34};
            nop->data_block[0] = PM4CmdNop::PayloadType::PatchedFlip;

            // Write event to update label
            auto* write_label = reinterpret_cast<PM4CmdWriteData*>(cmdbuf + 0x3b);
            write_label->header = PM4Type3Header{PM4ItOpcode::WriteData, 3};
            write_label->raw = 0x500u;
            write_label->dst_addr_lo = backup[2] & 0xffff'fffcu;
            write_label->dst_addr_hi = backup[3];
            write_label->data[0] = backup[4];
        }
        if (backup[1] == PM4CmdNop::PayloadType::PrepareFlipInterruptLabel) {
            nop->header = PM4Type3Header{PM4ItOpcode::Nop, 0x33};
            nop->data_block[0] = PM4CmdNop::PayloadType::PatchedFlip;

            auto* write_eop = reinterpret_cast<PM4CmdEventWriteEop*>(cmdbuf + 0x3a);
            write_eop->header = PM4Type3Header{PM4ItOpcode::EventWriteEop, 4};
            write_eop->event_control = (backup[5] & 0x3f) + 0x500u + (backup[6] & 0x3f) * 0x1000;
            write_eop->address_lo = backup[2] & 0xffff'fffcu;
            write_eop->data_control = (backup[3] & 0xffffu) | 0x2200'0000u;
            write_eop->data_lo = backup[4];
            write_eop->data_hi = 0u;
        }
        if (backup[1] == PM4CmdNop::PayloadType::PrepareFlipInterrupt) {
            nop->header = PM4Type3Header{PM4ItOpcode::Nop, 0x33};
            nop->data_block[0] = PM4CmdNop::PayloadType::PatchedFlip;

            auto* write_eop = reinterpret_cast<PM4CmdEventWriteEop*>(cmdbuf + 0x3a);
            write_eop->header = PM4Type3Header{PM4ItOpcode::EventWriteEop, 4};
            write_eop->event_control = (backup[5] & 0x3f) + 0x500u + (backup[6] & 0x3f) * 0x1000;
            write_eop->address_lo = 0u;
            write_eop->data_control = 0x100'0000u;
            write_eop->data_lo = 0u;
            write_eop->data_hi = 0u;
        }
    }

#ifdef __ANDROID__
    static std::atomic<u32> patched_flip_write_log_count{0};
    const u32 patched_flip_write_log =
        patched_flip_write_log_count.fetch_add(1, std::memory_order_relaxed);
    if (patched_flip_write_log < 8) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_PATCHED_FLIP_WRITTEN] ordinal=%u dcb=%p sizeDw=%u source=0x%x marker=0x%x count=%u buffer=%u",
            patched_flip_write_log + 1, cmdbuf - (size - 64), size, backup[1],
            nop->data_block[0], nop->header.count.Value(), buf_idx);
    }
#endif

    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSubmitAndFlipCommandBuffers(u32 count, u32* dcb_gpu_addrs[],
                                                   u32* dcb_sizes_in_bytes, u32* ccb_gpu_addrs[],
                                                   u32* ccb_sizes_in_bytes, u32 vo_handle,
                                                   u32 buf_idx, u32 flip_mode, s64 flip_arg) {
    return sceGnmSubmitAndFlipCommandBuffersForWorkload(
        count, count, dcb_gpu_addrs, dcb_sizes_in_bytes, ccb_gpu_addrs, ccb_sizes_in_bytes,
        vo_handle, buf_idx, flip_mode, flip_arg);
}

s32 PS4_SYSV_ABI sceGnmSubmitAndFlipCommandBuffersForWorkload(
    u32 workload, u32 count, u32* dcb_gpu_addrs[], u32* dcb_sizes_in_bytes, u32* ccb_gpu_addrs[],
    u32* ccb_sizes_in_bytes, u32 vo_handle, u32 buf_idx, u32 flip_mode, s64 flip_arg) {
    LOG_DEBUG(Lib_GnmDriver, "called [buf = {}]", buf_idx);
    LOG_INFO(Lib_GnmDriver,
             "EXECUTOR_GNM_FLIP_ATTEMPT api=sceGnmSubmitAndFlipCommandBuffersForWorkload "
             "workload={} count={} vo_handle={} buf_idx={} flip_mode={} flip_arg={}",
             workload, count, vo_handle, buf_idx, flip_mode, flip_arg);
    LOG_INFO(Lib_GnmDriver,
             "EXECUTOR_GNM_FLIP api=sceGnmSubmitAndFlipCommandBuffersForWorkload workload={} "
             "count={} vo_handle={} buf_idx={} flip_mode={} flip_arg={}",
             workload, count, vo_handle, buf_idx, flip_mode, flip_arg);

    auto* cmdbuf = dcb_gpu_addrs[count - 1];
    const auto size_dw = dcb_sizes_in_bytes[count - 1] / 4;

    const s32 patch_result =
        PatchFlipRequest(cmdbuf, size_dw, vo_handle, buf_idx, flip_mode, flip_arg, nullptr /*unk*/);
    if (patch_result != ORBIS_OK) {
        return patch_result;
    }

    return sceGnmSubmitCommandBuffers(count, const_cast<const u32**>(dcb_gpu_addrs),
                                      dcb_sizes_in_bytes, const_cast<const u32**>(ccb_gpu_addrs),
                                      ccb_sizes_in_bytes);
}

// EXECUTOR (Codex layer3 2026-06-02): public wrapper so the native runtime can probe presenter bring-up
// in isolation (no PM4 submit). EnsureGnmPresenter is file-local (anonymous namespace); this forwards.
bool ExecutorEnsureLiverpoolForReplay() {
    if (liverpool) {
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_REPLAY_LIVERPOOL] result=already");
#endif
        return true;
    }
    try {
        liverpool = std::make_unique<AmdGpu::Liverpool>();
        if (Config::copyGPUCmdBuffers()) {
            liverpool->ReserveCopyBufferSpace();
        }
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_REPLAY_LIVERPOOL] result=created");
#endif
        return true;
    } catch (const std::exception& e) {
        liverpool.reset();
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_REPLAY_LIVERPOOL] result=failed error=%s", e.what());
#endif
        return false;
    }
}

bool ExecutorEnsureGnmPresenter(const char* reason) {
    return EnsureGnmPresenter(reason);
}

// EXECUTOR .gnmcap replay: present the replay's color render target on the device screen (blit RT ->
// swapchain). Call after the replay submits while ExecutorReplayActive is still true (the VideoOut
// present thread is blocked then, so no concurrent present), before freeing replay memory.
bool ExecutorPresentReplayRT() {
#ifdef __ANDROID__
    g_live_present_rt_requests.fetch_add(1, std::memory_order_relaxed);
#endif
    if (presenter) {
        return presenter->ExecutorPresentReplayRT();
    }
    return false;
}

ExecutorGnmHudStats ExecutorGetGnmHudStats() noexcept {
#ifdef __ANDROID__
    return {
        .builder_total = g_live_gnm_builder_total.load(std::memory_order_relaxed),
        .actual_draw = g_live_actual_draws.load(std::memory_order_relaxed),
        .builder_shader = g_live_gnm_builder_shader.load(std::memory_order_relaxed),
        .submit_calls = g_live_submit_calls.load(std::memory_order_relaxed),
        .submit_done_calls = g_live_submit_done_calls.load(std::memory_order_relaxed),
        .present_requests = g_live_present_rt_requests.load(std::memory_order_relaxed),
    };
#else
    return {};
#endif
}

bool ExecutorLivePresentRtEnabled() {
#ifdef __ANDROID__
    // The marker is converted to an environment flag before the GPU/presenter is created. This
    // predicate is used in every draw and coroutine resume, so cache it instead of serializing all
    // GPU threads through bionic getenv().
    static const bool enabled = std::getenv("EXECUTOR_LIVE_PRESENT_RT") != nullptr;
    return enabled;
#else
    return false;
#endif
}

bool ExecutorLiveDirectPresentActive() {
#ifdef __ANDROID__
    return presenter && presenter->ExecutorLiveDirectPresentActive();
#else
    return false;
#endif
}

// EXECUTOR (Codex phase-2): capture-with-memory session. A submit opens a session (DCB/CCB + Regs);
// the Vulkan rasterizer, while drawing that submit on the GPU thread, records every guest memory range
// it actually reads (shaders, VB/IB/constants, textures, render targets) via the recorder API below.
// The session is finalized (written to <dir>/live-submit-<n>.gnmcap) when the NEXT submit arrives, so a
// real game frame becomes a replayable .gnmcap with all referenced memory. Capture is PC-writer only
// (addresses are valid host pointers there); on Android it stays off.
static bool g_gnm_capture_on_submit = false;
static std::string g_gnm_capture_dir;
static u32 g_gnm_capture_seq = 0;
static constexpr u32 kGnmCaptureMax = 8;
static constexpr u64 kGnmMaxRangeBytes = 16ull * 1024 * 1024;  // skip absurd single ranges

namespace {
struct CapRange {
    u64 addr{};
    u64 size{};
    u32 usage{};
    std::vector<u8> bytes;
};
struct CapShader {
    u32 stage{};
    u64 hash{};
    std::vector<u8> code;
};
struct CapSession {
    std::mutex mtx;
    bool active = false;
    u32 submit_id = 0;
    u32 draw_count = 0;
    std::vector<u32> dcb;
    std::vector<u32> ccb;
    std::vector<CapRange> ranges;
    std::vector<CapShader> shaders;
};
CapSession g_cap;

// Add a memory range to the active session. Caller MUST hold g_cap.mtx and have verified g_cap.active.
void RecordRangeUnlocked(u32 usage, u64 guest_addr, u64 size) {
    if (guest_addr == 0 || size == 0 || size > kGnmMaxRangeBytes) {
        return;
    }
    for (auto& r : g_cap.ranges) {  // dedupe: skip if already covered
        if (guest_addr >= r.addr && guest_addr + size <= r.addr + r.size) {
            return;
        }
    }
    CapRange r;
    r.addr = guest_addr;
    r.size = size;
    r.usage = usage;
    r.bytes.resize(static_cast<size_t>(size));
    std::memcpy(r.bytes.data(), reinterpret_cast<const void*>(static_cast<uintptr_t>(guest_addr)),
                static_cast<size_t>(size));
    g_cap.ranges.push_back(std::move(r));
}

// Build + serialize + write the current session, then clear it. Caller holds g_cap.mtx.
void FinalizeSessionLocked() {
    using namespace Executor::GnmCap;
    if (!g_cap.active || g_cap.dcb.empty()) {
        g_cap.active = false;
        return;
    }
    Capture cap;
    u32 idx = 1;  // section 0 = Submit
    const u32 dcb_idx = idx++;
    const u32 ccb_idx = g_cap.ccb.empty() ? 0xFFFFFFFFu : idx++;
    {
        auto& s = cap.AddSection(SectionType::Submit);
        SubmitRec rec{g_cap.submit_id, g_cap.ccb.empty() ? 1u : 2u, dcb_idx, ccb_idx};
        Capture::AppendPod(s.payload, rec);
    }
    {
        auto& s = cap.AddSection(SectionType::QueueBlob);
        QueueBlobHdr h{static_cast<uint32_t>(QueueType::Dcb), 0ull,
                       static_cast<uint32_t>(g_cap.dcb.size()), 0u};
        Capture::AppendPod(s.payload, h);
        Capture::AppendBytes(s.payload, g_cap.dcb.data(), g_cap.dcb.size() * sizeof(u32));
    }
    if (!g_cap.ccb.empty()) {
        auto& s = cap.AddSection(SectionType::QueueBlob);
        QueueBlobHdr h{static_cast<uint32_t>(QueueType::Ccb), 0ull,
                       static_cast<uint32_t>(g_cap.ccb.size()), 0u};
        Capture::AppendPod(s.payload, h);
        Capture::AppendBytes(s.payload, g_cap.ccb.data(), g_cap.ccb.size() * sizeof(u32));
    }
    for (const auto& sh : g_cap.shaders) {
        auto& s = cap.AddSection(SectionType::Shader);
        ShaderHdr h{sh.hash, sh.stage, static_cast<uint32_t>(sh.code.size()), 0u, 0u};
        Capture::AppendPod(s.payload, h);
        Capture::AppendBytes(s.payload, sh.code.data(), sh.code.size());
    }
    for (const auto& r : g_cap.ranges) {
        auto& s = cap.AddSection(SectionType::MemoryRange);
        MemoryRangeHdr h{r.addr, r.size, r.usage, 0u, 0ull, 0ull};
        Capture::AppendPod(s.payload, h);
        Capture::AppendBytes(s.payload, r.bytes.data(), r.bytes.size());
    }
    const std::vector<u8> bytes = Serialize(cap);
    char path[512];
    std::snprintf(path, sizeof(path), "%s/live-submit-%u.gnmcap", g_gnm_capture_dir.c_str(),
                  g_cap.submit_id);
    if (FILE* f = std::fopen(path, "wb")) {
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
        LOG_INFO(Lib_GnmDriver,
                 "EXECUTOR_GNM_CAPTURE finalize submit={} draws={} dcbDw={} shaders={} ranges={} "
                 "bytes={} path={}",
                 g_cap.submit_id, g_cap.draw_count, g_cap.dcb.size(), g_cap.shaders.size(),
                 g_cap.ranges.size(), bytes.size(), path);
    }
    g_cap.active = false;
    g_cap.dcb.clear();
    g_cap.ccb.clear();
    g_cap.ranges.clear();
    g_cap.shaders.clear();
    g_cap.draw_count = 0;
}
}  // namespace

void ExecutorSetGnmCaptureOnSubmit(bool enable, const char* dir) {
    g_gnm_capture_on_submit = enable;
    g_gnm_capture_dir = (enable && dir) ? dir : "";
    g_gnm_capture_seq = 0;
    LOG_INFO(Lib_GnmDriver, "EXECUTOR_GNM_CAPTURE config enable={} dir={}", enable, g_gnm_capture_dir);
}

bool ExecutorGnmCaptureActive() {
    return g_cap.active;
}

// --- Replay reloc-miss reporter (Codex phase-2) -----------------------------------------------------
namespace {
struct ReplayReloc {
    u64 old_base;
    u64 new_base;
    u64 size;
};
std::mutex g_replay_mtx;
bool g_replay_active = false;
std::vector<ReplayReloc> g_replay_relocs;
u32 g_replay_miss_count = 0;
}  // namespace

// Block until the GPU coroutine has fully consumed all submitted command buffers (submission_lock==0).
// The .gnmcap replay MUST call this after sceGnmSubmit* and before freeing the replay backing memory --
// the submit only enqueues work, so munmap/ClearReplayMemory while the GPU is still drawing is a
// use-after-free race (the source of the non-deterministic replay crashes).
void ExecutorReplayWaitGpuIdle() {
    if (!liverpool) {
        return;
    }
    // Per Codex replay sync contract: close the submit batch (triggers OnSubmit/Flush), then wait for the
    // PM4 coroutine (num_submits==0) AND the Vulkan queue (rasterizer->Finish). sceGnmSubmit* only
    // enqueues the ProcessGraphics task, so without this the replay frees overlay memory mid-draw (race).
    liverpool->SubmitDone();
    liverpool->WaitRasterizerIdleForExecutor();
}

void ExecutorReplaySetActive(bool active) {
    std::lock_guard lk(g_replay_mtx);
    g_replay_active = active;
    if (active) {
        g_replay_relocs.clear();
        g_replay_miss_count = 0;
    }
}

void ExecutorReplayAddReloc(u64 old_base, u64 new_base, u64 size) {
    std::lock_guard lk(g_replay_mtx);
    g_replay_relocs.push_back({old_base, new_base, size});
}

bool ExecutorReplayActive() {
    return g_replay_active;
}

bool ExecutorReplayForcePsConst() {
    static const bool v = [] {
        const char* e = std::getenv("EXECUTOR_FORCE_PS_CONST");
        return e && e[0] == '1';
    }();
    return v;
}

bool ExecutorReplayIsRelocated(u64 addr) {
    if (!g_replay_active || addr == 0) {
        return false;
    }
    std::lock_guard lk(g_replay_mtx);
    for (const auto& r : g_replay_relocs) {
        if (addr >= r.new_base && addr < r.new_base + r.size) {
            return true;
        }
    }
    return false;
}

// Translate a captured (old) guest address to its rebased (new) arena address. Used at descriptor
// use-sites (e.g. T# image base read from flattened_ud_buf, which is assembled at recompile time and
// therefore never passed through the DCB/table relocation passes). Returns 0 if addr is not inside any
// captured old range. If addr is already a rebased address, returns it unchanged.
u64 ExecutorReplayRelocate(u64 addr) {
    if (!g_replay_active || addr == 0) {
        return 0;
    }
    std::lock_guard lk(g_replay_mtx);
    // Descriptor bases (T#/V# from flattened_ud_buf) are ALWAYS captured (old) addresses, so translate
    // old->new FIRST. The mmap arena can overlap the original 0x7c... VA region, so an old address may
    // coincidentally land inside some unrelated new range -- checking "already rebased" first would then
    // wrongly return it unchanged. Old-range translation must win.
    for (const auto& r : g_replay_relocs) {
        if (addr >= r.old_base && addr < r.old_base + r.size) {
            return r.new_base + (addr - r.old_base);
        }
    }
    for (const auto& r : g_replay_relocs) {
        if (addr >= r.new_base && addr < r.new_base + r.size) {
            return addr;  // genuinely already rebased (not in any old range)
        }
    }
    return 0;
}

void ExecutorReplayCheckAddr(u32 usage, u64 addr, u64 size) {
    if (!g_replay_active || addr == 0) {
        return;
    }
    std::lock_guard lk(g_replay_mtx);
    if (!g_replay_active) {
        return;
    }
    for (const auto& r : g_replay_relocs) {
        if (addr >= r.new_base && addr < r.new_base + r.size) {
            return;  // already a rebased address
        }
        if (addr >= r.old_base && addr < r.old_base + r.size) {
            return;  // a captured old address (will be / should be rebased)
        }
    }
    if (g_replay_miss_count < 64) {  // cap the spam
        ++g_replay_miss_count;
        LOG_ERROR(Lib_GnmDriver,
                  "EXECUTOR_GNMCAP_RELOC_MISS usage={} addr=0x{:x} size={} (resource not captured/"
                  "relocated -- capture this on PC or patch its descriptor)",
                  usage, addr, size);
    }
}

void ExecutorGnmCaptureOnDraw() {
    if (g_cap.active) {
        ++g_cap.draw_count;
    }
}

void ExecutorGnmRecordActualDraw(u64 count) noexcept {
#ifdef __ANDROID__
    g_live_actual_draws.fetch_add(count, std::memory_order_relaxed);
#else
    (void)count;
#endif
}

// Record a guest memory range the rasterizer is about to read for the current draw. On the PC writer
// the address is a valid host pointer (shadPS4 maps PS4 memory), so copy the bytes for replay. usage:
// 0=shader 1=vertex 2=index 3=constant 4=texture 5=rendertarget.
void ExecutorGnmCaptureRecordResource(u32 usage, u64 guest_addr, u64 size) {
    if (!g_cap.active || guest_addr == 0 || size == 0 || size > kGnmMaxRangeBytes) {
        return;
    }
    std::lock_guard lk(g_cap.mtx);
    if (!g_cap.active) {
        return;
    }
    RecordRangeUnlocked(usage, guest_addr, size);
}

void ExecutorGnmCaptureRecordShader(u32 stage, u64 hash, u64 guest_addr, u32 code_bytes) {
    if (!g_cap.active || guest_addr == 0 || code_bytes == 0 || code_bytes > kGnmMaxRangeBytes) {
        return;
    }
    std::lock_guard lk(g_cap.mtx);
    if (!g_cap.active) {
        return;
    }
    for (auto& sh : g_cap.shaders) {
        if (sh.hash == hash && sh.stage == stage) {
            return;
        }
    }
    CapShader sh;
    sh.stage = stage;
    sh.hash = hash;
    sh.code.resize(code_bytes);
    std::memcpy(sh.code.data(), reinterpret_cast<const void*>(static_cast<uintptr_t>(guest_addr)),
                code_bytes);
    g_cap.shaders.push_back(std::move(sh));
    // Also record the raw shader memory range so replay can rebase the program address. Include a margin
    // PAST the code so the trailing BinaryInfo footer (which GetParams/SearchBinaryInfo scans for at
    // code_end on replay) is present -- otherwise the replay GetParams walks past the mapped range and
    // SIGSEGVs during pipeline compile.
    RecordRangeUnlocked(0u, guest_addr, static_cast<u64>(code_bytes) + 1024u);
}

// Open a session for this submit (finalize the previous one first). PC-writer only.
static void ExecutorCaptureSubmitGnmcap(std::span<const u32> dcb_span, std::span<const u32> ccb_span) {
    static std::once_flag env_once;
    static u32 g_gnm_capture_skip = 0;   // EXECUTOR_GNM_CAPTURE_SKIP: skip the first N submits (so a
    static u32 g_gnm_submit_seen = 0;    // longer game run can capture menu/gameplay frames, not loading)
    std::call_once(env_once, [] {
        if (!g_gnm_capture_on_submit) {
            if (const char* dir = std::getenv("EXECUTOR_GNM_CAPTURE_DIR"); dir && dir[0]) {
                g_gnm_capture_on_submit = true;
                g_gnm_capture_dir = dir;
                g_gnm_capture_seq = 0;
                if (const char* sk = std::getenv("EXECUTOR_GNM_CAPTURE_SKIP"); sk && sk[0]) {
                    g_gnm_capture_skip = static_cast<u32>(std::strtoul(sk, nullptr, 10));
                }
                LOG_INFO(Lib_GnmDriver, "EXECUTOR_GNM_CAPTURE env-enabled dir={} skip={}", dir,
                         g_gnm_capture_skip);
            }
        }
    });
    if (!g_gnm_capture_on_submit || g_gnm_capture_dir.empty() || dcb_span.empty()) {
        return;
    }

#ifdef __ANDROID__
    // Android live bring-up runs inside the app process and may see only a single submit before the game
    // blocks/exits. The phase-2 PC writer finalizes a capture on the NEXT submit because rasterizer hooks
    // append shader/memory ranges while the current submit draws; that is correct for PC captures but it
    // means a lone Android submit never leaves an artifact. For device validation, write an immediate
    // DCB/CCB+Regs .gnmcap here. This is intentionally DCB-only (memoryComplete=NO), so it does not copy
    // arbitrary guest pointers with reinterpret_cast on Android.
    if (g_gnm_submit_seen++ < g_gnm_capture_skip) {
        return;
    }
    if (g_gnm_capture_seq >= kGnmCaptureMax) {
        return;
    }
    const u32 submit_id = g_gnm_capture_seq++;
    const std::vector<u8> bytes =
        liverpool ? liverpool->ExecutorCaptureGnmcapFrame(dcb_span, ccb_span) : std::vector<u8>{};
    if (!bytes.empty()) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/live-submit-%u.gnmcap", g_gnm_capture_dir.c_str(),
                      submit_id);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(bytes.data(), 1, bytes.size(), f);
            std::fclose(f);
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_GNM_CAPTURE_ANDROID] submit=%u dcbDw=%zu ccbDw=%zu "
                                "bytes=%zu path=%s memoryComplete=NO",
                                submit_id, dcb_span.size(), ccb_span.size(), bytes.size(), path);
        } else {
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_GNM_CAPTURE_ANDROID] submit=%u write_failed path=%s",
                                submit_id, path);
        }
    }
    return;
#endif

    std::lock_guard lk(g_cap.mtx);
    if (g_cap.active) {
        FinalizeSessionLocked();  // write the previous submit (its draws have run)
    }
    if (g_gnm_submit_seen++ < g_gnm_capture_skip) {
        return;  // still in the skipped (loading) phase -- don't open a capture session yet
    }
    if (g_gnm_capture_seq >= kGnmCaptureMax) {
        return;
    }
    g_cap.submit_id = g_gnm_capture_seq++;
    g_cap.dcb.assign(dcb_span.begin(), dcb_span.end());
    g_cap.ccb.assign(ccb_span.begin(), ccb_span.end());
    g_cap.ranges.clear();
    g_cap.shaders.clear();
    g_cap.draw_count = 0;
    g_cap.active = true;
}

int PS4_SYSV_ABI sceGnmSubmitCommandBuffersForWorkload(u32 workload, u32 count,
                                                       const u32* dcb_gpu_addrs[],
                                                       u32* dcb_sizes_in_bytes,
                                                       const u32* ccb_gpu_addrs[],
                                                       u32* ccb_sizes_in_bytes) {
    HLE_TRACE;
    LOG_DEBUG(Lib_GnmDriver, "called");
#ifdef __ANDROID__
    g_live_submit_calls.fetch_add(1, std::memory_order_relaxed);
    ExecutorLiveFrameStats("submit_enter", frames_submitted);
    if (ExecutorTraceLiveWide()) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_SUBMIT] workload=%u count=%u dcb=%p dcb_sizes=%p ccb=%p ccb_sizes=%p",
                            workload, count, dcb_gpu_addrs, dcb_sizes_in_bytes, ccb_gpu_addrs,
                            ccb_sizes_in_bytes);
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_GNM_BUILDER_COUNTS] total=%llu draw=%llu shader=%llu submitdone=%llu",
            static_cast<unsigned long long>(g_live_gnm_builder_total.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_live_gnm_builder_draw.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(g_live_gnm_builder_shader.load(std::memory_order_relaxed)),
            static_cast<unsigned long long>(
                g_live_gnm_builder_submitdone.load(std::memory_order_relaxed)));
    }
    if (ExecutorTraceLiveWide() && dcb_gpu_addrs != nullptr && dcb_sizes_in_bytes != nullptr) {
        for (u32 i = 0; i < count && i < 8; ++i) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_SUBMIT_DCB] i=%u addr=%p bytes=%u", i,
                                dcb_gpu_addrs[i], dcb_sizes_in_bytes[i]);
        }
    }
#endif
    LOG_DEBUG(Lib_GnmDriver,
              "EXECUTOR_GNM_SUBMIT_ATTEMPT api=sceGnmSubmitCommandBuffersForWorkload workload={} "
              "count={}",
              workload, count);
    LOG_DEBUG(Lib_GnmDriver,
              "EXECUTOR_GNM_DRAW api=sceGnmSubmitCommandBuffersForWorkload workload={} count={}",
              workload, count);

    if (!dcb_gpu_addrs || !dcb_sizes_in_bytes) {
        LOG_ERROR(Lib_GnmDriver, "dcbGpuAddrs and dcbSizesInBytes must not be NULL");
        return 0x80d11000;
    }

    for (u32 i = 0; i < count; i++) {
        if (dcb_sizes_in_bytes[i] == 0) {
            LOG_ERROR(Lib_GnmDriver, "Submitting a null DCB {}", i);
            return 0x80d11000;
        }
        if (dcb_sizes_in_bytes[i] > 0x3ffffc) {
            LOG_ERROR(Lib_GnmDriver, "dcbSizesInBytes[{}] ({}) is limited to (2*20)-1 DWORDS", i,
                      dcb_sizes_in_bytes[i]);
            return 0x80d11000;
        }
        if (ccb_sizes_in_bytes && ccb_sizes_in_bytes[i] > 0x3ffffc) {
            LOG_ERROR(Lib_GnmDriver, "ccbSizesInBytes[{}] ({}) is limited to (2*20)-1 DWORDS", i,
                      ccb_sizes_in_bytes[i]);
            return 0x80d11000;
        }
    }

    // Lazily bring up the Vulkan presenter on the first real GNM GPU work. Android defers presenter
    // creation so Piglet/GLES keeps the surface until a GNM game actually submits. Do not gate the
    // submit on the result: if creation fails (e.g. no window yet) the existing `if (!presenter)`
    // fallbacks in VideoOutDriver still apply, so no working path regresses. (Codex layer3 design)
#ifdef __ANDROID__
    const bool live_sync_submit = ExecutorLiveGnmSyncSubmit();
    const bool live_gpu_trace = ExecutorTraceLiveWide();
    ExecutorLogLiveGnmStage("before_presenter", workload, 0, 0, 0);
#endif
    const bool presenter_ready = EnsureGnmPresenter("sceGnmSubmitCommandBuffersForWorkload");
#ifdef __ANDROID__
    if (live_gpu_trace) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_GNM_STAGE] stage=after_presenter workload=%u ready=%u sync=%u",
            workload, presenter_ready ? 1u : 0u, live_sync_submit ? 1u : 0u);
    }
    ExecutorLogLiveGnmStage("before_submission_lock_idle", workload, 0, 0, 0);
#endif

    if (!WaitGpuIdle("sceGnmSubmitCommandBuffersForWorkload")) {
        return ORBIS_OK;
    }
#ifdef __ANDROID__
    ExecutorLogLiveGnmStage("after_submission_lock_idle", workload, 0, 0, 0);

    auto submit_gfx_live = [&](std::span<const u32> submit_dcb, std::span<const u32> submit_ccb,
                               const char* label, u32 cbpair) {
        AmdGpu::ExecutorEopTraceHleSubmit(submit_dcb, submit_ccb, label, workload, cbpair);
        if (live_gpu_trace) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_LIVE_GNM_STAGE] stage=before_%s workload=%u cbpair=%u dcbDw=%zu ccbDw=%zu",
                label ? label : "submit", workload, cbpair, submit_dcb.size(), submit_ccb.size());
        }
        liverpool->SubmitGfx(submit_dcb, submit_ccb);
        if (live_gpu_trace) {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_STAGE] stage=queued_%s workload=%u cbpair=%u",
                                label ? label : "submit", workload, cbpair);
        }
        if (live_sync_submit) {
            if (live_gpu_trace) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_GNM_STAGE] stage=before_drain_%s workload=%u cbpair=%u",
                    label ? label : "submit", workload, cbpair);
            }
            liverpool->WaitRasterizerIdleForExecutor();
            if (live_gpu_trace) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_GNM_STAGE] stage=after_drain_%s workload=%u cbpair=%u",
                    label ? label : "submit", workload, cbpair);
            }
        }
    };
#else
    auto submit_gfx_live = [&](std::span<const u32> submit_dcb, std::span<const u32> submit_ccb,
                               const char*, u32) { liverpool->SubmitGfx(submit_dcb, submit_ccb); };
#endif

    if (DebugState.ShouldPauseInSubmit()) {
        DebugState.PauseGuestThreads();
    }

    if (send_init_packet) {
        if (sdk_version < Common::ElfInfo::FW_20) {
            submit_gfx_live(std::span<const u32>{InitSequence}, {}, "init_fw_lt20", 0);
        } else if (sdk_version < Common::ElfInfo::FW_40) {
            if (sceKernelIsNeoMode()) {
                if (!UseNeoCompatSequences) {
                    submit_gfx_live(std::span<const u32>{InitSequence200Neo}, {},
                                    "init_fw20_neo", 0);
                } else {
                    submit_gfx_live(std::span<const u32>{InitSequence200NeoCompat}, {},
                                    "init_fw20_neo_compat", 0);
                }
            } else {
                submit_gfx_live(std::span<const u32>{InitSequence200}, {}, "init_fw20", 0);
            }
        } else {
            if (sceKernelIsNeoMode()) {
                if (!UseNeoCompatSequences) {
                    submit_gfx_live(std::span<const u32>{InitSequence350Neo}, {},
                                    "init_fw35_neo", 0);
                } else {
                    submit_gfx_live(std::span<const u32>{InitSequence350NeoCompat}, {},
                                    "init_fw35_neo_compat", 0);
                }
            } else {
                submit_gfx_live(std::span<const u32>{InitSequence350}, {}, "init_fw35", 0);
            }
        }
        send_init_packet = false;
    }

    for (auto cbpair = 0u; cbpair < count; ++cbpair) {
        const auto* ccb = ccb_gpu_addrs ? ccb_gpu_addrs[cbpair] : nullptr;
        const auto ccb_size_in_bytes = ccb_sizes_in_bytes ? ccb_sizes_in_bytes[cbpair] : 0;

        const auto dcb_size_dw = dcb_sizes_in_bytes[cbpair] >> 2;
        const auto ccb_size_dw = ccb_size_in_bytes >> 2;

        const auto& dcb_span = std::span{dcb_gpu_addrs[cbpair], dcb_size_dw};
        const auto& ccb_span = std::span{ccb, ccb_size_dw};
#ifdef __ANDROID__
        char dcb_tag[64]{};
        std::snprintf(dcb_tag, sizeof(dcb_tag), "submit_work%u_pair%u", workload, cbpair);
        ExecutorLiveClassifyDcb(dcb_tag, dcb_gpu_addrs[cbpair], dcb_size_dw);
        ExecutorLiveRememberExplicitDcb(dcb_gpu_addrs[cbpair],
                                        ExecutorLiveAnalyzeDcb(dcb_gpu_addrs[cbpair], dcb_size_dw));
        if (ccb && ccb_size_dw != 0) {
            char ccb_tag[64]{};
            std::snprintf(ccb_tag, sizeof(ccb_tag), "submit_work%u_pair%u_ccb", workload,
                          cbpair);
            ExecutorLiveDumpDwords(ccb_tag, ccb, std::min<u32>(ccb_size_dw, 16u));
        }
#endif

        if (DebugState.DumpingCurrentFrame()) {
            static auto last_frame_num = -1LL;
            static u32 seq_num{};
            if (last_frame_num == frames_submitted && cbpair == 0) {
                ++seq_num;
            } else {
                last_frame_num = frames_submitted;
                seq_num = 0u;
            }

            using DebugStateType::QueueType;

            DebugState.PushQueueDump({
                .type = QueueType::dcb,
                .submit_num = seq_num,
                .num2 = cbpair,
                .data = {dcb_span.begin(), dcb_span.end()},
                .base_addr = reinterpret_cast<uintptr_t>(dcb_gpu_addrs[cbpair]),
            });
            DebugState.PushQueueDump({
                .type = QueueType::ccb,
                .submit_num = seq_num,
                .num2 = cbpair,
                .data = {ccb_span.begin(), ccb_span.end()},
                .base_addr = reinterpret_cast<uintptr_t>(ccb),
            });
        }
        // EXECUTOR (Codex step 2/3): capture this real submit to a replayable .gnmcap before it runs.
        ExecutorCaptureSubmitGnmcap(dcb_span, ccb_span);
        submit_gfx_live(dcb_span, ccb_span, "live_submit", cbpair);
    }

#ifdef __ANDROID__
    ExecutorLogLiveGnmStage("before_return_ok", workload, 0, 0, 0);
#endif
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmSubmitCommandBuffers(u32 count, const u32* dcb_gpu_addrs[],
                                            u32* dcb_sizes_in_bytes, const u32* ccb_gpu_addrs[],
                                            u32* ccb_sizes_in_bytes) {
    return sceGnmSubmitCommandBuffersForWorkload(count, count, dcb_gpu_addrs, dcb_sizes_in_bytes,
                                                 ccb_gpu_addrs, ccb_sizes_in_bytes);
}

int PS4_SYSV_ABI sceGnmSubmitDone() {
    HLE_TRACE;
    LOG_DEBUG(Lib_GnmDriver, "called");
#ifdef __ANDROID__
    // Register FEX guest stacks/TLS as Boehm GC roots so the conservative mark stops sweeping live
    // mono objects (the stochastic scene-build NullReferenceException root). Self-gated by the
    // run-gc-guest-roots marker + idempotent; runs on this guest-serving thread during loading frames.
    if (executor_lsx4_register_guest_gc_roots) {
        executor_lsx4_register_guest_gc_roots();
    }
    if (executor_mono_value_copy_dump) {
        executor_mono_value_copy_dump();
    }
    g_live_submit_done_calls.fetch_add(1, std::memory_order_relaxed);
    ExecutorLiveFrameStats("submit_done_enter", frames_submitted);
    static std::atomic<bool> mono_sigsegv_guard_requested{false};
    if (!mono_sigsegv_guard_requested.exchange(true) && std::getenv("MONO_DEBUG") &&
        std::strstr(std::getenv("MONO_DEBUG"), "explicit-null-checks") &&
        executor_lsx4_android_install_mono_explicit_sigsegv_guard) {
        executor_lsx4_android_install_mono_explicit_sigsegv_guard();
    }
    static std::atomic<u64> submit_done_log_count{0};
    const u64 submit_done_call = ++submit_done_log_count;
    if (submit_done_call == 64 &&
        std::getenv("EXECUTOR_LIVE_STALL_THREAD_DUMP") != nullptr) {
        std::thread([] {
            std::this_thread::sleep_for(std::chrono::seconds(6));
            if (executor_jit_dump_thread_states) {
                executor_jit_dump_thread_states("delayed-submitdone-stall");
            }
            if (executor_live_dump_mutex_wait_ledger) {
                executor_live_dump_mutex_wait_ledger("delayed-submitdone-stall", 64);
            }
        }).detach();
    }
    AmdGpu::ExecutorEopTraceSubmitDonePulse(submit_done_call);
    const bool submit_done_log =
        submit_done_call <= 16 || (submit_done_call & (submit_done_call - 1)) == 0 ||
        std::getenv("EXECUTOR_TRACE_LIVE_WIDE") != nullptr;
    if (submit_done_log) {
        if (executor_mono_vtnull_dump) {
            executor_mono_vtnull_dump();
        }
        if (executor_mono_vcall_entry_dump) {
            executor_mono_vcall_entry_dump();
        }
        const std::uint64_t imt_dyn =
            executor_imt_overflow_count ? executor_imt_overflow_count() : 0;
        const std::uint64_t imt_alloc_overflow =
            executor_imt_alloc_overflow_count ? executor_imt_alloc_overflow_count() : 0;
        const std::uint64_t imt_tail_overflow =
            executor_imt_tail_overflow_count ? executor_imt_tail_overflow_count() : 0;
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_SUBMIT_DONE] call=%llu stage=enter gpuIdle=%u "
                            "frames=%llu builderTotal=%llu builderDraw=%llu builderShader=%llu "
                            "builderSubmit=%llu imtDyn=%llu imtAllocOverflow=%llu "
                            "imtTailOverflow=%llu",
                            static_cast<unsigned long long>(submit_done_call),
                            liverpool && liverpool->IsGpuIdle() ? 1u : 0u,
                            static_cast<unsigned long long>(frames_submitted),
                            static_cast<unsigned long long>(
                                g_live_gnm_builder_total.load(std::memory_order_relaxed)),
                            static_cast<unsigned long long>(
                                g_live_gnm_builder_draw.load(std::memory_order_relaxed)),
                            static_cast<unsigned long long>(
                                g_live_gnm_builder_shader.load(std::memory_order_relaxed)),
                            static_cast<unsigned long long>(
                                g_live_gnm_builder_submitdone.load(std::memory_order_relaxed)),
                            static_cast<unsigned long long>(imt_dyn),
                            static_cast<unsigned long long>(imt_alloc_overflow),
                            static_cast<unsigned long long>(imt_tail_overflow));
    }
    // Guest-RIP pulse for the Unity/Game/mono threads: SubmitDone is the one reliable heartbeat
    // through the whole boot (Thread3 calls it every few seconds even while everything else is
    // stalled), so this exposes the poll sites of a mutual polling standoff with no contention.
    if (executor_live_dump_unity_thread_rips && submit_done_call >= 12 &&
        submit_done_call <= 24 && (submit_done_call % 4) == 0 &&
        g_live_gnm_builder_draw.load(std::memory_order_relaxed) == 0) {
        executor_live_dump_unity_thread_rips("submitdone", static_cast<int>(submit_done_call));
    }
    // One narrow synchronization-frontier sample. Run it from an observer after the tenth
    // heartbeat because a guest synchronization stall can prevent all subsequent SubmitDone calls.
    // The flight recorder itself remains opt-in through its runtime marker.
    static std::atomic<bool> sync_frontier_dump_started{false};
    if (executor_live_hle_flight_dump && submit_done_call == 10 &&
        g_live_gnm_builder_draw.load(std::memory_order_relaxed) == 0 &&
        !sync_frontier_dump_started.exchange(true, std::memory_order_relaxed)) {
        std::thread([] {
            std::this_thread::sleep_for(std::chrono::seconds(8));
            executor_live_hle_flight_dump(
                "submitdone-sync-frontier",
                g_live_gnm_builder_total.load(std::memory_order_relaxed),
                g_live_gnm_builder_draw.load(std::memory_order_relaxed),
                g_live_gnm_builder_shader.load(std::memory_order_relaxed),
                reinterpret_cast<u64>(g_live_active_cmdbuf.load(std::memory_order_relaxed)),
                g_live_active_cmdbuf_size.load(std::memory_order_relaxed));
            if (executor_live_dump_kernel_sem_records) {
                executor_live_dump_kernel_sem_records("submitdone-sync-frontier");
            }
            if (executor_live_dump_posix_sem_records) {
                executor_live_dump_posix_sem_records("submitdone-sync-frontier");
            }
            std::this_thread::sleep_for(std::chrono::seconds(47));
            executor_live_hle_flight_dump(
                "submitdone-late-sync-frontier",
                g_live_gnm_builder_total.load(std::memory_order_relaxed),
                g_live_gnm_builder_draw.load(std::memory_order_relaxed),
                g_live_gnm_builder_shader.load(std::memory_order_relaxed),
                reinterpret_cast<u64>(g_live_active_cmdbuf.load(std::memory_order_relaxed)),
                g_live_active_cmdbuf_size.load(std::memory_order_relaxed));
            if (executor_live_dump_kernel_sem_records) {
                executor_live_dump_kernel_sem_records("submitdone-late-sync-frontier");
            }
            if (executor_live_dump_posix_sem_records) {
                executor_live_dump_posix_sem_records("submitdone-late-sync-frontier");
            }
        }).detach();
    }
    if (std::getenv("EXECUTOR_LIVE_STALL_THREAD_DUMP") != nullptr &&
        executor_live_dump_mutex_wait_ledger != nullptr) {
        executor_live_dump_mutex_wait_ledger("submitdone",
                                             static_cast<int>(submit_done_call));
    }
    if ((std::getenv("EXECUTOR_LIVE_STALL_THREAD_DUMP") || ExecutorTraceLiveWide()) &&
        executor_jit_dump_thread_states &&
        g_live_gnm_builder_draw.load(std::memory_order_relaxed) == 0 &&
        ExecutorLiveShouldPulse(submit_done_call)) {
        if (executor_live_hle_flight_dump) {
            executor_live_hle_flight_dump(
                "submitdone-no-workload-draw",
                g_live_gnm_builder_total.load(std::memory_order_relaxed),
                g_live_gnm_builder_draw.load(std::memory_order_relaxed),
                g_live_gnm_builder_shader.load(std::memory_order_relaxed),
                reinterpret_cast<u64>(g_live_active_cmdbuf.load(std::memory_order_relaxed)),
                g_live_active_cmdbuf_size.load(std::memory_order_relaxed));
        }
        executor_jit_dump_thread_states("submitdone-no-workload-draw");
    }
    if (std::getenv("EXECUTOR_TRACE_LIVE_EMU_DUMP") &&
        executor_lsx4_android_dump_box64_emu_states) {
        executor_lsx4_android_dump_box64_emu_states("gnm-submitdone-enter");
    }
    if (executor_live_dump_posix_sem_records &&
        (std::getenv("EXECUTOR_TRACE_LIVE_SYNC") ||
         std::getenv("EXECUTOR_TRACE_LIVE_WIDE"))) {
        executor_live_dump_posix_sem_records("gnm-submitdone-enter");
    }
    if (ExecutorTraceLiveWide() && ExecutorLiveShouldPulse(submit_done_call)) {
        const auto active_cb =
            reinterpret_cast<const u32*>(g_live_active_cmdbuf.load(std::memory_order_relaxed));
        const auto active_size = g_live_active_cmdbuf_size.load(std::memory_order_relaxed);
        if (active_cb && active_size != 0) {
            ExecutorLiveClassifyDcb("active_builder_before_submitdone", active_cb,
                                    std::min<u32>(active_size, 4096u));
        } else {
            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_ACTIVE_CB] stage=submitdone none=1");
        }
    }
    if (std::getenv("EXECUTOR_LIVE_STALL_THREAD_DUMP") &&
        executor_lsx4_android_dump_box64_emu_states) {
        const auto active_cb =
            reinterpret_cast<const u32*>(g_live_active_cmdbuf.load(std::memory_order_relaxed));
        const auto active_size = g_live_active_cmdbuf_size.load(std::memory_order_relaxed);
        const auto active =
            ExecutorLiveAnalyzeDcb(active_cb, std::min<u32>(active_size, 4096u));
        const bool likely_stalled_after_first_submit =
            submit_done_call >= 2 &&
            g_live_gnm_builder_total.load(std::memory_order_relaxed) >= 8 &&
            g_live_gnm_builder_draw.load(std::memory_order_relaxed) == 0;
        const bool active_builder_has_draw =
            active_cb && active.valid_dwords != 0 && active.draws != 0;
        if ((likely_stalled_after_first_submit || active_builder_has_draw) &&
            ExecutorLiveShouldPulse(submit_done_call)) {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Native",
                "[EXECUTOR_LIVE_STALL_THREAD_DUMP] call=%llu activeCb=%p activeDw=%u "
                "activeDraws=%u activeHash=0x%llx builderTotal=%llu builderDraw=%llu",
                static_cast<unsigned long long>(submit_done_call), active_cb, active.valid_dwords,
                active.draws, static_cast<unsigned long long>(active.hash),
                static_cast<unsigned long long>(
                    g_live_gnm_builder_total.load(std::memory_order_relaxed)),
                static_cast<unsigned long long>(
                    g_live_gnm_builder_draw.load(std::memory_order_relaxed)));
            executor_lsx4_android_dump_box64_emu_states("live-stall-after-first-submit");
        }
    }
    const auto active_cb =
        reinterpret_cast<const u32*>(g_live_active_cmdbuf.load(std::memory_order_relaxed));
    const auto active_size = g_live_active_cmdbuf_size.load(std::memory_order_relaxed);
    const auto analysis = ExecutorLiveAnalyzeDcb(active_cb, std::min<u32>(active_size, 4096u));
    const u64 previous_hash = g_live_autosubmitted_hash.load(std::memory_order_relaxed);
    // PC oracle for CUSA00754 submits the completed frame DCB before SubmitDone. On Android/FEX the
    // Unity worker can reach SubmitDone with that same active builder DCB already complete but never
    // passed through sceGnmSubmit*. SubmitDone is the guest's ordering boundary, so drain exactly one
    // new active draw DCB here instead of letting it spin forever as an unsubmitted command stream.
    // JIT can issue an early init-only explicit submit and then build the first draw DCB in a
    // different buffer without passing that buffer through sceGnmSubmit*. A process-wide "any submit"
    // bit incorrectly suppresses the real frame forever. Suppress only an exact DCB that was already
    // submitted (same base, parsed length, and contents). Preserve Backend A's existing global gate.
    const bool guest_used_explicit_submit =
        g_live_submit_calls.load(std::memory_order_relaxed) != 0;
    const bool jit_active = executor_lsx4_android_runtime_jit_active &&
                                  executor_lsx4_android_runtime_jit_active() != 0;
    const bool active_dcb_already_submitted = ExecutorLiveWasExplicitlySubmitted(active_cb, analysis);
    // Reference JIT never turns SubmitDone into an implicit SubmitGfx. Replaying Sonic's
    // mutable DCB here queues a second copy which observes the next frame's rewritten WAIT_REG_MEM.
    const bool suppress_active_dcb =
        jit_active || active_dcb_already_submitted || guest_used_explicit_submit;
    if (!suppress_active_dcb && active_cb && analysis.draws != 0 &&
        analysis.valid_dwords != 0 && analysis.hash != previous_hash) {
        Core::g_executor_render_tid.store(static_cast<int>(gettid()), std::memory_order_relaxed);
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_LIVE_GNM_SUBMITDONE_DRAIN] stage=enter cb=%p validDw=%u draws=%u "
            "packets=%u hash=0x%llx previous=0x%llx call=%llu",
            active_cb, analysis.valid_dwords, analysis.draws, analysis.packets,
            static_cast<unsigned long long>(analysis.hash),
            static_cast<unsigned long long>(previous_hash),
            static_cast<unsigned long long>(submit_done_call));
        const bool presenter_ready = EnsureGnmPresenter("submitdone-drain-active-cb");
        if (presenter_ready && liverpool) {
            LOG_INFO(Lib_GnmDriver,
                     "EXECUTOR_GNM_DRAW api=sceGnmSubmitDoneDrainActiveCb count=1 dwords={} "
                     "draws={}",
                     analysis.valid_dwords, analysis.draws);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_GNM_DRAW] api=sceGnmSubmitDoneDrainActiveCb count=1 dwords=%u draws=%u",
                analysis.valid_dwords, analysis.draws);
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_SUBMITDONE_DRAIN] stage=before_submitgfx");
            liverpool->SubmitGfx(std::span<const u32>{active_cb, analysis.valid_dwords}, {});
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_SUBMITDONE_DRAIN] stage=before_waitidle");
            liverpool->WaitRasterizerIdleForExecutor();
            g_live_autosubmitted_hash.store(analysis.hash, std::memory_order_relaxed);
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_LIVE_GNM_SUBMITDONE_DRAIN] stage=done hash=0x%llx",
                                static_cast<unsigned long long>(analysis.hash));
        } else {
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Native",
                "[EXECUTOR_LIVE_GNM_SUBMITDONE_DRAIN] stage=skip presenterReady=%u liverpool=%u",
                presenter_ready ? 1u : 0u, liverpool ? 1u : 0u);
        }
    }
#endif
    if (!WaitGpuIdle("sceGnmSubmitDone")) {
        // Renderer terminal is a contained host failure. Preserve the guest ABI success while
        // refusing to enqueue more work into the wedged VkDevice.
        send_init_packet = true;
        return ORBIS_OK;
    }
    if (!liverpool->IsGpuIdle()) {
        submission_lock = true;
    }
    liverpool->SubmitDone();
#ifdef __ANDROID__
    if (executor_live_dump_posix_sem_records &&
        (std::getenv("EXECUTOR_TRACE_LIVE_SYNC") ||
         std::getenv("EXECUTOR_TRACE_LIVE_WIDE"))) {
        executor_live_dump_posix_sem_records("gnm-submitdone-after-submitdone");
    }
#endif
    send_init_packet = true;
    ++frames_submitted;
    DebugState.IncGnmFrameNum();
#ifdef __ANDROID__
    // Live direct presentation is owned exclusively by Liverpool's GPU thread after OnSubmit/Flush.
    // Calling it here raced the coroutine and could sample draws before they were submitted.
#endif
#ifdef __ANDROID__
    if (submit_done_log) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_GNM_SUBMIT_DONE] call=%llu stage=return frames=%llu",
                            static_cast<unsigned long long>(submit_done_call),
                            static_cast<unsigned long long>(frames_submitted));
    }
    if (std::getenv("EXECUTOR_TRACE_LIVE_EMU_DUMP") &&
        executor_lsx4_android_dump_box64_emu_states) {
        executor_lsx4_android_dump_box64_emu_states("gnm-submitdone-return");
    }
#endif
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmUnmapComputeQueue(u32 vqid) {
    liverpool->asc_queues.erase(Common::SlotId{vqid - 1});
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmUnregisterAllResourcesForOwner() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmUnregisterOwnerAndResources() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmUnregisterResource() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

s32 PS4_SYSV_ABI sceGnmUpdateGsShader(u32* cmdbuf, u32 size, const u32* gs_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size < 0x1d) {
        return -1;
    }

    if (!gs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (gs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x88u, gs_regs[0], 0u); // SPI_SHADER_PGM_LO_GS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x8au, gs_regs[2],
                                     gs_regs[3]); // SPI_SHADER_PGM_RSRC1_GS/SPI_SHADER_PGM_RSRC2_GS
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e02e5u,
                                           gs_regs[4]);
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e029bu,
                                           gs_regs[5]);
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e02e4u,
                                           gs_regs[6]);

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmUpdateHsShader(u32* cmdbuf, u32 size, const u32* hs_regs, u32 ls_hs_config) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x1c) {
        return -1;
    }

    if (!hs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (hs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x108u, hs_regs[0],
                                     0u); // SPI_SHADER_PGM_LO_HS/SPI_SHADER_PGM_HI_HS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x10au, hs_regs[2],
                                     hs_regs[3]); // SPI_SHADER_PGM_RSRC1_HS/SPI_SHADER_PGM_RSRC1_LS
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(
        cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0286u, hs_regs[5],
        hs_regs[6]); // VGT_HOS_MAX_TESS_LEVEL/VGT_HOS_MIN_TESS_LEVEL update
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e02dbu,
                                           hs_regs[4]); // VGT_TF_PARAM update
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e02d6u,
                                           ls_hs_config); // VGT_LS_HS_CONFIG update

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmUpdatePsShader(u32* cmdbuf, u32 size, const u32* ps_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x27) {
        return -1;
    }
    if (!ps_regs) {
        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, 0u,
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0203u,
                                               0u); // DB_SHADER_CONTROL update
        WriteTrailingNop<0x20>(cmdbuf);
    } else {
        if (ps_regs[1] != 0) {
            LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
            return -1;
        }

        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf =
            PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                    ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(
            cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01c4u, ps_regs[4],
            ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(
            cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b3u, ps_regs[6],
            ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b6u,
                                               ps_regs[8]); // SPI_PS_IN_CONTROL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b8u,
                                               ps_regs[9]); // SPI_BARYC_CNTL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0203u,
                                               ps_regs[10]); // DB_SHADER_CONTROL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e008fu,
                                               ps_regs[11]); // CB_SHADER_MASK update

        WriteTrailingNop<11>(cmdbuf);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmUpdatePsShader350(u32* cmdbuf, u32 size, const u32* ps_regs) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x27) {
        return -1;
    }
    if (!ps_regs) {
        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, 0u,
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0203u,
                                               0u);                 // DB_SHADER_CONTROL update
        cmdbuf = PM4CmdSetData::SetContextReg(cmdbuf, 0x8fu, 0xfu); // CB_SHADER_MASK

        WriteTrailingNop<0x1d>(cmdbuf);
    } else {
        if (ps_regs[1] != 0) {
            LOG_ERROR(Lib_GnmDriver, "Invalid shader address.");
            return -1;
        }

        cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 8u, ps_regs[0],
                                         0u); // SPI_SHADER_PGM_LO_PS/SPI_SHADER_PGM_HI_PS
        cmdbuf =
            PM4CmdSetData::SetShReg(cmdbuf, 10u, ps_regs[2],
                                    ps_regs[3]); // SPI_SHADER_PGM_RSRC1_PS/SPI_SHADER_PGM_RSRC2_PS
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(
            cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01c4u, ps_regs[4],
            ps_regs[5]); // SPI_SHADER_Z_FORMAT/SPI_SHADER_COL_FORMAT update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(
            cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b3u, ps_regs[6],
            ps_regs[7]); // SPI_PS_INPUT_ENA/SPI_PS_INPUT_ADDR update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b6u,
                                               ps_regs[8]); // SPI_PS_IN_CONTROL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b8u,
                                               ps_regs[9]); // SPI_BARYC_CNTL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0203u,
                                               ps_regs[10]); // DB_SHADER_CONTROL update
        cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e008fu,
                                               ps_regs[11]); // CB_SHADER_MASK update

        WriteTrailingNop<11>(cmdbuf);
    }
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmUpdateVsShader(u32* cmdbuf, u32 size, const u32* vs_regs,
                                      u32 shader_modifier) {
    LOG_TRACE(Lib_GnmDriver, "called");

    if (!cmdbuf || size <= 0x1c) {
        return -1;
    }

    if (!vs_regs) {
        LOG_ERROR(Lib_GnmDriver, "Null pointer passed as argument");
        return -1;
    }

    if (shader_modifier & 0xfcfffc3f) {
        LOG_ERROR(Lib_GnmDriver, "Invalid modifier mask");
        return -1;
    }

    if (vs_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "Invalid shader address");
        return -1;
    }

    const u32 var =
        shader_modifier == 0 ? vs_regs[2] : ((vs_regs[2] & 0xfcfffc3f) | shader_modifier);
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x48u, vs_regs[0], 0u);  // SPI_SHADER_PGM_LO_VS
    cmdbuf = PM4CmdSetData::SetShReg(cmdbuf, 0x4au, var, vs_regs[3]); // SPI_SHADER_PGM_RSRC1_VS
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e0207u,
                                           vs_regs[6]); // PA_CL_VS_OUT_CNTL update
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01b1u,
                                           vs_regs[4]); // PA_CL_VS_OUT_CNTL update
    cmdbuf = WritePacket<PM4ItOpcode::Nop>(cmdbuf, PM4ShaderType::ShaderGraphics, 0xc01e01c3u,
                                           vs_regs[5]); // PA_CL_VS_OUT_CNTL update

    WriteTrailingNop<11>(cmdbuf);
    return ORBIS_OK;
}

s32 PS4_SYSV_ABI sceGnmValidateCommandBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateDisableDiagnostics() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateDisableDiagnostics2() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateDispatchCommandBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateDrawCommandBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateGetDiagnosticInfo() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateGetDiagnostics() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidateGetVersion() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return 0;
}

bool PS4_SYSV_ABI sceGnmValidateOnSubmitEnabled() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

int PS4_SYSV_ABI sceGnmValidateResetState() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceGnmValidationRegisterMemoryCheckCallback() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_VALIDATION_NOT_ENABLED;
}

int PS4_SYSV_ABI sceRazorCaptureCommandBuffersOnlyImmediate() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_FAILED_INTERNAL;
}

int PS4_SYSV_ABI sceRazorCaptureCommandBuffersOnlySinceLastFlip() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_FAILED_INTERNAL;
}

int PS4_SYSV_ABI sceRazorCaptureImmediate() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_FAILED_INTERNAL;
}

int PS4_SYSV_ABI sceRazorCaptureSinceLastFlip() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_CAPTURE_FAILED_INTERNAL;
}

bool PS4_SYSV_ABI sceRazorIsLoaded() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return false;
}

int PS4_SYSV_ABI Func_063D065A2D6359C3() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_063D065A2D6359C3");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_0CABACAFB258429D() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_0CABACAFB258429D");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_150CF336FC2E99A3() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_150CF336FC2E99A3");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_17CA687F9EE52D49() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_17CA687F9EE52D49");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_1870B89F759C6B45() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_1870B89F759C6B45");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_26F9029EF68A955E() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_26F9029EF68A955E");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_301E3DBBAB092DB0() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_301E3DBBAB092DB0");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_30BAFE172AF17FEF() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_30BAFE172AF17FEF");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_3E6A3E8203D95317() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_3E6A3E8203D95317");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_40FEEF0C6534C434() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_40FEEF0C6534C434");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_416B9079DE4CBACE() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_416B9079DE4CBACE");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_4774D83BB4DDBF9A() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_4774D83BB4DDBF9A");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_50678F1CCEEB9A00() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_50678F1CCEEB9A00");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_54A2EC5FA4C62413() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_54A2EC5FA4C62413");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_5A9C52C83138AE6B() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_5A9C52C83138AE6B");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_5D22193A31EA1142() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_5D22193A31EA1142");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_725A36DEBB60948D() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_725A36DEBB60948D");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_8021A502FA61B9BB() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_8021A502FA61B9BB");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_9D002FE0FA40F0E6() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_9D002FE0FA40F0E6");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_9D297F36A7028B71() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_9D297F36A7028B71");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_A2D7EC7A7BCF79B3() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_A2D7EC7A7BCF79B3");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_AA12A3CB8990854A() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_AA12A3CB8990854A");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_ADC8DDC005020BC6() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_ADC8DDC005020BC6");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_B0A8688B679CB42D() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_B0A8688B679CB42D");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_B489020B5157A5FF() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_B489020B5157A5FF");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_BADE7B4C199140DD() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_BADE7B4C199140DD");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_D1511B9DCFFB3DD9() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_D1511B9DCFFB3DD9");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_D53446649B02E58E() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_D53446649B02E58E");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_D8B6E8E28E1EF0A3() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_D8B6E8E28E1EF0A3");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_D93D733A19DD7454() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_D93D733A19DD7454");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_DE995443BC2A8317() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_DE995443BC2A8317");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_DF6E9528150C23FF() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_DF6E9528150C23FF");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_ECB4C6BA41FE3350() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_ECB4C6BA41FE3350");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmDebugModuleReset() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmDebugReset() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI Func_C4C328B7CF3B4171() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_C4C328B7CF3B4171");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI sceGnmDrawInitToDefaultContextStateInternalCommand(u32* cmdbuf, u32 size) {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (sdk_version >= Common::ElfInfo::FW_40) {
        return sceGnmDrawInitToDefaultContextState400(cmdbuf, size);
    }
    return sceGnmDrawInitToDefaultContextState(cmdbuf, size);
}

int PS4_SYSV_ABI sceGnmDrawInitToDefaultContextStateInternalSize() {
    LOG_TRACE(Lib_GnmDriver, "called");
    if (sdk_version >= Common::ElfInfo::FW_40) {
        return 0x100;
    }
    return 0x20;
}

int PS4_SYSV_ABI sceGnmFindResources() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmGetResourceRegistrationBuffers() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI sceGnmRegisterOwnerForSystem() {
    LOG_TRACE(Lib_GnmDriver, "called");
    // Not available in retail firmware
    return ORBIS_GNM_ERROR_FAILURE;
}

int PS4_SYSV_ABI Func_1C43886B16EE5530() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_1C43886B16EE5530");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_81037019ECCD0E01() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_81037019ECCD0E01");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_BFB41C057478F0BF() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_BFB41C057478F0BF");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_E51D44DB8151238C() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_E51D44DB8151238C");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_F916890425496553() {
#ifdef __ANDROID__
    ExecutorLiveGnmUnknownLog("Func_F916890425496553");
#endif
    LOG_ERROR(Lib_GnmDriver, "(STUBBED) called");
    return ORBIS_OK;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LOG_INFO(Lib_GnmDriver, "Initializing presenter");
    liverpool = std::make_unique<AmdGpu::Liverpool>();
#if defined(__ANDROID__) && defined(EXECUTOR_ANDROID_NATIVE_CORE_PROBE_NO_DESKTOP_MEDIA_USB)
    const bool jit_active = executor_lsx4_android_runtime_jit_active &&
                                  executor_lsx4_android_runtime_jit_active() != 0;
    if (jit_active) {
        // JIT executes the guest concurrently with Liverpool's graphics coroutine. Copy
        // DCB/CCB bytes at submit time so Unity cannot recycle their guest storage while the GPU
        // thread is still parsing it. Menu-sized submits often hide this lifetime race; the first
        // FlusterCluck gameplay frame reliably exposes it as an Adreno device loss.
        Config::setCopyGPUCmdBuffers(true);
        // Match the reference lifecycle: Rasterizer exists before the guest creates its GPU-visible
        // direct-memory mappings, so MemoryManager callbacks populate mapped_ranges in time.
        presenter = std::make_unique<Vulkan::Presenter>(*g_window, liverpool.get());
        LOG_INFO(Lib_GnmDriver,
                 "EXECUTOR_VULKAN_WSI phase=presenter_eager result=OK "
                 "reason=jit_reference");
    } else {
    // Do NOT create the Vulkan presenter eagerly here. Its Swapchain ctor connects the
    // ANativeWindow to the Vulkan WSI API, which steals the surface from Piglet (the PS4
    // GLES driver that Piglet-based homebrew like Itemzflow renders through). When the
    // presenter grabbed the window first, Piglet's eglCreateWindowSurface failed with
    // EGL_BAD_ALLOC ("already connected to another API") and the launcher stayed black.
    // Defer presenter creation to the first real GNM submit (EnsureGnmPresenter), so
    // Piglet-only guests keep ownership of the surface and GNM games still get a presenter
    // lazily when they actually issue GPU work.
    LOG_INFO(Lib_GnmDriver,
             "EXECUTOR_VULKAN_WSI phase=presenter_deferred result=lazy "
             "reason=avoid_piglet_surface_conflict windowBridgeReady={}",
             g_window != nullptr ? "YES" : "NO");
    }
#else
    presenter = std::make_unique<Vulkan::Presenter>(*g_window, liverpool.get());
#endif

    const s32 result = sceKernelGetCompiledSdkVersion(&sdk_version);
    if (result != ORBIS_OK) {
        sdk_version = 0;
    }

    if (Config::copyGPUCmdBuffers()) {
        liverpool->ReserveCopyBufferSpace();
    }

    Platform::IrqC::Instance()->Register(Platform::InterruptId::GpuIdle, ResetSubmissionLock,
                                         nullptr);

    LIB_FUNCTION("b0xyllnVY-I", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmAddEqEvent);
    LIB_FUNCTION("b08AgtPlHPg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmAreSubmitsAllowed);
    LIB_FUNCTION("ihxrbsoSKWc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmBeginWorkload);
    LIB_FUNCTION("ffrNQOshows", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmComputeWaitOnAddress);
    LIB_FUNCTION("EJapNl2+pgU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmComputeWaitSemaphore);
    LIB_FUNCTION("5udAm+6boVg", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmCreateWorkloadStream);
    LIB_FUNCTION("jwCEzr7uEP4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerGetAddressWatch);
    LIB_FUNCTION("PNf0G7gvFHQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerHaltWavefront);
    LIB_FUNCTION("nO-tMnaxJiE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDebuggerReadGds);
    LIB_FUNCTION("t0HIQWnvK9E", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerReadSqIndirectRegister);
    LIB_FUNCTION("HsLtF4jKe48", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerResumeWavefront);
    LIB_FUNCTION("JRKSSV0YzwA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerResumeWavefrontCreation);
    LIB_FUNCTION("jpTMyYB8UBI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerSetAddressWatch);
    LIB_FUNCTION("MJG69Q7ti+s", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDebuggerWriteGds);
    LIB_FUNCTION("PaFw9w6f808", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDebuggerWriteSqIndirectRegister);
    LIB_FUNCTION("qpGITzPE+Zc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDebugHardwareStatus);
    LIB_FUNCTION("PVT+fuoS9gU", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDeleteEqEvent);
    LIB_FUNCTION("UtObDRQiGbs", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDestroyWorkloadStream);
    LIB_FUNCTION("bX5IbRvECXk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDingDong);
    LIB_FUNCTION("byXlqupd8cE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDingDongForWorkload);
    LIB_FUNCTION("HHo1BAljZO8", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDisableMipStatsReport);
    LIB_FUNCTION("0BzLGljcwBo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDispatchDirect);
    LIB_FUNCTION("Z43vKp5k7r0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDispatchIndirect);
    LIB_FUNCTION("wED4ZXCFJT0", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDispatchIndirectOnMec);
    LIB_FUNCTION("nF6bFRUBRAU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDispatchInitDefaultHardwareState);
    LIB_FUNCTION("HlTPoZ-oY7Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndex);
    LIB_FUNCTION("GGsn7jMTxw4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndexAuto);
    LIB_FUNCTION("ED9-Fjr8Ta4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndexIndirect);
    LIB_FUNCTION("thbPcG7E7qk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawIndexIndirectCountMulti);
    LIB_FUNCTION("5q95ravnueg", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawIndexIndirectMulti);
    LIB_FUNCTION("jHdPvIzlpKc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawIndexMultiInstanced);
    LIB_FUNCTION("oYM+YzfCm2Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndexOffset);
    LIB_FUNCTION("4v+otIIdjqg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndirect);
    LIB_FUNCTION("cUCo8OvArrw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawIndirectCountMulti);
    LIB_FUNCTION("f5QQLp9rzGk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawIndirectMulti);
    LIB_FUNCTION("Idffwf3yh8s", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitDefaultHardwareState);
    LIB_FUNCTION("QhnyReteJ1M", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitDefaultHardwareState175);
    LIB_FUNCTION("0H2vBYbTLHI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitDefaultHardwareState200);
    LIB_FUNCTION("yb2cRhagD1I", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitDefaultHardwareState350);
    LIB_FUNCTION("8lH54sfjfmU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitToDefaultContextState);
    LIB_FUNCTION("im2ZuItabu4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDrawInitToDefaultContextState400);
    LIB_FUNCTION("stDSYW2SBVM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmDrawOpaqueAuto);
    LIB_FUNCTION("TLV4mswiZ4A", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverCaptureInProgress);
    LIB_FUNCTION("ODEeJ1GfDtE", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterface);
    LIB_FUNCTION("4LSXsEKPTsE", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForGpuDebugger);
    LIB_FUNCTION("MpncRjHNYRE", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForGpuException);
    LIB_FUNCTION("EwjWGcIOgeM", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForHDRScopes);
    LIB_FUNCTION("3EXdrVC7WFk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForReplay);
    LIB_FUNCTION("P9iKqxAGeck", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForResourceRegistration);
    LIB_FUNCTION("t-vIc5cTEzg", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalRetrieveGnmInterfaceForValidation);
    LIB_FUNCTION("BvvO8Up88Zc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverInternalVirtualQuery);
    LIB_FUNCTION("R6z1xM3pW-w", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverTraceInProgress);
    LIB_FUNCTION("d88anrgNoKY", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmDriverTriggerCapture);
    LIB_FUNCTION("Fa3x75OOLRA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmEndWorkload);
    LIB_FUNCTION("4Mv9OXypBG8", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmFindResourcesPublic);
    LIB_FUNCTION("iBt3Oe00Kvc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmFlushGarlic);
    LIB_FUNCTION("GviyYfFQIkc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetCoredumpAddress);
    LIB_FUNCTION("meiO-5ZCVIE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetCoredumpMode);
    LIB_FUNCTION("O-7nHKgcNSQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetCoredumpProtectionFaultTimestamp);
    LIB_FUNCTION("bSJFzejYrJI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetDbgGcHandle);
    LIB_FUNCTION("pd4C7da6sEg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetDebugTimestamp);
    LIB_FUNCTION("UoYY0DWMC0U", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetEqEventType);
    LIB_FUNCTION("H7-fgvEutM0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetEqTimeStamp);
    LIB_FUNCTION("oL4hGI1PMpw", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetGpuBlockStatus);
    LIB_FUNCTION("Fwvh++m9IQI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetGpuCoreClockFrequency);
    LIB_FUNCTION("tZCSL5ulnB4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetGpuInfoStatus);
    LIB_FUNCTION("iFirFzgYsvw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetLastWaitedAddress);
    LIB_FUNCTION("KnldROUkWJY", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetNumTcaUnits);
    LIB_FUNCTION("FFVZcCu3zWU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetOffChipTessellationBufferSize);
    LIB_FUNCTION("QJjPjlmPAL0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetOwnerName);
    LIB_FUNCTION("dewXw5roLs0", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetPhysicalCounterFromVirtualized);
    LIB_FUNCTION("fzJdEihTFV4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetProtectionFaultTimeStamp);
    LIB_FUNCTION("4PKnYXOhcx4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetResourceBaseAddressAndSizeInBytes);
    LIB_FUNCTION("O0S96YnD04U", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetResourceName);
    LIB_FUNCTION("UBv7FkVfzcQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetResourceShaderGuid);
    LIB_FUNCTION("bdqdvIkLPIU", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetResourceType);
    LIB_FUNCTION("UoBuWAhKk7U", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetResourceUserData);
    LIB_FUNCTION("nEyFbYUloIM", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetShaderProgramBaseAddress);
    LIB_FUNCTION("k7iGTvDQPLQ", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGetShaderStatus);
    LIB_FUNCTION("ln33zjBrfjk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmGetTheTessellationFactorRingBufferBaseAddress);
    LIB_FUNCTION("QLdG7G-PBZo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGpuPaDebugEnter);
    LIB_FUNCTION("tVEdZe3wlbY", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmGpuPaDebugLeave);
    LIB_FUNCTION("NfvOrNzy6sk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmInsertDingDongMarker);
    LIB_FUNCTION("7qZVNgEu+SY", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmInsertPopMarker);
    LIB_FUNCTION("aPIZJTXC+cU", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmInsertPushColorMarker);
    LIB_FUNCTION("W1Etj-jlW7Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmInsertPushMarker);
    LIB_FUNCTION("aj3L-iaFmyk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmInsertSetColorMarker);
    LIB_FUNCTION("jiItzS6+22g", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmInsertSetMarker);
    LIB_FUNCTION("URDgJcXhQOs", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmInsertThreadTraceMarker);
    LIB_FUNCTION("1qXLHIpROPE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmInsertWaitFlipDone);
    LIB_FUNCTION("HRyNHoAjb6E", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmIsCoredumpValid);
    LIB_FUNCTION("jg33rEKLfVs", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmIsUserPaEnabled);
    LIB_FUNCTION("26PM5Mzl8zc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmLogicalCuIndexToPhysicalCuIndex);
    LIB_FUNCTION("RU74kek-N0c", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmLogicalCuMaskToPhysicalCuMask);
    LIB_FUNCTION("Kl0Z3LH07QI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmLogicalTcaUnitToPhysical);
    LIB_FUNCTION("29oKvKXzEZo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmMapComputeQueue);
    LIB_FUNCTION("A+uGq+3KFtQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmMapComputeQueueWithPriority);
    LIB_FUNCTION("+N+wrSYBLIw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmPaDisableFlipCallbacks);
    LIB_FUNCTION("8WDA9RiXLaw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmPaEnableFlipCallbacks);
    LIB_FUNCTION("tNuT48mApTc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmPaHeartbeat);
    LIB_FUNCTION("6IMbpR7nTzA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmQueryResourceRegistrationUserMemoryRequirements);
    LIB_FUNCTION("+rJnw2e9O+0", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRaiseUserExceptionEvent);
    LIB_FUNCTION("9Mv61HaMhfA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmRegisterGdsResource);
    LIB_FUNCTION("t7-VbMosbR4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRegisterGnmLiveCallbackConfig);
    LIB_FUNCTION("ZFqKFl23aMc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmRegisterOwner);
    LIB_FUNCTION("nvEwfYAImTs", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmRegisterResource);
    LIB_FUNCTION("gObODli-OH8", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRequestFlipAndSubmitDone);
    LIB_FUNCTION("6YRHhh5mHCs", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRequestFlipAndSubmitDoneForWorkload);
    LIB_FUNCTION("f85orjx7qts", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmRequestMipStatsReportAndReset);
    LIB_FUNCTION("MYRtYhojKdA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmResetVgtControl);
    LIB_FUNCTION("hS0MKPRdNr0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaClose);
    LIB_FUNCTION("31G6PB2oRYQ", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaConstFill);
    LIB_FUNCTION("Lg2isla2XeQ", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaCopyLinear);
    LIB_FUNCTION("-Se2FY+UTsI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaCopyTiled);
    LIB_FUNCTION("OlFgKnBsALE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaCopyWindow);
    LIB_FUNCTION("LQQN0SwQv8c", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaFlush);
    LIB_FUNCTION("suUlSjWr7CE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaGetMinCmdSize);
    LIB_FUNCTION("5AtqyMgO7fM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSdmaOpen);
    LIB_FUNCTION("KXltnCwEJHQ", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetCsShader);
    LIB_FUNCTION("Kx-h-nWQJ8A", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetCsShaderWithModifier);
    LIB_FUNCTION("X9Omw9dwv5M", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetEmbeddedPsShader);
    LIB_FUNCTION("+AFvOEXrKJk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetEmbeddedVsShader);
    LIB_FUNCTION("FUHG8sQ3R58", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetEsShader);
    LIB_FUNCTION("jtkqXpAOY6w", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetGsRingSizes);
    LIB_FUNCTION("UJwNuMBcUAk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetGsShader);
    LIB_FUNCTION("VJNjFtqiF5w", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetHsShader);
    LIB_FUNCTION("vckdzbQ46SI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetLsShader);
    LIB_FUNCTION("bQVd5YzCal0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetPsShader);
    LIB_FUNCTION("5uFKckiJYRM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetPsShader350);
    LIB_FUNCTION("q-qhDxP67Hg", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetResourceRegistrationUserMemory);
    LIB_FUNCTION("K3BKBBYKUSE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetResourceUserData);
    LIB_FUNCTION("0O3xxFaiObw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetSpiEnableSqCounters);
    LIB_FUNCTION("lN7Gk-p9u78", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetSpiEnableSqCountersForUnitInstance);
    LIB_FUNCTION("+xuDhxlWRPg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetupMipStatsReport);
    LIB_FUNCTION("cFCp0NX8wf0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetVgtControl);
    LIB_FUNCTION("gAhCn6UiU4Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSetVsShader);
    LIB_FUNCTION("y+iI2lkX+qI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetWaveLimitMultiplier);
    LIB_FUNCTION("XiyzNZ9J4nQ", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSetWaveLimitMultipliers);
    LIB_FUNCTION("kkn+iy-mhyg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmEndSpm);
    LIB_FUNCTION("aqhuK2Mj4X4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmInit);
    LIB_FUNCTION("KHpZ9hJo1c0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmInit2);
    LIB_FUNCTION("QEsMC+M3yjE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetDelay);
    LIB_FUNCTION("hljMAxTLNF0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetMuxRam);
    LIB_FUNCTION("bioGsp74SLM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetMuxRam2);
    LIB_FUNCTION("cMWWYeqQQlM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetSelectCounter);
    LIB_FUNCTION("-zJi8Vb4Du4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetSpmSelects);
    LIB_FUNCTION("xTsOqp-1bE4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmSetSpmSelects2);
    LIB_FUNCTION("AmmYLcJGTl0", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSpmStartSpm);
    LIB_FUNCTION("UHDiSFDxNao", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttFini);
    LIB_FUNCTION("a3tLC56vwug", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttFinishTrace);
    LIB_FUNCTION("L-owl1dSKKg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetBcInfo);
    LIB_FUNCTION("LQtzqghKQm4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetGpuClocks);
    LIB_FUNCTION("wYN5mmv6Ya8", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetHiWater);
    LIB_FUNCTION("9X4SkENMS0M", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetStatus);
    LIB_FUNCTION("lbMccQM2iqc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetTraceCounter);
    LIB_FUNCTION("DYAC6JUeZvM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetTraceWptr);
    LIB_FUNCTION("pS2tjBxzJr4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetWrapCounts);
    LIB_FUNCTION("rXV8az6X+fM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttGetWrapCounts2);
    LIB_FUNCTION("ARS+TNLopyk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttGetWritebackLabels);
    LIB_FUNCTION("X6yCBYPP7HA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttInit);
    LIB_FUNCTION("2IJhUyK8moE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSelectMode);
    LIB_FUNCTION("QA5h6Gh3r60", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSelectTarget);
    LIB_FUNCTION("F5XJY1XHa3Y", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSelectTokens);
    LIB_FUNCTION("wJtaTpNZfH4", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetCuPerfMask);
    LIB_FUNCTION("kY4dsQh+SH4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttSetDceEventWrite);
    LIB_FUNCTION("7XRH1CIfNpI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetHiWater);
    LIB_FUNCTION("05YzC2r3hHo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetTraceBuffer2);
    LIB_FUNCTION("ASUric-2EnI", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetTraceBuffers);
    LIB_FUNCTION("gPxYzPp2wlo", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttSetUserData);
    LIB_FUNCTION("d-YcZX7SIQA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttSetUserdataTimer);
    LIB_FUNCTION("ru8cb4he6O8", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttStartTrace);
    LIB_FUNCTION("gVuGo1nBnG8", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttStopTrace);
    LIB_FUNCTION("OpyolX6RwS0", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttSwitchTraceBuffer);
    LIB_FUNCTION("dl5u5eGBgNk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSqttSwitchTraceBuffer2);
    LIB_FUNCTION("QLzOwOF0t+A", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSqttWaitForEvent);
    LIB_FUNCTION("xbxNatawohc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSubmitAndFlipCommandBuffers);
    LIB_FUNCTION("Ga6r7H6Y0RI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSubmitAndFlipCommandBuffersForWorkload);
    LIB_FUNCTION("zwY0YV91TTI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSubmitCommandBuffers);
    LIB_FUNCTION("jRcI8VcgTz4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmSubmitCommandBuffersForWorkload);
    LIB_FUNCTION("yvZ73uQUqrk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmSubmitDone);
    LIB_FUNCTION("ArSg-TGinhk", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUnmapComputeQueue);
    LIB_FUNCTION("yhFCnaz5daw", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmUnregisterAllResourcesForOwner);
    LIB_FUNCTION("fhKwCVVj9nk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmUnregisterOwnerAndResources);
    LIB_FUNCTION("k8EXkhIP+lM", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUnregisterResource);
    LIB_FUNCTION("nLM2i2+65hA", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdateGsShader);
    LIB_FUNCTION("GNlx+y7xPdE", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdateHsShader);
    LIB_FUNCTION("4MgRw-bVNQU", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdatePsShader);
    LIB_FUNCTION("mLVL7N7BVBg", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdatePsShader350);
    LIB_FUNCTION("V31V01UiScY", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmUpdateVsShader);
    LIB_FUNCTION("iCO804ZgzdA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateCommandBuffers);
    LIB_FUNCTION("SXw4dZEkgpA", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateDisableDiagnostics);
    LIB_FUNCTION("BgM3t3LvcNk", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateDisableDiagnostics2);
    LIB_FUNCTION("qGP74T5OWJc", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateDispatchCommandBuffers);
    LIB_FUNCTION("hsZPf1lON7E", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateDrawCommandBuffers);
    LIB_FUNCTION("RX7XCNSaL6I", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateGetDiagnosticInfo);
    LIB_FUNCTION("5SHGNwLXBV4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateGetDiagnostics);
    LIB_FUNCTION("HzMN7ANqYEc", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmValidateGetVersion);
    LIB_FUNCTION("rTIV11nMQuM", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidateOnSubmitEnabled);
    LIB_FUNCTION("MBMa6EFu4Ko", "libSceGnmDriver", 1, "libSceGnmDriver", sceGnmValidateResetState);
    LIB_FUNCTION("Q7t4VEYLafI", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceGnmValidationRegisterMemoryCheckCallback);
    LIB_FUNCTION("xeTLfxVIQO4", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceRazorCaptureCommandBuffersOnlyImmediate);
    LIB_FUNCTION("9thMn+uB1is", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceRazorCaptureCommandBuffersOnlySinceLastFlip);
    LIB_FUNCTION("u9YKpRRHe-M", "libSceGnmDriver", 1, "libSceGnmDriver", sceRazorCaptureImmediate);
    LIB_FUNCTION("4UFagYlfuAM", "libSceGnmDriver", 1, "libSceGnmDriver",
                 sceRazorCaptureSinceLastFlip);
    LIB_FUNCTION("f33OrruQYbM", "libSceGnmDriver", 1, "libSceGnmDriver", sceRazorIsLoaded);
    LIB_FUNCTION("Bj0GWi1jWcM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_063D065A2D6359C3);
    LIB_FUNCTION("DKusr7JYQp0", "libSceGnmDriver", 1, "libSceGnmDriver", Func_0CABACAFB258429D);
    LIB_FUNCTION("FQzzNvwumaM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_150CF336FC2E99A3);
    LIB_FUNCTION("F8pof57lLUk", "libSceGnmDriver", 1, "libSceGnmDriver", Func_17CA687F9EE52D49);
    LIB_FUNCTION("GHC4n3Wca0U", "libSceGnmDriver", 1, "libSceGnmDriver", Func_1870B89F759C6B45);
    LIB_FUNCTION("JvkCnvaKlV4", "libSceGnmDriver", 1, "libSceGnmDriver", Func_26F9029EF68A955E);
    LIB_FUNCTION("MB49u6sJLbA", "libSceGnmDriver", 1, "libSceGnmDriver", Func_301E3DBBAB092DB0);
    LIB_FUNCTION("MLr+Fyrxf+8", "libSceGnmDriver", 1, "libSceGnmDriver", Func_30BAFE172AF17FEF);
    LIB_FUNCTION("Pmo+ggPZUxc", "libSceGnmDriver", 1, "libSceGnmDriver", Func_3E6A3E8203D95317);
    LIB_FUNCTION("QP7vDGU0xDQ", "libSceGnmDriver", 1, "libSceGnmDriver", Func_40FEEF0C6534C434);
    LIB_FUNCTION("QWuQed5Mus4", "libSceGnmDriver", 1, "libSceGnmDriver", Func_416B9079DE4CBACE);
    LIB_FUNCTION("R3TYO7Tdv5o", "libSceGnmDriver", 1, "libSceGnmDriver", Func_4774D83BB4DDBF9A);
    LIB_FUNCTION("UGePHM7rmgA", "libSceGnmDriver", 1, "libSceGnmDriver", Func_50678F1CCEEB9A00);
    LIB_FUNCTION("VKLsX6TGJBM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_54A2EC5FA4C62413);
    LIB_FUNCTION("WpxSyDE4rms", "libSceGnmDriver", 1, "libSceGnmDriver", Func_5A9C52C83138AE6B);
    LIB_FUNCTION("XSIZOjHqEUI", "libSceGnmDriver", 1, "libSceGnmDriver", Func_5D22193A31EA1142);
    LIB_FUNCTION("clo23rtglI0", "libSceGnmDriver", 1, "libSceGnmDriver", Func_725A36DEBB60948D);
    LIB_FUNCTION("gCGlAvphubs", "libSceGnmDriver", 1, "libSceGnmDriver", Func_8021A502FA61B9BB);
    LIB_FUNCTION("nQAv4PpA8OY", "libSceGnmDriver", 1, "libSceGnmDriver", Func_9D002FE0FA40F0E6);
    LIB_FUNCTION("nSl-NqcCi3E", "libSceGnmDriver", 1, "libSceGnmDriver", Func_9D297F36A7028B71);
    LIB_FUNCTION("otfsenvPebM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_A2D7EC7A7BCF79B3);
    LIB_FUNCTION("qhKjy4mQhUo", "libSceGnmDriver", 1, "libSceGnmDriver", Func_AA12A3CB8990854A);
    LIB_FUNCTION("rcjdwAUCC8Y", "libSceGnmDriver", 1, "libSceGnmDriver", Func_ADC8DDC005020BC6);
    LIB_FUNCTION("sKhoi2ectC0", "libSceGnmDriver", 1, "libSceGnmDriver", Func_B0A8688B679CB42D);
    LIB_FUNCTION("tIkCC1FXpf8", "libSceGnmDriver", 1, "libSceGnmDriver", Func_B489020B5157A5FF);
    LIB_FUNCTION("ut57TBmRQN0", "libSceGnmDriver", 1, "libSceGnmDriver", Func_BADE7B4C199140DD);
    LIB_FUNCTION("0VEbnc-7Pdk", "libSceGnmDriver", 1, "libSceGnmDriver", Func_D1511B9DCFFB3DD9);
    LIB_FUNCTION("1TRGZJsC5Y4", "libSceGnmDriver", 1, "libSceGnmDriver", Func_D53446649B02E58E);
    LIB_FUNCTION("2Lbo4o4e8KM", "libSceGnmDriver", 1, "libSceGnmDriver", Func_D8B6E8E28E1EF0A3);
    LIB_FUNCTION("2T1zOhnddFQ", "libSceGnmDriver", 1, "libSceGnmDriver", Func_D93D733A19DD7454);
    LIB_FUNCTION("3plUQ7wqgxc", "libSceGnmDriver", 1, "libSceGnmDriver", Func_DE995443BC2A8317);
    LIB_FUNCTION("326VKBUMI-8", "libSceGnmDriver", 1, "libSceGnmDriver", Func_DF6E9528150C23FF);
    LIB_FUNCTION("7LTGukH+M1A", "libSceGnmDriver", 1, "libSceGnmDriver", Func_ECB4C6BA41FE3350);
    LIB_FUNCTION("dqPBvjFVpTA", "libSceGnmDebugModuleReset", 1, "libSceGnmDriver",
                 sceGnmDebugModuleReset);
    LIB_FUNCTION("RNPAItiMLIg", "libSceGnmDebugReset", 1, "libSceGnmDriver", sceGnmDebugReset);
    LIB_FUNCTION("xMMot887QXE", "libSceGnmDebugReset", 1, "libSceGnmDriver", Func_C4C328B7CF3B4171);
    LIB_FUNCTION("pF1HQjbmQJ0", "libSceGnmDriverCompat", 1, "libSceGnmDriver",
                 sceGnmDrawInitToDefaultContextStateInternalCommand);
    LIB_FUNCTION("jajhf-Gi3AI", "libSceGnmDriverCompat", 1, "libSceGnmDriver",
                 sceGnmDrawInitToDefaultContextStateInternalSize);
    LIB_FUNCTION("vbcR4Ken6AA", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 sceGnmFindResources);
    LIB_FUNCTION("eLQbNsKeTkU", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 sceGnmGetResourceRegistrationBuffers);
    LIB_FUNCTION("j6mSQs3UgaY", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 sceGnmRegisterOwnerForSystem);
    LIB_FUNCTION("HEOIaxbuVTA", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_1C43886B16EE5530);
    LIB_FUNCTION("gQNwGezNDgE", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_81037019ECCD0E01);
    LIB_FUNCTION("v7QcBXR48L8", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_BFB41C057478F0BF);
    LIB_FUNCTION("5R1E24FRI4w", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_E51D44DB8151238C);
    LIB_FUNCTION("+RaJBCVJZVM", "libSceGnmDriverResourceRegistration", 1, "libSceGnmDriver",
                 Func_F916890425496553);
    LIB_FUNCTION("Fwvh++m9IQI", "libSceGnmGetGpuCoreClockFrequency", 1, "libSceGnmDriver",
                 sceGnmGetGpuCoreClockFrequency);
    LIB_FUNCTION("R3TYO7Tdv5o", "libSceGnmWaitFreeSubmit", 1, "libSceGnmDriver",
                 Func_4774D83BB4DDBF9A);
    LIB_FUNCTION("ut57TBmRQN0", "libSceGnmWaitFreeSubmit", 1, "libSceGnmDriver",
                 Func_BADE7B4C199140DD);
};

} // namespace Libraries::GnmDriver
