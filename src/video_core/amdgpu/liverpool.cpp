// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/preprocessor/stringize.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#ifdef __ANDROID__
#include <cstdlib>
#include <vector>
#include <android/log.h>
#endif

#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "executor/gnmcap_format.h"
#include "common/polyfill_thread.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/libraries/gnmdriver/gnmdriver.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/amdgpu/render_wave_trace.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#ifdef __ANDROID__
#include "video_core/renderer_vulkan/vk_scheduler.h"
#endif

namespace AmdGpu {

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

namespace {
// Bounds-checked SET_*_REG write. reg_addr = base + a fully guest-controlled 16-bit reg_offset, and
// `words` = a guest count; an unchecked memcpy into the fixed 0xD000-word reg_array overruns it into
// adjacent .so BSS (Liverpool's Vulkan scheduler std::mutex etc.), giving Sonic JIT's shape-
// shifting "destroyed mutex" FORTIFY/SIGSEGV corruption. Clamp/drop out-of-range writes.
bool ExecutorSafeRegArrayWrite(u32* reg_array, u32 capacity, u32 reg_addr, const void* payload,
                               u32 words, const char* which) {
    if (reg_addr >= capacity) {
#ifdef __ANDROID__
        static std::atomic<u32> b{0};
        if (b.fetch_add(1, std::memory_order_relaxed) < 16)
            __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                "[EXECUTOR_SETREG_DROP] which=%s reg_addr=0x%x cap=0x%x words=%u",
                                which, reg_addr, capacity, words);
#else
        (void)which;
#endif
        return false;
    }
    const u32 clamped = std::min(words, capacity - reg_addr);
    const auto bytes = static_cast<std::size_t>(clamped) * sizeof(u32);
    const bool changed = std::memcmp(reg_array + reg_addr, payload, bytes) != 0;
    if (changed) {
        std::memcpy(reg_array + reg_addr, payload, bytes);
    }
#ifdef __ANDROID__
    if (clamped != words) {
        static std::atomic<u32> b{0};
        if (b.fetch_add(1, std::memory_order_relaxed) < 16)
            __android_log_print(
                ANDROID_LOG_WARN, "LSX4Native",
                "[EXECUTOR_SETREG_CLAMP] which=%s reg_addr=0x%x cap=0x%x words=%u clamped=%u", which,
                reg_addr, capacity, words, clamped);
    }
#else
    (void)which;
#endif
    return changed;
}
} // namespace

#ifdef __ANDROID__
namespace {
bool ExecutorEnvFlag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool ExecutorLiveGnmChecks() {
    // Clean gameplay must not synchronously write one or more logcat records per PM4 packet.
    // Preserve the full forensic stream only behind the existing explicit trace markers.
    // FPS: the broad EXECUTOR_TRACE_LIVE_WIDE used by widecheck runs turned this on and produced
    // per-PM4-packet logcat on the render path (~23k lines/run). Require a dedicated opt-in instead so
    // ordinary widecheck runs stay fast; EXECUTOR_TRACE_LIVE_WIDE_HOT / _GNM_CHECKS restore it.
    static const bool enabled =
        !ExecutorEnvFlag("EXECUTOR_DISABLE_LIVE_GNM_CHECKS") &&
        (ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_WIDE_HOT") ||
         ExecutorEnvFlag("EXECUTOR_TRACE_LIVE_GNM_CHECKS"));
    return enabled;
}

bool ExecutorTracePm4() {
    // ProcessGraphics queries this at every coroutine resume and at many PM4 branches. Runtime
    // flags are established before Liverpool starts, so paying getenv() in the GPU wait loop is
    // pure overhead (65% of the sampled GpuCommandProcessor CPU on the first Fluster frame).
    static const bool enabled = ExecutorEnvFlag("EXECUTOR_TRACE_PM4");
    return enabled;
}

bool ExecutorSkipRasterizerFinish() {
    // The value is fixed before Liverpool starts.  This marker is an A/B escape hatch for restoring
    // the normal asynchronous PS4 submission model after exact VideoOut timeline retirement has
    // replaced the old queue-wide Android correctness wait.
    static const bool enabled = ExecutorEnvFlag("EXECUTOR_SKIP_RASTERIZER_FINISH");
    return enabled;
}

bool ExecutorBoundedRasterizerPipeline() {
    // This is the production Android path now. The synchronous per-frame Finish path serialized
    // roughly 180 ms of PM4 parsing with roughly 220 ms of GPU work and made the title run in slow
    // motion. Keep an explicit opt-out for compatibility bisection, rather than requiring every
    // launcher to remember an opt-in sentinel.
    static const bool enabled =
        !ExecutorEnvFlag("EXECUTOR_DISABLE_BOUNDED_RASTERIZER_PIPELINE");
    return enabled;
}

bool ExecutorEopWaitBatchingV2() {
    static const bool enabled = ExecutorEnvFlag("EXECUTOR_EOP_WAIT_BATCHING_V2");
    return enabled;
}

bool ExecutorShouldLogWaitYield(u32 yields) {
    return yields == 1 || yields == 64 || yields == 4096 || yields == 65536;
}

enum class ExecutorEopTraceStage : u32 {
    HleSubmit,
    HleEop,
    LiverpoolReceive,
    LiverpoolTerminalDrop,
    LiverpoolEnqueue,
    ParserBegin,
    ParserEop,
    IrqSignal,
    EqTrigger,
    EqWaitEnter,
    EqWaitExit,
    SemWaitEnter,
    SemWaitExit,
    SemPost,
    ParserEnd,
};

struct ExecutorEopTraceEvent {
    u64 seq{};
    u64 elapsed_us{};
    ExecutorEopTraceStage stage{};
    uintptr_t dcb{};
    u32 dcb_dw{};
    u32 offset_dw{};
    u32 aux0{};
    u32 aux1{};
    u64 aux2{};
    const char* label{};
};

struct ExecutorEopTraceState {
    static constexpr std::size_t RingSize = 96;

    std::mutex mutex;
    std::array<ExecutorEopTraceEvent, RingSize> ring{};
    u64 cursor{};
    const u64 start_us{
        static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count())};
    std::atomic<u64> hle_submits{};
    std::atomic<u64> hle_eops{};
    std::atomic<u64> liverpool_receives{};
    std::atomic<u64> terminal_drops{};
    std::atomic<u64> enqueued{};
    std::atomic<u64> parser_begin{};
    std::atomic<u64> parser_eop{};
    std::atomic<u64> irq_signal{};
    std::atomic<u64> eq_trigger{};
    std::atomic<u64> parser_end{};
    std::atomic<u64> last_progress_us{};
    std::atomic<u64> pulse_last_hle{};
    std::atomic<u32> stagnant_pulses{};
    std::atomic<u64> auto_dump_hle_epoch{~u64{0}};
    std::atomic<bool> explicit_dumped{};
};

bool ExecutorEopTraceEnabled() {
    static const bool enabled = ExecutorEnvFlag("EXECUTOR_TRACE_EOP_CHAIN");
    return enabled;
}

bool ExecutorEopTraceExplicitDumpEnabled() {
    static const bool enabled = ExecutorEnvFlag("EXECUTOR_TRACE_EOP_CHAIN_DUMP");
    return enabled;
}

ExecutorEopTraceState& ExecutorGetEopTraceState() {
    static ExecutorEopTraceState state;
    return state;
}

u64 ExecutorEopTraceNowUs(const ExecutorEopTraceState& state) {
    const auto now_us = static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now().time_since_epoch())
                                             .count());
    return now_us - state.start_us;
}

const char* ExecutorEopTraceStageName(ExecutorEopTraceStage stage) {
    switch (stage) {
    case ExecutorEopTraceStage::HleSubmit:
        return "HLE_SUBMIT";
    case ExecutorEopTraceStage::HleEop:
        return "HLE_EOP";
    case ExecutorEopTraceStage::LiverpoolReceive:
        return "LIVERPOOL_RECEIVE";
    case ExecutorEopTraceStage::LiverpoolTerminalDrop:
        return "LIVERPOOL_TERMINAL_DROP";
    case ExecutorEopTraceStage::LiverpoolEnqueue:
        return "LIVERPOOL_ENQUEUE";
    case ExecutorEopTraceStage::ParserBegin:
        return "PARSER_BEGIN";
    case ExecutorEopTraceStage::ParserEop:
        return "PARSER_EOP";
    case ExecutorEopTraceStage::IrqSignal:
        return "IRQ_SIGNAL";
    case ExecutorEopTraceStage::EqTrigger:
        return "EQ_TRIGGER";
    case ExecutorEopTraceStage::EqWaitEnter:
        return "EQ_WAIT_ENTER";
    case ExecutorEopTraceStage::EqWaitExit:
        return "EQ_WAIT_EXIT";
    case ExecutorEopTraceStage::SemWaitEnter:
        return "SEM_WAIT_ENTER";
    case ExecutorEopTraceStage::SemWaitExit:
        return "SEM_WAIT_EXIT";
    case ExecutorEopTraceStage::SemPost:
        return "SEM_POST";
    case ExecutorEopTraceStage::ParserEnd:
        return "PARSER_END";
    }
    return "UNKNOWN";
}

void ExecutorEopTraceRecord(ExecutorEopTraceStage stage, uintptr_t dcb, u32 dcb_dw,
                            u32 offset_dw = 0, u32 aux0 = 0, u32 aux1 = 0, u64 aux2 = 0,
                            const char* label = nullptr) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    auto& state = ExecutorGetEopTraceState();
    const u64 elapsed_us = ExecutorEopTraceNowUs(state);
    state.last_progress_us.store(elapsed_us, std::memory_order_relaxed);
    std::scoped_lock lock{state.mutex};
    const u64 seq = ++state.cursor;
    state.ring[(seq - 1) % state.ring.size()] = {
        .seq = seq,
        .elapsed_us = elapsed_us,
        .stage = stage,
        .dcb = dcb,
        .dcb_dw = dcb_dw,
        .offset_dw = offset_dw,
        .aux0 = aux0,
        .aux1 = aux1,
        .aux2 = aux2,
        .label = label,
    };
}

u64 ExecutorEopTraceGap(u64 upstream, u64 downstream) {
    return upstream > downstream ? upstream - downstream : 0;
}

void ExecutorEopTraceDump(const char* reason, u64 pulse) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    auto& state = ExecutorGetEopTraceState();
    std::array<ExecutorEopTraceEvent, ExecutorEopTraceState::RingSize> snapshot{};
    std::size_t count{};
    u64 total_events{};
    {
        std::scoped_lock lock{state.mutex};
        total_events = state.cursor;
        count = static_cast<std::size_t>(
            std::min<u64>(total_events, ExecutorEopTraceState::RingSize));
        const u64 first = total_events - count;
        for (std::size_t i = 0; i < count; ++i) {
            snapshot[i] = state.ring[(first + i) % state.ring.size()];
        }
    }

    const u64 hle_submits = state.hle_submits.load(std::memory_order_relaxed);
    const u64 hle_eops = state.hle_eops.load(std::memory_order_relaxed);
    const u64 receives = state.liverpool_receives.load(std::memory_order_relaxed);
    const u64 drops = state.terminal_drops.load(std::memory_order_relaxed);
    const u64 enqueued = state.enqueued.load(std::memory_order_relaxed);
    const u64 parser_begin = state.parser_begin.load(std::memory_order_relaxed);
    const u64 parser_eop = state.parser_eop.load(std::memory_order_relaxed);
    const u64 irq_signal = state.irq_signal.load(std::memory_order_relaxed);
    const u64 eq_trigger = state.eq_trigger.load(std::memory_order_relaxed);
    const u64 parser_end = state.parser_end.load(std::memory_order_relaxed);
    const u64 now_us = ExecutorEopTraceNowUs(state);
    const u64 last_progress_us = state.last_progress_us.load(std::memory_order_relaxed);
    const u64 age_ms = ExecutorEopTraceGap(now_us, last_progress_us) / 1000;

    __android_log_print(
        ANDROID_LOG_WARN, "LSX4Native",
        "[EXECUTOR_EOP_CHAIN_SUMMARY] reason=%s pulse=%llu ageMs=%llu events=%llu "
        "hleSubmit=%llu hleEop=%llu receive=%llu terminalDrop=%llu enqueue=%llu "
        "parserBegin=%llu parserEnd=%llu parserEop=%llu irqSignal=%llu eqTrigger=%llu "
        "gapHleReceive=%llu gapReceiveEnqueue=%llu gapBeginEnd=%llu gapHleEopParser=%llu "
        "gapParserEopIrq=%llu gapIrqEq=%llu",
        reason ? reason : "unknown", static_cast<unsigned long long>(pulse),
        static_cast<unsigned long long>(age_ms), static_cast<unsigned long long>(total_events),
        static_cast<unsigned long long>(hle_submits), static_cast<unsigned long long>(hle_eops),
        static_cast<unsigned long long>(receives), static_cast<unsigned long long>(drops),
        static_cast<unsigned long long>(enqueued), static_cast<unsigned long long>(parser_begin),
        static_cast<unsigned long long>(parser_end), static_cast<unsigned long long>(parser_eop),
        static_cast<unsigned long long>(irq_signal), static_cast<unsigned long long>(eq_trigger),
        static_cast<unsigned long long>(ExecutorEopTraceGap(hle_submits, receives)),
        static_cast<unsigned long long>(ExecutorEopTraceGap(receives, enqueued + drops)),
        static_cast<unsigned long long>(ExecutorEopTraceGap(parser_begin, parser_end)),
        static_cast<unsigned long long>(ExecutorEopTraceGap(hle_eops, parser_eop)),
        static_cast<unsigned long long>(ExecutorEopTraceGap(parser_eop, irq_signal)),
        static_cast<unsigned long long>(ExecutorEopTraceGap(irq_signal, eq_trigger)));
    for (std::size_t i = 0; i < count; ++i) {
        const auto& event = snapshot[i];
        __android_log_print(
            ANDROID_LOG_WARN, "LSX4Native",
            "[EXECUTOR_EOP_CHAIN_EVENT] seq=%llu us=%llu stage=%s dcb=0x%llx dcbDw=%u "
            "offDw=%u aux0=0x%x aux1=0x%x aux2=0x%llx label=%s",
            static_cast<unsigned long long>(event.seq),
            static_cast<unsigned long long>(event.elapsed_us),
            ExecutorEopTraceStageName(event.stage), static_cast<unsigned long long>(event.dcb),
            event.dcb_dw, event.offset_dw, event.aux0, event.aux1,
            static_cast<unsigned long long>(event.aux2), event.label ? event.label : "-");
    }
}

void ExecutorEopTraceLiverpoolReceive(std::span<const u32> dcb, std::span<const u32> ccb,
                                      bool terminal_drop) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    auto& state = ExecutorGetEopTraceState();
    state.liverpool_receives.fetch_add(1, std::memory_order_relaxed);
    ExecutorEopTraceRecord(ExecutorEopTraceStage::LiverpoolReceive,
                           reinterpret_cast<uintptr_t>(dcb.data()), static_cast<u32>(dcb.size()),
                           0, static_cast<u32>(ccb.size()));
    if (terminal_drop) {
        state.terminal_drops.fetch_add(1, std::memory_order_relaxed);
        ExecutorEopTraceRecord(ExecutorEopTraceStage::LiverpoolTerminalDrop,
                               reinterpret_cast<uintptr_t>(dcb.data()),
                               static_cast<u32>(dcb.size()));
    }
}

void ExecutorEopTraceLiverpoolEnqueue(std::span<const u32> dcb, std::span<const u32> ccb,
                                      u32 pending) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    auto& state = ExecutorGetEopTraceState();
    state.enqueued.fetch_add(1, std::memory_order_relaxed);
    ExecutorEopTraceRecord(ExecutorEopTraceStage::LiverpoolEnqueue,
                           reinterpret_cast<uintptr_t>(dcb.data()), static_cast<u32>(dcb.size()),
                           0, pending, static_cast<u32>(ccb.size()));
}

void ExecutorEopTraceParserBegin(uintptr_t dcb, u32 dcb_dw) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    ExecutorGetEopTraceState().parser_begin.fetch_add(1, std::memory_order_relaxed);
    ExecutorEopTraceRecord(ExecutorEopTraceStage::ParserBegin, dcb, dcb_dw);
}

void ExecutorEopTraceParserEop(uintptr_t dcb, u32 dcb_dw, u32 offset_dw, u32 int_sel,
                               u32 data_sel, u64 address) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    ExecutorGetEopTraceState().parser_eop.fetch_add(1, std::memory_order_relaxed);
    ExecutorEopTraceRecord(ExecutorEopTraceStage::ParserEop, dcb, dcb_dw, offset_dw, int_sel,
                           data_sel, address);
}

void ExecutorEopTraceIrqSignal(uintptr_t dcb, u32 dcb_dw, u32 offset_dw, u32 int_sel,
                               u32 data_sel, u64 address) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    ExecutorGetEopTraceState().irq_signal.fetch_add(1, std::memory_order_relaxed);
    ExecutorEopTraceRecord(ExecutorEopTraceStage::IrqSignal, dcb, dcb_dw, offset_dw, int_sel,
                           data_sel, address);
}

void ExecutorEopTraceParserEnd(uintptr_t dcb, u32 dcb_dw) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    ExecutorGetEopTraceState().parser_end.fetch_add(1, std::memory_order_relaxed);
    ExecutorEopTraceRecord(ExecutorEopTraceStage::ParserEnd, dcb, dcb_dw);
}

bool ExecutorWriteFenceToCpuAddress(const char* op, void* address, u64 data, u32 num_bytes) {
    if (Libraries::VideoOut::ExecutorTryWriteVoLabelAlias(address, &data, num_bytes)) {
        if (ExecutorLiveGnmChecks()) {
            static u32 logged_alias_writes = 0;
            if (logged_alias_writes++ < 32) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_PM4_FENCE_WRITE] op=%s kind=vo_label_alias raw=0x%llx "
                    "bytes=%u data=0x%llx",
                    op,
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address)),
                    num_bytes, static_cast<unsigned long long>(data));
            }
        }
        return true;
    }

    auto* memory = Core::Memory::Instance();
    if (memory && memory->TryWriteBacking(address, &data, num_bytes)) {
        if (ExecutorLiveGnmChecks()) {
            static u32 logged_backing_writes = 0;
            if (logged_backing_writes++ < 64) {
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_LIVE_PM4_FENCE_WRITE] op=%s kind=memory_backing raw=0x%llx "
                    "bytes=%u data=0x%llx",
                    op,
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address)),
                    num_bytes, static_cast<unsigned long long>(data));
            }
        }
        return true;
    }

    // Non-Android upstream path writes directly. On Android, only do that for addresses that do not
    // look like stale/tagged host pointers; guest VA writes should already have gone through
    // TryWriteBacking, and VO labels should already have gone through the alias table above.
    const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(address);
    if ((raw_addr >> 56) != 0 || raw_addr >= (1ull << 48)) {
        if (ExecutorLiveGnmChecks()) {
            static u32 logged_skips = 0;
            if (logged_skips++ < 32) {
                __android_log_print(
                    ANDROID_LOG_WARN, "LSX4Native",
                    "[EXECUTOR_LIVE_PM4_FENCE_WRITE_SKIP] op=%s raw=0x%llx bytes=%u data=0x%llx",
                    op,
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address)),
                    num_bytes, static_cast<unsigned long long>(data));
            }
        }
        return false;
    }

    std::memcpy(address, &data, num_bytes);
    if (ExecutorLiveGnmChecks()) {
        static u32 logged_direct_writes = 0;
        if (logged_direct_writes++ < 32) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_LIVE_PM4_FENCE_WRITE] op=%s kind=direct raw=0x%llx bytes=%u "
                "data=0x%llx",
                op, static_cast<unsigned long long>(raw_addr), num_bytes,
                static_cast<unsigned long long>(data));
        }
    }
    return true;
}

bool ExecutorDmaSrcIsMemory(DmaDataSrc src) {
    return src == DmaDataSrc::Memory || src == DmaDataSrc::MemoryUsingL2;
}

bool ExecutorDmaDstIsMemory(DmaDataDst dst) {
    return dst == DmaDataDst::Memory || dst == DmaDataDst::MemoryUsingL2;
}

bool ExecutorMirrorDmaDataToCpuBacking(const PM4DmaData& dma_data, const char* phase) {
    // Live DMA_DATA is an ordered GPU operation.  Mirroring it eagerly from the CPU mapping is
    // not equivalent: a Memory source may already contain newer GPU-produced bytes, in which case
    // CopySparseMemory observes stale backing and the mirror corrupts the destination immediately
    // before the real Vulkan copy.  Desktop shadPS4 therefore executes only FillBuffer/CopyBuffer.
    // CPU emulation is useful for isolated capture replay (whose resources live in a CPU overlay),
    // but must remain an explicit diagnostic outside that replay path.
    static const bool live_dma_cpu_mirror_enabled =
        ExecutorEnvFlag("EXECUTOR_ENABLE_LIVE_DMA_CPU_MIRROR");
    if (!live_dma_cpu_mirror_enabled) {
        return false;
    }

    if (!ExecutorDmaDstIsMemory(dma_data.dst_sel)) {
        // FPS: per-DMA skip-dst trace fires on the render path; opt-in via dedicated env so it is
        // not paid for under the broad live-gnm-checks diagnostics.
        static const bool trace_dma_mirror = std::getenv("EXECUTOR_TRACE_DMA_MIRROR") != nullptr;
        if (trace_dma_mirror && ExecutorLiveGnmChecks()) {
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_LIVE_DMA_MIRROR] phase=%s op=skip-dst srcSel=%u "
                                "dstSel=%u src=0x%llx dstLo=0x%x bytes=%u data=%08x",
                                phase, u32(dma_data.src_sel.Value()),
                                u32(dma_data.dst_sel.Value()),
                                static_cast<unsigned long long>(
                                    ExecutorDmaSrcIsMemory(dma_data.src_sel)
                                        ? dma_data.SrcAddress<VAddr>()
                                        : static_cast<VAddr>(0)),
                                dma_data.dst_addr_lo, dma_data.NumBytes(), dma_data.data);
        }
        return false;
    }

    const u32 num_bytes = dma_data.NumBytes();
    if (num_bytes == 0) {
        return false;
    }

    auto* memory = Core::Memory::Instance();
    const VAddr dst_addr = dma_data.DstAddress<VAddr>();
    if (!memory || !memory->IsValidMapping(dst_addr, num_bytes)) {
        if (ExecutorLiveGnmChecks()) {
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_LIVE_DMA_MIRROR] phase=%s op=miss-dst srcSel=%u dstSel=%u "
                                "dst=0x%llx bytes=%u",
                                phase, u32(dma_data.src_sel.Value()),
                                u32(dma_data.dst_sel.Value()),
                                static_cast<unsigned long long>(dst_addr), num_bytes);
        }
        return false;
    }

    std::vector<u8> data(num_bytes);
    const char* op = "unsupported";
    if (dma_data.src_sel == DmaDataSrc::Data) {
        op = "fill";
        for (u32 off = 0; off < num_bytes; off += sizeof(u32)) {
            const u32 chunk = std::min<u32>(sizeof(u32), num_bytes - off);
            std::memcpy(data.data() + off, &dma_data.data, chunk);
        }
    } else if (ExecutorDmaSrcIsMemory(dma_data.src_sel)) {
        const VAddr src_addr = dma_data.SrcAddress<VAddr>();
        op = "copy";
        if (!memory->IsValidMapping(src_addr, num_bytes)) {
            if (ExecutorLiveGnmChecks()) {
                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                    "[EXECUTOR_LIVE_DMA_MIRROR] phase=%s op=miss-src src=0x%llx "
                                    "dst=0x%llx bytes=%u",
                                    phase, static_cast<unsigned long long>(src_addr),
                                    static_cast<unsigned long long>(dst_addr), num_bytes);
            }
            return false;
        }
        memory->CopySparseMemory(src_addr, data.data(), num_bytes);
    } else {
        if (ExecutorLiveGnmChecks()) {
            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                "[EXECUTOR_LIVE_DMA_MIRROR] phase=%s op=skip-src srcSel=%u dst=0x%llx "
                                "bytes=%u",
                                phase, u32(dma_data.src_sel.Value()),
                                static_cast<unsigned long long>(dst_addr), num_bytes);
        }
        return false;
    }

    const bool wrote = memory->TryWriteBacking(std::bit_cast<void*>(dst_addr), data.data(), num_bytes);
    if (ExecutorLiveGnmChecks()) {
        u32 first = 0;
        if (!data.empty()) {
            std::memcpy(&first, data.data(), std::min<size_t>(sizeof(first), data.size()));
        }
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_LIVE_DMA_MIRROR] phase=%s op=%s wrote=%u srcSel=%u dstSel=%u "
                            "src=0x%llx dst=0x%llx bytes=%u first=%08x",
                            phase, op, wrote ? 1u : 0u, u32(dma_data.src_sel.Value()),
                            u32(dma_data.dst_sel.Value()),
                            static_cast<unsigned long long>(
                                ExecutorDmaSrcIsMemory(dma_data.src_sel)
                                    ? dma_data.SrcAddress<VAddr>()
                                    : static_cast<VAddr>(0)),
                            static_cast<unsigned long long>(dst_addr), num_bytes, first);
    }
    return wrote;
}

void ExecutorLogLiveDmaData(const PM4DmaData& dma_data, const char* phase, size_t off_dw,
                            bool mirrored, const char* decision) {
    if (!ExecutorLiveGnmChecks()) {
        return;
    }
    static u32 logged_dma = 0;
    if (logged_dma++ >= 96) {
        return;
    }
    const VAddr src_addr = ExecutorDmaSrcIsMemory(dma_data.src_sel) ? dma_data.SrcAddress<VAddr>() : 0;
    const VAddr dst_addr = ExecutorDmaDstIsMemory(dma_data.dst_sel) ? dma_data.DstAddress<VAddr>() : 0;
    __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                        "[EXECUTOR_LIVE_DMA_PACKET] phase=%s off=%zu decision=%s mirror=%u "
                        "srcSel=%u dstSel=%u src=0x%llx dst=0x%llx dstLo=0x%x bytes=%u data=%08x",
                        phase, off_dw, decision ? decision : "exec", mirrored ? 1u : 0u,
                        u32(dma_data.src_sel.Value()), u32(dma_data.dst_sel.Value()),
                        static_cast<unsigned long long>(src_addr),
                        static_cast<unsigned long long>(dst_addr), dma_data.dst_addr_lo,
                        dma_data.NumBytes(), dma_data.data);
}

void ExecutorLogMagicDmaPayload(const PM4DmaData& dma_data, const char* phase, size_t off_dw) {
    if (!ExecutorLiveGnmChecks() || dma_data.dst_addr_lo != 0x3022C ||
        !ExecutorDmaSrcIsMemory(dma_data.src_sel)) {
        return;
    }

    static u32 logged_magic = 0;
    if (logged_magic++ >= 16) {
        return;
    }

    auto* memory = Core::Memory::Instance();
    const VAddr src_addr = dma_data.SrcAddress<VAddr>();
    const u32 num_bytes = dma_data.NumBytes();
    const bool mapped = memory && num_bytes != 0 && memory->IsValidMapping(src_addr, num_bytes);
    std::array<u32, 8> words{};
    u32 nonzero_words = 0;
    if (mapped) {
        std::vector<u8> bytes(std::min<u32>(num_bytes, static_cast<u32>(words.size() * 4)));
        memory->CopySparseMemory(src_addr, bytes.data(), bytes.size());
        for (size_t i = 0; i < words.size() && (i * 4 + 4) <= bytes.size(); ++i) {
            std::memcpy(&words[i], bytes.data() + i * 4, sizeof(u32));
            if (words[i] != 0) {
                ++nonzero_words;
            }
        }
    }

    __android_log_print(
        ANDROID_LOG_INFO, "EXECUTOR",
        "[EXECUTOR_LIVE_DMA_MAGIC_PAYLOAD] phase=%s off=%zu src=0x%llx bytes=%u mapped=%u "
        "nonzeroWords=%u first=%08x %08x %08x %08x %08x %08x %08x %08x",
        phase, off_dw, static_cast<unsigned long long>(src_addr), num_bytes, mapped ? 1u : 0u,
        nonzero_words, words[0], words[1], words[2], words[3], words[4], words[5], words[6],
        words[7]);
}

struct ExecutorPm4LiveSummary {
    u32 packets{};
    u32 draws{};
    u32 dispatches{};
    u32 waits{};
    u32 dma{};
    u32 writes{};
    u32 set_context{};
    u32 set_sh{};
    u32 set_uconfig{};
    u32 indirect{};
    u32 events{};
    u32 wait_yields{};
    u32 last_opcode{};
    size_t last_off_dw{};
};
} // namespace

void ExecutorEopTraceHleSubmit(std::span<const u32> dcb, std::span<const u32> ccb,
                               const char* label, u32 workload, u32 cbpair) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }

    auto& state = ExecutorGetEopTraceState();
    state.hle_submits.fetch_add(1, std::memory_order_relaxed);
    const uintptr_t base = reinterpret_cast<uintptr_t>(dcb.data());
    const u32 dcb_dw = static_cast<u32>(dcb.size());
    ExecutorEopTraceRecord(ExecutorEopTraceStage::HleSubmit, base, dcb_dw, 0, workload, cbpair,
                           ccb.size(), label);

    // Scan only the bounded top-level DCB supplied by HLE. Indirect-buffer EOPs are still captured
    // by PARSER_EOP below; keeping this walk top-level makes malformed guest packet lengths harmless.
    std::size_t offset_dw = 0;
    bool saw_structured_irq_eop = false;
    while (offset_dw < dcb.size()) {
        const auto remaining = dcb.subspan(offset_dw);
        const auto* header = reinterpret_cast<const PM4Header*>(remaining.data());
        const u32 type = header->type.Value();
        u32 advance = 1;
        if (type == 3) {
            advance = header->type3.NumWords() + 1;
            if (advance == 0 || advance > remaining.size()) {
                break;
            }
            if (header->type3.opcode.Value() == PM4ItOpcode::EventWriteEop &&
                advance >= sizeof(PM4CmdEventWriteEop) / sizeof(u32)) {
                const auto* eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                state.hle_eops.fetch_add(1, std::memory_order_relaxed);
                saw_structured_irq_eop |= static_cast<u32>(eop->int_sel.Value()) != 0;
                ExecutorEopTraceRecord(
                    ExecutorEopTraceStage::HleEop, base, dcb_dw, static_cast<u32>(offset_dw),
                    static_cast<u32>(eop->int_sel.Value()),
                    static_cast<u32>(eop->data_sel.Value()),
                    reinterpret_cast<uintptr_t>(eop->Address<u32>()), label);
            }
        } else if (type != 2) {
            break;
        }
        offset_dw += advance;
    }

    // If an earlier malformed packet count makes the structured walk stop or jump over Unity's
    // six-DWORD IRQ-only EOP, retain that distinction in the bounded trace. Liverpool uses the
    // structured parser, so finding this raw signature proves "bytes arrived but packet framing
    // hid them" instead of incorrectly reporting that HLE never supplied the EOP at all.
    if (!saw_structured_irq_eop) {
        for (std::size_t i = 0; i + 6 <= dcb.size(); ++i) {
            if (dcb[i] != 0xc0044700u || dcb[i + 1] != 0x00000504u || dcb[i + 2] != 0u ||
                dcb[i + 3] != 0x01000000u || dcb[i + 4] != 0u || dcb[i + 5] != 0u) {
                continue;
            }
            state.hle_eops.fetch_add(1, std::memory_order_relaxed);
            ExecutorEopTraceRecord(ExecutorEopTraceStage::HleEop, base, dcb_dw,
                                   static_cast<u32>(i), 1, 0, 0, "raw-unparsed-unity-irq");
            break;
        }
    }
}

void ExecutorEopTraceEqTrigger(u64 id, s64 eq) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    auto& state = ExecutorGetEopTraceState();
    state.eq_trigger.fetch_add(1, std::memory_order_relaxed);
    ExecutorEopTraceRecord(ExecutorEopTraceStage::EqTrigger, 0, 0, 0, static_cast<u32>(id), 0,
                           static_cast<u64>(eq));
}

void ExecutorEopTraceEqWait(bool entering, s64 eq, s32 result, u64 id, s16 filter) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    ExecutorEopTraceRecord(
        entering ? ExecutorEopTraceStage::EqWaitEnter : ExecutorEopTraceStage::EqWaitExit,
        static_cast<uintptr_t>(eq), 0, 0, static_cast<u32>(id),
        static_cast<u32>(static_cast<s32>(filter)), static_cast<u64>(static_cast<u32>(result)));
}

void ExecutorEopTraceSemSync(u32 operation, uintptr_t slot, uintptr_t native, s32 before,
                             s32 after, s32 result, u32 thread_kind) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }
    const auto stage = operation == 0   ? ExecutorEopTraceStage::SemWaitEnter
                       : operation == 1 ? ExecutorEopTraceStage::SemWaitExit
                                        : ExecutorEopTraceStage::SemPost;
    ExecutorEopTraceRecord(stage, slot, static_cast<u32>(before),
                           static_cast<u32>(after), static_cast<u32>(result), thread_kind,
                           static_cast<u64>(native));
}

void ExecutorEopTraceSubmitDonePulse(u64 pulse) {
    if (!ExecutorEopTraceEnabled()) {
        return;
    }

    auto& state = ExecutorGetEopTraceState();
    const u64 hle_submits = state.hle_submits.load(std::memory_order_relaxed);
    if (ExecutorEopTraceExplicitDumpEnabled() && hle_submits != 0 &&
        !state.explicit_dumped.exchange(true, std::memory_order_acq_rel)) {
        ExecutorEopTraceDump("explicit-marker", pulse);
    }

    // Thread3's forced SubmitDone remains alive during the Coach wait. Two consecutive pulses with
    // no HLE submit and no parser/IRQ progress (~6 s in this title) are therefore a reliable bounded
    // stall trigger. Arm only after real graphics traffic so startup housekeeping cannot consume it.
    if (hle_submits == 0 || state.parser_begin.load(std::memory_order_relaxed) == 0) {
        return;
    }
    const u64 previous_hle = state.pulse_last_hle.exchange(hle_submits, std::memory_order_relaxed);
    if (previous_hle != hle_submits) {
        state.stagnant_pulses.store(0, std::memory_order_relaxed);
        return;
    }
    const u32 stagnant = state.stagnant_pulses.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 age_us = ExecutorEopTraceGap(
        ExecutorEopTraceNowUs(state), state.last_progress_us.load(std::memory_order_relaxed));
    if (stagnant < 2 || age_us < 5'000'000) {
        return;
    }

    u64 dumped_epoch = state.auto_dump_hle_epoch.load(std::memory_order_relaxed);
    while (dumped_epoch != hle_submits) {
        if (state.auto_dump_hle_epoch.compare_exchange_weak(
                dumped_epoch, hle_submits, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            ExecutorEopTraceDump("submitdone-stall", pulse);
            break;
        }
    }
}
#endif

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

bool Liverpool::CopyIndirectAtEncounter(Pm4Engine engine, VAddr address, u32 num_words,
                                        std::vector<u32>& owned,
                                        const Pm4SubmissionStatePtr& state) const {
    constexpr VAddr GpuAddressLimit = 1ULL << 40;
    constexpr u32 MaxIbWords = (1u << 20) - 1;
    constexpr u64 MaxSubmissionWords = 512_MB / sizeof(u32);
    constexpr u32 MaxSubmissionNodes = 65536;

    const char* const engine_name = [engine] {
        switch (engine) {
        case Pm4Engine::Graphics:
            return "gfx";
        case Pm4Engine::Constant:
            return "ce";
        case Pm4Engine::Compute:
            return "compute";
        }
        return "unknown";
    }();
    const auto fail = [&](const char* reason) {
        if (state) {
            state->abort = true;
        }
        LOG_ERROR(Lib_GnmDriver,
                  "Rejecting nested PM4 IB: engine={}, reason={}, address={:#x}, words={}",
                  engine_name, reason, address, num_words);
#ifdef __ANDROID__
        __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                            "[EXECUTOR_PM4_IB_REJECTED] engine=%s reason=%s addr=0x%llx words=%u",
                            engine_name, reason, static_cast<unsigned long long>(address),
                            num_words);
#endif
        owned.clear();
        return false;
    };

    if (!state) {
        return fail("missing-state");
    }
    if (address == 0 || (address & (alignof(u32) - 1)) != 0 || num_words == 0 ||
        num_words > MaxIbWords) {
        return fail("invalid-range");
    }
    const u64 num_bytes = u64{num_words} * sizeof(u32);
    if (address >= GpuAddressLimit || num_bytes > GpuAddressLimit - address) {
        return fail("outside-gpu-va");
    }
    const Pm4SnapshotKey key{address, num_words, engine};
    auto submit_snapshot = state->submit_snapshots.find(key);
    const bool gpu_modified = rasterizer && rasterizer->IsMemoryGpuModified(address, num_bytes);
    if (submit_snapshot != state->submit_snapshots.end() && !gpu_modified) {
        owned = submit_snapshot->second;
        return true;
    }
    const bool already_accounted = submit_snapshot != state->submit_snapshots.end();
    if (!already_accounted &&
        (state->copied_nodes >= MaxSubmissionNodes ||
         state->copied_words > MaxSubmissionWords - num_words)) {
        return fail("submission-budget");
    }

    // Match PC's point of observation. Any earlier GPU writer in this submission is made visible
    // first; only then do we freeze the child for the potentially long mobile parse/compile window.
    const bool gpu_backing_synchronized = rasterizer && rasterizer->ReadMemory(address, num_bytes);
    auto* memory = Core::Memory::Instance();
    owned.resize(num_words);
    if (!memory || !memory->TryCopyReadableMemory(address, owned.data(), num_bytes,
                                                   gpu_backing_synchronized)) {
        return fail("unreadable");
    }
    if (!already_accounted) {
        state->copied_words += num_words;
        ++state->copied_nodes;
        state->submit_snapshots.emplace(key, owned);
    } else {
        // A GPU writer produced a newer generation than the CPU submit snapshot. Keep the fresh
        // bytes as this submission's generation so a repeated reference cannot fall back to stale
        // submit-time contents after ReadMemory clears the dirty bit.
        submit_snapshot->second = owned;
    }
    return true;
}

static bool TestWaitRegMemValue(const PM4CmdWaitRegMem& wait_reg_mem, u32 value);
#ifdef __ANDROID__
static const char* ExecutorReadWaitRegMemValue(const PM4CmdWaitRegMem& wait_reg_mem,
                                               std::span<const u32> regs, u32* value);
#endif

void Liverpool::InvalidateIndirectSnapshotsForWrite(const Pm4SubmissionStatePtr& state,
                                                     VAddr address, u64 num_bytes) const {
    if (!state || num_bytes == 0 || address > std::numeric_limits<VAddr>::max() - num_bytes) {
        return;
    }

    const VAddr write_end = address + num_bytes;
    for (auto it = state->submit_snapshots.begin(); it != state->submit_snapshots.end();) {
        const auto& key = it->first;
        const u64 snapshot_bytes = u64{key.words} * sizeof(u32);
        if (key.address > std::numeric_limits<VAddr>::max() - snapshot_bytes) {
            it = state->submit_snapshots.erase(it);
            continue;
        }
        const VAddr snapshot_end = key.address + snapshot_bytes;
        if (address < snapshot_end && key.address < write_end) {
            it = state->submit_snapshots.erase(it);
        } else {
            ++it;
        }
    }
#ifdef __ANDROID__
    std::erase_if(state->completion_writes, [&](const auto& write) {
        if (write.address > std::numeric_limits<VAddr>::max() - write.num_bytes) {
            return true;
        }
        const VAddr completion_end = write.address + write.num_bytes;
        return address < completion_end && write.address < write_end;
    });
#endif

    // This invalidates future IB encounters only. An already-active immutable owned_dcb/owned_ccb
    // still cannot observe a self-write to its later packets; supporting that requires carrying each
    // owned stream's logical guest base and patching the active vector (GPU writes additionally need
    // ordered readback). Keep that larger contract explicit instead of pretending map erasure fixes it.
}

#ifdef __ANDROID__
void Liverpool::RecordCompletionWriteForExecutor(const Pm4SubmissionStatePtr& state,
                                                  Pm4Engine engine, VAddr address, u64 value,
                                                  u32 num_bytes, u64 gpu_tick) {
    if (!ExecutorEopWaitBatchingV2() || !state || address == 0 || gpu_tick == 0 ||
        (num_bytes != sizeof(u32) && num_bytes != sizeof(u64))) {
        return;
    }
    InvalidateIndirectSnapshotsForWrite(state, address, num_bytes);
    state->completion_writes.push_back(
        Pm4SubmissionState::CompletionWrite{address, value, gpu_tick, num_bytes, engine});
    constexpr u64 PageSize = 4_KB;
    std::vector<VAddr> pages_to_track;
    {
        std::scoped_lock lock{executor_pending_completion_mutex};
        ++executor_pending_completion_writes[ExecutorPendingCompletionWrite{
            address, gpu_tick, num_bytes}];
        const VAddr first_page = Common::AlignDown(address, PageSize);
        const VAddr last_page = Common::AlignDown(address + num_bytes - 1, PageSize);
        for (VAddr page = first_page;; page += PageSize) {
            if (rasterizer->IsMapped(page, PageSize)) {
                auto& refs = executor_pending_completion_page_refs[page];
                if (refs++ == 0) {
                    pages_to_track.push_back(page);
                }
            }
            if (page == last_page) {
                break;
            }
        }
    }
    for (const VAddr page : pages_to_track) {
        rasterizer->TrackPendingCompletionRead(page, PageSize);
    }
}

void Liverpool::CompletePendingCompletionWriteForExecutor(VAddr address, u32 num_bytes,
                                                          u64 gpu_tick) {
    if (!ExecutorEopWaitBatchingV2() || address == 0 || gpu_tick == 0) {
        return;
    }
    constexpr u64 PageSize = 4_KB;
    std::vector<VAddr> pages_to_untrack;
    {
        std::scoped_lock lock{executor_pending_completion_mutex};
        const ExecutorPendingCompletionWrite key{address, gpu_tick, num_bytes};
        const auto it = executor_pending_completion_writes.find(key);
        if (it != executor_pending_completion_writes.end()) {
            const u32 completed_refs = it->second;
            const VAddr first_page = Common::AlignDown(it->first.address, PageSize);
            const VAddr last_page =
                Common::AlignDown(it->first.address + it->first.num_bytes - 1, PageSize);
            for (VAddr page = first_page;; page += PageSize) {
                auto refs = executor_pending_completion_page_refs.find(page);
                if (refs != executor_pending_completion_page_refs.end()) {
                    ASSERT_MSG(refs->second >= completed_refs,
                               "Pending completion page watcher underflow: refs={} completed={}",
                               refs->second, completed_refs);
                    refs->second -= completed_refs;
                    if (refs->second == 0) {
                        pages_to_untrack.push_back(page);
                        executor_pending_completion_page_refs.erase(refs);
                    }
                }
                if (page == last_page) {
                    break;
                }
            }
            executor_pending_completion_writes.erase(it);
        }
    }
    for (const VAddr page : pages_to_untrack) {
        rasterizer->UntrackPendingCompletionRead(page, PageSize);
    }
}

u64 Liverpool::ConsumeCompletionWaitForExecutor(const Pm4SubmissionStatePtr& state,
                                                 Pm4Engine engine,
                                                 const PM4CmdWaitRegMem& wait) const {
    if (!ExecutorEopWaitBatchingV2() || !state ||
        wait.mem_space.Value() != PM4CmdWaitRegMem::MemSpace::Memory) {
        return 0;
    }
    const VAddr address = reinterpret_cast<VAddr>(wait.Address<u32*>());
    for (auto it = state->completion_writes.end(); it != state->completion_writes.begin();) {
        --it;
        if (it->engine == engine && it->address == address &&
            TestWaitRegMemValue(wait, static_cast<u32>(it->value))) {
            const u64 tick = it->gpu_tick;
            // One logical observation only. This prevents a later wait from matching an old EOP
            // after a direct guest-CPU reset that the PM4 parser cannot observe.
            state->completion_writes.erase(it);
            return tick;
        }
    }
    return 0;
}

bool Liverpool::SynchronizeUntrackedMemoryWaitForExecutor(const PM4CmdWaitRegMem& wait,
                                                           u32& value) {
    if (rasterizer == nullptr || IsRendererTerminal() ||
        wait.mem_space.Value() != PM4CmdWaitRegMem::MemSpace::Memory) {
        return false;
    }

    const VAddr address = reinterpret_cast<VAddr>(wait.Address<u32*>());

    // A WAIT is a real command-stream dependency even when its producer is not an EOP/RELEASE_MEM
    // completion tracked by RecordCompletionWriteForExecutor. In particular, a shader, DMA or
    // ordinary buffer write can be held in the scheduler's current, not-yet-submitted command
    // buffer. Polling CPU backing before publishing that batch creates a permanent 0xN -> 0xN+1
    // deadlock: the front Liverpool submission waits, while every later submission remains behind
    // it on the same guest queue.
    //
    // Logical EOP->WAIT pairs have already returned from ConsumeCompletionWaitForExecutor above, so
    // this is the correctness boundary for the remaining, genuine dependencies. Flush once before
    // the caller enters its yield loop. If BufferCache owns newer device bytes, perform the precise
    // readback as well; that wait is required by the guest packet and does not add serialization to
    // independent or logically folded waits.
    FlushGpuCompletionBatchForExecutor(true, true);
    if (rasterizer->IsMemoryGpuModified(address, sizeof(value))) {
        rasterizer->ReadMemory(address, sizeof(value));
    }

    const char* const source = ExecutorReadWaitRegMemValue(wait, regs.reg_array, &value);
    const bool resolved = TestWaitRegMemValue(wait, value);
    static std::atomic<u32> log_budget{0};
    const u32 ordinal = log_budget.fetch_add(1, std::memory_order_relaxed);
    if (ordinal < 32 || (ordinal != 0 && (ordinal & (ordinal - 1)) == 0)) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_UNTRACKED_MEMORY_WAIT_SYNC] n=%u addr=0x%llx value=0x%x ref=0x%x "
            "mask=0x%x source=%s resolved=%u",
            ordinal + 1, static_cast<unsigned long long>(address), value, wait.ref, wait.mask,
            source, resolved ? 1u : 0u);
    }
    return resolved;
}
#endif

bool Liverpool::ResolvePendingCompletionReadForExecutor(VAddr address, u64 size) {
#ifdef __ANDROID__
    if (!ExecutorEopWaitBatchingV2() || size == 0 ||
        address > std::numeric_limits<VAddr>::max() - size) {
        return false;
    }

    constexpr u64 PageSize = 4_KB;
    const VAddr first_read_page = Common::AlignDown(address, PageSize);
    const VAddr last_read_page = Common::AlignDown(address + size - 1, PageSize);
    u64 dependency_tick = 0;
    {
        std::scoped_lock lock{executor_pending_completion_mutex};
        for (const auto& [write, refs] : executor_pending_completion_writes) {
            (void)refs;
            if (write.address > std::numeric_limits<VAddr>::max() - write.num_bytes) {
                continue;
            }
            // PageManager protects at 4-KB granularity. A CPU load of unrelated bytes on the same
            // page faults on our completion watcher as well; matching only the exact fence bytes
            // leaves the page protected and re-executes that instruction forever.
            const VAddr first_write_page = Common::AlignDown(write.address, PageSize);
            const VAddr last_write_page =
                Common::AlignDown(write.address + write.num_bytes - 1, PageSize);
            if (first_read_page <= last_write_page &&
                first_write_page <= last_read_page) {
                dependency_tick = std::max(dependency_tick, write.gpu_tick);
            }
        }
    }
    if (dependency_tick == 0 || rasterizer == nullptr || IsRendererTerminal()) {
        return false;
    }

    SendCommand<true>([this, dependency_tick] {
        if (rasterizer == nullptr || IsRendererTerminal()) {
            return;
        }
        // This executes on Liverpool's sole command-processor thread. It is therefore safe to end
        // and submit the command buffer currently being recorded; doing this from the signal/fault
        // thread would race command-buffer ownership.
        auto& scheduler = rasterizer->GetScheduler();
        FlushGpuCompletionBatchForExecutor(true, true);
        scheduler.Wait(dependency_tick);
        scheduler.WaitRetirement(dependency_tick);
    });
    const u64 read_flush_ordinal =
        executor_pending_read_flushes.fetch_add(1, std::memory_order_relaxed) + 1;
    if (read_flush_ordinal <= 32 ||
        (read_flush_ordinal & (read_flush_ordinal - 1)) == 0) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_PENDING_COMPLETION_READ] n=%llu addr=0x%llx bytes=%llu tick=%llu "
            "action=flush_wait_retirement",
            static_cast<unsigned long long>(read_flush_ordinal),
            static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
            static_cast<unsigned long long>(dependency_tick));
    }
    return true;
#else
    return false;
#endif
}

void Liverpool::SnapshotCpuIndirectGraph(Pm4Engine engine, std::span<const u32> stream,
                                         const Pm4SubmissionStatePtr& state, u32 depth) const {
    constexpr VAddr GpuAddressLimit = 1ULL << 40;
    constexpr u32 MaxIbWords = (1u << 20) - 1;
    // The encounter-time parser retains a larger hard safety budget. Submit-time ownership is only
    // an optimization/correctness snapshot for ordinary CPU-authored command graphs and must remain
    // bounded enough for mobile memory pressure.
    constexpr u64 MaxSnapshotWords = 64_MB / sizeof(u32);
    constexpr u32 MaxSnapshotNodes = 8192;
    constexpr u32 MaxSnapshotDepth = 32;

    if (!state || depth > MaxSnapshotDepth) {
        return;
    }

    // This is a best-effort ownership walk, not a second PM4 interpreter. Invalid or dynamic data is
    // left for the production encounter-time path, which retains its strict validation and aborts.
    while (!stream.empty()) {
        const auto* header = reinterpret_cast<const PM4Header*>(stream.data());
        if (header->type == 2) {
            stream = stream.subspan(1);
            continue;
        }
        if (header->type != 3) {
            return;
        }

        const size_t packet_words = static_cast<size_t>(header->type3.NumWords()) + 1;
        if (packet_words == 0 || packet_words > stream.size()) {
            return;
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const bool is_graphics_ib =
            engine == Pm4Engine::Graphics && opcode == PM4ItOpcode::IndirectBuffer;
        const bool is_constant_ib =
            engine == Pm4Engine::Constant && opcode == PM4ItOpcode::IndirectBufferConst;
        const bool is_compute_ib =
            engine == Pm4Engine::Compute && opcode == PM4ItOpcode::IndirectBuffer;
        if ((is_graphics_ib || is_constant_ib || is_compute_ib) &&
            packet_words == sizeof(PM4CmdIndirectBuffer) / sizeof(u32)) {
            const auto* indirect = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            const bool chain = indirect->chain.Value() != 0;
            const VAddr address = reinterpret_cast<VAddr>(indirect->Address<const u32>());
            const u32 num_words = indirect->ib_size.Value();
            const u64 num_bytes = u64{num_words} * sizeof(u32);
            const bool valid_range = address != 0 && (address & (alignof(u32) - 1)) == 0 &&
                                     num_words != 0 && num_words <= MaxIbWords &&
                                     address < GpuAddressLimit &&
                                     num_bytes <= GpuAddressLimit - address;
            if (valid_range) {
                const Pm4SnapshotKey key{address, num_words, engine};
                // Never freeze stale CPU backing for a GPU-authored child. It must be synchronized
                // at encounter, after the producer packets ahead of it have executed.
                const bool gpu_modified_at_submit =
                    rasterizer && rasterizer->IsMemoryGpuModified(address, num_bytes);
                if (!state->submit_snapshots.contains(key) &&
                    !gpu_modified_at_submit && state->copied_nodes < MaxSnapshotNodes &&
                    state->copied_words <= MaxSnapshotWords - num_words) {
                    std::vector<u32> child(num_words);
                    auto* memory = Core::Memory::Instance();
                    // false means CPU-readable mappings only. GPU-only/generated IBs remain live and
                    // are synchronized by CopyIndirectAtEncounter after their producer executes.
                    if (memory && memory->TryCopyReadableMemory(address, child.data(), num_bytes,
                                                               false)) {
                        state->copied_words += num_words;
                        ++state->copied_nodes;
                        auto [it, inserted] =
                            state->submit_snapshots.emplace(key, std::move(child));
                        if (inserted) {
                            SnapshotCpuIndirectGraph(engine, it->second, state, depth + 1);
                        }
                    }
                }
            }
            // CHAIN transfers control permanently; bytes following this packet in the parent are
            // unreachable and must not consume snapshot budget or acquire false ownership.
            if (chain) {
                return;
            }
        }

        stream = stream.subspan(packet_words);
    }
}

static bool TestWaitRegMemValue(const PM4CmdWaitRegMem& wait_reg_mem, u32 value) {
    switch (wait_reg_mem.function.Value()) {
    case PM4CmdWaitRegMem::Function::Always:
        return true;
    case PM4CmdWaitRegMem::Function::LessThan:
        return (value & wait_reg_mem.mask) < wait_reg_mem.ref;
    case PM4CmdWaitRegMem::Function::LessThanEqual:
        return (value & wait_reg_mem.mask) <= wait_reg_mem.ref;
    case PM4CmdWaitRegMem::Function::Equal:
        return (value & wait_reg_mem.mask) == wait_reg_mem.ref;
    case PM4CmdWaitRegMem::Function::NotEqual:
        return (value & wait_reg_mem.mask) != wait_reg_mem.ref;
    case PM4CmdWaitRegMem::Function::GreaterThanEqual:
        return (value & wait_reg_mem.mask) >= wait_reg_mem.ref;
    case PM4CmdWaitRegMem::Function::GreaterThan:
        return (value & wait_reg_mem.mask) > wait_reg_mem.ref;
    case PM4CmdWaitRegMem::Function::Reserved:
    default:
        UNREACHABLE();
    }
}

static bool LooksLikeAndroidHostPointer(const void* address) {
#ifdef __ANDROID__
    const uintptr_t raw = reinterpret_cast<uintptr_t>(address);
    // PS4 GPU/guest addresses are below 2^48 in this runtime. Tagged Scudo/MTE pointers and
    // unowned host labels/fences must be resolved by an HLE alias, not by MemoryManager.
    return (raw >> 56) != 0 || raw >= (1ull << 48);
#else
    return false;
#endif
}

#ifdef __ANDROID__
static const char* ExecutorReadWaitRegMemValue(const PM4CmdWaitRegMem& wait_reg_mem,
                                               std::span<const u32> regs, u32* value) {
    if (!value) {
        return "invalid";
    }

    if (wait_reg_mem.mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Register) {
        *value = regs[wait_reg_mem.Reg()];
        return "register";
    }

    const u64* wait_addr = wait_reg_mem.Address<u64*>();
    if (Libraries::VideoOut::ExecutorTryReadVoLabelAlias(wait_addr, value)) {
        return "vo_label_alias";
    }

    const VAddr addr = reinterpret_cast<VAddr>(wait_addr);
    auto* memory = Core::Memory::Instance();
    if (memory && memory->IsValidMapping(addr, sizeof(u32))) {
        memory->CopySparseMemory(addr, reinterpret_cast<u8*>(value), sizeof(u32));
        return "memory_backing";
    }

    if (!LooksLikeAndroidHostPointer(wait_addr)) {
        std::memcpy(value, wait_addr, sizeof(u32));
        return "direct";
    }

    return "unresolved";
}

static u32 ExecutorReadMappedWaitValueFast(const u64* wait_addr) {
    const auto raw = reinterpret_cast<uintptr_t>(wait_addr);
    if ((raw & (alignof(u32) - 1)) != 0) {
        u32 value{};
        std::memcpy(&value, wait_addr, sizeof(value));
        return value;
    }

    // ExecutorReadWaitRegMemValue has already proved that this is a live guest mapping. A PM4
    // semaphore remains owned by the submitted command stream while this coroutine is suspended,
    // so repeating the VMA lookup and taking MemoryManager's shared mutex for every poll only
    // serializes the GPU command processor against unrelated guest map operations. The acquire load
    // preserves publication ordering with the CPU/GPU fence writer while matching the direct-memory
    // polling contract used by desktop shadPS4.
    auto* const value_ptr =
        reinterpret_cast<u32*>(const_cast<u64*>(wait_addr));
    return std::atomic_ref<u32>{*value_ptr}.load(std::memory_order_acquire);
}
#else
bool ExecutorWriteFenceToCpuAddress(const char*, void* address, u64 data, u32 num_bytes) {
    std::memcpy(address, &data, num_bytes);
    return true;
}

static const char* ExecutorReadWaitRegMemValue(const PM4CmdWaitRegMem& wait_reg_mem,
                                               std::span<const u32> regs, u32* value) {
    if (!value) {
        return "invalid";
    }
    if (wait_reg_mem.mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Register) {
        *value = regs[wait_reg_mem.Reg()];
        return "register";
    }
    *value = *wait_reg_mem.Address<u32*>();
    return "direct_memory";
}
#endif

Liverpool::Liverpool() {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

bool Liverpool::IsRendererTerminal() const noexcept {
    return rasterizer != nullptr && rasterizer->IsDeviceTerminal();
}

void Liverpool::MarkRendererTerminal(const char* where) noexcept {
    if (rasterizer != nullptr) {
        rasterizer->MarkDeviceTerminal(vk::Result::eTimeout, where);
    }
}

#ifdef __ANDROID__
u64 Liverpool::ArmGpuCompletionTickForExecutor() {
    auto& scheduler = rasterizer->GetScheduler();
    const u64 current_tick = scheduler.CurrentTick();
    ++executor_completion_point_count;
    if (scheduler.HasPendingWork()) {
        // Completion callbacks armed before the next dependency boundary can conservatively share
        // one tick. WAIT_REG_MEM forces the batch out before polling; independent submits remain in
        // one host command buffer until queue drain or SubmitDone.
        executor_completion_flush_pending = true;
        return current_tick;
    }
    // Presentation or an earlier completion point already flushed the relevant commands.
    return current_tick > 1 ? current_tick - 1 : 0;
}

void Liverpool::FlushGpuCompletionBatchForExecutor(bool force, bool synchronization_wait) {
    if (rasterizer == nullptr || (!force && !executor_completion_flush_pending)) {
        return;
    }
    if (synchronization_wait && executor_completion_flush_pending) {
        ++executor_metrics_wait_flushes;
    }
    rasterizer->Flush();
    executor_completion_flush_pending = false;
}
#endif

void Liverpool::WaitRasterizerIdleForExecutor() {
    {
        // Strict drain (Codex): wait until the PM4 coroutine has not only consumed the command buffer
        // (num_submits==0, num_commands==0) but ALSO run the submit-done OnSubmit/Flush block
        // (submit_done==false). Waiting on num_submits==0 alone races the OnSubmit/Flush tail.
        //
        // Time-bound the wait: when a title (Sonic/RSDK on JIT) submits a DCB whose tail flip
        // WAIT_REG_MEM never resolves — the flip needs the present, and the present is gated BEHIND
        // this very drain — the coroutine never reaches the idle predicate and the render thread hangs
        // here forever, so no frame is ever presented. Bounding the wait lets the render thread fall
        // through to the present with whatever the GPU has rendered instead of deadlocking.
        // Time-bound the lock acquisition itself: if a faulting GPU-side worker was parked while
        // holding submit_mutex (worker-park survival aid), a plain unique_lock here would deadlock
        // the render thread forever. try_lock_for lets it give up and present regardless.
        // Do NOT take submit_mutex here: num_submits/num_commands/submit_done are all std::atomic, so
        // they can be polled lock-free. Taking submit_mutex was fatal — a host-side heap corruption
        // (Sonic JIT) intermittently makes bionic FORTIFY-abort pthread_mutex_lock on this very
        // mutex ("destroyed mutex"), killing the render thread right here (stuck at before_waitidle,
        // no timeout log). Lock-free polling with a bound lets the render always fall through to the
        // present regardless of the mutex's state or a parked GPU worker.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        bool idle = false;
        while (std::chrono::steady_clock::now() < deadline) {
            if (num_submits.load(std::memory_order_acquire) == 0 &&
                num_commands.load(std::memory_order_acquire) == 0 &&
                !submit_done.load(std::memory_order_acquire)) {
                idle = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!idle) {
            static std::atomic<u32> s_timeout_budget{0};
            const u32 n = s_timeout_budget.fetch_add(1, std::memory_order_relaxed);
            if (n < 32) {
                __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                    "[EXECUTOR_RASTERIZER_IDLE_TIMEOUT] n=%u (lock-free poll; present "
                                    "without full drain)",
                                    n);
            }
        }
    }
    // rasterizer->Finish() drains the Vulkan queue via the scheduler, which locks a scheduler mutex.
    // When a host-side corruption has scribbled that mutex, Finish() FORTIFY-aborts and kills the
    // render thread right before it can present. Skippable via EXECUTOR_SKIP_RASTERIZER_FINISH so the
    // render can reach the present regardless (best-effort; may present a not-fully-drained frame).
    const bool skip_finish = ExecutorSkipRasterizerFinish();
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_RASTERIZER_IDLE] cv_passed=1 rasterizer=%d skipFinish=%d",
                        rasterizer != nullptr, skip_finish ? 1 : 0);
    if (rasterizer && !skip_finish && !IsRendererTerminal()) {
        rasterizer->Finish();
    }
    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                        "[EXECUTOR_RASTERIZER_IDLE] finish_done=1");
}

void Liverpool::ProcessCommands() {
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        callback();
    }
}

bool Liverpool::PublishSubmit(u32 qid, Task::Handle handle, u32* published_submit_count) {
    ASSERT(qid < NumTotalQueues);

    // The queue entry and the condition-variable predicate are one publication transaction.
    // submit_mutex serializes producers with the sleeper/recount contract; the MPSC queue itself
    // publishes its node with release ordering and requires no consumer-side mutex.
    u32 count = 0;
    {
        std::unique_lock submit_lock{submit_mutex};
        if (IsRendererTerminal()) {
            return false;
        }

        auto& queue = mapped_queues[qid];
        queue.submits.emplace(handle);
        submit_ready_mask.fetch_or(u64{1} << qid, std::memory_order_release);

        const u32 required_queue_count = qid + 1;
        const u32 current_queue_count = num_mapped_queues.load(std::memory_order_relaxed);
        if (required_queue_count > current_queue_count) {
            num_mapped_queues.store(required_queue_count, std::memory_order_release);
        }
        count = num_submits.fetch_add(1, std::memory_order_release) + 1;
    }
    if (published_submit_count != nullptr) {
        *published_submit_count = count;
    }
    // Wake only after releasing the publication lock.
    submit_cv.notify_one();
    return true;
}

u32 Liverpool::RecountQueuedSubmits() {
    // Called only by the sole consumer after an apparently empty round-robin pass. Holding
    // submit_mutex prevents a producer from linking another node, yielding an exact snapshot.
    std::unique_lock submit_lock{submit_mutex};
    const u32 queue_count = num_mapped_queues.load(std::memory_order_acquire);
    u32 actual_submits = 0;
    u64 actual_ready_mask = 0;
    for (u32 qid = 0; qid < queue_count; ++qid) {
        auto& queue = mapped_queues[qid];
        const u32 queued = static_cast<u32>(queue.submits.size());
        actual_submits += queued;
        if (queued != 0) {
            actual_ready_mask |= u64{1} << qid;
        }
    }

    submit_ready_mask.store(actual_ready_mask, std::memory_order_release);
    const u32 reported_submits = num_submits.exchange(actual_submits, std::memory_order_release);
#ifdef __ANDROID__
    if (reported_submits != actual_submits && (ExecutorTracePm4() || ExecutorLiveGnmChecks())) {
        __android_log_print(ANDROID_LOG_WARN, "EXECUTOR",
                            "[EXECUTOR_GNM_THREAD] submit_invariant_recount reported=%u actual=%u "
                            "commands=%u mappedQueues=%u action=repair_exact",
                            reported_submits, actual_submits, num_commands.load(), queue_count);
    }
#endif
    if (reported_submits != actual_submits || actual_submits == 0) {
        submit_cv.notify_all();
    }
    return actual_submits;
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
    // Liverpool is the sole consumer of every mapped queue. Keep a copy of each current front
    // handle while its coroutine is suspended; producers only append to the lock-free tail and
    // cannot change the task this consumer must resume.
    std::array<Task::Handle, NumTotalQueues> active_tasks{};
#ifdef __ANDROID__
    // A healthy Vulkan device remains healthy for virtually the whole process lifetime. Polling
    // its two acquire atomics before every tiny PM4 coroutine resume consumed a measurable slice
    // of the sole GPU command processor. Submit/flush boundaries below still check immediately;
    // only the hot inner round-robin loop is amortized. At most 63 queue steps are discarded after
    // another thread publishes a terminal device, which is far below one guest submission.
    u32 terminal_poll_counter = 0;
#endif

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
        }
        if (stoken.stop_requested()) {
            break;
        }

        // Every accepted wake must complete the guest-visible GpuIdle edge, including renderer
        // terminal/drop paths. Under EOP batching this is an enqueue/parse edge, not a GPU timeline
        // drain: completion writes retire asynchronously, and a real guest CPU read of a pending
        // completion page forces the exact tick through ResolvePendingCompletionReadForExecutor.
        // Waiting here used to turn every transient empty Liverpool queue into a CPU<->GPU round
        // trip (roughly ten per frame in Bloodborne), defeating the batching performed by the PM4
        // WAIT path itself.
        SCOPE_EXIT {
#ifdef __ANDROID__
            if (rasterizer && ExecutorBoundedRasterizerPipeline()) {
                // Admission gate: do not release gnmdriver's GpuIdle/submission_lock edge for frame
                // N+1 until frame N has completed. Frame N+1 remains in flight, so the guest and GPU
                // overlap, but the guest cannot author/overwrite resources for frame N+2 early.
                auto& scheduler = rasterizer->GetScheduler();
                const auto retire_front = [this] {
                    executor_async_frame_head =
                        (executor_async_frame_head + 1) % executor_async_frame_ticks.size();
                    --executor_async_frame_count;
                };
                // Retire every tick already visible in the timeline without entering the blocking
                // Vulkan wait path. This does not enlarge the two-frame correctness window.
                while (executor_async_frame_count != 0 &&
                       scheduler.IsFree(executor_async_frame_ticks[executor_async_frame_head])) {
                    retire_front();
                    ++executor_admission_free_retirements;
                }
                if (executor_async_frame_count == executor_async_frame_ticks.size()) {
                    const auto wait_begin = std::chrono::steady_clock::now();
                    scheduler.Wait(executor_async_frame_ticks[executor_async_frame_head]);
                    const u64 wait_ns = static_cast<u64>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - wait_begin)
                            .count());
                    ++executor_admission_wait_count;
                    executor_admission_wait_ns += wait_ns;
                    executor_admission_wait_max_ns =
                        std::max(executor_admission_wait_max_ns, wait_ns);
                    retire_front();
                }
            }
#endif
            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
        };

        VideoCore::StartCapture();

        curr_qid = -1;

        while (num_submits || num_commands) {
            ProcessCommands();
            if (num_submits.load(std::memory_order_acquire) == 0) {
                break;
            }

#ifdef __ANDROID__
            const bool renderer_terminal =
                ((terminal_poll_counter++ & 63u) == 0u) && IsRendererTerminal();
#else
            const bool renderer_terminal = IsRendererTerminal();
#endif
            if (renderer_terminal) {
                // A suspended WAIT_REG_MEM coroutine can otherwise keep num_submits non-zero even
                // after the Vulkan scheduler has entered terminal state. Destroy all suspended GPU
                // tasks, then complete this outer iteration's scope-guarded GpuIdle edge.
                {
                    std::unique_lock submit_lock{submit_mutex};
                    const u32 queue_count =
                        num_mapped_queues.load(std::memory_order_acquire);
                    for (u32 qid = 0; qid < queue_count; ++qid) {
                        auto& terminal_queue = mapped_queues[qid];
                        while (!terminal_queue.submits.empty()) {
                            auto task = terminal_queue.submits.front();
                            terminal_queue.submits.pop();
                            if (task) {
                                task.destroy();
                            }
                        }
                        active_tasks[qid] = {};
                    }
                    num_submits.store(0, std::memory_order_release);
                    submit_ready_mask.store(0, std::memory_order_release);
#ifdef __ANDROID__
                    executor_completion_flush_pending = false;
#endif
                    submit_done = false;
                    submit_cv.notify_all();
                }
                break;
            }

            const u32 queue_count = num_mapped_queues.load(std::memory_order_acquire);
            ASSERT(queue_count != 0);
            const u64 valid_queue_mask = (u64{1} << queue_count) - 1;
            u64 ready_mask =
                submit_ready_mask.load(std::memory_order_acquire) & valid_queue_mask;
            if (ready_mask == 0) {
                const u32 reported_submits = num_submits.load(std::memory_order_acquire);
                const u32 actual_submits = RecountQueuedSubmits();
#ifdef __ANDROID__
                if ((ExecutorTracePm4() || ExecutorLiveGnmChecks()) &&
                    reported_submits == actual_submits && actual_submits != 0) {
                    __android_log_print(
                        ANDROID_LOG_WARN, "EXECUTOR",
                        "[EXECUTOR_GNM_THREAD] ready_mask_empty_with_work submits=%u commands=%u "
                        "mappedQueues=%u action=rebuild",
                        actual_submits, num_commands.load(), queue_count);
                }
#endif
                if (actual_submits == 0) {
                    break;
                }
                ready_mask =
                    submit_ready_mask.load(std::memory_order_acquire) & valid_queue_mask;
                if (ready_mask == 0) {
                    continue;
                }
            }

            const u32 start_qid =
                curr_qid < 0 || static_cast<u32>(curr_qid + 1) >= queue_count
                    ? 0
                    : static_cast<u32>(curr_qid + 1);
            const u64 ready_at_or_after = ready_mask & (~u64{0} << start_qid);
            curr_qid = static_cast<s32>(std::countr_zero(ready_at_or_after != 0
                                                             ? ready_at_or_after
                                                             : ready_mask));

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task = active_tasks[curr_qid];
            if (!task) {
                task = queue.submits.front();
                if (!task) {
                    // A stale bit can be observed while the consumer retires the final node. Clear
                    // it, then recheck: a producer racing the clear either becomes visible here and
                    // has its bit restored, or publishes its own set after this recheck.
                    const u64 queue_bit = u64{1} << curr_qid;
                    submit_ready_mask.fetch_and(~queue_bit, std::memory_order_acq_rel);
                    task = queue.submits.front();
                    if (task) {
                        submit_ready_mask.fetch_or(queue_bit, std::memory_order_release);
                        active_tasks[curr_qid] = task;
                    } else {
                        continue;
                    }
                } else {
                    active_tasks[curr_qid] = task;
                }
            }
#ifdef __ANDROID__
            if (ExecutorTracePm4()) {
                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                    "[EXECUTOR_GNM_THREAD] before_task_resume q=%d submits=%u commands=%u",
                                    curr_qid, num_submits.load(), num_commands.load());
            }
#endif
            task.resume();
#ifdef __ANDROID__
            if (ExecutorTracePm4()) {
                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                    "[EXECUTOR_GNM_THREAD] after_task_resume q=%d done=%u submits=%u commands=%u",
                                    curr_qid, task.done() ? 1u : 0u, num_submits.load(),
                                    num_commands.load());
            }
#endif

            if (task.done()) {
#ifdef __ANDROID__
                if (ExecutorTracePm4()) {
                    __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                        "[EXECUTOR_GNM_THREAD] before_task_destroy q=%d submits=%u",
                                        curr_qid, num_submits.load());
                }
#endif
                active_tasks[curr_qid] = {};
                task.destroy();
#ifdef __ANDROID__
                ++executor_guest_submit_count;
                if (ExecutorTracePm4()) {
                    __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                        "[EXECUTOR_GNM_THREAD] after_task_destroy q=%d submits=%u",
                                        curr_qid, num_submits.load());
                }
#endif

                u32 remaining_submits = 0;
                {
                    ASSERT_MSG(!queue.submits.empty(),
                               "Liverpool submit queue emptied during execution");
                    queue.submits.pop();
                    if (queue.submits.empty()) {
                        const u64 queue_bit = u64{1} << curr_qid;
                        submit_ready_mask.fetch_and(~queue_bit, std::memory_order_acq_rel);
                        if (!queue.submits.empty()) {
                            submit_ready_mask.fetch_or(queue_bit, std::memory_order_release);
                        }
                    }
                    ASSERT_MSG(num_submits.load(std::memory_order_relaxed) != 0,
                               "Liverpool submit counter underflow");
                    remaining_submits =
                        num_submits.fetch_sub(1, std::memory_order_release) - 1;
#ifdef __ANDROID__
                    if (ExecutorTracePm4()) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_GNM_THREAD] after_submit_decrement q=%d submits=%u commands=%u submit_done=%u",
                                            curr_qid, num_submits.load(), num_commands.load(),
                                            submit_done ? 1u : 0u);
                    }
#endif
                }
                if (remaining_submits == 0) {
                    // Drainers only need the zero edge. Acquire their predicate mutex after
                    // publishing zero: a waiter is then either already asleep (and receives this
                    // notify) or observes zero before sleeping. Avoid serialising every intermediate
                    // task retirement against all MPSC submit producers.
                    std::scoped_lock submit_lock{submit_mutex};
                    submit_cv.notify_all();
                }
            }
        }

        if (submit_done) {
#ifdef __ANDROID__
            if (ExecutorTracePm4()) {
                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                    "[EXECUTOR_GNM_THREAD] before_submit_done_tail rasterizer=%u",
                                    rasterizer ? 1u : 0u);
            }
#endif
            VideoCore::EndCapture();
            if (rasterizer && !IsRendererTerminal()) {
                rasterizer->OnSubmit();
#ifdef __ANDROID__
                // Live full-resolution targets are selected by Presenter::PrepareFrame at the
                // PatchedFlip boundary. Keep swapchain acquire/present exclusively on VideoOut's
                // normal presentation path; a second WSI transaction from this submit tail can
                // deadlock the GPU processor against PresentThread.
#endif
#ifdef __ANDROID__
                const bool bounded_pipeline = ExecutorBoundedRasterizerPipeline();
                auto& scheduler = rasterizer->GetScheduler();
#endif
                // The live presenter flushes the shared draw scheduler with WSI semaphores. This
                // follow-up is an empty tick in that case, and remains required when there was no
                // full-resolution target or swapchain acquisition was skipped.
                FlushGpuCompletionBatchForExecutor(true, false);
#ifdef __ANDROID__
                if (bounded_pipeline) {
                    const u64 current_tick = scheduler.CurrentTick();
                    const u64 latest_submitted_tick = current_tick > 1 ? current_tick - 1 : 0;
                    if (latest_submitted_tick != 0 &&
                        latest_submitted_tick != executor_async_last_tick) {
                        const u32 tail = (executor_async_frame_head + executor_async_frame_count) %
                                         executor_async_frame_ticks.size();
                        executor_async_frame_ticks[tail] = latest_submitted_tick;
                        executor_async_last_tick = latest_submitted_tick;
                        ++executor_async_frame_count;
                    }
                }
                // The old Android guard drained the entire Vulkan queue here and removed all CPU/GPU
                // overlap. Exact GPU completion publication plus a two-frame tick ring now protect
                // guest resources without globally serializing every frame. Keep both the original
                // drain and the unbounded skip-Finish oracle available for controlled A/B.
                if (!IsRendererTerminal() && !bounded_pipeline &&
                    !ExecutorSkipRasterizerFinish()) {
                    rasterizer->Finish();
                }
#endif
            }
#ifdef __ANDROID__
            if (ExecutorTracePm4()) {
                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                    "[EXECUTOR_GNM_THREAD] after_submit_done_tail");
            }
#endif
            {
                std::scoped_lock lock2{submit_mutex};
                submit_done = false;
                // Wake WaitSubmitDoneDrainedForExecutor: OnSubmit/Flush are complete now, so the replay
                // thread can safely Finish + readback + free without racing this batch's submit-end work.
                submit_cv.notify_all();
            }
#ifdef __ANDROID__
            ++executor_submit_done_frame_count;
            if ((executor_submit_done_frame_count & 15u) == 0u && rasterizer != nullptr) {
                const u64 scheduler_tick = rasterizer->GetScheduler().CurrentTick();
                const u64 guest_delta =
                    executor_guest_submit_count - executor_metrics_last_guest_submits;
                const u64 completion_delta =
                    executor_completion_point_count - executor_metrics_last_completion_points;
                const u64 host_submit_delta =
                    scheduler_tick - executor_metrics_last_scheduler_tick;
                const u64 wait_flush_delta =
                    executor_metrics_wait_flushes - executor_metrics_last_wait_flushes;
                const u64 wait_stall_delta =
                    executor_completion_wait_stalls -
                    executor_metrics_last_completion_wait_stalls;
                const u64 wait_stall_ns_delta =
                    executor_completion_wait_stall_ns -
                    executor_metrics_last_completion_wait_stall_ns;
                const u64 logical_wait_delta =
                    executor_logical_wait_count - executor_metrics_last_logical_waits;
                const u64 admission_wait_delta =
                    executor_admission_wait_count - executor_metrics_last_admission_wait_count;
                const u64 admission_wait_ns_delta =
                    executor_admission_wait_ns - executor_metrics_last_admission_wait_ns;
                const u64 admission_free_delta =
                    executor_admission_free_retirements -
                    executor_metrics_last_admission_free_retirements;
                __android_log_print(
                    ANDROID_LOG_INFO, "LSX4Native",
                    "[EXECUTOR_GPU_SUBMIT_COALESCE] frames=16 guestSubmits=%llu "
                    "completionPoints=%llu hostSubmits=%llu waitFlushes=%llu "
                    "completionWaits=%llu completionWaitMs=%.3f completionWaitMaxMs=%.3f "
                    "logicalWaits=%llu admissionWaits=%llu admissionWaitMs=%.3f "
                    "admissionWaitMaxMs=%.3f admissionFree=%llu",
                    static_cast<unsigned long long>(guest_delta),
                    static_cast<unsigned long long>(completion_delta),
                    static_cast<unsigned long long>(host_submit_delta),
                    static_cast<unsigned long long>(wait_flush_delta),
                    static_cast<unsigned long long>(wait_stall_delta),
                    static_cast<double>(wait_stall_ns_delta) / 1'000'000.0,
                    static_cast<double>(executor_completion_wait_stall_max_ns) / 1'000'000.0,
                    static_cast<unsigned long long>(logical_wait_delta),
                    static_cast<unsigned long long>(admission_wait_delta),
                    static_cast<double>(admission_wait_ns_delta) / 1'000'000.0,
                    static_cast<double>(executor_admission_wait_max_ns) / 1'000'000.0,
                    static_cast<unsigned long long>(admission_free_delta));
                executor_metrics_last_guest_submits = executor_guest_submit_count;
                executor_metrics_last_completion_points = executor_completion_point_count;
                executor_metrics_last_scheduler_tick = scheduler_tick;
                executor_metrics_last_wait_flushes = executor_metrics_wait_flushes;
                executor_metrics_last_completion_wait_stalls =
                    executor_completion_wait_stalls;
                executor_metrics_last_completion_wait_stall_ns =
                    executor_completion_wait_stall_ns;
                executor_completion_wait_stall_max_ns = 0;
                executor_metrics_last_logical_waits = executor_logical_wait_count;
                executor_metrics_last_admission_wait_count = executor_admission_wait_count;
                executor_metrics_last_admission_wait_ns = executor_admission_wait_ns;
                executor_metrics_last_admission_free_retirements =
                    executor_admission_free_retirements;
                executor_admission_wait_max_ns = 0;
            }
#endif
        } else {
#ifdef __ANDROID__
            // Do not turn a transiently empty guest-submit queue into a host submission. A title
            // commonly feeds one frame in many short bursts, so this old fallback accounted for
            // nearly all remaining host ticks after logical EOP->WAIT folding. SubmitDone/flip
            // publishes the complete frame. If the guest CPU consumes an EOP fence before that
            // boundary, the protected pending-completion page invokes the precise flush/wait path.
            if (!ExecutorEopWaitBatchingV2()) {
                FlushGpuCompletionBatchForExecutor(false, false);
            }
#endif
        }
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb,
                                           std::vector<u32> owned_ccb,
                                           Pm4SubmissionStatePtr state, u32 ib_depth) {
    if (!owned_ccb.empty()) {
        ccb = std::span<const u32>{owned_ccb};
    }
    if (!state) {
        state = std::make_shared<Pm4SubmissionState>();
    }
    FIBER_ENTER(ccb_task_name);

    if (ib_depth > 32) {
        state->abort = true;
        FIBER_EXIT;
        co_return;
    }
    u32 chain_hops = 0;

    while (!ccb.empty() && !state->abort) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            state->abort = true;
            LOG_ERROR(Lib_GnmDriver, "Invalid CE PM4 type {}", type);
            break;
        }
        const size_t packet_words = static_cast<size_t>(header->type3.NumWords()) + 1;
        if (packet_words > ccb.size()) {
            state->abort = true;
            LOG_ERROR(Lib_GnmDriver,
                      "Truncated CE PM4 packet: packet words={}, remaining words={}", packet_words,
                      ccb.size());
            break;
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            // BOUNDS CHECK (Sonic JIT host-memory corruption root): constants_heap is a static
            // 48KB array in the .so BSS. Offset() is a 16-bit guest CCB field (0..65535) and Size() a
            // 14-bit count<<2 — both can exceed 48KB, so an unchecked memcpy overruns constants_heap
            // into adjacent BSS globals (e.g. the Vulkan scheduler std::mutex), giving the shape-
            // shifting "destroyed mutex" FORTIFY/SIGSEGV crashes. Clamp to the array; drop if the
            // offset itself is out of range (a real CE would ignore an out-of-range CONST_RAM write).
            const u32 heap_size = static_cast<u32>(cblock.constants_heap.size());
            const u32 off = write_const->Offset();
            const u32 sz = write_const->Size();
            if (off < heap_size) {
                const u32 clamped = std::min(sz, heap_size - off);
                memcpy(cblock.constants_heap.data() + off, &write_const->data, clamped);
                if (clamped != sz) {
                    static std::atomic<u32> b{0};
                    if (b.fetch_add(1, std::memory_order_relaxed) < 16)
                        __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                            "[EXECUTOR_CE_WRITE_CLAMP] off=%u size=%u heap=%u clamped=%u",
                                            off, sz, heap_size, clamped);
                }
            } else {
                static std::atomic<u32> b{0};
                if (b.fetch_add(1, std::memory_order_relaxed) < 16)
                    __android_log_print(ANDROID_LOG_WARN, "LSX4Native",
                                        "[EXECUTOR_CE_WRITE_DROP] off=%u size=%u heap=%u (out of range)",
                                        off, sz, heap_size);
            }
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            // Same bounds check on the READ side so a bad Offset()/Size() cannot read past the 48KB
            // array (and write that garbage to the guest dump target).
            const u32 heap_size = static_cast<u32>(cblock.constants_heap.size());
            const u32 off = dump_const->Offset();
            const u32 sz = dump_const->Size();
            if (off < heap_size) {
                const u32 clamped = std::min(sz, heap_size - off);
                void* const destination = dump_const->Address<void*>();
                memcpy(destination, cblock.constants_heap.data() + off, clamped);
                InvalidateIndirectSnapshotsForWrite(
                    state, reinterpret_cast<VAddr>(destination), clamped);
            }
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff && !state->abort) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            if (packet_words != sizeof(PM4CmdIndirectBuffer) / sizeof(u32)) {
                state->abort = true;
                LOG_ERROR(Lib_GnmDriver, "Malformed CE indirect-buffer packet size {}",
                          packet_words);
                break;
            }
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            const VAddr address =
                reinterpret_cast<VAddr>(indirect_buffer->Address<const u32>());
            const u32 num_words = indirect_buffer->ib_size.Value();
            const bool chain = indirect_buffer->chain.Value() != 0;
            std::vector<u32> child;
            if (!CopyIndirectAtEncounter(Pm4Engine::Constant, address, num_words, child, state)) {
                break;
            }
            if (chain) {
                if (++chain_hops > 4096) {
                    state->abort = true;
                    break;
                }
                owned_ccb = std::move(child);
                ccb = std::span<const u32>{owned_ccb};
                continue;
            }
            auto task = ProcessCeUpdate({}, std::move(child), state, ib_depth + 1);
            SCOPE_EXIT {
                if (task.handle) {
                    task.handle.destroy();
                }
            };
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            state->abort = true;
            LOG_ERROR(Lib_GnmDriver, "Unknown CE PM4 type 3 opcode {:#x} with count {}",
                      static_cast<u32>(opcode), header->type3.NumWords());
            break;
        }
        ccb = NextPacket(ccb, packet_words);
    }

    FIBER_EXIT;
}

Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb,
                                           std::vector<u32> owned_dcb,
                                           std::vector<u32> owned_ccb,
                                           Pm4SubmissionStatePtr state, u32 ib_depth,
                                           Task* inherited_ce_task, VAddr logical_dcb_base) {
    // Coroutine parameters live in the coroutine frame until task.destroy(). When command-buffer
    // copying is enabled, bind the parser spans to these per-submit owners instead of Liverpool's
    // shared scratch vectors. SubmitDone may reset the shared frame offsets while this task is still
    // queued, and JIT can enter submit/submit-done HLEs concurrently from native guest threads.
    if (!owned_dcb.empty()) {
        dcb = std::span<const u32>{owned_dcb};
    }
    if (!owned_ccb.empty()) {
        ccb = std::span<const u32>{owned_ccb};
    }
    if (!state) {
        state = std::make_shared<Pm4SubmissionState>();
    }
    if (logical_dcb_base == 0) {
        logical_dcb_base = reinterpret_cast<VAddr>(dcb.data());
    }
    FIBER_ENTER(dcb_task_name);

    if (ib_depth > 32) {
        state->abort = true;
        FIBER_EXIT;
        co_return;
    }
    u32 chain_hops = 0;

#ifdef __ANDROID__
    // EXECUTOR phase-2 replay diagnostic: when EXECUTOR_TRACE_PM4 is set, log the entry and the last
    // opcode reached, so a crash/abort during a rebased-frame replay is localized without a tombstone.
    static const bool kTracePm4 = std::getenv("EXECUTOR_TRACE_PM4") != nullptr;
    const bool kLiveChecks = ExecutorLiveGnmChecks();
    ExecutorPm4LiveSummary exec_live_summary{};
    if (kTracePm4) {
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR", "[EXECUTOR_PM4_TRACE] ProcessGraphics enter dcbDw=%zu",
                            dcb.size());
    }
    if (kLiveChecks) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_PM4_BEGIN] dcbDw=%zu ccbDw=%zu rasterizer=%u",
                            dcb.size(), ccb.size(), rasterizer ? 1u : 0u);
    }
#endif

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    const bool owns_ce_context = inherited_ce_task == nullptr;
    Task local_ce_task{};
    Task& ce_task = owns_ce_context ? local_ce_task : *inherited_ce_task;
    SCOPE_EXIT {
        if (owns_ce_context && ce_task.handle) {
            ce_task.handle.destroy();
        }
    };

    // An indirect graphics buffer is part of the same DE stream. Resetting counters or creating a
    // second CE coroutine at every nested call loses the root CCB and can dereference a null task in
    // WAIT_ON_CE_COUNTER. Only the published root owns/reset/drains this context; children inherit it.
    if (owns_ce_context) {
        cblock.Reset();
    }
    if (owns_ce_context && !ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb, {}, state, ib_depth);
        RESUME_GFX(ce_task);
        if (state->abort) {
            FIBER_EXIT;
            co_return;
        }
    }
    const bool host_markers_enabled = rasterizer && Config::getVkHostMarkersEnabled();
    const bool guest_markers_enabled = rasterizer && Config::getVkGuestMarkersEnabled();

    auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
#ifdef __ANDROID__
    u32 eop_trace_dcb_dw = static_cast<u32>(dcb.size());
    ExecutorEopTraceParserBegin(base_addr, eop_trace_dcb_dw);
#endif
    while (!dcb.empty() && !state->abort) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;
        size_t packet_words = 1;
        if (type == 3) {
            packet_words = static_cast<size_t>(header->type3.NumWords()) + 1;
            if (packet_words > dcb.size()) {
                state->abort = true;
                LOG_ERROR(Lib_GnmDriver,
                          "Truncated graphics PM4 packet: packet words={}, remaining words={}",
                          packet_words, dcb.size());
                break;
            }
        }

        switch (type) {
        default:
#ifdef __ANDROID__
            // A type other than 0/2/3 cannot be a valid PM4 packet.  Keep this diagnostic next to
            // the invariant (instead of relying on the filtered fmt logger) so Android tombstones
            // identify whether the corrupt word came from an immutable submitted copy or a nested
            // guest indirect buffer. Reject this submission without killing the host process.
            {
                const auto* const base = reinterpret_cast<const u32*>(base_addr);
                const auto* const current = dcb.data();
                const size_t offset_dw = static_cast<size_t>(current - base);
                const u32 word0 = dcb.size() > 0 ? dcb[0] : 0;
                const u32 word1 = dcb.size() > 1 ? dcb[1] : 0;
                const u32 word2 = dcb.size() > 2 ? dcb[2] : 0;
                const u32 word3 = dcb.size() > 3 ? dcb[3] : 0;
                const u32 prev1 = offset_dw > 0 ? base[offset_dw - 1] : 0;
                const u32 prev2 = offset_dw > 1 ? base[offset_dw - 2] : 0;
                __android_log_print(
                    ANDROID_LOG_ERROR, "LSX4Native",
                    "[EXECUTOR_PM4_INVALID_TYPE] base=0x%llx current=0x%llx offDw=%zu "
                    "remainingDw=%zu type=%u words=%08x,%08x,%08x,%08x prev=%08x,%08x",
                    static_cast<unsigned long long>(base_addr),
                    static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(current)),
                    offset_dw, dcb.size(), type, word0, word1, word2, word3, prev1, prev2);
            }
#endif
            state->abort = true;
            LOG_ERROR(Lib_GnmDriver, "Wrong PM4 type {}", type);
            break;
        case 0:
            // Keep the PC parser contract strict. The apparent Bloodborne PACKET0 stream was
            // `00000000` followed by a perfectly framed PACKET3 train: a mixed/reused command
            // buffer, not an intentional register write. Accepting it silently shifted parsing and
            // polluted shader/depth/raster state for every later draw.
#ifdef __ANDROID__
            __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                                "[EXECUTOR_PM4_TYPE0_REJECTED] base=0x%x words=%u "
                                "remainingDw=%zu next=%08x",
                                header->type0.base.Value(), header->type0.NumWords(), dcb.size(),
                                dcb.size() > 1 ? dcb[1] : 0u);
#endif
            state->abort = true;
            LOG_ERROR(Lib_GnmDriver, "Unimplemented PM4 type 0, base reg: {}, size: {}",
                      header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
#ifdef __ANDROID__
            if (kLiveChecks) {
                ++exec_live_summary.packets;
                exec_live_summary.last_opcode = static_cast<u32>(opcode);
                exec_live_summary.last_off_dw =
                    static_cast<size_t>(reinterpret_cast<const u32*>(header) -
                                        reinterpret_cast<const u32*>(base_addr));
                switch (opcode) {
                case PM4ItOpcode::SetContextReg:
                    ++exec_live_summary.set_context;
                    break;
                case PM4ItOpcode::SetShReg:
                    ++exec_live_summary.set_sh;
                    break;
                case PM4ItOpcode::SetUconfigReg:
                    ++exec_live_summary.set_uconfig;
                    break;
                case PM4ItOpcode::DmaData:
                    ++exec_live_summary.dma;
                    break;
                case PM4ItOpcode::WriteData:
                    ++exec_live_summary.writes;
                    break;
                case PM4ItOpcode::WaitRegMem:
                case PM4ItOpcode::WaitOnCeCounter:
                    ++exec_live_summary.waits;
                    break;
                case PM4ItOpcode::IndirectBuffer:
                case PM4ItOpcode::IndirectBufferConst:
                    ++exec_live_summary.indirect;
                    break;
                case PM4ItOpcode::EventWrite:
                case PM4ItOpcode::EventWriteEop:
                case PM4ItOpcode::EventWriteEos:
                    ++exec_live_summary.events;
                    break;
                case PM4ItOpcode::DispatchDirect:
                case PM4ItOpcode::DispatchIndirect:
                    ++exec_live_summary.dispatches;
                    break;
                default:
                    break;
                }
            }
            if (kTracePm4) {
                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                    "[EXECUTOR_PM4_TRACE] op=0x%02x count=%u off=%zu",
                                    static_cast<unsigned>(opcode), count,
                                    static_cast<size_t>(reinterpret_cast<const u32*>(header) -
                                                        reinterpret_cast<const u32*>(base_addr)));
            }
#endif
#ifdef __ANDROID__
            if (RenderWaveTrace::Enabled()) {
                const bool is_draw = [opcode] {
                    switch (opcode) {
                    case PM4ItOpcode::DrawIndex2:
                    case PM4ItOpcode::DrawIndexOffset2:
                    case PM4ItOpcode::DrawIndexAuto:
                    case PM4ItOpcode::DrawIndirect:
                    case PM4ItOpcode::DrawIndirectMulti:
                    case PM4ItOpcode::DrawIndexIndirect:
                    case PM4ItOpcode::DrawIndexIndirectMulti:
                    case PM4ItOpcode::DrawIndexIndirectCountMulti:
                        return true;
                    default:
                        return false;
                    }
                }();
                if (is_draw) {
                    RenderWaveTrace::Pm4Draw(
                        regs, state->trace_submit_sequence, state->trace_root_dcb,
                        state->trace_root_hash, logical_dcb_base, ib_depth,
                        static_cast<u32>(opcode), ++state->trace_semantic_sequence);
                }
            }
#endif
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
                if (nop->header.count.Value() == 0) {
                    break;
                }

                switch (nop->data_block[0]) {
                case PM4CmdNop::PayloadType::PatchedFlip: {
                    // There is no evidence that GPU CP drives flip events by parsing
                    // special NOP packets. For convenience lets assume that it does.
#ifdef __ANDROID__
                    // Dynamic late-frame oracle. The marker is consumed at this guest frame end;
                    // recording therefore starts after all draws in the current frame and ends at
                    // the next PatchedFlip. No title, shader hash, host clock or swapchain cadence
                    // participates in the selection.
                    if (RenderWaveTrace::ConsumeLateFrameArmMarker()) {
                        const u64 epoch = RenderWaveTrace::ArmLateFrameCapture();
                        Vulkan::ExecutorVkJournalArmFrameCapture(epoch);
                    } else if (RenderWaveTrace::LateFrameCaptureActive()) {
                        const u64 epoch = RenderWaveTrace::LateFrameCaptureEpoch();
                        Vulkan::ExecutorVkJournalFinishFrameCapture(epoch);
                        RenderWaveTrace::FinishLateFrameCapture();
                    }
                    static std::atomic<u32> patched_flip_log_count{0};
                    const u32 patched_flip_log =
                        patched_flip_log_count.fetch_add(1, std::memory_order_relaxed);
                    if (patched_flip_log < 8) {
                        __android_log_print(
                            ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_PATCHED_FLIP_PARSED] ordinal=%u dcb=0x%llx offDw=%zu count=%u",
                            patched_flip_log + 1,
                            static_cast<unsigned long long>(base_addr),
                            static_cast<size_t>(reinterpret_cast<const u32*>(header) -
                                                reinterpret_cast<const u32*>(base_addr)),
                            nop->header.count.Value());
                    }
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_NOP] patched_flip before_signal count=%u",
                                            nop->header.count.Value());
                    }
#endif
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_NOP] patched_flip after_signal");
                    }
#endif
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        rasterizer->ScopeMarkerBegin(label, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
                    if (guest_markers_enabled) {
                        const auto marker_sz = nop->header.count.Value() * 2;
                        const std::string_view label{
                            reinterpret_cast<const char*>(&nop->data_block[1]), marker_sz};
                        const u32 color = *reinterpret_cast<const u32*>(
                            reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                        rasterizer->ScopedMarkerInsertColor(label, color, true);
                    }
                    break;
                }
                case PM4CmdNop::PayloadType::DebugMarkerPop: {
                    if (guest_markers_enabled) {
                        rasterizer->ScopeMarkerEnd(true);
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                NotifyGraphicsStateChanged();
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                if (ExecutorSafeRegArrayWrite(regs.reg_array.data(), Regs::NumRegs, reg_addr,
                                              payload, count - 1, "config")) {
                    NotifyGraphicsStateChanged();
                }
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                if (ExecutorSafeRegArrayWrite(regs.reg_array.data(), Regs::NumRegs, reg_addr,
                                              payload, count - 1, "context")) {
                    NotifyGraphicsStateChanged();
                }

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    if (ExecutorSafeRegArrayWrite(
                            regs.reg_array.data(), Regs::NumRegs,
                            Regs::ShRegWordOffset + set_data->reg_offset, header + 2, count - 1,
                            "sh")) {
                        NotifyGraphicsStateChanged();
                    }
                }
#ifdef __ANDROID__
                if (RenderWaveTrace::Enabled()) {
                    struct StageRegister {
                        u32 offset;
                        RenderWaveTrace::Stage stage;
                    };
                    static constexpr std::array stage_registers{
                        StageRegister{0x08u, RenderWaveTrace::Stage::Ps},
                        StageRegister{0x48u, RenderWaveTrace::Stage::Vs},
                        StageRegister{0x88u, RenderWaveTrace::Stage::Gs},
                        StageRegister{0xc8u, RenderWaveTrace::Stage::Es},
                        StageRegister{0x108u, RenderWaveTrace::Stage::Hs},
                        StageRegister{0x148u, RenderWaveTrace::Stage::Ls},
                    };
                    const u32 payload_words = count > 1 ? count - 1 : 0;
                    const u64 write_begin = set_data->reg_offset;
                    const u64 write_end = write_begin + payload_words;
                    for (const auto& stage_register : stage_registers) {
                        if (stage_register.offset < write_begin || stage_register.offset >= write_end) {
                            continue;
                        }
                        const auto& program =
                            RenderWaveTrace::ProgramForStage(regs, stage_register.stage);
                        const auto shader = RenderWaveTrace::ResolveShader(program);
                        RenderWaveTrace::Pm4Shader(
                            stage_register.stage, shader, state->trace_submit_sequence,
                            state->trace_root_dcb, state->trace_root_hash, logical_dcb_base,
                            ib_depth, ++state->trace_semantic_sequence);
                    }
                }
#endif
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                if (ExecutorSafeRegArrayWrite(regs.reg_array.data(), Regs::NumRegs,
                                              Regs::UconfigRegWordOffset + set_data->reg_offset,
                                              header + 2, count - 1, "uconfig")) {
                    NotifyGraphicsStateChanged();
                }
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                #ifdef __ANDROID__
                if (kLiveChecks) {
                    ++exec_live_summary.draws;
                    if (exec_live_summary.draws <= 32) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_DRAW] n=%u op=DrawIndex2 off=%zu "
                                            "idx=%u max=%u baseLo=0x%x baseHi=0x%x rasterizer=%u",
                                            exec_live_summary.draws, exec_live_summary.last_off_dw,
                                            draw_index->index_count, draw_index->max_size,
                                            draw_index->index_base_lo, draw_index->index_base_hi,
                                            rasterizer ? 1u : 0u);
                    }
                }
                #endif
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndex2", cmd_address));
                        rasterizer->Draw(true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                #ifdef __ANDROID__
                if (kLiveChecks) {
                    ++exec_live_summary.draws;
                    if (exec_live_summary.draws <= 32) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_DRAW] n=%u op=DrawIndexOffset2 "
                                            "off=%zu idx=%u idxOff=%u max=%u rasterizer=%u",
                                            exec_live_summary.draws, exec_live_summary.last_off_dw,
                                            draw_index_off->index_count,
                                            draw_index_off->index_offset, draw_index_off->max_size,
                                            rasterizer ? 1u : 0u);
                    }
                }
                #endif
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexOffset2", cmd_address));
                        rasterizer->Draw(true, draw_index_off->index_offset);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true, draw_index_off->index_offset);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                #ifdef __ANDROID__
                if (kLiveChecks) {
                    ++exec_live_summary.draws;
                    if (exec_live_summary.draws <= 32) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_DRAW] n=%u op=DrawIndexAuto off=%zu "
                                            "idx=%u rasterizer=%u",
                                            exec_live_summary.draws, exec_live_summary.last_off_dw,
                                            draw_index->index_count, rasterizer ? 1u : 0u);
                    }
                }
                #endif
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexAuto", cmd_address));
                        rasterizer->Draw(false);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(false);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                #ifdef __ANDROID__
                if (kLiveChecks) {
                    ++exec_live_summary.draws;
                    if (exec_live_summary.draws <= 32) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_DRAW] n=%u op=DrawIndirect off=%zu "
                                            "dataOff=%u rasterizer=%u",
                                            exec_live_summary.draws, exec_live_summary.last_off_dw,
                                            draw_indirect->data_offset, rasterizer ? 1u : 0u);
                    }
                }
                #endif
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirect", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                #ifdef __ANDROID__
                if (kLiveChecks) {
                    ++exec_live_summary.draws;
                    if (exec_live_summary.draws <= 32) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_DRAW] n=%u op=DrawIndirectMulti "
                                            "off=%zu dataOff=%u count=%u rasterizer=%u",
                                            exec_live_summary.draws, exec_live_summary.last_off_dw,
                                            draw_indirect->data_offset, draw_indirect->count,
                                            rasterizer ? 1u : 0u);
                    }
                }
                #endif
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                #ifdef __ANDROID__
                if (kLiveChecks) {
                    ++exec_live_summary.draws;
                    if (exec_live_summary.draws <= 32) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_DRAW] n=%u op=DrawIndexIndirect "
                                            "off=%zu dataOff=%u rasterizer=%u",
                                            exec_live_summary.draws, exec_live_summary.last_off_dw,
                                            draw_index_indirect->data_offset, rasterizer ? 1u : 0u);
                    }
                }
                #endif
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirect", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                #ifdef __ANDROID__
                if (kLiveChecks) {
                    ++exec_live_summary.draws;
                    if (exec_live_summary.draws <= 32) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_DRAW] n=%u op=DrawIndexIndirectMulti "
                                            "off=%zu dataOff=%u count=%u rasterizer=%u",
                                            exec_live_summary.draws, exec_live_summary.last_off_dw,
                                            draw_index_indirect->data_offset,
                                            draw_index_indirect->count, rasterizer ? 1u : 0u);
                    }
                }
                #endif
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                #ifdef __ANDROID__
                if (kLiveChecks) {
                    ++exec_live_summary.draws;
                    if (exec_live_summary.draws <= 32) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_DRAW] n=%u "
                                            "op=DrawIndexIndirectCountMulti off=%zu dataOff=%u "
                                            "countAddr=0x%llx rasterizer=%u",
                                            exec_live_summary.draws, exec_live_summary.last_off_dw,
                                            draw_index_indirect->data_offset,
                                            (unsigned long long)draw_index_indirect->count_addr,
                                            rasterizer ? 1u : 0u);
                    }
                }
                #endif
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectCountMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchDirect", cmd_address));
                        rasterizer->DispatchDirect();
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchDirect();
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchIndirect", cmd_address));
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                    }
                }
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                } else if (event->event_index.Value() == EventIndex::ZpassDone) {
                    if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                        static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                        static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                        u64* const results_base = event->Address<u64*>();
                        u64* results = results_base;
                        for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                            *results = pixel_counter | OcclusionCounterValidMask;
                        }
                        if (num_counter_pairs != 0) {
                            const u64 result_span =
                                (u64{num_counter_pairs - 1} * 2 + 1) * sizeof(u64);
                            InvalidateIndirectSnapshotsForWrite(
                                state, reinterpret_cast<VAddr>(results_base), result_span);
                        }
                        pixel_counter += OcclusionCounterStep;
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEos: {
                const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                if (rasterizer) {
                    rasterizer->ProcessDownloadImages();
                }
                if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
                    ASSERT(event_eos->size == 1);
                    if (rasterizer) {
                        rasterizer->Finish();
                        const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                        auto* const destination = event_eos->Address();
                        *destination = value;
                        InvalidateIndirectSnapshotsForWrite(
                            state, reinterpret_cast<VAddr>(destination), sizeof(value));
                    }
                } else {
                    const bool defer_completion =
#ifdef __ANDROID__
                        rasterizer && ExecutorBoundedRasterizerPipeline();
#else
                        false;
#endif
                    if (defer_completion) {
                        // Pm4SubmissionState belongs to the GPU coroutine and is not thread-safe.
                        // Invalidate its CPU-side snapshot metadata conservatively here; the
                        // retirement worker below performs only the guest-memory publication.
                        InvalidateIndirectSnapshotsForWrite(
                            state, reinterpret_cast<VAddr>(event_eos->Address()), sizeof(u32));
                    }
#ifdef __ANDROID__
                    const u64 completion_tick =
                        defer_completion ? ArmGpuCompletionTickForExecutor() : 0;
#else
                    constexpr u64 completion_tick = 0;
#endif
                    auto signal_eos = [this, state, packet = *event_eos,
                                       invalidate_snapshot = !defer_completion,
                                       completion_tick] {
                        packet.SignalFence(
                            [this, state, invalidate_snapshot,
                             completion_tick](void* address, u64 data, u32 num_bytes) {
                                if (ExecutorWriteFenceToCpuAddress("eos", address, data,
                                                                   num_bytes)) {
                                    if (invalidate_snapshot) {
                                        InvalidateIndirectSnapshotsForWrite(
                                            state, reinterpret_cast<VAddr>(address), num_bytes);
                                    }
#ifdef __ANDROID__
                                    CompletePendingCompletionWriteForExecutor(
                                        reinterpret_cast<VAddr>(address), num_bytes,
                                        completion_tick);
#endif
                                }
                            });
                    };
#ifdef __ANDROID__
                    if (defer_completion) {
                        if (completion_tick != 0) {
                            RecordCompletionWriteForExecutor(
                                state, Pm4Engine::Graphics,
                                reinterpret_cast<VAddr>(event_eos->Address()),
                                event_eos->DataDWord(), sizeof(u32), completion_tick);
                            rasterizer->GetScheduler().DeferPriorityOperationAt(
                                completion_tick, std::move(signal_eos));
                        } else {
                            signal_eos();
                        }
                    } else
#endif
                    {
                        signal_eos();
                    }
                }
                break;
            }
            case PM4ItOpcode::EventWriteEop: {
                const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
                if (rasterizer) {
                    rasterizer->ProcessDownloadImages();
                }
#ifdef __ANDROID__
                const u32 eop_offset_dw = static_cast<u32>(
                    reinterpret_cast<const u32*>(header) - reinterpret_cast<const u32*>(base_addr));
                const u32 eop_int_sel = static_cast<u32>(event_eop->int_sel.Value());
                const u32 eop_data_sel = static_cast<u32>(event_eop->data_sel.Value());
                const u64 eop_address = reinterpret_cast<uintptr_t>(event_eop->Address<u32>());
                ExecutorEopTraceParserEop(base_addr, eop_trace_dcb_dw, eop_offset_dw, eop_int_sel,
                                          eop_data_sel, eop_address);
                const bool defer_completion =
                    rasterizer && ExecutorBoundedRasterizerPipeline();
                const u32 eop_write_size = [&] {
                    switch (event_eop->data_sel.Value()) {
                    case DataSelect::None:
                        return 0u;
                    case DataSelect::Data32Low:
                        return static_cast<u32>(sizeof(u32));
                    case DataSelect::Data64:
                    case DataSelect::GpuClock64:
                    case DataSelect::PerfCounter:
                        return static_cast<u32>(sizeof(u64));
                    default:
                        return 0u;
                    }
                }();
                if (defer_completion && eop_write_size != 0) {
                    // Keep Pm4SubmissionState mutations on the GPU coroutine. The actual fence
                    // bytes and IRQ remain deferred to the exact Vulkan completion tick below.
                    InvalidateIndirectSnapshotsForWrite(
                        state, reinterpret_cast<VAddr>(event_eop->Address<u32>()),
                        eop_write_size);
                }
                const u64 completion_tick =
                    defer_completion ? ArmGpuCompletionTickForExecutor() : 0;
                auto signal_eop =
                    [this, state, packet = *event_eop,
                     eop_dcb = reinterpret_cast<uintptr_t>(base_addr), eop_trace_dcb_dw,
                     eop_offset_dw, eop_int_sel, eop_data_sel, eop_address,
                     invalidate_snapshot = !defer_completion, completion_tick] {
                        packet.SignalFence(
                            [this, state, invalidate_snapshot,
                             completion_tick](void* address, u64 data, u32 num_bytes) {
                                if (ExecutorWriteFenceToCpuAddress("eop", address, data,
                                                                   num_bytes)) {
                                    if (invalidate_snapshot) {
                                        InvalidateIndirectSnapshotsForWrite(
                                            state, reinterpret_cast<VAddr>(address), num_bytes);
                                    }
                                    CompletePendingCompletionWriteForExecutor(
                                        reinterpret_cast<VAddr>(address), num_bytes,
                                        completion_tick);
                                }
                            },
                            [eop_dcb, eop_trace_dcb_dw, eop_offset_dw, eop_int_sel,
                             eop_data_sel, eop_address] {
                                ExecutorEopTraceIrqSignal(
                                    eop_dcb, eop_trace_dcb_dw, eop_offset_dw, eop_int_sel,
                                    eop_data_sel, eop_address);
                                Platform::IrqC::Instance()->Signal(
                                    Platform::InterruptId::GfxEop);
                            });
                    };
                if (defer_completion) {
                    if (completion_tick != 0) {
                        if (event_eop->data_sel.Value() == DataSelect::Data32Low) {
                            RecordCompletionWriteForExecutor(
                                state, Pm4Engine::Graphics,
                                reinterpret_cast<VAddr>(event_eop->Address<u32>()),
                                event_eop->DataDWord(), sizeof(u32), completion_tick);
                        } else if (event_eop->data_sel.Value() == DataSelect::Data64) {
                            RecordCompletionWriteForExecutor(
                                state, Pm4Engine::Graphics,
                                reinterpret_cast<VAddr>(event_eop->Address<u32>()),
                                event_eop->DataQWord(), sizeof(u64), completion_tick);
                        }
                        rasterizer->GetScheduler().DeferPriorityOperationAt(
                            completion_tick, std::move(signal_eop));
                    } else {
                        signal_eop();
                    }
                } else {
                    signal_eop();
                }
#else
                event_eop->SignalFence(
                    [this, state](void* address, u64 data, u32 num_bytes) {
                        if (ExecutorWriteFenceToCpuAddress("eop", address, data, num_bytes)) {
                            InvalidateIndirectSnapshotsForWrite(
                                state, reinterpret_cast<VAddr>(address), num_bytes);
                        }
                    },
                    [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
#endif
                break;
            }
            case PM4ItOpcode::DmaData: {
                const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
#ifdef __ANDROID__
                const bool cpu_mirrored = ExecutorMirrorDmaDataToCpuBacking(*dma_data, "gfx");
                const char* dma_decision = "exec";
                if (dma_data->dst_addr_lo == 0x3022C) {
                    dma_decision = "skip-magic-dst";
                    ExecutorLogMagicDmaPayload(*dma_data, "gfx", exec_live_summary.last_off_dw);
                } else if (!rasterizer) {
                    dma_decision = "skip-no-rasterizer";
                }
                ExecutorLogLiveDmaData(*dma_data, "gfx", exec_live_summary.last_off_dw,
                                       cpu_mirrored, dma_decision);
#endif
                if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                    break;
                }
                if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                           dma_data->data, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           dma_data->dst_sel == DmaDataDst::Gds) {
                    rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                           dma_data->NumBytes(), true, false);
                } else if (dma_data->src_sel == DmaDataSrc::Data &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                           dma_data->data, false);
                } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                           dma_data->NumBytes(), false, true);
                } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                            dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                           (dma_data->dst_sel == DmaDataDst::Memory ||
                            dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                    rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                           dma_data->SrcAddress<VAddr>(), dma_data->NumBytes(),
                                           false, false);
                } else {
                    UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                    u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
                }
                break;
            }
            case PM4ItOpcode::WriteData: {
                const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
                ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
                const u32 data_size = (header->type3.count.Value() - 2) * 4;
                void* address = write_data->Address<void*>();
                if (!write_data->wr_one_addr.Value()) {
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(address);
                        const uintptr_t canonical_addr =
                            Libraries::VideoOut::VideoOutPort::CanonicalLabelAddress(raw_addr);
                        const uintptr_t label_start =
                            vo_port ? Libraries::VideoOut::VideoOutPort::CanonicalLabelAddress(
                                          reinterpret_cast<uintptr_t>(vo_port->buffer_labels.data()))
                                    : 0;
                        const uintptr_t label_end =
                            vo_port ? label_start + sizeof(vo_port->buffer_labels) : 0;
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] raw=0x%llx canonical=0x%llx size=%u dst=%u labelStart=0x%llx labelEnd=0x%llx isVo=%u data0=0x%08x",
                                            static_cast<unsigned long long>(raw_addr),
                                            static_cast<unsigned long long>(canonical_addr),
                                            data_size, write_data->dst_sel.Value(),
                                            static_cast<unsigned long long>(label_start),
                                            static_cast<unsigned long long>(label_end),
                                            vo_port && vo_port->IsVoLabel(static_cast<const u64*>(address)) ? 1u : 0u,
                                            data_size ? write_data->data[0] : 0u);
                    }
#endif
                    bool is_vo_label =
                        vo_port && vo_port->IsVoLabel(static_cast<const u64*>(address));
#ifdef __ANDROID__
                    if (!is_vo_label && LooksLikeAndroidHostPointer(address)) {
                        if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                                "[EXECUTOR_PM4_WRITEDATA] skip_unowned_host_write_pre_alias raw=0x%llx size=%u data0=0x%08x",
                                                static_cast<unsigned long long>(
                                                    reinterpret_cast<uintptr_t>(address)),
                                                data_size,
                                                data_size ? write_data->data[0] : 0u);
                        }
                        break;
                    }
#endif
                    bool wrote_alias = false;
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] before_alias");
                    }
#endif
                    wrote_alias = Libraries::VideoOut::ExecutorTryWriteVoLabelAlias(
                        address, write_data->data, data_size);
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] after_alias hit=%u",
                                            wrote_alias ? 1u : 0u);
                    }
#endif
                    if (wrote_alias) {
                        break;
                    }
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] before_port_label_check");
                    }
#endif
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] after_port_label_check hit=%u",
                                            is_vo_label ? 1u : 0u);
                    }
#endif
                    if (is_vo_label) {
#ifdef __ANDROID__
                        if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                                "[EXECUTOR_PM4_WRITEDATA] before_vo_label_write");
                        }
#endif
                        if (vo_port->TryWriteVoLabel(address, write_data->data, data_size)) {
#ifdef __ANDROID__
                            if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                                __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                                    "[EXECUTOR_PM4_WRITEDATA] after_vo_label_write");
                            }
#endif
                            break;
                        }
                    }
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] before_hostptr_filter");
                    }
                    if (LooksLikeAndroidHostPointer(address)) {
                        if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                                "[EXECUTOR_PM4_WRITEDATA] skip_unowned_host_write raw=0x%llx size=%u data0=0x%08x",
                                                static_cast<unsigned long long>(
                                                    reinterpret_cast<uintptr_t>(address)),
                                                data_size,
                                                data_size ? write_data->data[0] : 0u);
                        }
                        break;
                    }
#endif
                    auto* memory = Core::Memory::Instance();
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] before_try_backing");
                    }
#endif
                    const bool wrote_backing =
                        memory && memory->TryWriteBacking(address, write_data->data, data_size);
                    if (wrote_backing) {
                        InvalidateIndirectSnapshotsForWrite(
                            state, reinterpret_cast<VAddr>(address), data_size);
                    }
#ifdef __ANDROID__
                    if (ExecutorLiveGnmChecks()) {
                        static u32 logged_write_data = 0;
                        if (logged_write_data++ < 64) {
                            __android_log_print(
                                ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_PM4_WRITEDATA_WRITE] phase=gfx result=%s raw=0x%llx "
                                "size=%u dstSel=%u data0=0x%08x",
                                wrote_backing ? "memory_backing" : "miss",
                                static_cast<unsigned long long>(
                                    reinterpret_cast<uintptr_t>(address)),
                                data_size, write_data->dst_sel.Value(),
                                data_size ? write_data->data[0] : 0u);
                        }
                    }
#endif
                    if (!wrote_backing) {
#ifdef __ANDROID__
                        if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                            __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                                "[EXECUTOR_PM4_WRITEDATA] skip_unowned_host_write raw=0x%llx size=%u data0=0x%08x",
                                                static_cast<unsigned long long>(
                                                    reinterpret_cast<uintptr_t>(address)),
                                                data_size,
                                                data_size ? write_data->data[0] : 0u);
                        }
#else
                        std::memcpy(address, write_data->data, data_size);
                        InvalidateIndirectSnapshotsForWrite(
                            state, reinterpret_cast<VAddr>(address), data_size);
#endif
                    }
                } else {
                    UNREACHABLE();
                }
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                const VAddr semaphore_address = mem_semaphore->Address<VAddr>();
                if (mem_semaphore->IsSignaling()) {
                    mem_semaphore->Signal();
                    InvalidateIndirectSnapshotsForWrite(state, semaphore_address, sizeof(u64));
                } else {
                    while (!mem_semaphore->Signaled() && !state->abort) {
                        YIELD_GFX();
                    }
                    if (!state->abort) {
                        mem_semaphore->Decrement();
                        InvalidateIndirectSnapshotsForWrite(state, semaphore_address, sizeof(u64));
                    }
                }
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                while (!rewind->Valid() && !state->abort) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
#ifdef __ANDROID__
                if (kLiveChecks && exec_live_summary.waits <= 32) {
                    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                        "[EXECUTOR_LIVE_PM4_WAIT] off=%zu memSpace=%u func=%u "
                                        "addr=0x%llx ref=0x%x mask=0x%x",
                                        exec_live_summary.last_off_dw,
                                        (unsigned)wait_reg_mem->mem_space.Value(),
                                        (unsigned)wait_reg_mem->function.Value(),
                                        (unsigned long long)reinterpret_cast<uintptr_t>(wait_addr),
                                        wait_reg_mem->ref, wait_reg_mem->mask);
                }
#endif
                u32 wait_value{};
#ifdef __ANDROID__
                const char* wait_source =
                    ExecutorReadWaitRegMemValue(*wait_reg_mem, regs.reg_array, &wait_value);
                if (kLiveChecks && exec_live_summary.waits <= 32) {
                    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                        "[EXECUTOR_LIVE_PM4_WAIT_VALUE] off=%zu src=%s "
                                        "value=0x%x pass=%u",
                                        exec_live_summary.last_off_dw, wait_source, wait_value,
                                        TestWaitRegMemValue(*wait_reg_mem, wait_value) ? 1u : 0u);
                }
#else
                const char* wait_source = "";
#endif
#ifdef __ANDROID__
                if (!TestWaitRegMemValue(*wait_reg_mem, wait_value)) {
                    const u64 dependency_tick = ConsumeCompletionWaitForExecutor(
                        state, Pm4Engine::Graphics, *wait_reg_mem);
                    if (dependency_tick != 0) {
                        rasterizer->CompletionWaitBarrier();
                        ++executor_logical_wait_count;
                        break;
                    }
                }
                if (!TestWaitRegMemValue(*wait_reg_mem, wait_value) &&
                    SynchronizeUntrackedMemoryWaitForExecutor(*wait_reg_mem, wait_value)) {
                    ++executor_logical_wait_count;
                    break;
                }
                const bool completion_wait_stall =
                    executor_completion_flush_pending &&
                    !TestWaitRegMemValue(*wait_reg_mem, wait_value);
                const auto completion_wait_start =
                    completion_wait_stall ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
                bool completion_wait_recorded = false;
                const auto record_completion_wait = [&] {
                    if (!completion_wait_stall || completion_wait_recorded) {
                        return;
                    }
                    completion_wait_recorded = true;
                    const u64 elapsed_ns = static_cast<u64>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - completion_wait_start)
                            .count());
                    ++executor_completion_wait_stalls;
                    executor_completion_wait_stall_ns += elapsed_ns;
                    executor_completion_wait_stall_max_ns =
                        std::max(executor_completion_wait_stall_max_ns, elapsed_ns);
                };
                if (executor_completion_flush_pending &&
                    !TestWaitRegMemValue(*wait_reg_mem, wait_value)) {
                    FlushGpuCompletionBatchForExecutor(false, true);
                }
#endif
                if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory &&
                    std::strcmp(wait_source, "vo_label_alias") == 0) {
                    // Match the PC scheduler contract: a host wait is safe only when every pending
                    // GPU submit is on the graphics queue. Otherwise blocking this one host thread
                    // also prevents the compute queue from reaching work that may precede the flip.
                    bool can_park_gpu_thread = false;
                    if (vo_port != nullptr) {
                        auto& gfx_queue = mapped_queues[GfxQueueId];
                        can_park_gpu_thread =
                            num_submits.load(std::memory_order_acquire) == gfx_queue.submits.size();
                    }
                    if (can_park_gpu_thread) {
                        vo_port->WaitVoLabel([&] {
                            return vo_port->TryReadVoLabelLocked(
                                       wait_addr, &wait_value, sizeof(wait_value)) &&
                                   TestWaitRegMemValue(*wait_reg_mem, wait_value);
                        });
                        record_completion_wait();
                        break;
                    }
                    u32 local_yields = 0;
                    while ((!Libraries::VideoOut::ExecutorTryReadVoLabelAlias(wait_addr,
                                                                              &wait_value) ||
                            !TestWaitRegMemValue(*wait_reg_mem, wait_value)) &&
                           !state->abort) {
#ifdef __ANDROID__
                        ++exec_live_summary.wait_yields;
                        ++local_yields;
                        if (kLiveChecks && ExecutorShouldLogWaitYield(local_yields)) {
                            __android_log_print(
                                ANDROID_LOG_INFO, "LSX4Native",
                                "[EXECUTOR_LIVE_PM4_WAIT_YIELD] kind=vo_alias "
                                "off=%zu yields=%u value=0x%x ref=0x%x mask=0x%x",
                                exec_live_summary.last_off_dw, local_yields, wait_value,
                                wait_reg_mem->ref, wait_reg_mem->mask);
                        }
#endif
                        YIELD_GFX();
                    }
                    record_completion_wait();
                    break;
                }
                if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory &&
                    std::strcmp(wait_source, "memory_backing") == 0) {
                    u32 local_yields = 0;
                    while (!state->abort &&
                           !TestWaitRegMemValue(*wait_reg_mem, wait_value)) {
#ifdef __ANDROID__
                        ++exec_live_summary.wait_yields;
                        ++local_yields;
                        if (kLiveChecks && ExecutorShouldLogWaitYield(local_yields)) {
                            __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                                "[EXECUTOR_LIVE_PM4_WAIT_YIELD] kind=memory_backing "
                                                "off=%zu yields=%u value=0x%x ref=0x%x mask=0x%x",
                                                exec_live_summary.last_off_dw, local_yields,
                                                wait_value, wait_reg_mem->ref,
                                                wait_reg_mem->mask);
                        }
#endif
                        YIELD_GFX();
                        wait_value = ExecutorReadMappedWaitValueFast(wait_addr);
                    }
                    record_completion_wait();
                    break;
                }
                if (vo_port && vo_port->IsVoLabel(wait_addr)) {
                    bool can_park_gpu_thread = false;
                    {
                        auto& gfx_queue = mapped_queues[GfxQueueId];
                        can_park_gpu_thread =
                            num_submits.load(std::memory_order_acquire) == gfx_queue.submits.size();
                    }
                    if (can_park_gpu_thread) {
                        vo_port->WaitVoLabel([&] {
                            u32 v = 0;
                            return vo_port->TryReadVoLabelLocked(wait_addr, &v, sizeof(v)) &&
                                   TestWaitRegMemValue(*wait_reg_mem, v);
                        });
                        record_completion_wait();
                        break;
                    }
                }
                u32 local_yields = 0;
                while (!wait_reg_mem->Test(regs.reg_array) && !state->abort) {
#ifdef __ANDROID__
                    ++exec_live_summary.wait_yields;
                    ++local_yields;
                    if (kLiveChecks && ExecutorShouldLogWaitYield(local_yields)) {
                        u32 current_value{};
                        const char* current_source =
                            ExecutorReadWaitRegMemValue(*wait_reg_mem, regs.reg_array,
                                                        &current_value);
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_WAIT_YIELD] kind=generic off=%zu "
                                            "yields=%u src=%s value=0x%x ref=0x%x mask=0x%x",
                                            exec_live_summary.last_off_dw, local_yields,
                                            current_source, current_value, wait_reg_mem->ref,
                                            wait_reg_mem->mask);
                    }
#endif
                    YIELD_GFX();
                }
                record_completion_wait();
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                if (packet_words != sizeof(PM4CmdIndirectBuffer) / sizeof(u32)) {
                    state->abort = true;
                    LOG_ERROR(Lib_GnmDriver, "Malformed graphics indirect-buffer packet size {}",
                              packet_words);
                    break;
                }
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                const VAddr address =
                    reinterpret_cast<VAddr>(indirect_buffer->Address<const u32>());
                const u32 num_words = indirect_buffer->ib_size.Value();
                const bool chain = indirect_buffer->chain.Value() != 0;
                std::vector<u32> child;
                if (!CopyIndirectAtEncounter(Pm4Engine::Graphics, address, num_words, child, state)) {
                    break;
                }
                if (chain) {
                    if (++chain_hops > 4096) {
                        state->abort = true;
                        break;
                    }
                    owned_dcb = std::move(child);
                    dcb = std::span<const u32>{owned_dcb};
                    base_addr = reinterpret_cast<uintptr_t>(dcb.data());
                    logical_dcb_base = address;
#ifdef __ANDROID__
                    eop_trace_dcb_dw = static_cast<u32>(dcb.size());
#endif
                    continue;
                }
                auto task = ProcessGraphics({}, {}, std::move(child), {}, state, ib_depth + 1,
                                            &ce_task, address);
                SCOPE_EXIT {
                    if (task.handle) {
                        task.handle.destroy();
                    }
                };
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                if (!ce_task.handle) {
                    state->abort = true;
                    LOG_ERROR(Lib_GnmDriver,
                              "WAIT_ON_CE_COUNTER encountered without a root constant-engine task");
                    break;
                }
                u32 local_yields = 0;
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done() &&
                       !state->abort) {
#ifdef __ANDROID__
                    ++exec_live_summary.wait_yields;
                    ++local_yields;
                    if (kLiveChecks &&
                        (local_yields == 1 || local_yields == 64 || local_yields == 4096)) {
                        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                                            "[EXECUTOR_LIVE_PM4_WAIT_YIELD] kind=ce off=%zu "
                                            "yields=%u ce=%u de=%u",
                                            exec_live_summary.last_off_dw, local_yields,
                                            cblock.ce_count, cblock.de_count);
                    }
#endif
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                if (rasterizer) {
                    rasterizer->CpSync();
                }
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                state->abort = true;
                LOG_ERROR(Lib_GnmDriver,
                          "Unknown graphics PM4 type 3 opcode {:#x} with count {}",
                          static_cast<u32>(opcode), count);
                break;
            }
            dcb = NextPacket(dcb, packet_words);
            break;
        }
    }

    if (owns_ce_context && ce_task.handle && !ce_task.handle.done()) {
        while (!ce_task.handle.done() && !state->abort) {
            RESUME_GFX(ce_task);
        }
    }

#ifdef __ANDROID__
    ExecutorEopTraceParserEnd(base_addr, eop_trace_dcb_dw);
    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                            "[EXECUTOR_PM4_TRACE] ProcessGraphics exit");
    }
    if (kLiveChecks) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_LIVE_PM4_END] packets=%u draws=%u dispatches=%u waits=%u waitYields=%u "
            "dma=%u writes=%u setCtx=%u setSh=%u setUcfg=%u indirect=%u events=%u "
            "lastOp=0x%02x lastOff=%zu",
            exec_live_summary.packets, exec_live_summary.draws, exec_live_summary.dispatches,
            exec_live_summary.waits, exec_live_summary.wait_yields, exec_live_summary.dma,
            exec_live_summary.writes, exec_live_summary.set_context, exec_live_summary.set_sh,
            exec_live_summary.set_uconfig, exec_live_summary.indirect, exec_live_summary.events,
            exec_live_summary.last_opcode, exec_live_summary.last_off_dw);
        if (exec_live_summary.draws == 0 && exec_live_summary.dispatches == 0) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_LIVE_NO_DRAW_SUBMIT] packets=%u waits=%u waitYields=%u dma=%u "
                "writes=%u setCtx=%u setSh=%u setUcfg=%u lastOp=0x%02x lastOff=%zu",
                exec_live_summary.packets, exec_live_summary.waits, exec_live_summary.wait_yields,
                exec_live_summary.dma, exec_live_summary.writes, exec_live_summary.set_context,
                exec_live_summary.set_sh, exec_live_summary.set_uconfig,
                exec_live_summary.last_opcode, exec_live_summary.last_off_dw);
        }
    }
#endif
    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid,
                                           std::vector<u32> owned_acb,
                                           Pm4SubmissionStatePtr state, u32 ib_depth,
                                           VAddr logical_acb_base) {
    if (logical_acb_base == 0) {
        logical_acb_base = reinterpret_cast<VAddr>(acb.data());
    }
    if (!owned_acb.empty()) {
        acb = std::span<const u32>{owned_acb};
    }
    if (!state) {
        state = std::make_shared<Pm4SubmissionState>();
    }
    FIBER_ENTER(acb_task_name[vqid]);
    if (ib_depth > 32) {
        state->abort = true;
        FIBER_EXIT;
        co_return;
    }
    u32 chain_hops = 0;
    auto& queue = asc_queues[{vqid}];
    const bool host_markers_enabled = rasterizer && Config::getVkHostMarkersEnabled();
    bool advances_ring = !is_indirect;

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty() && !state->abort) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 packet_total_words = 1;
        u32 next_dw_off = 1;

        // A root ACB may end in the middle of a type-3 packet. Keep appending later DingDong
        // segments until the packet is complete; never reinterpret a type-2 pad as a type-3 length.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            if (!advances_ring || queue.tmp_dwords > queue.tmp_packet.size()) {
                state->abort = true;
                break;
            }
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            if (header->type != 3) {
                state->abort = true;
                break;
            }
            packet_total_words = header->type3.NumWords() + 1;
            if (queue.tmp_dwords >= packet_total_words ||
                packet_total_words > queue.tmp_packet.size()) {
                state->abort = true;
                break;
            }
            next_dw_off = packet_total_words - queue.tmp_dwords;
            if (next_dw_off > acb.size()) {
                if (queue.tmp_dwords + acb.size() > queue.tmp_packet.size()) {
                    state->abort = true;
                    break;
                }
                std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                            acb.size_bytes());
                queue.tmp_dwords += static_cast<u32>(acb.size());
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
                break;
            }
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        } else if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            acb = NextPacket(acb, next_dw_off);
            if (advances_ring) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        } else if (header->type != 3) {
            state->abort = true;
            LOG_ERROR(Lib_GnmDriver, "Invalid compute PM4 type {}", header->type.Value());
            break;
        } else {
            packet_total_words = header->type3.NumWords() + 1;
            next_dw_off = packet_total_words;
            // If the packet is split across the current root-ring segment, preserve all available
            // DWORDs. Later segments may be shorter than the remainder, so the append path above is
            // deliberately multi-segment rather than assuming exactly one wrap.
            if (next_dw_off > acb.size()) [[unlikely]] {
                if (!advances_ring || packet_total_words > queue.tmp_packet.size() ||
                    acb.size() > queue.tmp_packet.size()) {
                    state->abort = true;
                    break;
                }
                std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
                queue.tmp_dwords = static_cast<u32>(acb.size());
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
                break;
            }
        }

        const PM4ItOpcode opcode = header->type3.opcode;

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            if (packet_total_words != sizeof(PM4CmdIndirectBuffer) / sizeof(u32)) {
                state->abort = true;
                LOG_ERROR(Lib_GnmDriver, "Malformed compute indirect-buffer packet size {}",
                          packet_total_words);
                break;
            }
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            const VAddr address =
                reinterpret_cast<VAddr>(indirect_buffer->Address<const u32>());
            const u32 num_words = indirect_buffer->ib_size.Value();
            const bool chain = indirect_buffer->chain.Value() != 0;
            std::vector<u32> child;
            if (!CopyIndirectAtEncounter(Pm4Engine::Compute, address, num_words, child, state)) {
                break;
            }
            if (chain) {
                if (++chain_hops > 4096) {
                    state->abort = true;
                    break;
                }
                if (advances_ring) {
                    *queue.read_addr += next_dw_off;
                    *queue.read_addr %= queue.ring_size_dw;
                }
                owned_acb = std::move(child);
                acb = std::span<const u32>{owned_acb};
                base_addr = reinterpret_cast<VAddr>(acb.data());
                acb_size = acb.size_bytes();
                logical_acb_base = address;
                advances_ring = false;
                indirect_patches.clear();
                continue;
            }
            auto task = ProcessCompute<true>({}, vqid, std::move(child), state, ib_depth + 1,
                                             address);
            SCOPE_EXIT {
                if (task.handle) {
                    task.handle.destroy();
                }
            };
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
#ifdef __ANDROID__
            const bool cpu_mirrored = ExecutorMirrorDmaDataToCpuBacking(*dma_data, "asc");
            const char* dma_decision = "exec";
            if (dma_data->dst_addr_lo == 0x3022C) {
                dma_decision = "skip-magic-dst";
                ExecutorLogMagicDmaPayload(*dma_data, "asc", 0);
            } else if (!rasterizer) {
                dma_decision = "skip-no-rasterizer";
            }
            ExecutorLogLiveDmaData(*dma_data, "asc", 0, cpu_mirrored, dma_decision);
#endif
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                       true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                       dma_data->NumBytes(), true, false);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                       dma_data->data, false);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                       dma_data->NumBytes(), false, true);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                // The task may execute an immutable host-owned snapshot while PM4 still names the
                // original guest ring. Preserve that logical base and translate the DMA target back
                // into the snapshot so self-patched DispatchDirect packets retain PC semantics.
                constexpr u64 patch_bytes =
                    sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                const bool valid_logical_range =
                    logical_acb_base <= std::numeric_limits<VAddr>::max() - acb_size;
                const VAddr logical_acb_end =
                    valid_logical_range ? logical_acb_base + acb_size : 0;
                const bool target_in_acb =
                    valid_logical_range && acb_size >= sizeof(PM4Header) + patch_bytes &&
                    dst_addr >= logical_acb_base + sizeof(PM4Header) &&
                    dst_addr <= logical_acb_end - patch_bytes && num_bytes == patch_bytes;
                const u64 header_offset =
                    target_in_acb ? dst_addr - logical_acb_base - sizeof(PM4Header) : 0;
                const auto* patched_header =
                    target_in_acb && header_offset + sizeof(PM4Header) <= acb_size
                        ? reinterpret_cast<const PM4Header*>(base_addr + header_offset)
                        : nullptr;
                if (patched_header != nullptr && patched_header->type == 3 &&
                    patched_header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(patched_header, src_addr);
                } else {
                    rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid() && !state->abort) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                ExecutorSafeRegArrayWrite(regs.reg_array.data(), Regs::NumRegs,
                                          Regs::ShRegWordOffset + set_data->reg_offset, header + 2,
                                          header->type3.NumWords() - 1, "sh-compute");
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                rasterizer->DispatchIndirect(it->indirect_addr, 0, size);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
                    rasterizer->DispatchDirect();
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchDirect();
                }
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                const auto cmd_address = reinterpret_cast<const void*>(header);
                if (host_markers_enabled) {
                    rasterizer->ScopeMarkerBegin(
                        fmt::format("asc[{}]:{}:DispatchIndirect", vqid, cmd_address));
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                    rasterizer->ScopeMarkerEnd();
                } else {
                    rasterizer->DispatchIndirect(ib_address, 0, size);
                }
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (!write_data->wr_one_addr.Value()) {
                void* address = write_data->Address<void*>();
                if (Libraries::VideoOut::ExecutorTryWriteVoLabelAlias(address, write_data->data,
                                                                       data_size)) {
                    break;
                }
                if (vo_port && vo_port->TryWriteVoLabel(address, write_data->data, data_size)) {
                    break;
                }
#ifdef __ANDROID__
                if (LooksLikeAndroidHostPointer(address)) {
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] skip_unowned_host_write raw=0x%llx size=%u data0=0x%08x",
                                            static_cast<unsigned long long>(
                                                reinterpret_cast<uintptr_t>(address)),
                                            data_size, data_size ? write_data->data[0] : 0u);
                    }
                    break;
                }
#endif
                auto* memory = Core::Memory::Instance();
                const bool wrote_backing =
                    memory && memory->TryWriteBacking(address, write_data->data, data_size);
                if (wrote_backing) {
                    InvalidateIndirectSnapshotsForWrite(
                        state, reinterpret_cast<VAddr>(address), data_size);
                }
#ifdef __ANDROID__
                if (ExecutorLiveGnmChecks()) {
                    static u32 logged_asc_write_data = 0;
                    if (logged_asc_write_data++ < 64) {
                        __android_log_print(
                            ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_LIVE_PM4_WRITEDATA_WRITE] phase=asc result=%s raw=0x%llx "
                            "size=%u dstSel=%u data0=0x%08x",
                            wrote_backing ? "memory_backing" : "miss",
                            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address)),
                            data_size, write_data->dst_sel.Value(),
                            data_size ? write_data->data[0] : 0u);
                    }
                }
#endif
                if (!wrote_backing) {
#ifdef __ANDROID__
                    if (std::getenv("EXECUTOR_TRACE_PM4") != nullptr) {
                        __android_log_print(ANDROID_LOG_INFO, "EXECUTOR",
                                            "[EXECUTOR_PM4_WRITEDATA] skip_unowned_host_write raw=0x%llx size=%u data0=0x%08x",
                                            static_cast<unsigned long long>(
                                                reinterpret_cast<uintptr_t>(address)),
                                            data_size, data_size ? write_data->data[0] : 0u);
                    }
#else
                    std::memcpy(address, write_data->data, data_size);
                    InvalidateIndirectSnapshotsForWrite(
                        state, reinterpret_cast<VAddr>(address), data_size);
#endif
                }
            } else {
                UNREACHABLE();
            }
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            const VAddr semaphore_address = mem_semaphore->Address<VAddr>();
            if (mem_semaphore->IsSignaling()) {
                mem_semaphore->Signal();
                InvalidateIndirectSnapshotsForWrite(state, semaphore_address, sizeof(u64));
            } else {
                while (!mem_semaphore->Signaled() && !state->abort) {
                    YIELD_ASC(vqid);
                }
                if (!state->abort) {
                    mem_semaphore->Decrement();
                    InvalidateIndirectSnapshotsForWrite(state, semaphore_address, sizeof(u64));
                }
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            const u64* wait_addr = wait_reg_mem->Address<u64*>();
            u32 wait_value{};
            if (wait_reg_mem->mem_space.Value() == PM4CmdWaitRegMem::MemSpace::Memory &&
                Libraries::VideoOut::ExecutorTryReadVoLabelAlias(wait_addr, &wait_value)) {
#ifdef __ANDROID__
                const bool completion_wait_stall =
                    executor_completion_flush_pending &&
                    !TestWaitRegMemValue(*wait_reg_mem, wait_value);
                const auto completion_wait_start =
                    completion_wait_stall ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
                if (executor_completion_flush_pending &&
                    !TestWaitRegMemValue(*wait_reg_mem, wait_value)) {
                    FlushGpuCompletionBatchForExecutor(false, true);
                }
#endif
                // One host thread services every Liverpool queue. A compute wait must suspend only
                // its coroutine; parking the host thread here can prevent the graphics queue from
                // ever reaching the PatchedFlip that releases this VideoOut label.
                while ((!Libraries::VideoOut::ExecutorTryReadVoLabelAlias(wait_addr, &wait_value) ||
                        !TestWaitRegMemValue(*wait_reg_mem, wait_value)) &&
                       !state->abort) {
                    YIELD_ASC(vqid);
                }
#ifdef __ANDROID__
                if (completion_wait_stall) {
                    const u64 elapsed_ns = static_cast<u64>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - completion_wait_start)
                            .count());
                    ++executor_completion_wait_stalls;
                    executor_completion_wait_stall_ns += elapsed_ns;
                    executor_completion_wait_stall_max_ns =
                        std::max(executor_completion_wait_stall_max_ns, elapsed_ns);
                }
#endif
                break;
            }
#ifdef __ANDROID__
            {
                u32 current_value{};
                ExecutorReadWaitRegMemValue(*wait_reg_mem, regs.reg_array, &current_value);
                if (!TestWaitRegMemValue(*wait_reg_mem, current_value)) {
                    const u64 dependency_tick = ConsumeCompletionWaitForExecutor(
                        state, Pm4Engine::Compute, *wait_reg_mem);
                    if (dependency_tick != 0) {
                        rasterizer->CompletionWaitBarrier();
                        ++executor_logical_wait_count;
                        break;
                    }
                }
                if (!TestWaitRegMemValue(*wait_reg_mem, current_value) &&
                    SynchronizeUntrackedMemoryWaitForExecutor(*wait_reg_mem, current_value)) {
                    ++executor_logical_wait_count;
                    break;
                }
            }
            const bool completion_wait_stall =
                executor_completion_flush_pending && !wait_reg_mem->Test(regs.reg_array);
            const auto completion_wait_start =
                completion_wait_stall ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
            if (executor_completion_flush_pending && !wait_reg_mem->Test(regs.reg_array)) {
                FlushGpuCompletionBatchForExecutor(false, true);
            }
#endif
            while (!wait_reg_mem->Test(regs.reg_array) && !state->abort) {
                YIELD_ASC(vqid);
            }
#ifdef __ANDROID__
            if (completion_wait_stall) {
                const u64 elapsed_ns = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - completion_wait_start)
                        .count());
                ++executor_completion_wait_stalls;
                executor_completion_wait_stall_ns += elapsed_ns;
                executor_completion_wait_stall_max_ns =
                    std::max(executor_completion_wait_stall_max_ns, elapsed_ns);
            }
#endif
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (rasterizer) {
                rasterizer->ProcessDownloadImages();
            }
            const u32 direct_write_size = [&] {
                switch (release_mem->data_sel.Value()) {
                case DataSelect::Data32Low:
                    return static_cast<u32>(sizeof(u32));
                case DataSelect::Data64:
                case DataSelect::GpuClock64:
                case DataSelect::PerfCounter:
                    return static_cast<u32>(sizeof(u64));
                default:
                    return 0u;
                }
            }();
#ifdef __ANDROID__
            if (rasterizer && ExecutorBoundedRasterizerPipeline()) {
                const auto packet = *release_mem;
                const auto signal_irq = [pipe_id = queue.pipe_id,
                                         int_sel = packet.int_sel.Value()] {
                    switch (int_sel) {
                    case InterruptSelect::None:
                        break;
                    case InterruptSelect::IrqUndocumented:
                    case InterruptSelect::IrqWhenWriteConfirm:
                        Platform::IrqC::Instance()->Signal(
                            static_cast<Platform::InterruptId>(pipe_id));
                        break;
                    default:
                        UNREACHABLE();
                    }
                };

                if (packet.data_sel.Value() == DataSelect::GdsMemStore) {
                    // Record the GDS copy now; only its completion interrupt belongs in the
                    // retirement callback.
                    rasterizer->CopyBuffer(packet.Address<VAddr>(), packet.gds_index,
                                           packet.num_dw * sizeof(u32), false, true);
                    const u64 completion_tick = ArmGpuCompletionTickForExecutor();
                    if (completion_tick != 0) {
                        rasterizer->GetScheduler().DeferPriorityOperationAt(
                            completion_tick, signal_irq);
                    } else {
                        signal_irq();
                    }
                } else {
                    const u64 completion_tick = ArmGpuCompletionTickForExecutor();
                    auto complete_release = [this, packet, signal_irq, completion_tick] {
                        void* const address = packet.Address<void*>();
                        bool wrote_completion = false;
                        u32 completion_bytes = 0;
                        switch (packet.data_sel.Value()) {
                        case DataSelect::Data32Low:
                            completion_bytes = sizeof(u32);
                            wrote_completion = ExecutorWriteFenceToCpuAddress(
                                "release_mem", address, packet.DataDWord(), sizeof(u32));
                            break;
                        case DataSelect::Data64:
                            completion_bytes = sizeof(u64);
                            wrote_completion = ExecutorWriteFenceToCpuAddress(
                                "release_mem", address, packet.DataQWord(), sizeof(u64));
                            break;
                        case DataSelect::GpuClock64:
                            completion_bytes = sizeof(u64);
                            wrote_completion = ExecutorWriteFenceToCpuAddress(
                                "release_mem", address, GetGpuClock64(), sizeof(u64));
                            break;
                        case DataSelect::PerfCounter:
                            completion_bytes = sizeof(u64);
                            wrote_completion = ExecutorWriteFenceToCpuAddress(
                                "release_mem", address, GetGpuPerfCounter(), sizeof(u64));
                            break;
                        default:
                            UNREACHABLE_MSG(
                                "Deferred direct RELEASE_MEM has unsupported data selector {}",
                                static_cast<u32>(packet.data_sel.Value()));
                        }
                        if (wrote_completion) {
                            CompletePendingCompletionWriteForExecutor(
                                reinterpret_cast<VAddr>(address), completion_bytes,
                                completion_tick);
                        }
                        signal_irq();
                    };
                    if (direct_write_size != 0) {
                        // The state map is GPU-coroutine-owned; invalidate it before handing only
                        // the memory publication/IRQ to the scheduler retirement worker.
                        InvalidateIndirectSnapshotsForWrite(
                            state, packet.Address<VAddr>(), direct_write_size);
                    }
                    if (completion_tick != 0) {
                        if (packet.data_sel.Value() == DataSelect::Data32Low) {
                            RecordCompletionWriteForExecutor(
                                state, Pm4Engine::Compute, packet.Address<VAddr>(),
                                packet.DataDWord(), sizeof(u32), completion_tick);
                        } else if (packet.data_sel.Value() == DataSelect::Data64) {
                            RecordCompletionWriteForExecutor(
                                state, Pm4Engine::Compute, packet.Address<VAddr>(),
                                packet.DataQWord(), sizeof(u64), completion_tick);
                        }
                        rasterizer->GetScheduler().DeferPriorityOperationAt(
                            completion_tick, std::move(complete_release));
                    } else {
                        complete_release();
                    }
                }
                break;
            }
#endif
            release_mem->SignalFence(
                [pipe_id = queue.pipe_id] {
                    Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
                },
                [this](VAddr dst, u16 gds_index, u16 num_dwords) {
                    rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
                });
            if (direct_write_size != 0) {
                InvalidateIndirectSnapshotsForWrite(
                    state, release_mem->Address<VAddr>(), direct_write_size);
            }
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            state->abort = true;
            LOG_ERROR(Lib_GnmDriver, "Unknown compute PM4 type 3 opcode {:#x} with count {}",
                      static_cast<u32>(opcode), header->type3.NumWords());
            break;
        }

        acb = NextPacket(acb, next_dw_off);

        if (advances_ring) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    if (state->abort && advances_ring) {
        // A malformed continuation must not poison the next independent DingDong segment.
        queue.tmp_dwords = 0;
    }
    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
#ifdef __ANDROID__
    ExecutorEopTraceLiverpoolReceive(dcb, ccb, IsRendererTerminal());
#endif
    if (IsRendererTerminal()) {
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
        return;
    }
    std::vector<u32> owned_dcb;
    std::vector<u32> owned_ccb;
    bool own_submitted_buffers = Config::copyGPUCmdBuffers();
#ifdef __ANDROID__
    // JIT maps each guest pthread to a native host pthread. Unity therefore starts rebuilding
    // its alternating command buffers as soon as the HLE submit returns, while Liverpool consumes
    // them asynchronously. A borrowed span can retain the draws yet lose the patched flip tail when
    // the next frame rewrites PrepareFlip in the same allocation. PC's timing often hides this race;
    // immutable per-task storage makes the actual PS4 ownership contract explicit on Android.
    own_submitted_buffers = true;
#endif
    if (own_submitted_buffers) {
        // The upstream shared copy arena assumes submit and SubmitDone cannot overlap. JIT
        // exposes the guest's 1:1 native threads, so that assumption is false: SubmitDone can reset
        // the arena offsets, or another submit can resize it, while a queued coroutine only retains
        // spans into the arena. Give each task immutable storage with exactly the task's lifetime.
        owned_dcb.assign(dcb.begin(), dcb.end());
        owned_ccb.assign(ccb.begin(), ccb.end());
    }

    const VAddr logical_dcb_base = reinterpret_cast<VAddr>(dcb.data());
    auto state = std::make_shared<Pm4SubmissionState>();
#ifdef __ANDROID__
    if (RenderWaveTrace::Enabled()) {
        state->trace_submit_sequence = RenderWaveTrace::NextSubmitSequence();
        state->trace_root_dcb = logical_dcb_base;
        state->trace_root_hash = RenderWaveTrace::HashWords(dcb);
        RenderWaveTrace::Submit(state->trace_submit_sequence, state->trace_root_dcb,
                                state->trace_root_hash, static_cast<u32>(dcb.size()),
                                static_cast<u32>(ccb.size()));
    }
#endif
    if (own_submitted_buffers) {
        SnapshotCpuIndirectGraph(Pm4Engine::Graphics, owned_dcb, state);
        SnapshotCpuIndirectGraph(Pm4Engine::Constant, owned_ccb, state);
    }
    auto task = ProcessGraphics(dcb, ccb, std::move(owned_dcb), std::move(owned_ccb),
                                std::move(state), 0, nullptr, logical_dcb_base);
    u32 published_submit_count = 0;
    if (!PublishSubmit(GfxQueueId, task.handle, &published_submit_count)) {
        task.handle.destroy();
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
        return;
    }
#ifdef __ANDROID__
    ExecutorEopTraceLiverpoolEnqueue(dcb, ccb, published_submit_count);
#endif
}

std::vector<u8> Liverpool::ExecutorCaptureGnmcapFrame(std::span<const u32> dcb,
                                                     std::span<const u32> ccb) const {
    using namespace Executor::GnmCap;
    Capture cap;
    // Submit record (references the DCB/CCB sections by index).
    const u32 dcb_section_idx = 1;  // section 0 = Submit, 1 = DCB, 2 = CCB (if present), 3 = Regs
    const u32 ccb_section_idx = ccb.empty() ? 0xFFFFFFFFu : 2u;
    {
        auto& s = cap.AddSection(SectionType::Submit);
        SubmitRec rec{0u, ccb.empty() ? 1u : 2u, dcb_section_idx, ccb_section_idx};
        Capture::AppendPod(s.payload, rec);
    }
    // DCB queue blob (raw dwords).
    {
        auto& s = cap.AddSection(SectionType::QueueBlob);
        QueueBlobHdr hdr{static_cast<uint32_t>(QueueType::Dcb), 0ull,
                         static_cast<uint32_t>(dcb.size()), 0u};
        Capture::AppendPod(s.payload, hdr);
        Capture::AppendBytes(s.payload, dcb.data(), dcb.size() * sizeof(u32));
    }
    // CCB queue blob (raw dwords), if any.
    if (!ccb.empty()) {
        auto& s = cap.AddSection(SectionType::QueueBlob);
        QueueBlobHdr hdr{static_cast<uint32_t>(QueueType::Ccb), 0ull,
                         static_cast<uint32_t>(ccb.size()), 0u};
        Capture::AppendPod(s.payload, hdr);
        Capture::AppendBytes(s.payload, ccb.data(), ccb.size() * sizeof(u32));
    }
    // Full raw Regs snapshot (what a real Liverpool replay needs to reconstruct render state).
    {
        auto& s = cap.AddSection(SectionType::Regs);
        RegsHdr hdr{0ull, static_cast<uint32_t>(sizeof(Regs)), 0u};
        Capture::AppendPod(s.payload, hdr);
        Capture::AppendBytes(s.payload, &regs, sizeof(Regs));
    }
    // Shader capture (phase-2): on the CPU writer side (PC shadPS4) the VS/PS program addresses in regs
    // are valid host pointers, so embed the real game GCN code + hash. On Android those addresses are
    // garbage -- this only runs where the memory is live (the capture side), giving the replayer a real
    // shader corpus to recompile (GetParams -> TranslateProgram -> EmitSPIRV -> vkCreateShaderModule).
    auto add_shader = [&](const auto& prog, uint32_t stage) {
        const u64 addr = reinterpret_cast<u64>(prog.template Address<const u8*>());
        if (addr == 0) {
            return;
        }
        try {
            const auto params = GetParams(prog);
            if (params.code.empty() || params.code.size() > (1u << 20)) {
                return;
            }
            auto& s = cap.AddSection(SectionType::Shader);
            ShaderHdr h{params.hash, stage,
                        static_cast<uint32_t>(params.code.size() * sizeof(u32)), 0u, 0u};
            Capture::AppendPod(s.payload, h);
            Capture::AppendBytes(s.payload, params.code.data(), params.code.size() * sizeof(u32));
        } catch (...) {
        }
    };
    // Capture whatever shader programs the live regs currently hold (a second Liverpool instance would
    // corrupt shared GPU state, so do NOT re-parse here). The submit-thread hook runs before the GPU
    // coroutine applies this DCB, so these can lag/be empty; a dedicated address-safe SetShReg extractor
    // is the follow-up for capturing every draw's shaders.
    add_shader(regs.ps_program, 0u);  // PS (stage 0)
    add_shader(regs.vs_program, 1u);  // VS (stage 1)
    return Serialize(cap);
}

Liverpool::ExecutorPm4ParseStats Liverpool::ExecutorParsePm4NoRaster(std::span<const u32> dcb) {
    ExecutorPm4ParseStats st{};
    st.rasterizer_bound = (rasterizer != nullptr);

    // Pre-walk the DCB using the REAL PM4 header struct: count type-3 packets + detect key opcodes.
    auto walk = dcb;
    while (walk.size() >= 1) {
        const auto* header = reinterpret_cast<const PM4Header*>(walk.data());
        if (header->type.Value() != 3) {
            break;  // our DCB is type-3 only
        }
        const u32 nwords = header->type3.NumWords();
        ++st.packets;
        switch (header->type3.opcode.Value()) {
        case PM4ItOpcode::SetContextReg:
            st.set_context_reg = true;
            break;
        case PM4ItOpcode::SetShReg:
            st.set_sh_reg = true;
            break;
        case PM4ItOpcode::DrawIndexAuto:
            st.draw_index_auto = true;
            break;
        default:
            break;
        }
        const u32 adv = nwords + 1;
        if (adv == 0 || adv > walk.size()) {
            break;
        }
        walk = walk.subspan(adv);
    }

    // Run the PRODUCTION parser synchronously. rasterizer is null, so DrawIndexAuto sets regs but never
    // issues a GPU draw (the draw body is guarded by `if (rasterizer)`). ProcessGraphics is a coroutine
    // suspended at start; resume until done. Our DCB has no co_await (no IndirectBuffer / GPU waits).
    auto task = ProcessGraphics(dcb, {});
    task.handle.resume();
    while (!task.handle.done()) {
        task.handle.resume();
    }
    task.handle.destroy();

    st.num_indices = regs.num_indices;
    st.regs_mutated = (regs.num_indices != 0) || st.set_context_reg || st.set_sh_reg;
    return st;
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    if (IsRendererTerminal()) {
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
        return;
    }
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    const auto vqid = gnm_vqid - 1;
    const VAddr logical_acb_base = reinterpret_cast<VAddr>(acb.data());
    // DingDong publishes this exact ring segment. Freeze it in the task just like a graphics root;
    // the guest may refill the ring after the HLE returns while the GPU coroutine is suspended.
    std::vector<u32> owned_acb{acb.begin(), acb.end()};
    auto state = std::make_shared<Pm4SubmissionState>();
    SnapshotCpuIndirectGraph(Pm4Engine::Compute, owned_acb, state);
    auto task = ProcessCompute(acb, vqid, std::move(owned_acb), std::move(state), 0,
                               logical_acb_base);
    if (!PublishSubmit(gnm_vqid, task.handle)) {
        task.handle.destroy();
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

} // namespace AmdGpu
