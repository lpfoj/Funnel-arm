// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/config.h"
#include "common/debug.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_driver_optimization.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#ifdef __ANDROID__
#include <array>
#include <atomic>
#include <cstdlib>
#include <android/log.h>
#endif

namespace Vulkan {

#ifdef __ANDROID__
namespace {

enum class ExecutorVkJournalOp : u32 {
    GraphicsDraw = 1,
    ComputeDispatch = 2,
    SubmitBegin = 3,
    SubmitComplete = 4,
};

struct ExecutorVkJournalEntry {
    std::atomic<u64> sequence{};
    ExecutorVkJournalOp op{};
    u64 stream_id{};
    u64 tick{};
    u64 pipeline_key_hash{};
    u64 pipeline_handle{};
    u64 vs_hash{};
    u64 ps_hash{};
    u64 cs_hash{};
    u64 address0{};
    u64 address1{};
    u64 color_address{};
    u64 depth_address{};
    u64 attachment_signature{};
    u32 op_variant{};
    u32 primitive_type{};
    u32 element_count{};
    u32 instance_count{};
    u32 max_draw_count{};
    u32 stride{};
    u32 color_width{};
    u32 color_height{};
    u32 color_format{};
    u32 depth_format{};
    u32 color_attachment_count{};
    u32 descriptor_writes{};
    u32 buffer_descriptors{};
    u32 image_descriptors{};
    s32 result{};
};

struct ExecutorVkGraphSpan {
    ExecutorVkJournalOp op{};
    u64 stream_id{};
    u64 first_sequence{};
    u64 last_sequence{};
    u64 first_tick{};
    u64 last_tick{};
    u64 attachment_signature{};
    u64 color_address{};
    u64 depth_address{};
    u64 first_pipeline_key_hash{};
    u64 last_pipeline_key_hash{};
    u64 first_vs_hash{};
    u64 last_vs_hash{};
    u64 first_ps_hash{};
    u64 last_ps_hash{};
    u64 first_cs_hash{};
    u64 last_cs_hash{};
    u64 total_elements{};
    u32 operation_count{};
    u32 pipeline_changes{};
    u32 draw_type_mask{};
    u32 max_elements{};
    u32 max_instances{};
    u32 color_width{};
    u32 color_height{};
    u32 color_format{};
    u32 depth_format{};
    u32 color_attachment_count{};
    u32 max_descriptor_writes{};
    u32 max_buffer_descriptors{};
    u32 max_image_descriptors{};
};

constexpr u32 ExecutorVkProbePreSpanCapacity = 12;
constexpr u32 ExecutorVkProbePostPrefixCapacity = 16;
constexpr u32 ExecutorVkProbePostTailCapacity = 16;

struct ExecutorVkProbeTrace {
    u64 stream_id{};
    u64 anchor_sequence{};
    u64 selected_draw_sequence{};
    u64 fragment_hash{};
    std::array<ExecutorVkGraphSpan, ExecutorVkProbePreSpanCapacity> pre_spans{};
    u32 pre_span_count{};
    u32 pre_total_spans{};
    ExecutorVkGraphSpan post_current{};
    bool post_current_valid{};
    std::array<ExecutorVkGraphSpan, ExecutorVkProbePostPrefixCapacity> post_prefix{};
    u32 post_prefix_count{};
    std::array<ExecutorVkGraphSpan, ExecutorVkProbePostTailCapacity> post_tail{};
    u32 post_tail_count{};
    u32 post_tail_next{};
    u32 post_total_spans{};
    u64 post_operations{};
};

constexpr u64 ExecutorVkJournalCapacity = 128;
std::array<ExecutorVkJournalEntry, ExecutorVkJournalCapacity> executor_vk_journal{};
std::atomic<u64> executor_vk_journal_sequence{0};
std::atomic<bool> executor_vk_forced_session{false};
std::atomic<bool> executor_vk_frame_capture_active{false};
std::atomic_flag executor_vk_journal_dumping = ATOMIC_FLAG_INIT;
constexpr u32 ExecutorVkProbeIdle = 0;
constexpr u32 ExecutorVkProbeInitializing = 1;
constexpr u32 ExecutorVkProbeArmed = 2;
constexpr u32 ExecutorVkProbeDumping = 3;
constexpr u32 ExecutorVkProbeDone = 4;
std::atomic<u32> executor_vk_probe_state{ExecutorVkProbeIdle};
std::atomic_flag executor_vk_probe_lock = ATOMIC_FLAG_INIT;
ExecutorVkProbeTrace executor_vk_probe_trace{};

constexpr u32 ExecutorVkFramePrefixCapacity = 64;
constexpr u32 ExecutorVkFrameTailCapacity = 64;
struct ExecutorVkFrameTrace {
    u64 epoch{};
    u64 first_sequence{};
    u64 last_sequence{};
    u64 operations{};
    ExecutorVkGraphSpan current{};
    bool current_valid{};
    std::array<ExecutorVkGraphSpan, ExecutorVkFramePrefixCapacity> prefix{};
    u32 prefix_count{};
    std::array<ExecutorVkGraphSpan, ExecutorVkFrameTailCapacity> tail{};
    u32 tail_count{};
    u32 tail_next{};
    u32 total_spans{};
};
constexpr u32 ExecutorVkFrameIdle = 0;
constexpr u32 ExecutorVkFrameArmed = 1;
constexpr u32 ExecutorVkFrameDumping = 2;
std::atomic<u32> executor_vk_frame_state{ExecutorVkFrameIdle};
std::atomic_flag executor_vk_frame_lock = ATOMIC_FLAG_INIT;
ExecutorVkFrameTrace executor_vk_frame_trace{};

void ExecutorVkProbeRecord(const ExecutorVkJournalEntry& entry, u64 sequence) noexcept;
void ExecutorVkFrameRecord(const ExecutorVkJournalEntry& entry, u64 sequence) noexcept;

bool ExecutorVkJournalRecordEnabled(bool force_record) noexcept {
    if (Config::managedGpuFastPath()) {
        return false;
    }
    if (force_record) {
        executor_vk_forced_session.store(true, std::memory_order_release);
        return true;
    }
    static const bool persistent_journal =
        std::getenv("EXECUTOR_VK_JOURNAL") != nullptr;
    return persistent_journal ||
           executor_vk_forced_session.load(std::memory_order_acquire) ||
           executor_vk_frame_capture_active.load(std::memory_order_acquire);
}

template <typename Fill>
u64 ExecutorVkJournalWrite(ExecutorVkJournalOp op, bool force_record, Fill&& fill) noexcept {
    if (!ExecutorVkJournalRecordEnabled(force_record)) [[likely]] {
        return 0;
    }
    const u64 sequence = executor_vk_journal_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    auto& entry = executor_vk_journal[(sequence - 1) % ExecutorVkJournalCapacity];
    entry.sequence.store(0, std::memory_order_relaxed);
    entry.op = op;
    fill(entry);
    entry.sequence.store(sequence, std::memory_order_release);
    ExecutorVkProbeRecord(entry, sequence);
    ExecutorVkFrameRecord(entry, sequence);
    return sequence;
}

bool ExecutorVkIsGraphOperation(const ExecutorVkJournalEntry& entry) noexcept {
    return entry.op == ExecutorVkJournalOp::GraphicsDraw ||
           entry.op == ExecutorVkJournalOp::ComputeDispatch;
}

ExecutorVkGraphSpan ExecutorVkMakeGraphSpan(const ExecutorVkJournalEntry& entry,
                                            u64 sequence) noexcept {
    ExecutorVkGraphSpan span{};
    span.op = entry.op;
    span.stream_id = entry.stream_id;
    span.first_sequence = sequence;
    span.last_sequence = sequence;
    span.first_tick = entry.tick;
    span.last_tick = entry.tick;
    span.attachment_signature = entry.attachment_signature;
    span.color_address = entry.color_address;
    span.depth_address = entry.depth_address;
    span.first_pipeline_key_hash = entry.pipeline_key_hash;
    span.last_pipeline_key_hash = entry.pipeline_key_hash;
    span.first_vs_hash = entry.vs_hash;
    span.last_vs_hash = entry.vs_hash;
    span.first_ps_hash = entry.ps_hash;
    span.last_ps_hash = entry.ps_hash;
    span.first_cs_hash = entry.cs_hash;
    span.last_cs_hash = entry.cs_hash;
    span.total_elements = entry.element_count;
    span.operation_count = 1;
    span.draw_type_mask = entry.op_variant < 32 ? 1U << entry.op_variant : 0;
    span.max_elements = entry.element_count;
    span.max_instances = entry.instance_count;
    span.color_width = entry.color_width;
    span.color_height = entry.color_height;
    span.color_format = entry.color_format;
    span.depth_format = entry.depth_format;
    span.color_attachment_count = entry.color_attachment_count;
    span.max_descriptor_writes = entry.descriptor_writes;
    span.max_buffer_descriptors = entry.buffer_descriptors;
    span.max_image_descriptors = entry.image_descriptors;
    return span;
}

bool ExecutorVkSameGraphSpan(const ExecutorVkGraphSpan& span,
                             const ExecutorVkJournalEntry& entry) noexcept {
    if (span.op != entry.op) {
        return false;
    }
    if (span.stream_id != entry.stream_id) {
        return false;
    }
    if (entry.op == ExecutorVkJournalOp::ComputeDispatch) {
        return span.last_cs_hash == entry.cs_hash;
    }
    return span.attachment_signature == entry.attachment_signature &&
           span.color_address == entry.color_address && span.depth_address == entry.depth_address &&
           span.color_width == entry.color_width && span.color_height == entry.color_height &&
           span.color_format == entry.color_format && span.depth_format == entry.depth_format &&
           span.color_attachment_count == entry.color_attachment_count;
}

void ExecutorVkExtendGraphSpan(ExecutorVkGraphSpan& span, const ExecutorVkJournalEntry& entry,
                               u64 sequence) noexcept {
    if (span.last_pipeline_key_hash != entry.pipeline_key_hash) {
        ++span.pipeline_changes;
    }
    span.last_sequence = sequence;
    span.last_tick = entry.tick;
    span.last_pipeline_key_hash = entry.pipeline_key_hash;
    span.last_vs_hash = entry.vs_hash;
    span.last_ps_hash = entry.ps_hash;
    span.last_cs_hash = entry.cs_hash;
    span.total_elements += entry.element_count;
    ++span.operation_count;
    if (entry.op_variant < 32) {
        span.draw_type_mask |= 1U << entry.op_variant;
    }
    span.max_elements = std::max(span.max_elements, entry.element_count);
    span.max_instances = std::max(span.max_instances, entry.instance_count);
    span.max_descriptor_writes = std::max(span.max_descriptor_writes, entry.descriptor_writes);
    span.max_buffer_descriptors =
        std::max(span.max_buffer_descriptors, entry.buffer_descriptors);
    span.max_image_descriptors = std::max(span.max_image_descriptors, entry.image_descriptors);
}

void ExecutorVkProbeLock() noexcept {
    while (executor_vk_probe_lock.test_and_set(std::memory_order_acquire)) {
    }
}

void ExecutorVkProbeUnlock() noexcept {
    executor_vk_probe_lock.clear(std::memory_order_release);
}

void ExecutorVkFrameLock() noexcept {
    while (executor_vk_frame_lock.test_and_set(std::memory_order_acquire)) {
    }
}

void ExecutorVkFrameUnlock() noexcept {
    executor_vk_frame_lock.clear(std::memory_order_release);
}

void ExecutorVkFrameStoreSpan(const ExecutorVkGraphSpan& span) noexcept {
    auto& trace = executor_vk_frame_trace;
    ++trace.total_spans;
    if (trace.prefix_count < trace.prefix.size()) {
        trace.prefix[trace.prefix_count++] = span;
        return;
    }
    trace.tail[trace.tail_next] = span;
    trace.tail_next = (trace.tail_next + 1) % trace.tail.size();
    trace.tail_count =
        std::min<u32>(trace.tail_count + 1, static_cast<u32>(trace.tail.size()));
}

void ExecutorVkFrameFinishCurrentSpan() noexcept {
    auto& trace = executor_vk_frame_trace;
    if (!trace.current_valid) {
        return;
    }
    ExecutorVkFrameStoreSpan(trace.current);
    trace.current = {};
    trace.current_valid = false;
}

void ExecutorVkFrameRecord(const ExecutorVkJournalEntry& entry, u64 sequence) noexcept {
    if (executor_vk_frame_state.load(std::memory_order_acquire) != ExecutorVkFrameArmed ||
        !ExecutorVkIsGraphOperation(entry)) {
        return;
    }

    ExecutorVkFrameLock();
    auto& trace = executor_vk_frame_trace;
    if (executor_vk_frame_state.load(std::memory_order_relaxed) != ExecutorVkFrameArmed) {
        ExecutorVkFrameUnlock();
        return;
    }
    if (trace.first_sequence == 0) {
        trace.first_sequence = sequence;
    }
    trace.last_sequence = sequence;
    ++trace.operations;
    if (!trace.current_valid) {
        trace.current = ExecutorVkMakeGraphSpan(entry, sequence);
        trace.current_valid = true;
    } else if (ExecutorVkSameGraphSpan(trace.current, entry)) {
        ExecutorVkExtendGraphSpan(trace.current, entry, sequence);
    } else {
        ExecutorVkFrameFinishCurrentSpan();
        trace.current = ExecutorVkMakeGraphSpan(entry, sequence);
        trace.current_valid = true;
    }
    ExecutorVkFrameUnlock();
}

void ExecutorVkProbeStorePostSpan(const ExecutorVkGraphSpan& span) noexcept {
    auto& trace = executor_vk_probe_trace;
    ++trace.post_total_spans;
    if (trace.post_prefix_count < trace.post_prefix.size()) {
        trace.post_prefix[trace.post_prefix_count++] = span;
        return;
    }
    trace.post_tail[trace.post_tail_next] = span;
    trace.post_tail_next = (trace.post_tail_next + 1) % trace.post_tail.size();
    trace.post_tail_count = std::min<u32>(trace.post_tail_count + 1,
                                          static_cast<u32>(trace.post_tail.size()));
}

void ExecutorVkProbeFinishCurrentPostSpan() noexcept {
    auto& trace = executor_vk_probe_trace;
    if (!trace.post_current_valid) {
        return;
    }
    ExecutorVkProbeStorePostSpan(trace.post_current);
    trace.post_current = {};
    trace.post_current_valid = false;
}

void ExecutorVkProbeRecord(const ExecutorVkJournalEntry& entry, u64 sequence) noexcept {
    if (executor_vk_probe_state.load(std::memory_order_acquire) != ExecutorVkProbeArmed ||
        !ExecutorVkIsGraphOperation(entry)) {
        return;
    }

    ExecutorVkProbeLock();
    auto& trace = executor_vk_probe_trace;
    if (executor_vk_probe_state.load(std::memory_order_relaxed) != ExecutorVkProbeArmed ||
        entry.stream_id != trace.stream_id || sequence <= trace.anchor_sequence) {
        ExecutorVkProbeUnlock();
        return;
    }

    ++trace.post_operations;
    if (!trace.post_current_valid) {
        trace.post_current = ExecutorVkMakeGraphSpan(entry, sequence);
        trace.post_current_valid = true;
    } else if (ExecutorVkSameGraphSpan(trace.post_current, entry)) {
        ExecutorVkExtendGraphSpan(trace.post_current, entry, sequence);
    } else {
        ExecutorVkProbeFinishCurrentPostSpan();
        trace.post_current = ExecutorVkMakeGraphSpan(entry, sequence);
        trace.post_current_valid = true;
    }
    ExecutorVkProbeUnlock();
}

void ExecutorVkProbePrintSpan(const char* phase, u32 index, const ExecutorVkGraphSpan& span,
                              u64 anchor_sequence) noexcept {
    const bool anchor = span.first_sequence <= anchor_sequence &&
                        anchor_sequence <= span.last_sequence;
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VK_GRAPH_SPAN] phase=%s index=%u kind=%s stream=0x%llx "
        "seq=%llu-%llu ticks=%llu-%llu "
        "ops=%u anchor=%u attach=0x%llx cb0=0x%llx db=0x%llx colors=%u extent=%ux%u "
        "colorFmt=%u depthFmt=%u firstKey=0x%llx lastKey=0x%llx keyChanges=%u "
        "firstVS=0x%llx lastVS=0x%llx firstPS=0x%llx lastPS=0x%llx "
        "firstCS=0x%llx lastCS=0x%llx drawTypes=0x%x totalElements=%llu "
        "maxElements=%u maxInstances=%u writesMax=%u buffersMax=%u imagesMax=%u",
        phase, index,
        span.op == ExecutorVkJournalOp::GraphicsDraw ? "draw" : "dispatch",
        static_cast<unsigned long long>(span.stream_id),
        static_cast<unsigned long long>(span.first_sequence),
        static_cast<unsigned long long>(span.last_sequence),
        static_cast<unsigned long long>(span.first_tick),
        static_cast<unsigned long long>(span.last_tick), span.operation_count, anchor ? 1u : 0u,
        static_cast<unsigned long long>(span.attachment_signature),
        static_cast<unsigned long long>(span.color_address),
        static_cast<unsigned long long>(span.depth_address), span.color_attachment_count,
        span.color_width, span.color_height, span.color_format, span.depth_format,
        static_cast<unsigned long long>(span.first_pipeline_key_hash),
        static_cast<unsigned long long>(span.last_pipeline_key_hash), span.pipeline_changes,
        static_cast<unsigned long long>(span.first_vs_hash),
        static_cast<unsigned long long>(span.last_vs_hash),
        static_cast<unsigned long long>(span.first_ps_hash),
        static_cast<unsigned long long>(span.last_ps_hash),
        static_cast<unsigned long long>(span.first_cs_hash),
        static_cast<unsigned long long>(span.last_cs_hash), span.draw_type_mask,
        static_cast<unsigned long long>(span.total_elements), span.max_elements,
        span.max_instances, span.max_descriptor_writes, span.max_buffer_descriptors,
        span.max_image_descriptors);
}

void ExecutorVkProbeDumpAtSubmit(const void* stream, u64 tick, vk::Result result) noexcept {
    if (executor_vk_probe_state.load(std::memory_order_acquire) != ExecutorVkProbeArmed ||
        reinterpret_cast<u64>(stream) != executor_vk_probe_trace.stream_id) {
        return;
    }
    u32 expected = ExecutorVkProbeArmed;
    if (!executor_vk_probe_state.compare_exchange_strong(
            expected, ExecutorVkProbeDumping, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return;
    }

    ExecutorVkProbeLock();
    ExecutorVkProbeFinishCurrentPostSpan();
    const auto& trace = executor_vk_probe_trace;
    const u32 retained_post = trace.post_prefix_count + trace.post_tail_count;
    const u32 omitted_post = trace.post_total_spans > retained_post
                                 ? trace.post_total_spans - retained_post
                                 : 0;
    const u32 omitted_pre = trace.pre_total_spans > trace.pre_span_count
                                ? trace.pre_total_spans - trace.pre_span_count
                                : 0;
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VK_GRAPH_BEGIN] stream=0x%llx selectedDraw=%llu selectedPS=0x%llx "
        "anchorSeq=%llu submitTick=%llu submitResult=%d preGroups=%u preOmitted=%u "
        "postGroups=%u postOperations=%llu postOmitted=%u",
        static_cast<unsigned long long>(trace.stream_id),
        static_cast<unsigned long long>(trace.selected_draw_sequence),
        static_cast<unsigned long long>(trace.fragment_hash),
        static_cast<unsigned long long>(trace.anchor_sequence),
        static_cast<unsigned long long>(tick), static_cast<s32>(result), trace.pre_total_spans,
        omitted_pre, trace.post_total_spans,
        static_cast<unsigned long long>(trace.post_operations), omitted_post);

    for (u32 i = 0; i < trace.pre_span_count; ++i) {
        ExecutorVkProbePrintSpan("before", i, trace.pre_spans[i], trace.anchor_sequence);
    }
    for (u32 i = 0; i < trace.post_prefix_count; ++i) {
        ExecutorVkProbePrintSpan("after_head", i, trace.post_prefix[i], trace.anchor_sequence);
    }
    const u32 tail_start = trace.post_tail_count == trace.post_tail.size()
                               ? trace.post_tail_next
                               : 0;
    for (u32 i = 0; i < trace.post_tail_count; ++i) {
        const u32 slot = (tail_start + i) % trace.post_tail.size();
        ExecutorVkProbePrintSpan("after_tail", i, trace.post_tail[slot],
                                 trace.anchor_sequence);
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VK_GRAPH_END] stream=0x%llx anchorSeq=%llu submitTick=%llu",
        static_cast<unsigned long long>(trace.stream_id),
        static_cast<unsigned long long>(trace.anchor_sequence),
        static_cast<unsigned long long>(tick));
    ExecutorVkProbeUnlock();
    executor_vk_probe_state.store(ExecutorVkProbeDone, std::memory_order_release);
}

}

u64 ExecutorVkJournalGraphicsDraw(const void* stream, u64 tick, u64 pipeline_key_hash,
                                  u64 pipeline_handle, u64 vs_hash, u64 ps_hash,
                                  ExecutorVkJournalDrawType draw_type, u32 primitive_type,
                                  u32 element_count,
                                  u32 instance_count, u32 max_draw_count, u32 stride,
                                  VAddr argument_address, VAddr count_address,
                                  VAddr color_address, VAddr depth_address,
                                  u64 attachment_signature, u32 color_attachment_count,
                                  u32 color_width, u32 color_height, u32 color_format,
                                  u32 depth_format, u32 descriptor_writes,
                                  u32 buffer_descriptors, u32 image_descriptors,
                                  bool force_record) noexcept {
    return ExecutorVkJournalWrite(ExecutorVkJournalOp::GraphicsDraw, force_record, [&](auto& entry) {
        entry.stream_id = reinterpret_cast<u64>(stream);
        entry.tick = tick;
        entry.pipeline_key_hash = pipeline_key_hash;
        entry.pipeline_handle = pipeline_handle;
        entry.vs_hash = vs_hash;
        entry.ps_hash = ps_hash;
        entry.cs_hash = 0;
        entry.address0 = argument_address;
        entry.address1 = count_address;
        entry.op_variant = static_cast<u32>(draw_type);
        entry.primitive_type = primitive_type;
        entry.element_count = element_count;
        entry.instance_count = instance_count;
        entry.max_draw_count = max_draw_count;
        entry.stride = stride;
        entry.color_address = color_address;
        entry.depth_address = depth_address;
        entry.attachment_signature = attachment_signature;
        entry.color_width = color_width;
        entry.color_height = color_height;
        entry.color_format = color_format;
        entry.depth_format = depth_format;
        entry.color_attachment_count = color_attachment_count;
        entry.descriptor_writes = descriptor_writes;
        entry.buffer_descriptors = buffer_descriptors;
        entry.image_descriptors = image_descriptors;
        entry.result = 0;
    });
}

void ExecutorVkJournalArmContentProbe(const void* stream, u64 journal_sequence,
                                      u64 selected_draw_sequence, u64 fragment_hash) noexcept {
    u32 expected = ExecutorVkProbeIdle;
    if (!executor_vk_probe_state.compare_exchange_strong(
            expected, ExecutorVkProbeInitializing, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return;
    }

    const u64 stream_id = reinterpret_cast<u64>(stream);
    if (journal_sequence == 0) {
        executor_vk_probe_state.store(ExecutorVkProbeIdle, std::memory_order_release);
        return;
    }
    const auto& anchor =
        executor_vk_journal[(journal_sequence - 1) % ExecutorVkJournalCapacity];
    if (anchor.sequence.load(std::memory_order_acquire) != journal_sequence ||
        anchor.stream_id != stream_id || anchor.op != ExecutorVkJournalOp::GraphicsDraw) {
        executor_vk_probe_state.store(ExecutorVkProbeIdle, std::memory_order_release);
        return;
    }

    executor_vk_probe_trace = {};
    auto& trace = executor_vk_probe_trace;
    trace.stream_id = stream_id;
    trace.anchor_sequence = journal_sequence;
    trace.selected_draw_sequence = selected_draw_sequence;
    trace.fragment_hash = fragment_hash;

    const u64 oldest = journal_sequence > ExecutorVkJournalCapacity
                           ? journal_sequence - ExecutorVkJournalCapacity + 1
                           : 1;
    u64 command_buffer_begin = oldest;
    for (u64 sequence = oldest; sequence < journal_sequence; ++sequence) {
        const auto& entry = executor_vk_journal[(sequence - 1) % ExecutorVkJournalCapacity];
        if (entry.sequence.load(std::memory_order_acquire) == sequence &&
            entry.stream_id == stream_id && entry.op == ExecutorVkJournalOp::SubmitComplete) {
            command_buffer_begin = sequence + 1;
        }
    }

    ExecutorVkGraphSpan current{};
    bool current_valid = false;
    const auto store_pre_span = [&](const ExecutorVkGraphSpan& span) {
        ++trace.pre_total_spans;
        if (trace.pre_span_count < trace.pre_spans.size()) {
            trace.pre_spans[trace.pre_span_count++] = span;
            return;
        }
        std::move(trace.pre_spans.begin() + 1, trace.pre_spans.end(),
                  trace.pre_spans.begin());
        trace.pre_spans.back() = span;
    };
    for (u64 sequence = command_buffer_begin; sequence <= journal_sequence; ++sequence) {
        const auto& entry = executor_vk_journal[(sequence - 1) % ExecutorVkJournalCapacity];
        if (entry.sequence.load(std::memory_order_acquire) != sequence ||
            entry.stream_id != stream_id || !ExecutorVkIsGraphOperation(entry)) {
            continue;
        }
        if (!current_valid) {
            current = ExecutorVkMakeGraphSpan(entry, sequence);
            current_valid = true;
        } else if (ExecutorVkSameGraphSpan(current, entry)) {
            ExecutorVkExtendGraphSpan(current, entry, sequence);
        } else {
            store_pre_span(current);
            current = ExecutorVkMakeGraphSpan(entry, sequence);
        }
    }
    if (current_valid) {
        store_pre_span(current);
    }

    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VK_GRAPH_ARM] stream=0x%llx selectedDraw=%llu selectedPS=0x%llx "
        "anchorSeq=%llu commandBufferBegin=%llu preGroups=%u retained=%u",
        static_cast<unsigned long long>(stream_id),
        static_cast<unsigned long long>(selected_draw_sequence),
        static_cast<unsigned long long>(fragment_hash),
        static_cast<unsigned long long>(journal_sequence),
        static_cast<unsigned long long>(command_buffer_begin), trace.pre_total_spans,
        trace.pre_span_count);
    executor_vk_probe_state.store(ExecutorVkProbeArmed, std::memory_order_release);
}

void ExecutorVkJournalArmFrameCapture(u64 epoch) noexcept {
    executor_vk_frame_state.store(ExecutorVkFrameDumping, std::memory_order_release);
    ExecutorVkFrameLock();
    executor_vk_frame_trace = {};
    executor_vk_frame_trace.epoch = epoch;
    ExecutorVkFrameUnlock();
    executor_vk_frame_capture_active.store(true, std::memory_order_release);
    executor_vk_frame_state.store(ExecutorVkFrameArmed, std::memory_order_release);
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VK_FRAME_GRAPH_ARM] epoch=%llu journalNext=%llu boundary=patched_flip",
        static_cast<unsigned long long>(epoch),
        static_cast<unsigned long long>(
            executor_vk_journal_sequence.load(std::memory_order_acquire) + 1));
}

void ExecutorVkJournalFinishFrameCapture(u64 epoch) noexcept {
    u32 expected = ExecutorVkFrameArmed;
    if (!executor_vk_frame_state.compare_exchange_strong(
            expected, ExecutorVkFrameDumping, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return;
    }
    executor_vk_frame_capture_active.store(false, std::memory_order_release);

    ExecutorVkFrameLock();
    ExecutorVkFrameFinishCurrentSpan();
    const auto& trace = executor_vk_frame_trace;
    if (trace.epoch != epoch) {
        ExecutorVkFrameUnlock();
        executor_vk_frame_state.store(ExecutorVkFrameIdle, std::memory_order_release);
        return;
    }
    const u32 retained = trace.prefix_count + trace.tail_count;
    const u32 omitted = trace.total_spans > retained ? trace.total_spans - retained : 0;
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VK_FRAME_GRAPH_BEGIN] epoch=%llu firstSeq=%llu lastSeq=%llu "
        "operations=%llu groups=%u retained=%u omitted=%u",
        static_cast<unsigned long long>(trace.epoch),
        static_cast<unsigned long long>(trace.first_sequence),
        static_cast<unsigned long long>(trace.last_sequence),
        static_cast<unsigned long long>(trace.operations), trace.total_spans, retained, omitted);
    for (u32 i = 0; i < trace.prefix_count; ++i) {
        ExecutorVkProbePrintSpan("frame_head", i, trace.prefix[i], 0);
    }
    const u32 tail_start = trace.tail_count == trace.tail.size() ? trace.tail_next : 0;
    for (u32 i = 0; i < trace.tail_count; ++i) {
        const u32 slot = (tail_start + i) % trace.tail.size();
        ExecutorVkProbePrintSpan("frame_tail", i, trace.tail[slot], 0);
    }
    __android_log_print(
        ANDROID_LOG_INFO, "LSX4Native",
        "[EXECUTOR_VK_FRAME_GRAPH_END] epoch=%llu operations=%llu groups=%u omitted=%u",
        static_cast<unsigned long long>(trace.epoch),
        static_cast<unsigned long long>(trace.operations), trace.total_spans, omitted);
    ExecutorVkFrameUnlock();
    executor_vk_frame_state.store(ExecutorVkFrameIdle, std::memory_order_release);
}

void ExecutorVkJournalComputeDispatch(const void* stream, u64 tick, u64 pipeline_key_hash,
                                      u64 pipeline_handle, u64 cs_hash, bool indirect,
                                      u32 dim_x, u32 dim_y, u32 dim_z,
                                      VAddr argument_address, u32 descriptor_writes,
                                      u32 buffer_descriptors, u32 image_descriptors,
                                      bool force_record) noexcept {
    ExecutorVkJournalWrite(ExecutorVkJournalOp::ComputeDispatch, force_record, [&](auto& entry) {
        entry.stream_id = reinterpret_cast<u64>(stream);
        entry.tick = tick;
        entry.pipeline_key_hash = pipeline_key_hash;
        entry.pipeline_handle = pipeline_handle;
        entry.vs_hash = 0;
        entry.ps_hash = 0;
        entry.cs_hash = cs_hash;
        entry.address0 = argument_address;
        entry.address1 = 0;
        entry.op_variant = indirect ? 1U : 0U;
        entry.element_count = dim_x;
        entry.instance_count = dim_y;
        entry.max_draw_count = dim_z;
        entry.stride = 0;
        entry.color_address = 0;
        entry.depth_address = 0;
        entry.attachment_signature = 0;
        entry.color_width = 0;
        entry.color_height = 0;
        entry.color_format = 0;
        entry.depth_format = 0;
        entry.color_attachment_count = 0;
        entry.descriptor_writes = descriptor_writes;
        entry.buffer_descriptors = buffer_descriptors;
        entry.image_descriptors = image_descriptors;
        entry.result = 0;
    });
}

void ExecutorVkJournalSubmit(const void* stream, u64 tick, bool completed,
                             vk::Result result) noexcept {
    ExecutorVkJournalWrite(completed ? ExecutorVkJournalOp::SubmitComplete
                                     : ExecutorVkJournalOp::SubmitBegin,
                           false, [&](auto& entry) {
                               entry.stream_id = reinterpret_cast<u64>(stream);
                               entry.tick = tick;
                               entry.pipeline_key_hash = 0;
                               entry.pipeline_handle = 0;
                               entry.vs_hash = 0;
                               entry.ps_hash = 0;
                               entry.cs_hash = 0;
                               entry.address0 = 0;
                               entry.address1 = 0;
                               entry.op_variant = 0;
                               entry.element_count = 0;
                               entry.instance_count = 0;
                               entry.max_draw_count = 0;
                               entry.stride = 0;
                               entry.color_address = 0;
                               entry.depth_address = 0;
                               entry.attachment_signature = 0;
                               entry.color_width = 0;
                               entry.color_height = 0;
                               entry.color_format = 0;
                               entry.depth_format = 0;
                               entry.color_attachment_count = 0;
                               entry.descriptor_writes = 0;
                               entry.buffer_descriptors = 0;
                               entry.image_descriptors = 0;
                               entry.result = static_cast<s32>(result);
                           });
    if (completed) {
        ExecutorVkProbeDumpAtSubmit(stream, tick, result);
    }
}

void ExecutorVkJournalDump(const char* reason, u64 failing_tick) noexcept {
    if (executor_vk_journal_dumping.test_and_set(std::memory_order_acq_rel)) {
        return;
    }

    const u64 newest = executor_vk_journal_sequence.load(std::memory_order_acquire);
    constexpr u64 DumpEntries = 32;
    const u64 oldest = newest > DumpEntries ? newest - DumpEntries + 1 : 1;
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_VK_JOURNAL_BEGIN] reason=%s failingTick=%llu oldest=%llu "
                        "newest=%llu",
                        reason ? reason : "unknown", static_cast<unsigned long long>(failing_tick),
                        static_cast<unsigned long long>(oldest),
                        static_cast<unsigned long long>(newest));

    for (u64 sequence = oldest; sequence <= newest; ++sequence) {
        const auto& entry = executor_vk_journal[(sequence - 1) % ExecutorVkJournalCapacity];
        if (entry.sequence.load(std::memory_order_acquire) != sequence) {
            continue;
        }
        if (entry.op == ExecutorVkJournalOp::GraphicsDraw) {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_VK_JOURNAL_DRAW] seq=%llu stream=0x%llx tick=%llu type=%u prim=%u "
                "key=0x%llx "
                "pipe=0x%llx vs=0x%llx ps=0x%llx elements=%u instances=%u "
                "maxDraws=%u stride=%u args=0x%llx countAddr=0x%llx attach=0x%llx "
                "cb0=0x%llx db=0x%llx colors=%u extent=%ux%u colorFmt=%u depthFmt=%u "
                "writes=%u buffers=%u images=%u",
                static_cast<unsigned long long>(sequence),
                static_cast<unsigned long long>(entry.stream_id),
                static_cast<unsigned long long>(entry.tick),
                entry.op_variant, entry.primitive_type,
                static_cast<unsigned long long>(entry.pipeline_key_hash),
                static_cast<unsigned long long>(entry.pipeline_handle),
                static_cast<unsigned long long>(entry.vs_hash),
                static_cast<unsigned long long>(entry.ps_hash), entry.element_count,
                entry.instance_count, entry.max_draw_count, entry.stride,
                static_cast<unsigned long long>(entry.address0),
                static_cast<unsigned long long>(entry.address1),
                static_cast<unsigned long long>(entry.attachment_signature),
                static_cast<unsigned long long>(entry.color_address),
                static_cast<unsigned long long>(entry.depth_address),
                entry.color_attachment_count, entry.color_width, entry.color_height,
                entry.color_format, entry.depth_format,
                entry.descriptor_writes, entry.buffer_descriptors, entry.image_descriptors);
        } else if (entry.op == ExecutorVkJournalOp::ComputeDispatch) {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_VK_JOURNAL_DISPATCH] seq=%llu stream=0x%llx tick=%llu indirect=%u key=0x%llx "
                "pipe=0x%llx cs=0x%llx dims=%ux%ux%u args=0x%llx writes=%u buffers=%u "
                "images=%u",
                static_cast<unsigned long long>(sequence),
                static_cast<unsigned long long>(entry.stream_id),
                static_cast<unsigned long long>(entry.tick), entry.op_variant,
                static_cast<unsigned long long>(entry.pipeline_key_hash),
                static_cast<unsigned long long>(entry.pipeline_handle),
                static_cast<unsigned long long>(entry.cs_hash), entry.element_count,
                entry.instance_count, entry.max_draw_count,
                static_cast<unsigned long long>(entry.address0), entry.descriptor_writes,
                entry.buffer_descriptors, entry.image_descriptors);
        } else {
            __android_log_print(
                ANDROID_LOG_ERROR, "LSX4Native",
                "[EXECUTOR_VK_JOURNAL_SUBMIT] seq=%llu stream=0x%llx tick=%llu phase=%s result=%d",
                static_cast<unsigned long long>(sequence),
                static_cast<unsigned long long>(entry.stream_id),
                static_cast<unsigned long long>(entry.tick),
                entry.op == ExecutorVkJournalOp::SubmitComplete ? "complete" : "begin",
                entry.result);
        }
    }
    __android_log_print(ANDROID_LOG_ERROR, "LSX4Native",
                        "[EXECUTOR_VK_JOURNAL_END] newest=%llu",
                        static_cast<unsigned long long>(newest));
}
#endif

#ifdef __ANDROID__
namespace {

struct ExecutorRenderScopeTelemetry {
    const Scheduler* owner{};
    std::array<u64, static_cast<std::size_t>(Scheduler::EndRenderingReason::Count)>
        end_reason_counts{};
    std::array<u64, static_cast<std::size_t>(Scheduler::EndRenderingReason::Count)>
        same_state_reopen_counts{};
    RenderState last_ended_render_state{};
    Scheduler::EndRenderingReason last_end_reason{Scheduler::EndRenderingReason::Unknown};
    u64 begin_count{};
    u64 end_count{};
    u64 same_state_reopen_count{};
    u64 consumed_clear_reuse_count{};
    bool has_last_ended_render_state{};
};

bool HasRenderClear(const RenderState& state) noexcept {
    for (u32 index = 0; index < state.num_color_attachments; ++index) {
        if (state.color_attachments[index].is_clear != 0) {
            return true;
        }
    }
    return state.depth_stencil_attachment.depth_clear ||
           state.depth_stencil_attachment.stencil_clear;
}

bool CanReuseAfterConsumedClear(const RenderState& active,
                                const RenderState& requested) noexcept {
    if (!HasRenderClear(active) || HasRenderClear(requested)) {
        return false;
    }

    RenderState normalized_active = active;
    RenderState normalized_requested = requested;
    for (u32 index = 0; index < normalized_active.color_attachments.size(); ++index) {
        normalized_active.color_attachments[index].is_clear = 0;
        normalized_active.color_attachments[index].clear_value = {};
        normalized_requested.color_attachments[index].is_clear = 0;
        normalized_requested.color_attachments[index].clear_value = {};
    }
    normalized_active.depth_stencil_attachment.depth_clear = false;
    normalized_active.depth_stencil_attachment.stencil_clear = false;
    normalized_active.depth_stencil_attachment.clear_value = {};
    normalized_requested.depth_stencil_attachment.depth_clear = false;
    normalized_requested.depth_stencil_attachment.stencil_clear = false;
    normalized_requested.depth_stencil_attachment.clear_value = {};
    return normalized_active == normalized_requested;
}

ExecutorRenderScopeTelemetry& RenderScopeTelemetryFor(const Scheduler* owner) {
    static thread_local ExecutorRenderScopeTelemetry telemetry{};
    if (telemetry.owner != owner) {
        telemetry = {};
        telemetry.owner = owner;
    }
    return telemetry;
}

} // namespace
#endif

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
    retirement_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::RetirementPendingOpsThread, this));
}

Scheduler::~Scheduler() {
#ifdef __ANDROID__
    if (pending_submit_thread.joinable()) {
        pending_submit_thread.request_stop();
        pending_submit_cv.notify_all();
        pending_submit_thread.join();
    }
    {
        std::scoped_lock lk{submit_mutex};
        SubmitPendingBatchLocked(nullptr, {}, 0, true);
    }
    const u64 current_tick = master_semaphore.CurrentTick();
    if (current_tick > 1 && !master_semaphore.IsDeviceLost()) {
        master_semaphore.Wait(current_tick - 1);
    }
#endif
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& func) {
    const u64 gpu_tick = CurrentTick();
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        std::scoped_lock lk{pending_ops_mutex};
        pending_ops.emplace(std::move(func), gpu_tick);
        const u64 previous_count = pending_ops_count.fetch_add(1, std::memory_order_release);
        if (previous_count == 0) {
            pending_ops_poll_sequence.store(0, std::memory_order_relaxed);
        }
        return;
    }

    {
        std::unique_lock lk(priority_pending_ops_mutex);
        priority_pending_ops.emplace(std::move(func), gpu_tick);
    }
    priority_pending_ops_cv.notify_one();
}

void Scheduler::DeferPriorityOperationAt(u64 gpu_tick, Common::UniqueFunction<void>&& func) {
    if (Libraries::GnmDriver::ExecutorReplayActive()) {
        std::scoped_lock lk{pending_ops_mutex};
        pending_ops.emplace(std::move(func), gpu_tick);
        const u64 previous_count = pending_ops_count.fetch_add(1, std::memory_order_release);
        if (previous_count == 0) {
            pending_ops_poll_sequence.store(0, std::memory_order_relaxed);
        }
        return;
    }

    bool notify_retirement_worker = false;
    {
        std::unique_lock lk(retirement_pending_ops_mutex);
        notify_retirement_worker = retirement_pending_ops.empty();
        retirement_pending_ops[gpu_tick].emplace(std::move(func), gpu_tick);
        ++retirement_outstanding_ops[gpu_tick];
        retirement_outstanding_count.fetch_add(1, std::memory_order_release);
    }
    if (notify_retirement_worker) {
        retirement_pending_ops_cv.notify_one();
    }
}

void Scheduler::MarkTerminal(vk::Result result, const char* where) noexcept {
    master_semaphore.MarkTerminal(result, where);
#ifdef __ANDROID__
    pending_submit_cv.notify_all();
#endif
    priority_pending_ops_cv.notify_all();
    retirement_pending_ops_cv.notify_all();
    event_cv.notify_all();
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (master_semaphore.IsDeviceLost()) {
        return;
    }
    if (is_rendering && render_state == new_state) {
        return;
    }
#ifdef __ANDROID__
    auto& telemetry = RenderScopeTelemetryFor(this);
    const bool managed_render_scope_reuse =
        VideoCore::IsManagedVulkanRenderScopeReuse();
    if (is_rendering && managed_render_scope_reuse &&
        CanReuseAfterConsumedClear(render_state, new_state)) {
        render_state = new_state;
        ++telemetry.consumed_clear_reuse_count;
        VideoCore::RecordVulkanRenderScopeReuse();
        return;
    }
    if (managed_render_scope_reuse) {
        ++telemetry.begin_count;
        if (!is_rendering && telemetry.has_last_ended_render_state &&
            telemetry.last_ended_render_state == new_state) {
            ++telemetry.same_state_reopen_count;
            ++telemetry.same_state_reopen_counts[static_cast<std::size_t>(
                telemetry.last_end_reason)];
        }
    }
    if (managed_render_scope_reuse &&
        (telemetry.begin_count == 256 ||
         (telemetry.begin_count & 4095u) == 0)) {
        const auto reason_count = [&](const EndRenderingReason reason) {
            return telemetry.end_reason_counts[static_cast<std::size_t>(reason)];
        };
        const auto reopen_count = [&](const EndRenderingReason reason) {
            return telemetry.same_state_reopen_counts[static_cast<std::size_t>(reason)];
        };
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_RENDER_SCOPE_TELEMETRY] begins=%llu ends=%llu sameStateReopen=%llu "
            "consumedClearReuse=%llu "
            "reasonEndReopen=unknown:%llu/%llu,state:%llu/%llu,pipeline:%llu/%llu,"
            "rastBarrier:%llu/%llu,draw:%llu/%llu,clear:%llu/%llu,copy:%llu/%llu,"
            "dispatch:%llu/%llu,upload:%llu/%llu,download:%llu/%llu,transit:%llu/%llu,"
            "refresh:%llu/%llu,bufFill:%llu/%llu,bufRead:%llu/%llu,"
            "bufCopy:%llu/%llu,bufProbe:%llu/%llu,bufUpload:%llu/%llu,"
            "faultCompute:%llu/%llu,tileCompute:%llu/%llu,blit:%llu/%llu,"
            "shaderHle:%llu/%llu,presenter:%llu/%llu,flush:%llu/%llu,submit:%llu/%llu",
            static_cast<unsigned long long>(telemetry.begin_count),
            static_cast<unsigned long long>(telemetry.end_count),
            static_cast<unsigned long long>(telemetry.same_state_reopen_count),
            static_cast<unsigned long long>(telemetry.consumed_clear_reuse_count),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::Unknown)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::Unknown)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::BeginStateChange)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::BeginStateChange)),
            static_cast<unsigned long long>(
                reason_count(EndRenderingReason::PipelineResourceBarrier)),
            static_cast<unsigned long long>(
                reopen_count(EndRenderingReason::PipelineResourceBarrier)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::RasterizerBarrier)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::RasterizerBarrier)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::RasterizerDraw)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::RasterizerDraw)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::RasterizerClear)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::RasterizerClear)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::RasterizerCopy)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::RasterizerCopy)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::RasterizerDispatch)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::RasterizerDispatch)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::ImageUpload)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::ImageUpload)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::ImageDownload)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::ImageDownload)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::ImageTransit)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::ImageTransit)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::TextureCacheRefresh)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::TextureCacheRefresh)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::BufferFill)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::BufferFill)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::BufferReadback)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::BufferReadback)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::BufferCopy)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::BufferCopy)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::BufferProbe)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::BufferProbe)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::BufferUpload)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::BufferUpload)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::FaultCompute)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::FaultCompute)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::TileCompute)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::TileCompute)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::BlitHelper)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::BlitHelper)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::ShaderHleCopy)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::ShaderHleCopy)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::Presenter)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::Presenter)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::Flush)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::Flush)),
            static_cast<unsigned long long>(reason_count(EndRenderingReason::Submit)),
            static_cast<unsigned long long>(reopen_count(EndRenderingReason::Submit)));
    }
#endif
    EndRendering(EndRenderingReason::BeginStateChange);
#ifdef __ANDROID__
    if (managed_render_scope_reuse) {
        telemetry.has_last_ended_render_state = false;
    }
#endif
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    ++rendering_epoch;
    command_recording_epoch.fetch_add(1, std::memory_order_relaxed);
    command_buffer_used = true;
    current_cmdbuf.beginRendering(rendering_info);
}

void Scheduler::EndRendering(const EndRenderingReason reason) {
    if (master_semaphore.IsDeviceLost()) {
        is_rendering = false;
#ifdef __ANDROID__
        RenderScopeTelemetryFor(this).has_last_ended_render_state = false;
#endif
        return;
    }
    if (!is_rendering) {
        return;
    }
#ifdef __ANDROID__
    if (VideoCore::IsManagedVulkanRenderScopeReuse()) {
        auto& telemetry = RenderScopeTelemetryFor(this);
        ++telemetry.end_count;
        ++telemetry.end_reason_counts[static_cast<std::size_t>(reason)];
        telemetry.last_ended_render_state = render_state;
        telemetry.last_end_reason = reason;
        telemetry.has_last_ended_render_state = true;
    }
#endif
    is_rendering = false;
    current_cmdbuf.endRendering();
}

bool Scheduler::TryCompletionBarrierInsideRendering() {
#ifdef __ANDROID__
    // Adreno's dynamic-rendering local-read implementation can corrupt the
    // command-buffer state after an in-rendering pipelineBarrier2 sequence.
    // The failure surfaces later in an unrelated barrier, so keep the normal
    // end-rendering dependency path on Qualcomm.
    constexpr u32 QualcommVendorId = 0x5143;
    if (!is_rendering || !VideoCore::IsManagedVulkanDriverOptimization() ||
        instance.GetVendorID() == QualcommVendorId ||
        !instance.IsDynamicRenderingLocalReadSupported() ||
        !pending_buffer_transfers.empty() || master_semaphore.IsDeviceLost()) {
        return false;
    }

    // While dynamic rendering is active, all compute, transfer, layout
    // transition and attachment-replacement paths have already ended the
    // scope. The remaining producer work is therefore graphics-shader work.
    // Keep framebuffer-space stages out of this dependency so the Vulkan
    // in-rendering synchronization rules remain satisfied.
    constexpr vk::PipelineStageFlags2 ShaderStages =
        vk::PipelineStageFlagBits2::eVertexShader |
        vk::PipelineStageFlagBits2::eTessellationControlShader |
        vk::PipelineStageFlagBits2::eTessellationEvaluationShader |
        vk::PipelineStageFlagBits2::eGeometryShader |
        vk::PipelineStageFlagBits2::eFragmentShader;
    constexpr vk::PipelineStageFlags2 ConsumerStages =
        ShaderStages | vk::PipelineStageFlagBits2::eDrawIndirect |
        vk::PipelineStageFlagBits2::eVertexInput;
    constexpr vk::MemoryBarrier2 shader_barrier{
        .srcStageMask = ShaderStages,
        .srcAccessMask =
            vk::AccessFlagBits2::eShaderWrite |
            vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = ConsumerStages,
        .dstAccessMask =
            vk::AccessFlagBits2::eShaderRead |
            vk::AccessFlagBits2::eShaderWrite |
            vk::AccessFlagBits2::eMemoryRead |
            vk::AccessFlagBits2::eMemoryWrite,
    };
    current_cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &shader_barrier,
    });

    // EOP/EOS completion also covers attachment writes. Keep that dependency
    // separate because Vulkan requires a barrier whose source includes a
    // framebuffer-space stage to target framebuffer-space stages only while
    // inside a rendering instance.
    constexpr vk::PipelineStageFlags2 FramebufferStages =
        vk::PipelineStageFlagBits2::eEarlyFragmentTests |
        vk::PipelineStageFlagBits2::eLateFragmentTests |
        vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    constexpr vk::MemoryBarrier2 framebuffer_barrier{
        .srcStageMask = FramebufferStages,
        .srcAccessMask =
            vk::AccessFlagBits2::eColorAttachmentWrite |
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
        .dstStageMask = FramebufferStages,
        .dstAccessMask =
            vk::AccessFlagBits2::eColorAttachmentRead |
            vk::AccessFlagBits2::eColorAttachmentWrite |
            vk::AccessFlagBits2::eDepthStencilAttachmentRead |
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
    };
    current_cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &framebuffer_barrier,
    });
    command_buffer_used = true;
    command_recording_epoch.fetch_add(1, std::memory_order_relaxed);
    return true;
#else
    return false;
#endif
}

void Scheduler::QueueBufferCopies(const vk::Buffer source, const vk::Buffer destination,
                                  const std::span<const vk::BufferCopy> regions,
                                  const VAddr source_guest_address, const u64 source_size,
                                  const VAddr destination_guest_address,
                                  const u64 destination_size,
                                  const BufferTransferOrigin origin) {
    if (regions.empty() || master_semaphore.IsDeviceLost()) {
        return;
    }
    ASSERT(origin < BufferTransferOrigin::Count);
    VideoCore::RecordVulkanTransferRequest(regions.size());
    static constexpr u64 MaxPendingRegions = 4096;
    static constexpr std::size_t MaxPendingGroups = 256;
    if (!pending_buffer_transfers.empty() &&
        (pending_buffer_transfers.size() >= MaxPendingGroups ||
         regions.size() > MaxPendingRegions - std::min(pending_buffer_transfer_regions,
                                                        MaxPendingRegions))) {
        FlushPendingBufferTransfers();
    }

    const auto range_end = [](const VAddr begin, const u64 size) {
        return size > std::numeric_limits<VAddr>::max() - begin
                   ? std::numeric_limits<VAddr>::max()
                   : begin + size;
    };
    const VAddr source_end = range_end(source_guest_address, source_size);
    const VAddr destination_end = range_end(destination_guest_address, destination_size);
    const auto overlaps = [](const VAddr lhs_begin, const VAddr lhs_end,
                             const VAddr rhs_begin, const VAddr rhs_end) {
        return lhs_begin < lhs_end && rhs_begin < rhs_end &&
               lhs_begin < rhs_end && rhs_begin < lhs_end;
    };
    vk::DeviceSize source_offset_begin = std::numeric_limits<vk::DeviceSize>::max();
    vk::DeviceSize source_offset_end = 0;
    vk::DeviceSize destination_offset_begin =
        std::numeric_limits<vk::DeviceSize>::max();
    vk::DeviceSize destination_offset_end = 0;
    for (const auto& region : regions) {
        source_offset_begin = std::min(source_offset_begin, region.srcOffset);
        source_offset_end = std::max(source_offset_end,
                                     region.srcOffset + region.size);
        destination_offset_begin =
            std::min(destination_offset_begin, region.dstOffset);
        destination_offset_end = std::max(destination_offset_end,
                                          region.dstOffset + region.size);
    }
    const auto offset_overlaps = [](const vk::DeviceSize lhs_begin,
                                    const vk::DeviceSize lhs_end,
                                    const vk::DeviceSize rhs_begin,
                                    const vk::DeviceSize rhs_end) {
        return lhs_begin < lhs_end && rhs_begin < rhs_end &&
               lhs_begin < rhs_end && rhs_begin < lhs_end;
    };
    const bool dependency_barrier_before = std::ranges::any_of(
        pending_buffer_transfers, [&](const PendingBufferTransfer& previous) {
            // Preserve RAW, WAR and WAW ordering while still allowing disjoint
            // transfer groups to share one render-scope break and barrier pair.
            const bool guest_hazard =
                overlaps(source_guest_address, source_end,
                         previous.destination_begin, previous.destination_end) ||
                overlaps(destination_guest_address, destination_end,
                         previous.source_begin, previous.source_end) ||
                overlaps(destination_guest_address, destination_end,
                         previous.destination_begin, previous.destination_end);
            const bool buffer_hazard =
                (source == previous.destination &&
                 offset_overlaps(source_offset_begin, source_offset_end,
                                 previous.destination_offset_begin,
                                 previous.destination_offset_end)) ||
                (destination == previous.source &&
                 offset_overlaps(destination_offset_begin,
                                 destination_offset_end,
                                 previous.source_offset_begin,
                                 previous.source_offset_end)) ||
                (destination == previous.destination &&
                 offset_overlaps(destination_offset_begin,
                                 destination_offset_end,
                                 previous.destination_offset_begin,
                                 previous.destination_offset_end));
            return guest_hazard || buffer_hazard;
        });

    PendingBufferTransfer* transfer = nullptr;
    if (!dependency_barrier_before && !pending_buffer_transfers.empty()) {
        for (auto candidate = pending_buffer_transfers.rbegin();
             candidate != pending_buffer_transfers.rend(); ++candidate) {
            if (candidate->source != source ||
                candidate->destination != destination) {
                continue;
            }
            transfer = &*candidate;
            candidate->source_begin =
                std::min(candidate->source_begin, source_guest_address);
            candidate->source_end = std::max(candidate->source_end, source_end);
            candidate->destination_begin =
                std::min(candidate->destination_begin,
                         destination_guest_address);
            candidate->destination_end =
                std::max(candidate->destination_end, destination_end);
            candidate->source_offset_begin =
                std::min(candidate->source_offset_begin, source_offset_begin);
            candidate->source_offset_end =
                std::max(candidate->source_offset_end, source_offset_end);
            candidate->destination_offset_begin =
                std::min(candidate->destination_offset_begin,
                         destination_offset_begin);
            candidate->destination_offset_end =
                std::max(candidate->destination_offset_end,
                         destination_offset_end);
            break;
        }
    }
    if (transfer == nullptr) {
        transfer = &pending_buffer_transfers.emplace_back(PendingBufferTransfer{
            .source = source,
            .destination = destination,
            .source_begin = source_guest_address,
            .source_end = source_end,
            .destination_begin = destination_guest_address,
            .destination_end = destination_end,
            .source_offset_begin = source_offset_begin,
            .source_offset_end = source_offset_end,
            .destination_offset_begin = destination_offset_begin,
            .destination_offset_end = destination_offset_end,
            .dependency_barrier_before = dependency_barrier_before,
        });
    }
    for (const vk::BufferCopy& region : regions) {
        bool coalesced = false;
        for (vk::BufferCopy& pending_region : transfer->regions) {
            if (pending_region.srcOffset + pending_region.size == region.srcOffset &&
                pending_region.dstOffset + pending_region.size == region.dstOffset) {
                pending_region.size += region.size;
                coalesced = true;
                break;
            }
            if (region.srcOffset + region.size == pending_region.srcOffset &&
                region.dstOffset + region.size == pending_region.dstOffset) {
                pending_region.srcOffset = region.srcOffset;
                pending_region.dstOffset = region.dstOffset;
                pending_region.size += region.size;
                coalesced = true;
                break;
            }
        }
        if (!coalesced) {
            transfer->regions.push_back(region);
        }
    }
    ++pending_buffer_transfer_requests;
    ++pending_buffer_transfer_request_epoch;
    pending_buffer_transfer_regions += regions.size();
    pending_buffer_transfer_origins |= u32{1} << static_cast<u32>(origin);
    pending_gnm_dma_transfers += origin == BufferTransferOrigin::GnmDma;

#ifdef __ANDROID__
    ++executor_buffer_transfers_queued;
    executor_buffer_transfer_regions_queued += regions.size();
    executor_buffer_transfer_max_queued =
        std::max<u64>(executor_buffer_transfer_max_queued, pending_buffer_transfers.size());
    switch (origin) {
    case BufferTransferOrigin::Hle:
        ++executor_hle_buffer_transfers_queued;
        break;
    case BufferTransferOrigin::Upload:
        ++executor_buffer_uploads_queued;
        break;
    case BufferTransferOrigin::GnmDma:
        ++executor_gnm_dma_transfers_queued;
        executor_gnm_dma_regions_queued += regions.size();
        break;
    case BufferTransferOrigin::Count:
        break;
    }
#endif
    if (!VideoCore::IsManagedVulkanTransferCoalescing()) {
        // A/B baseline: materialize every request immediately with the
        // conservative barrier path so no transfer calls are coalesced.
        FlushPendingBufferTransfers();
    }
}

void Scheduler::QueueBufferUploadCopies(const vk::Buffer source,
                                        const vk::Buffer destination,
                                        const std::span<const vk::BufferCopy> regions,
                                        const VAddr destination_guest_address,
                                        const u64 destination_size) {
    QueueBufferCopies(source, destination, regions, 0, 0,
                      destination_guest_address, destination_size,
                      BufferTransferOrigin::Upload);
}

void Scheduler::FlushPendingBufferTransfersForRange(const VAddr guest_address, const u64 size) {
    if (pending_buffer_transfers.empty() || size == 0) {
        return;
    }
    const VAddr guest_end =
        size > std::numeric_limits<VAddr>::max() - guest_address
            ? std::numeric_limits<VAddr>::max()
            : guest_address + size;
    const auto overlaps = [guest_address, guest_end](const VAddr begin, const VAddr end) {
        return guest_address < end && begin < guest_end;
    };
    const bool has_conflict = std::ranges::any_of(
        pending_buffer_transfers, [&](const PendingBufferTransfer& transfer) {
            return overlaps(transfer.source_begin, transfer.source_end) ||
                   overlaps(transfer.destination_begin, transfer.destination_end);
        });
    if (has_conflict) {
        FlushPendingBufferTransfers();
    }
}

void Scheduler::FlushPendingGnmDmaTransfers() {
    if (pending_gnm_dma_transfers != 0) {
        FlushPendingBufferTransfers();
    }
}

void Scheduler::FlushPendingBufferTransfers() {
    if (pending_buffer_transfers.empty() || flushing_buffer_transfers) {
        return;
    }
    if (master_semaphore.IsDeviceLost()) {
        pending_buffer_transfers.clear();
        pending_buffer_transfer_requests = 0;
        pending_buffer_transfer_regions = 0;
        pending_gnm_dma_transfers = 0;
        pending_buffer_transfer_origins = 0;
        return;
    }

    flushing_buffer_transfers = true;
    SCOPE_EXIT {
        flushing_buffer_transfers = false;
    };
    const bool has_gnm_dma =
        (pending_buffer_transfer_origins &
         (u32{1} << static_cast<u32>(BufferTransferOrigin::GnmDma))) != 0;
    EndRendering(has_gnm_dma ? EndRenderingReason::BufferCopy
                             : EndRenderingReason::ShaderHleCopy);
    const bool use_buffer_local_barriers =
        VideoCore::IsManagedVulkanTransferCoalescing();
    static constexpr vk::MemoryBarrier ReadBarrier{
        .srcAccessMask = vk::AccessFlagBits::eMemoryRead |
                         vk::AccessFlagBits::eMemoryWrite |
                         vk::AccessFlagBits::eTransferRead |
                         vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead |
                         vk::AccessFlagBits::eTransferWrite,
    };
    static constexpr vk::MemoryBarrier WriteBarrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferRead |
                         vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eMemoryRead |
                         vk::AccessFlagBits::eMemoryWrite,
    };
    static constexpr vk::MemoryBarrier DependencyBarrier{
        .srcAccessMask = vk::AccessFlagBits::eTransferRead |
                         vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = vk::AccessFlagBits::eTransferRead |
                         vk::AccessFlagBits::eTransferWrite,
    };
    struct BufferTransferRange {
        vk::Buffer buffer;
        vk::DeviceSize begin;
        vk::DeviceSize end;
        vk::AccessFlags2 transfer_access;
    };
    boost::container::small_vector<BufferTransferRange, 32> pre_ranges;
    boost::container::small_vector<BufferTransferRange, 32> post_ranges;
    const auto merge_range =
        [](auto& ranges, const vk::Buffer buffer, const vk::DeviceSize begin,
           const vk::DeviceSize end, const vk::AccessFlags2 transfer_access) {
            if (!buffer || begin >= end) {
                return;
            }
            const auto existing = std::ranges::find(ranges, buffer,
                                                     &BufferTransferRange::buffer);
            if (existing == ranges.end()) {
                ranges.emplace_back(BufferTransferRange{
                    .buffer = buffer,
                    .begin = begin,
                    .end = end,
                    .transfer_access = transfer_access,
                });
                return;
            }
            existing->begin = std::min(existing->begin, begin);
            existing->end = std::max(existing->end, end);
            existing->transfer_access |= transfer_access;
        };
    if (use_buffer_local_barriers) {
        for (const PendingBufferTransfer& transfer : pending_buffer_transfers) {
            merge_range(pre_ranges, transfer.source, transfer.source_offset_begin,
                        transfer.source_offset_end,
                        vk::AccessFlagBits2::eTransferRead);
            merge_range(pre_ranges, transfer.destination,
                        transfer.destination_offset_begin,
                        transfer.destination_offset_end,
                        vk::AccessFlagBits2::eTransferWrite);
            merge_range(post_ranges, transfer.destination,
                        transfer.destination_offset_begin,
                        transfer.destination_offset_end,
                        vk::AccessFlagBits2::eTransferWrite);
        }
    }
    boost::container::small_vector<vk::BufferMemoryBarrier2, 32> pre_barriers;
    boost::container::small_vector<vk::BufferMemoryBarrier2, 32> post_barriers;
    if (use_buffer_local_barriers) {
        pre_barriers.reserve(pre_ranges.size());
        for (const BufferTransferRange& range : pre_ranges) {
            pre_barriers.emplace_back(vk::BufferMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryRead |
                                 vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = range.transfer_access,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = range.buffer,
                .offset = range.begin,
                .size = range.end - range.begin,
            });
        }
        post_barriers.reserve(post_ranges.size());
        for (const BufferTransferRange& range : post_ranges) {
            post_barriers.emplace_back(vk::BufferMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
                                 vk::AccessFlagBits2::eMemoryWrite,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = range.buffer,
                .offset = range.begin,
                .size = range.end - range.begin,
            });
        }
    }

    const auto cmdbuf = CommandBuffer();
    if (use_buffer_local_barriers) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
            .pBufferMemoryBarriers = pre_barriers.data(),
        });
    } else {
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eTransfer,
                               vk::DependencyFlagBits::eByRegion, ReadBarrier, {}, {});
    }
    for (const auto& transfer : pending_buffer_transfers) {
        if (transfer.dependency_barrier_before) {
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion,
                                   DependencyBarrier, {}, {});
#ifdef __ANDROID__
            ++executor_buffer_transfer_dependency_barriers;
#endif
        }
        cmdbuf.copyBuffer(transfer.source, transfer.destination, transfer.regions);
    }
    if (use_buffer_local_barriers) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
            .pBufferMemoryBarriers = post_barriers.data(),
        });
    } else {
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                               vk::PipelineStageFlagBits::eAllCommands,
                               vk::DependencyFlagBits::eByRegion, WriteBarrier, {}, {});
    }

    const u64 dependency_barriers = std::ranges::count_if(
        pending_buffer_transfers, [](const PendingBufferTransfer& transfer) {
            return transfer.dependency_barrier_before;
        });
    u64 emitted_copy_regions = 0;
    for (const PendingBufferTransfer& transfer : pending_buffer_transfers) {
        emitted_copy_regions += transfer.regions.size();
    }
    const u64 requested_barriers =
        pending_buffer_transfer_requests * 2 + dependency_barriers;
    const u64 emitted_barriers = 2 + dependency_barriers;
    VideoCore::RecordVulkanTransferFlush(
        pending_buffer_transfer_regions, pending_buffer_transfers.size());
    VideoCore::RecordVulkanBarrierBatch(
        requested_barriers, emitted_barriers);

#ifdef __ANDROID__
    ++executor_buffer_transfer_flushes;
    executor_buffer_transfer_copy_calls += pending_buffer_transfers.size();
    executor_buffer_transfer_copy_regions_emitted += emitted_copy_regions;
    executor_buffer_transfer_barriers_requested += requested_barriers;
    executor_buffer_transfer_barriers_emitted += emitted_barriers;
    if (executor_buffer_transfer_flushes == 1 ||
        (executor_buffer_transfer_flushes & 4095u) == 0u) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_BUFFER_TRANSFER_BATCH] queued=%llu flushes=%llu copyCalls=%llu "
            "regions=requested:%llu,emitted:%llu pendingGroups=%zu maxGroups=%llu "
            "barriers=requested:%llu,emitted:%llu dependencyBarriers=%llu "
            "origin=h:%llu,u:%llu,g:%llu gnmRegions=%llu",
            static_cast<unsigned long long>(executor_buffer_transfers_queued),
            static_cast<unsigned long long>(executor_buffer_transfer_flushes),
            static_cast<unsigned long long>(executor_buffer_transfer_copy_calls),
            static_cast<unsigned long long>(executor_buffer_transfer_regions_queued),
            static_cast<unsigned long long>(
                executor_buffer_transfer_copy_regions_emitted),
            pending_buffer_transfers.size(),
            static_cast<unsigned long long>(executor_buffer_transfer_max_queued),
            static_cast<unsigned long long>(
                executor_buffer_transfer_barriers_requested),
            static_cast<unsigned long long>(
                executor_buffer_transfer_barriers_emitted),
            static_cast<unsigned long long>(
                executor_buffer_transfer_dependency_barriers),
            static_cast<unsigned long long>(executor_hle_buffer_transfers_queued),
            static_cast<unsigned long long>(executor_buffer_uploads_queued),
            static_cast<unsigned long long>(executor_gnm_dma_transfers_queued),
            static_cast<unsigned long long>(executor_gnm_dma_regions_queued));
    }
#endif
    pending_buffer_transfers.clear();
    pending_buffer_transfer_requests = 0;
    pending_buffer_transfer_regions = 0;
    pending_gnm_dma_transfers = 0;
    pending_buffer_transfer_origins = 0;
}

void Scheduler::Flush(SubmitInfo& info) {
    SubmitExecution(info, ExecutorSubmitKind::FlushInfo);
}

void Scheduler::Flush() {
    if (!command_buffer_used && pending_buffer_transfers.empty() &&
        pending_ops_count.load(std::memory_order_acquire) == 0) [[likely]] {
#ifdef __ANDROID__
        if (pending_submit_count.load(std::memory_order_acquire) != 0) {
            PublishPendingSubmitBatch();
        }
#endif
        return;
    }
    SubmitInfo info{};
    SubmitExecution(info, ExecutorSubmitKind::FlushPlain);
}

void Scheduler::Finish(const ExecutorFinishReason reason) {
    if (master_semaphore.IsDeviceLost()) {
        return;
    }
#ifdef __ANDROID__
    const auto executor_finish_begin = std::chrono::steady_clock::now();
    SCOPE_EXIT {
        const u64 elapsed_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - executor_finish_begin)
                .count());
        const std::size_t reason_index = static_cast<std::size_t>(reason);
        ++executor_finish_wait_counts[reason_index];
        executor_finish_wait_ns[reason_index] += elapsed_ns;
        executor_finish_wait_max_ns[reason_index] =
            std::max(executor_finish_wait_max_ns[reason_index], elapsed_ns);
        const u64 ordinal = ++executor_finish_wait_total;
        if ((ordinal & 63u) == 0u) {
            const std::size_t wave_index =
                static_cast<std::size_t>(ExecutorFinishReason::BufferReadbackWave);
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_VK_FINISH_WAIT] total=%llu reason=%u reasonCount=%llu "
                "reasonMs=%.3f reasonMaxMs=%.3f bufferWaveCount=%llu bufferWaveMs=%.3f "
                "bufferWaveMaxMs=%.3f",
                static_cast<unsigned long long>(ordinal), static_cast<unsigned>(reason),
                static_cast<unsigned long long>(executor_finish_wait_counts[reason_index]),
                static_cast<double>(executor_finish_wait_ns[reason_index]) / 1'000'000.0,
                static_cast<double>(executor_finish_wait_max_ns[reason_index]) / 1'000'000.0,
                static_cast<unsigned long long>(executor_finish_wait_counts[wave_index]),
                static_cast<double>(executor_finish_wait_ns[wave_index]) / 1'000'000.0,
                static_cast<double>(executor_finish_wait_max_ns[wave_index]) / 1'000'000.0);
        }
    };
#endif
    if (!command_buffer_used && pending_buffer_transfers.empty() &&
        pending_ops_count.load(std::memory_order_acquire) == 0) [[likely]] {
#ifdef __ANDROID__
        const u64 pending_tick =
            pending_submit_last_tick.load(std::memory_order_acquire);
        if (pending_tick != 0) {
            PublishPendingSubmitBatch();
            master_semaphore.Wait(pending_tick);
            return;
        }
#endif
        const u64 current_tick = CurrentTick();
        if (current_tick > 1) {
            Wait(current_tick - 1);
        }
        return;
    }
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info, ExecutorSubmitKind::Finish, reason);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    if (master_semaphore.IsDeviceLost()) {
        return;
    }
#ifdef __ANDROID__
    if (pending_submit_count.load(std::memory_order_acquire) != 0 &&
        tick <= pending_submit_last_tick.load(std::memory_order_acquire)) {
        PublishPendingSubmitBatch();
        if (master_semaphore.IsDeviceLost()) {
            return;
        }
    }
#endif
    if (tick >= master_semaphore.CurrentTick()) {
        SubmitInfo info{};
        SubmitExecution(info, ExecutorSubmitKind::WaitCurrent);
    }
    master_semaphore.Wait(tick);
}

void Scheduler::WaitRetirement(u64 tick) {
    if (tick == 0 || master_semaphore.IsDeviceLost()) {
        return;
    }
#ifdef __ANDROID__
    if (pending_submit_count.load(std::memory_order_acquire) != 0 &&
        tick <= pending_submit_last_tick.load(std::memory_order_acquire)) {
        PublishPendingSubmitBatch();
        if (master_semaphore.IsDeviceLost()) {
            return;
        }
    }
#endif
    std::unique_lock lk(retirement_pending_ops_mutex);
    retirement_pending_ops_cv.wait(lk, [&] {
        if (master_semaphore.IsDeviceLost()) {
            return true;
        }
        const auto oldest = retirement_outstanding_ops.begin();
        return oldest == retirement_outstanding_ops.end() || oldest->first > tick;
    });
}

void Scheduler::PopPendingOperations(bool force, bool timeline_current) {
    if (pending_ops_count.load(std::memory_order_acquire) == 0) [[likely]] {
        return;
    }

    if (!force &&
        (pending_ops_poll_sequence.fetch_add(1, std::memory_order_relaxed) & 0xffU) != 0) {
        return;
    }

    if (!timeline_current) {
        master_semaphore.Refresh();
    }
    if (master_semaphore.IsDeviceLost()) {
        return;
    }
    while (true) {
        Common::UniqueFunction<void> callback;
        {
            std::scoped_lock lk{pending_ops_mutex};
            if (pending_ops.empty() || !master_semaphore.IsFree(pending_ops.front().gpu_tick)) {
                break;
            }
            callback = std::move(pending_ops.front().callback);
            pending_ops.pop();
            const u64 previous_count =
                pending_ops_count.fetch_sub(1, std::memory_order_release);
            ASSERT_MSG(previous_count != 0,
                       "pending_ops_count underflow while draining deferred operation");
        }
        callback();
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    Check(current_cmdbuf.begin(begin_info));
    command_buffer_recording = true;
    command_buffer_used = false;

    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

#ifdef __ANDROID__
bool Scheduler::QueueSubmitLocked(
    const vk::ArrayProxy<const vk::SubmitInfo> submits, const vk::Fence fence,
    const std::span<const u64> logical_ticks, const u64 command_buffer_count,
    const bool forced_boundary) {
    ASSERT(!logical_ticks.empty());

    const bool executor_diagnostics_enabled = !Config::managedGpuFastPath();
    if (executor_diagnostics_enabled) {
        for (const u64 tick : logical_ticks) {
            ExecutorVkJournalSubmit(this, tick, false, vk::Result::eSuccess);
        }
    }
    const auto executor_submit_begin =
        executor_diagnostics_enabled ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};

    const vk::Result submit_result =
        instance.GetGraphicsQueue().submit(submits, fence);
    VideoCore::RecordVulkanQueueSubmit(logical_ticks.size(), command_buffer_count,
                                       forced_boundary);

    ++executor_submit_batches;
    executor_submit_logical_packets += logical_ticks.size();
    executor_submit_command_buffers += command_buffer_count;
    executor_submit_calls_saved +=
        logical_ticks.size() > 1 ? logical_ticks.size() - 1 : 0;
    executor_submit_forced_boundaries += forced_boundary ? 1 : 0;
    executor_submit_max_batch =
        std::max(executor_submit_max_batch,
                 static_cast<u64>(logical_ticks.size()));
    if (executor_submit_batches == 1 ||
        (executor_submit_batches & 255u) == 0u) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_VK_SUBMIT_COALESCING] physical=%llu logical=%llu "
            "commandBuffers=%llu callsSaved=%llu forced=%llu timer=%llu "
            "maxBatch=%llu currentBatch=%zu currentSubmitInfos=%u",
            static_cast<unsigned long long>(executor_submit_batches),
            static_cast<unsigned long long>(executor_submit_logical_packets),
            static_cast<unsigned long long>(executor_submit_command_buffers),
            static_cast<unsigned long long>(executor_submit_calls_saved),
            static_cast<unsigned long long>(executor_submit_forced_boundaries),
            static_cast<unsigned long long>(executor_submit_timer_flushes),
            static_cast<unsigned long long>(executor_submit_max_batch),
            logical_ticks.size(), static_cast<u32>(submits.size()));
    }

    if (executor_diagnostics_enabled) {
        const u64 executor_submit_ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - executor_submit_begin)
                .count());
        static u64 executor_submit_samples{};
        static u64 executor_submit_total_ns{};
        static u64 executor_submit_max_ns{};
        ++executor_submit_samples;
        executor_submit_total_ns += executor_submit_ns;
        executor_submit_max_ns =
            std::max(executor_submit_max_ns, executor_submit_ns);
        if ((executor_submit_samples & 1023u) == 0u) {
            __android_log_print(
                ANDROID_LOG_INFO, "LSX4Native",
                "[EXECUTOR_VK_SUBMIT_LATENCY] samples=1024 totalMs=%.3f "
                "avgUs=%.3f maxMs=%.3f",
                static_cast<double>(executor_submit_total_ns) / 1'000'000.0,
                static_cast<double>(executor_submit_total_ns) / 1'024'000.0,
                static_cast<double>(executor_submit_max_ns) / 1'000'000.0);
            executor_submit_total_ns = 0;
            executor_submit_max_ns = 0;
        }
        for (const u64 tick : logical_ticks) {
            ExecutorVkJournalSubmit(this, tick, true, submit_result);
        }
    }

    const u64 final_tick = logical_ticks.back();
    if (submit_result != vk::Result::eSuccess) [[unlikely]] {
        if (submit_result == vk::Result::eErrorDeviceLost) {
            instance.DumpDeviceFaultInfo("queue_submit", final_tick);
        }
        ExecutorVkJournalDump("queue_submit", final_tick);
        MarkTerminal(submit_result, "queue_submit");
        if (submit_result != vk::Result::eErrorDeviceLost) {
            LOG_ERROR(Render_Vulkan,
                      "Failed to submit Vulkan command buffer batch: {}",
                      vk::to_string(submit_result));
        }
        return false;
    }

    static const bool serialize_gnm_submits =
        std::getenv("EXECUTOR_ANDROID_SERIALIZE_GNM_SUBMITS") != nullptr;
    if (serialize_gnm_submits) {
        master_semaphore.Wait(final_tick);
        if (master_semaphore.IsDeviceLost()) {
            return false;
        }
    }

    master_semaphore.Refresh();
    if (master_semaphore.IsDeviceLost()) {
        return false;
    }
    return true;
}

bool Scheduler::SubmitPendingBatchLocked(
    const vk::SubmitInfo* trailing_submit, const vk::Fence trailing_fence,
    const u64 trailing_tick, const bool forced_boundary) {
    const bool has_pending = !pending_submit_command_buffers.empty();
    if (!has_pending && trailing_submit == nullptr) {
        return true;
    }
    if (master_semaphore.IsDeviceLost()) {
        pending_submit_command_buffers.clear();
        pending_submit_ticks.clear();
        pending_submit_count.store(0, std::memory_order_release);
        pending_submit_last_tick.store(0, std::memory_order_release);
        pending_submit_deadline = {};
        return false;
    }

    if (!has_pending) {
        const std::array<u64, 1> ticks{trailing_tick};
        return QueueSubmitLocked(
            vk::ArrayProxy<const vk::SubmitInfo>{1, trailing_submit},
            trailing_fence, ticks, 1, forced_boundary);
    }

    const vk::Semaphore timeline = master_semaphore.Handle();
    const std::array<vk::Semaphore, 1> signal_semas{timeline};
    const std::array<u64, 1> signal_ticks{pending_submit_ticks.back()};
    const vk::TimelineSemaphoreSubmitInfo timeline_info = {
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = signal_ticks.data(),
    };
    const vk::SubmitInfo pending_submit = {
        .pNext = &timeline_info,
        .commandBufferCount =
            static_cast<u32>(pending_submit_command_buffers.size()),
        .pCommandBuffers = pending_submit_command_buffers.data(),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = signal_semas.data(),
    };

    boost::container::small_vector<u64, 9> logical_ticks{
        pending_submit_ticks.begin(), pending_submit_ticks.end()};
    const u64 pending_command_buffers =
        pending_submit_command_buffers.size();
    bool success = false;
    if (trailing_submit != nullptr) {
        logical_ticks.push_back(trailing_tick);
        const std::array<vk::SubmitInfo, 2> submits{
            pending_submit, *trailing_submit};
        success = QueueSubmitLocked(submits, trailing_fence, logical_ticks,
                                    pending_command_buffers + 1,
                                    forced_boundary);
    } else {
        success = QueueSubmitLocked(
            vk::ArrayProxy<const vk::SubmitInfo>{1, &pending_submit}, {},
            logical_ticks, pending_command_buffers, forced_boundary);
    }

    pending_submit_command_buffers.clear();
    pending_submit_ticks.clear();
    pending_submit_count.store(0, std::memory_order_release);
    pending_submit_last_tick.store(0, std::memory_order_release);
    pending_submit_deadline = {};
    pending_submit_cv.notify_all();
    return success;
}

void Scheduler::PublishPendingSubmitBatch() {
    std::scoped_lock lk{submit_mutex};
    if (SubmitPendingBatchLocked(nullptr, {}, 0, true)) {
        PopPendingOperations(true, true);
    }
}

void Scheduler::PendingSubmitThread(const std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:VkSubmitCoalescer");
    std::unique_lock lk{submit_mutex};
    while (!stoken.stop_requested() && !master_semaphore.IsDeviceLost()) {
        if (pending_submit_command_buffers.empty()) {
            pending_submit_cv.wait(
                lk, stoken,
                [this] {
                    return master_semaphore.IsDeviceLost() ||
                           !pending_submit_command_buffers.empty();
                });
            continue;
        }
        const auto deadline = pending_submit_deadline;
        if (std::chrono::steady_clock::now() < deadline) {
            pending_submit_cv.wait_until(lk, deadline);
            continue;
        }
        ++executor_submit_timer_flushes;
        // Deferred-operation callbacks historically run on the scheduler's
        // caller thread. The timer only publishes GPU work; the next owner
        // interaction drains callbacks after observing timeline retirement.
        SubmitPendingBatchLocked();
    }
}
#endif

void Scheduler::SubmitExecution(SubmitInfo& info, ExecutorSubmitKind kind,
                                const ExecutorFinishReason finish_reason) {
    std::scoped_lock lk{submit_mutex};
    if (master_semaphore.IsDeviceLost()) {
        return;
    }
    FlushPendingBufferTransfers();
#ifdef __ANDROID__
    const bool executor_diagnostics_enabled = !Config::managedGpuFastPath();
    if (executor_diagnostics_enabled) {
        ++executor_submit_kind_counts[static_cast<std::size_t>(kind)];
        if (kind == ExecutorSubmitKind::Finish) {
            ++executor_finish_reason_counts[static_cast<std::size_t>(finish_reason)];
        }
    }
    const u64 executor_submit_kind_ordinal =
        executor_diagnostics_enabled ? ++executor_submit_kind_total : 0;
    if (executor_diagnostics_enabled && (executor_submit_kind_ordinal & 255u) == 0u) {
        __android_log_print(
            ANDROID_LOG_INFO, "LSX4Native",
            "[EXECUTOR_VK_SUBMIT_REASONS] scheduler=%p total=%llu flushInfo=%llu "
            "flushPlain=%llu finish=%llu waitCurrent=%llu finishUnknown=%llu "
            "finishBufWave=%llu finishBufDirect=%llu finishBufProbe=%llu "
            "finishTexRead=%llu finishTexProbe=%llu finishRasterizer=%llu "
            "finishPresenter=%llu",
            this, static_cast<unsigned long long>(executor_submit_kind_ordinal),
            static_cast<unsigned long long>(
                executor_submit_kind_counts[static_cast<std::size_t>(
                    ExecutorSubmitKind::FlushInfo)]),
            static_cast<unsigned long long>(
                executor_submit_kind_counts[static_cast<std::size_t>(
                    ExecutorSubmitKind::FlushPlain)]),
            static_cast<unsigned long long>(
                executor_submit_kind_counts[static_cast<std::size_t>(
                    ExecutorSubmitKind::Finish)]),
            static_cast<unsigned long long>(
                executor_submit_kind_counts[static_cast<std::size_t>(
                    ExecutorSubmitKind::WaitCurrent)]),
            static_cast<unsigned long long>(
                executor_finish_reason_counts[static_cast<std::size_t>(
                    ExecutorFinishReason::Unknown)]),
            static_cast<unsigned long long>(
                executor_finish_reason_counts[static_cast<std::size_t>(
                    ExecutorFinishReason::BufferReadbackWave)]),
            static_cast<unsigned long long>(
                executor_finish_reason_counts[static_cast<std::size_t>(
                    ExecutorFinishReason::BufferReadbackDirect)]),
            static_cast<unsigned long long>(
                executor_finish_reason_counts[static_cast<std::size_t>(
                    ExecutorFinishReason::BufferProbe)]),
            static_cast<unsigned long long>(
                executor_finish_reason_counts[static_cast<std::size_t>(
                    ExecutorFinishReason::TextureReadback)]),
            static_cast<unsigned long long>(
                executor_finish_reason_counts[static_cast<std::size_t>(
                    ExecutorFinishReason::TextureProbe)]),
            static_cast<unsigned long long>(
                executor_finish_reason_counts[static_cast<std::size_t>(
                    ExecutorFinishReason::Rasterizer)]),
            static_cast<unsigned long long>(
                executor_finish_reason_counts[static_cast<std::size_t>(
                    ExecutorFinishReason::PresenterShutdown)]));
    }
#endif
#ifdef __ANDROID__
    const bool managed_submit_coalescing =
        VideoCore::IsManagedVulkanSubmitCoalescing();
    static const bool serialize_gnm_submits =
        std::getenv("EXECUTOR_ANDROID_SERIALIZE_GNM_SUBMITS") != nullptr;
    const bool has_external_sync =
        info.num_wait_semas != 0 || info.num_signal_semas != 0 ||
        static_cast<VkFence>(info.fence) != VK_NULL_HANDLE;
    const bool can_defer_internal_submit =
        managed_submit_coalescing && !serialize_gnm_submits &&
        kind == ExecutorSubmitKind::FlushPlain && !has_external_sync;
#endif
    const u64 signal_value = master_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering(kind == ExecutorSubmitKind::FlushInfo ||
                         kind == ExecutorSubmitKind::FlushPlain
                     ? EndRenderingReason::Flush
                     : EndRenderingReason::Submit);
    Check(current_cmdbuf.end());
    command_buffer_recording = false;

    const vk::Semaphore timeline = master_semaphore.Handle();
#ifdef __ANDROID__
    if (signal_value > 1 && !managed_submit_coalescing) {
        const u64 preceding_signal = signal_value - 1;
        bool merged_wait = false;
        for (u32 index = 0; index < info.num_wait_semas; ++index) {
            if (info.wait_semas[index] != timeline) {
                continue;
            }
            info.wait_ticks[index] = std::max(info.wait_ticks[index], preceding_signal);
            info.wait_stage_masks[index] |= vk::PipelineStageFlagBits::eAllCommands;
            merged_wait = true;
            break;
        }
        if (!merged_wait) {
            info.AddWait(timeline, preceding_signal);
        }
    }
#endif
    info.AddSignal(timeline, signal_value);

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = info.wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &current_cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
#ifdef __ANDROID__
    if (can_defer_internal_submit) {
        constexpr std::size_t MaxCoalescedCommandBuffers = 16;
        // Keep collection and vkQueueSubmit on this scheduler thread.  The
        // former timer thread could submit one command buffer while Adreno
        // was recording the next buffer from the same pool, which triggered
        // a driver race in texture detiling and normally expired at batch=1.
        // A synchronous frame/boundary batch has no concurrent queue call;
        // explicit waits, external sync and presentation publish it through
        // the existing forced-boundary path.
        pending_submit_command_buffers.push_back(current_cmdbuf);
        pending_submit_ticks.push_back(signal_value);
        pending_submit_last_tick.store(signal_value,
                                       std::memory_order_release);
        pending_submit_count.store(
            static_cast<u32>(pending_submit_command_buffers.size()),
            std::memory_order_release);

        bool success = true;
        if (pending_submit_command_buffers.size() >=
            MaxCoalescedCommandBuffers) {
            success = SubmitPendingBatchLocked();
        }
        if (!success || master_semaphore.IsDeviceLost()) {
            return;
        }
        AllocateWorkerCommandBuffers();
        if (pending_submit_command_buffers.empty()) {
            PopPendingOperations(true, true);
        }
        return;
    }

    const std::array<u64, 1> logical_ticks{signal_value};
    const bool success =
        pending_submit_command_buffers.empty()
            ? QueueSubmitLocked(
                  vk::ArrayProxy<const vk::SubmitInfo>{1, &submit_info},
                  info.fence, logical_ticks, 1, false)
            : SubmitPendingBatchLocked(&submit_info, info.fence, signal_value,
                                       true);
    if (!success || master_semaphore.IsDeviceLost()) {
        return;
    }
    AllocateWorkerCommandBuffers();
    PopPendingOperations(true, true);
#else
    const vk::Result submit_result =
        instance.GetGraphicsQueue().submit(submit_info, info.fence);
    VideoCore::RecordVulkanQueueSubmit();
    if (submit_result != vk::Result::eSuccess) [[unlikely]] {
        MarkTerminal(submit_result, "queue_submit");
        if (submit_result != vk::Result::eErrorDeviceLost) {
            LOG_ERROR(Render_Vulkan, "Failed to submit Vulkan command buffer: {}",
                      vk::to_string(submit_result));
        }
        return;
    }
    master_semaphore.Refresh();
    if (master_semaphore.IsDeviceLost()) {
        return;
    }
    AllocateWorkerCommandBuffers();

    PopPendingOperations(true, true);
#endif
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(
                lk, stoken,
                [this] { return master_semaphore.IsDeviceLost() || !priority_pending_ops.empty(); });
            if (stoken.stop_requested() || master_semaphore.IsDeviceLost()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        master_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested() || master_semaphore.IsDeviceLost()) {
            break;
        }

        op.callback();
    }
}

void Scheduler::RetirementPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedRetirementOpsRunner");

    std::vector<PendingOp> batch;
    batch.reserve(256);
    while (!stoken.stop_requested()) {
        batch.clear();
        u64 gpu_tick{};
        {
            std::unique_lock lk(retirement_pending_ops_mutex);
            retirement_pending_ops_cv.wait(
                lk, stoken,
                [this] { return master_semaphore.IsDeviceLost() || !retirement_pending_ops.empty(); });
            if (stoken.stop_requested() || master_semaphore.IsDeviceLost()) {
                break;
            }

            auto oldest = retirement_pending_ops.begin();
            gpu_tick = oldest->first;
            auto& tick_ops = oldest->second;
            while (!tick_ops.empty()) {
                batch.emplace_back(std::move(tick_ops.front()));
                tick_ops.pop();
            }
            retirement_pending_ops.erase(oldest);
        }

        master_semaphore.Wait(gpu_tick);
        if (stoken.stop_requested() || master_semaphore.IsDeviceLost()) {
            break;
        }

        for (auto& op : batch) {
            if (stoken.stop_requested() || master_semaphore.IsDeviceLost()) {
                break;
            }
            op.callback();
        }
        {
            std::scoped_lock lk{retirement_pending_ops_mutex};
            auto outstanding = retirement_outstanding_ops.find(gpu_tick);
            ASSERT_MSG(outstanding != retirement_outstanding_ops.end() &&
                           outstanding->second >= batch.size(),
                       "Retirement callback accounting underflow at tick {}", gpu_tick);
            const u64 previous_outstanding =
                retirement_outstanding_count.fetch_sub(batch.size(),
                                                       std::memory_order_acq_rel);
            ASSERT_MSG(previous_outstanding >= batch.size(),
                       "Global retirement callback accounting underflow at tick {}",
                       gpu_tick);
            outstanding->second -= batch.size();
            if (outstanding->second == 0) {
                retirement_outstanding_ops.erase(outstanding);
            }
            u64 completed = retirement_completed_tick.load(std::memory_order_relaxed);
            while (completed < gpu_tick &&
                   !retirement_completed_tick.compare_exchange_weak(
                       completed, gpu_tick, std::memory_order_release,
                       std::memory_order_relaxed)) {
            }
        }
        retirement_pending_ops_cv.notify_all();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
#ifdef __ANDROID__
    dirty_state.rasterizer_discard_enable = false;
#else
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
#endif
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
    }
    if (dirty_state.fragment_shading_rate) {
        dirty_state.fragment_shading_rate = false;
        if (instance.IsCoarseFragmentShadingSupported()) {
            const std::array combiners = {
                vk::FragmentShadingRateCombinerOpKHR::eKeep,
                vk::FragmentShadingRateCombinerOpKHR::eKeep,
            };
            cmdbuf.setFragmentShadingRateKHR(fragment_shading_rate,
                                             combiners.data());
        }
    }
}

}
