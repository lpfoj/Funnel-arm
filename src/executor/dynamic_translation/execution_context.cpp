// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/execution_context.h"

#include "executor/dynamic_translation/live_state_port.h"

namespace Lsx4::Translation {
namespace {

thread_local CpuFrame* active_frame = nullptr;
thread_local CpuFrame replacement_frame{};
thread_local bool replacement_waiting = false;
thread_local std::uint64_t completion_value = 0;
thread_local bool completion_waiting = false;

} // namespace

ExecutionBinding::ExecutionBinding(CpuFrame& frame) noexcept
    : previous_{active_frame}, previous_storage_{ExchangeLiveStateStorage(&frame)} {
    active_frame = &frame;
}

ExecutionBinding::~ExecutionBinding() {
    active_frame = previous_;
    (void)ExchangeLiveStateStorage(previous_storage_);
}

CpuFrame* CurrentExecutionFrame() noexcept {
    return active_frame;
}

bool SnapshotExecutionFrame(CpuFrame& destination) noexcept {
    if (!active_frame) {
        return false;
    }
    destination = *active_frame;
    return true;
}

void PostExecutionReplacement(const CpuFrame& replacement) noexcept {
    replacement_frame = replacement;
    replacement_waiting = true;
}

bool TakeExecutionReplacement(CpuFrame& destination) noexcept {
    if (!replacement_waiting) {
        return false;
    }
    destination = replacement_frame;
    replacement_waiting = false;
    return true;
}

void PostExecutionCompletion(const std::uint64_t result) noexcept {
    completion_value = result;
    completion_waiting = true;
}

std::optional<std::uint64_t> TakeExecutionCompletion() noexcept {
    if (!completion_waiting) {
        return std::nullopt;
    }
    completion_waiting = false;
    return completion_value;
}

void ClearExecutionMessages() noexcept {
    replacement_waiting = false;
    completion_waiting = false;
}

} // namespace Lsx4::Translation
