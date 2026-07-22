// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/execution_driver.h"

#include "executor/dynamic_translation/native_compiler.h"
#include "executor/dynamic_translation/scalar_semantics.h"
#include "executor/dynamic_translation/vector_semantics.h"
#include "executor/dynamic_translation/x87_semantics.h"

namespace Lsx4::Translation {
namespace {

struct StepDecision {
    DriverStop stop{DriverStop::UnsupportedInstruction};
    std::uint64_t next{};
    std::uint64_t fault{};
    bool proceed{};
    bool complete{};
};

StepDecision ApplyBoundary(const Instruction& instruction, CpuFrame& frame,
                           const ExecutionPorts& ports) noexcept {
    if (!ports.boundary) {
        return {.proceed = true};
    }
    const BoundaryDecision decision =
        ports.boundary(instruction, frame, ports.boundary_context);
    switch (decision.action) {
    case BoundaryAction::NotHandled:
        return {.proceed = true};
    case BoundaryAction::Continue:
        return {.next = instruction.address + instruction.length, .proceed = true};
    case BoundaryAction::Transfer:
        return {.next = decision.next_address, .proceed = true};
    case BoundaryAction::Complete:
        return {.next = decision.result, .complete = true};
    case BoundaryAction::Fault:
        return {.stop = DriverStop::BoundaryFault,
                .fault = decision.fault_address};
    }
    return {};
}

StepDecision ExecuteSemantic(const Instruction& instruction, CpuFrame& frame,
                             const ExecutionPorts& ports,
                             const ExecutionLimits& limits) noexcept {
    const StepDecision boundary = ApplyBoundary(instruction, frame, ports);
    if (!boundary.proceed || boundary.complete || boundary.next != 0) {
        return boundary;
    }

    const SemanticResult scalar =
        ExecuteScalarInstruction(instruction, frame, ports.memory);
    switch (scalar.stop) {
    case SemanticStop::Continue:
    case SemanticStop::Transfer:
        return {.next = scalar.next_address, .proceed = true};
    case SemanticStop::Return:
        return scalar.next_address == limits.return_sentinel
                   ? StepDecision{.next = frame.integer[0], .complete = true}
                   : StepDecision{.next = scalar.next_address, .proceed = true};
    case SemanticStop::MemoryFault:
        return {.stop = DriverStop::MemoryFault, .fault = scalar.fault_address};
    case SemanticStop::Retry:
        return {.stop = DriverStop::RetryLimit};
    case SemanticStop::Unsupported:
        break;
    }

    const VectorSemanticResult vector =
        ExecuteVectorInstruction(instruction, frame, ports.memory);
    if (vector.stop == VectorStop::Continue) {
        return {.next = vector.next_address, .proceed = true};
    }
    if (vector.stop == VectorStop::MemoryFault) {
        return {.stop = DriverStop::MemoryFault, .fault = vector.fault_address};
    }

    const X87SemanticResult x87 = ExecuteX87Instruction(instruction, frame, ports.memory);
    if (x87.stop == X87Stop::Continue) {
        return {.next = x87.next_address, .proceed = true};
    }
    if (x87.stop == X87Stop::MemoryFault) {
        return {.stop = DriverStop::MemoryFault, .fault = x87.fault_address};
    }
    return {.stop = DriverStop::UnsupportedInstruction};
}

} // namespace

ExecutionDriver::ExecutionDriver(const std::size_t arena_slab_bytes)
    : service_{arena_slab_bytes} {}

DriverResult ExecutionDriver::Run(const std::uint64_t first_address, CpuFrame& frame,
                                  const ExecutionPorts& ports,
                                  const ExecutionLimits& limits) {
    if (!ports.fetch || !ports.read_code || limits.maximum_instructions == 0 ||
        limits.region_instruction_limit == 0) {
        return {DriverStop::GuestReadFailure, 0, first_address};
    }
    ClearExecutionMessages();
    ExecutionBinding binding{frame};
    std::uint64_t current = first_address;
    std::uint64_t executed = 0;
    while (executed < limits.maximum_instructions) {
        if (const auto completion = TakeExecutionCompletion()) {
            return {DriverStop::Complete, *completion, current, 0, executed};
        }
        CpuFrame replacement{};
        if (TakeExecutionReplacement(replacement)) {
            frame = replacement;
            current = frame.resume_address;
        }
        const AcquireResult acquired = service_.Acquire(
            current, ports.fetch, ports.fetch_context, ports.read_code,
            limits.region_instruction_limit);
        if (acquired.status == AcquireStatus::Ready && acquired.region) {
            const RegionPayload& payload = acquired.region->Payload();
            const auto entry = reinterpret_cast<NativeRegionEntry>(payload.entry_address);
            current = entry(&frame);
            executed += payload.plan.sequence.size();
            frame.resume_address = current;
            continue;
        }
        if (acquired.status != AcquireStatus::NeedsSemanticFallback) {
            const DriverStop stop =
                acquired.status == AcquireStatus::GuestReadFailure ||
                        acquired.status == AcquireStatus::WitnessFailure
                    ? DriverStop::GuestReadFailure
                    : DriverStop::CompileFailure;
            return {stop, frame.integer[0], current, 0, executed};
        }
        for (const PlannedInstruction& planned : acquired.fallback_plan.sequence) {
            const StepDecision step =
                ExecuteSemantic(planned.instruction, frame, ports, limits);
            ++executed;
            if (step.complete) {
                return {DriverStop::Complete, step.next, planned.instruction.address, 0,
                        executed};
            }
            if (!step.proceed) {
                return {step.stop, frame.integer[0], planned.instruction.address,
                        step.fault, executed};
            }
            current = step.next;
            frame.resume_address = current;
            if (planned.boundary != FlowBoundary::None) {
                break;
            }
        }
    }
    return {DriverStop::InstructionLimit, frame.integer[0], current, 0, executed};
}

TranslationService& ExecutionDriver::Service() noexcept {
    return service_;
}

} // namespace Lsx4::Translation
