// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/translation_service.h"

#include "executor/dynamic_translation/native_compiler.h"

#include <algorithm>
#include <limits>

namespace Lsx4::Translation {

TranslationService::TranslationService(const std::size_t arena_slab_bytes)
    : arena_{arena_slab_bytes} {}

TranslationService::~TranslationService() {
    Clear();
}

std::size_t TranslationService::AdmissionShard(std::uint64_t address) noexcept {
    address ^= address >> 27;
    address *= 0x3c79ac492ba7b653ull;
    address ^= address >> 33;
    return static_cast<std::size_t>(address % AdmissionShardCount);
}

bool TranslationService::RegionIsCurrent(const PublishedRegion& region,
                                         const ReadGuestBytes& reader) const {
    return mutations_.Validate(region.Payload().source_identity, reader);
}

void TranslationService::Retire(
    const std::shared_ptr<const PublishedRegion>& region) noexcept {
    if (!region) {
        return;
    }
    for (const auto& portal : region->Payload().outgoing_edges) {
        if (portal) {
            edges_.Detach(*portal);
        }
    }
    edges_.Revoke(region->GuestAddress());
}

std::vector<std::shared_ptr<ExitPortal>> TranslationService::BuildEdges(
    const RegionPlan& plan) {
    std::vector<std::shared_ptr<ExitPortal>> result{};
    if (plan.sequence.empty()) {
        return result;
    }
    const PlannedInstruction& tail = plan.sequence.back();
    std::uint64_t direct_target = 0;
    if ((tail.boundary == FlowBoundary::Jump ||
         tail.boundary == FlowBoundary::ConditionalJump ||
         tail.boundary == FlowBoundary::Call) &&
        ResolveRelativeTarget(tail.instruction, direct_target)) {
        auto portal = std::make_shared<ExitPortal>(direct_target, EdgePolicy::Chainable);
        edges_.Attach(*portal);
        result.push_back(std::move(portal));
    }
    if (tail.boundary == FlowBoundary::ConditionalJump) {
        auto fallthrough =
            std::make_shared<ExitPortal>(plan.continuation, EdgePolicy::Chainable);
        edges_.Attach(*fallthrough);
        result.push_back(std::move(fallthrough));
    }
    return result;
}

AcquireResult TranslationService::Acquire(const std::uint64_t guest_address,
                                          const InstructionFetcher fetch,
                                          void* const fetch_context,
                                          const ReadGuestBytes& reader,
                                          const std::size_t instruction_limit) {
    if (!fetch || !reader || instruction_limit == 0) {
        return {AcquireStatus::GuestReadFailure};
    }
    std::shared_lock publication{publication_gate_};
    if (auto existing = regions_.Find(guest_address)) {
        if (RegionIsCurrent(*existing, reader)) {
            lookup_hits_.fetch_add(1, std::memory_order_relaxed);
            return {AcquireStatus::Ready, std::move(existing)};
        }
    }

    std::lock_guard admission{admission_[AdmissionShard(guest_address)]};
    if (auto existing = regions_.Find(guest_address)) {
        if (RegionIsCurrent(*existing, reader)) {
            lookup_hits_.fetch_add(1, std::memory_order_relaxed);
            return {AcquireStatus::Ready, std::move(existing)};
        }
        Retire(regions_.Remove(guest_address));
        invalidated_regions_.fetch_add(1, std::memory_order_relaxed);
    }

    RegionPlan plan =
        PlanRegion(guest_address, instruction_limit, fetch, fetch_context);
    if (!plan.IsExecutable() || plan.continuation < plan.first_address) {
        return {plan.stop == RegionStop::ReadFailure ? AcquireStatus::GuestReadFailure
                                                     : AcquireStatus::InvalidRegion,
                {}, std::move(plan)};
    }
    const std::uint64_t extent = plan.continuation - plan.first_address;
    if (extent == 0 || extent > std::numeric_limits<std::size_t>::max()) {
        return {AcquireStatus::InvalidRegion, {}, std::move(plan)};
    }
    auto witness = mutations_.Capture(plan.first_address,
                                      static_cast<std::size_t>(extent), reader);
    if (!witness) {
        return {AcquireStatus::WitnessFailure, {}, std::move(plan)};
    }

    TranslationProgram program = BuildScalarProgram(plan);
    if (!program.semantic_steps.empty()) {
        semantic_regions_.fetch_add(1, std::memory_order_relaxed);
        return {AcquireStatus::NeedsSemanticFallback, {}, std::move(plan)};
    }
    NativeArtifact artifact = CompileScalarIr(program, arena_);
    if (!artifact.Succeeded()) {
        compile_failures_.fetch_add(1, std::memory_order_relaxed);
        return {AcquireStatus::NativeCompileFailure, {}, std::move(plan)};
    }

    RegionPayload payload{};
    payload.plan = std::move(plan);
    payload.source_identity = std::move(*witness);
    payload.entry_address = reinterpret_cast<std::uintptr_t>(artifact.entry);
    payload.code_address = reinterpret_cast<std::uintptr_t>(
        artifact.reservation.executable.data());
    payload.code_bytes = artifact.reservation.executable.size();
    payload.grade = ExecutionGrade::Native;
    payload.outgoing_edges = BuildEdges(payload.plan);
    auto published = regions_.Install(std::move(payload));
    if (!published) {
        compile_failures_.fetch_add(1, std::memory_order_relaxed);
        return {AcquireStatus::NativeCompileFailure};
    }
    const RegionPayload& installed = published->Payload();
    edges_.Announce(published->GuestAddress(), installed.entry_address,
                    installed.source_identity.fingerprint,
                    runtime_epoch_.load(std::memory_order_acquire));
    compiled_regions_.fetch_add(1, std::memory_order_relaxed);
    return {AcquireStatus::Ready, std::move(published)};
}

void TranslationService::NotifyMutation(const std::uint64_t guest_address,
                                        const std::size_t byte_count) {
    if (byte_count == 0) {
        return;
    }
    std::unique_lock publication{publication_gate_};
    mutations_.MarkChanged(guest_address, byte_count);
    runtime_epoch_.fetch_add(1, std::memory_order_acq_rel);
    edges_.RevokeAll();
    const std::uint64_t end =
        byte_count > std::numeric_limits<std::uint64_t>::max() - guest_address
            ? std::numeric_limits<std::uint64_t>::max()
            : guest_address + byte_count;
    for (const auto& region : regions_.Snapshot()) {
        const CodeWitness& witness = region->Payload().source_identity;
        const std::uint64_t witness_end =
            witness.byte_count >
                    std::numeric_limits<std::uint64_t>::max() - witness.guest_address
                ? std::numeric_limits<std::uint64_t>::max()
                : witness.guest_address + witness.byte_count;
        if (guest_address < witness_end && witness.guest_address < end) {
            Retire(regions_.Remove(region->GuestAddress()));
            invalidated_regions_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void TranslationService::Clear() noexcept {
    std::unique_lock publication{publication_gate_};
    for (const auto& region : regions_.Snapshot()) {
        Retire(region);
    }
    regions_.Clear();
    edges_.RevokeAll();
    runtime_epoch_.fetch_add(1, std::memory_order_release);
}

TranslationServiceCounters TranslationService::Counters() const noexcept {
    return {lookup_hits_.load(std::memory_order_relaxed),
            compiled_regions_.load(std::memory_order_relaxed),
            semantic_regions_.load(std::memory_order_relaxed),
            invalidated_regions_.load(std::memory_order_relaxed),
            compile_failures_.load(std::memory_order_relaxed)};
}

std::uint64_t TranslationService::RuntimeEpoch() const noexcept {
    return runtime_epoch_.load(std::memory_order_acquire);
}

std::size_t TranslationService::RegionCount() const noexcept {
    return regions_.ActiveCount();
}

std::size_t TranslationService::NativeBytes() const noexcept {
    return arena_.CommittedBytes();
}

} // namespace Lsx4::Translation
