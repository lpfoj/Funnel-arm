// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#include "executor/dynamic_translation/live_state_port.h"

#include <cstddef>
#include <cstring>

namespace Lsx4::Translation {
namespace {

struct SealedStorage {
    std::uintptr_t address{};
    std::uintptr_t proof{};
};

constexpr std::uintptr_t StorageKey =
    sizeof(std::uintptr_t) == 8 ? 0x52d6a10cf4b9873eull : 0xf4b9873eu;

thread_local SealedStorage active_storage{};
thread_local SealedStorage continuation_storage{};

void Seal(SealedStorage& slot, const void* storage) noexcept {
    slot.proof = 0;
    slot.address = reinterpret_cast<std::uintptr_t>(storage);
    slot.proof = slot.address ^ StorageKey;
}

void* Open(const SealedStorage& slot) noexcept {
    if ((slot.address ^ StorageKey) != slot.proof) {
        return nullptr;
    }
    return reinterpret_cast<void*>(slot.address);
}

std::byte* Field(void* storage, const std::size_t offset) noexcept {
    return static_cast<std::byte*>(storage) + offset;
}

const std::byte* Field(const void* storage, const std::size_t offset) noexcept {
    return static_cast<const std::byte*>(storage) + offset;
}

} // namespace

std::uint64_t LiveStateView::Read(const IntegerRegister id) const noexcept {
    std::uint64_t value{};
    if (storage_ != nullptr && RegisterIndex(id) < 16) {
        const std::size_t offset = offsetof(CpuFrame, integer) +
                                   RegisterIndex(id) * sizeof(value);
        std::memcpy(&value, Field(storage_, offset), sizeof(value));
    }
    return value;
}

void LiveStateView::Write(const IntegerRegister id,
                          const std::uint64_t value) const noexcept {
    if (storage_ == nullptr || RegisterIndex(id) >= 16) {
        return;
    }
    const std::size_t offset = offsetof(CpuFrame, integer) +
                               RegisterIndex(id) * sizeof(value);
    std::memcpy(Field(storage_, offset), &value, sizeof(value));
}

std::uint64_t LiveStateView::ResumeAddress() const noexcept {
    std::uint64_t value{};
    if (storage_ != nullptr) {
        std::memcpy(&value, Field(storage_, offsetof(CpuFrame, resume_address)),
                    sizeof(value));
    }
    return value;
}

void LiveStateView::SetResumeAddress(const std::uint64_t address) const noexcept {
    if (storage_ != nullptr) {
        std::memcpy(Field(storage_, offsetof(CpuFrame, resume_address)), &address,
                    sizeof(address));
    }
}

bool LiveStateView::CopyTo(CpuFrame& destination) const noexcept {
    if (storage_ == nullptr) {
        return false;
    }
    std::memcpy(&destination, storage_, sizeof(destination));
    return true;
}

bool LiveStateView::ReplaceWith(const CpuFrame& source) const noexcept {
    if (storage_ == nullptr) {
        return false;
    }
    std::memcpy(storage_, &source, sizeof(source));
    return true;
}

LiveStateView PublishedLiveState() noexcept {
    return LiveStateView{Open(active_storage)};
}

void* ExchangeLiveStateStorage(void* const storage) noexcept {
    void* const previous = Open(active_storage);
    Seal(active_storage, storage);
    return previous;
}

const void* ExchangeContinuationStorage(const void* const storage) noexcept {
    const void* const previous = Open(continuation_storage);
    Seal(continuation_storage, storage);
    return previous;
}

bool SnapshotContinuationState(CpuFrame& destination) noexcept {
    const void* const storage = Open(continuation_storage);
    if (storage == nullptr) {
        return false;
    }
    std::memcpy(&destination, storage, sizeof(destination));
    return true;
}

} // namespace Lsx4::Translation
