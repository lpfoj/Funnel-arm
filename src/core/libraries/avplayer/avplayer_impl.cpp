// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/libraries/avplayer/avplayer_common.h"
#include "core/libraries/avplayer/avplayer_error.h"
#include "core/libraries/avplayer/avplayer_impl.h"

#ifdef __ANDROID__
extern "C" int executor_lsx4_android_run_guest_callback4(
    void* callback, std::uint64_t arg0, std::uint64_t arg1, std::uint64_t arg2,
    std::uint64_t arg3, std::uint32_t arg_count, std::uint64_t* result_out,
    const char* reason) __attribute__((weak));
#endif

namespace Libraries::AvPlayer {

namespace {

int RunGuestReplacementCallback(void* callback, const u64 arg0, const u64 arg1, const u64 arg2,
                                const u64 arg3, const u32 arg_count, u64* result,
                                const char* reason) {
#ifdef __ANDROID__
    if (executor_lsx4_android_run_guest_callback4 != nullptr) {
        return executor_lsx4_android_run_guest_callback4(callback, arg0, arg1, arg2, arg3,
                                                            arg_count, result, reason);
    }
#else
    (void)callback;
    (void)arg0;
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg_count;
    (void)result;
    (void)reason;
#endif
    return 0;
}

}

void* PS4_SYSV_ABI AvPlayer::Allocate(void* handle, u32 alignment, u32 size) {
    const auto* const self = reinterpret_cast<AvPlayer*>(handle);
    const auto allocate = self->m_init_data_original.memory_replacement.allocate;
    const auto ptr = self->m_init_data_original.memory_replacement.object_ptr;
    u64 result = 0;
    const int bridge = RunGuestReplacementCallback(
        reinterpret_cast<void*>(allocate), reinterpret_cast<u64>(ptr), alignment, size, 0, 3,
        &result, "avplayer_allocate");
    if (bridge > 0) {
        return reinterpret_cast<void*>(result);
    }
    if (bridge < 0) {
        return nullptr;
    }
    return allocate(ptr, alignment, size);
}

void PS4_SYSV_ABI AvPlayer::Deallocate(void* handle, void* memory) {
    const auto* const self = reinterpret_cast<AvPlayer*>(handle);
    const auto deallocate = self->m_init_data_original.memory_replacement.deallocate;
    const auto ptr = self->m_init_data_original.memory_replacement.object_ptr;
    const int bridge = RunGuestReplacementCallback(
        reinterpret_cast<void*>(deallocate), reinterpret_cast<u64>(ptr),
        reinterpret_cast<u64>(memory), 0, 0, 2, nullptr, "avplayer_deallocate");
    if (bridge != 0) {
        return;
    }
    return deallocate(ptr, memory);
}

void* PS4_SYSV_ABI AvPlayer::AllocateTexture(void* handle, u32 alignment, u32 size) {
    const auto* const self = reinterpret_cast<AvPlayer*>(handle);
    const auto allocate = self->m_init_data_original.memory_replacement.allocate_texture;
    const auto ptr = self->m_init_data_original.memory_replacement.object_ptr;
    u64 result = 0;
    const int bridge = RunGuestReplacementCallback(
        reinterpret_cast<void*>(allocate), reinterpret_cast<u64>(ptr), alignment, size, 0, 3,
        &result, "avplayer_allocate_texture");
    if (bridge > 0) {
        return reinterpret_cast<void*>(result);
    }
    if (bridge < 0) {
        return nullptr;
    }
    return allocate(ptr, alignment, size);
}

void PS4_SYSV_ABI AvPlayer::DeallocateTexture(void* handle, void* memory) {
    const auto* const self = reinterpret_cast<AvPlayer*>(handle);
    const auto deallocate = self->m_init_data_original.memory_replacement.deallocate_texture;
    const auto ptr = self->m_init_data_original.memory_replacement.object_ptr;
    const int bridge = RunGuestReplacementCallback(
        reinterpret_cast<void*>(deallocate), reinterpret_cast<u64>(ptr),
        reinterpret_cast<u64>(memory), 0, 0, 2, nullptr, "avplayer_deallocate_texture");
    if (bridge != 0) {
        return;
    }
    return deallocate(ptr, memory);
}

int PS4_SYSV_ABI AvPlayer::OpenFile(void* handle, const char* filename) {
    auto const self = reinterpret_cast<AvPlayer*>(handle);
    std::lock_guard guard(self->m_file_io_mutex);

    const auto open = self->m_init_data_original.file_replacement.open;
    const auto ptr = self->m_init_data_original.file_replacement.object_ptr;
    u64 result = 0;
    const int bridge = RunGuestReplacementCallback(
        reinterpret_cast<void*>(open), reinterpret_cast<u64>(ptr),
        reinterpret_cast<u64>(filename), 0, 0, 2, &result, "avplayer_open_file");
    if (bridge > 0) {
        return static_cast<int>(result);
    }
    if (bridge < 0) {
        return -1;
    }
    return open(ptr, filename);
}

int PS4_SYSV_ABI AvPlayer::CloseFile(void* handle) {
    auto const self = reinterpret_cast<AvPlayer*>(handle);
    std::lock_guard guard(self->m_file_io_mutex);

    const auto close = self->m_init_data_original.file_replacement.close;
    const auto ptr = self->m_init_data_original.file_replacement.object_ptr;
    u64 result = 0;
    const int bridge = RunGuestReplacementCallback(
        reinterpret_cast<void*>(close), reinterpret_cast<u64>(ptr), 0, 0, 0, 1, &result,
        "avplayer_close_file");
    if (bridge > 0) {
        return static_cast<int>(result);
    }
    if (bridge < 0) {
        return -1;
    }
    return close(ptr);
}

int PS4_SYSV_ABI AvPlayer::ReadOffsetFile(void* handle, u8* buffer, u64 position, u32 length) {
    auto const self = reinterpret_cast<AvPlayer*>(handle);
    std::lock_guard guard(self->m_file_io_mutex);

    const auto read_offset = self->m_init_data_original.file_replacement.read_offset;
    const auto ptr = self->m_init_data_original.file_replacement.object_ptr;
    u64 result = 0;
    const int bridge = RunGuestReplacementCallback(
        reinterpret_cast<void*>(read_offset), reinterpret_cast<u64>(ptr),
        reinterpret_cast<u64>(buffer), position, length, 4, &result, "avplayer_read_offset");
    if (bridge > 0) {
        return static_cast<int>(result);
    }
    if (bridge < 0) {
        return -1;
    }
    return read_offset(ptr, buffer, position, length);
}

u64 PS4_SYSV_ABI AvPlayer::SizeFile(void* handle) {
    auto const self = reinterpret_cast<AvPlayer*>(handle);
    std::lock_guard guard(self->m_file_io_mutex);

    const auto size = self->m_init_data_original.file_replacement.size;
    const auto ptr = self->m_init_data_original.file_replacement.object_ptr;
    u64 result = 0;
    const int bridge = RunGuestReplacementCallback(
        reinterpret_cast<void*>(size), reinterpret_cast<u64>(ptr), 0, 0, 0, 1, &result,
        "avplayer_size_file");
    if (bridge > 0) {
        return result;
    }
    if (bridge < 0) {
        return 0;
    }
    return size(ptr);
}

AvPlayerInitData AvPlayer::StubInitData(const AvPlayerInitData& data) {
    AvPlayerInitData result = data;
    result.memory_replacement.object_ptr = this;
    result.memory_replacement.allocate = &AvPlayer::Allocate;
    result.memory_replacement.deallocate = &AvPlayer::Deallocate;
    result.memory_replacement.allocate_texture = &AvPlayer::AllocateTexture;
    result.memory_replacement.deallocate_texture = &AvPlayer::DeallocateTexture;
    if (data.file_replacement.open == nullptr || data.file_replacement.close == nullptr ||
        data.file_replacement.read_offset == nullptr || data.file_replacement.size == nullptr) {
        result.file_replacement = {};
    } else {
        result.file_replacement.object_ptr = this;
        result.file_replacement.open = &AvPlayer::OpenFile;
        result.file_replacement.close = &AvPlayer::CloseFile;
        result.file_replacement.read_offset = &AvPlayer::ReadOffsetFile;
        result.file_replacement.size = &AvPlayer::SizeFile;
    }
    return result;
}

AvPlayer::AvPlayer(const AvPlayerInitData& data)
    : m_init_data(StubInitData(data)), m_init_data_original(data),
      m_state(std::make_unique<AvPlayerState>(m_init_data)) {}

s32 AvPlayer::PostInit(const AvPlayerPostInitData& data) {
    m_state->PostInit(data);
    return ORBIS_OK;
}

s32 AvPlayer::AddSource(std::string_view path) {
    return AddSourceEx(path, AvPlayerSourceType::Unknown);
}

s32 AvPlayer::AddSourceEx(std::string_view path, AvPlayerSourceType source_type) {
    if (source_type == AvPlayerSourceType::Unknown) {
        source_type = GetSourceType(path);
    }
    if (source_type == AvPlayerSourceType::Hls) {
        LOG_ERROR(Lib_AvPlayer, "HTTP Live Streaming is not implemented");
        return ORBIS_AVPLAYER_ERROR_NOT_SUPPORTED;
    }
    if (!m_state->AddSource(path, GetSourceType(path))) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::GetStreamCount() {
    if (m_state == nullptr) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    const auto res = m_state->GetStreamCount();
    if (AVPLAYER_IS_ERROR(res)) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return res;
}

s32 AvPlayer::GetStreamInfo(u32 stream_index, AvPlayerStreamInfo& info) {
    if (!m_state->GetStreamInfo(stream_index, info)) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::EnableStream(u32 stream_index) {
    if (m_state == nullptr) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    if (!m_state->EnableStream(stream_index)) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::Start() {
    if (m_state == nullptr || !m_state->Start()) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::Pause() {
    if (m_state == nullptr || !m_state->Pause()) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::Resume() {
    if (m_state == nullptr || !m_state->Resume()) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

s32 AvPlayer::SetAvSyncMode(AvPlayerAvSyncMode sync_mode) {
    if (m_state == nullptr) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    m_state->SetAvSyncMode(sync_mode);
    return ORBIS_OK;
}

bool AvPlayer::GetVideoData(AvPlayerFrameInfo& video_info) {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->GetVideoData(video_info);
}

bool AvPlayer::GetVideoData(AvPlayerFrameInfoEx& video_info) {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->GetVideoData(video_info);
}

bool AvPlayer::GetAudioData(AvPlayerFrameInfo& audio_info) {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->GetAudioData(audio_info);
}

bool AvPlayer::IsActive() {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->IsActive();
}

u64 AvPlayer::CurrentTime() {
    if (m_state == nullptr) {
        return 0;
    }
    return m_state->CurrentTime();
}

s32 AvPlayer::Stop() {
    if (m_state == nullptr || !m_state->Stop()) {
        return ORBIS_AVPLAYER_ERROR_OPERATION_FAILED;
    }
    return ORBIS_OK;
}

bool AvPlayer::SetLooping(bool is_looping) {
    if (m_state == nullptr) {
        return false;
    }
    return m_state->SetLooping(is_looping);
}

}
