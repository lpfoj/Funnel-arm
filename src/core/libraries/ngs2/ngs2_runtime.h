// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/ngs2/ngs2.h"

namespace Libraries::Ngs2::Runtime {

s32 CreateSystem(const OrbisNgs2SystemOption* option,
                 const OrbisNgs2ContextBufferInfo* buffer_info,
                 OrbisNgs2Handle* out_handle);
s32 DestroySystem(OrbisNgs2Handle system_handle, OrbisNgs2ContextBufferInfo* out_buffer_info);
s32 GetSystemInfo(OrbisNgs2Handle system_handle, OrbisNgs2SystemInfo* out_info,
                  size_t info_size);
s32 GetSystemUserData(OrbisNgs2Handle system_handle, uintptr_t* out_user_data);
s32 SetSystemUserData(OrbisNgs2Handle system_handle, uintptr_t user_data);
s32 SetSystemGrainSamples(OrbisNgs2Handle system_handle, u32 num_samples);
s32 SetSystemSampleRate(OrbisNgs2Handle system_handle, u32 sample_rate);

s32 CreateRack(OrbisNgs2Handle system_handle, u32 rack_id,
               const OrbisNgs2RackOption* option,
               const OrbisNgs2ContextBufferInfo* buffer_info,
               OrbisNgs2Handle* out_handle);
s32 DestroyRack(OrbisNgs2Handle rack_handle, OrbisNgs2ContextBufferInfo* out_buffer_info);
s32 GetRackInfo(OrbisNgs2Handle rack_handle, OrbisNgs2RackInfo* out_info, size_t info_size);
s32 GetRackUserData(OrbisNgs2Handle rack_handle, uintptr_t* out_user_data);
s32 SetRackUserData(OrbisNgs2Handle rack_handle, uintptr_t user_data);
s32 GetVoiceHandle(OrbisNgs2Handle rack_handle, u32 voice_index,
                   OrbisNgs2Handle* out_handle);

s32 ControlVoice(OrbisNgs2Handle voice_handle, const OrbisNgs2VoiceParamHeader* params);
s32 GetVoiceOwner(OrbisNgs2Handle voice_handle, OrbisNgs2Handle* out_rack_handle,
                  u32* out_voice_index);
s32 GetVoiceState(OrbisNgs2Handle voice_handle, OrbisNgs2VoiceState* out_state,
                  size_t state_size);
s32 GetVoiceStateFlags(OrbisNgs2Handle voice_handle, u32* out_state_flags);
s32 GetVoiceMatrixInfo(OrbisNgs2Handle voice_handle, u32 matrix_id,
                       OrbisNgs2VoiceMatrixInfo* out_info, size_t info_size);
s32 GetVoicePortInfo(OrbisNgs2Handle voice_handle, u32 port,
                     OrbisNgs2VoicePortInfo* out_info, size_t info_size);

s32 Render(OrbisNgs2Handle system_handle, const OrbisNgs2RenderBufferInfo* buffers,
           u32 num_buffers);

}
