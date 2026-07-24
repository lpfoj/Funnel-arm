// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/libraries/kernel/threads/pthread.h"

namespace Libraries::Ngs2 {

static const int ORBIS_NGS2_SYSTEM_NAME_LENGTH = 16;
static const int ORBIS_NGS2_RACK_NAME_LENGTH = 16;

typedef uintptr_t OrbisNgs2Handle;

struct OrbisNgs2ContextBufferInfo {
    void* hostBuffer;
    size_t hostBufferSize;
    uintptr_t reserved[5];
    uintptr_t userData;
};

struct OrbisNgs2SystemOption {
    size_t size;
    char name[ORBIS_NGS2_SYSTEM_NAME_LENGTH];

    u32 flags;
    u32 maxGrainSamples;
    u32 numGrainSamples;
    u32 sampleRate;
    u32 aReserved[6];
};

using OrbisNgs2BufferAllocHandler =
    s32 PS4_SYSV_ABI (*)(OrbisNgs2ContextBufferInfo* io_buffer_info);
using OrbisNgs2BufferFreeHandler = s32 PS4_SYSV_ABI (*)(OrbisNgs2ContextBufferInfo* io_buffer_info);

struct OrbisNgs2SystemInfo {
    char name[ORBIS_NGS2_SYSTEM_NAME_LENGTH];

    OrbisNgs2Handle systemHandle;
    OrbisNgs2ContextBufferInfo bufferInfo;

    u32 uid;
    u32 minGrainSamples;
    u32 maxGrainSamples;

    u32 stateFlags;
    u32 rackCount;
    float lastRenderRatio;
    s64 lastRenderTick;
    s64 renderCount;
    u32 sampleRate;
    u32 numGrainSamples;
};

struct OrbisNgs2RackInfo {
    char name[ORBIS_NGS2_RACK_NAME_LENGTH];

    OrbisNgs2Handle rackHandle;
    OrbisNgs2ContextBufferInfo bufferInfo;

    OrbisNgs2Handle ownerSystemHandle;

    u32 type;
    u32 rackId;
    u32 uid;
    u32 minGrainSamples;
    u32 maxGrainSamples;
    u32 maxVoices;
    u32 maxChannelWorks;
    u32 maxInputs;
    u32 maxMatrices;
    u32 maxPorts;

    u32 stateFlags;
    float lastProcessRatio;
    u64 lastProcessTick;
    u64 renderCount;
    u32 activeVoiceCount;
    u32 activeChannelWorkCount;
};

struct StackBuffer {
    void** top;
    void* base;
    size_t size;
    size_t currentOffset;
    size_t usedSize;
    size_t totalSize;
    size_t alignment;
    u8 flags;
    char padding[7];
};

struct SystemInternal {
    char name[ORBIS_NGS2_SYSTEM_NAME_LENGTH];
    OrbisNgs2ContextBufferInfo bufferInfo;
    OrbisNgs2BufferFreeHandler hostFree;
    OrbisNgs2Handle systemHandle;
    void* unknown1;
    void* unknown2;
    OrbisNgs2Handle rackHandle;
    uintptr_t* userData;
    SystemInternal* systemList;
    StackBuffer* stackBuffer;
    OrbisNgs2SystemInfo ownerSystemInfo;

    struct rackList {
        void* prev;
        void* next;
        void* unknown;
    };

    rackList rackListPreset;
    rackList rackListNormal;
    rackList rackListMaster;

    void* unknown3;
    void* systemListPrev;
    void* unknown4;
    void* systemListNext;
    void* rackFunction;

    Kernel::PthreadMutex processLock;
    u32 hasProcessMutex;
    u32 unknown5;
    Kernel::PthreadMutex flushLock;
    u32 hasFlushMutex;
    u32 unknown6;

    u64 lastRenderTick;
    u64 renderCount;
    u32 isActive;
    std::atomic<int> lockCount;
    u32 uid;
    u32 systemType;

    struct {
        u8 isBufferValid : 1;
        u8 isRendering : 1;
        u8 isSorted : 1;
        u8 isFlushReady : 1;
    } flags;

    u16 currentMaxGrainSamples;
    u16 minGrainSamples;
    u16 maxGrainSamples;
    u16 numGrainSamples;
    u32 currentNumGrainSamples;
    u32 sampleRate;
    u32 currentSampleRate;
    u32 rackCount;
    float lastRenderRatio;
    float cpuLoad;
};

struct HandleInternal {
    HandleInternal* selfPtr;
    SystemInternal* systemData;
    std::atomic<int> refCount;
    u32 handleType;
    u32 handleID;
};

s32 StackBufferClose(StackBuffer* stackBuffer, size_t* outTotalSize);
s32 StackBufferOpen(StackBuffer* stackBuffer, void* buffer, size_t bufferSize, void** outBuffer,
                    u8 flags);
s32 SystemSetupCore(StackBuffer* stackBuffer, const OrbisNgs2SystemOption* option,
                    SystemInternal* outSystem);

s32 HandleReportInvalid(OrbisNgs2Handle handle, u32 handleType);
void* MemoryClear(void* buffer, size_t size);
s32 SystemCleanup(OrbisNgs2Handle systemHandle, OrbisNgs2ContextBufferInfo* outInfo);
s32 SystemSetup(const OrbisNgs2SystemOption* option, OrbisNgs2ContextBufferInfo* hostBufferInfo,
                OrbisNgs2BufferFreeHandler hostFree, OrbisNgs2Handle* outHandle);

}
