// SPDX-FileCopyrightText: Copyright (C) 2001-2024 Free Software Foundation, Inc.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/logging/log.h"
#include "core/loader/dwarf.h"

namespace Dwarf {

template <typename T>
T get(uintptr_t addr) {
    T val;
    memcpy(&val, reinterpret_cast<void*>(addr), sizeof(T));
    return val;
}

static uintptr_t getEncodedP(uintptr_t& addr, uintptr_t end, u8 encoding, uintptr_t datarelBase) {
    const uintptr_t startAddr = addr;
    const u8* p = (u8*)addr;
    uintptr_t result;

    switch (encoding & 0x0F) {
    case DW_EH_PE_ptr:
        result = get<uintptr_t>(addr);
        p += sizeof(uintptr_t);
        addr = (uintptr_t)p;
        break;
    case DW_EH_PE_udata2:
        result = get<u16>(addr);
        p += sizeof(u16);
        addr = (uintptr_t)p;
        break;
    case DW_EH_PE_udata4:
        result = get<u32>(addr);
        p += sizeof(u32);
        addr = (uintptr_t)p;
        break;
    case DW_EH_PE_udata8:
        result = get<u64>(addr);
        p += sizeof(u64);
        addr = (uintptr_t)p;
        break;
    case DW_EH_PE_sdata2:
        result = get<s16>(addr);
        p += sizeof(s16);
        addr = (uintptr_t)p;
        break;
    case DW_EH_PE_sdata4:
        result = get<s32>(addr);
        p += sizeof(s32);
        addr = (uintptr_t)p;
        break;
    case DW_EH_PE_sdata8:
        result = get<s64>(addr);
        p += sizeof(s64);
        addr = (uintptr_t)p;
        break;
    default:
        UNREACHABLE_MSG("unknown pointer encoding");
    }

    switch (encoding & 0x70) {
    case DW_EH_PE_absptr:
        break;
    case DW_EH_PE_pcrel:
        result += startAddr;
        break;
    case DW_EH_PE_textrel:
        UNREACHABLE_MSG("DW_EH_PE_textrel pointer encoding not supported");
        break;
    case DW_EH_PE_datarel:
        if (datarelBase == 0)
            UNREACHABLE_MSG("DW_EH_PE_datarel is invalid with a datarelBase of 0");
        result += datarelBase;
        break;
    case DW_EH_PE_funcrel:
        UNREACHABLE_MSG("DW_EH_PE_funcrel pointer encoding not supported");
        break;
    case DW_EH_PE_aligned:
        UNREACHABLE_MSG("DW_EH_PE_aligned pointer encoding not supported");
        break;
    default:
        UNREACHABLE_MSG("unknown pointer encoding");
        break;
    }

    if (encoding & DW_EH_PE_indirect) {
        result = get<uintptr_t>(result);
    }

    return result;
}

bool DecodeEHHdr(uintptr_t ehHdrStart, uintptr_t ehHdrEnd, EHHeaderInfo& ehHdrInfo) {
    auto p = ehHdrStart;
    if (ehHdrEnd - ehHdrStart < 4) {
        if (ehHdrEnd == ehHdrStart) {
            return false;
        }
        LOG_ERROR(Core_Linker,
                  "Unsupported .eh_frame_hdr at {:#x} "
                  "need at least 4 bytes of data but only got {:#x}",
                  ehHdrStart, ehHdrEnd - ehHdrStart);
        return false;
    }

    const u8 version = get<u8>(p++);
    if (version != 1) {
        LOG_CRITICAL(Core_Linker, "Unsupported .eh_frame_hdr version: {:#x} at {:#x}", version,
                     ehHdrStart);
        return false;
    }

    const u8 eh_frame_ptr_enc = get<u8>(p++);
    const u8 fde_count_enc = get<u8>(p++);
    ehHdrInfo.table_enc = get<u8>(p++);

    ehHdrInfo.eh_frame_ptr = getEncodedP(p, ehHdrEnd, eh_frame_ptr_enc, ehHdrStart);
    ehHdrInfo.fde_count =
        fde_count_enc == DW_EH_PE_omit ? 0 : getEncodedP(p, ehHdrEnd, fde_count_enc, ehHdrStart);
    ehHdrInfo.table = p;

    return true;
}

}
