// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/singleton.h"
#include "core/libraries/kernel/threads/pthread.h"
#include "core/libraries/libs.h"
#include "core/linker.h"
#include "core/tls.h"

namespace Libraries::Kernel {

static constexpr size_t TlsTcbSize = 0x40;
static constexpr size_t TlsTcbAlign = 0x20;

static std::shared_mutex RtldLock;

Core::Tcb* TcbCtor(Pthread* thread, int initial) {
    std::scoped_lock lk{RtldLock};

    auto* linker = Common::Singleton<Core::Linker>::Instance();
    auto* addr_out = linker->AllocateTlsForThread(initial);
    ASSERT_MSG(addr_out, "Unable to allocate guest TCB");

    const u32 num_dtvs = linker->MaxTlsIndex();
    const auto static_tls_size = linker->StaticTlsSize();
    auto* dtv_table = new Core::DtvEntry[num_dtvs + 2]{};

    u8* addr = reinterpret_cast<u8*>(addr_out);
    auto* tcb = reinterpret_cast<Core::Tcb*>(addr + static_tls_size);
    memset(addr_out, 0, static_tls_size);
    tcb->tcb_self = tcb;
    tcb->tcb_dtv = dtv_table;

    dtv_table[0].counter = linker->GenerationCounter();
    dtv_table[1].counter = num_dtvs;

    auto* module = linker->GetModule(0);
    u8* dest = reinterpret_cast<u8*>(addr + static_tls_size - module->tls.offset);

    if (module->tls.image_size != 0) {
        if (module->tls.image_virtual_addr != 0) {
            const u8* src = reinterpret_cast<const u8*>(module->tls.image_virtual_addr);
            memcpy(dest, src, module->tls.init_image_size);
        }
        ASSERT_MSG(module->tls.modid > 0 && module->tls.modid <= num_dtvs);
        tcb->tcb_dtv[module->tls.modid + 1].pointer = dest;
    }

    if (tcb) {
        tcb->tcb_thread = thread;
        tcb->tcb_fiber = nullptr;
    }
    return tcb;
}

void TcbDtor(Core::Tcb* oldtls) {
    std::scoped_lock lk{RtldLock};
    auto* dtv_table = oldtls->tcb_dtv;

    auto* linker = Common::Singleton<Core::Linker>::Instance();
    const u32 max_tls_index = linker->MaxTlsIndex();
    const u32 num_dtvs = dtv_table[1].counter;
    ASSERT_MSG(num_dtvs <= max_tls_index, "Out of bounds DTV access");

    const u32 static_tls_size = linker->StaticTlsSize();
    const u8* tls_base = (const u8*)oldtls - static_tls_size;

    for (int i = 1; i < num_dtvs; i++) {
        u8* dtv_ptr = dtv_table[i + 1].pointer;
        if (dtv_ptr && (dtv_ptr < tls_base || (const u8*)oldtls < dtv_ptr)) {
            const auto* module = linker->GetModule(i - 1);
            const size_t dynamic_size = module && module->tls.image_size ? module->tls.image_size : 1;
            linker->FreeTlsForNonPrimaryThread(dtv_ptr, dynamic_size);
        }
    }

#ifdef __ANDROID__
    std::memset(const_cast<u8*>(tls_base), 0, static_tls_size + TlsTcbSize);
#endif

    delete[] dtv_table;
}

struct TlsIndex {
    u64 ti_module;
    u64 ti_offset;
};

void* PS4_SYSV_ABI __tls_get_addr(TlsIndex* index) {
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    return linker->TlsGetAddr(index->ti_module, index->ti_offset);
}

void RegisterRtld(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("vNe1w4diLCs", "libkernel", 1, "libkernel", __tls_get_addr);
}

}
