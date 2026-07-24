// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/types.h"
#include "core/libraries/fiber/fiber.h"

namespace Libraries::Fiber {

extern "C" void PS4_SYSV_ABI _sceFiberForceQuit(u64 ret);

extern "C" s32 PS4_SYSV_ABI _sceFiberSetJmp(OrbisFiberContext* ctx) {
    if (ctx) {
        ctx->return_val = 0;
    }
    return 0;
}

extern "C" s32 PS4_SYSV_ABI _sceFiberLongJmp(OrbisFiberContext* ctx) {
    if (ctx) {
        ctx->return_val = 1;
    }
    return 1;
}

extern "C" void PS4_SYSV_ABI _sceFiberSwitchEntry(OrbisFiberData* data, bool set_fpu) {
    (void)set_fpu;
    if (!data || !data->entry) {
        return;
    }
    if (data->state) {
        *data->state = FiberState::Idle;
    }
    data->entry(data->arg_on_initialize, data->arg_on_run_to);
    _sceFiberForceQuit(1);
}

}
