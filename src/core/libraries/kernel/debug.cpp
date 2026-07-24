// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "core/libraries/kernel/file_system.h"
#include "core/libraries/kernel/orbis_error.h"
#include "core/libraries/libs.h"

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace Libraries::Kernel {

void PS4_SYSV_ABI sceKernelDebugOutText(void* unk, char* text) {
#ifdef __ANDROID__
    if (text != nullptr) {
        __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                            "[EXECUTOR_DEBUG_OUT] %.*s", 512, text);
    }
#endif
    if (text != nullptr) {
        sceKernelWrite(1, text, strlen(text));
    }
    return;
}

void RegisterDebug(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("9JYNqN6jAKI", "libkernel", 1, "libkernel", sceKernelDebugOutText);
}

}
