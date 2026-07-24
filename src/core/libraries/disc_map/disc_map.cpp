// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/logging/log.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "disc_map.h"
#include "disc_map_codes.h"

#ifdef __ANDROID__
#include <android/log.h>
#include <atomic>
#endif

namespace Libraries::DiscMap {

#ifdef __ANDROID__
namespace {

void ExecutorDiscMapLog(const char* fn, const char* path, s64 offset, s64 nbytes, const char* outs) {
    static std::atomic<int> budget{64};
    const int old = budget.fetch_sub(1, std::memory_order_relaxed);
    if (old <= 0) {
        return;
    }
    __android_log_print(ANDROID_LOG_INFO, "LSX4Native",
                        "[EXECUTOR_DISCMAP] fn=%s path=%s offset=%lld nbytes=%lld outs=%s "
                        "result=0x%x",
                        fn, path != nullptr ? path : "<null>", static_cast<long long>(offset),
                        static_cast<long long>(nbytes), outs, ORBIS_DISC_MAP_ERROR_NO_BITMAP_INFO);
}

}
#endif

int PS4_SYSV_ABI sceDiscMapGetPackageSize(s64 fflags, int* ret1, int* ret2) {
#ifdef __ANDROID__
    ExecutorDiscMapLog("GetPackageSize", nullptr, fflags, 0, "ret1,ret2");
#endif
    return ORBIS_DISC_MAP_ERROR_NO_BITMAP_INFO;
}

int PS4_SYSV_ABI sceDiscMapIsRequestOnHDD(char* path, s64 offset, s64 nbytes, int* ret) {
#ifdef __ANDROID__
    ExecutorDiscMapLog("IsRequestOnHDD", path, offset, nbytes, "ret");
#endif
    return ORBIS_DISC_MAP_ERROR_NO_BITMAP_INFO;
}

int PS4_SYSV_ABI Func_7C980FFB0AA27E7A(char* path, s64 offset, s64 nbytes, int* flags, int* ret1,
                                       int* ret2) {
    *flags = 0;
    *ret1 = 0;
    *ret2 = 0;
    return ORBIS_OK;
}

int PS4_SYSV_ABI Func_8A828CAEE7EDD5E9(char* path, s64 offset, s64 nbytes, int* flags, int* ret1,
                                       int* ret2) {
#ifdef __ANDROID__
    ExecutorDiscMapLog("Func_8A828CAEE7EDD5E9", path, offset, nbytes, "flags,ret1,ret2");
#endif
    return ORBIS_DISC_MAP_ERROR_NO_BITMAP_INFO;
}

int PS4_SYSV_ABI Func_E7EBCE96E92F91F8() {
    return ORBIS_DISC_MAP_ERROR_NO_BITMAP_INFO;
}

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    LIB_FUNCTION("fl1eoDnwQ4s", "libSceDiscMap", 1, "libSceDiscMap", sceDiscMapGetPackageSize);
    LIB_FUNCTION("lbQKqsERhtE", "libSceDiscMap", 1, "libSceDiscMap", sceDiscMapIsRequestOnHDD);
    LIB_FUNCTION("fJgP+wqifno", "libSceDiscMap", 1, "libSceDiscMap", Func_7C980FFB0AA27E7A);
    LIB_FUNCTION("ioKMruft1ek", "libSceDiscMap", 1, "libSceDiscMap", Func_8A828CAEE7EDD5E9);
    LIB_FUNCTION("5+vOlukvkfg", "libSceDiscMap", 1, "libSceDiscMap", Func_E7EBCE96E92F91F8);
};

}
