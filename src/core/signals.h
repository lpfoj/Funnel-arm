// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <set>
#include <signal.h>
#include "common/singleton.h"
#include "common/types.h"

#ifdef _WIN32
#define SIGSLEEP -1
#else
#define SIGSLEEP SIGVTALRM
#endif
namespace Core {

using AccessViolationHandler = bool (*)(void* context, void* fault_address);
using IllegalInstructionHandler = bool (*)(void* context);

#ifndef _WIN32
void SignalHandler(int sig, siginfo_t* info, void* raw_context);
#endif

#ifdef __ANDROID__
extern std::atomic<int> g_executor_render_tid;
#endif

class SignalDispatch {
public:
    SignalDispatch();
    ~SignalDispatch();

    void RegisterAccessViolationHandler(const AccessViolationHandler& handler, u32 priority) {
        access_violation_handlers.emplace(handler, priority);
    }

    void RegisterIllegalInstructionHandler(const IllegalInstructionHandler& handler, u32 priority) {
        illegal_instruction_handlers.emplace(handler, priority);
    }

    bool DispatchAccessViolation(void* context, void* fault_address) const;

    bool DispatchIllegalInstruction(void* context) const;

private:
    template <typename T>
    struct HandlerEntry {
        T handler;
        u32 priority;

        std::strong_ordering operator<=>(const HandlerEntry& right) const {
            return priority <=> right.priority;
        }
    };
    std::set<HandlerEntry<AccessViolationHandler>> access_violation_handlers;
    std::set<HandlerEntry<IllegalInstructionHandler>> illegal_instruction_handlers;

#ifdef _WIN32
    void* handle{};
#endif
};

using Signals = Common::Singleton<SignalDispatch>;

}
