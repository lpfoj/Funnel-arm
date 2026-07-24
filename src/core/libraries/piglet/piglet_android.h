// SPDX-FileCopyrightText: Copyright 2026 LSX4 Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Core::Loader {
class SymbolsResolver;
}

namespace Libraries::Piglet {

void RegisterLib(Core::Loader::SymbolsResolver* sym);
void* ResolveSymbolByName(const char* name);
void SetGuestSwapExpected(bool enabled, const char* reason);

}
