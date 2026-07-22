// SPDX-FileCopyrightText: Copyright 2026 LSX4
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

namespace Executor::Jit {
struct LsxDecodedOp;
struct LsxMachineImage;
}

namespace Lsx4::Translation {

[[nodiscard]] std::string RenderInstructionEvent(
    const Executor::Jit::LsxDecodedOp& instruction);
[[nodiscard]] std::string RenderRegisterEvent(
    const Executor::Jit::LsxMachineImage& state);

} // namespace Lsx4::Translation
