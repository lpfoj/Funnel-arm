<!--
SPDX-FileCopyrightText: 2026 Funnel-arm contributors
SPDX-License-Identifier: GPL-2.0-or-later
-->

<h1 align="center">Funnel-arm</h1>

<p align="center">
  Android/ARM64 adaptation layer consumed by LSX4.
</p>

Funnel-arm owns the desktop-emulation responsibilities adapted for the LSX4 Android runtime. It is a source layer, not a standalone desktop application or distribution.

## Integration contract

- `cmake/lsx4_arm_desktop_layer.cmake` is the single Android source manifest.
- `src/common`, `src/core`, `src/input`, `src/imgui`, `src/shader_recompiler`, and `src/video_core` provide the active emulation layer.
- `src/emulator.cpp` and `src/sdl_window.cpp` expose the Android-facing runtime boundary.
- LSX4 owns final target composition and third-party dependency integration; this repository does not vendor duplicate external projects.

Desktop installers, desktop entry points, distribution assets, Discord integration, ShadNet, Big Picture UI, unsupported host backends, and unused codec/service implementations are intentionally absent. Android client UI and lifecycle code remain in the LSX4 application repository.

Raw foundations are drawn from [shadPS4](https://github.com/shadps4-emu/shadPS4) and [Orbis](https://github.com/orbisdev).
