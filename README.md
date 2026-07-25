<!--
SPDX-FileCopyrightText: 2026 Funnel-arm contributors
SPDX-License-Identifier: GPL-2.0-or-later
-->

<p align="center">
  <img src="assets/funnel-arm-icon.png" width="280" alt="Funnel-arm">
</p>

<h1 align="center">Funnel-arm</h1>

<p align="center">
  ARM64 adaptation layer for x86-64 console emulation cores.
</p>

Funnels upstream PC emulators into ARM-ready building blocks: ported components, focused
performance adaptations, and integration groundwork.

Funnel-arm is the desktop-emulation layer adapted for the LSX4 Android runtime. It keeps
emulation-core responsibilities separate from the Android application, so each layer can
evolve and be optimized at its own boundary.

## What this repository owns

- CPU-independent core, library, input, shader, and Vulkan emulation components used by LSX4.
- ARM64 and Android host adaptations required by those components.
- The runtime bridge exposed to the LSX4 native executor.
- Focused performance work that belongs to the emulation layer rather than the Android client.

## Integration contract

- `cmake/lsx4_arm_desktop_layer.cmake` is the authoritative Android source manifest.
- `src/common`, `src/core`, `src/input`, `src/imgui`, `src/shader_recompiler`, and
  `src/video_core` provide the active emulation layer.
- `src/emulator.cpp` and `src/sdl_window.cpp` expose the Android-facing runtime boundary.
- LSX4 owns final target composition, Android lifecycle and UI, and third-party dependency
  integration.
- Active desktop-emulation sources are compiled from Funnel-arm; the LSX4 client does not
  maintain a duplicate desktop layer.

## Scope

Funnel-arm is a source layer, not a standalone desktop application or distribution. Desktop
installers, desktop entry points, distribution assets, Discord integration, ShadNet, Big
Picture UI, unsupported host backends, and unused codec or service implementations are
intentionally absent.

Raw foundations are drawn from [shadPS4](https://github.com/shadps4-emu/shadPS4) and [Orbis](https://github.com/orbisdev).
