<p align="center">
  <img src="assets/lsx4-banner.png" width="640" alt="LSX4, open-source PlayStation 4 emulator for Android">
</p>

# LSX4

LSX4 is an open-source PlayStation 4 emulator application for Android.

Development also makes use of current-generation large language models (LLMs).

LSX4 does not include games, firmware, or other copyrighted system content.

## Playable games

This list includes only games that have been tested. The broader set of supported games is larger and still requires testing.

| Game | Playable | Notes |
| --- | :---: | --- |
| Bloodborne | ✅ | Gameplay is temporarily limited to 7 FPS. |
| Deltarune Chapters 1 & 2 | ✅ | |
| Downwell | ✅ | |
| Hotline Miami 2: Wrong Number | ✅ | |
| Hyper Light Drifter | ✅ | |
| Just Shapes & Beats | ✅ | |
| Limbo | ✅ | |
| Minit | ✅ | |
| Nidhogg | ✅ | |
| Nidhogg 2 | ✅ | |
| Owlboy | ✅ | |
| Rain World | ✅ | |
| Shovel Knight | ✅ | |
| Sonic Mania | ✅ | |
| Undertale | ✅ | |

## Repository layout

| Path | Purpose |
| --- | --- |
| `android-app/` | LSX4 Android application |
| `src/` | Native runtime, translation, system, and graphics sources |
| `externals/` | Native build dependencies |
| `cmake/` | Android native build support |
| `scripts/` | Android dependency build helpers |
| `funnel-arm/` | ARM adaptation foundation linked from Funnel-arm |
| `assets/` | Project artwork |

## Android application

```sh
cd android-app
./gradlew assembleDebug
```

## Native runtime

```sh
git submodule update --init --recursive
powershell -File scripts/build-ffmpeg-android-pic.ps1
cmake --preset android-arm64-release
cmake --build --preset android-arm64-release
```
