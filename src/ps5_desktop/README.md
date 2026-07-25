# Funnel PS5 desktop layer

This directory is the only source root for the PS5 desktop implementation
consumed by LSX4. It is deliberately outside the PS4 `common`, `core`,
`video_core`, `shader_recompiler`, `input`, and `imgui` profiles.

Isolation rules:

1. The PS4 manifest must never include a source from this directory.
2. PS5 sources must not register libraries, handlers, caches, or global state
   in the PS4 runtime.
3. A source shared with PS4 must first move behind a platform-neutral contract
   and must retain PS4 verification. Including a PS4 implementation directly
   from this directory is not a shared contract.
4. Guest memory granularity, artifact-cache identity, HLE resolution, fault
   delivery, and GPU command processing are platform-owned configuration.
5. The LSX4 build option for this layer is off by default. Enabling it builds
   the separate `lsx4_executor_ps5_android` shared library and does not add
   PS5 objects to the PS4 shared library.

The executable probe checkpoint is complete: the isolated loader accepts
NextGen ELF and uncompressed SELF containers, maps 16 KiB guest pages, applies
the extended relocation set, binds imports to runtime-owned HLE thunks, and
reaches an HLE callback through the LSX4 AArch64 backend without initializing
the PS4 runtime.

The loader now carries the SharpEmu-style Variant II TLS layout across the
main executable and startup PRX images, including per-module IDs/static
offsets and `DTPMOD64`, `DTPOFF64`, and `TPOFF64` relocations. It also exposes
preinit/init/fini functions and the extended x86-64 relocation set needed by
real modules. It reads `DT_NEEDED`, publishes full/NID export aliases, records
deferred function/data relocations, and safely rebinds an already protected
eboot after adjacent PRX images are mapped. The runtime scans the same adjacent
module directories as SharpEmu, starts boot modules in dependency order, and
reports unresolved HLE versus critical data/TLS imports before launch.

Late modules are also installed into existing thread contexts: the runtime
uses a reserved `0x20000` static-TLS prefix, copies `tdata`, zeroes `tbss`,
updates DTV generation/module bounds, and publishes the new slot without
moving FS. Encrypted/compressed SELF segments and the production Prospero
HLE/GPU surface remain later milestones. The current game probe therefore
targets the same legally dumped, decrypted/uncompressed eboot and PRX inputs
used by SharpEmu.
