# Vendored upstream: PufferLib Ocean MOBA

- **Repo:** https://github.com/PufferAI/PufferLib
- **Commit:** `c5d3c637446047a6efbcaa74c039c5295d201ab0` (branch 4.0)
- **License:** MIT (see `vendor/LICENSE-pufferlib`)
- **Fetched:** 2026-08-01

## Rule: files under `vendor/upstream/` are byte-pristine

Never edit files in `vendor/upstream/`. All modifications are patch files in
`sim/patches/`, applied at build time into `build/` by `sim/apply_patches.sh`.
Pristineness is verifiable by diffing against the pinned upstream commit and
by the sha256 sums below.

## Files

| vendored file | upstream path | sha256 |
|---|---|---|
| `moba.h` | `ocean/moba/moba.h` | `3e705457337416c4390cb32ac5813e944ba7222010ad7a7070cc7af13a111c02` |
| `game_map.h` | `ocean/moba/game_map.h` | `e4cc1b91bc630a85120a3528b7a9ce60689eb3d40a187b7757a5bea9f16ae629` |
| `binding.c` | `ocean/moba/binding.c` | `7e795954b689fa7bcf37bcb8496384b68d2f4eb042e3a3887daf6327384bdbe0` |
| `moba.c` | `ocean/moba/moba.c` | `90ced112e4d381f19416e1b80e2179fa4955000cd1704190d9ae430a53292cce` |
| `puffernet.h` | `src/puffernet.h` | `f7f53ca1a1d1a56190bc8c73a099d5ac356013da3e4abdb0050342e33b88405b` |
| `moba_weights.bin` | `resources/moba/moba_weights.bin` | `394b19fa8e2894879e05f9bcdff1da78ae2e83ec161e16ca8ad4a5811674e896` |
| `moba.ini` | `config/moba.ini` | `38596e3630fe4758f784917c7ea27fcfbc5e0895973da6aceb465a1e530a5dfb` |

`vendor/LICENSE-pufferlib` is the upstream repo `LICENSE` file.

## Build toolchain

- emscripten: `emcc 6.0.5-git` (Homebrew). Recorded for build reproducibility;
  the wasm binaries are build outputs (gitignored), not committed.
