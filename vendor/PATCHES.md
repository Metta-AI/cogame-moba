# Patch set applied to the vendored sim

Files under `vendor/upstream/` are byte-pristine (see `vendor/UPSTREAM.md`).
All modifications are the patch files in `sim/patches/`, applied at build time
by `sim/apply_patches.sh` into `build/`:

- `build/src-pristine/` gets **0001 only** — the minimum required to compile
  the sim without raylib. This tree is the reference build for the fidelity
  test (`tests/test_fidelity.py`).
- `build/src-patched/` gets **all** patches. This is the production sim.

**Invariant:** patches must keep in-episode physics byte-identical. The
fidelity test drives both builds with identical seed + action logs and
requires byte-identical obs and reward streams. A patch that breaks it is
rejected.

## 0001-render-guard.patch

Guards `#include "raylib.h"` (+ the render-only `GLSL_VERSION` block) and the
entire renderer section of `moba.h` (everything from the `// Raylib client`
comment at former line 1970 through end of file: `COLORS[]`, `MapRenderer`,
`GameRenderer`, `c_render`, `close_game_renderer`) behind `#ifdef MOBA_RENDER`.

- Rationale: the server-side sim build must compile to STANDALONE_WASM with no
  raylib. Upstream's sim half (lines 1–1969) uses only libc/libm; the raylib
  include is unconditional, so without this guard nothing compiles headless.
- No sim lines change. Upstream's `c_close`/`free_allocated_moba` free only
  sim allocations (no renderer frees), so no in-function splits were needed.
  The `GameRenderer* client` pointer in `struct MOBA` stays: `GameRenderer` is
  forward-declared in the sim half and only ever dereferenced in the renderer.
- The viewer build (Phase 4) compiles the same tree with `-DMOBA_RENDER`.

## 0002-seed.patch

Adds `unsigned int seed;` to `struct MOBA` and calls `srand(env->seed);` at
the top of `init_moba()` — before the `CachedRNG` table is filled from
`rand()` and before any spawn jitter draws.

- Rationale: upstream 4.0 never calls `srand`, so every process replays the
  libc default stream (equivalent to seed 1). Upstream 3.0's
  `env_binding.h` did `srand(seed)` at env init; this restores that behavior
  so a league can run varied episodes. The seed value is provided by
  `sim/shim.c`'s `moba_init(seed, ...)`.
- In-episode physics are unchanged: for seed == 1 the stream is identical to
  the unseeded default (musl `srand(s)` stores `s - 1`; initial state is 0),
  which is exactly what the fidelity test relies on.

## 0003-done-flag.patch

Adds `int done; int winner;` to `struct MOBA`. In `c_step`'s win branch
(ancient death → `do_reset`), sets `env->done = 1` and `env->winner`
(0 = radiant, 1 = dire), keeps `add_log`, and **skips the internal
`c_reset(env)` auto-reset**.

- Rationale: upstream never writes `terminals` and silently restarts the
  episode inside `c_step` when an ancient dies. The Coworld server needs to
  detect the end of the episode and score it; after `done`, the server stops
  stepping. Affects only post-win behavior — every tick up to and including
  the winning tick is byte-identical (rewards for the winning tick are
  computed in `step_players`, before the win check).
- `done`/`winner` are cleared by the shim's `moba_reset()`, not by `c_reset`
  (keeps the patch surface minimal).
