# cogame-moba Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Design context (settled decisions, do not re-litigate): `docs/plans/2026-08-01-cogame-moba-design.md`.

**Goal:** A Coworld game repo (`Metta-AI/cogame-moba`) that runs PufferLib's Ocean MOBA with bit-exact
obs/action/physics via a wasm-compiled vendored sim, lockstep websocket server, static wasm replay
viewer, and a baseline player driven by upstream's pretrained weights.

**Architecture:** One vendored C sim (PufferLib @ `c5d3c637`) compiled twice with emscripten:
`moba_sim.wasm` (STANDALONE_WASM, hosted by a Python server via `wasmtime`) and a browser viewer
build (emscripten + raylib) that re-simulates replays. Replay = seed + config + per-tick action log.

**Tech Stack:** C (vendored sim), emscripten, Python 3.11+ (uv project: `aiohttp` or `websockets`,
`wasmtime`), pytest, Docker (linux/amd64), `coworld` package for build/certify.

**Working directory for ALL tasks:** `/Users/daveey/code/cogame-moba` (never the coworld-ctf worktree).

**Reference materials:**
- Pinned upstream clone: `/private/tmp/claude-501/…/scratchpad/pufferlib` (commit `c5d3c637446047a6efbcaa74c039c5295d201ab0`). If missing, re-clone: `git clone --depth 50 https://github.com/PufferAI/PufferLib && git -C PufferLib checkout c5d3c637`.
- Coworld runtime contract + manifest examples: `/Users/daveey/code/coworld-ctf` (`coworld_manifest.json`, `compose.yaml`, `Dockerfile`, `AGENTS.md`) and the Coworld Cookbook (raw Docker shape section shows the exact `COGAME_*` env contract).
- Research brief facts are restated inline where needed; when in doubt, **read the vendored source** — it is the ground truth.

**Global invariants (every task must respect):**
- Vendored upstream files under `vendor/upstream/` are byte-pristine. All changes are patch files in `sim/patches/`, applied at build time into `build/`.
- The 510-byte obs and `[7,7,3,2,2,2]` action encodings are opaque contracts — transported verbatim.
- Fidelity test (Phase 1) must pass after every sim-touching change.
- Env config values (vision_range, agent_speed, reward weights, etc.) must equal upstream training defaults from `config/moba.ini` + `ocean/moba/binding.c` — copy them into one place (`server/cogame_moba/defaults.py` mirrored in `sim/shim.c`) with a comment citing the upstream file.

---

## Phase 0: Scaffold

### Task 0.1: Repo skeleton + uv project

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md` (stub), `server/cogame_moba/__init__.py`, `tests/__init__.py`

**Step 1:** `cd /Users/daveey/code/cogame-moba && uv init --lib --name cogame-moba` then restructure: package lives at `server/cogame_moba/` (set `[tool.hatch.build]`/`[tool.uv]` accordingly or use a flat `packages = ["server/cogame_moba"]` setting in pyproject). Add deps: `uv add wasmtime aiohttp numpy` and dev deps `uv add --dev pytest pytest-asyncio`.

**Step 2:** `.gitignore`: `build/`, `dist/`, `__pycache__/`, `*.wasm` **except** none (wasm artifacts are build outputs, not committed), `.venv/`, `tmp/`.

**Step 3:** Verify: `uv run python -c "import wasmtime, aiohttp; print('ok')"` → `ok`.

**Step 4:** Commit: `git add -A && git commit -m "Scaffold uv project layout"`.

### Task 0.2: Vendor pinned upstream files

**Files:**
- Create: `vendor/upstream/moba.h`, `vendor/upstream/game_map.h`, `vendor/upstream/puffernet.h`, `vendor/upstream/moba_weights.bin`, `vendor/upstream/binding.c`, `vendor/upstream/moba.c`, `vendor/upstream/moba.ini`, `vendor/LICENSE-pufferlib`, `vendor/UPSTREAM.md`

**Step 1:** Copy from the pinned clone: `ocean/moba/{moba.h,game_map.h,binding.c,moba.c}`, `src/puffernet.h`, `resources/moba/moba_weights.bin`, `config/moba.ini`, repo `LICENSE` → `vendor/LICENSE-pufferlib`.

**Step 2:** `vendor/UPSTREAM.md`: record repo URL, commit `c5d3c637446047a6efbcaa74c039c5295d201ab0`, date fetched, file list with sha256 of each file (`shasum -a 256`), and the rule "files here are byte-pristine; changes go in sim/patches/".

**Step 3:** Verify pristineness: `diff vendor/upstream/moba.h <clone>/ocean/moba/moba.h` → no output.

**Step 4:** Commit: `"Vendor PufferLib moba @ c5d3c637 (MIT)"`.

### Task 0.3: Toolchain check

**Step 1:** `emcc --version` (brew-installed emscripten). If missing: `brew install emscripten`. Record version in `vendor/UPSTREAM.md` (build reproducibility note).

**Step 2:** Smoke: compile a hello.c to standalone wasm and run under wasmtime-py:
`emcc -O2 -sSTANDALONE_WASM --no-entry -sEXPORTED_FUNCTIONS=_add -o /tmp/t.wasm x.c` with `int add(int a,int b){return a+b;}`, then a 5-line wasmtime-py script instantiating with WASI and calling `add(2,3)` → `5`. This validates the exact host pattern Phase 1 uses.

**Step 3:** No commit needed (throwaway smoke), but note emcc version in UPSTREAM.md and commit that.

---

## Phase 1: Sim wasm build + Python host + fidelity test

### Task 1.1: Render-guard patch (`0001-render-guard.patch`)

**Files:**
- Create: `sim/patches/0001-render-guard.patch`, `sim/apply_patches.sh`, `vendor/PATCHES.md`

**Step 1:** Read `vendor/upstream/moba.h`; find the boundary between sim (~line 1968) and renderer. Produce a patch that (a) guards `#include "raylib.h"` and the entire renderer section (GameRenderer struct through `c_render`/`c_close` raylib parts) behind `#ifdef MOBA_RENDER`, (b) nothing else. If `c_close` mixes sim frees and render frees, split with the ifdef *inside* the function body.

**Step 2:** `sim/apply_patches.sh`: copies `vendor/upstream/*` → `build/src-pristine/` (no patches beyond 0001, which is required to compile at all) and → `build/src-patched/` (all patches). Both get 0001; "pristine" for the fidelity test means *0001 only*. Document this in `vendor/PATCHES.md` with rationale per patch.

**Step 3:** Verify both trees compile as plain native objects (no link): `cc -c -O2 -x c build/src-pristine/moba.h -o /dev/null` (or via a tiny `#include "moba.h"` TU). Expected: success without raylib installed.

**Step 4:** Commit.

### Task 1.2: Behavior patches (`0002-seed.patch`, `0003-done-flag.patch`)

**Files:**
- Create: `sim/patches/0002-seed.patch`, `sim/patches/0003-done-flag.patch`; update `vendor/PATCHES.md`

**Step 1 (0002):** Add `env->seed` field use: in `init_moba` (or a new `moba_seed(unsigned)` helper) call `srand(seed)`. Cite upstream 3.0's `env_binding.h` precedent in PATCHES.md.

**Step 2 (0003):** In `c_step`'s win branch (ancient health ≤ 0 → currently calls internal reset): set `env->done = 1; env->winner = <team>` and **skip the auto-reset**. Add fields to the MOBA struct. Read the actual code first — the research brief says the win check lives where towers/ancients die (`add_log` then reset); patch precisely there.

**Step 3:** Re-run Task 1.1's compile check on `build/src-patched/`.

**Step 4:** Commit.

### Task 1.3: wasm shim + build script

**Files:**
- Create: `sim/shim.c`, `sim/build_sim.sh`

**Step 1:** Write `sim/shim.c` against the patched tree. Shape (adjust to the real struct/API after reading `moba.h` and `binding.c`):

```c
#include <stdlib.h>
#include "moba.h"          // MOBA_RENDER not defined
#include "game_map.h"      // unsigned char game_map_npy[]

static MOBA env;

// Config defaults MUST mirror upstream config/moba.ini [env] + binding.c my_vec_init.
// (executor: read both, hardcode the trained-on values here, cite them)
__attribute__((export_name("moba_init")))
void moba_init(unsigned int seed, int num_agents) {
    env.num_agents = num_agents;        // 10
    env.vision_range = 5;               // from moba.ini
    /* ... agent_speed, reward_death, reward_xp, reward_tower, script_opponents=0 ... */
    env.seed = seed;
    allocate_moba(&env);                // allocates obs/actions/rewards + ai_paths, calls init_moba
    c_reset(&env);
}
__attribute__((export_name("moba_step")))  void moba_step(void)  { c_step(&env); }
__attribute__((export_name("moba_reset"))) void moba_reset(void) { c_reset(&env); }
__attribute__((export_name("obs_ptr")))    unsigned char* obs_ptr(void) { return env.observations; }
__attribute__((export_name("act_ptr")))    float* act_ptr(void)         { return env.actions; }
__attribute__((export_name("rew_ptr")))    float* rew_ptr(void)         { return env.rewards; }
__attribute__((export_name("moba_done")))  int moba_done(void)   { return env.done; }
__attribute__((export_name("moba_winner")))int moba_winner(void) { return env.winner; }
__attribute__((export_name("moba_tick")))  int moba_tick(void)   { return env.tick; }
// Per-agent score stats: expose whatever Entity fields exist (kills/deaths/level/…):
__attribute__((export_name("agent_stat"))) int agent_stat(int pid, int which);
```

For the *pristine* build, compile the same shim with `-DPRISTINE` guarding out `env.seed/done/winner` references (pristine tree lacks those fields; done/winner return 0).

**Step 2:** `sim/build_sim.sh`:

```bash
emcc -O2 -sSTANDALONE_WASM --no-entry \
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1gb -sABORTING_MALLOC=1 \
  -I build/src-patched sim/shim.c -o build/moba_sim.wasm
emcc -O2 -sSTANDALONE_WASM --no-entry -DPRISTINE \
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=1gb -sABORTING_MALLOC=1 \
  -I build/src-pristine sim/shim.c -o build/moba_sim_pristine.wasm
```

(`ABORTING_MALLOC` per the org's wasm-viewer OOM lesson: fail loudly, never NULL-write. 256 MB ai_paths + growth headroom fits under 1 GB.)

**Step 3:** Run it. Expected: two `.wasm` files. Failure modes to chase: emscripten missing `rand` (it won't be), undefined raylib symbols (patch 0001 incomplete), export name mismatches.

**Step 4:** Commit (scripts + patches only; `build/` is gitignored).

### Task 1.4: Python wasmtime host (`MobaSim`)

**Files:**
- Create: `server/cogame_moba/sim.py`
- Test: `tests/test_sim.py`

**Step 1: Failing test first:**

```python
import numpy as np
from cogame_moba.sim import MobaSim

NOOP = [3, 3, 0, 0, 0, 0]

def test_step_shapes():
    sim = MobaSim.load(seed=7)
    obs = sim.observations()
    assert obs.shape == (10, 510) and obs.dtype == np.uint8
    sim.set_actions(np.array([NOOP] * 10, dtype=np.float32))
    sim.step()
    assert sim.tick() == 1

def test_determinism():
    def run(seed):
        sim = MobaSim.load(seed=seed)
        rng = np.random.default_rng(0)
        h = []
        for _ in range(300):
            acts = np.stack([rng.integers(0, [7,7,3,2,2,2]) for _ in range(10)]).astype(np.float32)
            sim.set_actions(acts); sim.step()
            h.append(sim.observations().tobytes())
        return b"".join(h)
    assert run(7) == run(7)
    assert run(7) != run(8)   # seeding actually does something
```

**Step 2:** Run: `uv run pytest tests/test_sim.py -v` → FAIL (no module).

**Step 3:** Implement `MobaSim`: wasmtime `Engine/Store/Module/Linker` + `linker.define_wasi()`, `WasiConfig` with inherited stdout (sim printfs), instantiate, call `moba_init(seed, 10)`. `observations()` reads 10×510 bytes from exported memory at `obs_ptr()` into numpy (fresh copy per call); `set_actions` writes 60 float32 at `act_ptr()`; expose `rewards()`, `done()`, `winner()`, `tick()`, `agent_stat()`. Constructor arg `wasm_path` defaulting to `build/moba_sim.wasm` (resolve relative to repo root).

**Step 4:** Tests pass. **Step 5:** Commit.

### Task 1.5: Fidelity test (THE acceptance gate)

**Files:**
- Test: `tests/test_fidelity.py`

**Step 1:**

```python
def test_patched_matches_pristine():
    a = MobaSim.load(seed=1, wasm_path=PATCHED)     # patch 0002 srand(1) == pristine default seed 1
    b = MobaSim.load(seed=1, wasm_path=PRISTINE)
    rng = np.random.default_rng(42)
    for t in range(5000):
        acts = rng.integers(0, [7,7,3,2,2,2], size=(10, 6)).astype(np.float32)
        for s in (a, b): s.set_actions(acts); s.step()
        assert a.observations().tobytes() == b.observations().tobytes(), f"obs diverged at tick {t}"
        assert a.rewards().tobytes() == b.rewards().tobytes(), f"rewards diverged at tick {t}"
        if a.done(): break   # pristine auto-resets on win; stop comparing there
```

Load-bearing subtlety: `srand(1)` must reproduce the C default-seed stream — verify (musl: yes; if the first-episode streams differ, compare pristine against patched-with-no-srand-call instead and test 0002 separately for variety only).

**Step 2:** Run → must PASS. If it fails, a patch changed physics — fix the patch, not the test.

**Step 3:** Commit. Mark this test as the CI gate in the test's docstring.

---

## Phase 2: Game server (Coworld contract, lockstep)

Reference the Cookbook "Raw Docker Shape" for the exact env-var contract and coworld-ctf's `config.json` for config shape conventions.

### Task 2.1: Config + results models

**Files:** `server/cogame_moba/config.py`, `server/cogame_moba/defaults.py`; test `tests/test_config.py`

Game config JSON (arrives via `COGAME_CONFIG_URI`): `{seed?, max_ticks (default 40000), heroes_per_seat (1 or 5), tick_deadline_ms (default 1000), players: [{name}], tokens: [...]}`. Validate `len(players)*heroes_per_seat == 10`. TDD: parse/validate/defaults tests, then implement. Commit.

### Task 2.2: Lockstep engine (transport-free core)

**Files:** `server/cogame_moba/engine.py`; test `tests/test_engine.py`

Pure-async class: given a `MobaSim` and N seat queues, per tick: emit per-seat obs (slice heroes for team variant), await each seat's action with `tick_deadline_ms` timeout → missing/late/malformed = NOOP `[3,3,0,0,0,0]` per hero; step; accumulate per-seat rewards; stop on `done` or `max_ticks`. Result object: winner (0/1/None draw — on tick-cap draw, break ties by ancient HP via `agent_stat`/exported ancient health; if equal, true draw), per-seat score (win 1 / draw 0.5 / loss 0), stats. TDD with fake sim, then with the real wasm sim + scripted action fns. Commit per red/green.

### Task 2.3: Websocket server + COGAME contract

**Files:** `server/cogame_moba/server.py`, `server/cogame_moba/uris.py`; test `tests/test_server.py`

- aiohttp app: `GET /player` websocket (query `slot`, `token` — validate against config tokens), health route.
- Protocol (as implemented): server→player `{"tick": t, "obs": [b64(510B)×heroes]}` (no per-tick `done` field); final message `{"done": true, "result": {...}}`. player→server `{"tick": t, "actions": [[6 ints]×heroes]}`. Reject/NOOP wrong-tick or malformed messages (never crash the episode).
- `uris.py`: read/write `file://` and `http(s)://` URIs for `COGAME_CONFIG_URI`, `COGAME_RESULTS_URI`, `COGAME_SAVE_REPLAY_URI`, `COGAME_PLAYER_FAILURE_URI` (mirror how coworld-ctf/paintarena treat them; local file:// is enough for tests, http PUT for hosted).
- `player_connect_timeout_seconds` honored; a seat that never connects → episode proceeds with NOOPs and gets reported to `COGAME_PLAYER_FAILURE_URI`.
- Entry point: `python -m cogame_moba.server`.
- Tests: end-to-end in-process — start server on a random port with a temp config, connect 10 trivial ws clients sending random actions, assert episode completes, `results.json` written with 10 scores summing correctly, replay file exists. Also test the 2×5 team variant. Commit.

### Task 2.4: Replay writer + reader

**Files:** `server/cogame_moba/replay.py`; test `tests/test_replay.py`

Format v1 (as implemented): `MOBA` magic + u8 version + u32le header_len + header JSON (format_version, sim_wasm_sha256, full game config incl. seed and player names — tokens excluded, final result, **tick_count in the header JSON** — no trailer) + packed per-tick actions (10×6 uint8, post-clamp). Writer buffers the body in memory (episodes are ≤40000×60B = 2.4 MB) and renders the whole file at `finalize(result)`; `append_tick(tick, actions)` matches the engine's on_tick hook and validates tick sequentiality. Reader validates magic/version/lengths and iterates. Round-trip test + "reader re-sim reaches same winner/tick/final obs as recorded" test (uses MobaSim). Commit.

### Task 2.5: Replay-mode serving

**Files:** modify `server/cogame_moba/server.py`

When `COGAME_LOAD_REPLAY_URI` is set: don't run an episode; serve the static viewer bundle at `/client/replay` (bundle built in Phase 4 — until then, a placeholder page that loads the replay bytes at `/replay-data` and shows header JSON). Serve raw replay bytes at `/replay-data`. Test: start in replay mode with a recorded file, GET both routes. Commit.

---

## Phase 3: Players

### Task 3.1: Player client library + random player

**Files:** `players/client.py` (shared ws client: reads `COWORLD_PLAYER_WS_URL`, decode obs, send actions, reconnect-safe), `players/random_player.py`; test `tests/test_players.py` (random player completes an episode against the in-process server). Commit.

### Task 3.2: Baseline player (upstream weights via wasm puffernet)

**Files:** `sim/brain_shim.c`, extend `sim/build_sim.sh` (→ `build/moba_brain.wasm`), `players/baseline_player.py`; test `tests/test_baseline.py`

- `brain_shim.c`: include vendored `puffernet.h`; embed or load `moba_weights.bin` (embed via `xxd -i` at build time — simplest, no FS in wasm); exports: `brain_init()` (make_puffernet per upstream `moba.c`: `load_weights`, `make_puffernet(weights, 5, 510, 64, 5, logit_sizes, 6)` — copy the exact call from vendored `moba.c`), `brain_forward(agent_idx)` with obs-in/actions-out pointers; one recurrent state per agent index (10 states so one process can serve a whole team seat).
- `baseline_player.py`: hosts brain wasm via wasmtime; per tick: obs bytes → float array → forward → 6 ints → send.
- Test (slow, marked): baseline seats (5, team red) vs random seats (5): baseline wins ≥ 4 of 5 episodes with a modest `max_ticks`. If scripted-creep stomps make episodes long, cap ticks and score by ancient HP.
- Commit.

---

## Phase 4: Static replay viewer

### Task 4.1: Viewer wasm build (emscripten + raylib)

**Files:** `sim/viewer_main.c`, `sim/build_viewer.sh`, `viewer/index.html`

- Study upstream `build.sh`'s `--web` recipe (raylib via emscripten; upstream links a prebuilt web raylib — replicate: fetch raylib 5.x source, `make PLATFORM=PLATFORM_WEB` once into `build/raylib-web/`, cache it).
- `viewer_main.c`: compile patched tree **with** `-DMOBA_RENDER`. Main loop: JS passes replay bytes into wasm memory (`viewer_load(ptr,len)` parses header+actions); each render frame, advance the sim per a tick-schedule (12 render frames per sim tick like upstream demo, scaled by a playback-speed variable), feed recorded actions, call `c_render`. Exports: `viewer_load`, `viewer_seek(tick)` (re-sim from 0 — sim is cheap), `viewer_set_speed(x)`, `viewer_tick()`.
- `viewer/index.html`: canvas + minimal controls (play/pause, speed, seek bar, tick counter), HTML overlay listing seat/player names from the replay header (names come from replay bytes per the static-viewer contract). Fetches `/replay-data` by default; also accepts `?replay=<url>`.
- Assets: copy `resources/moba/{moba_assets.png,dota_map.png,*.glsl}` from the pinned clone into `viewer/assets/` (add to Task 0.2's vendor manifest) — the emscripten build preloads them (`--preload-file`).
- Check: `sim/build_viewer.sh` produces `viewer/dist/` (html+js+wasm+data). Manual check: `python -m http.server` in a dir with a recorded replay, verify playback renders and reaches the recorded winner. Automated check: headless — run `viewer_seek(end)` path under node (`emcc -sENVIRONMENT=web,node`) asserting final tick/winner match the header. Commit.

### Task 4.2: Wire bundle into server + replay smoke

Replace Task 2.5 placeholder with `viewer/dist/`. End-to-end test: record episode → serve replay mode → fetch `/client/replay` (bundle HTML) and `/replay-data`; node-based re-sim assertion from 4.1 runs against this replay. Commit.

---

## Phase 5: Packaging, CI, GitHub, certification

### Task 5.1: Dockerfile + compose + manifest template

**Files:** `Dockerfile`, `compose.yaml`, `coworld_manifest_template.json`, `config.json` (dev fixture)

- Single image (paintarena/coworld-ctf pattern): build stage runs emscripten builds (use `emscripten/emsdk` image) → runtime stage `python:3.11-slim` with `server/`, `players/`, `build/*.wasm`, `viewer/dist/`. Game entrypoint `python -m cogame_moba.server`; player entrypoints `python -m players.baseline_player` / `players.random_player`. `--platform=linux/amd64`.
- Manifest template: crib structure from coworld-ctf's `coworld_manifest.json` / paintarena's template — game runnable, certification fixture (10 baseline players), variants: `default` (10 seats ×1), `team` (2 seats ×5), static replay viewer bundle declaration, `source_url` → the GitHub repo.
- Check: `docker build --platform=linux/amd64 -t cogame-moba:local .` then the Cookbook raw-Docker smoke (game + 10 random players on a docker network) completes and writes artifacts. Commit.

### Task 5.2: coworld build + certify locally

- `uv run coworld build --project . --version 0.1.0` (adjust to this repo's layout; coworld-ctf shows the shape), then `uv run coworld certify dist/coworld_manifest.json --no-open-report`.
- Fix whatever the certifier flags (results schema, replay probe, player launch). Watch the replay once via the printed command per Cookbook guidance. Commit fixes.

### Task 5.3: CI + GitHub publish

**Files:** `.github/workflows/ci.yml`, `.github/workflows/coworld-upload.yml`, `README.md`, `AGENTS.md`

- CI: setup emsdk (`mymindstorm/setup-emsdk`), `uv sync`, build sim wasms, `uv run pytest` (fidelity + determinism + server tests; mark docker/slow tests to skip in CI or run in a second job).
- Upload workflow: the shared `Metta-AI/metta/.github/workflows/coworld-manual-upload.yml@main` reusable workflow, `workflow_dispatch`-only, default `confirm_upload: dry-run` (Cookbook rule).
- README: what it is, the fidelity guarantee and its test, quickstart (build wasm, run tests, run local episode, watch replay), protocol spec for policy authors, attribution to PufferAI (MIT).
- AGENTS.md: repo conventions for future agents (pristine-vendor rule, fidelity gate, where defaults live).
- Create repo `gh repo create Metta-AI/cogame-moba --public --source . --push` (public — it references only MIT upstream). Verify CI is green on GitHub; fix until green. 

### Task 5.4: Final verification sweep

Run the full local suite + one `coworld run-episode` with baseline players + open one replay in a browser. Confirm: results scored, replay plays, fidelity test green, CI green. Only then report done (superpowers:verification-before-completion).

---

---

## Phase 6: Hosted deployment (CoMOBA league, Pufferlib player)

Authorized by daveey 2026-08-01: mirror coworld-ctf's deployment. Prereqs: Phases 4–5 complete
(certified, uploaded to GitHub, CI green).

### Task 6.1: Hosted upload + push-to-main CI upload

- One-time local `uv run coworld upload-coworld dist/coworld_manifest.json` (reuses the certify
  proof) so the coworld exists and is canonical. Record the `cow_...` id.
- `.github/workflows/upload-coworld.yml` mirroring coworld-ctf's (same shape: push-to-main +
  workflow_dispatch with explicit version input, concurrency group, version = highest existing
  registry row patch-bumped via a `tools/ci/next_coworld_version.py` equivalent — NOT
  `coworld next-version` (orphan-row 409 wedge, see coworld-ctf's workflow comment)).
- Auth: repo secret `SOFTMAX_TOKEN` = a non-expiring CI credential (pattern: coworld-ctf's
  "coworld-ctf-ci" from softmax.com/observatory/credentials). If credential minting has no
  CLI/API path, ask the user to mint "cogame-moba-ci" and set the secret.

### Task 6.2: CoMOBA platform-ladder league

Follow the platform-ladder-league onboarding doc (fetched via cogtext; platform commissioner,
NOT a container commissioner — no commissioner runnable in the manifest):

1. `POST /v2/coworld-league-seeds` `{"coworld_name": <name>, "template": "commissioner_driven",
   "enabled": true, "overrides": {"commissioner_key": "platform"}}` (team token,
   `X-Use-Elevated-Privileges: true`). League display name: **CoMOBA**.
2. Declare divisions: single `Competition` (level 1).
3. Ladder settings (POST replaces whole doc — GET first, preserve siblings): strategy
   **`team_pair`** on the 10-seat variant (two champions × 5 cloned seats = 5v5 mirror,
   exactly the MOBA shape), `insufficient_players: "multiple_seats"` (so the league runs
   self-play mirrors while Pufferlib is the only player), ranking `elo`, write with
   `enabled: false` first, review, then re-POST `enabled: true`.
4. Unpause, `trigger-round`, verify: Temporal workflow `ladder-{league_id}`, a Competition
   round with frozen episode_plan, episodes complete, leaderboard publishes, replay opens.

### Task 6.2b: Scripted agent (added 2026-08-01)

`players/scripted_player.py` — a hand-coded (non-RL) policy that plays the MOBA, structural
inspiration from coworld-ctf's Nim scripted bots (read-only reference), implemented in Python on
`players/client.py`. Ground it in the vendored sim source (obs encoding from
`compute_observations` — mind the crop's stride-overlap quirk when deciding what is reliably
decodable; WAYPOINTS tables; skill cooldown semantics; tile-id constants). Baseline shape:
lane-push via embedded map waypoints using absolute self-position from the obs, attack-filter
actions, skills on cooldown when enemies are near, retreat-to-heal when low HP. Tests: valid
actions always; behavioral — scripted team beats random team (and report how it fares vs the
pretrained baseline, no pass/fail bar there). Submitted to CoMOBA as **daveey's own player**
(not the Pufferlib identity): `upload-policy` + `submit` with the user's default credentials.

### Task 6.3: Pufferlib player + policy + token in Secrets Manager

- Create a player identity named **Pufferlib** owned by daveey's account (check
  `uv run coworld player --help` / `softmax player --help` for the create command; if creation
  is UI-only, stop and ask).
- As that player (`coworld player use ply_...`): `upload-policy` the baseline player image
  (upstream weights via wasm puffernet) under name `pufferlib-baseline`, then
  `coworld submit pufferlib-baseline --league <CoMOBA league_...>`.
- Store the player's credential/token in AWS Secrets Manager, profile `softmax-org`, secret id
  `comoba/pufferlib-player-token` (value: the token + a JSON note of player id/name). The
  account will later be handed to an external user; the token must not live only in
  ~/.softmax/credentials.yaml.
- Verify placement: `coworld submissions --mine --league ... --json` → placed; membership
  active; next round seats the policy.

## Deviations log

- Baseline player inference: design doc said "numpy reimplementation"; plan uses upstream `puffernet.h` compiled to wasm instead (more faithful, less code). Deliverable unchanged.
- Hosted upload/league setup: originally out of scope; authorized and specified as Phase 6 on 2026-08-01 (CoMOBA league, Pufferlib player, CI auto-upload mirroring coworld-ctf).
