# cogame-moba — Design

A Coworld that runs PufferLib's Ocean MOBA ("Puffer MOBA", https://puffer.ai/ocean.html) with
**bit-exact observation/action spaces and physics**, so policies RL-trained on the original
environment play identically when submitted to a Coworld league.

Validated with daveey on 2026-08-01. Decisions recorded here are settled; implementation follows
this document without re-litigating them.

## Decisions

| Decision | Choice |
|---|---|
| Sim core | Vendored upstream C sim compiled to **WebAssembly**, used everywhere (server + viewer) |
| Upstream pin | `PufferAI/PufferLib` @ `c5d3c637446047a6efbcaa74c039c5295d201ab0` (branch 4.0, MIT license) |
| Repo | `Metta-AI/cogame-moba` on GitHub |
| Seats | Default variant: **10 seats × 1 hero**; `team` variant: **2 seats × 5 heroes** |
| Pacing | **Pure lockstep, as fast as possible** — no live human play; browser is replay-only |
| Server | Python websocket game server implementing the Coworld runtime contract, hosting the sim via `wasmtime` |
| Replay | Seed + config + per-tick action log; static wasm viewer bundle re-simulates for playback |

## Why wasm

The sim (`ocean/moba/moba.h`) is a single MIT C header using only libc/libm. Its only
portability hazard is libc `rand()` (glibc vs musl vs macOS produce different streams). Compiling
to wasm freezes one musl `rand()`, one float semantics, and one binary that behaves identically on
the hosted k8s runner, in CI, on dev Macs, and in the browser viewer. Upstream already ships an
emscripten web build (`build.sh moba --web`), so this is a supported path.

## Upstream contract being preserved

- **Observation**: 510 × uint8 per agent = 11×11×4 map-crop block (484) + 26 self-features.
  The crop writer uses stride 1 where 4 was intended, overlapping neighbor slots; the shipped
  pretrained weights were trained on this exact layout. We treat the 510 bytes as an **opaque
  contract** — transported verbatim, never "fixed".
- **Action**: MultiDiscrete `[7, 7, 3, 2, 2, 2]` per agent — vel_y, vel_x, target-filter,
  use_q, use_w, use_e. Delivered to the sim as 6 floats per agent, exactly as upstream decodes.
- **Agents**: 10 (5v5 mirror; roles support/assassin/burst/tank/carry per team).
  `script_opponents=0` so all 10 heroes are seat-controlled.
- **Tick**: `step_neutrals → step_creeps → step_towers → step_players`, tick-counted cooldowns,
  creep waves every 150 ticks, neutral camps every 600, no clocks or threads in the sim.
- **Map**: 128×128 uint8 grid compiled in from upstream `game_map.h`.
- **RNG**: libc `rand()` (direct + a 10k-entry cached table). Deterministic given seed + actions.

## Vendored patch set (minimal, documented, each with rationale)

Patches live as diffs in `vendor/PATCHES.md` and must keep in-episode physics byte-identical:

1. **Sim/render split** — guard the raylib renderer half of `moba.h` behind `#ifdef MOBA_RENDER`
   so the server-side sim build needs no raylib. Mechanical; no sim lines change.
2. **Done flag instead of auto-reset** — upstream `c_step` silently calls `c_reset` when an
   ancient dies and never writes `terminals`. We expose `env->done` + winning team and skip the
   internal reset, so the server can end the episode and score it. Affects only post-win behavior.
3. **Explicit seeding** — upstream 4.0 never calls `srand`, so every run replays seed-1. We add
   `srand(seed)` at init (matching upstream 3.0's behavior). Per-episode variety; in-episode
   physics unchanged.
4. **Tick cap** — the env has no truncation. Server-side config `max_ticks` (default generous,
   e.g. 40,000) truncates stalemates; scored as a draw unless ancients' HP differ.

**Fidelity invariant (acceptance test, CI-enforced)**: compile the *unpatched* upstream sim and
the patched sim to wasm; drive both with identical seed and recorded action logs for full
episodes; require **byte-identical 510-byte obs streams and reward streams**. A patch that breaks
this is rejected.

## Architecture

```
             ┌────────────────────────── cogame-moba repo ──────────────────────────┐
             │ vendor/pufferlib-moba/   pinned moba.h, game_map.h, puffernet.h, ... │
             │ sim/                     wasm builds: moba_sim.wasm (standalone),    │
             │                          viewer build (emscripten + raylib)         │
             │ server/                  Python: coworld contract + wasmtime host    │
             │ viewer/                  static replay bundle (html + wasm)          │
             │ players/                 baseline (puffer weights), random           │
             └──────────────────────────────────────────────────────────────────────┘
```

- **`moba_sim.wasm`** — emscripten `STANDALONE_WASM` build of the patched sim. Exports:
  init(seed, config), reset, step, and pointers into linear memory for obs (10×510 u8),
  actions (10×6 f32), rewards (10 f32), done/winner, plus per-agent stat counters for scoring.
  The 256 MB `ai_paths` BFS cache lives inside wasm linear memory (wasm32 max 4 GB — fine).
- **Game server** (Python, `server/`) — implements the standard Coworld game contract:
  `COGAME_CONFIG_URI`, `COGAME_RESULTS_URI`, `COGAME_SAVE_REPLAY_URI`,
  `COGAME_PLAYER_FAILURE_URI`, `COGAME_LOAD_REPLAY_URI` (replay mode), player websockets at
  `/player?slot=&token=`. Lockstep loop: broadcast per-seat obs, await all seats' actions with a
  per-tick deadline (config; deadline miss → no-op action `[3,3,0,0,0,0]` = stand still), step,
  repeat. No frame pacing — ticks run as fast as the slowest seat.
- **Player protocol** — JSON-over-WS, one message per tick each way:
  server→player `{tick, obs: [base64 510B × heroes_per_seat], done, ...}`;
  player→server `{tick, actions: [[6 ints] × heroes_per_seat]}`.
  10-seat variant: 1 hero per seat. Team variant: 5 heroes per seat, batched in role order.
- **Replay** — file = header (format version, sim build hash, config incl. seed, player/seat
  names — names must live in the replay bytes per the static-viewer contract) + per-tick packed
  actions (10×6 small ints). The **static viewer bundle** (declared in the manifest, per the
  Coworld static replay-viewer contract) loads the replay, re-simulates in the same wasm sim, and
  renders with upstream's own emscripten raylib renderer; scrub via re-sim from start plus
  periodic state snapshots if needed.
- **Scoring/results** — winner team from ancient kill (or draw on tick cap). `results.json`
  per-seat scores: win=1/loss=0/draw=0.5 (team-shared), plus per-agent stats (kills, deaths,
  level, tower kills) as auxiliary stats.

## Players

- **`players/baseline`** — loads upstream `moba_weights.bin` (95,616 float32 params:
  510→64 encoder, 5×minGRU(64), 24-logit decoder) reimplemented in numpy (~100 lines), argmax
  decode per upstream's puffernet path. This is the proof artifact that trained puffer policies
  drop in unchanged, and serves as the certification fixture player.
- **`players/random`** — uniform random actions; smoke-test opponent.

## Variants & league shape

- Manifest default variant: 10 seats (leagues mix 10 policies; team_n multiple-of-team-count
  seat rule applies as in paintbot).
- `team` variant: 2 seats × 5 heroes.
- Same sim, same wasm, same replay format for both; only seat batching differs.

## Testing

1. **Fidelity test** (the invariant above) — patched vs unpatched wasm, byte-identical streams.
2. **Determinism test** — same seed + action log twice → identical replay bytes and results.
3. **Server contract test** — headless episode via `coworld run-episode` shape: config in,
   results + replay out, player failure reporting on disconnect.
4. **Baseline sanity** — baseline (weights) beats random seats decisively over N episodes.
5. **Replay/viewer check** — viewer re-sim of a recorded episode reaches the same final state
   (winner, tick count) as the live run.

## Out of scope (deliberate)

- Live human/browser play (lockstep-only decision).
- Any "fixing" of upstream quirks (obs stride, unused reward_distance) — fidelity beats hygiene.
- GPU/vectorized training paths from pufferlib 4.0 — we host one env instance per episode.
