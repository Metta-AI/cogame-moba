# cogame-moba

A [Coworld](https://softmax.com) game that runs PufferLib's Ocean MOBA
("Puffer MOBA") with bit-exact observation/action spaces and physics, so
policies RL-trained on the original environment play identically when
submitted to a Coworld league.

The upstream C sim ([PufferAI/PufferLib](https://github.com/PufferAI/PufferLib)
@ `c5d3c637`, MIT) is vendored pristine under `vendor/upstream/`, patched at
build time (`sim/patches/`), and compiled to WebAssembly with emscripten. A
Python server hosts the wasm sim via `wasmtime` and referees lockstep
websocket episodes; replays re-simulate deterministically in a static wasm
browser viewer.

## The fidelity guarantee

The whole point of this port is that the served environment **is** the
trained-on environment. The gate is `tests/test_fidelity.py`: it runs the
patched production sim and a pristine (behavior-patch-free) build of the
vendored source side by side for thousands of ticks of identical random
actions and asserts every observation and reward byte matches. Sim-touching
changes must keep this test green; the test itself is inviolable — fix the
patch, never the test.

## Quickstart

```sh
uv sync
bash sim/build_sim.sh      # sim wasm (requires emcc; brew install emscripten)
bash sim/build_brain.sh    # baseline-policy brain wasm (requires xxd)
bash sim/build_viewer.sh   # browser replay viewer (downloads pinned raylib)
uv run pytest              # full suite, includes the fidelity gate
```

Run a local containerized episode (Docker required):

```sh
docker build --platform=linux/amd64 -t cogame-moba:local .
uv run coworld build --project . --version 0.1.0
uv run coworld run-episode dist/coworld_manifest.json
```

Watch a recorded replay:

```sh
uv run coworld replay dist/coworld_manifest.json <path/to/replay>
# or directly: COGAME_LOAD_REPLAY_URI=file://<replay> python -m cogame_moba.server
```

## The game

5v5 lane-pushing MOBA on a fixed map: two teams (radiant, pids 0-4; dire,
pids 5-9) of five fixed-role heroes (support, assassin, burst, tank, carry)
push creep waves down three lanes through enemy towers to destroy the
opposing Ancient. Heroes gain XP/levels, respawn on death, and have three
skills on cooldown. Episodes end when an Ancient falls or at the tick cap
(ties break by remaining Ancient health; equal is a draw).

Two variants: `default` (10 seats x 1 hero) and `team` (2 seats x 5 heroes).

## Protocol (for policy authors)

See [docs/PROTOCOL.md](docs/PROTOCOL.md). Short version: connect to the
websocket URL in `COWORLD_PLAYER_WS_URL`; each tick the server sends
`{"tick", "obs": [base64 x heroes]}` (each blob is the upstream 510-byte
observation: 11x11x4 map crop + 26 scalar bytes, per `compute_observations`
in `vendor/upstream/moba.h`) and you reply `{"tick", "actions": [[6 ints] x
heroes]}` in the upstream MultiDiscrete `[7,7,3,2,2,2]` action space. Late,
missing, or malformed replies play no-op; the encodings are transported
verbatim from upstream.

## Players

| player | command | what it is |
| --- | --- | --- |
| random | `python -m players.random_player` | uniform-random in-range actions |
| baseline | `python -m players.baseline_player` | upstream pretrained weights (`moba_weights.bin`) through vendored `puffernet.h` compiled to wasm — the bundled certification player (needs `build/moba_brain.wasm`) |
| scripted | `python -m players.scripted_player` | hand-coded lane-push FSM over the decoded obs (pure Python + aiohttp) |

## Repo layout

- `vendor/upstream/` — byte-pristine vendored upstream source (never edit;
  see `vendor/UPSTREAM.md`); all changes are patch files in `sim/patches/`
  (rationale in `vendor/PATCHES.md`)
- `sim/` — patches, wasm shims, build scripts (`apply_patches.sh`,
  `build_sim.sh`, `build_brain.sh`, `build_viewer.sh` -> `build/`, `viewer/dist/`)
- `server/cogame_moba/` — config, lockstep engine, websocket server, replay
  writer/reader, wasmtime sim host; entry `python -m cogame_moba.server`
- `players/` — see table above
- `tests/` — the full suite; `tests/test_fidelity.py` is the acceptance gate
- `Dockerfile`, `compose.yaml`, `coworld_manifest_template.json` — Coworld
  packaging (`uv run coworld build --project .`)

## Porting other PufferLib envs

This repo doubles as a worked example. [docs/PORTING.md](docs/PORTING.md) is
the recipe for turning another PufferLib Ocean env into a Coworld the same
way.

## Attribution

The simulation, renderer, map assets, network weights, and puffernet
inference library are from [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib),
pinned at commit `c5d3c637446047a6efbcaa74c039c5295d201ab0`, MIT license
(`vendor/LICENSE-pufferlib`). This repo adds the Coworld packaging around
them.
