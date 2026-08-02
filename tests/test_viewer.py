"""Viewer verification without a browser (Task 4.2).

Two layers:

- build outputs: sim/build_viewer.sh artifacts exist (skips with a clear
  message when the emscripten viewer build hasn't been run);
- headless re-sim: the viewer core (viewer_main.c compiled WITHOUT
  MOBA_RENDER, ENVIRONMENT=node) loads a real recorded replay under node
  and must reach the header's tick_count with the sim's winner matching
  the recorded result — proving the viewer's replay parsing and
  step-scheduling logic with no pixels involved.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cogame_moba import defaults, replay
from cogame_moba.config import GameConfig
from cogame_moba.engine import LockstepEngine
from cogame_moba.replay import ReplayWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_DIST = REPO_ROOT / "viewer" / "dist"
VIEWER_CORE_JS = REPO_ROOT / "build" / "viewer_core.js"
HARNESS = Path(__file__).parent / "viewer_core_harness.js"

NOT_BUILT = "viewer not built - run sim/build_viewer.sh first"


def test_build_viewer_outputs_exist():
    if not VIEWER_CORE_JS.exists():
        pytest.skip(NOT_BUILT)
    for name in ("index.html", "moba_viewer.js", "moba_viewer.wasm",
                 "moba_viewer.data"):
        assert (VIEWER_DIST / name).exists(), f"viewer/dist/{name} missing"
    assert (REPO_ROOT / "build" / "viewer_core.wasm").exists()


async def _record_replay(tmp_path: Path):
    """Record a real wasm episode (like tests/test_replay.py does)."""
    from cogame_moba.sim import MobaSim

    class RngSource:
        def __init__(self, seat):
            self.rng = np.random.default_rng(4000 + seat)

        async def get_actions(self, tick, obs):
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(1, 6)).tolist()

    cfg = GameConfig.from_dict({
        "players": [{"name": f"hero-{i}"} for i in range(10)],
        "tokens": [f"tok{i}" for i in range(10)],
        "seed": 424242,
        "max_ticks": 120,
        "tick_deadline_ms": 2000,
    })
    sim = MobaSim(seed=cfg.seed)
    writer = ReplayWriter(cfg, replay.sim_wasm_sha256())
    engine = LockstepEngine(
        sim, cfg, [RngSource(s) for s in range(10)],
        on_tick=writer.append_tick)
    result = await engine.run()
    data = writer.finalize({
        "winner": result.winner,
        "end_reason": result.end_reason,
        "final_tick": result.final_tick,
    })
    path = tmp_path / "replay.bin"
    path.write_bytes(data)
    return path, result, sim


async def test_headless_core_resimulates_recorded_replay(tmp_path):
    if not VIEWER_CORE_JS.exists():
        pytest.skip(NOT_BUILT)
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH")

    replay_path, result, sim = await _record_replay(tmp_path)

    proc = subprocess.run(
        [node, str(HARNESS), str(VIEWER_CORE_JS), str(replay_path)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"harness failed:\n{proc.stderr}"
    out = json.loads(proc.stdout)

    # replay body parse: C-side tick count == header tick count
    assert out["total"] == result.final_tick
    assert out["headerTickCount"] == result.final_tick

    # frame scheduling: 1 tick / 12 frames at 1x, 4 at 4x, none paused
    assert out["cadence1"] == 1
    assert out["cadence4"] == 4
    assert out["pausedTicks"] == 0

    # seek: mid lands exactly, end reaches tick_count and pauses (no loop)
    assert out["midTick"] == result.final_tick // 2
    assert out["endTick"] == result.final_tick
    assert out["playingAtEnd"] == 0
    assert out["playAtEndRefused"] == 1

    # the re-simulated episode reproduces the recorded outcome
    assert out["done"] == sim.done()
    if sim.done():
        assert out["winner"] == result.winner
