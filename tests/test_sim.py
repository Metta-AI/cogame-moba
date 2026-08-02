import numpy as np

from cogame_moba.sim import MobaSim

NOOP = [3, 3, 0, 0, 0, 0]  # MultiDiscrete [7,7,3,2,2,2] center: stand still
ACT_HIGH = [7, 7, 3, 2, 2, 2]


def test_step_shapes():
    sim = MobaSim.load(seed=7)
    obs = sim.observations()
    assert obs.shape == (10, 510)
    assert obs.dtype == np.uint8
    # obs is a fresh copy, not a view into wasm linear memory
    obs2 = sim.observations()
    assert obs2 is not obs

    rew = sim.rewards()
    assert rew.shape == (10,)
    assert rew.dtype == np.float32

    assert sim.tick() == 0
    assert sim.done() == 0

    sim.set_actions(np.array([NOOP] * 10, dtype=np.float32))
    sim.step()
    assert sim.tick() == 1

    # per-agent stats: everyone starts at level 1, 0 kills/deaths
    assert sim.agent_stat(0, 0) == 1
    assert sim.agent_stat(0, 1) == 0
    # both ancients start at full health (4500, TOWER_HEALTH[22..23])
    assert sim.ancient_health(0) == 4500.0
    assert sim.ancient_health(1) == 4500.0


def _run(seed, ticks=300):
    sim = MobaSim.load(seed=seed)
    rng = np.random.default_rng(0)
    chunks = []
    for _ in range(ticks):
        acts = rng.integers(0, ACT_HIGH, size=(10, 6)).astype(np.float32)
        sim.set_actions(acts)
        sim.step()
        chunks.append(sim.observations().tobytes())
        chunks.append(sim.rewards().tobytes())
    return b"".join(chunks)


def test_determinism_same_seed():
    assert _run(7) == _run(7)


def test_seed_changes_stream():
    # proves patch 0002 (srand seeding) actually does something
    assert _run(7) != _run(8)
