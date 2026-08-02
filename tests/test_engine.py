"""Tests for the transport-free lockstep episode engine."""

import asyncio

import numpy as np
import pytest

from cogame_moba import defaults
from cogame_moba.config import GameConfig
from cogame_moba.engine import EpisodeResult, LockstepEngine

NOOP = list(defaults.NOOP_ACTION)


def make_config(num_seats=10, **overrides):
    heroes = 10 // num_seats
    d = {
        "players": [{"name": f"p{i}"} for i in range(num_seats)],
        "tokens": [f"t{i}" for i in range(num_seats)],
        "heroes_per_seat": heroes,
        "seed": 5,
        "max_ticks": 10,
        "tick_deadline_ms": 200,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


class FakeSim:
    """Records actions fed per tick; obs row i is filled with byte value i."""

    def __init__(self, done_at=None, winner_team=0,
                 ancient_healths=(100.0, 100.0)):
        self._tick = 0
        self.done_at = done_at
        self.winner_team = winner_team
        self.ancient_healths = list(ancient_healths)
        self.fed_actions = []  # list of (10, 6) float32 arrays

    def observations(self):
        return np.tile(
            np.arange(10, dtype=np.uint8).reshape(10, 1), (1, 510))

    def set_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape == (10, 6)
        self.fed_actions.append(actions.copy())

    def step(self):
        self._tick += 1

    def rewards(self):
        # hero pid p earns reward p+1 each tick
        return np.arange(1, 11, dtype=np.float32)

    def done(self):
        return int(self.done_at is not None and self._tick >= self.done_at)

    def winner(self):
        return self.winner_team

    def tick(self):
        return self._tick

    def agent_stat(self, pid, which):
        return pid * 100 + which

    def ancient_health(self, team):
        return self.ancient_healths[team]


class ScriptedSource:
    """Returns a fixed per-hero action list every tick; records the obs seen."""

    def __init__(self, actions):
        self.actions = actions
        self.seen = []  # (tick, obs) pairs

    async def get_actions(self, tick, obs):
        self.seen.append((tick, obs.copy()))
        return self.actions


class SlowSource:
    async def get_actions(self, tick, obs):
        await asyncio.sleep(60)
        return [NOOP]


class NoneSource:
    async def get_actions(self, tick, obs):
        return None


class RaisingSource:
    async def get_actions(self, tick, obs):
        raise RuntimeError("player exploded")


class MalformedSource:
    def __init__(self, payload):
        self.payload = payload

    async def get_actions(self, tick, obs):
        return self.payload


# -- action routing ----------------------------------------------------------

async def test_scripted_actions_reach_sim_rows():
    sim = FakeSim()
    sources = [ScriptedSource([[i, i % 7, 1, 0, 1, 0]]) for i in range(10)]
    cfg = make_config(max_ticks=3)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 3
    assert len(sim.fed_actions) == 3
    for fed in sim.fed_actions:
        for pid in range(10):
            assert fed[pid].tolist() == [min(pid, 6), pid % 7, 1, 0, 1, 0]


async def test_team_variant_obs_slicing_and_row_mapping():
    sim = FakeSim()
    radiant = ScriptedSource([[1, 1, 0, 0, 0, 0]] * 5)
    dire = ScriptedSource([[2, 2, 1, 1, 1, 1]] * 5)
    cfg = make_config(num_seats=2, max_ticks=2)
    await LockstepEngine(sim, cfg, [radiant, dire]).run()
    # each seat saw exactly its heroes' obs rows (row p is filled with p)
    for tick, obs in radiant.seen:
        assert obs.shape == (5, 510)
        assert obs[:, 0].tolist() == [0, 1, 2, 3, 4]
    for tick, obs in dire.seen:
        assert obs[:, 0].tolist() == [5, 6, 7, 8, 9]
    # and each seat's actions landed on its heroes' rows
    fed = sim.fed_actions[0]
    for pid in range(5):
        assert fed[pid].tolist() == [1, 1, 0, 0, 0, 0]
    for pid in range(5, 10):
        assert fed[pid].tolist() == [2, 2, 1, 1, 1, 1]


async def test_solo_variant_obs_is_single_hero_row():
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(10)]
    cfg = make_config(max_ticks=1)
    await LockstepEngine(sim, cfg, sources).run()
    for seat, src in enumerate(sources):
        (tick, obs), = src.seen
        assert tick == 0
        assert obs.shape == (1, 510)
        assert obs[0, 0] == seat


# -- NOOP fallbacks ----------------------------------------------------------

@pytest.mark.parametrize("bad_source", [
    NoneSource(),
    RaisingSource(),
    MalformedSource("garbage"),
    MalformedSource([[1, 2, 3]]),                    # wrong shape
    MalformedSource([[1, 2, 3, 4, 5]]),              # 5 values not 6
    MalformedSource([[float("nan")] * 6]),           # non-finite
    MalformedSource([["a", "b", "c", "d", "e", "f"]]),
])
async def test_bad_sources_get_noop(bad_source):
    sim = FakeSim()
    sources = [ScriptedSource([[1, 1, 1, 1, 1, 1]]) for _ in range(9)]
    sources.append(bad_source)
    cfg = make_config(max_ticks=2)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 2  # episode never crashes
    for fed in sim.fed_actions:
        assert fed[9].tolist() == NOOP
        assert fed[0].tolist() == [1, 1, 1, 1, 1, 1]


async def test_deadline_timeout_gets_noop():
    sim = FakeSim()
    sources = [ScriptedSource([[2, 2, 0, 0, 0, 0]]) for _ in range(9)]
    sources.append(SlowSource())
    cfg = make_config(max_ticks=2, tick_deadline_ms=50)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 2
    for fed in sim.fed_actions:
        assert fed[9].tolist() == NOOP


async def test_out_of_range_actions_clamped():
    sim = FakeSim()
    sources = [ScriptedSource([[99, -3, 7, 2, 1, -1]])] + \
        [ScriptedSource([NOOP]) for _ in range(9)]
    cfg = make_config(max_ticks=1)
    ticks = []
    engine = LockstepEngine(sim, cfg, sources,
                            on_tick=lambda t, a: ticks.append((t, a.copy())))
    await engine.run()
    assert sim.fed_actions[0][0].tolist() == [6, 0, 2, 1, 1, 0]
    # replay hook sees the same post-clamp values, as uint8
    (t0, acts0), = ticks
    assert t0 == 0
    assert acts0.dtype == np.uint8
    assert acts0[0].tolist() == [6, 0, 2, 1, 1, 0]
    assert acts0[5].tolist() == NOOP


# -- termination + scoring ---------------------------------------------------

async def test_ancient_win_scores_by_team():
    sim = FakeSim(done_at=4, winner_team=1)
    sources = [ScriptedSource([NOOP]) for _ in range(10)]
    cfg = make_config(max_ticks=100)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert isinstance(result, EpisodeResult)
    assert result.end_reason == "ancient"
    assert result.winner == 1
    assert result.final_tick == 4
    assert list(result.seat_scores) == [0.0] * 5 + [1.0] * 5


async def test_tick_cap_tiebreak_by_ancient_health():
    sim = FakeSim(ancient_healths=(50.0, 200.0))
    sources = [ScriptedSource([NOOP]) for _ in range(10)]
    cfg = make_config(max_ticks=5)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.end_reason == "tick_cap"
    assert result.winner == 1
    assert result.final_tick == 5
    assert list(result.seat_scores) == [0.0] * 5 + [1.0] * 5
    assert result.ancient_healths == (50.0, 200.0)


async def test_tick_cap_equal_health_is_draw():
    sim = FakeSim(ancient_healths=(80.0, 80.0))
    sources = [ScriptedSource([NOOP]) for _ in range(10)]
    cfg = make_config(max_ticks=5)
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.winner is None
    assert list(result.seat_scores) == [0.5] * 10


async def test_team_variant_scores():
    sim = FakeSim(done_at=2, winner_team=0)
    cfg = make_config(num_seats=2, max_ticks=5)
    sources = [ScriptedSource([NOOP] * 5) for _ in range(2)]
    result = await LockstepEngine(sim, cfg, sources).run()
    assert list(result.seat_scores) == [1.0, 0.0]


async def test_reward_sums_and_stats():
    sim = FakeSim()
    sources = [ScriptedSource([NOOP]) for _ in range(10)]
    cfg = make_config(max_ticks=3)
    result = await LockstepEngine(sim, cfg, sources).run()
    # hero pid p earns p+1 per tick, 3 ticks, 1 hero per seat
    assert list(result.seat_reward_sums) == pytest.approx(
        [(p + 1) * 3 for p in range(10)])
    assert len(result.agent_stats) == 10
    assert result.agent_stats[2]["kills"] == 2 * 100 + 1
    assert result.agent_stats[7]["level"] == 7 * 100 + 0


async def test_team_variant_reward_sums():
    sim = FakeSim()
    cfg = make_config(num_seats=2, max_ticks=2)
    sources = [ScriptedSource([NOOP] * 5) for _ in range(2)]
    result = await LockstepEngine(sim, cfg, sources).run()
    # radiant heroes earn 1..5, dire 6..10, per tick x2 ticks
    assert list(result.seat_reward_sums) == pytest.approx([30.0, 80.0])


# -- real wasm sim end-to-end ------------------------------------------------

async def test_real_sim_end_to_end():
    from cogame_moba.sim import MobaSim

    class RngSource:
        def __init__(self, seat, heroes):
            self.rng = np.random.default_rng(seat)
            self.heroes = heroes

        async def get_actions(self, tick, obs):
            assert obs.shape == (self.heroes, 510)
            return self.rng.integers(
                0, defaults.ACT_HIGH, size=(self.heroes, 6)).tolist()

    sim = MobaSim(seed=11)
    cfg = make_config(max_ticks=40, tick_deadline_ms=2000)
    sources = [RngSource(seat, 1) for seat in range(10)]
    result = await LockstepEngine(sim, cfg, sources).run()
    assert result.final_tick == 40
    assert result.end_reason == "tick_cap"
    assert sim.tick() == 40
    assert all(np.isfinite(result.seat_reward_sums))
    assert result.ancient_healths[0] > 0 and result.ancient_healths[1] > 0
    assert all(s["level"] >= 1 for s in result.agent_stats)


async def test_real_sim_team_variant_slicing():
    from cogame_moba.sim import MobaSim

    seen = {}

    class CaptureSource:
        def __init__(self, seat):
            self.seat = seat

        async def get_actions(self, tick, obs):
            if tick == 0:
                seen[self.seat] = obs.copy()
            return [NOOP] * 5

    sim = MobaSim(seed=11)
    full_obs = sim.observations()
    cfg = make_config(num_seats=2, max_ticks=2, tick_deadline_ms=2000)
    await LockstepEngine(sim, cfg, [CaptureSource(0), CaptureSource(1)]).run()
    np.testing.assert_array_equal(seen[0], full_obs[0:5])
    np.testing.assert_array_equal(seen[1], full_obs[5:10])
