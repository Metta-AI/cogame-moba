"""Tests for the scripted player (Task 6.2b).

Unit tests: obs decode against hand-built vectors, crop ground truth
against a real sim observation, action validity under fuzzed garbage,
and the RETREAT/PUSH hysteresis driven by synthetic obs.

Slow behavioral tests (ServerHarness, as in tests/test_baseline.py):
scripted team vs random team must win decisively; scripted vs the
pretrained baseline is report-only (numbers printed, no assertion —
the trained policy is allowed to be better).
"""

import asyncio
import json

import numpy as np
import pytest

from cogame_moba import defaults
from cogame_moba.sim import MobaSim
from players import scripted_player
from players.client import play_episode
from players.random_player import RandomPolicy
from players.scripted_player import (CROP_TILES, CROP_W, OBS_SIZE, PUSH,
                                     RETREAT, SELF_OFF, SPAWN, TOWERS, VIS,
                                     WAYPOINTS, HeroObs, NavGrid,
                                     ScriptedPolicy, parse_obs)

from tests.test_server import ServerHarness, make_config


# -- obs construction helper -------------------------------------------------

def make_obs(x=50, y=50, level=1, health10=10, mana10=10, team=0,
             hero_type=0, q_timer=0, w_timer=0, e_timer=0,
             crop_tiles=()):
    """A synthetic 510-byte obs encoding the compute_observations
    layout (moba.h:460-533). ``crop_tiles`` is [(dy, dx, tile), ...]
    relative to self."""
    obs = bytearray(OBS_SIZE)
    e = SELF_OFF
    obs[e + 0] = x
    obs[e + 1] = y
    obs[e + 2] = level
    obs[e + 3] = health10
    obs[e + 4] = mana10
    obs[e + 5] = 1          # damage/50
    obs[e + 6] = 1          # move_speed
    obs[e + 7] = 1          # move_modifier
    obs[e + 10] = q_timer
    obs[e + 11] = w_timer
    obs[e + 12] = e_timer
    obs[e + 16] = team
    obs[e + 17 + hero_type] = 1
    # self tile at crop center
    obs[(VIS) * CROP_W + VIS] = 6 + hero_type + 5 * team
    for dy, dx, tile in crop_tiles:
        obs[(dy + VIS) * CROP_W + (dx + VIS)] = tile
    return bytes(obs)


# -- unit: obs decode --------------------------------------------------------

def test_parse_obs_decodes_self_features():
    obs = make_obs(x=42, y=97, level=7, health10=6, mana10=4, team=1,
                   hero_type=3, q_timer=9, w_timer=11, e_timer=13,
                   crop_tiles=[(-5, -5, 2), (5, 5, 4), (0, 1, 8)])
    s = parse_obs(obs)
    assert (s.x, s.y, s.level) == (42, 97, 7)
    assert (s.health10, s.mana10) == (6, 4)
    assert (s.team, s.hero_type) == (1, 3)
    assert (s.q_timer, s.w_timer, s.e_timer) == (9, 11, 13)
    assert len(s.crop) == CROP_TILES
    assert s.tile(-5, -5) == 2
    assert s.tile(5, 5) == 4
    assert s.tile(0, 1) == 8
    assert s.tile(0, 0) == 6 + 3 + 5   # own tile at center
    assert s.tile(6, 0) == -1          # outside the crop
    # tank at level 7: 200 + 7*50 = 550 max mana (moba.h:1722-1725)
    assert s.max_mana() == 550
    assert s.mana_at_least(100)        # 4/10 * 550 = 220 lower bound
    assert not s.mana_at_least(250)


def test_parse_obs_rejects_wrong_size():
    with pytest.raises(ValueError, match="510"):
        parse_obs(bytes(12))


def test_parse_obs_center_matches_real_sim():
    """Ground truth for the stride-1 crop conclusion: in a real first
    tick obs, byte (5*11+5) of the crop is the hero's own tile id
    (grid_id = 6 + hero_type + 5*team, moba.h:1665-1733)."""
    obs = MobaSim(seed=3).observations()
    for pid in range(10):
        s = parse_obs(obs[pid].tobytes())
        assert s.team == pid // 5
        assert s.hero_type == pid % 5
        assert s.tile(0, 0) == 6 + s.hero_type + 5 * s.team
        # self coords consistent with team spawn noise (+-7 of spawn)
        sy, sx = SPAWN[s.team]
        assert abs(s.y - sy) <= 8 and abs(s.x - sx) <= 8


# -- unit: nav grid ----------------------------------------------------------

def test_nav_grid_descends_toward_goal():
    nav = NavGrid()
    # radiant spawn to the first radiant mid waypoint: every step must
    # strictly reduce the BFS distance until adjacency
    y, x = SPAWN[0]
    goal = (int(WAYPOINTS[1][0][0]), int(WAYPOINTS[1][0][1]))
    field = nav._field(goal)
    for _ in range(400):
        dy, dx = nav.step_toward(y, x, goal)
        if (dy, dx) == (0, 0):
            break
        ny, nx = y + dy, x + dx
        assert field[ny * 128 + nx] < field[y * 128 + x]
        y, x = ny, nx
    assert abs(y - goal[0]) <= 1 and abs(x - goal[1]) <= 1


def test_nav_grid_walls_and_towers_blocked():
    nav = NavGrid()
    assert nav.is_wall(0, 0)            # border wall
    assert nav.is_wall(-1, 50)          # out of bounds
    ty, tx, _, _ = TOWERS[0]
    assert nav.is_wall(int(ty), int(tx))  # tower cells are blocked
    assert not nav.is_wall(*SPAWN[0])
    assert not nav.is_wall(*SPAWN[1])


# -- unit: action validity under fuzz ----------------------------------------

def test_act_high_in_sync_with_server_contract():
    # players/ deliberately duplicates ACT_HIGH; keep it in sync
    assert scripted_player.ACT_HIGH == defaults.ACT_HIGH


def test_actions_in_range_over_fuzzed_obs():
    """The policy must emit valid in-range actions for arbitrary obs
    bytes (robustness to partial observability and garbage)."""
    policy = ScriptedPolicy()
    rng = np.random.default_rng(0)
    for tick in range(60):
        rows = [rng.integers(0, 256, size=OBS_SIZE, dtype=np.uint8).tobytes()
                for _ in range(5)]
        actions = policy(tick, rows)
        assert len(actions) == 5
        for row in actions:
            assert len(row) == 6
            assert all(isinstance(a, int) for a in row)
            assert all(0 <= a < hi
                       for a, hi in zip(row, defaults.ACT_HIGH))


def test_single_hero_seat_shape():
    policy = ScriptedPolicy()
    actions = policy(0, [make_obs()])
    assert len(actions) == 1 and len(actions[0]) == 6


def test_policy_is_deterministic():
    obs_seq = [[make_obs(x=30 + t, y=60, health10=10 - t % 4)
                for _ in range(5)] for t in range(30)]
    a = ScriptedPolicy()
    b = ScriptedPolicy()
    assert [a(t, rows) for t, rows in enumerate(obs_seq)] == \
           [b(t, rows) for t, rows in enumerate(obs_seq)]


# -- unit: mode machine ------------------------------------------------------

def test_low_health_triggers_retreat_toward_spawn():
    policy = ScriptedPolicy()
    # radiant hero mid-map at low health: must enter RETREAT and step
    # toward the radiant fountain (down-left: y grows, x shrinks)
    obs = make_obs(x=60, y=60, health10=2, team=0, hero_type=4)
    ay, ax = policy(0, [obs])[0][:2]
    assert policy.heroes[0].mode == RETREAT
    assert ay >= 3 and ax <= 3 and (ay, ax) != (3, 3)

    # healing but under the exit threshold: still retreating
    policy(1, [make_obs(x=60, y=60, health10=7, team=0, hero_type=4)])
    assert policy.heroes[0].mode == RETREAT

    # healed past hysteresis: back to PUSH
    policy(2, [make_obs(x=60, y=60, health10=9, team=0, hero_type=4)])
    assert policy.heroes[0].mode == PUSH


def test_dire_hero_retreats_toward_dire_spawn():
    policy = ScriptedPolicy()
    obs = make_obs(x=60, y=60, health10=1, team=1, hero_type=1)
    ay, ax = policy(0, [obs])[0][:2]
    assert policy.heroes[0].mode == RETREAT
    # dire fountain is up-right: y shrinks, x grows
    assert ay <= 3 and ax >= 3 and (ay, ax) != (3, 3)


def test_push_mode_advances_along_lane():
    policy = ScriptedPolicy()
    # radiant burst (lane 1, mid) at its lane start: PUSH must move it
    # up the mid diagonal toward dire (y shrinks, x grows). (Lane 2's
    # upstream waypoint data doubles back on itself at the start, so
    # mid lane is the unambiguous direction check.)
    y0, x0 = int(WAYPOINTS[1][0][0]), int(WAYPOINTS[1][0][1])
    ay, ax = policy(0, [make_obs(x=x0, y=y0, hero_type=2)])[0][:2]
    assert policy.heroes[0].mode == PUSH
    assert policy.heroes[0].wp_index is not None
    assert ay <= 3 and ax >= 3 and (ay, ax) != (3, 3)


def test_skill_flags_by_role():
    policy = ScriptedPolicy()
    enemy_hero = [(2, 2, 11)]           # dire support tile near us
    enemy_creep = [(1, -1, 4)]          # dire creep

    # burst with full mana and an enemy hero: all three skills flagged
    row = policy(0, [make_obs(hero_type=2, level=5,
                              crop_tiles=enemy_hero)])[0]
    assert row[3:] == [1, 1, 1]

    # nothing hostile visible: no skills, heroes+towers filter
    policy2 = ScriptedPolicy()
    row = policy2(0, [make_obs(hero_type=2, level=5)])[0]
    assert row[2] == 2 and row[3:] == [0, 0, 0]

    # hostile creeps visible: scan-everything filter
    policy3 = ScriptedPolicy()
    row = policy3(0, [make_obs(hero_type=1, crop_tiles=enemy_creep)])[0]
    assert row[2] == 0
    assert row[3] == 1                  # assassin Q wants creeps

    # tank at low health prefers W self-heal over Q spam
    policy4 = ScriptedPolicy()
    row = policy4(0, [make_obs(hero_type=3, health10=4, level=3,
                               crop_tiles=enemy_creep)])[0]
    assert row[3] == 0 and row[4] == 1


def test_dead_tower_memory():
    policy = ScriptedPolicy()
    # stand where a known enemy tower cell is in the crop but shows
    # no TOWER tile: the tower must be remembered dead
    ty, tx, team, _tier = TOWERS[10]    # a dire tower
    ty, tx = int(ty), int(tx)
    obs = make_obs(x=tx - 3, y=ty - 3, team=0)
    policy(0, [obs])
    assert 10 in policy.dead_towers


def test_entrypoint_helpers(monkeypatch):
    monkeypatch.setenv("COGAME_PLAYER_SEED", "5")
    assert isinstance(scripted_player.policy_from_env(), ScriptedPolicy)
    monkeypatch.delenv("COGAME_PLAYER_SEED")
    assert isinstance(scripted_player.policy_from_env(), ScriptedPolicy)


# -- behavioral: scripted must beat random -----------------------------------

@pytest.mark.slow
async def test_scripted_beats_random_full_episode(tmp_path):
    """Full ws episode, team variant: scripted (team 0/radiant) vs
    random (team 1/dire). In-process calibration (seeds 7/13/21, both
    team assignments) kills the random side's ancient by tick
    ~1700-3000, so an outright ancient win inside the 6000-tick cap is
    the expected path; a decisive ancient-health lead at the cap also
    passes (load tolerance)."""
    cfg = make_config(num_seats=2, max_ticks=6000, seed=13)
    async with ServerHarness(cfg, tmp_path) as h:
        done_msgs = await asyncio.gather(
            play_episode(ScriptedPolicy(), h.ws_url(0, "token-0")),
            play_episode(RandomPolicy(seed=99), h.ws_url(1, "token-1")))
        result = await h.episode_task

    results = json.loads(h.results_path.read_text())
    scripted_towers = sum(s["towers_killed"]
                          for s in results["agent_stats"][:5])
    random_towers = sum(s["towers_killed"]
                        for s in results["agent_stats"][5:])
    print(f"\nscripted-vs-random outcome: winner={result.winner} "
          f"end_reason={result.end_reason} final_tick={result.final_tick} "
          f"ancients={result.ancient_healths} "
          f"towers(scripted={scripted_towers}, random={random_towers})")

    assert result.winner == 0
    if result.end_reason == "ancient":
        assert result.ancient_healths[1] == 0.0
    else:
        # tick-cap path: demand a decisive lead, not a coin-flip
        assert result.ancient_healths[0] > result.ancient_healths[1]
        assert scripted_towers > random_towers
    assert results["scores"] == [1.0, 0.0]
    assert all(m["winner"] == 0 for m in done_msgs)
    # healthy episode: no dead seats, negligible NOOP fallbacks
    # (load-tolerant, as in test_baseline)
    assert results["dead_seats"] == [False, False]
    assert all(n <= 5 for n in results["noop_ticks"]), results["noop_ticks"]


@pytest.mark.slow
async def test_scripted_vs_pretrained_baseline_report_only(tmp_path):
    """Scripted vs the pretrained wasm baseline. REPORT ONLY: the
    trained policy is allowed to win; this exists to print comparable
    outcome numbers and to prove the scripted seat stays healthy."""
    from players.baseline_player import BaselinePolicy

    cfg = make_config(num_seats=2, max_ticks=6000, seed=13)
    async with ServerHarness(cfg, tmp_path) as h:
        await asyncio.gather(
            play_episode(ScriptedPolicy(), h.ws_url(0, "token-0")),
            play_episode(BaselinePolicy(seed=1), h.ws_url(1, "token-1")))
        result = await h.episode_task

    results = json.loads(h.results_path.read_text())
    scripted_towers = sum(s["towers_killed"]
                          for s in results["agent_stats"][:5])
    baseline_towers = sum(s["towers_killed"]
                          for s in results["agent_stats"][5:])
    scripted_levels = [s["level"] for s in results["agent_stats"][:5]]
    baseline_levels = [s["level"] for s in results["agent_stats"][5:]]
    print(f"\nscripted-vs-baseline outcome (no assertion on winner): "
          f"winner={result.winner} end_reason={result.end_reason} "
          f"final_tick={result.final_tick} "
          f"ancients={result.ancient_healths} "
          f"towers(scripted={scripted_towers}, baseline={baseline_towers}) "
          f"levels(scripted={scripted_levels}, baseline={baseline_levels})")

    # only seat health is asserted — outcome is informational
    assert results["dead_seats"] == [False, False]
    assert all(n <= 5 for n in results["noop_ticks"]), results["noop_ticks"]
