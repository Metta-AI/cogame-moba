"""Transport-free lockstep episode engine.

Runs one MOBA episode against per-seat async action sources. Per tick:
slice per-seat observations, gather every seat's actions concurrently
under the config tick deadline, NOOP-fill anything missing/late/
malformed, feed the sim, step, accumulate rewards. A seat can never
crash the episode: any exception, timeout, or bad payload from a source
degrades to the no-op action for that seat's heroes.

The optional ``on_tick(tick, actions)`` callback receives the post-clamp
(10, 6) uint8 action matrix exactly as fed to the sim — the replay
writer hooks here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from . import defaults
from .config import GameConfig

# agent_stat `which` codes, mirroring sim/shim.c agent_stat()
STAT_NAMES = (
    "level", "kills", "deaths", "towers_killed", "creeps_killed",
    "neutrals_killed", "xp", "damage_dealt", "damage_received",
    "healing_dealt", "healing_received",
)

END_REASON_ANCIENT = "ancient"
END_REASON_TICK_CAP = "tick_cap"


class ActionSource(Protocol):
    """Per-seat action provider (websocket seat, scripted bot, ...)."""

    async def get_actions(
            self, tick: int, obs: np.ndarray
    ) -> Sequence[Sequence[int]] | None:
        """Actions for this seat's heroes at ``tick``.

        ``obs`` is the (heroes_per_seat, 510) uint8 slice for this seat's
        heroes in pid order. Returns a (heroes_per_seat, 6)-shaped nested
        sequence of action values, or None to play NOOP this tick.
        """
        ...


@dataclass(frozen=True)
class EpisodeResult:
    winner: int | None            # 0 radiant, 1 dire, None draw
    end_reason: str               # "ancient" | "tick_cap"
    seat_scores: tuple[float, ...]        # win 1 / draw 0.5 / loss 0
    seat_reward_sums: tuple[float, ...]   # sim reward sums per seat
    agent_stats: tuple[dict, ...]         # 10 dicts keyed by STAT_NAMES
    final_tick: int
    ancient_healths: tuple[float, float]  # (radiant, dire) at episode end


class LockstepEngine:
    def __init__(self, sim, config: GameConfig,
                 action_sources: Sequence[ActionSource],
                 on_tick: Callable[[int, np.ndarray], None] | None = None):
        if len(action_sources) != config.num_seats:
            raise ValueError(
                f"need {config.num_seats} action sources, "
                f"got {len(action_sources)}")
        self._sim = sim
        self._config = config
        self._sources = list(action_sources)
        self._on_tick = on_tick

    async def run(self) -> EpisodeResult:
        sim = self._sim
        cfg = self._config
        h = cfg.heroes_per_seat
        deadline = cfg.tick_deadline_ms / 1000.0
        reward_sums = np.zeros(cfg.num_seats, dtype=np.float64)
        noop_row = np.asarray(defaults.NOOP_ACTION, dtype=np.uint8)

        ticks_run = 0
        while not sim.done() and ticks_run < cfg.max_ticks:
            tick = sim.tick()
            obs = sim.observations()
            replies = await asyncio.gather(*(
                self._seat_actions(
                    source, tick,
                    obs[defaults.seat_hero_pids(seat, h).start:
                        defaults.seat_hero_pids(seat, h).stop],
                    deadline)
                for seat, source in enumerate(self._sources)))

            actions = np.tile(noop_row, (defaults.NUM_HEROES, 1))
            for seat, reply in enumerate(replies):
                sanitized = _sanitize(reply, h)
                if sanitized is not None:
                    pids = defaults.seat_hero_pids(seat, h)
                    actions[pids.start:pids.stop] = sanitized

            sim.set_actions(actions.astype(np.float32))
            sim.step()
            ticks_run += 1

            rewards = np.asarray(sim.rewards(), dtype=np.float64)
            for seat in range(cfg.num_seats):
                pids = defaults.seat_hero_pids(seat, h)
                reward_sums[seat] += float(rewards[pids.start:pids.stop].sum())

            if self._on_tick is not None:
                self._on_tick(tick, actions)

        return self._build_result(reward_sums)

    async def _seat_actions(self, source: ActionSource, tick: int,
                            seat_obs: np.ndarray, deadline: float):
        """One seat's reply, or None on timeout/error (degrade, never crash)."""
        try:
            return await asyncio.wait_for(
                source.get_actions(tick, seat_obs), deadline)
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

    def _build_result(self, reward_sums: np.ndarray) -> EpisodeResult:
        sim = self._sim
        cfg = self._config
        ancient_healths = (float(sim.ancient_health(0)),
                           float(sim.ancient_health(1)))

        if sim.done():
            end_reason = END_REASON_ANCIENT
            winner: int | None = int(sim.winner())
        else:
            end_reason = END_REASON_TICK_CAP
            radiant, dire = ancient_healths
            if radiant > dire:
                winner = 0
            elif dire > radiant:
                winner = 1
            else:
                winner = None

        scores = []
        for seat in range(cfg.num_seats):
            if winner is None:
                scores.append(0.5)
            else:
                team = defaults.team_for_seat(seat, cfg.heroes_per_seat)
                scores.append(1.0 if team == winner else 0.0)

        agent_stats = tuple(
            {name: int(sim.agent_stat(pid, which))
             for which, name in enumerate(STAT_NAMES)}
            for pid in range(defaults.NUM_HEROES))

        return EpisodeResult(
            winner=winner,
            end_reason=end_reason,
            seat_scores=tuple(scores),
            seat_reward_sums=tuple(float(r) for r in reward_sums),
            agent_stats=agent_stats,
            final_tick=int(sim.tick()),
            ancient_healths=ancient_healths,
        )


def _sanitize(reply, heroes_per_seat: int) -> np.ndarray | None:
    """Validate one seat's reply into (h, 6) uint8, or None if malformed."""
    if reply is None:
        return None
    try:
        arr = np.asarray(reply, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (heroes_per_seat, defaults.ACTIONS_PER_HERO):
        return None
    if not np.isfinite(arr).all():
        return None
    return defaults.clamp_actions(arr)
