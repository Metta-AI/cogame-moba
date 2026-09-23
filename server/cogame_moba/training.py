"""Headless training matches over the hosted Puffer MOBA simulator."""

from dataclasses import dataclass

import numpy as np

from players.baseline_player import BaselinePolicy

from . import defaults
from .sim import MobaSim, OBS_SIZE


@dataclass(frozen=True)
class TrainingStep:
    observation: np.ndarray
    action_mask: tuple[bool, ...]
    reward: float
    done: bool
    score: float | None


class TrainingMatch:
    """Train one seat in the ten-seat or two-team Coworld variant.

    Each controlled hero contributes the original six MultiDiscrete heads
    and 510 player-visible bytes. Opponent seats use independent copies of
    the bundled pretrained policy, as they do in hosted player containers.
    """

    def __init__(self, seat: int, heroes_per_seat: int = 1,
                 max_ticks: int = defaults.DEFAULT_MAX_TICKS):
        if heroes_per_seat not in defaults.VALID_HEROES_PER_SEAT:
            raise ValueError("heroes_per_seat must be 1 or 5")
        self.num_seats = defaults.seat_count(heroes_per_seat)
        if not 0 <= seat < self.num_seats or max_ticks < 1:
            raise ValueError("need a valid seat and positive tick limit")
        self.seat = seat
        self.heroes_per_seat = heroes_per_seat
        self.pids = defaults.seat_hero_pids(seat, heroes_per_seat)
        self.max_ticks = max_ticks
        self.action_sizes = defaults.ACT_HIGH * heroes_per_seat
        self.observation_size = OBS_SIZE * heroes_per_seat + 2

    def reset(self, seed: int) -> TrainingStep:
        self.sim = MobaSim(seed=seed)
        self.opponents = {
            seat: BaselinePolicy(seed=1)
            for seat in range(self.num_seats) if seat != self.seat
        }
        self.done = False
        return self._state(0.0, False)

    def _state(self, reward: float, done: bool) -> TrainingStep:
        visible = self.sim.observations()[self.pids].reshape(-1).astype(np.float32) / 255.0
        observation = np.concatenate((visible, np.array([
            self.seat / (self.num_seats - 1),
            max(0.0, 1.0 - self.sim.tick() / self.max_ticks),
        ], dtype=np.float32)))
        score = None
        if done:
            if self.sim.done():
                winner = self.sim.winner()
            else:
                radiant, dire = (self.sim.ancient_health(team)
                                 for team in range(defaults.NUM_TEAMS))
                winner = 0 if radiant > dire else 1 if dire > radiant else -1
            score = 0.5 if winner < 0 else float(
                winner == defaults.team_for_seat(self.seat, self.heroes_per_seat))
        return TrainingStep(observation,
                            (True,) * sum(self.action_sizes), reward, done, score)

    def step(self, action: list[int]) -> TrainingStep:
        if self.done:
            raise RuntimeError("training match has ended")
        if len(action) != len(self.action_sizes) or any(
            not 0 <= value < high
            for value, high in zip(action, self.action_sizes, strict=True)
        ):
            raise ValueError("action is outside the upstream action space")
        obs = self.sim.observations()
        actions = np.tile(np.asarray(defaults.NOOP_ACTION, dtype=np.float32),
                          (defaults.NUM_HEROES, 1))
        for seat, policy in self.opponents.items():
            pids = defaults.seat_hero_pids(seat, self.heroes_per_seat)
            actions[pids] = policy(self.sim.tick(), obs[pids].tolist())
        actions[self.pids] = np.asarray(action, dtype=np.float32).reshape(
            self.heroes_per_seat, defaults.ACTIONS_PER_HERO)
        self.sim.set_actions(actions)
        self.sim.step()
        if self.sim.fault():
            raise RuntimeError("Cogame MOBA simulator faulted")
        done = bool(self.sim.done()) or self.sim.tick() >= self.max_ticks
        self.done = done
        reward = float(self.sim.rewards()[self.pids].sum())
        return self._state(reward, done)
