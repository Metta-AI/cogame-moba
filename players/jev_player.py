"""Jev strategy policy over the ordinary MOBA observation and action stream."""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from urllib.parse import parse_qs, urlsplit

import aiohttp

from .client import PlayerError, run_policy_main, ws_url_from_env
from .scripted_player import RETREAT, SPAWN, ScriptedPolicy, parse_obs

HERO_NAMES = ("support", "assassin", "burst", "tank", "carry")
CRITERIA = {
    "lane_push": "Follow the scripted lane push, combat, and tower-safety plan.",
    "hero_focus": "Keep scripted movement but focus attacks on heroes and towers.",
    "clear_wave": "Keep scripted movement but attack creeps and use available skills.",
    "retreat": "Navigate toward your fountain until the next judgment.",
}


class JevPolicy:
    """Answer every tick while System One chooses modes in the background."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 interval_ticks: int = 120, max_calls: int = 20,
                 seat_slot: int | None = None):
        if interval_ticks <= 0 or max_calls <= 0:
            raise ValueError("Jev interval and call limit must be positive")
        if not base_url or not (api_key or seat_slot is not None):
            raise PlayerError("Jev requires a model transport and credential")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.interval_ticks = interval_ticks
        self.max_calls = max_calls
        self.seat_slot = seat_slot
        self.scripted = ScriptedPolicy()
        self.modes: list[str] = []
        self.pending: asyncio.Task[list[str]] | None = None
        self.next_decision_tick = 0
        self.calls = 0

    async def _query(self, tick: int, obs_rows: list[bytes]) -> list[str]:
        heroes = []
        for raw in obs_rows:
            hero = parse_obs(raw)
            enemy_hero, enemy_creep, friendly_creeps = (
                ScriptedPolicy._crop_scan(hero))
            heroes.append({
                "team": hero.team,
                "hero": HERO_NAMES[hero.hero_type],
                "x": hero.x, "y": hero.y,
                "health_tenths": hero.health10,
                "mana_tenths": hero.mana10,
                "level": hero.level,
                "nearest_enemy_hero": enemy_hero,
                "nearest_enemy_creep": enemy_creep,
                "nearby_friendly_creeps": len(friendly_creeps),
                "crop_11x11_row_major": list(hero.crop),
            })
        payload = {
            "model": self.model,
            "state": json.dumps({
                "game": "PufferLib Ocean MOBA",
                "goal": "Destroy the enemy Ancient while protecting your own.",
                "tick": tick,
                "heroes": heroes,
                "tile_ids": "0 empty, 1 wall, 2 tower, 3 radiant creep, "
                            "4 dire creep, 5 neutral, 6-10 radiant heroes, "
                            "11-15 dire heroes",
            }, separators=(",", ":")),
            "questions": {
                f"hero_{i}": {
                    "type": "choice",
                    "instructions": "Select the mode most likely to improve your team's chance of winning.",
                    "criteria": CRITERIA,
                } for i in range(len(heroes))
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["x-coworld-player-slot"] = str(self.seat_slot)
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.post(
                    f"{self.base_url}/v1/systemone", json=payload,
                    headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
        modes = []
        for i in range(len(heroes)):
            answer = result["answers"][f"hero_{i}"]
            probabilities = answer["probabilities"]
            if answer["type"] != "choice" or set(probabilities) != set(CRITERIA):
                raise ValueError("Jev returned a different choice menu")
            values = {name: float(probabilities[name]) for name in CRITERIA}
            if (any(not math.isfinite(value) or value < 0 or value > 1
                    for value in values.values())
                    or abs(sum(values.values()) - 1) > 0.01):
                raise ValueError("Jev returned invalid choice probabilities")
            modes.append(max(values, key=values.__getitem__))
        usage = result["usage"]
        print(f"jev tick={tick} modes={','.join(modes)} "
              f"input_tokens={usage['input_tokens']} "
              f"output_tokens={usage['output_tokens']}", file=sys.stderr)
        return modes

    def __call__(self, tick: int, obs_rows: list) -> list[list[int]]:
        observations = [bytes(row) for row in obs_rows]
        if self.pending is not None and self.pending.done():
            self.modes = self.pending.result()
            self.pending = None
        if len(self.modes) != len(observations):
            self.modes = ["lane_push"] * len(observations)
        if (self.pending is None and self.calls < self.max_calls
                and tick >= self.next_decision_tick):
            self.pending = asyncio.create_task(self._query(tick, observations))
            self.calls += 1
            self.next_decision_tick = tick + self.interval_ticks

        actions = self.scripted(tick, observations)
        for i, mode in enumerate(self.modes):
            if self.scripted.heroes[i].mode == RETREAT:
                continue
            if mode == "hero_focus":
                actions[i][2] = 2
            elif mode == "clear_wave":
                actions[i][2] = 0
            elif mode == "retreat":
                hero = parse_obs(observations[i])
                dy, dx = self.scripted.nav.step_toward(
                    hero.y, hero.x, SPAWN[hero.team])
                actions[i][:2] = [3 + 3 * dy, 3 + 3 * dx]
        return actions


def policy_from_env() -> JevPolicy:
    sidecar = os.environ.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "").strip()
    capture = os.environ.get("METTA_CAPTURE_URL", "").strip()
    if sidecar:
        base_url = sidecar
        model = "typesafe/jev-1.13"
        api_key = ""
        seat_slot = int(parse_qs(urlsplit(ws_url_from_env()).query)["slot"][0])
    elif capture:
        base_url = capture
        model = os.environ.get("METTA_CAPTURE_MODEL", "jev-latest")
        api_key = os.environ.get("METTA_CAPTURE_KEY", "").strip()
        seat_slot = None
    else:
        base_url = os.environ.get("TYPESAFE_BASE_URL", "https://api.typesafe.ai")
        model = os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        api_key = os.environ.get("TYPESAFE_API_KEY", "").strip()
        seat_slot = None
    return JevPolicy(
        base_url=base_url, api_key=api_key, model=model,
        interval_ticks=int(os.environ.get("JEV_INTERVAL_TICKS", "120")),
        max_calls=int(os.environ.get("JEV_MAX_CALLS", "20")),
        seat_slot=seat_slot)


def main() -> int:
    return run_policy_main(policy_from_env)


if __name__ == "__main__":
    sys.exit(main())
