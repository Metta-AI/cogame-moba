"""Side battery: scripted vs baseline over both side assignments.

Validation harness for the dire-side fix: plays full websocket episodes
(ServerHarness, exactly like tests/test_scripted.py's behavioral tests)
with the scripted player in each seat against the pretrained baseline
(or against a scripted mirror with --mirror), over a seed battery, and
prints a per-episode table plus a side summary.

Usage (from the repo root):
    uv run python tools/side_battery.py --seeds 13 7 21 42 99 1 3
    uv run python tools/side_battery.py --mirror --seeds 13 7 21
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from players.client import play_episode
from players.scripted_player import ScriptedPolicy

from tests.test_server import ServerHarness, make_config


async def run_episode(seed: int, scripted_seat: int, mirror: bool,
                      max_ticks: int) -> dict:
    if mirror:
        opponent = ScriptedPolicy()
    else:
        from players.baseline_player import BaselinePolicy
        opponent = BaselinePolicy(seed=1)
    policies = {scripted_seat: ScriptedPolicy(),
                1 - scripted_seat: opponent}
    cfg = make_config(num_seats=2, max_ticks=max_ticks, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        async with ServerHarness(cfg, Path(tmp)) as h:
            await asyncio.gather(
                play_episode(policies[0], h.ws_url(0, "token-0")),
                play_episode(policies[1], h.ws_url(1, "token-1")))
            result = await h.episode_task
        results = json.loads(h.results_path.read_text())
    stats = results["agent_stats"]
    lo = scripted_seat * 5
    towers = [sum(s["towers_killed"] for s in stats[:5]),
              sum(s["towers_killed"] for s in stats[5:])]
    return {
        "seed": seed,
        "scripted_seat": scripted_seat,
        "winner": result.winner,
        "scripted_won": result.winner == scripted_seat,
        "end_reason": result.end_reason,
        "final_tick": result.final_tick,
        "ancients": list(result.ancient_healths),
        "towers_scripted": towers[scripted_seat],
        "towers_opponent": towers[1 - scripted_seat],
        "levels_scripted": [s["level"] for s in stats[lo:lo + 5]],
        "dead_seats": results["dead_seats"],
        "noop_ticks": results["noop_ticks"],
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[13, 7, 21, 42, 99, 1, 3])
    ap.add_argument("--max-ticks", type=int, default=6000)
    ap.add_argument("--mirror", action="store_true",
                    help="scripted vs scripted instead of vs baseline")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    opponent = "scripted-mirror" if args.mirror else "baseline"
    rows = []
    wins = {0: 0, 1: 0}
    for scripted_seat in (0, 1):
        side = "radiant" if scripted_seat == 0 else "dire"
        for seed in args.seeds:
            r = await run_episode(seed, scripted_seat, args.mirror,
                                  args.max_ticks)
            rows.append(r)
            wins[scripted_seat] += r["scripted_won"]
            print(f"seed={seed:>3} scripted={side:<7} vs {opponent}: "
                  f"{'WIN ' if r['scripted_won'] else 'LOSS'} "
                  f"end={r['end_reason']:<8} tick={r['final_tick']:>4} "
                  f"ancients={r['ancients']} "
                  f"towers(s={r['towers_scripted']},"
                  f"o={r['towers_opponent']}) "
                  f"levels={r['levels_scripted']} "
                  f"noop={r['noop_ticks']}", flush=True)

    n = len(args.seeds)
    print(f"\nsummary vs {opponent}: "
          f"radiant {wins[0]}/{n}, dire {wins[1]}/{n}")
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
    # exit code signals the success bar for the vs-baseline battery
    if not args.mirror and (wins[1] < (n + 1) // 2 + 1 or wins[0] < n):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
