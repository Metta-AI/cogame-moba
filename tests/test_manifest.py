"""Tripwires: the manifest template stays in sync with the code.

The results schema is CLOSED (AGENTS.md): _results_doc, the manifest
results_schema, and docker_smoke.sh's expected key set must all list
exactly the same keys, and the end_reason enum must match the engine's.
Every variant/certification game_config in the template must parse
through GameConfig.from_dict (with runner-injected tokens).
"""

import json
import re
from pathlib import Path
from typing import get_args

from cogame_moba import defaults
from cogame_moba.config import GameConfig
from cogame_moba.engine import (NOOP_CAUSES, STAT_NAMES, EndReason,
                                EpisodeResult)
from cogame_moba.server import GameServer

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads(
    (REPO_ROOT / "coworld_manifest_template.json").read_text())
DOCKER_SMOKE = (REPO_ROOT / "tools" / "ci" / "docker_smoke.sh").read_text()


def _game_server(num_seats=2):
    heroes = defaults.NUM_HEROES // num_seats
    cfg = GameConfig.from_dict({
        "players": [{"name": f"p{i}"} for i in range(num_seats)],
        "tokens": [f"t{i}" for i in range(num_seats)],
        "heroes_per_seat": heroes,
        "seed": 1,
    })
    return GameServer(cfg)


def _dummy_result(num_seats=2):
    return EpisodeResult(
        winner=0,
        end_reason="ancient",
        seat_scores=(1.0,) + (0.0,) * (num_seats - 1),
        seat_reward_sums=(0.0,) * num_seats,
        agent_stats=tuple({n: 0 for n in STAT_NAMES}
                          for _ in range(defaults.NUM_HEROES)),
        final_tick=1,
        ancient_healths=(1.0, 1.0),
        seat_noop_ticks=(0,) * num_seats,
        seat_dead=(False,) * num_seats,
        seat_noop_causes=tuple(dict.fromkeys(NOOP_CAUSES, 0)
                               for _ in range(num_seats)),
    )


def _schema_keys():
    schema = MANIFEST["game"]["results_schema"]
    assert schema["additionalProperties"] is False, \
        "results_schema must stay closed"
    return set(schema["required"]), set(schema["properties"])


def test_results_doc_matches_manifest_results_schema():
    server = _game_server()
    doc_keys = set(server._results_doc(_dummy_result()))
    required, properties = _schema_keys()
    assert doc_keys == required, sorted(doc_keys ^ required)
    assert doc_keys == properties, sorted(doc_keys ^ properties)


def test_fault_results_doc_has_same_closed_key_set():
    server = _game_server()
    assert set(server._fault_results_doc(0)) == \
        set(server._results_doc(_dummy_result()))


def test_docker_smoke_expected_keys_match_results_doc():
    # third leg of the triple-sync rule: docker_smoke.sh's expected set
    match = re.search(r"expected = \{(.*?)\}", DOCKER_SMOKE, re.DOTALL)
    assert match, "docker_smoke.sh expected-keys block not found"
    smoke_keys = set(re.findall(r'"(\w+)"', match.group(1)))
    doc_keys = set(_game_server()._results_doc(_dummy_result()))
    assert smoke_keys == doc_keys, sorted(smoke_keys ^ doc_keys)


def test_end_reason_enum_matches_engine():
    schema_enum = set(
        MANIFEST["game"]["results_schema"]["properties"]["end_reason"]["enum"])
    assert schema_enum == set(get_args(EndReason))


def test_variant_and_certification_configs_parse():
    """Every game_config the manifest ships must be accepted by the
    server's parser once the runner injects tokens."""
    configs = [(v["id"], v["game_config"]) for v in MANIFEST["variants"]]
    configs.append(("certification", MANIFEST["certification"]["game_config"]))
    for label, game_config in configs:
        data = dict(game_config)
        data["tokens"] = [f"tok-{i}" for i in range(len(data["players"]))]
        cfg = GameConfig.from_dict(data)  # raises ConfigError on drift
        assert cfg.num_seats == len(game_config["players"]), label
