"""Jev uses the normal player websocket without delaying per-tick actions."""

import asyncio
import json
from time import perf_counter

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest

from cogame_moba.replay import Replay
from players.client import play_episode
from players.jev_player import JevPolicy, policy_from_env
from players.random_player import RandomPolicy
from players.scripted_player import DIRE_CREEP
from tests.test_scripted import make_obs
from tests.test_server import ServerHarness, make_config


async def test_jev_choice_applies_after_background_request():
    request_started = asyncio.Event()
    release_response = asyncio.Event()
    received = []

    async def systemone(request):
        payload = await request.json()
        received.append((request.headers["Authorization"], payload))
        request_started.set()
        await release_response.wait()
        return web.json_response({
            "answers": {"hero_0": {
                "type": "choice",
                "probabilities": {
                    name: float(name == "hero_focus")
                    for name in payload["questions"]["hero_0"]["criteria"]
                },
            }},
            "usage": {"input_tokens": 12, "output_tokens": 3},
        })

    app = web.Application()
    app.router.add_post("/v1/systemone", systemone)
    async with TestServer(app) as server:
        policy = JevPolicy(
            base_url=str(server.make_url("/")).rstrip("/"),
            api_key="mock", model="test-jev", interval_ticks=10,
            max_calls=1)
        obs = make_obs(crop_tiles=[(0, 1, DIRE_CREEP)])
        assert policy(0, [obs])[0][2] == 0
        await asyncio.wait_for(request_started.wait(), 1)
        start = perf_counter()
        assert policy(1, [obs])[0][2] == 0
        assert perf_counter() - start < 0.1
        release_response.set()
        await asyncio.sleep(0.05)
        assert policy(2, [obs])[0][2] == 2

    assert received[0][0] == "Bearer mock"
    state = json.loads(received[0][1]["state"])
    assert state["heroes"][0]["team"] == 0
    assert set(received[0][1]["questions"]) == {"hero_0"}


@pytest.mark.parametrize("num_seats", [2, 10])
async def test_jev_seat_keeps_up_with_game(tmp_path, num_seats):
    requests = []

    async def systemone(request):
        payload = await request.json()
        requests.append(payload)
        await asyncio.sleep(0.05)
        return web.json_response({
            "answers": {
                key: {"type": "choice", "probabilities": {
                    name: float(name == "hero_focus")
                    for name in question["criteria"]}}
                for key, question in payload["questions"].items()
            },
            "usage": {"input_tokens": 50, "output_tokens": 5},
        })

    app = web.Application()
    app.router.add_post("/v1/systemone", systemone)
    async with TestServer(app) as server:
        policy = JevPolicy(
            base_url=str(server.make_url("/")).rstrip("/"),
            api_key="mock", model="test-jev", interval_ticks=10,
            max_calls=1)
        cfg = make_config(num_seats=num_seats, max_ticks=100,
                          tick_deadline_ms=100)
        async with ServerHarness(cfg, tmp_path) as game:
            results = await asyncio.gather(
                play_episode(policy, game.ws_url(0, "token-0")),
                *(play_episode(RandomPolicy(seed=i),
                                game.ws_url(i, f"token-{i}"))
                  for i in range(1, num_seats)))
            await game.episode_task

    assert requests
    assert set(requests[0]["questions"]) == {
        f"hero_{i}" for i in range(10 // num_seats)}
    assert policy.modes == ["hero_focus"] * (10 // num_seats)
    assert results[0]["noop_ticks"] == [0] * num_seats
    replay = Replay.parse(game.replay_path.read_bytes())
    assert replay.tick_count == 100
    assert replay.header["result"]["noop_ticks"] == [0] * num_seats
    assert any(replay.actions(tick)[0, 2] == 2 for tick in range(1, 100))


def test_sidecar_transport_uses_seat_from_normal_ws_url(monkeypatch):
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://sidecar:8080")
    monkeypatch.setenv("COWORLD_PLAYER_WS_URL", "ws://game/player?slot=6&token=secret")
    policy = policy_from_env()
    assert policy.seat_slot == 6
    assert policy.model == "typesafe/jev-1.13"
