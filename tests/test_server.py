"""End-to-end tests for the Coworld-contract websocket game server.

In-process aiohttp test server + real websocket clients.
"""

import asyncio
import base64
import json

import aiohttp
import numpy as np
import pytest
from aiohttp import WSMsgType
from aiohttp.test_utils import TestServer

from cogame_moba import defaults, uris
from cogame_moba.config import GameConfig
from cogame_moba.replay import Replay
from cogame_moba.server import GameServer


def make_config(num_seats=10, **overrides):
    heroes = 10 // num_seats
    d = {
        "players": [{"name": f"bot-{i}"} for i in range(num_seats)],
        "tokens": [f"token-{i}" for i in range(num_seats)],
        "heroes_per_seat": heroes,
        "seed": 21,
        "max_ticks": 20,
        "tick_deadline_ms": 1000,
        "player_connect_timeout_seconds": 10,
    }
    d.update(overrides)
    return GameConfig.from_dict(d)


class ServerHarness:
    def __init__(self, cfg, tmp_path):
        self.results_path = tmp_path / "results.json"
        self.replay_path = tmp_path / "replay.bin"
        self.failure_path = tmp_path / "player_failure.json"
        self.server = GameServer(
            cfg,
            results_uri=f"file://{self.results_path}",
            save_replay_uri=f"file://{self.replay_path}",
            player_failure_uri=f"file://{self.failure_path}",
        )
        self.test_server = TestServer(self.server.make_app())
        self.episode_task = None

    async def __aenter__(self):
        await self.test_server.start_server()
        self.episode_task = asyncio.create_task(self.server.run_episode())
        return self

    async def __aexit__(self, *exc):
        if not self.episode_task.done():
            self.episode_task.cancel()
        try:
            await self.episode_task
        except asyncio.CancelledError:
            pass
        await self.test_server.close()

    def ws_url(self, slot, token):
        return str(self.test_server.make_url(
            f"/player?slot={slot}&token={token}"))


async def play_random_client(harness, slot, token, heroes):
    """A well-behaved player: random in-range actions until done."""
    rng = np.random.default_rng(slot)
    result = None
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(harness.ws_url(slot, token)) as ws:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                data = json.loads(msg.data)
                if data.get("done"):
                    result = data["result"]
                    break
                obs = [base64.b64decode(o) for o in data["obs"]]
                assert len(obs) == heroes
                assert all(len(o) == 510 for o in obs)
                acts = rng.integers(
                    0, defaults.ACT_HIGH, size=(heroes, 6)).tolist()
                await ws.send_str(json.dumps(
                    {"tick": data["tick"], "actions": acts}))
    return result


# -- full episodes -----------------------------------------------------------

async def test_full_episode_10_seats(tmp_path):
    cfg = make_config(max_ticks=15)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}", 1)
                   for s in range(10)]
        done_msgs = await asyncio.gather(*clients)
        result = await h.episode_task

    # every client got the done message with the result
    assert all(m is not None for m in done_msgs)
    assert done_msgs[0]["final_tick"] == result.final_tick

    results = json.loads(h.results_path.read_text())
    assert results["names"] == [f"bot-{i}" for i in range(10)]
    assert len(results["scores"]) == 10
    assert results["final_tick"] == result.final_tick
    assert results["end_reason"] in ("ancient", "tick_cap")
    assert results["seed"] == 21
    assert results["team"] == ["radiant"] * 5 + ["dire"] * 5
    assert len(results["agent_stats"]) == 10
    # scores consistent with winner
    if results["winner"] is None:
        assert results["scores"] == [0.5] * 10
    else:
        winners = [s for i, s in enumerate(results["scores"])
                   if defaults.team_for_seat(i, 1) == results["winner"]]
        losers = [s for i, s in enumerate(results["scores"])
                  if defaults.team_for_seat(i, 1) != results["winner"]]
        assert winners == [1.0] * 5 and losers == [0.0] * 5
    assert sum(results["scores"]) == 5.0

    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == result.final_tick
    assert replay.header["config"]["seed"] == 21
    assert [p["name"] for p in replay.header["config"]["players"]] == \
        results["names"]
    assert replay.header["result"]["winner"] == results["winner"]
    # no failures reported
    assert not h.failure_path.exists()


async def test_full_episode_team_variant(tmp_path):
    cfg = make_config(num_seats=2, max_ticks=12)
    async with ServerHarness(cfg, tmp_path) as h:
        done_msgs = await asyncio.gather(
            play_random_client(h, 0, "token-0", 5),
            play_random_client(h, 1, "token-1", 5))
        result = await h.episode_task

    assert all(m is not None for m in done_msgs)
    results = json.loads(h.results_path.read_text())
    assert len(results["scores"]) == 2
    assert results["team"] == ["radiant", "dire"]
    assert sum(results["scores"]) == 1.0
    replay = Replay.parse(h.replay_path.read_bytes())
    assert replay.tick_count == result.final_tick


# -- degraded players --------------------------------------------------------

async def test_missing_player_noop_and_failure_report(tmp_path):
    cfg = make_config(max_ticks=6, tick_deadline_ms=200,
                      player_connect_timeout_seconds=0.4)
    async with ServerHarness(cfg, tmp_path) as h:
        clients = [play_random_client(h, s, f"token-{s}", 1)
                   for s in range(9)]  # slot 9 never connects
        await asyncio.gather(*clients)
        result = await h.episode_task

    assert result.final_tick > 0
    failure = json.loads(h.failure_path.read_text())
    assert failure["failed_policy_index"] == 9
    assert "bot-9" in failure["message"]
    assert set(failure) == {"failed_policy_index", "message"}
    assert h.results_path.exists()
    assert h.replay_path.exists()


async def test_malformed_messages_never_crash_episode(tmp_path):
    cfg = make_config(max_ticks=6, tick_deadline_ms=150)

    async def malformed_client(h, slot, token):
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(h.ws_url(slot, token)) as ws:
                garbage = iter([
                    "not json at all",
                    json.dumps({"tick": -99, "actions": [[0] * 6]}),
                    json.dumps({"nonsense": True}),
                    json.dumps({"tick": None, "actions": "x"}),
                ])
                async for msg in ws:
                    if msg.type != WSMsgType.TEXT:
                        break
                    data = json.loads(msg.data)
                    if data.get("done"):
                        return data["result"]
                    try:
                        await ws.send_str(next(garbage))
                    except StopIteration:
                        # then wrong-shaped actions on the right tick
                        await ws.send_str(json.dumps(
                            {"tick": data["tick"], "actions": [[1, 2]]}))
        return None

    async with ServerHarness(cfg, tmp_path) as h:
        good = [play_random_client(h, s, f"token-{s}", 1) for s in range(9)]
        results = await asyncio.gather(*good, malformed_client(h, 9, "token-9"))
        result = await h.episode_task

    assert result.final_tick == 6
    # the malformed client stayed connected and still got the done message
    assert results[-1] is not None
    assert h.results_path.exists()


# -- auth + connection management --------------------------------------------

async def test_bad_token_rejected(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
                await session.ws_connect(h.ws_url(3, "wrong-token"))
            assert exc.value.status == 403


@pytest.mark.parametrize("slot", ["17", "-1", "abc", ""])
async def test_bad_slot_rejected(tmp_path, slot):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as exc:
                await session.ws_connect(
                    str(h.test_server.make_url(
                        f"/player?slot={slot}&token=token-0")))
            assert exc.value.status == 403


async def test_duplicate_slot_rejected_while_alive(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            ws1 = await session.ws_connect(h.ws_url(0, "token-0"))
            with pytest.raises(aiohttp.WSServerHandshakeError):
                await session.ws_connect(h.ws_url(0, "token-0"))
            await ws1.close()
            # dead connection may be replaced; the server's handler may
            # not have observed the close yet, so retry briefly
            for _ in range(40):
                try:
                    ws2 = await session.ws_connect(h.ws_url(0, "token-0"))
                    break
                except aiohttp.WSServerHandshakeError:
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("reconnect to a dead slot was never accepted")
            await ws2.close()


async def test_healthz(tmp_path):
    cfg = make_config(player_connect_timeout_seconds=5)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            async with session.get(h.test_server.make_url("/healthz")) as resp:
                assert resp.status == 200
                assert (await resp.json())["status"] == "ok"


# -- uris --------------------------------------------------------------------

async def test_file_uri_round_trip(tmp_path):
    target = tmp_path / "deep" / "nested" / "out.bin"
    await uris.write_uri(f"file://{target}", b"\x00\x01payload")
    assert await uris.read_uri(f"file://{target}") == b"\x00\x01payload"
    # plain paths (no scheme) also work, matching the runtime convention
    plain = tmp_path / "plain.txt"
    await uris.write_uri(str(plain), b"hello")
    assert await uris.read_uri(str(plain)) == b"hello"


async def test_coworld_mount_style_uri_path():
    # file:///coworld/out/results.json must resolve to /coworld/out/...
    assert uris.local_path("file:///coworld/out/results.json") is not None
    assert str(uris.local_path("file:///coworld/out/results.json")) == \
        "/coworld/out/results.json"


async def test_http_uri_read_write():
    from aiohttp import web

    stored = {}

    async def handle_get(request):
        return web.Response(body=stored.get("blob", b""))

    async def handle_put(request):
        stored["blob"] = await request.read()
        stored["content_type"] = request.content_type
        return web.Response(status=201)

    app = web.Application()
    app.router.add_get("/artifact", handle_get)
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        url = str(server.make_url("/artifact"))
        await uris.write_uri(url, b"http-bytes", "application/json")
        assert stored["blob"] == b"http-bytes"
        assert stored["content_type"] == "application/json"
        assert await uris.read_uri(url) == b"http-bytes"
    finally:
        await server.close()


async def test_unsupported_scheme_rejected():
    with pytest.raises(ValueError):
        await uris.read_uri("s3://bucket/key")
    with pytest.raises(ValueError):
        await uris.write_uri("ftp://host/file", b"x")


# -- replay mode (Task 2.5) --------------------------------------------------

def _write_replay_bytes():
    from cogame_moba.replay import ReplayWriter

    cfg = make_config()
    writer = ReplayWriter(cfg, "aa" * 32)
    rng = np.random.default_rng(3)
    for t in range(8):
        writer.append_tick(
            t, rng.integers(0, defaults.ACT_HIGH,
                            size=(10, 6)).astype(np.uint8))
    return writer.finalize({"winner": 0, "end_reason": "ancient",
                            "final_tick": 8})


async def test_replay_mode_serves_bytes_and_viewer():
    from cogame_moba.server import make_replay_app

    data = _write_replay_bytes()
    server = TestServer(make_replay_app(data))
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(server.make_url("/replay-data")) as resp:
                assert resp.status == 200
                assert resp.content_type == "application/octet-stream"
                assert await resp.read() == data
            async with session.get(server.make_url("/client/replay")) as resp:
                assert resp.status == 200
                assert resp.content_type == "text/html"
                html = await resp.text()
                assert "/replay-data" in html
            async with session.get(server.make_url("/healthz")) as resp:
                assert resp.status == 200
    finally:
        await server.close()


async def test_replay_mode_rejects_corrupt_replay():
    from cogame_moba.replay import ReplayError
    from cogame_moba.server import make_replay_app

    with pytest.raises(ReplayError):
        make_replay_app(b"not a replay")


# -- shutdown robustness (quality review) ------------------------------------

async def test_unresponsive_client_never_blocks_episode_exit(tmp_path):
    """A connected client that never reads or replies must not prevent
    run_episode from returning (bounded done-broadcast, strike rule)."""
    cfg = make_config(max_ticks=5, tick_deadline_ms=100,
                      player_connect_timeout_seconds=2)
    async with ServerHarness(cfg, tmp_path) as h:
        async with aiohttp.ClientSession() as session:
            silent_ws = await session.ws_connect(h.ws_url(9, "token-9"))
            good = [play_random_client(h, s, f"token-{s}", 1)
                    for s in range(9)]
            await asyncio.gather(*good)
            result = await asyncio.wait_for(h.episode_task, timeout=20)
            await silent_ws.close()
    assert result.final_tick == 5
    results = json.loads(h.results_path.read_text())
    assert results["noop_ticks"][9] == 5
    assert results["noop_ticks"][:9] == [0] * 9


async def test_failing_results_uri_does_not_block_replay_write(tmp_path):
    """Artifact writes are independent: a failing results URI must not
    prevent the replay write; the aggregate error is raised after."""
    cfg = make_config(max_ticks=3, tick_deadline_ms=50,
                      player_connect_timeout_seconds=0.1)
    replay_path = tmp_path / "replay.bin"
    server = GameServer(
        cfg,
        results_uri="badscheme://results",
        save_replay_uri=f"file://{replay_path}",
        player_failure_uri=f"file://{tmp_path / 'failure.json'}",
    )
    with pytest.raises(IOError):
        await server.run_episode()
    replay = Replay.parse(replay_path.read_bytes())
    assert replay.tick_count == 3


async def test_http_write_retries_then_succeeds():
    from aiohttp import web

    attempts = 0
    stored = {}

    async def handle_put(request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return web.Response(status=500)
        stored["blob"] = await request.read()
        return web.Response(status=200)

    app = web.Application()
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        url = str(server.make_url("/artifact"))
        await uris.write_uri(url, b"retried", "application/json",
                             backoff_seconds=0.01)
        assert attempts == 3
        assert stored["blob"] == b"retried"
    finally:
        await server.close()


async def test_http_write_raises_after_exhausted_retries():
    from aiohttp import web

    attempts = 0

    async def handle_put(request):
        nonlocal attempts
        attempts += 1
        return web.Response(status=503)

    app = web.Application()
    app.router.add_put("/artifact", handle_put)
    server = TestServer(app)
    await server.start_server()
    try:
        with pytest.raises(IOError):
            await uris.write_uri(str(server.make_url("/artifact")),
                                 b"x", backoff_seconds=0.01)
        assert attempts == 3
    finally:
        await server.close()
