"""aiohttp game server implementing the Coworld runtime contract.

Episode mode (default): read the game config from ``COGAME_CONFIG_URI``,
serve ``GET /player?slot=N&token=T`` websockets, run one lockstep episode
(starting when all seats are connected or the connect timeout elapses),
then write ``results.json`` to ``COGAME_RESULTS_URI`` and the replay
bytes to ``COGAME_SAVE_REPLAY_URI`` and exit 0. Seats that never connect
are declared to ``COGAME_PLAYER_FAILURE_URI`` and played as NOOP.

Wire protocol, one JSON text message per tick each way:

    server -> player  {"tick": t, "obs": ["<base64 510B>" x heroes]}
    player -> server  {"tick": t, "actions": [[6 ints] x heroes]}
    server -> player  {"done": true, "result": {...}}   (episode end)

Wrong-tick or malformed replies are treated as missing (NOOP for that
tick); the connection stays up and the episode never crashes.

Replay mode: when ``COGAME_LOAD_REPLAY_URI`` is set, no episode runs;
the recorded replay is served at ``GET /replay-data`` (raw bytes) and
``GET /client/replay`` (viewer page) and the process stays up.

Entry point: ``python -m cogame_moba.server``. Binds ``COGAME_HOST``/
``COGAME_PORT`` (default 0.0.0.0:8080).
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import os
import sys

import numpy as np
from aiohttp import WSMsgType, web

from . import defaults, uris
from .config import GameConfig
from .engine import EpisodeResult, LockstepEngine
from .replay import Replay, ReplayWriter, sim_wasm_sha256
from .sim import DEFAULT_WASM_PATH, MobaSim

# After artifacts are written, keep serving briefly so clients can finish
# reading the done message and close their websockets.
SHUTDOWN_GRACE_SECONDS = 1.0


class WsSeat:
    """One player seat: websocket connection state + engine ActionSource.

    ``get_actions`` sends the tick's obs to the connected player and waits
    for a matching-tick reply; the engine's deadline (asyncio.wait_for)
    cancels the wait when the player is late. No connection -> None
    immediately (the engine plays NOOP without burning the deadline).
    """

    def __init__(self, slot: int, name: str, heroes_per_seat: int):
        self.slot = slot
        self.name = name
        self.heroes_per_seat = heroes_per_seat
        self.ws: web.WebSocketResponse | None = None
        self.ever_connected = False
        self._waiter: tuple[int, asyncio.Future] | None = None

    @property
    def connected(self) -> bool:
        return self.ws is not None and not self.ws.closed

    async def get_actions(self, tick: int, obs: np.ndarray):
        ws = self.ws
        if ws is None or ws.closed:
            return None
        payload = json.dumps({
            "tick": tick,
            "obs": [base64.b64encode(row.tobytes()).decode("ascii")
                    for row in obs],
        })
        fut = asyncio.get_running_loop().create_future()
        self._waiter = (tick, fut)
        try:
            await ws.send_str(payload)
            return await fut
        except Exception:
            return None
        finally:
            self._waiter = None

    def deliver(self, data) -> None:
        """Route one decoded client message to the pending tick waiter.

        Wrong-tick or shapeless messages are dropped (the engine NOOPs on
        deadline); action-payload validation happens in the engine.
        """
        if not isinstance(data, dict):
            return
        waiter = self._waiter
        if waiter is None:
            return
        tick, fut = waiter
        if data.get("tick") != tick:
            return
        if not fut.done():
            fut.set_result(data.get("actions"))


class GameServer:
    def __init__(self, config: GameConfig, *,
                 results_uri: str | None = None,
                 save_replay_uri: str | None = None,
                 player_failure_uri: str | None = None,
                 sim_factory=MobaSim,
                 wasm_path=DEFAULT_WASM_PATH):
        self.config = config
        self.results_uri = results_uri
        self.save_replay_uri = save_replay_uri
        self.player_failure_uri = player_failure_uri
        self.sim_factory = sim_factory
        self.wasm_path = wasm_path
        self.seats = [
            WsSeat(slot, player.name, config.heroes_per_seat)
            for slot, player in enumerate(config.players)]
        self._all_connected = asyncio.Event()
        self.result: EpisodeResult | None = None

    # -- routes --------------------------------------------------------------

    def make_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/healthz", self._handle_healthz)
        app.router.add_get("/player", self._handle_player)
        return app

    async def _handle_healthz(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_player(self, request: web.Request):
        try:
            slot = int(request.query.get("slot", ""))
        except ValueError:
            raise web.HTTPForbidden(text="bad slot")
        if not 0 <= slot < len(self.seats):
            raise web.HTTPForbidden(text="bad slot")
        token = request.query.get("token", "")
        if not hmac.compare_digest(token, self.config.tokens[slot]):
            raise web.HTTPForbidden(text="bad token")

        seat = self.seats[slot]
        if seat.connected:
            # one connection per slot; replace only a dead connection
            raise web.HTTPConflict(text="slot already connected")

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        seat.ws = ws
        seat.ever_connected = True
        if all(s.connected for s in self.seats):
            self._all_connected.set()

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue  # malformed: never crash the episode
                seat.deliver(data)
        finally:
            if seat.ws is ws:
                seat.ws = None
        return ws

    # -- episode orchestration -----------------------------------------------

    async def run_episode(self) -> EpisodeResult:
        cfg = self.config
        try:
            await asyncio.wait_for(
                self._all_connected.wait(),
                cfg.player_connect_timeout_seconds)
        except (asyncio.TimeoutError, TimeoutError):
            pass
        for seat in self.seats:
            if not seat.ever_connected:
                await self._report_player_failure(seat)

        sim = self.sim_factory(seed=cfg.seed)
        writer = ReplayWriter(cfg, self._wasm_sha256())
        engine = LockstepEngine(
            sim, cfg, self.seats, on_tick=writer.append_tick)
        result = await engine.run()
        self.result = result

        results_doc = self._results_doc(result)
        if self.results_uri:
            await uris.write_uri(
                self.results_uri,
                (json.dumps(results_doc, indent=2) + "\n").encode("utf-8"),
                "application/json")
        if self.save_replay_uri:
            await uris.write_uri(
                self.save_replay_uri, writer.finalize(results_doc),
                "application/octet-stream")

        await self._broadcast_done(results_doc)
        return result

    def _wasm_sha256(self) -> str:
        try:
            return sim_wasm_sha256(self.wasm_path)
        except OSError:
            return "unknown"  # non-wasm sim_factory (tests with fakes)

    def _results_doc(self, result: EpisodeResult) -> dict:
        """results.json payload.

        Columnar arrays in config player/seat order (names, scores, win,
        team) follow the coworld-ctf convention; `scores` is the field
        cross-game Coworld consumers require. Episode metadata (winner,
        end_reason, final_tick, seed, stats) rides alongside and is
        declared by this game's manifest results_schema (Phase 5).
        """
        cfg = self.config
        return {
            "names": [p.name for p in cfg.players],
            "scores": list(result.seat_scores),
            "win": [score == 1.0 for score in result.seat_scores],
            "team": [
                "radiant" if defaults.team_for_seat(
                    seat, cfg.heroes_per_seat) == 0 else "dire"
                for seat in range(cfg.num_seats)],
            "reward_sums": list(result.seat_reward_sums),
            "winner": result.winner,
            "end_reason": result.end_reason,
            "final_tick": result.final_tick,
            "seed": cfg.seed,
            "ancient_healths": list(result.ancient_healths),
            "agent_stats": [dict(stats) for stats in result.agent_stats],
        }

    async def _report_player_failure(self, seat: WsSeat) -> None:
        """Declare a no-show seat to COGAME_PLAYER_FAILURE_URI.

        Payload shape is exactly what the platform runner parses
        (coworld.runner.io.GamePlayerFailure, extra="forbid"): message +
        failed_policy_index only; the seat name and reason ride in the
        message text.
        """
        print(f"seat {seat.slot} ({seat.name}) never connected; playing NOOP",
              file=sys.stderr)
        if not self.player_failure_uri:
            return
        payload = {
            "message": (
                f"player '{seat.name}' in slot {seat.slot} did not connect "
                f"within {self.config.player_connect_timeout_seconds:g}s "
                f"(reason: connect_timeout); seat played as NOOP"),
            "failed_policy_index": seat.slot,
        }
        try:
            await uris.write_uri(
                self.player_failure_uri,
                json.dumps(payload).encode("utf-8"),
                "application/json")
        except Exception as exc:  # best-effort: never blocks the episode
            print(f"player-failure report failed: {exc}", file=sys.stderr)

    async def _broadcast_done(self, results_doc: dict) -> None:
        message = json.dumps({"done": True, "result": results_doc})

        async def send_and_close(seat: WsSeat) -> None:
            ws = seat.ws
            if ws is None or ws.closed:
                return
            try:
                await ws.send_str(message)
                await ws.close()
            except Exception:
                pass

        await asyncio.gather(*(send_and_close(s) for s in self.seats))


# -- replay mode -------------------------------------------------------------

# Placeholder viewer page until the Phase 4 wasm re-sim viewer bundle
# replaces it: fetches /replay-data, parses the binary header client-side,
# and renders the header info.
REPLAY_PLACEHOLDER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>cogame-moba replay</title>
<style>
  body { font-family: ui-monospace, monospace; margin: 2rem; }
  dt { font-weight: bold; margin-top: .6rem; }
  .note { margin-top: 2rem; color: #666; }
</style>
</head>
<body>
<h1>cogame-moba replay</h1>
<dl id="info">loading /replay-data ...</dl>
<p class="note">Placeholder viewer: the full wasm re-simulation viewer
arrives in Phase 4.</p>
<script>
async function load() {
  const resp = await fetch("/replay-data");
  const buf = new Uint8Array(await resp.arrayBuffer());
  const magic = String.fromCharCode(...buf.slice(0, 4));
  if (magic !== "MOBA") throw new Error("bad magic: " + magic);
  const version = buf[4];
  if (version !== 1) throw new Error("unsupported version: " + version);
  const headerLen = new DataView(buf.buffer, 5, 4).getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(buf.slice(9, 9 + headerLen)));
  const players = header.config.players.map(p => p.name).join(", ");
  const winner = header.result.winner === null ? "draw"
    : (header.result.winner === 0 ? "radiant" : "dire");
  document.getElementById("info").innerHTML =
    "<dt>players</dt><dd>" + players + "</dd>" +
    "<dt>seed</dt><dd>" + header.config.seed + "</dd>" +
    "<dt>winner</dt><dd>" + winner + "</dd>" +
    "<dt>ticks</dt><dd>" + header.tick_count + "</dd>";
}
load().catch(e => {
  document.getElementById("info").textContent = "failed: " + e.message;
});
</script>
</body>
</html>
"""


def make_replay_app(replay_bytes: bytes) -> web.Application:
    """Replay-mode app: raw bytes at /replay-data, viewer at /client/replay.

    Raises ReplayError on corrupt bytes (fail at startup, not on request).
    """
    Replay.parse(replay_bytes)

    async def handle_replay_data(request: web.Request) -> web.Response:
        return web.Response(
            body=replay_bytes, content_type="application/octet-stream")

    async def handle_replay_client(request: web.Request) -> web.Response:
        return web.Response(
            text=REPLAY_PLACEHOLDER_HTML, content_type="text/html")

    async def handle_healthz(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    app = web.Application()
    app.router.add_get("/healthz", handle_healthz)
    app.router.add_get("/replay-data", handle_replay_data)
    app.router.add_get("/client/replay", handle_replay_client)
    return app


# -- process entry point -----------------------------------------------------

async def async_main() -> int:
    host = os.environ.get("COGAME_HOST", "0.0.0.0")
    port = int(os.environ.get("COGAME_PORT", "8080"))

    load_replay_uri = os.environ.get("COGAME_LOAD_REPLAY_URI", "")
    if load_replay_uri:
        # Replay mode: no episode, serve the recorded replay indefinitely.
        replay_bytes = await uris.read_uri(load_replay_uri)
        runner = web.AppRunner(make_replay_app(replay_bytes))
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        print(f"cogame-moba replay mode on {host}:{port} "
              f"({len(replay_bytes)} replay bytes)", file=sys.stderr)
        await asyncio.Event().wait()  # serve until the process is stopped
        return 0

    config_uri = os.environ.get("COGAME_CONFIG_URI", "")
    if not config_uri:
        print("COGAME_CONFIG_URI is required", file=sys.stderr)
        return 2
    config = GameConfig.from_dict(
        json.loads(await uris.read_uri(config_uri)))
    server = GameServer(
        config,
        results_uri=os.environ.get("COGAME_RESULTS_URI"),
        save_replay_uri=os.environ.get("COGAME_SAVE_REPLAY_URI"),
        player_failure_uri=os.environ.get("COGAME_PLAYER_FAILURE_URI"),
    )
    runner = web.AppRunner(server.make_app())
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"cogame-moba serving on {host}:{port} "
          f"({config.num_seats} seats x {config.heroes_per_seat} heroes)",
          file=sys.stderr)
    result = await server.run_episode()
    print(f"episode over: winner={result.winner} "
          f"end_reason={result.end_reason} tick={result.final_tick}",
          file=sys.stderr)
    await asyncio.sleep(SHUTDOWN_GRACE_SECONDS)
    await runner.cleanup()
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
