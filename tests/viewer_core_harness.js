#!/usr/bin/env node
// Headless verification harness for the viewer core wasm (no raylib).
//
// Usage: node viewer_core_harness.js <build/viewer_core.js> <replay.bin>
//
// Loads the replay exactly like viewer/index.html does (header JSON
// parsed JS-side, bytes copied into the wasm heap, seed passed from
// header.config.seed), exercises the core API, and prints one JSON
// object for tests/test_viewer.py to assert on. Exits non-zero on any
// failure.
"use strict";

const fs = require("fs");
const path = require("path");

const [, , coreJsPath, replayPath] = process.argv;
if (!coreJsPath || !replayPath) {
  console.error("usage: viewer_core_harness.js <viewer_core.js> <replay.bin>");
  process.exit(2);
}

const bytes = fs.readFileSync(replayPath);
if (bytes.toString("latin1", 0, 4) !== "MOBA" || bytes[4] !== 1) {
  console.error("bad replay magic/version");
  process.exit(2);
}
const headerLen = bytes.readUInt32LE(5);
const header = JSON.parse(bytes.toString("utf-8", 9, 9 + headerLen));

const createViewerCore = require(path.resolve(coreJsPath));

function run(M) {
  const call = (name, ret, args = [], vals = []) =>
    M.ccall(name, ret, args, vals);

  const ptr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, ptr);
  const seed = header.config.seed >>> 0;
  const total = call("viewer_load", "number",
    ["number", "number", "number"], [ptr, bytes.length, seed]);

  // Frame cadence: at speed s, s ticks per 12 advance_frame calls.
  const ticksOver = (frames) => {
    let n = 0;
    for (let i = 0; i < frames; i++)
      n += call("viewer_advance_frame", "number");
    return n;
  };
  call("viewer_set_playing", null, ["number"], [1]);
  const cadence1 = ticksOver(12);
  call("viewer_set_speed", null, ["number"], [4]);
  const cadence4 = ticksOver(12);
  const pausedTicks = (() => {  // paused: advance_frame must be a no-op
    call("viewer_set_playing", null, ["number"], [0]);
    return ticksOver(24);
  })();

  const mid = Math.floor(total / 2);
  call("viewer_seek", null, ["number"], [mid]);
  const midTick = call("viewer_tick", "number");

  call("viewer_seek", null, ["number"], [total]);
  const endTick = call("viewer_tick", "number");
  const playingAtEnd = call("viewer_playing", "number");
  // set_playing(1) at end must refuse (no silent restart/loop)
  call("viewer_set_playing", null, ["number"], [1]);
  const playAtEndRefused = call("viewer_playing", "number") === 0 ? 1 : 0;

  console.log(JSON.stringify({
    total, cadence1, cadence4, pausedTicks, midTick, endTick,
    playingAtEnd, playAtEndRefused,
    done: call("viewer_done", "number"),
    winner: call("viewer_winner", "number"),
    headerTickCount: header.tick_count,
  }));
}

createViewerCore().then(run).catch((e) => {
  console.error("harness failed:", e);
  process.exit(1);
});
