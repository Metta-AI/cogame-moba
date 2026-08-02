"""wasmtime host for the vendored Puffer MOBA sim compiled to wasm.

The wasm module is a WASI reactor (emscripten STANDALONE_WASM --no-entry)
built by sim/build_sim.sh. See sim/shim.c for the exported API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from wasmtime import (Config, Engine, Func, FuncType, Linker, Module, Store,
                      ValType, WasiConfig)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WASM_PATH = REPO_ROOT / "build" / "moba_sim.wasm"
PRISTINE_WASM_PATH = REPO_ROOT / "build" / "moba_sim_pristine.wasm"

NUM_AGENTS = 10
OBS_SIZE = 510  # 11*11*4 map crop + 26 self features (opaque contract)
NUM_ATNS = 6    # MultiDiscrete [7,7,3,2,2,2] delivered as 6 floats

# MultiDiscrete no-op: center velocity (3,3 -> 0,0), scan-all filter, no skills
NOOP_ACTION = [3, 3, 0, 0, 0, 0]


class MobaSim:
    """One MOBA episode simulator instance hosted in wasm."""

    @classmethod
    def load(cls, seed: int = 1, num_agents: int = NUM_AGENTS,
             wasm_path: str | Path = DEFAULT_WASM_PATH) -> "MobaSim":
        return cls(seed=seed, num_agents=num_agents, wasm_path=wasm_path)

    def __init__(self, seed: int = 1, num_agents: int = NUM_AGENTS,
                 wasm_path: str | Path = DEFAULT_WASM_PATH):
        wasm_path = Path(wasm_path)
        if not wasm_path.exists():
            raise FileNotFoundError(
                f"{wasm_path} not found - run sim/build_sim.sh first")
        self.num_agents = num_agents

        engine = Engine(Config())
        self._store = Store(engine)
        wasi = WasiConfig()
        wasi.inherit_stdout()  # sim printfs (glitch-state warnings etc.)
        wasi.inherit_stderr()
        self._store.set_wasi(wasi)

        module = Module.from_file(engine, str(wasm_path))
        linker = Linker(engine)
        linker.define_wasi()
        # -sALLOW_MEMORY_GROWTH emits this notification import; no-op host stub
        linker.define(
            self._store, "env", "emscripten_notify_memory_growth",
            Func(self._store, FuncType([ValType.i32()], []), lambda _idx: None))
        instance = linker.instantiate(self._store, module)
        self._exports = instance.exports(self._store)
        self._memory = self._exports["memory"]

        # WASI reactor: run emscripten static constructors before anything else
        self._exports["_initialize"](self._store)
        self._exports["moba_init"](self._store, seed, num_agents)

        self._obs_ptr = self._exports["obs_ptr"](self._store)
        self._act_ptr = self._exports["act_ptr"](self._store)
        self._rew_ptr = self._exports["rew_ptr"](self._store)

    # -- lockstep API ------------------------------------------------------

    def observations(self) -> np.ndarray:
        """Fresh (num_agents, 510) uint8 copy of the current observations."""
        raw = self._memory.read(
            self._store, self._obs_ptr,
            self._obs_ptr + self.num_agents * OBS_SIZE)
        return np.frombuffer(bytearray(raw), dtype=np.uint8).reshape(
            self.num_agents, OBS_SIZE)

    def set_actions(self, actions: np.ndarray) -> None:
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, NUM_ATNS):
            raise ValueError(
                f"actions must be ({self.num_agents}, {NUM_ATNS}), "
                f"got {actions.shape}")
        self._memory.write(self._store, actions.tobytes(), self._act_ptr)

    def step(self) -> None:
        self._exports["moba_step"](self._store)

    def reset(self) -> None:
        self._exports["moba_reset"](self._store)

    def rewards(self) -> np.ndarray:
        raw = self._memory.read(
            self._store, self._rew_ptr,
            self._rew_ptr + self.num_agents * 4)
        return np.frombuffer(bytearray(raw), dtype=np.float32)

    def done(self) -> int:
        return self._exports["moba_done"](self._store)

    def winner(self) -> int:
        return self._exports["moba_winner"](self._store)

    def tick(self) -> int:
        return self._exports["moba_tick"](self._store)

    def agent_stat(self, pid: int, which: int) -> int:
        return self._exports["agent_stat"](self._store, pid, which)

    def ancient_health(self, team: int) -> float:
        return self._exports["ancient_health"](self._store, team)
