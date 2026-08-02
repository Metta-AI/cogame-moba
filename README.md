# cogame-moba

A Coworld game that runs PufferLib's Ocean MOBA ("Puffer MOBA") with bit-exact
observation/action spaces and physics, so policies RL-trained on the original
environment play identically when submitted to a Coworld league.

The upstream C sim (PufferAI/PufferLib @ `c5d3c637`, MIT) is vendored pristine
under `vendor/upstream/`, patched at build time (`sim/patches/`), and compiled
to WebAssembly with emscripten. The Python server hosts the wasm sim via
`wasmtime`.

See `docs/plans/2026-08-01-cogame-moba-design.md` for the full design.

## Quickstart

```sh
uv sync
sim/build_sim.sh          # requires emcc (brew install emscripten)
uv run pytest             # includes the fidelity gate (tests/test_fidelity.py)
```
