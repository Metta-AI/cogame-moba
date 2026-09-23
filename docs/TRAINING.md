# Headless training

`server/cogame_moba/training.py` runs the same WebAssembly Puffer MOBA simulator
as the hosted game. Build the simulator and bundled baseline brain:

```sh
bash sim/build_sim.sh
bash sim/build_brain.sh
```

`TrainingMatch(seat, heroes_per_seat=1, max_ticks=40000)` covers the
`default` variant (10 seats, one hero each). Use `heroes_per_seat=5` for the
`team` variant (two seats, five heroes each). `reset(seed)` and `step(action)`
return a `TrainingStep` with the learner's visible observation, legal-action
mask, dense reward, terminal flag, and terminal score.

Each controlled hero contributes 510 observation bytes, normalized by 255,
and six upstream action heads of sizes `[7, 7, 3, 2, 2, 2]`. The final two
observation values are normalized seat index and remaining tick fraction.
The action argument is those six integers per hero, in seat order. Each
opponent seat runs an independent copy of the game's bundled pretrained
policy, matching hosted player containers. One tick advances the sim once.

Dense reward sums the controlled heroes' original simulator rewards. Terminal
score is 1 for a team win, 0 for a loss, and 0.5 for a draw. At the configured
tick cap, Ancient health breaks the tie, as in the hosted game. The adapter
does not simulate WebSocket deadlines or disconnected players.
