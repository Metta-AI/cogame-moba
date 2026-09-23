import numpy as np
import pytest

from cogame_moba import defaults
from cogame_moba.training import TrainingMatch


@pytest.mark.parametrize("heroes_per_seat", [1, 5])
def test_training_match_preserves_hosted_seat_contract(heroes_per_seat):
    def play(action):
        match = TrainingMatch(seat=0, heroes_per_seat=heroes_per_seat,
                              max_ticks=8)
        first = match.reset(seed=7)
        assert first.observation.shape == (510 * heroes_per_seat + 2,)
        assert len(first.action_mask) == 23 * heroes_per_seat
        assert all(first.action_mask)
        observations = [first.observation]
        rewards = []
        for _ in range(8):
            step = match.step(action)
            observations.append(step.observation)
            rewards.append(step.reward)
        assert step.done and match.sim.tick() == 8
        radiant, dire = (match.sim.ancient_health(team)
                         for team in range(defaults.NUM_TEAMS))
        assert step.score == (0.5 if radiant == dire else float(radiant > dire))
        with pytest.raises(RuntimeError, match="ended"):
            match.step(action)
        return np.stack(observations), rewards

    noop = list(defaults.NOOP_ACTION) * heroes_per_seat
    first = play(noop)
    repeat = play(noop)
    assert np.array_equal(first[0], repeat[0])
    assert first[1] == repeat[1]
    changed = play([0, 0, 0, 0, 0, 0] * heroes_per_seat)
    assert not np.array_equal(first[0], changed[0])
