"""OCBenchBenchmark against a fake MJWarp env: seeds per episode, partial resets, per-task limits."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vla_eval.benchmarks.ocbench.benchmark import OCBenchBenchmark
from vla_eval.recording import NullEpisodeRecorder


class FakeEnv:
    """State = [seed, steps]; terminates (with success) when the state's seed is odd and 3 steps have passed."""

    def __init__(self, nworld: int) -> None:
        self.nworld = nworld
        self.state = np.zeros((nworld, 2))
        self.actions: list[np.ndarray] = []
        self.calls: list[tuple[str, list[int]]] = []

    def reset(self, seeds):
        self.calls.append(("reset", list(range(self.nworld))))
        self.state[:] = np.stack([seeds, np.zeros(self.nworld)], 1)
        return self.state.copy(), {}

    def reset_worlds(self, world_ids, seeds):
        self.calls.append(("reset_worlds", list(world_ids)))
        self.state[world_ids] = np.stack([seeds, np.zeros(len(world_ids))], 1)
        obs = self.state + 100.0  # the other worlds' rows are not step()'s observation
        obs[world_ids] = self.state[world_ids]
        return obs, {}

    def step(self, action):
        self.actions.append(action)
        self.state[:, 1] += 1
        success = (self.state[:, 0] % 2 == 1) & (self.state[:, 1] >= 3)
        return (
            self.state.copy(),
            success.astype(np.float32),
            success,
            np.zeros(self.nworld, bool),
            {"success": success},
        )

    def close(self):
        pass


@pytest.fixture
def envs(monkeypatch):
    made: list[FakeEnv] = []
    ocbench = types.ModuleType("ocbench")
    ocbench.parse_env_spec = lambda name: (name.split("-")[0], "mjwarp", {}, 5)  # type: ignore[attr-defined]
    mjwarp = types.ModuleType("ocbench.mjwarp")

    def make_env(family, kwargs, nworld):
        made.append(FakeEnv(nworld))
        return made[-1]

    mjwarp.make_env = make_env  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ocbench", ocbench)
    monkeypatch.setitem(sys.modules, "ocbench.mjwarp", mjwarp)
    return made


def _reset(bench, slots, name, idxs):
    return bench.reset(slots, [{"name": name, "episode_idx": i} for i in idxs], [NullEpisodeRecorder()] * len(slots))


def test_episodes_reset_with_their_own_seeds_and_end_at_the_limit(envs):
    bench = OCBenchBenchmark(num_envs=3, tasks=["block-single-task1-v0"], seed=7)
    seeds = np.float32(
        np.random.SeedSequence([7, 10]).generate_state(5, dtype=np.uint32)
    )  # as the fake state holds them
    obs = _reset(bench, [0, 1], "block-single-task1-v0", [0, 1])
    assert [o[0] for o in obs] == [seeds[0], seeds[1]] and envs[0].calls == [("reset", [0, 1, 2])]
    out = bench.step({0: {"actions": np.ones(7)}, 1: {"actions": np.ones(7)}})
    assert np.all(envs[0].actions[-1][2] == 0) and np.all(envs[0].actions[-1][0] == 1)  # the idle world holds
    obs = _reset(bench, [2], "block-single-task1-v0", [4])  # a refill while slots 0 and 1 run
    assert obs[0][0] == seeds[4] and obs[0][1] == 0 and envs[0].calls[-1] == ("reset_worlds", [2])
    for _ in range(4):
        out = bench.step({0: {"actions": np.zeros(7)}, 1: {"actions": np.zeros(7)}})
    assert out[0].done and out[1].done  # the task's limit of 5 steps (or success)
    assert bench.get_step_result(0, out[0])["success"] == bool(envs[0].state[0, 0] % 2)


def test_a_new_task_builds_its_env(envs):
    bench = OCBenchBenchmark(num_envs=2, tasks=["block-single-task1-v0", "bowling-task1-v0"])
    _reset(bench, [0, 1], "block-single-task1-v0", [0, 1])
    _reset(bench, [0], "bowling-task1-v0", [0])
    assert len(envs) == 2 and envs[1].calls == [("reset", [0, 1])]
    out = bench.step({0: {"actions": np.zeros(7)}})
    assert "score" in bench.get_step_result(0, out[0]) and "score" in bench.get_metric_keys()
