"""Kinetix success follows the env's goal signal, not the shaped reward (allenai#186)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vla_eval.benchmarks.kinetix.benchmark import ACTION_DIM, KinetixBenchmark
from vla_eval.recording import NullEpisodeRecorder


def _scripted(steps: list[tuple[float, bool]]) -> KinetixBenchmark:
    """A benchmark whose env step returns ``(reward, GoalR)`` from ``steps``; no JAX."""
    bench = KinetixBenchmark(observation_type="symbolic")
    script = iter(steps)

    def step(rng, state, action, params):
        reward, goal = next(script)
        return np.zeros(3), state, reward, False, {"GoalR": np.bool_(goal)}

    bench._jax = SimpleNamespace(random=SimpleNamespace(split=lambda k: (k, k)))
    bench._jnp = SimpleNamespace(array=np.asarray)
    bench._rng, bench._jit_step, bench._recorder = 0, step, NullEpisodeRecorder()
    return bench


def _success(steps: list[tuple[float, bool]]) -> bool:
    bench = _scripted(steps)
    for _ in steps:
        bench.step({"actions": np.zeros(ACTION_DIM, np.float32)})
    return bench.get_step_result(None)["success"]  # ty: ignore[invalid-argument-type]


def test_shaping_reward_alone_is_not_success() -> None:
    assert not _success([(0.03, False), (0.04, False), (-0.995, False)])


def test_reaching_the_goal_is_success() -> None:
    assert _success([(0.02, False), (1.01, True)])
