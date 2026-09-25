"""RoboTwin environment reuse (``clear_cache_freq``) without the image: which resets build a new env or clear the cache."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import pytest

from vla_eval.benchmarks.robotwin import benchmark as rt


class _Env:
    created = 0
    step_lim = 2

    def __init__(self):
        _Env.created += 1
        self.closes: list[bool] = []
        self.take_action_cnt, self.eval_success = 0, False

    def close_env(self, clear_cache=False):
        self.closes.append(clear_cache)

    def setup_demo(self, **kwargs):
        self.take_action_cnt = 0

    def set_instruction(self, instruction):
        pass

    def get_obs(self):
        return {}

    def take_action(self, act, action_type):
        self.take_action_cnt += 1


class _Recorder:
    def record_video(self, frame):
        pass

    def record_step(self, **kwargs):
        pass


@pytest.fixture
def bench_factory(monkeypatch):
    monkeypatch.setattr(rt, "_patched_robot_set_planner", lambda enabled: nullcontext())
    monkeypatch.setattr(rt, "_patched_render_setup", lambda enabled: nullcontext())
    _Env.created = 0

    def make(clear_cache_freq):
        bench = rt.RoboTwinBenchmark(task_name="beat_block_hammer", clear_cache_freq=clear_cache_freq)
        bench._args, bench._recorder = {}, cast(Any, _Recorder())
        monkeypatch.setattr(bench, "_create_env", _Env)
        return bench

    return make


def _episode(bench, finish=True):
    bench.reset({"seed": 1, "episode_idx": 0, "instruction": "x"})
    for _ in range(bench._env.step_lim if finish else 1):
        bench.step({"actions": np.zeros(14)})
    return bench._env


def test_default_builds_a_new_env_and_clears_every_episode(bench_factory):
    bench = bench_factory(0)
    envs = [_episode(bench) for _ in range(3)]
    assert _Env.created == 3 and envs[0].closes == [True] and envs[1].closes == [True]


def test_reuse_clears_every_k_episodes(bench_factory):
    bench = bench_factory(2)
    envs = [_episode(bench) for _ in range(5)]
    assert _Env.created == 1 and len({id(e) for e in envs}) == 1
    assert envs[0].closes == [False, True, False, True]  # cleared before episodes 3 and 5


def test_unfinished_episode_gets_a_fresh_env(bench_factory):
    bench = bench_factory(5)
    first = _episode(bench, finish=False)
    second = _episode(bench)
    assert _Env.created == 2 and first is not second and first.closes == [True]


def test_clear_cache_freq_must_be_non_negative():
    with pytest.raises(ValueError, match="clear_cache_freq"):
        rt.RoboTwinBenchmark(task_name="beat_block_hammer", clear_cache_freq=-1)
