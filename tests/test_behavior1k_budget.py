"""BEHAVIOR-1K v3.9.4 time budget, with OmniGibson stubbed out."""

import sys
import time
from types import SimpleNamespace

import pytest

from vla_eval.benchmarks.behavior1k.benchmark import Behavior1KBenchmark
from vla_eval.recording import NullEpisodeRecorder


@pytest.fixture
def bench(monkeypatch):
    og = SimpleNamespace(sim=SimpleNamespace(get_rendering_dt=lambda: 1 / 30))
    net = SimpleNamespace(POLICY_RESPONSE_TIMEOUT=600)
    ev = SimpleNamespace(EVAL_TIMEOUT_MULTIPLIER=1.5)
    monkeypatch.setitem(sys.modules, "omnigibson", og)
    monkeypatch.setitem(sys.modules, "omnigibson.eval.utils.network_utils", net)
    monkeypatch.setitem(sys.modules, "omnigibson.eval.evaluator", ev)

    b = Behavior1KBenchmark(tasks=["turning_on_radio"])
    state = SimpleNamespace(obs={}, metrics=[], env_accessor=SimpleNamespace(success=True))
    b._evaluator = SimpleNamespace(max_steps=100, instance_eval_states=[state])
    b._recorder = NullEpisodeRecorder()
    b._instance_id, b._steps, b._failure, b._deadline = 301, 0, None, None
    b._obs_ready = time.monotonic()
    return b, net


def test_first_query_starts_budget(bench):
    b, _ = bench
    now = time.monotonic()
    assert b._check_budget(now) is None
    assert b._deadline == pytest.approx(now + 100)


@pytest.mark.parametrize("reason", ["policy_query_timeout", "group_time_budget_exceeded"])
def test_failed_rollout_scores_zero(bench, reason):
    b, net = bench
    if reason == "policy_query_timeout":
        net.POLICY_RESPONSE_TIMEOUT = 0
    else:
        b._deadline = time.monotonic() - 1
    result = b.step({"actions": [0.0] * 23})
    assert result.done
    out = b.get_step_result(result)
    assert out["failure_reason"] == reason
    assert out["success"] is False and out["q_score"] == 0.0
    assert out["normalized_time"] == pytest.approx(1 / 1.5)
    assert out["normalized_agent_distance_base"] == 0.0
