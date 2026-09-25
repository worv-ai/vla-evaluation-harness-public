"""RoboTwin adapter logic that runs without the image: step limits and the expert-check cap."""

from __future__ import annotations

import sys
import types

import pytest

from vla_eval.benchmarks.robotwin import benchmark as rt


@pytest.fixture
def robotwin_root(tmp_path, monkeypatch):
    (tmp_path / "task_config").mkdir()
    (tmp_path / rt.STEP_LIMIT_FILE).write_text(
        "beat_block_hammer: 400\nput_bottles_dustbin: 1700\nopen_microwave: 1500\n"
    )
    monkeypatch.setattr(rt, "ROBOTWIN_ROOT", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("task, limit", [("open_microwave", 1500), ("beat_block_hammer", 400)])
def test_max_steps_is_the_task_step_limit(robotwin_root, task, limit):
    assert rt.RoboTwinBenchmark(task_name=task).get_metadata()["max_steps"] == limit


def test_max_steps_falls_back_above_every_limit(robotwin_root, tmp_path, monkeypatch):
    assert rt.RoboTwinBenchmark(task_name="not_a_task").get_metadata()["max_steps"] == rt.MAX_STEP_LIMIT
    monkeypatch.setattr(rt, "ROBOTWIN_ROOT", str(tmp_path / "missing"))
    assert rt.RoboTwinBenchmark(task_name="open_microwave").get_metadata()["max_steps"] == rt.MAX_STEP_LIMIT


class _FailingEnv:
    def setup_demo(self, **kwargs):
        raise RuntimeError("planner unavailable")

    def close_env(self, **kwargs):
        pass


def test_expert_check_gives_up(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "generate_episode_instructions", types.SimpleNamespace(generate_episode_descriptions=None)
    )
    bench = rt.RoboTwinBenchmark(task_name="beat_block_hammer", test_num=2, max_expert_seeds=5)
    bench._args = {}
    monkeypatch.setattr(bench, "_create_env", _FailingEnv)
    with pytest.raises(RuntimeError, match="0/2 solvable seeds in 5 tries"):
        bench.get_tasks()


def test_expert_seed_cap_defaults_to_twenty_per_episode():
    assert rt.RoboTwinBenchmark(task_name="beat_block_hammer", test_num=100).max_expert_seeds == 2000
