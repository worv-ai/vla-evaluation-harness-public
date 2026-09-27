"""RoboTwin's CPU render path refuses what needs the CUDA-only planner."""

import pytest

from vla_eval.benchmarks.robotwin.benchmark import RoboTwinBenchmark


@pytest.fixture
def cpu_mode(monkeypatch):
    monkeypatch.setattr(RoboTwinBenchmark, "_render_mode", "cpu")


def test_cpu_needs_fast_init(cpu_mode):
    RoboTwinBenchmark(task_name="grab_roller")  # the bundled expert seeds (or skip_expert_check) serve get_tasks
    with pytest.raises(ValueError, match="CuRobo"):
        RoboTwinBenchmark(task_name="grab_roller", fast_init=False)


def test_gpu_accepts_the_defaults():
    assert RoboTwinBenchmark(task_name="grab_roller").fast_init is True
