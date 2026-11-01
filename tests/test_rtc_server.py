"""KinetixRTCServer's chunk schedule against the official eval_flow.py protocol, with a fake sampler."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.rtc import H, KinetixRTCServer, _RowState
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import StubBenchmark, start_server, stop_server
from vla_eval.benchmarks.base import StepResult


class FakeSampler(KinetixRTCServer):
    """Inference ``i`` (0 = the initial chunk) returns slot ``j`` as ``100 * i + j``."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(checkpoint_dir="unused", action_dim=1, **kw)
        self.calls = 0

    def warmup(self, rows: Any) -> None:
        self._states = {"lvl": None}

    def _infer(self, st: _RowState, obs: np.ndarray, prev: np.ndarray | None) -> tuple[np.ndarray, float]:
        i, self.calls = self.calls, self.calls + 1
        return (100.0 * i + np.arange(H, dtype=np.float32))[:, None], 0.0


class Recorder(StubBenchmark):
    """Observations carry ``env_step``; a hold before the first action does not step (Kinetix's gated start)."""

    def __init__(self, steps: int) -> None:
        super().__init__(done_at_step=steps, num_tasks=1)
        self.applied: list[float] = []
        self.pre_start = 0

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": "t", "level": "lvl"}]

    def reset(self, task: dict[str, Any]) -> Any:
        self.applied, self.pre_start = [], 0
        return super().reset(task)

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        return {"state": np.zeros(3, np.float32), "env_step": len(self.applied)}

    def get_hold_action(self, last_action: Any) -> dict[str, Any]:
        return {"actions": np.zeros(1, np.float32), "hold": True}

    def step(self, action: dict[str, Any]) -> Any:
        if action.get("hold") and not self.applied:
            self.pre_start += 1
            return StepResult(obs=None, reward=0.0, done=False, info={})
        self.applied.append(float(np.asarray(action["actions"]).reshape(-1)[0]))
        return super().step(action)


def official(d: int, s: int, n: int) -> list[float]:
    """eval_flow.eval's executed actions: prev[:d] ++ new[d:s] per boundary, then new[s:] zero-padded as prev."""
    prev, out, i = 100.0 * 0 + np.arange(H), [], 1
    while len(out) < n:
        new = 100.0 * i + np.arange(H)
        i += 1
        out += [*prev[:d], *new[d:s]]
        prev = np.concatenate([new[s:], np.zeros(s)])
    return out[:n]


@pytest.mark.anyio
@pytest.mark.parametrize("d,s", [(0, 1), (1, 1), (2, 2), (1, 3), (4, 4), (2, 5)])
async def test_sync_steps_latency_executes_the_official_schedule(free_port, d, s):
    task = await start_server(FakeSampler(delay=d, horizon=s), free_port)
    bench = Recorder(steps=20)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await SyncEpisodeRunner().run_episode(bench, bench.get_tasks()[0], conn, max_steps=20)
    finally:
        await stop_server(task)
    assert bench.applied == official(d, s, 20)


@pytest.mark.anyio
async def test_live_wall_latency_lands_mid_period_after_a_gated_start(free_port):
    d, s, n = 1, 2, 16
    task = await start_server(FakeSampler(delay=d, horizon=s, latency="wall"), free_port)
    bench = Recorder(steps=n)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await LiveEpisodeRunner(hz=10.0).run_episode(bench, bench.get_tasks()[0], conn, max_steps=n + 4)
    finally:
        await stop_server(task)
    assert bench.pre_start >= 1  # tick 0 comes before any action
    assert bench.applied == official(d, s, n)


def test_rejects_an_impossible_schedule():
    for make in (
        lambda: KinetixRTCServer("unused", delay=2, horizon=1),
        lambda: KinetixRTCServer("unused", delay=3, horizon=6),
        lambda: KinetixRTCServer("unused", horizon=0),
        lambda: KinetixRTCServer("unused", method=cast(Any, "x")),
    ):
        with pytest.raises(ValueError):
            make()
