"""Tests for SyncEpisodeRunner, LiveEpisodeRunner, ActionBuffer, batched PredictModelServer, and CI/LAAS."""

from __future__ import annotations

import asyncio
import anyio
import threading
import time
from functools import partial
from typing import Any

import numpy as np
import pytest

from vla_eval.benchmarks.base import repeat_last_hold
from vla_eval.connection import Connection
from vla_eval.runners.action_buffer import ActionBuffer
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.base import Row
from vla_eval.model_servers.serve import serve_async

from tests.conftest import StubBenchmark, wait_for_server, stop_server


@pytest.mark.anyio
async def test_sync_runner_completes(echo_server):
    """Runner finishes when benchmark reports done."""
    benchmark = StubBenchmark(done_at_step=3)
    runner = SyncEpisodeRunner()
    task = {"name": "task_0"}

    async with Connection(echo_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=50)

    assert result["metrics"]["success"] is True
    assert result["steps"] == 3


@pytest.mark.anyio
async def test_sync_runner_respects_max_steps(echo_server):
    """Runner stops at max_steps even if benchmark is not done."""
    benchmark = StubBenchmark(done_at_step=100)
    runner = SyncEpisodeRunner()
    task = {"name": "task_0"}

    async with Connection(echo_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=5)

    assert result["metrics"]["success"] is False
    assert result["steps"] == 5


@pytest.mark.anyio
async def test_random_action_server_completes(random_action_server):
    """StubBenchmark completes even when model returns arbitrary random actions."""
    benchmark = StubBenchmark(done_at_step=3)
    runner = SyncEpisodeRunner()
    task = {"name": "task_0"}

    async with Connection(random_action_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=50)

    assert result["metrics"]["success"] is True
    assert result["steps"] == 3


@pytest.mark.anyio
async def test_chunk_server_completes(chunk_server):
    """StubBenchmark completes with a chunk-based model server (chunk_size=4)."""
    benchmark = StubBenchmark(done_at_step=3)
    runner = SyncEpisodeRunner()
    task = {"name": "task_0"}

    async with Connection(chunk_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=50)

    assert result["metrics"]["success"] is True
    assert result["steps"] == 3


# ---------------------------------------------------------------------------
# Batched PredictModelServer tests (max_batch_size > 1)
# ---------------------------------------------------------------------------


class EchoBatchServer(PredictModelServer):
    def __init__(self):
        super().__init__(max_batch_size=4, max_wait_time=0.1)

    def predict_batch(self, obs, rows):
        return {"actions": np.asarray(obs["value"])[:, None] * np.ones(7)}


class RecordingBatchServer(PredictModelServer):
    def __init__(self):
        super().__init__(max_batch_size=4, max_wait_time=0.5)
        self.batch_sizes: list[int] = []

    def predict_batch(self, obs, rows):
        self.batch_sizes.append(len(rows))
        return {"actions": np.zeros((len(rows), 7))}


class FailingBatchServer(PredictModelServer):
    def __init__(self):
        super().__init__(max_batch_size=4, max_wait_time=0.1)

    def predict_batch(self, obs, rows):
        raise RuntimeError("test error")


@pytest.mark.anyio
async def test_batch_server_completes_episode(free_port):
    server = EchoBatchServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        benchmark = StubBenchmark(done_at_step=3)
        runner = SyncEpisodeRunner()
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            result = await runner.run_episode(benchmark, {"name": "task_0"}, conn, max_steps=50)
        assert result["metrics"]["success"] is True
        assert result["steps"] == 3
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_batch_server_batches_concurrent_requests(free_port):
    server = RecordingBatchServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        conns = [Connection(f"ws://127.0.0.1:{free_port}") for _ in range(4)]
        for c in conns:
            await c.connect()
        await anyio.sleep(0.05)

        results = await asyncio.gather(*(c.act({"value": 1.0}) for c in conns))
        assert len(results) == 4
        assert any(bs > 1 for bs in server.batch_sizes)

        for c in conns:
            await c.close()
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_batch_server_max_wait_time_dispatches_partial(free_port):
    server = EchoBatchServer()
    server.max_batch_size = 100
    server.max_wait_time = 0.05
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            action = await conn.act({"value": 2.0})
        assert "actions" in action
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_batch_server_predict_batch_exception_propagates(free_port):
    server = FailingBatchServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}", timeout=2.0) as conn:
            with pytest.raises(Exception):
                await conn.act({"value": 1.0})
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# ActionBuffer tests
# ---------------------------------------------------------------------------


# Reuse the shared hold helper (absolute-control repeat-last, action_dim=7).
_repeat_hold = partial(repeat_last_hold, action_dim=7)


def test_action_buffer_initial_state():
    """Buffer starts with no action and zero metrics."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    assert not buf.has_action()
    assert not buf.is_new()
    assert buf.update_count == 0
    assert buf.stale_count == 0


def test_action_buffer_update_and_get():
    """update() then get() returns the action."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    action = {"actions": np.ones(7)}
    buf.update(action)
    assert buf.has_action()
    assert buf.is_new()
    result = buf.get()
    assert np.array_equal(result["actions"], np.ones(7))
    assert buf.update_count == 1
    assert buf.stale_count == 0


def test_action_buffer_hold_repeats_last():
    """On a stale tick, hold_fn receives the last fresh action (repeat semantics)."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    buf.update({"actions": np.ones(7) * 2})
    buf.get()  # consume the "new" flag
    result = buf.get()  # stale → hold_fn(last_fresh) → repeat
    assert np.array_equal(result["actions"], np.ones(7) * 2)
    assert buf.stale_count == 1


def test_action_buffer_hold_null_action():
    """hold_fn can return a fixed null action (delta/velocity control)."""
    null = {"actions": np.zeros(7, dtype=np.float32)}
    buf = ActionBuffer(hold_fn=lambda last: null)
    buf.update({"actions": np.ones(7)})
    buf.get()  # consume new
    result = buf.get()  # stale → null
    assert np.allclose(result["actions"], np.zeros(7))
    assert buf.stale_count == 1


def test_action_buffer_reset():
    """reset() clears all state."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    buf.update({"actions": np.ones(7)})
    buf.get()
    buf.reset()
    assert not buf.has_action()
    assert buf.update_count == 0
    assert buf.stale_count == 0


def test_action_buffer_metrics():
    """get_metrics() returns correct stale ratio."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    buf.update({"actions": np.ones(7)})
    buf.get()  # 1 fresh get
    buf.get()  # 1 stale get
    buf.get()  # 1 stale get
    metrics = buf.get_metrics()
    assert metrics["update_count"] == 1
    assert metrics["fresh_count"] == 1
    assert metrics["stale_count"] == 2
    assert abs(metrics["stale_action_ratio"] - 2 / 3) < 1e-6


def test_action_buffer_metrics_count_ticks_not_arrivals():
    """Actions overwritten before any tick used them count as received, not as fresh ticks."""
    buf = ActionBuffer(hold_fn=_repeat_hold)
    buf.update({"actions": np.ones(7)})
    buf.update({"actions": np.ones(7) * 2})  # overwrites the first before a tick reads it
    buf.get()  # fresh
    buf.get()  # stale
    metrics = buf.get_metrics()
    assert metrics["update_count"] == 2
    assert metrics["fresh_count"] == 1
    assert metrics["stale_action_ratio"] == 0.5  # 1 of 2 ticks held, not 1 of 3


# ---------------------------------------------------------------------------
# LiveEpisodeRunner tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_live_runner_completes(echo_server):
    """LiveEpisodeRunner finishes when benchmark reports done."""
    benchmark = StubBenchmark(done_at_step=3)
    runner = LiveEpisodeRunner(hz=100.0)
    task = {"name": "task_0"}

    async with Connection(echo_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=50)

    assert result["metrics"]["success"] is True
    assert result["steps"] == 3
    assert "rt_metrics" in result


@pytest.mark.anyio
async def test_live_runner_respects_max_steps(echo_server):
    """LiveEpisodeRunner stops at max_steps even if benchmark is not done."""
    benchmark = StubBenchmark(done_at_step=100)
    runner = LiveEpisodeRunner(hz=100.0)
    task = {"name": "task_0"}

    async with Connection(echo_server) as conn:
        result = await runner.run_episode(benchmark, task, conn, max_steps=5)

    assert result["metrics"]["success"] is False
    assert result["steps"] == 5
    assert result["rt_metrics"]["stale_action_ratio"] >= 0.0


# ---------------------------------------------------------------------------
# predict() serialisation tests (max_batch_size=1 concurrency safety)
# ---------------------------------------------------------------------------


class ConcurrencyTrackingServer(PredictModelServer):
    """Records predict() entry/exit times and tracks max concurrent calls."""

    def __init__(self):
        super().__init__(max_batch_size=1)
        self._lock = threading.Lock()
        self._active = 0
        self.max_concurrent = 0
        self.call_count = 0

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        with self._lock:
            self._active += 1
            if self._active > self.max_concurrent:
                self.max_concurrent = self._active
            self.call_count += 1
        time.sleep(0.05)  # simulate GPU work
        with self._lock:
            self._active -= 1
        return {"actions": np.ones(7, dtype=np.float32)}


@pytest.mark.anyio
async def test_predict_serialised_under_concurrent_clients(free_port):
    """With max_batch_size=1, predict() must never run concurrently."""
    server = ConcurrencyTrackingServer()
    task = asyncio.create_task(serve_async(server, port=free_port))
    await wait_for_server(free_port)
    try:
        n_clients = 6
        conns = [Connection(f"ws://127.0.0.1:{free_port}") for _ in range(n_clients)]
        for c in conns:
            await c.connect()
        await anyio.sleep(0.05)

        # Fire all observations concurrently
        results = await asyncio.gather(*(c.act({"value": 1.0}) for c in conns))
        assert len(results) == n_clients
        # All calls should have completed
        assert server.call_count == n_clients
        # The critical assertion: at most 1 predict() was active at any time
        assert server.max_concurrent == 1, f"predict() ran {server.max_concurrent} times concurrently; expected 1"

        for c in conns:
            await c.close()
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# Continuous inference (CI) / LAAS
# ---------------------------------------------------------------------------


class CIEchoServer(PredictModelServer):
    def __init__(self, delay: float = 0.0, **kwargs: Any):
        super().__init__(continuous_inference=True, hz=10.0, **kwargs)
        self.delay = delay
        self.calls = 0

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        self.calls += 1
        time.sleep(self.delay)
        return {"actions": obs.get("value", 0) * np.ones(7, dtype=np.float32)}


class CILAASChunkServer(PredictModelServer):
    def __init__(self):
        super().__init__(continuous_inference=True, laas=True, hz=10.0, chunk_size=4)

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        time.sleep(0.25)  # 2.5 steps at 10 Hz
        return {"actions": np.arange(6 * 7, dtype=np.float32).reshape(6, 7)}


async def _serve(server: PredictModelServer, port: int) -> asyncio.Task:
    task = asyncio.create_task(serve_async(server, port=port))
    await wait_for_server(port)
    return task


@pytest.mark.anyio
async def test_ci_server_pushes_actions(free_port):
    task = await _serve(CIEchoServer(), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            assert conn.server_info["capabilities"]["vector_step"] is False
            for value in (2.0, 3.0):  # a second episode gets a fresh loop
                await conn.start_episode({"task": {"name": "t"}, "mode": "live"})
                action = await conn.act({"value": value})
                assert np.allclose(action["actions"], value)
                await conn.end_episode({})
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_ci_laas_skips_stale_actions(free_port):
    task = await _serve(CILAASChunkServer(), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}, "mode": "live"})
            action = await conn.act({"value": 1.0})
            chunk = np.arange(6 * 7, dtype=np.float32).reshape(6, 7)
            assert any(np.array_equal(action["actions"], chunk[k]) for k in (2, 3))  # never past chunk_size=4
            await conn.end_episode({})
    finally:
        await stop_server(task)


def test_pick_action_laas():
    server = CILAASChunkServer()
    chunk = np.arange(28, dtype=np.float32).reshape(4, 7)
    assert np.array_equal(server.pick_action(np.ones(7), time.monotonic()), np.ones(7))
    assert np.array_equal(server.pick_action(chunk, time.monotonic() - 0.25), chunk[2])
    assert np.array_equal(server.pick_action(chunk, time.monotonic() - 9.0), chunk[3])
    server.laas = False
    assert np.array_equal(server.pick_action(chunk, time.monotonic() - 0.25), chunk[0])


@pytest.mark.anyio
async def test_ci_infers_the_latest_observation(free_port):
    """Observations arriving during an inference are skipped except the newest."""
    server = CIEchoServer(delay=0.2)
    task = await _serve(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}, "mode": "live"})
            for value in (1.0, 2.0, 3.0):
                await conn.send_observation({"value": value})
            first = await conn.recv(timeout=2.0)
            second = await conn.recv(timeout=2.0)
            assert np.allclose(first.payload["actions"], 1.0) and np.allclose(second.payload["actions"], 3.0)
            assert server.calls == 2
            await conn.end_episode({})
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_ci_episode_end_drops_inflight_action(free_port):
    """EPISODE_END waits for the inference in flight; its action is never sent."""
    server = CIEchoServer(delay=0.3)
    task = await _serve(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}, "mode": "live"})
            await conn.send_observation({"value": 1.0})
            await anyio.sleep(0.05)
            await conn.end_episode({})
            await conn.start_episode({"task": {"name": "t"}, "mode": "live"})
            action = await conn.act({"value": 4.0})  # the first episode's action would arrive first
            assert np.allclose(action["actions"], 4.0)
            assert server.calls == 2
            await conn.end_episode({})
    finally:
        await stop_server(task)


class CIBatchServer(CIEchoServer):
    def __init__(self):
        super().__init__(max_batch_size=2, max_wait_time=0.5)
        self.sizes: list[int] = []

    def predict_batch(self, obs, rows):
        self.sizes.append(len(rows))
        return {"actions": np.asarray(obs["value"], np.float32)[:, None] * np.ones((1, 7), np.float32)}


@pytest.mark.anyio
async def test_ci_connections_batch(free_port):
    server = CIBatchServer()
    task = await _serve(server, free_port)
    try:
        conns = [Connection(f"ws://127.0.0.1:{free_port}") for _ in range(2)]
        for c in conns:
            await c.connect()
            await c.start_episode({"task": {"name": "t"}, "mode": "live"})
        actions = await asyncio.gather(*(c.act({"value": float(i + 1)}) for i, c in enumerate(conns)))
        assert [float(a["actions"][0]) for a in actions] == [1.0, 2.0]
        assert server.sizes == [2]
        for c in conns:
            await c.close()
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_live_runner_with_ci_server(free_port):
    task = await _serve(CIEchoServer(delay=0.02), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            result = await LiveEpisodeRunner(hz=100.0).run_episode(
                StubBenchmark(done_at_step=20), {"name": "t"}, conn, max_steps=50
            )
        assert result["metrics"]["success"] is True
        assert result["rt_metrics"]["fresh_count"] > 0  # pushed actions reached the env
    finally:
        await stop_server(task)


class CIEmptyChunkServer(CIEchoServer):
    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        if obs["value"] < 0:
            return {"actions": np.zeros((0, 7), np.float32)}
        return super().predict(obs, row)


@pytest.mark.anyio
async def test_ci_failure_errors_one_observation(free_port):
    task = await _serve(CIEmptyChunkServer(), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}, "mode": "live"})
            with pytest.raises(RuntimeError, match="observation_failed"):
                await conn.act({"value": -1.0})
            assert np.allclose((await conn.act({"value": 5.0}))["actions"], 5.0)
            await conn.end_episode({})
    finally:
        await stop_server(task)


class ThreadLogServer(PredictModelServer):
    def __init__(self):
        super().__init__(chunk_size=2)
        self.threads: set[int] = set()
        self.calls = 0

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        self.threads.add(threading.get_ident())
        self.calls += 1
        if self.calls == 1:
            time.sleep(1.5)  # past the first client's timeout
        return {"actions": np.ones((2, 7), np.float32)}

    def reset_episodes(self, rows):
        self.threads.add(threading.get_ident())

    def observe(self, obs, rows):
        self.threads.add(threading.get_ident())

    def finish_episodes(self, rows, results):
        self.threads.add(threading.get_ident())


@pytest.mark.anyio
async def test_model_calls_stay_on_one_thread(free_port):
    """A timed-out first call, reconnects and concurrent clients never move the model to another thread."""
    server = ThreadLogServer()
    task = await _serve(server, free_port)
    try:
        url = f"ws://127.0.0.1:{free_port}"
        async with Connection(url, timeout=0.5) as conn:
            await conn.start_episode({"task": {"name": "t"}})
            with pytest.raises(TimeoutError):
                await conn.act({"value": 1.0})

        async def client():
            async with Connection(url) as c:
                for _ in range(2):
                    await c.start_episode({"task": {"name": "t"}})
                    for _ in range(3):
                        await c.act({"value": 1.0})
                    await c.end_episode({})

        await asyncio.gather(client(), client())
        assert len(server.threads) == 1
    finally:
        await stop_server(task)


def test_a_second_ctrl_c_exits_without_waiting(tmp_path, free_port):
    """The first SIGINT waits for the predict() in flight; a second one exits at once."""
    import os
    import signal
    import subprocess
    import sys

    script = tmp_path / "srv.py"
    script.write_text(
        "import signal, time, numpy as np\n"
        "signal.signal(signal.SIGINT, signal.default_int_handler)\n"
        "from vla_eval.model_servers.predict import PredictModelServer\n"
        "from vla_eval.model_servers.serve import serve\n"
        "class S(PredictModelServer):\n"
        "    def predict(self, obs, row):\n"
        "        time.sleep(30)\n"
        "        return {'actions': np.zeros(7, np.float32)}\n"
        f"serve(S(), host='127.0.0.1', port={free_port})\n"
    )
    proc = subprocess.Popen([sys.executable, str(script)], env={**os.environ, "PYTHONPATH": os.getcwd()})
    try:
        url = f"ws://127.0.0.1:{free_port}"

        async def send() -> None:
            await wait_for_server(free_port)
            async with Connection(url, timeout=0.5) as conn:
                await conn.start_episode({"task": {"name": "t"}})
                with pytest.raises(TimeoutError):
                    await conn.act({"value": 1.0})

        anyio.run(send)
        proc.send_signal(signal.SIGINT)
        time.sleep(0.5)
        assert proc.poll() is None  # still finishing the call in flight
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=3) == 130
    finally:
        proc.kill()
