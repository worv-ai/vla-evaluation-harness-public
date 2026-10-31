"""The callback contract (send_action/send_error over Rows snapshots) and the servers built on it."""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import anyio
import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.base import ModelServer, Row, Rows
from vla_eval.model_servers.live import LiveModelServer
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.protocol.messages import MessageType

from tests.conftest import start_server, stop_server

LIVE = {"task": {"name": "t"}, "mode": "live"}


def _value(action: dict[str, Any]) -> float:
    return float(np.asarray(action["actions"]).reshape(-1)[0])


def _act(rows: Rows, values: Any) -> dict[str, Any]:
    return {"actions": np.asarray(values, np.float32).reshape(len(rows), 1) * np.ones((1, 7), np.float32)}


class ReplayServer(ModelServer):
    """A bare ModelServer that keeps every Rows it got, to replay stale replies with them."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[np.ndarray, Rows]] = []
        self.ended: list[dict[str, Any]] = []
        self.start_error: Exception | None = None
        self.mode = "echo"

    async def on_episode_start(self, rows: Rows) -> None:
        try:
            await self.send_action(rows, _act(rows, [0.0] * len(rows)))
        except RuntimeError as exc:
            self.start_error = exc

    async def on_observation(self, obs: dict[str, Any], rows: Rows) -> None:
        values = np.asarray(obs["value"], np.float32)
        if self.seen and self.mode == "replay":  # the previous observation's rows answer again, first
            old_values, old_rows = self.seen[-1]
            await self.send_action(old_rows, _act(old_rows, old_values + 100))
        self.seen.append((values, rows))
        if self.mode == "repeat":  # row 0 listed twice in one reply: it answers row 0 once, never row 1
            await self.send_action(rows.take([0, 0]), _act(rows.take([0, 0]), [values[0], -1.0]))
            assert rows._pending is not None and not rows._pending.done.is_set()
            await self.send_action(rows.take([1]), _act(rows.take([1]), values[1:]))
            return
        if self.mode == "partial":
            await self.send_action(rows.take([0]), _act(rows.take([0]), values[:1]))
            raise ValueError("the rest failed")
        if self.mode != "silent":
            await self.send_action(rows, _act(rows, values))
            await self.send_action(rows, _act(rows, values + 1000))  # a duplicate: dropped

    async def on_episode_end(self, rows: Rows, results: list[dict[str, Any]]) -> None:
        self.ended += results


@pytest.mark.anyio
async def test_a_duplicate_or_delayed_reply_never_answers_another_observation(free_port):
    server = ReplayServer()
    server.mode = "replay"
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            assert [_value(await conn.act({"value": v})) for v in (1.0, 2.0, 3.0)] == [1.0, 2.0, 3.0]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_reply_through_a_reused_row_is_dropped(free_port):
    server = ReplayServer()
    server.mode = "replay"
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            reply = await conn.exchange(_vstep([0], [[0, {}]], 1.0))
            reply = await conn.exchange(_vstep([0], [[0, {}]], 2.0))  # slot 0 restarts on the same row id
            assert server.seen[0][1].ids.tolist() == server.seen[1][1].ids.tolist() == [0]
            assert _value({"actions": reply["actions"]["actions"][0]}) == 2.0 and reply["errors"] == []
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_reply_without_an_observation_is_rejected(free_port):
    server = ReplayServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            assert _value(await conn.act({"value": 4.0})) == 4.0
        assert isinstance(server.start_error, RuntimeError)
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_rows_answered_before_an_exception_keep_their_actions(free_port):
    server = ReplayServer()
    server.mode = "partial"
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            reply = await conn.exchange(_vstep([0, 1, 2], [[s, {}] for s in range(3)], [1.0, 2.0, 3.0]))
        assert [_value({"actions": a}) for a in reply["actions"]["actions"]] == [1.0]
        assert sorted(s for s, _ in reply["errors"]) == [1, 2]
        assert all("ValueError: the rest failed" in e for _, e in reply["errors"])
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_row_listed_twice_in_a_reply_is_answered_once(free_port):
    server = ReplayServer()
    server.mode = "repeat"
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            reply = await conn.exchange(_vstep([0, 1], [[0, {}], [1, {}]], [1.0, 2.0]))
        assert [_value({"actions": a}) for a in reply["actions"]["actions"]] == [1.0, 2.0] and reply["errors"] == []
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_disconnect_ends_an_episode_waiting_for_a_reply(free_port):
    server = ReplayServer()
    server.mode = "silent"
    task = await start_server(server, free_port)
    try:
        conn = Connection(f"ws://127.0.0.1:{free_port}", timeout=0.2)
        await conn.connect()
        await conn.start_episode({"task": {"name": "t"}})
        with pytest.raises(TimeoutError):
            await conn.act({"value": 1.0})
        await conn.close()
        with anyio.fail_after(3):
            while not server.ended:
                await anyio.sleep(0.02)
        assert server.ended == [{"status": "aborted", "failure_reason": "connection_closed"}]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_an_episode_end_sent_before_a_disconnect_still_counts(free_port):
    """The harness times out, sends EPISODE_END with its status, then closes: the status reaches the server."""
    server = ReplayServer()
    server.mode = "silent"
    task = await start_server(server, free_port)
    try:
        conn = Connection(f"ws://127.0.0.1:{free_port}", timeout=0.2)
        await conn.connect()
        await conn.start_episode({"task": {"name": "t"}})
        with pytest.raises(TimeoutError):
            await conn.act({"value": 1.0})
        await conn.end_episode({"status": "error", "failure_reason": "timeout"})
        await conn.close()
        with anyio.fail_after(3):
            while not server.ended:
                await anyio.sleep(0.02)
        assert server.ended == [{"status": "error", "failure_reason": "timeout"}]
    finally:
        await stop_server(task)


def _vstep(slots: list[int], starts: list, values: Any) -> dict[str, Any]:
    v = np.broadcast_to(np.asarray(values, np.float64), (len(slots),)).copy()
    return {"ends": [], "starts": starts, "slots": slots, "observations": {"value": v}}


# ---------------------------------------------------------------------------
# PredictModelServer
# ---------------------------------------------------------------------------


class HistoryServer(PredictModelServer):
    """predict() logs each observation it infers with the row's step; reset/finish log the rows."""

    def __init__(self, delay: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delay = delay
        self.predicted: list[tuple[float, int]] = []

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        self.predicted.append((float(obs["value"]), row.step))
        time.sleep(self.delay)
        n = self.chunk_size or 1
        return {"actions": float(obs["value"]) * np.ones((n, 7), np.float32)}


class FailOnceServer(HistoryServer):
    def predict(self, obs, row):
        if obs["value"] < 0:
            raise ValueError("bad")
        return super().predict(obs, row)


@pytest.mark.anyio
async def test_a_failed_observation_still_counts_as_a_step(free_port):
    server = FailOnceServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            with pytest.raises(RuntimeError, match="observation_failed"):
                await conn.act({"value": -1.0})
            await conn.act({"value": 2.0})
        assert server.predicted == [(2.0, 1)]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_vector_step_never_waits_for_a_batch(free_port):
    """One slot left, or some slots served from their buffers: no max_wait_time stall."""
    server = HistoryServer(chunk_size=2, max_batch_size=8, max_wait_time=5.0)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            t0 = time.monotonic()
            await conn.exchange(_vstep([0], [[0, {}]], 1.0))  # one row
            await conn.exchange(_vstep([0, 1], [[1, {}]], 2.0))  # slot 0 buffered, slot 1 inferred
            await conn.exchange(_vstep([0, 1], [], 3.0))  # slot 0 inferred, slot 1 buffered
            assert time.monotonic() - t0 < 2.0
        assert server.predicted == [(1.0, 0), (2.0, 0), (3.0, 2)]
    finally:
        await stop_server(task)


class BrokenStartServer(HistoryServer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.finished: list[list[int]] = []

    def reset_episodes(self, rows: Rows):
        return {i: ValueError("no") for i, e in enumerate(rows.episodes) if e.config["task"].get("broken")}

    def finish_episodes(self, rows, results):
        self.finished.append(rows.ids.tolist())


@pytest.mark.anyio
async def test_a_failed_start_frees_its_row_without_an_end(free_port):
    server = BrokenStartServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            reply = await conn.exchange(_vstep([0, 1], [[0, {"task": {"broken": True}}], [1, {"task": {}}]], 1.0))
            assert [s for s, _ in reply["errors"]] == [0]
            reply = await conn.exchange(_vstep([0], [[0, {"task": {}}]], 2.0))
            assert reply["errors"] == []
        with anyio.fail_after(3):  # the disconnect ends both
            while not server.finished:
                await anyio.sleep(0.02)
        assert server.finished == [[0, 1]]  # row 0 was free again; the broken start never ended
    finally:
        await stop_server(task)


class SlowEndLogServer(HistoryServer):
    def __init__(self) -> None:
        super().__init__(delay=0.3)
        self.results: list[dict[str, Any]] = []

    def finish_episodes(self, rows: Rows, results) -> None:
        self.results += results


@pytest.mark.anyio
async def test_a_vector_end_sent_before_a_disconnect_still_counts(free_port):
    """A step still inferring when the client closes; the ends it sent after it keep their status."""
    server = SlowEndLogServer()
    task = await start_server(server, free_port)
    try:
        conn = Connection(f"ws://127.0.0.1:{free_port}")
        await conn.connect()
        await conn.send(MessageType.VECTOR_STEP, _vstep([0], [[0, {}]], 1.0))
        end = {"status": "error", "failure_reason": "timeout"}
        await conn.send(MessageType.VECTOR_STEP, {"ends": [[0, end]], "starts": [], "slots": [], "observations": {}})
        await conn.close()
        with anyio.fail_after(3):
            while not server.results:
                await anyio.sleep(0.02)
        assert server.results == [end]
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# LiveModelServer: the latest observation of each episode, inferred in a loop
# ---------------------------------------------------------------------------


class LiveLogServer(LiveModelServer):
    def __init__(self, delay: float = 0.0, out: str = "echo", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delay, self.out = delay, out
        self.predicted: list[tuple[float, int]] = []
        self.threads: set[str] = set()
        self.busy = False
        self.finished_busy: list[bool] = []

    def reset_episodes(self, rows: Rows) -> None:
        self.threads.add(threading.current_thread().name)

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        self.threads.add(threading.current_thread().name)
        v = float(obs["value"])
        self.predicted.append((v, row.step))
        self.busy = True
        time.sleep(self.delay)
        self.busy = False
        if v < 0:
            raise ValueError("boom")
        if self.out == "chunk":
            return {"actions": v + np.arange(42, dtype=np.float64).reshape(6, 7)}
        if self.out == "noactions":
            return {"foo": v}
        return {"actions": v * np.ones(7)}

    def finish_episodes(self, rows: Rows, results) -> None:
        self.threads.add(threading.current_thread().name)
        self.finished_busy.append(self.busy)


async def _live(server: LiveModelServer, port: int, values: list[float], gap: float = 0.0) -> list[Any]:
    """Send ``values`` in one live episode (``gap`` apart), then collect the replies until they stop."""
    task = await start_server(server, port)
    try:
        async with Connection(f"ws://127.0.0.1:{port}") as conn:
            assert conn.server_info["capabilities"]["vector_step"] is False
            await conn.start_episode(LIVE)
            seqs = []
            for v in values:
                seqs.append(await conn.send(MessageType.OBSERVATION, {"value": v}))
                await anyio.sleep(gap)
            got = []
            while True:
                try:
                    m = await conn.recv(timeout=1.0)
                except Exception:
                    break
                got.append((m.type, seqs.index(m.seq), m.payload))
            await conn.end_episode({})
        return got
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_live_coalesces_to_the_latest_observation_and_counts_every_step(free_port):
    server = LiveLogServer(delay=0.3)
    got = await _live(server, free_port, [1.0, 2.0, 3.0])
    assert [(t, i, _value(p)) for t, i, p in got] == [(MessageType.ACTION, 0, 1.0), (MessageType.ACTION, 2, 3.0)]
    assert server.predicted == [(1.0, 0), (3.0, 2)]


@pytest.mark.anyio
async def test_live_sends_a_chunks_first_action_as_float32(free_port):
    got = await _live(LiveLogServer(out="chunk", chunk_size=4), free_port, [1.0], gap=0.2)
    (_, _, payload), chunk = got[0], 1.0 + np.arange(42).reshape(6, 7)
    assert np.asarray(payload["actions"]).dtype == np.float32 and np.array_equal(payload["actions"], chunk[0])


@pytest.mark.parametrize("chunk_size", [4, None])
@pytest.mark.anyio
async def test_live_laas_counts_latency_from_admission(free_port, chunk_size):
    server = LiveLogServer(delay=0.25, out="chunk", chunk_size=chunk_size, laas=True, hz=10.0)
    got = await _live(server, free_port, [1.0])
    chunk = 1.0 + np.arange(42).reshape(6, 7)
    assert any(np.array_equal(got[0][2]["actions"], chunk[k]) for k in (2, 3))


@pytest.mark.anyio
async def test_live_passes_an_action_without_actions_through(free_port):
    got = await _live(LiveLogServer(out="noactions"), free_port, [1.0, 2.0], gap=0.2)
    assert [(i, p) for _, i, p in got] == [(0, {"foo": 1.0}), (1, {"foo": 2.0})]


@pytest.mark.anyio
async def test_live_failure_errors_its_observation_and_the_loop_goes_on(free_port):
    got = await _live(LiveLogServer(), free_port, [-1.0, 2.0], gap=0.2)
    assert [(t, i) for t, i, _ in got] == [(MessageType.ERROR, 0), (MessageType.ACTION, 1)]
    assert "boom" in got[0][2]["error"] and _value(got[1][2]) == 2.0


class SelectLastServer(LiveLogServer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.latencies: list[float] = []

    def select(self, action, latency, row):
        self.latencies.append(latency)
        return {"actions": np.asarray(action["actions"][-1], np.float32)}


@pytest.mark.anyio
async def test_live_select_chooses_the_action_after_the_inference(free_port):
    server = SelectLastServer(delay=0.2, out="chunk")
    got = await _live(server, free_port, [1.0])
    assert np.array_equal(got[0][2]["actions"], 1.0 + np.arange(35, 42))
    assert server.latencies[0] >= 0.2


@pytest.mark.anyio
async def test_live_episode_end_waits_for_the_inference_in_flight(free_port):
    server = LiveLogServer(delay=0.3)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode(LIVE)
            await conn.send_observation({"value": 1.0})
            await anyio.sleep(0.1)
            await conn.end_episode({})
            await conn.start_episode(LIVE)  # the same row id, once the first episode finished
            assert _value(await conn.act({"value": 2.0})) == 2.0
        assert server.finished_busy[:1] == [False]
        assert server.threads == {"model_0"}
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# VLANeXt on the base: dense state history, its own chunk queue, per-row errors
# ---------------------------------------------------------------------------


@pytest.fixture
def vlanext(monkeypatch):
    """VLANeXtModelServer with torch and the model stubbed: the chunk is ``value + arange``, the history logged."""
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.delitem(sys.modules, "vla_eval.model_servers.vlanext", raising=False)
    from vla_eval.model_servers import vlanext as m

    monkeypatch.setitem(sys.modules, "vla_eval.model_servers.vlanext", m)  # dropped again after the test
    log: list[tuple[float, list[float], int, str]] = []

    def predict_chunk(self, obs, row):
        v = float(obs["value"])
        log.append(
            (v, [float(s[0]) for s in self._state_histories[row.id]], row.step, threading.current_thread().name)
        )
        if v < 0:
            raise ValueError("bad state")
        return {"actions": v + np.arange(56, dtype=np.float64).reshape(8, 7)}

    monkeypatch.setattr(m.VLANeXtModelServer, "_init_model", lambda self: None)
    monkeypatch.setattr(m.VLANeXtModelServer, "_predict_chunk", predict_chunk)
    return m.VLANeXtModelServer, log


def _vobs(values: Any) -> dict[str, Any]:
    v = np.atleast_1d(np.asarray(values, np.float64))
    return {"value": v, "state": v[:, None] * np.ones((1, 7))}


@pytest.mark.anyio
async def test_vlanext_history_is_dense_and_buffered_steps_come_from_its_queue(vlanext, free_port):
    cls, log = vlanext
    task = await start_server(cls("x", chunk_size=3), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            got = [await conn.act({"value": float(v), "state": np.full(7, float(v))}) for v in range(1, 6)]
            await conn.start_episode({"task": {"name": "t"}})  # a new episode starts a new history
            await conn.act({"value": 9.0, "state": np.full(7, 9.0)})
        chunk = np.arange(56, dtype=np.float32).reshape(8, 7)
        expected = [1 + chunk[0], 1 + chunk[1], 1 + chunk[2], 4 + chunk[0], 4 + chunk[1]]
        assert all(np.array_equal(a["actions"], e) and a["actions"].dtype == np.float32 for a, e in zip(got, expected))
        assert [x[:3] for x in log] == [(1.0, [1.0], 0), (4.0, [1.0, 2.0, 3.0, 4.0], 3), (9.0, [9.0], 0)]
        assert {x[3] for x in log} == {"model_0"}
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_vlanext_a_rows_failure_is_its_own_in_a_vector_step(vlanext, free_port):
    cls, log = vlanext
    task = await start_server(cls("x", chunk_size=2), free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            step = {"ends": [], "starts": [[s, {}] for s in range(3)], "slots": [0, 1, 2]}
            reply = await conn.exchange({**step, "observations": _vobs([1.0, -1.0, 3.0])})
            assert [s for s, _ in reply["errors"]] == [1] and "bad state" in reply["errors"][0][1]
            assert [float(a[0]) for a in reply["actions"]["actions"]] == [1.0, 3.0]
            reply = await conn.exchange({**step, "starts": [], "observations": _vobs([2.0, 5.0, 4.0])})
            assert reply["errors"] == []
            assert [float(a[0]) for a in reply["actions"]["actions"]] == [1.0 + 7, 5.0, 3.0 + 7]
        assert [x[:2] for x in log] == [(1.0, [1.0]), (-1.0, [-1.0]), (3.0, [3.0]), (5.0, [-1.0, 5.0])]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_a_cancelled_model_call_never_runs_or_returns_only_after_it_ran():
    server = ReplayServer()
    ran: list[str] = []

    def slow(tag: str) -> None:
        time.sleep(0.2)
        ran.append(tag)

    with anyio.CancelScope() as scope:
        scope.cancel()
        await server.run_on_model_thread(slow, "cancelled before")
    with anyio.move_on_after(0.05):
        await server.run_on_model_thread(slow, "cancelled during")
    assert ran == ["cancelled during"]  # the await outlived the call
    await anyio.sleep(0.3)
    assert ran == ["cancelled during"]


@pytest.mark.anyio
async def test_a_server_without_vector_step_refuses_it(free_port):
    task = await start_server(LiveLogServer(), free_port)
    conn = Connection(f"ws://127.0.0.1:{free_port}")
    await conn.connect()
    try:
        assert conn.server_info["capabilities"] == {"vector_step": False}
        step = {"ends": [], "starts": [[0, {"task": {"name": "t"}}]], "slots": [0], "observations": {"v": [1.0]}}
        with pytest.raises(RuntimeError, match="vector_step_unsupported"):
            await conn.exchange(step)
    finally:
        await conn.close()
        await stop_server(task)
