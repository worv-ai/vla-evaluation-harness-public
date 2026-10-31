"""The callback contract: send_action/send_error over Rows snapshots; observe sees every observation."""

from __future__ import annotations

import time
from typing import Any

import anyio
import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.base import ModelServer, Row, Rows
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
# PredictModelServer: observe sees every observation once, in order, before it is inferred
# ---------------------------------------------------------------------------


class HistoryServer(PredictModelServer):
    """observe() keeps each row's value history; predict() logs what it saw with its observation."""

    def __init__(self, delay: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.delay = delay
        self.history: dict[int, list[float]] = {}
        self.predicted: list[tuple[float, list[float], int]] = []
        self.finished: list[list[float]] = []

    def reset_episodes(self, rows: Rows) -> None:
        for rid in rows.ids.tolist():
            self.history[rid] = []

    def observe(self, obs, rows: Rows) -> None:
        for rid, v in zip(rows.ids.tolist(), np.asarray(obs["value"]).tolist()):
            self.history[rid].append(v)

    def finish_episodes(self, rows: Rows, results) -> None:
        self.finished += [list(self.history[rid]) for rid in rows.ids.tolist()]

    def predict(self, obs: dict[str, Any], row: Row) -> dict[str, Any]:
        self.predicted.append((float(obs["value"]), list(self.history[row.id]), row.step))
        time.sleep(self.delay)
        n = self.chunk_size or 1
        return {"actions": float(obs["value"]) * np.ones((n, 7), np.float32)}


@pytest.mark.anyio
async def test_observe_sees_buffered_observations_once_in_order(free_port):
    server = HistoryServer(chunk_size=3)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            for v in range(1, 6):
                await conn.act({"value": float(v)})
        assert server.history[0] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert server.predicted == [(1.0, [1.0], 0), (4.0, [1.0, 2.0, 3.0, 4.0], 3)]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_ci_observes_coalesced_observations_and_infers_aligned_history(free_port):
    """An observation arriving during an inference is observed only before the next one: history matches obs."""
    server = HistoryServer(delay=0.3, continuous_inference=True)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode(LIVE)
            await conn.send_observation({"value": 1.0})
            await anyio.sleep(0.1)
            for v in (2.0, 3.0):
                await conn.send_observation({"value": v})
            got = [_value((await conn.recv(timeout=2.0)).payload) for _ in range(2)]
            await conn.end_episode({})
        assert got == [1.0, 3.0]
        # step counts every received observation, the coalesced one included
        assert server.predicted == [(1.0, [1.0], 0), (3.0, [1.0, 2.0, 3.0], 2)]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_ci_observes_every_observation_before_the_episode_ends(free_port):
    server = HistoryServer(delay=0.3, continuous_inference=True)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode(LIVE)
            await conn.send_observation({"value": 1.0})
            await anyio.sleep(0.1)
            await conn.send_observation({"value": 2.0})  # arrives during the inference of 1.0
            await conn.end_episode({})
            await conn.start_episode(LIVE)  # waits for the end: an inference was in flight
            assert _value(await conn.act({"value": 3.0})) == 3.0
        assert server.finished[0] == [1.0, 2.0]
    finally:
        await stop_server(task)


class FailObserveServer(HistoryServer):
    def observe(self, obs, rows: Rows) -> None:
        if float(np.asarray(obs["value"])[0]) == 2.0:
            raise ValueError("cannot observe 2")
        super().observe(obs, rows)


@pytest.mark.anyio
async def test_ci_reports_a_skipped_observations_failure_as_its_own(free_port):
    """Observations 2 and 3 arrive during an inference; observe(2) fails: 2 gets the error, 3 is still inferred."""
    server = FailObserveServer(delay=0.3, continuous_inference=True)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode(LIVE)
            await conn.send_observation({"value": 1.0})
            await anyio.sleep(0.1)
            seq2 = await conn.send(MessageType.OBSERVATION, {"value": 2.0})
            seq3 = await conn.send(MessageType.OBSERVATION, {"value": 3.0})
            got = [await conn.recv(timeout=2.0) for _ in range(3)]
            await conn.end_episode({})
        assert [(m.type, m.seq) for m in got[1:]] == [(MessageType.ERROR, seq2), (MessageType.ACTION, seq3)]
        assert "cannot observe 2" in got[1].payload["error"] and _value(got[2].payload) == 3.0
    finally:
        await stop_server(task)


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
        assert server.predicted == [(2.0, [-1.0, 2.0], 1)]
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
        assert server.history == {0: [1.0, 2.0, 3.0], 1: [2.0, 3.0]}
    finally:
        await stop_server(task)


class BrokenStartServer(HistoryServer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.finished: list[list[int]] = []

    def reset_episodes(self, rows: Rows):
        super().reset_episodes(rows)
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


def test_old_hook_names_fail_loudly():
    with pytest.raises(TypeError, match="override reset_episodes"):

        class Old(PredictModelServer):
            def on_episode_start(self, rows):
                pass


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
