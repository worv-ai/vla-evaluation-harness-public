"""The model server over rows: vector steps, lifecycle in batches, chunk buffers per row, row ids."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.base import Row, Rows
from vla_eval.model_servers.predict import PredictModelServer

from tests.conftest import start_server, stop_server


class CountingServer(PredictModelServer):
    """Echoes ``value`` as a chunk of ``chunk_size`` actions; logs each batch's row ids and steps."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.batches: list[list[int]] = []
        self.steps: list[list[int]] = []
        self.started: list[list[int]] = []
        self.ended: list[list[int]] = []

    def predict_batch(self, obs, rows):
        self.batches.append(rows.ids.tolist())
        self.steps.append(rows.step.tolist())
        n = self.chunk_size or 1
        return {"actions": np.asarray(obs["value"])[:, None, None] * np.ones((len(rows), n, 7), np.float32)}

    def on_episode_start(self, rows: Rows):
        self.started.append(rows.ids.tolist())
        for e in rows.episodes:
            if "suite_chunk" in e.config.get("task", {}):
                e.chunk_size = e.config["task"]["suite_chunk"]
        broken = [i for i, e in enumerate(rows.episodes) if e.config.get("task", {}).get("broken")]
        return {i: ValueError("cannot load this task") for i in broken}

    def on_episode_end(self, rows: Rows, results):
        self.ended.append(rows.ids.tolist())


async def _connect(server: PredictModelServer, port: int) -> tuple[Any, Connection]:
    task = await start_server(server, port)
    conn = Connection(f"ws://127.0.0.1:{port}")
    await conn.connect()
    return task, conn


def _start(slot: int, **task: Any) -> list[Any]:
    return [slot, {"task": {"name": "t", **task}}]


def _step(slots: list[int], starts: list[list[Any]] | None = None, ends: list[list[Any]] | None = None) -> dict:
    obs = {"value": np.array([float(s + 1) for s in slots])}
    return {"ends": ends or [], "starts": starts or [], "slots": slots, "observations": obs}


def _actions(reply: dict[str, Any], slots: list[int]) -> dict[int, float]:
    """The reply's action rows, by slot: the slots without an error, in order."""
    failed = {s for s, _ in reply["errors"]}
    ok = [s for s in slots if s not in failed]
    rows = reply["actions"].get("actions", [])
    assert len(rows) == len(ok)
    return {s: float(np.asarray(a).reshape(-1)[0]) for s, a in zip(ok, rows)}


@pytest.mark.anyio
async def test_a_step_infers_its_rows_in_batches_of_max_batch_size(free_port):
    server = CountingServer(max_batch_size=2)
    task, conn = await _connect(server, free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        assert _actions(reply, [0, 1, 2]) == {0: 1.0, 1: 2.0, 2: 3.0} and reply["errors"] == []
        assert server.batches == [[0, 1], [2]] and server.started == [[0, 1, 2]]  # one start call for the step
        assert conn.server_info["capabilities"] == {"vector_step": True}
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_a_failed_start_fails_its_slot_alone(free_port):
    task, conn = await _connect(CountingServer(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(0), _start(1, broken=True), _start(2)]))
        assert _actions(reply, [0, 1, 2]) == {0: 1.0, 2: 3.0}  # the rows of the slots that did not fail
        ((slot, detail),) = reply["errors"]
        assert slot == 1 and "episode_start_failed: ValueError: cannot load this task" in detail
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_each_row_has_its_own_chunk_buffer_and_counts_every_observation(free_port):
    server = CountingServer(chunk_size=3)
    task, conn = await _connect(server, free_port)
    try:
        await conn.exchange(_step([0], [_start(0)]))
        # slot 1 starts a step later: its chunk is out of phase with slot 0's
        for k in range(5):
            reply = await conn.exchange(_step([0, 1], [_start(1)] if k == 0 else []))
            assert _actions(reply, [0, 1]) == {0: 1.0, 1: 2.0}
        assert server.batches == [[0], [1], [0], [1]]  # 6 and 5 steps of chunk 3: inferred at 0,3 / 1,4
        assert server.steps == [[0], [0], [3], [3]]  # buffered observations count
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_on_episode_start_sets_an_episodes_chunk_size(free_port):
    server = CountingServer(chunk_size=4, max_batch_size=None)
    task, conn = await _connect(server, free_port)
    try:
        await conn.exchange(_step([0, 1], [_start(0, suite_chunk=2), _start(1)]))
        for _ in range(3):
            await conn.exchange(_step([0, 1]))
        assert server.batches == [[0, 1], [0]]  # slot 0 refills after 2 steps, slot 1 after 4
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_rows_are_reused_once_their_episodes_end(free_port):
    server = CountingServer(max_batch_size=None)
    task, conn = await _connect(server, free_port)
    try:
        await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        await conn.exchange(_step([0, 2, 1], [_start(1)], ends=[[1, {"success": True}]]))
        assert server.ended == [[1]] and server.started[-1] == [1]  # slot 1's new episode takes row 1 again
        assert server.batches[-1] == [0, 2, 1] and server.steps[-1] == [1, 1, 0]
    finally:
        await conn.close()
        await stop_server(task)
    assert sorted(server.ended[-1]) == [0, 1, 2]  # the disconnect ended the open episodes


@pytest.mark.anyio
async def test_max_rows_refuses_more_episodes(free_port):
    task, conn = await _connect(CountingServer(max_rows=2), free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        assert [s for s, _ in reply["errors"]] == [2] and "at most 2 episodes" in reply["errors"][0][1]
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_an_observation_for_a_slot_without_an_episode_is_an_error(free_port):
    task, conn = await _connect(CountingServer(), free_port)
    try:
        await conn.exchange(_step([0], [_start(0)]))
        reply = await conn.exchange(_step([0], ends=[[0, {}]]))
        assert reply["actions"] == {} and "no running episode" in reply["errors"][0][1]
    finally:
        await conn.close()
        await stop_server(task)


class Columns(PredictModelServer):
    """Infers column batches directly; rejects a batch holding a negative value."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[int] = []

    def predict_batch(self, obs, rows):
        values = np.asarray(obs["value"])
        self.calls.append(len(values))
        if (values < 0).any():
            raise ValueError("negative value")
        return {"actions": np.repeat(values[:, None], 7, axis=1)}


@pytest.mark.anyio
async def test_predict_batch_takes_the_columns_in_slices(free_port):
    server = Columns(max_batch_size=2)
    task, conn = await _connect(server, free_port)
    try:
        reply = await conn.exchange(_step([4, 0, 2], [_start(s) for s in (0, 2, 4)]))
        assert isinstance(reply["actions"]["actions"], np.ndarray) and server.calls == [2, 1]
        assert _actions(reply, [4, 0, 2]) == {4: 5.0, 0: 1.0, 2: 3.0}  # in the request's order
        bad = _step([0, 2, 4])
        bad["observations"]["value"][0] = -1.0  # fails the first slice (slots 0 and 2) only
        reply = await conn.exchange(bad)
        assert sorted(s for s, _ in reply["errors"]) == [0, 2] and _actions(reply, [0, 2, 4]) == {4: 5.0}
    finally:
        await conn.close()
        await stop_server(task)


class ListOfDicts(PredictModelServer):
    """The old predict_batch contract: a list of dicts."""

    def predict_batch(self, obs, rows):
        return [{"actions": np.zeros(7)} for _ in range(len(rows) + 1)]


@pytest.mark.anyio
async def test_an_old_style_predict_batch_names_the_change(free_port):
    task, conn = await _connect(ListOfDicts(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1], [_start(0), _start(1)]))
        assert len(reply["errors"]) == 2 and "column batches" in reply["errors"][0][1]
    finally:
        await conn.close()
        await stop_server(task)


class PerRow(PredictModelServer):
    """``predict`` per row: chunks of different lengths, keys in other orders, and a failing row."""

    def predict(self, obs, row: Row):
        v = float(obs["value"])
        if v == 3.0:
            raise ValueError("row 2 fails")
        a, meta = np.ones((int(v), 7), dtype=np.float32), {"row": row.id}
        return {"meta": meta, "actions": a} if v == 2.0 else {"actions": a, "meta": meta}


@pytest.mark.anyio
async def test_predict_per_row_isolates_a_failure_and_keeps_ragged_actions(free_port):
    task, conn = await _connect(PerRow(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        assert [s for s, _ in reply["errors"]] == [2]
        assert [np.asarray(a).shape for a in reply["actions"]["actions"]] == [(1, 7), (2, 7)]
        assert reply["actions"]["meta"] == {"row": [0, 1]}  # a nested dict column
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_a_vector_step_does_not_wait_for_more_requests(free_port):
    import time

    server = CountingServer(max_batch_size=8, max_wait_time=2.0)
    task, conn = await _connect(server, free_port)
    try:
        t0 = time.monotonic()
        await conn.exchange(_step([0, 1], [_start(0), _start(1)]))
        assert time.monotonic() - t0 < 1.0  # fewer rows than max_batch_size, yet no 2 s wait
    finally:
        await conn.close()
        await stop_server(task)


class EmptyChunk(PredictModelServer):
    """Slot 1's chunk is empty."""

    def __init__(self) -> None:
        super().__init__(chunk_size=3, max_batch_size=None)

    def predict_batch(self, obs, rows):
        return {"actions": [np.ones((0 if v == 2.0 else 3, 7), np.float32) for v in np.asarray(obs["value"])]}


@pytest.mark.anyio
async def test_a_bad_chunk_fails_its_row_alone(free_port):
    task, conn = await _connect(EmptyChunk(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        assert [s for s, _ in reply["errors"]] == [1] and len(reply["actions"]["actions"]) == 2
    finally:
        await conn.close()
        await stop_server(task)


class NonNumeric(PredictModelServer):
    def predict(self, obs, row: Row):
        v = float(obs["value"])
        return {"actions": "oops"} if v == 2.0 else {} if v == 4.0 else {"actions": np.ones(7)}


@pytest.mark.anyio
async def test_a_row_with_unusable_actions_fails_alone(free_port):
    task, conn = await _connect(NonNumeric(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2, 3], [_start(s) for s in range(4)]))
        assert sorted(s for s, _ in reply["errors"]) == [1, 3]  # unusable actions, other keys
        assert _actions(reply, [0, 1, 2, 3]) == {0: 1.0, 2: 1.0}
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_buffered_actions_do_not_wait_for_a_batch(free_port):
    import time

    server = CountingServer(chunk_size=3, max_batch_size=8, max_wait_time=0.5)
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({"task": {"name": "t"}})
            await conn.act({"value": 1.0})  # infers after waiting for a batch
            t0 = time.monotonic()
            await conn.act({"value": 1.0})
            await conn.act({"value": 1.0})
            assert time.monotonic() - t0 < 0.4 and len(server.batches) == 1  # served from the buffer at once
    finally:
        await stop_server(task)
