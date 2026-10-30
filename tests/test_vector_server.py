"""The server side of a vector step: one VECTOR_STEP per step, each slot its own session."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vla_eval.connection import Connection
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer

from tests.conftest import start_server, stop_server


class CountingServer(PredictModelServer):
    """Echoes ``value`` as a chunk of ``chunk_size`` actions; logs each batch's sessions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.batches: list[list[str]] = []

    def predict_batch(self, obs_batch, ctx_batch):
        self.batches.append([c.session_id for c in ctx_batch])
        n = self.chunk_size or 1
        return [{"actions": o["value"] * np.ones((n, 7), dtype=np.float32)} for o in obs_batch]

    async def on_episode_start(self, config: dict[str, Any], ctx: SessionContext) -> None:
        if config.get("task", {}).get("broken"):
            raise ValueError("cannot load this task")
        await super().on_episode_start(config, ctx)


class OwnObservation(PredictModelServer):
    """A subclass that handles each observation itself, as VLANeXt does to keep a dense history."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[float] = []

    async def on_observation(self, obs, ctx):
        self.seen.append(obs["value"])
        await ctx.send_action({"actions": np.full(7, obs["value"], dtype=np.float32)})


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
async def test_a_step_infers_its_slots_in_batches_of_max_batch_size(free_port):
    server = CountingServer(max_batch_size=2)
    task, conn = await _connect(server, free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(s) for s in range(3)]))
        assert _actions(reply, [0, 1, 2]) == {0: 1.0, 1: 2.0, 2: 3.0} and reply["errors"] == []
        assert [len(b) for b in server.batches] == [2, 1]
        assert len({sid for b in server.batches for sid in b}) == 3  # a session per slot
        assert conn.server_info["capabilities"] == {"vector_step": True}
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_a_failed_start_fails_its_slot_alone(free_port):
    task, conn = await _connect(CountingServer(max_batch_size=4), free_port)
    try:
        reply = await conn.exchange(_step([0, 1, 2], [_start(0), _start(1, broken=True), _start(2)]))
        assert _actions(reply, [0, 1, 2]) == {0: 1.0, 2: 3.0}  # the rows of the slots that did not fail
        ((slot, detail),) = reply["errors"]
        assert slot == 1 and "episode_start_failed: ValueError: cannot load this task" in detail
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_each_slot_has_its_own_chunk_buffer(free_port):
    server = CountingServer(max_batch_size=4, chunk_size=3)
    task, conn = await _connect(server, free_port)
    try:
        await conn.exchange(_step([0], [_start(0)]))
        # slot 1 starts a step later: its chunk is out of phase with slot 0's
        for _ in range(5):
            reply = await conn.exchange(_step([0, 1], [_start(1)] if _ == 0 else []))
            assert _actions(reply, [0, 1]) == {0: 1.0, 1: 2.0}
        assert [len(b) for b in server.batches] == [1, 1, 1, 1]  # 6 and 5 steps of chunk 3: inferred at 0,3 / 1,4
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_a_subclass_on_observation_sees_every_item(free_port):
    server = OwnObservation()
    task, conn = await _connect(server, free_port)
    try:
        reply = await conn.exchange(_step([0, 1], [_start(0), _start(1)]))
        assert _actions(reply, [0, 1]) == {0: 1.0, 1: 2.0} and sorted(server.seen) == [1.0, 2.0]
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_an_observation_for_a_slot_without_an_episode_is_an_error(free_port):
    task, conn = await _connect(CountingServer(max_batch_size=2), free_port)
    try:
        await conn.exchange(_step([0], [_start(0)]))
        reply = await conn.exchange(_step([0], ends=[[0, {}]]))
        assert reply["actions"] == {} and "no running episode" in reply["errors"][0][1]
    finally:
        await conn.close()
        await stop_server(task)


class Stacked(PredictModelServer):
    """Infers column batches directly; rejects a batch holding a negative value."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[int] = []

    def predict_stacked(self, obs, ctx_batch):
        values = np.asarray(obs["value"])
        self.calls.append(len(values))
        if (values < 0).any():
            raise ValueError("negative value")
        return {"actions": np.repeat(values[:, None], 7, axis=1)}


@pytest.mark.anyio
async def test_predict_stacked_takes_the_columns_in_slices(free_port):
    server = Stacked(max_batch_size=2)
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


class Ragged(PredictModelServer):
    """Chunks of different lengths per slot: the action column stays a list."""

    def predict(self, obs, ctx):
        n = int(obs["value"])
        return {"actions": np.ones((n, 7), dtype=np.float32)}


@pytest.mark.anyio
async def test_actions_of_different_shapes_travel_as_a_list(free_port):
    task, conn = await _connect(Ragged(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1], [_start(0), _start(1)]))
        assert [np.asarray(a).shape for a in reply["actions"]["actions"]] == [(1, 7), (2, 7)]
    finally:
        await conn.close()
        await stop_server(task)


class Reordered(PredictModelServer):
    """Slot 1 builds its action dict in another key order."""

    def predict(self, obs, ctx):
        a, m = np.full(7, obs["value"], dtype=np.float32), {"slot": int(obs["value"])}
        return {"meta": m, "actions": a} if obs["value"] == 2.0 else {"actions": a, "meta": m}


@pytest.mark.anyio
async def test_actions_with_the_same_keys_in_another_order_are_one_batch(free_port):
    task, conn = await _connect(Reordered(), free_port)
    try:
        reply = await conn.exchange(_step([0, 1], [_start(0), _start(1)]))
        assert reply["errors"] == [] and [m["slot"] for m in reply["actions"]["meta"]] == [1, 2]
    finally:
        await conn.close()
        await stop_server(task)
