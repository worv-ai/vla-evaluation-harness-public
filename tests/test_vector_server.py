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


def _obs(slots: list[int]) -> list[list[Any]]:
    return [[s, {"value": float(s + 1)}] for s in slots]


def _actions(reply: dict[str, Any]) -> dict[int, float]:
    return {s: float(np.asarray(a["actions"]).reshape(-1)[0]) for s, a in reply["actions"]}


@pytest.mark.anyio
async def test_a_step_infers_its_slots_in_batches_of_max_batch_size(free_port):
    server = CountingServer(max_batch_size=2)
    task, conn = await _connect(server, free_port)
    try:
        reply = await conn.exchange(
            {"ends": [], "starts": [_start(s) for s in range(3)], "observations": _obs([0, 1, 2])}
        )
        assert _actions(reply) == {0: 1.0, 1: 2.0, 2: 3.0} and reply["errors"] == []
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
        step = {"ends": [], "starts": [_start(0), _start(1, broken=True)], "observations": _obs([0, 1])}
        reply = await conn.exchange(step)
        assert _actions(reply) == {0: 1.0}
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
        await conn.exchange({"ends": [], "starts": [_start(0)], "observations": _obs([0])})
        # slot 1 starts a step later: its chunk is out of phase with slot 0's
        for _ in range(5):
            reply = await conn.exchange(
                {"ends": [], "starts": [_start(1)] if _ == 0 else [], "observations": _obs([0, 1])}
            )
            assert _actions(reply) == {0: 1.0, 1: 2.0}
        assert [len(b) for b in server.batches] == [1, 1, 1, 1]  # 6 and 5 steps of chunk 3: inferred at 0,3 / 1,4
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_a_subclass_on_observation_sees_every_item(free_port):
    server = OwnObservation()
    task, conn = await _connect(server, free_port)
    try:
        reply = await conn.exchange({"ends": [], "starts": [_start(0), _start(1)], "observations": _obs([0, 1])})
        assert _actions(reply) == {0: 1.0, 1: 2.0} and sorted(server.seen) == [1.0, 2.0]
    finally:
        await conn.close()
        await stop_server(task)


@pytest.mark.anyio
async def test_an_observation_for_a_slot_without_an_episode_is_an_error(free_port):
    task, conn = await _connect(CountingServer(max_batch_size=2), free_port)
    try:
        await conn.exchange({"ends": [], "starts": [_start(0)], "observations": _obs([0])})
        reply = await conn.exchange({"ends": [[0, {}]], "starts": [], "observations": _obs([0])})
        assert reply["actions"] == [] and "no running episode" in reply["errors"][0][1]
    finally:
        await conn.close()
        await stop_server(task)
