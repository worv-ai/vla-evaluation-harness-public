"""The harness side of the callback contract: every started server episode ends with its status, and a
connection never hands an earlier episode's reply to the current one."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import anyio
import numpy as np
import pytest
from websockets.asyncio.server import serve as ws_serve

from vla_eval.connection import Connection
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.orchestrator import Orchestrator
from vla_eval.protocol.messages import Message, MessageType, pack_message, unpack_message
from vla_eval.runners.live_runner import LiveEpisodeRunner
from vla_eval.runners.sync_runner import SyncEpisodeRunner

from tests.conftest import StubBenchmark, start_server, stop_server
from tests.test_vector import SlotOneResultErrorStub, _config


# ---------------------------------------------------------------------------
# The harness ends every server episode with its status
# ---------------------------------------------------------------------------


class EndLogServer(PredictModelServer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.results: list[dict[str, Any]] = []

    def predict(self, obs, row):
        if obs.get("value", 0) < 0:
            raise ValueError("bad observation")
        return {"actions": np.ones(7, np.float32)}

    def finish_episodes(self, rows, results):
        self.results += [{k: r.get(k) for k in ("status", "failure_reason")} for r in results]


class NegativeBenchmark(StubBenchmark):
    def make_obs(self, raw_obs, task):
        return {"value": -1.0}


class BrokenStepBenchmark(StubBenchmark):
    def step(self, action):
        raise RuntimeError("physics exploded")


async def _ends_of(server: EndLogServer, port: int, n: int) -> list[dict[str, Any]]:
    with anyio.fail_after(3):
        while len(server.results) < n:
            await anyio.sleep(0.02)
    return server.results


@pytest.mark.anyio
async def test_sync_runner_ends_every_episode_with_its_status(free_port):
    server = EndLogServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            runner = SyncEpisodeRunner()
            await runner.run_episode(StubBenchmark(done_at_step=2), {"name": "t"}, conn, max_steps=5)
            await runner.run_episode(StubBenchmark(done_at_step=9), {"name": "t"}, conn, max_steps=2)
            with pytest.raises(Exception):
                await runner.run_episode(NegativeBenchmark(), {"name": "t"}, conn, max_steps=5)
            with pytest.raises(Exception):
                await runner.run_episode(BrokenStepBenchmark(), {"name": "t"}, conn, max_steps=5)
            assert await _ends_of(server, free_port, 4) == [
                {"status": "success", "failure_reason": None},
                {"status": "fail", "failure_reason": None},
                {"status": "error", "failure_reason": "model_act"},
                {"status": "error", "failure_reason": "env_step"},
            ]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_live_runner_ends_a_failed_episode_with_its_status(free_port):
    server = EndLogServer()
    task = await start_server(server, free_port)
    try:
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            with pytest.raises(Exception):
                await LiveEpisodeRunner(hz=100.0).run_episode(BrokenStepBenchmark(), {"name": "t"}, conn, max_steps=5)
            assert await _ends_of(server, free_port, 1) == [{"status": "error", "failure_reason": "env_step"}]
    finally:
        await stop_server(task)


@pytest.mark.anyio
async def test_vector_runner_ends_every_episode_with_its_status(free_port, tmp_path):
    server = EndLogServer()
    task = await start_server(server, free_port)
    try:
        config = _config(f"ws://127.0.0.1:{free_port}", tmp_path, episodes=3, num_tasks=1)
        with patch("vla_eval.orchestrator.resolve_import_string", return_value=SlotOneResultErrorStub):
            await Orchestrator(config, no_save=True).run()
        ends = await _ends_of(server, free_port, 3)
        assert sorted(ends, key=str) == [  # slot 1's result raised
            {"status": "error", "failure_reason": "env_step"},
            {"status": "success", "failure_reason": None},
            {"status": "success", "failure_reason": None},
        ]
    finally:
        await stop_server(task)


# ---------------------------------------------------------------------------
# Connection: replies of an earlier episode never reach the current one
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_connection_drops_replies_from_before_episode_start(free_port):
    """A raw server that answers each observation, and replays the last action after every EPISODE_START."""

    async def handler(ws: Any) -> None:
        last: Message | None = None
        async for raw in ws:
            msg = unpack_message(raw)
            if msg.type == MessageType.HELLO:
                await ws.send(
                    pack_message(Message(type=MessageType.HELLO, payload={"harness_version": "x"}, seq=msg.seq))
                )
            elif msg.type == MessageType.EPISODE_START and last is not None:
                await ws.send(pack_message(last))
            elif msg.type == MessageType.OBSERVATION:
                last = Message(type=MessageType.ACTION, payload={"actions": [msg.payload["value"]]}, seq=msg.seq)
                await ws.send(pack_message(last))

    async with ws_serve(handler, "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            await conn.start_episode({})
            assert (await conn.act({"value": 1.0}))["actions"] == [1.0]
            await conn.start_episode({})  # the server replays action 1.0
            assert (await conn.act({"value": 2.0}))["actions"] == [2.0]

            got: list[Any] = []
            conn.on_action(lambda a: got.append(a["actions"]))
            async with conn.listening():
                await conn.start_episode({})  # replays 2.0: dropped
                await conn.send_observation({"value": 3.0})
                with anyio.fail_after(2):
                    while not got:
                        await anyio.sleep(0.01)
            assert got == [[3.0]]


@pytest.mark.anyio
async def test_the_live_listener_drops_an_action_older_than_one_applied(free_port):
    """A raw server that answers two observations newest first."""

    async def handler(ws: Any) -> None:
        held: list[Message] = []
        async for raw in ws:
            msg = unpack_message(raw)
            if msg.type == MessageType.HELLO:
                await ws.send(
                    pack_message(Message(type=MessageType.HELLO, payload={"harness_version": "x"}, seq=msg.seq))
                )
            elif msg.type == MessageType.OBSERVATION:
                held.append(Message(type=MessageType.ACTION, payload={"actions": [msg.payload["value"]]}, seq=msg.seq))
                if len(held) == 2:
                    for m in reversed(held):
                        await ws.send(pack_message(m))

    async with ws_serve(handler, "127.0.0.1", free_port):
        async with Connection(f"ws://127.0.0.1:{free_port}") as conn:
            got: list[Any] = []
            conn.on_action(lambda a: got.append(a["actions"]))
            await conn.start_episode({})
            async with conn.listening():
                await conn.send_observation({"value": 1.0})
                await conn.send_observation({"value": 2.0})
                with anyio.fail_after(2):
                    while not got:
                        await anyio.sleep(0.01)
                await anyio.sleep(0.1)
            assert got == [[2.0]]
