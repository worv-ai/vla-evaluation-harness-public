"""End-to-end tests for the unified recording API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest

from vla_eval import recording
from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.benchmarks.data_recording import EpisodeRecorder
from vla_eval.connection import Connection
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.model_servers.serve import serve_async
from vla_eval.protocol.messages import (
    Message,
    MessageType,
    make_record_payload,
    pack_message,
    unpack_message,
)
from vla_eval.runners.sync_runner import SyncEpisodeRunner


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fresh_recording() -> recording.Recording:
    """A standalone Recording instance for tests that need isolation from the default."""
    return recording.Recording()


def test_record_uses_local_buffer_without_sender():
    r = _fresh_recording()
    ctx = SessionContext(session_id="sess-local", episode_id="ep")
    r.record(ctx, foo=1, bar="x")
    assert r.drain("sess-local") == {"foo": 1, "bar": "x"}
    assert r.drain("sess-local") == {}


def test_record_routes_via_sender_when_registered():
    r = _fresh_recording()
    sent: list[tuple[str, dict[str, Any]]] = []
    r.register_sender("sess-wire", lambda sid, fields: sent.append((sid, fields)))
    try:
        ctx = SessionContext(session_id="sess-wire", episode_id="ep")
        r.record(ctx, foo=1)
        assert sent == [("sess-wire", {"foo": 1})]
        assert r.drain("sess-wire") == {}
    finally:
        r.unregister_sender("sess-wire")


def test_record_warns_when_server_mode_missing_sender(caplog):
    """Server-side bug: a sender was registered for some session, but not this one."""
    r = _fresh_recording()
    r.register_sender("other", lambda sid, fields: None)

    ctx = SessionContext(session_id="sess-missing", episode_id="ep")
    with caplog.at_level("WARNING"):
        r.record(ctx, foo=1)
    # Fields are dropped (NOT buffered) in server mode when sender missing.
    assert r.drain("sess-missing") == {}
    assert any("no sender registered" in rec.message for rec in caplog.records)


def test_ingest_appends_to_buffer():
    r = _fresh_recording()
    r.ingest("sess-x", {"a": 1})
    r.ingest("sess-x", {"b": 2})
    assert r.drain("sess-x") == {"a": 1, "b": 2}


def test_record_drops_when_session_id_missing(caplog):
    r = _fresh_recording()

    class _Ctx:
        session_id = None

    with caplog.at_level("WARNING"):
        r.record(_Ctx(), foo=1)
    assert r.drain("") == {}
    assert any("no session_id" in rec.message for rec in caplog.records)


def test_session_scope_clears_on_entry_and_exit():
    r = _fresh_recording()
    r.ingest("sess-scope", {"stale": True})
    with r.session_scope("sess-scope"):
        assert r.current_session() == "sess-scope"
        # Entry cleared the stale data.
        assert r.drain_current() == {}
        r.ingest("sess-scope", {"fresh": 1})
        assert r.drain_current() == {"fresh": 1}
    # Exit restored ContextVar and cleared the buffer.
    assert r.current_session() is None
    assert r.drain("sess-scope") == {}


def test_session_scope_none_is_noop():
    r = _fresh_recording()
    with r.session_scope(None):
        assert r.current_session() is None


def test_get_default_returns_singleton():
    assert recording.get_default() is recording.get_default()


def test_record_payload_roundtrip():
    payload = make_record_payload("sess-1", {"foo": 1, "bar": "x"})
    msg = Message(type=MessageType.RECORD, payload=payload, seq=7)
    restored = unpack_message(pack_message(msg))
    assert restored.type == MessageType.RECORD
    assert restored.payload["session_id"] == "sess-1"
    assert restored.payload["fields"] == {"foo": 1, "bar": "x"}


def test_record_message_type_is_additive():
    assert MessageType.RECORD.value == "record"
    assert {t.value for t in MessageType} >= {
        "hello",
        "observation",
        "action",
        "episode_start",
        "episode_end",
        "error",
    }


def test_episode_recorder_merges_drained_fields(tmp_path: Path):
    rec = EpisodeRecorder(tmp_path, record_video=False, record_step=True)
    rec.start({"env_id": "Test", "episode_idx": 0})

    default = recording.get_default()
    with default.session_scope("sess-merge"):
        default.ingest("sess-merge", {"s2_subgoal": "pick", "s2_fired": True})
        rec.record_step({"step": 0, "reward": 1.0})
        default.ingest("sess-merge", {"chunk_idx": 3})
        rec.record_step({"step": 1, "reward": 0.5})

    rec.save(status="success")
    jsonl_path = tmp_path / "Test_ep0000_success.jsonl"
    assert jsonl_path.exists()
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert rows[0] == {"step": 0, "reward": 1.0, "s2_subgoal": "pick", "s2_fired": True}
    assert rows[1] == {"step": 1, "reward": 0.5, "chunk_idx": 3}


def test_record_step_collision_logs_and_data_wins(tmp_path: Path, caplog):
    """Benchmark-recorded fields are authoritative on collision."""
    rec = EpisodeRecorder(tmp_path, record_video=False, record_step=True)
    rec.start({"env_id": "Test", "episode_idx": 0})

    default = recording.get_default()
    with default.session_scope("sess-collide"):
        default.ingest("sess-collide", {"step": 999, "reward": 9.9, "extra": "ok"})
        with caplog.at_level("WARNING"):
            rec.record_step({"step": 1, "reward": 0.5})
    rec.save(status="success")

    jsonl_path = tmp_path / "Test_ep0000_success.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert rows == [{"step": 1, "reward": 0.5, "extra": "ok"}]
    msgs = [rec.message for rec in caplog.records]
    assert any("collide with benchmark schema" in m and "'reward'" in m and "'step'" in m for m in msgs)


class _RecordingServer(PredictModelServer):
    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        recording.record(ctx, server_foo=1, server_bar="x", server_step=ctx.step)
        return {"actions": np.zeros(7, dtype=np.float32)}


class _RecordingBenchmark(StepBenchmark):
    """Benchmark that calls recording.record() during step() and writes jsonl rows."""

    def __init__(self, output_dir: Path, done_at_step: int = 3, **kwargs: Any) -> None:
        super().__init__()
        self.done_at_step = done_at_step
        self._step_count = 0
        self._recorder = EpisodeRecorder(output_dir, record_video=False, record_step=True)

    def get_tasks(self) -> list[dict[str, Any]]:
        return [{"name": "task0", "env_id": "Test", "episode_idx": 0}]

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": 50}

    def reset(self, task: dict[str, Any]) -> Any:
        self._step_count = 0
        self._recorder.start({"env_id": task["env_id"], "episode_idx": task["episode_idx"]})
        return {"value": 1.0}

    def step(self, action: dict[str, Any]) -> StepResult:
        # In real deployments the server and harness live in separate
        # processes; here they share one, so the harness-side benchmark
        # uses ``ingest`` directly to avoid double-routing through the
        # wire sender bound by the in-process server.
        default = recording.get_default()
        sid = default.current_session()
        if sid is not None:
            default.ingest(sid, {"bench_reward": 0.5, "bench_step": self._step_count})
        self._step_count += 1
        done = self._step_count >= self.done_at_step
        self._recorder.record_step({"step": self._step_count - 1, "env_done": done})
        return StepResult(obs={"value": 1.0}, reward=1.0 if done else 0.0, done=done, info={})

    def make_obs(self, raw_obs: Any, task: dict[str, Any]) -> dict[str, Any]:
        return {"value": 1.0, "task_description": task.get("name", "")}

    def get_step_result(self, step_result: StepResult) -> dict[str, Any]:
        self._recorder.save(status="success" if step_result.done else "fail")
        return {"success": step_result.done}


async def _wait_for_server(port: int, timeout: float = 5.0) -> None:
    import time as _time
    import websockets as _ws

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        try:
            async with _ws.connect(f"ws://127.0.0.1:{port}"):
                return
        except (OSError, _ws.exceptions.InvalidHandshake):
            await anyio.sleep(0.05)
    raise TimeoutError(f"Server on port {port} did not start within {timeout}s")


@pytest.mark.anyio
async def test_end_to_end_recording_roundtrip(tmp_path: Path):
    port = _free_port()
    server_task = asyncio.create_task(serve_async(_RecordingServer(), port=port))
    await _wait_for_server(port)

    try:
        conn = Connection(f"ws://127.0.0.1:{port}")
        await conn.connect()
        assert conn.session_id is not None

        benchmark = _RecordingBenchmark(tmp_path, done_at_step=3)
        runner = SyncEpisodeRunner()
        result = await runner.run_episode(benchmark, benchmark.get_tasks()[0], conn, max_steps=10)
        assert result["metrics"]["success"] is True

        await conn.close()
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, anyio.get_cancelled_exc_class()):
            await server_task

    jsonl_path = tmp_path / "Test_ep0000_success.jsonl"
    assert jsonl_path.exists(), f"missing {jsonl_path}"
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(rows) == 3
    for row in rows:
        # Benchmark-side fields
        assert "step" in row and "env_done" in row
        assert "bench_reward" in row and row["bench_reward"] == 0.5
        # Server-side fields shipped over the wire as RECORD messages.
        assert row["server_foo"] == 1
        assert row["server_bar"] == "x"
        assert "server_step" in row


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
