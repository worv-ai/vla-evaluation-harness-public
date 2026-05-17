"""End-to-end tests for the unified recording API with strong (sid, step) ordering."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
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
    make_record_commit_payload,
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


def _ctx(sid: str, step: int = 0) -> SessionContext:
    c = SessionContext(session_id=sid, episode_id="ep")
    for _ in range(step):
        c._increment_step()
    return c


# ── step_scope lifecycle ────────────────────────────────────────────


def test_step_scope_commits_merged_fields_locally():
    """Without a sender, the commit lands in the local (sid, step) bucket."""
    r = _fresh_recording()
    ctx = _ctx("sess-local", step=3)
    with r.step_scope(ctx) as step:
        step.record(foo=1, bar="x")
        step.record(baz=2)
    assert r.drain_step("sess-local", 3) == {"foo": 1, "bar": "x", "baz": 2}
    # Second drain is empty (commit is removed).
    assert r.drain_step("sess-local", 3) == {}


def test_step_scope_routes_via_sender_as_one_commit():
    """Three record() calls in scope → ONE sender invocation."""
    r = _fresh_recording()
    sent: list[tuple[str, int, dict[str, Any]]] = []
    r.register_sender("sess-wire", lambda sid, step_id, fields: sent.append((sid, step_id, fields)))
    try:
        ctx = _ctx("sess-wire", step=5)
        with r.step_scope(ctx) as step:
            step.record(foo=1)
            step.record(bar=2)
            step.record(baz=3)
        assert sent == [("sess-wire", 5, {"foo": 1, "bar": 2, "baz": 3})]
    finally:
        r.unregister_sender("sess-wire")


def test_step_scope_commits_partial_fields_on_exception():
    """Records made before the exception still ship; the exception propagates."""
    r = _fresh_recording()
    ctx = _ctx("sess-exc", step=7)
    with pytest.raises(RuntimeError, match="boom"):
        with r.step_scope(ctx) as step:
            step.record(saved=True)
            raise RuntimeError("boom")
    assert r.drain_step("sess-exc", 7) == {"saved": True}


def test_step_scope_empty_does_not_emit_commit():
    """No record() calls inside scope → no sender invocation, no bucket entry."""
    r = _fresh_recording()
    sent: list[Any] = []
    r.register_sender("sess-empty", lambda *args: sent.append(args))
    try:
        ctx = _ctx("sess-empty", step=0)
        with r.step_scope(ctx):
            pass
        assert sent == []
    finally:
        r.unregister_sender("sess-empty")


def test_step_scope_double_entry_same_key_raises():
    r = _fresh_recording()
    ctx = _ctx("sess-dbl", step=2)
    with r.step_scope(ctx):
        with pytest.raises(RuntimeError, match="double-entry"):
            with r.step_scope(ctx):
                pass


def test_record_outside_step_scope_raises():
    r = _fresh_recording()
    ctx = _ctx("sess-out", step=0)
    with pytest.raises(RuntimeError, match="step_scope"):
        r.record(ctx, foo=1)


def test_record_module_helper_outside_step_scope_raises():
    """The module-level recording.record() free function also enforces the scope."""
    ctx = _ctx("sess-module-out", step=0)
    with pytest.raises(RuntimeError, match="step_scope"):
        recording.record(ctx, foo=1)


def test_record_inside_scope_with_mismatched_ctx_raises():
    r = _fresh_recording()
    ctx_outer = _ctx("sess-A", step=0)
    ctx_inner = _ctx("sess-B", step=0)
    with r.step_scope(ctx_outer):
        with pytest.raises(RuntimeError, match="does not match"):
            r.record(ctx_inner, foo=1)


def test_record_inside_scope_delegates_to_step():
    """recording.record(ctx, ...) inside step_scope appends to the active Step."""
    r = _fresh_recording()
    ctx = _ctx("sess-deleg", step=4)
    with r.step_scope(ctx):
        r.record(ctx, foo=1, bar=2)
    assert r.drain_step("sess-deleg", 4) == {"foo": 1, "bar": 2}


def test_step_scope_no_session_id_is_noop():
    r = _fresh_recording()

    class _Ctx:
        session_id = None
        step = 0

    with r.step_scope(_Ctx()) as step:
        step.record(foo=1)
    # Nothing buffered.
    assert r.drain_step("", 0) == {}


# ── ingest / drain semantics ─────────────────────────────────────────


def test_ingest_appends_to_per_step_bucket():
    r = _fresh_recording()
    r.ingest("sess-x", 1, {"a": 1})
    r.ingest("sess-x", 1, {"b": 2})
    r.ingest("sess-x", 2, {"c": 3})
    assert r.drain_step("sess-x", 1) == {"a": 1, "b": 2}
    assert r.drain_step("sess-x", 2) == {"c": 3}


def test_late_record_commit_lands_in_correct_bucket():
    """Out-of-order ingest: step 5 arrives before step 3; both stay separate."""
    r = _fresh_recording()
    r.ingest("sess-late", 5, {"late": True})
    r.ingest("sess-late", 3, {"early": True})
    assert r.drain_step("sess-late", 3) == {"early": True}
    assert r.drain_step("sess-late", 5) == {"late": True}


def test_drain_step_brief_wait_returns_when_ingest_lands():
    """drain_step with timeout waits for a late ingest in another thread."""
    r = _fresh_recording()

    def _delayed_ingest() -> None:
        time.sleep(0.05)
        r.ingest("sess-wait", 9, {"woke_up": True})

    threading.Thread(target=_delayed_ingest, daemon=True).start()
    t0 = time.monotonic()
    result = r.drain_step("sess-wait", 9, timeout=1.0)
    elapsed = time.monotonic() - t0
    assert result == {"woke_up": True}
    assert elapsed < 0.5  # woke before the full timeout


def test_drain_step_timeout_returns_empty():
    r = _fresh_recording()
    t0 = time.monotonic()
    assert r.drain_step("sess-gone", 0, timeout=0.05) == {}
    assert time.monotonic() - t0 >= 0.04


def test_session_scope_clears_stale_buckets_on_exit(caplog):
    r = _fresh_recording()
    r.ingest("sess-stale", 0, {"left_over": 1})
    with caplog.at_level("WARNING"):
        with r.session_scope("sess-stale"):
            r.ingest("sess-stale", 1, {"during": True})
    # Both stale and during are cleared.
    assert r.drain_step("sess-stale", 0) == {}
    assert r.drain_step("sess-stale", 1) == {}
    assert any("unmerged RECORD_COMMIT bucket" in rec.message for rec in caplog.records)


def test_session_scope_none_is_noop():
    r = _fresh_recording()
    with r.session_scope(None):
        assert r.current_session() is None


# ── singleton + protocol roundtrip ───────────────────────────────────


def test_get_default_returns_singleton():
    assert recording.get_default() is recording.get_default()


def test_record_commit_payload_roundtrip():
    payload = make_record_commit_payload("sess-1", 42, {"foo": 1, "bar": "x"})
    msg = Message(type=MessageType.RECORD_COMMIT, payload=payload, seq=7)
    restored = unpack_message(pack_message(msg))
    assert restored.type == MessageType.RECORD_COMMIT
    assert restored.payload["session_id"] == "sess-1"
    assert restored.payload["step_id"] == 42
    assert restored.payload["fields"] == {"foo": 1, "bar": "x"}


def test_record_commit_message_type_is_additive():
    assert MessageType.RECORD_COMMIT.value == "record_commit"
    assert {t.value for t in MessageType} >= {
        "hello",
        "observation",
        "action",
        "episode_start",
        "episode_end",
        "error",
    }


# ── EpisodeRecorder: bucket merge + episode-batched write ────────────


def test_episode_recorder_merges_per_step_buckets(tmp_path: Path):
    rec = EpisodeRecorder(tmp_path, record_video=False, record_step=True)
    rec.start({"env_id": "Test", "episode_idx": 0})

    default = recording.get_default()
    with default.session_scope("sess-merge"):
        default.ingest("sess-merge", 0, {"s2_subgoal": "pick", "s2_fired": True})
        default.ingest("sess-merge", 1, {"chunk_idx": 3})
        rec.record_step({"step": 0, "reward": 1.0})
        rec.record_step({"step": 1, "reward": 0.5})

    # Before save(), nothing on disk.
    assert not list(tmp_path.glob("*.jsonl"))
    rec.save(status="success")
    jsonl_path = tmp_path / "Test_ep0000_success.jsonl"
    assert jsonl_path.exists()
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert rows[0] == {"step": 0, "reward": 1.0, "s2_subgoal": "pick", "s2_fired": True}
    assert rows[1] == {"step": 1, "reward": 0.5, "chunk_idx": 3}


def test_episode_recorder_late_arrival_into_correct_step_bucket(tmp_path: Path):
    """RECORD_COMMIT for step 0 that arrives BEFORE record_step(step=0) is captured."""
    rec = EpisodeRecorder(tmp_path, record_video=False, record_step=True)
    rec.start({"env_id": "Test", "episode_idx": 0})

    default = recording.get_default()
    with default.session_scope("sess-late"):
        # Server-side commit for step 0 arrives.
        default.ingest("sess-late", 0, {"server_foo": 1})
        # Server-side commit for step 1 also arrives (early).
        default.ingest("sess-late", 1, {"server_bar": 2})
        rec.record_step({"step": 0, "env_done": False})
        rec.record_step({"step": 1, "env_done": True})

    rec.save(status="success")
    rows = [json.loads(line) for line in (tmp_path / "Test_ep0000_success.jsonl").read_text().splitlines()]
    assert rows[0]["server_foo"] == 1
    assert rows[1]["server_bar"] == 2


def test_episode_recorder_missing_commit_emits_bench_only_after_brief_wait(tmp_path: Path):
    rec = EpisodeRecorder(tmp_path, record_video=False, record_step=True)
    rec.start({"env_id": "Test", "episode_idx": 0})

    default = recording.get_default()
    with default.session_scope("sess-miss"):
        # No ingest for step 0. record_step waits briefly, then proceeds without extras.
        t0 = time.monotonic()
        rec.record_step({"step": 0, "reward": 1.0})
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.05  # waited at least the brief-await
    rec.save(status="success")

    rows = [json.loads(line) for line in (tmp_path / "Test_ep0000_success.jsonl").read_text().splitlines()]
    assert rows == [{"step": 0, "reward": 1.0}]


def test_record_step_collision_logs_and_data_wins(tmp_path: Path, caplog):
    """Benchmark-recorded fields are authoritative on collision."""
    rec = EpisodeRecorder(tmp_path, record_video=False, record_step=True)
    rec.start({"env_id": "Test", "episode_idx": 0})

    default = recording.get_default()
    with default.session_scope("sess-collide"):
        default.ingest("sess-collide", 1, {"step": 999, "reward": 9.9, "extra": "ok"})
        with caplog.at_level("WARNING"):
            rec.record_step({"step": 1, "reward": 0.5})
    rec.save(status="success")

    jsonl_path = tmp_path / "Test_ep0000_success.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert rows == [{"step": 1, "reward": 0.5, "extra": "ok"}]
    msgs = [rec.message for rec in caplog.records]
    assert any("collide with benchmark schema" in m and "'reward'" in m and "'step'" in m for m in msgs)


def test_record_step_requires_integer_step_key(tmp_path: Path):
    rec = EpisodeRecorder(tmp_path, record_video=False, record_step=True)
    rec.start({"env_id": "Test", "episode_idx": 0})
    with pytest.raises(ValueError, match="integer 'step' key"):
        rec.record_step({"reward": 1.0})  # missing 'step'


def test_episode_recorder_discard_does_not_write(tmp_path: Path):
    rec = EpisodeRecorder(tmp_path, record_video=False, record_step=True)
    rec.start({"env_id": "Test", "episode_idx": 0})
    rec.record_step({"step": 0, "reward": 1.0})
    rec.discard()
    assert not list(tmp_path.glob("*.jsonl"))


# ── multi-session interleaved ───────────────────────────────────────


def test_multi_session_per_step_isolation():
    r = _fresh_recording()
    r.ingest("sA", 0, {"side": "A0"})
    r.ingest("sB", 0, {"side": "B0"})
    r.ingest("sA", 1, {"side": "A1"})
    assert r.drain_step("sA", 0) == {"side": "A0"}
    assert r.drain_step("sB", 0) == {"side": "B0"}
    assert r.drain_step("sA", 1) == {"side": "A1"}


# ── end-to-end with model server + harness ───────────────────────────


class _RecordingServer(PredictModelServer):
    def predict(self, obs: dict[str, Any], ctx: SessionContext) -> dict[str, Any]:
        with recording.step_scope(ctx) as step:
            step.record(server_foo=1, server_bar="x", server_step=ctx.step)
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
        # In production the harness-side benchmark calls recording.record(...)
        # via its own ctx; here we share one process with the server so we
        # ingest directly to avoid being looped back through the in-process
        # wire sender bound by the server.
        default = recording.get_default()
        sid = default.current_session()
        step_id = self._step_count
        if sid is not None:
            default.ingest(sid, step_id, {"bench_reward": 0.5, "bench_step": step_id})
        self._step_count += 1
        done = self._step_count >= self.done_at_step
        self._recorder.record_step({"step": step_id, "env_done": done})
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
    for i, row in enumerate(rows):
        # Benchmark-side fields
        assert row["step"] == i
        assert "env_done" in row
        assert row["bench_reward"] == 0.5
        assert row["bench_step"] == i
        # Server-side fields shipped over the wire as RECORD_COMMIT messages,
        # keyed by (session_id, step_id) — must align step-for-step.
        assert row["server_foo"] == 1
        assert row["server_bar"] == "x"
        assert row["server_step"] == i


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
