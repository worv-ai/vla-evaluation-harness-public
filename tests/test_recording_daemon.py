"""Unit tests for the recording daemon (in-process server + emitter).

The daemon runs in the test's asyncio loop while emitters run in their
own background thread. We use ``asyncio.sleep`` (not ``time.sleep``) so
the daemon's loop keeps processing frames between assertions.

The wire protocol is intentionally small (STEP / RESULT / EVAL_END) and
buckets/evals are created implicitly on first arrival — these tests
exercise that contract.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import pytest
import websockets

from vla_eval.recording_daemon.client import RecordingClient
from vla_eval.recording_daemon.config import RecordingDaemonConfig
from vla_eval.recording_daemon.daemon import RecordingDaemon
from vla_eval.recording_daemon.messages import (
    RecMessageType,
    RecordingFrame,
    pack_frame,
    unpack_frame,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wait_bind(url: str, deadline: float = 5.0) -> None:
    """Poll until the daemon's WS server is accepting connections."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            async with websockets.connect(url):
                return
        except OSError:
            await asyncio.sleep(0.02)
    raise TimeoutError(f"daemon never bound on {url}")


async def _wait_path(path: Path, deadline: float = 3.0) -> bool:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if path.exists():
            return True
        await asyncio.sleep(0.05)
    return False


def _push_result(
    client: RecordingClient,
    *,
    eval_id: str,
    sid: str,
    eid: str,
    status: str,
    metrics: dict,
    jsonl_path: Path,
    aggregate_dir: Path,
    aggregate_filename: str,
    task_name: str = "default",
    episode_id: int = 0,
    context: dict | None = None,
    bench_metadata: dict | None = None,
) -> None:
    """Helper that fills in the now-mandatory RESULT fields with sensible defaults."""
    client.result(
        eval_id=eval_id,
        sid=sid,
        eid=eid,
        task_name=task_name,
        episode_id=episode_id,
        status=status,
        metrics=metrics,
        steps=len(metrics) * 0,  # exact count doesn't matter for these tests
        elapsed_sec=0.0,
        context=context or {},
        jsonl_path=str(jsonl_path),
        aggregate_dir=str(aggregate_dir),
        aggregate_filename=aggregate_filename,
        bench_metadata=bench_metadata
        or {
            "benchmark": "demo",
            "mode": "sync",
            "metric_keys": {"success": "mean"},
            "config": {},
            "harness_version": "test",
        },
    )


async def _start_daemon(
    port: int, out_dir: Path, idle_timeout: float = 600.0
) -> tuple[RecordingDaemon, str, asyncio.Task]:
    cfg = RecordingDaemonConfig(
        bind=f"ws://127.0.0.1:{port}",
        out_dir=str(out_dir),
        idle_timeout_sec=idle_timeout,
    )
    d = RecordingDaemon(cfg)
    task = asyncio.create_task(d.start())
    await _wait_bind(cfg.bind)
    return d, cfg.bind, task


async def _stop_daemon(d: RecordingDaemon, task: asyncio.Task) -> None:
    await d.stop()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except (asyncio.TimeoutError, Exception):
        pass


@pytest.fixture
async def daemon(free_port: int, tmp_path: Path) -> AsyncIterator[tuple[RecordingDaemon, str, Path]]:
    d, url, task = await _start_daemon(free_port, tmp_path)
    try:
        yield d, url, tmp_path
    finally:
        await _stop_daemon(d, task)


# ---------------------------------------------------------------------------
# Envelope round-trip
# ---------------------------------------------------------------------------


def test_envelope_roundtrip_all_types() -> None:
    for mtype in RecMessageType:
        payload = {"k": "v", "arr": np.arange(6, dtype=np.float32).reshape(2, 3)}
        frame = RecordingFrame(type=mtype, payload=payload)
        out = unpack_frame(pack_frame(frame))
        assert out.type == mtype
        assert out.payload["k"] == "v"
        np.testing.assert_array_equal(out.payload["arr"], payload["arr"])


def test_unpack_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        unpack_frame(b"\x00\x01garbage")
    import msgpack

    not_a_dict = msgpack.packb([1, 2, 3])
    with pytest.raises(ValueError):
        unpack_frame(not_a_dict)


# ---------------------------------------------------------------------------
# Happy path: STEP × N → RESULT writes jsonl
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_implicit_bucket_creation_and_result_flush(
    daemon: tuple[RecordingDaemon, str, Path],
) -> None:
    d, url, tmp = daemon
    client = RecordingClient(url)
    sid, eid = "shard-0", "ep-0"
    jsonl_path = tmp / "ep0.jsonl"
    try:
        client.step(sid, eid, 0, {"reward": 0.1})
        client.step(sid, eid, 1, {"reward": 0.2})
        client.step(sid, eid, 2, {"reward": 0.3})
        _push_result(
            client,
            eval_id="eval-x",
            sid=sid,
            eid=eid,
            status="success",
            metrics={"success": True},
            context={"env_id": "demo", "episode_idx": 0},
            jsonl_path=jsonl_path,
            aggregate_dir=tmp,
            aggregate_filename="agg.json",
        )
        # Wait for the daemon to flush.
        assert await _wait_path(jsonl_path), f"jsonl never appeared at {jsonl_path}"
    finally:
        client.close()

    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert rows[0]["reward"] == pytest.approx(0.1)
    assert d.bucket_count == 0  # bucket popped on RESULT


# ---------------------------------------------------------------------------
# Aggregate JSON across multiple episodes in the same eval
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_eval_end_flushes_aggregate(
    daemon: tuple[RecordingDaemon, str, Path],
) -> None:
    """EVAL_END produces a BenchmarkResult-shaped JSON with metadata,
    per-task aggregation, and global mean_success."""
    d, url, tmp = daemon
    agg_path = tmp / "agg.json"
    client = RecordingClient(url)
    bench_metadata = {
        "benchmark": "demo",
        "mode": "sync",
        "metric_keys": {"success": "mean"},
        "config": {"params": {"seed": 7}},
        "harness_version": "test-version",
        "server_info": {"model_server": "EchoServer"},
    }
    try:
        for i in range(3):
            sid, eid = "shard-0", f"ep-{i}"
            client.step(sid, eid, 0, {"reward": float(i)})
            _push_result(
                client,
                eval_id="eval-x",
                sid=sid,
                eid=eid,
                status="success" if i % 2 == 0 else "fail",
                metrics={"success": i % 2 == 0},
                context={"env_id": "demo", "episode_idx": i},
                jsonl_path=tmp / f"ep{i}.jsonl",
                aggregate_dir=tmp,
                aggregate_filename="agg.json",
                task_name="demo_task",
                episode_id=i,
                bench_metadata=bench_metadata,
            )
        client.eval_end("eval-x")
        assert await _wait_path(agg_path), f"aggregate never appeared at {agg_path}"
    finally:
        client.close()

    body = json.loads(agg_path.read_text())
    # Metadata round-trip.
    assert body["benchmark"] == "demo"
    assert body["mode"] == "sync"
    assert body["harness_version"] == "test-version"
    assert body["server_info"] == {"model_server": "EchoServer"}
    assert body["seed"] == 7
    assert body["metric_keys"] == {"success": "mean"}
    assert body["eval_id"] == "eval-x"
    assert "unclosed" not in body  # happy path

    # Per-task structure with aggregation.
    assert [t["task"] for t in body["tasks"]] == ["demo_task"]
    task = body["tasks"][0]
    assert task["num_episodes"] == 3
    # 2 successes out of 3.
    assert task["mean_success"] == pytest.approx(2 / 3, abs=1e-4)
    assert body["mean_success"] == pytest.approx(2 / 3, abs=1e-4)
    assert {ep["eid"] for ep in task["episodes"]} == {"ep-0", "ep-1", "ep-2"}
    assert d.eval_count == 0


# ---------------------------------------------------------------------------
# Two emitters fan in to the same bucket
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_two_emitters_merge_into_same_bucket(
    daemon: tuple[RecordingDaemon, str, Path],
) -> None:
    """Two RecordingClients (separate WS connections) → same bucket.

    Ordering between connections is not guaranteed, so we drain the
    secondary emitter (close → joins its background thread, ensuring the
    daemon has received everything it queued) BEFORE the orchestrator
    declares the episode done with ``result()``. Any STEP arriving after
    RESULT is dropped by contract — see :func:`test_late_step_after_result_dropped`.
    """
    _, url, tmp = daemon
    sid, eid = "shard-A", "ep-merged"
    jsonl_path = tmp / "ep-merged.jsonl"

    e_orch = RecordingClient(url)
    e_model = RecordingClient(url)
    try:
        # Orchestrator-side step rows
        e_orch.step(sid, eid, 0, {"orch": "first"})
        # Model-server-side step rows for the same (sid, eid)
        e_model.step(sid, eid, 0, {"model": "stamp"})  # merges into step 0
        e_model.step(sid, eid, 1, {"model": "alpha"})
        # Drain the model emitter before declaring the episode done — otherwise
        # a STEP can race past RESULT and land in a fresh (orphan) bucket.
        e_model.close()
        # Small yield so the daemon's loop drains the connection's last frames.
        await asyncio.sleep(0.1)
        # Orchestrator declares the episode done
        _push_result(
            e_orch,
            eval_id="eval-merge",
            sid=sid,
            eid=eid,
            status="success",
            metrics={"success": True},
            context={"env_id": "merge"},
            jsonl_path=jsonl_path,
            aggregate_dir=tmp,
            aggregate_filename="agg.json",
        )
        assert await _wait_path(jsonl_path)
    finally:
        e_orch.close()

    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    # Step 0 merged: holds both orch and model fields
    by_step = {r["step"]: r for r in rows}
    assert by_step[0] == {"step": 0, "orch": "first", "model": "stamp"}
    assert by_step[1] == {"step": 1, "model": "alpha"}


# ---------------------------------------------------------------------------
# Late STEP after RESULT is dropped (bucket already popped)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_late_step_after_result_dropped(
    daemon: tuple[RecordingDaemon, str, Path],
) -> None:
    _, url, tmp = daemon
    sid, eid = "shard-0", "late"
    jsonl_path = tmp / "late.jsonl"
    client = RecordingClient(url)
    try:
        client.step(sid, eid, 0, {"a": 1})
        _push_result(
            client,
            eval_id="e",
            sid=sid,
            eid=eid,
            status="success",
            metrics={},
            jsonl_path=jsonl_path,
            aggregate_dir=tmp,
            aggregate_filename="agg.json",
        )
        assert await _wait_path(jsonl_path)
        # This STEP arrives after RESULT — should land in a new bucket which never
        # closes, so the file content for the original episode is unaffected.
        client.step(sid, eid, 99, {"a": "late"})
        await asyncio.sleep(0.2)
    finally:
        client.close()

    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert rows == [{"step": 0, "a": 1}]


# ---------------------------------------------------------------------------
# Age-out: bucket with steps but no RESULT → _unclosed.jsonl
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_bucket_age_out_to_unclosed(free_port: int, tmp_path: Path) -> None:
    d, url, task = await _start_daemon(free_port, tmp_path, idle_timeout=0.05)
    try:
        client = RecordingClient(url)
        client.step("s", "e", 0, {"v": 1})
        # Wait long enough for the age-out loop to act. The loop runs every
        # _AGE_OUT_INTERVAL_SEC = 15s in production, but for tests we just trigger
        # it manually to keep the suite fast.
        await asyncio.sleep(0.1)
        await d._age_out_once()
        client.close()
        path = tmp_path / "s-e_unclosed.jsonl"
        assert await _wait_path(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows == [{"step": 0, "v": 1}]
    finally:
        await _stop_daemon(d, task)


@pytest.mark.anyio
async def test_eval_age_out_to_unclosed_aggregate(free_port: int, tmp_path: Path) -> None:
    d, url, task = await _start_daemon(free_port, tmp_path, idle_timeout=0.05)
    try:
        client = RecordingClient(url)
        # One RESULT opens the eval but no EVAL_END follows.
        _push_result(
            client,
            eval_id="ev-age",
            sid="s",
            eid="e",
            status="success",
            metrics={},
            jsonl_path=tmp_path / "e.jsonl",
            aggregate_dir=tmp_path,
            aggregate_filename="age.json",
        )
        await asyncio.sleep(0.1)
        await d._age_out_once()
        client.close()
        agg = tmp_path / "age_unclosed.json"
        assert await _wait_path(agg)
        body = json.loads(agg.read_text())
        assert body["unclosed"] is True
        # One episode landed under the default task name from _push_result.
        assert sum(len(t["episodes"]) for t in body["tasks"]) == 1
    finally:
        await _stop_daemon(d, task)


# ---------------------------------------------------------------------------
# Emitter resilience: daemon-down does not raise; frames queue + reconnect
# ---------------------------------------------------------------------------


def test_emitter_with_no_daemon_drops_silently(free_port: int) -> None:
    """Sending to a dead daemon must never raise — recording is best-effort."""
    client = RecordingClient(f"ws://127.0.0.1:{free_port}")
    try:
        client.step("s", "e", 0, {"x": 1})
        _push_result(
            client,
            eval_id="e",
            sid="s",
            eid="e",
            status="success",
            metrics={},
            jsonl_path=Path("/tmp/x.jsonl"),
            aggregate_dir=Path("/tmp"),
            aggregate_filename="a.json",
        )
        client.eval_end("e")
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Shutdown flush
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_shutdown_flushes_in_flight_buckets(free_port: int, tmp_path: Path) -> None:
    d, url, task = await _start_daemon(free_port, tmp_path)
    client = RecordingClient(url)
    try:
        client.step("s", "x", 0, {"a": 1})
        client.step("s", "x", 1, {"a": 2})
        await asyncio.sleep(0.2)
    finally:
        client.close()
    await _stop_daemon(d, task)
    # Daemon shutdown should have flushed the bucket with _unclosed.
    path = tmp_path / "s-x_unclosed.jsonl"
    assert path.exists(), f"shutdown did not flush bucket; expected {path}"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [{"step": 0, "a": 1}, {"step": 1, "a": 2}]
