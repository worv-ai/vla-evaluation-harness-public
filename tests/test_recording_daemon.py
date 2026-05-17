"""Unit tests for the recording daemon (in-process server + emitter).

The daemon runs in the test's asyncio loop while emitters run in their own
background thread.  We use ``asyncio.sleep`` (not ``time.sleep``) so the
daemon's loop keeps processing frames between assertions.
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

from vla_eval.recording_daemon.config import RecordingDaemonConfig
from vla_eval.recording_daemon.daemon import RecordingDaemon
from vla_eval.recording_daemon.client import RecordingClient
from vla_eval.recording_daemon.messages import (
    RecordingFrame,
    RecordingMessageType,
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


async def _start_daemon(
    port: int, out_dir: Path, idle_timeout: float = 600.0
) -> tuple[RecordingDaemon, str, asyncio.Task]:
    cfg = RecordingDaemonConfig(
        bind=f"ws://127.0.0.1:{port}",
        out_dir=str(out_dir),
        idle_timeout_sec=idle_timeout,
        health_port=0,
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
    for mtype in RecordingMessageType:
        payload = {"k": "v", "arr": np.arange(6, dtype=np.float32).reshape(2, 3)}
        frame = RecordingFrame(type=mtype, payload=payload)
        out = unpack_frame(pack_frame(frame))
        assert out.type == mtype
        assert out.payload["k"] == "v"
        np.testing.assert_array_equal(out.payload["arr"], payload["arr"])


def test_unpack_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        unpack_frame(b"not-msgpack")


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_episode_lifecycle(daemon: tuple) -> None:
    d, url, out_dir = daemon
    emitter = RecordingClient(url)
    try:
        emitter.start_eval("e1", str(out_dir), "agg.json", expected_count=1)
        emitter.start_episode("e1", "s1", "ep1", str(out_dir), "{task}_ep{idx:04d}_{status}", {"task": "T", "idx": 0})
        for step in range(3):
            emitter.record("s1", "ep1", step, {"reward": float(step)})
        working = out_dir / ".recorder-fake.mp4"
        working.write_bytes(b"\x00\x00\x00\x18ftypmp42")
        emitter.video("s1", "ep1", str(working))
        emitter.result("e1", "s1", "ep1", "success", {"success": True})
        emitter.end_episode("s1", "ep1")

        jsonl = out_dir / "T_ep0000_success.jsonl"
        mp4 = out_dir / "T_ep0000_success.mp4"
        assert await _wait_path(jsonl), f"missing jsonl in {list(out_dir.iterdir())}"
        assert await _wait_path(mp4), f"missing mp4 in {list(out_dir.iterdir())}"
        lines = jsonl.read_text().strip().splitlines()
        assert len(lines) == 3
        assert [json.loads(line)["step"] for line in lines] == [0, 1, 2]
    finally:
        emitter.close()


@pytest.mark.anyio
async def test_two_emitters_merge(daemon: tuple) -> None:
    d, url, out_dir = daemon
    a = RecordingClient(url)
    b = RecordingClient(url)
    try:
        a.start_episode("e1", "s1", "ep1", str(out_dir), "merge_{idx:04d}_{status}", {"idx": 0})
        b.start_episode("e1", "s1", "ep1", str(out_dir), "merge_{idx:04d}_{status}", {"idx": 0})
        a.record("s1", "ep1", 0, {"from_a": 1})
        b.record("s1", "ep1", 0, {"from_b": 2})
        a.record("s1", "ep1", 1, {"from_a": 1})
        a.result("e1", "s1", "ep1", "fail", {"success": False})
        a.end_episode("s1", "ep1")

        jsonl = out_dir / "merge_0000_fail.jsonl"
        assert await _wait_path(jsonl), list(out_dir.iterdir())
        rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert rows[0]["from_a"] == 1
        assert rows[0]["from_b"] == 2
        assert rows[1]["from_a"] == 1
    finally:
        a.close()
        b.close()


@pytest.mark.anyio
async def test_late_frame_dropped(daemon: tuple) -> None:
    d, url, out_dir = daemon
    emitter = RecordingClient(url)
    try:
        emitter.start_episode("e1", "s1", "ep1", str(out_dir), "late_{idx:04d}_{status}", {"idx": 0})
        emitter.record("s1", "ep1", 0, {"x": 1})
        emitter.result("e1", "s1", "ep1", "success", {"success": True})
        emitter.end_episode("s1", "ep1")
        jsonl = out_dir / "late_0000_success.jsonl"
        assert await _wait_path(jsonl)
        emitter.record("s1", "ep1", 1, {"x": 2})  # bucket already gone
        await asyncio.sleep(0.3)
        rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert [r["step"] for r in rows] == [0]
    finally:
        emitter.close()


@pytest.mark.anyio
async def test_unknown_episode_id_dropped(daemon: tuple) -> None:
    d, url, out_dir = daemon
    emitter = RecordingClient(url)
    try:
        emitter.record("nope", "nope", 0, {"x": 1})
        await asyncio.sleep(0.3)
        assert list(out_dir.glob("*.jsonl")) == []
    finally:
        emitter.close()


@pytest.mark.anyio
async def test_video_rename_failure(daemon: tuple) -> None:
    d, url, out_dir = daemon
    emitter = RecordingClient(url)
    try:
        emitter.start_episode("e1", "s1", "ep1", str(out_dir), "vid_{idx:04d}_{status}", {"idx": 0})
        emitter.record("s1", "ep1", 0, {"x": 1})
        emitter.video("s1", "ep1", str(out_dir / "does-not-exist.mp4"))
        emitter.result("e1", "s1", "ep1", "success", {"success": True})
        emitter.end_episode("s1", "ep1")
        jsonl = out_dir / "vid_0000_success.jsonl"
        assert await _wait_path(jsonl)
        assert not (out_dir / "vid_0000_success.mp4").exists()
    finally:
        emitter.close()


@pytest.mark.anyio
async def test_eval_aggregate(daemon: tuple) -> None:
    d, url, out_dir = daemon
    emitter = RecordingClient(url)
    try:
        emitter.start_eval("agg1", str(out_dir), "myrun.json", expected_count=2)
        for i in range(2):
            emitter.start_episode("agg1", "s1", f"ep{i}", str(out_dir), "agg_{idx:04d}_{status}", {"idx": i})
            emitter.result("agg1", "s1", f"ep{i}", "success", {"success": True, "idx": i})
            emitter.end_episode("s1", f"ep{i}")
        emitter.end_eval("agg1")

        aggregate = out_dir / "myrun.json"
        assert await _wait_path(aggregate, deadline=4.0)
        data = json.loads(aggregate.read_text())
        assert data["expected_count"] == 2
        assert data["received_count"] == 2
        assert {r["eid"] for r in data["results"]} == {"ep0", "ep1"}
    finally:
        emitter.close()


# ---------------------------------------------------------------------------
# Age-out
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_age_out_partial_flush_bucket(free_port: int, tmp_path: Path) -> None:
    d, url, task = await _start_daemon(free_port, tmp_path, idle_timeout=0.1)
    emitter = RecordingClient(url)
    try:
        emitter.start_episode("e1", "s1", "ep1", str(tmp_path), "age_{idx:04d}_{status}", {"idx": 0})
        emitter.record("s1", "ep1", 0, {"x": 1})
        await asyncio.sleep(0.5)
        await d._age_out_once()
        files = sorted(tmp_path.glob("age_*"))
        assert any("_unclosed" in str(p) for p in files), files
    finally:
        emitter.close()
        await _stop_daemon(d, task)


@pytest.mark.anyio
async def test_age_out_partial_flush_eval(free_port: int, tmp_path: Path) -> None:
    d, url, task = await _start_daemon(free_port, tmp_path, idle_timeout=0.1)
    emitter = RecordingClient(url)
    try:
        emitter.start_eval("e1", str(tmp_path), "aggpartial.json", expected_count=10)
        await asyncio.sleep(0.5)
        await d._age_out_once()
        out = tmp_path / "aggpartial_unclosed.json"
        assert out.exists()
    finally:
        emitter.close()
        await _stop_daemon(d, task)


# ---------------------------------------------------------------------------
# Emitter when daemon is unreachable
# ---------------------------------------------------------------------------


def test_emitter_daemon_down_drops_silently(free_port: int) -> None:
    # Nothing listening — emitter must not raise; close cleanly within timeout.
    emitter = RecordingClient(f"ws://127.0.0.1:{free_port}")
    try:
        emitter.start_episode("e1", "s1", "ep1", "/tmp/none", "x_{status}", {})
        emitter.record("s1", "ep1", 0, {"x": 1})
        time.sleep(0.3)
    finally:
        emitter.close(timeout=1.0)


# ---------------------------------------------------------------------------
# Shutdown flush
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_shutdown_flush(free_port: int, tmp_path: Path) -> None:
    d, url, task = await _start_daemon(free_port, tmp_path)
    emitter = RecordingClient(url)
    try:
        emitter.start_episode("e1", "s1", "ep1", str(tmp_path), "sh_{idx:04d}_{status}", {"idx": 0})
        emitter.record("s1", "ep1", 0, {"x": 1})
        await asyncio.sleep(0.3)
    finally:
        emitter.close()
        await _stop_daemon(d, task)
    files = sorted(tmp_path.glob("sh_*"))
    assert any("_unclosed" in str(p) for p in files), files
