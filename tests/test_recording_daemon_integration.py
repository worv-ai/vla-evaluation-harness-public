"""End-to-end integration tests for the recording daemon.

These tests boot the daemon in a subprocess (via ``vla-eval recording-daemon``)
so signal handling, the ``process_request`` health endpoint, and atexit
flushes are exercised against the real entrypoint.

Slow-ish (~1–3 s each) because they wait for the daemon to bind and
flush real files; mark them ``integration``-style by keeping the
subprocess interactions focused.
"""

from __future__ import annotations

import json
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from vla_eval.recording_daemon.client import RecordingClient


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port_open(port: int, deadline: float = 10.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"daemon never bound port {port}")


def _wait_for_path(path: Path, deadline: float = 5.0) -> bool:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def _spawn_daemon(out_dir: Path, *, port: int, idle: float = 600.0) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vla_eval.cli.main",
        "recording-daemon",
        "--bind",
        f"ws://127.0.0.1:{port}",
        "--out-dir",
        str(out_dir),
        "--idle-timeout-sec",
        str(idle),
        "-v",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def test_end_to_end_run_via_subprocess_daemon(tmp_path: Path) -> None:
    port = _free_port()
    proc = _spawn_daemon(tmp_path, port=port)
    try:
        _wait_port_open(port)
        url = f"ws://127.0.0.1:{port}"
        client = RecordingClient(url)
        try:
            sid, eid = "shard-0", "ep-0"
            client.step(sid, eid, 0, {"r": 0.0})
            client.step(sid, eid, 1, {"r": 0.1})
            client.result(
                eval_id="ev",
                sid=sid,
                eid=eid,
                status="success",
                metrics={"success": True},
                context={"env_id": "demo", "episode_idx": 0},
                jsonl_path=str(tmp_path / "demo_ep0000_success.jsonl"),
                aggregate_dir=str(tmp_path),
                aggregate_filename="demo_aggregate.json",
            )
            client.eval_end("ev")
        finally:
            client.close()

        jsonl = tmp_path / "demo_ep0000_success.jsonl"
        agg = tmp_path / "demo_aggregate.json"
        assert _wait_for_path(jsonl), "per-episode jsonl never appeared"
        assert _wait_for_path(agg), "aggregate JSON never appeared"
        rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
        assert [r["step"] for r in rows] == [0, 1]
        body = json.loads(agg.read_text())
        assert body["count"] == 1
        assert body["results"][0]["env_id"] == "demo"
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)


def test_health_endpoint_on_same_ws_port(tmp_path: Path) -> None:
    port = _free_port()
    proc = _spawn_daemon(tmp_path, port=port)
    try:
        _wait_port_open(port)
        # Give the daemon a moment past bind so its WS server is registered for HTTP.
        time.sleep(0.2)
        for _ in range(20):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.0) as r:
                    body = json.loads(r.read())
                    assert "buckets" in body and "evals" in body and "uptime_s" in body
                    return
            except (urllib.error.URLError, ConnectionResetError):
                time.sleep(0.1)
        pytest.fail("health endpoint never responded")
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)


def test_sigterm_flushes_in_flight_buckets(tmp_path: Path) -> None:
    port = _free_port()
    proc = _spawn_daemon(tmp_path, port=port)
    try:
        _wait_port_open(port)
        client = RecordingClient(f"ws://127.0.0.1:{port}")
        try:
            client.step("s", "x", 0, {"v": 1})
            client.step("s", "x", 1, {"v": 2})
            # Give the daemon time to ingest before we kill it.
            time.sleep(0.3)
        finally:
            client.close()
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
        # The daemon should have flushed the bucket with _unclosed during shutdown.
        path = tmp_path / "s-x_unclosed.jsonl"
        assert path.exists(), f"shutdown flush missing: {path}"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows == [{"step": 0, "v": 1}, {"step": 1, "v": 2}]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_two_shard_emitters_fan_in(tmp_path: Path) -> None:
    """Two RecordingClients (separate connections) → same daemon, same eval.

    Drains both shard emitters before pushing ``eval_end`` via a third
    client so the aggregate doesn't close while one shard's ``result`` is
    still in flight.
    """
    port = _free_port()
    proc = _spawn_daemon(tmp_path, port=port)
    try:
        _wait_port_open(port)
        url = f"ws://127.0.0.1:{port}"
        c1 = RecordingClient(url)
        c2 = RecordingClient(url)
        try:
            for emit, sid in ((c1, "shard-A"), (c2, "shard-B")):
                emit.step(sid, "ep", 0, {"who": sid})
                emit.result(
                    eval_id="merged",
                    sid=sid,
                    eid="ep",
                    status="success",
                    metrics={"success": True},
                    context={"env_id": "x", "episode_idx": 0},
                    jsonl_path=str(tmp_path / f"{sid}_ep.jsonl"),
                    aggregate_dir=str(tmp_path),
                    aggregate_filename="merged.json",
                )
        finally:
            # Drain both shards before declaring the eval done.
            c1.close()
            c2.close()
        time.sleep(0.2)
        closer = RecordingClient(url)
        try:
            closer.eval_end("merged")
        finally:
            closer.close()

        agg = tmp_path / "merged.json"
        assert _wait_for_path(agg)
        body = json.loads(agg.read_text())
        assert body["count"] == 2
        assert {r["sid"] for r in body["results"]} == {"shard-A", "shard-B"}
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
