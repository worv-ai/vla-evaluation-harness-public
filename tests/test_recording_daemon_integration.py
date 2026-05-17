"""Integration tests — daemon runs as a real subprocess."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from vla_eval.recording_daemon.emitter import RecordingEmitter


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listen(port: int, deadline: float = 5.0) -> bool:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except OSError:
                time.sleep(0.05)
    return False


def _wait_path(path: Path, deadline: float = 5.0) -> bool:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def _spawn_daemon(tmp_path: Path) -> tuple[subprocess.Popen, int, int]:
    ws_port = _free_port()
    health_port = _free_port()
    cmd = [
        sys.executable,
        "-m",
        "vla_eval.cli.main",
        "recording-daemon",
        "--bind",
        f"ws://127.0.0.1:{ws_port}",
        "--out-dir",
        str(tmp_path),
        "--health-port",
        str(health_port),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_listen(ws_port):
        proc.terminate()
        raise RuntimeError("daemon subprocess never bound")
    return proc, ws_port, health_port


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_end_to_end(tmp_path: Path) -> None:
    proc, ws_port, _ = _spawn_daemon(tmp_path)
    try:
        emitter = RecordingEmitter(f"ws://127.0.0.1:{ws_port}")
        try:
            emitter.push_eval_start("e2e", str(tmp_path), "e2e_run.json", expected_count=3)
            for i in range(3):
                emitter.push_episode_start(
                    "e2e",
                    "shard0",
                    f"ep{i}",
                    str(tmp_path),
                    "e2e_{idx:04d}_{status}",
                    {"idx": i},
                )
                for s in range(2):
                    emitter.push_record_commit("shard0", f"ep{i}", s, {"reward": s + 0.5})
                emitter.push_episode_result("e2e", "shard0", f"ep{i}", "success", {"success": True})
                emitter.push_episode_end("shard0", f"ep{i}")
            emitter.push_eval_end("e2e")
        finally:
            emitter.close(timeout=3.0)

        aggregate = tmp_path / "e2e_run.json"
        assert _wait_path(aggregate)
        data = json.loads(aggregate.read_text())
        assert data["received_count"] == 3
        for i in range(3):
            assert _wait_path(tmp_path / f"e2e_{i:04d}_success.jsonl")
    finally:
        _kill(proc)


def test_two_shard_emitters_one_daemon(tmp_path: Path) -> None:
    proc, ws_port, _ = _spawn_daemon(tmp_path)
    try:
        url = f"ws://127.0.0.1:{ws_port}"
        a = RecordingEmitter(url)
        b = RecordingEmitter(url)
        try:
            for emitter, sid, idx in [(a, "sA", 0), (b, "sB", 1)]:
                emitter.push_episode_start(
                    "shared",
                    sid,
                    f"e{idx}",
                    str(tmp_path),
                    "shard_{idx:04d}_{status}",
                    {"idx": idx},
                )
                emitter.push_record_commit(sid, f"e{idx}", 0, {"sid": sid})
                emitter.push_episode_result("shared", sid, f"e{idx}", "success", {"success": True})
                emitter.push_episode_end(sid, f"e{idx}")
        finally:
            a.close(timeout=2.0)
            b.close(timeout=2.0)

        for idx in range(2):
            assert _wait_path(tmp_path / f"shard_{idx:04d}_success.jsonl")
    finally:
        _kill(proc)


def test_health_endpoint(tmp_path: Path) -> None:
    proc, _, health_port = _spawn_daemon(tmp_path)
    try:
        # Health server runs in a thread; give it a beat to bind.
        deadline = time.monotonic() + 3.0
        last_err = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{health_port}/health", timeout=1.0) as resp:
                    assert resp.status == 200
                    body = json.loads(resp.read().decode("utf-8"))
                    assert "buckets" in body and "evals" in body and "uptime_s" in body
                    return
            except Exception as exc:
                last_err = exc
                time.sleep(0.1)
        raise AssertionError(f"health endpoint never answered: {last_err}")
    finally:
        _kill(proc)


@pytest.mark.skipif(sys.platform != "linux", reason="signal-based subprocess shutdown is POSIX-only")
def test_signal_shutdown_flushes_unclosed(tmp_path: Path) -> None:
    """SIGTERM during an open episode: daemon should flush bucket as _unclosed."""
    import os
    import signal

    proc, ws_port, _ = _spawn_daemon(tmp_path)
    try:
        emitter = RecordingEmitter(f"ws://127.0.0.1:{ws_port}")
        try:
            emitter.push_episode_start("e1", "s1", "ep1", str(tmp_path), "sig_{idx:04d}_{status}", {"idx": 0})
            emitter.push_record_commit("s1", "ep1", 0, {"x": 1})
            time.sleep(0.5)
        finally:
            emitter.close(timeout=2.0)

        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5.0)
        # Daemon should have flushed the open bucket with _unclosed suffix.
        out = next(tmp_path.glob("sig_*"), None)
        assert out is not None and "_unclosed" in str(out)
    finally:
        _kill(proc)
