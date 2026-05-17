"""Unit tests for :class:`vla_eval.recording.EpisodeRecorder` and ``NullEpisodeRecorder``.

These cover the composition layer (video on disk + step push over WS) without
booting a real daemon — a captured stand-in RecordingClient verifies the calls
:meth:`close` makes on the happy path and that ``NullEpisodeRecorder`` is a
strict no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from vla_eval.recording import EpisodeRecorder, NullEpisodeRecorder
from vla_eval.recording_daemon.client import RecordingClient


class _CapturingClient:
    """Minimal in-memory stand-in for ``RecordingClient`` — records every call."""

    def __init__(self) -> None:
        self.steps: list[tuple[str, str, int, dict[str, Any]]] = []
        self.results: list[dict[str, Any]] = []
        self.evals_ended: list[str] = []

    def step(self, sid: str, eid: str, step_id: int, fields: dict[str, Any]) -> None:
        self.steps.append((sid, eid, step_id, fields))

    def result(self, **kw: Any) -> None:
        self.results.append(kw)

    def eval_end(self, eval_id: str) -> None:
        self.evals_ended.append(eval_id)


def _frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_null_recorder_is_strict_noop() -> None:
    rec = NullEpisodeRecorder()
    rec.record_video(_frame())
    rec.record_step({"reward": 1.0})
    rec.close(status="success", metrics={"success": True})
    # close on an already-closed null is fine too
    rec.close(status="success", metrics={})
    assert rec.sid == ""
    assert rec.eid == ""
    assert rec.eval_id == ""


def test_record_step_auto_increments(tmp_path: Path) -> None:
    client = _CapturingClient()
    rec = EpisodeRecorder(
        client=cast(RecordingClient, client),
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=tmp_path,
        filename_stem="ep_{status}",
        context={},
        aggregate_dir=tmp_path,
        aggregate_filename="agg.json",
        record_video=False,  # video uses a real imageio writer; skip in this unit test
    )
    rec.record_step({"a": 1})
    rec.record_step({"a": 2})
    rec.record_step({"a": 3})
    assert [s[2] for s in client.steps] == [0, 1, 2]
    assert [s[3]["a"] for s in client.steps] == [1, 2, 3]


def test_record_step_explicit_step_wins(tmp_path: Path) -> None:
    client = _CapturingClient()
    rec = EpisodeRecorder(
        client=cast(RecordingClient, client),
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=tmp_path,
        filename_stem="ep_{status}",
        context={},
        aggregate_dir=tmp_path,
        aggregate_filename="agg.json",
        record_video=False,
    )
    rec.record_step({"step": 7, "a": "x"})
    rec.record_step({"a": "y"})  # auto picks next > 7
    assert client.steps[0][2] == 7
    assert client.steps[1][2] == 8


def test_close_sends_result_with_resolved_jsonl_path(tmp_path: Path) -> None:
    client = _CapturingClient()
    rec = EpisodeRecorder(
        client=cast(RecordingClient, client),
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=tmp_path,
        filename_stem="{env_id}_ep{episode_idx:04d}_{status}",
        context={"env_id": "demo", "episode_idx": 3},
        aggregate_dir=tmp_path,
        aggregate_filename="agg.json",
        record_video=False,
    )
    rec.record_step({"reward": 0.5})
    rec.close(status="fail", metrics={"success": False})

    assert len(client.results) == 1
    r = client.results[0]
    assert r["eval_id"] == "ev"
    assert r["status"] == "fail"
    assert r["metrics"] == {"success": False}
    assert r["context"] == {"env_id": "demo", "episode_idx": 3}
    assert r["jsonl_path"] == str(tmp_path / "demo_ep0003_fail.jsonl")
    assert r["aggregate_dir"] == str(tmp_path)
    assert r["aggregate_filename"] == "agg.json"


def test_close_is_idempotent(tmp_path: Path) -> None:
    client = _CapturingClient()
    rec = EpisodeRecorder(
        client=cast(RecordingClient, client),
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=tmp_path,
        filename_stem="ep_{status}",
        context={},
        aggregate_dir=tmp_path,
        aggregate_filename="agg.json",
        record_video=False,
    )
    rec.close(status="success", metrics={"success": True})
    rec.close(status="success", metrics={"success": True})
    assert len(client.results) == 1


def test_record_step_disabled_skips_client(tmp_path: Path) -> None:
    client = _CapturingClient()
    rec = EpisodeRecorder(
        client=cast(RecordingClient, client),
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=tmp_path,
        filename_stem="ep_{status}",
        context={},
        aggregate_dir=tmp_path,
        aggregate_filename="agg.json",
        record_video=False,
        record_step=False,
    )
    rec.record_step({"a": 1})
    rec.record_step({"a": 2})
    assert client.steps == []
    # close still pushes RESULT regardless of step capture being off
    rec.close(status="success", metrics={})
    assert len(client.results) == 1


def test_video_writes_real_mp4(tmp_path: Path) -> None:
    """End-to-end with the actual EpisodeVideoRecorder underneath."""
    client = _CapturingClient()
    rec = EpisodeRecorder(
        client=cast(RecordingClient, client),
        sid="s",
        eid="e",
        eval_id="ev",
        output_dir=tmp_path,
        filename_stem="vid_{status}",
        context={},
        aggregate_dir=tmp_path,
        aggregate_filename="agg.json",
        record_video=True,
    )
    for _ in range(3):
        rec.record_video(_frame())
    rec.close(status="success", metrics={})
    mp4 = tmp_path / "vid_success.mp4"
    assert mp4.exists(), "video file not written"
    assert mp4.stat().st_size > 0
