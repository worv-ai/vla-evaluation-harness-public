"""Unit tests for ``vla_eval.benchmarks.data_recording``.

Covers the new ``EpisodeRecorder.from_config`` / ``record_row`` /
``NullEpisodeRecorder`` surface that benchmark adapters now lean on
unconditionally (no per-call ``if recorder is not None`` checks).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vla_eval.benchmarks.data_recording import (
    EpisodeRecorder,
    NullEpisodeRecorder,
)

_ALLOWED = frozenset({"reward", "done", "success"})


def _frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


def test_from_config_none_returns_null() -> None:
    rec = EpisodeRecorder.from_config(None, allowed_fields=_ALLOWED)
    assert isinstance(rec, NullEpisodeRecorder)
    assert rec.active is False


def test_from_config_empty_dict_returns_null() -> None:
    assert isinstance(EpisodeRecorder.from_config({}, allowed_fields=_ALLOWED), NullEpisodeRecorder)


def test_from_config_builds_real_recorder(tmp_path: Path) -> None:
    rec = EpisodeRecorder.from_config(
        {"output_dir": str(tmp_path), "record_video": False},
        allowed_fields=_ALLOWED,
    )
    assert isinstance(rec, EpisodeRecorder)


def test_from_config_rejects_unknown_step_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown step_fields"):
        EpisodeRecorder.from_config(
            {"output_dir": str(tmp_path), "step_fields": ["bogus"]},
            allowed_fields=_ALLOWED,
        )


def test_from_config_default_fps_used_when_unset(tmp_path: Path) -> None:
    rec = EpisodeRecorder.from_config(
        {"output_dir": str(tmp_path), "record_step": False},
        allowed_fields=_ALLOWED,
        default_fps=30,
    )
    assert isinstance(rec, EpisodeRecorder)
    assert rec._video is not None
    assert rec._video.fps == 30  # type: ignore[attr-defined]


def test_from_config_explicit_fps_overrides_default(tmp_path: Path) -> None:
    rec = EpisodeRecorder.from_config(
        {"output_dir": str(tmp_path), "fps": 15, "record_step": False},
        allowed_fields=_ALLOWED,
        default_fps=30,
    )
    assert isinstance(rec, EpisodeRecorder)
    assert rec._video is not None
    assert rec._video.fps == 15  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# NullEpisodeRecorder no-op surface
# ---------------------------------------------------------------------------


def test_null_recorder_methods_are_noops() -> None:
    rec = NullEpisodeRecorder()
    # All these should be callable without state and never raise.
    rec.start({"env_id": "x", "episode_idx": 0})
    rec.record_frame(_frame())
    rec.record_frame(None)
    rec.record_step({"step": 0, "reward": 1.0})
    rec.record_row(reward=1.0, done=False)
    rec.save(status="success")
    rec.discard()
    assert rec.active is False


# ---------------------------------------------------------------------------
# record_row — filter + step counter
# ---------------------------------------------------------------------------


def test_record_row_filters_and_stamps_step(tmp_path: Path) -> None:
    rec = EpisodeRecorder.from_config(
        {"output_dir": str(tmp_path), "record_video": False, "step_fields": ["reward", "done"]},
        allowed_fields=_ALLOWED,
    )
    assert isinstance(rec, EpisodeRecorder)
    rec.start({"env_id": "Test", "episode_idx": 0})
    rec.record_row(reward=1.0, done=False, success=True)  # success dropped by filter
    rec.record_row(reward=2.0, done=True, success=False)
    rec.save(status="success")

    rows = _read_jsonl(tmp_path / "Test_ep0000_success.jsonl")
    assert rows == [
        {"step": 0, "reward": 1.0, "done": False},
        {"step": 1, "reward": 2.0, "done": True},
    ]


def test_record_row_default_step_fields_is_full_allowed_set(tmp_path: Path) -> None:
    rec = EpisodeRecorder.from_config(
        {"output_dir": str(tmp_path), "record_video": False},
        allowed_fields=_ALLOWED,
    )
    assert isinstance(rec, EpisodeRecorder)
    rec.start({"env_id": "T", "episode_idx": 0})
    rec.record_row(reward=1.0, done=False, success=True, extra="dropped")
    rec.save(status="fail")

    rows = _read_jsonl(tmp_path / "T_ep0000_fail.jsonl")
    assert rows == [{"step": 0, "reward": 1.0, "done": False, "success": True}]


def test_record_row_before_start_is_noop(tmp_path: Path) -> None:
    rec = EpisodeRecorder.from_config(
        {"output_dir": str(tmp_path), "record_video": False},
        allowed_fields=_ALLOWED,
    )
    assert isinstance(rec, EpisodeRecorder)
    # No start() — record_row must silently drop.
    rec.record_row(reward=1.0)
    assert list(tmp_path.glob("*.jsonl")) == []


def test_record_row_step_resets_per_episode(tmp_path: Path) -> None:
    rec = EpisodeRecorder.from_config(
        {"output_dir": str(tmp_path), "record_video": False},
        allowed_fields=_ALLOWED,
    )
    assert isinstance(rec, EpisodeRecorder)
    for ep in range(2):
        rec.start({"env_id": "T", "episode_idx": ep})
        rec.record_row(reward=float(ep))
        rec.record_row(reward=float(ep) + 0.5)
        rec.save(status="success")

    for ep in range(2):
        rows = _read_jsonl(tmp_path / f"T_ep{ep:04d}_success.jsonl")
        assert [r["step"] for r in rows] == [0, 1]


# ---------------------------------------------------------------------------
# record_frame(None) is a no-op (used by benchmarks whose extract may fail)
# ---------------------------------------------------------------------------


def test_record_frame_none_does_not_raise(tmp_path: Path) -> None:
    rec = EpisodeRecorder.from_config(
        {"output_dir": str(tmp_path), "record_step": False},
        allowed_fields=_ALLOWED,
    )
    assert isinstance(rec, EpisodeRecorder)
    rec.start({"env_id": "T", "episode_idx": 0})
    rec.record_frame(None)
    rec.record_frame(_frame())
    rec.save(status="success")
    assert (tmp_path / "T_ep0000_success.mp4").exists()
