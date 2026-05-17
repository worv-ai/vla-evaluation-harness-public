"""Composite per-episode recorder — video + structured JSONL in one directory.

Single ``output_dir`` receives both ``.mp4`` and ``.jsonl`` with matching
filenames (e.g. ``BinFill_ep0000_fail.mp4`` + ``BinFill_ep0000_fail.jsonl``).

Typical usage::

    recorder = EpisodeRecorder("/workspace/results/episodes")
    recorder.start({"env_id": "BinFill", "episode_idx": 0})
    recorder.record_frame(front_rgb)
    recorder.record_step({"step": n, "gt_subgoal": "pick up cube", ...})
    recorder.save(status="fail")

Per-step jsonl is staged in memory and written atomically once at
``save()``: by then all server-side ``RECORD_COMMIT`` frames for the
episode have either landed in their ``(sid, step_id)`` bucket or are
discarded with a WARNING.  This eliminates the prior race where late
RECORD frames were mis-attributed to the next step.
"""

from __future__ import annotations

import json
import logging
import os
import string
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vla_eval import recording as _recording
from vla_eval.benchmarks.recording import EpisodeVideoRecorder

logger = logging.getLogger(__name__)

# Brief wait for a matching RECORD_COMMIT to land in the bucket before
# record_step gives up and emits the row without server-side extras.
# Sync mode never hits the wait (commits arrive before ACTION); realtime
# mode occasionally needs it under inversion.
_RECORD_COMMIT_WAIT_SEC: float = 0.1


@dataclass
class RecordingConfig:
    """Configuration for per-episode recording, passed via benchmark YAML ``params.recording``."""

    output_dir: str = "/workspace/results/episodes"
    record_video: bool = True
    record_step: bool = True
    step_fields: list[str] = field(default_factory=list)


__all__ = ["EpisodeRecorder", "RecordingConfig"]


class EpisodeRecorder:
    """Composite recorder: optional video (mp4) + optional data (JSONL)."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        record_video: bool = True,
        record_step: bool = True,
        filename_stem: str = "{env_id}_ep{episode_idx:04d}_{status}",
        fps: int = 20,
    ) -> None:
        out = Path(output_dir)
        if record_video or record_step:
            out.mkdir(parents=True, exist_ok=True)
        self._video: EpisodeVideoRecorder | None = (
            EpisodeVideoRecorder(output_dir=out, filename=filename_stem + ".mp4", fps=fps) if record_video else None
        )
        self._data_dir = out if record_step else None
        self._data_filename = filename_stem + ".jsonl"
        self._staged_rows: list[dict[str, Any]] = []
        self._episode_active = False
        self._context: dict[str, Any] = {}
        self._required_context = tuple(
            bare
            for _, field_name, _, _ in string.Formatter().parse(filename_stem)
            if field_name
            for bare in [field_name.split(".")[0].split("[")[0]]
            if bare and bare != "status"
        )

    @property
    def active(self) -> bool:
        return (self._video is not None and self._video.active) or self._episode_active

    def start(self, context: Mapping[str, Any]) -> None:
        missing = [k for k in self._required_context if k not in context]
        if missing:
            raise ValueError(f"EpisodeRecorder.start: missing required context keys: {missing}")
        self._context = dict(context)
        if self._video is not None:
            self._video.start(context)
        if self._data_dir is not None:
            self._staged_rows = []
            self._episode_active = True

    def record_frame(self, frame: np.ndarray) -> None:
        if self._video is not None:
            self._video.record(frame)

    def record_step(self, data: dict[str, Any]) -> None:
        if not self._episode_active:
            return
        step_id = data.get("step")
        if not isinstance(step_id, int):
            raise ValueError(
                f"EpisodeRecorder.record_step: data must include integer 'step' key for "
                f"(session, step) bucket matching; got step={step_id!r}"
            )
        rec = _recording.get_default()
        sid = rec.current_session()
        extras: dict[str, Any] = {}
        if sid is not None:
            extras = rec.drain_step(sid, step_id, timeout=_RECORD_COMMIT_WAIT_SEC)
        if extras:
            collisions = sorted(set(data) & set(extras))
            if collisions:
                logger.warning(
                    "record_step: dropping recording field(s) that collide with benchmark schema: %s",
                    collisions,
                )
            row = {**extras, **data}
        else:
            row = dict(data)
        self._staged_rows.append(row)

    def save(self, **extra: Any) -> None:
        status_kwargs = {**self._context, **extra}
        if self._video is not None:
            self._video.save(**extra)
        if self._episode_active and self._data_dir is not None:
            try:
                final_name = self._data_filename.format(**status_kwargs)
                final_path = self._data_dir / final_name
                uid = uuid.uuid4().hex[:12]
                working = self._data_dir / f".data-{uid}.jsonl"
                with open(working, "w", encoding="utf-8") as fh:
                    for row in self._staged_rows:
                        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                if final_path.exists():
                    final_path.unlink()
                working.rename(final_path)
                logger.info("Saved episode data: %s (%d rows)", final_path, len(self._staged_rows))
            except Exception:
                logger.warning("Failed to save episode data", exc_info=True)
        self._episode_active = False
        self._staged_rows = []

    def discard(self) -> None:
        if self._video is not None:
            self._video.discard()
        self._episode_active = False
        self._staged_rows = []
