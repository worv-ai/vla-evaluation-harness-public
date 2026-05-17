"""Per-episode recorder — thin wrapper around the recording-daemon client.

Forwards ``record_step`` / ``record_frame`` / ``save`` to ``vla_eval.recording.client()``.
No-op when no client is installed (recording disabled).
"""

from __future__ import annotations

import logging
import string
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vla_eval import recording

logger = logging.getLogger(__name__)


@dataclass
class RecordingConfig:
    """Configuration for per-episode recording, passed via benchmark YAML ``params.recording``."""

    output_dir: str = "/workspace/results/episodes"
    record_video: bool = True
    record_step: bool = True
    step_fields: list[str] = field(default_factory=list)


__all__ = ["EpisodeRecorder", "RecordingConfig"]


class EpisodeRecorder:
    """Per-episode recorder. Forwards events to the recording daemon when a client is installed."""

    def __init__(
        self,
        output_dir: str | Any = "",
        *,
        record_video: bool = True,
        record_step: bool = True,
        filename_stem: str = "{env_id}_ep{episode_idx:04d}_{status}",
        fps: int = 20,
    ) -> None:
        # output_dir is kept on the recorder so the orchestrator can publish it in EPISODE_START.
        self.output_dir = str(output_dir) if output_dir else ""
        self._record_video = record_video
        self._record_step = record_step
        self._filename_stem = filename_stem
        self.fps = fps
        self._required_context = tuple(
            bare
            for _, field_name, _, _ in string.Formatter().parse(filename_stem)
            if field_name
            for bare in [field_name.split(".")[0].split("[")[0]]
            if bare and bare != "status"
        )
        self._context: dict[str, Any] = {}
        self._sid: str | None = None
        self._eid: str | None = None
        self._step_counter = 0

    @property
    def filename_template(self) -> str:
        return self._filename_stem

    @property
    def required_context(self) -> tuple[str, ...]:
        return self._required_context

    def bind(self, sid: str | None, eid: str | None) -> None:
        """Set the (sid, eid) for the next episode. Pass None to disable."""
        self._sid = sid
        self._eid = eid

    def _bound_client(self) -> tuple[Any, str, str] | None:
        """Return (client, sid, eid) iff all three are set; else None."""
        c = recording.client()
        if c is None or self._sid is None or self._eid is None:
            return None
        return c, self._sid, self._eid

    @property
    def active(self) -> bool:
        return self._bound_client() is not None

    def start(self, context: Mapping[str, Any]) -> None:
        missing = [k for k in self._required_context if k not in context]
        if missing:
            raise ValueError(f"EpisodeRecorder.start: missing required context keys: {missing}")
        self._context = dict(context)
        self._step_counter = 0

    def record_frame(self, frame: np.ndarray) -> None:
        if not self._record_video:
            return
        bound = self._bound_client()
        if bound is None:
            return
        client, sid, eid = bound
        client.video_frame(sid, eid, frame)

    def record_step(self, data: dict[str, Any]) -> None:
        if not self._record_step:
            return
        bound = self._bound_client()
        if bound is None:
            return
        client, sid, eid = bound
        client.record(sid, eid, self._step_counter, dict(data))
        self._step_counter += 1

    def save(self, **extra: Any) -> None:
        bound = self._bound_client()
        if bound is None:
            return
        client, sid, eid = bound
        client.end_episode(sid, eid)

    def discard(self) -> None:
        bound = self._bound_client()
        if bound is None:
            return
        client, sid, eid = bound
        client.end_episode(sid, eid)
