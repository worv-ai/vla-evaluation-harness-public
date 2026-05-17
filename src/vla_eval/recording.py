"""Recording API used by the harness and (optionally) by model servers.

Two surfaces:

- :func:`set_client` / :func:`client`: install the :class:`RecordingClient`
  for this process. Callers that live inside a model server (e.g.
  reflex-train's ``_record_step_trace``) can ``recording.client().step(...)``
  to push step rows tagged with the active ``(sid, eid)`` they received in
  ``EPISODE_START``.

- :class:`EpisodeRecorder`: per-episode helper built by the orchestrator
  and handed to the benchmark via ``Benchmark.start_episode(task, recorder=...)``.
  Benchmarks call ``record_video()`` / ``record_step()`` without ever
  touching ``sid``/``eid``/the WS client directly. The orchestrator owns
  the lifecycle: it constructs the recorder, the benchmark uses it during
  the episode, and the orchestrator calls ``close(status, metrics)`` once
  the result is known — even on exception paths.

- :class:`NullEpisodeRecorder`: drop-in stand-in when recording is
  disabled. All methods are no-ops so benchmarks can call without
  ``if recorder is None`` guards.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from vla_eval.recording_daemon.client import RecordingClient


@dataclass
class RecordingContext:
    """Per-episode recording metadata supplied by the benchmark.

    The orchestrator merges this with ``(sid, eid, eval_id, client, aggregate_*)``
    to build the :class:`EpisodeRecorder` that's handed back to the
    benchmark via ``start_episode``.

    Attributes:
        output_dir: Where the per-episode mp4 + jsonl land.
        filename_stem: ``str.format`` template for the per-episode filename
            (no extension). MUST include ``{status}`` since ``close()``
            injects it. Keys present in ``context`` are also substituted.
            Example: ``"{env_id}_ep{episode_idx:04d}_{status}"``.
        context: Task identifiers consumed by ``filename_stem`` and copied
            into each aggregate result row (so post-hoc analysis can group
            by env / task / episode).
        record_video: Whether to write a per-episode mp4. Default True.
        record_step: Whether to push per-step rows to the daemon. Default True.
        video_fps: Frame rate for the mp4 writer. Default 20.
    """

    output_dir: str | Path
    filename_stem: str
    context: dict[str, Any] = field(default_factory=dict)
    record_video: bool = True
    record_step: bool = True
    video_fps: int = 20


logger = logging.getLogger(__name__)

_lock = threading.Lock()
_emitter: "RecordingClient | None" = None


def set_client(emitter: "RecordingClient | None") -> None:
    """Install (or clear) the process-wide recording emitter. Call once at startup."""
    global _emitter
    with _lock:
        _emitter = emitter


def client() -> "RecordingClient | None":
    """Return the installed emitter, or ``None`` if recording is disabled."""
    return _emitter


class EpisodeRecorder:
    """Per-episode recorder built by the orchestrator.

    Owns:
      - an :class:`~vla_eval.benchmarks.video.EpisodeVideoRecorder` (optional)
        that writes to a final mp4 path on the same host (binary never
        crosses the wire);
      - a reference to the :class:`RecordingClient` for ``step`` / ``result``
        emits to the daemon.

    Constructed once per episode; safe to use across the episode's reset
    and step calls. Call :meth:`close` exactly once with the resolved
    status and metrics.
    """

    def __init__(
        self,
        *,
        client: "RecordingClient | None",
        sid: str,
        eid: str,
        eval_id: str,
        output_dir: str | Path,
        filename_stem: str,
        context: dict[str, Any],
        aggregate_dir: str | Path,
        aggregate_filename: str,
        record_video: bool = True,
        record_step: bool = True,
        video_fps: int = 20,
    ) -> None:
        self._client = client
        self._sid = sid
        self._eid = eid
        self._eval_id = eval_id
        self._output_dir = Path(output_dir)
        self._filename_stem = filename_stem
        self._context = dict(context)
        self._aggregate_dir = str(aggregate_dir)
        self._aggregate_filename = aggregate_filename
        self._record_step = record_step
        self._step_counter = 0
        self._closed = False
        self._video: Any = None  # EpisodeVideoRecorder | None — lazy import to avoid cycle

        if record_video:
            from vla_eval.benchmarks.video import EpisodeVideoRecorder

            self._video = EpisodeVideoRecorder(
                output_dir=self._output_dir,
                filename=filename_stem + ".mp4",
                fps=video_fps,
            )
            try:
                self._video.start(self._context)
            except Exception:
                logger.exception("EpisodeVideoRecorder.start failed; video disabled for this episode")
                self._video = None

    # ------------------------------------------------------------------
    # Identifiers (read by benchmarks that want to attribute external work)
    # ------------------------------------------------------------------

    @property
    def sid(self) -> str:
        return self._sid

    @property
    def eid(self) -> str:
        return self._eid

    @property
    def eval_id(self) -> str:
        return self._eval_id

    # ------------------------------------------------------------------
    # Capture API (called per frame / per step)
    # ------------------------------------------------------------------

    def record_video(self, frame: "np.ndarray") -> None:
        if self._video is not None:
            self._video.record(frame)

    def record_step(self, row: dict[str, Any]) -> None:
        if not self._record_step or self._client is None:
            return
        step_id = int(row.pop("step", self._step_counter))
        self._step_counter = max(self._step_counter, step_id + 1)
        self._client.step(self._sid, self._eid, step_id, row)

    # ------------------------------------------------------------------
    # Close (orchestrator owns this — happy and exception paths)
    # ------------------------------------------------------------------

    def close(self, *, status: str, metrics: dict[str, Any]) -> None:
        if self._closed:
            return
        self._closed = True

        if self._video is not None:
            try:
                self._video.save(status=status)
            except FileExistsError as exc:
                logger.warning("Episode video already exists: %s", exc)
            except Exception:
                logger.exception("EpisodeVideoRecorder.save failed for sid=%s eid=%s", self._sid, self._eid)
            self._video = None

        if self._client is None:
            return

        jsonl_name = (self._filename_stem + ".jsonl").format(status=status, **self._context)
        jsonl_path = str(self._output_dir / jsonl_name)
        try:
            self._client.result(
                eval_id=self._eval_id,
                sid=self._sid,
                eid=self._eid,
                status=status,
                metrics=metrics,
                context=self._context,
                jsonl_path=jsonl_path,
                aggregate_dir=self._aggregate_dir,
                aggregate_filename=self._aggregate_filename,
            )
        except Exception:
            logger.exception("EpisodeRecorder.close: failed to push RESULT")


class NullEpisodeRecorder(EpisodeRecorder):
    """No-op recorder. All methods return immediately.

    Hand this to a benchmark when recording is disabled so the benchmark
    doesn't need ``if recorder is not None`` guards on every call.
    """

    def __init__(self) -> None:  # type: ignore[override]
        # Skip the parent's heavy setup; mark closed so close() is a no-op too.
        self._closed = True
        self._sid = ""
        self._eid = ""
        self._eval_id = ""

    def record_video(self, frame: "np.ndarray") -> None:  # type: ignore[override]
        pass

    def record_step(self, row: dict[str, Any]) -> None:  # type: ignore[override]
        pass

    def close(self, *, status: str, metrics: dict[str, Any]) -> None:  # type: ignore[override]
        pass
