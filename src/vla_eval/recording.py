"""Recording: per-episode step rows + episode results + per-eval metadata in SQLite.

Why SQLite?

The harness records two things per episode:

1. **Step rows** — schema-flex JSON documents keyed by ``(sid, eid, step_id)``.
   Both the orchestrator (running the benchmark sim) and the model server
   (e.g. reflex-train pushing inference traces) may write rows for the same
   ``(sid, eid, step_id)`` with **disjoint or overlapping field sets** —
   the store must field-union them atomically. SQLite + ``json_patch``
   UPSERT does this in one line; coordinating the same merge across two
   processes any other way (file locks, append-only logs + post-merge,
   running a daemon) costs significantly more code and complexity.

2. **Episode results + eval metadata** — small structured records the
   orchestrator writes once per episode / once per eval. Stable schema,
   keyed by ``(sid, eid)`` and ``eval_id`` respectively.

A *single* SQLite file per eval (``recording-<eval_id>.sqlite``) lets all
shards on one host concurrent-write under WAL mode and lets the model
server (a third process) join in. ``vla-eval merge`` reads the DB after
the run and emits the human-readable jsonl + aggregate JSON downstream
tooling expects.

This file is the entire data plane. There is no daemon, no WebSocket, no
``recording.set_client`` global emitter — external callers receive the
DB path via the ``EPISODE_START`` payload and open their own connection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import numpy as np


logger = logging.getLogger(__name__)


EpisodeStatus = Literal["success", "fail", "error"]


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=30000;

CREATE TABLE IF NOT EXISTS eval_metadata (
    eval_id    TEXT PRIMARY KEY,
    safe_name  TEXT NOT NULL,
    metadata   TEXT NOT NULL  -- JSON: benchmark, mode, config, harness_version, server_info, metric_keys
);

CREATE TABLE IF NOT EXISTS episode_results (
    sid             TEXT NOT NULL,
    eid             TEXT NOT NULL,
    eval_id         TEXT NOT NULL,
    task_name       TEXT,
    episode_id      INTEGER,
    status          TEXT,            -- 'success' | 'fail' | 'error'
    metrics         TEXT,            -- JSON
    steps           INTEGER,
    elapsed_sec     REAL,
    context         TEXT,            -- JSON
    jsonl_path      TEXT,            -- resolved final filename for ``vla-eval merge``
    failure_reason  TEXT,
    failure_detail  TEXT,
    PRIMARY KEY (sid, eid)
);
CREATE INDEX IF NOT EXISTS idx_episode_results_eval ON episode_results(eval_id);

CREATE TABLE IF NOT EXISTS step_rows (
    sid      TEXT NOT NULL,
    eid      TEXT NOT NULL,
    step_id  INTEGER NOT NULL,
    fields   TEXT NOT NULL,  -- JSON document; multi-writer field-union via json_patch
    PRIMARY KEY (sid, eid, step_id)
);
"""


# ---------------------------------------------------------------------------
# RecordingContext — benchmark tells orchestrator how to name files
# ---------------------------------------------------------------------------


@dataclass
class RecordingContext:
    """Per-episode metadata the benchmark hands the orchestrator.

    The orchestrator combines this with ``(sid, eid, eval_id, store)`` to
    build the :class:`EpisodeRecorder` passed back to ``start_episode``.

    Attributes:
        output_dir: Where the final mp4 and jsonl land (after ``vla-eval merge``).
        filename_stem: ``str.format`` template that must include ``{status}``.
            Keys from ``context`` are also substituted. Example:
            ``"{env_id}_ep{episode_idx:04d}_{status}"``.
        context: Task identifiers consumed by ``filename_stem`` and copied
            into each aggregate result row.
        record_video: Whether to write a per-episode mp4.
        record_step: Whether to push per-step rows to the recording DB.
        video_fps: Frame rate for the mp4 writer.
    """

    output_dir: str | Path
    filename_stem: str
    context: dict[str, Any] = field(default_factory=dict)
    record_video: bool = True
    record_step: bool = True
    video_fps: int = 20


# ---------------------------------------------------------------------------
# RecordingStore — SQLite connection + idempotent writes
# ---------------------------------------------------------------------------


def db_path_for_eval(output_dir: str | Path, eval_id: str) -> Path:
    """Canonical SQLite path for an eval. All shards on one host point here."""
    return Path(output_dir) / f"recording-{eval_id}.sqlite"


class RecordingStore:
    """SQLite connection holder. One per process; multiple stores against the
    same file are race-safe via WAL.

    Used both by the orchestrator (full recorder lifecycle) and by external
    callers — model-server code (e.g. reflex-train) receives the DB path via
    ``EPISODE_START.recording.db_path`` and opens its own ``RecordingStore``
    to push step rows.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None, timeout=30.0)
        self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def upsert_eval_metadata(self, eval_id: str, safe_name: str, metadata: dict[str, Any]) -> None:
        """First-writer-wins. Subsequent shards' identical metadata is a no-op."""
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO eval_metadata (eval_id, safe_name, metadata) VALUES (?, ?, ?)",
                (eval_id, safe_name, json.dumps(metadata, default=str)),
            )

    def upsert_episode_result(
        self,
        *,
        sid: str,
        eid: str,
        eval_id: str,
        task_name: str,
        episode_id: int,
        status: str,
        metrics: dict[str, Any],
        steps: int,
        elapsed_sec: float,
        context: dict[str, Any],
        jsonl_path: str,
        failure_reason: str | None,
        failure_detail: str | None,
    ) -> None:
        """Replace-or-insert. Only the orchestrator writes episode_results,
        and exactly once per episode, so the REPLACE handles a retry path
        (e.g. orchestrator crash + restart with same eval_id)."""
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO episode_results
                  (sid, eid, eval_id, task_name, episode_id, status, metrics,
                   steps, elapsed_sec, context, jsonl_path,
                   failure_reason, failure_detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    eid,
                    eval_id,
                    task_name,
                    episode_id,
                    status,
                    json.dumps(metrics, default=str),
                    steps,
                    elapsed_sec,
                    json.dumps(context, default=str),
                    jsonl_path,
                    failure_reason,
                    failure_detail,
                ),
            )

    def upsert_step_rows(self, sid: str, eid: str, rows: dict[int, dict[str, Any]]) -> None:
        """Field-union UPSERT for the multi-writer case.

        Two writers may concurrently insert rows for the same
        ``(sid, eid, step_id)`` with different field sets — for example the
        orchestrator-side benchmark records ``{reward, gt_subgoal, robot_state}``
        while the model-server-side records ``{inference_ms, action_logits}``.
        ``json_patch`` merges them: keys from the new value overwrite keys with
        the same name; keys absent from the new value are preserved.
        """
        if not rows:
            return
        payload = [(sid, eid, step_id, json.dumps(fields, default=str)) for step_id, fields in rows.items()]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO step_rows (sid, eid, step_id, fields)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sid, eid, step_id)
                  DO UPDATE SET fields = json_patch(fields, excluded.fields)
                """,
                payload,
            )


# ---------------------------------------------------------------------------
# EpisodeRecorder — orchestrator side (owns video + episode lifecycle)
# ---------------------------------------------------------------------------


class EpisodeRecorder:
    """Per-episode recorder built by the orchestrator and handed to the
    benchmark via ``start_episode(task, recorder=...)``.

    Benchmarks call ``record_video(frame)`` and ``record_step(row)``. The
    orchestrator calls ``close(status, metrics, ...)`` exactly once (happy
    or exception path); ``close`` flushes the in-memory step buffer to
    SQLite in one transaction, saves the mp4 to its final path, and writes
    the ``episode_results`` row that ``vla-eval merge`` keys off.
    """

    def __init__(
        self,
        *,
        store: RecordingStore,
        sid: str,
        eid: str,
        eval_id: str,
        output_dir: str | Path,
        filename_stem: str,
        context: dict[str, Any],
        record_video: bool = True,
        record_step: bool = True,
        video_fps: int = 20,
    ) -> None:
        self._store = store
        self._sid = sid
        self._eid = eid
        self._eval_id = eval_id
        self._output_dir = Path(output_dir)
        self._filename_stem = filename_stem
        self._context = dict(context)
        self._record_step = record_step
        self._steps: dict[int, dict[str, Any]] = {}
        self._next_step = 0
        self._closed = False
        self._video: Any = None
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

    # -- Identifiers -------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return True

    @property
    def sid(self) -> str:
        return self._sid

    @property
    def eid(self) -> str:
        return self._eid

    @property
    def eval_id(self) -> str:
        return self._eval_id

    @property
    def db_path(self) -> str:
        return str(self._store.db_path)

    # -- Capture API -------------------------------------------------------

    def record_video(self, frame: "np.ndarray") -> None:
        if self._video is not None:
            self._video.record(frame)

    def record_step(self, row: dict[str, Any]) -> None:
        if not self._record_step:
            return
        step_id = int(row.pop("step", self._next_step))
        self._next_step = step_id + 1
        existing = self._steps.setdefault(step_id, {})
        existing.update(row)

    # -- Close (orchestrator) ---------------------------------------------

    def close(
        self,
        *,
        status: EpisodeStatus,
        metrics: dict[str, Any],
        task_name: str = "",
        episode_id: int = 0,
        steps: int = 0,
        elapsed_sec: float = 0.0,
        failure_reason: str | None = None,
        failure_detail: str | None = None,
    ) -> None:
        if self._closed:
            return
        self._closed = True

        if self._video is not None:
            try:
                self._video.save(status=status)
            except FileExistsError as exc:
                logger.warning("Episode video already exists: %s", exc)
            except Exception:
                logger.exception("video.save failed for sid=%s eid=%s", self._sid, self._eid)
            self._video = None

        try:
            jsonl_name = (self._filename_stem + ".jsonl").format(status=status, **self._context)
        except Exception:
            logger.exception("filename_stem render failed; using fallback name")
            jsonl_name = f"{self._sid}-{self._eid}_{status}.jsonl"
        jsonl_path = str(self._output_dir / jsonl_name)

        try:
            self._store.upsert_step_rows(self._sid, self._eid, self._steps)
        except Exception:
            logger.exception("Failed to upsert step rows for sid=%s eid=%s", self._sid, self._eid)

        try:
            self._store.upsert_episode_result(
                sid=self._sid,
                eid=self._eid,
                eval_id=self._eval_id,
                task_name=task_name,
                episode_id=episode_id,
                status=status,
                metrics=metrics,
                steps=steps,
                elapsed_sec=elapsed_sec,
                context=self._context,
                jsonl_path=jsonl_path,
                failure_reason=failure_reason,
                failure_detail=failure_detail,
            )
        except Exception:
            logger.exception("Failed to upsert episode result for sid=%s eid=%s", self._sid, self._eid)


class NullEpisodeRecorder(EpisodeRecorder):
    """No-op recorder. Benchmarks call ``record_video`` / ``record_step``
    unconditionally; with a Null instance every call returns immediately.

    Used when the orchestrator runs without recording (``--no-save``) or
    when the benchmark opts out of recording (``get_recording_context``
    returns ``None``).
    """

    def __init__(self) -> None:  # type: ignore[override]
        # Skip parent setup entirely — no store, no video, no buffer.
        self._closed = True
        self._video = None
        self._steps = {}

    @property
    def is_active(self) -> bool:  # type: ignore[override]
        return False

    @property
    def sid(self) -> str:  # type: ignore[override]
        return ""

    @property
    def eid(self) -> str:  # type: ignore[override]
        return ""

    @property
    def eval_id(self) -> str:  # type: ignore[override]
        return ""

    @property
    def db_path(self) -> str:  # type: ignore[override]
        return ""

    def record_video(self, frame: "np.ndarray") -> None:  # type: ignore[override]
        pass

    def record_step(self, row: dict[str, Any]) -> None:  # type: ignore[override]
        pass

    def close(  # type: ignore[override]
        self,
        *,
        status: EpisodeStatus,
        metrics: dict[str, Any],
        task_name: str = "",
        episode_id: int = 0,
        steps: int = 0,
        elapsed_sec: float = 0.0,
        failure_reason: str | None = None,
        failure_detail: str | None = None,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# StepRecorder — lightweight external API (model server side)
# ---------------------------------------------------------------------------


class StepRecorder:
    """Per-episode step-row writer for external callers.

    Use case: model server code (e.g. reflex-train) wants to record per-step
    inference traces alongside the benchmark's per-step records. The harness
    forwards ``(sid, eid, eval_id, db_path)`` in the ``EPISODE_START`` WS
    payload; the model server opens a :class:`StepRecorder`, accumulates
    rows, and ``close()`` flushes them to the same SQLite the harness owns.

    Field-union semantics: if the model server records
    ``{"inference_ms": 12.3, "action_logits": [...]}`` for ``step_id=42``
    and the benchmark records ``{"reward": 0.5, "robot_state": [...]}`` for
    the same step, the final row contains all four fields. ``json_patch``
    UPSERT inside the store handles this atomically regardless of which
    process commits first.
    """

    def __init__(self, db_path: str | Path, sid: str, eid: str) -> None:
        self._store = RecordingStore(db_path)
        self._sid = sid
        self._eid = eid
        self._steps: dict[int, dict[str, Any]] = {}
        self._next_step = 0
        self._closed = False

    def record(self, row: dict[str, Any]) -> None:
        step_id = int(row.pop("step", self._next_step))
        self._next_step = step_id + 1
        existing = self._steps.setdefault(step_id, {})
        existing.update(row)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._store.upsert_step_rows(self._sid, self._eid, self._steps)
        except Exception:
            logger.exception("StepRecorder: failed to upsert step rows for sid=%s eid=%s", self._sid, self._eid)
        self._store.close()

    def __enter__(self) -> "StepRecorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
