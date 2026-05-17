"""Recording daemon — collects per-episode artefacts via WebSocket.

One process per host; many emitters fan in.  Lifecycle is documented in
``plans/continue-fuzzy-parasol.md``.  Failure modes (age-out, missing
EPISODE_END, missing EVAL_END, video rename failure) are decided behaviors
listed there; this module implements them as called out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import websockets

from vla_eval.recording_daemon.config import RecordingDaemonConfig
from vla_eval.recording_daemon.health import HealthServer
from vla_eval.recording_daemon.messages import RecordingFrame, RecordingMessageType, unpack_frame

logger = logging.getLogger(__name__)

_AGE_OUT_INTERVAL_SEC = 15.0


@dataclass
class _EpisodeBucket:
    sid: str
    eid: str
    eval_id: str | None
    output_dir: Path
    filename_template: str
    context: dict[str, Any]
    rows: dict[int, dict[str, Any]] = field(default_factory=dict)
    status: str | None = None
    metrics: dict[str, Any] | None = None
    video_working_path: Path | None = None
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


@dataclass
class _EvalAggregate:
    eval_id: str
    output_dir: Path
    aggregate_filename: str
    expected_count: int | None
    result_rows: list[dict[str, Any]] = field(default_factory=list)
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class RecordingDaemon:
    """Async daemon: WS server + bucket store + atomic disk writer."""

    def __init__(self, config: RecordingDaemonConfig) -> None:
        self.config = config
        self.out_dir = Path(config.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._buckets: dict[tuple[str, str], _EpisodeBucket] = {}
        self._evals: dict[str, _EvalAggregate] = {}
        self._lock = asyncio.Lock()
        self._stop_event: asyncio.Event | None = None
        self._started_at = time.monotonic()
        self._health = HealthServer(self, config.health_port)

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    @property
    def eval_count(self) -> int:
        return len(self._evals)

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._started_at

    async def start(self) -> None:
        """Run the daemon until :meth:`stop` is called or SIGTERM/SIGINT arrives."""
        self._stop_event = asyncio.Event()
        host, port = _parse_bind(self.config.bind)
        self._health.start()
        age_task = asyncio.create_task(self._age_out_loop())
        try:
            async with websockets.serve(
                self._handle_connection,
                host,
                port,
                compression=None,
                max_size=None,
                ping_interval=None,
            ):
                logger.info("RecordingDaemon listening on %s (out=%s)", self.config.bind, self.out_dir)
                await self._stop_event.wait()
        finally:
            age_task.cancel()
            try:
                await age_task
            except (asyncio.CancelledError, Exception):
                pass
            await self._flush_all_on_shutdown()
            self._health.stop()

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    async def _handle_connection(self, ws: Any) -> None:
        try:
            async for raw in ws:
                try:
                    frame = unpack_frame(raw)
                except ValueError as exc:
                    logger.warning("Discarding malformed recording frame: %s", exc)
                    continue
                await self._dispatch(frame)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.exception("Recording emitter handler crashed")

    async def _dispatch(self, frame: RecordingFrame) -> None:
        handler = _DISPATCH.get(frame.type)
        if handler is None:
            logger.warning("Unknown recording frame type: %s", frame.type)
            return
        async with self._lock:
            await handler(self, frame.payload)

    # ------------------------------------------------------------------
    # Per-type handlers
    # ------------------------------------------------------------------

    async def _on_eval_start(self, payload: dict[str, Any]) -> None:
        eval_id = payload["eval_id"]
        if eval_id in self._evals:
            logger.warning("EVAL_START for already-known eval_id=%s; ignoring", eval_id)
            self._evals[eval_id].touch()
            return
        self._evals[eval_id] = _EvalAggregate(
            eval_id=eval_id,
            output_dir=Path(payload["output_dir"]),
            aggregate_filename=payload["aggregate_filename"],
            expected_count=payload.get("expected_count"),
        )

    async def _on_episode_start(self, payload: dict[str, Any]) -> None:
        sid, eid = payload["sid"], payload["eid"]
        key = (sid, eid)
        bucket = self._buckets.get(key)
        if bucket is None:
            self._buckets[key] = _EpisodeBucket(
                sid=sid,
                eid=eid,
                eval_id=payload.get("eval_id"),
                output_dir=Path(payload["output_dir"]),
                filename_template=payload["filename_template"],
                context=dict(payload.get("context") or {}),
            )
        else:
            # Context-merge — second arrival adds fields without clobbering.
            for k, v in (payload.get("context") or {}).items():
                bucket.context.setdefault(k, v)
            if bucket.eval_id is None and payload.get("eval_id"):
                bucket.eval_id = payload["eval_id"]
            bucket.touch()

    async def _on_record_commit(self, payload: dict[str, Any]) -> None:
        key = (payload["sid"], payload["eid"])
        bucket = self._buckets.get(key)
        if bucket is None:
            logger.warning("RECORD_COMMIT for unknown bucket %s; dropping", key)
            return
        step_id = payload["step_id"]
        fields = payload.get("fields") or {}
        row = bucket.rows.setdefault(step_id, {"step": step_id})
        row.update(fields)
        bucket.touch()

    async def _on_episode_result(self, payload: dict[str, Any]) -> None:
        key = (payload["sid"], payload["eid"])
        bucket = self._buckets.get(key)
        if bucket is None:
            logger.warning("EPISODE_RESULT for unknown bucket %s; dropping", key)
            return
        bucket.status = payload.get("status")
        bucket.metrics = dict(payload.get("metrics") or {})
        bucket.touch()
        eval_id = payload.get("eval_id") or bucket.eval_id
        if eval_id and eval_id in self._evals:
            self._evals[eval_id].result_rows.append(
                {
                    "sid": bucket.sid,
                    "eid": bucket.eid,
                    "status": bucket.status,
                    "metrics": bucket.metrics,
                    **bucket.context,
                }
            )
            self._evals[eval_id].touch()

    async def _on_video_artifact(self, payload: dict[str, Any]) -> None:
        key = (payload["sid"], payload["eid"])
        bucket = self._buckets.get(key)
        if bucket is None:
            logger.warning("VIDEO_ARTIFACT for unknown bucket %s; dropping", key)
            return
        bucket.video_working_path = Path(payload["working_path"])
        bucket.touch()

    async def _on_episode_end(self, payload: dict[str, Any]) -> None:
        key = (payload["sid"], payload["eid"])
        bucket = self._buckets.pop(key, None)
        if bucket is None:
            return  # second EPISODE_END (first-wins); silent.
        self._flush_bucket(bucket, unclosed=False)

    async def _on_eval_end(self, payload: dict[str, Any]) -> None:
        eval_id = payload["eval_id"]
        agg = self._evals.pop(eval_id, None)
        if agg is None:
            logger.warning("EVAL_END for unknown eval_id=%s", eval_id)
            return
        self._flush_eval(agg, unclosed=False)

    # ------------------------------------------------------------------
    # Disk writers (atomic via tmp + os.replace; same-fs only)
    # ------------------------------------------------------------------

    def _flush_bucket(self, bucket: _EpisodeBucket, *, unclosed: bool) -> None:
        try:
            bucket.output_dir.mkdir(parents=True, exist_ok=True)
            status = bucket.status or ("unknown" if not unclosed else "unknown")
            stem_ctx = {**bucket.context, "status": status}
            try:
                stem = bucket.filename_template.format(**stem_ctx)
            except KeyError as exc:
                logger.warning("Filename template missing key %s for bucket %s/%s", exc, bucket.sid, bucket.eid)
                return
            suffix = "_unclosed" if unclosed else ""
            jsonl_path = bucket.output_dir / f"{stem}{suffix}.jsonl"
            self._write_jsonl_atomic(jsonl_path, bucket.rows)
            logger.info("Flushed episode jsonl: %s (%d rows)", jsonl_path, len(bucket.rows))
            if bucket.video_working_path is not None:
                mp4_path = bucket.output_dir / f"{stem}{suffix}.mp4"
                try:
                    if mp4_path.exists():
                        mp4_path.unlink()
                    os.rename(str(bucket.video_working_path), str(mp4_path))
                    logger.info("Renamed video into place: %s", mp4_path)
                except OSError as exc:
                    logger.warning(
                        "Video rename failed (%s); mp4 dropped, jsonl still written: %s",
                        exc,
                        bucket.video_working_path,
                    )
        except Exception:
            logger.exception("Failed to flush bucket %s/%s", bucket.sid, bucket.eid)

    def _flush_eval(self, agg: _EvalAggregate, *, unclosed: bool) -> None:
        try:
            agg.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = "_unclosed" if unclosed else ""
            stem, _, ext = agg.aggregate_filename.rpartition(".")
            if not stem:
                stem, ext = agg.aggregate_filename, "json"
            final = agg.output_dir / f"{stem}{suffix}.{ext}"
            payload = {
                "eval_id": agg.eval_id,
                "expected_count": agg.expected_count,
                "received_count": len(agg.result_rows),
                "results": agg.result_rows,
                "unclosed": unclosed,
            }
            self._write_json_atomic(final, payload)
            logger.info("Flushed eval aggregate: %s (%d rows)", final, len(agg.result_rows))
        except Exception:
            logger.exception("Failed to flush eval %s", agg.eval_id)

    @staticmethod
    def _write_jsonl_atomic(path: Path, rows: dict[int, dict[str, Any]]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for step_id in sorted(rows):
                f.write(json.dumps(rows[step_id], default=str) + "\n")
        os.replace(str(tmp), str(path))

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        os.replace(str(tmp), str(path))

    # ------------------------------------------------------------------
    # Age-out + shutdown
    # ------------------------------------------------------------------

    async def _age_out_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_AGE_OUT_INTERVAL_SEC)
                await self._age_out_once()
        except asyncio.CancelledError:
            raise

    async def _age_out_once(self) -> None:
        now = time.monotonic()
        timeout = self.config.idle_timeout_sec
        async with self._lock:
            stale_buckets = [k for k, b in self._buckets.items() if (now - b.last_activity) > timeout]
            for key in stale_buckets:
                bucket = self._buckets.pop(key)
                logger.warning("Bucket %s/%s aged out — flushing with _unclosed suffix", bucket.sid, bucket.eid)
                self._flush_bucket(bucket, unclosed=True)
            stale_evals = [k for k, e in self._evals.items() if (now - e.last_activity) > timeout]
            for eid in stale_evals:
                agg = self._evals.pop(eid)
                logger.warning("Eval %s aged out — flushing partial aggregate with _unclosed suffix", agg.eval_id)
                self._flush_eval(agg, unclosed=True)

    async def _flush_all_on_shutdown(self) -> None:
        async with self._lock:
            for key in list(self._buckets):
                bucket = self._buckets.pop(key)
                logger.warning("Shutdown flush: bucket %s/%s -> _unclosed", bucket.sid, bucket.eid)
                self._flush_bucket(bucket, unclosed=True)
            for eid in list(self._evals):
                agg = self._evals.pop(eid)
                logger.warning("Shutdown flush: eval %s -> _unclosed", agg.eval_id)
                self._flush_eval(agg, unclosed=True)


_DISPATCH = {
    RecordingMessageType.EVAL_START: RecordingDaemon._on_eval_start,
    RecordingMessageType.EPISODE_START: RecordingDaemon._on_episode_start,
    RecordingMessageType.RECORD_COMMIT: RecordingDaemon._on_record_commit,
    RecordingMessageType.EPISODE_RESULT: RecordingDaemon._on_episode_result,
    RecordingMessageType.VIDEO_ARTIFACT: RecordingDaemon._on_video_artifact,
    RecordingMessageType.EPISODE_END: RecordingDaemon._on_episode_end,
    RecordingMessageType.EVAL_END: RecordingDaemon._on_eval_end,
}


def _parse_bind(bind: str) -> tuple[str, int]:
    """Accept ``ws://host:port`` or ``tcp://host:port`` or bare ``host:port``."""
    if "://" in bind:
        parsed = urlparse(bind)
        host = parsed.hostname or "0.0.0.0"
        port = parsed.port or 9001
        return host, port
    host, _, port_s = bind.partition(":")
    return host or "0.0.0.0", int(port_s or "9001")
