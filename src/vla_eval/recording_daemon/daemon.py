"""Recording daemon — collects per-episode step rows + per-eval aggregate.

One process per host; many emitters fan in over WebSocket. The daemon
speaks the 3-message recording protocol (see ``messages.py``):

- ``STEP``: implicitly creates the ``(sid, eid)`` bucket on first arrival
  and accumulates step rows in memory.
- ``RESULT``: flushes the bucket to ``jsonl_path`` and appends a row to
  the eval-level aggregate. Implicitly creates the eval on first arrival.
- ``EVAL_END``: flushes the aggregate JSON to
  ``aggregate_dir/aggregate_filename``.

The daemon does no filename rendering and no status injection. Paths are
carried on the wire by the orchestrator.

Failure modes:

- *Age-out* (idle longer than ``idle_timeout_sec``): buckets/evals flush
  with an ``_unclosed`` suffix and a warning. Safety net — not the happy
  path.
- *Daemon shutdown* (SIGTERM/SIGINT): every live bucket/eval flushes
  with ``_unclosed`` before the process exits.
- *Disk write failure*: logged at ``ERROR``; the daemon keeps running.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import websockets

from vla_eval.recording_daemon.config import RecordingDaemonConfig
from vla_eval.recording_daemon.messages import RecMessageType, RecordingFrame, unpack_frame
from vla_eval.results.collector import _aggregate_metrics, _build_task_result, _extract_seed

logger = logging.getLogger(__name__)

_AGE_OUT_INTERVAL_SEC = 15.0


@dataclass
class _Bucket:
    sid: str
    eid: str
    rows: dict[int, dict[str, Any]] = field(default_factory=dict)
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


@dataclass
class _Eval:
    eval_id: str
    aggregate_dir: Path
    aggregate_filename: str
    # task_name → list of EpisodeResult-shaped dicts (the same shape ResultCollector
    # builds in-memory, so we can reuse collector helpers to compute aggregates).
    tasks: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Filled in on first RESULT for this eval_id; subsequent RESULTs ignore it
    # (all shards send the same benchmark-level metadata).
    metadata: dict[str, Any] = field(default_factory=dict)
    last_activity: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class RecordingDaemon:
    """Async daemon: WS server + (sid, eid) step buffer + eval aggregator.

    No lock: the asyncio event loop is single-threaded and the dispatch is
    serial, so concurrent mutation of internal dicts cannot occur. All
    disk writes are dispatched to a thread so the loop is never blocked.
    """

    def __init__(self, config: RecordingDaemonConfig) -> None:
        self.config = config
        self.out_dir = Path(config.out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._evals: dict[str, _Eval] = {}
        self._stop_event: asyncio.Event | None = None
        self._started_at = time.monotonic()
        self._dispatch_table = {
            RecMessageType.STEP: self._on_step,
            RecMessageType.RESULT: self._on_result,
            RecMessageType.EVAL_END: self._on_eval_end,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Run the daemon until :meth:`stop` is called or SIGTERM/SIGINT arrives."""
        self._stop_event = asyncio.Event()
        host, port = _parse_bind(self.config.bind)
        age_task = asyncio.create_task(self._age_out_loop())
        try:
            async with websockets.serve(
                self._handle_connection,
                host,
                port,
                process_request=self._make_process_request(),
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

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    # ------------------------------------------------------------------
    # Properties (read by /health)
    # ------------------------------------------------------------------

    @property
    def bucket_count(self) -> int:
        return len(self._buckets)

    @property
    def eval_count(self) -> int:
        return len(self._evals)

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._started_at

    # ------------------------------------------------------------------
    # WS server
    # ------------------------------------------------------------------

    def _make_process_request(self) -> Any:
        """``GET /health`` is served on the same port as the WS endpoint.

        Mirrors :func:`vla_eval.model_servers.serve._make_process_request` —
        a small ``process_request`` callback that short-circuits the WS
        handshake for HTTP control-plane requests.
        """

        def process_request(connection: Any, request: Any) -> Any:
            if urlparse(request.path).path != "/health":
                return None  # proceed with WS handshake
            body = json.dumps(
                {
                    "buckets": self.bucket_count,
                    "evals": self.eval_count,
                    "uptime_s": round(self.uptime_sec, 2),
                }
            )
            return connection.respond(HTTPStatus.OK, body + "\n")

        return process_request

    async def _handle_connection(self, ws: Any) -> None:
        try:
            async for raw in ws:
                # websockets yields either bytes (binary frames) or str (text);
                # our wire format is msgpack binary only, so guard against an
                # accidentally text-encoded frame from a misconfigured client.
                if not isinstance(raw, (bytes, bytearray)):
                    logger.warning("Discarding non-binary recording frame (%s)", type(raw).__name__)
                    continue
                try:
                    frame = unpack_frame(bytes(raw))
                except ValueError as exc:
                    logger.warning("Discarding malformed recording frame: %s", exc)
                    continue
                await self._dispatch(frame)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            logger.exception("Recording emitter handler crashed")

    async def _dispatch(self, frame: RecordingFrame) -> None:
        handler = self._dispatch_table.get(frame.type)
        if handler is None:  # pragma: no cover — enum exhaustive
            logger.warning("Unknown recording frame type: %s", frame.type)
            return
        result = handler(frame.payload)
        if asyncio.iscoroutine(result):
            await result

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_step(self, payload: dict[str, Any]) -> None:
        sid, eid = payload["sid"], payload["eid"]
        key = (sid, eid)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(sid=sid, eid=eid)
            self._buckets[key] = bucket
        step_id = int(payload["step_id"])
        fields = payload.get("fields") or {}
        row = bucket.rows.setdefault(step_id, {"step": step_id})
        row.update(fields)
        bucket.touch()

    async def _on_result(self, payload: dict[str, Any]) -> None:
        sid, eid = payload["sid"], payload["eid"]
        eval_id = payload["eval_id"]
        bucket = self._buckets.pop((sid, eid), None)

        # Skip the jsonl write when no step rows accumulated. Two cases:
        #   - duplicate RESULT for an already-flushed episode (defensive — the
        #     happy path's single orchestrator owner shouldn't produce this)
        #   - the episode failed before any STEP arrived (e.g. benchmark.reset
        #     raised; orchestrator pushed RESULT with status="error"). The
        #     aggregate row still captures the failure; an empty jsonl is just
        #     noise on disk and could clobber a real file from a duplicate.
        if bucket is None or not bucket.rows:
            if bucket is None:
                logger.warning("RESULT for unknown bucket sid=%s eid=%s; skipping jsonl write", sid, eid)
        else:
            jsonl_path = Path(payload["jsonl_path"])
            try:
                await asyncio.to_thread(_write_jsonl_atomic, jsonl_path, dict(bucket.rows))
                logger.info("Flushed episode jsonl: %s (%d rows)", jsonl_path, len(bucket.rows))
            except Exception:
                logger.exception("Failed to write episode jsonl: %s", jsonl_path)

        agg = self._evals.get(eval_id)
        if agg is None:
            agg = _Eval(
                eval_id=eval_id,
                aggregate_dir=Path(payload["aggregate_dir"]),
                aggregate_filename=payload["aggregate_filename"],
                metadata=dict(payload.get("bench_metadata") or {}),
            )
            self._evals[eval_id] = agg
        elif not agg.metadata:
            # First arrival had no metadata (e.g. an external caller); accept
            # whatever this RESULT carries as the canonical metadata.
            agg.metadata = dict(payload.get("bench_metadata") or {})

        # Build the EpisodeResult shape that ResultCollector / _build_task_result
        # expect. Daemon-only context fields (sid, eid) are preserved alongside
        # so per-episode rows remain identifiable.
        episode_row: dict[str, Any] = {
            "sid": sid,
            "eid": eid,
            "episode_id": payload.get("episode_id", 0),
            "metrics": dict(payload.get("metrics") or {}),
            "steps": payload.get("steps", 0),
            "elapsed_sec": payload.get("elapsed_sec", 0.0),
            **(payload.get("context") or {}),
        }
        if payload.get("failure_reason"):
            episode_row["failure_reason"] = payload["failure_reason"]
        if payload.get("failure_detail"):
            episode_row["failure_detail"] = payload["failure_detail"]

        task_name = str(payload.get("task_name") or "_unknown")
        agg.tasks.setdefault(task_name, []).append(episode_row)
        agg.touch()

    async def _on_eval_end(self, payload: dict[str, Any]) -> None:
        eval_id = payload["eval_id"]
        agg = self._evals.pop(eval_id, None)
        if agg is None:
            logger.warning("EVAL_END for unknown eval_id=%s; nothing to flush", eval_id)
            return
        await self._flush_eval(agg, unclosed=False)

    # ------------------------------------------------------------------
    # Disk writers
    # ------------------------------------------------------------------

    async def _flush_eval(self, agg: _Eval, *, unclosed: bool) -> None:
        try:
            stem, _, ext = agg.aggregate_filename.rpartition(".")
            if not stem:
                stem, ext = agg.aggregate_filename, "json"
            suffix = "_unclosed" if unclosed else ""
            final = agg.aggregate_dir / f"{stem}{suffix}.{ext}"
            body = self._build_aggregate(agg, unclosed=unclosed)
            await asyncio.to_thread(_write_json_atomic, final, body)
            total = sum(len(eps) for eps in agg.tasks.values())
            logger.info("Flushed eval aggregate: %s (%d episodes)", final, total)
        except Exception:
            logger.exception("Failed to flush eval %s", agg.eval_id)

    @staticmethod
    def _build_aggregate(agg: _Eval, *, unclosed: bool) -> dict[str, Any]:
        """Construct the on-disk BenchmarkResult-shaped aggregate.

        Mirrors what :meth:`vla_eval.results.collector.ResultCollector.get_benchmark_result`
        produces in-memory — same shape so downstream tooling (leaderboard
        scripts, manual inspection) keys off the same fields whether the
        result came from the daemon or from a `vla-eval run --no-save`
        in-memory return value.
        """
        metric_keys = dict(agg.metadata.get("metric_keys") or {})
        all_episodes: list[dict[str, Any]] = []
        tasks_out: list[Any] = []  # TaskResult is a TypedDict; widen for json
        for task_name in sorted(agg.tasks):
            episodes = agg.tasks[task_name]
            all_episodes.extend(episodes)
            tasks_out.append(_build_task_result(task_name, episodes, metric_keys))

        body: dict[str, Any] = {
            "benchmark": agg.metadata.get("benchmark", agg.eval_id),
            "mode": agg.metadata.get("mode"),
            "harness_version": agg.metadata.get("harness_version"),
            "tasks": tasks_out,
            "config": agg.metadata.get("config") or {},
            "eval_id": agg.eval_id,
        }
        if "server_info" in agg.metadata:
            body["server_info"] = agg.metadata["server_info"]
        seed = _extract_seed(body["config"])
        if seed is not None:
            body["seed"] = seed
        if metric_keys:
            body["metric_keys"] = metric_keys
            _aggregate_metrics(body, all_episodes, metric_keys)
        if unclosed:
            body["unclosed"] = True
        return body

    async def _flush_orphan_bucket(self, bucket: _Bucket) -> None:
        """Bucket that aged out without a RESULT — no jsonl_path on the wire.

        Falls back to ``out_dir/<sid>-<eid>_unclosed.jsonl`` so we at least
        keep the step rows for diagnosis.
        """
        path = self.out_dir / f"{bucket.sid}-{bucket.eid}_unclosed.jsonl"
        try:
            await asyncio.to_thread(_write_jsonl_atomic, path, bucket.rows)
            logger.warning(
                "Aged-out bucket %s/%s flushed to %s (%d rows)",
                bucket.sid,
                bucket.eid,
                path,
                len(bucket.rows),
            )
        except Exception:
            logger.exception("Failed to flush orphan bucket %s/%s", bucket.sid, bucket.eid)

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
        stale_buckets = [k for k, b in self._buckets.items() if (now - b.last_activity) > timeout]
        for key in stale_buckets:
            bucket = self._buckets.pop(key)
            await self._flush_orphan_bucket(bucket)
        stale_evals = [k for k, e in self._evals.items() if (now - e.last_activity) > timeout]
        for eid in stale_evals:
            agg = self._evals.pop(eid)
            logger.warning("Eval %s aged out — flushing partial aggregate with _unclosed suffix", eid)
            await self._flush_eval(agg, unclosed=True)

    async def _flush_all_on_shutdown(self) -> None:
        for key in list(self._buckets):
            bucket = self._buckets.pop(key)
            logger.warning("Shutdown flush: bucket %s/%s -> _unclosed", bucket.sid, bucket.eid)
            await self._flush_orphan_bucket(bucket)
        for eid in list(self._evals):
            agg = self._evals.pop(eid)
            logger.warning("Shutdown flush: eval %s -> _unclosed", agg.eval_id)
            await self._flush_eval(agg, unclosed=True)


# ---------------------------------------------------------------------------
# Atomic writers (run in thread)
# ---------------------------------------------------------------------------


def _write_jsonl_atomic(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for step_id in sorted(rows):
            f.write(json.dumps(rows[step_id], default=str) + "\n")
    os.replace(str(tmp), str(path))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _parse_bind(bind: str) -> tuple[str, int]:
    """Accept ``ws://host:port`` or ``tcp://host:port`` or bare ``host:port``."""
    if "://" in bind:
        parsed = urlparse(bind)
        host = parsed.hostname or "0.0.0.0"
        port = parsed.port or 9001
        return host, port
    host, _, port_s = bind.partition(":")
    return host or "0.0.0.0", int(port_s or "9001")
