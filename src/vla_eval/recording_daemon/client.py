"""WebSocket client that pushes recording frames to the daemon.

Single connection per process, fire-and-forget from the caller's POV. A
background thread runs an asyncio loop that drains a bounded queue,
dialing the daemon with exponential backoff and dropping frames on
queue overflow. Recording must never stall the predict / step loops.

This client speaks the 3-message recording protocol (``STEP`` /
``RESULT`` / ``EVAL_END``) — it has no notion of video, no notion of
filenames. Video lives entirely on the harness side (see
:class:`vla_eval.benchmarks.video.EpisodeVideoRecorder`); the daemon
never sees binary frames.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

import websockets

from vla_eval.recording_daemon.messages import RecMessageType, RecordingFrame, pack_frame

logger = logging.getLogger(__name__)

# Bounded queue: when a slow daemon stalls the WS write, the emitter drops
# rather than backing up the caller. Sized for ~30 s of episode chatter at
# 50 Hz per shard; one drop is logged per session.
_DEFAULT_QUEUE_SIZE = 8192

# Backoff bounded to a small cap because gaps in recording are worse than
# tight retry — the emitter is in-process to the predict loop.
_RECONNECT_INITIAL_SEC = 0.1
_RECONNECT_MAX_SEC = 5.0


class RecordingClient:
    """Thread-safe client that sends recording frames to the daemon over WS."""

    def __init__(self, daemon_url: str, queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._url = daemon_url
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=queue_size)
        self._drop_count = 0
        self._warned_overflow = False
        self._warned_unreachable = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False
        self._start()

    # ------------------------------------------------------------------
    # Public push API (3 message types only)
    # ------------------------------------------------------------------

    def step(self, sid: str, eid: str, step_id: int, fields: dict[str, Any]) -> None:
        """Push one step row. Daemon buffers under ``(sid, eid)``."""
        self._enqueue(
            RecMessageType.STEP,
            {"sid": sid, "eid": eid, "step_id": step_id, "fields": fields},
        )

    def result(
        self,
        *,
        eval_id: str,
        sid: str,
        eid: str,
        status: str,
        metrics: dict[str, Any],
        context: dict[str, Any],
        jsonl_path: str,
        aggregate_dir: str,
        aggregate_filename: str,
    ) -> None:
        """Push episode result. Daemon writes the bucket's rows to ``jsonl_path``
        and appends ``{sid, eid, status, metrics, **context}`` to the eval aggregate.
        ``aggregate_dir``/``aggregate_filename`` tell the daemon where ``EVAL_END``
        will eventually flush the aggregate JSON.
        """
        self._enqueue(
            RecMessageType.RESULT,
            {
                "eval_id": eval_id,
                "sid": sid,
                "eid": eid,
                "status": status,
                "metrics": metrics,
                "context": context,
                "jsonl_path": jsonl_path,
                "aggregate_dir": aggregate_dir,
                "aggregate_filename": aggregate_filename,
            },
        )

    def eval_end(self, eval_id: str) -> None:
        """Push end-of-eval. Daemon flushes the aggregate JSON for this eval."""
        self._enqueue(RecMessageType.EVAL_END, {"eval_id": eval_id})

    def close(self, timeout: float = 5.0) -> None:
        """Flush queued frames, close the WS connection, stop the thread."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _enqueue(self, msg_type: RecMessageType, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        frame = RecordingFrame(type=msg_type, payload=payload)
        try:
            data = pack_frame(frame)
        except Exception:
            logger.exception("Failed to pack recording frame type=%s", msg_type.value)
            return
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            self._drop_count += 1
            if not self._warned_overflow:
                self._warned_overflow = True
                logger.warning(
                    "RecordingClient queue full at %d frames; dropping. Further drops silent this session.",
                    self._queue.maxsize,
                )

    def _start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, name="recording-emitter", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._sender_loop())
        except Exception:
            logger.exception("RecordingClient sender loop crashed")
        finally:
            if self._loop is not None:
                self._loop.close()

    async def _sender_loop(self) -> None:
        backoff = _RECONNECT_INITIAL_SEC
        while True:
            try:
                ws = await websockets.connect(self._url, compression=None, max_size=None, ping_interval=None)
            except Exception as exc:
                if not self._warned_unreachable:
                    self._warned_unreachable = True
                    logger.warning(
                        "Recording daemon at %s unreachable (%s); retrying. Frames will be queued.",
                        self._url,
                        type(exc).__name__,
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX_SEC)
                continue

            self._warned_unreachable = False
            backoff = _RECONNECT_INITIAL_SEC
            try:
                await self._drain_into(ws)
                # Sentinel reached: graceful shutdown.
                await ws.close()
                return
            except (
                websockets.exceptions.ConnectionClosed,
                ConnectionError,
                OSError,
            ) as exc:
                logger.warning("Recording daemon connection lost (%s); reconnecting.", type(exc).__name__)
                continue

    async def _drain_into(self, ws: Any) -> None:
        """Pop frames from the cross-thread queue, send each over ws.

        Returns normally when the sentinel ``None`` is dequeued; raises on
        connection errors so the outer loop can reconnect.
        """
        loop = asyncio.get_running_loop()
        while True:
            data = await loop.run_in_executor(None, self._queue.get)
            if data is None:
                return
            await ws.send(data)
