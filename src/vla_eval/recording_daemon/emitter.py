"""WebSocket client that pushes recording frames to the daemon.

Single connection per process, fire-and-forget from the caller's POV.  A
background thread runs an asyncio loop that drains a bounded queue, dialing
the daemon with exponential backoff and dropping frames on queue overflow.
Recording must never stall the predict / step loops.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Any

import websockets

from vla_eval.recording_daemon.messages import RecordingFrame, RecordingMessageType, pack_frame

logger = logging.getLogger(__name__)

# Bounded queue: when a slow daemon stalls the WS write, the emitter drops
# rather than backing up the caller.  Sized for ~30 s of episode chatter at
# 50 Hz per shard; one drop is logged per session.
_DEFAULT_QUEUE_SIZE = 8192

# Backoff bounded to a small cap because gaps in recording are worse than
# tight retry — the emitter is in-process to the predict loop.
_RECONNECT_INITIAL_SEC = 0.1
_RECONNECT_MAX_SEC = 5.0


class RecordingEmitter:
    """Thread-safe emitter that sends frames to a recording daemon over WS."""

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
    # Public push API
    # ------------------------------------------------------------------

    def push_eval_start(
        self,
        eval_id: str,
        output_dir: str,
        aggregate_filename: str,
        expected_count: int | None = None,
    ) -> None:
        self._enqueue(
            RecordingMessageType.EVAL_START,
            {
                "eval_id": eval_id,
                "output_dir": output_dir,
                "aggregate_filename": aggregate_filename,
                "expected_count": expected_count,
            },
        )

    def push_episode_start(
        self,
        eval_id: str,
        sid: str,
        eid: str,
        output_dir: str,
        filename_template: str,
        context: dict[str, Any],
    ) -> None:
        self._enqueue(
            RecordingMessageType.EPISODE_START,
            {
                "eval_id": eval_id,
                "sid": sid,
                "eid": eid,
                "output_dir": output_dir,
                "filename_template": filename_template,
                "context": context,
            },
        )

    def push_record_commit(self, sid: str, eid: str, step_id: int, fields: dict[str, Any]) -> None:
        self._enqueue(
            RecordingMessageType.RECORD_COMMIT,
            {"sid": sid, "eid": eid, "step_id": step_id, "fields": fields},
        )

    def push_episode_result(self, eval_id: str, sid: str, eid: str, status: str, metrics: dict[str, Any]) -> None:
        self._enqueue(
            RecordingMessageType.EPISODE_RESULT,
            {"eval_id": eval_id, "sid": sid, "eid": eid, "status": status, "metrics": metrics},
        )

    def push_video_artifact(self, sid: str, eid: str, working_path: str) -> None:
        self._enqueue(
            RecordingMessageType.VIDEO_ARTIFACT,
            {"sid": sid, "eid": eid, "working_path": working_path},
        )

    def push_episode_end(self, sid: str, eid: str) -> None:
        self._enqueue(RecordingMessageType.EPISODE_END, {"sid": sid, "eid": eid})

    def push_eval_end(self, eval_id: str) -> None:
        self._enqueue(RecordingMessageType.EVAL_END, {"eval_id": eval_id})

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

    def _enqueue(self, msg_type: RecordingMessageType, payload: dict[str, Any]) -> None:
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
                    "RecordingEmitter queue full at %d frames; dropping. Further drops silent this session.",
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
            logger.exception("RecordingEmitter sender loop crashed")
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
                        "Recording daemon at %s unreachable (%s); retrying with backoff. Frames will be queued.",
                        self._url,
                        type(exc).__name__,
                    )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX_SEC)
                continue

            # Connected — reset warning state so a future drop re-logs once.
            self._warned_unreachable = False
            backoff = _RECONNECT_INITIAL_SEC
            try:
                await self._drain_into(ws)
                # Sentinel reached: graceful shutdown requested.
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
