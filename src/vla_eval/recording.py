"""Unified recording API for surfacing debug fields into the per-step jsonl.

The same call site works on both ends of the WebSocket:

    from vla_eval import recording
    recording.record(ctx, foo=1, bar="x")

On the model-server side a sender bound for ``ctx.session_id`` ships the
fields to the harness via a ``RECORD`` message.  On the harness side
``record`` writes directly into a per-session buffer that the
orchestrator drains at step boundary into the same jsonl row the
benchmark already produces.  ``ctx`` must expose ``.session_id`` so
batched server RPCs can keep sessions separate.

The wire backend and the in-process backend are co-located on purpose:
the public surface (``record`` / ``drain`` / ``register_sender`` /
``ingest``) is the seam at which a future stand-alone recording daemon
could replace the wire transport without touching call sites.
"""

from __future__ import annotations

import logging
import threading
from contextvars import ContextVar
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

SenderFn = Callable[[str, dict[str, Any]], None]


class _HasSessionId(Protocol):
    @property
    def session_id(self) -> str: ...


class Recording:
    """Per-session buffer + sender registry shared by the harness and the server.

    Thread-safe; a single instance is exposed at module scope.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffers: dict[str, dict[str, Any]] = {}
        self._senders: dict[str, SenderFn] = {}
        self._active_session: ContextVar[str | None] = ContextVar("active_session", default=None)

    def register_sender(self, session_id: str, sender: SenderFn) -> None:
        """Route ``record`` calls for *session_id* over the wire."""
        with self._lock:
            self._senders[session_id] = sender

    def unregister_sender(self, session_id: str) -> None:
        with self._lock:
            self._senders.pop(session_id, None)

    def record(self, ctx: _HasSessionId, **fields: Any) -> None:
        if not fields:
            return
        session_id = ctx.session_id
        with self._lock:
            sender = self._senders.get(session_id)
        if sender is not None:
            try:
                sender(session_id, dict(fields))
            except Exception:
                logger.exception("recording sender failed; dropping fields for session=%s", session_id[:8])
            return
        self._append(session_id, fields)

    def ingest(self, session_id: str, fields: dict[str, Any]) -> None:
        """Merge fields that arrived over the wire into the local buffer."""
        self._append(session_id, fields)

    def drain(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return self._buffers.pop(session_id, {})

    def drain_current(self) -> dict[str, Any]:
        sid = self._active_session.get()
        if sid is None:
            return {}
        return self.drain(sid)

    def current_session(self) -> str | None:
        return self._active_session.get()

    def set_active_session(self, session_id: str | None) -> Any:
        return self._active_session.set(session_id)

    def reset_active_session(self, token: Any) -> None:
        self._active_session.reset(token)

    def clear(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._buffers.clear()
            else:
                self._buffers.pop(session_id, None)

    def _append(self, session_id: str, fields: dict[str, Any]) -> None:
        with self._lock:
            bucket = self._buffers.setdefault(session_id, {})
            bucket.update(fields)


_recording = Recording()


def record(ctx: _HasSessionId, **fields: Any) -> None:
    _recording.record(ctx, **fields)


def ingest(session_id: str, fields: dict[str, Any]) -> None:
    _recording.ingest(session_id, fields)


def drain(session_id: str) -> dict[str, Any]:
    return _recording.drain(session_id)


def drain_current() -> dict[str, Any]:
    return _recording.drain_current()


def current_session() -> str | None:
    return _recording.current_session()


def register_sender(session_id: str, sender: SenderFn) -> None:
    _recording.register_sender(session_id, sender)


def unregister_sender(session_id: str) -> None:
    _recording.unregister_sender(session_id)


def set_active_session(session_id: str | None) -> Any:
    return _recording.set_active_session(session_id)


def reset_active_session(token: Any) -> None:
    _recording.reset_active_session(token)


def clear(session_id: str | None = None) -> None:
    _recording.clear(session_id)


__all__ = [
    "Recording",
    "SenderFn",
    "clear",
    "current_session",
    "drain",
    "drain_current",
    "ingest",
    "record",
    "register_sender",
    "reset_active_session",
    "set_active_session",
    "unregister_sender",
]
