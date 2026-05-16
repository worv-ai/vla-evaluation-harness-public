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
the public surface (``record`` / ``session_scope`` / ``ingest`` /
``register_sender``) is the seam at which a future stand-alone recording
daemon could replace the wire transport without touching call sites.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Protocol

logger = logging.getLogger(__name__)

SenderFn = Callable[[str, dict[str, Any]], None]


class _HasSessionId(Protocol):
    @property
    def session_id(self) -> str | None: ...


class Recording:
    """Per-session buffer + sender registry shared by the harness and the server.

    Thread-safe; a single instance is exposed at module scope via
    :func:`get_default`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buffers: dict[str, dict[str, Any]] = {}
        self._senders: dict[str, SenderFn] = {}
        self._active_session: ContextVar[str | None] = ContextVar("active_session", default=None)
        self._ever_registered_sender = False

    def register_sender(self, session_id: str, sender: SenderFn) -> None:
        """Route ``record`` calls for *session_id* over the wire."""
        with self._lock:
            self._senders[session_id] = sender
            self._ever_registered_sender = True

    def unregister_sender(self, session_id: str) -> None:
        with self._lock:
            self._senders.pop(session_id, None)

    def record(self, ctx: _HasSessionId, **fields: Any) -> None:
        if not fields:
            return
        session_id = ctx.session_id
        if not session_id:
            logger.warning("Dropped recording fields: no session_id in context")
            return
        with self._lock:
            ever_registered = self._ever_registered_sender
            sender = self._senders.get(session_id)
        if not ever_registered:
            self._append(session_id, fields)
            return
        if sender is None:
            logger.warning("record(): no sender registered for session=%s; fields dropped", session_id[:8])
            return
        try:
            sender(session_id, fields)
        except Exception:
            logger.exception("recording sender failed; dropping fields for session=%s", session_id[:8])

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

    @contextmanager
    def session_scope(self, session_id: str | None) -> Iterator[None]:
        """Bind *session_id* as the active session for the duration of the block.

        Clears the per-session buffer on entry and exit so the runner sees a
        fresh slate per episode.  ``async with`` is supported via
        ``contextlib.asynccontextmanager``-equivalent semantics: the generator
        yields once and the body of ``with`` / ``async with`` runs around it.
        ``session_id=None`` is a no-op scope (used when the connection has no
        session yet).
        """
        if not session_id:
            yield
            return
        token = self.set_active_session(session_id)
        self.clear(session_id)
        try:
            yield
        finally:
            self.reset_active_session(token)
            self.clear(session_id)

    def _append(self, session_id: str, fields: dict[str, Any]) -> None:
        with self._lock:
            bucket = self._buffers.setdefault(session_id, {})
            bucket.update(fields)


_default = Recording()


def get_default() -> Recording:
    """Return the process-wide default :class:`Recording` instance.

    Framework internals (runners, :class:`EpisodeRecorder`) need direct
    instance access; user code should reach for the module-level helpers.
    """
    return _default


def record(ctx: _HasSessionId, **fields: Any) -> None:
    _default.record(ctx, **fields)


def ingest(session_id: str, fields: dict[str, Any]) -> None:
    _default.ingest(session_id, fields)


def register_sender(session_id: str, sender: SenderFn) -> None:
    _default.register_sender(session_id, sender)


def unregister_sender(session_id: str) -> None:
    _default.unregister_sender(session_id)


def session_scope(session_id: str | None) -> Any:
    return _default.session_scope(session_id)


__all__ = [
    "Recording",
    "SenderFn",
    "get_default",
    "ingest",
    "record",
    "register_sender",
    "session_scope",
    "unregister_sender",
]
