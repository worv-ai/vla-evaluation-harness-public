"""Unified recording API with strong (sid, step_id) ordering guarantees.

The same call site works on both ends of the WebSocket:

    from vla_eval import recording

    with recording.step_scope(ctx) as step:
        step.record(foo=1, bar="x")
        step.record(baz=2)
    # __exit__: all accumulated fields ship as ONE RECORD_COMMIT frame
    # (or land in the local buffer if no wire sender is registered).

On the model-server side a sender bound for ``ctx.session_id`` ships the
committed fields to the harness as a single atomic ``RECORD_COMMIT``
message keyed by ``(session_id, step_id)``.  On the harness side the
orchestrator drains the matching ``(sid, step_id)`` bucket into the same
jsonl row the benchmark already produces.  ``ctx`` must expose
``.session_id`` and ``.step`` so batched server RPCs can keep sessions
and steps separate.

Lifecycle is enforced: ``record(ctx, ...)`` outside of a ``step_scope``
raises ``RuntimeError``.  Re-entering ``step_scope`` for the same
``(sid, step_id)`` raises ``RuntimeError``.  Exceptions inside the scope
still commit the fields collected so far, then propagate.

The wire backend and the in-process backend are co-located on purpose:
the public surface (``record`` / ``step_scope`` / ``session_scope`` /
``ingest`` / ``register_sender``) is the seam at which a future
stand-alone recording daemon could replace the wire transport without
touching call sites.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Protocol

logger = logging.getLogger(__name__)

# Sender callback ships a single atomic commit for one (session, step).
SenderFn = Callable[[str, int, dict[str, Any]], None]


class _HasSessionAndStep(Protocol):
    @property
    def session_id(self) -> str | None: ...
    @property
    def step(self) -> int: ...


class Step:
    """Handle returned by :meth:`Recording.step_scope`.

    Accumulates fields locally; the enclosing scope commits them atomically
    on ``__exit__``.  Calls outside the scope are no-ops on the discarded
    handle (the scope's commit happens at most once).
    """

    __slots__ = ("_sid", "_step_id", "_fields", "_lock")

    def __init__(self, sid: str, step_id: int) -> None:
        self._sid = sid
        self._step_id = step_id
        self._fields: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        return self._sid

    @property
    def step_id(self) -> int:
        return self._step_id

    def record(self, **fields: Any) -> None:
        if not fields:
            return
        with self._lock:
            self._fields.update(fields)


class _NullStep:
    """No-op step yielded when the context has no session_id."""

    session_id = ""
    step_id = -1

    def record(self, **fields: Any) -> None:
        return


class Recording:
    """Per-(session, step) buffer + sender registry shared by harness and server.

    Thread-safe; a single instance is exposed at module scope via
    :func:`get_default`.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._buffers: dict[tuple[str, int], dict[str, Any]] = {}
        self._senders: dict[str, SenderFn] = {}
        self._active_session: ContextVar[str | None] = ContextVar("active_session", default=None)
        self._active_step: ContextVar[Step | None] = ContextVar("active_step", default=None)
        self._active_keys: set[tuple[str, int]] = set()
        self._ever_registered_sender = False

    # ── Sender registry ─────────────────────────────────────────────

    def register_sender(self, session_id: str, sender: SenderFn) -> None:
        """Route committed fields for *session_id* over the wire."""
        with self._cond:
            self._senders[session_id] = sender
            self._ever_registered_sender = True

    def unregister_sender(self, session_id: str) -> None:
        with self._cond:
            self._senders.pop(session_id, None)

    # ── Lifecycle scopes ────────────────────────────────────────────

    @contextmanager
    def session_scope(self, session_id: str | None) -> Iterator[None]:
        """Bind *session_id* as the active session for the duration of the block.

        Clears the per-session buckets on entry and exit so the runner sees
        a fresh slate per episode.  On exit any leftover buckets for this
        session are dropped with a WARNING.  ``session_id=None`` is a
        no-op scope (used when the connection has no session yet).
        """
        if not session_id:
            yield
            return
        token = self._active_session.set(session_id)
        self._clear_session(session_id, warn=False)
        try:
            yield
        finally:
            self._active_session.reset(token)
            self._clear_session(session_id, warn=True)

    @contextmanager
    def step_scope(self, ctx: _HasSessionAndStep) -> Iterator[Step | _NullStep]:
        """Bind a ``(session_id, step_id)`` scope for the duration of the block.

        The yielded :class:`Step` accumulates fields via ``step.record(...)``.
        On exit (including via exception) all accumulated fields ship as ONE
        atomic ``RECORD_COMMIT`` frame (when a wire sender is registered)
        or land in the local buffer.  Re-entering for the same ``(sid,
        step_id)`` raises ``RuntimeError``.
        """
        sid = ctx.session_id
        if not sid:
            yield _NullStep()
            return
        step_id = ctx.step
        key = (sid, step_id)
        with self._cond:
            if key in self._active_keys:
                raise RuntimeError(f"step_scope: double-entry for session={sid[:8]} step={step_id}")
            self._active_keys.add(key)
        step = Step(sid, step_id)
        token = self._active_step.set(step)
        try:
            yield step
        finally:
            self._active_step.reset(token)
            with self._cond:
                self._active_keys.discard(key)
            # Always commit fields-so-far, even when the body raised.
            self._commit_step(sid, step_id, step._fields)

    # ── Recording API ───────────────────────────────────────────────

    def record(self, ctx: _HasSessionAndStep, **fields: Any) -> None:
        """Record fields into the currently active step_scope.

        Raises ``RuntimeError`` if called outside ``step_scope`` or if the
        active scope's session does not match ``ctx.session_id``.
        """
        if not fields:
            return
        step = self._active_step.get()
        if step is None:
            raise RuntimeError("recording.record() must be called inside `with recording.step_scope(ctx):`")
        if step._sid != ctx.session_id:
            raise RuntimeError(
                f"recording.record(): ctx.session_id={ctx.session_id!r} does not match "
                f"active step_scope session={step._sid!r}"
            )
        step.record(**fields)

    # ── Harness-side ingest / drain ─────────────────────────────────

    def ingest(self, session_id: str, step_id: int, fields: dict[str, Any]) -> None:
        """Merge committed fields that arrived over the wire into the bucket."""
        with self._cond:
            bucket = self._buffers.setdefault((session_id, step_id), {})
            bucket.update(fields)
            self._cond.notify_all()

    def drain_step(self, session_id: str, step_id: int, *, timeout: float = 0.0) -> dict[str, Any]:
        """Return and remove the bucket for ``(session_id, step_id)``.

        If the bucket is missing and *timeout* > 0, briefly wait for a
        matching ``ingest()``.  Returns ``{}`` if the wait times out.
        """
        key = (session_id, step_id)
        with self._cond:
            if key in self._buffers:
                return self._buffers.pop(key)
            if timeout <= 0:
                return {}
            self._cond.wait_for(lambda: key in self._buffers, timeout=timeout)
            return self._buffers.pop(key, {})

    def has_pending(self, session_id: str) -> list[int]:
        """Return step_ids still buffered for *session_id* (diagnostic)."""
        with self._cond:
            return sorted(s for sid, s in self._buffers if sid == session_id)

    def current_session(self) -> str | None:
        return self._active_session.get()

    # ── Internals ───────────────────────────────────────────────────

    def _commit_step(self, sid: str, step_id: int, fields: dict[str, Any]) -> None:
        if not fields:
            return
        with self._cond:
            sender = self._senders.get(sid)
            ever_registered = self._ever_registered_sender
        if ever_registered:
            if sender is None:
                logger.warning(
                    "step_scope commit: no sender registered for session=%s step=%d; %d field(s) dropped",
                    sid[:8],
                    step_id,
                    len(fields),
                )
                return
            try:
                sender(sid, step_id, fields)
            except Exception:
                logger.exception("recording sender failed; dropping fields for session=%s step=%d", sid[:8], step_id)
            return
        # No sender ever registered → in-process mode (harness side or tests).
        self.ingest(sid, step_id, fields)

    def _clear_session(self, sid: str, *, warn: bool) -> None:
        with self._cond:
            stale = [(s, st) for (s, st) in self._buffers if s == sid]
            if stale and warn:
                logger.warning(
                    "session_scope exit: dropping %d unmerged RECORD_COMMIT bucket(s) for session=%s steps=%s",
                    len(stale),
                    sid[:8],
                    sorted(st for _, st in stale),
                )
            for key in stale:
                self._buffers.pop(key, None)


_default = Recording()


def get_default() -> Recording:
    """Return the process-wide default :class:`Recording` instance.

    Framework internals (runners, :class:`EpisodeRecorder`) need direct
    instance access; user code should reach for the module-level helpers.
    """
    return _default


def record(ctx: _HasSessionAndStep, **fields: Any) -> None:
    _default.record(ctx, **fields)


def ingest(session_id: str, step_id: int, fields: dict[str, Any]) -> None:
    _default.ingest(session_id, step_id, fields)


def register_sender(session_id: str, sender: SenderFn) -> None:
    _default.register_sender(session_id, sender)


def unregister_sender(session_id: str) -> None:
    _default.unregister_sender(session_id)


def session_scope(session_id: str | None) -> Any:
    return _default.session_scope(session_id)


def step_scope(ctx: _HasSessionAndStep) -> Any:
    return _default.step_scope(ctx)


__all__ = [
    "Recording",
    "SenderFn",
    "Step",
    "get_default",
    "ingest",
    "record",
    "register_sender",
    "session_scope",
    "step_scope",
    "unregister_sender",
]
