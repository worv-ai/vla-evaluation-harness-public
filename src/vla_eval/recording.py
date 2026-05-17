"""Recording API used by call sites (model server, harness).

Public surface:

- :func:`set_client` / :func:`client` — wire the
  RecordingClient at process startup.  Optional: with no emitter installed,
  step_scope is a no-op and a one-shot WARN is logged.
- :func:`step_scope` — context manager that accumulates fields and emits one
  ``RECORD_COMMIT`` frame on exit, keyed by ``(session_id, step_id)``.

Reflex-train's ``_record_step_trace`` calls this surface directly; the
shape matches the PR #9 API (it does not).  See plan
``continue-fuzzy-parasol.md`` §Reflex-train emit path.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Protocol

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_emitter: Any | None = None
_warned_no_emitter = False


class _HasSessionAndStep(Protocol):
    @property
    def session_id(self) -> str: ...
    @property
    def episode_id(self) -> str: ...
    @property
    def step(self) -> int: ...


class Step:
    """Handle yielded by :func:`step_scope`.  Accumulates fields; commits on scope exit."""

    __slots__ = ("_sid", "_eid", "_step_id", "_fields", "_lock")

    def __init__(self, sid: str, eid: str, step_id: int) -> None:
        self._sid = sid
        self._eid = eid
        self._step_id = step_id
        self._fields: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        return self._sid

    @property
    def episode_id(self) -> str:
        return self._eid

    @property
    def step_id(self) -> int:
        return self._step_id

    def record(self, **fields: Any) -> None:
        if not fields:
            return
        with self._lock:
            self._fields.update(fields)


def set_client(emitter: Any) -> None:
    """Install the daemon emitter for this process. Call once at startup."""
    global _emitter, _warned_no_emitter
    with _lock:
        _emitter = emitter
        _warned_no_emitter = False


def client() -> Any | None:
    return _emitter


@contextmanager
def step_scope(ctx: _HasSessionAndStep) -> Iterator[Step]:
    """Open a scope that batches record(...) calls into one RECORD_COMMIT.

    The scope commits on exit even if the body raises (propagates).  Without
    an installed emitter, records are dropped and a one-shot WARN is logged.
    """
    step = Step(sid=ctx.session_id, eid=ctx.episode_id, step_id=ctx.step)
    try:
        yield step
    finally:
        _commit_step(step)


def _commit_step(step: Step) -> None:
    if not step._fields:
        return
    emitter = _emitter
    if emitter is None:
        global _warned_no_emitter
        with _lock:
            if not _warned_no_emitter:
                _warned_no_emitter = True
                logger.warning(
                    "recording.step_scope called but no emitter installed; "
                    "records dropped. Install with recording.set_client() at startup."
                )
        return
    try:
        emitter.record(step._sid, step._eid, step._step_id, dict(step._fields))
    except Exception:
        logger.exception("Failed to push RECORD_COMMIT for sid=%s eid=%s step=%d", step._sid, step._eid, step._step_id)
