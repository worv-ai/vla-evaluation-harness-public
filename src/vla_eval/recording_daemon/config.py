"""Daemon configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecordingDaemonConfig:
    """Configuration for :class:`vla_eval.recording_daemon.daemon.RecordingDaemon`.

    Attributes:
        bind: WebSocket bind URL (``ws://host:port``).
        out_dir: Default filesystem directory used by the daemon when it
            has to write an ``_unclosed`` aggregate JSON for an eval that
            never received its ``RESULT.aggregate_dir``. On the happy path
            the daemon writes only to paths it receives over the wire, so
            this is a fallback for the safety net.
        idle_timeout_sec: Age-out window for buckets/evals that never
            receive their terminal ``RESULT`` / ``EVAL_END``. After this
            many seconds without activity, the bucket flushes with an
            ``_unclosed`` suffix.
    """

    bind: str = "ws://0.0.0.0:9001"
    out_dir: str = "./recording-out"
    idle_timeout_sec: float = 600.0
