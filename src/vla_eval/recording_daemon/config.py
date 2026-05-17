"""Daemon configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecordingDaemonConfig:
    """Configuration for :class:`vla_eval.recording_daemon.daemon.RecordingDaemon`.

    Attributes:
        bind: WebSocket bind URL (``ws://host:port`` or ``tcp://host:port``).
        out_dir: Filesystem directory where per-episode jsonl/mp4 and per-eval
            aggregate JSON land.
        idle_timeout_sec: Age-out window for buckets/evals that never received
            their terminal ``EPISODE_END`` / ``EVAL_END``.  After this many
            seconds without activity, the bucket flushes with an ``_unclosed``
            suffix.
        health_port: HTTP port for the ``/health`` endpoint.
    """

    bind: str = "ws://0.0.0.0:9001"
    out_dir: str = "./recording-out"
    idle_timeout_sec: float = 600.0
    health_port: int = 9002
