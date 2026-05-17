"""Wire types for the recording daemon.

Three message types — that's it:

- ``STEP``: one step row in a per-episode jsonl. Sent by orchestrator
  (via :class:`vla_eval.recording.EpisodeRecorder`) or by external code
  living inside the model server (e.g. reflex-train).
- ``RESULT``: the orchestrator declares the episode finished. Carries the
  fully-resolved ``jsonl_path`` so the daemon does not render filenames.
  Daemon flushes the bucket's accumulated rows to that path and appends a
  row to the eval-level aggregate.
- ``EVAL_END``: the orchestrator declares the run finished. Daemon flushes
  the aggregate to ``aggregate_dir/aggregate_filename`` (carried on the
  first ``RESULT`` of the eval).

Episode and eval lifecycles are *implicit*: buckets/evals are created
lazily on first arrival and removed on ``RESULT`` / ``EVAL_END``. There is
no ``EPISODE_START`` or ``EVAL_START`` — neither side needs them.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

import msgpack

from vla_eval.protocol.numpy_codec import decode_ndarray, encode_ndarray


class RecMessageType(str, enum.Enum):
    STEP = "step"
    RESULT = "result"
    EVAL_END = "eval_end"


@dataclass
class RecordingFrame:
    type: RecMessageType
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)


def pack_frame(frame: RecordingFrame) -> bytes:
    raw = {"type": frame.type.value, "ts": frame.ts, "payload": frame.payload}
    return msgpack.packb(raw, default=encode_ndarray, use_bin_type=True)


def unpack_frame(data: bytes) -> RecordingFrame:
    """Decode bytes to a :class:`RecordingFrame`. Raises ``ValueError`` on malformed input."""
    try:
        raw = msgpack.unpackb(data, object_hook=decode_ndarray, raw=False)
    except Exception as exc:
        raise ValueError(f"Failed to decode recording frame ({len(data)} bytes): {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Expected msgpack dict, got {type(raw).__name__}")

    missing = [k for k in ("type", "ts", "payload") if k not in raw]
    if missing:
        raise ValueError(f"Recording frame missing required fields: {missing}")

    try:
        msg_type = RecMessageType(raw["type"])
    except ValueError:
        raise ValueError(f"Unknown recording frame type: {raw['type']!r}")

    return RecordingFrame(type=msg_type, payload=raw["payload"], ts=raw["ts"])
