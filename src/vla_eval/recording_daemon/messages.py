"""Wire types for the recording daemon.

Each frame is a single msgpack dict ``{type, ts, payload}`` packed with the
numpy-aware codec from :mod:`vla_eval.protocol.numpy_codec`.  See the design
notes (``plans/continue-fuzzy-parasol.md``) for the lifecycle and the per-type
payload contracts.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any

import msgpack

from vla_eval.protocol.numpy_codec import decode_ndarray, encode_ndarray


class RecordingMessageType(str, enum.Enum):
    EVAL_START = "eval_start"
    EVAL_END = "eval_end"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    EPISODE_RESULT = "episode_result"
    RECORD_COMMIT = "record_commit"
    VIDEO_ARTIFACT = "video_artifact"


@dataclass
class RecordingFrame:
    type: RecordingMessageType
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
        msg_type = RecordingMessageType(raw["type"])
    except ValueError:
        raise ValueError(f"Unknown recording frame type: {raw['type']!r}")

    return RecordingFrame(type=msg_type, payload=raw["payload"], ts=raw["ts"])
