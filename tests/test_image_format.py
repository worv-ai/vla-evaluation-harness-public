"""Tests for server.image_format — the observation wire codec.

Encoding runs inline in the live episode loop, so this setting caps the
achievable control rate. png costs ~40 ms per 640x480 image; two cameras put a
30 Hz benchmark under 13 Hz on encode alone. These cover the config plumbing and
the round-trip, not the timing (measured separately).
"""

from __future__ import annotations

import numpy as np
import pytest

from vla_eval.config import ServerConfig
from vla_eval.protocol.numpy_codec import decode_ndarray, encode_ndarray, get_image_format, set_image_format


@pytest.fixture(autouse=True)
def _restore_format():
    before = get_image_format()
    yield
    set_image_format(before)


def test_defaults_to_png() -> None:
    assert ServerConfig.from_dict({"url": "ws://x:1"}).image_format == "png"
    assert ServerConfig().image_format == "png"


@pytest.mark.parametrize("fmt", ["png", "jpeg", "raw"])
def test_accepts_each_supported_format(fmt: str) -> None:
    assert ServerConfig.from_dict({"image_format": fmt}).image_format == fmt


def test_rejects_unknown_format() -> None:
    # A typo must not silently fall back to the slow default.
    with pytest.raises(ValueError, match="image_format"):
        ServerConfig.from_dict({"image_format": "jpg"})


def test_orchestrator_applies_the_configured_format() -> None:
    from vla_eval.orchestrator import Orchestrator

    set_image_format("png")
    Orchestrator({"server": {"url": "ws://x:1", "image_format": "jpeg"}, "benchmarks": []}, no_save=True)
    assert get_image_format() == "jpeg"


@pytest.mark.parametrize("fmt", ["png", "jpeg", "raw"])
def test_image_round_trips_in_every_format(fmt: str) -> None:
    set_image_format(fmt)
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    img[8:24, 12:36] = 200  # a block survives lossy encoding; noise would not
    out = decode_ndarray(encode_ndarray(img))
    assert out.shape == img.shape and out.dtype == np.uint8
    # jpeg is lossy: assert the structure survived, not bit equality.
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 3


def test_non_image_arrays_are_unaffected_by_the_format() -> None:
    # Actions/states are float32 — they must round-trip exactly regardless.
    set_image_format("jpeg")
    state = np.linspace(-1, 1, 14, dtype=np.float32)
    np.testing.assert_array_equal(decode_ndarray(encode_ndarray(state)), state)


def test_decode_detects_format_from_the_payload() -> None:
    # The receiver never configures a format: a jpeg-encoded obs must decode on
    # a server left at the png default. Otherwise the setting would need to be
    # applied on both ends and would silently corrupt when they disagreed.
    set_image_format("jpeg")
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    payload = encode_ndarray(img)
    set_image_format("png")  # receiver's setting differs
    out = decode_ndarray(payload)
    assert out.shape == img.shape
    assert np.abs(out.astype(int) - 128).mean() < 3
