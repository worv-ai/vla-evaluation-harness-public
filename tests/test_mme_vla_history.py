"""Tests for the MME-VLA streaming memory buffer.

The suite (``mme_vla_suite``) is cloned at runtime and needs JAX + weights, so
these drive ``MmeVlaModelServer``'s buffer bookkeeping against a fake policy that
reproduces the two behaviours that matter, quoted from
``mme_vla_suite/policies/policy.py``:

* ``infer`` asserts ``len(self.mem_buffer._history_feats) > 0`` with
  "history feats is empty, add buffer first" for any non-symbolic variant.
* ``add_buffer`` appends ``len(images)`` frames and advances ``step_idx`` by
  that much, so batches must be incremental — re-sending held frames corrupts
  the timeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vla_eval.model_servers.mme_vla import MmeVlaModelServer
from vla_eval.model_servers.predict import PredictModelServer


class FakePolicy:
    """Stands in for MME_VLA_Policy: same assertion, same step_idx accounting."""

    def __init__(self) -> None:
        self.history_feats: list[np.ndarray] = []
        self.step_idx = -1
        self.exec_start_idx = 0
        self.add_buffer_calls: list[dict[str, Any]] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.history_feats = []
        self.step_idx = -1
        self.exec_start_idx = 0
        self.reset_count += 1

    def add_buffer(self, obs: dict) -> None:
        images = obs["images"]
        if obs.get("exec_start_idx", 0) > 0:
            self.exec_start_idx = obs["exec_start_idx"]
        self.add_buffer_calls.append({k: v for k, v in obs.items()})
        self.history_feats.extend(images)
        self.step_idx += len(images)

    def infer(self, obs: dict) -> dict:
        assert len(self.history_feats) > 0, "history feats is empty, add buffer first"
        return {"actions": np.zeros((10, 14), dtype=np.float32)}


class Ctx:
    def __init__(self, session_id: str = "s0", is_first: bool = False, step: int = 0) -> None:
        self.session_id = session_id
        self.is_first = is_first
        self.step = step


def _server(**kw) -> MmeVlaModelServer:
    """Build the server without loading a real policy."""
    srv = MmeVlaModelServer.__new__(MmeVlaModelServer)
    srv.use_history = kw.get("use_history", True)
    srv.image_key = "observation/image"
    srv.wrist_image_key = "observation/wrist_image"
    srv.state_key = "observation/state"
    srv.state_dim = kw.get("state_dim", 14)
    srv.image_resolution = None
    srv._state_warned = False
    srv._pending = {}
    srv._policy = FakePolicy()
    return srv


def _obs(v: int = 0, history: list | None = None) -> dict:
    o: dict[str, Any] = {
        "images": {
            "cam_high": np.full((4, 4, 3), v, dtype=np.uint8),
            "cam_wrist": np.full((4, 4, 3), v, dtype=np.uint8),
        },
        "states": np.full(14, float(v), dtype=np.float32),
        "task_description": "Put Fruits",
    }
    if history is not None:
        o["video_history"] = history
    return o


def test_no_demo_task_still_seeds_the_buffer() -> None:
    # The bug: task1 has no demo clip, nothing ever called add_buffer, and every
    # observation died on the policy's assertion. The episode's own first frame
    # is a valid seed — exec_start_idx 0 means "no demo", not "no buffer".
    srv = _server()
    ctx = Ctx(is_first=True)
    srv._stage_frame(_obs(1), ctx)
    srv.predict(_obs(1), ctx)  # must not raise

    assert len(srv._policy.add_buffer_calls) == 1
    call = srv._policy.add_buffer_calls[0]
    assert call["exec_start_idx"] == 0
    assert call["images"].shape == (1, 1, 4, 4, 3)  # (T, num_views=1, H, W, 3)


def test_demo_prefixes_the_first_batch_and_sets_exec_start_idx() -> None:
    srv = _server()
    ctx = Ctx(is_first=True)
    demo = [np.full((4, 4, 3), 9, dtype=np.uint8) for _ in range(3)]
    srv._stage_frame(_obs(1, history=demo), ctx)
    srv.predict(_obs(1, history=demo), ctx)

    call = srv._policy.add_buffer_calls[0]
    assert call["images"].shape[0] == 4  # 3 demo frames + the first live frame
    assert call["exec_start_idx"] == 3  # execution starts at the live frame
    assert srv._policy.exec_start_idx == 3


def test_batches_are_incremental_across_inferences() -> None:
    # add_buffer advances the policy's step_idx by len(images); re-sending frames
    # it already holds would double-count the timeline.
    srv = _server()
    ctx = Ctx(is_first=True)
    srv._stage_frame(_obs(0), ctx)
    srv.predict(_obs(0), ctx)  # inference 1: 1 frame

    ctx.is_first = False
    for v in (1, 2, 3):  # served from the chunk buffer — staged, not inferred
        srv._stage_frame(_obs(v), ctx)
    srv.predict(_obs(3), ctx)  # inference 2: the 3 staged since

    sizes = [c["images"].shape[0] for c in srv._policy.add_buffer_calls]
    assert sizes == [1, 3]
    assert srv._policy.step_idx == 3  # -1 + 1 + 3 → one slot per real frame
    assert srv._policy.add_buffer_calls[1]["exec_start_idx"] == 0  # only the first batch sets it


def test_every_observation_is_remembered_not_just_inferred_ones() -> None:
    # Memory must see the episode at full rate. Staging in predict() would drop
    # every frame served from the chunk buffer.
    srv = _server()
    ctx = Ctx(is_first=True)
    srv._stage_frame(_obs(0), ctx)
    srv.predict(_obs(0), ctx)
    ctx.is_first = False
    for v in range(1, 25):
        srv._stage_frame(_obs(v), ctx)
    srv.predict(_obs(25), ctx)  # note: 25 not staged; 24 were
    assert len(srv._policy.history_feats) == 25  # 1 + 24, no gaps


def test_states_are_real_for_live_frames_and_zero_for_demo() -> None:
    srv = _server()
    ctx = Ctx(is_first=True)
    demo = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    srv._stage_frame(_obs(7, history=demo), ctx)
    srv.predict(_obs(7, history=demo), ctx)

    states = srv._policy.add_buffer_calls[0]["state"]
    assert states.shape == (3, 14)
    assert np.allclose(states[:2], 0.0)  # demo frames carry no proprioception
    assert np.allclose(states[2], 7.0)  # the live frame carries the real state


def test_episode_start_resets_policy_and_drops_stale_frames() -> None:
    import anyio
    from unittest.mock import AsyncMock, patch

    srv = _server()
    ctx = Ctx(is_first=True)
    srv._stage_frame(_obs(1), ctx)
    assert srv._pending[ctx.session_id]["images"]

    with patch.object(PredictModelServer, "on_episode_start", new=AsyncMock()):
        anyio.run(lambda: MmeVlaModelServer.on_episode_start(srv, {}, ctx))

    assert srv._policy.reset_count == 1
    # Frames staged before the reset belong to the previous episode. Leaving them
    # would splice the tail of one episode onto the head of the next.
    assert ctx.session_id not in srv._pending


def test_sessions_do_not_share_a_buffer() -> None:
    srv = _server()
    a, b = Ctx("sa", is_first=True), Ctx("sb", is_first=True)
    srv._stage_frame(_obs(1), a)
    srv._stage_frame(_obs(2), b)
    srv._stage_frame(_obs(3), a)
    assert len(srv._pending["sa"]["images"]) == 2
    assert len(srv._pending["sb"]["images"]) == 1


def test_baseline_without_history_never_touches_the_buffer() -> None:
    # pi0.5 has no memory; add_buffer must not be called, and its infer has no
    # assertion to satisfy.
    srv = _server(use_history=False)
    srv._policy.history_feats = [np.zeros(1)]  # baseline's infer wouldn't assert
    ctx = Ctx(is_first=True)
    srv.predict(_obs(1), ctx)
    assert srv._policy.add_buffer_calls == []


def test_flush_without_staged_frames_is_a_noop() -> None:
    srv = _server()
    srv._flush_buffer(Ctx())
    assert srv._policy.add_buffer_calls == []


@pytest.mark.parametrize("state_dim", [8, 14])
def test_staged_state_is_truncated_to_the_model_width(state_dim: int) -> None:
    srv = _server(state_dim=state_dim)
    ctx = Ctx(is_first=True)
    srv._stage_frame(_obs(3), ctx)
    srv.predict(_obs(3), ctx)
    assert srv._policy.add_buffer_calls[0]["state"].shape == (1, state_dim)
