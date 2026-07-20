"""Exceptions a benchmark can raise to steer the orchestrator's error recovery.

Most exceptions from a benchmark are per-episode: the orchestrator records the
episode as an error and moves on. The types here opt out of that isolation when
continuing would be wrong.
"""

from __future__ import annotations

from typing import Any


class _PacingError(Exception):
    """Base for the live-runner errors that carry their own evidence.

    ``rt_metrics`` is the episode's pacing measurement (tick_hz, stale ratio,
    loop times). It rides on the exception so the orchestrator can persist it on
    the failure path: the numbers that justify refusing to score the episode are
    exactly the numbers a reader needs to act on it, and they are gone otherwise.
    """

    def __init__(self, message: str, rt_metrics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.rt_metrics: dict[str, Any] = rt_metrics or {}


class NoActionsError(_PacingError):
    """The model server returned nothing for an entire episode.

    Raised by the real-time runner when an episode ends having received zero
    fresh actions. The loop keeps ticking on the hold action when the server is
    silent — that is deliberate, and correct for a server that is merely slow.
    Across a *whole* episode it means something else: every observation failed
    server-side, or the connection is one-way. Nothing was evaluated, so the
    resulting ``success=False`` would be a fabricated model result, identical on
    paper to a policy that genuinely tried and failed.

    The orchestrator records the episode with ``failure_reason="no_actions"`` and
    continues; the fault is usually per-server-config, not per-episode, so the
    next episode gets a chance to show the same thing rather than the run dying
    on a transient.
    """


class TimingContractError(_PacingError):
    """A paced live episode could not hold its target step rate.

    ``tick_hz`` far below the configured ``hz`` means the loop body itself did
    not fit in a step period — the harness taxing the embodiment, not a slow
    model (that shows up as ``stale_action_ratio``, and is a legitimate result).

    It is raised rather than warned because a chunk-based policy converts the
    shortfall directly into *speed*: the runner commands one action per tick, so
    a policy whose chunk spans one second at ``hz`` only ever reaches its first
    ``tick_hz`` entries before the next chunk replaces it. The embodiment then
    executes the whole trajectory at ``tick_hz / hz`` of the demonstrated speed,
    and a wall-clock episode cap turns that into a timeout that is indistinguishable
    on paper from a policy that tried and failed. Real 30 Hz runs sat at ~8 Hz —
    0.26x speed — for five days, scored as model failures, because this was a log
    line nobody read.

    The orchestrator records the episode with ``failure_reason="timing_contract"``
    and continues: the cause is usually per-setup (image encoding, sensor reads,
    network) rather than per-episode, so the next episode should get the chance to
    show the same thing rather than the run dying on a one-off hiccup.
    """


class HardwareFaultError(Exception):
    """The embodiment is unusable; abort the benchmark instead of continuing.

    Raise this from a real-robot benchmark when the hardware has entered a state
    the harness cannot recover from in software — a latched controller fault, a
    tripped e-stop, an arm that will not connect. Per-episode isolation is wrong
    for these: every remaining episode would fail identically, burning the
    operator's run and reporting a 0% success rate that reads as a model result
    when the robot never moved.

    The orchestrator records the episode with ``failure_reason="hardware_fault"``
    and finalizes the benchmark as partial, so the run ends with an accurate
    reason rather than a fabricated score.
    """
