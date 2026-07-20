"""LiveEpisodeRunner: real-time wall-clock evaluation.

Ties the environment clock to wall-clock time.  The environment advances at
a fixed Hz whether or not the model has returned an action.

Pacing is controlled by a :class:`~vla_eval.runners.clock.Clock` instance.
Pass ``Clock(pace=math.inf)`` to run at maximum speed (no sleeping) — useful
for fast simulation where the deployment gap should emerge from actual
inference latency rather than wall-clock pacing.

Two different failures both look like "it ran slow", and ``rt_metrics`` keeps
them apart:

* **The model is slow.** The loop still ticks at ``hz``; the buffer has no fresh
  action, so the hold is commanded. Read ``stale_action_ratio`` and
  ``effective_control_hz``. This is the deployment gap the harness exists to
  expose — it is not a bug.
* **The loop is slow.** The loop body itself does not fit in a step period, so
  everything stretches: observation rate, command rate, control rate. Read
  ``tick_hz`` against ``hz``, and ``loop_time_mean``. This *is* a bug — the
  harness taxing the robot — and it silently invalidates the timing contract a
  policy trained at a fixed rate depends on.
"""

from __future__ import annotations

import itertools
import logging
import math
import time as _time
from typing import Any

import anyio

import numpy as np

from vla_eval.benchmarks.base import Benchmark
from vla_eval.exceptions import NoActionsError, TimingContractError
from vla_eval.recording import EpisodeRecorder
from vla_eval.runners.action_buffer import ActionBuffer
from vla_eval.runners.base import EpisodeRunner
from vla_eval.runners.clock import Clock
from vla_eval.types import EpisodeResult, Task

logger = logging.getLogger(__name__)


class LiveEpisodeRunner(EpisodeRunner):
    """Real-time episode runner with clock-based pacing.

    Args:
        hz: Environment step frequency (default 10.0).
        clock: Clock instance for pacing. Defaults to real-time (pace=1.0).
        min_tick_hz_ratio: Fraction of ``hz`` the loop must actually achieve for
            the episode to count. Below it the episode is failed with
            :class:`~vla_eval.exceptions.TimingContractError` instead of scored.
            Only applies to a *paced* clock (an unpaced one is asked to run flat
            out, so its tick rate is meaningless). ``0`` disables the check.

    The stale-tick hold action comes from ``benchmark.get_hold_action`` — the
    embodiment owns it (see :class:`~vla_eval.runners.action_buffer.ActionBuffer`).
    """

    # Judging a rate needs enough of a window that startup and a prompt operator
    # label don't dominate the average; below this the episode is too short for
    # tick_hz to mean anything and the check is skipped.
    _MIN_JUDGE_SEC = 1.0

    def __init__(
        self,
        hz: float = 10.0,
        clock: Clock | None = None,
        wait_first_action: bool = False,
        min_tick_hz_ratio: float = 0.8,
    ) -> None:
        self.hz = hz
        self.clock = clock or Clock()
        self.wait_first_action = wait_first_action
        self.min_tick_hz_ratio = float(min_tick_hz_ratio)

    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        """Run a single real-time episode.

        The clock resets right before the first observation is sent, so setup
        time (env reset, model server episode_start) is NOT counted.  The step
        loop starts immediately — no waiting for the first action.  Until the
        model server responds, the benchmark's hold action (``get_hold_action``)
        supplies actions, faithfully reflecting real deployment where physics does
        not pause for the controller to warm up.
        """
        clock = self.clock

        # --- Setup phase (not timed) ---
        await benchmark.start_episode(task, recorder=recorder)
        obs_dict = await benchmark.get_observation()

        task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
        ep_payload: dict[str, Any] = {"task": task_info, "mode": "live"}
        if recorder is not None and recorder.is_active:
            ep_payload["recording"] = {
                "sid": recorder.sid,
                "eid": recorder.eid,
                "eval_id": recorder.eval_id,
                "db_path": recorder.db_path,
            }
        await conn.start_episode(ep_payload)

        # Stale-tick hold is embodiment-owned; get_hold_action(None) also covers
        # the pre-first-action fallback. Raises if the benchmark hasn't declared it.
        action_buffer = ActionBuffer(hold_fn=benchmark.get_hold_action)
        conn.on_action(lambda a: action_buffer.update(a))
        await conn.start_listener()

        step_period = 1.0 / self.hz
        step_times: list[float] = []
        loop_times: list[float] = []
        step_count = 0

        try:
            # --- Episode begins: clock starts, first obs sent ---
            clock.reset()
            await conn.send_observation(obs_dict)

            # By default, no waiting for first action: the step loop starts
            # immediately.  The model server computes concurrently; until it
            # responds, action_buffer.get() returns a zero/held action — just
            # like real deployment where physics does not pause for inference.
            #
            # When wait_first_action=True, we block until the first action
            # arrives.  This is useful for sanity-checking that the live
            # pipeline matches sync results (eliminates step-0 zero action).
            if self.wait_first_action:
                deadline = _time.monotonic() + 30.0
                while not action_buffer.has_action():
                    if _time.monotonic() > deadline:
                        raise TimeoutError("wait_first_action: no action received within 30s")
                    await anyio.sleep(0.0001)

            steps = range(max_steps) if max_steps is not None else itertools.count()
            for step in steps:
                step_start = clock.time()
                _loop_t0 = _time.monotonic()

                action = action_buffer.get()

                _t0 = _time.monotonic()
                await benchmark.apply_action(action)
                step_times.append(_time.monotonic() - _t0)
                step_count += 1
                if await benchmark.is_done():
                    break
                obs_dict = await benchmark.get_observation()

                # Send next observation
                await conn.send_observation(obs_dict)

                # Everything above is the loop's own cost — what has to fit in a
                # step period to hold the target rate. Timed as a whole because
                # the expensive parts are NOT in apply_action: get_observation
                # carries sensor reads, and send_observation carries image
                # encoding. Timing only apply_action hides them (see the
                # step_time vs loop_time split in rt_metrics).
                loop_times.append(_time.monotonic() - _loop_t0)

                # Pacing via clock
                await clock.wait_until(step_start + step_period)

        finally:
            await conn.stop_listener()

        elapsed = clock.time()
        bench_metrics = await benchmark.get_result()
        episode_result: dict = {"metrics": bench_metrics, "steps": step_count, "elapsed_sec": round(elapsed, 3)}

        # Real-time metrics. Three rates that are easy to conflate:
        #   tick_hz    — how fast the loop actually ran (what the robot felt).
        #   effective_control_hz — how fast FRESH actions arrived from the server.
        #   self.hz    — what was asked for.
        # tick < asked means the loop couldn't keep up (loop_time); control <
        # tick means the model couldn't (stale ticks). They fail differently.
        metrics = action_buffer.get_metrics()
        step_mean = float(np.mean(step_times)) if step_times else 0.0
        step_max = float(np.max(step_times)) if step_times else 0.0
        loop_mean = float(np.mean(loop_times)) if loop_times else 0.0
        loop_max = float(np.max(loop_times)) if loop_times else 0.0
        effective_hz = metrics["update_count"] / elapsed if elapsed > 0 else 0.0
        tick_hz = step_count / elapsed if elapsed > 0 else 0.0
        episode_result["rt_metrics"] = {
            **metrics,
            "effective_control_hz": effective_hz,
            "tick_hz": tick_hz,
            "step_time_mean": step_mean,
            "step_time_max": step_max,
            "loop_time_mean": loop_mean,
            "loop_time_max": loop_max,
        }

        logger.info(
            "Episode done: %d steps %.1fs | stale=%.0f%% | tick=%.1f/%.1fHz | control=%.1fHz | "
            "loop mean=%.1fms max=%.1fms (env.step %.1fms)",
            step_count,
            elapsed,
            metrics["stale_action_ratio"] * 100,
            tick_hz,
            self.hz,
            effective_hz,
            loop_mean * 1000,
            loop_max * 1000,
            step_mean * 1000,
        )

        # Warn on the loop's whole cost, not just apply_action. The 2026-07-15
        # real-robot runs sat at 8 Hz against a 30 Hz target for days without
        # tripping this, because the cost was PNG-encoding observations inside
        # send_observation — nowhere near the old apply_action-only measurement.
        if loop_mean > step_period:
            logger.warning(
                "Loop body (%.1fms) exceeds the step period (%.1fms at %.0fHz) — cannot hold the "
                "target rate; actual tick was %.1fHz. env.step (apply_action) is only %.1fms of it, "
                "so look at get_observation (sensor reads) and send_observation (image encoding: "
                "see server.image_format).",
                loop_mean * 1000,
                step_period * 1000,
                self.hz,
                tick_hz,
                step_mean * 1000,
            )

        # Zero fresh actions across a whole episode is not a model result. Every
        # observation failed (or the server never answered), so the embodiment
        # ran the hold action start to finish and nothing was evaluated — but the
        # episode still ends with metrics, and success=False is indistinguishable
        # from a policy that tried and failed. Say so instead of scoring it.
        if step_count > 0 and metrics["update_count"] == 0:
            raise NoActionsError(
                f"model server returned no actions for the entire episode ({step_count} steps, "
                f"{elapsed:.0f}s): the embodiment held its position throughout and nothing was "
                "evaluated. Check the server log — observations are most likely erroring server-side.",
                episode_result["rt_metrics"],
            )

        # A loop that never held its target rate did not run the experiment that
        # was configured: with one action commanded per tick, a shortfall here is
        # a proportional slowdown of the embodiment (see TimingContractError).
        # Scoring it would file a harness bug as a model result.
        floor = self.min_tick_hz_ratio * self.hz
        if (
            self.min_tick_hz_ratio > 0
            and not math.isinf(clock.pace)
            and elapsed >= self._MIN_JUDGE_SEC
            and tick_hz < floor
        ):
            raise TimingContractError(
                f"loop held only {tick_hz:.1f}Hz of the configured {self.hz:.0f}Hz "
                f"({tick_hz / self.hz:.0%}, floor {floor:.1f}Hz) across {step_count} steps / "
                f"{elapsed:.0f}s. The loop body averaged {loop_mean * 1000:.1f}ms against a "
                f"{step_period * 1000:.1f}ms step period, of which env.step (apply_action) was "
                f"{step_mean * 1000:.1f}ms — so look at get_observation (sensor reads) and "
                f"send_observation (image encoding: see server.image_format). An action-chunk "
                f"policy executes at roughly this fraction of its demonstrated speed, so the "
                f"episode was not scored. Set min_tick_hz_ratio: 0 to evaluate anyway.",
                episode_result["rt_metrics"],
            )

        await conn.end_episode(episode_result)
        return episode_result
