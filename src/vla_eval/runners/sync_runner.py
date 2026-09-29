"""SyncEpisodeRunner: waits for inference before stepping."""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_eval import watchdog
from vla_eval.benchmarks.base import Benchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.runners.base import EpisodeRunner
from vla_eval.types import Action, EpisodeResult, Task


def split_action_chunk(action: Action) -> list[Action]:
    """One action per env step; a 2-D ``actions`` array is a chunk the server sent whole."""
    actions = action.get("actions")
    if actions is None or np.ndim(actions) < 2:
        return [action]
    rows = [{**action, "actions": row} for row in np.asarray(actions)]
    if not rows:
        raise ValueError("the model server returned an empty action chunk")  # else the step loop would spin
    return rows


class SyncEpisodeRunner(EpisodeRunner):
    """Synchronous episode runner: one observation → one action per step.

    ``open_loop=True``: ask the server for whole action chunks and execute them without observing in between
    (``benchmark.observation_needed`` is False on those steps, so a render-bound simulator may skip rendering).

    Episode flow:
        1. ``benchmark.start_episode(task, recorder=...)``
        2. ``benchmark.get_observation()`` → initial observation.
        3. ``conn.start_episode(task_info)``
        4. Step loop (up to ``max_steps``):
           a. ``conn.act(obs)`` → action (or, open-loop, a chunk) from the model server
           b. ``benchmark.apply_action(action)`` per env step
           c. If ``benchmark.is_done()``: break
           d. ``benchmark.get_observation()`` → next observation
        5. ``conn.end_episode()``
    """

    def __init__(self, open_loop: bool = False) -> None:
        self.open_loop = open_loop

    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
        recorder: EpisodeRecorder | None = None,
    ) -> EpisodeResult:
        """Run a synchronous episode."""
        await benchmark.start_episode(task, recorder=recorder)
        obs_dict = await benchmark.get_observation()

        task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
        ep_payload: dict[str, Any] = {"task": task_info}
        if self.open_loop:
            ep_payload["open_loop"] = True
        if recorder is not None and recorder.is_active:
            ep_payload["recording"] = {
                "sid": recorder.sid,
                "eid": recorder.eid,
                "eval_id": recorder.eval_id,
                "db_path": recorder.db_path,
            }
        await conn.start_episode(ep_payload)

        step = 0
        done = False

        def more() -> bool:
            return not done and (max_steps is None or step < max_steps)

        while more():
            action = await conn.act(obs_dict)
            # Closed-loop, a 2-D action reaches the benchmark as is (adapters may take its first row).
            chunk = split_action_chunk(action) if self.open_loop else [action]
            for i, action in enumerate(chunk):
                benchmark.observation_needed = i == len(chunk) - 1
                await benchmark.apply_action(action)
                watchdog.pet()  # a slow simulator's episode can outlast the stall timeout
                step += 1
                done = await benchmark.is_done()
                if not more():
                    break
            benchmark.observation_needed = True
            if more():
                obs_dict = await benchmark.get_observation()

        elapsed = await benchmark.get_time()
        metrics = await benchmark.get_result()
        episode_result: dict = {"metrics": metrics, "steps": step, "elapsed_sec": round(elapsed, 3)}

        await conn.end_episode(episode_result)
        return episode_result
