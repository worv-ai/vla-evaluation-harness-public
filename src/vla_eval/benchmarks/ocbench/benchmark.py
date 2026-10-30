"""OCBench on MuJoCo Warp: ``num_envs`` worlds of one task stepped together on the GPU.

OCBench (arXiv:2610.07056, ``pip install "ocbench[mjwarp]"``) has 28 manipulation tasks with
state observations.  The protocol follows OCBench's ``evaluate_mjwarp``: an episode ends when
its world terminates or after the task's step limit, and succeeds when ``info["success"]`` is set
then; bowling, which never flags success, scores its return (``score``, the paper's metric, is
the return x 100).  Episode ``i`` of a task resets its world with seed
``SeedSequence([seed, 10]).generate_state(i + 1)[i]``, the seeds OCBench evaluations draw
for a seed root, so an episode is the same whatever world or ``num_envs`` runs it.

Observation: ``{"state": float32[D]}``.  Action: ``{"actions": float[7]}``, OCBench's joint
action (six arm deltas and the gripper, in [-1, 1]).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from vla_eval.benchmarks.base import StepResult, VectorStepBenchmark
from vla_eval.recording import EpisodeRecorder
from vla_eval.specs import RAW, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

TASKS = [
    *(f"block-{v}-v0" for v in ("single-task1", "double-task1", "double-task2", "triple-task1", "triple-task2")),
    *(f"block-quadruple-task{i}-v0" for i in (1, 2, 3)),
    *(f"chamber-easy-task{i}-v0" for i in (1, 2, 3)),
    *(f"chamber-{d}-task{i}-v0" for d in ("medium", "hard") for i in (1, 2)),
    *(f"switch-{g}-task{i}-v0" for g in ("3x3", "4x4", "5x5") for i in (1, 2)),
    *(f"hanoi-{d}-task{i}-v0" for d in ("single", "double", "triple") for i in (1, 2)),
    "bowling-task1-v0",
]


class OCBenchBenchmark(VectorStepBenchmark):
    """Params: ``num_envs`` (worlds), ``tasks`` (OCBench MJWarp env names; default the 28 tasks),
    ``seed`` (the evaluation's seed root)."""

    partial_reset: ClassVar[bool] = True

    def __init__(self, num_envs: int = 500, tasks: list[str] | None = None, seed: int = 0) -> None:
        super().__init__(num_envs)
        self.task_names = list(tasks or TASKS)
        self.seed = seed
        self._env: Any = None
        self._env_name = ""
        self._limit = 0
        self._steps = np.zeros(num_envs, np.int64)
        self._returns = np.zeros(num_envs)
        self._obs: np.ndarray | None = None
        self._recorders: dict[int, EpisodeRecorder] = {}

    def get_tasks(self) -> list[Task]:
        return [{"name": name} for name in self.task_names]

    def _episode_seed(self, episode_idx: int) -> int:
        return int(np.random.SeedSequence([self.seed, 10]).generate_state(episode_idx + 1, dtype=np.uint32)[-1])

    def reset(self, slots: list[int], tasks: list[Task], recorders: list[EpisodeRecorder]) -> list[Any]:
        import ocbench

        name = str(tasks[0]["name"])  # the harness runs one task at a time
        seeds = np.array([self._episode_seed(int(t["episode_idx"])) for t in tasks], np.uint32)
        if name != self._env_name:
            if self._env is not None:
                self._env.close()
            family, _, kwargs, self._limit = ocbench.parse_env_spec(name)
            from ocbench.mjwarp import make_env

            self._env, self._env_name = make_env(family, kwargs, self.num_envs), name
            full = np.zeros(self.num_envs, np.uint32)  # a first reset is of every world
            full[slots] = seeds
            obs, _ = self._env.reset(seeds=full)
        else:
            obs, _ = self._env.reset_worlds(np.asarray(slots, np.int32), seeds=seeds)
        # only the reset worlds' rows: reset_worlds recomputes the other worlds' observations, not as step() does
        self._obs = np.asarray(obs, np.float32)
        self._steps[slots] = 0
        self._returns[slots] = 0.0
        self._recorders.update(zip(slots, recorders))
        return [self._obs[slot] for slot in slots]

    def step(self, actions: dict[int, Action]) -> dict[int, StepResult]:
        slots = np.fromiter(actions, np.int64, len(actions))
        action = np.zeros((self.num_envs, 7))  # idle worlds hold still, as in evaluate_mjwarp
        action[slots] = np.stack([np.asarray(actions[s]["actions"], np.float64).reshape(7) for s in slots])
        obs, reward, terminated, _, info = self._env.step(action)
        self._obs = np.asarray(obs, np.float32)
        self._steps[slots] += 1
        self._returns[slots] += np.asarray(reward, np.float64)[slots]
        done = np.asarray(terminated, bool) | (self._steps >= self._limit)
        success = np.asarray(info["success"], bool)
        failure = np.asarray(info.get("failure", np.zeros(self.num_envs, bool)), bool)
        for s in slots:
            self._recorders[int(s)].record_step(reward=float(reward[s]), done=bool(done[s]), success=bool(success[s]))
        return {
            int(s): StepResult(
                obs=self._obs[s],
                reward=float(reward[s]),
                done=bool(done[s]),
                info={"success": bool(success[s]), "failure": bool(failure[s]), "return": float(self._returns[s])},
            )
            for s in slots
        }

    def make_obs(self, raw_obs: Any, slot: int, task: Task) -> Observation:
        return {"state": raw_obs, "task_description": task["name"]}

    def get_step_result(self, slot: int, step_result: StepResult) -> EpisodeResult:
        info = step_result.info
        out: EpisodeResult = {"success": info["success"], "failure": info["failure"], "return": info["return"]}
        if self._env_name.startswith("bowling"):
            out["score"] = 100.0 * info["return"]
        return out

    def get_metric_keys(self) -> dict[str, str]:
        return {"success": "mean", "failure": "mean", "return": "mean", "score": "mean"}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": None}  # each task's own limit (1250-5000), applied in step()

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {"actions": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {"state": RAW}

    def cleanup(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
