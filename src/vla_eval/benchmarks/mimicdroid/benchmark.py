"""MimicDroid-RoboCasa benchmark (arXiv 2509.09769) — Abstract (DemoTwoHand) embodiment.

Few-shot in-context imitation: 12 kitchen tasks across three generalization levels
(L1 seen objects/seen kitchens, L2 unseen objects, L3 unseen kitchens), 14-D bimanual
absolute wrist-pose + grasp actions at 20 Hz. The evaluation protocol follows the
authors' repo guidance (UT-Austin-RPL/mimicdroid-robocasa issues #4/#5/#6):

- ``env.reset()`` randomizes the object pose; the robot is then *warm-started*
  toward the first in-context demonstration's initial configuration by stepping
  that demo's first action for ``warm_start_steps`` ticks (issue #6).
- Observations are the three official agent views; live renders are vertically
  flipped to match the recorded-demo orientation the policies train on.
- Episode resets are drawn from a deterministic per-(task, episode) seed so that
  different model servers evaluate on an identical, paired reset matrix.

The in-context demonstrations themselves are the *model server's* concern: the
benchmark sends ``task_description`` = the task name, and a few-shot server loads
that task's demo files from its own copy of the dataset.

Dataset layout (HuggingFace ``Rutav/MimicDroidDataset``)::

    <data_dir>/TaskDemos/<task>/003/demo.hdf5     # env_args + demo states/actions
"""

from __future__ import annotations

import os
import zlib
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ALL_TASKS: list[dict[str, str]] = [
    {"name": "PnPSinkToRightCounterPlate", "suite": "L1"},
    {"name": "PnPSinkToCabinet", "suite": "L1"},
    {"name": "TurnOnFaucet", "suite": "L1"},
    {"name": "CloseLeftCabinetDoor", "suite": "L1"},
    {"name": "PnPSinkToRightCounterPlateL2", "suite": "L2"},
    {"name": "PnPSinkToCabinetL2", "suite": "L2"},
    {"name": "CloseRightCabinetDoorL2", "suite": "L2"},
    {"name": "CloseLeftCabinetDoorL2", "suite": "L2"},
    {"name": "PnPSinkToRightCounterPlateL3", "suite": "L3"},
    {"name": "CloseLeftCabinetDoorL3", "suite": "L3"},
    {"name": "PnPSinkToMicrowaveTopL3", "suite": "L3"},
    {"name": "TurnOnFaucetL3", "suite": "L3"},
]

CAMERA_NAMES = [
    "robot0_agentview_center_top",
    "robot0_agentview_sideleft",
    "robot0_agentview_sideright",
]

ACTION_DIM = 14
PROPRIO_DIM = 16


def _episode_seed(task_name: str, episode_idx: int) -> int:
    """Deterministic per-(task, episode) seed — identical across servers/arms."""
    return zlib.crc32(f"{task_name}|{episode_idx}".encode()) & 0x7FFFFFFF


class MimicDroidBenchmark(StepBenchmark):
    """MimicDroid Abstract-embodiment benchmark on RoboCasa kitchens.

    Args (config YAML ``params:``):
        data_dir: dataset root containing ``TaskDemos/``.
        robot: robosuite robot name (paper Abstract embodiment: ``DemoTwoHand``).
        env_instance: TaskDemos instance subdirectory (the release ships ``003``).
        camera_size: render resolution (the policies train on 128).
        warm_start_steps: how many ticks to repeat the first demo action after
            reset (issue #6 protocol); 0 disables warm-start.
        suite: optional filter — ``L1``/``L2``/``L3``.
        max_steps: episode horizon at 20 Hz.
    """

    def __init__(
        self,
        data_dir: str,
        robot: str = "DemoTwoHand",
        env_instance: str = "003",
        camera_size: int = 128,
        warm_start_steps: int = 20,
        gripper_mode: str = "sign",
        suite: str | None = None,
        max_steps: int = 350,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.robot = robot
        self.env_instance = env_instance
        self.camera_size = camera_size
        self.warm_start_steps = warm_start_steps
        self.gripper_mode = gripper_mode
        self.suite = suite
        self.max_steps = max_steps
        self._env: Any = None
        self._env_task: str | None = None
        self._task_name: str = ""
        self._warm_actions: dict[str, np.ndarray] = {}

    # --- required API ---

    def get_tasks(self) -> list[Task]:
        tasks = [dict(t) for t in ALL_TASKS]
        if self.suite:
            tasks = [t for t in tasks if t["suite"] == self.suite]
        return tasks

    def reset(self, task: Task) -> Any:
        name = task["name"]
        if self._env is None or self._env_task != name:
            self.cleanup()
            self._env = self._make_env(name)
            self._env_task = name
        self._task_name = name
        seed = _episode_seed(name, int(task.get("episode_idx", 0)))
        np.random.seed(seed)
        obs = self._env.reset()
        warm = self._warm_action(name)
        for _ in range(self.warm_start_steps):
            obs, _, _, _ = self._env.step(warm)
        return obs

    def step(self, action: Action) -> StepResult:
        act = np.asarray(action["actions"], dtype=np.float64).reshape(-1)[:ACTION_DIM].copy()
        if self.gripper_mode == "sign":  # binarize grasp dims (sim-side convention, cf. LIBERO)
            act[12] = 1.0 if act[12] > 0.0 else -1.0
            act[13] = 1.0 if act[13] > 0.0 else -1.0
        obs, reward, done, info = self._env.step(act)
        success = bool(self._env._check_success())
        return StepResult(obs=obs, reward=float(reward), done=done or success, info={"success": success})

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        # Live renders are upside-down vs the recorded demos the policies train on.
        images = {c: np.ascontiguousarray(np.asarray(raw_obs[f"{c}_image"])[::-1]) for c in CAMERA_NAMES}
        return {
            "images": images,
            "state": self._build_proprio(raw_obs),
            "task_description": task["name"],
        }

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": bool(step_result.info.get("success", False))}

    # --- optional API ---

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "actions": DimSpec(
                name="bimanual_wrist_pose",
                dims=ACTION_DIM,
                format="absolute_pos_aa_grasp_x2",
                description="2x [pos(3), axis-angle(3)] + 2x grasp (binarized sim-side when gripper_mode='sign')",
            )
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec = {f"images.{c}": IMAGE_RGB for c in CAMERA_NAMES}
        spec["state"] = DimSpec(
            name="robot_obs",
            dims=PROPRIO_DIM,
            format="world_eef_pos_quat_grip_x2",
            description="[L pos(3)+quat(4), R pos(3)+quat(4), L grip, R grip]",
        )
        spec["task_description"] = LANGUAGE
        return spec

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps}

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
                self._env_task = None

    # --- helpers ---

    def _demo_path(self, task_name: str) -> str:
        return os.path.join(self.data_dir, "TaskDemos", task_name, self.env_instance, "demo.hdf5")

    def _make_env(self, task_name: str) -> Any:
        import json

        import h5py
        import robosuite

        with h5py.File(self._demo_path(task_name), "r") as f:
            env_kwargs = json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        env_kwargs["robots"] = [self.robot]
        for k in ("has_renderer", "use_camera_obs", "renderer", "camera_segmentations", "eval_mode"):
            env_kwargs.pop(k, None)
        return robosuite.make(
            has_renderer=False,
            use_camera_obs=True,
            renderer="mjviewer",
            control_freq=20,
            camera_names=CAMERA_NAMES,
            camera_widths=self.camera_size,
            camera_heights=self.camera_size,
            **env_kwargs,
        )

    def _warm_action(self, task_name: str) -> np.ndarray:
        if task_name not in self._warm_actions:
            import h5py

            with h5py.File(self._demo_path(task_name), "r") as f:
                first = sorted(f["data"].keys())[0]
                self._warm_actions[task_name] = np.asarray(f["data"][first]["actions"][0], dtype=np.float64)
        return self._warm_actions[task_name]

    @staticmethod
    def _build_proprio(raw_obs: Any) -> np.ndarray:
        """16-D world-frame ``robot_obs``: [L eef pos(3)+quat(4), R eef pos(3)+quat(4),
        L grip qpos[-1], R grip qpos[-1]] — byte-identical to the training key."""
        if "robot_obs" in raw_obs:
            return np.asarray(raw_obs["robot_obs"], dtype=np.float32)[:PROPRIO_DIM]
        parts = (
            raw_obs["robot0_left_eef_pos"],
            raw_obs["robot0_left_eef_quat"],
            raw_obs["robot0_right_eef_pos"],
            raw_obs["robot0_right_eef_quat"],
            np.asarray(raw_obs["robot0_left_gripper_qpos"])[-1:],
            np.asarray(raw_obs["robot0_right_gripper_qpos"])[-1:],
        )
        return np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])[:PROPRIO_DIM]
