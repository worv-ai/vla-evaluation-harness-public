"""MIKASA-Robo benchmark implementation.

Memory-intensive robotic manipulation tasks built on ManiSkill3/SAPIEN.
32 tasks across categories: remember, shell-game, rotate, intercept, etc.

Key details:
- ManiSkill3 gymnasium API with obs_mode="rgb", Panda robot
- StateOnlyTensorToDictWrapper required for observations
- Action: 8D (7 joint deltas + 1 gripper) for default pd_joint_delta_pos
- Success: info["success"] (batched tensor, index [0] for single env)
- max_episode_steps varies per task (60–180)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.benchmarks.data_recording import EpisodeRecorder, RecordingConfig
from vla_eval.specs import GRIPPER_RAW, IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

os.environ.setdefault("DISPLAY", "")

# Representative subset — one task per category
DEFAULT_TASKS = [
    "RememberColor3-v0",
    "ShellGameTouch-v0",
    "RotateLenientPos-v0",
    "InterceptSlow-v0",
    "TakeItBack-v0",
]

# Human-readable descriptions
TASK_DESCRIPTIONS: dict[str, str] = {
    "RememberColor3-v0": "Remember the color of the target cube, then touch it after it reappears",
    "RememberColor5-v0": "Remember the target color among 5 cubes",
    "RememberColor9-v0": "Remember the target color among 9 cubes",
    "RememberShape3-v0": "Remember the shape of the target object",
    "RememberShape6-v0": "Remember the target shape among 6 objects",
    "RememberShape9-v0": "Remember the target shape among 9 objects",
    "ShellGameTouch-v0": "Track a ball hidden under shuffling cups, then touch the correct cup",
    "ShellGamePush-v0": "Track and push the cup hiding the ball",
    "ShellGamePick-v0": "Track and pick up the cup hiding the ball",
    "RotateLenientPos-v0": "Rotate object to the target angle (lenient, positive only)",
    "RotateLenientPosNeg-v0": "Rotate object to the target angle (lenient, both directions)",
    "RotateStrictPos-v0": "Rotate object to exact target angle (strict, positive)",
    "RotateStrictPosNeg-v0": "Rotate object to exact target angle (strict, both)",
    "InterceptSlow-v0": "Intercept a slow-moving object",
    "InterceptMedium-v0": "Intercept a medium-speed object",
    "InterceptFast-v0": "Intercept a fast-moving object",
    "InterceptGrabSlow-v0": "Grab a slow-moving object",
    "InterceptGrabMedium-v0": "Grab a medium-speed object",
    "InterceptGrabFast-v0": "Grab a fast-moving object",
    "TakeItBack-v0": "Pick up object and return it to its original position",
    "BunchOfColors3-v0": "Touch all cubes of the target color in a bunch of 3",
    "BunchOfColors5-v0": "Touch all cubes of the target color in a bunch of 5",
    "BunchOfColors7-v0": "Touch all cubes of the target color in a bunch of 7",
    "SeqOfColors3-v0": "Touch 3 colored cubes in the memorised sequence",
    "SeqOfColors5-v0": "Touch 5 colored cubes in sequence",
    "SeqOfColors7-v0": "Touch 7 colored cubes in sequence",
    "ChainOfColors3-v0": "Follow a chain of 3 colors",
    "ChainOfColors5-v0": "Follow a chain of 5 colors",
    "ChainOfColors7-v0": "Follow a chain of 7 colors",
    "RememberShapeAndColor3x2-v0": "Remember shape and color (3 shapes x 2 colors)",
    "RememberShapeAndColor3x3-v0": "Remember shape and color (3x3)",
    "RememberShapeAndColor5x3-v0": "Remember shape and color (5x3)",
}


class MIKASABenchmark(StepBenchmark):
    """MIKASA-Robo memory-intensive manipulation benchmark.

    Args:
        recording: A ``RecordingConfig`` dict (or ``None`` to disable).
            Controls per-episode video + JSONL data recording.
    """

    _ALL_RECORD_FIELDS = frozenset({"reward", "terminated", "truncated", "success"})

    def __init__(
        self,
        tasks: list[str] | None = None,
        episodes_per_task: int = 10,
        max_episode_steps: int | None = None,
        render_resolution: list[int] | tuple[int, int] = (256, 256),
        recording: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._task_names = tasks or DEFAULT_TASKS
        self._max_steps_override = max_episode_steps
        self._render_resolution = tuple(render_resolution)
        self._env: Any = None
        self._current_task: str | None = None
        self._task_desc: str = ""

        rec = RecordingConfig(**recording) if recording else None
        if rec and rec.step_fields:
            unknown = set(rec.step_fields) - self._ALL_RECORD_FIELDS
            if unknown:
                raise ValueError(f"Unknown step_fields: {unknown}. Valid: {sorted(self._ALL_RECORD_FIELDS)}")
        self._record_fields: set[str] = (
            set(rec.step_fields) if rec and rec.step_fields else set(self._ALL_RECORD_FIELDS)
        )
        self._recorder: EpisodeRecorder | None = (
            EpisodeRecorder(
                output_dir=rec.output_dir,
                record_video=rec.record_video,
                record_step=rec.record_step,
                fps=rec.fps,
            )
            if rec
            else None
        )
        self._step_counter: int = 0

    def cleanup(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None
        if self._recorder is not None:
            self._recorder.discard()

    def get_tasks(self) -> list[Task]:
        return [{"name": t, "env_id": t} for t in self._task_names]

    def reset(self, task: Task) -> Any:
        import gymnasium as gym
        import mikasa_robo_suite  # noqa: F401 — registers envs

        env_name = task["name"]

        if self._env is None or self._current_task != env_name:
            if self._env is not None:
                self._env.close()
            self._env = gym.make(
                env_name,
                num_envs=1,
                obs_mode="rgb",
                render_mode="cameras",
            )
            from mikasa_robo_suite.utils.wrappers import StateOnlyTensorToDictWrapper

            self._env = StateOnlyTensorToDictWrapper(self._env)
            self._current_task = env_name

        obs, info = self._env.reset()
        self._task_desc = TASK_DESCRIPTIONS.get(env_name, f"Complete {env_name}")

        if self._recorder is not None:
            self._recorder.start({"env_id": task["env_id"], "episode_idx": task.get("episode_idx", 0)})
            frame = self._extract_frame(obs)
            if frame is not None:
                self._recorder.record_frame(frame)
        self._step_counter = 0

        return obs

    def step(self, action: Action) -> StepResult:
        import torch

        raw = action.get("actions", action.get("action"))
        if raw is None:
            raw = np.zeros(8, dtype=np.float32)
        raw = np.asarray(raw, dtype=np.float32).flatten()
        assert raw.shape[-1] == 8, f"Action dimension mismatch: got {raw.shape[-1]}, expected 8"

        # Pad/truncate to match action space
        act_dim = self._env.action_space.shape[-1]
        if raw.shape[0] < act_dim:
            raw = np.pad(raw, (0, act_dim - raw.shape[0]))
        raw = raw[:act_dim]

        act_tensor = torch.from_numpy(raw).unsqueeze(0)
        obs, reward, terminated, truncated, info = self._env.step(act_tensor)

        term_bool = bool(terminated.any())
        trunc_bool = bool(truncated.any())
        done = term_bool or trunc_bool
        rew = float(reward.sum())
        success = bool(info.get("success", torch.tensor(False)).any())

        if self._recorder is not None and self._recorder.active:
            frame = self._extract_frame(obs)
            if frame is not None:
                self._recorder.record_frame(frame)
            fields = self._record_fields
            row: dict[str, Any] = {"step": self._step_counter}
            if "reward" in fields:
                row["reward"] = rew
            if "terminated" in fields:
                row["terminated"] = term_bool
            if "truncated" in fields:
                row["truncated"] = trunc_bool
            if "success" in fields:
                row["success"] = success
            self._recorder.record_step(row)
        self._step_counter += 1

        return StepResult(obs=obs, reward=rew, done=done, info={"success": success})

    def _extract_frame(self, obs: Any) -> np.ndarray | None:
        """Pull the first camera's RGB frame, normalising tensor / batched layouts."""
        if not isinstance(obs, dict):
            return None
        sensor = obs.get("sensor_data")
        if not isinstance(sensor, dict):
            return None
        for cam_data in sensor.values():
            if isinstance(cam_data, dict) and "rgb" in cam_data:
                img = cam_data["rgb"]
                if hasattr(img, "cpu"):
                    img = img.cpu().numpy()
                img = np.asarray(img)
                if img.ndim == 4:
                    img = img[0]
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                return img
        return None

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        images: dict[str, np.ndarray] = {}
        if isinstance(raw_obs, dict) and "sensor_data" in raw_obs:
            for cam_name, cam_data in raw_obs["sensor_data"].items():
                if "rgb" in cam_data:
                    img = cam_data["rgb"]
                    if hasattr(img, "cpu"):
                        img = img.cpu().numpy()
                    if img.ndim == 4:
                        img = img[0]
                    images[cam_name] = img
        if not images:
            images["base_camera"] = np.zeros((*self._render_resolution, 3), dtype=np.uint8)
        return {"images": images, "task_description": self._task_desc}

    def check_done(self, step_result: StepResult) -> bool:
        return step_result.done

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        success = bool(step_result.info.get("success", False))
        if self._recorder is not None:
            self._recorder.save(status="success" if success else "fail")
        return {"success": success}

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self._max_steps_override or 90}

    def get_action_spec(self) -> dict[str, DimSpec]:
        # 8D: 7 joint deltas + 1 gripper (pd_joint_delta_pos)
        return {
            "joints": DimSpec("joints", 7, "joint_delta_pos"),
            "gripper": GRIPPER_RAW,
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        return {
            "base_camera": IMAGE_RGB,
            "language": LANGUAGE,
        }
