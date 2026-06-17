"""RoboCasa-DC cross-embodiment benchmark (SeeTraceAct, arXiv 2606.02745).

24 RoboCasa kitchen tasks evaluated cross-embodiment: a GR-1 humanoid demo (loaded
by the *model server*) conditions a Panda-arm (PandaOmron) rollout. Each eval episode
restores a predefined scene from ``{predefined_envs_root}/{task}/{seed}/`` (loose
``environment.xml`` + ``ep_meta.json`` + ``initial_state.npz``); the integer ``seed``
doubles as the GR-1 demo index the server reads (``obs["episode_idx"]``).

This benchmark is the sim half of the eval; the demo conditioning lives entirely in the
reflex-dual model server. The benchmark reuses SeeTraceAct's own tested
``PredefinedRestoredRoboCasaEvalWrapper`` for the env restore + success check, so the
restore is byte-identical to the validated serial pipeline (CloseDrawer ~24% cog-on).

Obs ``state`` (53-d) and the 3 flipped camera views are pre-transformed here exactly as
SeeTraceAct's ``prepare_single_observation`` does (joint_pos = atan2(sin, cos);
quat xyzw->wxyz on the 3 quat keys) — the server consumes the ready vectors.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

import numpy as np

from vla_eval.benchmarks.base import StepBenchmark, StepResult
from vla_eval.specs import IMAGE_RGB, LANGUAGE, DimSpec
from vla_eval.types import Action, EpisodeResult, Observation, Task

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

# RoboCasa obs key -> GR00T video key (server reads these names, POLICY_CAMERAS order).
CAMERA_MAP: dict[str, str] = {
    "robot0_agentview_left_image": "video.ego_view",
    "robot0_agentview_right_image": "video.ego_view_right",
    "robot0_eye_in_hand_image": "video.hand_view",
}
CAMERA_MAP_ALT: dict[str, str] = {
    "robot0_agentview_left": "video.ego_view",
    "robot0_agentview_right": "video.ego_view_right",
    "robot0_eye_in_hand": "video.hand_view",
}
# 12 components, concatenated in THIS order -> the 53-d observation.state the policy trains on.
STATE_KEYS: list[str] = [
    "robot0_joint_pos",
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
    "robot0_base_pos",
    "robot0_base_quat",
    "robot0_base_to_eef_pos",
    "robot0_base_to_eef_quat",
]
QUAT_KEYS = {"robot0_eef_quat", "robot0_base_quat", "robot0_base_to_eef_quat"}
STATE_DIM = 53
ACTION_DIM = 12  # [arm_pos(3), arm_rot(3), gripper(1), base(3), torso(1), mode(1)]

# Canonical RoboCasa-DC task list (SeeTraceAct GROOT_EVAL_TASKS), tagged by the two unseen
# eval splits used for the Table-1 breakdown; the rest are "other".
CAT_BAL = {"CloseDrawer", "TurnOffSinkFaucet", "OpenDoubleDoor", "CoffeeServeMug", "PnPCounterToMicrowave"}
PREC_SENS = {"TurnOffStove", "CoffeePressButton", "TurnOffSinkFaucet", "PnPCounterToSink", "PnPStoveToCounter"}
GROOT_EVAL_TASKS = [
    "CloseDoubleDoor",
    "CloseDrawer",
    "CloseSingleDoor",
    "CoffeePressButton",
    "CoffeeServeMug",
    "CoffeeSetupMug",
    "OpenDoubleDoor",
    "OpenDrawer",
    "OpenSingleDoor",
    "PnPCabToCounter",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "PnPMicrowaveToCounter",
    "PnPSinkToCounter",
    "PnPStoveToCounter",
    "TurnOffMicrowave",
    "TurnOffSinkFaucet",
    "TurnOffStove",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "TurnOnStove",
    "TurnSinkSpout",
]


def _suite_of(task: str) -> str:
    if task in CAT_BAL:
        return "cat_bal"
    if task in PREC_SENS:
        return "prec_sens"
    return "other"


class RoboCasaDCBenchmark(StepBenchmark):
    """Cross-embodiment RoboCasa-DC: predefined-scene restore + Panda rollout.

    Args (config YAML ``params:``):
        predefined_envs_root: dir with ``<task>/<seed>/{environment.xml,ep_meta.json,initial_state.npz}``.
        robocasa_eval_py: path to SeeTraceAct's ``eval.py`` (reused for the tested restore wrapper).
        robots / controllers / camera_size: env-make knobs (PandaOmron / OSC_POSE / 128, the trained config).
        tasks: optional explicit task subset; defaults to all on-disk GROOT tasks.
        suite: optional filter "cat_bal" / "prec_sens" / "other".
        max_steps: episode horizon at 20 Hz (SeeTraceAct uses 300 for DC).
    """

    def __init__(
        self,
        predefined_envs_root: str,
        robocasa_eval_py: str,
        robots: str = "PandaOmron",
        controllers: str = "OSC_POSE",
        camera_size: int = 128,
        tasks: list[str] | None = None,
        suite: str | None = None,
        max_steps: int = 300,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.predefined_envs_root = predefined_envs_root
        self.robocasa_eval_py = robocasa_eval_py
        self.robots = robots
        self.controllers = controllers
        self.camera_size = camera_size
        self.tasks = tasks
        self.suite = suite
        self.max_steps = max_steps
        self._camera_names = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
        self._base_env: Any = None
        self._base_env_task: str | None = None
        self._wrapper: Any = None
        self._cur_seed: int = 0
        self._step_dbg: int = 0
        self._seed_cache: dict[str, list[int]] = {}
        self._eval_mod: Any = None
        self._create_eval_env: Any = None

    # --- required API ---

    def get_tasks(self) -> list[Task]:
        names = self.tasks or [
            t for t in GROOT_EVAL_TASKS if os.path.isdir(os.path.join(self.predefined_envs_root, t))
        ]
        tasks = [{"name": n, "suite": _suite_of(n)} for n in names]
        if self.suite:
            tasks = [t for t in tasks if t["suite"] == self.suite]
        return tasks

    def reset(self, task: Task) -> Any:
        name = task["name"]
        seeds = self._seeds_for(name)
        ep_idx = int(task.get("episode_idx", 0))
        if ep_idx >= len(seeds):
            raise IndexError(f"{name}: episode_idx {ep_idx} >= {len(seeds)} available seeds")
        seed = seeds[ep_idx]

        if self._base_env is None or self._base_env_task != name:
            self.cleanup()
            self._base_env = self._make_base_env(name)
            self._base_env_task = name
        self._cur_seed = seed
        self._step_dbg = 0

        wrapper_cls = self._eval_module().PredefinedRestoredRoboCasaEvalWrapper
        self._wrapper = wrapper_cls(
            self._base_env,
            env_name=name,
            predefined_envs_root=self.predefined_envs_root,
            seed_assignments=[seed],
            num_parallel_envs=1,
            env_idx=0,
        )
        return self._wrapper.reset()

    def step(self, action: Action) -> StepResult:
        raw_act = np.asarray(action["actions"], dtype=np.float64)
        act = raw_act.reshape(-1)[:ACTION_DIM].copy()
        if self._step_dbg < 5 or self._step_dbg in (16, 17):
            print(
                f"[stepdbg] step={self._step_dbg} raw_shape={raw_act.shape} act={np.round(act, 3).tolist()}",
                flush=True,
            )
        self._step_dbg += 1
        obs, reward, done, info = self._wrapper.step(act)
        success = bool(info.get("success", False))
        return StepResult(obs=obs, reward=float(reward), done=bool(done) or success, info={"success": success})

    def make_obs(self, raw_obs: Any, task: Task) -> Observation:
        return {
            "images": self._build_images(raw_obs),
            "state": self._build_state(raw_obs),
            "task_description": str(raw_obs.get("_task_description", "") or task["name"]),
            "task_name": task["name"],  # demo lookup (video_key_map_gr1) keys on the task NAME, not the lang
            "episode_idx": int(self._cur_seed),
        }

    def get_step_result(self, step_result: StepResult) -> EpisodeResult:
        return {"success": bool(step_result.info.get("success", False))}

    # --- optional API ---

    def get_action_spec(self) -> dict[str, DimSpec]:
        return {
            "actions": DimSpec(
                name="panda_omron_action",
                dims=ACTION_DIM,
                format="arm_pos3_rot3_grip1_base3_torso1_mode1",
                description="PandaOmron 12-D action; server emits a 16-step chunk, one applied per env step.",
            )
        }

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec: dict[str, DimSpec] = {f"images.{v}": IMAGE_RGB for v in CAMERA_MAP.values()}
        spec["state"] = DimSpec(
            name="robot_obs",
            dims=STATE_DIM,
            format="robocasa_dc_53d",
            description="12 GR00T state components concatenated; joint_pos=atan2(sin,cos), quats wxyz.",
        )
        spec["task_description"] = LANGUAGE
        return spec

    def get_metadata(self) -> dict[str, Any]:
        return {"max_steps": self.max_steps}

    def cleanup(self) -> None:
        self._wrapper = None
        if self._base_env is not None:
            try:
                self._base_env.close()
            finally:
                self._base_env = None
                self._base_env_task = None

    # --- helpers ---

    def _seeds_for(self, task_name: str) -> list[int]:
        if task_name not in self._seed_cache:
            base = os.path.join(self.predefined_envs_root, task_name)
            seeds = sorted(int(d) for d in os.listdir(base) if d.isdigit() and os.path.isdir(os.path.join(base, d)))
            if not seeds:
                raise FileNotFoundError(f"no seed dirs under {base}")
            self._seed_cache[task_name] = seeds
        return self._seed_cache[task_name]

    def _eval_module(self) -> Any:
        """Import SeeTraceAct eval.py once (its tested restore wrapper + helpers)."""
        if self._eval_mod is None:
            eval_dir = os.path.dirname(os.path.abspath(self.robocasa_eval_py))
            if eval_dir not in sys.path:
                sys.path.insert(0, eval_dir)
            spec = importlib.util.spec_from_file_location("seetraceact_eval", self.robocasa_eval_py)
            assert spec and spec.loader, f"cannot load {self.robocasa_eval_py}"
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._eval_mod = mod
        return self._eval_mod

    def _make_base_env(self, task_name: str) -> Any:
        # Importing SeeTraceAct eval.py monkeypatches robosuite 1.5 back to a load_controller_config
        # shim that robocasa.utils.eval_utils needs — must happen BEFORE that import.
        self._eval_module()
        from robocasa.utils.eval_utils import create_eval_env

        seed0 = self._seeds_for(task_name)[0]
        env = create_eval_env(
            env_name=task_name,
            robots=self.robots,
            controllers=self.controllers,
            camera_names=self._camera_names,
            camera_widths=self.camera_size,
            camera_heights=self.camera_size,
            seed=seed0,
        )
        # PandaOmron leaves robot0_joint_pos inactive (only cos/sin); the 53-d state needs raw.
        try:
            env.modify_observable("robot0_joint_pos", "active", True)
        except Exception as e:  # noqa: BLE001
            print(f"[robocasa_dc] could not activate robot0_joint_pos observable: {e}")
        return env

    @staticmethod
    def _build_images(raw_obs: Any) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for mapping in (CAMERA_MAP, CAMERA_MAP_ALT):
            for rk, gk in mapping.items():
                if rk in raw_obs and gk not in images:
                    img = np.asarray(raw_obs[rk])[::-1, :, :]  # live render is upside-down vs demos
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    images[gk] = np.ascontiguousarray(img)
        return images

    @staticmethod
    def _build_state(raw_obs: Any) -> np.ndarray:
        obs = raw_obs
        if "robot0_joint_pos" not in obs and "robot0_joint_pos_cos" in obs and "robot0_joint_pos_sin" in obs:
            obs = dict(obs)
            obs["robot0_joint_pos"] = np.arctan2(
                np.asarray(obs["robot0_joint_pos_sin"]), np.asarray(obs["robot0_joint_pos_cos"])
            )
        # NOTE: the LeRobot training parquet stores eef/base/base_to_eef quats as xyzw and the
        # RoboCasaDCEpisodeStore reads observation.state verbatim (no quat conversion), so the
        # model was trained on xyzw. RoboCasa's live obs are ALSO xyzw — feed them through
        # unchanged. (SeeTraceAct's reference eval.py converts xyzw->wxyz for its native GR00T
        # data-config which re-encodes to rotation_6d; reflex's raw-53D pipeline must NOT, or the
        # 3 orientation blocks get scrambled vs training and the flow head floors.)
        parts = []
        for key in STATE_KEYS:
            v = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            parts.append(v)
        state = np.concatenate(parts)
        assert state.shape[0] == STATE_DIM, f"assembled state {state.shape[0]}d, expected {STATE_DIM}"
        return state
