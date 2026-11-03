"""Time RoboTwin's per-step components inside the robotwin image: setup, physics (take_action) and render (get_obs).

    python scripts/profile/robotwin_step_timing.py --task beat_block_hammer --steps 30 --variant rt32,rt8,rt4,raster

Variants change the camera shader before the scene is built: rtN = RoboTwin's ray tracer at N samples per pixel
(32 is the shipped setting); raster = SAPIEN's default rasterizer. Runs one process per variant (the shader binds
at scene creation), so pass one --variant per invocation from the shell loop in the sbatch.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np


def patch_shader(variant: str):
    import sapien.render as r

    orig = {"dir": r.set_camera_shader_dir, "spp": r.set_ray_tracing_samples_per_pixel}
    orig_denoiser = r.set_ray_tracing_denoiser
    import torch

    if not torch.cuda.is_available():  # OIDN's denoiser is CUDA-only
        r.set_ray_tracing_denoiser = lambda n: orig_denoiser("none")
    if variant == "raster":
        r.set_camera_shader_dir = lambda d: orig["dir"]("default")
        r.set_ray_tracing_samples_per_pixel = lambda n: None
        r.set_ray_tracing_path_depth = lambda n: None
        r.set_ray_tracing_denoiser = lambda n: None
    elif variant.startswith("rt"):
        spp = int(variant[2:])
        r.set_ray_tracing_samples_per_pixel = lambda n: orig["spp"](spp)
    elif variant != "shipped":
        raise SystemExit(f"unknown variant {variant}")


def stub_curobo_without_gpu() -> None:
    """RoboTwin's planner.py defines CuroboPlanner inside a try that needs CUDA; robot.py imports the name
    unconditionally. With no GPU, load planner.py with a placeholder class so eval (which never plans) can import."""
    import importlib.abc
    import importlib.machinery
    import importlib.util
    import sys

    import torch

    if torch.cuda.is_available():
        return

    class Loader(importlib.abc.Loader):
        def __init__(self, path: str) -> None:
            self.path = path

        def exec_module(self, module):  # noqa: ANN001
            src = open(self.path).read() + "\n\nif 'CuroboPlanner' not in globals():\n    class CuroboPlanner: ...\n"
            exec(compile(src, self.path, "exec"), module.__dict__)

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):  # noqa: ANN001
            if fullname != "envs.robot.planner" or not path:
                return None
            file = f"{list(path)[0]}/planner.py"
            return importlib.util.spec_from_file_location(fullname, file, loader=Loader(file))

    sys.meta_path.insert(0, Finder())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="beat_block_hammer")
    p.add_argument("--config", default="demo_clean")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--variant", default="shipped")
    p.add_argument("--episodes", type=int, default=2)
    a = p.parse_args()

    from vla_eval.benchmarks.robotwin.benchmark import RoboTwinBenchmark

    stub_curobo_without_gpu()
    patch_shader(a.variant)
    b = RoboTwinBenchmark(task_name=a.task, task_config=a.config, seed=0, test_num=a.episodes, skip_expert_check=True)
    from vla_eval.recording import NullEpisodeRecorder

    b._recorder = NullEpisodeRecorder()
    tasks = b.get_tasks()
    out = {"variant": a.variant, "task": a.task, "episodes": []}
    for task in tasks:
        t0 = time.perf_counter()
        obs = b.reset(task)
        t_setup = time.perf_counter() - t0
        env = b._env
        phys, rend = [], []
        for _ in range(a.steps):
            q = np.asarray(obs["joint_action"]["vector"], np.float64)
            t1 = time.perf_counter()
            env.take_action(q, action_type="qpos")
            t2 = time.perf_counter()
            obs = env.get_obs()
            t3 = time.perf_counter()
            phys.append(t2 - t1)
            rend.append(t3 - t2)
        img = obs["observation"]["head_camera"]["rgb"]
        out["episodes"].append(
            {
                "setup_s": round(t_setup, 2),
                "physics_s_mean": round(float(np.mean(phys)), 4),
                "render_s_mean": round(float(np.mean(rend)), 4),
                "render_s_p90": round(float(np.percentile(rend, 90)), 4),
                "step_s": round(float(np.mean(phys) + np.mean(rend)), 4),
                "img_shape": list(img.shape),
                "img_mean": round(float(img.mean()), 2),
            }
        )
        with contextlib.suppress(Exception):
            np.save(f"/workspace/results/frame-{a.variant}-{a.task}-ep{task['episode_idx']}.npy", img)
        print(json.dumps(out["episodes"][-1]), flush=True)
    print("RESULT " + json.dumps(out), flush=True)
    b.cleanup()


if __name__ == "__main__":
    main()
