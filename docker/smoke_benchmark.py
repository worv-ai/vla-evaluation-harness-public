"""Reset, apply one explicit action, and verify rendered observations without a model server."""

import argparse
import asyncio
import hashlib
import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml

from vla_eval.recording import EpisodeRecorder, RecordingStore


async def main(config, action, render=None):
    with open(config) as f:
        document = yaml.safe_load(f)
    cfg = document["benchmarks"][0]
    module, name = cfg["benchmark"].split(":")
    benchmark_type = getattr(importlib.import_module(module), name)
    benchmark_type.configure_render(render or document.get("render", "gpu"))
    bench = benchmark_type(**cfg.get("params", {}))
    scratch = tempfile.TemporaryDirectory(prefix="vla-smoke-")
    store = RecordingStore(Path(scratch.name) / "smoke.sqlite")
    recorder = EpisodeRecorder(
        store=store,
        sid="smoke",
        eid="0",
        eval_id="smoke",
        output_dir=scratch.name,
        filename_stem="smoke",
        context={},
        record_video=False,
    )
    try:
        task = dict(bench.get_tasks()[0], episode_idx=0)
        await bench.start_episode(task, recorder=recorder)
        await bench.get_observation()
        await bench.apply_action({"actions": np.asarray(action, dtype=np.float32)})
        after = await bench.get_observation()
        frames = {}

        def inspect(value, path=""):
            if isinstance(value, dict):
                for key, item in value.items():
                    inspect(item, path + "/" + str(key))
            elif isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in (3, 4):
                frames[path] = {
                    "shape": list(value.shape),
                    "std": float(value.std()),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }

        inspect(after)
        assert frames and any(x["std"] > 1 for x in frames.values()), frames
        print(json.dumps({"benchmark": name, "frames": frames, "done": bool(await bench.is_done())}))
    finally:
        try:
            bench.cleanup()
        finally:
            recorder.close(status="fail", metrics={"smoke_only": True})
            store.close()
            scratch.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Evaluation YAML; uses its first benchmark and task")
    parser.add_argument("--action", required=True, type=json.loads, help="JSON action vector for this simulator")
    parser.add_argument("--render", choices=("cpu", "gpu"), help="Override the config render backend")
    args = parser.parse_args()
    asyncio.run(main(args.config, args.action, args.render))
