"""Summarize a profiling run: env steps/s, episode timings, shard balance and job load (run_profile.sh output)."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sqlite3
import statistics as st
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("eval_id")
    p.add_argument("--shards", type=int)
    p.add_argument("--wall", type=float, help="seconds from first shard launch to last shard exit")
    p.add_argument("--render", default="?")
    p.add_argument("--gpus", type=int, default=0)
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--render-every", type=int, default=1)
    a = p.parse_args()
    out = Path(a.out)

    db = out / f"recording-{a.eval_id}.sqlite"
    rows = []
    if db.exists():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        rows = con.execute("SELECT sid, task_name, status, steps, elapsed_sec FROM episode_results").fetchall()
    n_ep = len(rows)
    steps = sum(r[3] or 0 for r in rows)
    ep_sec = [r[4] for r in rows if r[4]]
    errors = sum(1 for r in rows if r[2] == "error")
    by_shard: dict[str, float] = {}
    for r in rows:
        by_shard[r[0]] = by_shard.get(r[0], 0.0) + (r[4] or 0.0)

    # shard wall from the logs: first "Starting benchmark" to the last episode line
    shard_walls = []
    ts = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
    for log in sorted(glob.glob(str(out / "logs" / "shard*.log"))):
        stamps = [m.group(1) for ln in open(log, errors="replace") if (m := ts.match(ln))]
        if len(stamps) >= 2:
            from datetime import datetime

            t0 = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(stamps[-1], "%Y-%m-%d %H:%M:%S")
            shard_walls.append((t1 - t0).total_seconds())

    load = {}
    lp = out / "logs" / "load.log"
    if lp.exists():
        samples = [ln.split() for ln in open(lp) if ln.startswith(tuple("0123456789"))]
        if len(samples) >= 2:

            def kv(s, k):  # noqa: E306
                return next((x.split("=", 1)[1] for x in s if x.startswith(k + "=")), "")

            t = [int(s[0]) for s in samples]
            cpu = [int(kv(s, "cpu_s") or 0) for s in samples]
            mem = [int(kv(s, "mem_gb") or 0) for s in samples]
            peak = [int(kv(s, "peak_gb") or 0) for s in samples]
            load = {
                "avg_cores": round((cpu[-1] - cpu[0]) / max(1, t[-1] - t[0]), 1),
                "mem_gb_p50": st.median(mem),
                "mem_gb_peak": max(peak + mem),
            }
            g = [kv(s, "gpu") for s in samples if kv(s, "gpu")]
            if g:
                utils, mems = [], []
                for s in g:
                    for dev in s.strip(";").split(";"):
                        parts = dev.split(",")
                        if len(parts) == 3:
                            utils.append(int(parts[1]))
                            mems.append(int(parts[2]))
                if utils:
                    load["gpu_util_p50"] = st.median(utils)
                    load["gpu_mem_mib_max"] = max(mems)

    summary = {
        "eval_id": a.eval_id,
        "shards": a.shards,
        "render": a.render,
        "render_gpus": a.gpus,
        "episodes": n_ep,
        "errors": errors,
        "env_steps": steps,
        "wall_s": a.wall,
        "steps_per_s": round(steps / a.wall, 2) if a.wall else None,
        "ep_s_mean": round(st.mean(ep_sec), 1) if ep_sec else None,
        "ep_s_p90": round(sorted(ep_sec)[int(0.9 * (len(ep_sec) - 1))], 1) if ep_sec else None,
        "step_s_in_episode": round(sum(ep_sec) / steps, 4) if steps and ep_sec else None,
        "shard_wall_min_max": [round(min(shard_walls)), round(max(shard_walls))] if shard_walls else None,
        "shard_busy_s_min_max": [round(min(by_shard.values())), round(max(by_shard.values()))] if by_shard else None,
        **load,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
