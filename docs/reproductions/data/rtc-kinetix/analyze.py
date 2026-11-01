"""Per-level success + protocol instruments from one vla-eval recording DB (arm C or L).

usage: python ana.py <recording.sqlite> <d> <s> [--live] [--json out.json]
"""

import argparse
import collections
import json
import sqlite3

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("db")
p.add_argument("d", type=int)
p.add_argument("s", type=int)
p.add_argument("--live", action="store_true")
p.add_argument("--json")
p.add_argument("--hz", type=float, default=10.0)
a = p.parse_args()
d, s = a.d, a.s
c = sqlite3.connect(a.db)
eps = c.execute("SELECT sid, eid, context, status, metrics, steps FROM episode_results").fetchall()
lvl = collections.defaultdict(lambda: [0, 0])
errors = collections.Counter()
steps_by_ep = {}
rt = collections.defaultdict(list)
for sid, eid, ctx, status, metrics, steps in eps:
    ctx = json.loads(ctx)
    if status == "error":
        errors[ctx["level"]] += 1
        continue
    succ = bool(json.loads(metrics).get("success"))
    lvl[ctx["level"]][0] += succ
    lvl[ctx["level"]][1] += 1
    steps_by_ep[(sid, eid)] = steps
    for k, v in (ctx.get("rt_metrics") or {}).items():
        rt[k].append(v)

inst = collections.Counter()
eff = collections.Counter()  # effective delay - d, per boundary
lag, margin_lo, margin_hi, infer_ms = [], [], [], []
rows = collections.defaultdict(dict)
for sid, eid, step, fields in c.execute("SELECT sid, eid, step_id, fields FROM step_rows"):
    rows[(sid, eid)][step] = json.loads(fields)
dt = 1.0 / a.hz
for key, st in rows.items():
    n = steps_by_ep.get(key)
    if n is None:
        continue
    first = {}  # chunk k -> (first protocol step it was dispensed, its harness step)
    adm = {}  # protocol step -> admission of its latest observation
    for step in sorted(st):
        if "p" in st[step] and st[step]["p"] >= 0:
            adm[st[step]["p"]] = st[step]["admitted_wall"]
    infer_steps = {r["p"] if "p" in r else h for h, r in st.items() if "infer_k" in r}
    last_j = -1
    for step in range(n):
        r = st.get(step) or {}
        if r.get("pre_start"):
            inst["pre_start_ticks"] += 1
            continue
        if "action" not in r:
            inst["missing_bench_row"] += 1
            continue
        if "intended_action" not in r:
            if r.get("p", 0) >= 0:
                inst["missing_server_row"] += 1
            continue
        j = r.get("p", step)  # protocol step (the server's), else the harness index
        if j < 0:
            continue
        inst["steps"] += 1
        last_j = max(last_j, j)
        k = r.get("chunk_k")
        if k is not None and k not in first:
            first[k] = (j, step)
        if not np.allclose(r["action"], r["intended_action"], atol=1e-6):
            inst["stale"] += 1
        elif "dispense_wall" in r and "t_wall" in r:
            lag.append(r["t_wall"] - r["dispense_wall"])
        if "infer_ms" in r:
            infer_ms.append(r["infer_ms"])
    inst["cadence_violation"] += len(infer_steps ^ {j for j in range(last_j + 1) if j % s == 0})
    for k, (j, h) in first.items():
        if k < 0:
            continue
        t = k * s
        eff[j - t - d] += 1
        lw = st[h].get("landed_wall")
        if a.live and lw is not None and d > 0:  # landed between the admissions of obs t+d-1 and obs t+d
            if t + d - 1 in adm:
                margin_lo.append(lw - adm[t + d - 1])
            margin_hi.append(st[h]["admitted_wall"] - lw)
    expected = {k for k in range(last_j // s + 1) if k * s + d <= last_j}
    inst["boundary_never_dispensed"] += len(expected - {k for k in first if k >= 0})

out = {"levels": {}, "errors": dict(errors)}
tot = [0, 0]
for name in sorted(lvl):
    k, m = lvl[name]
    out["levels"][name] = {"solved": k, "n": m, "p": k / m}
    tot[0] += k
    tot[1] += m
out["avg"] = float(np.mean([v["p"] for v in out["levels"].values()])) if lvl else None
out["n_total"] = tot[1]
out["instruments"] = dict(inst)
nb = sum(eff.values())
out["eff_delay_hist"] = {str(k): v for k, v in sorted(eff.items())}
out["eff_delay_exact_frac"] = eff[0] / nb if nb else None
if inst["steps"]:
    out["stale_frac"] = inst["stale"] / inst["steps"]
if lag:
    out["dispense_to_apply_ms"] = {
        q: float(np.percentile(lag, x) * 1e3) for q, x in (("p50", 50), ("p99", 99), ("max", 100))
    }
if infer_ms:
    out["infer_ms"] = {q: float(np.percentile(infer_ms, x)) for q, x in (("p50", 50), ("p99", 99), ("max", 100))}
if margin_lo:
    out["land_after_prev_obs_ms_min"] = float(np.min(margin_lo) * 1e3)
if margin_hi:
    out["land_before_obs_ms_min"] = float(np.min(margin_hi) * 1e3)
if rt.get("step_time_max"):
    out["eps_step_over_half_period"] = int(sum(x > 0.5 / a.hz for x in rt["step_time_max"]))
for k in ("step_time_max", "stale_action_ratio"):
    if rt.get(k):
        out[f"rt_{k}_max"] = float(np.max(rt[k]))
        out[f"rt_{k}_mean"] = float(np.mean(rt[k]))
if rt.get("elapsed_sec") is None and rt:
    pass
print(json.dumps(out, indent=1))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
