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
    first = {}  # chunk k -> first protocol step it was dispensed
    off = 1 if (st.get(0) or {}).get("p") == -1 else 0  # live / initial_hold: protocol step 0 is harness step 1
    for step in range(n):
        r = st.get(step)
        j = step - off  # protocol step
        if j < 0:
            continue
        if r is None or "action" not in r:
            inst["missing_bench_row"] += 1
            continue
        if "intended_action" not in r:
            inst["missing_server_row"] += 1
            continue
        inst["steps"] += 1
        if ("infer_k" in r) != (j % s == 0):
            inst["cadence_violation"] += 1
        k = r.get("chunk_k")
        if k is not None and k not in first:
            first[k] = j
        same = np.allclose(r["action"], r["intended_action"], atol=1e-6)
        if not same:
            inst["stale"] += 1
        elif "dispense_wall" in r and "t_wall" in r:
            lag.append(r["t_wall"] - r["dispense_wall"])
        if "infer_ms" in r:
            infer_ms.append(r["infer_ms"])
    for k, j in first.items():
        if k < 0:
            continue
        t = k * s
        eff[j - t - d] += 1
        lw = st[j + off].get("landed_wall")
        if a.live and lw is not None and d > 0:  # landed between obs t+d-1 and obs t+d admissions
            prev_adm = st.get(t + d - 1 + off, {}).get("admitted_wall")
            if prev_adm is not None:
                margin_lo.append(lw - prev_adm)
            margin_hi.append(st[j + off]["admitted_wall"] - lw)
    ks = sorted(k for k in first if k >= 0)
    expected = {k for k in range(0, (n - off - 1) // s + 1) if k * s + d < n - off}
    inst["boundary_never_dispensed"] += len(expected - set(ks))

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
for k in ("step_time_max", "stale_action_ratio"):
    if rt.get(k):
        out[f"rt_{k}_max"] = float(np.max(rt[k]))
        out[f"rt_{k}_mean"] = float(np.mean(rt[k]))
if rt.get("elapsed_sec") is None and rt:
    pass
print(json.dumps(out, indent=1))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
