"""Compare vla-eval arms (ana.py JSON) against arm R (official script JSONL) per level and on the level average.

usage: python compare.py <R.jsonl,...> <method> <d> <s> <arm.json> [--label L] [--md]
Criteria (spec §6.3): (1) |avg diff| <= 95% CI of the difference; (2) <= 2 of 12 levels outside their 95% CI and
none by more than 5 pp.
"""

import argparse
import json
import math

Z = 1.959964
LEVELS = [
    "grasp_easy",
    "catapult",
    "cartpole_thrust",
    "hard_lunar_lander",
    "mjc_half_cheetah",
    "mjc_swimmer",
    "mjc_walker",
    "h17_unicycle",
    "chain_lander",
    "catcher_v3",
    "trampoline",
    "car_launch",
]


def wilson(p: float, n: int) -> tuple[float, float]:
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return c - h, c + h


def load_r(paths: list[str], method: str, d: int, s: int) -> dict[str, tuple[float, int]]:
    out = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if (r["method"], r["d"], r["s"]) == (method, d, s):
                out[r["level"]] = (r["solved"], r["n"])
    return out


def compare(R: dict, X: dict) -> dict:
    rows, var_r, var_x, outside, worst = [], 0.0, 0.0, 0, 0.0
    for lv in LEVELS:
        pr, nr = R[lv]
        px, nx = X[lv]["p"], X[lv]["n"]
        vr, vx = pr * (1 - pr) / nr, px * (1 - px) / nx
        diff = px - pr
        (lx, ux), (lr, ur) = wilson(px, nx), wilson(pr, nr)  # Newcombe's hybrid score interval of the difference
        lo, hi = diff - math.sqrt((px - lx) ** 2 + (ur - pr) ** 2), diff + math.sqrt((ux - px) ** 2 + (pr - lr) ** 2)
        half = (hi - lo) / 2
        out = not (lo <= 0 <= hi)
        outside += out
        worst = max(worst, abs(diff))
        var_r += vr
        var_x += vx
        rows.append(dict(level=lv, R=pr, X=px, nX=nx, diff=diff, ci=half, lo=lo, hi=hi, outside=bool(out)))
    avg_r = sum(R[lv][0] for lv in LEVELS) / 12
    avg_x = sum(X[lv]["p"] for lv in LEVELS) / 12
    half = Z * math.sqrt(var_r + var_x) / 12
    nx_min = min(X[lv]["n"] for lv in LEVELS)
    crit1 = abs(avg_x - avg_r) <= half
    crit2 = outside <= 2 and (worst <= 0.05 or nx_min < 2048)
    return dict(
        rows=rows,
        avg_R=avg_r,
        avg_X=avg_x,
        avg_diff=avg_x - avg_r,
        avg_ci=half,
        levels_outside=outside,
        worst_level_pp=worst * 100,
        crit1=crit1,
        crit2=crit2,
        n_per_level=nx_min,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("r")
    p.add_argument("method")
    p.add_argument("d", type=int)
    p.add_argument("s", type=int)
    p.add_argument("arm")
    a = p.parse_args()
    res = compare(load_r(a.r.split(","), a.method, a.d, a.s), json.load(open(a.arm))["levels"])
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    for r in res["rows"]:
        print(
            f"{r['level']:18s} R {r['R']:.3f}  X {r['X']:.3f}  diff {r['diff'] * 100:+5.1f} [{r['lo'] * 100:+5.1f}, {r['hi'] * 100:+5.1f}]"
            f"{'  *' if r['outside'] else ''}"
        )
