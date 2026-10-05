#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drift-balanced A/B over a sequence of Spec-Bench runs (e.g. A B B A B A A B, which is orthogonal to linear and
quadratic drift). Per run: ms/round = sum(decode_window_s) / sum(steps) over the prompts common to all runs. Fit
ms = mu + X * [config B] + b1 * t + b2 * t^2 by least squares; report X as % of A, with a prompt bootstrap.
usage: sb_balanced.py A:run0.json B:run1.json ... [--boot 2000] [--choice FILE B_VALUE A_VALUE]"""
import json, sys
import numpy as np

args, boot, choice = [], 2000, None
it = iter(sys.argv[1:])
for a in it:
    if a == "--boot": boot = int(next(it))
    elif a == "--choice": choice = (next(it), next(it), next(it))
    else: args.append(a)
cfg = [a.split(":", 1)[0] for a in args]
runs = [{(r["group"], r["question_id"]): r for r in json.load(open(a.split(":", 1)[1]))["rows"]} for a in args]
keys = sorted(set.intersection(*[set(k for k, r in d.items() if r["steps"] > 0) for d in runs]))
E = np.array([[d[k]["decode_window_s"] for k in keys] for d in runs])  # runs x prompts
S = np.array([[d[k]["steps"] for k in keys] for d in runs], dtype=np.float64)
t = np.arange(len(runs), dtype=np.float64)
Xd = np.stack([np.ones_like(t), np.array([c == "B" for c in cfg], dtype=np.float64), t, t * t], 1)

def fit(idx):
    ms = E[:, idx].sum(1) / S[:, idx].sum(1) * 1e3
    coef = np.linalg.lstsq(Xd, ms, rcond=None)[0]
    return coef[1] / coef[0], ms

x, ms = fit(np.arange(len(keys)))
rng = np.random.default_rng(0)
bs = [fit(rng.integers(0, len(keys), len(keys)))[0] for _ in range(boot)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"{len(keys)} prompts, runs: " + " ".join(f"{c}{m:.4f}" for c, m in zip(cfg, ms)))
print(f"B vs A ms/round {100 * x:+.3f}% prompt bootstrap [{100 * lo:+.3f}, {100 * hi:+.3f}]  (linear + quadratic drift removed)")
# Run-level noise: residual standard error of the 4-parameter fit (df = runs - 4), t-quantile from a small table.
coef, res = np.linalg.lstsq(Xd, ms, rcond=None)[:2]
df = len(runs) - Xd.shape[1]
tq = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31}.get(df, 2.0)
s2 = float(res[0]) / df if df > 0 and len(res) else float("nan")
se = np.sqrt(s2 * np.linalg.inv(Xd.T @ Xd)[1, 1]) / coef[0]
print(f"B vs A ms/round {100 * x:+.3f}% run-level 95% [{100 * (x - tq * se):+.3f}, {100 * (x + tq * se):+.3f}]  (residual sd {100 * np.sqrt(s2) / coef[0]:.3f}%, df {df})")
if choice:
    f, bval, aval = choice
    win = x + tq * se < 0
    open(f, "w").write((bval if win else aval) + "\n")
    print(f"choice -> {bval if win else aval} ({'B' if win else 'A'}: {'run-level interval below 0' if win else 'not a clear win'})")
