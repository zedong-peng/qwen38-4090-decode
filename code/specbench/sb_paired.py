#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired per-prompt comparison of two Spec-Bench runs over ALL prompts (outputs may differ): ms per round, tokens per
step and aggregate tok/s, with prompt-bootstrap 95% intervals (prompt difficulty cancels; content draws remain).
usage: sb_paired.py BASE.json RUN.json [BOOT=2000]"""
import json, sys
import numpy as np

def load(p):
    return {(r["group"], r["question_id"]): r for r in json.load(open(p))["rows"]}

b, r = load(sys.argv[1]), load(sys.argv[2]); nb = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
keys = sorted(k for k in b if k in r and b[k]["steps"] > 0 and r[k]["steps"] > 0)
def arr(d, f): return np.array([f(d[k]) for k in keys], dtype=np.float64)
tb, tr = arr(b, lambda x: x["completion_tokens"]), arr(r, lambda x: x["completion_tokens"])
sb, sr = arr(b, lambda x: x["steps"]), arr(r, lambda x: x["steps"])
eb, er = arr(b, lambda x: x["decode_window_s"]), arr(r, lambda x: x["decode_window_s"])
same = np.array([b[k]["content_sha256"] == r[k]["content_sha256"] for k in keys])
rng = np.random.default_rng(0); n = len(keys)
def boot(f):
    v = f(np.arange(n)); bs = [f(rng.integers(0, n, n)) for _ in range(nb)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return f"{100*(v-1):+.2f}% [{100*(lo-1):+.2f}, {100*(hi-1):+.2f}]"
print(f"{n} prompts, {same.sum()} identical outputs")
print("ms/round      ", boot(lambda i: (er[i].sum() / sr[i].sum()) / (eb[i].sum() / sb[i].sum())))
print("tok/step      ", boot(lambda i: (tr[i].sum() / sr[i].sum()) / (tb[i].sum() / sb[i].sum())))
print("tok/s         ", boot(lambda i: (tr[i].sum() / er[i].sum()) / (tb[i].sum() / eb[i].sum())))
d = np.where(~same)[0]
if len(d):  # the same ratio over the prompts whose outputs differ, resampled among those prompts
    def boot_d(f):
        v = f(d); bs = [f(d[rng.integers(0, len(d), len(d))]) for _ in range(nb)]
        lo, hi = np.percentile(bs, [2.5, 97.5]); return f"{100*(v-1):+.2f}% [{100*(lo-1):+.2f}, {100*(hi-1):+.2f}]"
    print(f"tok/step, {len(d)} differing", boot_d(lambda i: (tr[i].sum() / sr[i].sum()) / (tb[i].sum() / sb[i].sum())))
