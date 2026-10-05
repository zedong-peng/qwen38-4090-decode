#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Steady-state round budget from an nsys sqlite: kernels between the first and last recurrent_fold_staged_kernel,
per round: wall span, busy (union of kernel intervals), idle gaps, and busy time by kernel class.
usage: round_budget.py SQLITE"""
import os, re, sqlite3, sys
MARKER = os.environ.get("ROUND_MARKER", "recurrent_fold_staged_kernel")  # one call per decode round
from collections import defaultdict
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select k.start, k.end, s.value, k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k "
                  "join StringIds s on s.id = k.shortName order by k.start").fetchall()
folds = [r[0] for r in rows if r[2] == MARKER]
t0, t1 = folds[0], folds[-1]
rounds = len(folds) - 1
ks = [r for r in rows if t0 <= r[0] < t1]
busy = 0; cur_s = cur_e = None
for s, e, n, g in ks:
    if cur_e is None or s > cur_e:
        if cur_e is not None: busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
cls = defaultdict(float); cnt = defaultdict(int)
def klass(n, g):
    if n.startswith(("ada_small_t", "q4_rowsplit", "q8_ksplit", "q4_", "q8_")): return "gemm"
    if n == "screen_kernel": return "head screen"
    if "fold" in n: return "fold"
    if "record" in n: return "gdn record"
    if "attention" in n: return "attention"
    if "rmsnorm" in n or "norm" in n: return "norms"
    if "conv" in n: return "conv"
    return "other"
for s, e, n, g in ks:
    c = klass(n, g); cls[c] += e - s; cnt[c] += 1
span = t1 - t0
print(f"rounds {rounds}  span/round {span/rounds/1e3:.1f} us  busy {busy/rounds/1e3:.1f}  idle {(span-busy)/rounds/1e3:.1f}  kernels/round {len(ks)/rounds:.0f}")
for c, v in sorted(cls.items(), key=lambda x: -x[1]):
    print(f"  {c:12s} {v/rounds/1e3:8.1f} us  {cnt[c]/rounds:6.1f} calls")
other = defaultdict(float); oc = defaultdict(int)
for s, e, n, g in ks:
    if klass(n, g) == "other": other[n] += e - s; oc[n] += 1
print("other:")
for n, v in sorted(other.items(), key=lambda x: -x[1])[:25]:
    print(f"  {v/rounds/1e3:7.1f} us {oc[n]/rounds:5.1f}  {n[:70]}")
