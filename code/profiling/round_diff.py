#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-round kernel time difference between two nsys sqlite profiles (steady rounds between
recurrent_fold_staged_kernel calls): calls per round and us per round by (kernel name, grid), sorted by |delta|.
usage: round_diff.py A.sqlite B.sqlite [TOP=25]"""
import os, sqlite3, sys, statistics, collections
MARKER = os.environ.get("ROUND_MARKER", "recurrent_fold_staged_kernel")  # one call per decode round


def load(path):
    db = sqlite3.connect(path)
    rows = db.execute("select k.start, k.end, s.value, k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on s.id = k.shortName order by k.start").fetchall()
    fold = [i for i, r in enumerate(rows) if MARKER in r[2]]
    lo, hi = fold[len(fold) // 4], fold[-2]
    nr = sum(1 for i in fold if lo <= i < hi)
    by = collections.defaultdict(list)
    for i in range(lo, hi):
        by[(rows[i][2][:56], rows[i][3])].append((rows[i][1] - rows[i][0]) / 1e3)
    span = (rows[hi][0] - rows[lo][0]) / 1e3 / nr
    return nr, span, {k: (len(v) / nr, statistics.median(v), sum(v) / nr) for k, v in by.items()}


na, sa, a = load(sys.argv[1])
nb, sb, b = load(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
keys = set(a) | set(b)
rows = []
for k in keys:
    ca, ma, ta = a.get(k, (0, 0, 0))
    cb, mb, tb = b.get(k, (0, 0, 0))
    rows.append((tb - ta, k, ca, ma, ta, cb, mb, tb))
rows.sort(key=lambda r: -abs(r[0]))
print(f"A {na} rounds span {sa:.1f} us/round; B {nb} rounds span {sb:.1f} us/round; delta {sb - sa:+.1f}")
print(f"{'delta us/rd':>11} {'A calls':>7} {'A med':>7} {'B calls':>7} {'B med':>7}  grid  kernel")
for d, (n, g), ca, ma, ta, cb, mb, tb in rows[:top]:
    print(f"{d:+11.1f} {ca:7.1f} {ma:7.2f} {cb:7.1f} {mb:7.2f} {g:5d}  {n}")
