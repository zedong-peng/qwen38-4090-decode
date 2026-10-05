#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Steady-state rounds of an nsys sqlite (between recurrent_fold_staged_kernel calls): every kernel name with calls
per round, median us, us per round. usage: round_kernels.py DB [MATCH]"""
import os, sqlite3, sys, statistics, collections
MARKER = os.environ.get("ROUND_MARKER", "recurrent_fold_staged_kernel")  # one call per decode round
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select k.start, k.end, s.value, k.gridX from CUPTI_ACTIVITY_KIND_KERNEL k "
                  "join StringIds s on s.id = k.shortName order by k.start").fetchall()
fold = [i for i, r in enumerate(rows) if MARKER in r[2]]
lo, hi = fold[len(fold) // 4], fold[-2]
nr = sum(1 for i in fold if lo <= i < hi)
by = collections.defaultdict(list)
for i in range(lo, hi):
    by[(rows[i][2][:58], rows[i][3])].append((rows[i][1] - rows[i][0]) / 1e3)
pat = sys.argv[2] if len(sys.argv) > 2 else ""
tot = 0.0
for (n, g), v in sorted(by.items(), key=lambda kv: -sum(kv[1])):
    if pat and pat not in n: continue
    tot += sum(v) / nr
    print(f"{len(v) / nr:6.1f}/round  med {statistics.median(v):7.2f} us  {sum(v) / nr:8.1f} us/round  grid {g:5d}  {n}")
print(f"{nr} rounds, listed total {tot:.1f} us/round")
