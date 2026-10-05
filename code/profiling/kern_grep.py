#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-call kernel durations from an nsys sqlite, by (short name, grid), for names matching REGEX: calls, median and
mean in us. Per-call times do not depend on how many rounds a run had, and an A/B of two kernel sets is read off the
medians without the session drift of end-to-end runs.
usage: kern_grep.py SQLITE REGEX [MIN_CALLS]"""
import re, sqlite3, statistics, sys
from collections import defaultdict
db = sqlite3.connect(sys.argv[1])
pat = re.compile(sys.argv[2])
min_calls = int(sys.argv[3]) if len(sys.argv) > 3 else 1
rows = db.execute("select s.value, k.gridX, k.gridY, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k "
                  "join StringIds s on s.id = k.shortName").fetchall()
d = defaultdict(list)
for n, gx, gy, t in rows:
    if pat.search(n):
        d[(n, gx, gy)].append(t / 1e3)
for (n, gx, gy), v in sorted(d.items(), key=lambda x: -statistics.median(x[1]) * len(x[1])):
    if len(v) < min_calls:
        continue
    print(f"{len(v):6d} calls  median {statistics.median(v):8.2f} us  mean {statistics.fmean(v):8.2f} us  "
          f"grid {gx}x{gy}  {n[:70]}")
