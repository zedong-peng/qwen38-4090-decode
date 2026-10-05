#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Where the steady-state round idles: gaps in the union of kernel and memcpy/memset intervals between the first and
last recurrent_fold_staged_kernel, attributed to (last activity before -> first activity after), summed per round.
Also the graph membership of the kernels around the largest gaps.
usage: gap_list.py SQLITE [TOP]"""
import os, sqlite3, sys
MARKER = os.environ.get("ROUND_MARKER", "recurrent_fold_staged_kernel")  # one call per decode round
from collections import defaultdict
db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
kcols = [r[1] for r in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
gsel = "k.graphId" if "graphId" in kcols else "0"
acts = [(s, e, n, g, "K") for s, e, n, g in db.execute(
    f"select k.start, k.end, s.value, {gsel} from CUPTI_ACTIVITY_KIND_KERNEL k "
    "join StringIds s on s.id = k.shortName")]
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
for t, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
    if t in tables:
        cols = [r[1] for r in db.execute(f"pragma table_info({t})")]
        kind = "copyKind" if "copyKind" in cols else None
        q = f"select start, end, {kind or 0}, {'graphId' if 'graphId' in cols else 0} from {t}"
        acts += [(s, e, f"{label}{'(' + str(k) + ')' if kind else ''}", g, "M") for s, e, k, g in db.execute(q)]
acts.sort()
folds = [a[0] for a in acts if a[2] == MARKER]
t0, t1 = folds[0], folds[-1]
rounds = len(folds) - 1
xs = [a for a in acts if t0 <= a[0] < t1]
gaps = defaultdict(float); cnt = defaultdict(int); total = 0.0
cur_e, cur_n, cur_g = None, None, None
for s, e, n, g, k in xs:
    if cur_e is not None and s > cur_e:
        key = (cur_n, cur_g, n, g)
        gaps[key] += s - cur_e; cnt[key] += 1; total += s - cur_e
    if cur_e is None or e > cur_e:
        cur_e, cur_n, cur_g = e, n, g
print(f"rounds {rounds}  idle/round {total / rounds / 1e3:.1f} us  ({len(gaps)} distinct boundaries)")
for (pn, pg, nn, ng), v in sorted(gaps.items(), key=lambda x: -x[1])[:top]:
    c = cnt[(pn, pg, nn, ng)]
    print(f"  {v / rounds / 1e3:7.1f} us/round  {c / rounds:5.2f}/round  {v / c / 1e3:7.1f} us each  "
          f"{pn[:38]}[g{pg}] -> {nn[:38]}[g{ng}]")
if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
    rt = defaultdict(float); rc = defaultdict(int)
    for s, e, n in db.execute("select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r "
                              "join StringIds s on s.id = r.nameId where r.start >= ? and r.start < ?", (t0, t1)):
        rt[n] += e - s; rc[n] += 1
    print("host runtime calls per round (time inside the call):")
    for n, v in sorted(rt.items(), key=lambda x: -x[1])[:12]:
        print(f"  {v / rounds / 1e3:8.1f} us  {rc[n] / rounds:6.2f}/round  {n[:60]}")
