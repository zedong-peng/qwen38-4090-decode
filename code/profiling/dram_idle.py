#!/usr/bin/env python3
"""DRAM-idle opportunity per verify round from an nsys sqlite: time in DRAM-streaming kernels (weight GEMMs, head
screens, fold) vs latency-bound kernels vs gaps, and the run lengths of non-streaming stretches between streaming
kernels. usage: dram_idle.py DB"""
import sqlite3, sys, collections, statistics
db = sqlite3.connect(sys.argv[1])
rows = db.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName "
                  "order by k.start").fetchall()
fold = [i for i, r in enumerate(rows) if "recurrent_fold_staged" in r[2]]
STREAM = ("ada_small_t_mma", "screen_kernel", "recurrent_fold", "q8_ksplit", "context_kv_grouped")
rounds = []
for a, b in zip(fold[2:-1], fold[3:]):
    rd = rows[a + 1:b + 1]
    t0, t1 = rd[0][0], rd[-1][1]
    stream = lat = 0.0
    stretches = []  # non-streaming stretch lengths (latency kernels + gaps) between streaming kernels
    cur = 0.0; prev_end = t0
    by = collections.Counter()
    for st, en, name in rd:
        gap = max(0, st - prev_end) / 1000
        dur = (en - st) / 1000
        if any(k in name for k in STREAM):
            stream += dur; cur += gap
            if cur > 0: stretches.append(cur)
            cur = 0.0
        else:
            lat += dur; cur += gap + dur; by[name] += dur
        prev_end = max(prev_end, en)
    if cur > 0: stretches.append(cur)
    span = (t1 - t0) / 1000
    rounds.append((span, stream, lat, span - stream - lat, stretches, by))
n = len(rounds)
span = statistics.median(r[0] for r in rounds); stream = statistics.median(r[1] for r in rounds)
lat = statistics.median(r[2] for r in rounds); gaps = statistics.median(r[3] for r in rounds)
print(f"{n} rounds: span {span:.0f} us = streaming kernels {stream:.0f} + latency-bound kernels {lat:.0f} + gaps {gaps:.0f}")
allst = [s for r in rounds for s in r[4]]
allst.sort()
tot = sum(allst) / n
print(f"non-streaming stretches per round: {len(allst)/n:.0f}, total {tot:.0f} us; median {statistics.median(allst):.1f} us")
for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 1e9)):
    sel = [s for s in allst if lo <= s < hi]
    print(f"  {lo:>4}-{hi if hi < 1e9 else 'inf':>4} us: {len(sel)/n:6.1f} per round, {sum(sel)/n:7.1f} us")
by = collections.Counter()
for r in rounds: by.update(r[5])
print("top latency-bound kernels (us/round):")
for name, v in by.most_common(14): print(f"  {v/n:7.1f}  {name[:70]}")
