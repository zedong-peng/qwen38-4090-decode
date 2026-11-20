#!/usr/bin/env python3
"""Mean GPU time per kernel name for prefill passes in token-size bins (nsys sqlite export).

usage: nsys_passkern.py prof.sqlite LO-HI [LO-HI ...]
"""
import bisect
import collections
import sqlite3
import sys

c = sqlite3.connect(sys.argv[1])
bins = [tuple(map(int, a.split("-"))) for a in sys.argv[2:]]
S = dict(c.execute("select id, value from StringIds"))
ch = collections.defaultdict(list)
for s, e, n, t, g, p in c.execute("select start, end, text, textId, globalTid, uint64Value from NVTX_EVENTS "
                                  "where eventType in (59, 60) and end is not null order by start"):
    if (n or S.get(t)) == "prefill.chunk":
        ch[g].append((s, e, p))
st = {g: [x[0] for x in v] for g, v in ch.items()}
rt = {cid: (s, g) for s, g, cid in c.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
per = collections.defaultdict(collections.Counter)  # chunk start -> kernel -> ns
cnt = collections.defaultdict(collections.Counter)
size = {}
for s, e, cid, nid in c.execute("select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    if cid not in rt:
        continue
    hs, g = rt[cid]
    v = ch.get(g)
    if not v:
        continue
    i = bisect.bisect_right(st[g], hs) - 1
    if i >= 0 and v[i][1] >= hs:
        per[v[i][0]][S.get(nid, "?")] += e - s
        cnt[v[i][0]][S.get(nid, "?")] += 1
        size[v[i][0]] = v[i][2]
for lo, hi in bins:
    keys = [k for k, p in size.items() if lo <= p <= hi]
    if not keys:
        continue
    agg, n_ = collections.Counter(), collections.Counter()
    for k in keys:
        agg.update(per[k])
        n_.update(cnt[k])
    tot = sum(agg.values()) / len(keys) / 1e6
    print(f"\npasses of {lo}..{hi} tokens: n={len(keys)}, mean tokens {sum(size[k] for k in keys) / len(keys):.0f}, "
          f"kernel time {tot:.1f} ms per pass")
    for name, ns in agg.most_common(14):
        print(f"  {ns / len(keys) / 1e6:7.2f} ms  {n_[name] / len(keys):6.0f} launches  {name[:90]}")
