#!/usr/bin/env python3
"""Wall-time budget from arrival to the first decode round for each resumed request in an nsys sqlite export.

usage: nsys_budget.py prof.sqlite requests.jsonl [RATE_MS_PER_TOKEN]

A prefill pass is one prefill.chunk range plus the device.wait that follows it on the same thread. Its wall time is
split into the token-proportional part (tokens x RATE, default 0.443 ms, the cost per token of a 1,024-token pass in
this capture) and the excess. Gaps are the host time between passes (commits, state captures copied to the host).
"""
import bisect
import collections
import json
import sqlite3
import statistics
import sys

db, rqpath = sys.argv[1], sys.argv[2]
RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.443
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
T0 = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
nvtx = [(s, e, n or S.get(t, "?"), g, p) for s, e, n, t, g, p in c.execute(
    "select start, end, text, textId, globalTid, uint64Value from NVTX_EVENTS "
    "where eventType in (59, 60) and end is not null order by start")]
copies = sorted((s, e, k) for s, e, k in c.execute("select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"))
ms = lambda ns: ns / 1e6

recs = [json.loads(l) for l in open(rqpath)]
reqs = []
for d in recs:
    if d.get("event") != "request_done":
        continue
    tm, res = d["timings_seconds"], d["result"]
    done = d["timestamp_unix_ms"] * 1_000_000 - T0
    arr = done - int(tm["total"] * 1e9)
    if done > 0:
        reqs.append(dict(arr=arr, ttft=arr + int(tm["ttft"] * 1e9), computed=res["computed_prefill_tokens"],
                         prompt=res["prompt_tokens"], path=res["prefix_reuse_path"], t=tm))
reqs.sort(key=lambda r: r["arr"])
by_name = collections.defaultdict(list)
for r in nvtx:
    by_name[r[2]].append(r)
starts = {k: [r[0] for r in v] for k, v in by_name.items()}


def ranges(name, lo, hi):
    v, st = by_name.get(name, []), starts.get(name, [])
    return [r for r in v[bisect.bisect_left(st, lo):bisect.bisect_right(st, hi)]]


def copy_ms(lo, hi, kind):
    return sum(ms(min(e, hi) - max(s, lo)) for s, e, k in copies if k == kind and e > lo and s < hi)


rows = []
for i, r in enumerate(reqs):
    nxt = reqs[i + 1]["arr"] if i + 1 < len(reqs) else r["ttft"] + 10**9
    fp = ranges("frontend.prepare", r["arr"] - 5_000_000, r["arr"] + 50_000_000)
    dec = ranges("decode", r["arr"], nxt)
    if not fp or not dec:
        continue
    fp, first_decode = fp[0], dec[0][0]
    chunks = ranges("prefill.chunk", fp[1], first_decode)
    waits = ranges("device.wait", fp[1], first_decode)
    if not chunks:
        continue
    passes = []
    for ch in chunks:
        w = [x for x in waits if x[3] == ch[3] and x[0] >= ch[1]]
        end = w[0][1] if w else ch[1]
        passes.append((ch[0], end, ch[4]))
    gaps = sum(ms(b[0] - a[1]) for a, b in zip(passes, passes[1:]))
    tiny = [p for p in passes if p[2] <= 16]
    other = [p for p in passes if p[2] > 16]
    rows.append(dict(
        path=r["path"], computed=r["computed"], prompt=r["prompt"],
        total=ms(first_decode - r["arr"]), prepare=ms(fp[1] - fp[0]),
        before=ms(passes[0][0] - fp[1]), h2d=copy_ms(fp[1], passes[0][0], 1),
        n=len(passes), sizes=[p[2] for p in passes],
        tiny_n=len(tiny), tiny=sum(ms(p[1] - p[0]) for p in tiny),
        other=sum(ms(p[1] - p[0]) for p in other), other_tok=sum(p[2] for p in other),
        gaps=gaps, d2h=sum(copy_ms(a[1], b[0], 2) for a, b in zip(passes, passes[1:])),
        tail=ms(first_decode - passes[-1][1])))

print(f"{len(rows)} requests with a decode inside the capture; rate {RATE} ms/token")
print(f"{'path':24s} {'prompt':>6s} {'comp':>5s} {'total':>6s} {'prep':>5s} {'befor':>5s} {'h2d':>4s} "
      f"{'tinyN':>5s} {'tiny':>5s} {'other':>6s} {'oexc':>6s} {'gaps':>5s} {'d2h':>4s}  passes")
for x in rows:
    print(f"{x['path'][:24]:24s} {x['prompt']:6d} {x['computed']:5d} {x['total']:6.1f} {x['prepare']:5.1f} "
          f"{x['before']:5.1f} {x['h2d']:4.1f} {x['tiny_n']:5d} {x['tiny']:5.1f} {x['other']:6.1f} "
          f"{x['other'] - x['other_tok'] * RATE:6.1f} {x['gaps']:5.1f} {x['d2h']:4.1f}  {x['sizes']}")

res = [x for x in rows if x["path"] != "root"]
if res:
    n = len(res)
    m = lambda k: statistics.mean(x[k] for x in res)
    comp = m("computed")
    print(f"\nmeans over the {n} resumed requests (computed {comp:.0f} tokens, {m('n'):.1f} passes):")
    print(f"  arrival to first decode     {m('total'):7.1f} ms  (tokens x rate would be {comp * RATE:.1f})")
    print(f"  frontend.prepare            {m('prepare'):7.1f}")
    print(f"  prepare end to first pass   {m('before'):7.1f}  (of which host-to-device copies {m('h2d'):.1f})")
    print(f"  passes of <= 16 tokens      {m('tiny'):7.1f}  ({m('tiny_n'):.1f} per request)")
    print(f"  other passes                {m('other'):7.1f}  (token part {m('other_tok') * RATE:.1f}, "
          f"excess {m('other') - m('other_tok') * RATE:.1f})")
    print(f"  between passes              {m('gaps'):7.1f}  (of which device-to-host copies {m('d2h'):.1f})")
    print(f"  last pass to first decode   {m('tail'):7.1f}")
    pos = collections.Counter()
    for x in res:
        for k, s in enumerate(x["sizes"]):
            if s <= 16:
                pos["first" if k == 0 else "last" if k == len(x["sizes"]) - 1 else "middle"] += 1
    print(f"  tiny passes by position: {dict(pos)}")
