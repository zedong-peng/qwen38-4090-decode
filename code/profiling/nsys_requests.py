#!/usr/bin/env python3
"""Lay out ninfer-serve requests from arrival to first token in an nsys sqlite export.

usage: nsys_requests.py prof.sqlite requests.jsonl [LO HI [SHOW]]
  LO..HI   computed prefill tokens of the requests to lay out (default 100..600)
  SHOW     how many of them to print in full (default 3); the rest only enter the averages

GPU work is charged to the NVTX range that was open on the launching thread when its runtime call was made
(kernels of a graph launch carry the launch's correlation id), so a chunk's GPU time is what the chunk enqueued.
"""
import bisect
import collections
import json
import sqlite3
import statistics
import sys

db, rqpath = sys.argv[1], sys.argv[2]
LO, HI = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (100, 600)
SHOW = int(sys.argv[5]) if len(sys.argv) > 5 else 3
GAP_NS = 300_000  # a GPU gap worth listing

c = sqlite3.connect(db)
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
S = dict(c.execute("select id, value from StringIds"))
T0 = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]

nvtx = [(s, e, n or S.get(t, "?"), g, p) for s, e, n, t, g, p in c.execute(
    "select start, end, text, textId, globalTid, uint64Value from NVTX_EVENTS "
    "where eventType in (59, 60) and end is not null order by start")]
eng_threads = collections.Counter(g for *_, g, _ in nvtx)

rt = {}  # correlation id -> (host start, thread, api name)
for s, e, g, cid, nid in c.execute("select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"):
    rt[cid] = (s, g, S.get(nid, "?"), e)
rt_calls = sorted((s, e, g, name) for (s, g, name, e) in rt.values())

gpu = []  # (start, end, kind, name, correlation id, bytes)
for s, e, cid, nid in c.execute("select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    gpu.append((s, e, "kernel", S.get(nid, "?"), cid, 0))
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    kinds = {1: "memcpy HtoD", 2: "memcpy DtoH", 8: "memcpy DtoD", 10: "memcpy PtoP"}
    for s, e, cid, b, k in c.execute("select start, end, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"):
        gpu.append((s, e, "memcpy", kinds.get(k, f"memcpy kind {k}"), cid, b))
if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
    for s, e, cid, b in c.execute("select start, end, correlationId, bytes from CUPTI_ACTIVITY_KIND_MEMSET"):
        gpu.append((s, e, "memset", "memset", cid, b))
gpu.sort()
gpu_starts = [x[0] for x in gpu]

osrt = []
if "OSRT_API" in tables:
    osrt = sorted((s, e, g, S.get(n, "?")) for s, e, g, n in c.execute("select start, end, globalTid, nameId from OSRT_API"))

# innermost open NVTX range per (thread, host time): ranges per thread sorted by start
by_thread = collections.defaultdict(list)
for r in nvtx:
    by_thread[r[3]].append(r)
starts_by_thread = {g: [r[0] for r in v] for g, v in by_thread.items()}


def enclosing(g, t):
    """NVTX ranges open on thread g at host time t, outermost first."""
    v, st = by_thread.get(g, []), starts_by_thread.get(g, [])
    i = bisect.bisect_right(st, t)
    out = [r for r in v[max(0, i - 400):i] if r[1] >= t]
    return sorted(out, key=lambda r: (r[0], -r[1]))


chunks_by_thread = collections.defaultdict(list)
for r in nvtx:
    if r[2] == "prefill.chunk":
        chunks_by_thread[r[3]].append(r)
chunk_starts = {g: [r[0] for r in v] for g, v in chunks_by_thread.items()}


def chunk_of(cid):
    """the prefill.chunk range open on the launching thread when GPU activity cid was enqueued"""
    if cid not in rt:
        return None
    s, g, _, _ = rt[cid]
    st = chunk_starts.get(g)
    if not st:
        return None
    i = bisect.bisect_right(st, s) - 1
    if i >= 0 and chunks_by_thread[g][i][1] >= s:
        return chunks_by_thread[g][i]
    return None


def owner(cid):
    """innermost NVTX range open on the launching thread when GPU activity cid was enqueued"""
    if cid not in rt:
        return "(no runtime call)"
    s, g, api, _ = rt[cid]
    enc = enclosing(g, s)
    return enc[-1][2] if enc else f"(outside nvtx: {api})"


def window(lo, hi, seq, starts):
    i = bisect.bisect_left(starts, lo - 5_000_000_000)
    return [x for x in seq[i:bisect.bisect_right(starts, hi)] if x[1] >= lo]


nvtx_starts = [x[0] for x in nvtx]
rt_starts = [x[0] for x in rt_calls]
os_starts = [x[0] for x in osrt]


def merged(acts, lo, hi):
    """busy intervals of the GPU within [lo, hi]"""
    out = []
    for s, e, *_ in acts:
        s, e = max(s, lo), min(e, hi)
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


ms = lambda ns: ns / 1e6

# requests
recs = [json.loads(l) for l in open(rqpath)]
done = [d for d in recs if d.get("event") == "request_done"]
reqs = []
for d in done:
    tm, res, et = d["timings_seconds"], d["result"], d.get("engine_timing", {})
    t_done = d["timestamp_unix_ms"] * 1_000_000 - T0
    t_arr = t_done - int(tm["total"] * 1e9)
    reqs.append(dict(id=d["request"]["request_id"], arr=t_arr, ttft=t_arr + int(tm["ttft"] * 1e9), done=t_done,
                     computed=res.get("computed_prefill_tokens", 0), prompt=res.get("prompt_tokens", 0),
                     path=res.get("prefix_reuse_path"), units=et.get("units", {}).get("prefill"),
                     t=tm, plan=d.get("materialization", {}).get("planning_elapsed_ns", 0)))
reqs = [r for r in reqs if r["done"] > 0]
print(f"{len(reqs)} requests inside the capture; nvtx ranges {len(nvtx)}, gpu activities {len(gpu)}, osrt {len(osrt)}")

# clock check: first frontend.prepare after each arrival
fp = [r for r in nvtx if r[2] == "frontend.prepare"]
fps = [r[0] for r in fp]
offs = []
for r in reqs:
    i = bisect.bisect_left(fps, r["arr"] - 50_000_000)
    if i < len(fps):
        offs.append(ms(fps[i] - r["arr"]))
if offs:
    print(f"frontend.prepare start minus logged arrival: median {statistics.median(offs):.1f} ms, "
          f"min {min(offs):.1f}, max {max(offs):.1f} (should be a few ms; ms-resolution log)")

# every chunk of the capture: tokens, host time, GPU time it enqueued
chunk_gpu = collections.defaultdict(int)
chunk_kern = collections.defaultdict(collections.Counter)
for s, e, kind, name, cid, b in gpu:
    ch = chunk_of(cid)
    if ch is not None:
        chunk_gpu[ch[0]] += e - s
        chunk_kern[ch[0]][name] += e - s
chunks = [r for r in nvtx if r[2] == "prefill.chunk"]
print(f"\n{len(chunks)} prefill chunks. tokens -> GPU ms they enqueued (and host ms the range was open):")
pts = [(r[4], ms(chunk_gpu[r[0]]), ms(r[1] - r[0])) for r in chunks]
for lo_, hi_ in [(1, 16), (17, 64), (65, 256), (257, 512), (513, 1023), (1024, 1024)]:
    sel = [p for p in pts if lo_ <= p[0] <= hi_]
    if sel:
        print(f"  {lo_:>4}..{hi_:<4} n={len(sel):3d}  tokens {statistics.mean(p[0] for p in sel):6.0f}  "
              f"GPU {statistics.mean(p[1] for p in sel):7.1f} ms  host {statistics.mean(p[2] for p in sel):7.1f} ms")
if len(pts) > 2:
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx:
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        print(f"  fit: GPU ms = {my - b * mx:.1f} + {b * 1000:.3f} ms per 1000 tokens")
small = [r[0] for r in chunks if r[4] <= 64]
if small:
    agg = collections.Counter()
    for k in small:
        agg.update(chunk_kern[k])
    print(f"  kernels of chunks with <= 64 tokens (mean per chunk, n={len(small)}):")
    for name, ns in agg.most_common(12):
        print(f"    {ms(ns) / len(small):7.2f} ms  {name[:100]}")

# the selected requests
sel = [r for r in reqs if LO <= r["computed"] <= HI]
print(f"\n{len(sel)} requests recompute {LO}..{HI} tokens")
tot = collections.defaultdict(list)
by_range = collections.Counter()
by_kernel = collections.Counter()
by_api = collections.Counter()
by_os = collections.Counter()
for k, r in enumerate(sel):
    lo, hi = r["arr"] - 2_000_000, r["ttft"] + 2_000_000
    acts = window(lo, hi, gpu, gpu_starts)
    busy = merged(acts, lo, hi)
    busy_ns = sum(e - s for s, e in busy)
    first_gpu = busy[0][0] if busy else hi
    tot["window"].append(ms(hi - lo))
    tot["gpu busy"].append(ms(busy_ns))
    tot["arrival to first gpu"].append(ms(first_gpu - r["arr"]))
    tot["log prepare"].append(r["t"]["prepare"] * 1e3)
    tot["log prefill"].append(r["t"]["prefill"] * 1e3)
    tot["log ttft"].append(r["t"]["ttft"] * 1e3)
    tot["planning"].append(r["plan"] / 1e6)
    for s, e, kind, name, cid, b in acts:
        by_kernel[name] += min(e, hi) - max(s, lo)
        by_range[owner(cid)] += min(e, hi) - max(s, lo)
    for s, e, g, name in window(lo, hi, rt_calls, rt_starts):
        if g in eng_threads:
            by_api[name] += min(e, hi) - max(s, lo)
    for s, e, g, name in window(lo, hi, osrt, os_starts):
        if g in eng_threads:
            by_os[name] += min(e, hi) - max(s, lo)
    if k >= SHOW:
        continue
    print(f"\nrequest {r['id']}: prompt {r['prompt']}, computed {r['computed']}, {r['path']}, units {r['units']}; "
          f"log: prepare {r['t']['prepare'] * 1e3:.1f} ms, prefill {r['t']['prefill'] * 1e3:.1f} ms, "
          f"ttft {r['t']['ttft'] * 1e3:.1f} ms, planning {r['plan'] / 1e6:.1f} ms")
    print(f"  GPU busy {ms(busy_ns):.1f} of {ms(hi - lo):.1f} ms; first GPU work {ms(first_gpu - r['arr']):+.1f} ms")
    print("  host ranges (offset from arrival, duration, payload):")
    for s, e, name, g, p in window(lo, hi, nvtx, nvtx_starts):
        if name.startswith(("prefill.layer", "prefill.attention", "prefill.gdn", "prefill.post_mixer", "dflash.layer",
                            "dflash.attention", "dflash.mlp", "verify.")):
            continue
        if e - s < 200_000 and name not in ("prefill.chunk", "frontend.prepare"):
            continue
        depth = len(enclosing(g, s)) - 1
        print(f"    {ms(s - r['arr']):+8.1f} {ms(e - s):7.1f} ms  {'  ' * depth}{name}" + (f" [{p}]" if p else ""))
    print("  GPU segments (split at gaps > 0.3 ms):")
    seg = []
    for s, e, kind, name, cid, b in acts:
        if seg and s - seg[-1]["e"] <= GAP_NS:
            seg[-1]["e"] = max(seg[-1]["e"], e)
            seg[-1]["n"] += 1
            seg[-1]["k"][name] += e - s
        else:
            seg.append(dict(s=s, e=e, n=1, k=collections.Counter({name: e - s})))
    for x in seg:
        if x["e"] < lo or x["s"] > hi:
            continue
        top = ", ".join(f"{n[:40]} {ms(v):.1f}" for n, v in x["k"].most_common(3))
        print(f"    {ms(x['s'] - r['arr']):+8.1f} {ms(x['e'] - x['s']):7.1f} ms  {x['n']:5d} ops  {top}")
    print("  gaps > 0.3 ms and the engine threads' host calls inside them:")
    for (s0, e0), (s1, e1) in zip(busy, busy[1:]):
        if s1 - e0 < GAP_NS:
            continue
        calls = collections.Counter()
        for s, e, g, name in window(e0, s1, rt_calls, rt_starts) + window(e0, s1, osrt, os_starts):
            if g in eng_threads:
                calls[name] += min(e, s1) - max(s, e0)
        top = ", ".join(f"{n} {ms(v):.1f}" for n, v in calls.most_common(4))
        print(f"    {ms(e0 - r['arr']):+8.1f} {ms(s1 - e0):7.1f} ms  {top}")

if sel:
    n = len(sel)
    print(f"\nmeans over the {n} requests:")
    for k, v in tot.items():
        print(f"  {k:22s} {statistics.mean(v):8.1f} ms")
    print("  GPU time by the NVTX range that enqueued it (ms per request):")
    for name, ns in by_range.most_common(12):
        print(f"    {ms(ns) / n:7.2f}  {name}")
    print("  GPU time by kernel (ms per request):")
    for name, ns in by_kernel.most_common(15):
        print(f"    {ms(ns) / n:7.2f}  {name[:100]}")
    print("  CUDA runtime calls on engine threads (host ms per request):")
    for name, ns in by_api.most_common(10):
        print(f"    {ms(ns) / n:7.2f}  {name}")
    print("  OS runtime calls on engine threads (host ms per request):")
    for name, ns in by_os.most_common(10):
        print(f"    {ms(ns) / n:7.2f}  {name}")
