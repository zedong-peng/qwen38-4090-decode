#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One steady-state decode round's kernel timeline from an nsys sqlite (median-span round between consecutive
recurrent_fold_staged_kernel calls), for the write-up's interactive round figure.
usage: round_timeline.py OUT.json LABEL=SQLITE ..."""
import json, sqlite3, statistics, sys
MARKER = "recurrent_fold_staged_kernel"
def klass(n):
    if n.startswith(("ada_small_t", "q4_rowsplit", "q8_ksplit", "q4_", "q8_", "q5_", "q3_", "bf16_gemm", "gemm")): return "gemm"
    if n in ("screen_kernel",) or "screen" in n or "rescore" in n or "topk" in n: return "head"
    if "attention" in n or "attn" in n: return "attention"
    if "fold" in n or "record" in n or "gdn" in n or "recurrent" in n or "conv" in n: return "gdn"
    if "norm" in n: return "norm"
    return "other"
out = {}
for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    db = sqlite3.connect(path)
    rows = db.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on s.id = k.shortName order by k.start").fetchall()
    folds = [r[0] for r in rows if r[2] == MARKER]
    spans = [(folds[i + 1] - folds[i], i) for i in range(len(folds) - 1)]
    med = sorted(spans)[len(spans) // 2][1]
    t0, t1 = folds[med], folds[med + 1]
    ks = [r for r in rows if t0 <= r[0] < t1]
    names = sorted({n for _, _, n in ks})
    idx = {n: i for i, n in enumerate(names)}
    kern = [[round((s - t0) / 1e3, 2), round((e - s) / 1e3, 2), klass(n), idx[n]] for s, e, n in ks]
    tot = {}
    for k in kern: tot[k[2]] = tot.get(k[2], 0) + k[1]
    out[label] = {"span_us": round((t1 - t0) / 1e3, 1), "rounds": len(spans), "median_round": med,
                  "median_span_us": round(statistics.median(s for s, _ in spans) / 1e3, 1),
                  "kernels": kern, "names": names, "class_us": {k: round(v, 1) for k, v in tot.items()}}
    print(label, out[label]["span_us"], len(kern), out[label]["class_us"], file=sys.stderr)
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
