#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How far two NINFER_TOKEN_DUMP runs agree before greedy decoding diverges. Per request: the common prefix of the
generated ids. usage: ids_prefix.py REF.ids.jsonl RUN.ids.jsonl"""
import json, statistics, sys
a = [json.loads(l)["ids"] for l in open(sys.argv[1])]
b = [json.loads(l)["ids"] for l in open(sys.argv[2])]
pre = []
for x, y in zip(a, b):
    n = next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    pre.append((n, min(len(x), len(y)), x == y))
same = sum(p[2] for p in pre)
frac = sum(p[0] for p in pre) / max(1, sum(p[1] for p in pre))
div = [p[0] for p in pre if not p[2]]
print(f"identical {same}/{len(pre)}; common prefix {frac:.3f} of generated tokens; "
      f"diverged requests: median first difference at token {statistics.median(div) if div else '-'}")
