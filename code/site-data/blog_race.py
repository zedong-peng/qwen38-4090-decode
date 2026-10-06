#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Race data for the write-up: per-chunk (= per verify round) arrival times and texts from race_capture.py, with
tokens per chunk from re-tokenizing the streamed text (token start offsets inside each chunk's character span).
usage: blog_race.py TOKENIZER.json OUT.json LABEL=FILE ..."""
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
out = {}
for arg in sys.argv[3:]:
    lab, f = arg.split("=", 1)
    d = json.load(open(f))
    rows = []
    for r in d["rows"]:
        txt = "".join(c[1] for c in r["chunks"])
        enc = tok.encode(txt, add_special_tokens=False)
        starts = [o[0] for o in enc.offsets]
        bounds, pos = [], 0
        for c in r["chunks"]:
            bounds.append((pos, pos + len(c[1]))); pos += len(c[1])
        ks = [sum(1 for s in starts if a <= s < b) for a, b in bounds]
        n_usage = (r.get("usage") or {}).get("completion_tokens")
        rows.append({"qid": r["question_id"], "cat": r["category"], "prompt": r["prompt"],
                     "t": [c[0] for c in r["chunks"]], "x": [c[1] for c in r["chunks"]], "k": ks,
                     "n_retok": len(enc.ids), "n_usage": n_usage})
        print(lab, r["question_id"], "chunks", len(ks), "retok", len(enc.ids), "usage", n_usage, "zero-k", sum(1 for k in ks if k == 0),
              "max k", max(ks), file=sys.stderr)
    out[lab] = rows
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, separators=(",", ":"))
