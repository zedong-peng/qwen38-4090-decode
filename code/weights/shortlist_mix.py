#!/usr/bin/env python3
"""Proposal-head shortlist ranked by a mixture of the target's own output unigram distribution (generated drafter
fine-tune responses, data/draft-ft/*/index.jsonl from resp0) and the converter's corpus ranking.
Held-out check: fit on prompts with even line index, report misses (target response tokens outside the shortlist)
on odd ones, against the container's current shortlist. Writes the full-data counts file for tools.convert --ranking.
usage: shortlist_mix.py OUT.counts.i64 [LAMBDA=0.5] [DATASETS=c48a,c48b,r2a,r2b]   (run with tools/vllm-env python
from the repo root)"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from tools.artifact.reader import Artifact
out = sys.argv[1]; lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
sets = (sys.argv[3] if len(sys.argv) > 3 else "c48a,c48b,r2a,r2b").split(",")
V = 248320
corpus = np.fromfile("tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64", dtype=np.int64)[:V].astype(np.float64)
seqs = []
for d in sets:
    for l in open(f"../../data/draft-ft/{d}/index.jsonl"):
        r = json.loads(l); seqs.append(np.array(r["ids"][r["resp0"]:], dtype=np.int64))
fit = np.concatenate(seqs[0::2]); test = np.concatenate(seqs[1::2]); full = np.concatenate(seqs)


def counts(lam, toks):
    o = np.bincount(toks, minlength=V)[:V].astype(np.float64)
    p = lam * o / o.sum() + (1 - lam) * corpus / corpus.sum()
    return np.round(p * 1e15).astype(np.int64)


def shortlist(c, n):
    return np.argsort(-c, kind="stable")[:n]


rep = json.load(open("../../models/c96-ft1/qwen3_8_27b.ninfer.conversion.json"))
oid = [x for x in rep["methods"] if any("proposal/token_ids" in p for p in x["parameters"])][0]["object"]
cur = np.frombuffer(Artifact("../../models/c96-ft1/qwen3_8_27b.ninfer").read_object(oid), dtype=np.int32)
special = np.arange(248044, V)  # chat/control tokens are always kept by the converter's force-include; keep them here too


def miss(ids, toks):
    inside = np.zeros(V, bool); inside[ids] = True; inside[special] = True
    return 1 - inside[toks].mean()


print(f"fit {fit.size} tokens, test {test.size}; current shortlist ({cur.size}) test miss {miss(cur, test):.4f}")
for l in [0.25, 0.5, 0.75]:
    c = counts(l, fit)
    print(f"lambda {l}: " + "  ".join(f"{n}:{miss(shortlist(c, n), test):.4f}" for n in [49152, 65536, 98304, 131072]))
counts(lam, full).tofile(out)
print("wrote", out, "lambda", lam)
