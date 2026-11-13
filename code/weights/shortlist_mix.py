#!/usr/bin/env python3
"""Proposal-head shortlist ranked by a mixture of the target's own output unigram distribution (the responses in the
drafter fine-tune index files, positions resp0 on) and the converter's corpus ranking.
Held-out check: fit on prompts with even line index, report misses (target response tokens outside the shortlist)
on odd ones; with --container also for the shortlist that container holds (it needs FILE.ninfer.conversion.json next
to it). Writes the full-data counts file for tools.convert --ranking.
usage: shortlist_mix.py OUT.counts.i64 INDEX.jsonl [INDEX.jsonl ...] [--lambda 0.5] [--container FILE.ninfer]
Run from the engine tree: it reads the corpus ranking in tools/freq_corpus and, for --container, tools.artifact."""
import argparse, json, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("index", nargs="+", help="draft_ft_features.py index files")
ap.add_argument("--lambda", dest="lam", type=float, default=0.5); ap.add_argument("--container")
a = ap.parse_args()
V = 248320
corpus = np.fromfile("tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64", dtype=np.int64)[:V].astype(np.float64)
seqs = []
for path in a.index:
    for line in open(path):
        r = json.loads(line); seqs.append(np.array(r["ids"][r["resp0"]:], dtype=np.int64))
fit = np.concatenate(seqs[0::2]); test = np.concatenate(seqs[1::2]); full = np.concatenate(seqs)


def counts(lam, toks):
    o = np.bincount(toks, minlength=V)[:V].astype(np.float64)
    p = lam * o / o.sum() + (1 - lam) * corpus / corpus.sum()
    return np.round(p * 1e15).astype(np.int64)


def shortlist(c, n):
    return np.argsort(-c, kind="stable")[:n]


special = np.arange(248044, V)  # chat/control tokens are always kept by the converter's force-include; keep them here too


def miss(ids, toks):
    inside = np.zeros(V, bool); inside[ids] = True; inside[special] = True
    return 1 - inside[toks].mean()


print(f"fit {fit.size} tokens, test {test.size}")
if a.container:
    sys.path.insert(0, ".")
    from tools.artifact.reader import Artifact
    rep = json.load(open(a.container + ".conversion.json"))
    oid = next(x for x in rep["methods"] if any("proposal/token_ids" in p for p in x["parameters"]))["object"]
    cur = np.frombuffer(Artifact(a.container).read_object(oid), dtype=np.int32)
    print(f"shortlist of {a.container} ({cur.size}): test miss {miss(cur, test):.4f}")
for lam in [0.25, 0.5, 0.75]:
    c = counts(lam, fit)
    print(f"lambda {lam}: " + "  ".join(f"{n}:{miss(shortlist(c, n), test):.4f}" for n in [49152, 65536, 98304, 131072]))
counts(a.lam, full).tofile(a.out)
print("wrote", a.out, "lambda", a.lam)
