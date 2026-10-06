#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify trees for the write-up from NINFER_TREE_DUMP + NINFER_TOKEN_DUMP (same parsing and best-first growth as
tools/tree_sim.py at the deployed budget N = 15 and tree temperature). Each round: the 15 drafted nodes (parent, depth,
token text), which of them the target accepted, and the target's own bonus token. The replay must match the run's
own per-round acceptance (next root - this root); rounds where it does not are reported and dropped.
usage: blog_trees.py DUMPDIR TOKENIZER.json OUT.json [TEMP=1.25] [FIRST_REQ=2]"""
import json, sys
import numpy as np
from tokenizers import Tokenizer

d, tokf, out = sys.argv[1:4]
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else 1.25
FIRST = int(sys.argv[5]) if len(sys.argv) > 5 else 2
tok = Tokenizer.from_file(tokf)
raw = np.fromfile(f"{d}/tree.bin", dtype=np.int32)
reqs = [json.loads(l) for l in open(f"{d}/tokens.jsonl")]
recs, off = [], 0
while off < raw.size:
    K = int(raw[off]); n = 6 + K + 16 * K + 256 * K
    r = raw[off:off + n]
    recs.append(dict(K=K, root=int(r[1]), anchor=int(r[2]), extent=int(r[3]), lcount=int(r[4]),
                     llp=float(r[5:6].view(np.float32)[0]), lookup=r[6:6 + K].copy(),
                     ids=r[6 + K:6 + 17 * K].reshape(K, 16).copy(),
                     edges=r[6 + 17 * K:n].view(np.float32).reshape(K, 16, 16).copy()))
    off += n
assign, ri, last = [], 0, -1
for rc in recs:
    for step in range(0, 8):
        if ri + step >= len(reqs): break
        rq = reqs[ri + step]; j = rc["root"] - rq["prompt_tokens"]
        if (step > 0 or rc["root"] > last) and 0 <= j < len(rq["ids"]) and rq["ids"][j] == rc["anchor"]:
            ri += step; last = rc["root"]; assign.append(ri); break
    else:
        assign.append(-1)
groups = [[] for _ in reqs]
for rc, a in zip(recs, assign):
    if a >= 0: groups[a].append(rc)
BIAS = np.zeros(15, dtype=np.float32)
NMAX = 15


def grow(rc, inv_t):
    K, extent, lc = rc["K"], max(0, min(rc["extent"], rc["K"])), max(0, min(rc["lcount"], rc["K"]))
    e = (rc["edges"] * np.float32(inv_t)).astype(np.float32)
    m = e.max(axis=2, keepdims=True)
    lsm = (e - (m + np.log(np.exp(e - m).sum(axis=2, keepdims=True, dtype=np.float32)) - BIAS[:K].reshape(-1, 1, 1))).astype(np.float32)
    ids, lookup, llp = rc["ids"], rc["lookup"], np.float32(rc["llp"])
    lrank = [-1] * K
    for s_ in range(lc):
        hit = np.nonzero(ids[s_] == lookup[s_])[0]
        lrank[s_] = int(hit[0]) if hit.size else -1
    fs, finfo = [], []

    def push(parent, step, rank, pscore, on_lookup):
        chain = on_lookup and step < lc
        crank = lrank[step] if chain else -1
        sc = (pscore + lsm[step, rank]).astype(np.float32)
        for c in range(16):
            s_, onl = sc[c], False
            if c == crank:
                s_, onl = max(s_, np.float32((step + 1) * llp)), True
            fs.append(s_); finfo.append((parent, step, c, onl))
        if chain and crank < 0:
            fs.append(np.float32((step + 1) * llp)); finfo.append((parent, step, -1, True))
    nodes = []
    if extent > 0:
        push(0, 0, 0, np.float32(0), True)
    for node in range(1, NMAX + 1):
        if not fs: break
        fsa = np.array(fs, dtype=np.float32)
        best = int(np.argmax(fsa))
        if not np.isfinite(fsa[best]): break
        bscore = fsa[best]; fs[best] = np.float32(-np.inf)
        parent, step, rank, onl = finfo[best]
        t = int(lookup[step]) if rank < 0 else int(ids[step, rank])
        nodes.append((parent, step, t, float(bscore), rank < 0 or onl))
        if step + 1 < extent:
            if rank < 0:
                if step + 1 < lc:
                    fs.append(np.float32((step + 2) * llp)); finfo.append((node, step + 1, -1, True))
            else:
                push(node, step + 1, rank, bscore, onl)
    return nodes


def piece(i):
    return tok.decode([i], skip_special_tokens=False)


res = []
for qi in range(FIRST, len(reqs)):
    rq, g = reqs[qi], groups[qi]
    P, ids = rq["prompt_tokens"], rq["ids"]
    rounds, bad = [], 0
    for i, rc in enumerate(g):
        j = rc["root"] - P
        nodes = grow(rc, 1.0 / TEMP)
        target = ids[j + 1:j + 1 + rc["K"]]
        acc, parent = [], 0
        for t in target:
            hit = next((n + 1 for n, (p, s_, tk, sc, lk) in enumerate(nodes) if p == parent and tk == t), None)
            if hit is None: break
            acc.append(hit); parent = hit
        if i + 1 < len(g):
            gap = g[i + 1]["root"] - rc["root"]
            if gap != len(acc) + 1:
                bad += 1
                continue
        bonus = ids[j + 1 + len(acc)] if j + 1 + len(acc) < len(ids) else None
        rounds.append({"pos": j, "nodes": [[p, s_, piece(tk), round(sc, 3), int(lk)] for p, s_, tk, sc, lk in nodes],
                       "acc": acc, "anchor": piece(ids[j]), "bonus": piece(bonus) if bonus is not None else None,
                       "ctx": tok.decode(ids[:j + 1])})
    res.append({"prompt_tokens": P, "n_ids": len(ids), "rounds": rounds, "dropped": bad, "text": tok.decode(ids)})
    print(f"request {qi}: {len(g)} rounds, replay disagreed on {bad}, kept {len(rounds)}; mean accepted+1 "
          f"{np.mean([len(r['acc']) + 1 for r in rounds]):.2f}", file=sys.stderr)
json.dump(res, open(out, "w"), ensure_ascii=False, separators=(",", ":"))
