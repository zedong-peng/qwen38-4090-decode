#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fidelity of a candidate engine/quantization against the BF16 reference over identical windows.
usage: fidelity.py WINDOWS.bin REF_PREFIX CAND_PREFIX [CAND_REPORT.json]
  REF_PREFIX.topk.bin/.target.bin from ref_bf16.py; CAND_PREFIX.topk.bin/.atref.bin from a
  Cinference scoring dump run with NINFER_SCORE_REF=REF_PREFIX.topk.bin.
Metrics per stream domain: KL(ref||cand) on the reference's top-64 support (exact candidate
log-probs at those ids), the reference mass that support covers, top-1 agreement, reference PPL,
and the candidate PPL from its ninfer-perplexity report."""
import json, struct, sys
import numpy as np

TOP = 64
win_path, ref, cand = sys.argv[1:4]
report = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None

d = open(win_path, "rb").read(); off = 0; cols_stream = []; stream = -1
while off < len(d):
    n, first = struct.unpack_from("<II", d, off); off += 8 + 4 * n
    if first == 1:
        stream += 1
    cols_stream += [stream] * (n - first)
cols_stream = np.array(cols_stream)
rec = lambda p: np.fromfile(p, dtype=np.float32).reshape(-1, 1 + 2 * TOP)
R = rec(ref + ".topk.bin"); C = rec(cand + ".topk.bin")
A = np.fromfile(cand + ".atref.bin", dtype=np.float32).reshape(-1, TOP)
T = np.fromfile(ref + ".target.bin", dtype=np.float32)
n = len(cols_stream)
assert R.shape[0] == n == C.shape[0] == A.shape[0] == T.shape[0], (R.shape, C.shape, A.shape, T.shape, n)
ref_lse, ref_ids, ref_logit = R[:, 0], R[:, 1:1 + TOP].view(np.int32), R[:, 1 + TOP:]
cand_lse, cand_ids = C[:, 0], C[:, 1:1 + TOP].view(np.int32)
logp = ref_logit - ref_lse[:, None]; p = np.exp(logp)
logq = A - cand_lse[:, None]
kl = (p * (logp - logq)).sum(1)
mass = p.sum(1)
top1 = ref_ids[:, 0] == cand_ids[:, 0]
domains = report and [s["domain"] for s in report["streams"]]
cand_ppl = {}
import os
CT = np.fromfile(cand + ".target.bin", dtype=np.float32) if os.path.exists(cand + ".target.bin") else None
if report:
    for s in report["streams"]:
        cand_ppl[s["domain"]] = s["perplexity"]
    cand_ppl["overall"] = report["overall"]["perplexity"]
print(f"{'domain':10s} {'cols':>6s} {'KL64':>9s} {'mass':>6s} {'top1':>7s} {'PPL ref':>8s} {'PPL cand':>8s}")
groups = [(domains[s] if domains else f"s{s}", cols_stream == s) for s in range(stream + 1)] + [("overall", np.ones(n, bool))]
out = {}
for name, m in groups:
    row = {"cols": int(m.sum()), "kl64": float(kl[m].mean()), "mass": float(mass[m].mean()),
           "top1": float(top1[m].mean()), "ppl_ref": float(np.exp(-T[m].mean())),
           "ppl_cand": float(np.exp(-CT[m].mean())) if CT is not None else cand_ppl.get(name)}
    out[name] = row
    pc = "-" if row["ppl_cand"] is None else f"{row['ppl_cand']:.4f}"
    print(f"{name:10s} {row['cols']:6d} {row['kl64']:9.5f} {row['mass']:6.3f} {row['top1']:7.4f} {row['ppl_ref']:8.4f} {pc:>8s}")
json.dump(out, open(cand + ".fidelity.json", "w"), indent=1)
