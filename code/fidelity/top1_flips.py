#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Top-1 agreement flips between candidate dumps against the same reference: for each candidate, agreement count
and, against the base, positions gained / lost. usage: top1_flips.py REF_PREFIX BASE_PREFIX CAND_PREFIX..."""
import sys
import numpy as np
TOP = 64
rec = lambda p: np.fromfile(p, dtype=np.float32).reshape(-1, 1 + 2 * TOP)
R = rec(sys.argv[1] + ".topk.bin")
ref = R[:, 1].view(np.int32)
base = rec(sys.argv[2] + ".topk.bin")[:, 1].view(np.int32) == ref
print(f"{'candidate':34s} {'agree':>7s} {'top1':>8s} {'gained':>7s} {'lost':>6s} {'net':>6s}")
print(f"{sys.argv[2].split('/')[-1]:34s} {base.sum():7d} {base.mean():8.5f}")
for c in sys.argv[3:]:
    a = rec(c + ".topk.bin")[:, 1].view(np.int32) == ref
    g, l = int((a & ~base).sum()), int((~a & base).sum())
    print(f"{c.split('/')[-1]:34s} {a.sum():7d} {a.mean():8.5f} {g:7d} {l:6d} {g - l:+6d}")
