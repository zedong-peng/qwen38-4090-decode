#!/usr/bin/env python3
"""Occupancy audit of an nsys sqlite: per (kernel, grid, block) the registers, shared memory, resident CTAs per SM
(Ada: 65536 regs, 100 KB smem, 1536 threads, 24 CTAs), waves, and us per round. usage: occ_audit.py DB [TOP=40]"""
import sqlite3, sys, statistics, collections
db = sqlite3.connect(sys.argv[1]); top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
rows = db.execute("select k.start, k.end, s.value, k.gridX*k.gridY*k.gridZ, k.blockX*k.blockY*k.blockZ, "
                  "k.registersPerThread, k.staticSharedMemory + k.dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL k "
                  "join StringIds s on s.id = k.shortName order by k.start").fetchall()
fold = [i for i, r in enumerate(rows) if "recurrent_fold_staged" in r[2]]
rows = rows[fold[2]:fold[-1]]; rounds = len(fold) - 3
g = collections.defaultdict(list)
for st, en, name, grid, block, regs, smem in rows:
    g[(name, grid, block, regs, smem)].append((en - st) / 1000)
SMS = 128
def occ(block, regs, smem):
    warps = (block + 31) // 32
    regs_w = ((regs * 32 + 255) // 256) * 256
    by_regs = 65536 // (regs_w * warps) if regs else 99
    by_smem = (102400 // (smem + 1024)) if smem else 99
    by_thr = 1536 // block
    return min(by_regs, by_smem, by_thr, 24), by_regs, by_smem, by_thr
out = []
for (name, grid, block, regs, smem), d in g.items():
    per = sum(d) / rounds
    o, br, bs, bt = occ(block, regs, smem)
    waves = grid / (o * SMS) if o else 0
    out.append((per, name, grid, block, regs, smem, o, br, bs, bt, waves, statistics.median(d), len(d) / rounds))
out.sort(reverse=True)
print(f"{rounds} rounds")
print("  us/rd   calls  med_us  grid  block regs  smem  occ(r/s/t)  waves  kernel")
for per, name, grid, block, regs, smem, o, br, bs, bt, waves, med, calls in out[:top]:
    nxt = ""
    # registers that would give one more CTA per SM
    warps = (block + 31) // 32
    if br == o and o < min(bs, bt, 24):
        need = (65536 // ((o + 1) * warps)) // 256 * 256 // 32
        nxt = f"  (<= {need} regs -> {o+1})"
    print(f"{per:7.1f} {calls:6.1f} {med:7.2f} {grid:5d} {block:5d} {regs:4d} {smem:6d}  {o:2d}({br}/{bs}/{bt})  {waves:5.2f}  {name[:60]}{nxt}")
