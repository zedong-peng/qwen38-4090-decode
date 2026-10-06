#!/usr/bin/env python3
"""Split a windows file into consecutive parts of N windows: OUT_PREFIX.partK.windows.bin.
usage: split_windows.py IN.windows.bin N OUT_PREFIX"""
import struct, sys
src, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = open(src, "rb").read(); off = 0; recs = []
while off < len(d):
    L, first = struct.unpack_from("<II", d, off); recs.append(d[off:off + 8 + 4 * L]); off += 8 + 4 * L
for k in range(0, len(recs), n):
    open(f"{out}.part{k // n}.windows.bin", "wb").write(b"".join(recs[k:k + n]))
print(len(recs), "windows ->", (len(recs) + n - 1) // n, "parts")
