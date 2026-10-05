#!/usr/bin/env python3
# Compare per-kernel registers (cuobjdump -res-usage) of two builds: counts of kernels with more / fewer / equal
# registers, and the named hot kernels side by side.
# usage: res_cmp.py A.res B.res [REGEX]
import re, sys
from collections import Counter


def load(path):
    out, name = {}, None
    for line in open(path, errors='replace'):
        m = re.match(r'\s*Function (\S+):', line)
        if m:
            name = m.group(1)
            # rdc builds prefix kernels of anonymous namespaces with a TU-unique tag
            name = re.sub(r'__nv_static_\d+__[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]+_\d+__', '', name)
            name = re.sub(r'_GLOBAL__N__[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]+_\d+', '_GLOBAL__N_', name)
            continue
        m = re.search(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', line)
        if m and name:
            out[name] = tuple(int(v) for v in m.groups())
            name = None
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
c = Counter()
for k in common:
    c['more' if b[k][0] > a[k][0] else 'fewer' if b[k][0] < a[k][0] else 'same'] += 1
    if b[k][1] != a[k][1]:
        c['stack differs'] += 1
print(f'kernels: A {len(a)}  B {len(b)}  common {len(common)}  B vs A registers: {dict(c)}')
pat = re.compile(sys.argv[3] if len(sys.argv) > 3 else
                 r'ada_small_t_mma|screen_kernel|causal_attention_small_t_k8v4|rmsnorm|gdn|chunked|merge_chunk')
rows = []
for k in common:
    if pat.search(k) and a[k] != b[k]:
        short = re.sub(r'ninfer3ops6detail', '', k)[:150]
        rows.append((b[k][0] - a[k][0], a[k][0], b[k][0], a[k][1], b[k][1], short))
for d, ra, rb, sa, sb, k in sorted(rows)[:40] + (sorted(rows)[-15:] if len(rows) > 40 else []):
    print(f'{ra:4d} -> {rb:4d}  stack {sa}->{sb}  {k}')
