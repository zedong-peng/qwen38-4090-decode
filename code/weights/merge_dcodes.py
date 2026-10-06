#!/usr/bin/env python3
"""Merge two drafter codes stores (tools/draft_gptq.py): tensors whose name contains one of ROLES come from B,
the rest from A (config.json and draft_gptq.json from A).
usage: merge_dcodes.py A B OUT ROLE[,ROLE...]   e.g. ROLE list mlp.gate_proj,mlp.up_proj,mlp.down_proj"""
import os, shutil, sys
from safetensors import safe_open
from safetensors.torch import save_file

a, b, out, roles = sys.argv[1], sys.argv[2], sys.argv[3], tuple(sys.argv[4].split(","))
os.makedirs(out, exist_ok=True)
fa = safe_open(os.path.join(a, "model.safetensors"), "pt")
fb = safe_open(os.path.join(b, "model.safetensors"), "pt")
merged, taken = {}, 0
for k in fa.keys():
    if any(r in k for r in roles):
        merged[k] = fb.get_tensor(k); taken += 1
    else:
        merged[k] = fa.get_tensor(k)
save_file(merged, os.path.join(out, "model.safetensors"))
for f in ("config.json", "draft_gptq.json"):
    if os.path.exists(os.path.join(a, f)):
        shutil.copy(os.path.join(a, f), out)
print(f"merged {len(merged)} tensors, {taken} from {b}")
