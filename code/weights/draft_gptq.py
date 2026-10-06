#!/usr/bin/env python3
"""GPTQ for the DFlash2 drafter's projections from engine-dumped inputs (NINFER_DRAFT_DUMP).
Writes a codes store (PREFIX.gptq_codes I8 [N,K], PREFIX.gptq_scales F16 [N,K/G]) that the converter
imports (tools/gptq_recipe.py with --source dcodes=DIR).
  --bits B         target grid: B-bit G64 with the canonical FP16 absmax scale (clip search optional)
  --emulate-q8     store the B-bit grid as Q8 G32 codes (code*2^(8-B), scale/2^(8-B)) for the Q8 kernels
  --rtn            no error compensation (same grid; the RTN baseline)
usage: draft_gptq.py DUMP_DIR DRAFTER_DIR OUT_DIR [--bits 4] [--act-order] [--clip-grid 1,.95,...]"""
import argparse, json, os, shutil, sys, time
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fq_eval  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dump"); ap.add_argument("drafter"); ap.add_argument("out")
ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--act-order", action="store_true")
ap.add_argument("--clip-grid", default="1.0")
ap.add_argument("--emulate-q8", action="store_true")
ap.add_argument("--rtn", action="store_true")
ap.add_argument("--max-cols", type=int, default=0, help="use at most this many dumped columns per input")
ap.add_argument("--damp", type=float, default=0.01)
a = ap.parse_args()
fq_eval.ACT_ORDER = a.act_order
fq_eval.ALPHAS = tuple(float(x) for x in a.clip_grid.split(","))
qmax = (1 << (a.bits - 1)) - 1
qmin = -qmax if a.emulate_q8 else -(1 << (a.bits - 1))
dev = os.environ.get("DEV", "cuda")  # DEV=cpu when no GPU is free
cfg = json.load(open(os.path.join(a.drafter, "config.json")))
layers = cfg.get("num_hidden_layers", 5)
f = safe_open(os.path.join(a.drafter, "model.safetensors"), framework="pt", device="cpu")

JOBS = [("fc", ["fc"])]
for i in range(layers):
    JOBS += [(f"L{i}.o", [f"layers.{i}.self_attn.o_proj"]),
             (f"L{i}.mlp", [f"layers.{i}.mlp.gate_proj", f"layers.{i}.mlp.up_proj"]),
             (f"L{i}.down", [f"layers.{i}.mlp.down_proj"])]


def hessian(name, k):
    path = os.path.join(a.dump, name + ".bf16")
    cols = os.path.getsize(path) // (2 * k)
    if a.max_cols:
        cols = min(cols, a.max_cols)
    H = torch.zeros(k, k, dtype=torch.float32, device=dev)
    mm = np.memmap(path, dtype=np.uint16, mode="r", shape=(cols, k))
    for c0 in range(0, cols, 4096):
        x = torch.from_numpy(np.ascontiguousarray(mm[c0:c0 + 4096]).view(np.int16)).view(torch.bfloat16)
        x = x.to(dev).float()
        H.addmm_(x.t(), x)
    return H / max(cols, 1), cols


def proxy(w, q, H):
    d = (w - q)
    return float(((d @ H) * d).sum() / ((w @ H) * w).sum())


os.makedirs(a.out, exist_ok=True)
shutil.copy(os.path.join(a.drafter, "config.json"), a.out)
saved, report = {}, {}
t0 = time.time()
for dump, prefixes in JOBS:
    w0 = f.get_tensor(prefixes[0] + ".weight")
    k = w0.shape[1]
    H, cols = hessian(dump, k)
    for prefix in prefixes:
        w = f.get_tensor(prefix + ".weight").to(dev).float()
        n = w.shape[0]
        qn = torch.full((n,), float(qmin), device=dev); qx = torch.full((n,), float(qmax), device=dev)
        rq = fq_eval.rtn(w, qn, qx, 64, codes=True, alphas=(1.0,))
        if a.rtn:
            Q, C, S = rq[0], rq[1], rq[2]
        else:
            Q, C, S = fq_eval.gptq(w, H, qn, qx, 64, 128, a.damp)
        e_rtn, e_q = proxy(w, rq[0], H), proxy(w, Q, H)
        report[prefix] = {"cols": cols, "rtn": e_rtn, "out": e_q}
        print(f"{prefix:32s} cols {cols:7d}  rel err rtn {e_rtn:.5f}  out {e_q:.5f}  {time.time() - t0:.0f}s", flush=True)
        if a.emulate_q8:
            m = 1 << (8 - a.bits)
            C = (C.to(torch.int16) * m).to(torch.int8)
            S = (S.float() / m).half().repeat_interleave(2, dim=1)
            bad = ((C.float().reshape(n, -1, 32) * S.float()[..., None]).reshape(n, k) != Q).float().mean().item()
            report[prefix]["emulation_mismatch"] = bad
        saved[prefix + ".gptq_codes"] = C.cpu().contiguous()
        saved[prefix + ".gptq_scales"] = S.cpu().contiguous()
        del w, Q, C, S, rq
    del H
    torch.cuda.empty_cache()
save_file(saved, os.path.join(a.out, "model.safetensors"))
json.dump({"args": vars(a), "report": report}, open(os.path.join(a.out, "draft_gptq.json"), "w"), indent=1)
print("wrote", a.out)
