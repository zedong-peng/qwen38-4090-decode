#!/usr/bin/env python3
"""Layer-streaming Qwen3.8-27B evaluator with fake-quantized weights (FP32 compute).

Modes:
  ref     original BF16 weights, FP32 compute (the reference)
  rtn     NInfer qwen3_8_27b recipe, exactly as the converter quantizes (grouped absmax, FP16 scales):
          Q4 attention query/key, GDN query/key, MLP gate/up; Q5 everything else in the layers;
          Q8 (G32) embedding and head; GDN a/b projections stay BF16
  gptq    same formats (or --recipe allq4: Q4 for every layer projection), GPTQ error compensation
          with Hessians from calibration windows (sequential: layer i calibrates on outputs of the
          already-quantized layers < i)
Writes OUT.topk.bin / OUT.target.bin (and OUT.atref.bin with --ref REF.topk.bin) over the windows
of a Cinference scoring dump, in its record format.
  --asym A (gptq): asymmetric calibration. A second calibration stream runs the unquantized model (held on the
          host); each projection is fitted to the FP model's output on the FP stream instead of its own output on
          the quantized stream: GPTQ (Hessian H = X_q^T X_q) on W + A * W D^T X_q H_d^-1, D = X_fp - X_q, the
          least-squares weight for min |X_q Q^T - X_fp W^T| (A = 1). Later layers then absorb earlier error.
  --mlp-ap STEPS (gptq): learned rounding for each MLP after GPTQ. Latent codes U (initialized to the GPTQ codes)
          and log group scales are trained with Adam through a straight-through round so that the quantized MLP
          reproduces a target on the calibration tokens (sym: the FP MLP on the quantized stream's MLP input;
          asym: the FP layer's output on the FP stream minus the quantized stream's residual, i.e. the whole layer
          output matches the FP model); 1 in 20 calibration windows is held out and the best held-out step is kept.
usage: fq_eval.py MODEL_DIR WINDOWS.bin OUT --mode ref|rtn|gptq [--recipe official|allq4]
                  [--calib calib.windows.bin] [--ref REF.topk.bin] [--save-codes DIR]"""
import argparse, copy, json, os, shutil, struct, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
                                                          Qwen3_5TextRotaryEmbedding)

TOP = 64
FMT = {"q4": (-8, 7, 64), "q5": (-16, 15, 64), "q8": (-127, 127, 32)}


def read_windows(path):
    data = open(path, "rb").read(); out, off = [], 0
    while off < len(data):
        n, first = struct.unpack_from("<II", data, off); off += 8
        out.append((np.frombuffer(data, dtype=np.int32, count=n, offset=off).copy(), first)); off += 4 * n
    return out


class Weights:
    def __init__(self, d):
        self.dir = d; self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]; self.h = {}

    def get(self, name):
        f = self.map[name]
        if f not in self.h:
            self.h[f] = safe_open(os.path.join(self.dir, f), framework="pt", device="cpu")
        return self.h[f].get_tensor(name)


SCALE_GRID, ROW_SUPER = None, None
# --grid half: layer-projection levels (c + 1/2) * s for codes c in [qmin, qmax], s = absmax / (qmax + 1/2).
# --save-codes then stores them as odd codes 2c + 1 with scales s / 2.
HALF = 0.0
LS8_LUT = torch.from_numpy(np.exp2(-np.arange(256, dtype=np.float64) / 32.0).astype(np.float32))


def ls8_round(s):
    """s: float32 scales with leading dim = rows. Nearest LS8 grid value under ROW_SUPER."""
    sup = ROW_SUPER.view(-1, *([1] * (s.dim() - 1)))
    j = torch.round(-32.0 * torch.log2(torch.clamp(s, min=1e-30) / sup)).clamp(0, 255).long()
    return sup * LS8_LUT.to(s.device)[j]


def canonical_scale(absmax, qmax):
    if SCALE_GRID == "ls8":
        s = ls8_round((absmax.double() / (qmax + HALF)).float())
        recip = torch.where(s > 0, (1.0 / s.double()).float(), torch.zeros_like(s))
        return s, recip
    s = (absmax.double() / (qmax + HALF)).float().half()
    s = torch.where((s == 0) & (absmax > 0), torch.tensor(2.0 ** -24, dtype=torch.half, device=s.device), s)
    recip = torch.where(s > 0, (1.0 / s.double()).float(), torch.zeros_like(s, dtype=torch.float))
    return s.float(), recip


def search_scale(g, qmin, qmax, alphas, weight=None):
    """g [..., group] values; qmin/qmax broadcastable to g[..., 0]. Per group, the FP16 scale among
    fp16(alpha * absmax / qmax) minimizing sum(weight * (g - q(g))^2). Returns (scale, reciprocal)."""
    absmax = g.abs().amax(-1)
    if len(alphas) == 1:
        return canonical_scale(absmax * alphas[0], qmax)
    best = None
    for alpha in alphas:
        s, r = canonical_scale(absmax * alpha, qmax)
        q = torch.clamp(torch.round(g * r[..., None] - HALF), qmin[..., None] if torch.is_tensor(qmin) else qmin,
                        qmax[..., None] if torch.is_tensor(qmax) else qmax) + HALF
        e = (g - q * s[..., None]) ** 2
        e = (e * weight).sum(-1) if weight is not None else e.sum(-1)
        if best is None:
            best = [e, s, r]
        else:
            take = e < best[0]
            best = [torch.where(take, e, best[0]), torch.where(take, s, best[1]), torch.where(take, r, best[2])]
    return best[1], best[2]


ALPHAS = (1.0,)


def set_row_super(w, qmax, group):
    global ROW_SUPER
    if SCALE_GRID == "ls8":
        qm = qmax.float() if torch.is_tensor(qmax) else torch.tensor(float(qmax), device=w.device)
        ROW_SUPER = (w.float().reshape(w.shape[0], -1, group).abs().amax(-1).amax(-1) / (qm + HALF)).clamp(min=1e-30)


def rtn(w, qmin, qmax, group, codes=False, alphas=None):
    n, k = w.shape
    if n <= 16384:
        set_row_super(w, qmax, group)
    if n > 16384:  # bound the temporaries (the vocabulary matrices)
        parts = [rtn(w[i:i + 16384], qmin[i:i + 16384] if torch.is_tensor(qmin) else qmin,
                     qmax[i:i + 16384] if torch.is_tensor(qmax) else qmax, group, codes, alphas)
                 for i in range(0, n, 16384)]
        if not codes:
            return torch.cat(parts)
        return tuple(torch.cat([p[j] for p in parts]) for j in range(3))
    g = w.float().reshape(n, k // group, group)
    qmx = qmax[:, None] if torch.is_tensor(qmax) else qmax
    qmn = qmin[:, None] if torch.is_tensor(qmin) else qmin
    s, r = search_scale(g, qmn, qmx, alphas or ALPHAS)
    q = torch.clamp(torch.round(g * r[..., None] - HALF), (qmin[:, None, None] if torch.is_tensor(qmin) else qmin),
                    (qmax[:, None, None] if torch.is_tensor(qmax) else qmax)) + HALF
    dq = (q * s[..., None]).reshape(n, k)
    return (dq, (q - HALF).reshape(n, k).to(torch.int8), s.half()) if codes else dq


ACT_ORDER = False
DAMP_DIAG = [0.0]  # --damp-diag: H += d diag(H) (per-channel shrinkage of the calibration Hessian), before --damp
KEEP_U = [False, None]  # --mlp-ap: gptq() leaves the continuous pre-rounding codes (col / s) in KEEP_U[1]


def gptq(w, H, qmin, qmax, group=64, block=128, damp=0.01):
    if ACT_ORDER:
        return gptq_act_order(w, H, qmin, qmax, group, block, damp)
    """qmin/qmax: per-row tensors. Returns the dequantized FP32 weight, int8 codes, FP16 scales."""
    W = w.float().clone(); n, k = W.shape
    set_row_super(W, qmax, group)
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    if DAMP_DIAG[0]:
        H += DAMP_DIAG[0] * torch.diag(torch.diag(H))
    H += damp * torch.mean(torch.diag(H)) * torch.eye(k, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    C = torch.zeros((n, k), dtype=torch.int8, device=W.device)
    S = torch.zeros((n, k // group), dtype=torch.half, device=W.device)
    qmin_c, qmax_c = qmin[:, None].float(), qmax[:, None].float()
    U = torch.zeros_like(W) if KEEP_U[0] else None
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        s = r = None
        for i in range(cnt):
            if (i1 + i) % group == 0:
                s, r = search_scale(W1[:, i:i + group], qmin.float(), qmax.float(), ALPHAS,
                                    torch.diag(H)[i1 + i:i1 + i + group][None, :])
                S[:, (i1 + i) // group] = s.half()
            col = W1[:, i]
            if U is not None:
                U[:, i1 + i] = col * r - HALF
            q = torch.clamp(torch.round(col * r - HALF), qmin_c[:, 0], qmax_c[:, 0]) + HALF
            dq = q * s
            Q1[:, i] = dq
            C[:, i1 + i] = (q - HALF).to(torch.int8)
            e = (col - dq) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    KEEP_U[1] = U
    return Q, C, S


def gptq_act_order(w, H, qmin, qmax, group=64, block=128, damp=0.01):
    """GPTQ in descending diag(H) column order with static groups: every group's FP16 scale is
    chosen up front from the original weights (diag(H)-weighted clip search), so the stored
    group layout is unchanged."""
    W = w.float().clone(); n, k = W.shape
    set_row_super(W, qmax, group)
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    gw = W.reshape(n, k // group, group)
    S, R = search_scale(gw, qmin[:, None].float(), qmax[:, None].float(), ALPHAS,
                        torch.diag(H).reshape(k // group, group)[None])
    perm = torch.argsort(torch.diag(H), descending=True)
    inv = torch.argsort(perm)
    W = W[:, perm]; H = H[perm][:, perm]
    col_group = perm // group
    if DAMP_DIAG[0]:
        H += DAMP_DIAG[0] * torch.diag(torch.diag(H))
    H += damp * torch.mean(torch.diag(H)) * torch.eye(k, device=H.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    C = torch.zeros((n, k), dtype=torch.int8, device=W.device)
    qmin_c, qmax_c = qmin.float(), qmax.float()
    U = torch.zeros_like(W) if KEEP_U[0] else None
    for i1 in range(0, k, block):
        i2 = min(i1 + block, k); cnt = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            g = col_group[i1 + i]
            s, r = S[:, g], R[:, g]
            col = W1[:, i]
            if U is not None:
                U[:, i1 + i] = col * r - HALF
            q = torch.clamp(torch.round(col * r - HALF), qmin_c, qmax_c) + HALF
            dq = q * s
            Q1[:, i] = dq
            C[:, i1 + i] = (q - HALF).to(torch.int8)
            e = (col - dq) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None, :]
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    KEEP_U[1] = U[:, inv] if U is not None else None
    return Q[:, inv], C[:, inv], S.half()


Q4_GROUPS, Q4_LAYERS = set(), None


def paley20():
    q = 19
    squares = {(x * x) % q for x in range(1, q)}
    chi = lambda v: 0 if v % q == 0 else (1 if v % q in squares else -1)
    Q = torch.tensor([[chi(j - i) for j in range(q)] for i in range(q)], dtype=torch.float64)
    S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    S[0, 1:] = 1; S[1:, 0] = -1; S[1:, 1:] = Q
    H = torch.eye(q + 1, dtype=torch.float64) + S
    assert torch.equal(H @ H.T, (q + 1) * torch.eye(q + 1, dtype=torch.float64))
    return H


def rotation(spec, n, dev):
    kind, _, seed = spec.partition(":")
    g = torch.Generator().manual_seed(int(seed or 0))
    if kind == "haar":
        Q, Rr = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
        R = Q * torch.sign(torch.diagonal(Rr))[None, :]
    elif kind == "had":
        assert n == 5120
        H = torch.tensor([[1.0]], dtype=torch.float64)
        for _ in range(8):
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        D = torch.randint(0, 2, (n,), generator=g).double() * 2 - 1
        R = torch.kron(paley20(), H) * D[None, :] / n ** 0.5
    else:
        raise ValueError(spec)
    assert torch.allclose(R @ R.T, torch.eye(n, dtype=torch.float64), atol=1e-10)
    return R.float().to(dev)


def rotate_layer(layer, R):
    """Fold the layer's two RMSNorm scales into their readers and rotate readers and writers."""
    def fold(norm, readers):
        s = 1.0 + norm.weight.data.float()
        for m in readers:
            m.weight.data = (m.weight.data * s[None, :]) @ R.T
        norm.weight.data.zero_()
    if hasattr(layer, "linear_attn"):
        a = layer.linear_attn
        fold(layer.input_layernorm, [a.in_proj_qkv, a.in_proj_z, a.in_proj_b, a.in_proj_a])
        for m in (a.in_proj_b, a.in_proj_a):
            m.weight.data = m.weight.data.bfloat16().float()
        writer = a.out_proj
    else:
        a = layer.self_attn
        fold(layer.input_layernorm, [a.q_proj, a.k_proj, a.v_proj])
        writer = a.o_proj
    fold(layer.post_attention_layernorm, [layer.mlp.gate_proj, layer.mlp.up_proj])
    for m in (writer, layer.mlp.down_proj):
        m.weight.data = R @ m.weight.data


def row_formats(name, rows, recipe, cfg):
    """Per-row format for an HF projection weight under the NInfer qwen3_8_27b recipe (plus the
    --q4-groups parents switched to Q4)."""
    q4 = np.zeros(rows, dtype=bool)
    if recipe == "allq5":  # every layer projection Q5 (a source of Q5 rows for hybrids)
        return q4
    layer = int(name.split("layers.")[1].split(".")[0])
    if Q4_GROUPS and (Q4_LAYERS is None or layer in Q4_LAYERS):
        q4 |= group_rows(name.removesuffix(".weight"), rows, Q4_GROUPS, cfg).numpy()
    if recipe == "allq4" or (recipe == "q4down" and name.endswith("mlp.down_proj.weight")):
        q4[:] = True
    elif name.endswith("self_attn.q_proj.weight"):
        d = cfg.head_dim
        for h in range(cfg.num_attention_heads):
            q4[h * 2 * d: h * 2 * d + d] = True  # query rows Q4, gate rows Q5
    elif name.endswith(("self_attn.k_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight")):
        q4[:] = True
    elif name.endswith("linear_attn.in_proj_qkv.weight"):
        kg = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        q4[: 2 * kg] = True  # query, key Q4; value Q5
    return q4


def group_rows(stem, n, groups, cfg):
    """Rows of HF tensor `stem` that belong to the engine parent groups (see build_hybrid.py)."""
    m = torch.zeros(n, dtype=torch.bool)
    d, heads = cfg.head_dim, cfg.num_attention_heads
    kg = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    if "gdn_vz" in groups and stem.endswith("linear_attn.in_proj_qkv"):
        m[2 * kg:] = True
    if "gdn_vz" in groups and stem.endswith("linear_attn.in_proj_z"):
        m[:] = True
    if "gdn_out" in groups and stem.endswith("linear_attn.out_proj"):
        m[:] = True
    if "attn_gv" in groups and stem.endswith("self_attn.q_proj"):
        for h in range(heads):
            m[h * 2 * d + d: h * 2 * d + 2 * d] = True
    if "attn_gv" in groups and stem.endswith("self_attn.v_proj"):
        m[:] = True
    if "attn_out" in groups and stem.endswith("self_attn.o_proj"):
        m[:] = True
    if "down" in groups and stem.endswith("mlp.down_proj"):
        m[:] = True
    if "gate_up" in groups and stem.endswith(("mlp.gate_proj", "mlp.up_proj")):
        m[:] = True
    if "gdn_qk" in groups and stem.endswith("linear_attn.in_proj_qkv"):
        m[:2 * kg] = True
    if "attn_qk" in groups and stem.endswith("self_attn.q_proj"):
        for h in range(heads):
            m[h * 2 * d: h * 2 * d + d] = True
    if "attn_qk" in groups and stem.endswith("self_attn.k_proj"):
        m[:] = True
    return m


STAGE_OF = {"in_proj_qkv": 0, "in_proj_z": 0, "q_proj": 0, "k_proj": 0, "v_proj": 0, "out_proj": 1, "o_proj": 1,
            "gate_proj": 2, "up_proj": 2, "down_proj": 3}
QUANT_SUFFIXES = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "in_proj_qkv.weight",
                  "in_proj_z.weight", "out_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight")


# ---- engine KV cache codecs (--kv; see tools/patch_fq_kv.py) ----
_HADAMARD = {}


def _h256(device):
    if device not in _HADAMARD:
        i = torch.arange(256)
        bits = (i[:, None] & i[None, :])
        parity = torch.zeros_like(bits)
        for b in range(8):
            parity ^= (bits >> b) & 1
        _HADAMARD[device] = ((1 - 2 * parity).float() / 16.0).to(device)
    return _HADAMARD[device]


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _fp8_rows(x):
    a = x.abs().amax(-1, keepdim=True)
    s = (a / 448.0).clamp(2.0 ** -24, 65504.0).half().float()
    s = torch.where(a == 0, torch.zeros_like(s), s)
    inv = torch.where(s == 0, torch.zeros_like(s), 1.0 / s)
    code = (x * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return code * s


def _nvfp4_g16(x):
    g = x.reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    a = g.abs().amax(-1, keepdim=True)
    s = (a / 6.0).clamp(2.0 ** -9, 448.0).to(torch.float8_e4m3fn).float()
    s = torch.where(a == 0, torch.zeros_like(s), s)
    q = torch.where(s == 0, torch.zeros_like(g), g / torch.where(s == 0, torch.ones_like(s), s))
    mag = _E2M1.to(x.device)[torch.bucketize(q.abs(), _E2M1_MID.to(x.device))]
    return (torch.sign(q) * mag * s).reshape(x.shape)


def _int8_g64(x):
    g = x.reshape(*x.shape[:-1], x.shape[-1] // 64, 64)
    a = g.abs().amax(-1, keepdim=True)
    s = (a / 127.0).half().float()
    inv = torch.where(s == 0, torch.zeros_like(s), 1.0 / s)
    return (torch.round(g * inv).clamp(-127, 127) * s).reshape(x.shape)


def kv_codec(x, which, kind):
    """Decoded engine cache row of K (which='k') or V ('v'), straight-through; x is [..., 256]."""
    if kind == "bf16":
        return x
    xf = x.float()
    H = _h256(x.device)
    if kind == "k8v4":
        y = (_fp8_rows(xf @ H) if which == "k" else _nvfp4_g16(xf @ H)) @ H
    elif kind == "fp8":
        y = _fp8_rows(xf @ H) @ H if which == "k" else _fp8_rows(xf)
    elif kind == "k8vi8":
        y = (_fp8_rows(xf @ H) if which == "k" else _int8_g64(xf @ H)) @ H
    elif kind == "k8v8":
        y = _fp8_rows(xf @ H) @ H
    elif kind == "int8":
        y = _int8_g64(xf @ H) @ H if which == "k" else _int8_g64(xf)
    else:
        raise ValueError(kind)
    y = y.to(x.dtype)
    return x + (y - x).detach()


def enable_kv_codec(kind):
    """Route every sdpa attention call through kv_codec (the HF Qwen3.5 attention passes post-RoPE K/V)."""
    if kind == "bf16":
        return
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    base = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_kv(module, query, key, value, *args, **kwargs):
        assert key.shape[-1] == 256 and value.shape[-1] == 256, key.shape
        return base(module, query, kv_codec(key, "k", kind), kv_codec(value, "v", kind), *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_kv
    print(f"KV codec in attention: {kind}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("windows"); ap.add_argument("out")
    ap.add_argument("--mode", required=True, choices=["ref", "rtn", "gptq", "codes", "exl3"])
    ap.add_argument("--codes", help="--mode codes: evaluate the codes store DIR (tools/gptq_recipe.py format)")
    ap.add_argument("--exl3", help="--mode exl3: ExLlamaV3 model DIR whose layer projections replace the weights")
    ap.add_argument("--codes-b", help="--mode codes: second store; rows of --groups come from it")
    ap.add_argument("--q4-groups", default="", help="gptq/rtn: also store these parent groups as Q4")
    ap.add_argument("--q4-layers", default="", help="layers where --q4-groups apply (default all)")
    ap.add_argument("--groups", default="", help="parent groups taken from --codes-b (build_hybrid.py names)")
    ap.add_argument("--codes-c", help="--mode codes: third store; rows of --groups-c come from it")
    ap.add_argument("--groups-c", default="", help="per-group layer ranges taken from --codes-c (gate_up:0-15,...)")
    ap.add_argument("--group-layers", default="", help="layers (e.g. 0-15,40) where --groups apply; default all")
    ap.add_argument("--act-order", action="store_true", help="GPTQ in descending diag(H) order, static groups")
    ap.add_argument("--scale-grid", default="fp16", choices=["fp16", "ls8"],
                    help="layer-projection group scales: FP16, or 8-bit log grid under a per-row FP32 super-scale")
    ap.add_argument("--clip-grid", default="1.0",
                    help="comma-separated scale shrink factors searched per group (absmax*alpha/qmax)")
    ap.add_argument("--recipe", default="official", choices=["official", "allq4", "q4down", "allq3", "allq5"])
    ap.add_argument("--grid", default="int", choices=["int", "half"], help="layer-projection code grid (half: (c+1/2)*s)")
    ap.add_argument("--calib"); ap.add_argument("--ref"); ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--calib-host", action="store_true",
                    help="keep the calibration stream in pinned host memory (FP32), batches move to the GPU on use: large --calib sets")
    ap.add_argument("--vocab-rows", type=int, default=248077); ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--damp-diag", type=float, default=0.0, help="GPTQ: add d * diag(H) to H before --damp")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-codes", help="write a converter codes store (tools/gptq_recipe.py) to DIR")
    ap.add_argument("--save-hidden", action="store_true", help="also write OUT.hidden.pt (final-layer states)")
    ap.add_argument("--rotate", default="", help="KIND:SEED (haar|had): residual rotation folded into the weights")
    ap.add_argument("--asym", type=float, default=0.0, help="gptq: asymmetric calibration strength (see top)")
    ap.add_argument("--asym-damp", type=float, default=0.01, help="relative damping of the asymmetric least squares")
    ap.add_argument("--h-fp", action="store_true",
                    help="GPTQ Hessians from the unquantized model's inputs (FP stream) instead of the quantized stream")
    ap.add_argument("--true-seq", action="store_true", help="gptq: quantize each layer in four stages, recapturing "
                    "inputs after each (mixer in, mixer out, MLP gate/up, MLP down)")
    ap.add_argument("--mlp-ap", type=int, default=0, help="gptq: learned-rounding steps per MLP (0: off; see top)")
    ap.add_argument("--mlp-ap-layers", default="", help="layers for --mlp-ap, e.g. 16-31 (default all)")
    ap.add_argument("--mlp-ap-lr", type=float, default=5e-3, help="Adam lr of the latent codes (code units)")
    ap.add_argument("--mlp-ap-init", choices=("cont", "codes"), default="cont",
                    help="latent init: GPTQ's continuous pre-rounding values (cont) or the integer codes")
    ap.add_argument("--mlp-ap-slr", type=float, default=1e-3, help="Adam lr of the log group scales")
    ap.add_argument("--mlp-ap-batch", type=int, default=4096, help="--mlp-ap tokens per step")
    ap.add_argument("--mlp-ap-target", default="auto", choices=["auto", "sym", "asym"],
                    help="--mlp-ap target (auto: asym with --asym, else sym)")
    ap.add_argument("--q3-groups", default="", help="gptq/rtn: parent groups on the 3-bit grid [-4, 3] with layer "
                    "ranges, e.g. gate_up:0-15,gdn_qk:0-31 (ranges joined by '+'); those rows must be Q4 in the recipe")
    ap.add_argument("--kv", default="bf16", choices=["bf16", "k8v4", "k8v8", "k8vi8", "fp8", "int8"],
                    help="engine KV cache codec applied to K/V in every attention layer (tools/patch_fq_kv.py)")
    ap.add_argument("--rotate-inputs", default="", help="KIND:SEED: rotate only the normed inputs of the residual "
                    "readers (online in an engine): quantize W R^T, apply Q(W R^T) R; writers and norms untouched")
    ap.add_argument("--mlp-stats", help="--mode ref: write per-layer MLP channel statistics to FILE and stop")
    ap.add_argument("--row-mask", help="--mode codes: torch dict {stem: bool[rows]}; those rows come from --codes-b")
    a = ap.parse_args()
    global ALPHAS, ACT_ORDER, Q4_GROUPS, Q4_LAYERS
    Q4_GROUPS = set(g for g in a.q4_groups.split(",") if g)
    if a.q4_layers:
        Q4_LAYERS = set()
        for part in a.q4_layers.split(","):
            lo, _, hi = part.partition("-"); Q4_LAYERS.update(range(int(lo), int(hi or lo) + 1))
    ALPHAS = tuple(float(v) for v in a.clip_grid.split(","))
    global SCALE_GRID, HALF
    LAYER_GRID = None if a.scale_grid == "fp16" else a.scale_grid
    ACT_ORDER = a.act_order
    DAMP_DIAG[0] = a.damp_diag
    store = Weights(a.codes) if a.mode == "codes" else None
    if a.mode == "exl3":
        from exllamav3.modules.quant.exl3 import LinearEXL3
        store = Weights(a.exl3); bits = {}
    store_b = Weights(a.codes_b) if a.mode == "codes" and a.codes_b else None
    row_mask = torch.load(a.row_mask) if a.row_mask else None
    assert row_mask is None or store_b is not None, "--row-mask needs --codes-b"
    assert not a.mlp_stats or a.mode == "ref", "--mlp-stats needs --mode ref"
    groups = set(g for g in a.groups.split(",") if g)
    group_layers = None
    if a.group_layers:
        group_layers = set()
        for part in a.group_layers.split(","):
            lo, _, hi = part.partition("-"); group_layers.update(range(int(lo), int(hi or lo) + 1))
    # --groups may also name per-group layer ranges: gate_up:0-15,down:0-15,gdn_qk:0-31 (ranges joined by '+').
    per_group = {}
    for g in list(groups):
        if ":" in g:
            name, _, spec = g.partition(":")
            lay = set()
            for part in spec.split("+"):
                lo, _, hi = part.partition("-"); lay.update(range(int(lo), int(hi or lo) + 1))
            per_group[name] = lay
            groups.discard(g)
    store_c = Weights(a.codes_c) if a.mode == "codes" and a.codes_c else None
    per_group_c = {}
    for g in (g for g in a.groups_c.split(",") if g):
        name, _, spec = g.partition(":")
        lay = set()
        for part in spec.split("+"):
            lo, _, hi = part.partition("-"); lay.update(range(int(lo), int(hi or lo) + 1))
        per_group_c[name] = lay
    assert not per_group_c or store_c is not None, "--groups-c needs --codes-c"
    q3_groups = {}
    for g in (g for g in a.q3_groups.split(",") if g):
        name, _, spec = g.partition(":")
        lay = set()
        for part in spec.split("+"):
            lo, _, hi = part.partition("-"); lay.update(range(int(lo), int(hi or lo) + 1))
        q3_groups[name] = lay
    use_fp = bool(a.asym) or a.h_fp  # the FP calibration stream is needed
    assert not (use_fp and (a.rotate or a.rotate_inputs)), "--asym/--h-fp with rotations is not supported"
    assert not (a.h_fp and a.asym), "--h-fp and --asym are exclusive"
    assert not use_fp or a.mode == "gptq", "--asym/--h-fp need --mode gptq"
    ap_layers = None
    if a.mlp_ap_layers:
        ap_layers = set()
        for part in a.mlp_ap_layers.split(","):
            lo, _, hi = part.partition("-"); ap_layers.update(range(int(lo), int(hi or lo) + 1))
    ap_asym = a.mlp_ap_target == "asym" or (a.mlp_ap_target == "auto" and a.asym > 0)
    assert not a.mlp_ap or (a.mode == "gptq" and a.grid == "int" and not a.rotate and not a.rotate_inputs)
    assert not (a.mlp_ap and ap_asym and not a.asym), "--mlp-ap-target asym needs --asym (the FP stream)"
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cfg = Qwen3_5TextConfig(**json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"])
    cfg._attn_implementation = "sdpa"
    enable_kv_codec(a.kv)
    W = Weights(a.model_dir)
    quant = a.mode != "ref"
    sets = {"eval": read_windows(a.windows)}
    if a.mode == "gptq":
        sets["calib"] = read_windows(a.calib)
    Lw = {k: len(v[0][0]) for k, v in sets.items()}
    for k, v in sets.items():
        assert all(len(t) == Lw[k] for t, _ in v)
    t0 = time.time()
    emb = W.get("model.language_model.embed_tokens.weight").to(dev)
    R = rotation(a.rotate, cfg.hidden_size, dev) if a.rotate else None
    RI = rotation(a.rotate_inputs, cfg.hidden_size, dev) if a.rotate_inputs else None
    READERS = ("in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
    if R is not None:
        emb = emb.float()
        for c0 in range(0, emb.shape[0], 16384):
            emb[c0:c0 + 16384] = emb[c0:c0 + 16384] @ R.T
    calib_fp = None
    if use_fp:  # the FP stream starts from the unquantized embedding and lives in pinned host memory
        ids = torch.tensor(np.stack([t for t, _ in sets["calib"]]), dtype=torch.long, device=dev)
        calib_fp = torch.empty((ids.shape[0], Lw["calib"], cfg.hidden_size), dtype=torch.float32).pin_memory()
        for b in range(ids.shape[0]):
            calib_fp[b] = emb[ids[b]].float().cpu()
        del ids
    if quant:
        lo, hi, g = FMT["q8"]; emb = rtn(emb, lo, hi, g, alphas=(1.0,))  # the converter's absmax
    hidden = {k: emb.float()[torch.tensor(np.stack([t for t, _ in v]), dtype=torch.long, device=dev)] for k, v in sets.items()
              if not (k == "calib" and a.calib_host)}
    if "calib" in sets and a.calib_host:
        ids = torch.tensor(np.stack([t for t, _ in sets["calib"]]), dtype=torch.long, device=dev)
        hidden["calib"] = torch.empty((ids.shape[0], Lw["calib"], cfg.hidden_size), dtype=torch.float32).pin_memory()
        for b in range(ids.shape[0]):
            hidden["calib"][b] = emb[ids[b]].float().cpu()
        del ids
    del emb; torch.cuda.empty_cache()
    rot = Qwen3_5TextRotaryEmbedding(cfg).to(dev)
    pes = {k: rot(hidden[k][: a.batch].to(dev), torch.arange(Lw[k], device=dev)[None].expand(a.batch, Lw[k])) for k in sets}
    poss = {k: torch.arange(Lw[k], device=dev)[None].expand(a.batch, Lw[k]) for k in sets}

    ACT = os.environ.get("FQ_ACT", "fp32")
    assert ACT in ("fp32", "bf16", "bf16r32"), ACT
    if ACT != "fp32":
        print(f"FQ_ACT={ACT}: projection inputs/outputs rounded to BF16, residual "
              + ("BF16" if ACT == "bf16" else "FP32"), flush=True)

    def bf(t):
        return t.bfloat16().float()

    def act_forward(layer, x, pe, pos):
        """Qwen3_5DecoderLayer.forward with the engine's BF16 storage points (FQ_ACT)."""
        hooks = []
        for mod in layer.modules():
            if isinstance(mod, torch.nn.Linear):
                hooks.append(mod.register_forward_pre_hook(lambda m_, inp: (bf(inp[0]),) + tuple(inp[1:])))
                hooks.append(mod.register_forward_hook(lambda m_, inp, out: bf(out)))
        try:
            rnd = bf if ACT == "bf16" else (lambda t: t)
            r = rnd(x)
            h = layer.input_layernorm(r)
            if layer.block_type == "linear_attention":
                h = layer.linear_attn(hidden_states=h, cache_params=None, attention_mask=None)
            else:
                h, _ = layer.self_attn(hidden_states=h, attention_mask=None, position_ids=pos, past_key_values=None,
                                       position_embeddings=pe)
            r = rnd(r + h)
            r = rnd(r + layer.mlp(layer.post_attention_layernorm(r)))
            return r
        finally:
            for hk in hooks:
                hk.remove()

    def run(layer, key, write=True):
        h = hidden[key]
        with torch.no_grad():
            for b in range(0, h.shape[0], a.batch):
                x = h[b: b + a.batch].to(dev, non_blocking=True); m = x.shape[0]
                pe = (pes[key][0][:m], pes[key][1][:m])
                if ACT != "fp32" and key == "eval":
                    y = act_forward(layer, x, pe, poss[key][:m])
                else:
                    y = layer(x, position_embeddings=pe, attention_mask=None, position_ids=poss[key][:m])
                if write:
                    h[b: b + a.batch].copy_(y)

    class _Stop(Exception):
        pass

    def capture(layer, layer_fp, lin, last):
        """H = X_q^T X_q for the projections `lin` (name -> module of `layer`) on the quantized calibration stream,
        `layer` as quantized so far; with --asym also C = X_q^T (X_fp - X_q), X_fp from the unquantized `layer_fp` on
        the FP stream, which advances when `last`. A forward stops once every projection in `lin` has its input."""
        Hs = {n: torch.zeros(mm.in_features, mm.in_features, device=dev) for n, mm in lin.items()}
        Cs = {n: torch.zeros_like(Hs[n]) for n in lin} if a.asym else None
        cnt = {n: 0 for n in lin}
        fp_mods = dict(layer_fp.named_modules()) if use_fp else None
        hq = hidden["calib"]
        xfp, seen = {}, set()

        def record(name):
            def f(mod, inp):
                xfp[name] = inp[0].reshape(-1, inp[0].shape[-1]).float(); seen.add(name)
                if not last and len(seen) == len(lin):
                    raise _Stop
            return f

        def acc(name):
            def f(mod, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                xh = xfp[name] if a.h_fp else x
                Hs[name].addmm_(xh.T, xh); cnt[name] += x.shape[0]
                if Cs is not None:
                    Cs[name].addmm_(x.T, xfp[name] - x)
                seen.add(name)
                if len(seen) == len(lin):
                    raise _Stop
            return f
        with torch.no_grad():
            for b in range(0, hq.shape[0], a.batch):
                m_ = min(a.batch, hq.shape[0] - b)
                kw = dict(position_embeddings=(pes["calib"][0][:m_], pes["calib"][1][:m_]), attention_mask=None,
                          position_ids=poss["calib"][:m_])
                if use_fp:
                    seen.clear()
                    hs = [fp_mods[n].register_forward_pre_hook(record(n)) for n in lin]
                    try:
                        y = layer_fp(calib_fp[b: b + m_].to(dev, non_blocking=True), **kw)
                        if last:
                            calib_fp[b: b + m_].copy_(y.cpu())
                        del y
                    except _Stop:
                        pass
                    finally:
                        for hh in hs:
                            hh.remove()
                seen.clear()
                hs = [mm.register_forward_pre_hook(acc(n)) for n, mm in lin.items()]
                try:
                    layer(hq[b: b + m_].to(dev, non_blocking=True), **kw)
                except _Stop:
                    pass
                finally:
                    for hh in hs:
                        hh.remove()
                xfp.clear()
        return Hs, cnt, Cs

    ap_buf = {}

    def mlp_ap(layer, i, qres, saved, prefix):
        """--mlp-ap for layer i's MLP; qres["mlp.NAME"] = [codes, scales, qmin, qmax, FP weight] from GPTQ."""
        H = cfg.hidden_size
        hq = hidden["calib"]; N, L = hq.shape[0], hq.shape[1]
        if not ap_buf:  # host staging for inputs and targets, reused across layers
            ap_buf["X"] = torch.empty((N * L, H), dtype=torch.bfloat16).pin_memory()
            ap_buf["T"] = torch.empty((N * L, H), dtype=torch.bfloat16).pin_memory()
        X, T = ap_buf["X"], ap_buf["T"]
        names = ("gate_proj", "up_proj", "down_proj")
        fp = {nm: qres["mlp." + nm][4] for nm in names}
        got = {}

        def pre_norm(mod, inp):
            got["r"] = inp[0]

        def pre_mlp(mod, inp):
            got["x"] = inp[0]
            raise _Stop
        hs = [layer.post_attention_layernorm.register_forward_pre_hook(pre_norm),
              layer.mlp.register_forward_pre_hook(pre_mlp)]
        with torch.no_grad():
            for b in range(0, N, a.batch):
                m_ = min(a.batch, N - b)
                try:
                    layer(hq[b: b + m_].to(dev, non_blocking=True), position_embeddings=(pes["calib"][0][:m_], pes["calib"][1][:m_]),
                          attention_mask=None, position_ids=poss["calib"][:m_])
                except _Stop:
                    pass
                x = got["x"].reshape(-1, H).float()
                if ap_asym:
                    t = calib_fp[b: b + m_].to(dev).reshape(-1, H) - got["r"].reshape(-1, H).float()
                else:
                    t = (torch.nn.functional.silu(x @ fp["gate_proj"].T) * (x @ fp["up_proj"].T)) @ fp["down_proj"].T
                X[b * L: (b + m_) * L].copy_(x.bfloat16().cpu()); T[b * L: (b + m_) * L].copy_(t.bfloat16().cpu())
                del x, t
        for hh in hs:
            hh.remove()
        got.clear()
        win = torch.arange(N * L) // L
        val_idx = torch.nonzero(win % 20 == 19)[:, 0]
        tr_idx = torch.nonzero(win % 20 != 19)[:, 0]
        if val_idx.numel() == 0:  # fewer than 20 windows (smoke tests): hold out nothing
            val_idx = tr_idx
        val_idx = val_idx[torch.randperm(val_idx.numel(), generator=torch.Generator().manual_seed(1))[:16384]]
        Xv, Tv = X[val_idx].to(dev).float(), T[val_idx].to(dev).float()
        P = {}
        for nm in names:
            codes, scales, qmin, qmax, _, U0 = qres["mlp." + nm]
            if U0 is None or a.mlp_ap_init == "codes":
                U0 = codes.float().clone()
            else:  # GPTQ's own continuous values: round(U0) = codes, so training starts at the GPTQ solution with
                # each weight's distance to its rounding boundary intact (integer init would put every weight 0.5 away)
                lo_, hi_ = qmin.float()[:, None] - 0.49, qmax.float()[:, None] + 0.49
                U0 = torch.minimum(torch.maximum(U0.float(), lo_), hi_)
                assert torch.equal(torch.clamp(torch.round(U0), qmin.float()[:, None], qmax.float()[:, None]), codes.float()), "U0 does not round to the codes"
            qres["mlp." + nm][5] = None
            P[nm] = (U0.requires_grad_(True),
                     torch.log(scales.float().clamp(min=1e-12)).requires_grad_(True),
                     qmin.float()[:, None, None], qmax.float()[:, None, None])

        def qdq(nm):
            U, S, lo, hi = P[nm]; rows, k = U.shape
            Ug = U.view(rows, k // 64, 64)
            q = Ug + (torch.clamp(torch.round(Ug), lo, hi) - Ug).detach()
            return (q * torch.exp(S)[..., None]).view(rows, k)

        def fwd(x):
            return (torch.nn.functional.silu(x @ qdq("gate_proj").T) * (x @ qdq("up_proj").T)) @ qdq("down_proj").T

        def snapshot():
            with torch.no_grad():
                return {nm: (torch.clamp(torch.round(P[nm][0].view(P[nm][0].shape[0], -1, 64)), P[nm][2], P[nm][3])
                             .view(P[nm][0].shape).to(torch.int8), torch.exp(P[nm][1]).half()) for nm in names}

        def val_loss():
            with torch.no_grad():
                Wg, Wu, Wd = qdq("gate_proj"), qdq("up_proj"), qdq("down_proj")
                err = 0.0
                for c0 in range(0, Xv.shape[0], 2048):
                    x = Xv[c0: c0 + 2048]
                    err += ((((torch.nn.functional.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T) - Tv[c0: c0 + 2048]) ** 2).sum().item()
                return err / Xv.shape[0] / H / (Tv ** 2).mean().item()
        opt = torch.optim.Adam([{"params": [P[nm][0] for nm in names], "lr": a.mlp_ap_lr},
                                {"params": [P[nm][1] for nm in names], "lr": a.mlp_ap_slr}])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
        v0 = best = val_loss(); best_state = snapshot(); best_step = 0
        perm = tr_idx[torch.randperm(tr_idx.numel(), generator=torch.Generator().manual_seed(i))]
        B = a.mlp_ap_batch; pos = 0; tl = float("nan")
        for step in range(a.mlp_ap):
            if pos + B > perm.numel():
                perm = perm[torch.randperm(perm.numel())]; pos = 0
            idx = perm[pos: pos + B]; pos += B
            x = X[idx].to(dev, non_blocking=True).float(); t = T[idx].to(dev, non_blocking=True).float()
            loss = ((fwd(x) - t) ** 2).mean() / (t ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tl = loss.item() if step % 50 == 49 else tl
            if step % 50 == 49 or step == a.mlp_ap - 1:
                v = val_loss()
                if v < best:
                    best, best_state, best_step = v, snapshot(), step + 1
        torch.backends.cuda.matmul.allow_tf32 = False
        changed = []
        for nm in names:
            codes, scales = best_state[nm]
            changed.append((codes != qres["mlp." + nm][0]).float().mean().item())
            mod = getattr(layer.mlp, nm)
            mod.weight.data = (codes.float().view(codes.shape[0], -1, 64) * scales.float()[..., None]).view(codes.shape)
            stem = prefix + "mlp." + nm
            if a.save_codes:
                saved[stem + ".gptq_codes"] = codes.cpu().contiguous()
                saved[stem + ".gptq_scales"] = scales.cpu().contiguous()
        ap_stats.append((i, v0, best, best_step, changed))
        print(f"  mlp-ap layer {i}: held-out rel mse {v0:.5f} -> {best:.5f} at step {best_step} (train {tl:.5f}); "
              f"codes changed {' '.join(f'{c:.4f}' for c in changed)}; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        del P, opt, Xv, Tv, best_state

    mstats, nrm = {}, [0]

    def mlp_stat_hooks(layer, i):
        f64 = dict(dtype=torch.float64, device=dev)
        st = mstats.setdefault(i, {"n": 0, "ex2": torch.zeros(cfg.hidden_size, **f64),
                                   **{k: torch.zeros(cfg.intermediate_size, **f64) for k in ("A", "B", "H")}})
        buf = {}

        def g_hook(mod, inp, out):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            st["ex2"] += (x * x).sum(0).double(); st["n"] += x.shape[0]
            buf["g"] = out.reshape(-1, out.shape[-1]).float()

        def u_hook(mod, inp, out):
            g = buf.pop("g"); u = out.reshape(-1, out.shape[-1]).float()
            sg = torch.sigmoid(g); sl = g * sg
            st["A"] += ((sg * (1 + g * (1 - sg)) * u) ** 2).sum(0).double()
            st["B"] += (sl * sl).sum(0).double()
            st["H"] += ((sl * u) ** 2).sum(0).double()

        return [layer.mlp.gate_proj.register_forward_hook(g_hook), layer.mlp.up_proj.register_forward_hook(u_hook)]

    stats = []
    ap_stats = []
    asym_stats = []
    weight_map = {}
    if a.save_codes:
        os.makedirs(a.save_codes, exist_ok=True)
        shutil.copy(os.path.join(a.model_dir, "config.json"), a.save_codes)
    for i in range(int(os.environ.get("FQ_MAX_LAYERS", cfg.num_hidden_layers))):  # FQ_MAX_LAYERS: smoke tests
        saved = {}
        layer = Qwen3_5DecoderLayer(cfg, i)
        prefix = f"model.language_model.layers.{i}."
        sd = {k[len(prefix):]: W.get(k) for k in W.map if k.startswith(prefix)}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        assert not unexpected and not [m for m in missing if not m.endswith("inv_freq")], (missing, unexpected)
        layer = layer.to(dev, dtype=torch.float32)
        if R is not None:
            rotate_layer(layer, R)
        linears = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear) and (n + ".weight").endswith(QUANT_SUFFIXES)}
        # --true-seq: quantize the layer in stages (mixer inputs, mixer output, MLP gate/up, MLP down), recapturing
        # each stage's inputs with the earlier stages already quantized (GPTQ's "true sequential").
        if a.true_seq and a.mode == "gptq":
            stages = [[n for n in linears if STAGE_OF[n.split(".")[-1]] == st] for st in range(4)]
            stages = [st for st in stages if st]
        else:
            stages = [list(linears)]
        layer_fp = copy.deepcopy(layer) if (use_fp and len(stages) > 1) else layer
        ap_now = a.mlp_ap > 0 and (ap_layers is None or i in ap_layers)
        qres = {}
        if quant:
            SCALE_GRID = LAYER_GRID
            HALF = 0.5 if a.grid == "half" else 0.0
        for si, stage in enumerate(stages):
            if a.mode == "gptq":
                Hs, cnt, Cs = capture(layer, layer_fp, {n: linears[n] for n in stage}, si == len(stages) - 1)
            if not quant:
                continue
            for n in stage:
                m = linears[n]
                full = prefix + n + ".weight"
                q4 = torch.tensor(row_formats(full, m.out_features, a.recipe, cfg), device=dev)
                qmin = torch.where(q4, torch.tensor(-8.0, device=dev), torch.tensor(-16.0, device=dev))
                qmax = torch.where(q4, torch.tensor(7.0, device=dev), torch.tensor(15.0, device=dev))
                if a.recipe == "allq3":  # every layer projection on the 3-bit grid [-4, 3]
                    qmin, qmax = torch.full_like(qmin, -4.0), torch.full_like(qmax, 3.0)
                q3_active = {g for g, lay in q3_groups.items() if i in lay}
                if q3_active:
                    q3m = group_rows(full.removesuffix(".weight"), m.out_features, q3_active, cfg).to(dev)
                    assert bool(q4[q3m].all()), f"{full}: --q3-groups rows must be Q4 in the recipe"
                    qmin = torch.where(q3m, torch.tensor(-4.0, device=dev), qmin)
                    qmax = torch.where(q3m, torch.tensor(3.0, device=dev), qmax)
                w = m.weight.data
                rin = RI is not None and n.split(".")[-1] in READERS
                if rin:
                    w = w @ RI.T
                if a.mode == "codes":
                    stem = full.removesuffix(".weight")
                    codes, scales = store.get(stem + ".gptq_codes").to(dev), store.get(stem + ".gptq_scales").to(dev)
                    active = groups | {g for g, lay in per_group.items() if i in lay}
                    if store_b is not None and active and (group_layers is None or i in group_layers or per_group):
                        mask = group_rows(stem, codes.shape[0], active if per_group else groups, cfg).to(dev)
                        if bool(mask.any()):
                            codes[mask] = store_b.get(stem + ".gptq_codes").to(dev)[mask]
                            scales[mask] = store_b.get(stem + ".gptq_scales").to(dev)[mask]
                    active_c = {g for g, lay in per_group_c.items() if i in lay}
                    if active_c:
                        mask_c = group_rows(stem, codes.shape[0], active_c, cfg).to(dev)
                        if bool(mask_c.any()):
                            if store_b is not None and active:
                                assert not bool((mask_c & group_rows(stem, codes.shape[0], active if per_group else groups,
                                                                     cfg).to(dev)).any()), f"{full}: rows in both B and C"
                            codes[mask_c] = store_c.get(stem + ".gptq_codes").to(dev)[mask_c]
                            scales[mask_c] = store_c.get(stem + ".gptq_scales").to(dev)[mask_c]
                    if row_mask is not None and stem in row_mask:
                        rm = row_mask[stem].to(dev)
                        assert rm.shape[0] == codes.shape[0], (stem, rm.shape)
                        codes[rm] = store_b.get(stem + ".gptq_codes").to(dev)[rm]
                        scales[rm] = store_b.get(stem + ".gptq_scales").to(dev)[rm]
                        nrm[0] += int(rm.sum())
                    assert bool((codes >= -16).all() and (codes <= 15).all()), full  # Q4 or Q5 rows
                    sc = scales.float()
                    if a.scale_grid == "ls8":
                        sup = sc.amax(1, keepdim=True)
                        j = torch.where(sc > 0, torch.round(-32.0 * torch.log2(sc / sup)).clamp(0, 255), torch.full_like(sc, 255.0))
                        sc = torch.where(sc > 0, sup * torch.exp2(-j / 32.0), torch.zeros_like(sc))
                    new = (codes.float().reshape(w.shape[0], -1, 64) * sc[..., None]).reshape(w.shape)
                elif a.mode == "exl3":
                    stem = full.removesuffix(".weight")
                    t = {k: store.get(stem + "." + k).to(dev) for k in ("trellis", "suh", "svh", "mcg", "mul1")
                         if stem + "." + k in store.map}
                    lin = LinearEXL3(None, m.in_features, m.out_features, **t, key=stem)
                    new = lin.get_weight_tensor().float().T.contiguous(); bits[stem] = lin.K
                    assert new.shape == w.shape, (full, new.shape, w.shape)
                    del lin, t
                elif a.mode == "rtn":
                    new, codes, scales = rtn(w, qmin, qmax, 64, codes=True)
                else:
                    H = Hs[n] * (2.0 / max(1, cnt[n]))
                    if rin:
                        H = RI @ H @ RI.T
                    if a.asym:  # least-squares target: W + A * (W C^T) H_d^-1, C = X_q^T (X_fp - X_q)
                        C = Cs.pop(n) * (2.0 / max(1, cnt[n]))
                        Hd = H.clone()
                        dead = torch.diag(Hd) == 0
                        Hd[dead, dead] = 1
                        Hd += a.asym_damp * torch.mean(torch.diag(Hd)) * torch.eye(Hd.shape[0], device=dev)
                        Lc = torch.linalg.cholesky(Hd)
                        delta = torch.cholesky_solve((w.float() @ C.T).T.contiguous(), Lc).T
                        delta[:, dead] = 0
                        asym_stats.append((i, n, (delta.norm() / w.norm()).item()))
                        w = w.float() + a.asym * delta
                        del C, Hd, Lc, delta
                    KEEP_U[0] = ap_now and n.startswith("mlp.")
                    new, codes, scales = gptq(w, H, qmin, qmax, damp=a.damp)
                    KEEP_U[0] = False
                    del H
                if a.save_codes:
                    stem = full.removesuffix(".weight")
                    if HALF:  # (c + 1/2) s = (2c + 1)(s / 2): odd integer codes, exact, importable as Q4/Q5 codes
                        codes, scales = (2 * codes.to(torch.int16) + 1).to(torch.int8), (scales.float() / 2).half()
                    saved[stem + ".gptq_codes"] = codes.cpu().contiguous()
                    saved[stem + ".gptq_scales"] = scales.cpu().contiguous()
                if ap_now and n.startswith("mlp."):
                    qres[n] = [codes, scales, qmin, qmax, m.weight.data.clone(), KEEP_U[1]]
                KEEP_U[1] = None
                rel = ((new - m.weight.data).norm() / m.weight.data.norm()).item() if not rin else \
                    ((new - w).norm() / w.norm()).item()
                stats.append((i, n, rel))
                m.weight.data = new @ RI if rin else new
            if a.mode == "gptq":
                del Hs, Cs
        if ap_now:
            del layer_fp
            layer_fp = None
            mlp_ap(layer, i, qres, saved, prefix)
        del qres
        if quant:
            SCALE_GRID = None
            HALF = 0.0
        del layer_fp
        if saved:
            fname = f"codes-{i:03d}.safetensors"
            save_file(saved, os.path.join(a.save_codes, fname))
            weight_map.update({t: fname for t in saved})
        hooks = mlp_stat_hooks(layer, i) if a.mlp_stats else []
        for key in sets:
            run(layer, key)
        for hk in hooks:
            hk.remove()
        del layer, sd; torch.cuda.empty_cache()
        print(f"layer {i} done {time.time() - t0:.0f}s" + (f"  mean rel err {np.mean([s[2] for s in stats if s[0] == i]):.4f}" if quant else "")
              + (f"  asym |dW|/|W| {np.mean([s[2] for s in asym_stats if s[0] == i]):.4f}" if a.asym else ""), flush=True)
    if nrm[0]:
        print(f"--row-mask: {nrm[0]} rows from --codes-b", flush=True)
    if a.mlp_stats:
        torch.save({i: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st.items()} for i, st in mstats.items()},
                   a.mlp_stats)
        print(f"MLP channel stats of {len(mstats)} layers -> {a.mlp_stats}; done {time.time() - t0:.0f}s", flush=True)
        return
    hidden.pop("calib", None); calib_fp = None; torch.cuda.empty_cache()
    if a.save_hidden:
        torch.save(hidden["eval"].cpu(), a.out + ".hidden.pt")  # final-layer states, for head-only variants
    norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = W.get("model.language_model.norm.weight").float(); norm = norm.to(dev)
    head = W.get("lm_head.weight").to(dev)[: a.vocab_rows].float()
    if R is not None:
        scale = 1.0 + norm.weight.data
        for c0 in range(0, head.shape[0], 16384):
            head[c0:c0 + 16384] = (head[c0:c0 + 16384] * scale[None, :]) @ R.T
        norm.weight.data.zero_()
    if quant:
        lo, hi, g = FMT["q8"]; head = rtn(head, lo, hi, g, alphas=(1.0,))
    ref = np.fromfile(a.ref, dtype=np.float32).reshape(-1, 1 + 2 * TOP) if a.ref else None
    col = 0
    with open(a.out + ".topk.bin", "wb") as f, open(a.out + ".target.bin", "wb") as ft, \
            (open(a.out + ".atref.bin", "wb") if a.ref else open(os.devnull, "wb")) as fa, torch.no_grad():
        for w_i, (toks, first) in enumerate(sets["eval"]):
            L = len(toks)
            h = norm(hidden["eval"][w_i: w_i + 1])[0][first - 1: L - 1]
            if ACT != "fp32":
                h = bf(h)  # the engine's head reads the BF16 final-norm output
            targets = torch.tensor(toks[first:L], dtype=torch.long, device=dev)
            for c0 in range(0, h.shape[0], 512):
                logits = h[c0: c0 + 512] @ head.T
                lse = torch.logsumexp(logits, -1)
                ft.write((logits.gather(1, targets[c0: c0 + 512, None])[:, 0] - lse).cpu().numpy().astype(np.float32).tobytes())
                val, idx = torch.topk(logits, TOP, -1)
                rec = np.zeros((logits.shape[0], 1 + 2 * TOP), dtype=np.float32)
                rec[:, 0] = lse.cpu().numpy(); rec[:, 1:1 + TOP] = idx.int().cpu().numpy().view(np.float32); rec[:, 1 + TOP:] = val.cpu().numpy()
                f.write(rec.tobytes())
                if ref is not None:
                    rid = torch.from_numpy(ref[col: col + logits.shape[0], 1:1 + TOP].copy().view(np.int32)).long().to(dev)
                    fa.write(logits.gather(1, rid).cpu().numpy().astype(np.float32).tobytes())
                col += logits.shape[0]
    if stats:
        json.dump(stats, open(a.out + ".relerr.json", "w"))
    if asym_stats:
        json.dump(asym_stats, open(a.out + ".asym.json", "w"))
    if ap_stats:
        json.dump(ap_stats, open(a.out + ".mlpap.json", "w"))
    if a.mode == "exl3":
        json.dump(bits, open(a.out + ".exl3bits.json", "w"))
    if a.save_codes:
        json.dump({"metadata": {"mode": a.mode, "recipe": a.recipe, "damp": a.damp, "damp_diag": a.damp_diag, "calib": a.calib, "asym": a.asym, "h_fp": a.h_fp,
                                "asym_damp": a.asym_damp, "q3_groups": a.q3_groups,
                                "true_seq": a.true_seq, "mlp_ap": a.mlp_ap, "mlp_ap_layers": a.mlp_ap_layers},
                   "weight_map": weight_map}, open(os.path.join(a.save_codes, "model.safetensors.index.json"), "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
