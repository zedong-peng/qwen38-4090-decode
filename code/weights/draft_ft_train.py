#!/usr/bin/env python3
"""DFlash2 drafter fine-tuning on target self-distillation data (tools/draft_ft_features.py).

Forward semantics follow z-lab/dflash dflash/model.py (DFlash2DraftModel): the block is
[anchor, mask x (B-1)] embedded with the target's input embeddings; every layer attends
bidirectionally inside the block and to the context keys/values, which are k/v projections of
hidden_norm(fc(target features)) for the positions before the anchor (2048-token sliding window).
Grouped dynamic causal convolutions run along the block. Logits use the target's lm_head; the
candidate selector scores the top-k candidates with a bilinear (predecessor, hidden, successor) term.

Training: many anchors per sequence (one context pass shared by all of them), CE on positions
1..B-1 with weights DECAY^(k-1); selector CE over the top-k candidates (teacher-forced
predecessor, detached unary). With --soft W and dumped target final-norm states (draft_ft_features.py
FINAL=1), the position loss is (1-W)·CE(greedy token) + W·CE(target top-K distribution), the target
distribution rebuilt as softmax(final · lm_headᵀ) restricted to its top --soft-topk tokens.
LoRA on fc and every layer projection; norms, conv kernels and the
selector are trained fully. --eval-only reports per-position top-1 accuracy and greedy chain length.

usage: draft_ft_train.py --data DIR [--data DIR ...] --drafter DIR --target DIR --out DIR [options]"""
import argparse, json, math, os, random, threading, queue, time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--data", action="append", required=True, help="dir with index.jsonl and dump/fc.bf16")
ap.add_argument("--drafter", required=True); ap.add_argument("--target", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--block", type=int, default=16)
ap.add_argument("--anchors", type=int, default=32, help="anchors per sequence per step")
ap.add_argument("--rank", type=int, default=64); ap.add_argument("--alpha", type=float, default=64)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr-full", type=float, default=2e-5)
ap.add_argument("--decay", type=float, default=0.9); ap.add_argument("--selector-weight", type=float, default=0.5)
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--holdout", type=float, default=0.03); ap.add_argument("--seed", type=int, default=20261003)
ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--eval-seqs", type=int, default=200)
ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--load-lora", default="")
ap.add_argument("--soft", type=float, default=0.0); ap.add_argument("--soft-topk", type=int, default=64)
ap.add_argument("--soft-temp", type=float, default=1.0, help="target temperature; < 1 sharpens toward the greedy argmax")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
os.makedirs(a.out, exist_ok=True)
cfg = json.load(open(os.path.join(a.drafter, "config.json")))
dc = cfg["dflash_config"]
H, NH, NKV, HD = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
NL, EPS, WIN = cfg["num_hidden_layers"], cfg["rms_norm_eps"], cfg.get("sliding_window", 2048)
THETA = cfg.get("rope_parameters", {}).get("rope_theta", cfg.get("rope_theta", 1e7))
MASK, TOPK, GS, KS = dc["mask_token_id"], dc["selector_top_k"], dc["conv_group_size"], dc["conv_kernel_size"]
B = a.block

# ----------------------------------------------------------------------------- weights
dw = safe_open(os.path.join(a.drafter, "model.safetensors"), "pt", device="cpu")
P = {k: dw.get_tensor(k) for k in dw.keys()}
tmap = json.load(open(os.path.join(a.target, "model.safetensors.index.json")))["weight_map"]
def tget(name):
    with safe_open(os.path.join(a.target, tmap[name]), "pt", device="cpu") as f:
        return f.get_tensor(name)
embed = tget("model.language_model.embed_tokens.weight")            # CPU, gathered per batch
head = tget("lm_head.weight").to(dev)                                # [V, H] BF16


class LoRALinear(torch.nn.Module):
    def __init__(self, w, rank, alpha):
        super().__init__()
        self.w = w.to(dev, torch.bfloat16)                           # frozen
        out_f, in_f = self.w.shape
        self.A = torch.nn.Parameter(torch.randn(rank, in_f, device=dev) / math.sqrt(in_f))
        self.B = torch.nn.Parameter(torch.zeros(out_f, rank, device=dev))
        self.s = alpha / rank

    def forward(self, x):
        y = x @ self.w.T
        return y + ((x.float() @ self.A.T) @ self.B.T * self.s).to(y.dtype)

    def merged(self):
        return (self.w.float() + self.s * self.B @ self.A).to(torch.bfloat16)


def full(name):
    return torch.nn.Parameter(P[name].float().to(dev))


class Drafter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        r, al = a.rank, a.alpha
        self.fc = LoRALinear(P["fc.weight"], r, al)
        self.hidden_norm = full("hidden_norm.weight"); self.norm = full("norm.weight")
        self.layers = torch.nn.ModuleList()
        for i in range(NL):
            p = f"layers.{i}."
            L = torch.nn.Module()
            for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(L, n, LoRALinear(P[p + "self_attn." + n + ".weight"], r, al))
            for n in ("gate_proj", "up_proj", "down_proj"):
                setattr(L, n, LoRALinear(P[p + "mlp." + n + ".weight"], r, al))
            L.q_norm = full(p + "self_attn.q_norm.weight"); L.k_norm = full(p + "self_attn.k_norm.weight")
            L.in_norm = full(p + "input_layernorm.weight"); L.post_norm = full(p + "post_attention_layernorm.weight")
            for c in ("attention_conv", "mlp_conv"):
                setattr(L, c + "_base", full(p + c + ".base_kernel"))
                setattr(L, c + "_proj", LoRALinear(P[p + c + ".kernel_projection.weight"], r, al))
            self.layers.append(L)
        self.sel_hidden = full("candidate_selector.hidden_projection.weight")
        self.sel_pred = full("candidate_selector.predecessor_codebook")
        self.sel_succ = full("candidate_selector.successor_codebook")


def rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w).to(torch.bfloat16)


def rope(pos):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=dev).float() / HD))
    f = pos.float()[..., None] * inv
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):          # x [..., L, heads, HD]; cos [..., L, HD]
    x1, x2 = x.float().chunk(2, -1)
    rot = torch.cat([-x2, x1], -1)
    return (x.float() * cos[..., None, :] + rot * sin[..., None, :]).to(torch.bfloat16)


def dyn_conv(h, dyn, base):           # h [M, B, H]; dyn [M, B, KS, G]; base [KS, H]
    M_, L_, _ = h.shape; G = H // GS
    blocks = h.view(M_, L_, G, GS)
    out = torch.zeros_like(blocks, dtype=torch.float32)
    for off in range(KS):
        v = blocks if off == 0 else F.pad(blocks[:, :-off], (0, 0, 0, 0, off, 0))
        out = out + base[off].view(1, 1, G, GS) * v.float() + dyn[:, :, off, :, None].float() * v.float()
    return out.to(torch.bfloat16).view_as(h)


def conv_prepare(L, x, which):
    proj = getattr(L, which + "_proj")(x).view(*x.shape[:-1], 2, KS, H // GS)
    return dyn_conv(x, proj[..., 0, :, :], getattr(L, which + "_base")[0]), proj[..., 1, :, :]


def forward(m, feats, ids_t, anchors):
    """feats [Lc, 5H] BF16 (positions 0..Lc-1), anchors [M] (positions, all <= Lc).
    Returns final hidden [M, B-1, H]."""
    M_ = anchors.shape[0]
    Lc = feats.shape[0]
    ctx = rms(m.fc(feats), m.hidden_norm)                                   # [Lc, H]
    blk_ids = torch.full((M_, B), MASK, dtype=torch.long)
    blk_ids[:, 0] = ids_t[anchors.cpu()]
    h = embed[blk_ids].to(dev)                                               # [M, B, H]
    qpos = anchors[:, None] + torch.arange(B, device=dev)[None]             # [M, B]
    qcos, qsin = rope(qpos)
    kcos, ksin = rope(torch.arange(Lc, device=dev))
    kpos = torch.arange(Lc, device=dev)
    vis_ctx = (kpos[None, None, :] < anchors[:, None, None]) & (qpos[:, :, None] - kpos[None, None, :] < WIN)
    mask = torch.cat([vis_ctx, torch.ones(M_, B, B, dtype=torch.bool, device=dev)], -1)[:, None]  # [M,1,B,Lc+B]
    for L in m.layers:
        res = h
        x = rms(h, L.in_norm)
        x, kern = conv_prepare(L, x, "attention_conv")
        q = rms(L.q_proj(x).view(M_, B, NH, HD), L.q_norm)
        kb = rms(L.k_proj(x).view(M_, B, NKV, HD), L.k_norm)
        vb = L.v_proj(x).view(M_, B, NKV, HD)
        kc = rms(L.k_proj(ctx).view(Lc, NKV, HD), L.k_norm)
        vc = L.v_proj(ctx).view(Lc, NKV, HD)
        q = apply_rope(q, qcos, qsin); kb = apply_rope(kb, qcos, qsin); kc = apply_rope(kc, kcos, ksin)
        k = torch.cat([kc[None].expand(M_, -1, -1, -1), kb], 1).transpose(1, 2)    # [M, NKV, Lc+B, HD]
        v = torch.cat([vc[None].expand(M_, -1, -1, -1), vb], 1).transpose(1, 2)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k, v, attn_mask=mask, scale=HD ** -0.5, enable_gqa=True)
        o = L.o_proj(o.transpose(1, 2).reshape(M_, B, NH * HD))
        o = dyn_conv(o, kern, L.attention_conv_base[1])
        h = res + o
        res = h
        x = rms(h, L.post_norm)
        x, kern = conv_prepare(L, x, "mlp_conv")
        x = L.down_proj(F.silu(L.gate_proj(x)) * L.up_proj(x))
        x = dyn_conv(x, kern, L.mlp_conv_base[1])
        h = res + x
    return rms(h, m.norm)[:, 1:]


def selector_scores(m, hid, logits_top, cand, pred_ids):
    """hid [M, K, H]; cand [M, K, TOPK]; pred_ids [M, K] -> scores [M, K, TOPK] (unary + bilinear)."""
    hp = hid.float() @ m.sel_hidden.T                                        # [M, K, R]
    return logits_top + torch.einsum("mkr,mktr->mkt", m.sel_pred[pred_ids] * hp, m.sel_succ[cand])


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, dirs):
        self.recs = []
        for d in dirs:
            fc = np.memmap(os.path.join(d, "dump", "fc.bf16"), dtype=np.uint16, mode="r").reshape(-1, 5 * H)
            fp = os.path.join(d, "dump", "final.bf16")
            fin = np.memmap(fp, dtype=np.uint16, mode="r").reshape(-1, H) if os.path.exists(fp) else None
            for line in open(os.path.join(d, "index.jsonl")):
                r = json.loads(line)
                if r["ncols"] - r["resp0"] >= 4:
                    r["fc"] = fc; self.recs.append(r)
                    if fin is not None and "fcol0" in r: r["final"] = fin
        rng = random.Random(a.seed); rng.shuffle(self.recs)
        n_hold = max(1, int(len(self.recs) * a.holdout))
        self.hold, self.train = self.recs[:n_hold], self.recs[n_hold:]

    @staticmethod
    def load(r):
        f = torch.from_numpy(np.ascontiguousarray(r["fc"][r["col0"]: r["col0"] + r["ncols"]])).view(torch.bfloat16)
        fin = None
        if "final" in r:
            fin = torch.from_numpy(np.ascontiguousarray(r["final"][r["fcol0"]: r["fcol0"] + r["ncols"]])).view(torch.bfloat16)
        return f, torch.tensor(r["ids"], dtype=torch.long), fin


def sample_anchors(r, n, rng):
    lo, hi = r["resp0"], r["ncols"] - 2          # anchor token known, at least one target
    pos = list(range(lo, hi + 1))
    return sorted(rng.sample(pos, min(n, len(pos))))


def batch_loss(m, feats, ids, anchors, train=True, fin=None):
    an = torch.tensor(anchors, device=dev)
    Lc = int(an.max().item())
    hid = forward(m, feats[:Lc].to(dev, non_blocking=True), ids, an)        # [M, B-1, H]
    tpos = an[:, None] + torch.arange(1, B, device=dev)[None]               # [M, B-1]
    valid = tpos < ids.shape[0]
    tgt = ids.to(dev)[tpos.clamp(max=ids.shape[0] - 1)]
    logits = (hid @ head.T).float()                                          # [M, B-1, V]
    w = (a.decay ** torch.arange(B - 1, device=dev).float())[None].expand_as(valid) * valid
    lsm = logits.log_softmax(-1)
    ce = -lsm.gather(-1, tgt[..., None])[..., 0]
    if train and a.soft > 0 and fin is not None:     # eval keeps the hard CE, comparable across runs
        with torch.no_grad():                         # target next-token distribution, from position tpos-1
            tl = (fin.to(dev, non_blocking=True)[(tpos - 1).clamp(max=ids.shape[0] - 1)] @ head.T).float() / a.soft_temp
            lz = tl.logsumexp(-1, keepdim=True)
            tv, ti = tl.topk(a.soft_topk, -1); del tl
            tp = (tv - lz).exp()                      # top-K target probabilities (not renormalized)
        ce = (1 - a.soft) * ce - a.soft * (tp * lsm.gather(-1, ti)).sum(-1)
    loss = (ce * w).sum() / w.sum()
    top_val, cand = logits.detach().topk(TOPK, -1)
    pred = torch.cat([ids.to(dev)[an][:, None], tgt[:, :-1]], 1)            # teacher-forced predecessor
    sc = selector_scores(m, hid, top_val, cand, pred)
    hit = (cand == tgt[..., None])
    has = hit.any(-1) & valid
    sel = F.cross_entropy(sc[has], hit[has].float().argmax(-1)) if has.any() else torch.zeros((), device=dev)
    stats = {}
    if not train:
        with torch.no_grad():
            arg = logits.argmax(-1); ok = (arg == tgt) & valid
            stats["pos_ok"] = ok.sum(0).cpu(); stats["pos_n"] = valid.sum(0).cpu()
            stats["chain"] = ok.int().cumprod(1).sum(1).float().sum().item(); stats["anchors"] = an.shape[0]
            # selector greedy path (teacher-free): previous = chosen candidate
            prev = ids.to(dev)[an]; alive = torch.ones_like(prev, dtype=torch.bool); acc = torch.zeros_like(prev)
            hp = hid.float() @ m.sel_hidden.T
            for k in range(B - 1):
                s = top_val[:, k] + torch.einsum("mr,mtr->mt", m.sel_pred[prev] * hp[:, k], m.sel_succ[cand[:, k]])
                prev = cand[:, k].gather(-1, s.argmax(-1, keepdim=True))[:, 0]
                alive = alive & (prev == tgt[:, k]) & valid[:, k]; acc += alive
            stats["chain_sel"] = acc.float().sum().item()
    return loss, sel, stats


def evaluate(m, data, tag):
    rng = random.Random(1)
    tot = {"pos_ok": torch.zeros(B - 1), "pos_n": torch.zeros(B - 1), "chain": 0.0, "chain_sel": 0.0, "anchors": 0}
    lsum = ssum = 0.0; n = 0
    with torch.no_grad():
        for r in data.hold[: a.eval_seqs]:
            feats, ids, fin = Data.load(r)
            anchors = sample_anchors(r, 64, rng)
            loss, sel, st = batch_loss(m, feats, ids, anchors, train=False, fin=fin)
            lsum += loss.item(); ssum += sel.item(); n += 1
            for k in ("pos_ok", "pos_n"): tot[k] += st[k].float()
            for k in ("chain", "chain_sel", "anchors"): tot[k] += st[k]
    acc = (tot["pos_ok"] / tot["pos_n"].clamp(min=1)).tolist()
    res = {"tag": tag, "loss": lsum / n, "sel_loss": ssum / n, "chain_argmax": tot["chain"] / tot["anchors"],
           "chain_selector": tot["chain_sel"] / tot["anchors"], "top1_by_pos": [round(x, 4) for x in acc]}
    print(json.dumps(res), flush=True)
    with open(os.path.join(a.out, "eval.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")
    return res


def save(m, path):
    os.makedirs(path, exist_ok=True)
    out = {k: v.clone() for k, v in P.items()}
    out["fc.weight"] = m.fc.merged().cpu()
    out["hidden_norm.weight"] = m.hidden_norm.detach().to(torch.bfloat16).cpu()
    out["norm.weight"] = m.norm.detach().to(torch.bfloat16).cpu()
    for i, L in enumerate(m.layers):
        p = f"layers.{i}."
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[p + "self_attn." + n + ".weight"] = getattr(L, n).merged().cpu()
        for n in ("gate_proj", "up_proj", "down_proj"):
            out[p + "mlp." + n + ".weight"] = getattr(L, n).merged().cpu()
        out[p + "self_attn.q_norm.weight"] = L.q_norm.detach().to(torch.bfloat16).cpu()
        out[p + "self_attn.k_norm.weight"] = L.k_norm.detach().to(torch.bfloat16).cpu()
        out[p + "input_layernorm.weight"] = L.in_norm.detach().to(torch.bfloat16).cpu()
        out[p + "post_attention_layernorm.weight"] = L.post_norm.detach().to(torch.bfloat16).cpu()
        for c in ("attention_conv", "mlp_conv"):
            out[p + c + ".base_kernel"] = getattr(L, c + "_base").detach().to(torch.bfloat16).cpu()
            out[p + c + ".kernel_projection.weight"] = getattr(L, c + "_proj").merged().cpu()
    out["candidate_selector.hidden_projection.weight"] = m.sel_hidden.detach().to(torch.bfloat16).cpu()
    out["candidate_selector.predecessor_codebook"] = m.sel_pred.detach().to(torch.bfloat16).cpu()
    out["candidate_selector.successor_codebook"] = m.sel_succ.detach().to(torch.bfloat16).cpu()
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"), indent=2)   # unchanged; the engine sets K
    json.dump({"train_block": B, "args": vars(a)}, open(os.path.join(path, "finetune.json"), "w"), indent=2)
    lora = {n: p.detach().cpu() for n, p in m.named_parameters()}
    torch.save(lora, os.path.join(path, "trainable.pt"))


m = Drafter()
del P["fc.weight"]  # merged copies are rebuilt from the module at save time
P["fc.weight"] = m.fc.w.cpu()
if a.load_lora:
    sd = torch.load(a.load_lora); m.load_state_dict(sd, strict=False)
data = Data(a.data)
print(f"sequences: train {len(data.train)}, holdout {len(data.hold)}; block {B}", flush=True)
evaluate(m, data, "base")
if a.eval_only:
    raise SystemExit
lora_p = [p for n, p in m.named_parameters() if n.endswith(".A") or n.endswith(".B")]
full_p = [p for n, p in m.named_parameters() if not (n.endswith(".A") or n.endswith(".B"))]
opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr, "weight_decay": 0.0},
                         {"params": full_p, "lr": a.lr_full, "weight_decay": 0.0}], betas=(0.9, 0.98))
steps = int(len(data.train) * a.epochs)
if a.max_steps: steps = min(steps, a.max_steps)
base_lr = [g["lr"] for g in opt.param_groups]


def prefetch(q, order):
    for r in order:
        q.put((r, *Data.load(r)))
    q.put(None)


order = [data.train[i % len(data.train)] for i in range(steps)]
q = queue.Queue(maxsize=8); threading.Thread(target=prefetch, args=(q, order), daemon=True).start()
rng = random.Random(a.seed + 1); t0 = time.time(); step = 0; run = run_s = 0.0
while True:
    item = q.get()
    if item is None: break
    r, feats, ids, fin = item
    f = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / max(1, steps)))
    for g, lr in zip(opt.param_groups, base_lr): g["lr"] = lr * f
    loss, sel, _ = batch_loss(m, feats, ids, sample_anchors(r, a.anchors, rng), fin=fin)
    (loss + a.selector_weight * sel).backward()
    torch.nn.utils.clip_grad_norm_(lora_p + full_p, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
    run_s = 0.98 * run_s + 0.02 * sel.item() if step else sel.item()
    step += 1
    if step % 50 == 0:
        print(f"step {step}/{steps} loss {run:.4f} sel {run_s:.4f} {time.time() - t0:.0f}s", flush=True)
    if step % a.eval_every == 0:
        evaluate(m, data, f"step{step}")
evaluate(m, data, "final")
save(m, os.path.join(a.out, "drafter"))
print("saved", os.path.join(a.out, "drafter"), flush=True)
