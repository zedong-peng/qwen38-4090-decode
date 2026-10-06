#!/usr/bin/env python3
"""End-to-end scale tuning (EfficientQAT's E2E-QP) of a quantized Qwen3.8-27B target.

The integer codes of a GPTQ store stay fixed. Every group scale of every layer projection (log-parametrized, FP32 master)
is trained by distillation from the FP32 model: the loss is KL(ref || student) on the reference's top-64 support plus the
lumped tail mass (a proper divergence whose top-64 part is the fidelity metric KL64), over calibration windows whose
reference records come from `fq_eval.py --mode ref` (TEACHER = their concatenated .topk.bin).

Layer-major steps. A step takes --wpb windows: (1) forward without grad, layer by layer, boundary activations to pinned
host memory; (2) final norm + Q8 head + loss, the gradient w.r.t. the last hidden state; (3) layers in reverse: recompute
each layer's forward from its stored input with grad, backpropagate, and apply Adam to that layer's scales at once.
Each layer's codes (int4-packed on the host when they fit in [-8, 7], int8 otherwise) visit the GPU twice per step, so the
model never sits on the GPU whole. Matmuls run in TF32, everything else in FP32 (the --mode codes simulation is FP32).
The embedding and head are Q8 (the converter's absmax, as in the simulation) and frozen, like every non-projection weight.

--param row trains one log-scale offset per output row instead (all groups of a row move together: ~100x fewer
parameters, so a small batch's gradient is far less noisy); --norms also trains the RMSNorm gains of every layer
(exported as a patched model dir, --model-out, whose index points those tensors at a BF16 patch file). --bf16 runs the
projection matmuls in BF16 (FP32 accumulate) instead of TF32. --train-ranges restricts training to windows the GPTQ never
saw (on its own calibration windows the quantized model is far closer to the reference: 0.011 vs 0.052 held out).

Every --eval-every steps the held-out windows (every --holdout-every-th calibration window) are scored; the best held-out
scales are kept. The output is a codes store (same codes, tuned FP16 scales), readable by fq_eval --mode codes and by the
converter (tools/gptq_recipe.py: --source gptq/q3/q5 may all point at it).
usage: e2e_scales.py MODEL_DIR CALIB.windows.bin TEACHER.topk.bin OUT_STORE --codes A [--codes-b B --groups G]
                     [--codes-c C --groups-c G] [--epochs 1] [--wpb 4] [--lr 5e-4] [--param group|row]
                     [--norms --lr-norm 5e-4 --model-out DIR] [--bf16] [--train-ranges 96-383,480-767]"""
import argparse, json, math, os, shutil, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fq_eval as fq  # noqa: E402  (read_windows, Weights, group_rows, rtn, FMT, QUANT_SUFFIXES, TOP)
import transformers.models.qwen3_5.modeling_qwen3_5 as mq  # noqa: E402
from fast_gdr import fast_chunk_gated_delta_rule  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5DecoderLayer, Qwen3_5RMSNorm,  # noqa: E402
                                                          Qwen3_5TextRotaryEmbedding)

TOP = fq.TOP


MM_DTYPE = [torch.float32]  # torch.bfloat16 with --bf16


class QLinFn(torch.autograd.Function):
    """y = x W^T with W = codes * exp(theta) per 64-group; saves only x, the int8 codes and theta."""

    @staticmethod
    def forward(ctx, x, codes, theta):
        o, i = codes.shape
        dt = MM_DTYPE[0]
        w = (codes.view(o, -1, 64).float() * theta.exp()[..., None]).view(o, i).to(dt)
        ctx.save_for_backward(x.to(dt), codes, theta)
        return (x.to(dt) @ w.T).float()

    @staticmethod
    def backward(ctx, gy):
        x, codes, theta = ctx.saved_tensors
        o, i = codes.shape
        dt = MM_DTYPE[0]
        s = theta.exp()
        c = codes.view(o, -1, 64).float()
        g = gy.to(dt)
        gx = (g @ (c * s[..., None]).view(o, i).to(dt)).float()
        gw = (g.reshape(-1, o).T @ x.reshape(-1, i)).float()
        gt = (gw.view(o, -1, 64) * c).sum(-1) * s
        return gx, None, gt


class QLinear(torch.nn.Module):
    def __init__(self, stem):
        super().__init__()
        self.stem, self.codes, self.theta = stem, None, None  # theta: a leaf tensor owned by the trainer

    def forward(self, x):
        return QLinFn.apply(x, self.codes, self.theta)


def pack(codes):
    """int8 [o, i] -> (uint8 [o, i/2] nibble pairs, True) when every code lies in [-8, 7], else (codes, False)."""
    if int(codes.min()) >= -8 and int(codes.max()) <= 7:
        u = (codes.to(torch.int16) & 15).to(torch.uint8)
        return (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous(), True
    return codes.contiguous(), False


def unpack(p, packed, dev):
    p = p.to(dev, non_blocking=True)
    if not packed:
        return p
    lo, hi = (p & 15).to(torch.int8), (p >> 4).to(torch.int8)
    lo, hi = (lo ^ 8) - 8, (hi ^ 8) - 8
    return torch.stack((lo, hi), -1).view(p.shape[0], -1)


TOP1 = [0.0, 0.5, 0.25]  # --top1 weight, --top1-gap (teacher top-2 log-prob gap), --top1-margin (logits)


def kd_loss(logits, t):
    """logits [n, V] fp32; t [n, 1 + 2 TOP] reference records (lse, ids, logits). Per-row KL with the tail lumped,
    plus the optional top-1 hinge (TOP1); also returns the per-row top-1 agreement with the teacher."""
    lse_t, ids, vt = t[:, 0], t[:, 1:1 + TOP].contiguous().view(torch.int32).long(), t[:, 1 + TOP:]
    lp_t = vt - lse_t[:, None]
    p_t = lp_t.exp()
    lp_s = logits.gather(1, ids) - torch.logsumexp(logits, -1, keepdim=True)
    tail_t = (1 - p_t.sum(-1)).clamp_min(1e-9)
    tail_s = (1 - lp_s.exp().sum(-1)).clamp_min(1e-9)
    kl64 = (p_t * (lp_t - lp_s)).sum(-1)
    loss = kl64 + tail_t * (tail_t.log() - tail_s.log())
    a1 = ids.gather(1, vt.argmax(-1, keepdim=True))
    v2, i2 = logits.topk(2, -1)
    hit = (i2[:, :1] == a1).squeeze(-1).float()
    if TOP1[0] > 0:
        za = logits.gather(1, a1).squeeze(-1)
        zo = torch.where(i2[:, 0] == a1[:, 0], v2[:, 1], v2[:, 0])
        g2 = lp_t.topk(2, -1).values
        on = ((g2[:, 0] - g2[:, 1]) >= TOP1[1]).float()
        loss = loss + TOP1[0] * on * torch.relu(zo - za + TOP1[2])
    return loss, kl64, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("calib"); ap.add_argument("teacher"); ap.add_argument("out")
    ap.add_argument("--codes", required=True)
    ap.add_argument("--codes-b"); ap.add_argument("--groups", default="")
    ap.add_argument("--codes-c"); ap.add_argument("--groups-c", default="")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--wpb", type=int, default=4, help="windows per step")
    ap.add_argument("--mb", type=int, default=2, help="windows per microbatch inside a layer")
    ap.add_argument("--lr", type=float, default=5e-4, help="Adam lr on log scales")
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--holdout-every", type=int, default=48)
    ap.add_argument("--holdout-ranges", default="", help="hold out every --holdout-every-th window of these window ranges "
                    "(e.g. 96-383,480-767: windows the GPTQ calibration never saw); default all windows")
    ap.add_argument("--eval-every", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0, help="smoke tests: only the first N layers")
    ap.add_argument("--vocab-rows", type=int, default=248077)
    ap.add_argument("--chunk", type=int, default=256, help="positions per head/loss chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--slow-gdr", action="store_true", help="keep transformers' loop form of the GDN chunk rule")
    ap.add_argument("--param", choices=("group", "row"), default="group")
    ap.add_argument("--kv", default="bf16", choices=["bf16", "k8v4", "k8v8", "k8vi8", "fp8", "int8"],
                    help="train and evaluate through the engine KV cache codec (fq_eval.enable_kv_codec)")
    ap.add_argument("--norms", action="store_true", help="also train every layer's RMSNorm gains")
    ap.add_argument("--lr-norm", type=float, default=5e-4)
    ap.add_argument("--model-out", default="", help="with --norms: patched model dir (symlinked shards + norm patch)")
    ap.add_argument("--bf16", action="store_true", help="BF16 projection matmuls (FP32 accumulate)")
    ap.add_argument("--window-weights", default="", help="LO-HI:W,... loss weight per window range (default 1)")
    ap.add_argument("--train-ranges", default="", help="train only on these window ranges (minus the held-out ones)")
    ap.add_argument("--top1", type=float, default=0.0, help="weight of the top-1 hinge (0 = off)")
    ap.add_argument("--top1-gap", type=float, default=0.5, help="hinge only where the teacher's top-2 gap >= this")
    ap.add_argument("--top1-margin", type=float, default=0.25, help="hinge margin in logits")
    a = ap.parse_args()
    TOP1[:] = [a.top1, a.top1_gap, a.top1_margin]
    if a.bf16:
        MM_DTYPE[0] = torch.bfloat16
    if not a.slow_gdr:  # same math as the loop form (tools/fast_gdr.py), one triangular solve per chunk
        mq.torch_chunk_gated_delta_rule = fast_chunk_gated_delta_rule
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cuda = dev.type == "cuda"
    pin = (lambda t: t.pin_memory()) if cuda else (lambda t: t)
    gib = (lambda f: f() / 2**30) if cuda else (lambda f: 0.0)
    t0 = time.time()
    cfg = Qwen3_5TextConfig(**json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"])
    cfg._attn_implementation = "sdpa"
    fq.enable_kv_codec(a.kv)
    NL = a.layers or cfg.num_hidden_layers
    W = fq.Weights(a.model_dir)
    wins = fq.read_windows(a.calib)
    L = len(wins[0][0]); assert all(len(t) == L for t, _ in wins)
    ncol = [L - f for _, f in wins]; start = np.concatenate([[0], np.cumsum(ncol)])
    teacher = np.memmap(a.teacher, dtype=np.float32, mode="r").reshape(-1, 1 + 2 * TOP)
    assert teacher.shape[0] == start[-1], (teacher.shape, start[-1])
    def wranges(spec):
        out = []
        for part in (spec.split(",") if spec else [f"0-{len(wins) - 1}"]):
            lo, _, hi = part.partition("-"); out.append(list(range(int(lo), int(hi or lo) + 1)))
        return out
    cand = wranges(a.holdout_ranges)
    hold = sorted(w for c in cand for w in c[a.holdout_every - 1::a.holdout_every])
    hs = set(hold); train = [w for c in wranges(a.train_ranges) for w in c if w not in hs]
    wts = np.ones(len(wins)); dom = np.full(len(wins), -1)
    for d, part in enumerate(p for p in a.window_weights.split(",") if p):
        r, _, wv = part.rpartition(":"); lo, _, hi = r.partition("-")
        wts[int(lo):int(hi or lo) + 1] = float(wv); dom[int(lo):int(hi or lo) + 1] = d
    dacc = {}
    print(f"{len(wins)} windows: {len(train)} train, {len(hold)} held out; {NL} layers", flush=True)

    # stores and row groups, as fq_eval --mode codes
    stores = {"a": fq.Weights(a.codes)}
    if a.codes_b: stores["b"] = fq.Weights(a.codes_b)
    if a.codes_c: stores["c"] = fq.Weights(a.codes_c)

    def ranges(spec):
        out = {}
        for g in (g for g in spec.split(",") if g):
            name, _, sp = g.partition(":"); lay = set()
            for part in sp.split("+"):
                lo, _, hi = part.partition("-"); lay.update(range(int(lo), int(hi or lo) + 1))
            out[name] = lay
        return out
    gb, gc = ranges(a.groups), ranges(a.groups_c)
    assert all(":" in g for g in a.groups.split(",") if g), "--groups needs per-group layer ranges"

    # layers: meta construction, projections replaced by QLinear, the rest loaded and frozen
    layers, qlins, codes_host, theta, s0, norm_params = [], [], {}, {}, {}, {}
    for i in range(NL):
        with torch.device("meta"):
            layer = Qwen3_5DecoderLayer(cfg, i)
        prefix = f"model.language_model.layers.{i}."
        names = [n for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear) and (n + ".weight").endswith(fq.QUANT_SUFFIXES)]
        ql = {}
        for n in names:
            parent, _, attr = n.rpartition(".")
            q = QLinear(prefix + n); setattr(layer.get_submodule(parent), attr, q); ql[n] = q
        layer = layer.to_empty(device=dev)
        sd = {k[len(prefix):]: W.get(k).float() for k in W.map if k.startswith(prefix)
              and k[len(prefix):].removesuffix(".weight") not in ql}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        assert not unexpected and not missing, (missing, unexpected)
        for p in layer.parameters():
            p.requires_grad_(False)
        layer.eval()
        if a.norms:
            for n, mod in layer.named_modules():
                if isinstance(mod, (Qwen3_5RMSNorm, mq.Qwen3_5RMSNormGated)):
                    norm_params[prefix + n + ".weight"] = (i, mod.weight)
        for n, q in ql.items():
            stem = q.stem
            codes = stores["a"].get(stem + ".gptq_codes"); scales = stores["a"].get(stem + ".gptq_scales").float()
            for key, groups in (("b", gb), ("c", gc)):
                act = {g for g, lay in groups.items() if i in lay}
                if act:
                    m = fq.group_rows(stem, codes.shape[0], act, cfg)
                    if bool(m.any()):
                        codes[m] = stores[key].get(stem + ".gptq_codes")[m]
                        scales[m] = stores[key].get(stem + ".gptq_scales").float()[m]
            assert bool((codes >= -16).all() and (codes <= 15).all()), stem
            pk, packed = pack(codes)
            codes_host[stem] = (pin(pk), packed)
            s0[stem] = scales.half()
            theta[stem] = torch.log(scales.clamp_min(1e-30)).to(dev).requires_grad_(False)
            q.theta = theta[stem]
        layers.append(layer); qlins.append(ql)
        if i % 8 == 0:
            print(f"layer {i} set up {time.time() - t0:.0f}s", flush=True)
    nparam = sum(t.numel() for t in theta.values())
    print(f"{len(theta)} projections, {nparam / 1e6:.1f} M scales; host codes "
          f"{sum(c.numel() for c, _ in codes_host.values()) / 2**30:.1f} GiB", flush=True)
    pshape = (lambda t: t[:, :1]) if a.param == "row" else (lambda t: t)
    adam_m = {k: torch.zeros_like(pshape(t)) for k, t in theta.items()}
    adam_v = {k: torch.zeros_like(pshape(t)) for k, t in theta.items()}
    live = {k: (s0[k].to(dev) > 0).float() for k in theta}  # dead groups (scale 0) stay dead
    norm0 = {k: p.detach().clone() for k, (_, p) in norm_params.items()}
    nm_m = {k: torch.zeros_like(p) for k, (_, p) in norm_params.items()}
    nm_v = {k: torch.zeros_like(p) for k, (_, p) in norm_params.items()}
    layer_norms = [[(k, p) for k, (j, p) in norm_params.items() if j == i] for i in range(NL)]
    print(f"param {a.param}: {sum(m.numel() for m in adam_m.values()) / 1e6:.2f} M trainable scales; "
          f"{sum(p.numel() for p in norm0.values()) / 1e6:.2f} M norm gains ({len(norm0)} norms)", flush=True)

    emb = W.get("model.language_model.embed_tokens.weight")
    lo, hi, g = fq.FMT["q8"]; emb = fq.rtn(emb.to(dev), lo, hi, g, alphas=(1.0,)).float().cpu()
    norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = W.get("model.language_model.norm.weight").float(); norm = norm.to(dev).requires_grad_(False)
    head = W.get("lm_head.weight")[: a.vocab_rows].to(dev)
    head = fq.rtn(head, lo, hi, g, alphas=(1.0,)).float()
    rot = Qwen3_5TextRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(L, device=dev)[None].expand(a.mb, L)
    pe_full = rot(torch.zeros(a.mb, L, cfg.hidden_size, device=dev), pos)
    H = cfg.hidden_size
    acts = pin(torch.empty((NL + 1, a.wpb, L, H), dtype=torch.float32))
    print(f"setup done {time.time() - t0:.0f}s; GPU {gib(torch.cuda.memory_allocated):.1f} GiB, "
          f"host acts {acts.numel() * 4 / 2**30:.1f} GiB", flush=True)

    def load(i):
        for q in qlins[i].values():
            pk, packed = codes_host[q.stem]
            q.codes = unpack(pk, packed, dev)

    def unload(i):
        for q in qlins[i].values():
            q.codes = None

    def run_layer(i, x):
        m = x.shape[0]
        return layers[i](x, position_embeddings=(pe_full[0][:m], pe_full[1][:m]), attention_mask=None, position_ids=pos[:m])

    def forward(ws):
        nw = len(ws)
        for j, w in enumerate(ws):
            acts[0, j].copy_(emb[torch.from_numpy(wins[w][0]).long()])
        with torch.no_grad():
            for i in range(NL):
                load(i)
                for b in range(0, nw, a.mb):
                    y = run_layer(i, acts[i, b:min(nw, b + a.mb)].to(dev, non_blocking=True))
                    acts[i + 1, b:b + y.shape[0]].copy_(y)
                unload(i)

    def loss_and_grad(ws, want_grad):
        nw = len(ws); tot = sum(wts[w] * ncol[w] for w in ws); tot_u = sum(ncol[w] for w in ws)
        grad = torch.zeros((nw, L, H), device=dev) if want_grad else None
        s_loss = s_kl64 = 0.0
        for j, w in enumerate(ws):
            first = wins[w][1]
            h = acts[NL, j].to(dev)
            tw = torch.from_numpy(np.array(teacher[start[w]:start[w + 1]])).to(dev)
            for c0 in range(first - 1, L - 1, a.chunk):
                c1 = min(L - 1, c0 + a.chunk)
                hc = h[c0:c1].clone().requires_grad_(want_grad)
                with torch.set_grad_enabled(want_grad):
                    logits = norm(hc) @ head.T
                    lrow, k64, hit = kd_loss(logits, tw[c0 - first + 1:c1 - first + 1])
                    lsum = lrow.sum() * float(wts[w])
                if want_grad:
                    (lsum / tot).backward()
                    grad[j, c0:c1] = hc.grad
                s_loss += float(lsum.detach()); s_kl64 += float(k64.detach().sum())
                acc = dacc.setdefault(int(dom[w]), [0.0, 0, 0.0]); acc[0] += float(k64.detach().sum()); acc[1] += c1 - c0
                acc[2] += float(hit.sum())
                del logits, lrow, k64, hc, hit
        return s_loss / tot, s_kl64 / tot_u, grad

    step_no = [0]

    def backward(ws, grad, lr):
        nw = len(ws)
        step_no[0] += 1; t = step_no[0]
        b1, b2, eps = 0.9, 0.999, 1e-20
        gn = 0.0
        for i in reversed(range(NL)):
            load(i)
            for q in qlins[i].values():
                q.theta.requires_grad_(True); q.theta.grad = None
            for _, p in layer_norms[i]:
                p.requires_grad_(True); p.grad = None
            for b in range(0, nw, a.mb):
                x = acts[i, b:min(nw, b + a.mb)].to(dev).requires_grad_(True)
                y = run_layer(i, x)
                y.backward(grad[b:b + x.shape[0]])
                grad[b:b + x.shape[0]] = x.grad
                del x, y
            unload(i)
            with torch.no_grad():
                for q in qlins[i].values():
                    k, th, gr = q.stem, q.theta, q.theta.grad * live[q.stem]
                    if a.param == "row":
                        gr = gr.sum(1, keepdim=True)
                    gn += float((gr * gr).sum())
                    adam_m[k].mul_(b1).add_(gr, alpha=1 - b1)
                    adam_v[k].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    upd = (adam_m[k] / (1 - b1 ** t)) / ((adam_v[k] / (1 - b2 ** t)).sqrt() + eps)
                    th.sub_(lr * upd * live[k])
                    th.grad = None; th.requires_grad_(False)
                lrn = lr * a.lr_norm / a.lr
                for k, p in layer_norms[i]:
                    gr = p.grad
                    gn += float((gr * gr).sum())
                    nm_m[k].mul_(b1).add_(gr, alpha=1 - b1)
                    nm_v[k].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                    p.sub_(lrn * (nm_m[k] / (1 - b1 ** t)) / ((nm_v[k] / (1 - b2 ** t)).sqrt() + eps))
                    p.grad = None; p.requires_grad_(False)
        return math.sqrt(gn)

    def evaluate():
        tl = tk = n = nw_ = 0.0
        dacc.clear()
        for b in range(0, len(hold), a.wpb):
            ws = hold[b:b + a.wpb]
            forward(ws)
            l, k, _ = loss_and_grad(ws, False)
            c = sum(ncol[w] for w in ws); cw = sum(wts[w] * ncol[w] for w in ws)
            tl += l * cw; tk += k * c; n += c; nw_ += cw
        if a.window_weights:
            print("  held-out kl64 by range: " + " ".join(f"{d}:{v[0] / max(1, v[1]):.5f}" for d, v in sorted(dacc.items())),
                  flush=True)
        n1 = sum(v[1] for v in dacc.values())
        print(f"  held-out top1 {sum(v[2] for v in dacc.values()) / max(1, n1):.4f} by range: "
              + " ".join(f"{d}:{v[2] / max(1, v[1]):.4f}" for d, v in sorted(dacc.items())), flush=True)
        dacc.clear()
        return tl / nw_, tk / n

    def drift():
        num = den = 0.0
        for k, th in theta.items():
            d = (th - torch.log(s0[k].to(dev).float().clamp_min(1e-30))) * live[k]
            num += float((d * d).sum()); den += float(live[k].sum())
        return math.sqrt(num / max(den, 1))

    def ndrift():
        num = sum(float(((p - norm0[k]) ** 2).sum()) for k, (_, p) in norm_params.items())
        return math.sqrt(num / max(1, sum(p.numel() for p in norm0.values())))

    def snapshot():
        return ({k: t.detach().cpu().clone() for k, t in theta.items()},
                {k: p.detach().cpu().clone() for k, (_, p) in norm_params.items()})

    rng = np.random.default_rng(a.seed)
    steps_per_epoch = len(train) // a.wpb
    total = a.max_steps or int(round(a.epochs * steps_per_epoch))
    hl, hk = evaluate()
    best = (hl, 0); best_theta, best_norms = snapshot()
    print(f"step 0 held-out loss {hl:.5f} kl64 {hk:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    order = []
    for st in range(1, total + 1):
        if not order:
            order = list(rng.permutation(train))
        ws = [order.pop() for _ in range(min(a.wpb, len(order)))]
        lr = a.lr * min(1.0, st / a.warmup) * 0.5 * (1 + math.cos(math.pi * (st - 1) / total))
        ts = time.time()
        forward(ws)
        l, k, grad = loss_and_grad(ws, True)
        gn = backward(ws, grad, lr)
        del grad
        print(f"step {st}/{total} loss {l:.5f} kl64 {k:.5f} |g| {gn:.3e} lr {lr:.2e} drift {drift():.5f} "
              + (f"ndrift {ndrift():.5f} " if norm_params else "") +
              f"{time.time() - ts:.0f}s peak {gib(torch.cuda.max_memory_allocated):.1f} GiB", flush=True)
        if st % a.eval_every == 0 or st == total:
            hl, hk = evaluate()
            mark = ""
            if hl < best[0]:
                best = (hl, st); best_theta, best_norms = snapshot(); mark = " *"
            print(f"step {st} held-out loss {hl:.5f} kl64 {hk:.5f}{mark}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"best held-out loss {best[0]:.5f} at step {best[1]}", flush=True)

    # export: same codes, tuned FP16 scales (dead groups stay 0)
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(a.model_dir, "config.json"), a.out)
    weight_map = {}
    for i in range(NL):
        saved = {}
        for q in qlins[i].values():
            pk, packed = codes_host[q.stem]
            c = unpack(pk, packed, "cpu") if packed else pk.clone()
            s = torch.where(s0[q.stem] > 0, best_theta[q.stem].exp(), torch.zeros_like(best_theta[q.stem])).half()
            saved[q.stem + ".gptq_codes"] = c.contiguous(); saved[q.stem + ".gptq_scales"] = s.contiguous()
        fname = f"codes-{i:03d}.safetensors"
        save_file(saved, os.path.join(a.out, fname)); weight_map.update({t: fname for t in saved})
    json.dump({"metadata": {"mode": "e2e-scales", "codes": a.codes, "codes_b": a.codes_b, "groups": a.groups,
                            "codes_c": a.codes_c, "groups_c": a.groups_c, "calib": a.calib, "teacher": a.teacher,
                            "lr": a.lr, "wpb": a.wpb, "steps": total, "best_step": best[1], "best_heldout": best[0],
                            "param": a.param, "norms": a.norms, "lr_norm": a.lr_norm, "bf16": a.bf16,
                            "train_ranges": a.train_ranges, "window_weights": a.window_weights, "holdout_ranges": a.holdout_ranges,
                            "top1": TOP1, "model_dir": a.model_dir, "kv": a.kv},
               "weight_map": weight_map}, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    if norm_params:  # BF16 patch file + a model dir whose index points the norm tensors at it
        patch = {k: v.to(torch.bfloat16).contiguous() for k, v in best_norms.items()}
        save_file(patch, os.path.join(a.out, "norms.safetensors"))
        if a.model_out:
            os.makedirs(a.model_out, exist_ok=True)
            src = os.path.abspath(a.model_dir)
            for f in os.listdir(src):
                if f not in ("model.safetensors.index.json", "e2e-norms.safetensors") and \
                        not os.path.exists(os.path.join(a.model_out, f)):
                    os.symlink(os.path.join(src, f), os.path.join(a.model_out, f))
            idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
            shutil.copy(os.path.join(a.out, "norms.safetensors"), os.path.join(a.model_out, "e2e-norms.safetensors"))
            for k in patch:
                assert k in idx["weight_map"], k
                idx["weight_map"][k] = "e2e-norms.safetensors"
            json.dump(idx, open(os.path.join(a.model_out, "model.safetensors.index.json"), "w"), indent=1)
            print(f"patched model dir {a.model_out}: {len(patch)} norm tensors", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
