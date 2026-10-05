#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ref_bf16.py for windows of different lengths (decode-path fidelity, tools/dfid.py).
Same reference: BF16 checkpoint layer-streamed through the transformers qwen3_5 modules (GDN on the FP32 torch
fallback), FP32 logits from the BF16 normed hidden and the BF16 head. Every window runs alone (batch 1); the hidden
states of all windows stay on the GPU between layers.
Writes OUT.topk.bin (per scored column: f32 logsumexp, i32 ids[64], f32 logits[64], descending) and OUT.target.bin
(f32 target log-probability per column), windows in file order, columns first-1 .. n-2 of each.
usage: ref_var.py MODEL_DIR WINDOWS.bin OUT_PREFIX [--device cuda:0]"""
import argparse, json, os, struct, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
                                                          Qwen3_5TextRotaryEmbedding)

TOP = 64


def read_windows(path):
    data = open(path, "rb").read()
    out, off = [], 0
    while off < len(data):
        n, first = struct.unpack_from("<II", data, off); off += 8
        toks = np.frombuffer(data, dtype=np.int32, count=n, offset=off).copy(); off += 4 * n
        out.append((toks, first))
    return out


class Weights:
    def __init__(self, model_dir):
        self.dir = model_dir
        self.map = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}

    def get(self, name, device):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(os.path.join(self.dir, f), framework="pt", device="cpu")
        return self.handles[f].get_tensor(name).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("windows"); ap.add_argument("out")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cfg_all = json.load(open(os.path.join(a.model_dir, "config.json")))
    cfg = Qwen3_5TextConfig(**cfg_all["text_config"])
    cfg._attn_implementation = "sdpa"
    W = Weights(a.model_dir)
    windows = read_windows(a.windows)
    lens = [len(t) for t, _ in windows]
    offs = np.concatenate([[0], np.cumsum(lens)])
    L = max(lens)
    print(f"{len(windows)} windows, {offs[-1]} tokens, longest {L}, "
          f"{sum(n - f for n, (_, f) in zip(lens, windows))} scored columns", flush=True)
    ids = torch.tensor(np.concatenate([t for t, _ in windows]), dtype=torch.long, device=dev)
    t0 = time.time()
    emb = W.get("model.language_model.embed_tokens.weight", dev)
    hidden = emb[ids]  # [tokens, H] bf16
    del emb
    rot = Qwen3_5TextRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(L, device=dev)[None]
    pe_full = rot(hidden[None, :1], pos)
    for i in range(cfg.num_hidden_layers):
        layer = Qwen3_5DecoderLayer(cfg, i)
        prefix = f"model.language_model.layers.{i}."
        sd = {k[len(prefix):]: W.get(k, "cpu") for k in W.map if k.startswith(prefix)}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        assert not unexpected and not [m for m in missing if not m.endswith("inv_freq")], (missing, unexpected)
        layer = layer.to(dev, dtype=torch.bfloat16)
        if hasattr(layer, "linear_attn"):  # keep the FP32 parameters the checkpoint stores in FP32
            for name in ("A_log", "dt_bias"):
                p = getattr(layer.linear_attn, name)
                p.data = sd[f"linear_attn.{name}"].to(dev)
        with torch.no_grad():
            for w, n in enumerate(lens):
                h = hidden[offs[w]: offs[w] + n][None]
                pe = (pe_full[0][:, :n], pe_full[1][:, :n])
                hidden[offs[w]: offs[w] + n] = layer(h, position_embeddings=pe, attention_mask=None,
                                                     position_ids=pos[:, :n])[0]
        del layer, sd
        torch.cuda.empty_cache()
        print(f"layer {i} done {time.time() - t0:.0f}s", flush=True)
    norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight.data = W.get("model.language_model.norm.weight", "cpu").float()
    norm = norm.to(dev)
    head32 = W.get("lm_head.weight", dev).float()
    with open(a.out + ".topk.bin", "wb") as f, open(a.out + ".target.bin", "wb") as ft, torch.no_grad():
        for w, (toks, first) in enumerate(windows):
            n = lens[w]
            h = norm(hidden[offs[w]: offs[w] + n][None])[0]  # bf16 (norm in fp32, cast back like the model)
            h = h[first - 1: n - 1]
            targets = torch.tensor(toks[first:n], dtype=torch.long, device=dev)
            for c0 in range(0, h.shape[0], 512):
                logits = torch.matmul(h[c0: c0 + 512].float(), head32.T)
                lse = torch.logsumexp(logits, dim=-1)
                tgt = targets[c0: c0 + 512]
                ft.write((logits.gather(1, tgt[:, None])[:, 0] - lse).cpu().numpy().astype(np.float32).tobytes())
                val, idx = torch.topk(logits, TOP, dim=-1)
                rec = np.zeros((logits.shape[0], 1 + 2 * TOP), dtype=np.float32)
                rec[:, 0] = lse.cpu().numpy()
                rec[:, 1:1 + TOP] = idx.int().cpu().numpy().view(np.float32)
                rec[:, 1 + TOP:] = val.cpu().numpy()
                f.write(rec.tobytes())
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
