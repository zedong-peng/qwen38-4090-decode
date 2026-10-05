#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Decode-path fidelity from NINFER_HIDDEN_DUMP runs (tools/patch_hidden_dump.py) on forced Spec-Bench text.

windows DUMP IDS.jsonl OUT [--every K]
    Scoring windows prompt_ids + ids (ids = the run's NINFER_TOKEN_DUMP ids, ids[0] = prefill token) with first
    target P + 1, for every K-th request: OUT.windows.bin (ref_var.py input) and OUT.windows.json (request indices).
score OUT REF_PREFIX MODEL_DIR NAME=DUMP:IDS.jsonl ... [--rows SPECBENCH.json] [--device cuda:0]
    For each run: logits of its accepted-path head inputs with the checkpoint's BF16 head (FP32 products, as
    ref_var.py), then against REF_PREFIX (ref_var.py on OUT.windows.bin):
    KL64(ref||run) on the reference's top-64 support, top-1 agreement, run perplexity of the text; per Spec-Bench
    group (--rows) and overall. Runs after the first also get top-1 flips against the first.
    A request is skipped (and counted) when its prompt or ids differ from the windows' text."""
import json, os, struct, sys
import numpy as np

TOP = 64


def read_dump(path):
    """[(prompt_ids, [(frontier, hidden[count, H] bf16-as-u16), ...]), ...] in request order."""
    d = np.memmap(path, dtype=np.uint8, mode="r")
    out, off, H = [], 0, None
    while off < len(d):
        tag = int(np.frombuffer(d, dtype=np.int32, count=1, offset=off)[0])
        if tag == 1:
            P = int(np.frombuffer(d, dtype=np.int32, count=1, offset=off + 4)[0])
            ids = np.frombuffer(d, dtype=np.int32, count=P, offset=off + 8).copy()
            out.append((ids, []))
            off += 8 + 4 * P
        elif tag == 2:
            frontier, count = (int(x) for x in np.frombuffer(d, dtype=np.int32, count=2, offset=off + 4))
            off += 12
            if H is None:  # hidden size from the distance to the next tag
                for h in (5120, 4096, 6144, 8192, 3072, 2048):
                    nxt = off + 2 * count * h
                    if nxt == len(d) or (nxt + 4 <= len(d) and
                                         int(np.frombuffer(d, dtype=np.int32, count=1, offset=nxt)[0]) in (1, 2)):
                        H = h
                        break
                assert H is not None, "cannot infer the hidden size"
            hid = np.frombuffer(d, dtype=np.uint16, count=count * H, offset=off).reshape(count, H)
            off += 2 * count * H
            assert out, "round before any prompt"
            out[-1][1].append((frontier, hid))
        else:
            raise ValueError(f"bad tag {tag} at {off}")
    return out, H


def read_ids(path):
    return [json.loads(l) for l in open(path)]


def align(reqs, ents):
    """Dump requests matched in order to the token-dump entries by prompt length (extra dump prompts, e.g. a
    server warmup, are dropped); None for an entry without one."""
    out, r = [], 0
    for e in ents:
        while r < len(reqs) and len(reqs[r][0]) != e["prompt_tokens"]:
            r += 1
        out.append(reqs[r] if r < len(reqs) else None)
        r += 1
    return out


def write_windows(dump, ids_path, out, every):
    reqs, _ = read_dump(dump)
    ents = read_ids(ids_path)
    reqs = align(reqs, ents)
    sel = []
    with open(out + ".windows.bin", "wb") as f:
        for r, (req, e) in enumerate(zip(reqs, ents)):
            if r % every or len(e["ids"]) < 2 or req is None:
                continue
            prompt = req[0]
            toks = np.concatenate([prompt, np.array(e["ids"], dtype=np.int32)]).astype(np.int32)
            f.write(struct.pack("<II", len(toks), len(prompt) + 1))
            f.write(toks.tobytes())
            sel.append(r)
    json.dump({"requests": sel, "dump": dump, "ids": ids_path}, open(out + ".windows.json", "w"))
    print(f"{len(sel)} windows from {len(reqs)} requests")


def run_columns(reqs, ents, windows, sel):
    """Per selected request: hidden rows [n-1, H] predicting ids[1..n-1], or None when the text differs."""
    out = []
    for (toks, first), r in zip(windows, sel):
        if r >= len(reqs) or reqs[r] is None:
            out.append(None)
            continue
        prompt, rounds = reqs[r]
        P = first - 1
        ids = toks[P:]
        if len(prompt) != P or not np.array_equal(prompt, toks[:P]) or \
                not np.array_equal(np.array(ents[r]["ids"][:len(ids)], dtype=np.int32), ids):
            out.append(None)
            continue
        cols = [None] * (len(ids) - 1)
        for frontier, hid in rounds:
            j0 = frontier - P + 1
            for c in range(hid.shape[0]):
                j = j0 + c
                if 1 <= j < len(ids):
                    cols[j - 1] = hid[c]
        out.append(None if any(c is None for c in cols) else np.stack(cols))
    return out


def score(argv):
    import torch
    from safetensors import safe_open
    out, ref, model_dir = argv[:3]
    runs = [a for a in argv[3:] if "=" in a and not a.startswith("--")]
    opts = dict(zip(argv[3:][::1], argv[4:] + [None]))
    rows_path = opts.get("--rows")
    dev = torch.device(opts.get("--device") or "cuda:0")
    meta = json.load(open(out + ".windows.json"))
    sel = meta["requests"]
    data = open(out + ".windows.bin", "rb").read(); off = 0; windows = []
    while off < len(data):
        n, first = struct.unpack_from("<II", data, off); off += 8
        windows.append((np.frombuffer(data, dtype=np.int32, count=n, offset=off).copy(), first)); off += 4 * n
    ncols = [len(t) - f for t, f in windows]
    R = np.fromfile(ref + ".topk.bin", dtype=np.float32).reshape(-1, 1 + 2 * TOP)
    T = np.fromfile(ref + ".target.bin", dtype=np.float32)
    assert R.shape[0] == sum(ncols) == T.shape[0], (R.shape, sum(ncols), T.shape)
    starts = np.concatenate([[0], np.cumsum(ncols)])
    ref_lse, ref_ids, ref_logit = R[:, 0], R[:, 1:1 + TOP].view(np.int32), R[:, 1 + TOP:]
    logp = ref_logit - ref_lse[:, None]; p = np.exp(logp)

    # Spec-Bench group per request: rows (480) matched in order by prompt length to the 482 dump entries.
    group = {}
    ents0 = read_ids(runs[0].split("=", 1)[1].split(":", 1)[1])
    if rows_path:
        rows = json.load(open(rows_path))["rows"]; e = 0
        for row in rows:
            while e < len(ents0) and ents0[e]["prompt_tokens"] != row["prompt_tokens"]:
                e += 1
            if e < len(ents0):
                group[e] = row["group"]; e += 1

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(model_dir, idx["lm_head.weight"]), framework="pt", device="cpu") as f:
        head32 = f.get_tensor("lm_head.weight").to(dev).float()

    base_top1 = None
    for spec in runs:
        name, rest = spec.split("=", 1)
        dump, ids_path = rest.split(":", 1)
        reqs, H = read_dump(dump)
        ents = read_ids(ids_path)
        reqs = align(reqs, ents)
        cols = run_columns(reqs, ents, windows, sel)
        kl = np.full(len(T), np.nan); top1 = np.zeros(len(T), dtype=bool); tgt = np.full(len(T), np.nan)
        valid = np.zeros(len(T), dtype=bool); skipped = 0
        with torch.no_grad():
            for w, hc in enumerate(cols):
                if hc is None:
                    skipped += 1
                    continue
                s, n = starts[w], ncols[w]
                h = torch.from_numpy(hc.view(np.int16).copy()).to(dev).view(torch.bfloat16)
                targets = torch.from_numpy(windows[w][0][windows[w][1]:].astype(np.int64)).to(dev)
                rid = torch.from_numpy(ref_ids[s:s + n].astype(np.int64)).to(dev)
                for c0 in range(0, n, 512):
                    lg = torch.matmul(h[c0:c0 + 512].float(), head32.T)
                    lse = torch.logsumexp(lg, dim=-1)
                    at = lg.gather(1, rid[c0:c0 + 512]) - lse[:, None]
                    sl = slice(s + c0, s + c0 + lg.shape[0])
                    kl[sl] = (p[sl] * (logp[sl] - at.cpu().numpy())).sum(1)
                    top1[sl] = (lg.argmax(-1).cpu().numpy() == ref_ids[sl, 0])
                    tgt[sl] = (lg.gather(1, targets[c0:c0 + 512, None])[:, 0] - lse).cpu().numpy()
                valid[s:s + n] = True
        print(f"== {name}: {valid.sum()} columns, {skipped} of {len(cols)} requests skipped (text differs)")
        req_of_col = np.repeat(np.arange(len(sel)), ncols)
        groups = {"all": valid}
        for gname in sorted(set(group.values())):
            m = np.array([group.get(sel[r]) == gname for r in req_of_col]) & valid
            if m.any():
                groups[gname] = m
        for gname, m in groups.items():
            print(f"  {gname:18s} n {m.sum():6d}  KL64 {kl[m].mean():.5f}  top-1 {top1[m].mean():.4f} "
                  f"({top1[m].sum()})  PPL {np.exp(-tgt[m].mean()):.4f}  ref PPL {np.exp(-T[m].mean()):.4f}")
        if base_top1 is None:
            base_top1, base_valid = top1, valid
        else:
            m = valid & base_valid
            print(f"  vs first: lost {np.sum(base_top1 & ~top1 & m)}, gained {np.sum(~base_top1 & top1 & m)} "
                  f"(on {m.sum()} common columns)")
        np.save(f"{out}.{name}.kl.npy", kl)


if __name__ == "__main__":
    if sys.argv[1] == "windows":
        a = sys.argv[2:]
        every = int(a[a.index("--every") + 1]) if "--every" in a else 1
        write_windows(a[0], a[1], a[2], every)
    elif sys.argv[1] == "score":
        score(sys.argv[2:])
    else:
        sys.exit(__doc__)
