#!/usr/bin/env python3
"""Builds the self-contained write-up: article.html (prose; <figure data-fig="NAME"> slots; {{v:..}} values;
{{ref:NAME}} figure references) + data.json and data/*.json (numbers) + style.css + js/*.js (interactive SVG figures,
vanilla JS) -> one index.html with everything inline. No network, no external assets.
usage: build.py [OUT=index.html]"""
import glob, html, json, re, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "index.html"
D = json.load(open("data.json"))
esc = lambda s: html.escape(str(s), quote=True)
lad = {r["key"]: r for r in D["ladder"]}
fid = D["fidelity"]

# ------------------------------------------------------------------ race (real streams, one chunk per round)
race = json.load(open("data/blog-race.json"))
LANES = [
    ("ar", "No speculation", "stock engine, official weights, one token per pass"),
    ("stock", "Stock engine", "official weights, 7-token draft chain"),
    ("releng", "Our engine", "official weights, 16-node verify tree"),
    ("final", "Our engine + our weights", "re-quantized target, fine-tuned drafter"),
]
CATS = {"coding": "Code", "writing": "Writing", "math_reasoning": "Math", "summarization": "Summarization"}
prompts = []
for i, row in enumerate(race["final"]):
    p = {"qid": row["qid"], "cat": row["cat"].replace("_", " "), "label": CATS.get(row["cat"], row["cat"]),
         "prompt": row["prompt"], "runs": {}}
    for k, *_ in LANES:
        rr = race[k][i]
        assert rr["qid"] == row["qid"]
        p["runs"][k] = {"t": [round(x, 4) for x in rr["t"]], "x": rr["x"], "k": rr["k"]}
    prompts.append(p)
RACE = {"max_tokens": 256, "lanes": [{"key": k, "title": t, "sub": s} for k, t, s in LANES], "prompts": prompts}

# ------------------------------------------------------------------ throughput plane
G = json.load(open("data/sb-groups.json"))
PLANE_CFG = [
    ("sb-stock", "stock", "Stock engine, official weights (7-token chain)", "stock engine", 12, 22, "start"),
    ("sb-rel-def", "rel_def", "Our engine, official weights, defaults (16-node tree)", "+ verify tree", 14, -20, "start"),
    ("sb-rel-eng", "rel_eng", "Our engine, official weights, all engine changes", "+ engine kernels", 0, -34, "middle"),
    ("sb-final-a", "final", "Our engine + our weights and drafter", "+ our weights", -14, -26, "end"),
]
plane_cfgs = []
for f, key, label, short, lx, ly, anc in PLANE_CFG:
    g = G[f]
    o = g["overall"]
    plane_cfgs.append({"key": key, "label": label, "short": short, "tau": o["tau"], "ms": o["ms"], "tps": lad[key]["tps"] if key != "final" else lad["final"]["tps"],
                       "lx": lx, "ly": ly, "anchor": anc, "groups": {k: v for k, v in g.items() if k != "overall"}})
PLANE = {"configs": plane_cfgs, "steps": [
    {"title": "16-node verify tree", "text": "Upstream's verify trees, merged into the Ada port, with 15 drafts instead of a 7-token chain.",
     "lines": ["verify tree", "+29% tokens/round"], "dx": 10, "dy": 0, "anchor": "start"},
    {"title": "Engine kernels", "text": "Two-level LM head, L2 fills, faster small kernels, whole-program build, INT8 V cache, FP16 state, tree temperature.",
     "lines": ["kernels", "−6% round time"], "dx": 0, "dy": 30, "anchor": "middle"},
    {"title": "Our weights and drafter", "text": "GPTQ + 3-bit MLP layers + end-to-end scale tuning; fine-tuned, re-quantized drafter.",
     "lines": ["our weights + drafter", "−20% round time, +5% tokens"], "dx": 0, "dy": 34, "anchor": "middle"},
]}

# ------------------------------------------------------------------ round timelines (nsys)
RR = json.load(open("data/blog-rounds.json"))
TRACES = [("first", "Oct 2", "official weights, first tree"), ("start", "Oct 3", "official weights, tuned GEMMs"),
          ("final", "Oct 6", "final, our weights")]
traces = []
for key, day, sub in TRACES:
    tr = RR[key]
    names, ks = tr["names"], tr["kernels"]
    nm = lambda k: names[k[3]]
    for k in ks:  # LM heads by role: the Q8/Q4 head GEMMs early on, the 3-bit screens later
        if (nm(k) == "q4_ksplit_mma_kernel" and k[1] > 300) or (nm(k) == "q8_ksplit_mma_kernel" and k[1] > 1000):
            k[2] = "head"
    i_ph = next(i for i, k in enumerate(ks) if k[2] == "head")
    i_vh = max(i for i, k in enumerate(ks) if k[2] == "head")
    i_sel = max(i for i, k in enumerate(ks) if names[k[3]].startswith("selector"))
    end = lambda i: ks[i][0] + ks[i][1]
    span = tr["span_us"]
    phases = [["commit", 0, end(0)], ["host", end(0), ks[1][0]], ["drafter", ks[1][0], ks[i_ph][0]],
              ["proposal head + tree", ks[i_ph][0], end(i_sel)], ["target verify: 64 layers", end(i_sel), ks[i_vh][0]],
              ["verify head", ks[i_vh][0], end(i_vh)], ["accept", end(i_vh), span]]
    qk = [k[0] for k in ks if nm(k) == "qk_rmsnorm_rope_kernel"]
    busy, cs, ce = 0.0, None, None
    for k in ks:
        if ce is None or k[0] > ce:
            if ce is not None: busy += ce - cs
            cs, ce = k[0], k[0] + k[1]
        else:
            ce = max(ce, k[0] + k[1])
    busy += ce - cs
    cls = {}
    for k in ks: cls[k[2]] = cls.get(k[2], 0) + k[1]
    traces.append({"key": key, "label": day, "sub": sub, "span": span, "k": [[round(a, 1), round(b, 2), c, n] for a, b, c, n in ks],
                   "names": names, "phases": [[p, round(a, 1), round(b, 1)] for p, a, b in phases],
                   "ranges": {"drafter": [ks[1][0], end(i_sel)], "block": [qk[7], qk[8]], "tail": [ks[i_vh][0] - 250, span]},
                   "idle_us": round(span - busy, 1), "class_us": {k: round(v, 1) for k, v in cls.items()}})
ROUNDS = {"traces": traces}

# ------------------------------------------------------------------ trees
TT = json.load(open("data/blog-trees.json"))
for q, p in zip(TT, prompts):
    q["label"] = p["label"]
    for r in q["rounds"]:
        r["ctx"] = r["ctx"][-480:]
    q.pop("text", None)

# ------------------------------------------------------------------ Q3 sensitivity map (log, 2026-10-04 02:20)
GU = "Q3 kernel 80.7 vs 97.7 µs per layer (T = 16): 0.27 ms less per round"
DN = "Q3 kernel 42.1 vs 50.1 µs per layer: 0.13 ms less per round"
GD = "no time saved: our GDN Q3 routes lost bandwidth on Ada"
cells = [
    ("gate_up", 0, 0, 0.0025, 374, True, GU, 272, False), ("gate_up", 1, 1, 0.0068, 374, True, GU, 272, False),
    ("gate_up", 2, 2, 0.0318, 374, False, GU, 272, False), ("gate_up", 3, 3, 0.0227, 374, False, GU, 272, False),
    ("down", 0, 0, 0.0023, 187, True, DN, 128, False), ("down", 1, 1, 0.0058, 187, False, DN, 128, False),
    ("down", 2, 2, 0.0233, 187, False, DN, 128, False), ("down", 3, 3, 0.0148, 187, False, DN, 128, False),
    ("gdn_vz", 0, 1, 0.0041, 197, False, GD, 0, True), ("gdn_vz", 2, 3, 0.0153, 197, False, GD, 0, True),
    ("gdn_qk", 0, 1, 0.00095, 66, False, GD, 0, True), ("gdn_qk", 2, 3, 0.0024, 66, False, GD, 0, True),
    ("gdn_out", 0, 1, 0.0032, 98, False, GD, 0, True),
]
ROWN = {"gate_up": "MLP gate/up", "down": "MLP down", "gdn_vz": "GDN value/z", "gdn_qk": "GDN query/key", "gdn_out": "GDN output"}
SENS = {"rows": [{"key": "gate_up", "label": "MLP gate/up", "sub": "34,816 × 5,120"}, {"key": "down", "label": "MLP down", "sub": "5,120 × 17,408"},
                 {"key": "gdn_vz", "label": "GDN value/z", "sub": "48 GDN layers"}, {"key": "gdn_qk", "label": "GDN query/key", "sub": ""},
                 {"key": "gdn_out", "label": "GDN output", "sub": ""}],
        "cells": [{"id": f"{r}{a}{b}", "row": r, "c0": a, "c1": b, "dkl": d, "mb": mb, "adopted": ad, "time": tm, "us": us, "slow": sl,
                   "name": f"{ROWN[r]}, layers {16 * a}–{16 * b + 15}"} for r, a, b, d, mb, ad, tm, us, sl in cells],
        "budget": 0.0064, "final_kl": fid["final"]["kl"]["all"], "release_kl": fid["release"]["kl"]["all"]}

# ------------------------------------------------------------------ drift-balanced A/B (job 126)
DRIFT = {"runs": [["A", 16.7856], ["B", 16.7542], ["B", 16.7671], ["A", 16.7937], ["B", 16.7751], ["A", 16.8127], ["A", 16.8028], ["B", 16.7899]],
         "a_label": "8 MB attention fill", "b_label": "4 MB attention fill"}

WORKLOADS = [["json", "JSON", "77-token prompt"], ["code", "code", "69-token prompt"], ["prose", "prose", "34-token prompt"],
             ["sustained_decode", "long answer", "1,024 tokens out"], ["short_refactor", "refactor", "2K-token prompt"],
             ["medium_feature", "feature", "11K-token prompt"], ["long_review", "review", "32K-token prompt"],
             ["context_copy", "copy", "26K-token prompt"]]

DATA = {"race": RACE, "plane": PLANE, "rounds": ROUNDS, "trees": TT, "sens": SENS, "drift": DRIFT,
        "quality_path": D["quality_path"], "progress": D["progress"], "suite": D["suite"], "suite_0310": D["suite_0310"],
        "workloads": WORKLOADS}

# ------------------------------------------------------------------ values used in the prose
final_tps = lad["final"]["tps"]
st, fi = next(t for t in traces if t["key"] == "start"), next(t for t in traces if t["key"] == "final")
V = {
    "final_tps": f"{final_tps:.1f}", "final_tps0": f"{final_tps:.0f}",
    "releng_tps": f"{lad['rel_eng']['tps']:.1f}", "releng_tps0": f"{lad['rel_eng']['tps']:.0f}",
    "reldef_tps": f"{lad['rel_def']['tps']:.1f}", "stock_tps": f"{lad['stock']['tps']:.1f}",
    "speedup": f"{final_tps / lad['stock']['tps']:.2f}", "engine_x": f"{lad['rel_eng']['tps'] / lad['stock']['tps']:.2f}",
    "model_x": f"{final_tps / lad['rel_eng']['tps']:.2f}",
    "kl_final": f"{fid['final']['kl']['all']:.4f}", "kl_rel": f"{fid['release']['kl']['all']:.4f}",
    "ppl_final": f"{fid['final']['ppl']['all']:.2f}", "ppl_rel": f"{fid['release']['ppl']['all']:.2f}",
    "n_pred": f"{fid['n']:,}", "round_ms": f"{1000 * lad['final']['tps_step'] / final_tps:.1f}",
    "start_round_ms": f"{st['span'] / 1000:.1f}", "start_stream_ms": f"{(st['class_us'].get('gemm', 0) + st['class_us'].get('head', 0)) / 1000:.1f}",
    "final_idle_ms": f"{fi['idle_us'] / 1000:.2f}", "final_stream_ms": f"{(fi['class_us'].get('gemm', 0) + fi['class_us'].get('head', 0)) / 1000:.1f}",
    "final_round_trace_ms": f"{fi['span'] / 1000:.1f}",
    "run_cmd": D.get("run_cmd", ""),
}

# ---------------------------------------------------------------- tables
def table_quality():
    rel, fin, ref = fid["release"], fid["final"], fid["ref_ppl"]
    def row(name, d, bold=False):
        b = (lambda s: f"<b>{s}</b>") if bold else (lambda s: s)
        return (f"<tr><td>{name}</td><td>{b(f'{d['kl']['prose']:.4f}')}</td><td>{b(f'{d['kl']['code']:.4f}')}</td>"
                f"<td>{b(f'{d['kl']['all']:.4f}')}</td><td>{b(f'{d['top1']:.4f}')}</td>"
                f"<td>{b(f'{d['ppl']['all']:.3f}')}</td></tr>")
    return ('<div class="tbl"><table><thead><tr><th></th><th>KL<sub>64</sub> prose</th><th>KL<sub>64</sub> code</th>'
            '<th>KL<sub>64</sub></th><th>top-1</th><th>PPL</th></tr></thead><tbody>'
            + row("release (RTN Q4/Q5)", rel) + row("ours", fin, True)
            + f'</tbody></table><p class="tnote">{fid["n"]:,} predictions; FP32 reference perplexity {ref["all"]:.3f} '
              f'(prose {ref["prose"]:.2f}, code {ref["code"]:.3f}). Lower KL and PPL are better. Top-1: '
              f'{fin["top1_n"]:,} vs {rel["top1_n"]:,} agreements.</p></div>')


NEG = [
    ("Trellis quantization (EXL3-style)", "not adopted", "Matching our scalar quality needs ≈3.75 bits per weight, at most 9% fewer bytes, and the decoder is slower."),
    ("Entropy-coded weights", "closed", "Decoding needs ≈2 T codes/s; the GPU decodes 1.5–1.65 T/s."),
    ("32-node verify trees", "+11% tokens, +12–39% GEMM time", "The activations outgrow the 16-column tiles."),
    ("Prefetching weights from a side stream", "2–3% slower", "It slows the co-scheduled kernel (1.7 → 16 µs), and the graph joins eat the gain."),
    ("GDN or attention work on a second stream", "±0", "The kernels share DRAM and stretch each other one for one."),
    ("RMSNorms folded into GEMM epilogues", "±0", "Removing the norm also removes the L2-fill window it carried."),
    ("Larger L2 fills", "0.2–0.3% slower", "The stretches they ride on are already full."),
    ("Four CTAs per SM for gate/up", "4 µs slower per call", "More CTAs mean more activation reloads and more L2 contention."),
    ("More drafter fine-tuning", "±0", "Higher rank, more epochs, soft labels and new data all saturate after one run."),
    ("Q3 GDN projections, Q5 MLP down", "0.8% slower", "Those kernels lose bandwidth; only Q3 MLP layers turn bytes into time."),
    ("2-bit LM-head screen", "rejected", "Top-1 recall is 0.987."),
    ("FP32 residual stream", "top-1 unchanged", "The top-1 loss comes from BF16 projection inputs and outputs, not the residual."),
    ("SM-balanced grids, NUMA pinning, split CUDA graphs", "±0", ""),
]


def table_negative():
    rows = "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td></tr>" for a, b, c in NEG)
    return ('<div class="tbl"><table class="neg"><thead><tr><th>idea</th><th>result</th><th>why</th></tr></thead>'
            f"<tbody>{rows}</tbody></table></div>")



TABLES = {"quality": table_quality, "negative": table_negative}

art = open("article.html").read()
art = re.sub(r"\{\{table:([a-z]+)\}\}", lambda m: TABLES[m[1]](), art)
order = re.findall(r'<figure data-fig="([a-z]+)"', art)
num = {n: i + 1 for i, n in enumerate(order)}
missing = [n for n in re.findall(r"\{\{ref:([a-z]+)\}\}", art) if n not in num]
assert not missing, f"unknown figure refs: {missing}"
art = re.sub(r'<figure data-fig="([a-z]+)"([^>]*)>\s*<figcaption>',
             lambda m: f'<figure data-fig="{m[1]}" id="fig-{m[1]}"{m[2]}><figcaption><b>Figure {num[m[1]]}.</b> ', art)
art = re.sub(r"\{\{ref:([a-z]+)\}\}", lambda m: f'<a href="#fig-{m[1]}">Figure {num[m[1]]}</a>', art)
art = re.sub(r"\{\{v:([a-z0-9_]+)\}\}", lambda m: esc(V[m[1]]), art)
assert "{{" not in art, re.findall(r"\{\{[^}]*\}\}", art)[:5]

css = open("style.css").read()
js = "\n".join(open(f).read() for f in sorted(glob.glob("js/*.js")))
js += "\nfor (const f of document.querySelectorAll('figure[data-fig]')) { const fn = FIGS[f.dataset.fig]; " \
      "if (fn) try { fn(f); } catch (e) { console.error(f.dataset.fig, e); } }\n"
data_js = "window.__DATA__=" + json.dumps(DATA, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/") + ";"
title = re.sub(r"<[^>]+>", "", re.search(r"<h1>(.*?)</h1>", art, re.S)[1]).strip()
page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="Batch-1 speculative decoding of Qwen3.8-27B on one RTX 4090: {V['releng_tps']} tok/s on the official weights, {V['final_tps']} with re-quantized weights that are closer to full precision than the release.">
<style>{css}</style>
</head>
<body>
<main>
{art}
</main>
<noscript><p class="noscript">The figures on this page are drawn with JavaScript.</p></noscript>
<script>{data_js}</script>
<script>(() => {{"use strict";
{js}
}})();</script>
</body>
</html>
"""
open(OUT, "w").write(page)
print(f"wrote {OUT}: {len(page) // 1024} KB, figures: {', '.join(f'{n}={i}' for n, i in num.items())}")
