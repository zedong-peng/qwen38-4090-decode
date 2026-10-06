#!/usr/bin/env python3
"""Builds the self-contained write-up: article.html (prose; <figure data-fig="NAME"> slots; {{v:..}} values;
{{ref:NAME}} figure references) + data.json and data/*.json (numbers) + style.css + js/*.js (interactive SVG figures,
vanilla JS) -> one index.html with everything inline. No network, no external assets.
usage: build.py [OUT=index.html]"""
import glob, html, json, re, sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "index.html"
D = json.load(open("data.json"))
esc = lambda s: html.escape(str(s), quote=True)
fid = D["fidelity"]

# ------------------------------------------------------------------ race (real streams, one chunk per round)
race = json.load(open("data/blog-race.json"))
FM = json.load(open("data/famous.json"))  # same-session engine comparison (jobs 131-132)
SB = FM["sb"]
LN = max((n for n in (4, 7) if f"sb-llama-n{n}" in SB), key=lambda n: SB[f"sb-llama-n{n}"]["overall"]["tps"])  # llama.cpp: the better recipe
LANES = [
    ("ar", "No speculation", "Cinference, one token per forward pass"),
    ("llama", "llama.cpp", f"UD-Q4_K_XL + DFlash2 drafter, up to {LN} drafts"),
    ("vllm", "vLLM", "community RTX 4090 recipe, 7 drafts"),
    ("stock", "Cinference-4090", "the engine we started from, 7-token chain"),
    ("releng", "Ours", "release weights, 16-node verify tree"),
    ("final", "Ours + our weights", "re-quantized target, fine-tuned drafter"),
]
RACE_SRC = {"llama": f"llama-n{LN}"}
CATS = {"coding": "Code", "writing": "Writing", "math_reasoning": "Math", "summarization": "Summarization"}
prompts = []
for i, row in enumerate(race["final"]):
    p = {"qid": row["qid"], "cat": row["cat"].replace("_", " "), "label": CATS.get(row["cat"], row["cat"]),
         "prompt": row["prompt"], "runs": {}}
    for k, *_ in LANES:
        rr = race[RACE_SRC.get(k, k)][i]
        assert rr["qid"] == row["qid"]
        # aligned at the lane's first token: the race is about decode speed (llama.cpp's first token takes ~0.55 s)
        p["runs"][k] = {"t": [round(x - rr["t"][0], 4) for x in rr["t"]], "x": rr["x"], "k": rr["k"]}
    prompts.append(p)
RACE = {"max_tokens": 256, "lanes": [{"key": k, "title": t, "sub": s} for k, t, s in LANES], "prompts": prompts}

# ------------------------------------------------------------------ one session (jobs 131-132): leaderboard, plane, ladder
def mean_sb(a, b):
    out = {}
    for g in a:
        x, y = a[g], b[g]
        tps, tau = (x["tps"] + y["tps"]) / 2, (x["tau"] + y["tau"]) / 2
        out[g] = {"n": x["n"], "tps": round(tps, 2), "tau": round(tau, 3), "ms": round(1000 * tau / tps, 2)}
    return out
SB["final"] = mean_sb(SB["sb-final-a"], SB["sb-final-b"])
suite_of = lambda label: FM["suite"][label]
BOARD_E = [
    ("llama", "llama.cpp", "llama.cpp", f"UD-Q4_K_XL, ≤{LN} drafts", SB[f"sb-llama-n{LN}"], suite_of(f"ss-llama-n{LN}"), False,
     f"llama.cpp a4d880f: unsloth UD-Q4_K_XL target, DFlash2 Q4_K_M drafter, up to {LN} drafts, F16 KV cache"),
    ("vllm", "vLLM", "vLLM", "AutoRound W4A16, 7 drafts", SB["sb-vllm-k7"], suite_of("ss-vllm-k7"), False,
     "vLLM 0.27.1 with sidnaZ's single-user RTX 4090 recipe: AutoRound W4A16 target, W4A16 DFlash2 drafter, 7 drafts, BF16 KV cache"),
    ("stock", "Cinference-4090", "Cinference-4090", "release weights, 7-token chain", SB["sb-stock"], suite_of("ss-stock"), False,
     "jram4's Ada port of Cinference (70ebb12), the engine we started from, on the NInfer release container"),
    ("rel_eng", "Ours", "ours (release weights)", "same release weights", SB["sb-rel-eng"], suite_of("ss-rel-eng"), True,
     "our engine on the same NInfer release container as Cinference-4090 (Part I)"),
    ("final", "Ours + our weights", "ours", "re-quantized target and drafter", SB["final"], suite_of("ss-final"), True,
     "our engine with our re-quantized target and fine-tuned drafter (Part II); Spec-Bench is the mean of two runs"),
]
BOARD_E.sort(key=lambda e: (e[6], e[4]["overall"]["tps"]))
BOARD = {"engines": [{"key": k, "name": n, "short": sh, "sub": sub, "groups": g, "suite": su, "ours": o, "long": lo}
                     for k, n, sh, sub, g, su, o, lo in BOARD_E], "base": "vllm", "bases": ["vllm", "llama", "stock"]}
BE = {e["key"]: e["groups"]["overall"] for e in BOARD["engines"]}
BS = {e["key"]: {w: v["tps"] for w, v in e["suite"].items()} for e in BOARD["engines"]}

# ------------------------------------------------------------------ throughput plane
PLANE_CFG = [
    ("sb-stock", "stock", "Cinference-4090, release weights (7-token chain)", "Cinference-4090", 12, 22, "start"),
    ("sb-rel-def", "rel_def", "Our engine, release weights, defaults (16-node tree)", "+ verify tree", 14, -20, "start"),
    ("sb-rel-eng", "rel_eng", "Our engine, release weights, all engine changes", "+ engine kernels", 0, -34, "middle"),
    ("final", "final", "Our engine + our weights and drafter", "+ our weights", -14, -26, "end"),
]
plane_cfgs = []
for f, key, label, short, lx, ly, anc in PLANE_CFG:
    o = SB[f]["overall"]
    plane_cfgs.append({"key": key, "label": label, "short": short, "tau": o["tau"], "ms": o["ms"], "tps": o["tps"],
                       "lx": lx, "ly": ly, "anchor": anc, "groups": {k: v for k, v in SB[f].items() if k != "overall"}})
P0, P1, P2, P3 = (c for c in plane_cfgs)
chg = lambda a, b: f"{'+' if b >= a else '−'}{abs(100 * (b / a - 1)):.0f}%"
STEPS = {"tree_tau": chg(P0["tau"], P1["tau"]), "tree_ms": chg(P0["ms"], P1["ms"]), "kern_ms": chg(P1["ms"], P2["ms"]),
         "w_ms": chg(P2["ms"], P3["ms"]), "w_tau": chg(P2["tau"], P3["tau"])}
OTHERS = [{"key": "vllm", "label": "vLLM, community RTX 4090 recipe (7 drafts)", "short": "vLLM", "lx": -14, "ly": 4, "anchor": "end", **BE["vllm"]},
          {"key": "llama", "label": f"llama.cpp, UD-Q4_K_XL + DFlash2 (up to {LN} drafts)", "short": "llama.cpp", "lx": 14, "ly": 4, "anchor": "start", **BE["llama"]}]
_ms = [c["ms"] for c in plane_cfgs + OTHERS]; _ta = [c["tau"] for c in plane_cfgs + OTHERS]
# x stays on our range; an engine with longer rounds is drawn at the right edge as an off-chart marker
DOMS = {"overall": [15.5, 24, min(3.6, min(_ta) - 0.25), max(5.9, max(_ta) + 0.3)],
        "tasks": [15.5, 24, 2.8, 9.6]}
PLANE = {"configs": plane_cfgs, "others": OTHERS, "doms": DOMS, "steps": [
    {"title": "16-node verify tree", "text": "Upstream's verify trees, merged into the Ada port, with 15 drafts instead of a 7-token chain.",
     "lines": ["verify tree", f"{STEPS['tree_tau']} tokens/round"], "dx": 10, "dy": 0, "anchor": "start"},
    {"title": "Engine kernels", "text": "Two-level LM head, L2 fills, faster small kernels, whole-program build, INT8 V cache, FP16 state, tree temperature.",
     "lines": ["kernels", f"{STEPS['kern_ms']} round time"], "dx": 0, "dy": 30, "anchor": "middle"},
    {"title": "Our weights and drafter", "text": "GPTQ + 3-bit MLP layers + end-to-end scale tuning; fine-tuned, re-quantized drafter.",
     "lines": ["our weights + drafter", f"{STEPS['w_ms']} round time, {STEPS['w_tau']} tokens"], "dx": 0, "dy": 34, "anchor": "middle"},
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

DATA = {"race": RACE, "board": BOARD, "plane": PLANE, "rounds": ROUNDS, "trees": TT, "sens": SENS, "drift": DRIFT,
        "workloads": WORKLOADS}

# ------------------------------------------------------------------ values used in the prose (all Spec-Bench numbers: one session)
st, fi = next(t for t in traces if t["key"] == "start"), next(t for t in traces if t["key"] == "final")
F, RE, RD, ST = BE["final"]["tps"], BE["rel_eng"]["tps"], SB["sb-rel-def"]["overall"]["tps"], BE["stock"]["tps"]
ratio = lambda a, b, w: BS[a][w] / BS[b][w]
LONG = ("medium_feature", "long_review")
V = {
    "final_tps": f"{F:.1f}", "final_tps0": f"{F:.0f}", "releng_tps": f"{RE:.1f}", "releng_tps0": f"{RE:.0f}",
    "reldef_tps": f"{RD:.1f}", "stock_tps": f"{ST:.1f}",
    "speedup": f"{F / ST:.2f}", "engine_x": f"{RE / ST:.2f}", "model_x": f"{F / RE:.2f}",
    "tree_tau": STEPS["tree_tau"], "kern_ms": STEPS["kern_ms"], "w_ms": STEPS["w_ms"], "w_tau": STEPS["w_tau"],
    "kl_final": f"{fid['final']['kl']['all']:.4f}", "kl_rel": f"{fid['release']['kl']['all']:.4f}",
    "ppl_final": f"{fid['final']['ppl']['all']:.2f}", "ppl_rel": f"{fid['release']['ppl']['all']:.2f}",
    "n_pred": f"{fid['n']:,}", "round_ms": f"{BE['final']['ms']:.1f}",
    "start_round_ms": f"{st['span'] / 1000:.1f}", "start_stream_ms": f"{(st['class_us'].get('gemm', 0) + st['class_us'].get('head', 0)) / 1000:.1f}",
    "final_idle_ms": f"{fi['idle_us'] / 1000:.2f}", "final_round_trace_ms": f"{fi['span'] / 1000:.1f}",
    "run_cmd": D.get("run_cmd", ""),
    "vllm_tps": f"{BE['vllm']['tps']:.1f}", "vllm_tps0": f"{BE['vllm']['tps']:.0f}", "llama_tps": f"{BE['llama']['tps']:.1f}",
    "llama_tps0": f"{BE['llama']['tps']:.0f}", "llama_n": str(LN),
    "x_vllm_eng": f"{RE / BE['vllm']['tps']:.2f}", "x_vllm_final": f"{F / BE['vllm']['tps']:.2f}",
    "x_llama_eng": f"{RE / BE['llama']['tps']:.2f}", "x_llama_final": f"{F / BE['llama']['tps']:.2f}",
    "vllm_tau": f"{BE['vllm']['tau']:.2f}", "vllm_ms": f"{BE['vllm']['ms']:.1f}", "llama_tau": f"{BE['llama']['tau']:.2f}",
    "llama_ms": f"{BE['llama']['ms']:.1f}", "final_tau": f"{BE['final']['tau']:.2f}",
    "long_x_stock": f"{min(ratio('final', 'stock', w) for w in LONG):.2f}",
    "long_x_vllm": f"{min(ratio('final', 'vllm', w) for w in LONG):.2f}",
    "best_x_vllm": f"{max(ratio('final', 'vllm', w) for w, *_ in WORKLOADS):.2f}",
    "tree_tau_n": f"{abs(100 * (P1['tau'] / P0['tau'] - 1)):.0f}",
    "session": "in one session",
}
_fs = next(e for e in BOARD["engines"] if e["key"] == "final")["suite"]
_lm = [1000 * _fs[w]["tau"] / _fs[w]["tps"] for w in LONG if _fs[w].get("tau")]
V["long_ms"] = f"{min(_lm):.0f}–{max(_lm):.0f}" if len(_lm) == 2 else "18–20"

# ---------------------------------------------------------------- tables
def table_quality():
    """Part II in numbers: every adopted weight step, release to final (engine KL against the FP32 reference)."""
    rel, fin, ref = fid["release"], fid["final"], fid["ref_ppl"]
    steps = [("release (RTN Q4/Q5)", D["quality_path"][0], rel, False), ("GPTQ, at the release's bit allocation", D["quality_path"][1], None, False),
             ("all projections Q4, 8-bit group scales", D["quality_path"][2], None, False),
             ("+ Q3 MLP gate/up, layers 0–15", D["quality_path"][3], None, False),
             ("+ end-to-end scale tuning, Q3 gate/up 0–31", D["quality_path"][4], None, False),
             ("final: + tuning v5, Q3 MLP down 0–15, INT8 V cache, FP16 state", D["quality_path"][6], fin, True)]
    f4 = lambda v: "–" if v is None else f"{v:.4f}"
    rows = ""
    for name, q, full, bold in steps:
        b = (lambda x: f"<b>{x}</b>") if bold else (lambda x: x)
        cells = [f"{q['gb']:.2f}", f4(full["kl"]["prose"]) if full else "–", f4(full["kl"]["code"]) if full else "–", f4(q["kl"]),
                 f4(q.get("top1")), f"{full['ppl']['all']:.3f}" if full else "–"]
        rows += f"<tr><td>{name}</td>" + "".join(f"<td>{b(c)}</td>" for c in cells) + "</tr>"
    return ('<div class="tbl"><table><thead><tr><th></th><th>target GB per round</th><th>KL<sub>64</sub> prose</th>'
            '<th>KL<sub>64</sub> code</th><th>KL<sub>64</sub></th><th>top-1</th><th>PPL</th></tr></thead><tbody>' + rows
            + f'</tbody></table><p class="tnote">Engine fidelity against the FP32 reference over {fid["n"]:,} predictions '
              f'(reference perplexity {ref["all"]:.3f}). Lower KL and PPL are better. Bytes are the target weights read per '
              f'verify round, from the bit allocation. The two Q3 steps were scored on KL only; the final model has '
              f'{fin["top1_n"]:,} top-1 agreements against the release\'s {rel["top1_n"]:,}.</p></div>')


NEG = [
    ("Trellis quantization (EXL3-style)", "not adopted", "Matching our scalar quality needs ≈3.75 bits per weight, at most 9% fewer bytes, and the decoder is slower."),
    ("Entropy-coded weights", "closed", "Decoding needs ≈2 T codes/s; the GPU decodes 1.5–1.65 T/s."),
    ("GDN or attention work on a second stream", "±0", "The kernels share DRAM and stretch each other one for one."),
    ("Four CTAs per SM for gate/up", "4 µs slower per call", "More CTAs mean more activation reloads and more L2 contention."),
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
<meta name="description" content="Batch-1 speculative decoding of Qwen3.8-27B on one RTX 4090: {V['releng_tps0']} tok/s on the official weights and {V['final_tps0']} with re-quantized ones, against {V['vllm_tps0']} for vLLM and {V['llama_tps0']} for llama.cpp on the same card.">
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
