#!/usr/bin/env python3
"""Builds the self-contained write-up: article.html (prose with {{v:..}}, {{fig:..}}, {{table:..}} slots) +
data.json (numbers) -> index.html with inline CSS and hand-drawn SVG figures. No network, no external assets.
usage: build.py [OUT=index.html]"""
import html, json, math, re, sys

D = json.load(open("data.json"))
OUT = sys.argv[1] if len(sys.argv) > 1 else "index.html"
esc = lambda s: html.escape(str(s), quote=True)

lad = {r["key"]: r for r in D["ladder"]}
fid = D["fidelity"]
final_tps = lad["final"]["tps"]
V = {
    "final_tps": f"{final_tps:.1f}",
    "final_tps0": f"{final_tps:.0f}",
    "stock_tps": f"{lad['stock']['tps']:.1f}",
    "speedup": f"{final_tps / lad['stock']['tps']:.2f}",
    "kl_final": f"{fid['final']['kl']['all']:.4f}",
    "kl_rel": f"{fid['release']['kl']['all']:.4f}",
    "ppl_final": f"{fid['final']['ppl']['all']:.2f}",
    "ppl_rel": f"{fid['release']['ppl']['all']:.2f}",
    "n_pred": f"{fid['n']:,}",
    "round_ms": f"{1000 * lad['final']['tps_step'] / final_tps:.1f}",
    "engine_x": f"{lad['rel_eng']['tps'] / lad['stock']['tps']:.2f}",
    "model_x": f"{final_tps / lad['rel_eng']['tps']:.2f}",
    "run_cmd": D.get("run_cmd", ""),
}


def figure(fid_, svg, caption, wide=False):
    return (f'<figure id="fig-{fid_}" class="{"wide" if wide else ""}">{svg}'
            f'<figcaption>{caption}</figcaption></figure>')


def svg_open(w, h, label):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
            f'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">')


def text(x, y, s, cls="t", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'


# ---------------------------------------------------------------- Figure: ladder
def fig_ladder():
    rows = [r for r in D["ladder"] if r.get("tps")]
    W, left, right, rh, top = 760, 16, 210, 58, 10
    H = top + rh * len(rows) + 30
    vmax = max(r["tps"] for r in rows) * 1.02
    sx = lambda v: left + (W - left - right) * v / vmax
    cls = {"stock": "c-gray", "rel_def": "c-acc1", "rel_eng": "c-acc2", "final": "c-acc3"}
    out = [svg_open(W, H, "Spec-Bench-480 tok/s ladder")]
    ya = top + rh * len(rows) + 4
    for v in range(0, int(vmax) + 1, 50):
        out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top + 18}" y2="{ya}" class="grid"/>')
        out.append(text(sx(v), ya + 16, str(v), "t-axis", "middle"))
    out.append(text(sx(vmax) + 30, ya + 16, "tok/s", "t-axis"))
    for i, r in enumerate(rows):
        y = top + i * rh
        out.append(text(left, y + 14, r["label"], "t-label"))
        out.append(f'<rect x="{left}" y="{y + 20}" width="{sx(r["tps"]) - left:.1f}" height="24" rx="3" class="{cls.get(r["key"], "c-acc2")}"/>')
        out.append(text(sx(r["tps"]) + 8, y + 37, f'{r["tps"]:.1f}', "t-value"))
        if r.get("tps_step"):
            ms = 1000 * r["tps_step"] / r["tps"]
            out.append(text(sx(r["tps"]) + 54, y + 37, f'{r["tps_step"]:.2f} tok/round · {ms:.1f} ms', "t-small"))
    out.append("</svg>")
    cap = ("<b>Figure 1.</b> Spec-Bench-480 decode throughput on one RTX 4090: 480 prompts, greedy, batch 1, at most "
           "256 new tokens. All bars were measured in one session on one card. The middle two bars keep the released "
           "weights and isolate the engine; the last bar adds our quantized model and fine-tuned drafter. "
           "tok/round is the mean number of tokens accepted per verify round. The engine/model split is not unique: "
           "engine changes save a roughly fixed time per round, which is a larger share of the final model's shorter rounds.")
    return figure("ladder", "".join(out), cap)


# ---------------------------------------------------------------- Figure: round anatomy
def fig_anatomy():
    A = D["anatomy"]
    segs = [("gemm", "weight GEMMs", "c-acc3"), ("heads", "LM-head screens", "c-acc2"),
            ("small", "small kernels", "c-warm"), ("idle", "idle", "c-idle")]  # legend shows start -> final in ms
    W, left, right, top, bh, gap = 760, 150, 70, 14, 34, 26
    tmax = max(sum(A[k][s] for s, _, _ in segs) for k in ("start", "final")) * 1.0
    sx = lambda us: (W - left - right) * us / tmax
    out = [svg_open(W, top + 2 * (bh + gap) + 50, "verify round anatomy")]
    for i, k in enumerate(("start", "final")):
        y = top + i * (bh + gap)
        out.append(text(left - 10, y + bh / 2 + 5, A[k]["label"].split(",")[0], "t-label", "end"))
        out.append(text(left - 10, y + bh / 2 + 20, A[k]["label"].split(",", 1)[1].strip(), "t-small", "end"))
        x = left
        for s, name, c in segs:
            w = sx(A[k][s])
            if w <= 0:
                continue
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bh}" class="{c}"/>')
            if w > 46:
                out.append(text(x + w / 2, y + bh / 2 + 5, f'{A[k][s] / 1000:.1f}', "t-in", "middle"))

            x += w
        tot = sum(A[k][s] for s, _, _ in segs)
        out.append(text(x + 8, y + bh / 2 + 5, f'{tot / 1000:.1f} ms', "t-value"))
    # legend
    ly = top + 2 * (bh + gap) + 6
    lx = 14
    for s, name, c in segs:
        a, b = A["start"][s], A["final"][s]
        lab = f"{name}  {a / 1000:.2f} → {b / 1000:.2f}" if a else f"{name}  {b / 1000:.2f}"
        out.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" class="{c}"/>')
        out.append(text(lx + 17, ly + 11, lab, "t-small"))
        lx += 26 + 6.3 * len(lab) + 10
    out.append("</svg>")
    cap = ("<b>Figure 2.</b> GPU time per verify round in ms, from nsys traces of a code prompt. Top: the first "
           "trace of our tuned engine with the released weights. Its idle time was not measured separately and is "
           "folded into the other segments. Bottom: the final configuration. Weight GEMMs include the drafter's "
           "projections. nsys adds a few percent to every kernel.")
    return figure("anatomy", "".join(out), cap)


# ---------------------------------------------------------------- Figure: quality vs bytes
def fig_quality():
    P = D["quality_path"]
    W, H, l, r, t, b = 760, 420, 70, 30, 20, 50
    x0, x1, y0, y1 = 11.3, 14.7, 0.036, 0.072
    sx = lambda v: l + (W - l - r) * (v - x0) / (x1 - x0)
    sy = lambda v: t + (H - t - b) * (1 - (v - y0) / (y1 - y0))
    out = [svg_open(W, H, "KL versus bytes per round")]
    for v in [11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5]:
        out.append(f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{t}" y2="{H - b}" class="grid"/>')
        out.append(text(sx(v), H - b + 18, f"{v:.1f}", "t-axis", "middle"))
    for v in [0.04, 0.05, 0.06, 0.07]:
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>')
        out.append(text(l - 8, sy(v) + 4, f"{v:.2f}", "t-axis", "end"))
    out.append(text((l + W - r) / 2, H - 10, "target weight bytes read per verify round (GB, from the bit allocation)", "t-axis", "middle"))
    out.append(text(16, t + (H - t - b) / 2, "KL64 vs FP32", "t-axis", "middle",
                    f'transform="rotate(-90 16 {t + (H - t - b) / 2:.1f})"'))
    rel = P[0]
    out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(rel["kl"]):.1f}" y2="{sy(rel["kl"]):.1f}" class="bar-line"/>')
    out.append(text(sx(13.0), sy(rel["kl"]) - 7, "release KL (the bar)", "t-small", "middle"))
    out.append('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="c-ink"/></marker></defs>')
    for a, bpt in zip(P, P[1:]):
        if (a["gb"], a["kl"]) == (bpt["gb"], bpt["kl"]):
            continue
        xa, ya, xb, yb = sx(a["gb"]), sy(a["kl"]), sx(bpt["gb"]), sy(bpt["kl"])
        d = math.hypot(xb - xa, yb - ya) or 1
        sh = 11 / d
        out.append(f'<line x1="{xa + (xb - xa) * sh:.1f}" y1="{ya + (yb - ya) * sh:.1f}" x2="{xb - (xb - xa) * sh:.1f}" '
                   f'y2="{yb - (yb - ya) * sh:.1f}" class="path" marker-end="url(#arr)"/>')
    # numbered points; points at the same spot share a number; labels go in an in-plot legend
    nums, seen = [], {}
    for p in P:
        key = (p["gb"], round(p["kl"], 3))
        if key not in seen:
            seen[key] = len(seen) + 1
        nums.append(seen[key])
    for i, p in enumerate(P):
        if i and nums[i] == nums[i - 1]:
            continue
        c = "c-gray" if i == 0 else ("c-acc3" if nums[i] == nums[-1] else "c-acc2")
        cx, cy = sx(p["gb"]), sy(p["kl"])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" class="{c}"/>')
        out.append(text(cx, cy + 4, str(nums[i]), "t-num", "middle"))
    labels = {}
    for i, p in enumerate(P):
        labels.setdefault(nums[i], []).append(p["label"])
    lx, ly = sx(11.45), sy(0.0545)
    lw = 9 + 6.4 * max(len(f"{k}  " + "; ".join(v)) for k, v in labels.items())
    out.append(f'<rect x="{lx - 10:.1f}" y="{ly - 16:.1f}" width="{lw:.1f}" height="{18 * len(labels) + 10}" rx="6" class="legend"/>')
    for k in sorted(labels):
        out.append(text(lx, ly, f"{k}  " + "; ".join(labels[k]), "t-small"))
        ly += 18
    out.append("</svg>")
    cap = ("<b>Figure 3.</b> Engine KL<sub>64</sub> against an FP32 reference, plotted against the target's weight "
           "bytes per round (computed from the bit allocation; the drafter and LM heads are excluded). GPTQ buys "
           "quality at the release's own size, and each later step spends some of it on bytes. The final model reads "
           f"{100 * (1 - P[-1]['gb'] / P[0]['gb']):.0f}% fewer target bytes than the release at lower KL. Top-1 "
           "agreement, not KL, ended up being the binding constraint.")
    return figure("quality", "".join(out), cap)


# ---------------------------------------------------------------- Figure: L2 fill schematic
def fig_fill():
    W, H = 760, 222
    out = [svg_open(W, H, "L2 fill schematic")]
    lx, x0 = 140, 150
    # timeline scale: GEMM A 0-200, small kernel 200-260, GEMM B 260-460 (no fill) / 260-430 (fill)
    sc = 1.15
    X = lambda v: x0 + v * sc
    def lane(y, label, sub):
        out.append(text(lx, y + 16, label, "t-label", "end"))
        out.append(text(lx, y + 31, sub, "t-small", "end"))
    # Without fill
    out.append(text(x0, 16, "Without fill", "t-head"))
    lane(26, "kernels", "on the SMs")
    out.append(f'<rect x="{X(0)}" y="26" width="{200 * sc}" height="26" class="c-acc3"/>')
    out.append(text(X(100), 44, "GEMM A", "t-in", "middle"))
    out.append(f'<rect x="{X(200)}" y="26" width="{60 * sc}" height="26" class="c-warm"/>')
    out.append(text(X(230), 44, "norm", "t-in", "middle"))
    out.append(f'<rect x="{X(260)}" y="26" width="{200 * sc}" height="26" class="c-acc3"/>')
    out.append(text(X(360), 44, "GEMM B (starts cold)", "t-in", "middle"))
    lane(60, "DRAM", "weight stream")
    out.append(f'<rect x="{X(0)}" y="62" width="{200 * sc}" height="14" class="c-acc3 op"/>')
    out.append(f'<rect x="{X(200)}" y="62" width="{60 * sc}" height="14" class="c-idle"/>')
    out.append(text(X(230), 92, "DRAM idle", "t-small", "middle"))
    out.append(f'<rect x="{X(260)}" y="62" width="{200 * sc}" height="14" class="c-acc3 op"/>')
    # With fill
    y2 = 128
    out.append(text(x0, y2 - 10, "With fill", "t-head"))
    lane(y2, "kernels", "on the SMs")
    out.append(f'<rect x="{X(0)}" y="{y2}" width="{200 * sc}" height="26" class="c-acc3"/>')
    out.append(text(X(100), y2 + 18, "GEMM A", "t-in", "middle"))
    out.append(f'<rect x="{X(200)}" y="{y2}" width="{62 * sc}" height="26" class="c-warm"/>')
    out.append(text(X(231), y2 + 18, "norm", "t-in", "middle"))
    out.append(f'<rect x="{X(262)}" y="{y2}" width="{172 * sc}" height="26" class="c-acc3"/>')
    out.append(text(X(348), y2 + 18, "GEMM B (head already in L2)", "t-in", "middle"))
    lane(y2 + 34, "DRAM", "weight stream")
    out.append(f'<rect x="{X(0)}" y="{y2 + 36}" width="{200 * sc}" height="14" class="c-acc3 op"/>')
    out.append(f'<rect x="{X(200)}" y="{y2 + 36}" width="{62 * sc}" height="14" class="c-fill"/>')
    out.append(f'<rect x="{X(262)}" y="{y2 + 36}" width="{172 * sc}" height="14" class="c-acc3 op"/>')
    out.append(text(X(231), y2 + 66, "extra CTAs of the norm load B's first MBs into L2", "t-small", "middle"))
    out.append(f'<line x1="{X(434)}" x2="{X(460)}" y1="{y2 + 13}" y2="{y2 + 13}" class="path" marker-end="url(#arr2)"/>')
    out.append(f'<line x1="{X(460)}" x2="{X(434)}" y1="{y2 + 13}" y2="{y2 + 13}" class="path" marker-end="url(#arr2)"/>')
    out.append(text(X(447), y2 - 4, "saved", "t-small", "middle"))
    out.append('<defs><marker id="arr2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
               'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="c-ink"/></marker></defs>')
    out.append("</svg>")
    cap = ("<b>Figure 4.</b> L2 fills, schematic and not to scale. A latency-bound kernel such as an RMSNorm "
           "leaves DRAM idle for a few microseconds. Extra CTAs in the same launch, issued after the kernel's own "
           "loads, read the first megabytes of the next GEMM's weights into L2 (the 4090 has 72 MB). The GEMM then "
           "starts warm. Results are bit-identical, because the fill only loads data.")
    return figure("fill", "".join(out), cap)


# ---------------------------------------------------------------- Figure: progress
def fig_progress():
    P = D["progress"]
    W, H, l, r, t, b = 760, 300, 56, 20, 18, 58
    y0, y1 = 230, 330
    n = len(P) + (1 if lad["final"]["tps"] > P[-1][1] + 0.05 else 0)
    pts = list(P) + ([["10-06", lad["final"]["tps"], "kernel", "final (balanced A/B)"]] if n > len(P) else [])
    sx = lambda i: l + (W - l - r) * (i + 0.5) / len(pts)
    sy = lambda v: t + (H - t - b) * (1 - (v - y0) / (y1 - y0))
    out = [svg_open(W, H, "progress over four days")]
    # day bands
    days = []
    for i, p in enumerate(pts):
        if not days or days[-1][0] != p[0]:
            days.append([p[0], i, i])
        days[-1][2] = i
    for j, (dname, a, bb) in enumerate(days):
        xa, xb = sx(a) - (W - l - r) / len(pts) / 2, sx(bb) + (W - l - r) / len(pts) / 2
        if j % 2 == 0:
            out.append(f'<rect x="{xa:.1f}" y="{t}" width="{xb - xa:.1f}" height="{H - t - b}" class="band"/>')
        out.append(text((xa + xb) / 2, H - b + 18, "Oct " + dname.split("-")[1].lstrip("0"), "t-axis", "middle"))
    for v in range(y0, y1 + 1, 20):
        out.append(f'<line x1="{l}" x2="{W - r}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="grid"/>')
        out.append(text(l - 8, sy(v) + 4, str(v), "t-axis", "end"))
    out.append(text(14, t + (H - t - b) / 2, "Spec-Bench tok/s", "t-axis", "middle",
                    f'transform="rotate(-90 14 {t + (H - t - b) / 2:.1f})"'))
    path = []
    for i, p in enumerate(pts):
        if i:
            path.append(f"L{sx(i):.1f},{sy(pts[i - 1][1]):.1f}")
        path.append(f"{'M' if i == 0 else 'L'}{sx(i):.1f},{sy(p[1]):.1f}")
    out.append(f'<path d="{" ".join(path)}" class="step"/>')
    kc = {"engine": "c-gray", "quant": "c-acc3", "drafter": "c-warm", "kernel": "c-acc2"}
    for i, p in enumerate(pts):
        out.append(f'<circle cx="{sx(i):.1f}" cy="{sy(p[1]):.1f}" r="5" class="{kc[p[2]]}"><title>{esc(p[3])}: {p[1]} tok/s</title></circle>')
    for i in (0, len(pts) - 1):
        out.append(text(sx(i) + (2 if i == 0 else -10), sy(pts[i][1]) - 10,
                        f"{pts[i][1]:.1f}", "t-value", "start" if i == 0 else "end"))
    lx, ly = l + 10, H - 14
    for k, name in (("quant", "quantization"), ("drafter", "drafter and tree"), ("kernel", "kernels and engine")):
        out.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" class="{kc[k]}"/>')
        out.append(text(lx + 14, ly, name, "t-small"))
        lx += 34 + 7 * len(name)
    out.append("</svg>")
    cap = ("<b>Figure 5.</b> Every adopted change, in order, from the release weights on our engine branch to the "
           "final configuration. Each point is one Spec-Bench-480 run made when the change was adopted; drift "
           "between sessions is about ±0.5%, so the last point (322.9) and the same-session value in Figure 1 "
           f"({lad['final']['tps']:.1f}) differ by that much. Hover over a point for the change it adds.")
    return figure("progress", "".join(out), cap)


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


WL = [("json", "JSON (77-token prompt)"), ("code", "code (69)"), ("prose", "prose (34)"),
      ("sustained_decode", "1,024-token answer"), ("short_refactor", "2K-token prompt"),
      ("medium_feature", "11K-token prompt"), ("long_review", "32K-token prompt"), ("context_copy", "26K-token copy task")]


def table_suite():
    S, O = D["suite"], D["suite_0310"]
    cols = [("llama.cpp DFlash2 recipe*", O["llama"]), ("vLLM DFlash2 recipe*", O["vllm"])]
    if not S.get("pending"):
        cols += [("stock Cinference", S["stock"]), ("our engine, release weights", S["rel_def"]), ("ours", S["final"])]
    head = "".join(f"<th>{esc(c)}</th>" for c, _ in cols)
    body = ""
    for k, name in WL:
        cells = "".join(f"<td>{'<b>' if c == 'ours' else ''}{d.get(k, float('nan')):.1f}{'</b>' if c == 'ours' else ''}</td>" for c, d in cols)
        body += f"<tr><td>{esc(name)}</td>{cells}</tr>"
    return (f'<div class="tbl"><table><thead><tr><th>workload</th>{head}</tr></thead><tbody>{body}</tbody></table>'
            '<p class="tnote">Decode tok/s as seen by a streaming client, median of 3, batch 1, greedy. '
            '* Measured 2026-10-03 with the same client on the same box; the others in the same session as Figure 1. '
            'Ours ran with a 49,152-token KV cache (see Limitations), the others with 65,536; rerunning our engine with '
            'release weights at 49,152 reproduced its column within 0.1 tok/s. '
            'Single prompts, so one near-tie can move a cell by several percent.</p></div>')


CSS = r"""
:root{--bg:#fdfdfb;--fg:#1d1d1f;--mut:#5f6368;--rule:#e3e3df;--card:#f4f4f0;--acc1:#bcd3ea;--acc2:#6c9bd1;--acc3:#1f5aa6;
--warm:#e0913a;--fill:#f2c38a;--idle:#d6d6d0;--gray:#a5a5a0;--ink:#333;--band:#f1f1ec;--link:#1f5aa6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141518;--fg:#e8e8e6;--mut:#a0a3a8;--rule:#2c2e33;
--card:#1c1e22;--acc1:#2f4a6b;--acc2:#4f7fbf;--acc3:#86b4ee;--warm:#e8a35a;--fill:#8a6234;--idle:#3a3c40;--gray:#6d6f73;
--ink:#ddd;--band:#1b1d21;--link:#8ab8f0}}
:root[data-theme="dark"]{--bg:#141518;--fg:#e8e8e6;--mut:#a0a3a8;--rule:#2c2e33;--card:#1c1e22;--acc1:#2f4a6b;--acc2:#4f7fbf;
--acc3:#86b4ee;--warm:#e8a35a;--fill:#8a6234;--idle:#3a3c40;--gray:#6d6f73;--ink:#ddd;--band:#1b1d21;--link:#8ab8f0}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.65 Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif}
main{max-width:720px;margin:0 auto;padding:48px 16px 96px}
h1,h2,h3,.kicker,.byline,.links,figcaption,.tnote,table,.tldr h2,nav{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",
"Helvetica Neue",Arial,sans-serif}
h1{font-size:2.2rem;line-height:1.18;letter-spacing:-.01em;margin:.2em 0 .4em}
h2{font-size:1.35rem;margin:2.4em 0 .6em;letter-spacing:-.005em}
.kicker{text-transform:uppercase;letter-spacing:.08em;font-size:.75rem;color:var(--mut);margin:0}
.dek{font-size:1.15rem;color:var(--mut);margin:.2em 0 1em}
.byline{font-size:.9rem;color:var(--mut);margin:0}.byline a{color:inherit;text-decoration:none;border-bottom:1px solid var(--rule)}
.links{margin:.8em 0 0;display:flex;gap:10px;flex-wrap:wrap}
.links a{font-size:.85rem;border:1px solid var(--rule);border-radius:999px;padding:3px 12px;text-decoration:none;color:var(--fg)}
a{color:var(--link)}
.tldr{background:var(--card);border-radius:10px;padding:14px 22px 6px;margin:28px 0}
.tldr h2{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin:.4em 0 .2em}
.tldr li{margin:.45em 0}
ul,ol{padding-left:1.3em}
li{margin:.3em 0}
li>ul{margin:.2em 0}
.steps>li{margin:.7em 0}
.eq{text-align:center;font-size:1.05rem;margin:1.1em 0}
code{white-space:nowrap;font:.86em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--card);padding:1px 4px;border-radius:4px}
pre{font:.8rem/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--card);padding:12px 14px;border-radius:8px;
overflow-x:auto;white-space:pre-wrap;word-break:break-word}
pre.cmd{white-space:pre}
figure{margin:30px 0 34px}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:.84rem;line-height:1.5;color:var(--mut);margin-top:8px}
@media (min-width:1000px){figure.wide{margin-left:-80px;margin-right:-80px}}
.tbl{overflow-x:auto;margin:18px 0 26px}
table{border-collapse:collapse;font-size:.84rem;width:100%;line-height:1.4}
th,td{padding:6px 8px;border-bottom:1px solid var(--rule);text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}
table.neg td,table.neg th{text-align:left}
thead th{font-weight:600;border-bottom:1.5px solid var(--fg)}
.tnote{font-size:.8rem;color:var(--mut);margin:.5em 0 0}
.note{font-size:.92rem;color:var(--mut)}
svg text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif;fill:var(--fg)}
.t-label{font-size:14px}.t-small{font-size:12px;fill:var(--mut)}.t-value{font-size:14px;font-weight:600}
.t-axis{font-size:12px;fill:var(--mut)}.t-in{font-size:12.5px;fill:#fff;font-weight:500}.t-head{font-size:13px;font-weight:600}.t-num{font-size:11px;font-weight:700;fill:#fff}
.c-acc1{fill:var(--acc1)}.c-acc2{fill:var(--acc2)}.c-acc3{fill:var(--acc3)}.c-warm{fill:var(--warm)}.c-gray{fill:var(--gray)}
.c-idle{fill:var(--idle)}.c-fill{fill:var(--fill)}.c-ink{fill:var(--ink)}.op{opacity:.55}
.grid{stroke:var(--rule);stroke-width:1}.legend{fill:var(--bg);stroke:var(--rule)}.band{fill:var(--band)}
.bar-line{stroke:var(--gray);stroke-width:1.5;stroke-dasharray:5 4}
.path{stroke:var(--ink);stroke-width:1.3;fill:none}
.step{stroke:var(--mut);stroke-width:1.6;fill:none}
@media (max-width:560px){body{font-size:16px}h1{font-size:1.7rem}.t-label{font-size:16px}.t-small,.t-axis{font-size:14px}}
"""


def build():
    src = open("article.html").read()
    figs = {"ladder": fig_ladder, "anatomy": fig_anatomy, "quality": fig_quality, "fill": fig_fill, "progress": fig_progress}
    tabs = {"quality": table_quality, "negative": table_negative, "suite": table_suite}
    src = re.sub(r"\{\{fig:(\w+)\}\}", lambda m: figs[m.group(1)](), src)
    src = re.sub(r"\{\{table:(\w+)\}\}", lambda m: tabs[m.group(1)](), src)
    src = re.sub(r"\{\{v:(\w+)\}\}", lambda m: esc(V[m.group(1)]), src)
    left = re.findall(r"\{\{[^}]+\}\}", src)
    assert not left, left
    title = re.search(r"<h1>(.*?)</h1>", src, re.S).group(1)
    desc = ("How Qwen3.8-27B reaches " + V["final_tps0"] + " tok/s on Spec-Bench on a single RTX 4090 with speculative "
            "decoding: fewer bytes per round, more tokens per round, and DRAM kept busy, at better-than-release quality.")
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><meta name="description" content="{esc(desc)}">'
            f'<style>{CSS}</style></head><body><main>{src}</main></body></html>\n')
    open(OUT, "w").write(page)
    print(f"wrote {OUT}: {len(page) / 1024:.0f} KB")


build()
