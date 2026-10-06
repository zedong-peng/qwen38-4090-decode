# Qwen3.8-27B speculative decoding on one RTX 4090

**Write-up:** [zedong-peng.github.io/assets/html/qwen38-4090-decode.html](https://zedong-peng.github.io/assets/html/qwen38-4090-decode.html) · **Weights:** [zedongpeng/Qwen3.8-27B-NInfer-4090](https://huggingface.co/zedongpeng/Qwen3.8-27B-NInfer-4090)

245 tok/s on the official weights (1.38× the stock engine) and 322 tok/s with re-quantized weights that are closer to
full precision than the official release (1.81×), on Spec-Bench, batch 1, greedy, one RTX 4090.

The write-up, engine patches and measurement tools from a four-day speed study: Qwen3.8-27B with a DFlash2 drafter,
batch 1, greedy, on a single RTX 4090.

| path | contents |
|---|---|
| `site/` | The final write-up as one self-contained HTML page (`index.html`, no network needed) with ten interactive figures. `build.py` inlines `article.html` (prose), `style.css`, `js/*.js` (vanilla-JS SVG figures) and the data: `data.json` plus `data/` (recorded decoding streams with per-round timing, verify-tree dumps, nsys round timelines, Spec-Bench group statistics). |
| `code/engine-series/` | 53 patches on the Cinference `ada-dflash2` tree. Every research feature sits behind an environment flag that is off by default. |
| `code/specbench/`, `code/fidelity/`, `code/profiling/`, `code/microbench/` | Forced-text and drift-balanced A/Bs, prefill and decode-path fidelity against an FP32 reference, nsys round accounting, and kernel microbenchmarks. See `code/README.md`. |
| `blog/` | Earlier drafts (posts 1–5), kept for reference. The site page supersedes them. |

Rebuild the page: `cd site && python3 build.py`.

Code is Apache-2.0 (see `code/LICENSE` and `code/NOTICE`), following the upstream NInfer / Cinference license.
