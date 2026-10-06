# Qwen3.8-27B speculative decoding on one RTX 4090

The write-up, engine patches and measurement tools from a four-day speed study: Qwen3.8-27B with a DFlash2 drafter,
batch 1, greedy, on a single RTX 4090.

| path | contents |
|---|---|
| `site/` | The final write-up as one self-contained HTML page (`index.html`, no network needed). It is built from `article.html` and `data.json` by `build.py`, which draws the figures as inline SVG. |
| `code/engine-series/` | 53 patches on the Cinference `ada-dflash2` tree. Every research feature sits behind an environment flag that is off by default. |
| `code/specbench/`, `code/fidelity/`, `code/profiling/`, `code/microbench/` | Forced-text and drift-balanced A/Bs, prefill and decode-path fidelity against an FP32 reference, nsys round accounting, and kernel microbenchmarks. See `code/README.md`. |
| `blog/` | Earlier drafts (posts 1–5), kept for reference. The site page supersedes them. |

Rebuild the page: `cd site && python3 build.py`.

Code is Apache-2.0 (see `code/LICENSE` and `code/NOTICE`), following the upstream NInfer / Cinference license.
