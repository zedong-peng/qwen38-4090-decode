# Qwen3.8-27B speculative decoding on one RTX 4090

**Write-up:** [zedongpeng.com/assets/html/qwen38-4090-decode.html](https://zedongpeng.com/assets/html/qwen38-4090-decode.html) · **Weights:** [zedongpeng/Qwen3.8-27B-NInfer-4090](https://huggingface.co/zedongpeng/Qwen3.8-27B-NInfer-4090)

On Spec-Bench (batch 1, greedy) on one RTX 4090: 245.2 tok/s on the official release weights and 322.3 tok/s with
re-quantized weights that are closer to full precision than that release. vLLM's community recipe for this card
decodes 200.3 tok/s on the same benchmark and llama.cpp's 102.2 tok/s.

| path | contents |
|---|---|
| `site/` | The write-up as one self-contained HTML page (`index.html`, works offline) with eight interactive figures. `build.py` inlines `article.html`, `style.css`, `js/*.js` and the data in `data.json` and `data/` (recorded decoding streams, verify-tree dumps, nsys round timelines, the engine comparison). |
| `code/engine-series/` | 53 patches on the Cinference `ada-dflash2` tree. Every research feature sits behind an environment flag that is off by default. |
| `code/specbench/`, `code/fidelity/`, `code/profiling/`, `code/microbench/` | Forced-text and drift-balanced A/Bs, prefill and decode-path fidelity against an FP32 reference, nsys round accounting, and kernel microbenchmarks. See `code/README.md`. |
| `code/site-data/`, `code/release/` | Scripts that turn raw runs into the page's data; the Hugging Face model card and the container-provenance sanitizer. |

Rebuild the page: `cd site && python3 build.py`.

Code is Apache-2.0 (see `code/LICENSE` and `code/NOTICE`), following the upstream NInfer / Cinference license.
