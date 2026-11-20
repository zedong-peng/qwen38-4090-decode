# Measurement tools for speculative decoding on one GPU

These are the tools behind the Qwen3.8-27B + DFlash2 speed study in the write-up (../README.md). They answer three
questions:

1. **Is B faster than A?** This covers changes as small as 0.3%, including changes that alter the generated text.
2. **Did the change cost quality?** Measured against the BF16 checkpoint's own distributions, on both the prefill
   path and the decode (verify) path.
3. **Where does a decode round spend its time?** Read from nsys traces.

The Python tools need only numpy (plus torch and transformers for the reference); the Spec-Bench client uses the stdlib
only. The engine patches target the Cinference `ada-dflash2` tree
([NInfer](https://github.com/Neroued/ninfer) → [Cinference](https://github.com/satellitedown/cinference)).

## specbench/: A/B on Spec-Bench

| tool | what it does |
|---|---|
| `specbench.py` | Runs the 480 Spec-Bench first turns against any OpenAI-compatible server. Concurrency 1, greedy, thinking off, one excluded warmup. Writes per-prompt rows: tokens, client-side decode window, verify steps, output hash. |
| `sb_paired.py BASE.json RUN.json` | Paired per-prompt comparison with prompt-bootstrap 95% intervals on ms/round, tokens/step and tok/s. |
| `sb_balanced.py A:r0.json B:r1.json ...` | Drift-balanced A/B over a run sequence such as A B B A B A A B: least-squares fit of ms/round with linear and quadratic drift, reported with a prompt bootstrap and a run-level t interval. |
| `ids_cmp.py`, `ids_prefix.py` | Greedy agreement of two token dumps: identical requests, common prefix, median first divergence. |

Protocol that made 0.3% measurable:
1. **Forced text.** With `NINFER_FORCE_TOKENS=REF.ids.jsonl` (engine patch below), every configuration walks the
   reference's tokens. Round time is then compared on identical text, and near-tie flips no longer change content.
2. **Alternate on one GPU.** Keep the other GPU's load constant and run no builds on the box, not even niced ones.
   Discard one warm-up run (a run that starts on an idle, cool GPU reads fast). Same-config drift is 0.05–0.6% per
   job, so prefer A B B A B A A B and `sb_balanced.py`, which removes linear and quadratic drift.
3. **`sb_paired.py` for the intervals.** Use nsys per-call medians (`profiling/kern_grep.py`) to explain the result.

## fidelity/: quality against the unquantized checkpoint

| tool | what it does |
|---|---|
| `ref_bf16.py MODEL_DIR WINDOWS.bin OUT` | Reference distributions: the BF16 checkpoint layer-streamed through the transformers `qwen3_5` modules on one 24 GB GPU, BF16 activations and FP32 logits. Writes the top-64 per scored column. |
| `ref_var.py` | The same for windows of different lengths. |
| `fidelity.py WINDOWS.bin REF CAND [report.json]` | KL64 (KL(ref‖cand) on the reference's top-64 support, with the candidate's exact logits), top-1 agreement and PPL, per domain. |
| `top1_flips.py REF BASE CAND...` | Top-1 agreements gained and lost against a base. Shows when a top-1 change is noise. |
| `dfid.py` | **Decode-path fidelity.** Scores a forced run's verify-path head inputs (`NINFER_HIDDEN_DUMP`) against `ref_var.py` on the same text. |

The prefill numbers in the write-up use a third reference, `../weights/fq_eval.py --mode ref`: the same layer
streaming with FP32 activations, in the same record format. `ref_var.py` is the reference of the decode-path numbers.

Why `dfid.py` exists: a prefill-based scorer never runs the verify forward. Per-round numerics are invisible to it,
for example GDN state stored in FP16, or a norm folded into a GEMM epilogue.

## profiling/: nsys round analysis

Each tool but `res_cmp.py` takes the sqlite export of an nsys trace. Steady rounds are delimited by a kernel that runs once per decode
round; set it with `ROUND_MARKER` (default `recurrent_fold_staged_kernel`).

| tool | what it does |
|---|---|
| `round_budget.py` | Span, busy time (union of kernels) and idle per round, plus busy time by kernel class. |
| `gap_list.py` | Where the round idles: gaps attributed to the (previous → next) activity pair, plus host runtime calls per round. |
| `round_kernels.py`, `round_diff.py` | Per-kernel calls and µs per round, for one trace or the difference of two. |
| `kern_grep.py SQLITE REGEX` | Per-call medians by kernel and grid. This is the A/B that does not depend on run length. |
| `dram_idle.py` | DRAM-streaming kernels against latency-bound kernels and gaps per round, and how long the non-streaming stretches are. |
| `occ_audit.py` | Registers, shared memory and resident CTAs per SM for every kernel of a round, and the register count that would fit one more CTA. |
| `res_cmp.py A.res B.res` | Per-kernel registers of two builds, from `cuobjdump -res-usage` output instead of a trace (the `-rdc` comparison). |
| `nsys_budget.py SQLITE REQUESTS.jsonl` | Serving traces (`ninfer-serve` with its request log): each request from arrival to its first decode round, split into frontend, planning and restore, prefill passes (tiny and longer, the latter into token cost and excess), and the gaps between passes. |
| `nsys_passkern.py SQLITE LO-HI ...` | Serving traces: GPU time per kernel of the prefill passes in each token-count bin. |
| `nsys_requests.py SQLITE REQUESTS.jsonl [LO HI [SHOW]]` | Serving traces: prefill chunk cost by size, and a full host/GPU layout of a few requests that recompute LO..HI tokens. |

nsys inflates host-side time severalfold, so price host costs from unprofiled request logs instead. Prefill is
GPU-bound: on one SWE-bench task of the agent replay, profiled and unprofiled prefill agreed within 4%.

## engine-patches/: research hooks for Cinference

Idempotent source patchers (`python3 patch_x.py REPO`), all default off. They are patches 0005, 0032 and 0049 of
engine-series/ as scripts, for a Cinference tree without the series (each applies to upstream Cinference at `383e5db`). A tree built from the
series already has them.
- `patch_force_tokens.py`: `NINFER_FORCE_TOKENS=FILE` forces the greedy tree walk onto reference ids, for forced-text
  A/Bs. The reference ids are a `NINFER_TOKEN_DUMP` file, which patch 0015 of the series writes.
- `patch_score_dump.py`: `NINFER_SCORE_DUMP=PREFIX` and `NINFER_SCORE_REF=REF.topk.bin` make the perplexity scorer
  write the top-64 and the at-reference logits.
- `patch_hidden_dump.py`: `NINFER_HIDDEN_DUMP=FILE` writes the prompt ids and the accepted path's head inputs of every
  verify round. Run it with `--no-cuda-graph`.

## weights/: making the re-quantized container

weights/README.md walks through the pipeline behind the Hugging Face container, step by step, with the scripts and the
calibration and training windows.

## engine-series/: the engine commits

53 `git format-patch` commits: the Ada port fixes and the research knobs, each default off, with its measured
effect in the commit message. engine-series/README.md describes the base to apply them on.

## License

Apache-2.0 (see ../LICENSE). The engine patches modify Apache-2.0 code from NInfer and Cinference; see NOTICE.
