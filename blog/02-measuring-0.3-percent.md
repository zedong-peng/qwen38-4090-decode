# Measuring 0.3%: forced-text A/Bs for speculative decoding

*Draft, 2026-10-05. Second post of the single-RTX-4090 Qwen3.8-27B series. The tools are in `code/specbench` and
`code/engine-patches`.*

Late in an optimization project most wins are 0.3–1% of round time. Speculative decoding makes those hard to see:
- The output depends on numerics at near-ties.
- Acceptance depends on the output.
- tok/s is acceptance divided by round time.

This post covers what went wrong in our A/Bs and the protocol that fixed it.

## Unforced A/Bs measure content, not speed

Spec-Bench-480 is 480 prompts × 256 greedy tokens. Any kernel change that is not bit-exact flips a few near-tie
tokens. From the first flip on, that prompt generates different text, and different text has different acceptance:
summarizing a list drafts better than free prose.

Three ways this misled us:

1. **Identical-output subsets are biased.** It is tempting to compare only the prompts whose outputs did not change.
   Those prompts are the confident ones: they average 6.8 tokens per step against 5.4 overall.
2. **Every comparison shares one content draw.** All candidates are compared against the same baseline outputs, so
   the baseline's luck biases all of them in the same direction. A drafter perturbation we knew to be null (conv
   projections on an 8-bit grid, numerically almost exact) read **+0.44% tok/step**, and +0.67% on its differing
   prompts.
3. **Real effects hide inside that band.** Moving the drafter MLP to Q3 costs 0.22% acceptance. The unforced A/B could
   not tell that apart from the content draw.

Round time (ms/round) is robust either way: a round costs the same whatever the text. Acceptance is not.

## Forced text

The engine's greedy tree walk normally compares each draft node against the target's argmax. With
`NINFER_FORCE_TOKENS=REF.jsonl`, the walk takes the reference id for that position instead. The reference is the token
dump of a baseline run.
- The target still computes every column, so the round costs exactly what it did.
- The only extra work is two reads of a mapped host buffer per round, which measured +0.07% [−0.03, +0.16].
- Every configuration now walks the same 480 texts. tok/step and ms/round are compared on identical tokens, for
  drafter changes and target changes alike.

**Validate the instrument first.** Forcing a configuration onto its own reference must reproduce its outputs exactly.
Our first version gave **0/482** identical outputs and −3.2% tok/step:
- At the first round the frontier is P, not P + 1, so the root re-emitted the prefill token.
- Everything after that was the reference shifted by one.

With the base fixed, the outputs were 482/482 identical with equal tok/step. Without that check, the bug would have
looked like a plausible −3% drafter regression.

What forced text revealed:
- **The tree absorbs about two-thirds of any change in chain acceptance.**
  - Drafter MLP Q3 costs −0.77% teacher-forced on a chain, but only −0.22% engine tok/step on forced text.
  - The fine-tuned drafter shows the same ratio: +6.8% chain, +1.4% engine.
- **Acceptance resolves to about ±0.35% (95%).** That comes from per-prompt spread on identical text, where the
  unforced run could not separate effects from content draws.

## A B A B, and read the drift

Clocks, thermals and background load drift during a 30-minute job. So we run four full Spec-Bench runs on one GPU:
1. A, B, A, B.
2. Report A→B and the second A→B separately.
3. Read A→A and B→B as the drift.

| change (forced, 4 × 480 prompts) | A→B | second A→B | A→A drift |
|---|---|---|---|
| GDN conv fused into the GEMM epilogue | ms/round −0.40% [−0.42, −0.39] | −0.38% [−0.40, −0.37] | −0.05% |
| GDN state stored in FP16 | tok/s +0.90% [+0.72, +1.07] | +0.84% [+0.67, +1.01] | ms/round −0.32% |
| RMSNorms folded into GEMM epilogues | ms/round **+1.52%** [+1.48, +1.56] | +1.57% [+1.53, +1.61] | −0.14% |
| null: forced repeat of the same config | ms/round +0.20% [+0.18, +0.23] | | |

- **Drift is the same size as the effects.** A→A ranges from 0.05% to 0.32%. A single A→B pair would have misread
  the FP16-state change by a third.
- **The intervals are tight because the bootstrap is paired.** `sb_paired.py` resamples prompts and compares the same
  prompt across runs, so prompt difficulty cancels. On the same runs, unpaired ms/round intervals are 5–13× wider:
  0.40% against 0.03–0.09%.

## Box discipline

Each of these rules cost a voided run before it was a rule.

- **No builds during timing runs, not even niced ones.** An 8-job `nice 19` rebuild made an identical configuration
  read **0.93% slower**.
- **Keep the other GPU's load constant.** A run starts only when the sibling card is idle or busy with one fixed job;
  the run log records its utilization and clock.
- **Clocks under a power cap.** At the 450 W cap the SM clock sits at 2490–2535 MHz instead of 2715, while the memory
  clock stays fixed. The GEMMs are memory-bound, so that does not move them, but it is logged anyway.

## Explain with per-call kernel times

An end-to-end A/B says *whether*; nsys says *why*. We compare per-call **median** kernel times grouped by
(kernel, grid). They do not depend on how many rounds a run had.

Example: the RMSNorm fold above was expected to save ≈0.9% and lost 1.5%. Per-call medians showed why:
- The 80 removed norm launches saved 139 µs per round.
- The producer GEMM grew from 20.0 to 25.5 µs per call, 64 times per round.
- The likely cause, not yet confirmed by a microbenchmark: its new epilogue sends 5,120 64-bit atomics per call into
  one 128-byte L2 line.

**Do not price host time from nsys.** Under tracing, `cudaGraphLaunch` blocks the host for 467 µs. The engine's own
unprofiled request log shows 132 µs of exposed host time per round.

## Checklist

1. Make the text fixed: forced tokens from a reference run. Check that a self-forced run reproduces 100% of outputs.
2. Run A B A B on one GPU, with no builds and constant sibling load.
3. Bootstrap paired per prompt on ms/round, tok/step and tok/s. Report both pairs and the drift.
4. Explain with per-call kernel medians. Take host costs from unprofiled logs.
5. Judge quality separately, against an FP32 reference. Forced text says nothing about quality.
