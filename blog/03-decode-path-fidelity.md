# Your fidelity test never runs your decode kernels

*Draft, 2026-10-05. Third post of the single-RTX-4090 Qwen3.8-27B series. Tools: `code/fidelity/dfid.py`,
`ref_var.py` and `code/engine-patches/patch_hidden_dump.py`.*

The usual way to check a quantized or optimized LLM engine is to score a text corpus through it and compare its
next-token distributions with a high-precision reference: KL, top-1 agreement and perplexity. We do the same, against an
FP32-compute reference, on 46,059 predictions.

That check runs the **prefill** path: long chunks, big GEMMs, one pass. A speculative decoder spends its life in a
different path. Every round, a 16-token verify forward reads the recurrent state written by the previous round and
writes it back. Some optimizations touch only that path:
- storing the Gated DeltaNet state in FP16 between rounds;
- folding a norm into a GEMM epilogue on verify widths;
- a different kernel route at T = 16.

The prefill tool cannot see any of them.

## Measuring the verify path directly

The idea: run the real decode loop on fixed text, and capture what the target computes for each token it accepts.

1. **Fixed text.** Forced decoding (`NINFER_FORCE_TOKENS`, from the previous post) makes every configuration walk the
   same reference continuation of each Spec-Bench prompt. The verify forward still runs normally.
2. **Dump the head inputs.** `NINFER_HIDDEN_DUMP` writes two things:
   - the prompt ids when each request starts;
   - after every verify round, the final-normed hidden state of each accepted column, together with the round's
     frontier.

   Column *c* of a round predicts the token at frontier + 1 + *c*. Graph capture is off for these runs, because host
   code inside a captured round runs only once.
3. **Same arithmetic as the reference.** Offline, hidden × the checkpoint's BF16 head gives FP32 logits, exactly as the
   reference computes them. Engine head quantization drops out, so only the backbone's decode numerics are measured.
4. **Reference on the same text.** prompt + continuation goes through the BF16 checkpoint, layer-streamed on one 24 GB
   card (`ref_var.py`, variable-length windows): 241 windows in 14 minutes.

Before trusting it, check the alignment. 236 of 240 dumped columns reproduce the forced next token under the BF16 head,
and the 4 misses are near-ties. Every column is present, and the first round sits at frontier = prompt length.

## FP32 vs FP16 recurrent state, on the decode path

| state storage | columns | KL64 | top-1 | PPL |
|---|---|---|---|---|
| FP32 | 40,889 | 0.02125 | 0.9588 | 1.1942 |
| FP16 | 40,634 | 0.02079 | 0.9586 | 1.1930 |

- Per Spec-Bench group, on identical columns, KL moves by ±0.5% in both directions, and PPL agrees to the fourth digit.
- Top-1 on the common columns: FP16 loses 79 agreements and gains 56, a net −23 (sign test p ≈ 0.06).

## The near-tie tax

Is −23 a real cost of FP16? The prefill tool's probes answer it. There, the carried state was rounded to fewer
mantissa bits at every 64-token boundary:

| state rounding | net top-1 vs FP32 (of 46,059) |
|---|---|
| 10 mantissa bits | −9 |
| 8 bits | −11 |
| 7 bits | −11 |
| 5 bits | −14 |
| exact FP16 | −22 |
| FP16 storage | −24 |

Every perturbation is net negative by a similar amount, and the size does not track the precision: 5 bits costs less
than FP16. The explanation is selection, not damage.
- Near a tie, the engine's argmax agrees with the reference more often than chance, because the two computations are
  close.
- Any added noise re-randomizes some of those near-ties.
- On average, that turns agreements into disagreements.

So **every non-bit-exact change pays ≈10–25 top-1 agreements per 40–46k predictions, with no KL or PPL signal.**

That matters for quality bars. A rule like "top-1 must not drop below the baseline" is biased against every change,
however harmless. The bar we now use:
- KL and PPL not worse than the reference release;
- top-1 above the release's.

Within that bar, the top-1 headroom is a budget that each non-bit-exact change spends. FP16 GDN state spent nearly
all of ours: 42,606 against the release's 42,605. It buys +0.7–0.9% tok/s.

## Checklist

- If an optimization only runs at decode widths, prefill fidelity says nothing about it. Dump the verify path on forced
  text and score that.
- Remove what is common to all configurations (the head) and compute logits exactly like the reference.
- Before the expensive reference, validate the alignment with argmax against the forced token.
- Read top-1 flips as gained/lost counts with a sign test. Expect a small negative net for any perturbation.
