# Notes for whoever picks this up

The write-up explains what was built and why it works. REPRODUCE.md and code/weights/README.md say how to rebuild the
engine and the weights. This page covers the rest: how the project got here, the lessons that cost the most to learn,
and where to start next.

## Timeline

Five days on one machine (4× RTX 4090, two of them ours), Oct 2–6, 2026. An AI coding agent ran the experiments and
kept a research log of about 3,800 lines; the milestones below come from it. Within-day numbers come from different
sessions, so treat differences under 1% between days as noise.

| day | what happened | Spec-Bench-480, tok/s |
|---|---|---|
| Oct 2 | Baselines on the card: llama.cpp, vLLM (community recipe), Cinference-4090 on the official container. | 178 (Cinference-4090, measured Oct 6) |
| Oct 3 | Upstream Cinference merged into the Ada port (verify trees, DFlash2 passes); Ada small-T GEMV kernels; two-level LM head. Fidelity protocol: FP32 reference, KL64 on prose and code. GPTQ beats RTN at the same bits; all-Q4 with c96 calibration; drafter fine-tune. | 292 |
| Oct 4 | 3-bit MLP via load-time planes ("Q3 shadow", no new file format); GPTQ Q4 drafter; end-to-end scale tuning pays back the 3-bit cost; c384 calibration; forced-text A/Bs. | 299 → 309 |
| Oct 5 | E2E v5 (16 more 3-bit layers); INT8 V; proposal-head screen; L2 fills; FP16 GDN state; decode-path fidelity tool. | 310 → 319 |
| Oct 6 | Whole-program CUDA build; last fill sizes; converged. Same-session comparison with vLLM and llama.cpp. | 322 |

## Lessons

**Measuring**
- At this level most changes are worth 0.3–1%, below the run-to-run drift of an unforced Spec-Bench run. Forced text
  (every configuration walks the same reference tokens), drift-balanced run orders and paired per-prompt intervals
  made them measurable. Part III of the write-up and code/README.md have the protocol.
- A run that starts on a cool, idle GPU reads fast. Discard one warm-up run.
- Check the instrument before trusting it. Forcing a configuration onto its own outputs must reproduce them; ours first
  reproduced 0 of 482 because of an off-by-one.
- A prefill-path quality scorer never runs the verify forward, so per-round numerics (FP16 state, fused norms) are
  invisible to it. `dfid.py` scores the decode path.
- GPTQ's in-sample KL is about 5× its held-out KL. Anything tuned after GPTQ (E2E) must train on windows GPTQ never saw.
- Simulated KL matched the engine only after the simulation ran the engine's own KV codec (gap 0.002–0.003 → 0.0005).

**Speed**
- At batch 1 a round is DRAM time plus idle gaps. Wins came from reading fewer bytes (3-bit MLP, Q4 drafter, screened
  heads) or from closing gaps (fused kernels, L2 fills). Bytes alone are not time: 3-bit routes that lost bandwidth
  saved nothing.
- On Ada, `prefetch.global.L2` does nothing; real line loads issued during latency-bound kernels do.
- The verify tree absorbs about two-thirds of any change in the drafter's chain acceptance. A drafter improvement
  measured teacher-forced shrinks by about 3× in the engine.
- A noisier target lowers acceptance. Judge every quantization step on round time and tokens per round together.

**Operating the box**
- Never overwrite a running shell script in place (scp onto the same file). The shell continues at its old offset in
  the new text. Write a temp file and `mv`.
- `pkill -f PATTERN` over ssh also matches the ssh command itself. Anchor the pattern.
- A remote retry loop survives a local ssh timeout. Hash-check every download.
- One job queue per GPU (`gpuq.sh`), no builds during timing runs, and never touch other users' GPUs.
- github.com is unreliable from the server; mirrors (ghfast.top, hf-mirror.com) work for reading.

## Where to start next

- **More 3-bit layers.** MLP gate/up 32–63 and down 16–31 are still Q4. Each block of 16 layers moved to 3 bits bought
  0.8–1.6% once end-to-end tuning recovered the quality, and E2E v5 was still improving at its last step. See
  code/weights/README.md.
- **Prefill on the 3-bit planes.** It would drop the duplicate Q4 copy (about 4 GiB) and allow a 65,536-token context.
- **Long contexts.** At 11K–32K tokens verify attention reads the whole cache and the advantage over the other engines
  shrinks (Limitations in the write-up). Verify attention at long context is the least-tuned path.
- **Sampling.** Only greedy decoding is supported: the two-level head returns no dense logits.
- **The drafter.** Fine-tuning plateaued after one round: more epochs, rank, data and soft labels all landed at the same
  +1.2–1.5%. A different objective, not more data, is the open question.
