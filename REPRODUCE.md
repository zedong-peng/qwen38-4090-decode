# Reproducing the Spec-Bench numbers

Spec-Bench's 480 first turns on one RTX 4090: batch 1, greedy, thinking off, 256 new tokens per prompt, decode rate
measured by the client. All rows were measured in one session (Oct 6, 2026, jobs 131–132 in the research log).

| configuration | weights | tok/s | tokens/round |
|---|---|---|---|
| our engine, `serve.sh official` | official NInfer v3 container | 245.2 | 5.18 |
| our engine, `serve.sh ours` | our container | 322.3 | 5.43 |
| our engine, `serve.sh defaults` (every research knob off) | official NInfer v3 container | 229.8 | 5.19 |
| vLLM 0.27.1, community single-user recipe | AutoRound W4A16 + DFlash2 W4A16 | 200.3 | 4.07 |
| llama.cpp a4d880f | UD-Q4_K_XL + DFlash2 Q4_K_M | 102.2 | 4.09 |

We checked this page end to end on Oct 7, 2026: a fresh clone through `code/reproduce/reproduce.sh` gave 244.3 tok/s
(5.18 tokens/round) for `official` and 321.0 (5.43) for `ours`, and the patched source tree was identical to the one
behind our binary.

Expect a few percent of difference between cards: clocks, power limit and temperature all move the round time.

## The short way

```bash
git clone https://github.com/zedong-peng/qwen38-4090-decode && cd qwen38-4090-decode
bash code/reproduce/reproduce.sh official      # or: ours, defaults
```

The script does steps 1 to 4 below: it builds the engine once, downloads and checks the container once, starts the server
and runs Spec-Bench. Pick the GPU with `CUDA_VISIBLE_DEVICES`. The steps are spelled out below for anyone who wants them
one by one.

## 1. Build the engine

Needs Linux, an RTX 4090 (sm_89), CUDA 12.8, GCC 13, CMake 3.28 or newer, Ninja and the FFmpeg development libraries.

```bash
git clone https://github.com/zedong-peng/qwen38-4090-decode
git clone https://github.com/jram4/ninfer-4090 engine && cd engine
git checkout 70ebb1290dc7abe246c20696a24d21f77faee8d4
git apply --index ../qwen38-4090-decode/code/engine-series/base.diff
git commit -m "Base: Cinference 383e5db merged into jram4 70ebb12, cbf7b7e reverted"
git am ../qwen38-4090-decode/code/engine-series/*.patch
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DNINFER_BUILD_APPS=ON -DNINFER_RDC=OFF
cmake --build build
```

The server is `build/apps/ninfer-serve`. `base.diff` is the merge of Cinference into the Ada port with its conflicts
resolved (code/engine-series/README.md says what it contains); the 53 patches apply on top of it.

## 2. Download the weights

```bash
hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer \
  --revision 1cbd84e7221e51186bd7f093a149912d2489625b --local-dir official
hf download zedongpeng/Qwen3.8-27B-NInfer-4090 qwen3_8_27b.ninfer --local-dir ours
sha256sum official/qwen3_8_27b.ninfer ours/qwen3_8_27b.ninfer
```

| file | bytes | SHA-256 |
|---|---|---|
| `official/qwen3_8_27b.ninfer` | 20,437,521,664 | `81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da` |
| `ours/qwen3_8_27b.ninfer` | 18,186,359,296 | `37cb1507aca20b442a35f4d7833b950e0507767cf766e7140e81fad4faf43d7f` |

## 3. Start the server

```bash
export NINFER_SERVE=$PWD/engine/build/apps/ninfer-serve
bash qwen38-4090-decode/code/reproduce/serve.sh official official/qwen3_8_27b.ninfer 8080   # 245 tok/s
bash qwen38-4090-decode/code/reproduce/serve.sh ours ours/qwen3_8_27b.ninfer 8080           # 322 tok/s
```

The three modes differ only in environment knobs (`code/reproduce/serve.sh` lists them):
- `official` sets the engine knobs: the verify-tree head and its screens, the KV and GDN-state formats, the fused conv
  epilogue and the L2 prefetch sizes.
- `ours` adds the four knobs that are part of the weight work in Part II of the write-up: 8-bit group scales and a Q4
  drafter conv copy made at load, and 3-bit planes for layers whose codes fit them. Only our container has such layers.
- `defaults` turns every research knob off and keeps verify trees.

The server flags are the same in all three: DFlash2 drafts of 15 tokens, verify trees, an 8-bit K / 4-bit V cache,
a 32,768-token context, one sequence.

## 4. Run Spec-Bench

```bash
curl -sLO https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl
python3 qwen38-4090-decode/code/specbench/specbench.py --questions question.jsonl \
  --base-url http://127.0.0.1:8080 --label official --out results --max-tokens 256
```

`question.jsonl` has 480 lines, SHA-256 `4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04`. A run takes
about ten minutes and prints decode tok/s and tokens per round overall and per task group. Keep the GPU otherwise idle.
A run that starts on a cool, idle GPU reads slightly fast; code/README.md describes the protocol we used for small
differences.

## Other engines

We ran their published recipes without tuning, on the same card and with the same client.

**vLLM 0.27.1.** [sidnaZ/qwen38-27b-rtx4090](https://github.com/sidnaZ/qwen38-27b-rtx4090) at `d2c39e3`: run
`scripts/prepare.sh`, then `runtime/start-single-user.sh` with `MAX_SEQS=2 KV_MEM=4800000000 MAX_LEN=49152` and
`--max-num-batched-tokens 1024` (the script has 2048). Eight sequences ran out of memory at CUDA-graph capture on our
card. Target: the recipe's `Qwen3.8-27B-W4A16-AutoRound-fast`. Drafter: `syvai/Qwen3.8-27B-DFlash2-W4A16` at `4d30ec7`,
7 draft tokens. Run Spec-Bench with `--engine vllm`. Its stream carries no draft statistics, so tokens per round come from the server's
Prometheus counters read before and after the run: 1 + accepted tokens / drafts (`code/site-data/famous_compile.py`).

**llama.cpp** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at `a4d880f`. Target:
`Qwen3.8-27B-UD-Q4_K_XL.gguf` from `unsloth/Qwen3.8-27B-GGUF` at `4ca7207`. Drafter: `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
from `z-lab/Qwen3.8-27B-DFlash2-GGUF`.

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf -ngl all -fa on -c 65536 -np 1 --fit off --no-cache-prompt \
  --reasoning off --alias qwen3.8-27b --port 8080 --metrics \
  -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 7 -ngld all
```

Run Spec-Bench with `--engine llama`. A maximum of 7 drafts was faster than the recipe's 4 (97.8 tok/s).

## The agent workload: AA-AgentPerf-Local

Spec-Bench measures decoding alone. [AA-AgentPerf-Local](https://github.com/ArtificialAnalysis/aa-agentperf-local)
0.3.4's default replay is an agent workload: 8 tasks, 168 turns, a 65,536-token context, one sequence, every turn
generated to its recorded length, the tool's standard sampling (temperature 0.7, top_p 0.8, top_k 20). We ran it
unmodified on the same card on Nov 14, 2026. The [RTX 4090 leaderboard](https://zedongpeng.com/qwen38-4090-inference/)
ranks the first row against other engines on the same replay.

| configuration | weights | minutes to finish | output tok/s | tokens/round |
|---|---|---|---|---|
| our engine, the engine knobs that keep the dense verify head, `NINFER_ANCHOR_BEFORE_LAST_USER=1` | official NInfer v3 container | 4.18 (4.17, 4.18, 4.20) | 275 | 6.6 |
| our engine, every research knob off, `NINFER_ANCHOR_BEFORE_LAST_USER=1` | official NInfer v3 container | 4.22 (4.27, 4.22, 4.17) | 270 | 6.4 |
| our engine, every research knob off | official NInfer v3 container | 15.82 (16.01, 15.82, 15.76) | 266 | 6.4 |
| our engine, the engine knobs that keep the dense verify head | official NInfer v3 container | 15.85 (15.90, 15.75, 15.85) | 268 | 6.5 |
| our engine, `serve.sh official` | official NInfer v3 container | not comparable | | |
| our engine, the same knobs and flag plus the four weight knobs of `serve.sh ours` | our container | 4.12 (4.12, 4.14, 4.03) | 294 | 6.1 |
| our engine, `serve.sh ours` (two-level head) | our container | does not start | | |
| llama.cpp b11236, UD-Q4_K_M, DFlash 7 (AA's recipe) | | 9.02 | 90 | |

Minutes are the median of three replays; output speed and tokens per round are from the median replay. All fifteen
replays finished 168 of 168 turns at the recorded lengths, with the 65,536-token context reported and not reduced. The
row on our container was measured on Nov 19.

- **Decoding carries over.** Output is 266 to 275 tok/s against llama.cpp's 90, and verify trees accept 6.4 tokens per
  round under sampling (5.18 on Spec-Bench, which is greedy and other text).
- **As the server ships, the replay is slower than llama.cpp, because decoding is 12% of it.** Of 949 s, prefill takes
  821 s: the server recomputed 1.29 M of the 2.58 M prompt tokens. The misses are all in the three SWE-bench tasks
  (108 turns). Each of their prompts ends with a turn-counter user message ("Tool-call turn 4 of 100. Remaining turns:
  96.") that the next prompt no longer contains. `ninfer-serve` resumes a conversation from the checkpoint it keeps
  at the end of the prompt, which sits after that message, so every one of these turns is prefilled from the first
  token: 798 s on the three tasks against llama.cpp's 255 s.
- **`NINFER_ANCHOR_BEFORE_LAST_USER=1` keeps that checkpoint in front of a trailing user message instead**
  (`code/reproduce/anchor_patch.py`, 17 lines in the chat template code, off by default). The next request then
  resumes from the previous turn. With every research knob off the server recomputes 0.21 M prompt tokens (llama.cpp
  0.19 M), the three SWE-bench tasks take 107 s and the other five 146 s (llama.cpp: 255 s and 287 s), and the replay
  finishes in 253 s, of which prefill is 124 s and decoding 114 s. The cost: a client that does keep its last user
  message has that one message prefilled again on the next turn, and a prompt that ends this way renders its chat
  template once more to find the boundary. On the same 102 SWE-bench prompts with and without the flag that is 17 ms
  of preparation each (the prompts average 11 k tokens), about 1.8 s of the replay.
- **The engine knobs do not matter here**: with or without the flag the two knob sets differ by less than 1%, inside
  the spread of the three replays.
- **`serve.sh official` is not comparable.** Its two-level head returns argmax tokens and no logits, so the sampled
  requests are served greedily (one replay without the flag: 15.7 min, 296 output tok/s).
- **Our weights change little here.** `serve.sh ours` does not start at this context (with the two-level head its
  decode-only weight copies leave room for 57,344 tokens at most), but with the dense verify head, which sampling
  needs anyway, the container fits with 345 MiB to spare. A round then takes 20.5 ms instead of 24.0. Two of the three
  replays accepted fewer tokens per round than the official weights did (6.0, 6.0 and 6.6 against 6.6 to 6.8), so
  decoding took 104 s against 110 s and the whole replay is about 1% faster: on Spec-Bench the same weights are 31%
  faster. Half of this replay is prefill, which reads the 4-bit copies either way.
- **What did not work first**: asking the engine for a private long anchor at the same boundary. The anchor was stored
  every turn, but resuming from one keeps its source continuation alive, the server then has to evict a continuation
  for the running request, and it values each by how many recent requests could have used it, so the newest anchor was
  the one evicted (60% of prompt tokens from the cache on one task, against 93% with the checkpoint moved).
- **Where a resumed turn's time goes** (an nsys timeline of one SWE-bench task with the first row's settings, Nov 20).
  Its 27 resumed turns recompute 449 tokens on average and take 452 ms from arrival to the first decode round. At the
  cost per token of a full 1,024-token pass those tokens would take 199 ms. Every turn runs five prefill passes, for
  example 3, 4, 94, 27 and 2 tokens: the engine ends a pass after each assistant header and after each (empty) think
  block, so that a state computed from the first token and one resumed from a checkpoint stay bit-identical, and at
  the checkpoint kept before the turn counter. Passes of up to 16 tokens use the small-T kernels and take 18 ms, about
  one read of the weights. Longer ones switch to the generic GEMMs and the long-prompt attention kernel: 60 ms at
  26 tokens, about 50 ms above their token cost up to several hundred tokens.

  | part of a resumed turn | ms |
  |---|---|
  | chat template and tokenizer | 47 |
  | planning, and the checkpoint copied back from host memory | 23 (copy 15) |
  | the three passes of 2–4 tokens | 57 |
  | the two longer passes, above their token cost | 114 |
  | between passes: the new checkpoint copied to host memory | 16 (copy 14) |
  | the longer passes' 440 tokens at a full pass's cost | 195 |
  | arrival to the first decode round | 452 |

  On these turns profiled and unprofiled prefill agree within 4% (367 against 355 ms). In a test on Nov 14, with the
  first version of the checkpoint change, four extra device state slots (`--device-state-slots 4`) cut copy time on the
  same task by a third but left prefill time unchanged. To take the timeline: start the server under
  `nsys launch --trace=cuda,nvtx,osrt --cuda-graph-trace=node`, then `nsys start --sample=none --cpuctxsw=none`,
  replay one task, `nsys stop`, export to sqlite, and run `code/profiling/nsys_budget.py` and `nsys_passkern.py` on the
  export and the server's request log.

To run it, add two protocol fields to the server (neither touches the engine: `ignore_eos`, and a `length` finish
reason whenever the output limit ends a turn) and the flag, rebuild, install the tool, and start a replay:

```bash
python3 qwen38-4090-decode/code/reproduce/aa_serve_patch.py engine
python3 qwen38-4090-decode/code/reproduce/anchor_patch.py engine
cmake --build engine/build --target ninfer-serve
pip install git+https://github.com/ArtificialAnalysis/aa-agentperf-local@1fe8056      # 0.3.4
bash qwen38-4090-decode/code/reproduce/aa_agentperf.sh engine-anchor official/qwen3_8_27b.ninfer aa-results
```

The script starts the server as below with the mode's environment variables in front, runs the tool's six-turn check
and then three full replays of about 4 minutes each (`agentperf-local run --base-url http://127.0.0.1:8080/v1 --model
qwen3.8-27b`). The modes `anchor`, `engine` and `defaults` give the other rows, and `ours-anchor` with
`ours/qwen3_8_27b.ninfer` the row on our container; without the flag a replay takes about
16 minutes. The sampler, thinking and prefix reuse stay at the server's defaults, which is what the benchmark expects;
`serve.sh` turns all three off for Spec-Bench.

```bash
ninfer-serve official/qwen3_8_27b.ninfer --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b \
  --spec dflash2 --draft-tokens 15 --lm-head-draft --verify-tree --kv-dtype k8v4 \
  --max-context 65536 --kv-capacity 65536 --max-concurrency 1 --prefill-chunk 1024
```
