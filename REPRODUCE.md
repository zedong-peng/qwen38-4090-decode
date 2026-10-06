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
