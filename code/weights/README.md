# Making the re-quantized weights

These are the scripts and the order of steps behind the container on Hugging Face
([zedongpeng/Qwen3.8-27B-NInfer-4090](https://huggingface.co/zedongpeng/Qwen3.8-27B-NInfer-4090)), with the arguments we
ran. Part II of the write-up explains why each step exists and what it bought; this page is the how.

```
Qwen3.8-27B BF16 ─► GPTQ, 768 calibration windows ─► Q4 store + Q3 store ─► E2E scale tuning ─────────┐
                                                                                                       ├─► convert ─► container
DFlash2 drafter ─► self-distillation data ─► LoRA fine-tune ─► GPTQ (Q4; Q3 MLP) ──────────────────────┤
                   (8,000 prompts)                                                                     │
proposal shortlist re-ranked by the target's own outputs (data/mix05.counts.i64) ──────────────────────┘
```

Inputs: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at `1d4bf0f` (BF16, 52 GB) and
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) at `50307d4`. Hardware: one 24 GB GPU and at
least 64 GB of free host RAM (GPTQ on 768 windows keeps 34 GB of calibration activations on the host; E2E peaks at 64 GB).
Python: torch with CUDA, safetensors, tokenizers, numpy, pyarrow. The converter and the engine come from the engine
build in ../../REPRODUCE.md (`tools/convert` in that tree).

Everything below runs from this directory. Shorthands:

```bash
SRC=/path/to/Qwen3.8-27B; DRAFT=/path/to/z-lab-Qwen3.8-27B-DFlash2; ENGINE=/path/to/engine   # built as in REPRODUCE.md
D=data; O=work; mkdir -p $O
```

## data/

| file | what it is |
|---|---|
| `eval.windows.bin` | The fidelity evaluation: 44 windows of 2,048 tokens (46,059 scored positions) from `wiki100k.txt` (wikitext-2 *test*) and `cpython-difflib.py`, the same windows the engine's perplexity tool cuts from `manifest.json`. Every KL64 / top-1 / PPL number in the write-up is on these. |
| `calib-wiki384-code384.windows.bin` | GPTQ calibration (c384): 384 wikitext-2 *train* windows + 384 windows of CPython stdlib source (`make_calib4.py`). |
| `e5.windows.bin` | E2E training windows, disjoint from c384 (`make_e2e_set.py`): the next 384 wiki windows + 384 code windows. |
| `draft-calib-prompts.jsonl` | Prompts decoded to collect drafter GPTQ inputs (`draft_calib_prompts.py`). |
| `mix05.counts.i64` | The proposal shortlist ranking in the container (`shortlist_mix.py`, λ = 0.5). |

The window files are token ids (`fq_eval.py` reads them). The code windows depend on the exact Python 3.12 sources
present when they were cut, so use these files rather than regenerating them.

## 1. Reference distributions

FP32 compute on the BF16 weights, top-64 per scored position. One run for the evaluation windows, and one per
128-window part for the E2E teacher:

```bash
python fq_eval.py $SRC $D/eval.windows.bin $O/ref-fp32 --mode ref
python split_windows.py $D/e5.windows.bin 128 $O/e5
for k in 0 1 2 3 4 5; do python fq_eval.py $SRC $O/e5.part$k.windows.bin $O/ref-e5.part$k --mode ref; done
cat $O/ref-e5.part{0,1,2,3,4,5}.topk.bin > $O/teacher-e5.topk.bin
```

`../fidelity/fidelity.py $D/eval.windows.bin $O/ref-fp32 CANDIDATE` scores any candidate against the reference.

## 2. GPTQ codes for the target

Two stores on the same calibration: every layer projection at Q4, and at Q3 (integer grid [−4, 3]). Act-order, a
clipping search, group 64, sequential layer by layer:

```bash
python fq_eval.py $SRC $D/eval.windows.bin $O/fq-c384-allq4 --mode gptq --recipe allq4 --act-order \
  --clip-grid 1.0,0.95,0.9,0.85,0.8 --calib $D/calib-wiki384-code384.windows.bin --calib-host \
  --ref $O/ref-fp32.topk.bin --save-codes $O/codes-c384-allq4
python fq_eval.py $SRC $D/eval.windows.bin $O/fq-c384-allq3 --mode gptq --recipe allq3 --act-order \
  --clip-grid 1.0,0.95,0.9,0.85,0.8,0.75,0.7 --calib $D/calib-wiki384-code384.windows.bin --calib-host \
  --ref $O/ref-fp32.topk.bin --save-codes $O/codes-c384-allq3
```

## 3. E2E scale tuning

The integer codes stay fixed. Row log-scale offsets and the RMSNorm gains are trained by distillation from the FP32
teacher, through the engine's K8V4 KV codec, with MLP gate/up of layers 0–31 and down of layers 0–15 taken from the Q3
store. Wiki windows count twice. 276 steps, 3.6 hours on one 4090:

```bash
python e2e_scales2.py $SRC $D/e5.windows.bin $O/teacher-e5.topk.bin $O/codes-e2e5 \
  --codes $O/codes-c384-allq4 --codes-b $O/codes-c384-allq3 --groups gate_up:0-31,down:0-15 \
  --param row --norms --lr 5e-4 --lr-norm 5e-4 --bf16 --wpb 4 --epochs 1.5 --kv k8v4 \
  --window-weights 0-383:2,384-767:1 --train-ranges 0-767 --holdout-ranges 0-767 --holdout-every 24 --eval-every 24 \
  --model-out $O/src-e2e5
python fq_eval.py $O/src-e2e5 $D/eval.windows.bin $O/fq-e2e5 --mode codes --codes $O/codes-e2e5 \
  --scale-grid ls8 --kv k8v4 --ref $O/ref-fp32.topk.bin      # the simulated fidelity of the result
```

`codes-e2e5` holds the codes with the offsets folded into the scales; `src-e2e5` is a model directory carrying the
trained norm gains.

## 4. Drafter fine-tuning (self-distillation)

8,000 prompts (UltraChat first turns 50%, GSM8K *train* 20%, Magicoder OSS-Instruct 20%, CodeAlpaca 10%; none is a
Spec-Bench source). The target answers them greedily (≤ 512 tokens); the engine then dumps the drafter's input features
(`fc`, 5 × 5,120 BF16 per token, about 200 GB) by prefilling prompt + answer; LoRA rank 64 trains on block 16 for one
epoch (61 minutes).

```bash
python draft_ft_prompts.py $O/prompts-8k.jsonl 8000          # reads the four datasets from data/draft-ft/src
# a server on a quantized target (we used an earlier all-Q4 GPTQ container; any close one works):
bash ../reproduce/serve.sh ours MODEL.ninfer 8080 &
python draft_ft_generate.py http://127.0.0.1:8080 $O/prompts-8k.jsonl $O/gen.jsonl 512
# restart it eager, dumping features: NINFER_DRAFT_DUMP=$O/dump NINFER_DRAFT_DUMP_ONLY=fc, server flag --no-cuda-graph
python draft_ft_features.py http://127.0.0.1:8080 $O/gen.jsonl $O/dump $O/index.jsonl $SRC 2048
python draft_ft_train.py --data $O --drafter $DRAFT --target $SRC --out $O/run1 --block 16 --epochs 1 \
  --eval-every 1000 --eval-seqs 200                          # merged BF16 drafter in $O/run1/drafter
```

## 5. Drafter GPTQ

Hessians come from the drafter's real inputs while decoding: start a server on the official container in eager mode
with `NINFER_DRAFT_DUMP=$O/dcalib` (flag `--no-cuda-graph`) and send it the calibration prompts. Then GPTQ at Q4 for every
projection and at Q3 for the MLP, merged (the fused qkv input stays Q8 in the container):

```bash
python draft_calib_run.py http://127.0.0.1:8080 $D/draft-calib-prompts.jsonl 384
python draft_gptq.py $O/dcalib $O/run1/drafter $O/dcodes-q4 --bits 4 --act-order --clip-grid 1.0,0.95,0.9,0.85,0.8
python draft_gptq.py $O/dcalib $O/run1/drafter $O/dcodes-q3 --bits 3 --act-order --clip-grid 1.0,0.95,0.9,0.85,0.8,0.75,0.7
python merge_dcodes.py $O/dcodes-q4 $O/dcodes-q3 $O/dcodes-q4q3mlp mlp.gate_proj,mlp.up_proj,mlp.down_proj
```

## 6. Proposal shortlist

The drafter proposes from a 131,072-token shortlist. The released ranking comes from a generic corpus and misses 0.95%
of the target's output tokens, mostly markdown. `shortlist_mix.py` mixes in the target's own output frequencies from the
fine-tune responses (λ = 0.5; held-out misses 0.35%). The result is `data/mix05.counts.i64`.

## 7. Convert

```bash
cd $ENGINE
GPTQ_VARIANT=allq4 GPTQ_Q3_SPEC="mlp/gate,mlp/up:0-31;mlp/down:0-15" DRAFTER_FORMAT=q4 \
python -m tools.convert --model $O/src-e2e5 --recipe /path/to/code/weights/gptq_recipe.py \
  --source gptq=$O/codes-e2e5 --source q3=$O/codes-e2e5 --source dcodes=$O/dcodes-q4q3mlp \
  --source dflash2=$O/run1/drafter --components text,vision,mtp,dflash2 \
  --proposal --ranking /path/to/code/weights/data/mix05.counts.i64 --name qwen3.8-27b --out qwen3_8_27b.ninfer --device cpu
```

About 27 minutes on the CPU. (We converted the target once and then swapped drafters with `convert_splice.py`, which
copies every unchanged object byte for byte; the result is the same container.) Then check quality with the engine's
perplexity tool against `$O/ref-fp32` (`../fidelity/`), and speed with ../../REPRODUCE.md.

## Where to go from here

- **More 3-bit layers.** Gate/up 32–63 and down 16–31 are still Q4. E2E v5 was still improving at its last step, and
  each block of 16 layers moved to Q3 bought 0.8–1.6% of speed once E2E recovered the quality. A v6 that adds a block and trains longer
  is the obvious next round.
- **Prefill kernels that read the 3-bit planes** would drop the duplicate Q4 copy (about 4 GiB) and allow 65,536 tokens of
  context.
- **Drafter fine-tuning has plateaued.** More epochs, rank 128, a second round of new data and soft labels all landed
  at the same +1.2–1.5% over the untuned drafter as run1, although some were better on their own holdout. A third data
  round (24,000 prompts, about 15 GPU hours) was parked. A different objective, not more data, is the open question.
