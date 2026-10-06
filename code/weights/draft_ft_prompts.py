#!/usr/bin/env python3
"""Prompt mix for drafter fine-tuning data (target self-distillation). Sources (none is a Spec-Bench
source): UltraChat-200k train_sft first user turns (chat), GSM8K train questions (math), Magicoder
OSS-Instruct problems and CodeAlpaca instructions (code). Prompts over MAX_CHARS are skipped.
usage: draft_ft_prompts.py OUT.jsonl N [SEED]"""
import glob, json, random, sys
import pyarrow.parquet as pq
out, n = sys.argv[1], int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261003
S = "data/draft-ft/src"; MAX_CHARS = 4000
rng = random.Random(seed)
pools = {}
uc = pq.read_table(glob.glob(f"{S}/ultrachat/data/*.parquet")[0], columns=["messages"]).to_pylist()
pools["chat"] = [r["messages"][0]["content"] for r in uc if r["messages"] and r["messages"][0]["role"] == "user"]
pools["math"] = [r["question"] for r in pq.read_table(glob.glob(f"{S}/gsm8k/main/*.parquet")[0]).to_pylist()]
pools["code_oss"] = [json.loads(l)["problem"] for l in open(glob.glob(f"{S}/magicoder/*.jsonl")[0])]
pools["code_alpaca"] = [(r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")) for r in json.load(open(glob.glob(f"{S}/codealpaca/*.json")[0]))]
mix = {"chat": 0.50, "math": 0.20, "code_oss": 0.20, "code_alpaca": 0.10}
with open(out, "w") as f:
    i = 0
    for src, frac in mix.items():
        pool = [p for p in pools[src] if 0 < len(p) <= MAX_CHARS]
        for p in rng.sample(pool, round(n * frac)):
            f.write(json.dumps({"id": f"{src}-{i:06d}", "source": src, "messages": [{"role": "user", "content": p}]}) + "\n"); i += 1
print({k: len(v) for k, v in pools.items()}, "wrote", i)
