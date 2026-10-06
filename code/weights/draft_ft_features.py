#!/usr/bin/env python3
"""Target features (drafter fc inputs, 5 x 5120 BF16 per token) for drafter fine-tuning.
The server runs the research binary in eager mode with NINFER_DRAFT_DUMP=DUMP_DIR and
NINFER_DRAFT_DUMP_ONLY=fc. Each generated response is sent back through /v1/messages as an assistant
prefill (continuation) with max_tokens 1, so the prefill dumps one fc column per token of
prompt + response, in order. The token ids come from the HF chat template with
continue_final_message; a record is kept only when its id count equals the dumped column count.
Writes INDEX.jsonl: id, source, col0, ncols, ids, resp0 (first response position). When the server also
dumps the target's final-norm hidden states (NINFER_DRAFT_DUMP_ONLY=fc,final; tools/patch_final_dump.py),
DUMP_DIR/final.bf16 must grow by the same column count and the record gets fcol0 (its first column).
usage: draft_ft_features.py BASE_URL GENERATED.jsonl DUMP_DIR INDEX.jsonl MODEL_DIR [MAX_TOKENS_TOTAL]"""
import json, os, sys, time, urllib.request
from transformers import AutoTokenizer
base, gen, dump, index, model_dir = sys.argv[1:6]
max_total = int(sys.argv[6]) if len(sys.argv) > 6 else 2048
tok = AutoTokenizer.from_pretrained(model_dir)
FC = os.path.join(dump, "fc.bf16"); COL = 25600 * 2
FIN = os.path.join(dump, "final.bf16"); FCOL = 5120 * 2; final = os.environ.get("FINAL", "") == "1"
done = set()
if os.path.exists(index):
    done = {json.loads(l)["id"] for l in open(index)}
size = lambda path=FC: os.path.getsize(path) if os.path.exists(path) else 0
kept = skipped = 0; t0 = time.time()
with open(index, "a") as fi:
    for line in open(gen):
        g = json.loads(line)
        if g["id"] in done or not g["response"].strip():
            continue
        resp = g["response"].rstrip()
        user = g["messages"]
        prompt_ids = tok.apply_chat_template(user, add_generation_prompt=True, enable_thinking=False, tokenize=True)
        ids = tok.apply_chat_template(user + [{"role": "assistant", "content": resp}], continue_final_message=True,
                                      enable_thinking=False, tokenize=True)
        if hasattr(ids, "input_ids"): ids, prompt_ids = ids["input_ids"], prompt_ids["input_ids"]
        if len(ids) > max_total or ids[:len(prompt_ids)] != prompt_ids:
            skipped += 1; continue
        body = {"model": "qwen3.8-27b", "max_tokens": 1, "temperature": 0,
                "messages": [{"role": "user", "content": user[0]["content"]}, {"role": "assistant", "content": resp}]}
        s0, f0 = size(), size(FIN)
        req = urllib.request.Request(base + "/v1/messages", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        try:
            json.load(urllib.request.urlopen(req, timeout=600))
        except Exception as e:
            print("request failed", g["id"], e, flush=True); skipped += 1; continue
        s1, f1 = size(), size(FIN); assert (s1 - s0) % COL == 0 and (f1 - f0) % FCOL == 0
        ncols = (s1 - s0) // COL
        if ncols != len(ids) or (final and (f1 - f0) // FCOL != ncols):
            print(f"mismatch {g['id']}: dumped {ncols}/{(f1 - f0) // FCOL} vs ids {len(ids)}", flush=True)
            skipped += 1; continue
        rec = {"id": g["id"], "source": g["source"], "col0": s0 // COL, "ncols": ncols, "resp0": len(prompt_ids), "ids": ids}
        if final:
            rec["fcol0"] = f0 // FCOL
        fi.write(json.dumps(rec) + "\n"); fi.flush()
        kept += 1
        if kept % 100 == 0:
            print(f"kept {kept} skipped {skipped} cols {s1 // COL} {time.time() - t0:.0f}s", flush=True)
print(f"kept {kept} skipped {skipped}", flush=True)
