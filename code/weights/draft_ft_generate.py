#!/usr/bin/env python3
"""Target responses for drafter fine-tuning: greedy, non-streaming, resumable (ids already in OUT are
skipped, as are ids in the DONE env list of files). Prompts are split across workers by index:
WORKER/NWORKERS, or shared dynamically when the CLAIMS env names a directory (a prompt is taken by
creating CLAIMS/<id>; clear it only while no worker runs). With DEADLINE (Unix seconds) set, no new prompt is
taken after that time, so a queued job can run in chunks.
usage: draft_ft_generate.py BASE_URL PROMPTS.jsonl OUT.jsonl MAX_TOKENS [WORKER NWORKERS]"""
import json, os, sys, time, urllib.request
base, path, out, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
wk, nw = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
done = set()
for path_ in [out] + [x for x in os.environ.get("DONE", "").split(",") if x]:
    if os.path.exists(path_):
        for l in open(path_):
            try:
                done.add(json.loads(l)["id"])
            except ValueError:
                pass  # a line cut by a killed writer
claims = os.environ.get("CLAIMS", "")
if claims:
    os.makedirs(claims, exist_ok=True)


def claim(pid):
    try:
        os.close(os.open(os.path.join(claims, pid), os.O_CREAT | os.O_EXCL)); return True
    except FileExistsError:
        return False


deadline = float(os.environ.get("DEADLINE", "0") or 0)
total, t0, k = 0, time.time(), 0
with open(out, "a") as f:
    for i, line in enumerate(open(path)):
        p = json.loads(line)
        if deadline and time.time() > deadline:
            break
        if i % nw != wk or p["id"] in done or (claims and not claim(p["id"])):
            continue
        body = {"model": "qwen3.8-27b", "messages": p["messages"], "max_tokens": max_tokens, "temperature": 0}
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        c = r["choices"][0]
        f.write(json.dumps({"id": p["id"], "source": p["source"], "messages": p["messages"],
                            "response": c["message"]["content"], "finish_reason": c.get("finish_reason"),
                            "usage": r.get("usage")}) + "\n"); f.flush()
        total += (r.get("usage") or {}).get("completion_tokens", 0); k += 1
        if k % 50 == 0:
            print(f"{k} done, {total} tokens, {total / (time.time() - t0):.0f} tok/s wall", flush=True)
print("tokens", total, flush=True)
