#!/usr/bin/env python3
"""Send calibration prompts to a running server (greedy, non-streaming).
usage: draft_calib_run.py BASE_URL PROMPTS.jsonl MAX_TOKENS"""
import json, sys, time, urllib.request
base, path, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
total, t0 = 0, time.time()
for line in open(path):
    p = json.loads(line)
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": p["prompt"]}],
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    n = r.get("usage", {}).get("completion_tokens", 0)
    total += n
    print(p["id"], p["kind"], n, f"{time.time() - t0:.0f}s", flush=True)
print("tokens", total)
