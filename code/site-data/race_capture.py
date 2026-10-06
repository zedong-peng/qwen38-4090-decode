#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-round replay capture for the write-up: streams Spec-Bench prompts and records every SSE content chunk with its
arrival time (one chunk per verify round on Cinference). Same request body as bench_stream.stream_chat.
usage: race_capture.py --base-url URL --label L --out FILE --qids 122,88,413,258 [--max-tokens 256]"""
import argparse, json, os, time, urllib.request

QFILE = os.environ.get("SPEC_BENCH_QUESTIONS", "data/spec-bench/question.jsonl")  # hemingkx/Spec-Bench


def run(base, content, max_tokens):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); chunks = []; usage = timings = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage") or usage
            timings = obj.get("timings") or timings
            for ch in obj.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    chunks.append([round(time.perf_counter() - t0, 6), piece])
    return {"chunks": chunks, "usage": usage, "timings": timings}


ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True); ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--qids", required=True)
ap.add_argument("--max-tokens", type=int, default=256)
a = ap.parse_args()
qs = {q["question_id"]: q for q in map(json.loads, open(QFILE))}
run(a.base_url, "Say hello.", 16)
res = {"label": a.label, "rows": []}
for qid in map(int, a.qids.split(",")):
    q = qs[qid]
    r = run(a.base_url, q["turns"][0], a.max_tokens)
    r.update(question_id=qid, category=q["category"], prompt=q["turns"][0])
    res["rows"].append(r)
    print(a.label, qid, q["category"], len(r["chunks"]), (r["usage"] or {}).get("completion_tokens"),
          round(r["chunks"][-1][0] - r["chunks"][0][0], 3) if r["chunks"] else None)
json.dump(res, open(a.out, "w"), indent=0, ensure_ascii=False)
