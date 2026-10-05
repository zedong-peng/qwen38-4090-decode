#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Spec-Bench (hemingkx/Spec-Bench data/spec_bench/question.jsonl, 480 prompts) against an OpenAI-compatible server at
concurrency 1: first turn only, greedy, thinking off, fixed max_tokens, one excluded warmup. Writes LABEL.json (one row
per prompt: tokens, decode window, verify steps, output hash and text) and LABEL.summary.json, and prints per-group
aggregate decode tok/s (sum of tokens - 1 over sum of decode windows) and tokens per verify step.

Decode window = first to last streamed content chunk, measured client side. Steps come from the server's llama.cpp-style
`timings` (draft_n, draft_n_accepted) when it reports them; otherwise tokens per step is left empty.
Stdlib only.
usage: specbench.py --questions question.jsonl --base-url URL --label L --out DIR [--per-category N] [--max-tokens 256]
                    [--engine llama|vllm|other]"""
import argparse, collections, datetime, hashlib, json, time, urllib.request
from pathlib import Path

MT = ("writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities")
QSHA = "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"  # the file these results used


def stream_chat(args, content, max_tokens, salt):
    body = {"model": args.model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False}}
    if args.engine == "llama":
        body["cache_prompt"] = False  # defeat prefix caching
    elif args.engine == "vllm":
        body["cache_salt"] = salt
    req = urllib.request.Request(args.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = last = None; text = []; usage = timings = None
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
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
                    now = time.perf_counter(); first = first or now; last = now; text.append(piece)
    return {"t0": t0, "first": first, "last": last, "usage": usage, "timings": timings, "content": "".join(text)}


def select(path, per_category):
    raw = open(path, "rb").read()
    if hashlib.sha256(raw).hexdigest() != QSHA:
        print("warning: question.jsonl differs from the file these tools were validated on")
    qs = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
    groups = collections.defaultdict(list)
    for q in qs:
        groups["mt_bench" if q["category"] in MT else q["category"]].append(q)
    if per_category:
        picked = {}
        for g, items in groups.items():
            if g == "mt_bench":  # spread over the 8 MT-bench subcategories
                by = collections.defaultdict(list)
                for q in items:
                    by[q["category"]].append(q)
                out = []
                while len(out) < per_category and any(by.values()):
                    for c in MT:
                        if by[c] and len(out) < per_category:
                            out.append(by[c].pop(0))
                picked[g] = out
            else:
                picked[g] = items[:per_category]
        groups = picked
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True); ap.add_argument("--base-url", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="default"); ap.add_argument("--engine", default="other")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--per-category", type=int, default=0); ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--config", action="append", default=[], help="KEY=VALUE recorded in the metadata")
    args = ap.parse_args()
    groups = select(args.questions, args.per_category)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = {"label": args.label, "base_url": args.base_url, "per_category": args.per_category,
            "max_tokens": args.max_tokens, "config": dict(c.split("=", 1) for c in args.config),
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    stream_chat(args, "Say hello.", 16, "warm")
    rows = []
    for g, items in groups.items():
        for q in items:
            r = stream_chat(args, q["turns"][0], args.max_tokens, f"{args.label}:{q['question_id']}:{time.time()}")
            n = (r["usage"] or {}).get("completion_tokens")
            t = r["timings"] or {}
            steps = max(1, n - t.get("draft_n_accepted", 0)) if t.get("draft_n") is not None and n else None
            rows.append({"group": g, "category": q["category"], "question_id": q["question_id"],
                         "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "completion_tokens": n,
                         "decode_window_s": (r["last"] - r["first"]) if r["first"] else None,
                         "ttft_s": (r["first"] - r["t0"]) if r["first"] else None, "steps": steps,
                         "content_sha256": hashlib.sha256(r["content"].encode()).hexdigest(), "content": r["content"]})
        (out / f"{args.label}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    summ = {}
    for g in list(groups) + ["overall"]:
        v = [r for r in rows if (g == "overall" or r["group"] == g) and r["completion_tokens"] and r["decode_window_s"]]
        tok = sum(r["completion_tokens"] - 1 for r in v); win = sum(r["decode_window_s"] for r in v)
        st = [r for r in v if r["steps"]]
        summ[g] = {"n": len(v), "decode_tps": tok / win if win else None,
                   "tokens_per_step": sum(r["completion_tokens"] for r in st) / sum(r["steps"] for r in st) if st else None,
                   "mean_prompt": sum(r["prompt_tokens"] or 0 for r in v) / max(1, len(v)),
                   "mean_out": sum(r["completion_tokens"] for r in v) / max(1, len(v))}
    (out / f"{args.label}.summary.json").write_text(json.dumps({"meta": meta, "summary": summ}, indent=1))
    print(args.label)
    print(f"{'group':15s} {'n':>4s} {'prompt':>7s} {'out':>5s} {'tok/s':>8s} {'tok/step':>8s}")
    f = lambda x, d=1: "-" if x is None else f"{x:.{d}f}"
    for g, s in summ.items():
        print(f"{g:15s} {s['n']:4d} {f(s['mean_prompt'], 0):>7s} {f(s['mean_out'], 0):>5s} "
              f"{f(s['decode_tps']):>8s} {f(s['tokens_per_step'], 2):>8s}")


if __name__ == "__main__":
    main()
