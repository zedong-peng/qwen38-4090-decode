#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collects the same-session engine comparison (jobs 131-132) for the write-up: Spec-Bench-480 per group (decode tok/s,
tokens per round, ms per round) for every engine, the 8-workload suite medians, and vLLM's tokens per round from its
Prometheus counters (its stream carries no per-request draft statistics).
usage: famous_compile.py RESULTS_DIR OUT.json"""
import json, re, sys

R, OUT = sys.argv[1], sys.argv[2]


def sb(label, tau_override=None):
    s = json.load(open(f"{R}/{label}.summary.json"))["summary"]
    out = {}
    for g, v in s.items():
        tau = v.get("tokens_per_step")
        if tau is None and g == "overall":
            tau = tau_override
        out[g] = {"n": v["n"], "tps": round(v["decode_tps"], 2)}
        if tau:
            out[g].update(tau=round(tau, 3), ms=round(1000 * tau / v["decode_tps"], 2))
    return out


def prom(path):
    tot = {}
    for line in open(path):
        m = re.match(r"^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens))_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            tot[m[1]] = tot.get(m[1], 0.0) + float(m[2])
    return tot


def suite(label):
    s = json.load(open(f"{R}/{label}.summary.json"))["summary"]
    return {w["workload"]: {"tps": round(w["decode_tps"], 1), "tau": round(w["tokens_per_step"], 2) if w.get("tokens_per_step") else None}
            for w in s}


m0, m1 = prom(f"{R}/vllm-k7.metrics0.txt"), prom(f"{R}/vllm-k7.metrics1.txt")
d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
drafts, acc = d.get("vllm:spec_decode_num_drafts", 0), d.get("vllm:spec_decode_num_accepted_tokens", 0)
vllm_tau = 1 + acc / drafts if drafts else None
res = {"sb": {}, "suite": {}, "vllm_counters": d, "vllm_tau": vllm_tau}
for lab in ["sb-final-a", "sb-final-b", "sb-rel-eng", "sb-rel-def", "sb-stock", "sb-llama-n4", "sb-llama-n7"]:
    try:
        res["sb"][lab] = sb(lab)
    except FileNotFoundError:
        print("missing", lab, file=sys.stderr)
res["sb"]["sb-vllm-k7"] = sb("sb-vllm-k7", vllm_tau)
for lab in ["ss-vllm-k7", "ss-llama-n4", "ss-llama-n7", "ss-stock", "ss-rel-eng", "ss-final"]:
    try:
        res["suite"][lab] = suite(lab)
    except FileNotFoundError:
        print("missing", lab, file=sys.stderr)
json.dump(res, open(OUT, "w"), indent=1)
for k, v in res["sb"].items():
    o = v["overall"]
    print(f"{k:14s} {o['tps']:7.1f} tok/s  tau {o.get('tau', float('nan')):.3f}  {o.get('ms', float('nan')):.2f} ms")
