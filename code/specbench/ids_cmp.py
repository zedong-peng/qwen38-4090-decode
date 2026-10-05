#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two NINFER_TOKEN_DUMP files request by request. usage: ids_cmp.py A.ids.jsonl B.ids.jsonl"""
import json, sys
a = [json.loads(l) for l in open(sys.argv[1])]
b = [json.loads(l) for l in open(sys.argv[2])]
same = sum(1 for x, y in zip(a, b) if x["ids"] == y["ids"])
first = next((i for i, (x, y) in enumerate(zip(a, b)) if x["ids"] != y["ids"]), None)
print(f"ids identical {same}/{min(len(a), len(b))} (lengths {len(a)}, {len(b)}); first difference at request {first}")
