#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rewrites build-machine paths in a NInfer v3 container's provenance to workspace-relative ones, in place.
The directory JSON is a fixed-size region padded with spaces (tools/artifact/writer.py), so a shorter JSON is padded
back to the same length and the payload offset, every tensor byte and the artifact id stay unchanged.
usage: sanitize_container.py FILE.ninfer PREFIX   (e.g. PREFIX=/data/me/workspace/ -> paths become relative)"""
import json, struct, sys

HEADER = struct.Struct("<8sQ16s")
path, prefix = sys.argv[1], sys.argv[2]
with open(path, "r+b") as f:
    magic, n, _ = HEADER.unpack(f.read(HEADER.size))
    assert magic == b"NINFER\x00\x03", "not a NInfer v3 entry"
    raw = f.read(n)
    old = json.loads(raw)
    text = raw.decode().rstrip(" ")
    new_text = text.replace(prefix, "")
    new = json.loads(new_text)
    assert {k: v for k, v in new.items() if k != "provenance"} == {k: v for k, v in old.items() if k != "provenance"}, \
        "prefix occurs outside provenance"
    body = new_text.encode()
    assert len(body) <= n
    f.seek(HEADER.size)
    f.write(body + b" " * (n - len(body)))
print(json.dumps(new["provenance"], indent=1))
