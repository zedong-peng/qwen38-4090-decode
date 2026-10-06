#!/usr/bin/env python3
"""E2E training windows disjoint from the c384 GPTQ calibration (data/fidelity/calib-wiki384-code384): GPTQ's in-sample
KL is ~5x below its held-out KL, so tuning scales on the windows the codes were fitted to would under-weight the error.
Wiki: wikitext-2 train windows N_SKIP .. N_SKIP + N_WIKI (make_calib4.py's stream; the first N_SKIP are c384's).
Code: make_calib4.py's stdlib stream from window N_SKIP on, then Python sources of a few site-packages libraries (sorted
paths, each file capped at 60k characters) until N_CODE windows; nothing named difflib (the eval code stream).
usage: make_e2e_set.py OUT.windows.bin N_WIKI N_CODE [N_SKIP=384]"""
import glob, os, struct, sys, sysconfig

out, n_wiki, n_code = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
skip = int(sys.argv[4]) if len(sys.argv) > 4 else 384
L = 2048
W = os.environ.get("WORK", ".")  # needs W/models/src/Qwen3.8-27B (tokenizer) and W/data/fidelity/wikitext2-train.parquet
# make_calib4.py's module lists, read from its source so the stdlib stream is exactly the same
import pyarrow.parquet as pq
from tokenizers import Tokenizer

tok = Tokenizer.from_file(f"{W}/models/src/Qwen3.8-27B/tokenizer.json")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_calib4.py")).read()
mods = eval(src.split("mods = ", 1)[1].split("\nmore = ", 1)[0])
more = eval(src.split("more = ", 1)[1].split("\nassert", 1)[0])
lib = os.environ.get("PYLIB", sysconfig.get_paths()["stdlib"]) + "/"  # a Python 3.12 lib dir holding the stdlib and site-packages
wiki = "".join(pq.read_table(f"{W}/data/fidelity/wikitext2-train.parquet").column("text").to_pylist())
seen, parts = set(), []
for m in mods + [m for m in more if os.path.exists(lib + m)]:
    if m in seen:
        continue
    seen.add(m); parts.append(open(lib + m).read())
code_ids = tok.encode("".join(parts), add_special_tokens=False).ids[skip * L:]
need = n_code * L
extra = []
for pkg in ("aiohttp", "jinja2", "requests", "urllib3", "pydantic", "fastapi", "starlette", "click", "rich", "yaml",
            "tqdm", "packaging", "pyarrow", "safetensors", "tokenizers", "huggingface_hub", "transformers/generation",
            "transformers/utils", "torch/_dynamo", "torch/fx", "torch/distributed"):
    for f in sorted(glob.glob(f"{lib}site-packages/{pkg}/**/*.py", recursive=True)):
        if "difflib" in f or os.path.getsize(f) < 2000:
            continue
        extra.append(open(f, errors="replace").read()[:60000])
    if len(code_ids) + sum(len(e) for e in extra) // 3 > need * 1.3:
        break
code_ids += tok.encode("".join(extra), add_special_tokens=False).ids
wiki_ids = tok.encode(wiki, add_special_tokens=False).ids[skip * L:]
windows = []
for name, ids, n in (("wiki", wiki_ids, n_wiki), ("code", code_ids, n_code)):
    print(name, len(ids), "tokens available after the first", skip, "windows,", n * L, "needed")
    assert len(ids) >= n * L, (name, len(ids), n * L)
    windows += [ids[i * L:(i + 1) * L] for i in range(n)]
with open(out, "wb") as f:
    for w in windows:
        f.write(struct.pack("<II", L, 1)); f.write(struct.pack(f"<{L}i", *w))
print(len(windows), "windows")
