#!/usr/bin/env python3
"""Chat prompts for drafter calibration (GPTQ Hessians from real decoding). Built only from text that
is in none of the evaluation sets: wikitext-2 *train* paragraphs and CPython stdlib modules outside the
fidelity windows (difflib) and the target-GPTQ calibration (argparse, inspect, typing, dataclasses,
pathlib); math problems are templated with a fixed seed. usage: draft_calib_prompts.py OUT.jsonl"""
import ast, json, os, random, sys, sysconfig
import pyarrow.parquet as pq

W = os.environ.get("WORK", ".")  # needs W/models/src/Qwen3.8-27B (tokenizer) and W/data/fidelity/wikitext2-train.parquet
rng = random.Random(20261003)
out = []

text = pq.read_table(f"{W}/data/fidelity/wikitext2-train.parquet").column("text").to_pylist()
paras = [t.strip() for t in text if len(t.strip()) > 700 and not t.strip().startswith("=")]
rng.shuffle(paras)
paras = [p[:1400].rsplit(" ", 1)[0].replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".") for p in paras]
it = iter(paras)
for _ in range(16):
    out.append(("wiki-continue", f"Continue writing this encyclopedia article in the same style:\n\n{next(it)}"))
for _ in range(12):
    out.append(("wiki-summary", f"Summarize the following text in five bullet points, then give it a title.\n\n{next(it)}"))
for _ in range(10):
    out.append(("wiki-qa", f"Read the text and write four quiz questions about it, each followed by its answer.\n\n{next(it)}"))
for lang in ("German", "French", "Chinese", "Spanish", "Japanese", "Russian", "Italian", "Portuguese"):
    out.append(("translate", f"Translate the following paragraph into {lang}.\n\n{next(it)}"))

lib = os.environ.get("PYLIB", sysconfig.get_paths()["stdlib"]) + "/"  # a Python 3.12 lib dir holding the stdlib and site-packages
mods = ["json/decoder.py", "csv.py", "textwrap.py", "shutil.py", "heapq.py", "functools.py", "statistics.py",
        "fractions.py", "calendar.py", "ipaddress.py", "configparser.py", "tokenize.py", "fnmatch.py", "glob.py",
        "string/__init__.py" if os.path.exists(lib + "string/__init__.py") else "string.py", "bisect.py"]
funcs = []
for m in mods:
    src = open(lib + m).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(src, node)
            if seg and 400 < len(seg) < 2500:
                funcs.append((m, seg))
rng.shuffle(funcs)
tasks = ["Explain step by step what this Python function does, then point out any edge cases.",
         "Add type hints and a Google-style docstring to this function. Return the full rewritten code.",
         "Write pytest unit tests that cover this function thoroughly.",
         "Rewrite this function in modern idiomatic Rust.",
         "Rewrite this function in TypeScript.",
         "Refactor this function for readability without changing its behaviour, and explain each change."]
for i in range(30):
    m, seg = funcs[i]
    out.append(("code", f"{tasks[i % len(tasks)]}\n\n```python\n{seg}\n```"))
for i in range(10):
    out.append(("code-gen", ["Write a Python class implementing an LRU cache with O(1) get and put, with tests.",
                             "Implement Dijkstra's algorithm in Python for a graph given as an adjacency dict.",
                             "Write a bash script that finds the ten largest files under a directory.",
                             "Implement a thread-safe bounded queue in Java.",
                             "Write a SQL schema for a library system and five example queries.",
                             "Write a Python function that parses ISO 8601 durations like P3DT4H5M.",
                             "Implement merge sort and quicksort in C and compare them.",
                             "Write a FastAPI service with CRUD endpoints for a todo list.",
                             "Implement a trie with insert, search and prefix listing in Go.",
                             "Write a Python script that computes word frequencies of a text file and plots them."][i]))

names = ["Ana", "Ben", "Chen", "Dina", "Eli", "Fatima", "Goro", "Hana"]
for i in range(16):
    a, b, c = rng.randint(3, 60), rng.randint(2, 15), rng.randint(5, 95)
    n = names[i % len(names)]
    out.append(("math", rng.choice([
        f"{n} buys {a} notebooks at ${b} each and gets a {c}% discount on the total. How much does {n} pay? Solve step by step.",
        f"A train travels {a * 10} km at {b * 10} km/h, then {c} km at {b * 10 + 20} km/h. What is the average speed for the whole trip? Show your work.",
        f"A tank holds {a * 100} liters. One pipe fills it in {b} hours and another drains it in {b + c % 7 + 1} hours. How long does it take to fill the empty tank with both open? Explain.",
        f"Find all integer solutions of x^2 - {a}x + {a * b % 97} = 0, or show there are none. Reason carefully.",
        f"The sum of {b} consecutive integers is {a * b + b * (b - 1) // 2}. What are the integers? Explain."])))
for i, (kind, prompt) in enumerate(out):
    pass
with open(sys.argv[1], "w") as f:
    for i, (kind, prompt) in enumerate(out):
        f.write(json.dumps({"id": i, "kind": kind, "prompt": prompt}) + "\n")
print(len(out), "prompts")
