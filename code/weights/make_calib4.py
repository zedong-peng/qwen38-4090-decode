#!/usr/bin/env python3
"""Calibration sets beyond c192: N_WIKI wikitext-2 train windows + N_CODE windows of CPython stdlib source. The module list
starts with make_calib3.py's twenty-three (so its windows are a prefix of these) and appends more stdlib modules until the
code stream is long enough; none is difflib (the eval code stream).
usage: make_calib4.py OUT.windows.bin N_WIKI N_CODE"""
import os, struct, sys, sysconfig
import pyarrow.parquet as pq
from tokenizers import Tokenizer

out, n_wiki, n_code = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
L = 2048
W = os.environ.get("WORK", ".")  # needs W/models/src/Qwen3.8-27B (tokenizer) and W/data/fidelity/wikitext2-train.parquet
tok = Tokenizer.from_file(f"{W}/models/src/Qwen3.8-27B/tokenizer.json")
wiki = "".join(pq.read_table(f"{W}/data/fidelity/wikitext2-train.parquet").column("text").to_pylist())
lib = os.environ.get("PYLIB", sysconfig.get_paths()["stdlib"]) + "/"  # a Python 3.12 lib dir holding the stdlib and site-packages
mods = ["argparse.py", "inspect.py", "typing.py", "dataclasses.py", "pathlib.py", "subprocess.py",
        "threading.py", "asyncio/base_events.py", "logging/__init__.py", "http/client.py", "unittest/case.py",
        "enum.py", "zipfile/__init__.py", "tarfile.py", "ast.py",
        "_pydecimal.py", "pydoc.py", "_pydatetime.py", "functools.py", "collections/__init__.py", "shutil.py",
        "configparser.py", "pickle.py"]
more = ["email/_header_value_parser.py", "mailbox.py", "imaplib.py", "smtplib.py", "ftplib.py", "statistics.py",
        "fractions.py", "random.py", "textwrap.py", "optparse.py", "gettext.py", "locale.py", "calendar.py", "csv.py",
        "json/encoder.py", "json/decoder.py", "xml/etree/ElementTree.py", "xml/dom/minidom.py", "http/server.py",
        "http/cookiejar.py", "urllib/request.py", "urllib/parse.py", "socketserver.py", "selectors.py", "ssl.py",
        "socket.py", "os.py", "posixpath.py", "ntpath.py", "glob.py", "tempfile.py", "heapq.py", "copy.py", "pprint.py",
        "traceback.py", "warnings.py", "weakref.py", "contextlib.py", "queue.py", "multiprocessing/pool.py",
        "concurrent/futures/_base.py", "concurrent/futures/process.py", "asyncio/tasks.py", "asyncio/streams.py",
        "unittest/mock.py", "doctest.py", "pdb.py", "bdb.py", "cmd.py", "dis.py", "tokenize.py", "trace.py",
        "profile.py", "pstats.py", "timeit.py", "re/_parser.py", "re/_compiler.py", "gzip.py", "lzma.py",
        "importlib/_bootstrap.py", "importlib/_bootstrap_external.py", "plistlib.py", "base64.py", "uuid.py",
        "ipaddress.py", "zoneinfo/_zoneinfo.py", "sqlite3/dump.py", "turtle.py", "tkinter/__init__.py", "smtpd.py",
        "aifc.py", "wave.py", "sre_compile.py", "string.py", "numbers.py", "operator.py", "selectors.py"]
assert not any("difflib" in m for m in mods + more)
seen = set(); code_parts = []
for m in mods + [m for m in more if os.path.exists(lib + m)]:
    if m in seen: continue
    seen.add(m); code_parts.append(open(lib + m).read())
windows = []
for name, text, n in (("wiki", wiki, n_wiki), ("code", "".join(code_parts), n_code)):
    ids = tok.encode(text, add_special_tokens=False).ids
    print(name, len(ids), "tokens available,", n * L, "needed")
    assert len(ids) >= n * L, (len(ids), n * L)
    windows += [ids[i * L:(i + 1) * L] for i in range(n)]
with open(out, "wb") as f:
    for w in windows:
        f.write(struct.pack("<II", L, 1)); f.write(struct.pack(f"<{L}i", *w))
print(len(windows), "windows from", len(seen), "modules")
