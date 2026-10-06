#!/usr/bin/env python3
"""tools.convert with every non-drafter object copied byte-for-byte from an existing container.
SPLICE_FROM=OLD.ninfer (with OLD.ninfer.conversion.json): a weight job whose parameter list starts
with no "dflash2/" (or SPLICE_CONVERT prefix) name and matches an object of the old report (same parameters, format, shape and
byte size) streams that object's bytes instead of re-encoding. Drafter objects are converted as
usual, so a new drafter checkpoint or drafter format costs minutes instead of a full conversion.
usage: SPLICE_FROM=OLD.ninfer python convert_splice.py [tools.convert arguments]   (cwd = repo)"""
import dataclasses, json, os, sys
sys.path.insert(0, os.getcwd())
import tools.convert.__main__ as cm
from tools.convert.recipe import Recipe
from tools.artifact.reader import Artifact

old_path = os.environ["SPLICE_FROM"]
old = Artifact(old_path)
report = json.load(open(old_path + ".conversion.json"))
by_params = {tuple(m["parameters"]): m for m in report["methods"]}
orig_prepare = Recipe.prepare
stats = {"copied": 0, "converted": 0}
# SPLICE_CONVERT=proposal/,...: parameter prefixes that are always re-converted (e.g. a new proposal ranking).
FORCE = tuple(p for p in os.environ.get("SPLICE_CONVERT", "").split(",") if p)


def prepare(self, **kw):
    pr = orig_prepare(self, **kw)
    jobs = []
    for job in pr.weights:
        m = by_params.get(tuple(job.parameters))
        drafter = any(p.startswith(("dflash2/",) + FORCE) for p in job.parameters)
        if m is not None and not drafter and m["format"] == job.spec.format and tuple(m["shape"]) == tuple(job.spec.shape):
            oid = m["object"]

            def produce(out, oid=oid):
                if old.object(oid).bytes != out.object.bytes:
                    raise ValueError(f"{oid}: {old.object(oid).bytes} bytes in {old_path}, {out.object.bytes} expected")
                off = 0
                for chunk in old.iter_object(oid):
                    out.write_bytes(off, chunk); off += len(chunk)

            job = dataclasses.replace(job, prepared=dataclasses.replace(job.prepared, produce=produce))
            stats["copied"] += 1
        else:
            stats["converted"] += 1
            print("convert", job.spec.id, job.parameters[:2], job.spec.format, flush=True)
        jobs.append(job)
    print(f"splice: {stats['copied']} objects copied from {old_path}, {stats['converted']} converted", flush=True)
    return dataclasses.replace(pr, weights=tuple(jobs))


Recipe.prepare = prepare
cm.main(sys.argv[1:])
