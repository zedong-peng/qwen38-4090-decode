"""qwen3_8_27b with externally chosen target codes (tools/fq_eval.py --save-codes).

Target layer projections import codes from --source gptq=DIR. GPTQ_VARIANT selects formats:
official (the release's Q4/Q5 split), q4down (official with Q4 MLP down) or allq4. The embedding and
head stay Q8 absmax, GDN a/b BF16. DRAFTER_FORMAT (q8 default, the released container's choice;
q6, q5, q4) sets the DFlash2 drafter projections except its fused attention query/key/value input,
which stays Q8 (its only native form). With --source dcodes=DIR (tools/draft_gptq.py) those drafter
projections import their codes from DIR in DRAFTER_FORMAT instead of being rounded here.
DRAFTER_KEEP_Q8 (comma-separated name suffixes, e.g. feature_projection,/attention/output) keeps
those drafter projections Q8 whatever DRAFTER_FORMAT says; DRAFTER_Q4 (same syntax) rounds those to Q4.
GPTQ_Q4_GROUPS (gdn_vz,gdn_out,attn_gv,attn_out,down) and GPTQ_Q4_LAYERS (e.g. 0-31) switch further
parents to Q4 on top of the official split. GPTQ_Q3_LAYERS (e.g. 0-15) with --source q3=DIR (a Q3-grid store,
tools/fq_eval.py --recipe allq3) imports those layers' GPTQ_Q3_ROLES (default mlp gate,up) from DIR, still as
Q4 objects: their codes lie in [-4, 3], which the engine's NINFER_Q3_SHADOW=1 repacks into Q3 planes.
GPTQ_Q3_SPEC (e.g. "mlp/gate,mlp/up:0-15;gdn/query,gdn/key:0-31") gives per-role layer ranges instead; a
half-grid store (odd codes) works the same way.
GPTQ_Q5_SPEC (same syntax, e.g. "mlp/down:32-47;gdn/value,gdn/z:32-63") with --source q5=DIR (tools/fq_eval.py
--recipe allq5) makes those objects Q5 and imports their codes from DIR (bit rebalancing: Q5 where Q4 costs the most KL).
usage: python -m tools.convert ... --recipe W/tools/gptq_recipe.py --source gptq=DIR
"""
import os

from tools.convert.methods import import_grouped
from tools.convert.official_recipes import Q4, Q5, Q6, Q8, _assign

OFFICIAL_Q4 = ("/attention/query", "/attention/key", "/gdn/query", "/gdn/key", "/mlp/gate", "/mlp/up")


DRAFTER = {"q8": Q8, "q6": Q6, "q5": Q5, "q4": Q4}[os.environ.get("DRAFTER_FORMAT", "q8")]
KEEP_Q8 = tuple(x for x in os.environ.get("DRAFTER_KEEP_Q8", "").split(",") if x)
FORCE_Q4 = tuple(x for x in os.environ.get("DRAFTER_Q4", "").split(",") if x)


def _optional_q8_drafter(model, recipe, dcodes=None):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(("/moe/router", "/moe/shared_score", "/attention_conv/kernel_projection",
                              "/mlp_conv/kernel_projection", "/candidate_selector/hidden_projection")):
                continue
            if name.startswith("dflash2/") and not name.endswith(
                    ("/attention/query", "/attention/key", "/attention/value")):
                if name.endswith(KEEP_Q8):
                    _assign(recipe, name, Q8)
                elif name.endswith(FORCE_Q4):
                    _assign(recipe, name, Q4)
                elif dcodes is not None:
                    recipe.assign(name, format=DRAFTER, method=import_grouped,
                                  source=model.source(name, dcodes, DRAFTER))
                else:
                    _assign(recipe, name, DRAFTER)
            else:
                _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        for layer in range(model.components[backend]["config"]["num_hidden_layers"]):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


GROUP_ROLES = {"gdn_vz": ("/gdn/value", "/gdn/z"), "gdn_out": ("/gdn/output",),
               "attn_gv": ("/attention/gate", "/attention/value"), "attn_out": ("/attention/output",),
               "down": ("/mlp/down",)}


def _layers(spec):
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def configure(model, recipe, sources):
    variant = os.environ.get("GPTQ_VARIANT", "official")
    groups = [g for g in os.environ.get("GPTQ_Q4_GROUPS", "").split(",") if g]
    group_layers = _layers(os.environ.get("GPTQ_Q4_LAYERS", ""))
    extra_q4 = tuple(r for g in groups for r in GROUP_ROLES[g])
    if variant not in ("official", "q4down", "allq4"):
        raise ValueError(f"unknown GPTQ_VARIANT {variant}")
    store = sources["gptq"]
    q3_layers = _layers(os.environ.get("GPTQ_Q3_LAYERS", ""))
    q3_roles = tuple("/" + r for r in os.environ.get("GPTQ_Q3_ROLES", "mlp/gate,mlp/up").split(",") if r)
    q3_spec = []  # [(roles, layers)]
    for entry in (e for e in os.environ.get("GPTQ_Q3_SPEC", "").split(";") if e):
        roles, _, spec = entry.partition(":")
        q3_spec.append((tuple("/" + r for r in roles.split(",") if r), _layers(spec)))
    if q3_layers:
        q3_spec.append((q3_roles, q3_layers))
    q5_spec = []  # [(roles, layers)]
    for entry in (e for e in os.environ.get("GPTQ_Q5_SPEC", "").split(";") if e):
        roles, _, spec = entry.partition(":")
        q5_spec.append((tuple("/" + r for r in roles.split(",") if r), _layers(spec)))
    # SourceInputs raises ValueError (not KeyError) for absent names, so test membership by iteration.
    _optional_q8_drafter(model, recipe, sources["dcodes"] if "dcodes" in list(sources) else None)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        layer = int(name.split("/")[2])
        q4 = (variant == "allq4" or name.endswith(OFFICIAL_Q4)
              or (variant == "q4down" and name.endswith("/mlp/down"))
              or (extra_q4 and name.endswith(extra_q4) and (group_layers is None or layer in group_layers)))
        format = Q4 if q4 else Q5
        source = store
        if any(layer in lay and name.endswith(roles) for roles, lay in q5_spec):
            if any(layer in lay and name.endswith(roles) for roles, lay in q3_spec):
                raise ValueError(f"{name}: in both GPTQ_Q3_SPEC and GPTQ_Q5_SPEC")
            format, source = Q5, sources["q5"]
        if any(layer in lay and name.endswith(roles) for roles, lay in q3_spec):
            if format != Q4:
                raise ValueError(f"{name}: GPTQ_Q3_LAYERS needs a Q4 object")
            source = sources["q3"]
        recipe.assign(name, format=format, method=import_grouped, source=model.source(name, source, format))
    if variant == "allq4" or groups:
        # Keep the release's two-parent inputs (the engine's Q4 paths are paired-parent); with every
        # role Q4 the packing groups would otherwise fuse all four roles into one parent.
        for layer in range(model.config["num_hidden_layers"]):
            p = f"text/layers/{layer}/"
            if p + "gdn/query" in model.parameters:
                recipe.group([p + "gdn/query", p + "gdn/key"])
                recipe.group([p + "gdn/value", p + "gdn/z"])
            else:
                recipe.group([p + "attention/query", p + "attention/key"])
                recipe.group([p + "attention/gate", p + "attention/value"])
