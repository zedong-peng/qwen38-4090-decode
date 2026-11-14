#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# AA-AgentPerf-Local's default replay against our engine on one RTX 4090: the tool unmodified, a 65,536-token context,
# one sequence, exact output lengths, the tool's standard sampling. Starts ninfer-serve, runs the tool's small check
# (aa-mini-v1) and then REPS full replays; each replay's result directory is OUTDIR/MODE/N.
# usage: aa_agentperf.sh defaults|engine|official MODEL.ninfer OUTDIR [REPS=3]
#   env NINFER_SERVE  ninfer-serve built from the patched tree after aa_serve_patch.py
#       AGENTPERF     agentperf-local 0.3.4 (default: agentperf-local on PATH)
#       GPU           card index (default 0); nothing else may run on it
#   defaults  every research knob off
#   engine    the engine knobs that keep the dense verify head: sampled requests take the ordinary route
#   official  serve.sh official. Its two-level head returns argmax tokens only, so sampled requests are served greedily:
#             the replay finishes, but it is not the benchmark's sampling
# Unlike serve.sh this leaves the sampler, thinking and prefix reuse at the server's defaults, as the benchmark expects.
set -u
MODE=$1 MODEL=$2 OUT=$3 REPS=${4:-3}
BIN=${NINFER_SERVE:-ninfer-serve} AA=${AGENTPERF:-agentperf-local} G=${GPU:-0} PORT=${PORT:-8080}
ENGINE=(NINFER_PROPOSAL_SCREEN=3 NINFER_TREE_TEMPERATURE=1.25 NINFER_KV_VI8=1 NINFER_GDN_STATE_F16=1
  NINFER_GDN_CHUNKED_RECORD=1 NINFER_CONV_EPILOGUE=1 NINFER_ATTN_ROWSPLIT=3 NINFER_L2_FILL=121
  NINFER_L2_FILL_NORM_KB=1536 NINFER_L2_FILL_POST_KB=768 NINFER_L2_FILL_GATED_KB=768 NINFER_L2_FILL_RECORD_KB=2048
  NINFER_L2_FILL_ATTN_KB=4096)
HEAD=(NINFER_TWO_LEVEL_HEAD=1 NINFER_HEAD_SCREEN=3 NINFER_SCREEN_TILED=1)
case $MODE in
  defaults) KNOBS=() ;;
  engine) KNOBS=("${ENGINE[@]}") ;;
  official) KNOBS=("${ENGINE[@]}" "${HEAD[@]}") ;;
  *) echo "usage: aa_agentperf.sh defaults|engine|official MODEL.ninfer OUTDIR [REPS]" >&2; exit 2 ;;
esac
mkdir -p "$OUT/$MODE"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G setsid env ${KNOBS[@]+"${KNOBS[@]}"} "$BIN" "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --model-id qwen3.8-27b --spec dflash2 --draft-tokens 15 --lm-head-draft \
  --verify-tree --kv-dtype k8v4 --max-context 65536 --kv-capacity 65536 --max-concurrency 1 --prefill-chunk 1024 \
  --request-log-jsonl "$OUT/$MODE/requests.jsonl" > "$OUT/$MODE/server.log" 2>&1 < /dev/null &
SPID=$!
trap 'kill -- -$SPID 2>/dev/null' EXIT
until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
  kill -0 $SPID 2>/dev/null || { echo "server did not start:"; tail -3 "$OUT/$MODE/server.log"; exit 1; }
  sleep 3
done
for rep in mini $(seq 1 "$REPS"); do
  extra=()
  [ "$rep" = mini ] && extra=(--replay aa-mini-v1)
  "$AA" run --base-url "http://127.0.0.1:$PORT/v1" --model qwen3.8-27b --device "$G" --no-progress \
    --output-dir "$OUT/$MODE/$rep" ${extra[@]+"${extra[@]}"} > "$OUT/$MODE-$rep.log" 2>&1 || { echo "$MODE $rep failed"; exit 1; }
  python3 - "$OUT/$MODE/$rep/summary.json" "$MODE $rep" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); t = s["totals"]
print(f"{sys.argv[2]}: {t['successful_turns']}/{t['turns']} turns, {t['total_inference_latency_ms'] / 60000:.2f} min, "
      f"{s.get('output_tokens_per_second') or 0:.0f} output tok/s")
PY
done
