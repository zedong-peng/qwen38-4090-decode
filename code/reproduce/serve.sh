#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Starts ninfer-serve in one of the configurations measured in the write-up (one RTX 4090, batch 1, greedy).
# usage: serve.sh official|ours|defaults MODEL.ninfer [PORT]     env NINFER_SERVE: the built ninfer-serve binary
#   official  our engine on the official release container: the engine knobs only     (Spec-Bench 245.2 tok/s)
#   ours      plus the four knobs that belong to our weights (Part II of the write-up) (Spec-Bench 322.3 tok/s)
#   defaults  our engine with every research knob off, verify trees on                 (Spec-Bench 229.8 tok/s)
set -eu
MODE=$1 MODEL=$2 PORT=${3:-8080}
BIN=${NINFER_SERVE:-ninfer-serve}
# Engine knobs: verify-tree head and screens, KV/GDN state formats, the fused conv epilogue, L2 prefetch sizes.
ENGINE=(NINFER_TWO_LEVEL_HEAD=1 NINFER_HEAD_SCREEN=3 NINFER_SCREEN_TILED=1 NINFER_PROPOSAL_SCREEN=3
  NINFER_TREE_TEMPERATURE=1.25 NINFER_KV_VI8=1 NINFER_GDN_STATE_F16=1 NINFER_GDN_CHUNKED_RECORD=1
  NINFER_CONV_EPILOGUE=1 NINFER_ATTN_ROWSPLIT=3 NINFER_L2_FILL=121 NINFER_L2_FILL_NORM_KB=1536
  NINFER_L2_FILL_POST_KB=768 NINFER_L2_FILL_GATED_KB=768 NINFER_L2_FILL_RECORD_KB=2048 NINFER_L2_FILL_ATTN_KB=4096)
# Weight knobs: 8-bit group scales and a Q4 drafter conv copy made at load, and 3-bit planes for layers whose codes fit
# them (only our re-quantized container has such layers). Their quality cost is what Part II scores.
WEIGHTS=(NINFER_LS8=1 NINFER_DRAFT_CONV_Q4=1 NINFER_Q3_SHADOW=1 NINFER_Q3_TILED=1)
case $MODE in
  official) KNOBS=("${ENGINE[@]}") ;;
  ours) KNOBS=("${ENGINE[@]}" "${WEIGHTS[@]}") ;;
  defaults) KNOBS=() ;;
  *) echo "usage: serve.sh official|ours|defaults MODEL.ninfer [PORT]" >&2; exit 2 ;;
esac
exec env ${KNOBS[@]+"${KNOBS[@]}"} "$BIN" "$MODEL" --host 127.0.0.1 --port "$PORT" --model-id qwen3.8-27b \
  --spec dflash2 --draft-tokens 15 --lm-head-draft --verify-tree --kv-dtype k8v4 \
  --max-context 32768 --kv-capacity 32768 --max-concurrency 1 --prefill-chunk 1024 \
  --temperature 0 --no-thinking --no-prefix-reuse
