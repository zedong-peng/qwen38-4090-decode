#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# From a fresh clone of this repository to a Spec-Bench number on one RTX 4090, in one command:
#   bash code/reproduce/reproduce.sh official|ours|defaults [WORKDIR]
#     official  our engine on the official NInfer v3 container   (we measured 245.2 tok/s)
#     ours      our engine on our container                      (322.3 tok/s)
#     defaults  our engine, every research knob off, official container (229.8 tok/s)
# It builds the engine (once), downloads and checks the container (once), starts the server, runs the 480 Spec-Bench
# first turns and prints decode tok/s and tokens per round. Pick the GPU with CUDA_VISIBLE_DEVICES.
# Needs git, CUDA 12.8, GCC 13, CMake >= 3.28, Ninja, the FFmpeg development libraries, python3 and the hf CLI
# (pip install -U huggingface_hub). CMake honours CC, CXX, CUDACXX and CMAKE_PREFIX_PATH from the environment.
set -euo pipefail
MODE=${1:-}
case $MODE in official | ours | defaults) ;; *) echo "usage: reproduce.sh official|ours|defaults [WORKDIR]" >&2; exit 2 ;; esac
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WD=${2:-$REPO/reproduce-work}
PORT=${PORT:-8080}
ENGINE_URL=${ENGINE_URL:-https://github.com/jram4/ninfer-4090}
SPEC_BENCH_URL=${SPEC_BENCH_URL:-https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl}
mkdir -p "$WD" && cd "$WD"
step() { echo "== $(date +%H:%M:%S) $*"; }

# 1. Engine: jram4's Ada port at 70ebb12, base.diff (Cinference 383e5db merged, cbf7b7e reverted), our 53 patches.
SERVE=$WD/engine/build/apps/ninfer-serve
if [ ! -x "$SERVE" ]; then
  if [ ! -f engine/.qwen38-patched ]; then
    step "fetching the engine"
    rm -rf engine && git clone -q "$ENGINE_URL" engine
    git -C engine checkout -q 70ebb1290dc7abe246c20696a24d21f77faee8d4
    step "applying base.diff and the 53 patches"
    git -C engine apply --index "$REPO/code/engine-series/base.diff"
    G=(git -C engine -c user.name=reproduce -c user.email=reproduce@localhost)
    "${G[@]}" commit -q -m "Base: Cinference 383e5db merged into jram4 70ebb12, cbf7b7e reverted"
    "${G[@]}" am -q "$REPO"/code/engine-series/0*.patch
    touch engine/.qwen38-patched
  fi
  step "building (several minutes)"
  cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DNINFER_BUILD_APPS=ON -DNINFER_RDC=OFF > build.log 2>&1 || { tail -20 build.log; exit 1; }
  cmake --build engine/build -j "${JOBS:-$(nproc)}" >> build.log 2>&1 || { grep -m 10 -i error build.log; exit 1; }
fi

# 2. Weights.
if [ "$MODE" = ours ]; then
  HF_REPO=zedongpeng/Qwen3.8-27B-NInfer-4090 REV=main DIR=weights-ours
  SHA=37cb1507aca20b442a35f4d7833b950e0507767cf766e7140e81fad4faf43d7f
else
  HF_REPO=neroued/Qwen3.8-27B-NInfer REV=1cbd84e7221e51186bd7f093a149912d2489625b DIR=weights-official
  SHA=81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da
fi
MODEL=$WD/$DIR/qwen3_8_27b.ninfer
if [ ! -f "$DIR/.checked" ]; then
  [ -f "$MODEL" ] || { step "downloading $HF_REPO"; hf download "$HF_REPO" qwen3_8_27b.ninfer --revision "$REV" --local-dir "$DIR"; }
  step "checking SHA-256"
  echo "$SHA  $MODEL" | sha256sum -c --quiet - && touch "$DIR/.checked"
fi

# 3. Spec-Bench questions (480 lines).
[ -f question.jsonl ] || curl -fsSL -o question.jsonl "$SPEC_BENCH_URL"
echo "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04  question.jsonl" | sha256sum -c --quiet -

# 4. Serve and measure.
step "starting the server ($MODE)"
NINFER_SERVE=$SERVE bash "$REPO/code/reproduce/serve.sh" "$MODE" "$MODEL" "$PORT" > "server-$MODE.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT
until curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q -i ok; do
  kill -0 $SP 2>/dev/null || { echo "the server exited:"; tail -20 "server-$MODE.log"; exit 1; }
  sleep 2
done
step "running Spec-Bench (about ten minutes)"
python3 "$REPO/code/specbench/specbench.py" --questions question.jsonl --base-url "http://127.0.0.1:$PORT" \
  --label "$MODE" --out results --max-tokens 256
