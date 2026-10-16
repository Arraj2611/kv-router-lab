#!/usr/bin/env bash
# Run the calibration experiments against an already-running server. Run inside WSL2.
#   bench/run_experiments.sh <engine-label> on|off [model]
# e.g. bench/run_experiments.sh vllm-0.31.0 off Qwen/Qwen2.5-1.5B-Instruct
# Cold sweep is only meaningful with caching off; with it on, unique prompts still miss, so both are recorded.
set -euo pipefail
ENGINE="${1:?engine label}"; CACHE="${2:?on|off}"; MODEL="${3:-meta-llama/Llama-3.2-1B-Instruct}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results/$(echo "$MODEL" | tr '/' '_')__${ENGINE}.csv"
mkdir -p "$HERE/results"
PY="${PYTHON:-python3}"
for exp in ${EXPERIMENTS:-cold paired shared load}; do
  "$PY" "$HERE/bench_client.py" --model "$MODEL" --engine "$ENGINE" --prefix-caching "$CACHE" \
    --experiment "$exp" --out "$OUT" --n "${N:-30}"
done
echo "wrote $OUT"
