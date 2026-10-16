#!/usr/bin/env bash
# Serve a model with vLLM for calibration. Run inside WSL2.
#   bench/serve_vllm.sh on|off [model]
# Env: VLLM_ENV (venv path, default ~/kvbench/vllm-env), HF_TOKEN read from repo .env if unset.
set -euo pipefail
CACHE="${1:?usage: serve_vllm.sh on|off [model]}"
MODEL="${2:-meta-llama/Llama-3.2-1B-Instruct}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HERE/../.env" ]; then
  set -a; . "$HERE/../.env"; set +a
fi
source "${VLLM_ENV:-$HOME/kvbench/vllm-env}/bin/activate"

# WSL has no CUDA toolkit (nvcc), so FlashInfer's JIT-compiled sampler can't build.
# Use the native sampler; attention stays on the bundled FLASH_ATTN backend. Negligible at max_tokens=1.
export VLLM_USE_FLASHINFER_SAMPLER=0

if [ "$CACHE" = "on" ]; then CACHE_FLAG="--enable-prefix-caching"; else CACHE_FLAG="--no-enable-prefix-caching"; fi

# 8 GB laptop GPU: keep context small and leave headroom for the Windows display.
exec vllm serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --dtype float16 \
  --enable-prompt-tokens-details \
  $CACHE_FLAG
