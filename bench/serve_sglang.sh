#!/usr/bin/env bash
# Serve a model with SGLang for calibration. Run inside WSL2.
#   bench/serve_sglang.sh on|off [model]
# "on" = RadixAttention prefix cache enabled (SGLang default); "off" = --disable-radix-cache.
# Env: SGLANG_ENV (venv path, default ~/kvbench/sglang-env), HF_TOKEN read from repo .env if unset.
# Extra flags can be passed via SGLANG_EXTRA (e.g. "--attention-backend triton").
set -euo pipefail
CACHE="${1:?usage: serve_sglang.sh on|off [model]}"
MODEL="${2:-meta-llama/Llama-3.2-1B-Instruct}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HERE/../.env" ]; then
  set -a; . "$HERE/../.env"; set +a
fi
source "${SGLANG_ENV:-$HOME/kvbench/sglang-env}/bin/activate"

# FlashInfer JIT-compiles kernels and needs nvcc; the toolkit must match torch's CUDA (13.0 here).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"

if [ "$CACHE" = "on" ]; then CACHE_FLAG=""; else CACHE_FLAG="--disable-radix-cache"; fi

# Same limits as serve_vllm.sh so the two engines are comparable on the 8 GB card.
exec python -m sglang.launch_server --model-path "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --context-length 8192 \
  --mem-fraction-static 0.80 \
  --dtype float16 \
  --enable-cache-report \
  $CACHE_FLAG ${SGLANG_EXTRA:-}
