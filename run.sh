#!/usr/bin/env bash
# Launches the live pose-estimation server.
set -euo pipefail
cd "$(dirname "$0")"

SITE=".venv/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cufft/lib:${LD_LIBRARY_PATH:-}"

source .venv/bin/activate
cd backend
exec python -m uvicorn main:app --host 0.0.0.0 --port 8000
