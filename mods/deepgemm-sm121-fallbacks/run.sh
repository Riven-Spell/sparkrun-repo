#!/bin/bash
set -e

SRC="${PWD}/sm12x_deep_gemm_fallbacks.py"
DST="/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py"

echo "Installing SM12x DeepGEMM fallbacks (hazyumps bf16 fast-path + 1GB chunks)"

if [ -d "$(dirname "$DST")" ]; then
  cp "$SRC" "$DST"
  echo "  sm12x_deep_gemm_fallbacks.py installed"
else
  echo "  Target directory $(dirname "$DST") does not exist, skipping"
fi
