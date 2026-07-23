#!/bin/bash
set -e

SRC_DIR="/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls/"

cd "$SRC_DIR"

echo "Creating sm121→sm120 symlinks for DeepGEMM JIT headers"

for f in sm120_*.cuh; do
  dest="${f/sm120/sm121}"
  if [ ! -e "$dest" ]; then
    ln -s "$f" "$dest"
    echo "  $dest → $f"
  else
    echo "  $dest already exists, skipping"
  fi
done

echo "DeepGEMM sm121 symlinks complete"
