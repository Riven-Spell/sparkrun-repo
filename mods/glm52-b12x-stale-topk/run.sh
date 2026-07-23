#!/bin/bash
set -e

PATCHFILE="${PWD}/b12x-stale-topk-buffer.patch"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying GLM-5.2 b12x-stale-topk-buffer (prevents ~85%→30% MTP acceptance drop)"
if git apply $PATCHFILE; then
  echo "- GLM-5.2 b12x-stale-topk-buffer applied successfully"
else
  echo "- GLM-5.2 b12x-stale-topk-buffer can't be applied, skipping"
fi
