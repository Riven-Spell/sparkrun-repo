#!/bin/bash
set -e

cd /usr/local/lib/python3.12/dist-packages
echo "Applying PR #45544"
if curl -fsL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/45544.diff | git apply --exclude="tests/*"; then
  echo "- PR #45544 applied successfully"
else
  echo "- PR #45544 can't be applied, skipping"
fi
