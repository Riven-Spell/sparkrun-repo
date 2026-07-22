#!/bin/bash
set -e

PATCHFILE="${PWD}/diff.patch"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying PR 48624 NVFP4 MoE padding fix"
if git apply $PATCHFILE --exclude="tests/*"; then
  echo "- PR 48624 applied successfully"
else
  echo "- PR 48624 can't be applied, skipping"
fi
