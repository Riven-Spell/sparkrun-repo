#!/bin/bash
set -e

# Copied verbatim from https://github.com/jschmied/vllm/commit/72bfbcf53f6633b012fdec141cb1aae40f0a3fa2.diff

PATCHFILE="${PWD}/diff.patch"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying PR 40898 forwardport"
if git apply $PATCHFILE --exclude="tests/*"; then
  echo "- PR 40898 forwardport applied successfully"
else
  echo "- PR 40898 forwardport can't be applied, skipping"
fi
