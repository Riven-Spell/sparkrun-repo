#!/bin/bash
set -e

PATCHFILE="${PWD}/memcheck.diff"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying VLLM memory check skip patch"
if git apply $PATCHFILE --exclude="tests/*"; then
  echo "- Memory check skip patch applied successfully"
else
  echo "- Memory check skip patch can't be applied, skipping"
fi
