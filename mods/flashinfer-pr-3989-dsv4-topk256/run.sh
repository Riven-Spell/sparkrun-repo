#!/bin/bash
set -e

PATCHFILE="${PWD}/diff.patch"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying flashinfer PR 3989 DSv4 SM120 topk=256 dispatch"
if git apply $PATCHFILE --exclude="tests/*"; then
  echo "- PR 3989 applied successfully"
else
  echo "- PR 3989 can't be applied, skipping"
fi
