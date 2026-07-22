#!/bin/bash
set -e

PATCHFILE="${PWD}/diff.patch"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying PR 41834 DSv4 Flash SM12x support + fixes"
if git apply $PATCHFILE --exclude="tests/*"; then
  echo "- PR 41834 applied successfully"
else
  echo "- PR 41834 can't be applied, skipping"
fi
