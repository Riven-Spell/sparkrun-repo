#!/bin/bash
set -e

PATCHFILE="${PWD}/mtp-overhang-fix.patch"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying MTP fix"
if git apply $PATCHFILE --exclude="tests/*"; then
  echo "- MTP fix applied successfully"
else
  echo "- MTP fix can't be applied, skipping"
fi
