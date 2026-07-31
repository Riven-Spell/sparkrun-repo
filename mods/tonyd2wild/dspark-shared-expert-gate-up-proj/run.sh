#!/bin/bash
set -e

PATCHFILE="${PWD}/0004-dspark-shared-expert-gate-up-proj.patch"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying DSpark shared expert gate up proj mod"
if git apply $PATCHFILE; then
  echo "- Mod applied successfully"
else
  echo "- Mod can't be applied, skipping"
fi
