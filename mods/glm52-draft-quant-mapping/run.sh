#!/bin/bash
set -e

PATCHFILE="${PWD}/draft-quant-packed-mapping.patch"

cd /usr/local/lib/python3.12/dist-packages
echo "Applying GLM-5.2 draft-quant-packed-mapping (prevents fp8 MTP acceptance collapse)"
if git apply $PATCHFILE; then
  echo "- GLM-5.2 draft-quant-packed-mapping applied successfully"
else
  echo "- GLM-5.2 draft-quant-packed-mapping can't be applied, skipping"
fi
