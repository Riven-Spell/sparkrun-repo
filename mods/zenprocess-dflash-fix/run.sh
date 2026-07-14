#!/bin/bash
set -e

cd /usr/local/lib/python3.12/dist-packages
echo "Applying DFlash fixes"
if curl -fsL https://gist.githubusercontent.com/zenprocess/ee97fef69503515fe5cb97ac8afb582d/raw/7071d25016aeb89f3c47cee8e3b15686ebd9d2e2/dflash-hybrid-port-vs-dev701.diff | git apply --exclude="tests/*"; then
  echo "- DFlash fixes applied successfully"
else
  echo "- DFlash fixes can't be applied, skipping"
fi