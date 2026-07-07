#!/bin/bash
set -e

cd ~
pip download nvidia-nccl-cu13==2.30.4 -d /tmp/nccl --no-deps
mkdir -p /var/tmp/nccl-2.30.4
cd /tmp/nccl && unzip -o nvidia_nccl_cu13-2.30.4*.whl 'nvidia/nccl/lib/libnccl.so.2'
cp nvidia/nccl/lib/libnccl.so.2 /var/tmp/nccl-2.30.4/