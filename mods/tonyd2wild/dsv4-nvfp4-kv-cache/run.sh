#!/usr/bin/env bash
# Port tonyd2wild's nvfp4_ds_mla KV-cache datatype (DeepSeek V4 padded 584B
# page probe, stage-C final state) to vLLM v0.26.0 as a runtime mod.
#
# Source: https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
#   - recipe/nvfp4/Dockerfile.stage-{a,b,c}
#   - patches/official-main-b12x-nvfp4-python.patch (KV-cache half only)
#
# Anchors verified against vLLM v0.26.0 and clear of the
# mods/vllm-pr-41834-dsv4-flash hunks, but list this mod AFTER that one.
# Dormant unless --kv-cache-dtype nvfp4 (or nvfp4_ds_mla) is passed.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SITE_PACKAGES="${SITE_PACKAGES:-/usr/local/lib/python3.12/dist-packages}"

python3 "${MOD_DIR}/apply_nvfp4_ds_mla.py"

python3 - <<'PY'
from pathlib import Path
import os
import py_compile

root = Path(os.environ.get("SITE_PACKAGES", "/usr/local/lib/python3.12/dist-packages")) / "vllm"
files = (
    "config/cache.py",
    "config/vllm.py",
    "utils/torch_utils.py",
    "v1/kv_cache_interface.py",
    "models/deepseek_v4/attention.py",
    "models/deepseek_v4/sparse_mla.py",
    "models/deepseek_v4/compressor.py",
    "models/deepseek_v4/nvidia/flashinfer_sparse.py",
    "v1/attention/backends/mla/flashmla_sparse.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
)
for relative in files:
    py_compile.compile(str(root / relative), doraise=True)

def text(rel):
    return (root / rel).read_text()

assert '"nvfp4_ds_mla",' in text("config/cache.py")
assert 'cache_config.cache_dtype = "nvfp4_ds_mla"' in text("config/vllm.py")
assert '"nvfp4_ds_mla": torch.uint8' in text("utils/torch_utils.py")
assert 'kv_cache_dtype == "nvfp4_ds_mla"' in text("utils/torch_utils.py")
kvi = text("v1/kv_cache_interface.py")
assert 'kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla")' in kvi
assert 'cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla")' in kvi
att = text("models/deepseek_v4/attention.py")
assert 'kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla")' in att
assert '"nvfp4_ds_mla", torch.uint8' in att
assert 'self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")' in att
smla = text("models/deepseek_v4/sparse_mla.py")
assert '"nvfp4_ds_mla",' in smla
assert 'cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla")' in smla
fms = text("v1/attention/backends/mla/flashmla_sparse.py")
assert '"nvfp4_ds_mla",' in fms
assert 'kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")' in fms
swa = text("v1/attention/backends/mla/sparse_swa.py")
assert 'cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla")' in swa
assert '"nvfp4_ds_mla",' in swa
assert '"nvfp4_ds_mla",' in text("models/deepseek_v4/compressor.py")
fis = text("models/deepseek_v4/nvidia/flashinfer_sparse.py")
assert '"nvfp4_ds_mla",' in fis
assert '"fp8_ds_mla",\n                "nvfp4_ds_mla",' in fis
sm120 = text("v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py")
assert 'kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla")' in sm120
# attention.py indexer-cache alignment must also treat nvfp4_ds_mla as ds_mla.
assert att.count('cache_dtype in (\n            "fp8_ds_mla",\n            "nvfp4_ds_mla",\n        )') >= 1
print("[dsv4-nvfp4-kv-cache] compile and marker checks passed")
PY
