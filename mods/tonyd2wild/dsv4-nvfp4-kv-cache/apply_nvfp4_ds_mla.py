#!/usr/bin/env python3
"""Apply the nvfp4_ds_mla KV-cache datatype port to vLLM v0.26.0.

Anchor-based, idempotent string replacement (same approach as tonyd2wild's
recipe/nvfp4/Dockerfile.stage-{a,b,c}). Sources:

  https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark
    - recipe/nvfp4/Dockerfile.stage-a  (dtype plumbing)
    - recipe/nvfp4/Dockerfile.stage-b  (DSv4 attention probe)
    - recipe/nvfp4/Dockerfile.stage-c  (padded 584B page probe, final state)
    - patches/official-main-b12x-nvfp4-python.patch (modern-lineage rebase)

Semantics: nvfp4_ds_mla reuses DeepSeek V4's proven fp8_ds_mla 584-byte
per-token cache envelope everywhere (stage-C final state). The fp8 path is
left byte-for-byte untouched; nvfp4_ds_mla only activates when
--kv-cache-dtype nvfp4 / nvfp4_ds_mla is passed.

Only the KV-cache datatype is ported. The B12X MXFP4 MoE expert half of the
official-main patch (envs.py / oracle/mxfp4.py / b12x_mxfp4_moe.py) is
deliberately out of scope.
"""

import os
import sys
from pathlib import Path

SITE_PACKAGES = Path(
    os.environ.get("SITE_PACKAGES", "/usr/local/lib/python3.12/dist-packages")
)
ROOT = SITE_PACKAGES / "vllm"

# (relative_path, old, new) — applied in order, once each.
REPLACEMENTS = [
    # --- Stage A: dtype plumbing -------------------------------------------
    (
        "config/cache.py",
        '    "fp8_ds_mla",\n    "turboquant_k8v4",',
        '    "fp8_ds_mla",\n    "nvfp4_ds_mla",\n    "turboquant_k8v4",',
    ),
    (
        "config/vllm.py",
        '''        if self.cache_config.cache_dtype == "nvfp4" and self.model_config.use_mla:
            raise ValueError(
                "nvfp4 KV cache is not supported with MLA (Multi-head Latent "
                "Attention) backends. Please use a different --kv-cache-dtype "
                "(e.g., 'fp8' or 'auto') for MLA models such as DeepSeek."
            )
''',
        '''        if self.cache_config.cache_dtype == "nvfp4" and self.model_config.use_mla:
            # MLA + nvfp4 resolves to the DeepSeek V4 padded nvfp4_ds_mla layout.
            self.cache_config.cache_dtype = "nvfp4_ds_mla"
''',
    ),
    (
        "utils/torch_utils.py",
        '    "fp8_ds_mla": torch.uint8,\n    "turboquant_k8v4": torch.uint8,',
        '    "fp8_ds_mla": torch.uint8,\n'
        '    "nvfp4_ds_mla": torch.uint8,\n'
        '    "turboquant_k8v4": torch.uint8,',
    ),
    (
        "utils/torch_utils.py",
        '''        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype == "nvfp4"
''',
        '''        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype == "nvfp4"
        or kv_cache_dtype == "nvfp4_ds_mla"
''',
    ),
    # --- KV-cache specs: same 584B/token envelope as fp8_ds_mla ------------
    (
        "v1/kv_cache_interface.py",
        '    if kv_cache_dtype == "nvfp4":\n        return KVQuantMode.NVFP4\n',
        '    if kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla"):\n'
        "        return KVQuantMode.NVFP4\n",
    ),
    (
        # MLAAttentionSpec.real_page_size_bytes: nvfp4_ds_mla takes the
        # deepseek_v4 584B/token branch, exactly like fp8_ds_mla.
        "v1/kv_cache_interface.py",
        '        if self.cache_dtype_str == "fp8_ds_mla":\n'
        '            if self.model_version == "deepseek_v4":',
        '        if self.cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):\n'
        '            if self.model_version == "deepseek_v4":',
    ),
    (
        # SlidingWindowMLASpec.real_page_size_bytes (official-main patch).
        "v1/kv_cache_interface.py",
        '        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":',
        '''        if self.model_version == "deepseek_v4" and self.cache_dtype_str in (
            "fp8_ds_mla",
            "nvfp4_ds_mla",
        ):''',
    ),
    # --- DeepSeek V4 attention (official-main patch, v0.26.0-adapted) ------
    (
        # _resolve_dsv4_kv_cache_dtype: nvfp4 -> nvfp4_ds_mla coercion.
        "models/deepseek_v4/attention.py",
        '''    page-size specs pick the 576B per-token slot). Plain-row backends store each
    token's KV row in its element dtype: bf16 or per-tensor FP8 E4M3.
    """
    if use_fp8_ds_mla_layout:
''',
        '''    page-size specs pick the 576B per-token slot). Plain-row backends store each
    token's KV row in its element dtype: bf16 or per-tensor FP8 E4M3.
    """
    if kv_cache_dtype in ("nvfp4", "nvfp4_ds_mla"):
        assert use_fp8_ds_mla_layout, (
            "DeepseekV4 nvfp4 KV cache requires the sparse MLA padded layout"
        )
        if cache_config is not None:
            cache_config.cache_dtype = "nvfp4_ds_mla"
        logger.info_once("Using DeepSeek V4 padded nvfp4_ds_mla KV cache format.")
        return "nvfp4_ds_mla", torch.uint8

    if use_fp8_ds_mla_layout:
''',
    ),
    (
        # get_kv_cache_spec: uint8 dtype + fp8_ds_mla alignment for
        # nvfp4_ds_mla. Keeps v0.26.0's 576/512 alignment and kv_quant_mode
        # field (the official-main tree used 584/None without kv_quant_mode).
        "models/deepseek_v4/attention.py",
        '''        # fp8_ds_mla is a UE8M0 block-scaled uint8 layout and needs 576B
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
''',
        '''        # fp8_ds_mla/nvfp4_ds_mla are padded uint8 layouts and need a fixed
        # alignment; plain bf16 / per-tensor fp8 rows use natural element-size
        # pages.
        uses_ds_mla_layout = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )
''',
    ),
    # --- Backend dtype support (official-main patch) -----------------------
    (
        "models/deepseek_v4/sparse_mla.py",
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]
''',
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]
''',
    ),
    (
        "models/deepseek_v4/sparse_mla.py",
        '        if cache_dtype_str == "fp8_ds_mla":\n'
        "            # DeepseekV4 main MLA: 584B per token"
        " (448 NoPE + 128 RoPE + 8 fp8 scale).",
        '        if cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):\n'
        "            # DeepseekV4 main MLA: 584B per token"
        " (448 NoPE + 128 RoPE + 8 fp8 scale).",
    ),
    (
        "v1/attention/backends/mla/flashmla_sparse.py",
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]
''',
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]
''',
    ),
    (
        "v1/attention/backends/mla/flashmla_sparse.py",
        '''        if is_quantized_kv_cache(kv_cache_dtype):
            assert kv_cache_dtype == "fp8_ds_mla", (
                "FlashMLA Sparse Attention backend fp8 only supports "
                "fp8_ds_mla kv-cache dtype"
            )

        if kv_cache_dtype == "fp8_ds_mla":
''',
        '''        if is_quantized_kv_cache(kv_cache_dtype):
            assert kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla"), (
                "FlashMLA Sparse Attention backend only supports "
                "fp8_ds_mla/nvfp4_ds_mla kv-cache dtype"
            )

        if kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla"):
''',
    ),
    # --- SWA / indexer / compressor caches: same envelope as an fp8 run ----
    # These caches size/align off cache_config.cache_dtype; without these
    # edits an nvfp4_ds_mla run silently gets 512-wide SWA pages while the
    # SM120 packed path expects the 584B fp8_ds_mla layout.
    (
        "v1/attention/backends/mla/sparse_swa.py",
        '        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"\n',
        "        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype in (\n"
        '            "fp8_ds_mla",\n'
        '            "nvfp4_ds_mla",\n'
        "        )\n",
    ),
    (
        # DeepseekSparseSWABackend.get_kv_cache_shape: 584B/token SWA layout.
        "v1/attention/backends/mla/sparse_swa.py",
        '        if cache_dtype_str == "fp8_ds_mla":\n'
        "            # DeepseekV4 SWA: 584B per token"
        " (448 NoPE + 128 RoPE + 8 fp8 scale).",
        '        if cache_dtype_str in ("fp8_ds_mla", "nvfp4_ds_mla"):\n'
        "            # DeepseekV4 SWA: 584B per token"
        " (448 NoPE + 128 RoPE + 8 fp8 scale).",
    ),
    (
        # DeepseekV4IndexerCache.get_kv_cache_spec: keep 576B alignment.
        "models/deepseek_v4/attention.py",
        '        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"\n',
        "        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype in (\n"
        '            "fp8_ds_mla",\n'
        '            "nvfp4_ds_mla",\n'
        "        )\n",
    ),
    (
        # Compressor state cache: keep 576B alignment.
        "models/deepseek_v4/compressor.py",
        '        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"\n',
        "        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype in (\n"
        '            "fp8_ds_mla",\n'
        '            "nvfp4_ds_mla",\n'
        "        )\n",
    ),
    # --- SM12x default path: FlashInfer DSV4 sparse backend ----------------
    # On SM12, _select_dsv4_attn_cls defaults to DeepseekV4FlashInferSM120Attention
    # (backend FLASHINFER_MLA_SPARSE_DSV4). Its gates must accept nvfp4_ds_mla
    # or the backend is rejected / raises at init.
    (
        "models/deepseek_v4/nvidia/flashinfer_sparse.py",
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]
''',
        '''    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
        "nvfp4_ds_mla",
    ]
''',
    ),
    (
        "models/deepseek_v4/nvidia/flashinfer_sparse.py",
        '            if kv_cache_dtype not in ("fp8", "fp8_e4m3", "fp8_ds_mla"):\n'
        '                return "kv_cache_dtype not supported"',
        '            if kv_cache_dtype not in (\n'
        '                "fp8",\n'
        '                "fp8_e4m3",\n'
        '                "fp8_ds_mla",\n'
        '                "nvfp4_ds_mla",\n'
        "            ):\n"
        '                return "kv_cache_dtype not supported"',
    ),
    (
        "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py",
        '''        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )
''',
        '''        if self.kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla"):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed "
                "fp8_ds_mla/nvfp4_ds_mla KV cache layout; got "
                f"kv_cache_dtype={kv_cache_dtype!r}."
            )
''',
    ),
]


def main() -> int:
    missing = []
    applied = 0
    skipped = 0
    for rel, old, new in REPLACEMENTS:
        path = ROOT / rel
        if not path.is_file():
            missing.append(f"{rel}: file not found ({path})")
            continue
        text = path.read_text()
        if new in text:
            skipped += 1
            print(f"[dsv4-nvfp4-kv-cache] already applied: {rel}")
            continue
        if old not in text:
            missing.append(f"{rel}: anchor not found:\n{old}")
            continue
        path.write_text(text.replace(old, new, 1))
        applied += 1
        print(f"[dsv4-nvfp4-kv-cache] patched: {rel}")

    if missing:
        print(
            "[dsv4-nvfp4-kv-cache] ERROR: anchors missing; "
            "the vLLM tree does not match v0.26.0 expectations:",
            file=sys.stderr,
        )
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1
    print(
        f"[dsv4-nvfp4-kv-cache] done: {applied} file(s) patched, "
        f"{skipped} already applied"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
