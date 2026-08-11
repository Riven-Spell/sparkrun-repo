# vllm-skip-memory-check

This patch introduces an environment variable on vLLM 0.26.0: `VLLM_STRICT_MEMORY_CHECK`. Set it to `0` to disable the memory checks.

This will probably hit swap. zram is strongly recommended instead of swap (peep those SSD prices...)