# End-to-End LLM + Engram Benchmark

Measure the throughput impact of Engram memory pooling on SGLang inference,
comparable to Table 2 in arXiv:2603.10087.

## Plan

```
bench_e2e.py:
  1. Start SGLang server (Qwen3-4B/8B) with Engram module
  2. Engram pool uses ctypes to call lib/urma_rw or lib/ubsmem via C library
  3. Run inference requests (prefill + decode)
  4. Measure:
     - Baseline throughput (no Engram)
     - + Engram via UBS-MEM (load/store)
     - + Engram via URMA (RDMA READ)
     - + Engram via TCP
  5. Output: throughput (tokens/s) and overhead % per transport

engram_pool.py:
  - Python ctypes wrapper for lib/urma_rw.so and libubsm_sdk.so
  - Provides EngarmPool.prefetch(token_ids) → embeddings
  - Plugs into sglang/srt/engram/engram_prefetcher.py
```

## Status

Not yet implemented. Depends on:
- [ ] lib/ C libraries working (done)
- [ ] Python ctypes wrapper
- [ ] SGLang Engram module integration
