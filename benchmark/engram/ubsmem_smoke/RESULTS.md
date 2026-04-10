# Cross-Node Engram Access Benchmark Results

**Date**: 2026-04-10
**Hardware**: Node1 (141.61.84.245) → Node2 (192.168.84.247), UB fabric, udma2
**Table**: 10000 rows × 341 dim × float32 = 1364 bytes/row, 128 MB shmem
**Iterations**: 500 per benchmark

## Single-Row Read Latency (1364 bytes)

| Mode | Avg (us) | Min (us) | Max (us) | Throughput | vs TCP |
|------|----------|----------|----------|------------|--------|
| LOCAL DRAM | 0.168 | 0.030 | 0.360 | 8.1 GB/s | 359x |
| **UBS-MEM (cache)** | **0.163** | **0.040** | **0.400** | **8.4 GB/s** | **370x** |
| URMA (RDMA READ) | 2.104 | 2.010 | 11.740 | 648 MB/s | 29x |
| TCP | 60.238 | 56.820 | 110.390 | 22.6 MB/s | 1x |

## Single Float Load (4 bytes)

| Mode | Avg (ns) | Min (ns) | Max (ns) |
|------|----------|----------|----------|
| LOCAL DRAM | 36.3 | 20 | 310 |
| **UBS-MEM (cache)** | **42.8** | **20** | **190** |

## Batch Read

| Mode | Batch=32 | Batch=64 | Batch=128 | Batch=256 |
|------|----------|----------|-----------|-----------|
| | us / row, GB/s | us / row, GB/s | us / row, GB/s | us / row, GB/s |
| LOCAL DRAM | 0.097, 13.4 | 0.065, 20.1 | 0.061, 21.3 | 0.062, 21.1 |
| **UBS-MEM (cache)** | **0.110, 11.9** | **0.074, 17.5** | **0.064, 20.5** | **0.063, 20.5** |
| URMA (RDMA READ) | 0.156, 8.3 | 0.113, 11.5 | 0.094, 13.8 | 0.084, 15.4 |
| TCP | 7.398, 0.18 | 5.589, 0.23 | 4.550, 0.29 | 4.032, 0.32 |

## Full Engram Prefetch (12 tables × N tokens)

| Mode | 32 tok | 64 tok | 128 tok | 256 tok |
|------|--------|--------|---------|---------|
| | ms, GB/s | ms, GB/s | ms, GB/s | ms, GB/s |
| LOCAL DRAM | 0.024, 20.5 | 0.051, 19.6 | 0.109, 18.4 | 0.223, 17.9 |
| **UBS-MEM (cache)** | **0.025, 20.0** | **0.052, 19.1** | **0.111, 18.0** | **0.226, 17.7** |
| URMA (RDMA READ) | 0.031, 16.0 | 0.060, 16.7 | 0.117, 17.0 | 0.230, 17.4 |
| TCP | 1.467, 0.34 | 2.762, 0.36 | 5.421, 0.37 | 10.484, 0.38 |

## Cold/Hot Analysis (first access vs cached, 1364 bytes)

| Mode | Row | Cold (us) | Hot (us) |
|------|-----|-----------|----------|
| LOCAL DRAM | 0 | 0.120 | 0.050 |
| LOCAL DRAM | 5000 | 0.080 | 0.040 |
| UBS-MEM (cache) | 0 | 0.130 | 0.040 |
| UBS-MEM (cache) | 5000 | 0.240 | 0.040 |

## Key Findings

1. **UBS-MEM (cache) cross-node ≈ LOCAL DRAM**: Single-row 0.163us vs 0.168us — UBMMU
   page table caching makes remote memory indistinguishable from local after warmup.

2. **UBS-MEM vs URMA**: 13x faster for single random access (0.163 vs 2.104 us).
   Gap narrows at batch: 20.5 vs 15.4 GB/s (batch=256). URMA's post+poll overhead
   is amortized in batch but never eliminated.

3. **UBS-MEM vs TCP**: 370x faster single-row, 47x faster Engram prefetch.

4. **Cold vs Hot**: UBS-MEM first access ~0.13-0.24us (page fault + UBMMU translation),
   subsequent access ~0.04us (TLB cached). Cold penalty is small (~3-6x of hot).

5. **Engram 128-token target**: 0.111 ms via UBS-MEM, well under the CXL paper's
   <1ms target. Compared to paper's CXL result (~0.2ms), UB achieves ~2x better.

## Comparison with CXL Paper (arXiv:2603.10087)

| Metric | CXL (paper) | UB UBS-MEM | UB URMA |
|--------|-------------|------------|---------|
| Single read | ~0.2 us | 0.163 us | 2.1 us |
| 128-token prefetch | <1 ms | **0.111 ms** | 0.117 ms |
| Throughput impact | 1-7% loss | ~0% loss* | ~2% loss* |

*Estimated from throughput ratio vs local DRAM.
