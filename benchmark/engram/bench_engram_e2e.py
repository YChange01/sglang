#!/usr/bin/env python3
"""End-to-end Engram pool performance benchmark.

Simulates the full Engram inference pipeline with realistic parameters
from arXiv:2603.10087, without requiring a trained Engram model.

Flow:
  1. Generate random Engram embedding tables (matching real model size)
  2. Load tables into yuanrong-datasystem pool (via KVClient / URMA)
  3. Simulate inference loop:
     - Generate random N-gram hash indices (as a real model would)
     - Prefetch embeddings from pool asynchronously
     - Simulate Transformer compute (GPU sleep or real model)
     - Gather prefetched results
     - Measure: prefetch latency, overlap ratio, throughput impact
  4. Report results comparable to the paper's Table 1 & Figure 5

Usage:
  # Mock mode (no Worker needed, tests logic + CPU overhead):
  python bench_engram_e2e.py --mock

  # With real Worker:
  python bench_engram_e2e.py --hosts 10.0.0.1 --ports 18482

  # With real Worker + real model (measures actual throughput impact):
  python bench_engram_e2e.py --hosts 10.0.0.1 --ports 18482 \
      --with-model --model-path Qwen/Qwen2.5-7B

Paper reference parameters (DeepSeek-like Engram, 100B params ≈ 200GB bf16):
  - 12 sub-tables (over_embedding_k=3, over_embedding_n=5 → 3*(5-1)=12)
  - ~5 KB per token per Engram layer
  - oe_hidden_dim = model_dim / n_grams (e.g. 4096/12 ≈ 341)
  - table rows: M + 2*i + 1, where M ≈ 24,400,000 (for 100B total params)
  - Total: 12 tables × 24.4M rows × 341 dim × 2B(bf16) ≈ 200 GB
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
#  Engram model parameters (matching paper / DeepSeek-like config)
# ---------------------------------------------------------------------------

@dataclass
class EngramModelSpec:
    """Simulated Engram model specification.

    Default: 100B params ≈ 200GB (bf16), matching paper's DeepSeek Engram.
    Use --total-size-gb to scale up/down for your hardware.
    """
    model_dim: int = 4096           # hidden_size
    num_tables: int = 12            # n_grams = k * (n-1)
    over_embedding_m: int = 0       # base modulus (0 = auto from total_size_gb)
    over_embedding_k: int = 3       # hash functions per order
    over_embedding_n: int = 5       # max N-gram order
    vocab_size: int = 131072        # token vocabulary
    dtype_bytes: int = 2            # 2=bf16, 4=float32
    total_size_gb: float = 200.0    # target total size in GB

    def __post_init__(self):
        if self.over_embedding_m == 0:
            # Auto-compute M from total_size_gb
            # total_bytes = num_tables * M * oe_hidden_dim * dtype_bytes
            # M = total_bytes / (num_tables * oe_hidden_dim * dtype_bytes)
            total_bytes = self.total_size_gb * (1024 ** 3)
            dim = self.model_dim // self.num_tables
            self.over_embedding_m = int(
                total_bytes / (self.num_tables * dim * self.dtype_bytes)
            )

    @property
    def oe_hidden_dim(self) -> int:
        return self.model_dim // self.num_tables

    @property
    def bytes_per_token_per_table(self) -> int:
        return self.oe_hidden_dim * self.dtype_bytes

    @property
    def bytes_per_token_total(self) -> int:
        return self.bytes_per_token_per_table * self.num_tables

    def table_rows(self, table_idx: int) -> int:
        """Number of rows in sub-table i."""
        return self.over_embedding_m + 2 * table_idx + 1

    def total_params(self) -> int:
        """Total embedding parameters across all sub-tables."""
        return sum(
            self.table_rows(i) * self.oe_hidden_dim
            for i in range(self.num_tables)
        )

    def total_bytes(self) -> int:
        return self.total_params() * self.dtype_bytes

    def total_gb(self) -> float:
        return self.total_bytes() / (1024 ** 3)

    def summary(self) -> str:
        per_tok_kb = self.bytes_per_token_total / 1024
        dtype_name = "bf16" if self.dtype_bytes == 2 else "fp32"
        return (
            f"Engram spec: {self.num_tables} tables, "
            f"dim={self.oe_hidden_dim}, M={self.over_embedding_m:,}, "
            f"dtype={dtype_name}\n"
            f"  Total: {self.total_gb():.1f} GB "
            f"({self.total_params()/1e9:.1f}B params), "
            f"per_token={per_tok_kb:.1f} KB"
        )


# ---------------------------------------------------------------------------
#  Mock KVClient (for testing without Worker)
# ---------------------------------------------------------------------------

class MockKVClient:
    def __init__(self, **kw):
        self._store = {}
        self._read_count = 0

    def init(self):
        pass

    def mset(self, keys, vals, **kw):
        for k, v in zip(keys, vals):
            self._store[k] = v
        return []

    def get(self, keys, **kw):
        self._read_count += len(keys)
        return [self._store.get(k) for k in keys]


# ---------------------------------------------------------------------------
#  Table generation & loading
# ---------------------------------------------------------------------------

def create_kv_client(host, port, use_mock=False):
    if use_mock:
        return MockKVClient()
    from yr.datasystem import KVClient
    client = KVClient(host=host, port=port, timeout_ms=60000, req_timeout_ms=10000)
    client.init()
    return client


def generate_and_load_tables(
    client, spec: EngramModelSpec, key_prefix="engram",
    max_rows_per_table: int = 0,
) -> dict:
    """Generate random embedding tables and load into KVClient.

    Uses the same chunk-addressed key format as EngramPool
    (``{prefix}:t{table_id}:c{chunk_id}``) so that both the benchmark's
    direct reads and EngramPool.batch_lookup() can access the same data.

    Args:
        max_rows_per_table: If > 0, cap each table at this many rows.

    Returns metadata dict with timing info.
    """
    total_bytes = 0
    t0 = time.perf_counter()
    chunk_size = 1  # one row per key, matching EngramPool default

    for t in range(spec.num_tables):
        full_rows = spec.table_rows(t)
        nrows = min(full_rows, max_rows_per_table) if max_rows_per_table > 0 else full_rows
        dim = spec.oe_hidden_dim
        # Always store as float32 (matching EngramPool.load_table)
        row_bytes = dim * 4

        # Generate and write in batches of 2000 (KVClient limit)
        batch_size = 2000
        for start in range(0, nrows, batch_size):
            end = min(start + batch_size, nrows)
            chunk = np.random.randn(end - start, dim).astype(np.float32)
            # Use chunk-addressed keys: c{chunk_id} where chunk_id == row_id
            # when chunk_size == 1
            keys = [f"{key_prefix}:t{t}:c{r}" for r in range(start, end)]
            vals = [chunk[r - start].tobytes() for r in range(start, end)]
            failed = client.mset(keys, vals)
            if failed:
                print(f"  WARNING: {len(failed)} keys failed on table {t}")

        table_mb = nrows * row_bytes / 1e6
        total_bytes += nrows * row_bytes
        elapsed = time.perf_counter() - t0
        full_mb = full_rows * row_bytes / 1e6
        suffix = f" (capped from {full_rows:,} = {full_mb:.0f}MB)" if nrows < full_rows else ""
        print(
            f"  Table {t:2d}: {nrows:,} rows x {dim}d = {table_mb:6.1f} MB{suffix} "
            f"[{elapsed:.1f}s elapsed]",
            flush=True,
        )

    total_sec = time.perf_counter() - t0
    total_mb = total_bytes / 1e6
    full_total_gb = spec.total_gb()
    print(
        f"  Loaded: {total_mb:.0f} MB in {total_sec:.1f}s "
        f"({total_mb/total_sec:.0f} MB/s)",
        flush=True,
    )
    print(
        f"  Full Engram size (if uncapped): {full_total_gb:.1f} GB",
        flush=True,
    )
    return {"total_bytes": total_bytes, "load_time_s": total_sec}


# ---------------------------------------------------------------------------
#  Simulate Engram inference
# ---------------------------------------------------------------------------

def simulate_ngram_indices(
    batch_tokens: int, spec: EngramModelSpec, max_rows: int = 0
) -> List[tuple]:
    """Generate random hash indices simulating compute_n_gram_ids.

    Returns list of (table_id, row_ids_array).
    """
    result = []
    for t in range(spec.num_tables):
        full_rows = spec.table_rows(t)
        nrows = min(full_rows, max_rows) if max_rows > 0 else full_rows
        row_ids = np.random.randint(0, nrows, size=batch_tokens)
        result.append((t, row_ids))
    return result


def prefetch_from_pool(
    client, table_indices: List[tuple], spec: EngramModelSpec,
    key_prefix="engram", executor=None
) -> tuple:
    """Fetch embeddings from pool. Returns (data_bytes, elapsed_sec)."""
    dim = spec.oe_hidden_dim
    row_bytes = dim * 4  # stored as float32

    t0 = time.perf_counter()
    total_bytes = 0

    def fetch_table(t, row_ids):
        # Use chunk-addressed keys matching EngramPool and generate_and_load_tables
        keys = [f"{key_prefix}:t{t}:c{r}" for r in row_ids]
        # Batch into 10000 chunks (API limit)
        results = []
        for i in range(0, len(keys), 10000):
            results.extend(client.get(keys[i:i+10000]))
        return len(results) * row_bytes

    if executor:
        futures = []
        for t, row_ids in table_indices:
            futures.append(executor.submit(fetch_table, t, row_ids))
        for f in futures:
            total_bytes += f.result()
    else:
        for t, row_ids in table_indices:
            total_bytes += fetch_table(t, row_ids)

    elapsed = time.perf_counter() - t0
    return total_bytes, elapsed


def simulate_transformer_compute(duration_ms: float):
    """Simulate GPU compute time."""
    time.sleep(duration_ms / 1000.0)


# ---------------------------------------------------------------------------
#  Benchmarks
# ---------------------------------------------------------------------------

def bench_prefetch_latency(
    client, spec: EngramModelSpec, batch_sizes=(32, 64, 128, 256),
    num_iters=10, executor=None, max_rows=0
):
    """Measure prefetch latency for various batch sizes."""
    print("\n[Bench 1] Prefetch Latency vs Batch Size")
    print(f"  {'Batch':>8} {'Tokens':>8} {'Data/iter':>10} {'Avg ms':>10} {'Throughput':>12}")
    print("  " + "-" * 55)

    for bs in batch_sizes:
        latencies = []
        for _ in range(num_iters):
            indices = simulate_ngram_indices(bs, spec, max_rows)
            data_bytes, elapsed = prefetch_from_pool(
                client, indices, spec, executor=executor
            )
            latencies.append(elapsed * 1000)

        avg_ms = np.mean(latencies)
        data_kb = spec.bytes_per_token_total * bs / 1024
        throughput = (data_kb / 1024) / (avg_ms / 1000) if avg_ms > 0 else 0
        print(
            f"  {bs:>8} {bs:>8} {data_kb:>8.1f}KB {avg_ms:>9.2f} {throughput:>9.1f} MB/s"
        )


def bench_overlap_efficiency(
    client, spec: EngramModelSpec, batch_size=128,
    compute_ms=5.0, num_iters=20, executor=None, max_rows=0
):
    """Measure how well prefetch overlaps with Transformer compute.

    If prefetch takes P ms and compute takes C ms:
    - Sequential: P + C ms
    - Overlap: max(P, C) ms
    - Efficiency = 1 - (total - C) / P  (1.0 = perfect overlap)
    """
    print(f"\n[Bench 2] Overlap Efficiency (batch={batch_size}, compute={compute_ms}ms)")

    # First measure standalone prefetch time
    prefetch_times = []
    for _ in range(num_iters):
        indices = simulate_ngram_indices(batch_size, spec, max_rows)
        _, elapsed = prefetch_from_pool(client, indices, spec, executor=executor)
        prefetch_times.append(elapsed * 1000)
    avg_prefetch_ms = np.mean(prefetch_times)

    # Then measure overlapped
    overlap_times = []
    thread_pool = ThreadPoolExecutor(max_workers=4)
    for _ in range(num_iters):
        indices = simulate_ngram_indices(batch_size, spec, max_rows)
        t0 = time.perf_counter()
        # Launch prefetch in background
        fut = thread_pool.submit(
            prefetch_from_pool, client, indices, spec, "engram", executor
        )
        # Simulate compute
        simulate_transformer_compute(compute_ms)
        # Wait for prefetch
        fut.result()
        total_ms = (time.perf_counter() - t0) * 1000
        overlap_times.append(total_ms)
    thread_pool.shutdown(wait=False)

    avg_total_ms = np.mean(overlap_times)
    sequential_ms = avg_prefetch_ms + compute_ms
    overhead_ms = max(0, avg_total_ms - compute_ms)
    efficiency = 1.0 - overhead_ms / avg_prefetch_ms if avg_prefetch_ms > 0 else 0

    print(f"  Prefetch alone:    {avg_prefetch_ms:>8.2f} ms")
    print(f"  Compute alone:     {compute_ms:>8.2f} ms")
    print(f"  Sequential (P+C):  {sequential_ms:>8.2f} ms")
    print(f"  Overlapped:        {avg_total_ms:>8.2f} ms")
    print(f"  Overhead:          {overhead_ms:>8.2f} ms")
    print(f"  Overlap efficiency:{efficiency:>8.1%}")
    if efficiency > 0.8:
        print("  => GOOD: prefetch mostly hidden behind compute")
    elif efficiency > 0.5:
        print("  => OK: partial overlap, consider increasing prefetch_ahead")
    else:
        print("  => POOR: prefetch dominates, need faster transport")


def bench_scaling(
    hosts, ports, spec: EngramModelSpec, batch_size=128,
    num_iters=10, use_mock=False, max_rows=0
):
    """Measure throughput scaling with multiple Workers."""
    print(f"\n[Bench 3] Multi-Worker Scaling (batch={batch_size})")
    print(f"  {'Workers':>8} {'Avg ms':>10} {'Throughput':>12} {'Speedup':>10}")
    print("  " + "-" * 45)

    base_throughput = None
    for n in range(1, len(hosts) + 1):
        clients = [create_kv_client(hosts[i], ports[i], use_mock) for i in range(n)]

        # Load tables to this subset (simplified: all to each)
        for c in clients:
            generate_and_load_tables(c, spec, max_rows_per_table=max_rows)

        # Measure with round-robin across clients
        latencies = []
        for it in range(num_iters):
            client = clients[it % n]
            indices = simulate_ngram_indices(batch_size, spec, max_rows)
            _, elapsed = prefetch_from_pool(client, indices, spec)
            latencies.append(elapsed * 1000)

        avg_ms = np.mean(latencies)
        data_kb = spec.bytes_per_token_total * batch_size / 1024
        throughput = (data_kb / 1024) / (avg_ms / 1000) if avg_ms > 0 else 0

        if base_throughput is None:
            base_throughput = throughput
        speedup = throughput / base_throughput if base_throughput > 0 else 1

        print(f"  {n:>8} {avg_ms:>9.2f} {throughput:>9.1f} MB/s {speedup:>9.2f}x")


def bench_packet_size_breakdown(
    client, num_iters=200
):
    """Raw latency vs value size (like paper's Figure: CXL vs RDMA)."""
    print("\n[Bench 4] Raw Read Latency vs Packet Size")
    print(f"  {'Size':>10} {'Avg us':>10} {'P99 us':>10} {'Throughput':>12}")
    print("  " + "-" * 48)

    sizes = [64, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536]

    for size in sizes:
        key = f"bench:raw:{size}"
        val = bytes(np.random.bytes(size))
        client.mset([key], [val])

        # Warmup
        for _ in range(5):
            client.get([key])

        lats = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            client.get([key])
            lats.append((time.perf_counter() - t0) * 1e6)

        avg = np.mean(lats)
        p99 = np.percentile(lats, 99)
        tp = (size / 1e6) / (avg / 1e6) if avg > 0 else 0

        if size >= 1024:
            label = f"{size//1024} KB"
        else:
            label = f"{size} B"

        print(f"  {label:>10} {avg:>9.1f} {p99:>9.1f} {tp:>9.1f} MB/s")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Engram E2E performance benchmark (UB/URMA vs CXL comparison)"
    )
    parser.add_argument("--mock", action="store_true",
                        help="Use in-memory mock (no Worker needed)")
    parser.add_argument("--hosts", default="127.0.0.1",
                        help="Comma-separated Worker IPs")
    parser.add_argument("--ports", default="18482",
                        help="Comma-separated Worker ports")
    # Engram model spec
    parser.add_argument("--model-dim", type=int, default=4096)
    parser.add_argument("--num-tables", type=int, default=12)
    parser.add_argument("--total-size-gb", type=float, default=200.0,
                        help="Target total Engram table size in GB (default 200 ≈ 100B bf16 params)")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--max-rows", type=int, default=50000,
                        help="Cap rows per table for quick bench (0=full size, WARNING: 200GB)")
    # Benchmark params
    parser.add_argument("--batch-sizes", default="32,64,128,256",
                        help="Comma-separated batch sizes")
    parser.add_argument("--compute-ms", type=float, default=5.0,
                        help="Simulated Transformer layer compute time (ms)")
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--skip-load", action="store_true",
                        help="Skip table loading (assume already loaded)")
    parser.add_argument("--prefetch-threads", type=int, default=4)
    args = parser.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",")]
    ports = [int(p.strip()) for p in args.ports.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    spec = EngramModelSpec(
        model_dim=args.model_dim,
        num_tables=args.num_tables,
        total_size_gb=args.total_size_gb,
        dtype_bytes=2 if args.dtype == "bf16" else 4,
    )

    print("=" * 60)
    print("Engram E2E Performance Benchmark")
    print("=" * 60)
    print(f"Mode:      {'MOCK (in-memory)' if args.mock else 'LIVE (UB/URMA)'}")
    print(f"Workers:   {', '.join(f'{h}:{p}' for h,p in zip(hosts, ports))}")
    print(f"{spec.summary()}")
    print(f"Compute:   {args.compute_ms} ms (simulated Transformer layer)")
    print("=" * 60)

    # Connect
    client = create_kv_client(hosts[0], ports[0], args.mock)

    # Load tables
    if not args.skip_load:
        if args.max_rows > 0:
            print(f"\n[Setup] Loading Engram tables (capped at {args.max_rows:,} rows/table for quick bench)...")
        else:
            print(f"\n[Setup] Loading FULL Engram tables ({spec.total_gb():.0f} GB, this may take a while)...")
        generate_and_load_tables(client, spec, max_rows_per_table=args.max_rows)
    else:
        print("\n[Setup] Skipping table load (--skip-load)")

    # Prefetch thread pool (simulates multi-table parallel fetch)
    executor = ThreadPoolExecutor(max_workers=args.prefetch_threads)

    # Run benchmarks
    bench_packet_size_breakdown(client, num_iters=args.num_iters * 20)

    bench_prefetch_latency(
        client, spec, batch_sizes, args.num_iters, executor,
        max_rows=args.max_rows,
    )

    bench_overlap_efficiency(
        client, spec, batch_size=128,
        compute_ms=args.compute_ms, num_iters=args.num_iters,
        executor=executor, max_rows=args.max_rows,
    )

    if len(hosts) > 1 and not args.mock:
        bench_scaling(hosts, ports, spec, batch_size=128,
                      num_iters=args.num_iters, use_mock=args.mock,
                      max_rows=args.max_rows)

    executor.shutdown(wait=False)

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("Compare results with paper Table 1:")
    print(f"  Paper (CXL):  1-7% throughput loss vs DRAM")
    print(f"  Paper (RDMA): <25% peak BW for 64B messages")
    print(f"  This (URMA):  see results above")
    print("=" * 60)


if __name__ == "__main__":
    main()
