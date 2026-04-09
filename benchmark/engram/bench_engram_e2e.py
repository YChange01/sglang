#!/usr/bin/env python3
"""End-to-end Engram pool performance benchmark.

Simulates the full Engram inference pipeline with realistic parameters
from arXiv:2603.10087, without requiring a trained Engram model.

Two read modes:
  --mode kv     : Per-key KVClient.get() (baseline, high RPC overhead)
  --mode shm    : ObjectClient shared memory mmap (like CXL paper's approach)

The shared memory mode stores each table as one contiguous buffer and reads
rows by byte offset — zero per-key RPC, matching CXL's mmap semantics.

Usage:
  # Shared memory mode (recommended, matches paper)
  python bench_engram_e2e.py --mode shm --hosts 10.0.0.1 --ports 18483

  # KV mode (baseline comparison)
  python bench_engram_e2e.py --mode kv --hosts 10.0.0.1 --ports 18483

  # Mock mode (no Worker)
  python bench_engram_e2e.py --mode mock

Paper reference: 200GB Engram (100B bf16 params), 12 sub-tables, ~8KB/token
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
#  Engram model parameters
# ---------------------------------------------------------------------------

@dataclass
class EngramModelSpec:
    """Simulated Engram model specification.
    Default: 100B params ~ 200GB (bf16), matching paper.
    """
    model_dim: int = 4096
    num_tables: int = 12
    over_embedding_m: int = 0       # 0 = auto from total_size_gb
    dtype_bytes: int = 2            # 2=bf16, 4=float32
    total_size_gb: float = 200.0

    def __post_init__(self):
        if self.over_embedding_m == 0:
            total_bytes = self.total_size_gb * (1024 ** 3)
            dim = self.model_dim // self.num_tables
            self.over_embedding_m = int(
                total_bytes / (self.num_tables * dim * self.dtype_bytes)
            )

    @property
    def oe_hidden_dim(self) -> int:
        return self.model_dim // self.num_tables

    @property
    def bytes_per_row(self) -> int:
        return self.oe_hidden_dim * 4  # always stored as float32

    @property
    def bytes_per_token_total(self) -> int:
        return self.bytes_per_row * self.num_tables

    def table_rows(self, table_idx: int) -> int:
        return self.over_embedding_m + 2 * table_idx + 1

    def total_gb(self) -> float:
        total = sum(
            self.table_rows(i) * self.oe_hidden_dim * self.dtype_bytes
            for i in range(self.num_tables)
        )
        return total / (1024 ** 3)

    def summary(self) -> str:
        per_tok_kb = self.bytes_per_token_total / 1024
        return (
            f"Engram: {self.num_tables} tables, dim={self.oe_hidden_dim}, "
            f"M={self.over_embedding_m:,}\n"
            f"  Full size: {self.total_gb():.1f} GB, "
            f"per_token={per_tok_kb:.1f} KB"
        )


# ---------------------------------------------------------------------------
#  Shared Memory Table Manager (ObjectClient-based, like CXL mmap)
# ---------------------------------------------------------------------------

class ShmTableManager:
    """Stores each Engram table as one contiguous shared memory buffer.

    Read = byte offset into memoryview. Zero RPC per row.
    This matches CXL paper's mmap approach.
    """

    def __init__(self, host, port, use_mock=False):
        self.host = host
        self.port = port
        self.use_mock = use_mock
        self._client = None
        self._buffers = {}   # table_id -> memoryview
        self._table_meta = {}  # table_id -> {nrows, dim, row_bytes}

    def initialize(self):
        if self.use_mock:
            return
        from yr.datasystem import ObjectClient
        self._client = ObjectClient(
            host=self.host, port=self.port, timeout_ms=60000,
        )
        self._client.init()

    def load_table(self, table_id: int, nrows: int, dim: int):
        """Generate random data and store as one contiguous buffer."""
        row_bytes = dim * 4  # float32
        total_bytes = nrows * row_bytes
        data = np.random.randn(nrows, dim).astype(np.float32).tobytes()

        key = f"engram_shm:t{table_id}"
        self._table_meta[table_id] = {
            "nrows": nrows, "dim": dim, "row_bytes": row_bytes
        }

        if self.use_mock:
            self._buffers[table_id] = memoryview(bytearray(data))
            return

        # Write to ObjectClient
        self._client.g_increase_ref([key])
        self._client.put(key, data)

    def open_table(self, table_id: int):
        """Get read-only memoryview to the table buffer."""
        if self.use_mock:
            return  # already in _buffers

        key = f"engram_shm:t{table_id}"
        buffers = self._client.get([key], timeout_ms=10000)
        buf = buffers[0]
        self._buffers[table_id] = buf.immutable_data()

    def read_rows(self, table_id: int, row_ids: np.ndarray) -> np.ndarray:
        """Read specific rows by byte offset. Zero RPC."""
        meta = self._table_meta[table_id]
        dim = meta["dim"]
        row_bytes = meta["row_bytes"]
        mv = self._buffers[table_id]

        result = np.empty((len(row_ids), dim), dtype=np.float32)
        for i, rid in enumerate(row_ids):
            start = int(rid) * row_bytes
            end = start + row_bytes
            result[i] = np.frombuffer(mv[start:end], dtype=np.float32)
        return result

    def read_rows_fast(self, table_id: int, row_ids: np.ndarray) -> np.ndarray:
        """Vectorized read using numpy fancy indexing on the raw buffer."""
        meta = self._table_meta[table_id]
        dim = meta["dim"]
        mv = self._buffers[table_id]

        # Interpret entire buffer as float32 2D array
        flat = np.frombuffer(mv, dtype=np.float32).reshape(-1, dim)
        return flat[row_ids]

    def shutdown(self):
        self._buffers.clear()


# ---------------------------------------------------------------------------
#  KV-based reader (baseline, for comparison)
# ---------------------------------------------------------------------------

class KVTableManager:
    """Per-key KVClient reads. High RPC overhead baseline."""

    def __init__(self, host, port, use_ipc=False):
        self.host = host
        self.port = port
        self.use_ipc = use_ipc
        self._client = None
        self._table_meta = {}

    def initialize(self):
        from yr.datasystem import KVClient
        self._client = KVClient(
            host=self.host, port=self.port, timeout_ms=60000,
            req_timeout_ms=10000, enable_exclusive_connection=self.use_ipc,
        )
        self._client.init()

    def load_table(self, table_id: int, nrows: int, dim: int):
        row_bytes = dim * 4
        self._table_meta[table_id] = {"nrows": nrows, "dim": dim, "row_bytes": row_bytes}
        batch_size = 2000
        for start in range(0, nrows, batch_size):
            end = min(start + batch_size, nrows)
            chunk = np.random.randn(end - start, dim).astype(np.float32)
            keys = [f"engram:t{table_id}:c{r}" for r in range(start, end)]
            vals = [chunk[r - start].tobytes() for r in range(start, end)]
            self._client.mset(keys, vals)

    def open_table(self, table_id: int):
        pass  # nothing to open for KV mode

    def read_rows(self, table_id: int, row_ids: np.ndarray) -> np.ndarray:
        meta = self._table_meta[table_id]
        dim = meta["dim"]
        keys = [f"engram:t{table_id}:c{r}" for r in row_ids]
        vals = self._client.get(keys)
        result = np.empty((len(row_ids), dim), dtype=np.float32)
        for i, v in enumerate(vals):
            result[i] = np.frombuffer(v, dtype=np.float32)
        return result

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
#  Benchmarks
# ---------------------------------------------------------------------------

def bench_single_read_latency(mgr, spec, max_rows, num_iters=200):
    """Measure latency for reading a single row."""
    print("\n[Bench 1] Single Row Read Latency")

    table_id = 0
    latencies = []
    for _ in range(num_iters):
        rid = np.random.randint(0, max_rows, size=1)
        t0 = time.perf_counter()
        mgr.read_rows(table_id, rid)
        latencies.append((time.perf_counter() - t0) * 1e6)

    avg = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    print(f"  Avg: {avg:.1f} us, P50: {p50:.1f} us, P99: {p99:.1f} us")
    return avg


def bench_batch_read_latency(mgr, spec, max_rows, num_iters=50):
    """Measure batch read for different sizes."""
    print("\n[Bench 2] Batch Read Latency (single table)")
    print(f"  {'Rows':>8} {'Data':>10} {'Avg ms':>10} {'Throughput':>12}")
    print("  " + "-" * 45)

    for nrows in [32, 128, 512, 2048]:
        latencies = []
        for _ in range(num_iters):
            rids = np.random.randint(0, max_rows, size=nrows)
            t0 = time.perf_counter()
            mgr.read_rows(0, rids)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(latencies)
        data_kb = nrows * spec.bytes_per_row / 1024
        tp = (data_kb / 1024) / (avg_ms / 1000) if avg_ms > 0 else 0
        print(f"  {nrows:>8} {data_kb:>8.1f}KB {avg_ms:>9.3f} {tp:>9.1f} MB/s")


def bench_prefetch_latency(mgr, spec, batch_sizes, max_rows, num_iters=10):
    """Full Engram prefetch: all tables × batch_size tokens."""
    print("\n[Bench 3] Full Engram Prefetch (12 tables × N tokens)")
    print(f"  {'Batch':>8} {'Data':>10} {'Avg ms':>10} {'Throughput':>12}")
    print("  " + "-" * 45)

    for bs in batch_sizes:
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            for t in range(spec.num_tables):
                rids = np.random.randint(0, max_rows, size=bs)
                mgr.read_rows(t, rids)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

        avg_ms = np.mean(latencies)
        data_kb = spec.bytes_per_token_total * bs / 1024
        tp = (data_kb / 1024) / (avg_ms / 1000) if avg_ms > 0 else 0
        print(f"  {bs:>8} {data_kb:>8.1f}KB {avg_ms:>9.2f} {tp:>9.1f} MB/s")


def bench_prefetch_threaded(mgr, spec, batch_sizes, max_rows, num_iters=10,
                            num_threads=4):
    """Full Engram prefetch with parallel table reads."""
    print(f"\n[Bench 4] Threaded Prefetch ({num_threads} threads)")
    print(f"  {'Batch':>8} {'Data':>10} {'Avg ms':>10} {'Throughput':>12}")
    print("  " + "-" * 45)

    executor = ThreadPoolExecutor(max_workers=num_threads)

    for bs in batch_sizes:
        latencies = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            futures = []
            for t in range(spec.num_tables):
                rids = np.random.randint(0, max_rows, size=bs)
                futures.append(executor.submit(mgr.read_rows, t, rids))
            for f in futures:
                f.result()
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)

        avg_ms = np.mean(latencies)
        data_kb = spec.bytes_per_token_total * bs / 1024
        tp = (data_kb / 1024) / (avg_ms / 1000) if avg_ms > 0 else 0
        print(f"  {bs:>8} {data_kb:>8.1f}KB {avg_ms:>9.2f} {tp:>9.1f} MB/s")

    executor.shutdown(wait=False)


def bench_overlap(mgr, spec, max_rows, compute_ms=5.0, batch_size=128,
                  num_iters=10):
    """Measure prefetch + compute overlap efficiency."""
    print(f"\n[Bench 5] Overlap Efficiency (batch={batch_size}, compute={compute_ms}ms)")

    # Standalone prefetch time
    prefetch_times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        for t in range(spec.num_tables):
            rids = np.random.randint(0, max_rows, size=batch_size)
            mgr.read_rows(t, rids)
        prefetch_times.append((time.perf_counter() - t0) * 1000)
    avg_prefetch = np.mean(prefetch_times)

    # Overlapped
    pool = ThreadPoolExecutor(max_workers=1)
    overlap_times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        def do_prefetch():
            for t in range(spec.num_tables):
                rids = np.random.randint(0, max_rows, size=batch_size)
                mgr.read_rows(t, rids)
        fut = pool.submit(do_prefetch)
        time.sleep(compute_ms / 1000.0)
        fut.result()
        overlap_times.append((time.perf_counter() - t0) * 1000)
    pool.shutdown(wait=False)

    avg_overlap = np.mean(overlap_times)
    sequential = avg_prefetch + compute_ms
    overhead = max(0, avg_overlap - compute_ms)
    efficiency = 1.0 - overhead / avg_prefetch if avg_prefetch > 0 else 0

    print(f"  Prefetch alone:    {avg_prefetch:>8.2f} ms")
    print(f"  Compute alone:     {compute_ms:>8.2f} ms")
    print(f"  Sequential (P+C):  {sequential:>8.2f} ms")
    print(f"  Overlapped:        {avg_overlap:>8.2f} ms")
    print(f"  Overhead:          {overhead:>8.2f} ms")
    print(f"  Overlap efficiency:{efficiency:>8.1%}")
    if avg_prefetch < compute_ms:
        print("  => EXCELLENT: prefetch fully hidden behind compute")
    elif efficiency > 0.8:
        print("  => GOOD: prefetch mostly hidden")
    elif efficiency > 0.5:
        print("  => OK: partial overlap")
    else:
        print("  => POOR: prefetch dominates")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Engram E2E benchmark (shared memory vs KV baseline)"
    )
    parser.add_argument("--mode", choices=["shm", "kv", "mock"], default="shm",
                        help="shm=ObjectClient mmap (like CXL), kv=KVClient per-key, mock=in-memory")
    parser.add_argument("--ipc", action="store_true",
                        help="KV mode: use IPC instead of TCP")
    parser.add_argument("--hosts", default="127.0.0.1")
    parser.add_argument("--ports", default="18482")
    parser.add_argument("--model-dim", type=int, default=4096)
    parser.add_argument("--num-tables", type=int, default=12)
    parser.add_argument("--total-size-gb", type=float, default=200.0)
    parser.add_argument("--max-rows", type=int, default=10000,
                        help="Rows per table to actually load (caps memory use)")
    parser.add_argument("--batch-sizes", default="32,64,128,256")
    parser.add_argument("--compute-ms", type=float, default=5.0)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--prefetch-threads", type=int, default=4)
    args = parser.parse_args()

    host = args.hosts.split(",")[0].strip()
    port = int(args.ports.split(",")[0].strip())
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    spec = EngramModelSpec(
        model_dim=args.model_dim,
        num_tables=args.num_tables,
        total_size_gb=args.total_size_gb,
    )

    print("=" * 60)
    print("Engram E2E Performance Benchmark")
    print("=" * 60)
    mode_labels = {"shm": "SHM (ObjectClient mmap, like CXL)",
                   "kv": f"KV ({'IPC' if args.ipc else 'TCP'})",
                   "mock": "MOCK (in-memory numpy)"}
    print(f"Mode:      {mode_labels[args.mode]}")
    print(f"Worker:    {host}:{port}")
    print(f"{spec.summary()}")
    print(f"Max rows:  {args.max_rows:,} per table")
    print(f"Compute:   {args.compute_ms} ms (simulated)")
    print("=" * 60)

    # Create manager
    if args.mode == "shm":
        mgr = ShmTableManager(host, port, use_mock=False)
    elif args.mode == "mock":
        mgr = ShmTableManager(host, port, use_mock=True)
    else:
        mgr = KVTableManager(host, port, use_ipc=args.ipc)

    mgr.initialize()

    # Load tables
    max_rows = args.max_rows
    print(f"\n[Setup] Loading {spec.num_tables} tables ({max_rows:,} rows each)...")
    t0 = time.perf_counter()
    for t in range(spec.num_tables):
        full_rows = spec.table_rows(t)
        nrows = min(full_rows, max_rows) if max_rows > 0 else full_rows
        mgr.load_table(t, nrows, spec.oe_hidden_dim)
        elapsed = time.perf_counter() - t0
        mb = nrows * spec.bytes_per_row / 1e6
        print(f"  Table {t:2d}: {nrows:,} rows = {mb:.1f} MB [{elapsed:.1f}s]", flush=True)

    # Open buffers (for SHM mode, maps shared memory)
    for t in range(spec.num_tables):
        mgr.open_table(t)

    total_mb = spec.num_tables * max_rows * spec.bytes_per_row / 1e6
    print(f"  Total loaded: {total_mb:.0f} MB in {time.perf_counter()-t0:.1f}s")
    print(f"  Full Engram (uncapped): {spec.total_gb():.0f} GB")

    # Run benchmarks
    bench_single_read_latency(mgr, spec, max_rows, num_iters=args.num_iters * 50)

    bench_batch_read_latency(mgr, spec, max_rows, num_iters=args.num_iters * 5)

    bench_prefetch_latency(mgr, spec, batch_sizes, max_rows, args.num_iters)

    bench_prefetch_threaded(mgr, spec, batch_sizes, max_rows, args.num_iters,
                            args.prefetch_threads)

    bench_overlap(mgr, spec, max_rows, args.compute_ms, batch_size=128,
                  num_iters=args.num_iters)

    mgr.shutdown()

    print("\n" + "=" * 60)
    print("Compare with paper (arXiv:2603.10087):")
    print("  CXL read latency:  ~0.2 us (load/store)")
    print("  CXL prefetch 128t: <1 ms")
    print("  CXL throughput:    1-7% loss vs DRAM")
    print("=" * 60)


if __name__ == "__main__":
    main()
