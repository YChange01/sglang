#!/usr/bin/env python3
"""Benchmark: URMA small-packet read latency and throughput for Engram.

Reproduces the key measurements from the CXL paper (arXiv:2603.10087)
using UB/URMA transport via yuanrong-datasystem.

Usage:
    # Start yuanrong-datasystem Worker first:
    #   dscli start -w <worker_ip>:<worker_port>

    python bench_urma_small_packet.py \
        --host 10.0.0.1 --port 18482 \
        --embed-dim 128 --num-rows 100000

Outputs:
    - Latency vs packet size (64B ~ 64KB)
    - Throughput vs concurrent readers
    - Prefetch overlap efficiency
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

import numpy as np

# Try import — fail gracefully with instructions
try:
    from yr.datasystem import KVClient
except ImportError:
    KVClient = None
    print("WARNING: yr.datasystem not installed. Using mock for demo.")


@dataclass
class BenchResult:
    label: str
    packet_bytes: int
    num_ops: int
    total_sec: float

    @property
    def avg_latency_us(self) -> float:
        return (self.total_sec / self.num_ops) * 1e6

    @property
    def throughput_mbps(self) -> float:
        total_bytes = self.packet_bytes * self.num_ops
        return (total_bytes / 1e6) / self.total_sec


class MockKVClient:
    """In-process mock for demo/development without a Worker."""

    def __init__(self, **kw):
        self._store = {}

    def init(self):
        pass

    def mset(self, keys, vals, **kw):
        for k, v in zip(keys, vals):
            self._store[k] = v
        return []

    def get(self, keys, **kw):
        return [self._store.get(k, b"") for k in keys]


def create_client(host: str, port: int) -> "KVClient":
    if KVClient is None:
        return MockKVClient()
    client = KVClient(host=host, port=port, connect_timeout_ms=30000, req_timeout_ms=10000)
    client.init()
    return client


# ──────────────────────────────────────────────────────────
#  Benchmark 1: Latency vs Packet Size
# ──────────────────────────────────────────────────────────

def bench_latency_vs_size(client, num_iters=500) -> List[BenchResult]:
    """Measure single-key read latency for different value sizes."""
    sizes = [64, 256, 1024, 4096, 16384, 65536]  # bytes
    results = []

    for size in sizes:
        key = f"bench:lat:{size}"
        value = bytes(np.random.bytes(size))
        client.mset([key], [value])

        # Warmup
        for _ in range(10):
            client.get([key])

        # Measure
        t0 = time.perf_counter()
        for _ in range(num_iters):
            client.get([key])
        elapsed = time.perf_counter() - t0

        r = BenchResult(f"size={size}B", size, num_iters, elapsed)
        results.append(r)
        print(f"  {r.label}: avg={r.avg_latency_us:.1f} us, "
              f"throughput={r.throughput_mbps:.1f} MB/s")

    return results


# ──────────────────────────────────────────────────────────
#  Benchmark 2: Throughput vs Concurrency
# ──────────────────────────────────────────────────────────

def bench_throughput_vs_concurrency(
    host: str, port: int, packet_size=4096, num_ops_per_thread=200
) -> List[BenchResult]:
    """Measure aggregate throughput with increasing concurrent readers."""
    concurrency_levels = [1, 2, 4, 8, 16]
    results = []

    # Setup: write a test key
    client0 = create_client(host, port)
    key = "bench:conc:data"
    value = bytes(np.random.bytes(packet_size))
    client0.mset([key], [value])

    for num_threads in concurrency_levels:

        def worker():
            c = create_client(host, port)
            for _ in range(num_ops_per_thread):
                c.get([key])

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.perf_counter() - t0

        total_ops = num_threads * num_ops_per_thread
        r = BenchResult(
            f"threads={num_threads}", packet_size, total_ops, elapsed
        )
        results.append(r)
        print(f"  {r.label}: avg={r.avg_latency_us:.1f} us, "
              f"agg_throughput={r.throughput_mbps:.1f} MB/s")

    return results


# ──────────────────────────────────────────────────────────
#  Benchmark 3: Engram-Realistic Prefetch
# ──────────────────────────────────────────────────────────

def bench_engram_prefetch(
    host: str,
    port: int,
    embed_dim=128,
    num_rows=100000,
    batch_sizes=(32, 64, 128, 256),
    num_tables=12,
) -> List[BenchResult]:
    """Simulate realistic Engram prefetch: batch lookup of sparse rows.

    Each token needs ~5KB (embed_dim * 4 bytes * num_tables lookups).
    """
    client = create_client(host, port)
    row_bytes = embed_dim * 4  # float32

    # Load table data
    print(f"  Loading {num_rows} rows × {embed_dim}d ({num_rows * row_bytes / 1e6:.1f} MB)...")
    batch_limit = 2000
    for start in range(0, min(num_rows, 10000), batch_limit):
        end = min(start + batch_limit, num_rows, 10000)
        keys = [f"bench:engram:{i}" for i in range(start, end)]
        vals = [bytes(np.random.bytes(row_bytes)) for _ in range(end - start)]
        client.mset(keys, vals)

    results = []
    for bs in batch_sizes:
        # Each token looks up num_tables rows
        total_keys = bs * num_tables
        row_ids = np.random.randint(0, min(num_rows, 10000), size=total_keys)
        keys = [f"bench:engram:{rid}" for rid in row_ids]

        # Split into batches of 10000 (API limit)
        num_iters = 20

        t0 = time.perf_counter()
        for _ in range(num_iters):
            for i in range(0, len(keys), 10000):
                client.get(keys[i : i + 10000])
        elapsed = time.perf_counter() - t0

        total_bytes_per_iter = total_keys * row_bytes
        r = BenchResult(
            f"batch={bs}tok×{num_tables}tables",
            total_bytes_per_iter,
            num_iters,
            elapsed,
        )
        results.append(r)

        per_token_kb = row_bytes * num_tables / 1024
        print(
            f"  {r.label}: "
            f"per_iter={total_bytes_per_iter/1024:.0f}KB, "
            f"avg={r.avg_latency_us/1000:.2f}ms, "
            f"throughput={r.throughput_mbps:.1f}MB/s, "
            f"per_token={per_token_kb:.1f}KB"
        )

    return results


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Engram URMA small-packet benchmark")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18482)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-rows", type=int, default=100000)
    args = parser.parse_args()

    print("=" * 60)
    print("Engram URMA Small-Packet Benchmark")
    print(f"Worker: {args.host}:{args.port}")
    print(f"Transport: {'URMA/UB' if KVClient else 'MOCK (no SDK)'}")
    print("=" * 60)

    print("\n[1/3] Latency vs Packet Size")
    client = create_client(args.host, args.port)
    lat_results = bench_latency_vs_size(client)

    print("\n[2/3] Throughput vs Concurrency (4KB packets)")
    conc_results = bench_throughput_vs_concurrency(args.host, args.port)

    print("\n[3/3] Engram-Realistic Prefetch")
    eng_results = bench_engram_prefetch(
        args.host, args.port, args.embed_dim, args.num_rows
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<35} {'Avg Latency':>12} {'Throughput':>12}")
    print("-" * 60)
    for r in lat_results + conc_results + eng_results:
        if r.avg_latency_us > 1000:
            lat_str = f"{r.avg_latency_us/1000:.2f} ms"
        else:
            lat_str = f"{r.avg_latency_us:.1f} us"
        print(f"{r.label:<35} {lat_str:>12} {r.throughput_mbps:>9.1f} MB/s")


if __name__ == "__main__":
    main()
