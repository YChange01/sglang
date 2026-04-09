#!/usr/bin/env python3
"""Unified benchmark comparing all 5 cross-node Engram read schemes.

Schemes:
  1. KV TCP:       KVClient → TCP RPC → Worker B → TCP RPC → Worker A
  2. KV URMA:      KVClient → TCP RPC → Worker B → urma_write → Worker A
  3. KV IPC+URMA:  KVClient → IPC → Worker B → urma_write → Worker A
  4. SHM Prefetch: ObjectClient → memoryview (data pre-fetched to local)
  5. URMA MMAP:    liburma_mmap → load/store → remote memory via UBMMU

Usage:
  # Single node (schemes 1, 4 only)
  python bench_all_schemes.py --node1-host 141.61.84.245 --node1-port 18483

  # Two nodes (all 5 schemes)
  python bench_all_schemes.py \
    --node1-host 10.0.0.1 --node1-port 18482 \
    --node2-host 10.0.0.2 --node2-port 18482 \
    --seg-info seg_info.json  # for scheme 5

  # Specific scheme only
  python bench_all_schemes.py --scheme 5 --seg-info seg_info.json
"""

import argparse
import time
from typing import List, Optional

import numpy as np


# ------------------------------------------------------------------ #
#  Common parameters
# ------------------------------------------------------------------ #

NUM_TABLES = 12
DIM = 341
ROW_BYTES = DIM * 4  # float32


def generate_row_ids(batch_tokens: int, max_rows: int, num_tables: int = NUM_TABLES):
    """Generate random row IDs simulating N-gram hash output."""
    return [(t, np.random.randint(0, max_rows, size=batch_tokens))
            for t in range(num_tables)]


def measure(func, num_iters=50, warmup=5, label=""):
    """Run func num_iters times, report avg/p99 latency."""
    for _ in range(warmup):
        func()
    latencies = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        func()
        latencies.append((time.perf_counter() - t0) * 1e6)
    avg = np.mean(latencies)
    p99 = np.percentile(latencies, 99)
    return avg, p99


# ------------------------------------------------------------------ #
#  Scheme 1: KV TCP
# ------------------------------------------------------------------ #

def bench_scheme1(host, port, max_rows, batch_sizes, num_iters):
    print("\n" + "=" * 60)
    print("[Scheme 1] KV TCP (baseline)")
    print("=" * 60)
    from yr.datasystem import KVClient
    c = KVClient(host=host, port=port, timeout_ms=60000, req_timeout_ms=10000)
    c.init()

    # Write test data
    _load_kv_data(c, max_rows)
    _run_kv_bench(c, max_rows, batch_sizes, num_iters)


def bench_scheme2(host, port, max_rows, batch_sizes, num_iters):
    print("\n" + "=" * 60)
    print("[Scheme 2] KV TCP + Worker-URMA (connect to remote-data Worker)")
    print("=" * 60)
    from yr.datasystem import KVClient
    c = KVClient(host=host, port=port, timeout_ms=60000, req_timeout_ms=10000)
    c.init()
    _run_kv_bench(c, max_rows, batch_sizes, num_iters)


def bench_scheme3(host, port, max_rows, batch_sizes, num_iters):
    """IPC mode — only works when client is on the same node as the Worker."""
    print("\n" + "=" * 60)
    print("[Scheme 3] KV IPC (same-node shared memory)")
    print("=" * 60)
    from yr.datasystem import KVClient
    c = KVClient(host=host, port=port, timeout_ms=60000, req_timeout_ms=10000,
                 enable_exclusive_connection=True)
    c.init()
    _run_kv_bench(c, max_rows, batch_sizes, num_iters)


def _load_kv_data(client, max_rows):
    """Load test data into KV store."""
    print(f"  Loading {max_rows} rows x {DIM}d per table...")
    for t in range(NUM_TABLES):
        keys, vals = [], []
        for start in range(0, max_rows, 2000):
            end = min(start + 2000, max_rows)
            chunk = np.random.randn(end - start, DIM).astype(np.float32)
            keys = [f"engram:t{t}:c{r}" for r in range(start, end)]
            vals = [chunk[i].tobytes() for i in range(end - start)]
            client.mset(keys, vals)
    print("  Data loaded.")


def _run_kv_bench(client, max_rows, batch_sizes, num_iters):
    # Single row
    def single_read():
        rid = np.random.randint(0, max_rows)
        client.get([f"engram:t0:c{rid}"])

    avg, p99 = measure(single_read, num_iters * 10, label="single")
    print(f"  Single row:  avg={avg:.1f} us, p99={p99:.1f} us")

    # Batch prefetch
    print(f"  {'Batch':>8} {'Avg ms':>10} {'Throughput':>12}")
    for bs in batch_sizes:
        def batch_read():
            for t in range(NUM_TABLES):
                rids = np.random.randint(0, max_rows, size=bs)
                keys = [f"engram:t{t}:c{r}" for r in rids]
                client.get(keys)

        avg, _ = measure(batch_read, num_iters)
        data_kb = NUM_TABLES * bs * ROW_BYTES / 1024
        tp = (data_kb / 1024) / (avg / 1e6) if avg > 0 else 0
        print(f"  {bs:>8} {avg/1000:>9.2f} {tp:>9.1f} MB/s")


# ------------------------------------------------------------------ #
#  Scheme 4: ObjectClient SHM
# ------------------------------------------------------------------ #

def bench_scheme4(host, port, max_rows, batch_sizes, num_iters):
    print("\n" + "=" * 60)
    print("[Scheme 4] ObjectClient SHM (memoryview direct read)")
    print("=" * 60)
    from yr.datasystem import ObjectClient

    client = ObjectClient(host=host, port=port, timeout_ms=60000)
    client.init()

    # Load tables as large contiguous buffers
    print(f"  Loading {NUM_TABLES} tables ({max_rows} rows x {DIM}d each)...")
    for t in range(NUM_TABLES):
        key = f"engram_shm:t{t}"
        data = np.random.randn(max_rows, DIM).astype(np.float32).tobytes()
        client.put(key, data)
        client.g_increase_ref([key])  # ref after put

    # Open memoryviews
    keys = [f"engram_shm:t{t}" for t in range(NUM_TABLES)]
    buffers = client.get(keys, timeout_ms=10000)
    mvs = [buf.immutable_data() for buf in buffers]
    # Wrap as numpy
    np_tables = [np.frombuffer(mv, dtype=np.float32).reshape(-1, DIM) for mv in mvs]
    print(f"  Mapped {NUM_TABLES} tables, {max_rows} rows each")

    # Single row
    def single_read():
        rid = np.random.randint(0, max_rows)
        _ = np_tables[0][rid].copy()

    avg, p99 = measure(single_read, num_iters * 10)
    print(f"  Single row:  avg={avg:.2f} us, p99={p99:.2f} us")

    # Batch prefetch
    print(f"  {'Batch':>8} {'Avg ms':>10} {'Throughput':>12}")
    for bs in batch_sizes:
        def batch_read():
            for t in range(NUM_TABLES):
                rids = np.random.randint(0, max_rows, size=bs)
                _ = np_tables[t][rids]

        avg, _ = measure(batch_read, num_iters)
        data_kb = NUM_TABLES * bs * ROW_BYTES / 1024
        tp = (data_kb / 1024) / (avg / 1e6) if avg > 0 else 0
        print(f"  {bs:>8} {avg/1000:>9.3f} {tp:>9.1f} MB/s")


# ------------------------------------------------------------------ #
#  Scheme 5: URMA MMAP
# ------------------------------------------------------------------ #

def bench_scheme5(seg_info_path, batch_sizes, num_iters):
    print("\n" + "=" * 60)
    print("[Scheme 5] URMA MMAP (load/store via UBMMU)")
    print("=" * 60)

    from scheme5_urma_mmap.urma_mmap_py import UrmaMmapClient, SegInfo, bench_mmap_read

    info = SegInfo.load(seg_info_path)
    client = UrmaMmapClient()
    table = client.mmap_import(info)
    print(f"  Mapped: {table.shape}, row[0][:3]={table[0][:3]}")

    # Single row
    nrows = table.shape[0]
    def single_read():
        rid = np.random.randint(0, nrows)
        _ = table[rid].copy()

    avg, p99 = measure(single_read, num_iters * 10)
    print(f"  Single row:  avg={avg:.2f} us, p99={p99:.2f} us")

    # Batch (simulate 12 tables reading from same mapped region)
    print(f"  {'Batch':>8} {'Avg ms':>10} {'Throughput':>12}")
    for bs in batch_sizes:
        def batch_read():
            for _ in range(NUM_TABLES):
                rids = np.random.randint(0, nrows, size=bs)
                _ = table[rids]

        avg, _ = measure(batch_read, num_iters)
        data_kb = NUM_TABLES * bs * ROW_BYTES / 1024
        tp = (data_kb / 1024) / (avg / 1e6) if avg > 0 else 0
        print(f"  {bs:>8} {avg/1000:>9.3f} {tp:>9.1f} MB/s")

    client.destroy()


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="5-scheme Engram benchmark")
    parser.add_argument("--node1-host", default="127.0.0.1", help="Data node Worker IP")
    parser.add_argument("--node1-port", type=int, default=18482, help="Data node Worker port")
    parser.add_argument("--node2-host", default="", help="Compute node Worker IP (for cross-node)")
    parser.add_argument("--node2-port", type=int, default=18482)
    parser.add_argument("--seg-info", default="", help="seg_info.json path (for scheme 5)")
    parser.add_argument("--max-rows", type=int, default=10000)
    parser.add_argument("--batch-sizes", default="32,128,256")
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--scheme", type=int, default=0, help="Run specific scheme (0=all)")
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    host = args.node1_host
    port = args.node1_port
    cross_host = args.node2_host or host
    cross_port = args.node2_port

    print("=" * 60)
    print("Engram 5-Scheme Cross-Node Benchmark")
    print("=" * 60)
    print(f"Node1 (data):    {host}:{port}")
    print(f"Node2 (compute): {cross_host}:{cross_port}")
    print(f"Tables: {NUM_TABLES}, Dim: {DIM}, MaxRows: {args.max_rows}")
    print(f"Batch sizes: {batch_sizes}")

    run_all = args.scheme == 0
    results = {}

    if run_all or args.scheme == 1:
        bench_scheme1(host, port, args.max_rows, batch_sizes, args.num_iters)

    if run_all or args.scheme == 2:
        if args.node2_host:
            bench_scheme2(cross_host, cross_port, args.max_rows, batch_sizes, args.num_iters)
        else:
            print("\n[Scheme 2] SKIPPED (need --node2-host for cross-node)")

    if run_all or args.scheme == 3:
        bench_scheme3(host, port, args.max_rows, batch_sizes, args.num_iters)

    if run_all or args.scheme == 4:
        bench_scheme4(host, port, args.max_rows, batch_sizes, args.num_iters)

    if run_all or args.scheme == 5:
        if args.seg_info:
            bench_scheme5(args.seg_info, batch_sizes, args.num_iters)
        else:
            print("\n[Scheme 5] SKIPPED (need --seg-info for URMA MMAP)")

    print("\n" + "=" * 60)
    print("Done. Compare single-row latency across schemes:")
    print("  Scheme 1 (KV TCP):     ~150 us (RPC bound)")
    print("  Scheme 2 (KV URMA):    ~150 us (RPC bound, urma_write internal)")
    print("  Scheme 3 (KV IPC):     ~100 us (IPC bound)")
    print("  Scheme 4 (SHM):        ~0.5 us (memoryview direct)")
    print("  Scheme 5 (URMA MMAP):  ~0.5 us (load/store via UBMMU)")
    print("=" * 60)


if __name__ == "__main__":
    main()
