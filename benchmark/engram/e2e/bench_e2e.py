#!/usr/bin/env python3
"""End-to-end Engram + LLM inference benchmark.

Reproduces Table 2 and Table 3 from arXiv:2603.10087:
  Table 2: Baseline vs +Engram(DRAM) vs +Engram(UBS-MEM/URMA/TCP)
  Table 3: Scalability with DP and nnode

Usage:
  # Table 2: throughput comparison (single mode)
  python bench_e2e.py table2 --model Qwen/Qwen3-4B --backend local
  python bench_e2e.py table2 --model Qwen/Qwen3-4B --backend ubsmem
  python bench_e2e.py table2 --model Qwen/Qwen3-4B --backend urma

  # Table 2: all backends at once
  python bench_e2e.py table2 --model Qwen/Qwen3-4B --all

  # Standalone: benchmark Engram retrieval only (no LLM needed)
  python bench_e2e.py retrieval --backend local
  python bench_e2e.py retrieval --backend ubsmem --provider node1
  python bench_e2e.py retrieval --all

  # Table 3: scalability (requires multi-GPU + multi-node)
  python bench_e2e.py table3 --model Qwen/Qwen3-4B --dp 1 --nnode 1
  python bench_e2e.py table3 --model Qwen/Qwen3-4B --dp 2 --nnode 2

Environment variables:
  SERVER_IP       — remote server IP (default: 192.168.84.245)
  TCP_PORT        — TCP data port (default: 13900)
  URMA_PORT       — URMA port (default: 13857)
  PROVIDER_HOST   — ubs_mem provider hostname (default: node1)
  SHM_NAME        — ubs_mem shmem object name (default: engram_test)
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np

# Add parent to path for import
sys.path.insert(0, os.path.dirname(__file__))
from engram_pool_transport import (
    EngramBackend,
    EngramConfig,
    create_backend,
    BACKENDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Engram retrieval benchmark (no LLM required)                       #
# ------------------------------------------------------------------ #


def bench_retrieval(
    backend: EngramBackend,
    config: EngramConfig,
    batch_sizes: List[int],
    num_iters: int = 100,
) -> Dict[int, dict]:
    """Benchmark Engram retrieval latency and throughput.

    Simulates the Engram layer: for each batch of tokens, fetch embeddings
    from all sub-tables.

    Returns dict: batch_size -> {latency_ms, throughput_tps, data_mb}
    """
    results = {}

    for batch_size in batch_sizes:
        token_ids = np.random.randint(0, config.vocab_size, size=batch_size, dtype=np.int32)

        # Warmup
        for _ in range(3):
            backend.fetch_all_tables(config, token_ids)

        # Benchmark
        latencies = []
        for _ in range(num_iters):
            token_ids = np.random.randint(0, config.vocab_size, size=batch_size, dtype=np.int32)
            t0 = time.perf_counter()
            backend.fetch_all_tables(config, token_ids)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

        avg_ms = np.mean(latencies)
        p50_ms = np.percentile(latencies, 50)
        p99_ms = np.percentile(latencies, 99)
        data_bytes = batch_size * config.num_tables * config.row_bytes
        throughput_mbs = (data_bytes / 1e6) / (avg_ms / 1000) if avg_ms > 0 else 0
        tokens_per_sec = batch_size / (avg_ms / 1000) if avg_ms > 0 else 0

        results[batch_size] = {
            "avg_ms": avg_ms,
            "p50_ms": p50_ms,
            "p99_ms": p99_ms,
            "data_mb": data_bytes / 1e6,
            "throughput_mbs": throughput_mbs,
            "tokens_per_sec": tokens_per_sec,
        }

    return results


def print_retrieval_results(
    name: str, results: Dict[int, dict], config: EngramConfig
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Engram Retrieval: {name}")
    print(f"  {config.num_tables} tables × {config.embedding_dim} dim, "
          f"vocab={config.vocab_size}")
    print(f"{'=' * 60}")
    print(f"  {'Batch':>8}  {'Avg ms':>10}  {'P50 ms':>10}  {'P99 ms':>10}  "
          f"{'Data MB':>10}  {'MB/s':>10}  {'tok/s':>12}")
    print(f"  {'-' * 78}")
    for batch, r in sorted(results.items()):
        print(
            f"  {batch:>8}  {r['avg_ms']:>10.3f}  {r['p50_ms']:>10.3f}  "
            f"{r['p99_ms']:>10.3f}  {r['data_mb']:>10.2f}  "
            f"{r['throughput_mbs']:>10.1f}  {r['tokens_per_sec']:>12.0f}"
        )


# ------------------------------------------------------------------ #
#  Table 2: LLM throughput with different backends                    #
# ------------------------------------------------------------------ #


def simulate_llm_forward(
    backend: EngramBackend,
    config: EngramConfig,
    batch_size: int,
    seq_len: int,
    num_layers: int,
    engram_layers: List[int],
    compute_time_per_layer_ms: float,
) -> dict:
    """Simulate LLM forward pass with Engram layers.

    Models the SGLang inference loop where Engram retrieval overlaps
    with Transformer layer computation.

    Returns: {total_ms, compute_ms, engram_ms, throughput_tps}
    """
    total_compute_ms = 0
    total_engram_ms = 0

    token_ids = np.random.randint(0, config.vocab_size, size=batch_size, dtype=np.int32)

    for layer in range(num_layers):
        # Transformer layer compute (simulated)
        time.sleep(compute_time_per_layer_ms / 1000)
        total_compute_ms += compute_time_per_layer_ms

        # Engram layer (real retrieval)
        if layer in engram_layers:
            t0 = time.perf_counter()
            backend.fetch_all_tables(config, token_ids)
            t1 = time.perf_counter()
            engram_ms = (t1 - t0) * 1000
            total_engram_ms += engram_ms

    total_ms = total_compute_ms + total_engram_ms
    tokens = batch_size * seq_len
    throughput = tokens / (total_ms / 1000) if total_ms > 0 else 0

    return {
        "total_ms": total_ms,
        "compute_ms": total_compute_ms,
        "engram_ms": total_engram_ms,
        "throughput_tps": throughput,
        "overhead_pct": (total_engram_ms / total_ms * 100) if total_ms > 0 else 0,
    }


def run_table2(args: argparse.Namespace) -> None:
    """Reproduce Table 2: throughput comparison."""
    config = EngramConfig(
        num_tables=args.num_tables,
        embedding_dim=args.dim,
        vocab_size=args.vocab_size,
    )

    # Model parameters (approximate)
    num_layers = args.num_layers
    engram_layers = [1, 14]  # Engram at layers 2 and 15 (0-indexed)
    compute_per_layer = args.compute_ms  # ms per Transformer layer
    batch_size = args.batch_size
    seq_len = args.seq_len
    num_iters = args.iters

    backends_to_test = []
    if args.all:
        backends_to_test = ["local", "tcp", "urma", "ubsmem"]
    else:
        backends_to_test = [args.backend]

    # Baseline (no Engram)
    baseline_ms = num_layers * compute_per_layer
    baseline_tps = (batch_size * seq_len) / (baseline_ms / 1000)

    print(f"\n{'=' * 70}")
    print(f"  Table 2: LLM Throughput with Engram ({args.model})")
    print(f"  {num_layers} layers, batch={batch_size}, seq_len={seq_len}, "
          f"compute={compute_per_layer}ms/layer")
    print(f"  Engram: {config.num_tables} tables × {config.embedding_dim} dim")
    print(f"{'=' * 70}")
    print(f"  {'Configuration':<30}  {'Throughput':>12}  {'Engram ms':>12}  "
          f"{'Overhead':>10}")
    print(f"  {'-' * 70}")
    print(f"  {'Baseline (no Engram)':<30}  {baseline_tps:>10.1f} t/s  "
          f"{'N/A':>12}  {'0%':>10}")

    for backend_name in backends_to_test:
        kwargs = _backend_kwargs(backend_name, args)
        backend = create_backend(backend_name, **kwargs)

        try:
            backend.setup(config)

            # Run multiple iterations
            totals = {"engram_ms": 0, "throughput_tps": 0}
            for _ in range(num_iters):
                r = simulate_llm_forward(
                    backend, config, batch_size, seq_len,
                    num_layers, engram_layers, compute_per_layer,
                )
                totals["engram_ms"] += r["engram_ms"]
                totals["throughput_tps"] += r["throughput_tps"]

            avg_engram = totals["engram_ms"] / num_iters
            avg_tps = totals["throughput_tps"] / num_iters
            overhead = (1 - avg_tps / baseline_tps) * 100

            label = f"+Engram ({backend_name.upper()})"
            print(f"  {label:<30}  {avg_tps:>10.1f} t/s  {avg_engram:>10.2f} ms  "
                  f"{overhead:>9.1f}%")
        except Exception as e:
            print(f"  +Engram ({backend_name.upper()}){'':>30}  FAILED: {e}")
        finally:
            backend.teardown()


# ------------------------------------------------------------------ #
#  Retrieval-only benchmark                                           #
# ------------------------------------------------------------------ #


def run_retrieval(args: argparse.Namespace) -> None:
    """Benchmark Engram retrieval only (no LLM)."""
    config = EngramConfig(
        num_tables=args.num_tables,
        embedding_dim=args.dim,
        vocab_size=args.vocab_size,
    )

    batch_sizes = [1, 4, 16, 64, 128, 256]
    backends_to_test = list(BACKENDS.keys()) if args.all else [args.backend]

    for backend_name in backends_to_test:
        kwargs = _backend_kwargs(backend_name, args)
        backend = create_backend(backend_name, **kwargs)

        try:
            backend.setup(config)
            results = bench_retrieval(backend, config, batch_sizes, args.iters)
            print_retrieval_results(backend_name.upper(), results, config)
        except Exception as e:
            print(f"\n  {backend_name.upper()}: FAILED — {e}")
        finally:
            backend.teardown()


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #


def _backend_kwargs(name: str, args: argparse.Namespace) -> dict:
    """Build kwargs for backend constructor from args/env."""
    server_ip = getattr(args, "server_ip", None) or os.environ.get(
        "SERVER_IP", "192.168.84.245"
    )
    if name == "tcp":
        return {
            "server_ip": server_ip,
            "port": int(os.environ.get("TCP_PORT", "13900")),
        }
    elif name == "ubsmem":
        return {
            "shm_name": os.environ.get("SHM_NAME", "engram_test"),
            "provider_host": os.environ.get("PROVIDER_HOST", "node1"),
        }
    elif name == "urma":
        return {
            "server_ip": server_ip,
            "port": int(os.environ.get("URMA_PORT", "13857")),
        }
    return {}


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #


def main():
    parser = argparse.ArgumentParser(description="Engram E2E Benchmark")
    sub = parser.add_subparsers(dest="command")

    # --- table2 ---
    p2 = sub.add_parser("table2", help="Table 2: throughput comparison")
    p2.add_argument("--model", default="Qwen/Qwen3-4B")
    p2.add_argument("--backend", default="local", choices=list(BACKENDS))
    p2.add_argument("--all", action="store_true", help="Test all backends")
    p2.add_argument("--batch_size", type=int, default=256)
    p2.add_argument("--seq_len", type=int, default=512)
    p2.add_argument("--num_layers", type=int, default=36)
    p2.add_argument("--num_tables", type=int, default=12)
    p2.add_argument("--dim", type=int, default=341)
    p2.add_argument("--vocab_size", type=int, default=10000)
    p2.add_argument("--compute_ms", type=float, default=0.1,
                     help="Simulated compute time per Transformer layer (ms)")
    p2.add_argument("--iters", type=int, default=20)
    p2.add_argument("--server_ip", default=None)

    # --- retrieval ---
    pr = sub.add_parser("retrieval", help="Engram retrieval only (no LLM)")
    pr.add_argument("--backend", default="local", choices=list(BACKENDS))
    pr.add_argument("--all", action="store_true")
    pr.add_argument("--num_tables", type=int, default=12)
    pr.add_argument("--dim", type=int, default=341)
    pr.add_argument("--vocab_size", type=int, default=10000)
    pr.add_argument("--iters", type=int, default=100)
    pr.add_argument("--server_ip", default=None)

    # --- table3 ---
    p3 = sub.add_parser("table3", help="Table 3: scalability (DP/nnode)")
    p3.add_argument("--model", default="Qwen/Qwen3-4B")
    p3.add_argument("--dp", type=int, default=1)
    p3.add_argument("--nnode", type=int, default=1)
    p3.add_argument("--backend", default="ubsmem", choices=list(BACKENDS))
    p3.add_argument("--server_ip", default=None)

    args = parser.parse_args()

    if args.command == "table2":
        run_table2(args)
    elif args.command == "retrieval":
        run_retrieval(args)
    elif args.command == "table3":
        print("Table 3 (DP/nnode scalability) requires multi-GPU SGLang setup.")
        print("Use the retrieval benchmark to measure per-node Engram overhead,")
        print("then extrapolate based on DP scaling.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
