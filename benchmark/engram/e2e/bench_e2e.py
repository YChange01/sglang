#!/usr/bin/env python3
"""End-to-end Engram + LLM inference benchmark.

Reproduces Table 2 and Table 3 from arXiv:2603.10087.

Commands:
  retrieval — Engram retrieval only (no LLM, pure transport benchmark)
  table2    — Real SGLang + Qwen3-8B inference throughput

Usage:
  # Retrieval only (no GPU needed)
  python bench_e2e.py retrieval --all

  # Real Table 2 with SGLang (requires GPU)
  python bench_e2e.py table2 --model /tmp/g00872988/Qwen3-8B

Environment variables:
  SERVER_IP, TCP_PORT, URMA_PORT, PROVIDER_HOST, SHM_NAME
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np

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

E2E_BACKENDS = ["local", "urma", "ubsmem"]


# ================================================================== #
#  Engram retrieval benchmark (no LLM)                                #
# ================================================================== #


def bench_retrieval(
    backend: EngramBackend,
    config: EngramConfig,
    batch_sizes: List[int],
    num_iters: int = 100,
) -> Dict[int, dict]:
    results = {}
    for batch_size in batch_sizes:
        # Warmup
        token_ids = np.random.randint(0, config.vocab_size, size=batch_size, dtype=np.int32)
        for _ in range(3):
            backend.fetch_all_tables(config, token_ids)

        latencies = []
        for _ in range(num_iters):
            token_ids = np.random.randint(0, config.vocab_size, size=batch_size, dtype=np.int32)
            t0 = time.perf_counter()
            backend.fetch_all_tables(config, token_ids)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

        avg_ms = np.mean(latencies)
        p50_ms = np.percentile(latencies, 50)
        p99_ms = np.percentile(latencies, 99)
        data_bytes = batch_size * config.num_tables * config.row_bytes
        throughput_mbs = (data_bytes / 1e6) / (avg_ms / 1000) if avg_ms > 0 else 0
        tokens_per_sec = batch_size / (avg_ms / 1000) if avg_ms > 0 else 0

        results[batch_size] = {
            "avg_ms": avg_ms, "p50_ms": p50_ms, "p99_ms": p99_ms,
            "data_mb": data_bytes / 1e6,
            "throughput_mbs": throughput_mbs,
            "tokens_per_sec": tokens_per_sec,
        }
    return results


def print_retrieval_results(name: str, results: Dict[int, dict], config: EngramConfig) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Engram Retrieval: {name}")
    print(f"  {config.num_tables} tables x {config.embedding_dim} dim, vocab={config.vocab_size}")
    print(f"{'=' * 60}")
    print(f"  {'Batch':>8}  {'Avg ms':>10}  {'P50 ms':>10}  {'P99 ms':>10}  "
          f"{'Data MB':>10}  {'MB/s':>10}  {'tok/s':>12}")
    print(f"  {'-' * 78}")
    for batch, r in sorted(results.items()):
        print(f"  {batch:>8}  {r['avg_ms']:>10.3f}  {r['p50_ms']:>10.3f}  "
              f"{r['p99_ms']:>10.3f}  {r['data_mb']:>10.2f}  "
              f"{r['throughput_mbs']:>10.1f}  {r['tokens_per_sec']:>12.0f}")


def run_retrieval(args: argparse.Namespace) -> None:
    config = EngramConfig(
        num_tables=args.num_tables, embedding_dim=args.dim, vocab_size=args.vocab_size,
    )
    batch_sizes = [1, 4, 16, 64, 128, 256]
    backends_to_test = E2E_BACKENDS if args.all else [args.backend]

    for backend_name in backends_to_test:
        kwargs = _backend_kwargs(backend_name, args)
        backend = create_backend(backend_name, **kwargs)
        try:
            backend.setup(config)
            results = bench_retrieval(backend, config, batch_sizes, args.iters)
            print_retrieval_results(backend_name.upper(), results, config)
        except Exception as e:
            print(f"\n  {backend_name.upper()}: FAILED - {e}")
        finally:
            backend.teardown()



# ================================================================== #
#  Table 2 real: SGLang + Qwen3-8B                                    #
# ================================================================== #

SGLANG_PORT = 30000
WARMUP_REQUESTS = 5
BENCH_REQUESTS = 50


def wait_for_server(port: int, timeout: int = 300) -> bool:
    """Wait for SGLang server to be ready."""
    import urllib.request
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def send_requests(
    port: int, prompts: List[str], max_tokens: int = 128
) -> List[dict]:
    """Send completion requests to SGLang server, return timing info."""
    import urllib.request
    results = []
    url = f"http://localhost:{port}/v1/completions"

    for prompt in prompts:
        payload = json.dumps({
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()

        t0 = time.perf_counter()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        t1 = time.perf_counter()

        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        results.append({
            "elapsed_s": t1 - t0,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        })
    return results


def measure_throughput(port: int, num_requests: int, max_tokens: int = 128) -> dict:
    """Run benchmark requests and return throughput stats."""
    prompts = [
        "Explain the concept of memory pooling in distributed systems.",
        "What are the advantages of CXL over traditional RDMA?",
        "Describe how N-gram embeddings work in language models.",
        "What is the difference between load/store and DMA access?",
        "How does hardware page table translation enable remote memory access?",
    ]
    # Cycle through prompts
    test_prompts = [prompts[i % len(prompts)] for i in range(num_requests)]

    # Warmup
    logger.info("Warmup: %d requests...", WARMUP_REQUESTS)
    send_requests(port, test_prompts[:WARMUP_REQUESTS], max_tokens)

    # Benchmark
    logger.info("Benchmark: %d requests, max_tokens=%d...", num_requests, max_tokens)
    t0 = time.perf_counter()
    results = send_requests(port, test_prompts, max_tokens)
    total_time = time.perf_counter() - t0

    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_tokens = sum(r["total_tokens"] for r in results)
    throughput = total_completion_tokens / total_time if total_time > 0 else 0

    return {
        "total_time_s": total_time,
        "num_requests": num_requests,
        "total_tokens": total_tokens,
        "completion_tokens": total_completion_tokens,
        "throughput_tps": throughput,
        "avg_latency_s": total_time / num_requests,
    }


def launch_sglang(model_path: str, port: int, extra_args: List[str] = None) -> subprocess.Popen:
    """Launch SGLang server as subprocess."""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
    ]
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Launching: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc


def kill_server(proc: subprocess.Popen) -> None:
    """Kill SGLang server."""
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_table2_real(args: argparse.Namespace) -> None:
    """Real Table 2: launch SGLang, benchmark, compare."""
    port = args.port
    model = args.model
    num_requests = args.num_requests
    max_tokens = args.max_tokens

    configs = [
        ("Baseline (no Engram)", []),
        ("+Engram (LOCAL/DRAM)", [
            "--engram-pool-enabled",
            "--engram-pool-backend", "local",
        ]),
    ]

    # Add UBS-MEM config if server is running
    server_ip = getattr(args, "server_ip", None) or os.environ.get("SERVER_IP", "192.168.84.245")
    configs.append(("+Engram (UBS-MEM)", [
        "--engram-pool-enabled",
        "--engram-pool-backend", "ubsmem",
        "--engram-pool-server-ip", server_ip,
    ]))

    print(f"\n{'=' * 70}")
    print(f"  Table 2: Real LLM Throughput with Engram")
    print(f"  Model: {model}")
    print(f"  Requests: {num_requests}, max_tokens: {max_tokens}")
    print(f"{'=' * 70}")
    print(f"  {'Configuration':<30}  {'Throughput':>12}  {'Avg Latency':>12}  {'Overhead':>10}")
    print(f"  {'-' * 70}")

    baseline_tps = None

    for config_name, extra_args in configs:
        proc = None
        try:
            proc = launch_sglang(model, port, extra_args)
            logger.info("Waiting for server to be ready...")
            if not wait_for_server(port, timeout=300):
                print(f"  {config_name:<30}  FAILED: server timeout")
                continue

            result = measure_throughput(port, num_requests, max_tokens)
            tps = result["throughput_tps"]
            avg_lat = result["avg_latency_s"]

            if baseline_tps is None:
                baseline_tps = tps
                overhead_str = "0%"
            else:
                overhead = (1 - tps / baseline_tps) * 100 if baseline_tps > 0 else 0
                overhead_str = f"{overhead:.1f}%"

            print(f"  {config_name:<30}  {tps:>10.1f} t/s  {avg_lat:>10.2f} s  "
                  f"{overhead_str:>10}")

        except Exception as e:
            print(f"  {config_name:<30}  FAILED: {e}")
        finally:
            if proc:
                kill_server(proc)
                time.sleep(5)  # wait for port release


# ================================================================== #
#  Helpers                                                            #
# ================================================================== #


def _backend_kwargs(name: str, args: argparse.Namespace) -> dict:
    server_ip = getattr(args, "server_ip", None) or os.environ.get("SERVER_IP", "192.168.84.245")
    if name == "tcp":
        return {"server_ip": server_ip, "port": int(os.environ.get("TCP_PORT", "13900"))}
    elif name == "ubsmem":
        return {
            "shm_name": os.environ.get("SHM_NAME", "engram_test"),
            "provider_host": os.environ.get("PROVIDER_HOST", "node1"),
        }
    elif name == "urma":
        return {"server_ip": server_ip, "port": int(os.environ.get("URMA_PORT", "13857"))}
    return {}


# ================================================================== #
#  Main                                                               #
# ================================================================== #


def main():
    parser = argparse.ArgumentParser(description="Engram E2E Benchmark")
    sub = parser.add_subparsers(dest="command")

    # --- retrieval ---
    pr = sub.add_parser("retrieval", help="Engram retrieval only (no LLM)")
    pr.add_argument("--backend", default="local", choices=list(BACKENDS))
    pr.add_argument("--all", action="store_true")
    pr.add_argument("--num_tables", type=int, default=12)
    pr.add_argument("--dim", type=int, default=341)
    pr.add_argument("--vocab_size", type=int, default=10000)
    pr.add_argument("--iters", type=int, default=100)
    pr.add_argument("--server_ip", default=None)

    # --- table2 (real) ---
    p2 = sub.add_parser("table2", help="Real Table 2: SGLang + Qwen3-8B")
    p2.add_argument("--model", default="/tmp/g00872988/Qwen3-8B")
    p2.add_argument("--port", type=int, default=SGLANG_PORT)
    p2.add_argument("--num_requests", type=int, default=50)
    p2.add_argument("--max_tokens", type=int, default=128)
    p2.add_argument("--server_ip", default=None)

    args = parser.parse_args()

    if args.command == "retrieval":
        run_retrieval(args)
    elif args.command == "table2":
        run_table2_real(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
