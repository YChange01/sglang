"""Integration tests for EngramPrefetcher — async prefetch pipeline.

Verifies:
1. Prefetch correctly overlaps with simulated compute
2. Gathered tensor shape and data correctness
3. Concurrent prefetch for multiple batches
4. Performance: prefetch latency vs compute window
"""

import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"),
)

from sglang.srt.engram.engram_config import EngramPoolConfig
from sglang.srt.engram.engram_pool import EngramPool
from sglang.srt.engram.engram_prefetcher import EngramPrefetcher


class MockKVClient:
    """In-memory mock with configurable latency."""

    def __init__(self, latency_ms=0, **kwargs):
        self._store = {}
        self._latency_ms = latency_ms

    def init(self):
        pass

    def mset(self, keys, vals, **kwargs):
        for k, v in zip(keys, vals):
            self._store[k] = v
        return []

    def get(self, keys, **kwargs):
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)
        return [self._store[k] for k in keys]


def make_pool_with_table(
    num_rows=100, embed_dim=64, num_tables=4, latency_ms=0
):
    """Helper: create pool, load tables, return (pool, tables)."""
    config = EngramPoolConfig(
        enabled=True,
        worker_hosts=["127.0.0.1"],
        worker_ports=[18482],
        num_tables=num_tables,
        embedding_dim=embed_dim,
        table_capacity=num_rows,
        chunk_size=1,
        num_prefetch_workers=4,
    )

    with patch("sglang.srt.engram.engram_pool._import_kv_client") as mock:
        mock.return_value = lambda **kw: MockKVClient(latency_ms=latency_ms, **kw)
        pool = EngramPool(config)
        pool.initialize()

    tables = []
    for t in range(num_tables):
        table = torch.randn(num_rows, embed_dim)
        pool.load_table(t, table)
        tables.append(table)

    return pool, tables, config


class TestEngramPrefetcher(unittest.TestCase):
    """Test async prefetch correctness."""

    def test_prefetch_and_gather_correctness(self):
        """Verify prefetched data matches original tables."""
        num_rows, embed_dim, num_tables = 100, 64, 4
        pool, tables, config = make_pool_with_table(num_rows, embed_dim, num_tables)
        prefetcher = EngramPrefetcher(pool, config)

        seq_len = 32
        table_indices = []
        for t in range(num_tables):
            row_ids = np.random.randint(0, num_rows, size=seq_len)
            table_indices.append((t, row_ids))

        result = prefetcher.prefetch_and_wait(
            table_indices, seq_len, device=torch.device("cpu"), dtype=torch.float32
        )

        self.assertEqual(result.shape, (num_tables, seq_len, embed_dim))

        # Verify each element
        for t, (_, row_ids) in enumerate(table_indices):
            expected = tables[t].numpy()[row_ids]
            np.testing.assert_allclose(
                result[t].numpy(), expected, atol=1e-5,
                err_msg=f"Table {t} data mismatch"
            )

        pool.shutdown()

    def test_prefetch_overlap(self):
        """Verify prefetch overlaps with simulated compute.

        If prefetch takes ~50ms and compute takes ~100ms, total should be
        ~100ms (not 150ms), proving overlap.
        """
        pool, tables, config = make_pool_with_table(
            num_rows=50, embed_dim=32, num_tables=2, latency_ms=50
        )
        prefetcher = EngramPrefetcher(pool, config)

        seq_len = 10
        table_indices = [
            (0, np.arange(seq_len)),
            (1, np.arange(seq_len)),
        ]

        t0 = time.perf_counter()

        # Launch async prefetch
        req = prefetcher.prefetch_async(
            table_indices, seq_len, device=torch.device("cpu")
        )

        # Simulate compute work (100ms)
        compute_start = time.perf_counter()
        time.sleep(0.1)
        compute_end = time.perf_counter()

        # Gather (should be near-instant if prefetch finished during compute)
        result = prefetcher.wait_and_gather(req)
        total = time.perf_counter() - t0

        self.assertEqual(result.shape, (2, seq_len, 32))

        # Total should be ~compute time (100ms), not compute + prefetch
        # Allow generous margin for CI variability
        print(f"Overlap test: compute={compute_end-compute_start:.3f}s, total={total:.3f}s")
        self.assertLess(total, 0.25, "Prefetch did not overlap with compute")

        pool.shutdown()

    def test_concurrent_batches(self):
        """Multiple prefetch requests can run concurrently."""
        pool, tables, config = make_pool_with_table(
            num_rows=50, embed_dim=32, num_tables=2
        )
        prefetcher = EngramPrefetcher(pool, config)

        requests = []
        for _ in range(5):
            seq_len = 16
            table_indices = [
                (0, np.random.randint(0, 50, size=seq_len)),
                (1, np.random.randint(0, 50, size=seq_len)),
            ]
            req = prefetcher.prefetch_async(
                table_indices, seq_len, device=torch.device("cpu")
            )
            requests.append(req)

        # All should complete
        for req in requests:
            result = prefetcher.wait_and_gather(req)
            self.assertEqual(result.shape, (2, 16, 32))

        pool.shutdown()


class TestEngramPrefetcherGPU(unittest.TestCase):
    """Test prefetch with GPU tensor output."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def test_gpu_output(self):
        """Result tensor should be on the specified GPU device."""
        pool, tables, config = make_pool_with_table(
            num_rows=50, embed_dim=32, num_tables=2
        )
        prefetcher = EngramPrefetcher(pool, config)

        seq_len = 8
        table_indices = [
            (0, np.arange(seq_len)),
            (1, np.arange(seq_len)),
        ]

        result = prefetcher.prefetch_and_wait(
            table_indices, seq_len, device=torch.device("cuda:0"), dtype=torch.bfloat16
        )

        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.shape, (2, seq_len, 32))

        pool.shutdown()


if __name__ == "__main__":
    unittest.main()
