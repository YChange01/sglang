"""Unit tests for EngramPool — remote embedding pool client.

These tests verify:
1. Pool initialization and connection to yuanrong-datasystem Workers
2. Table loading (write) correctness
3. Batch lookup (read) correctness and data integrity
4. Sharding consistency
5. Async lookup functionality

Requirements:
- yuanrong-datasystem Worker running on localhost:18482
  (start with: dscli start -w 127.0.0.1:18482)
- yr.datasystem Python SDK installed

For CI without a live Worker, use MockEngramPool (see below).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Add project root to path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"),
)

from sglang.srt.engram.engram_config import EngramPoolConfig


class MockKVClient:
    """In-memory mock of yr.datasystem.KVClient for testing without a Worker."""

    def __init__(self, **kwargs):
        self._store = {}

    def init(self):
        pass

    def mset(self, keys, vals, **kwargs):
        for k, v in zip(keys, vals):
            self._store[k] = v
        return []  # no failures

    def get(self, keys, **kwargs):
        return [self._store[k] for k in keys]


class TestEngramPoolConfig(unittest.TestCase):
    """Test configuration validation."""

    def test_default_config(self):
        config = EngramPoolConfig()
        self.assertFalse(config.enabled)

    def test_validation_no_hosts(self):
        config = EngramPoolConfig(enabled=True, worker_hosts=[], worker_ports=[])
        with self.assertRaises(ValueError):
            config.validate()

    def test_validation_mismatched_hosts_ports(self):
        config = EngramPoolConfig(
            enabled=True,
            worker_hosts=["10.0.0.1", "10.0.0.2"],
            worker_ports=[18482],
        )
        with self.assertRaises(ValueError):
            config.validate()

    def test_validation_ok(self):
        config = EngramPoolConfig(
            enabled=True,
            worker_hosts=["10.0.0.1"],
            worker_ports=[18482],
        )
        config.validate()  # Should not raise

    def test_num_workers(self):
        config = EngramPoolConfig(
            enabled=True,
            worker_hosts=["10.0.0.1", "10.0.0.2", "10.0.0.3"],
            worker_ports=[18482, 18482, 18482],
        )
        self.assertEqual(config.num_workers, 3)


class TestEngramPoolWithMock(unittest.TestCase):
    """Test EngramPool logic using mock KVClient (no live Worker needed)."""

    def setUp(self):
        self.config = EngramPoolConfig(
            enabled=True,
            worker_hosts=["127.0.0.1"],
            worker_ports=[18482],
            num_tables=4,
            embedding_dim=64,
            table_capacity=1000,
            chunk_size=1,
            num_prefetch_workers=2,
        )

    @patch("sglang.srt.engram.engram_pool._import_kv_client")
    def test_load_and_lookup(self, mock_import):
        """Test round-trip: load table → batch lookup → verify data."""
        mock_import.return_value = MockKVClient

        from sglang.srt.engram.engram_pool import EngramPool

        pool = EngramPool(self.config)
        pool.initialize()

        # Create a small embedding table
        num_rows, embed_dim = 100, 64
        table = torch.randn(num_rows, embed_dim)
        pool.load_table(0, table)

        # Verify table is loaded
        self.assertTrue(pool.is_table_loaded(0))
        meta = pool.get_table_meta(0)
        self.assertEqual(meta["num_rows"], 100)
        self.assertEqual(meta["embed_dim"], 64)

        # Batch lookup
        row_ids = np.array([0, 5, 42, 99])
        result = pool.batch_lookup(0, row_ids)

        self.assertEqual(result.shape, (4, 64))

        # Verify correctness
        expected = table.numpy()
        np.testing.assert_allclose(result[0], expected[0], atol=1e-6)
        np.testing.assert_allclose(result[1], expected[5], atol=1e-6)
        np.testing.assert_allclose(result[2], expected[42], atol=1e-6)
        np.testing.assert_allclose(result[3], expected[99], atol=1e-6)

        pool.shutdown()

    @patch("sglang.srt.engram.engram_pool._import_kv_client")
    def test_async_lookup(self, mock_import):
        """Test async batch lookup returns correct results."""
        mock_import.return_value = MockKVClient

        from sglang.srt.engram.engram_pool import EngramPool

        pool = EngramPool(self.config)
        pool.initialize()

        table = torch.randn(50, 64)
        pool.load_table(0, table)

        row_ids = np.array([0, 10, 20, 30, 40, 49])
        future = pool.batch_lookup_async(0, row_ids)
        result = future.result()

        self.assertEqual(result.shape, (6, 64))
        expected = table.numpy()
        for i, rid in enumerate(row_ids):
            np.testing.assert_allclose(result[i], expected[rid], atol=1e-6)

        pool.shutdown()

    @patch("sglang.srt.engram.engram_pool._import_kv_client")
    def test_empty_lookup(self, mock_import):
        """Test lookup with empty row_ids."""
        mock_import.return_value = MockKVClient

        from sglang.srt.engram.engram_pool import EngramPool

        pool = EngramPool(self.config)
        pool.initialize()
        pool.load_table(0, torch.randn(10, 64))

        result = pool.batch_lookup(0, np.array([], dtype=np.int64))
        self.assertEqual(result.shape, (0, 64))

        pool.shutdown()

    @patch("sglang.srt.engram.engram_pool._import_kv_client")
    def test_chunked_storage(self, mock_import):
        """Test with chunk_size > 1 (multiple rows per KV entry)."""
        mock_import.return_value = MockKVClient
        self.config.chunk_size = 4

        from sglang.srt.engram.engram_pool import EngramPool

        pool = EngramPool(self.config)
        pool.initialize()

        table = torch.randn(20, 64)
        pool.load_table(0, table)

        # Fetch rows from different chunks
        row_ids = np.array([1, 5, 13, 19])
        result = pool.batch_lookup(0, row_ids)

        expected = table.numpy()
        for i, rid in enumerate(row_ids):
            np.testing.assert_allclose(result[i], expected[rid], atol=1e-6)

        pool.shutdown()

    @patch("sglang.srt.engram.engram_pool._import_kv_client")
    def test_multi_worker_sharding(self, mock_import):
        """Test that sharding distributes across Workers."""
        mock_import.return_value = MockKVClient
        config = EngramPoolConfig(
            enabled=True,
            worker_hosts=["127.0.0.1", "127.0.0.2"],
            worker_ports=[18482, 18483],
            num_tables=4,
            embedding_dim=32,
            table_capacity=100,
            chunk_size=1,
            shard_strategy="hash",
        )

        from sglang.srt.engram.engram_pool import EngramPool

        pool = EngramPool(config)
        pool.initialize()

        # Verify different chunks map to different workers
        shard0 = pool._shard_for_chunk(0, 0)
        shard1 = pool._shard_for_chunk(0, 1)
        # At least one pair should differ (probabilistic but very likely)
        shards = {pool._shard_for_chunk(0, i) for i in range(10)}
        self.assertTrue(len(shards) > 1, "All chunks mapped to same Worker")

        pool.shutdown()


class TestEngramPoolLive(unittest.TestCase):
    """Integration tests requiring a live yuanrong-datasystem Worker.

    Skip unless ENGRAM_LIVE_TEST=1 is set.
    """

    @classmethod
    def setUpClass(cls):
        if os.environ.get("ENGRAM_LIVE_TEST") != "1":
            raise unittest.SkipTest(
                "Live Worker tests disabled. Set ENGRAM_LIVE_TEST=1 to enable."
            )

    def test_live_round_trip(self):
        """Test with a real Worker at localhost:18482."""
        from sglang.srt.engram.engram_pool import EngramPool

        config = EngramPoolConfig(
            enabled=True,
            worker_hosts=["127.0.0.1"],
            worker_ports=[18482],
            embedding_dim=128,
            chunk_size=1,
            req_timeout_ms=10000,
        )
        pool = EngramPool(config)
        pool.initialize()

        table = torch.randn(500, 128)
        pool.load_table(0, table)

        row_ids = np.array([0, 100, 200, 300, 499])
        result = pool.batch_lookup(0, row_ids)

        expected = table.numpy()
        for i, rid in enumerate(row_ids):
            np.testing.assert_allclose(result[i], expected[rid], atol=1e-6)

        pool.shutdown()

    def test_live_latency(self):
        """Measure single-request latency (should be < 5ms for local Worker)."""
        import time

        from sglang.srt.engram.engram_pool import EngramPool

        config = EngramPoolConfig(
            enabled=True,
            worker_hosts=["127.0.0.1"],
            worker_ports=[18482],
            embedding_dim=128,
            chunk_size=1,
        )
        pool = EngramPool(config)
        pool.initialize()

        table = torch.randn(1000, 128)
        pool.load_table(99, table)

        # Warmup
        pool.batch_lookup(99, np.array([0]))

        # Measure
        latencies = []
        for _ in range(100):
            rid = np.random.randint(0, 1000, size=1)
            t0 = time.perf_counter()
            pool.batch_lookup(99, rid)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(latencies)
        p99_ms = np.percentile(latencies, 99)
        print(f"Latency: avg={avg_ms:.2f}ms, p99={p99_ms:.2f}ms")
        self.assertLess(avg_ms, 10.0, "Average latency too high")

        pool.shutdown()


if __name__ == "__main__":
    unittest.main()
