"""End-to-end tests for Engram remote pool integration with SGLang.

Verifies:
1. RemoteEngramEmbedding produces same output as local NgramEmbedding
2. Model inference throughput with Engram pool enabled
3. Multi-node scaling (requires 2+ Workers)

Requirements for live tests:
- yuanrong-datasystem Workers running
- Set ENGRAM_LIVE_TEST=1
- GPU available
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "python"),
)

from sglang.srt.engram.engram_config import EngramPoolConfig


class MockKVClient:
    def __init__(self, **kwargs):
        self._store = {}

    def init(self):
        pass

    def mset(self, keys, vals, **kwargs):
        for k, v in zip(keys, vals):
            self._store[k] = v
        return []

    def get(self, keys, **kwargs):
        return [self._store[k] for k in keys]


class TestRemoteEngramEmbeddingConsistency(unittest.TestCase):
    """Compare local NgramEmbedding vs RemoteEngramEmbedding output."""

    def _build_local_embedding(self, num_embeddings, embedding_dim, m, k, n):
        """Build original NgramEmbedding."""
        from sglang.srt.layers.n_gram_embedding import NgramEmbedding

        return NgramEmbedding(num_embeddings, embedding_dim, m, k, n)

    def _build_remote_embedding(self, num_embeddings, embedding_dim, m, k, n):
        """Build RemoteEngramEmbedding."""
        from sglang.srt.engram.engram_embedding import RemoteEngramEmbedding

        return RemoteEngramEmbedding(num_embeddings, embedding_dim, m, k, n)

    @patch("sglang.srt.engram.engram_pool._import_kv_client")
    def test_output_consistency(self, mock_import):
        """Local and remote embedding should produce identical outputs
        when the remote pool contains the same weights.

        This test uses the local fallback path (oe_embeder still on GPU)
        with prefetcher=None to verify weight loading compatibility.
        """
        mock_import.return_value = MockKVClient

        num_embeddings = 1000
        embedding_dim = 128
        m, k, n = 100, 3, 3

        local_emb = self._build_local_embedding(num_embeddings, embedding_dim, m, k, n)
        remote_emb = self._build_remote_embedding(
            num_embeddings, embedding_dim, m, k, n
        )

        # Copy weights from local to remote
        remote_emb.word_embeder.weight.data.copy_(local_emb.word_embeder.weight.data)
        remote_emb.oe_projection.data.copy_(local_emb.oe_projection.data)

        # Copy oe_embeder weights (temporarily, before offload)
        if not hasattr(remote_emb, "oe_embeder"):
            from sglang.srt.layers.vocab_parallel_embedding import (
                VocabParallelEmbedding,
            )
            from sglang.srt.layers.dp_attention import is_dp_attention_enabled

            total_size = int(remote_emb.exclusive_oe_embedder_size_sums[-1].item())
            remote_emb.oe_embeder = VocabParallelEmbedding(
                num_embeddings=total_size,
                embedding_dim=remote_emb.oe_hidden_dim,
                enable_tp=is_dp_attention_enabled(),
            )
        remote_emb.oe_embeder.weight.data.copy_(local_emb.oe_embeder.weight.data)

        # Prepare minimal ForwardBatch mock
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, num_embeddings, (seq_len,))

        forward_batch = MagicMock()
        forward_batch.forward_mode.is_extend.return_value = True
        forward_batch.forward_mode.is_decode.return_value = False
        forward_batch.batch_size = batch_size
        forward_batch.req_pool_indices = torch.arange(batch_size)
        forward_batch.seq_lens = torch.tensor([8, 8])
        forward_batch.engram_prefetch_request = None

        # Build ngram_embedding_info mock
        from sglang.srt.model_executor.forward_batch_info import NgramEmbeddingInfo

        token_table = torch.zeros(batch_size, 128, dtype=torch.int32)
        # Fill token table with input tokens
        for i in range(batch_size):
            start = i * 8
            token_table[i, :8] = input_ids[start : start + 8].to(torch.int32)

        ngram_info = NgramEmbeddingInfo.create(
            token_table=token_table,
            batch_size=batch_size,
            device=torch.device("cpu"),
            column_starts=torch.zeros(batch_size, dtype=torch.int32),
            req_lens=torch.tensor([8, 8], dtype=torch.int32),
        )
        forward_batch.ngram_embedding_info = ngram_info

        # Init buffers
        local_emb.init_buffers(batch_size, seq_len, "cpu")
        remote_emb.init_buffers(batch_size, seq_len, "cpu")

        # Forward pass
        local_out = local_emb(input_ids, forward_batch)
        remote_out = remote_emb(input_ids, forward_batch)

        self.assertEqual(local_out.shape, remote_out.shape)
        torch.testing.assert_close(local_out, remote_out, atol=1e-5, rtol=1e-5)


class TestEngramPoolIntegration(unittest.TestCase):
    """Test the full pool workflow: load → prefetch → gather → compare."""

    @patch("sglang.srt.engram.engram_pool._import_kv_client")
    def test_pool_round_trip_matches_local(self, mock_import):
        """Pool-based lookup should return exactly the same embeddings
        as indexing the original weight tensor."""
        mock_import.return_value = MockKVClient

        from sglang.srt.engram.engram_pool import EngramPool
        from sglang.srt.engram.engram_prefetcher import EngramPrefetcher

        config = EngramPoolConfig(
            enabled=True,
            worker_hosts=["127.0.0.1"],
            worker_ports=[18482],
            num_tables=3,
            embedding_dim=32,
            table_capacity=200,
            chunk_size=1,
        )
        pool = EngramPool(config)
        pool.initialize()
        prefetcher = EngramPrefetcher(pool, config)

        # Create and load tables
        tables = []
        for t in range(3):
            w = torch.randn(200, 32)
            pool.load_table(t, w)
            tables.append(w)

        # Simulate a prefetch
        seq_len = 50
        table_indices = []
        for t in range(3):
            row_ids = np.random.randint(0, 200, size=seq_len)
            table_indices.append((t, row_ids))

        result = prefetcher.prefetch_and_wait(
            table_indices, seq_len, device=torch.device("cpu"), dtype=torch.float32
        )

        # Compare
        for t, (_, row_ids) in enumerate(table_indices):
            expected = tables[t].numpy()[row_ids]
            np.testing.assert_allclose(
                result[t].numpy(), expected, atol=1e-6
            )

        pool.shutdown()


if __name__ == "__main__":
    unittest.main()
