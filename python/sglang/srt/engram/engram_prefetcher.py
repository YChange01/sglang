"""Asynchronous prefetch engine for Engram embedding retrieval.

Overlaps embedding lookup from the remote pool with upstream Transformer
layer computation, hiding the cross-node retrieval latency behind
GPU compute.
"""

import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.engram.engram_config import EngramPoolConfig
from sglang.srt.engram.engram_pool import EngramPool

logger = logging.getLogger(__name__)


@dataclass
class PrefetchRequest:
    """A pending prefetch for one forward batch."""

    # Per-table async lookup futures: table_id -> Future[np.ndarray]
    table_futures: Dict[int, Future]
    # Metadata for assembly
    seq_len: int
    num_tables: int
    embed_dim: int
    target_device: torch.device
    target_dtype: torch.dtype
    start_time: float


class EngramPrefetcher:
    """Manages asynchronous prefetch of Engram embeddings.

    Usage in ModelRunner forward loop::

        # Before upstream Transformer layers
        req = prefetcher.prefetch_async(table_indices, seq_len, device)

        # ... upstream Transformer layers execute on GPU ...

        # Before Engram layer
        oe_hidden = prefetcher.wait_and_gather(req)
    """

    def __init__(self, pool: EngramPool, config: EngramPoolConfig):
        self.pool = pool
        self.config = config
        self._np_dtype = np.float32

    def prefetch_async(
        self,
        table_indices: List[Tuple[int, np.ndarray]],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> PrefetchRequest:
        """Launch asynchronous embedding lookups for all sub-tables.

        Args:
            table_indices: List of (table_id, row_ids) from
                ``engram_utils.split_indices_by_table``.
            seq_len: Number of tokens in the batch.
            device: Target GPU device for the final tensor.
            dtype: Target dtype for the final tensor.

        Returns:
            PrefetchRequest that can be passed to ``wait_and_gather``.
        """
        table_futures: Dict[int, Future] = {}

        for table_id, row_ids in table_indices:
            fut = self.pool.batch_lookup_async(
                table_id, row_ids, dtype=self._np_dtype
            )
            table_futures[table_id] = fut

        embed_dim = self.config.embedding_dim
        meta = self.pool.get_table_meta(table_indices[0][0]) if table_indices else None
        if meta:
            embed_dim = meta["embed_dim"]

        return PrefetchRequest(
            table_futures=table_futures,
            seq_len=seq_len,
            num_tables=len(table_indices),
            embed_dim=embed_dim,
            target_device=device,
            target_dtype=dtype,
            start_time=time.monotonic(),
        )

    def wait_and_gather(self, request: PrefetchRequest) -> torch.Tensor:
        """Wait for all prefetch futures and assemble into a GPU tensor.

        Args:
            request: PrefetchRequest from ``prefetch_async``.

        Returns:
            Tensor of shape [num_tables, seq_len, embed_dim] on the
            target device, containing the retrieved embeddings.
        """
        num_tables = request.num_tables
        seq_len = request.seq_len
        embed_dim = request.embed_dim

        # Collect results in order
        results_np = np.empty(
            (num_tables, seq_len, embed_dim), dtype=self._np_dtype
        )

        for idx, (table_id, fut) in enumerate(
            sorted(request.table_futures.items())
        ):
            arr = fut.result()  # blocks until this table's lookup completes
            if arr.shape != (seq_len, embed_dim):
                raise RuntimeError(
                    f"Engram prefetch: table {table_id} shape mismatch: "
                    f"expected ({seq_len}, {embed_dim}), got {arr.shape}"
                )
            results_np[idx] = arr

        # Transfer to GPU
        result_tensor = torch.from_numpy(results_np).to(
            device=request.target_device, dtype=request.target_dtype
        )

        elapsed_ms = (time.monotonic() - request.start_time) * 1000
        total_bytes = num_tables * seq_len * embed_dim * 4
        logger.debug(
            "Engram prefetch: %d tables, %d tokens, %.1f KB in %.2f ms "
            "(%.1f MB/s)",
            num_tables,
            seq_len,
            total_bytes / 1024,
            elapsed_ms,
            (total_bytes / 1e6) / (elapsed_ms / 1000 + 1e-9),
        )

        return result_tensor

    def prefetch_and_wait(
        self,
        table_indices: List[Tuple[int, np.ndarray]],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Convenience: synchronous prefetch + gather in one call."""
        req = self.prefetch_async(table_indices, seq_len, device, dtype)
        return self.wait_and_gather(req)
