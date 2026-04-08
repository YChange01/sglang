"""Utility functions for Engram N-gram hash index computation."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_ngram_hash_indices(
    input_ids: torch.Tensor,
    token_table: torch.Tensor,
    req_pool_indices: torch.Tensor,
    column_starts: torch.Tensor,
    req_lens: torch.Tensor,
    over_embedding_n: int,
    over_embedding_k: int,
    over_embedding_m: int,
    num_embeddings: int,
    exclusive_oe_embedder_size_sums: torch.Tensor,
) -> torch.Tensor:
    """Compute N-gram hash indices for Engram embedding lookup.

    This mirrors the logic in NgramEmbedding.forward / compute_n_gram_ids,
    but returns the raw hash indices suitable for remote pool lookup.

    Args:
        input_ids: [seq_len] token IDs.
        token_table: [max_reqs, context_len] historical token table.
        req_pool_indices: [batch_size] request pool row indices.
        column_starts: [batch_size] starting column per request.
        req_lens: [batch_size] token count per request.
        over_embedding_n: N-gram order.
        over_embedding_k: Number of hash functions per N-gram order.
        over_embedding_m: Base modulus parameter.
        num_embeddings: Vocabulary size (used in hash polynomial).
        exclusive_oe_embedder_size_sums: [n_grams + 1] prefix sums for sub-table sizes.

    Returns:
        Tensor [seq_len, n_grams] of integer indices into the unified
        embedding table (with sub-table offsets).
    """
    # Delegate to the existing JIT kernel if available
    try:
        from sglang.jit_kernel.ngram_embedding import compute_n_gram_ids

        n_grams = over_embedding_k * (over_embedding_n - 1)
        oe_n_gram_ids = torch.zeros(
            (len(input_ids), n_grams), dtype=torch.int32, device=input_ids.device
        )

        # Build mod and weight tables
        oe_mods = torch.zeros(
            (over_embedding_n - 1, over_embedding_k), dtype=torch.int32
        )
        oe_weights = torch.zeros(
            (over_embedding_n - 1, over_embedding_k, over_embedding_n),
            dtype=torch.int32,
        )
        for n in range(2, over_embedding_n + 1):
            for k in range(over_embedding_k):
                mod = over_embedding_m + 2 * ((n - 2) * over_embedding_k + k) + 1
                oe_mods[n - 2][k] = mod
                for delta in range(over_embedding_n):
                    oe_weights[n - 2][k][delta] = pow(num_embeddings, delta, mod)

        # Compute prefix sums
        exclusive_req_len_sums = torch.zeros(
            len(req_lens) + 1, dtype=torch.int32, device=input_ids.device
        )
        torch.cumsum(
            req_lens, dim=0, dtype=torch.int32, out=exclusive_req_len_sums[1:]
        )

        compute_n_gram_ids(
            ne_n=over_embedding_n,
            ne_k=over_embedding_k,
            ne_weights=oe_weights,
            ne_mods=oe_mods,
            tokens=input_ids.to(torch.int32),
            exclusive_ne_embedder_size_sums=exclusive_oe_embedder_size_sums,
            exclusive_req_len_sums=exclusive_req_len_sums,
            ne_token_table=token_table,
            row_indices=req_pool_indices,
            column_starts=column_starts,
            n_gram_ids=oe_n_gram_ids,
        )
        return oe_n_gram_ids

    except ImportError:
        logger.warning(
            "JIT kernel not available, using CPU fallback for N-gram hash. "
            "Results differ from JIT version — do not use for production."
        )
        return _compute_ngram_hash_cpu(
            input_ids,
            over_embedding_n,
            over_embedding_k,
            over_embedding_m,
            num_embeddings,
            exclusive_oe_embedder_size_sums,
        )


def _compute_ngram_hash_cpu(
    input_ids: torch.Tensor,
    over_embedding_n: int,
    over_embedding_k: int,
    over_embedding_m: int,
    num_embeddings: int,
    exclusive_oe_embedder_size_sums: torch.Tensor,
) -> torch.Tensor:
    """CPU fallback for N-gram hash computation (simplified, no token_table)."""
    seq_len = len(input_ids)
    n_grams = over_embedding_k * (over_embedding_n - 1)
    ids_np = input_ids.cpu().numpy().astype(np.int64)
    result = np.zeros((seq_len, n_grams), dtype=np.int32)

    for n in range(2, over_embedding_n + 1):
        for k in range(over_embedding_k):
            col = (n - 2) * over_embedding_k + k
            mod = over_embedding_m + 2 * col + 1
            offset = int(exclusive_oe_embedder_size_sums[col].item())
            for i in range(seq_len):
                h = 0
                for delta in range(min(n, i + 1)):
                    h = (h + ids_np[i - delta] * pow(num_embeddings, delta, mod)) % mod
                result[i, col] = offset + h

    return torch.from_numpy(result).to(input_ids.device)


def split_indices_by_table(
    ngram_ids: torch.Tensor,
    exclusive_size_sums: torch.Tensor,
    num_tables: int,
) -> list:
    """Split unified ngram_ids into per-table local row indices.

    Args:
        ngram_ids: [seq_len, n_grams] indices into unified embedding table.
        exclusive_size_sums: [n_grams + 1] prefix sums of per-table sizes.
        num_tables: Number of sub-tables (== n_grams).

    Returns:
        List of (table_id, row_ids_np) tuples where row_ids_np is a 1-D
        int array of local row indices within that sub-table.
    """
    ids_np = ngram_ids.cpu().numpy()
    sums_np = exclusive_size_sums.cpu().numpy()
    result = []
    for t in range(num_tables):
        col_ids = ids_np[:, t]
        local_ids = col_ids - int(sums_np[t])
        result.append((t, local_ids.astype(np.int64)))
    return result
