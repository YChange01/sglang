"""SGLang-facing wrappers for TurboQuant store + attend kernels.

The vLLM Triton kernels assume paged K/V with shape
``(num_blocks, block_size, H_kv, ...)`` and index each slot as
``phys_blocks * block_size + tok_in_block``. In SGLang the KV pool
is flat ``(num_slots, H_kv, ...)`` and ``req_to_token[req_id, pos]``
returns the token slot index directly.

Key insight: the block-major indexing math

    base = phys_blocks * (block_size * H_kv * D) + tok_in_block * (H_kv * D)

reduces to ``slot * H_kv * D`` when ``block_size == 1``. We therefore
call the upstream kernels with ``block_size=1`` and pass SGLang's flat
buffers (reshaped with a leading 1-dim) plus ``req_to_token`` as the
``block_table``. No kernel rewrite required.

Differences that still need shimming:
 * slot_mapping   == out_cache_loc (both are 1-D abs slot indices)
 * block_table    == req_to_token  (same semantics with block_size=1)
 * block_size     = 1

For extend/prefill we build ``query_start_loc`` from ``extend_seq_lens``;
for decode it's identity ``[0, 1, 2, ..., bs]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .attend_split_tc import turboquant_paged_attention_split_tc
from .attend_tc import turboquant_paged_attention_tc
from .store import (
    turboquant_store_kv,
    turboquant_store_split,
    turboquant_store_v,
)

if TYPE_CHECKING:
    from .codebook import QuantState
    from .outlier import SplitQuantState


# ---------------------------------------------------------------------------
# Buffer reshape: SGLang (num_slots, H_kv, D) -> vLLM (num_slots, 1, H_kv, D)
# ---------------------------------------------------------------------------
def _as_paged(buf: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Insert a size-1 block dim so vLLM kernels (which expect
    block-major layout) index the flat SGLang pool correctly with
    block_size=1.
    """
    if buf is None:
        return None
    # (num_slots, H_kv, D) -> (num_slots, 1, H_kv, D)
    # (num_slots, H_kv)    -> (num_slots, 1, H_kv)
    return buf.unsqueeze(1)


# ---------------------------------------------------------------------------
# Store (write) path
# ---------------------------------------------------------------------------
def sglang_store_kv(
    new_k: torch.Tensor,
    out_cache_loc: torch.Tensor,
    state: "QuantState",
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_k_qjl_sign: Optional[torch.Tensor] = None,
    cache_k_rnorm: Optional[torch.Tensor] = None,
) -> None:
    """Write quantized K into the SGLang flat pool.

    Args:
        new_k:        (T, H_kv, d)  bf16/fp16 K values to write
        out_cache_loc: (T,) int64  destination slot indices
        cache_k_idx:  (num_slots, H_kv, idx_dim)  uint8  flat pool buffer
        cache_k_norm: (num_slots, H_kv)
        cache_k_qjl_sign: (num_slots, H_kv, d/8)  (prod only)
        cache_k_rnorm:    (num_slots, H_kv)        (prod only)
    """
    turboquant_store_kv(
        new_k=new_k,
        cache_k_idx=_as_paged(cache_k_idx),
        cache_k_norm=_as_paged(cache_k_norm),
        slot_mapping=out_cache_loc,
        state=state,
        block_size=1,
        cache_k_qjl_sign=_as_paged(cache_k_qjl_sign),
        cache_k_rnorm=_as_paged(cache_k_rnorm),
    )


def sglang_store_v(
    new_v: torch.Tensor,
    out_cache_loc: torch.Tensor,
    state: "QuantState",
    cache_v_idx: torch.Tensor,
    cache_v_norm: torch.Tensor,
    cache_v_qjl_sign: Optional[torch.Tensor] = None,
    cache_v_rnorm: Optional[torch.Tensor] = None,
) -> None:
    turboquant_store_v(
        new_v=new_v,
        cache_v_idx=_as_paged(cache_v_idx),
        cache_v_norm=_as_paged(cache_v_norm),
        slot_mapping=out_cache_loc,
        state=state,
        block_size=1,
        cache_v_qjl_sign=_as_paged(cache_v_qjl_sign),
        cache_v_rnorm=_as_paged(cache_v_rnorm),
    )


# ---------------------------------------------------------------------------
# Attend (read) path
# ---------------------------------------------------------------------------
def sglang_paged_attention_tc(
    q: torch.Tensor,
    cache_k_idx: torch.Tensor,
    cache_k_norm: torch.Tensor,
    cache_v_idx: torch.Tensor,
    cache_v_norm: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state: "QuantState",
    cache_k_qjl_sign: Optional[torch.Tensor] = None,
    cache_k_rnorm: Optional[torch.Tensor] = None,
    cache_v_qjl_sign: Optional[torch.Tensor] = None,
    cache_v_rnorm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """TurboQuant paged attention against SGLang's flat pool + req table.

    Args:
        q:             (num_query_tokens, H_q, d)  bf16/fp16
        cache_k_idx/_norm/_qjl_sign/_rnorm:  (num_slots, H_kv, ...)
        cache_v_idx/_norm/_qjl_sign/_rnorm:  same
        req_to_token: (num_req_slots, max_context_len) int32
            row[req_pool_indices[b], :seq_lens[b]] gives absolute slot
            indices for batch element b.
        req_pool_indices: (bs,) int32
            maps batch element to row in req_to_token.
        seq_lens:     (bs,) int32/64  total tokens per batch elem (incl. new)
        query_start_loc: (bs+1,) int32  prefix-sum of query token counts
        state:        per-layer QuantState (codebook + Pi + S + ...)

    Returns:
        output: (num_query_tokens, H_q, d) in q's dtype
    """
    # Reorder req_to_token rows so the "batch-idx" axis expected by the
    # vLLM kernel matches. vLLM assumes block_table[batch_idx, *] gives
    # the blocks for batch element batch_idx; SGLang's req_to_token is
    # indexed by req_pool_idx, not batch_idx. Gather into a
    # batch-indexed view (cheap; bs is small).
    block_table = req_to_token[req_pool_indices]

    return turboquant_paged_attention_tc(
        q=q,
        cache_k_idx=_as_paged(cache_k_idx),
        cache_k_norm=_as_paged(cache_k_norm),
        cache_v_idx=_as_paged(cache_v_idx),
        cache_v_norm=_as_paged(cache_v_norm),
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state=state,
        cache_k_qjl_sign=_as_paged(cache_k_qjl_sign),
        cache_k_rnorm=_as_paged(cache_k_rnorm),
        cache_v_qjl_sign=_as_paged(cache_v_qjl_sign),
        cache_v_rnorm=_as_paged(cache_v_rnorm),
    )


# ---------------------------------------------------------------------------
# Split-mode store + attend (paper §4.3: outlier channel splitting)
# ---------------------------------------------------------------------------
def sglang_store_split(
    new_x: torch.Tensor,
    out_cache_loc: torch.Tensor,
    state_split: "SplitQuantState",
    # Outlier slice buffers (flat SGLang layout)
    cache_idx_out: torch.Tensor,
    cache_norm_out: torch.Tensor,
    cache_qjl_sign_out: Optional[torch.Tensor],
    cache_rnorm_out: Optional[torch.Tensor],
    # Regular slice buffers
    cache_idx_reg: torch.Tensor,
    cache_norm_reg: torch.Tensor,
    cache_qjl_sign_reg: Optional[torch.Tensor],
    cache_rnorm_reg: Optional[torch.Tensor],
) -> None:
    """Write split-quantized K or V into the SGLang flat pool."""
    turboquant_store_split(
        new_x=new_x,
        state_split=state_split,
        cache_idx_out=_as_paged(cache_idx_out),
        cache_norm_out=_as_paged(cache_norm_out),
        cache_qjl_sign_out=_as_paged(cache_qjl_sign_out),
        cache_rnorm_out=_as_paged(cache_rnorm_out),
        cache_idx_reg=_as_paged(cache_idx_reg),
        cache_norm_reg=_as_paged(cache_norm_reg),
        cache_qjl_sign_reg=_as_paged(cache_qjl_sign_reg),
        cache_rnorm_reg=_as_paged(cache_rnorm_reg),
        slot_mapping=out_cache_loc,
        block_size=1,
    )


def sglang_paged_attention_split_tc(
    q: torch.Tensor,
    # Outlier buffers (K + V, flat SGLang layout)
    cache_k_idx_out: torch.Tensor,
    cache_k_norm_out: torch.Tensor,
    cache_v_idx_out: torch.Tensor,
    cache_v_norm_out: torch.Tensor,
    cache_k_qjl_sign_out: Optional[torch.Tensor],
    cache_k_rnorm_out: Optional[torch.Tensor],
    cache_v_qjl_sign_out: Optional[torch.Tensor],
    cache_v_rnorm_out: Optional[torch.Tensor],
    # Regular buffers (K + V)
    cache_k_idx_reg: torch.Tensor,
    cache_k_norm_reg: torch.Tensor,
    cache_v_idx_reg: torch.Tensor,
    cache_v_norm_reg: torch.Tensor,
    cache_k_qjl_sign_reg: Optional[torch.Tensor],
    cache_k_rnorm_reg: Optional[torch.Tensor],
    cache_v_qjl_sign_reg: Optional[torch.Tensor],
    cache_v_rnorm_reg: Optional[torch.Tensor],
    # SGLang-style metadata
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_k: "SplitQuantState",
    state_v: "SplitQuantState",
) -> torch.Tensor:
    """TurboQuant split-mode paged attention against SGLang's flat pool."""
    block_table = req_to_token[req_pool_indices]

    return turboquant_paged_attention_split_tc(
        q=q,
        cache_k_idx_out=_as_paged(cache_k_idx_out),
        cache_k_norm_out=_as_paged(cache_k_norm_out),
        cache_v_idx_out=_as_paged(cache_v_idx_out),
        cache_v_norm_out=_as_paged(cache_v_norm_out),
        cache_k_qjl_sign_out=_as_paged(cache_k_qjl_sign_out),
        cache_k_rnorm_out=_as_paged(cache_k_rnorm_out),
        cache_v_qjl_sign_out=_as_paged(cache_v_qjl_sign_out),
        cache_v_rnorm_out=_as_paged(cache_v_rnorm_out),
        cache_k_idx_reg=_as_paged(cache_k_idx_reg),
        cache_k_norm_reg=_as_paged(cache_k_norm_reg),
        cache_v_idx_reg=_as_paged(cache_v_idx_reg),
        cache_v_norm_reg=_as_paged(cache_v_norm_reg),
        cache_k_qjl_sign_reg=_as_paged(cache_k_qjl_sign_reg),
        cache_k_rnorm_reg=_as_paged(cache_k_rnorm_reg),
        cache_v_qjl_sign_reg=_as_paged(cache_v_qjl_sign_reg),
        cache_v_rnorm_reg=_as_paged(cache_v_rnorm_reg),
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_k=state_k,
        state_v=state_v,
    )


# ---------------------------------------------------------------------------
# ForwardBatch helpers
# ---------------------------------------------------------------------------
def build_query_start_loc(
    extend_seq_lens: Optional[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a vLLM-style query_start_loc (bs+1,) int32 indptr.

    In forward_extend: one row per sample, length = extend_seq_lens[i].
    In forward_decode: one row per sample, length 1 each.
    """
    if extend_seq_lens is None:
        query_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    else:
        query_lens = extend_seq_lens.to(device=device, dtype=torch.int32)
    qsl = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    qsl[1:] = torch.cumsum(query_lens, dim=0)
    return qsl
