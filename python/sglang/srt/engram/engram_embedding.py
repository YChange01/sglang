"""Remote-pooled Engram embedding layer.

Replaces the large N-gram embedding tables in NgramEmbedding with remote
lookups via EngramPrefetcher, keeping only the small projection matrices
and standard word embedding on GPU.
"""

import logging
from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.engram.engram_prefetcher import EngramPrefetcher, PrefetchRequest
from sglang.srt.engram.engram_utils import (
    compute_ngram_hash_indices,
    split_indices_by_table,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class RemoteEngramEmbedding(torch.nn.Module):
    """N-gram embedding layer with large tables offloaded to remote pool.

    Architecture::

        input_ids ──► word_embeder (local GPU) ──────────────────┐
                 │                                                │
                 └► N-gram hash → remote pool lookup → project ──┤
                        (via EngramPrefetcher)                    │
                                                            mean(dim=0)
                                                                  │
                                                           hidden_states

    The remote pool lookup is triggered asynchronously *before* this layer's
    forward() is called, overlapping with upstream Transformer computation.
    By the time forward() runs, results are already (mostly) available.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        over_embedding_m: int,
        over_embedding_k: int,
        over_embedding_n: int,
        prefetcher: Optional[EngramPrefetcher] = None,
    ):
        super().__init__()
        assert over_embedding_n > 1

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.over_embedding_m = over_embedding_m
        self.over_embedding_k = over_embedding_k
        self.over_embedding_n = over_embedding_n
        self.prefetcher = prefetcher

        # Local: standard word embedding (small, stays on GPU)
        self.word_embeder = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            enable_tp=is_dp_attention_enabled(),
        )

        # N-gram sub-table structure metadata
        self.n_grams = over_embedding_k * (over_embedding_n - 1)
        self.oe_hidden_dim = embedding_dim // self.n_grams

        # Prefix sums for sub-table sizes (needed for index splitting)
        self.exclusive_oe_embedder_size_sums = torch.zeros(
            self.n_grams + 1, dtype=torch.int32, device="cuda"
        )
        for i in range(self.n_grams):
            self.exclusive_oe_embedder_size_sums[i + 1] = (
                self.exclusive_oe_embedder_size_sums[i]
                + int(over_embedding_m + i * 2 + 1)
            )

        # Local: projection matrices (small, stays on GPU)
        # Shape: [n_grams, oe_hidden_dim, embedding_dim]
        self.oe_projection = nn.Parameter(
            torch.empty(self.n_grams, self.oe_hidden_dim, embedding_dim),
            requires_grad=False,
        )

        # N-gram hash parameters (CPU, small)
        self.oe_mods = torch.zeros(
            (over_embedding_n - 1, over_embedding_k), dtype=torch.int32
        )
        self.oe_weights = torch.zeros(
            (over_embedding_n - 1, over_embedding_k, over_embedding_n),
            dtype=torch.int32,
        )
        for n in range(2, over_embedding_n + 1):
            for k in range(over_embedding_k):
                mod = over_embedding_m + 2 * ((n - 2) * over_embedding_k + k) + 1
                self.oe_mods[n - 2][k] = mod
                for delta in range(over_embedding_n):
                    self.oe_weights[n - 2][k][delta] = pow(
                        num_embeddings, delta, mod
                    )

        # NOTE: We do NOT create self.oe_embeder (the large embedding table).
        # It is offloaded to the remote pool.

    def init_buffers(
        self, max_running_requests: int, chunked_prefill_size: int, device: str
    ):
        """Initialize runtime buffers (mirrors NgramEmbedding.init_buffers)."""
        max_tokens = max(chunked_prefill_size, max_running_requests)
        self.oe_n_gram_ids = torch.zeros(
            (max_tokens, self.n_grams), dtype=torch.int32, device=device
        )
        self.exclusive_req_len_sums = torch.zeros(
            max_running_requests + 1, dtype=torch.int32, device=device
        )

    def get_oe_table_weights(self) -> Optional[torch.Tensor]:
        """Return the local oe_embeder weights if they exist (for loading to pool).

        Returns None if weights have already been offloaded.
        """
        if hasattr(self, "oe_embeder"):
            return self.oe_embeder.weight.data
        return None

    def offload_oe_table(self):
        """Delete the local oe_embeder to free GPU memory after loading to pool."""
        if hasattr(self, "oe_embeder"):
            del self.oe_embeder
            logger.info(
                "RemoteEngramEmbedding: offloaded oe_embeder from GPU"
            )

    def load_weight(
        self, param: Parameter, weight_name: str, loaded_weight: torch.Tensor
    ):
        """Weight loader compatible with NgramEmbedding (for model loading)."""
        if ".embed_tokens." in weight_name:
            param.weight_loader(param, loaded_weight)
        elif "model.ngram_embeddings.embedders." in weight_name:
            # Temporarily store in a local oe_embeder for later pool upload
            if not hasattr(self, "oe_embeder"):
                total_size = int(self.exclusive_oe_embedder_size_sums[-1].item())
                self.oe_embeder = VocabParallelEmbedding(
                    num_embeddings=total_size,
                    embedding_dim=self.oe_hidden_dim,
                    enable_tp=is_dp_attention_enabled(),
                )
            index = int(
                weight_name.replace("model.ngram_embeddings.embedders.", "")
                .replace(".weight", "")
            )
            oe_weight_start = self.exclusive_oe_embedder_size_sums[index]
            oe_weight_end = self.exclusive_oe_embedder_size_sums[index + 1]
            tp_start = self.oe_embeder.shard_indices.org_vocab_start_index
            tp_end = self.oe_embeder.shard_indices.org_vocab_end_index
            to_load_start = max(oe_weight_start, tp_start)
            to_load_end = min(oe_weight_end, tp_end)
            if to_load_start < to_load_end:
                src_start = to_load_start - oe_weight_start
                src_end = to_load_end - oe_weight_start
                dest_start = to_load_start - tp_start
                dest_end = to_load_end - tp_start
                self.oe_embeder.weight.data[dest_start:dest_end] = loaded_weight[
                    src_start:src_end
                ]
        elif "model.ngram_embeddings.post_projs." in weight_name:
            index = int(
                weight_name.replace("model.ngram_embeddings.post_projs.", "")
                .replace(".weight", "")
            )
            self.oe_projection[index].copy_(loaded_weight.data.t())
        else:
            logger.warning("Unknown ngram embedding weight name: %s", weight_name)

    def trigger_prefetch(
        self, forward_batch: ForwardBatch
    ) -> Optional[PrefetchRequest]:
        """Compute N-gram indices and launch async prefetch.

        Called by ModelRunner BEFORE the upstream Transformer layers,
        so retrieval overlaps with compute.

        Returns:
            PrefetchRequest to be stored in forward_batch, or None if
            prefetcher is not configured.
        """
        if self.prefetcher is None:
            return None

        if not (
            forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_decode()
        ):
            return None

        input_ids = forward_batch.input_ids
        ngram_info = forward_batch.ngram_embedding_info
        if ngram_info is None:
            return None

        # Compute N-gram hash indices
        torch.cumsum(
            ngram_info.req_lens,
            dim=0,
            dtype=torch.int32,
            out=self.exclusive_req_len_sums[1 : 1 + forward_batch.batch_size],
        )

        from sglang.jit_kernel.ngram_embedding import compute_n_gram_ids

        oe_n_gram_ids = self.oe_n_gram_ids[: len(input_ids)]
        compute_n_gram_ids(
            ne_n=self.over_embedding_n,
            ne_k=self.over_embedding_k,
            ne_weights=self.oe_weights,
            ne_mods=self.oe_mods,
            tokens=input_ids.to(torch.int32),
            exclusive_ne_embedder_size_sums=self.exclusive_oe_embedder_size_sums,
            exclusive_req_len_sums=self.exclusive_req_len_sums[
                : forward_batch.batch_size + 1
            ],
            ne_token_table=ngram_info.token_table,
            row_indices=forward_batch.req_pool_indices,
            column_starts=ngram_info.column_starts,
            n_gram_ids=oe_n_gram_ids,
        )

        # Split into per-table local indices
        table_indices = split_indices_by_table(
            oe_n_gram_ids,
            self.exclusive_oe_embedder_size_sums,
            self.n_grams,
        )

        # Launch async prefetch
        return self.prefetcher.prefetch_async(
            table_indices,
            seq_len=len(input_ids),
            device=input_ids.device,
            dtype=self.oe_projection.dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Forward pass using prefetched remote embeddings.

        If a PrefetchRequest was set on forward_batch, uses it.
        Otherwise falls back to local embedding (if oe_embeder exists).
        """
        # Standard word embedding (always local)
        word_hidden = self.word_embeder(input_ids)

        prefetch_req = getattr(forward_batch, "engram_prefetch_request", None)

        if prefetch_req is not None and self.prefetcher is not None:
            # Use prefetched remote embeddings
            # Shape: [n_grams, seq_len, oe_hidden_dim]
            oe_hidden_states = self.prefetcher.wait_and_gather(prefetch_req)

            # Project each sub-table's embeddings to model dim
            # oe_hidden_states: [n_grams, seq_len, oe_hidden_dim]
            # oe_projection: [n_grams, oe_hidden_dim, embedding_dim]
            # result: [n_grams, seq_len, embedding_dim]
            projected = torch.bmm(oe_hidden_states, self.oe_projection)

            # Combine with word embedding
            all_hidden = torch.cat(
                [word_hidden.unsqueeze(0), projected], dim=0
            )
            return all_hidden.mean(dim=0)

        elif hasattr(self, "oe_embeder"):
            # Fallback: local embedding (before offloading, or pool disabled)
            ngram_info = forward_batch.ngram_embedding_info
            if ngram_info is not None and (
                forward_batch.forward_mode.is_extend()
                or forward_batch.forward_mode.is_decode()
            ):
                torch.cumsum(
                    ngram_info.req_lens,
                    dim=0,
                    dtype=torch.int32,
                    out=self.exclusive_req_len_sums[
                        1 : 1 + forward_batch.batch_size
                    ],
                )
                from sglang.jit_kernel.ngram_embedding import compute_n_gram_ids

                compute_n_gram_ids(
                    ne_n=self.over_embedding_n,
                    ne_k=self.over_embedding_k,
                    ne_weights=self.oe_weights,
                    ne_mods=self.oe_mods,
                    tokens=input_ids.to(torch.int32),
                    exclusive_ne_embedder_size_sums=self.exclusive_oe_embedder_size_sums,
                    exclusive_req_len_sums=self.exclusive_req_len_sums[
                        : forward_batch.batch_size + 1
                    ],
                    ne_token_table=ngram_info.token_table,
                    row_indices=forward_batch.req_pool_indices,
                    column_starts=ngram_info.column_starts,
                    n_gram_ids=self.oe_n_gram_ids[: len(input_ids)],
                )

            all_hidden = torch.empty(
                (self.n_grams + 1, len(input_ids), self.embedding_dim),
                dtype=self.oe_projection.dtype,
                device=input_ids.device,
            )
            all_hidden[0] = word_hidden
            oe_hidden = self.oe_embeder(
                self.oe_n_gram_ids[: len(input_ids)].permute(1, 0).contiguous()
            )
            torch.bmm(oe_hidden, self.oe_projection, out=all_hidden[1:])
            return all_hidden.mean(dim=0)

        else:
            # No N-gram embeddings available: word embedding only
            return word_hidden
