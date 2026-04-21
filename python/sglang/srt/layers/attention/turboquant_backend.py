"""TurboQuant attention backend for SGLang (paper arXiv:2504.19874).

MVP scope
---------
* Homogeneous and split modes both supported (Days 1-5 ported the
  algorithm + kernels from vllm_fork).
* Enforce-eager only (no CUDA graph capture in Phase 1).
* Sliding window / ALiBi / logit soft-cap not supported.
* Speculative decoding not supported.

Construction is wired via ``@register_attention_backend("turboquant")``
in attention_registry.py. The server picks the backend via
``--attention-backend turboquant``; the matching KV pool
(``TurboQuantMHAPool``) is installed conditionally in
``model_runner_kv_cache_mixin``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.turboquant import (
    QuantState,
    SplitQuantState,
    TurboQuantConfig,
    build_query_start_loc,
    sglang_paged_attention_split_tc,
    sglang_paged_attention_tc,
    sglang_store_kv,
    sglang_store_split,
    sglang_store_v,
)
from sglang.srt.mem_cache.turboquant_pool import TurboQuantMHAPool

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class _ForwardMetadata:
    """Per-forward-pass metadata shared across all layers."""
    # (bs+1,) int32 indptr over query tokens. Decode: [0,1,...,bs].
    query_start_loc: torch.Tensor


class TurboQuantAttnBackend(AttentionBackend):
    """Paper-faithful TurboQuant attention for SGLang."""

    def __init__(self, model_runner: "ModelRunner") -> None:
        super().__init__()
        self.model_runner = model_runner
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

        pool = model_runner.token_to_kv_pool
        if not isinstance(pool, TurboQuantMHAPool):
            raise RuntimeError(
                "--attention-backend turboquant requires a "
                "TurboQuantMHAPool. Got type "
                f"{type(pool).__name__}. Check model_runner_kv_cache_mixin."
            )
        self.pool: TurboQuantMHAPool = pool
        self.cfg: TurboQuantConfig = pool.turboquant_config

        # Lazy per-layer state maps (keyed by layer.layer_id).
        self._layer_state: dict[int, QuantState] = {}
        self._layer_split_state: dict[
            int, tuple[SplitQuantState, SplitQuantState]
        ] = {}

        self.forward_metadata: Optional[_ForwardMetadata] = None

    # ------------------------------------------------------------------
    # Lazy state init
    # ------------------------------------------------------------------
    def _get_homog_state(
        self, layer: "RadixAttention", dtype: torch.dtype, device: torch.device
    ) -> QuantState:
        lid = layer.layer_id
        state = self._layer_state.get(lid)
        if state is None:
            tight = (
                self.cfg.tight_pack
                and self.cfg.is_homog
                and self.cfg.bits == 4
            )
            state = QuantState(
                algo=self.cfg.algo,
                bits=self.cfg.bits,
                head_dim=layer.qk_head_dim,
                seed=lid,
                dtype=dtype,
                device=device,
                tight_pack=tight,
            )
            self._layer_state[lid] = state
        return state

    def _get_split_state(
        self, layer: "RadixAttention", dtype: torch.dtype, device: torch.device
    ) -> tuple[SplitQuantState, SplitQuantState]:
        lid = layer.layer_id
        pair = self._layer_split_state.get(lid)
        if pair is None:
            mask = self.pool.outlier_mask
            assert mask is not None
            if mask.head_dim != layer.qk_head_dim:
                raise ValueError(
                    f"Outlier mask head_dim {mask.head_dim} != layer "
                    f"head_dim {layer.qk_head_dim}. Re-run calibration."
                )
            k_outlier_idx = mask.for_layer(lid % mask.num_layers, "k").to(device)
            # K mask drives layout; V reuses K indices so both kernels
            # see a consistent channel partition (see vllm backend note).
            k_state = SplitQuantState(
                algo=self.cfg.algo,
                bits_outlier=self.cfg.bits_outlier,
                bits_regular=self.cfg.bits_regular,
                head_dim=layer.qk_head_dim,
                outlier_idx=k_outlier_idx,
                seed=lid,
                dtype=dtype,
                device=device,
            )
            v_state = SplitQuantState(
                algo=self.cfg.algo,
                bits_outlier=self.cfg.bits_outlier,
                bits_regular=self.cfg.bits_regular,
                head_dim=layer.qk_head_dim,
                outlier_idx=k_outlier_idx,
                seed=lid ^ 0x7FFFFFFF,
                dtype=dtype,
                device=device,
            )
            pair = (k_state, v_state)
            self._layer_split_state[lid] = pair
        return pair

    # ------------------------------------------------------------------
    # Forward metadata
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        bs = forward_batch.batch_size
        extend = forward_batch.extend_seq_lens
        is_extend = (
            forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_mixed()
        )
        qsl = build_query_start_loc(
            extend_seq_lens=extend if is_extend else None,
            batch_size=bs,
            device=self.device,
        )
        self.forward_metadata = _ForwardMetadata(query_start_loc=qsl)

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 0

    # CUDA graph hooks intentionally left unimplemented (MVP).
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        raise NotImplementedError(
            "TurboQuant backend requires --disable-cuda-graph in MVP."
        )

    # ------------------------------------------------------------------
    # Forward extend / decode
    # ------------------------------------------------------------------
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward_core(
            q, k, v, layer, forward_batch, save_kv_cache,
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward_core(
            q, k, v, layer, forward_batch, save_kv_cache,
        )

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def _forward_core(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool,
    ) -> torch.Tensor:
        num_query_tokens = q.shape[0]
        H_q = layer.tp_q_head_num
        H_kv = layer.tp_k_head_num
        d = layer.qk_head_dim
        if layer.qk_head_dim != layer.v_head_dim:
            raise NotImplementedError(
                "TurboQuant MVP requires qk_head_dim == v_head_dim "
                f"(got {d} vs {layer.v_head_dim})."
            )
        q = q.reshape(num_query_tokens, H_q, d)
        if k is not None and v is not None:
            k = k.reshape(num_query_tokens, H_kv, d)
            v = v.reshape(num_query_tokens, H_kv, d)

        device = q.device
        bufs = self.pool.get_quant_buffers(layer.layer_id)

        if self.cfg.is_split:
            k_state, v_state = self._get_split_state(layer, q.dtype, device)
            if save_kv_cache and k is not None:
                self._store_split(
                    k, v, forward_batch.out_cache_loc,
                    k_state, v_state, bufs,
                )
            attn_out = sglang_paged_attention_split_tc(
                q=q,
                cache_k_idx_out=bufs["cache_k_idx_out"],
                cache_k_norm_out=bufs["cache_k_norm_out"],
                cache_v_idx_out=bufs["cache_v_idx_out"],
                cache_v_norm_out=bufs["cache_v_norm_out"],
                cache_k_qjl_sign_out=bufs["cache_k_qjl_sign_out"],
                cache_k_rnorm_out=bufs["cache_k_rnorm_out"],
                cache_v_qjl_sign_out=bufs["cache_v_qjl_sign_out"],
                cache_v_rnorm_out=bufs["cache_v_rnorm_out"],
                cache_k_idx_reg=bufs["cache_k_idx_reg"],
                cache_k_norm_reg=bufs["cache_k_norm_reg"],
                cache_v_idx_reg=bufs["cache_v_idx_reg"],
                cache_v_norm_reg=bufs["cache_v_norm_reg"],
                cache_k_qjl_sign_reg=bufs["cache_k_qjl_sign_reg"],
                cache_k_rnorm_reg=bufs["cache_k_rnorm_reg"],
                cache_v_qjl_sign_reg=bufs["cache_v_qjl_sign_reg"],
                cache_v_rnorm_reg=bufs["cache_v_rnorm_reg"],
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                query_start_loc=self.forward_metadata.query_start_loc,
                state_k=k_state,
                state_v=v_state,
            )
        else:
            state = self._get_homog_state(layer, q.dtype, device)
            if save_kv_cache and k is not None:
                sglang_store_kv(
                    new_k=k,
                    out_cache_loc=forward_batch.out_cache_loc,
                    state=state,
                    cache_k_idx=bufs["cache_k_idx"],
                    cache_k_norm=bufs["cache_k_norm"],
                    cache_k_qjl_sign=bufs["cache_k_qjl_sign"],
                    cache_k_rnorm=bufs["cache_k_rnorm"],
                )
                sglang_store_v(
                    new_v=v,
                    out_cache_loc=forward_batch.out_cache_loc,
                    state=state,
                    cache_v_idx=bufs["cache_v_idx"],
                    cache_v_norm=bufs["cache_v_norm"],
                    cache_v_qjl_sign=bufs["cache_v_qjl_sign"],
                    cache_v_rnorm=bufs["cache_v_rnorm"],
                )
            attn_out = sglang_paged_attention_tc(
                q=q,
                cache_k_idx=bufs["cache_k_idx"],
                cache_k_norm=bufs["cache_k_norm"],
                cache_v_idx=bufs["cache_v_idx"],
                cache_v_norm=bufs["cache_v_norm"],
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                query_start_loc=self.forward_metadata.query_start_loc,
                state=state,
                cache_k_qjl_sign=bufs["cache_k_qjl_sign"],
                cache_k_rnorm=bufs["cache_k_rnorm"],
                cache_v_qjl_sign=bufs["cache_v_qjl_sign"],
                cache_v_rnorm=bufs["cache_v_rnorm"],
            )

        # (num_query_tokens, H_q, d) -> (num_query_tokens, H_q*d) so the
        # model's down-projection sees the standard layout.
        return attn_out.reshape(num_query_tokens, H_q * d)

    def _store_split(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_cache_loc: torch.Tensor,
        k_state: SplitQuantState,
        v_state: SplitQuantState,
        bufs: dict,
    ) -> None:
        sglang_store_split(
            new_x=k,
            out_cache_loc=out_cache_loc,
            state_split=k_state,
            cache_idx_out=bufs["cache_k_idx_out"],
            cache_norm_out=bufs["cache_k_norm_out"],
            cache_qjl_sign_out=bufs["cache_k_qjl_sign_out"],
            cache_rnorm_out=bufs["cache_k_rnorm_out"],
            cache_idx_reg=bufs["cache_k_idx_reg"],
            cache_norm_reg=bufs["cache_k_norm_reg"],
            cache_qjl_sign_reg=bufs["cache_k_qjl_sign_reg"],
            cache_rnorm_reg=bufs["cache_k_rnorm_reg"],
        )
        sglang_store_split(
            new_x=v,
            out_cache_loc=out_cache_loc,
            state_split=v_state,
            cache_idx_out=bufs["cache_v_idx_out"],
            cache_norm_out=bufs["cache_v_norm_out"],
            cache_qjl_sign_out=bufs["cache_v_qjl_sign_out"],
            cache_rnorm_out=bufs["cache_v_rnorm_out"],
            cache_idx_reg=bufs["cache_v_idx_reg"],
            cache_norm_reg=bufs["cache_v_norm_reg"],
            cache_qjl_sign_reg=bufs["cache_v_qjl_sign_reg"],
            cache_rnorm_reg=bufs["cache_v_rnorm_reg"],
        )
