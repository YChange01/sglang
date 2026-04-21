"""KV cache pool subclass for TurboQuant paper-faithful quantization.

Allocates the quant-specific buffers (idx / norm / qjl_sign / rnorm)
alongside the parent's bf16 K/V buffers. The attention backend reads
the quant buffers via ``get_quant_buffers(layer_id)`` instead of the
standard ``get_key_buffer`` / ``get_value_buffer``.

MVP scope (Day 2)
-----------------
Keeps the parent bf16 allocation as a compatibility shim so unrelated
SGLang code (radix cache, disagg transfer, CPU offload) continues to
work. Day 3-4 wires the Triton ``turboquant_store_kv`` kernel into
``set_kv_buffer`` -- until then, quant buffers stay zero-initialized.

Layout (per layer, flat token-major to match SGLang's page_size<=16):

    cache_k_idx:      (size+page_size, H_kv, idx_last_dim)  uint8
    cache_k_norm:     (size+page_size, H_kv)                fp32|fp16
    cache_k_qjl_sign: (size+page_size, H_kv, head_dim//8)   uint8  (prod only)
    cache_k_rnorm:    (size+page_size, H_kv)                fp32|fp16|uint8  (prod only)

V side is identical. Homog mode only -- split mode lands in Day 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.turboquant import OutlierMask, TurboQuantConfig
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


_NORM_TORCH_DTYPE = {"fp32": torch.float32, "fp16": torch.float16}
_RNORM_TORCH_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "uint8": torch.uint8,
}


def _pow2_ceil(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _idx_last_dim_homog(cfg: TurboQuantConfig, head_dim: int) -> int:
    """Bytes per (token, head) for the idx buffer in homog mode."""
    main_bits = cfg.main_bits
    pack_bits = _pow2_ceil(main_bits)
    if head_dim * pack_bits % 8 != 0:
        raise ValueError(
            f"head_dim={head_dim} * pack_bits={pack_bits} must be "
            "byte-aligned for TurboQuant packing"
        )
    return head_dim * pack_bits // 8


class TurboQuantMHAPool(MHATokenToKVPool):
    """Subclass that owns TurboQuant-specific buffers in addition to
    the parent's bf16 K/V cache.

    Construction mirrors ``MHATokenToKVPool`` -- only extra arg is
    ``turboquant_config``. All existing SGLang callers that pass a
    standard pool construction kwargs dict keep working; the backend
    picks up the quant buffers via ``get_quant_buffers(layer_id)``.
    """

    def __init__(
        self,
        *args,
        turboquant_config: TurboQuantConfig,
        **kwargs,
    ) -> None:
        turboquant_config.validate()
        self._tq_cfg = turboquant_config

        # Split mode: load the outlier mask file once so we know the
        # per-slice channel counts before allocating buffers.
        self._outlier_mask: Optional[OutlierMask] = None
        if turboquant_config.is_split:
            self._outlier_mask = OutlierMask.load(
                turboquant_config.outlier_mask_path
            )

        super().__init__(*args, **kwargs)
        self._create_quant_buffers()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------
    def _create_quant_buffers(self) -> None:
        cfg = self._tq_cfg
        if cfg.is_split:
            self._create_split_buffers()
        else:
            self._create_homog_buffers()

    def _create_homog_buffers(self) -> None:
        cfg = self._tq_cfg
        num_slots = self.size + self.page_size
        H_kv = self.head_num
        d = self.head_dim
        assert d == self.v_head_dim, (
            "TurboQuant MVP assumes K and V share head_dim; got "
            f"qk={d}, v={self.v_head_dim}"
        )

        idx_last = _idx_last_dim_homog(cfg, d)
        norm_dtype = _NORM_TORCH_DTYPE[cfg.norm_dtype]
        rnorm_dtype = _RNORM_TORCH_DTYPE[cfg.rnorm_dtype]
        device = self.device

        def _u8(shape):
            return torch.zeros(shape, dtype=torch.uint8, device=device)

        def _norm(shape):
            return torch.zeros(shape, dtype=norm_dtype, device=device)

        def _rnorm(shape):
            return torch.zeros(shape, dtype=rnorm_dtype, device=device)

        shape_idx = (num_slots, H_kv, idx_last)
        shape_meta = (num_slots, H_kv)
        shape_qjl = (num_slots, H_kv, d // 8)

        L = self.layer_num
        self._tq_k_idx = [_u8(shape_idx) for _ in range(L)]
        self._tq_v_idx = [_u8(shape_idx) for _ in range(L)]
        self._tq_k_norm = [_norm(shape_meta) for _ in range(L)]
        self._tq_v_norm = [_norm(shape_meta) for _ in range(L)]

        if cfg.algo == "prod":
            # tight_pack merges qjl into idx nibble -- no separate sign buffer
            if cfg.tight_pack and cfg.is_homog and cfg.bits == 4:
                self._tq_k_qjl_sign: list[torch.Tensor] = []
                self._tq_v_qjl_sign: list[torch.Tensor] = []
            else:
                if d % 8 != 0:
                    raise ValueError(
                        f"head_dim={d} must be divisible by 8 for QJL bit-pack"
                    )
                self._tq_k_qjl_sign = [_u8(shape_qjl) for _ in range(L)]
                self._tq_v_qjl_sign = [_u8(shape_qjl) for _ in range(L)]
            self._tq_k_rnorm = [_rnorm(shape_meta) for _ in range(L)]
            self._tq_v_rnorm = [_rnorm(shape_meta) for _ in range(L)]
        else:
            self._tq_k_qjl_sign = []
            self._tq_v_qjl_sign = []
            self._tq_k_rnorm = []
            self._tq_v_rnorm = []

    def _create_split_buffers(self) -> None:
        """Split mode: 8 buffers per (K side, V side) = 16 per layer.

        Outlier slice (d_out channels, bits_outlier):
          idx / norm / qjl_sign / rnorm
        Regular slice (d_reg channels, bits_regular): same.
        K and V each get their own pair.
        """
        cfg = self._tq_cfg
        assert self._outlier_mask is not None
        num_slots = self.size + self.page_size
        H_kv = self.head_num
        d = self.head_dim
        d_out = self._outlier_mask.num_outliers
        d_reg = d - d_out
        device = self.device
        norm_dtype = _NORM_TORCH_DTYPE[cfg.norm_dtype]
        rnorm_dtype = _RNORM_TORCH_DTYPE[cfg.rnorm_dtype]

        main_out = cfg.bits_outlier - 1 if cfg.algo == "prod" else cfg.bits_outlier
        main_reg = cfg.bits_regular - 1 if cfg.algo == "prod" else cfg.bits_regular
        pack_out = _pow2_ceil(main_out)
        pack_reg = _pow2_ceil(main_reg)

        if (d_out * pack_out) % 8 != 0:
            raise ValueError(
                f"d_out={d_out} * pack_bits={pack_out} must be byte-aligned"
            )
        if (d_reg * pack_reg) % 8 != 0:
            raise ValueError(
                f"d_reg={d_reg} * pack_bits={pack_reg} must be byte-aligned"
            )
        if cfg.algo == "prod" and (d_out % 8 != 0 or d_reg % 8 != 0):
            raise ValueError(
                f"Both d_out={d_out} and d_reg={d_reg} must be "
                "divisible by 8 for QJL bit-pack"
            )

        idx_last_out = d_out * pack_out // 8
        idx_last_reg = d_reg * pack_reg // 8

        def _u8(shape):
            return torch.zeros(shape, dtype=torch.uint8, device=device)

        def _norm(shape):
            return torch.zeros(shape, dtype=norm_dtype, device=device)

        def _rnorm(shape):
            return torch.zeros(shape, dtype=rnorm_dtype, device=device)

        shape_idx_out = (num_slots, H_kv, idx_last_out)
        shape_idx_reg = (num_slots, H_kv, idx_last_reg)
        shape_meta = (num_slots, H_kv)
        shape_qjl_out = (num_slots, H_kv, d_out // 8)
        shape_qjl_reg = (num_slots, H_kv, d_reg // 8)

        L = self.layer_num
        is_prod = cfg.algo == "prod"

        # K side
        self._tq_k_idx_out = [_u8(shape_idx_out) for _ in range(L)]
        self._tq_k_idx_reg = [_u8(shape_idx_reg) for _ in range(L)]
        self._tq_k_norm_out = [_norm(shape_meta) for _ in range(L)]
        self._tq_k_norm_reg = [_norm(shape_meta) for _ in range(L)]
        self._tq_k_qjl_sign_out = (
            [_u8(shape_qjl_out) for _ in range(L)] if is_prod else []
        )
        self._tq_k_qjl_sign_reg = (
            [_u8(shape_qjl_reg) for _ in range(L)] if is_prod else []
        )
        self._tq_k_rnorm_out = (
            [_rnorm(shape_meta) for _ in range(L)] if is_prod else []
        )
        self._tq_k_rnorm_reg = (
            [_rnorm(shape_meta) for _ in range(L)] if is_prod else []
        )
        # V side mirrors K
        self._tq_v_idx_out = [_u8(shape_idx_out) for _ in range(L)]
        self._tq_v_idx_reg = [_u8(shape_idx_reg) for _ in range(L)]
        self._tq_v_norm_out = [_norm(shape_meta) for _ in range(L)]
        self._tq_v_norm_reg = [_norm(shape_meta) for _ in range(L)]
        self._tq_v_qjl_sign_out = (
            [_u8(shape_qjl_out) for _ in range(L)] if is_prod else []
        )
        self._tq_v_qjl_sign_reg = (
            [_u8(shape_qjl_reg) for _ in range(L)] if is_prod else []
        )
        self._tq_v_rnorm_out = (
            [_rnorm(shape_meta) for _ in range(L)] if is_prod else []
        )
        self._tq_v_rnorm_reg = (
            [_rnorm(shape_meta) for _ in range(L)] if is_prod else []
        )

    # ------------------------------------------------------------------
    # Accessors for the attention backend
    # ------------------------------------------------------------------
    def get_quant_buffers(self, layer_id: int) -> dict[str, Optional[torch.Tensor]]:
        """Return the per-layer TurboQuant buffers as a dict.

        Homog mode: cache_k_idx, cache_k_norm, cache_v_idx, cache_v_norm
            (+ cache_k_qjl_sign, cache_k_rnorm, cache_v_qjl_sign,
            cache_v_rnorm for Q_prod; qjl_sign fields are None in
            tight-pack mode).

        Split mode: 16 keys with _out / _reg suffix for each slice.
        """
        i = layer_id - self.start_layer

        def _maybe(tensors: list[torch.Tensor]) -> Optional[torch.Tensor]:
            return tensors[i] if tensors else None

        if self._tq_cfg.is_split:
            return {
                # K outlier
                "cache_k_idx_out":      self._tq_k_idx_out[i],
                "cache_k_norm_out":     self._tq_k_norm_out[i],
                "cache_k_qjl_sign_out": _maybe(self._tq_k_qjl_sign_out),
                "cache_k_rnorm_out":    _maybe(self._tq_k_rnorm_out),
                # K regular
                "cache_k_idx_reg":      self._tq_k_idx_reg[i],
                "cache_k_norm_reg":     self._tq_k_norm_reg[i],
                "cache_k_qjl_sign_reg": _maybe(self._tq_k_qjl_sign_reg),
                "cache_k_rnorm_reg":    _maybe(self._tq_k_rnorm_reg),
                # V outlier
                "cache_v_idx_out":      self._tq_v_idx_out[i],
                "cache_v_norm_out":     self._tq_v_norm_out[i],
                "cache_v_qjl_sign_out": _maybe(self._tq_v_qjl_sign_out),
                "cache_v_rnorm_out":    _maybe(self._tq_v_rnorm_out),
                # V regular
                "cache_v_idx_reg":      self._tq_v_idx_reg[i],
                "cache_v_norm_reg":     self._tq_v_norm_reg[i],
                "cache_v_qjl_sign_reg": _maybe(self._tq_v_qjl_sign_reg),
                "cache_v_rnorm_reg":    _maybe(self._tq_v_rnorm_reg),
            }
        return {
            "cache_k_idx":      self._tq_k_idx[i],
            "cache_k_norm":     self._tq_k_norm[i],
            "cache_v_idx":      self._tq_v_idx[i],
            "cache_v_norm":     self._tq_v_norm[i],
            "cache_k_qjl_sign": _maybe(self._tq_k_qjl_sign),
            "cache_k_rnorm":    _maybe(self._tq_k_rnorm),
            "cache_v_qjl_sign": _maybe(self._tq_v_qjl_sign),
            "cache_v_rnorm":    _maybe(self._tq_v_rnorm),
        }

    @property
    def turboquant_config(self) -> TurboQuantConfig:
        return self._tq_cfg

    @property
    def outlier_mask(self) -> Optional[OutlierMask]:
        return self._outlier_mask

    # ------------------------------------------------------------------
    # KV write (Day 3-4 wires in real store; MVP falls through to bf16)
    # ------------------------------------------------------------------
    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ) -> None:
        # Preserve the standard bf16 path so radix cache / disagg /
        # cpu offload keep working during MVP.
        super().set_kv_buffer(
            layer, loc, cache_k, cache_v,
            k_scale=k_scale, v_scale=v_scale,
            layer_id_override=layer_id_override,
        )
        # TODO(Day 3-4): turboquant_store_kv(...) writes quant buffers.
        # Shape: cache_k is (num_tokens, H_kv*head_dim); loc is
        # 1-D absolute slot indices. Stays no-op until store.py is
        # ported from vllm_fork/vllm/turboquant/store.py.

    def get_kv_size_bytes(self):
        # Report bf16 + quant bytes so logs reflect actual GPU footprint.
        k_bytes, v_bytes = super().get_kv_size_bytes()
        extra = 0
        if self._tq_cfg.is_split:
            buf_lists = (
                self._tq_k_idx_out, self._tq_k_idx_reg,
                self._tq_k_norm_out, self._tq_k_norm_reg,
                self._tq_k_qjl_sign_out, self._tq_k_qjl_sign_reg,
                self._tq_k_rnorm_out, self._tq_k_rnorm_reg,
                self._tq_v_idx_out, self._tq_v_idx_reg,
                self._tq_v_norm_out, self._tq_v_norm_reg,
                self._tq_v_qjl_sign_out, self._tq_v_qjl_sign_reg,
                self._tq_v_rnorm_out, self._tq_v_rnorm_reg,
            )
        else:
            buf_lists = (
                self._tq_k_idx, self._tq_v_idx,
                self._tq_k_norm, self._tq_v_norm,
                self._tq_k_qjl_sign, self._tq_v_qjl_sign,
                self._tq_k_rnorm, self._tq_v_rnorm,
            )
        for lst in buf_lists:
            for t in lst:
                extra += t.numel() * t.element_size()
        # Split roughly between k and v (they're symmetric).
        return k_bytes + extra // 2, v_bytes + (extra - extra // 2)
