"""TurboQuant paper-faithful KV quantization (arXiv:2504.19874).

SGLang port of the algorithm layer originally implemented in
vllm/turboquant/. The algorithm code here has no SGLang dependency;
integration lives in:
  - sglang/srt/mem_cache/turboquant_pool.py  (KV storage subclass)
  - sglang/srt/layers/attention/turboquant_backend.py  (attention backend)

Exports the public surface used by the backend and external tests.
"""

from __future__ import annotations

from .codebook import QuantState
from .config import TurboQuantConfig
from .outlier import OutlierMask, SplitQuantState
from .sglang_adapter import (
    build_query_start_loc,
    sglang_paged_attention_tc,
    sglang_store_kv,
    sglang_store_v,
)
from .stages import (
    DEPRECATED_ALIASES,
    SILENT_ALIASES,
    STAGES,
    UnknownStageError,
    all_stage_names_including_aliases,
    list_stage_names,
    resolve_stage,
)

__all__ = [
    "TurboQuantConfig",
    "QuantState",
    "OutlierMask",
    "SplitQuantState",
    "STAGES",
    "SILENT_ALIASES",
    "DEPRECATED_ALIASES",
    "UnknownStageError",
    "resolve_stage",
    "list_stage_names",
    "all_stage_names_including_aliases",
    # SGLang attention adapter
    "sglang_store_kv",
    "sglang_store_v",
    "sglang_paged_attention_tc",
    "build_query_start_loc",
]
