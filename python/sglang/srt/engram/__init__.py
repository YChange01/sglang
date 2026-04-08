"""Engram: Cross-node embedding pool for LLM inference via UB/URMA.

Offloads large Engram embedding tables to yuanrong-datasystem distributed
cache, using URMA fine-grained remote memory access for low-latency
sparse retrieval during inference.
"""

from sglang.srt.engram.engram_config import EngramPoolConfig

__all__ = ["EngramPoolConfig"]
