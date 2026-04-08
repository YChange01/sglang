"""Configuration for Engram remote embedding pool."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngramPoolConfig:
    """Configuration for Engram embedding pool backed by yuanrong-datasystem.

    The pool stores large Engram embedding tables on remote Worker nodes,
    accessed via URMA/UB for fine-grained, low-latency sparse retrieval.
    """

    # Whether Engram pooling is enabled
    enabled: bool = False

    # --- Connection ---
    # yuanrong-datasystem Worker addresses
    worker_hosts: List[str] = field(default_factory=list)
    worker_ports: List[int] = field(default_factory=list)
    # ETCD address for cluster discovery (optional, direct connection if empty)
    etcd_address: str = ""
    # Authentication
    token: str = ""
    access_key: str = ""
    secret_key: str = ""

    # --- Engram Table Structure ---
    # Number of N-gram sub-tables (over_embedding_k * (over_embedding_n - 1))
    num_tables: int = 12
    # Embedding dimension per sub-table
    embedding_dim: int = 512
    # Max rows per sub-table
    table_capacity: int = 1_000_000

    # --- Transport ---
    connect_timeout_ms: int = 60000
    # Per-request timeout (small-packet reads should be fast)
    req_timeout_ms: int = 5000
    # Enable cross-node connection fallback
    enable_cross_node_connection: bool = False

    # --- Prefetch ---
    # Number of layers to prefetch ahead of the Engram layer
    prefetch_ahead_layers: int = 2
    # Max keys per batch KVClient.get() call
    max_batch_keys: int = 10000
    # Number of prefetch worker threads
    num_prefetch_workers: int = 4

    # --- Storage ---
    # Key prefix in yuanrong-datasystem
    key_prefix: str = "engram"
    # Shard strategy: "hash" (row_id % num_workers) or "range"
    shard_strategy: str = "hash"
    # Chunk size: number of rows packed into a single KV entry
    # Larger chunks reduce key count but increase per-read size.
    # Default 1 = one row per key (finest granularity, best for sparse access)
    chunk_size: int = 1

    @classmethod
    def from_server_args(cls, server_args) -> "EngramPoolConfig":
        """Build config from SGLang ServerArgs."""
        if not getattr(server_args, "engram_pool_enabled", False):
            return cls(enabled=False)

        hosts = getattr(server_args, "engram_pool_hosts", "").split(",")
        hosts = [h.strip() for h in hosts if h.strip()]

        ports_str = getattr(server_args, "engram_pool_ports", "")
        if ports_str:
            ports = [int(p.strip()) for p in ports_str.split(",") if p.strip()]
        else:
            ports = [18482] * len(hosts)

        return cls(
            enabled=True,
            worker_hosts=hosts,
            worker_ports=ports,
            etcd_address=getattr(server_args, "engram_pool_etcd", ""),
            connect_timeout_ms=getattr(
                server_args, "engram_pool_timeout_ms", 60000
            ),
            req_timeout_ms=getattr(server_args, "engram_pool_req_timeout_ms", 5000),
            prefetch_ahead_layers=getattr(
                server_args, "engram_pool_prefetch_ahead", 2
            ),
            num_prefetch_workers=getattr(
                server_args, "engram_pool_num_workers", 4
            ),
            chunk_size=getattr(server_args, "engram_pool_chunk_size", 1),
            key_prefix=getattr(server_args, "engram_pool_key_prefix", "engram"),
            shard_strategy=getattr(
                server_args, "engram_pool_shard_strategy", "hash"
            ),
            enable_cross_node_connection=getattr(
                server_args, "engram_pool_cross_node", False
            ),
        )

    @property
    def num_workers(self) -> int:
        return len(self.worker_hosts)

    def validate(self):
        """Validate config consistency."""
        if not self.enabled:
            return
        if not self.worker_hosts:
            raise ValueError("engram_pool_hosts must be set when engram pool is enabled")
        if len(self.worker_hosts) != len(self.worker_ports):
            raise ValueError(
                f"worker_hosts ({len(self.worker_hosts)}) and "
                f"worker_ports ({len(self.worker_ports)}) must have same length"
            )
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.shard_strategy not in ("hash", "range"):
            raise ValueError(
                f"shard_strategy must be 'hash' or 'range', got {self.shard_strategy}"
            )
