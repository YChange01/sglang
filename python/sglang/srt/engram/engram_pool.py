"""Engram remote embedding pool backed by yuanrong-datasystem.

Stores large Engram embedding tables on remote Worker nodes and provides
batch lookup via KVClient. The underlying UB/URMA transport delivers
fine-grained, low-latency access suitable for sparse embedding retrieval.
"""

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.engram.engram_config import EngramPoolConfig

logger = logging.getLogger(__name__)

# Lazy import yuanrong-datasystem SDK to avoid hard dependency
_kv_client_cls = None


def _import_kv_client():
    global _kv_client_cls
    if _kv_client_cls is None:
        try:
            from yr.datasystem import KVClient

            _kv_client_cls = KVClient
        except ImportError:
            raise ImportError(
                "yr.datasystem is required for Engram pool. "
                "Install yuanrong-datasystem Python SDK first."
            )
    return _kv_client_cls


class EngramPool:
    """Client wrapper for Engram embedding storage on yuanrong-datasystem.

    Manages connections to multiple Worker nodes and provides batch
    embedding lookup with sharding support.

    Key format: ``{prefix}:t{table_id}:c{chunk_id}``
    where chunk_id = row_id // chunk_size.

    Each KV value stores ``chunk_size`` consecutive embedding rows packed
    as raw float bytes.
    """

    def __init__(self, config: EngramPoolConfig):
        config.validate()
        self.config = config
        self._clients: List = []  # KVClient per Worker
        # Separate executors to prevent nested-submit deadlock:
        # _io_executor: handles KVClient.get() calls inside batch_lookup
        # _prefetch_executor: handles batch_lookup_async() calls from prefetcher
        self._io_executor = ThreadPoolExecutor(
            max_workers=max(config.num_workers, 1) * 2,
            thread_name_prefix="engram-io",
        )
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=max(config.num_prefetch_workers, config.num_tables),
            thread_name_prefix="engram-prefetch",
        )
        self._init_lock = threading.Lock()
        self._initialized = False
        # Cache: table metadata after loading
        self._table_meta: Dict[int, dict] = {}

    def initialize(self):
        """Establish connections to all Workers (thread-safe)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._do_initialize()

    def _do_initialize(self):
        KVClient = _import_kv_client()

        for host, port in zip(self.config.worker_hosts, self.config.worker_ports):
            client = KVClient(
                host=host,
                port=port,
                timeout_ms=self.config.connect_timeout_ms,
                req_timeout_ms=self.config.req_timeout_ms,
                enable_cross_node_connection=self.config.enable_cross_node_connection,
            )
            client.init()
            self._clients.append(client)
            logger.info("Engram pool: connected to Worker %s:%d", host, port)

        self._initialized = True
        logger.info(
            "Engram pool initialized with %d Workers, shard=%s, chunk=%d",
            len(self._clients),
            self.config.shard_strategy,
            self.config.chunk_size,
        )

    def _shard_for_chunk(self, table_id: int, chunk_id: int) -> int:
        """Determine which Worker stores this chunk."""
        if self.config.num_workers == 1:
            return 0
        if self.config.shard_strategy == "hash":
            return (table_id * 1000003 + chunk_id) % self.config.num_workers
        else:  # range
            total_chunks = self.config.table_capacity // max(self.config.chunk_size, 1)
            chunks_per_worker = max(total_chunks // self.config.num_workers, 1)
            return min(chunk_id // chunks_per_worker, self.config.num_workers - 1)

    def _make_key(self, table_id: int, chunk_id: int) -> str:
        return f"{self.config.key_prefix}:t{table_id}:c{chunk_id}"

    # ------------------------------------------------------------------ #
    #  Table Loading (called once at initialization by rank 0)
    # ------------------------------------------------------------------ #

    def load_table(self, table_id: int, weights: torch.Tensor):
        """Write embedding table to the remote pool.

        Args:
            table_id: Sub-table index.
            weights: Tensor of shape [num_rows, embedding_dim], float type.
        """
        num_rows, embed_dim = weights.shape
        chunk_size = self.config.chunk_size
        weights_np = weights.detach().cpu().to(torch.float32).numpy()

        # Group chunks by target Worker for batch mset
        worker_batches: Dict[int, Tuple[List[str], List[bytes]]] = {
            i: ([], []) for i in range(self.config.num_workers)
        }

        num_chunks = (num_rows + chunk_size - 1) // chunk_size
        for cid in range(num_chunks):
            row_start = cid * chunk_size
            row_end = min(row_start + chunk_size, num_rows)
            chunk_data = weights_np[row_start:row_end].tobytes()
            key = self._make_key(table_id, cid)
            worker_id = self._shard_for_chunk(table_id, cid)
            worker_batches[worker_id][0].append(key)
            worker_batches[worker_id][1].append(chunk_data)

        # Write to each Worker in parallel
        futures = []
        for worker_id, (keys, vals) in worker_batches.items():
            if not keys:
                continue
            fut = self._io_executor.submit(self._mset_worker, worker_id, keys, vals)
            futures.append(fut)

        for fut in futures:
            fut.result()  # Raise on error

        self._table_meta[table_id] = {
            "num_rows": num_rows,
            "embed_dim": embed_dim,
            "num_chunks": num_chunks,
            "store_dtype": np.float32,
        }
        logger.info(
            "Engram pool: loaded table %d (%d rows, dim=%d) → %d chunks across %d Workers",
            table_id,
            num_rows,
            embed_dim,
            num_chunks,
            self.config.num_workers,
        )

    def _mset_worker(self, worker_id: int, keys: List[str], vals: List[bytes]):
        """Batch-set on a single Worker, respecting the 2000-key limit."""
        client = self._clients[worker_id]
        batch_limit = 2000
        for i in range(0, len(keys), batch_limit):
            batch_keys = keys[i : i + batch_limit]
            batch_vals = vals[i : i + batch_limit]
            failed = client.mset(batch_keys, batch_vals)
            if failed:
                raise RuntimeError(
                    f"Engram pool: mset failed on Worker {worker_id}, "
                    f"{len(failed)} keys failed"
                )

    # ------------------------------------------------------------------ #
    #  Batch Lookup (called per forward pass)
    # ------------------------------------------------------------------ #

    def batch_lookup(
        self,
        table_id: int,
        row_ids: np.ndarray,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """Synchronous batch lookup of embedding rows.

        Args:
            table_id: Sub-table index.
            row_ids: 1-D int array of row indices to fetch.
            dtype: Target numpy dtype (default float32).

        Returns:
            np.ndarray of shape [len(row_ids), embedding_dim].
        """
        if len(row_ids) == 0:
            meta = self._table_meta.get(table_id, {})
            embed_dim = meta.get("embed_dim", self.config.embedding_dim)
            return np.empty((0, embed_dim), dtype=dtype)

        if table_id not in self._table_meta:
            raise RuntimeError(
                f"Engram pool: table {table_id} not loaded — call load_table first"
            )

        chunk_size = self.config.chunk_size
        embed_dim = self._table_meta[table_id]["embed_dim"]
        store_dtype = self._table_meta[table_id].get("store_dtype", np.float32)
        row_bytes = embed_dim * np.dtype(store_dtype).itemsize

        # Deduplicate chunks and group by Worker
        chunk_ids = row_ids // chunk_size
        unique_chunks = np.unique(chunk_ids)

        worker_chunks: Dict[int, List[int]] = {}
        for cid in unique_chunks:
            wid = self._shard_for_chunk(table_id, int(cid))
            worker_chunks.setdefault(wid, []).append(int(cid))

        # Fetch from Workers using _io_executor (never _prefetch_executor,
        # since batch_lookup may itself be running inside _prefetch_executor)
        chunk_data: Dict[int, bytes] = {}

        if len(worker_chunks) <= 1:
            for wid, cids in worker_chunks.items():
                keys = [self._make_key(table_id, cid) for cid in cids]
                values = self._get_worker(wid, keys)
                for cid, val in zip(cids, values):
                    if val is None:
                        raise RuntimeError(
                            f"Engram pool: key miss for table={table_id} chunk={cid}"
                        )
                    chunk_data[cid] = val
        else:
            fetch_futures: List[Tuple[int, List[int], Future]] = []
            for wid, cids in worker_chunks.items():
                keys = [self._make_key(table_id, cid) for cid in cids]
                fut = self._io_executor.submit(self._get_worker, wid, keys)
                fetch_futures.append((wid, cids, fut))
            for wid, cids, fut in fetch_futures:
                values = fut.result()
                for cid, val in zip(cids, values):
                    if val is None:
                        raise RuntimeError(
                            f"Engram pool: key miss for table={table_id} chunk={cid}"
                        )
                    chunk_data[cid] = val

        # Assemble result
        result = np.empty((len(row_ids), embed_dim), dtype=store_dtype)
        for i, rid in enumerate(row_ids):
            cid = rid // chunk_size
            offset_in_chunk = rid % chunk_size
            raw = chunk_data[cid]
            start = offset_in_chunk * row_bytes
            end = start + row_bytes
            if len(raw) < end:
                raise RuntimeError(
                    f"Engram pool: truncated data for table={table_id} "
                    f"chunk={cid}, expected {end} bytes got {len(raw)}"
                )
            result[i] = np.frombuffer(raw[start:end], dtype=store_dtype)

        if dtype != store_dtype:
            result = result.astype(dtype)
        return result

    def batch_lookup_async(
        self,
        table_id: int,
        row_ids: np.ndarray,
        dtype: np.dtype = np.float32,
    ) -> Future:
        """Asynchronous batch lookup. Returns a Future[np.ndarray]."""
        return self._prefetch_executor.submit(self.batch_lookup, table_id, row_ids, dtype)

    def _get_worker(self, worker_id: int, keys: List[str]) -> List[bytes]:
        """Batch-get from a single Worker."""
        client = self._clients[worker_id]
        values = client.get(keys, sub_timeout_ms=self.config.req_timeout_ms)
        return values

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def is_table_loaded(self, table_id: int) -> bool:
        return table_id in self._table_meta

    def get_table_meta(self, table_id: int) -> Optional[dict]:
        return self._table_meta.get(table_id)

    def shutdown(self):
        """Release resources (waits for in-flight tasks to complete)."""
        self._prefetch_executor.shutdown(wait=True)
        self._io_executor.shutdown(wait=True)
        self._clients.clear()
        self._initialized = False
        logger.info("Engram pool shut down")
