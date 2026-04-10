"""Engram pool with pluggable transport backends.

Backends:
  - local:   Embeddings in local DRAM (numpy array)
  - tcp:     Fetch from remote server via TCP socket
  - ubsmem:  ubs_mem load/store (UBMMU hardware page translation)
  - urma:    URMA RDMA READ (one-sided, via liburma_rw.so)

Each backend implements the same interface: given token IDs, return
embedding rows from all sub-tables.
"""

import ctypes
import logging
import os
import socket
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EngramConfig:
    """Engram table parameters."""

    num_tables: int = 12
    embedding_dim: int = 341
    vocab_size: int = 10000
    dtype: np.dtype = np.float32

    @property
    def row_bytes(self) -> int:
        return self.embedding_dim * np.dtype(self.dtype).itemsize

    @property
    def table_bytes(self) -> int:
        return self.vocab_size * self.row_bytes

    @property
    def total_bytes(self) -> int:
        return self.num_tables * self.table_bytes


# ------------------------------------------------------------------ #
#  Backend interface                                                  #
# ------------------------------------------------------------------ #


class EngramBackend(ABC):
    """Abstract backend for Engram embedding retrieval."""

    @abstractmethod
    def setup(self, config: EngramConfig) -> None:
        """Initialize the backend."""

    @abstractmethod
    def fetch_rows(
        self, table_id: int, row_ids: np.ndarray
    ) -> np.ndarray:
        """Fetch embedding rows for one sub-table.

        Args:
            table_id: Sub-table index [0, num_tables).
            row_ids: 1-D int32 array of row indices.

        Returns:
            np.ndarray of shape [len(row_ids), embedding_dim], float32.
        """

    @abstractmethod
    def teardown(self) -> None:
        """Release resources."""

    def fetch_all_tables(
        self, config: EngramConfig, token_ids: np.ndarray
    ) -> np.ndarray:
        """Fetch embeddings from all sub-tables for a batch of tokens.

        Simulates the Engram layer: each token looks up all sub-tables.

        Args:
            config: Engram configuration.
            token_ids: 1-D int array of token IDs (hashed to row indices).

        Returns:
            np.ndarray of shape [num_tables, len(token_ids), embedding_dim].
        """
        num_tokens = len(token_ids)
        result = np.empty(
            (config.num_tables, num_tokens, config.embedding_dim),
            dtype=config.dtype,
        )
        for t in range(config.num_tables):
            # Each table uses a different hash → different row IDs
            row_ids = (token_ids * (t + 1) + t * 1000003) % config.vocab_size
            row_ids = row_ids.astype(np.int32)
            result[t] = self.fetch_rows(t, row_ids)
        return result


# ------------------------------------------------------------------ #
#  Backend: Local DRAM                                                #
# ------------------------------------------------------------------ #


class LocalBackend(EngramBackend):
    """Embeddings stored in local numpy arrays."""

    def __init__(self) -> None:
        self.tables: List[np.ndarray] = []

    def setup(self, config: EngramConfig) -> None:
        self.tables = []
        for t in range(config.num_tables):
            # Deterministic data pattern (same as C server)
            table = np.arange(
                config.vocab_size * config.embedding_dim, dtype=np.float32
            ).reshape(config.vocab_size, config.embedding_dim)
            table *= 0.001
            self.tables.append(table)
        logger.info(
            "LocalBackend: %d tables, %d rows × %d dim (%.1f MB total)",
            config.num_tables,
            config.vocab_size,
            config.embedding_dim,
            config.total_bytes / 1e6,
        )

    def fetch_rows(self, table_id: int, row_ids: np.ndarray) -> np.ndarray:
        return self.tables[table_id][row_ids]

    def teardown(self) -> None:
        self.tables.clear()


# ------------------------------------------------------------------ #
#  Backend: TCP                                                       #
# ------------------------------------------------------------------ #


class TCPBackend(EngramBackend):
    """Fetch embeddings from remote server via TCP socket."""

    def __init__(self, server_ip: str, port: int = 13900) -> None:
        self.server_ip = server_ip
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.row_bytes = 0

    def setup(self, config: EngramConfig) -> None:
        self.row_bytes = config.row_bytes
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.server_ip, self.port))
        logger.info("TCPBackend: connected to %s:%d", self.server_ip, self.port)

    def fetch_rows(self, table_id: int, row_ids: np.ndarray) -> np.ndarray:
        # Server has flat buffer: offset = row_id * row_bytes
        # For multi-table: offset = (table_id * vocab_size + row_id) * row_bytes
        # Current server has single flat buffer, use row_id directly
        buf = bytearray(len(row_ids) * self.row_bytes)

        # Pipeline: send all requests, then read all responses
        for rid in row_ids:
            offset = int(rid) * self.row_bytes
            req = struct.pack("<QI", offset, self.row_bytes)
            self.sock.sendall(req)

        view = memoryview(buf)
        for i in range(len(row_ids)):
            start = i * self.row_bytes
            end = start + self.row_bytes
            self._recv_exact(view[start:end])

        return np.frombuffer(buf, dtype=np.float32).reshape(
            len(row_ids), -1
        )

    def _recv_exact(self, buf: memoryview) -> None:
        n = len(buf)
        pos = 0
        while pos < n:
            received = self.sock.recv_into(buf[pos:])
            if received == 0:
                raise ConnectionError("TCP connection closed")
            pos += received

    def teardown(self) -> None:
        if self.sock:
            # Send EOF
            eof = struct.pack("<QI", 0xFFFFFFFFFFFFFFFF, 0)
            try:
                self.sock.sendall(eof)
            except OSError:
                pass
            self.sock.close()
            self.sock = None


# ------------------------------------------------------------------ #
#  Backend: UBS-MEM (load/store)                                      #
# ------------------------------------------------------------------ #


class UBSMemBackend(EngramBackend):
    """ubs_mem shared memory via UBMMU load/store."""

    def __init__(
        self,
        shm_name: str = "engram_test",
        provider_host: str = "node1",
    ) -> None:
        self.shm_name = shm_name
        self.provider_host = provider_host
        self.lib: Optional[ctypes.CDLL] = None
        self.ptr: Optional[ctypes.c_void_p] = None
        self.buf_size = 0
        self.data: Optional[np.ndarray] = None

    def setup(self, config: EngramConfig) -> None:
        self.lib = ctypes.CDLL("libubsm_sdk.so")
        self.buf_size = config.vocab_size * config.row_bytes
        # Ensure buf_size >= 128 MB (obmm block alignment)
        if self.buf_size < 128 * 1024 * 1024:
            self.buf_size = 128 * 1024 * 1024

        # Initialize
        opts = ctypes.create_string_buffer(256)  # ubsmem_options_t
        self.lib.ubsmem_init_attributes(opts)
        ret = self.lib.ubsmem_initialize(opts)
        if ret != 0:
            raise RuntimeError(f"ubsmem_initialize failed: {ret}")
        self.lib.ubsmem_set_logger_level(3)  # error only

        # Map existing shmem
        self.ptr = ctypes.c_void_p()
        name_b = self.shm_name.encode()

        PROT_READ = 0x1
        MAP_SHARED = 0x01

        ret = self.lib.ubsmem_shmem_map(
            None,
            ctypes.c_size_t(self.buf_size),
            ctypes.c_int(PROT_READ),
            ctypes.c_int(MAP_SHARED),
            name_b,
            ctypes.c_long(0),
            ctypes.byref(self.ptr),
        )
        if ret != 0:
            raise RuntimeError(f"ubsmem_shmem_map failed: {ret}")

        # Wrap as numpy array (zero-copy)
        arr_type = ctypes.c_float * (self.buf_size // 4)
        arr = arr_type.from_address(self.ptr.value)
        self.data = np.ctypeslib.as_array(arr).reshape(-1, config.embedding_dim)
        logger.info(
            "UBSMemBackend: mapped %s, ptr=%s, shape=%s",
            self.shm_name,
            hex(self.ptr.value),
            self.data.shape,
        )

    def fetch_rows(self, table_id: int, row_ids: np.ndarray) -> np.ndarray:
        return self.data[row_ids].copy()  # memcpy from mapped remote memory

    def teardown(self) -> None:
        if self.lib and self.ptr and self.ptr.value:
            self.lib.ubsmem_shmem_unmap(self.ptr, ctypes.c_size_t(self.buf_size))
            self.lib.ubsmem_finalize()
        self.data = None


# ------------------------------------------------------------------ #
#  Backend: URMA (RDMA READ)                                          #
# ------------------------------------------------------------------ #


class URMABackend(EngramBackend):
    """URMA one-sided RDMA READ via liburma_rw.so."""

    def __init__(
        self, server_ip: str = "192.168.84.245", port: int = 13857
    ) -> None:
        self.server_ip = server_ip
        self.port = port
        self.lib: Optional[ctypes.CDLL] = None
        self.ctx: Optional[ctypes.c_void_p] = None
        self.row_bytes = 0
        self.local_buf: Optional[np.ndarray] = None

    def setup(self, config: EngramConfig) -> None:
        self.row_bytes = config.row_bytes

        # Load liburma_rw.so (built from lib/urma_rw.c)
        lib_path = os.environ.get(
            "URMA_RW_LIB", os.path.join(os.path.dirname(__file__), "../lib/liburma_rw.so")
        )
        self.lib = ctypes.CDLL(lib_path)

        # Set return types
        self.lib.urma_rw_init.restype = ctypes.c_void_p
        self.lib.urma_rw_get_buffer.restype = ctypes.c_void_p

        buf_size = 128 * 1024 * 1024  # 128 MB local buffer
        self.ctx = self.lib.urma_rw_init(None, ctypes.c_uint64(buf_size))
        if not self.ctx:
            raise RuntimeError("urma_rw_init failed")

        ret = self.lib.urma_rw_client_connect(
            self.ctx, self.server_ip.encode(), ctypes.c_uint16(self.port)
        )
        if ret != 0:
            raise RuntimeError(f"urma_rw_client_connect failed: {ret}")

        # Wrap local buffer as numpy array
        local_ptr = self.lib.urma_rw_get_buffer(self.ctx)
        arr_type = ctypes.c_float * (buf_size // 4)
        arr = arr_type.from_address(local_ptr)
        self.local_buf = np.ctypeslib.as_array(arr)
        logger.info("URMABackend: connected to %s:%d", self.server_ip, self.port)

    def fetch_rows(self, table_id: int, row_ids: np.ndarray) -> np.ndarray:
        n = len(row_ids)
        row_bytes = self.row_bytes

        locals_arr = (ctypes.c_uint64 * n)()
        remotes_arr = (ctypes.c_uint64 * n)()
        lens_arr = (ctypes.c_uint32 * n)()

        for i, rid in enumerate(row_ids):
            locals_arr[i] = i * row_bytes
            remotes_arr[i] = int(rid) * row_bytes
            lens_arr[i] = row_bytes

        ret = self.lib.urma_rw_read_batch(
            self.ctx, locals_arr, remotes_arr, lens_arr, ctypes.c_uint32(n)
        )
        if ret != 0:
            raise RuntimeError(f"urma_rw_read_batch failed: {ret}")

        dim = row_bytes // 4
        return self.local_buf[: n * dim].reshape(n, dim).copy()

    def teardown(self) -> None:
        if self.lib and self.ctx:
            self.lib.urma_rw_destroy(self.ctx)
        self.ctx = None
        self.local_buf = None


# ------------------------------------------------------------------ #
#  Factory                                                            #
# ------------------------------------------------------------------ #

BACKENDS = {
    "local": LocalBackend,
    "tcp": TCPBackend,
    "ubsmem": UBSMemBackend,
    "urma": URMABackend,
}


def create_backend(name: str, **kwargs) -> EngramBackend:
    """Create a backend by name."""
    cls = BACKENDS.get(name)
    if cls is None:
        raise ValueError(f"Unknown backend: {name}. Choose from {list(BACKENDS)}")
    return cls(**kwargs)
