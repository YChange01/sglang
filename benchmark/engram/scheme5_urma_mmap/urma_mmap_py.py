"""Python wrapper for liburma_mmap.so — direct URMA memory mapping.

Usage (server side, Node1):
    server = UrmaMmapServer()
    info = server.register(data_array)
    info.save("seg_info.json")  # share with client

Usage (client side, Node2):
    client = UrmaMmapClient()
    info = SegInfo.load("seg_info.json")
    table = client.mmap_import(info, num_rows=10000, dim=341)
    result = table[row_ids]  # direct load/store, ~0.1 us
"""

import ctypes
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Find the shared library
_LIB_PATHS = [
    Path(__file__).parent / "liburma_mmap.so",
    Path("./liburma_mmap.so"),
]

_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    for p in _LIB_PATHS:
        if p.exists():
            _lib = ctypes.CDLL(str(p))
            return _lib
    raise RuntimeError(
        "liburma_mmap.so not found. Build it first:\n"
        "  cd benchmark/engram/scheme5_urma_mmap && make"
    )


# ------------------------------------------------------------------ #
#  Segment info (exchanged between server and client)
# ------------------------------------------------------------------ #

class SegInfo:
    """Segment info for URMA import."""

    def __init__(self, eid_hex: str, uasid: int, seg_va: int, seg_len: int,
                 seg_id: int, token: int = 0xACFE,
                 num_rows: int = 0, dim: int = 0):
        self.eid_hex = eid_hex
        self.uasid = uasid
        self.seg_va = seg_va
        self.seg_len = seg_len
        self.seg_id = seg_id
        self.token = token
        self.num_rows = num_rows
        self.dim = dim

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SegInfo":
        with open(path) as f:
            return cls(**json.load(f))

    def to_c_struct(self):
        """Convert to ctypes struct for C API."""

        class CSegInfo(ctypes.Structure):
            _fields_ = [
                ("eid", ctypes.c_char * 64),
                ("uasid", ctypes.c_uint32),
                ("seg_va", ctypes.c_uint64),
                ("seg_len", ctypes.c_uint64),
                ("seg_id", ctypes.c_uint32),
                ("token", ctypes.c_uint32),
            ]

        info = CSegInfo()
        eid_bytes = bytes.fromhex(self.eid_hex)
        info.eid = eid_bytes[:64].ljust(64, b'\x00')
        info.uasid = self.uasid
        info.seg_va = self.seg_va
        info.seg_len = self.seg_len
        info.seg_id = self.seg_id
        info.token = self.token
        return info


# ------------------------------------------------------------------ #
#  Server: register local memory
# ------------------------------------------------------------------ #

class UrmaMmapServer:
    """Register local memory as URMA segment for remote access."""

    def __init__(self, dev_name: Optional[str] = None, eid_index: int = -1):
        lib = _load_lib()
        lib.urma_mmap_init.restype = ctypes.c_void_p
        self._ctx = lib.urma_mmap_init(
            dev_name.encode() if dev_name else None,
            eid_index,
        )
        if not self._ctx:
            raise RuntimeError("urma_mmap_init failed")
        self._lib = lib

    def register(self, data: np.ndarray) -> SegInfo:
        """Register a numpy array as URMA segment.

        The array must stay alive as long as the segment is registered.
        """
        if not data.flags["C_CONTIGUOUS"]:
            raise ValueError("Array must be C-contiguous")

        class CSegInfo(ctypes.Structure):
            _fields_ = [
                ("eid", ctypes.c_char * 64),
                ("uasid", ctypes.c_uint32),
                ("seg_va", ctypes.c_uint64),
                ("seg_len", ctypes.c_uint64),
                ("seg_id", ctypes.c_uint32),
                ("token", ctypes.c_uint32),
            ]

        info = CSegInfo()
        addr = data.ctypes.data_as(ctypes.c_void_p)
        size = ctypes.c_uint64(data.nbytes)

        rc = self._lib.urma_mmap_register(
            ctypes.c_void_p(self._ctx), addr, size, ctypes.byref(info)
        )
        if rc != 0:
            raise RuntimeError(f"urma_mmap_register failed: {rc}")

        eid_hex = bytes(info.eid).hex()
        return SegInfo(
            eid_hex=eid_hex,
            uasid=info.uasid,
            seg_va=info.seg_va,
            seg_len=info.seg_len,
            seg_id=info.seg_id,
            token=info.token,
            num_rows=data.shape[0] if data.ndim >= 1 else 0,
            dim=data.shape[1] if data.ndim >= 2 else data.shape[0],
        )

    def destroy(self):
        if self._ctx:
            self._lib.urma_mmap_destroy(ctypes.c_void_p(self._ctx))
            self._ctx = None


# ------------------------------------------------------------------ #
#  Client: import + mmap remote memory
# ------------------------------------------------------------------ #

class UrmaMmapClient:
    """Import remote URMA segment and access via direct load/store."""

    def __init__(self, dev_name: Optional[str] = None, eid_index: int = -1):
        lib = _load_lib()
        lib.urma_mmap_init.restype = ctypes.c_void_p
        self._ctx = lib.urma_mmap_init(
            dev_name.encode() if dev_name else None,
            eid_index,
        )
        if not self._ctx:
            raise RuntimeError("urma_mmap_init failed")
        self._lib = lib
        self._mapped_tables = {}  # seg_id -> numpy array

    def mmap_import(self, info: SegInfo, num_rows: int = 0,
                    dim: int = 0) -> np.ndarray:
        """Import remote segment and return as numpy array.

        The returned array points directly to the mapped remote memory.
        Every element access is a load/store through UBMMU — no RPC.
        """
        c_info = info.to_c_struct()
        mapped_ptr = ctypes.c_void_p()

        rc = self._lib.urma_mmap_import(
            ctypes.c_void_p(self._ctx),
            ctypes.byref(c_info),
            ctypes.byref(mapped_ptr),
        )
        if rc != 0:
            raise RuntimeError(f"urma_mmap_import failed: {rc}")

        nrows = num_rows or info.num_rows
        d = dim or info.dim
        if nrows == 0 or d == 0:
            raise ValueError("num_rows and dim must be specified")

        # Wrap the mapped address as a numpy array (zero-copy)
        arr_type = ctypes.c_float * (nrows * d)
        arr = np.ctypeslib.as_array(
            arr_type.from_address(mapped_ptr.value)
        ).reshape(nrows, d)

        self._mapped_tables[info.seg_id] = arr
        return arr

    def destroy(self):
        if self._ctx:
            self._lib.urma_mmap_destroy(ctypes.c_void_p(self._ctx))
            self._ctx = None
            self._mapped_tables.clear()


# ------------------------------------------------------------------ #
#  Standalone benchmark
# ------------------------------------------------------------------ #

def bench_mmap_read(table: np.ndarray, num_iters: int = 1000):
    """Benchmark direct load/store read from mapped remote memory."""
    nrows, dim = table.shape

    # Single row
    latencies = []
    for _ in range(num_iters):
        rid = np.random.randint(0, nrows)
        t0 = time.perf_counter()
        _ = table[rid].copy()  # force actual read
        latencies.append((time.perf_counter() - t0) * 1e6)

    avg = np.mean(latencies)
    p99 = np.percentile(latencies, 99)
    print(f"  Single row: avg={avg:.2f} us, p99={p99:.2f} us")

    # Batch read
    for batch in [32, 128, 256]:
        latencies = []
        for _ in range(num_iters // 5):
            rids = np.random.randint(0, nrows, size=batch)
            t0 = time.perf_counter()
            _ = table[rids].copy()  # fancy indexing on mapped memory
            latencies.append((time.perf_counter() - t0) * 1e6)

        avg = np.mean(latencies)
        data_kb = batch * dim * 4 / 1024
        tp = (data_kb / 1024) / (avg / 1e6) if avg > 0 else 0
        print(f"  Batch {batch:>4}: avg={avg:.1f} us, {tp:.0f} MB/s ({data_kb:.0f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-info", required=True, help="Path to seg_info.json")
    parser.add_argument("--num-rows", type=int, default=0)
    parser.add_argument("--dim", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=1000)
    args = parser.parse_args()

    info = SegInfo.load(args.seg_info)
    print(f"Importing: seg_va=0x{info.seg_va:x}, len={info.seg_len}, "
          f"rows={info.num_rows}, dim={info.dim}")

    client = UrmaMmapClient()
    table = client.mmap_import(info, args.num_rows, args.dim)
    print(f"Mapped: shape={table.shape}, dtype={table.dtype}")
    print(f"  row[0][:4] = {table[0][:4]}")
    print(f"  row[42][:4] = {table[42][:4]}")

    print("\nBenchmark:")
    bench_mmap_read(table, args.num_iters)

    client.destroy()
