# Scheme 5: URMA MMAP — Direct Remote Memory Access

This is the highest-performance scheme, equivalent to the CXL paper's mmap approach
but using UB/URMA hardware instead of CXL.

## How it works

```
Node1 (server):  malloc → fill data → urma_register_seg(URMA_SEG_MAP)
                                              ↓
                                    share seg_info.json
                                              ↓
Node2 (client):  urma_import_seg(URMA_SEG_MAP) → local VA mapped
                 table = (float*)mapped_addr
                 result = table[row_id * dim]  ← UBMMU routes to Node1 memory
```

## Build

On the machine with URMA SDK installed (umdk-urma-devel):

```bash
cd benchmark/engram/scheme5_urma_mmap
make
```

This produces `liburma_mmap.so` (for Python) and `test_server`/`test_client` (for C testing).

## Usage: C test programs

**Node1 (data server):**
```bash
./test_server 10000 341
# Prints segment info JSON — copy to Node2
# Keeps running until you press Enter
```

**Node2 (client):**
```bash
./test_client <seg_va> <seg_len> <seg_id> <eid_hex> <uasid> 10000 341
# Reads data and measures latency
```

## Usage: Python

**Node1:**
```python
from urma_mmap_py import UrmaMmapServer, SegInfo
import numpy as np

data = np.random.randn(10000, 341).astype(np.float32)
server = UrmaMmapServer()
info = server.register(data)
info.save("seg_info.json")  # copy to Node2
input("Press Enter to stop...")
server.destroy()
```

**Node2:**
```python
from urma_mmap_py import UrmaMmapClient, SegInfo

info = SegInfo.load("seg_info.json")
client = UrmaMmapClient()
table = client.mmap_import(info)

# Direct read — load/store via UBMMU, ~0.1 us per row
result = table[42]  # read row 42
batch = table[[0, 5, 42, 999]]  # batch read
```

## Unified benchmark

```bash
# Run all 5 schemes
python bench_all_schemes.py \
    --node1-host 10.0.0.1 --node1-port 18482 \
    --node2-host 10.0.0.2 --node2-port 18482 \
    --seg-info seg_info.json

# Run scheme 5 only
python bench_all_schemes.py --scheme 5 --seg-info seg_info.json
```

## Expected performance

| Metric | Scheme 5 (URMA MMAP) | Scheme 1 (KV TCP) | Speedup |
|--------|----------------------|--------------------|---------|
| Single row | ~0.5 us | ~150 us | 300x |
| 128 tokens | ~0.5 ms | ~18 ms | 36x |
| Throughput | ~2 GB/s | ~60 MB/s | 33x |
