# Scheme 5: URMA RW (urma_read-based cross-node access)

Cross-node Engram reading using **URMA one-sided read** (`urma_read` / `URMA_OPC_READ`).

This is the practical replacement for the true load/store approach, since the
current URMA SDK does not implement `URMA_SEG_MAPPED` on hns3 hardware.

## Architecture

```
Node1 (Server)                     Node2 (Client)
├── allocate 2GB buffer             ├── allocate local recv buf
├── fill with Engram data           ├── urma_init, create jetty
├── urma_register_seg               ├── TCP connect → exchange info
├── TCP listen on port 13857        ├── urma_import_seg(remote)
└── wait for client    ───────►     ├── urma_import_jetty(remote)
                                    ├── Loop:
                                    │   urma_post_jetty_send_wr(READ)
                                    │   urma_poll_jfc(completion)
                                    └── benchmark latency/throughput
```

## Build

On each node:

```bash
cd benchmark/engram/scheme5_urma_rw
make
```

## Usage

**Node1 (data node):**
```bash
./server 13857 2048    # port=13857, buffer=2048MB
# Fills buffer with deterministic pattern data[i] = i * 0.001
# Waits for client, keeps data alive
```

**Node2 (compute node):**
```bash
./client 192.168.84.245 13857 10000 341 200
#        <server_ip>    <port> <rows> <dim> <iters>
```

## Expected Output

```
[Bench 1] Single row read latency
  Avg: 3.5 us, Min: 2.8 us, Max: 12.1 us
  Throughput (single-row): 390 MB/s

[Bench 2] Batch read latency
  Batch    Data      Avg us     Throughput
  --------------------------------------------
  32       42.6 KB   45.2       943 MB/s
  128      170.5 KB  150.1      1136 MB/s
  256      341.0 KB  280.4      1217 MB/s

[Bench 3] Full Engram prefetch (12 tables x N tokens)
  Tokens   Data      Avg ms     Throughput
  --------------------------------------------
  32       511 KB    0.45       1135 MB/s
  64       1023 KB   0.82       1247 MB/s
  128      2046 KB   1.58       1295 MB/s
  256      4092 KB   3.05       1342 MB/s
```

## Performance Comparison

| Scheme | Single Read | 128 tokens | Note |
|--------|------------|------------|------|
| KV TCP (via ZMQ) | ~150 us | ~18 ms | RPC bound |
| KV IPC | ~100 us | ~17 ms | IPC still has RPC |
| ObjectClient SHM (same-node) | ~0.5 us | <1 ms | memoryview direct |
| **URMA RW (this, cross-node)** | **~3-5 us** | **~1-2 ms** | One-sided read |
| URMA MMAP (not impl) | ~0.5 us | <1 ms | Would need driver patch |

## Why This Works

- **No RPC, no ZMQ**: Direct from client process to remote memory
- **No Worker intermediary**: Bypasses yuanrong-datasystem entirely
- **One-sided**: Remote side doesn't need to do anything (remote CPU unused)
- **Batch amortization**: One WR submission for many rows, one poll for all

## Why It's Not Pure Load/Store

- Each read still requires posting a Work Request (~1-3 us overhead)
- Completion polling adds latency
- Minimum granularity is 64B due to WR fixed overhead
- True load/store would be <0.5 us per row
