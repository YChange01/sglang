# Scheme 6: UB Shared Memory (ubs_mem)

Cross-node load/store access to Engram tables using the **ubs_mem SDK**
(Matrix Core Memory Package), which layers on top of the `obmm` kernel
UAPI. After `ubsmem_shmem_map` returns, the caller has a plain virtual
address whose backing pages live on a peer node and are routed by UBMMU
over the UB fabric. No explicit read/write calls, no post/poll — just
`memcpy` or direct pointer dereference.

## vs scheme 5 (urma_read)

| aspect | scheme 5 | scheme 6 |
|---|---|---|
| Data path | `urma_post_jetty_send_wr(READ) + urma_poll_jfc` | Plain load/store via UBMMU |
| Setup | TCP handle exchange + jetty/JFC/JFR | Name-based discovery via `ubsmd` |
| Host overhead per read | ~50 instructions to post + poll | 1 load instruction |
| Single-read latency | 2.65 us | **sub-microsecond expected** |
| Lines of user code | ~600 | ~400 |

## Architecture

```
  App (server/client)
       │
       │ #include "ubmem_rw.h"
       │
       └─> libubsm_sdk.so (/usr/local/ubs_mem/lib)
               │
               └─> UDS /run/matrix/memory/MxmAgentIpcServer
                        │
                        └─> ubsmd daemon
                                │
                                └─> ioctl(/dev/obmm) + mmap
                                          │
                                          └─> obmm + ummu + ubmem_vmmu kernel
                                                    │
                                                    └─> UBMMU HW + UB fabric
```

## Prerequisites

1. Install the ubs_mem package if not already present:
   ```
   rpm -q ubs_mem    # should show ubs_mem-1.0.0-1.oe2203sp13
   ```
2. Ensure the daemon is running:
   ```
   systemctl status ubsmd
   ```
3. Ensure the kernel driver stack is loaded (install `ub-pkg-mem` once
   per node to guarantee):
   ```
   sudo dnf install -y ub-pkg-mem
   lsmod | grep -E 'obmm|ummu'    # should list obmm, ummu, ummu_core, ubmem_vmmu
   ls -l /dev/obmm                 # should exist (major 10)
   ```
4. **Run the sanity probe first** and confirm both nodes see each other:
   ```
   make ubsm_probe
   ./ubsm_probe
   ```
   Expected: `=== PROBE PASSED — scheme6 ready ===`, `host_num = 2`.

## Build

```
make clean && make
```

Produces three binaries:
- `ubsm_probe` — sanity check (see above)
- `server`     — allocator / data owner
- `client`     — reader / benchmark

All three dynamically link `libubsm_sdk.so` with `-Wl,-rpath` so no
`LD_LIBRARY_PATH` juggling is needed.

## Run

### On Node1 (server / data owner)

```
cd benchmark/engram/scheme6_ubmem
./server
```

Defaults: 10000 rows × 341 float dims = 13.6 MB, region
`engram_pool`, object `engram_table_0`. Fills with the same
deterministic pattern as scheme5 (`data[i] = i * 0.001f`) so the
client's verify step can check bit-for-bit.

Command-line: `./server [num_rows] [dim] [region_name] [object_name]`.

The server waits until Ctrl+C, then cleanly unmaps, deallocates, and
finalizes.

### On Node2 (client / reader)

```
cd benchmark/engram/scheme6_ubmem
./client
```

Command-line: `./client [num_rows] [dim] [num_iters] [region_name] [object_name]`.

The client:
1. Initializes the SDK
2. Ensures the same region exists (idempotent if server created it)
3. Maps the same named object read-only
4. Verifies `row[42][0..3]` matches the server's fill pattern
5. Runs three benches:
   - **Bench 1**: single-row read latency (`memcpy` of one row)
   - **Bench 2**: batched reads, batch sizes 32/64/128/256
   - **Bench 3**: full Engram prefetch, 12 tables × {32,64,128,256} tokens
6. Unmaps and finalizes

The benches use `memcpy` from the mapped VA into a local sink buffer
so the reads are real (not CSE'd away) and the timing measures UBMMU
fetch cost.

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| `ubsmem_initialize failed rc=6050` | ubsmd daemon not running | `systemctl start ubsmd` |
| `ubsmem_initialize failed rc=6051` | obmm kernel module not loaded | `sudo modprobe obmm` or install `ub-pkg-mem` |
| `ubsem_create_region failed rc=6020` on only one node | hostname mismatch — the SDK compares hostnames literally | verify both nodes see each other via `ubsm_probe`, and match the hostnames in server.c / client.c |
| Client map fails with `NOT_FOUND` | Server hasn't allocated the object yet, or name differs | start server first, double-check object name arg |
| Verification fails | Pattern fill incomplete or cache stale | check server log for "fill done"; re-run server |
| Build: `unknown type name 'bool'` | Including `ubs_mem.h` directly without `<stdbool.h>` | always include `ubmem_rw.h` (which pulls in `stdbool.h` first) |

## Known SDK quirks

- **Vendor SDK header bug**: `ubs_mem_def.h:102` uses `bool` without
  `<stdbool.h>`. `ubmem_rw.h` works around this by pulling in
  `<stdbool.h>` first. Always include `ubmem_rw.h`, not raw `ubs_mem.h`,
  in your own code.
- **Version skew**: the `ubs_mem` RPM on this cluster is built for
  openEuler 22.03 sp13 but runs on 24.03 sp3. It works, but don't
  assume API/ABI stability across SDK upgrades.
- **Region naming collisions**: regions are a global namespace within
  ubsmd. Pick unique names or handle `UBSM_ERR_ALREADY_EXIST` as a
  no-op (the wrapper does the latter).
