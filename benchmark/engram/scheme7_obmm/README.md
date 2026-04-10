# Scheme 7 — direct obmm UAPI for cross-node load/store

## What this is

A load/store path for cross-node Engram access that **bypasses every
Huawei userspace component** — no `libubsm_sdk.so`, no open-source
`libubs_mem.so`, no `libubse.so`, no `ubsmd` daemon. We speak directly
to the GPL kernel UAPI at `/dev/obmm` via ioctl, and to the per-mem_id
data device `/dev/obmm_shmdev<id>` via `mmap(2)`.

This was written after reverse-engineering the open-source
[ubs-mem](https://gitcode.com/openeuler/ubs-mem) project, specifically:

- `src/app_lib/mxm_shm_lib/RackMemShm.cpp:131-177` — the mmap loop
- `src/app_lib/common/rack_mem_lib_common.h:47-65` — `ObmmOpenInternal`
- `src/app_lib/common/rack_mem_libobmm.h:26` — `/dev/obmm_shmdev%lu` path

These three files reveal the actual load/store flow that `ubs-mem` uses
at the SDK layer, and show that after an EXPORT ioctl we can drive the
data plane entirely with plain `open(2) + mmap(2)`.

## Architecture

```
user process                    /dev/obmm             obmm driver      UBMMU/UB fabric
────────────                    ────────────          ──────────       ───────────────
1. open /dev/obmm        ──►    ctl fd
2. ioctl EXPORT          ──►    allocate length bytes
                                 on NUMA 0 via buddy
                                 highmem, register
                                 mem_id + tokenid
                         ◄──    {mem_id, tokenid, uba}
                                 kernel auto-creates
                                 /dev/obmm_shmdev<id>

3. open shmdev<id>       ──►    data fd
4. mmap(data_fd,                establish user VA
    offset=0             ──►    backed by the page
    or HUGETLB_PMD)              pool via UBMMU

5. *(volatile T*)va      ──────────────────────────►  direct access
                                                       (single-node: local DRAM)
                                                       (cross-node:  UB fabric)

6. munmap / close
7. ioctl UNEXPORT        ──►    releases pages
```

The magic is step 4: the `offset` parameter is **not a byte offset**,
it's a bitfield of kernel flags. Valid values:

- `0` — normal 4 KB pages
- `OBMM_MMAP_FLAG_HUGETLB_PMD` (`1UL << 63`) — 2 MB huge pages

That one insight (offset = flag, not bytes) is what unstuck scheme 7
after two days of probing mmap offsets on the wrong device.

## Files

- `smoke_test.c` — fully self-contained single-node loopback test.
  No custom headers, no helper libraries. Build and run it, and
  either we see `=== smoke test PASSED ===` and scheme 7 is unblocked,
  or we get a precise kernel errno telling us exactly which step
  failed. **PASSED on Node1 2026-04-10.**
- `server.c` — cross-node data server. EXPORT 2 GB, fill with
  scheme5-compatible Engram pattern (`data[i] = i*0.001f`), TCP
  listen, send handle struct on each accept. Mirrors
  `scheme5_urma_rw/server.c`.
- `client.c` — cross-node reader. TCP connect, recv handle, IMPORT,
  open/mmap, verify row 42, run the same 3 bench loops as
  `scheme5_urma_rw/client.c`: single-row read, batch read
  (32/64/128/256), full Engram prefetch (12 tables × N tokens).
- `Makefile` — three C files, three binaries, no helper lib.

Deleted files (based on the prior wrong mental model):
- `smoke_multi.c`, `smoke_export.c` — probed mmap offsets on
  `/dev/obmm` (control device), which is the wrong target.
- `obmm_rw.c`, `obmm_rw.h` — wrapped `OBMM_CMD_EXPORT_PID` which
  always returns ENOTSUP on this kernel (export-existing-VA path
  is reserved for in-kernel callers).

## How to build and run

### Single-node smoke test (first step)

```bash
cd benchmark/engram/scheme7_obmm
make clean && make
sudo ./smoke_test            # 4 MB default
sudo ./smoke_test 16         # 16 MB
```

Expected output on success:

```
=== scheme7 obmm loopback smoke test ===
  buffer = 4.00 MB (4MB-aligned, 4194304 bytes)
  strategy: EXPORT → /dev/obmm_shmdev<id> → mmap → load/store

[1] /dev/obmm -> fd_ctl=3
[2] EXPORT ok: mem_id=1 (0x1)  tokenid=0x4a9  uba=0xffffffc00000
[3] opening /dev/obmm_shmdev1 ...
    exists: mode=0600, rdev=...
    opened -> fd_data=4
[4.0] mmap offset=0 (normal pages) ... OK, va=0x...
    using variant #0
[5] load/store test ...
    [PASS] 1 KB pattern roundtrip OK
[6] micro-bench: 100000 loads in X.XX us = Y.YY ns/load (sink=..., dce-guard)
[7] UNEXPORT ok

=== smoke test PASSED ===
```

### What can go wrong — reading the diagnostic output

The test prints the result of every stage, so failures are localized:

| Symptom | Meaning | Next step |
|---|---|---|
| `open(/dev/obmm): Permission denied` | need root | run with `sudo` |
| `open(/dev/obmm): No such file or directory` | kernel module not loaded | `lsmod \| grep obmm`; `modprobe obmm` |
| `EXPORT: ENOMEM` | pool exhausted | check `/sys/module/obmm/parameters/mempool_size` |
| `EXPORT: EINVAL` | struct mismatch (rare) | compare installed `<ub/obmm.h>` vs the one in this source tree |
| `stat /dev/obmm_shmdev<id>: ENOENT` | device not auto-created despite EXPORT success | check dmesg; kernel may be missing the udev rule or running a stripped build |
| `open shmdev: EACCES` | wrong ownership / mode | `ls -l /dev/obmm_shmdev*`; ensure root or add udev rule |
| `mmap offset=0: EINVAL`, then HUGETLB_PMD also EINVAL | neither page size accepted | dmesg will tell us why; could be NUMA mismatch or cacheable bit |
| `[FAIL] N pattern mismatches` | mapping worked but wrote to different pages | almost impossible in loopback; means shmdev mmap backed something stale |

All of these are debuggable because the kernel source is open
(openEuler, `drivers/ub/mem/obmm/` for the ub-pkg-mem build).

### Cross-node test (after smoke_test passes)

```bash
# Node1 (data server)
cd benchmark/engram/scheme7_obmm
make clean && make
sudo ./server                              # default: port 13857, 2 GB

# Node2 (compute client)
cd benchmark/engram/scheme7_obmm
make clean && make
sudo ./client 141.61.84.245 13857 10000 341 200
```

Client output format matches scheme5 exactly, so numbers can be
diff'd directly. Key columns:

- **[Verify]**: `row[42][0..3]` should match server's pattern.
- **[Bench 1]**: single-row latency. Scheme5 = 2.65 μs (post+poll).
  Scheme7 target: 200-500 ns (pure UB fabric load).
- **[Bench 2]**: batch read. Scheme5 amortizes to 0.10 μs/row @256.
  Scheme7 is per-row (no batching API); CPU prefetcher may help.
- **[Bench 3]**: full Engram prefetch. Scheme5 hits 0.14 ms for 128
  tokens (7× under the CXL paper target). **This is the number
  that decides scheme7 vs scheme5 for the final integration.**

### Cross-node failure modes

| Symptom | Meaning | Fix |
|---|---|---|
| `OBMM_CMD_IMPORT: EINVAL` | identity fields (seid/scna zero) rejected | check dmesg; borrow EID from URMA (scheme5 already does) |
| `OBMM_CMD_IMPORT: ENOMEM` | shmdev minor numbers or UBMMU entries exhausted | restart obmm module, reduce size |
| `OBMM_CMD_IMPORT: EACCES` | tokenid mismatch | wire struct must be binary-identical both sides (same-endian aarch64 = OK) |
| `[Verify] row[42]` shows garbage | mapping valid but wrong pages | UBMMU misprogramming; dmesg for translation errors |
| Bench 1 > 1 μs | fabric is the limit, not OBMM overhead | scheme 7 has no edge over scheme 5 — skip integration |
| Bench 1 < 500 ns | load/store is faster than urma_read | confirm Bench 3 for ratio that matters |

### SGLang integration (after cross-node bench)

If scheme 7 Bench 3 beats scheme 5's 0.14 ms by >2×, integrate via
Python ctypes wrapping in `engram_prefetcher.py`. Otherwise keep
scheme 5 (less plumbing).

## Design notes

### Why no `obmm_rw.c` abstraction layer

Because the raw call sequence is already so short (5 syscalls:
`open`, `ioctl`, `open`, `mmap`, `munmap`) that wrapping it in a
library adds noise without saving lines. For cross-node, we'll
add a 30-line helper for the TCP handshake and let the server /
client use the same raw syscalls directly.

### Why we still require 4 MB alignment

Kernel param `/sys/module/obmm/parameters/mem_allocator_granu` is
`2M`, but ubs-mem's SDK enforces 4 MB and empirically 4 MB works
reliably. Rather than hit an unknown edge case, we keep 4 MB.

### Why no linking against `libobmm.so.1`

Even though ubs-mem's SDK dlopens this for one function
(`obmm_set_ownership`), we don't need it in scheme 7. Our use case
is "open wide, read/write freely, then unexport" — no dynamic
permission switching. Zero `dlopen`, zero `ldconfig` headaches.

Verify with `ldd ./smoke_test`:

```
$ ldd smoke_test
    linux-vdso.so.1 (0x...)
    libc.so.6 => /lib64/libc.so.6 (0x...)
    /lib64/ld-linux-aarch64.so.1 (0x...)
```

Nothing from `libubsm_*`, `libubse*`, or `libobmm*` should appear.

## Relationship to other schemes

| Scheme | Userland deps | Kernel path | Status |
|---|---|---|---|
| 5 (urma_read) | `liburma.so` | URMA verbs | **working**, 0.14 ms/128 tokens |
| 6 (libubsm_sdk) | `libubsm_sdk.so` + `libubse.so` + `ubsmd` | obmm via UBSE | abandoned (4 MB alignment was the actual blocker, not the SDK) |
| **7 (obmm direct)** | **none** | obmm direct ioctl + shmdev mmap | **smoke test pending** |
| Stretch: URMA_SEG_MAPPED | `liburma.so` + kernel patch | hns3 driver patch | deferred (2-3 weeks work) |
