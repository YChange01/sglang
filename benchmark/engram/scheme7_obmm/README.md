# Scheme 7 — direct obmm UAPI for cross-node load/store

## What this is

An alternative load/store path for cross-node Engram access that
**bypasses the closed-source `libubsm_sdk.so` entirely** and talks
directly to the GPL kernel UAPI at `/dev/obmm`.

This was written after scheme 6 got stuck on a `daemon error 800`
inside `libubsm_sdk.so` that we could not debug because the SDK is
a binary blob. The kernel UAPI, by contrast, is documented at
`/usr/include/ub/obmm.h` (186 lines, `SPDX-License-Identifier:
GPL-2.0+`) and we can read the kernel source if needed.

## Architecture

```
app                      obmm_rw.c          /dev/obmm       ummu/UBMMU    UB fabric
───                      ──────────          ─────────       ──────────    ─────────
export_pid(va, len) ──► ioctl EXPORT_PID ──► kernel pins pages,
                                             returns mem_id+tokenid+uba
                                                    │
                                 TCP handle (mem_id, tokenid, seid, deid, length)
                                                    ▼
import(handle)      ──► ioctl IMPORT       ──► kernel allocates local VA,
                                                programs UBMMU to forward
                                                faults over UB fabric,
                                                returns local addr
*(T*)addr           ──────────────────────► UBMMU page walks ──► UB fabric ──► remote
                                                                                 DRAM
```

**Key difference from scheme 5 (URMA urma_read):** no `post_jetty_send_wr`,
no completion queue, no polling. A load of `*addr` faults through UBMMU
which resolves it across the fabric. Expected to drop single-read
latency from scheme 5's ~2.65 μs (post+poll overhead) to something
closer to raw fabric round-trip (~0.2 μs per the CXL paper).

**Key difference from scheme 6 (libubsm_sdk.so):** we own every byte
of the ioctl payload. No closed SDK between us and the kernel. Every
failure mode has a readable source of truth.

## Files

- `obmm_rw.h` / `obmm_rw.c` — thin wrapper over `/dev/obmm` ioctls.
  Opaque `obmm_rw_ctx_t` holds the fd; `obmm_rw_handle_t` is the
  80-byte wire-format handle that crosses nodes via TCP.
- `smoke_test.c` — **single-node loopback sanity check**. Export a
  local buffer, import it back in the same process, verify the
  pattern reads correctly through the imported VA. Also measures a
  quick loopback load latency as a sanity bench.
- `server.c` / `client.c` — **(to be added)** cross-node
  exporter + importer with TCP handle exchange. Mirror structure of
  `scheme5_urma_rw/`.
- `Makefile` — no link to `libubsm_sdk.so`, only standard libc +
  kernel headers.

## How to build and run

### Single-node smoke test (first step)

```bash
cd benchmark/engram/scheme7_obmm
make clean && make smoke_test
sudo ./smoke_test            # /dev/obmm typically requires root
# or:  sudo ./smoke_test 16   # 16 MB buffer
```

Expected output on success:

```
=== scheme7 obmm smoke test ===
  buffer = 4.00 MB (4MB-aligned)

[obmm_rw INFO] opened /dev/obmm -> fd=3
  allocated va = 0x...
  wrote 1 KB pattern via original VA
[obmm_rw INFO] ioctl EXPORT_PID: va=0x... length=... pid=... flags=0x1
[obmm_rw INFO] EXPORT_PID ok: mem_id=0x... tokenid=0x... uba=0x...
  exported handle: mem_id=... tokenid=... length=... uba=... ...
[obmm_rw INFO] ioctl IMPORT: mem_id=0x... tokenid=0x... length=... flags=0x1
[obmm_rw INFO] IMPORT ok: local addr=0x... length=...
  imported va = 0x...
  [PASS] 1 KB pattern matches via imported VA
  [bench] 1000 loopback loads: ... ns total, ... ns/load (acc=..., prevents DCE)

  === smoke test PASSED ===
```

### What can go wrong at this stage

Likely failure points on the first run and how to read them:

| Symptom | Meaning | Next step |
|---|---|---|
| `open(/dev/obmm): Permission denied` | need root or an ACL group | run with sudo |
| `open(/dev/obmm): No such file or directory` | `obmm` kernel module not loaded | `lsmod \| grep obmm`, then `modprobe obmm` via ub-pkg-mem init |
| `EXPORT_PID failed: Invalid argument (EINVAL)` | one of our struct fields doesn't match what the kernel expects | try non-zero `pxm_numa`, or try `OBMM_CMD_EXPORT` (size[]-based) instead of `EXPORT_PID` |
| `IMPORT failed: EPERM` | tokenid / EID check — try matching seid to something real | borrow our local EID from URMA (see scheme5) and feed both ends |
| `IMPORT returned addr=0` | kernel accepted the call but didn't map — flag missing | make sure `OBMM_EXPORT_FLAG_ALLOW_MMAP` was set at export time |
| `[FAIL] N mismatches` | ioctl succeeded but read data is wrong — map went to different pages | we're on the wrong path; possibly the addr field is input (hint) not output |

All of these are debuggable because the kernel source is open
(openEuler kernel, likely at `drivers/ub/mem/`).

### Cross-node test (after smoke test passes)

```bash
# Node1
./server

# Node2
./client <node1-ip> <port>
```

(commands will be added with server.c / client.c)

## Design notes / non-obvious choices

### Why `OBMM_CMD_EXPORT_PID` and not `OBMM_CMD_EXPORT`

`EXPORT_PID` takes `(va, length, pid, flags)` — simple and matches
our "I have a buffer, make it cross-node visible" use case.

`EXPORT` takes `(size[OBMM_MAX_LOCAL_NUMA_NODES], length, flags, uba, tokenid)` —
it seems to describe a per-NUMA allocation request rather than
exporting an existing VA. More verbose, meant for cases where you
want the kernel to choose NUMA placement.

For Engram, we already have a buffer (filled with embedding data)
and we want that buffer made cross-node. `EXPORT_PID` fits.

### Why we preserve the 4 MB alignment rule from scheme 6

The `"size does not align with 4194304"` error we hit in scheme 6
came from inside `libubsm_sdk.so`, but the alignment constraint is
almost certainly enforced by the underlying obmm kernel module
(it controls physical page registration). We keep the rule here
to avoid rediscovering it the hard way.

### Why seid/deid/scna/dcna start as zero

For the loopback smoke test we just need the simplest possible
call. If the kernel rejects zero EIDs with EINVAL we'll iterate:

- borrow the local EID from URMA (one extra open+close on `/dev/uburma/udma2`)
- or parse it from sysfs if exposed
- or use PID-based addressing exclusively when both sides are the same process

The important thing is that we have a clean `obmm_rw_ctx` that we
can extend without touching callers.

### Why we don't link libubsm_sdk.so

That's the whole point of scheme 7. The only headers we include are:

- standard libc (stdio, stdlib, unistd, sys/ioctl, sys/mman)
- `<ub/obmm.h>` — GPL kernel UAPI

If any of the other `/usr/local/ubs_mem/` files (the SDK headers,
libubsm_sdk.so binary, ubsmd daemon) accidentally get pulled in,
we've failed to isolate. Check with `ldd ./smoke_test` — should
not show any libubsm_* entries.
