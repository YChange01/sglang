/*
 * obmm_rw.h — direct UAPI wrapper over /dev/obmm for cross-node load/store.
 *
 * Scheme 7 bypasses the closed libubsm_sdk.so entirely and calls the
 * GPL kernel UAPI `/usr/include/ub/obmm.h` via ioctl on `/dev/obmm`.
 *
 * Data flow (cross-node):
 *
 *     Node A (owner)                      Node B (accessor)
 *     ──────────                          ──────────────
 *     obmm_rw_open()                      obmm_rw_open()
 *     obmm_rw_export(va, len)             (wait for TCP handshake)
 *        → {mem_id, tokenid, uba}
 *                  │
 *                  │  TCP: exchange handle (mem_id, tokenid, seid, deid, len)
 *                  ▼
 *                                         obmm_rw_import(handle)
 *                                            → addr (local VA)
 *                                         memcpy(dst, (void*)addr+off, n)
 *                                         obmm_rw_unimport()
 *     obmm_rw_unexport()                  obmm_rw_close()
 *     obmm_rw_close()
 *
 * For a single-node loopback smoke test, steps 2-5 all happen in one
 * process on one /dev/obmm fd — no TCP, seid=deid=0.
 *
 * This file is header-only for the data structures and declarations;
 * the actual SDK interaction lives in obmm_rw.c.
 */

#ifndef OBMM_RW_H
#define OBMM_RW_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/types.h>

/* The kernel UAPI header. 186 lines, GPL-2.0+, Huawei copyright. */
#include <ub/obmm.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================== */
/*  Return codes                                                       */
/* ================================================================== */

#define OBMM_RW_OK             0
#define OBMM_RW_ERR           -1
#define OBMM_RW_ERR_PARAM     -2
#define OBMM_RW_ERR_IOCTL     -3
#define OBMM_RW_ERR_NODEV     -4

/* ================================================================== */
/*  Handle that crosses nodes                                          */
/* ================================================================== */

/* This is the "importer ticket" the owner gives to a peer so the peer
 * can map the remote memory. It is self-contained: send these bytes
 * over TCP, pass to obmm_rw_import on the peer.
 *
 * 80 bytes, packed for wire stability. */
typedef struct {
    uint64_t mem_id;       /* output of export, input of import */
    uint64_t length;       /* byte length of the exported region */
    uint64_t uba;          /* UB fabric address, for diagnostics only */
    uint32_t tokenid;      /* access credential, export output → import input */
    uint32_t scna;         /* source CNA (owner's node address) */
    uint8_t  seid[16];     /* source endpoint ID (owner's EID, 128 bit) */
    uint8_t  deid[16];     /* destination endpoint ID (peer's EID, 128 bit) */
    int32_t  pxm_numa;     /* proximity NUMA hint */
    uint8_t  base_dist;    /* NUMA distance hint */
    uint8_t  reserved[3];  /* align */
} __attribute__((packed)) obmm_rw_handle_t;

/* ================================================================== */
/*  Context                                                            */
/* ================================================================== */

typedef struct obmm_rw_ctx obmm_rw_ctx_t;

/* Open /dev/obmm and allocate a ctx. Returns NULL on failure. */
obmm_rw_ctx_t* obmm_rw_open(void);

/* Close and free the ctx. Safe to call with NULL. */
void obmm_rw_close(obmm_rw_ctx_t *ctx);

/* Returns the underlying /dev/obmm fd (useful for mmap experiments). */
int obmm_rw_fd(const obmm_rw_ctx_t *ctx);

/* ================================================================== */
/*  Export (server / owner side)                                       */
/* ================================================================== */

/* Export an existing local buffer to be visible across the UB fabric.
 *
 * `va` must be a valid, populated virtual address owned by this process
 * (from malloc, mmap anonymous, etc). `length` should be a page multiple;
 * the kernel may round up. `seid` / `deid` may be zero for loopback
 * smoke test; otherwise fill seid with our local EID (can be borrowed
 * from URMA briefly) and deid with the expected peer EID or 0 for "any".
 *
 * On success, `out_handle` is populated with everything the peer needs
 * to import this region. */
int obmm_rw_export(obmm_rw_ctx_t *ctx,
                   void           *va,
                   size_t          length,
                   const uint8_t   seid[16],
                   const uint8_t   deid[16],
                   obmm_rw_handle_t *out_handle);

/* Release a prior export. Idempotent. */
int obmm_rw_unexport(obmm_rw_ctx_t *ctx, const obmm_rw_handle_t *handle);

/* ================================================================== */
/*  Import (client / accessor side)                                    */
/* ================================================================== */

/* Import a remote region using the handle received from the owner.
 * Returns a local virtual address in `*out_addr` that is backed by
 * the remote memory — *(T*)(*out_addr + offset) is a direct load
 * routed by UBMMU over UB fabric.
 *
 * `local_seid` is this node's own EID (we're the "source" of the read
 *  request flowing to the remote "destination"). Can be zero for
 * loopback. */
int obmm_rw_import(obmm_rw_ctx_t          *ctx,
                   const obmm_rw_handle_t *handle,
                   const uint8_t           local_seid[16],
                   void                  **out_addr);

/* Release a prior import. Must be called before close to avoid leaks. */
int obmm_rw_unimport(obmm_rw_ctx_t *ctx, const obmm_rw_handle_t *handle);

/* ================================================================== */
/*  Utility                                                            */
/* ================================================================== */

/* Human-readable string for an errno from an obmm ioctl. */
const char* obmm_rw_strerror(int rc);

/* Pretty-print a handle for debug logging. */
void obmm_rw_print_handle(const char *tag, const obmm_rw_handle_t *h);

#ifdef __cplusplus
}
#endif

#endif /* OBMM_RW_H */
