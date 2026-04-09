/*
 * ubmem_rw.h — thin wrapper around the UB Shared Memory (ubs_mem) SDK.
 *
 * Scheme 6 uses the ubs_mem SDK (/usr/local/ubs_mem) which lets a process
 * on Node A allocate a named shared-memory object and a process on Node B
 * mmap it into its own virtual address space. After import the remote
 * memory is accessible via plain load/store instructions routed by UBMMU
 * over the UB fabric — no explicit read/write calls, no post/poll.
 *
 * This header hides a few awkward SDK details:
 *   - ubs_mem_def.h uses `bool` without including <stdbool.h> — we pull
 *     <stdbool.h> in first to work around that SDK header bug.
 *   - Every SDK call returns a UBSM_ERR_* integer; we convert the common
 *     ones to strings for readable logging.
 *   - ubsmem_initialize needs a zeroed options struct (there are no
 *     configurable fields yet) — wrapper takes care of that.
 *
 * Usage pattern (same on both server and client):
 *     urw = ubmem_rw_init();
 *     ubmem_rw_ensure_region(urw, "engram_pool", hosts, n_hosts);
 *     // Server side:
 *     ubmem_rw_allocate(urw, "engram_pool", "engram_table_0", size, flags);
 *     ubmem_rw_map("engram_table_0", size, PROT_READ|PROT_WRITE, &ptr);
 *     // Client side:
 *     ubmem_rw_map("engram_table_0", size, PROT_READ, &ptr);
 *     // common:
 *     ubmem_rw_unmap(ptr, size);
 *     ubmem_rw_destroy(urw);
 */

#ifndef UBMEM_RW_H
#define UBMEM_RW_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>    /* must precede ubs_mem.h — SDK header forgets it */
#include <sys/types.h>  /* off_t, mode_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes returned by ubmem_rw_* */
#define UBMEM_RW_OK           0
#define UBMEM_RW_ERR         -1
#define UBMEM_RW_ERR_PARAM   -2
#define UBMEM_RW_ERR_NOTFOUND -3

/* Allocation alignment required by the ubs_mem SDK.
 *
 * Empirical finding: ubsmem_shmem_allocate() rejects any size that is
 * not a multiple of 4 MB with UBSM_ERR_PARAM_INVALID and an internal
 * log line like:
 *     "The size N does not align with 4194304"
 *
 * This is enforced regardless of the HUGEPAGE flag — it appears to be
 * the minimum granularity the OBMM kernel driver works in. Both
 * allocation and mapping must use the same aligned size. Callers
 * should always pass a size that is UBMEM_RW_ALIGN_UP'd.
 */
#define UBMEM_RW_ALIGN_BYTES  (4UL * 1024UL * 1024UL)
#define UBMEM_RW_ALIGN_UP(x)  \
    (((size_t)(x) + UBMEM_RW_ALIGN_BYTES - 1UL) & ~(UBMEM_RW_ALIGN_BYTES - 1UL))

/* Opaque context */
typedef struct ubmem_rw_ctx ubmem_rw_ctx_t;

/* Describes one host that participates in a region. */
typedef struct {
    const char *hostname;   /* e.g. "node1" — must match what ubsmd reports */
    bool        affinity;   /* true if backing memory should prefer this host */
} ubmem_rw_host_t;

/* ================================================================== */
/*  Init / shutdown                                                    */
/* ================================================================== */

/* Initialize the ubs_mem SDK. Also sets a quieter log level so the
 * stdout/stderr isn't drowned in HCOM/SDK trace lines.
 * Returns NULL on failure. */
ubmem_rw_ctx_t* ubmem_rw_init(void);

/* Finalize the SDK and release the ctx. Safe to call with NULL. */
void ubmem_rw_destroy(ubmem_rw_ctx_t *ctx);

/* ================================================================== */
/*  Region management (name-scoped memory pool spanning hosts)         */
/* ================================================================== */

/* Ensure a region with the given name exists that spans `hosts[0..n-1]`.
 * If it already exists (UBSM_ERR_ALREADY_EXIST), treated as success.
 *
 * This should be called on BOTH server and client with the same (name,
 * hosts) arguments. */
int ubmem_rw_ensure_region(ubmem_rw_ctx_t *ctx,
                           const char *region_name,
                           const ubmem_rw_host_t *hosts,
                           int n_hosts);

/* Destroy a region. Ignores UBSM_ERR_NOT_FOUND so it's idempotent. */
int ubmem_rw_destroy_region(ubmem_rw_ctx_t *ctx, const char *region_name);

/* ================================================================== */
/*  Shared memory object (named buffer)                                */
/* ================================================================== */

/* Flag shortcuts for allocate() */
#define UBMEM_RW_FLAG_CACHE       0x0UL  /* cacheable (default, best for read-heavy) */
#define UBMEM_RW_FLAG_NONCACHE    0x2UL  /* O_SYNC, bypass cache */
#define UBMEM_RW_FLAG_HUGEPAGE    0x20UL /* map at 2MB PMD granularity */

/* Allocate (or ensure exists) a named shared-memory object within a
 * region. If an object with the same name already exists, this is
 * treated as success (idempotent), but note that size/flags of the
 * pre-existing object are NOT verified here.
 *
 * `mode` is the unix-style permission bits (e.g. 0644).
 * `flags` is a bitwise-OR of UBMEM_RW_FLAG_* above. */
int ubmem_rw_allocate(ubmem_rw_ctx_t *ctx,
                      const char *region_name,
                      const char *object_name,
                      size_t      size,
                      unsigned    mode,
                      uint64_t    flags);

/* Deallocate a previously-allocated object. Ignores NOT_FOUND. */
int ubmem_rw_deallocate(ubmem_rw_ctx_t *ctx, const char *object_name);

/* ================================================================== */
/*  Mapping                                                            */
/* ================================================================== */

/* Map an object into the local virtual address space. `prot` and `flags`
 * follow mmap(2) semantics (e.g. PROT_READ|PROT_WRITE, MAP_SHARED).
 *
 * After success, `*out_ptr` can be used directly with load/store /
 * memcpy — UBMMU routes reads to the backing host over UB fabric. */
int ubmem_rw_map(ubmem_rw_ctx_t *ctx,
                 const char *object_name,
                 size_t length,
                 int prot,
                 int flags,
                 off_t offset,
                 void **out_ptr);

/* Unmap. `length` must match what was passed to map(). */
int ubmem_rw_unmap(ubmem_rw_ctx_t *ctx, void *ptr, size_t length);

/* ================================================================== */
/*  Utility                                                            */
/* ================================================================== */

/* Return the local supernode ID (nid). */
int ubmem_rw_local_nid(ubmem_rw_ctx_t *ctx, uint32_t *out_nid);

/* Human-readable name for a UBSM_ERR_* or UBMEM_RW_ERR_* code. */
const char* ubmem_rw_strerror(int code);

#ifdef __cplusplus
}
#endif

#endif /* UBMEM_RW_H */
