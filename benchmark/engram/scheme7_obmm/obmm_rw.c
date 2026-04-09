/*
 * obmm_rw.c — /dev/obmm ioctl wrapper implementation.
 *
 * See obmm_rw.h for the data flow. Every function here is a thin
 * translation to one or two ioctl() calls on /dev/obmm, with the
 * non-obvious fields filled in according to our current best reading
 * of `/usr/include/ub/obmm.h`. If a field's semantics turn out to be
 * different, it should be fixable here without touching callers.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "obmm_rw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>        /* UINT32_MAX */
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>      /* mmap fallback for addr field semantics probe */

#define OBMM_DEVICE_PATH "/dev/obmm"

#define LOG_ERR(fmt, ...) \
    fprintf(stderr, "[obmm_rw ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[obmm_rw INFO]  " fmt "\n", ##__VA_ARGS__)

/* Single-entry tracking of an mmap() we did during import via the
 * fallback path. The smoke test imports one region; if we later need
 * to juggle multiple, promote this to a small array or hashmap. */
struct obmm_rw_ctx {
    int     fd;
    void   *imported_via_mmap;    /* non-NULL iff import went through mmap() */
    size_t  imported_length;      /* length passed to mmap, needed for munmap */
};

/* ================================================================== */
/*  open / close                                                       */
/* ================================================================== */

obmm_rw_ctx_t* obmm_rw_open(void)
{
    obmm_rw_ctx_t *ctx = (obmm_rw_ctx_t*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->fd = open(OBMM_DEVICE_PATH, O_RDWR | O_CLOEXEC);
    if (ctx->fd < 0) {
        LOG_ERR("open(%s) failed: %s", OBMM_DEVICE_PATH, strerror(errno));
        free(ctx);
        return NULL;
    }
    LOG_INFO("opened %s -> fd=%d", OBMM_DEVICE_PATH, ctx->fd);
    return ctx;
}

void obmm_rw_close(obmm_rw_ctx_t *ctx)
{
    if (!ctx) return;
    /* Defensive: if caller forgot to unimport, free the mmap. */
    if (ctx->imported_via_mmap && ctx->imported_length) {
        munmap(ctx->imported_via_mmap, ctx->imported_length);
    }
    if (ctx->fd >= 0) close(ctx->fd);
    free(ctx);
}

int obmm_rw_fd(const obmm_rw_ctx_t *ctx)
{
    return ctx ? ctx->fd : -1;
}

/* ================================================================== */
/*  Export                                                             */
/* ================================================================== */

int obmm_rw_export(obmm_rw_ctx_t *ctx,
                   void           *va,
                   size_t          length,
                   const uint8_t   seid[16],
                   const uint8_t   deid[16],
                   obmm_rw_handle_t *out_handle)
{
    if (!ctx || !va || length == 0 || !out_handle) return OBMM_RW_ERR_PARAM;

    /* Clear the caller's handle on entry so an early ioctl failure
     * doesn't leave them with stale bytes. */
    memset(out_handle, 0, sizeof(*out_handle));

    struct obmm_cmd_export_pid cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.va         = va;
    cmd.length     = length;
    cmd.pid        = getpid();
    cmd.flags      = OBMM_EXPORT_FLAG_ALLOW_MMAP;
    cmd.pxm_numa   = -1;       /* signed s32; -1 == UINT32_MAX bit pattern == "any" */
    cmd.tokenid    = 0;        /* output field; kernel fills */
    cmd.mem_id     = 0;        /* output field; kernel fills */
    cmd.uba        = 0;        /* output field; kernel fills */
    cmd.priv_len   = 0;
    cmd.vendor_len = 0;
    cmd.priv       = NULL;
    cmd.vendor_info = NULL;

    if (seid) memcpy(cmd.seid, seid, 16);
    if (deid) memcpy(cmd.deid, deid, 16);

    LOG_INFO("ioctl EXPORT_PID: va=%p length=%zu pid=%d flags=0x%lx pxm_numa=%d",
             va, length, (int)cmd.pid, (unsigned long)cmd.flags,
             (int)cmd.pxm_numa);

    if (ioctl(ctx->fd, OBMM_CMD_EXPORT_PID, &cmd) < 0) {
        LOG_ERR("OBMM_CMD_EXPORT_PID failed: %s (errno=%d)",
                strerror(errno), errno);
        return OBMM_RW_ERR_IOCTL;
    }

    out_handle->mem_id    = cmd.mem_id;
    out_handle->length    = length;
    out_handle->uba       = cmd.uba;
    out_handle->tokenid   = cmd.tokenid;
    out_handle->pxm_numa  = cmd.pxm_numa;
    out_handle->base_dist = 0;
    /* NOTE: obmm_cmd_export_pid has NO scna field — only the importer
     * struct does. So the exporter cannot populate handle->scna from
     * the ioctl result. Callers that need a real scna must query it
     * out of band (URMA device attr / sysfs / ubsm_probe). For the
     * loopback smoke test we leave it zero and import uses UINT32_MAX
     * as the "any CNA" sentinel. */
    out_handle->scna      = 0;
    memcpy(out_handle->seid, cmd.seid, 16);
    memcpy(out_handle->deid, cmd.deid, 16);

    LOG_INFO("EXPORT_PID ok: mem_id=0x%lx tokenid=0x%x uba=0x%lx",
             (unsigned long)cmd.mem_id,
             (unsigned)cmd.tokenid,
             (unsigned long)cmd.uba);
    return OBMM_RW_OK;
}

int obmm_rw_unexport(obmm_rw_ctx_t *ctx, const obmm_rw_handle_t *handle)
{
    if (!ctx || !handle) return OBMM_RW_ERR_PARAM;

    struct obmm_cmd_unexport cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.mem_id = handle->mem_id;
    cmd.flags  = 0;

    if (ioctl(ctx->fd, OBMM_CMD_UNEXPORT, &cmd) < 0) {
        if (errno == ENOENT) return OBMM_RW_OK;  /* already gone */
        LOG_ERR("OBMM_CMD_UNEXPORT failed: %s (errno=%d)",
                strerror(errno), errno);
        return OBMM_RW_ERR_IOCTL;
    }
    return OBMM_RW_OK;
}

/* ================================================================== */
/*  Import                                                             */
/* ================================================================== */

/* Is an address plausibly a user-space VA returned by the kernel? This is
 * a fuzzy sanity check to avoid dereferencing an uninitialized or small-
 * integer value and segfaulting before we can log anything useful. aarch64
 * Linux typically places mmap'd regions in the range
 * [0x0000000400000000, 0x00007fffffffffff]. Values below one page or above
 * the canonical user-space ceiling are almost certainly NOT a VA. */
static int addr_looks_like_user_va(uint64_t addr)
{
    if (addr < 0x1000ULL) return 0;                   /* below first page */
    if (addr > 0x0000800000000000ULL) return 0;       /* kernel half */
    return 1;
}

int obmm_rw_import(obmm_rw_ctx_t          *ctx,
                   const obmm_rw_handle_t *handle,
                   const uint8_t           local_seid[16],
                   void                  **out_addr)
{
    if (!ctx || !handle || !out_addr) return OBMM_RW_ERR_PARAM;
    *out_addr = NULL;

    struct obmm_cmd_import cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.flags     = OBMM_IMPORT_FLAG_ALLOW_MMAP;
    cmd.mem_id    = handle->mem_id;
    cmd.addr      = 0;                          /* 0 on input = "let kernel pick" */
    cmd.length    = handle->length;
    cmd.tokenid   = handle->tokenid;
    /* UINT32_MAX is the "any / not specified" sentinel per ubs_mem SDK
     * convention (see scheme6 docstring for allocate_with_provider).
     * Passing 0 here would mean "CNA 0 specifically" which is almost
     * always wrong and is the most likely cause of daemon error 800 in
     * scheme6. */
    cmd.scna      = (handle->scna != 0) ? handle->scna : UINT32_MAX;
    cmd.dcna      = UINT32_MAX;
    cmd.numa_id   = -1;                         /* s32 -1 == UINT32_MAX bits == any */
    cmd.priv_len  = 0;
    cmd.base_dist = handle->base_dist;
    cmd.priv      = NULL;

    /* For loopback smoke test, local_seid == handle->seid. For cross
     * node, local_seid is OUR EID (the accessor), and the handle's
     * seid is the OWNER's EID. We put our side in deid and the owner
     * side in seid (the "source of the memory" is seid). */
    memcpy(cmd.seid, handle->seid, 16);
    if (local_seid) {
        memcpy(cmd.deid, local_seid, 16);
    } else {
        memcpy(cmd.deid, handle->deid, 16);
    }

    /* If we're clearly NOT on the same node as the owner, hint
     * NUMA_REMOTE. For the smoke test the two EIDs are equal so we
     * leave the flag off. */
    if (local_seid && memcmp(local_seid, handle->seid, 16) != 0) {
        cmd.flags |= OBMM_IMPORT_FLAG_NUMA_REMOTE;
    }

    LOG_INFO("ioctl IMPORT: mem_id=0x%lx tokenid=0x%x length=%lu flags=0x%lx "
             "scna=0x%x dcna=0x%x numa_id=%d",
             (unsigned long)cmd.mem_id, (unsigned)cmd.tokenid,
             (unsigned long)cmd.length, (unsigned long)cmd.flags,
             (unsigned)cmd.scna, (unsigned)cmd.dcna, (int)cmd.numa_id);

    if (ioctl(ctx->fd, OBMM_CMD_IMPORT, &cmd) < 0) {
        LOG_ERR("OBMM_CMD_IMPORT failed: %s (errno=%d)",
                strerror(errno), errno);
        return OBMM_RW_ERR_IOCTL;
    }

    LOG_INFO("IMPORT ioctl ok: cmd.addr=0x%lx (length=%lu)",
             (unsigned long)cmd.addr, (unsigned long)handle->length);

    /* The direction / meaning of cmd.addr is NOT documented in obmm.h.
     * We handle three plausible interpretations defensively:
     *
     *   (1) addr is an already-mapped user VA the kernel set up for us
     *       → use it directly
     *   (2) addr is an mmap offset/cookie the caller is expected to feed
     *       into mmap(2) on the /dev/obmm fd
     *   (3) addr was ignored and ALLOW_MMAP means "now call mmap on fd
     *       with offset = mem_id"
     *
     * We pick (1) when addr looks like a valid user VA, otherwise fall
     * through to an mmap(2) probe using cmd.addr, then cmd.mem_id, as
     * possible offset values. Whichever works, we log so we know which
     * interpretation is correct. */
    if (addr_looks_like_user_va(cmd.addr)) {
        *out_addr = (void*)(uintptr_t)cmd.addr;
        ctx->imported_via_mmap = NULL;   /* kernel owns the mapping */
        ctx->imported_length   = 0;
        LOG_INFO("  → interpreting as direct VA: %p", *out_addr);
        return OBMM_RW_OK;
    }

    LOG_INFO("  → addr=0x%lx does not look like a user VA; trying mmap(2) fallback",
             (unsigned long)cmd.addr);

    uint64_t try_offsets[2] = { cmd.addr, cmd.mem_id };
    const char *offset_names[2] = { "cmd.addr", "cmd.mem_id" };
    for (int i = 0; i < 2; i++) {
        void *m = mmap(NULL, handle->length,
                       PROT_READ | PROT_WRITE, MAP_SHARED,
                       ctx->fd, (off_t)try_offsets[i]);
        if (m != MAP_FAILED) {
            *out_addr = m;
            ctx->imported_via_mmap = m;
            ctx->imported_length   = handle->length;
            LOG_INFO("  → mmap(fd, offset=%s=0x%lx) succeeded, va=%p",
                     offset_names[i], (unsigned long)try_offsets[i], m);
            return OBMM_RW_OK;
        }
        LOG_INFO("  → mmap(fd, offset=%s=0x%lx) failed: %s",
                 offset_names[i], (unsigned long)try_offsets[i],
                 strerror(errno));
    }

    LOG_ERR("OBMM_CMD_IMPORT returned addr=0x%lx and mmap fallback failed — "
            "ALLOW_MMAP semantics unclear. Next: strace the call and compare "
            "against libubsm_sdk.so to see how the SDK uses the addr field.",
            (unsigned long)cmd.addr);
    return OBMM_RW_ERR;
}

int obmm_rw_unimport(obmm_rw_ctx_t *ctx, const obmm_rw_handle_t *handle)
{
    if (!ctx || !handle) return OBMM_RW_ERR_PARAM;

    /* If import went through the mmap(2) fallback path, we own the
     * mapping and must munmap before releasing the kernel handle. The
     * direct-VA path leaves imported_via_mmap == NULL. */
    if (ctx->imported_via_mmap && ctx->imported_length) {
        if (munmap(ctx->imported_via_mmap, ctx->imported_length) < 0) {
            LOG_ERR("munmap(%p, %zu) failed: %s (continuing to unimport)",
                    ctx->imported_via_mmap, ctx->imported_length,
                    strerror(errno));
        }
        ctx->imported_via_mmap = NULL;
        ctx->imported_length   = 0;
    }

    struct obmm_cmd_unimport cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.mem_id = handle->mem_id;
    cmd.flags  = 0;

    if (ioctl(ctx->fd, OBMM_CMD_UNIMPORT, &cmd) < 0) {
        if (errno == ENOENT) return OBMM_RW_OK;
        LOG_ERR("OBMM_CMD_UNIMPORT failed: %s (errno=%d)",
                strerror(errno), errno);
        return OBMM_RW_ERR_IOCTL;
    }
    return OBMM_RW_OK;
}

/* ================================================================== */
/*  Utility                                                            */
/* ================================================================== */

const char* obmm_rw_strerror(int rc)
{
    switch (rc) {
    case OBMM_RW_OK:         return "OK";
    case OBMM_RW_ERR:        return "OBMM_RW_ERR (generic)";
    case OBMM_RW_ERR_PARAM:  return "OBMM_RW_ERR_PARAM (bad arg)";
    case OBMM_RW_ERR_IOCTL:  return "OBMM_RW_ERR_IOCTL (see earlier errno)";
    case OBMM_RW_ERR_NODEV:  return "OBMM_RW_ERR_NODEV (/dev/obmm not accessible)";
    default:                 return "UNKNOWN";
    }
}

void obmm_rw_print_handle(const char *tag, const obmm_rw_handle_t *h)
{
    if (!h) {
        printf("  %s: (null)\n", tag ? tag : "handle");
        return;
    }
    printf("  %s: mem_id=0x%lx tokenid=0x%x length=%lu uba=0x%lx "
           "scna=%u pxm_numa=%d base_dist=%u\n",
           tag ? tag : "handle",
           (unsigned long)h->mem_id, (unsigned)h->tokenid,
           (unsigned long)h->length, (unsigned long)h->uba,
           (unsigned)h->scna, (int)h->pxm_numa, (unsigned)h->base_dist);
    printf("    seid = ");
    for (int i = 0; i < 16; i++) printf("%02x", h->seid[i]);
    printf("\n    deid = ");
    for (int i = 0; i < 16; i++) printf("%02x", h->deid[i]);
    printf("\n");
}
