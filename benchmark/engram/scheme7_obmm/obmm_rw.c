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
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#define OBMM_DEVICE_PATH "/dev/obmm"

#define LOG_ERR(fmt, ...) \
    fprintf(stderr, "[obmm_rw ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[obmm_rw INFO]  " fmt "\n", ##__VA_ARGS__)

struct obmm_rw_ctx {
    int fd;
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

    struct obmm_cmd_export_pid cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.va         = va;
    cmd.length     = length;
    cmd.pid        = getpid();
    cmd.flags      = OBMM_EXPORT_FLAG_ALLOW_MMAP;
    cmd.pxm_numa   = -1;       /* any local NUMA — may need 0 or specific */
    cmd.tokenid    = 0;        /* output field; kernel fills */
    cmd.mem_id     = 0;        /* output field; kernel fills */
    cmd.uba        = 0;        /* output field; kernel fills */
    cmd.priv_len   = 0;
    cmd.vendor_len = 0;
    cmd.priv       = NULL;
    cmd.vendor_info = NULL;

    if (seid) memcpy(cmd.seid, seid, 16);
    if (deid) memcpy(cmd.deid, deid, 16);

    LOG_INFO("ioctl EXPORT_PID: va=%p length=%zu pid=%d flags=0x%lx",
             va, length, (int)cmd.pid, (unsigned long)cmd.flags);

    if (ioctl(ctx->fd, OBMM_CMD_EXPORT_PID, &cmd) < 0) {
        LOG_ERR("OBMM_CMD_EXPORT_PID failed: %s (errno=%d)",
                strerror(errno), errno);
        return OBMM_RW_ERR_IOCTL;
    }

    memset(out_handle, 0, sizeof(*out_handle));
    out_handle->mem_id    = cmd.mem_id;
    out_handle->length    = length;
    out_handle->uba       = cmd.uba;
    out_handle->tokenid   = cmd.tokenid;
    out_handle->pxm_numa  = cmd.pxm_numa;
    out_handle->base_dist = 0;
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

int obmm_rw_import(obmm_rw_ctx_t          *ctx,
                   const obmm_rw_handle_t *handle,
                   const uint8_t           local_seid[16],
                   void                  **out_addr)
{
    if (!ctx || !handle || !out_addr) return OBMM_RW_ERR_PARAM;

    struct obmm_cmd_import cmd;
    memset(&cmd, 0, sizeof(cmd));

    cmd.flags     = OBMM_IMPORT_FLAG_ALLOW_MMAP;
    cmd.mem_id    = handle->mem_id;
    cmd.addr      = 0;                          /* output: local VA */
    cmd.length    = handle->length;
    cmd.tokenid   = handle->tokenid;
    cmd.scna      = handle->scna;
    cmd.dcna      = 0;                          /* destination CNA; unknown, try 0 */
    cmd.numa_id   = -1;                         /* any local NUMA */
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

    LOG_INFO("ioctl IMPORT: mem_id=0x%lx tokenid=0x%x length=%lu flags=0x%lx",
             (unsigned long)cmd.mem_id, (unsigned)cmd.tokenid,
             (unsigned long)cmd.length, (unsigned long)cmd.flags);

    if (ioctl(ctx->fd, OBMM_CMD_IMPORT, &cmd) < 0) {
        LOG_ERR("OBMM_CMD_IMPORT failed: %s (errno=%d)",
                strerror(errno), errno);
        return OBMM_RW_ERR_IOCTL;
    }

    if (cmd.addr == 0) {
        LOG_ERR("OBMM_CMD_IMPORT returned addr=0 — mapping did not happen");
        return OBMM_RW_ERR;
    }

    *out_addr = (void*)(uintptr_t)cmd.addr;
    LOG_INFO("IMPORT ok: local addr=%p length=%lu",
             *out_addr, (unsigned long)handle->length);
    return OBMM_RW_OK;
}

int obmm_rw_unimport(obmm_rw_ctx_t *ctx, const obmm_rw_handle_t *handle)
{
    if (!ctx || !handle) return OBMM_RW_ERR_PARAM;

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
