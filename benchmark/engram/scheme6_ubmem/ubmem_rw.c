/*
 * ubmem_rw.c — implementation of the thin ubs_mem SDK wrapper.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ubmem_rw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>        /* gethostname */
#include <sys/types.h>
#include <sys/stat.h>

/* stdbool.h already included via ubmem_rw.h before the SDK headers */
#include <ubs_mem.h>
#include <ubs_mem_def.h>

#define LOG_ERR(fmt, ...) \
    fprintf(stderr, "[ubmem_rw ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[ubmem_rw INFO] " fmt "\n", ##__VA_ARGS__)

/* ================================================================== */
/*  Context                                                            */
/* ================================================================== */

struct ubmem_rw_ctx {
    bool initialized;
};

/* ================================================================== */
/*  Error strings                                                      */
/* ================================================================== */

const char* ubmem_rw_strerror(int code)
{
    switch (code) {
    /* ubmem_rw wrapper codes */
    case UBMEM_RW_OK:           return "OK";
    case UBMEM_RW_ERR:          return "UBMEM_RW_ERR";
    case UBMEM_RW_ERR_PARAM:    return "UBMEM_RW_ERR_PARAM";
    case UBMEM_RW_ERR_NOTFOUND: return "UBMEM_RW_ERR_NOTFOUND";
    /* Raw SDK codes from ubs_mem_def.h */
    case 6010: return "UBSM_ERR_PARAM_INVALID";
    case 6011: return "UBSM_ERR_NOPERM";
    case 6012: return "UBSM_ERR_MEMORY";
    case 6013: return "UBSM_ERR_UNIMPL";
    case 6014: return "UBSM_CHECK_RESOURCE_ERROR";
    case 6015: return "UBSM_ERR_MEMLIB";
    case 6016: return "UBSM_ERR_NO_NEEDED";
    case 6017: return "UBSM_ERR_BUSY";
    case 6020: return "UBSM_ERR_NOT_FOUND";
    case 6021: return "UBSM_ERR_ALREADY_EXIST";
    case 6022: return "UBSM_ERR_MALLOC_FAIL";
    case 6023: return "UBSM_ERR_RECORD";
    case 6024: return "UBSM_ERR_IN_USING";
    case 6040: return "UBSM_ERR_NET";
    case 6050: return "UBSM_ERR_UBSE";
    case 6051: return "UBSM_ERR_OBMM";
    case 6060: return "UBSM_ERR_LOCK_NOT_SUPPORTED";
    case 6061: return "UBSM_ERR_LOCK_ALREADY_LOCKED";
    case 6062: return "UBSM_ERR_DLOCK";
    case 6099: return "UBSM_ERR_BUFF";
    default:   return "UNKNOWN";
    }
}

/* ================================================================== */
/*  Init / destroy                                                     */
/* ================================================================== */

ubmem_rw_ctx_t* ubmem_rw_init(void)
{
    ubmem_rw_ctx_t *ctx = (ubmem_rw_ctx_t*)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    /* The SDK is very chatty at INFO. Crank it to ERROR (3). We set it
     * twice — once before initialize (in case the logger respects the
     * level pre-init and suppresses init-time spam) and once after
     * (in case the logger is only armed inside initialize). Whichever
     * of the two matters for the installed SDK version, we cover it. */
    (void)ubsmem_set_logger_level(3);

    ubsmem_options_t opts;
    memset(&opts, 0, sizeof(opts));
    (void)ubsmem_init_attributes(&opts);

    int rc = ubsmem_initialize(&opts);
    if (rc != 0) {
        LOG_ERR("ubsmem_initialize failed: rc=%d (%s)",
                rc, ubmem_rw_strerror(rc));
        free(ctx);
        return NULL;
    }
    /* Re-apply log level post-init so subsequent calls are quiet. */
    (void)ubsmem_set_logger_level(3);

    ctx->initialized = true;
    return ctx;
}

void ubmem_rw_destroy(ubmem_rw_ctx_t *ctx)
{
    if (!ctx) return;
    if (ctx->initialized) {
        (void)ubsmem_finalize();
    }
    free(ctx);
}

int ubmem_rw_local_nid(ubmem_rw_ctx_t *ctx, uint32_t *out_nid)
{
    if (!ctx || !out_nid) return UBMEM_RW_ERR_PARAM;
    int rc = ubsmem_local_nid_query(out_nid);
    return (rc == 0) ? UBMEM_RW_OK : rc;
}

int ubmem_rw_query_cluster(ubmem_rw_ctx_t *ctx, ubmem_rw_cluster_t *out)
{
    if (!ctx || !out) return UBMEM_RW_ERR_PARAM;
    memset(out, 0, sizeof(*out));
    out->local_host_idx = -1;

    /* (1) local nid for printing / local-host-idx cross-check */
    int rc = ubsmem_local_nid_query(&out->local_nid);
    if (rc != 0) {
        LOG_ERR("ubsmem_local_nid_query failed: rc=%d (%s)",
                rc, ubmem_rw_strerror(rc));
        return rc;
    }

    /* (2) enumerate hosts as the SDK knows them */
    ubsmem_cluster_info_t ci;
    memset(&ci, 0, sizeof(ci));
    rc = ubsmem_lookup_cluster_statistic(&ci);
    if (rc != 0) {
        LOG_ERR("ubsmem_lookup_cluster_statistic failed: rc=%d (%s)",
                rc, ubmem_rw_strerror(rc));
        return rc;
    }
    if (ci.host_num <= 0) {
        LOG_ERR("cluster has 0 hosts — ubsmd discovery not ready");
        return UBMEM_RW_ERR;
    }

    int nh = ci.host_num;
    if (nh > UBMEM_RW_MAX_HOSTS) nh = UBMEM_RW_MAX_HOSTS;
    out->n_hosts = nh;
    for (int h = 0; h < nh; h++) {
        /* ubsmem_host_info_t.host_name is a fixed-size char[] in the SDK.
         * Copy defensively with a NUL terminator. */
        strncpy(out->hostnames[h], ci.host[h].host_name,
                UBMEM_RW_MAX_HOSTNAME - 1);
        out->hostnames[h][UBMEM_RW_MAX_HOSTNAME - 1] = '\0';
    }

    /* (3) try to find which entry corresponds to the local node.
     * The SDK doesn't expose a direct "which host am I" call, so we
     * fall back to comparing gethostname() against the reported names. */
    char myhost[UBMEM_RW_MAX_HOSTNAME] = {0};
    if (gethostname(myhost, sizeof(myhost) - 1) == 0) {
        /* Some clusters report FQDN; truncate at the first dot for a
         * lenient match ("node1" == "node1.cluster.local"). */
        for (int h = 0; h < nh; h++) {
            if (strcmp(out->hostnames[h], myhost) == 0) {
                out->local_host_idx = h;
                break;
            }
        }
        if (out->local_host_idx < 0) {
            const char *dot = strchr(myhost, '.');
            size_t plain_len = dot ? (size_t)(dot - myhost) : strlen(myhost);
            for (int h = 0; h < nh; h++) {
                if (strncmp(out->hostnames[h], myhost, plain_len) == 0 &&
                    (out->hostnames[h][plain_len] == '\0' ||
                     out->hostnames[h][plain_len] == '.')) {
                    out->local_host_idx = h;
                    break;
                }
            }
        }
    }

    return UBMEM_RW_OK;
}

/* ================================================================== */
/*  Region                                                             */
/* ================================================================== */

int ubmem_rw_ensure_region(ubmem_rw_ctx_t *ctx,
                           const char *region_name,
                           const ubmem_rw_host_t *hosts,
                           int n_hosts)
{
    if (!ctx || !region_name || !hosts || n_hosts <= 0 ||
        n_hosts > MAX_REGION_NODE_NUM) {
        return UBMEM_RW_ERR_PARAM;
    }

    ubsmem_region_attributes_t attr;
    memset(&attr, 0, sizeof(attr));
    attr.host_num = n_hosts;
    for (int i = 0; i < n_hosts; i++) {
        if (!hosts[i].hostname) return UBMEM_RW_ERR_PARAM;
        size_t hn_len = strlen(hosts[i].hostname);
        if (hn_len >= MAX_HOST_NAME_DESC_LENGTH) {
            LOG_ERR("hostname too long: %s", hosts[i].hostname);
            return UBMEM_RW_ERR_PARAM;
        }
        memcpy(attr.hosts[i].host_name, hosts[i].hostname, hn_len + 1);
        attr.hosts[i].affinity = hosts[i].affinity;
    }

    /* size=0 per SDK docstring ("930 no use, default 0") */
    int rc = ubsmem_create_region(region_name, 0, &attr);
    if (rc == 0) {
        LOG_INFO("created region \"%s\" with %d hosts", region_name, n_hosts);
        return UBMEM_RW_OK;
    }
    if (rc == UBSM_ERR_ALREADY_EXIST) {
        LOG_INFO("region \"%s\" already exists — using it", region_name);
        return UBMEM_RW_OK;
    }
    LOG_ERR("ubsmem_create_region(\"%s\") failed: rc=%d (%s)",
            region_name, rc, ubmem_rw_strerror(rc));
    return rc;
}

int ubmem_rw_destroy_region(ubmem_rw_ctx_t *ctx, const char *region_name)
{
    if (!ctx || !region_name) return UBMEM_RW_ERR_PARAM;
    int rc = ubsmem_destroy_region(region_name);
    if (rc == 0 || rc == UBSM_ERR_NOT_FOUND) return UBMEM_RW_OK;
    LOG_ERR("ubsmem_destroy_region(\"%s\") failed: rc=%d (%s)",
            region_name, rc, ubmem_rw_strerror(rc));
    return rc;
}

/* ================================================================== */
/*  Shared memory object                                               */
/* ================================================================== */

int ubmem_rw_allocate(ubmem_rw_ctx_t *ctx,
                      const char *region_name,
                      const char *object_name,
                      size_t      size,
                      unsigned    mode,
                      uint64_t    flags)
{
    if (!ctx || !region_name || !object_name || size == 0) {
        return UBMEM_RW_ERR_PARAM;
    }
    size_t on_len = strlen(object_name);
    if (on_len == 0 || on_len > MAX_SHM_NAME_LENGTH) {
        LOG_ERR("object_name length %zu out of range (max %d)",
                on_len, MAX_SHM_NAME_LENGTH);
        return UBMEM_RW_ERR_PARAM;
    }

    int rc = ubsmem_shmem_allocate(region_name, object_name, size,
                                   (mode_t)mode, flags);
    if (rc == 0) {
        LOG_INFO("allocated shmem \"%s\" size=%zu bytes (%.2f MB) flags=0x%lx",
                 object_name, size, size / (1024.0 * 1024.0),
                 (unsigned long)flags);
        return UBMEM_RW_OK;
    }
    if (rc == UBSM_ERR_ALREADY_EXIST) {
        LOG_INFO("shmem \"%s\" already exists — using it", object_name);
        return UBMEM_RW_OK;
    }
    LOG_ERR("ubsmem_shmem_allocate(\"%s\") failed: rc=%d (%s)",
            object_name, rc, ubmem_rw_strerror(rc));
    return rc;
}

int ubmem_rw_deallocate(ubmem_rw_ctx_t *ctx, const char *object_name)
{
    if (!ctx || !object_name) return UBMEM_RW_ERR_PARAM;
    int rc = ubsmem_shmem_deallocate(object_name);
    if (rc == 0 || rc == UBSM_ERR_NOT_FOUND) return UBMEM_RW_OK;
    LOG_ERR("ubsmem_shmem_deallocate(\"%s\") failed: rc=%d (%s)",
            object_name, rc, ubmem_rw_strerror(rc));
    return rc;
}

/* ================================================================== */
/*  Mapping                                                            */
/* ================================================================== */

int ubmem_rw_map(ubmem_rw_ctx_t *ctx,
                 const char *object_name,
                 size_t length,
                 int prot,
                 int flags,
                 off_t offset,
                 void **out_ptr)
{
    if (!ctx || !object_name || length == 0 || !out_ptr) {
        return UBMEM_RW_ERR_PARAM;
    }

    void *ptr = NULL;
    int rc = ubsmem_shmem_map(NULL, length, prot, flags,
                              object_name, offset, &ptr);
    if (rc != 0 || !ptr) {
        LOG_ERR("ubsmem_shmem_map(\"%s\", len=%zu) failed: rc=%d (%s) ptr=%p",
                object_name, length, rc, ubmem_rw_strerror(rc), ptr);
        return (rc == 0) ? UBMEM_RW_ERR : rc;
    }
    *out_ptr = ptr;
    LOG_INFO("mapped \"%s\" -> %p (len=%zu)", object_name, ptr, length);
    return UBMEM_RW_OK;
}

int ubmem_rw_unmap(ubmem_rw_ctx_t *ctx, void *ptr, size_t length)
{
    if (!ctx || !ptr || length == 0) return UBMEM_RW_ERR_PARAM;
    int rc = ubsmem_shmem_unmap(ptr, length);
    if (rc != 0) {
        LOG_ERR("ubsmem_shmem_unmap(%p, %zu) failed: rc=%d (%s)",
                ptr, length, rc, ubmem_rw_strerror(rc));
        return rc;
    }
    return UBMEM_RW_OK;
}
