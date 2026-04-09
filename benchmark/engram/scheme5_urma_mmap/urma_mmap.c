/**
 * URMA MMAP: Direct remote memory mapping via UB/URMA.
 *
 * Implementation using liburma.so API. The key difference from
 * yuanrong-datasystem's UrmaManager is:
 *   importSegmentFlag.mapping = URMA_SEG_MAP  (not NOMAP)
 * This enables direct load/store to remote memory via UBMMU.
 */

#include "urma_mmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ub/umdk/urma/urma_api.h>
#include <ub/umdk/urma/urma_opcode.h>
#ifdef URMA_OVER_UB
#include <ub/umdk/urma/urma_ubagg.h>
#endif

#define LOG_ERR(fmt, ...) fprintf(stderr, "[urma_mmap ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) fprintf(stdout, "[urma_mmap INFO] " fmt "\n", ##__VA_ARGS__)

/* Use the token from the header */
#define DEFAULT_TOKEN URMA_MMAP_DEFAULT_TOKEN

struct urma_mmap_ctx {
    urma_context_t*  urma_ctx;
    urma_device_t*   urma_dev;
    urma_token_t     token;

    /* Registered local segments (server side) */
    urma_target_seg_t* local_seg;

    /* Imported remote segments (client side) */
    urma_target_seg_t* imported_seg;

    /* Flags */
    urma_reg_seg_flag_t   reg_flag;
    urma_import_seg_flag_t import_flag;
};


/* ------------------------------------------------------------------ */
/*  Init / Destroy                                                     */
/* ------------------------------------------------------------------ */

urma_mmap_ctx_t* urma_mmap_init(const char* dev_name, int eid_index)
{
    urma_mmap_ctx_t* ctx = (urma_mmap_ctx_t*)calloc(1, sizeof(urma_mmap_ctx_t));
    if (!ctx) return NULL;

    ctx->token.token = DEFAULT_TOKEN;

    /* Initialize URMA library */
    urma_init_attr_t init_attr = {0, 0};
    urma_status_t rc = urma_init(&init_attr);
    if (rc != URMA_SUCCESS) {
        LOG_ERR("urma_init failed: %d", rc);
        free(ctx);
        return NULL;
    }

    /* Get device */
    const char* name = dev_name ? dev_name : "ubcore";
    ctx->urma_dev = urma_get_device_by_name((char*)name);
    if (!ctx->urma_dev) {
        /* Try bonding device as fallback */
        ctx->urma_dev = urma_get_device_by_name("bonding_dev_0");
    }
    if (!ctx->urma_dev) {
        LOG_ERR("urma_get_device_by_name(%s) failed", name);
        urma_uninit();
        free(ctx);
        return NULL;
    }
    LOG_INFO("Device: %s, type: %d", name, ctx->urma_dev->type);

    /* Get EID index */
    uint32_t eid_idx = 0;
    if (eid_index >= 0) {
        eid_idx = (uint32_t)eid_index;
    } else {
        uint32_t eid_count = 0;
        urma_eid_info_t* eid_list = urma_get_eid_list(ctx->urma_dev, &eid_count);
        if (eid_list && eid_count > 0) {
            eid_idx = eid_list[0].eid_index;
            urma_free_eid_list(eid_list);
        }
    }

    /* Create context */
    ctx->urma_ctx = urma_create_context(ctx->urma_dev, eid_idx);
    if (!ctx->urma_ctx) {
        LOG_ERR("urma_create_context failed (eid_index=%u)", eid_idx);
        urma_uninit();
        free(ctx);
        return NULL;
    }

    /* Setup registration flags */
    ctx->reg_flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    ctx->reg_flag.bs.token_id_valid = URMA_TOKEN_ID_INVALID;
    ctx->reg_flag.bs.cacheable = URMA_NON_CACHEABLE;
    ctx->reg_flag.bs.reserved = 0;
#ifdef URMA_OVER_UB
    ctx->reg_flag.bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE | URMA_ACCESS_ATOMIC;
#else
    ctx->reg_flag.bs.access = URMA_ACCESS_LOCAL_WRITE | URMA_ACCESS_REMOTE_READ |
                              URMA_ACCESS_REMOTE_WRITE | URMA_ACCESS_REMOTE_ATOMIC;
#endif

    /* Setup import flags — KEY DIFFERENCE: URMA_SEG_MAP instead of NOMAP */
    ctx->import_flag.bs.cacheable = URMA_NON_CACHEABLE;
    ctx->import_flag.bs.mapping = URMA_SEG_MAP;  /* ← Enable memory mapping! */
    ctx->import_flag.bs.reserved = 0;
#ifdef URMA_OVER_UB
    ctx->import_flag.bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE | URMA_ACCESS_ATOMIC;
#else
    ctx->import_flag.bs.access = URMA_ACCESS_LOCAL_WRITE | URMA_ACCESS_REMOTE_READ |
                                 URMA_ACCESS_REMOTE_WRITE | URMA_ACCESS_REMOTE_ATOMIC;
#endif

    LOG_INFO("Context created (eid_index=%u)", eid_idx);
    return ctx;
}

void urma_mmap_destroy(urma_mmap_ctx_t* ctx)
{
    if (!ctx) return;

    if (ctx->imported_seg) {
        urma_unimport_seg(ctx->imported_seg);
        ctx->imported_seg = NULL;
    }
    if (ctx->local_seg) {
        urma_unregister_seg(ctx->local_seg);
        ctx->local_seg = NULL;
    }
    if (ctx->urma_ctx) {
        urma_delete_context(ctx->urma_ctx);
        ctx->urma_ctx = NULL;
    }
    urma_uninit();
    free(ctx);
    LOG_INFO("Context destroyed");
}

int urma_mmap_get_eid(urma_mmap_ctx_t* ctx, char* eid_out, size_t eid_len)
{
    if (!ctx || !ctx->urma_ctx || !eid_out) return URMA_MMAP_ERR_PARAM;
    size_t copy_len = eid_len < URMA_EID_SIZE ? eid_len : URMA_EID_SIZE;
    memcpy(eid_out, ctx->urma_ctx->eid.raw, copy_len);
    return URMA_MMAP_OK;
}


/* ------------------------------------------------------------------ */
/*  Server side: Register local memory                                 */
/* ------------------------------------------------------------------ */

int urma_mmap_register(urma_mmap_ctx_t* ctx, void* addr, uint64_t len,
                       urma_mmap_seg_info_t* info_out)
{
    if (!ctx || !addr || len == 0 || !info_out) return URMA_MMAP_ERR_PARAM;

    urma_seg_cfg_t seg_cfg;
    memset(&seg_cfg, 0, sizeof(seg_cfg));
    seg_cfg.va = (uint64_t)addr;
    seg_cfg.len = len;
    seg_cfg.token_value = ctx->token;
    seg_cfg.flag = ctx->reg_flag;
    seg_cfg.iova = 0;
    seg_cfg.token_id = NULL;
    seg_cfg.user_ctx = 0;

    urma_target_seg_t* seg = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    if (!seg) {
        LOG_ERR("urma_register_seg failed (addr=%p, len=%lu)", addr, len);
        return URMA_MMAP_ERR_REG;
    }

    ctx->local_seg = seg;

    /* Fill output info for client-side import */
    memcpy(info_out->eid, ctx->urma_ctx->eid.raw, sizeof(info_out->eid));
    info_out->uasid = ctx->urma_ctx->uasid;
    info_out->seg_va = seg->seg.va;
    info_out->seg_len = seg->seg.len;
    info_out->seg_id = seg->seg.seg_id;
    info_out->token = DEFAULT_TOKEN;

    LOG_INFO("Registered segment: va=0x%lx, len=%lu, seg_id=%u",
             seg->seg.va, seg->seg.len, seg->seg.seg_id);
    return URMA_MMAP_OK;
}

int urma_mmap_unregister(urma_mmap_ctx_t* ctx, urma_mmap_seg_info_t* info)
{
    if (!ctx || !ctx->local_seg) return URMA_MMAP_ERR_PARAM;
    urma_status_t rc = urma_unregister_seg(ctx->local_seg);
    ctx->local_seg = NULL;
    return (rc == URMA_SUCCESS) ? URMA_MMAP_OK : URMA_MMAP_ERR_REG;
}


/* ------------------------------------------------------------------ */
/*  Client side: Import remote segment with MMAP                       */
/* ------------------------------------------------------------------ */

int urma_mmap_import(urma_mmap_ctx_t* ctx, const urma_mmap_seg_info_t* remote_info,
                     void** mapped_addr)
{
    if (!ctx || !remote_info || !mapped_addr) return URMA_MMAP_ERR_PARAM;

    /* Build remote segment descriptor */
    urma_seg_t remote_seg;
    memset(&remote_seg, 0, sizeof(remote_seg));
    memcpy(remote_seg.ubva.eid.raw, remote_info->eid, URMA_EID_SIZE);
    remote_seg.ubva.uasid = remote_info->uasid;
    remote_seg.ubva.va = remote_info->seg_va;
    remote_seg.len = remote_info->seg_len;
    remote_seg.seg_id = remote_info->seg_id;

    urma_token_t token;
    token.token = remote_info->token;

    /* Import with URMA_SEG_MAP — this creates a local VA mapping */
    urma_target_seg_t* imported = urma_import_seg(
        ctx->urma_ctx, &remote_seg, &token, 0, ctx->import_flag);

    if (!imported) {
        LOG_ERR("urma_import_seg (MAP) failed for remote eid, seg_va=0x%lx",
                remote_info->seg_va);
        return URMA_MMAP_ERR_IMPORT;
    }

    ctx->imported_seg = imported;

    /* The mapped address — UBMMU handles translation on access */
    *mapped_addr = (void*)imported->seg.va;

    LOG_INFO("Imported+mapped remote segment: local_va=%p, len=%lu",
             *mapped_addr, imported->seg.len);
    return URMA_MMAP_OK;
}

int urma_mmap_unimport(urma_mmap_ctx_t* ctx, const urma_mmap_seg_info_t* remote_info)
{
    if (!ctx || !ctx->imported_seg) return URMA_MMAP_ERR_PARAM;
    urma_status_t rc = urma_unimport_seg(ctx->imported_seg);
    ctx->imported_seg = NULL;
    return (rc == URMA_SUCCESS) ? URMA_MMAP_OK : URMA_MMAP_ERR_IMPORT;
}


/* ------------------------------------------------------------------ */
/*  Utility                                                            */
/* ------------------------------------------------------------------ */

void urma_mmap_dump_info(const urma_mmap_seg_info_t* info)
{
    if (!info) return;
    printf("SegInfo: seg_va=0x%lx, seg_len=%lu, seg_id=%u, uasid=%u, token=0x%x\n",
           info->seg_va, info->seg_len, info->seg_id, info->uasid, info->token);
}
