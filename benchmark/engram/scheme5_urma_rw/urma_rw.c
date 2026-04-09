/**
 * URMA RW implementation — based on openEuler umdk urma_sample.c pattern.
 */

#define _GNU_SOURCE
#include "urma_rw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/mman.h>
#include <ub/umdk/urma/urma_api.h>
#include <ub/umdk/urma/urma_opcode.h>

#define PAGE_SIZE 4096
#define JETTY_DEPTH 256
#define MAX_POLL_TRY 1000000       /* ~1 second with 1us sleep */
#define DEFAULT_TOKEN 0xACFE

#define LOG_ERR(fmt, ...) \
    fprintf(stderr, "[urma_rw ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) \
    fprintf(stdout, "[urma_rw INFO] " fmt "\n", ##__VA_ARGS__)

/* ================================================================== */
/*  Wire format for seg + jetty exchange                              */
/* ================================================================== */

typedef struct seg_jetty_info {
    urma_eid_t eid;
    uint32_t uasid;
    uint64_t seg_va;
    uint64_t seg_len;
    uint32_t seg_flag;
    uint32_t seg_token_id;
    urma_jetty_id_t jetty_id;
} __attribute__((packed)) seg_jetty_info_t;


/* ================================================================== */
/*  Context                                                            */
/* ================================================================== */

struct urma_rw_ctx {
    urma_context_t* urma_ctx;
    urma_device_t*  urma_dev;

    urma_jfce_t*   jfce;
    urma_jfc_t*    jfc;
    urma_jfr_t*    jfr;
    urma_jetty_t*  jetty;

    void*          buf;
    uint64_t       buf_size;
    urma_target_seg_t* local_tseg;

    /* After connection established */
    urma_target_seg_t*   remote_tseg;     /* imported remote seg */
    urma_target_jetty_t* remote_tjetty;   /* imported remote jetty */

    urma_token_t   token;
    uint64_t       rid;  /* request id counter */

    int listen_fd;
    int client_fd;
    bool is_server;
};


/* ================================================================== */
/*  Initialization                                                     */
/* ================================================================== */

static int do_urma_init(void)
{
    urma_init_attr_t attr = {0};
    urma_status_t rc = urma_init(&attr);
    if (rc != URMA_SUCCESS) {
        LOG_ERR("urma_init failed: %d", rc);
        return -1;
    }
    return 0;
}

urma_rw_ctx_t* urma_rw_init(const char* dev_name, uint64_t buf_size)
{
    urma_rw_ctx_t* ctx = (urma_rw_ctx_t*)calloc(1, sizeof(urma_rw_ctx_t));
    if (!ctx) return NULL;

    ctx->buf_size = buf_size;
    ctx->token.token = DEFAULT_TOKEN;
    ctx->listen_fd = -1;
    ctx->client_fd = -1;

    if (do_urma_init() != 0) goto FREE_CTX;

    /* Find device */
    const char* name = dev_name ? dev_name : "udma2";
    ctx->urma_dev = urma_get_device_by_name((char*)name);
    if (!ctx->urma_dev) {
        /* try alternatives */
        const char* alternatives[] = {"udma2", "udma3", "udma4", "udma5",
                                       "bonding_dev_0", "ubcore", NULL};
        for (int i = 0; alternatives[i]; i++) {
            ctx->urma_dev = urma_get_device_by_name((char*)alternatives[i]);
            if (ctx->urma_dev) {
                LOG_INFO("Auto-selected device: %s", alternatives[i]);
                break;
            }
        }
    }
    if (!ctx->urma_dev) {
        LOG_ERR("No URMA device found (tried: %s and alternatives)", name);
        goto UNINIT;
    }

    /* Get eid index */
    uint32_t eid_cnt = 0;
    urma_eid_info_t* eids = urma_get_eid_list(ctx->urma_dev, &eid_cnt);
    int eid_idx = 0;
    if (eids && eid_cnt > 0) {
        eid_idx = eids[0].eid_index;
        urma_free_eid_list(eids);
    }

    /* Create context */
    ctx->urma_ctx = urma_create_context(ctx->urma_dev, (uint32_t)eid_idx);
    if (!ctx->urma_ctx) {
        LOG_ERR("urma_create_context failed");
        goto UNINIT;
    }

    /* JFCE */
    ctx->jfce = urma_create_jfce(ctx->urma_ctx);
    if (!ctx->jfce) {
        LOG_ERR("urma_create_jfce failed");
        goto DEL_CTX;
    }

    /* JFC */
    urma_device_attr_t dev_attr;
    if (urma_query_device(ctx->urma_dev, &dev_attr) != URMA_SUCCESS) {
        LOG_ERR("urma_query_device failed");
        goto DEL_JFCE;
    }
    urma_jfc_cfg_t jfc_cfg = {
        .depth = dev_attr.dev_cap.max_jfc_depth,
        .flag = {.value = 0},
        .jfce = ctx->jfce,
        .user_ctx = 0,
    };
    ctx->jfc = urma_create_jfc(ctx->urma_ctx, &jfc_cfg);
    if (!ctx->jfc) {
        LOG_ERR("urma_create_jfc failed");
        goto DEL_JFCE;
    }

    /* JFR */
    urma_jfr_cfg_t jfr_cfg = {
        .depth = JETTY_DEPTH,
        .flag.bs.tag_matching = URMA_NO_TAG_MATCHING,
        .flag.bs.order_type = 0,
        .trans_mode = URMA_TM_RM,  /* Reliable Message */
        .min_rnr_timer = URMA_TYPICAL_MIN_RNR_TIMER,
        .jfc = ctx->jfc,
        .token_value = ctx->token,
        .id = 0,
        .max_sge = 1,
    };
    ctx->jfr = urma_create_jfr(ctx->urma_ctx, &jfr_cfg);
    if (!ctx->jfr) {
        LOG_ERR("urma_create_jfr failed");
        goto DEL_JFC;
    }

    /* Jetty (combines JFS + shared JFR) */
    urma_jfs_cfg_t jfs_cfg = {
        .depth = JETTY_DEPTH,
        .flag.bs.order_type = 0,
        .flag.bs.multi_path = 0,
        .trans_mode = URMA_TM_RM,
        .priority = URMA_MAX_PRIORITY,
        .max_sge = 1,
        .max_inline_data = 0,
        .rnr_retry = URMA_TYPICAL_RNR_RETRY,
        .err_timeout = URMA_TYPICAL_ERR_TIMEOUT,
        .jfc = ctx->jfc,
        .user_ctx = 0,
    };
    urma_jetty_cfg_t jetty_cfg = {
        .flag.bs.share_jfr = 1,
        .jfs_cfg = jfs_cfg,
        .shared.jfr = ctx->jfr,
    };
    ctx->jetty = urma_create_jetty(ctx->urma_ctx, &jetty_cfg);
    if (!ctx->jetty) {
        LOG_ERR("urma_create_jetty failed");
        goto DEL_JFR;
    }

    /* Allocate and register memory */
    ctx->buf = memalign(PAGE_SIZE, buf_size);
    if (!ctx->buf) {
        LOG_ERR("memalign %lu bytes failed", (unsigned long)buf_size);
        goto DEL_JETTY;
    }
    memset(ctx->buf, 0, buf_size);

    urma_reg_seg_flag_t reg_flag = {
        .bs.token_policy = URMA_TOKEN_NONE,
        .bs.cacheable = URMA_NON_CACHEABLE,
        .bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE | URMA_ACCESS_ATOMIC,
        .bs.token_id_valid = 0,
        .bs.reserved = 0,
    };
    urma_seg_cfg_t seg_cfg = {
        .va = (uint64_t)ctx->buf,
        .len = buf_size,
        .token_id = NULL,
        .token_value = ctx->token,
        .flag = reg_flag,
        .user_ctx = 0,
        .iova = 0,
    };
    ctx->local_tseg = urma_register_seg(ctx->urma_ctx, &seg_cfg);
    if (!ctx->local_tseg) {
        LOG_ERR("urma_register_seg failed");
        goto FREE_BUF;
    }

    LOG_INFO("Initialized: device=%s, buf=%p, size=%lu MB",
             ctx->urma_dev->name, ctx->buf,
             (unsigned long)(buf_size / (1024 * 1024)));
    return ctx;

FREE_BUF:
    free(ctx->buf);
DEL_JETTY:
    urma_delete_jetty(ctx->jetty);
DEL_JFR:
    urma_delete_jfr(ctx->jfr);
DEL_JFC:
    urma_delete_jfc(ctx->jfc);
DEL_JFCE:
    urma_delete_jfce(ctx->jfce);
DEL_CTX:
    urma_delete_context(ctx->urma_ctx);
UNINIT:
    urma_uninit();
FREE_CTX:
    free(ctx);
    return NULL;
}

void urma_rw_destroy(urma_rw_ctx_t* ctx)
{
    if (!ctx) return;

    if (ctx->client_fd >= 0) close(ctx->client_fd);
    if (ctx->listen_fd >= 0) close(ctx->listen_fd);

    if (ctx->remote_tjetty) urma_unimport_jetty(ctx->remote_tjetty);
    if (ctx->remote_tseg) urma_unimport_seg(ctx->remote_tseg);
    if (ctx->local_tseg) urma_unregister_seg(ctx->local_tseg);
    if (ctx->buf) free(ctx->buf);
    if (ctx->jetty) urma_delete_jetty(ctx->jetty);
    if (ctx->jfr) urma_delete_jfr(ctx->jfr);
    if (ctx->jfc) urma_delete_jfc(ctx->jfc);
    if (ctx->jfce) urma_delete_jfce(ctx->jfce);
    if (ctx->urma_ctx) urma_delete_context(ctx->urma_ctx);
    urma_uninit();
    free(ctx);
}

void* urma_rw_get_buffer(urma_rw_ctx_t* ctx)
{
    return ctx ? ctx->buf : NULL;
}


/* ================================================================== */
/*  TCP seg/jetty info exchange                                        */
/* ================================================================== */

static int sock_sync(int fd, int size, char* local, char* remote)
{
    ssize_t w = write(fd, local, size);
    if (w < size) {
        LOG_ERR("sock_sync write: %s", strerror(errno));
        return -1;
    }
    int total = 0;
    while (total < size) {
        ssize_t r = read(fd, remote + total, size - total);
        if (r <= 0) {
            LOG_ERR("sock_sync read: %s", strerror(errno));
            return -1;
        }
        total += r;
    }
    return 0;
}

static void pack_info(seg_jetty_info_t* info, const urma_rw_ctx_t* ctx)
{
    memset(info, 0, sizeof(*info));
    info->eid = ctx->urma_ctx->eid;
    info->uasid = ctx->urma_ctx->uasid;
    info->seg_va = ctx->local_tseg->seg.ubva.va;
    info->seg_len = ctx->local_tseg->seg.len;
    info->seg_flag = ctx->local_tseg->seg.attr.value;
    info->seg_token_id = ctx->local_tseg->seg.token_id;
    info->jetty_id = ctx->jetty->jetty_id;
}

static int import_remote(urma_rw_ctx_t* ctx, const seg_jetty_info_t* remote)
{
    /* Build remote seg descriptor */
    urma_seg_t rseg = {0};
    rseg.ubva.eid = remote->eid;
    rseg.ubva.uasid = remote->uasid;
    rseg.ubva.va = remote->seg_va;
    rseg.len = remote->seg_len;
    rseg.attr.value = remote->seg_flag;
    rseg.token_id = remote->seg_token_id;

    urma_import_seg_flag_t seg_flag = {
        .bs.cacheable = URMA_NON_CACHEABLE,
        .bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE | URMA_ACCESS_ATOMIC,
        .bs.mapping = URMA_SEG_NOMAP,
        .bs.reserved = 0,
    };
    ctx->remote_tseg = urma_import_seg(ctx->urma_ctx, &rseg, &ctx->token, 0, seg_flag);
    if (!ctx->remote_tseg) {
        LOG_ERR("urma_import_seg failed");
        return -1;
    }

    /* Import remote jetty */
    urma_rjetty_t rjetty = {
        .jetty_id = remote->jetty_id,
        .trans_mode = URMA_TM_RM,
        .type = URMA_JETTY,
        .tp_type = URMA_RTP,
        .flag.bs.order_type = 0,
        .flag.bs.share_tp = 0,
    };
    ctx->remote_tjetty = urma_import_jetty(ctx->urma_ctx, &rjetty, &ctx->token);
    if (!ctx->remote_tjetty) {
        LOG_ERR("urma_import_jetty failed");
        urma_unimport_seg(ctx->remote_tseg);
        ctx->remote_tseg = NULL;
        return -1;
    }

    LOG_INFO("Imported remote seg (va=0x%lx, len=%lu) and jetty",
             (unsigned long)remote->seg_va, (unsigned long)remote->seg_len);
    return 0;
}


/* ================================================================== */
/*  Server side                                                        */
/* ================================================================== */

int urma_rw_server_accept(urma_rw_ctx_t* ctx, uint16_t port)
{
    int enable = 1;
    ctx->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (ctx->listen_fd < 0) return URMA_RW_ERR_SOCKET;

    setsockopt(ctx->listen_fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_addr.s_addr = htonl(INADDR_ANY),
        .sin_port = htons(port),
    };
    if (bind(ctx->listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERR("bind port %u: %s", port, strerror(errno));
        return URMA_RW_ERR_SOCKET;
    }
    if (listen(ctx->listen_fd, 1) < 0) {
        LOG_ERR("listen: %s", strerror(errno));
        return URMA_RW_ERR_SOCKET;
    }

    LOG_INFO("Server listening on port %u, waiting for client...", port);
    ctx->client_fd = accept(ctx->listen_fd, NULL, NULL);
    if (ctx->client_fd < 0) {
        LOG_ERR("accept: %s", strerror(errno));
        return URMA_RW_ERR_SOCKET;
    }
    LOG_INFO("Client connected");

    /* Exchange info */
    seg_jetty_info_t local, remote;
    pack_info(&local, ctx);
    if (sock_sync(ctx->client_fd, sizeof(local), (char*)&local, (char*)&remote) != 0) {
        return URMA_RW_ERR_SOCKET;
    }

    /* Import remote (client) jetty so bidirectional works.
     * Even though client initiates reads, URMA requires both sides
     * to import each other's jetty for the connection. */
    if (import_remote(ctx, &remote) != 0) {
        return URMA_RW_ERR_IMPORT;
    }

    /* Final sync */
    char sync_msg = 0;
    sock_sync(ctx->client_fd, 1, "S", &sync_msg);

    ctx->is_server = true;
    LOG_INFO("Server ready — data is accessible to client");
    return URMA_RW_OK;
}


/* ================================================================== */
/*  Client side                                                        */
/* ================================================================== */

int urma_rw_client_connect(urma_rw_ctx_t* ctx, const char* server_ip, uint16_t port)
{
    ctx->client_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (ctx->client_fd < 0) return URMA_RW_ERR_SOCKET;

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
        .sin_addr.s_addr = inet_addr(server_ip),
    };
    LOG_INFO("Connecting to %s:%u...", server_ip, port);
    if (connect(ctx->client_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERR("connect: %s", strerror(errno));
        return URMA_RW_ERR_SOCKET;
    }

    /* Exchange info */
    seg_jetty_info_t local, remote;
    pack_info(&local, ctx);
    if (sock_sync(ctx->client_fd, sizeof(local), (char*)&local, (char*)&remote) != 0) {
        return URMA_RW_ERR_SOCKET;
    }

    if (import_remote(ctx, &remote) != 0) {
        return URMA_RW_ERR_IMPORT;
    }

    /* Final sync */
    char sync_msg = 0;
    sock_sync(ctx->client_fd, 1, "S", &sync_msg);

    ctx->is_server = false;
    LOG_INFO("Client ready — can now urma_read from remote");
    return URMA_RW_OK;
}


/* ================================================================== */
/*  Data transfer: urma_read                                           */
/* ================================================================== */

static int poll_completion(urma_rw_ctx_t* ctx, uint64_t expected_rid)
{
    urma_cr_t cr = {0};
    for (int try = 0; try < MAX_POLL_TRY; try++) {
        int n = urma_poll_jfc(ctx->jfc, 1, &cr);
        if (n < 0) {
            LOG_ERR("urma_poll_jfc: %d", n);
            return -1;
        }
        if (n > 0) {
            if (cr.status != URMA_CR_SUCCESS) {
                LOG_ERR("CR failed: status=%d, rid=%lu",
                        cr.status, (unsigned long)cr.user_ctx);
                return -1;
            }
            return 0;
        }
    }
    LOG_ERR("poll_completion timeout waiting for rid=%lu",
            (unsigned long)expected_rid);
    return -1;
}

static int poll_n_completions(urma_rw_ctx_t* ctx, uint32_t n)
{
    urma_cr_t cr;
    uint32_t done = 0;
    int tries = 0;
    while (done < n && tries < MAX_POLL_TRY) {
        int got = urma_poll_jfc(ctx->jfc, 1, &cr);
        if (got < 0) {
            LOG_ERR("urma_poll_jfc: %d", got);
            return -1;
        }
        if (got > 0) {
            if (cr.status != URMA_CR_SUCCESS) {
                LOG_ERR("CR failed: status=%d", cr.status);
                return -1;
            }
            done++;
        } else {
            tries++;
        }
    }
    return (done == n) ? 0 : -1;
}

int urma_rw_read(urma_rw_ctx_t* ctx, uint64_t local_offset,
                 uint64_t remote_offset, uint32_t len)
{
    if (!ctx || !ctx->remote_tseg || !ctx->remote_tjetty) {
        return URMA_RW_ERR_PARAM;
    }

    /* For urma_read:
     *   src = remote VA
     *   dst = local VA
     * We use the post_jetty_send_wr path with opcode=READ. */
    urma_sge_t src_sge = {
        .addr = ctx->remote_tseg->seg.ubva.va + remote_offset,
        .len = len,
        .tseg = ctx->remote_tseg,
    };
    urma_sge_t dst_sge = {
        .addr = (uint64_t)ctx->buf + local_offset,
        .len = len,
        .tseg = ctx->local_tseg,
    };
    urma_sg_t src_sg = {.sge = &src_sge, .num_sge = 1};
    urma_sg_t dst_sg = {.sge = &dst_sge, .num_sge = 1};
    urma_rw_wr_t rw = {.src = src_sg, .dst = dst_sg};

    uint64_t rid = __atomic_fetch_add(&ctx->rid, 1, __ATOMIC_RELAXED);
    urma_jfs_wr_t wr = {
        .opcode = URMA_OPC_READ,
        .flag.bs.complete_enable = 1,
        .flag.bs.inline_flag = 0,
        .tjetty = ctx->remote_tjetty,
        .user_ctx = rid,
        .rw = rw,
        .next = NULL,
    };
    urma_jfs_wr_t* bad_wr = NULL;
    if (urma_post_jetty_send_wr(ctx->jetty, &wr, &bad_wr) != URMA_SUCCESS) {
        LOG_ERR("urma_post_jetty_send_wr READ failed");
        return URMA_RW_ERR_POST;
    }

    return poll_completion(ctx, rid);
}

int urma_rw_read_batch(urma_rw_ctx_t* ctx,
                       const uint64_t* local_offsets,
                       const uint64_t* remote_offsets,
                       const uint32_t* lens,
                       uint32_t count)
{
    if (!ctx || !local_offsets || !remote_offsets || !lens || count == 0 ||
        count > URMA_RW_MAX_BATCH) {
        return URMA_RW_ERR_PARAM;
    }

    /* Submit all WRs first (chained) */
    urma_sge_t  src_sges[URMA_RW_MAX_BATCH];
    urma_sge_t  dst_sges[URMA_RW_MAX_BATCH];
    urma_sg_t   src_sgs[URMA_RW_MAX_BATCH];
    urma_sg_t   dst_sgs[URMA_RW_MAX_BATCH];
    urma_jfs_wr_t wrs[URMA_RW_MAX_BATCH];

    for (uint32_t i = 0; i < count; i++) {
        src_sges[i].addr = ctx->remote_tseg->seg.ubva.va + remote_offsets[i];
        src_sges[i].len = lens[i];
        src_sges[i].tseg = ctx->remote_tseg;

        dst_sges[i].addr = (uint64_t)ctx->buf + local_offsets[i];
        dst_sges[i].len = lens[i];
        dst_sges[i].tseg = ctx->local_tseg;

        src_sgs[i].sge = &src_sges[i];
        src_sgs[i].num_sge = 1;
        dst_sgs[i].sge = &dst_sges[i];
        dst_sgs[i].num_sge = 1;

        wrs[i].opcode = URMA_OPC_READ;
        wrs[i].flag.value = 0;
        wrs[i].flag.bs.complete_enable = 1;
        wrs[i].tjetty = ctx->remote_tjetty;
        wrs[i].user_ctx = __atomic_fetch_add(&ctx->rid, 1, __ATOMIC_RELAXED);
        wrs[i].rw.src = src_sgs[i];
        wrs[i].rw.dst = dst_sgs[i];
        wrs[i].next = (i + 1 < count) ? &wrs[i + 1] : NULL;
    }

    urma_jfs_wr_t* bad_wr = NULL;
    if (urma_post_jetty_send_wr(ctx->jetty, &wrs[0], &bad_wr) != URMA_SUCCESS) {
        LOG_ERR("urma_post_jetty_send_wr batch READ failed");
        return URMA_RW_ERR_POST;
    }

    return poll_n_completions(ctx, count);
}
