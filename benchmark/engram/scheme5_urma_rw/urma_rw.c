/**
 * URMA RW implementation — based on openEuler umdk urma_sample.c pattern.
 */

#define _GNU_SOURCE
#include "urma_rw.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/mman.h>
#include <ub/umdk/urma/urma_api.h>
#include <ub/umdk/urma/urma_opcode.h>

#ifndef PAGE_SIZE
#define PAGE_SIZE 4096
#endif
#define JETTY_DEPTH 8192  /* > URMA_RW_MAX_BATCH(4096) for headroom */
/* Pure spin — no sleep. On a modern aarch64 core this bounds a poll to a few
 * seconds of wall time (dominated by urma_poll_jfc overhead per call). Sized
 * generously so genuine completions are never lost under load, but low enough
 * to surface true hangs within a reasonable time. */
#define MAX_POLL_TRY 10000000
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

    /* Pre-allocated batch buffers (heap, not stack — DMA-safe) */
    urma_sge_t*    batch_src_sges;
    urma_sge_t*    batch_dst_sges;
    urma_sg_t*     batch_src_sgs;
    urma_sg_t*     batch_dst_sgs;
    urma_jfs_wr_t* batch_wrs;

    int listen_fd;
    int client_fd;
    bool is_server;
};


/* ================================================================== */
/*  Initialization                                                     */
/* ================================================================== */

/* Process-global URMA refcount (urma_init/uninit are ref-counted globally) */
static atomic_int g_urma_refcount = 0;

static int do_urma_init(void)
{
    if (atomic_fetch_add(&g_urma_refcount, 1) == 0) {
        urma_init_attr_t attr = {0};
        urma_status_t rc = urma_init(&attr);
        if (rc != URMA_SUCCESS) {
            LOG_ERR("urma_init failed: %d", rc);
            atomic_fetch_sub(&g_urma_refcount, 1);
            return -1;
        }
    }
    return 0;
}

static void do_urma_uninit(void)
{
    if (atomic_fetch_sub(&g_urma_refcount, 1) == 1) {
        urma_uninit();
    }
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
    } else {
        /* No eids visible — fall back to index 0, but warn loudly. Typically
         * means the device has no route configured, and create_context will
         * fail below with a cryptic message. */
        LOG_ERR("urma_get_eid_list returned %u entries; falling back to eid_idx=0",
                eid_cnt);
        if (eids) urma_free_eid_list(eids);
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

    /* Allocate and register memory (page-aligned) */
    if (posix_memalign(&ctx->buf, PAGE_SIZE, buf_size) != 0 || !ctx->buf) {
        LOG_ERR("posix_memalign %lu bytes failed: %s",
                (unsigned long)buf_size, strerror(errno));
        ctx->buf = NULL;
        goto DEL_JETTY;
    }
    memset(ctx->buf, 0, buf_size);

    /* Pre-allocate heap-based batch work buffers (DMA-safe, not on stack) */
    ctx->batch_src_sges = calloc(URMA_RW_MAX_BATCH, sizeof(urma_sge_t));
    ctx->batch_dst_sges = calloc(URMA_RW_MAX_BATCH, sizeof(urma_sge_t));
    ctx->batch_src_sgs  = calloc(URMA_RW_MAX_BATCH, sizeof(urma_sg_t));
    ctx->batch_dst_sgs  = calloc(URMA_RW_MAX_BATCH, sizeof(urma_sg_t));
    ctx->batch_wrs      = calloc(URMA_RW_MAX_BATCH, sizeof(urma_jfs_wr_t));
    if (!ctx->batch_src_sges || !ctx->batch_dst_sges ||
        !ctx->batch_src_sgs || !ctx->batch_dst_sgs || !ctx->batch_wrs) {
        LOG_ERR("Failed to allocate batch work buffers");
        goto FREE_BUF;
    }

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
    /* Free batch work buffers first (free(NULL) is safe, so partial-alloc
     * and already-NULL cases both work). Then the data buffer. */
    free(ctx->batch_src_sges);
    free(ctx->batch_dst_sges);
    free(ctx->batch_src_sgs);
    free(ctx->batch_dst_sgs);
    free(ctx->batch_wrs);
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
    do_urma_uninit();
FREE_CTX:
    free(ctx);
    return NULL;
}

void urma_rw_destroy(urma_rw_ctx_t* ctx)
{
    if (!ctx) return;

    /* Close sockets first — independent of URMA */
    if (ctx->client_fd >= 0) close(ctx->client_fd);
    if (ctx->listen_fd >= 0) close(ctx->listen_fd);

    /* Teardown order: destroy active objects before unregistering / freeing memory.
     * Jetty/JFR/JFC hold references to the JFC/JFCE hierarchy and may have
     * pending DMA operations — must be destroyed before freeing the buffer. */
    if (ctx->remote_tjetty) urma_unimport_jetty(ctx->remote_tjetty);
    if (ctx->jetty) urma_delete_jetty(ctx->jetty);
    if (ctx->jfr) urma_delete_jfr(ctx->jfr);
    if (ctx->jfc) urma_delete_jfc(ctx->jfc);
    if (ctx->jfce) urma_delete_jfce(ctx->jfce);
    if (ctx->remote_tseg) urma_unimport_seg(ctx->remote_tseg);
    if (ctx->local_tseg) urma_unregister_seg(ctx->local_tseg);
    /* Now safe to free buffer — no in-flight DMA to it */
    if (ctx->buf) free(ctx->buf);

    /* Batch work buffers (heap) */
    free(ctx->batch_src_sges);
    free(ctx->batch_dst_sges);
    free(ctx->batch_src_sgs);
    free(ctx->batch_dst_sgs);
    free(ctx->batch_wrs);

    if (ctx->urma_ctx) urma_delete_context(ctx->urma_ctx);
    do_urma_uninit();
    free(ctx);
}

void* urma_rw_get_buffer(urma_rw_ctx_t* ctx)
{
    return ctx ? ctx->buf : NULL;
}


/* ================================================================== */
/*  TCP seg/jetty info exchange                                        */
/* ================================================================== */

static int sock_send_all(int fd, const void* buf, size_t size)
{
    const char* p = (const char*)buf;
    size_t done = 0;
    while (done < size) {
        ssize_t w = write(fd, p + done, size - done);
        if (w < 0) {
            if (errno == EINTR) continue;
            LOG_ERR("sock_send_all: %s", strerror(errno));
            return -1;
        }
        if (w == 0) {
            LOG_ERR("sock_send_all: peer closed");
            return -1;
        }
        done += (size_t)w;
    }
    return 0;
}

static int sock_recv_all(int fd, void* buf, size_t size)
{
    char* p = (char*)buf;
    size_t done = 0;
    while (done < size) {
        ssize_t r = read(fd, p + done, size - done);
        if (r < 0) {
            if (errno == EINTR) continue;
            LOG_ERR("sock_recv_all: %s", strerror(errno));
            return -1;
        }
        if (r == 0) {
            LOG_ERR("sock_recv_all: peer closed (EOF)");
            return -1;
        }
        done += (size_t)r;
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
        close(ctx->listen_fd);
        ctx->listen_fd = -1;
        return URMA_RW_ERR_SOCKET;
    }
    if (listen(ctx->listen_fd, 1) < 0) {
        LOG_ERR("listen: %s", strerror(errno));
        close(ctx->listen_fd);
        ctx->listen_fd = -1;
        return URMA_RW_ERR_SOCKET;
    }

    LOG_INFO("Server listening on port %u, waiting for client...", port);
    ctx->client_fd = accept(ctx->listen_fd, NULL, NULL);
    if (ctx->client_fd < 0) {
        LOG_ERR("accept: %s", strerror(errno));
        return URMA_RW_ERR_SOCKET;
    }
    LOG_INFO("Client connected");

    /* Half-duplex exchange: server receives first, then sends.
     * Client does the opposite — no deadlock even with zero buffer. */
    seg_jetty_info_t local, remote;
    pack_info(&local, ctx);
    if (sock_recv_all(ctx->client_fd, &remote, sizeof(remote)) != 0) {
        return URMA_RW_ERR_SOCKET;
    }
    if (sock_send_all(ctx->client_fd, &local, sizeof(local)) != 0) {
        return URMA_RW_ERR_SOCKET;
    }

    /* Import remote (client) jetty so bidirectional works.
     * Even though client initiates reads, URMA requires both sides
     * to import each other's jetty for the connection. */
    if (import_remote(ctx, &remote) != 0) {
        return URMA_RW_ERR_IMPORT;
    }

    /* Final sync: server receives then sends one ready byte */
    char byte_in = 0;
    if (sock_recv_all(ctx->client_fd, &byte_in, 1) != 0 ||
        sock_send_all(ctx->client_fd, "R", 1) != 0) {
        return URMA_RW_ERR_SOCKET;
    }

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
    };
    if (inet_pton(AF_INET, server_ip, &addr.sin_addr) != 1) {
        LOG_ERR("invalid server IP: %s", server_ip);
        close(ctx->client_fd);
        ctx->client_fd = -1;
        return URMA_RW_ERR_SOCKET;
    }
    LOG_INFO("Connecting to %s:%u...", server_ip, port);
    if (connect(ctx->client_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERR("connect: %s", strerror(errno));
        return URMA_RW_ERR_SOCKET;
    }

    /* Half-duplex exchange: client sends first, then receives.
     * Server does the opposite — no deadlock. */
    seg_jetty_info_t local, remote;
    pack_info(&local, ctx);
    if (sock_send_all(ctx->client_fd, &local, sizeof(local)) != 0) {
        return URMA_RW_ERR_SOCKET;
    }
    if (sock_recv_all(ctx->client_fd, &remote, sizeof(remote)) != 0) {
        return URMA_RW_ERR_SOCKET;
    }

    if (import_remote(ctx, &remote) != 0) {
        return URMA_RW_ERR_IMPORT;
    }

    /* Final sync: client sends then receives ready byte */
    char byte_in = 0;
    if (sock_send_all(ctx->client_fd, "R", 1) != 0 ||
        sock_recv_all(ctx->client_fd, &byte_in, 1) != 0) {
        return URMA_RW_ERR_SOCKET;
    }

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
            if ((uint64_t)cr.user_ctx != expected_rid) {
                LOG_ERR("Unexpected rid: got %lu, want %lu",
                        (unsigned long)cr.user_ctx, (unsigned long)expected_rid);
                return -1;
            }
            return 0;
        }
    }
    LOG_ERR("poll_completion timeout waiting for rid=%lu",
            (unsigned long)expected_rid);
    return -1;
}

int urma_rw_read(urma_rw_ctx_t* ctx, uint64_t local_offset,
                 uint64_t remote_offset, uint32_t len)
{
    if (!ctx || !ctx->remote_tseg || !ctx->remote_tjetty || len == 0) {
        return URMA_RW_ERR_PARAM;
    }
    if (local_offset + len > ctx->buf_size ||
        remote_offset + len > ctx->remote_tseg->seg.len) {
        LOG_ERR("read out of bounds: local=%lu+%u (buf=%lu), remote=%lu+%u (rseg=%lu)",
                (unsigned long)local_offset, len, (unsigned long)ctx->buf_size,
                (unsigned long)remote_offset, len,
                (unsigned long)ctx->remote_tseg->seg.len);
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
    if (!ctx->remote_tseg || !ctx->remote_tjetty) {
        return URMA_RW_ERR_PARAM;
    }

    /* Bounds-check all offsets */
    uint64_t remote_len = ctx->remote_tseg->seg.len;
    for (uint32_t i = 0; i < count; i++) {
        if (local_offsets[i] + lens[i] > ctx->buf_size ||
            remote_offsets[i] + lens[i] > remote_len) {
            LOG_ERR("batch[%u] out of bounds: local=%lu+%u (buf=%lu), remote=%lu+%u (rseg=%lu)",
                    i, (unsigned long)local_offsets[i], lens[i],
                    (unsigned long)ctx->buf_size,
                    (unsigned long)remote_offsets[i], lens[i],
                    (unsigned long)remote_len);
            return URMA_RW_ERR_PARAM;
        }
    }

    /* Use pre-allocated heap buffers (DMA-safe) */
    urma_sge_t* src_sges = ctx->batch_src_sges;
    urma_sge_t* dst_sges = ctx->batch_dst_sges;
    urma_sg_t*  src_sgs  = ctx->batch_src_sgs;
    urma_sg_t*  dst_sgs  = ctx->batch_dst_sgs;
    urma_jfs_wr_t* wrs   = ctx->batch_wrs;

    /* Reserve a contiguous rid range for the whole batch. The last WR's rid
     * is what we'll match against in the completion below. */
    uint64_t base_rid = __atomic_fetch_add(&ctx->rid, count, __ATOMIC_RELAXED);
    uint64_t last_rid = base_rid + (count - 1);

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
        /* Only enable completion on the last WR in the chain — one completion
         * per batch, avoiding hardware coalescing ambiguity. */
        wrs[i].flag.bs.complete_enable = (i == count - 1) ? 1 : 0;
        wrs[i].tjetty = ctx->remote_tjetty;
        wrs[i].user_ctx = base_rid + i;
        wrs[i].rw.src = src_sgs[i];
        wrs[i].rw.dst = dst_sgs[i];
        wrs[i].next = (i + 1 < count) ? &wrs[i + 1] : NULL;
    }

    urma_jfs_wr_t* bad_wr = NULL;
    if (urma_post_jetty_send_wr(ctx->jetty, &wrs[0], &bad_wr) != URMA_SUCCESS) {
        LOG_ERR("urma_post_jetty_send_wr batch READ failed");
        return URMA_RW_ERR_POST;
    }

    /* Wait for the single completion (only the last WR had complete_enable=1).
     * Verify the CQE's user_ctx matches last_rid so we don't silently consume
     * a stale entry from a prior operation. */
    urma_cr_t cr = {0};
    int tries = 0;
    while (tries < MAX_POLL_TRY) {
        int got = urma_poll_jfc(ctx->jfc, 1, &cr);
        if (got < 0) {
            LOG_ERR("urma_poll_jfc: %d", got);
            return URMA_RW_ERR_POLL;
        }
        if (got > 0) {
            if (cr.status != URMA_CR_SUCCESS) {
                LOG_ERR("batch CR failed: status=%d, rid=%lu",
                        cr.status, (unsigned long)cr.user_ctx);
                return URMA_RW_ERR_POLL;
            }
            if ((uint64_t)cr.user_ctx != last_rid) {
                LOG_ERR("batch CR rid mismatch: got %lu, want %lu",
                        (unsigned long)cr.user_ctx, (unsigned long)last_rid);
                return URMA_RW_ERR_POLL;
            }
            return URMA_RW_OK;
        }
        tries++;
    }
    LOG_ERR("batch completion timeout (last_rid=%lu)",
            (unsigned long)last_rid);
    return URMA_RW_ERR_POLL;
}
