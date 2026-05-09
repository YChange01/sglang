/**
 * URMA posted-WRITE ping-pong microbenchmark.
 *
 * This is the URMA counterpart of the UB-MEM noncache write puncture:
 * client writes a sequence-stamped message into the server's registered
 * receive slot, server polls that local slot and writes a response into the
 * client's receive slot, and client reports full RTT plus 1/2 RTT.
 *
 * Default mode uses posted WRITE without CQ polling. Use --cq_mod N to request
 * and poll one completion every N WRs, matching the periodic-signaled style
 * used by UMDK perftest. Use --signaled to wait for every WRITE.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../lib/urma_rw.h"

#define DEFAULT_PORT 13858
#define DEFAULT_BUF_MB 4
#define DEFAULT_ITERS 100000
#define DEFAULT_WARMUP 20000
#define DEFAULT_BYTES 512
#define MIN_MSG_BYTES 16
#define MAX_MSG_BYTES 4096
#define RECV_OFFSET 0
#define SEND_OFFSET 4096

typedef enum {
    ROLE_SERVER,
    ROLE_CLIENT,
} bench_role_t;

typedef struct {
    uint64_t seq;
    uint32_t bytes;
    uint32_t reserved;
} __attribute__((packed)) msg_hdr_t;

typedef struct {
    bench_role_t role;
    const char *server_ip;
    uint16_t port;
    uint32_t bytes;
    uint64_t iters;
    uint64_t warmup;
    uint64_t buf_size;
    const char *dev_name;
    uint64_t cq_mod;
    bool signaled;
} bench_opts_t;

#if defined(__aarch64__)
static inline uint64_t rdcntvct(void)
{
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

static inline uint64_t rdcntfrq(void)
{
    uint64_t f;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}
#else
static inline uint64_t rdcntvct(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint64_t rdcntfrq(void)
{
    return 1000000000ULL;
}
#endif

static int cmp_u64(const void *a, const void *b)
{
    uint64_t av = *(const uint64_t *)a;
    uint64_t bv = *(const uint64_t *)b;
    return (av > bv) - (av < bv);
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

static const char *mode_desc(const bench_opts_t *opts)
{
    static char buf[64];
    if (opts->signaled) return "signaled";
    if (opts->cq_mod == 0) return "posted";
    snprintf(buf, sizeof(buf), "posted+cq_mod=%lu", (unsigned long)opts->cq_mod);
    return buf;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --role server|client [options]\n"
            "Options:\n"
            "  --server_ip IP     server IP for client (default: 192.168.84.245)\n"
            "  --port N           TCP port for URMA exchange (default: 13858)\n"
            "  --bytes N          URMA WRITE bytes, including 16B header (default: 512)\n"
            "  --iters N          measured ping-pong iterations (default: 100000)\n"
            "  --warmup N         warmup ping-pong iterations (default: 20000)\n"
            "  --buf_mb N         registered local buffer size in MB (default: 4)\n"
            "  --dev_name DEV     URMA device name, overrides URMA_DEV env\n"
            "  --urma_dev DEV     alias for --dev_name\n"
            "  --cq_mod N         poll one send completion every N WRITEs (0 disables)\n"
            "  --signaled         wait CQ completion for every WRITE\n",
            prog);
}

static int parse_opts(int argc, char **argv, bench_opts_t *opts)
{
    memset(opts, 0, sizeof(*opts));
    opts->role = ROLE_CLIENT;
    opts->server_ip = "192.168.84.245";
    opts->port = DEFAULT_PORT;
    opts->bytes = DEFAULT_BYTES;
    opts->iters = DEFAULT_ITERS;
    opts->warmup = DEFAULT_WARMUP;
    opts->buf_size = DEFAULT_BUF_MB * 1024ULL * 1024ULL;
    opts->dev_name = getenv("URMA_DEV");
    opts->cq_mod = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--role") == 0 && i + 1 < argc) {
            const char *role = argv[++i];
            if (strcmp(role, "server") == 0) {
                opts->role = ROLE_SERVER;
            } else if (strcmp(role, "client") == 0) {
                opts->role = ROLE_CLIENT;
            } else {
                return -1;
            }
        } else if (strcmp(argv[i], "--server_ip") == 0 && i + 1 < argc) {
            opts->server_ip = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            uint64_t v;
            if (parse_u64(argv[++i], &v) != 0 || v == 0 || v > UINT16_MAX) return -1;
            opts->port = (uint16_t)v;
        } else if (strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            uint64_t v;
            if (parse_u64(argv[++i], &v) != 0 || v < MIN_MSG_BYTES || v > MAX_MSG_BYTES) {
                return -1;
            }
            opts->bytes = (uint32_t)v;
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            if (parse_u64(argv[++i], &opts->iters) != 0 || opts->iters == 0) return -1;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (parse_u64(argv[++i], &opts->warmup) != 0) return -1;
        } else if (strcmp(argv[i], "--buf_mb") == 0 && i + 1 < argc) {
            uint64_t v;
            if (parse_u64(argv[++i], &v) != 0 || v == 0) return -1;
            opts->buf_size = v * 1024ULL * 1024ULL;
        } else if ((strcmp(argv[i], "--dev_name") == 0 ||
                    strcmp(argv[i], "--urma_dev") == 0) && i + 1 < argc) {
            opts->dev_name = argv[++i];
        } else if (strcmp(argv[i], "--cq_mod") == 0 && i + 1 < argc) {
            if (parse_u64(argv[++i], &opts->cq_mod) != 0) return -1;
        } else if (strcmp(argv[i], "--signaled") == 0) {
            opts->signaled = true;
            opts->cq_mod = 1;
        } else {
            return -1;
        }
    }

    if (SEND_OFFSET + opts->bytes > opts->buf_size ||
        RECV_OFFSET + opts->bytes > SEND_OFFSET) {
        fprintf(stderr, "registered buffer too small for bytes=%u\n", opts->bytes);
        return -1;
    }
    return 0;
}

static void prepare_msg(void *base, uint64_t seq, uint32_t bytes)
{
    uint8_t *p = (uint8_t *)base;
    msg_hdr_t *hdr = (msg_hdr_t *)p;
    hdr->seq = 0;
    hdr->bytes = bytes;
    hdr->reserved = 0;

    for (uint32_t i = sizeof(*hdr); i < bytes; i++) {
        p[i] = (uint8_t)((seq + i) & 0xff);
    }

    __atomic_thread_fence(__ATOMIC_RELEASE);
    hdr->seq = seq;
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

static void wait_msg(const void *base, uint64_t seq)
{
    const uint64_t *seqp = (const uint64_t *)base;
    while (__atomic_load_n(seqp, __ATOMIC_ACQUIRE) != seq) {
        __asm__ __volatile__("" ::: "memory");
    }
}

static int write_msg_seq(urma_rw_ctx_t *ctx, const bench_opts_t *opts, uint64_t seq)
{
    if (opts->signaled || (opts->cq_mod != 0 && seq % opts->cq_mod == 0)) {
        return urma_rw_write(ctx, SEND_OFFSET, RECV_OFFSET, opts->bytes);
    }
    return urma_rw_write_post(ctx, SEND_OFFSET, RECV_OFFSET, opts->bytes);
}

static void run_server(urma_rw_ctx_t *ctx, const bench_opts_t *opts)
{
    uint8_t *buf = (uint8_t *)urma_rw_get_buffer(ctx);
    uint64_t total = opts->warmup + opts->iters;

    printf("server ready: bytes=%u total=%lu mode=%s\n",
           opts->bytes, (unsigned long)total,
           mode_desc(opts));

    for (uint64_t seq = 1; seq <= total; seq++) {
        wait_msg(buf + RECV_OFFSET, seq);
        prepare_msg(buf + SEND_OFFSET, seq, opts->bytes);
        int ret = write_msg_seq(ctx, opts, seq);
        if (ret != URMA_RW_OK) {
            fprintf(stderr, "server WRITE failed at seq=%lu ret=%d\n",
                    (unsigned long)seq, ret);
            return;
        }
    }
    printf("server done: handled %lu messages\n", (unsigned long)total);
}

static void run_client(urma_rw_ctx_t *ctx, const bench_opts_t *opts)
{
    uint8_t *buf = (uint8_t *)urma_rw_get_buffer(ctx);
    uint64_t total = opts->warmup + opts->iters;
    uint64_t *samples = (uint64_t *)calloc(opts->iters, sizeof(uint64_t));
    if (!samples) {
        fprintf(stderr, "calloc samples failed\n");
        return;
    }

    printf("client start: bytes=%u warmup=%lu iters=%lu mode=%s\n",
           opts->bytes, (unsigned long)opts->warmup,
           (unsigned long)opts->iters,
           mode_desc(opts));

    for (uint64_t seq = 1; seq <= total; seq++) {
        prepare_msg(buf + SEND_OFFSET, seq, opts->bytes);
        uint64_t t0 = rdcntvct();
        int ret = write_msg_seq(ctx, opts, seq);
        if (ret != URMA_RW_OK) {
            fprintf(stderr, "client WRITE failed at seq=%lu ret=%d\n",
                    (unsigned long)seq, ret);
            free(samples);
            return;
        }
        wait_msg(buf + RECV_OFFSET, seq);
        uint64_t t1 = rdcntvct();
        if (seq > opts->warmup) {
            samples[seq - opts->warmup - 1] = t1 - t0;
        }
    }

    uint64_t frq = rdcntfrq();
    if (frq == 0) frq = 1;
    double ns_per_tick = 1e9 / (double)frq;

    long double sum = 0.0;
    uint64_t min = UINT64_MAX;
    uint64_t max = 0;
    for (uint64_t i = 0; i < opts->iters; i++) {
        uint64_t v = samples[i];
        sum += (long double)v;
        if (v < min) min = v;
        if (v > max) max = v;
    }

    qsort(samples, opts->iters, sizeof(uint64_t), cmp_u64);
    uint64_t p50 = samples[opts->iters / 2];
    uint64_t p99 = samples[(opts->iters * 99) / 100];
    double avg_rtt_ns = (double)(sum / opts->iters) * ns_per_tick;

    printf("\n[URMA WRITE pingpong]\n");
    printf("  bytes=%u iters=%lu warmup=%lu mode=%s counter_freq=%lu\n",
           opts->bytes, (unsigned long)opts->iters,
           (unsigned long)opts->warmup,
           mode_desc(opts),
           (unsigned long)frq);
    printf("  full RTT avg=%.1f ns min=%.1f ns p50=%.1f ns p99=%.1f ns max=%.1f ns\n",
           avg_rtt_ns, (double)min * ns_per_tick, (double)p50 * ns_per_tick,
           (double)p99 * ns_per_tick, (double)max * ns_per_tick);
    printf("  1/2 RTT avg=%.1f ns min=%.1f ns p50=%.1f ns p99=%.1f ns max=%.1f ns\n",
           avg_rtt_ns / 2.0, (double)min * ns_per_tick / 2.0,
           (double)p50 * ns_per_tick / 2.0,
           (double)p99 * ns_per_tick / 2.0,
           (double)max * ns_per_tick / 2.0);

    free(samples);
}

int main(int argc, char **argv)
{
    bench_opts_t opts;
    if (parse_opts(argc, argv, &opts) != 0) {
        usage(argv[0]);
        return 1;
    }

    printf("=== URMA WRITE Pingpong ===\n");
    printf("role=%s server=%s port=%u bytes=%u buf=%lu MB dev=%s mode=%s\n",
           opts.role == ROLE_SERVER ? "server" : "client",
           opts.server_ip, opts.port, opts.bytes,
           (unsigned long)(opts.buf_size / (1024 * 1024)),
           opts.dev_name ? opts.dev_name : "(auto)",
           mode_desc(&opts));

    urma_rw_ctx_t *ctx = urma_rw_init(opts.dev_name, opts.buf_size);
    if (!ctx) {
        fprintf(stderr, "urma_rw_init failed\n");
        return 1;
    }

    int ret;
    if (opts.role == ROLE_SERVER) {
        ret = urma_rw_server_accept(ctx, opts.port);
    } else {
        ret = urma_rw_client_connect(ctx, opts.server_ip, opts.port);
    }
    if (ret != URMA_RW_OK) {
        fprintf(stderr, "URMA connect/accept failed: %d\n", ret);
        urma_rw_destroy(ctx);
        return 1;
    }

    if (opts.role == ROLE_SERVER) {
        run_server(ctx, &opts);
    } else {
        run_client(ctx, &opts);
    }

    urma_rw_destroy(ctx);
    return 0;
}
