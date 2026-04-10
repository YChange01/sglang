/**
 * Unified Cross-Node Benchmark — All Transport Modes
 *
 * Modes:
 *   local       — Local DRAM memcpy (theoretical ceiling)
 *   tcp         — TCP socket read from server
 *   urma        — URMA urma_read (RDMA one-sided READ)
 *   ubsmem      — ubs_mem load/store (CACHE)
 *   ubsmem-nc   — ubs_mem load/store (NONCACHE / O_SYNC)
 *   ubsmem-huge — ubs_mem load/store (2MB hugepage)
 *   all         — Run all modes sequentially
 *
 * Usage:
 *   ./bench_all <mode> [options]
 *
 * Options:
 *   --size_mb N       shmem/buffer size in MB (default 128)
 *   --name NAME       shmem object name (default "engram_test")
 *   --rows N          number of rows (default 10000)
 *   --dim N           embedding dimension (default 341)
 *   --iters N         iterations per benchmark (default 500)
 *   --server_ip IP    server IP for TCP/URMA (default "192.168.84.245")
 *   --tcp_port N      TCP data port (default 13900)
 *   --urma_port N     URMA seg exchange port (default 13857)
 *   --provider HOST   ubs_mem provider hostname (default "node1")
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <ubs_mem.h>

/* URMA RW library */
#include "../lib/urma_rw.h"

#define NUM_TABLES 12
#define DEFAULT_TCP_PORT  13900
#define DEFAULT_URMA_PORT 13857

/* ------------------------------------------------------------------ */
/*  Timing helpers                                                    */
/* ------------------------------------------------------------------ */

static double diff_us(struct timespec *a, struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

static double diff_ns(struct timespec *a, struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e9 + (b->tv_nsec - a->tv_nsec);
}

/* ------------------------------------------------------------------ */
/*  TCP client helpers                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t offset;
    uint32_t length;
} __attribute__((packed)) tcp_req_t;

static int tcp_recv_all(int fd, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = recv(fd, (char *)buf + done, len - done, 0);
        if (n <= 0) return -1;
        done += n;
    }
    return 0;
}

static int tcp_send_all(int fd, const void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = send(fd, (const char *)buf + done, len - done, 0);
        if (n <= 0) return -1;
        done += n;
    }
    return 0;
}

static int tcp_connect(const char *ip, uint16_t port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &addr.sin_addr);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect"); close(fd); return -1;
    }

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    return fd;
}

static void tcp_disconnect(int fd)
{
    tcp_req_t eof = { .offset = 0xFFFFFFFFFFFFFFFFULL, .length = 0 };
    tcp_send_all(fd, &eof, sizeof(eof));
    close(fd);
}

/* ------------------------------------------------------------------ */
/*  Read abstraction                                                  */
/* ------------------------------------------------------------------ */

typedef enum {
    MODE_LOCAL,
    MODE_TCP,
    MODE_URMA,
    MODE_UBSMEM,
    MODE_UBSMEM_NC,
    MODE_UBSMEM_HUGE,
} bench_mode_t;

typedef struct {
    bench_mode_t mode;
    const char *mode_name;

    /* For local / ubsmem: pointer to mapped data */
    const float *data_ptr;

    /* For TCP */
    int tcp_fd;

    /* For URMA */
    urma_rw_ctx_t *urma_ctx;

    /* Local receive buffer */
    float *local_buf;
    size_t local_buf_size;
} bench_ctx_t;

/* Read one row into local_buf. Returns 0 on success. */
static int bench_read_row(bench_ctx_t *ctx, int row, int dim)
{
    int row_bytes = dim * (int)sizeof(float);

    switch (ctx->mode) {
    case MODE_LOCAL:
    case MODE_UBSMEM:
    case MODE_UBSMEM_NC:
    case MODE_UBSMEM_HUGE:
        memcpy(ctx->local_buf, ctx->data_ptr + (size_t)row * dim, row_bytes);
        return 0;

    case MODE_TCP: {
        tcp_req_t req = { .offset = (uint64_t)row * row_bytes, .length = row_bytes };
        if (tcp_send_all(ctx->tcp_fd, &req, sizeof(req)) != 0) return -1;
        if (tcp_recv_all(ctx->tcp_fd, ctx->local_buf, row_bytes) != 0) return -1;
        return 0;
    }

    case MODE_URMA:
        return urma_rw_read(ctx->urma_ctx, 0,
                            (uint64_t)row * row_bytes, row_bytes);
    }
    return -1;
}

/* Read N rows. For URMA uses batch post+poll. */
static int bench_read_batch(bench_ctx_t *ctx, const int *rows, int count, int dim)
{
    int row_bytes = dim * (int)sizeof(float);

    switch (ctx->mode) {
    case MODE_LOCAL:
    case MODE_UBSMEM:
    case MODE_UBSMEM_NC:
    case MODE_UBSMEM_HUGE:
        for (int i = 0; i < count; i++) {
            memcpy(ctx->local_buf + (size_t)i * dim,
                   ctx->data_ptr + (size_t)rows[i] * dim, row_bytes);
        }
        return 0;

    case MODE_TCP: {
        for (int i = 0; i < count; i++) {
            tcp_req_t req = { .offset = (uint64_t)rows[i] * row_bytes,
                              .length = row_bytes };
            if (tcp_send_all(ctx->tcp_fd, &req, sizeof(req)) != 0) return -1;
        }
        for (int i = 0; i < count; i++) {
            if (tcp_recv_all(ctx->tcp_fd, ctx->local_buf + (size_t)i * dim,
                             row_bytes) != 0) return -1;
        }
        return 0;
    }

    case MODE_URMA: {
        if (count > URMA_RW_MAX_BATCH) return -1;
        /* Use static buffers to avoid malloc in hot path */
        static uint64_t s_locals[URMA_RW_MAX_BATCH];
        static uint64_t s_remotes[URMA_RW_MAX_BATCH];
        static uint32_t s_lens[URMA_RW_MAX_BATCH];
        for (int i = 0; i < count; i++) {
            s_locals[i]  = (uint64_t)i * row_bytes;
            s_remotes[i] = (uint64_t)rows[i] * row_bytes;
            s_lens[i]    = row_bytes;
        }
        return urma_rw_read_batch(ctx->urma_ctx, s_locals, s_remotes, s_lens, count);
    }
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/*  Benchmarks                                                        */
/* ------------------------------------------------------------------ */

static void bench_single_load(bench_ctx_t *ctx, int num_rows, int dim,
                              int num_iters)
{
    if (ctx->mode == MODE_TCP || ctx->mode == MODE_URMA) {
        printf("\n  [Single float load] (skipped for %s)\n", ctx->mode_name);
        return;
    }

    printf("\n  [Single float load] 4 bytes, %d iters\n", num_iters);

    volatile float sink = 0;
    for (int i = 0; i < 100; i++) sink += ctx->data_ptr[i];

    size_t total_floats = (size_t)num_rows * dim;
    size_t *indices = (size_t *)malloc(num_iters * sizeof(size_t));
    if (!indices) return;
    srand(123);
    for (int i = 0; i < num_iters; i++)
        indices[i] = (size_t)(rand() % (int)(total_floats > INT_MAX ? INT_MAX : total_floats));

    struct timespec t0, t1;
    double total_ns = 0, min_ns = 1e12, max_ns = 0;

    for (int i = 0; i < num_iters; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        sink = ctx->data_ptr[indices[i]];
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = diff_ns(&t0, &t1);
        total_ns += ns;
        if (ns < min_ns) min_ns = ns;
        if (ns > max_ns) max_ns = ns;
    }

    printf("    Avg: %.1f ns, Min: %.1f ns, Max: %.1f ns\n",
           total_ns / num_iters, min_ns, max_ns);
    (void)sink;
    free(indices);
}

static void bench_single_row(bench_ctx_t *ctx, int num_rows, int dim,
                              int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    printf("\n  [Single-row read] %d bytes, %d iters\n", row_bytes, num_iters);

    for (int i = 0; i < 10; i++) bench_read_row(ctx, i % num_rows, dim);

    struct timespec t0, t1;
    double total_us = 0, min_us = 1e9, max_us = 0;

    srand(42);
    for (int i = 0; i < num_iters; i++) {
        int row = rand() % num_rows;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        bench_read_row(ctx, row, dim);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = diff_us(&t0, &t1);
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }

    double avg = total_us / num_iters;
    printf("    Avg: %.3f us, Min: %.3f us, Max: %.3f us\n", avg, min_us, max_us);
    printf("    Throughput: %.1f MB/s\n", (double)row_bytes / avg);
}

static void bench_batch(bench_ctx_t *ctx, int num_rows, int dim, int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    int batches[] = {32, 64, 128, 256};

    printf("\n  [Batch read]\n");
    printf("    %-8s %-10s %-12s %-12s %-12s\n",
           "Batch", "Data", "Total us", "Per-row us", "Throughput");
    printf("    -----------------------------------------------------------\n");

    for (int b = 0; b < 4; b++) {
        int batch = batches[b];
        int *rows = (int *)malloc(batch * sizeof(int));
        if (!rows) continue;

        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            for (int i = 0; i < batch; i++) rows[i] = rand() % num_rows;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            bench_read_batch(ctx, rows, batch, dim);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg = total_us / num_iters;
        double per_row = avg / batch;
        double data_kb = (double)batch * row_bytes / 1024.0;
        double throughput = (data_kb / 1024.0) / (avg / 1e6);
        printf("    %-8d %-8.1fKB %-10.2f %-10.3f %.1f MB/s\n",
               batch, data_kb, avg, per_row, throughput);
        free(rows);
    }
}

static void bench_engram_prefetch(bench_ctx_t *ctx, int num_rows, int dim,
                                  int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    int token_counts[] = {32, 64, 128, 256};

    printf("\n  [Engram prefetch] %d tables x N tokens\n", NUM_TABLES);
    printf("    %-8s %-8s %-10s %-12s %-12s\n",
           "Tokens", "Reads", "Data", "Avg ms", "Throughput");
    printf("    -----------------------------------------------------------\n");

    for (int b = 0; b < 4; b++) {
        int tokens = token_counts[b];
        int total_reads = NUM_TABLES * tokens;

        int *rows = (int *)malloc(total_reads * sizeof(int));
        if (!rows) continue;

        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            for (int i = 0; i < total_reads; i++) rows[i] = rand() % num_rows;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            bench_read_batch(ctx, rows, total_reads, dim);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg_ms = total_us / num_iters / 1000.0;
        double data_mb = (double)total_reads * row_bytes / (1024.0 * 1024.0);
        double throughput = data_mb / (avg_ms / 1e3);
        printf("    %-8d %-8d %-8.1fMB %-10.4f %.1f MB/s\n",
               tokens, total_reads, data_mb, avg_ms, throughput);
        free(rows);
    }
}

static void bench_cold_hot(bench_ctx_t *ctx, int num_rows, int dim)
{
    if (ctx->mode == MODE_TCP || ctx->mode == MODE_URMA) {
        printf("\n  [Cold/Hot analysis] (skipped for %s)\n", ctx->mode_name);
        return;
    }

    int row_bytes = dim * (int)sizeof(float);
    printf("\n  [Cold/Hot analysis] first access vs cached\n");

    struct timespec t0, t1;

    int cold_rows[] = {0, num_rows/4, num_rows/2, num_rows*3/4, num_rows-1};
    int ncold = 5;
    if (cold_rows[4] >= num_rows) cold_rows[4] = num_rows - 1;

    printf("    %-12s %-12s %-12s\n", "Row", "Cold (us)", "Hot (us)");
    printf("    ------------------------------------\n");

    for (int r = 0; r < ncold; r++) {
        int row = cold_rows[r];
        if (row >= num_rows) continue;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(ctx->local_buf, ctx->data_ptr + (size_t)row * dim, row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double cold_us = diff_us(&t0, &t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(ctx->local_buf, ctx->data_ptr + (size_t)row * dim, row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double hot_us = diff_us(&t0, &t1);

        printf("    %-12d %-12.3f %-12.3f\n", row, cold_us, hot_us);
    }
}

/* ------------------------------------------------------------------ */
/*  Paper reproduction: Engram-27B (arXiv:2603.10087 Figure 3/5)      */
/* ------------------------------------------------------------------ */

/*
 * Engram-27B parameters:
 *   vocab_size = 2,262,400; emb_dim = 1,280
 *   Per token: 8 hash-mapped segments, each 320 bytes, sparse addresses
 *   Batch sizes: 1, 4, 16, 64, 256, 1024
 */
#define PAPER_SEGS_PER_TOKEN  8
#define PAPER_SEG_BYTES       320
#define PAPER_SEG_FLOATS      (PAPER_SEG_BYTES / (int)sizeof(float))  /* 80 */

static void bench_paper_read_batch(bench_ctx_t *ctx, const uint64_t *offsets,
                                    int count)
{
    switch (ctx->mode) {
    case MODE_LOCAL:
    case MODE_UBSMEM:
    case MODE_UBSMEM_NC:
    case MODE_UBSMEM_HUGE:
        for (int i = 0; i < count; i++) {
            memcpy((char *)ctx->local_buf + (size_t)i * PAPER_SEG_BYTES,
                   (const char *)ctx->data_ptr + offsets[i],
                   PAPER_SEG_BYTES);
        }
        break;

    case MODE_TCP:
        for (int i = 0; i < count; i++) {
            tcp_req_t req = { .offset = offsets[i], .length = PAPER_SEG_BYTES };
            tcp_send_all(ctx->tcp_fd, &req, sizeof(req));
        }
        for (int i = 0; i < count; i++) {
            tcp_recv_all(ctx->tcp_fd, (char *)ctx->local_buf + (size_t)i * PAPER_SEG_BYTES,
                         PAPER_SEG_BYTES);
        }
        break;

    case MODE_URMA: {
        static uint64_t s_locals[URMA_RW_MAX_BATCH];
        static uint64_t s_remotes[URMA_RW_MAX_BATCH];
        static uint32_t s_lens[URMA_RW_MAX_BATCH];
        for (int i = 0; i < count; i++) {
            s_locals[i]  = (uint64_t)i * PAPER_SEG_BYTES;
            s_remotes[i] = offsets[i];
            s_lens[i]    = PAPER_SEG_BYTES;
        }
        urma_rw_read_batch(ctx->urma_ctx, s_locals, s_remotes, s_lens, count);
        break;
    }
    }
}

static void bench_paper_27b(bench_ctx_t *ctx, size_t data_size, int num_iters)
{
    int batch_sizes[] = {1, 4, 16, 64, 256, 1024};
    int nbatches = 6;

    printf("\n  [Paper: Engram-27B] 8 segs x 320B per token, sparse\n");
    printf("    %-8s %-8s %-10s %-12s %-12s\n",
           "Batch", "Reads", "Data", "Latency", "Throughput");
    printf("    -----------------------------------------------------------\n");

    /* Max offset: ensure segment fits within data_size */
    uint64_t max_offset = data_size - PAPER_SEG_BYTES;

    for (int b = 0; b < nbatches; b++) {
        int batch = batch_sizes[b];
        int total_segs = batch * PAPER_SEGS_PER_TOKEN;

        if (total_segs > URMA_RW_MAX_BATCH && ctx->mode == MODE_URMA) {
            printf("    %-8d (skipped: %d > URMA_RW_MAX_BATCH)\n", batch, total_segs);
            continue;
        }

        uint64_t *offsets = (uint64_t *)malloc(total_segs * sizeof(uint64_t));
        if (!offsets) continue;

        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            /* Generate sparse random offsets (320-byte aligned for realism) */
            for (int i = 0; i < total_segs; i++) {
                offsets[i] = ((uint64_t)(rand() % (int)(max_offset / PAPER_SEG_BYTES)))
                             * PAPER_SEG_BYTES;
            }
            clock_gettime(CLOCK_MONOTONIC, &t0);
            bench_paper_read_batch(ctx, offsets, total_segs);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg_us = total_us / num_iters;
        double data_kb = (double)total_segs * PAPER_SEG_BYTES / 1024.0;
        double throughput_mbs = (data_kb / 1024.0) / (avg_us / 1e6);

        if (avg_us >= 1000.0) {
            printf("    %-8d %-8d %-8.1fKB %-10.2fms %.1f MB/s\n",
                   batch, total_segs, data_kb, avg_us / 1000.0, throughput_mbs);
        } else {
            printf("    %-8d %-8d %-8.1fKB %-10.2fus %.1f MB/s\n",
                   batch, total_segs, data_kb, avg_us, throughput_mbs);
        }
        free(offsets);
    }
}

/* ------------------------------------------------------------------ */
/*  Run all benchmarks for one mode                                   */
/* ------------------------------------------------------------------ */

static void run_benchmarks(bench_ctx_t *ctx, int num_rows, int dim, int num_iters)
{
    printf("\n======== Mode: %s ========\n", ctx->mode_name);

    /* Verify data (skip for noncache/hugepage — separate objects, data uninitialized) */
    bench_read_row(ctx, 42, dim);
    float *verify_buf = (ctx->mode == MODE_URMA)
        ? (float *)urma_rw_get_buffer(ctx->urma_ctx)
        : ctx->local_buf;
    float expected = 42.0f * dim * 0.001f;
    float actual = verify_buf[0];
    if (ctx->mode == MODE_UBSMEM_NC || ctx->mode == MODE_UBSMEM_HUGE) {
        printf("  Verify: skipped (separate shmem object, data may differ)\n");
    } else {
        printf("  Verify row[42][0]: got=%.4f expect=%.4f %s\n",
               actual, expected, fabsf(actual - expected) < 0.01f ? "OK" : "MISMATCH");
    }

    bench_single_load(ctx, num_rows, dim, num_iters);
    bench_single_row(ctx, num_rows, dim, num_iters);
    bench_batch(ctx, num_rows, dim, num_iters);
    bench_engram_prefetch(ctx, num_rows, dim, num_iters);
    bench_paper_27b(ctx, (size_t)num_rows * dim * sizeof(float), num_iters);
    bench_cold_hot(ctx, num_rows, dim);
}

/* ------------------------------------------------------------------ */
/*  Setup helpers                                                     */
/* ------------------------------------------------------------------ */

static const float *setup_local(size_t buf_size)
{
    float *buf = (float *)malloc(buf_size);
    if (!buf) return NULL;
    size_t n = buf_size / sizeof(float);
    for (size_t i = 0; i < n; i++) buf[i] = (float)i * 0.001f;
    printf("  Local DRAM buffer: %zu MB\n", buf_size / (1024*1024));
    return buf;
}

/*
 * Setup ubs_mem mapping.
 *
 * NONCACHE / HUGETLB flags are set at allocate time, not map time.
 * The server creates the default CACHE object. For noncache/hugepage,
 * we create a separate shmem object with a suffixed name and the
 * desired flags via allocate_with_provider.
 *
 * For CACHE mode we reuse the server's existing object.
 */
static const float *setup_ubsmem(const char *shm_name, size_t buf_size,
                                  const char *provider_host, uint64_t flags,
                                  void **out_ptr, char *used_name, size_t name_len)
{
    /* Build the actual shmem name: base name + suffix for non-default flags */
    if (flags == UBSM_FLAG_CACHE) {
        snprintf(used_name, name_len, "%s", shm_name);
    } else if (flags == UBSM_FLAG_NONCACHE) {
        snprintf(used_name, name_len, "%s_nc", shm_name);
    } else if (flags == UBSM_FLAG_MMAP_HUGETLB_PMD) {
        snprintf(used_name, name_len, "%s_huge", shm_name);
    } else {
        snprintf(used_name, name_len, "%s_0x%lx", shm_name, (unsigned long)flags);
    }

    ubsmem_shmem_info_t info;
    int ret = ubsmem_shmem_lookup(used_name, &info);
    if (ret != 0) {
        /* Need to create — use allocate_with_provider for cross-node */
        ubs_mem_provider_t prov;
        memset(&prov, 0, sizeof(prov));
        snprintf(prov.host_name, sizeof(prov.host_name), "%s", provider_host);
        prov.socket_id = UINT32_MAX;
        prov.numa_id = UINT32_MAX;
        prov.port_id = UINT32_MAX;
        ret = ubsmem_shmem_allocate_with_provider(&prov, used_name, buf_size,
                                                   0666, flags);
        if (ret != 0 && ret != UBSM_ERR_ALREADY_EXIST) {
            fprintf(stderr, "  ubsmem allocate_with_provider(%s, flags=0x%lx) failed: %d\n",
                    used_name, (unsigned long)flags, ret);
        }
    }

    void *ptr = NULL;
    ret = ubsmem_shmem_map(NULL, buf_size, PROT_READ, MAP_SHARED, used_name, 0, &ptr);
    if (ret != 0 || !ptr) {
        fprintf(stderr, "  ubsmem_shmem_map(%s) failed: %d\n", used_name, ret);
        return NULL;
    }
    printf("  ubs_mem mapped: name=%s, ptr=%p, flags=0x%lx\n",
           used_name, ptr, (unsigned long)flags);
    *out_ptr = ptr;
    return (const float *)ptr;
}

static urma_rw_ctx_t *setup_urma(const char *server_ip, uint16_t port,
                                  size_t local_buf_size)
{
    urma_rw_ctx_t *ctx = urma_rw_init(NULL, local_buf_size);
    if (!ctx) {
        fprintf(stderr, "  urma_rw_init failed\n");
        return NULL;
    }
    printf("  Connecting to URMA server %s:%u...\n", server_ip, port);
    if (urma_rw_client_connect(ctx, server_ip, port) != 0) {
        fprintf(stderr, "  urma_rw_client_connect failed\n");
        urma_rw_destroy(ctx);
        return NULL;
    }
    printf("  URMA connected\n");
    return ctx;
}

/* ------------------------------------------------------------------ */
/*  Mode matching helper                                              */
/* ------------------------------------------------------------------ */

static bool mode_match(const char *mode_str, const char *target)
{
    return strcmp(mode_str, target) == 0 || strcmp(mode_str, "all") == 0;
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    /* Defaults */
    size_t size_mb = 128;
    const char *shm_name = "engram_test";
    int num_rows = 10000;
    int dim = 341;
    int num_iters = 500;
    const char *server_ip = "192.168.84.245";
    uint16_t tcp_port = DEFAULT_TCP_PORT;
    uint16_t urma_port = DEFAULT_URMA_PORT;
    const char *provider_host = "node1";
    const char *mode_str = "all";

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <mode> [options]\n"
                "  mode: local, tcp, urma, ubsmem, ubsmem-nc, ubsmem-huge, all\n"
                "  --size_mb N  --name NAME  --rows N  --dim N  --iters N\n"
                "  --server_ip IP  --tcp_port N  --urma_port N  --provider HOST\n",
                argv[0]);
        return 1;
    }
    mode_str = argv[1];

    for (int i = 2; i < argc; i++) {
        if (i + 1 >= argc) break;  /* all flags need a value */
        if (strcmp(argv[i], "--size_mb") == 0)       size_mb = (size_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--name") == 0)      shm_name = argv[++i];
        else if (strcmp(argv[i], "--rows") == 0)      num_rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dim") == 0)       dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0)     num_iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--server_ip") == 0) server_ip = argv[++i];
        else if (strcmp(argv[i], "--tcp_port") == 0)  tcp_port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--urma_port") == 0) urma_port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--provider") == 0)  provider_host = argv[++i];
    }

    size_t buf_size = size_mb * 1024 * 1024;
    int row_bytes = dim * (int)sizeof(float);
    size_t local_buf_size = (size_t)NUM_TABLES * 256 * row_bytes;
    if (local_buf_size < buf_size) local_buf_size = buf_size;

    printf("=== Unified Cross-Node Benchmark ===\n");
    printf("Mode: %s\n", mode_str);
    printf("Table: %d rows x %d dim = %d bytes/row\n", num_rows, dim, row_bytes);
    printf("shmem: %s, size: %zu MB, iters: %d\n", shm_name, size_mb, num_iters);
    printf("server: %s, tcp:%u, urma:%u, provider: %s\n\n",
           server_ip, tcp_port, urma_port, provider_host);

    float *local_buf = (float *)malloc(local_buf_size);
    if (!local_buf) { perror("malloc local_buf"); return 1; }

    /* Initialize ubs_mem library if needed */
    bool ubsmem_inited = false;
    if (mode_match(mode_str, "ubsmem") || mode_match(mode_str, "ubsmem-nc") ||
        mode_match(mode_str, "ubsmem-huge")) {
        ubsmem_options_t opts;
        if (ubsmem_init_attributes(&opts) == 0 && ubsmem_initialize(&opts) == 0) {
            ubsmem_set_logger_level(2);
            ubsmem_inited = true;
            printf("[init] ubs_mem library OK\n");
        } else {
            fprintf(stderr, "[init] ubs_mem library FAILED\n");
        }
    }

    /* ---- LOCAL ---- */
    if (mode_match(mode_str, "local")) {
        const float *ldata = setup_local(buf_size);
        if (ldata) {
            bench_ctx_t ctx = { .mode = MODE_LOCAL, .mode_name = "LOCAL DRAM",
                                .data_ptr = ldata, .local_buf = local_buf,
                                .local_buf_size = local_buf_size };
            run_benchmarks(&ctx, num_rows, dim, num_iters);
            free((void *)ldata);
        }
    }

    /* ---- TCP ---- */
    if (mode_match(mode_str, "tcp")) {
        int tcp_fd = tcp_connect(server_ip, tcp_port);
        if (tcp_fd >= 0) {
            bench_ctx_t ctx = { .mode = MODE_TCP, .mode_name = "TCP",
                                .tcp_fd = tcp_fd, .local_buf = local_buf,
                                .local_buf_size = local_buf_size };
            run_benchmarks(&ctx, num_rows, dim, num_iters);
            tcp_disconnect(tcp_fd);
        } else {
            fprintf(stderr, "TCP connect to %s:%u failed\n", server_ip, tcp_port);
        }
    }

    /* ---- URMA (RDMA READ) ---- */
    if (mode_match(mode_str, "urma")) {
        urma_rw_ctx_t *uctx = setup_urma(server_ip, urma_port, local_buf_size);
        if (uctx) {
            bench_ctx_t ctx = { .mode = MODE_URMA,
                                .mode_name = "URMA (RDMA READ)",
                                .urma_ctx = uctx, .local_buf = local_buf,
                                .local_buf_size = local_buf_size };
            run_benchmarks(&ctx, num_rows, dim, num_iters);
            urma_rw_destroy(uctx);
        }
    }

    /* ---- UBSMEM (CACHE) ---- */
    if (mode_match(mode_str, "ubsmem") && ubsmem_inited) {
        void *uptr = NULL;
        char uname[MAX_SHM_NAME_LENGTH + 1];
        const float *udata = setup_ubsmem(shm_name, buf_size, provider_host,
                                           UBSM_FLAG_CACHE, &uptr, uname, sizeof(uname));
        if (udata) {
            bench_ctx_t ctx = { .mode = MODE_UBSMEM,
                                .mode_name = "UBS-MEM (cache)",
                                .data_ptr = udata, .local_buf = local_buf,
                                .local_buf_size = local_buf_size };
            run_benchmarks(&ctx, num_rows, dim, num_iters);
            ubsmem_shmem_unmap(uptr, buf_size);
        }
    }

    /* ---- UBSMEM NONCACHE ---- */
    if (mode_match(mode_str, "ubsmem-nc") && ubsmem_inited) {
        void *uptr = NULL;
        char uname[MAX_SHM_NAME_LENGTH + 1];
        const float *udata = setup_ubsmem(shm_name, buf_size, provider_host,
                                           UBSM_FLAG_NONCACHE, &uptr, uname, sizeof(uname));
        if (udata) {
            bench_ctx_t ctx = { .mode = MODE_UBSMEM_NC,
                                .mode_name = "UBS-MEM (noncache)",
                                .data_ptr = udata, .local_buf = local_buf,
                                .local_buf_size = local_buf_size };
            run_benchmarks(&ctx, num_rows, dim, num_iters);
            ubsmem_shmem_unmap(uptr, buf_size);
        }
    }

    /* ---- UBSMEM HUGEPAGE ---- */
    if (mode_match(mode_str, "ubsmem-huge") && ubsmem_inited) {
        void *uptr = NULL;
        char uname[MAX_SHM_NAME_LENGTH + 1];
        const float *udata = setup_ubsmem(shm_name, buf_size, provider_host,
                                           UBSM_FLAG_MMAP_HUGETLB_PMD, &uptr, uname, sizeof(uname));
        if (udata) {
            bench_ctx_t ctx = { .mode = MODE_UBSMEM_HUGE,
                                .mode_name = "UBS-MEM (2MB hugepage)",
                                .data_ptr = udata, .local_buf = local_buf,
                                .local_buf_size = local_buf_size };
            run_benchmarks(&ctx, num_rows, dim, num_iters);
            ubsmem_shmem_unmap(uptr, buf_size);
        }
    }

    /* Cleanup */
    free(local_buf);
    if (ubsmem_inited) ubsmem_finalize();

    printf("\n=== Benchmark complete ===\n");
    return 0;
}
