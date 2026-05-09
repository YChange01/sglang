/**
 * URMA RDMA READ benchmark.
 *
 * Measures one-sided URMA READ against the Engram benchmark server. This is a
 * completion-based read path: every measured read waits until local data is
 * available in the registered client buffer.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../lib/urma_rw.h"

#define DEFAULT_URMA_PORT 13857
#define PAPER_SEGS_PER_TOKEN 8
#define PAPER_SEG_BYTES 320

typedef struct {
    size_t size_mb;
    int rows;
    int dim;
    int iters;
    const char *server_ip;
    const char *urma_dev;
    uint16_t urma_port;
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

static inline uint64_t rdcntfrq(void) { return 1000000000ULL; }
#endif

static int cmp_u64(const void *a, const void *b)
{
    uint64_t av = *(const uint64_t *)a;
    uint64_t bv = *(const uint64_t *)b;
    return (av > bv) - (av < bv);
}

static uint64_t next_rand64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static uint64_t random_segment_offset(uint64_t *state, uint64_t data_size,
                                      uint32_t len, uint32_t align)
{
    uint64_t max_offset = data_size > len ? data_size - len : 0;
    uint64_t slots = max_offset / align;
    if (slots == 0) return 0;
    return (next_rand64(state) % slots) * align;
}

static void print_ns_stats(const char *label, uint64_t *samples, int count,
                           uint64_t counter_freq)
{
    if (count <= 0) return;
    long double sum = 0.0;
    uint64_t min = UINT64_MAX;
    uint64_t max = 0;
    for (int i = 0; i < count; i++) {
        uint64_t v = samples[i];
        sum += (long double)v;
        if (v < min) min = v;
        if (v > max) max = v;
    }

    qsort(samples, (size_t)count, sizeof(uint64_t), cmp_u64);
    double ns_per_tick = 1e9 / (double)counter_freq;
    double avg_ns = (double)(sum / count) * ns_per_tick;
    double min_ns = (double)min * ns_per_tick;
    double p50_ns = (double)samples[count / 2] * ns_per_tick;
    double p99_ns = (double)samples[(count * 99) / 100] * ns_per_tick;
    double max_ns = (double)max * ns_per_tick;
    printf("    %s avg=%.1f ns min=%.1f ns p50=%.1f ns p99=%.1f ns max=%.1f ns\n",
           label, avg_ns, min_ns, p50_ns, p99_ns, max_ns);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --size_mb N    remote buffer size in MB (default 128)\n"
            "  --rows N       number of rows (default 10000)\n"
            "  --dim N        embedding dimension (default 341)\n"
            "  --iters N      iterations per benchmark (default 500)\n"
            "  --server_ip IP URMA server IP (default 192.168.84.245)\n"
            "  --urma_port N  URMA exchange port (default 13857)\n"
            "  --dev_name DEV URMA device name, overrides URMA_DEV env\n"
            "  --urma_dev DEV alias for --dev_name\n",
            prog);
}

static int parse_opts(int argc, char **argv, bench_opts_t *opts)
{
    opts->size_mb = 128;
    opts->rows = 10000;
    opts->dim = 341;
    opts->iters = 500;
    opts->server_ip = "192.168.84.245";
    opts->urma_dev = getenv("URMA_DEV");
    opts->urma_port = DEFAULT_URMA_PORT;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(0);
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for option: %s\n", argv[i]);
            return -1;
        }
        if (strcmp(argv[i], "--size_mb") == 0) opts->size_mb = (size_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--rows") == 0) opts->rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dim") == 0) opts->dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0) opts->iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--server_ip") == 0) opts->server_ip = argv[++i];
        else if (strcmp(argv[i], "--urma_port") == 0) opts->urma_port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--dev_name") == 0 ||
                 strcmp(argv[i], "--urma_dev") == 0) opts->urma_dev = argv[++i];
        else if (strcmp(argv[i], "--name") == 0 ||
                 strcmp(argv[i], "--tcp_port") == 0 ||
                 strcmp(argv[i], "--provider") == 0) {
            i++;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

static urma_rw_ctx_t *setup_urma(const char *server_ip, uint16_t port,
                                 const char *dev_name, size_t local_buf_size)
{
    urma_rw_ctx_t *ctx = urma_rw_init(dev_name, local_buf_size);
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

static int read_row(urma_rw_ctx_t *ctx, int row, int dim)
{
    uint32_t row_bytes = (uint32_t)(dim * (int)sizeof(float));
    return urma_rw_read(ctx, 0, (uint64_t)row * row_bytes, row_bytes);
}

static int read_segments(urma_rw_ctx_t *ctx, const uint64_t *offsets, int count)
{
    static uint64_t local_offsets[URMA_RW_MAX_BATCH];
    static uint64_t remote_offsets[URMA_RW_MAX_BATCH];
    static uint32_t lens[URMA_RW_MAX_BATCH];

    int remaining = count;
    int done = 0;
    while (remaining > 0) {
        int chunk = remaining > URMA_RW_MAX_BATCH ? URMA_RW_MAX_BATCH : remaining;
        for (int i = 0; i < chunk; i++) {
            local_offsets[i] = (uint64_t)(done + i) * PAPER_SEG_BYTES;
            remote_offsets[i] = offsets[done + i];
            lens[i] = PAPER_SEG_BYTES;
        }
        if (urma_rw_read_batch(ctx, local_offsets, remote_offsets, lens,
                               (uint32_t)chunk) != 0) {
            return -1;
        }
        done += chunk;
        remaining -= chunk;
    }
    return 0;
}

static void bench_single_seg(urma_rw_ctx_t *ctx, size_t data_size, int iters)
{
    printf("\n  [Single float load] (skipped for URMA READ)\n");
    printf("\n  [Single-seg read] %d bytes, %d iters\n", PAPER_SEG_BYTES, iters);

    uint64_t max_offset = data_size - PAPER_SEG_BYTES;
    uint64_t off0 = 0;
    for (int i = 0; i < 10; i++) {
        if (read_segments(ctx, &off0, 1) != 0) {
            fprintf(stderr, "    warmup read failed\n");
            return;
        }
        off0 = (off0 + PAPER_SEG_BYTES) % max_offset;
    }

    uint64_t *offsets = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    uint64_t *samples = (uint64_t *)calloc((size_t)iters, sizeof(uint64_t));
    if (!offsets || !samples) {
        free(offsets);
        free(samples);
        return;
    }
    uint64_t rng = 42;
    for (int i = 0; i < iters; i++) {
        offsets[i] = random_segment_offset(&rng, data_size, PAPER_SEG_BYTES,
                                            PAPER_SEG_BYTES);
    }

    int ok_iters = 0;
    uint64_t frq = rdcntfrq();
    if (frq == 0) frq = 1;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = rdcntvct();
        if (read_segments(ctx, &offsets[i], 1) != 0) {
            fprintf(stderr, "    URMA segment read failed\n");
            break;
        }
        uint64_t t1 = rdcntvct();
        samples[ok_iters] = t1 - t0;
        ok_iters++;
    }

    if (ok_iters > 0) {
        long double sum = 0.0;
        for (int i = 0; i < ok_iters; i++) sum += (long double)samples[i];
        double avg_ns = (double)(sum / ok_iters) * 1e9 / (double)frq;
        print_ns_stats("Latency:", samples, ok_iters, frq);
        printf("    Throughput: %.1f MB/s\n",
               (double)PAPER_SEG_BYTES / (avg_ns / 1e9) / (1024.0 * 1024.0));
    }
    free(samples);
    free(offsets);
}

static void bench_engram_27b(urma_rw_ctx_t *ctx, size_t data_size, int iters)
{
    int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                         1024, 2048, 4096, 8192, 16384};
    int nbatches = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));

    printf("\n  [Engram-27B] 8 segs x 320B per token, sparse\n");
    printf("    %-8s %-8s %-10s %-12s %-12s\n",
           "Batch", "Reads", "Data", "Latency", "Throughput");
    printf("    -----------------------------------------------------------\n");

    for (int b = 0; b < nbatches; b++) {
        int batch = batch_sizes[b];
        int total_segs = batch * PAPER_SEGS_PER_TOKEN;
        uint64_t *offsets = (uint64_t *)malloc((size_t)total_segs * sizeof(uint64_t));
        if (!offsets) continue;

        uint64_t frq = rdcntfrq();
        if (frq == 0) frq = 1;
        long double total_ticks = 0.0;
        int ok_iters = 0;
        uint64_t rng = 0x9e3779b97f4a7c15ULL ^ (uint64_t)batch;
        for (int iter = 0; iter < iters; iter++) {
            for (int i = 0; i < total_segs; i++) {
                offsets[i] = random_segment_offset(&rng, data_size,
                                                    PAPER_SEG_BYTES,
                                                    PAPER_SEG_BYTES);
            }
            uint64_t t0 = rdcntvct();
            if (read_segments(ctx, offsets, total_segs) != 0) {
                fprintf(stderr, "    URMA batch read failed\n");
                break;
            }
            uint64_t t1 = rdcntvct();
            total_ticks += (long double)(t1 - t0);
            ok_iters++;
        }

        if (ok_iters == 0) {
            free(offsets);
            break;
        }

        double avg_us = (double)(total_ticks / ok_iters) * 1e6 / (double)frq;
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

static void bench_urma_fast(urma_rw_ctx_t *ctx, size_t data_size, int iters)
{
    const uint32_t len = PAPER_SEG_BYTES;
    uint64_t frq = rdcntfrq();
    if (frq == 0) frq = 1;

    uint64_t *offsets = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    if (!offsets) return;
    uint64_t rng = 4242;
    for (int i = 0; i < iters; i++) {
        offsets[i] = random_segment_offset(&rng, data_size, len, len);
    }

    for (int i = 0; i < 100; i++) {
        urma_rw_read(ctx, 0, offsets[i % iters], len);
        urma_rw_read_fast(ctx, 0, offsets[i % iters], len);
    }

    uint64_t *samples_old = (uint64_t *)calloc((size_t)iters, sizeof(uint64_t));
    uint64_t *samples_new = (uint64_t *)calloc((size_t)iters, sizeof(uint64_t));
    if (!samples_old || !samples_new) {
        free(samples_old);
        free(samples_new);
        free(offsets);
        return;
    }
    uint64_t total_old = 0, min_old = UINT64_MAX, max_old = 0;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = rdcntvct();
        urma_rw_read(ctx, 0, offsets[i], len);
        uint64_t t1 = rdcntvct();
        uint64_t d = t1 - t0;
        samples_old[i] = d;
        total_old += d;
        if (d < min_old) min_old = d;
        if (d > max_old) max_old = d;
    }

    uint64_t total_new = 0, min_new = UINT64_MAX, max_new = 0;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = rdcntvct();
        urma_rw_read_fast(ctx, 0, offsets[i], len);
        uint64_t t1 = rdcntvct();
        uint64_t d = t1 - t0;
        samples_new[i] = d;
        total_new += d;
        if (d < min_new) min_new = d;
        if (d > max_new) max_new = d;
    }

    double ns_per_tick = 1e9 / (double)frq;
    double avg_old_ns = (double)total_old / iters * ns_per_tick;
    double avg_new_ns = (double)total_new / iters * ns_per_tick;
    double min_old_ns = (double)min_old * ns_per_tick;
    double min_new_ns = (double)min_new * ns_per_tick;
    double max_old_ns = (double)max_old * ns_per_tick;
    double max_new_ns = (double)max_new * ns_per_tick;

    printf("\n  [URMA fast A/B] 320B read, %d iters, CNTVCT (freq=%lu Hz)\n",
           iters, (unsigned long)frq);
    qsort(samples_old, (size_t)iters, sizeof(uint64_t), cmp_u64);
    qsort(samples_new, (size_t)iters, sizeof(uint64_t), cmp_u64);
    printf("    %-20s %-12s %-12s %-12s %-12s %-12s\n",
           "path", "avg(ns)", "min(ns)", "p50(ns)", "p99(ns)", "max(ns)");
    printf("    -------------------------------------------------------------------------\n");
    printf("    %-20s %-12.1f %-12.1f %-12.1f %-12.1f %-12.1f\n",
           "urma_rw_read", avg_old_ns, min_old_ns,
           (double)samples_old[iters / 2] * ns_per_tick,
           (double)samples_old[(iters * 99) / 100] * ns_per_tick,
           max_old_ns);
    printf("    %-20s %-12.1f %-12.1f %-12.1f %-12.1f %-12.1f\n",
           "urma_rw_read_fast", avg_new_ns, min_new_ns,
           (double)samples_new[iters / 2] * ns_per_tick,
           (double)samples_new[(iters * 99) / 100] * ns_per_tick,
           max_new_ns);
    printf("    savings: %.1f ns (%.1f%%)\n",
           avg_old_ns - avg_new_ns,
           avg_old_ns > 0.0 ? (avg_old_ns - avg_new_ns) / avg_old_ns * 100.0 : 0.0);

    free(samples_old);
    free(samples_new);
    free(offsets);
}

int main(int argc, char **argv)
{
    bench_opts_t opts;
    if (parse_opts(argc, argv, &opts) != 0) return 1;

    size_t data_size = (size_t)opts.rows * opts.dim * sizeof(float);
    size_t buf_size = opts.size_mb * 1024ULL * 1024ULL;
    size_t local_buf_size = (size_t)16384 * PAPER_SEGS_PER_TOKEN * PAPER_SEG_BYTES;
    if (local_buf_size < buf_size) local_buf_size = buf_size;

    printf("=== URMA RDMA READ Benchmark ===\n");
    printf("Table: %d rows x %d dim = %d bytes/row\n",
           opts.rows, opts.dim, (int)(opts.dim * sizeof(float)));
    printf("remote buffer: %zu MB, iters: %d\n", opts.size_mb, opts.iters);
    printf("server: %s:%u, dev: %s\n",
           opts.server_ip, opts.urma_port,
           opts.urma_dev ? opts.urma_dev : "(auto)");

    urma_rw_ctx_t *ctx = setup_urma(opts.server_ip, opts.urma_port,
                                    opts.urma_dev, local_buf_size);
    if (!ctx) return 1;

    if (read_row(ctx, 42, opts.dim) == 0) {
        float *verify_buf = (float *)urma_rw_get_buffer(ctx);
        float expected = 42.0f * opts.dim * 0.001f;
        printf("  Verify row[42][0]: got=%.4f expect=%.4f %s\n",
               verify_buf[0], expected,
               fabsf(verify_buf[0] - expected) < 0.01f ? "OK" : "MISMATCH");
    } else {
        fprintf(stderr, "  Verify read failed\n");
    }

    bench_single_seg(ctx, data_size, opts.iters);
    bench_engram_27b(ctx, data_size, opts.iters);
    printf("\n  [Cold/Hot analysis] (skipped for URMA READ)\n");
    bench_urma_fast(ctx, data_size, opts.iters);

    urma_rw_destroy(ctx);
    printf("\n=== Benchmark complete ===\n");
    return 0;
}
