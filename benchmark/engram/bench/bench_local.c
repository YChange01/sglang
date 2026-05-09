/**
 * Local DRAM read benchmark.
 *
 * This is the standalone local baseline for Engram cross-node experiments.
 * It keeps the same measurement shape as the remote schemes:
 *   - single float load
 *   - single sparse 320B segment read
 *   - Engram-27B batch pattern: 8 x 320B sparse segments per token
 *   - cold/hot row read check
 */

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define PAPER_SEGS_PER_TOKEN 8
#define PAPER_SEG_BYTES 320

typedef struct {
    size_t size_mb;
    int rows;
    int dim;
    int iters;
} bench_opts_t;

static double diff_us(const struct timespec *a, const struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

static double diff_ns(const struct timespec *a, const struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e9 + (b->tv_nsec - a->tv_nsec);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --size_mb N  buffer size in MB (default 128)\n"
            "  --rows N     number of rows (default 10000)\n"
            "  --dim N      embedding dimension (default 341)\n"
            "  --iters N    iterations per benchmark (default 500)\n",
            prog);
}

static int parse_opts(int argc, char **argv, bench_opts_t *opts)
{
    opts->size_mb = 128;
    opts->rows = 10000;
    opts->dim = 341;
    opts->iters = 500;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            exit(0);
        }
        if (i + 1 >= argc) break;
        if (strcmp(argv[i], "--size_mb") == 0) opts->size_mb = (size_t)atol(argv[++i]);
        else if (strcmp(argv[i], "--rows") == 0) opts->rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dim") == 0) opts->dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0) opts->iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--name") == 0 ||
                 strcmp(argv[i], "--server_ip") == 0 ||
                 strcmp(argv[i], "--tcp_port") == 0 ||
                 strcmp(argv[i], "--urma_port") == 0 ||
                 strcmp(argv[i], "--provider") == 0 ||
                 strcmp(argv[i], "--dev_name") == 0 ||
                 strcmp(argv[i], "--urma_dev") == 0) {
            i++;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

static void fill_local(float *buf, size_t bytes)
{
    size_t n = bytes / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        buf[i] = (float)i * 0.001f;
    }
}

static void read_segments(const float *data, float *local_buf,
                          const uint64_t *offsets, int count)
{
    for (int i = 0; i < count; i++) {
        memcpy((char *)local_buf + (size_t)i * PAPER_SEG_BYTES,
               (const char *)data + offsets[i],
               PAPER_SEG_BYTES);
    }
}

static void bench_single_float(const float *data, int rows, int dim, int iters)
{
    printf("\n  [Single float load] 4 bytes, %d iters\n", iters);

    volatile float sink = 0;
    size_t total_floats = (size_t)rows * dim;
    size_t *indices = (size_t *)malloc((size_t)iters * sizeof(size_t));
    if (!indices) return;

    srand(123);
    size_t rand_limit = total_floats > INT_MAX ? INT_MAX : total_floats;
    for (int i = 0; i < iters; i++) {
        indices[i] = (size_t)(rand() % (int)rand_limit);
    }

    struct timespec t0, t1;
    double total_ns = 0, min_ns = 1e12, max_ns = 0;
    for (int i = 0; i < iters; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        sink = data[indices[i]];
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = diff_ns(&t0, &t1);
        total_ns += ns;
        if (ns < min_ns) min_ns = ns;
        if (ns > max_ns) max_ns = ns;
    }
    printf("    Avg: %.1f ns, Min: %.1f ns, Max: %.1f ns\n",
           total_ns / iters, min_ns, max_ns);
    (void)sink;
    free(indices);
}

static void bench_single_seg(const float *data, float *local_buf,
                             size_t data_size, int iters)
{
    printf("\n  [Single-seg read] %d bytes, %d iters\n", PAPER_SEG_BYTES, iters);

    uint64_t max_offset = data_size - PAPER_SEG_BYTES;
    uint64_t off0 = 0;
    for (int i = 0; i < 10; i++) {
        read_segments(data, local_buf, &off0, 1);
        off0 = (off0 + PAPER_SEG_BYTES) % max_offset;
    }

    uint64_t *offsets = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    if (!offsets) return;
    srand(42);
    for (int i = 0; i < iters; i++) {
        offsets[i] = ((uint64_t)(rand() % (int)(max_offset / PAPER_SEG_BYTES))) *
                     PAPER_SEG_BYTES;
    }

    struct timespec t0, t1;
    double total_us = 0, min_us = 1e9, max_us = 0;
    for (int i = 0; i < iters; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        read_segments(data, local_buf, &offsets[i], 1);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = diff_us(&t0, &t1);
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }

    double avg = total_us / iters;
    printf("    Avg: %.3f us, Min: %.3f us, Max: %.3f us\n", avg, min_us, max_us);
    printf("    Throughput: %.1f MB/s\n", (double)PAPER_SEG_BYTES / avg);
    free(offsets);
}

static void bench_engram_27b(const float *data, float *local_buf,
                             size_t data_size, int iters)
{
    int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                         1024, 2048, 4096, 8192, 16384};
    int nbatches = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));
    uint64_t max_offset = data_size - PAPER_SEG_BYTES;

    printf("\n  [Engram-27B] 8 segs x 320B per token, sparse\n");
    printf("    %-8s %-8s %-10s %-12s %-12s\n",
           "Batch", "Reads", "Data", "Latency", "Throughput");
    printf("    -----------------------------------------------------------\n");

    for (int b = 0; b < nbatches; b++) {
        int batch = batch_sizes[b];
        int total_segs = batch * PAPER_SEGS_PER_TOKEN;
        uint64_t *offsets = (uint64_t *)malloc((size_t)total_segs * sizeof(uint64_t));
        if (!offsets) continue;

        struct timespec t0, t1;
        double total_us = 0;
        for (int iter = 0; iter < iters; iter++) {
            for (int i = 0; i < total_segs; i++) {
                offsets[i] = ((uint64_t)(rand() % (int)(max_offset / PAPER_SEG_BYTES))) *
                             PAPER_SEG_BYTES;
            }
            clock_gettime(CLOCK_MONOTONIC, &t0);
            read_segments(data, local_buf, offsets, total_segs);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg_us = total_us / iters;
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

static void bench_cold_hot(const float *data, float *local_buf, int rows, int dim)
{
    int row_bytes = dim * (int)sizeof(float);
    int cold_rows[] = {0, rows / 4, rows / 2, rows * 3 / 4, rows - 1};
    struct timespec t0, t1;

    printf("\n  [Cold/Hot analysis] first access vs cached\n");
    printf("    %-12s %-12s %-12s\n", "Row", "Cold (us)", "Hot (us)");
    printf("    ------------------------------------\n");

    for (int i = 0; i < 5; i++) {
        int row = cold_rows[i];
        if (row < 0 || row >= rows) continue;

        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(local_buf, data + (size_t)row * dim, row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double cold_us = diff_us(&t0, &t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(local_buf, data + (size_t)row * dim, row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double hot_us = diff_us(&t0, &t1);

        printf("    %-12d %-12.3f %-12.3f\n", row, cold_us, hot_us);
    }
}

int main(int argc, char **argv)
{
    bench_opts_t opts;
    if (parse_opts(argc, argv, &opts) != 0) return 1;

    size_t buf_size = opts.size_mb * 1024ULL * 1024ULL;
    size_t data_size = (size_t)opts.rows * opts.dim * sizeof(float);
    size_t local_buf_size = (size_t)16384 * PAPER_SEGS_PER_TOKEN * PAPER_SEG_BYTES;
    if (local_buf_size < buf_size) local_buf_size = buf_size;

    printf("=== LOCAL DRAM Read Benchmark ===\n");
    printf("Table: %d rows x %d dim = %d bytes/row\n",
           opts.rows, opts.dim, (int)(opts.dim * sizeof(float)));
    printf("buffer: %zu MB, iters: %d\n", opts.size_mb, opts.iters);

    float *data = (float *)malloc(buf_size);
    float *local_buf = (float *)malloc(local_buf_size);
    if (!data || !local_buf) {
        perror("malloc");
        free(data);
        free(local_buf);
        return 1;
    }

    fill_local(data, buf_size);

    memcpy(local_buf, data + (size_t)42 * opts.dim, (size_t)opts.dim * sizeof(float));
    float expected = 42.0f * opts.dim * 0.001f;
    printf("  Verify row[42][0]: got=%.4f expect=%.4f %s\n",
           local_buf[0], expected, fabsf(local_buf[0] - expected) < 0.01f ? "OK" : "MISMATCH");

    bench_single_float(data, opts.rows, opts.dim, opts.iters);
    bench_single_seg(data, local_buf, data_size, opts.iters);
    bench_engram_27b(data, local_buf, data_size, opts.iters);
    bench_cold_hot(data, local_buf, opts.rows, opts.dim);

    free(local_buf);
    free(data);
    printf("\n=== Benchmark complete ===\n");
    return 0;
}
