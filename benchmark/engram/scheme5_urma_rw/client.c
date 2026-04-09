/**
 * Client: connects to server, imports remote segment, runs urma_read benchmarks.
 *
 * Usage:
 *   ./client <server_ip> [port] [num_rows] [dim]
 *
 * Example:
 *   ./client 192.168.84.245 13857 10000 341
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "urma_rw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

/* Max batch size used by bench_batch_read. Must be >= max(batches[]) below.
 * Kept as a static cap so fixed-size stack arrays below are sized correctly. */
#define MAX_BENCH_BATCH 256

static double diff_us(struct timespec* a, struct timespec* b)
{
    return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

static void bench_single_read(urma_rw_ctx_t* ctx, int num_rows, int dim,
                              int num_iters)
{
    int row_bytes = dim * sizeof(float);
    printf("\n[Bench 1] Single row read latency (1 row = %d bytes)\n", row_bytes);

    struct timespec t0, t1;
    double total_us = 0;
    double min_us = 1e9, max_us = 0;

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        urma_rw_read(ctx, 0, 0, row_bytes);
    }

    for (int i = 0; i < num_iters; i++) {
        int row = rand() % num_rows;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (urma_rw_read(ctx, 0, (uint64_t)row * row_bytes, row_bytes) != 0) {
            fprintf(stderr, "read failed\n");
            return;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = diff_us(&t0, &t1);
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }

    double avg = total_us / num_iters;
    printf("  Avg: %.2f us, Min: %.2f us, Max: %.2f us\n", avg, min_us, max_us);
    printf("  Throughput (single-row): %.1f MB/s\n",
           row_bytes / avg);
}

static void bench_batch_read(urma_rw_ctx_t* ctx, int num_rows, int dim,
                              int num_iters)
{
    int row_bytes = dim * sizeof(float);
    int batches[] = {32, 64, 128, 256};

    printf("\n[Bench 2] Batch read latency (varying batch size)\n");
    printf("  %-8s %-10s %-10s %-12s\n", "Batch", "Data", "Avg us", "Throughput");
    printf("  --------------------------------------------\n");

    for (int b = 0; b < 4; b++) {
        int batch = batches[b];
        if (batch > MAX_BENCH_BATCH) {
            fprintf(stderr, "batch %d exceeds MAX_BENCH_BATCH=%d, skipping\n",
                    batch, MAX_BENCH_BATCH);
            continue;
        }
        uint64_t locals[MAX_BENCH_BATCH];
        uint64_t remotes[MAX_BENCH_BATCH];
        uint32_t lens[MAX_BENCH_BATCH];
        for (int i = 0; i < batch; i++) lens[i] = row_bytes;

        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            for (int i = 0; i < batch; i++) {
                int row = rand() % num_rows;
                locals[i] = (uint64_t)i * row_bytes;
                remotes[i] = (uint64_t)row * row_bytes;
            }
            clock_gettime(CLOCK_MONOTONIC, &t0);
            urma_rw_read_batch(ctx, locals, remotes, lens, batch);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg = total_us / num_iters;
        double data_kb = (double)batch * row_bytes / 1024.0;
        double throughput = (data_kb / 1024.0) / (avg / 1e6);
        printf("  %-8d %-8.1fKB %-9.1f %.1f MB/s\n",
               batch, data_kb, avg, throughput);
    }
}

static void bench_engram_prefetch(urma_rw_ctx_t* ctx, int num_rows, int dim,
                                   int num_iters)
{
    int row_bytes = dim * sizeof(float);
    int NUM_TABLES = 12;
    int batch_sizes[] = {32, 64, 128, 256};

    printf("\n[Bench 3] Full Engram prefetch (%d tables x N tokens)\n", NUM_TABLES);
    printf("  %-8s %-10s %-10s %-12s\n", "Tokens", "Data", "Avg ms", "Throughput");
    printf("  --------------------------------------------\n");

    /* Heap-allocated, reused across batch sizes */
    uint64_t* locals = malloc(URMA_RW_MAX_BATCH * sizeof(uint64_t));
    uint64_t* remotes = malloc(URMA_RW_MAX_BATCH * sizeof(uint64_t));
    uint32_t* lens = malloc(URMA_RW_MAX_BATCH * sizeof(uint32_t));
    if (!locals || !remotes || !lens) {
        fprintf(stderr, "malloc failed\n");
        goto cleanup;
    }

    for (int b = 0; b < 4; b++) {
        int tokens = batch_sizes[b];
        int total_reads = NUM_TABLES * tokens;
        if (total_reads > URMA_RW_MAX_BATCH) {
            printf("  %-8d  (skipped: %d > URMA_RW_MAX_BATCH=%d)\n",
                   tokens, total_reads, URMA_RW_MAX_BATCH);
            continue;
        }

        for (int i = 0; i < total_reads; i++) lens[i] = row_bytes;

        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            for (int i = 0; i < total_reads; i++) {
                int row = rand() % num_rows;
                locals[i] = (uint64_t)i * row_bytes;
                remotes[i] = (uint64_t)row * row_bytes;
            }
            clock_gettime(CLOCK_MONOTONIC, &t0);
            if (urma_rw_read_batch(ctx, locals, remotes, lens, total_reads) != 0) {
                fprintf(stderr, "batch read failed at tokens=%d\n", tokens);
                break;
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg_ms = total_us / num_iters / 1000.0;
        double data_kb = (double)total_reads * row_bytes / 1024.0;
        double throughput = (data_kb / 1024.0) / (avg_ms / 1e3);
        printf("  %-8d %-8.1fKB %-9.2f %.1f MB/s\n",
               tokens, data_kb, avg_ms, throughput);
    }

cleanup:
    free(locals);
    free(remotes);
    free(lens);
}

static void verify_read(urma_rw_ctx_t* ctx, int dim)
{
    int row_bytes = dim * sizeof(float);
    printf("\n[Verify] Reading and checking data...\n");

    /* Read row 42 */
    if (urma_rw_read(ctx, 0, (uint64_t)42 * row_bytes, row_bytes) != 0) {
        printf("  FAILED\n");
        return;
    }
    float* local = (float*)urma_rw_get_buffer(ctx);
    printf("  row[42][0..3] = %.6f, %.6f, %.6f, %.6f\n",
           local[0], local[1], local[2], local[3]);
    printf("  Expected:       %.6f, %.6f, %.6f, %.6f (if server filled pattern)\n",
           (42 * dim + 0) * 0.001f,
           (42 * dim + 1) * 0.001f,
           (42 * dim + 2) * 0.001f,
           (42 * dim + 3) * 0.001f);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <server_ip> [port] [num_rows] [dim] [num_iters]\n", argv[0]);
        return 1;
    }

    const char* server_ip = argv[1];
    uint16_t port = (argc > 2) ? (uint16_t)atoi(argv[2]) : URMA_RW_DEFAULT_PORT;
    int num_rows = (argc > 3) ? atoi(argv[3]) : 10000;
    int dim = (argc > 4) ? atoi(argv[4]) : 341;
    int num_iters = (argc > 5) ? atoi(argv[5]) : 200;

    /* Local buffer: enough for the biggest batch (URMA_RW_MAX_BATCH rows).
     * With URMA_RW_MAX_BATCH=4096 and dim=341: ~5.5 MB minimum.
     * Use 64 MB for headroom. */
    uint64_t local_size = (uint64_t)URMA_RW_MAX_BATCH * dim * sizeof(float);
    if (local_size < 64 * 1024 * 1024) local_size = 64 * 1024 * 1024;

    printf("=== URMA RW Client ===\n");
    printf("Server: %s:%u\n", server_ip, port);
    printf("Table: %d rows x %d dim = %.1f MB\n",
           num_rows, dim, (double)num_rows * dim * 4 / 1e6);
    printf("Local buf: %.1f MB, num_iters: %d\n",
           (double)local_size / 1e6, num_iters);

    urma_rw_ctx_t* ctx = urma_rw_init(NULL, local_size);
    if (!ctx) {
        fprintf(stderr, "urma_rw_init failed\n");
        return 1;
    }

    if (urma_rw_client_connect(ctx, server_ip, port) != 0) {
        fprintf(stderr, "connect failed\n");
        urma_rw_destroy(ctx);
        return 1;
    }

    srand(time(NULL));

    verify_read(ctx, dim);
    bench_single_read(ctx, num_rows, dim, num_iters);
    bench_batch_read(ctx, num_rows, dim, num_iters);
    bench_engram_prefetch(ctx, num_rows, dim, num_iters);

    urma_rw_destroy(ctx);
    printf("\nDone.\n");
    return 0;
}
