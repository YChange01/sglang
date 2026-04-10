/**
 * UBS-MEM Smoke Test — Client (reader / benchmarker)
 *
 * Maps the same named shmem object created by server, reads it via
 * plain load/store (pointer dereference), verifies data, and benchmarks.
 *
 * Usage:
 *   ./client [size_mb] [shmem_name] [num_rows] [dim] [num_iters]
 *   Default: size_mb=16, shmem_name="engram_test", num_rows=10000, dim=341
 *
 * Prerequisites:
 *   - Server must be running first (shmem object must exist)
 *   - ubse.service and ubsmd.service running on this node
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <ubs_mem.h>

static double diff_us(struct timespec *a, struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

static double diff_ns(struct timespec *a, struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e9 + (b->tv_nsec - a->tv_nsec);
}

/* ------------------------------------------------------------------ */
/*  Verify correctness                                                */
/* ------------------------------------------------------------------ */

static int verify_data(const float *data, int num_rows, int dim)
{
    printf("\n[Verify] Checking data correctness...\n");
    int errors = 0;

    /* Check a few known rows */
    int check_rows[] = {0, 1, 42, 999};
    for (int r = 0; r < 4; r++) {
        int row = check_rows[r];
        if (row >= num_rows) continue;

        size_t base = (size_t)row * dim;
        float expected0 = (float)base * 0.001f;
        float expected1 = (float)(base + 1) * 0.001f;
        float actual0 = data[base];
        float actual1 = data[base + 1];

        int ok = (fabsf(actual0 - expected0) < 0.01f &&
                  fabsf(actual1 - expected1) < 0.01f);
        printf("  row[%4d]: got (%.4f, %.4f), expect (%.4f, %.4f) %s\n",
               row, actual0, actual1, expected0, expected1,
               ok ? "OK" : "MISMATCH");
        if (!ok) errors++;
    }
    return errors;
}

/* ------------------------------------------------------------------ */
/*  Bench 1: Single-row memcpy latency                               */
/* ------------------------------------------------------------------ */

static void bench_single_read(const float *remote, int num_rows, int dim,
                              int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    float *local = (float *)malloc(row_bytes);
    if (!local) return;

    printf("\n[Bench 1] Single-row load/store latency (1 row = %d bytes)\n",
           row_bytes);

    /* Warmup — touch pages */
    for (int i = 0; i < 10; i++) {
        memcpy(local, remote + (i % num_rows) * dim, row_bytes);
    }

    struct timespec t0, t1;
    double total_us = 0, min_us = 1e9, max_us = 0;

    srand(42);
    for (int i = 0; i < num_iters; i++) {
        int row = rand() % num_rows;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(local, remote + (size_t)row * dim, row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = diff_us(&t0, &t1);
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }

    double avg = total_us / num_iters;
    printf("  Avg: %.3f us, Min: %.3f us, Max: %.3f us\n", avg, min_us, max_us);
    printf("  Throughput: %.1f MB/s\n",
           (double)row_bytes / avg);  /* row_bytes/us = MB/s */
    free(local);
}

/* ------------------------------------------------------------------ */
/*  Bench 2: Batch sequential read (simulate Engram table scan)       */
/* ------------------------------------------------------------------ */

static void bench_batch_read(const float *remote, int num_rows, int dim,
                             int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    int batches[] = {32, 64, 128, 256};

    printf("\n[Bench 2] Batch read latency (sequential memcpy)\n");
    printf("  %-8s %-10s %-12s %-12s %-12s\n",
           "Batch", "Data", "Total us", "Per-row us", "Throughput");
    printf("  -----------------------------------------------------------\n");

    for (int b = 0; b < 4; b++) {
        int batch = batches[b];
        float *local = (float *)malloc((size_t)batch * row_bytes);
        if (!local) continue;

        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < batch; i++) {
                int row = rand() % num_rows;
                memcpy(local + (size_t)i * dim,
                       remote + (size_t)row * dim,
                       row_bytes);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg = total_us / num_iters;
        double per_row = avg / batch;
        double data_kb = (double)batch * row_bytes / 1024.0;
        double throughput = (data_kb / 1024.0) / (avg / 1e6);  /* MB/s */
        printf("  %-8d %-8.1fKB %-10.2f %-10.3f %.1f MB/s\n",
               batch, data_kb, avg, per_row, throughput);
        free(local);
    }
}

/* ------------------------------------------------------------------ */
/*  Bench 3: Full Engram prefetch (12 tables x N tokens)              */
/* ------------------------------------------------------------------ */

static void bench_engram_prefetch(const float *remote, int num_rows, int dim,
                                  int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    int NUM_TABLES = 12;
    int token_counts[] = {32, 64, 128, 256};

    printf("\n[Bench 3] Full Engram prefetch (%d tables x N tokens)\n",
           NUM_TABLES);
    printf("  %-8s %-8s %-10s %-12s %-12s\n",
           "Tokens", "Reads", "Data", "Avg ms", "Throughput");
    printf("  -----------------------------------------------------------\n");

    for (int b = 0; b < 4; b++) {
        int tokens = token_counts[b];
        int total_reads = NUM_TABLES * tokens;

        float *local = (float *)malloc((size_t)total_reads * row_bytes);
        if (!local) continue;

        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < total_reads; i++) {
                int row = rand() % num_rows;
                memcpy(local + (size_t)i * dim,
                       remote + (size_t)row * dim,
                       row_bytes);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg_ms = total_us / num_iters / 1000.0;
        double data_mb = (double)total_reads * row_bytes / (1024.0 * 1024.0);
        double throughput = data_mb / (avg_ms / 1e3);  /* MB/s */
        printf("  %-8d %-8d %-8.1fMB %-10.4f %.1f MB/s\n",
               tokens, total_reads, data_mb, avg_ms, throughput);
        free(local);
    }
}

/* ------------------------------------------------------------------ */
/*  Bench 4: Single float load latency (finest grain)                 */
/* ------------------------------------------------------------------ */

static void bench_single_load(const float *remote, int num_rows, int dim,
                              int num_iters)
{
    printf("\n[Bench 4] Single float load latency (4 bytes)\n");

    /* Warmup */
    volatile float sink = 0;
    for (int i = 0; i < 100; i++) {
        sink += remote[i];
    }

    struct timespec t0, t1;
    double total_ns = 0, min_ns = 1e12, max_ns = 0;
    size_t total_floats = (size_t)num_rows * dim;

    /* Pre-generate random indices to keep rand() out of timed region */
    size_t *indices = (size_t *)malloc(num_iters * sizeof(size_t));
    if (!indices) return;
    srand(123);
    for (int i = 0; i < num_iters; i++)
        indices[i] = (size_t)(rand() % (int)(total_floats > INT_MAX ? INT_MAX : total_floats));

    for (int i = 0; i < num_iters; i++) {
        size_t idx = indices[i];
        clock_gettime(CLOCK_MONOTONIC, &t0);
        sink = remote[idx];
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = diff_ns(&t0, &t1);
        total_ns += ns;
        if (ns < min_ns) min_ns = ns;
        if (ns > max_ns) max_ns = ns;
    }

    double avg = total_ns / num_iters;
    printf("  Avg: %.1f ns, Min: %.1f ns, Max: %.1f ns\n",
           avg, min_ns, max_ns);
    printf("  (clock_gettime overhead ~20-30ns included)\n");
    (void)sink;
    free(indices);
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    size_t size_mb = 16;
    const char *shm_name = "engram_test";
    int num_rows = 10000;
    int dim = 341;
    int num_iters = 500;

    if (argc > 1) size_mb = (size_t)atol(argv[1]);
    if (argc > 2) shm_name = argv[2];
    if (argc > 3) num_rows = atoi(argv[3]);
    if (argc > 4) dim = atoi(argv[4]);
    if (argc > 5) num_iters = atoi(argv[5]);

    size_t buf_size = size_mb * 1024 * 1024;

    printf("=== UBS-MEM Smoke Client ===\n");
    printf("shmem name: %s\n", shm_name);
    printf("size: %zu MB, table: %d rows x %d dim\n", size_mb, num_rows, dim);
    printf("iters: %d\n\n", num_iters);

    /* Step 1: Initialize */
    ubsmem_options_t opts;
    if (ubsmem_init_attributes(&opts) != 0) {
        fprintf(stderr, "ubsmem_init_attributes failed\n");
        return 1;
    }
    if (ubsmem_initialize(&opts) != 0) {
        fprintf(stderr, "ubsmem_initialize failed\n");
        return 1;
    }
    ubsmem_set_logger_level(1);
    printf("[1/3] ubsmem_initialize OK\n");

    /* Step 2: Lookup shmem info */
    ubsmem_shmem_info_t shm_info;
    int ret = ubsmem_shmem_lookup(shm_name, &shm_info);
    if (ret != 0) {
        fprintf(stderr, "[2/3] ubsmem_shmem_lookup(\"%s\") failed: %d\n",
                shm_name, ret);
        fprintf(stderr, "  Is the server running?\n");
        ubsmem_finalize();
        return 1;
    }
    printf("[2/3] shmem lookup OK: name=%s, size=%zu, mem_num=%u\n",
           shm_info.name, shm_info.size, shm_info.mem_num);

    /* Step 3: Map to local VA */
    void *ptr = NULL;
    ret = ubsmem_shmem_map(NULL, buf_size,
                           PROT_READ,
                           MAP_SHARED,
                           shm_name, 0, &ptr);
    if (ret != 0 || !ptr) {
        fprintf(stderr, "[3/3] ubsmem_shmem_map failed: %d\n", ret);
        ubsmem_finalize();
        return 1;
    }
    printf("[3/3] ubsmem_shmem_map OK: ptr=%p\n\n", ptr);

    const float *data = (const float *)ptr;

    /* Run benchmarks */
    int errors = verify_data(data, num_rows, dim);
    if (errors > 0) {
        printf("\n*** %d verification errors — data mismatch ***\n", errors);
    }

    bench_single_load(data, num_rows, dim, num_iters);
    bench_single_read(data, num_rows, dim, num_iters);
    bench_batch_read(data, num_rows, dim, num_iters);
    bench_engram_prefetch(data, num_rows, dim, num_iters);

    /* Cleanup */
    ubsmem_shmem_unmap(ptr, buf_size);
    ubsmem_finalize();
    printf("\nDone.\n");
    return 0;
}
