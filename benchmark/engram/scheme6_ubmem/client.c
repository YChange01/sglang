/*
 * client.c — scheme6 client for UB Shared Memory (ubs_mem) benchmark.
 *
 * Role (mirror of server.c):
 *   1. Initialize the ubs_mem SDK
 *   2. Ensure the same named region exists (idempotent)
 *   3. Map the same shared-memory object read-only
 *   4. Verify one row of the Engram pattern
 *   5. Run three latency/throughput benchmarks matching scheme5:
 *        Bench 1: single-row read latency
 *        Bench 2: batched reads (batch sizes 32/64/128/256)
 *        Bench 3: full Engram prefetch (12 tables x N tokens)
 *   6. Tear down
 *
 * The data path is CPU load/store (via memcpy) into UBMMU-mapped VA,
 * so there is no explicit get/post/poll in the hot loop.
 *
 * Usage:
 *   ./client [num_rows] [dim] [num_iters] [region_name] [object_name]
 *   Defaults: num_rows=10000, dim=341, num_iters=200,
 *             region_name="engram_pool", object_name="engram_table_0"
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ubmem_rw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

#define MAX_BENCH_BATCH 256
#define NUM_ENGRAM_TABLES 12
#define URMA_RW_MAX_BATCH 4096  /* keep parity with scheme5 bench upper bound */

static double diff_us(const struct timespec *a, const struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e6 +
           (b->tv_nsec - a->tv_nsec) / 1e3;
}

/* ================================================================== */
/*  Verification                                                       */
/* ================================================================== */

static int verify_row(const void *base, long dim, long row)
{
    const float *row_ptr = (const float*)base + row * dim;
    /* memcpy to force a real UBMMU-backed read, not a compiler CSE */
    float local[4];
    memcpy(local, row_ptr, sizeof(local));

    printf("\n[Verify] row[%ld][0..3] via memcpy from mapped VA:\n", row);
    printf("  got      = %.6f, %.6f, %.6f, %.6f\n",
           local[0], local[1], local[2], local[3]);
    printf("  expected = %.6f, %.6f, %.6f, %.6f\n",
           (row * dim + 0) * 0.001f,
           (row * dim + 1) * 0.001f,
           (row * dim + 2) * 0.001f,
           (row * dim + 3) * 0.001f);

    int ok = 1;
    for (int i = 0; i < 4; i++) {
        float want = (row * dim + i) * 0.001f;
        if (local[i] != want) ok = 0;
    }
    printf("  result   = %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : -1;
}

/* ================================================================== */
/*  Bench 1: single-row read latency                                   */
/* ================================================================== */

static void bench_single(const void *base, long num_rows, long dim, int num_iters)
{
    size_t row_bytes = (size_t)dim * sizeof(float);
    printf("\n[Bench 1] Single row read latency (1 row = %zu bytes)\n", row_bytes);

    /* Local sink — must be outside the read to avoid dead-store elimination.
     * Use a small heap buffer so it doesn't bloat stack. */
    float *sink = (float*)malloc(row_bytes);
    if (!sink) { fprintf(stderr, "malloc sink failed\n"); return; }

    /* Warmup — first few touches include TLB miss / UBMMU page-in. */
    for (int i = 0; i < 32; i++) {
        memcpy(sink, (const char*)base + (long)(i % num_rows) * row_bytes, row_bytes);
    }

    struct timespec t0, t1;
    double total_us = 0;
    double min_us = 1e18, max_us = 0;
    for (int i = 0; i < num_iters; i++) {
        long row = rand() % num_rows;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(sink, (const char*)base + row * row_bytes, row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = diff_us(&t0, &t1);
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }
    double avg = total_us / num_iters;
    printf("  Avg: %.3f us  Min: %.3f us  Max: %.3f us\n", avg, min_us, max_us);
    printf("  Throughput (single-row): %.1f MB/s\n", row_bytes / avg);
    free(sink);
}

/* ================================================================== */
/*  Bench 2: batched reads, varying batch size                         */
/* ================================================================== */

static void bench_batch(const void *base, long num_rows, long dim, int num_iters)
{
    size_t row_bytes = (size_t)dim * sizeof(float);
    const int batches[] = {32, 64, 128, 256};

    printf("\n[Bench 2] Batch read latency (memcpy in a loop)\n");
    printf("  %-8s %-12s %-10s %-14s\n", "Batch", "Data", "Avg us", "Throughput");
    printf("  ----------------------------------------------\n");

    /* Local receive buffer, sized for the biggest batch */
    float *sink = (float*)malloc((size_t)MAX_BENCH_BATCH * row_bytes);
    if (!sink) { fprintf(stderr, "malloc sink failed\n"); return; }

    for (size_t b = 0; b < sizeof(batches) / sizeof(batches[0]); b++) {
        int batch = batches[b];
        if (batch > MAX_BENCH_BATCH) continue;

        double total_us = 0;
        for (int iter = 0; iter < num_iters; iter++) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < batch; i++) {
                long row = rand() % num_rows;
                memcpy(sink + (size_t)i * dim,
                       (const char*)base + row * row_bytes,
                       row_bytes);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }
        double avg = total_us / num_iters;
        double data_kb = (double)batch * row_bytes / 1024.0;
        double throughput_mbs = (data_kb / 1024.0) / (avg / 1e6);
        printf("  %-8d %-8.1f KB %-9.2f %.1f MB/s\n",
               batch, data_kb, avg, throughput_mbs);
    }
    free(sink);
}

/* ================================================================== */
/*  Bench 3: full Engram prefetch (12 tables x N tokens)               */
/* ================================================================== */

static void bench_engram_prefetch(const void *base, long num_rows, long dim,
                                   int num_iters)
{
    size_t row_bytes = (size_t)dim * sizeof(float);
    const int token_sizes[] = {32, 64, 128, 256};

    printf("\n[Bench 3] Full Engram prefetch (%d tables x N tokens)\n",
           NUM_ENGRAM_TABLES);
    printf("  %-8s %-12s %-10s %-14s\n", "Tokens", "Data", "Avg ms", "Throughput");
    printf("  ----------------------------------------------\n");

    /* Heap-allocated sink, sized for the biggest case */
    float *sink = (float*)malloc((size_t)URMA_RW_MAX_BATCH * row_bytes);
    if (!sink) { fprintf(stderr, "malloc sink failed\n"); return; }

    for (size_t b = 0; b < sizeof(token_sizes) / sizeof(token_sizes[0]); b++) {
        int tokens = token_sizes[b];
        long total_reads = (long)NUM_ENGRAM_TABLES * tokens;
        if (total_reads > URMA_RW_MAX_BATCH) {
            printf("  %-8d  (skipped: %ld > MAX=%d)\n",
                   tokens, total_reads, URMA_RW_MAX_BATCH);
            continue;
        }

        double total_us = 0;
        for (int iter = 0; iter < num_iters; iter++) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (long i = 0; i < total_reads; i++) {
                long row = rand() % num_rows;
                memcpy(sink + (size_t)i * dim,
                       (const char*)base + row * row_bytes,
                       row_bytes);
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }
        double avg_us = total_us / num_iters;
        double avg_ms = avg_us / 1000.0;
        double data_kb = (double)total_reads * row_bytes / 1024.0;
        double throughput_mbs = (data_kb / 1024.0) / (avg_ms / 1e3);
        printf("  %-8d %-8.1f KB %-9.3f %.1f MB/s\n",
               tokens, data_kb, avg_ms, throughput_mbs);
    }
    free(sink);
}

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */

int main(int argc, char *argv[])
{
    long num_rows  = (argc > 1) ? atol(argv[1]) : 10000;
    long dim       = (argc > 2) ? atol(argv[2]) : 341;
    int  num_iters = (argc > 3) ? atoi(argv[3]) : 200;
    const char *region_name = (argc > 4) ? argv[4] : "engram_pool";
    const char *object_name = (argc > 5) ? argv[5] : "engram_table_0";

    if (num_rows <= 0 || dim <= 0 || num_iters <= 0) {
        fprintf(stderr, "invalid args\n");
        return 1;
    }

    size_t row_bytes  = (size_t)dim * sizeof(float);
    size_t total_size = (size_t)num_rows * row_bytes;
    /* Must match the server's aligned allocation — ubs_mem 4 MB min. */
    size_t alloc_size = UBMEM_RW_ALIGN_UP(total_size);

    printf("=== scheme6 ubs_mem client ===\n");
    printf("  region = \"%s\"\n", region_name);
    printf("  object = \"%s\"\n", object_name);
    printf("  table  = %ld rows x %ld dim = %.2f MB (logical)\n",
           num_rows, dim, total_size / (1024.0 * 1024.0));
    printf("  alloc  = %.2f MB (4MB-aligned for ubs_mem)\n",
           alloc_size / (1024.0 * 1024.0));
    printf("  iters  = %d\n", num_iters);

    ubmem_rw_ctx_t *urw = ubmem_rw_init();
    if (!urw) {
        fprintf(stderr, "ubmem_rw_init failed\n");
        return 1;
    }

    /* Query real cluster hostnames (same reason as in server.c — never
     * hardcode). We don't destroy the region here; the server owns it. */
    ubmem_rw_cluster_t cluster;
    if (ubmem_rw_query_cluster(urw, &cluster) != UBMEM_RW_OK) {
        goto fail;
    }
    printf("  cluster host_num = %d, local_nid = %u, local_host_idx = %d\n",
           cluster.n_hosts, cluster.local_nid, cluster.local_host_idx);
    for (int h = 0; h < cluster.n_hosts; h++) {
        printf("    host[%d] = \"%s\"%s\n",
               h, cluster.hostnames[h],
               (h == cluster.local_host_idx) ? "  (local)" : "");
    }
    if (cluster.n_hosts < 2) {
        fprintf(stderr, "  [FAIL] cluster has < 2 hosts; scheme6 needs cross-node\n");
        goto fail;
    }

    /* Build the same host list the server used. Affinity doesn't matter
     * on the ALREADY_EXIST path (the region is already defined), but
     * we keep it consistent so the create call is meaningful if the
     * client ever happens to race ahead of the server. The data actually
     * lives on the "remote" host from the client's perspective — i.e.
     * the non-local entry should have affinity. */
    ubmem_rw_host_t hosts[UBMEM_RW_MAX_HOSTS];
    for (int h = 0; h < cluster.n_hosts; h++) {
        hosts[h].hostname = cluster.hostnames[h];
        hosts[h].affinity = (h != cluster.local_host_idx);
    }
    if (ubmem_rw_ensure_region(urw, region_name, hosts, cluster.n_hosts) != UBMEM_RW_OK) {
        goto fail;
    }

    /* Map the shmem object read-only. The backing memory lives on the
     * server; this call returns a local VA routed by UBMMU to UB fabric.
     * Map length must match the server's allocate() — the 4MB-aligned
     * alloc_size, not the logical total_size. */
    void *ptr = NULL;
    if (ubmem_rw_map(urw, object_name, alloc_size,
                     PROT_READ, MAP_SHARED, 0, &ptr) != UBMEM_RW_OK) {
        fprintf(stderr, "map failed — is the server running and object allocated?\n");
        goto fail;
    }

    srand((unsigned)time(NULL));

    /* Verify the pattern first. If this fails, bench numbers are meaningless. */
    if (verify_row(ptr, dim, 42) != 0) {
        fprintf(stderr, "verification FAILED — bailing out before benches\n");
        (void)ubmem_rw_unmap(urw, ptr, alloc_size);
        goto fail;
    }

    bench_single(ptr, num_rows, dim, num_iters);
    bench_batch (ptr, num_rows, dim, num_iters);
    bench_engram_prefetch(ptr, num_rows, dim, num_iters);

    (void)ubmem_rw_unmap(urw, ptr, alloc_size);
    ubmem_rw_destroy(urw);
    printf("\nDone.\n");
    return 0;

fail:
    ubmem_rw_destroy(urw);
    return 1;
}
