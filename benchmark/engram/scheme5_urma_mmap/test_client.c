/**
 * Test client: Import remote segment via URMA_SEG_MAP and read data.
 *
 * Usage:
 *   ./test_client <seg_va> <seg_len> <seg_id> <eid_hex> <uasid> [num_rows] [dim]
 *
 * The seg info values come from the server's JSON output.
 * Reads a few rows and measures latency.
 */

#include "urma_mmap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void hex_to_bytes(const char* hex, char* out, size_t out_len)
{
    memset(out, 0, out_len);
    size_t hex_len = strlen(hex);
    for (size_t i = 0; i < hex_len / 2 && i < out_len; i++) {
        unsigned int byte;
        sscanf(hex + 2 * i, "%02x", &byte);
        out[i] = (char)byte;
    }
}

int main(int argc, char* argv[])
{
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <seg_va> <seg_len> <seg_id> <eid_hex> <uasid> [num_rows] [dim]\n", argv[0]);
        return 1;
    }

    urma_mmap_seg_info_t remote_info;
    memset(&remote_info, 0, sizeof(remote_info));
    remote_info.seg_va = (uint64_t)atol(argv[1]);
    remote_info.seg_len = (uint64_t)atol(argv[2]);
    remote_info.seg_id = (uint32_t)atoi(argv[3]);
    hex_to_bytes(argv[4], remote_info.eid, sizeof(remote_info.eid));
    remote_info.uasid = (uint32_t)atoi(argv[5]);
    remote_info.token = 0xACFE;

    int num_rows = (argc > 6) ? atoi(argv[6]) : 10000;
    int dim = (argc > 7) ? atoi(argv[7]) : 341;

    printf("Importing remote segment: va=0x%lx, len=%lu, seg_id=%u\n",
           remote_info.seg_va, remote_info.seg_len, remote_info.seg_id);

    /* Init URMA */
    urma_mmap_ctx_t* ctx = urma_mmap_init(NULL, -1);
    if (!ctx) {
        fprintf(stderr, "urma_mmap_init failed\n");
        return 1;
    }

    /* Import + mmap remote segment */
    void* mapped_addr = NULL;
    int rc = urma_mmap_import(ctx, &remote_info, &mapped_addr);
    if (rc != URMA_MMAP_OK) {
        fprintf(stderr, "urma_mmap_import failed: %d\n", rc);
        urma_mmap_destroy(ctx);
        return 1;
    }

    printf("Mapped remote memory at local address: %p\n", mapped_addr);

    /* Direct read via load/store */
    float* table = (float*)mapped_addr;
    printf("\nDirect read test:\n");
    printf("  row[0][0..3]  = %.6f, %.6f, %.6f, %.6f\n",
           table[0], table[1], table[2], table[3]);
    printf("  row[42][0..3] = %.6f, %.6f, %.6f, %.6f\n",
           table[42 * dim], table[42 * dim + 1], table[42 * dim + 2], table[42 * dim + 3]);

    /* Latency benchmark */
    printf("\nLatency benchmark:\n");
    int test_iters = 10000;

    /* Single row read */
    struct timespec t0, t1;
    double total_ns = 0;
    for (int i = 0; i < test_iters; i++) {
        int row = rand() % num_rows;
        float sum = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int d = 0; d < dim; d++) {
            sum += table[row * dim + d];
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        total_ns += ns;
        (void)sum;  /* prevent optimization */
    }
    printf("  Single row read (%d dim): avg %.1f ns (%.3f us)\n",
           dim, total_ns / test_iters, total_ns / test_iters / 1000);

    /* Batch read (128 rows) */
    int batch = 128;
    total_ns = 0;
    for (int i = 0; i < test_iters / 10; i++) {
        float sum = 0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int b = 0; b < batch; b++) {
            int row = rand() % num_rows;
            for (int d = 0; d < dim; d++) {
                sum += table[row * dim + d];
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
        total_ns += ns;
        (void)sum;
    }
    double batch_avg_us = total_ns / (test_iters / 10) / 1000;
    double batch_throughput = (double)batch * dim * sizeof(float) / 1e6 / (batch_avg_us / 1e6);
    printf("  Batch read (%d rows): avg %.1f us (%.1f MB/s)\n",
           batch, batch_avg_us, batch_throughput);

    /* Cleanup */
    urma_mmap_unimport(ctx, &remote_info);
    urma_mmap_destroy(ctx);
    printf("\nDone.\n");
    return 0;
}
