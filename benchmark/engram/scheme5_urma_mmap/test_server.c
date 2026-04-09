/**
 * Test server: Register a memory region with Engram-like data.
 *
 * Usage:
 *   ./test_server [num_rows] [dim]
 *
 * Prints segment info in JSON format for the client to import.
 * Waits until user presses Enter, then cleans up.
 */

#include "urma_mmap.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char* argv[])
{
    int num_rows = 10000;
    int dim = 341;

    if (argc > 1) num_rows = atoi(argv[1]);
    if (argc > 2) dim = atoi(argv[2]);

    size_t row_bytes = dim * sizeof(float);
    size_t total_bytes = (size_t)num_rows * row_bytes;

    printf("Allocating %.1f MB for %d rows x %d dim...\n",
           total_bytes / 1e6, num_rows, dim);

    /* Allocate page-aligned memory (URMA requires page alignment) */
    float* data = NULL;
    int alloc_rc = posix_memalign((void**)&data, 4096, total_bytes);
    if (alloc_rc != 0 || !data) {
        fprintf(stderr, "Failed to allocate %zu bytes (rc=%d)\n", total_bytes, alloc_rc);
        return 1;
    }

    srand((unsigned)time(NULL));
    for (size_t i = 0; i < (size_t)num_rows * dim; i++) {
        data[i] = (float)rand() / RAND_MAX;
    }

    /* Print first few values for verification */
    printf("data[0][0..3] = %.6f, %.6f, %.6f, %.6f\n",
           data[0], data[1], data[2], data[3]);
    printf("data[42][0..3] = %.6f, %.6f, %.6f, %.6f\n",
           data[42 * dim], data[42 * dim + 1], data[42 * dim + 2], data[42 * dim + 3]);

    /* Init URMA */
    urma_mmap_ctx_t* ctx = urma_mmap_init(NULL, -1);
    if (!ctx) {
        fprintf(stderr, "urma_mmap_init failed\n");
        free(data);
        return 1;
    }

    /* Register memory */
    urma_mmap_seg_info_t info;
    int rc = urma_mmap_register(ctx, data, total_bytes, &info);
    if (rc != URMA_MMAP_OK) {
        fprintf(stderr, "urma_mmap_register failed: %d\n", rc);
        urma_mmap_destroy(ctx);
        free(data);
        return 1;
    }

    /* Print segment info as JSON for client */
    printf("\n=== SEGMENT INFO (copy to client) ===\n");
    printf("{\n");
    printf("  \"eid\": \"");
    for (int i = 0; i < 64; i++) printf("%02x", (unsigned char)info.eid[i]);
    printf("\",\n");
    printf("  \"uasid\": %u,\n", info.uasid);
    printf("  \"seg_va\": %lu,\n", info.seg_va);
    printf("  \"seg_len\": %lu,\n", info.seg_len);
    printf("  \"token_id\": %u,\n", info.token_id);
    printf("  \"token\": %u,\n", info.token);
    printf("  \"num_rows\": %d,\n", num_rows);
    printf("  \"dim\": %d\n", dim);
    printf("}\n");
    printf("=====================================\n\n");

    urma_mmap_dump_info(&info);

    printf("Server ready. Press Enter to stop...\n");
    getchar();

    urma_mmap_unregister(ctx, &info);
    urma_mmap_destroy(ctx);
    free(data);
    printf("Server stopped.\n");
    return 0;
}
