/*
 * server.c — scheme6 server for UB Shared Memory (ubs_mem) benchmark.
 *
 * Role:
 *   1. Initialize the ubs_mem SDK
 *   2. Ensure a named region exists that spans Node1 and Node2
 *   3. Allocate a named shared-memory object within the region
 *   4. Map it locally, fill with the Engram pattern (data[i] = i * 0.001f)
 *   5. Wait for Ctrl+C, then tear down
 *
 * The client on the peer node runs the mirror: lookup the same region,
 * map the same object, and read. No TCP / handle exchange — the
 * (region, object) name is the handle.
 *
 * Usage:
 *   ./server [num_rows] [dim] [region_name] [object_name]
 *   Defaults: num_rows=10000, dim=341, region_name="engram_pool",
 *             object_name="engram_table_0"
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ubmem_rw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;

static void sigint_handler(int sig)
{
    (void)sig;
    g_stop = 1;
}

int main(int argc, char *argv[])
{
    long num_rows = (argc > 1) ? atol(argv[1]) : 10000;
    long dim      = (argc > 2) ? atol(argv[2]) : 341;
    const char *region_name = (argc > 3) ? argv[3] : "engram_pool";
    const char *object_name = (argc > 4) ? argv[4] : "engram_table_0";

    if (num_rows <= 0 || dim <= 0) {
        fprintf(stderr, "invalid num_rows=%ld dim=%ld\n", num_rows, dim);
        return 1;
    }

    size_t row_bytes  = (size_t)dim * sizeof(float);
    size_t total_size = (size_t)num_rows * row_bytes;

    printf("=== scheme6 ubs_mem server ===\n");
    printf("  region = \"%s\"\n", region_name);
    printf("  object = \"%s\"\n", object_name);
    printf("  table  = %ld rows x %ld dim = %.2f MB\n",
           num_rows, dim, total_size / (1024.0 * 1024.0));
    printf("\n");

    /* 1. SDK init */
    ubmem_rw_ctx_t *urw = ubmem_rw_init();
    if (!urw) {
        fprintf(stderr, "ubmem_rw_init failed\n");
        return 1;
    }

    uint32_t nid = 0;
    if (ubmem_rw_local_nid(urw, &nid) == 0) {
        printf("  local supernode nid = %u\n", nid);
    }

    /* 2. Ensure region exists spanning both nodes.
     *    The server has local affinity; client will not. */
    ubmem_rw_host_t hosts[2] = {
        { .hostname = "node1", .affinity = true  },
        { .hostname = "node2", .affinity = false },
    };
    if (ubmem_rw_ensure_region(urw, region_name, hosts, 2) != UBMEM_RW_OK) {
        goto fail;
    }

    /* 3. Clean up any stale object from a previous run, then allocate.
     *    Ignore NOT_FOUND on the deallocate path. */
    (void)ubmem_rw_deallocate(urw, object_name);

    uint64_t alloc_flags = UBMEM_RW_FLAG_CACHE | UBMEM_RW_FLAG_HUGEPAGE;
    if (ubmem_rw_allocate(urw, region_name, object_name,
                          total_size, 0644, alloc_flags) != UBMEM_RW_OK) {
        goto fail;
    }

    /* 4. Map locally for writing. Use MAP_SHARED so the object is visible
     *    cluster-wide. Hugepage flag is already set at allocate time. */
    void *ptr = NULL;
    if (ubmem_rw_map(urw, object_name, total_size,
                     PROT_READ | PROT_WRITE, MAP_SHARED, 0, &ptr) != UBMEM_RW_OK) {
        goto fail_dealloc;
    }

    /* 5. Fill with the Engram verification pattern — same as scheme5,
     *    so a reader using row r expects data[r*dim + k] == (r*dim+k)*0.001f. */
    printf("\nfilling %.2f MB with pattern data[i] = i * 0.001f ...\n",
           total_size / (1024.0 * 1024.0));
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    float *data = (float*)ptr;
    size_t num_floats = total_size / sizeof(float);
    for (size_t i = 0; i < num_floats; i++) {
        data[i] = (float)i * 0.001f;
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double fill_ms = (t1.tv_sec - t0.tv_sec) * 1e3 +
                     (t1.tv_nsec - t0.tv_nsec) / 1e6;
    printf("  fill done in %.2f ms\n", fill_ms);

    printf("  sample data[0]      = %.6f (expect 0.000000)\n", data[0]);
    if (num_floats > 42) {
        printf("  sample data[42]     = %.6f (expect %.6f)\n",
               data[42], 42 * 0.001f);
    }
    if (num_floats > 14322) {
        /* row 42, col 0 when dim=341: index 42*341 = 14322 */
        printf("  sample row[42][0]   = %.6f (expect %.6f)\n",
               data[42 * dim], (42 * dim) * 0.001f);
    }

    printf("\nserver ready — waiting for client. Press Ctrl+C to stop.\n");
    fflush(stdout);

    signal(SIGINT, sigint_handler);
    while (!g_stop) {
        sleep(1);
    }

    /* 6. Tear down in reverse order of creation. */
    printf("\nstopping, tearing down ...\n");
    (void)ubmem_rw_unmap(urw, ptr, total_size);
    (void)ubmem_rw_deallocate(urw, object_name);
    /* Leave the region intact — other users might share it. Uncomment if
     * you want to destroy the region on exit:
     *     (void)ubmem_rw_destroy_region(urw, region_name);
     */
    ubmem_rw_destroy(urw);
    printf("server stopped cleanly.\n");
    return 0;

fail_dealloc:
    (void)ubmem_rw_deallocate(urw, object_name);
fail:
    ubmem_rw_destroy(urw);
    return 1;
}
