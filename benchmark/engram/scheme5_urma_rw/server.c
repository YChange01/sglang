/**
 * Server: allocates Engram tables, registers URMA segment, waits for client.
 *
 * Usage:
 *   ./server [port] [total_size_mb]
 *   Default: port=URMA_RW_DEFAULT_PORT, total_size_mb=2048 (2GB, enough for
 *            ~10000 rows x 12 tables)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "urma_rw.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

static volatile sig_atomic_t g_stop = 0;

static void sigint_handler(int sig)
{
    (void)sig;
    g_stop = 1;
}

int main(int argc, char* argv[])
{
    uint16_t port = URMA_RW_DEFAULT_PORT;
    uint64_t total_mb = 2048;  /* 2 GB */

    if (argc > 1) port = (uint16_t)atoi(argv[1]);
    if (argc > 2) total_mb = strtoull(argv[2], NULL, 10);

    uint64_t buf_size = total_mb * 1024 * 1024;

    printf("=== URMA RW Server ===\n");
    printf("Port: %u, Buffer: %lu MB\n", port, (unsigned long)total_mb);
    printf("Initializing URMA context...\n");

    urma_rw_ctx_t* ctx = urma_rw_init(NULL, buf_size);
    if (!ctx) {
        fprintf(stderr, "urma_rw_init failed\n");
        return 1;
    }

    /* Fill buffer with deterministic pattern for verification */
    float* data = (float*)urma_rw_get_buffer(ctx);
    uint64_t num_floats = buf_size / sizeof(float);
    printf("Filling %lu floats with pattern (data[i] = i * 0.001)...\n",
           (unsigned long)num_floats);
    srand(42);
    for (uint64_t i = 0; i < num_floats; i++) {
        data[i] = (float)i * 0.001f;
    }

    printf("Sample values:\n");
    if (num_floats > 0)      printf("  data[0]      = %.6f\n", data[0]);
    if (num_floats > 42)     printf("  data[42]     = %.6f\n", data[42]);
    if (num_floats > 1000)   printf("  data[1000]   = %.6f\n", data[1000]);
    if (num_floats > 100000) printf("  data[100000] = %.6f\n", data[100000]);

    signal(SIGINT, sigint_handler);

    if (urma_rw_server_accept(ctx, port) != 0) {
        fprintf(stderr, "server_accept failed\n");
        urma_rw_destroy(ctx);
        return 1;
    }

    printf("\nServer is ready. Press Ctrl+C to stop.\n");
    while (!g_stop) {
        sleep(1);
    }

    printf("\nStopping server...\n");
    urma_rw_destroy(ctx);
    printf("Server stopped.\n");
    return 0;
}
