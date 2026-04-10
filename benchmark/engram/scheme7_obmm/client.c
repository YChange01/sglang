/*
 * scheme7 client — cross-node Engram reader via obmm direct UAPI.
 *
 * Usage:
 *   sudo ./client <server_ip> [port] [num_rows] [dim] [num_iters]
 *   Example:
 *     sudo ./client 192.168.84.245 13857 10000 341 200
 *
 * Flow:
 *   1. TCP connect + recv scheme7_wire_handle
 *   2. open /dev/obmm
 *   3. ioctl OBMM_CMD_IMPORT  → kernel programs UBMMU, returns local mem_id
 *   4. open /dev/obmm_shmdev<local_mem_id>
 *   5. mmap                  → local VA backed by remote memory via UB fabric
 *   6. verify row 42 matches server's fill pattern
 *   7. Bench 1: single-row read latency
 *   8. Bench 2: batch read latency (32/64/128/256)
 *   9. Bench 3: full Engram prefetch (12 tables × N tokens)
 *   10. cleanup: munmap, close, UNIMPORT
 *
 * This mirrors scheme5_urma_rw/client.c exactly in terms of bench
 * output format, so numbers can be diff'd side by side.
 *
 * The crucial difference: scheme5 uses urma_post_jetty_send_wr +
 * poll_cq (RDMA-style post/poll). scheme7 uses plain memcpy(), which
 * the CPU turns into UB fabric loads routed by UBMMU. No completion
 * queue, no wait, no batching API — just loads.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <ub/obmm.h>

#define DEFAULT_PORT       13857
#define DEFAULT_NUM_ROWS   10000
#define DEFAULT_DIM        341
#define DEFAULT_NUM_ITERS  200

/* Max batch size for the bench_engram_prefetch heap allocations —
 * 12 tables × 256 tokens = 3072 rows. Keep a little headroom. */
#define MAX_PREFETCH_READS 4096

struct scheme7_wire_handle {
    uint64_t uba;
    uint64_t length;
    uint32_t tokenid;
    uint32_t scna;
    uint32_t dcna;
    int32_t  pxm_numa;
    uint8_t  seid[16];
    uint8_t  deid[16];
} __attribute__((packed));

/*
 * Read this node's primary CNA from sysfs.
 */
static uint32_t read_local_cna(void)
{
    const char *path = "/sys/devices/ub_bus_controller1/00002/primary_cna";
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[warn] cannot read %s: %s\n", path, strerror(errno));
        return 0;
    }
    unsigned int cna = 0;
    if (fscanf(f, "%x", &cna) != 1) {
        fprintf(stderr, "[warn] failed to parse CNA from %s\n", path);
    }
    fclose(f);
    return (uint32_t)cna;
}

/* ---------- utilities ---------- */

static double diff_us(const struct timespec *a, const struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1.0e6
         + (b->tv_nsec - a->tv_nsec) / 1.0e3;
}

static int recv_all(int fd, void *buf, size_t n)
{
    char *p = (char *)buf;
    while (n > 0) {
        ssize_t k = recv(fd, p, n, 0);
        if (k == 0) return -1;  /* peer closed */
        if (k < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += k;
        n -= (size_t)k;
    }
    return 0;
}

static int tcp_connect(const char *ip, uint16_t port)
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); return -1; }

    struct sockaddr_in sa = {0};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &sa.sin_addr) != 1) {
        fprintf(stderr, "inet_pton(%s) failed\n", ip);
        close(s);
        return -1;
    }
    if (connect(s, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        perror("connect");
        close(s);
        return -1;
    }
    return s;
}

/* ---------- benchmarks ---------- */

static void verify_read(const void *remote_va, int dim)
{
    int row_bytes = dim * (int)sizeof(float);
    printf("\n[Verify] Reading and checking data...\n");

    /* Copy row 42 into a local scratch. This is an honest "memcpy from
     * remote VA" — the CPU performs loads from UB-routed memory. */
    float local[4];
    memcpy(local,
           (const char *)remote_va + (uint64_t)42 * row_bytes,
           sizeof(local));

    printf("  row[42][0..3] = %.6f, %.6f, %.6f, %.6f\n",
           local[0], local[1], local[2], local[3]);
    printf("  Expected:       %.6f, %.6f, %.6f, %.6f "
           "(server fill pattern: data[i] = i*0.001)\n",
           (42 * dim + 0) * 0.001f,
           (42 * dim + 1) * 0.001f,
           (42 * dim + 2) * 0.001f,
           (42 * dim + 3) * 0.001f);
}

static void bench_single_read(const void *remote_va, void *local_buf,
                              int num_rows, int dim, int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    printf("\n[Bench 1] Single row read latency (1 row = %d bytes)\n", row_bytes);

    /* Warmup (10 reads, not timed) */
    for (int i = 0; i < 10; i++) {
        memcpy(local_buf, remote_va, (size_t)row_bytes);
    }

    struct timespec t0, t1;
    double total_us = 0, min_us = 1e9, max_us = 0;
    for (int i = 0; i < num_iters; i++) {
        int row = rand() % num_rows;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(local_buf,
               (const char *)remote_va + (uint64_t)row * row_bytes,
               (size_t)row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = diff_us(&t0, &t1);
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
    }

    double avg = total_us / num_iters;
    printf("  Avg: %.2f us, Min: %.2f us, Max: %.2f us\n", avg, min_us, max_us);
    printf("  Throughput (single-row): %.1f MB/s\n", row_bytes / avg);
}

static void bench_batch_read(const void *remote_va, void *local_buf,
                             int num_rows, int dim, int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    int batches[] = {32, 64, 128, 256};

    printf("\n[Bench 2] Batch read latency (varying batch size)\n");
    printf("  %-8s %-10s %-10s %-12s\n", "Batch", "Data", "Avg us", "Throughput");
    printf("  --------------------------------------------\n");

    for (int b = 0; b < 4; b++) {
        int batch = batches[b];
        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < batch; i++) {
                int row = rand() % num_rows;
                memcpy((char *)local_buf + (uint64_t)i * row_bytes,
                       (const char *)remote_va + (uint64_t)row * row_bytes,
                       (size_t)row_bytes);
            }
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

static void bench_engram_prefetch(const void *remote_va, void *local_buf,
                                  int num_rows, int dim, int num_iters)
{
    int row_bytes = dim * (int)sizeof(float);
    int NUM_TABLES = 12;
    int token_sizes[] = {32, 64, 128, 256};

    printf("\n[Bench 3] Full Engram prefetch (%d tables x N tokens)\n", NUM_TABLES);
    printf("  %-8s %-10s %-10s %-12s\n", "Tokens", "Data", "Avg ms", "Throughput");
    printf("  --------------------------------------------\n");

    for (int b = 0; b < 4; b++) {
        int tokens = token_sizes[b];
        int total_reads = NUM_TABLES * tokens;
        if (total_reads > MAX_PREFETCH_READS) {
            printf("  %-8d  (skipped: %d > %d)\n",
                   tokens, total_reads, MAX_PREFETCH_READS);
            continue;
        }

        struct timespec t0, t1;
        double total_us = 0;

        for (int iter = 0; iter < num_iters; iter++) {
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int i = 0; i < total_reads; i++) {
                int row = rand() % num_rows;
                memcpy((char *)local_buf + (uint64_t)i * row_bytes,
                       (const char *)remote_va + (uint64_t)row * row_bytes,
                       (size_t)row_bytes);
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
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <server_ip> [port] [num_rows] [dim] [num_iters]\n",
                argv[0]);
        return 1;
    }

    const char *server_ip = argv[1];
    uint16_t port = (argc > 2) ? (uint16_t)atoi(argv[2]) : DEFAULT_PORT;
    int num_rows = (argc > 3) ? atoi(argv[3]) : DEFAULT_NUM_ROWS;
    int dim      = (argc > 4) ? atoi(argv[4]) : DEFAULT_DIM;
    int num_iters = (argc > 5) ? atoi(argv[5]) : DEFAULT_NUM_ITERS;

    size_t row_bytes = (size_t)dim * sizeof(float);
    size_t local_buf_size = MAX_PREFETCH_READS * row_bytes;
    if (local_buf_size < 64UL * 1024UL * 1024UL) {
        local_buf_size = 64UL * 1024UL * 1024UL;  /* match scheme5 headroom */
    }

    printf("=== scheme7 obmm client ===\n");
    printf("  server   : %s:%u\n", server_ip, port);
    printf("  table    : %d rows x %d dim = %.1f MB\n",
           num_rows, dim, (double)num_rows * dim * 4 / 1e6);
    printf("  local_buf: %.1f MB\n", (double)local_buf_size / 1e6);
    printf("  iters    : %d\n\n", num_iters);

    /* 1. TCP connect + recv handle */
    int cs = tcp_connect(server_ip, port);
    if (cs < 0) return 1;

    struct scheme7_wire_handle wire;
    if (recv_all(cs, &wire, sizeof(wire)) != 0) {
        fprintf(stderr, "[FAIL 1] recv handle: %s\n", strerror(errno));
        close(cs);
        return 1;
    }
    close(cs);
    printf("[1] handle recv ok\n");
    printf("    uba     = 0x%" PRIx64 "\n", wire.uba);
    printf("    length  = %" PRIu64 " (%.1f MB)\n",
           wire.length, wire.length / (1024.0 * 1024.0));
    printf("    tokenid = 0x%x\n", wire.tokenid);

    /* Sanity check that the remote table is at least as big as we
     * expect based on num_rows × dim. */
    size_t expected = (size_t)num_rows * row_bytes;
    if (wire.length < expected) {
        fprintf(stderr,
                "[warn] remote length %" PRIu64 " < expected %zu — "
                "num_rows or dim too large for server's buffer\n",
                wire.length, expected);
    }

    /* 2. open /dev/obmm + read local CNA */
    int fd_ctl = open("/dev/obmm", O_RDWR);
    if (fd_ctl < 0) { perror("[2] open /dev/obmm"); return 1; }

    uint32_t local_cna = read_local_cna();
    printf("[2] /dev/obmm fd_ctl=%d, local CNA=0x%04x\n", fd_ctl, local_cna);

    /* 3. IMPORT. Kernel requires EXACTLY ONE of {ALLOW_MMAP, NUMA_REMOTE}.
     *    For cross-node: use NUMA_REMOTE alone.
     *
     *    scna = server's CNA (source of data, received in handle)
     *    dcna = our own CNA  (destination / importer)
     *
     *    dmesg errors we've seen and fixed:
     *      "Exactly one of {ALLOW_MMAP, NUMA_REMOTE}" → use one flag
     *      "0x0 is not a known scna"                  → fill real CNA
     */
    struct obmm_cmd_import imp;
    memset(&imp, 0, sizeof(imp));
    imp.flags    = OBMM_IMPORT_FLAG_NUMA_REMOTE;
    imp.addr     = wire.uba;
    imp.length   = wire.length;
    imp.tokenid  = wire.tokenid;
    imp.scna     = wire.scna;     /* server's CNA (from handle) */
    imp.dcna     = local_cna;     /* our own CNA */
    imp.numa_id  = -1;            /* any NUMA */
    imp.base_dist = 0;
    memcpy(imp.seid, wire.seid, 16);
    memcpy(imp.deid, wire.deid, 16);

    printf("    IMPORT params: scna=0x%04x dcna=0x%04x addr=0x%" PRIx64
           " tokenid=0x%x\n", imp.scna, imp.dcna, (uint64_t)imp.addr,
           imp.tokenid);

    if (ioctl(fd_ctl, OBMM_CMD_IMPORT, &imp) < 0) {
        fprintf(stderr, "[FAIL 3] OBMM_CMD_IMPORT: %s (errno=%d)\n",
                strerror(errno), errno);
        fprintf(stderr, "         check dmesg for obmm kernel message\n");
        close(fd_ctl);
        return 1;
    }
    printf("[3] IMPORT ok: local mem_id=%" PRIu64 "\n", (uint64_t)imp.mem_id);

    /* 4. open /dev/obmm_shmdev<local_mem_id> */
    char path[64];
    snprintf(path, sizeof(path),
             "/dev/obmm_shmdev%" PRIu64, (uint64_t)imp.mem_id);
    int fd_data = open(path, O_RDWR);
    if (fd_data < 0) {
        fprintf(stderr, "[FAIL 4] open %s: %s\n", path, strerror(errno));
        goto fail_import;
    }
    printf("[4] %s fd_data=%d\n", path, fd_data);

    /* 5. mmap — remote memory visible as local VA */
    void *remote_va = mmap(NULL, wire.length, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd_data, 0);
    if (remote_va == MAP_FAILED) {
        fprintf(stderr, "[FAIL 5] mmap: %s\n", strerror(errno));
        close(fd_data);
        goto fail_import;
    }
    printf("[5] remote VA = %p\n", remote_va);

    /* 6. Allocate local scratch buffer (pure host DRAM) for the
     *    benchmarks to memcpy INTO. This is the "destination" of
     *    each read. */
    void *local_buf = malloc(local_buf_size);
    if (!local_buf) {
        fprintf(stderr, "[FAIL 6] malloc local_buf\n");
        goto fail_mmap;
    }

    srand((unsigned)time(NULL));

    /* 7. verify + benches */
    verify_read(remote_va, dim);
    bench_single_read(remote_va, local_buf, num_rows, dim, num_iters);
    bench_batch_read(remote_va, local_buf, num_rows, dim, num_iters);
    bench_engram_prefetch(remote_va, local_buf, num_rows, dim, num_iters);

    /* 8. cleanup */
    free(local_buf);
    munmap(remote_va, wire.length);
    close(fd_data);
    {
        struct obmm_cmd_unimport un;
        memset(&un, 0, sizeof(un));
        un.mem_id = imp.mem_id;
        if (ioctl(fd_ctl, OBMM_CMD_UNIMPORT, &un) < 0) {
            fprintf(stderr, "[warn] UNIMPORT: %s\n", strerror(errno));
        } else {
            printf("\n[cleanup] UNIMPORT ok\n");
        }
    }
    close(fd_ctl);
    printf("Done.\n");
    return 0;

fail_mmap:
    munmap(remote_va, wire.length);
    close(fd_data);
fail_import:
    {
        struct obmm_cmd_unimport un;
        memset(&un, 0, sizeof(un));
        un.mem_id = imp.mem_id;
        (void)ioctl(fd_ctl, OBMM_CMD_UNIMPORT, &un);
    }
    close(fd_ctl);
    return 1;
}
