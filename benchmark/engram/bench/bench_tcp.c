/**
 * TCP socket read benchmark.
 *
 * Client sends {offset, length} requests to the Engram benchmark server and
 * receives the requested bytes back over TCP. Batch reads pipeline all
 * requests first, then drain the responses.
 */

#include <arpa/inet.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define DEFAULT_TCP_PORT 13900
#define PAPER_SEGS_PER_TOKEN 8
#define PAPER_SEG_BYTES 320

typedef struct {
    uint64_t offset;
    uint32_t length;
} __attribute__((packed)) tcp_req_t;

typedef struct {
    size_t size_mb;
    int rows;
    int dim;
    int iters;
    const char *server_ip;
    uint16_t tcp_port;
} bench_opts_t;

static double diff_us(const struct timespec *a, const struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --size_mb N    remote buffer size in MB (default 128)\n"
            "  --rows N       number of rows (default 10000)\n"
            "  --dim N        embedding dimension (default 341)\n"
            "  --iters N      iterations per benchmark (default 500)\n"
            "  --server_ip IP TCP server IP (default 192.168.84.245)\n"
            "  --tcp_port N   TCP server port (default 13900)\n",
            prog);
}

static int parse_opts(int argc, char **argv, bench_opts_t *opts)
{
    opts->size_mb = 128;
    opts->rows = 10000;
    opts->dim = 341;
    opts->iters = 500;
    opts->server_ip = "192.168.84.245";
    opts->tcp_port = DEFAULT_TCP_PORT;

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
        else if (strcmp(argv[i], "--tcp_port") == 0) opts->tcp_port = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--name") == 0 ||
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

static int recv_all(int fd, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = recv(fd, (char *)buf + done, len - done, 0);
        if (n <= 0) return -1;
        done += (size_t)n;
    }
    return 0;
}

static int send_all(int fd, const void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = send(fd, (const char *)buf + done, len - done, 0);
        if (n <= 0) return -1;
        done += (size_t)n;
    }
    return 0;
}

static int tcp_connect_server(const char *ip, uint16_t port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        perror("socket");
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) {
        fprintf(stderr, "Invalid server IP: %s\n", ip);
        close(fd);
        return -1;
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(fd);
        return -1;
    }

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
    return fd;
}

static void tcp_disconnect_server(int fd)
{
    tcp_req_t eof = { .offset = 0xFFFFFFFFFFFFFFFFULL, .length = 0 };
    send_all(fd, &eof, sizeof(eof));
    close(fd);
}

static int read_row(int fd, float *local_buf, int row, int dim)
{
    int row_bytes = dim * (int)sizeof(float);
    tcp_req_t req = {
        .offset = (uint64_t)row * (uint64_t)row_bytes,
        .length = (uint32_t)row_bytes,
    };
    if (send_all(fd, &req, sizeof(req)) != 0) return -1;
    return recv_all(fd, local_buf, (size_t)row_bytes);
}

static int read_segments(int fd, float *local_buf, const uint64_t *offsets, int count)
{
    for (int i = 0; i < count; i++) {
        tcp_req_t req = { .offset = offsets[i], .length = PAPER_SEG_BYTES };
        if (send_all(fd, &req, sizeof(req)) != 0) return -1;
    }
    for (int i = 0; i < count; i++) {
        if (recv_all(fd, (char *)local_buf + (size_t)i * PAPER_SEG_BYTES,
                     PAPER_SEG_BYTES) != 0) {
            return -1;
        }
    }
    return 0;
}

static void bench_single_seg(int fd, float *local_buf, size_t data_size, int iters)
{
    printf("\n  [Single float load] (skipped for TCP)\n");
    printf("\n  [Single-seg read] %d bytes, %d iters\n", PAPER_SEG_BYTES, iters);

    uint64_t max_offset = data_size - PAPER_SEG_BYTES;
    uint64_t off0 = 0;
    for (int i = 0; i < 10; i++) {
        if (read_segments(fd, local_buf, &off0, 1) != 0) {
            fprintf(stderr, "    warmup read failed\n");
            return;
        }
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
    int ok_iters = 0;
    for (int i = 0; i < iters; i++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        if (read_segments(fd, local_buf, &offsets[i], 1) != 0) {
            fprintf(stderr, "    TCP segment read failed\n");
            break;
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = diff_us(&t0, &t1);
        total_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
        ok_iters++;
    }

    if (ok_iters > 0) {
        double avg = total_us / ok_iters;
        printf("    Avg: %.3f us, Min: %.3f us, Max: %.3f us\n", avg, min_us, max_us);
        printf("    Throughput: %.1f MB/s\n", (double)PAPER_SEG_BYTES / avg);
    }
    free(offsets);
}

static void bench_engram_27b(int fd, float *local_buf, size_t data_size, int iters)
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
        int ok_iters = 0;
        for (int iter = 0; iter < iters; iter++) {
            for (int i = 0; i < total_segs; i++) {
                offsets[i] = ((uint64_t)(rand() % (int)(max_offset / PAPER_SEG_BYTES))) *
                             PAPER_SEG_BYTES;
            }
            clock_gettime(CLOCK_MONOTONIC, &t0);
            if (read_segments(fd, local_buf, offsets, total_segs) != 0) {
                fprintf(stderr, "    TCP batch read failed\n");
                break;
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
            ok_iters++;
        }

        if (ok_iters == 0) {
            free(offsets);
            break;
        }

        double avg_us = total_us / ok_iters;
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

int main(int argc, char **argv)
{
    bench_opts_t opts;
    if (parse_opts(argc, argv, &opts) != 0) return 1;

    size_t data_size = (size_t)opts.rows * opts.dim * sizeof(float);
    size_t buf_size = opts.size_mb * 1024ULL * 1024ULL;
    size_t local_buf_size = (size_t)16384 * PAPER_SEGS_PER_TOKEN * PAPER_SEG_BYTES;
    if (local_buf_size < buf_size) local_buf_size = buf_size;

    printf("=== TCP Read Benchmark ===\n");
    printf("Table: %d rows x %d dim = %d bytes/row\n",
           opts.rows, opts.dim, (int)(opts.dim * sizeof(float)));
    printf("remote buffer: %zu MB, iters: %d\n", opts.size_mb, opts.iters);
    printf("server: %s:%u\n", opts.server_ip, opts.tcp_port);

    float *local_buf = (float *)malloc(local_buf_size);
    if (!local_buf) {
        perror("malloc local_buf");
        return 1;
    }

    int fd = tcp_connect_server(opts.server_ip, opts.tcp_port);
    if (fd < 0) {
        free(local_buf);
        return 1;
    }

    if (read_row(fd, local_buf, 42, opts.dim) == 0) {
        float expected = 42.0f * opts.dim * 0.001f;
        printf("  Verify row[42][0]: got=%.4f expect=%.4f %s\n",
               local_buf[0], expected,
               fabsf(local_buf[0] - expected) < 0.01f ? "OK" : "MISMATCH");
    } else {
        fprintf(stderr, "  Verify read failed\n");
    }

    bench_single_seg(fd, local_buf, data_size, opts.iters);
    bench_engram_27b(fd, local_buf, data_size, opts.iters);
    printf("\n  [Cold/Hot analysis] (skipped for TCP)\n");

    tcp_disconnect_server(fd);
    free(local_buf);
    printf("\n=== Benchmark complete ===\n");
    return 0;
}
