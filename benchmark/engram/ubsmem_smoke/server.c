/**
 * UBS-MEM Benchmark Server (data provider)
 *
 * 1. Allocates a named shmem object via ubs_mem SDK
 * 2. Fills it with a deterministic pattern
 * 3. Starts a TCP data server for TCP benchmark baseline
 * 4. Keeps shmem alive for ubs_mem clients to map
 *
 * Usage:
 *   ./server [size_mb] [shmem_name] [tcp_port]
 *   Default: size_mb=128, shmem_name="engram_test", tcp_port=13900
 *
 * Prerequisites:
 *   - ubse.service and ubsmd.service running
 *   - obmm kernel module loaded
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <ubs_mem.h>

#define DEFAULT_TCP_PORT 13900

static volatile sig_atomic_t g_stop = 0;

static void sigint_handler(int sig)
{
    (void)sig;
    g_stop = 1;
}

/* ------------------------------------------------------------------ */
/*  TCP data server                                                   */
/* ------------------------------------------------------------------ */

/*
 * Protocol (per request):
 *   Client sends: { uint64_t offset, uint32_t length }  (12 bytes)
 *   Server sends: data[offset .. offset+length]
 *
 * offset=0xFFFFFFFFFFFFFFFF signals end of session.
 *
 * For batch: client sends N requests back-to-back, reads N responses.
 */

typedef struct {
    uint64_t offset;
    uint32_t length;
} __attribute__((packed)) tcp_req_t;

typedef struct {
    const float *data;
    size_t data_size;
    uint16_t port;
} tcp_server_args_t;

static int recv_all(int fd, void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = recv(fd, (char *)buf + done, len - done, 0);
        if (n <= 0) return -1;
        done += n;
    }
    return 0;
}

static int send_all(int fd, const void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = send(fd, (const char *)buf + done, len - done, 0);
        if (n <= 0) return -1;
        done += n;
    }
    return 0;
}

static void *tcp_server_thread(void *arg)
{
    tcp_server_args_t *args = (tcp_server_args_t *)arg;
    const char *data = (const char *)args->data;
    size_t data_size = args->data_size;

    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd < 0) { perror("socket"); return NULL; }

    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(args->port);

    if (bind(listenfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(listenfd); return NULL;
    }
    listen(listenfd, 1);
    printf("[TCP] Listening on port %u\n", args->port);

    while (!g_stop) {
        struct sockaddr_in cli;
        socklen_t clen = sizeof(cli);
        int connfd = accept(listenfd, (struct sockaddr *)&cli, &clen);
        if (connfd < 0) { if (g_stop) break; continue; }

        /* Disable Nagle for latency */
        int flag = 1;
        setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        printf("[TCP] Client connected from %s:%d\n",
               inet_ntoa(cli.sin_addr), ntohs(cli.sin_port));

        while (!g_stop) {
            tcp_req_t req;
            if (recv_all(connfd, &req, sizeof(req)) != 0) break;
            if (req.offset == 0xFFFFFFFFFFFFFFFFULL) break;  /* EOF */

            if (req.offset + req.length > data_size) {
                fprintf(stderr, "[TCP] OOB: offset=%lu len=%u max=%zu\n",
                        (unsigned long)req.offset, req.length, data_size);
                break;
            }
            if (send_all(connfd, data + req.offset, req.length) != 0) break;
        }

        printf("[TCP] Client disconnected\n");
        close(connfd);
    }

    close(listenfd);
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char *argv[])
{
    size_t size_mb = 128;
    const char *shm_name = "engram_test";
    uint16_t tcp_port = DEFAULT_TCP_PORT;

    if (argc > 1) size_mb = (size_t)atol(argv[1]);
    if (argc > 2) shm_name = argv[2];
    if (argc > 3) tcp_port = (uint16_t)atoi(argv[3]);

    size_t buf_size = size_mb * 1024 * 1024;

    printf("=== UBS-MEM Benchmark Server ===\n");
    printf("shmem: %s, size: %zu MB, tcp_port: %u\n", shm_name, size_mb, tcp_port);

    /* Step 1: Initialize ubs_mem library */
    ubsmem_options_t opts;
    if (ubsmem_init_attributes(&opts) != 0) {
        fprintf(stderr, "ubsmem_init_attributes failed\n");
        return 1;
    }
    if (ubsmem_initialize(&opts) != 0) {
        fprintf(stderr, "ubsmem_initialize failed\n");
        return 1;
    }
    ubsmem_set_logger_level(2);  /* warning only */
    printf("[1/4] ubsmem_initialize OK\n");

    int ret;

    /* Step 2: Query cluster info */
    ubsmem_cluster_info_t cinfo;
    if (ubsmem_lookup_cluster_statistic(&cinfo) == 0) {
        printf("[2/4] Cluster: %d hosts\n", cinfo.host_num);
        for (int h = 0; h < cinfo.host_num; h++) {
            printf("  host[%d]: %s, %d numas\n", h,
                   cinfo.host[h].host_name, cinfo.host[h].numa_num);
        }
    } else {
        printf("[2/4] cluster query failed (non-fatal)\n");
    }

    /* Step 3: Allocate + map named shmem object */
    const char *region_name = "default";
    ubsmem_shmem_deallocate(shm_name);  /* cleanup leftover */

    ret = ubsmem_shmem_allocate(region_name, shm_name, buf_size,
                                0666, UBSM_FLAG_CACHE);
    if (ret != 0) {
        fprintf(stderr, "[3/4] ubsmem_shmem_allocate failed: %d\n", ret);
        ubsmem_finalize();
        return 1;
    }

    void *ptr = NULL;
    ret = ubsmem_shmem_map(NULL, buf_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, shm_name, 0, &ptr);
    if (ret != 0 || !ptr) {
        fprintf(stderr, "[3/4] ubsmem_shmem_map failed: %d\n", ret);
        ubsmem_shmem_deallocate(shm_name);
        ubsmem_finalize();
        return 1;
    }
    printf("[3/4] shmem allocate+map OK: ptr=%p\n", ptr);

    /* Step 4: Fill with deterministic pattern */
    float *data = (float *)ptr;
    size_t num_floats = buf_size / sizeof(float);
    printf("[4/4] Filling %zu floats...\n", num_floats);
    for (size_t i = 0; i < num_floats; i++) {
        data[i] = (float)i * 0.001f;
    }
    printf("  data[0]=%.4f  data[42]=%.4f  data[1000]=%.4f\n",
           data[0], data[42], data[1000]);

    /* Start TCP server thread */
    tcp_server_args_t tcp_args = { data, buf_size, tcp_port };
    pthread_t tcp_tid;
    pthread_create(&tcp_tid, NULL, tcp_server_thread, &tcp_args);

    signal(SIGINT, sigint_handler);
    printf("\nServer ready. Shmem \"%s\" + TCP :%u. Press Ctrl+C to stop.\n",
           shm_name, tcp_port);

    while (!g_stop) {
        sleep(1);
    }

    printf("\nCleaning up...\n");
    pthread_cancel(tcp_tid);
    pthread_join(tcp_tid, NULL);
    ubsmem_shmem_unmap(ptr, buf_size);
    ubsmem_shmem_deallocate(shm_name);
    ubsmem_finalize();
    printf("Server stopped.\n");
    return 0;
}
