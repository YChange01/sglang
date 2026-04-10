/**
 * Unified Engram Benchmark Server
 *
 * One process serves ALL transport modes:
 *   1. ubs_mem shmem — allocate + fill data (clients map remotely)
 *   2. TCP thread    — request/response data serving
 *   3. URMA thread   — RDMA segment registration + accept client
 *
 * Usage:
 *   ./server [size_mb] [shmem_name] [tcp_port] [urma_port]
 *   Default: 128 engram_test 13900 13857
 *
 * Prerequisites:
 *   - ubse.service and ubsmd.service running
 *   - obmm kernel module loaded
 *   - numactl binding required (e.g., numactl --cpunodebind=0 --membind=0)
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
#include "../lib/urma_rw.h"

#define DEFAULT_TCP_PORT  13900
#define DEFAULT_URMA_PORT 13857

static volatile sig_atomic_t g_stop = 0;

static void sigint_handler(int sig)
{
    (void)sig;
    g_stop = 1;
}

/* ------------------------------------------------------------------ */
/*  TCP data server thread                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t offset;
    uint32_t length;
} __attribute__((packed)) tcp_req_t;

typedef struct {
    const char *data;
    size_t data_size;
    uint16_t port;
} tcp_args_t;

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

static void *tcp_thread(void *arg)
{
    tcp_args_t *a = (tcp_args_t *)arg;

    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd < 0) { perror("tcp socket"); return NULL; }

    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(a->port);

    if (bind(listenfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("tcp bind"); close(listenfd); return NULL;
    }
    listen(listenfd, 2);
    printf("[TCP] Listening on port %u\n", a->port);

    while (!g_stop) {
        struct sockaddr_in cli;
        socklen_t clen = sizeof(cli);
        int connfd = accept(listenfd, (struct sockaddr *)&cli, &clen);
        if (connfd < 0) { if (g_stop) break; continue; }

        int flag = 1;
        setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        printf("[TCP] Client %s:%d connected\n",
               inet_ntoa(cli.sin_addr), ntohs(cli.sin_port));

        while (!g_stop) {
            tcp_req_t req;
            if (recv_all(connfd, &req, sizeof(req)) != 0) break;
            if (req.offset == 0xFFFFFFFFFFFFFFFFULL) break;
            if (req.offset + req.length > a->data_size) {
                fprintf(stderr, "[TCP] OOB request\n");
                break;
            }
            if (send_all(connfd, a->data + req.offset, req.length) != 0) break;
        }
        printf("[TCP] Client disconnected\n");
        close(connfd);
    }
    close(listenfd);
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  URMA server thread                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    float *data;
    size_t data_size;
    uint16_t port;
} urma_args_t;

static void *urma_thread(void *arg)
{
    urma_args_t *a = (urma_args_t *)arg;

    urma_rw_ctx_t *ctx = urma_rw_init(NULL, a->data_size);
    if (!ctx) {
        fprintf(stderr, "[URMA] urma_rw_init failed\n");
        return NULL;
    }

    /* Copy data into URMA registered buffer (must be in registered memory) */
    float *urma_buf = (float *)urma_rw_get_buffer(ctx);
    memcpy(urma_buf, a->data, a->data_size);
    printf("[URMA] Buffer filled, %zu MB\n", a->data_size / (1024*1024));

    printf("[URMA] Waiting for client on port %u...\n", a->port);
    if (urma_rw_server_accept(ctx, a->port) != 0) {
        fprintf(stderr, "[URMA] server_accept failed\n");
        urma_rw_destroy(ctx);
        return NULL;
    }
    printf("[URMA] Client connected\n");

    while (!g_stop) {
        sleep(1);
    }

    urma_rw_destroy(ctx);
    printf("[URMA] Stopped\n");
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
    uint16_t urma_port = DEFAULT_URMA_PORT;

    if (argc > 1) size_mb = (size_t)atol(argv[1]);
    if (argc > 2) shm_name = argv[2];
    if (argc > 3) tcp_port = (uint16_t)atoi(argv[3]);
    if (argc > 4) urma_port = (uint16_t)atoi(argv[4]);

    size_t buf_size = size_mb * 1024 * 1024;

    printf("=== Unified Engram Server ===\n");
    printf("shmem: %s, size: %zu MB, tcp: %u, urma: %u\n",
           shm_name, size_mb, tcp_port, urma_port);

    /* Step 1: ubs_mem initialize */
    ubsmem_options_t opts;
    if (ubsmem_init_attributes(&opts) != 0 || ubsmem_initialize(&opts) != 0) {
        fprintf(stderr, "ubsmem_initialize failed\n");
        return 1;
    }
    ubsmem_set_logger_level(2);
    printf("[1/4] ubs_mem OK\n");

    /* Step 2: Cluster info */
    ubsmem_cluster_info_t cinfo;
    if (ubsmem_lookup_cluster_statistic(&cinfo) == 0) {
        printf("[2/4] Cluster: %d hosts\n", cinfo.host_num);
        for (int h = 0; h < cinfo.host_num; h++)
            printf("  host[%d]: %s\n", h, cinfo.host[h].host_name);
    }

    /* Step 3: Allocate + map shmem */
    const char *region_name = "default";
    ubsmem_shmem_deallocate(shm_name);

    int ret = ubsmem_shmem_allocate(region_name, shm_name, buf_size,
                                    0666, UBSM_FLAG_CACHE);
    if (ret != 0) {
        fprintf(stderr, "[3/4] shmem_allocate failed: %d\n", ret);
        ubsmem_finalize();
        return 1;
    }

    void *ptr = NULL;
    ret = ubsmem_shmem_map(NULL, buf_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, shm_name, 0, &ptr);
    if (ret != 0 || !ptr) {
        fprintf(stderr, "[3/4] shmem_map failed: %d\n", ret);
        ubsmem_shmem_deallocate(shm_name);
        ubsmem_finalize();
        return 1;
    }
    printf("[3/4] shmem OK: %s, ptr=%p\n", shm_name, ptr);

    /* Step 4: Fill data */
    float *data = (float *)ptr;
    size_t num_floats = buf_size / sizeof(float);
    for (size_t i = 0; i < num_floats; i++)
        data[i] = (float)i * 0.001f;
    printf("[4/4] Data filled: %zu floats\n", num_floats);

    /* Start TCP thread */
    tcp_args_t ta = { (const char *)data, buf_size, tcp_port };
    pthread_t tcp_tid;
    pthread_create(&tcp_tid, NULL, tcp_thread, &ta);

    /* Start URMA thread */
    urma_args_t ua = { data, buf_size, urma_port };
    pthread_t urma_tid;
    pthread_create(&urma_tid, NULL, urma_thread, &ua);

    signal(SIGINT, sigint_handler);
    printf("\nServer ready. Ctrl+C to stop.\n");
    printf("  shmem: %s (%zu MB)\n", shm_name, size_mb);
    printf("  TCP:   :%u\n", tcp_port);
    printf("  URMA:  :%u\n\n", urma_port);

    while (!g_stop) sleep(1);

    printf("\nShutting down...\n");
    pthread_cancel(tcp_tid);
    pthread_join(tcp_tid, NULL);
    /* URMA thread will exit on g_stop */
    pthread_join(urma_tid, NULL);

    ubsmem_shmem_unmap(ptr, buf_size);
    ubsmem_shmem_deallocate(shm_name);
    ubsmem_finalize();
    printf("Done.\n");
    return 0;
}
