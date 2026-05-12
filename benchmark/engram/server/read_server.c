/**
 * Lightweight Engram read server.
 *
 * This server is for TCP and URMA READ benchmarks only. It allocates a local
 * DRAM buffer, fills it with the same deterministic test pattern as the unified
 * server, and exposes it over TCP and/or URMA. It deliberately avoids UBS-MEM
 * initialization so TCP/URMA read tests can run even when ubsmd is unhealthy.
 *
 * Usage:
 *   READ_SERVER_MODES=tcp,urma ./read_server [size_mb] [name] [tcp_port] [urma_port]
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../lib/urma_rw.h"

#define DEFAULT_TCP_PORT  13900
#define DEFAULT_URMA_PORT 13857

static volatile sig_atomic_t g_stop = 0;

static void sigint_handler(int sig)
{
    (void)sig;
    g_stop = 1;
}

static bool mode_enabled(const char *modes, const char *token)
{
    if (!modes || modes[0] == '\0' || strcmp(modes, "all") == 0) {
        return true;
    }

    size_t token_len = strlen(token);
    const char *p = modes;
    while (*p) {
        while (*p == ',' || *p == ' ' || *p == '\t') {
            p++;
        }
        const char *start = p;
        while (*p && *p != ',' && *p != ' ' && *p != '\t') {
            p++;
        }
        if ((size_t)(p - start) == token_len &&
            strncmp(start, token, token_len) == 0) {
            return true;
        }
    }
    return false;
}

static void fill_data(float *buf, size_t bytes)
{
    size_t n = bytes / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        buf[i] = (float)i * 0.001f;
    }
}

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
        if (n <= 0) {
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static int send_all(int fd, const void *buf, size_t len)
{
    size_t done = 0;
    while (done < len) {
        ssize_t n = send(fd, (const char *)buf + done, len - done, 0);
        if (n <= 0) {
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static void *tcp_thread(void *arg)
{
    tcp_args_t *a = (tcp_args_t *)arg;

    int listenfd = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd < 0) {
        perror("tcp socket");
        return NULL;
    }

    int opt = 1;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(a->port);

    if (bind(listenfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("tcp bind");
        close(listenfd);
        return NULL;
    }
    listen(listenfd, 2);
    printf("[TCP] Listening on port %u\n", a->port);

    while (!g_stop) {
        struct sockaddr_in cli;
        socklen_t clen = sizeof(cli);
        int connfd = accept(listenfd, (struct sockaddr *)&cli, &clen);
        if (connfd < 0) {
            if (g_stop) {
                break;
            }
            continue;
        }

        int flag = 1;
        setsockopt(connfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
        printf("[TCP] Client %s:%d connected\n",
               inet_ntoa(cli.sin_addr), ntohs(cli.sin_port));

        while (!g_stop) {
            tcp_req_t req;
            if (recv_all(connfd, &req, sizeof(req)) != 0) {
                break;
            }
            if (req.offset == 0xFFFFFFFFFFFFFFFFULL) {
                break;
            }
            if (req.offset + req.length > a->data_size) {
                fprintf(stderr, "[TCP] OOB request\n");
                break;
            }
            if (send_all(connfd, a->data + req.offset, req.length) != 0) {
                break;
            }
        }
        printf("[TCP] Client disconnected\n");
        close(connfd);
    }
    close(listenfd);
    return NULL;
}

typedef struct {
    float *data;
    size_t data_size;
    uint16_t port;
} urma_args_t;

static void *urma_thread(void *arg)
{
    urma_args_t *a = (urma_args_t *)arg;
    const char *dev_name = getenv("URMA_DEV");

    printf("[URMA] Device: %s\n", dev_name ? dev_name : "(auto)");
    urma_rw_ctx_t *ctx = urma_rw_init(dev_name, a->data_size);
    if (!ctx) {
        fprintf(stderr, "[URMA] urma_rw_init failed\n");
        return NULL;
    }

    float *urma_buf = (float *)urma_rw_get_buffer(ctx);
    memcpy(urma_buf, a->data, a->data_size);
    printf("[URMA] Buffer filled, %zu MB\n", a->data_size / (1024 * 1024));

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

int main(int argc, char *argv[])
{
    size_t size_mb = 128;
    const char *name = "engram_test";
    uint16_t tcp_port = DEFAULT_TCP_PORT;
    uint16_t urma_port = DEFAULT_URMA_PORT;

    if (argc > 1) {
        size_mb = (size_t)atol(argv[1]);
    }
    if (argc > 2) {
        name = argv[2];
    }
    if (argc > 3) {
        tcp_port = (uint16_t)atoi(argv[3]);
    }
    if (argc > 4) {
        urma_port = (uint16_t)atoi(argv[4]);
    }

    const char *modes = getenv("READ_SERVER_MODES");
    if (!modes || modes[0] == '\0') {
        modes = "tcp,urma";
    }
    bool enable_tcp = mode_enabled(modes, "tcp");
    bool enable_urma = mode_enabled(modes, "urma");
    if (!enable_tcp && !enable_urma) {
        fprintf(stderr, "No read server mode enabled: %s\n", modes);
        return 1;
    }

    size_t buf_size = size_mb * 1024ULL * 1024ULL;
    float *data = NULL;
    int ret = posix_memalign((void **)&data, 4096, buf_size);
    if (ret != 0 || !data) {
        fprintf(stderr, "posix_memalign failed: %d\n", ret);
        return 1;
    }
    fill_data(data, buf_size);

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    printf("=== Engram Lightweight Read Server ===\n");
    printf("name: %s, size: %zu MB, modes: %s\n", name, size_mb, modes);
    printf("TCP: %s", enable_tcp ? "enabled" : "disabled");
    if (enable_tcp) {
        printf(" (:%u)", tcp_port);
    }
    printf(", URMA: %s", enable_urma ? "enabled" : "disabled");
    if (enable_urma) {
        printf(" (:%u)", urma_port);
    }
    printf("\n");

    pthread_t tcp_tid;
    pthread_t urma_tid;
    bool tcp_started = false;
    bool urma_started = false;

    tcp_args_t ta = { (const char *)data, buf_size, tcp_port };
    if (enable_tcp) {
        if (pthread_create(&tcp_tid, NULL, tcp_thread, &ta) != 0) {
            fprintf(stderr, "pthread_create tcp failed\n");
            free(data);
            return 1;
        }
        tcp_started = true;
    }

    urma_args_t ua = { data, buf_size, urma_port };
    if (enable_urma) {
        if (pthread_create(&urma_tid, NULL, urma_thread, &ua) != 0) {
            fprintf(stderr, "pthread_create urma failed\n");
            g_stop = 1;
            if (tcp_started) {
                pthread_cancel(tcp_tid);
                pthread_join(tcp_tid, NULL);
            }
            free(data);
            return 1;
        }
        pthread_detach(urma_tid);
        urma_started = true;
    }

    printf("Server ready. Ctrl+C to stop.\n\n");
    while (!g_stop) {
        sleep(1);
    }

    printf("\nShutting down...\n");
    if (tcp_started) {
        pthread_cancel(tcp_tid);
        pthread_join(tcp_tid, NULL);
    }
    if (urma_started) {
        pthread_cancel(urma_tid);
    }
    if (!urma_started) {
        free(data);
    }
    printf("Done.\n");
    return 0;
}
