/*
 * scheme7 server — cross-node Engram data provider via obmm direct UAPI.
 *
 * Usage:
 *   sudo ./server [port] [total_size_mb]
 *   Defaults: port=13857, size=2048 (2 GB)
 *
 * Flow:
 *   1. open /dev/obmm
 *   2. ioctl OBMM_CMD_EXPORT  → get mem_id + tokenid + uba
 *   3. open /dev/obmm_shmdev<mem_id>
 *   4. mmap(shmdev_fd)       → local VA we can fill
 *   5. fill with scheme5-compatible pattern: data[i] = i * 0.001f
 *   6. TCP listen
 *   7. On accept: send handle struct, close client conn, keep listening
 *   8. On SIGINT: munmap, close, UNEXPORT, exit
 *
 * This mirrors scheme5_urma_rw/server.c in structure so the bench
 * numbers come out in the same format and can be diff'd directly.
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
#include <signal.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <ub/obmm.h>

#define DEFAULT_PORT      13857
#define DEFAULT_SIZE_MB   2048UL
#define ALIGN_4MB         (4UL * 1024UL * 1024UL)
#define ALIGN_UP(x, a)    (((x) + (a) - 1UL) & ~((a) - 1UL))

/*
 * Wire format shared with client.c — keep the two definitions in sync.
 * 72 bytes, naturally aligned, same-endian only (aarch64 ↔ aarch64).
 */
struct scheme7_wire_handle {
    uint64_t uba;          /* remote UB fabric address (from EXPORT) */
    uint64_t pa;           /* physical address (from ADDR_QUERY) */
    uint64_t length;       /* bytes */
    uint32_t tokenid;      /* access credential (from EXPORT) */
    uint32_t scna;         /* source CNA (exporter's) */
    uint32_t dcna;         /* dest CNA (0 = any) */
    int32_t  pxm_numa;     /* NUMA hint */
    uint8_t  seid[16];     /* source endpoint ID (exporter's) */
    uint8_t  deid[16];     /* dest endpoint ID (0 = any) */
} __attribute__((packed));

static volatile sig_atomic_t g_stop = 0;

static void on_sigint(int sig) { (void)sig; g_stop = 1; }

/*
 * Read this node's primary CNA from sysfs. The kernel requires a
 * valid scna for cross-node IMPORT routing — zero gets rejected with
 * "0x0 is not a known scna".
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

static int tcp_listen(uint16_t port)
{
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); return -1; }

    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in sa = {0};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons(port);

    if (bind(s, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        perror("bind");
        close(s);
        return -1;
    }
    if (listen(s, 4) < 0) {
        perror("listen");
        close(s);
        return -1;
    }
    return s;
}

static int send_all(int fd, const void *buf, size_t n)
{
    const char *p = (const char *)buf;
    while (n > 0) {
        ssize_t k = send(fd, p, n, 0);
        if (k <= 0) {
            if (k < 0 && errno == EINTR) continue;
            return -1;
        }
        p += k;
        n -= (size_t)k;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    uint16_t port = DEFAULT_PORT;
    uint64_t size_mb = DEFAULT_SIZE_MB;

    if (argc > 1) port = (uint16_t)atoi(argv[1]);
    if (argc > 2) size_mb = strtoull(argv[2], NULL, 10);

    uint64_t length = ALIGN_UP(size_mb * 1024UL * 1024UL, ALIGN_4MB);

    printf("=== scheme7 obmm server ===\n");
    printf("  port    : %u\n", port);
    printf("  buffer  : %.2f MB (%" PRIu64 " bytes, 4MB-aligned)\n",
           length / (1024.0 * 1024.0), length);
    printf("\n");

    /* 1. open /dev/obmm (control) */
    int fd_ctl = open("/dev/obmm", O_RDWR);
    if (fd_ctl < 0) { perror("[1] open /dev/obmm"); return 1; }
    printf("[1] /dev/obmm fd_ctl=%d\n", fd_ctl);

    /* 2. EXPORT. Same all-zero approach as smoke_test.c: zero all
     *    identity fields; if the kernel rejects cross-node later we'll
     *    iterate with real scna/seid. Single-node loopback worked with
     *    all zeros so it's a reasonable starting point. */
    struct obmm_cmd_export exp;
    memset(&exp, 0, sizeof(exp));
    exp.size[0] = length;
    exp.length  = 1;
    exp.flags   = OBMM_EXPORT_FLAG_ALLOW_MMAP;
    /* Everything else (pxm_numa, seid, deid, vendor_info, priv) stays 0. */

    if (ioctl(fd_ctl, OBMM_CMD_EXPORT, &exp) < 0) {
        fprintf(stderr, "[FAIL 2] OBMM_CMD_EXPORT: %s\n", strerror(errno));
        close(fd_ctl);
        return 1;
    }
    printf("[2] EXPORT ok: mem_id=%" PRIu64 " tokenid=0x%x uba=0x%" PRIx64 "\n",
           (uint64_t)exp.mem_id, exp.tokenid, (uint64_t)exp.uba);

    /* 2b. ADDR_QUERY: translate (mem_id, offset=0) → PA.
     *     The consultant told us: "export出来的是 VA, import 的是 PA,
     *     要转一下". This is the "转一下" — convert UBA to PA so the
     *     client can use PA for DECLARE_PREIMPORT + IMPORT. */
    struct obmm_cmd_addr_query aq;
    memset(&aq, 0, sizeof(aq));
    aq.key_type = OBMM_QUERY_BY_ID_OFFSET;
    aq.mem_id   = exp.mem_id;
    aq.offset   = 0;
    if (ioctl(fd_ctl, OBMM_CMD_ADDR_QUERY, &aq) < 0) {
        fprintf(stderr, "[warn] ADDR_QUERY failed: %s (errno=%d) — pa will be 0\n",
                strerror(errno), errno);
        aq.pa = 0;
    } else {
        printf("[2b] ADDR_QUERY ok: pa=0x%" PRIx64 "\n", (uint64_t)aq.pa);
    }

    /* 3. open /dev/obmm_shmdev<mem_id> */
    char path[64];
    snprintf(path, sizeof(path), "/dev/obmm_shmdev%" PRIu64, (uint64_t)exp.mem_id);
    int fd_data = open(path, O_RDWR);
    if (fd_data < 0) {
        fprintf(stderr, "[FAIL 3] open %s: %s\n", path, strerror(errno));
        goto fail_export;
    }
    printf("[3] %s fd_data=%d\n", path, fd_data);

    /* 4. mmap local access */
    void *va = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED,
                    fd_data, 0);
    if (va == MAP_FAILED) {
        fprintf(stderr, "[FAIL 4] mmap: %s\n", strerror(errno));
        close(fd_data);
        goto fail_export;
    }
    printf("[4] mmap va=%p\n", va);

    /* 5. Fill pattern — MUST match scheme5_urma_rw/server.c so that
     *    client verify + benches compare apples-to-apples. */
    float *data = (float *)va;
    uint64_t num_floats = length / sizeof(float);
    printf("[5] filling %" PRIu64 " floats with pattern (data[i] = i * 0.001)\n",
           num_floats);
    for (uint64_t i = 0; i < num_floats; i++) {
        data[i] = (float)i * 0.001f;
    }
    printf("    sample values:\n");
    printf("      data[0]      = %.6f\n", data[0]);
    printf("      data[42]     = %.6f\n", data[42]);
    printf("      data[1000]   = %.6f\n", data[1000]);
    if (num_floats > 100000) {
        printf("      data[100000] = %.6f\n", data[100000]);
    }

    /* 6. TCP listen */
    int lsk = tcp_listen(port);
    if (lsk < 0) { goto fail_mmap; }
    printf("[6] listening on port %u ...\n", port);

    /* Read our local CNA from sysfs — needed for cross-node routing. */
    uint32_t local_cna = read_local_cna();
    printf("[*] local CNA = 0x%04x\n", local_cna);

    /* Prepare handle once; same bytes sent to every client. */
    struct scheme7_wire_handle wire;
    memset(&wire, 0, sizeof(wire));
    wire.uba      = (uint64_t)exp.uba;
    wire.pa       = (uint64_t)aq.pa;
    wire.length   = length;
    wire.tokenid  = exp.tokenid;
    wire.scna     = local_cna;   /* server's CNA — client uses as scna */
    wire.dcna     = 0;           /* client fills its own */
    wire.pxm_numa = exp.pxm_numa;
    memcpy(wire.seid, exp.seid, 16);
    memcpy(wire.deid, exp.deid, 16);

    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);

    printf("\nServer ready. Press Ctrl+C to stop.\n\n");

    /* 7. Accept loop — each client gets one handle, then we keep
     *    serving so multiple clients can attach to the same memory. */
    while (!g_stop) {
        struct sockaddr_in ca;
        socklen_t cl = sizeof(ca);
        int cs = accept(lsk, (struct sockaddr *)&ca, &cl);
        if (cs < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            break;
        }
        printf("  client %s:%u connected\n",
               inet_ntoa(ca.sin_addr), ntohs(ca.sin_port));

        if (send_all(cs, &wire, sizeof(wire)) != 0) {
            fprintf(stderr, "  send handle failed: %s\n", strerror(errno));
        } else {
            printf("  sent handle (uba=0x%" PRIx64 " tokenid=0x%x length=%" PRIu64 ")\n",
                   wire.uba, wire.tokenid, wire.length);
        }
        close(cs);
    }

    printf("\nStopping server...\n");
    close(lsk);

    /* 8. cleanup */
fail_mmap:
    munmap(va, length);
    close(fd_data);
fail_export:
    {
        struct obmm_cmd_unexport un;
        memset(&un, 0, sizeof(un));
        un.mem_id = exp.mem_id;
        if (ioctl(fd_ctl, OBMM_CMD_UNEXPORT, &un) < 0) {
            fprintf(stderr, "[warn] UNEXPORT: %s\n", strerror(errno));
        } else {
            printf("[cleanup] UNEXPORT ok\n");
        }
    }
    close(fd_ctl);
    printf("Server stopped.\n");
    return g_stop ? 0 : 1;
}
