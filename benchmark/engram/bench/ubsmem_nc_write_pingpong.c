/**
 * UB-MEM noncache write ping-pong microbenchmark.
 *
 * This measures the same kind of number as the vendor "write/read" puncture:
 * one side writes a message into the peer's imported noncache mapping, the peer
 * polls it and writes a response back, and the client reports 1/2 RTT.
 *
 * Each process allocates one local shared memory object:
 *   server: <name>_wr_s
 *   client: <name>_wr_c
 *
 * Both objects use UBSM_FLAG_ONLY_IMPORT_NONCACHE | UBSM_FLAG_MEM_ANONYMOUS:
 * export side stays cacheable, import side is noncache. This is the low-latency
 * write-puncture mode used by the standalone pingpong sample.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <ubs_mem.h>

#define DEFAULT_SHM_MB 4
#define DEFAULT_ITERS 100000
#define DEFAULT_WARMUP 20000
#define DEFAULT_BYTES 512
#define DEFAULT_MAP_RETRIES 60
#define DEFAULT_RETRY_SLEEP_MS 1000
#define CACHELINE_BYTES 64
#define CTRL_OFFSET 0
#define PAYLOAD_OFFSET CACHELINE_BYTES
#define FIRST_LINE_PAYLOAD_BYTES 48
#define MAX_PAYLOAD_BYTES 4096
#define MAX_MSG_BLOCKS \
    (1 + ((MAX_PAYLOAD_BYTES - FIRST_LINE_PAYLOAD_BYTES + CACHELINE_BYTES - 1) / CACHELINE_BYTES))

typedef struct {
    uint8_t bytes[CACHELINE_BYTES];
} __attribute__((aligned(CACHELINE_BYTES))) cacheline_t;

typedef struct {
    uint8_t payload[FIRST_LINE_PAYLOAD_BYTES];
    uint64_t seq;
    uint32_t len;
    uint32_t blocks;
} __attribute__((aligned(CACHELINE_BYTES))) ctrl_line_t;

typedef enum {
    ROLE_SERVER,
    ROLE_CLIENT,
} bench_role_t;

typedef struct {
    bench_role_t role;
    const char *name;
    const char *provider_host;
    size_t shm_size;
    uint32_t bytes;
    uint64_t iters;
    uint64_t warmup;
    int map_retries;
    int retry_sleep_ms;
} bench_opts_t;

typedef struct {
    void *local_addr;
    void *remote_addr;
    char local_name[MAX_SHM_NAME_LENGTH + 1];
    char remote_name[MAX_SHM_NAME_LENGTH + 1];
    size_t shm_size;
} shm_pair_t;

static void teardown_shm_pair(shm_pair_t *pair);

#if defined(__aarch64__)
static inline uint64_t rdcntvct(void)
{
    uint64_t v;
    __asm__ __volatile__("mrs %0, cntvct_el0" : "=r"(v));
    return v;
}

static inline uint64_t rdcntfrq(void)
{
    uint64_t f;
    __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(f));
    return f;
}
#else
static inline uint64_t rdcntvct(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline uint64_t rdcntfrq(void)
{
    return 1000000000ULL;
}
#endif

static int cmp_u64(const void *a, const void *b)
{
    uint64_t av = *(const uint64_t *)a;
    uint64_t bv = *(const uint64_t *)b;
    return (av > bv) - (av < bv);
}

static uint32_t blocks_for_bytes(uint32_t bytes)
{
    if (bytes <= FIRST_LINE_PAYLOAD_BYTES) {
        return 1;
    }
    uint32_t tail = bytes - FIRST_LINE_PAYLOAD_BYTES;
    return 1 + (tail + CACHELINE_BYTES - 1) / CACHELINE_BYTES;
}

static void sleep_ms(int ms)
{
    struct timespec ts = {
        .tv_sec = ms / 1000,
        .tv_nsec = (long)(ms % 1000) * 1000000L,
    };
    nanosleep(&ts, NULL);
}

static inline void store64b(void *dst, const void *src)
{
#if defined(__aarch64__)
    __asm__ __volatile__(
        "mov x12, %0\n"
        "mov x13, %1\n"
        "ldr x4, [x12]\n"
        "ldr x5, [x12, #8]\n"
        "ldr x6, [x12, #16]\n"
        "ldr x7, [x12, #24]\n"
        "ldr x8, [x12, #32]\n"
        "ldr x9, [x12, #40]\n"
        "ldr x10, [x12, #48]\n"
        "ldr x11, [x12, #56]\n"
        "st64b x4, [x13]\n"
        :
        : "r"(src), "r"(dst)
        : "memory", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13");
#else
    memcpy(dst, src, CACHELINE_BYTES);
#endif
}

static inline void publish_barrier(void)
{
#if defined(__aarch64__)
    __asm__ __volatile__("dmb oshst" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static void fill_payload(cacheline_t *payload, uint32_t blocks, uint64_t seq)
{
    for (uint32_t b = 0; b < blocks; b++) {
        uint64_t *q = (uint64_t *)payload[b].bytes;
        for (uint32_t i = 0; i < CACHELINE_BYTES / sizeof(uint64_t); i++) {
            q[i] = seq ^ ((uint64_t)b << 32) ^ i;
        }
    }
}

static void send_msg(void *remote_addr, uint64_t seq, uint32_t bytes,
                     const cacheline_t *payload)
{
    uint32_t blocks = blocks_for_bytes(bytes);
    uint8_t *base = (uint8_t *)remote_addr;

    for (uint32_t i = 1; i < blocks; i++) {
        store64b(base + PAYLOAD_OFFSET + (size_t)(i - 1) * CACHELINE_BYTES,
                 &payload[i]);
    }
    publish_barrier();

    ctrl_line_t ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    memcpy(ctrl.payload, payload[0].bytes, FIRST_LINE_PAYLOAD_BYTES);
    ctrl.seq = seq;
    ctrl.len = bytes;
    ctrl.blocks = blocks;
    store64b(base + CTRL_OFFSET, &ctrl);
}

static void wait_msg(const void *local_addr, uint64_t seq)
{
    const volatile ctrl_line_t *ctrl = (const volatile ctrl_line_t *)
        ((const uint8_t *)local_addr + CTRL_OFFSET);
    while (ctrl->seq != seq) {
        __asm__ __volatile__("" ::: "memory");
    }
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --role server|client [options]\n"
            "Options:\n"
            "  --name NAME             base shmem name (default: engram_test)\n"
            "  --bytes N               payload bytes per message (default: 512)\n"
            "  --iters N               measured ping-pong iterations (default: 100000)\n"
            "  --warmup N              warmup ping-pong iterations (default: 20000)\n"
            "  --size_mb N             each local shmem size in MB (default: 4)\n"
            "  --provider HOST         local provider hostname (default: gethostname)\n"
            "  --map_retries N         remote map retries (default: 60)\n"
            "  --retry_sleep_ms N      sleep per map retry (default: 1000)\n",
            prog);
}

static int parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return -1;
    *out = (uint64_t)v;
    return 0;
}

static int parse_opts(int argc, char **argv, bench_opts_t *opts)
{
    memset(opts, 0, sizeof(*opts));
    opts->role = ROLE_CLIENT;
    opts->name = "engram_test";
    opts->shm_size = DEFAULT_SHM_MB * 1024ULL * 1024ULL;
    opts->bytes = DEFAULT_BYTES;
    opts->iters = DEFAULT_ITERS;
    opts->warmup = DEFAULT_WARMUP;
    opts->map_retries = DEFAULT_MAP_RETRIES;
    opts->retry_sleep_ms = DEFAULT_RETRY_SLEEP_MS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--role") == 0 && i + 1 < argc) {
            const char *role = argv[++i];
            if (strcmp(role, "server") == 0) {
                opts->role = ROLE_SERVER;
            } else if (strcmp(role, "client") == 0) {
                opts->role = ROLE_CLIENT;
            } else {
                return -1;
            }
        } else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
            opts->name = argv[++i];
        } else if (strcmp(argv[i], "--provider") == 0 && i + 1 < argc) {
            opts->provider_host = argv[++i];
        } else if (strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            uint64_t v;
            if (parse_u64(argv[++i], &v) != 0 || v == 0 || v > MAX_PAYLOAD_BYTES) return -1;
            opts->bytes = (uint32_t)v;
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            if (parse_u64(argv[++i], &opts->iters) != 0 || opts->iters == 0) return -1;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (parse_u64(argv[++i], &opts->warmup) != 0) return -1;
        } else if (strcmp(argv[i], "--size_mb") == 0 && i + 1 < argc) {
            uint64_t v;
            if (parse_u64(argv[++i], &v) != 0 || v == 0) return -1;
            opts->shm_size = v * 1024ULL * 1024ULL;
        } else if (strcmp(argv[i], "--map_retries") == 0 && i + 1 < argc) {
            uint64_t v;
            if (parse_u64(argv[++i], &v) != 0 || v > INT_MAX) return -1;
            opts->map_retries = (int)v;
        } else if (strcmp(argv[i], "--retry_sleep_ms") == 0 && i + 1 < argc) {
            uint64_t v;
            if (parse_u64(argv[++i], &v) != 0 || v > INT_MAX) return -1;
            opts->retry_sleep_ms = (int)v;
        } else {
            return -1;
        }
    }

    size_t needed = PAYLOAD_OFFSET +
        (size_t)(blocks_for_bytes(opts->bytes) - 1) * CACHELINE_BYTES;
    if (opts->shm_size < needed) {
        fprintf(stderr, "shmem too small: need at least %zu bytes\n", needed);
        return -1;
    }
    return 0;
}

static int make_names(const bench_opts_t *opts, shm_pair_t *pair)
{
    const char *local_suffix = opts->role == ROLE_SERVER ? "_wr_s" : "_wr_c";
    const char *remote_suffix = opts->role == ROLE_SERVER ? "_wr_c" : "_wr_s";
    int n1 = snprintf(pair->local_name, sizeof(pair->local_name), "%s%s",
                      opts->name, local_suffix);
    int n2 = snprintf(pair->remote_name, sizeof(pair->remote_name), "%s%s",
                      opts->name, remote_suffix);
    if (n1 < 0 || n1 >= (int)sizeof(pair->local_name) ||
        n2 < 0 || n2 >= (int)sizeof(pair->remote_name)) {
        fprintf(stderr, "shmem name too long\n");
        return -1;
    }
    return 0;
}

static int setup_shm_pair(const bench_opts_t *opts, shm_pair_t *pair)
{
    memset(pair, 0, sizeof(*pair));
    pair->shm_size = opts->shm_size;
    if (make_names(opts, pair) != 0) return -1;

    char local_host[MAX_HOST_NAME_DESC_LENGTH] = {0};
    if (opts->provider_host) {
        snprintf(local_host, sizeof(local_host), "%s", opts->provider_host);
    } else if (gethostname(local_host, sizeof(local_host)) != 0) {
        perror("gethostname");
        return -1;
    }

    ubs_mem_provider_t provider;
    memset(&provider, 0, sizeof(provider));
    snprintf(provider.host_name, sizeof(provider.host_name), "%s", local_host);
    provider.socket_id = UINT32_MAX;
    provider.numa_id = UINT32_MAX;
    provider.port_id = UINT32_MAX;

    uint64_t flags = UBSM_FLAG_ONLY_IMPORT_NONCACHE | UBSM_FLAG_MEM_ANONYMOUS;

    ubsmem_shmem_deallocate(pair->local_name);
    int ret = ubsmem_shmem_allocate_with_provider(&provider, pair->local_name,
                                                  pair->shm_size, 0666, flags);
    if (ret != 0 && ret != UBSM_ERR_ALREADY_EXIST) {
        fprintf(stderr, "allocate %s failed: %d\n", pair->local_name, ret);
        return -1;
    }

    ret = ubsmem_shmem_map(NULL, pair->shm_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, pair->local_name, 0, &pair->local_addr);
    if (ret != 0 || !pair->local_addr) {
        fprintf(stderr, "map local %s failed: %d\n", pair->local_name, ret);
        return -1;
    }
    memset(pair->local_addr, 0, pair->shm_size);

    printf("local shmem:  %s host=%s ptr=%p flags=0x%lx\n",
           pair->local_name, local_host, pair->local_addr, (unsigned long)flags);
    printf("remote shmem: %s waiting", pair->remote_name);
    fflush(stdout);

    for (int i = 0; i < opts->map_retries; i++) {
        ret = ubsmem_shmem_map(NULL, pair->shm_size, PROT_READ | PROT_WRITE,
                               MAP_SHARED, pair->remote_name, 0, &pair->remote_addr);
        if (ret == 0 && pair->remote_addr) {
            printf("\nremote mapped: %s ptr=%p\n", pair->remote_name, pair->remote_addr);
            return 0;
        }
        printf(".");
        fflush(stdout);
        sleep_ms(opts->retry_sleep_ms);
    }
    printf("\n");
    fprintf(stderr, "map remote %s failed after %d retries\n",
            pair->remote_name, opts->map_retries);
    return -1;
}

static void teardown_shm_pair(shm_pair_t *pair)
{
    if (pair->remote_addr) {
        ubsmem_shmem_unmap(pair->remote_addr, pair->shm_size);
        pair->remote_addr = NULL;
    }
    if (pair->local_addr) {
        ubsmem_shmem_unmap(pair->local_addr, pair->shm_size);
        pair->local_addr = NULL;
    }
    if (pair->local_name[0] != '\0') {
        ubsmem_shmem_deallocate(pair->local_name);
    }
}

static void run_server(const bench_opts_t *opts, shm_pair_t *pair)
{
    uint64_t total = opts->warmup + opts->iters;
    uint32_t blocks = blocks_for_bytes(opts->bytes);
    cacheline_t payload[MAX_MSG_BLOCKS];
    memset(payload, 0, sizeof(payload));

    printf("server ready: bytes=%u st64b/msg=%u total=%lu\n",
           opts->bytes, blocks, (unsigned long)total);
    for (uint64_t seq = 1; seq <= total; seq++) {
        wait_msg(pair->local_addr, seq);
        fill_payload(payload, blocks, seq);
        send_msg(pair->remote_addr, seq, opts->bytes, payload);
    }
    printf("server done: handled %lu messages\n", (unsigned long)total);
}

static void run_client(const bench_opts_t *opts, shm_pair_t *pair)
{
    uint64_t total = opts->warmup + opts->iters;
    uint32_t blocks = blocks_for_bytes(opts->bytes);
    cacheline_t payload[MAX_MSG_BLOCKS];
    uint64_t *samples = (uint64_t *)calloc(opts->iters, sizeof(uint64_t));
    if (!samples) {
        fprintf(stderr, "calloc samples failed\n");
        return;
    }

    printf("client start: bytes=%u st64b/msg=%u warmup=%lu iters=%lu\n",
           opts->bytes, blocks, (unsigned long)opts->warmup,
           (unsigned long)opts->iters);

    for (uint64_t seq = 1; seq <= total; seq++) {
        fill_payload(payload, blocks, seq);
        uint64_t t0 = rdcntvct();
        send_msg(pair->remote_addr, seq, opts->bytes, payload);
        wait_msg(pair->local_addr, seq);
        uint64_t t1 = rdcntvct();
        if (seq > opts->warmup) {
            samples[seq - opts->warmup - 1] = t1 - t0;
        }
    }

    uint64_t frq = rdcntfrq();
    if (frq == 0) frq = 1;
    double ns_per_tick = 1e9 / (double)frq;

    long double sum = 0.0;
    uint64_t min = UINT64_MAX;
    uint64_t max = 0;
    for (uint64_t i = 0; i < opts->iters; i++) {
        uint64_t v = samples[i];
        sum += (long double)v;
        if (v < min) min = v;
        if (v > max) max = v;
    }

    qsort(samples, opts->iters, sizeof(uint64_t), cmp_u64);
    uint64_t p50 = samples[opts->iters / 2];
    uint64_t p99 = samples[(opts->iters * 99) / 100];
    double avg_rtt_ns = (double)(sum / opts->iters) * ns_per_tick;

    printf("\n[UB-MEM noncache write pingpong]\n");
    printf("  bytes=%u st64b/msg=%u iters=%lu warmup=%lu counter_freq=%lu\n",
           opts->bytes, blocks, (unsigned long)opts->iters,
           (unsigned long)opts->warmup, (unsigned long)frq);
    printf("  full RTT avg=%.1f ns min=%.1f ns p50=%.1f ns p99=%.1f ns max=%.1f ns\n",
           avg_rtt_ns, (double)min * ns_per_tick, (double)p50 * ns_per_tick,
           (double)p99 * ns_per_tick, (double)max * ns_per_tick);
    printf("  1/2 RTT avg=%.1f ns min=%.1f ns p50=%.1f ns p99=%.1f ns max=%.1f ns\n",
           avg_rtt_ns / 2.0, (double)min * ns_per_tick / 2.0,
           (double)p50 * ns_per_tick / 2.0,
           (double)p99 * ns_per_tick / 2.0,
           (double)max * ns_per_tick / 2.0);

    free(samples);
}

int main(int argc, char **argv)
{
    bench_opts_t opts;
    if (parse_opts(argc, argv, &opts) != 0) {
        usage(argv[0]);
        return 1;
    }

    if (opts.bytes > MAX_PAYLOAD_BYTES) {
        fprintf(stderr, "bytes must be <= %d\n", MAX_PAYLOAD_BYTES);
        return 1;
    }

    ubsmem_options_t init_opts;
    if (ubsmem_init_attributes(&init_opts) != 0 ||
        ubsmem_initialize(&init_opts) != 0) {
        fprintf(stderr, "ubsmem_initialize failed\n");
        return 1;
    }
    ubsmem_set_logger_level(2);

    printf("=== UB-MEM Noncache Write Pingpong ===\n");
    printf("role=%s name=%s bytes=%u shm=%zu MB\n",
           opts.role == ROLE_SERVER ? "server" : "client",
           opts.name, opts.bytes, opts.shm_size / (1024 * 1024));

    shm_pair_t pair;
    int rc = 0;
    if (setup_shm_pair(&opts, &pair) != 0) {
        teardown_shm_pair(&pair);
        rc = 1;
        goto OUT;
    }

    if (opts.role == ROLE_SERVER) {
        run_server(&opts, &pair);
    } else {
        run_client(&opts, &pair);
    }

    teardown_shm_pair(&pair);

OUT:
    ubsmem_finalize();
    return rc;
}
