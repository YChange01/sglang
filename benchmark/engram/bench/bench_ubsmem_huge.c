/**
 * UBS-MEM hugepage import-noncache load benchmark.
 *
 * Maps the server-created "<name>_huge" object and reads it with CPU
 * loads/memcpy. This uses the import-noncache + anonymous + 2MB hugepage flag
 * combination expected by the UBS-MEM hugepage path.
 */

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include <ubs_mem.h>

#define PAPER_SEGS_PER_TOKEN 8
#define PAPER_SEG_BYTES 320
#define BENCH_TITLE "UBS-MEM Hugepage Read Benchmark"
#define MODE_NAME "UBS-MEM (import noncache + 2MB hugepage)"
#define UBSMEM_FLAGS (UBSM_FLAG_ONLY_IMPORT_NONCACHE | \
                      UBSM_FLAG_MEM_ANONYMOUS | \
                      UBSM_FLAG_MMAP_HUGETLB_PMD)
#define FIXED_LOAD_64B 64

typedef struct {
    size_t size_mb;
    const char *name;
    int rows;
    int dim;
    int iters;
    const char *provider;
} bench_opts_t;

static double diff_us(const struct timespec *a, const struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1e6 + (b->tv_nsec - a->tv_nsec) / 1e3;
}

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

static inline uint64_t rdcntfrq(void) { return 1000000000ULL; }
#endif

static int cmp_u64(const void *a, const void *b)
{
    uint64_t av = *(const uint64_t *)a;
    uint64_t bv = *(const uint64_t *)b;
    return (av > bv) - (av < bv);
}

static uint64_t next_rand64(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static uint64_t random_aligned_offset(uint64_t *state, uint64_t data_size,
                                      uint32_t len, uint32_t align)
{
    uint64_t max_offset = data_size > len ? data_size - len : 0;
    uint64_t slots = max_offset / align;
    if (slots == 0) return 0;
    return (next_rand64(state) % slots) * align;
}

static double print_ns_stats(const char *label, uint64_t *samples, int count,
                             uint64_t counter_freq)
{
    if (count <= 0) return 0.0;
    long double sum = 0.0;
    uint64_t min = UINT64_MAX;
    uint64_t max = 0;
    for (int i = 0; i < count; i++) {
        uint64_t v = samples[i];
        sum += (long double)v;
        if (v < min) min = v;
        if (v > max) max = v;
    }

    qsort(samples, (size_t)count, sizeof(uint64_t), cmp_u64);
    double ns_per_tick = 1e9 / (double)counter_freq;
    double avg_ns = (double)(sum / count) * ns_per_tick;
    printf("    %s avg=%.1f ns min=%.1f ns p50=%.1f ns p99=%.1f ns max=%.1f ns\n",
           label, avg_ns, (double)min * ns_per_tick,
           (double)samples[count / 2] * ns_per_tick,
           (double)samples[(count * 99) / 100] * ns_per_tick,
           (double)max * ns_per_tick);
    return avg_ns;
}

static uint64_t load_u64_bytes(const void *ptr, uint32_t bytes)
{
    volatile const uint64_t *p = (volatile const uint64_t *)ptr;
    uint64_t acc = 0;
    for (uint32_t i = 0; i < bytes / sizeof(uint64_t); i++) {
        acc += p[i];
    }
    __asm__ __volatile__("" : "+r"(acc) :: "memory");
    return acc;
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --size_mb N   shmem size in MB (default 128)\n"
            "  --name NAME   base shmem name (default engram_test)\n"
            "  --rows N      number of rows (default 10000)\n"
            "  --dim N       embedding dimension (default 341)\n"
            "  --iters N     iterations per benchmark (default 500)\n"
            "  --provider H  ubs_mem provider hostname (default node1)\n",
            prog);
}

static int parse_opts(int argc, char **argv, bench_opts_t *opts)
{
    opts->size_mb = 128;
    opts->name = "engram_test";
    opts->rows = 10000;
    opts->dim = 341;
    opts->iters = 500;
    opts->provider = "node1";

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
        else if (strcmp(argv[i], "--name") == 0) opts->name = argv[++i];
        else if (strcmp(argv[i], "--rows") == 0) opts->rows = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dim") == 0) opts->dim = atoi(argv[++i]);
        else if (strcmp(argv[i], "--iters") == 0) opts->iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--provider") == 0) opts->provider = argv[++i];
        else if (strcmp(argv[i], "--server_ip") == 0 ||
                 strcmp(argv[i], "--tcp_port") == 0 ||
                 strcmp(argv[i], "--urma_port") == 0 ||
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

static void build_object_name(const char *base, char *out, size_t out_len)
{
    snprintf(out, out_len, "%s_huge", base);
}

static const float *setup_ubsmem(const bench_opts_t *opts, size_t buf_size,
                                 void **out_ptr, char *used_name, size_t name_len)
{
    build_object_name(opts->name, used_name, name_len);

    ubsmem_shmem_info_t info;
    int ret = ubsmem_shmem_lookup(used_name, &info);
    if (ret != 0) {
        ubs_mem_provider_t provider;
        memset(&provider, 0, sizeof(provider));
        snprintf(provider.host_name, sizeof(provider.host_name), "%s", opts->provider);
        provider.socket_id = UINT32_MAX;
        provider.numa_id = UINT32_MAX;
        provider.port_id = UINT32_MAX;

        ret = ubsmem_shmem_allocate_with_provider(&provider, used_name, buf_size,
                                                  0666, UBSMEM_FLAGS);
        if (ret != 0 && ret != UBSM_ERR_ALREADY_EXIST) {
            fprintf(stderr, "  ubsmem allocate_with_provider(%s, flags=0x%lx) failed: %d\n",
                    used_name, (unsigned long)UBSMEM_FLAGS, ret);
        }
    }

    void *ptr = NULL;
    ret = ubsmem_shmem_map(NULL, buf_size, PROT_READ, MAP_SHARED, used_name, 0, &ptr);
    if (ret != 0 || !ptr) {
        fprintf(stderr, "  ubsmem_shmem_map(%s) failed: %d\n", used_name, ret);
        return NULL;
    }

    printf("  ubs_mem mapped: name=%s, ptr=%p, flags=0x%lx\n",
           used_name, ptr, (unsigned long)UBSMEM_FLAGS);
    *out_ptr = ptr;
    return (const float *)ptr;
}

static void read_segments(const float *data, float *local_buf,
                          const uint64_t *offsets, int count)
{
    for (int i = 0; i < count; i++) {
        memcpy((char *)local_buf + (size_t)i * PAPER_SEG_BYTES,
               (const char *)data + offsets[i],
               PAPER_SEG_BYTES);
    }
}

static void bench_single_float(const float *data, int rows, int dim, int iters)
{
    printf("\n  [Single float load] 4 bytes, %d iters\n", iters);

    volatile float sink = 0;
    for (int i = 0; i < 100; i++) sink += data[i];

    size_t total_floats = (size_t)rows * dim;
    size_t *indices = (size_t *)malloc((size_t)iters * sizeof(size_t));
    if (!indices) return;

    uint64_t rng = 123;
    for (int i = 0; i < iters; i++) {
        indices[i] = (size_t)(next_rand64(&rng) % total_floats);
    }

    uint64_t *samples = (uint64_t *)calloc((size_t)iters, sizeof(uint64_t));
    if (!samples) {
        free(indices);
        return;
    }
    uint64_t frq = rdcntfrq();
    if (frq == 0) frq = 1;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = rdcntvct();
        sink = data[indices[i]];
        uint64_t t1 = rdcntvct();
        samples[i] = t1 - t0;
    }

    print_ns_stats("Latency:", samples, iters, frq);
    (void)sink;
    free(samples);
    free(indices);
}

static void bench_single_seg(const float *data, float *local_buf,
                             size_t data_size, int iters)
{
    printf("\n  [Single-seg read] %d bytes, %d iters\n", PAPER_SEG_BYTES, iters);

    uint64_t max_offset = data_size - PAPER_SEG_BYTES;
    uint64_t off0 = 0;
    for (int i = 0; i < 10; i++) {
        read_segments(data, local_buf, &off0, 1);
        off0 = (off0 + PAPER_SEG_BYTES) % max_offset;
    }

    uint64_t *offsets = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
    uint64_t *samples = (uint64_t *)calloc((size_t)iters, sizeof(uint64_t));
    if (!offsets || !samples) {
        free(offsets);
        free(samples);
        return;
    }
    uint64_t rng = 42;
    for (int i = 0; i < iters; i++) {
        offsets[i] = random_aligned_offset(&rng, data_size, PAPER_SEG_BYTES,
                                            PAPER_SEG_BYTES);
    }

    uint64_t frq = rdcntfrq();
    if (frq == 0) frq = 1;
    for (int i = 0; i < iters; i++) {
        uint64_t t0 = rdcntvct();
        read_segments(data, local_buf, &offsets[i], 1);
        uint64_t t1 = rdcntvct();
        samples[i] = t1 - t0;
    }

    double avg_ns = print_ns_stats("Latency:", samples, iters, frq);
    printf("    Throughput: %.1f MB/s\n",
           (double)PAPER_SEG_BYTES / (avg_ns / 1e9) / (1024.0 * 1024.0));
    free(samples);
    free(offsets);
}

static void bench_fixed_loads(const float *data, size_t data_size, int iters)
{
    const uint32_t sizes[] = {FIXED_LOAD_64B, PAPER_SEG_BYTES};
    uint64_t frq = rdcntfrq();
    if (frq == 0) frq = 1;

    printf("\n  [Fixed-width load puncture] load-only, %d iters\n", iters);
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        uint32_t bytes = sizes[s];
        if (data_size < bytes) continue;

        uint64_t *offsets = (uint64_t *)malloc((size_t)iters * sizeof(uint64_t));
        uint64_t *samples = (uint64_t *)calloc((size_t)iters, sizeof(uint64_t));
        if (!offsets || !samples) {
            free(offsets);
            free(samples);
            return;
        }

        uint64_t rng = 0xfeed0000ULL + bytes;
        for (int i = 0; i < iters; i++) {
            offsets[i] = random_aligned_offset(&rng, data_size, bytes, FIXED_LOAD_64B);
        }

        volatile uint64_t sink = 0;
        for (int i = 0; i < 100; i++) {
            sink ^= load_u64_bytes((const char *)data + offsets[i % iters], bytes);
        }
        for (int i = 0; i < iters; i++) {
            uint64_t t0 = rdcntvct();
            sink ^= load_u64_bytes((const char *)data + offsets[i], bytes);
            uint64_t t1 = rdcntvct();
            samples[i] = t1 - t0;
        }

        char label[64];
        snprintf(label, sizeof(label), "%uB load:", bytes);
        double avg_ns = print_ns_stats(label, samples, iters, frq);
        printf("    %uB throughput: %.1f MB/s\n",
               bytes, (double)bytes / (avg_ns / 1e9) / (1024.0 * 1024.0));
        (void)sink;
        free(samples);
        free(offsets);
    }
}

static void bench_engram_27b(const float *data, float *local_buf,
                             size_t data_size, int iters)
{
    int batch_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                         1024, 2048, 4096, 8192, 16384};
    int nbatches = (int)(sizeof(batch_sizes) / sizeof(batch_sizes[0]));

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
        uint64_t rng = 0x12345678ULL ^ (uint64_t)batch;
        for (int iter = 0; iter < iters; iter++) {
            for (int i = 0; i < total_segs; i++) {
                offsets[i] = random_aligned_offset(&rng, data_size,
                                                    PAPER_SEG_BYTES,
                                                    PAPER_SEG_BYTES);
            }
            clock_gettime(CLOCK_MONOTONIC, &t0);
            read_segments(data, local_buf, offsets, total_segs);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            total_us += diff_us(&t0, &t1);
        }

        double avg_us = total_us / iters;
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

static void bench_cold_hot(const float *data, float *local_buf, int rows, int dim)
{
    int row_bytes = dim * (int)sizeof(float);
    int cold_rows[] = {0, rows / 4, rows / 2, rows * 3 / 4, rows - 1};

    printf("\n  [Cold/Hot analysis] first access vs cached\n");
    printf("    %-12s %-12s %-12s\n", "Row", "Cold (us)", "Hot (us)");
    printf("    ------------------------------------\n");

    for (int i = 0; i < 5; i++) {
        int row = cold_rows[i];
        if (row < 0 || row >= rows) continue;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(local_buf, data + (size_t)row * dim, row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double cold_us = diff_us(&t0, &t1);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(local_buf, data + (size_t)row * dim, row_bytes);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double hot_us = diff_us(&t0, &t1);

        printf("    %-12d %-12.3f %-12.3f\n", row, cold_us, hot_us);
    }
}

int main(int argc, char **argv)
{
    bench_opts_t opts;
    if (parse_opts(argc, argv, &opts) != 0) return 1;

    size_t buf_size = opts.size_mb * 1024ULL * 1024ULL;
    size_t data_size = (size_t)opts.rows * opts.dim * sizeof(float);
    size_t local_buf_size = (size_t)16384 * PAPER_SEGS_PER_TOKEN * PAPER_SEG_BYTES;
    if (local_buf_size < buf_size) local_buf_size = buf_size;

    printf("=== %s ===\n", BENCH_TITLE);
    printf("Table: %d rows x %d dim = %d bytes/row\n",
           opts.rows, opts.dim, (int)(opts.dim * sizeof(float)));
    printf("shmem: %s_huge, size: %zu MB, iters: %d, provider: %s\n",
           opts.name, opts.size_mb, opts.iters, opts.provider);

    ubsmem_options_t init_opts;
    if (ubsmem_init_attributes(&init_opts) != 0 ||
        ubsmem_initialize(&init_opts) != 0) {
        fprintf(stderr, "[init] ubs_mem library FAILED\n");
        return 1;
    }
    ubsmem_set_logger_level(2);

    float *local_buf = (float *)malloc(local_buf_size);
    if (!local_buf) {
        perror("malloc local_buf");
        ubsmem_finalize();
        return 1;
    }

    void *mapped_ptr = NULL;
    char used_name[MAX_SHM_NAME_LENGTH + 1];
    const float *data = setup_ubsmem(&opts, buf_size, &mapped_ptr,
                                     used_name, sizeof(used_name));
    if (!data) {
        free(local_buf);
        ubsmem_finalize();
        return 1;
    }

    memcpy(local_buf, data + (size_t)42 * opts.dim, (size_t)opts.dim * sizeof(float));
    float expected = 42.0f * opts.dim * 0.001f;
    printf("  Verify row[42][0]: got=%.4f expect=%.4f %s\n",
           local_buf[0], expected,
           fabsf(local_buf[0] - expected) < 0.01f ? "OK" : "MISMATCH");

    printf("\n======== Mode: %s ========\n", MODE_NAME);
    bench_single_float(data, opts.rows, opts.dim, opts.iters);
    bench_single_seg(data, local_buf, data_size, opts.iters);
    bench_fixed_loads(data, data_size, opts.iters);
    bench_engram_27b(data, local_buf, data_size, opts.iters);
    bench_cold_hot(data, local_buf, opts.rows, opts.dim);

    ubsmem_shmem_unmap(mapped_ptr, buf_size);
    free(local_buf);
    ubsmem_finalize();
    printf("\n=== Benchmark complete ===\n");
    return 0;
}
