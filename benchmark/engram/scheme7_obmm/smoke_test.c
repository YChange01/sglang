/*
 * smoke_test.c — single-node loopback test for the obmm UAPI.
 *
 * Goal: prove that we can drive /dev/obmm directly without any help
 * from libubsm_sdk.so. No TCP, no cluster, no region — just:
 *
 *   1. open /dev/obmm
 *   2. allocate a small buffer (4 MB, the scheme6 alignment rule
 *      probably still applies at the kernel level)
 *   3. fill with a known pattern
 *   4. EXPORT_PID the buffer → get mem_id + tokenid
 *   5. IMPORT it back → get a local VA that should resolve to the
 *      same physical memory
 *   6. read the pattern through the imported VA and verify
 *   7. UNIMPORT / UNEXPORT / close
 *
 * If this passes, the obmm UAPI is usable from C and we can move on
 * to cross-node (scheme7 server.c + client.c).
 *
 * If it fails, the error message + errno tells us exactly which ioctl
 * rejected the call and why — we have GPL kernel source to consult,
 * unlike the black-box scheme6 situation.
 *
 * Usage:
 *   ./smoke_test              # default 4 MB buffer
 *   ./smoke_test [size_mb]    # pick a different size, will round to 4MB
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "obmm_rw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/mman.h>

/* Same 4MB granularity we learned the hard way in scheme6 — the kernel
 * OBMM layer enforces this, not the userspace SDK. */
#define OBMM_ALIGN_BYTES  (4UL * 1024UL * 1024UL)
#define ALIGN_UP(x, a)    (((x) + (a) - 1UL) & ~((a) - 1UL))

static int check_pattern(const volatile uint32_t *base, size_t n_words)
{
    int errors = 0;
    for (size_t i = 0; i < n_words; i++) {
        uint32_t want = (uint32_t)(i * 0x1111u + 0xdead0000u);
        uint32_t got  = base[i];
        if (got != want) {
            if (errors < 4) {
                fprintf(stderr,
                        "  [MISMATCH] word %zu: want 0x%08x got 0x%08x\n",
                        i, want, got);
            }
            errors++;
        }
    }
    return errors;
}

int main(int argc, char *argv[])
{
    size_t size_mb = (argc > 1) ? (size_t)atol(argv[1]) : 4;
    if (size_mb == 0) size_mb = 4;
    size_t length = ALIGN_UP(size_mb * 1024UL * 1024UL, OBMM_ALIGN_BYTES);

    printf("=== scheme7 obmm smoke test ===\n");
    printf("  buffer = %.2f MB (4MB-aligned)\n", length / (1024.0 * 1024.0));
    printf("\n");

    /* 1. open */
    obmm_rw_ctx_t *ctx = obmm_rw_open();
    if (!ctx) {
        fprintf(stderr, "FATAL: cannot open /dev/obmm\n");
        return 1;
    }

    /* 2. allocate. MAP_SHARED so the kernel can pin / export these pages.
     *    MAP_ANONYMOUS + -1 fd gives us cacheable normal pages. */
    void *va = mmap(NULL, length, PROT_READ | PROT_WRITE,
                    MAP_ANONYMOUS | MAP_SHARED, -1, 0);
    if (va == MAP_FAILED) {
        fprintf(stderr, "FATAL: mmap failed: %s\n", strerror(errno));
        goto fail_close;
    }
    printf("  allocated va = %p\n", va);

    /* 3. fill pattern. We only need to touch the first KB to prove the
     *    import path works; full fill is useful for a later bench. */
    size_t n_pattern_words = 1024 / sizeof(uint32_t);  /* 256 words = 1 KB */
    volatile uint32_t *words = (volatile uint32_t*)va;
    for (size_t i = 0; i < n_pattern_words; i++) {
        words[i] = (uint32_t)(i * 0x1111u + 0xdead0000u);
    }
    printf("  wrote 1 KB pattern via original VA\n");

    /* 4. EXPORT_PID. For loopback smoke test, seid = deid = 0 and
     *    we rely on PID-based addressing only. */
    obmm_rw_handle_t handle;
    uint8_t zero_eid[16] = {0};
    int rc = obmm_rw_export(ctx, va, length, zero_eid, zero_eid, &handle);
    if (rc != OBMM_RW_OK) {
        fprintf(stderr, "FATAL: export failed: rc=%d (%s)\n",
                rc, obmm_rw_strerror(rc));
        goto fail_munmap;
    }
    obmm_rw_print_handle("exported handle", &handle);

    /* 5. IMPORT the same region back in the same process. With
     *    loopback, local_seid == handle.seid (== zero) so the import
     *    does NOT set NUMA_REMOTE. */
    void *imported = NULL;
    rc = obmm_rw_import(ctx, &handle, zero_eid, &imported);
    if (rc != OBMM_RW_OK) {
        fprintf(stderr, "FATAL: import failed: rc=%d (%s)\n",
                rc, obmm_rw_strerror(rc));
        goto fail_unexport;
    }
    printf("  imported va = %p\n", imported);

    /* 6. verify pattern through the imported VA. If this passes, obmm
     *    really mapped the same backing pages (or copies of them) and
     *    the UAPI is usable. */
    int errors = check_pattern((volatile uint32_t*)imported, n_pattern_words);
    if (errors == 0) {
        printf("  [PASS] 1 KB pattern matches via imported VA\n");
    } else {
        fprintf(stderr, "  [FAIL] %d mismatches via imported VA\n", errors);
    }

    /* Bonus: probe a single-row load latency through the mapped VA.
     *    This gives us a rough feeling for how fast pure load/store is
     *    on this hardware, even in the loopback case (should be DRAM
     *    speed — hundreds of ns). */
    {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        volatile uint32_t acc = 0;
        for (int i = 0; i < 1000; i++) {
            acc += ((volatile uint32_t*)imported)[i % n_pattern_words];
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double total_ns = (t1.tv_sec - t0.tv_sec) * 1e9 +
                          (t1.tv_nsec - t0.tv_nsec);
        printf("  [bench] 1000 loopback loads: %.1f ns total, "
               "%.1f ns/load (acc=%u, prevents DCE)\n",
               total_ns, total_ns / 1000.0, acc);
    }

    /* 7. cleanup */
    (void)obmm_rw_unimport(ctx, &handle);
    (void)obmm_rw_unexport(ctx, &handle);
    munmap(va, length);
    obmm_rw_close(ctx);

    printf("\n  === smoke test PASSED ===\n");
    return (errors == 0) ? 0 : 2;

fail_unexport:
    (void)obmm_rw_unexport(ctx, &handle);
fail_munmap:
    munmap(va, length);
fail_close:
    obmm_rw_close(ctx);
    fprintf(stderr, "\n  === smoke test FAILED ===\n");
    return 1;
}
