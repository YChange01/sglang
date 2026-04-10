/*
 * smoke_test.c — single-node loopback load/store test for obmm UAPI.
 *
 * Based on the flow reverse-engineered from the open-source ubs-mem
 * project (src/app_lib/mxm_shm_lib/RackMemShm.cpp:131-177 and
 * src/app_lib/common/rack_mem_libobmm.h:24-28). The new model is:
 *
 *     1. open /dev/obmm              — control device
 *     2. ioctl(OBMM_CMD_EXPORT)      — kernel allocates; returns mem_id
 *     3. open /dev/obmm_shmdev<mem_id>
 *                                    — per-mem_id data device, auto-
 *                                      created by the kernel on EXPORT
 *     4. mmap(fd_data, len, MAP_SHARED, offset=0 or HUGETLB_PMD)
 *                                    — offset is a FLAG, not a byte
 *                                      offset. Returns a real local VA.
 *     5. *(volatile T*)va = x         — direct load/store on UBMMU
 *     6. munmap / close / UNEXPORT   — cleanup
 *
 * Why loopback first: no cluster, no TCP, no partner node, no
 * libubse.so, no ubsmd. If this single-process test passes, the kernel
 * obmm → UBMMU → DRAM path is fully usable from pure C, and we only
 * need to add TCP handshake + IMPORT ioctl to go cross-node.
 *
 * What killed our previous attempts (now we know why):
 *   • EXPORT_PID (register existing VA): always ENOTSUP — kernel
 *     restricts that path to in-kernel callers (qemu/KVM/DPU).
 *   • mmap on /dev/obmm with byte offsets: wrong device AND wrong
 *     offset semantics. Data plane is /dev/obmm_shmdev<id>, and the
 *     offset encodes kernel flags (bit 63 = HUGETLB_PMD), not bytes.
 *
 * Run as root: sudo ./smoke_test [size_mb]
 *   size_mb defaults to 4 (the 4MB SDK-side alignment minimum);
 *   will be rounded up to the 4MB granule on input.
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
#include <sys/stat.h>

/* The GPL UAPI header installed by ub-pkg-mem.
 * Using the system header rather than a vendored copy so we always
 * match whatever kernel module is live on this box. */
#include <ub/obmm.h>

/* ---------- constants ---------- */

/* Kernel mempool granule per /sys/module/obmm/parameters/mem_allocator_granu
 * is 2MB, but ubs-mem's SDK enforces 4MB, and empirically 4MB is what
 * reliably works with this build. */
#define OBMM_ALIGN_BYTES   (4UL * 1024UL * 1024UL)
#define ALIGN_UP(x, a)     (((x) + (a) - 1UL) & ~((a) - 1UL))

/* Max path for /dev/obmm_shmdev<mem_id>. ubs-mem uses 64. */
#define SHMDEV_PATH_LEN    64

/* ---------- helpers ---------- */

static void die_errno(const char *what)
{
    fprintf(stderr, "[FAIL] %s: %s (errno=%d)\n", what, strerror(errno), errno);
    exit(1);
}

/* Build "/dev/obmm_shmdev<mem_id>" — format matches ubs-mem/
 * src/app_lib/common/rack_mem_libobmm.h:26 exactly. */
static void build_shmdev_path(uint64_t mem_id, char *out, size_t cap)
{
    int n = snprintf(out, cap, "/dev/obmm_shmdev%" PRIu64, mem_id);
    if (n < 0 || (size_t)n >= cap) {
        fprintf(stderr, "[FAIL] shmdev path overflow for mem_id=%" PRIu64 "\n", mem_id);
        exit(1);
    }
}

static double time_diff_us(const struct timespec *t0, const struct timespec *t1)
{
    return (t1->tv_sec - t0->tv_sec) * 1.0e6
         + (t1->tv_nsec - t0->tv_nsec) / 1.0e3;
}

/* Fill `va` with a deterministic 32-bit pattern so we can verify
 * reads. Pattern: word[i] = 0xdead0000 + i * 0x1111. */
static void fill_pattern(volatile uint32_t *va, size_t n_words)
{
    for (size_t i = 0; i < n_words; i++) {
        va[i] = (uint32_t)(0xdead0000u + (uint32_t)i * 0x1111u);
    }
}

static int verify_pattern(const volatile uint32_t *va, size_t n_words)
{
    int errors = 0;
    for (size_t i = 0; i < n_words; i++) {
        uint32_t want = (uint32_t)(0xdead0000u + (uint32_t)i * 0x1111u);
        uint32_t got  = va[i];
        if (got != want) {
            if (errors < 4) {
                fprintf(stderr, "  word[%zu] want=0x%08x got=0x%08x\n",
                        i, want, got);
            }
            errors++;
        }
    }
    return errors;
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    /* 0. parse args */
    size_t size_mb = (argc > 1) ? (size_t)strtoul(argv[1], NULL, 10) : 4;
    if (size_mb == 0) size_mb = 4;
    size_t length = ALIGN_UP(size_mb * 1024UL * 1024UL, OBMM_ALIGN_BYTES);

    printf("=== scheme7 obmm loopback smoke test ===\n");
    printf("  buffer = %.2f MB (4MB-aligned, %zu bytes)\n",
           length / (1024.0 * 1024.0), length);
    printf("  strategy: EXPORT → /dev/obmm_shmdev<id> → mmap → load/store\n\n");

    /* 1. Open /dev/obmm (control device) */
    int fd_ctl = open("/dev/obmm", O_RDWR);
    if (fd_ctl < 0) die_errno("[1] open /dev/obmm");
    printf("[1] /dev/obmm -> fd_ctl=%d\n", fd_ctl);

    /* 2. EXPORT: ask kernel to allocate `length` bytes on NUMA 0.
     *
     *   Per the installed obmm.h, obmm_cmd_export has a per-NUMA size
     *   array (size[OBMM_MAX_LOCAL_NUMA_NODES]) and a `length` field
     *   which is the COUNT of size[] entries used (max 16), NOT bytes.
     *   We set size[0] = full allocation and length = 1.
     *
     *   The only flag we need for load/store access is ALLOW_MMAP.
     *   Everything else (cacheable, priv, seid/deid, pxm_numa, etc.)
     *   may or may not exist on this header version — we rely on the
     *   memset to zero them out to defaults. */
    struct obmm_cmd_export exp;
    memset(&exp, 0, sizeof(exp));
    exp.size[0] = length;
    exp.length  = 1;
    exp.flags   = OBMM_EXPORT_FLAG_ALLOW_MMAP;

    if (ioctl(fd_ctl, OBMM_CMD_EXPORT, &exp) < 0) {
        fprintf(stderr, "[FAIL 2] OBMM_CMD_EXPORT: %s (errno=%d)\n",
                strerror(errno), errno);
        close(fd_ctl);
        return 1;
    }
    printf("[2] EXPORT ok: mem_id=%" PRIu64 " (0x%" PRIx64 ")"
           "  tokenid=0x%x  uba=0x%" PRIx64 "\n",
           (uint64_t)exp.mem_id, (uint64_t)exp.mem_id,
           exp.tokenid, (uint64_t)exp.uba);

    /* 3. Open the per-mem_id shmdev data device.
     *
     *   Critical: this path is `/dev/obmm_shmdev<decimal_mem_id>`,
     *   no underscore, no separator. See ubs-mem/src/app_lib/common/
     *   rack_mem_libobmm.h:26 — it uses `sprintf_s(... "%lu", id)`.
     *
     *   The device is auto-created by the kernel when EXPORT succeeds,
     *   so it should be present by the time we open() it. If not, the
     *   kernel is either out of free major/minor slots or the mem_id
     *   registration race'd with us. */
    char shmdev_path[SHMDEV_PATH_LEN];
    build_shmdev_path((uint64_t)exp.mem_id, shmdev_path, sizeof(shmdev_path));
    printf("[3] opening %s ...\n", shmdev_path);

    /* stat it first so we see nicely whether it actually exists on disk */
    struct stat sb;
    if (stat(shmdev_path, &sb) == 0) {
        printf("    exists: mode=0%o, rdev=%lu\n",
               (unsigned)sb.st_mode & 07777, (unsigned long)sb.st_rdev);
    } else {
        fprintf(stderr, "    stat: %s (errno=%d)\n", strerror(errno), errno);
    }

    int fd_data = open(shmdev_path, O_RDWR);
    if (fd_data < 0) {
        fprintf(stderr, "[FAIL 3] open %s: %s (errno=%d)\n",
                shmdev_path, strerror(errno), errno);
        goto fail_after_export;
    }
    printf("    opened -> fd_data=%d\n", fd_data);

    /* 4. mmap with the flag-encoded offset. The offset parameter to
     *    mmap on /dev/obmm_shmdev<id> is NOT a byte offset; it is a
     *    bitfield of kernel flags.  Valid values seen in ubs-mem:
     *       0                           — normal (4KB pages)
     *       OBMM_MMAP_FLAG_HUGETLB_PMD  — 2MB huge pages (bit 63)
     *
     *    Try flags=0 first (simpler, no hugetlb requirement). */
    struct { off_t off; const char *name; } mmap_variants[] = {
        { 0,                             "offset=0 (normal pages)" },
        { (off_t)OBMM_MMAP_FLAG_HUGETLB_PMD, "offset=HUGETLB_PMD (bit63)" },
    };

    void *va = MAP_FAILED;
    size_t chosen = 0;
    for (size_t i = 0; i < sizeof(mmap_variants)/sizeof(mmap_variants[0]); i++) {
        printf("[4.%zu] mmap %s ... ", i, mmap_variants[i].name);
        fflush(stdout);

        void *attempt = mmap(NULL, length,
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED,
                             fd_data,
                             mmap_variants[i].off);
        if (attempt != MAP_FAILED) {
            printf("OK, va=%p\n", attempt);
            va = attempt;
            chosen = i;
            break;
        }
        printf("FAIL (%s)\n", strerror(errno));
    }

    if (va == MAP_FAILED) {
        fprintf(stderr, "[FAIL 4] all mmap variants rejected\n");
        fprintf(stderr, "         check dmesg for kernel obmm messages\n");
        close(fd_data);
        goto fail_after_export;
    }
    printf("    using variant #%zu\n", chosen);

    /* 5. Load/store verify. Touch every 4KB page to be sure the
     *    mapping is real (not lazy-faulted into nothing), then fill
     *    a 1 KB pattern at the start and verify read-back. */
    printf("[5] load/store test ...\n");
    {
        volatile uint8_t *bytes = (volatile uint8_t *)va;
        /* pre-touch every page */
        for (size_t off = 0; off < length; off += 4096) {
            bytes[off] = (uint8_t)(off >> 12);
        }
        /* read them back too, just to be sure the kernel didn't hand us
         * a write-only window somehow */
        volatile uint8_t acc = 0;
        for (size_t off = 0; off < length; off += 4096) {
            acc ^= bytes[off];
        }
        (void)acc;
    }

    const size_t n_pattern_words = 256;   /* 1 KB */
    volatile uint32_t *words = (volatile uint32_t *)va;
    fill_pattern(words, n_pattern_words);

    int errors = verify_pattern(words, n_pattern_words);
    if (errors == 0) {
        printf("    [PASS] 1 KB pattern roundtrip OK\n");
    } else {
        fprintf(stderr, "    [FAIL] %d pattern mismatches\n", errors);
    }

    /* 6. Micro-benchmark: pure local load latency on this VA.
     *    This is the floor — cross-node load/store should come within
     *    ~2x of this, otherwise the fabric is the bottleneck. */
    {
        struct timespec t0, t1;
        volatile uint64_t sink = 0;
        const size_t n_iters = 100000;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (size_t i = 0; i < n_iters; i++) {
            sink += ((volatile uint64_t *)va)[i % (length / 8)];
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double us = time_diff_us(&t0, &t1);
        printf("[6] micro-bench: %zu loads in %.2f us = %.2f ns/load"
               " (sink=%" PRIu64 ", dce-guard)\n",
               n_iters, us, us * 1000.0 / n_iters, sink);
    }

    /* 7. Cleanup: munmap, close data fd, UNEXPORT, close ctl fd */
    if (munmap(va, length) < 0) {
        fprintf(stderr, "[warn] munmap: %s\n", strerror(errno));
    }
    close(fd_data);

    {
        struct obmm_cmd_unexport un;
        memset(&un, 0, sizeof(un));
        un.mem_id = exp.mem_id;
        if (ioctl(fd_ctl, OBMM_CMD_UNEXPORT, &un) < 0) {
            fprintf(stderr, "[warn] UNEXPORT: %s\n", strerror(errno));
        } else {
            printf("[7] UNEXPORT ok\n");
        }
    }
    close(fd_ctl);

    if (errors == 0) {
        printf("\n=== smoke test PASSED ===\n");
        return 0;
    }
    fprintf(stderr, "\n=== smoke test FAILED (%d mismatches) ===\n", errors);
    return 2;

fail_after_export:
    {
        struct obmm_cmd_unexport un;
        memset(&un, 0, sizeof(un));
        un.mem_id = exp.mem_id;
        (void)ioctl(fd_ctl, OBMM_CMD_UNEXPORT, &un);
    }
    close(fd_ctl);
    fprintf(stderr, "\n=== smoke test FAILED at setup ===\n");
    return 1;
}
