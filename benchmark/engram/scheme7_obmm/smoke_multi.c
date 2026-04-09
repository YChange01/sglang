/*
 * smoke_multi.c — multi-variant diagnostic for OBMM_CMD_EXPORT_PID EINVAL.
 *
 * The basic `smoke_test` tries one configuration and fails with EINVAL
 * from the kernel. Since obmm.h gives us no field-level documentation,
 * we iterate: try multiple variants of EXPORT_PID (different backings,
 * different flags, different seid/deid patterns) and also the non-pid
 * OBMM_CMD_EXPORT path, logging which variant the kernel accepts.
 *
 * We still use the obmm_rw.h return codes and error strings, but we
 * call the raw ioctls directly here so each variant can tweak exactly
 * one field.
 *
 * Usage:
 *     sudo ./smoke_multi
 *
 * Output: one line per variant attempt with the result. The first
 * success short-circuits (we can always rerun if we want to see later
 * variants).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "obmm_rw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#define OBMM_DEV "/dev/obmm"
#define LEN (4UL * 1024UL * 1024UL)    /* 4 MB, the scheme6 alignment rule */

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

static int sys_memfd_create(const char *name, unsigned int flags)
{
    return (int)syscall(SYS_memfd_create, name, flags);
}

/* Fully prefault every 4KB page in [va, va+length). */
static void prefault(void *va, size_t length)
{
    volatile uint8_t *p = (volatile uint8_t*)va;
    for (size_t i = 0; i < length; i += 4096) p[i] = 0;
}

/* Pretty-print errno after an ioctl. */
static void report(const char *variant, int ok, int err)
{
    if (ok) {
        printf("  [PASS] %s\n", variant);
    } else {
        printf("  [FAIL] %-48s  errno=%d (%s)\n",
               variant, err, strerror(err));
    }
}

/* ================================================================== */
/*  Shared helper: try an EXPORT_PID variant with a prepared va        */
/* ================================================================== */

static int try_export_pid(int fd, void *va, size_t length,
                          int32_t pxm_numa, uint64_t flags,
                          const uint8_t seid[16], const uint8_t deid[16],
                          const char *label)
{
    struct obmm_cmd_export_pid cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.va       = va;
    cmd.length   = length;     /* EXPORT_PID: length IS byte count */
    cmd.pid      = getpid();
    cmd.flags    = flags;
    cmd.pxm_numa = pxm_numa;
    if (seid) memcpy(cmd.seid, seid, 16);
    if (deid) memcpy(cmd.deid, deid, 16);

    int rc = ioctl(fd, OBMM_CMD_EXPORT_PID, &cmd);
    int err = errno;
    report(label, rc >= 0, err);
    if (rc >= 0) {
        printf("        → mem_id=0x%lx tokenid=0x%x uba=0x%lx\n",
               (unsigned long)cmd.mem_id, (unsigned)cmd.tokenid,
               (unsigned long)cmd.uba);
        /* Clean up so subsequent variants don't collide. */
        struct obmm_cmd_unexport un = { .mem_id = cmd.mem_id, .flags = 0 };
        (void)ioctl(fd, OBMM_CMD_UNEXPORT, &un);
        return 1;
    }
    return 0;
}

/* OBMM_CMD_EXPORT (non-pid) — kernel allocates.
 *
 * Discovered via dmesg:
 *   "OBMM: Size list is too long: max=16, actual_length=4194304"
 *
 * Meaning: cmd.length is NOT the byte count — it's the number of
 * entries in cmd.size[] (max 16 = OBMM_MAX_LOCAL_NUMA_NODES).
 * cmd.size[i] is the byte count for NUMA i.
 *
 * So to request 4 MB on NUMA 0:
 *   cmd.length  = 1;             // one size[] entry
 *   cmd.size[0] = 4*1024*1024;   // 4 MB on NUMA 0
 */
static int try_export_non_pid(int fd, size_t byte_count, int32_t pxm_numa,
                              uint64_t flags,
                              const uint8_t seid[16], const uint8_t deid[16],
                              const char *label)
{
    struct obmm_cmd_export cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.size[0]  = byte_count;   /* bytes to alloc on NUMA 0 */
    cmd.length   = 1;            /* one valid entry in size[] */
    cmd.flags    = flags;
    cmd.pxm_numa = pxm_numa;
    if (seid) memcpy(cmd.seid, seid, 16);
    if (deid) memcpy(cmd.deid, deid, 16);

    int rc = ioctl(fd, OBMM_CMD_EXPORT, &cmd);
    int err = errno;
    report(label, rc >= 0, err);
    if (rc >= 0) {
        printf("        → mem_id=0x%lx tokenid=0x%x uba=0x%lx\n",
               (unsigned long)cmd.mem_id, (unsigned)cmd.tokenid,
               (unsigned long)cmd.uba);
        struct obmm_cmd_unexport un = { .mem_id = cmd.mem_id, .flags = 0 };
        (void)ioctl(fd, OBMM_CMD_UNEXPORT, &un);
        return 1;
    }
    return 0;
}

/* ================================================================== */
/*  Variants                                                           */
/* ================================================================== */

int main(void)
{
    printf("=== scheme7 obmm multi-variant diagnostic ===\n");
    printf("  length = %lu bytes (4 MB)\n\n", LEN);

    int fd = open(OBMM_DEV, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "open(%s) failed: %s\n", OBMM_DEV, strerror(errno));
        return 1;
    }

    uint8_t eid_zero[16] = {0};
    uint8_t eid_fake[16];
    for (int i = 0; i < 16; i++) eid_fake[i] = 0x11 * (i + 1);

    /* Knowledge from the first dmesg run:
     *   - "ALLOW_MMAP flag is not allowed in export_user_addr"
     *     → EXPORT_PID must NOT set OBMM_EXPORT_FLAG_ALLOW_MMAP
     *   - "Size list is too long: max=16, actual_length=4194304"
     *     → OBMM_CMD_EXPORT.length is the COUNT of size[] entries,
     *       not a byte count. Helper is already fixed.
     *
     * This second pass retries with the corrected semantics. */
    int any_pass = 0;

    /* ========== EXPORT_PID variants (no ALLOW_MMAP) ========== */

    /* v1: baseline with flags=0 */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                0, eid_zero, eid_zero,
                "v1: EXPORT_PID ANON|SHARED, pxm=0, flags=0, eid=0");
            munmap(va, LEN);
        } else {
            printf("  [SKIP] v1: mmap ANON|SHARED failed: %s\n", strerror(errno));
        }
    }

    /* v2: same but flags=FAST */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                OBMM_EXPORT_FLAG_FAST, eid_zero, eid_zero,
                "v2: EXPORT_PID same, flags=FAST");
            munmap(va, LEN);
        }
    }

    /* v3: ANON|PRIVATE backing */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_PRIVATE|MAP_POPULATE, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                0, eid_zero, eid_zero,
                "v3: EXPORT_PID ANON|PRIVATE, flags=0");
            munmap(va, LEN);
        }
    }

    /* v4: memfd backing */
    {
        int mfd = sys_memfd_create("obmm_smoke", MFD_CLOEXEC);
        if (mfd >= 0) {
            if (ftruncate(mfd, LEN) == 0) {
                void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                                MAP_SHARED, mfd, 0);
                if (va != MAP_FAILED) {
                    prefault(va, LEN);
                    any_pass |= try_export_pid(fd, va, LEN, 0,
                        0, eid_zero, eid_zero,
                        "v4: EXPORT_PID memfd backing, flags=0");
                    munmap(va, LEN);
                }
            }
            close(mfd);
        }
    }

    /* v5: MAP_LOCKED */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE|MAP_LOCKED, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                0, eid_zero, eid_zero,
                "v5: EXPORT_PID + MAP_LOCKED, flags=0");
            munmap(va, LEN);
        } else {
            printf("  [SKIP] v5: mmap(MAP_LOCKED) failed: %s\n", strerror(errno));
        }
    }

    /* v6: 2MB-aligned (obmm allocator granularity per sysfs) */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE|MAP_HUGETLB,
                        -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                0, eid_zero, eid_zero,
                "v6: EXPORT_PID + MAP_HUGETLB (2MB pages)");
            munmap(va, LEN);
        } else {
            printf("  [SKIP] v6: mmap(MAP_HUGETLB) failed: %s (nr_hugepages?)\n",
                   strerror(errno));
        }
    }

    /* ========== OBMM_CMD_EXPORT variants (kernel allocates) ========== */

    /* v7: baseline — 4 MB on NUMA 0, flags=0 */
    any_pass |= try_export_non_pid(fd, LEN, 0, 0, eid_zero, eid_zero,
        "v7: EXPORT size[0]=4MB, pxm=0, flags=0, eid=0");

    /* v8: with ALLOW_MMAP (should work — kernel is the allocator) */
    any_pass |= try_export_non_pid(fd, LEN, 0,
        OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_zero, eid_zero,
        "v8: EXPORT size[0]=4MB, flags=ALLOW_MMAP");

    /* v9: with FAST */
    any_pass |= try_export_non_pid(fd, LEN, 0,
        OBMM_EXPORT_FLAG_FAST, eid_zero, eid_zero,
        "v9: EXPORT size[0]=4MB, flags=FAST");

    /* v10: ALLOW_MMAP|FAST */
    any_pass |= try_export_non_pid(fd, LEN, 0,
        OBMM_EXPORT_FLAG_ALLOW_MMAP | OBMM_EXPORT_FLAG_FAST,
        eid_zero, eid_zero,
        "v10: EXPORT size[0]=4MB, flags=ALLOW_MMAP|FAST");

    /* v11: fake eid */
    any_pass |= try_export_non_pid(fd, LEN, 0,
        OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_fake, eid_fake,
        "v11: EXPORT flags=ALLOW_MMAP, fake eid");

    printf("\n");
    if (any_pass) {
        printf("  === at least one variant PASSED — see above for which ===\n");
        close(fd);
        return 0;
    } else {
        printf("  === ALL variants FAILED — next: read dmesg, examine /sys/module/obmm ===\n");
        close(fd);
        return 1;
    }
}
