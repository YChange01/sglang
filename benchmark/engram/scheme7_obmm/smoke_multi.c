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
    cmd.length   = length;
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

static int try_export_non_pid(int fd, size_t length, int32_t pxm_numa,
                              uint64_t flags,
                              const uint8_t seid[16], const uint8_t deid[16],
                              const char *label)
{
    struct obmm_cmd_export cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.size[0]  = length;     /* ask kernel to allocate from NUMA 0 */
    cmd.length   = length;
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

    int any_pass = 0;

    /* ========== variant 1: MAP_ANONYMOUS|MAP_SHARED + prefault + pxm=0 + zero eid ========== */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_zero, eid_zero,
                "v1: ANON|SHARED|POPULATE, pxm=0, flags=ALLOW_MMAP, eid=0");
            munmap(va, LEN);
        } else {
            printf("  [SKIP] v1: mmap ANON|SHARED failed: %s\n", strerror(errno));
        }
    }

    /* ========== variant 2: same va, pxm_numa=-1 ========== */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, -1,
                OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_zero, eid_zero,
                "v2: same, pxm=-1");
            munmap(va, LEN);
        }
    }

    /* ========== variant 3: flags = ALLOW_MMAP | FAST ========== */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                OBMM_EXPORT_FLAG_ALLOW_MMAP | OBMM_EXPORT_FLAG_FAST,
                eid_zero, eid_zero,
                "v3: same, flags=ALLOW_MMAP|FAST");
            munmap(va, LEN);
        }
    }

    /* ========== variant 4: non-zero seid/deid ========== */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_fake, eid_fake,
                "v4: same, eid=0x11,0x22,... (fake)");
            munmap(va, LEN);
        }
    }

    /* ========== variant 5: memfd_create backing ========== */
    {
        int mfd = sys_memfd_create("obmm_smoke", MFD_CLOEXEC);
        if (mfd >= 0) {
            if (ftruncate(mfd, LEN) == 0) {
                void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                                MAP_SHARED, mfd, 0);
                if (va != MAP_FAILED) {
                    prefault(va, LEN);
                    any_pass |= try_export_pid(fd, va, LEN, 0,
                        OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_zero, eid_zero,
                        "v5: memfd_create backing, pxm=0");
                    munmap(va, LEN);
                } else {
                    printf("  [SKIP] v5: mmap(memfd) failed: %s\n", strerror(errno));
                }
            } else {
                printf("  [SKIP] v5: ftruncate(memfd) failed: %s\n", strerror(errno));
            }
            close(mfd);
        } else {
            printf("  [SKIP] v5: memfd_create failed: %s\n", strerror(errno));
        }
    }

    /* ========== variant 6: MAP_PRIVATE instead of MAP_SHARED ========== */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_PRIVATE|MAP_POPULATE, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_zero, eid_zero,
                "v6: ANON|PRIVATE|POPULATE");
            munmap(va, LEN);
        }
    }

    /* ========== variant 7: mlocked anonymous ========== */
    {
        void *va = mmap(NULL, LEN, PROT_READ|PROT_WRITE,
                        MAP_ANONYMOUS|MAP_SHARED|MAP_POPULATE|MAP_LOCKED, -1, 0);
        if (va != MAP_FAILED) {
            prefault(va, LEN);
            any_pass |= try_export_pid(fd, va, LEN, 0,
                OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_zero, eid_zero,
                "v7: same + MAP_LOCKED");
            munmap(va, LEN);
        } else {
            printf("  [SKIP] v7: mmap(MAP_LOCKED) failed: %s\n", strerror(errno));
        }
    }

    /* ========== variant 8: OBMM_CMD_EXPORT (non-pid), size[0]=LEN, pxm=0, eid=0 ========== */
    {
        any_pass |= try_export_non_pid(fd, LEN, 0,
            OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_zero, eid_zero,
            "v8: OBMM_CMD_EXPORT (non-pid), size[0]=LEN, pxm=0, eid=0");
    }

    /* ========== variant 9: OBMM_CMD_EXPORT, pxm=-1 ========== */
    {
        any_pass |= try_export_non_pid(fd, LEN, -1,
            OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_zero, eid_zero,
            "v9: OBMM_CMD_EXPORT, pxm=-1");
    }

    /* ========== variant 10: OBMM_CMD_EXPORT, fake eid, pxm=0 ========== */
    {
        any_pass |= try_export_non_pid(fd, LEN, 0,
            OBMM_EXPORT_FLAG_ALLOW_MMAP, eid_fake, eid_fake,
            "v10: OBMM_CMD_EXPORT, fake eid");
    }

    /* ========== variant 11: OBMM_CMD_EXPORT, flags=ALLOW_MMAP|FAST ========== */
    {
        any_pass |= try_export_non_pid(fd, LEN, 0,
            OBMM_EXPORT_FLAG_ALLOW_MMAP | OBMM_EXPORT_FLAG_FAST,
            eid_zero, eid_zero,
            "v11: OBMM_CMD_EXPORT, flags=ALLOW_MMAP|FAST");
    }

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
