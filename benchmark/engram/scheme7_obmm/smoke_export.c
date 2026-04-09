/*
 * smoke_export.c — find the mmap offset that makes an obmm-allocated
 * region visible to user-space.
 *
 * smoke_multi proved that OBMM_CMD_EXPORT with length=1 + size[0]=4MB
 * always succeeds. What it does NOT tell us is how to get a user-space
 * VA for the allocated region. mmap(2) on /dev/obmm needs an offset;
 * the driver interprets it. This probe tries a bunch of likely offsets
 * and reports which (if any) yields a usable VA.
 *
 * Flow per candidate:
 *   1. ioctl(EXPORT) — fresh region
 *   2. mmap(fd, offset=CANDIDATE) — try to map
 *   3. If mmap succeeded, write a pattern to the first KB, read it
 *      back, verify
 *   4. munmap + ioctl(UNEXPORT)
 *   5. Log pass/fail with the candidate offset
 *
 * The first candidate that yields a readable/writeable VA is the
 * correct offset convention for this driver. That unblocks the real
 * server/client data path.
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
#include <sys/ioctl.h>
#include <sys/mman.h>

#define OBMM_DEV "/dev/obmm"
#define LEN (4UL * 1024UL * 1024UL)

/* Do one EXPORT, return mem_id + tokenid + uba via out params. */
static int do_export(int fd, uint64_t *out_mem_id,
                     uint32_t *out_tokenid, uint64_t *out_uba)
{
    struct obmm_cmd_export cmd;
    memset(&cmd, 0, sizeof(cmd));
    cmd.size[0]  = LEN;
    cmd.length   = 1;
    cmd.flags    = OBMM_EXPORT_FLAG_ALLOW_MMAP;
    cmd.pxm_numa = 0;

    if (ioctl(fd, OBMM_CMD_EXPORT, &cmd) < 0) {
        fprintf(stderr, "  EXPORT failed: %s\n", strerror(errno));
        return -1;
    }
    *out_mem_id  = cmd.mem_id;
    *out_tokenid = cmd.tokenid;
    *out_uba     = cmd.uba;
    return 0;
}

static void do_unexport(int fd, uint64_t mem_id)
{
    struct obmm_cmd_unexport un;
    memset(&un, 0, sizeof(un));
    un.mem_id = mem_id;
    (void)ioctl(fd, OBMM_CMD_UNEXPORT, &un);
}

/* Attempt mmap with a given offset, test readability, report. Returns
 * 1 if mmap succeeded AND we could round-trip a pattern through it. */
static int try_offset(int fd, uint64_t mem_id, off_t offset, const char *label)
{
    void *va = mmap(NULL, LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    if (va == MAP_FAILED) {
        printf("  [FAIL] %-40s offset=0x%lx  mmap: %s\n",
               label, (unsigned long)offset, strerror(errno));
        return 0;
    }

    printf("  [ OK ] %-40s offset=0x%lx  va=%p\n",
           label, (unsigned long)offset, va);

    /* Write + read pattern to confirm the mapping is real. Wrap in
     * a signal-safe path: if the mapping is bogus the store will
     * SIGSEGV. We can't install a SIGSEGV handler without setjmp
     * gymnastics, so trust the kernel for now and log the attempt
     * before the write so any crash clearly names the candidate. */
    printf("         writing test pattern via mmap VA ... ");
    fflush(stdout);
    volatile uint32_t *w = (volatile uint32_t*)va;
    for (int i = 0; i < 16; i++) {
        w[i] = 0xa5a50000u | (uint32_t)i;
    }
    printf("done\n");

    printf("         reading back via same VA ... ");
    fflush(stdout);
    int ok = 1;
    for (int i = 0; i < 16; i++) {
        uint32_t expect = 0xa5a50000u | (uint32_t)i;
        uint32_t got = w[i];
        if (got != expect) {
            printf("MISMATCH at %d: want 0x%08x got 0x%08x\n",
                   i, expect, got);
            ok = 0;
            break;
        }
    }
    if (ok) printf("PASS\n");

    munmap(va, LEN);
    (void)mem_id;
    return ok;
}

int main(void)
{
    printf("=== scheme7 obmm EXPORT + mmap offset probe ===\n");
    printf("  length = %lu bytes (4 MB)\n\n", LEN);

    int fd = open(OBMM_DEV, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        fprintf(stderr, "open(%s) failed: %s\n", OBMM_DEV, strerror(errno));
        return 1;
    }

    uint64_t mem_id = 0, uba = 0;
    uint32_t tokenid = 0;

    /* Each candidate gets a fresh export / unexport cycle so one
     * failed mmap doesn't poison the next try. */
    struct { const char *label; off_t (*compute)(uint64_t, uint64_t); } cands[] = {
        { "offset=0",                  NULL },  /* special case */
        { "offset=uba",                NULL },
        { "offset=uba / PAGE_SIZE",    NULL },
        { "offset=mem_id",             NULL },
        { "offset=mem_id << 12",       NULL },
        { "offset=mem_id << PAGE_SHIFT", NULL },
        { "offset=tokenid",            NULL },
    };
    int nc = sizeof(cands) / sizeof(cands[0]);

    int winners = 0;
    for (int i = 0; i < nc; i++) {
        if (do_export(fd, &mem_id, &tokenid, &uba) < 0) {
            printf("  [FAIL] %s — export failed, skipping\n", cands[i].label);
            continue;
        }
        printf("\n[%d/%d] Fresh export: mem_id=0x%lx tokenid=0x%x uba=0x%lx\n",
               i + 1, nc,
               (unsigned long)mem_id, (unsigned)tokenid, (unsigned long)uba);

        off_t offset;
        switch (i) {
        case 0: offset = 0; break;
        case 1: offset = (off_t)uba; break;
        case 2: offset = (off_t)(uba / 4096); break;
        case 3: offset = (off_t)mem_id; break;
        case 4: offset = (off_t)(mem_id << 12); break;
        case 5: offset = (off_t)(mem_id << 12); break;  /* same as 4 */
        case 6: offset = (off_t)tokenid; break;
        default: offset = 0;
        }

        int won = try_offset(fd, mem_id, offset, cands[i].label);
        if (won) winners++;

        do_unexport(fd, mem_id);
    }

    printf("\n");
    printf("=== %d candidate(s) yielded a working mmap+pattern ===\n", winners);
    close(fd);
    return winners > 0 ? 0 : 1;
}
