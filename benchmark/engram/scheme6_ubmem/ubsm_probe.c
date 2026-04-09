/*
 * ubsm_probe.c — Cluster sanity check for UB Shared Memory (ubs_mem) SDK.
 *
 * Purpose:
 *   Before building scheme6 in earnest, verify that the ubs_mem SDK on this
 *   machine is usable and that the cluster discovery sees both nodes. Run
 *   this probe on BOTH Node1 and Node2; if they each report >= 2 hosts in
 *   the cluster view, scheme6 can proceed directly.
 *
 * What it checks:
 *   [1] ubsmem_initialize()            — local ubsmd reachable, SDK init
 *   [2] ubsmem_local_nid_query()       — hardware supernode ID
 *   [3] ubsmem_lookup_cluster_statistic() — cluster membership + memory stats
 *   [4] ubsmem_lookup_regions()        — pre-existing named regions (optional)
 *   [5] ubsmem_finalize()              — clean shutdown
 *
 * Build:
 *   make
 *
 * Run (on each node):
 *   ./ubsm_probe
 *
 * Exit codes:
 *   0 — all checks succeeded AND cluster has >= 2 hosts (ready for scheme6)
 *   1 — an SDK call failed (see error output)
 *   2 — all SDK calls succeeded but cluster has < 2 hosts (not ready yet)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>

#include <ubs_mem.h>
#include <ubs_mem_def.h>

/* Map ubs_mem SDK return codes (from ubs_mem_def.h) to strings.
 * The SDK returns 0 on success, or one of the UBSM_ERR_* codes in 6010-6099. */
static const char* ubsm_strerror(int code)
{
    switch (code) {
    case 0:    return "UBSM_OK";
    case 6010: return "UBSM_ERR_PARAM_INVALID";
    case 6011: return "UBSM_ERR_NOPERM";
    case 6012: return "UBSM_ERR_MEMORY";
    case 6013: return "UBSM_ERR_UNIMPL";
    case 6014: return "UBSM_CHECK_RESOURCE_ERROR";
    case 6015: return "UBSM_ERR_MEMLIB";
    case 6016: return "UBSM_ERR_NO_NEEDED";
    case 6017: return "UBSM_ERR_BUSY";
    case 6020: return "UBSM_ERR_NOT_FOUND";
    case 6021: return "UBSM_ERR_ALREADY_EXIST";
    case 6022: return "UBSM_ERR_MALLOC_FAIL";
    case 6023: return "UBSM_ERR_RECORD";
    case 6024: return "UBSM_ERR_IN_USING";
    case 6040: return "UBSM_ERR_NET";
    case 6050: return "UBSM_ERR_UBSE";
    case 6051: return "UBSM_ERR_OBMM";
    case 6060: return "UBSM_ERR_LOCK_NOT_SUPPORTED";
    case 6061: return "UBSM_ERR_LOCK_ALREADY_LOCKED";
    case 6062: return "UBSM_ERR_DLOCK";
    case 6099: return "UBSM_ERR_BUFF";
    default:   return "UBSM_ERR_???";
    }
}

#define CHECK(expr, label) do {                                          \
    int _rc = (expr);                                                    \
    if (_rc != 0) {                                                      \
        fprintf(stderr,                                                  \
                "  [FAIL] %s: rc=%d (%s)\n",                             \
                (label), _rc, ubsm_strerror(_rc));                       \
        goto fail;                                                       \
    } else {                                                             \
        printf("  [ OK ] %s\n", (label));                                \
    }                                                                    \
} while (0)

int main(void)
{
    int initialized = 0;

    printf("=== ubs_mem SDK sanity probe ===\n");
    printf("  Header : /usr/local/ubs_mem/include/ubs_mem.h\n");
    printf("  Library: /usr/local/ubs_mem/lib/libubsm_sdk.so\n");
    printf("  Daemon : ubsmd.service (expected: active/running)\n");
    printf("\n");

    /* ---------------- [1] initialize ---------------- */
    printf("[1] ubsmem_initialize\n");
    ubsmem_options_t opts;
    memset(&opts, 0, sizeof(opts));

    /* init_attributes fills defaults. The struct is empty in this SDK build
     * (see ubs_mem_def.h comment "todo") so the call is effectively a no-op,
     * but we still invoke it so that when future SDK versions add fields the
     * probe picks them up automatically. */
    int rc_attr = ubsmem_init_attributes(&opts);
    if (rc_attr != 0) {
        fprintf(stderr,
                "  [WARN] ubsmem_init_attributes rc=%d (%s) — continuing with zero-filled opts\n",
                rc_attr, ubsm_strerror(rc_attr));
    }

    CHECK(ubsmem_initialize(&opts), "ubsmem_initialize");
    initialized = 1;

    /* ---------------- [2] local NID ---------------- */
    printf("\n[2] ubsmem_local_nid_query\n");
    uint32_t nid = 0xffffffffu;
    CHECK(ubsmem_local_nid_query(&nid), "ubsmem_local_nid_query");
    printf("       local supernode ID (nid) = %u (0x%x)\n", nid, nid);

    /* ---------------- [3] cluster statistics ---------------- */
    printf("\n[3] ubsmem_lookup_cluster_statistic\n");
    ubsmem_cluster_info_t ci;
    memset(&ci, 0, sizeof(ci));
    CHECK(ubsmem_lookup_cluster_statistic(&ci), "ubsmem_lookup_cluster_statistic");

    printf("       host_num = %d\n", ci.host_num);
    if (ci.host_num <= 0) {
        fprintf(stderr, "  [FAIL] cluster has 0 hosts — ubsmd discovery not ready\n");
        goto fail;
    }

    int cap = (ci.host_num < MAX_HOST_NUM) ? ci.host_num : MAX_HOST_NUM;
    for (int h = 0; h < cap; h++) {
        ubsmem_host_info_t *hi = &ci.host[h];
        printf("       host[%d] = \"%s\", numa_num=%d\n",
               h, hi->host_name, hi->numa_num);

        int numa_show = (hi->numa_num < 4) ? hi->numa_num : 4;
        for (int n = 0; n < numa_show; n++) {
            ubsmem_numa_mem_t *nm = &hi->numa[n];
            printf("         numa[%d] slot=%u socket=%u numa_id=%u "
                   "total=%.2f GB free=%.2f GB borrow=%.2f GB lend=%.2f GB\n",
                   n,
                   nm->slot_id, nm->socket_id, nm->numa_id,
                   nm->mem_total / 1e9,
                   nm->mem_free  / 1e9,
                   nm->mem_borrow / 1e9,
                   nm->mem_lend  / 1e9);
        }
        if (hi->numa_num > numa_show) {
            printf("         (... and %d more numa nodes)\n",
                   hi->numa_num - numa_show);
        }
    }

    int cluster_ok = (ci.host_num >= 2);
    if (!cluster_ok) {
        printf("\n  [WARN] only 1 host visible in cluster — cross-node scheme6 "
               "will NOT work from this node yet.\n");
        printf("         Possible causes:\n");
        printf("           - peer ubsmd not running / not reachable\n");
        printf("           - discovery not joined (check /etc/ubsmd or discovery config)\n");
        printf("           - UB fabric route missing\n");
    } else {
        printf("\n  [ OK ] cluster has %d hosts — cross-node path available.\n", ci.host_num);
    }

    /* ---------------- [4] pre-existing regions ---------------- */
    printf("\n[4] ubsmem_lookup_regions\n");
    ubsmem_regions_t regions;
    memset(&regions, 0, sizeof(regions));
    int rc_reg = ubsmem_lookup_regions(&regions);
    if (rc_reg != 0) {
        fprintf(stderr,
                "  [INFO] ubsmem_lookup_regions rc=%d (%s) — "
                "likely no pre-existing regions, which is fine for first run\n",
                rc_reg, ubsm_strerror(rc_reg));
    } else {
        printf("  [ OK ] ubsmem_lookup_regions rc=0, regions.num=%d\n", regions.num);
        int rcap = (regions.num < MAX_REGIONS_NUM) ? regions.num : MAX_REGIONS_NUM;
        for (int i = 0; i < rcap; i++) {
            printf("       region[%d]: host_num=%d\n",
                   i, regions.region[i].host_num);
        }
    }

    /* ---------------- [5] finalize ---------------- */
    printf("\n[5] ubsmem_finalize\n");
    CHECK(ubsmem_finalize(), "ubsmem_finalize");
    initialized = 0;

    printf("\n");
    if (cluster_ok) {
        printf("=== PROBE PASSED — scheme6 ready ===\n");
        return 0;
    } else {
        printf("=== PROBE PARTIAL — SDK works, cluster not yet multi-node ===\n");
        return 2;
    }

fail:
    fprintf(stderr, "\n=== PROBE FAILED ===\n");
    if (initialized) {
        (void)ubsmem_finalize();
    }
    return 1;
}
