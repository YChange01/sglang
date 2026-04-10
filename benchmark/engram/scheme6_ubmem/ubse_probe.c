/*
 * ubse_probe.c — bypass ubsmd, directly probe UBSE allocation paths.
 *
 * Tests MULTIPLE allocation strategies to find one that works:
 *   A. ubs_mem_shm_create (flag=0)              — original attempt
 *   B. ubs_mem_shm_create (flag=CACHEABLE)       — different cache mode
 *   C. ubs_mem_shm_create (flag=CACHEABLE|NO_WR) — disable write-relay
 *   D. ubs_mem_fd_create  (distance=L0)          — lease model, different code path
 *
 * Compile: gcc -o ubse_probe ubse_probe.c -ldl
 * Run:     sudo ./ubse_probe
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <netinet/in.h>

/* ---- Minimal struct definitions from UBSE headers ---- */

#define UBS_TOPO_SOCKET_NUM   2
#define UBS_TOPO_IPADDR_NUM  50
#define UBS_TOPO_NUMA_NUM     4
#define UBS_MEM_MAX_SLOT_NUM 16
#define UBS_MEM_MAX_MEMID_NUM 2048
#define UBS_MEM_MAX_NAME_LENGTH 48
#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 64
#endif

/* flags for shm_create */
#define UBS_MEM_FLAG_NO_WR_DELAY  0x1
#define UBS_MEM_FLAG_SHM_ANONYMOUS 0x2
#define UBS_MEM_FLAG_CACHEABLE    0x4

/* distance for fd_create */
typedef enum { MEM_DISTANCE_L0 = 0 } ubs_mem_distance_t;

typedef struct {
    int32_t af;
    struct in_addr ipv4;
    struct in6_addr ipv6;
} ubs_topo_ip_address_t;

typedef struct {
    uint32_t slot_id;
    uint32_t socket_id[UBS_TOPO_SOCKET_NUM];
    uint32_t numa_ids[UBS_TOPO_SOCKET_NUM][UBS_TOPO_NUMA_NUM];
    ubs_topo_ip_address_t ips[UBS_TOPO_IPADDR_NUM];
    char host_name[HOST_NAME_MAX];
} ubs_topo_node_t;

typedef struct {
    uint32_t node_cnt;
    uint32_t slot_ids[UBS_MEM_MAX_SLOT_NUM];
} ubs_mem_nodes_t;

typedef struct {
    uid_t uid;
    gid_t gid;
    pid_t pid;
} ubs_mem_fd_owner_t;

typedef struct {
    char name[UBS_MEM_MAX_NAME_LENGTH];
    uint32_t memid_cnt;
    uint64_t memids[UBS_MEM_MAX_MEMID_NUM];
    uint64_t mem_size;
    uint64_t unit_size;
    /* there may be more fields but we only need memid_cnt + memids */
    char _pad[4096];
} ubs_mem_fd_desc_t;

/* ---- Function pointer types ---- */

typedef int32_t (*fn_client_init)(const char *);
typedef void    (*fn_client_fini)(void);
typedef int32_t (*fn_topo_list)(ubs_topo_node_t **, uint32_t *);
typedef int32_t (*fn_topo_local)(ubs_topo_node_t *);
typedef int32_t (*fn_shm_create)(const char *, uint64_t, uint8_t[32],
                                 uint64_t, const ubs_mem_nodes_t *,
                                 const ubs_mem_nodes_t *);
typedef int32_t (*fn_shm_delete)(const char *);
typedef int32_t (*fn_fd_create)(const char *, uint64_t,
                                const ubs_mem_fd_owner_t *, unsigned int,
                                ubs_mem_distance_t, ubs_mem_fd_desc_t *);
typedef int32_t (*fn_fd_delete)(const char *);

/* ---- helpers ---- */

static const char *err_name(int32_t rc) {
    switch (rc) {
    case 0:    return "OK";
    case 1002: return "CONNECTION_FAILED";
    case 1005: return "INTERNAL";
    case 1006: return "EXISTED";
    case 1007: return "NOT_EXIST";
    case 1013: return "ALLOCATE";
    case 1014: return "SHM_NO_CREATE";
    default:   return "?";
    }
}

/* ---- main ---- */

int main(void)
{
    printf("=== UBSE multi-path probe ===\n\n");

    void *h = dlopen("libubse-client.so", RTLD_NOW);
    if (!h) { fprintf(stderr, "[FAIL] dlopen: %s\n", dlerror()); return 1; }

    fn_client_init p_init  = (fn_client_init) dlsym(h, "ubs_engine_client_initialize");
    fn_client_fini p_fini  = (fn_client_fini) dlsym(h, "ubs_engine_client_finalize");
    fn_topo_list   p_tlist = (fn_topo_list)   dlsym(h, "ubs_topo_node_list");
    fn_topo_local  p_local = (fn_topo_local)  dlsym(h, "ubs_topo_node_local_get");
    fn_shm_create  p_shm_c = (fn_shm_create) dlsym(h, "ubs_mem_shm_create");
    fn_shm_delete  p_shm_d = (fn_shm_delete) dlsym(h, "ubs_mem_shm_delete");
    fn_fd_create   p_fd_c  = (fn_fd_create)  dlsym(h, "ubs_mem_fd_create");
    fn_fd_delete   p_fd_d  = (fn_fd_delete)  dlsym(h, "ubs_mem_fd_delete");

    if (!p_init || !p_tlist || !p_local || !p_shm_c) {
        fprintf(stderr, "[FAIL] dlsym missing critical symbols\n");
        dlclose(h); return 1;
    }
    printf("[1] dlopen + dlsym OK (fd_create=%p)\n", (void*)p_fd_c);

    /* init */
    int32_t rc = p_init(NULL);
    if (rc != 0) { fprintf(stderr, "[FAIL] client_init: rc=%d\n", rc); dlclose(h); return 1; }
    printf("[2] client_initialize OK\n");

    /* topo */
    ubs_topo_node_t *nodes = NULL;
    uint32_t ncnt = 0;
    rc = p_tlist(&nodes, &ncnt);
    if (rc != 0) { fprintf(stderr, "[FAIL] topo: rc=%d\n", rc); goto done; }

    ubs_mem_nodes_t region;
    memset(&region, 0, sizeof(region));
    region.node_cnt = (ncnt > UBS_MEM_MAX_SLOT_NUM) ? UBS_MEM_MAX_SLOT_NUM : ncnt;
    for (uint32_t i = 0; i < region.node_cnt; i++) {
        region.slot_ids[i] = nodes[i].slot_id;
        printf("    node[%u]: slot=%u host=%s\n", i, nodes[i].slot_id, nodes[i].host_name);
    }

    /* local node for provider hint */
    ubs_topo_node_t local_nd;
    p_local(&local_nd);
    printf("[3] local: slot=%u host=%s\n\n", local_nd.slot_id, local_nd.host_name);

    /* provider = local node only */
    ubs_mem_nodes_t provider;
    memset(&provider, 0, sizeof(provider));
    provider.node_cnt = 1;
    provider.slot_ids[0] = local_nd.slot_id;

    uint8_t usr_info[32];
    memset(usr_info, 0, sizeof(usr_info));
    uint64_t size = 4ULL * 1024 * 1024;

    /* ================================================================ */
    /* Test A: shm_create flag=0 (original, expected to fail with 1013) */
    /* ================================================================ */
    {
        const char *name = "probe_A";
        if (p_shm_d) (void)p_shm_d(name);
        printf("[A] shm_create flag=0, provider=NULL ... ");
        rc = p_shm_c(name, size, usr_info, 0, &region, NULL);
        printf("rc=%d (%s)\n", rc, err_name(rc));
        if (rc == 0 && p_shm_d) p_shm_d(name);
    }

    /* ================================================================ */
    /* Test B: shm_create flag=CACHEABLE                                */
    /* ================================================================ */
    {
        const char *name = "probe_B";
        if (p_shm_d) (void)p_shm_d(name);
        printf("[B] shm_create flag=CACHEABLE(0x4), provider=NULL ... ");
        rc = p_shm_c(name, size, usr_info, UBS_MEM_FLAG_CACHEABLE, &region, NULL);
        printf("rc=%d (%s)\n", rc, err_name(rc));
        if (rc == 0 && p_shm_d) p_shm_d(name);
    }

    /* ================================================================ */
    /* Test C: shm_create flag=CACHEABLE|NO_WR, provider=local          */
    /* ================================================================ */
    {
        const char *name = "probe_C";
        if (p_shm_d) (void)p_shm_d(name);
        printf("[C] shm_create flag=0x5, provider=local(slot=%u) ... ",
               local_nd.slot_id);
        rc = p_shm_c(name, size, usr_info,
                      UBS_MEM_FLAG_CACHEABLE | UBS_MEM_FLAG_NO_WR_DELAY,
                      &region, &provider);
        printf("rc=%d (%s)\n", rc, err_name(rc));
        if (rc == 0 && p_shm_d) p_shm_d(name);
    }

    /* ================================================================ */
    /* Test D: fd_create (lease model, completely different code path)   */
    /* ================================================================ */
    if (p_fd_c) {
        const char *name = "probe_D";
        if (p_fd_d) (void)p_fd_d(name);

        ubs_mem_fd_owner_t owner;
        owner.uid = getuid();
        owner.gid = getgid();
        owner.pid = getpid();

        ubs_mem_fd_desc_t desc;
        memset(&desc, 0, sizeof(desc));

        printf("[D] fd_create distance=L0 ... ");
        rc = p_fd_c(name, size, &owner, 0644, MEM_DISTANCE_L0, &desc);
        printf("rc=%d (%s)\n", rc, err_name(rc));
        if (rc == 0) {
            printf("    SUCCESS! memid_cnt=%u, mem_size=%lu, unit_size=%lu\n",
                   desc.memid_cnt, (unsigned long)desc.mem_size,
                   (unsigned long)desc.unit_size);
            for (uint32_t i = 0; i < desc.memid_cnt && i < 4; i++) {
                printf("    memid[%u] = %lu → /dev/obmm_shmdev%lu\n",
                       i, (unsigned long)desc.memids[i],
                       (unsigned long)desc.memids[i]);
            }
            if (p_fd_d) p_fd_d(name);
        }
    } else {
        printf("[D] fd_create: symbol not found, skipped\n");
    }

    /* ================================================================ */
    /* Test E: shm_create with single-node region (only local)          */
    /* ================================================================ */
    {
        const char *name = "probe_E";
        if (p_shm_d) (void)p_shm_d(name);
        ubs_mem_nodes_t single_region;
        memset(&single_region, 0, sizeof(single_region));
        single_region.node_cnt = 1;
        single_region.slot_ids[0] = local_nd.slot_id;

        printf("[E] shm_create region=local-only(slot=%u), flag=0x5 ... ",
               local_nd.slot_id);
        rc = p_shm_c(name, size, usr_info,
                      UBS_MEM_FLAG_CACHEABLE | UBS_MEM_FLAG_NO_WR_DELAY,
                      &single_region, &provider);
        printf("rc=%d (%s)\n", rc, err_name(rc));
        if (rc == 0 && p_shm_d) p_shm_d(name);
    }

    printf("\n");

done:
    if (nodes) free(nodes);
    p_fini();
    dlclose(h);
    printf("Done.\n");
    return 0;
}
