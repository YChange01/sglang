/*
 * ubse_probe.c — bypass ubsmd, directly connect to UBSE daemon
 *                and test if shmem create works.
 *
 * This is a diagnostic tool. If this works but scheme6 server.c
 * doesn't, the bug is in ubsmd (it doesn't call
 * ubs_engine_client_initialize before making UBSE calls).
 *
 * If this ALSO fails, the UBSE daemon itself has a problem.
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
/* (Inlined so we don't need an include path to ubs_engine*.h) */

#define UBS_TOPO_SOCKET_NUM   2
#define UBS_TOPO_IPADDR_NUM  50
#define UBS_TOPO_NUMA_NUM     4
#define UBS_MEM_MAX_SLOT_NUM 16
#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 64
#endif

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

/* ---- Function pointer types ---- */

typedef int32_t (*fn_client_init)(const char *);
typedef void    (*fn_client_fini)(void);
typedef int32_t (*fn_topo_list)(ubs_topo_node_t **, uint32_t *);
typedef int32_t (*fn_topo_local)(ubs_topo_node_t *);
typedef int32_t (*fn_shm_create)(const char *, uint64_t, uint8_t[32],
                                 uint64_t, const ubs_mem_nodes_t *,
                                 const ubs_mem_nodes_t *);
typedef int32_t (*fn_shm_delete)(const char *);

/* ---- main ---- */

int main(void)
{
    printf("=== UBSE direct probe (bypass ubsmd) ===\n\n");

    /* 1. dlopen */
    void *h = dlopen("libubse-client.so", RTLD_NOW);
    if (!h) {
        fprintf(stderr, "[FAIL] dlopen: %s\n", dlerror());
        return 1;
    }
    printf("[1] dlopen libubse-client.so OK\n");

    /* 2. dlsym */
    fn_client_init  p_init  = (fn_client_init) dlsym(h, "ubs_engine_client_initialize");
    fn_client_fini  p_fini  = (fn_client_fini) dlsym(h, "ubs_engine_client_finalize");
    fn_topo_list    p_tlist = (fn_topo_list)   dlsym(h, "ubs_topo_node_list");
    fn_topo_local   p_local = (fn_topo_local)  dlsym(h, "ubs_topo_node_local_get");
    fn_shm_create   p_shm_c = (fn_shm_create) dlsym(h, "ubs_mem_shm_create");
    fn_shm_delete   p_shm_d = (fn_shm_delete) dlsym(h, "ubs_mem_shm_delete");

    if (!p_init || !p_fini || !p_tlist || !p_local || !p_shm_c) {
        fprintf(stderr, "[FAIL] dlsym missing: init=%p fini=%p tlist=%p local=%p shm_c=%p\n",
                (void*)p_init, (void*)p_fini, (void*)p_tlist, (void*)p_local, (void*)p_shm_c);
        dlclose(h);
        return 1;
    }
    printf("[2] dlsym all OK\n");

    /* 3. ubs_engine_client_initialize — THE MISSING CALL */
    printf("[3] ubs_engine_client_initialize(NULL)...\n");
    int32_t rc = p_init(NULL);  /* NULL = use default /var/run/ubse/ubse.sock */
    if (rc != 0) {
        fprintf(stderr, "    FAIL: rc=%d\n", rc);
        dlclose(h);
        return 1;
    }
    printf("    OK\n");

    /* 4. topo: get node list */
    ubs_topo_node_t *nodes = NULL;
    uint32_t ncnt = 0;
    rc = p_tlist(&nodes, &ncnt);
    if (rc != 0) {
        fprintf(stderr, "[FAIL 4] topo_node_list: rc=%d\n", rc);
        goto done;
    }
    printf("[4] topo_node_list: %u nodes\n", ncnt);
    ubs_mem_nodes_t region;
    memset(&region, 0, sizeof(region));
    region.node_cnt = (ncnt > UBS_MEM_MAX_SLOT_NUM) ? UBS_MEM_MAX_SLOT_NUM : ncnt;
    for (uint32_t i = 0; i < region.node_cnt; i++) {
        region.slot_ids[i] = nodes[i].slot_id;
        printf("    node[%u]: slot_id=%u hostname=%s\n",
               i, nodes[i].slot_id, nodes[i].host_name);
    }
    free(nodes);

    /* 5. topo: local node */
    ubs_topo_node_t local_node;
    memset(&local_node, 0, sizeof(local_node));
    rc = p_local(&local_node);
    if (rc != 0) {
        fprintf(stderr, "[FAIL 5] topo_node_local_get: rc=%d\n", rc);
        goto done;
    }
    printf("[5] local: slot_id=%u hostname=%s\n",
           local_node.slot_id, local_node.host_name);

    /* 6. shm_create — the call that ubsmd fails on with error 800 */
    const char *test_name = "ubse_probe_test";

    /* Clean up any stale test object first */
    if (p_shm_d) {
        (void)p_shm_d(test_name);
    }

    uint8_t usr_info[32];
    memset(usr_info, 0, sizeof(usr_info));
    uint64_t size = 4ULL * 1024 * 1024;  /* 4 MB minimum */
    uint64_t flag = 0;

    printf("[6] ubs_mem_shm_create(\"%s\", %llu, region.cnt=%u)...\n",
           test_name, (unsigned long long)size, region.node_cnt);
    rc = p_shm_c(test_name, size, usr_info, flag, &region, NULL);
    if (rc != 0) {
        fprintf(stderr, "    FAIL: rc=%d\n", rc);
        fprintf(stderr, "    (0=OK, 1002=CONNECTION_FAILED, 1005=INTERNAL, "
                "1006=EXISTED, 1013=ALLOCATE)\n");
        goto done;
    }
    printf("    OK! shmem created successfully via direct UBSE call\n");

    /* 7. cleanup */
    printf("[7] cleaning up...\n");
    if (p_shm_d) {
        rc = p_shm_d(test_name);
        printf("    shm_delete: rc=%d\n", rc);
    }

done:
    p_fini();
    dlclose(h);
    printf("\nDone.\n");
    return 0;
}
