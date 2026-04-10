/**
 * UBS-MEM Smoke Test — Server (data provider)
 *
 * Allocates a named shmem object via ubs_mem SDK, fills it with a
 * deterministic pattern, and keeps it alive for the client to map.
 *
 * Usage:
 *   ./server [size_mb] [shmem_name]
 *   Default: size_mb=16, shmem_name="engram_test"
 *
 * Prerequisites:
 *   - ubse.service and ubsmd.service running
 *   - obmm kernel module loaded
 *   - region already created (ubsmd manages default region)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ubs_mem.h>

static volatile sig_atomic_t g_stop = 0;

static void sigint_handler(int sig)
{
    (void)sig;
    g_stop = 1;
}

int main(int argc, char *argv[])
{
    size_t size_mb = 16;
    const char *shm_name = "engram_test";

    if (argc > 1) size_mb = (size_t)atol(argv[1]);
    if (argc > 2) shm_name = argv[2];

    size_t buf_size = size_mb * 1024 * 1024;

    printf("=== UBS-MEM Smoke Server ===\n");
    printf("shmem name: %s\n", shm_name);
    printf("size: %zu MB (%zu bytes)\n", size_mb, buf_size);

    /* Step 1: Initialize ubs_mem library */
    ubsmem_options_t opts;
    if (ubsmem_init_attributes(&opts) != 0) {
        fprintf(stderr, "ubsmem_init_attributes failed\n");
        return 1;
    }
    if (ubsmem_initialize(&opts) != 0) {
        fprintf(stderr, "ubsmem_initialize failed\n");
        return 1;
    }
    ubsmem_set_logger_level(1);  /* info */
    printf("[1/6] ubsmem_initialize OK\n");

    int ret;

    /* Step 2: Query cluster info */
    ubsmem_cluster_info_t cinfo;
    if (ubsmem_lookup_cluster_statistic(&cinfo) == 0) {
        printf("[2/6] Cluster: %d hosts\n", cinfo.host_num);
        for (int h = 0; h < cinfo.host_num; h++) {
            printf("  host[%d]: %s, %d numas\n", h,
                   cinfo.host[h].host_name, cinfo.host[h].numa_num);
            for (int n = 0; n < cinfo.host[h].numa_num; n++) {
                ubsmem_numa_mem_t *nm = &cinfo.host[h].numa[n];
                printf("    numa%u: total=%zu MB, free=%zu MB\n",
                       nm->numa_id,
                       (size_t)(nm->mem_total / (1024*1024)),
                       (size_t)(nm->mem_free / (1024*1024)));
            }
        }
    } else {
        printf("[2/6] cluster query failed (non-fatal)\n");
    }

    /* Step 3: Discover region name */
    ubsmem_regions_t regions;
    char region_name[MAX_REGION_NAME_DESC_LENGTH] = {0};
    ret = ubsmem_lookup_regions(&regions);
    if (ret != 0 || regions.num == 0) {
        fprintf(stderr, "[3/6] ubsmem_lookup_regions failed: %d (num=%d)\n",
                ret, regions.num);
        ubsmem_finalize();
        return 1;
    }
    printf("[3/6] Found %d region(s):\n", regions.num);
    for (int i = 0; i < regions.num; i++) {
        printf("  region[%d]: %d hosts\n", i,
               regions.region[i].host_num);
        for (int h = 0; h < regions.region[i].host_num; h++) {
            printf("    host: %s (affinity=%d)\n",
                   regions.region[i].hosts[h].host_name,
                   regions.region[i].hosts[h].affinity);
        }
    }
    /* Try well-known default region name, or list by prefix */
    ubsmem_shmem_desc_t shm_list[MAX_SHM_CNT];
    uint32_t shm_cnt = MAX_SHM_CNT;
    ret = ubsmem_shmem_list_lookup("", shm_list, &shm_cnt);
    printf("  existing shmem objects (prefix=\"\"): %u (ret=%d)\n", shm_cnt, ret);
    for (uint32_t i = 0; i < shm_cnt && i < 10; i++) {
        printf("    [%u] name=\"%s\", size=%zu\n", i, shm_list[i].name, shm_list[i].size);
    }

    /* Look up region by name — try "default" first, then node hostname */
    const char *region_candidates[] = {"default", "node1", "node2", NULL};
    int found_region = 0;
    for (int i = 0; region_candidates[i]; i++) {
        ubsmem_region_desc_t rdesc;
        ret = ubsmem_lookup_region(region_candidates[i], &rdesc);
        if (ret == 0) {
            strncpy(region_name, region_candidates[i],
                    MAX_REGION_NAME_DESC_LENGTH - 1);
            printf("  Using region: \"%s\" (size=%zu)\n",
                   region_name, rdesc.size);
            found_region = 1;
            break;
        }
    }
    if (!found_region) {
        fprintf(stderr, "  No known region found. Try creating one or check ubsmd config.\n");
        ubsmem_finalize();
        return 1;
    }

    /* Step 4: Allocate named shmem object */
    /* Try to deallocate first in case leftover from previous run */
    ret = ubsmem_shmem_deallocate(shm_name);
    if (ret == 0)
        printf("  (cleaned up leftover shmem \"%s\")\n", shm_name);

    ret = ubsmem_shmem_allocate(region_name, shm_name, buf_size,
                                    0666, UBSM_FLAG_CACHE);
    if (ret != 0) {
        fprintf(stderr, "[4/6] ubsmem_shmem_allocate failed: %d\n", ret);
        ubsmem_finalize();
        return 1;
    }
    printf("[4/6] ubsmem_shmem_allocate OK (name=%s, size=%zu)\n",
           shm_name, buf_size);

    /* Step 5: Map it locally */
    void *ptr = NULL;
    ret = ubsmem_shmem_map(NULL, buf_size,
                           PROT_READ | PROT_WRITE,
                           MAP_SHARED,
                           shm_name, 0, &ptr);
    if (ret != 0 || !ptr) {
        fprintf(stderr, "[5/6] ubsmem_shmem_map failed: %d\n", ret);
        ubsmem_shmem_deallocate(shm_name);
        ubsmem_finalize();
        return 1;
    }
    printf("[5/6] ubsmem_shmem_map OK: ptr=%p\n", ptr);

    /* Step 6: Fill with deterministic pattern — same as scheme5 server */
    float *data = (float *)ptr;
    size_t num_floats = buf_size / sizeof(float);
    printf("[6/6] Filling %zu floats (data[i] = i * 0.001f)...\n", num_floats);
    for (size_t i = 0; i < num_floats; i++) {
        data[i] = (float)i * 0.001f;
    }

    printf("\nSample values:\n");
    printf("  data[0]      = %.6f\n", data[0]);
    printf("  data[42]     = %.6f\n", data[42]);
    printf("  data[1000]   = %.6f\n", data[1000]);
    if (num_floats > 100000)
        printf("  data[100000] = %.6f\n", data[100000]);

    signal(SIGINT, sigint_handler);
    printf("\nServer ready. Shmem \"%s\" is live. Press Ctrl+C to stop.\n",
           shm_name);

    while (!g_stop) {
        sleep(1);
    }

    printf("\nCleaning up...\n");
    ubsmem_shmem_unmap(ptr, buf_size);
    ubsmem_shmem_deallocate(shm_name);
    ubsmem_finalize();
    printf("Server stopped.\n");
    return 0;
}
