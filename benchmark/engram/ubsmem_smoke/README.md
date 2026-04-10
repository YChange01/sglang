# UBS-MEM Smoke Test (load/store path)

Cross-node Engram data access via `ubs_mem` SDK — true load/store through
UBMMU hardware page table translation, bypassing URMA async post+poll.

## Architecture

```
Node1 (server)                           Node2 (client)
ubsmem_initialize()                      ubsmem_initialize()
ubsmem_shmem_allocate("engram_test")     ubsmem_shmem_lookup("engram_test")
ubsmem_shmem_map() → ptr                 ubsmem_shmem_map() → ptr
fill data via ptr                        memcpy / direct load via ptr
wait...                                  benchmark
```

Both nodes connect to their local `ubsmd` daemon, which coordinates
memory allocation and address translation via the obmm kernel module.

## Build

```bash
# On Node1 or Node2 (must have ubs_mem SDK installed)
make
```

## Run

```bash
# Node1 (data provider)
./server 16 engram_test          # 16 MB, shmem name

# Node2 (reader / benchmarker)
./client 16 engram_test 10000 341 500
#        ^  ^           ^     ^   ^
#        |  shmem name  rows  dim iterations
#        size_mb
```

## Compare with Scheme 5 (urma_read)

| Metric | Scheme 5 (urma_read) | UBS-MEM (load/store) |
|--------|---------------------|---------------------|
| API | urma_post_jetty_send_wr | memcpy / pointer deref |
| Mechanism | async post + poll CQ | sync page fault + HW translation |
| Single read | ~2.65 us | **TBD** (expect < 1 us) |
| Batch 128-tok | 0.14 ms | **TBD** |
