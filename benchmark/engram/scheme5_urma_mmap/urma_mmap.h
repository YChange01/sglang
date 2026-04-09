/**
 * URMA MMAP: Direct remote memory mapping via UB/URMA.
 *
 * Provides a minimal C API for:
 *   1. Server side: register a local memory region as URMA segment
 *   2. Client side: import and mmap a remote segment into local address space
 *   3. Direct load/store access to remote memory (UBMMU handles routing)
 *
 * This bypasses all RPC/ZMQ overhead, giving CXL-equivalent performance
 * over UB interconnect.
 *
 * Build: see Makefile (links against liburma.so)
 * Usage: see urma_mmap_py.py (Python ctypes wrapper)
 */

#ifndef URMA_MMAP_H
#define URMA_MMAP_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define URMA_MMAP_OK          0
#define URMA_MMAP_ERR_INIT   -1
#define URMA_MMAP_ERR_DEVICE -2
#define URMA_MMAP_ERR_CTX    -3
#define URMA_MMAP_ERR_REG    -4
#define URMA_MMAP_ERR_IMPORT -5
#define URMA_MMAP_ERR_PARAM  -6

/**
 * Opaque handle for an URMA mmap context.
 */
typedef struct urma_mmap_ctx urma_mmap_ctx_t;

/**
 * Segment info exchanged between server and client.
 * Server fills this after register, client uses it to import.
 */
typedef struct {
    char eid[64];           /* Endpoint ID (string representation) */
    uint32_t uasid;         /* User Address Space ID */
    uint64_t seg_va;        /* Segment virtual address */
    uint64_t seg_len;       /* Segment length in bytes */
    uint32_t seg_id;        /* Segment ID */
    uint32_t token;         /* Access token */
} urma_mmap_seg_info_t;

/**
 * Initialize URMA and create context.
 *
 * @param dev_name  UB device name (e.g. "ubcore", or NULL for auto-detect)
 * @param eid_index EID index on the device (-1 for auto)
 * @return Context handle, or NULL on failure.
 */
urma_mmap_ctx_t* urma_mmap_init(const char* dev_name, int eid_index);

/**
 * Clean up URMA context and release all resources.
 */
void urma_mmap_destroy(urma_mmap_ctx_t* ctx);

/**
 * Get the EID of this context (for sharing with remote side).
 */
int urma_mmap_get_eid(urma_mmap_ctx_t* ctx, char* eid_out, size_t eid_len);

/**
 * SERVER SIDE: Register a local memory region as an URMA segment.
 *
 * After registration, the segment info can be shared with the client
 * side (e.g. via file, TCP, or ETCD) for import.
 *
 * @param ctx       URMA context.
 * @param addr      Starting address of the memory to register.
 * @param len       Length in bytes.
 * @param info_out  Filled with segment info for client-side import.
 * @return 0 on success, negative error code on failure.
 */
int urma_mmap_register(urma_mmap_ctx_t* ctx, void* addr, uint64_t len,
                       urma_mmap_seg_info_t* info_out);

/**
 * SERVER SIDE: Unregister a previously registered segment.
 */
int urma_mmap_unregister(urma_mmap_ctx_t* ctx, urma_mmap_seg_info_t* info);

/**
 * CLIENT SIDE: Import a remote segment and map it into local address space.
 *
 * After this call, the returned pointer can be used for direct load/store
 * access to the remote memory. UBMMU handles address translation and
 * routing transparently.
 *
 * @param ctx           URMA context.
 * @param remote_info   Segment info from the server side.
 * @param mapped_addr   Output: local virtual address of the mapped segment.
 * @return 0 on success, negative error code on failure.
 */
int urma_mmap_import(urma_mmap_ctx_t* ctx, const urma_mmap_seg_info_t* remote_info,
                     void** mapped_addr);

/**
 * CLIENT SIDE: Unmap and release an imported segment.
 */
int urma_mmap_unimport(urma_mmap_ctx_t* ctx, const urma_mmap_seg_info_t* remote_info);

/**
 * Utility: dump segment info to stdout (for debugging).
 */
void urma_mmap_dump_info(const urma_mmap_seg_info_t* info);

#ifdef __cplusplus
}
#endif

#endif /* URMA_MMAP_H */
