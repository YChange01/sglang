/**
 * URMA RW: Cross-node Engram memory access via urma_read (one-sided read).
 *
 * Client (Node2) uses urma_read to directly pull data from Server (Node1)
 * memory, bypassing any RPC or intermediate Worker process. This is the
 * closest practical approach to true load/store given the current URMA
 * driver limitations.
 *
 * Architecture:
 *   Server: allocates memory, registers URMA segment, listens on TCP
 *           for clients to exchange segment/jetty info
 *   Client: TCP-connects to server, imports remote segment + jetty,
 *           posts READ work requests to fetch specific rows
 *
 * Build: see Makefile (links against liburma.so)
 *
 * Thread safety:
 *   A urma_rw_ctx_t is NOT thread-safe. The internal jetty / completion queue
 *   can only be driven from a single producer thread at a time. If you need
 *   parallel readers, create one context per thread.
 *
 * Wire protocol limits:
 *   The TCP seg/jetty exchange sends a packed C struct containing raw SDK
 *   types (urma_eid_t, urma_jetty_id_t). This assumes:
 *     1. Server and client are the same endian (true for aarch64↔aarch64)
 *     2. Server and client use the same umdk SDK version (struct layout)
 *   Do not use across heterogeneous hardware or SDK versions without
 *   introducing explicit marshalling.
 */

#ifndef URMA_RW_H
#define URMA_RW_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define URMA_RW_OK           0
#define URMA_RW_ERR_INIT    -1
#define URMA_RW_ERR_DEVICE  -2
#define URMA_RW_ERR_CTX     -3
#define URMA_RW_ERR_REG     -4
#define URMA_RW_ERR_IMPORT  -5
#define URMA_RW_ERR_SOCKET  -6
#define URMA_RW_ERR_POST    -7
#define URMA_RW_ERR_POLL    -8
#define URMA_RW_ERR_PARAM   -9

/* Default TCP port for seg info exchange */
#define URMA_RW_DEFAULT_PORT 13857

/* Max outstanding read requests in one batch.
 * Sized for Engram: 12 tables × 256 tokens = 3072 reads. */
#define URMA_RW_MAX_BATCH 4096

/* Opaque context handle */
typedef struct urma_rw_ctx urma_rw_ctx_t;

/* ================================================================== */
/*  Initialization                                                     */
/* ================================================================== */

/**
 * Initialize URMA context on server or client.
 *
 * @param dev_name UB device name (e.g. "udma2", "bonding_dev_0", or NULL to auto-detect)
 * @param buf_size Size of the local memory buffer to register (bytes)
 * @return Context handle, or NULL on error
 */
urma_rw_ctx_t* urma_rw_init(const char* dev_name, uint64_t buf_size);

/**
 * Destroy URMA context and release all resources.
 */
void urma_rw_destroy(urma_rw_ctx_t* ctx);

/**
 * Get pointer to the local registered memory buffer.
 * On server: this is where data should be written (Engram tables).
 * On client: this is the destination for urma_read results.
 */
void* urma_rw_get_buffer(urma_rw_ctx_t* ctx);

/* ================================================================== */
/*  Server side: accept client connections                             */
/* ================================================================== */

/**
 * Start TCP listener and accept a single client connection.
 * Exchanges segment/jetty info over TCP and imports remote jetty.
 *
 * @param ctx URMA context
 * @param port TCP port to listen on
 * @return 0 on success, negative error code on failure
 */
int urma_rw_server_accept(urma_rw_ctx_t* ctx, uint16_t port);

/* ================================================================== */
/*  Client side: connect to server                                     */
/* ================================================================== */

/**
 * TCP-connect to a server, exchange segment/jetty info, import remote
 * segment and jetty. After this call, urma_rw_read() can be used.
 *
 * @param ctx URMA context
 * @param server_ip Server IP address (e.g. "192.168.84.245")
 * @param port Server TCP port
 * @return 0 on success, negative error code on failure
 */
int urma_rw_client_connect(urma_rw_ctx_t* ctx, const char* server_ip, uint16_t port);

/* ================================================================== */
/*  Data transfer                                                      */
/* ================================================================== */

/**
 * Read `len` bytes from remote offset `remote_offset` into local offset
 * `local_offset`. Blocks until the read completes.
 *
 * @param ctx       URMA context (must be a connected client)
 * @param local_offset  Offset within local buffer to write into
 * @param remote_offset Offset within remote buffer to read from
 * @param len       Number of bytes to read
 * @return 0 on success, negative error code on failure
 */
int urma_rw_read(urma_rw_ctx_t* ctx, uint64_t local_offset,
                 uint64_t remote_offset, uint32_t len);

/**
 * Batch read: fetch `count` ranges in a single submission + single poll.
 * All ranges must be within the registered segment sizes.
 *
 * @param ctx            URMA context
 * @param local_offsets  Array of local buffer offsets (size `count`)
 * @param remote_offsets Array of remote buffer offsets (size `count`)
 * @param lens           Array of byte lengths (size `count`)
 * @param count          Number of reads (must be <= URMA_RW_MAX_BATCH)
 * @return 0 on success, negative error code on failure
 */
int urma_rw_read_batch(urma_rw_ctx_t* ctx,
                       const uint64_t* local_offsets,
                       const uint64_t* remote_offsets,
                       const uint32_t* lens,
                       uint32_t count);

#ifdef __cplusplus
}
#endif

#endif /* URMA_RW_H */
