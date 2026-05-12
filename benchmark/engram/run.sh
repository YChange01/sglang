#!/bin/bash
# Engram Cross-Node Benchmark Runner
#
# Usage:
#   ./run.sh server                         — Node1: start unified server
#   ./run.sh bench [mode] [options]         — Node2: run benchmark
#   ./run.sh paper [options]                — Node2: paper reproduction (Engram-27B)
#   ./run.sh all [options]                  — Node2: all benchmarks
#
# Examples:
#   ./run.sh server                                              # Node1
#   ./run.sh all --server_ip 192.168.84.245                      # Node2: all read modes
#   ./run.sh ubsmem-nc-read --iters 1000                         # Node2: one read scheme
#   ./run.sh bench ubsmem --provider node1                       # Node2: ubsmem only
#   ./run.sh bench urma --server_ip 192.168.84.245               # Node2: URMA only
#   ./run.sh nc-write server --bytes 512                         # Node1: UB-M NC write ponger
#   ./run.sh nc-write client --bytes 512                         # Node2: UB-M NC write pinger
#   ./run.sh urma-write server --bytes 512                       # Node1: URMA write ponger
#   ./run.sh urma-write client --bytes 512                       # Node2: URMA write pinger
#
# Environment:
#   NUMA_NODE    — NUMA node to bind (default: 0; set "none" to disable)
#   SIZE_MB      — shmem size in MB (default: 128)
#   SHM_NAME     — shmem object name (default: engram_test)
#   SERVER_IP    — server IP (default: 192.168.84.245)
#   TCP_PORT     — TCP data port (default: 13900)
#   URMA_PORT    — URMA seg exchange port (default: 13857)
#   URMA_DEV     — URMA device name (e.g. udma2; default auto)
#   URMA_TP_TYPE — URMA TP type for RM mode: ctp|rtp|utp (default: ctp)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NUMA_NODE=${NUMA_NODE:-0}
SIZE_MB=${SIZE_MB:-128}
SHM_NAME=${SHM_NAME:-engram_test}
SERVER_IP=${SERVER_IP:-192.168.84.245}
TCP_PORT=${TCP_PORT:-13900}
URMA_PORT=${URMA_PORT:-13857}
URMA_PP_PORT=${URMA_PP_PORT:-13858}
PROVIDER=${PROVIDER:-node1}
URMA_DEV=${URMA_DEV:-}
URMA_TP_TYPE=${URMA_TP_TYPE:-ctp}

if [ "$NUMA_NODE" = "none" ] || [ "$NUMA_NODE" = "off" ] || [ "$NUMA_NODE" = "-1" ]; then
    NUMA_CMD=""
    NUMA_DESC="disabled"
elif command -v numactl >/dev/null 2>&1; then
    NUMA_CMD="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
    NUMA_DESC="node $NUMA_NODE"
else
    NUMA_CMD=""
    NUMA_DESC="disabled (numactl not found)"
    echo "[WARN] numactl not found; running without NUMA binding. Install numactl or set NUMA_NODE=none to silence this." >&2
fi

read_bin_for_mode() {
    case "$1" in
        local) echo "bench_local" ;;
        tcp) echo "bench_tcp" ;;
        urma) echo "bench_urma_read" ;;
        ubsmem) echo "bench_ubsmem_cache" ;;
        ubsmem-nc) echo "bench_ubsmem_noncache" ;;
        ubsmem-import-nc|ubsmem-inc) echo "bench_ubsmem_import_nc" ;;
        ubsmem-huge) echo "bench_ubsmem_huge" ;;
        *) return 1 ;;
    esac
}

run_read_bench() {
    local bin="$1"
    shift
    $NUMA_CMD "./bench/$bin" \
        --size_mb "$SIZE_MB" --name "$SHM_NAME" \
        --server_ip "$SERVER_IP" --tcp_port "$TCP_PORT" \
        --urma_port "$URMA_PORT" --provider "$PROVIDER" \
        "$@"
}

run_read_mode() {
    local mode="$1"
    shift
    local bin
    bin="$(read_bin_for_mode "$mode")" || {
        echo "Unknown read mode: $mode" >&2
        return 1
    }
    run_read_bench "$bin" "$@"
}

case "${1:-help}" in
    server)
        echo "=== Starting Unified Server ==="
        echo "  shmem: $SHM_NAME ($SIZE_MB MB)"
        echo "  TCP: :$TCP_PORT, URMA: :$URMA_PORT"
        echo "  NUMA: $NUMA_DESC"
        echo "  URMA_DEV: ${URMA_DEV:-auto}"
        echo ""
        $NUMA_CMD ./server/server $SIZE_MB $SHM_NAME $TCP_PORT $URMA_PORT
        ;;

    bench)
        shift
        MODE=${1:-all}
        shift 2>/dev/null || true
        echo "=== Running Benchmark: $MODE ==="
        if [ "$MODE" = "all" ]; then
            for mode in local tcp urma ubsmem ubsmem-nc ubsmem-import-nc ubsmem-huge; do
                echo ""
                echo "--- Mode: $mode ---"
                run_read_mode "$mode" "$@"
            done
        else
            run_read_mode "$MODE" "$@"
        fi
        ;;

    paper)
        shift 2>/dev/null || true
        echo "=== Paper Reproduction (all modes) ==="
        for mode in local tcp urma ubsmem; do
            echo ""
            echo "--- Mode: $mode ---"
            run_read_mode "$mode" "$@"
        done
        ;;

    all)
        shift 2>/dev/null || true
        echo "=== Full Benchmark Suite ==="
        for mode in local tcp urma ubsmem ubsmem-nc ubsmem-import-nc ubsmem-huge; do
            echo ""
            echo "--- Mode: $mode ---"
            run_read_mode "$mode" "$@"
        done
        ;;

    local-read)
        shift 2>/dev/null || true
        echo "=== LOCAL Read Benchmark ==="
        run_read_bench bench_local "$@"
        ;;

    tcp-read)
        shift 2>/dev/null || true
        echo "=== TCP Read Benchmark ==="
        run_read_bench bench_tcp "$@"
        ;;

    urma-read)
        shift 2>/dev/null || true
        echo "=== URMA READ Benchmark ==="
        echo "  URMA_DEV: ${URMA_DEV:-auto}"
        run_read_bench bench_urma_read "$@"
        ;;

    ubsmem-cache-read)
        shift 2>/dev/null || true
        echo "=== UB-MEM Cache Read Benchmark ==="
        run_read_bench bench_ubsmem_cache "$@"
        ;;

    ubsmem-nc-read)
        shift 2>/dev/null || true
        echo "=== UB-MEM Noncache Read Benchmark ==="
        run_read_bench bench_ubsmem_noncache "$@"
        ;;

    ubsmem-import-nc-read)
        shift 2>/dev/null || true
        echo "=== UB-MEM Import-Noncache Read Benchmark ==="
        run_read_bench bench_ubsmem_import_nc "$@"
        ;;

    ubsmem-huge-read)
        shift 2>/dev/null || true
        echo "=== UB-MEM Hugepage Read Benchmark ==="
        run_read_bench bench_ubsmem_huge "$@"
        ;;

    e2e)
        shift
        CMD=${1:-retrieval}
        shift 2>/dev/null || true
        echo "=== E2E Benchmark: $CMD ==="
        export SERVER_IP TCP_PORT URMA_PORT
        export PROVIDER_HOST=$PROVIDER
        export SHM_NAME
        python3 e2e/bench_e2e.py "$CMD" --server_ip $SERVER_IP "$@"
        ;;

    nc-write)
        shift
        ROLE=${1:-client}
        shift 2>/dev/null || true
        echo "=== UB-MEM Noncache Write Pingpong: $ROLE ==="
        echo "  shmem base: $SHM_NAME"
        echo "  NUMA: $NUMA_DESC"
        echo ""
        $NUMA_CMD ./bench/ubsmem_nc_write_pingpong \
            --role "$ROLE" --name "$SHM_NAME" "$@"
        ;;

    urma-write)
        shift
        ROLE=${1:-client}
        shift 2>/dev/null || true
        echo "=== URMA WRITE Pingpong: $ROLE ==="
        echo "  server: $SERVER_IP:$URMA_PP_PORT"
        echo "  NUMA: $NUMA_DESC"
        echo "  URMA_DEV: ${URMA_DEV:-auto}"
        echo "  URMA_TP_TYPE: $URMA_TP_TYPE"
        echo ""
        export URMA_TP_TYPE
        $NUMA_CMD ./bench/urma_write_pingpong \
            --role "$ROLE" --server_ip "$SERVER_IP" --port "$URMA_PP_PORT" "$@"
        ;;

    help|*)
        echo "Usage: $0 {server|bench|paper|e2e|local-read|tcp-read|urma-read|ubsmem-cache-read|ubsmem-nc-read|ubsmem-import-nc-read|ubsmem-huge-read|nc-write|urma-write|all} [options]"
        echo ""
        echo "  server              Start unified server (Node1)"
        echo "  bench [mode] [...]  Run micro benchmark (Node2)"
        echo "    modes: local, tcp, urma, ubsmem, ubsmem-nc, ubsmem-import-nc, ubsmem-huge, all"
        echo "  local-read          Standalone LOCAL read benchmark"
        echo "  tcp-read            Standalone TCP read benchmark"
        echo "  urma-read           Standalone URMA READ benchmark"
        echo "  ubsmem-cache-read   Standalone UB-MEM cache read benchmark"
        echo "  ubsmem-nc-read      Standalone UB-MEM noncache read benchmark"
        echo "  ubsmem-import-nc-read"
        echo "                      Standalone UB-MEM import-noncache read benchmark"
        echo "  ubsmem-huge-read    Standalone UB-MEM hugepage read benchmark"
        echo "  paper               Paper reproduction only"
        echo "  e2e [cmd] [...]     End-to-end benchmark (Node2)"
        echo "    cmds: retrieval, table2"
        echo "    e.g.: ./run.sh e2e retrieval --all"
        echo "          ./run.sh e2e table2 --model /tmp/g00872988/Qwen3-8B"
        echo "  nc-write server|client"
        echo "                      UB-MEM noncache write 1/2 RTT ping-pong"
        echo "    e.g.: ./run.sh nc-write server --bytes 512 --iters 100000"
        echo "          ./run.sh nc-write client --bytes 512 --iters 100000"
        echo "  urma-write server|client"
        echo "                      URMA posted WRITE 1/2 RTT ping-pong"
        echo "    e.g.: ./run.sh urma-write server --bytes 512 --iters 100000"
        echo "          ./run.sh urma-write client --bytes 512 --iters 100000"
        echo "  all                 Full micro benchmark suite"
        echo ""
        echo "Environment variables: NUMA_NODE, SIZE_MB, SHM_NAME, SERVER_IP,"
        echo "  TCP_PORT, URMA_PORT, URMA_PP_PORT, PROVIDER, URMA_DEV"
        ;;
esac
