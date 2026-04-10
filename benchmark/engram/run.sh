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
#   ./run.sh bench all --server_ip 192.168.84.245                # Node2: all modes
#   ./run.sh bench ubsmem --provider node1                       # Node2: ubsmem only
#   ./run.sh bench urma --server_ip 192.168.84.245               # Node2: URMA only
#
# Environment:
#   NUMA_NODE    — NUMA node to bind (default: 0)
#   SIZE_MB      — shmem size in MB (default: 128)
#   SHM_NAME     — shmem object name (default: engram_test)
#   SERVER_IP    — server IP (default: 192.168.84.245)
#   TCP_PORT     — TCP data port (default: 13900)
#   URMA_PORT    — URMA seg exchange port (default: 13857)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NUMA_NODE=${NUMA_NODE:-0}
SIZE_MB=${SIZE_MB:-128}
SHM_NAME=${SHM_NAME:-engram_test}
SERVER_IP=${SERVER_IP:-192.168.84.245}
TCP_PORT=${TCP_PORT:-13900}
URMA_PORT=${URMA_PORT:-13857}
PROVIDER=${PROVIDER:-node1}

NUMA_CMD="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"

case "${1:-help}" in
    server)
        echo "=== Starting Unified Server ==="
        echo "  shmem: $SHM_NAME ($SIZE_MB MB)"
        echo "  TCP: :$TCP_PORT, URMA: :$URMA_PORT"
        echo "  NUMA: node $NUMA_NODE"
        echo ""
        $NUMA_CMD ./server/server $SIZE_MB $SHM_NAME $TCP_PORT $URMA_PORT
        ;;

    bench)
        shift
        MODE=${1:-all}
        shift 2>/dev/null || true
        echo "=== Running Benchmark: $MODE ==="
        $NUMA_CMD ./bench/bench_all "$MODE" \
            --size_mb $SIZE_MB --name $SHM_NAME \
            --server_ip $SERVER_IP --tcp_port $TCP_PORT \
            --urma_port $URMA_PORT --provider $PROVIDER \
            "$@"
        ;;

    paper)
        shift 2>/dev/null || true
        echo "=== Paper Reproduction (all modes) ==="
        for mode in local tcp urma ubsmem; do
            echo ""
            echo "--- Mode: $mode ---"
            $NUMA_CMD ./bench/bench_all "$mode" \
                --size_mb $SIZE_MB --name $SHM_NAME \
                --server_ip $SERVER_IP --tcp_port $TCP_PORT \
                --urma_port $URMA_PORT --provider $PROVIDER \
                "$@" 2>&1 | grep -A 20 "Paper: Engram-27B"
        done
        ;;

    all)
        shift 2>/dev/null || true
        echo "=== Full Benchmark Suite ==="
        $NUMA_CMD ./bench/bench_all all \
            --size_mb $SIZE_MB --name $SHM_NAME \
            --server_ip $SERVER_IP --tcp_port $TCP_PORT \
            --urma_port $URMA_PORT --provider $PROVIDER \
            "$@"
        ;;

    help|*)
        echo "Usage: $0 {server|bench|paper|all} [options]"
        echo ""
        echo "  server              Start unified server (Node1)"
        echo "  bench [mode] [...]  Run benchmark (Node2)"
        echo "    modes: local, tcp, urma, ubsmem, ubsmem-nc, ubsmem-huge, all"
        echo "  paper               Paper reproduction only"
        echo "  all                 Full benchmark suite"
        echo ""
        echo "Environment variables: NUMA_NODE, SIZE_MB, SHM_NAME, SERVER_IP,"
        echo "  TCP_PORT, URMA_PORT, PROVIDER"
        ;;
esac
