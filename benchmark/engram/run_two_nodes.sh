#!/usr/bin/env bash
# Two-node wrapper for Engram microbenchmarks.
#
# Run this script from either node. The first argument selects the local node
# role, and the second argument selects the benchmark action.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

NUMA_NODE=${NUMA_NODE:-0}
SIZE_MB=${SIZE_MB:-128}
SHM_NAME=${SHM_NAME:-engram_test}
SERVER_IP=${SERVER_IP:-192.168.84.245}
TCP_PORT=${TCP_PORT:-13900}
URMA_PORT=${URMA_PORT:-13857}
URMA_PP_PORT=${URMA_PP_PORT:-13858}
URMA_DEV=${URMA_DEV:-}
URMA_TP_TYPE=${URMA_TP_TYPE:-ctp}
URMA_PRIORITY=${URMA_PRIORITY:-}

# UBS-MEM provider hostnames. For read benchmarks, clients map objects created
# by Node1, so READ_PROVIDER defaults to NODE1_PROVIDER.
NODE1_PROVIDER=${NODE1_PROVIDER:-node1}
NODE2_PROVIDER=${NODE2_PROVIDER:-node2}
READ_PROVIDER=${READ_PROVIDER:-$NODE1_PROVIDER}

# UMDK perftest-style periodic completion for URMA WRITE ping-pong.
# Set URMA_WRITE_CQ_MOD=0 for pure posted WRITE with no CQ polling.
URMA_WRITE_CQ_MOD=${URMA_WRITE_CQ_MOD:-64}

export NUMA_NODE SIZE_MB SHM_NAME SERVER_IP TCP_PORT URMA_PORT URMA_PP_PORT URMA_DEV URMA_TP_TYPE URMA_PRIORITY

usage() {
    cat <<'EOF'
Usage:
  ./run_two_nodes.sh node1 build
  ./run_two_nodes.sh node2 build

  ./run_two_nodes.sh node1 server
  SERVER_IP=<node1_ip> ./run_two_nodes.sh node2 read [mode] [options]
  SERVER_IP=<node1_ip> ./run_two_nodes.sh node2 read-compare [options]
  SERVER_IP=<node1_ip> ./run_two_nodes.sh node2 read-all [options]

  ./run_two_nodes.sh node1 nc-write [options]
  ./run_two_nodes.sh node2 nc-write [options]

  ./run_two_nodes.sh node1 urma-write [options]
  SERVER_IP=<node1_ip> ./run_two_nodes.sh node2 urma-write [options]

Nodes:
  node1                 Server-side node
  node2                 Client-side node

Read modes:
  local
  tcp
  urma
  ubsmem
  ubsmem-nc
  ubsmem-import-nc      Recommended for UB-M-NC comparison
  ubsmem-huge
  all

Common environment:
  NUMA_NODE             default: 0; set "none" to disable binding
  SIZE_MB               default: 128
  SHM_NAME              default: engram_test
  SERVER_IP             default: 192.168.84.245
  TCP_PORT              default: 13900
  URMA_PORT             default: 13857
  URMA_PP_PORT          default: 13858
  URMA_DEV              e.g. udma2
  URMA_TP_TYPE          default: ctp; matches urma_sample -m 0 -t 1
  URMA_PRIORITY         optional JFS priority override, e.g. 6, 7, or 15
  NODE1_PROVIDER        default: node1
  NODE2_PROVIDER        default: node2
  READ_PROVIDER         default: NODE1_PROVIDER
  URMA_WRITE_CQ_MOD     default: 64; set 0 for pure posted WRITE

Examples:
  # Node1: build and start unified read server
  URMA_DEV=udma2 ./run_two_nodes.sh node1 build
  URMA_DEV=udma2 ./run_two_nodes.sh node1 server

  # Node2: compare read puncture modes
  SERVER_IP=192.168.84.245 URMA_DEV=udma2 \
    ./run_two_nodes.sh node2 read-compare --iters 100000

  # UB-M noncache WRITE ping-pong, run Node1 first
  ./run_two_nodes.sh node1 nc-write --bytes 512 --iters 100000
  ./run_two_nodes.sh node2 nc-write --bytes 512 --iters 100000

  # URMA WRITE ping-pong, run Node1 first
  URMA_DEV=udma2 ./run_two_nodes.sh node1 urma-write --bytes 512
  SERVER_IP=192.168.84.245 URMA_DEV=udma2 \
    ./run_two_nodes.sh node2 urma-write --bytes 512
EOF
}

print_env() {
    echo "NUMA_NODE=$NUMA_NODE"
    echo "SIZE_MB=$SIZE_MB"
    echo "SHM_NAME=$SHM_NAME"
    echo "SERVER_IP=$SERVER_IP"
    echo "TCP_PORT=$TCP_PORT"
    echo "URMA_PORT=$URMA_PORT"
    echo "URMA_PP_PORT=$URMA_PP_PORT"
    echo "URMA_DEV=${URMA_DEV:-auto}"
    echo "URMA_TP_TYPE=$URMA_TP_TYPE"
    echo "URMA_PRIORITY=${URMA_PRIORITY:-auto}"
    echo "NODE1_PROVIDER=$NODE1_PROVIDER"
    echo "NODE2_PROVIDER=$NODE2_PROVIDER"
    echo "READ_PROVIDER=$READ_PROVIDER"
    echo "URMA_WRITE_CQ_MOD=$URMA_WRITE_CQ_MOD"
}

normalize_node() {
    case "$1" in
        node1|n1|server)
            echo "node1"
            ;;
        node2|n2|client)
            echo "node2"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

need_node() {
    local want="$1"
    if [ "$NODE" != "$want" ]; then
        echo "Action '$ACTION' must be run on $want, but local role is $NODE" >&2
        exit 1
    fi
}

run_build() {
    make server bench
}

run_server() {
    need_node node1
    exec ./run.sh server "$@"
}

run_read() {
    need_node node2

    local mode="ubsmem-import-nc"
    if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
        mode="$1"
        shift
    fi

    export PROVIDER="$READ_PROVIDER"
    case "$mode" in
        all)
            exec ./run.sh all "$@"
            ;;
        local|tcp|urma|ubsmem|ubsmem-nc|ubsmem-import-nc|ubsmem-inc|ubsmem-huge)
            exec ./run.sh bench "$mode" "$@"
            ;;
        local-read|tcp-read|urma-read|ubsmem-cache-read|ubsmem-nc-read|ubsmem-import-nc-read|ubsmem-huge-read)
            exec ./run.sh "$mode" "$@"
            ;;
        *)
            echo "Unknown read mode: $mode" >&2
            usage
            exit 1
            ;;
    esac
}

run_read_compare() {
    need_node node2
    export PROVIDER="$READ_PROVIDER"

    ./run.sh ubsmem-import-nc-read "$@"
    ./run.sh ubsmem-nc-read "$@"
    ./run.sh urma-read "$@"
}

run_nc_write() {
    local role provider
    if [ "$NODE" = "node1" ]; then
        role="server"
        provider="$NODE1_PROVIDER"
    else
        role="client"
        provider="$NODE2_PROVIDER"
    fi

    exec ./run.sh nc-write "$role" --provider "$provider" "$@"
}

run_urma_write() {
    local role
    if [ "$NODE" = "node1" ]; then
        role="server"
    else
        role="client"
    fi

    local cq_args=()
    if [ "$URMA_WRITE_CQ_MOD" != "0" ]; then
        cq_args=(--cq_mod "$URMA_WRITE_CQ_MOD")
    fi

    exec ./run.sh urma-write "$role" "${cq_args[@]}" "$@"
}

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "${1:-}" = "help" ]; then
    usage
    exit 0
fi

RAW_NODE="$1"
NODE="$(normalize_node "$RAW_NODE")"
shift
if [ "$NODE" = "unknown" ]; then
    echo "Unknown node role: $RAW_NODE" >&2
    usage
    exit 1
fi

ACTION="${1:-help}"
if [ $# -gt 0 ]; then
    shift
fi

case "$ACTION" in
    help|-h|--help)
        usage
        ;;
    env)
        print_env
        ;;
    build)
        run_build "$@"
        ;;
    server)
        run_server "$@"
        ;;
    read)
        run_read "$@"
        ;;
    read-compare)
        run_read_compare "$@"
        ;;
    read-all)
        run_read all "$@"
        ;;
    nc-write)
        run_nc_write "$@"
        ;;
    urma-write)
        run_urma_write "$@"
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        usage
        exit 1
        ;;
esac
