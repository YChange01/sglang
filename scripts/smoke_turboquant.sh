#!/usr/bin/env bash
# Day 7 MVP smoke: launch SGLang with TurboQuant backend + send one
# completion request + compare against FLASH_ATTN / triton baseline.
#
# Usage:
#   bash scripts/smoke_turboquant.sh                         # TURBOQUANT_b4 default
#   TQ_STAGE=TURBOQUANT_split_3_5bit_fu bash scripts/smoke_turboquant.sh
#
# Env overrides (match vllm_fork/test/*.sh conventions):
#   MODEL       model path
#   GPU         CUDA_VISIBLE_DEVICES
#   PORT        server port
#   MAX_LEN     --context-length
#   MEM_FRAC    --mem-fraction-static
#   TQ_STAGE    stage name from stages.py (e.g. TURBOQUANT_b4, split_3_5bit_fu)
#   OUTLIER_MASK  path to calibrated outlier mask (split stages only)
#   BASELINE    also spin up triton backend for side-by-side compare (1/0)

set -u

MODEL="${MODEL:-/mnt/nvme3n1/g00872988/models/Llama-3.1-8B-Instruct}"
GPU="${GPU:-0}"
PORT="${PORT:-30000}"
MAX_LEN="${MAX_LEN:-32768}"
MEM_FRAC="${MEM_FRAC:-0.5}"
TQ_STAGE="${TQ_STAGE:-TURBOQUANT_b4}"
OUTLIER_MASK="${OUTLIER_MASK:-/tmp/outliers_llama-3_1-8b_32.pt}"
BASELINE="${BASELINE:-0}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs/smoke_turboquant_$TS"
mkdir -p "$LOG_DIR"

# Resolve stage -> env vars via the vendored registry (no CUDA import).
TQ_ENV=$(python3 -c "
import sys
sys.path.insert(0, '$ROOT_DIR/python')
from sglang.srt.layers.attention.turboquant import resolve_stage
_, cfg = resolve_stage('$TQ_STAGE')
d = cfg.to_env_dict()
if cfg.is_split:
    d['TURBOQUANT_OUTLIER_MASK'] = '$OUTLIER_MASK'
print(' '.join(f'{k}={v}' for k, v in d.items()))
") || { echo "[smoke] stage resolution failed"; exit 1; }

echo "[smoke] stage=$TQ_STAGE"
echo "[smoke] env: $TQ_ENV"
echo "[smoke] model=$MODEL gpu=$GPU max_len=$MAX_LEN mem_frac=$MEM_FRAC"
echo "[smoke] log_dir=$LOG_DIR"

# Split mode needs the outlier mask file present.
if [[ "$TQ_STAGE" == *split* ]] && [ ! -f "$OUTLIER_MASK" ]; then
    echo "[smoke] ERROR: split stage requires calibrated outlier mask at" >&2
    echo "             $OUTLIER_MASK  (missing)" >&2
    echo "             Copy from vllm_fork or re-run scripts/calibrate.sh" >&2
    exit 1
fi

launch_tq() {
    local log="$1"
    echo "[smoke] launching TurboQuant server on port $PORT ..."
    export CUDA_VISIBLE_DEVICES="$GPU"
    env $TQ_ENV PYTHONUNBUFFERED=1 setsid python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --attention-backend turboquant \
        --disable-cuda-graph \
        --disable-radix-cache \
        --page-size 16 \
        --context-length "$MAX_LEN" \
        --mem-fraction-static "$MEM_FRAC" \
        --port "$PORT" \
        --host 127.0.0.1 \
        >"$log" 2>&1 &
    TQ_PGID=$!
    echo "[smoke] server pgid=$TQ_PGID   log tail:  tail -f $log"
}

wait_healthy() {
    local port="$1" pid="$2"
    echo "[smoke] waiting for /health (up to 15 min for first JIT) ..."
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:${port}/health_generate" >/dev/null 2>&1 \
            || curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            echo "[smoke] server healthy"
            return 0
        fi
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[smoke] server exited early; tail of log:" >&2
            tail -n 60 "$LOG_DIR/server.log" >&2
            return 1
        fi
        sleep 5
    done
    echo "[smoke] never healthy" >&2
    tail -n 60 "$LOG_DIR/server.log" >&2
    return 1
}

stop_server() {
    if [ -n "${TQ_PGID:-}" ]; then
        echo "[smoke] stopping pgid $TQ_PGID"
        kill -TERM -"$TQ_PGID" 2>/dev/null || kill -TERM "$TQ_PGID" 2>/dev/null || true
        wait "$TQ_PGID" 2>/dev/null || true
        TQ_PGID=""
    fi
}
trap stop_server EXIT INT TERM

# Single prompt test
send_prompt() {
    local tag="$1"
    local prompt='The capital of France is'
    echo ""
    echo "=============================================="
    echo "[smoke] $tag single-prompt test"
    echo "=============================================="
    curl -s -X POST "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"$MODEL\",
            \"prompt\": \"$prompt\",
            \"max_tokens\": 32,
            \"temperature\": 0.0
        }" | tee "$LOG_DIR/${tag}_response.json"
    echo ""
}

launch_tq "$LOG_DIR/server.log"
if ! wait_healthy "$PORT" "$TQ_PGID"; then
    exit 1
fi
send_prompt "$TQ_STAGE"

stop_server

if [ "$BASELINE" = "1" ]; then
    echo ""
    echo "=============================================="
    echo "[smoke] relaunching with triton backend for comparison"
    echo "=============================================="
    export CUDA_VISIBLE_DEVICES="$GPU"
    PYTHONUNBUFFERED=1 setsid python3 -m sglang.launch_server \
        --model-path "$MODEL" \
        --attention-backend triton \
        --disable-cuda-graph \
        --disable-radix-cache \
        --context-length "$MAX_LEN" \
        --mem-fraction-static "$MEM_FRAC" \
        --port "$PORT" \
        --host 127.0.0.1 \
        >"$LOG_DIR/baseline_server.log" 2>&1 &
    TQ_PGID=$!
    if wait_healthy "$PORT" "$TQ_PGID"; then
        send_prompt "baseline_triton"
    fi
    stop_server
fi

echo ""
echo "[smoke] done -- logs + responses in $LOG_DIR/"
