#!/bin/bash
#SBATCH --job-name=vllm_gradio
#SBATCH --comment=pytorch
##SBATCH --partition=mig_amd_a100_4
##SBATCH --partition=gh200_1
##SBATCH --partition=eme_h200nv_8
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=cas_v100nv_8
##SBATCH --partition=cas_v100nv_4
##SBATCH --partition=cas_v100_4
##SBATCH --partition=bigmem
##SBATCH --partition=gdebug01
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set +e

#######################################
# Defaults (overridable by CLI or env)
#######################################
SERVER="$(hostname)"
GRADIO_PORT="${GRADIO_PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL_DEFAULT="Qwen/Qwen3-0.6B"
VLLM_MODEL="${VLLM_MODEL:-$VLLM_MODEL_DEFAULT}"
SIF_PATH="${SIF_PATH:-/scratch/$USER/gpt-oss-with-vllm-on-supercomputer/vllm-gptoss.sif}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
SCHED_STEPS="${SCHED_STEPS:-1}"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/${USER}/gpt-oss-with-vllm-on-supercomputer/.vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/.gradio_cache}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/tmp}"

#######################################
# CLI parsing
#######################################
print_help() {
  cat <<EOF
Usage: $0 [--model <hf_model>] [--vllm-port <port>] [--gradio-port <port>] [--sif </path/to.sif>]

Examples:
  $0 --model openai/gpt-oss-20b
  sbatch --export=ALL,VLLM_MODEL=openai/gpt-oss-20b,VLLM_PORT=9000,GRADIO_PORT=7000 $0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        VLLM_MODEL="$2"; shift 2;;
    --vllm-port)    VLLM_PORT="$2"; shift 2;;
    --gradio-port)  GRADIO_PORT="$2"; shift 2;;
    --sif)          SIF_PATH="$2"; shift 2;;
    -h|--help)      print_help; exit 0;;
    *) echo "Unknown arg: $1"; print_help; exit 1;;
  esac
done

#######################################
# Paths & logs
#######################################
WORK_DIR="/scratch/$USER/gpt-oss-with-vllm-on-supercomputer"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$WORK_DIR" "$LOG_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "${VLLM_CACHE_ROOT}"

JOB_ID="${SLURM_JOB_ID:-none}"
if [ "$JOB_ID" = "none" ]; then
  VLLM_LOG="${LOG_DIR}/vllm_server.log"
  GRADIO_LOG="${LOG_DIR}/gradio_server.log"
  PORT_FWD_FILE="${WORK_DIR}/port_forwarding.txt"
else
  VLLM_LOG="${LOG_DIR}/vllm_server_${JOB_ID}.log"
  GRADIO_LOG="${LOG_DIR}/gradio_server_${JOB_ID}.log"
  PORT_FWD_FILE="${WORK_DIR}/port_forwarding_${JOB_ID}.txt"
fi

#######################################
# HF cache (show and ensure)
#######################################
if [ -n "${HF_HOME}" ]; then
  HF_PRESET=1
else
  HF_HOME="/scratch/${USER}/.huggingface"
  HF_PRESET=0
fi
mkdir -p "${HF_HOME}/hub"

# Show disk free on the cache filesystem (best-effort)
CACHE_ROOT="${HF_HOME}/hub"
CACHE_FREE="$(df -h "${CACHE_ROOT}" 2>/dev/null | awk 'NR==2{print $4}')"
[ -z "$CACHE_FREE" ] && CACHE_FREE="?"

#######################################
# Cleanup
#######################################
cleanup() {
  echo "[$(date)] Cleaning up…"
  [ -n "$GRADIO_PID" ] && kill -TERM "$GRADIO_PID" 2>/dev/null && sleep 2 && kill -9 "$GRADIO_PID" 2>/dev/null
  [ -n "$VLLM_PID" ]   && kill -TERM "$VLLM_PID"   2>/dev/null && sleep 2 && kill -9 "$VLLM_PID"   2>/dev/null
  [ -n "$WATCH_PID" ]  && kill -TERM "$WATCH_PID"  2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  echo "[$(date)] Done."
}
trap cleanup EXIT INT TERM

#######################################
# Info banner
#######################################
echo "========================================"
echo "Starting vLLM + Gradio"
echo "Date: $(date)"
echo "Server: $SERVER"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Gradio Port: $GRADIO_PORT"
echo "vLLM Port: $VLLM_PORT"
echo "Model: ${VLLM_MODEL:-$VLLM_MODEL_DEFAULT}"
echo "HF_HOME: ${HF_HOME} (pre-set: ${HF_PRESET})"
echo "vLLM download dir: ${CACHE_ROOT} (free: ${CACHE_FREE})"
echo "SIF: ${SIF_PATH}"
echo "========================================"

echo "ssh -L localhost:${GRADIO_PORT}:${SERVER}:${GRADIO_PORT} -L localhost:${VLLM_PORT}:${SERVER}:${VLLM_PORT} ${USER}@neuron.ksc.re.kr" > "$PORT_FWD_FILE"

#######################################
# Env / modules / Python selection
#######################################
if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
module load gcc/10.2.0 cuda/12.1

# Best-effort conda activation (handles /scratch/$USER/miniconda3 installs)
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate vllm-hpc 2>/dev/null || true
  fi
else
  # Try common path
  if [ -f "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    . "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh"
    conda activate vllm-hpc 2>/dev/null || true
  fi
fi

PYTHON_BIN="$(command -v python3 || true)"
PY_VER_OK="$($PYTHON_BIN - <<'PY'
import sys
print(int(sys.version_info[:2] >= (3,8)))
PY
2>/dev/null)"

if [ "$PY_VER_OK" != "1" ]; then
  # Try explicit env path
  if [ -x "/scratch/${USER}/miniconda3/envs/vllm-hpc/bin/python3" ]; then
    PYTHON_BIN="/scratch/${USER}/miniconda3/envs/vllm-hpc/bin/python3"
  fi
fi

PY_VERSION="$($PYTHON_BIN -c 'import sys;print(".".join(map(str,sys.version_info[:3])))' 2>/dev/null || echo '?')"
echo "Using Python: ${PYTHON_BIN} (Python ${PY_VERSION})"
$PYTHON_BIN - <<'PY' 2>/dev/null || { echo "Python >= 3.8 required; please ensure conda env vllm-hpc is available."; exit 1; }
import sys
assert sys.version_info >= (3,8), f"Python >= 3.8 required, found {sys.version}"
PY

#######################################
# Clean stale logs / procs
#######################################
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "vllm_web.py" 2>/dev/null || true
rm -f "$VLLM_LOG" "$GRADIO_LOG"

#######################################
# Start vLLM (Singularity EXEC)
#######################################
echo "🚀 Starting vLLM server..."
cd "$WORK_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Run with positional model arg; include tqdm bars in log on first run
nohup singularity exec --nv "$SIF_PATH" \
  vllm serve "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --dtype auto \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --num-scheduler-steps "${SCHED_STEPS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --generation-config vllm \
    --trust-remote-code \
    --use-tqdm-on-load \
    --download-dir "${CACHE_ROOT}" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

#######################################
# Progress watcher (deduped)
#######################################
progress_watch() {
  stdbuf -oL tail -n +0 -F "$VLLM_LOG" 2>/dev/null \
  | awk '!seen[$0]++' \
  | while IFS= read -r line; do
      case "$line" in
        *"Resolved architecture:"*)
          echo "🔧 $(echo "$line" | sed -E 's/.*Resolved architecture: *//')"
          ;;
        *"Starting to load model "*)
          echo "📦 Loading model weights… (first run may download to ${CACHE_ROOT})"
          ;;
        *"Loading weights took "*)
          echo "✅ Weights loaded ($(echo "$line" | sed -E 's/.*Loading weights took *//') )"
          ;;
        *"Dynamo bytecode transform time:"*)
          echo "🧠 torch.compile (dynamo transform) $(echo "$line" | sed -E 's/.*time: *//')"
          ;;
        *"GPU KV cache size:"*)
          echo "🗄️ $(echo "$line" | sed -E 's/.*GPU KV cache size: *//')"
          ;;
        *"Maximum concurrency for "*)
          echo "🚦 $(echo "$line" | sed -E 's/.*Maximum concurrency for *//')"
          ;;
        *"Graph capturing finished in "*)
          secs=$(echo "$line" | sed -E 's/.*finished in *([^,]+).*/\1/')
          extra=$(echo "$line" | sed -nE 's/.*took *([^ ]+) .*$/\1/p')
          [ -n "$extra" ] && echo "🕸️ CUDA graph captured (${secs}, extra ${extra} VRAM)" || echo "🕸️ CUDA graph captured (${secs})"
          ;;
        *"Starting vLLM API server "*)
          echo "🚪 Starting HTTP server…"
          ;;
        *"Application startup complete."*)
          echo "✅ vLLM API is ready!"
          break
          ;;
      esac
    done
}

progress_watch &  WATCH_PID=$!

# Friendly "still preparing" pulses until /v1/models is up
ELAPSED=0
until curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
  sleep 10
  ELAPSED=$((ELAPSED+10))
  echo "⏳ Still preparing vLLM API… (${ELAPSED}s). Large models may take a while on first run (cache: ${CACHE_ROOT})."
done

# Stop the watcher if still running
kill "$WATCH_PID" 2>/dev/null || true

# Show init time if present
INIT_LINE=$(grep -m1 "init engine (profile, create kv cache, warmup model) took" "$VLLM_LOG" || true)
if [ -n "$INIT_LINE" ]; then
  TOOK=$(echo "$INIT_LINE" | sed -E 's/.*took *([0-9.]+) seconds/\1s/')
  echo "⏱  Engine initialization took ~${TOOK} (per vLLM)."
fi

#######################################
# Start Gradio (points to vLLM /v1)
#######################################
echo "🌐 Starting Gradio web interface..."
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export OPENAI_API_KEY="sk-local"
export DEFAULT_MODEL="${VLLM_MODEL}"

nohup "$PYTHON_BIN" vllm_web.py --host=0.0.0.0 --port="${GRADIO_PORT}" > "$GRADIO_LOG" 2>&1 &
GRADIO_PID=$!
echo "Gradio PID: $GRADIO_PID"

#######################################
# Wait for Gradio UI
#######################################
GRADIO_URL="http://127.0.0.1:${GRADIO_PORT}/"
echo "⏳ Waiting for Gradio UI at ${GRADIO_URL} ..."
GRADIO_MAX_WAIT=900
GRADIO_ELAPSED=0
while ! curl -fsS --max-time 5 "${GRADIO_URL}" >/dev/null 2>&1; do
  if ! kill -0 "$GRADIO_PID" 2>/dev/null; then
    echo "❌ Gradio process exited. Last logs:"
    tail -n 120 "$GRADIO_LOG" || true
    exit 1
  fi
  sleep 2
  GRADIO_ELAPSED=$((GRADIO_ELAPSED+2))
  if (( GRADIO_ELAPSED % 10 == 0 )); then
    echo "  ... still waiting (${GRADIO_ELAPSED}s)"
  fi
  if (( GRADIO_ELAPSED >= GRADIO_MAX_WAIT )); then
    echo "⚠️  Gradio still not responding after ${GRADIO_MAX_WAIT}s; showing recent logs and continuing."
    tail -n 200 "$GRADIO_LOG" || true
    break
  fi
done
if (( GRADIO_ELAPSED < GRADIO_MAX_WAIT )); then
  echo "✅ Gradio UI is up!"
fi

#######################################
# Summary
#######################################
echo "========================================="
echo "🎉 All services started successfully!"
echo "Gradio URL:  http://${SERVER}:${GRADIO_PORT}"
echo "Local access: http://localhost:${GRADIO_PORT} (after port forwarding)"
echo "vLLM API:    http://${SERVER}:${VLLM_PORT}/v1"
echo "Port forward for both:"
echo "ssh -L localhost:${GRADIO_PORT}:${SERVER}:${GRADIO_PORT} -L localhost:${VLLM_PORT}:${SERVER}:${VLLM_PORT} ${USER}@neuron.ksc.re.kr"
echo "Logs:"
echo "  vLLM:   $VLLM_LOG"
echo "  Gradio: $GRADIO_LOG"
echo "========================================="

#######################################
# Monitor with GPU stats & health checks
#######################################
LAST_HEARTBEAT=$(date +%s)
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date)] ERROR: vLLM process died"
    tail -60 "$VLLM_LOG"
    break
  fi
  if ! kill -0 "$GRADIO_PID" 2>/dev/null; then
    echo "[$(date)] ERROR: Gradio process died"
    tail -60 "$GRADIO_LOG"
    break
  fi

  NOW=$(date +%s)
  if (( NOW - LAST_HEARTBEAT >= 300 )); then
    echo "[$(date)] 💓 Heartbeat: services running"

    echo "🔍 GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx name used total util temp; do
      used=${used// /}; total=${total// /}; util=${util// /}; temp=${temp// /}
      mem_percent=$(( (used * 100) / (total == 0 ? 1 : total) ))
      printf "  GPU%s (%s): %sMB/%sMB (%s%%) | Util: %s%% | Temp: %s°C\n" \
        "${idx// /}" "${name}" "${used}" "${total}" "${mem_percent}" "${util}" "${temp}"
    done || true

    if curl -s --max-time 5 "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "✅ vLLM API responsive"
    else
      echo "⚠️  vLLM API not responding"
    fi

    if curl -s --max-time 5 "http://127.0.0.1:${GRADIO_PORT}" >/dev/null 2>&1; then
      echo "✅ Gradio UI responsive"
    else
      echo "⚠️  Gradio UI not responding"
    fi

    echo "----------------------------------------"
    LAST_HEARTBEAT=$NOW
  fi
  sleep 30
done

