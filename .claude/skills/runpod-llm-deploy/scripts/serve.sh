#!/usr/bin/env bash
# Launch vLLM (OpenAI-compatible) INSIDE a RunPod pod and expose it via the
# RunPod HTTP proxy. Safe to re-run: if a healthy server is up, just prints info.
#
# Usage (inside the pod):  ./serve.sh
# Overrides: MODEL=... PORT=... MAX_LEN=... GPU_UTIL=... ./serve.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
LOG="${LOG:-/workspace/vllm.log}"
PID_FILE="/workspace/vllm.pid"
KEY_FILE="/workspace/.vllm_api_key"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"   # secs; first run includes weight download

# --- sanity: are we inside a RunPod pod with a GPU? ---
if [ -z "${RUNPOD_POD_ID:-}" ]; then
  echo "⚠️  RUNPOD_POD_ID not set — not inside a RunPod pod? Proxy URL can't be derived." >&2
fi
nvidia-smi -L || { echo "❌ no GPU visible"; exit 1; }

# --- persist HF cache on the volume: survive pod stop/start, no re-download ---
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
mkdir -p "$HF_HOME"

# --- API key: reuse persisted one, else generate ---
if [ -z "${VLLM_API_KEY:-}" ]; then
  if [ -f "$KEY_FILE" ]; then
    VLLM_API_KEY="$(cat "$KEY_FILE")"
  else
    VLLM_API_KEY="$(openssl rand -hex 16)"
    umask 077 && echo "$VLLM_API_KEY" > "$KEY_FILE"
    echo "🔑 generated new API key -> $KEY_FILE"
  fi
fi
export VLLM_API_KEY

PUBLIC_URL="https://${RUNPOD_POD_ID:-<POD_ID>}-${PORT}.proxy.runpod.net"

print_summary() {
  cat <<INFO

✅ Chat API is live
   Model      : $MODEL
   Public URL : ${PUBLIC_URL}/v1
   API key    : $KEY_FILE  (value: \$VLLM_API_KEY)
   Log        : $LOG

   Test from your LOCAL machine:
     export API_URL="${PUBLIC_URL}/v1"
     export VLLM_API_KEY="$VLLM_API_KEY"
     ./scripts/test_chat.sh          # or: python client/chat_client.py

💰 Stop the pod in the RunPod console when idle (URL changes on restart;
   weights are cached in /workspace so restart is fast — just re-run serve.sh).
INFO
}

# --- already running & healthy? just report ---
if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "ℹ️  server already healthy on port ${PORT} — not starting another."
  print_summary
  exit 0
fi

# --- launch (nohup: survives SSH disconnect) ---
if command -v vllm >/dev/null 2>&1; then
  CMD=(vllm serve "$MODEL")
else
  CMD=(python3 -m vllm.entrypoints.openai.api_server --model "$MODEL")
fi
CMD+=(--host 0.0.0.0 --port "$PORT" --max-model-len "$MAX_LEN"
      --gpu-memory-utilization "$GPU_UTIL" --api-key "$VLLM_API_KEY")

echo "🚀 launching: ${CMD[*]}"
nohup "${CMD[@]}" > "$LOG" 2>&1 &
echo $! > "$PID_FILE"
echo "   pid $(cat "$PID_FILE"), log -> $LOG"

# --- wait for health (first run downloads ~6GB of weights) ---
echo -n "⏳ waiting for /health"
elapsed=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo; echo "❌ server process died — last log lines:"; tail -n 30 "$LOG"; exit 1
  fi
  if [ "$elapsed" -ge "$BOOT_TIMEOUT" ]; then
    echo; echo "❌ timed out after ${BOOT_TIMEOUT}s — last log lines:"; tail -n 30 "$LOG"; exit 1
  fi
  sleep 5; elapsed=$((elapsed + 5)); echo -n "."
done
echo " up (${elapsed}s)"

# --- verify EXTERNAL proxy path, not just localhost ---
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  if curl -sf "${PUBLIC_URL}/health" >/dev/null 2>&1; then
    echo "🌐 external proxy check OK"
  else
    echo "⚠️  localhost healthy but ${PUBLIC_URL}/health failed."
    echo "    Most likely port ${PORT} is NOT in the pod's 'Expose HTTP Ports' — fix in console."
  fi
fi

print_summary
