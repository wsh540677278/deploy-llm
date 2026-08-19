#!/usr/bin/env bash
# Serve a fine-tuned model through vLLM on port 8000 (same public URL/key).
# Usage:
#   ./serve_finetuned.sh /workspace/models/qwen3-8b-tuned          # merged model
#   ./serve_finetuned.sh --adapter /workspace/finetunes/run1       # LoRA on base (no merge)
set -euo pipefail

PORT="${PORT:-8000}"
KEY="${VLLM_API_KEY:-sk-p9orpe46e9mvq8}"
LOG=/workspace/vllm-ft.log
export HF_HOME="${HF_HOME:-/workspace/huggingface}"

echo "🛑 stopping any running vLLM…"
pkill -f vllm 2>/dev/null || true
sleep 5

if [ "${1:-}" = "--adapter" ]; then
  ADAPTER="${2:?usage: --adapter <adapter_dir>}"
  echo "🚀 serving base Qwen/Qwen3-8B + LoRA adapter (request model: 'tuned')"
  nohup vllm serve Qwen/Qwen3-8B \
    --enable-lora --lora-modules "tuned=${ADAPTER}" \
    --host 0.0.0.0 --port "$PORT" --api-key "$KEY" \
    --max-model-len 8192 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
  MODEL_NAME="tuned"
else
  MODEL_DIR="${1:?usage: ./serve_finetuned.sh <merged_model_dir>}"
  NAME="$(basename "$MODEL_DIR")"
  echo "🚀 serving merged model $MODEL_DIR (request model: '$NAME')"
  nohup vllm serve "$MODEL_DIR" --served-model-name "$NAME" \
    --host 0.0.0.0 --port "$PORT" --api-key "$KEY" \
    --max-model-len 8192 --gpu-memory-utilization 0.90 > "$LOG" 2>&1 &
  MODEL_NAME="$NAME"
fi

echo -n "⏳ waiting for /health"
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo " up."
    echo "✅ API: https://${RUNPOD_POD_ID:-<POD_ID>}-${PORT}.proxy.runpod.net/v1  model: ${MODEL_NAME}"
    echo "   (update MODEL=${MODEL_NAME} in local .env for the webapp)"
    exit 0
  fi
  sleep 5; echo -n "."
done
echo; echo "❌ server did not become healthy — tail $LOG:"; tail -n 30 "$LOG"; exit 1
