#!/usr/bin/env bash
# Stop the vLLM server inside the pod (the pod itself keeps running/billing).
set -euo pipefail
PID_FILE="/workspace/vllm.pid"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  kill "$(cat "$PID_FILE")" && rm -f "$PID_FILE"
  echo "🛑 vLLM server stopped."
else
  pkill -f "vllm.entrypoints.openai.api_server|vllm serve" 2>/dev/null \
    && echo "🛑 vLLM server stopped (found by name)." \
    || echo "ℹ️  no running vLLM server found."
fi

echo "💰 Reminder: the POD is still running and billing —"
echo "   stop it from the RunPod console if you're done for now."
