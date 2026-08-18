#!/usr/bin/env bash
# Create a COMMUNITY-cloud A100 pod running vLLM (OpenAI-compatible) on RunPod.
# Requires: RUNPOD_API_KEY, VLLM_API_KEY env vars (see .env.example).
# NOTE: GPU type IDs and API surface drift — if this errors, check
#       https://docs.runpod.io and adjust GPU_TYPE / endpoint accordingly.
set -euo pipefail

: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY (RunPod Settings -> API Keys)}"
: "${VLLM_API_KEY:?set VLLM_API_KEY (any random string; protects your endpoint)}"

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
GPU_TYPE="${GPU_TYPE:-NVIDIA A100 80GB PCIe}"   # alt: "NVIDIA A100-SXM4-80GB"
POD_NAME="${POD_NAME:-vllm-$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')}"
MAX_LEN="${MAX_LEN:-8192}"

DOCKER_ARGS="--model $MODEL --host 0.0.0.0 --port 8000 --max-model-len $MAX_LEN --api-key $VLLM_API_KEY"

read -r -d '' QUERY <<EOF || true
mutation {
  podFindAndDeployOnDemand(input: {
    cloudType: COMMUNITY,
    gpuCount: 1,
    gpuTypeId: "$GPU_TYPE",
    name: "$POD_NAME",
    imageName: "vllm/vllm-openai:latest",
    dockerArgs: "$DOCKER_ARGS",
    ports: "8000/http",
    containerDiskInGb: 30,
    volumeInGb: 0,
    minVcpuCount: 8,
    minMemoryInGb: 32
  }) { id machineId }
}
EOF

RESPONSE=$(curl -sf "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"query": sys.stdin.read()}))' <<< "$QUERY")")

echo "$RESPONSE" | python3 -m json.tool

POD_ID=$(echo "$RESPONSE" | python3 -c \
  'import json,sys; print(json.load(sys.stdin)["data"]["podFindAndDeployOnDemand"]["id"])')

cat <<INFO

✅ Pod created: $POD_ID
   Model     : $MODEL
   API URL   : https://${POD_ID}-8000.proxy.runpod.net/v1
   API key   : \$VLLM_API_KEY
   Wait ~2-5 min for image pull + weight load, then test:
     API_URL="https://${POD_ID}-8000.proxy.runpod.net/v1" ./test_chat.sh

💰 Remember: pods bill per minute while running.
   Stop when idle:  ./stop_pod.sh $POD_ID
INFO
