#!/usr/bin/env bash
# Smoke-test the deployed chat endpoint from local.
# Usage: API_URL="https://<POD_ID>-8000.proxy.runpod.net/v1" ./test_chat.sh
set -euo pipefail
: "${API_URL:?set API_URL, e.g. https://<POD_ID>-8000.proxy.runpod.net/v1}"
: "${VLLM_API_KEY:?set VLLM_API_KEY}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"

curl -sf "${API_URL}/chat/completions" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"${MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in one sentence.\"}],
    \"max_tokens\": 64
  }" | python3 -m json.tool
