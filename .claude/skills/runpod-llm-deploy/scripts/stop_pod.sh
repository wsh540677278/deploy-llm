#!/usr/bin/env bash
# Stop a RunPod pod (keeps disk, stops GPU billing). Usage: ./stop_pod.sh <POD_ID>
set -euo pipefail
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
POD_ID="${1:?usage: ./stop_pod.sh <POD_ID>}"

curl -sf "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$POD_ID\\\"}) { id desiredStatus } }\"}" \
  | python3 -m json.tool

echo "🛑 Pod $POD_ID stopped (GPU billing halted; small disk fee remains)."
echo "   Terminate fully (zero cost) from console, or podTerminate mutation."
