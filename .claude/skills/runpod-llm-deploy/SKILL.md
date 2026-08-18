---
name: runpod-llm-deploy
description: Deploy a small open-source LLM (default Qwen2.5-3B-Instruct) on RunPod with an A100 GPU via vLLM, exposing an OpenAI-compatible chat API URL callable from the user's local machine, at minimal cost. Use when the user wants to deploy/serve an LLM on RunPod, get the inference API URL, test the chat endpoint, or stop/terminate the deployment to save money.
---

# RunPod LLM Deployment Skill

Goal: stand up a **chat-style (OpenAI-compatible) API** on RunPod backed by an
**A100 GPU**, return the **API URL + key** for local calls, and keep **cost as
low as possible**.

## Key decisions (defaults)

| Decision | Default | Why |
|---|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` | small, strong, Apache-2.0, no HF gating |
| Framework | **vLLM** (`vllm/vllm-openai:latest` image) | OpenAI-compatible `/v1/chat/completions` out of the box; RunPod also has a pre-built vLLM template. SGLang (`lmsysorg/sglang:latest`) is a fine alternative — same OpenAI-compatible API |
| GPU | A100 80GB (user requirement) | note: a 3B model fits easily on a cheaper RTX 4090/A5000 — offer this to cut cost ~3-4x if user is flexible |
| Cloud type | **COMMUNITY** cloud | cheaper than SECURE cloud |
| API style | OpenAI-compatible chat completions | works with any OpenAI SDK/client from local |

## Cost minimization rules (IMPORTANT — apply always)

1. **Prefer RunPod Serverless vLLM endpoint** for intermittent chat use:
   flex workers **scale to zero** — you pay per request-second, $0 while idle.
   A pod bills every minute it runs, even idle.
2. If using a Pod: **COMMUNITY cloud**, and **stop the pod whenever idle**
   (stopped pods only bill disk, ~cents/day). Terminate fully when done.
3. Keep `--max-model-len` modest (e.g. 8192) — smaller KV cache, faster start.
4. Don't over-provision disk: 20–30 GB container disk is plenty for a 3B model.
5. Remind the user at the end of every deploy: "stop it when you're done."

## Prerequisites

- RunPod account with credits: https://runpod.io
- `RUNPOD_API_KEY` env var (create at Settings → API Keys)
- Choose a `VLLM_API_KEY` (any random string) to protect the endpoint.
- Put both in `.env` (git-ignored). Never commit keys.

## Path A — GPU Pod (always-on URL, simplest; use scripts/)

1. Verify current GPU type IDs and pricing first (they drift):
   query RunPod's API or check the console pricing page. If anything in the
   scripts 404s or errors, check https://docs.runpod.io — the API surface has
   both a GraphQL endpoint (`api.runpod.io/graphql`) and a newer REST API;
   prefer whichever the current docs recommend.
2. Run `scripts/create_pod.sh` — creates a COMMUNITY-cloud A100 pod running
   `vllm/vllm-openai:latest` serving the model on port 8000 (exposed via
   RunPod's HTTP proxy).
3. Wait for the pod to pull the image + load weights (~2–5 min). The API URL is:

   `https://<POD_ID>-8000.proxy.runpod.net/v1`

4. Test with `scripts/test_chat.sh` (or `client/chat_client.py` locally).
5. When the user is done: `scripts/stop_pod.sh <POD_ID>` (stop = keep disk,
   minimal cost) or terminate from the console ($0).

## Path B — Serverless vLLM endpoint (cheapest for intermittent use)

RunPod has a **pre-built "Serverless vLLM" quick-deploy**: Console → Serverless
→ Quick Deploy → vLLM. Configure:
- Model: `Qwen/Qwen2.5-3B-Instruct`
- GPU: A100 80GB; **Active workers: 0**, Max workers: 1 (scale-to-zero)
- The endpoint URL is `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`
  and is OpenAI-compatible; auth uses the RunPod API key as Bearer token.

Cold starts add ~30–60s on the first request after idle — the price of $0 idle
cost. Recommend Path B when the user's usage is bursty/occasional (typical
personal chatbot), Path A when they want a warm, always-on endpoint.

## Calling from local (either path)

```python
from openai import OpenAI
client = OpenAI(base_url="<API_URL>", api_key="<KEY>")
r = client.chat.completions.create(
    model="Qwen/Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "hello"}],
)
print(r.choices[0].message.content)
```

## Troubleshooting

- **502/503 from proxy URL**: pod still booting or vLLM still loading weights —
  check pod logs in console; wait for "Uvicorn running on 0.0.0.0:8000".
- **CUDA OOM**: lower `--max-model-len`, or `--gpu-memory-utilization 0.90`.
- **Gated model (Llama)**: needs `HF_TOKEN` env var in the pod; Qwen doesn't.
- **GPU unavailable on COMMUNITY**: retry, or fall back to SECURE cloud /
  another A100 variant (`NVIDIA A100-SXM4-80GB` vs `NVIDIA A100 80GB PCIe`).

## Checklist for Claude when executing this skill

1. Confirm `RUNPOD_API_KEY` is set; generate a `VLLM_API_KEY` if absent.
2. Ask (or infer) Path A vs B based on usage pattern; default to B for cost.
3. Verify current GPU IDs/pricing before creating resources.
4. After deploy, ALWAYS print: API URL, API key env var name, model name,
   estimated $/hr, and the stop/terminate command.
5. Never leave a pod running after the user says they're done.
