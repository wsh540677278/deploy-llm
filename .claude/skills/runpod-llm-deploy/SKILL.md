---
name: runpod-llm-deploy
description: Launch a vLLM OpenAI-compatible chat API for a small open-source LLM (default Qwen2.5-3B-Instruct) from INSIDE a RunPod GPU pod, and expose it via RunPod's HTTP proxy so the user can call it from their local machine. Use when running inside a RunPod pod to start/stop/health-check the model server, get the public API URL, or troubleshoot serving. Covers server launch + network access only — the user creates the pod and SSHes in themselves.
---

# RunPod In-Pod LLM Serving Skill

Division of labor:
- **User (manual, RunPod console):** create the A100 pod from a vLLM image,
  SSH in, clone this repo.
- **This skill (inside the pod):** launch the vLLM server, verify health,
  expose the OpenAI-compatible chat API through RunPod's proxy, hand back the
  public URL + key so the user can call it from local.

## Pod-creation checklist (user does this once, in the console)

The skill can only expose what the pod allows — these settings matter:

| Setting | Value | Why |
|---|---|---|
| GPU | A100 80GB, **Community Cloud** | requirement; community = cheaper |
| Image | `vllm/vllm-openai:latest` (or RunPod's vLLM template / any CUDA+Python image) | vLLM preinstalled |
| **Container Start Command** | `sleep infinity` | prevents the image entrypoint from auto-starting a server — the skill controls launch |
| **Expose HTTP Ports** | **8000** | REQUIRED — this is what makes `https://<POD_ID>-8000.proxy.runpod.net` work |
| Volume (`/workspace`) | 30 GB+ | persists model weights + API key across pod stop/start |
| SSH | enabled (default) | user's access path |

If port 8000 was not exposed at creation: edit the pod's config (or recreate).
Without it there is no proxy URL — that's the #1 gotcha.

## In-pod workflow (what the skill does)

1. `scripts/serve.sh` — launches the server. It:
   - defaults `MODEL=Qwen/Qwen2.5-3B-Instruct`, `PORT=8000`, `MAX_LEN=8192`
   - sets `HF_HOME=/workspace/huggingface` so **weights persist across
     pod stop/start** (restart = no re-download = faster + cheaper)
   - generates a `VLLM_API_KEY` if absent and persists it to
     `/workspace/.vllm_api_key` (reused on restarts)
   - if something is already healthy on the port, just reports the URL
   - otherwise starts vLLM with `nohup` (survives SSH disconnect), logs to
     `/workspace/vllm.log`, waits for `/health` to go green
   - prints the public URL: `https://$RUNPOD_POD_ID-8000.proxy.runpod.net/v1`
2. `scripts/stop_serve.sh` — stops the server process (pod keeps running).
3. Verify externally: the proxy URL's `/health` should return 200 from
   anywhere; then run `scripts/test_chat.sh` from the user's local machine.

## Calling from local (user's machine)

```bash
export API_URL="https://<POD_ID>-8000.proxy.runpod.net/v1"
export VLLM_API_KEY="<printed by serve.sh>"
./scripts/test_chat.sh                 # curl smoke test
python client/chat_client.py           # streaming chatbot REPL (pip install openai)
```

Any OpenAI SDK works: `OpenAI(base_url=API_URL, api_key=VLLM_API_KEY)`.

## Cost rules (apply always)

1. **Stop the pod from the console whenever idle** — an A100 pod bills every
   minute it runs. Stopped pods pay only pennies/day for the volume, and
   because weights + key live in `/workspace`, restart is fast:
   start pod → `serve.sh` → same key, cached weights, new POD_ID/URL.
2. Note: **POD_ID changes on stop/start** → the URL changes; re-run
   `serve.sh` to print the fresh URL and update local `API_URL`.
3. Keep `--max-model-len` modest (8192) and don't over-provision disk.
4. If usage becomes rare/bursty, suggest migrating to RunPod **Serverless
   vLLM** (scale-to-zero, $0 idle) as the cheaper long-term shape.
5. End every session by reminding the user to stop the pod.

## Troubleshooting

- **Proxy URL 404/refused:** port 8000 not in "Expose HTTP Ports" → fix pod
  config. `RUNPOD_POD_ID` env var missing → not inside a RunPod pod.
- **Proxy URL 502/503:** server still loading weights — `tail -f /workspace/vllm.log`,
  wait for "Uvicorn running on 0.0.0.0:8000".
- **CUDA OOM:** lower `MAX_LEN`, or `GPU_UTIL=0.85 ./serve.sh`.
- **`vllm: command not found`:** serve.sh falls back to
  `python3 -m vllm.entrypoints.openai.api_server`; if neither exists,
  `pip install vllm` (or use the vllm/vllm-openai image).
- **Gated model (Llama etc.):** `export HF_TOKEN=...` before serve.sh;
  default Qwen model needs none.
- **SGLang instead:** same pattern — `python3 -m sglang.launch_server
  --model-path $MODEL --host 0.0.0.0 --port 8000` — the proxy URL logic
  is identical.

## Checklist for Claude when executing this skill (inside the pod)

1. Confirm we're in a pod: `RUNPOD_POD_ID` is set; GPU visible via `nvidia-smi`.
2. Run `serve.sh`; don't duplicate servers — health-check first.
3. After launch, ALWAYS print: public API URL, where the key lives, model
   name, and the local test command.
4. Verify the **external** proxy URL responds (not just localhost) before
   declaring success.
5. Remind the user: stop the pod when done; URL changes on restart.
