# deploy-llm

Serve a small open-source LLM (default **Qwen2.5-3B-Instruct**) as a chat API
from a **RunPod A100 pod**, callable from your local machine.

Workflow: **you** create the pod + SSH in; the **skill** (run inside the pod)
handles server launch + network exposure and hands you the public URL.

## 1. Create the pod (RunPod console — once)

- GPU: **A100 80GB**, Community Cloud (cheaper)
- Image: `vllm/vllm-openai:latest`
- Container Start Command: `sleep infinity`  ← so the skill controls the server
- **Expose HTTP Ports: `8000`**  ← required for the public URL
- Volume `/workspace`: 30 GB+ (persists weights/key across stop/start)

## 2. Inside the pod

```bash
ssh <pod>                                  # from RunPod console's connect tab
git clone <this-repo-url> && cd deploy-llm
./.claude/skills/runpod-llm-deploy/scripts/serve.sh
# → prints:  Public URL: https://<POD_ID>-8000.proxy.runpod.net/v1  + API key
```

(Or run Claude Code in the pod and just ask it to deploy — the skill in
`.claude/skills/runpod-llm-deploy/` drives the same flow.)

## 3. From your local machine

```bash
export API_URL="https://<POD_ID>-8000.proxy.runpod.net/v1"
export VLLM_API_KEY="<key from serve.sh>"
./.claude/skills/runpod-llm-deploy/scripts/test_chat.sh     # smoke test
python .claude/skills/runpod-llm-deploy/client/chat_client.py  # chat REPL
```

## 💰 Cost

- **Stop the pod in the console whenever idle** — that's the whole game.
- Weights are cached in `/workspace`: restart = start pod → `serve.sh` again
  (note: POD_ID and therefore the URL change after a restart).
- Bursty/rare usage? Consider RunPod **Serverless vLLM** (scale-to-zero) instead.

See `.claude/skills/runpod-llm-deploy/SKILL.md` for full details and troubleshooting.
