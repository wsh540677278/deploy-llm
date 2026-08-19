"""Minimal chat webapp backed by the RunPod vLLM endpoint.

Keeps the API key server-side and streams tokens to the browser via SSE.

Run:
    pip install -r requirements.txt
    python app.py            # http://localhost:7860
Config comes from ../.env or environment: API_URL, VLLM_API_KEY, MODEL.
"""
import json
import os
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

ROOT = Path(__file__).parent


def load_env() -> None:
    """Tiny .env loader (repo root), no python-dotenv dependency."""
    env_file = ROOT.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.split("#")[0].strip())


load_env()
API_URL = os.environ["API_URL"].rstrip("/")          # e.g. https://<pod>-8000.proxy.runpod.net/v1
API_KEY = os.environ["VLLM_API_KEY"]
MODEL = os.environ.get("MODEL", "")                   # discovered from /v1/models if empty

app = FastAPI()


async def resolve_model() -> str:
    """If MODEL unset, ask the backend what it's serving."""
    global MODEL
    if not MODEL:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{API_URL}/models",
                                 headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
            r.raise_for_status()
            MODEL = r.json()["data"][0]["id"]
    return MODEL


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/config")
async def config():
    return {"model": await resolve_model()}


@app.post("/api/chat")
async def chat(body: dict):
    """Proxy chat messages to vLLM, streaming SSE straight through."""
    payload = {
        "model": await resolve_model(),
        "messages": body["messages"],
        "stream": True,
        "max_tokens": body.get("max_tokens", 2048),
        "temperature": body.get("temperature", 0.7),
        # Qwen3 is a hybrid thinking model; skip <think> blocks for snappy chat.
        # Flip to True (or remove) if you want chain-of-thought in responses.
        "chat_template_kwargs": {"enable_thinking": body.get("thinking", False)},
    }

    async def stream():
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as client:
            async with client.stream(
                "POST", f"{API_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json=payload,
            ) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode(errors="replace")[:500]
                    yield f"data: {json.dumps({'error': f'{r.status_code}: {detail}'})}\n\n"
                    return
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        yield line + "\n\n"     # pass OpenAI SSE frames through untouched

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    print(f"backend: {API_URL}  model: {MODEL or '(auto-discover)'}")
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("WEBAPP_PORT", 7860)))
