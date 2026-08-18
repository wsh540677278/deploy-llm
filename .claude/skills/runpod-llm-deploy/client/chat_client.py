"""Minimal local chatbot client for the RunPod-deployed endpoint.

Usage:
    export API_URL="https://<POD_ID>-8000.proxy.runpod.net/v1"
    export VLLM_API_KEY="<your key>"
    pip install openai
    python chat_client.py
"""
import os

from openai import OpenAI

API_URL = os.environ["API_URL"]
API_KEY = os.environ["VLLM_API_KEY"]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-3B-Instruct")

client = OpenAI(base_url=API_URL, api_key=API_KEY)
history = []

print(f"Chatting with {MODEL} @ {API_URL}  (Ctrl-C or 'exit' to quit)")
while True:
    try:
        user = input("you > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user or user.lower() in {"exit", "quit"}:
        break
    history.append({"role": "user", "content": user})
    stream = client.chat.completions.create(
        model=MODEL, messages=history, stream=True, max_tokens=1024
    )
    print("bot > ", end="", flush=True)
    reply = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        reply.append(delta)
        print(delta, end="", flush=True)
    print()
    history.append({"role": "assistant", "content": "".join(reply)})
