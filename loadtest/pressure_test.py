"""KV-cache pressure test: drive vLLM's GPU KV cache usage toward 100%.

Sends N concurrent long-context requests with UNIQUE prefixes (a UUID at
token 0 defeats prefix caching, so every request occupies its own KV blocks),
while polling the server's /metrics endpoint to display live KV usage.

Self-calibrates prompt sizing against the server's tokenizer (one probe request
reading usage.prompt_tokens), so --prompt-tokens means real tokens.

Usage (defaults sized to saturate an A100-80GB Qwen3-8B pool = 433,520 tokens,
max context 8128; 72 x (5800+1200) = 504K demand -> ~100% usage + queueing):
    python pressure_test.py
    python pressure_test.py --concurrency 16     # partial fill (~25%)
Config from ../.env: API_URL, VLLM_API_KEY, MODEL.
Ctrl-C aborts. Expect several minutes at full defaults; watch the pod's
loggers.py lines too (Running/Waiting/KV usage should mirror this output).
"""
import argparse
import asyncio
import os
import random
import re
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango").split()


def load_env():
    env = ROOT.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.split("#")[0].strip())


def make_prompt(n_words: int) -> str:
    # UUID FIRST -> unique token 0 -> zero prefix-cache sharing between requests.
    filler = " ".join(random.choice(WORDS) for _ in range(n_words))
    return (f"{uuid.uuid4()} You are a storyteller. Using the word list below, "
            f"write the longest story you can.\n{filler}")


async def calibrate(client, url, key, model, target_tokens: int) -> int:
    """Measure the server tokenizer's tokens/word on our filler and return the
    word count that yields ~target_tokens prompt tokens (3% safety margin)."""
    probe_words = 1000
    r = await client.post(f"{url}/chat/completions",
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "max_tokens": 1,
                                "messages": [{"role": "user",
                                              "content": make_prompt(probe_words)}]})
    r.raise_for_status()
    measured = r.json()["usage"]["prompt_tokens"]
    ratio = measured / probe_words                    # includes template overhead -> conservative
    n_words = int(target_tokens / ratio * 0.97)
    print(f"calibration: {probe_words} words -> {measured} tokens "
          f"(ratio {ratio:.2f}) -> using {n_words} words/request")
    return n_words


async def one_request(client, url, key, model, args, stats):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": make_prompt(args.n_words)}],
        "max_tokens": args.max_tokens,
        "temperature": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},   # thinking = extra KV growth
    }
    try:
        r = await client.post(f"{url}/chat/completions", json=body,
                              headers={"Authorization": f"Bearer {key}"})
        stats["ok" if r.status_code == 200 else "err"] += 1
        if r.status_code != 200:
            stats.setdefault("last_err", r.text[:200])
    except Exception as e:  # timeouts/aborts still served their KV-pressure purpose
        stats["err"] += 1
        stats.setdefault("last_err", repr(e)[:200])


async def monitor(client, base, key, stop, stats):
    """Poll /metrics; print live KV usage / running / waiting."""
    # metric is kv_cache_usage_perc on recent vLLM, gpu_cache_usage_perc on older
    pat_usage = re.compile(r"(?:kv|gpu)_cache_usage\w*(?:{[^}]*})?\s+([0-9.eE+-]+)")
    pat_run = re.compile(r"num_requests_running(?:{[^}]*})?\s+([0-9.eE+-]+)")
    pat_wait = re.compile(r"num_requests_waiting(?:{[^}]*})?\s+([0-9.eE+-]+)")
    warned = False
    while not stop.is_set():
        try:
            r = await client.get(f"{base}/metrics",
                                 headers={"Authorization": f"Bearer {key}"})
            m = pat_usage.search(r.text)
            if m:
                usage = float(m.group(1)) * 100
                stats["peak"] = max(stats.get("peak", 0.0), usage)
                run = pat_run.search(r.text)
                wait = pat_wait.search(r.text)
                bar = "#" * int(usage // 2)
                print(f"\rKV cache: {usage:5.1f}% |{bar:<50}| "
                      f"running={int(float(run.group(1))) if run else '?':>3} "
                      f"waiting={int(float(wait.group(1))) if wait else '?':>3} "
                      f"done={stats['ok']} err={stats['err']}   ", end="", flush=True)
            elif not warned:
                warned = True
                print("\n(no gpu_cache_usage metric found at /metrics — "
                      "watch the pod's loggers.py log lines instead)")
        except Exception:
            pass  # transient poll failure; keep going
        await asyncio.sleep(2)


async def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__)
    # Server max context is 8128; 5800 + 1200 = 7000/request stays safely under,
    # and 72 x 7000 = 504K tokens > the 433K-token KV pool -> saturation + queueing.
    ap.add_argument("--concurrency", type=int, default=72)
    ap.add_argument("--prompt-tokens", type=int, default=5800)
    ap.add_argument("--max-tokens", type=int, default=1200)
    args = ap.parse_args()

    url = os.environ["API_URL"].rstrip("/")
    base = url.removesuffix("/v1")
    key = os.environ["VLLM_API_KEY"]
    model = os.environ.get("MODEL", "Qwen/Qwen3-8B")

    total = args.concurrency * (args.prompt_tokens + args.max_tokens)
    print(f"target: {url}  model: {model}")
    print(f"wave: {args.concurrency} concurrent x (~{args.prompt_tokens} prompt "
          f"+ {args.max_tokens} gen) ≈ {total/1000:.0f}K tokens of KV demand")
    print("Ctrl-C to abort. Watch the bar (and the pod logs) …\n")

    stats = {"ok": 0, "err": 0}
    stop = asyncio.Event()
    t0 = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30)) as client:
        args.n_words = await calibrate(client, url, key, model, args.prompt_tokens)
        mon = asyncio.create_task(monitor(client, base, key, stop, stats))
        try:
            await asyncio.gather(*(one_request(client, url, key, model, args, stats)
                                   for _ in range(args.concurrency)))
        finally:
            stop.set()
            await mon
    print(f"\n\ndone in {time.time()-t0:.0f}s | ok={stats['ok']} err={stats['err']} "
          f"| PEAK KV cache usage: {stats.get('peak', 0.0):.1f}%")
    if "last_err" in stats:
        print(f"last error: {stats['last_err']}")


if __name__ == "__main__":
    asyncio.run(main())
