# Why Qwen3-8B + vLLM Shows 76/80 GB VRAM Used

*Context: RunPod pod, NVIDIA A100-SXM4-80GB, vLLM serving `Qwen/Qwen3-8B` (bf16).
`nvidia-smi` reports ~76 GB used while the 8B model's weights are only ~16 GB.*

**TL;DR: 76 GB is not what the model needs — it's what vLLM deliberately
pre-allocates.** The bulk is an intentionally maximal **PagedAttention KV-cache
pool** — serving capacity, not model footprint. Confirmed empirically below:
a pressure test drove the pool to **99.9% usage with zero errors**.

## Ground truth from the server itself

vLLM publishes its startup accounting in the `/metrics` endpoint's
`vllm:cache_config_info` gauge. Ours reports:

```
gpu_memory_utilization = 0.95          # grab 95% of VRAM
block_size             = 16            # tokens per PagedAttention block
num_gpu_blocks         = 27095
kv_cache_size_tokens   = 433520        # = 27,095 blocks × 16 tokens
kv_cache_max_concurrency = 53.336      # = 433,520 ÷ max_model_len
enable_prefix_caching  = True
```

Two derived facts: the **KV pool holds exactly 433,520 tokens**, and
`433,520 ÷ 53.336 = 8128` ⇒ the server's **max context length is 8128**.

## Where 433,520 comes from (vLLM's startup arithmetic)

```
1. Budget      = total VRAM × gpu_memory_utilization = 80 GiB × 0.95 = 76.0 GiB
2. − Weights   = 8.2B params × 2 B (bf16)                           ≈ 15.3 GiB
3. − Profiled activation peak + CUDA graphs + context               ≈  1.2 GiB
      (measured by a dummy forward pass at startup)
4. = KV pool                                                        ≈ 59.5 GiB
5. ÷ KV bytes/token (Qwen3-8B: 2 (K+V) × 36 layers × 8 KV heads
      × 128 head_dim × 2 B)                        = 147,456 B ≈ 144 KiB/token
6. 59.5 GiB ÷ 144 KiB/token ≈ 433K tokens → round down to whole blocks
   → 27,095 blocks × 16 = 433,520 tokens
```

Reconciliation check: 433,520 × 147,456 B = **59.5 GiB pool** + 15.3 GiB
weights + ~1.2 GiB activations = 76 GiB budget ✓ (use GiB consistently —
decimal-GB napkin math will look ~5% off).

## Memory breakdown (GiB-accurate)

| Component | Size | What it is |
|---|---|---|
| Model weights | 15.3 GiB | 8.2B params × 2 bytes (bf16) |
| Activations / CUDA graphs / context | ~1.2 GiB | profiled peak + captured graphs + runtime |
| **KV-cache block pool** | **59.5 GiB** | 27,095 × 16-token PagedAttention pages |
| **Total** | **76 GiB** | = 80 GiB × `gpu_memory_utilization 0.95` |

## Why vLLM grabs it all upfront

1. **KV pool = serving capacity.** Every concurrent request's context lives in
   KV cache; bigger pool → more sequences batched → higher throughput.
2. **No runtime allocation → no fragmentation, no mid-request OOM.** One pool,
   fixed-size pages, block tables (the core PagedAttention idea).
3. **Exact admission control.** Known block count → the scheduler can batch,
   queue, and preempt against a precise budget. (Demonstrated below.)

## Empirical validation: pressure test to 99.9%

`loadtest/pressure_test.py` saturates the pool: **72 concurrent requests ×
(~5800 prompt + 1200 gen) ≈ 504K tokens of demand > 433K capacity**. Each
prompt starts with a UUID (unique token 0) so prefix caching can't dedupe
blocks, and prompt sizing **self-calibrates** against the server tokenizer via
a probe request reading `usage.prompt_tokens` (word-count heuristics were off
by 20–60% and caused max-context rejections).

Observed (run of 2026-08-19, 111 s total, 72/72 ok):

```
SATURATION:  KV 99.9% | running=66-69 | waiting=3-6   ← queueing at the budget
DRAIN:       97% → 95% (waiting→0) → 68% → 32% → 9% → done
PEAK: 99.9%   errors: 0
```

Takeaways:

- The pool is a **hard budget**: usage plateaus at ~100%, never OOMs; excess
  requests sit in `waiting` until finishers free blocks — admission control
  against `num_gpu_blocks`, live.
- `running` capped ~66–69 (above `kv_cache_max_concurrency = 53.3` because
  most requests used ~7K < the 8128 max assumed by that bound).
- Metric note: usage gauge is `vllm:kv_cache_usage_perc` on this version
  (`gpu_cache_usage_perc` on older releases).

## Tuning (only if co-locating other work on the GPU)

Reserved ≠ burning anything on a dedicated pod — `nvidia-smi` memory shows
**allocation, not activity**. Leave it if the pod only serves chat. To free
VRAM for co-located jobs (e.g. training experiments):

```bash
--gpu-memory-utilization 0.45   # ~36 GiB total: weights + ample KV for a few users
--max-model-len 8192            # caps per-request KV + shrinks profiled activations
--kv-cache-dtype fp8            # halves KV bytes/token (regain capacity)
```

## Related

- PagedAttention paper: Kwon et al., *Efficient Memory Management for Large
  Language Model Serving with PagedAttention* (vLLM).
- Repro tool: [`loadtest/pressure_test.py`](../loadtest/pressure_test.py) —
  live `/metrics` monitor + saturation wave.
- Fleet-metrics lesson: **allocated vs utilized** — pre-allocated pools make
  memory-based idle detection useless; watch SM utilization / power draw, and
  `kv_cache_usage_perc` (occupancy) vs `prefix cache hit rate` (compute reuse).
