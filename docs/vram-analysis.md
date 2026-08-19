# Why Qwen3-8B + vLLM Shows 76/80 GB VRAM Used

*Context: RunPod pod, NVIDIA A100-SXM4-80GB, vLLM serving `Qwen/Qwen3-8B` (bf16).
`nvidia-smi` reports ~76 GB used while the 8B model's weights are only ~16 GB.*

**TL;DR: 76 GB is not what the model needs — it's what vLLM deliberately
pre-allocates.** The bulk is an intentionally maximal **PagedAttention KV-cache
pool**, i.e. serving capacity, not model footprint.

## Memory breakdown

| Component | ~Size | What it is |
|---|---|---|
| Model weights | **~16 GB** | 8B params × 2 bytes (bf16) |
| Activations / CUDA graphs / workspace | ~2–4 GB | profiled peak for max batch, captured CUDA graphs, comm buffers |
| CUDA context | ~1 GB | driver/runtime overhead |
| **KV-cache block pool** | **~55–57 GB** | pre-allocated PagedAttention pages (the big one) |

Startup sequence: load weights → profiling forward pass to measure peak
activation memory → **allocate all remaining VRAM up to
`--gpu-memory-utilization` as a fixed pool of KV-cache blocks**.
76/80 ≈ 95% ⇒ the template runs `--gpu-memory-utilization 0.95`
(vLLM default is 0.90 ⇒ would land ~73 GB with overheads).

## Why vLLM grabs it all upfront

1. **KV pool = serving capacity.** Every concurrent request's context lives in
   KV cache; a bigger pool → more sequences batched → higher throughput. vLLM
   is a throughput-oriented server: idle VRAM is wasted capacity, not savings.
2. **No runtime allocation → no fragmentation, no mid-request OOM.** One big
   pool managed as fixed-size pages with block tables makes memory management
   deterministic (the core PagedAttention idea).
3. **Exact admission control.** Known block count → the scheduler can batch /
   queue / preempt requests against a precise budget.

## What the ~56 GB pool buys

Qwen3-8B: 36 layers, 8 KV heads (GQA), head_dim 128, bf16:

```
KV bytes/token = 2 (K+V) × 36 layers × 8 kv_heads × 128 dim × 2 bytes ≈ 144 KB/token
56 GB ÷ 144 KB/token ≈ ~390K tokens of KV capacity
                     ≈ ~95 concurrent requests @ 4K context each
```

So the reservation is headroom for ~100 simultaneous chats — oversized for a
single-user webapp, but that's a serving engine's default posture.

## Tuning (only if co-locating other work on the GPU)

Reserved ≠ burning anything on a dedicated pod — `nvidia-smi` memory shows
**allocation, not activity**. Leave it if the pod only serves chat. To free
VRAM for co-located jobs (e.g. training experiments):

```bash
--gpu-memory-utilization 0.45   # ~36 GB total: weights + ample KV for a few users
--max-model-len 8192            # caps per-request KV + shrinks profiled activations
--kv-cache-dtype fp8            # halves KV bytes/token (regain capacity)
```

At `0.45`, ~40 GB frees up for training while single-user chat stays snappy.

## Related

- PagedAttention paper: Kwon et al., *Efficient Memory Management for Large
  Language Model Serving with PagedAttention* (vLLM).
- Distinction worth internalizing for GPU fleet metrics: **allocated vs
  utilized** — pre-allocated pools make memory-based idle detection useless;
  use SM utilization / power draw instead.
