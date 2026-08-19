---
name: qwen-finetune
description: Fine-tune Qwen3-8B with LoRA/QLoRA (TRL + PEFT) on the RunPod A100 pod, then merge and re-serve the tuned model through vLLM on the same chat API. Use when the user wants to fine-tune/SFT/LoRA a Qwen model, prepare training data, merge adapters, or serve a fine-tuned model. Handles the GPU handoff between the running vLLM server and training.
---

# Qwen Fine-tuning Skill (in-pod, single A100-80GB)

Pipeline: **data → (stop serving) → LoRA SFT → merge → re-serve → test → cleanup**.
All artifacts live in `/workspace` (survives pod restarts).

## Method decision (A100-80GB, Qwen3-8B)

| Method | VRAM | When |
|---|---|---|
| **LoRA bf16 (default)** | ~30-40 GiB | almost always — quality ≈ full FT for most tasks |
| **QLoRA (4-bit)** | ~15-20 GiB | OOM with LoRA, or longer seq/bigger batch needed |
| Full fine-tune | ~96+ GiB | ✗ does not fit one A100-80B; needs multi-GPU/ZeRO |

Starting hyperparameters: `r=16, alpha=32, dropout=0.05`, all linear proj targets,
`lr=2e-4` (LoRA) , 1-3 epochs, effective batch 16 (2 × grad-accum 8), seq len 2048.

## ⚠️ The GPU handoff (this pod's specific gotcha)

The template-launched vLLM server pre-allocates **76/80 GiB** (see
`docs/vram-analysis.md`). Training will OOM instantly while it runs.

- **Before training:** `pkill -f vllm` (frees the GPU; verify with `nvidia-smi`).
- **After training:** relaunch via `scripts/serve_finetuned.sh` (tuned model),
  or restart the pod to restore the original template server.
- The chat API URL stays the same either way — only the model behind it changes.

## Steps

1. **Setup once:** `scripts/setup_env.sh` — creates `/workspace/ft-venv`
   (persistent) with torch/transformers/trl/peft/bitsandbytes.
   Qwen3 needs `transformers >= 4.51`. `HF_HOME=/workspace/huggingface`
   already holds the Qwen3-8B weights from vLLM — no re-download.
2. **Data:** JSONL, one `{"messages": [{role, content}, ...]}` per line
   (see `data/example_sft.jsonl`). Put real data in `/workspace/data/`.
   Format is applied via the Qwen3 chat template with `enable_thinking=False`
   (train on plain responses; thinking-mode SFT needs reasoning traces).
3. **Train:** stop vLLM, then
   `python scripts/train_lora.py --data /workspace/data/train.jsonl --out /workspace/finetunes/<run>`
   (`--qlora` for 4-bit). Logs loss to stdout; checkpoints per epoch.
4. **Merge:** `python scripts/merge_adapter.py --adapter /workspace/finetunes/<run> --out /workspace/models/<name>`
   (merges LoRA into the base weights → a normal HF model dir vLLM can serve).
5. **Serve:** `scripts/serve_finetuned.sh /workspace/models/<name>` — kills any
   vLLM, serves the tuned model on port 8000 with the same API key. The local
   webapp/chat client works unchanged (model name = the dir path or
   `--served-model-name`). Alternative (no merge, saves disk): vLLM can serve
   the adapter directly — `--enable-lora --lora-modules tuned=<adapter_dir>`,
   request with `"model": "tuned"`.
6. **Sanity-check:** ask a held-out question through the API; compare against
   the base model's answer. For real evals, keep a small held-out set.

## Cost rules

- LoRA on a few thousand samples ≈ minutes-to-an-hour on the A100 — the pod
  bills throughout; batch your experiments.
- Everything persists in `/workspace`: venv, data, adapters, merged models —
  **stop the pod between experiment sessions**, restart and continue.
- Adapters are ~100-200 MB; merged models are ~16 GB each — prune old merges.

## Troubleshooting

- **CUDA OOM:** vLLM still running? (`nvidia-smi`, `pkill -f vllm`). Else:
  `--qlora`, halve `--batch` (raise `--grad-accum`), lower `--max-seq-len`.
- **TRL/PEFT API drift:** these libs rename args often (e.g.
  `max_seq_length`→`max_length`, `tokenizer=`→`processing_class=`). On a
  TypeError, check the installed version's signature and adapt the script.
- **Qwen3 unrecognized:** `transformers < 4.51` — upgrade in the venv.
- **Slow training:** confirm bf16 + gradient checkpointing on; optionally
  `pip install flash-attn` (long build) and `attn_implementation="flash_attention_2"`.
- **Loss not dropping:** lr too low/high, or dataset too small — LoRA on <100
  samples memorizes fast but generalizes poorly.

## Checklist for Claude when executing

1. `nvidia-smi` first — never start training with vLLM resident.
2. Verify dataset exists and parses (one JSON object with `messages` per line).
3. Run training in `nohup`/background with logs to `/workspace/ft-<run>.log`;
   report loss trajectory.
4. After serving the tuned model, verify `/health` + one chat completion, and
   print the API URL + model name for the user.
5. Remind: stop the pod when the session is done.
