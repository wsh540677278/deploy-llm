"""Merge a LoRA adapter into the base model -> a plain HF dir vLLM can serve.

    python merge_adapter.py --adapter /workspace/finetunes/run1 \
        --out /workspace/models/qwen3-8b-tuned
Runs on CPU (no GPU needed; safe while anything else uses the GPU). ~16 GB out.
"""
import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"loading base {BASE} on CPU (bf16)…")
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cpu")
    print(f"applying adapter {args.adapter} …")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()          # bake LoRA deltas into the weights

    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"✅ merged model -> {args.out}")
    print(f"serve it:  ./serve_finetuned.sh {args.out}")


if __name__ == "__main__":
    main()
