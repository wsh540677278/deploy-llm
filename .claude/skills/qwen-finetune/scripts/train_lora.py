"""LoRA / QLoRA SFT for Qwen3 on a single A100.

Data: JSONL, one {"messages": [{"role": ..., "content": ...}, ...]} per line.
Run (inside /workspace/ft-venv, with vLLM STOPPED — check nvidia-smi first):
    python train_lora.py --data /workspace/data/train.jsonl \
        --out /workspace/finetunes/run1 [--qlora] [--epochs 2]

NOTE: TRL/PEFT rename kwargs across versions (max_seq_length vs max_length,
tokenizer vs processing_class). If a TypeError fires, adapt to the installed
version — see SKILL.md troubleshooting.
"""
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/workspace/finetunes/run1")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--qlora", action="store_true", help="4-bit base (NF4)")
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(BASE)

    ds = load_dataset("json", data_files=args.data, split="train")

    def to_text(ex):
        # Plain-response SFT: disable Qwen3 thinking in the template.
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)
    print(f"dataset: {len(ds)} samples | sample[0][:300]:\n{ds[0]['text'][:300]}")

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, quantization_config=quant,
        device_map="auto")
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False          # incompatible with grad checkpointing

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, gradient_checkpointing=True,
        max_seq_length=args.max_seq_len,      # older TRL; newer: max_length
        dataset_text_field="text",
        logging_steps=5, save_strategy="epoch", report_to="none")

    trainer = SFTTrainer(model=model, train_dataset=ds, peft_config=lora, args=cfg)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"✅ adapter saved -> {args.out}")
    print(f"next: merge_adapter.py --adapter {args.out} --out /workspace/models/<name>")


if __name__ == "__main__":
    main()
