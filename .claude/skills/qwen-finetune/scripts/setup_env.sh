#!/usr/bin/env bash
# One-time training env setup, persisted in /workspace (survives pod restarts).
set -euo pipefail

VENV=/workspace/ft-venv
export HF_HOME="${HF_HOME:-/workspace/huggingface}"   # reuse vLLM's model cache

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install -q --upgrade pip
# torch usually ships in the vLLM image's system python but NOT in a fresh venv;
# install the CUDA build (large; one-time — venv persists in /workspace).
pip install -q torch --index-url https://download.pytorch.org/whl/cu121 || pip install -q torch
pip install -q "transformers>=4.51" datasets accelerate peft trl bitsandbytes

python - <<'EOF'
import torch, transformers, trl, peft
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} "
      f"| transformers {transformers.__version__} | trl {trl.__version__} | peft {peft.__version__}")
EOF
echo "✅ env ready: source $VENV/bin/activate   (HF_HOME=$HF_HOME)"
