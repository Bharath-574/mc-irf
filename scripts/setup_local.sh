#!/usr/bin/env bash
# MC-IRF Local Setup (RTX 3050 6GB)
# Run on research center PC after git clone

set -euo pipefail

echo "=== MC-IRF Local Setup ==="

# 1. Python env
echo "Creating venv..."
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
uv sync

# 2. Download GGUF models (Phi-3-mini + Qwen2.5-7B fit in 6GB sequentially)
echo "Downloading models..."
mkdir -p models/gguf
cd models/gguf

# Phi-3-mini-3.8B (2.5GB)
wget -nc "https://huggingface.co/bartowski/Phi-3-mini-3.8B-instruct-GGUF/resolve/main/Phi-3-mini-3.8B-instruct-Q4_K_M.gguf"

# Qwen2.5-7B (4.5GB)
wget -nc "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

cd ../..

# 3. Verify GPU
echo "GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 4. Quick sanity test
echo "Running sanity test..."
python -c "
from llama_cpp import Llama
llm = Llama(model_path='models/gguf/Phi-3-mini-3.8B-instruct-Q4_K_M.gguf', n_gpu_layers=-1, n_ctx=4096, verbose=False)
out = llm('2+2=', max_tokens=10, temperature=0)
print('Phi-3-mini:', out['choices'][0]['text'].strip())
"

echo "=== Setup complete ==="
echo "Run experiments: python scripts/run_experiment.py --exp E1 --local"