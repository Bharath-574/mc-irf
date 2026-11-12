#!/bin/bash
# MC-IRF H200 Setup Script
# Run this in JupyterLab terminal after getting MIG slice access

set -e

echo "=========================================="
echo "MC-IRF H200 Environment Setup"
echo "Target: 1× MIG 1g.18GB slice via JupyterLab"
echo "=========================================="

# Check GPU
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Check Python version
echo "Python version:"
python --version
echo ""

# Install system dependencies (if needed)
# Note: JupyterLab usually has these pre-installed
# apt-get update && apt-get install -y git wget curl 2>/dev/null || true

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip

# Core ML stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# vLLM with AWQ support
pip install vllm

# Transformers and analysis stack
pip install transformers accelerate bitsandbytes scipy scikit-learn

# Experiment tracking
pip install pyyaml tqdm pandas matplotlib seaborn

# Statistical packages
pip install statsmodels

# JupyterLab extensions (if needed)
# pip install jupyterlab-widgets ipywidgets

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import bitsandbytes; print(f'bitsandbytes: OK')"
python -c "from llama_cpp import Llama; print('llama-cpp-python: OK')" 2>/dev/null || echo "llama-cpp-python: Not installed (local only)"

# Check GPU memory
echo ""
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv

# Setup project directory
PROJECT_DIR="/home/bharath/Research/mc-irf"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory not found at $PROJECT_DIR"
    echo "Please clone the repository first:"
    echo "  git clone <your-repo> $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# Download models
echo ""
echo "Downloading models..."
python scripts/download_models.py

# Verify model configs
echo ""
echo "Verifying model configs..."
python -c "
from mc_irf.models.registry import get_registry
reg = get_registry('configs/models_h200.yaml')
print('Available models:', reg.list_models())
print('FP16 models:', reg.list_models(backend='vllm', max_size_gb=17))
print('AWQ models:', [m for m in reg.list_models(backend='vllm') if 'awq' in m])
print('HF models:', reg.list_models(backend='hf'))
"

# Validate pipeline
echo ""
echo "Running pipeline validation..."
python scripts/validate_pipeline.py

echo ""
echo "=========================================="
echo "H200 Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run local validation: python scripts/validate_pipeline.py --test-vllm"
echo "  2. Start with E0: jupyter notebook notebooks/01_E0_law_calibration.ipynb"
echo "  3. Or run experiment directly: python -m mc_irf.experiments.runner E0 h200"
echo ""
echo "Monitor GPU: nvidia-smi -l 1"