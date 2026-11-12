#!/usr/bin/env python3
# MC-IRF Model Downloader for H200
# Pre-downloads models to HF cache to avoid delays during experiments

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import snapshot_download
import yaml

def load_models_config():
    with open("configs/models_h200.yaml") as f:
        return yaml.safe_load(f)

def download_model(hf_id: str, local_dir: str = None):
    """Download model from HF Hub to cache."""
    print(f"Downloading {hf_id}...")
    try:
        path = snapshot_download(
            repo_id=hf_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  ✓ Downloaded to {path}")
        return path
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None

def main():
    config = load_models_config()
    
    # Models that fit in single 16GB MIG slice (E1-E6, E8)
    single_slice_models = [
        "phi3-mini-3.8b",
        "qwen2.5-7b", 
        "llama3-8b",
        "qwen2.5-14b",
    ]
    
    # Models needing 2 slices (TP=2) - for E7, E9
    multi_slice_models = [
        "qwen2.5-32b",
        "llama3-70b",
    ]
    
    print("=" * 60)
    print("MC-IRF Model Download (H200)")
    print("=" * 60)
    
    # Download single-slice models first (priority)
    print("\n📥 Single-slice models (E1-E6, E8):")
    for model_key in single_slice_models:
        if model_key in config["models"]:
            hf_id = config["models"][model_key]["hf_id"]
            download_model(hf_id)
    
    # Ask about multi-slice models
    print("\n📥 Multi-slice models (E7, E9) - need 2 MIG slices:")
    for model_key in multi_slice_models:
        if model_key in config["models"]:
            hf_id = config["models"][model_key]["hf_id"]
            response = input(f"Download {hf_id} ({config['models'][model_key]['size_gb']}GB)? [y/N]: ")
            if response.lower() == 'y':
                download_model(hf_id)
            else:
                print(f"  ⏭ Skipped {hf_id}")
    
    print("\n✅ Download complete!")
    print(f"Models cached in: {os.environ.get('HF_HOME', '~/.cache/huggingface')}")

if __name__ == "__main__":
    main()