#!/usr/bin/env python3
"""
MC-IRF Model Download Script for H200 JupyterLab
Downloads models needed for experiments on 1× MIG 18GB slice.
"""

import subprocess
import sys
import yaml
from pathlib import Path


def download_model(hf_id: str, local_dir: str = None):
    """Download model using huggingface-cli."""
    cmd = ["huggingface-cli", "download", hf_id]
    if local_dir:
        cmd.extend(["--local-dir", local_dir])
    
    print(f"Downloading {hf_id}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Downloaded {hf_id}")
        return True
    else:
        print(f"  ✗ Failed: {result.stderr}")
        return False


def download_gguf_model(repo: str, file: str, local_dir: str):
    """Download specific GGUF file from HF repo."""
    cmd = ["huggingface-cli", "download", repo, file, "--local-dir", local_dir]
    
    print(f"Downloading {repo}/{file}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"  ✓ Downloaded {file}")
        return True
    else:
        print(f"  ✗ Failed: {result.stderr}")
        return False


def main():
    # Load model configs
    with open("configs/models_h200.yaml") as f:
        config = yaml.safe_load(f)
    
    models = config.get("models", {})
    download_priority = config.get("download_priority", {})
    
    print("=" * 60)
    print("MC-IRF Model Download for H200 MIG 18GB")
    print("=" * 60)
    
    # Phase 1: Essential models
    print("\n📥 Phase 1: Essential Models")
    for key in download_priority.get("phase1_essential", []):
        if key in models:
            spec = models[key]
            download_model(spec["hf_id"])
    
    # Phase 2: Scale models (AWQ)
    print("\n📥 Phase 2: Scale Models (AWQ 4-bit)")
    for key in download_priority.get("phase2_scale", []):
        if key in models:
            spec = models[key]
            download_model(spec["hf_id"])
    
    # Phase 3: Analysis models (HF)
    print("\n📥 Phase 3: Analysis Models (HF 8-bit)")
    for key in download_priority.get("phase3_analysis", []):
        if key in models:
            spec = models[key]
            download_model(spec["hf_id"])
    
    # Phase 4: Optional
    print("\n📥 Phase 4: Optional Models")
    for key in download_priority.get("phase4_optional", []):
        if key in models:
            spec = models[key]
            download_model(spec["hf_id"])
    
    print("\n✅ Download complete!")
    print("Models cached in ~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()