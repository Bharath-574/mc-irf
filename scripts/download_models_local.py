#!/usr/bin/env python3
"""
MC-IRF Local Model Download Script (RTX 3050 6GB)
Downloads GGUF models for local validation.
"""

import subprocess
import sys
import yaml
from pathlib import Path


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
    with open("configs/models_local.yaml") as f:
        config = yaml.safe_load(f)
    
    models = config.get("models", {})
    download_urls = config.get("download_urls", {})
    
    print("=" * 60)
    print("MC-IRF Local Model Download (RTX 3050 6GB)")
    print("=" * 60)
    
    models_dir = Path("/home/bharath/Research/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Downloading GGUF Models (Q4_K_M)")
    
    for key in ["phi-3.5-mini-gguf", "qwen2.5-7b-gguf"]:
        if key in download_urls:
            spec = download_urls[key]
            local_dir = models_dir / spec["file"].replace(".gguf", "")
            download_gguf_model(spec["repo"], spec["file"], str(local_dir))
    
    print("\n✅ Download complete!")
    print(f"Models saved to {models_dir}/")


if __name__ == "__main__":
    main()