#!/usr/bin/env python3
# Discover best current models for MC-IRF on H200
# Run: python scripts/discover_models.py

from huggingface_hub import HfApi
import json

api = HfApi()

# Target model families for math reasoning
target_families = {
    "phi": "microsoft",
    "qwen": "Qwen", 
    "llama": "meta-llama",
    "nemotron": "nvidia",
    "mistral": "mistralai",
    "gemma": "google",
    "deepseek": "deepseek-ai",
    "yi": "01-ai",
    "olmo": "allenai",
    "starcoder": "bigcode",
}

# Size buckets we care about
size_buckets = {
    "3-4B": (3e9, 5e9),
    "7-8B": (7e9, 9e9),
    "13-15B": (12e9, 16e9),
    "30-35B": (28e9, 38e9),
    "70-72B": (65e9, 75e9),
}

def estimate_params(model_id: str) -> float:
    """Extract param count from model ID."""
    import re
    # Look for patterns like 3.8B, 7B, 14B, 32B, 70B, 72B
    matches = re.findall(r'(\d+\.?\d*)[Bb]', model_id)
    if matches:
        val = float(matches[-1])
        if val < 100:  # Assume B if < 100
            return val * 1e9
    return 0

def is_instruct(model_id: str) -> bool:
    """Check if model is instruction-tuned."""
    lower = model_id.lower()
    return any(x in lower for x in ['instruct', 'chat', 'it-', '-it', 'sft', 'rlhf', 'dpo'])

def main():
    print("=" * 80)
    print("DISCOVERING BEST MODELS FOR MC-IRF (2026)")
    print("=" * 80)
    
    all_candidates = []
    
    for family, org in target_families.items():
        try:
            print(f"\n🔍 Searching {org}...")
            models = api.list_models(
                author=org, 
                filter='text-generation-inference', 
                sort='lastModified', 
                direction=-1, 
                limit=30
            )
            
            for m in models:
                if not is_instruct(m.modelId):
                    continue
                
                params = estimate_params(m.modelId)
                if params == 0:
                    continue
                
                # Determine size bucket
                bucket = "other"
                for name, (lo, hi) in size_buckets.items():
                    if lo <= params <= hi:
                        bucket = name
                        break
                
                # Get last modified
                last_mod = getattr(m, 'lastModified', None)
                if hasattr(last_mod, 'isoformat'):
                    last_mod = last_mod.isoformat()
                elif hasattr(last_mod, 'strftime'):
                    last_mod = last_mod.strftime('%Y-%m-%d')
                
                candidate = {
                    "family": family,
                    "model_id": m.modelId,
                    "params_b": round(params / 1e9, 1),
                    "bucket": bucket,
                    "last_modified": str(last_mod) if last_mod else "unknown",
                    "tags": getattr(m, 'tags', []),
                    "downloads": getattr(m, 'downloads', 0),
                    "likes": getattr(m, 'likes', 0),
                }
                all_candidates.append(candidate)
                print(f"  {m.modelId} ({params/1e9:.1f}B) [{bucket}] - {last_mod}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Sort by bucket, then by downloads (popularity proxy)
    print("\n" + "=" * 80)
    print("SUMMARY BY SIZE BUCKET")
    print("=" * 80)
    
    for bucket_name in ["3-4B", "7-8B", "13-15B", "30-35B", "70-72B"]:
        bucket_models = [c for c in all_candidates if c["bucket"] == bucket_name]
        bucket_models.sort(key=lambda x: x["downloads"], reverse=True)
        
        print(f"\n📦 {bucket_name}:")
        for c in bucket_models[:5]:
            print(f"  {c['model_id']} ({c['params_b']}B) - ⬇{c['downloads']} ❤{c['likes']} - {c['last_modified']}")
    
    # Save full results
    with open("configs/model_discovery.json", "w") as f:
        json.dump(all_candidates, f, indent=2)
    print(f"\n💾 Saved {len(all_candidates)} candidates to configs/model_discovery.json")

if __name__ == "__main__":
    main()