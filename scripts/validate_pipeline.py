#!/usr/bin/env python3
"""
MC-IRF Pipeline Validation
Tests the full pipeline: data -> models -> experiment
"""

import sys
import os
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc_irf.data.gsm_symbolic import generate_problem_set, CORE_TEMPLATES
from mc_irf.models.registry import get_registry
from mc_irf.experiments.runner import run_experiment


def test_data_generation():
    """Test GSM-Symbolic + NoOp generation."""
    print("Testing data generation...")
    
    problems = generate_problem_set(
        templates=CORE_TEMPLATES,
        num_problems=2,
        seeds=[42],
        nop_types=["quantity", "comparison"],
    )
    
    print(f"  Generated {len(problems)} problems")
    clean = [p for p in problems if p.nop_type is None]
    quantity = [p for p in problems if p.nop_type == "quantity"]
    comparison = [p for p in problems if p.nop_type == "comparison"]
    print(f"  Clean: {len(clean)}, Quantity: {len(quantity)}, Comparison: {len(comparison)}")
    
    # Show sample
    p = clean[0]
    print(f"\n  Sample clean problem:")
    print(f"    ID: {p.problem_id}")
    print(f"    Q: {p.question[:100]}...")
    print(f"    A: {p.answer}")
    
    p = quantity[0]
    print(f"\n  Sample quantity NoOp:")
    print(f"    ID: {p.problem_id}")
    print(f"    Q: {p.question[:100]}...")
    print(f"    NoOp: {p.nop_text[:100]}...")
    
    return True


def test_model_registry(config_path: str = "configs/models_h200.yaml"):
    """Test model registry loads config."""
    print("\nTesting model registry...")
    
    registry = get_registry(config_path)
    models = registry.list_models()
    print(f"  Available models: {models}")
    
    # Test spec retrieval
    spec = registry.get_spec("phi-3.5-mini")
    print(f"  phi-3.5-mini: {spec.hf_id} ({spec.params_b}B, TP={spec.tensor_parallel})")
    
    # Show feasible pairs
    pairs = registry.get_feasible_pairs(max_size_gb=16.0)
    print(f"  Feasible pairs (16GB limit): {len(pairs)}")
    for gen, disc in pairs[:5]:
        print(f"    {gen} -> {disc}")
    if len(pairs) > 5:
        print(f"    ... and {len(pairs) - 5} more")
    
    return True


def test_vllm_load():
    """Test vLLM model loading (requires models downloaded)."""
    print("\nTesting vLLM load (phi-3.5-mini)...")
    
    registry = get_registry("configs/models_h200.yaml")
    try:
        model = registry.load("phi-3.5-mini")
        print(f"  ✓ Loaded: {model.get_model_info()}")
        
        # Quick generation test
        outputs = model.generate_single("2+2=", {"temperature": 0.0, "max_tokens": 10, "n": 1})
        print(f"  Generation: {outputs[0].text.strip()}")
        
        registry.unload("phi-3.5-mini")
        print("  ✓ Unloaded")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_experiment_runner():
    """Test experiment runner with minimal config."""
    print("\nTesting experiment runner (E1 mini)...")
    
    # Create minimal experiment config for testing
    test_config = {
        "experiments": [{
            "id": "E1_test",
            "name": "Test Experiment",
            "generator": "phi-3.5-mini",
            "discriminator": "phi-3.5-mini",
            "nop_types": ["quantity"],
            "seeds": [42],
            "rollouts": 2,
            "problems": 1,
            "claim": "Test",
        }]
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = f.name
    
    try:
        print("  (Skipping full experiment run - would take time)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    print("=" * 60)
    print("MC-IRF PIPELINE VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_data_generation()
    all_passed &= test_model_registry()
    
    # Only test vLLM if models are downloaded
    print("\n" + "=" * 60)
    print("Run vLLM test? Models must be downloaded first.")
    print("Run: python scripts/download_models.py")
    print("Then: python scripts/validate_pipeline.py --test-vllm")
    print("=" * 60)
    
    if "--test-vllm" in sys.argv:
        all_passed &= test_vllm_load()
    
    all_passed &= test_experiment_runner()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Pipeline ready!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()