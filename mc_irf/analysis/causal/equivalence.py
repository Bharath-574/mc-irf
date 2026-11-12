"""
MC-IRF Causal Equivalence Testing
Theorem 2: β=1 iff causal intervention ≡ mutual consistency

Tests whether patching the generator's causal layers (from E4)
achieves the same defense as mutual consistency with a discriminator.
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from mc_irf.models.registry import get_registry
from mc_irf.data.gsm_symbolic import GSMProblem, load_problems, CORE_TEMPLATES
from mc_irf.models.wrappers.hf import HFWrapper
from mc_irf.analysis.causal.intervene import CausalTracer


@dataclass
class EquivalenceResult:
    """Result from causal equivalence test for one NoOp type."""
    nop_type: str
    generator: str
    discriminator: str
    
    # Mutual consistency metrics
    mc_gen_acc: float
    mc_mutual_acc: float
    mc_defense_gain: float
    
    # Causal intervention metrics
    patch_gen_acc: float
    patch_defense_gain: float
    
    # Equivalence metrics
    accuracy_diff: float          # |patch_acc - mc_acc|
    defense_gain_diff: float      # |patch_gain - mc_gain|
    relative_diff: float          # accuracy_diff / mc_acc
    equivalent: bool              # Within threshold
    p_value: float                # Statistical test
    
    # Details
    intervention_layers: List[int]
    recovery_per_layer: Dict[int, float]
    per_problem: List[Dict]


class CausalEquivalenceTester:
    """
    Tests Theorem 2: Causal intervention on ℒ* ≈ Mutual Consistency.
    
    Procedure:
    1. Run E4 to identify top-k causal layers ℒ* for each NoOp type
    2. During generation, patch activations at ℒ* from clean run
    3. Compare patched generator accuracy vs mutual consistency accuracy
    4. Statistical test for equivalence
    """
    
    def __init__(self, config_path: str = "configs/mcrf_law.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.equivalence_threshold = self.config["causal_equivalence"]["equivalence_threshold"]
        self.alpha = self.config["causal_equivalence"]["alpha"]
        self.multiple_comparison = self.config["causal_equivalence"]["multiple_comparison"]
    
    def load_causal_layers(self, causal_summary_path: str, top_k: int = 5) -> Dict[str, List[int]]:
        """Load top-k causal layers per NoOp type from E4 results."""
        with open(causal_summary_path) as f:
            causal_data = yaml.safe_load(f)
        
        layers_by_type = {}
        for nop_type in ["quantity", "comparison", "filter", "discount"]:
            # Get from per_nop_type if available
            if "per_nop_type" in causal_data and nop_type in causal_data["per_nop_type"]:
                layers = causal_data["per_nop_type"][nop_type]
                layers_by_type[nop_type] = [l["layer"] for l in layers[:top_k]]
            else:
                # Fallback to overall top layers
                top_layers = causal_data.get("top_layers", [])
                layers_by_type[nop_type] = [l["layer"] for l in top_layers[:top_k]]
        
        return layers_by_type
    
    def run_patched_generation(self,
                                generator_wrapper,
                                problem: GSMProblem,
                                clean_activations: Dict[int, torch.Tensor],
                                intervention_layers: List[int],
                                sampling_params: Dict) -> Tuple[str, float]:
        """
        Run generation with activation patching at intervention layers.
        
        Returns: (generation_text, extracted_answer)
        """
        # This requires the HF wrapper's run_with_patching method
        if not hasattr(generator_wrapper, 'run_with_patching'):
            raise ValueError("Generator wrapper must support run_with_patching (HFWrapper)")
        
        # For patching during generation, we need to patch at each generation step
        # This is complex - simplified version: patch the final forward pass
        # A full implementation would patch during each token generation
        
        # For now, we'll use a simpler approach: 
        # Run the generator normally but with a modified prompt that includes
        # the clean reasoning as context (approximation)
        pass
    
    def compute_equivalence(self,
                             mc_metrics: Dict,
                             patch_metrics: Dict,
                             nop_type: str) -> EquivalenceResult:
        """
        Compute equivalence metrics between MC and causal intervention.
        
        mc_metrics: {gen_acc, mutual_acc, defense_gain} from mutual consistency
        patch_metrics: {gen_acc, defense_gain} from causal patching
        """
        mc_gen = mc_metrics["gen_acc"]
        mc_mutual = mc_metrics["mutual_acc"]
        mc_gain = mc_metrics["defense_gain"]
        
        patch_gen = patch_metrics["gen_acc"]
        patch_gain = patch_metrics["defense_gain"]
        
        accuracy_diff = abs(patch_gen - mc_mutual)
        gain_diff = abs(patch_gain - mc_gain)
        relative_diff = accuracy_diff / mc_mutual if mc_mutual > 0 else 1.0
        
        equivalent = (accuracy_diff < self.equivalence_threshold and 
                      gain_diff < self.equivalence_threshold)
        
        # Statistical test would require per-problem data
        p_value = 1.0  # Placeholder
        
        return EquivalenceResult(
            nop_type=nop_type,
            generator=mc_metrics.get("generator", ""),
            discriminator=mc_metrics.get("discriminator", ""),
            mc_gen_acc=mc_gen,
            mc_mutual_acc=mc_mutual,
            mc_defense_gain=mc_gain,
            patch_gen_acc=patch_gen,
            patch_defense_gain=patch_gain,
            accuracy_diff=accuracy_diff,
            defense_gain_diff=gain_diff,
            relative_diff=relative_diff,
            equivalent=equivalent,
            p_value=p_value,
            intervention_layers=patch_metrics.get("intervention_layers", []),
            recovery_per_layer=patch_metrics.get("recovery_per_layer", {}),
            per_problem=patch_metrics.get("per_problem", []),
        )
    
    def test_equivalence_across_types(self,
                                       mc_results: Dict[str, Dict],
                                       patch_results: Dict[str, Dict]) -> Dict:
        """Test equivalence for all NoOp types with multiple comparison correction."""
        results = {}
        p_values = []
        
        for nop_type in ["quantity", "comparison", "filter", "discount"]:
            if nop_type in mc_results and nop_type in patch_results:
                result = self.compute_equivalence(mc_results[nop_type], patch_results[nop_type], nop_type)
                results[nop_type] = result
                p_values.append(result.p_value)
        
        # Multiple comparison correction (Bonferroni-Holm)
        if p_values and self.multiple_comparison == "bonferroni":
            try:
                from statsmodels.stats.multitest import multipletests
                rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method='holm')
                for i, nop_type in enumerate(results):
                    results[nop_type].p_value = pvals_corrected[i]
                    results[nop_type].equivalent = (pvals_corrected[i] > self.alpha and 
                                                    results[nop_type].accuracy_diff < self.equivalence_threshold)
            except ImportError:
                # Fallback: simple Bonferroni
                alpha_corrected = self.alpha / len(p_values)
                for nop_type in results:
                    results[nop_type].equivalent = (results[nop_type].p_value > alpha_corrected and 
                                                    results[nop_type].accuracy_diff < self.equivalence_threshold)
        
        return results


def run_equivalence_test(
    generator_key: str = "phi-3.5-mini-hf",
    discriminator_key: str = "phi-3.5-mini-hf",
    exp_id: str = "E10",
    causal_summary_path: str = "experiments/results/E4/causal/causal_summary.yaml",
    max_problems: int = 50,
    output_dir: str = "experiments/results/E10/equivalence",
):
    """
    Run full causal equivalence test.
    
    This is a complex experiment that requires:
    1. Mutual consistency results from E2 (homogeneous Φ→Φ)
    2. Causal layer identification from E4
    3. Patched generation on NoOp problems
    """
    print(f"Running causal equivalence test for {exp_id}")
    
    # Load causal layers
    tester = CausalEquivalenceTester()
    intervention_layers = tester.load_causal_layers(causal_summary_path, top_k=5)
    
    # Load problems
    problem_path = f"data/processed/{exp_id}_problems.jsonl"
    if Path(problem_path).exists():
        problems = load_problems(problem_path)
    else:
        from mc_irf.experiments.registry import get_experiment
        from mc_irf.data.gsm_symbolic import generate_problem_set
        exp_config = get_experiment(exp_id)
        problems = generate_problem_set(
            templates=CORE_TEMPLATES,
            num_problems=exp_config["problems"],
            seeds=exp_config["seeds"],
            nop_types=exp_config["nop_types"],
        )
    
    # Load MC results from E2
    mc_results = load_mc_results("E2")
    
    # For each NoOp type, run patched generation
    patch_results = {}
    
    for nop_type in ["quantity", "comparison", "filter", "discount"]:
        layers = intervention_layers.get(nop_type, [])
        if not layers:
            print(f"No causal layers for {nop_type}, skipping")
            continue
        
        print(f"\nTesting {nop_type} with layers {layers}")
        
        # This would run patched generation - requires implementing
        # the patching during generation loop
        patch_metrics = run_patched_generation_for_type(
            generator_key, problems, nop_type, layers
        )
        patch_results[nop_type] = patch_metrics
    
    # Test equivalence
    equivalence_results = tester.test_equivalence_across_types(mc_results, patch_results)
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_equivalence_results(equivalence_results, output_dir)
    
    return equivalence_results


def run_patched_generation_for_type(
    generator_key: str,
    problems: List[GSMProblem],
    nop_type: str,
    intervention_layers: List[int],
) -> Dict:
    """
    Run generation with causal patching for a specific NoOp type.
    
    This is a placeholder - full implementation requires:
    1. Load clean/corrupted problem pairs
    2. Get clean activations
    3. Run generation with patching at each step
    4. Evaluate accuracy
    """
    # Simplified for now - would be implemented in the notebook
    return {
        "gen_acc": 0.0,
        "defense_gain": 0.0,
        "intervention_layers": intervention_layers,
        "recovery_per_layer": {},
        "per_problem": [],
    }


def load_mc_results(exp_id: str) -> Dict[str, Dict]:
    """Load mutual consistency metrics from experiment results."""
    metrics_path = f"experiments/results/{exp_id}/mcrf_metrics.yaml"
    if not Path(metrics_path).exists():
        return {}
    
    with open(metrics_path) as f:
        data = yaml.safe_load(f)
    
    # Convert to per-NoOp-type format
    results = {}
    # This would need the per-problem breakdown
    return results


def save_equivalence_results(results: Dict, output_dir: str):
    """Save causal equivalence test results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    data = {}
    for nop_type, result in results.items():
        data[nop_type] = {
            "nop_type": result.nop_type,
            "generator": result.generator,
            "discriminator": result.discriminator,
            "mc_gen_acc": result.mc_gen_acc,
            "mc_mutual_acc": result.mc_mutual_acc,
            "mc_defense_gain": result.mc_defense_gain,
            "patch_gen_acc": result.patch_gen_acc,
            "patch_defense_gain": result.patch_defense_gain,
            "accuracy_diff": result.accuracy_diff,
            "defense_gain_diff": result.defense_gain_diff,
            "relative_diff": result.relative_diff,
            "equivalent": result.equivalent,
            "p_value": result.p_value,
            "intervention_layers": result.intervention_layers,
            "recovery_per_layer": result.recovery_per_layer,
            "per_problem": result.per_problem,
        }
    
    with open(f"{output_dir}/equivalence_results.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    
    # Also save summary
    summary = {
        "all_equivalent": all(r.equivalent for r in results.values()),
        "nop_types_tested": list(results.keys()),
        "mean_accuracy_diff": np.mean([r.accuracy_diff for r in results.values()]),
        "mean_defense_gain_diff": np.mean([r.defense_gain_diff for r in results.values()]),
    }
    with open(f"{output_dir}/equivalence_summary.yaml", "w") as f:
        yaml.dump(summary, f, default_flow_style=False)


if __name__ == "__main__":
    import sys
    gen = sys.argv[1] if len(sys.argv) > 1 else "phi-3.5-mini-hf"
    disc = sys.argv[2] if len(sys.argv) > 2 else "phi-3.5-mini-hf"
    run_equivalence_test(gen, disc)