"""
MC-IRF Causal Tracing
Activation patching / causal mediation analysis for NoOp susceptibility.
"""

import numpy as np
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from mc_irf.models.registry import get_registry
from mc_irf.data.gsm_symbolic import GSMProblem, load_problems, CORE_TEMPLATES
from mc_irf.models.wrappers.hf import HFWrapper


@dataclass
class CausalTraceResult:
    """Causal tracing result for a problem pair."""
    clean_problem_id: str
    corrupted_problem_id: str
    nop_type: str
    baseline_correct_prob: float
    target_token: str
    target_token_id: int
    layers: Dict[int, Dict]  # layer -> {patched_prob, recovery, improvement}


class CausalTracer:
    """
    Causal tracing via activation patching.
    
    Based on: "Locating and Editing Factual Knowledge in GPT" (Meng et al.)
    and "Causal Tracing for Language Models" (Wang et al.)
    
    For NoOp: We trace how NoOp tokens affect the final answer prediction
    by patching activations from clean runs into corrupted (NoOp) runs.
    """
    
    def __init__(self, model_key: str):
        self.model_key = model_key
        self.registry = get_registry("configs/models_h200.yaml")
        self.wrapper = self.registry.load(model_key, backend="hf")
        
        if not isinstance(self.wrapper, HFWrapper):
            raise ValueError("Causal tracing requires HFWrapper backend")
        
        self.model = self.wrapper.model
        self.tokenizer = self.wrapper.tokenizer
        self.device = self.wrapper.device
        
        # Hook storage
        self.hooks = []
        self.cached_activations = {}
    
    def _register_hooks(self, layer_indices: List[int]):
        """Register forward hooks to cache activations."""
        self.cached_activations = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output is tuple (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                self.cached_activations[layer_idx] = hidden.detach().clone()
            return hook
        
        for layer_idx in layer_indices:
            if layer_idx < len(self.model.model.layers):
                layer = self.model.model.layers[layer_idx]
                hook = layer.register_forward_hook(make_hook(layer_idx))
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_clean_activations(self, clean_problem: GSMProblem, layer_indices: List[int]) -> Dict[int, torch.Tensor]:
        """Run clean problem and cache activations at each layer."""
        self._register_hooks(layer_indices)
        
        with torch.no_grad():
            inputs = self.tokenizer(clean_problem.question, return_tensors="pt").to(self.device)
            _ = self.model(**inputs)
        
        activations = {k: v.clone() for k, v in self.cached_activations.items()}
        self._remove_hooks()
        
        return activations
    
    def run_with_patching(
        self, 
        corrupted_problem: GSMProblem, 
        clean_activations: Dict[int, torch.Tensor],
        patch_layer: int,
        patch_position: str = "noop_span",  # or "last_token", "all"
    ) -> torch.Tensor:
        """
        Run corrupted problem with activations patched from clean run at one layer.
        
        Returns: logits for final token prediction
        """
        def patching_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Patch the activations
            if patch_layer in clean_activations:
                clean_act = clean_activations[patch_layer]
                
                if patch_position == "last_token":
                    # Patch only last token position
                    hidden[:, -1, :] = clean_act[:, -1, :]
                elif patch_position == "all":
                    # Patch all positions (trim to match)
                    min_len = min(hidden.shape[1], clean_act.shape[1])
                    hidden[:, :min_len, :] = clean_act[:, :min_len, :]
                elif patch_position == "noop_span":
                    # Patch last 20% (NoOp span heuristic)
                    seq_len = hidden.shape[1]
                    start = int(seq_len * 0.8)
                    hidden[:, start:, :] = clean_act[:, start:, :]
            
            return (hidden,) if isinstance(output, tuple) else hidden
        
        # Register patching hook only at target layer
        target_layer = self.model.model.layers[patch_layer]
        hook = target_layer.register_forward_hook(patching_hook)
        
        with torch.no_grad():
            inputs = self.tokenizer(corrupted_problem.question, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        
        hook.remove()
        
        # Return logits for last token
        logits = outputs.logits[:, -1, :]
        return logits
    
    def trace_noop_effect(
        self,
        clean_problem: GSMProblem,
        corrupted_problem: GSMProblem,
        layer_indices: List[int] = None,
        target_answer_token: str = None,
    ) -> CausalTraceResult:
        """
        Trace the causal effect of NoOp by patching each layer.
        
        Measures: How much does patching layer L from clean -> corrupted
        recover the correct answer probability?
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.model.model.layers)))
        
        # Get clean activations
        print(f"  Getting clean activations for {clean_problem.problem_id}...")
        clean_acts = self.get_clean_activations(clean_problem, layer_indices)
        
        # Baseline: corrupted run without patching
        with torch.no_grad():
            inputs = self.tokenizer(corrupted_problem.question, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            corrupted_logits = outputs.logits[:, -1, :]
        
        # Get target token ID
        if target_answer_token is None:
            # Use the correct answer
            target_answer_token = str(int(corrupted_problem.answer))
        
        target_id = self.tokenizer.encode(target_answer_token, add_special_tokens=False)
        if len(target_id) > 1:
            target_id = target_id[0]  # Take first token
        else:
            target_id = target_id[0]
        
        # Baseline probability of correct answer
        corrupted_probs = torch.softmax(corrupted_logits, dim=-1)
        baseline_prob = corrupted_probs[0, target_id].item()
        
        # Patch each layer
        layers_result = {}
        
        for layer_idx in layer_indices:
            if layer_idx not in clean_acts:
                continue
            
            print(f"  Patching layer {layer_idx}...")
            patched_logits = self.run_with_patching(
                corrupted_problem, clean_acts, layer_idx, patch_position="noop_span"
            )
            
            patched_probs = torch.softmax(patched_logits, dim=-1)
            patched_prob = patched_probs[0, target_id].item()
            
            # Recovery = (patched - baseline) / (1 - baseline)  # Normalized
            recovery = (patched_prob - baseline_prob) / (1 - baseline_prob + 1e-8)
            
            layers_result[layer_idx] = {
                "patched_correct_prob": patched_prob,
                "recovery": recovery,
                "improvement": patched_prob - baseline_prob,
            }
        
        return CausalTraceResult(
            clean_problem_id=clean_problem.problem_id,
            corrupted_problem_id=corrupted_problem.problem_id,
            nop_type=corrupted_problem.nop_type,
            baseline_correct_prob=baseline_prob,
            target_token=target_answer_token,
            target_token_id=target_id,
            layers=layers_result,
        )
    
    def trace_head_contributions(
        self,
        clean_problem: GSMProblem,
        corrupted_problem: GSMProblem,
        layer_idx: int,
        target_answer_token: str = None,
    ) -> Dict[str, Any]:
        """
        Trace individual attention head contributions at a specific layer.
        Patches per-head activations.
        """
        # This requires accessing attention output per head
        # More granular - patch individual heads
        raise NotImplementedError("Head-level tracing requires attention output access")


def run_causal_tracing(
    model_key: str = "phi-3.5-mini-hf",
    exp_id: str = "E4",
    max_pairs: int = 20,
    layer_indices: List[int] = None,
    output_dir: str = "experiments/results_h200/E4/causal",
):
    """Run causal tracing for clean/NoOp problem pairs."""
    print(f"Running causal tracing for {exp_id} with {model_key}")
    
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
    
    # Group clean/NoOp pairs
    clean_problems = [p for p in problems if p.nop_type is None]
    nop_problems = [p for p in problems if p.nop_type is not None]
    
    # Match by template_id and seed
    pairs = []
    for clean in clean_problems:
        for nop in nop_problems:
            # Same template and seed
            clean_key = clean.problem_id.replace("_clean", "")
            nop_key = nop.problem_id.replace(f"_{nop.nop_type}", "")
            if clean_key == nop_key:
                pairs.append((clean, nop))
    
    print(f"Found {len(pairs)} clean/NoOp pairs")
    
    # Initialize tracer
    tracer = CausalTracer(model_key)
    
    if layer_indices is None:
        layer_indices = list(range(len(tracer.model.model.layers)))
    
    # Run tracing on subset
    all_results = {}
    
    for i, (clean, nop) in enumerate(pairs[:max_pairs]):
        print(f"\nPair {i+1}/{min(max_pairs, len(pairs))}: {clean.problem_id} vs {nop.problem_id}")
        
        try:
            result = tracer.trace_noop_effect(
                clean, nop, layer_indices=layer_indices
            )
            
            pair_key = f"{clean.template_id}_seed{clean.problem_id.split('seed')[1].split('_')[0]}_{nop.nop_type}"
            all_results[pair_key] = result
            
        except Exception as e:
            print(f"  Error: {e}")
            all_results[pair_key] = {"error": str(e)}
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    import json
    
    def convert(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        elif isinstance(obj, (np.floating, torch.floating)):
            return float(obj)
        elif isinstance(obj, (np.integer, torch.integer)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return convert(obj.__dict__)
        return obj
    
    with open(f"{output_dir}/causal_traces.json", "w") as f:
        json.dump(convert(all_results), f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Aggregate summary
    summarize_causal_results(all_results, output_dir)
    
    return all_results


def summarize_causal_results(results: Dict, output_dir: str):
    """Summarize causal tracing results across pairs."""
    import json
    
    # Aggregate recovery by layer
    layer_recovery = {}
    
    for pair_key, result in results.items():
        if "error" in result or "layers" not in result:
            continue
        
        for layer_idx, layer_data in result["layers"].items():
            if layer_idx not in layer_recovery:
                layer_recovery[layer_idx] = []
            layer_recovery[layer_idx].append(layer_data["recovery"])
    
    # Compute mean recovery per layer
    summary = {}
    for layer_idx, recoveries in layer_recovery.items():
        summary[layer_idx] = {
            "mean_recovery": float(np.mean(recoveries)),
            "std_recovery": float(np.std(recoveries)),
            "num_samples": len(recoveries),
        }
    
    # Find most causal layers
    sorted_layers = sorted(summary.items(), key=lambda x: x[1]["mean_recovery"], reverse=True)
    
    print("\n=== Causal Layer Ranking ===")
    for layer_idx, data in sorted_layers[:10]:
        print(f"  Layer {layer_idx}: recovery={data['mean_recovery']:.3f} ± {data['std_recovery']:.3f}")
    
    # Per NoOp type breakdown
    nop_type_recovery = {}
    for pair_key, result in results.items():
        if "error" in result or "layers" not in result:
            continue
        nop_type = result.get("nop_type", "unknown")
        if nop_type not in nop_type_recovery:
            nop_type_recovery[nop_type] = {}
        for layer_idx, layer_data in result["layers"].items():
            if layer_idx not in nop_type_recovery[nop_type]:
                nop_type_recovery[nop_type][layer_idx] = []
            nop_type_recovery[nop_type][layer_idx].append(layer_data["recovery"])
    
    for nop_type, layers in nop_type_recovery.items():
        print(f"\n--- {nop_type} ---")
        for layer_idx, recoveries in sorted(layers.items(), key=lambda x: np.mean(x[1]), reverse=True)[:5]:
            print(f"  Layer {layer_idx}: recovery={np.mean(recoveries):.3f} ± {np.std(recoveries):.3f}")
    
    with open(f"{output_dir}/causal_summary.yaml", "w") as f:
        yaml.dump({
            "layer_recovery": summary,
            "top_layers": [{"layer": l, **d} for l, d in sorted_layers[:20]],
            "per_nop_type": {nt: {l: {"mean": float(np.mean(v)), "std": float(np.std(v))} 
                                for l, v in layers.items()} 
                           for nt, layers in nop_type_recovery.items()},
        }, f, indent=2)


def load_causal_results(input_path: str) -> Dict:
    """Load causal tracing results."""
    import json
    with open(input_path) as f:
        return json.load(f)


def get_top_causal_layers(causal_summary_path: str, top_k: int = 5) -> List[int]:
    """Get top-k causal layers from summary."""
    with open(causal_summary_path) as f:
        data = yaml.safe_load(f)
    
    top_layers = data.get("top_layers", [])
    return [l["layer"] for l in top_layers[:top_k]]


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "phi-3.5-mini-hf"
    exp = sys.argv[2] if len(sys.argv) > 2 else "E4"
    run_causal_tracing(model, exp)