"""
MC-IRF Attention Analysis
Attention weight extraction and NoOp leakage quantification.
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from mc_irf.models.registry import get_registry
from mc_irf.data.gsm_symbolic import GSMProblem, load_problems, CORE_TEMPLATES


@dataclass
class AttentionData:
    """Attention weights for a single problem."""
    problem_id: str
    tokens: List[str]
    layers: Dict[str, Dict]  # layer_idx -> {attention, shape, num_heads, seq_len}
    prompt: str
    nop_type: Optional[str] = None
    nop_text: Optional[str] = None


@dataclass
class LeakageResult:
    """NoOp attention leakage for a problem."""
    problem_id: str
    nop_type: str
    layer_leakage: Dict[int, Dict]  # layer -> {r2n_mean, r2n_std, n2r_mean, n2r_std}
    head_leakage: Dict[int, Dict]   # layer -> head -> {r2n, n2r}


def extract_attention_for_problems(
    model_key: str,
    problems: List[GSMProblem],
    layer_indices: List[int] = None,
    max_problems: int = 50,
) -> Dict[str, AttentionData]:
    """Extract attention weights for a set of problems."""
    registry = get_registry("configs/models_h200.yaml")
    wrapper = registry.load(model_key, backend="hf")
    
    if not hasattr(wrapper, 'get_attention_weights'):
        raise ValueError("Attention extraction requires HFWrapper backend")
    
    results = {}
    
    for i, problem in enumerate(problems[:max_problems]):
        print(f"  Extracting attention for {problem.problem_id} ({i+1}/{min(max_problems, len(problems))})")
        
        try:
            attn_data = wrapper.get_attention_weights(problem.question, layer_indices)
            results[problem.problem_id] = AttentionData(
                problem_id=problem.problem_id,
                tokens=attn_data.get("tokens", []),
                layers={k: v for k, v in attn_data.items() if k.startswith("layer_")},
                prompt=problem.question,
                nop_type=problem.nop_type,
                nop_text=problem.nop_text,
            )
        except Exception as e:
            print(f"    Error: {e}")
            results[problem.problem_id] = AttentionData(
                problem_id=problem.problem_id,
                tokens=[],
                layers={},
                prompt=problem.question,
                nop_type=problem.nop_type,
            )
    
    return results


def identify_noop_token_span(attention_data: AttentionData, problem: GSMProblem) -> tuple:
    """Identify NoOp token span in the sequence."""
    tokens = attention_data.tokens
    nop_text = problem.nop_text or ""
    
    if not nop_text:
        return None, None
    
    # Tokenize NoOp text separately to find overlap
    # Heuristic: NoOp tokens are typically at the end before the question
    seq_len = len(tokens)
    nop_start = int(seq_len * 0.75)  # Last 25% as default
    nop_indices = list(range(nop_start, seq_len))
    reasoning_indices = list(range(0, nop_start))
    
    return reasoning_indices, nop_indices


def quantify_noop_attention_leakage(
    attention_data: Dict[str, AttentionData],
    problems: List[GSMProblem],
) -> Dict[str, LeakageResult]:
    """
    Quantify attention leakage to NoOp tokens.
    
    For each problem with NoOp, measure:
    - Average attention FROM reasoning tokens TO NoOp tokens (r2n)
    - Average attention FROM NoOp tokens TO reasoning tokens (n2r)
    - Per-layer and per-head breakdown
    """
    # Group by NoOp type
    by_nop_type = {}
    for prob in problems:
        nt = prob.nop_type or "clean"
        if nt not in by_nop_type:
            by_nop_type[nt] = []
        by_nop_type[nt].append(prob)
    
    leakage_results = {}
    
    for nop_type, nop_problems in by_nop_type.items():
        if nop_type == "clean":
            continue
        
        all_leakages = []
        
        for problem in nop_problems:
            if problem.problem_id not in attention_data:
                continue
            
            attn = attention_data[problem.problem_id]
            if not attn.layers:
                continue
            
            reasoning_idx, nop_idx = identify_noop_token_span(attn, problem)
            if reasoning_idx is None or len(reasoning_idx) == 0 or len(nop_idx) == 0:
                continue
            
            layer_leakage = {}
            head_leakage = {}
            
            for layer_key, layer_data in attn.layers.items():
                if not layer_key.startswith("layer_"):
                    continue
                
                layer_num = int(layer_key.split("_")[1])
                attention = layer_data["attention"]  # (heads, seq, seq)
                num_heads = attention.shape[0]
                
                # Attention FROM reasoning TO NoOp
                if len(reasoning_idx) > 0 and len(nop_idx) > 0:
                    r2n = attention[:, reasoning_idx, :][:, :, nop_idx]
                    r2n_mean = float(r2n.mean())
                    r2n_std = float(r2n.std())
                    
                    # Attention FROM NoOp TO reasoning
                    n2r = attention[:, nop_idx, :][:, :, reasoning_idx]
                    n2r_mean = float(n2r.mean())
                    n2r_std = float(n2r.std())
                    
                    layer_leakage[layer_num] = {
                        "r2n_mean": r2n_mean,
                        "r2n_std": r2n_std,
                        "n2r_mean": n2r_mean,
                        "n2r_std": n2r_std,
                    }
                    
                    # Per-head
                    head_leakage[layer_num] = {}
                    for h in range(num_heads):
                        h_r2n = attention[h, reasoning_idx, :][:, nop_idx].mean()
                        h_n2r = attention[h, nop_idx, :][:, reasoning_idx].mean()
                        head_leakage[layer_num][h] = {
                            "r2n": float(h_r2n),
                            "n2r": float(h_n2r),
                        }
            
            if layer_leakage:
                all_leakages.append(LeakageResult(
                    problem_id=problem.problem_id,
                    nop_type=nop_type,
                    layer_leakage=layer_leakage,
                    head_leakage=head_leakage,
                ))
        
        if all_leakages:
            # Aggregate across problems
            aggregated = aggregate_leakage(all_leakages)
            leakage_results[nop_type] = aggregated
    
    return leakage_results


def aggregate_leakage(leakages: List[LeakageResult]) -> Dict:
    """Aggregate leakage across problems."""
    # Collect per-layer stats
    layer_stats = {}
    head_stats = {}
    
    for leak in leakages:
        for layer, data in leak.layer_leakage.items():
            if layer not in layer_stats:
                layer_stats[layer] = {"r2n": [], "n2r": []}
            layer_stats[layer]["r2n"].append(data["r2n_mean"])
            layer_stats[layer]["n2r"].append(data["n2r_mean"])
        
        for layer, heads in leak.head_leakage.items():
            if layer not in head_stats:
                head_stats[layer] = {}
            for head, hdata in heads.items():
                if head not in head_stats[layer]:
                    head_stats[layer][head] = {"r2n": [], "n2r": []}
                head_stats[layer][head]["r2n"].append(hdata["r2n"])
                head_stats[layer][head]["n2r"].append(hdata["n2r"])
    
    # Compute summary
    summary = {"layers": {}, "heads": {}}
    
    for layer, data in layer_stats.items():
        summary["layers"][layer] = {
            "mean_r2n": float(np.mean(data["r2n"])),
            "std_r2n": float(np.std(data["r2n"])),
            "mean_n2r": float(np.mean(data["n2r"])),
            "std_n2r": float(np.std(data["n2r"])),
            "n_problems": len(data["r2n"]),
        }
    
    for layer, heads in head_stats.items():
        summary["heads"][layer] = {}
        for head, hdata in heads.items():
            summary["heads"][layer][head] = {
                "mean_r2n": float(np.mean(hdata["r2n"])),
                "std_r2n": float(np.std(hdata["r2n"])),
                "mean_n2r": float(np.mean(hdata["n2r"])),
                "std_n2r": float(np.std(hdata["n2r"])),
            }
    
    return {
        "per_problem": [l.__dict__ for l in leakages],
        "summary": summary,
    }


def predict_rho_from_leakage(leakage_results: Dict, config: Dict) -> Dict[str, float]:
    """
    Predict error correlation ρ from attention leakage.
    
    Hypothesis: Higher r2n leakage → higher ρ (more correlated errors)
    """
    predictions = {}
    
    for nop_type, data in leakage_results.items():
        if "summary" not in data or "layers" not in data["summary"]:
            continue
        
        # Use mean r2n across all layers as predictor
        r2n_values = [layer_data["mean_r2n"] for layer_data in data["summary"]["layers"].values()]
        mean_r2n = np.mean(r2n_values) if r2n_values else 0.0
        
        # Simple linear prediction (would be calibrated in E0)
        # ρ ≈ α * mean_r2n + β
        # For now, just report the leakage statistic
        predictions[nop_type] = {
            "mean_r2n_leakage": mean_r2n,
            "mean_n2r_leakage": np.mean([layer_data["mean_n2r"] for layer_data in data["summary"]["layers"].values()]),
            "predicted_rho": float(np.clip(mean_r2n * 2, 0, 1)),  # Rough heuristic
        }
    
    return predictions


def save_attention_results(results: Dict, output_path: str):
    """Save attention analysis results."""
    import json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif hasattr(obj, '__dict__'):
            return convert(obj.__dict__)
        return obj
    
    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)


def run_attention_analysis(
    model_key: str = "phi-3.5-mini-hf",
    exp_id: str = "E3",
    max_problems: int = 50,
    output_dir: str = "experiments/results_h200/E3/attention",
):
    """Run full attention analysis for an experiment."""
    print(f"Running attention analysis for {exp_id} with {model_key}")
    
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
    
    # Extract attention
    print("Extracting attention weights...")
    attention_data = extract_attention_for_problems(
        model_key, problems, max_problems=max_problems
    )
    
    # Quantify leakage
    print("Quantifying NoOp attention leakage...")
    leakage = quantify_noop_attention_leakage(attention_data, problems)
    
    # Predict ρ
    print("Predicting ρ from leakage...")
    with open("configs/mcrf_law.yaml") as f:
        mcrf_config = yaml.safe_load(f)
    rho_predictions = predict_rho_from_leakage(leakage, mcrf_config)
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_attention_results(attention_data, f"{output_dir}/attention_weights.json")
    save_attention_results(leakage, f"{output_dir}/noop_leakage.json")
    save_attention_results(rho_predictions, f"{output_dir}/rho_predictions.json")
    
    print(f"Results saved to {output_dir}")
    
    return {
        "attention_weights": attention_data,
        "leakage": leakage,
        "rho_predictions": rho_predictions,
    }


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "phi-3.5-mini-hf"
    exp = sys.argv[2] if len(sys.argv) > 2 else "E3"
    run_attention_analysis(model, exp)