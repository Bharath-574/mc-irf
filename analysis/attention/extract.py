# MC-IRF Attention Analysis
# Attention weight extraction and NoOp leakage quantification

from typing import Dict, List, Any, Optional
import numpy as np
import torch
from pathlib import Path

from models.wrappers.hf import HFWrapper
from models.registry import get_registry
from data.gsm_symbolic import GSMProblem, load_problems, CORE_TEMPLATES


def extract_attention_for_problems(
    model_key: str,
    problems: List[GSMProblem],
    layer_indices: List[int] = None,
    max_problems: int = 50,
) -> Dict[str, Any]:
    """Extract attention weights for a set of problems."""
    registry = get_registry()
    wrapper = registry.load(model_key, backend="hf")
    
    if not isinstance(wrapper, HFWrapper):
        raise ValueError("Attention extraction requires HFWrapper backend")
    
    results = {}
    
    for i, problem in enumerate(problems[:max_problems]):
        print(f"  Extracting attention for {problem.problem_id} ({i+1}/{min(max_problems, len(problems))})")
        
        try:
            attn_data = wrapper.get_attention_weights(problem.question, layer_indices)
            results[problem.problem_id] = attn_data
        except Exception as e:
            print(f"    Error: {e}")
            results[problem.problem_id] = {"error": str(e)}
    
    return results


def quantify_noop_attention_leakage(
    attention_data: Dict[str, Any],
    problems: List[GSMProblem],
    nop_token_indices: Dict[str, List[int]] = None,
) -> Dict[str, Any]:
    """
    Quantify attention leakage to NoOp tokens.
    
    For each problem with NoOp, measure:
    - Average attention from reasoning tokens to NoOp tokens
    - Average attention from NoOp tokens to reasoning tokens
    - Compare clean vs NoOp problems
    """
    # Group by NoOp type
    by_nop_type = {}
    for prob in problems:
        nt = prob.nop_type or "clean"
        if nt not in by_nop_type:
            by_nop_type[nt] = []
        by_nop_type[nt].append(prob)
    
    # For each NoOp problem, we need to identify NoOp token positions
    # This requires tokenizing and finding the NoOp span
    leakage_results = {}
    
    for nop_type, nop_problems in by_nop_type.items():
        if nop_type == "clean":
            continue
        
        leakages = []
        
        for problem in nop_problems:
            if problem.problem_id not in attention_data:
                continue
            
            attn = attention_data[problem.problem_id]
            tokens = attn.get("tokens", [])
            
            # Find NoOp token indices (heuristic: last N tokens where NoOp text appears)
            # In practice, need to tokenize NoOp text separately and find overlap
            nop_text = problem.nop_text or ""
            if not nop_text:
                continue
            
            # Simple heuristic: NoOp tokens are at the end
            # Better: tokenize question and NoOp separately, find boundary
            clean_q = problem.question.replace(problem.nop_text, "").strip()
            
            # For now, assume last 20% tokens are NoOp region
            seq_len = len(tokens)
            nop_start = int(seq_len * 0.8)
            nop_indices = list(range(nop_start, seq_len))
            reasoning_indices = list(range(0, nop_start))
            
            # Compute leakage per layer
            for layer_key, layer_data in attn.items():
                if not layer_key.startswith("layer_"):
                    continue
                
                attention = layer_data["attention"]  # (heads, seq, seq)
                num_heads = attention.shape[0]
                
                # Attention FROM reasoning TO NoOp
                if len(reasoning_indices) > 0 and len(nop_indices) > 0:
                    reasoning_to_noop = attention[:, reasoning_indices, :][:, :, nop_indices]
                    avg_leakage = reasoning_to_noop.mean()
                    
                    # Attention FROM NoOp TO reasoning
                    noop_to_reasoning = attention[:, nop_indices, :][:, :, reasoning_indices]
                    avg_backflow = noop_to_reasoning.mean()
                    
                    leakages.append({
                        "problem_id": problem.problem_id,
                        "layer": layer_key,
                        "reasoning_to_noop": float(avg_leakage),
                        "noop_to_reasoning": float(avg_backflow),
                        "num_heads": num_heads,
                    })
        
        if leakages:
            # Aggregate
            by_layer = {}
            for l in leakages:
                layer = l["layer"]
                if layer not in by_layer:
                    by_layer[layer] = {"r2n": [], "n2r": []}
                by_layer[layer]["r2n"].append(l["reasoning_to_noop"])
                by_layer[layer]["n2r"].append(l["noop_to_reasoning"])
            
            summary = {}
            for layer, data in by_layer.items():
                summary[layer] = {
                    "mean_reasoning_to_noop": float(np.mean(data["r2n"])),
                    "std_reasoning_to_noop": float(np.std(data["r2n"])),
                    "mean_noop_to_reasoning": float(np.mean(data["n2r"])),
                    "std_noop_to_reasoning": float(np.std(data["n2r"])),
                }
            
            leakage_results[nop_type] = {
                "per_problem": leakages,
                "summary": summary,
            }
    
    return leakage_results


def compare_clean_vs_noop_attention(
    clean_attention: Dict[str, Any],
    noop_attention: Dict[str, Any],
    layer_indices: List[int] = None,
) -> Dict[str, Any]:
    """Compare attention patterns between clean and NoOp problems."""
    # This would compare layer-wise attention statistics
    # Placeholder for now
    return {}


def save_attention_results(results: Dict, output_path: str):
    """Save attention analysis results."""
    import json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
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
        return obj
    
    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)


def run_attention_analysis(
    model_key: str = "phi-3.5-mini",
    exp_id: str = "E3",
    max_problems: int = 50,
    output_dir: str = "experiments/results/E3/attention",
):
    """Run full attention analysis for an experiment."""
    print(f"Running attention analysis for {exp_id} with {model_key}")
    
    # Load problems
    problem_path = f"data/processed/{exp_id}_problems.jsonl"
    if Path(problem_path).exists():
        problems = load_problems(problem_path)
    else:
        from data.gsm_symbolic import generate_problem_set
        from experiments.registry import get_experiment
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
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_attention_results(attention_data, f"{output_dir}/attention_weights.json")
    save_attention_results(leakage, f"{output_dir}/noop_leakage.json")
    
    print(f"Results saved to {output_dir}")
    
    return {
        "attention_weights": attention_data,
        "leakage": leakage,
    }


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "phi-3.5-mini"
    exp = sys.argv[2] if len(sys.argv) > 2 else "E3"
    run_attention_analysis(model, exp)