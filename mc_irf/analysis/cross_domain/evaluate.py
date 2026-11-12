"""
MC-IRF Cross-Domain Analysis
E11: Cross-Domain NoOp Generalization
E12: SCoRe Distillation
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from mc_irf.models.registry import get_registry
from mc_irf.data.gsm_symbolic import GSMProblem
from mc_irf.theory.mcrf_law import MCRFLaw, ExperimentMetrics


@dataclass
class CrossDomainResult:
    """Cross-domain MCRF law test result."""
    domain: str
    nop_type: str
    gen_acc: float
    disc_acc: float
    mutual_acc: float
    sc_acc: float
    rho: float
    beta: float
    beta_ci: Tuple[float, float]
    math_beta: float
    transfer_score: float  # Similarity to math β


def load_cross_domain_problems(domain: str, nop_types: List[str], 
                               problems_per_type: int = 30) -> List[GSMProblem]:
    """Load or generate cross-domain NoOp problems."""
    # This would load from generated data files
    # For now, return empty - actual generation in notebook
    return []


def compute_cross_domain_metrics(
    domain: str,
    generator_key: str,
    discriminator_key: str,
    problems: List[GSMProblem],
) -> Dict[str, CrossDomainResult]:
    """Compute MCRF metrics for a cross-domain experiment."""
    # Similar to main metrics but for cross-domain
    results = {}
    
    for nop_type in ["quantity", "comparison", "filter", "discount"]:
        nt_problems = [p for p in problems if p.nop_type == nop_type]
        if not nt_problems:
            continue
        
        # Run generator and discriminator
        # This would be done in the notebook
        
        results[nop_type] = CrossDomainResult(
            domain=domain,
            nop_type=nop_type,
            gen_acc=0.0,
            disc_acc=0.0,
            mutual_acc=0.0,
            sc_acc=0.0,
            rho=0.0,
            beta=0.0,
            beta_ci=(0.0, 1.0),
            math_beta=0.0,
            transfer_score=0.0,
        )
    
    return results


def analyze_beta_transfer(math_betas: Dict[str, float], 
                          domain_betas: Dict[str, Dict[str, float]]) -> Dict:
    """
    Analyze β transfer across domains.
    
    Theorem 3: β_domain1(τ) ≈ β_domain2(τ) ⟺ 
    TopK_Attention_Leakage_Heads(domain1) ≈ TopK_Attention_Leakage_Heads(domain2)
    """
    transfer_analysis = {}
    
    for domain, betas in domain_betas.items():
        transfer_analysis[domain] = {}
        for nop_type, beta in betas.items():
            math_beta = math_betas.get(nop_type, 0.5)
            transfer_score = 1.0 - abs(beta - math_beta) / max(beta, math_beta, 0.01)
            transfer_analysis[domain][nop_type] = {
                "math_beta": math_beta,
                "domain_beta": beta,
                "transfer_score": transfer_score,
                "transfers": transfer_score > 0.7,
            }
    
    return transfer_analysis


def attention_leakage_similarity(leakage1: Dict, leakage2: Dict, 
                                  k: int = 10) -> float:
    """
    Compute similarity of attention leakage topologies.
    
    Returns overlap of top-k heads with highest leakage.
    """
    def get_top_heads(leakage: Dict, k: int) -> set:
        heads = []
        for layer, data in leakage.get("summary", {}).get("heads", {}).items():
            for head, hdata in data.items():
                score = hdata.get("mean_r2n", 0) + hdata.get("mean_n2r", 0)
                heads.append((f"L{layer}H{head}", score))
        heads.sort(key=lambda x: x[1], reverse=True)
        return set(h[0] for h in heads[:k])
    
    top1 = get_top_heads(leakage1, k)
    top2 = get_top_heads(leakage2, k)
    
    if not top1 or not top2:
        return 0.0
    
    return len(top1 & top2) / len(top1 | top2)


def save_cross_domain_results(results: Dict, output_dir: str):
    """Save cross-domain analysis results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    data = {}
    for domain, domain_results in results.items():
        data[domain] = {}
        for nop_type, result in domain_results.items():
            data[domain][nop_type] = {
                "domain": result.domain,
                "nop_type": result.nop_type,
                "gen_acc": result.gen_acc,
                "disc_acc": result.disc_acc,
                "mutual_acc": result.mutual_acc,
                "sc_acc": result.sc_acc,
                "rho": result.rho,
                "beta": result.beta,
                "beta_ci": list(result.beta_ci),
                "math_beta": result.math_beta,
                "transfer_score": result.transfer_score,
            }
    
    with open(f"{output_dir}/cross_domain_results.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)


@dataclass
class SCoReConfig:
    """Configuration for SCoRe (Self-Consistency with Relevance Scoring)."""
    generator_key: str
    scorer_type: str  # "linear_probe", "mlp", "attention"
    layer_indices: List[int]
    hidden_dim: int
    training_samples: int
    validation_split: float = 0.2


class SCoReScorer:
    """
    SCoRe (Self-Consistency with Relevance Scoring)
    
    Distills mutual consistency into a single model + lightweight relevance scorer.
    
    Training:
    1. Generate reasoning traces with generator
    2. Label steps as relevant/irrelevant based on NoOp injection
    3. Train linear probe on hidden states
    4. During inference, weight rollouts by relevance score
    """
    
    def __init__(self, config: SCoReConfig):
        self.config = config
        self.scorer = None
        self.registry = get_registry("configs/models_h200.yaml")
    
    def prepare_training_data(self, problems: List[GSMProblem]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for relevance classifier.
        
        Positive: Clean reasoning steps
        Negative: NoOp-injected steps
        """
        # This would extract hidden states from generator
        # and label them based on whether they come from clean or NoOp problems
        pass
    
    def train(self, problems: List[GSMProblem]):
        """Train the relevance scorer."""
        # Load generator with HF backend for hidden states
        generator = self.registry.load(self.config.generator_key, backend="hf")
        
        # Extract hidden states for clean and NoOp problems
        clean_problems = [p for p in problems if p.nop_type is None]
        nop_problems = [p for p in problems if p.nop_type is not None]
        
        # Get hidden states at specified layers
        # Train linear probe
        pass
    
    def score_rollout(self, generation: str, hidden_states: List[np.ndarray]) -> float:
        """Score a single rollout for relevance."""
        if self.scorer is None:
            return 1.0  # Default to uniform
        
        # Average relevance across steps
        scores = []
        for hidden in hidden_states:
            # hidden: (seq_len, hidden_dim)
            # Score each position
            step_scores = self.scorer.predict_proba(hidden)[:, 1]  # P(relevant)
            scores.append(step_scores.mean())
        
        return np.mean(scores) if scores else 1.0
    
    def score_generations(self, generations: List[Dict]) -> List[float]:
        """Score multiple generations."""
        return [self.score_rollout(g["generation"], g.get("hidden_states", [])) 
                for g in generations]


def run_scorer_distillation(
    generator_key: str = "phi-3.5-mini",
    exp_id: str = "E12",
    problems: List[GSMProblem] = None,
    output_dir: str = "experiments/results_h200/E12/scorer",
):
    """Run SCoRe distillation experiment."""
    print(f"Running SCoRe distillation for {exp_id}")
    
    # This would be implemented in the notebook
    # Key steps:
    # 1. Generate training data (clean + NoOp problems)
    # 2. Extract hidden states from generator
    # 3. Train linear probe on layer hidden states
    # 4. Evaluate on test set: compare SCoRe vs MC vs SC
    pass


def evaluate_scorer(
    scorer: SCoReScorer,
    test_problems: List[GSMProblem],
    generator_key: str,
    discriminator_key: str = None,
) -> Dict:
    """
    Evaluate SCoRe against baselines.
    
    Compares:
    - Self-Consistency (SC)
    - Mutual Consistency (MC) - if discriminator available
    - SCoRe (single model + scorer)
    """
    results = {
        "sc_acc": 0.0,
        "mc_acc": 0.0,
        "scorer_acc": 0.0,
        "scorer_auroc": 0.0,
    }
    
    # Implementation in notebook
    return results


if __name__ == "__main__":
    # Test attention leakage similarity
    leakage_math = {
        "summary": {
            "heads": {
                8: {2: {"mean_r2n": 0.1, "mean_n2r": 0.05}},
                12: {5: {"mean_r2n": 0.15, "mean_n2r": 0.08}},
            }
        }
    }
    
    leakage_code = {
        "summary": {
            "heads": {
                8: {2: {"mean_r2n": 0.12, "mean_n2r": 0.06}},
                12: {5: {"mean_r2n": 0.14, "mean_n2r": 0.07}},
            }
        }
    }
    
    sim = attention_leakage_similarity(leakage_math, leakage_code, k=2)
    print(f"Attention leakage similarity: {sim:.3f}")