"""
MC-IRF Experiment Metrics
Complete metrics suite including MCRF law parameters.
"""

import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

from mc_irf.theory.mcrf_law import MCRFLaw, ExperimentMetrics
from mc_irf.theory.information_theory import error_correlation_discrete, bootstrap_error_correlation


@dataclass
class MCRFMetrics:
    """MCRF law metrics for an experiment."""
    exp_id: str
    generator: str
    discriminator: Optional[str]
    nop_type: str
    
    # Core accuracies
    gen_acc: float
    disc_acc: Optional[float]
    mutual_acc: float
    sc_acc: Optional[float] = None
    
    # MCRF parameters
    rho: Optional[float] = None
    rho_ci: Optional[Tuple[float, float]] = None
    beta: Optional[float] = None
    beta_ci: Optional[Tuple[float, float]] = None
    epsilon: Optional[float] = None
    defense_gain: float = 0.0
    
    # Per-problem breakdown
    per_problem: Dict = field(default_factory=dict)
    
    # Bootstrap samples (for meta-analysis)
    bootstrap_rho: List[float] = field(default_factory=list)
    bootstrap_beta: List[float] = field(default_factory=list)


class MetricsComputer:
    """Computes all experiment metrics including MCRF law parameters."""
    
    def __init__(self, config_path: str = "configs/mcrf_law.yaml"):
        self.mcrf_law = MCRFLaw(config_path)
        self.nop_types = ["quantity", "comparison", "filter", "discount"]
    
    def compute_experiment_metrics(self, 
                                    exp_id: str,
                                    config: Dict,
                                    generations: List[Dict],
                                    discriminator_scores: List[Dict],
                                    problems: List) -> MCRFMetrics:
        """Compute complete metrics for an experiment."""
        
        # Group by problem and NoOp type
        nop_type = config.get("nop_types", ["quantity"])[0]  # Primary type
        
        # Generator accuracy (majority vote over rollouts)
        gen_acc = self._compute_generator_accuracy(generations, problems)
        
        # Discriminator consistency rate
        disc_acc = self._compute_discriminator_accuracy(discriminator_scores)
        
        # Mutual consistency accuracy
        mutual_acc = self._compute_mutual_consistency(generations, discriminator_scores, problems)
        
        # Self-consistency (if available)
        sc_acc = self._compute_self_consistency(generations, problems)
        
        # Error correlation ρ
        rho, rho_ci = self._compute_rho(generations, discriminator_scores, problems)
        
        # Beta estimation
        beta, beta_ci = None, None
        if disc_acc is not None and rho is not None:
            beta = self.mcrf_law.estimate_beta(gen_acc, disc_acc, mutual_acc, rho)
            beta_ci = self._bootstrap_beta(generations, discriminator_scores, problems, rho)
        
        # Defense gain
        defense_gain = mutual_acc - gen_acc
        
        # Per-problem breakdown
        per_problem = self._per_problem_breakdown(generations, discriminator_scores, problems)
        
        return MCRFMetrics(
            exp_id=exp_id,
            generator=config.get("generator", ""),
            discriminator=config.get("discriminator"),
            nop_type=nop_type,
            gen_acc=gen_acc,
            disc_acc=disc_acc,
            mutual_acc=mutual_acc,
            sc_acc=sc_acc,
            rho=rho,
            rho_ci=rho_ci,
            beta=beta,
            beta_ci=beta_ci,
            defense_gain=defense_gain,
            per_problem=per_problem,
        )
    
    def _compute_generator_accuracy(self, generations: List[Dict], problems: List) -> float:
        """Compute generator accuracy via majority vote over rollouts."""
        # Group by problem
        by_problem = {}
        for gen in generations:
            pid = gen["problem_id"]
            if pid not in by_problem:
                by_problem[pid] = []
            by_problem[pid].append(gen)
        
        correct = 0
        total = 0
        for pid, gens in by_problem.items():
            # Majority vote
            answers = [g.get("extracted_answer") for g in gens if g.get("extracted_answer") is not None]
            if not answers:
                continue
            majority = max(set(answers), key=answers.count)
            
            # Find true answer
            problem = next(p for p in problems if p.problem_id == pid)
            true_ans = problem.answer
            
            if abs(majority - true_ans) < 1e-3:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_discriminator_accuracy(self, scores: List[Dict]) -> Optional[float]:
        """Compute discriminator consistency rate."""
        if not scores:
            return None
        consistent = sum(1 for s in scores if s.get("is_consistent", False))
        return consistent / len(scores)
    
    def _compute_mutual_consistency(self, generations: List[Dict], scores: List[Dict], problems: List) -> float:
        """Compute mutual consistency: generator correct AND discriminator agrees."""
        by_problem = {}
        for gen in generations:
            pid = gen["problem_id"]
            if pid not in by_problem:
                by_problem[pid] = {"generations": [], "scores": []}
            by_problem[pid]["generations"].append(gen)
        
        for score in scores:
            pid = score["problem_id"]
            if pid in by_problem:
                by_problem[pid]["scores"].append(score)
        
        mutual_correct = 0
        total = 0
        for pid, data in by_problem.items():
            problem = next(p for p in problems if p.problem_id == pid)
            true_ans = problem.answer
            
            for gen, score in zip(data["generations"], data["scores"]):
                gen_ans = gen.get("extracted_answer")
                gen_correct = gen_ans is not None and abs(gen_ans - true_ans) < 1e-3
                disc_consistent = score.get("is_consistent", False)
                
                if gen_correct and disc_consistent:
                    mutual_correct += 1
                total += 1
        
        return mutual_correct / total if total > 0 else 0.0
    
    def _compute_self_consistency(self, generations: List[Dict], problems: List) -> Optional[float]:
        """Compute self-consistency (majority vote accuracy)."""
        # This is the same as generator accuracy with majority vote
        return self._compute_generator_accuracy(generations, problems)
    
    def _compute_rho(self, generations: List[Dict], scores: List[Dict], problems: List) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        """Compute error correlation ρ with bootstrap CI."""
        # Align generations and scores by problem
        gen_errors = []
        disc_errors = []
        
        for gen in generations:
            pid = gen["problem_id"]
            problem = next(p for p in problems if p.problem_id == pid)
            true_ans = problem.answer
            gen_ans = gen.get("extracted_answer")
            gen_err = 1 if (gen_ans is None or abs(gen_ans - true_ans) >= 1e-3) else 0
            gen_errors.append(gen_err)
        
        for score in scores:
            disc_err = 0 if score.get("is_consistent", False) else 1
            disc_errors.append(disc_err)
        
        # Align by index (assuming same order)
        min_len = min(len(gen_errors), len(disc_errors))
        gen_errors = np.array(gen_errors[:min_len])
        disc_errors = np.array(disc_errors[:min_len])
        
        if len(gen_errors) < 10:
            return None, None
        
        # Compute ρ with bootstrap CI
        rho, ci = bootstrap_error_correlation(
            gen_errors, disc_errors,
            n_bootstrap=1000,
            method="discrete_mi",
            confidence=0.95
        )
        
        return rho, ci
    
    def _bootstrap_beta(self, generations: List[Dict], scores: List[Dict], 
                        problems: List, rho: float) -> Optional[Tuple[float, float]]:
        """Bootstrap CI for β."""
        n_bootstrap = 1000
        boot_betas = []
        
        for _ in range(n_bootstrap):
            # Resample problems
            idx = np.random.choice(len(problems), len(problems), replace=True)
            
            # Compute accuracies on bootstrap sample
            gen_acc = self._compute_accuracy_on_subset(generations, problems, idx)
            disc_acc = self._compute_disc_acc_on_subset(scores, idx)
            mutual_acc = self._compute_mutual_on_subset(generations, scores, problems, idx)
            
            if disc_acc is not None and disc_acc > gen_acc:
                beta = self.mcrf_law.estimate_beta(gen_acc, disc_acc, mutual_acc, rho)
                boot_betas.append(beta)
        
        if boot_betas:
            return (np.percentile(boot_betas, 2.5), np.percentile(boot_betas, 97.5))
        return None
    
    def _compute_accuracy_on_subset(self, generations, problems, indices) -> float:
        """Helper for bootstrap."""
        subset_problems = [problems[i] for i in indices]
        subset_pids = set(p.problem_id for p in subset_problems)
        subset_gens = [g for g in generations if g["problem_id"] in subset_pids]
        return self._compute_generator_accuracy(subset_gens, subset_problems)
    
    def _compute_disc_acc_on_subset(self, scores, indices) -> float:
        """Helper for bootstrap."""
        # Simplified - would need proper alignment
        return np.mean([s.get("is_consistent", False) for s in scores]) if scores else 0.0
    
    def _compute_mutual_on_subset(self, generations, scores, problems, indices) -> float:
        """Helper for bootstrap."""
        subset_problems = [problems[i] for i in indices]
        subset_pids = set(p.problem_id for p in subset_problems)
        subset_gens = [g for g in generations if g["problem_id"] in subset_pids]
        subset_scores = [s for s in scores if s["problem_id"] in subset_pids]
        return self._compute_mutual_consistency(subset_gens, subset_scores, subset_problems)
    
    def _per_problem_breakdown(self, generations: List[Dict], scores: List[Dict], problems: List) -> Dict:
        """Detailed per-problem metrics."""
        by_problem = {}
        for gen in generations:
            pid = gen["problem_id"]
            if pid not in by_problem:
                by_problem[pid] = {"generations": [], "scores": []}
            by_problem[pid]["generations"].append(gen)
        
        for score in scores:
            pid = score["problem_id"]
            if pid in by_problem:
                by_problem[pid]["scores"].append(score)
        
        breakdown = {}
        for pid, data in by_problem.items():
            problem = next(p for p in problems if p.problem_id == pid)
            true_ans = problem.answer
            nop_type = problem.nop_type or "clean"
            
            gen_correct = []
            for gen in data["generations"]:
                gen_ans = gen.get("extracted_answer")
                gen_correct.append(1 if (gen_ans is not None and abs(gen_ans - true_ans) < 1e-3) else 0)
            
            disc_consistent = [s.get("is_consistent", False) for s in data["scores"]]
            
            breakdown[pid] = {
                "nop_type": nop_type,
                "gen_acc": np.mean(gen_correct) if gen_correct else 0,
                "disc_consistency": np.mean(disc_consistent) if disc_consistent else 0,
                "mutual": np.mean([g and d for g, d in zip(gen_correct, disc_consistent)]) if gen_correct and disc_consistent else 0,
                "n_rollouts": len(gen_correct),
            }
        
        return breakdown
    
    def compute_mcrf_law_fit(self, all_metrics: List[MCRFMetrics]) -> Dict:
        """Fit MCRF law across all experiments."""
        # Convert to ExperimentMetrics format
        exp_metrics = []
        for m in all_metrics:
            if m.disc_acc is not None:
                exp_metrics.append(ExperimentMetrics(
                    exp_id=m.exp_id,
                    generator=m.generator,
                    discriminator=m.discriminator or "",
                    nop_type=m.nop_type,
                    gen_acc=m.gen_acc,
                    disc_acc=m.disc_acc,
                    mutual_acc=m.mutual_acc,
                    sc_acc=m.sc_acc,
                    rho=m.rho,
                ))
        
        if len(exp_metrics) < 2:
            return {"error": "Need at least 2 experiments with discriminator"}
        
        # Fit law
        params = self.mcrf_law.fit_law(exp_metrics, per_nop_type=True)
        
        # Cross-validate
        cv = self.mcrf_law.validate_law(exp_metrics)
        
        return {
            "parameters": {
                "beta": params.beta,
                "beta_ci": {k: list(v) for k, v in params.beta_ci.items()},
                "rho": params.rho,
                "rho_ci": {k: list(v) for k, v in params.rho_ci.items()},
                "epsilon": params.epsilon,
                "epsilon_ci": list(params.epsilon_ci),
                "r2": params.r2,
                "mse": params.mse,
            },
            "cross_validation": cv,
            "n_experiments": len(exp_metrics),
            "model_pairs": params.model_pairs,
        }
    
    def save_metrics(self, metrics: MCRFMetrics, output_dir: str):
        """Save metrics to YAML."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        data = {
            "exp_id": metrics.exp_id,
            "generator": metrics.generator,
            "discriminator": metrics.discriminator,
            "nop_type": metrics.nop_type,
            "gen_acc": metrics.gen_acc,
            "disc_acc": metrics.disc_acc,
            "mutual_acc": metrics.mutual_acc,
            "sc_acc": metrics.sc_acc,
            "rho": metrics.rho,
            "rho_ci": list(metrics.rho_ci) if metrics.rho_ci else None,
            "beta": metrics.beta,
            "beta_ci": list(metrics.beta_ci) if metrics.beta_ci else None,
            "defense_gain": metrics.defense_gain,
            "per_problem": metrics.per_problem,
        }
        
        with open(Path(output_dir) / "mcrf_metrics.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def aggregate_across_experiments(self, metrics_list: List[MCRFMetrics]) -> Dict:
        """Aggregate metrics across multiple experiments."""
        # Group by NoOp type
        by_type = {nt: [] for nt in self.nop_types}
        for m in metrics_list:
            if m.nop_type in by_type:
                by_type[m.nop_type].append(m)
        
        result = {}
        for nt, metrics in by_type.items():
            if not metrics:
                continue
            result[nt] = {
                "gen_acc": np.mean([m.gen_acc for m in metrics]),
                "disc_acc": np.mean([m.disc_acc for m in metrics if m.disc_acc is not None]) if any(m.disc_acc for m in metrics) else None,
                "mutual_acc": np.mean([m.mutual_acc for m in metrics]),
                "sc_acc": np.mean([m.sc_acc for m in metrics if m.sc_acc is not None]) if any(m.sc_acc for m in metrics) else None,
                "rho": np.mean([m.rho for m in metrics if m.rho is not None]) if any(m.rho for m in metrics) else None,
                "beta": np.mean([m.beta for m in metrics if m.beta is not None]) if any(m.beta for m in metrics) else None,
                "defense_gain": np.mean([m.defense_gain for m in metrics]),
                "n_experiments": len(metrics),
            }
        
        return result


def save_metrics_yaml(metrics: Dict, output_path: str):
    """Save metrics dict to YAML."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)


def load_metrics_yaml(input_path: str) -> Dict:
    """Load metrics from YAML."""
    with open(input_path) as f:
        return yaml.safe_load(f)