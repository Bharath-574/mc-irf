# MC-IRF Metrics Analysis
# Defense gain, bootstrap CI, error correlation, scaling laws

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json
import yaml
from pathlib import Path
from scipy import stats
from scipy.stats import bootstrap


def compute_defense_gain(
    baseline_accuracy: float,
    defense_accuracy: float,
) -> float:
    """
    Compute defense gain: (defense_acc - baseline_acc) / (1 - baseline_acc)
    Normalized to [0, 1] where 1 = perfect defense.
    """
    if baseline_accuracy >= 1.0:
        return 0.0
    return (defense_accuracy - baseline_accuracy) / (1 - baseline_accuracy)


def bootstrap_ci(
    data: List[float], 
    n_resamples: int = 10000, 
    confidence_level: float = 0.95,
    statistic: callable = np.mean,
) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    if len(data) < 2:
        return (data[0], data[0]) if data else (0.0, 0.0)
    
    data_arr = np.array(data)
    result = bootstrap(
        (data_arr,), 
        statistic, 
        n_resamples=n_resamples, 
        confidence_level=confidence_level,
        method='percentile',
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def paired_bootstrap_ci(
    baseline: List[float],
    defense: List[float],
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Paired bootstrap CI for difference."""
    if len(baseline) != len(defense) or len(baseline) < 2:
        return (0.0, 0.0)
    
    diffs = np.array(defense) - np.array(baseline)
    result = bootstrap(
        (diffs,),
        np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method='percentile',
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))


def compute_error_correlation(
    generator_errors: List[bool],  # True = correct, False = error
    discriminator_errors: List[bool],
) -> float:
    """
    Compute error correlation between generator and discriminator.
    Lower correlation = more independent errors = better defense.
    """
    gen_arr = np.array([0 if e else 1 for e in generator_errors])  # 1 = error
    disc_arr = np.array([0 if e else 1 for e in discriminator_errors])
    
    if len(gen_arr) < 2:
        return 1.0
    
    # Phi coefficient (for binary variables)
    n = len(gen_arr)
    n11 = np.sum((gen_arr == 1) & (disc_arr == 1))
    n10 = np.sum((gen_arr == 1) & (disc_arr == 0))
    n01 = np.sum((gen_arr == 0) & (disc_arr == 1))
    n00 = np.sum((gen_arr == 0) & (disc_arr == 0))
    
    # Phi = (n11*n00 - n10*n01) / sqrt((n11+n10)(n11+n01)(n00+n10)(n00+n01))
    numerator = n11 * n00 - n10 * n01
    denom = np.sqrt((n11+n10)*(n11+n01)*(n00+n10)*(n00+n01))
    
    if denom == 0:
        return 1.0
    
    return numerator / denom


def compute_defense_gain_vs_correlation(
    experiments_results: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Test the law: Defense Gain ∝ 1 / Error Correlation
    
    For each experiment with heterogeneous generator/discriminator:
    - Compute defense gain (vs homogeneous baseline)
    - Compute error correlation
    - Fit relationship
    """
    results = []
    
    for exp_id, metrics in experiments_results.items():
        if "error_correlation" in metrics and "defense_gain" in metrics:
            results.append({
                "exp_id": exp_id,
                "error_correlation": metrics["error_correlation"],
                "defense_gain": metrics["defense_gain"],
                "generator": metrics.get("generator"),
                "discriminator": metrics.get("discriminator"),
            })
    
    if len(results) < 3:
        return {"error": "Need at least 3 experiments"}
    
    # Fit: defense_gain = a / (error_corr + b) + c
    # Or simpler: defense_gain = k / error_correlation
    
    corrs = [r["error_correlation"] for r in results]
    gains = [r["defense_gain"] for r in results]
    
    # Linear fit on reciprocal
    inv_corrs = [1.0 / (c + 1e-6) for c in corrs]
    
    # Simple linear regression
    A = np.vstack([inv_corrs, np.ones(len(inv_corrs))]).T
    k, c = np.linalg.lstsq(A, gains, rcond=None)[0]
    
    # Predictions
    pred_gains = k * np.array(inv_corrs) + c
    r_squared = 1 - np.sum((np.array(gains) - pred_gains)**2) / np.sum((np.array(gains) - np.mean(gains))**2)
    
    return {
        "law": "defense_gain = k / error_correlation + c",
        "k": float(k),
        "c": float(c),
        "r_squared": float(r_squared),
        "data": results,
        "predictions": [
            {"exp_id": r["exp_id"], "predicted_gain": float(pg), "actual_gain": float(g)}
            for r, pg, g in zip(results, pred_gains, gains)
        ],
    }


def analyze_scale_ladder(
    generator_scales: List[float],  # param counts in B
    generator_accuracies: List[float],
    discriminator_scale: float,
    discriminator_accuracy: float,
) -> Dict[str, Any]:
    """
    Analyze generator scale effect with fixed discriminator.
    Tests: Generator scale helps but cannot close gap without discriminator scale.
    """
    # Fit: accuracy = a * log(scale) + b
    log_scales = np.log(generator_scales)
    accs = np.array(generator_accuracies)
    
    A = np.vstack([log_scales, np.ones(len(log_scales))]).T
    a, b = np.linalg.lstsq(A, accs, rcond=None)[0]
    
    # Predicted at discriminator scale
    pred_at_disc = a * np.log(discriminator_scale) + b
    
    # Gap
    gap = discriminator_accuracy - pred_at_disc
    
    return {
        "generator_scale_law": f"accuracy = {a:.4f} * log(scale) + {b:.4f}",
        "a": float(a),
        "b": float(b),
        "predicted_at_discriminator_scale": float(pred_at_disc),
        "actual_discriminator_accuracy": float(discriminator_accuracy),
        "gap": float(gap),
        "interpretation": "Positive gap means discriminator scale adds value beyond generator scale extrapolation",
    }


def analyze_discriminator_scale(
    discriminator_scales: List[float],
    discriminator_accuracies: List[float],
    generator_scale: float,
    generator_accuracy: float,
) -> Dict[str, Any]:
    """Analyze discriminator scale effect."""
    log_scales = np.log(discriminator_scales)
    accs = np.array(discriminator_accuracies)
    
    A = np.vstack([log_scales, np.ones(len(log_scales))]).T
    a, b = np.linalg.lstsq(A, accs, rcond=None)[0]
    
    return {
        "discriminator_scale_law": f"accuracy = {a:.4f} * log(scale) + {b:.4f}",
        "a": float(a),
        "b": float(b),
    }


def load_experiment_results(exp_dir: str) -> Dict[str, Any]:
    """Load results from experiment directory."""
    metrics_path = Path(exp_dir) / "metrics.yaml"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return yaml.safe_load(f)
    return {}


def aggregate_all_results(results_root: str = "experiments/results") -> Dict[str, Any]:
    """Aggregate metrics across all experiments."""
    root = Path(results_root)
    all_metrics = {}
    
    for exp_dir in sorted(root.iterdir()):
        if exp_dir.is_dir() and exp_dir.name.startswith("E"):
            metrics = load_experiment_results(str(exp_dir))
            if metrics:
                all_metrics[exp_dir.name] = metrics
    
    return all_metrics


def compute_all_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all derived metrics from raw experiment results."""
    # This would be called after loading raw generations/scores
    # Placeholder for now
    return {}


def save_metrics(metrics: Dict, output_path: str):
    """Save metrics to YAML."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(metrics, f)


if __name__ == "__main__":
    # Test bootstrap CI
    data = np.random.normal(0.7, 0.1, 100).tolist()
    ci = bootstrap_ci(data)
    print(f"Bootstrap CI: {ci}")
    
    # Test error correlation
    gen_errors = [False, True, False, False, True, False, False, True]
    disc_errors = [False, True, False, True, True, False, False, False]
    corr = compute_error_correlation(gen_errors, disc_errors)
    print(f"Error correlation: {corr:.3f}")