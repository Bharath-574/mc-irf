"""
MC-IRF Experiment Registry
Centralized experiment definitions and loading.
"""

from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path


# Experiment registry loaded from YAML
_EXPERIMENTS: List[Dict] = None
_TIERS: Dict[str, List[str]] = None


def load_experiments(config_path: str = "configs/experiment_matrix.yaml") -> List[Dict]:
    """Load experiments from YAML config."""
    global _EXPERIMENTS, _TIERS
    if _EXPERIMENTS is not None:
        return _EXPERIMENTS
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    _EXPERIMENTS = config.get("experiments", [])
    _TIERS = config.get("hardware", {}).get("tiers", {
        "local": ["E0_local", "E1", "E2", "E5_local", "E3_local", "E4_local"],
        "h200": ["E0", "E1", "E2", "E5", "E6", "E6b", "E6c", "E3", "E4", "E10", "E7", "E9", "E9_clean", "E11", "E12"],
    })
    
    return _EXPERIMENTS


def get_experiment(exp_id: str, config_path: str = "configs/experiment_matrix.yaml") -> Dict:
    """Get experiment config by ID."""
    experiments = load_experiments(config_path)
    for exp in experiments:
        if exp["id"] == exp_id:
            return exp
    raise ValueError(f"Unknown experiment: {exp_id}")


def get_tier_experiments(tier: str, config_path: str = "configs/experiment_matrix.yaml") -> List[str]:
    """Get experiment IDs for a tier."""
    load_experiments(config_path)
    return _TIERS.get(tier, [])


def list_experiments(config_path: str = "configs/experiment_matrix.yaml") -> List[Dict]:
    """List all experiments."""
    return load_experiments(config_path)


def list_experiments_by_environment(environment: str, config_path: str = "configs/experiment_matrix.yaml") -> List[Dict]:
    """List experiments filtered by environment (local/h200)."""
    experiments = load_experiments(config_path)
    tier = "local" if environment == "local" else "h200"
    tier_exp_ids = get_tier_experiments(tier, config_path)
    return [e for e in experiments if e["id"] in tier_exp_ids]


if __name__ == "__main__":
    # Test loading
    exps = list_experiments()
    print(f"Total experiments: {len(exps)}")
    for e in exps:
        print(f"  {e['id']}: {e['name']} (tier {e.get('tier', '?')})")