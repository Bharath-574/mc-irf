"""
MCRF Scaling Law Implementation
Core theory: Mutual Consistency Relevance Filter

Law: M_acc = G_acc + β(τ) * (1 - ρ) * (D_acc - G_acc) + ε

Where:
- M_acc: Mutual consistency accuracy
- G_acc: Generator accuracy  
- D_acc: Discriminator accuracy
- β(τ): Relevance filtering efficiency per NoOp type τ
- ρ: Error correlation between generator and discriminator
- ε: Residual (higher-order interactions)
"""

import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MCRFParameters:
    """Fitted MCRF law parameters with confidence intervals."""
    beta: Dict[str, float]                    # Per NoOp type
    beta_ci: Dict[str, Tuple[float, float]]   # 95% CI
    rho: Dict[str, float]                     # Per NoOp type (or per pair)
    rho_ci: Dict[str, Tuple[float, float]]
    epsilon: float
    epsilon_ci: Tuple[float, float]
    r2: float
    mse: float
    n_samples: int
    model_pairs: List[str]
    nop_types: List[str]


@dataclass
class ExperimentMetrics:
    """Metrics from a single experiment."""
    exp_id: str
    generator: str
    discriminator: Optional[str]
    nop_type: str
    gen_acc: float
    disc_acc: Optional[float]
    mutual_acc: float
    sc_acc: Optional[float] = None
    rho: Optional[float] = None
    beta: Optional[float] = None
    n_problems: int = 0
    n_rollouts: int = 0
    seeds: List[int] = field(default_factory=list)


class MCRFLaw:
    """
    MCRF Scaling Law estimator with bootstrap confidence intervals.
    
    Implements:
    1. Error correlation ρ via mutual information
    2. β(τ) estimation per NoOp type
    3. Bootstrap CI for all parameters
    4. Prediction and validation
    """
    
    def __init__(self, config_path: str = "configs/mcrf_law.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.nop_types = ["quantity", "comparison", "filter", "discount"]
        self.bootstrap_config = self.config["bootstrap"]
        self.error_corr_config = self.config["error_correlation"]
        self.beta_config = self.config["beta_estimation"]
    
    def compute_error_correlation(self, 
                                   gen_errors: np.ndarray, 
                                   disc_errors: np.ndarray) -> float:
        """
        Compute error correlation ρ between generator and discriminator.
        
        Uses mutual information normalized by joint entropy.
        """
        method = self.error_corr_config["method"]
        
        if method == "mutual_information":
            return self._mutual_information_correlation(gen_errors, disc_errors)
        elif method == "pearson":
            return np.corrcoef(gen_errors, disc_errors)[0, 1]
        elif method == "spearman":
            return stats.spearmanr(gen_errors, disc_errors)[0]
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _mutual_information_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Normalized mutual information: I(X;Y) / H(X,Y)"""
        # For binary errors, use exact discrete MI
        return self._normalized_discrete_mi(x, y)
    
    def _normalized_discrete_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Normalized discrete MI: I(X;Y) / H(X,Y)"""
        mi, hx, hy, hxy = self._discrete_mutual_information(x, y)
        if hxy > 1e-10:
            return mi / hxy
        return 0.0
    
    def _discrete_mutual_information(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Exact discrete MI for binary/categorical variables.
        
        Returns: (MI, H(X), H(Y), H(X,Y))
        """
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        
        px = np.array([np.mean(x == ux) for ux in unique_x])
        py = np.array([np.mean(y == uy) for uy in unique_y])
        
        # Joint
        pxy = np.zeros((len(unique_x), len(unique_y)))
        for i, ux in enumerate(unique_x):
            for j, uy in enumerate(unique_y):
                pxy[i, j] = np.mean((x == ux) & (y == uy))
        
        # Entropies
        hx = -np.sum(px * np.log(px + 1e-10))
        hy = -np.sum(py * np.log(py + 1e-10))
        hxy = -np.sum(pxy * np.log(pxy + 1e-10))
        
        # MI
        mi = 0.0
        for i in range(len(unique_x)):
            for j in range(len(unique_y)):
                if pxy[i, j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j] + 1e-10))
        
        return mi, hx, hy, hxy
    
    def estimate_beta(self, 
                      gen_acc: float, 
                      disc_acc: float, 
                      mutual_acc: float, 
                      rho: float) -> float:
        """
        Estimate β from observed accuracies and ρ.
        
        β = (M_acc - G_acc) / [(1 - ρ) * (D_acc - G_acc)]
        """
        denominator = (1 - rho) * (disc_acc - gen_acc)
        if abs(denominator) < 1e-8:
            return 0.0
        beta = (mutual_acc - gen_acc) / denominator
        return np.clip(beta, 0.0, 1.0)
    
    def fit_law(self, 
                experiments: List[ExperimentMetrics],
                per_nop_type: bool = True) -> MCRFParameters:
        """
        Fit MCRF law parameters across experiments.
        
        Uses MLE with bootstrap CI.
        """
        # Group by NoOp type
        if per_nop_type:
            groups = {nt: [] for nt in self.nop_types}
            for exp in experiments:
                if exp.nop_type in groups:
                    groups[exp.nop_type].append(exp)
        else:
            groups = {"all": experiments}
        
        beta = {}
        beta_ci = {}
        rho = {}
        rho_ci = {}
        
        all_pred = []
        all_true = []
        
        for nt, exps in groups.items():
            if not exps:
                continue
            
            # Compute ρ and β for each experiment
            rhos = []
            betas = []
            preds = []
            trues = []
            
            for exp in exps:
                if exp.disc_acc is None or exp.rho is None:
                    continue
                
                rho_val = exp.rho
                beta_val = self.estimate_beta(exp.gen_acc, exp.disc_acc, exp.mutual_acc, rho_val)
                
                rhos.append(rho_val)
                betas.append(beta_val)
                
                # Prediction
                pred = exp.gen_acc + beta_val * (1 - rho_val) * (exp.disc_acc - exp.gen_acc)
                preds.append(pred)
                trues.append(exp.mutual_acc)
            
            if rhos:
                rho[nt] = np.mean(rhos)
                beta[nt] = np.mean(betas)
                all_pred.extend(preds)
                all_true.extend(trues)
        
        # Bootstrap confidence intervals
        beta_ci = self._bootstrap_beta(experiments, per_nop_type)
        rho_ci = self._bootstrap_rho(experiments, per_nop_type)
        epsilon_ci = self._bootstrap_epsilon(all_true, all_pred)
        
        # Overall fit metrics
        all_pred = np.array(all_pred)
        all_true = np.array(all_true)
        mse = np.mean((all_pred - all_true) ** 2)
        r2 = 1 - np.sum((all_true - all_pred) ** 2) / np.sum((all_true - np.mean(all_true)) ** 2)
        
        return MCRFParameters(
            beta=beta,
            beta_ci=beta_ci,
            rho=rho,
            rho_ci=rho_ci,
            epsilon=np.mean(all_true - all_pred),
            epsilon_ci=epsilon_ci,
            r2=r2,
            mse=mse,
            n_samples=len(all_true),
            model_pairs=list(set(f"{e.generator}→{e.discriminator}" for e in experiments if e.discriminator)),
            nop_types=self.nop_types
        )
    
    def _bootstrap_beta(self, experiments: List[ExperimentMetrics], per_nop_type: bool) -> Dict:
        """Bootstrap CI for β."""
        n_samples = self.bootstrap_config["n_samples"]
        ci_level = self.bootstrap_config["confidence_level"]
        alpha = (1 - ci_level) / 2
        
        if per_nop_type:
            result = {}
            for nt in self.nop_types:
                nt_exps = [e for e in experiments if e.nop_type == nt and e.disc_acc is not None]
                if len(nt_exps) < 2:
                    result[nt] = (0.0, 1.0)
                    continue
                
                boot_betas = []
                for _ in range(n_samples):
                    sample = resample(nt_exps, replace=True, random_state=np.random.randint(1e9))
                    betas = []
                    for exp in sample:
                        if exp.rho is not None:
                            b = self.estimate_beta(exp.gen_acc, exp.disc_acc, exp.mutual_acc, exp.rho)
                            betas.append(b)
                    if betas:
                        boot_betas.append(np.mean(betas))
                
                if boot_betas:
                    result[nt] = (np.percentile(boot_betas, alpha * 100), 
                                  np.percentile(boot_betas, (1 - alpha) * 100))
                else:
                    result[nt] = (0.0, 1.0)
            return result
        else:
            all_exps = [e for e in experiments if e.disc_acc is not None]
            boot_betas = []
            for _ in range(n_samples):
                sample = resample(all_exps, replace=True, random_state=np.random.randint(1e9))
                betas = [self.estimate_beta(e.gen_acc, e.disc_acc, e.mutual_acc, e.rho) 
                         for e in sample if e.rho is not None]
                if betas:
                    boot_betas.append(np.mean(betas))
            return {"all": (np.percentile(boot_betas, alpha * 100), 
                            np.percentile(boot_betas, (1 - alpha) * 100))}
    
    def _bootstrap_rho(self, experiments: List[ExperimentMetrics], per_nop_type: bool) -> Dict:
        """Bootstrap CI for ρ."""
        n_samples = self.bootstrap_config["n_samples"]
        ci_level = self.bootstrap_config["confidence_level"]
        alpha = (1 - ci_level) / 2
        
        if per_nop_type:
            result = {}
            for nt in self.nop_types:
                nt_exps = [e for e in experiments if e.nop_type == nt and e.rho is not None]
                if len(nt_exps) < 2:
                    result[nt] = (0.0, 1.0)
                    continue
                
                boot_rhos = []
                for _ in range(n_samples):
                    sample = resample(nt_exps, replace=True, random_state=np.random.randint(1e9))
                    rhos = [e.rho for e in sample]
                    boot_rhos.append(np.mean(rhos))
                
                result[nt] = (np.percentile(boot_rhos, alpha * 100), 
                              np.percentile(boot_rhos, (1 - alpha) * 100))
            return result
        else:
            all_exps = [e for e in experiments if e.rho is not None]
            boot_rhos = []
            for _ in range(n_samples):
                sample = resample(all_exps, replace=True, random_state=np.random.randint(1e9))
                boot_rhos.append(np.mean([e.rho for e in sample]))
            return {"all": (np.percentile(boot_rhos, alpha * 100), 
                            np.percentile(boot_rhos, (1 - alpha) * 100))}
    
    def _bootstrap_epsilon(self, y_true: List, y_pred: List) -> Tuple[float, float]:
        """Bootstrap CI for ε (residual mean)."""
        n_samples = self.bootstrap_config["n_samples"]
        ci_level = self.bootstrap_config["confidence_level"]
        alpha = (1 - ci_level) / 2
        
        residuals = np.array(y_true) - np.array(y_pred)
        boot_eps = []
        for _ in range(n_samples):
            sample = resample(residuals, replace=True, random_state=np.random.randint(1e9))
            boot_eps.append(np.mean(sample))
        
        return (np.percentile(boot_eps, alpha * 100), 
                np.percentile(boot_eps, (1 - alpha) * 100))
    
    def predict(self, 
                gen_acc: float, 
                disc_acc: float, 
                rho: float, 
                nop_type: str,
                beta: Optional[float] = None) -> float:
        """Predict mutual consistency accuracy using fitted law."""
        if beta is None:
            beta = self.fitted_beta.get(nop_type, 0.5)
        return gen_acc + beta * (1 - rho) * (disc_acc - gen_acc)
    
    def validate_law(self, 
                     experiments: List[ExperimentMetrics],
                     holdout_fraction: float = 0.2) -> Dict:
        """Cross-validate the law."""
        np.random.shuffle(experiments)
        n_holdout = int(len(experiments) * holdout_fraction)
        train = experiments[n_holdout:]
        test = experiments[:n_holdout]
        
        params = self.fit_law(train)
        
        predictions = []
        truths = []
        for exp in test:
            if exp.disc_acc is not None and exp.rho is not None:
                pred = self.predict(exp.gen_acc, exp.disc_acc, exp.rho, exp.nop_type)
                predictions.append(pred)
                truths.append(exp.mutual_acc)
        
        if not predictions:
            return {"mse": None, "mae": None, "r2": None}
        
        predictions = np.array(predictions)
        truths = np.array(truths)
        
        return {
            "mse": float(np.mean((predictions - truths) ** 2)),
            "mae": float(np.mean(np.abs(predictions - truths))),
            "r2": float(1 - np.sum((truths - predictions) ** 2) / np.sum((truths - np.mean(truths)) ** 2)),
            "n_test": len(predictions)
        }


def save_mcrf_params(params: MCRFParameters, output_path: str):
    """Save MCRF parameters to YAML."""
    data = {
        "beta": params.beta,
        "beta_ci": {k: list(v) for k, v in params.beta_ci.items()},
        "rho": params.rho,
        "rho_ci": {k: list(v) for k, v in params.rho_ci.items()},
        "epsilon": params.epsilon,
        "epsilon_ci": list(params.epsilon_ci),
        "r2": params.r2,
        "mse": params.mse,
        "n_samples": params.n_samples,
        "model_pairs": params.model_pairs,
        "nop_types": params.nop_types
    }
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_mcrf_params(input_path: str) -> MCRFParameters:
    """Load MCRF parameters from YAML."""
    with open(input_path) as f:
        data = yaml.safe_load(f)
    return MCRFParameters(
        beta=data["beta"],
        beta_ci={k: tuple(v) for k, v in data["beta_ci"].items()},
        rho=data["rho"],
        rho_ci={k: tuple(v) for k, v in data["rho_ci"].items()},
        epsilon=data["epsilon"],
        epsilon_ci=tuple(data["epsilon_ci"]),
        r2=data["r2"],
        mse=data["mse"],
        n_samples=data["n_samples"],
        model_pairs=data["model_pairs"],
        nop_types=data["nop_types"]
    )


if __name__ == "__main__":
    # Test with synthetic data
    law = MCRFLaw()
    
    experiments = [
        ExperimentMetrics("E1", "phi", "phi", "quantity", 0.3, 0.3, 0.45, rho=0.9),
        ExperimentMetrics("E2", "phi", "phi", "comparison", 0.25, 0.25, 0.4, rho=0.85),
        ExperimentMetrics("E5", "phi", "qwen7b", "quantity", 0.3, 0.5, 0.65, rho=0.4),
        ExperimentMetrics("E6", "phi", "qwen14b", "quantity", 0.3, 0.7, 0.75, rho=0.2),
    ]
    
    params = law.fit_law(experiments)
    print("Fitted MCRF Parameters:")
    print(f"  β: {params.beta}")
    print(f"  β CI: {params.beta_ci}")
    print(f"  ρ: {params.rho}")
    print(f"  ρ CI: {params.rho_ci}")
    print(f"  ε: {params.epsilon:.4f} [{params.epsilon_ci[0]:.4f}, {params.epsilon_ci[1]:.4f}]")
    print(f"  R²: {params.r2:.4f}")
    print(f"  MSE: {params.mse:.4f}")