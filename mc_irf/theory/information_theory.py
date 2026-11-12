"""
Information-Theoretic Error Correlation
Implements ρ = I(E_G; E_D) / H(E_G, E_D) for MCRF law.

Uses exact discrete MI for binary errors, Kraskov k-NN for continuous.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


def mutual_information_kraskov(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Kraskov k-NN mutual information estimator.
    
    Works for continuous or discrete variables.
    I(X;Y) = ψ(k) + ψ(N) - <ψ(n_x + 1) + ψ(n_y + 1)>
    
    where ψ is digamma function, n_x/n_y are neighbor counts in marginal spaces.
    """
    N = len(x)
    if N < k + 1:
        return 0.0
    
    # Ensure 2D
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Joint space
    xy = np.hstack([x, y])
    
    # Find k-th nearest neighbor distances in joint space
    nbrs_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(xy)
    distances, _ = nbrs_joint.kneighbors(xy)
    eps = distances[:, k]  # k-th neighbor (0-indexed, so k-th is index k)
    
    # Count neighbors in marginal spaces within eps
    nbrs_x = NearestNeighbors(metric='chebyshev').fit(x)
    nbrs_y = NearestNeighbors(metric='chebyshev').fit(y)
    
    nx = np.array([len(nbrs_x.radius_neighbors([xi], radius=eps[i], return_distance=False)[0]) - 1 
                   for i, xi in enumerate(x)])
    ny = np.array([len(nbrs_y.radius_neighbors([yi], radius=eps[i], return_distance=False)[0]) - 1 
                   for i, yi in enumerate(y)])
    
    # Digamma
    from scipy.special import digamma
    mi = digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    
    return max(0.0, mi)


def entropy_kraskov(x: np.ndarray, k: int = 5) -> float:
    """Kraskov entropy estimator for continuous variables."""
    N = len(x)
    if N < k + 1:
        return 0.0
    
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(x)
    distances, _ = nbrs.kneighbors(x)
    eps = distances[:, k]
    
    from scipy.special import digamma
    # Volume of d-dimensional ball
    d = x.shape[1]
    vol = np.pi ** (d/2) / np.math.gamma(d/2 + 1) * (2*eps) ** d
    
    h = digamma(N) - digamma(k) + d * np.mean(np.log(eps + 1e-10)) + np.log(vol + 1e-10)
    return max(0.0, h)


def normalized_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    Normalized MI: I(X;Y) / H(X,Y)
    
    This gives ρ ∈ [0, 1] for error correlation.
    """
    mi = mutual_information_kraskov(x, y, k)
    
    # Joint entropy
    xy = np.hstack([x.reshape(-1, 1) if x.ndim == 1 else x,
                    y.reshape(-1, 1) if y.ndim == 1 else y])
    h_xy = entropy_kraskov(xy, k)
    
    if h_xy > 1e-10:
        return mi / h_xy
    return 0.0


def discrete_mutual_information(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Exact discrete MI for binary/categorical variables.
    
    Returns: (MI, H(X), H(Y), H(X,Y))
    """
    # Joint distribution
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


def normalized_discrete_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized discrete MI: I(X;Y) / H(X,Y)"""
    mi, hx, hy, hxy = discrete_mutual_information(x, y)
    if hxy > 1e-10:
        return mi / hxy
    return 0.0


def error_correlation_discrete(gen_errors: np.ndarray, disc_errors: np.ndarray) -> float:
    """
    Exact normalized MI for binary errors.
    
    This is the preferred method for binary error vectors.
    """
    return normalized_discrete_mi(gen_errors, disc_errors)


def error_correlation_mi(gen_errors: np.ndarray, disc_errors: np.ndarray, k: int = 5) -> float:
    """
    Compute error correlation ρ using normalized mutual information.
    
    Args:
        gen_errors: Binary array (1=error, 0=correct) for generator
        disc_errors: Binary array (1=error, 0=correct) for discriminator
        k: k-NN parameter for Kraskov estimator
    
    Returns:
        ρ ∈ [0, 1] normalized mutual information
    """
    # For binary errors, we can also use exact discrete MI
    # But Kraskov works and generalizes to continuous confidence scores
    return normalized_mutual_information(gen_errors.astype(float), disc_errors.astype(float), k)


def error_correlation_pearson(gen_errors: np.ndarray, disc_errors: np.ndarray) -> float:
    """Pearson correlation for binary errors (phi coefficient)."""
    return np.corrcoef(gen_errors, disc_errors)[0, 1]


def error_correlation_spearman(gen_errors: np.ndarray, disc_errors: np.ndarray) -> float:
    """Spearman rank correlation."""
    return stats.spearmanr(gen_errors, disc_errors)[0]


def error_correlation_matrix(gen_errors: np.ndarray, disc_errors: np.ndarray, 
                              method: str = "discrete_mi") -> Dict[str, float]:
    """
    Compute multiple correlation metrics for comparison.
    
    Returns dict with different correlation measures.
    """
    results = {}
    
    if method in ["discrete_mi", "all"]:
        results["discrete_mi"] = error_correlation_discrete(gen_errors, disc_errors)
    
    if method in ["kraskov_mi", "all"]:
        results["kraskov_mi"] = error_correlation_mi(gen_errors, disc_errors)
    
    if method in ["pearson", "all"]:
        results["pearson"] = error_correlation_pearson(gen_errors, disc_errors)
    
    if method in ["spearman", "all"]:
        results["spearman"] = error_correlation_spearman(gen_errors, disc_errors)
    
    # Cohen's kappa for agreement
    if method in ["kappa", "all"]:
        from sklearn.metrics import cohen_kappa_score
        results["kappa"] = cohen_kappa_score(gen_errors, disc_errors)
    
    return results


def bootstrap_error_correlation(gen_errors: np.ndarray, disc_errors: np.ndarray,
                                  n_bootstrap: int = 10000, 
                                  method: str = "discrete_mi",
                                  confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap confidence interval for error correlation.
    
    Returns: (point_estimate, (ci_lower, ci_upper))
    """
    n = len(gen_errors)
    point_est = error_correlation_matrix(gen_errors, disc_errors, method)[method]
    
    boot_estimates = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        boot_gen = gen_errors[idx]
        boot_disc = disc_errors[idx]
        boot_est = error_correlation_matrix(boot_gen, boot_disc, method)[method]
        boot_estimates.append(boot_est)
    
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(boot_estimates, alpha * 100)
    ci_upper = np.percentile(boot_estimates, (1 - alpha) * 100)
    
    return point_est, (ci_lower, ci_upper)


def conditional_error_correlation(gen_errors: np.ndarray, disc_errors: np.ndarray,
                                    condition: np.ndarray) -> Dict[str, float]:
    """
    Compute error correlation conditioned on a third variable.
    
    E.g., condition on NoOp type, problem difficulty, etc.
    """
    results = {}
    for cond_val in np.unique(condition):
        mask = condition == cond_val
        if mask.sum() > 10:
            results[str(cond_val)] = error_correlation_matrix(
                gen_errors[mask], disc_errors[mask]
            )
    return results


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 1000
    
    # Independent errors
    gen_err = np.random.binomial(1, 0.3, n)
    disc_err = np.random.binomial(1, 0.3, n)
    
    print("Independent errors:")
    print(f"  Discrete MI ρ: {error_correlation_discrete(gen_err, disc_err):.4f}")
    print(f"  Pearson ρ: {error_correlation_pearson(gen_err, disc_err):.4f}")
    print(f"  Kappa: {error_correlation_matrix(gen_err, disc_err, 'kappa')['kappa']:.4f}")
    
    # Correlated errors
    gen_err = np.random.binomial(1, 0.3, n)
    # Discriminator errors correlate with generator
    disc_err = np.where(gen_err == 1, 
                        np.random.binomial(1, 0.7, n),  # High error when gen errors
                        np.random.binomial(1, 0.1, n))  # Low error when gen correct
    
    print("\nCorrelated errors (ρ≈0.6):")
    print(f"  Discrete MI ρ: {error_correlation_discrete(gen_err, disc_err):.4f}")
    print(f"  Pearson ρ: {error_correlation_pearson(gen_err, disc_err):.4f}")
    print(f"  Kappa: {error_correlation_matrix(gen_err, disc_err, 'kappa')['kappa']:.4f}")
    
    # Bootstrap CI
    point, ci = bootstrap_error_correlation(gen_err, disc_err, n_bootstrap=1000)
    print(f"  Bootstrap CI: [{ci[0]:.4f}, {ci[1]:.4f}]")