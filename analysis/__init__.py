# MC-IRF Analysis Package

from analysis.attention.extract import (
    extract_attention_for_problems,
    quantify_noop_attention_leakage,
    run_attention_analysis,
)
from analysis.causal.intervene import (
    CausalTracer,
    run_causal_tracing,
)
from analysis.metrics.defense import (
    compute_defense_gain,
    bootstrap_ci,
    paired_bootstrap_ci,
    compute_error_correlation,
    compute_defense_gain_vs_correlation,
    analyze_scale_ladder,
    aggregate_all_results,
)

__all__ = [
    "extract_attention_for_problems",
    "quantify_noop_attention_leakage",
    "run_attention_analysis",
    "CausalTracer",
    "run_causal_tracing",
    "compute_defense_gain",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "compute_error_correlation",
    "compute_defense_gain_vs_correlation",
    "analyze_scale_ladder",
    "aggregate_all_results",
]