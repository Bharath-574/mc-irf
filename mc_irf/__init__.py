"""
MC-IRF Main Package
"""

from mc_irf.theory.mcrf_law import MCRFLaw, MCRFParameters, ExperimentMetrics
from mc_irf.theory.causal_equivalence import CausalEquivalenceTester, EquivalenceResult
from mc_irf.theory.information_theory import (
    error_correlation_discrete,
    error_correlation_mi,
    normalized_mutual_information,
    bootstrap_error_correlation,
)

from mc_irf.models.registry import get_registry, ModelRegistry, SequentialLoader
from mc_irf.models.wrappers.base import ModelWrapper, ModelConfig, GenerationOutput
from mc_irf.models.wrappers.vllm import VLLMWrapper
from mc_irf.models.wrappers.hf import HFWrapper
from mc_irf.models.wrappers.llamacpp import LlamaCppWrapper

from mc_irf.data.gsm_symbolic import (
    GSMProblem, GSMTemplate, CORE_TEMPLATES,
    generate_problem_set, save_problems, load_problems,
    inject_noop, NOOP_TEMPLATES, IRRELEVANT_POOLS,
)

from mc_irf.experiments.runner import ExperimentRunner, ExperimentConfig, run_experiment
from mc_irf.experiments.checkpoint import ExperimentStateMachine, load_experiment_state
from mc_irf.experiments.metrics import MetricsComputer, MCRFMetrics

from mc_irf.analysis.attention.extract import (
    extract_attention_for_problems,
    quantify_noop_attention_leakage,
    run_attention_analysis,
)
from mc_irf.analysis.causal.intervene import (
    CausalTracer, run_causal_tracing, summarize_causal_results,
)
from mc_irf.analysis.causal.equivalence import (
    CausalEquivalenceTester, run_equivalence_test,
)

__version__ = "1.0.0"

__all__ = [
    # Theory
    "MCRFLaw", "MCRFParameters", "ExperimentMetrics",
    "CausalEquivalenceTester", "EquivalenceResult",
    "error_correlation_discrete", "error_correlation_mi",
    "normalized_mutual_information", "bootstrap_error_correlation",
    
    # Models
    "get_registry", "ModelRegistry", "SequentialLoader",
    "ModelWrapper", "ModelConfig", "GenerationOutput",
    "VLLMWrapper", "HFWrapper", "LlamaCppWrapper",
    
    # Data
    "GSMProblem", "GSMTemplate", "CORE_TEMPLATES",
    "generate_problem_set", "save_problems", "load_problems",
    "inject_noop", "NOOP_TEMPLATES", "IRRELEVANT_POOLS",
    
    # Experiments
    "ExperimentRunner", "ExperimentConfig", "run_experiment",
    "ExperimentStateMachine", "load_experiment_state",
    "MetricsComputer", "MCRFMetrics",
    
    # Analysis
    "extract_attention_for_problems", "quantify_noop_attention_leakage", "run_attention_analysis",
    "CausalTracer", "run_causal_tracing", "summarize_causal_results",
    "CausalEquivalenceTester", "run_equivalence_test",
]