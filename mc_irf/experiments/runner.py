"""
MC-IRF Experiment Runner
Core pipeline for running experiments with checkpointing and MCRF metrics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import torch

from mc_irf.models.registry import get_registry, ModelRegistry, SequentialLoader
from mc_irf.data.gsm_symbolic import GSMProblem, load_problems, generate_experiment_problems, CORE_TEMPLATES
from mc_irf.experiments.checkpoint import ExperimentStateMachine, load_experiment_state
from mc_irf.experiments.metrics import MetricsComputer, MCRFMetrics
from mc_irf.theory.information_theory import error_correlation_discrete


@dataclass
class ExperimentConfig:
    """Experiment configuration from YAML."""
    exp_id: str
    name: str
    generator: str
    discriminator: Optional[str]
    nop_types: List[str]
    seeds: List[int]
    rollouts: int
    problems: int
    claim: str
    baselines: List[str] = field(default_factory=list)
    tracing: Optional[str] = None
    generators: List[str] = field(default_factory=list)
    intervention_layers: Optional[List[int]] = None


@dataclass
class GenerationResult:
    """Single generation result."""
    problem_id: str
    seed: int
    rollout: int
    prompt: str
    generation: str
    tokens: List[int]
    extracted_answer: Optional[float]
    is_correct: bool
    logprobs: Optional[List[float]] = None


@dataclass
class DiscriminatorScore:
    """Discriminator score for a generation."""
    problem_id: str
    seed: int
    rollout: int
    score: float
    is_consistent: bool
    reasoning: str


# ============================================================
# Prompt Templates
# ============================================================

GENERATOR_PROMPT = """Solve this math problem step by step. Put your final answer in \\boxed{}.

Problem: {question}

Let's think step by step:"""

DISCRIMINATOR_PROMPT = """You are a careful verifier. Check if the proposed solution is correct.

Problem: {question}

Proposed solution:
{solution}

Is the solution correct? Answer with "Yes" or "No" and brief reasoning.
Format: 
Verdict: Yes/No
Reasoning: ..."""


# ============================================================
# Answer Extraction
# ============================================================

def extract_answer(text: str) -> Optional[float]:
    """Extract numerical answer from generation text."""
    import re
    
    # \boxed{...}
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        try:
            return float(boxed.group(1).replace(',', ''))
        except:
            pass
    
    # "answer is X" or "Answer: X"
    patterns = [
        r'answer is\s*([\d.]+)',
        r'answer:\s*([\d.]+)',
        r'final answer.*?([\d.]+)',
        r'= ([\d.]+)\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(',', ''))
            except:
                pass
    
    # Last number in text
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass
    
    return None


def check_correctness(extracted: Optional[float], true_answer: float, tolerance: float = 1e-3) -> bool:
    """Check if extracted answer matches true answer."""
    if extracted is None:
        return False
    return abs(extracted - true_answer) < tolerance


# ============================================================
# Experiment Runner
# ============================================================

class ExperimentRunner:
    """Runs a single experiment with checkpointing and MCRF metrics."""
    
    def __init__(self, config: ExperimentConfig, registry: ModelRegistry = None, environment: str = "h200"):
        self.config = config
        self.environment = environment
        self.registry = registry or get_registry(
            "configs/models_h200.yaml" if environment == "h200" else "configs/models_local.yaml"
        )
        self.metrics_computer = MetricsComputer()
        
        # Load or generate problems
        self.problems = self._load_problems()
        
        # Initialize state machine
        checkpoint_dir = f"experiments/checkpoints_{environment}"
        self.state_machine = ExperimentStateMachine(config.exp_id, config.__dict__, checkpoint_dir)
        self.state_machine.build_problem_order(self.problems, config.seeds, config.rollouts)
        
        # Try to resume
        self.state_machine.load_checkpoint()
    
    def _load_problems(self) -> List[GSMProblem]:
        """Load or generate problems for this experiment."""
        env_suffix = "h200" if self.environment == "h200" else "local"
        problem_path = f"data/processed/{self.config.exp_id}_problems.jsonl"
        if Path(problem_path).exists():
            return load_problems(problem_path)
        
        # Generate from templates
        problems = generate_experiment_problems(
            exp_config=self.config.__dict__,
            templates=CORE_TEMPLATES,
        )
        return problems
    
    def run_generator(self, generator, problem: GSMProblem, seed: int, rollout: int) -> GenerationResult:
        """Run single generator rollout."""
        prompt = GENERATOR_PROMPT.format(question=problem.question)
        
        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 512,
            "n": 1,
            "seed": seed + rollout * 1000,  # Deterministic per rollout
        }
        
        outputs = generator.generate_single(prompt, sampling_params)
        gen = outputs[0]
        
        extracted = extract_answer(gen.text)
        is_correct = check_correctness(extracted, problem.answer)
        
        return GenerationResult(
            problem_id=problem.problem_id,
            seed=seed,
            rollout=rollout,
            prompt=prompt,
            generation=gen.text,
            tokens=gen.tokens,
            extracted_answer=extracted,
            is_correct=is_correct,
            logprobs=gen.logprobs,
        )
    
    def run_discriminator(self, discriminator, problem: GSMProblem, generation_text: str) -> DiscriminatorScore:
        """Run discriminator on a generation."""
        prompt = DISCRIMINATOR_PROMPT.format(
            question=problem.question,
            solution=generation_text
        )
        
        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "n": 1,
        }
        
        outputs = discriminator.generate_single(prompt, sampling_params)
        disc_text = outputs[0].text
        
        # Parse verdict
        verdict = "No"
        if "Verdict: Yes" in disc_text or "verdict: yes" in disc_text.lower():
            verdict = "Yes"
        
        is_consistent = (verdict == "Yes")
        score = 1.0 if is_consistent else 0.0
        
        return DiscriminatorScore(
            problem_id=problem.problem_id,
            seed=0,  # Filled later
            rollout=0,
            score=score,
            is_consistent=is_consistent,
            reasoning=disc_text,
        )
    
    def run_generator_phase(self) -> List[GenerationResult]:
        """Run generator phase with checkpointing."""
        print(f"Running generator phase for {self.config.exp_id}")
        
        # Load generator
        generator = self.registry.load(self.config.generator)
        
        # Process tasks
        while True:
            task = self.state_machine.get_next_task()
            if task is None:
                break
            
            problem, seed, rollout = task
            
            gen_result = self.run_generator(generator, problem, seed, rollout)
            
            # Convert to dict for storage
            gen_dict = {
                "problem_id": gen_result.problem_id,
                "seed": gen_result.seed,
                "rollout": gen_result.rollout,
                "prompt": gen_result.prompt,
                "generation": gen_result.generation,
                "tokens": gen_result.tokens,
                "extracted_answer": gen_result.extracted_answer,
                "is_correct": gen_result.is_correct,
                "logprobs": gen_result.logprobs,
            }
            
            self.state_machine.add_generation(gen_dict)
            self.state_machine.advance()
            
            # Checkpoint every 5 problems
            self.state_machine.checkpoint()
        
        # Unload generator
        self.registry.unload(self.config.generator)
        
        return self.state_machine.generations
    
    def run_discriminator_phase(self) -> List[DiscriminatorScore]:
        """Run discriminator phase with checkpointing."""
        if not self.config.discriminator:
            print("No discriminator configured, skipping discriminator phase")
            return []
        
        print(f"Running discriminator phase for {self.config.exp_id}")
        
        # Load discriminator
        discriminator = self.registry.load(self.config.discriminator)
        
        # Reset for discriminator phase
        self.state_machine.phase = "discriminator"
        self.state_machine.current_problem_idx = 0
        
        # Process each generation
        for gen_dict in self.state_machine.generations:
            problem = next(p for p in self.problems if p.problem_id == gen_dict["problem_id"])
            
            score = self.run_discriminator(discriminator, problem, gen_dict["generation"])
            score.seed = gen_dict["seed"]
            score.rollout = gen_dict["rollout"]
            
            score_dict = {
                "problem_id": score.problem_id,
                "seed": score.seed,
                "rollout": score.rollout,
                "score": score.score,
                "is_consistent": score.is_consistent,
                "reasoning": score.reasoning,
            }
            
            self.state_machine.add_discriminator_score(score_dict)
            self.state_machine.advance()
            self.state_machine.checkpoint()
        
        # Unload discriminator
        self.registry.unload(self.config.discriminator)
        
        return self.state_machine.discriminator_scores
    
    def run_experiment(self) -> MCRFMetrics:
        """Run full experiment."""
        print(f"\n{'='*60}")
        print(f"Running {self.config.exp_id}: {self.config.name}")
        print(f"Generator: {self.config.generator}")
        print(f"Discriminator: {self.config.discriminator}")
        print(f"Problems: {len(self.problems)}")
        print(f"Rollouts: {self.config.rollouts}")
        print(f"Seeds: {self.config.seeds}")
        print(f"Environment: {self.environment}")
        print(f"{'='*60}\n")
        
        # Run generator phase
        if self.state_machine.phase == "generator":
            self.run_generator_phase()
        
        # Run discriminator phase
        if self.config.discriminator and self.state_machine.phase == "discriminator":
            self.run_discriminator_phase()
        
        # Compute metrics
        print("Computing metrics...")
        self.state_machine.metrics = self.metrics_computer.compute_experiment_metrics(
            self.config.exp_id,
            self.config.__dict__,
            self.state_machine.generations,
            self.state_machine.discriminator_scores,
            self.problems,
        ).__dict__
        
        # Save final results
        self.state_machine.save_final_results()
        self.metrics_computer.save_metrics(
            self.metrics_computer.compute_experiment_metrics(
                self.config.exp_id,
                self.config.__dict__,
                self.state_machine.generations,
                self.state_machine.discriminator_scores,
                self.problems,
            ),
            f"experiments/results_{self.environment}/{self.config.exp_id}"
        )
        
        # Print summary
        m = self.state_machine.metrics
        print(f"\n{'='*60}")
        print(f"{self.config.exp_id} Complete")
        print(f"Generator Accuracy: {m.get('gen_acc', 0):.3f}")
        if m.get('disc_acc'):
            print(f"Discriminator Consistency: {m['disc_acc']:.3f}")
        print(f"Mutual Consistency: {m.get('mutual_acc', 0):.3f}")
        if m.get('rho'):
            print(f"Error Correlation (ρ): {m['rho']:.3f}")
        if m.get('beta'):
            print(f"Filtering Efficiency (β): {m['beta']:.3f}")
        print(f"Defense Gain: {m.get('defense_gain', 0):.3f}")
        print(f"{'='*60}\n")
        
        return self.metrics_computer.compute_experiment_metrics(
            self.config.exp_id,
            self.config.__dict__,
            self.state_machine.generations,
            self.state_machine.discriminator_scores,
            self.problems,
        )


def load_experiment_config(exp_id: str, config_path: str = "configs/experiment_matrix.yaml") -> ExperimentConfig:
    """Load experiment config from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for exp in config.get("experiments", []):
        if exp["id"] == exp_id:
            return ExperimentConfig(**exp)
    
    raise ValueError(f"Experiment {exp_id} not found in {config_path}")


def run_experiment(exp_id: str, environment: str = "h200") -> MCRFMetrics:
    """Main entry point to run an experiment."""
    config = load_experiment_config(exp_id)
    runner = ExperimentRunner(config, environment=environment)
    return runner.run_experiment()


if __name__ == "__main__":
    import sys
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "E1"
    env = sys.argv[2] if len(sys.argv) > 2 else "h200"
    result = run_experiment(exp_id, env)
    print(f"\n=== {exp_id} Complete ===")
    print(f"Generator Accuracy: {result.gen_acc:.3f}")
    if result.disc_acc:
        print(f"Discriminator Consistency: {result.disc_acc:.3f}")
    print(f"Mutual Consistency: {result.mutual_acc:.3f}")