# MC-IRF Experiment Runner - Core Pipeline
# Runs E1-E9 experiments with mutual consistency evaluation

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from models.registry import get_registry, ModelRegistry
from models.wrappers.base import GenerationOutput
from data.gsm_symbolic import GSMProblem, load_problems, generate_problem_set, CORE_TEMPLATES


@dataclass
class ExperimentConfig:
    """Experiment configuration from YAML."""
    exp_id: str
    name: str
    generator: str
    discriminator: str
    nop_types: List[str]
    seeds: List[int]
    rollouts: int
    problems: int
    claim: str
    baselines: List[str] = field(default_factory=list)
    tracing: Optional[str] = None
    generators: List[str] = field(default_factory=list)
    offload: Optional[str] = None


@dataclass
class GenerationResult:
    """Single generation result."""
    problem_id: str
    seed: int
    rollout: int
    prompt: str
    generation: GenerationOutput
    is_correct: bool
    extracted_answer: Optional[float]


@dataclass
class DiscriminatorScore:
    """Discriminator score for a generation."""
    problem_id: str
    seed: int
    rollout: int
    score: float  # 0-1 confidence
    is_consistent: bool  # Agrees with generator answer
    reasoning: str


@dataclass
class ExperimentResult:
    """Full experiment results."""
    exp_id: str
    config: ExperimentConfig
    generations: List[GenerationResult] = field(default_factory=list)
    discriminator_scores: List[DiscriminatorScore] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================
# Answer Extraction
# ============================================================

def extract_answer(text: str) -> Optional[float]:
    """Extract numerical answer from generation text."""
    # Look for \boxed{...} or "answer is X" or final number
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
# Experiment Runner
# ============================================================

class ExperimentRunner:
    """Runs a single experiment (E1-E9)."""
    
    def __init__(self, config: ExperimentConfig, registry: ModelRegistry = None):
        self.config = config
        self.registry = registry or get_registry()
        self.results = ExperimentResult(exp_id=config.exp_id, config=config)
        
        # Load problems
        self.problems = self._load_problems()
    
    def _load_problems(self) -> List[GSMProblem]:
        """Load or generate problems for this experiment."""
        # Try to load from processed
        problem_path = f"data/processed/{self.config.exp_id}_problems.jsonl"
        if Path(problem_path).exists():
            return load_problems(problem_path)
        
        # Generate from templates
        problems = generate_problem_set(
            templates=CORE_TEMPLATES,
            num_problems=self.config.problems,
            seeds=self.config.seeds,
            nop_types=self.config.nop_types,
        )
        
        # Filter to this experiment's nop_types
        if self.config.nop_types:
            problems = [p for p in problems if p.nop_type in self.config.nop_types or p.nop_type is None]
        
        return problems
    
    def run_generator(self, generator_wrapper, problem: GSMProblem, seed: int, rollout: int) -> GenerationResult:
        """Run single generator rollout."""
        prompt = GENERATOR_PROMPT.format(question=problem.question)
        
        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 512,
            "n": 1,
        }
        
        outputs = generator_wrapper.generate_single(prompt, sampling_params)
        gen = outputs[0]
        
        extracted = extract_answer(gen.text)
        is_correct = check_correctness(extracted, problem.answer)
        
        return GenerationResult(
            problem_id=problem.problem_id,
            seed=seed,
            rollout=rollout,
            prompt=prompt,
            generation=gen,
            is_correct=is_correct,
            extracted_answer=extracted,
        )
    
    def run_discriminator(self, discriminator_wrapper, problem: GSMProblem, generation: GenerationOutput) -> DiscriminatorScore:
        """Run discriminator on a generation."""
        prompt = DISCRIMINATOR_PROMPT.format(
            question=problem.question,
            solution=generation.text
        )
        
        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "n": 1,
        }
        
        outputs = discriminator_wrapper.generate_single(prompt, sampling_params)
        disc_text = outputs[0].text
        
        # Parse verdict
        verdict = "No"
        if "Verdict: Yes" in disc_text or "verdict: yes" in disc_text.lower():
            verdict = "Yes"
        
        is_consistent = (verdict == "Yes")
        
        # Extract confidence (simplified)
        score = 1.0 if is_consistent else 0.0
        
        return DiscriminatorScore(
            problem_id=problem.problem_id,
            seed=0,  # Filled later
            rollout=0,
            score=score,
            is_consistent=is_consistent,
            reasoning=disc_text,
        )
    
    def run_experiment(self) -> ExperimentResult:
        """Run full experiment."""
        print(f"Running {self.config.exp_id}: {self.config.name}")
        print(f"  Generator: {self.config.generator}")
        print(f"  Discriminator: {self.config.discriminator}")
        print(f"  Problems: {len(self.problems)}")
        print(f"  Rollouts: {self.config.rollouts}")
        print(f"  Seeds: {self.config.seeds}")
        
        # Load generator
        generator = self.registry.load(self.config.generator)
        
        # Generate all rollouts
        print("  Generating...")
        all_generations = []
        
        for problem in self.problems:
            for seed in self.config.seeds:
                for rollout in range(self.config.rollouts):
                    gen_result = self.run_generator(generator, problem, seed, rollout)
                    gen_result.seed = seed
                    gen_result.rollout = rollout
                    all_generations.append(gen_result)
        
        self.results.generations = all_generations
        print(f"  Generated {len(all_generations)} completions")
        
        # Unload generator, load discriminator
        self.registry.unload(self.config.generator)
        discriminator = self.registry.load(self.config.discriminator)
        
        # Run discriminator
        print("  Discriminating...")
        all_scores = []
        
        for gen in all_generations:
            # Find corresponding problem
            problem = next(p for p in self.problems if p.problem_id == gen.problem_id)
            score = self.run_discriminator(discriminator, problem, gen.generation)
            score.seed = gen.seed
            score.rollout = gen.rollout
            all_scores.append(score)
        
        self.results.discriminator_scores = all_scores
        print(f"  Scored {len(all_scores)} generations")
        
        # Compute metrics
        self.results.metrics = self._compute_metrics()
        
        # Save results
        self._save_results()
        
        # Unload discriminator
        self.registry.unload(self.config.discriminator)
        
        return self.results
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute experiment metrics."""
        gens = self.results.generations
        scores = self.results.discriminator_scores
        
        # Group by problem_id
        by_problem = {}
        for g in gens:
            if g.problem_id not in by_problem:
                by_problem[g.problem_id] = {"generations": [], "scores": []}
            by_problem[g.problem_id]["generations"].append(g)
        
        for s in scores:
            if s.problem_id in by_problem:
                by_problem[s.problem_id]["scores"].append(s)
        
        # Per-problem metrics
        problem_accuracies = []
        problem_consistencies = []
        mutual_consistencies = []
        
        for pid, data in by_problem.items():
            gen_correct = [g.is_correct for g in data["generations"]]
            disc_consistent = [s.is_consistent for s in data["scores"]]
            
            # Generator accuracy (majority vote)
            gen_acc = sum(gen_correct) / len(gen_correct) if gen_correct else 0
            problem_accuracies.append(gen_acc)
            
            # Discriminator consistency rate
            disc_cons = sum(disc_consistent) / len(disc_consistent) if disc_consistent else 0
            problem_consistencies.append(disc_cons)
            
            # Mutual consistency: generator correct AND discriminator agrees
            mutual = sum(1 for g, s in zip(data["generations"], data["scores"]) 
                        if g.is_correct and s.is_consistent) / len(data["generations"])
            mutual_consistencies.append(mutual)
        
        # Overall metrics
        metrics = {
            "generator_accuracy": np.mean(problem_accuracies),
            "discriminator_consistency_rate": np.mean(problem_consistencies),
            "mutual_consistency_rate": np.mean(mutual_consistencies),
            "num_problems": len(by_problem),
            "num_generations": len(gens),
            "num_discriminator_scores": len(scores),
            "per_problem": {
                "accuracies": problem_accuracies,
                "consistencies": problem_consistencies,
                "mutual_consistencies": mutual_consistencies,
            }
        }
        
        # NoOp-type breakdown
        nop_types = set(p.nop_type for p in self.problems if p.nop_type)
        for nt in nop_types:
            nt_problems = [pid for pid, data in by_problem.items() 
                          if any(p.nop_type == nt for p in self.problems if p.problem_id == pid)]
            if nt_problems:
                nt_acc = np.mean([by_problem[pid]["generations"][0].is_correct for pid in nt_problems])
                metrics[f"accuracy_{nt}"] = float(nt_acc)
        
        return metrics
    
    def _save_results(self) -> None:
        """Save experiment results."""
        output_dir = Path(f"experiments/results/{self.config.exp_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save generations
        with open(output_dir / "generations.jsonl", "w") as f:
            for g in self.results.generations:
                f.write(json.dumps({
                    "problem_id": g.problem_id,
                    "seed": g.seed,
                    "rollout": g.rollout,
                    "prompt": g.prompt,
                    "generation": g.generation.text,
                    "tokens": g.generation.tokens,
                    "extracted_answer": g.extracted_answer,
                    "is_correct": g.is_correct,
                }) + "\n")
        
        # Save discriminator scores
        with open(output_dir / "discriminator_scores.jsonl", "w") as f:
            for s in self.results.discriminator_scores:
                f.write(json.dumps({
                    "problem_id": s.problem_id,
                    "seed": s.seed,
                    "rollout": s.rollout,
                    "score": s.score,
                    "is_consistent": s.is_consistent,
                    "reasoning": s.reasoning,
                }) + "\n")
        
        # Save metrics
        with open(output_dir / "metrics.yaml", "w") as f:
            yaml.dump(self.results.metrics, f)
        
        # Save config snapshot
        config_dict = {
            "exp_id": self.config.exp_id,
            "name": self.config.name,
            "generator": self.config.generator,
            "discriminator": self.config.discriminator,
            "nop_types": self.config.nop_types,
            "seeds": self.config.seeds,
            "rollouts": self.config.rollouts,
            "problems": self.config.problems,
            "claim": self.config.claim,
        }
        with open(output_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(config_dict, f)
        
        print(f"  ✓ Results saved to {output_dir}")


def load_experiment_config(exp_id: str, config_path: str = "configs/experiment_matrix.yaml") -> ExperimentConfig:
    """Load experiment config from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for exp in config.get("experiments", []):
        if exp["id"] == exp_id:
            return ExperimentConfig(**exp)
    
    raise ValueError(f"Experiment {exp_id} not found in {config_path}")


def run_experiment(exp_id: str) -> ExperimentResult:
    """Main entry point to run an experiment."""
    config = load_experiment_config(exp_id)
    runner = ExperimentRunner(config)
    return runner.run_experiment()


if __name__ == "__main__":
    import sys
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "E1"
    result = run_experiment(exp_id)
    print(f"\n=== {exp_id} Complete ===")
    print(f"Generator Accuracy: {result.metrics['generator_accuracy']:.3f}")
    print(f"Discriminator Consistency: {result.metrics['discriminator_consistency_rate']:.3f}")
    print(f"Mutual Consistency: {result.metrics['mutual_consistency_rate']:.3f}")