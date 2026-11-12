"""
MC-IRF Experiment Checkpointing
Resumable state machine for JupyterLab execution.
"""

import pickle
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import torch
import numpy as np


@dataclass
class CheckpointData:
    """Complete experiment state for resumability."""
    exp_id: str
    config: Dict
    phase: str  # "generator", "discriminator", "complete"
    current_problem_idx: int
    current_seed_idx: int
    current_rollout: int
    
    generations: List[Dict] = field(default_factory=list)
    discriminator_scores: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    
    rng_state_python: Any = None
    rng_state_numpy: Any = None
    rng_state_torch: Any = None
    rng_state_cuda: Any = None
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"


class ExperimentStateMachine:
    """
    Resumable experiment execution with full state persistence.
    
    Designed for JupyterLab where kernel restarts kill processes.
    Checkpoints every N problems, saves RNG state for exact reproducibility.
    """
    
    def __init__(self, exp_id: str, config: Dict, checkpoint_dir: str = "experiments/checkpoints_h200"):
        self.exp_id = exp_id
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_path = self.checkpoint_dir / f"{exp_id}_state.pkl"
        self.results_dir = Path(f"experiments/results_h200/{exp_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.phase = "generator"
        self.current_problem_idx = 0
        self.current_seed_idx = 0
        self.current_rollout = 0
        
        self.generations = []
        self.discriminator_scores = []
        self.metrics = {}
        
        self.problems = []
        self.problem_order = []  # List of (problem_idx, seed_idx, rollout)
    
    def save_rng_state(self):
        """Capture all RNG states for exact reproducibility."""
        import random
        self.rng_state_python = random.getstate()
        self.rng_state_numpy = np.random.get_state()
        self.rng_state_torch = torch.get_rng_state()
        if torch.cuda.is_available():
            self.rng_state_cuda = torch.cuda.get_rng_state_all()
    
    def restore_rng_state(self):
        """Restore all RNG states."""
        import random
        if self.rng_state_python:
            random.setstate(self.rng_state_python)
        if self.rng_state_numpy:
            np.random.set_state(self.rng_state_numpy)
        if self.rng_state_torch:
            torch.set_rng_state(self.rng_state_torch)
        if self.rng_state_cuda and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.rng_state_cuda)
    
    def build_problem_order(self, problems: List, seeds: List[int], rollouts: int):
        """Build deterministic problem order for resumability."""
        self.problems = problems
        self.problem_order = []
        for p_idx, problem in enumerate(problems):
            for s_idx, seed in enumerate(seeds):
                for r in range(rollouts):
                    self.problem_order.append((p_idx, s_idx, r))
    
    def get_next_task(self) -> Optional[tuple]:
        """Get next (problem, seed, rollout) to process."""
        if self.current_problem_idx >= len(self.problem_order):
            return None
        
        p_idx, s_idx, r = self.problem_order[self.current_problem_idx]
        return (self.problems[p_idx], 
                self.config["seeds"][s_idx], 
                r)
    
    def advance(self):
        """Advance to next task."""
        self.current_problem_idx += 1
        if self.current_problem_idx >= len(self.problem_order):
            self.phase = "discriminator"
            self.current_problem_idx = 0
    
    def add_generation(self, gen_data: Dict):
        """Add a generation result."""
        self.generations.append(gen_data)
    
    def add_discriminator_score(self, score_data: Dict):
        """Add a discriminator score."""
        self.discriminator_scores.append(score_data)
    
    def checkpoint(self, force: bool = False):
        """Save checkpoint to disk."""
        # Save every 5 problems or on phase change or force
        should_save = (force or 
                       self.current_problem_idx % 5 == 0 or
                       self.phase == "discriminator" or
                       self.current_problem_idx >= len(self.problem_order))
        
        if not should_save:
            return
        
        self.save_rng_state()
        
        data = CheckpointData(
            exp_id=self.exp_id,
            config=self.config,
            phase=self.phase,
            current_problem_idx=self.current_problem_idx,
            current_seed_idx=self.current_seed_idx,
            current_rollout=self.current_rollout,
            generations=self.generations,
            discriminator_scores=self.discriminator_scores,
            metrics=self.metrics,
            rng_state_python=self.rng_state_python,
            rng_state_numpy=self.rng_state_numpy,
            rng_state_torch=self.rng_state_torch,
            rng_state_cuda=self.rng_state_cuda,
        )
        
        with open(self.state_path, "wb") as f:
            pickle.dump(data, f)
        
        # Also save human-readable JSON
        json_path = self.checkpoint_dir / f"{self.exp_id}_state.json"
        with open(json_path, "w") as f:
            json.dump({
                "exp_id": self.exp_id,
                "phase": self.phase,
                "progress": f"{self.current_problem_idx}/{len(self.problem_order)}",
                "generations": len(self.generations),
                "discriminator_scores": len(self.discriminator_scores),
                "timestamp": data.timestamp,
            }, f, indent=2)
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        if not self.state_path.exists():
            return False
        
        print(f"Loading checkpoint from {self.state_path}")
        with open(self.state_path, "rb") as f:
            data = pickle.load(f)
        
        # Verify config matches
        if data.config != self.config:
            print("WARNING: Config mismatch with checkpoint!")
        
        self.phase = data.phase
        self.current_problem_idx = data.current_problem_idx
        self.current_seed_idx = data.current_seed_idx
        self.current_rollout = data.current_rollout
        self.generations = data.generations
        self.discriminator_scores = data.discriminator_scores
        self.metrics = data.metrics
        self.rng_state_python = data.rng_state_python
        self.rng_state_numpy = data.rng_state_numpy
        self.rng_state_torch = data.rng_state_torch
        self.rng_state_cuda = data.rng_state_cuda
        
        # Restore RNG state
        self.restore_rng_state()
        
        print(f"Resumed: phase={self.phase}, progress={self.current_problem_idx}/{len(self.problem_order)}")
        return True
    
    def save_final_results(self):
        """Save final results in standard format."""
        # Save generations
        with open(self.results_dir / "generations.jsonl", "w") as f:
            for g in self.generations:
                f.write(json.dumps(g) + "\n")
        
        # Save discriminator scores
        with open(self.results_dir / "discriminator_scores.jsonl", "w") as f:
            for s in self.discriminator_scores:
                f.write(json.dumps(s) + "\n")
        
        # Save metrics
        with open(self.results_dir / "metrics.yaml", "w") as f:
            yaml.dump(self.metrics, f)
        
        # Save config snapshot
        with open(self.results_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(self.config, f)
        
        print(f"Final results saved to {self.results_dir}")
    
    def get_progress(self) -> Dict:
        """Get current progress info."""
        total = len(self.problem_order)
        if self.phase == "generator":
            done = self.current_problem_idx
        else:
            done = total + self.current_problem_idx
        total_all = total * (2 if self.config.get("discriminator") else 1)
        
        return {
            "phase": self.phase,
            "current": self.current_problem_idx,
            "total": total,
            "percent": done / total_all * 100 if total_all > 0 else 100,
            "generations": len(self.generations),
            "scores": len(self.discriminator_scores),
        }


def load_experiment_state(exp_id: str, checkpoint_dir: str = "experiments/checkpoints_h200") -> Optional[CheckpointData]:
    """Load experiment state from checkpoint."""
    path = Path(checkpoint_dir) / f"{exp_id}_state.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def list_checkpoints(checkpoint_dir: str = "experiments/checkpoints_h200") -> List[Dict]:
    """List all available checkpoints."""
    checkpoints = []
    for path in Path(checkpoint_dir).glob("*_state.pkl"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            checkpoints.append({
                "exp_id": data.exp_id,
                "phase": data.phase,
                "timestamp": data.timestamp,
                "generations": len(data.generations),
                "scores": len(data.discriminator_scores),
            })
        except:
            pass
    return checkpoints