# mc_irf/experiments/checkpoint.py
class ExperimentStateMachine:
    """Resumable experiment execution with full state persistence."""
    
    def __init__(self, exp_id: str, config: ExperimentConfig):
        self.exp_id = exp_id
        self.config = config
        self.state_path = Path(f"experiments/checkpoints/{exp_id}_state.pkl")
        self.results_path = Path(f"experiments/results/{exp_id}")
        
    def run_generator_phase(self, generator_llm) -> GenerationBatch:
        """Run generator on next batch, checkpoint after each problem."""
        
    def run_discriminator_phase(self, discriminator_llm) -> DiscriminatorBatch:
        """Run discriminator on completed generations."""
        
    def checkpoint(self):
        """Save full state: generations, scores, metrics, RNG state."""
        
    def resume(self):
        """Restore from latest checkpoint."""
        
    def compute_mcrf_metrics(self) -> MCRFMetrics:
        """Compute ρ, β, defense_gain with bootstrap CI."""
