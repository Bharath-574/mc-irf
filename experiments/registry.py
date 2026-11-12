# MC-IRF Experiment Registry
# Defines all experiments E1-E9

EXPERIMENTS = [
    # Tier 1: Core Claims (run first)
    {
        "id": "E1",
        "name": "NoOp Persistence Baseline",
        "generator": "phi-3.5-mini",
        "discriminator": "phi-3.5-mini",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "claim": "C1: NoOp attacks persist across model scales without defense",
    },
    {
        "id": "E2",
        "name": "Mutual Consistency Defense (Homogeneous)",
        "generator": "phi-3.5-mini",
        "discriminator": "phi-3.5-mini",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "claim": "C2: Mutual consistency defends against NoOp",
    },
    {
        "id": "E5",
        "name": "Heterogeneous Generator-Discriminator",
        "generator": "phi-3.5-mini",
        "discriminator": "qwen2.5-7b",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "claim": "C3: Defense gain ∝ 1/error_correlation (error independence law)",
    },
    {
        "id": "E6",
        "name": "Discriminator Scale Advantage",
        "generator": "phi-3.5-mini",
        "discriminator": "qwen2.5-14b",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "claim": "C4: Discriminator scale > Generator scale for defense",
    },
    
    # Tier 2: Mechanism Validation
    {
        "id": "E3",
        "name": "Attention Leakage Analysis",
        "generator": "phi-3.5-mini",
        "discriminator": "phi-3.5-mini",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "tracing": "attention",
        "claim": "C5: Attention leaks to NoOp tokens (similarity ≠ causal relevance)",
    },
    {
        "id": "E4",
        "name": "Causal Tracing Localization",
        "generator": "phi-3.5-mini",
        "discriminator": "phi-3.5-mini",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "tracing": "causal",
        "claim": "C6: Causal tracing localizes NoOp susceptibility to specific layers/heads",
    },
    {
        "id": "E8",
        "name": "Self-Consistency vs Mutual Consistency",
        "generator": "phi-3.5-mini",
        "discriminator": "phi-3.5-mini",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "baselines": ["self-consistency", "majority-vote"],
        "claim": "C8: Mutual consistency > Self-consistency for adversarial irrelevance",
    },
    
    # Tier 3: Scale Ladder (H200 multi-slice)
    {
        "id": "E7",
        "name": "Generator Scale Ladder",
        "generator": "phi-3.5-mini",  # First in ladder
        "discriminator": "qwen2.5-14b",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "generators": ["phi-3.5-mini", "llama-3.1-8b", "qwen2.5-14b", "qwen2.5-32b"],
        "claim": "C7: Generator scale helps but cannot close gap without discriminator",
    },
    {
        "id": "E9",
        "name": "70B Discriminator (CPU Offload)",
        "generator": "llama-3.1-8b",
        "discriminator": "llama-3.1-70b",
        "nop_types": ["quantity", "comparison", "filter", "discount"],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "offload": "cpu",
        "claim": "C9: 70B discriminator saturates defense; error independence plateaus",
    },
    {
        "id": "E9_clean",
        "name": "70B Clean Baseline (No NoOp)",
        "generator": "llama-3.1-8b",
        "discriminator": "llama-3.1-70b",
        "nop_types": [],
        "seeds": [42, 123, 456],
        "rollouts": 32,
        "problems": 20,
        "offload": "cpu",
        "claim": "Upper bound: mutual consistency on clean GSM-Symbolic",
    },
]

# Experiment tiers for execution order
TIERS = {
    "tier1_core": ["E1", "E2", "E5", "E6"],
    "tier2_mechanism": ["E3", "E4", "E8"],
    "tier3_scale": ["E7", "E9", "E9_clean"],
}

def get_experiment(exp_id: str) -> dict:
    """Get experiment config by ID."""
    for exp in EXPERIMENTS:
        if exp["id"] == exp_id:
            return exp
    raise ValueError(f"Unknown experiment: {exp_id}")

def get_tier_experiments(tier: str) -> List[str]:
    """Get experiment IDs for a tier."""
    return TIERS.get(tier, [])

def list_experiments() -> List[dict]:
    """List all experiments."""
    return EXPERIMENTS