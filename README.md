# MC-IRF: Mutual Consistency as Implicit Relevance Filter

> **Defending Against GSM-NoOp Adversarial Attacks via Heterogeneous Generator–Discriminator Verification**

Target venues: **ICLR / NeurIPS / ICML 2026-2027**

## Core Thesis

**Mutual consistency (rStar) acts as an implicit relevance filter** against GSM-NoOp adversarial irrelevant-information attacks. The defense gain scales with **error independence** between generator and discriminator — heterogeneous pairs outperform homogeneous ones.

## Theoretical Contributions

| Claim | Description |
|-------|-------------|
| **C1** | NoOp attacks persist across model scales without explicit defense |
| **C2** | Mutual consistency provides non-trivial defense against NoOp |
| **C3** | Defense gain ∝ 1/error_correlation (quantitative law) |
| **C4** | Discriminator scale > Generator scale for defense |
| **C5** | Attention leaks to NoOp tokens (similarity ≠ causal relevance) |
| **C6** | Causal tracing localizes NoOp susceptibility to specific layers/heads |
| **C7** | Generator scale helps but cannot close gap without discriminator |
| **C8** | Mutual consistency > Self-consistency for adversarial irrelevance |
| **C9** | 70B discriminator saturates defense; error independence plateaus |

## Experiment Matrix

| Tier | Experiments | Models | Hardware |
|------|-------------|--------|----------|
| **Tier 1** (Core) | E1, E2, E5, E6 | Phi-3-mini, Qwen2.5-7B/14B | Local RTX 3050 (llama.cpp GGUF) |
| **Tier 2** (Mechanism) | E3, E4, E8 | Phi-3-mini | Local RTX 3050 + Attention/Causal Tracing |
| **Tier 3** (Scale) | E7, E9, E9_clean | LLaMA3-8B, Qwen2.5-14B/32B, LLaMA3-70B | DGX H200 (9 MIG slices) |

**Total**: 9 experiments × 20 problems × 3 seeds × 32 rollouts = ~1,728 generations/experiment

## Quick Start

```bash
# Local validation (RTX 3050 6GB)
uv sync
python scripts/download_models.py --models phi3-mini,qwen2.5-7b
python scripts/run_experiment.py --exp E1 --local

# H200 deployment
python scripts/deploy_h200.py --exp E7,E9 --parallel 8
```

## Repository Structure

```
mc-irf/
├── configs/           # Experiment matrix, model configs, hardware specs
├── data/              # GSM-Symbolic templates, NoOp perturbations
├── experiments/       # Experiment runners, result aggregation
├── models/            # Model wrappers (llama.cpp, vLLM, HF)
├── analysis/          # Attention tracing, causal tracing, metrics
├── scripts/           # Download, deploy, eval, plotting
└── docs/              # Paper draft, experiment logs, theorems
```

## Hardware Strategy

| Environment | GPU | Backend | Models |
|-------------|-----|---------|--------|
| **Local** | RTX 3050 6GB | llama.cpp GGUF | Phi-3-mini (2.5GB), Qwen2.5-7B (4.5GB) |
| **DGX H200** | 8× H200 (18GB MIG) | vLLM / llama.cpp | Up to Qwen2.5-32B, LLaMA3-70B (CPU offload) |

**Local → H200 Pipeline**: Validate on RTX 3050 sequentially (gen→disc via shared FS), then scale to H200 with 8-way parallel.

## Citation

```bibtex
@misc{mc-irf2026,
  title={Mutual Consistency as Implicit Relevance Filter: Defending Against GSM-NoOp Adversarial Attacks via Heterogeneous Generator–Discriminator Verification},
  author={Bharath, P.},
  year={2026},
  note={Under review at ICLR/NeurIPS/ICML}
}
```

## License

MIT (core framework) — Enterprise components private.