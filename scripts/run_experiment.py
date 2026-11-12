#!/usr/bin/env python3
# MC-IRF Main Entry Point
# Usage: python scripts/run_experiment.py --exp E1

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import run_experiment, list_experiments, get_tier_experiments


def main():
    parser = argparse.ArgumentParser(description="MC-IRF Experiment Runner")
    parser.add_argument("--exp", type=str, help="Experiment ID (e.g., E1, E2, ...)")
    parser.add_argument("--tier", type=str, choices=["tier1_core", "tier2_mechanism", "tier3_scale", "all"], 
                        help="Run all experiments in a tier")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    parser.add_argument("--list-tier", type=str, help="List experiments in a tier")
    
    args = parser.parse_args()
    
    if args.list:
        print("All experiments:")
        for exp in list_experiments():
            print(f"  {exp['id']}: {exp['name']}")
            print(f"    Gen: {exp['generator']} | Disc: {exp['discriminator']}")
            print(f"    Claim: {exp['claim']}")
        return
    
    if args.list_tier:
        exps = get_tier_experiments(args.list_tier)
        print(f"Experiments in {args.list_tier}:")
        for exp_id in exps:
            print(f"  {exp_id}")
        return
    
    if args.tier:
        exp_ids = get_tier_experiments(args.tier)
        print(f"Running tier {args.tier}: {exp_ids}")
        for exp_id in exp_ids:
            print(f"\n{'='*60}")
            print(f"Running {exp_id}")
            print(f"{'='*60}")
            try:
                result = run_experiment(exp_id)
                print(f"✓ {exp_id} complete")
                print(f"  Generator Accuracy: {result.metrics['generator_accuracy']:.3f}")
                print(f"  Mutual Consistency: {result.metrics['mutual_consistency_rate']:.3f}")
            except Exception as e:
                print(f"✗ {exp_id} failed: {e}")
        return
    
    if args.exp:
        print(f"Running {args.exp}...")
        try:
            result = run_experiment(args.exp)
            print(f"\n=== {args.exp} Complete ===")
            print(f"Generator Accuracy: {result.metrics['generator_accuracy']:.3f}")
            print(f"Discriminator Consistency: {result.metrics['discriminator_consistency_rate']:.3f}")
            print(f"Mutual Consistency: {result.metrics['mutual_consistency_rate']:.3f}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()