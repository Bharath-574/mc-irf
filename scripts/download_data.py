#!/usr/bin/env python3
# MC-IRF Data Download & Generation Script
# Generates GSM-Symbolic problems + NoOp perturbations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.gsm_symbolic import (
    generate_problem_set, 
    save_problems, 
    CORE_TEMPLATES,
    NOOP_TEMPLATES,
    IRRELEVANT_POOLS,
)
from experiments.registry import EXPERIMENTS, TIERS
from pathlib import Path
import json


def main():
    print("=" * 60)
    print("MC-IRF Data Generation")
    print("=" * 60)
    
    # 1. Show available templates
    print(f"\n📋 Core GSM-Symbolic templates: {len(CORE_TEMPLATES)}")
    for t in CORE_TEMPLATES:
        print(f"  {t.template_id}: {t.question_template[:80]}...")
    
    print(f"\n🎯 NoOp types: {list(NOOP_TEMPLATES.keys())}")
    for nt, templates in NOOP_TEMPLATES.items():
        print(f"  {nt}: {len(templates)} templates")
    
    # 2. Generate problems for all experiments
    print("\n🔄 Generating problem sets for all experiments...")
    
    all_problems = []
    
    for exp in EXPERIMENTS:
        exp_id = exp["id"]
        print(f"\n  Generating for {exp_id}: {exp['name']}")
        
        problems = generate_problem_set(
            templates=CORE_TEMPLATES,
            num_problems=exp["problems"],
            seeds=exp["seeds"],
            nop_types=exp["nop_types"],
        )
        
        # Filter to this experiment's nop_types
        if exp["nop_types"]:
            filtered = [p for p in problems if p.nop_type in exp["nop_types"] or p.nop_type is None]
        else:
            filtered = [p for p in problems if p.nop_type is None]
        
        # Save per-experiment
        output_path = f"data/processed/{exp_id}_problems.jsonl"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_problems(filtered, output_path)
        
        print(f"    Problems: {len(filtered)}")
        print(f"    Saved to: {output_path}")
        
        all_problems.extend(filtered)
    
    # 3. Save combined master file
    print("\n💾 Saving master problem set...")
    master_path = "data/processed/all_problems.jsonl"
    save_problems(all_problems, master_path)
    print(f"  Total problems: {len(all_problems)}")
    print(f"  Saved to: {master_path}")
    
    # 4. Summary statistics
    print("\n📊 Summary:")
    print(f"  Total unique problems: {len(all_problems)}")
    
    by_type = {}
    for p in all_problems:
        nt = p.nop_type or "clean"
        by_type[nt] = by_type.get(nt, 0) + 1
    
    for nt, count in sorted(by_type.items()):
        print(f"  {nt}: {count}")
    
    # 5. Verify a few samples
    print("\n🔍 Sample problems:")
    for p in all_problems[:3]:
        print(f"\n  {p.problem_id}")
        print(f"  Q: {p.question[:150]}...")
        print(f"  A: {p.answer}")
        if p.nop_type:
            print(f"  NoOp ({p.nop_type}): {p.nop_text[:100]}...")
    
    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    main()