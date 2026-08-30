#!/usr/bin/env python3
"""Collect and aggregate fold metrics from 100_subject_adaptation experiments."""

import json
import os
import numpy as np
from pathlib import Path

def load_fold_metrics(base_dir):
    """Load metrics from all 10 folds for a given adaptation mode."""
    fold_metrics = []
    for fold_num in range(1, 11):
        fold_dir = os.path.join(base_dir, f"fold_{fold_num:02d}")
        metrics_file = os.path.join(fold_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                fold_metrics.append(data)
    return fold_metrics

def compute_stats(values):
    """Compute mean and std for a list of values."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr))
    }

def aggregate_metrics(fold_data_list):
    """Aggregate metrics across all folds."""
    # Overall metrics
    overall_keys = ['accuracy', 'kappa', 'macro_f1', 'weighted_f1', 'mgm']
    overall_stats = {}
    
    for key in overall_keys:
        values = []
        for fold_data in fold_data_list:
            # Try test_metrics first, then top-level
            if 'test_metrics' in fold_data and key in fold_data['test_metrics']:
                values.append(fold_data['test_metrics'][key])
            elif key in fold_data:
                values.append(fold_data[key])
        if values:
            overall_stats[key] = compute_stats(values)
    
    # Per-class metrics
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    per_class_stats = {}
    
    for class_name in class_names:
        class_metrics = {}
        for metric_key in ['precision', 'recall', 'f1']:
            values = []
            for fold_data in fold_data_list:
                # Try test_metrics first, then top-level
                per_class = None
                if 'test_metrics' in fold_data and 'per_class' in fold_data['test_metrics']:
                    per_class = fold_data['test_metrics']['per_class']
                elif 'per_class' in fold_data:
                    per_class = fold_data['per_class']
                
                if per_class and class_name in per_class and metric_key in per_class[class_name]:
                    values.append(per_class[class_name][metric_key])
            if values:
                class_metrics[metric_key] = compute_stats(values)
        per_class_stats[class_name] = class_metrics
    
    return {
        "overall": overall_stats,
        "per_class": per_class_stats
    }

def main():
    base_dir = Path("/home/shamique/projects/sleep/results/100_subject_adaptation")
    
    # Define the four adaptation modes
    modes = {
        "frozen": base_dir / "frozen",
        "lora_head": base_dir / "lora_r8_head",
        "lora_cnn_head": base_dir / "lora_r8_enc.0.pw_enc.1.pw_head",
        "full_finetune": base_dir / "full_finetune"
    }
    
    all_results = {}
    
    for mode_name, mode_dir in modes.items():
        print(f"Processing {mode_name}...")
        fold_data = load_fold_metrics(mode_dir)
        if fold_data:
            stats = aggregate_metrics(fold_data)
            all_results[mode_name] = stats
            
            # Also store raw fold data
            all_results[mode_name]["raw_folds"] = []
            for fold_data_item in fold_data:
                fold_info = {
                    "fold": fold_data_item.get("fold"),
                    "accuracy": fold_data_item.get("accuracy") or fold_data_item.get("test_metrics", {}).get("accuracy"),
                    "kappa": fold_data_item.get("kappa") or fold_data_item.get("test_metrics", {}).get("kappa"),
                    "macro_f1": fold_data_item.get("macro_f1") or fold_data_item.get("test_metrics", {}).get("macro_f1"),
                    "weighted_f1": fold_data_item.get("weighted_f1") or fold_data_item.get("test_metrics", {}).get("weighted_f1"),
                    "mgm": fold_data_item.get("mgm") or fold_data_item.get("test_metrics", {}).get("mgm"),
                    "per_class": fold_data_item.get("per_class") or fold_data_item.get("test_metrics", {}).get("per_class", {})
                }
                all_results[mode_name]["raw_folds"].append(fold_info)
    
    # Save the comprehensive results
    output_file = base_dir / "comprehensive_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComprehensive metrics saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for mode_name, stats in all_results.items():
        if mode_name == "raw_folds":
            continue
        print(f"\n{mode_name.upper()}")
        print("-"*40)
        print("Overall Metrics:")
        for metric, values in stats["overall"].items():
            print(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
        
        print("\nPer-class Metrics:")
        for class_name, metrics in stats["per_class"].items():
            print(f"  {class_name}:")
            for metric_key, values in metrics.items():
                print(f"    {metric_key}: {values['mean']:.4f} ± {values['std']:.4f}")

if __name__ == "__main__":
    main()