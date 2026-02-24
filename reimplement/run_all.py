#!/usr/bin/env python3
"""
Main orchestration script for PtychoNet baseline training and evaluation.
Trains MSE and DSE variants at all three overlap conditions.
Designed to fit within 2-hour compute budget.
"""
import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train_ptychonet, evaluate, plot_training_curves
from model import PtychoNet
from forward_model import create_gaussian_probe, create_scan_positions, compute_object_size

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(SAVE_DIR, exist_ok=True)

# Time budget management
START_TIME = time.time()
MAX_TIME = 50 * 60  # 50 minutes for baseline (leave time for improved method)


def time_remaining():
    return MAX_TIME - (time.time() - START_TIME)


def should_continue():
    return time_remaining() > 120  # Keep 2 min buffer


def main():
    results = {}

    # Training configs - prioritize sparse condition, train all conditions
    # Each config: (step_size, loss_type, max_steps)
    configs = [
        # MSE loss first (simpler, faster)
        (60, 'mse', 2000),   # Sparse - hardest, most important
        (40, 'mse', 1500),   # Moderate
        (20, 'mse', 1500),   # Dense
        # DSE loss (includes DFT constraint)
        (60, 'dse', 2000),
        (40, 'dse', 1500),
        (20, 'dse', 1500),
    ]

    for step_size, loss_type, max_steps in configs:
        if not should_continue():
            print(f"\nTime limit approaching, skipping step_size={step_size}, loss={loss_type}")
            continue

        try:
            model, metrics, logs = train_ptychonet(
                step_size=step_size,
                loss_type=loss_type,
                epochs=3,
                batch_size=16,
                lr=2e-4,
                nf=64,
                save_dir=SAVE_DIR,
                probe_sigma=30.0,
                num_workers=4,
                eval_every=400,
                max_steps=max_steps,
            )

            key = f'step{step_size}_{loss_type}'
            results[key] = metrics
            print(f"\n{'='*60}")
            print(f"Results for {key}:")
            print(f"  Amp NRMSE: {metrics['amp_nrmse_mean']:.4f} ± {metrics['amp_nrmse_std']:.4f}")
            print(f"  Phase NRMSE: {metrics['phase_nrmse_mean']:.4f} ± {metrics['phase_nrmse_std']:.4f}")
            print(f"  Time elapsed: {(time.time() - START_TIME)/60:.1f} min")
            print(f"{'='*60}")

            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error training step_size={step_size}, loss={loss_type}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary
    with open(os.path.join(SAVE_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Config':<20} {'Amp NRMSE':<20} {'Phase NRMSE':<20}")
    print("-" * 60)
    for key, metrics in results.items():
        print(f"{key:<20} {metrics['amp_nrmse_mean']:.4f} ± {metrics['amp_nrmse_std']:.4f}  "
              f"{metrics['phase_nrmse_mean']:.4f} ± {metrics['phase_nrmse_std']:.4f}")

    total_time = time.time() - START_TIME
    print(f"\nTotal time: {total_time/60:.1f} minutes")

    return results


if __name__ == '__main__':
    main()
