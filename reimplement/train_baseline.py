#!/usr/bin/env python3
"""
Efficient training script for PtychoNet baseline.
Trains MSE and DSE variants at all three overlap conditions.
Time budget: ~45 minutes for all 6 configs.
"""
import os
import sys
import json
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import PtychoNet
from forward_model import (
    create_gaussian_probe, create_scan_positions,
    compute_object_size, simulate_ptychography_from_complex
)
from dataset import PtychographyDataset

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_nrmse(pred, target):
    """Normalized Root Mean Square Error per sample."""
    mse = torch.mean((pred - target) ** 2, dim=(-2, -1))
    norm = torch.mean(target ** 2, dim=(-2, -1))
    return torch.sqrt(mse / (norm + 1e-10))


def correct_global_phase(pred_phase, gt_phase):
    """Correct global phase shift: c = mean(pred - gt)."""
    diff = pred_phase - gt_phase
    c = diff.mean(dim=(-2, -1), keepdim=True)
    return pred_phase - c


def compute_dft_loss(output, probe, positions, diff_amps_target):
    """DFT loss: consistency in Fourier space."""
    B = output.shape[0]
    h, w = probe.shape[-2], probe.shape[-1]
    device = output.device

    pred_amp = output[:, 0].double()
    pred_phase = output[:, 1].double()
    obj_complex = pred_amp * torch.exp(1j * pred_phase)

    probe_dev = probe.to(device)
    if probe_dev.ndim == 2:
        probe_dev = probe_dev.unsqueeze(0)

    total_loss = torch.tensor(0.0, device=device)
    N = len(positions)

    for j in range(N):
        top, left = positions[j]
        patch = obj_complex[:, top:top + h, left:left + w]
        exit_wave = patch * probe_dev
        ft = torch.fft.fft2(exit_wave, norm='ortho')
        pred_diff_amp = torch.abs(ft).float()
        target = diff_amps_target[:, j]
        total_loss = total_loss + torch.mean((pred_diff_amp - target) ** 2)

    return total_loss / N


def evaluate_model(model, test_loader, positions, object_size, device, max_batches=None):
    """Full evaluation returning per-sample NRMSE."""
    model.eval()
    all_amp_nrmse = []
    all_phase_nrmse = []

    with torch.no_grad():
        for batch_idx, (diff_amps, gt_amp, gt_phase) in enumerate(test_loader):
            if max_batches and batch_idx >= max_batches:
                break

            diff_amps = diff_amps.to(device)
            gt_amp = gt_amp.to(device)
            gt_phase = gt_phase.to(device)

            output = model(diff_amps, positions, object_size)
            pred_amp = output[:, 0]
            pred_phase = output[:, 1]

            pred_phase_corr = correct_global_phase(pred_phase, gt_phase)

            all_amp_nrmse.append(compute_nrmse(pred_amp, gt_amp).cpu())
            all_phase_nrmse.append(compute_nrmse(pred_phase_corr, gt_phase).cpu())

    all_amp_nrmse = torch.cat(all_amp_nrmse)
    all_phase_nrmse = torch.cat(all_phase_nrmse)

    return {
        'amp_nrmse_mean': all_amp_nrmse.mean().item(),
        'amp_nrmse_std': all_amp_nrmse.std().item(),
        'phase_nrmse_mean': all_phase_nrmse.mean().item(),
        'phase_nrmse_std': all_phase_nrmse.std().item(),
    }


def train_single_config(step_size, loss_type, hf_dataset_train, hf_dataset_test,
                         max_steps=3000, batch_size=32, lr=2e-4, nf=64,
                         num_workers=4, eval_every=500, probe_sigma=30.0):
    """Train one configuration."""
    device = torch.device('cuda')
    probe_size = 128
    object_size = compute_object_size(probe_size, step_size, n_steps=6)
    probe = create_gaussian_probe(probe_size, sigma=probe_sigma)
    positions = create_scan_positions(object_size, probe_size, step_size)

    config_name = f'step{step_size}_{loss_type}'
    print(f"\n{'='*70}")
    print(f"Training {config_name}: obj_size={object_size}, n_pos={len(positions)}, max_steps={max_steps}")
    print(f"{'='*70}")

    train_dataset = PtychographyDataset(
        hf_dataset_train, probe, positions,
        probe_size=probe_size, object_size=object_size
    )
    test_dataset = PtychographyDataset(
        hf_dataset_test, probe, positions,
        probe_size=probe_size, object_size=object_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0
    )

    model = PtychoNet(nf=nf).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.01)

    train_losses = []
    val_metrics_log = []
    best_val = float('inf')
    global_step = 0
    start_time = time.time()

    while global_step < max_steps:
        model.train()
        for diff_amps, gt_amp, gt_phase in train_loader:
            if global_step >= max_steps:
                break

            diff_amps = diff_amps.to(device, non_blocking=True)
            gt_amp = gt_amp.to(device, non_blocking=True)
            gt_phase = gt_phase.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            output = model(diff_amps, positions, object_size)
            pred_amp = output[:, 0]
            pred_phase = output[:, 1]

            mse_loss = nn.functional.mse_loss(pred_amp, gt_amp) + \
                       nn.functional.mse_loss(pred_phase, gt_phase)

            if loss_type == 'dse':
                dft_loss = compute_dft_loss(output, probe, positions, diff_amps)
                loss = mse_loss + dft_loss
            else:
                loss = mse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            train_losses.append({'step': global_step, 'loss': loss.item(),
                                  'time': time.time() - start_time})

            if global_step % 100 == 0:
                print(f"  Step {global_step}/{max_steps}, loss={loss.item():.6f}, "
                      f"lr={scheduler.get_last_lr()[0]:.6f}, "
                      f"time={time.time()-start_time:.0f}s")

            if global_step % eval_every == 0 or global_step == max_steps:
                metrics = evaluate_model(model, test_loader, positions, object_size,
                                          device, max_batches=20)
                metrics['step'] = global_step
                metrics['time'] = time.time() - start_time
                val_metrics_log.append(metrics)
                print(f"  >> Val: Amp NRMSE={metrics['amp_nrmse_mean']:.4f}±{metrics['amp_nrmse_std']:.4f}, "
                      f"Phase NRMSE={metrics['phase_nrmse_mean']:.4f}±{metrics['phase_nrmse_std']:.4f}")

                val_score = metrics['amp_nrmse_mean'] + metrics['phase_nrmse_mean']
                if val_score < best_val:
                    best_val = val_score
                    torch.save(model.state_dict(),
                               os.path.join(SAVE_DIR, f'best_{config_name}.pt'))
                model.train()

    # Load best model for final eval
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'best_{config_name}.pt'),
                                      weights_only=True))
    print("\nFinal evaluation on full test set...")
    final_metrics = evaluate_model(model, test_loader, positions, object_size, device)
    print(f"  FINAL: Amp NRMSE={final_metrics['amp_nrmse_mean']:.4f}±{final_metrics['amp_nrmse_std']:.4f}, "
          f"Phase NRMSE={final_metrics['phase_nrmse_mean']:.4f}±{final_metrics['phase_nrmse_std']:.4f}")

    # Save logs
    logs = {
        'config': {'step_size': step_size, 'loss_type': loss_type,
                    'batch_size': batch_size, 'lr': lr, 'nf': nf,
                    'object_size': object_size, 'n_positions': len(positions),
                    'n_params': model.count_parameters(), 'max_steps': max_steps},
        'train_losses': train_losses,
        'val_metrics': val_metrics_log,
        'final_metrics': final_metrics,
        'total_time': time.time() - start_time,
    }
    with open(os.path.join(SAVE_DIR, f'log_{config_name}.json'), 'w') as f:
        json.dump(logs, f)

    # Plot
    plot_curves(train_losses, val_metrics_log, config_name)

    # Cleanup
    del model, optimizer, scheduler
    del train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return final_metrics, logs


def plot_curves(train_losses, val_metrics, config_name):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = [t['step'] for t in train_losses]
    losses = [t['loss'] for t in train_losses]
    ax1.plot(steps, losses, alpha=0.3, color='blue')
    if len(losses) > 50:
        w = min(50, len(losses) // 5)
        sm = np.convolve(losses, np.ones(w) / w, mode='valid')
        ax1.plot(steps[w - 1:], sm, color='blue', lw=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss ({config_name})')
    ax1.set_yscale('log')

    if val_metrics:
        vs = [m['step'] for m in val_metrics]
        ax2.plot(vs, [m['amp_nrmse_mean'] for m in val_metrics], 'o-', label='Amp NRMSE')
        ax2.plot(vs, [m['phase_nrmse_mean'] for m in val_metrics], 's-', label='Phase NRMSE')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('NRMSE')
        ax2.set_title(f'Validation NRMSE ({config_name})')
        ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'curves_{config_name}.png'), dpi=150)
    plt.close()


def main():
    from datasets import load_dataset

    print("Loading Flickr30K dataset...")
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    total = len(dataset)
    train_size = 28600
    test_size = total - train_size
    print(f"Total: {total}, Train: {train_size}, Test: {test_size}")

    train_data = dataset.select(range(train_size))
    test_data = dataset.select(range(train_size, total))

    all_results = {}
    start = time.time()

    # Training schedule - allocate time per config
    # MSE: faster (no DFT loss), DSE: slower
    # Sparse condition is hardest -> train longer
    configs = [
        (60, 'mse', 2500),
        (40, 'mse', 2000),
        (20, 'mse', 2000),
        (60, 'dse', 2000),
        (40, 'dse', 1500),
        (20, 'dse', 1500),
    ]

    for step_size, loss_type, max_steps in configs:
        elapsed = time.time() - start
        if elapsed > 45 * 60:  # 45 min budget
            print(f"\nTime budget exceeded ({elapsed/60:.1f} min), stopping.")
            break

        key = f'step{step_size}_{loss_type}'
        metrics, logs = train_single_config(
            step_size, loss_type,
            train_data, test_data,
            max_steps=max_steps,
            batch_size=32,
            lr=2e-4,
            nf=64,
            num_workers=4,
            eval_every=500,
        )
        all_results[key] = metrics

    # Save summary
    with open(os.path.join(SAVE_DIR, 'baseline_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print table
    print("\n" + "=" * 90)
    print("PTYCHONET BASELINE RESULTS")
    print("=" * 90)
    print(f"{'Config':<25} {'Amp NRMSE':>20} {'Phase NRMSE':>20}")
    print("-" * 65)
    for key in sorted(all_results.keys()):
        m = all_results[key]
        print(f"{key:<25} {m['amp_nrmse_mean']:.4f} ± {m['amp_nrmse_std']:.4f}    "
              f"{m['phase_nrmse_mean']:.4f} ± {m['phase_nrmse_std']:.4f}")

    print(f"\nTotal baseline training time: {(time.time()-start)/60:.1f} minutes")
    return all_results


if __name__ == '__main__':
    main()
