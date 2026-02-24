#!/usr/bin/env python3
"""
Training script for the improved PhysNet model.
Trains at all three overlap conditions with multiple loss variants.
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

# Add paths - proposal first so we import the right model
proposal_dir = os.path.dirname(os.path.abspath(__file__))
reimplement_dir = os.path.join(proposal_dir, '..', 'reimplement')
sys.path.insert(0, proposal_dir)

# Import from proposal
from model import PhysNet, PhysNetWithRefinement

# Import from reimplement
sys.path.insert(0, reimplement_dir)
from forward_model import (
    create_gaussian_probe, create_scan_positions,
    compute_object_size
)
from dataset import PtychographyDataset

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_nrmse(pred, target):
    mse = torch.mean((pred - target) ** 2, dim=(-2, -1))
    norm = torch.mean(target ** 2, dim=(-2, -1))
    return torch.sqrt(mse / (norm + 1e-10))


def correct_global_phase(pred_phase, gt_phase):
    diff = pred_phase - gt_phase
    c = diff.mean(dim=(-2, -1), keepdim=True)
    return pred_phase - c


def compute_dft_loss(output, probe, positions, diff_amps_target):
    """DFT loss: Fourier space consistency."""
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


def ssim_loss(pred, target, window_size=11):
    """Structural similarity loss (1 - SSIM)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Simple SSIM using average pooling
    pad = window_size // 2
    mu_pred = F.avg_pool2d(pred.unsqueeze(1) if pred.dim() == 3 else pred,
                            window_size, stride=1, padding=pad)
    mu_target = F.avg_pool2d(target.unsqueeze(1) if target.dim() == 3 else target,
                              window_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d((pred.unsqueeze(1) if pred.dim() == 3 else pred) ** 2,
                                   window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d((target.unsqueeze(1) if target.dim() == 3 else target) ** 2,
                                     window_size, stride=1, padding=pad) - mu_target_sq
    sigma_cross = F.avg_pool2d(
        (pred.unsqueeze(1) if pred.dim() == 3 else pred) *
        (target.unsqueeze(1) if target.dim() == 3 else target),
        window_size, stride=1, padding=pad
    ) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return 1.0 - ssim_map.mean()


import torch.nn.functional as F


def combined_loss(output, gt_amp, gt_phase, diff_amps, probe, positions,
                  lambda_dft=0.5, lambda_ssim=0.1):
    """
    Combined loss function:
    L = MSE(amp) + MSE(phase) + lambda_dft * DFT_loss + lambda_ssim * SSIM_loss

    The MSE in both real and Fourier space provides dual-space supervision.
    SSIM encourages structural preservation.
    """
    pred_amp = output[:, 0]
    pred_phase = output[:, 1]

    # MSE loss
    mse_amp = nn.functional.mse_loss(pred_amp, gt_amp)
    mse_phase = nn.functional.mse_loss(pred_phase, gt_phase)
    mse_loss = mse_amp + mse_phase

    # L1 loss for sharper predictions
    l1_loss = nn.functional.l1_loss(pred_amp, gt_amp) + \
              nn.functional.l1_loss(pred_phase, gt_phase)

    total = mse_loss + 0.1 * l1_loss

    # DFT consistency loss
    if lambda_dft > 0:
        dft_loss = compute_dft_loss(output, probe, positions, diff_amps)
        total = total + lambda_dft * dft_loss

    return total


def evaluate_model(model, test_loader, positions, object_size, device, max_batches=None):
    """Evaluate model."""
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


def evaluate_with_refinement(model, test_loader, positions, object_size, device,
                              probe, n_refine_steps=5, refine_lr=0.05, max_batches=None):
    """Evaluate model with physics-based iterative refinement at test time."""
    model.eval()
    all_amp_nrmse = []
    all_phase_nrmse = []
    h, w = probe.shape[-2], probe.shape[-1]
    probe_dev = probe.to(device)

    with torch.no_grad():
        for batch_idx, (diff_amps, gt_amp, gt_phase) in enumerate(test_loader):
            if max_batches and batch_idx >= max_batches:
                break
            diff_amps = diff_amps.to(device)
            gt_amp = gt_amp.to(device)
            gt_phase = gt_phase.to(device)

            output = model(diff_amps, positions, object_size)

            # Iterative refinement
            pred_amp = output[:, 0].clone()
            pred_phase = output[:, 1].clone()

            for step in range(n_refine_steps):
                pred_amp.requires_grad_(True)
                pred_phase.requires_grad_(True)

                obj_complex = pred_amp.double() * torch.exp(1j * pred_phase.double())
                loss = torch.tensor(0.0, device=device)

                for j, (top, left) in enumerate(positions):
                    patch = obj_complex[:, top:top+h, left:left+w]
                    exit_wave = patch * probe_dev.unsqueeze(0)
                    ft = torch.fft.fft2(exit_wave, norm='ortho')
                    pred_diff = torch.abs(ft).float()
                    loss = loss + torch.mean((pred_diff - diff_amps[:, j]) ** 2)
                loss = loss / len(positions)

                grads = torch.autograd.grad(loss, [pred_amp, pred_phase])

                with torch.no_grad():
                    pred_amp = pred_amp - refine_lr * grads[0]
                    pred_phase = pred_phase - refine_lr * grads[1]
                    pred_amp = pred_amp.clamp(0.5, 1.0)
                    pred_phase = pred_phase.clamp(-np.pi / 3.0, 0.0)

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


def train_single_config(step_size, hf_dataset_train, hf_dataset_test,
                         max_steps=3000, batch_size=24, lr=2e-4, nf=48,
                         refine_nf=32, num_workers=4, eval_every=500,
                         probe_sigma=30.0, lambda_dft=0.5):
    """Train PhysNet for one overlap condition."""
    device = torch.device('cuda')
    probe_size = 128
    object_size = compute_object_size(probe_size, step_size, n_steps=6)
    probe = create_gaussian_probe(probe_size, sigma=probe_sigma)
    positions = create_scan_positions(object_size, probe_size, step_size)

    config_name = f'physnet_step{step_size}'
    print(f"\n{'='*70}")
    print(f"Training {config_name}: obj_size={object_size}, n_pos={len(positions)}")
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

    model = PhysNet(nf=nf, refine_nf=refine_nf).to(device)
    print(f"PhysNet parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # Warmup + cosine schedule
    warmup_steps = 200
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses = []
    val_metrics_log = []
    best_val = float('inf')
    global_step = 0
    start_time = time.time()

    # Curriculum: start with MSE only, gradually add DFT loss
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

            # Curriculum: ramp up DFT loss
            dft_weight = min(lambda_dft, lambda_dft * global_step / 500)

            loss = combined_loss(
                output, gt_amp, gt_phase, diff_amps, probe, positions,
                lambda_dft=dft_weight, lambda_ssim=0.0
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            train_losses.append({
                'step': global_step,
                'loss': loss.item(),
                'time': time.time() - start_time
            })

            if global_step % 100 == 0:
                print(f"  Step {global_step}/{max_steps}, loss={loss.item():.6f}, "
                      f"lr={scheduler.get_last_lr()[0]:.6f}, "
                      f"dft_w={dft_weight:.3f}, "
                      f"time={time.time()-start_time:.0f}s")

            if global_step % eval_every == 0 or global_step == max_steps:
                metrics = evaluate_model(model, test_loader, positions, object_size,
                                          device, max_batches=20)
                metrics['step'] = global_step
                metrics['time'] = time.time() - start_time
                val_metrics_log.append(metrics)
                print(f"  >> Val: Amp={metrics['amp_nrmse_mean']:.4f}±{metrics['amp_nrmse_std']:.4f}, "
                      f"Phase={metrics['phase_nrmse_mean']:.4f}±{metrics['phase_nrmse_std']:.4f}")

                val_score = metrics['amp_nrmse_mean'] + metrics['phase_nrmse_mean']
                if val_score < best_val:
                    best_val = val_score
                    torch.save(model.state_dict(),
                               os.path.join(SAVE_DIR, f'best_{config_name}.pt'))
                model.train()

    # Load best and evaluate
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'best_{config_name}.pt'),
                                      weights_only=True))

    # DL-only evaluation
    print("\nFinal evaluation (DL only)...")
    dl_metrics = evaluate_model(model, test_loader, positions, object_size, device)
    print(f"  DL-only: Amp={dl_metrics['amp_nrmse_mean']:.4f}±{dl_metrics['amp_nrmse_std']:.4f}, "
          f"Phase={dl_metrics['phase_nrmse_mean']:.4f}±{dl_metrics['phase_nrmse_std']:.4f}")

    # Hybrid evaluation (DL + iterative refinement)
    print("Final evaluation (DL + refinement)...")
    hybrid_metrics = evaluate_with_refinement(
        model, test_loader, positions, object_size, device, probe,
        n_refine_steps=5, refine_lr=0.05, max_batches=20
    )
    print(f"  Hybrid: Amp={hybrid_metrics['amp_nrmse_mean']:.4f}±{hybrid_metrics['amp_nrmse_std']:.4f}, "
          f"Phase={hybrid_metrics['phase_nrmse_mean']:.4f}±{hybrid_metrics['phase_nrmse_std']:.4f}")

    # Save logs
    logs = {
        'config': {'step_size': step_size, 'batch_size': batch_size,
                    'lr': lr, 'nf': nf, 'refine_nf': refine_nf,
                    'object_size': object_size, 'n_positions': len(positions),
                    'n_params': model.count_parameters(), 'max_steps': max_steps,
                    'lambda_dft': lambda_dft},
        'train_losses': train_losses,
        'val_metrics': val_metrics_log,
        'dl_metrics': dl_metrics,
        'hybrid_metrics': hybrid_metrics,
        'total_time': time.time() - start_time,
    }
    with open(os.path.join(SAVE_DIR, f'log_{config_name}.json'), 'w') as f:
        json.dump(logs, f)

    plot_curves(train_losses, val_metrics_log, config_name)

    del model, optimizer, scheduler, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return dl_metrics, hybrid_metrics, logs


def plot_curves(train_losses, val_metrics, config_name):
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
    print(f"Total: {total}, Train: {train_size}, Test: {total - train_size}")

    train_data = dataset.select(range(train_size))
    test_data = dataset.select(range(train_size, total))

    all_results = {}
    start = time.time()

    # Train for all three conditions
    # Prioritize sparse (hardest) with more steps
    configs = [
        (60, 3000),   # Sparse - most steps
        (40, 2500),   # Moderate
        (20, 2500),   # Dense
    ]

    for step_size, max_steps in configs:
        elapsed = time.time() - start
        if elapsed > 55 * 60:  # 55 min budget for improved method
            print(f"\nTime budget exceeded ({elapsed/60:.1f} min), stopping.")
            break

        key = f'physnet_step{step_size}'
        dl_metrics, hybrid_metrics, logs = train_single_config(
            step_size, train_data, test_data,
            max_steps=max_steps,
            batch_size=24,
            lr=3e-4,
            nf=48,
            refine_nf=32,
            num_workers=4,
            eval_every=500,
            lambda_dft=0.5,
        )
        all_results[f'{key}_dl'] = dl_metrics
        all_results[f'{key}_hybrid'] = hybrid_metrics

    # Save summary
    with open(os.path.join(SAVE_DIR, 'improved_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 90)
    print("PHYSNET IMPROVED RESULTS")
    print("=" * 90)
    print(f"{'Config':<35} {'Amp NRMSE':>20} {'Phase NRMSE':>20}")
    print("-" * 75)
    for key in sorted(all_results.keys()):
        m = all_results[key]
        print(f"{key:<35} {m['amp_nrmse_mean']:.4f} ± {m['amp_nrmse_std']:.4f}    "
              f"{m['phase_nrmse_mean']:.4f} ± {m['phase_nrmse_std']:.4f}")

    print(f"\nTotal improved training time: {(time.time()-start)/60:.1f} minutes")
    return all_results


if __name__ == '__main__':
    main()
