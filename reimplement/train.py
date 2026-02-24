"""
Training script for PtychoNet baseline.
Trains with MSE and DSE losses at all three overlap conditions.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import PtychoNet
from forward_model import (
    create_gaussian_probe, create_scan_positions,
    compute_object_size, simulate_ptychography_from_complex
)
from dataset import PtychographyDataset


def compute_nrmse(pred, target):
    """Compute Normalized Root Mean Square Error."""
    mse = torch.mean((pred - target) ** 2, dim=(-2, -1))
    norm = torch.mean(target ** 2, dim=(-2, -1))
    nrmse = torch.sqrt(mse / (norm + 1e-10))
    return nrmse


def correct_global_phase(pred_phase, gt_phase):
    """
    Correct global phase shift.
    Find constant c that minimizes ||pred_phase - gt_phase - c||.
    c = mean(pred_phase - gt_phase)
    """
    diff = pred_phase - gt_phase
    c = diff.mean(dim=(-2, -1), keepdim=True)
    return pred_phase - c


def compute_dft_loss(output, probe, positions, diff_amps_target):
    """
    Compute DFT loss (Fourier space consistency).
    Re-simulate ptychography from predicted object and compare with input.
    """
    B = output.shape[0]
    h, w = probe.shape
    device = output.device

    pred_amp = output[:, 0]   # (B, H, W)
    pred_phase = output[:, 1]  # (B, H, W)

    # Build complex object from prediction
    obj_complex = pred_amp * torch.exp(1j * pred_phase.double())

    probe_dev = probe.to(device)

    total_loss = 0.0
    N = len(positions)

    for j in range(N):
        top, left = positions[j]
        patch = obj_complex[:, top:top + h, left:left + w]
        exit_wave = patch * probe_dev.unsqueeze(0)
        ft = torch.fft.fft2(exit_wave, norm='ortho')
        pred_diff_amp = torch.abs(ft).float()
        target = diff_amps_target[:, j]
        total_loss += torch.mean((pred_diff_amp - target) ** 2)

    return total_loss / N


def evaluate(model, test_loader, positions, object_size, device, max_batches=None):
    """Evaluate model on test set."""
    model.eval()
    amp_nrmses = []
    phase_nrmses = []

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

            # Phase correction
            pred_phase_corrected = correct_global_phase(pred_phase, gt_phase)

            # NRMSE
            amp_nrmse = compute_nrmse(pred_amp, gt_amp)
            phase_nrmse = compute_nrmse(pred_phase_corrected, gt_phase)

            amp_nrmses.append(amp_nrmse.cpu())
            phase_nrmses.append(phase_nrmse.cpu())

    amp_nrmses = torch.cat(amp_nrmses)
    phase_nrmses = torch.cat(phase_nrmses)

    return {
        'amp_nrmse_mean': amp_nrmses.mean().item(),
        'amp_nrmse_std': amp_nrmses.std().item(),
        'phase_nrmse_mean': phase_nrmses.mean().item(),
        'phase_nrmse_std': phase_nrmses.std().item(),
    }


def train_ptychonet(step_size, loss_type='mse', epochs=5, batch_size=16,
                    lr=2e-4, nf=64, save_dir='checkpoints', probe_sigma=30.0,
                    max_train_samples=None, num_workers=4, eval_every=500,
                    max_steps=None):
    """
    Train PtychoNet for a given overlap condition.

    Args:
        step_size: lateral offset (20, 40, or 60)
        loss_type: 'mse' or 'dse'
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        nf: base number of filters
        save_dir: directory for saving checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training PtychoNet - step_size={step_size}, loss={loss_type}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    # Setup
    probe_size = 128
    object_size = compute_object_size(probe_size, step_size, n_steps=6)
    probe = create_gaussian_probe(probe_size, sigma=probe_sigma)
    positions = create_scan_positions(object_size, probe_size, step_size)
    n_positions = len(positions)

    print(f"Object size: {object_size}x{object_size}")
    print(f"Number of scan positions: {n_positions}")
    overlap = max(0, (probe_size - step_size) / probe_size * 100)
    print(f"Approximate overlap: {overlap:.1f}%")

    # Load dataset
    from datasets import load_dataset
    print("Loading Flickr30K dataset...")
    dataset = load_dataset("nlphuji/flickr30k", split="test")

    train_size = 28600
    test_size = 3183
    if max_train_samples:
        train_size = min(train_size, max_train_samples)

    train_data = dataset.select(range(train_size))
    test_data = dataset.select(range(28600, 28600 + test_size))

    train_dataset = PtychographyDataset(
        train_data, probe, positions,
        probe_size=probe_size, object_size=object_size
    )
    test_dataset = PtychographyDataset(
        test_data, probe, positions,
        probe_size=probe_size, object_size=object_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    # Model
    model = PtychoNet(nf=nf).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer (Adam with beta1=0.5 as in paper)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=lr * 0.01
    )

    # Training logs
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f'log_step{step_size}_{loss_type}.json')
    train_losses = []
    val_metrics = []

    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (diff_amps, gt_amp, gt_phase) in enumerate(pbar):
            diff_amps = diff_amps.to(device)
            gt_amp = gt_amp.to(device)
            gt_phase = gt_phase.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(diff_amps, positions, object_size)

            pred_amp = output[:, 0]
            pred_phase = output[:, 1]

            # MSE loss
            mse_loss = nn.functional.mse_loss(pred_amp, gt_amp) + \
                       nn.functional.mse_loss(pred_phase, gt_phase)

            if loss_type == 'dse':
                # Add DFT loss
                dft_loss = compute_dft_loss(
                    output, probe, positions, diff_amps
                )
                loss = mse_loss + dft_loss
            else:
                loss = mse_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            train_losses.append({
                'step': global_step,
                'loss': loss.item(),
                'time': time.time() - start_time
            })

            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

            # Periodic evaluation
            if global_step % eval_every == 0:
                metrics = evaluate(model, test_loader, positions, object_size,
                                   device, max_batches=10)
                metrics['step'] = global_step
                metrics['time'] = time.time() - start_time
                val_metrics.append(metrics)
                print(f"\n  [Step {global_step}] Val Amp NRMSE: {metrics['amp_nrmse_mean']:.4f} ± {metrics['amp_nrmse_std']:.4f}")
                print(f"  [Step {global_step}] Val Phase NRMSE: {metrics['phase_nrmse_mean']:.4f} ± {metrics['phase_nrmse_std']:.4f}")

                # Save best model
                val_loss = metrics['amp_nrmse_mean'] + metrics['phase_nrmse_mean']
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': global_step,
                        'metrics': metrics,
                    }, os.path.join(save_dir, f'best_step{step_size}_{loss_type}.pt'))

                model.train()

            if max_steps and global_step >= max_steps:
                break

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.6f}")

        if max_steps and global_step >= max_steps:
            break

    # Final evaluation on full test set
    print("\nFinal evaluation on full test set...")
    final_metrics = evaluate(model, test_loader, positions, object_size, device)
    print(f"Final Amp NRMSE: {final_metrics['amp_nrmse_mean']:.4f} ± {final_metrics['amp_nrmse_std']:.4f}")
    print(f"Final Phase NRMSE: {final_metrics['phase_nrmse_mean']:.4f} ± {final_metrics['phase_nrmse_std']:.4f}")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': global_step,
        'metrics': final_metrics,
    }, os.path.join(save_dir, f'final_step{step_size}_{loss_type}.pt'))

    # Save logs
    logs = {
        'config': {
            'step_size': step_size,
            'loss_type': loss_type,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'nf': nf,
            'object_size': object_size,
            'n_positions': n_positions,
            'n_params': model.count_parameters(),
        },
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'final_metrics': final_metrics,
        'total_time': time.time() - start_time,
    }

    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)

    # Plot training curves
    plot_training_curves(train_losses, val_metrics, step_size, loss_type, save_dir)

    return model, final_metrics, logs


def plot_training_curves(train_losses, val_metrics, step_size, loss_type, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Training loss
    steps = [t['step'] for t in train_losses]
    losses = [t['loss'] for t in train_losses]
    axes[0].plot(steps, losses, alpha=0.3, color='blue')
    # Smoothed
    if len(losses) > 50:
        window = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        axes[0].plot(steps[window - 1:], smoothed, color='blue', linewidth=2)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training Loss (step={step_size}, {loss_type})')
    axes[0].set_yscale('log')

    # Validation NRMSE
    if val_metrics:
        val_steps = [m['step'] for m in val_metrics]
        amp_nrmse = [m['amp_nrmse_mean'] for m in val_metrics]
        phase_nrmse = [m['phase_nrmse_mean'] for m in val_metrics]
        axes[1].plot(val_steps, amp_nrmse, 'o-', label='Amplitude NRMSE')
        axes[1].plot(val_steps, phase_nrmse, 's-', label='Phase NRMSE')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('NRMSE')
        axes[1].set_title(f'Validation NRMSE (step={step_size}, {loss_type})')
        axes[1].legend()

    # Time vs step
    times = [t['time'] / 60 for t in train_losses]
    axes[2].plot(steps, times)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Time (min)')
    axes[2].set_title('Training Time')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'curves_step{step_size}_{loss_type}.png'), dpi=150)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step_size', type=int, default=20, choices=[20, 40, 60])
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'dse'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--probe_sigma', type=float, default=30.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--max_train_samples', type=int, default=None)

    args = parser.parse_args()
    train_ptychonet(**vars(args))
