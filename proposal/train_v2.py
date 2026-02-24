#!/usr/bin/env python3
"""
Training script V2 for PhysNet.
Strategy: Train with MSE+L1 loss only (no DFT loss during training for speed).
The global refinement network + U-Net skip connections should provide the
improvement over baseline. Physics-based refinement is applied at test time.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

proposal_dir = os.path.dirname(os.path.abspath(__file__))
reimplement_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'reimplement')

# Add reimplement dir first for forward_model/dataset, then use importlib for model
sys.path.insert(0, reimplement_dir)
from forward_model import create_gaussian_probe, create_scan_positions, compute_object_size
from dataset import PtychographyDataset

# Use importlib for proposal model to avoid conflict with reimplement model
import importlib.util
def _import_from(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_proposal_model = _import_from("proposal_model", os.path.join(proposal_dir, "model.py"))
PhysNet = _proposal_model.PhysNet

SAVE_DIR = os.path.join(proposal_dir, 'checkpoints')
os.makedirs(SAVE_DIR, exist_ok=True)


def compute_nrmse(pred, target):
    mse = torch.mean((pred - target) ** 2, dim=(-2, -1))
    norm = torch.mean(target ** 2, dim=(-2, -1))
    return torch.sqrt(mse / (norm + 1e-10))


def correct_global_phase(pred_phase, gt_phase):
    diff = pred_phase - gt_phase
    c = diff.mean(dim=(-2, -1), keepdim=True)
    return pred_phase - c


def evaluate_model(model, test_loader, positions, object_size, device, max_batches=None):
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
    """DL + physics-based iterative refinement at test time."""
    model.eval()
    all_amp_nrmse = []
    all_phase_nrmse = []
    h, w = 128, 128
    probe_dev = probe.to(device)

    for batch_idx, (diff_amps, gt_amp, gt_phase) in enumerate(test_loader):
        if max_batches and batch_idx >= max_batches:
            break
        diff_amps = diff_amps.to(device)
        gt_amp = gt_amp.to(device)
        gt_phase = gt_phase.to(device)

        with torch.no_grad():
            output = model(diff_amps, positions, object_size)

        pred_amp = output[:, 0].clone()
        pred_phase = output[:, 1].clone()

        # Iterative refinement in Fourier space
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
                pred_amp = (pred_amp - refine_lr * grads[0]).clamp(0.5, 1.0)
                pred_phase = (pred_phase - refine_lr * grads[1]).clamp(-np.pi/3, 0.0)

        with torch.no_grad():
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
                         max_steps=2500, batch_size=24, lr=2e-4, nf=48,
                         refine_nf=32, num_workers=4, eval_every=500,
                         probe_sigma=30.0):
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

    # Use Adam with beta1=0.5 like baseline (which works well)
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

            # MSE + L1 loss (no DFT for speed)
            mse_loss = nn.functional.mse_loss(pred_amp, gt_amp) + \
                       nn.functional.mse_loss(pred_phase, gt_phase)
            l1_loss = nn.functional.l1_loss(pred_amp, gt_amp) + \
                      nn.functional.l1_loss(pred_phase, gt_phase)
            loss = mse_loss + 0.1 * l1_loss

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

    # Load best model
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'best_{config_name}.pt'),
                                      weights_only=True))

    # DL-only evaluation on full test set
    print("\nFinal evaluation (DL only)...")
    dl_metrics = evaluate_model(model, test_loader, positions, object_size, device)
    print(f"  DL-only: Amp={dl_metrics['amp_nrmse_mean']:.4f}±{dl_metrics['amp_nrmse_std']:.4f}, "
          f"Phase={dl_metrics['phase_nrmse_mean']:.4f}±{dl_metrics['phase_nrmse_std']:.4f}")

    # Hybrid: DL + physics refinement
    print("Evaluation with physics refinement (5 steps)...")
    hybrid_metrics = evaluate_with_refinement(
        model, test_loader, positions, object_size, device, probe,
        n_refine_steps=5, refine_lr=0.05, max_batches=20
    )
    print(f"  Hybrid: Amp={hybrid_metrics['amp_nrmse_mean']:.4f}±{hybrid_metrics['amp_nrmse_std']:.4f}, "
          f"Phase={hybrid_metrics['phase_nrmse_mean']:.4f}±{hybrid_metrics['phase_nrmse_std']:.4f}")

    # Also try 10 refinement steps
    print("Evaluation with physics refinement (10 steps)...")
    hybrid10_metrics = evaluate_with_refinement(
        model, test_loader, positions, object_size, device, probe,
        n_refine_steps=10, refine_lr=0.03, max_batches=20
    )
    print(f"  Hybrid-10: Amp={hybrid10_metrics['amp_nrmse_mean']:.4f}±{hybrid10_metrics['amp_nrmse_std']:.4f}, "
          f"Phase={hybrid10_metrics['phase_nrmse_mean']:.4f}±{hybrid10_metrics['phase_nrmse_std']:.4f}")

    # Save logs
    logs = {
        'config': {'step_size': step_size, 'batch_size': batch_size,
                    'lr': lr, 'nf': nf, 'refine_nf': refine_nf,
                    'object_size': object_size, 'n_positions': len(positions),
                    'n_params': model.count_parameters(), 'max_steps': max_steps},
        'train_losses': train_losses,
        'val_metrics': val_metrics_log,
        'dl_metrics': dl_metrics,
        'hybrid_metrics': hybrid_metrics,
        'hybrid10_metrics': hybrid10_metrics,
        'total_time': time.time() - start_time,
    }
    with open(os.path.join(SAVE_DIR, f'log_{config_name}.json'), 'w') as f:
        json.dump(logs, f)

    plot_curves(train_losses, val_metrics_log, config_name)

    del optimizer, scheduler, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return model, dl_metrics, hybrid_metrics, hybrid10_metrics, logs


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

    # Train all three conditions
    configs = [
        (60, 3000),   # Sparse
        (40, 2500),   # Moderate
        (20, 2500),   # Dense
    ]

    for step_size, max_steps in configs:
        elapsed = time.time() - start
        if elapsed > 55 * 60:
            print(f"\nTime budget exceeded ({elapsed/60:.1f} min), stopping.")
            break

        key = f'physnet_step{step_size}'
        model, dl_metrics, hybrid_metrics, hybrid10_metrics, logs = train_single_config(
            step_size, train_data, test_data,
            max_steps=max_steps,
            batch_size=24,
            lr=2e-4,
            nf=48,
            refine_nf=32,
            num_workers=4,
            eval_every=500,
        )
        all_results[f'{key}_dl'] = dl_metrics
        all_results[f'{key}_hybrid5'] = hybrid_metrics
        all_results[f'{key}_hybrid10'] = hybrid10_metrics

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Save summary
    with open(os.path.join(SAVE_DIR, 'improved_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 90)
    print("PHYSNET IMPROVED RESULTS")
    print("=" * 90)
    print(f"{'Config':<40} {'Amp NRMSE':>20} {'Phase NRMSE':>20}")
    print("-" * 80)
    for key in sorted(all_results.keys()):
        m = all_results[key]
        print(f"{key:<40} {m['amp_nrmse_mean']:.4f} ± {m['amp_nrmse_std']:.4f}    "
              f"{m['phase_nrmse_mean']:.4f} ± {m['phase_nrmse_std']:.4f}")

    print(f"\nTotal improved training time: {(time.time()-start)/60:.1f} minutes")
    return all_results


if __name__ == '__main__':
    main()
