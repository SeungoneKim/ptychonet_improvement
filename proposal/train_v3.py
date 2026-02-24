#!/usr/bin/env python3
"""
Training script V3 for PtychoNet+.
Same training regime as baseline PtychoNet (proven to work well),
but with the improved architecture.
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
reimplement_dir = os.path.join(proposal_dir, '..', 'reimplement')

# Add reimplement dir for shared modules
sys.path.insert(0, reimplement_dir)
from forward_model import create_gaussian_probe, create_scan_positions, compute_object_size
from dataset import PtychographyDataset

# Import proposal model with importlib
import importlib.util
def _import_from(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_proposal_model = _import_from("proposal_model", os.path.join(proposal_dir, "model.py"))
PtychoNetPlus = _proposal_model.PtychoNetPlus

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
    all_amp = []
    all_phase = []
    with torch.no_grad():
        for idx, (diff_amps, gt_amp, gt_phase) in enumerate(test_loader):
            if max_batches and idx >= max_batches:
                break
            diff_amps = diff_amps.to(device)
            gt_amp = gt_amp.to(device)
            gt_phase = gt_phase.to(device)
            output = model(diff_amps, positions, object_size)
            pred_phase_corr = correct_global_phase(output[:, 1], gt_phase)
            all_amp.append(compute_nrmse(output[:, 0], gt_amp).cpu())
            all_phase.append(compute_nrmse(pred_phase_corr, gt_phase).cpu())
    all_amp = torch.cat(all_amp)
    all_phase = torch.cat(all_phase)
    return {
        'amp_nrmse_mean': all_amp.mean().item(),
        'amp_nrmse_std': all_amp.std().item(),
        'phase_nrmse_mean': all_phase.mean().item(),
        'phase_nrmse_std': all_phase.std().item(),
    }


def evaluate_with_refinement(model, test_loader, positions, object_size, device,
                              probe, n_steps=5, lr=0.05, max_batches=None):
    """Hybrid: DL prediction + physics-based gradient descent."""
    model.eval()
    all_amp = []
    all_phase = []
    h, w = 128, 128
    probe_dev = probe.to(device)

    for idx, (diff_amps, gt_amp, gt_phase) in enumerate(test_loader):
        if max_batches and idx >= max_batches:
            break
        diff_amps = diff_amps.to(device)
        gt_amp = gt_amp.to(device)
        gt_phase = gt_phase.to(device)

        with torch.no_grad():
            output = model(diff_amps, positions, object_size)

        pred_amp = output[:, 0].clone()
        pred_phase = output[:, 1].clone()

        for step in range(n_steps):
            pred_amp.requires_grad_(True)
            pred_phase.requires_grad_(True)

            obj_c = pred_amp.double() * torch.exp(1j * pred_phase.double())
            loss = torch.tensor(0.0, device=device)
            for j, (top, left) in enumerate(positions):
                patch = obj_c[:, top:top+h, left:left+w]
                ew = patch * probe_dev.unsqueeze(0)
                ft = torch.fft.fft2(ew, norm='ortho')
                loss = loss + torch.mean((torch.abs(ft).float() - diff_amps[:, j]) ** 2)
            loss = loss / len(positions)

            grads = torch.autograd.grad(loss, [pred_amp, pred_phase])
            with torch.no_grad():
                pred_amp = (pred_amp - lr * grads[0]).clamp(0.5, 1.0)
                pred_phase = (pred_phase - lr * grads[1]).clamp(-np.pi/3, 0.0)

        with torch.no_grad():
            pred_phase_corr = correct_global_phase(pred_phase, gt_phase)
            all_amp.append(compute_nrmse(pred_amp, gt_amp).cpu())
            all_phase.append(compute_nrmse(pred_phase_corr, gt_phase).cpu())

    all_amp = torch.cat(all_amp)
    all_phase = torch.cat(all_phase)
    return {
        'amp_nrmse_mean': all_amp.mean().item(),
        'amp_nrmse_std': all_amp.std().item(),
        'phase_nrmse_mean': all_phase.mean().item(),
        'phase_nrmse_std': all_phase.std().item(),
    }


def train_single(step_size, train_data, test_data, max_steps=2500,
                  batch_size=32, lr=2e-4, nf=64, refine_nf=48,
                  num_workers=4, eval_every=500, probe_sigma=30.0):
    device = torch.device('cuda')
    probe_size = 128
    object_size = compute_object_size(probe_size, step_size, n_steps=6)
    probe = create_gaussian_probe(probe_size, sigma=probe_sigma)
    positions = create_scan_positions(object_size, probe_size, step_size)

    name = f'ptychonetplus_step{step_size}'
    print(f"\n{'='*70}")
    print(f"Training {name}: obj_size={object_size}, n_pos={len(positions)}")
    print(f"{'='*70}")

    train_ds = PtychographyDataset(train_data, probe, positions, probe_size=probe_size, object_size=object_size)
    test_ds = PtychographyDataset(test_data, probe, positions, probe_size=probe_size, object_size=object_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0
    )

    model = PtychoNetPlus(nf=nf, refine_nf=refine_nf).to(device)
    print(f"Params: {model.count_parameters():,}")

    # Same optimizer as baseline (proven to work)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr*0.01)

    train_losses = []
    val_log = []
    best_val = float('inf')
    step = 0
    t0 = time.time()

    while step < max_steps:
        model.train()
        for diff_amps, gt_amp, gt_phase in train_loader:
            if step >= max_steps:
                break

            diff_amps = diff_amps.to(device, non_blocking=True)
            gt_amp = gt_amp.to(device, non_blocking=True)
            gt_phase = gt_phase.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(diff_amps, positions, object_size)

            # MSE loss (same as baseline MSE)
            loss = nn.functional.mse_loss(output[:, 0], gt_amp) + \
                   nn.functional.mse_loss(output[:, 1], gt_phase)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1
            train_losses.append({'step': step, 'loss': loss.item(), 'time': time.time()-t0})

            if step % 100 == 0:
                print(f"  Step {step}/{max_steps}, loss={loss.item():.6f}, "
                      f"lr={scheduler.get_last_lr()[0]:.6f}, time={time.time()-t0:.0f}s")

            if step % eval_every == 0 or step == max_steps:
                m = evaluate_model(model, test_loader, positions, object_size, device, max_batches=20)
                m['step'] = step
                m['time'] = time.time() - t0
                val_log.append(m)
                print(f"  >> Val: Amp={m['amp_nrmse_mean']:.4f}±{m['amp_nrmse_std']:.4f}, "
                      f"Phase={m['phase_nrmse_mean']:.4f}±{m['phase_nrmse_std']:.4f}")
                vs = m['amp_nrmse_mean'] + m['phase_nrmse_mean']
                if vs < best_val:
                    best_val = vs
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'best_{name}.pt'))
                model.train()

    # Load best
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, f'best_{name}.pt'), weights_only=True))

    print("\nFinal eval (DL only)...")
    dl_m = evaluate_model(model, test_loader, positions, object_size, device)
    print(f"  DL: Amp={dl_m['amp_nrmse_mean']:.4f}±{dl_m['amp_nrmse_std']:.4f}, "
          f"Phase={dl_m['phase_nrmse_mean']:.4f}±{dl_m['phase_nrmse_std']:.4f}")

    print("Eval with refinement (5 steps)...")
    h5 = evaluate_with_refinement(model, test_loader, positions, object_size, device, probe,
                                    n_steps=5, lr=0.05, max_batches=20)
    print(f"  Hybrid-5: Amp={h5['amp_nrmse_mean']:.4f}±{h5['amp_nrmse_std']:.4f}, "
          f"Phase={h5['phase_nrmse_mean']:.4f}±{h5['phase_nrmse_std']:.4f}")

    print("Eval with refinement (10 steps)...")
    h10 = evaluate_with_refinement(model, test_loader, positions, object_size, device, probe,
                                     n_steps=10, lr=0.03, max_batches=20)
    print(f"  Hybrid-10: Amp={h10['amp_nrmse_mean']:.4f}±{h10['amp_nrmse_std']:.4f}, "
          f"Phase={h10['phase_nrmse_mean']:.4f}±{h10['phase_nrmse_std']:.4f}")

    logs = {
        'config': {'step_size': step_size, 'nf': nf, 'refine_nf': refine_nf,
                    'n_params': model.count_parameters(), 'max_steps': max_steps,
                    'batch_size': batch_size, 'lr': lr,
                    'object_size': object_size, 'n_positions': len(positions)},
        'train_losses': train_losses,
        'val_metrics': val_log,
        'dl_metrics': dl_m,
        'hybrid5_metrics': h5,
        'hybrid10_metrics': h10,
        'total_time': time.time() - t0,
    }
    with open(os.path.join(SAVE_DIR, f'log_{name}.json'), 'w') as f:
        json.dump(logs, f)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    steps = [t['step'] for t in train_losses]
    losses = [t['loss'] for t in train_losses]
    ax1.plot(steps, losses, alpha=0.3, color='blue')
    if len(losses) > 50:
        w = min(50, len(losses) // 5)
        sm = np.convolve(losses, np.ones(w) / w, mode='valid')
        ax1.plot(steps[w-1:], sm, color='blue', lw=2)
    ax1.set_xlabel('Step'); ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss ({name})'); ax1.set_yscale('log')
    if val_log:
        vs = [v['step'] for v in val_log]
        ax2.plot(vs, [v['amp_nrmse_mean'] for v in val_log], 'o-', label='Amp')
        ax2.plot(vs, [v['phase_nrmse_mean'] for v in val_log], 's-', label='Phase')
        ax2.set_xlabel('Step'); ax2.set_ylabel('NRMSE')
        ax2.set_title(f'Val NRMSE ({name})'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'curves_{name}.png'), dpi=150)
    plt.close()

    del optimizer, scheduler, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    return model, dl_m, h5, h10, logs


def main():
    from datasets import load_dataset

    print("Loading Flickr30K...")
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    total = len(dataset)
    train_data = dataset.select(range(28600))
    test_data = dataset.select(range(28600, total))
    print(f"Total: {total}, Train: 28600, Test: {total-28600}")

    results = {}
    start = time.time()

    configs = [
        (60, 2500),
        (40, 2000),
        (20, 2000),
    ]

    for step_size, max_steps in configs:
        if time.time() - start > 55 * 60:
            print("Time budget exceeded")
            break

        key = f'ptychonetplus_step{step_size}'
        model, dl_m, h5, h10, logs = train_single(
            step_size, train_data, test_data,
            max_steps=max_steps, batch_size=32,
            lr=2e-4, nf=64, refine_nf=48,
            num_workers=4, eval_every=500,
        )
        results[f'{key}_dl'] = dl_m
        results[f'{key}_hybrid5'] = h5
        results[f'{key}_hybrid10'] = h10
        del model
        torch.cuda.empty_cache()
        gc.collect()

    with open(os.path.join(SAVE_DIR, 'improved_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 90)
    print("PTYCHONET+ RESULTS")
    print("=" * 90)
    for key in sorted(results.keys()):
        m = results[key]
        print(f"  {key:<45} Amp={m['amp_nrmse_mean']:.4f}±{m['amp_nrmse_std']:.4f}  "
              f"Phase={m['phase_nrmse_mean']:.4f}±{m['phase_nrmse_std']:.4f}")

    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")


if __name__ == '__main__':
    main()
