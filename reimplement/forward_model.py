"""
Forward model for ptychographic imaging simulation.
Implements the exact Fourier optics forward model:
    I_j(q) = |DFT[P(r - r_j) * T(r)]|^2
"""
import torch
import numpy as np


def create_gaussian_probe(size=128, sigma=30.0):
    """Create a Gaussian probe of given size."""
    y = torch.linspace(-size // 2, size // 2 - 1, size)
    x = torch.linspace(-size // 2, size // 2 - 1, size)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r2 = xx ** 2 + yy ** 2
    amplitude = torch.exp(-r2 / (2 * sigma ** 2))
    # Add some phase variation (Fresnel-like)
    phase = torch.exp(1j * 0.5 * r2 / (size * 2.0))
    probe = amplitude * phase
    return probe  # complex128


def create_scan_positions(object_size, probe_size, step_size):
    """
    Create a regular grid scan pattern.
    Returns list of (top, left) positions for each scan point.
    For 6x6 grid as specified.
    """
    positions = []
    n_steps = 6  # 6x6 grid as specified
    for i in range(n_steps):
        for j in range(n_steps):
            top = i * step_size
            left = j * step_size
            # Make sure probe fits within object
            if top + probe_size <= object_size and left + probe_size <= object_size:
                positions.append((top, left))
    return positions


def compute_object_size(probe_size, step_size, n_steps=6):
    """Compute the required object size for the scan pattern."""
    return (n_steps - 1) * step_size + probe_size


def simulate_ptychography(obj_amplitude, obj_phase, probe, positions):
    """
    Simulate ptychographic measurements.

    Args:
        obj_amplitude: (B, H, W) amplitude in [0.5, 1.0]
        obj_phase: (B, H, W) phase in [-pi/3, 0]
        probe: (h, w) complex probe
        positions: list of (top, left) tuples

    Returns:
        diffraction_amplitudes: (B, N, h, w) sqrt of intensity patterns
    """
    B = obj_amplitude.shape[0]
    h, w = probe.shape
    N = len(positions)

    # Build complex object
    obj_complex = obj_amplitude * torch.exp(1j * obj_phase)

    device = obj_amplitude.device
    probe = probe.to(device)

    diff_amps = torch.zeros(B, N, h, w, device=device)

    for idx, (top, left) in enumerate(positions):
        # Extract patch
        patch = obj_complex[:, top:top + h, left:left + w]
        # Multiply by probe
        exit_wave = patch * probe.unsqueeze(0)
        # DFT
        ft = torch.fft.fft2(exit_wave, norm='ortho')
        # Intensity -> amplitude
        intensity = torch.abs(ft) ** 2
        diff_amps[:, idx] = torch.sqrt(intensity)

    return diff_amps


def simulate_ptychography_from_complex(obj_complex, probe, positions):
    """
    Simulate ptychography from a complex object directly.
    Used in DSE loss computation.

    Args:
        obj_complex: (B, H, W) complex object
        probe: (h, w) complex probe
        positions: list of (top, left) tuples

    Returns:
        diffraction_amplitudes: (B, N, h, w)
    """
    B = obj_complex.shape[0]
    h, w = probe.shape
    N = len(positions)
    device = obj_complex.device
    probe = probe.to(device)

    diff_amps = torch.zeros(B, N, h, w, device=device)

    for idx, (top, left) in enumerate(positions):
        patch = obj_complex[:, top:top + h, left:left + w]
        exit_wave = patch * probe.unsqueeze(0)
        ft = torch.fft.fft2(exit_wave, norm='ortho')
        diff_amps[:, idx] = torch.abs(ft)

    return diff_amps
