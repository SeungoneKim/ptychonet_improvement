"""
Dataset for ptychographic phase retrieval.
Downloads Flickr30K from HuggingFace, simulates ptychographic measurements.
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from forward_model import (
    create_gaussian_probe, create_scan_positions,
    compute_object_size, simulate_ptychography
)


class PtychographyDataset(Dataset):
    """
    Dataset that generates ptychographic measurements from images.
    Uses Flickr30K images as ground truth objects.
    """

    def __init__(self, hf_dataset, probe, positions, probe_size=128,
                 object_size=248, precompute=False, cache_dir=None):
        """
        Args:
            hf_dataset: HuggingFace dataset split
            probe: complex probe tensor (h, w)
            positions: list of (top, left) scan positions
            probe_size: probe size in pixels
            object_size: required object size
            precompute: if True, precompute all diffraction patterns
            cache_dir: directory for caching
        """
        self.hf_dataset = hf_dataset
        self.probe = probe
        self.positions = positions
        self.probe_size = probe_size
        self.object_size = object_size
        self.precompute = precompute
        self.cache_dir = cache_dir

        self.transform = transforms.Compose([
            transforms.Resize(object_size),
            transforms.CenterCrop(object_size),
            transforms.Grayscale(),
            transforms.ToTensor(),  # [0, 1]
        ])

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # Get image from HuggingFace dataset
        item = self.hf_dataset[idx]
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')

        gray = self.transform(img)  # (1, H, W) in [0, 1]
        gray = gray.squeeze(0)  # (H, W)

        # Create amplitude and phase ground truth
        # Amplitude: [0.5, 1.0], Phase: [-pi/3, 0]
        amplitude = gray * 0.5 + 0.5  # [0.5, 1.0]
        phase = -gray * (np.pi / 3.0)  # [-pi/3, 0]

        # Simulate ptychography
        amp_batch = amplitude.unsqueeze(0).double()
        phase_batch = phase.unsqueeze(0).double()

        with torch.no_grad():
            diff_amps = simulate_ptychography(
                amp_batch, phase_batch, self.probe, self.positions
            )

        diff_amps = diff_amps.squeeze(0).float()  # (N, h, w)
        amplitude = amplitude.float()
        phase = phase.float()

        return diff_amps, amplitude, phase


def get_dataloaders(step_size=20, batch_size=16, num_workers=4,
                    train_size=28600, test_size=3183, probe_sigma=30.0):
    """
    Create train and test dataloaders.

    Args:
        step_size: lateral offset in pixels (20, 40, or 60)
        batch_size: batch size for training
        num_workers: number of data loading workers

    Returns:
        train_loader, test_loader, probe, positions, object_size
    """
    from datasets import load_dataset

    print("Loading Flickr30K dataset...")
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    train_data = dataset.select(range(train_size))
    test_data = dataset.select(range(train_size, train_size + test_size))

    probe_size = 128
    object_size = compute_object_size(probe_size, step_size, n_steps=6)
    print(f"Object size for step_size={step_size}: {object_size}x{object_size}")

    probe = create_gaussian_probe(probe_size, sigma=probe_sigma)
    positions = create_scan_positions(object_size, probe_size, step_size)
    print(f"Number of scan positions: {len(positions)}")

    train_dataset = PtychographyDataset(
        train_data, probe, positions,
        probe_size=probe_size, object_size=object_size
    )
    test_dataset = PtychographyDataset(
        test_data, probe, positions,
        probe_size=probe_size, object_size=object_size
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, test_loader, probe, positions, object_size
