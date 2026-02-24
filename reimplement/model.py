"""
PtychoNet: Encoder-Decoder CNN for ptychographic phase retrieval.

Architecture from Guan & Tsai (BMVC 2019):
- Encoder: Conv 4x4 stride 2 + LeakyReLU(0.2) + BN
- Decoder: ConvTranspose 4x4 stride 2 + ReLU + BN
- Output: sigmoid activation
- Processes each diffraction pattern independently
- Stitches patches together via averaging in overlapping regions
"""
import torch
import torch.nn as nn
import numpy as np


class PtychoNetEncoder(nn.Module):
    """Encoder part of PtychoNet. Maps 128x128 diffraction pattern to latent."""

    def __init__(self, nf=64):
        super().__init__()
        # Input: 1 x 128 x 128
        # Following DCGAN-style encoder with 4x4 conv, stride 2
        self.layers = nn.Sequential(
            # 1 x 128 x 128 -> nf x 64 x 64
            nn.Conv2d(1, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # nf x 64 x 64 -> 2nf x 32 x 32
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 2nf x 32 x 32 -> 4nf x 16 x 16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4nf x 16 x 16 -> 8nf x 8 x 8
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8nf x 8 x 8 -> 8nf x 4 x 4
            nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8nf x 4 x 4 -> 8nf x 2 x 2
            nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 8nf x 2 x 2 -> 8nf x 1 x 1
            nn.Conv2d(nf * 8, nf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class PtychoNetDecoder(nn.Module):
    """Decoder part of PtychoNet. Maps latent to 2x128x128 (amplitude + phase)."""

    def __init__(self, nf=64):
        super().__init__()
        # Mirror of encoder with ConvTranspose
        self.layers = nn.Sequential(
            # 8nf x 1 x 1 -> 8nf x 2 x 2
            nn.ConvTranspose2d(nf * 8, nf * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),

            # 8nf x 2 x 2 -> 8nf x 4 x 4
            nn.ConvTranspose2d(nf * 8, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),

            # 8nf x 4 x 4 -> 8nf x 8 x 8
            nn.ConvTranspose2d(nf * 8, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(inplace=True),

            # 8nf x 8 x 8 -> 4nf x 16 x 16
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(inplace=True),

            # 4nf x 16 x 16 -> 2nf x 32 x 32
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),

            # 2nf x 32 x 32 -> nf x 64 x 64
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            # nf x 64 x 64 -> 2 x 128 x 128
            nn.ConvTranspose2d(nf, 2, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class PtychoNet(nn.Module):
    """
    Full PtychoNet model.
    Processes each diffraction pattern through encoder-decoder,
    then stitches patches together.

    Output channels: 2 (amplitude and phase)
    - Channel 0: amplitude, sigmoid maps to [0, 1], then rescaled to [0.5, 1.0]
    - Channel 1: phase, sigmoid maps to [0, 1], then rescaled to [-pi/3, 0]
    """

    def __init__(self, nf=64):
        super().__init__()
        self.encoder = PtychoNetEncoder(nf=nf)
        self.decoder = PtychoNetDecoder(nf=nf)

    def forward_single(self, diffraction_pattern):
        """
        Process a single diffraction pattern.

        Args:
            diffraction_pattern: (B, 1, h, w)

        Returns:
            patch: (B, 2, h, w) - amplitude and phase channels
        """
        z = self.encoder(diffraction_pattern)
        patch = self.decoder(z)
        return patch

    def forward(self, diffraction_patterns, positions, object_size):
        """
        Full forward pass: process all patterns and stitch.

        Args:
            diffraction_patterns: (B, N, h, w) - all diffraction amplitudes
            positions: list of (top, left) tuples
            object_size: int, size of output object

        Returns:
            output: (B, 2, H, W) - reconstructed amplitude and phase
        """
        B, N, h, w = diffraction_patterns.shape
        device = diffraction_patterns.device

        # Initialize output and counter
        output = torch.zeros(B, 2, object_size, object_size, device=device)
        counter = torch.zeros(B, 2, object_size, object_size, device=device)

        for j in range(N):
            # Get single diffraction pattern
            dp = diffraction_patterns[:, j:j + 1, :, :]  # (B, 1, h, w)

            # Process through encoder-decoder
            patch = self.forward_single(dp)  # (B, 2, h, w)

            # Stitch into output
            top, left = positions[j]
            output[:, :, top:top + h, left:left + w] += patch
            counter[:, :, top:top + h, left:left + w] += 1.0

        # Average overlapping regions
        output = output / counter.clamp(min=1.0)

        # Rescale: amplitude [0.5, 1.0], phase [-pi/3, 0]
        amp = output[:, 0:1] * 0.5 + 0.5   # sigmoid [0,1] -> [0.5, 1.0]
        phase = -output[:, 1:2] * (np.pi / 3.0)  # sigmoid [0,1] -> [-pi/3, 0]

        return torch.cat([amp, phase], dim=1)  # (B, 2, H, W)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
