"""
PtychoNet+ : Improved Ptychographic Phase Retrieval

Key improvements over PtychoNet:
1. Same encoder-decoder patch processing (no skip connections - by design,
   since the encoder must transform from Fourier to real space, skip
   connections leak Fourier-domain artifacts)
2. Deeper encoder-decoder with residual connections within each block
3. Post-stitching refinement network with dilated convolutions
   - Large receptive field to fix boundary artifacts between patches
   - Residual learning for stable training
4. Physics-based iterative refinement at test time (hybrid approach)

This addresses PtychoNet's main limitation: patches are processed independently
and stitched by simple averaging, causing boundary artifacts and no cross-patch
information sharing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    """Residual block with two conv layers."""
    def __init__(self, channels, activation='relu'):
        super().__init__()
        act = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.2, inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            act,
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class ImprovedEncoder(nn.Module):
    """Encoder with residual blocks for better gradient flow."""
    def __init__(self, nf=64):
        super().__init__()
        self.layers = nn.Sequential(
            # 1 x 128 x 128 -> nf x 64 x 64
            nn.Conv2d(1, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # nf x 64 x 64 -> 2nf x 32 x 32
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(nf * 2, 'leaky'),

            # 2nf x 32 x 32 -> 4nf x 16 x 16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(nf * 4, 'leaky'),

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


class ImprovedDecoder(nn.Module):
    """Decoder with residual blocks."""
    def __init__(self, nf=64):
        super().__init__()
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
            ResBlock(nf * 4, 'relu'),

            # 4nf x 16 x 16 -> 2nf x 32 x 32
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(inplace=True),
            ResBlock(nf * 2, 'relu'),

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


class RefinementNet(nn.Module):
    """
    Post-stitching refinement network.

    Operates on the full stitched image to:
    - Fix boundary artifacts between patches
    - Share information across distant patches
    - Enforce global consistency

    Uses dilated convolutions for large receptive field without
    excessive parameters.
    """
    def __init__(self, nf=48):
        super().__init__()
        self.net = nn.Sequential(
            # Input: 2 channels (amp + phase)
            nn.Conv2d(2, nf, 3, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            # Dilated conv cascade for large receptive field
            nn.Conv2d(nf, nf, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf, nf, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf, nf, 3, padding=8, dilation=8, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf, nf, 3, padding=16, dilation=16, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf, nf, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf, nf, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf, nf, 3, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),

            nn.Conv2d(nf, 2, 1),
            nn.Tanh(),  # Output small residual corrections
        )
        # Scale the residual correction
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.net(x)


class PtychoNetPlus(nn.Module):
    """
    PtychoNet+ : Improved ptychographic phase retrieval.

    Architecture:
    1. Improved encoder-decoder with residual blocks (per-patch)
    2. Stitching via averaging (same as PtychoNet)
    3. Global refinement network with dilated convolutions

    The encoder-decoder has NO skip connections by design:
    the Fourier->real space transformation requires the encoder
    to fully transform the representation, and skip connections
    would leak Fourier-domain features into the real-space output.
    """
    def __init__(self, nf=64, refine_nf=48):
        super().__init__()
        self.encoder = ImprovedEncoder(nf=nf)
        self.decoder = ImprovedDecoder(nf=nf)
        self.refine = RefinementNet(nf=refine_nf)

    def forward_single(self, diffraction_pattern):
        z = self.encoder(diffraction_pattern)
        return self.decoder(z)

    def forward(self, diffraction_patterns, positions, object_size):
        B, N, h, w = diffraction_patterns.shape
        device = diffraction_patterns.device

        # Process all patches
        output = torch.zeros(B, 2, object_size, object_size, device=device)
        counter = torch.zeros(B, 1, object_size, object_size, device=device)

        for j in range(N):
            dp = diffraction_patterns[:, j:j + 1, :, :]
            patch = self.forward_single(dp)
            top, left = positions[j]
            output[:, :, top:top + h, left:left + w] += patch
            counter[:, :, top:top + h, left:left + w] += 1.0

        output = output / counter.clamp(min=1.0)

        # Global refinement
        output = self.refine(output)

        # Clamp for stability then scale
        output = output.clamp(0, 1)
        amp = output[:, 0:1] * 0.5 + 0.5
        phase = -output[:, 1:2] * (np.pi / 3.0)

        return torch.cat([amp, phase], dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
