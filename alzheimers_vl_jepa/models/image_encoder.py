"""
image_encoder.py
----------------
Lightweight CNN image encoder optimised for laptop training.

Architecture: 4-block CNN with BatchNorm + ReLU + MaxPool
Input: [B, 3, 128, 128]  (RGB MRI images)
Output: [B, embedding_dim]  (dense feature vector)

Why CNN instead of ViT?
- Uses far less memory than even tiny ViT on CPU
- Converges faster on small medical datasets
- No positional embeddings to tune
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # halves spatial dims
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightCNNEncoder(nn.Module):
    """
    4-stage CNN encoder.
    Spatial progression (128→64→32→16→8):
      Stage 1: 3   → 32   channels
      Stage 2: 32  → 64   channels
      Stage 3: 64  → 128  channels
      Stage 4: 128 → 256  channels
    Global average pool → linear projection → embedding_dim
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,   32),   # 128 → 64
            ConvBlock(32,  64),   # 64  → 32
            ConvBlock(64,  128),  # 32  → 16
            ConvBlock(128, 256),  # 16  → 8
        )

        # Global average pool reduces [B, 256, 8, 8] → [B, 256]
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.projector = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            embedding: [B, embedding_dim]
        """
        feat = self.features(x)             # [B, 256, 8, 8]
        feat = self.pool(feat).flatten(1)   # [B, 256]
        embedding = self.projector(feat)    # [B, embedding_dim]
        return embedding

    @property
    def output_dim(self) -> int:
        return self.projector[-2].out_features  # Linear layer out_features
