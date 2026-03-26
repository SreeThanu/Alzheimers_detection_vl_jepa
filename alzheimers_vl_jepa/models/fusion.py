"""
fusion.py
---------
Vision–language fusion: additive baseline vs single-head cross-attention.

Tensor layout for attention (``batch_first=True``):
  * Query  ``q``:  (B, N_q, D)  — here N_q = 1 (one image token per sample)
  * Key    ``k``:  (B, N_kv, D)
  * Value  ``v``:  (B, N_kv, D)  — here N_kv = 1 (one text token per sample)

Output: (B, N_q, D), then squeeze sequence dim → (B, D) fused representation.
"""

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Image queries attend to text keys/values.

    Args:
        embed_dim: Channel dimension D (must match projected image/text features).
        num_heads: Number of attention heads (default 1).
        dropout: Dropout on attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: (B, D) global image representation (acts as one query token).
            txt_feat: (B, D) global text representation (one key/value token).

        Returns:
            fused: (B, D) attention output for the image token.
        """
        # (B, D) → (B, 1, D) so MultiheadAttention sees (B, N, D)
        q = img_feat.unsqueeze(1)
        k = txt_feat.unsqueeze(1)
        v = txt_feat.unsqueeze(1)
        out, _ = self.attn(q, k, v, need_weights=False)
        return out.squeeze(1)


class SumFusion(nn.Module):
    """Element-wise sum (legacy fusion)."""

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor) -> torch.Tensor:
        return img_feat + txt_feat
