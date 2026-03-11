"""
text_encoder.py
---------------
Lightweight text encoder that converts token-id sequences (from
SimpleTokenizer) into a single dense embedding vector.

Architecture:
  Embedding layer → mean-pooling across sequence → Linear projection

No external NLP library needed — works with the custom SimpleTokenizer
defined in data/dataset_loader.py.

Input:  [B, max_seq_len]  (LongTensor of token ids)
Output: [B, embedding_dim]
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Encode short text prompts into fixed-size embeddings.

    Steps:
      1. Embed each token id → [B, L, token_dim]
      2. Mean-pool over sequence length → [B, token_dim]
      3. Linear projection → [B, embedding_dim]
      4. LayerNorm for stable training
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        token_dim: int = 64,       # Internal token embedding dim (small = fast)
        dropout: float = 0.1,
        padding_idx: int = 0,      # <PAD> token index
    ):
        super().__init__()

        # Lookup table: vocab_size × token_dim
        self.token_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_dim,
            padding_idx=padding_idx,
        )

        self.projector = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(token_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L]  LongTensor of token indices
        Returns:
            embedding: [B, embedding_dim]
        """
        # Step 1: embed tokens
        embed = self.token_embed(token_ids)       # [B, L, token_dim]

        # Step 2: mean-pool (ignores padding implicitly via zero embeddings)
        pooled = embed.mean(dim=1)                # [B, token_dim]

        # Step 3: project to shared embedding space
        out = self.projector(pooled)              # [B, embedding_dim]
        return out
