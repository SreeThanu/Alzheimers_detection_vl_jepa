"""
vl_jepa_model.py
----------------
Vision-Language JEPA-style model for Alzheimer's classification.

Architecture overview:
  ┌──────────────┐      ┌──────────────┐
  │ Image Encoder│      │ Text Encoder │
  └──────┬───────┘      └──────┬───────┘
         │ img_emb              │ txt_emb
         ▼                      ▼
  ┌──────────────┐      ┌──────────────┐
  │  Img Proj    │      │  Txt Proj    │   ← projection heads (JEPA-style)
  └──────┬───────┘      └──────┬───────┘
         │                      │
         └──────────┬───────────┘
                    │  fused = img_proj + txt_proj (element-wise sum)
                    ▼
           ┌───────────────┐
           │Classification │   → [B, num_classes]
           │    Head        │
           └───────────────┘

Optional: contrastive alignment loss between image and text branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_encoder import LightweightCNNEncoder
from .text_encoder import TextEncoder


class ProjectionHead(nn.Module):
    """
    Small 2-layer MLP projection head used in JEPA / CLIP-style models.
    Maps embedding_dim → projection_dim → embedding_dim.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VLJEPAModel(nn.Module):
    """
    Full vision-language classification model.

    Args:
        vocab_size:    Size of the text vocabulary (from SimpleTokenizer)
        embedding_dim: Shared image/text embedding dimension
        projection_dim: Hidden dim in projection heads
        num_classes:   Number of Alzheimer's stages (4)
        dropout:       Dropout probability
        use_text:      If False, only the image branch is used (ablation mode)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
        use_text: bool = True,
    ):
        super().__init__()

        self.use_text = use_text
        self.embedding_dim = embedding_dim

        # ── Encoders ────────────────────────────────────────────────
        self.image_encoder = LightweightCNNEncoder(
            embedding_dim=embedding_dim, dropout=dropout
        )
        if use_text:
            self.text_encoder = TextEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                dropout=0.1,
            )

        # ── Projection heads (JEPA-style) ───────────────────────────
        self.img_proj = ProjectionHead(
            in_dim=embedding_dim,
            hidden_dim=projection_dim,
            out_dim=embedding_dim,
            dropout=dropout,
        )
        if use_text:
            self.txt_proj = ProjectionHead(
                in_dim=embedding_dim,
                hidden_dim=projection_dim,
                out_dim=embedding_dim,
                dropout=dropout,
            )

        # ── Classification head ──────────────────────────────────────
        # Input dim: embedding_dim (fused)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> dict:
        """
        Args:
            images:    [B, 3, H, W]
            token_ids: [B, L]

        Returns:
            dict with:
              'logits'     : [B, num_classes]
              'img_proj'   : [B, embedding_dim]  (for contrastive loss)
              'txt_proj'   : [B, embedding_dim]  (for contrastive loss)
        """
        # Encode image
        img_emb  = self.image_encoder(images)    # [B, E]
        img_proj = self.img_proj(img_emb)        # [B, E]

        if self.use_text:
            # Encode text
            txt_emb  = self.text_encoder(token_ids)  # [B, E]
            txt_proj = self.txt_proj(txt_emb)         # [B, E]

            # Fuse: element-wise sum (lightweight alternative to cross-attention)
            fused = img_proj + txt_proj               # [B, E]
        else:
            txt_proj = img_proj.detach()              # Dummy for return dict
            fused = img_proj

        # Classify
        logits = self.classifier(fused)               # [B, num_classes]

        return {
            "logits":    logits,
            "img_proj":  F.normalize(img_proj, dim=-1),
            "txt_proj":  F.normalize(txt_proj, dim=-1),
        }

    def contrastive_loss(
        self,
        img_proj: torch.Tensor,
        txt_proj: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Symmetric InfoNCE / CLIP-style contrastive loss.
        Aligns image and text embeddings in shared space.

        Args:
            img_proj: [B, E]  (already L2-normalised from forward())
            txt_proj: [B, E]  (already L2-normalised from forward())
        """
        batch_size = img_proj.size(0)
        labels = torch.arange(batch_size, device=img_proj.device)

        # Similarity matrix [B, B]
        sim = torch.matmul(img_proj, txt_proj.T) / temperature

        # Image→Text and Text→Image cross-entropy
        loss_i2t = F.cross_entropy(sim, labels)
        loss_t2i = F.cross_entropy(sim.T, labels)
        return (loss_i2t + loss_t2i) / 2.0
