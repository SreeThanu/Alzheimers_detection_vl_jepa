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
  │  Img Proj    │      │  Txt Proj    │
  └──────┬───────┘      └──────┬───────┘
         │                      │
         └──────────┬───────────┘
                    │  fusion: sum or cross-attention (Q=image, K/V=text)
                    ▼
           ┌───────────────┐
           │Classification │
           └───────────────┘

Optional: contrastive alignment loss between image and text branches.
Optional: ``cache_text_embeddings`` — precomputed per-class text projections for fast eval.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import CrossAttentionFusion, SumFusion
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
        cache_text_embeddings: If True, maintain ``cached_text_embeddings`` and use them
            in eval mode when ``labels`` is passed (skips text encoder on that path).
        class_token_ids: [num_classes, L] token ids for each class prompt; required when
            ``cache_text_embeddings`` is True and ``use_text`` is True.
        use_attention_fusion: If True, fuse with cross-attention (Q=image, K/V=text);
            else element-wise sum.
        fusion_dropout: Attention dropout when using cross-attention.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
        use_text: bool = True,
        cache_text_embeddings: bool = False,
        class_token_ids: Optional[torch.Tensor] = None,
        use_attention_fusion: bool = False,
        fusion_dropout: float = 0.0,
    ):
        super().__init__()

        self.use_text = use_text
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.cache_text_embeddings = bool(cache_text_embeddings) and use_text
        self.use_attention_fusion = bool(use_attention_fusion) and use_text

        if self.cache_text_embeddings and class_token_ids is None:
            raise ValueError(
                "class_token_ids [num_classes, seq_len] is required when "
                "cache_text_embeddings=True and use_text=True"
            )

        self.image_encoder = LightweightCNNEncoder(
            embedding_dim=embedding_dim, dropout=dropout
        )
        if use_text:
            self.text_encoder = TextEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                dropout=0.1,
            )

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
            if self.use_attention_fusion:
                self.fusion = CrossAttentionFusion(
                    embed_dim=embedding_dim,
                    num_heads=1,
                    dropout=fusion_dropout,
                )
            else:
                self.fusion = SumFusion()
        else:
            self.fusion = None  # unused; forward uses image branch only

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, num_classes),
        )

        if use_text and class_token_ids is not None:
            self.register_buffer("_class_tokens", class_token_ids.clone().long(), persistent=True)
        else:
            self.register_buffer("_class_tokens", torch.zeros(0, 0, dtype=torch.long), persistent=False)

        self.register_buffer(
            "cached_text_embeddings",
            torch.zeros(num_classes, embedding_dim),
            persistent=True,
        )

        if self.cache_text_embeddings:
            self.update_text_embedding_cache()

    @torch.no_grad()
    def update_text_embedding_cache(self) -> None:
        """
        Fill ``cached_text_embeddings`` [C, E] from current ``text_encoder`` + ``txt_proj``
        using the fixed per-class token rows in ``_class_tokens``.
        """
        if not self.use_text or not self.cache_text_embeddings:
            return
        if self._class_tokens.numel() == 0:
            return
        device = self.cached_text_embeddings.device
        tokens = self._class_tokens.to(device)
        was_te = self.text_encoder.training
        was_tp = self.txt_proj.training
        self.text_encoder.eval()
        self.txt_proj.eval()
        z = self.txt_proj(self.text_encoder(tokens))
        self.cached_text_embeddings.copy_(z.detach())
        self.text_encoder.train(was_te)
        self.txt_proj.train(was_tp)

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            images:    [B, 3, H, W]
            token_ids: [B, L]
            labels:    [B] optional; when ``cache_text_embeddings`` is True and the model
                       is in eval mode, used to gather cached text projections (fast path).
                       When omitted or training, the text encoder runs on ``token_ids``.

        Returns:
            dict with logits, img_proj, txt_proj (L2-normalised for contrastive loss).
        """
        img_emb = self.image_encoder(images)
        img_proj = self.img_proj(img_emb)

        if self.use_text:
            use_cache = (
                self.cache_text_embeddings
                and labels is not None
                and not self.training
            )
            if use_cache:
                txt_proj = self.cached_text_embeddings[labels.long()]
            else:
                txt_emb = self.text_encoder(token_ids)
                txt_proj = self.txt_proj(txt_emb)

            fused = self.fusion(img_proj, txt_proj)
        else:
            txt_proj = img_proj.detach()
            fused = img_proj

        logits = self.classifier(fused)

        return {
            "logits": logits,
            "img_proj": F.normalize(img_proj, dim=-1),
            "txt_proj": F.normalize(txt_proj, dim=-1),
        }

    def contrastive_loss(
        self,
        img_proj: torch.Tensor,
        txt_proj: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        batch_size = img_proj.size(0)
        lab = torch.arange(batch_size, device=img_proj.device)

        sim = torch.matmul(img_proj, txt_proj.T) / temperature

        loss_i2t = F.cross_entropy(sim, lab)
        loss_t2i = F.cross_entropy(sim.T, lab)
        return (loss_i2t + loss_t2i) / 2.0
