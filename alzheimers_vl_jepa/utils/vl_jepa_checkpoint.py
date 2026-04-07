"""
vl_jepa_checkpoint.py
---------------------
Infer VL-JEPA architecture flags from a saved ``state_dict`` so inference matches
training when ``configs/model_config.yaml`` was changed after the checkpoint was saved.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


def infer_arch_from_state_dict(state_dict: Mapping[str, Any]) -> Tuple[bool, bool]:
    """
    Infer ``(use_attention_fusion, cache_text_embeddings)`` from checkpoint weights.

    - **Attention fusion:** present iff keys like ``fusion.attn.in_proj_weight`` exist
      (``SumFusion`` has no parameters).
    - **Cached text embeddings:** present iff ``_class_tokens`` is saved with
      ``numel() > 0`` (non-empty persistent buffer). If missing or empty, the model
      was trained with ``cache_text_embeddings=False``.
    """
    use_attention = any(str(k).startswith("fusion.attn") for k in state_dict)
    ct = state_dict.get("_class_tokens")
    if ct is None:
        cache_text = False
    else:
        cache_text = bool(ct.numel() > 0)
    return use_attention, cache_text


def apply_checkpoint_arch_to_model_kwargs(
    cfg: Dict[str, Any],
    state_dict: Mapping[str, Any],
    vocab_size: int,
    class_tokens_stacked: Any,
) -> Dict[str, Any]:
    """
    Build keyword args for ``VLJEPAModel`` that match ``state_dict`` and config.

    Uses checkpoint-derived flags when they differ from YAML so ``load_state_dict``
    succeeds.
    """
    vlj = cfg["vl_jepa"]
    use_txt = bool(vlj.get("use_text_branch", True))
    attn_ckpt, cache_ckpt = infer_arch_from_state_dict(state_dict)

    use_attention = attn_ckpt and use_txt
    cache_txt = cache_ckpt and use_txt

    cfg_attn = bool(vlj.get("use_attention_fusion", False)) and use_txt
    cfg_cache = bool(vlj.get("cache_text_embeddings", False)) and use_txt
    if use_attention != cfg_attn or cache_txt != cfg_cache:
        print(
            "[checkpoint] Architecture from weights → "
            f"use_attention_fusion={use_attention}, cache_text_embeddings={cache_txt} "
            f"(config YAML: attention={cfg_attn}, cache={cfg_cache})"
        )

    class_token_ids = class_tokens_stacked if cache_txt else None

    return {
        "vocab_size": vocab_size,
        "embedding_dim": cfg["image_encoder"]["embedding_dim"],
        "projection_dim": vlj["projection_dim"],
        "num_classes": int(vlj["num_classes"]),
        "dropout": vlj["dropout"],
        "use_text": use_txt,
        "cache_text_embeddings": cache_txt,
        "class_token_ids": class_token_ids,
        "use_attention_fusion": use_attention,
        "fusion_dropout": float(vlj.get("fusion_dropout", 0.0)),
    }
