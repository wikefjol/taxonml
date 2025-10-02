from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional
import hashlib, json

import torch
from torch import nn

try:
    from transformers import BertConfig, BertModel  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "transformers is required. Install with `pip install transformers`."
    ) from e

logger = logging.getLogger(__name__)


def set_frozen(module: nn.Module, frozen: bool) -> None:
    """
    Freeze (or unfreeze) all parameters in a module.
    """
    for p in module.parameters():
        p.requires_grad = not frozen


import re
from typing import Any, Dict, List, Tuple
import torch
from transformers import BertModel
import logging

logger = logging.getLogger(__name__)

def _extract_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    """
    Accept several common checkpoint layouts and return a flat state_dict.
    """
    if isinstance(raw, dict):
        # common wrappers
        for k in ("model_state_dict", "state_dict", "model"):
            if k in raw and isinstance(raw[k], dict):
                return {kk: vv for kk, vv in raw[k].items() if isinstance(vv, torch.Tensor)}
        # raw sd or mixed dict
        return {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
    # unexpected layout
    raise ValueError("Unsupported checkpoint format: expected dict-like object.")

def _strip_known_prefixes(key: str, expect_prefix: str | None) -> str:
    # only harmless/global wrappers; DO NOT strip 'encoder.' (BERT uses it legitimately)
    prefixes = [p for p in (expect_prefix, "module.", "model.", "bert.") if p]
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key

def load_pretrained_backbone(
    encoder: BertModel,
    ckpt_path: str,
    *,
    strict: bool = True,
    expect_prefix: str | None = None,
    map_location: str = "cpu",
) -> List[str]:
    """
    Verify and load encoder weights from `ckpt_path` into `encoder` (in place).

    strict=True (default): every encoder parameter must be present with matching shape.
                           Any mismatch -> raise with a compact diff.
    strict=False:          load the subset that matches; still error if *none* match.

    Returns: list of loaded encoder keys (post-normalization).
    """
    raw = torch.load(ckpt_path, map_location=map_location)
    src = _extract_state_dict(raw)

    tgt_sd = encoder.state_dict()
    mapped: Dict[str, torch.Tensor] = {}
    loaded_keys: List[str] = []

    # 1) normalize keys & select candidates
    for k, v in src.items():
        t = _strip_known_prefixes(k, expect_prefix)
        # accept only exact-name hits in encoder
        if t in tgt_sd:
            mapped[t] = v
            loaded_keys.append(t)

    if not loaded_keys:
        raise RuntimeError(
            f"No overlapping tensors between checkpoint and encoder.\n"
            f"ckpt={ckpt_path}\n"
            f"examples(ckpt)={list(sorted(src.keys()))[:5]}"
        )

    # 2) shape audit (before mutating the model)
    shape_mismatches: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for k, v in mapped.items():
        if tuple(v.shape) != tuple(tgt_sd[k].shape):
            shape_mismatches.append((k, tuple(v.shape), tuple(tgt_sd[k].shape)))

    # 3) completeness audit
    missing_in_ckpt = [k for k in tgt_sd.keys() if k not in mapped]

    if strict:
        errs = []
        if shape_mismatches:
            ex = ", ".join(f"{k}: {a}->{b}" for k, a, b in shape_mismatches[:5])
            errs.append(f"shape_mismatches={len(shape_mismatches)} (e.g., {ex}{' …' if len(shape_mismatches)>5 else ''})")
        if missing_in_ckpt:
            errs.append(f"missing_in_ckpt={len(missing_in_ckpt)} (e.g., {missing_in_ckpt[:5]}{' …' if len(missing_in_ckpt)>5 else ''})")
        # unexpected keys in ckpt are fine (heads/optimizer/etc), we ignore them
        if errs:
            raise RuntimeError(
                "Checkpoint does not match encoder config under strict loading:\n  " + "\n  ".join(errs)
            )

    # 4) if not strict, drop shape-mismatched tensors from the mapping
    if not strict and shape_mismatches:
        bad = {k for k, _, _ in shape_mismatches}
        mapped = {k: v for k, v in mapped.items() if k not in bad}
        loaded_keys = [k for k in loaded_keys if k not in bad]
        if not loaded_keys:
            raise RuntimeError("After removing mismatches, nothing left to load.")

    # 5) actually load (in place)
    encoder.load_state_dict(mapped, strict=False)

    # 6) log concise summary
    total_params = sum(1 for _ in tgt_sd.keys())
    logger.info(
        f"Backbone load ok: loaded={len(loaded_keys)}/{total_params} tensors "
        f"(strict={strict}); ckpt={ckpt_path}"
    )
    if not strict and shape_mismatches:
        logger.info(f"Ignored {len(shape_mismatches)} shape-mismatched tensors.")

    return loaded_keys



def describe_encoder(encoder: BertModel) -> str:
    """
    Human-readable summary string for logs.
    """
    num_params = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    cfg = encoder.config
    lines = [
        "BERT encoder:",
        f"  hidden_size={cfg.hidden_size}, layers={cfg.num_hidden_layers}, heads={cfg.num_attention_heads}",
        f"  intermediate_size={cfg.intermediate_size}, max_position_embeddings={cfg.max_position_embeddings}",
        f"  params(total/trainable)={num_params:,}/{trainable:,}",
    ]
    return "\n".join(lines)

def _short_hash(obj, n=8):
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode()).hexdigest()[:n]

def derive_arch_id_from_cfg(
    cfg: BertConfig,
    *,
    include_vocab: bool = False,
    include_positions: bool = True,
    with_hash: bool = False,
    hash_len: int = 8,
) -> str:
    parts = [
        "bert",
        f"h{cfg.hidden_size}",
        f"L{cfg.num_hidden_layers}",
        f"H{cfg.num_attention_heads}",
        f"i{cfg.intermediate_size}",
    ]
    if include_positions:
        parts.append(f"P{cfg.max_position_embeddings}")
    if include_vocab:
        parts.append(f"V{cfg.vocab_size}")
    if with_hash:
        sig = {
            "h": cfg.hidden_size,
            "L": cfg.num_hidden_layers,
            "H": cfg.num_attention_heads,
            "i": cfg.intermediate_size,
            "P": cfg.max_position_embeddings if include_positions else None,
            "V": cfg.vocab_size if include_vocab else None,
        }
        parts.append(_short_hash(sig, n=hash_len))
    return "_".join(parts)