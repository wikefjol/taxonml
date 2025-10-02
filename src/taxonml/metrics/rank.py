# taxonml/metrics/rank.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Sequence, Callable, Mapping
import torch

# Metric function signature: returns (numerator, denominator) for a scalar metric
RankMetricFn = Callable[
    [
        torch.Tensor,                      # logits_global: [B, C_full]
        torch.Tensor,                      # labels_global: [B]
        Optional[Tuple[torch.Tensor, torch.Tensor]],  # mask: (active_idx, remap) or None
        int,                               # ignore_index
    ],
    Tuple[int, int]
]

# --------------------------- Metric implementations ---------------------------

def accuracy(
    logits_global: torch.Tensor,
    labels_global: torch.Tensor,
    mask: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ignore_index: int,
) -> Tuple[int, int]:
    """
    Active-class accuracy: argmax over masked logits; compare to remapped labels.
    Denominator counts only valid (non-ignored) samples.
    """
    if mask is not None:
        active_idx, remap = mask
        logits_local = logits_global.index_select(1, active_idx)
        labels_local = remap[labels_global]
        valid = (labels_local != ignore_index)
        if not valid.any():
            return 0, 0
        pred = logits_local.argmax(dim=-1)
        correct = int((pred[valid] == labels_local[valid]).sum().item())
        total   = int(valid.sum().item())
        return correct, total
    
    # unmasked path
    valid = (labels_global != ignore_index)
    if not valid.any():
        return 0, 0
    pred = logits_global.argmax(dim=-1)
    correct = int((pred[valid] == labels_global[valid]).sum().item())
    total   = int(valid.sum().item())
    return correct, total

# Synonym if you like the name:
valid_accuracy: RankMetricFn = accuracy

# --------------------------- Public API / Registry ----------------------------

RANK_METRICS: Mapping[str, RankMetricFn] = {
    "accuracy": accuracy,
    "valid_accuracy": valid_accuracy,  # same behavior; pick the name you prefer
}

def compute_rank_metrics(
    *,
    which: Sequence[str],
    logits_global: torch.Tensor,
    labels_global: torch.Tensor,
    mask: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ignore_index: int,
) -> Dict[str, Dict[str, int]]:
    """
    Compute a set of rank-local metrics for one (level, batch).

    Returns {metric_name: {"num": int, "den": int}} so the caller can aggregate
    across batches without average-of-averages issues.
    """
    out: Dict[str, Dict[str, int]] = {}
    for name in which:
        fn = RANK_METRICS[name]
        num, den = fn(logits_global, labels_global, mask, ignore_index)
        out[name] = {"num": int(num), "den": int(den)}
    return out
