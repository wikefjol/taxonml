# taxonml/metrics/rank.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Sequence, Callable, Mapping, List
import torch

RankMetricFn = Callable[
    [torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], int],
    Tuple[int, int]
]

def accuracy(logits_global, labels_global, mask, ignore_index) -> Tuple[int,int]:
    if mask is not None:
        active_idx, remap = mask
        logits_local = logits_global.index_select(1, active_idx)
        labels_local = remap[labels_global]
        valid = (labels_local != ignore_index)
        if not valid.any(): return 0, 0
        pred = logits_local.argmax(dim=-1)
        return int((pred[valid] == labels_local[valid]).sum().item()), int(valid.sum().item())
    valid = (labels_global != ignore_index)
    if not valid.any(): return 0, 0
    pred = logits_global.argmax(dim=-1)
    return int((pred[valid] == labels_global[valid]).sum().item()), int(valid.sum().item())

valid_accuracy: RankMetricFn = accuracy

RANK_METRICS: Dict[str, RankMetricFn] = {
    "accuracy": accuracy,
    "valid_accuracy": valid_accuracy,
}

def register_accuracy_by_class_size_bins(
    *,
    prefix: str,                                 # e.g. "acc_size"
    class_sizes_global: torch.Tensor,            # [C_full], CPU or CUDA ok
    bins: Sequence[Tuple[int,int]],              # [(1,1),(2,2),...]
    labels: Sequence[str],                       # ["1","2",...]
) -> List[str]:
    """
    Registers one scalar metric per bin in RANK_METRICS and returns their names.
    Each metric conforms to RankMetricFn and returns (num, den).
    """
    assert len(bins) == len(labels)
    names: List[str] = []
    # Keep a CPU view for index_select; labels_global might be on CUDA
    cs_cpu = class_sizes_global.detach().to("cpu")

    for (lo, hi), lab in zip(bins, labels):
        name = f"{prefix}[{lab}]"

        def _make(lo_: int, hi_: int) -> RankMetricFn:
            def _fn(logits_global, labels_global, mask, ignore_index) -> Tuple[int,int]:
                if mask is not None:
                    active_idx, remap = mask
                    logits_local = logits_global.index_select(1, active_idx)
                    labels_local = remap[labels_global]
                    valid = (labels_local != ignore_index)
                    if not valid.any(): return 0, 0
                    pred_ok = (logits_local.argmax(dim=-1)[valid] == labels_local[valid])
                    yg = labels_global[valid].detach().to("cpu")
                else:
                    valid = (labels_global != ignore_index)
                    if not valid.any(): return 0, 0
                    pred_ok = (logits_global.argmax(dim=-1)[valid] == labels_global[valid])
                    yg = labels_global[valid].detach().to("cpu")

                cs = cs_cpu.index_select(0, yg)                    # [N_valid] global class sizes
                in_bin = (cs >= lo_) & (cs <= hi_)
                den = int(in_bin.sum().item())
                if den == 0: return 0, 0
                # pred_ok may reside on CUDA; move to CPU boolean
                num = int((pred_ok.detach().to("cpu") & in_bin).sum().item())
                return num, den
            return _fn

        RANK_METRICS[name] = _make(int(lo), int(hi))
        names.append(name)
    return names

def compute_rank_metrics(
    *,
    which: Sequence[str],
    logits_global: torch.Tensor,
    labels_global: torch.Tensor,
    mask: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ignore_index: int,
) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for name in which:
        fn = RANK_METRICS[name]
        num, den = fn(logits_global, labels_global, mask, ignore_index)
        out[name] = {"num": int(num), "den": int(den)}
    return out
