# collators.py
from typing import Dict, List
import torch

def collate_simple(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # Items are already fixed length → just stack
    out = {k: [] for k in ("input_ids","attention_mask")}
    for item in batch:
        out["input_ids"].append(item["input_ids"])
        out["attention_mask"].append(item["attention_mask"])
    result = {k: torch.stack(v, 0) for k,v in out.items()}
    # pass through others verbatim if present (MLM: labels; CLS: labels_by_level)
    if "labels" in batch[0]:
        result["labels"] = torch.stack([x["labels"] for x in batch], 0)
    if "labels_by_level" in batch[0]:
        lvls = list(batch[0]["labels_by_level"].keys())
        result["labels_by_level"] = {l: torch.stack([x["labels_by_level"][l] for x in batch], 0) for l in lvls}
        result["species_idx"]    = torch.stack([x["species_idx"] for x in batch], 0)
    return result
