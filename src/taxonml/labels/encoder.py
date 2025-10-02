from __future__ import annotations
from typing import Dict, List, Iterable, Optional
from dataclasses import dataclass
from .space import LabelSpace

UnknownPolicy = str  # "error" | "skip" | "assign_last" etc. (we implement "error" and "skip")


@dataclass
class LabelEncoder:
    """
    Deterministic string<->index encoding built on a LabelSpace.

    Encoding policies:
      - unknown='error': raise if label not in LabelSpace
      - unknown='skip':  return None for that label
    """
    space: LabelSpace
    unknown: UnknownPolicy = "error"

    def encode(self, level: str, label: str) -> Optional[int]:
        d = self.space.label_to_idx[level]
        if label in d:
            return d[label]
        if self.unknown == "skip":
            return None
        raise KeyError(f"Unknown label '{label}' at level '{level}'")

    def decode(self, level: str, idx: int) -> str:
        return self.space.idx_to_label[level][idx]

    def encode_row(self, labels_by_level: Dict[str, str]) -> Dict[str, Optional[int]]:
        out: Dict[str, Optional[int]] = {}
        for lvl in self.space.levels:
            if lvl in labels_by_level:
                out[lvl] = self.encode(lvl, labels_by_level[lvl])
        return out

    def encode_batch(self, batch_by_level: Dict[str, Iterable[str]]) -> Dict[str, List[Optional[int]]]:
        result: Dict[str, List[Optional[int]]] = {}
        for lvl, labels in batch_by_level.items():
            result[lvl] = [self.encode(lvl, lab) for lab in labels]
        return result

    def spec(self, subset_levels: Optional[Iterable[str]] = None):
        return self.space.spec(subset_levels=subset_levels)
