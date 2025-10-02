from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable
import json
import pandas as pd
from collections import defaultdict

from taxonml.core.constants import CANON_LEVELS

@dataclass
class LabelSpace:
    """
    Canonical description of your taxonomy label universe.

    - levels: ordered taxonomy ranks you care about (subset of CANON_LEVELS, consecutive).
    - label_to_idx / idx_to_label: compact integer encoding per level.
    - counts: per-level frequency arrays aligned to idx ordering.
    - counts_by_label: human-readable dict view (duplicated for clarity).
    - lineage.child_to_parent[level]: child(label)->parent(label) for each non-root level.
    - lineage.parent_to_children[level]: parent(label)->[children] for each parent level.
    - ancestors[label]: cached full path from root to this label (for fast validation/lookups).

    Build from:
      - DataFrame: `from_dataframe(df, levels=...)`
      - CSV:       `from_csv(path, levels=...)`
      - JSON:      `from_json(path)`

    Persist with `to_json(path)`.
    """
    levels: List[str]
    label_to_idx: Dict[str, Dict[str, int]]
    idx_to_label: Dict[str, Dict[int, str]]
    counts: Dict[str, List[int]]
    counts_by_label: Dict[str, Dict[str, int]]
    child_to_parent: Dict[str, Dict[str, str]]
    parent_to_children: Dict[str, Dict[str, List[str]]]
    ancestors: Dict[str, List[str]] = field(default_factory=dict)

    # --------- Constructors --------- #

    @staticmethod
    def validate_levels(levels: Iterable[str]) -> List[str]:
        levels = list(levels)
        # ensure they are a consecutive slice of CANON_LEVELS
        if not levels:
            raise ValueError("levels must be a non-empty list of taxonomy ranks.")
        try:
            start = CANON_LEVELS.index(levels[0])
        except ValueError as e:
            raise ValueError(f"Unknown level '{levels[0]}'. Must be one of {CANON_LEVELS}.") from e
        if CANON_LEVELS[start:start + len(levels)] != levels:
            raise ValueError(
                f"Levels must be consecutive in canonical order {CANON_LEVELS}. "
                f"Got {levels}."
            )
        return levels

    @classmethod
    def from_csv(cls, path: str, *, levels: Iterable[str]) -> "LabelSpace":
        df = pd.read_csv(path)
        return cls.from_dataframe(df, levels=levels)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, *, levels: Iterable[str]) -> "LabelSpace":
        levels = cls.validate_levels(levels)

        # 1) Build vocabularies per level (sorted for determinism)
        label_to_idx: Dict[str, Dict[str, int]] = {}
        idx_to_label: Dict[str, Dict[int, str]] = {}
        counts: Dict[str, List[int]] = {}
        counts_by_label: Dict[str, Dict[str, int]] = {}

        for lvl in levels:
            if lvl not in df.columns:
                raise ValueError(f"Column '{lvl}' missing from dataframe.")
            vc = df[lvl].astype(str).fillna("").replace({"nan": ""}).value_counts()
            labels = sorted(vc.index.tolist())
            l2i = {lab: i for i, lab in enumerate(labels)}
            i2l = {i: lab for lab, i in l2i.items()}
            cnts = [int(vc.get(lab, 0)) for lab in labels]
            label_to_idx[lvl] = l2i
            idx_to_label[lvl] = i2l
            counts[lvl] = cnts
            counts_by_label[lvl] = {lab: int(vc.get(lab, 0)) for lab in labels}

        # 2) Build lineage maps from rows (immediate relations)
        child_to_parent: Dict[str, Dict[str, str]] = {lvl: {} for lvl in levels[1:]}
        parent_to_children: Dict[str, Dict[str, List[str]]] = {lvl: defaultdict(list) for lvl in levels[:-1]}

        for _, row in df[levels].astype(str).fillna("").replace({"nan": ""}).iterrows():
            for i in range(1, len(levels)):
                parent_lvl, child_lvl = levels[i-1], levels[i]
                parent, child = row[parent_lvl], row[child_lvl]
                if not parent or not child:
                    continue
                # enforce 1 parent per child
                prev = child_to_parent[child_lvl].get(child)
                if prev is not None and prev != parent:
                    raise ValueError(
                        f"Inconsistent lineage: child '{child}' seen with parents "
                        f"'{prev}' and '{parent}' at level '{child_lvl}'."
                    )
                child_to_parent[child_lvl][child] = parent
                if child not in parent_to_children[parent_lvl][parent]:
                    parent_to_children[parent_lvl][parent].append(child)

        # 3) Build ancestor cache (full path to root) for quick validation
        ancestors: Dict[str, List[str]] = {}

        def compute_path(label: str, level: str) -> List[str]:
            """Return full path [root,...,label] for any label at any level."""
            i = levels.index(level)
            if i == 0:
                return [label]
            # climb up using child_to_parent progressively
            path = [label]
            cur_label = label
            cur_level_idx = i
            while cur_level_idx > 0:
                child_lvl = levels[cur_level_idx]
                parent_lvl = levels[cur_level_idx - 1]
                parent = child_to_parent.get(child_lvl, {}).get(cur_label)
                if parent is None:
                    # No parent recorded — return partial (still useful)
                    break
                path.append(parent)
                cur_label = parent
                cur_level_idx -= 1
            path.reverse()
            return path

        # precompute for all labels we know
        for lvl in levels:
            for lab in label_to_idx[lvl].keys():
                ancestors[lab] = compute_path(lab, lvl)

        return cls(
            levels=levels,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            counts=counts,
            counts_by_label=counts_by_label,
            child_to_parent=child_to_parent,
            parent_to_children={k: dict(v) for k, v in parent_to_children.items()},
            ancestors=ancestors,
        )

    # --------- Persistence --------- #

    def to_json(self, path: str) -> None:
        payload = {
            "version": 1,
            "levels": self.levels,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": {lvl: {str(k): v for k, v in d.items()} for lvl, d in self.idx_to_label.items()},
            "counts": self.counts,
            "counts_by_label": self.counts_by_label,
            "child_to_parent": self.child_to_parent,
            "parent_to_children": self.parent_to_children,
            "ancestors": self.ancestors,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "LabelSpace":
        with open(path, "r") as f:
            payload = json.load(f)
        levels = cls.validate_levels(payload["levels"])
        # coerce idx_to_label keys back to ints
        idx_to_label = {lvl: {int(k): v for k, v in d.items()} for lvl, d in payload["idx_to_label"].items()}
        return cls(
            levels=levels,
            label_to_idx=payload["label_to_idx"],
            idx_to_label=idx_to_label,
            counts=payload["counts"],
            counts_by_label=payload["counts_by_label"],
            child_to_parent=payload["child_to_parent"],
            parent_to_children=payload["parent_to_children"],
            ancestors=payload.get("ancestors", {}),
        )

    # --------- Introspection / helpers --------- #

    def num_classes(self, level: str) -> int:
        return len(self.label_to_idx[level])

    def spec(self, subset_levels: Optional[Iterable[str]] = None) -> List[Tuple[str, int]]:
        lvls = self.validate_levels(subset_levels) if subset_levels else self.levels
        return [(lvl, self.num_classes(lvl)) for lvl in lvls]

    def counts_dict(self, level: str) -> Dict[str, int]:
        return dict(self.counts_by_label[level])

    def full_lineage(self, label: str) -> List[str]:
        return list(self.ancestors.get(label, [label]))

    def validate_path(self, pred_by_level: Dict[str, str]) -> bool:
        """Validate child→parent consistency across provided consecutive levels."""
        lvls = [lvl for lvl in self.levels if lvl in pred_by_level]
        for i in range(1, len(lvls)):
            parent_lvl, child_lvl = lvls[i-1], lvls[i]
            parent, child = pred_by_level[parent_lvl], pred_by_level[child_lvl]
            if not parent or not child:
                return False
            if self.child_to_parent.get(child_lvl, {}).get(child) != parent:
                return False
        return True
