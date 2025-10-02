# src/taxml/data/datasets.py
from __future__ import annotations
from typing import Dict, List
import numpy as np, torch
import json
from torch.utils.data import Dataset

from taxml.labels.encoder import LabelEncoder
from taxml.labels.space import LabelSpace
from taxml.preprocessing.preprocessor import Preprocessor
from taxml.core.constants import IGNORE_INDEX   # <- fix import

class MLMDataset(Dataset):
    """
    Returns:
      {
        "input_ids":      LongTensor[T],
        "labels":     LongTensor[T],   # original IDs at masked positions, IGNORE_INDEX elsewhere
        "attention_mask": LongTensor[T]
      }
    Behavior:
      - Add [CLS] + [SEP]
      - BERT-style masking p=masking_percentage with 80/10/10
    """
    def __init__(self, sequences: List[str], preprocessor: Preprocessor, masking_percentage: float):
        if hasattr(sequences, "reset_index"):                  # pandas Series or DataFrame column
            sequences = sequences.reset_index(drop=True)
        self.seq = sequences
        self.prep = preprocessor
        self.mask_p = float(masking_percentage)
        self.v = self.prep.vocab
        self.id_PAD = self.v.get_id("PAD")
        self.id_CLS = self.v.get_id("CLS")
        self.id_SEP = self.v.get_id("SEP")
        self.id_MASK = self.v.get_id("MASK")
        # Your current vocab.get_special_tokens() returns the IDs (good)
        self.special_ids = set(self.v.get_special_tokens())
        self.rand_ids = np.array([i for i in range(len(self.v)) if i not in self.special_ids],
                                 dtype=np.int64)

    def __len__(self) -> int:
        return len(self.seq)

    def _add_cls_sep(self, ids: List[int]) -> List[int]:
        return [self.id_CLS] + ids + [self.id_SEP]

    def _mask_ids(self, ids: List[int]) -> tuple[list[int], list[int]]:
        out = ids[:]  # possibly modified inputs
        labels = [IGNORE_INDEX] * len(ids)
        rng = np.random.default_rng()

        for i, t in enumerate(ids):
            if t in self.special_ids:
                continue
            if rng.random() >= self.mask_p:
                continue

            labels[i] = t  # supervise this position

            r = rng.random()
            if r < 0.8:
                out[i] = self.id_MASK
            elif r < 0.9:
                out[i] = int(rng.choice(self.rand_ids))
            else:
                # keep original token
                pass
        return out, labels

    def _attn_mask(self, ids: List[int]) -> List[int]:
        return [0 if t == self.id_PAD else 1 for t in ids]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.prep.process(self.seq[idx])      # List[int], length=T_no_specials
        ids = self._add_cls_sep(ids)                # length=T
        inp, lab = self._mask_ids(ids)
        am = self._attn_mask(inp)
        return {
            "input_ids": torch.tensor(inp, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),      # <- key name expected by verify_pipeline
            "attention_mask": torch.tensor(am, dtype=torch.long),
        }


class ClassifyDataset(Dataset):
    """
    Supervised classification dataset.

    Inputs:
      - df: pandas DataFrame with columns: "sequence", *levels (e.g. "phylum", ...), and "species"
      - preprocessor: Preprocessor (handles augmentation → tokenization → pad → truncate → map to ids)
      - encoder: LabelEncoder (owns the taxonomy/space and unknown policy)
      - levels: active levels to supervise over (order matters)

    Returns per item:
      {
        "input_ids":       LongTensor[T],
        "attention_mask":  LongTensor[T],
        "labels_by_level": { level: LongTensor[] },  # scalar class index per level
        "species_idx":     LongTensor[]              # scalar species index (if "species" column exists)
      }

    Notes:
      - No early validation against a LabelSpace; any unknowns surface via `encoder.encode(...)`.
      - Adds [CLS] at start and [SEP] at end (IDs from preprocessor.vocab).
      - Attention mask: 1 for non-PAD, 0 for PAD.
    """

    def __init__(
        self,
        df,
        preprocessor: Preprocessor,
        encoder: LabelEncoder,
        levels: List[str],
    ) -> None:
        # Normalize/clean frame
        self.df = df.reset_index(drop=True)
        self.levels = list(levels)
        self.df = self.df.dropna(subset=["sequence", *self.levels]).reset_index(drop=True)

        # Keep references
        self.prep = preprocessor
        self.enc = encoder

        # Cache special token IDs from vocab
        v = self.prep.vocab
        if v is None:
            raise ValueError("Preprocessor.vocab is required for ClassifyDataset.")
        self._id_PAD = v.get_id("PAD")
        self._id_CLS = v.get_id("CLS")
        self._id_SEP = v.get_id("SEP")

        # Whether we have a species column
        self._has_species = "species" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def _add_cls_sep(self, ids: List[int]) -> List[int]:
        return [self._id_CLS] + ids + [self._id_SEP]

    def _attn_mask(self, ids: List[int]) -> List[int]:
        return [0 if t == self._id_PAD else 1 for t in ids]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.df.iloc[i]

        # Sequence → ids (+CLS/SEP) → attention mask
        ids = self.prep.process(row["sequence"])
        ids = self._add_cls_sep(ids)
        am = self._attn_mask(ids)

        # Encode labels per active level
        # Let LabelEncoder own the unknown-policy behavior (it may raise or map to UNK)
        labels_by_level = {
            lvl: self.enc.encode(lvl, str(row[lvl]))
            for lvl in self.levels
        }

        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "labels_by_level": {k: torch.tensor(v, dtype=torch.long) for k, v in labels_by_level.items()},
        }

        # Optional: species index (if available)
        if self._has_species:
            item["species_idx"] = torch.tensor(self.enc.encode("species", str(row["species"])), dtype=torch.long)

        return item

    def __repr__(self) -> str:
        n = len(self)
        has_species = "yes" if self._has_species else "no"
        # Count unique labels per level in this dataset
        uniq_counts = {lvl: self.df[lvl].nunique() for lvl in self.levels}
        counts_str = json.dumps(uniq_counts, indent=2)
        return (
            "ClassifyDataset(\n"
            f"  rows={n}, levels=[{', '.join(self.levels)}], species_col={has_species}\n"
            f"  unique_labels={{ {counts_str} }}\n"
            f"  preprocessor={self.prep.__class__.__name__}\n"
            f"  encoder={self.enc.__class__.__name__}\n"
            ")"
        )

